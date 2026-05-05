#!/usr/bin/env python3
"""Build a compact P03 main-result HTML report.

The report is intentionally figure-first for lecture use:

* one-frame oracle map vs ego-motion accumulated oracle map;
* DoA-estimator maps from angle FFT, MUSIC, and RadarCubeDoANet;
* DoA and downstream map metrics in one table.
"""
from __future__ import annotations

import argparse
import base64
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mapping import (
    EgoPose,
    accumulate_probability_map,
    map_metrics,
    point_cloud_from_measurements,
    point_cloud_grid,
)
from model import build_model
from train import (
    MappingDetectionDataset,
    _mapping_metrics_for_angles,
    _method_metrics,
    _predict_model_angles,
    _predict_signal_processing_angles,
)

BASE = Path(__file__).resolve().parent


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def flip_for_plot(grid: np.ndarray) -> np.ndarray:
    return np.flipud(np.asarray(grid))


def poses_for_scene(dataset: MappingDetectionDataset, scene_idx: int) -> list[EgoPose]:
    return [
        EgoPose(
            x_m=float(p[0]),
            y_m=float(p[1]),
            heading_deg=float(p[2]),
            speed_mps=float(p[3]),
        )
        for p in dataset.poses[scene_idx]
    ]


def collect_frame_lists(
    dataset: MappingDetectionDataset,
    pred_angle: np.ndarray,
    scene_idx: int,
    frame_limit: int | None = None,
    use_dynamic: bool = False,
) -> tuple[list[EgoPose], list[list[float]], list[list[float]]]:
    poses = poses_for_scene(dataset, scene_idx)
    if frame_limit is not None:
        poses = poses[:frame_limit]
    per_frame_angles = [[] for _ in poses]
    per_frame_ranges = [[] for _ in poses]

    det_mask = dataset.scene_idx == scene_idx
    if not use_dynamic:
        det_mask &= ~dataset.is_dynamic
    if frame_limit is not None:
        det_mask &= dataset.frame_idx < frame_limit

    for i in np.nonzero(det_mask)[0]:
        frame = int(dataset.frame_idx[i])
        if 0 <= frame < len(poses):
            per_frame_angles[frame].append(float(pred_angle[i]))
            per_frame_ranges[frame].append(float(dataset.range_m[i]))
    return poses, per_frame_angles, per_frame_ranges


def build_ogm_for_scene(
    dataset: MappingDetectionDataset,
    pred_angle: np.ndarray,
    scene_idx: int,
    frame_limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    poses, per_frame_angles, per_frame_ranges = collect_frame_lists(
        dataset,
        pred_angle,
        scene_idx,
        frame_limit=frame_limit,
        use_dynamic=False,
    )
    ogm_prob, ogm_bin = accumulate_probability_map(
        poses,
        per_frame_angles,
        per_frame_ranges,
        grid_size=dataset.grid_size,
        grid_range_m=dataset.grid_range_m,
        grid_spec=dataset.grid_spec,
        max_range_m=dataset.radar_max_range_m,
        beam_width_deg=5.0,
        p_occ=0.60,
        p_free=0.45,
    )
    points = point_cloud_from_measurements(poses, per_frame_ranges, per_frame_angles)
    pc_grid = point_cloud_grid(
        points,
        grid_size=dataset.grid_size,
        grid_range_m=dataset.grid_range_m,
        grid_spec=dataset.grid_spec,
        sigma_cells=1.0,
    )
    return ogm_prob, ogm_bin, pc_grid


def plot_panel(ax, grid: np.ndarray, title: str, dataset: MappingDetectionDataset, poses: list[EgoPose] | None = None) -> None:
    ax.imshow(
        flip_for_plot(grid),
        origin="lower",
        extent=dataset.grid_spec.extent,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    if poses:
        ax.plot([p.x_m for p in poses], [p.y_m for p in poses], "c.-", lw=1.2, ms=3)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-12, 12)
    ax.set_ylim(0, 34)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def metric_row(metrics: dict, doa_key: str, map_key: str, label: str) -> dict[str, float | str]:
    doa = metrics.get(doa_key, {})
    mp = metrics.get(map_key, {})
    return {
        "method": label,
        "mae_deg": doa.get("mae_deg", float("nan")),
        "rmse_deg": doa.get("rmse_deg", float("nan")),
        "within_2deg_acc": doa.get("within_2deg_acc", float("nan")),
        "ogm_iou": mp.get("ogm_thr0p5_iou", float("nan")),
        "ogm_f1": mp.get("ogm_thr0p5_f1", float("nan")),
        "pc_iou": mp.get("point_cloud_grid_iou", float("nan")),
        "point_error_mean_m": mp.get("point_error_mean_m", float("nan")),
        "point_error_p90_m": mp.get("point_error_p90_m", float("nan")),
    }


def format_float(v: object, digits: int = 3) -> str:
    if isinstance(v, (int, float)) and np.isfinite(v):
        return f"{float(v):.{digits}f}"
    return "—"


def build_report_metrics(
    dataset: MappingDetectionDataset,
    pred_dl: np.ndarray,
    pred_fft: np.ndarray,
    pred_music: np.ndarray,
    pred_oracle: np.ndarray,
) -> dict:
    """Compute a self-consistent metrics block from this report's dataset."""

    true = dataset.angle_deg
    return {
        "deep_learning_doa": _method_metrics(true, pred_dl, dataset),
        "signal_processing_angle_fft_doa": _method_metrics(true, pred_fft, dataset),
        "signal_processing_music_doa": _method_metrics(true, pred_music, dataset),
        "oracle_gt_doa": _method_metrics(true, pred_oracle, dataset),
        "map_from_deep_learning_doa": _mapping_metrics_for_angles(dataset, pred_dl),
        "map_from_angle_fft_doa": _mapping_metrics_for_angles(dataset, pred_fft),
        "map_from_music_doa": _mapping_metrics_for_angles(dataset, pred_music),
        "map_from_oracle_gt_doa": _mapping_metrics_for_angles(dataset, pred_oracle),
        "data_contract": {
            "source": "recomputed_by_make_main_result_report.py",
            "n_test_detections": int(len(dataset)),
            "n_test_scenes": int(dataset.gt_ogm.shape[0]),
            "grid_size": int(dataset.grid_size),
            "grid_nx": int(dataset.grid_nx),
            "grid_ny": int(dataset.grid_ny),
            "grid_range_m": float(dataset.grid_range_m),
            "grid_x_min_m": float(dataset.grid_spec.x_min_m),
            "grid_x_max_m": float(dataset.grid_spec.x_max_m),
            "grid_y_min_m": float(dataset.grid_spec.y_min_m),
            "grid_y_max_m": float(dataset.grid_spec.y_max_m),
            "grid_cell_x_m": float(dataset.grid_cell_x_m),
            "grid_cell_y_m": float(dataset.grid_cell_y_m),
            "grid_square_cell": bool(dataset.grid_spec.is_square_cell),
            "radar_max_range_m": float(dataset.radar_max_range_m),
            "wall_spacing_m": float(dataset.wall_spacing_m),
            "radar_bw_hz": float(dataset.radar_bw_hz),
            "radar_n_fast": int(dataset.radar_n_fast),
            "radar_range_res_m": float(dataset.radar_range_res_m),
            "n_steps": int(dataset.n_steps),
        },
    }


def classify_report(config: dict) -> tuple[str, str, list[str]]:
    """Return (kind, label, caveats) for the report config."""

    caveats = [
        "Main map evaluation uses simulator-exact range and perfect ego poses; range-resolution and ego-motion errors are appendix experiments.",
        "All DoA methods receive the same RD-selected antenna-vector detections, so the comparison isolates DoA quality rather than detection or association quality.",
        "The GT map is radar-scatterer occupancy, not semantic wall reconstruction.",
        "Map-grid IoU is a raster metric; the canonical grid uses uniform square cells, and continuous point error remains the grid-invariant reference.",
    ]
    canonical = (
        abs(float(config["radar_bw_mhz"]) - 200.0) <= 1e-6
        and int(config["radar_n_fast"]) == 1024
        and config["wall_spacing_m"] is not None
        and float(config["wall_spacing_m"]) <= 0.350001
        and int(config["n_steps"]) >= 10
        and int(config["n_test_scenes"]) >= 4
        and bool(config.get("grid_square_cell", False))
        and int(config.get("grid_nx", config["grid_size"])) == 128
        and int(config.get("grid_ny", config["grid_size"])) == 128
        and np.isclose(float(config.get("grid_x_min_m", -config["grid_range_m"])), -20.0)
        and np.isclose(float(config.get("grid_x_max_m", config["grid_range_m"])), 20.0)
        and np.isclose(float(config.get("grid_y_min_m", 0.0)), 0.0)
        and np.isclose(float(config.get("grid_y_max_m", config["grid_range_m"])), 40.0)
    )
    if canonical:
        return "main_canonical_uniform_grid", "Canonical uniform-grid result", caveats
    caveats.insert(
        0,
        "This run does not meet the lecture canonical setting "
        "(200 MHz, N_fast=1024, wall spacing ≤0.35 m, ≥10 frames, ≥4 test scenes, "
        "128×128 uniform square cells over x=[-20,20] m/y=[0,40] m).",
    )
    return "main_preview", "Preview result", caveats


def _contract_value(contract: dict, key: str) -> object:
    if key in contract:
        return contract[key]
    # Older train.py metrics use data_contract without n_steps.
    return None


def metrics_mismatch_warnings(config: dict, metrics: dict) -> list[str]:
    """Detect when an external metrics file belongs to a different dataset."""

    contract = metrics.get("data_contract", {}) if isinstance(metrics, dict) else {}
    if not contract:
        return ["Metrics file has no data_contract block; recomputing metrics from this dataset."]

    checks = [
        ("n_test_detections", int(config["n_test_detections"]), int),
        ("n_test_scenes", int(config["n_test_scenes"]), int),
        ("grid_size", int(config["grid_size"]), int),
        ("grid_range_m", float(config["grid_range_m"]), float),
        ("grid_nx", int(config.get("grid_nx", config["grid_size"])), int),
        ("grid_ny", int(config.get("grid_ny", config["grid_size"])), int),
        ("grid_x_min_m", float(config.get("grid_x_min_m", -config["grid_range_m"])), float),
        ("grid_x_max_m", float(config.get("grid_x_max_m", config["grid_range_m"])), float),
        ("grid_y_min_m", float(config.get("grid_y_min_m", 0.0)), float),
        ("grid_y_max_m", float(config.get("grid_y_max_m", config["grid_range_m"])), float),
        ("grid_cell_x_m", float(config.get("grid_cell_x_m", np.nan)), float),
        ("grid_cell_y_m", float(config.get("grid_cell_y_m", np.nan)), float),
        ("radar_max_range_m", float(config.get("radar_max_range_m", config["grid_range_m"])), float),
        ("wall_spacing_m", float(config["wall_spacing_m"]) if config["wall_spacing_m"] is not None else None, float),
        ("radar_bw_hz", float(config["radar_bw_mhz"]) * 1e6, float),
        ("radar_n_fast", int(config["radar_n_fast"]), int),
        ("radar_range_res_m", float(config["range_resolution_m"]), float),
    ]
    warnings: list[str] = []
    for key, expected, caster in checks:
        observed = _contract_value(contract, key)
        if observed is None or expected is None:
            continue
        observed = caster(observed)
        if caster is int:
            if observed != expected:
                warnings.append(f"metrics data_contract mismatch: {key} metrics={observed} report_dataset={expected}")
        else:
            if not np.isclose(float(observed), float(expected), rtol=1e-4, atol=1e-4):
                warnings.append(
                    f"metrics data_contract mismatch: {key} metrics={float(observed):.6g} "
                    f"report_dataset={float(expected):.6g}"
                )
    return warnings


def html_rows_from_dict(items: dict[str, object]) -> str:
    return "\n".join(
        f"<tr><th>{key}</th><td><code>{value}</code></td></tr>"
        for key, value in items.items()
    )


def make_report(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = MappingDetectionDataset(Path(args.dataset))
    metrics_path = Path(args.metrics)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = build_model(n_rx=dataset.x.shape[-1], grid_size=dataset.y.shape[-1]).to(device)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Refusing to generate a report with a random model."
        )
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    pred_dl = _predict_model_angles(model, dataset, device=device)
    pred_fft, pred_music = _predict_signal_processing_angles(dataset)
    pred_oracle = dataset.angle_deg.astype(np.float32)

    scene_idx = int(args.scene_idx)
    poses = poses_for_scene(dataset, scene_idx)
    gt = dataset.gt_ogm[scene_idx]

    oracle_single, _, _ = build_ogm_for_scene(dataset, pred_oracle, scene_idx, frame_limit=1)
    oracle_ego, _, _ = build_ogm_for_scene(dataset, pred_oracle, scene_idx, frame_limit=None)
    fft_ego, _, _ = build_ogm_for_scene(dataset, pred_fft, scene_idx)
    music_ego, _, _ = build_ogm_for_scene(dataset, pred_music, scene_idx)
    dl_ego, _, _ = build_ogm_for_scene(dataset, pred_dl, scene_idx)

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.4), constrained_layout=True)
    plot_panel(axes[0, 0], gt, "GT static map", dataset, poses)
    plot_panel(axes[0, 1], oracle_single, "Oracle DoA\nsingle frame", dataset, poses[:1])
    plot_panel(axes[0, 2], oracle_ego, "Oracle DoA\nego-motion map", dataset, poses)
    plot_panel(axes[1, 0], fft_ego, "Signal processing\nangle FFT map", dataset, poses)
    plot_panel(axes[1, 1], music_ego, "Signal processing\nMUSIC map", dataset, poses)
    plot_panel(axes[1, 2], dl_ego, "Deep learning\nRadarCubeDoANet map", dataset, poses)
    config = {
        "dataset": str(Path(args.dataset)),
        "checkpoint": str(checkpoint),
        "metrics": str(metrics_path),
        "n_test_detections": int(len(dataset)),
        "n_test_scenes": int(dataset.gt_ogm.shape[0]),
        "grid_size": int(dataset.grid_size),
        "grid_nx": int(dataset.grid_nx),
        "grid_ny": int(dataset.grid_ny),
        "grid_range_m": float(dataset.grid_range_m),
        "grid_x_min_m": float(dataset.grid_spec.x_min_m),
        "grid_x_max_m": float(dataset.grid_spec.x_max_m),
        "grid_y_min_m": float(dataset.grid_spec.y_min_m),
        "grid_y_max_m": float(dataset.grid_spec.y_max_m),
        "grid_cell_x_m": float(dataset.grid_cell_x_m),
        "grid_cell_y_m": float(dataset.grid_cell_y_m),
        "grid_square_cell": bool(dataset.grid_spec.is_square_cell),
        "radar_max_range_m": float(dataset.radar_max_range_m),
        "radar_bw_mhz": float(dataset.radar_bw_hz / 1e6),
        "radar_n_fast": int(dataset.radar_n_fast),
        "range_resolution_m": float(dataset.radar_range_res_m),
        "wall_spacing_m": float(dataset.wall_spacing_m) if np.isfinite(dataset.wall_spacing_m) else None,
        "n_steps": int(dataset.n_steps),
    }
    report_kind, report_label, caveats = classify_report(config)

    fig.suptitle(
        f"P03 {report_label}: DoA quality as radar-map quality\n"
        f"BW={dataset.radar_bw_hz/1e6:g} MHz, ΔR={dataset.radar_range_res_m:.2f} m, "
        f"frames={dataset.n_steps}, wall spacing={dataset.wall_spacing_m if np.isfinite(dataset.wall_spacing_m) else float('nan'):.2f} m",
        fontsize=13,
    )
    panel_path = out_dir / "p03_main_result_maps.png"
    fig.savefig(panel_path, dpi=180)
    plt.close(fig)

    external_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    warnings = metrics_mismatch_warnings(config, external_metrics)
    metrics = build_report_metrics(dataset, pred_dl, pred_fft, pred_music, pred_oracle)
    metrics_source = "recomputed_by_report_from_loaded_checkpoint"

    rows = [
        metric_row(metrics, "oracle_gt_doa", "map_from_oracle_gt_doa", "Oracle GT DoA"),
        metric_row(metrics, "signal_processing_music_doa", "map_from_music_doa", "MUSIC"),
        metric_row(metrics, "signal_processing_angle_fft_doa", "map_from_angle_fft_doa", "Coarse angle FFT"),
        metric_row(metrics, "deep_learning_doa", "map_from_deep_learning_doa", "RadarCubeDoANet"),
    ]
    table_rows = "\n".join(
        "<tr>"
        f"<td>{r['method']}</td>"
        f"<td>{format_float(r['mae_deg'])}</td>"
        f"<td>{format_float(r['rmse_deg'])}</td>"
        f"<td>{format_float(r['within_2deg_acc'])}</td>"
        f"<td>{format_float(r['ogm_iou'])}</td>"
        f"<td>{format_float(r['ogm_f1'])}</td>"
        f"<td>{format_float(r['pc_iou'])}</td>"
        f"<td>{format_float(r['point_error_mean_m'])}</td>"
        f"<td>{format_float(r['point_error_p90_m'])}</td>"
        "</tr>"
        for r in rows
    )

    best_sp = max(rows[1:3], key=lambda r: float(r["pc_iou"]) if np.isfinite(r["pc_iou"]) else -1.0)
    dl = rows[3]
    oracle = rows[0]
    note = (
        f"In this run, {best_sp['method']} is the stronger signal-processing point-cloud baseline "
        f"(point-grid IoU {format_float(best_sp['pc_iou'])}). "
        f"RadarCubeDoANet reaches point-grid IoU {format_float(dl['pc_iou'])}; "
        f"the oracle point-cloud grid IoU is {format_float(oracle['pc_iou'])} with zero DoA point error. "
        "OGM threshold metrics can be slightly non-monotonic because the inverse-sensor model and rasterized GT are discrete, "
        "so read OGM together with point-cloud IoU and point-error. "
        "The visual gap between single-frame and ego-motion oracle maps is the key lecture point: "
        "known ego motion converts repeated range-bearing detections into a sharper world map."
    )

    config.update({
        "report_kind": report_kind,
        "report_label": report_label,
        "metrics_source_used": metrics_source,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "caveats": caveats,
        "warnings": warnings,
    })
    config_path = out_dir / "p03_main_result_config.json"
    config_path.write_text(json.dumps({"config": config, "metrics_rows": rows}, indent=2), encoding="utf-8")
    caveat_html = "\n".join(f"<li>{c}</li>" for c in caveats)
    warning_html = "\n".join(f"<li>{w}</li>" for w in warnings)
    warning_block = f"<div class=\"warn\"><strong>Reference metrics warning.</strong><ul>{warning_html}</ul><p>Report table metrics were recomputed from the loaded checkpoint and report dataset to avoid mixing result pools.</p></div>" if warnings else ""
    config_rows = html_rows_from_dict({
        "Radar": (
            f"{config['radar_bw_mhz']:g} MHz bandwidth, N_fast={config['radar_n_fast']}, "
            f"range resolution ≈ {config['range_resolution_m']:.3f} m"
        ),
        "Scene": f"dense point-scatterer corridor, wall spacing {config['wall_spacing_m']} m",
        "Ego trajectory": f"{config['n_steps']} frames with perfect ego pose",
        "Evaluation set": f"{config['n_test_scenes']} held-out scenes, {config['n_test_detections']} detections",
        "Map grid": (
            f"{config['grid_nx']}×{config['grid_ny']}, "
            f"x=[{config['grid_x_min_m']:.1f},{config['grid_x_max_m']:.1f}] m, "
            f"y=[{config['grid_y_min_m']:.1f},{config['grid_y_max_m']:.1f}] m, "
            f"cell={config['grid_cell_x_m']:.4f}×{config['grid_cell_y_m']:.4f} m"
        ),
        "Radar max range": f"{config['radar_max_range_m']} m",
    })

    html = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>P03 {report_label} — DoA to Radar Map</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 32px; color: #111827; line-height: 1.55; }}
main {{ max-width: 1120px; margin: auto; }}
.note {{ background: #eff6ff; border-left: 4px solid #2563eb; padding: 12px 16px; margin: 18px 0; }}
.warn {{ background: #fff7ed; border-left: 4px solid #f97316; padding: 12px 16px; margin: 18px 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 18px 0 28px; }}
th, td {{ border: 1px solid #d1d5db; padding: 7px 9px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #f3f4f6; }}
img {{ max-width: 100%; border: 1px solid #d1d5db; border-radius: 8px; }}
code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }}
</style>
</head>
<body><main>
<h1>P03 {report_label} — DoA to Radar Map</h1>
<div class="note"><strong>Lecture message.</strong> DoA is not only an angle number. Once range-bearing detections are projected through a moving ego radar, DoA quality becomes visible as point-cloud and probabilistic-map quality.</div>
<div class="note"><strong>Run summary.</strong> {note}</div>
{warning_block}
<h2>Map panels</h2>
<img alt="P03 main-result map panels" src="data:image/png;base64,{encode_image(panel_path)}" />
<h2>Metrics</h2>
<table>
<thead><tr>
<th>Method</th><th>DoA MAE [deg]</th><th>DoA RMSE [deg]</th><th>≤2° acc.</th>
<th>OGM IoU</th><th>OGM F1</th><th>Point-grid IoU</th><th>Mean point error [m]</th><th>P90 point error [m]</th>
</tr></thead>
<tbody>{table_rows}</tbody>
</table>
<h2>Configuration</h2>
<table><tbody>{config_rows}</tbody></table>
<h2>Interpretation caveats</h2>
<ul>{caveat_html}</ul>
<h2>Suggested appendix order</h2>
<p>After this main figure, use the angular-projection appendix for <code>R·Δθ</code>, the range-resolution appendix for bandwidth/range-cell limits, the off-grid appendix for raster-IoU caveats, and the ego-motion appendix for pose-error limits.</p>
</main></body></html>"""
    html_path = out_dir / "p03_main_result_report.html"
    html_path.write_text(html, encoding="utf-8")
    return {
        "html": str(html_path),
        "panel": str(panel_path),
        "config": str(config_path),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate P03 main-result report")
    parser.add_argument("--dataset", default=str(BASE / "data_mapping" / "test.h5"))
    parser.add_argument("--checkpoint", default=str(BASE / "artifacts" / "best_model.pt"))
    parser.add_argument("--metrics", default=str(BASE / "artifacts" / "metrics.json"))
    parser.add_argument("--out_dir", default=str(BASE / "artifacts" / "main_result"))
    parser.add_argument("--scene_idx", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    result = make_report(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
