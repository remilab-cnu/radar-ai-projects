#!/usr/bin/env python3
"""Generate the P03 ego-motion-error appendix report.

This appendix intentionally fixes DoA/range to oracle values and perturbs only
world-frame ego poses.  It therefore isolates localization/odometry error from
the main P03 DoA-estimator comparison.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mapping import (
    EgoPose,
    accumulate_probability_map,
    map_metrics,
    perturb_ego_poses,
    point_cloud_from_measurements,
    point_cloud_grid,
    point_from_measurement,
)
from train import MappingDetectionDataset

BASE = Path(__file__).resolve().parent


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def fmt(v: object, digits: int = 3) -> str:
    if isinstance(v, (float, int)) and np.isfinite(v):
        return f"{float(v):.{digits}f}"
    return "—"


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


def scene_frame_lists(
    dataset: MappingDetectionDataset,
    scene_idx: int,
) -> tuple[list[list[float]], list[list[float]], np.ndarray]:
    per_frame_angles = [[] for _ in range(dataset.n_steps)]
    per_frame_ranges = [[] for _ in range(dataset.n_steps)]
    det_mask = (dataset.scene_idx == scene_idx) & (~dataset.is_dynamic)
    det_indices = np.nonzero(det_mask)[0]
    for i in det_indices:
        frame = int(dataset.frame_idx[i])
        if 0 <= frame < dataset.n_steps:
            per_frame_angles[frame].append(float(dataset.angle_deg[i]))
            per_frame_ranges[frame].append(float(dataset.range_m[i]))
    return per_frame_angles, per_frame_ranges, det_indices


def build_map(
    dataset: MappingDetectionDataset,
    poses: list[EgoPose],
    per_frame_angles: list[list[float]],
    per_frame_ranges: list[list[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def detection_pose_error(
    dataset: MappingDetectionDataset,
    scene_idx: int,
    true_poses: list[EgoPose],
    perturbed_poses: list[EgoPose],
    det_indices: np.ndarray,
) -> np.ndarray:
    errs = []
    for i in det_indices:
        frame = int(dataset.frame_idx[i])
        true_pt = point_from_measurement(true_poses[frame], float(dataset.range_m[i]), float(dataset.angle_deg[i]))
        pert_pt = point_from_measurement(perturbed_poses[frame], float(dataset.range_m[i]), float(dataset.angle_deg[i]))
        errs.append(float(np.linalg.norm(pert_pt - true_pt)))
    return np.asarray(errs, dtype=np.float64)


def evaluate_profile(
    dataset: MappingDetectionDataset,
    name: str,
    yaw_bias_deg: float = 0.0,
    dx_m: float = 0.0,
    dy_m: float = 0.0,
    drift_y_per_step_m: float = 0.0,
) -> dict[str, float | str]:
    ogm_rows = []
    pc_rows = []
    point_errors = []
    for scene_idx in range(dataset.gt_ogm.shape[0]):
        true_poses = poses_for_scene(dataset, scene_idx)
        pert_poses = perturb_ego_poses(
            true_poses,
            dx_m=dx_m,
            dy_m=dy_m,
            yaw_bias_deg=yaw_bias_deg,
            drift_per_step_m=(0.0, drift_y_per_step_m),
        )
        per_frame_angles, per_frame_ranges, det_indices = scene_frame_lists(dataset, scene_idx)
        _, ogm_bin, pc_grid = build_map(dataset, pert_poses, per_frame_angles, per_frame_ranges)
        gt = dataset.gt_ogm[scene_idx]
        ogm_rows.append(map_metrics(gt, ogm_bin))
        pc_rows.append(map_metrics(gt, pc_grid))
        point_errors.append(detection_pose_error(dataset, scene_idx, true_poses, pert_poses, det_indices))
    err = np.concatenate(point_errors) if point_errors else np.asarray([], dtype=np.float64)
    return {
        "name": name,
        "yaw_bias_deg": float(yaw_bias_deg),
        "dx_m": float(dx_m),
        "dy_m": float(dy_m),
        "drift_y_per_step_m": float(drift_y_per_step_m),
        "mean_point_error_m": float(np.mean(err)) if len(err) else float("nan"),
        "p90_point_error_m": float(np.percentile(err, 90)) if len(err) else float("nan"),
        "ogm_iou": float(np.mean([m["iou"] for m in ogm_rows])),
        "ogm_f1": float(np.mean([m["f1"] for m in ogm_rows])),
        "point_grid_iou": float(np.mean([m["iou"] for m in pc_rows])),
        "point_grid_f1": float(np.mean([m["f1"] for m in pc_rows])),
    }


def flip_for_plot(grid: np.ndarray) -> np.ndarray:
    return np.flipud(np.asarray(grid))


def plot_map_panel(ax, grid: np.ndarray, dataset: MappingDetectionDataset, title: str, poses: list[EgoPose]) -> None:
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
    ax.plot([p.x_m for p in poses], [p.y_m for p in poses], "c.-", lw=1.2, ms=3)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-12, 12)
    ax.set_ylim(0, 34)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def make_report(args: argparse.Namespace) -> dict:
    dataset = MappingDetectionDataset(Path(args.dataset))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    yaw_values = [0.0, 0.25, 0.5, 1.0, 2.0]
    translation_values = [0.0, 0.1, 0.25, 0.5, 1.0]
    drift_values = [0.0, 0.02, 0.05, 0.10]

    yaw_rows = [evaluate_profile(dataset, f"yaw {v:g}°", yaw_bias_deg=v) for v in yaw_values]
    trans_rows = [evaluate_profile(dataset, f"lateral dx {v:g} m", dx_m=v) for v in translation_values]
    drift_rows = [evaluate_profile(dataset, f"forward drift {v:g} m/step", drift_y_per_step_m=v) for v in drift_values]

    metrics = {
        "config": {
            "dataset": str(Path(args.dataset)),
            "radar_bw_mhz": float(dataset.radar_bw_hz / 1e6),
            "radar_n_fast": int(dataset.radar_n_fast),
            "range_resolution_m": float(dataset.radar_range_res_m),
            "wall_spacing_m": float(dataset.wall_spacing_m),
            "grid_nx": int(dataset.grid_nx),
            "grid_ny": int(dataset.grid_ny),
            "grid_x_min_m": float(dataset.grid_spec.x_min_m),
            "grid_x_max_m": float(dataset.grid_spec.x_max_m),
            "grid_y_min_m": float(dataset.grid_spec.y_min_m),
            "grid_y_max_m": float(dataset.grid_spec.y_max_m),
            "grid_cell_x_m": float(dataset.grid_cell_x_m),
            "grid_cell_y_m": float(dataset.grid_cell_y_m),
            "n_steps": int(dataset.n_steps),
            "n_test_scenes": int(dataset.gt_ogm.shape[0]),
            "n_test_detections": int(len(dataset)),
            "assumption": "GT DoA/range fixed; only ego poses are perturbed.",
        },
        "yaw_bias_sweep": yaw_rows,
        "translation_bias_sweep": trans_rows,
        "forward_drift_sweep": drift_rows,
    }
    metrics_path = out_dir / "p03_ego_motion_error_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Curves.
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), constrained_layout=True)
    axes[0].plot([r["yaw_bias_deg"] for r in yaw_rows], [r["mean_point_error_m"] for r in yaw_rows], "o-", label="mean")
    axes[0].plot([r["yaw_bias_deg"] for r in yaw_rows], [r["p90_point_error_m"] for r in yaw_rows], "s--", label="p90")
    axes[0].set_title("Yaw-bias-only error")
    axes[0].set_xlabel("Yaw bias [deg]")
    axes[0].set_ylabel("World point error [m]")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot([r["dx_m"] for r in trans_rows], [r["mean_point_error_m"] for r in trans_rows], "o-", label="mean")
    axes[1].plot([r["dx_m"] for r in trans_rows], [r["p90_point_error_m"] for r in trans_rows], "s--", label="p90")
    axes[1].set_title("Lateral translation-bias error")
    axes[1].set_xlabel("Lateral pose bias [m]")
    axes[1].set_ylabel("World point error [m]")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    axes[2].plot([r["drift_y_per_step_m"] for r in drift_rows], [r["mean_point_error_m"] for r in drift_rows], "o-", label="mean")
    axes[2].plot([r["drift_y_per_step_m"] for r in drift_rows], [r["p90_point_error_m"] for r in drift_rows], "s--", label="p90")
    axes[2].set_title("Forward drift accumulation")
    axes[2].set_xlabel("Forward drift [m/frame]")
    axes[2].set_ylabel("World point error [m]")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()
    fig.suptitle("P03 ego-motion-error appendix: oracle DoA/range, perturbed pose only")
    curve_path = out_dir / "p03_ego_motion_error_curves.png"
    fig.savefig(curve_path, dpi=180)
    plt.close(fig)

    # Map panels for one scene.
    scene_idx = int(args.scene_idx)
    true_poses = poses_for_scene(dataset, scene_idx)
    per_frame_angles, per_frame_ranges, _ = scene_frame_lists(dataset, scene_idx)
    panel_profiles = [
        ("GT static map", None),
        ("Oracle pose", perturb_ego_poses(true_poses)),
        ("Yaw +1°", perturb_ego_poses(true_poses, yaw_bias_deg=1.0)),
        ("Yaw +2°", perturb_ego_poses(true_poses, yaw_bias_deg=2.0)),
        ("Lateral +0.5 m", perturb_ego_poses(true_poses, dx_m=0.5)),
        ("Forward drift 0.1 m/frame", perturb_ego_poses(true_poses, drift_per_step_m=(0.0, 0.10))),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.4), constrained_layout=True)
    for ax, (title, poses) in zip(axes.ravel(), panel_profiles):
        if poses is None:
            plot_map_panel(ax, dataset.gt_ogm[scene_idx], dataset, title, true_poses)
        else:
            ogm_prob, _, _ = build_map(dataset, poses, per_frame_angles, per_frame_ranges)
            plot_map_panel(ax, ogm_prob, dataset, title, poses)
    fig.suptitle("P03 ego-motion-error map panels: GT DoA/range fixed")
    panel_path = out_dir / "p03_ego_motion_error_maps.png"
    fig.savefig(panel_path, dpi=180)
    plt.close(fig)

    def rows_html(rows: list[dict]) -> str:
        return "\n".join(
            "<tr>"
            f"<td>{r['name']}</td>"
            f"<td>{fmt(r['mean_point_error_m'])}</td>"
            f"<td>{fmt(r['p90_point_error_m'])}</td>"
            f"<td>{fmt(r['point_grid_iou'])}</td>"
            f"<td>{fmt(r['ogm_iou'])}</td>"
            "</tr>"
            for r in rows
        )

    html = f"""<!doctype html>
<html lang=\"ko\"><head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>P03 Ego-Motion Error Appendix</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 32px; color: #111827; line-height: 1.55; }}
main {{ max-width: 1120px; margin: auto; }}
.note {{ background: #eff6ff; border-left: 4px solid #2563eb; padding: 12px 16px; margin: 18px 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 18px 0 28px; }}
th, td {{ border: 1px solid #d1d5db; padding: 7px 9px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #f3f4f6; }}
img {{ max-width: 100%; border: 1px solid #d1d5db; border-radius: 8px; margin: 12px 0 24px; }}
code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }}
</style></head><body><main>
<h1>P03 Ego-Motion Error Appendix</h1>
<div class=\"note\"><strong>Isolation rule.</strong> This appendix uses GT DoA and GT range for every detection. Only the ego poses used for world projection are perturbed, so these curves are not DoA-estimator performance.</div>
<ul>
<li>Dataset: <code>{metrics['config']['dataset']}</code></li>
<li>Radar: {metrics['config']['radar_bw_mhz']:g} MHz, N_fast={metrics['config']['radar_n_fast']}, ΔR={metrics['config']['range_resolution_m']:.3f} m.</li>
<li>Scene: wall spacing {metrics['config']['wall_spacing_m']:.3f} m, {metrics['config']['n_steps']} frames, {metrics['config']['n_test_scenes']} scenes.</li>
<li>Map grid: {metrics['config']['grid_nx']}×{metrics['config']['grid_ny']}, x=[{metrics['config']['grid_x_min_m']:.1f},{metrics['config']['grid_x_max_m']:.1f}] m, y=[{metrics['config']['grid_y_min_m']:.1f},{metrics['config']['grid_y_max_m']:.1f}] m, cell={metrics['config']['grid_cell_x_m']:.4f}×{metrics['config']['grid_cell_y_m']:.4f} m.</li>
</ul>
<h2>Error curves</h2>
<img alt=\"P03 ego-motion error curves\" src=\"data:image/png;base64,{encode_image(curve_path)}\" />
<div class=\"note\"><strong>Metric reading.</strong> Mean/P90 point error is the primary monotonic ego-error signal. Point-grid IoU and OGM IoU are thresholded raster metrics, so small pose perturbations can occasionally improve overlap with a finite grid even though the physical point error increased.</div>
<h2>Map panels</h2>
<img alt=\"P03 ego-motion error maps\" src=\"data:image/png;base64,{encode_image(panel_path)}\" />
<h2>Yaw-bias sweep</h2>
<table><thead><tr><th>Profile</th><th>Mean point error [m]</th><th>P90 point error [m]</th><th>Point-grid IoU</th><th>OGM IoU</th></tr></thead><tbody>{rows_html(yaw_rows)}</tbody></table>
<h2>Lateral-translation sweep</h2>
<table><thead><tr><th>Profile</th><th>Mean point error [m]</th><th>P90 point error [m]</th><th>Point-grid IoU</th><th>OGM IoU</th></tr></thead><tbody>{rows_html(trans_rows)}</tbody></table>
<h2>Forward-drift sweep</h2>
<table><thead><tr><th>Profile</th><th>Mean point error [m]</th><th>P90 point error [m]</th><th>Point-grid IoU</th><th>OGM IoU</th></tr></thead><tbody>{rows_html(drift_rows)}</tbody></table>
<p><strong>Lecture use:</strong> keep the main P03 report as a DoA-isolation experiment with perfect ego-motion. Use this appendix when discussing odometry/calibration sensitivity.</p>
</main></body></html>"""
    html_path = out_dir / "p03_ego_motion_error_report.html"
    html_path.write_text(html, encoding="utf-8")
    return {
        "html": str(html_path),
        "curves": str(curve_path),
        "maps": str(panel_path),
        "metrics": str(metrics_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate P03 ego-motion-error appendix")
    parser.add_argument("--dataset", default=str(BASE / "data_mapping" / "test.h5"))
    parser.add_argument("--out_dir", default=str(BASE / "artifacts" / "ego_motion_appendix"))
    parser.add_argument("--scene_idx", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(make_report(args), indent=2))


if __name__ == "__main__":
    main()
