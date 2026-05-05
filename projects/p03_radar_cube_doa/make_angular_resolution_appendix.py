#!/usr/bin/env python3
"""Generate the P03 angular/cross-range resolution appendix.

This report converts DoA error into map-space lateral error and runs a compact
single-frame point-pair probe at controlled angular separations.  The pair probe
uses the same RD-selected antenna-vector interface as P03, but each point is
simulated independently; it is therefore an estimator/projection teaching probe,
not a simultaneous multi-target super-resolution benchmark.
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
import torch

from generate_data import ANGLE_GRID, build_p03_radar
from generate_mapping_data import _rcs_for_requested_snr
from mapping import RadarMeasurement, simulate_rd_selected_vector
from model import build_model
from train import (
    angle_fft_spectrum,
    estimate_angle_from_spectrum,
    music_spectrum_single_snapshot,
    steering_matrix,
)

BASE = Path(__file__).resolve().parent


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def fmt(v: object, digits: int = 3) -> str:
    if isinstance(v, (float, int)) and np.isfinite(v):
        return f"{float(v):.{digits}f}"
    return "—"


def lateral_separation_m(range_m: float, angle_sep_deg: float) -> float:
    return float(2.0 * range_m * np.sin(np.deg2rad(angle_sep_deg) / 2.0))


def xy_from_range_angle(range_m: float, angle_deg: float) -> tuple[float, float]:
    a = np.deg2rad(float(angle_deg))
    return float(range_m * np.sin(a)), float(range_m * np.cos(a))


def predict_doanet(model: torch.nn.Module, x_rows: np.ndarray, device: str) -> np.ndarray:
    x = torch.as_tensor(x_rows, dtype=torch.float32, device=device)
    with torch.no_grad():
        spec = torch.sigmoid(model(x)).detach().cpu().numpy()
    return np.asarray([estimate_angle_from_spectrum(s) for s in spec], dtype=np.float32)


def simulate_pair_predictions(
    model: torch.nn.Module,
    radar,
    range_m: float,
    sep_deg: float,
    device: str,
    seed: int,
    snr_db: float = 20.0,
) -> dict:
    true_angles = np.asarray([-sep_deg / 2.0, sep_deg / 2.0], dtype=np.float32)
    x_rows = []
    x_complex = []
    for j, angle in enumerate(true_angles):
        meas = RadarMeasurement(
            range_m=float(range_m),
            angle_deg=float(angle),
            radial_velocity_mps=0.0,
            world_x_m=float(range_m * np.sin(np.deg2rad(angle))),
            world_y_m=float(range_m * np.cos(np.deg2rad(angle))),
            target_id=j,
            target_type="angular_pair_probe",
            is_dynamic=False,
        )
        rcs = _rcs_for_requested_snr(radar, float(range_m), float(angle))
        ant_vec, _ = simulate_rd_selected_vector(
            radar,
            meas,
            snr_db=snr_db,
            rcs_m2=rcs,
            seed=seed + j,
        )
        x_complex.append(ant_vec)
        x_rows.append(np.stack([ant_vec.real, ant_vec.imag], axis=0).astype(np.float32))
    x_rows = np.stack(x_rows, axis=0)
    A = steering_matrix(ANGLE_GRID, n_rx=radar.N_rx)
    pred = {
        "Oracle GT DoA": true_angles.astype(np.float32),
        "MUSIC": np.asarray([
            estimate_angle_from_spectrum(music_spectrum_single_snapshot(x, A)) for x in x_complex
        ], dtype=np.float32),
        "RadarCubeDoANet": predict_doanet(model, x_rows, device=device),
        "Coarse angle FFT": np.asarray([
            estimate_angle_from_spectrum(angle_fft_spectrum(x)) for x in x_complex
        ], dtype=np.float32),
    }
    rows = []
    for method, pred_angles in pred.items():
        angle_err = np.abs(pred_angles - true_angles)
        true_xy = np.asarray([xy_from_range_angle(range_m, a) for a in true_angles])
        pred_xy = np.asarray([xy_from_range_angle(range_m, a) for a in pred_angles])
        xy_err = np.linalg.norm(pred_xy - true_xy, axis=1)
        rows.append({
            "range_m": float(range_m),
            "angle_sep_deg": float(sep_deg),
            "true_lateral_sep_m": lateral_separation_m(range_m, sep_deg),
            "method": method,
            "true_angles_deg": [float(a) for a in true_angles],
            "pred_angles_deg": [float(a) for a in pred_angles],
            "pred_lateral_sep_m": float(abs(pred_xy[1, 0] - pred_xy[0, 0])),
            "mean_abs_angle_error_deg": float(np.mean(angle_err)),
            "mean_lateral_point_error_m": float(np.mean(xy_err)),
            "pair_order_preserved": bool(pred_angles[0] < pred_angles[1]),
        })
    return {"rows": rows, "predictions": pred}


def make_report(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    radar = build_p03_radar(bandwidth_hz=args.radar_bw_mhz * 1e6, n_fast=args.radar_n_fast)
    model = build_model(n_rx=radar.N_rx, grid_size=len(ANGLE_GRID)).to(device)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    ranges = [10.0, 20.0, 30.0]
    separations = [0.5, 1.0, 2.0, 5.0]

    conversion_rows = [
        {"range_m": r, "angle_sep_deg": s, "lateral_sep_m": lateral_separation_m(r, s)}
        for r in ranges
        for s in separations
    ]

    probe_rows = []
    panel_predictions = {}
    seed = int(args.seed)
    for r in ranges:
        for s in separations:
            result = simulate_pair_predictions(model, radar, r, s, device=device, seed=seed, snr_db=args.snr_db)
            probe_rows.extend(result["rows"])
            if abs(r - 20.0) < 1e-6:
                panel_predictions[s] = result["predictions"]
            seed += 101

    metrics = {
        "config": {
            "checkpoint": str(Path(args.checkpoint)),
            "reference_metrics": str(Path(args.reference_metrics)) if args.reference_metrics else None,
            "radar_bw_mhz": float(args.radar_bw_mhz),
            "radar_n_fast": int(args.radar_n_fast),
            "range_resolution_m": float(radar.range_res),
            "snr_db": float(args.snr_db),
            "assumption": "Point-pair probe simulates each point independently; this is not a same-RD simultaneous multi-target resolution benchmark.",
        },
        "conversion_table": conversion_rows,
        "probe_rows": probe_rows,
    }
    metrics_path = out_dir / "p03_angular_resolution_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Conversion figure.
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    for r in ranges:
        vals = [lateral_separation_m(r, s) for s in separations]
        ax.plot(separations, vals, "o-", label=f"R={r:g} m")
    ax.set_xlabel("Angular separation [deg]")
    ax.set_ylabel("Lateral separation [m]")
    ax.set_title("Cross-range separation scales approximately as R·Δθ")
    ax.grid(True, alpha=0.25)
    ax.legend()
    conversion_path = out_dir / "p03_angular_resolution_conversion.png"
    fig.savefig(conversion_path, dpi=180)
    plt.close(fig)

    # Pair projection panels at 20 m.
    methods = ["Oracle GT DoA", "MUSIC", "RadarCubeDoANet", "Coarse angle FFT"]
    fig, axes = plt.subplots(len(methods), len(separations), figsize=(13.2, 9.0), constrained_layout=True)
    for row_idx, method in enumerate(methods):
        for col_idx, sep in enumerate(separations):
            ax = axes[row_idx, col_idx]
            true_angles = np.asarray([-sep / 2.0, sep / 2.0], dtype=np.float32)
            true_xy = np.asarray([xy_from_range_angle(20.0, a) for a in true_angles])
            pred_angles = panel_predictions[sep][method]
            pred_xy = np.asarray([xy_from_range_angle(20.0, a) for a in pred_angles])
            ax.scatter(true_xy[:, 0], true_xy[:, 1], marker="x", s=70, c="black", label="true")
            ax.scatter(pred_xy[:, 0], pred_xy[:, 1], marker="o", s=42, c="#2563eb", label="pred")
            for t, p in zip(true_xy, pred_xy):
                ax.plot([t[0], p[0]], [t[1], p[1]], "r-", alpha=0.45, lw=1.0)
            ax.set_xlim(-2.2, 2.2)
            ax.set_ylim(19.4, 20.2)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25)
            if row_idx == 0:
                ax.set_title(f"Δθ={sep:g}°")
            if col_idx == 0:
                ax.set_ylabel(method)
            if row_idx == len(methods) - 1:
                ax.set_xlabel("x [m]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("20 m point-pair projection: DoA error becomes cross-range map error")
    panels_path = out_dir / "p03_angular_resolution_pair_projection.png"
    fig.savefig(panels_path, dpi=180)
    plt.close(fig)

    reference_note = (
        "Use the analytic conversion table below with the DoA error values from the main result."
    )
    if args.reference_metrics and Path(args.reference_metrics).exists():
        ref = json.loads(Path(args.reference_metrics).read_text())
        dl_mae = float(ref.get("deep_learning_doa", {}).get("mae_deg", float("nan")))
        fft_mae = float(ref.get("signal_processing_angle_fft_doa", {}).get("mae_deg", float("nan")))
        if np.isfinite(dl_mae) and np.isfinite(fft_mae):
            dl_lat = float(20.0 * np.deg2rad(dl_mae))
            fft_lat = float(20.0 * np.deg2rad(fft_mae))
            reference_note = (
                "Using the supplied main-result metrics as examples, "
                f"RadarCubeDoANet MAE {dl_mae:.3f}° corresponds to about {dl_lat:.2f} m lateral error at 20 m, "
                f"while coarse angle FFT MAE {fft_mae:.3f}° corresponds to about {fft_lat:.2f} m."
            )

    def conversion_html() -> str:
        return "\n".join(
            f"<tr><td>{fmt(r['range_m'], 0)}</td><td>{fmt(r['angle_sep_deg'], 1)}</td><td>{fmt(r['lateral_sep_m'])}</td></tr>"
            for r in conversion_rows
        )

    def probe_html() -> str:
        selected = [r for r in probe_rows if abs(r["range_m"] - 20.0) < 1e-6]
        return "\n".join(
            "<tr>"
            f"<td>{fmt(r['angle_sep_deg'], 1)}</td><td>{r['method']}</td>"
            f"<td>{fmt(r['true_lateral_sep_m'])}</td><td>{fmt(r['pred_lateral_sep_m'])}</td>"
            f"<td>{fmt(r['mean_abs_angle_error_deg'])}</td><td>{fmt(r['mean_lateral_point_error_m'])}</td>"
            f"<td>{'yes' if r['pair_order_preserved'] else 'no'}</td>"
            "</tr>"
            for r in selected
        )

    html = f"""<!doctype html>
<html lang=\"ko\"><head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>P03 Angular / Cross-Range Projection Appendix</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 32px; color: #111827; line-height: 1.55; }}
main {{ max-width: 1120px; margin: auto; }}
.note {{ background: #eff6ff; border-left: 4px solid #2563eb; padding: 12px 16px; margin: 18px 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 18px 0 28px; }}
th, td {{ border: 1px solid #d1d5db; padding: 7px 9px; text-align: right; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
th {{ background: #f3f4f6; }}
img {{ max-width: 100%; border: 1px solid #d1d5db; border-radius: 8px; margin: 12px 0 24px; }}
code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }}
</style></head><body><main>
<h1>P03 Angular / Cross-Range Projection Appendix</h1>
<div class=\"note\"><strong>Lecture message.</strong> A small DoA error becomes a lateral map error of approximately <code>R·Δθ</code> radians. {reference_note}</div>
<div class=\"note\"><strong>Scope caveat.</strong> The point-pair probe simulates each point independently through the P03 RD-selected antenna-vector interface. It is intended to teach projection sensitivity; it is not a simultaneous same-RD multi-target super-resolution benchmark.</div>
<ul>
<li>Checkpoint: <code>{metrics['config']['checkpoint']}</code></li>
<li>Radar: {args.radar_bw_mhz:g} MHz, N_fast={args.radar_n_fast}, ΔR={radar.range_res:.3f} m, SNR={args.snr_db:g} dB.</li>
</ul>
<h2>Analytic conversion</h2>
<img alt=\"P03 angular to cross-range conversion\" src=\"data:image/png;base64,{encode_image(conversion_path)}\" />
<table><thead><tr><th>Range [m]</th><th>Angular separation [deg]</th><th>Lateral separation [m]</th></tr></thead><tbody>{conversion_html()}</tbody></table>
<h2>20 m point-pair projection probe</h2>
<img alt=\"P03 angular point-pair projection\" src=\"data:image/png;base64,{encode_image(panels_path)}\" />
<table><thead><tr><th>Δθ [deg]</th><th>Method</th><th>True lateral sep. [m]</th><th>Pred lateral sep. [m]</th><th>Mean angle err. [deg]</th><th>Mean lateral err. [m]</th><th>Order preserved</th></tr></thead><tbody>{probe_html()}</tbody></table>
<p><strong>Lecture use:</strong> pair this appendix with the main map report to explain why sub-degree DoA matters before students inspect OGM/point-cloud maps.</p>
</main></body></html>"""
    html_path = out_dir / "p03_angular_resolution_report.html"
    html_path.write_text(html, encoding="utf-8")
    return {
        "html": str(html_path),
        "conversion": str(conversion_path),
        "panels": str(panels_path),
        "metrics": str(metrics_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate P03 angular/cross-range resolution appendix")
    parser.add_argument("--checkpoint", default=str(BASE / "artifacts" / "best_model.pt"))
    parser.add_argument("--out_dir", default=str(BASE / "artifacts" / "angular_resolution_appendix"))
    parser.add_argument("--reference_metrics", default=str(BASE / "artifacts" / "runs" / "doanet_200_canonical_eval_20260502" / "metrics.json"))
    parser.add_argument("--radar_bw_mhz", type=float, default=200.0)
    parser.add_argument("--radar_n_fast", type=int, default=1024)
    parser.add_argument("--snr_db", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    print(json.dumps(make_report(args), indent=2))


if __name__ == "__main__":
    main()
