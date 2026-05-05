#!/usr/bin/env python3
"""Generate a DoA-only diagnostic report for P03 RadarCubeDoANet."""
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

from model import build_model
from train import (
    MappingDetectionDataset,
    _predict_model_angles,
    _predict_signal_processing_angles,
)

BASE = Path(__file__).resolve().parent


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def doa_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = np.abs(np.asarray(pred) - np.asarray(true))
    return {
        "n": int(len(err)),
        "mae_deg": float(np.mean(err)),
        "rmse_deg": float(np.sqrt(np.mean(err**2))),
        "median_deg": float(np.median(err)),
        "p90_deg": float(np.percentile(err, 90)),
        "p95_deg": float(np.percentile(err, 95)),
        "max_deg": float(np.max(err)),
        "within_1deg_acc": float(np.mean(err <= 1.0)),
        "within_2deg_acc": float(np.mean(err <= 2.0)),
        "within_5deg_acc": float(np.mean(err <= 5.0)),
    }


def binned_metrics(true: np.ndarray, pred: np.ndarray, bins: list[tuple[float, float]]) -> list[dict]:
    err = np.abs(np.asarray(pred) - np.asarray(true))
    rows = []
    for lo, hi in bins:
        mask = (true >= lo) & (true < hi)
        if np.any(mask):
            rows.append({
                "bin": f"{lo:g}..{hi:g}",
                "n": int(np.sum(mask)),
                "mae_deg": float(np.mean(err[mask])),
                "p90_deg": float(np.percentile(err[mask], 90)),
                "max_deg": float(np.max(err[mask])),
            })
    return rows


def snr_metrics(snr: np.ndarray, true: np.ndarray, pred: np.ndarray) -> list[dict]:
    err = np.abs(np.asarray(pred) - np.asarray(true))
    rows = []
    for lo, hi in [(5, 10), (10, 15), (15, 20), (20, 25.1), (25.1, 35)]:
        mask = (snr >= lo) & (snr < hi)
        if np.any(mask):
            rows.append({
                "bin": f"{lo:g}..{hi:g}",
                "n": int(np.sum(mask)),
                "mae_deg": float(np.mean(err[mask])),
                "p90_deg": float(np.percentile(err[mask], 90)),
            })
    return rows


def fmt(v: object, digits: int = 3) -> str:
    if isinstance(v, (float, int)) and np.isfinite(v):
        return f"{float(v):.{digits}f}"
    return "—"


def load_predictions(dataset_dir: Path, checkpoint: Path, device: str) -> dict:
    out = {}
    model = None
    for split in ("train", "val", "test"):
        ds = MappingDetectionDataset(dataset_dir / f"{split}.h5")
        if model is None:
            model = build_model(n_rx=ds.x.shape[-1], grid_size=ds.y.shape[-1]).to(device)
            model.load_state_dict(torch.load(checkpoint, map_location=device))
        pred_dl = _predict_model_angles(model, ds, device=device)
        pred_fft, pred_music = _predict_signal_processing_angles(ds)
        out[split] = {
            "dataset": ds,
            "true": ds.angle_deg,
            "dl": pred_dl,
            "music": pred_music,
            "angle_fft": pred_fft,
        }
    return out


def make_report(args: argparse.Namespace) -> dict:
    dataset_dir = Path(args.dataset_dir)
    checkpoint = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    preds = load_predictions(dataset_dir, checkpoint, device=device)

    method_keys = [
        ("dl", "RadarCubeDoANet"),
        ("music", "MUSIC"),
        ("angle_fft", "Coarse angle FFT"),
    ]
    summary = {}
    for split, row in preds.items():
        summary[split] = {
            label: doa_metrics(row["true"], row[key])
            for key, label in method_keys
        }

    test = preds["test"]
    true = test["true"]
    dl = test["dl"]
    music = test["music"]
    fft = test["angle_fft"]
    err_dl = np.abs(dl - true)
    idx = np.argsort(err_dl)[-12:][::-1]
    outliers = [
        {
            "idx": int(i),
            "true_deg": float(true[i]),
            "pred_deg": float(dl[i]),
            "err_deg": float(err_dl[i]),
            "snr_db": float(test["dataset"].snr_db[i]),
            "range_m": float(test["dataset"].range_m[i]),
            "frame": int(test["dataset"].frame_idx[i]),
            "target_id": int(test["dataset"].target_id[i]),
        }
        for i in idx
    ]

    # Figure 1: prediction scatter.
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
    for ax, pred, title in [
        (axes[0], dl, "RadarCubeDoANet"),
        (axes[1], music, "MUSIC"),
        (axes[2], fft, "Coarse angle FFT"),
    ]:
        ax.scatter(true, pred, s=8, alpha=0.55)
        ax.plot([-65, 65], [-65, 65], "k--", lw=1)
        ax.set_xlim(-65, 65)
        ax.set_ylim(-65, 65)
        ax.set_xlabel("true DoA [deg]")
        ax.set_ylabel("predicted DoA [deg]")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    fig.suptitle("P03 DoA-only diagnostic: prediction scatter on held-out test split")
    scatter_path = out_dir / "p03_doa_prediction_scatter.png"
    fig.savefig(scatter_path, dpi=180)
    plt.close(fig)

    # Figure 2: error by angle and SNR.
    angle_bins = [(-90, -45), (-45, -20), (-20, 0), (0, 20), (20, 45), (45, 90)]
    angle_rows = binned_metrics(true, dl, angle_bins)
    snr_rows = snr_metrics(test["dataset"].snr_db, true, dl)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    axes[0].bar([r["bin"] for r in angle_rows], [r["mae_deg"] for r in angle_rows], color="#2563eb")
    axes[0].set_title("DoANet MAE by true-angle bin")
    axes[0].set_ylabel("MAE [deg]")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].bar([r["bin"] for r in snr_rows], [r["mae_deg"] for r in snr_rows], color="#16a34a")
    axes[1].set_title("DoANet MAE by SNR bin")
    axes[1].set_ylabel("MAE [deg]")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(True, axis="y", alpha=0.25)
    bin_path = out_dir / "p03_doa_error_bins.png"
    fig.savefig(bin_path, dpi=180)
    plt.close(fig)

    table_rows = "\n".join(
        "<tr>"
        f"<td>{split}</td><td>{label}</td>"
        f"<td>{m['n']}</td><td>{fmt(m['mae_deg'])}</td><td>{fmt(m['rmse_deg'])}</td>"
        f"<td>{fmt(m['p90_deg'])}</td><td>{fmt(m['within_1deg_acc'])}</td>"
        f"<td>{fmt(m['within_2deg_acc'])}</td><td>{fmt(m['within_5deg_acc'])}</td>"
        "</tr>"
        for split in ("train", "val", "test")
        for label, m in summary[split].items()
    )
    angle_table = "\n".join(
        f"<tr><td>{r['bin']}</td><td>{r['n']}</td><td>{fmt(r['mae_deg'])}</td><td>{fmt(r['p90_deg'])}</td><td>{fmt(r['max_deg'])}</td></tr>"
        for r in angle_rows
    )
    snr_table = "\n".join(
        f"<tr><td>{r['bin']}</td><td>{r['n']}</td><td>{fmt(r['mae_deg'])}</td><td>{fmt(r['p90_deg'])}</td></tr>"
        for r in snr_rows
    )
    outlier_table = "\n".join(
        f"<tr><td>{r['idx']}</td><td>{fmt(r['true_deg'])}</td><td>{fmt(r['pred_deg'])}</td><td>{fmt(r['err_deg'])}</td>"
        f"<td>{fmt(r['snr_db'])}</td><td>{fmt(r['range_m'])}</td><td>{r['frame']}</td><td>{r['target_id']}</td></tr>"
        for r in outliers
    )

    cfg = {
        "dataset_dir": str(dataset_dir),
        "checkpoint": str(checkpoint),
        "device": device,
        "train_n": summary["train"]["RadarCubeDoANet"]["n"],
        "val_n": summary["val"]["RadarCubeDoANet"]["n"],
        "test_n": summary["test"]["RadarCubeDoANet"]["n"],
        "radar_bw_mhz": float(test["dataset"].radar_bw_hz / 1e6),
        "range_resolution_m": float(test["dataset"].radar_range_res_m),
        "wall_spacing_m": float(test["dataset"].wall_spacing_m),
    }
    (out_dir / "p03_doa_diagnostics.json").write_text(
        json.dumps({"config": cfg, "summary": summary, "angle_bins": angle_rows, "snr_bins": snr_rows, "outliers": outliers}, indent=2),
        encoding="utf-8",
    )

    dl_test = summary["test"]["RadarCubeDoANet"]
    music_test = summary["test"]["MUSIC"]
    fft_test = summary["test"]["Coarse angle FFT"]
    html = f"""<!doctype html>
<html lang="ko"><head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>P03 DoA-only Diagnostics</title>
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
<h1>P03 DoA-only Diagnostics</h1>
<p>Generated from <code>make_doa_diagnostics.py</code>.</p>
<div class="note"><strong>Result.</strong> For this dataset/checkpoint pair, RadarCubeDoANet reaches test MAE {fmt(dl_test['mae_deg'])}° and ≤2° accuracy {fmt(dl_test['within_2deg_acc'])}. MUSIC remains the best clean selected-vector reference at MAE {fmt(music_test['mae_deg'])}°, while the coarse native angle-FFT baseline is MAE {fmt(fft_test['mae_deg'])}°.</div>
<ul>
<li>Dataset: train {cfg['train_n']}, val {cfg['val_n']}, test {cfg['test_n']} detections.</li>
<li>Radar: {cfg['radar_bw_mhz']:g} MHz, ΔR={cfg['range_resolution_m']:.3f} m.</li>
<li>Scene wall spacing: {cfg['wall_spacing_m']:.3f} m. Keep this config separate from canonical final-result reports.</li>
<li>Training provenance is not inferred by this report; keep these diagnostics separate from canonical map-evaluation metrics.</li>
</ul>
<h2>Summary metrics</h2>
<table><thead><tr><th>Split</th><th>Method</th><th>N</th><th>MAE [deg]</th><th>RMSE [deg]</th><th>P90 [deg]</th><th>≤1°</th><th>≤2°</th><th>≤5°</th></tr></thead><tbody>{table_rows}</tbody></table>
<h2>Prediction scatter</h2>
<img alt="P03 DoA prediction scatter" src="data:image/png;base64,{encode_image(scatter_path)}" />
<h2>DoANet error bins</h2>
<img alt="P03 DoA error bins" src="data:image/png;base64,{encode_image(bin_path)}" />
<h3>Angle bins</h3>
<table><thead><tr><th>True angle bin [deg]</th><th>N</th><th>MAE [deg]</th><th>P90 [deg]</th><th>Max [deg]</th></tr></thead><tbody>{angle_table}</tbody></table>
<h3>SNR bins</h3>
<table><thead><tr><th>SNR bin [dB]</th><th>N</th><th>MAE [deg]</th><th>P90 [deg]</th></tr></thead><tbody>{snr_table}</tbody></table>
<h3>Largest DoANet test errors</h3>
<table><thead><tr><th>Idx</th><th>True</th><th>Pred</th><th>Error</th><th>SNR</th><th>Range</th><th>Frame</th><th>Target</th></tr></thead><tbody>{outlier_table}</tbody></table>
</main></body></html>"""
    html_path = out_dir / "p03_doa_diagnostics.html"
    html_path.write_text(html, encoding="utf-8")
    return {"html": str(html_path), "scatter": str(scatter_path), "bins": str(bin_path), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate P03 DoA-only diagnostics")
    parser.add_argument("--dataset_dir", default=str(BASE / "data_mapping"))
    parser.add_argument("--checkpoint", default=str(BASE / "artifacts" / "best_model.pt"))
    parser.add_argument("--out_dir", default=str(BASE / "artifacts" / "doa_diagnostics"))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    print(json.dumps(make_report(args), indent=2))


if __name__ == "__main__":
    main()
