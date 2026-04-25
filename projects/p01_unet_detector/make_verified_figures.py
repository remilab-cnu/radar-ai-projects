#!/usr/bin/env python3
"""Create compact P01 verified-experiment figures from JSON artifacts."""
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def figure_curves(cfar_val, unet_val, cfar_test, unet_test, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6.4, 4), constrained_layout=True)
    series = [
        (cfar_val, cfar_test, "CA-CFAR", "#2563eb", "o"),
        (unet_val, unet_test, "U-Net", "#dc2626", "s"),
    ]
    for val_payload, test_payload, label, color, marker in series:
        # Validation sweep selects an operating policy; held-out test point reports it.
        # Plotting both prevents test metrics from being mistaken for tuned points.
        payload = val_payload
        rows = payload.get("results", [])
        if not rows:
            continue
        pfa = [r["Pfa"] for r in rows]
        pd = [r["Pd"] for r in rows]
        ax.plot(pfa, pd, marker=marker, ms=3, lw=1.2, color=color, alpha=0.55, label=f"{label} validation sweep")
        val_sel = val_payload.get("selected_policy", {})
        if val_sel:
            ax.scatter([val_sel["Pfa"]], [val_sel["Pd"]], s=70, color="white", edgecolor=color, linewidth=1.8, zorder=5, label=f"{label} selected on val")
        test_sel = test_payload.get("selected_policy", {})
        if test_sel:
            ax.scatter([test_sel["Pfa"]], [test_sel["Pd"]], s=95, color=color, edgecolor="black", marker="*", zorder=6, label=f"{label} held-out test")
    ax.set_xscale("log")
    ax.set_xlabel("False-alarm probability (pixel-level Pfa)")
    ax.set_ylabel("Detection probability (Pd)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
    fig.savefig(out_dir / "p01_verified_pd_pfa_curve.png", dpi=170)
    plt.close(fig)


def figure_metric_table(cfar_test, unet_test, out_dir: Path):
    cfar = cfar_test["selected_policy"]
    unet = unet_test["selected_policy"]
    metrics = ["Pd", "Pfa", "Precision", "F1"]
    data = [[unet[m] for m in metrics], [cfar[m] for m in metrics]]
    fig, ax = plt.subplots(figsize=(7.2, 2.4), constrained_layout=True)
    ax.axis("off")
    cell_text = [[f"{v:.4g}" for v in row] for row in data]
    table = ax.table(cellText=cell_text, rowLabels=["U-Net", "CA-CFAR"], colLabels=metrics, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    fig.savefig(out_dir / "p01_verified_metric_table.png", dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts/verified_p01")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    art = Path(args.artifacts)
    out_dir = Path(args.out_dir) if args.out_dir else art / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfar_val = load_json(art / "p01_cfar_sweep_val.json")
    cfar_test = load_json(art / "p01_cfar_selected_test.json")
    unet_val = load_json(art / "p01_unet_threshold_sweep_val.json")
    unet_test = load_json(art / "p01_unet_selected_test.json")
    figure_curves(cfar_val, unet_val, cfar_test, unet_test, out_dir)
    figure_metric_table(cfar_test, unet_test, out_dir)
    summary = {
        "figures": ["p01_verified_pd_pfa_curve.png", "p01_verified_metric_table.png"],
        "figure_contract": "curves show validation sweeps; star markers and table show held-out test outcomes for the selected policies",
        "cfar_test": cfar_test["selected_policy"],
        "unet_test": unet_test["selected_policy"],
    }
    (out_dir / "p01_figure_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
