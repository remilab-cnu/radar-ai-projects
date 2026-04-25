#!/usr/bin/env python3
"""Validate P01 schema-v2 data and create label/axis sanity artifacts."""
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import matplotlib.pyplot as plt
import numpy as np

from eval_utils import assert_schema_v2, split_path


def analyze(path: Path, out_dir: Path, max_examples: int = 4) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "r") as f:
        assert_schema_v2(f)
        n = len(f["x"])
        y = f["y"][:, 0]
        pos = y.reshape(n, -1).sum(axis=1)
        ratio = pos / y.reshape(n, -1).shape[1]
        nt = f["n_targets"][:]
        rb = f["target_range_bin"][:]
        vb = f["target_doppler_bin"][:]
        _, _, nd, nr = f["x"].shape
        invalid_bins = int(np.sum(
            (rb.astype(int) < -1) | (rb.astype(int) >= nr) |
            (vb.astype(int) < -1) | (vb.astype(int) >= nd)
        ))
        empty_with_targets = int(np.sum((nt > 0) & (pos == 0)))

        fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
        ax.hist(ratio, bins=30, color="#2563eb", alpha=0.85)
        ax.set_xlabel("positive-pixel ratio")
        ax.set_ylabel("samples")
        ax.set_title("P01 label sparsity")
        fig.savefig(out_dir / "p01_positive_ratio_hist.png", dpi=160)
        plt.close(fig)

        n_show = min(max_examples, n)
        fig, axes = plt.subplots(n_show, 2, figsize=(8, 2.6 * n_show), constrained_layout=True)
        if n_show == 1:
            axes = np.array([axes])
        for row in range(n_show):
            mag = f["rdm_mag_linear"][row]
            mask = y[row]
            db = 20 * np.log10(mag / (np.median(mag) + 1e-12) + 1e-12)
            axes[row, 0].imshow(db, aspect="auto", origin="lower", cmap="viridis")
            axes[row, 0].contour(mask, levels=[0.5], colors="r", linewidths=0.8)
            axes[row, 0].set_title(f"RDM + mask #{row}")
            axes[row, 1].imshow(mask, aspect="auto", origin="lower", cmap="gray_r")
            axes[row, 1].set_title(f"mask positives={int(pos[row])}")
        fig.savefig(out_dir / "p01_label_overlay_examples.png", dpi=160)
        plt.close(fig)

    return {
        "split_path": str(path),
        "n_samples": int(n),
        "positive_ratio_mean": float(np.mean(ratio)),
        "positive_ratio_min": float(np.min(ratio)),
        "positive_ratio_max": float(np.max(ratio)),
        "empty_masks_with_targets": empty_with_targets,
        "invalid_bin_markers": invalid_bins,
        "artifacts": ["p01_positive_ratio_hist.png", "p01_label_overlay_examples.png"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--out_dir", default="artifacts/verified_p01")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    result = analyze(split_path(args.data_dir, args.split), out_dir)
    out = out_dir / "p01_label_sanity.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
