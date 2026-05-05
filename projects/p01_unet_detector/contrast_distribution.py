#!/usr/bin/env python3
"""Compute per-target local contrast (sigma units) over a split.

For each labelled target in the test split, measure
    contrast_sigma = (peak - local_median) / local_mad
on the linear-magnitude RDM, where the local window excludes guard cells
around the target. Saves a single JSON for the figure script.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np

from eval_utils import assert_schema_v2, split_path


def local_contrast(rdm: np.ndarray, vi: int, ri: int,
                   guard: int = 2, train: int = 8) -> float:
    Nc, Nr = rdm.shape
    v_lo, v_hi = max(0, vi - train - guard), min(Nc, vi + train + guard + 1)
    r_lo, r_hi = max(0, ri - train - guard), min(Nr, ri + train + guard + 1)
    block = rdm[v_lo:v_hi, r_lo:r_hi].copy()
    # Mask the guard region around the target
    gv_lo = max(0, vi - guard) - v_lo
    gv_hi = min(Nc, vi + guard + 1) - v_lo
    gr_lo = max(0, ri - guard) - r_lo
    gr_hi = min(Nr, ri + guard + 1) - r_lo
    mask = np.ones_like(block, dtype=bool)
    mask[gv_lo:gv_hi, gr_lo:gr_hi] = False
    bg = block[mask]
    med = float(np.median(bg))
    mad = float(np.median(np.abs(bg - med))) + 1e-30
    sigma = mad * 1.4826
    peak = float(rdm[vi, ri])
    return (peak - med) / sigma


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    contrasts = []
    rcs_list = []
    snr_list = []
    path = split_path(args.data_dir, args.split)
    with h5py.File(path, "r") as f:
        assert_schema_v2(f)
        n = len(f["x"])
        if args.max_samples is not None:
            n = min(n, int(args.max_samples))
        for i in range(n):
            rdm = f["rdm_mag_linear"][i]
            snr = float(f["snr_db"][i])
            rb = f["target_range_bin"][i]
            db = f["target_doppler_bin"][i]
            rcs = f["target_rcs"][i]
            for j in range(len(rb)):
                if rb[j] < 0 or db[j] < 0:
                    continue
                c = local_contrast(rdm, int(db[j]), int(rb[j]))
                contrasts.append(float(c))
                rcs_list.append(float(rcs[j]) if not np.isnan(rcs[j]) else float("nan"))
                snr_list.append(snr)
    payload = {
        "kind": "p01_contrast_distribution",
        "split": args.split,
        "n_targets": len(contrasts),
        "contrasts": contrasts,
        "rcs": rcs_list,
        "snr_db": snr_list,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    print(f"saved {out}  ({len(contrasts)} targets)")


if __name__ == "__main__":
    main()
