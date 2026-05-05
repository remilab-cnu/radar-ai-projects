#!/usr/bin/env python3
"""SNR-binned breakdown of CFAR vs U-Net detection performance.

Reads val-selected policies, applies them to test split, and reports
per-SNR-bin Pd / Pfa / F1 for both detectors. Produces a single JSON
that the figure script can plot.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np
import torch

from eval_utils import (
    add_counts,
    assert_schema_v2,
    confusion_counts,
    load_policy,
    metrics_from_counts,
    split_path,
)
from model import UNetDetector
from shared.fmcw_simulator import ca_cfar_2d


def _bin_index(snr: float, edges: np.ndarray) -> int:
    return int(np.clip(np.searchsorted(edges, snr, side="right") - 1, 0, len(edges) - 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--cfar_policy", required=True)
    ap.add_argument("--unet_policy", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base_ch", type=int, default=32)
    ap.add_argument("--bin_edges", default="5,10,15,20,25",
                    help="comma-separated SNR bin edges in dB")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    edges = np.array([float(x) for x in args.bin_edges.split(",")], dtype=float)
    n_bins = len(edges) - 1
    bin_labels = [f"{edges[i]:.0f}-{edges[i+1]:.0f}" for i in range(n_bins)]

    cfar_pol = load_policy(args.cfar_policy)
    unet_pol = load_policy(args.unet_policy)
    guard = tuple(cfar_pol["guard"])
    train = tuple(cfar_pol["train"])
    pfa = float(cfar_pol.get("pfa_design", cfar_pol.get("pfa", 1e-4)))
    threshold = float(unet_pol["threshold"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetDetector(in_channels=2, base_ch=args.base_ch).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    cfar_counts = [{"tp": 0, "fp": 0, "fn": 0, "tn": 0} for _ in range(n_bins)]
    unet_counts = [{"tp": 0, "fp": 0, "fn": 0, "tn": 0} for _ in range(n_bins)]
    n_per_bin = [0] * n_bins

    path = split_path(args.data_dir, args.split)
    batch_size = 16
    with h5py.File(path, "r") as f:
        assert_schema_v2(f)
        n = len(f["x"])
        if args.max_samples is not None:
            n = min(n, int(args.max_samples))
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            with torch.no_grad():
                bx = torch.as_tensor(f["x"][start:end], dtype=torch.float32, device=device)
                probs = model(bx).cpu().numpy()[:, 0]
            for k in range(end - start):
                i = start + k
                snr = float(f["snr_db"][i])
                b = _bin_index(snr, edges)
                gt = f["y"][i, 0] > 0.5
                n_per_bin[b] += 1
                cfar_det = ca_cfar_2d(f["rdm_mag_linear"][i], guard=guard, train=train, pfa=pfa)
                add_counts(cfar_counts[b], confusion_counts(cfar_det, gt))
                unet_det = probs[k] > threshold
                add_counts(unet_counts[b], confusion_counts(unet_det, gt))
            print(f"  {end}/{n}", flush=True)

    rows = []
    for i, label in enumerate(bin_labels):
        cfar_m = metrics_from_counts(cfar_counts[i])
        unet_m = metrics_from_counts(unet_counts[i])
        rows.append({
            "snr_bin": label,
            "snr_lo": float(edges[i]),
            "snr_hi": float(edges[i + 1]),
            "n_samples": int(n_per_bin[i]),
            "cfar": cfar_m,
            "unet": unet_m,
        })

    payload = {
        "kind": "p01_snr_breakdown",
        "split": args.split,
        "data_path": str(path),
        "cfar_policy": cfar_pol,
        "unet_policy": unet_pol,
        "bin_edges_db": edges.tolist(),
        "bins": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
