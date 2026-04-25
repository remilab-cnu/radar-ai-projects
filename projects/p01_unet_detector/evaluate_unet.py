#!/usr/bin/env python3
"""Run P01 U-Net threshold sweeps from validation-selected policies."""
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
    choose_by_max_f1,
    confusion_counts,
    load_policy,
    metrics_from_counts,
    split_path,
)
from model import UNetDetector


def predict_probs(model, x, device):
    with torch.no_grad():
        batch = torch.as_tensor(x, dtype=torch.float32, device=device)
        return model(batch).cpu().numpy()[:, 0]


def evaluate_thresholds(
    path: Path,
    checkpoint: Path,
    thresholds: list[float],
    max_samples: int | None,
    base_ch: int,
):
    device = "cpu"
    model = UNetDetector(in_channels=2, base_ch=base_ch).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    counts = [{"tp": 0, "fp": 0, "fn": 0, "tn": 0} for _ in thresholds]
    n_done = 0
    with h5py.File(path, "r") as f:
        assert_schema_v2(f)
        n = len(f["x"])
        if max_samples is not None:
            n = min(n, max_samples)
        batch_size = 16
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            probs = predict_probs(model, f["x"][start:end], device)
            gt = f["y"][start:end, 0] > 0.5
            for ti, thr in enumerate(thresholds):
                add_counts(counts[ti], confusion_counts(probs > thr, gt))
            n_done += end - start
    results = []
    for thr, c in zip(thresholds, counts):
        row = metrics_from_counts(c)
        row.update({
            "threshold": float(thr),
            "n_eval": int(n_done),
            "false_alarms_per_rdm": float(c["fp"] / max(n_done, 1)),
        })
        results.append(row)
    selected = choose_by_max_f1(results)
    return selected, results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--policy_from", "--policy-from", dest="policy_from")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--base_ch", type=int, default=32)
    args = ap.parse_args()

    path = split_path(args.data_dir, args.split)
    if args.policy_from:
        pol = load_policy(args.policy_from)
        thresholds = [float(pol["threshold"])] if "threshold" in pol else [float(args.threshold)]
    elif args.sweep:
        thresholds = [round(x, 3) for x in np.linspace(0.05, 0.95, 19)]
    else:
        thresholds = [args.threshold]
    selected, results = evaluate_thresholds(
        path,
        Path(args.checkpoint),
        thresholds,
        args.max_samples,
        args.base_ch,
    )
    payload = {
        "kind": "p01_unet",
        "split": args.split,
        "data_path": str(path),
        "checkpoint": str(args.checkpoint),
        "selected_policy": selected,
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(selected, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
