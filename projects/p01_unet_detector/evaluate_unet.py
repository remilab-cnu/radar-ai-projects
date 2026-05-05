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
    add_target_counts,
    add_target_metrics,
    assert_schema_current,
    choose_by_max_f1,
    confusion_counts,
    load_policy,
    metrics_from_counts,
    split_path,
    target_detection_counts,
)
from model import UNetDetector


def apply_input_mode(x: np.ndarray, input_mode: str) -> np.ndarray:
    if input_mode == "mag_phase":
        return x
    if input_mode == "mag_only":
        x = np.array(x, copy=True)
        x[:, 1] = 0.0
        return x
    raise ValueError(f"unknown input_mode={input_mode!r}")


def predict_probs(model, x, device, input_mode: str):
    with torch.no_grad():
        x = apply_input_mode(x, input_mode)
        batch = torch.as_tensor(x, dtype=torch.float32, device=device)
        return model(batch).cpu().numpy()[:, 0]


def evaluate_thresholds(
    path: Path,
    checkpoint: Path,
    thresholds: list[float],
    max_samples: int | None,
    base_ch: int,
    input_mode: str,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetDetector(in_channels=2, base_ch=base_ch).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    counts = [{"tp": 0, "fp": 0, "fn": 0, "tn": 0} for _ in thresholds]
    target_counts = [{"target_detected": 0, "target_total": 0} for _ in thresholds]
    n_done = 0
    with h5py.File(path, "r") as f:
        assert_schema_current(f)
        n = len(f["x"])
        if max_samples is not None:
            n = min(n, max_samples)
        batch_size = 16
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            probs = predict_probs(model, f["x"][start:end], device, input_mode)
            gt = f["y"][start:end, 0] > 0.5
            for ti, thr in enumerate(thresholds):
                pred = probs > thr
                add_counts(counts[ti], confusion_counts(pred, gt))
                for bi in range(end - start):
                    add_target_counts(
                        target_counts[ti],
                        target_detection_counts(
                            pred[bi],
                            f["target_range_bin"][start + bi],
                            f["target_doppler_bin"][start + bi],
                        ),
                    )
            n_done += end - start
    results = []
    for thr, c, tc in zip(thresholds, counts, target_counts):
        row = metrics_from_counts(c)
        add_target_metrics(row, tc)
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
    ap.add_argument("--input_mode", choices=["mag_phase", "mag_only"], default="mag_phase")
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
        args.input_mode,
    )
    payload = {
        "kind": "p01_unet",
        "split": args.split,
        "data_path": str(path),
        "checkpoint": str(args.checkpoint),
        "input_mode": args.input_mode,
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
