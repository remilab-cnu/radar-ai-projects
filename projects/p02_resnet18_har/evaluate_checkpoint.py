#!/usr/bin/env python3
"""Evaluate a trained P02 neural checkpoint on a chosen HAR split.

This is intentionally small and reuses the training contract so stress-set
results are comparable to `train.py` test evaluations.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import make_har_model
from train import ACTIVITY_LABELS, N_CLASSES, evaluate, load_har_dataset


def _load_state_dict(path: Path, device: str) -> dict:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate P02 TinyCNN/ResNet18 checkpoint")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", choices=["tiny_cnn", "resnet18"], required=True)
    ap.add_argument("--max_samples", type=int, default=3000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    checkpoint = Path(args.checkpoint)
    split_path = data_dir / f"har_{args.split}.h5"
    if not split_path.exists():
        raise FileNotFoundError(f"missing evaluation split: {split_path}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = load_har_dataset(split_path)
    model = make_har_model(args.model, n_classes=N_CLASSES).to(device)
    state = _load_state_dict(checkpoint, device)
    model.load_state_dict(state)

    results = evaluate(model, dataset, device, max_samples=args.max_samples)
    results.update({
        "kind": "p02_checkpoint_eval",
        "model": args.model,
        "class_names": list(ACTIVITY_LABELS),
        "n_classes": int(N_CLASSES),
        "data_dir": str(data_dir),
        "split": args.split,
        "checkpoint": str(checkpoint),
        "max_samples": int(args.max_samples),
        "eval_samples": int(min(len(dataset), args.max_samples)),
        "device": device,
    })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
