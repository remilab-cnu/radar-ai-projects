#!/usr/bin/env python3
"""Evaluate a P05 checkpoint by SNR bins."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model import make_waveform_model
from shared.waveform_library import WAVEFORM_CLASSES
from train import assert_current_schema


def load_split(path: Path) -> tuple[TensorDataset, np.ndarray]:
    assert_current_schema(path)
    with h5py.File(path, "r") as f:
        x = torch.as_tensor(f["x"][:], dtype=torch.float32)
        y = torch.as_tensor(f["y"][:], dtype=torch.long)
        snr = f["snr_db"][:].astype(np.float32)
    return TensorDataset(x, y), snr


@torch.no_grad()
def predict(model: torch.nn.Module, dataset: TensorDataset, device: str, max_samples: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    preds, labels = [], []
    n_seen = 0
    for x, y in loader:
        if max_samples is not None and n_seen >= max_samples:
            break
        logits = model(x.to(device))
        preds.append(logits.argmax(1).cpu())
        labels.append(y)
        n_seen += len(y)
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(labels).numpy()
    if max_samples is not None:
        y_pred = y_pred[:max_samples]
        y_true = y_true[:max_samples]
    return y_true, y_pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--checkpoint", default="artifacts/best_model.pt")
    ap.add_argument("--base_ch", type=int, default=16)
    ap.add_argument("--bins", type=float, nargs="*", default=[-6, 0, 6, 12, 18])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="artifacts/snr_sweep_results.json")
    args = ap.parse_args()

    dataset, snr = load_split(Path(args.data_dir) / f"waveform_{args.split}.h5")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_waveform_model(n_classes=len(WAVEFORM_CLASSES), base_ch=args.base_ch).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    max_samples = 64 if args.smoke else None
    y_true, y_pred = predict(model, dataset, device, max_samples=max_samples)
    snr = snr[:len(y_true)]

    bins = np.asarray(args.bins, dtype=np.float32)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (snr >= lo) & (snr < hi)
        if not np.any(mask):
            continue
        rows.append({
            "snr_lo_db": float(lo),
            "snr_hi_db": float(hi),
            "n": int(np.sum(mask)),
            "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])),
            "f1_macro": float(f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true[mask], y_pred[mask], labels=list(range(len(WAVEFORM_CLASSES)))).tolist(),
        })

    payload = {
        "kind": "p05_snr_sweep",
        "split": args.split,
        "class_names": list(WAVEFORM_CLASSES),
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "overall_f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "bins": rows,
        "n_eval": int(len(y_true)),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
