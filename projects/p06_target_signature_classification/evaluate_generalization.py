#!/usr/bin/env python3
"""Evaluate a P06 checkpoint on a standard or generated generalization split."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model import make_signature_model
from shared.target_signature import TARGET_CLASSES
from train import assert_current_schema


def load_split(path: Path) -> tuple[TensorDataset, np.ndarray, np.ndarray]:
    assert_current_schema(path)
    with h5py.File(path, "r") as f:
        x = torch.as_tensor(f["x"][:], dtype=torch.float32)
        y = torch.as_tensor(f["y"][:], dtype=torch.long)
        snr = f["snr_db"][:].astype(np.float32)
        aspect = f["center_aspect_deg"][:].astype(np.float32)
    return TensorDataset(x, y), snr, aspect


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


def _group_metric(name: str, values: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, bins: list[float]) -> list[dict]:
    rows = []
    edges = np.asarray(bins, dtype=np.float32)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (values >= lo) & (values < hi)
        if not np.any(mask):
            continue
        rows.append({
            f"{name}_lo": float(lo),
            f"{name}_hi": float(hi),
            "n": int(np.sum(mask)),
            "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])),
            "f1_macro": float(f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_generalization")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--checkpoint", default="artifacts/best_model.pt")
    ap.add_argument("--base_ch", type=int, default=16)
    ap.add_argument("--generate", action="store_true", help="Generate a held-out aspect/SNR data directory before evaluation")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n_eval", type=int, default=300)
    ap.add_argument("--snr_lo", type=float, default=0.0)
    ap.add_argument("--snr_hi", type=float, default=8.0)
    ap.add_argument("--aspect_lo", type=float, default=60.0)
    ap.add_argument("--aspect_hi", type=float, default=90.0)
    ap.add_argument("--out", default="artifacts/generalization_results.json")
    args = ap.parse_args()

    root = Path(__file__).parent
    data_dir = Path(args.data_dir)
    if args.generate:
        n = 90 if args.smoke else args.n_eval
        cmd = [
            sys.executable, str(root / "generate_data.py"),
            "--n_train", str(n),
            "--n_val", str(n),
            "--n_test", str(n),
            "--snr_lo", str(args.snr_lo),
            "--snr_hi", str(args.snr_hi),
            "--aspect_lo", str(args.aspect_lo),
            "--aspect_hi", str(args.aspect_hi),
            "--out_dir", str(data_dir),
            "--seed", "9090",
        ]
        subprocess.run(cmd, check=True)

    dataset, snr, aspect = load_split(data_dir / f"signature_{args.split}.h5")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_signature_model(n_classes=len(TARGET_CLASSES), base_ch=args.base_ch).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    max_samples = 90 if args.smoke else None
    y_true, y_pred = predict(model, dataset, device, max_samples=max_samples)
    snr = snr[:len(y_true)]
    aspect = aspect[:len(y_true)]

    payload = {
        "kind": "p06_generalization_eval",
        "split": args.split,
        "class_names": list(TARGET_CLASSES),
        "n_eval": int(len(y_true)),
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "overall_f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(TARGET_CLASSES)))).tolist(),
        "snr_bins": _group_metric("snr_db", snr, y_true, y_pred, [0, 4, 8, 12, 18, 24]),
        "aspect_abs_bins": _group_metric("abs_aspect_deg", np.abs(aspect), y_true, y_pred, [0, 30, 60, 90]),
        "data_dir": str(data_dir),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
