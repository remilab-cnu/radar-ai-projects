#!/usr/bin/env python3
"""P06 — Train a lightweight target signature classifier."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.cli import base_parser
from common.hdf5_io import load_hdf5
from common.metrics import classification_report
from common.seed import seed_everything
from common.train_utils import count_parameters, training_loop
from model import make_signature_model
from shared.target_signature import SCHEMA_VERSION, TARGET_CLASSES


def assert_current_schema(path: Path) -> None:
    with h5py.File(path, "r") as f:
        version = int(f["schema_version"][0]) if "schema_version" in f else -1
        required = ["target_class_names", "snr_db", "center_aspect_deg", "features", "matlab_reference_url"]
        missing = [key for key in required if key not in f]
    if version != SCHEMA_VERSION or missing:
        raise ValueError(
            f"{path} is stale or incompatible (schema_version={version}, missing={missing}); "
            "regenerate P06 data before training/eval."
        )


def load_signature_dataset(path: Path) -> TensorDataset:
    assert_current_schema(path)
    data = load_hdf5(path, ["x", "y"])
    x = torch.as_tensor(data["x"], dtype=torch.float32)
    y = torch.as_tensor(data["y"], dtype=torch.long)
    return TensorDataset(x, y)


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataset: TensorDataset, device: str, max_samples: int = 3000) -> dict:
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    preds, labels = [], []
    total = 0
    for x, y in loader:
        if total >= max_samples:
            break
        logits = model(x.to(device))
        preds.append(logits.argmax(1).cpu())
        labels.append(y)
        total += len(y)
    y_true = torch.cat(labels).numpy()[:max_samples]
    y_pred = torch.cat(preds).numpy()[:max_samples]
    report = classification_report(y_true, y_pred, labels=list(range(len(TARGET_CLASSES))))
    report.update({
        "class_names": list(TARGET_CLASSES),
        "n_eval": int(len(y_true)),
    })
    return report


def main() -> None:
    p = base_parser("P06: Lightweight Target Signature Classification")
    p.add_argument("--n_train", type=int, default=3000)
    p.add_argument("--n_val", type=int, default=600)
    p.add_argument("--n_test", type=int, default=600)
    p.add_argument("--snr_lo", type=float, default=6.0)
    p.add_argument("--snr_hi", type=float, default=24.0)
    p.add_argument("--aspect_lo", type=float, default=-45.0)
    p.add_argument("--aspect_hi", type=float, default=45.0)
    p.add_argument("--base_ch", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--artifact_dir", type=str, default=None)
    args = p.parse_args()

    if args.smoke:
        args.n_train, args.n_val, args.n_test = 240, 60, 60
        args.epochs = 2
        args.batch_size = 16
        args.base_ch = min(args.base_ch, 8)

    seed_everything(args.seed)
    root = Path(__file__).parent
    data_dir = Path(args.data_dir) if args.data_dir else root / "data"
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if args.generate:
        cmd = [
            sys.executable, str(root / "generate_data.py"),
            "--n_train", str(args.n_train),
            "--n_val", str(args.n_val),
            "--n_test", str(args.n_test),
            "--snr_lo", str(args.snr_lo),
            "--snr_hi", str(args.snr_hi),
            "--aspect_lo", str(args.aspect_lo),
            "--aspect_hi", str(args.aspect_hi),
            "--out_dir", str(data_dir),
            "--seed", str(args.seed),
        ]
        if args.smoke:
            cmd.append("--smoke")
        subprocess.run(cmd, check=True)

    train_path = data_dir / "signature_train.h5"
    val_path = data_dir / "signature_val.h5"
    test_path = data_dir / "signature_test.h5"
    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Use --generate first.")

    train_ds = load_signature_dataset(train_path)
    val_ds = load_signature_dataset(val_path)
    test_ds = load_signature_dataset(test_path)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"  Classes ({len(TARGET_CLASSES)}): {list(TARGET_CLASSES)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_signature_model(
        n_classes=len(TARGET_CLASSES),
        base_ch=args.base_ch,
        dropout=args.dropout,
    ).to(device)
    print(f"  Device: {device}")
    print(f"  Parameters: {count_parameters(model):,}")

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"  Loaded: {args.checkpoint}")

    if args.eval_only:
        results = evaluate(model, test_ds, device)
        results.update({"model": "tiny_signature_cnn", "base_ch": args.base_ch})
        out = artifact_dir / "eval_results.json"
        out.write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))
        print(f"saved {out}")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    history = training_loop(model, train_loader, val_loader, criterion, optimizer, args.epochs, artifact_dir, device, scheduler)

    model.load_state_dict(torch.load(artifact_dir / "best_model.pt", map_location=device))
    results = evaluate(model, test_ds, device)
    results.update({
        "model": "tiny_signature_cnn",
        "base_ch": args.base_ch,
        "epochs": int(args.epochs),
        "history_final_val_loss": float(history["val_loss"][-1]) if history["val_loss"] else None,
    })
    out = artifact_dir / "eval_results.json"
    out.write_text(json.dumps(results, indent=2))
    print("\n=== Test Evaluation ===")
    print(json.dumps(results, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
