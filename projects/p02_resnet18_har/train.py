#!/usr/bin/env python3
"""P02 — Train ResNet-18 Micro-Doppler HAR.

Usage:
    # Generate data + train
    python train.py --generate --epochs 30

    # Train only (existing data)
    python train.py --epochs 30

    # Smoke test (CPU, tiny data, 2 epochs)
    python train.py --generate --smoke

    # Eval only
    python train.py --eval_only --checkpoint artifacts/best_model.pt
"""

import json
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
from common.seed import seed_everything
from common.train_utils import training_loop, count_parameters
from model import make_har_model

try:
    from shared.micro_doppler import ACTIVITY_LABELS, N_CLASSES
except ImportError:
    ACTIVITY_LABELS = [f"class_{i}" for i in range(6)]
    N_CLASSES = 6

EXPECTED_SCHEMA_VERSION = 6
DEFAULT_ASPECT_ANGLE_RANGE_DEG = (0.0, 60.0)


def assert_current_schema(path: Path) -> None:
    with h5py.File(path, "r") as f:
        version = int(f["schema_version"][0]) if "schema_version" in f else -1
        required = [
            "aspect_angle_deg", "aspect_angle_range_deg", "slow_time_prf_hz",
            "slow_time_samples", "scatter_model", "range_processing", "doppler_source",
            "max_abs_radial_velocity_mps", "radar_max_unambiguous_velocity_mps",
            "doppler_alias_margin_mps", "aspect_convention",
        ]
        missing = [key for key in required if key not in f]
    if version != EXPECTED_SCHEMA_VERSION or missing:
        raise ValueError(
            f"{path} is stale or incompatible (schema_version={version}, "
            f"missing={missing}); regenerate P02 data before training/eval."
        )


# ---------------------------------------------------------------------------
# Dataset helper: load spectrogram + integer labels from HDF5
# ---------------------------------------------------------------------------

def load_har_dataset(path):
    """Return TensorDataset with (spec, label) tensors."""
    assert_current_schema(Path(path))
    data = load_hdf5(path, ["x", "y"])
    x = torch.as_tensor(data["x"], dtype=torch.float32)   # (N, 1, H, W)
    y = torch.as_tensor(data["y"], dtype=torch.long)       # (N,)
    return TensorDataset(x, y)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataset, device, max_samples=3000):
    """Accuracy + per-class accuracy."""
    model.eval()
    n_eval = min(len(dataset), max_samples)
    preds, labels = [], []

    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    total = 0
    for x, y in loader:
        if total >= n_eval:
            break
        out = model(x.to(device))
        preds.append(out.argmax(1).cpu())
        labels.append(y)
        total += len(y)

    preds  = torch.cat(preds).numpy()[:n_eval]
    labels = torch.cat(labels).numpy()[:n_eval]

    acc = float(np.mean(preds == labels))
    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for true, pred in zip(labels, preds):
        if 0 <= true < N_CLASSES and 0 <= pred < N_CLASSES:
            confusion[int(true), int(pred)] += 1
    per_class = {}
    for i, name in enumerate(ACTIVITY_LABELS):
        mask = labels == i
        if mask.sum() > 0:
            per_class[name] = float(np.mean(preds[mask] == labels[mask]))

    return {
        "accuracy": acc,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "class_names": list(ACTIVITY_LABELS),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = base_parser("P02: ResNet-18 Micro-Doppler HAR")
    p.add_argument("--n_train", type=int, default=30000)
    p.add_argument("--n_val",   type=int, default=3000)
    p.add_argument("--n_test",  type=int, default=3000)
    p.add_argument("--snr_lo",  type=float, default=5.0)
    p.add_argument("--snr_hi",  type=float, default=25.0)
    p.add_argument("--aspect_lo", type=float, default=DEFAULT_ASPECT_ANGLE_RANGE_DEG[0])
    p.add_argument("--aspect_hi", type=float, default=DEFAULT_ASPECT_ANGLE_RANGE_DEG[1])
    p.add_argument("--range_lo", type=float, default=6.0)
    p.add_argument("--range_hi", type=float, default=18.0)
    p.add_argument("--model", choices=["resnet18", "tiny_cnn"], default="resnet18")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--artifact_dir", type=str, default=None)
    args = p.parse_args()

    if args.smoke:
        args.n_train = 256
        args.n_val   = 64
        args.n_test  = 64
        args.epochs  = 2
        args.batch_size = 16

    seed_everything(args.seed)

    root = Path(__file__).parent
    data_dir = Path(args.data_dir) if args.data_dir else root / "data"
    ckpt_dir = Path(args.artifact_dir) if args.artifact_dir else root / "artifacts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Data generation ---
    if args.generate:
        import subprocess
        cmd = [
            sys.executable, str(root / "generate_data.py"),
            "--n_train", str(args.n_train),
            "--n_val",   str(args.n_val),
            "--n_test",  str(args.n_test),
            "--snr_lo",  str(args.snr_lo),
            "--snr_hi",  str(args.snr_hi),
            "--aspect_lo", str(args.aspect_lo),
            "--aspect_hi", str(args.aspect_hi),
            "--range_lo", str(args.range_lo),
            "--range_hi", str(args.range_hi),
            "--out_dir", str(data_dir),
            "--seed",    str(args.seed),
        ]
        if args.smoke:
            cmd.append("--smoke")
        subprocess.run(cmd, check=True)

    # --- Datasets ---
    train_path = data_dir / "har_train.h5"
    val_path   = data_dir / "har_val.h5"
    test_path  = data_dir / "har_test.h5"

    for p_check in [train_path, val_path, test_path]:
        if not p_check.exists():
            print(f"ERROR: {p_check} not found. Use --generate flag.")
            return

    train_ds = load_har_dataset(train_path)
    val_ds   = load_har_dataset(val_path)
    test_ds  = load_har_dataset(test_path)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"  Classes ({N_CLASSES}): {ACTIVITY_LABELS}")

    # --- Model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_har_model(args.model, n_classes=N_CLASSES).to(device)
    print(f"  Device: {device}")
    print(f"  Model: {args.model}")
    print(f"  Parameters: {count_parameters(model):,}")

    # --- Load checkpoint ---
    if args.checkpoint and Path(args.checkpoint).exists():
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"  Loaded: {args.checkpoint}")

    # --- Eval only ---
    if args.eval_only:
        results = evaluate(model, test_ds, device)
        results["model"] = args.model
        print(f"\n=== Test Evaluation ===")
        print(f"  Overall accuracy: {results['accuracy']:.1%}")
        for cls, acc in results["per_class"].items():
            print(f"    {cls:<14s}  {acc:.1%}")
        with open(ckpt_dir / "eval_results.json", "w") as f:
            json.dump(results, f, indent=2)
        return

    # --- Training ---
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    print("\n=== P02: ResNet-18 HAR ===")
    training_loop(
        model, train_loader, val_loader, criterion, optimizer,
        epochs=args.epochs, checkpoint_dir=ckpt_dir,
        device=device, scheduler=scheduler,
    )

    # --- Final evaluation ---
    print("\n=== Test Set Evaluation ===")
    best_ckpt = ckpt_dir / "best_model.pt"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
    results = evaluate(model, test_ds, device)
    results["model"] = args.model
    print(f"  Overall accuracy: {results['accuracy']:.1%}")
    for cls, acc in results["per_class"].items():
        print(f"    {cls:<14s}  {acc:.1%}")
    with open(ckpt_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {ckpt_dir}/eval_results.json")


if __name__ == "__main__":
    main()
