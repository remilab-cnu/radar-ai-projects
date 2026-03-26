#!/usr/bin/env python3
"""P01 — Train U-Net Radar Detector.

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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.cli import base_parser
from common.hdf5_io import HDF5Dataset
from common.seed import seed_everything
from common.train_utils import training_loop, count_parameters
from model import UNetDetector, FocalDiceLoss


# ---------------------------------------------------------------------------
# Dataset wrapper: HDF5 stores y as (1, Nc, Nr); return (rdm, mask) tensors
# ---------------------------------------------------------------------------

class DetectionDataset(HDF5Dataset):
    """Thin wrapper: x=(2,Nc,Nr) float32, y=(1,Nc,Nr) float32."""

    def __init__(self, path):
        super().__init__(path, x_key="x", y_key="y",
                         x_dtype=torch.float32, y_dtype=torch.float32)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataset, device, threshold=0.5, max_samples=2000):
    """Pixel-level Pd / Pfa / Precision / F1."""
    model.eval()
    n_eval = min(len(dataset), max_samples)
    tp = fp = fn = tn = 0

    for i in range(n_eval):
        rdm, mask = dataset[i]
        gt = (mask[0].numpy() > 0.5)
        pred_prob = model(rdm.unsqueeze(0).to(device))[0, 0].cpu().numpy()
        pred_bin = pred_prob > threshold

        tp += int(np.sum(pred_bin & gt))
        fp += int(np.sum(pred_bin & ~gt))
        fn += int(np.sum(~pred_bin & gt))
        tn += int(np.sum(~pred_bin & ~gt))

    pd = tp / (tp + fn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    f1 = 2 * precision * pd / (precision + pd + 1e-10)
    total_neg = fp + tn
    pfa = fp / (total_neg + 1e-10)
    return {"Pd": float(pd), "Pfa": float(pfa), "Precision": float(precision), "F1": float(f1)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = base_parser("P01: U-Net Radar Detector")
    p.add_argument("--n_train",  type=int, default=50000)
    p.add_argument("--n_val",    type=int, default=5000)
    p.add_argument("--n_test",   type=int, default=5000)
    p.add_argument("--base_ch",  type=int, default=32)
    p.add_argument("--data_dir", type=str, default=None)
    args = p.parse_args()

    if args.smoke:
        args.n_train = 256
        args.n_val   = 64
        args.n_test  = 64
        args.epochs  = 2
        args.batch_size = 8

    seed_everything(args.seed)

    root = Path(__file__).parent
    data_dir = Path(args.data_dir) if args.data_dir else root / "data"
    ckpt_dir = root / "artifacts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Data generation ---
    if args.generate:
        import subprocess
        cmd = [
            sys.executable, str(root / "generate_data.py"),
            "--n_train", str(args.n_train),
            "--n_val",   str(args.n_val),
            "--n_test",  str(args.n_test),
            "--out_dir", str(data_dir),
            "--seed",    str(args.seed),
        ]
        if args.smoke:
            cmd.append("--smoke")
        subprocess.run(cmd, check=True)

    # --- Datasets ---
    train_path = data_dir / "det_train.h5"
    val_path   = data_dir / "det_val.h5"
    test_path  = data_dir / "det_test.h5"

    for p_check in [train_path, val_path, test_path]:
        if not p_check.exists():
            print(f"ERROR: {p_check} not found. Use --generate flag.")
            return

    train_ds = DetectionDataset(train_path)
    val_ds   = DetectionDataset(val_path)
    test_ds  = DetectionDataset(test_path)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # --- Model ---
    device = "cpu"
    model = UNetDetector(in_channels=2, base_ch=args.base_ch).to(device)
    print(f"  Model: UNetDetector (base_ch={args.base_ch})")
    print(f"  Parameters: {count_parameters(model):,}")

    # --- Load checkpoint ---
    if args.checkpoint and Path(args.checkpoint).exists():
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"  Loaded: {args.checkpoint}")

    # --- Eval only ---
    if args.eval_only:
        results = evaluate(model, test_ds, device)
        print("\n=== Test Evaluation ===")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")
        with open(ckpt_dir / "eval_results.json", "w") as f:
            json.dump(results, f, indent=2)
        return

    # --- Training ---
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    criterion = FocalDiceLoss(alpha=0.75, gamma=2.0, dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    print("\n=== P01: U-Net Radar Detector ===")
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
    print(f"  Pd={results['Pd']:.4f}  Pfa={results['Pfa']:.2e}  "
          f"Prec={results['Precision']:.4f}  F1={results['F1']:.4f}")
    with open(ckpt_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {ckpt_dir}/eval_results.json")


if __name__ == "__main__":
    main()
