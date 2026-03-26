#!/usr/bin/env python3
"""P04 -- Train DnCNN-SAR for SAR despeckling.

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
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.cli import base_parser
from common.seed import seed_everything
from common.train_utils import training_loop, count_parameters
from model import DnCNNSAR, DespecklingLoss, lee_filter, frost_filter, median_filter
from model import compute_psnr, compute_ssim, compute_enl


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DespecklingDataset(Dataset):
    """HDF5 SAR despeckling dataset: returns (noisy, clean) tensors."""

    def __init__(self, path: Path):
        self.path = path
        with h5py.File(path, 'r') as f:
            self.n_samples  = f['noisy'].shape[0]
            self.image_size = f['noisy'].shape[2]
        self._file = None

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self.path, 'r')

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        self._open()
        noisy = self._file['noisy'][idx]  # (1, H, W) float32
        clean = self._file['clean'][idx]  # (1, H, W) float32
        return torch.from_numpy(noisy), torch.from_numpy(clean)

    def get_meta(self, idx):
        self._open()
        return {
            'n_looks':   int(self._file['n_looks'][idx]),
            'n_targets': int(self._file['n_targets'][idx]),
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataset, device, max_samples=1000):
    """DnCNN-SAR vs Lee / Frost / Median: PSNR and SSIM."""
    model.eval()
    n_eval = min(len(dataset), max_samples)
    methods = ['dncnn', 'lee', 'frost', 'median']
    psnr_sums = {m: 0.0 for m in methods}
    ssim_sums = {m: 0.0 for m in methods}

    for i in range(n_eval):
        noisy_t, clean_t = dataset[i]
        noisy_np = noisy_t[0].numpy()
        clean_np = clean_t[0].numpy()

        # DnCNN
        pred_t  = model(noisy_t.unsqueeze(0).to(device))
        pred_np = pred_t.cpu().numpy()[0, 0]

        outputs = {
            'dncnn':  pred_np,
            'lee':    lee_filter(noisy_np),
            'frost':  frost_filter(noisy_np),
            'median': median_filter(noisy_np),
        }
        for m, out in outputs.items():
            psnr_sums[m] += compute_psnr(out, clean_np)
            ssim_sums[m] += compute_ssim(out, clean_np)

    results = {}
    for m in methods:
        results[f"{m}_psnr"] = float(psnr_sums[m] / n_eval)
        results[f"{m}_ssim"] = float(ssim_sums[m] / n_eval)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = base_parser("P04: DnCNN-SAR Despeckling")
    p.add_argument("--n_train",      type=int,   default=25000)
    p.add_argument("--n_val",        type=int,   default=5000)
    p.add_argument("--n_test",       type=int,   default=5000)
    p.add_argument("--image_size",   type=int,   default=256)
    p.add_argument("--n_filters",    type=int,   default=64)
    p.add_argument("--n_layers",     type=int,   default=17)
    p.add_argument("--w_char",       type=float, default=0.8)
    p.add_argument("--w_ssim",       type=float, default=0.2)
    p.add_argument("--eval_samples", type=int,   default=1000)
    p.add_argument("--data_dir",     type=str,   default=None)
    args = p.parse_args()

    if args.smoke:
        args.n_train     = 256
        args.n_val       = 64
        args.n_test      = 64
        args.epochs      = 2
        args.batch_size  = 4
        args.eval_samples = 20

    seed_everything(args.seed)

    root     = Path(__file__).parent
    data_dir = Path(args.data_dir) if args.data_dir else root / "data"
    ckpt_dir = root / "artifacts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Data generation ---
    if args.generate:
        import subprocess
        cmd = [
            sys.executable, str(root / "generate_data.py"),
            "--n_train",    str(args.n_train),
            "--n_val",      str(args.n_val),
            "--n_test",     str(args.n_test),
            "--image_size", str(args.image_size),
            "--out_dir",    str(data_dir),
            "--seed",       str(args.seed),
        ]
        if args.smoke:
            cmd.append("--smoke")
        subprocess.run(cmd, check=True)

    # --- Datasets ---
    train_path = data_dir / "despeckling_train.h5"
    val_path   = data_dir / "despeckling_val.h5"
    test_path  = data_dir / "despeckling_test.h5"

    for p_check in [train_path, val_path, test_path]:
        if not p_check.exists():
            print(f"ERROR: {p_check} not found. Use --generate flag.")
            return

    train_ds = DespecklingDataset(train_path)
    val_ds   = DespecklingDataset(val_path)
    test_ds  = DespecklingDataset(test_path)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"  Image size: {train_ds.image_size}x{train_ds.image_size}")

    # --- Model ---
    device = "cpu"
    model = DnCNNSAR(n_channels=1, n_filters=args.n_filters,
                     n_layers=args.n_layers).to(device)
    print(f"  Parameters: {count_parameters(model):,}")

    # --- Load checkpoint ---
    if args.checkpoint and Path(args.checkpoint).exists():
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"  Loaded: {args.checkpoint}")

    # --- Eval only ---
    if args.eval_only:
        results = evaluate(model, test_ds, device, args.eval_samples)
        print("\n=== Test Evaluation ===")
        _print_results(results)
        with open(ckpt_dir / "eval_results.json", "w") as f:
            json.dump(results, f, indent=2)
        return

    # --- Training ---
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    criterion = DespecklingLoss(w_char=args.w_char, w_ssim=args.w_ssim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    print("\n=== P04: DnCNN-SAR Despeckling ===")
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
    results = evaluate(model, test_ds, device, args.eval_samples)
    _print_results(results)
    with open(ckpt_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {ckpt_dir}/eval_results.json")


def _print_results(results):
    methods = ['dncnn', 'lee', 'frost', 'median']
    print(f"  {'Method':<10}  {'PSNR':>7}  {'SSIM':>7}")
    print("  " + "-" * 28)
    for m in methods:
        psnr = results.get(f"{m}_psnr", float('nan'))
        ssim = results.get(f"{m}_ssim", float('nan'))
        print(f"  {m.upper():<10}  {psnr:>7.3f}  {ssim:>7.4f}")


if __name__ == "__main__":
    main()
