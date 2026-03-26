#!/usr/bin/env python3
"""P03 -- Train DeepMUSIC CNN for DoA estimation.

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
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.cli import base_parser
from common.seed import seed_everything
from common.train_utils import training_loop, count_parameters
from model import DeepMUSIC

try:
    from shared.doa_utils import (
        music_spectrum, find_spectrum_peaks, compute_doa_rmse,
    )
    _HAS_DOA_UTILS = True
except ImportError:
    _HAS_DOA_UTILS = False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DoADataset(Dataset):
    """HDF5 DoA dataset: returns (covariance, spectrum) tensors."""

    def __init__(self, path: Path):
        self.path = path
        with h5py.File(path, 'r') as f:
            self.n_samples = f['covariance'].shape[0]
            self.n_rx      = int(f.attrs['n_rx'])
            self.grid_size = int(f.attrs['grid_size'])
        self._file = None

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self.path, 'r')

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        self._open()
        cov  = self._file['covariance'][idx]  # (2, N_rx, N_rx)
        spec = self._file['spectrum'][idx]    # (grid_size,)
        return torch.from_numpy(cov), torch.from_numpy(spec)

    def get_meta(self, idx):
        self._open()
        return {
            'snr_db':      float(self._file['snr_db'][idx]),
            'n_sources':   int(self._file['n_sources'][idx]),
            'n_snapshots': int(self._file['n_snapshots'][idx]),
            'coherent':    bool(self._file['coherent'][idx]),
            'angles':      self._file['angles'][idx].tolist(),
        }

    def get_raw_covariance(self, idx):
        self._open()
        cov = self._file['covariance'][idx]
        return cov[0] + 1j * cov[1]


# ---------------------------------------------------------------------------
# Evaluation: DNN RMSE vs MUSIC
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataset, device, max_samples=2000):
    """Compute mean RMSE (degrees) for DeepMUSIC vs MUSIC."""
    model.eval()
    n_eval = min(len(dataset), max_samples)
    angle_grid = np.linspace(-90, 90, dataset.grid_size)

    rmse_dnn = []
    rmse_music = []

    for i in range(n_eval):
        cov_t, _ = dataset[i]
        meta = dataset.get_meta(i)
        true_angles = meta['angles']
        K = meta['n_sources']

        # DeepMUSIC prediction
        pred_spec = model(cov_t.unsqueeze(0).to(device)).cpu().numpy()[0]
        # Simple peak finding: top-K peaks
        peaks = np.argsort(pred_spec)[-K:]
        est_dnn = sorted(angle_grid[peaks].tolist())
        rmse_dnn.append(_rmse(est_dnn, true_angles))

        # MUSIC (classical baseline)
        if _HAS_DOA_UTILS:
            try:
                R = dataset.get_raw_covariance(i)
                P_music = music_spectrum(R, dataset.n_rx, angle_grid, K)
                est_music = find_spectrum_peaks(P_music, angle_grid, K)
                rmse_music.append(compute_doa_rmse(est_music, true_angles))
            except Exception:
                rmse_music.append(90.0)
        else:
            rmse_music.append(float('nan'))

    results = {
        "dnn_rmse_mean":   float(np.mean(rmse_dnn)),
        "dnn_rmse_median": float(np.median(rmse_dnn)),
    }
    if rmse_music and not np.isnan(rmse_music[0]):
        results["music_rmse_mean"]   = float(np.nanmean(rmse_music))
        results["music_rmse_median"] = float(np.nanmedian(rmse_music))
    return results


def _rmse(estimated, true_angles):
    """Match estimated to true angles (greedy nearest-neighbor), return RMSE."""
    if len(estimated) == 0 or len(true_angles) == 0:
        return 90.0
    est = sorted(estimated)
    tru = sorted(true_angles)
    # Pad shorter list with 90-degree penalties
    while len(est) < len(tru):
        est.append(90.0)
    while len(tru) < len(est):
        tru.append(0.0)
    errors = [abs(e - t) for e, t in zip(est, tru)]
    return float(np.sqrt(np.mean(np.array(errors) ** 2)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = base_parser("P03: DeepMUSIC CNN DoA Estimator")
    p.add_argument("--n_train",       type=int,   default=100000)
    p.add_argument("--n_val",         type=int,   default=20000)
    p.add_argument("--n_test",        type=int,   default=20000)
    p.add_argument("--n_rx",          type=int,   default=12)
    p.add_argument("--grid_size",     type=int,   default=181)
    p.add_argument("--dropout",       type=float, default=0.3)
    p.add_argument("--eval_samples",  type=int,   default=2000)
    p.add_argument("--data_dir",      type=str,   default=None)
    args = p.parse_args()

    if args.smoke:
        args.n_train     = 256
        args.n_val       = 64
        args.n_test      = 64
        args.epochs      = 2
        args.batch_size  = 32
        args.eval_samples = 50

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
            "--n_train",   str(args.n_train),
            "--n_val",     str(args.n_val),
            "--n_test",    str(args.n_test),
            "--n_rx",      str(args.n_rx),
            "--grid_size", str(args.grid_size),
            "--out_dir",   str(data_dir),
            "--seed",      str(args.seed),
        ]
        if args.smoke:
            cmd.append("--smoke")
        subprocess.run(cmd, check=True)

    # --- Datasets ---
    train_path = data_dir / "doa_train.h5"
    val_path   = data_dir / "doa_val.h5"
    test_path  = data_dir / "doa_test.h5"

    for p_check in [train_path, val_path, test_path]:
        if not p_check.exists():
            print(f"ERROR: {p_check} not found. Use --generate flag.")
            return

    train_ds = DoADataset(train_path)
    val_ds   = DoADataset(val_path)
    test_ds  = DoADataset(test_path)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"  N_rx={train_ds.n_rx}, grid_size={train_ds.grid_size}")

    # --- Model ---
    device = "cpu"
    model = DeepMUSIC(n_rx=train_ds.n_rx, grid_size=train_ds.grid_size,
                      dropout=args.dropout).to(device)
    print(f"  Parameters: {count_parameters(model):,}")

    # --- Load checkpoint ---
    if args.checkpoint and Path(args.checkpoint).exists():
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"  Loaded: {args.checkpoint}")

    # --- Eval only ---
    if args.eval_only:
        results = evaluate(model, test_ds, device, args.eval_samples)
        print("\n=== Test Evaluation ===")
        for k, v in results.items():
            print(f"  {k}: {v:.3f} deg")
        with open(ckpt_dir / "eval_results.json", "w") as f:
            json.dump(results, f, indent=2)
        return

    # --- Training ---
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    print("\n=== P03: DeepMUSIC CNN ===")
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
    for k, v in results.items():
        print(f"  {k}: {v:.3f} deg")
    with open(ckpt_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {ckpt_dir}/eval_results.json")


if __name__ == "__main__":
    main()
