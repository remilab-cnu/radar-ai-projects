"""P09 RD Super-Resolution — 학습 스크립트

Loss: L1 (primary) + gradient loss (optional)
Metrics: PSNR, NMSE, peak localization error
Baseline: bicubic interpolation

사용법:
    python train.py --generate --smoke        # 데이터 생성 + smoke 학습
    python train.py --smoke                   # 기존 데이터로 smoke 학습
    python train.py --generate --epochs 50   # 전체 학습
    python train.py --eval_only --checkpoint artifacts/best_model.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import subprocess

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from common.cli import base_parser
from common.hdf5_io import load_hdf5
from common.metrics import psnr, nmse
from common.seed import seed_everything
from common.train_utils import count_parameters, training_loop

from model import build_model

ARTIFACTS = Path(__file__).parent / "artifacts"
DATA = Path(__file__).parent / "data"


# ─── Loss ──────────────────────────────────────────────────────────────────────

class GradientLoss(nn.Module):
    """Sobel gradient loss for sharpness."""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gx_pred = F.conv2d(pred, self.kx, padding=1)
        gy_pred = F.conv2d(pred, self.ky, padding=1)
        gx_tgt = F.conv2d(target, self.kx, padding=1)
        gy_tgt = F.conv2d(target, self.ky, padding=1)
        return F.l1_loss(gx_pred, gx_tgt) + F.l1_loss(gy_pred, gy_tgt)


class SRLoss(nn.Module):
    """L1 + weighted gradient loss."""

    def __init__(self, grad_weight: float = 0.1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.grad = GradientLoss()
        self.grad_weight = grad_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.l1(pred, target) + self.grad_weight * self.grad(pred, target)


# ─── Dataset ───────────────────────────────────────────────────────────────────

def load_split(name: str) -> TensorDataset:
    path = DATA / f"{name}.h5"
    data = load_hdf5(path, ["x_lr", "y_hr"])
    x = torch.as_tensor(data["x_lr"], dtype=torch.float32)
    y = torch.as_tensor(data["y_hr"], dtype=torch.float32)
    return TensorDataset(x, y)


def load_split_full(name: str) -> dict:
    path = DATA / f"{name}.h5"
    return load_hdf5(path, ["x_lr", "y_hr", "peak_mask", "n_targets", "snr_db"])


def load_split_attrs(name: str) -> dict:
    """Load scalar HDF5 attrs for data-contract reporting."""
    path = DATA / f"{name}.h5"
    with h5py.File(path, "r") as f:
        attrs = {}
        for key, value in f.attrs.items():
            if isinstance(value, np.generic):
                value = value.item()
            attrs[key] = value
    return attrs


# ─── Bicubic baseline ──────────────────────────────────────────────────────────

def bicubic_upsample(x_lr: torch.Tensor) -> torch.Tensor:
    """Bicubic interpolation: (B, 1, 32, 32) → (B, 1, 64, 64)."""
    return F.interpolate(x_lr, scale_factor=2, mode="bicubic", align_corners=False)


def zero_pad_upsample(x_lr: torch.Tensor) -> torch.Tensor:
    """Zero-insert baseline: copy LR bins onto even HR bins, fill gaps with 0.

    This is an intentionally simple image-domain zero-padding baseline. It is
    reported separately from bicubic and the learned model, and is not claimed
    to recover missing physical bandwidth or chirps.
    """
    b, c, h, w = x_lr.shape
    out = torch.zeros((b, c, h * 2, w * 2), dtype=x_lr.dtype, device=x_lr.device)
    out[..., ::2, ::2] = x_lr
    return out


# ─── Peak localization metric ──────────────────────────────────────────────────

def peak_localization_error(pred_np: np.ndarray, peak_mask_np: np.ndarray) -> float:
    """GT peak mask와 예측 map 사이의 평균 localization error.

    Parameters
    ----------
    pred_np : (N, 1, 64, 64) float — model output
    peak_mask_np : (N, 1, 64, 64) float — binary GT mask

    Returns
    -------
    mean pixel distance to nearest GT peak
    """
    from scipy.ndimage import label, center_of_mass
    from scipy.spatial.distance import cdist

    errors = []
    N = pred_np.shape[0]

    for i in range(N):
        pred_map = pred_np[i, 0]
        mask = peak_mask_np[i, 0]

        # GT peak centroids from mask
        labeled, n_obj = label(mask > 0.5)
        if n_obj == 0:
            continue
        gt_centers = np.array(center_of_mass(mask, labeled, range(1, n_obj + 1)))
        if gt_centers.ndim == 1:
            gt_centers = gt_centers[np.newaxis]

        # Predicted peaks: top-k bright bins. This is a simple localization
        # proxy, not a strict local-maxima detector.
        flat_idx = np.argsort(pred_map.ravel())[::-1][:n_obj * 3]
        row_idx = flat_idx // pred_map.shape[1]
        col_idx = flat_idx % pred_map.shape[1]
        pred_centers = np.stack([row_idx, col_idx], axis=1).astype(float)

        if len(pred_centers) == 0:
            errors.append(pred_map.shape[0])
            continue

        # Match each GT to nearest predicted center
        D = cdist(gt_centers, pred_centers)
        matched_dist = D.min(axis=1)
        errors.extend(matched_dist.tolist())

    return float(np.mean(errors)) if errors else 0.0


# ─── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, test_data: dict, device: str) -> dict:
    model.eval()
    x_lr = torch.as_tensor(test_data["x_lr"], dtype=torch.float32)
    y_hr = test_data["y_hr"]
    peak_mask = test_data["peak_mask"]

    # Batch predict
    batch_size = 128
    preds = []
    for i in range(0, len(x_lr), batch_size):
        xb = x_lr[i:i+batch_size].to(device)
        preds.append(model(xb).cpu().numpy())
    pred_np = np.concatenate(preds, axis=0)   # (N, 1, 64, 64)

    # Bicubic baseline
    bicubic_np = bicubic_upsample(x_lr).numpy()
    zero_pad_np = zero_pad_upsample(x_lr).numpy()

    # Metrics
    model_psnr = psnr(y_hr, pred_np)
    model_nmse = nmse(y_hr, pred_np)
    bicubic_psnr = psnr(y_hr, bicubic_np)
    bicubic_nmse = nmse(y_hr, bicubic_np)
    zero_pad_psnr = psnr(y_hr, zero_pad_np)
    zero_pad_nmse = nmse(y_hr, zero_pad_np)
    peak_err = peak_localization_error(pred_np, peak_mask)
    bicubic_peak_err = peak_localization_error(bicubic_np, peak_mask)
    zero_pad_peak_err = peak_localization_error(zero_pad_np, peak_mask)

    return {
        "model": {
            "psnr_db": round(model_psnr, 3),
            "nmse": round(model_nmse, 6),
            "peak_loc_err_px": round(peak_err, 3),
        },
        "baseline_bicubic": {
            "psnr_db": round(bicubic_psnr, 3),
            "nmse": round(bicubic_nmse, 6),
            "peak_loc_err_px": round(bicubic_peak_err, 3),
        },
        "baseline_zero_pad": {
            "psnr_db": round(zero_pad_psnr, 3),
            "nmse": round(zero_pad_nmse, 6),
            "peak_loc_err_px": round(zero_pad_peak_err, 3),
        },
    }


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = base_parser("P09 RD Super-Resolution Training")
    args = parser.parse_args()

    seed_everything(args.seed)
    ARTIFACTS.mkdir(exist_ok=True)
    device = "cpu"

    # 1. 데이터 생성
    if args.generate:
        cmd = [sys.executable, str(Path(__file__).parent / "generate_data.py")]
        if args.smoke:
            cmd.append("--smoke")
        cmd += ["--seed", str(args.seed)]
        print("=== Generating data ===")
        subprocess.run(cmd, check=True)

    # 2. 데이터 로드
    print("\n=== Loading data ===")
    train_ds = load_split("train")
    val_ds = load_split("val")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    epochs = 2 if args.smoke else args.epochs
    batch_size = args.batch_size

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # 3. 모델
    model = build_model().to(device)
    print(f"\n=== Model ===")
    print(f"  Parameters: {count_parameters(model):,}")

    criterion = SRLoss(grad_weight=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # 4. 학습 / checkpoint 로드
    if args.eval_only:
        ckpt = args.checkpoint or str(ARTIFACTS / "best_model.pt")
        print(f"\n=== Eval only — loading {ckpt} ===")
        model.load_state_dict(torch.load(ckpt, map_location=device))
    else:
        if args.checkpoint:
            print(f"  Resuming from {args.checkpoint}")
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))

        print(f"\n=== Training ({epochs} epochs) ===")
        training_loop(
            model, train_loader, val_loader, criterion, optimizer,
            epochs=epochs, checkpoint_dir=ARTIFACTS, device=device,
            scheduler=scheduler,
        )
        model.load_state_dict(torch.load(ARTIFACTS / "best_model.pt", map_location=device))

    # 5. 평가
    print("\n=== Evaluation ===")
    test_data = load_split_full("test")
    metrics = evaluate(model, test_data, device)
    test_attrs = load_split_attrs("test")
    metrics["data_contract"] = {
        key: test_attrs[key]
        for key in (
            "generation_mode",
            "hr_bw_hz",
            "lr_bw_hz",
            "hr_n_chirps",
            "lr_n_chirps",
            "hr_range_bin_spacing_m",
            "lr_range_bin_spacing_m",
            "hr_doppler_bin_spacing_mps",
            "lr_doppler_bin_spacing_mps",
        )
        if key in test_attrs
    }

    for group, vals in metrics.items():
        print(f"  [{group}]")
        if isinstance(vals, dict):
            for k, v in vals.items():
                print(f"    {k}: {v}")
        else:
            print(f"    {vals}")

    metrics_path = ARTIFACTS / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
