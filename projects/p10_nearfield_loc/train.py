"""P10 Near-Field Source Localization — 학습 스크립트

Loss: BCE(near_label) + 0.5*MSE(sin/cos angle) + 0.5*near_mask*SmoothL1(range)
Metrics: near/far F1, angle MAE (deg), range MAE (near-field subset)
Baseline: far-field MUSIC for angle + threshold-based near/far decision

사용법:
    python train.py --generate --smoke
    python train.py --generate --epochs 50
    python train.py --eval_only --checkpoint artifacts/best_model.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from common.cli import base_parser
from common.hdf5_io import load_hdf5
from common.metrics import classification_report, regression_report
from common.seed import seed_everything
from common.train_utils import count_parameters
from shared.doa_utils import find_spectrum_peaks

from model import build_model

ARTIFACTS = Path(__file__).parent / "artifacts"
DATA = Path(__file__).parent / "data"

# near-field range normalization: scale to roughly [0, 1] for training
RANGE_SCALE = 5.0   # NEAR_RANGE_MAX


# ─── Dataset ───────────────────────────────────────────────────────────────────

class NearFieldDataset(torch.utils.data.Dataset):
    """Multi-output dataset for near-field localization."""

    def __init__(self, path: Path):
        data = load_hdf5(path, ["x", "near_label", "angle_deg", "range_m"])
        self.x = torch.as_tensor(data["x"], dtype=torch.float32)
        self.near_label = torch.as_tensor(data["near_label"], dtype=torch.float32)
        self.angle_deg = torch.as_tensor(data["angle_deg"], dtype=torch.float32)
        self.range_m = torch.as_tensor(data["range_m"], dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.near_label[idx],
            self.angle_deg[idx],
            self.range_m[idx],
        )


# ─── Loss ──────────────────────────────────────────────────────────────────────

def compute_loss(
    near_logit: torch.Tensor,   # (B, 1)
    angle_sc: torch.Tensor,     # (B, 2)  [sin, cos]
    range_out: torch.Tensor,    # (B, 1)
    near_label: torch.Tensor,   # (B,)
    angle_deg: torch.Tensor,    # (B,)
    range_m: torch.Tensor,      # (B,)
) -> torch.Tensor:
    """Multi-task loss."""
    B = near_label.shape[0]

    # 1. Binary cross entropy for near/far
    bce = F.binary_cross_entropy_with_logits(
        near_logit.squeeze(1), near_label
    )

    # 2. Angle loss via sin/cos representation
    angle_rad = torch.deg2rad(angle_deg)
    angle_sc_gt = torch.stack([torch.sin(angle_rad), torch.cos(angle_rad)], dim=1)  # (B, 2)
    angle_loss = F.mse_loss(angle_sc, angle_sc_gt)

    # 3. Range loss — only for near-field samples
    near_mask = near_label.bool()  # (B,)
    if near_mask.sum() > 0:
        range_pred_near = range_out[near_mask, 0]                          # (K,)
        range_gt_near = range_m[near_mask] / RANGE_SCALE                   # normalize
        range_loss = F.smooth_l1_loss(range_pred_near, range_gt_near)
    else:
        range_loss = torch.tensor(0.0, device=near_logit.device)

    return bce + 0.5 * angle_loss + 0.5 * range_loss


# ─── Training helpers ──────────────────────────────────────────────────────────

# NOTE: common.train_utils.training_loop은 (x, y) 단일-입력/단일-출력 모델만 지원한다.
# NearFieldNet은 (near_logit, angle_sc, range_out) 세 출력을 가지며,
# compute_loss가 (near_logit, angle_sc, range_out, near_label, angle_deg, range_m) 여섯 인수를
# 받아야 하므로 전용 루프를 직접 구현한다.
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, near_lbl, angle, rng_m in loader:
        x = x.to(device)
        near_lbl = near_lbl.to(device)
        angle = angle.to(device)
        rng_m = rng_m.to(device)

        optimizer.zero_grad()
        near_logit, angle_sc, range_out = model(x)
        loss = compute_loss(near_logit, angle_sc, range_out, near_lbl, angle, rng_m)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate_loss(model, loader, device):
    model.eval()
    total_loss = 0.0
    for x, near_lbl, angle, rng_m in loader:
        x = x.to(device)
        near_lbl = near_lbl.to(device)
        angle = angle.to(device)
        rng_m = rng_m.to(device)

        near_logit, angle_sc, range_out = model(x)
        loss = compute_loss(near_logit, angle_sc, range_out, near_lbl, angle, rng_m)
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def run_training(model, train_loader, val_loader, epochs, device):
    ARTIFACTS.mkdir(exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    import time
    import json as _json

    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []}

    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Training for {epochs} epochs on {device}...")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate_loss(model, val_loader, device)
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        improved = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ARTIFACTS / "best_model.pt")
            improved = " *"

        if epoch <= 3 or epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:3d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}"
                  f"  {dt:.1f}s{improved}")

    with open(ARTIFACTS / "history.json", "w") as f:
        _json.dump(history, f, indent=2)

    print(f"  Done. Best val loss: {best_val:.4f}")


# ─── Near-field steering vector (matches generate_data.py) ─────────────────────

def _nearfield_steering_vector(
    theta_deg: float,
    r_m: float,
    N: int,
    d_over_lam: float = 0.5,
) -> np.ndarray:
    """구면파 근거리 steering vector (generate_data.py의 nearfield_steering_vector와 동일)."""
    theta = np.radians(theta_deg)
    x_n = np.arange(N) * d_over_lam
    src_x = r_m * np.sin(theta)
    src_y = r_m * np.cos(theta)
    dist_n = np.sqrt((x_n - src_x) ** 2 + src_y ** 2)
    phase = -2 * np.pi * (dist_n - r_m)
    return np.exp(1j * phase)


# ─── MUSIC baseline ────────────────────────────────────────────────────────────

def music_angle_estimate(x_np: np.ndarray, n_sources: int = 1) -> float:
    """Near-field MUSIC으로 단일 각도 추정 (구면파 모델, generate_data.py와 일치).

    Near-field 조건에서 range를 모르므로 NEAR_RANGE_MIN~MAX 구간의
    대표 거리들에서 near-field steering vector를 생성하고,
    각 거리에서 계산된 MUSIC 스펙트럼을 평균하여 각도를 추정한다.

    Parameters
    ----------
    x_np : (2, N_RX, N_SNAPSHOTS)

    Returns
    -------
    angle_est : float [deg]
    """
    X = x_np[0] + 1j * x_np[1]   # (N_RX, N_SNAPSHOTS)
    T = X.shape[1]
    R = (X @ X.conj().T) / T
    R = R / (np.linalg.norm(R, 'fro') + 1e-10)

    angles_grid = np.linspace(-90, 90, 361)
    N_rx = X.shape[0]

    # Representative near-field ranges spanning NEAR_RANGE_MIN to NEAR_RANGE_MAX
    range_grid = np.linspace(RANGE_SCALE * 0.1, RANGE_SCALE, 5)  # 0.5m ~ 5.0m

    try:
        eigvals, eigvecs = np.linalg.eigh(R)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]
        En = eigvecs[:, n_sources:]  # noise subspace

        # Average MUSIC spectrum over representative near-field ranges
        P_sum = np.zeros(len(angles_grid))
        for r_m in range_grid:
            P_r = np.zeros(len(angles_grid))
            for i, theta in enumerate(angles_grid):
                a = _nearfield_steering_vector(theta, r_m, N_rx)
                denom = np.real(a.conj() @ En @ En.conj().T @ a)
                P_r[i] = 1.0 / (denom + 1e-20)
            P_sum += P_r
        P = P_sum / len(range_grid)
    except Exception:
        return 0.0

    peaks = find_spectrum_peaks(P, angles_grid, n_sources=1)
    return float(peaks[0]) if len(peaks) > 0 else 0.0


# ─── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, test_data: dict, device: str) -> dict:
    model.eval()

    x_np = test_data["x"]
    near_label_np = test_data["near_label"]
    angle_deg_np = test_data["angle_deg"]
    range_m_np = test_data["range_m"]
    n_sources_np = test_data["n_sources"]

    x_t = torch.as_tensor(x_np, dtype=torch.float32)

    # Batch predict
    batch_size = 256
    near_probs, angle_preds, range_preds = [], [], []

    for i in range(0, len(x_t), batch_size):
        xb = x_t[i:i+batch_size].to(device)
        nl, asc, ro = model(xb)
        near_probs.append(torch.sigmoid(nl).cpu().numpy())
        angle_preds.append(asc.cpu().numpy())
        range_preds.append(ro.cpu().numpy())

    near_prob_np = np.concatenate(near_probs)[:, 0]      # (N,)
    angle_sc_np = np.concatenate(angle_preds)            # (N, 2)
    range_pred_np = np.concatenate(range_preds)[:, 0] * RANGE_SCALE  # (N,)

    # Near/far classification
    near_pred_np = (near_prob_np >= 0.5).astype(int)
    cls = classification_report(near_label_np, near_pred_np, y_prob=near_prob_np)

    # Angle MAE (deg) via atan2(sin, cos)
    angle_pred_deg = np.degrees(np.arctan2(angle_sc_np[:, 0], angle_sc_np[:, 1]))
    angle_mae = float(np.mean(np.abs(angle_pred_deg - angle_deg_np)))

    # Range MAE — only near-field subset
    near_mask = near_label_np == 1
    if near_mask.sum() > 0:
        range_mae_near = float(np.mean(np.abs(range_pred_np[near_mask] - range_m_np[near_mask])))
    else:
        range_mae_near = float("nan")

    # Joint localization accuracy: correct near/far AND angle error < 5 deg
    correct_cls = (near_pred_np == near_label_np)
    correct_angle = (np.abs(angle_pred_deg - angle_deg_np) < 5.0)
    joint_acc = float(np.mean(correct_cls & correct_angle))

    # Baseline: near-field MUSIC angle + threshold near/far (always predict far-field)
    print("  Computing near-field MUSIC baseline (may take a while)...")
    n_baseline = min(500, len(x_np))
    music_angles = np.array([
        music_angle_estimate(x_np[i]) for i in range(n_baseline)
    ])
    music_angle_mae = float(np.mean(np.abs(music_angles - angle_deg_np[:n_baseline])))
    # Threshold: always predict far-field → near/far F1 = ~0.5 (all-far)
    baseline_near_pred = np.zeros(n_baseline, dtype=int)
    baseline_cls = classification_report(near_label_np[:n_baseline], baseline_near_pred)

    return {
        "model": {
            "near_far_f1_macro": round(cls["f1_macro"], 4),
            "near_far_accuracy": round(cls["accuracy"], 4),
            "angle_mae_deg": round(angle_mae, 3),
            "range_mae_near_m": round(range_mae_near, 3) if not np.isnan(range_mae_near) else None,
            "joint_loc_acc_5deg": round(joint_acc, 4),
        },
        "baseline_music_nearfield": {
            "near_far_f1_macro": round(baseline_cls["f1_macro"], 4),
            "near_far_accuracy": round(baseline_cls["accuracy"], 4),
            "angle_mae_deg": round(music_angle_mae, 3),
            "range_mae_near_m": None,
            "joint_loc_acc_5deg": None,
        },
    }


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = base_parser("P10 Near-Field Source Localization Training")
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
    train_ds = NearFieldDataset(DATA / "train.h5")
    val_ds = NearFieldDataset(DATA / "val.h5")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    epochs = 2 if args.smoke else args.epochs
    batch_size = args.batch_size

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # 3. 모델
    model = build_model().to(device)
    print(f"\n=== Model ===")
    print(f"  Parameters: {count_parameters(model):,}")

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
        run_training(model, train_loader, val_loader, epochs, device)
        model.load_state_dict(torch.load(ARTIFACTS / "best_model.pt", map_location=device))

    # 5. 평가
    print("\n=== Evaluation ===")
    test_data = load_hdf5(DATA / "test.h5",
                          ["x", "near_label", "angle_deg", "range_m", "n_sources"])
    metrics = evaluate(model, test_data, device)

    for group, vals in metrics.items():
        print(f"  [{group}]")
        for k, v in vals.items():
            print(f"    {k}: {v}")

    metrics_path = ARTIFACTS / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
