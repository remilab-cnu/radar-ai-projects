"""P07 Full-Duplex SIC — 학습 스크립트

Usage:
    python train.py --generate --smoke            # 데이터 생성 + 스모크 학습
    python train.py --generate --epochs 30        # 풀 학습
    python train.py --eval_only --checkpoint artifacts/best_model.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import subprocess

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common.cli import base_parser
from common.hdf5_io import load_hdf5
from common.seed import seed_everything
from common.train_utils import count_parameters, training_loop

from model import SICUNet

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
ARTIFACT_DIR = PROJECT_DIR / "artifacts"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_dataset(split: str) -> TensorDataset:
    """HDF5에서 데이터 로드 → TensorDataset 반환.

    x: (N, 4, 512) = concat(tx_ref, rx_mix)
    y: (N, 2, 512) = y_si  (SI 추정 목표)
    """
    path = DATA_DIR / f"{split}.h5"
    data = load_hdf5(path, ["tx_ref", "rx_mix", "y_si", "y_clean"])
    x = np.concatenate([data["tx_ref"], data["rx_mix"]], axis=1)  # (N, 4, 512)
    x_t = torch.as_tensor(x, dtype=torch.float32)
    y_si_t = torch.as_tensor(data["y_si"], dtype=torch.float32)
    y_clean_t = torch.as_tensor(data["y_clean"], dtype=torch.float32)
    # y_si_t와 y_clean_t를 함께 묶어서 반환 (cat along ch dim)
    y_t = torch.cat([y_si_t, y_clean_t], dim=1)  # (N, 4, 512): [y_si(2), y_clean(2)]
    return TensorDataset(x_t, y_t)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class SICLoss(nn.Module):
    """Combined SIC loss.

    L = 0.7 * SmoothL1(si_hat, y_si) + 0.3 * SmoothL1(clean_hat, y_clean)

    clean_hat = rx_mix - si_hat  (residual)
    """

    def __init__(self):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, si_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        si_hat : (B, 2, 512)
        y      : (B, 4, 512) = [y_si(2), y_clean(2)]
        """
        y_si = y[:, :2, :]     # (B, 2, 512)
        y_clean = y[:, 2:, :]  # (B, 2, 512)
        rx_mix = y_si + y_clean  # rx_mix 복원 (y_si + y_clean = rx_mix)

        clean_hat = rx_mix - si_hat

        loss_si = self.smooth_l1(si_hat, y_si)
        loss_clean = self.smooth_l1(clean_hat, y_clean)
        return 0.7 * loss_si + 0.3 * loss_clean


# ---------------------------------------------------------------------------
# Baseline: Complex NLMS
# ---------------------------------------------------------------------------

def nlms_baseline(tx_ref: np.ndarray, rx_mix: np.ndarray,
                  n_taps: int = 32, mu: float = 0.05) -> np.ndarray:
    """Complex NLMS adaptive filter (벡터화 미적용 — 교육용 단순 구현).

    Parameters
    ----------
    tx_ref : (N,) complex — TX 기준 신호
    rx_mix : (N,) complex — 수신 혼합 신호
    n_taps : int — 필터 탭 수
    mu : float — 스텝 크기

    Returns
    -------
    si_hat : (N,) complex — SI 추정
    """
    N = len(tx_ref)
    w = np.zeros(n_taps, dtype=np.complex64)
    si_hat = np.zeros(N, dtype=np.complex64)

    for n in range(n_taps, N):
        x_buf = tx_ref[n:n - n_taps:-1]  # (n_taps,) — 최근 탭 버퍼 (역순)
        y_hat = np.dot(w.conj(), x_buf)
        si_hat[n] = y_hat
        e = rx_mix[n] - y_hat            # 오차 = 관측 - SI 추정
        denom = np.dot(x_buf.conj(), x_buf).real + 1e-10
        w += (mu / denom) * np.conj(e) * x_buf

    return si_hat


def evaluate_nlms_baseline(split: str = "test") -> dict:
    """NLMS 기준선 성능 평가."""
    data = load_hdf5(DATA_DIR / f"{split}.h5",
                     ["tx_ref", "rx_mix", "y_si", "y_clean"])

    cancellation_dbs = []
    n_eval = min(200, data["tx_ref"].shape[0])

    for i in range(n_eval):
        tx = data["tx_ref"][i, 0] + 1j * data["tx_ref"][i, 1]   # (512,) complex
        rx = data["rx_mix"][i, 0] + 1j * data["rx_mix"][i, 1]
        y_si = data["y_si"][i, 0] + 1j * data["y_si"][i, 1]

        si_hat = nlms_baseline(tx, rx)
        residual = y_si - si_hat

        p_si = np.mean(np.abs(y_si) ** 2)
        p_res = np.mean(np.abs(residual) ** 2) + 1e-20
        canc_db = 10 * np.log10(p_si / p_res)
        cancellation_dbs.append(canc_db)

    return {"nlms_cancellation_db_mean": float(np.mean(cancellation_dbs)),
            "nlms_cancellation_db_std": float(np.std(cancellation_dbs))}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model: nn.Module, split: str = "test",
                   device: str = "cpu") -> dict:
    """모델 성능 평가.

    Metrics:
    - cancellation_db: 10*log10(||y_si||^2 / ||y_si - si_hat||^2)
    - output_sir_gain_db: 출력 SIR - 입력 SIR
    - clean_nmse: ||y_clean - clean_hat||^2 / ||y_clean||^2
    """
    model.eval()
    dataset = load_dataset(split)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    cancellation_dbs = []
    sir_gains = []
    clean_nmses = []

    data_raw = load_hdf5(DATA_DIR / f"{split}.h5", ["sir_db"])
    input_sir = data_raw["sir_db"]
    idx = 0

    for x, y in loader:
        x = x.to(device)
        si_hat = model(x)  # (B, 2, 512)

        y_si = y[:, :2, :]      # (B, 2, 512)
        y_clean = y[:, 2:, :]   # (B, 2, 512)
        rx_mix = y_si + y_clean

        clean_hat = rx_mix - si_hat

        B = x.shape[0]
        si_hat_np = si_hat.cpu().numpy()
        y_si_np = y_si.numpy()
        y_clean_np = y_clean.numpy()
        clean_hat_np = clean_hat.cpu().numpy()

        for b in range(B):
            # SI cancellation depth
            p_si = np.mean(y_si_np[b] ** 2) + 1e-20
            residual = y_si_np[b] - si_hat_np[b]
            p_res = np.mean(residual ** 2) + 1e-20
            canc_db = 10 * np.log10(p_si / p_res)
            cancellation_dbs.append(canc_db)

            # Output SIR gain
            p_clean = np.mean(y_clean_np[b] ** 2) + 1e-20
            p_rx_si = np.mean(y_si_np[b] ** 2) + 1e-20
            input_sir_b = 10 * np.log10(p_clean / p_rx_si)  # clean/SI
            p_clean_hat = np.mean(clean_hat_np[b] ** 2) + 1e-20
            p_residual_si = np.mean((y_si_np[b] - si_hat_np[b]) ** 2) + 1e-20
            output_sir_b = 10 * np.log10(p_clean_hat / (p_residual_si + 1e-20))
            sir_gains.append(output_sir_b - input_sir_b)

            # Clean NMSE
            p_y_clean = np.mean(y_clean_np[b] ** 2) + 1e-20
            p_err_clean = np.mean((y_clean_np[b] - clean_hat_np[b]) ** 2)
            clean_nmses.append(p_err_clean / p_y_clean)

        idx += B

    return {
        "cancellation_db_mean": float(np.mean(cancellation_dbs)),
        "cancellation_db_std": float(np.std(cancellation_dbs)),
        "output_sir_gain_db_mean": float(np.mean(sir_gains)),
        "clean_nmse_mean": float(np.mean(clean_nmses)),
        "clean_nmse_db_mean": float(10 * np.log10(np.mean(clean_nmses) + 1e-20)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = base_parser("P07: Full-Duplex Self-Interference Cancellation")
    args = parser.parse_args()

    seed_everything(args.seed)
    ARTIFACT_DIR.mkdir(exist_ok=True)

    # 1. 데이터 생성
    if args.generate:
        cmd = [sys.executable, str(PROJECT_DIR / "generate_data.py")]
        if args.smoke:
            cmd.append("--smoke")
        cmd += ["--seed", str(args.seed)]
        subprocess.run(cmd, check=True)

    # 2. 데이터 로드
    print("\nLoading datasets...")
    train_ds = load_dataset("train")
    val_ds = load_dataset("val")
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

    epochs = 2 if args.smoke else args.epochs
    batch_size = args.batch_size
    device = "cpu"

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

    # 3. 모델
    model = SICUNet().to(device)
    print(f"\nModel parameters: {count_parameters(model):,}")

    if args.eval_only:
        if args.checkpoint is None:
            args.checkpoint = str(ARTIFACT_DIR / "best_model.pt")
        print(f"Loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        # 4. 학습
        criterion = SICLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        # training_loop은 (x, y)를 그대로 criterion(out, y)에 전달하므로
        # SICLoss가 y[:, :2] = y_si, y[:, 2:] = y_clean를 올바르게 처리
        print("\nTraining...")
        history = training_loop(
            model, train_loader, val_loader, criterion, optimizer,
            epochs=epochs, checkpoint_dir=ARTIFACT_DIR,
            device=device, scheduler=scheduler,
        )

        # 최적 모델 로드
        model.load_state_dict(
            torch.load(ARTIFACT_DIR / "best_model.pt", map_location=device)
        )

    # 5. 평가
    print("\nEvaluating model on test set...")
    test_metrics = evaluate_model(model, "test", device)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # NLMS 기준선
    print("\nEvaluating NLMS baseline...")
    nlms_metrics = evaluate_nlms_baseline("test")
    for k, v in nlms_metrics.items():
        print(f"  {k}: {v:.4f}")

    # 6. 메트릭 저장
    all_metrics = {"model": test_metrics, "baseline_nlms": nlms_metrics}
    metrics_path = ARTIFACT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
