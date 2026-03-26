"""P08 Jammer Null Steering — 학습 스크립트

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
from common.train_utils import count_parameters

from model import CovNet
from shared.doa_utils import steering_vector, music_spectrum, find_spectrum_peaks

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
ARTIFACT_DIR = PROJECT_DIR / "artifacts"

N_RX = 8
D_OVER_LAM = 0.5


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class JammerDataset(torch.utils.data.Dataset):
    """재머 방향 회귀 데이터셋.

    x1 : cov        (2, 8, 8)
    x2 : look_angle (1,)      [degrees]
    y  : sin(jammer_angle)    scalar
    """

    def __init__(self, path: Path):
        data = load_hdf5(path, ["cov", "look_angle_deg", "jammer_angle_deg"])
        self.cov = torch.as_tensor(data["cov"], dtype=torch.float32)
        self.look = torch.as_tensor(
            data["look_angle_deg"][:, None], dtype=torch.float32
        )  # (N, 1)
        # 타겟: sin(jammer_angle_deg in radians)
        jammer_rad = np.radians(data["jammer_angle_deg"])
        self.sin_jammer = torch.as_tensor(
            np.sin(jammer_rad)[:, None], dtype=torch.float32
        )  # (N, 1)
        # 원래 각도 보존 (평가용)
        self.jammer_angle_deg = data["jammer_angle_deg"]

    def __len__(self):
        return len(self.cov)

    def __getitem__(self, idx):
        return self.cov[idx], self.look[idx], self.sin_jammer[idx]


# ---------------------------------------------------------------------------
# Wrapper for training_loop compatibility
# ---------------------------------------------------------------------------

class JammerDatasetFlat(torch.utils.data.Dataset):
    """training_loop과 호환되도록 (x, y) 쌍으로 반환.

    x: (2, 8, 8) cov 만 (look_angle은 별도 처리 필요)
    → training_loop 대신 직접 학습 루프 사용
    """
    pass


# ---------------------------------------------------------------------------
# LCMV Beamformer 유틸
# ---------------------------------------------------------------------------

def lcmv_null_depth(R: np.ndarray, look_angle_deg: float,
                    null_angle_deg: float) -> float:
    """LCMV 빔포머로 null 방향 전력을 측정 (null depth).

    Constraints:
      - 원하는 방향 응답 = 1 (distortionless)
      - null 방향 응답 = 0

    Returns
    -------
    null_depth_db : float — null 방향 전력 [dB] (낮을수록 좋음)
    """
    a_look = steering_vector(look_angle_deg, N_RX, D_OVER_LAM)  # (8,)
    a_null = steering_vector(null_angle_deg, N_RX, D_OVER_LAM)  # (8,)

    # 제약 행렬 C = [a_look, a_null], 목표 f = [1, 0]
    C = np.stack([a_look, a_null], axis=1)  # (8, 2)
    f = np.array([1.0, 0.0], dtype=np.complex64)

    try:
        R_inv = np.linalg.inv(R + 1e-8 * np.eye(N_RX))
        # LCMV 가중치: w = R_inv @ C @ (C^H @ R_inv @ C)^-1 @ f
        RiC = R_inv @ C
        CRiC = C.conj().T @ RiC  # (2, 2)
        w = RiC @ np.linalg.inv(CRiC + 1e-12 * np.eye(2)) @ f
    except np.linalg.LinAlgError:
        return 0.0  # 역행렬 실패 시 0 dB

    # null 방향 응답 전력
    null_response = abs(w.conj() @ a_null) ** 2
    null_depth_db = float(10 * np.log10(null_response + 1e-20))
    return null_depth_db


# ---------------------------------------------------------------------------
# Baseline: MUSIC + LCMV
# ---------------------------------------------------------------------------

def music_lcmv_baseline(split: str = "test", n_eval: int = 500) -> dict:
    """MUSIC으로 재머 방향 추정 → LCMV null steering 성능 측정."""
    data = load_hdf5(DATA_DIR / f"{split}.h5",
                     ["cov", "look_angle_deg", "jammer_angle_deg", "n_jammers"])

    angle_grid = np.linspace(-90, 90, 361)
    mae_list = []
    within2_list = []
    null_depth_list = []

    n_eval = min(n_eval, data["cov"].shape[0])

    for i in range(n_eval):
        cov_2ch = data["cov"][i]  # (2, 8, 8)
        R = cov_2ch[0] + 1j * cov_2ch[1]  # (8, 8) complex

        look_deg = float(data["look_angle_deg"][i])
        true_jammer = float(data["jammer_angle_deg"][i])
        n_jam = int(data["n_jammers"][i])

        # MUSIC 스펙트럼 (재머 수 = n_jam, 원하는 신호 제외)
        try:
            P = music_spectrum(R, N_RX, angle_grid, n_sources=n_jam + 1)
            peaks = find_spectrum_peaks(P, angle_grid, n_sources=n_jam + 1)
            # look_angle 근처 제거 → 나머지 중 가장 강한 피크 = 재머 추정
            jammer_cands = [a for a in peaks if abs(a - look_deg) > 5.0]
            if len(jammer_cands) == 0:
                est_jammer = float(angle_grid[np.argmax(P)])
            else:
                est_jammer = jammer_cands[0]
        except Exception:
            est_jammer = 0.0

        err = abs(true_jammer - est_jammer)
        mae_list.append(err)
        within2_list.append(float(err <= 2.0))

        # LCMV null depth (예측 방향에 null steering)
        R_denorm = R * (np.linalg.norm(R, 'fro') + 1e-10)  # 정규화 역변환 (근사)
        nd = lcmv_null_depth(R_denorm, look_deg, est_jammer)
        null_depth_list.append(nd)

    return {
        "music_angle_mae_deg": float(np.mean(mae_list)),
        "music_within2deg_acc": float(np.mean(within2_list)),
        "music_lcmv_null_depth_db_mean": float(np.mean(null_depth_list)),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model: nn.Module, split: str = "test",
                   device: str = "cpu") -> dict:
    """모델 성능 평가.

    Metrics:
    - angle MAE (deg): |jammer_true - jammer_pred| 평균
    - accuracy within ±2 deg
    - LCMV null depth using predicted angle
    """
    model.eval()
    dataset = JammerDataset(DATA_DIR / f"{split}.h5")
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    all_pred_deg = []
    all_true_deg = []
    null_depths = []

    raw_data = load_hdf5(DATA_DIR / f"{split}.h5",
                         ["cov", "look_angle_deg", "jammer_angle_deg"])

    idx = 0
    for cov, look, sin_target in loader:
        cov, look = cov.to(device), look.to(device)
        sin_pred = model(cov, look).cpu().numpy()  # (B, 1)
        # sin → angle (degrees)
        pred_deg = np.degrees(np.arcsin(np.clip(sin_pred, -1.0, 1.0))).squeeze(1)

        B = cov.shape[0]
        true_deg = raw_data["jammer_angle_deg"][idx:idx + B]
        look_deg_np = raw_data["look_angle_deg"][idx:idx + B]
        cov_np = raw_data["cov"][idx:idx + B]

        all_pred_deg.extend(pred_deg.tolist())
        all_true_deg.extend(true_deg.tolist())

        # LCMV null depth (일부 샘플만 계산, 속도를 위해)
        for b in range(min(B, 50)):  # 배치당 최대 50개
            R = cov_np[b, 0] + 1j * cov_np[b, 1]
            nd = lcmv_null_depth(R, float(look_deg_np[b]), float(pred_deg[b]))
            null_depths.append(nd)

        idx += B

    all_pred_deg = np.array(all_pred_deg)
    all_true_deg = np.array(all_true_deg)
    errors = np.abs(all_true_deg - all_pred_deg)

    return {
        "angle_mae_deg": float(np.mean(errors)),
        "angle_rmse_deg": float(np.sqrt(np.mean(errors ** 2))),
        "within_2deg_acc": float(np.mean(errors <= 2.0)),
        "within_5deg_acc": float(np.mean(errors <= 5.0)),
        "lcmv_null_depth_db_mean": float(np.mean(null_depths)) if null_depths else float("nan"),
    }


# ---------------------------------------------------------------------------
# 직접 학습 루프 (training_loop 대신 — 두 입력 필요)
# ---------------------------------------------------------------------------

def train_epoch(model: nn.Module, loader: DataLoader,
                criterion: nn.Module, optimizer: torch.optim.Optimizer,
                device: str) -> float:
    model.train()
    total_loss = 0.0
    for cov, look, y in loader:
        cov, look, y = cov.to(device), look.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(cov, look)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * cov.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def val_epoch(model: nn.Module, loader: DataLoader,
              criterion: nn.Module, device: str) -> float:
    model.eval()
    total_loss = 0.0
    for cov, look, y in loader:
        cov, look, y = cov.to(device), look.to(device), y.to(device)
        pred = model(cov, look)
        loss = criterion(pred, y)
        total_loss += loss.item() * cov.size(0)
    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = base_parser("P08: Jammer Null Steering")
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
    train_ds = JammerDataset(DATA_DIR / "train.h5")
    val_ds = JammerDataset(DATA_DIR / "val.h5")
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

    epochs = 2 if args.smoke else args.epochs
    batch_size = args.batch_size
    device = "cpu"

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

    # 3. 모델
    model = CovNet().to(device)
    print(f"\nModel parameters: {count_parameters(model):,}")

    if args.eval_only:
        if args.checkpoint is None:
            args.checkpoint = str(ARTIFACT_DIR / "best_model.pt")
        print(f"Loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        # 4. 학습
        criterion = nn.SmoothL1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        import time
        best_val = float("inf")
        history = {"train_loss": [], "val_loss": [], "epoch_time": []}

        print(f"\nTraining for {epochs} epochs on {device}...")
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            va_loss = val_epoch(model, val_loader, criterion, device)
            dt = time.time() - t0

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            history["epoch_time"].append(dt)

            scheduler.step(va_loss)

            improved = ""
            if va_loss < best_val:
                best_val = va_loss
                torch.save(model.state_dict(), ARTIFACT_DIR / "best_model.pt")
                improved = " *"

            if epoch <= 3 or epoch % 5 == 0 or epoch == epochs:
                print(f"  Epoch {epoch:3d}/{epochs}  train={tr_loss:.5f}"
                      f"  val={va_loss:.5f}  {dt:.1f}s{improved}")

        with open(ARTIFACT_DIR / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        total_time = sum(history["epoch_time"])
        print(f"  Done. Best val loss: {best_val:.5f}  Total: {total_time:.0f}s")

        # 최적 모델 로드
        model.load_state_dict(
            torch.load(ARTIFACT_DIR / "best_model.pt", map_location=device)
        )

    # 5. 평가
    print("\nEvaluating model on test set...")
    test_metrics = evaluate_model(model, "test", device)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # MUSIC + LCMV 기준선
    print("\nEvaluating MUSIC+LCMV baseline (subset)...")
    baseline_metrics = music_lcmv_baseline("test", n_eval=300)
    for k, v in baseline_metrics.items():
        print(f"  {k}: {v:.4f}")

    # 6. 메트릭 저장
    all_metrics = {"model": test_metrics, "baseline_music_lcmv": baseline_metrics}
    metrics_path = ARTIFACT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
