"""P06 I/Q Imbalance Correction — 학습 및 평가

Loss = SmoothL1Loss(params) + λ * L1Loss(corrected_signal, clean_signal)

Usage:
  python train.py --generate --smoke
  python train.py --generate --epochs 30
  python train.py --eval_only --checkpoint artifacts/best_model.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common.cli import base_parser
from common.hdf5_io import load_hdf5
from common.train_utils import count_parameters
from common.metrics import regression_report, nmse
from common.seed import seed_everything

BASE = Path(__file__).parent

# Loss weight for signal reconstruction term
LAMBDA_SIGNAL = 0.5

# Parameter normalization scales (for balanced loss)
# [gain_db, phase_deg, dc_i, dc_q]
PARAM_SCALE = torch.tensor([3.0, 15.0, 0.05, 0.05], dtype=torch.float32)


def apply_iq_correction(
    x_corrupt: torch.Tensor,  # (B, 2, N)
    params: torch.Tensor,     # (B, 4) — predicted [gain_db, phase_deg, dc_i, dc_q]
) -> torch.Tensor:
    """Analytic inverse I/Q imbalance correction using predicted params.

    Forward model:
      I_out = I + dc_i
      Q_out = g * (I*sin(phi) + Q*cos(phi)) + dc_q

    Inverse (closed-form):
      I_rec = I_out - dc_i
      Q_rec = (Q_out - dc_q - g*sin(phi)*(I_out - dc_i)) / (g*cos(phi))
    """
    gain_db = params[:, 0:1]    # (B, 1)
    phase_deg = params[:, 1:2]  # (B, 1)
    dc_i = params[:, 2:3]       # (B, 1)
    dc_q = params[:, 3:4]       # (B, 1)

    g = torch.pow(torch.tensor(10.0, device=params.device), gain_db / 20.0)  # (B, 1)
    phi = phase_deg * (torch.pi / 180.0)

    I_c = x_corrupt[:, 0, :]   # (B, N)
    Q_c = x_corrupt[:, 1, :]   # (B, N)

    # Inverse correction
    I_rec = I_c - dc_i
    Q_rec = (Q_c - dc_q - g * torch.sin(phi) * I_rec) / (g * torch.cos(phi) + 1e-8)

    return torch.stack([I_rec, Q_rec], dim=1)  # (B, 2, N)


class CombinedLoss(nn.Module):
    """SmoothL1 on normalized params + L1 on corrected signal."""

    def __init__(self, lambda_signal: float = LAMBDA_SIGNAL):
        super().__init__()
        self.lambda_signal = lambda_signal
        self.smooth_l1 = nn.SmoothL1Loss()
        self.l1 = nn.L1Loss()
        self.register_buffer("scale", PARAM_SCALE)

    def forward(
        self,
        pred_params: torch.Tensor,   # (B, 4)
        true_params: torch.Tensor,   # (B, 4)
        x_corrupt: torch.Tensor,     # (B, 2, N)
        y_clean: torch.Tensor,       # (B, 2, N)
    ) -> torch.Tensor:
        # Normalize params by expected scale for balanced gradients
        scale = self.scale.to(pred_params.device)
        loss_params = self.smooth_l1(pred_params / scale, true_params / scale)

        # Signal reconstruction loss
        x_corrected = apply_iq_correction(x_corrupt, pred_params)
        loss_signal = self.l1(x_corrected, y_clean)

        return loss_params + self.lambda_signal * loss_signal


class IQDataset(Dataset):
    """Custom dataset that holds x_corrupt, y_params, y_clean."""

    def __init__(self, x_corrupt: torch.Tensor, y_params: torch.Tensor,
                 y_clean: torch.Tensor):
        self.x_corrupt = x_corrupt
        self.y_params = y_params
        self.y_clean = y_clean

    def __len__(self):
        return len(self.x_corrupt)

    def __getitem__(self, idx):
        return self.x_corrupt[idx], self.y_params[idx], self.y_clean[idx]


def load_split(split: str, device: str = "cpu"):
    path = BASE / "data" / f"{split}.h5"
    data = load_hdf5(path, ["x_corrupt", "y_params", "y_clean", "gain_db",
                             "phase_deg", "snr_db"])
    x = torch.as_tensor(data["x_corrupt"], dtype=torch.float32).to(device)
    yp = torch.as_tensor(data["y_params"], dtype=torch.float32).to(device)
    yc = torch.as_tensor(data["y_clean"], dtype=torch.float32).to(device)
    return x, yp, yc, data["gain_db"], data["phase_deg"], data["snr_db"]


# NOTE: common.train_utils.training_loop은 (x, y) 단일-입력/단일-출력 모델만 지원한다.
# P06의 CombinedLoss는 (pred_params, true_params, x_corrupt, y_clean) 네 인수를 받아야 하므로
# 배치마다 세 텐서를 모두 전달하는 전용 루프를 직접 구현한다.
def train_one_epoch_iq(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x_c, y_p, y_cl in loader:
        x_c, y_p, y_cl = x_c.to(device), y_p.to(device), y_cl.to(device)
        optimizer.zero_grad()
        pred = model(x_c)
        loss = criterion(pred, y_p, x_c, y_cl)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_c.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate_iq(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for x_c, y_p, y_cl in loader:
        x_c, y_p, y_cl = x_c.to(device), y_p.to(device), y_cl.to(device)
        pred = model(x_c)
        loss = criterion(pred, y_p, x_c, y_cl)
        total_loss += loss.item() * x_c.size(0)
    return total_loss / len(loader.dataset)


def compute_image_rejection_ratio(
    x_corrupt_np: np.ndarray,  # (N, 2, 512)
    x_corrected_np: np.ndarray,
    x_clean_np: np.ndarray,
) -> tuple[float, float]:
    """Image Rejection Ratio (IRR) improvement.

    IRR = power of desired signal / power of image component.
    Computed via FFT: image is the spectral mirror of the desired signal.
    Returns (irr_corrupt_db, irr_corrected_db).
    """
    def _irr(iq):
        # iq: (N, 2, N) → complex
        c = iq[:, 0, :] + 1j * iq[:, 1, :]  # (N, L)
        spec = np.fft.fft(c, axis=-1)
        L = spec.shape[-1]
        half = L // 2
        # Desired: positive freq, Image: negative freq (mirror)
        desired = np.mean(np.abs(spec[:, 1:half]) ** 2, axis=-1)
        image = np.mean(np.abs(spec[:, half + 1:]) ** 2, axis=-1)
        irr = 10 * np.log10(desired / (image + 1e-30) + 1e-30)
        return float(np.mean(irr))

    irr_corrupt = _irr(x_corrupt_np)
    irr_corrected = _irr(x_corrected_np)
    return irr_corrupt, irr_corrected


def gram_schmidt_correction(x_corrupt_np: np.ndarray) -> np.ndarray:
    """Blind Gram-Schmidt orthogonalization baseline.

    Corrects phase and gain imbalance without knowing true parameters.
    I_out = I / ||I||
    Q_out = (Q - <Q,I_norm> * I_norm) normalized
    """
    I = x_corrupt_np[:, 0, :].copy()   # (N, L)
    Q = x_corrupt_np[:, 1, :].copy()

    # Normalize I
    I_norm = I / (np.sqrt(np.mean(I ** 2, axis=-1, keepdims=True)) + 1e-30)

    # Remove I component from Q (orthogonalization)
    dot = np.sum(Q * I_norm, axis=-1, keepdims=True) / I.shape[-1]
    Q_orth = Q - dot * I_norm

    # Normalize Q
    Q_norm = Q_orth / (np.sqrt(np.mean(Q_orth ** 2, axis=-1, keepdims=True)) + 1e-30)

    # Scale back to original I magnitude
    scale = np.sqrt(np.mean(I ** 2, axis=-1, keepdims=True)) + 1e-30
    I_out = I_norm * scale
    Q_out = Q_norm * scale

    return np.stack([I_out, Q_out], axis=1)


def evaluate(
    model: nn.Module,
    x_test: torch.Tensor,
    y_params: torch.Tensor,
    y_clean: torch.Tensor,
    gain_np: np.ndarray,
    phase_np: np.ndarray,
    device: str = "cpu",
) -> dict:
    """전체 평가 지표 계산."""
    model.eval()
    with torch.no_grad():
        pred_params = model(x_test.to(device)).cpu()
        x_corrected = apply_iq_correction(x_test.cpu(), pred_params)

    pred_np = pred_params.numpy()   # (N, 4)
    true_np = y_params.cpu().numpy()
    x_corr_np = x_corrected.numpy()
    x_clean_np = y_clean.cpu().numpy()
    x_corrupt_np = x_test.cpu().numpy()

    # Parameter MAE
    gain_mae = float(np.mean(np.abs(pred_np[:, 0] - true_np[:, 0])))
    phase_mae = float(np.mean(np.abs(pred_np[:, 1] - true_np[:, 1])))
    dc_i_mae = float(np.mean(np.abs(pred_np[:, 2] - true_np[:, 2])))
    dc_q_mae = float(np.mean(np.abs(pred_np[:, 3] - true_np[:, 3])))

    # Signal NMSE
    nmse_corrupt = nmse(x_clean_np, x_corrupt_np)
    nmse_corrected = nmse(x_clean_np, x_corr_np)

    # IRR
    irr_corrupt, irr_corrected = compute_image_rejection_ratio(
        x_corrupt_np, x_corr_np, x_clean_np
    )

    return {
        "gain_mae_db": gain_mae,
        "phase_mae_deg": phase_mae,
        "dc_i_mae": dc_i_mae,
        "dc_q_mae": dc_q_mae,
        "nmse_corrupt": float(nmse_corrupt),
        "nmse_corrected": float(nmse_corrected),
        "nmse_improvement_db": float(10 * np.log10((nmse_corrupt + 1e-30) / (nmse_corrected + 1e-30))),
        "irr_corrupt_db": irr_corrupt,
        "irr_corrected_db": irr_corrected,
        "irr_improvement_db": irr_corrected - irr_corrupt,
    }


def evaluate_baseline(x_test_np: np.ndarray, y_clean_np: np.ndarray) -> dict:
    """Gram-Schmidt baseline 평가."""
    x_gs = gram_schmidt_correction(x_test_np)
    nmse_corrupt = nmse(y_clean_np, x_test_np)
    nmse_gs = nmse(y_clean_np, x_gs)
    irr_corrupt, irr_gs = compute_image_rejection_ratio(x_test_np, x_gs, y_clean_np)
    return {
        "nmse_corrupt": float(nmse_corrupt),
        "nmse_corrected": float(nmse_gs),
        "nmse_improvement_db": float(10 * np.log10((nmse_corrupt + 1e-30) / (nmse_gs + 1e-30))),
        "irr_corrupt_db": irr_corrupt,
        "irr_corrected_db": irr_gs,
        "irr_improvement_db": irr_gs - irr_corrupt,
    }


def main():
    parser = base_parser("P06 I/Q Imbalance Correction — 학습 및 평가")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = "cpu"

    artifact_dir = BASE / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # --- 데이터 생성 ---
    if args.generate:
        import subprocess
        gen_cmd = [sys.executable, str(BASE / "generate_data.py")]
        if args.smoke:
            gen_cmd.append("--smoke")
        gen_cmd += ["--seed", str(args.seed)]
        subprocess.run(gen_cmd, check=True)

    # --- 모델 ---
    from model import build_model
    model = build_model().to(device)
    print(f"\n[Model] IQImbalanceCNN  params={count_parameters(model):,}")

    criterion = CombinedLoss(lambda_signal=LAMBDA_SIGNAL)

    if not args.eval_only:
        x_tr, yp_tr, yc_tr, _, _, _ = load_split("train", device)
        x_val, yp_val, yc_val, _, _, _ = load_split("val", device)

        epochs = 2 if args.smoke else args.epochs

        train_ds = IQDataset(x_tr, yp_tr, yc_tr)
        val_ds = IQDataset(x_val, yp_val, yc_val)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        # Custom training loop (needs 3 tensors per batch)
        import time
        history = {"train_loss": [], "val_loss": [], "epoch_time": []}
        best_val = float("inf")

        print(f"\n[Train] {epochs} epochs on {device}...")
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr_loss = train_one_epoch_iq(model, train_loader, criterion, optimizer, device)
            vl_loss = validate_iq(model, val_loader, criterion, device)
            dt = time.time() - t0

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(vl_loss)
            history["epoch_time"].append(dt)
            scheduler.step(vl_loss)

            if vl_loss < best_val:
                best_val = vl_loss
                torch.save(model.state_dict(), artifact_dir / "best_model.pt")
                star = " *"
            else:
                star = ""

            if epoch <= 3 or epoch % 5 == 0 or epoch == epochs:
                print(f"  Epoch {epoch:3d}/{epochs}  train={tr_loss:.4f}  val={vl_loss:.4f}"
                      f"  {dt:.1f}s{star}")

        with open(artifact_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"  Done. Best val loss: {best_val:.4f}")

    # --- 체크포인트 로드 ---
    ckpt = args.checkpoint or str(artifact_dir / "best_model.pt")
    if Path(ckpt).exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"\n[Eval] Loaded checkpoint: {ckpt}")
    else:
        print(f"\n[Eval] No checkpoint at {ckpt}, using current weights.")

    # --- 평가 ---
    x_test, yp_test, yc_test, gain_np, phase_np, snr_np = load_split("test", device)

    print("\n[Eval] IQImbalanceCNN on test set...")
    metrics = evaluate(model, x_test, yp_test, yc_test, gain_np, phase_np, device)

    print(f"  Gain MAE:           {metrics['gain_mae_db']:.4f} dB")
    print(f"  Phase MAE:          {metrics['phase_mae_deg']:.4f} deg")
    print(f"  DC-I MAE:           {metrics['dc_i_mae']:.6f}")
    print(f"  DC-Q MAE:           {metrics['dc_q_mae']:.6f}")
    print(f"  Signal NMSE (corrupt):    {metrics['nmse_corrupt']:.4f}")
    print(f"  Signal NMSE (corrected):  {metrics['nmse_corrected']:.4f}")
    print(f"  NMSE improvement:   {metrics['nmse_improvement_db']:.2f} dB")
    print(f"  IRR (corrupt):      {metrics['irr_corrupt_db']:.2f} dB")
    print(f"  IRR (corrected):    {metrics['irr_corrected_db']:.2f} dB")
    print(f"  IRR improvement:    {metrics['irr_improvement_db']:.2f} dB")

    # --- Gram-Schmidt baseline ---
    print("\n[Baseline] Gram-Schmidt orthogonalization...")
    gs_metrics = evaluate_baseline(x_test.cpu().numpy(), yc_test.cpu().numpy())
    print(f"  Signal NMSE (corrupt):    {gs_metrics['nmse_corrupt']:.4f}")
    print(f"  Signal NMSE (corrected):  {gs_metrics['nmse_corrected']:.4f}")
    print(f"  NMSE improvement:   {gs_metrics['nmse_improvement_db']:.2f} dB")
    print(f"  IRR (corrupt):      {gs_metrics['irr_corrupt_db']:.2f} dB")
    print(f"  IRR (corrected):    {gs_metrics['irr_corrected_db']:.2f} dB")
    print(f"  IRR improvement:    {gs_metrics['irr_improvement_db']:.2f} dB")

    # --- 저장 ---
    all_metrics = {
        "iq_imbalance_cnn": metrics,
        "gram_schmidt_baseline": gs_metrics,
    }
    metrics_path = artifact_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[Saved] {metrics_path}")


if __name__ == "__main__":
    main()
