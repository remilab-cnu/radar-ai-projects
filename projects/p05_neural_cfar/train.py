"""P05 Neural CFAR — 학습 및 평가

Usage:
  python train.py --generate --smoke          # 데이터 생성 + smoke test
  python train.py --generate --epochs 30      # 전체 학습
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
from torch.utils.data import DataLoader, TensorDataset

from common.cli import base_parser
from common.hdf5_io import load_hdf5
from common.train_utils import training_loop, count_parameters
from common.metrics import pd_at_pfa, classification_report
from common.seed import seed_everything
from shared.fmcw_simulator import ca_cfar_2d, range_doppler_map
from shared.clutter_model import generate_scene_with_clutter

BASE = Path(__file__).parent


def load_dataset(split: str, device: str = "cpu"):
    data = load_hdf5(BASE / "data" / f"{split}.h5", ["x", "y", "snr_db"])
    x = torch.as_tensor(data["x"], dtype=torch.float32).to(device)
    y = torch.as_tensor(data["y"], dtype=torch.float32).to(device)
    snr = data["snr_db"]
    return x, y, snr


def evaluate(model: nn.Module, x: torch.Tensor, y_true: torch.Tensor,
             snr: np.ndarray, device: str = "cpu") -> dict:
    """ROC-AUC, Pd@Pfa, balanced accuracy 계산."""
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device)).squeeze(1)
        y_prob = torch.sigmoid(logits).cpu().numpy()

    y_np = y_true.cpu().numpy()
    y_pred = (y_prob >= 0.5).astype(int)

    base_metrics = classification_report(y_np, y_pred, y_prob)
    metrics = {
        "roc_auc": base_metrics.get("roc_auc", 0.0),
        "balanced_accuracy": base_metrics["balanced_accuracy"],
        "pd_at_pfa_1e2": pd_at_pfa(y_np, y_prob, 1e-2),
        "pd_at_pfa_1e3": pd_at_pfa(y_np, y_prob, 1e-3),
    }

    # Per-SNR Pd@Pfa=1e-2
    snr_bins = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    per_snr = {}
    for snr_val in snr_bins:
        mask = np.isclose(snr, snr_val)
        if mask.sum() == 0:
            continue
        pd = pd_at_pfa(y_np[mask], y_prob[mask], 1e-2)
        per_snr[f"pd_pfa1e2_snr{int(snr_val)}dB"] = float(pd)
    metrics["per_snr"] = per_snr

    return metrics


def cfar_baseline(x_np: np.ndarray, y_np: np.ndarray, pfa: float = 1e-2) -> float:
    """CA-CFAR baseline: 패치 중심 셀에 대해 주변 training cells로 문턱값 결정.

    패치 크기 15x15, guard=1, train=3 (패치 내 적용)
    """
    N = len(x_np)
    # 중심 셀 magnitude (ch0 기준)
    half = 7  # PATCH // 2

    # 패치 내 CA-CFAR: center vs training region
    # guard_ring=1, train_ring=3 → inner 3x3 guard, outer ring for training
    g = 1
    t = 3
    n_train_cells = (2 * (g + t) + 1) ** 2 - (2 * g + 1) ** 2
    alpha = n_train_cells * (pfa ** (-1.0 / n_train_cells) - 1)

    detections = []
    for i in range(N):
        patch = x_np[i, 0]  # (15, 15) ch0
        # power of center
        center_pow = patch[half, half] ** 2

        # training cells mask: exclude guard ring
        mask = np.zeros((15, 15), dtype=bool)
        mask[half - (g + t): half + (g + t) + 1,
             half - (g + t): half + (g + t) + 1] = True
        mask[half - g: half + g + 1,
             half - g: half + g + 1] = False

        noise_est = np.mean(patch[mask] ** 2)
        T = alpha * noise_est
        detections.append(1 if center_pow > T else 0)

    y_pred = np.array(detections)
    tp = np.sum((y_pred == 1) & (y_np == 1))
    fp = np.sum((y_pred == 1) & (y_np == 0))
    fn = np.sum((y_pred == 0) & (y_np == 1))
    tn = np.sum((y_pred == 0) & (y_np == 0))

    pd = tp / (tp + fn + 1e-9)
    pfa_actual = fp / (fp + tn + 1e-9)
    bal_acc = 0.5 * (tp / (tp + fn + 1e-9) + tn / (tn + fp + 1e-9))
    return {"pd": float(pd), "pfa": float(pfa_actual), "balanced_accuracy": float(bal_acc)}


def main():
    parser = base_parser("P05 Neural CFAR — 학습 및 평가")
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
    print(f"\n[Model] NeuralCFAR  params={count_parameters(model):,}")

    if not args.eval_only:
        # --- 학습 데이터 로드 ---
        x_tr, y_tr, _ = load_dataset("train")
        x_val, y_val, _ = load_dataset("val")

        epochs = 2 if args.smoke else args.epochs

        train_ds = TensorDataset(x_tr, y_tr.unsqueeze(1))
        val_ds = TensorDataset(x_val, y_val.unsqueeze(1))
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        print("\n[Train]")
        training_loop(model, train_loader, val_loader, criterion, optimizer,
                      epochs=epochs, checkpoint_dir=artifact_dir,
                      device=device, scheduler=scheduler)

    # --- 체크포인트 로드 ---
    ckpt = args.checkpoint or str(artifact_dir / "best_model.pt")
    if Path(ckpt).exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"\n[Eval] Loaded checkpoint: {ckpt}")
    else:
        print(f"\n[Eval] No checkpoint found at {ckpt}, evaluating current weights.")

    # --- 평가 ---
    x_test, y_test, snr_test = load_dataset("test")
    print("\n[Eval] Neural CFAR on test set...")
    metrics = evaluate(model, x_test, y_test, snr_test, device)

    print(f"  ROC-AUC:            {metrics['roc_auc']:.4f}")
    print(f"  Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
    print(f"  Pd @ Pfa=1e-2:      {metrics['pd_at_pfa_1e2']:.4f}")
    print(f"  Pd @ Pfa=1e-3:      {metrics['pd_at_pfa_1e3']:.4f}")
    print("  Per-SNR Pd@Pfa=1e-2:")
    for k, v in metrics["per_snr"].items():
        print(f"    {k}: {v:.4f}")

    # --- CA-CFAR baseline ---
    x_np = x_test.cpu().numpy()
    y_np = y_test.cpu().numpy()
    print("\n[Baseline] CA-CFAR (patch-level)...")
    cfar_metrics = cfar_baseline(x_np, y_np, pfa=1e-2)
    print(f"  Pd:                 {cfar_metrics['pd']:.4f}")
    print(f"  Pfa (actual):       {cfar_metrics['pfa']:.4f}")
    print(f"  Balanced Accuracy:  {cfar_metrics['balanced_accuracy']:.4f}")

    # --- 저장 ---
    all_metrics = {
        "neural_cfar": metrics,
        "ca_cfar_baseline": cfar_metrics,
    }
    metrics_path = artifact_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[Saved] {metrics_path}")


if __name__ == "__main__":
    main()
