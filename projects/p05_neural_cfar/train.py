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
from cfar_utils import evaluate_patch_ca_cfar

BASE = Path(__file__).parent


def load_dataset(split: str, device: str = "cpu", include_cfar: bool = False):
    keys = ["x", "y", "snr_db"]
    if include_cfar:
        keys += ["patch_power", "cut_range_bin", "cut_doppler_bin", "target_distance_bins", "clutter_type"]
    try:
        data = load_hdf5(BASE / "data" / f"{split}.h5", keys)
    except KeyError as exc:
        raise RuntimeError(
            "P05 dataset is missing linear-power CFAR metadata. "
            "Regenerate it with `python train.py --generate --smoke` or `python generate_data.py`."
        ) from exc

    x = torch.as_tensor(data["x"], dtype=torch.float32).to(device)
    y = torch.as_tensor(data["y"], dtype=torch.float32).to(device)
    snr = data["snr_db"]
    if not include_cfar:
        return x, y, snr
    cfar_meta = {
        "patch_power": data["patch_power"],
        "cut_range_bin": data["cut_range_bin"],
        "cut_doppler_bin": data["cut_doppler_bin"],
        "target_distance_bins": data["target_distance_bins"],
        "clutter_type": data["clutter_type"],
    }
    return x, y, snr, cfar_meta


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


def cfar_baseline(patch_power: np.ndarray, y_np: np.ndarray, pfa: float = 1e-2) -> dict[str, float]:
    """Patch CA-CFAR baseline on native linear RDM power cells."""
    return evaluate_patch_ca_cfar(patch_power, y_np, pfa=pfa)


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
    x_test, y_test, snr_test, cfar_meta = load_dataset("test", include_cfar=True)
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
    y_np = y_test.cpu().numpy()
    patch_power = cfar_meta["patch_power"]
    print("\n[Baseline] Patch CA-CFAR on linear RDM power...")
    cfar_metrics = {
        "pfa_1e2": cfar_baseline(patch_power, y_np, pfa=1e-2),
        "pfa_1e3": cfar_baseline(patch_power, y_np, pfa=1e-3),
    }
    for label, vals in cfar_metrics.items():
        print(f"  {label} target Pfa={vals['target_pfa']:.0e}")
        print(f"    Pd:               {vals['pd']:.4f}")
        print(f"    Pfa (empirical):  {vals['pfa']:.4f}")
        print(f"    Balanced Acc.:    {vals['balanced_accuracy']:.4f}")

    # --- 저장 ---
    all_metrics = {
        "neural_cfar": metrics,
        "patch_ca_cfar_linear_power": cfar_metrics,
    }
    metrics_path = artifact_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[Saved] {metrics_path}")


if __name__ == "__main__":
    main()
