#!/usr/bin/env python3
"""P06 — Generate lightweight target-signature classification datasets."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.cli import base_parser
from common.hdf5_io import save_hdf5
from common.seed import seed_everything
from shared.target_signature import (
    SCHEMA_VERSION,
    TARGET_CLASSES,
    TargetSignatureConfig,
    class_name_bytes,
    generate_target_signature_sample,
)

MATLAB_REFERENCE_URL = "https://www.mathworks.com/help/radar/ug/radar-target-classification-using-machine-learning-and-deep-learning.html"


def _generate_split(path: Path, n_samples: int, seed: int, config: TargetSignatureConfig) -> None:
    rng = np.random.default_rng(seed)
    dummy_x, _, dummy_feat, _ = generate_target_signature_sample(TARGET_CLASSES[0], rng, config, snr_db=12.0)
    channels, length = dummy_x.shape
    n_features = int(dummy_feat.size)
    n_classes = len(TARGET_CLASSES)
    n_per_class = max(1, n_samples // n_classes) if n_samples > 0 else 0
    n_total = n_per_class * n_classes

    x_all = np.empty((n_total, channels, length), dtype=np.float32)
    y_all = np.empty(n_total, dtype=np.int64)
    features_all = np.empty((n_total, n_features), dtype=np.float32)
    snr_all = np.empty(n_total, dtype=np.float32)
    center_aspect_all = np.empty(n_total, dtype=np.float32)
    aspect_min_all = np.empty(n_total, dtype=np.float32)
    aspect_max_all = np.empty(n_total, dtype=np.float32)
    vibration_all = np.empty(n_total, dtype=np.float32)
    n_scatterers_all = np.empty(n_total, dtype=np.int16)

    t0 = time.time()
    idx = 0
    progress_every = max(100, min(1000, max(n_total, 1) // 5))
    for class_name in TARGET_CLASSES:
        for _ in range(n_per_class):
            x, label, features, meta = generate_target_signature_sample(class_name, rng, config)
            x_all[idx] = x
            y_all[idx] = int(label)
            features_all[idx] = features
            snr_all[idx] = float(meta["snr_db"])
            center_aspect_all[idx] = float(meta["center_aspect_deg"])
            aspect_min_all[idx] = float(meta["aspect_min_deg"])
            aspect_max_all[idx] = float(meta["aspect_max_deg"])
            vibration_all[idx] = float(meta["vibration_deg"])
            n_scatterers_all[idx] = int(meta["n_scatterers"])
            idx += 1
            if idx % progress_every == 0:
                elapsed = time.time() - t0
                rate = idx / max(elapsed, 1e-9)
                print(f"    [{idx:>6d}/{n_total}]  {rate:.0f} samples/s")

    order = rng.permutation(n_total)
    save_hdf5(
        path,
        x=x_all[order],
        y=y_all[order],
        features=features_all[order],
        snr_db=snr_all[order],
        center_aspect_deg=center_aspect_all[order],
        aspect_min_deg=aspect_min_all[order],
        aspect_max_deg=aspect_max_all[order],
        vibration_deg=vibration_all[order],
        n_scatterers=n_scatterers_all[order],
        target_class_names=class_name_bytes(),
        target_signature_channels=np.array([b"magnitude_db_norm", b"phase_unwrapped_norm"], dtype="S32"),
        fc_hz=np.array([config.fc_hz], dtype=np.float64),
        wavelength_m=np.array([config.wavelength_m], dtype=np.float64),
        n_samples=np.array([config.n_samples], dtype=np.int32),
        aspect_range_deg=np.array(config.aspect_range_deg, dtype=np.float32),
        snr_range_db=np.array(config.snr_range_db, dtype=np.float32),
        vibration_range_deg=np.array(config.vibration_range_deg, dtype=np.float32),
        aspect_jitter_std_deg=np.array([config.aspect_jitter_std_deg], dtype=np.float32),
        matlab_reference_url=np.array([MATLAB_REFERENCE_URL.encode("utf-8")], dtype="S160"),
        representation=np.array([b"aspect_varying_complex_signature_mag_phase"], dtype="S80"),
        simulator_scope=np.array([b"lightweight_point_scatterer_teaching_model"], dtype="S80"),
        schema_version=np.array([SCHEMA_VERSION], dtype=np.int32),
    )


def main() -> None:
    p = base_parser("Generate P06 lightweight target-signature datasets")
    p.add_argument("--n_train", type=int, default=3000)
    p.add_argument("--n_val", type=int, default=600)
    p.add_argument("--n_test", type=int, default=600)
    p.add_argument("--snr_lo", type=float, default=6.0)
    p.add_argument("--snr_hi", type=float, default=24.0)
    p.add_argument("--aspect_lo", type=float, default=-45.0)
    p.add_argument("--aspect_hi", type=float, default=45.0)
    p.add_argument("--out_dir", type=str, default=None)
    args = p.parse_args()

    if args.smoke:
        args.n_train, args.n_val, args.n_test = 240, 60, 60

    seed_everything(args.seed)
    config = TargetSignatureConfig(
        aspect_range_deg=(float(args.aspect_lo), float(args.aspect_hi)),
        snr_range_db=(float(args.snr_lo), float(args.snr_hi)),
    )
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== P06: Generate Target Signature Classification Data ===")
    print(f"  Classes ({len(TARGET_CLASSES)}): {list(TARGET_CLASSES)}")
    print(f"  Aspect range: [{args.aspect_lo}, {args.aspect_hi}] deg")
    print(f"  SNR range: [{args.snr_lo}, {args.snr_hi}] dB")
    print("  Representation: 2-channel magnitude/phase signature")
    print("  MATLAB reference: Radar Target Classification Using Machine Learning and Deep Learning")

    for name, n, seed in [
        ("signature_train.h5", args.n_train, args.seed),
        ("signature_val.h5", args.n_val, args.seed + 1000),
        ("signature_test.h5", args.n_test, args.seed + 2000),
    ]:
        print(f"\n  {name} ({n} requested samples)...")
        _generate_split(out_dir / name, n, seed, config)

    print(f"\nDone. Data saved to {out_dir}")


if __name__ == "__main__":
    main()
