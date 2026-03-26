#!/usr/bin/env python3
"""P02 — Generate micro-Doppler HAR HDF5 datasets.

Usage:
    python generate_data.py                          # default: 30K/3K/3K
    python generate_data.py --smoke                  # 256/64/64 samples
    python generate_data.py --n_train 10000
    python generate_data.py --out_dir custom/path
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.cli import base_parser
from common.hdf5_io import save_hdf5
from common.seed import seed_everything
from shared.micro_doppler import (
    generate_har_sample,
    extract_handcrafted_features,
    ACTIVITY_LABELS,
    N_CLASSES,
)


def _generate_split(path: Path, n_samples: int, seed: int,
                    snr_lo: float = 5.0, snr_hi: float = 25.0) -> None:
    """Generate one balanced HDF5 split file."""
    rng = np.random.default_rng(seed)
    H, W = 128, 128

    dummy_feat = extract_handcrafted_features(np.zeros((H, W), dtype=np.float32))
    n_features = len(dummy_feat)

    n_per_class = n_samples // N_CLASSES
    n_total = n_per_class * N_CLASSES

    spec_all = np.empty((n_total, H, W), dtype=np.float32)
    label_all = np.empty(n_total, dtype=np.int32)
    feat_all = np.empty((n_total, n_features), dtype=np.float32)
    snr_all = np.empty(n_total, dtype=np.float32)

    t0 = time.time()
    idx = 0
    for cls_idx, activity in enumerate(ACTIVITY_LABELS):
        for _ in range(n_per_class):
            snr_db = rng.uniform(snr_lo, snr_hi)
            spec, label, _ = generate_har_sample(activity, rng, snr_db=snr_db, aspect_angle=0.0)
            feat = extract_handcrafted_features(spec)

            spec_all[idx] = spec
            label_all[idx] = label
            feat_all[idx] = feat
            snr_all[idx] = snr_db
            idx += 1

            if idx % 2000 == 0:
                elapsed = time.time() - t0
                rate = idx / elapsed
                eta = (n_total - idx) / rate
                print(f"    [{idx:>7d}/{n_total}]  {rate:.0f} samples/s  ETA {eta:.0f}s")

    # Store spectrogram as (N, 1, H, W) for HDF5Dataset x/y convention
    save_hdf5(
        path,
        x=spec_all[:, np.newaxis, :, :],
        y=label_all,
        features=feat_all,
        snr_db=snr_all,
    )


def main():
    p = base_parser("Generate P02 micro-Doppler HAR datasets")
    p.add_argument("--n_train", type=int, default=30000)
    p.add_argument("--n_val",   type=int, default=3000)
    p.add_argument("--n_test",  type=int, default=3000)
    p.add_argument("--snr_lo",  type=float, default=5.0)
    p.add_argument("--snr_hi",  type=float, default=25.0)
    p.add_argument("--out_dir", type=str, default=None)
    args = p.parse_args()

    if args.smoke:
        args.n_train, args.n_val, args.n_test = 256, 64, 64

    seed_everything(args.seed)

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== P02: Generate Micro-Doppler HAR Datasets ===")
    print(f"  Classes ({N_CLASSES}): {ACTIVITY_LABELS}")
    print(f"  SNR range: [{args.snr_lo}, {args.snr_hi}] dB")

    for name, n, seed in [
        ("har_train.h5", args.n_train, args.seed),
        ("har_val.h5",   args.n_val,   args.seed + 1000),
        ("har_test.h5",  args.n_test,  args.seed + 2000),
    ]:
        print(f"\n  {name} ({n} samples)...")
        _generate_split(out_dir / name, n, seed, args.snr_lo, args.snr_hi)

    print("\nDone.")


if __name__ == "__main__":
    main()
