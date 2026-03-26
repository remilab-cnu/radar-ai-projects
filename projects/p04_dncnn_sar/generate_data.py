#!/usr/bin/env python3
"""P04 -- Generate SAR despeckling HDF5 datasets.

Usage:
    python generate_data.py                          # default: 25K/5K/5K
    python generate_data.py --smoke                  # 256/64/64 samples
    python generate_data.py --n_train 10000
    python generate_data.py --out_dir custom/path
"""

import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.cli import base_parser
from common.seed import seed_everything
from shared.sar_simulator import StripmapSAR, generate_sar_image, add_speckle, sar_to_db


def _generate_random_scene(sar, rng, n_targets_range=(10, 100),
                            rcs_range=(0.1, 50.0), snr_db_range=(15.0, 30.0)):
    """Generate a random point-target SAR scene. Returns (complex_image, n_targets)."""
    n_targets = int(rng.integers(n_targets_range[0], n_targets_range[1] + 1))
    max_x = sar.az_extent * 0.4
    max_r = sar.rg_extent * 0.4
    targets = [
        {
            'x': float(rng.uniform(-max_x, max_x)),
            'r': float(rng.uniform(-max_r, max_r)),
            'rcs': float(rng.uniform(rcs_range[0], rcs_range[1])),
        }
        for _ in range(n_targets)
    ]
    snr_db = float(rng.uniform(snr_db_range[0], snr_db_range[1]))
    image = generate_sar_image(sar, targets, snr_db=snr_db, rng=rng)
    return image, n_targets


def _generate_split(path: Path, n_samples: int, seed: int,
                    image_size: int = 256,
                    n_targets_range: tuple = (10, 100),
                    rcs_range: tuple = (0.1, 50.0),
                    snr_db_range: tuple = (15.0, 30.0),
                    looks_choices: list = None) -> None:
    """Generate one HDF5 split with noisy/clean SAR image pairs."""
    if looks_choices is None:
        looks_choices = [1, 2, 3, 4, 5]

    rng = np.random.default_rng(seed)
    sar = StripmapSAR(N_az=image_size, N_rg=image_size)
    H, W = image_size, image_size
    chunk = min(256, n_samples)

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as f:
        ds_noisy = f.create_dataset('noisy', shape=(n_samples, 1, H, W),
                                    dtype='float32', chunks=(chunk, 1, H, W),
                                    compression='gzip', compression_opts=4)
        ds_clean = f.create_dataset('clean', shape=(n_samples, 1, H, W),
                                    dtype='float32', chunks=(chunk, 1, H, W),
                                    compression='gzip', compression_opts=4)
        ds_looks = f.create_dataset('n_looks',   shape=(n_samples,), dtype='int32')
        ds_ntgt  = f.create_dataset('n_targets', shape=(n_samples,), dtype='int32')

        f.attrs['image_size'] = image_size
        f.attrs['seed']       = seed
        f.attrs['n_samples']  = n_samples

        t0 = time.time()
        report_interval = max(1, n_samples // 10)

        for i in range(n_samples):
            clean_complex, n_tgt = _generate_random_scene(
                sar, rng,
                n_targets_range=n_targets_range,
                rcs_range=rcs_range,
                snr_db_range=snr_db_range,
            )

            ref_max = np.abs(clean_complex).max()
            clean_db = sar_to_db(clean_complex, ref_max=ref_max)

            L = int(rng.choice(looks_choices))
            speckled_complex = add_speckle(clean_complex, n_looks=L, rng=rng)
            noisy_db = sar_to_db(speckled_complex, ref_max=ref_max)

            lo, hi = clean_db.min(), clean_db.max()
            if hi - lo > 1e-6:
                clean_db = (clean_db - lo) / (hi - lo)
                noisy_db = np.clip((noisy_db - lo) / (hi - lo), 0.0, 1.0)

            ds_noisy[i] = noisy_db[np.newaxis, :, :]
            ds_clean[i] = clean_db[np.newaxis, :, :]
            ds_looks[i] = L
            ds_ntgt[i]  = n_tgt

            if (i + 1) % report_interval == 0 or (i + 1) == n_samples:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n_samples - i - 1) / rate
                pct = (i + 1) / n_samples * 100
                print(f"    [{i+1:>7d}/{n_samples}] {pct:5.1f}%  "
                      f"{rate:.0f} samples/s  ETA {eta:.0f}s")

    elapsed = time.time() - t0
    size_mb = path.stat().st_size / (1024 ** 2)
    print(f"  Saved {path.name}  ({n_samples} samples, {size_mb:.1f} MB, {elapsed:.1f}s)")


def main():
    p = base_parser("Generate P04 SAR despeckling datasets")
    p.add_argument("--n_train",    type=int, default=25000)
    p.add_argument("--n_val",      type=int, default=5000)
    p.add_argument("--n_test",     type=int, default=5000)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--out_dir",    type=str, default=None)
    args = p.parse_args()

    if args.smoke:
        args.n_train, args.n_val, args.n_test = 256, 64, 64

    seed_everything(args.seed)

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_targets_range = (10, 100)
    rcs_range       = (0.1, 50.0)
    snr_db_range    = (15.0, 30.0)
    looks_choices   = [1, 2, 3, 4, 5]

    print("=== P04: Generate SAR Despeckling Datasets ===")
    print(f"  image_size   = {args.image_size}x{args.image_size}")
    print(f"  n_targets    = {n_targets_range}")
    print(f"  rcs_range    = {rcs_range} m^2")
    print(f"  snr_db_range = {snr_db_range} dB")
    print(f"  looks        = {looks_choices}")

    for name, n, seed in [
        ("despeckling_train.h5", args.n_train, args.seed),
        ("despeckling_val.h5",   args.n_val,   args.seed + 1000),
        ("despeckling_test.h5",  args.n_test,  args.seed + 2000),
    ]:
        print(f"\n  {name} ({n} samples)...")
        _generate_split(
            out_dir / name, n, seed,
            image_size=args.image_size,
            n_targets_range=n_targets_range,
            rcs_range=rcs_range,
            snr_db_range=snr_db_range,
            looks_choices=looks_choices,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
