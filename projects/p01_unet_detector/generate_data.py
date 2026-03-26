#!/usr/bin/env python3
"""P01 — Generate RDM detection HDF5 datasets.

Usage:
    python generate_data.py                          # default: 50K/5K/5K
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
from shared.fmcw_simulator import FMCWRadar
from shared.clutter_model import generate_random_scene


RADAR = FMCWRadar(fc=77e9, bw=1e9, T_chirp=50e-6, N_chirps=128, fs=10e6)
N_CHANNELS = 2  # mag + phase


def _generate_split(path: Path, n_samples: int, seed: int) -> None:
    """Generate one HDF5 split file."""
    radar = RADAR
    rng = np.random.default_rng(seed)
    Nc = radar.N_chirps
    Nr = radar.N_samples // 2

    rdm_all = np.empty((n_samples, N_CHANNELS, Nc, Nr), dtype=np.float32)
    mask_all = np.empty((n_samples, Nc, Nr), dtype=np.float32)
    snr_all = np.empty(n_samples, dtype=np.float32)
    ntgt_all = np.empty(n_samples, dtype=np.int32)

    t0 = time.time()
    for i in range(n_samples):
        rdm_input, target_mask, meta = generate_random_scene(radar, rng)
        rdm_all[i] = rdm_input[:N_CHANNELS]
        mask_all[i] = target_mask
        snr_all[i] = meta['snr_db']
        ntgt_all[i] = meta['n_targets']

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_samples - i - 1) / rate
            print(f"    [{i+1:>7d}/{n_samples}]  {rate:.0f} samples/s  ETA {eta:.0f}s")

    # mask stored as (N, 1, Nc, Nr) to match HDF5Dataset x_key/y_key convention
    save_hdf5(
        path,
        x=rdm_all,
        y=mask_all[:, np.newaxis, :, :],
        snr_db=snr_all,
        n_targets=ntgt_all,
    )


def main():
    p = base_parser("Generate P01 RDM detection datasets")
    p.add_argument("--n_train", type=int, default=50000)
    p.add_argument("--n_val",   type=int, default=5000)
    p.add_argument("--n_test",  type=int, default=5000)
    p.add_argument("--out_dir", type=str, default=None)
    args = p.parse_args()

    if args.smoke:
        args.n_train, args.n_val, args.n_test = 256, 64, 64

    seed_everything(args.seed)

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    radar = RADAR
    print("=== P01: Generate RDM Detection Datasets ===")
    print(f"  Radar: Nc={radar.N_chirps}, Nr={radar.N_samples // 2}")
    print(f"  range_res={radar.range_res:.3f} m, vel_res={radar.vel_res:.3f} m/s")
    print(f"  Channels: {N_CHANNELS}")

    for name, n, seed in [
        ("det_train.h5", args.n_train, args.seed),
        ("det_val.h5",   args.n_val,   args.seed + 1000),
        ("det_test.h5",  args.n_test,  args.seed + 2000),
    ]:
        print(f"\n  {name} ({n} samples)...")
        _generate_split(out_dir / name, n, seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
