#!/usr/bin/env python3
"""P03 -- Generate DoA HDF5 datasets for DeepMUSIC training.

Usage:
    python generate_data.py                          # default: 100K/20K/20K
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
from shared.doa_utils import generate_doa_sample


def _generate_split(path: Path, n_samples: int, seed: int,
                    n_rx: int = 12, grid_size: int = 181,
                    coherent_prob: float = 0.2,
                    n_sources_range: tuple = (1, 3),
                    snr_range: tuple = (0.0, 20.0),
                    n_snapshots_range: tuple = (10, 200)) -> None:
    """Generate one HDF5 split with covariance matrices and spectra."""
    rng = np.random.default_rng(seed)

    gen_kwargs = dict(
        grid_size=grid_size,
        coherent_prob=coherent_prob,
        moving_prob=0.0,
        max_drift_deg=5.0,
        n_sources_range=n_sources_range,
        snr_range=snr_range,
        n_snapshots_range=n_snapshots_range,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as f:
        ds_cov = f.create_dataset(
            'covariance', shape=(n_samples, 2, n_rx, n_rx),
            dtype='float32', chunks=(min(512, n_samples), 2, n_rx, n_rx),
            compression='gzip', compression_opts=4,
        )
        ds_spec = f.create_dataset(
            'spectrum', shape=(n_samples, grid_size),
            dtype='float32', chunks=(min(512, n_samples), grid_size),
            compression='gzip', compression_opts=4,
        )
        dt_vlen = h5py.vlen_dtype(np.float64)
        ds_angles = f.create_dataset('angles', shape=(n_samples,), dtype=dt_vlen)
        ds_snr    = f.create_dataset('snr_db',      shape=(n_samples,), dtype='float32')
        ds_nsrc   = f.create_dataset('n_sources',   shape=(n_samples,), dtype='int32')
        ds_nsnap  = f.create_dataset('n_snapshots', shape=(n_samples,), dtype='int32')
        ds_coh    = f.create_dataset('coherent',    shape=(n_samples,), dtype='bool')
        ds_mov    = f.create_dataset('moving',      shape=(n_samples,), dtype='bool')

        f.attrs['n_rx']      = n_rx
        f.attrs['grid_size'] = grid_size
        f.attrs['seed']      = seed
        f.attrs['n_samples'] = n_samples

        t0 = time.time()
        for i in range(n_samples):
            cov, spec, angles, meta = generate_doa_sample(N_rx=n_rx, rng=rng, **gen_kwargs)
            ds_cov[i]    = cov
            ds_spec[i]   = spec
            ds_angles[i] = np.array(angles)
            ds_snr[i]    = meta['snr_db']
            ds_nsrc[i]   = meta['n_sources']
            ds_nsnap[i]  = meta['n_snapshots']
            ds_coh[i]    = meta['coherent']
            ds_mov[i]    = meta.get('moving', False)

            if (i + 1) % 10000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n_samples - i - 1) / rate
                print(f"    [{i+1:>7d}/{n_samples}]  {rate:.0f} samples/s  ETA {eta:.0f}s")

    elapsed = time.time() - t0
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  Saved {path.name}  ({n_samples} samples, {size_mb:.1f} MB, {elapsed:.1f}s)")


def main():
    p = base_parser("Generate P03 DoA datasets")
    p.add_argument("--n_train",         type=int,   default=100000)
    p.add_argument("--n_val",           type=int,   default=20000)
    p.add_argument("--n_test",          type=int,   default=20000)
    p.add_argument("--n_rx",            type=int,   default=12)
    p.add_argument("--grid_size",       type=int,   default=181)
    p.add_argument("--coherent_prob",   type=float, default=0.2)
    p.add_argument("--max_sources",     type=int,   default=3)
    p.add_argument("--snr_lo",          type=float, default=0.0)
    p.add_argument("--snr_hi",          type=float, default=20.0)
    p.add_argument("--min_snapshots",   type=int,   default=10)
    p.add_argument("--max_snapshots",   type=int,   default=200)
    p.add_argument("--out_dir",         type=str,   default=None)
    args = p.parse_args()

    if args.smoke:
        args.n_train, args.n_val, args.n_test = 256, 64, 64

    seed_everything(args.seed)

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== P03: Generate DoA Datasets ===")
    print(f"  N_rx={args.n_rx}, grid_size={args.grid_size}")
    print(f"  coherent_prob={args.coherent_prob}")
    print(f"  n_sources_range=(1, {args.max_sources})")
    print(f"  snr_range=({args.snr_lo}, {args.snr_hi}) dB")
    print(f"  n_snapshots_range=({args.min_snapshots}, {args.max_snapshots})")

    gen_kwargs = dict(
        n_rx=args.n_rx,
        grid_size=args.grid_size,
        coherent_prob=args.coherent_prob,
        n_sources_range=(1, args.max_sources),
        snr_range=(args.snr_lo, args.snr_hi),
        n_snapshots_range=(args.min_snapshots, args.max_snapshots),
    )

    for name, n, seed in [
        ("doa_train.h5", args.n_train, args.seed),
        ("doa_val.h5",   args.n_val,   args.seed + 1000),
        ("doa_test.h5",  args.n_test,  args.seed + 2000),
    ]:
        print(f"\n  {name} ({n} samples)...")
        _generate_split(out_dir / name, n, seed, **gen_kwargs)

    print("\nDone.")


if __name__ == "__main__":
    main()
