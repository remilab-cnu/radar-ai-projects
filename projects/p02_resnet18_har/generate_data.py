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
from shared.fmcw_simulator import FMCWRadar, range_axis
from shared.micro_doppler import (
    generate_har_sample,
    extract_handcrafted_features,
    ACTIVITY_LABELS,
    N_CLASSES,
    P02_DOPPLER_ALIAS_SAFETY_FACTOR,
    radar_max_unambiguous_velocity_mps,
)

RADAR = FMCWRadar(
    fc=77.0e9,
    bw=50e6,
    T_chirp=2e-6,
    PRI=100e-6,
    N_chirps=64,
    fs=200e6,  # 4 × signal bandwidth
    N_rx=1,
)
SCHEMA_VERSION = 6
SCATTER_MODEL = "radar_deconv_inspired_pedestrian_scatterers"
ASPECT_ANGLE_RANGE_DEG = (0.0, 60.0)
ASPECT_CONVENTION = "absolute_degrees_symmetric_2d_radial_cosine_projection"
TARGET_RANGE_M_RANGE = (6.0, 18.0)


def _generate_split(path: Path, n_samples: int, seed: int,
                    snr_lo: float = 5.0, snr_hi: float = 25.0,
                    aspect_angle_range_deg=ASPECT_ANGLE_RANGE_DEG,
                    target_range_m_range=TARGET_RANGE_M_RANGE) -> None:
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
    range_all = np.empty(n_total, dtype=np.float32)
    aspect_angle_all = np.empty(n_total, dtype=np.float32)
    target_range_bin_all = np.empty(n_total, dtype=np.int32)
    target_range_m_all = np.empty(n_total, dtype=np.float32)
    slow_time_prf_all = np.empty(n_total, dtype=np.float32)
    slow_time_samples_all = np.empty(n_total, dtype=np.int32)
    max_abs_radial_velocity_all = np.empty(n_total, dtype=np.float32)
    radar_max_unambiguous_velocity_all = np.empty(n_total, dtype=np.float32)
    doppler_alias_margin_all = np.empty(n_total, dtype=np.float32)
    n_scatterers_all = np.empty(n_total, dtype=np.int16)
    torso_scatterers_all = np.empty(n_total, dtype=np.int16)
    head_scatterers_all = np.empty(n_total, dtype=np.int16)
    limb_scatterers_all = np.empty(n_total, dtype=np.int16)

    t0 = time.time()
    idx = 0
    progress_every = max(100, min(2000, max(n_total, 1) // 10))
    for cls_idx, activity in enumerate(ACTIVITY_LABELS):
        for _ in range(n_per_class):
            snr_db = rng.uniform(snr_lo, snr_hi)
            range_m = float(rng.uniform(*target_range_m_range))
            aspect_angle = float(rng.uniform(*aspect_angle_range_deg))
            spec, label, meta = generate_har_sample(
                activity,
                rng,
                snr_db=snr_db,
                aspect_angle=aspect_angle,
                range_m=range_m,
                radar=RADAR,
            )
            if meta["doppler_alias_margin_mps"] < 0.0:
                raise RuntimeError(
                    "P02 default teaching data exceeded the Doppler alias guard: "
                    f"activity={activity}, max_abs_radial_velocity_mps="
                    f"{meta['max_abs_radial_velocity_mps']:.2f}, guard="
                    f"{P02_DOPPLER_ALIAS_SAFETY_FACTOR * meta['radar_max_unambiguous_velocity_mps']:.2f}. "
                    "Tighten activity ranges or increase slow-time PRF before regenerating."
                )
            feat = extract_handcrafted_features(spec)

            spec_all[idx] = spec
            label_all[idx] = label
            feat_all[idx] = feat
            snr_all[idx] = snr_db
            range_all[idx] = range_m
            aspect_angle_all[idx] = aspect_angle
            target_range_bin_all[idx] = meta["target_range_bin"]
            target_range_m_all[idx] = meta["target_range_m"]
            slow_time_prf_all[idx] = meta["slow_time_prf_hz"]
            slow_time_samples_all[idx] = meta["slow_time_samples"]
            max_abs_radial_velocity_all[idx] = meta["max_abs_radial_velocity_mps"]
            radar_max_unambiguous_velocity_all[idx] = meta["radar_max_unambiguous_velocity_mps"]
            doppler_alias_margin_all[idx] = meta["doppler_alias_margin_mps"]
            n_scatterers_all[idx] = meta["n_scatterers"]
            torso_scatterers_all[idx] = meta["scatter_kind_counts"]["torso"]
            head_scatterers_all[idx] = meta["scatter_kind_counts"]["head"]
            limb_scatterers_all[idx] = meta["scatter_kind_counts"]["limb"]
            idx += 1

            if idx % progress_every == 0:
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
        range_m=range_all,
        aspect_angle_deg=aspect_angle_all,
        target_range_bin=target_range_bin_all,
        target_range_m=target_range_m_all,
        slow_time_prf_hz=slow_time_prf_all,
        slow_time_samples=slow_time_samples_all,
        max_abs_radial_velocity_mps=max_abs_radial_velocity_all,
        radar_max_unambiguous_velocity_mps=radar_max_unambiguous_velocity_all,
        doppler_alias_margin_mps=doppler_alias_margin_all,
        n_scatterers=n_scatterers_all,
        torso_scatterers=torso_scatterers_all,
        head_scatterers=head_scatterers_all,
        limb_scatterers=limb_scatterers_all,
        range_axis_m=range_axis(RADAR).astype(np.float32),
        radar_fc_hz=np.array([RADAR.fc], dtype=np.float64),
        radar_bw_hz=np.array([RADAR.bw], dtype=np.float64),
        radar_fs_hz=np.array([RADAR.fs], dtype=np.float64),
        radar_pri_s=np.array([RADAR.PRI], dtype=np.float64),
        radar_config_n_chirps=np.array([RADAR.N_chirps], dtype=np.int32),
        fs_over_bandwidth=np.array([RADAR.fs / RADAR.bw], dtype=np.float32),
        aspect_angle_range_deg=np.array(aspect_angle_range_deg, dtype=np.float32),
        aspect_convention=np.array([ASPECT_CONVENTION.encode("utf-8")], dtype="S64"),
        target_range_m_range=np.array(target_range_m_range, dtype=np.float32),
        doppler_alias_safety_factor=np.array([P02_DOPPLER_ALIAS_SAFETY_FACTOR], dtype=np.float32),
        scatter_model=np.array([SCATTER_MODEL.encode("utf-8")], dtype="S64"),
        scatter_model_scope=np.array([b"p02_only"], dtype="S16"),
        range_processing=np.array([b"local_range_compressed_frame"], dtype="S64"),
        doppler_source=np.array([b"stft_of_target_range_signal"], dtype="S64"),
        schema_version=np.array([SCHEMA_VERSION], dtype=np.int32),
    )


def main():
    p = base_parser("Generate P02 micro-Doppler HAR datasets")
    p.add_argument("--n_train", type=int, default=30000)
    p.add_argument("--n_val",   type=int, default=3000)
    p.add_argument("--n_test",  type=int, default=3000)
    p.add_argument("--snr_lo",  type=float, default=5.0)
    p.add_argument("--snr_hi",  type=float, default=25.0)
    p.add_argument("--aspect_lo", type=float, default=ASPECT_ANGLE_RANGE_DEG[0])
    p.add_argument("--aspect_hi", type=float, default=ASPECT_ANGLE_RANGE_DEG[1])
    p.add_argument("--range_lo", type=float, default=TARGET_RANGE_M_RANGE[0])
    p.add_argument("--range_hi", type=float, default=TARGET_RANGE_M_RANGE[1])
    p.add_argument("--out_dir", type=str, default=None)
    args = p.parse_args()

    if args.smoke:
        args.n_train, args.n_val, args.n_test = 256, 64, 64

    seed_everything(args.seed)

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    aspect_range = (float(args.aspect_lo), float(args.aspect_hi))
    target_range = (float(args.range_lo), float(args.range_hi))

    print("=== P02: Generate Micro-Doppler HAR Datasets ===")
    print(f"  Classes ({N_CLASSES}): {ACTIVITY_LABELS}")
    print(f"  SNR range: [{args.snr_lo}, {args.snr_hi}] dB")
    print(
        f"  Aspect range: {aspect_range} deg "
        "(absolute angle; ± signs are symmetric in the current 2-D radial model)"
    )
    print(f"  Target range: {target_range} m")
    print(
        "  Doppler alias guard: max scatter velocity should remain below "
        f"{P02_DOPPLER_ALIAS_SAFETY_FACTOR:.0%} of "
        f"{radar_max_unambiguous_velocity_mps(RADAR):.2f} m/s"
    )
    print(
        "  Simulator: pedestrian scatterers -> local range-compressed frame "
        "-> target range signal -> STFT "
        f"(fc={RADAR.fc/1e9:.1f} GHz, fs/BW={RADAR.fs/RADAR.bw:.1f})"
    )

    for name, n, seed in [
        ("har_train.h5", args.n_train, args.seed),
        ("har_val.h5",   args.n_val,   args.seed + 1000),
        ("har_test.h5",  args.n_test,  args.seed + 2000),
    ]:
        print(f"\n  {name} ({n} samples)...")
        _generate_split(
            out_dir / name,
            n,
            seed,
            args.snr_lo,
            args.snr_hi,
            aspect_angle_range_deg=aspect_range,
            target_range_m_range=target_range,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
