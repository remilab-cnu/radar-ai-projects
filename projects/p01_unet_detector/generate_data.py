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
from shared.fmcw_simulator import FMCWRadar, range_axis, velocity_axis
from shared.clutter_model import generate_random_scene


RADAR = FMCWRadar(
    fc=9.6e9,
    bw=50e6,
    T_chirp=2e-6,
    PRI=100e-6,
    N_chirps=64,
    fs=200e6,  # 4 × signal bandwidth
    N_rx=1,
)
N_CHANNELS = 2  # mag + phase
MAX_TARGETS = 15
SCHEMA_VERSION = 9
MIN_LABEL_SNR_DB = 6.0
ADC_IQ_BITS = 16
ADC_IQ_FULL_SCALE = 6.0e-5
MTI_MODE = "mean_removal"
TARGET_RANGE_MIN_M = 15.0
TARGET_MIN_ABS_VELOCITY_MPS = 2.0 * RADAR.vel_res
TARGET_RCS_LOG10_RANGE = (-1.3, 0.5)  # 0.05 ~ 3.16 m^2
N_TARGETS_RANGE = (1, 8)
CLUTTER_POWER_RANGE_DB = (-24.0, -16.0)
CLUTTER_RCS_MAX = 0.15
X_STORAGE_DTYPE = np.float16
Y_STORAGE_DTYPE = np.uint8


def _generate_split(path: Path, n_samples: int, seed: int) -> None:
    """Generate one HDF5 split file."""
    radar = RADAR
    rng = np.random.default_rng(seed)
    Nc = radar.N_chirps
    Nr = radar.N_range_bins

    rdm_all = np.empty((n_samples, N_CHANNELS, Nc, Nr), dtype=X_STORAGE_DTYPE)
    rdm_mag_all = np.empty((n_samples, Nc, Nr), dtype=np.float32)
    mask_all = np.empty((n_samples, Nc, Nr), dtype=Y_STORAGE_DTYPE)
    snr_all = np.empty(n_samples, dtype=np.float32)
    ntgt_all = np.empty(n_samples, dtype=np.int32)
    clutter_power_all = np.empty(n_samples, dtype=np.float32)
    noise_floor_all = np.empty(n_samples, dtype=np.float32)
    adc_clipped_fraction_all = np.empty(n_samples, dtype=np.float32)
    mti_applied_all = np.empty(n_samples, dtype=np.bool_)
    target_range_all = np.full((n_samples, MAX_TARGETS), np.nan, dtype=np.float32)
    target_velocity_all = np.full((n_samples, MAX_TARGETS), np.nan, dtype=np.float32)
    target_rcs_all = np.full((n_samples, MAX_TARGETS), np.nan, dtype=np.float32)
    target_actual_snr_all = np.full((n_samples, MAX_TARGETS), np.nan, dtype=np.float32)
    target_peak_snr_all = np.full((n_samples, MAX_TARGETS), np.nan, dtype=np.float32)
    target_global_peak_snr_all = np.full((n_samples, MAX_TARGETS), np.nan, dtype=np.float32)
    target_local_bg_floor_all = np.full((n_samples, MAX_TARGETS), np.nan, dtype=np.float32)
    target_effective_bg_floor_all = np.full((n_samples, MAX_TARGETS), np.nan, dtype=np.float32)
    target_range_bin_all = np.full((n_samples, MAX_TARGETS), -1, dtype=np.int32)
    target_doppler_bin_all = np.full((n_samples, MAX_TARGETS), -1, dtype=np.int32)

    t0 = time.time()
    for i in range(n_samples):
        rdm_input, target_mask, meta = generate_random_scene(
            radar,
            rng,
            n_targets_range=N_TARGETS_RANGE,
            clutter_power_range=CLUTTER_POWER_RANGE_DB,
            target_range_min_m=TARGET_RANGE_MIN_M,
            target_min_abs_velocity_mps=TARGET_MIN_ABS_VELOCITY_MPS,
            target_rcs_log10_range=TARGET_RCS_LOG10_RANGE,
            clutter_rcs_max=CLUTTER_RCS_MAX,
            min_label_snr_db=MIN_LABEL_SNR_DB,
            adc_bits=ADC_IQ_BITS,
            adc_full_scale=ADC_IQ_FULL_SCALE,
            mti_mode=MTI_MODE,
            require_no_adc_clipping=True,
            return_raw=True,
        )
        rdm_all[i] = rdm_input[:N_CHANNELS].astype(X_STORAGE_DTYPE)
        rdm_mag_all[i] = meta['rdm_mag_linear']
        mask_all[i] = target_mask.astype(Y_STORAGE_DTYPE)
        snr_all[i] = meta['snr_db']
        ntgt_all[i] = meta['n_targets']
        clutter_power_all[i] = meta['clutter_power_db']
        noise_floor_all[i] = meta['noise_floor']
        adc_clipped_fraction_all[i] = meta['adc_clipped_fraction']
        mti_applied_all[i] = meta['mti_applied']
        for j, info in enumerate(meta['target_info'][:MAX_TARGETS]):
            target_range_all[i, j] = info['range']
            target_velocity_all[i, j] = info['velocity']
            target_rcs_all[i, j] = info['rcs']
            target_actual_snr_all[i, j] = info.get('actual_snr_db', np.nan)
            target_peak_snr_all[i, j] = info.get('peak_snr_db', np.nan)
            target_global_peak_snr_all[i, j] = info.get('global_peak_snr_db', np.nan)
            target_local_bg_floor_all[i, j] = info.get('local_background_floor', np.nan)
            target_effective_bg_floor_all[i, j] = info.get('effective_background_floor', np.nan)
            target_range_bin_all[i, j] = info['range_bin']
            target_doppler_bin_all[i, j] = info['doppler_bin']

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_samples - i - 1) / rate
            print(f"    [{i+1:>7d}/{n_samples}]  {rate:.0f} samples/s  ETA {eta:.0f}s")

    velocity_axis_mps = velocity_axis(radar)

    # mask stored as (N, 1, Nc, Nr) to match HDF5Dataset x_key/y_key convention
    save_hdf5(
        path,
        x=rdm_all,
        y=mask_all[:, np.newaxis, :, :],
        rdm_mag_linear=rdm_mag_all,
        snr_db=snr_all,
        n_targets=ntgt_all,
        clutter_power_db=clutter_power_all,
        noise_floor=noise_floor_all,
        adc_clipped_fraction=adc_clipped_fraction_all,
        mti_applied=mti_applied_all,
        target_range_m=target_range_all,
        target_velocity_mps=target_velocity_all,
        target_rcs=target_rcs_all,
        target_actual_snr_db=target_actual_snr_all,
        target_peak_snr_db=target_peak_snr_all,
        target_global_peak_snr_db=target_global_peak_snr_all,
        target_local_bg_floor=target_local_bg_floor_all,
        target_effective_bg_floor=target_effective_bg_floor_all,
        target_range_bin=target_range_bin_all,
        target_doppler_bin=target_doppler_bin_all,
        range_axis_m=range_axis(radar)[:Nr],
        velocity_axis_mps=velocity_axis_mps,
        radar_fc_hz=np.array([radar.fc], dtype=np.float64),
        radar_bw_hz=np.array([radar.bw], dtype=np.float64),
        radar_fs_hz=np.array([radar.fs], dtype=np.float64),
        fs_over_bandwidth=np.array([radar.fs / radar.bw], dtype=np.float32),
        min_label_snr_db=np.array([MIN_LABEL_SNR_DB], dtype=np.float32),
        adc_iq_bits=np.array([ADC_IQ_BITS], dtype=np.int32),
        adc_iq_full_scale=np.array([ADC_IQ_FULL_SCALE], dtype=np.float64),
        adc_iq_component_dtype=np.array([b"int16"]),
        mti_mode=np.array([b"slow_time_mean_removal_dc_notch"]),
        target_range_min_m=np.array([TARGET_RANGE_MIN_M], dtype=np.float32),
        target_min_abs_velocity_mps=np.array([TARGET_MIN_ABS_VELOCITY_MPS], dtype=np.float32),
        target_rcs_log10_range=np.array(TARGET_RCS_LOG10_RANGE, dtype=np.float32),
        n_targets_range=np.array(N_TARGETS_RANGE, dtype=np.int32),
        clutter_power_range_db=np.array(CLUTTER_POWER_RANGE_DB, dtype=np.float32),
        clutter_rcs_max=np.array([CLUTTER_RCS_MAX], dtype=np.float32),
        clutter_type=np.array([b"static"]),
        x_storage_dtype=np.array([str(np.dtype(X_STORAGE_DTYPE)).encode()]),
        y_storage_dtype=np.array([str(np.dtype(Y_STORAGE_DTYPE)).encode()]),
        schema_version=np.array([SCHEMA_VERSION], dtype=np.int32),
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
    print(f"  Radar: Nc={radar.N_chirps}, Nr={radar.N_range_bins}")
    print(f"  range_res={radar.range_res:.3f} m, vel_res={radar.vel_res:.3f} m/s")
    print(f"  Channels: {N_CHANNELS}")
    print(f"  Simulator: shared FMCW dechirp/mixing core (fs/BW={radar.fs/radar.bw:.1f})")
    print(
        f"  Schema: v{SCHEMA_VERSION} with linear rdm_mag_linear, "
        f"target_actual_snr_db, target_peak_snr_db, "
        f"and min_label_snr_db={MIN_LABEL_SNR_DB:.1f} dB"
    )
    print(
        f"  P1 clutter: static-only, power {CLUTTER_POWER_RANGE_DB} dB, "
        "no multipath / no Doppler tail; MTI DC-notch input"
    )
    print(
        "  Label gate: effective(global median, local RD background) "
        f">= {MIN_LABEL_SNR_DB:.1f} dB; target count range {N_TARGETS_RANGE}; "
        f"|velocity| >= {TARGET_MIN_ABS_VELOCITY_MPS:.2f} m/s to avoid static-clutter ambiguity"
    )
    print(
        f"  ADC: complex {ADC_IQ_BITS}-bit I/Q = int16 I + int16 Q components, "
        f"fixed full-scale={ADC_IQ_FULL_SCALE:.2e}; stored x={np.dtype(X_STORAGE_DTYPE)}, "
        f"y={np.dtype(Y_STORAGE_DTYPE)}"
    )

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
