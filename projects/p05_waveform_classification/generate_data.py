#!/usr/bin/env python3
"""P05 — Generate lightweight radar waveform-classification datasets."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.cli import base_parser
from common.hdf5_io import save_hdf5
from common.seed import seed_everything
from shared.waveform_library import (
    SCHEMA_VERSION,
    WAVEFORM_CLASSES,
    WaveformConfig,
    class_name_bytes,
    generate_waveform_sample,
)

MATLAB_REFERENCE_URL = "https://www.mathworks.com/help/radar/ug/radar-and-communications-waveform-classification-using-deep-learning.html"
MATLAB_LPI_REFERENCE_URL = "https://www.mathworks.com/help/radar/ug/LPI-radar-waveform-classification-using-time-frequency-CNN.html"


def _generate_split(path: Path, n_samples: int, seed: int, config: WaveformConfig) -> None:
    rng = np.random.default_rng(seed)
    h, w = config.image_size
    dummy_img, _, dummy_feat, _ = generate_waveform_sample(WAVEFORM_CLASSES[0], rng, config, snr_db=12.0)
    n_features = int(dummy_feat.size)
    n_classes = len(WAVEFORM_CLASSES)
    n_per_class = max(1, n_samples // n_classes) if n_samples > 0 else 0
    n_total = n_per_class * n_classes

    x_all = np.empty((n_total, 1, h, w), dtype=np.float32)
    y_all = np.empty(n_total, dtype=np.int64)
    features_all = np.empty((n_total, n_features), dtype=np.float32)
    snr_all = np.empty(n_total, dtype=np.float32)
    pulse_start_all = np.empty(n_total, dtype=np.int32)
    pulse_width_all = np.empty(n_total, dtype=np.int32)
    bandwidth_all = np.empty(n_total, dtype=np.float32)
    freq_offset_all = np.empty(n_total, dtype=np.float32)
    barker_len_all = np.empty(n_total, dtype=np.int16)

    t0 = time.time()
    idx = 0
    progress_every = max(100, min(1000, max(n_total, 1) // 5))
    for class_name in WAVEFORM_CLASSES:
        for _ in range(n_per_class):
            image, label, features, meta = generate_waveform_sample(class_name, rng, config)
            x_all[idx, 0] = image
            y_all[idx] = int(label)
            features_all[idx] = features
            snr_all[idx] = float(meta["snr_db"])
            pulse_start_all[idx] = int(meta["pulse_start_sample"])
            pulse_width_all[idx] = int(meta["pulse_width_samples"])
            bandwidth_all[idx] = float(meta["bandwidth_hz"])
            freq_offset_all[idx] = float(meta.get("frequency_offset_hz", 0.0))
            barker_len_all[idx] = int(meta.get("barker_length", 0))
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
        pulse_start_sample=pulse_start_all[order],
        pulse_width_samples=pulse_width_all[order],
        bandwidth_hz=bandwidth_all[order],
        frequency_offset_hz=freq_offset_all[order],
        barker_length=barker_len_all[order],
        waveform_class_names=class_name_bytes(),
        sample_rate_hz=np.array([config.sample_rate_hz], dtype=np.float64),
        waveform_duration_s=np.array([config.duration_s], dtype=np.float64),
        n_samples=np.array([config.n_samples], dtype=np.int32),
        image_size=np.array(config.image_size, dtype=np.int32),
        stft_window=np.array([config.stft_window], dtype=np.int32),
        stft_overlap=np.array([config.stft_overlap], dtype=np.int32),
        stft_nfft=np.array([config.stft_nfft], dtype=np.int32),
        snr_range_db=np.array(config.snr_range_db, dtype=np.float32),
        bandwidth_range_hz=np.array(config.bandwidth_range_hz, dtype=np.float64),
        pulse_fraction_range=np.array(config.pulse_fraction_range, dtype=np.float32),
        max_frequency_offset_hz=np.array([config.max_frequency_offset_hz], dtype=np.float64),
        matlab_reference_url=np.array([MATLAB_REFERENCE_URL.encode("utf-8")], dtype="S160"),
        matlab_lpi_reference_url=np.array([MATLAB_LPI_REFERENCE_URL.encode("utf-8")], dtype="S160"),
        representation=np.array([b"stft_log_magnitude_image"], dtype="S64"),
        schema_version=np.array([SCHEMA_VERSION], dtype=np.int32),
    )


def main() -> None:
    p = base_parser("Generate P05 lightweight waveform-classification datasets")
    p.add_argument("--n_train", type=int, default=4000)
    p.add_argument("--n_val", type=int, default=800)
    p.add_argument("--n_test", type=int, default=800)
    p.add_argument("--snr_lo", type=float, default=-6.0)
    p.add_argument("--snr_hi", type=float, default=18.0)
    p.add_argument("--out_dir", type=str, default=None)
    args = p.parse_args()

    if args.smoke:
        args.n_train, args.n_val, args.n_test = 256, 64, 64

    seed_everything(args.seed)
    config = WaveformConfig(snr_range_db=(float(args.snr_lo), float(args.snr_hi)))
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== P05: Generate Radar Waveform Classification Data ===")
    print(f"  Classes ({len(WAVEFORM_CLASSES)}): {list(WAVEFORM_CLASSES)}")
    print(f"  SNR range: [{args.snr_lo}, {args.snr_hi}] dB")
    print(f"  Representation: {config.image_size[0]}x{config.image_size[1]} STFT log-magnitude image")
    print("  MATLAB reference: Radar and Communications Waveform Classification")

    for name, n, seed in [
        ("waveform_train.h5", args.n_train, args.seed),
        ("waveform_val.h5", args.n_val, args.seed + 1000),
        ("waveform_test.h5", args.n_test, args.seed + 2000),
    ]:
        print(f"\n  {name} ({n} requested samples)...")
        _generate_split(out_dir / name, n, seed, config)

    print(f"\nDone. Data saved to {out_dir}")


if __name__ == "__main__":
    main()
