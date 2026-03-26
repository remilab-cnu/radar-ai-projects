"""P06 I/Q Imbalance Correction — 데이터 생성

FMCW beat signal에 I/Q imbalance를 인가하고,
CNN이 imbalance 파라미터를 추정하도록 학습 데이터를 생성한다.

I/Q Imbalance 모델:
  y_I(t) = I(t) + dc_i
  y_Q(t) = g * (I(t)*sin(φ) + Q(t)*cos(φ)) + dc_q

  where g = 10^(gain_db/20), φ = phase_deg * π/180

HDF5 keys:
  x_corrupt  (N, 2, 512)  — 손상된 I/Q (ch0: real, ch1: imag)
  y_params   (N, 4)       — [gain_db, phase_deg, dc_i, dc_q]
  y_clean    (N, 2, 512)  — 원본 I/Q (ch0: real, ch1: imag)
  gain_db    (N,)         — gain mismatch [dB]
  phase_deg  (N,)         — phase mismatch [deg]
  snr_db     (N,)         — SNR [dB]

Splits: train 18K / val 3K / test 3K (smoke: 256/64/64)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import numpy as np

from shared.fmcw_simulator import FMCWRadar, generate_scene
from common.hdf5_io import save_hdf5
from common.seed import seed_everything

N_SAMPLES = 512  # beat signal 길이 (1 chirp, fast-time samples)


def make_radar() -> FMCWRadar:
    """512 샘플을 갖는 FMCW 레이다 설정."""
    return FMCWRadar(
        fc=77e9,
        bw=1e9,
        T_chirp=51.2e-6,   # 51.2 us @ 10 MHz → exactly 512 samples
        N_chirps=1,         # 단일 chirp (fast-time 신호만 사용)
        fs=10e6,
        N_rx=1,
    )


def apply_iq_imbalance(
    signal_iq: np.ndarray,  # (2, N) — clean I/Q
    gain_db: float,
    phase_deg: float,
    dc_i: float,
    dc_q: float,
) -> np.ndarray:
    """I/Q imbalance 인가.

    Parameters
    ----------
    signal_iq : (2, N) — row0: I, row1: Q
    gain_db   : amplitude gain mismatch (Q channel gain error) [dB]
    phase_deg : quadrature phase mismatch [deg]
    dc_i, dc_q: DC offset on I and Q channels

    Returns
    -------
    corrupted : (2, N)
    """
    I = signal_iq[0]
    Q = signal_iq[1]

    g = 10.0 ** (gain_db / 20.0)   # linear gain
    phi = np.deg2rad(phase_deg)

    # Standard I/Q imbalance model
    I_out = I + dc_i
    Q_out = g * (I * np.sin(phi) + Q * np.cos(phi)) + dc_q

    return np.stack([I_out, Q_out], axis=0).astype(np.float32)


def generate_one_sample(
    radar: FMCWRadar,
    rng: np.random.Generator,
    snr_db: float,
    gain_range: tuple[float, float] = (0.5, 3.0),
    phase_range: tuple[float, float] = (1.0, 15.0),
    dc_range: tuple[float, float] = (-0.05, 0.05),
    seed: int = 0,
) -> dict:
    """단일 샘플 생성."""
    # 1~3개 표적
    n_tgt = rng.integers(1, 4)
    targets = []
    for _ in range(n_tgt):
        r = rng.uniform(5.0, radar.max_range * 0.85)
        rcs = 10 ** rng.uniform(-1, 1)
        targets.append({'range': r, 'velocity': 0.0, 'rcs': rcs})

    # Clean beat signal (single chirp)
    signal = generate_scene(radar, targets, snr_db=snr_db, seed=seed)
    # signal: (1, 1, N_samples) — N_rx=1, N_chirps=1
    beat = signal[0, 0, :N_SAMPLES]  # (N_samples,) complex

    # Normalize to prevent scale issues
    scale = np.sqrt(np.mean(np.abs(beat) ** 2)) + 1e-30
    beat_norm = beat / scale

    I_clean = beat_norm.real.astype(np.float32)
    Q_clean = beat_norm.imag.astype(np.float32)
    clean_iq = np.stack([I_clean, Q_clean], axis=0)  # (2, N)

    # Sample imbalance parameters
    gain_db = float(rng.uniform(gain_range[0], gain_range[1]))
    # Random sign for gain (can be positive or negative imbalance)
    gain_db *= float(rng.choice([-1.0, 1.0]))
    gain_db = float(np.clip(gain_db, -gain_range[1], gain_range[1]))

    phase_deg = float(rng.uniform(phase_range[0], phase_range[1]))
    phase_deg *= float(rng.choice([-1.0, 1.0]))

    dc_i = float(rng.uniform(dc_range[0], dc_range[1]))
    dc_q = float(rng.uniform(dc_range[0], dc_range[1]))

    corrupt_iq = apply_iq_imbalance(clean_iq, gain_db, phase_deg, dc_i, dc_q)

    return {
        "x_corrupt": corrupt_iq,           # (2, 512)
        "y_params": np.array([gain_db, phase_deg, dc_i, dc_q], dtype=np.float32),
        "y_clean": clean_iq,               # (2, 512)
        "gain_db": np.float32(gain_db),
        "phase_deg": np.float32(phase_deg),
        "snr_db": np.float32(snr_db),
    }


def generate_split(
    n_samples: int,
    radar: FMCWRadar,
    rng: np.random.Generator,
    seed_base: int,
    snr_range: tuple[float, float] = (5.0, 25.0),
) -> dict[str, np.ndarray]:
    """n_samples개의 샘플 생성."""
    all_x, all_yp, all_yc = [], [], []
    all_gain, all_phase, all_snr = [], [], []

    for i in range(n_samples):
        snr_db = float(rng.uniform(snr_range[0], snr_range[1]))
        sample = generate_one_sample(
            radar, rng, snr_db, seed=seed_base + i
        )
        all_x.append(sample["x_corrupt"])
        all_yp.append(sample["y_params"])
        all_yc.append(sample["y_clean"])
        all_gain.append(sample["gain_db"])
        all_phase.append(sample["phase_deg"])
        all_snr.append(sample["snr_db"])

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n_samples}...")

    return {
        "x_corrupt": np.stack(all_x, axis=0),    # (N, 2, 512)
        "y_params": np.stack(all_yp, axis=0),     # (N, 4)
        "y_clean": np.stack(all_yc, axis=0),      # (N, 2, 512)
        "gain_db": np.array(all_gain, dtype=np.float32),
        "phase_deg": np.array(all_phase, dtype=np.float32),
        "snr_db": np.array(all_snr, dtype=np.float32),
    }


def main():
    parser = argparse.ArgumentParser(description="P06 I/Q Imbalance — 데이터 생성")
    parser.add_argument("--smoke", action="store_true", help="소규모 smoke test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)

    base = Path(__file__).parent
    out_dir = Path(args.out_dir) if args.out_dir else base / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        splits = {"train": 256, "val": 64, "test": 64}
    else:
        splits = {"train": 18000, "val": 3000, "test": 3000}

    radar = make_radar()
    print(f"Radar: N_samples={radar.N_samples}, max_range={radar.max_range:.1f}m")

    for split_name, n_samples in splits.items():
        print(f"\n[{split_name}] Generating {n_samples} samples...")
        seed_base = args.seed + {"train": 0, "val": 100000, "test": 200000}[split_name]
        rng = np.random.default_rng(seed_base)
        data = generate_split(n_samples, radar, rng, seed_base)
        print(f"  x_corrupt: {data['x_corrupt'].shape}")
        print(f"  gain_db range: [{data['gain_db'].min():.2f}, {data['gain_db'].max():.2f}]")
        print(f"  phase_deg range: [{data['phase_deg'].min():.2f}, {data['phase_deg'].max():.2f}]")
        save_hdf5(out_dir / f"{split_name}.h5", **data)

    print("\nData generation complete.")


if __name__ == "__main__":
    main()
