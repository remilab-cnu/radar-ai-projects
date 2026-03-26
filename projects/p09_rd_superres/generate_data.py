"""P09 RD Super-Resolution — 데이터 생성

HR RD map (64x64) 생성 후 32x32로 다운샘플링하여 LR/HR 쌍 구성.
- LR input : x_lr (N, 1, 32, 32) — dB 단위 저해상도 RD map
- HR target: y_hr (N, 1, 64, 64) — dB 단위 고해상도 RD map
- Peak mask: peak_mask (N, 1, 64, 64) — HR 공간에서 표적 위치 binary mask

사용법:
    python generate_data.py --smoke          # 빠른 테스트
    python generate_data.py                  # 전체 생성
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import numpy as np
from scipy.ndimage import zoom

from shared.fmcw_simulator import FMCWRadar, generate_scene, range_doppler_map, to_db
from common.hdf5_io import save_hdf5
from common.seed import seed_everything

# ─── 레이다 파라미터 ────────────────────────────────────────────────────────────
HR_RANGE = 64   # HR RD map range bins
HR_DOPPLER = 64  # HR RD map doppler bins
LR_RANGE = 32   # LR (downsampled)
LR_DOPPLER = 32

DB_MIN = -60.0
DB_MAX = 0.0


def make_radar():
    """HR RD map 생성용 레이다 (64 chirps, 64 range samples)."""
    return FMCWRadar(
        fc=77e9,
        bw=1e9,
        T_chirp=50e-6,
        N_chirps=HR_DOPPLER,   # 64 chirps → Doppler axis 64
        fs=10e6,
        N_rx=1,
    )


def rdm_to_db_normalized(rdm_complex: np.ndarray) -> np.ndarray:
    """복소 RDM → dB, 정규화 [DB_MIN, DB_MAX] → [-1, 1]."""
    mag = np.abs(rdm_complex)
    db = 20.0 * np.log10(mag / (mag.max() + 1e-30) + 1e-30)
    db = np.clip(db, DB_MIN, DB_MAX)
    # Normalize to [-1, 1]
    db_norm = (db - DB_MIN) / (DB_MAX - DB_MIN) * 2.0 - 1.0
    return db_norm.astype(np.float32)


def generate_one_sample(radar: FMCWRadar, rng: np.random.Generator, sample_seed: int):
    """단일 샘플 생성.

    Returns
    -------
    x_lr : (1, 32, 32) float32
    y_hr : (1, 64, 64) float32
    peak_mask : (1, 64, 64) float32
    n_targets : int
    snr_db : float
    """
    # 1. 표적 파라미터 샘플링
    n_tgt = int(rng.integers(1, 5))   # 1~4 targets
    snr_db = float(rng.uniform(5.0, 25.0))

    targets = []
    r_max = radar.max_range * 0.8
    v_max = radar.max_vel * 0.8

    for _ in range(n_tgt):
        targets.append({
            'range': float(rng.uniform(3.0, r_max)),
            'velocity': float(rng.uniform(-v_max, v_max)),
            'rcs': float(10 ** rng.uniform(-1, 1)),
        })

    # 2. Beat signal 생성 → HR RDM
    signal = generate_scene(radar, targets, snr_db=snr_db, seed=sample_seed)
    # signal: (1, N_chirps, N_samples) — N_rx=1
    # RDM: (1, N_chirps, N_samples) complex
    rdm = range_doppler_map(signal, window_range='hann', window_doppler='hann')
    # Crop to HR_RANGE range bins (positive range only)
    rdm_hr = rdm[0, :, :HR_RANGE]   # (64, 64) complex

    # 3. HR map in dB
    y_map = rdm_to_db_normalized(rdm_hr)   # (64, 64)

    # 4. LR by 2x spatial downsampling (average pooling via zoom)
    x_map_lr = zoom(y_map, 0.5, order=1)   # (32, 32)

    # 5. Peak mask: mark target locations in HR grid
    peak_mask = np.zeros((HR_DOPPLER, HR_RANGE), dtype=np.float32)
    vel_axis = np.fft.fftshift(np.fft.fftfreq(radar.N_chirps)) * radar.lam / (2 * radar.T_chirp)
    range_res = radar.range_res

    for tgt in targets:
        r_bin = int(round(tgt['range'] / range_res))
        v_bin = int(np.argmin(np.abs(vel_axis - tgt['velocity'])))
        if 0 <= r_bin < HR_RANGE and 0 <= v_bin < HR_DOPPLER:
            # 3x3 neighborhood for peak marking
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    vi = np.clip(v_bin + di, 0, HR_DOPPLER - 1)
                    ri = np.clip(r_bin + dj, 0, HR_RANGE - 1)
                    peak_mask[vi, ri] = 1.0

    return (
        x_map_lr[np.newaxis].astype(np.float32),   # (1, 32, 32)
        y_map[np.newaxis].astype(np.float32),        # (1, 64, 64)
        peak_mask[np.newaxis].astype(np.float32),    # (1, 64, 64)
        n_tgt,
        snr_db,
    )


def generate_split(
    name: str,
    n: int,
    radar: FMCWRadar,
    rng: np.random.Generator,
    out_dir: Path,
    seed_offset: int = 0,
):
    """데이터셋 분할 생성 및 저장."""
    print(f"  Generating {name} split ({n} samples)...")

    x_lr_list, y_hr_list, mask_list = [], [], []
    n_targets_list, snr_list = [], []

    for i in range(n):
        sample_seed = seed_offset + i
        x_lr, y_hr, peak_mask, n_tgt, snr_db = generate_one_sample(radar, rng, sample_seed)
        x_lr_list.append(x_lr)
        y_hr_list.append(y_hr)
        mask_list.append(peak_mask)
        n_targets_list.append(n_tgt)
        snr_list.append(snr_db)

        if (i + 1) % 1000 == 0:
            print(f"    {i+1}/{n}")

    save_hdf5(
        out_dir / f"{name}.h5",
        x_lr=np.stack(x_lr_list),
        y_hr=np.stack(y_hr_list),
        peak_mask=np.stack(mask_list),
        n_targets=np.array(n_targets_list, dtype=np.int32),
        snr_db=np.array(snr_list, dtype=np.float32),
    )


def main():
    parser = argparse.ArgumentParser(description="P09 RD Super-Resolution 데이터 생성")
    parser.add_argument("--smoke", action="store_true", help="소규모 smoke 테스트")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)

    radar = make_radar()
    radar.print_params()

    if args.smoke:
        splits = [("train", 256), ("val", 64), ("test", 64)]
    else:
        splits = [("train", 12000), ("val", 2000), ("test", 2000)]

    print(f"\nGenerating {'smoke' if args.smoke else 'full'} dataset...")
    for name, n in splits:
        seed_offset = {"train": 0, "val": 100000, "test": 200000}[name]
        generate_split(name, n, radar, rng, out_dir, seed_offset=seed_offset)

    print("\nDone. Files written to:", out_dir)


if __name__ == "__main__":
    main()
