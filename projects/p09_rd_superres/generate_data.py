"""P09 RD Super-Resolution — 데이터 생성

동일한 표적 scene을 두 개의 물리 레이다 설정으로 관측해 LR/HR 쌍을 구성한다.
- LR input : x_lr (N, 1, 32, 32) — 저대역폭/저 chirp 수 레이다 RDM
- HR target: y_hr (N, 1, 64, 64) — 고대역폭/고 chirp 수 레이다 RDM
- Peak mask: peak_mask (N, 1, 64, 64) — HR 공간에서 표적 위치 binary mask

사용법:
    python generate_data.py --smoke          # 빠른 테스트
    python generate_data.py                  # 전체 생성
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np

from common.cli import base_parser

from shared.fmcw_simulator import FMCWRadar, generate_scene, range_doppler_map
from common.seed import seed_everything

# ─── 레이다 파라미터 ────────────────────────────────────────────────────────────
HR_RANGE = 64     # HR RD map range bins
HR_DOPPLER = 64   # HR RD map doppler bins
LR_RANGE = 32     # LR physical range bins
LR_DOPPLER = 32   # LR physical doppler bins

DB_MIN = -60.0
DB_MAX = 0.0


def make_hr_radar() -> FMCWRadar:
    """HR RD map 생성용 레이다 (1 GHz bandwidth, 64 chirps)."""
    return FMCWRadar(
        fc=77e9,
        bw=1e9,
        T_chirp=50e-6,
        N_chirps=HR_DOPPLER,
        fs=10e6,
        N_rx=1,
    )


def make_lr_radar() -> FMCWRadar:
    """LR RD map 생성용 물리 레이다 (0.5 GHz bandwidth, 32 chirps).

    HR map을 이미지로 다운샘플링하지 않는다. 대역폭을 절반으로 낮춰 range
    bin spacing을 2배로 만들고, chirp 수를 절반으로 낮춰 Doppler bin spacing을
    2배로 만든다. 따라서 32×32 LR map은 64×64 HR map과 거의 같은 물리
    extent를 더 성긴 bin으로 관측한다.
    """
    return FMCWRadar(
        fc=77e9,
        bw=0.5e9,
        T_chirp=50e-6,
        N_chirps=LR_DOPPLER,
        fs=10e6,
        N_rx=1,
    )


def radar_config_attrs(hr_radar: FMCWRadar, lr_radar: FMCWRadar) -> dict[str, float | int | str]:
    """HDF5에 저장할 LR/HR 레이다 설정 및 bin spacing metadata."""
    return {
        "generation_mode": "physical_lr_hr_radar_configs",
        "normalization": f"per-map magnitude dB clipped to [{DB_MIN}, {DB_MAX}] then scaled to [-1, 1]",
        "hr_fc_hz": hr_radar.fc,
        "hr_bw_hz": hr_radar.bw,
        "hr_t_chirp_s": hr_radar.T_chirp,
        "hr_n_chirps": hr_radar.N_chirps,
        "hr_fs_hz": hr_radar.fs,
        "hr_range_bins": HR_RANGE,
        "hr_doppler_bins": HR_DOPPLER,
        "hr_range_bin_spacing_m": hr_radar.range_res,
        "hr_doppler_bin_spacing_mps": hr_radar.vel_res,
        "lr_fc_hz": lr_radar.fc,
        "lr_bw_hz": lr_radar.bw,
        "lr_t_chirp_s": lr_radar.T_chirp,
        "lr_n_chirps": lr_radar.N_chirps,
        "lr_fs_hz": lr_radar.fs,
        "lr_range_bins": LR_RANGE,
        "lr_doppler_bins": LR_DOPPLER,
        "lr_range_bin_spacing_m": lr_radar.range_res,
        "lr_doppler_bin_spacing_mps": lr_radar.vel_res,
        "shared_range_extent_m": min(HR_RANGE * hr_radar.range_res, LR_RANGE * lr_radar.range_res),
        "shared_velocity_extent_mps": 2.0 * min(hr_radar.max_vel, lr_radar.max_vel),
    }


def rdm_to_db_normalized(rdm_complex: np.ndarray) -> np.ndarray:
    """복소 RDM → dB, 정규화 [DB_MIN, DB_MAX] → [-1, 1]."""
    mag = np.abs(rdm_complex)
    db = 20.0 * np.log10(mag / (mag.max() + 1e-30) + 1e-30)
    db = np.clip(db, DB_MIN, DB_MAX)
    db_norm = (db - DB_MIN) / (DB_MAX - DB_MIN) * 2.0 - 1.0
    return db_norm.astype(np.float32)


def doppler_axis(radar: FMCWRadar) -> np.ndarray:
    """Doppler bin centers in m/s for fftshifted RDM output."""
    return np.fft.fftshift(np.fft.fftfreq(radar.N_chirps)) * radar.lam / (2 * radar.T_chirp)


def simulate_rdm_map(
    radar: FMCWRadar,
    targets: list[dict[str, float]],
    snr_db: float,
    sample_seed: int,
    n_range_bins: int,
) -> np.ndarray:
    """Simulate one radar config and return normalized dB RDM crop."""
    signal = generate_scene(radar, targets, snr_db=snr_db, seed=sample_seed)
    rdm = range_doppler_map(signal, window_range="hann", window_doppler="hann")
    return rdm_to_db_normalized(rdm[0, :, :n_range_bins])


def generate_one_sample(
    hr_radar: FMCWRadar,
    lr_radar: FMCWRadar,
    rng: np.random.Generator,
    sample_seed: int,
):
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
    shared_range_extent = min(HR_RANGE * hr_radar.range_res, LR_RANGE * lr_radar.range_res)
    r_max = shared_range_extent * 0.8
    v_max = min(hr_radar.max_vel, lr_radar.max_vel) * 0.8

    for _ in range(n_tgt):
        targets.append({
            'range': float(rng.uniform(3.0, r_max)),
            'velocity': float(rng.uniform(-v_max, v_max)),
            'rcs': float(10 ** rng.uniform(-1, 1)),
        })

    # 2. 동일 scene을 물리적으로 다른 LR/HR radar configs로 관측한다.
    # LR은 HR dB image의 post-FFT downsample이 아니라 낮은 BW/chirp 수의 RDM이다.
    y_map = simulate_rdm_map(hr_radar, targets, snr_db, sample_seed, HR_RANGE)
    x_map_lr = simulate_rdm_map(lr_radar, targets, snr_db, sample_seed + 10_000_000, LR_RANGE)

    # 3. Peak mask: mark target locations in HR grid
    peak_mask = np.zeros((HR_DOPPLER, HR_RANGE), dtype=np.float32)
    vel_axis = doppler_axis(hr_radar)
    range_res = hr_radar.range_res

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
        y_map[np.newaxis].astype(np.float32),      # (1, 64, 64)
        peak_mask[np.newaxis].astype(np.float32),  # (1, 64, 64)
        n_tgt,
        snr_db,
    )


def generate_split(
    name: str,
    n: int,
    hr_radar: FMCWRadar,
    lr_radar: FMCWRadar,
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
        x_lr, y_hr, peak_mask, n_tgt, snr_db = generate_one_sample(
            hr_radar, lr_radar, rng, sample_seed
        )
        x_lr_list.append(x_lr)
        y_hr_list.append(y_hr)
        mask_list.append(peak_mask)
        n_targets_list.append(n_tgt)
        snr_list.append(snr_db)

        if (i + 1) % 1000 == 0:
            print(f"    {i+1}/{n}")

    arrays = {
        "x_lr": np.stack(x_lr_list),
        "y_hr": np.stack(y_hr_list),
        "peak_mask": np.stack(mask_list),
        "n_targets": np.array(n_targets_list, dtype=np.int32),
        "snr_db": np.array(snr_list, dtype=np.float32),
    }
    attrs = radar_config_attrs(hr_radar, lr_radar) | {"split": name, "n_samples": n}
    path = out_dir / f"{name}.h5"
    with h5py.File(path, "w") as f:
        for key, arr in arrays.items():
            f.create_dataset(key, data=arr, compression="gzip", compression_opts=4)
        for key, value in attrs.items():
            f.attrs[key] = value
    print(f"  Saved {path.name}: {', '.join(f'{k} {v.shape}' for k, v in arrays.items())}")
    print(
        "    attrs: "
        f"LR ΔR={attrs['lr_range_bin_spacing_m']:.3f} m, "
        f"HR ΔR={attrs['hr_range_bin_spacing_m']:.3f} m, "
        f"LR Δv={attrs['lr_doppler_bin_spacing_mps']:.3f} m/s, "
        f"HR Δv={attrs['hr_doppler_bin_spacing_mps']:.3f} m/s"
    )


def main():
    parser = base_parser("P09 RD Super-Resolution 데이터 생성")
    args = parser.parse_args()

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)

    hr_radar = make_hr_radar()
    lr_radar = make_lr_radar()
    print("=== HR radar ===")
    hr_radar.print_params()
    print("\n=== LR radar ===")
    lr_radar.print_params()

    if args.smoke:
        splits = [("train", 256), ("val", 64), ("test", 64)]
    else:
        splits = [("train", 12000), ("val", 2000), ("test", 2000)]

    print(f"\nGenerating {'smoke' if args.smoke else 'full'} dataset...")
    for name, n in splits:
        seed_offset = {"train": 0, "val": 100000, "test": 200000}[name]
        generate_split(name, n, hr_radar, lr_radar, rng, out_dir, seed_offset=seed_offset)

    print("\nDone. Files written to:", out_dir)


if __name__ == "__main__":
    main()
