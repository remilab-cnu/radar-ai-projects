"""P05 Neural CFAR — 데이터 생성

Range-Doppler Map에서 15x15 패치를 추출하여
표적 존재 여부를 분류하는 학습 데이터를 생성한다.

HDF5 keys:
  x                     (N, 2, 15, 15)  — ch0: dB magnitude, ch1: locally normalized
  patch_power           (N, 15, 15)     — linear RDM power for CA-CFAR baseline
  y                     (N,)             — binary label (1=target CUT, 0=clutter/noise CUT)
  snr_db                (N,)             — 해당 샘플의 SNR 레벨
  cut_range_bin         (N,)             — CUT range-bin index
  cut_doppler_bin       (N,)             — CUT Doppler-bin index
  target_distance_bins  (N,)             — nearest target-bin distance from CUT
  clutter_type          (N,)             — fixed-width string clutter type

Splits: train 24K / val 6K / test 6K (smoke: 256/64/64)
Balance: 50/50, 6 SNR bins (0,5,10,15,20,25 dB)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from common.cli import base_parser
from shared.fmcw_simulator import FMCWRadar, range_doppler_map
from shared.clutter_model import generate_scene_with_clutter
from common.hdf5_io import save_hdf5
from common.seed import seed_everything

PATCH = 15
HALF = PATCH // 2  # 7

SNR_BINS = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]  # 6 bins


def make_radar() -> FMCWRadar:
    return FMCWRadar(fc=77e9, bw=1e9, T_chirp=50e-6, N_chirps=128, fs=10e6, N_rx=1)


def extract_patch(rdm_mag: np.ndarray, r_bin: int, d_bin: int) -> np.ndarray | None:
    """15x15 패치 추출. 경계를 벗어나면 None 반환."""
    Nd, Nr = rdm_mag.shape
    if (r_bin < HALF or r_bin >= Nr - HALF or
            d_bin < HALF or d_bin >= Nd - HALF):
        return None
    patch = rdm_mag[d_bin - HALF: d_bin + HALF + 1,
                    r_bin - HALF: r_bin + HALF + 1]  # (15, 15)
    return patch


def nearest_target_distance_bins(r_bin: int, d_bin: int, target_info: list[dict]) -> float:
    """Return Euclidean bin distance from the CUT to the nearest true target bin."""
    if not target_info:
        return float("inf")
    distances = [
        np.hypot(float(d_bin - info["doppler_bin"]), float(r_bin - info["range_bin"]))
        for info in target_info
    ]
    return float(min(distances))


def rdm_products(signal: np.ndarray, radar: FMCWRadar) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized network magnitude image and native linear-power RDM."""
    rdm = range_doppler_map(signal[0:1])
    rdm_half = rdm[0, :, :radar.N_samples // 2]
    mag = np.abs(rdm_half)
    linear_power = (mag ** 2).astype(np.float32)
    noise_floor = np.median(mag)
    mag_db = 20 * np.log10(mag / (noise_floor + 1e-30) + 1e-30)
    mag_norm = np.clip(mag_db, -20.0, 40.0) / 60.0 + 1.0 / 3.0
    return mag_norm.astype(np.float32), linear_power


def patch_to_channels(patch: np.ndarray) -> np.ndarray:
    """패치 → 2채널 (N, 2, 15, 15).

    ch0: 전역 noise-floor 기준 dB magnitude (generate 시 이미 normalized)
    ch1: 패치 내 로컬 정규화 magnitude
    """
    # ch0: patch는 이미 noise-floor ref log-mag (clutter_model 출력)
    ch0 = patch.astype(np.float32)

    # ch1: 패치 내에서 [0,1] min-max 정규화
    p_min, p_max = ch0.min(), ch0.max()
    ch1 = (ch0 - p_min) / (p_max - p_min + 1e-8)

    return np.stack([ch0, ch1], axis=0)  # (2, 15, 15)


def generate_split(
    n_target: int,
    n_noise: int,
    snr_bins: list[float],
    radar: FMCWRadar,
    rng: np.random.Generator,
    seed_base: int,
) -> tuple[np.ndarray, ...]:
    """표적/비표적 패치 생성.

    각 SNR bin에서 n_target//len(bins) 표적과 n_noise//len(bins) 비표적을 생성한다.
    """
    xs, ys, snrs = [], [], []
    patch_powers, cut_range_bins, cut_doppler_bins = [], [], []
    target_distances, clutter_types = [], []

    per_bin_tgt = max(1, n_target // len(snr_bins))
    per_bin_noise = max(1, n_noise // len(snr_bins))

    Nr_half = radar.N_samples // 2
    Nd = radar.N_chirps

    for bin_idx, snr_db in enumerate(snr_bins):
        seed_offset = seed_base + bin_idx * 10000

        # --- 표적 패치 ---
        collected_tgt = 0
        scene_idx = 0
        while collected_tgt < per_bin_tgt:
            scene_seed = seed_offset + scene_idx
            scene_idx += 1

            # 1~3개 표적 배치
            n_tgt = rng.integers(1, 4)
            targets = []
            for _ in range(n_tgt):
                r = rng.uniform(5.0, radar.max_range * 0.85)
                v = rng.uniform(-radar.max_vel * 0.8, radar.max_vel * 0.8)
                targets.append({'range': r, 'velocity': v, 'rcs': 1.0})

            signal, target_mask, target_info = generate_scene_with_clutter(
                radar, targets,
                snr_db=snr_db,
                clutter_type='mixed',
                clutter_power_db=-10.0,
                seed=int(scene_seed),
            )

            mag_norm, linear_power = rdm_products(signal, radar)

            for info in target_info:
                if collected_tgt >= per_bin_tgt:
                    break
                r_bin = info['range_bin']
                d_bin = info['doppler_bin']
                patch = extract_patch(mag_norm, r_bin, d_bin)
                power_patch = extract_patch(linear_power, r_bin, d_bin)
                if patch is None or power_patch is None:
                    continue
                xs.append(patch_to_channels(patch))
                patch_powers.append(power_patch.astype(np.float32))
                ys.append(1)
                snrs.append(snr_db)
                cut_range_bins.append(r_bin)
                cut_doppler_bins.append(d_bin)
                target_distances.append(nearest_target_distance_bins(r_bin, d_bin, target_info))
                clutter_types.append("mixed")
                collected_tgt += 1

            if scene_idx > per_bin_tgt * 5 + 20:
                break  # 무한루프 방지

        # --- 비표적 패치 (noise/clutter cell) ---
        collected_noise = 0
        scene_idx = 0
        while collected_noise < per_bin_noise:
            scene_seed = seed_offset + 5000 + scene_idx
            scene_idx += 1

            targets = [{'range': rng.uniform(5.0, radar.max_range * 0.85),
                        'velocity': rng.uniform(-radar.max_vel * 0.8, radar.max_vel * 0.8),
                        'rcs': 1.0}]

            signal, target_mask, target_info = generate_scene_with_clutter(
                radar, targets,
                snr_db=snr_db,
                clutter_type='mixed',
                clutter_power_db=-10.0,
                seed=int(scene_seed),
            )

            mag_norm, linear_power = rdm_products(signal, radar)

            # 표적 마스크가 0인 셀에서 랜덤 추출
            zero_cells = np.argwhere(target_mask == 0)
            if len(zero_cells) == 0:
                continue

            rng.shuffle(zero_cells)
            for cell in zero_cells:
                if collected_noise >= per_bin_noise:
                    break
                d_bin, r_bin = int(cell[0]), int(cell[1])
                patch = extract_patch(mag_norm, r_bin, d_bin)
                power_patch = extract_patch(linear_power, r_bin, d_bin)
                if patch is None or power_patch is None:
                    continue
                xs.append(patch_to_channels(patch))
                patch_powers.append(power_patch.astype(np.float32))
                ys.append(0)
                snrs.append(snr_db)
                cut_range_bins.append(r_bin)
                cut_doppler_bins.append(d_bin)
                target_distances.append(nearest_target_distance_bins(r_bin, d_bin, target_info))
                clutter_types.append("mixed")
                collected_noise += 1

            if scene_idx > per_bin_noise * 3 + 20:
                break

    x_arr = np.stack(xs, axis=0).astype(np.float32)
    power_arr = np.stack(patch_powers, axis=0).astype(np.float32)
    y_arr = np.array(ys, dtype=np.float32)
    snr_arr = np.array(snrs, dtype=np.float32)
    cut_r_arr = np.array(cut_range_bins, dtype=np.int32)
    cut_d_arr = np.array(cut_doppler_bins, dtype=np.int32)
    target_dist_arr = np.array(target_distances, dtype=np.float32)
    clutter_arr = np.array(clutter_types, dtype="S16")

    # 셔플
    idx = rng.permutation(len(x_arr))
    return (
        x_arr[idx],
        power_arr[idx],
        y_arr[idx],
        snr_arr[idx],
        cut_r_arr[idx],
        cut_d_arr[idx],
        target_dist_arr[idx],
        clutter_arr[idx],
    )


def main():
    parser = base_parser("P05 Neural CFAR — 데이터 생성")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)

    base = Path(__file__).parent
    out_dir = Path(args.out_dir) if args.out_dir else base / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        splits = {"train": (128, 128), "val": (32, 32), "test": (32, 32)}
    else:
        splits = {"train": (12000, 12000), "val": (3000, 3000), "test": (3000, 3000)}

    radar = make_radar()

    for split_name, (n_tgt, n_noise) in splits.items():
        print(f"\n[{split_name}] Generating {n_tgt} target + {n_noise} no-target patches...")
        seed_base = args.seed + {"train": 0, "val": 100000, "test": 200000}[split_name]
        x, patch_power, y, snr, cut_r, cut_d, target_dist, clutter = generate_split(
            n_tgt, n_noise, SNR_BINS, radar, np.random.default_rng(seed_base), seed_base
        )
        print(f"  x: {x.shape}, patch_power: {patch_power.shape}, y: {y.shape}, balance: {y.mean():.3f}")
        save_hdf5(
            out_dir / f"{split_name}.h5",
            x=x,
            patch_power=patch_power,
            y=y,
            snr_db=snr,
            cut_range_bin=cut_r,
            cut_doppler_bin=cut_d,
            target_distance_bins=target_dist,
            clutter_type=clutter,
        )

    print("\nData generation complete.")


if __name__ == "__main__":
    main()
