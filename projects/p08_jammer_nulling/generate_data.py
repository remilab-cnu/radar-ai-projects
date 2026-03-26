"""P08 Jammer Null Steering — 데이터 생성

8-element ULA 수신 공분산 행렬로부터 재머 방향(DoA)을 회귀 추정하기 위한
학습 데이터를 생성한다.

Usage:
    python generate_data.py --smoke
    python generate_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import numpy as np
from common.hdf5_io import save_hdf5
from common.seed import seed_everything
from shared.doa_utils import steering_vector

N_RX = 8          # ULA 안테나 수
D_OVER_LAM = 0.5  # 안테나 간격 / 파장


# ---------------------------------------------------------------------------
# 시나리오 생성
# ---------------------------------------------------------------------------

def generate_sample(rng: np.random.Generator,
                    n_jammers: int,
                    look_angle_deg: float,
                    jnr_db: float,
                    snr_db: float,
                    n_snapshots: int = 100) -> dict:
    """단일 재머 시나리오 생성.

    Parameters
    ----------
    rng : Generator
    n_jammers : int — 재머 수 (1 or 2)
    look_angle_deg : float — 원하는 신호 방향 [deg]
    jnr_db : float — Jammer-to-Noise Ratio [dB]
    snr_db : float — Signal-to-Noise Ratio [dB]
    n_snapshots : int — 스냅샷 수 (공분산 추정용)

    Returns
    -------
    dict with keys: cov (2,8,8), look_angle_deg (scalar),
                    jammer_angle_deg (scalar, stronger jammer),
                    jnr_db, snr_db, n_jammers
    """
    # 재머 각도: look_angle과 최소 10도 이상 분리
    jammer_angles = _sample_jammer_angles(rng, n_jammers, look_angle_deg,
                                          min_sep=10.0)

    # 원하는 신호 스티어링 벡터
    a_s = steering_vector(look_angle_deg, N_RX, D_OVER_LAM)  # (8,)

    # 재머 스티어링 벡터
    jnr_lin = 10 ** (jnr_db / 10.0)
    snr_lin = 10 ** (snr_db / 10.0)

    # 재머 전력 (2개일 때: 첫 번째가 더 강함)
    if n_jammers == 1:
        jammer_powers = [jnr_lin]
    else:
        # 두 번째 재머는 1~6dB 약하게
        power_diff = rng.uniform(1.0, 6.0)
        jammer_powers = [jnr_lin, jnr_lin / (10 ** (power_diff / 10.0))]

    # 수신 신호 시뮬레이션: X = A_j * S_j + a_s * s + noise
    X = np.zeros((N_RX, n_snapshots), dtype=np.complex64)

    # 재머 신호
    for j_idx, (j_angle, j_pwr) in enumerate(zip(jammer_angles, jammer_powers)):
        a_j = steering_vector(j_angle, N_RX, D_OVER_LAM)  # (8,)
        s_j = np.sqrt(j_pwr / 2) * (
            rng.standard_normal(n_snapshots) +
            1j * rng.standard_normal(n_snapshots)
        ).astype(np.complex64)
        X += np.outer(a_j, s_j)

    # 원하는 신호
    s_desired = np.sqrt(snr_lin / 2) * (
        rng.standard_normal(n_snapshots) +
        1j * rng.standard_normal(n_snapshots)
    ).astype(np.complex64)
    X += np.outer(a_s, s_desired)

    # AWGN (noise power = 1)
    noise = (
        rng.standard_normal((N_RX, n_snapshots)) +
        1j * rng.standard_normal((N_RX, n_snapshots))
    ).astype(np.complex64) / np.sqrt(2)
    X += noise

    # 샘플 공분산 행렬
    R = (X @ X.conj().T) / n_snapshots  # (8, 8)

    # Frobenius norm 정규화
    R_norm = R / (np.linalg.norm(R, 'fro') + 1e-10)

    # (2, 8, 8) real/imag 스택
    cov = np.stack([R_norm.real, R_norm.imag], axis=0).astype(np.float32)

    # 강한 재머 각도 (회귀 타겟)
    primary_jammer_angle = float(jammer_angles[0])  # 항상 첫 번째가 강한 재머

    return {
        "cov": cov,                                        # (2, 8, 8)
        "look_angle_deg": np.float32(look_angle_deg),
        "jammer_angle_deg": np.float32(primary_jammer_angle),
        "jnr_db": np.float32(jnr_db),
        "snr_db": np.float32(snr_db),
        "n_jammers": np.int32(n_jammers),
    }


def _sample_jammer_angles(rng: np.random.Generator,
                          n_jammers: int,
                          look_angle: float,
                          min_sep: float = 10.0,
                          angle_range: tuple[float, float] = (-60.0, 60.0),
                          max_attempts: int = 200) -> list[float]:
    """look_angle과 최소 min_sep 이상 분리된 재머 각도 샘플링."""
    angles = []
    for _ in range(n_jammers):
        for _ in range(max_attempts):
            a = rng.uniform(angle_range[0], angle_range[1])
            # look_angle과 분리
            if abs(a - look_angle) < min_sep:
                continue
            # 기존 재머들과 분리
            if all(abs(a - prev) >= min_sep for prev in angles):
                angles.append(float(a))
                break
        else:
            # fallback: look_angle에서 ±(min_sep + 5) 위치
            sign = 1.0 if len(angles) % 2 == 0 else -1.0
            fallback = look_angle + sign * (min_sep + 5.0)
            fallback = np.clip(fallback, angle_range[0], angle_range[1])
            angles.append(float(fallback))
    return angles


def generate_split(n: int, seed: int,
                   jnr_range: tuple[float, float] = (10.0, 40.0),
                   snr_range: tuple[float, float] = (0.0, 20.0)) -> dict:
    """N개 샘플 생성.

    Returns
    -------
    dict with arrays:
      cov         : (N, 2, 8, 8)
      look_angle_deg : (N,)
      jammer_angle_deg : (N,)
      jnr_db      : (N,)
      snr_db      : (N,)
      n_jammers   : (N,)
    """
    rng = np.random.default_rng(seed)

    covs, look_angles, jammer_angles = [], [], []
    jnr_dbs, snr_dbs, n_jammers_list = [], [], []

    for i in range(n):
        jnr_db = rng.uniform(jnr_range[0], jnr_range[1])
        snr_db = rng.uniform(snr_range[0], snr_range[1])
        look_angle = rng.uniform(-50.0, 50.0)
        n_jammers = int(rng.integers(1, 3))  # 1 or 2

        sample = generate_sample(rng, n_jammers, look_angle, jnr_db, snr_db)

        covs.append(sample["cov"])
        look_angles.append(sample["look_angle_deg"])
        jammer_angles.append(sample["jammer_angle_deg"])
        jnr_dbs.append(sample["jnr_db"])
        snr_dbs.append(sample["snr_db"])
        n_jammers_list.append(sample["n_jammers"])

    return {
        "cov": np.stack(covs, axis=0).astype(np.float32),              # (N, 2, 8, 8)
        "look_angle_deg": np.array(look_angles, dtype=np.float32),     # (N,)
        "jammer_angle_deg": np.array(jammer_angles, dtype=np.float32), # (N,)
        "jnr_db": np.array(jnr_dbs, dtype=np.float32),                 # (N,)
        "snr_db": np.array(snr_dbs, dtype=np.float32),                 # (N,)
        "n_jammers": np.array(n_jammers_list, dtype=np.int32),         # (N,)
    }


def main():
    p = argparse.ArgumentParser(description="P08: Jammer Null Steering 데이터 생성")
    p.add_argument("--smoke", action="store_true", help="소규모 스모크 테스트 (256/64/64)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    if args.smoke:
        splits = {"train": 256, "val": 64, "test": 64}
    else:
        splits = {"train": 24_000, "val": 4_000, "test": 4_000}

    seed_everything(args.seed)

    for split, n in splits.items():
        print(f"Generating {split}: {n} samples...")
        seed_offset = {"train": 0, "val": 1, "test": 2}[split]
        data = generate_split(n, seed=args.seed + seed_offset)
        save_hdf5(data_dir / f"{split}.h5", **data)

    print("Done.")


if __name__ == "__main__":
    main()
