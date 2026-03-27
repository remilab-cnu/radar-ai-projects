"""P10 Near-Field Source Localization — 데이터 생성

near-field / far-field 혼합 시나리오에서 ULA array snapshot 생성.
- Near-field: 구면파 모델 (range 0.5~5 m)
- Far-field:  평면파 모델 (range > 10 m → doa_utils.steering_vector)

출력 텐서:
- x          : (N, 2, 8, 64) — real/imag array snapshots
- near_label : (N,)          — 1=near-field, 0=far-field
- angle_deg  : (N,)          — 소스 각도 [deg] (첫 번째 소스 기준)
- range_m    : (N,)          — 거리 [m] (near-field만 의미 있음)
- snr_db     : (N,)
- n_sources  : (N,)

사용법:
    python generate_data.py --smoke
    python generate_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from common.cli import base_parser
from shared.doa_utils import steering_vector
from common.hdf5_io import save_hdf5
from common.seed import seed_everything

# ─── 어레이 파라미터 ────────────────────────────────────────────────────────────
N_RX = 8          # 안테나 수
D_OVER_LAM = 0.5  # d = λ/2
N_SNAPSHOTS = 64  # 시간 샘플 수

NEAR_RANGE_MIN = 0.5   # [m]
NEAR_RANGE_MAX = 5.0   # [m]
FAR_RANGE_MIN = 10.0   # [m] (far-field 기준)
ANGLE_MIN = -60.0      # [deg]
ANGLE_MAX = 60.0       # [deg]
MIN_ANGLE_SEP = 10.0   # 최소 각도 간격 [deg]


def nearfield_steering_vector(
    theta_deg: float,
    r_m: float,
    N: int,
    d_over_lam: float = 0.5,
) -> np.ndarray:
    """구면파 근거리 steering vector.

    각 소자 x_n = n * d (n=0..N-1), 파장 λ=1 (d_over_lam=0.5 기준)
    phase_n = -2π/λ * (dist_n - r)
    dist_n = sqrt((x_n - r*sin(θ))^2 + (r*cos(θ))^2)

    Parameters
    ----------
    theta_deg : float
        소스 각도 [deg]
    r_m : float
        소스 거리 [m] (단위는 파장 단위로 정규화)
    N : int
        안테나 수
    d_over_lam : float
        소자 간격 / 파장

    Returns
    -------
    a : (N,) complex128
    """
    theta = np.radians(theta_deg)
    # 소자 위치: x_n = n * d_over_lam  (파장 = 1 단위)
    x_n = np.arange(N) * d_over_lam        # (N,)
    # 소스 위치 (파장 단위)
    src_x = r_m * np.sin(theta)
    src_y = r_m * np.cos(theta)
    # 소자-소스 거리
    dist_n = np.sqrt((x_n - src_x) ** 2 + src_y ** 2)  # (N,)
    # 위상차: -2π(dist_n - r_m)  [파장=1이므로 2π/λ = 2π]
    phase = -2 * np.pi * (dist_n - r_m)
    return np.exp(1j * phase)


def sample_angles(K: int, rng: np.random.Generator) -> np.ndarray:
    """최소 간격 보장하며 K개 각도 샘플링."""
    for _ in range(200):
        angles = rng.uniform(ANGLE_MIN, ANGLE_MAX, size=K)
        angles = np.sort(angles)
        if K == 1 or np.all(np.diff(angles) >= MIN_ANGLE_SEP):
            return angles
    return np.linspace(ANGLE_MIN + 10, ANGLE_MAX - 10, K)


def generate_one_sample(
    is_near: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, float, float, float, int]:
    """단일 샘플 생성.

    Returns
    -------
    x      : (2, N_RX, N_SNAPSHOTS) float32
    label  : int (1=near, 0=far)
    angle  : float [deg]
    range_ : float [m]
    snr_db : float
    n_src  : int
    """
    n_src = int(rng.integers(1, 3))   # 1 or 2 sources
    snr_db = float(rng.uniform(0.0, 20.0))
    snr_lin = 10 ** (snr_db / 10.0)

    angles = sample_angles(n_src, rng)

    # Primary source angle & range (for label)
    primary_angle = float(angles[0])

    if is_near:
        range_m = float(rng.uniform(NEAR_RANGE_MIN, NEAR_RANGE_MAX))
        label = 1
    else:
        range_m = float(rng.uniform(FAR_RANGE_MIN, FAR_RANGE_MIN * 5))
        label = 0

    # Steering vectors
    A_cols = []
    for k, theta in enumerate(angles):
        if is_near:
            a = nearfield_steering_vector(theta, range_m, N_RX, D_OVER_LAM)
        else:
            # far-field: plane wave
            a = steering_vector(float(theta), N_RX, D_OVER_LAM)
        A_cols.append(a)

    A = np.stack(A_cols, axis=1)  # (N_RX, n_src)

    # Source signals: uncorrelated
    S = np.sqrt(snr_lin / n_src) * (
        rng.standard_normal((n_src, N_SNAPSHOTS)) +
        1j * rng.standard_normal((n_src, N_SNAPSHOTS))
    ) / np.sqrt(2)

    # Noise
    noise = (
        rng.standard_normal((N_RX, N_SNAPSHOTS)) +
        1j * rng.standard_normal((N_RX, N_SNAPSHOTS))
    ) / np.sqrt(2)

    X = A @ S + noise  # (N_RX, N_SNAPSHOTS)

    # Normalize by Frobenius norm of the array snapshot matrix
    norm = np.linalg.norm(X, 'fro') + 1e-10
    X = X / norm

    # Stack real/imag: (2, N_RX, N_SNAPSHOTS)
    x = np.stack([X.real, X.imag], axis=0).astype(np.float32)

    return x, label, primary_angle, range_m, snr_db, n_src


def generate_split(
    name: str,
    n: int,
    rng: np.random.Generator,
    out_dir: Path,
):
    """데이터셋 분할 생성 및 저장."""
    print(f"  Generating {name} split ({n} samples, 50/50 near/far)...")

    x_list, labels, angles, ranges, snrs, n_srcs = [], [], [], [], [], []

    n_near = n // 2
    n_far = n - n_near

    for i in range(n):
        is_near = (i < n_near)
        x, lbl, ang, rng_m, snr, n_src = generate_one_sample(is_near, rng)
        x_list.append(x)
        labels.append(lbl)
        angles.append(ang)
        ranges.append(rng_m)
        snrs.append(snr)
        n_srcs.append(n_src)

        if (i + 1) % 2000 == 0:
            print(f"    {i+1}/{n}")

    save_hdf5(
        out_dir / f"{name}.h5",
        x=np.stack(x_list),
        near_label=np.array(labels, dtype=np.int32),
        angle_deg=np.array(angles, dtype=np.float32),
        range_m=np.array(ranges, dtype=np.float32),
        snr_db=np.array(snrs, dtype=np.float32),
        n_sources=np.array(n_srcs, dtype=np.int32),
    )


def main():
    parser = base_parser("P10 Near-Field 데이터 생성")
    args = parser.parse_args()

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)

    if args.smoke:
        splits = [("train", 256), ("val", 64), ("test", 64)]
    else:
        splits = [("train", 20000), ("val", 4000), ("test", 4000)]

    print(f"Generating {'smoke' if args.smoke else 'full'} dataset...")
    print(f"  Config: N_RX={N_RX}, N_SNAPSHOTS={N_SNAPSHOTS}")
    print(f"  Near-field range: {NEAR_RANGE_MIN}~{NEAR_RANGE_MAX} m")
    print(f"  Far-field range:  >{FAR_RANGE_MIN} m")

    for name, n in splits:
        generate_split(name, n, rng, out_dir)

    print("\nDone. Files written to:", out_dir)


if __name__ == "__main__":
    main()
