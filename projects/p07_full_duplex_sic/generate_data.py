"""P07 Full-Duplex Self-Interference Cancellation — 데이터 생성

TX 기준 신호와 수신 혼합 신호로부터 자기 간섭(SI) 신호를 추정하기 위한
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

# 신호 길이
N_SAMPLES = 512


# ---------------------------------------------------------------------------
# 신호 생성 함수
# ---------------------------------------------------------------------------

def generate_chirp(N: int, rng: np.random.Generator) -> np.ndarray:
    """랜덤 초기 위상을 가진 FMCW chirp 신호 생성.

    Returns
    -------
    chirp : ndarray, shape (N,) complex64
    """
    t = np.arange(N)
    # 랜덤 chirp rate (정규화된 주파수 단위)
    f0 = rng.uniform(0.0, 0.1)
    f1 = rng.uniform(0.3, 0.45)
    k = (f1 - f0) / N
    phase0 = rng.uniform(0, 2 * np.pi)
    chirp = np.exp(1j * (2 * np.pi * (f0 * t + 0.5 * k * t ** 2) + phase0))
    return chirp.astype(np.complex64)


def apply_si_channel(tx: np.ndarray, rng: np.random.Generator,
                     n_taps_range: tuple[int, int] = (2, 5),
                     nonlinear: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """자기 간섭(SI) 채널 적용 (FIR 필터 + 선택적 비선형).

    Parameters
    ----------
    tx : ndarray, shape (N,) complex
    rng : Generator
    n_taps_range : (min_taps, max_taps)
    nonlinear : bool
        True면 3차 비선형항 추가

    Returns
    -------
    y_si : ndarray, shape (N,) complex — SI 컴포넌트
    h : ndarray — FIR 계수 (디버그용)
    """
    n_taps = rng.integers(n_taps_range[0], n_taps_range[1] + 1)
    # 복소 FIR 계수 (지수감쇠 포락선)
    decay = rng.uniform(0.3, 0.7)
    h_real = rng.standard_normal(n_taps) * decay ** np.arange(n_taps)
    h_imag = rng.standard_normal(n_taps) * decay ** np.arange(n_taps)
    h = (h_real + 1j * h_imag).astype(np.complex64)
    # 전력 정규화
    h = h / (np.sqrt(np.sum(np.abs(h) ** 2)) + 1e-12)

    # FIR 컨볼루션 (same length)
    N = len(tx)
    y_si = np.zeros(N, dtype=np.complex64)
    for k, coef in enumerate(h):
        if k == 0:
            y_si += coef * tx
        else:
            y_si[k:] += coef * tx[:-k]

    # 선택적 3차 비선형
    if nonlinear:
        alpha3 = rng.uniform(0.05, 0.2)
        y_si += alpha3 * tx * np.abs(tx) ** 2

    return y_si, h


def generate_target_echo(N: int, rng: np.random.Generator) -> np.ndarray:
    """간단한 표적 에코 신호 (지연 + 도플러 시프트).

    Returns
    -------
    echo : ndarray, shape (N,) complex64
    """
    delay = rng.integers(5, 50)
    doppler = rng.uniform(-0.02, 0.02)  # 정규화 주파수

    # 임의 표적 반사 계수
    amp = rng.uniform(0.1, 0.5) * np.exp(1j * rng.uniform(0, 2 * np.pi))

    echo = np.zeros(N, dtype=np.complex64)
    t = np.arange(N)
    # 지연된 chirp 대신 단순 협대역 신호 사용 (교육 목적)
    base = amp * np.exp(1j * 2 * np.pi * doppler * t)
    echo[delay:] = base[:N - delay]
    return echo


def make_sample(rng: np.random.Generator,
                sir_db: float, snr_db: float) -> dict[str, np.ndarray]:
    """단일 데이터 샘플 생성.

    Returns dict with keys: tx_ref, rx_mix, y_si, y_clean
    """
    N = N_SAMPLES

    # TX 기준 신호 (chirp)
    tx = generate_chirp(N, rng)

    # SI 채널 (일부 샘플에서 비선형 적용)
    nonlinear = rng.random() < 0.3
    y_si, _ = apply_si_channel(tx, rng, nonlinear=nonlinear)

    # 표적 에코
    echo = generate_target_echo(N, rng)

    # SIR 기반 전력 조정
    # SIR = P_si / P_echo → P_si = 10^(sir_db/10) * P_echo
    p_si = np.mean(np.abs(y_si) ** 2) + 1e-20
    p_echo = np.mean(np.abs(echo) ** 2) + 1e-20
    sir_lin = 10 ** (sir_db / 10.0)
    # y_si 전력을 기준으로 echo 조정
    scale_echo = np.sqrt(p_si / (sir_lin * p_echo + 1e-20))
    echo = echo * scale_echo

    # 잡음 (SNR 기준: SNR = P_si / P_noise)
    snr_lin = 10 ** (snr_db / 10.0)
    p_noise = p_si / snr_lin
    noise_std = np.sqrt(p_noise / 2)
    noise = noise_std * (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(np.complex64)

    # 수신 혼합 신호
    y_clean = echo + noise  # 표적 에코 + 잡음
    rx_mix = y_si + y_clean

    # (2, N) 텐서로 변환: [real, imag]
    def to_2ch(sig):
        return np.stack([sig.real, sig.imag], axis=0).astype(np.float32)

    return {
        "tx_ref": to_2ch(tx),       # (2, 512)
        "rx_mix": to_2ch(rx_mix),   # (2, 512)
        "y_si": to_2ch(y_si),       # (2, 512) — 추정 목표
        "y_clean": to_2ch(y_clean), # (2, 512) — 클린 신호 (eval용)
    }


def generate_split(n: int, seed: int,
                   sir_range: tuple[float, float] = (-10.0, 20.0),
                   snr_range: tuple[float, float] = (5.0, 25.0)) -> dict[str, np.ndarray]:
    """N개 샘플 생성.

    Returns
    -------
    dict with arrays of shape (N, 2, 512) for tx_ref/rx_mix/y_si/y_clean,
    and (N,) for sir_db/snr_db
    """
    rng = np.random.default_rng(seed)

    tx_refs, rx_mixes, y_sis, y_cleans = [], [], [], []
    sir_dbs, snr_dbs = [], []

    for _ in range(n):
        sir_db = rng.uniform(sir_range[0], sir_range[1])
        snr_db = rng.uniform(snr_range[0], snr_range[1])
        sample = make_sample(rng, sir_db, snr_db)
        tx_refs.append(sample["tx_ref"])
        rx_mixes.append(sample["rx_mix"])
        y_sis.append(sample["y_si"])
        y_cleans.append(sample["y_clean"])
        sir_dbs.append(sir_db)
        snr_dbs.append(snr_db)

    return {
        "tx_ref": np.stack(tx_refs, axis=0).astype(np.float32),    # (N, 2, 512)
        "rx_mix": np.stack(rx_mixes, axis=0).astype(np.float32),   # (N, 2, 512)
        "y_si": np.stack(y_sis, axis=0).astype(np.float32),        # (N, 2, 512)
        "y_clean": np.stack(y_cleans, axis=0).astype(np.float32),  # (N, 2, 512)
        "sir_db": np.array(sir_dbs, dtype=np.float32),             # (N,)
        "snr_db": np.array(snr_dbs, dtype=np.float32),             # (N,)
    }


def main():
    p = argparse.ArgumentParser(description="P07: Full-Duplex SIC 데이터 생성")
    p.add_argument("--smoke", action="store_true", help="소규모 스모크 테스트 (256/64/64)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    if args.smoke:
        splits = {"train": 256, "val": 64, "test": 64}
    else:
        splits = {"train": 18_000, "val": 3_000, "test": 3_000}

    seed_everything(args.seed)

    for split, n in splits.items():
        print(f"Generating {split}: {n} samples...")
        seed_offset = {"train": 0, "val": 1, "test": 2}[split]
        data = generate_split(n, seed=args.seed + seed_offset)
        save_hdf5(data_dir / f"{split}.h5", **data)

    print("Done.")


if __name__ == "__main__":
    main()
