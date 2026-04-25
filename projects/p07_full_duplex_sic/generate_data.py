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

import numpy as np
from common.cli import base_parser
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
    y_linear = np.zeros(N, dtype=np.complex64)
    for k, coef in enumerate(h):
        if k == 0:
            y_linear += coef * tx
        else:
            y_linear[k:] += coef * tx[:-k]

    y_si = y_linear.copy()

    # 선택적 3차 비선형
    if nonlinear:
        alpha3 = rng.uniform(0.1, 0.3) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        # A constant-envelope chirp would make tx*|tx|^2 indistinguishable
        # from a linear path. Add a small cubic phase-leakage term so NLMS has
        # a genuine nonlinear residual to expose in the teaching baseline.
        y_si += alpha3 * tx ** 3

    return y_si, h


def generate_target_echo(tx_ref: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """TX 기준 파형의 지연 + 도플러 스케일 표적 에코.

    Returns
    -------
    echo : ndarray, shape (N,) complex64
    """
    N = len(tx_ref)
    delay = rng.integers(5, 50)
    doppler = rng.uniform(-0.02, 0.02)  # 정규화 주파수

    # 임의 표적 반사 계수
    amp = rng.uniform(0.1, 0.5) * np.exp(1j * rng.uniform(0, 2 * np.pi))

    echo = np.zeros(N, dtype=np.complex64)
    t = np.arange(N - delay)
    doppler_phase = np.exp(1j * 2 * np.pi * doppler * t)
    echo[delay:] = amp * tx_ref[:N - delay] * doppler_phase
    return echo


def make_sample(rng: np.random.Generator,
                isr_db: float | None = None,
                snr_db: float | None = None,
                *,
                sir_db: float | None = None,
                force_nonlinear: bool | None = None) -> dict[str, np.ndarray]:
    """단일 데이터 샘플 생성.

    Returns dict with keys: tx_ref, rx_mix, y_si, y_clean
    """
    if isr_db is None:
        if sir_db is None:
            raise ValueError("make_sample requires isr_db (or legacy sir_db).")
        isr_db = sir_db
    elif sir_db is not None and not np.isclose(isr_db, sir_db):
        raise ValueError("isr_db and legacy sir_db disagree.")
    if snr_db is None:
        raise ValueError("make_sample requires snr_db.")

    N = N_SAMPLES

    # TX 기준 신호 (chirp)
    tx = generate_chirp(N, rng)

    # SI 채널 (일부 샘플에서 비선형 적용)
    nonlinear = (rng.random() < 0.3) if force_nonlinear is None else force_nonlinear
    y_si, _ = apply_si_channel(tx, rng, nonlinear=nonlinear)

    # 표적 에코: 같은 TX 기준 파형의 지연/도플러 변환
    echo = generate_target_echo(tx, rng)

    # ISR(SI-to-echo power ratio) 기반 전력 조정.
    # Historical key name is "sir_db", but the stored value is P_si / P_echo.
    p_si = np.mean(np.abs(y_si) ** 2) + 1e-20
    p_echo = np.mean(np.abs(echo) ** 2) + 1e-20
    si_to_echo_lin = 10 ** (isr_db / 10.0)
    # y_si 전력을 기준으로 echo 조정
    scale_echo = np.sqrt(p_si / (si_to_echo_lin * p_echo + 1e-20))
    echo = echo * scale_echo

    # 잡음 (SNR 기준: SNR = P_echo / P_noise)
    p_echo = np.mean(np.abs(echo) ** 2) + 1e-20
    snr_lin = 10 ** (snr_db / 10.0)
    p_noise = p_echo / snr_lin
    noise_std = np.sqrt(p_noise / 2)
    noise = noise_std * (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(np.complex64)
    measured_isr_db = 10.0 * np.log10(p_si / (p_echo + 1e-20))
    measured_snr_db = 10.0 * np.log10(p_echo / (p_noise + 1e-20))

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
        "nonlinear": np.array(nonlinear, dtype=np.uint8),
        "si_power": np.float32(p_si),
        "target_echo_power": np.float32(p_echo),
        "noise_power": np.float32(p_noise),
        "measured_isr_db": np.float32(measured_isr_db),
        "measured_snr_db": np.float32(measured_snr_db),
    }


def generate_split(n: int, seed: int,
                   isr_range: tuple[float, float] = (-10.0, 20.0),
                   snr_range: tuple[float, float] = (5.0, 25.0),
                   *,
                   sir_range: tuple[float, float] | None = None) -> dict[str, np.ndarray]:
    """N개 샘플 생성.

    Returns
    -------
    dict with arrays of shape (N, 2, 512) for tx_ref/rx_mix/y_si/y_clean,
    and (N,) for isr_db/sir_db/snr_db. The legacy `sir_db` key is retained
    as an alias for ISR = P_si/P_echo in dB, so larger values mean stronger
    self-interference. `snr_db` is referenced to target echo power:
    SNR = P_echo/P_noise.
    """
    if sir_range is not None:
        isr_range = sir_range

    rng = np.random.default_rng(seed)

    tx_refs, rx_mixes, y_sis, y_cleans = [], [], [], []
    isr_dbs, snr_dbs, nonlinear_flags = [], [], []
    si_powers, echo_powers, noise_powers = [], [], []
    measured_isrs, measured_snrs = [], []

    for _ in range(n):
        isr_db = rng.uniform(isr_range[0], isr_range[1])
        snr_db = rng.uniform(snr_range[0], snr_range[1])
        sample = make_sample(rng, isr_db, snr_db)
        tx_refs.append(sample["tx_ref"])
        rx_mixes.append(sample["rx_mix"])
        y_sis.append(sample["y_si"])
        y_cleans.append(sample["y_clean"])
        isr_dbs.append(isr_db)
        snr_dbs.append(snr_db)
        nonlinear_flags.append(sample["nonlinear"])
        si_powers.append(sample["si_power"])
        echo_powers.append(sample["target_echo_power"])
        noise_powers.append(sample["noise_power"])
        measured_isrs.append(sample["measured_isr_db"])
        measured_snrs.append(sample["measured_snr_db"])

    isr_db_arr = np.array(isr_dbs, dtype=np.float32)
    return {
        "tx_ref": np.stack(tx_refs, axis=0).astype(np.float32),    # (N, 2, 512)
        "rx_mix": np.stack(rx_mixes, axis=0).astype(np.float32),   # (N, 2, 512)
        "y_si": np.stack(y_sis, axis=0).astype(np.float32),        # (N, 2, 512)
        "y_clean": np.stack(y_cleans, axis=0).astype(np.float32),  # (N, 2, 512)
        "isr_db": isr_db_arr,                                      # (N,)
        "sir_db": isr_db_arr.copy(),                               # (N,) legacy alias
        "snr_db": np.array(snr_dbs, dtype=np.float32),             # (N,)
        "nonlinear": np.array(nonlinear_flags, dtype=np.uint8),     # (N,)
        "si_power": np.array(si_powers, dtype=np.float32),          # (N,)
        "target_echo_power": np.array(echo_powers, dtype=np.float32), # (N,)
        "noise_power": np.array(noise_powers, dtype=np.float32),    # (N,)
        "measured_isr_db": np.array(measured_isrs, dtype=np.float32), # (N,)
        "measured_snr_db": np.array(measured_snrs, dtype=np.float32), # (N,)
    }


def main():
    p = base_parser("P07: Full-Duplex SIC 데이터 생성")
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
