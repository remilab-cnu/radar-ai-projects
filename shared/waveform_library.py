"""Lightweight radar waveform generators for P05.

The functions in this module intentionally mirror the small teaching subset used
by the MathWorks radar waveform-classification examples: rectangular pulses,
linear-FM pulses, Barker phase-coded pulses, and noise-only observations.  The
implementation stays NumPy/SciPy-only so students can inspect every step before
training a small Python classifier.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.signal import stft

from .fmcw_simulator import add_complex_awgn

WAVEFORM_CLASSES = ("rect", "lfm", "barker", "noise_only")
SCHEMA_VERSION = 1
BARKER_CODES: dict[int, tuple[int, ...]] = {
    2: (1, -1),
    3: (1, 1, -1),
    4: (1, 1, -1, 1),
    5: (1, 1, 1, -1, 1),
    7: (1, 1, 1, -1, -1, 1, -1),
    11: (1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1),
    13: (1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1),
}


@dataclass(frozen=True)
class WaveformConfig:
    """Configuration for fixed-size baseband waveform examples."""

    sample_rate_hz: float = 20.0e6
    n_samples: int = 512
    image_size: tuple[int, int] = (64, 64)
    stft_window: int = 64
    stft_overlap: int = 48
    stft_nfft: int = 128
    snr_range_db: tuple[float, float] = (-6.0, 18.0)
    bandwidth_range_hz: tuple[float, float] = (2.0e6, 8.0e6)
    pulse_fraction_range: tuple[float, float] = (0.35, 0.85)
    max_frequency_offset_hz: float = 1.0e6

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sample_rate_hz


def _rng(rng: np.random.Generator | None) -> np.random.Generator:
    return np.random.default_rng() if rng is None else rng


def rectangular_pulse(n_samples: int, start: int, width: int) -> np.ndarray:
    """Return a unit-amplitude complex rectangular pulse."""

    x = np.zeros(n_samples, dtype=np.complex128)
    stop = min(n_samples, start + max(1, width))
    x[start:stop] = 1.0 + 0.0j
    return x


def lfm_pulse(n_samples: int, start: int, width: int, bandwidth_hz: float, sample_rate_hz: float) -> np.ndarray:
    """Return a baseband linear-FM pulse embedded in a fixed observation window."""

    x = np.zeros(n_samples, dtype=np.complex128)
    width = max(8, int(width))
    stop = min(n_samples, start + width)
    n = max(0, stop - start)
    if n == 0:
        return x
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    centered = t - (n / sample_rate_hz) / 2.0
    slope = float(bandwidth_hz) / max(n / sample_rate_hz, 1e-12)
    x[start:stop] = np.exp(1j * np.pi * slope * centered**2)
    return x


def barker_phase_code(n_samples: int, start: int, width: int, code_length: int = 13) -> np.ndarray:
    """Return a Barker-coded BPSK pulse embedded in a fixed observation window."""

    code = np.asarray(BARKER_CODES[int(code_length)], dtype=np.float64)
    x = np.zeros(n_samples, dtype=np.complex128)
    width = max(len(code), int(width))
    stop = min(n_samples, start + width)
    n = max(0, stop - start)
    if n == 0:
        return x
    chip_edges = np.linspace(0, n, len(code) + 1).round().astype(int)
    pulse = np.zeros(n, dtype=np.float64)
    for chip, a, b in zip(code, chip_edges[:-1], chip_edges[1:]):
        pulse[a:b] = chip
    x[start:stop] = pulse.astype(np.complex128)
    return x


def apply_frequency_offset(signal: np.ndarray, sample_rate_hz: float, offset_hz: float) -> np.ndarray:
    """Apply a complex carrier-frequency offset to a baseband observation."""

    n = np.arange(signal.size, dtype=np.float64)
    return np.asarray(signal, dtype=np.complex128) * np.exp(1j * 2.0 * np.pi * float(offset_hz) * n / sample_rate_hz)


def _resize_2d(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a 2-D array with separable linear interpolation."""

    out_h, out_w = shape
    src_h, src_w = image.shape
    if (src_h, src_w) == (out_h, out_w):
        return image.astype(np.float32)
    y_old = np.linspace(0.0, 1.0, src_h)
    y_new = np.linspace(0.0, 1.0, out_h)
    tmp = np.empty((out_h, src_w), dtype=np.float64)
    for j in range(src_w):
        tmp[:, j] = np.interp(y_new, y_old, image[:, j])
    x_old = np.linspace(0.0, 1.0, src_w)
    x_new = np.linspace(0.0, 1.0, out_w)
    out = np.empty((out_h, out_w), dtype=np.float64)
    for i in range(out_h):
        out[i] = np.interp(x_new, x_old, tmp[i])
    return out.astype(np.float32)


def stft_log_image(
    signal: np.ndarray,
    sample_rate_hz: float,
    *,
    image_size: tuple[int, int] = (64, 64),
    nperseg: int = 64,
    noverlap: int = 48,
    nfft: int = 128,
    floor_db: float = -60.0,
) -> np.ndarray:
    """Convert a complex waveform to a normalized log-magnitude STFT image."""

    _, _, z = stft(
        np.asarray(signal, dtype=np.complex128),
        fs=sample_rate_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        return_onesided=False,
        boundary=None,
        padded=False,
    )
    mag = np.fft.fftshift(np.abs(z), axes=0)
    ref = max(float(np.max(mag)), 1e-12)
    db = 20.0 * np.log10(mag / ref + 1e-12)
    norm = np.clip((db - floor_db) / abs(floor_db), 0.0, 1.0)
    return _resize_2d(norm, image_size).astype(np.float32)


def extract_waveform_features(signal: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    """Small handcrafted descriptor for classical P05 baselines."""

    x = np.asarray(signal, dtype=np.complex128)
    mag = np.abs(x)
    power = mag**2
    total_power = float(np.sum(power) + 1e-12)
    freqs = np.fft.fftshift(np.fft.fftfreq(x.size, d=1.0 / sample_rate_hz))
    spec = np.fft.fftshift(np.abs(np.fft.fft(x)))
    spec_power = spec**2
    spec_total = float(np.sum(spec_power) + 1e-12)
    centroid = float(np.sum(freqs * spec_power) / spec_total) / (sample_rate_hz / 2.0)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid * sample_rate_hz / 2.0) ** 2) * spec_power) / spec_total))
    bandwidth /= sample_rate_hz / 2.0
    envelope_threshold = 0.25 * max(float(np.max(mag)), 1e-12)
    duty = float(np.mean(mag > envelope_threshold))
    phase = np.unwrap(np.angle(x + 1e-12))
    phase_diff = np.diff(phase)
    return np.asarray(
        [
            float(np.mean(mag)),
            float(np.std(mag)),
            float(np.max(mag)),
            float(np.sqrt(np.mean(power))),
            float(np.max(power) / (np.mean(power) + 1e-12)),
            duty,
            centroid,
            bandwidth,
            float(np.std(spec / (np.max(spec) + 1e-12))),
            float(np.mean(np.abs(phase_diff))) if phase_diff.size else 0.0,
            float(np.std(phase_diff)) if phase_diff.size else 0.0,
        ],
        dtype=np.float32,
    )


def _random_window(config: WaveformConfig, rng: np.random.Generator) -> tuple[int, int]:
    frac = rng.uniform(*config.pulse_fraction_range)
    width = int(np.clip(round(frac * config.n_samples), 16, config.n_samples))
    max_start = max(0, config.n_samples - width)
    start = int(rng.integers(0, max_start + 1)) if max_start else 0
    return start, width


def synthesize_waveform(
    class_name: str,
    config: WaveformConfig,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Create one clean waveform and metadata before receiver impairments."""

    rng = _rng(rng)
    class_name = str(class_name)
    if class_name not in WAVEFORM_CLASSES:
        raise ValueError(f"unknown waveform class: {class_name!r}")

    start, width = _random_window(config, rng)
    bandwidth = float(rng.uniform(*config.bandwidth_range_hz))
    code_length = int(rng.choice([7, 11, 13]))

    if class_name == "rect":
        clean = rectangular_pulse(config.n_samples, start, width)
    elif class_name == "lfm":
        clean = lfm_pulse(config.n_samples, start, width, bandwidth, config.sample_rate_hz)
    elif class_name == "barker":
        clean = barker_phase_code(config.n_samples, start, width, code_length=code_length)
    elif class_name == "noise_only":
        clean = np.zeros(config.n_samples, dtype=np.complex128)
    else:  # pragma: no cover - guarded above
        raise ValueError(class_name)

    if class_name != "noise_only":
        clean *= np.exp(1j * rng.uniform(0.0, 2.0 * np.pi))
        rms = np.sqrt(np.mean(np.abs(clean) ** 2) + 1e-12)
        clean = clean / rms

    meta = {
        "class_name": class_name,
        "pulse_start_sample": start,
        "pulse_width_samples": width,
        "bandwidth_hz": bandwidth,
        "barker_length": code_length if class_name == "barker" else 0,
    }
    return clean.astype(np.complex128), meta


def generate_waveform_sample(
    class_name: str,
    rng: np.random.Generator | None = None,
    config: WaveformConfig | None = None,
    *,
    snr_db: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int | str]]:
    """Return `(image, label, features, meta)` for one P05 observation."""

    rng = _rng(rng)
    config = WaveformConfig() if config is None else config
    clean, meta = synthesize_waveform(class_name, config, rng)
    class_idx = WAVEFORM_CLASSES.index(class_name)
    snr = float(rng.uniform(*config.snr_range_db) if snr_db is None else snr_db)

    if class_name == "noise_only":
        observed = (
            rng.standard_normal(config.n_samples) + 1j * rng.standard_normal(config.n_samples)
        ) / np.sqrt(2.0)
    else:
        offset = float(rng.uniform(-config.max_frequency_offset_hz, config.max_frequency_offset_hz))
        shifted = apply_frequency_offset(clean, config.sample_rate_hz, offset)
        observed = add_complex_awgn(shifted, snr, rng, reference_power=1.0)
        meta["frequency_offset_hz"] = offset

    image = stft_log_image(
        observed,
        config.sample_rate_hz,
        image_size=config.image_size,
        nperseg=config.stft_window,
        noverlap=config.stft_overlap,
        nfft=config.stft_nfft,
    )
    features = extract_waveform_features(observed, config.sample_rate_hz)
    meta.update({
        "snr_db": snr,
        "class_index": class_idx,
        "sample_rate_hz": float(config.sample_rate_hz),
        "n_samples": int(config.n_samples),
    })
    return image, np.asarray(class_idx, dtype=np.int64), features, meta


def class_name_bytes() -> np.ndarray:
    return np.asarray([name.encode("utf-8") for name in WAVEFORM_CLASSES], dtype="S32")
