"""Clean coherent burst radar simulator.

This module is adapted from the target-echo path of the research
``radar-interference`` burst simulator, but deliberately excludes all
interference/RIS/dirty-signal functionality.  It is a small, self-contained core
for teaching matched-filter pulse compression:

1. generate a complex baseband LFM pulse,
2. inject delayed moving target echoes using the monostatic radar equation,
3. add receiver thermal noise,
4. matched-filter each pulse,
5. Doppler FFT across the coherent pulse interval.

No RF passband sampling is performed; ``fc`` is used for wavelength, carrier
phase, Doppler, and radar-equation power only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

C = 299_792_458.0
K_BOLTZMANN = 1.380_649e-23


@dataclass(frozen=True)
class BurstRadar:
    """Parameters for a clean coherent pulse burst radar."""

    fc: float = 9.6e9
    bandwidth: float = 20e6
    pulse_width: float = 4e-6
    prf: float = 10e3
    pulses_per_cpi: int = 64
    sampling_rate: float | None = None
    max_range: float = 3_000.0
    tx_power_w: float = 100.0
    tx_gain_db: float = 20.0
    rx_gain_db: float = 20.0
    noise_figure_db: float = 5.0
    system_loss_db: float = 6.0
    temperature_k: float = 290.0

    def __post_init__(self) -> None:
        if self.bandwidth <= 0 or self.pulse_width <= 0 or self.prf <= 0:
            raise ValueError("bandwidth, pulse_width, and prf must be positive")
        if self.pulses_per_cpi <= 0:
            raise ValueError("pulses_per_cpi must be positive")
        if self.max_range <= 0:
            raise ValueError("max_range must be positive")
        fs = self.fs
        if fs <= 0:
            raise ValueError("sampling_rate must be positive")
        if self.n_samples <= self.pulse_samples + 4:
            raise ValueError("receive window is too short for pulse compression")

    @property
    def fs(self) -> float:
        return float(self.sampling_rate if self.sampling_rate is not None else 4.0 * self.bandwidth)

    @property
    def wavelength(self) -> float:
        return C / self.fc

    @property
    def pri(self) -> float:
        return 1.0 / self.prf

    @property
    def slope(self) -> float:
        return self.bandwidth / self.pulse_width

    @property
    def pulse_samples(self) -> int:
        return max(8, int(round(self.pulse_width * self.fs)))

    @property
    def n_samples(self) -> int:
        useful = int(np.ceil((2.0 * self.max_range / C) * self.fs))
        guard = self.pulse_samples + 8
        return min(int(np.floor(self.pri * self.fs)), useful + guard)

    @property
    def range_bin_spacing(self) -> float:
        return C / (2.0 * self.fs)

    @property
    def range_resolution(self) -> float:
        return C / (2.0 * self.bandwidth)

    @property
    def velocity_resolution(self) -> float:
        return self.wavelength / (2.0 * self.pulses_per_cpi * self.pri)

    @property
    def max_velocity(self) -> float:
        return self.wavelength / (4.0 * self.pri)

    @property
    def tx_gain_linear(self) -> float:
        return 10.0 ** (self.tx_gain_db / 10.0)

    @property
    def rx_gain_linear(self) -> float:
        return 10.0 ** (self.rx_gain_db / 10.0)

    @property
    def system_loss_linear(self) -> float:
        return 10.0 ** (self.system_loss_db / 10.0)

    @property
    def noise_power_w(self) -> float:
        nf = 10.0 ** (self.noise_figure_db / 10.0)
        return K_BOLTZMANN * self.temperature_k * self.bandwidth * nf

    def received_power(self, range_m: float, rcs_m2: float = 1.0) -> float:
        r = max(float(range_m), self.range_bin_spacing)
        sigma = max(float(rcs_m2), 1e-12)
        numerator = self.tx_power_w * self.tx_gain_linear * self.rx_gain_linear * self.wavelength**2 * sigma
        denominator = ((4.0 * np.pi) ** 3) * r**4 * self.system_loss_linear
        return float(numerator / denominator)


def lfm_pulse(bandwidth: float, pulse_width: float, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Generate a unit-amplitude complex baseband LFM pulse."""

    n = max(8, int(round(pulse_width * fs)))
    t = np.arange(n, dtype=np.float64) / fs
    centered = t - pulse_width / 2.0
    slope = bandwidth / pulse_width
    s = np.exp(1j * np.pi * slope * centered**2)
    return t, s.astype(np.complex128)


class CleanBurstSimulator:
    """Target-only coherent burst simulator with matched-filter processing."""

    def __init__(self, radar: BurstRadar):
        self.radar = radar
        self.fast_time = np.arange(radar.n_samples, dtype=np.float64) / radar.fs
        _, self.reference = lfm_pulse(radar.bandwidth, radar.pulse_width, radar.fs)

    def simulate_burst(
        self,
        targets: Iterable[Mapping[str, float]],
        *,
        seed: int | None = None,
        add_noise: bool = True,
        return_meta: bool = False,
    ):
        """Simulate raw ADC fast-time/slow-time data for clean targets.

        ``targets`` entries accept ``range`` [m], ``velocity`` [m/s, positive
        closing], and ``rcs`` [m²].  Interference configuration is intentionally
        not part of this API.
        """

        rng = np.random.default_rng(seed)
        r = self.radar
        adc = np.zeros((r.n_samples, r.pulses_per_cpi), dtype=np.complex128)
        slow_time = np.arange(r.pulses_per_cpi, dtype=np.float64) * r.pri
        meta_targets: list[dict[str, float | int]] = []

        for target in targets:
            r0 = float(target["range"])
            v = float(target.get("velocity", 0.0))
            rcs = max(float(target.get("rcs", 1.0)), 1e-12)
            scatter_phase = float(target.get("phase", rng.uniform(0.0, 2.0 * np.pi)))

            ranges = r0 - v * slow_time
            valid = ranges > 0.0
            fd = 2.0 * v / r.wavelength
            amp = np.sqrt([r.received_power(x, rcs) if ok else 0.0 for x, ok in zip(ranges, valid)])
            tau = 2.0 * ranges / C

            for m in np.where(valid)[0]:
                delay = tau[m]
                sample_delay = int(np.floor(delay * r.fs))
                if sample_delay >= r.n_samples:
                    continue
                stop = min(r.n_samples, sample_delay + len(self.reference))
                n = stop - sample_delay
                if n <= 0:
                    continue
                idx = np.arange(sample_delay, stop)
                carrier_phase = np.exp(-1j * 2.0 * np.pi * r.fc * delay)
                doppler_phase = np.exp(1j * 2.0 * np.pi * fd * self.fast_time[idx])
                adc[idx, m] += amp[m] * self.reference[:n] * carrier_phase * doppler_phase * np.exp(1j * scatter_phase)

            r_bin = int(round((2.0 * r0 / C) * r.fs))
            d_bin = int(np.argmin(np.abs(velocity_axis(r) - v)))
            meta_targets.append({
                "range": r0,
                "velocity": v,
                "rcs": rcs,
                "range_bin": r_bin,
                "doppler_bin": d_bin,
            })

        if add_noise:
            sigma = np.sqrt(r.noise_power_w / 2.0)
            adc += sigma * (rng.standard_normal(adc.shape) + 1j * rng.standard_normal(adc.shape))

        if return_meta:
            return adc, self.reference.copy(), {
                "simulator": "clean_lfm_burst_no_interference",
                "noise_power_w": float(r.noise_power_w),
                "target_info": meta_targets,
                "fs_over_bandwidth": float(r.fs / r.bandwidth),
            }
        return adc, self.reference.copy()

    def process_burst(self, adc_data: np.ndarray, ref_chirp: np.ndarray | None = None, *, window: str | None = None):
        """Matched-filter fast time, then Doppler FFT across pulses."""

        ref = self.reference if ref_chirp is None else np.asarray(ref_chirp)
        if window not in (None, "none", "rect"):
            ref = ref * _window(window, len(ref))
        h = np.conj(ref[::-1])
        n_fast = adc_data.shape[0]
        n_fft = 1 << int(np.ceil(np.log2(n_fast + len(h) - 1)))
        H = np.fft.fft(h, n_fft)
        Y = np.fft.fft(adc_data, n_fft, axis=0) * H[:, None]
        full = np.fft.ifft(Y, axis=0)
        start = len(h) - 1
        pc = full[start:start + n_fast, :]
        rd = np.fft.fftshift(np.fft.fft(pc, axis=1), axes=1)
        return pc, rd


def range_axis(radar: BurstRadar) -> np.ndarray:
    return np.arange(radar.n_samples, dtype=np.float64) * radar.range_bin_spacing


def velocity_axis(radar: BurstRadar) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftfreq(radar.pulses_per_cpi, d=radar.pri)) * radar.wavelength / 2.0


def _window(name: str | None, n: int) -> np.ndarray:
    if name in (None, "none", "rect"):
        return np.ones(n)
    if name == "hann":
        return np.hanning(n)
    if name == "hamming":
        return np.hamming(n)
    if name == "blackman":
        return np.blackman(n)
    raise ValueError(f"unknown window: {name}")
