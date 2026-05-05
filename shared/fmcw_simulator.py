"""FMCW baseband dechirp simulator — teaching common core.

The active simulator is a complex-baseband FMCW model.  It synthesizes the
transmitted chirp and delayed/Doppler-shifted received echoes, then performs the
mixer/dechirp operation ``rx * conj(tx)``.  RF passband up-conversion and
receiver down-conversion are intentionally not sampled; carrier frequency is used
for wavelength, carrier phase, Doppler, antenna phase, and radar-equation power.

Public names from the earlier teaching simulator are kept so older project
scripts and lecture utilities can migrate without API churn.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d

C = 299_792_458.0
K_BOLTZMANN = 1.380_649e-23


class FMCWRadar:
    """Complex-baseband FMCW radar parameters.

    ``fs`` defaults to ``4*bw`` per the active sampling contract.  ``pulse_width``
    is accepted as a compatibility alias for sweep duration: if it is supplied
    and ``PRI`` is not supplied, ``T_chirp`` is treated as the slow-time
    repetition interval and ``pulse_width`` is used as the FMCW sweep time.
    Active project wiring should pass ``T_chirp`` as sweep duration and ``PRI``
    explicitly when they differ.
    """

    def __init__(
        self,
        fc: float = 77.0e9,
        bw: float = 50e6,
        T_chirp: float = 2e-6,
        N_chirps: int = 64,
        fs: float | None = None,
        N_rx: int = 1,
        d_rx: float | None = None,
        pulse_width: float | None = None,
        PRI: float | None = None,
        N_samples: int | None = None,
        tx_power_w: float = 100.0,
        tx_gain_db: float = 20.0,
        rx_gain_db: float = 20.0,
        noise_figure_db: float = 5.0,
        system_loss_db: float = 6.0,
        temperature_k: float = 290.0,
        reference_range_m: float | None = None,
        reference_rcs_m2: float = 1.0,
        simulate_range_walk: bool = True,
        phase_noise_std_rad: float = 0.0,
        enforce_4x_sampling: bool = True,
    ):
        self.fc = float(fc)
        self.bw = float(bw)
        if self.bw <= 0:
            raise ValueError("bw must be positive")
        self.fs = float(fs if fs is not None else 4.0 * self.bw)
        if enforce_4x_sampling and not np.isclose(self.fs, 4.0 * self.bw, rtol=1e-9, atol=1e-6):
            raise ValueError("active FMCW configs must use fs = 4 * bw")

        # Compatibility mode: when pulse_width is supplied without PRI, treat
        # T_chirp as the repetition interval and pulse_width as the LFM duration.
        if pulse_width is not None and PRI is None:
            self.T_chirp = float(pulse_width)
            self.PRI = float(T_chirp)
        else:
            self.T_chirp = float(T_chirp)
            self.PRI = float(PRI if PRI is not None else T_chirp)
        self.pulse_width = self.T_chirp  # compatibility alias
        if self.T_chirp <= 0 or self.PRI <= 0:
            raise ValueError("T_chirp and PRI must be positive")

        self.N_chirps = int(N_chirps)
        self.N_rx = int(N_rx)
        if self.N_chirps <= 0 or self.N_rx <= 0:
            raise ValueError("N_chirps and N_rx must be positive")

        self.N_samples = int(N_samples if N_samples is not None else max(32, round(self.T_chirp * self.fs)))
        if self.N_samples < 16:
            raise ValueError("N_samples must be at least 16")

        self.tx_power_w = float(tx_power_w)
        self.tx_gain_db = float(tx_gain_db)
        self.rx_gain_db = float(rx_gain_db)
        self.noise_figure_db = float(noise_figure_db)
        self.system_loss_db = float(system_loss_db)
        self.temperature_k = float(temperature_k)
        self.simulate_range_walk = bool(simulate_range_walk)
        self.phase_noise_std_rad = float(phase_noise_std_rad)
        self.enforce_4x_sampling = bool(enforce_4x_sampling)

        self.lam = C / self.fc
        self.d_rx = float(d_rx if d_rx is not None else self.lam / 2.0)
        self.mu = self.bw / self.T_chirp
        self.slope = self.mu
        self.reference_range_m = (
            float(reference_range_m) if reference_range_m is not None else max(self.range_res, 0.35 * self.max_range)
        )
        self.reference_rcs_m2 = float(reference_rcs_m2)

    @property
    def range_res(self) -> float:
        return C / (2.0 * self.bw)

    @property
    def range_bin_spacing(self) -> float:
        return C * self.fs / (2.0 * self.slope * self.N_samples)

    @property
    def N_range_bins(self) -> int:
        return max(1, self.N_samples // 2)

    @property
    def max_range(self) -> float:
        return self.range_bin_spacing * (self.N_range_bins - 1)

    @property
    def vel_res(self) -> float:
        return self.lam / (2.0 * self.N_chirps * self.PRI)

    @property
    def max_vel(self) -> float:
        return self.lam / (4.0 * self.PRI)

    @property
    def noise_bandwidth(self) -> float:
        return self.bw

    @property
    def thermal_noise_power(self) -> float:
        nf = 10.0 ** (self.noise_figure_db / 10.0)
        return K_BOLTZMANN * self.temperature_k * self.noise_bandwidth * nf

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
    def fast_time(self) -> np.ndarray:
        return np.arange(self.N_samples, dtype=np.float64) / self.fs

    def waveform(self) -> np.ndarray:
        return fmcw_chirp(self.bw, self.T_chirp, self.fs, self.N_samples)[1]

    def received_power(self, range_m, rcs_m2=1.0, angle_deg=0.0, tx_power_scale=1.0):
        r = max(float(range_m), self.range_bin_spacing)
        sigma = max(float(rcs_m2), 1e-12)
        angle_gain = _element_power_gain(angle_deg)
        numerator = self.tx_power_w * tx_power_scale * self.tx_gain_linear * self.rx_gain_linear * self.lam**2 * sigma * angle_gain
        denominator = ((4.0 * np.pi) ** 3) * r**4 * self.system_loss_linear
        return float(numerator / denominator)

    def print_params(self):
        print("=== FMCW Baseband Dechirp Radar Parameters ===")
        print(f"  fc        = {self.fc/1e9:.2f} GHz")
        print(f"  BW        = {self.bw/1e6:.1f} MHz")
        print(f"  sweep     = {self.T_chirp*1e6:.2f} us")
        print(f"  PRI       = {self.PRI*1e6:.2f} us")
        print(f"  N_chirps  = {self.N_chirps}")
        print(f"  fs        = {self.fs/1e6:.1f} MHz ({self.fs/self.bw:.1f}x BW)")
        print(f"  N_fast    = {self.N_samples}")
        print(f"  N_rx      = {self.N_rx}")
        print("  ---")
        print(f"  range_res = {self.range_res:.3f} m")
        print(f"  bin_space = {self.range_bin_spacing:.3f} m")
        print(f"  max_range = {self.max_range:.1f} m")
        print(f"  vel_res   = {self.vel_res:.3f} m/s")
        print(f"  max_vel   = {self.max_vel:.1f} m/s ({self.max_vel*3.6:.1f} km/h)")
        print(f"  noise     = {10*np.log10(self.thermal_noise_power/1e-3):.1f} dBm")


def fmcw_chirp(bw, T_chirp, fs, N_samples: int | None = None):
    """Generate a unit-amplitude complex baseband up-chirp."""

    n = int(N_samples if N_samples is not None else max(16, round(T_chirp * fs)))
    t = np.arange(n, dtype=np.float64) / fs
    slope = bw / T_chirp
    inside = t < T_chirp
    s = np.zeros(n, dtype=np.complex128)
    s[inside] = np.exp(1j * np.pi * slope * t[inside] ** 2)
    return t, s


# Historical alias used by older scripts.
def lfm_waveform(bw, pulse_width, fs):
    return fmcw_chirp(bw, pulse_width, fs)


def add_complex_awgn(signal, snr_db, rng, reference_power=None):
    if reference_power is None:
        reference_power = float(np.mean(np.abs(signal) ** 2))
    noise_power = reference_power / (10.0 ** (snr_db / 10.0))
    noise_std = np.sqrt(max(noise_power, 1e-30) / 2.0)
    return signal + noise_std * (rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape))


def encode_complex_iq_signed(
    signal,
    bits: int = 16,
    full_scale: float | None = None,
    return_meta: bool = False,
):
    """Encode complex I/Q samples as signed integer I,Q component codes.

    There is no standard NumPy/HDF5 "complex int16" scalar dtype.  Receiver-like
    complex 16-bit data means two signed 16-bit components per sample:
    ``[..., 0]`` = I and ``[..., 1]`` = Q.  For ``bits <= 16`` this function
    returns ``int16`` component codes; larger bit depths use ``int32``.
    """

    if bits < 2:
        raise ValueError("bits must be >= 2")
    x = np.asarray(signal)
    if not np.iscomplexobj(x):
        raise ValueError("signal must be complex I/Q")
    max_code = (2 ** (bits - 1)) - 1
    min_code = -(2 ** (bits - 1))
    if full_scale is None:
        full_scale = float(max(np.max(np.abs(x.real)), np.max(np.abs(x.imag)), 1e-30))
    full_scale = float(full_scale)
    if full_scale <= 0:
        raise ValueError("full_scale must be positive")

    scale = max_code / full_scale
    i_unclipped = np.rint(x.real * scale)
    q_unclipped = np.rint(x.imag * scale)
    i_code = np.clip(i_unclipped, min_code, max_code)
    q_code = np.clip(q_unclipped, min_code, max_code)
    dtype = np.int16 if bits <= 16 else np.int32
    codes = np.stack([i_code, q_code], axis=-1).astype(dtype)

    if return_meta:
        clipped = np.count_nonzero((i_unclipped != i_code) | (q_unclipped != q_code))
        return codes, {
            "bits": int(bits),
            "full_scale": full_scale,
            "lsb": float(1.0 / scale),
            "clipped_fraction": float(clipped / x.size),
            "component_dtype": str(np.dtype(dtype)),
            "component_axis": "last_dim_iq",
        }
    return codes


def decode_complex_iq_signed(codes, full_scale: float, bits: int = 16):
    """Decode signed I,Q component codes back to complex floating-point I/Q."""

    arr = np.asarray(codes)
    if arr.shape[-1] != 2:
        raise ValueError("codes must have last dimension [I, Q]")
    max_code = (2 ** (bits - 1)) - 1
    scale = max_code / float(full_scale)
    return (arr[..., 0].astype(np.float64) + 1j * arr[..., 1].astype(np.float64)) / scale


def quantize_complex_iq(signal, bits: int = 16, full_scale: float | None = None, return_meta: bool = False):
    """Uniform signed integer quantization for complex I/Q diagnostics.

    This helper models an ADC-style signed quantizer independently on I and Q.
    It is intentionally diagnostic-only: active datasets still store floating
    point arrays unless a project explicitly opts into quantized data.

    Parameters
    ----------
    signal : ndarray
        Complex baseband I/Q samples.
    bits : int
        Signed integer precision. ``bits=16`` gives code range [-32768, 32767].
    full_scale : float or None
        Absolute I/Q value mapped to the largest positive code. If omitted, the
        maximum absolute real/imag sample in this frame is used, which measures
        best-case per-frame quantization rather than fixed receiver gain.
    return_meta : bool
        Return quantization metadata with scale and clipping fraction.
    """

    codes, meta = encode_complex_iq_signed(signal, bits=bits, full_scale=full_scale, return_meta=True)
    y = decode_complex_iq_signed(codes, full_scale=meta["full_scale"], bits=bits)
    if return_meta:
        return y, meta
    return y


def generate_scene(radar: FMCWRadar, targets, snr_db=20.0, seed=42, return_meta=False):
    """Generate dechirped FMCW beat data with shape ``(N_rx, N_chirps, N_samples)``."""

    rng = np.random.default_rng(seed)
    tx = radar.waveform()
    fast_time = radar.fast_time
    slow_time = np.arange(radar.N_chirps, dtype=np.float64) * radar.PRI
    rx_data = np.zeros((radar.N_rx, radar.N_chirps, radar.N_samples), dtype=np.complex128)
    rx_idx = np.arange(radar.N_rx, dtype=np.float64)

    tx_power_scale = 1.0
    processing_gain = _nominal_rd_processing_gain(radar)
    if snr_db is not None:
        ref_power = radar.received_power(radar.reference_range_m, radar.reference_rcs_m2)
        tx_power_scale = 10.0 ** (snr_db / 10.0) * radar.thermal_noise_power / ((ref_power + 1e-300) * processing_gain)

    target_meta = []
    for tgt in targets:
        r0 = float(tgt["range"])
        v = float(tgt.get("velocity", 0.0))  # positive is closing: range decreases over slow time
        rcs = _rcs_linear(tgt.get("rcs", 1.0))
        theta_deg = float(tgt.get("angle", tgt.get("angle_deg", 0.0)))
        scatter_phase = float(tgt.get("phase", rng.uniform(0.0, 2.0 * np.pi)))
        fd = 2.0 * v / radar.lam
        spatial = np.exp(1j * 2.0 * np.pi * rx_idx * radar.d_rx * np.sin(np.deg2rad(theta_deg)) / radar.lam)

        ranges = r0 - v * slow_time if radar.simulate_range_walk else np.full_like(slow_time, r0)
        valid = ranges > 0.0
        tau = 2.0 * ranges / C
        local_time = fast_time[None, :] - tau[:, None]
        inside = (local_time >= 0.0) & (local_time < radar.T_chirp) & valid[:, None]

        delayed_tx = np.zeros_like(local_time, dtype=np.complex128)
        delayed_tx[inside] = np.exp(1j * np.pi * radar.slope * local_time[inside] ** 2)
        p_rx = _received_power_array(radar, ranges, rcs, theta_deg, tx_power_scale=tx_power_scale)
        amp = np.sqrt(p_rx)[:, None]
        carrier_phase = np.exp(-1j * 2.0 * np.pi * radar.fc * tau)[:, None]
        doppler_phase = np.exp(1j * 2.0 * np.pi * fd * fast_time[None, :])
        rx_echo = amp * delayed_tx * carrier_phase * doppler_phase * np.exp(1j * scatter_phase)
        beat = rx_echo * np.conj(tx)[None, :]
        rx_data += spatial[:, None, None] * beat[None, :, :]

        r_bin, d_bin = target_rd_bins(radar, r0, v)
        p0 = radar.received_power(r0, rcs, theta_deg, tx_power_scale=tx_power_scale)
        target_meta.append({
            "range": r0,
            "velocity": v,
            "angle_deg": theta_deg,
            "rcs": rcs,
            "range_bin": int(r_bin),
            "doppler_bin": int(d_bin),
            "actual_snr_db": float(10.0 * np.log10((p0 + 1e-300) * processing_gain / radar.thermal_noise_power)),
            "sample_snr_db": float(10.0 * np.log10((p0 + 1e-300) / radar.thermal_noise_power)),
            "range_walk_m": float(-v * (radar.N_chirps - 1) * radar.PRI),
            "is_clutter": bool(tgt.get("is_clutter", False)),
        })

    phase_noise = np.zeros(radar.N_chirps, dtype=np.float64)
    if radar.phase_noise_std_rad > 0.0:
        phase_noise = _pulse_phase_noise(rng, radar.N_chirps, radar.phase_noise_std_rad)
        rx_data *= np.exp(1j * phase_noise)[None, :, None]

    noise_std = np.sqrt(radar.thermal_noise_power / 2.0)
    rx_data += noise_std * (rng.standard_normal(rx_data.shape) + 1j * rng.standard_normal(rx_data.shape))

    if return_meta:
        return rx_data, {
            "simulator": "fmcw_baseband_dechirp_mixing",
            "nominal_snr_db": None if snr_db is None else float(snr_db),
            "noise_power_w": float(radar.thermal_noise_power),
            "tx_power_scale": float(tx_power_scale),
            "target_info": target_meta,
            "reference_range_m": float(radar.reference_range_m),
            "reference_rcs_m2": float(radar.reference_rcs_m2),
            "phase_noise_std_rad": float(radar.phase_noise_std_rad),
            "phase_noise_applied_std_rad": float(np.std(phase_noise)),
            "chirp_slope_hz_per_s": float(radar.slope),
            "fs_over_bandwidth": float(radar.fs / radar.bw),
            "up_down_conversion": "excluded_baseband_only",
        }
    return rx_data


def _positive_range_spectrum(spectrum: np.ndarray, n_bins: int) -> np.ndarray:
    """Return positive-range bins for beat convention ``rx * conj(tx)``.

    With an up-chirp and ``rx * conj(tx)``, positive target range appears at
    negative beat frequency.  Bin 0 is DC; bins 1.. map from FFT indices -1, -2,
    ... so returned range bins ascend from zero range outward.
    """

    if n_bins <= 1:
        return spectrum[..., :1]
    neg = spectrum[..., -(n_bins - 1):][..., ::-1]
    return np.concatenate([spectrum[..., :1], neg], axis=-1)


def range_fft(signal, window="hann", radar: FMCWRadar | None = None):
    """Range FFT along fast time.

    With ``radar``, returns positive-range FMCW beat bins.  Without ``radar``,
    returns the raw FFT ordering.
    """

    w = _get_window(window, signal.shape[-1])
    spec = np.fft.fft(signal * w, axis=-1)
    if radar is not None:
        return _positive_range_spectrum(spec, radar.N_range_bins)
    return spec


def pulse_compress(signal: np.ndarray, radar: FMCWRadar, window="hann") -> np.ndarray:
    """Deprecated compatibility wrapper; active FMCW uses ``range_fft``."""

    return range_fft(signal, window=window, radar=radar)


def range_doppler_map(signal, radar: FMCWRadar | None = None, window_range="hann", window_doppler="hann"):
    """Range-Doppler map from range FFT followed by slow-time Doppler FFT."""

    nc = signal.shape[-2]
    rng_data = range_fft(signal, window=window_range, radar=radar)
    w_d = _get_window(window_doppler, nc)
    shape = [1] * rng_data.ndim
    shape[-2] = nc
    rdm = np.fft.fftshift(np.fft.fft(rng_data * w_d.reshape(shape), axis=-2), axes=-2)
    return rdm


def range_axis(radar: FMCWRadar) -> np.ndarray:
    return np.arange(radar.N_range_bins, dtype=np.float32) * radar.range_bin_spacing


def velocity_axis(radar: FMCWRadar) -> np.ndarray:
    return (np.fft.fftshift(np.fft.fftfreq(radar.N_chirps, d=radar.PRI)) * radar.lam / 2.0).astype(np.float32)


def target_rd_bins(radar: FMCWRadar, range_m, velocity_mps):
    beat_freq = radar.slope * (2.0 * float(range_m) / C)
    bin_spacing_hz = radar.fs / radar.N_samples
    r_bin = int(round(beat_freq / bin_spacing_hz))
    r_bin = int(np.clip(r_bin, 0, radar.N_range_bins - 1))
    vel = velocity_axis(radar)
    d_bin = int(np.argmin(np.abs(vel - float(velocity_mps))))
    return r_bin, d_bin


def range_angle_map(signal, radar, window_range="hann", window_angle="hann", N_angle=64):
    rng = range_fft(signal, radar=radar, window=window_range)
    rng_avg = np.mean(rng, axis=1)
    w_a = _get_window(window_angle, signal.shape[0])
    rng_avg_windowed = rng_avg * w_a[:, np.newaxis]
    padded = np.zeros((N_angle, rng_avg.shape[1]), dtype=np.complex128)
    padded[:signal.shape[0], :] = rng_avg_windowed
    ram = np.fft.fftshift(np.fft.fft(padded, axis=0), axes=0)
    u = np.fft.fftshift(np.fft.fftfreq(N_angle))
    angle_axis_deg = np.degrees(np.arcsin(np.clip(u / (radar.d_rx / radar.lam), -1, 1)))
    return ram, angle_axis_deg


def _rcs_linear(rcs):
    return max(float(rcs), 1e-12)


def _received_power_array(radar: FMCWRadar, range_m: np.ndarray, rcs_m2: float, angle_deg: float, tx_power_scale: float = 1.0) -> np.ndarray:
    r = np.maximum(np.asarray(range_m, dtype=np.float64), radar.range_bin_spacing)
    sigma = max(float(rcs_m2), 1e-12)
    angle_gain = _element_power_gain(angle_deg)
    numerator = radar.tx_power_w * tx_power_scale * radar.tx_gain_linear * radar.rx_gain_linear * radar.lam**2 * sigma * angle_gain
    denominator = ((4.0 * np.pi) ** 3) * r**4 * radar.system_loss_linear
    return numerator / denominator


def _element_power_gain(angle_deg):
    theta = abs(float(angle_deg))
    if theta > 90.0:
        return 0.0
    return float(np.cos(np.radians(theta)) ** 2)


def _nominal_rd_processing_gain(radar: FMCWRadar):
    wd = np.hanning(radar.N_chirps)
    doppler_gain = (np.sum(wd) ** 2) / (np.sum(wd**2) + 1e-30)
    wr = np.hanning(radar.N_samples)
    range_gain = (np.sum(wr) ** 2) / (np.sum(wr**2) + 1e-30)
    return max(float(range_gain * doppler_gain), 1.0)


def _pulse_phase_noise(rng: np.random.Generator, n_pulses: int, std_rad: float) -> np.ndarray:
    if n_pulses <= 0 or std_rad <= 0.0:
        return np.zeros(max(n_pulses, 0), dtype=np.float64)
    white = rng.normal(0.0, std_rad, n_pulses)
    drift_steps = rng.normal(0.0, std_rad * 0.05, n_pulses)
    drift = np.cumsum(drift_steps)
    drift -= np.mean(drift)
    return white + drift


def ca_cfar_1d(signal_mag, guard_cells=2, train_cells=8, pfa=1e-4):
    """1D CA-CFAR detector."""

    N = len(signal_mag)
    power = signal_mag**2
    alpha = 2 * train_cells * (pfa ** (-1.0 / (2 * train_cells)) - 1)
    detections = np.zeros(N, dtype=bool)
    threshold = np.zeros(N)
    margin = guard_cells + train_cells
    for i in range(margin, N - margin):
        left = power[i - margin:i - guard_cells]
        right = power[i + guard_cells + 1:i + margin + 1]
        noise_est = np.mean(np.concatenate([left, right]))
        t = alpha * noise_est
        threshold[i] = t
        detections[i] = power[i] > t
    return detections, np.sqrt(threshold)


def ca_cfar_2d(rdm_mag, guard=(2, 2), train=(4, 4), pfa=1e-4):
    """2D CA-CFAR detector on magnitude data."""

    Nd, Nr = rdm_mag.shape
    power = rdm_mag**2
    gd, gr = guard
    td, tr = train
    n_train = (2 * (gd + td) + 1) * (2 * (gr + tr) + 1) - (2 * gd + 1) * (2 * gr + 1)
    alpha = n_train * (pfa ** (-1.0 / n_train) - 1)
    kernel = np.ones((2 * (gd + td) + 1, 2 * (gr + tr) + 1), dtype=np.float64)
    kernel[td:td + 2 * gd + 1, tr:tr + 2 * gr + 1] = 0.0
    noise_sum = convolve2d(power, kernel, mode="same", boundary="fill", fillvalue=0.0)
    threshold = alpha * noise_sum / n_train
    detections = power > threshold
    margin_d = gd + td
    margin_r = gr + tr
    if margin_d:
        detections[:margin_d, :] = False
        detections[Nd - margin_d:, :] = False
    if margin_r:
        detections[:, :margin_r] = False
        detections[:, Nr - margin_r:] = False
    return detections


def to_db(x, ref=None):
    mag = np.abs(x)
    if ref is None:
        ref = np.max(mag)
    return 20.0 * np.log10(mag / (ref + 1e-30) + 1e-30)


def _get_window(name, N):
    if name in (None, "rect", "none"):
        return np.ones(N)
    if name == "hann":
        return np.hanning(N)
    if name == "hamming":
        return np.hamming(N)
    if name == "blackman":
        return np.blackman(N)
    return np.hanning(N)
