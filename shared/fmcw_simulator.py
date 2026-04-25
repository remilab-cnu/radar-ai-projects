"""FMCW 레이다 시뮬레이터 — 교육용

연구 코드(research/fmcw-isac/simulator/)를 교육 목적으로 단순화.
학생이 읽고 수정하기 쉽도록 한 파일에 핵심 기능만 포함.

사용법:
    from shared.fmcw_simulator import FMCWRadar, generate_scene, range_fft, range_doppler_map
"""

import numpy as np
from scipy.signal import convolve2d, find_peaks

C = 299_792_458.0  # 광속 [m/s]


class FMCWRadar:
    """FMCW 레이다 시스템 파라미터 및 신호 생성.

    Parameters
    ----------
    fc : float
        캐리어 주파수 [Hz] (기본값: 77 GHz)
    bw : float
        chirp 대역폭 [Hz] (기본값: 1 GHz)
    T_chirp : float
        chirp 지속 시간 [s] (기본값: 50 us)
    N_chirps : int
        프레임당 chirp 수 (기본값: 128)
    fs : float
        ADC 샘플링 레이트 [Hz] (기본값: 10 MHz)
    N_rx : int
        수신 안테나 수 (기본값: 1)
    d_rx : float
        수신 안테나 간격 [m] (기본값: lambda/2)
    """

    def __init__(
        self,
        fc=77e9,
        bw=1e9,
        T_chirp=50e-6,
        N_chirps=128,
        fs=10e6,
        N_rx=1,
        d_rx=None,
    ):
        self.fc = fc
        self.bw = bw
        self.T_chirp = T_chirp
        self.N_chirps = N_chirps
        self.fs = fs
        self.N_rx = N_rx

        # 유도 파라미터
        self.lam = C / fc               # 파장 [m]
        self.mu = bw / T_chirp           # chirp rate [Hz/s]
        self.N_samples = int(T_chirp * fs)  # chirp당 ADC 샘플 수
        self.d_rx = d_rx if d_rx else self.lam / 2  # 안테나 간격

    @property
    def range_res(self):
        """거리 분해능 [m]"""
        return C / (2 * self.bw)

    @property
    def max_range(self):
        """최대 비모호 거리 [m]"""
        return self.fs * C * self.T_chirp / (4 * self.bw)

    @property
    def vel_res(self):
        """속도 분해능 [m/s]"""
        return self.lam / (2 * self.N_chirps * self.T_chirp)

    @property
    def max_vel(self):
        """최대 비모호 속도 [m/s]"""
        return self.lam / (4 * self.T_chirp)

    def print_params(self):
        """시스템 파라미터 출력."""
        print(f"=== FMCW Radar Parameters ===")
        print(f"  fc        = {self.fc/1e9:.1f} GHz")
        print(f"  BW        = {self.bw/1e6:.0f} MHz")
        print(f"  T_chirp   = {self.T_chirp*1e6:.1f} us")
        print(f"  N_chirps  = {self.N_chirps}")
        print(f"  fs        = {self.fs/1e6:.1f} MHz")
        print(f"  N_samples = {self.N_samples}")
        print(f"  N_rx      = {self.N_rx}")
        print(f"  ---")
        print(f"  range_res = {self.range_res:.3f} m")
        print(f"  max_range = {self.max_range:.1f} m")
        print(f"  vel_res   = {self.vel_res:.3f} m/s")
        print(f"  max_vel   = {self.max_vel:.1f} m/s ({self.max_vel*3.6:.1f} km/h)")


def generate_scene(radar, targets, snr_db=20.0, seed=42):
    """다중 표적 FMCW beat signal 생성 (해석적 모델).

    Parameters
    ----------
    radar : FMCWRadar
        레이다 시스템 객체
    targets : list of dict
        각 표적: {'range': R [m], 'velocity': v [m/s], 'rcs': sigma, 'angle': theta [deg]}
        angle은 N_rx > 1일 때만 사용
    snr_db : float
        단일 샘플 기준 SNR [dB] (가장 강한 표적 기준)
    seed : int
        랜덤 시드

    Returns
    -------
    signal : ndarray, shape (N_rx, N_chirps, N_samples)
        beat signal 텐서 (complex)
    """
    rng = np.random.default_rng(seed)
    N_rx = radar.N_rx
    Nc = radar.N_chirps
    Ns = radar.N_samples

    signal = np.zeros((N_rx, Nc, Ns), dtype=np.complex128)

    # chirp 내 시간 (fast-time)
    t_fast = np.arange(Ns) / radar.fs  # (Ns,)

    for tgt in targets:
        R = tgt['range']
        v = tgt.get('velocity', 0.0)
        rcs = tgt.get('rcs', 1.0)
        theta_deg = tgt.get('angle', 0.0)

        # beat frequency와 도플러 주파수
        tau = 2 * R / C
        f_beat = radar.mu * tau  # beat 주파수
        f_d = 2 * v / radar.lam  # 도플러 주파수

        # 진폭 (RCS에 비례, 단순화)
        amp = np.sqrt(rcs)

        for m in range(Nc):
            # fast-time: beat frequency
            fast_phase = 2 * np.pi * f_beat * t_fast
            # slow-time: 도플러 위상 (chirp 간)
            slow_phase = 2 * np.pi * f_d * m * radar.T_chirp
            # 거리 변화에 의한 beat 주파수 미세 변화 (무시 가능하지만 포함)
            beat = amp * np.exp(1j * (fast_phase + slow_phase))

            for rx in range(N_rx):
                # 공간 위상 (안테나 간 위상 차이)
                spatial_phase = 2 * np.pi * rx * radar.d_rx * np.sin(np.radians(theta_deg)) / radar.lam
                signal[rx, m, :] += beat * np.exp(1j * spatial_phase)

    # 잡음 추가
    max_power = np.max(np.abs(signal) ** 2)
    noise_power = max_power / (10 ** (snr_db / 10))
    noise_std = np.sqrt(noise_power / 2)
    noise = noise_std * (rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape))
    signal += noise

    return signal


def range_fft(signal, window='hann'):
    """Range FFT (fast-time 방향).

    Parameters
    ----------
    signal : ndarray, shape (..., N_samples)
        beat signal (마지막 축이 fast-time)
    window : str
        윈도 함수 ('hann', 'hamming', 'blackman', 'rect')

    Returns
    -------
    range_profile : ndarray
        range FFT 결과 (complex)
    """
    Ns = signal.shape[-1]
    w = _get_window(window, Ns)
    return np.fft.fft(signal * w, axis=-1)


def range_doppler_map(signal, window_range='hann', window_doppler='hann'):
    """Range-Doppler Map 생성 (2D FFT).

    Parameters
    ----------
    signal : ndarray, shape (..., N_chirps, N_samples)
        beat signal 텐서
    window_range : str
        range FFT 윈도
    window_doppler : str
        Doppler FFT 윈도

    Returns
    -------
    rdm : ndarray, shape (..., N_chirps, N_samples)
        Range-Doppler Map (complex, fftshift 적용됨)
    """
    Nc = signal.shape[-2]
    Ns = signal.shape[-1]

    # 1st FFT: range (fast-time, 마지막 축)
    w_r = _get_window(window_range, Ns)
    rng_fft = np.fft.fft(signal * w_r, axis=-1)

    # 2nd FFT: Doppler (slow-time, 뒤에서 두번째 축)
    w_d = _get_window(window_doppler, Nc)
    # w_d shape을 signal 차원에 맞춤
    w_d_shape = [1] * signal.ndim
    w_d_shape[-2] = Nc
    w_d = w_d.reshape(w_d_shape)

    rdm = np.fft.fftshift(np.fft.fft(rng_fft * w_d, axis=-2), axes=-2)
    return rdm


def range_angle_map(signal, radar, window_range='hann', window_angle='hann', N_angle=64):
    """Range-Angle Map 생성.

    Parameters
    ----------
    signal : ndarray, shape (N_rx, N_chirps, N_samples)
    radar : FMCWRadar
    N_angle : int
        angle FFT 크기 (zero-padding)

    Returns
    -------
    ram : ndarray, shape (N_angle, N_samples)
        Range-Angle Map (크기, 도플러는 코히런트 적분)
    angle_axis : ndarray
        각도 축 [deg]
    """
    N_rx = signal.shape[0]

    # 먼저 range FFT
    w_r = _get_window(window_range, signal.shape[-1])
    rng_fft = np.fft.fft(signal * w_r, axis=-1)

    # 도플러 방향 코히런트 적분 (평균)
    rng_avg = np.mean(rng_fft, axis=1)  # (N_rx, N_samples)

    # spatial FFT (안테나 축)
    w_a = _get_window(window_angle, N_rx)
    rng_avg_windowed = rng_avg * w_a[:, np.newaxis]

    # zero-padding → angle FFT
    padded = np.zeros((N_angle, rng_avg.shape[1]), dtype=np.complex128)
    padded[:N_rx, :] = rng_avg_windowed
    ram = np.fft.fftshift(np.fft.fft(padded, axis=0), axes=0)

    # 각도 축 계산
    u = np.fft.fftshift(np.fft.fftfreq(N_angle))  # -0.5 ~ 0.5
    angle_axis = np.degrees(np.arcsin(np.clip(u / (radar.d_rx / radar.lam), -1, 1)))

    return ram, angle_axis


def ca_cfar_1d(signal_mag, guard_cells=2, train_cells=8, pfa=1e-4):
    """1D CA-CFAR 탐지.

    Parameters
    ----------
    signal_mag : ndarray, shape (N,)
        입력 신호의 크기 (magnitude 또는 power)
    guard_cells : int
        한쪽 guard cell 수
    train_cells : int
        한쪽 training cell 수
    pfa : float
        목표 오경보 확률

    Returns
    -------
    detections : ndarray (bool)
        탐지 여부
    threshold : ndarray
        적응 문턱값
    """
    N = len(signal_mag)
    power = signal_mag ** 2
    alpha = 2 * train_cells * (pfa ** (-1.0 / (2 * train_cells)) - 1)

    detections = np.zeros(N, dtype=bool)
    threshold = np.zeros(N)

    margin = guard_cells + train_cells

    for i in range(margin, N - margin):
        # 왼쪽 training cells
        left = power[i - margin:i - guard_cells]
        # 오른쪽 training cells
        right = power[i + guard_cells + 1:i + margin + 1]
        noise_est = np.mean(np.concatenate([left, right]))
        T = alpha * noise_est
        threshold[i] = T
        if power[i] > T:
            detections[i] = True

    return detections, np.sqrt(threshold)


def ca_cfar_2d(rdm_mag, guard=(2, 2), train=(4, 4), pfa=1e-4):
    """2D CA-CFAR 탐지.

    Parameters
    ----------
    rdm_mag : ndarray, shape (Nd, Nr)
    guard : tuple (guard_doppler, guard_range)
    train : tuple (train_doppler, train_range)
    pfa : float

    Returns
    -------
    detections : ndarray (bool), shape (Nd, Nr)
    """
    Nd, Nr = rdm_mag.shape
    power = rdm_mag ** 2
    gd, gr = guard
    td, tr = train

    n_train = (2*(gd+td)+1) * (2*(gr+tr)+1) - (2*gd+1) * (2*gr+1)
    alpha = n_train * (pfa ** (-1.0 / n_train) - 1)

    kernel = np.ones((2*(gd+td)+1, 2*(gr+tr)+1), dtype=np.float64)
    kernel[td:td+2*gd+1, tr:tr+2*gr+1] = 0.0
    noise_sum = convolve2d(power, kernel, mode='same', boundary='fill', fillvalue=0.0)
    threshold = alpha * noise_sum / n_train

    detections = power > threshold
    margin_d = gd + td
    margin_r = gr + tr
    if margin_d:
        detections[:margin_d, :] = False
        detections[Nd-margin_d:, :] = False
    if margin_r:
        detections[:, :margin_r] = False
        detections[:, Nr-margin_r:] = False
    return detections


def to_db(x, ref=None):
    """복소수 또는 크기를 dB로 변환."""
    mag = np.abs(x)
    if ref is None:
        ref = np.max(mag)
    return 20 * np.log10(mag / (ref + 1e-30) + 1e-30)


def _get_window(name, N):
    """윈도 함수 생성."""
    if name == 'hann':
        return np.hanning(N)
    elif name == 'hamming':
        return np.hamming(N)
    elif name == 'blackman':
        return np.blackman(N)
    elif name == 'rect' or name == 'none':
        return np.ones(N)
    else:
        return np.hanning(N)
