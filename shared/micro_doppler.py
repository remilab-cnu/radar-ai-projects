"""Micro-Doppler 시뮬레이터 — 교육용

Boulic kinematic body model (11-segment)을 이용한
인체 micro-Doppler 서명 생성 및 STFT 스펙트로그램 변환.

사용법:
    from shared.micro_doppler import generate_har_sample, ACTIVITY_LABELS
"""

import numpy as np
from scipy.signal import stft as scipy_stft
from scipy.ndimage import zoom

C = 299_792_458.0

ACTIVITY_LABELS = ['walk', 'run', 'sit_down', 'fall', 'wave', 'idle']
ACTIVITY_TO_IDX = {a: i for i, a in enumerate(ACTIVITY_LABELS)}
N_CLASSES = len(ACTIVITY_LABELS)


# ─── Body Model ───────────────────────────────────────────────────────────────

BODY_SEGMENTS = {
    'torso':       {'length': 0.50, 'rcs': 1.00},
    'head':        {'length': 0.20, 'rcs': 0.30},
    'upper_arm_L': {'length': 0.28, 'rcs': 0.15},
    'lower_arm_L': {'length': 0.25, 'rcs': 0.10},
    'upper_arm_R': {'length': 0.28, 'rcs': 0.15},
    'lower_arm_R': {'length': 0.25, 'rcs': 0.10},
    'upper_leg_L': {'length': 0.43, 'rcs': 0.25},
    'lower_leg_L': {'length': 0.43, 'rcs': 0.15},
    'upper_leg_R': {'length': 0.43, 'rcs': 0.25},
    'lower_leg_R': {'length': 0.43, 'rcs': 0.15},
}

HIP_HEIGHT = 0.86  # upper_leg + lower_leg when standing


# ─── Activity Joint Angle Models ──────────────────────────────────────────────

def _joint_angles_walk(t, params):
    """Walking: Boulic-inspired periodic gait."""
    f = params.get('gait_freq', 1.0)
    phi = 2 * np.pi * f * t + params.get('phase_offset', 0)
    v = params.get('bulk_vel', 1.2) * params.get('direction', 1.0)

    A_hip = np.radians(params.get('hip_amp', 25))
    A_knee = np.radians(params.get('knee_amp', 40))
    A_sh = np.radians(params.get('shoulder_amp', 15))
    A_el = np.radians(params.get('elbow_amp', 20))

    return {
        'hip_x': v * t,
        'hip_z': HIP_HEIGHT + 0.02 * np.sin(2 * phi),
        'torso_lean': np.full_like(t, np.radians(5)),
        'hip_L': A_hip * np.sin(phi),
        'knee_L': -A_knee * np.clip(np.sin(phi - 0.8), 0, 1),
        'hip_R': A_hip * np.sin(phi + np.pi),
        'knee_R': -A_knee * np.clip(np.sin(phi + np.pi - 0.8), 0, 1),
        'shoulder_L': -A_sh * np.sin(phi),
        'elbow_L': -A_el * np.clip(np.sin(phi + 0.5), 0, 1),
        'shoulder_R': -A_sh * np.sin(phi + np.pi),
        'elbow_R': -A_el * np.clip(np.sin(phi + np.pi + 0.5), 0, 1),
    }


def _joint_angles_run(t, params):
    """Running: faster gait, larger amplitudes."""
    f = params.get('gait_freq', 2.5)
    phi = 2 * np.pi * f * t + params.get('phase_offset', 0)
    v = params.get('bulk_vel', 3.0) * params.get('direction', 1.0)

    A_hip = np.radians(params.get('hip_amp', 40))
    A_knee = np.radians(params.get('knee_amp', 60))
    A_sh = np.radians(params.get('shoulder_amp', 30))
    A_el = np.radians(params.get('elbow_amp', 50))

    return {
        'hip_x': v * t,
        'hip_z': HIP_HEIGHT + 0.05 * np.sin(2 * phi),
        'torso_lean': np.full_like(t, np.radians(10)),
        'hip_L': A_hip * np.sin(phi),
        'knee_L': -A_knee * np.clip(np.sin(phi - 0.6), 0, 1),
        'hip_R': A_hip * np.sin(phi + np.pi),
        'knee_R': -A_knee * np.clip(np.sin(phi + np.pi - 0.6), 0, 1),
        'shoulder_L': -A_sh * np.sin(phi),
        'elbow_L': -A_el * np.clip(np.sin(phi + 0.3), 0, 1) - np.radians(20),
        'shoulder_R': -A_sh * np.sin(phi + np.pi),
        'elbow_R': -A_el * np.clip(np.sin(phi + np.pi + 0.3), 0, 1) - np.radians(20),
    }


def _joint_angles_sit_down(t, params):
    """Sit-down: transient hip/knee flexion."""
    duration = params.get('duration', 1.5)
    t_start = params.get('t_start', 0.3)
    t_eff = np.clip(t - t_start, 0, None)
    progress = np.clip(t_eff / duration, 0, 1)
    s = 0.5 * (1 - np.cos(np.pi * progress))

    return {
        'hip_x': np.zeros_like(t),
        'hip_z': HIP_HEIGHT - 0.40 * s,
        'torso_lean': np.radians(25) * s,
        'hip_L': np.radians(90) * s,
        'knee_L': -np.radians(90) * s,
        'hip_R': np.radians(90) * s,
        'knee_R': -np.radians(90) * s,
        'shoulder_L': np.radians(10) * s,
        'elbow_L': -np.radians(30) * s,
        'shoulder_R': np.radians(10) * s,
        'elbow_R': -np.radians(30) * s,
    }


def _joint_angles_fall(t, params):
    """Fall: rapid collapse with flailing limbs."""
    t_collapse = params.get('t_collapse', 0.5)
    t_start = params.get('t_start', 0.2)
    direction = params.get('direction', 1.0)
    t_eff = np.clip(t - t_start, 0, None)
    progress = np.clip(t_eff / t_collapse, 0, 1)
    s = progress ** 2  # accelerating

    flail = (1 - progress) * np.sin(8 * np.pi * t_eff)

    return {
        'hip_x': direction * 0.3 * s,
        'hip_z': HIP_HEIGHT - 0.70 * s,
        'torso_lean': direction * np.radians(80) * s,
        'hip_L': np.radians(30) * s + np.radians(20) * flail,
        'knee_L': -np.radians(50) * s,
        'hip_R': np.radians(30) * s + np.radians(15) * flail,
        'knee_R': -np.radians(50) * s,
        'shoulder_L': direction * np.radians(60) * s + np.radians(30) * flail,
        'elbow_L': -np.radians(40) * s,
        'shoulder_R': direction * np.radians(60) * s + np.radians(25) * flail,
        'elbow_R': -np.radians(40) * s,
    }


def _joint_angles_wave(t, params):
    """Wave: right arm oscillation, body mostly still."""
    f = params.get('wave_freq', 2.5)
    phi = 2 * np.pi * f * t + params.get('phase_offset', 0)

    return {
        'hip_x': np.zeros_like(t),
        'hip_z': np.full_like(t, HIP_HEIGHT),
        'torso_lean': np.radians(2) * np.sin(0.5 * phi),
        'hip_L': np.radians(2) * np.sin(0.3 * phi),
        'knee_L': np.zeros_like(t),
        'hip_R': np.radians(2) * np.sin(0.3 * phi + np.pi),
        'knee_R': np.zeros_like(t),
        'shoulder_L': np.radians(5) * np.sin(0.5 * phi),
        'elbow_L': np.full_like(t, -np.radians(15)),
        'shoulder_R': np.radians(120) + np.radians(30) * np.sin(phi),
        'elbow_R': -np.radians(40) + np.radians(30) * np.sin(phi * 1.5),
    }


def _joint_angles_idle(t, params):
    """Idle: breathing + slight sway."""
    f_br = params.get('breath_freq', 0.25)
    phi = 2 * np.pi * f_br * t
    sway = params.get('sway', 0.005)

    return {
        'hip_x': sway * np.sin(0.3 * phi),
        'hip_z': HIP_HEIGHT + 0.005 * np.sin(phi),
        'torso_lean': np.radians(1) * np.sin(phi),
        'hip_L': np.radians(1) * np.sin(0.2 * phi),
        'knee_L': np.zeros_like(t),
        'hip_R': np.radians(1) * np.sin(0.2 * phi + np.pi),
        'knee_R': np.zeros_like(t),
        'shoulder_L': np.radians(1) * np.sin(0.3 * phi),
        'elbow_L': np.zeros_like(t),
        'shoulder_R': np.radians(1) * np.sin(0.3 * phi + np.pi),
        'elbow_R': np.zeros_like(t),
    }


_ACTIVITY_FN = {
    'walk': _joint_angles_walk,
    'run': _joint_angles_run,
    'sit_down': _joint_angles_sit_down,
    'fall': _joint_angles_fall,
    'wave': _joint_angles_wave,
    'idle': _joint_angles_idle,
}


# ─── Forward Kinematics ──────────────────────────────────────────────────────

def _forward_kinematics(angles):
    """Compute segment midpoint positions (x, z) from joint angles.

    Angle conventions (all in radians):
        torso_lean — forward lean from vertical (positive = forward)
        hip_L/R — leg swing from vertical-down (positive = forward)
        knee_L/R — relative to upper leg (negative = flex backward)
        shoulder_L/R — arm swing from vertical-down (positive = forward)
        elbow_L/R — relative to upper arm (negative = flex backward)
    """
    hip_x = angles['hip_x']
    hip_z = angles['hip_z']
    L = BODY_SEGMENTS
    positions = {}

    # Torso: from hip upward
    torso_abs = np.pi / 2 - angles['torso_lean']
    torso_end_x = hip_x + L['torso']['length'] * np.cos(torso_abs)
    torso_end_z = hip_z + L['torso']['length'] * np.sin(torso_abs)
    positions['torso'] = ((hip_x + torso_end_x) / 2,
                          (hip_z + torso_end_z) / 2)

    # Head: continues from torso end
    head_end_x = torso_end_x + L['head']['length'] * np.cos(torso_abs)
    head_end_z = torso_end_z + L['head']['length'] * np.sin(torso_abs)
    positions['head'] = ((torso_end_x + head_end_x) / 2,
                         (torso_end_z + head_end_z) / 2)

    # Shoulders at torso top
    sh_x, sh_z = torso_end_x, torso_end_z

    for side in ['L', 'R']:
        # Legs (from hip downward)
        hip_abs = -np.pi / 2 + angles[f'hip_{side}']
        ul = L[f'upper_leg_{side}']['length']
        ul_end_x = hip_x + ul * np.cos(hip_abs)
        ul_end_z = hip_z + ul * np.sin(hip_abs)
        positions[f'upper_leg_{side}'] = ((hip_x + ul_end_x) / 2,
                                          (hip_z + ul_end_z) / 2)

        knee_abs = hip_abs + angles[f'knee_{side}']
        ll = L[f'lower_leg_{side}']['length']
        ll_end_x = ul_end_x + ll * np.cos(knee_abs)
        ll_end_z = ul_end_z + ll * np.sin(knee_abs)
        positions[f'lower_leg_{side}'] = ((ul_end_x + ll_end_x) / 2,
                                          (ul_end_z + ll_end_z) / 2)

        # Arms (from shoulder downward)
        sh_abs = -np.pi / 2 + angles[f'shoulder_{side}']
        ua = L[f'upper_arm_{side}']['length']
        ua_end_x = sh_x + ua * np.cos(sh_abs)
        ua_end_z = sh_z + ua * np.sin(sh_abs)
        positions[f'upper_arm_{side}'] = ((sh_x + ua_end_x) / 2,
                                          (sh_z + ua_end_z) / 2)

        el_abs = sh_abs + angles[f'elbow_{side}']
        la = L[f'lower_arm_{side}']['length']
        la_end_x = ua_end_x + la * np.cos(el_abs)
        la_end_z = ua_end_z + la * np.sin(el_abs)
        positions[f'lower_arm_{side}'] = ((ua_end_x + la_end_x) / 2,
                                          (ua_end_z + la_end_z) / 2)

    return positions


# ─── Signal Generation ────────────────────────────────────────────────────────

def generate_micro_doppler_signal(activity, params, fc=77e9, prf=10000,
                                   duration=3.0, snr_db=15.0, rng=None,
                                   aspect_angle=0.0):
    """Generate CW micro-Doppler baseband signal.

    Parameters
    ----------
    activity : str
    params : dict — activity-specific parameters
    fc : float — carrier frequency [Hz]
    prf : float — sampling rate [Hz]
    duration : float — observation time [s]
    snr_db : float
    rng : np.random.Generator
    aspect_angle : float — angle between radar LOS and walking direction [deg]

    Returns
    -------
    signal : ndarray (N,) complex
    """
    if rng is None:
        rng = np.random.default_rng()

    lam = C / fc
    N = int(prf * duration)
    t = np.arange(N) / prf

    angles = _ACTIVITY_FN[activity](t, params)
    positions = _forward_kinematics(angles)

    cos_aspect = np.cos(np.radians(aspect_angle))

    signal = np.zeros(N, dtype=np.complex128)
    for seg_name, seg_info in BODY_SEGMENTS.items():
        x_k = positions[seg_name][0] * cos_aspect  # radial direction
        phase = -4 * np.pi / lam * x_k
        signal += seg_info['rcs'] * np.exp(1j * phase)

    # Noise
    sig_power = np.mean(np.abs(signal) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(N) + 1j * rng.standard_normal(N))
    signal += noise

    return signal


def signal_to_spectrogram(signal, prf, n_fft=256, hop=64, output_size=(128, 128)):
    """STFT spectrogram (dB, normalized [0,1]).

    Parameters
    ----------
    signal : ndarray (N,) complex
    prf : float
    n_fft, hop : int
    output_size : tuple (H, W)

    Returns
    -------
    spec : ndarray (H, W) float32
    """
    _, _, Zxx = scipy_stft(signal, fs=prf, nperseg=n_fft,
                           noverlap=n_fft - hop, return_onesided=False)
    Zxx = np.fft.fftshift(Zxx, axes=0)

    mag = np.abs(Zxx)
    mag_db = 20 * np.log10(mag / (mag.max() + 1e-30) + 1e-30)
    mag_db = np.clip(mag_db, -60, 0)
    spec = (mag_db + 60) / 60.0

    if spec.shape != output_size:
        zoom_factors = (output_size[0] / spec.shape[0],
                        output_size[1] / spec.shape[1])
        spec = zoom(spec, zoom_factors, order=1)

    return spec.astype(np.float32)


# ─── Random Parameter Generation ─────────────────────────────────────────────

def _random_activity_params(activity, rng):
    """Randomized activity parameters for data diversity."""
    if activity == 'walk':
        return {
            'gait_freq': rng.uniform(0.8, 1.2),
            'bulk_vel': rng.uniform(0.8, 1.6),
            'hip_amp': rng.uniform(20, 30),
            'knee_amp': rng.uniform(30, 50),
            'shoulder_amp': rng.uniform(10, 20),
            'elbow_amp': rng.uniform(15, 25),
            'direction': rng.choice([-1.0, 1.0]),
            'phase_offset': rng.uniform(0, 2 * np.pi),
        }
    elif activity == 'run':
        return {
            'gait_freq': rng.uniform(2.0, 3.0),
            'bulk_vel': rng.uniform(2.0, 4.5),
            'hip_amp': rng.uniform(35, 50),
            'knee_amp': rng.uniform(50, 70),
            'shoulder_amp': rng.uniform(25, 40),
            'elbow_amp': rng.uniform(40, 60),
            'direction': rng.choice([-1.0, 1.0]),
            'phase_offset': rng.uniform(0, 2 * np.pi),
        }
    elif activity == 'sit_down':
        return {
            'duration': rng.uniform(1.0, 2.5),
            't_start': rng.uniform(0.1, 0.5),
        }
    elif activity == 'fall':
        return {
            't_collapse': rng.uniform(0.3, 0.8),
            't_start': rng.uniform(0.1, 0.5),
            'direction': rng.choice([-1.0, 1.0]),
        }
    elif activity == 'wave':
        return {
            'wave_freq': rng.uniform(1.5, 3.5),
            'phase_offset': rng.uniform(0, 2 * np.pi),
        }
    elif activity == 'idle':
        return {
            'breath_freq': rng.uniform(0.15, 0.35),
            'sway': rng.uniform(0.002, 0.01),
        }
    return {}


def generate_har_sample(activity, rng, fc=77e9, prf=10000, duration=3.0,
                         snr_db=15.0, output_size=(128, 128),
                         aspect_angle=None):
    """End-to-end: activity → spectrogram + label.

    Parameters
    ----------
    aspect_angle : float or None — radar-to-motion aspect angle [deg].
        If None, defaults to 0.0 (backward compatible).

    Returns
    -------
    spectrogram : ndarray (H, W) float32
    label : int
    meta : dict
    """
    if aspect_angle is None:
        aspect_angle = 0.0

    params = _random_activity_params(activity, rng)
    signal = generate_micro_doppler_signal(
        activity, params, fc=fc, prf=prf,
        duration=duration, snr_db=snr_db, rng=rng,
        aspect_angle=aspect_angle)
    spec = signal_to_spectrogram(signal, prf, output_size=output_size)

    return spec, ACTIVITY_TO_IDX[activity], {
        'activity': activity, 'snr_db': snr_db,
        'aspect_angle': aspect_angle,
    }


# ─── Handcrafted Features (SVM baseline) ─────────────────────────────────────

def extract_handcrafted_features(spectrogram, n_svd=5):
    """Extract features: bandwidth, centroid, SVD, entropy, periodicity.

    Parameters
    ----------
    spectrogram : ndarray (H, W) — [0, 1] normalized

    Returns
    -------
    features : ndarray (N_FEATURES,) float32
    """
    spec = spectrogram.copy()
    H, W = spec.shape
    features = []
    freq_axis = np.linspace(-1, 1, H)

    # Doppler profile (time-averaged)
    doppler_profile = spec.mean(axis=1)

    # 1. Bandwidth (-3dB)
    thresh = 0.5 * doppler_profile.max()
    features.append(np.sum(doppler_profile > thresh) / H)

    # 2-3. Centroid & std
    dp_sum = doppler_profile.sum() + 1e-10
    centroid = np.sum(freq_axis * doppler_profile) / dp_sum
    std = np.sqrt(np.sum((freq_axis - centroid) ** 2 * doppler_profile) / dp_sum)
    features.extend([centroid, std])

    # 4-5. Time-varying bandwidth stats
    time_bw = np.zeros(W)
    for j in range(W):
        col = spec[:, j]
        cs = col.sum() + 1e-10
        cc = np.sum(freq_axis * col) / cs
        time_bw[j] = np.sqrt(np.sum((freq_axis - cc) ** 2 * col) / cs)
    features.extend([time_bw.mean(), time_bw.std()])

    # 6-10. SVD singular values
    U, S, Vh = np.linalg.svd(spec, full_matrices=False)
    features.extend(S[:n_svd].tolist())

    # 11. SVD concentration
    features.append(S[0] / (S.sum() + 1e-10))

    # 12. Spectral entropy (averaged)
    entropies = np.zeros(W)
    for j in range(W):
        p = spec[:, j] / (spec[:, j].sum() + 1e-10)
        p = p[p > 0]
        entropies[j] = -np.sum(p * np.log2(p + 1e-30))
    features.append(entropies.mean())

    # 13. Total power
    features.append(spec.mean())

    # 14-15. Periodicity (autocorrelation)
    power_t = spec.sum(axis=0)
    power_t = power_t - power_t.mean()
    acf = np.correlate(power_t, power_t, mode='full')
    acf = acf[len(acf) // 2:]
    acf = acf / (acf[0] + 1e-10)
    peaks = [(i, acf[i]) for i in range(2, len(acf) - 1)
             if acf[i] > acf[i - 1] and acf[i] > acf[i + 1]]
    if peaks:
        best = max(peaks, key=lambda x: x[1])
        features.extend([best[0] / W, best[1]])
    else:
        features.extend([0.0, 0.0])

    # 16. Max Doppler extent
    features.append(doppler_profile.max())

    # ── Physical Doppler features (17-19) ──

    # 17. Max occupied Doppler extent (fraction of bins above noise floor)
    #     Distinguishes idle(~0) vs walk(~0.3) vs run(~0.7)
    noise_floor = np.percentile(doppler_profile, 10)
    occupied = doppler_profile > (noise_floor + 0.1 * (doppler_profile.max() - noise_floor))
    max_occ_extent = 0.0
    if occupied.any():
        occ_indices = np.where(occupied)[0]
        max_occ_extent = (occ_indices[-1] - occ_indices[0] + 1) / H
    features.append(max_occ_extent)

    # 18. DC-to-total energy ratio (zero-Doppler concentration)
    #     High for idle/wave (stationary body), low for walk/run (moving body)
    dc_band = max(1, H // 16)  # ~±3% of Doppler range
    dc_energy = spec[H // 2 - dc_band:H // 2 + dc_band, :].sum()
    total_energy = spec.sum() + 1e-10
    features.append(dc_energy / total_energy)

    # 19. Positive/negative Doppler asymmetry (excluding DC bin)
    #     Walk/run are roughly symmetric; wave is strongly asymmetric (one arm)
    dc_half = max(1, H // 32)  # exclude narrow DC band from both sides
    pos_energy = spec[H // 2 + dc_half:, :].sum()
    neg_energy = spec[:H // 2 - dc_half, :].sum()
    features.append((pos_energy - neg_energy) / (pos_energy + neg_energy + 1e-10))

    # ── Temporal transient features (20-22) ──
    # Use off-DC envelope to avoid stationary body energy dominating

    dc_suppress = spec.copy()
    dc_suppress[H // 2 - dc_band:H // 2 + dc_band, :] = 0  # suppress DC band
    power_envelope = dc_suppress.sum(axis=0)  # (W,) — motion-only power per frame
    pe_max = power_envelope.max() + 1e-10
    pe_norm = power_envelope / pe_max

    # 20. Onset time (normalized): when does significant motion start?
    #     Fall/sit_down start mid-sequence; walk/run are continuous
    onset_thresh = 0.3
    onset_frames = np.where(pe_norm > onset_thresh)[0]
    onset_time = onset_frames[0] / W if len(onset_frames) > 0 else 1.0
    features.append(onset_time)

    # 21. Power decay rate: slope of envelope in second half
    #     Fall: sharp rise then flat; sit_down: gradual rise then flat
    half = W // 2
    if half > 2:
        second_half = pe_norm[half:]
        t = np.arange(len(second_half), dtype=np.float64)
        if t.std() > 0:
            slope = np.polyfit(t, second_half, 1)[0]
        else:
            slope = 0.0
        features.append(float(slope))
    else:
        features.append(0.0)

    # 22. Active duration ratio: fraction of frames with significant power
    #     Walk/run ≈ 1.0, fall ≈ 0.3, sit_down ≈ 0.5
    active_ratio = np.sum(pe_norm > onset_thresh) / W
    features.append(active_ratio)

    return np.array(features, dtype=np.float32)
