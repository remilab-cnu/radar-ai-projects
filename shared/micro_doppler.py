"""Micro-Doppler 시뮬레이터 — 교육용

Boulic kinematic body model (11-segment)을 이용한
인체 micro-Doppler 서명 생성 및 STFT 스펙트로그램 변환.

사용법:
    from shared.micro_doppler import generate_har_sample, ACTIVITY_LABELS
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import stft as scipy_stft
from scipy.ndimage import zoom

from shared.fmcw_simulator import FMCWRadar, add_complex_awgn, range_axis

C = 299_792_458.0
P02_DOPPLER_ALIAS_SAFETY_FACTOR = 0.90

ACTIVITY_LABELS = ['walk', 'run', 'sit_down', 'fall', 'wave', 'idle']
ACTIVITY_TO_IDX = {a: i for i, a in enumerate(ACTIVITY_LABELS)}
N_CLASSES = len(ACTIVITY_LABELS)


def radar_max_unambiguous_velocity_mps(radar, prf=None):
    """Return one-sided slow-time Doppler Nyquist velocity for monostatic radar."""
    slow_prf = (1.0 / radar.PRI) if prf is None else float(prf)
    return float(radar.lam * slow_prf / 4.0)


@dataclass(frozen=True)
class P02PedestrianScatterer:
    """One P02-only pedestrian scatterer.

    This mirrors the useful part of the radar-deconv scatter scene idea
    (multiple weighted scatterers per target, optional micro-displacement)
    without depending on that research package or changing P01/P03.
    """

    name: str
    parent_segment: str
    scatter_kind: str
    range_m: np.ndarray
    radial_velocity_mps: np.ndarray
    rcs_m2: float
    amplitude_weight: float
    range_offset_m: float
    initial_phase_rad: float
    micro_disp_amp_m: float
    micro_freq_hz: float
    micro_phase_rad: float


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


def _segment_scatter_kind(segment_name):
    if segment_name == "torso":
        return "torso"
    if segment_name == "head":
        return "head"
    return "limb"


def _activity_micro_frequency_hz(activity, params, rng):
    """Activity-aware residual micro-displacement frequency.

    The main Doppler comes from the Boulic-style segment kinematics.  These
    small residual oscillations follow the radar-deconv pedestrian-scatterer
    style and add sub-scatterer diversity without replacing the body model.
    """
    if activity == "run":
        base = params.get("gait_freq", 2.5)
        # Keep the classroom run class inside the P02 slow-time Doppler
        # Nyquist region.  Full sprint limb tips at 77 GHz with 10 kHz PRF can
        # alias, which is a useful advanced lesson but not the default HAR set.
        return float(rng.uniform(1.5, 2.5) * base)
    if activity == "walk":
        base = params.get("gait_freq", 1.0)
        return float(rng.uniform(6.0, 10.0) * base)
    if activity == "wave":
        base = params.get("wave_freq", 2.5)
        return float(rng.uniform(1.5, 3.0) * base)
    if activity == "fall":
        base = 1.0 / max(float(params.get("t_collapse", 0.5)), 1e-3)
        return float(rng.uniform(1.0, 3.0) * base)
    if activity == "sit_down":
        base = 1.0 / max(float(params.get("duration", 1.5)), 1e-3)
        return float(rng.uniform(2.0, 5.0) * base)
    if activity == "idle":
        return float(rng.uniform(0.4, 1.5))
    return float(rng.uniform(4.0, 12.0))


def _segment_scatter_templates(segment_name, rng):
    """Return P02 scatter templates for one body segment.

    The template shapes intentionally follow the radar-deconv pedestrian model:
    torso/head are stable scatterers; limbs are several lower-weight scatterers
    with small residual micro-displacements.
    """
    kind = _segment_scatter_kind(segment_name)
    length = float(BODY_SEGMENTS[segment_name]["length"])

    if kind == "torso":
        # radar-deconv pedestrian torso: 2--3 high-weight, no micro motion.
        n = int(rng.integers(2, 4))
        return [
            {
                "offset": float(rng.normal(0.0, 0.08)),
                "weight": float(rng.uniform(0.6, 1.0)),
                "micro_amp": 0.0,
                "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
            }
            for _ in range(n)
        ]

    if kind == "head":
        return [
            {
                "offset": float(rng.normal(0.0, 0.04)),
                "weight": 1.0,
                "micro_amp": 0.0,
                "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
            }
        ]

    # radar-deconv pedestrian limbs: several weaker scatterers across the
    # pedestrian extent with small micro-displacement terms.
    n = 2
    return [
        {
            "offset": float(rng.uniform(-0.45 * length, 0.45 * length)),
            "weight": float(rng.uniform(0.12, 0.40)),
            "micro_amp": float(rng.uniform(0.004, 0.010)),
            "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
        }
        for _ in range(n)
    ]


def build_p02_pedestrian_scatterers(
    activity, params, positions, range_m, cos_aspect, t, rng
):
    """Expand body segments into P02-only radar-deconv-inspired scatterers."""
    scatterers = []
    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0

    for segment_name, segment_info in BODY_SEGMENTS.items():
        templates = _segment_scatter_templates(segment_name, rng)
        template_weights = np.asarray(
            [tpl["weight"] for tpl in templates], dtype=np.float64
        )
        template_weights = np.maximum(template_weights, 1e-6)
        template_weights /= float(template_weights.sum())
        base_rcs = float(segment_info["rcs"])
        base_radial = (
            np.asarray(positions[segment_name][0], dtype=np.float64) * cos_aspect
        )
        kind = _segment_scatter_kind(segment_name)

        for idx, tpl in enumerate(templates):
            micro_amp = float(tpl["micro_amp"])
            micro_freq = 0.0
            micro_phase = float(tpl["phase"])
            if micro_amp > 0.0:
                micro_freq = _activity_micro_frequency_hz(activity, params, rng)
            micro_disp = (
                micro_amp * np.sin(2.0 * np.pi * micro_freq * t + micro_phase)
                if micro_amp > 0.0 and micro_freq > 0.0
                else 0.0
            )
            # Template offsets and residual micro-displacements are along the
            # walking/radial kinematic axis in this 2-D teaching model, so they
            # should follow the same aspect projection as the segment motion.
            scatter_range = (
                range_m
                + base_radial
                + (float(tpl["offset"]) + micro_disp) * cos_aspect
            )
            if len(t) > 1:
                radial_velocity = np.gradient(scatter_range, dt).astype(np.float64)
            else:
                radial_velocity = np.zeros_like(scatter_range, dtype=np.float64)

            scatterers.append(
                P02PedestrianScatterer(
                    name=f"{segment_name}_{idx}",
                    parent_segment=segment_name,
                    scatter_kind=kind,
                    range_m=np.asarray(scatter_range, dtype=np.float64),
                    radial_velocity_mps=radial_velocity,
                    rcs_m2=float(base_rcs * template_weights[idx]),
                    amplitude_weight=float(template_weights[idx]),
                    range_offset_m=float(tpl["offset"]),
                    initial_phase_rad=float(rng.uniform(0.0, 2.0 * np.pi)),
                    micro_disp_amp_m=micro_amp,
                    micro_freq_hz=float(micro_freq),
                    micro_phase_rad=micro_phase,
                )
            )

    return scatterers


# ─── Signal Generation ────────────────────────────────────────────────────────

def generate_micro_doppler_signal(activity, params, fc=9.6e9, prf=10000,
                                   duration=3.0, snr_db=15.0, rng=None,
                                   aspect_angle=0.0, radar=None,
                                   range_m=10.0):
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
    radar : FMCWRadar or None
        공통 레이다 physics core. None이면 fc 기준 기본 FMCWRadar를 생성한다.
    range_m : float
        인체 중심 기준 거리 [m]. 세그먼트별 미세 radial displacement는 이 값에
        더해져 radar equation의 R^-4 amplitude에 반영된다.

    Returns
    -------
    signal : ndarray (N,) complex
    """
    if rng is None:
        rng = np.random.default_rng()

    if radar is None:
        radar = FMCWRadar(
            fc=fc,
            bw=50e6,
            T_chirp=2e-6,
            PRI=100e-6,
            N_chirps=64,
            fs=200e6,
            N_rx=1,
        )

    lam = radar.lam
    N = int(prf * duration)
    t = np.arange(N) / prf

    angles = _ACTIVITY_FN[activity](t, params)
    positions = _forward_kinematics(angles)

    cos_aspect = np.cos(np.radians(aspect_angle))

    signal = np.zeros(N, dtype=np.complex128)
    for seg_name, seg_info in BODY_SEGMENTS.items():
        x_k = positions[seg_name][0] * cos_aspect  # radial direction
        phase = -4 * np.pi / lam * x_k
        seg_range = np.maximum(range_m + x_k, radar.range_res)
        sigma = max(float(seg_info['rcs']), 1e-12)
        p_rx = (
            radar.tx_power_w
            * radar.tx_gain_linear
            * radar.rx_gain_linear
            * radar.lam ** 2
            * sigma
            / (((4.0 * np.pi) ** 3) * seg_range ** 4 * radar.system_loss_linear)
        )
        amp = np.sqrt(p_rx)
        signal += amp * np.exp(1j * phase)

    return add_complex_awgn(signal, snr_db=snr_db, rng=rng)


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
            # Controlled teaching run: visually distinct from walking but
            # bounded to avoid Doppler aliasing for the default 77 GHz / 10 kHz
            # slow-time PRF configuration.
            'gait_freq': rng.uniform(1.5, 2.05),
            'bulk_vel': rng.uniform(1.8, 3.1),
            'hip_amp': rng.uniform(25, 36),
            'knee_amp': rng.uniform(35, 50),
            'shoulder_amp': rng.uniform(20, 32),
            'elbow_amp': rng.uniform(30, 46),
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


def generate_range_compressed_micro_doppler_frame(
    activity,
    params,
    fc=9.6e9,
    prf=10000,
    duration=3.0,
    snr_db=15.0,
    rng=None,
    aspect_angle=0.0,
    radar=None,
    range_m=10.0,
    range_window_bins=9,
):
    """Generate a local range-compressed frame for micro-Doppler extraction.

    P02 builds a slow-time range-compressed frame around the human target range,
    then extracts the complex slow-time signal at the simulator-known target
    range bin and computes Doppler/STFT from that selected range.

    The frame is local around the target range instead of a full hundreds-bin
    cube so dataset generation stays tractable.  Each local bin is a physically
    meaningful range bin from the shared radar configuration.

    Returns
    -------
    frame : ndarray (N_chirps, N_local_range) complex
        Local range-compressed frame.
    meta : dict
        Includes global/local range-bin indices and simulator metadata.
    """
    if rng is None:
        rng = np.random.default_rng()

    if radar is None:
        radar = FMCWRadar(
            fc=fc,
            bw=50e6,
            T_chirp=2e-6,
            PRI=1.0 / prf,
            N_chirps=64,
            fs=200e6,
            N_rx=1,
        )
    else:
        fc = radar.fc
        radar_prf = 1.0 / radar.PRI
        if not np.isclose(prf, radar_prf, rtol=1e-6, atol=1e-9):
            prf = radar_prf

    n_slow = int(prf * duration)
    t = np.arange(n_slow, dtype=np.float64) / prf
    angles = _ACTIVITY_FN[activity](t, params)
    positions = _forward_kinematics(angles)

    # Local range-bin window around the human target range.
    full_range_axis = range_axis(radar).astype(np.float64)
    target_bin = int(
        np.clip(round(range_m / radar.range_bin_spacing), 0, radar.N_range_bins - 1)
    )
    half = max(0, int(range_window_bins) // 2)
    start = max(0, target_bin - half)
    stop = min(radar.N_range_bins, target_bin + half + 1)
    range_bin_indices = np.arange(start, stop, dtype=np.int32)
    local_range_axis = full_range_axis[range_bin_indices]

    cos_aspect = float(np.cos(np.radians(aspect_angle)))
    scatterers = build_p02_pedestrian_scatterers(
        activity=activity,
        params=params,
        positions=positions,
        range_m=float(range_m),
        cos_aspect=cos_aspect,
        t=t,
        rng=rng,
    )
    scatter_kind_counts = {
        kind: sum(1 for sc in scatterers if sc.scatter_kind == kind)
        for kind in ("torso", "head", "limb")
    }
    max_abs_radial_velocity_mps = max(
        (
            float(np.max(np.abs(sc.radial_velocity_mps)))
            for sc in scatterers
            if sc.radial_velocity_mps.size
        ),
        default=0.0,
    )
    max_unambiguous_velocity_mps = radar_max_unambiguous_velocity_mps(radar, prf)

    frame = np.zeros((n_slow, len(range_bin_indices)), dtype=np.complex128)
    lam = radar.lam

    for scatterer in scatterers:
        seg_range = np.maximum(scatterer.range_m, radar.range_res)
        sigma = max(float(scatterer.rcs_m2), 1e-12)

        p_rx = (
            radar.tx_power_w
            * radar.tx_gain_linear
            * radar.rx_gain_linear
            * radar.lam**2
            * sigma
            / (((4.0 * np.pi) ** 3) * seg_range**4 * radar.system_loss_linear)
        )
        amp = np.sqrt(p_rx)
        carrier_phase = np.exp(
            -1j * 4.0 * np.pi * seg_range / lam
            + 1j * scatterer.initial_phase_rad
        )

        # Range-compressed point-spread proxy.  The sinc kernel maps each
        # scatterer's continuous range to the local bins; time variation of
        # carrier phase over slow time creates the micro-Doppler signature.
        range_response = np.sinc(
            (local_range_axis[None, :] - seg_range[:, None]) / radar.range_bin_spacing
        )
        frame += (amp * carrier_phase)[:, None] * range_response

    target_range_local = int(np.argmin(np.abs(range_bin_indices - target_bin)))
    reference_power = float(np.mean(np.abs(frame[:, target_range_local]) ** 2))
    frame = add_complex_awgn(
        frame, snr_db=snr_db, rng=rng, reference_power=reference_power
    )

    return frame.astype(np.complex64), {
        "simulator": "range_compressed_target_range_micro_doppler",
        "radar_fc_hz": float(radar.fc),
        "radar_bw_hz": float(radar.bw),
        "radar_fs_hz": float(radar.fs),
        "fs_over_bandwidth": float(radar.fs / radar.bw),
        "slow_time_prf_hz": float(prf),
        "slow_time_samples": int(n_slow),
        "slow_time_duration_s": float(duration),
        "max_abs_radial_velocity_mps": float(max_abs_radial_velocity_mps),
        "radar_max_unambiguous_velocity_mps": float(max_unambiguous_velocity_mps),
        "doppler_alias_safety_factor": float(P02_DOPPLER_ALIAS_SAFETY_FACTOR),
        "doppler_alias_margin_mps": float(
            P02_DOPPLER_ALIAS_SAFETY_FACTOR * max_unambiguous_velocity_mps
            - max_abs_radial_velocity_mps
        ),
        "radar_config_n_chirps": int(radar.N_chirps),
        "up_down_conversion": "excluded_baseband_only",
        "range_processing": "local_range_compressed_frame",
        "doppler_source": "stft_of_target_range_signal",
        "scatter_model": "radar_deconv_inspired_pedestrian_scatterers",
        "scatter_model_scope": "p02_only",
        "n_scatterers": int(len(scatterers)),
        "scatter_kind_counts": scatter_kind_counts,
        "scatter_parent_segments": sorted({sc.parent_segment for sc in scatterers}),
        "target_range_bin": int(target_bin),
        "target_range_m": float(full_range_axis[target_bin]),
        "range_bin_indices": range_bin_indices,
        "range_axis_m": local_range_axis.astype(np.float32),
        "target_range_local_index": int(target_range_local),
        "range_window_bins": int(len(range_bin_indices)),
        "scatterer_summary": [
            {
                "name": sc.name,
                "parent_segment": sc.parent_segment,
                "scatter_kind": sc.scatter_kind,
                "rcs_m2": float(sc.rcs_m2),
                "amplitude_weight": float(sc.amplitude_weight),
                "range_offset_m": float(sc.range_offset_m),
                "micro_disp_amp_m": float(sc.micro_disp_amp_m),
                "micro_freq_hz": float(sc.micro_freq_hz),
                "mean_radial_velocity_mps": float(np.mean(sc.radial_velocity_mps)),
                "max_abs_radial_velocity_mps": float(
                    np.max(np.abs(sc.radial_velocity_mps))
                ),
            }
            for sc in scatterers
        ],
    }


def generate_fmcw_micro_doppler_frame(*args, **kwargs):
    """Deprecated alias; P02 uses target-range extraction, not full FMCW dechirp."""
    return generate_range_compressed_micro_doppler_frame(*args, **kwargs)


def extract_target_range_signal(frame, meta, range_half_width=0):
    """Extract the slow-time signal at the simulator-known target range bin."""
    center = int(meta["target_range_local_index"])
    lo = max(0, center - int(range_half_width))
    hi = min(frame.shape[1], center + int(range_half_width) + 1)
    selected = frame[:, lo:hi]
    if selected.shape[1] == 1:
        return selected[:, 0]
    weights = np.hanning(selected.shape[1] + 2)[1:-1]
    weights = weights / (weights.sum() + 1e-12)
    return selected @ weights


def generate_har_sample(activity, rng, fc=9.6e9, prf=10000, duration=3.0,
                         snr_db=15.0, output_size=(128, 128),
                         aspect_angle=None, range_m=10.0, radar=None,
                         return_debug=False):
    """End-to-end: activity → spectrogram + label.

    Parameters
    ----------
    aspect_angle : float or None — radar-to-motion aspect angle [deg].
        If None, defaults to 0.0 (backward compatible).
    radar : FMCWRadar or None
        Shared radar configuration. The sample is generated as a local
        range-compressed frame, then the complex slow-time signal at the target
        range bin is extracted before STFT. This avoids the earlier range-free
        shortcut.

    Returns
    -------
    spectrogram : ndarray (H, W) float32
    label : int
    meta : dict
    """
    if aspect_angle is None:
        aspect_angle = 0.0

    params = _random_activity_params(activity, rng)
    frame, frame_meta = generate_range_compressed_micro_doppler_frame(
        activity,
        params,
        fc=fc,
        prf=prf,
        duration=duration,
        snr_db=snr_db,
        rng=rng,
        aspect_angle=aspect_angle,
        radar=radar,
        range_m=range_m,
        range_window_bins=9 if return_debug else 1,
    )
    signal = extract_target_range_signal(frame, frame_meta, range_half_width=0)
    spec = signal_to_spectrogram(
        signal, frame_meta['slow_time_prf_hz'], output_size=output_size
    )

    meta = {
        'activity': activity, 'snr_db': snr_db,
        'aspect_angle': aspect_angle,
        'range_m': range_m,
        'target_range_bin': frame_meta['target_range_bin'],
        'target_range_m': frame_meta['target_range_m'],
    }
    meta.update({
        k: v for k, v in frame_meta.items()
        if k not in {'range_bin_indices', 'range_axis_m', 'scatterer_summary'}
    })
    if return_debug:
        meta['range_frame'] = frame
        meta['range_axis_m'] = frame_meta['range_axis_m']
        meta['target_range_signal'] = signal.astype(np.complex64)
        meta['range_bin_indices'] = frame_meta['range_bin_indices']
        meta['scatterer_summary'] = frame_meta['scatterer_summary']
    return spec, ACTIVITY_TO_IDX[activity], meta


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
    #
    # Full 128x128 SVD dominated P02 data generation on shared CPUs.  The SVD
    # terms are only coarse morphology descriptors for the classical baseline,
    # so compute them on a 32x32 averaged image.  This keeps the feature family
    # from the earlier lecture material while making generation practical.
    svd_spec = spec
    max_svd_size = 32
    if H > max_svd_size or W > max_svd_size:
        if H % max_svd_size == 0 and W % max_svd_size == 0:
            fh = H // max_svd_size
            fw = W // max_svd_size
            svd_spec = spec.reshape(max_svd_size, fh, max_svd_size, fw).mean(axis=(1, 3))
        else:
            svd_spec = zoom(
                spec,
                (max_svd_size / H, max_svd_size / W),
                order=1,
            )
    _, S, _ = np.linalg.svd(svd_spec, full_matrices=False)
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
