"""클러터 모델 확장 — 교육용

fmcw_simulator.py의 generate_scene()을 확장하여
다양한 클러터 환경을 시뮬레이션.

사용법:
    from shared.clutter_model import generate_scene_with_clutter
"""

import numpy as np
from shared.fmcw_simulator import FMCWRadar, generate_scene, range_doppler_map, to_db

C = 299_792_458.0


def generate_scene_with_clutter(
    radar,
    targets,
    snr_db=15.0,
    clutter_type='mixed',
    clutter_power_db=-10.0,
    n_clutter_scatterers=30,
    multipath_prob=0.3,
    seed=42,
):
    """다양한 클러터가 포함된 FMCW beat signal 생성.

    Parameters
    ----------
    radar : FMCWRadar
    targets : list of dict
        실제 표적들 {'range', 'velocity', 'rcs'}
    snr_db : float
        기준 SNR
    clutter_type : str
        'zero_doppler' — 정지 클러터만
        'distributed' — 분산 클러터
        'multipath' — 다중경로 고스트
        'mixed' — 모든 유형 혼합
    clutter_power_db : float
        클러터 전력 (표적 대비 dB)
    n_clutter_scatterers : int
        분산 클러터 산란체 수
    multipath_prob : float
        다중경로 고스트 생성 확률
    seed : int

    Returns
    -------
    signal : ndarray (N_rx, N_chirps, N_samples)
    target_mask : ndarray (N_chirps, N_range_bins)
        GT binary mask (positive range only)
    target_info : list of dict
        각 표적의 bin 좌표 포함
    """
    rng = np.random.default_rng(seed)

    all_scatterers = list(targets)
    clutter_list = []

    clutter_power = 10 ** (clutter_power_db / 10.0)

    if clutter_type in ('zero_doppler', 'mixed'):
        # 정지 클러터: 도플러 ~0 근처, range 전역
        n_zd = n_clutter_scatterers // 2
        for _ in range(n_zd):
            r = rng.uniform(3.0, radar.max_range * 0.9)
            v = rng.normal(0.0, 0.3)  # 도플러 ~0 근처
            rcs = clutter_power * rng.exponential(1.0)
            clutter_list.append({'range': r, 'velocity': v, 'rcs': rcs})

    if clutter_type in ('distributed', 'mixed'):
        # 분산 클러터: range-dependent power (가까울수록 강함)
        n_dist = n_clutter_scatterers // 2
        for _ in range(n_dist):
            r = rng.uniform(3.0, radar.max_range * 0.9)
            v = rng.uniform(-radar.max_vel * 0.5, radar.max_vel * 0.5)
            # Range-dependent: closer = stronger
            range_factor = (1.0 - r / radar.max_range) ** 2
            rcs = clutter_power * range_factor * rng.exponential(0.5)
            clutter_list.append({'range': r, 'velocity': v, 'rcs': rcs})

    if clutter_type in ('multipath', 'mixed'):
        # 다중경로: 실제 표적의 2배 거리에 약한 고스트
        for tgt in targets:
            if rng.random() < multipath_prob:
                ghost_range = tgt['range'] * 2
                if ghost_range < radar.max_range * 0.95:
                    clutter_list.append({
                        'range': ghost_range,
                        'velocity': tgt.get('velocity', 0.0),
                        'rcs': tgt.get('rcs', 1.0) * 0.1,
                    })

    all_scatterers = targets + clutter_list

    # 신호 생성
    signal = generate_scene(radar, all_scatterers, snr_db=snr_db, seed=seed)

    # GT mask 생성 (표적만, 클러터 제외)
    Nc = radar.N_chirps
    Nr = radar.N_samples // 2  # positive range only
    target_mask = np.zeros((Nc, Nr), dtype=np.float32)
    target_info = []

    # Use exact FFT frequency axis (not linspace — 1-bin systematic offset bug)
    vel_axis = np.fft.fftshift(np.fft.fftfreq(Nc)) * radar.lam / (2 * radar.T_chirp)

    for tgt in targets:
        r_bin = round(tgt['range'] / radar.range_res)  # round, not int() floor
        v = tgt.get('velocity', 0.0)
        v_bin = np.argmin(np.abs(vel_axis - v))

        if 0 <= r_bin < Nr and 0 <= v_bin < Nc:
            # Binary cross mask (5 pixels: center + 4-connected)
            # Matches Hann-windowed PSF mainlobe — avoids F1 ceiling from
            # diagonal pixels where PSF amplitude is only 0.25
            for di, dj in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                vi = v_bin + di
                ri = r_bin + dj
                if 0 <= vi < Nc and 0 <= ri < Nr:
                    target_mask[vi, ri] = 1.0

            target_info.append({
                'range': tgt['range'],
                'velocity': v,
                'rcs': tgt.get('rcs', 1.0),
                'range_bin': r_bin,
                'doppler_bin': v_bin,
            })

    return signal, target_mask, target_info


def generate_random_scene(radar, rng, n_targets_range=(1, 15),
                          snr_range=(5.0, 25.0), clutter_power_range=(-15.0, -5.0)):
    """랜덤 시나리오 생성 (학습 데이터용).

    Returns
    -------
    rdm_input : ndarray (2, N_chirps, N_range_half) — noise-floor-ref log-mag + phase
    target_mask : ndarray (N_chirps, N_range_half)
    meta : dict
    """
    K = rng.integers(n_targets_range[0], n_targets_range[1] + 1)
    snr_db = rng.uniform(snr_range[0], snr_range[1])
    clutter_power_db = rng.uniform(clutter_power_range[0], clutter_power_range[1])

    targets = []
    for _ in range(K):
        r = rng.uniform(5.0, radar.max_range * 0.85)
        v = rng.uniform(-radar.max_vel * 0.8, radar.max_vel * 0.8)
        rcs = 10 ** rng.uniform(-2, 1)  # 0.01 ~ 10
        targets.append({'range': r, 'velocity': v, 'rcs': rcs})

    signal, target_mask, target_info = generate_scene_with_clutter(
        radar, targets,
        snr_db=snr_db,
        clutter_type='mixed',
        clutter_power_db=clutter_power_db,
        seed=int(rng.integers(0, 2**31)),
    )

    # RDM 생성
    rdm = range_doppler_map(signal[0:1], window_range='hann', window_doppler='hann')
    rdm_half = rdm[0, :, :radar.N_samples // 2]  # (Nc, Nr_half), complex

    # 2-channel: log-magnitude (noise-floor referenced) + phase
    mag = np.abs(rdm_half)
    # Noise-floor reference: median is robust to target peaks + clutter outliers
    noise_floor = np.median(mag)
    mag_db = 20 * np.log10(mag / (noise_floor + 1e-30) + 1e-30)
    # 0 dB = noise floor; targets are positive dB; deep noise is negative
    # Clip to [-20, 40] dB (60 dB range), noise floor maps to 1/3
    mag_db = np.clip(mag_db, -20, 40) / 60.0 + 1.0 / 3.0  # [0, 1], noise ≈ 0.33
    phase = np.angle(rdm_half) / np.pi  # normalize to [-1, 1]

    rdm_input = np.stack([mag_db, phase], axis=0).astype(np.float32)  # (2, Nc, Nr)

    meta = {
        'snr_db': float(snr_db),
        'n_targets': int(K),
        'clutter_power_db': float(clutter_power_db),
        'noise_floor': float(noise_floor),
    }

    return rdm_input, target_mask, meta
