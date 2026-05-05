"""클러터 모델 확장 — 교육용

fmcw_simulator.py의 generate_scene()을 확장하여
P1 탐지용 정적 클러터 환경을 시뮬레이션.

사용법:
    from shared.clutter_model import generate_scene_with_clutter
"""

import numpy as np
from shared.fmcw_simulator import (
    generate_scene,
    quantize_complex_iq,
    range_doppler_map,
)


def _local_rd_background_floor(mag, d_bin, r_bin, guard=(1, 1), train=(4, 4)):
    """Median RD background around a target, excluding the target mainlobe.

    A global map median can understate the background around the static-clutter
    Doppler ridge.  P1 labels should therefore be gated against a local
    CFAR-like training ring for the target bin.
    """
    nd, nr = mag.shape
    gd, gr = guard
    td, tr = train

    d0 = max(0, int(d_bin) - gd - td)
    d1 = min(nd, int(d_bin) + gd + td + 1)
    r0 = max(0, int(r_bin) - gr - tr)
    r1 = min(nr, int(r_bin) + gr + tr + 1)
    patch = mag[d0:d1, r0:r1]

    dd, rr = np.ogrid[d0:d1, r0:r1]
    guard_mask = (
        (np.abs(dd - int(d_bin)) <= gd)
        & (np.abs(rr - int(r_bin)) <= gr)
    )
    train_cells = patch[~guard_mask]
    if train_cells.size == 0:
        return float(np.median(mag))
    return float(np.median(train_cells))


def apply_slow_time_mti(signal, mode="mean_removal"):
    """Apply a simple slow-time MTI/DC-notch while preserving tensor shape."""
    if mode in (None, False, "none"):
        return signal
    if mode != "mean_removal":
        raise ValueError("Supported P1 MTI modes: 'mean_removal' or 'none'")
    return signal - np.mean(signal, axis=-2, keepdims=True)


def generate_scene_with_clutter(
    radar,
    targets,
    snr_db=15.0,
    clutter_type='static',
    clutter_power_db=-16.0,
    n_clutter_scatterers=30,
    clutter_rcs_max=0.25,
    min_label_snr_db=None,
    adc_bits=None,
    adc_full_scale=None,
    mti_mode='mean_removal',
    seed=42,
    return_meta=False,
):
    """정적 클러터가 포함된 FMCW beat signal 생성.

    Parameters
    ----------
    radar : FMCWRadar
    targets : list of dict
        실제 표적들 {'range', 'velocity', 'rcs'}
    snr_db : float
        기준 SNR
    clutter_type : str
        'static' or 'zero_doppler' only.  P1 intentionally excludes multipath
        ghosts and Doppler-tail clutter after the simulator cleanup.
    clutter_power_db : float
        클러터 전력 (표적 대비 dB)
    n_clutter_scatterers : int
        분산 클러터 산란체 수
    clutter_rcs_max : float
        Hard cap for static clutter RCS.  The lognormal draw is clipped so a
        rare clutter outlier cannot determine ADC full-scale or clipping.
    min_label_snr_db : float or None
        If set, targets below this realised post-processing SNR remain in the
        raw signal but are excluded from detection labels. This avoids forcing
        impossible below-noise positives in detection datasets.
    seed : int

    Returns
    -------
    signal : ndarray (N_rx, N_chirps, N_samples)
    target_mask : ndarray (N_chirps, N_range_bins)
        GT binary mask (positive range only)
    target_info : list of dict
        각 표적의 bin 좌표 포함
    """
    if clutter_type != 'static':
        raise ValueError("P1 clutter_type is static-only; use 'static'")

    rng = np.random.default_rng(seed)

    all_scatterers = list(targets)
    clutter_list = []

    clutter_power = 10 ** (clutter_power_db / 10.0)

    def clutter_rcs(range_m: float, scale: float = 1.0) -> float:
        # Approximate unresolved ground/background resolution-cell footprint.
        footprint = max(range_m / 100.0, 0.25) ** 1.2
        raw = clutter_power * scale * footprint * rng.lognormal(mean=-0.3, sigma=0.8)
        return float(np.clip(raw, 0.0, clutter_rcs_max))

    # Static ground/background clutter only: exactly zero Doppler.  No Doppler
    # tail and no target-linked multipath ghosts are generated.
    for _ in range(n_clutter_scatterers):
        r = rng.uniform(3.0, radar.max_range * 0.9)
        rcs = clutter_rcs(r, scale=1.0)
        clutter_list.append({'range': r, 'velocity': 0.0, 'rcs': rcs, 'is_clutter': True})

    all_scatterers = targets + clutter_list

    # 신호 생성
    signal, scene_meta = generate_scene(
        radar, all_scatterers, snr_db=snr_db, seed=seed, return_meta=True
    )
    adc_meta = {
        'enabled': False,
        'bits': None,
        'full_scale': None,
        'clipped_fraction': 0.0,
    }
    if adc_bits is not None:
        signal, adc_meta = quantize_complex_iq(
            signal,
            bits=int(adc_bits),
            full_scale=adc_full_scale,
            return_meta=True,
        )
        adc_meta = dict(adc_meta)
        adc_meta['enabled'] = True

    signal = apply_slow_time_mti(signal, mode=mti_mode)

    label_mag = None
    label_noise_floor = None
    if min_label_snr_db is not None:
        # Gate labels on the processed RD map that P1 actually sees, not on
        # scene maximum or only the thermal-SNR model.  This keeps effectively
        # buried targets out of positive masks.
        label_rdm = range_doppler_map(signal[0:1], radar=radar, window_range='hann', window_doppler='hann')
        label_mag = np.abs(label_rdm[0, :, :radar.N_range_bins])
        label_noise_floor = float(np.median(label_mag))

    # GT mask 생성 (표적만, 클러터 제외)
    Nc = radar.N_chirps
    Nr = radar.N_range_bins
    target_mask = np.zeros((Nc, Nr), dtype=np.float32)
    target_info = []

    true_target_meta = [info for info in scene_meta['target_info'] if not info.get('is_clutter', False)]

    for tgt, sim_info in zip(targets, true_target_meta):
        r_bin = sim_info['range_bin']  # round, not int() floor
        v = tgt.get('velocity', 0.0)
        v_bin = sim_info['doppler_bin']

        if 0 <= r_bin < Nr and 0 <= v_bin < Nc:
            peak_snr_db = sim_info.get('actual_snr_db', snr_db)
            global_peak_snr_db = sim_info.get('actual_snr_db', snr_db)
            local_background_floor = np.nan
            effective_background_floor = np.nan
            if label_mag is not None and label_noise_floor is not None:
                peak = float(label_mag[v_bin, r_bin])
                global_peak_snr_db = float(
                    20.0 * np.log10((peak + 1e-30) / (label_noise_floor + 1e-30))
                )
                local_background_floor = _local_rd_background_floor(label_mag, v_bin, r_bin)
                effective_background_floor = max(label_noise_floor, local_background_floor)
                peak_snr_db = float(
                    20.0 * np.log10((peak + 1e-30) / (effective_background_floor + 1e-30))
                )
            if min_label_snr_db is not None and peak_snr_db < min_label_snr_db:
                continue

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
                'actual_snr_db': sim_info.get('actual_snr_db', snr_db),
                'peak_snr_db': peak_snr_db,
                'global_peak_snr_db': global_peak_snr_db,
                'local_background_floor': local_background_floor,
                'effective_background_floor': effective_background_floor,
            })

    meta = {
        'clutter_type': 'static',
        'clutter_info': clutter_list,
        'n_clutter_scatterers': int(len(clutter_list)),
        'clutter_power_db': float(clutter_power_db),
        'clutter_rcs_max': float(clutter_rcs_max),
        'adc_quantization': adc_meta,
        'mti_applied': mti_mode not in (None, False, "none"),
        'mti_mode': "slow_time_mean_removal_dc_notch" if mti_mode == "mean_removal" else "none",
    }

    if return_meta:
        return signal, target_mask, target_info, meta
    return signal, target_mask, target_info


def generate_random_scene(radar, rng, n_targets_range=(1, 15),
                          snr_range=(5.0, 25.0), clutter_power_range=(-22.0, -12.0),
                          target_range_min_m=15.0, target_rcs_log10_range=(-2.0, 0.3),
                          target_min_abs_velocity_mps=None,
                          clutter_rcs_max=0.25, min_label_snr_db=0.0,
                          adc_bits=16, adc_full_scale=6.0e-5,
                          mti_mode='mean_removal',
                          require_no_adc_clipping=True, return_raw=False):
    """랜덤 시나리오 생성 (학습 데이터용).

    Returns
    -------
    rdm_input : ndarray (2, N_chirps, N_range_half) — noise-floor-ref log-mag + phase
    target_mask : ndarray (N_chirps, N_range_half)
    meta : dict
        If return_raw=True, includes rdm_mag_linear, rdm_power, and target_info
        for baseline-grade evaluation.
    """
    # Generate at least one labelled positive in the common case while still
    # allowing rare hard-negative scenes.  Labels below the configured processed
    # target-bin peak/noise-floor SNR are excluded because they are effectively
    # below the map floor and make the detection target ill-posed.
    if target_min_abs_velocity_mps is None:
        target_min_abs_velocity_mps = 2.0 * radar.vel_res
    max_attempts = 32
    for attempt in range(max_attempts):
        K = rng.integers(n_targets_range[0], n_targets_range[1] + 1)
        snr_db = rng.uniform(snr_range[0], snr_range[1])
        clutter_power_db = rng.uniform(clutter_power_range[0], clutter_power_range[1])

        targets = []
        for _ in range(K):
            r = rng.uniform(target_range_min_m, radar.max_range * 0.85)
            min_abs_v = max(0.0, float(target_min_abs_velocity_mps))
            max_abs_v = radar.max_vel * 0.8
            if min_abs_v > 0.0 and min_abs_v < max_abs_v:
                sign = rng.choice([-1.0, 1.0])
                v = sign * rng.uniform(min_abs_v, max_abs_v)
            else:
                v = rng.uniform(-max_abs_v, max_abs_v)
            rcs = 10 ** rng.uniform(*target_rcs_log10_range)
            targets.append({'range': r, 'velocity': v, 'rcs': rcs})

        signal, target_mask, target_info, scene_meta = generate_scene_with_clutter(
            radar, targets,
            snr_db=snr_db,
            clutter_type='static',
            clutter_power_db=clutter_power_db,
            clutter_rcs_max=clutter_rcs_max,
            min_label_snr_db=min_label_snr_db,
            adc_bits=adc_bits,
            adc_full_scale=adc_full_scale,
            mti_mode=mti_mode,
            seed=int(rng.integers(0, 2**31)),
            return_meta=True,
        )
        clipped = scene_meta['adc_quantization']['clipped_fraction'] > 0.0
        if target_info and (not require_no_adc_clipping or not clipped):
            break
    else:
        fallback_v = max(float(target_min_abs_velocity_mps), 2.5 * radar.vel_res)
        fallback_targets = [{
            'range': max(float(target_range_min_m) + 10.0, 0.35 * radar.max_range),
            'velocity': min(fallback_v, 0.5 * radar.max_vel),
            'rcs': min(1.0, 10 ** float(target_rcs_log10_range[1])),
        }]
        for _ in range(8):
            snr_db = max(float(snr_range[1]), 20.0)
            clutter_power_db = float(clutter_power_range[0])
            signal, target_mask, target_info, scene_meta = generate_scene_with_clutter(
                radar, fallback_targets,
                snr_db=snr_db,
                clutter_type='static',
                clutter_power_db=clutter_power_db,
                clutter_rcs_max=clutter_rcs_max,
                min_label_snr_db=min_label_snr_db,
                adc_bits=adc_bits,
                adc_full_scale=adc_full_scale,
                mti_mode=mti_mode,
                seed=int(rng.integers(0, 2**31)),
                return_meta=True,
            )
            clipped = scene_meta['adc_quantization']['clipped_fraction'] > 0.0
            if target_info and (not require_no_adc_clipping or not clipped):
                break
        else:
            raise RuntimeError(
                "P1 generator could not produce a labelled non-clipped scene under "
                "the current teaching constraints; adjust ADC full-scale or scene ranges."
            )

    # RDM 생성
    rdm = range_doppler_map(signal[0:1], radar=radar, window_range='hann', window_doppler='hann')
    rdm_half = rdm[0, :, :radar.N_range_bins]  # (Nc, Nr), complex

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
        'n_targets': int(len(target_info)),
        'clutter_power_db': float(clutter_power_db),
        'noise_floor': float(noise_floor),
        'min_label_snr_db': float(min_label_snr_db),
        'clutter_type': 'static',
        'adc_bits': int(adc_bits) if adc_bits is not None else 0,
        'adc_full_scale': float(adc_full_scale) if adc_full_scale is not None else np.nan,
        'adc_clipped_fraction': float(scene_meta['adc_quantization']['clipped_fraction']),
        'mti_applied': bool(scene_meta['mti_applied']),
        'mti_mode': str(scene_meta['mti_mode']),
        'target_range_min_m': float(target_range_min_m),
        'target_min_abs_velocity_mps': float(target_min_abs_velocity_mps),
        'target_rcs_log10_min': float(target_rcs_log10_range[0]),
        'target_rcs_log10_max': float(target_rcs_log10_range[1]),
        'clutter_rcs_max': float(clutter_rcs_max),
    }
    if return_raw:
        meta.update({
            'rdm_mag_linear': mag.astype(np.float32),
            'rdm_power': (mag ** 2).astype(np.float32),
            'target_info': target_info,
            'clutter_info': scene_meta['clutter_info'],
        })

    return rdm_input, target_mask, meta
