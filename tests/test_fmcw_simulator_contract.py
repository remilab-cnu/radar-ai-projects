import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.fmcw_simulator import (
    FMCWRadar,
    generate_scene,
    range_doppler_map,
    range_fft,
    target_rd_bins,
)


def _radar(**kwargs):
    params = dict(
        fc=10.0e9,
        bw=10.0e6,
        T_chirp=12.8e-6,
        PRI=100e-6,
        N_chirps=64,
        N_samples=512,
        temperature_k=1e-9,
        phase_noise_std_rad=0.0,
    )
    params.update(kwargs)
    return FMCWRadar(**params)


def test_fmcw_sampling_contract_defaults_to_four_times_bandwidth():
    radar = FMCWRadar(bw=12.5e6, T_chirp=4e-6, PRI=80e-6, N_samples=256)
    assert radar.fs == pytest.approx(4.0 * radar.bw)
    assert radar.fs / radar.bw == pytest.approx(4.0)


def test_fmcw_sampling_contract_rejects_active_mismatch():
    with pytest.raises(ValueError, match=r"fs = 4 \* bw"):
        FMCWRadar(bw=10e6, fs=20e6, T_chirp=4e-6, N_samples=128)


def test_fmcw_range_and_doppler_peaks_match_metadata_bins():
    radar = _radar()
    target = {"range": 600.0, "velocity": 3.0, "rcs": 1e8, "phase": 0.0}
    raw, meta = generate_scene(radar, [target], snr_db=None, seed=123, return_meta=True)
    rd = range_doppler_map(raw, radar=radar, window_range="rect", window_doppler="rect")[0]
    peak_d, peak_r = np.unravel_index(np.argmax(np.abs(rd)), rd.shape)
    expected_r, expected_d = target_rd_bins(radar, target["range"], target["velocity"])

    assert meta["simulator"] == "fmcw_baseband_dechirp_mixing"
    assert meta["fs_over_bandwidth"] == pytest.approx(4.0)
    assert meta["up_down_conversion"] == "excluded_baseband_only"
    assert abs(peak_r - expected_r) <= 1
    assert abs(peak_d - expected_d) <= 1


def test_fmcw_multi_rx_phase_slope_matches_steering_angle():
    radar = _radar(N_rx=4)
    angle_deg = 25.0
    target = {"range": 450.0, "velocity": 0.0, "angle": angle_deg, "rcs": 1e8, "phase": 0.0}
    raw = generate_scene(radar, [target], snr_db=None, seed=321, return_meta=False)
    rng = range_fft(raw, radar=radar, window="rect")
    r_bin, _ = target_rd_bins(radar, target["range"], 0.0)
    vec = rng[:, 0, r_bin]
    measured = np.angle(vec[1:] * np.conj(vec[:-1]))
    expected = 2.0 * np.pi * radar.d_rx * np.sin(np.deg2rad(angle_deg)) / radar.lam
    expected = np.angle(np.exp(1j * expected))
    assert np.allclose(measured, expected, atol=0.08)
