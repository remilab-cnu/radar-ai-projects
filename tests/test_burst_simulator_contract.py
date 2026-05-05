import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.burst_simulator import BurstRadar, CleanBurstSimulator, range_axis, velocity_axis


def _radar(**kwargs):
    params = dict(
        fc=9.6e9,
        bandwidth=20e6,
        pulse_width=4e-6,
        prf=10e3,
        pulses_per_cpi=64,
        max_range=2_000.0,
        temperature_k=1e-9,
    )
    params.update(kwargs)
    return BurstRadar(**params)


def test_clean_burst_rejects_interference_configuration_by_api():
    sim = CleanBurstSimulator(_radar())
    with pytest.raises(TypeError):
        sim.simulate_burst([], interference_config={})  # type: ignore[call-arg]


def test_clean_burst_is_deterministic_with_fixed_seed():
    sim = CleanBurstSimulator(_radar())
    targets = [{"range": 750.0, "velocity": 0.0, "rcs": 1.0}]
    a, ref_a, meta_a = sim.simulate_burst(targets, seed=7, return_meta=True)
    b, ref_b, meta_b = sim.simulate_burst(targets, seed=7, return_meta=True)
    assert np.array_equal(a, b)
    assert np.array_equal(ref_a, ref_b)
    assert meta_a["simulator"] == "clean_lfm_burst_no_interference"
    assert meta_b["fs_over_bandwidth"] == pytest.approx(4.0)


def test_clean_burst_range_and_doppler_peak_match_expected_bins():
    radar = _radar(temperature_k=1e-9)
    sim = CleanBurstSimulator(radar)
    target = {"range": 750.0, "velocity": 5.0, "rcs": 1e8, "phase": 0.0}
    adc, ref, meta = sim.simulate_burst([target], seed=11, add_noise=False, return_meta=True)
    _, rd = sim.process_burst(adc, ref)
    peak_r, peak_d = np.unravel_index(np.argmax(np.abs(rd)), rd.shape)
    info = meta["target_info"][0]
    assert abs(peak_r - info["range_bin"]) <= 1
    assert abs(peak_d - info["doppler_bin"]) <= 1
    assert range_axis(radar)[peak_r] == pytest.approx(target["range"], abs=radar.range_bin_spacing)
    assert velocity_axis(radar)[peak_d] == pytest.approx(target["velocity"], abs=radar.velocity_resolution)


def test_clean_burst_noise_power_uses_ktb_noise_figure():
    radar = _radar(temperature_k=290.0, noise_figure_db=6.0)
    expected = 1.380_649e-23 * 290.0 * radar.bandwidth * (10.0 ** (6.0 / 10.0))
    assert radar.noise_power_w == pytest.approx(expected)
