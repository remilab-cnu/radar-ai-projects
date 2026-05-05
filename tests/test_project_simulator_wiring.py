import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from projects.p01_unet_detector.generate_data import RADAR as P01_RADAR
from projects.p02_resnet18_har.generate_data import RADAR as P02_RADAR
from projects.p03_radar_cube_doa.generate_data import P03_RADAR, generate_one_sample
from shared.micro_doppler import generate_har_sample


def test_p01_uses_explicit_shared_fmcw_dechirp_config():
    assert P01_RADAR.fs / P01_RADAR.bw == pytest.approx(4.0)
    assert P01_RADAR.T_chirp == pytest.approx(2e-6)
    assert P01_RADAR.PRI == pytest.approx(100e-6)
    assert P01_RADAR.N_range_bins == P01_RADAR.N_samples // 2


def test_p02_har_sample_uses_shared_physics_radar_metadata():
    spec, label, meta = generate_har_sample(
        "walk",
        np.random.default_rng(123),
        duration=0.25,
        output_size=(32, 32),
        radar=P02_RADAR,
    )
    assert spec.shape == (32, 32)
    assert label >= 0
    assert meta["simulator"] == "range_compressed_target_range_micro_doppler"
    assert meta["radar_fc_hz"] == pytest.approx(77.0e9)
    assert meta["fs_over_bandwidth"] == pytest.approx(4.0)
    assert meta["slow_time_prf_hz"] == pytest.approx(1.0 / P02_RADAR.PRI)
    assert meta["slow_time_samples"] == 2500
    assert meta["radar_config_n_chirps"] == P02_RADAR.N_chirps
    assert meta["up_down_conversion"] == "excluded_baseband_only"
    assert meta["range_processing"] == "local_range_compressed_frame"
    assert meta["doppler_source"] == "stft_of_target_range_signal"
    assert meta["scatter_model"] == "radar_deconv_inspired_pedestrian_scatterers"
    assert meta["scatter_model_scope"] == "p02_only"
    assert meta["n_scatterers"] >= 18
    assert meta["scatter_kind_counts"]["limb"] >= 16
    assert 0 <= meta["target_range_bin"] < P02_RADAR.N_range_bins


def test_p03_generator_uses_shared_fmcw_and_outputs_selected_antenna_vector():
    sample = generate_one_sample(np.random.default_rng(321))
    assert P03_RADAR.fs / P03_RADAR.bw == pytest.approx(4.0)
    assert sample["x_ant"].shape == (2, P03_RADAR.N_rx)
    assert sample["y_spectrum"].shape == (181,)
    assert 0 <= int(sample["r_bin"]) < P03_RADAR.N_range_bins
    assert 0 <= int(sample["d_bin"]) < P03_RADAR.N_chirps
    assert sample["n_targets"] == np.int32(1)
    assert sample["fs_over_bandwidth"] == pytest.approx(4.0)
    ant = sample["x_ant"][0] + 1j * sample["x_ant"][1]
    assert np.mean(np.abs(ant) ** 2) == pytest.approx(1.0, abs=1e-5)
