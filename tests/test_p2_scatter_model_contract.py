import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from projects.p02_resnet18_har.generate_data import (
    ASPECT_ANGLE_RANGE_DEG,
    ASPECT_CONVENTION,
    RADAR as P02_RADAR,
    SCATTER_MODEL,
    SCHEMA_VERSION,
    _generate_split,
)
from shared.micro_doppler import generate_har_sample
from shared.micro_doppler import (
    _random_activity_params,
    generate_range_compressed_micro_doppler_frame,
    radar_max_unambiguous_velocity_mps,
)


def test_p2_target_range_micro_doppler_uses_pedestrian_scatter_model():
    spec, label, meta = generate_har_sample(
        "walk",
        np.random.default_rng(20260430),
        duration=0.25,
        output_size=(32, 32),
        radar=P02_RADAR,
        return_debug=True,
    )

    assert spec.shape == (32, 32)
    assert label == 0
    assert meta["simulator"] == "range_compressed_target_range_micro_doppler"
    assert meta["scatter_model"] == "radar_deconv_inspired_pedestrian_scatterers"
    assert meta["scatter_model_scope"] == "p02_only"
    assert meta["slow_time_prf_hz"] == 1.0 / P02_RADAR.PRI
    assert meta["slow_time_samples"] == 2500
    assert meta["max_abs_radial_velocity_mps"] < meta["radar_max_unambiguous_velocity_mps"]
    assert meta["doppler_alias_margin_mps"] > 0.0
    assert meta["radar_config_n_chirps"] == P02_RADAR.N_chirps
    assert meta["n_scatterers"] >= 18
    assert meta["scatter_kind_counts"]["torso"] >= 2
    assert meta["scatter_kind_counts"]["head"] == 1
    assert meta["scatter_kind_counts"]["limb"] >= 16

    scatterers = meta["scatterer_summary"]
    assert len(scatterers) == meta["n_scatterers"]
    assert any(sc["scatter_kind"] == "limb" and sc["micro_disp_amp_m"] > 0 for sc in scatterers)
    assert any(sc["max_abs_radial_velocity_mps"] > 0.1 for sc in scatterers)

    frame = meta["range_frame"]
    target_range_signal = meta["target_range_signal"]
    assert frame.shape[0] == target_range_signal.shape[0]
    assert frame.shape[1] == meta["range_window_bins"]
    assert np.iscomplexobj(frame)
    assert np.iscomplexobj(target_range_signal)


def test_p2_hdf5_records_scatter_model_metadata(tmp_path):
    out = tmp_path / "har_smoke.h5"
    _generate_split(out, n_samples=6, seed=20260501)

    with h5py.File(out, "r") as f:
        assert f["schema_version"][0] == SCHEMA_VERSION
        assert f["scatter_model"][0] == SCATTER_MODEL.encode("utf-8")
        assert f["scatter_model_scope"][0] == b"p02_only"
        assert f["aspect_convention"][0] == ASPECT_CONVENTION.encode("utf-8")
        assert tuple(f["aspect_angle_range_deg"][:]) == ASPECT_ANGLE_RANGE_DEG
        assert np.all(f["aspect_angle_deg"][:] >= ASPECT_ANGLE_RANGE_DEG[0])
        assert np.all(f["aspect_angle_deg"][:] <= ASPECT_ANGLE_RANGE_DEG[1])
        assert np.all(f["slow_time_prf_hz"][:] == 1.0 / P02_RADAR.PRI)
        assert np.all(f["slow_time_samples"][:] == 30000)
        assert np.all(f["max_abs_radial_velocity_mps"][:] < f["radar_max_unambiguous_velocity_mps"][:])
        assert np.all(f["doppler_alias_margin_mps"][:] > 0.0)
        assert f["radar_config_n_chirps"][0] == P02_RADAR.N_chirps
        assert f["range_processing"][0] == b"local_range_compressed_frame"
        assert f["doppler_source"][0] == b"stft_of_target_range_signal"
        assert np.all(f["n_scatterers"][:] >= 18)
        assert np.all(f["limb_scatterers"][:] >= 16)
        assert np.all(f["head_scatterers"][:] == 1)


def test_p2_aspect_projection_reduces_radial_micro_doppler():
    params = _random_activity_params("walk", np.random.default_rng(77))
    max_velocities = []
    for aspect in [0.0, 60.0, 90.0]:
        _, meta = generate_range_compressed_micro_doppler_frame(
            "walk",
            params,
            duration=0.5,
            snr_db=80.0,
            rng=np.random.default_rng(1234),
            aspect_angle=aspect,
            radar=P02_RADAR,
        )
        max_velocities.append(
            max(sc["max_abs_radial_velocity_mps"] for sc in meta["scatterer_summary"])
        )

    assert max_velocities[1] < 0.65 * max_velocities[0]
    assert max_velocities[2] < 0.05 * max_velocities[0]


def test_p2_default_run_motion_stays_inside_doppler_nyquist():
    rng = np.random.default_rng(20260430)
    max_unambiguous = radar_max_unambiguous_velocity_mps(P02_RADAR)
    safety_limit = 0.9 * max_unambiguous
    observed = []
    for _ in range(8):
        params = _random_activity_params("run", rng)
        _, meta = generate_range_compressed_micro_doppler_frame(
            "run",
            params,
            duration=0.5,
            snr_db=80.0,
            rng=rng,
            aspect_angle=float(rng.uniform(*ASPECT_ANGLE_RANGE_DEG)),
            radar=P02_RADAR,
        )
        observed.append(meta["max_abs_radial_velocity_mps"])

    assert max(observed) < safety_limit
