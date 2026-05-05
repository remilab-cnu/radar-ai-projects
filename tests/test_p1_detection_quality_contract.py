import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from projects.p01_unet_detector.generate_data import (
    ADC_IQ_BITS,
    ADC_IQ_FULL_SCALE,
    RADAR as P01_RADAR,
    SCHEMA_VERSION,
    TARGET_MIN_ABS_VELOCITY_MPS,
    X_STORAGE_DTYPE,
    Y_STORAGE_DTYPE,
    _generate_split,
)
from projects.p01_unet_detector.eval_utils import target_detection_counts
from shared.clutter_model import generate_random_scene, generate_scene_with_clutter
from shared.fmcw_simulator import encode_complex_iq_signed, quantize_complex_iq, range_doppler_map


def test_p1_default_labels_exclude_sub_zero_db_targets():
    rng = np.random.default_rng(20260430)
    for _ in range(40):
        _, _, meta = generate_random_scene(P01_RADAR, rng, return_raw=True)
        assert meta["adc_clipped_fraction"] == pytest.approx(0.0)
        for info in meta["target_info"]:
            assert info["peak_snr_db"] >= 0.0
            assert abs(info["velocity"]) >= TARGET_MIN_ABS_VELOCITY_MPS
            assert np.isfinite(info["global_peak_snr_db"])
            assert np.isfinite(info["local_background_floor"])
            assert np.isfinite(info["effective_background_floor"])
            assert info["global_peak_snr_db"] >= 0.0


def test_p1_default_generator_keeps_positive_labels_common():
    rng = np.random.default_rng(777)
    n_labels = []
    for _ in range(80):
        _, _, meta = generate_random_scene(P01_RADAR, rng, return_raw=True)
        n_labels.append(meta["n_targets"])
    # Some hard-negative scenes are acceptable, but the default generator should
    # not mostly produce impossible/no-positive labels after the SNR gate.
    assert np.mean(np.asarray(n_labels) > 0) >= 0.90


def test_p1_clutter_is_static_only_without_multipath_or_doppler_tail():
    targets = [{"range": 80.0, "velocity": 12.0, "rcs": 0.5}]
    _, _, _, meta = generate_scene_with_clutter(
        P01_RADAR,
        targets,
        clutter_type="static",
        min_label_snr_db=0.0,
        adc_bits=16,
        adc_full_scale=ADC_IQ_FULL_SCALE,
        return_meta=True,
        seed=123,
    )
    clutter = meta["clutter_info"]
    assert clutter
    assert all(c["is_clutter"] for c in clutter)
    assert all(c["velocity"] == pytest.approx(0.0) for c in clutter)
    # Static-only means no target-linked multipath ghosts at 2x target range.
    assert not any(c["range"] == pytest.approx(160.0) and c["velocity"] == pytest.approx(12.0) for c in clutter)
    assert meta["mti_applied"]
    assert meta["mti_mode"] == "slow_time_mean_removal_dc_notch"


def test_complex_16bit_iq_is_two_int16_components_not_scalar_int16():
    raw = np.array([1.0 + 0.5j, -0.25 - 0.75j], dtype=np.complex64)
    codes, meta = encode_complex_iq_signed(raw, bits=16, full_scale=1.0, return_meta=True)
    assert codes.dtype == np.int16
    assert codes.shape == (2, 2)  # last axis is [I, Q]
    assert meta["component_axis"] == "last_dim_iq"
    assert meta["component_dtype"] == "int16"


def test_p1_hdf5_uses_compact_storage_and_clipping_free_fixed_adc(tmp_path):
    out = tmp_path / "det_smoke.h5"
    _generate_split(out, n_samples=8, seed=9191)
    with h5py.File(out, "r") as f:
        assert f["x"].dtype == np.dtype(X_STORAGE_DTYPE)
        assert f["y"].dtype == np.dtype(Y_STORAGE_DTYPE)
        assert f["schema_version"][0] == SCHEMA_VERSION
        assert f["adc_iq_bits"][0] == ADC_IQ_BITS
        assert f["adc_iq_component_dtype"][0] == b"int16"
        assert f["clutter_type"][0] == b"static"
        assert f["mti_mode"][0] == b"slow_time_mean_removal_dc_notch"
        assert np.all(f["mti_applied"][:])
        assert f["target_min_abs_velocity_mps"][0] == pytest.approx(TARGET_MIN_ABS_VELOCITY_MPS)
        assert np.max(f["adc_clipped_fraction"][:]) == pytest.approx(0.0)
        valid = f["target_range_bin"][:] >= 0
        assert np.all(f["target_peak_snr_db"][:][valid] >= 0.0)
        assert np.all(np.abs(f["target_velocity_mps"][:][valid]) >= TARGET_MIN_ABS_VELOCITY_MPS)
        assert np.all(np.isfinite(f["target_global_peak_snr_db"][:][valid]))
        assert np.all(np.isfinite(f["target_local_bg_floor"][:][valid]))
        assert np.all(np.isfinite(f["target_effective_bg_floor"][:][valid]))
        assert np.all(f["target_global_peak_snr_db"][:][valid] >= 0.0)


def test_p1_target_detection_metric_counts_targets_with_bin_tolerance():
    pred = np.zeros((64, 200), dtype=bool)
    pred[11, 21] = True
    pred[30, 100] = True
    counts = target_detection_counts(
        pred,
        target_range_bins=np.array([20, 100, -1]),
        target_doppler_bins=np.array([10, 33, -1]),
        tolerance=(1, 1),
    )
    assert counts == {"target_detected": 1, "target_total": 2}


def test_16bit_complex_iq_quantization_preserves_p1_rdm_peaks_for_teaching_scene():
    targets = [
        {"range": 30.0, "velocity": 3.0, "rcs": 1.0, "phase": 0.1},
        {"range": 150.0, "velocity": 15.0, "rcs": 0.25, "phase": 1.0},
    ]
    from shared.fmcw_simulator import generate_scene

    raw, meta = generate_scene(P01_RADAR, targets, snr_db=25.0, seed=123, return_meta=True)
    quantized, qmeta = quantize_complex_iq(raw, bits=16, return_meta=True)
    rd = np.abs(range_doppler_map(raw[0:1], radar=P01_RADAR, window_range="hann", window_doppler="hann")[0])
    rd_q = np.abs(range_doppler_map(quantized[0:1], radar=P01_RADAR, window_range="hann", window_doppler="hann")[0])

    assert qmeta["bits"] == 16
    assert qmeta["clipped_fraction"] == pytest.approx(0.0)
    for info in meta["target_info"]:
        r = int(info["range_bin"])
        d = int(info["doppler_bin"])
        before = rd[max(0, d - 1): d + 2, max(0, r - 1): r + 2].max()
        after = rd_q[max(0, d - 1): d + 2, max(0, r - 1): r + 2].max()
        delta_db = 20.0 * np.log10((after + 1e-30) / (before + 1e-30))
        assert abs(delta_db) < 0.25
