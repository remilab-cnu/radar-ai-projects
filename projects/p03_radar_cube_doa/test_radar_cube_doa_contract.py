"""Contract tests for P03 radar-cube DoA data and baselines."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR.parents[1]))
sys.path.insert(0, str(PROJECT_DIR))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gen = _load_module("p03_generate_data", PROJECT_DIR / "generate_data.py")
mapping = _load_module("p03_mapping", PROJECT_DIR / "mapping.py")
mapping_gen = _load_module("p03_generate_mapping_data", PROJECT_DIR / "generate_mapping_data.py")
train = _load_module("p03_train", PROJECT_DIR / "train.py")


def test_range_doppler_fft_skips_angle_fft_and_selects_antenna_vector() -> None:
    rng = np.random.default_rng(123)
    sample = gen.generate_one_sample(rng)
    assert sample["x_ant"].shape == (2, gen.N_RX)
    assert sample["y_spectrum"].shape == (len(gen.ANGLE_GRID),)
    assert 0 <= int(sample["r_bin"]) < gen.N_FAST
    assert 0 <= int(sample["d_bin"]) < gen.N_CHIRPS
    assert -90.0 <= float(sample["angle_deg"]) <= 90.0
    assert float(sample["snr_db"]) > 0.0
    assert int(sample["n_targets"]) == 1
    assert "scene_mode" not in sample
    assert "target_rcs_linear" not in sample
    assert "collision_angle_deg" not in sample


def test_music_baseline_peaks_near_clean_steering_angle() -> None:
    angle = 23.0
    x = gen.steering_vector(angle).astype(np.complex64)
    A = train.steering_matrix(train.ANGLE_GRID, n_rx=len(x))

    music_angle = train.estimate_angle_from_spectrum(train.music_spectrum_single_snapshot(x, A))

    assert abs(music_angle - angle) <= 1.0


def test_ego_motion_projection_uses_simulator_closing_velocity() -> None:
    pose = mapping.EgoPose(x_m=0.0, y_m=0.0, heading_deg=0.0, speed_mps=10.0)
    theta = 30.0
    target = mapping.WorldTarget(
        x_m=20.0 * np.sin(np.deg2rad(theta)),
        y_m=20.0 * np.cos(np.deg2rad(theta)),
        target_id=7,
    )

    meas = mapping.measurement_from_world(pose, target)

    assert meas is not None
    assert np.isclose(meas.range_m, 20.0)
    assert np.isclose(meas.angle_deg, theta)
    # Simulator convention: positive velocity means closing range.
    assert np.isclose(meas.radial_velocity_mps, 10.0 * np.cos(np.deg2rad(theta)))


def test_p03_main_scene_uses_urban_intersection_without_resolution_probes() -> None:
    targets = mapping.build_p03_mapping_targets(
        seed=0,
        wall_spacing_m=0.35,
        include_dynamic=False,
        include_resolution_probes=False,
    )

    assert not any(t.target_type.startswith("resolution_pair") for t in targets)
    assert sum(t.target_type.endswith("_facade") for t in targets) >= 300
    assert sum(t.target_type.startswith("parked_vehicle") for t in targets) >= 40
    assert len(mapping.build_p03_urban_occluders()) == 8

    spec = mapping.P03_URBAN_GRID
    grid = mapping.occupancy_grid_from_targets(
        targets,
        grid_spec=spec,
        include_dynamic=False,
        sigma_cells=1.0,
    )
    rows = [mapping.world_to_grid_spec(-6.0, y, spec)[0] for y in np.linspace(-20.0, 18.0, 80)]
    col = mapping.world_to_grid_spec(-6.0, 0.0, spec)[1]
    lo, hi = min(rows), max(rows)
    wall_trace = grid[lo:hi + 1, max(0, col - 1):col + 2].max(axis=1) > 0

    # The road-facing building facade should read as a connected surface chain.
    assert float(np.mean(wall_trace)) > 0.85


def test_visible_measurements_blocks_targets_behind_opaque_building() -> None:
    pose = mapping.EgoPose(x_m=0.0, y_m=0.0, heading_deg=0.0, speed_mps=10.0)
    visible_front = mapping.WorldTarget(x_m=-6.0, y_m=8.0, target_id=1, target_type="sw_east_facade")
    blocked_far = mapping.WorldTarget(x_m=-15.0, y_m=8.0, target_id=2, target_type="behind_storefront")

    measurements = mapping.visible_measurements(
        pose,
        [visible_front, blocked_far],
        max_range_m=45.0,
        fov_deg=120.0,
        occluders=mapping.build_p03_urban_occluders(include_parked_vehicles=False),
    )

    assert [m.target_id for m in measurements] == [1]


def test_first_hit_rule_keeps_nearest_ray_endpoint_only() -> None:
    pose = mapping.EgoPose(x_m=0.0, y_m=0.0, heading_deg=0.0, speed_mps=10.0)
    near = mapping.WorldTarget(x_m=0.0, y_m=10.0, target_id=1)
    far = mapping.WorldTarget(x_m=0.0, y_m=20.0, target_id=2)

    measurements = mapping.visible_measurements(
        pose,
        [far, near],
        max_range_m=45.0,
        fov_deg=120.0,
        first_hit_occlusion=True,
        occlusion_bin_deg=0.25,
    )

    assert [m.target_id for m in measurements] == [1]


def test_logodds_first_hit_keeps_behind_surface_unknown() -> None:
    pose = mapping.EgoPose(x_m=0.0, y_m=0.0, heading_deg=0.0, speed_mps=10.0)
    spec = mapping.MapGridSpec.uniform_square(grid_size=80, x_min_m=-10, x_max_m=10, y_min_m=0, y_max_m=30)
    prob, _ = mapping.accumulate_probability_map(
        [pose],
        per_frame_angles=[[0.0, 0.0]],
        per_frame_ranges=[[10.0, 20.0]],
        grid_spec=spec,
        max_range_m=30.0,
        beam_width_deg=2.0,
        p_occ=0.65,
        p_free=0.35,
        first_hit_occlusion=True,
        occlusion_bin_deg=0.25,
        free_ray_width_deg=4.0,
    )
    near_row, near_col = mapping.world_to_grid_spec(0.0, 10.0, spec)
    far_row, far_col = mapping.world_to_grid_spec(0.0, 20.0, spec)
    free_row, free_col = mapping.world_to_grid_spec(0.0, 5.0, spec)

    assert prob[near_row, near_col] > 0.5
    assert np.isclose(prob[far_row, far_col], 0.4, atol=1e-3)
    assert prob[free_row, free_col] < 0.4


def test_dynamic_object_is_not_accumulated_into_static_reference_map() -> None:
    targets = mapping.build_p03_mapping_targets(
        seed=0,
        wall_spacing_m=0.35,
        include_dynamic=True,
        include_resolution_probes=False,
    )
    dynamic_targets = [t for t in targets if t.is_dynamic]
    assert len(dynamic_targets) == 1

    spec = mapping.P03_URBAN_GRID
    static_grid = mapping.occupancy_grid_from_targets(
        targets,
        grid_spec=spec,
        include_dynamic=False,
        sigma_cells=0.0,
    )
    dynamic_grid = mapping.occupancy_grid_from_targets(
        targets,
        grid_spec=spec,
        include_dynamic=True,
        sigma_cells=0.0,
    )
    dyn = dynamic_targets[0]
    row, col = mapping.world_to_grid_spec(dyn.x_m, dyn.y_m, spec)

    assert static_grid[row, col] == 0.0
    assert dynamic_grid[row, col] == 1.0


def test_urban_ego_route_uses_right_hand_lane_offset() -> None:
    poses = mapping.generate_p03_urban_ego_trajectory(n_steps=4, dt_s=0.2, speed_mps=10.0)

    assert all(np.isclose(p.x_m, 1.75) for p in poses)
    assert np.isclose(poses[0].y_m, -20.0)
    assert np.allclose([p.y_m for p in poses], [-20.0, -18.0, -16.0, -14.0])


def test_uniform_map_grid_spec_decouples_width_from_radar_range() -> None:
    spec = mapping.MapGridSpec.uniform_square(
        grid_size=128,
        x_min_m=-20.0,
        x_max_m=20.0,
        y_min_m=0.0,
        y_max_m=40.0,
    )

    assert spec.shape == (128, 128)
    assert spec.is_square_cell
    assert np.isclose(spec.cell_x_m, 0.3125)
    assert np.isclose(spec.cell_y_m, 0.3125)
    assert mapping.world_to_grid_spec(0.0, 40.0, spec) == (0, 64)
    assert mapping.world_to_grid_spec(20.0, 0.0, spec) == (127, 127)

    legacy = mapping.MapGridSpec.legacy(grid_size=128, grid_range_m=40.0)
    assert not legacy.is_square_cell
    assert np.isclose(legacy.cell_x_m, 0.625)
    assert np.isclose(legacy.cell_y_m, 0.3125)


def test_p03_resolution_probes_are_appendix_only() -> None:
    main_targets = mapping.build_p03_mapping_targets(seed=1, include_resolution_probes=False)
    appendix_targets = mapping.build_p03_mapping_targets(seed=1, include_resolution_probes=True)

    assert not any(t.target_type.startswith("resolution_pair") for t in main_targets)
    probe_targets = [t for t in appendix_targets if t.target_type.startswith("resolution_pair")]
    assert len(probe_targets) == 8


def test_mapping_dataset_schema_and_oracle_point_cloud_error(tmp_path) -> None:
    data = mapping_gen.generate_mapping_split(
        n_scenes=1,
        seed=5,
        n_steps=1,
        wall_spacing_m=4.0,
        grid_size=32,
        grid_range_m=40.0,
        include_dynamic=False,
        radar=gen.P03_RADAR,
    )
    path = tmp_path / "mapping.h5"
    from common.hdf5_io import save_hdf5

    save_hdf5(path, **data)
    ds = train.MappingDetectionDataset(path)

    assert ds.x.shape[1:] == (2, gen.N_RX)
    assert ds.gt_ogm.shape == (1, 32, 32)
    assert ds.poses.shape == (1, 1, 4)
    assert ds.grid_spec.is_square_cell
    assert np.isclose(ds.grid_cell_x_m, ds.grid_cell_y_m)
    assert np.isclose(ds.grid_spec.x_min_m, -20.0)
    assert np.isclose(ds.grid_spec.x_max_m, 20.0)
    assert np.isclose(ds.radar_max_range_m, 40.0)
    assert float(np.max(ds.range_m)) > 20.0
    assert ds.radar_bw_hz == gen.P03_RADAR.bw
    assert np.all(ds.velocity_mps > -gen.P03_RADAR.max_vel)
    assert np.all(ds.velocity_mps < gen.P03_RADAR.max_vel)

    oracle = train._mapping_metrics_for_angles(ds, ds.angle_deg)
    biased = train._mapping_metrics_for_angles(ds, ds.angle_deg + 8.0)

    assert oracle["point_error_mean_m"] < 1e-5
    assert biased["point_error_mean_m"] > 1.0
    assert "ogm_thr0p5_iou" in oracle


def test_results_radar_preset_uses_200mhz_resolution_standard() -> None:
    radar = gen.build_p03_radar(bandwidth_hz=200e6, n_fast=1024)
    assert radar.fs / radar.bw == 4.0
    assert np.isclose(radar.range_res, 299_792_458.0 / (2.0 * 200e6))
    assert radar.range_res < 1.0
    assert radar.T_chirp == gen.P03_RADAR.T_chirp
