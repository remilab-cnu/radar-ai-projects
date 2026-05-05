#!/usr/bin/env python3
"""P03 -- Generate moving-ego radar mapping / DoA comparison data.

This is the mapping-first P03 data path.  It keeps the neural-learning unit as a
per-detection DoA task, but evaluates it through downstream environmental
perception artifacts: point-cloud maps and probabilistic occupancy maps.

Mainline assumptions:
  * ego-motion is known exactly,
  * range/Doppler association is provided by simulator metadata for labels,
  * all DoA methods consume the same RD-selected antenna vector,
  * ego-motion-error studies are appendix-only and use GT DoA/range.

HDF5 schema per split:
  x_ant            (N_det, 2, N_rx)       selected antenna vector [real, imag]
  y_spectrum       (N_det, 181)           Gaussian DoA spectrum label
  angle_deg        (N_det,)               GT DoA relative to ego heading
  range_m          (N_det,)               GT range from ego to target
  velocity_mps     (N_det,)               simulator closing velocity
  scene_idx        (N_det,)               scene index in this split
  frame_idx        (N_det,)               ego frame index
  target_id        (N_det,)               stable target/scatterer id
  is_dynamic       (N_det,)               dynamic-target flag
  gt_ogm           (N_scene, G, G)        persistent static occupancy map
  poses            (N_scene, T, 4)        [x, y, heading_deg, speed_mps]

Schema v2 adds explicit map bounds so map-cell geometry is decoupled from
radar maximum range.  The lecture default is a physically uniform grid:
``x=[-20,20] m``, ``y=[0,40] m``, ``128×128`` cells.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.cli import base_parser
from common.hdf5_io import save_hdf5
from common.seed import seed_everything
from generate_data import ANGLE_GRID, P03_RADAR, build_p03_radar, _label_spectrum
from mapping import (
    MapGridSpec,
    build_p03_mapping_targets,
    generate_ego_trajectory,
    occupancy_grid_from_targets,
    simulate_rd_selected_vector,
    visible_measurements,
)

BASE = Path(__file__).parent
MAPPING_SCHEMA_VERSION = 2
RESULTS_BW_MHZ = 200.0
RESULTS_N_FAST = 1024
DEFAULT_MAP_X_MIN_M = -20.0
DEFAULT_MAP_X_MAX_M = 20.0
DEFAULT_MAP_Y_MIN_M = 0.0
DEFAULT_MAP_Y_MAX_M = 40.0
DEFAULT_RADAR_MAX_RANGE_M = 40.0


def _target_lookup(targets):
    return {int(t.target_id): t for t in targets}


def _rcs_for_requested_snr(radar, range_m: float, angle_deg: float) -> float:
    """Path-loss compensation for the selected mapping radar preset."""

    angle_gain = max(float(np.cos(np.deg2rad(angle_deg)) ** 2), 0.10)
    range_gain = (float(range_m) / radar.reference_range_m) ** 4
    return float(radar.reference_rcs_m2 * range_gain / angle_gain)


def generate_mapping_split(
    n_scenes: int,
    seed: int,
    n_steps: int = 6,
    dt_s: float = 0.20,
    ego_speed_mps: float = 8.0,
    wall_spacing_m: float = 0.35,
    grid_size: int = 128,
    grid_range_m: float = 40.0,
    map_x_min_m: float = DEFAULT_MAP_X_MIN_M,
    map_x_max_m: float = DEFAULT_MAP_X_MAX_M,
    map_y_min_m: float = DEFAULT_MAP_Y_MIN_M,
    map_y_max_m: float = DEFAULT_MAP_Y_MAX_M,
    radar_max_range_m: float = DEFAULT_RADAR_MAX_RANGE_M,
    include_dynamic: bool = False,
    snr_db_range: tuple[float, float] = (10.0, 25.0),
    radar=P03_RADAR,
) -> dict[str, np.ndarray]:
    """Generate one mapping split as flat detections plus scene-level maps."""

    rng = np.random.default_rng(seed)
    grid_spec = MapGridSpec.uniform_square(
        grid_size=grid_size,
        x_min_m=map_x_min_m,
        x_max_m=map_x_max_m,
        y_min_m=map_y_min_m,
        y_max_m=map_y_max_m,
    )
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    angle_rows: list[np.float32] = []
    range_rows: list[np.float32] = []
    vel_rows: list[np.float32] = []
    snr_rows: list[np.float32] = []
    requested_snr_rows: list[np.float32] = []
    rcs_rows: list[np.float32] = []
    rbin_rows: list[np.int32] = []
    dbin_rows: list[np.int32] = []
    scene_rows: list[np.int32] = []
    frame_rows: list[np.int32] = []
    target_rows: list[np.int32] = []
    dynamic_rows: list[np.int32] = []
    n_targets_rows: list[np.int32] = []

    poses_all = np.zeros((n_scenes, n_steps, 4), dtype=np.float32)
    gt_ogm_all = np.zeros((n_scenes, grid_spec.ny, grid_spec.nx), dtype=np.float32)
    n_visible_per_scene = np.zeros(n_scenes, dtype=np.int32)

    for scene_idx in range(n_scenes):
        scene_seed = int(seed + 1009 * scene_idx)
        targets = build_p03_mapping_targets(
            seed=scene_seed,
            wall_spacing_m=wall_spacing_m,
            include_dynamic=True,
            include_resolution_probes=False,
        )
        target_by_id = _target_lookup(targets)
        gt_ogm_all[scene_idx] = occupancy_grid_from_targets(
            targets,
            grid_size=grid_size,
            grid_range_m=grid_range_m,
            grid_spec=grid_spec,
            include_dynamic=False,
            sigma_cells=1.0,
        )

        # Small deterministic lateral/heading variation per scene keeps the
        # lecture scenario recognizable while preventing one fixed trajectory
        # from becoming the entire training distribution.
        start_x = float(rng.uniform(-0.35, 0.35))
        heading = float(rng.uniform(-1.0, 1.0))
        poses = generate_ego_trajectory(
            n_steps=n_steps,
            dt_s=dt_s,
            speed_mps=ego_speed_mps,
            start_x_m=start_x,
            start_y_m=0.0,
            heading_deg=heading,
        )
        poses_all[scene_idx] = np.asarray(
            [[p.x_m, p.y_m, p.heading_deg, p.speed_mps] for p in poses],
            dtype=np.float32,
        )

        for frame_idx, pose in enumerate(poses):
            measurements = visible_measurements(
                pose,
                targets,
                max_range_m=radar_max_range_m,
                fov_deg=120.0,
                include_dynamic=include_dynamic,
            )
            for meas in measurements:
                target = target_by_id[meas.target_id]
                requested_snr_db = float(rng.uniform(*snr_db_range))
                # Compensate range/angle path loss so DoA difficulty is not just
                # a hidden proxy for target distance.  Preserve a mild RCS factor
                # from the map target so walls/points are not identical.
                rcs = _rcs_for_requested_snr(radar, meas.range_m, meas.angle_deg) * max(target.rcs_m2, 0.1)
                ant_vec, sim_meta = simulate_rd_selected_vector(
                    radar,
                    meas,
                    snr_db=requested_snr_db,
                    rcs_m2=rcs,
                    seed=int(rng.integers(0, 2**31)),
                )
                x_rows.append(np.stack([ant_vec.real, ant_vec.imag], axis=0).astype(np.float32))
                y_rows.append(_label_spectrum(meas.angle_deg))
                angle_rows.append(np.float32(meas.angle_deg))
                range_rows.append(np.float32(meas.range_m))
                vel_rows.append(np.float32(meas.radial_velocity_mps))
                snr_rows.append(np.float32(sim_meta["actual_snr_db"]))
                requested_snr_rows.append(np.float32(requested_snr_db))
                rcs_rows.append(np.float32(rcs))
                rbin_rows.append(np.int32(sim_meta["r_bin"]))
                dbin_rows.append(np.int32(sim_meta["d_bin"]))
                scene_rows.append(np.int32(scene_idx))
                frame_rows.append(np.int32(frame_idx))
                target_rows.append(np.int32(meas.target_id))
                dynamic_rows.append(np.int32(meas.is_dynamic))
                n_targets_rows.append(np.int32(1))
                n_visible_per_scene[scene_idx] += 1

    if not x_rows:
        raise RuntimeError("mapping split generated zero detections; widen FoV/range or inspect scenario")

    return {
        "x_ant": np.stack(x_rows, axis=0).astype(np.float32),
        "y_spectrum": np.stack(y_rows, axis=0).astype(np.float32),
        "angle_deg": np.asarray(angle_rows, dtype=np.float32),
        "range_m": np.asarray(range_rows, dtype=np.float32),
        "velocity_mps": np.asarray(vel_rows, dtype=np.float32),
        "snr_db": np.asarray(snr_rows, dtype=np.float32),
        "requested_snr_db": np.asarray(requested_snr_rows, dtype=np.float32),
        "target_rcs_m2": np.asarray(rcs_rows, dtype=np.float32),
        "r_bin": np.asarray(rbin_rows, dtype=np.int32),
        "d_bin": np.asarray(dbin_rows, dtype=np.int32),
        "scene_idx": np.asarray(scene_rows, dtype=np.int32),
        "frame_idx": np.asarray(frame_rows, dtype=np.int32),
        "target_id": np.asarray(target_rows, dtype=np.int32),
        "is_dynamic": np.asarray(dynamic_rows, dtype=np.int32),
        "n_targets": np.asarray(n_targets_rows, dtype=np.int32),
        "gt_ogm": gt_ogm_all.astype(np.float32),
        "poses": poses_all.astype(np.float32),
        "n_visible_per_scene": n_visible_per_scene.astype(np.int32),
        "angle_grid_deg": ANGLE_GRID.astype(np.float32),
        "radar_fc_hz": np.array([radar.fc], dtype=np.float64),
        "radar_bw_hz": np.array([radar.bw], dtype=np.float64),
        "radar_fs_hz": np.array([radar.fs], dtype=np.float64),
        "radar_n_fast": np.array([radar.N_samples], dtype=np.int32),
        "radar_range_res_m": np.array([radar.range_res], dtype=np.float32),
        "fs_over_bandwidth": np.array([radar.fs / radar.bw], dtype=np.float32),
        "grid_size": np.array([grid_size], dtype=np.int32),
        "grid_range_m": np.array([grid_range_m], dtype=np.float32),
        "grid_nx": np.array([grid_spec.nx], dtype=np.int32),
        "grid_ny": np.array([grid_spec.ny], dtype=np.int32),
        "grid_x_min_m": np.array([grid_spec.x_min_m], dtype=np.float32),
        "grid_x_max_m": np.array([grid_spec.x_max_m], dtype=np.float32),
        "grid_y_min_m": np.array([grid_spec.y_min_m], dtype=np.float32),
        "grid_y_max_m": np.array([grid_spec.y_max_m], dtype=np.float32),
        "grid_cell_x_m": np.array([grid_spec.cell_x_m], dtype=np.float32),
        "grid_cell_y_m": np.array([grid_spec.cell_y_m], dtype=np.float32),
        "radar_max_range_m": np.array([radar_max_range_m], dtype=np.float32),
        "n_steps": np.array([n_steps], dtype=np.int32),
        "dt_s": np.array([dt_s], dtype=np.float32),
        "ego_speed_mps": np.array([ego_speed_mps], dtype=np.float32),
        "wall_spacing_m": np.array([wall_spacing_m], dtype=np.float32),
        "include_dynamic": np.array([int(include_dynamic)], dtype=np.int32),
        "schema_version": np.array([MAPPING_SCHEMA_VERSION], dtype=np.int32),
    }


def main() -> None:
    parser = base_parser("Generate P03 moving-ego mapping DoA datasets")
    parser.add_argument("--n_train_scenes", type=int, default=80)
    parser.add_argument("--n_val_scenes", type=int, default=16)
    parser.add_argument("--n_test_scenes", type=int, default=16)
    parser.add_argument("--n_steps", type=int, default=6)
    parser.add_argument("--dt_s", type=float, default=0.20)
    parser.add_argument("--ego_speed_mps", type=float, default=8.0)
    parser.add_argument("--wall_spacing_m", type=float, default=0.35,
                        help="Dense wall scatterer spacing. Result default 0.35 m; smoke may relax to 1 m.")
    parser.add_argument("--grid_size", type=int, default=128)
    parser.add_argument("--grid_range_m", type=float, default=40.0)
    parser.add_argument("--map_x_min_m", type=float, default=DEFAULT_MAP_X_MIN_M)
    parser.add_argument("--map_x_max_m", type=float, default=DEFAULT_MAP_X_MAX_M)
    parser.add_argument("--map_y_min_m", type=float, default=DEFAULT_MAP_Y_MIN_M)
    parser.add_argument("--map_y_max_m", type=float, default=DEFAULT_MAP_Y_MAX_M)
    parser.add_argument("--radar_max_range_m", type=float, default=DEFAULT_RADAR_MAX_RANGE_M)
    parser.add_argument("--radar_bw_mhz", type=float, default=RESULTS_BW_MHZ,
                        help="Mapping radar bandwidth. Results standard: 200 MHz; low-res stress: 50 MHz.")
    parser.add_argument("--radar_n_fast", type=int, default=RESULTS_N_FAST,
                        help="Fast-time samples. Use 1024 with 200 MHz to preserve the 1.28 us sweep.")
    parser.add_argument("--include_dynamic", action="store_true", help="Include dynamic target detections; GT map remains static")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    if args.smoke:
        args.n_train_scenes, args.n_val_scenes, args.n_test_scenes = 2, 1, 1
        args.n_steps = min(args.n_steps, 4)
        args.grid_size = min(args.grid_size, 48)
        args.wall_spacing_m = max(args.wall_spacing_m, 1.0)
        if "--radar_bw_mhz" not in sys.argv:
            args.radar_bw_mhz = 50.0
        if "--radar_n_fast" not in sys.argv:
            args.radar_n_fast = 256
    if "--grid_range_m" not in sys.argv:
        # Keep legacy metadata aligned with the forward map extent unless a
        # caller explicitly requests a different compatibility value.
        args.grid_range_m = float(args.map_y_max_m)

    seed_everything(args.seed)
    out_dir = Path(args.out_dir) if args.out_dir else BASE / "data_mapping"
    out_dir.mkdir(parents=True, exist_ok=True)
    radar = build_p03_radar(bandwidth_hz=args.radar_bw_mhz * 1e6, n_fast=args.radar_n_fast)

    print("=== P03 Moving-Ego Radar Mapping Dataset ===")
    print(
        f"  radar: fc={radar.fc/1e9:.1f} GHz, B={radar.bw/1e6:.1f} MHz, "
        f"N_fast={radar.N_samples}, range_res={radar.range_res:.3f} m, fs/BW={radar.fs/radar.bw:.1f}"
    )
    print(f"  ego: speed={args.ego_speed_mps:.1f} m/s, steps={args.n_steps}, dt={args.dt_s:.2f}s, perfect ego-motion")
    print(f"  scene: dense point-scatterer walls, spacing={args.wall_spacing_m:.2f} m, resolution probes=appendix-only")
    map_spec = MapGridSpec.uniform_square(
        grid_size=args.grid_size,
        x_min_m=args.map_x_min_m,
        x_max_m=args.map_x_max_m,
        y_min_m=args.map_y_min_m,
        y_max_m=args.map_y_max_m,
    )
    print(
        f"  map: grid={map_spec.nx}x{map_spec.ny}, "
        f"x=[{map_spec.x_min_m:.1f},{map_spec.x_max_m:.1f}] m, "
        f"y=[{map_spec.y_min_m:.1f},{map_spec.y_max_m:.1f}] m, "
        f"cell={map_spec.cell_x_m:.4f}x{map_spec.cell_y_m:.4f} m, "
        f"radar_max_range={args.radar_max_range_m:.1f} m, include_dynamic={args.include_dynamic}"
    )
    print("  output: per-detection x_ant + GT static OGM + ego poses")

    splits = [
        ("train", args.n_train_scenes, args.seed),
        ("val", args.n_val_scenes, args.seed + 100000),
        ("test", args.n_test_scenes, args.seed + 200000),
    ]
    for name, n_scenes, split_seed in splits:
        print(f"\n[{name}] Generating {n_scenes} scenes...")
        data = generate_mapping_split(
            n_scenes=n_scenes,
            seed=split_seed,
            n_steps=args.n_steps,
            dt_s=args.dt_s,
            ego_speed_mps=args.ego_speed_mps,
            wall_spacing_m=args.wall_spacing_m,
            grid_size=args.grid_size,
            grid_range_m=args.grid_range_m,
            map_x_min_m=args.map_x_min_m,
            map_x_max_m=args.map_x_max_m,
            map_y_min_m=args.map_y_min_m,
            map_y_max_m=args.map_y_max_m,
            radar_max_range_m=args.radar_max_range_m,
            include_dynamic=args.include_dynamic,
            radar=radar,
        )
        print(f"  detections={data['x_ant'].shape[0]}, x_ant={data['x_ant'].shape}, gt_ogm={data['gt_ogm'].shape}")
        save_hdf5(out_dir / f"{name}.h5", **data)


if __name__ == "__main__":
    main()
