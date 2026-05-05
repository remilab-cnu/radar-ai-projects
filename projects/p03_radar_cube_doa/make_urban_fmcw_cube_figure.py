#!/usr/bin/env python3
"""Generate scene-consistent FMCW cube slice figure for P03 lecture.

The figure uses the current urban T-intersection scene, right-hand lane ego
route, opaque LoS filtering, and first-hit static detections.  It replaces the
older Week12 generic FMCW cube example so the signal-domain panels match the
mapping scene used later in the lecture.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.fmcw_simulator import generate_scene, range_axis, range_doppler_map, to_db, velocity_axis
from generate_data import build_p03_radar
from mapping import (
    AxisAlignedBox,
    EgoPose,
    WorldTarget,
    build_p03_cross_traffic_target,
    build_p03_mapping_targets,
    build_p03_urban_occluders,
    generate_p03_urban_ego_trajectory,
    measurement_from_world,
    visible_measurements,
)

BASE = Path(__file__).resolve().parent


def draw_occluders(ax, occluders: list[AxisAlignedBox], alpha: float = 0.32) -> None:
    for box in occluders:
        fc = "#94a3b8"
        if box.name.startswith("parked"):
            fc = "#f59e0b"
        ax.add_patch(
            Rectangle(
                (box.x_min_m, box.y_min_m),
                box.x_max_m - box.x_min_m,
                box.y_max_m - box.y_min_m,
                facecolor=fc,
                edgecolor="#334155",
                lw=1.0,
                alpha=alpha,
            )
        )


def draw_fov(ax, pose: EgoPose, rng: float = 28.0, alpha: float = 0.16) -> None:
    pts = [(pose.x_m, pose.y_m)]
    for a_deg in (pose.heading_deg - 60.0, pose.heading_deg + 60.0):
        a = np.deg2rad(a_deg)
        pts.append((pose.x_m + rng * np.sin(a), pose.y_m + rng * np.cos(a)))
    ax.add_patch(Polygon(pts, closed=True, facecolor="#64748b", edgecolor="#64748b", alpha=alpha, lw=0.8))


def angle_fft_cube(rd_cube: np.ndarray, radar, n_angle: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Angle FFT over antenna for an RD cube with shape (rx, doppler, range)."""

    n_rx, n_dop, n_range = rd_cube.shape
    w = np.hanning(n_rx).astype(np.float64)
    padded = np.zeros((n_angle, n_dop, n_range), dtype=np.complex128)
    padded[:n_rx, :, :] = rd_cube * w[:, None, None]
    spec = np.fft.fftshift(np.fft.fft(padded, axis=0), axes=0)
    u = np.fft.fftshift(np.fft.fftfreq(n_angle))
    angle_axis = np.degrees(np.arcsin(np.clip(u / (radar.d_rx / radar.lam), -1.0, 1.0)))
    return spec, angle_axis.astype(np.float32)


def target_dicts_from_scene(
    pose: EgoPose,
    static_targets: list[WorldTarget],
    dynamic_target: WorldTarget,
    occluders: list[AxisAlignedBox],
    radar_max_range_m: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Build simulator targets from visible current-scene measurements."""

    target_by_id = {t.target_id: t for t in static_targets + [dynamic_target]}
    static_meas = visible_measurements(
        pose,
        static_targets,
        max_range_m=radar_max_range_m,
        fov_deg=120.0,
        include_dynamic=False,
        occluders=occluders,
        first_hit_occlusion=True,
        occlusion_bin_deg=0.25,
    )
    dynamic_meas = visible_measurements(
        pose,
        [dynamic_target],
        max_range_m=radar_max_range_m,
        fov_deg=120.0,
        include_dynamic=True,
        occluders=occluders,
        first_hit_occlusion=False,
    )
    all_meas = static_meas + dynamic_meas
    sim_targets: list[dict] = []
    marker_rows: list[dict] = []
    for i, meas in enumerate(all_meas):
        src = target_by_id[int(meas.target_id)]
        # Keep per-scatterer amplitudes moderate; the goal is a readable cube
        # slice, not calibrated RCS inference.
        rcs = float(np.clip(src.rcs_m2, 0.5, 12.0))
        if meas.is_dynamic:
            rcs *= 2.0
        sim_targets.append(
            {
                "range": float(meas.range_m),
                "velocity": float(meas.radial_velocity_mps),
                "angle": float(meas.angle_deg),
                "rcs": rcs,
                "phase": float(0.37 * (i + 1)),
            }
        )
        marker_rows.append(
            {
                "range_m": float(meas.range_m),
                "velocity_mps": float(meas.radial_velocity_mps),
                "angle_deg": float(meas.angle_deg),
                "world_x_m": float(meas.world_x_m),
                "world_y_m": float(meas.world_y_m),
                "target_type": str(meas.target_type),
                "is_dynamic": bool(meas.is_dynamic),
            }
        )
    return sim_targets, marker_rows, [m for m in marker_rows if m["is_dynamic"]]


def make_figure(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    radar = build_p03_radar(bandwidth_hz=args.bw_mhz * 1e6, n_fast=args.n_fast)
    static_targets = build_p03_mapping_targets(seed=args.seed, wall_spacing_m=0.35, include_dynamic=False, include_resolution_probes=False)
    occluders = list(build_p03_urban_occluders(include_parked_vehicles=True))
    poses = generate_p03_urban_ego_trajectory(n_steps=24, dt_s=0.2, speed_mps=10.0)
    frame_idx = int(args.frame_idx)
    pose = poses[frame_idx]
    dynamic = build_p03_cross_traffic_target(frame_idx, len(poses))

    sim_targets, markers, dyn_markers = target_dicts_from_scene(
        pose,
        static_targets,
        dynamic,
        occluders,
        radar_max_range_m=args.radar_max_range_m,
    )
    if not sim_targets:
        raise RuntimeError("no visible scene targets for FMCW cube figure")
    raw, meta = generate_scene(radar, sim_targets, snr_db=args.snr_db, seed=args.seed + 9000, return_meta=True)
    rd_cube = range_doppler_map(raw, radar=radar, window_range="hann", window_doppler="hann").astype(np.complex64)
    ranges = range_axis(radar)
    velocities = velocity_axis(radar)
    angle_cube, angles = angle_fft_cube(rd_cube, radar, n_angle=args.n_angle)

    rd_power = np.mean(np.abs(rd_cube), axis=0)  # doppler, range
    ra_power = np.max(np.abs(angle_cube), axis=1)  # angle, range; max over Doppler
    if dyn_markers:
        r_select = dyn_markers[0]["range_m"]
    else:
        strongest = int(np.argmax(np.max(rd_power, axis=0)))
        r_select = float(ranges[strongest])
    r_bin = int(np.argmin(np.abs(ranges - r_select)))
    # Use a small range slab to make the DA panel readable for extended surfaces.
    r0 = max(0, r_bin - 1)
    r1 = min(len(ranges), r_bin + 2)
    da_power = np.max(np.abs(angle_cube[:, :, r0:r1]), axis=2)  # angle, doppler

    rd_db = np.clip(to_db(rd_power), -45, 0)
    ra_db = np.clip(to_db(ra_power), -35, 0)
    da_db = np.clip(to_db(da_power), -35, 0)

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2), constrained_layout=True)
    ax = axes[0, 0]
    ax.add_patch(Rectangle((-4, -25), 8, 55, facecolor="#e2e8f0", edgecolor="none", alpha=0.55, label="ego road"))
    ax.add_patch(Rectangle((-25, 20), 60, 10, facecolor="#e2e8f0", edgecolor="none", alpha=0.55, label="cross street"))
    draw_occluders(ax, occluders, alpha=0.35)
    draw_fov(ax, pose, rng=28.0)
    static_markers = [m for m in markers if not m["is_dynamic"]]
    if static_markers:
        show = static_markers[:: max(len(static_markers) // 180, 1)]
        ax.scatter([m["world_x_m"] for m in show], [m["world_y_m"] for m in show], s=8, c="#fde047", label="visible first-hit static")
    if dyn_markers:
        ax.scatter([m["world_x_m"] for m in dyn_markers], [m["world_y_m"] for m in dyn_markers], s=135, marker="*", c="#ef4444", edgecolor="white", label="visible moving vehicle")
    ax.plot([p.x_m for p in poses[: frame_idx + 1]], [p.y_m for p in poses[: frame_idx + 1]], "c.-", lw=2.0, ms=4, label="right-lane ego route")
    ax.scatter([pose.x_m], [pose.y_m], s=60, c="#06b6d4", edgecolor="white", zorder=6)
    ax.set_xlim(-26, 35)
    ax.set_ylim(-23, 55)
    ax.set_aspect("equal")
    ax.set_title(f"(a) Current urban frame for FMCW cube (frame {frame_idx + 1})")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", fontsize=7.0, framealpha=0.9)

    ax = axes[0, 1]
    im = ax.imshow(rd_db, origin="lower", aspect="auto", extent=[ranges[0], ranges[-1], velocities[0], velocities[-1]], cmap="viridis", vmin=-45, vmax=0)
    for m in markers[:: max(len(markers) // 80, 1)]:
        ax.plot(m["range_m"], m["velocity_mps"], ".", color="#e5e7eb", ms=2.2, alpha=0.75)
    if dyn_markers:
        m = dyn_markers[0]
        ax.plot(m["range_m"], m["velocity_mps"], "*", color="#ef4444", ms=12, mec="white")
        ax.text(m["range_m"] + 0.8, m["velocity_mps"] + 0.5, "moving", color="white", fontsize=8, weight="bold")
    ax.set_xlim(0, args.radar_max_range_m)
    ax.set_ylim(-15, 15)
    ax.set_xlabel("Range [m]")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title("(b) Range-Doppler (Rx-averaged)")
    fig.colorbar(im, ax=ax, label="Power [dB]", shrink=0.86)

    ax = axes[1, 0]
    im = ax.imshow(ra_db, origin="lower", aspect="auto", extent=[ranges[0], ranges[-1], angles[0], angles[-1]], cmap="viridis", vmin=-35, vmax=0)
    for m in markers[:: max(len(markers) // 80, 1)]:
        ax.plot(m["range_m"], m["angle_deg"], ".", color="#e5e7eb", ms=2.2, alpha=0.75)
    if dyn_markers:
        m = dyn_markers[0]
        ax.plot(m["range_m"], m["angle_deg"], "*", color="#ef4444", ms=12, mec="white")
    ax.set_xlim(0, args.radar_max_range_m)
    ax.set_ylim(-65, 65)
    ax.set_xlabel("Range [m]")
    ax.set_ylabel("Angle [deg]")
    ax.set_title("(c) Range-Angle (Doppler-integrated)")
    fig.colorbar(im, ax=ax, label="Power [dB]", shrink=0.86)

    ax = axes[1, 1]
    im = ax.imshow(da_db, origin="lower", aspect="auto", extent=[velocities[0], velocities[-1], angles[0], angles[-1]], cmap="viridis", vmin=-35, vmax=0)
    if dyn_markers:
        m = dyn_markers[0]
        ax.plot(m["velocity_mps"], m["angle_deg"], "*", color="#ef4444", ms=12, mec="white")
    # Static markers close to selected range.
    close = [m for m in markers if abs(m["range_m"] - float(ranges[r_bin])) <= 1.5 and not m["is_dynamic"]]
    for m in close[:: max(len(close) // 50, 1) if close else 1]:
        ax.plot(m["velocity_mps"], m["angle_deg"], ".", color="#e5e7eb", ms=2.5, alpha=0.8)
    ax.set_xlim(-15, 15)
    ax.set_ylim(-65, 65)
    ax.set_xlabel("Velocity [m/s]")
    ax.set_ylabel("Angle [deg]")
    ax.set_title(f"(d) Doppler-Angle near R={ranges[r_bin]:.1f} m")
    fig.colorbar(im, ax=ax, label="Power [dB]", shrink=0.86)

    fig.suptitle(
        "Scene-consistent FMCW cube slices: urban T-intersection, right-lane ego route",
        fontsize=14,
    )
    out_png = out_dir / "p03_urban_fmcw_cube_slices.png"
    fig.savefig(out_png, dpi=170)
    plt.close(fig)

    out_json = out_dir / "p03_urban_fmcw_cube_slices.json"
    out_json.write_text(
        json.dumps(
            {
                "figure": str(out_png),
                "frame_idx": frame_idx,
                "pose": {"x_m": pose.x_m, "y_m": pose.y_m, "heading_deg": pose.heading_deg, "speed_mps": pose.speed_mps},
                "radar": {"bw_mhz": args.bw_mhz, "n_fast": args.n_fast, "range_res_m": radar.range_res, "n_rx": radar.N_rx, "n_chirps": radar.N_chirps},
                "n_visible_targets_for_sim": len(sim_targets),
                "n_dynamic_visible": len(dyn_markers),
                "selected_da_range_m": float(ranges[r_bin]),
                "markers_sample": markers[:20],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"figure": str(out_png), "metadata": str(out_json), "n_targets": len(sim_targets), "n_dynamic_visible": len(dyn_markers)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate P03 urban FMCW cube slice figure")
    parser.add_argument("--out_dir", default=str(BASE / "artifacts" / "grad_week12_urban_rebuild"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame_idx", type=int, default=13)
    parser.add_argument("--bw_mhz", type=float, default=200.0)
    parser.add_argument("--n_fast", type=int, default=1024)
    parser.add_argument("--n_angle", type=int, default=128)
    parser.add_argument("--radar_max_range_m", type=float, default=45.0)
    parser.add_argument("--snr_db", type=float, default=32.0)
    args = parser.parse_args()
    print(json.dumps(make_figure(args), indent=2))


if __name__ == "__main__":
    main()
