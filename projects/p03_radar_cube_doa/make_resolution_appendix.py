#!/usr/bin/env python3
"""Generate P03 range-resolution appendix maps for the urban mapping scene.

The appendix isolates range resolution: DoA is oracle, ego pose is perfect, and
only the range-bin center changes with bandwidth.  The scene is the same
visibility-aware urban T-intersection used by the main P03 lecture material:
opaque buildings/parked vehicles block line of sight, first-hit detections are
used for the inverse sensor model, and cross-traffic is not accumulated into the
static map.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np

from mapping import (
    AxisAlignedBox,
    EgoPose,
    MapGridSpec,
    WorldTarget,
    accumulate_resolution_probability_map,
    build_p03_cross_traffic_target,
    build_p03_mapping_targets,
    build_p03_urban_occluders,
    generate_p03_urban_ego_trajectory,
    map_metrics,
    occupancy_grid_from_targets,
    visible_measurements,
)

C = 299_792_458.0
BASE = Path(__file__).resolve().parent


def quantize_range(range_m: float, range_resolution_m: float) -> float:
    """Quantize a true range to the nearest range-bin center."""

    return float(np.round(float(range_m) / float(range_resolution_m)) * float(range_resolution_m))


def frame_measurements(
    poses: Sequence[EgoPose],
    targets: Sequence[WorldTarget],
    range_resolution_m: float,
    radar_max_range_m: float,
    occluders: Sequence[AxisAlignedBox],
    occlusion_bin_deg: float,
) -> tuple[list[list[float]], list[list[float]], np.ndarray, list[int]]:
    """Return oracle-DoA / quantized-range measurements for each ego pose."""

    per_angles: list[list[float]] = []
    per_ranges: list[list[float]] = []
    counts: list[int] = []
    abs_range_err: list[float] = []
    for pose in poses:
        angles: list[float] = []
        ranges: list[float] = []
        measurements = visible_measurements(
            pose,
            targets,
            max_range_m=radar_max_range_m,
            fov_deg=120.0,
            include_dynamic=False,
            occluders=occluders,
            first_hit_occlusion=True,
            occlusion_bin_deg=occlusion_bin_deg,
        )
        for meas in measurements:
            rq = quantize_range(meas.range_m, range_resolution_m)
            angles.append(float(meas.angle_deg))
            ranges.append(float(rq))
            abs_range_err.append(abs(float(rq) - float(meas.range_m)))
        per_angles.append(angles)
        per_ranges.append(ranges)
        counts.append(len(angles))
    return per_angles, per_ranges, np.asarray(abs_range_err, dtype=np.float64), counts


def flip_for_plot(grid: np.ndarray) -> np.ndarray:
    return np.flipud(np.asarray(grid))


def draw_occluders(ax, occluders: Sequence[AxisAlignedBox], alpha: float = 0.22, label: bool = False) -> None:
    used = False
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
                label="opaque building / parked vehicle" if label and not used else None,
            )
        )
        used = True


def draw_fov(ax, pose: EgoPose, rng: float = 24.0, alpha: float = 0.15) -> None:
    pts = [(pose.x_m, pose.y_m)]
    for a_deg in (pose.heading_deg - 60.0, pose.heading_deg + 60.0):
        a = np.deg2rad(a_deg)
        pts.append((pose.x_m + rng * np.sin(a), pose.y_m + rng * np.cos(a)))
    ax.add_patch(Polygon(pts, closed=True, facecolor="#64748b", edgecolor="#64748b", alpha=alpha, lw=0.8))


def set_axes(ax, grid_spec: MapGridSpec) -> None:
    ax.set_xlim(-26.0, 35.0)
    ax.set_ylim(-23.0, 55.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def plot_map(ax, grid: np.ndarray, title: str, grid_spec: MapGridSpec, poses: Sequence[EgoPose] | None = None) -> None:
    ax.imshow(
        flip_for_plot(grid),
        origin="lower",
        extent=grid_spec.extent,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    if poses:
        ax.plot([p.x_m for p in poses], [p.y_m for p in poses], "c.-", lw=1.4, ms=3, label="ego route")
    ax.set_title(title, fontsize=10)
    set_axes(ax, grid_spec)
    ax.grid(False)


def scene_inventory(targets: Sequence[WorldTarget]) -> list[dict]:
    rows: list[dict] = []
    for target_type in sorted({t.target_type for t in targets}):
        group = [t for t in targets if t.target_type == target_type]
        xs = np.asarray([t.x_m for t in group], dtype=np.float64)
        ys = np.asarray([t.y_m for t in group], dtype=np.float64)
        rows.append(
            {
                "target_type": target_type,
                "count": len(group),
                "x_min_m": float(xs.min()),
                "x_max_m": float(xs.max()),
                "y_min_m": float(ys.min()),
                "y_max_m": float(ys.max()),
                "dynamic": bool(any(t.is_dynamic for t in group)),
            }
        )
    return rows


def plot_scene_overview(
    targets: Sequence[WorldTarget],
    occluders: Sequence[AxisAlignedBox],
    poses: Sequence[EgoPose],
    gt: np.ndarray,
    grid_spec: MapGridSpec,
    out_dir: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.5), constrained_layout=True)

    ax = axes[0]
    ax.add_patch(Rectangle((-4.0, -25.0), 8.0, 55.0, facecolor="#e2e8f0", edgecolor="none", alpha=0.55, label="ego road"))
    ax.add_patch(Rectangle((-25.0, 20.0), 60.0, 10.0, facecolor="#e2e8f0", edgecolor="none", alpha=0.55, label="cross street"))
    draw_occluders(ax, occluders, alpha=0.36, label=True)

    facade = [t for t in targets if t.target_type.endswith("_facade")]
    if facade:
        ax.scatter([t.x_m for t in facade], [t.y_m for t in facade], s=4, c="#0f172a", alpha=0.40, label="static facade scatterers")
    parked = [t for t in targets if t.target_type.startswith("parked_vehicle")]
    if parked:
        ax.scatter([t.x_m for t in parked], [t.y_m for t in parked], s=5, c="#92400e", alpha=0.55, label="parked-vehicle edge scatterers")
    poles = [t for t in targets if t.target_type == "pole_or_sign"]
    if poles:
        ax.scatter([t.x_m for t in poles], [t.y_m for t in poles], s=45, marker="D", c="#7c3aed", label="poles / signs")

    xs = [p.x_m for p in poses]
    ys = [p.y_m for p in poses]
    ax.plot(xs, ys, "c.-", lw=2.2, ms=4, label="right-lane ego route")
    for idx in [0, min(8, len(poses) - 1), min(15, len(poses) - 1), len(poses) - 1]:
        draw_fov(ax, poses[idx], rng=16.0, alpha=0.16)
    dyn0 = build_p03_cross_traffic_target(0, len(poses))
    dyn1 = build_p03_cross_traffic_target(len(poses) - 1, len(poses))
    ax.plot([dyn0.x_m, dyn1.x_m], [dyn0.y_m, dyn1.y_m], "r--", lw=1.3, alpha=0.75, label="cross-traffic path")
    ax.axvline(0.0, color="#64748b", lw=0.8, ls=":", alpha=0.7)
    ax.text(0.25, -21.5, "road center", fontsize=8, color="#64748b")
    ax.text(xs[0] + 0.35, ys[0] + 2.4, "right-hand\nlane offset", fontsize=8, color="#0891b2")
    ax.set_title("Urban T-intersection with right-hand lane ego route")
    set_axes(ax, grid_spec)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left", fontsize=7.2, framealpha=0.92)

    ax = axes[1]
    ax.imshow(flip_for_plot(gt), origin="lower", extent=grid_spec.extent, cmap="gray", vmin=0, vmax=1, interpolation="nearest", aspect="equal")
    draw_occluders(ax, occluders, alpha=0.16)
    ax.plot(xs, ys, "c.-", lw=2.0, ms=4)
    ax.set_title("Static reference surfaces (dynamic object excluded)")
    set_axes(ax, grid_spec)
    ax.grid(True, alpha=0.18)

    path = out_dir / "p03_resolution_urban_scene_overview.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def make_report(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_spec = MapGridSpec.uniform_square(
        grid_size=args.grid_size,
        x_min_m=args.map_x_min_m,
        x_max_m=args.map_x_max_m,
        y_min_m=args.map_y_min_m,
        y_max_m=args.map_y_max_m,
    )
    targets = build_p03_mapping_targets(
        seed=args.seed,
        wall_spacing_m=args.wall_spacing_m,
        include_dynamic=False,
        include_resolution_probes=args.add_probes,
        scene="urban_intersection",
    )
    occluders = build_p03_urban_occluders(include_parked_vehicles=True)
    poses = generate_p03_urban_ego_trajectory(
        n_steps=args.n_steps,
        dt_s=args.dt_s,
        speed_mps=args.ego_speed_mps,
        start_x_m=args.ego_start_x_m,
        start_y_m=args.ego_start_y_m,
        heading_deg=0.0,
    )
    single_poses = poses[:1]
    gt = occupancy_grid_from_targets(targets, grid_spec=grid_spec, include_dynamic=False, sigma_cells=1.0)
    scene_path = plot_scene_overview(targets, occluders, poses, gt, grid_spec, out_dir)

    bws_mhz = [float(x) for x in args.bw_mhz]
    rows: list[dict] = []
    single_maps: list[np.ndarray] = []
    ego_maps: list[np.ndarray] = []
    frame_counts: dict[str, list[int]] = {}

    for bw_mhz in bws_mhz:
        dr = C / (2.0 * bw_mhz * 1e6)
        single_angles, single_ranges, single_err, single_counts = frame_measurements(
            single_poses, targets, dr, args.radar_max_range_m, occluders, args.occlusion_bin_deg
        )
        ego_angles, ego_ranges, ego_err, ego_counts = frame_measurements(
            poses, targets, dr, args.radar_max_range_m, occluders, args.occlusion_bin_deg
        )
        single_prob, single_bin = accumulate_resolution_probability_map(
            single_poses,
            single_angles,
            single_ranges,
            range_resolution_m=dr,
            grid_spec=grid_spec,
            max_range_m=args.radar_max_range_m,
            beam_width_deg=args.beam_width_deg,
            p_occ=0.65,
            p_free=0.35,
            first_hit_occlusion=True,
            occlusion_bin_deg=args.occlusion_bin_deg,
            free_ray_width_deg=args.free_ray_width_deg,
        )
        ego_prob, ego_bin = accumulate_resolution_probability_map(
            poses,
            ego_angles,
            ego_ranges,
            range_resolution_m=dr,
            grid_spec=grid_spec,
            max_range_m=args.radar_max_range_m,
            beam_width_deg=args.beam_width_deg,
            p_occ=0.65,
            p_free=0.35,
            first_hit_occlusion=True,
            occlusion_bin_deg=args.occlusion_bin_deg,
            free_ray_width_deg=args.free_ray_width_deg,
        )
        sm = map_metrics(gt, single_bin)
        em = map_metrics(gt, ego_bin)
        rows.append(
            {
                "bw_mhz": bw_mhz,
                "range_res_m": dr,
                "single_iou": sm["iou"],
                "single_f1": sm["f1"],
                "ego_iou": em["iou"],
                "ego_f1": em["f1"],
                "single_range_abs_err_mean_m": float(single_err.mean()) if len(single_err) else float("nan"),
                "ego_range_abs_err_mean_m": float(ego_err.mean()) if len(ego_err) else float("nan"),
                "n_single_meas": int(sum(single_counts)),
                "n_ego_meas": int(sum(ego_counts)),
            }
        )
        frame_counts[f"{bw_mhz:g}MHz"] = ego_counts
        single_maps.append(single_prob)
        ego_maps.append(ego_prob)

    fig, axes = plt.subplots(2, len(bws_mhz) + 1, figsize=(3.0 * (len(bws_mhz) + 1), 7.6), constrained_layout=True)
    plot_map(axes[0, 0], gt, "Static reference\nurban scene", grid_spec, single_poses)
    plot_map(axes[1, 0], gt, "Static reference\nright-lane route", grid_spec, poses)
    for j, bw_mhz in enumerate(bws_mhz, start=1):
        dr = rows[j - 1]["range_res_m"]
        plot_map(axes[0, j], single_maps[j - 1], f"Single frame\n{bw_mhz:g} MHz, ΔR={dr:.2f} m", grid_spec, single_poses)
        plot_map(axes[1, j], ego_maps[j - 1], f"Ego-motion map\n{bw_mhz:g} MHz, ΔR={dr:.2f} m", grid_spec, poses)
    fig.suptitle(
        "P03 resolution appendix on the urban T-intersection scene\n"
        f"oracle DoA, first-hit visibility, right-lane ego route x={args.ego_start_x_m:g} m",
        fontsize=14,
    )
    png_path = out_dir / "p03_resolution_single_vs_ego.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    bw = np.asarray([r["bw_mhz"] for r in rows])
    ax.semilogx(bw, [r["single_f1"] for r in rows], "s--", label="single-frame F1")
    ax.semilogx(bw, [r["ego_f1"] for r in rows], "s-", label="ego-motion F1")
    ax.set_xlabel("Bandwidth [MHz]")
    ax.set_ylabel("Map F1 against full static reference")
    ax.set_ylim(0, 1.0)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title("Diagnostic metric only: geometry panels are the lecture focus")
    curve_path = out_dir / "p03_resolution_metrics.png"
    fig2.savefig(curve_path, dpi=180)
    plt.close(fig2)

    inventory = scene_inventory(targets)
    metrics_path = out_dir / "p03_resolution_metrics.json"
    config = vars(args).copy()
    config.update(
        {
            "grid_nx": grid_spec.nx,
            "grid_ny": grid_spec.ny,
            "grid_x_min_m": grid_spec.x_min_m,
            "grid_x_max_m": grid_spec.x_max_m,
            "grid_y_min_m": grid_spec.y_min_m,
            "grid_y_max_m": grid_spec.y_max_m,
            "grid_cell_x_m": grid_spec.cell_x_m,
            "grid_cell_y_m": grid_spec.cell_y_m,
            "grid_square_cell": grid_spec.is_square_cell,
            "occlusion_model": "opaque_boxes + LoS first-hit + unknown behind first hit",
            "frame_counts_by_bandwidth": frame_counts,
        }
    )
    metrics_path.write_text(json.dumps({"rows": rows, "scene_inventory": inventory, "config": config}, indent=2), encoding="utf-8")

    table_rows = "\n".join(
        f"<tr><td>{r['bw_mhz']:g}</td><td>{r['range_res_m']:.3f}</td>"
        f"<td>{r['n_single_meas']}</td><td>{r['n_ego_meas']}</td>"
        f"<td>{r['single_f1']:.3f}</td><td>{r['ego_f1']:.3f}</td>"
        f"<td>{r['ego_range_abs_err_mean_m']:.3f}</td></tr>"
        for r in rows
    )
    scene_table_rows = "\n".join(
        f"<tr><td>{r['target_type']}</td><td>{r['count']}</td>"
        f"<td>{r['x_min_m']:.2f} .. {r['x_max_m']:.2f}</td>"
        f"<td>{r['y_min_m']:.2f} .. {r['y_max_m']:.2f}</td></tr>"
        for r in inventory
    )
    html = f"""<!doctype html>
<html lang=\"ko\"><head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>P03 Resolution Appendix — Urban T-intersection</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 32px; line-height: 1.58; color: #111827; }}
main {{ max-width: 1180px; margin: auto; }}
.note {{ background: #eff6ff; border-left: 4px solid #2563eb; padding: 12px 16px; margin: 18px 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 18px 0; }}
th, td {{ border: 1px solid #d1d5db; padding: 7px 9px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #f3f4f6; }}
img {{ max-width: 100%; border: 1px solid #d1d5db; border-radius: 8px; margin: 12px 0 24px; }}
code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }}
</style></head><body><main>
<h1>P03 Resolution Appendix — Urban T-intersection</h1>
<div class=\"note\"><strong>핵심.</strong> 이 appendix는 DoA를 oracle로 고정하고 bandwidth만 바꾸어 range resolution이 urban radar map 두께에 미치는 영향을 보여준다. Detection은 opaque building/vehicle LoS와 first-hit rule을 통과한 표면만 사용하며, ego route는 우측통행 차선 중심을 반영해 road center가 아니라 x={args.ego_start_x_m:g} m에 둔다.</div>
<ul>
<li>Scene: urban T-intersection, building facades, parked-vehicle edges, poles/signs.</li>
<li>Ego route: {args.n_steps} frames, {args.ego_speed_mps:g} m/s, Δt={args.dt_s:g}s, start=({args.ego_start_x_m:g},{args.ego_start_y_m:g}) m, perfect ego-motion.</li>
<li>Map grid: {grid_spec.nx}×{grid_spec.ny}, x=[{grid_spec.x_min_m:g},{grid_spec.x_max_m:g}] m, y=[{grid_spec.y_min_m:g},{grid_spec.y_max_m:g}] m, cell={grid_spec.cell_x_m:.4f}×{grid_spec.cell_y_m:.4f} m.</li>
<li>Inverse sensor model: first-hit endpoint occupied, ray before endpoint free, behind first-hit/occluded region unknown.</li>
</ul>
<h2>Scene overview</h2>
<img alt=\"P03 urban resolution scene\" src=\"data:image/png;base64,{encode_image(scene_path)}\" />
<h2>Range-resolution map panels</h2>
<img alt=\"P03 urban resolution maps\" src=\"data:image/png;base64,{encode_image(png_path)}\" />
<h2>Scene inventory</h2>
<table><thead><tr><th>Target class</th><th>Count</th><th>x range [m]</th><th>y range [m]</th></tr></thead><tbody>{scene_table_rows}</tbody></table>
<h2>Diagnostic table</h2>
<table><thead><tr><th>BW [MHz]</th><th>ΔR [m]</th><th>Single detections</th><th>Ego detections</th><th>Single F1</th><th>Ego F1</th><th>Mean |range-bin error| [m]</th></tr></thead><tbody>{table_rows}</tbody></table>
<p>정량 metric은 full static reference와의 보조 진단값이다. 강의에서는 좌우로 치우친 ego route에서 single frame footprint가 넓고, ego-motion 누적 후에는 cross-range uncertainty가 줄어들며, 낮은 bandwidth에서는 range 방향 두께가 남는다는 geometry를 중심으로 해석한다.</p>
</main></body></html>"""
    html_path = out_dir / "p03_resolution_report.html"
    html_path.write_text(html, encoding="utf-8")
    return {
        "html": str(html_path),
        "scene": str(scene_path),
        "png": str(png_path),
        "curve": str(curve_path),
        "metrics": str(metrics_path),
        "rows": rows,
        "scene_inventory": inventory,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate P03 urban range-resolution appendix maps")
    parser.add_argument("--out_dir", default=str(BASE / "artifacts" / "resolution_appendix_urban_lane_uniform256"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid_size", type=int, default=256)
    parser.add_argument("--grid_range_m", type=float, default=55.0)
    parser.add_argument("--map_x_min_m", type=float, default=-40.0)
    parser.add_argument("--map_x_max_m", type=float, default=40.0)
    parser.add_argument("--map_y_min_m", type=float, default=-25.0)
    parser.add_argument("--map_y_max_m", type=float, default=55.0)
    parser.add_argument("--radar_max_range_m", type=float, default=45.0)
    parser.add_argument("--n_steps", type=int, default=24)
    parser.add_argument("--dt_s", type=float, default=0.2)
    parser.add_argument("--ego_speed_mps", type=float, default=10.0)
    parser.add_argument("--ego_start_x_m", type=float, default=1.75)
    parser.add_argument("--ego_start_y_m", type=float, default=-20.0)
    parser.add_argument("--beam_width_deg", type=float, default=4.0)
    parser.add_argument("--occlusion_bin_deg", type=float, default=0.25)
    parser.add_argument("--free_ray_width_deg", type=float, default=1.0)
    parser.add_argument("--wall_spacing_m", type=float, default=0.35)
    parser.add_argument("--bw_mhz", type=float, nargs="+", default=[50, 100, 200, 400, 800])
    parser.add_argument("--add_probes", action="store_true", default=False)
    args = parser.parse_args()
    if "--grid_range_m" not in sys.argv:
        args.grid_range_m = float(args.map_y_max_m)
    result = make_report(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
