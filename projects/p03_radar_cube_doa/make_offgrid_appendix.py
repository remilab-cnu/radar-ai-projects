#!/usr/bin/env python3
"""Generate a P03 off-grid / raster-alignment appendix.

This appendix keeps DoA/range oracle and ego pose perfect, then shifts the
continuous scene by sub-cell offsets.  The purpose is not to create a new DoA
benchmark; it demonstrates that OGM/point-grid IoU are finite-raster metrics
that can move with target-to-cell alignment even when the physical point error
is zero.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mapping import (
    EgoPose,
    MapGridSpec,
    WorldTarget,
    accumulate_probability_map,
    build_p03_mapping_targets,
    generate_ego_trajectory,
    map_metrics,
    occupancy_grid_from_targets,
    point_cloud_from_measurements,
    point_cloud_grid,
    visible_measurements,
)

BASE = Path(__file__).resolve().parent


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def fmt(v: object, digits: int = 3) -> str:
    if isinstance(v, (float, int)) and np.isfinite(v):
        return f"{float(v):.{digits}f}"
    return "—"


def shifted_targets(targets: list[WorldTarget], dx_m: float, dy_m: float) -> list[WorldTarget]:
    """Translate all world targets without changing velocities/RCS/classes."""

    return [
        WorldTarget(
            x_m=float(t.x_m + dx_m),
            y_m=float(t.y_m + dy_m),
            vx_mps=float(t.vx_mps),
            vy_mps=float(t.vy_mps),
            rcs_m2=float(t.rcs_m2),
            target_id=int(t.target_id),
            target_type=str(t.target_type),
            is_dynamic=bool(t.is_dynamic),
        )
        for t in targets
    ]


def oracle_frame_lists(
    poses: list[EgoPose],
    targets: list[WorldTarget],
    radar_max_range_m: float,
) -> tuple[list[list[float]], list[list[float]]]:
    per_frame_angles: list[list[float]] = []
    per_frame_ranges: list[list[float]] = []
    for pose in poses:
        measurements = visible_measurements(
            pose,
            targets,
            max_range_m=radar_max_range_m,
            fov_deg=120.0,
            include_dynamic=False,
        )
        per_frame_angles.append([float(m.angle_deg) for m in measurements])
        per_frame_ranges.append([float(m.range_m) for m in measurements])
    return per_frame_angles, per_frame_ranges


def flip_for_plot(grid: np.ndarray) -> np.ndarray:
    return np.flipud(np.asarray(grid))


def plot_grid(ax, grid: np.ndarray, grid_spec: MapGridSpec, title: str, poses: list[EgoPose]) -> None:
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
    ax.plot([p.x_m for p in poses], [p.y_m for p in poses], "c.-", lw=1.2, ms=3)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-12, 14)
    ax.set_ylim(0, 34)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def evaluate_shift(
    base_targets: list[WorldTarget],
    poses: list[EgoPose],
    grid_spec: MapGridSpec,
    dx_m: float,
    dy_m: float,
    radar_max_range_m: float,
    sigma_cells: float,
) -> dict:
    targets = shifted_targets(base_targets, dx_m=dx_m, dy_m=dy_m)
    gt = occupancy_grid_from_targets(
        targets,
        grid_spec=grid_spec,
        include_dynamic=False,
        sigma_cells=sigma_cells,
    )
    per_frame_angles, per_frame_ranges = oracle_frame_lists(poses, targets, radar_max_range_m=radar_max_range_m)
    ogm_prob, ogm_bin = accumulate_probability_map(
        poses,
        per_frame_angles,
        per_frame_ranges,
        grid_spec=grid_spec,
        grid_range_m=grid_spec.y_max_m,
        max_range_m=radar_max_range_m,
        beam_width_deg=5.0,
        p_occ=0.60,
        p_free=0.45,
    )
    points = point_cloud_from_measurements(poses, per_frame_ranges, per_frame_angles)
    pc_grid = point_cloud_grid(points, grid_spec=grid_spec, grid_range_m=grid_spec.y_max_m, sigma_cells=sigma_cells)
    ogm = map_metrics(gt, ogm_bin)
    pc = map_metrics(gt, pc_grid)
    return {
        "dx_m": float(dx_m),
        "dy_m": float(dy_m),
        "n_oracle_detections": int(sum(len(x) for x in per_frame_angles)),
        "gt": gt,
        "ogm_prob": ogm_prob,
        "pc_grid": pc_grid,
        "ogm_iou": ogm["iou"],
        "ogm_f1": ogm["f1"],
        "point_grid_iou": pc["iou"],
        "point_grid_f1": pc["f1"],
        "oracle_point_error_mean_m": 0.0,
    }


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
    base_targets = build_p03_mapping_targets(
        seed=args.seed,
        wall_spacing_m=args.wall_spacing_m,
        include_dynamic=True,
        include_resolution_probes=False,
    )
    poses = generate_ego_trajectory(
        n_steps=args.n_steps,
        dt_s=args.dt_s,
        speed_mps=args.ego_speed_mps,
        start_x_m=0.0,
        start_y_m=0.0,
        heading_deg=0.0,
    )
    shifts = [
        ("reference", 0.0, 0.0),
        ("x half-cell", grid_spec.cell_x_m / 2.0, 0.0),
        ("y half-cell", 0.0, grid_spec.cell_y_m / 2.0),
        ("x/y half-cell", grid_spec.cell_x_m / 2.0, grid_spec.cell_y_m / 2.0),
    ]
    results = []
    for label, dx, dy in shifts:
        row = evaluate_shift(
            base_targets=base_targets,
            poses=poses,
            grid_spec=grid_spec,
            dx_m=dx,
            dy_m=dy,
            radar_max_range_m=args.radar_max_range_m,
            sigma_cells=args.sigma_cells,
        )
        row["label"] = label
        results.append(row)

    fig, axes = plt.subplots(2, len(results), figsize=(3.0 * len(results), 6.8), constrained_layout=True)
    for col, row in enumerate(results):
        title = f"{row['label']}\nshift=({row['dx_m']:.3f},{row['dy_m']:.3f}) m"
        plot_grid(axes[0, col], row["gt"], grid_spec, "GT " + title, poses)
        plot_grid(axes[1, col], row["pc_grid"], grid_spec, "Oracle PC " + title, poses)
    fig.suptitle("P03 off-grid appendix: oracle DoA/range, only target-to-cell alignment changes")
    panel_path = out_dir / "p03_offgrid_maps.png"
    fig.savefig(panel_path, dpi=180)
    plt.close(fig)

    rows = [
        {
            k: v
            for k, v in row.items()
            if k not in {"gt", "ogm_prob", "pc_grid"}
        }
        for row in results
    ]
    metrics = {
        "config": {
            "grid_nx": grid_spec.nx,
            "grid_ny": grid_spec.ny,
            "grid_x_min_m": grid_spec.x_min_m,
            "grid_x_max_m": grid_spec.x_max_m,
            "grid_y_min_m": grid_spec.y_min_m,
            "grid_y_max_m": grid_spec.y_max_m,
            "grid_cell_x_m": grid_spec.cell_x_m,
            "grid_cell_y_m": grid_spec.cell_y_m,
            "wall_spacing_m": args.wall_spacing_m,
            "n_steps": args.n_steps,
            "radar_max_range_m": args.radar_max_range_m,
            "sigma_cells": args.sigma_cells,
        },
        "rows": rows,
    }
    metrics_path = out_dir / "p03_offgrid_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    table_rows = "\n".join(
        "<tr>"
        f"<td>{r['label']}</td>"
        f"<td>{fmt(r['dx_m'], 4)}</td>"
        f"<td>{fmt(r['dy_m'], 4)}</td>"
        f"<td>{r['n_oracle_detections']}</td>"
        f"<td>{fmt(r['point_grid_iou'])}</td>"
        f"<td>{fmt(r['ogm_iou'])}</td>"
        f"<td>{fmt(r['oracle_point_error_mean_m'])}</td>"
        "</tr>"
        for r in rows
    )
    html = f"""<!doctype html>
<html lang="ko"><head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>P03 Off-Grid Raster Appendix</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 32px; color: #111827; line-height: 1.55; }}
main {{ max-width: 1120px; margin: auto; }}
.note {{ background: #eff6ff; border-left: 4px solid #2563eb; padding: 12px 16px; margin: 18px 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 18px 0 28px; }}
th, td {{ border: 1px solid #d1d5db; padding: 7px 9px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #f3f4f6; }}
img {{ max-width: 100%; border: 1px solid #d1d5db; border-radius: 8px; }}
code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }}
</style></head><body><main>
<h1>P03 Off-Grid Raster Appendix</h1>
<div class="note"><strong>Isolation rule.</strong> This appendix uses oracle DoA, oracle range, and perfect ego motion.  The continuous scene is shifted by sub-cell offsets only.  Therefore the physical oracle point error stays zero while raster IoU can move.</div>
<ul>
<li>Map grid: {grid_spec.nx}×{grid_spec.ny}, x=[{grid_spec.x_min_m:g},{grid_spec.x_max_m:g}] m, y=[{grid_spec.y_min_m:g},{grid_spec.y_max_m:g}] m, cell={grid_spec.cell_x_m:.4f}×{grid_spec.cell_y_m:.4f} m.</li>
<li>Scene: dense wall scatterers, wall spacing {args.wall_spacing_m:g} m, {args.n_steps} ego frames.</li>
<li>Lecture use: keep this as a caveat/appendix; do not mix it with the main DoA estimator comparison.</li>
</ul>
<h2>GT and oracle point-cloud grids</h2>
<img alt="P03 off-grid map panels" src="data:image/png;base64,{encode_image(panel_path)}" />
<h2>Raster sensitivity table</h2>
<table><thead><tr><th>Shift profile</th><th>dx [m]</th><th>dy [m]</th><th>Detections</th><th>Point-grid IoU</th><th>OGM IoU</th><th>Oracle point error [m]</th></tr></thead><tbody>{table_rows}</tbody></table>
<p><strong>Interpretation:</strong> if point error is unchanged but grid IoU changes, the change is due to finite rasterization/thresholding.  Use continuous point error and DoA error for primary scientific claims; use IoU as a visual/map-quality teaching metric.</p>
</main></body></html>"""
    html_path = out_dir / "p03_offgrid_appendix.html"
    html_path.write_text(html, encoding="utf-8")
    return {
        "html": str(html_path),
        "panel": str(panel_path),
        "metrics": str(metrics_path),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate P03 off-grid/raster-alignment appendix")
    parser.add_argument("--out_dir", default=str(BASE / "artifacts" / "offgrid_appendix"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid_size", type=int, default=128)
    parser.add_argument("--map_x_min_m", type=float, default=-20.0)
    parser.add_argument("--map_x_max_m", type=float, default=20.0)
    parser.add_argument("--map_y_min_m", type=float, default=0.0)
    parser.add_argument("--map_y_max_m", type=float, default=40.0)
    parser.add_argument("--radar_max_range_m", type=float, default=40.0)
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--dt_s", type=float, default=0.2)
    parser.add_argument("--ego_speed_mps", type=float, default=8.0)
    parser.add_argument("--wall_spacing_m", type=float, default=0.35)
    parser.add_argument("--sigma_cells", type=float, default=1.0)
    args = parser.parse_args()
    print(json.dumps(make_report(args), indent=2))


if __name__ == "__main__":
    main()
