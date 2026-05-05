"""P03 mapping and ego-motion utilities.

P03 uses these helpers to connect DoA estimation to downstream environmental
perception.  The mainline assumes perfect ego-motion: pose/speed are known and
only DoA estimators differ.  Ego-motion error is therefore modelled as an
appendix perturbation on top of these perfect-pose transforms, not as part of the
main benchmark.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.fmcw_simulator import FMCWRadar, generate_scene, range_doppler_map


@dataclass(frozen=True)
class EgoPose:
    """2D ego pose in the lecture world frame.

    Coordinates follow the Week-12 convention: ``x`` is lateral/right, ``y`` is
    forward/north, and heading ``0 deg`` points along ``+y``.  ``speed_mps`` is
    the known ego speed along the heading direction.
    """

    x_m: float
    y_m: float
    heading_deg: float = 0.0
    speed_mps: float = 0.0


@dataclass(frozen=True)
class WorldTarget:
    """Point-scatterer target in the world frame."""

    x_m: float
    y_m: float
    vx_mps: float = 0.0
    vy_mps: float = 0.0
    rcs_m2: float = 1.0
    target_id: int = 0
    target_type: str = "point"
    is_dynamic: bool = False


@dataclass(frozen=True)
class WallSegment:
    """Straight wall primitive that is rendered as dense radar scatterers."""

    name: str
    x0_m: float
    y0_m: float
    x1_m: float
    y1_m: float


@dataclass(frozen=True)
class AxisAlignedBox:
    """Opaque axis-aligned rectangle used for urban visibility checks."""

    name: str
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float

    def __post_init__(self) -> None:
        if not (self.x_max_m > self.x_min_m and self.y_max_m > self.y_min_m):
            raise ValueError(
                f"AxisAlignedBox bounds must be increasing for {self.name}: "
                f"x=[{self.x_min_m}, {self.x_max_m}], y=[{self.y_min_m}, {self.y_max_m}]"
            )

    def contains_open(self, x_m: float, y_m: float, eps: float = 1e-9) -> bool:
        """Return True only for points strictly inside the opaque body."""

        return bool(
            self.x_min_m + eps < float(x_m) < self.x_max_m - eps
            and self.y_min_m + eps < float(y_m) < self.y_max_m - eps
        )


@dataclass(frozen=True)
class RadarMeasurement:
    """Perfect-pose target measurement used to drive the FMCW simulator."""

    range_m: float
    angle_deg: float
    radial_velocity_mps: float
    world_x_m: float
    world_y_m: float
    target_id: int
    target_type: str
    is_dynamic: bool


@dataclass(frozen=True)
class MapGridSpec:
    """World-frame occupancy-grid bounds with physical cell dimensions.

    P03 originally used one ``grid_range_m`` for both ``x ∈ [-range,+range]``
    and ``y ∈ [0,range]``.  A square ``G×G`` array then had rectangular cells,
    which made map quality and IoU hard to explain in lecture.  This spec keeps
    map bounds explicit so P03 can use physically uniform square cells while
    still loading older legacy datasets.

    Rows are stored in map convention, not image convention:
      * row 0 is ``y_max_m`` (far/forward)
      * row ``ny-1`` is ``y_min_m`` (near/ego side)
      * col 0 is ``x_min_m`` (left)
      * col ``nx-1`` is ``x_max_m`` (right)
    """

    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    nx: int
    ny: int

    def __post_init__(self) -> None:
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError(f"MapGridSpec dimensions must be positive, got nx={self.nx}, ny={self.ny}")
        if not (self.x_max_m > self.x_min_m and self.y_max_m > self.y_min_m):
            raise ValueError(
                "MapGridSpec bounds must be increasing: "
                f"x=[{self.x_min_m}, {self.x_max_m}], y=[{self.y_min_m}, {self.y_max_m}]"
            )

    @classmethod
    def legacy(cls, grid_size: int = 64, grid_range_m: float = 40.0) -> "MapGridSpec":
        """Legacy P03/shared-radar-scene map: x=[-range,+range], y=[0,range]."""

        gr = float(grid_range_m)
        gs = int(grid_size)
        return cls(x_min_m=-gr, x_max_m=gr, y_min_m=0.0, y_max_m=gr, nx=gs, ny=gs)

    @classmethod
    def uniform_square(
        cls,
        grid_size: int = 128,
        x_min_m: float = -20.0,
        x_max_m: float = 20.0,
        y_min_m: float = 0.0,
        y_max_m: float = 40.0,
    ) -> "MapGridSpec":
        """Canonical lecture map: square cells over an explicit world box."""

        return cls(
            x_min_m=float(x_min_m),
            x_max_m=float(x_max_m),
            y_min_m=float(y_min_m),
            y_max_m=float(y_max_m),
            nx=int(grid_size),
            ny=int(grid_size),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.ny), int(self.nx))

    @property
    def width_m(self) -> float:
        return float(self.x_max_m - self.x_min_m)

    @property
    def height_m(self) -> float:
        return float(self.y_max_m - self.y_min_m)

    @property
    def cell_x_m(self) -> float:
        return self.width_m / float(self.nx)

    @property
    def cell_y_m(self) -> float:
        return self.height_m / float(self.ny)

    @property
    def cell_size_m(self) -> float:
        """Representative cell size for isotropic kernels/ISM buffers."""

        return float(min(self.cell_x_m, self.cell_y_m))

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Matplotlib extent: ``[x_min, x_max, y_min, y_max]``."""

        return (float(self.x_min_m), float(self.x_max_m), float(self.y_min_m), float(self.y_max_m))

    @property
    def is_square_cell(self) -> bool:
        return bool(np.isclose(self.cell_x_m, self.cell_y_m, rtol=1e-6, atol=1e-9))

    def contains(self, x_m: float, y_m: float) -> bool:
        return bool(self.x_min_m <= float(x_m) <= self.x_max_m and self.y_min_m <= float(y_m) <= self.y_max_m)


CANONICAL_UNIFORM_GRID = MapGridSpec.uniform_square()
P03_URBAN_GRID = MapGridSpec.uniform_square(
    grid_size=200,
    x_min_m=-40.0,
    x_max_m=40.0,
    y_min_m=-25.0,
    y_max_m=55.0,
)


P03_WALL_SEGMENTS: tuple[WallSegment, ...] = (
    WallSegment("left_wall", -6.0, 6.0, -6.0, 30.0),
    WallSegment("right_wall", 6.0, 8.0, 6.0, 30.0),
    WallSegment("back_wall", -5.0, 31.5, 5.0, 31.5),
)

P03_STATIC_POINT_REFLECTORS: tuple[tuple[float, float], ...] = (
    (-2.5, 14.0),
    (3.5, 18.0),
    (1.5, 26.0),
    (-4.0, 24.0),
)

P03_RESOLUTION_PROBES: tuple[tuple[float, float, float], ...] = (
    # (bearing from ego boresight [deg], near range [m], radial separation [m])
    (18.0, 18.0, 0.5),
    (24.0, 16.0, 1.0),
    (30.0, 20.0, 1.5),
    (38.0, 22.0, 3.0),
)


P03_URBAN_BUILDINGS: tuple[AxisAlignedBox, ...] = (
    AxisAlignedBox("southwest_storefront", -25.0, -6.0, -25.0, 18.0),
    AxisAlignedBox("southeast_storefront", 6.0, 35.0, -25.0, 18.0),
    AxisAlignedBox("northwest_building", -25.0, -6.0, 30.0, 55.0),
    AxisAlignedBox("northeast_building", 6.0, 35.0, 30.0, 55.0),
)

P03_URBAN_PARKED_VEHICLES: tuple[AxisAlignedBox, ...] = (
    AxisAlignedBox("parked_vehicle_left_near", -5.7, -4.3, -11.0, -6.0),
    AxisAlignedBox("parked_vehicle_right_near", 4.3, 5.7, 5.0, 10.2),
    AxisAlignedBox("parked_vehicle_left_cross", -18.0, -12.8, 20.5, 22.5),
    AxisAlignedBox("parked_vehicle_right_cross", 13.0, 18.0, 27.5, 29.5),
)


def wrap_angle_deg(angle_deg: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) to ``[-180, 180)`` degrees."""

    wrapped = (np.asarray(angle_deg) + 180.0) % 360.0 - 180.0
    return float(wrapped) if np.isscalar(angle_deg) else wrapped


def heading_unit_vector(heading_deg: float) -> np.ndarray:
    """Return world-frame unit vector for a heading measured from +y."""

    h = np.deg2rad(float(heading_deg))
    return np.array([np.sin(h), np.cos(h)], dtype=np.float64)


def velocity_from_heading(speed_mps: float, heading_deg: float) -> np.ndarray:
    return float(speed_mps) * heading_unit_vector(heading_deg)


def _segment_box_intersection_interval(
    p0: np.ndarray,
    p1: np.ndarray,
    box: AxisAlignedBox,
    eps: float = 1e-12,
) -> tuple[float, float] | None:
    """Liang-Barsky interval for segment intersection with an axis-aligned box."""

    t0, t1 = 0.0, 1.0
    d = np.asarray(p1, dtype=np.float64) - np.asarray(p0, dtype=np.float64)
    bounds = (
        (float(box.x_min_m), float(box.x_max_m), float(p0[0]), float(d[0])),
        (float(box.y_min_m), float(box.y_max_m), float(p0[1]), float(d[1])),
    )
    for lo, hi, origin, direction in bounds:
        if abs(direction) <= eps:
            if origin < lo or origin > hi:
                return None
            continue
        ta = (lo - origin) / direction
        tb = (hi - origin) / direction
        if ta > tb:
            ta, tb = tb, ta
        t0 = max(t0, ta)
        t1 = min(t1, tb)
        if t0 > t1:
            return None
    return (float(t0), float(t1))


def segment_blocked_by_box(
    start_xy: Sequence[float],
    end_xy: Sequence[float],
    box: AxisAlignedBox,
    eps: float = 1e-6,
) -> bool:
    """Return True when an opaque box intersects the LoS before the endpoint.

    Endpoints are allowed to lie on the visible facade.  A target on the front
    boundary therefore remains visible, but a target on a far boundary or behind
    the building is blocked because the segment enters the rectangle before
    reaching the endpoint.
    """

    p0 = np.asarray(start_xy, dtype=np.float64)
    p1 = np.asarray(end_xy, dtype=np.float64)
    if np.linalg.norm(p1 - p0) <= eps:
        return False
    interval = _segment_box_intersection_interval(p0, p1, box)
    if interval is None:
        return False
    t_enter, t_exit = interval
    # Ignore contact at the radar origin and at the endpoint facade only.
    return bool(t_exit > eps and t_enter < 1.0 - eps)


def is_line_of_sight_clear(
    pose: EgoPose,
    target: WorldTarget,
    occluders: Sequence[AxisAlignedBox] | None,
    eps: float = 1e-6,
) -> bool:
    """Direct-path visibility test from the radar pose to a target scatterer."""

    if not occluders:
        return True
    start = (float(pose.x_m), float(pose.y_m))
    end = (float(target.x_m), float(target.y_m))
    for box in occluders:
        if segment_blocked_by_box(start, end, box, eps=eps):
            return False
    return True


def build_p03_urban_occluders(include_parked_vehicles: bool = True) -> tuple[AxisAlignedBox, ...]:
    """Opaque geometry for the P03 T-intersection scene."""

    if include_parked_vehicles:
        return P03_URBAN_BUILDINGS + P03_URBAN_PARKED_VEHICLES
    return P03_URBAN_BUILDINGS


def measurement_from_world(pose: EgoPose, target: WorldTarget) -> RadarMeasurement | None:
    """Convert a world target into radar-frame range, DoA, and radial velocity.

    The simulator convention is used: positive radial velocity means **closing**
    range.  For a static target straight ahead of an ego vehicle moving at
    ``V`` m/s, the returned velocity is therefore ``+V``.  For a static target at
    bearing ``theta``, it is ``V*cos(theta)``; this is the P03 ego-motion
    correction that was missing from the DoA-only dataset.
    """

    dx = float(target.x_m) - float(pose.x_m)
    dy = float(target.y_m) - float(pose.y_m)
    range_m = float(np.hypot(dx, dy))
    if range_m <= 1e-9:
        return None

    angle_world_deg = float(np.degrees(np.arctan2(dx, dy)))
    rel_angle_deg = float(wrap_angle_deg(angle_world_deg - float(pose.heading_deg)))
    los_unit = np.array([dx, dy], dtype=np.float64) / range_m
    ego_velocity = velocity_from_heading(pose.speed_mps, pose.heading_deg)
    target_velocity = np.array([target.vx_mps, target.vy_mps], dtype=np.float64)
    closing_velocity = float(np.dot(ego_velocity - target_velocity, los_unit))

    return RadarMeasurement(
        range_m=range_m,
        angle_deg=rel_angle_deg,
        radial_velocity_mps=closing_velocity,
        world_x_m=float(target.x_m),
        world_y_m=float(target.y_m),
        target_id=int(target.target_id),
        target_type=str(target.target_type),
        is_dynamic=bool(target.is_dynamic),
    )


def point_from_measurement(pose: EgoPose, range_m: float, angle_deg: float) -> np.ndarray:
    """Project ``(range, DoA)`` back to a world-frame point with perfect pose."""

    world_bearing_deg = float(pose.heading_deg) + float(angle_deg)
    b = np.deg2rad(world_bearing_deg)
    return np.array([
        float(pose.x_m) + float(range_m) * np.sin(b),
        float(pose.y_m) + float(range_m) * np.cos(b),
    ], dtype=np.float64)


def visible_measurements(
    pose: EgoPose,
    targets: Sequence[WorldTarget],
    max_range_m: float,
    fov_deg: float = 120.0,
    include_dynamic: bool = True,
    occluders: Sequence[AxisAlignedBox] | None = None,
    first_hit_occlusion: bool = False,
    occlusion_bin_deg: float = 0.25,
) -> list[RadarMeasurement]:
    """Return direct-path targets inside radar range/FoV for one ego pose."""

    out: list[RadarMeasurement] = []
    half_fov = float(fov_deg) / 2.0
    for target in targets:
        if target.is_dynamic and not include_dynamic:
            continue
        if not is_line_of_sight_clear(pose, target, occluders):
            continue
        meas = measurement_from_world(pose, target)
        if meas is None:
            continue
        if 0.5 <= meas.range_m <= max_range_m and abs(meas.angle_deg) <= half_fov:
            out.append(meas)
    if first_hit_occlusion and out:
        bin_width = max(float(occlusion_bin_deg), 1e-6)
        first_by_bin: dict[int, RadarMeasurement] = {}
        for meas in out:
            bin_idx = int(np.floor((float(meas.angle_deg) + 180.0) / bin_width))
            prev = first_by_bin.get(bin_idx)
            if prev is None or float(meas.range_m) < float(prev.range_m):
                first_by_bin[bin_idx] = meas
        out = sorted(first_by_bin.values(), key=lambda m: (m.range_m, m.angle_deg, m.target_id))
    return out


def classify_measurement_visibility(
    pose: EgoPose,
    targets: Sequence[WorldTarget],
    max_range_m: float,
    fov_deg: float = 120.0,
    include_dynamic: bool = True,
    occluders: Sequence[AxisAlignedBox] | None = None,
) -> dict[str, list[WorldTarget]]:
    """Classify target candidates for lecture visibility figures."""

    visible: list[WorldTarget] = []
    occluded: list[WorldTarget] = []
    out_of_view: list[WorldTarget] = []
    half_fov = float(fov_deg) / 2.0
    for target in targets:
        if target.is_dynamic and not include_dynamic:
            continue
        meas = measurement_from_world(pose, target)
        if meas is None or not (0.5 <= meas.range_m <= max_range_m and abs(meas.angle_deg) <= half_fov):
            out_of_view.append(target)
            continue
        if is_line_of_sight_clear(pose, target, occluders):
            visible.append(target)
        else:
            occluded.append(target)
    return {"visible": visible, "occluded": occluded, "out_of_view": out_of_view}


def _append_resolution_probe_targets(targets: list[WorldTarget]) -> None:
    """Append same-bearing radial point pairs used only by the resolution appendix."""

    next_id = max((t.target_id for t in targets), default=-1) + 1
    for bearing_deg, base_range_m, sep_m in P03_RESOLUTION_PROBES:
        for rr in (base_range_m, base_range_m + sep_m):
            targets.append(WorldTarget(
                x_m=float(rr * np.sin(np.deg2rad(bearing_deg))),
                y_m=float(rr * np.cos(np.deg2rad(bearing_deg))),
                rcs_m2=8.0,
                target_id=next_id,
                target_type=f"resolution_pair_{sep_m:g}m",
            ))
            next_id += 1


def _next_target_id(targets: Sequence[WorldTarget]) -> int:
    return max((int(t.target_id) for t in targets), default=-1) + 1


def _append_segment_targets(
    targets: list[WorldTarget],
    name: str,
    x0_m: float,
    y0_m: float,
    x1_m: float,
    y1_m: float,
    spacing_m: float,
    rng: np.random.Generator,
    rcs_range: tuple[float, float] = (0.8, 3.0),
) -> None:
    """Sample a visible surface segment into dense point scatterers."""

    tid = _next_target_id(targets)
    length = float(np.hypot(float(x1_m) - float(x0_m), float(y1_m) - float(y0_m)))
    n = max(2, int(np.ceil(length / float(spacing_m))) + 1)
    for alpha in np.linspace(0.0, 1.0, n):
        targets.append(WorldTarget(
            x_m=float((1.0 - alpha) * float(x0_m) + alpha * float(x1_m)),
            y_m=float((1.0 - alpha) * float(y0_m) + alpha * float(y1_m)),
            rcs_m2=float(rng.uniform(*rcs_range)),
            target_id=tid,
            target_type=name,
        ))
        tid += 1


def _append_box_edge_targets(
    targets: list[WorldTarget],
    box: AxisAlignedBox,
    spacing_m: float,
    rng: np.random.Generator,
    target_type: str,
    rcs_range: tuple[float, float] = (2.0, 8.0),
) -> None:
    """Sample an extended rectangle perimeter; occluders decide visible faces."""

    _append_segment_targets(targets, target_type, box.x_min_m, box.y_min_m, box.x_max_m, box.y_min_m,
                            spacing_m, rng, rcs_range)
    _append_segment_targets(targets, target_type, box.x_max_m, box.y_min_m, box.x_max_m, box.y_max_m,
                            spacing_m, rng, rcs_range)
    _append_segment_targets(targets, target_type, box.x_max_m, box.y_max_m, box.x_min_m, box.y_max_m,
                            spacing_m, rng, rcs_range)
    _append_segment_targets(targets, target_type, box.x_min_m, box.y_max_m, box.x_min_m, box.y_min_m,
                            spacing_m, rng, rcs_range)


def build_p03_corridor_targets(
    seed: int = 0,
    wall_spacing_m: float = 0.35,
    include_dynamic: bool = True,
    include_resolution_probes: bool = False,
) -> list[WorldTarget]:
    """Legacy corridor scene retained for appendix/backward comparison."""

    rng = np.random.default_rng(seed)
    targets: list[WorldTarget] = []

    for segment in P03_WALL_SEGMENTS:
        _append_segment_targets(
            targets,
            segment.name,
            segment.x0_m,
            segment.y0_m,
            segment.x1_m,
            segment.y1_m,
            wall_spacing_m,
            rng,
            rcs_range=(0.8, 3.0),
        )

    tid = _next_target_id(targets)
    for x, y in P03_STATIC_POINT_REFLECTORS:
        targets.append(WorldTarget(
            x_m=float(x + rng.normal(0.0, 0.15)),
            y_m=float(y + rng.normal(0.0, 0.20)),
            rcs_m2=float(rng.uniform(4.0, 12.0)),
            target_id=tid,
            target_type="static_point",
        ))
        tid += 1

    if include_dynamic:
        targets.append(WorldTarget(
            x_m=2.0,
            y_m=20.0,
            vx_mps=0.0,
            vy_mps=-4.0,
            rcs_m2=8.0,
            target_id=tid,
            target_type="moving_object",
            is_dynamic=True,
        ))

    if include_resolution_probes:
        _append_resolution_probe_targets(targets)
    return targets


def build_p03_urban_intersection_targets(
    seed: int = 0,
    wall_spacing_m: float = 0.35,
    include_dynamic: bool = True,
    include_resolution_probes: bool = False,
) -> list[WorldTarget]:
    """Physically visibility-aware P03 main scene: an urban T-intersection.

    The returned scatterers are candidate surfaces.  Direct-path visibility is
    enforced by ``visible_measurements(..., occluders=build_p03_urban_occluders())``.
    Building and vehicle bodies are opaque; backside/interior scatterers do not
    become measurements unless the ray-casting test permits line of sight.
    """

    rng = np.random.default_rng(seed)
    targets: list[WorldTarget] = []

    # Building facades facing the ego road or cross street.  The occluder test
    # decides which parts are actually visible from each ego pose.
    facades = (
        ("sw_east_facade", -6.0, -25.0, -6.0, 18.0),
        ("se_west_facade", 6.0, -25.0, 6.0, 18.0),
        ("sw_cross_facade", -25.0, 18.0, -6.0, 18.0),
        ("se_cross_facade", 6.0, 18.0, 35.0, 18.0),
        ("nw_east_facade", -6.0, 30.0, -6.0, 55.0),
        ("ne_west_facade", 6.0, 30.0, 6.0, 55.0),
        ("nw_cross_facade", -25.0, 30.0, -6.0, 30.0),
        ("ne_cross_facade", 6.0, 30.0, 35.0, 30.0),
    )
    for name, x0, y0, x1, y1 in facades:
        _append_segment_targets(targets, name, x0, y0, x1, y1, wall_spacing_m, rng, rcs_range=(0.8, 3.2))

    for box in P03_URBAN_PARKED_VEHICLES:
        _append_box_edge_targets(
            targets,
            box,
            spacing_m=max(float(wall_spacing_m), 0.40),
            rng=rng,
            target_type=box.name,
            rcs_range=(3.0, 12.0),
        )

    tid = _next_target_id(targets)
    for x, y in [(-4.2, -1.5), (4.4, 15.5), (-4.5, 24.0), (4.5, 24.8), (0.6, 31.0)]:
        targets.append(WorldTarget(
            x_m=float(x + rng.normal(0.0, 0.05)),
            y_m=float(y + rng.normal(0.0, 0.08)),
            rcs_m2=float(rng.uniform(5.0, 14.0)),
            target_id=tid,
            target_type="pole_or_sign",
        ))
        tid += 1

    if include_dynamic:
        # A cross-traffic vehicle snapshot.  Lecture animations should move this
        # with ``build_p03_cross_traffic_target``; this static instance preserves
        # the existing include_dynamic contract for datasets.
        targets.append(WorldTarget(
            x_m=-18.0,
            y_m=25.0,
            vx_mps=10.0,
            vy_mps=0.0,
            rcs_m2=10.0,
            target_id=tid,
            target_type="cross_traffic_vehicle",
            is_dynamic=True,
        ))

    if include_resolution_probes:
        _append_resolution_probe_targets(targets)
    return targets


def build_p03_cross_traffic_target(
    frame_idx: int,
    n_frames: int,
    target_id: int = 100_000,
    speed_mps: float = 10.0,
) -> WorldTarget:
    """Frame-indexed dynamic vehicle for lecture visibility animations."""

    denom = max(int(n_frames) - 1, 1)
    alpha = float(np.clip(int(frame_idx) / denom, 0.0, 1.0))
    x = -20.0 + alpha * 45.0
    return WorldTarget(
        x_m=float(x),
        y_m=25.0,
        vx_mps=float(speed_mps),
        vy_mps=0.0,
        rcs_m2=10.0,
        target_id=int(target_id),
        target_type="cross_traffic_vehicle",
        is_dynamic=True,
    )


def build_p03_mapping_targets(
    seed: int = 0,
    wall_spacing_m: float = 0.35,
    include_dynamic: bool = True,
    include_resolution_probes: bool = False,
    scene: str = "urban_intersection",
) -> list[WorldTarget]:
    """Create the P03 lecture mapping scene.

    Default P03 uses a physically gated urban T-intersection.  A corridor scene
    can also be requested with ``scene="corridor"`` for compact demonstrations.
    """

    if scene == "urban_intersection":
        return build_p03_urban_intersection_targets(
            seed=seed,
            wall_spacing_m=wall_spacing_m,
            include_dynamic=include_dynamic,
            include_resolution_probes=include_resolution_probes,
        )
    if scene == "corridor":
        return build_p03_corridor_targets(
            seed=seed,
            wall_spacing_m=wall_spacing_m,
            include_dynamic=include_dynamic,
            include_resolution_probes=include_resolution_probes,
        )
    raise ValueError(f"Unknown P03 mapping scene: {scene!r}")


def build_lecture_mapping_targets(seed: int = 0, wall_spacing_m: float = 0.35) -> list[WorldTarget]:
    """Backward-compatible alias for the main P03 urban-intersection scene."""

    return build_p03_mapping_targets(
        seed=seed,
        wall_spacing_m=wall_spacing_m,
        include_dynamic=True,
        include_resolution_probes=False,
        scene="urban_intersection",
    )


def generate_ego_trajectory(
    n_steps: int = 8,
    dt_s: float = 0.2,
    speed_mps: float = 8.0,
    start_x_m: float = 0.0,
    start_y_m: float = 0.0,
    heading_deg: float = 0.0,
) -> list[EgoPose]:
    """Straight ego trajectory with known, error-free ego-motion."""

    step = float(speed_mps) * float(dt_s)
    direction = heading_unit_vector(heading_deg)
    poses: list[EgoPose] = []
    for i in range(int(n_steps)):
        offset = i * step * direction
        poses.append(EgoPose(
            x_m=float(start_x_m + offset[0]),
            y_m=float(start_y_m + offset[1]),
            heading_deg=float(heading_deg),
            speed_mps=float(speed_mps),
        ))
    return poses


def generate_p03_urban_ego_trajectory(
    n_steps: int = 24,
    dt_s: float = 0.20,
    speed_mps: float = 10.0,
    start_x_m: float = 1.75,
    start_y_m: float = -20.0,
    heading_deg: float = 0.0,
) -> list[EgoPose]:
    """Straight, perfect-pose T-intersection approach used by P03 mainline.

    The ego road spans roughly ``x ∈ [-4, 4]``.  For a northbound vehicle in a
    right-hand-traffic setting, the lane centre is on the positive-x side of
    the road centreline; the default ``x=1.75 m`` reflects that asymmetric lane
    placement instead of driving unrealistically on the road centreline.
    """

    return generate_ego_trajectory(
        n_steps=n_steps,
        dt_s=dt_s,
        speed_mps=speed_mps,
        start_x_m=start_x_m,
        start_y_m=start_y_m,
        heading_deg=heading_deg,
    )


def normalize_antenna_vector(ant_vec: np.ndarray) -> np.ndarray:
    ant_vec = np.asarray(ant_vec, dtype=np.complex64)
    return (ant_vec / (np.sqrt(np.mean(np.abs(ant_vec) ** 2)) + 1e-12)).astype(np.complex64)


def simulate_rd_selected_vector(
    radar: FMCWRadar,
    measurement: RadarMeasurement,
    snr_db: float,
    rcs_m2: float,
    seed: int,
    normalize: bool = True,
) -> tuple[np.ndarray, dict]:
    """Simulate one measurement and extract its RD-selected antenna vector."""

    target = {
        "range": float(measurement.range_m),
        "velocity": float(measurement.radial_velocity_mps),
        "angle": float(measurement.angle_deg),
        "rcs": float(rcs_m2),
    }
    raw, scene_meta = generate_scene(radar, [target], snr_db=snr_db, seed=seed, return_meta=True)
    rd_cube = range_doppler_map(raw, radar=radar, window_range="hann", window_doppler="hann").astype(np.complex64)
    sim_info = scene_meta["target_info"][0]
    ant_vec = rd_cube[:, int(sim_info["doppler_bin"]), int(sim_info["range_bin"])]
    if normalize:
        ant_vec = normalize_antenna_vector(ant_vec)
    meta = {
        "r_bin": int(sim_info["range_bin"]),
        "d_bin": int(sim_info["doppler_bin"]),
        "actual_snr_db": float(sim_info["actual_snr_db"]),
        "sample_snr_db": float(sim_info["sample_snr_db"]),
        "fs_over_bandwidth": float(scene_meta["fs_over_bandwidth"]),
    }
    return ant_vec.astype(np.complex64), meta


def coerce_map_grid_spec(
    grid_spec: MapGridSpec | None = None,
    grid_size: int = 64,
    grid_range_m: float = 40.0,
) -> MapGridSpec:
    """Return an explicit grid spec, defaulting to the legacy P03 convention."""

    return grid_spec if grid_spec is not None else MapGridSpec.legacy(grid_size=grid_size, grid_range_m=grid_range_m)


def map_grid_centres(grid_spec: MapGridSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x_centres, y_centres)`` arrays with shape ``grid_spec.shape``."""

    rows, cols = np.mgrid[0:grid_spec.ny, 0:grid_spec.nx]
    x_centres = float(grid_spec.x_min_m) + (cols + 0.5) * grid_spec.cell_x_m
    y_centres = float(grid_spec.y_max_m) - (rows + 0.5) * grid_spec.cell_y_m
    return x_centres.astype(np.float64), y_centres.astype(np.float64)


def world_to_grid(x_m: float, y_m: float, grid_size: int, grid_range_m: float) -> tuple[int, int]:
    """Legacy world-to-grid transform for x=[-range,+range], y=[0,range]."""

    col = int((float(x_m) + float(grid_range_m)) / (2.0 * float(grid_range_m)) * int(grid_size))
    row = int((float(grid_range_m) - float(y_m)) / float(grid_range_m) * int(grid_size))
    return int(np.clip(row, 0, grid_size - 1)), int(np.clip(col, 0, grid_size - 1))


def world_to_grid_spec(x_m: float, y_m: float, grid_spec: MapGridSpec) -> tuple[int, int]:
    """World-to-grid transform for an explicit ``MapGridSpec``."""

    col = int((float(x_m) - float(grid_spec.x_min_m)) / grid_spec.width_m * int(grid_spec.nx))
    row = int((float(grid_spec.y_max_m) - float(y_m)) / grid_spec.height_m * int(grid_spec.ny))
    return int(np.clip(row, 0, grid_spec.ny - 1)), int(np.clip(col, 0, grid_spec.nx - 1))


def occupancy_grid_from_targets(
    targets: Sequence[WorldTarget],
    grid_size: int = 64,
    grid_range_m: float = 40.0,
    include_dynamic: bool = False,
    sigma_cells: float = 1.0,
    grid_spec: MapGridSpec | None = None,
) -> np.ndarray:
    """Build binary GT OGM from world targets."""

    spec = coerce_map_grid_spec(grid_spec, grid_size=grid_size, grid_range_m=grid_range_m)
    grid = np.zeros(spec.shape, dtype=np.float32)
    for target in targets:
        if target.is_dynamic and not include_dynamic:
            continue
        if not spec.contains(target.x_m, target.y_m):
            continue
        row, col = world_to_grid_spec(target.x_m, target.y_m, spec)
        grid[row, col] = 1.0
    if sigma_cells > 0:
        soft = gaussian_filter(grid, sigma=sigma_cells)
        if soft.max() > 0:
            soft = soft / soft.max()
        grid = (soft >= 0.5).astype(np.float32)
    return grid.astype(np.float32)


def _accumulate_logodds_map(
    poses: Sequence[EgoPose],
    per_frame_angles: Sequence[Sequence[float]],
    per_frame_ranges: Sequence[Sequence[float]] | None,
    grid_spec: MapGridSpec,
    max_range_m: float,
    beam_width_deg: float,
    p_occ: float,
    p_free: float,
    occ_near_m: float,
    occ_far_m: float,
    first_hit_occlusion: bool = False,
    occlusion_bin_deg: float = 0.25,
    free_ray_width_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared range-aware inverse-sensor-model update for explicit map bounds."""

    x_centres, y_centres = map_grid_centres(grid_spec)
    prior_logodds = np.log(0.4 / 0.6)
    l_occ = np.log(float(p_occ) / (1.0 - float(p_occ)))
    l_free = np.log(float(p_free) / (1.0 - float(p_free)))
    ogm_logodds = np.full(grid_spec.shape, prior_logodds, dtype=np.float64)

    for frame_idx, (pose, frame_angles) in enumerate(zip(poses, per_frame_angles)):
        if len(frame_angles) == 0:
            continue
        frame_ranges = None if per_frame_ranges is None else list(per_frame_ranges[frame_idx])
        first_hit_indices: set[int] | None = None
        if first_hit_occlusion and frame_ranges is not None:
            first_by_bin: dict[int, int] = {}
            bin_width = max(float(occlusion_bin_deg), 1e-6)
            for i, (angle_deg, range_m) in enumerate(zip(frame_angles, frame_ranges)):
                bin_idx = int(np.floor((float(angle_deg) + 180.0) / bin_width))
                prev = first_by_bin.get(bin_idx)
                if prev is None or float(range_m) < float(frame_ranges[prev]):
                    first_by_bin[bin_idx] = i
            first_hit_indices = set(first_by_bin.values())
        free_half_bw = np.radians(
            (float(free_ray_width_deg) if free_ray_width_deg is not None else float(beam_width_deg)) / 2.0
        )
        ex, ey = float(pose.x_m), float(pose.y_m)
        head_rad = np.radians(float(pose.heading_deg))
        dx = x_centres - ex
        dy = y_centres - ey
        r_cell = np.sqrt(dx**2 + dy**2)

        for i, angle_deg in enumerate(frame_angles):
            beam_rad = head_rad + np.radians(float(angle_deg))
            beam_dx = np.sin(beam_rad)
            beam_dy = np.cos(beam_rad)
            cell_angle_world = np.arctan2(dx, dy)
            cell_angle_rel = (cell_angle_world - beam_rad + np.pi) % (2 * np.pi) - np.pi
            in_beam = np.abs(cell_angle_rel) < np.radians(float(beam_width_deg) / 2.0)
            proj = dx * beam_dx + dy * beam_dy
            in_front = proj > 0
            in_range = r_cell <= float(max_range_m)

            if frame_ranges is None:
                ray_mask = in_beam & in_front & in_range
                ogm_logodds[ray_mask] += l_occ
                continue

            r0 = float(frame_ranges[i])
            occ_mask = (
                in_beam
                & in_front
                & in_range
                & (r_cell >= r0 - float(occ_near_m))
                & (r_cell <= r0 + float(occ_far_m))
            )
            free_mask = in_beam & in_front & in_range & (r_cell < r0 - float(occ_near_m))
            if first_hit_indices is not None:
                if i not in first_hit_indices:
                    occ_mask = np.zeros_like(occ_mask, dtype=bool)
                    free_mask = np.zeros_like(free_mask, dtype=bool)
                else:
                    free_mask = (
                        (np.abs(cell_angle_rel) < free_half_bw)
                        & in_front
                        & in_range
                        & (r_cell < r0 - float(occ_near_m))
                    )

            ogm_logodds[occ_mask] += l_occ
            ogm_logodds[free_mask] += l_free

    ogm_logodds = np.clip(ogm_logodds, -10, 10)
    ogm_prob = (1.0 / (1.0 + np.exp(-ogm_logodds))).astype(np.float32)
    ogm_binary = (ogm_prob >= 0.5).astype(np.float32)
    return ogm_prob, ogm_binary


def accumulate_probability_map(
    poses: Sequence[EgoPose],
    per_frame_angles: Sequence[Sequence[float]],
    per_frame_ranges: Sequence[Sequence[float]],
    grid_size: int = 64,
    grid_range_m: float = 40.0,
    grid_spec: MapGridSpec | None = None,
    max_range_m: float | None = None,
    beam_width_deg: float = 5.0,
    p_occ: float = 0.53,
    p_free: float = 0.3,
    first_hit_occlusion: bool = False,
    occlusion_bin_deg: float = 0.25,
    free_ray_width_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate a Bayesian OGM from range+DoA detections."""

    spec = coerce_map_grid_spec(grid_spec, grid_size=grid_size, grid_range_m=grid_range_m)
    cell_size = spec.cell_size_m
    return _accumulate_logodds_map(
        poses=poses,
        per_frame_angles=per_frame_angles,
        per_frame_ranges=per_frame_ranges,
        grid_spec=spec,
        max_range_m=float(max_range_m) if max_range_m is not None else float(grid_range_m),
        beam_width_deg=beam_width_deg,
        p_occ=p_occ,
        p_free=p_free,
        occ_near_m=cell_size * 1.0,
        occ_far_m=cell_size * 1.5,
        first_hit_occlusion=first_hit_occlusion,
        occlusion_bin_deg=occlusion_bin_deg,
        free_ray_width_deg=free_ray_width_deg,
    )


def accumulate_resolution_probability_map(
    poses: Sequence[EgoPose],
    per_frame_angles: Sequence[Sequence[float]],
    per_frame_ranges: Sequence[Sequence[float]],
    range_resolution_m: float,
    grid_size: int = 64,
    grid_range_m: float = 40.0,
    grid_spec: MapGridSpec | None = None,
    max_range_m: float | None = None,
    beam_width_deg: float = 4.0,
    p_occ: float = 0.65,
    p_free: float = 0.35,
    first_hit_occlusion: bool = False,
    occlusion_bin_deg: float = 0.25,
    free_ray_width_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate an OGM while explicitly modelling range-cell thickness.

    This helper is for the P03 resolution appendix.  Unlike the main DoA
    comparison path, it should be fed range-bin-centre or range-likelihood
    measurements, not simulator-exact ranges.  The occupied band around each
    endpoint is widened to roughly one range-resolution cell so low-bandwidth
    radars produce thick walls even with oracle DoA.
    """

    spec = coerce_map_grid_spec(grid_spec, grid_size=grid_size, grid_range_m=grid_range_m)
    cell_size = spec.cell_size_m
    half_range = max(float(range_resolution_m) / 2.0, cell_size / 2.0)

    return _accumulate_logodds_map(
        poses=poses,
        per_frame_angles=per_frame_angles,
        per_frame_ranges=per_frame_ranges,
        grid_spec=spec,
        max_range_m=float(max_range_m) if max_range_m is not None else float(grid_range_m),
        beam_width_deg=beam_width_deg,
        p_occ=p_occ,
        p_free=p_free,
        occ_near_m=half_range,
        occ_far_m=half_range,
        first_hit_occlusion=first_hit_occlusion,
        occlusion_bin_deg=occlusion_bin_deg,
        free_ray_width_deg=free_ray_width_deg,
    )


def point_cloud_from_measurements(
    poses: Sequence[EgoPose],
    per_frame_ranges: Sequence[Sequence[float]],
    per_frame_angles: Sequence[Sequence[float]],
) -> np.ndarray:
    points: list[np.ndarray] = []
    for pose, ranges, angles in zip(poses, per_frame_ranges, per_frame_angles):
        for r, a in zip(ranges, angles):
            points.append(point_from_measurement(pose, float(r), float(a)))
    if not points:
        return np.zeros((0, 2), dtype=np.float64)
    return np.vstack(points)


def point_cloud_grid(
    points_xy: np.ndarray,
    grid_size: int = 64,
    grid_range_m: float = 40.0,
    sigma_cells: float = 1.0,
    grid_spec: MapGridSpec | None = None,
) -> np.ndarray:
    spec = coerce_map_grid_spec(grid_spec, grid_size=grid_size, grid_range_m=grid_range_m)
    grid = np.zeros(spec.shape, dtype=np.float32)
    for x, y in np.asarray(points_xy):
        if spec.contains(float(x), float(y)):
            row, col = world_to_grid_spec(float(x), float(y), spec)
            grid[row, col] = 1.0
    if sigma_cells > 0 and grid.max() > 0:
        soft = gaussian_filter(grid, sigma=sigma_cells)
        grid = (soft / (soft.max() + 1e-12) >= 0.5).astype(np.float32)
    return grid


def localization_errors_m(range_m: np.ndarray, true_angle_deg: np.ndarray, pred_angle_deg: np.ndarray) -> np.ndarray:
    """Euclidean point-cloud error caused by DoA error at known range."""

    r = np.asarray(range_m, dtype=np.float64)
    a_true = np.deg2rad(np.asarray(true_angle_deg, dtype=np.float64))
    a_pred = np.deg2rad(np.asarray(pred_angle_deg, dtype=np.float64))
    true_xy = np.stack([r * np.sin(a_true), r * np.cos(a_true)], axis=-1)
    pred_xy = np.stack([r * np.sin(a_pred), r * np.cos(a_pred)], axis=-1)
    return np.linalg.norm(pred_xy - true_xy, axis=-1)


def map_metrics(gt_binary: np.ndarray, pred_binary: np.ndarray) -> dict[str, float]:
    gt = np.asarray(gt_binary).astype(bool)
    pred = np.asarray(pred_binary).astype(bool)
    tp = int(np.count_nonzero(gt & pred))
    fp = int(np.count_nonzero(~gt & pred))
    fn = int(np.count_nonzero(gt & ~pred))
    union = tp + fp + fn
    iou = float(tp / union) if union else 1.0
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def yaw_biased_angles(angle_deg: Iterable[float], yaw_bias_deg: float) -> list[float]:
    """Appendix helper: emulate an ego-heading error as a DoA-frame bias."""

    return [float(wrap_angle_deg(float(a) + float(yaw_bias_deg))) for a in angle_deg]


def perturb_ego_poses(
    poses: Sequence[EgoPose],
    dx_m: float = 0.0,
    dy_m: float = 0.0,
    yaw_bias_deg: float = 0.0,
    speed_bias_mps: float = 0.0,
    drift_per_step_m: tuple[float, float] = (0.0, 0.0),
) -> list[EgoPose]:
    """Appendix helper for ego-motion-error-only map studies.

    Use this with GT DoA/range detections when studying the P03 appendix.  The
    mainline should pass unperturbed poses so map differences are attributable
    to DoA, not odometry.
    """

    out: list[EgoPose] = []
    drift_x, drift_y = drift_per_step_m
    for i, pose in enumerate(poses):
        out.append(EgoPose(
            x_m=float(pose.x_m + dx_m + i * drift_x),
            y_m=float(pose.y_m + dy_m + i * drift_y),
            heading_deg=float(pose.heading_deg + yaw_bias_deg),
            speed_mps=float(pose.speed_mps + speed_bias_mps),
        ))
    return out
