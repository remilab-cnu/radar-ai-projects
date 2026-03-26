"""2D Radar Scene Simulator — 교육용 (DoA + OGM)

에고 차량 시점에서 정적/동적 표적이 있는 2D 환경을 시뮬레이션.
CNN이 공분산 행렬 시퀀스로부터 DoA를 추정하고, 그 결과로 Occupancy Grid Map을
빌드하는 P3 프로젝트의 데이터 생성에 사용.

사용법:
    from shared.radar_scene import RadarScene, polar_to_cartesian, generate_ogm_gt
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from shared.fmcw_simulator import FMCWRadar, generate_scene, range_doppler_map

C = 299_792_458.0  # 광속 [m/s]


# ─── Antenna Element Pattern ───────────────────────────────────────────────────

def patch_element_pattern(theta_deg: float) -> float:
    """Patch antenna element pattern — one-way voltage gain.

    Uses cos(theta) model for E-plane.
    Returns 0 for |theta| > 90 deg (back hemisphere).

    Parameters
    ----------
    theta_deg : float
        Angle from boresight [deg]. 0 = straight ahead.

    Returns
    -------
    gain : float
        One-way voltage gain in [0, 1].
    """
    if abs(theta_deg) > 90.0:
        return 0.0
    return float(np.cos(np.radians(theta_deg)))


# ─── Coordinate Helpers ────────────────────────────────────────────────────────

def polar_to_cartesian(r: float, theta_deg: float):
    """극좌표 → 직교좌표 변환.

    Parameters
    ----------
    r : float
        거리 [m]
    theta_deg : float
        방위각 [deg] — 0°는 정면(y축 방향), 양수는 우측

    Returns
    -------
    x : float  lateral [m]
    y : float  forward [m]
    """
    theta_rad = np.radians(theta_deg)
    x = r * np.sin(theta_rad)
    y = r * np.cos(theta_rad)
    return float(x), float(y)


def cartesian_to_grid(x: float, y: float, grid_size: int, grid_range: float):
    """직교좌표 → OGM 그리드 인덱스 변환.

    OGM 레이아웃:
      - 행(row=0): 최대 거리(전방 끝)
      - 행(row=grid_size-1): 거리 0 (에고 위치)
      - 열(col=0): 좌측 끝 (-grid_range)
      - 열(col=grid_size-1): 우측 끝 (+grid_range)

    Parameters
    ----------
    x : float  lateral [m]  (좌음, 우양)
    y : float  forward [m]  (0 ~ grid_range)
    grid_size : int
    grid_range : float  [m]

    Returns
    -------
    row : int
    col : int
    """
    col = int((x + grid_range) / (2 * grid_range) * grid_size)
    row = int((grid_range - y) / grid_range * grid_size)
    col = int(np.clip(col, 0, grid_size - 1))
    row = int(np.clip(row, 0, grid_size - 1))
    return row, col


def generate_ogm_gt(
    targets: list,
    grid_size: int,
    grid_range: float,
    sigma: float = 1.5,
) -> np.ndarray:
    """표적 목록 → 이진 Occupancy Grid Map.

    Parameters
    ----------
    targets : list of dict
        각 표적: {'range': R [m], 'angle_deg': theta, 'rcs': sigma_rcs}
    grid_size : int
    grid_range : float  [m]
    sigma : float
        표적 확산 커널의 표준편차 [cells]

    Returns
    -------
    ogm : ndarray, shape (grid_size, grid_size), dtype float32
        값 범위 [0, 1]. 표적 위치에 Gaussian blob.
        0.5 이상을 occupied로 간주 (binary threshold 용).
    """
    ogm_raw = np.zeros((grid_size, grid_size), dtype=np.float32)

    # 각 표적을 독립적으로 Gaussian blob 생성 후 max 합성
    # (global normalization은 겹치는 표적이 고립 표적을 억압하는 문제가 있음)
    for tgt in targets:
        x, y = polar_to_cartesian(tgt['range'], tgt['angle_deg'])
        row, col = cartesian_to_grid(x, y, grid_size, grid_range)
        single = np.zeros((grid_size, grid_size), dtype=np.float32)
        single[row, col] = 1.0
        if sigma > 0:
            single = gaussian_filter(single, sigma=sigma)
            peak = single.max()
            if peak > 0:
                single /= peak  # 개별 표적 blob을 [0, 1]로 정규화
        ogm_raw = np.maximum(ogm_raw, single)

    return ogm_raw


# ─── RadarScene ────────────────────────────────────────────────────────────────

class RadarScene:
    """2D 레이다 환경 시뮬레이터.

    에고 차량이 정면을 바라보는 반원형 시야 (angle ∈ [-60°, 60°],
    range ∈ [0, grid_range]) 내 표적 분포를 모델링.
    공분산 행렬 시퀀스와 GT OGM을 함께 생성.

    Parameters
    ----------
    grid_size : int
        OGM 해상도 (grid_size × grid_size 픽셀).
    grid_range : float
        OGM의 물리적 전방 거리 범위 [m].
        가로 범위는 [-grid_range, +grid_range].
    seed : int or None
        전역 랜덤 시드.
    """

    def __init__(self, grid_size: int = 128, grid_range: float = 50.0, seed=None):
        self.grid_size = grid_size
        self.grid_range = grid_range
        self.rng = np.random.default_rng(seed)

    # ── Scene Generation ──────────────────────────────────────────────────────

    def generate_random_scene(
        self,
        n_targets_range: tuple = (3, 12),
        range_limits: tuple = (2, 40),
        angle_limits: tuple = (-60, 60),
        rcs_range: tuple = (0.5, 20),
    ) -> tuple:
        """무작위 표적 배치 및 GT OGM 생성.

        Parameters
        ----------
        n_targets_range : (int, int)
            표적 수 범위 [min, max].
        range_limits : (float, float)
            거리 범위 [m].
        angle_limits : (float, float)
            방위각 범위 [deg].
        rcs_range : (float, float)
            RCS 범위 [m²].

        Returns
        -------
        targets : list of dict
            {'range', 'angle_deg', 'rcs'} 키를 갖는 표적 목록.
        gt_ogm : ndarray, shape (grid_size, grid_size)
            이진 Occupancy Grid (0/1, float32).
        meta : dict
            씬 메타데이터.
        """
        n_tgt = int(self.rng.integers(n_targets_range[0], n_targets_range[1] + 1))

        targets = []
        for _ in range(n_tgt):
            r = float(self.rng.uniform(range_limits[0], range_limits[1]))
            theta = float(self.rng.uniform(angle_limits[0], angle_limits[1]))
            rcs = float(self.rng.uniform(rcs_range[0], rcs_range[1]))
            targets.append({'range': r, 'angle_deg': theta, 'rcs': rcs})

        gt_ogm = self.targets_to_ogm(targets)

        meta = {
            'n_targets': n_tgt,
            'range_limits': range_limits,
            'angle_limits': angle_limits,
        }
        return targets, gt_ogm, meta

    def targets_to_ogm(self, targets: list) -> np.ndarray:
        """표적 목록 → 이진 OGM.

        Parameters
        ----------
        targets : list of dict
            {'range', 'angle_deg'} 키 필수.

        Returns
        -------
        ogm_binary : ndarray, shape (grid_size, grid_size), dtype float32
            Gaussian blob 후 0.5 임계값으로 이진화.
        """
        ogm_soft = generate_ogm_gt(
            targets, self.grid_size, self.grid_range, sigma=1.5
        )
        ogm_binary = (ogm_soft >= 0.5).astype(np.float32)
        return ogm_binary

    # ── Covariance Sequence ───────────────────────────────────────────────────

    def generate_covariance_sequence(
        self,
        targets: list,
        n_frames: int = 20,
        n_rx: int = 8,
        fc: float = 77e9,
        d_lambda: float = 0.5,
        n_snapshots: int = 64,
        snr_db_range: tuple = (-5, 20),
    ) -> tuple:
        """공분산 행렬 시퀀스 생성.

        각 프레임은 동일 표적군을 독립적인 노이즈 실현으로 관측.
        소폭의 에고모션(±2° 방위각 변화)을 선택적으로 적용.

        Parameters
        ----------
        targets : list of dict
            {'range', 'angle_deg', 'rcs'} 키를 갖는 표적 목록.
        n_frames : int
            프레임 수.
        n_rx : int
            수신 안테나 수 (ULA).
        fc : float
            캐리어 주파수 [Hz].
        d_lambda : float
            안테나 간격 / 파장 (기본값: 0.5).
        n_snapshots : int
            프레임당 스냅샷 수 (공분산 추정 품질).
        snr_db_range : (float, float)
            프레임별 SNR 범위 [dB] (프레임마다 독립 샘플링).

        Returns
        -------
        cov_sequence : ndarray, shape (n_frames, 3, n_rx, n_rx)
            3채널: [Re(R_norm), Im(R_norm), log(|R_unnorm| + eps)].
            ch0-1은 Frobenius 정규화된 R_norm, ch2는 원본 R의 log 크기.
            MUSIC은 ch0+j*ch1로 복원 (eigenstructure는 scale-invariant).
            float32.
        frame_meta : list of dict
            프레임별 메타데이터 (snr_db, heading_offset_deg).
        """
        lam = C / fc
        d = d_lambda * lam  # 안테나 간격 [m]
        n = np.arange(n_rx)  # 안테나 인덱스

        cov_sequence = np.zeros((n_frames, 3, n_rx, n_rx), dtype=np.float32)
        frame_meta = []

        for f in range(n_frames):
            # 프레임별 파라미터 샘플링
            snr_db = float(self.rng.uniform(snr_db_range[0], snr_db_range[1]))
            snr_lin = 10 ** (snr_db / 10.0)
            heading_offset = float(self.rng.uniform(-2.0, 2.0))  # 에고모션 [deg]

            # 신호 행렬 X: (n_rx, n_snapshots)
            X = np.zeros((n_rx, n_snapshots), dtype=np.complex128)

            for tgt in targets:
                theta_eff = tgt['angle_deg'] + heading_offset
                rcs = tgt.get('rcs', 1.0)
                amp = np.sqrt(snr_lin * rcs)

                # 스티어링 벡터 a: (n_rx,)
                phase = 2 * np.pi * d / lam * np.sin(np.radians(theta_eff))
                a = np.exp(1j * n * phase)  # (n_rx,)

                # 복소 신호 waveform: (n_snapshots,)
                s = (self.rng.standard_normal(n_snapshots) +
                     1j * self.rng.standard_normal(n_snapshots)) / np.sqrt(2)

                X += amp * np.outer(a, s)

            # 잡음 (단위 분산)
            noise = (self.rng.standard_normal((n_rx, n_snapshots)) +
                     1j * self.rng.standard_normal((n_rx, n_snapshots))) / np.sqrt(2)
            X += noise

            # 샘플 공분산 행렬: R = (1/T) X X^H
            R = (X @ X.conj().T) / n_snapshots  # (n_rx, n_rx)

            # Frobenius norm 정규화
            R_norm = R / (np.linalg.norm(R, 'fro') + 1e-10)

            # 3채널 표현
            log_mag = np.log(np.abs(R) + 1e-10).astype(np.float32)
            cov_sequence[f, 0] = R_norm.real.astype(np.float32)
            cov_sequence[f, 1] = R_norm.imag.astype(np.float32)
            cov_sequence[f, 2] = log_mag

            frame_meta.append({
                'snr_db': snr_db,
                'heading_offset_deg': heading_offset,
            })

        return cov_sequence, frame_meta


# ─── TrajectoryScene ──────────────────────────────────────────────────────────

class TrajectoryScene:
    """Simple 2D scene with ego trajectory for OGM mapping.

    Generates random maps (walls + point targets), ego trajectories,
    per-frame covariance matrices, and GT OGM.  Used for P3 v7:
    the DNN estimates DoA only; OGM is built classically via ISM.

    Coordinate system (world frame):
      - x: lateral (positive = right)
      - y: forward (positive = up/north)
      - Grid: x in [-grid_range, +grid_range], y in [0, grid_range]
      - OGM: row 0 = far (y=grid_range), row (grid_size-1) = near (y=0)

    Parameters
    ----------
    grid_size : int
        OGM resolution (grid_size x grid_size).
    grid_range : float
        Forward range [m].  Lateral spans [-grid_range, +grid_range].
    n_rx : int
        Number of ULA receive antennas.
    d_over_lam : float
        Antenna spacing / wavelength (default 0.5).
    """

    def __init__(
        self,
        grid_size: int = 64,
        grid_range: float = 40.0,
        n_rx: int = 8,
        d_over_lam: float = 0.5,
        seed=None,
    ):
        self.grid_size = grid_size
        self.grid_range = grid_range
        self.n_rx = n_rx
        self.d_over_lam = d_over_lam
        self.rng = np.random.default_rng(seed)

    # ── Map Generation ────────────────────────────────────────────────────

    def generate_map(self, n_walls=None, wall_spacing=0.5):
        """Generate structured 2D map for DoA quality visualization.

        Layout design:

        Ego vehicle drives straight forward along +y axis.
        Radar boresight = 0 deg = straight ahead (+y direction).

        LEFT side (x < 0, negative angles from ego):
            Multiple VERTICAL wall segments — all parallel to the ego
            trajectory (+y axis).  Each wall has a fixed x-coordinate
            and extends along y.  As ego drives forward, each wall sweeps
            from wide to narrow angles, producing extended-target
            reflections at varying negative angles over time.

        RIGHT side (x > 0, positive angles from ego):
            Point targets at various positions.  Some clustered (resolution
            test), some isolated.  Includes wide-angle targets to show the
            patch element pattern gain roll-off.

        The resulting OGM comparison visualises DoA estimator quality across:
            - Close/far range  (range-dependent SNR)
            - Boresight vs wide angle  (patch element pattern roll-off)
            - Point pairs near/at/above Rayleigh limit  (resolution)

        Parameters
        ----------
        n_walls : int or None
            Number of vertical wall segments on the left side (2-4).
            If None, sampled uniformly from [2, 4].
        wall_spacing : float
            Approximate spacing [m] between scatterers along a wall.

        Returns
        -------
        all_targets : list of dict
            Each: {'x': float, 'y': float, 'rcs': float, 'type': str}.
        gt_ogm : ndarray, shape (grid_size, grid_size), float32
            Binary GT OGM (0/1).
        walls : list of dict
            Raw wall definitions for visualisation:
            {'x1', 'y1', 'x2', 'y2'}.
        map_meta : dict
            Scene metadata including angular resolution pair info.
        """
        rng = self.rng
        gr = self.grid_range  # 40 m

        if n_walls is None:
            n_walls = int(rng.integers(2, 5))  # 2, 3, or 4

        all_targets = []
        walls = []

        # ── LEFT SIDE: Vertical walls (parallel to ego trajectory) ────────
        # Each wall has a fixed x-coordinate (negative = left of ego) and
        # extends along y.  All walls have x1 == x2 (pure longitudinal).
        # Different walls at different x-offsets and y-ranges, different lengths.
        # As ego drives forward along +y, each wall enters FOV at a wide angle,
        # sweeps through narrower angles, then exits — producing extended-target
        # reflections that vary from wide to narrow angles over time.
        # Closer walls (small |x|) appear at wider angles; farther walls
        # (large |x|) appear at narrower angles with range attenuation.
        #
        # Fixed wall layout (positions randomised slightly around anchors):
        wall_anchors = [
            # (x_wall, y_start, y_end, rcs_range)  — all x negative
            (-8.0,   5.0,  15.0, (1.0, 4.0)),  # 10 m long, close lateral
            (-15.0, 12.0,  24.0, (1.0, 4.0)),  # 12 m long, mid lateral
            (-5.0,  20.0,  27.0, (1.0, 4.0)),  #  7 m long, very close
            (-20.0,  8.0,  23.0, (1.0, 4.0)),  # 15 m long, far lateral
        ]
        # Use as many anchors as n_walls (cycle if needed)
        used_anchors = [wall_anchors[i % len(wall_anchors)] for i in range(n_walls)]

        for x_w, y_s, y_e, rcs_rng in used_anchors:
            # Small random jitter on x position (±1.5 m) and y endpoints (±2 m)
            x_wall = float(np.clip(x_w + rng.uniform(-1.5, 1.5), -gr + 1.0, -1.0))
            y1 = float(np.clip(y_s + rng.uniform(-2.0, 2.0), 1.0, gr - 2.0))
            y2 = float(np.clip(y_e + rng.uniform(-2.0, 2.0), 1.0, gr - 2.0))
            if y1 > y2:
                y1, y2 = y2, y1  # ensure y1 < y2

            walls.append({'x1': x_wall, 'y1': y1, 'x2': x_wall, 'y2': y2})

            # Discretise wall into scatterer points along y-extent
            wall_len = abs(y2 - y1)
            n_pts = max(2, int(np.ceil(wall_len / wall_spacing)))
            for t in np.linspace(0, 1, n_pts):
                wy = y1 + t * (y2 - y1)
                all_targets.append({
                    'x': float(x_wall),
                    'y': float(wy),
                    'rcs': float(rng.uniform(*rcs_rng)),
                    'type': 'wall',
                })

        # ── RIGHT SIDE: Point targets ──────────────────────────────────────
        # All at positive x (right of ego, positive angles).
        # Mix of:
        #   - Clustered pairs for resolution testing
        #   - Isolated targets at various ranges
        #   - Wide-angle targets (>45 deg) for element pattern demo

        point_targets = []
        map_meta_pairs = []

        def _place_pair(range_m, centre_angle_deg, separation_deg,
                        rcs_range=(3.0, 12.0)):
            """Place a pair of point targets at given range and angular sep."""
            a1 = centre_angle_deg - separation_deg / 2.0
            a2 = centre_angle_deg + separation_deg / 2.0
            for ang in (a1, a2):
                ang_rad = np.radians(ang)
                px = range_m * np.sin(ang_rad)
                py = range_m * np.cos(ang_rad)
                py += float(rng.uniform(-0.5, 0.5))  # small range jitter
                px = float(np.clip(px, 0.5, gr - 0.5))
                py = float(np.clip(py, 1.0, gr - 1.0))
                point_targets.append({
                    'x': px, 'y': py,
                    'rcs': float(rng.uniform(*rcs_range)),
                    'type': 'point',
                })
            return a1, a2

        # Pair 1: sub-Rayleigh (~3 deg) — DNN required to resolve
        r1 = float(rng.uniform(15.0, 25.0))
        c1 = float(rng.uniform(8.0, 18.0))
        _place_pair(r1, c1, 3.0)
        map_meta_pairs.append({
            'label': 'sub-Rayleigh (~3 deg)',
            'separation_deg': 3.0,
            'centre_angle_deg': c1,
            'range_m': r1,
        })

        # Pair 2: near-Rayleigh (~8 deg) — borderline for MUSIC
        r2 = float(rng.uniform(10.0, 20.0))
        c2 = float(rng.uniform(25.0, 38.0))
        _place_pair(r2, c2, 8.0)
        map_meta_pairs.append({
            'label': 'near-Rayleigh (~8 deg)',
            'separation_deg': 8.0,
            'centre_angle_deg': c2,
            'range_m': r2,
        })

        # Pair 3: well-resolved (~15 deg) — all methods should separate
        r3 = float(rng.uniform(8.0, 18.0))
        c3 = float(rng.uniform(10.0, 22.0))
        _place_pair(r3, c3, 15.0)
        map_meta_pairs.append({
            'label': 'well-resolved (~15 deg)',
            'separation_deg': 15.0,
            'centre_angle_deg': c3,
            'range_m': r3,
        })

        # Isolated targets at various ranges (moderate angles)
        n_isolated = int(rng.integers(2, 5))
        for _ in range(n_isolated):
            ang = float(rng.uniform(5.0, 45.0))
            rng_m = float(rng.uniform(5.0, gr * 0.85))
            ang_rad = np.radians(ang)
            px = float(np.clip(rng_m * np.sin(ang_rad), 0.5, gr - 0.5))
            py = float(np.clip(rng_m * np.cos(ang_rad), 1.0, gr - 1.0))
            point_targets.append({
                'x': px, 'y': py,
                'rcs': float(rng.uniform(4.0, 15.0)),
                'type': 'point',
            })

        # Wide-angle targets (>45 deg) — patch element pattern attenuates these
        n_wide = int(rng.integers(1, 3))
        for _ in range(n_wide):
            ang = float(rng.uniform(48.0, 58.0))
            rng_m = float(rng.uniform(5.0, gr * 0.7))
            ang_rad = np.radians(ang)
            px = float(np.clip(rng_m * np.sin(ang_rad), 0.5, gr - 0.5))
            py = float(np.clip(rng_m * np.cos(ang_rad), 1.0, gr - 1.0))
            point_targets.append({
                'x': px, 'y': py,
                'rcs': float(rng.uniform(4.0, 15.0)),
                'type': 'point_wide_angle',
            })

        all_targets.extend(point_targets)

        # GT OGM
        gt_ogm = self._targets_to_ogm(all_targets)

        map_meta = {
            'n_walls': n_walls,
            'n_point_targets': len(point_targets),
            'angular_resolution_pairs': map_meta_pairs,
            'layout': (
                'LEFT=vertical-walls-parallel-to-y(negative-x), '
                'RIGHT=points(angular-resolution-test+wide-angle)'
            ),
        }

        return all_targets, gt_ogm, walls, map_meta

    def _targets_to_ogm(self, targets, sigma=1.0):
        """Convert target list to binary OGM.

        Parameters
        ----------
        targets : list of dict with 'x', 'y' keys.
        sigma : float
            Gaussian blur sigma in cells.

        Returns
        -------
        ogm : ndarray (grid_size, grid_size), float32, values 0/1.
        """
        gs = self.grid_size
        gr = self.grid_range
        ogm_raw = np.zeros((gs, gs), dtype=np.float32)

        for tgt in targets:
            col = int((tgt['x'] + gr) / (2 * gr) * gs)
            row = int((gr - tgt['y']) / gr * gs)
            col = int(np.clip(col, 0, gs - 1))
            row = int(np.clip(row, 0, gs - 1))
            single = np.zeros((gs, gs), dtype=np.float32)
            single[row, col] = 1.0
            if sigma > 0:
                single = gaussian_filter(single, sigma=sigma)
                peak = single.max()
                if peak > 0:
                    single /= peak
            ogm_raw = np.maximum(ogm_raw, single)

        return (ogm_raw >= 0.5).astype(np.float32)

    # ── Trajectory ────────────────────────────────────────────────────────

    def generate_trajectory(self, n_steps=50, step_size=0.5,
                            start_x=0.0, start_y=2.0, heading_deg=0.0):
        """Generate straight-line ego trajectory.

        Parameters
        ----------
        n_steps : int
            Number of positions along trajectory.
        step_size : float
            Distance per step [m].
        start_x, start_y : float
            Starting position [m].
        heading_deg : float
            Direction of travel [deg]. 0 = along +y axis.

        Returns
        -------
        positions : ndarray (n_steps, 2)  — [x, y] per step.
        headings : ndarray (n_steps,)     — heading in degrees.
        """
        heading_rad = np.radians(heading_deg)
        dx = step_size * np.sin(heading_rad)
        dy = step_size * np.cos(heading_rad)

        positions = np.zeros((n_steps, 2), dtype=np.float64)
        for t in range(n_steps):
            positions[t, 0] = start_x + t * dx
            positions[t, 1] = start_y + t * dy

        headings = np.full(n_steps, heading_deg, dtype=np.float64)
        return positions, headings

    # ── Visibility & Covariance ───────────────────────────────────────────

    def get_visible_targets(self, ego_pos, ego_heading_deg, all_targets,
                            max_range=None, fov_deg=120.0):
        """Get targets visible from ego position.

        Parameters
        ----------
        ego_pos : (2,) array  — [x, y].
        ego_heading_deg : float
            Ego heading (0 = +y direction).
        all_targets : list of dict with 'x', 'y', 'rcs'.
        max_range : float or None
            Max detection range [m]. None = grid_range.
        fov_deg : float
            Total field of view [deg] (symmetric around heading).

        Returns
        -------
        visible : list of dict
            Each: {'range': r, 'angle_deg': theta, 'rcs': rcs}.
            angle_deg is relative to ego heading (0=boresight, +right).
        """
        if max_range is None:
            max_range = self.grid_range
        half_fov = fov_deg / 2.0

        visible = []
        ex, ey = ego_pos[0], ego_pos[1]
        head_rad = np.radians(ego_heading_deg)

        for tgt in all_targets:
            dx = tgt['x'] - ex
            dy = tgt['y'] - ey
            r = np.sqrt(dx**2 + dy**2)

            if r < 0.5 or r > max_range:
                continue

            # Angle of target in world frame (0 = +y)
            tgt_angle_world = np.degrees(np.arctan2(dx, dy))
            # Relative angle (positive = right of heading)
            rel_angle = tgt_angle_world - ego_heading_deg
            # Wrap to [-180, 180]
            rel_angle = (rel_angle + 180) % 360 - 180

            if abs(rel_angle) > half_fov:
                continue

            visible.append({
                'range': float(r),
                'angle_deg': float(rel_angle),
                'rcs': tgt.get('rcs', 1.0),
            })

        return visible

    def generate_covariance_at_position(self, visible_targets,
                                        snr_db=20.0, n_snapshots=64,
                                        ref_range=10.0, rcs_ref=1.0):
        """Generate array covariance matrix for one frame.

        Incorporates:
        1. Patch antenna element pattern — cos(theta) one-way voltage gain.
           Two-way power: cos^2(theta).  Targets at wide angles are strongly
           attenuated; targets in the back hemisphere (|theta|>90) are absent.
        2. Range-dependent SNR — radar equation: received power ∝ RCS / R^4,
           so amplitude ∝ sqrt(RCS) / R^2.

        Signal model: X = sum_k amp_k * a(theta_k) * s_k^T + noise
        R = X X^H / T

        Parameters
        ----------
        visible_targets : list of dict with 'angle_deg', 'range', 'rcs'.
        snr_db : float
            Reference SNR [dB] at ref_range for a target with rcs = rcs_ref.
        n_snapshots : int
            Number of snapshots for covariance estimation.
        ref_range : float
            Reference range for SNR definition [m]. Default 10 m.
        rcs_ref : float
            Reference RCS for SNR definition [m^2]. Default 1.0.

        Returns
        -------
        cov_3ch : ndarray (3, n_rx, n_rx), float32
            [Re(R_norm), Im(R_norm), log|R|].
        R_complex : ndarray (n_rx, n_rx), complex128
            Un-normalised sample covariance (for MUSIC).
        """
        rng = self.rng
        n_rx = self.n_rx
        d_lam = self.d_over_lam
        n = np.arange(n_rx)

        snr_ref_lin = 10 ** (snr_db / 10.0)  # SNR at ref_range, rcs_ref

        X = np.zeros((n_rx, n_snapshots), dtype=np.complex128)

        for tgt in visible_targets:
            theta_deg = float(tgt['angle_deg'])
            rcs = float(tgt.get('rcs', 1.0))
            r = float(tgt.get('range', ref_range))

            # ── Patch element pattern (one-way voltage gain) ───────────────
            element_gain = patch_element_pattern(theta_deg)
            if element_gain <= 0.0:
                continue  # target in back hemisphere — invisible

            # ── Range-dependent SNR (radar equation) ──────────────────────
            # received_power ∝ rcs / R^4  →  amplitude ∝ sqrt(rcs) / R^2
            # Normalised to snr_ref_lin at ref_range for rcs_ref:
            snr_lin = snr_ref_lin * (rcs / rcs_ref) * (ref_range / r) ** 4

            # ── Effective voltage amplitude including element pattern ──────
            # Two-way: transmit gain * receive gain = element_gain^2 (power)
            # In voltage: element_gain (one-way) contributes once per path,
            # total two-way voltage amplitude factor = element_gain^2.
            # Combined with sqrt(snr_lin) for base amplitude:
            base_amp = np.sqrt(snr_lin)
            effective_amp = base_amp * (element_gain ** 2)

            # ── Steering vector ────────────────────────────────────────────
            theta_rad = np.radians(theta_deg)
            phase = 2 * np.pi * d_lam * np.sin(theta_rad)
            a = np.exp(1j * n * phase)  # (n_rx,)

            # Complex baseband signal waveform
            s = (rng.standard_normal(n_snapshots) +
                 1j * rng.standard_normal(n_snapshots)) / np.sqrt(2)
            X += effective_amp * np.outer(a, s)

        # Noise (unit variance)
        noise = (rng.standard_normal((n_rx, n_snapshots)) +
                 1j * rng.standard_normal((n_rx, n_snapshots))) / np.sqrt(2)
        X += noise

        # Sample covariance: R = (1/T) X X^H
        R = (X @ X.conj().T) / n_snapshots

        # Frobenius-normalised version (eigenstructure preserved, scale removed)
        R_norm = R / (np.linalg.norm(R, 'fro') + 1e-10)

        cov_3ch = np.zeros((3, n_rx, n_rx), dtype=np.float32)
        cov_3ch[0] = R_norm.real.astype(np.float32)
        cov_3ch[1] = R_norm.imag.astype(np.float32)
        cov_3ch[2] = np.log(np.abs(R) + 1e-10).astype(np.float32)

        return cov_3ch, R

    # ── FMCW Signal Generation ─────────────────────────────────────────────

    def generate_fmcw_signal_at_position(self, visible_targets, radar,
                                         snr_db=15.0, seed=None):
        """Generate full FMCW multi-antenna signal at one ego position.

        Uses FMCWRadar + generate_scene() to create realistic FMCW beat
        signals with proper range (beat frequency) and angle (inter-antenna
        phase).

        Parameters
        ----------
        visible_targets : list of dict
            Each: {'range': r, 'angle_deg': theta, 'rcs': rcs}.
        radar : FMCWRadar
            Radar system object (must have N_rx matching scene's n_rx).
        snr_db : float
            SNR for generate_scene().
        seed : int or None
            Random seed for noise realisation.

        Returns
        -------
        signal : ndarray, shape (N_rx, N_chirps, N_samples), complex
            Raw FMCW beat signal tensor.
        """
        if seed is None:
            seed = int(self.rng.integers(0, 2**31))

        # Convert visible_targets to generate_scene format
        targets_for_scene = []
        for tgt in visible_targets:
            targets_for_scene.append({
                'range': float(tgt['range']),
                'velocity': 0.0,   # static scene for OGM mapping
                'rcs': float(tgt.get('rcs', 1.0)),
                'angle': float(tgt['angle_deg']),
            })

        if len(targets_for_scene) == 0:
            # No visible targets: return noise-only signal
            rng = np.random.default_rng(seed)
            Ns = radar.N_samples
            Nc = radar.N_chirps
            N_rx = radar.N_rx
            noise = (rng.standard_normal((N_rx, Nc, Ns)) +
                     1j * rng.standard_normal((N_rx, Nc, Ns))) / np.sqrt(2)
            return noise

        return generate_scene(radar, targets_for_scene,
                              snr_db=snr_db, seed=seed)

    # ── ISM + Bayesian OGM ────────────────────────────────────────────────

    def ism_update(self, ogm_logodds, ego_pos, ego_heading_deg,
                   detected_angles, detected_ranges=None,
                   beam_width_deg=5.0,
                   max_range=None, p_occ=0.53, p_free=0.3):
        """Inverse Sensor Model: update OGM log-odds from one frame.

        Two modes depending on detected_ranges:

        Bearing-only mode (detected_ranges=None):
            No range information available.  Apply a weak occupied increment
            along the entire beam ray within max_range.  Free-space update is
            skipped — without range we have no knowledge of what lies between
            ego and the target.  Uses p_occ=0.53 (default) to prevent
            over-saturation when only convergent bearings matter.

        Range-aware mode (detected_ranges provided):
            Classic ISM: free-space decrement up to target, occupied increment
            around target range.

        Parameters
        ----------
        ogm_logodds : ndarray (grid_size, grid_size)
        ego_pos : (2,) — [x, y] in world frame.
        ego_heading_deg : float
        detected_angles : array-like of float
            Detected DoA angles [deg] relative to ego heading.
        detected_ranges : array-like of float or None
            Estimated range [m] for each detected angle.
            If None, bearing-only mode is used.
        beam_width_deg : float
            Width of each update beam [deg].
        max_range : float or None
        p_occ : float
            Probability assigned to occupied cells (>0.5).
            Default 0.53 is intentionally weak for bearing-only mode so that
            a single bearing does not saturate the grid; repeated convergent
            bearings from multiple frames accumulate to cross the 0.5 threshold.
        p_free : float
            Probability assigned to free cells (<0.5).
            Default 0.3 gives l_free = log(0.3/0.7) ≈ -0.847, which is
            genuinely below the prior log-odds of -0.405 (prob=0.4).

        Returns
        -------
        ogm_logodds : ndarray (grid_size, grid_size) — updated.
        """
        if max_range is None:
            max_range = self.grid_range
        gs = self.grid_size
        gr = self.grid_range

        # Pre-compute cell centres in world frame
        cell_y = gr / gs
        cell_x = 2.0 * gr / gs
        rows, cols = np.mgrid[0:gs, 0:gs]
        y_centres = (gs - 0.5 - rows) * cell_y
        x_centres = (cols + 0.5 - gs / 2.0) * cell_x

        # Log-odds increments
        l_occ = np.log(p_occ / (1.0 - p_occ))       # positive
        l_free = np.log(p_free / (1.0 - p_free))     # negative (unused in bearing-only)

        ex, ey = ego_pos[0], ego_pos[1]
        head_rad = np.radians(ego_heading_deg)

        bearing_only = (detected_ranges is None)
        detected_angles = list(detected_angles)
        if not bearing_only:
            detected_ranges = list(detected_ranges)

        for i, angle_deg in enumerate(detected_angles):
            # Beam direction in world frame
            beam_rad = head_rad + np.radians(angle_deg)
            beam_dx = np.sin(beam_rad)
            beam_dy = np.cos(beam_rad)

            # Vector from ego to each cell
            dx = x_centres - ex
            dy = y_centres - ey
            r_cell = np.sqrt(dx**2 + dy**2)

            # Angle of each cell relative to beam direction
            cell_angle_world = np.arctan2(dx, dy)
            cell_angle_rel = cell_angle_world - beam_rad
            # Wrap to [-pi, pi]
            cell_angle_rel = (cell_angle_rel + np.pi) % (2 * np.pi) - np.pi

            # Cells within beam cone
            half_bw = np.radians(beam_width_deg / 2.0)
            in_beam = np.abs(cell_angle_rel) < half_bw

            # Cells in front of ego (along beam direction)
            proj = dx * beam_dx + dy * beam_dy
            in_front = proj > 0

            # Range mask: only update within sensor range
            in_range = r_cell <= max_range

            if bearing_only:
                # Bearing-only: apply weak l_occ along entire ray.
                # No free-space update — no range = no knowledge of clearance.
                ray_mask = in_beam & in_front & in_range
                ogm_logodds[ray_mask] += l_occ
            else:
                tgt_range = detected_ranges[i]
                # Occupied band: localised window around detected target range.
                # near=1.0 cell behind, far=1.5 cells ahead — empirically
                # optimal for single-cell Gaussian blobs (sigma=1.0 cell,
                # threshold 0.5).
                cell_size = gr / gs
                occ_near = cell_size * 1.0   # 1 cell behind detected range
                occ_far  = cell_size * 1.5   # 1.5 cells ahead of detected range

                # Occupied: cells in [tgt_range - occ_near, tgt_range + occ_far]
                occ_mask = (in_beam & in_front & in_range
                            & (r_cell >= tgt_range - occ_near)
                            & (r_cell <= tgt_range + occ_far))
                # Free: cells clearly between ego and target (leave a buffer)
                free_mask = (in_beam & in_front & in_range
                             & (r_cell < tgt_range - occ_near))

                ogm_logodds[occ_mask] += l_occ
                ogm_logodds[free_mask] += l_free

        return ogm_logodds

    def build_ogm_from_trajectory(self, positions, headings,
                                  per_frame_angles, per_frame_ranges=None,
                                  **ism_kwargs):
        """Accumulate OGM over full trajectory using Bayesian log-odds.

        Parameters
        ----------
        positions : ndarray (n_steps, 2)
        headings : ndarray (n_steps,)
        per_frame_angles : list of array-like
            Detected angles per frame [deg], relative to ego heading.
        per_frame_ranges : list of array-like or None
            Detected ranges per frame [m], one per angle.
            If None, ism_update falls back to max_range (inaccurate).
        **ism_kwargs : passed to ism_update.

        Returns
        -------
        ogm_prob : ndarray (grid_size, grid_size), float32
            Occupancy probability in [0, 1].
        ogm_binary : ndarray (grid_size, grid_size), float32
            Binary (threshold 0.5).
        """
        gs = self.grid_size
        # Initialise with a weak free-space prior (prob=0.4 → logodds≈-0.405)
        # so cells that are never visited by any beam have prob < 0.5 and are
        # classified as free.  Cells that receive occupied updates will cross
        # above 0.5 only with evidence.
        prior_logodds = np.log(0.4 / 0.6)   # ≈ -0.405
        ogm_logodds = np.full((gs, gs), prior_logodds, dtype=np.float64)

        for t in range(len(positions)):
            if len(per_frame_angles[t]) > 0:
                frame_ranges = (per_frame_ranges[t]
                                if per_frame_ranges is not None else None)
                ogm_logodds = self.ism_update(
                    ogm_logodds,
                    positions[t], headings[t],
                    per_frame_angles[t],
                    detected_ranges=frame_ranges,
                    **ism_kwargs,
                )

        # Clip log-odds to prevent overflow in sigmoid
        ogm_logodds = np.clip(ogm_logodds, -10, 10)
        ogm_prob = (1.0 / (1.0 + np.exp(-ogm_logodds))).astype(np.float32)
        ogm_binary = (ogm_prob >= 0.5).astype(np.float32)

        return ogm_prob, ogm_binary


# ─── Smoke Test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    scene = RadarScene(grid_size=128, grid_range=50.0, seed=0)

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    fig.suptitle('RadarScene Smoke Test — 3 Random Scenes', fontsize=13)

    for i in range(3):
        targets, gt_ogm, meta = scene.generate_random_scene(
            n_targets_range=(3, 8),
            range_limits=(2, 40),
            angle_limits=(-60, 60),
        )

        # --- Left: 극좌표 표적 위치 ---
        ax_polar = axes[i, 0]
        ranges = [t['range'] for t in targets]
        angles = [t['angle_deg'] for t in targets]
        rcs_vals = [t['rcs'] for t in targets]
        xs = [r * np.sin(np.radians(a)) for r, a in zip(ranges, angles)]
        ys = [r * np.cos(np.radians(a)) for r, a in zip(ranges, angles)]
        sc = ax_polar.scatter(xs, ys, c=rcs_vals, cmap='plasma',
                               s=80, vmin=0, vmax=20, edgecolors='k', linewidths=0.5)
        ax_polar.set_xlim(-50, 50)
        ax_polar.set_ylim(0, 50)
        ax_polar.set_xlabel('Lateral x [m]')
        ax_polar.set_ylabel('Forward y [m]')
        ax_polar.set_title(f'Scene {i+1}: {meta["n_targets"]} targets (RCS color)')
        ax_polar.set_aspect('equal')
        fig.colorbar(sc, ax=ax_polar, label='RCS [m²]')

        # --- Right: GT OGM ---
        ax_ogm = axes[i, 1]
        ax_ogm.imshow(
            gt_ogm,
            origin='upper',
            extent=[-50, 50, 0, 50],
            cmap='gray_r',
            vmin=0, vmax=1,
            aspect='auto',
        )
        ax_ogm.set_xlabel('Lateral x [m]')
        ax_ogm.set_ylabel('Forward y [m]')
        ax_ogm.set_title(f'Scene {i+1}: GT OGM (binary)')

        # --- Covariance sequence (print only) ---
        cov_seq, fmeta = scene.generate_covariance_sequence(
            targets, n_frames=20, n_rx=8, n_snapshots=64
        )
        if i == 0:
            print(f'Covariance sequence shape: {cov_seq.shape}')
            print(f'  dtype: {cov_seq.dtype}')
            print(f'  ch0 (Re) range: [{cov_seq[:, 0].min():.3f}, {cov_seq[:, 0].max():.3f}]')
            print(f'  ch1 (Im) range: [{cov_seq[:, 1].min():.3f}, {cov_seq[:, 1].max():.3f}]')
            print(f'  ch2 (log|R|) range: [{cov_seq[:, 2].min():.3f}, {cov_seq[:, 2].max():.3f}]')
            print(f'  frame 0 SNR: {fmeta[0]["snr_db"]:.1f} dB, '
                  f'heading offset: {fmeta[0]["heading_offset_deg"]:.2f} deg')

    plt.tight_layout()
    out_path = '/tmp/radar_scene_smoke.png'
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    print(f'\nPlot saved to {out_path}')
