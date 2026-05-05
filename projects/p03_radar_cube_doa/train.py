#!/usr/bin/env python3
"""P03 -- Train/evaluate DoA network and downstream radar maps.

Usage:
  python train.py --mapping --generate --smoke
  python train.py --mapping --generate --epochs 30
  python train.py --generate --smoke          # retained DoA-unit lane
  python train.py --eval_only --checkpoint artifacts/best_model.pt
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.cli import base_parser
from common.hdf5_io import load_hdf5
from common.seed import seed_everything
from common.train_utils import count_parameters
from mapping import (
    EgoPose,
    MapGridSpec,
    accumulate_probability_map,
    localization_errors_m,
    map_metrics,
    point_cloud_from_measurements,
    point_cloud_grid,
)
from model import build_model

BASE = Path(__file__).parent
ANGLE_GRID = np.linspace(-90.0, 90.0, 181, dtype=np.float32)
D_OVER_LAM = 0.5


class DoAVectorDataset(Dataset):
    def __init__(self, path: Path):
        data = load_hdf5(path)
        self.x = torch.as_tensor(data["x_ant"], dtype=torch.float32)
        self.y = torch.as_tensor(data["y_spectrum"], dtype=torch.float32)
        self.angle_deg = data["angle_deg"].astype(np.float32)
        self.snr_db = data["snr_db"].astype(np.float32)
        self.range_m = data["range_m"].astype(np.float32)
        self.velocity_mps = data["velocity_mps"].astype(np.float32)
        self.r_bin = data["r_bin"].astype(np.int32)
        self.d_bin = data["d_bin"].astype(np.int32)
        self.n_targets = data["n_targets"].astype(np.int32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class MappingDetectionDataset(DoAVectorDataset):
    """Scenario-derived per-detection DoA data plus scene-level map labels."""

    def __init__(self, path: Path):
        super().__init__(path)
        data = load_hdf5(path)
        self.scene_idx = data["scene_idx"].astype(np.int32)
        self.frame_idx = data["frame_idx"].astype(np.int32)
        self.target_id = data["target_id"].astype(np.int32)
        self.is_dynamic = data["is_dynamic"].astype(bool)
        self.poses = data["poses"].astype(np.float32)  # (S, T, [x,y,heading,speed])
        self.gt_ogm = data["gt_ogm"].astype(np.float32)
        self.grid_size = int(data["grid_size"][0])
        self.grid_range_m = float(data["grid_range_m"][0])
        self.grid_nx = int(data["grid_nx"][0]) if "grid_nx" in data else self.grid_size
        self.grid_ny = int(data["grid_ny"][0]) if "grid_ny" in data else self.grid_size
        if "grid_x_min_m" in data:
            self.grid_spec = MapGridSpec(
                x_min_m=float(data["grid_x_min_m"][0]),
                x_max_m=float(data["grid_x_max_m"][0]),
                y_min_m=float(data["grid_y_min_m"][0]),
                y_max_m=float(data["grid_y_max_m"][0]),
                nx=self.grid_nx,
                ny=self.grid_ny,
            )
        else:
            self.grid_spec = MapGridSpec.legacy(grid_size=self.grid_size, grid_range_m=self.grid_range_m)
        self.grid_cell_x_m = float(data["grid_cell_x_m"][0]) if "grid_cell_x_m" in data else self.grid_spec.cell_x_m
        self.grid_cell_y_m = float(data["grid_cell_y_m"][0]) if "grid_cell_y_m" in data else self.grid_spec.cell_y_m
        self.radar_max_range_m = float(data["radar_max_range_m"][0]) if "radar_max_range_m" in data else self.grid_range_m
        self.n_steps = int(data["n_steps"][0])
        self.include_dynamic = bool(int(data["include_dynamic"][0]))
        self.wall_spacing_m = float(data["wall_spacing_m"][0]) if "wall_spacing_m" in data else float("nan")
        self.radar_bw_hz = float(data["radar_bw_hz"][0])
        self.radar_n_fast = int(data["radar_n_fast"][0])
        self.radar_range_res_m = float(data["radar_range_res_m"][0])


def build_angle_balanced_sampler(
    dataset: DoAVectorDataset,
    bin_edges: np.ndarray | None = None,
) -> WeightedRandomSampler:
    """Oversample underrepresented DoA ranges for edge-angle stability."""

    if bin_edges is None:
        bin_edges = np.asarray([-90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 90], dtype=np.float32)
    angle = np.asarray(dataset.angle_deg, dtype=np.float32)
    bins = np.clip(np.digitize(angle, bin_edges[1:-1], right=False), 0, len(bin_edges) - 2)
    counts = np.bincount(bins, minlength=len(bin_edges) - 1).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = 1.0 / counts[bins]
    weights = weights / np.mean(weights)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Vector-input classical DoA baselines
# ---------------------------------------------------------------------------

def steering_matrix(angle_grid: np.ndarray = ANGLE_GRID, n_rx: int = 8) -> np.ndarray:
    ant = np.arange(n_rx, dtype=np.float64)[:, None]
    phase = 2.0 * np.pi * D_OVER_LAM * np.sin(np.deg2rad(angle_grid))[None, :] * ant
    return np.exp(1j * phase).astype(np.complex64)  # (A, G)


def normalize_spectrum(p: np.ndarray) -> np.ndarray:
    p = np.maximum(np.real(p), 0.0)
    return p / (np.max(p) + 1e-12)


def angle_fft_spectrum(x: np.ndarray, angle_grid: np.ndarray = ANGLE_GRID) -> np.ndarray:
    """Crude non-parametric angle-FFT baseline on the native antenna aperture.

    This intentionally does not zero-pad to a dense angular grid, so it behaves
    like the coarse/weak baseline students would get from a direct antenna FFT.
    """
    n_rx = x.shape[0]
    spec = np.abs(np.fft.fftshift(np.fft.fft(x, n=n_rx))) ** 2
    u_fft = np.linspace(-1.0, 1.0, len(spec), endpoint=False)
    u_grid = np.sin(np.deg2rad(angle_grid))
    interp = np.interp(u_grid, u_fft, spec, left=0.0, right=0.0)
    return normalize_spectrum(interp)


def music_spectrum_single_snapshot(x: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Rank-1 MUSIC reference from one RD-selected antenna vector.

    The active SNR-only benchmark contains one target per sample, so a rank-1
    covariance surrogate is the intended MUSIC baseline. Covariance/eigendecomposition
    remains inside the classical baseline only; the neural network receives only
    the raw RD-selected antenna vector.
    """
    n_rx = x.shape[0]
    R = np.outer(x, x.conj()) / (np.vdot(x, x).real + 1e-12)
    eigvals, eigvecs = np.linalg.eigh(R)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    En = eigvecs[:, 1:]
    denom = np.sum(np.abs(En.conj().T @ A) ** 2, axis=0)
    return normalize_spectrum(1.0 / (denom + 1e-12))


def estimate_angle_from_spectrum(spec: np.ndarray, angle_grid: np.ndarray = ANGLE_GRID) -> float:
    p = normalize_spectrum(spec)
    peaks, props = find_peaks(p, height=0.2, distance=2)
    if len(peaks) == 0:
        return float(angle_grid[int(np.argmax(p))])
    best = peaks[int(np.argmax(props["peak_heights"]))]
    return float(angle_grid[best])


def _metrics(true_angle: np.ndarray, pred_angle: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(pred_angle)
    if not np.any(valid):
        return {
            "mae_deg": float("nan"),
            "rmse_deg": float("nan"),
            "within_1deg_acc": float("nan"),
            "within_2deg_acc": float("nan"),
            "within_5deg_acc": float("nan"),
            "valid_fraction": 0.0,
            "n_valid": 0,
        }
    err = np.abs(pred_angle[valid] - true_angle[valid])
    out = {
        "mae_deg": float(np.mean(err)),
        "rmse_deg": float(np.sqrt(np.mean(err ** 2))),
        "within_1deg_acc": float(np.mean(err <= 1.0)),
        "within_2deg_acc": float(np.mean(err <= 2.0)),
        "within_5deg_acc": float(np.mean(err <= 5.0)),
    }
    if not np.all(valid):
        out["valid_fraction"] = float(np.mean(valid))
        out["n_valid"] = int(np.sum(valid))
    return out


def _errors(true_angle: np.ndarray, pred_angle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(pred_angle)
    return np.abs(pred_angle[valid] - true_angle[valid]), valid


def _snr_breakdown(true_angle: np.ndarray, pred_angle: np.ndarray, snr_db: np.ndarray) -> dict[str, float]:
    bins = [(1, 5), (5, 10), (10, 15), (15, 20), (20, 25.1)]
    out = {}
    err, valid = _errors(true_angle, pred_angle)
    snr_valid = snr_db[valid]
    for lo, hi in bins:
        mask = (snr_valid >= lo) & (snr_valid < hi)
        if np.any(mask):
            prefix = f"snr_{lo:g}_{hi:g}dB"
            out[f"mae_{prefix}"] = float(np.mean(err[mask]))
            out[f"within_1deg_acc_{prefix}"] = float(np.mean(err[mask] <= 1.0))
            out[f"within_2deg_acc_{prefix}"] = float(np.mean(err[mask] <= 2.0))
            out[f"within_5deg_acc_{prefix}"] = float(np.mean(err[mask] <= 5.0))
            out[f"n_{prefix}"] = int(np.sum(mask))
    return out


def _method_metrics(
    true_angle: np.ndarray,
    pred_angle: np.ndarray,
    dataset: DoAVectorDataset,
) -> dict[str, float]:
    return {
        **_metrics(true_angle, pred_angle),
        **_snr_breakdown(true_angle, pred_angle, dataset.snr_db),
    }


def _angle_velocity_breakdown(
    true_angle: np.ndarray,
    pred_angle: np.ndarray,
    velocity_mps: np.ndarray,
) -> dict[str, float]:
    out = {}
    err, valid = _errors(true_angle, pred_angle)
    abs_angle = np.abs(true_angle[valid])
    for lo, hi in [(0, 20), (20, 45), (45, 90.1)]:
        mask = (abs_angle >= lo) & (abs_angle < hi)
        if np.any(mask):
            out[f"mae_abs_angle_{lo:g}_{hi:g}deg"] = float(np.mean(err[mask]))
    abs_vel = np.abs(velocity_mps[valid])
    for lo, hi in [(0, 5), (5, 10), (10, 20)]:
        mask = (abs_vel >= lo) & (abs_vel < hi)
        if np.any(mask):
            out[f"mae_abs_velocity_{lo:g}_{hi:g}mps"] = float(np.mean(err[mask]))
    return out


@torch.no_grad()
def _predict_model_angles(model: nn.Module, dataset: DoAVectorDataset, device: str = "cpu") -> np.ndarray:
    model.eval()
    x = dataset.x.to(device)
    logits = []
    batch = 512
    for i in range(0, len(x), batch):
        logits.append(model(x[i:i + batch]).cpu())
    pred_spec = torch.sigmoid(torch.cat(logits, dim=0)).numpy()
    return np.array([estimate_angle_from_spectrum(s) for s in pred_spec], dtype=np.float32)


def _predict_signal_processing_angles(dataset: DoAVectorDataset) -> tuple[np.ndarray, np.ndarray]:
    A = steering_matrix(ANGLE_GRID, n_rx=dataset.x.shape[-1])
    x_complex = dataset.x.numpy()[:, 0, :] + 1j * dataset.x.numpy()[:, 1, :]
    pred_fft, pred_music = [], []
    for row in x_complex:
        pred_fft.append(estimate_angle_from_spectrum(angle_fft_spectrum(row, ANGLE_GRID)))
        pred_music.append(estimate_angle_from_spectrum(music_spectrum_single_snapshot(row, A)))

    pred_fft = np.asarray(pred_fft, dtype=np.float32)
    pred_music = np.asarray(pred_music, dtype=np.float32)
    return pred_fft, pred_music


@torch.no_grad()
def evaluate(model: nn.Module, dataset: DoAVectorDataset, device: str = "cpu") -> dict:
    true = dataset.angle_deg
    pred_model = _predict_model_angles(model, dataset, device)
    pred_fft, pred_music = _predict_signal_processing_angles(dataset)

    return {
        "model": {
            **_method_metrics(true, pred_model, dataset),
            **_angle_velocity_breakdown(true, pred_model, dataset.velocity_mps),
        },
        "baseline_angle_fft": _method_metrics(true, pred_fft, dataset),
        "baseline_music_single_snapshot": _method_metrics(true, pred_music, dataset),
        "data_contract": {
            "input": "shared FMCW dechirp beat cube -> range FFT -> Doppler FFT -> selected RD antenna vector; angle FFT skipped for neural input",
            "label_source": "simulator-known single target angle/range/velocity/SNR metadata",
            "difficulty_model": "positive selected-target SNR only; one target per sample; no same-RD collision",
            "simulator": "shared.fmcw_simulator.FMCWRadar",
            "up_down_conversion": "excluded_baseband_only",
            "n_test": int(len(dataset)),
            "angle_grid_min_deg": float(ANGLE_GRID[0]),
            "angle_grid_max_deg": float(ANGLE_GRID[-1]),
        },
    }


def _mean_metric_dict(rows: list[dict[str, float]], prefix: str = "") -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row.keys() if isinstance(row.get(k), (int, float))})
    out = {}
    for key in keys:
        vals = np.asarray([row[key] for row in rows if key in row and np.isfinite(row[key])], dtype=np.float64)
        if len(vals):
            out[f"{prefix}{key}"] = float(np.mean(vals))
    return out


def _poses_for_scene(dataset: MappingDetectionDataset, scene_idx: int) -> list[EgoPose]:
    return [
        EgoPose(
            x_m=float(p[0]),
            y_m=float(p[1]),
            heading_deg=float(p[2]),
            speed_mps=float(p[3]),
        )
        for p in dataset.poses[scene_idx]
    ]


def _mapping_metrics_for_angles(
    dataset: MappingDetectionDataset,
    pred_angle: np.ndarray,
    use_dynamic: bool = False,
) -> dict[str, float]:
    """Project predicted DoA into point-cloud/OGM maps and score them."""

    scene_metrics_05: list[dict[str, float]] = []
    scene_metrics_04: list[dict[str, float]] = []
    pc_grid_metrics: list[dict[str, float]] = []
    n_scenes = dataset.gt_ogm.shape[0]

    eval_mask = np.ones(len(dataset.angle_deg), dtype=bool)
    if not use_dynamic:
        eval_mask &= ~dataset.is_dynamic
    loc_err = localization_errors_m(
        dataset.range_m[eval_mask],
        dataset.angle_deg[eval_mask],
        pred_angle[eval_mask],
    )

    for scene_idx in range(n_scenes):
        poses = _poses_for_scene(dataset, scene_idx)
        per_frame_angles = [[] for _ in range(dataset.n_steps)]
        per_frame_ranges = [[] for _ in range(dataset.n_steps)]
        det_mask = dataset.scene_idx == scene_idx
        if not use_dynamic:
            det_mask &= ~dataset.is_dynamic

        for i in np.nonzero(det_mask)[0]:
            frame = int(dataset.frame_idx[i])
            if 0 <= frame < dataset.n_steps:
                per_frame_angles[frame].append(float(pred_angle[i]))
                per_frame_ranges[frame].append(float(dataset.range_m[i]))

        ogm_prob, ogm_bin_05 = accumulate_probability_map(
            poses,
            per_frame_angles,
            per_frame_ranges,
            grid_size=dataset.grid_size,
            grid_range_m=dataset.grid_range_m,
            grid_spec=dataset.grid_spec,
            max_range_m=dataset.radar_max_range_m,
            beam_width_deg=5.0,
            p_occ=0.60,
            p_free=0.45,
        )
        # Week-12 often uses 0.4; use a strict comparison so prior-only cells
        # at exactly p=0.4 do not become occupied everywhere.
        ogm_bin_04 = (ogm_prob > 0.400001).astype(np.float32)
        gt = dataset.gt_ogm[scene_idx]
        scene_metrics_05.append(map_metrics(gt, ogm_bin_05))
        scene_metrics_04.append(map_metrics(gt, ogm_bin_04))

        points = point_cloud_from_measurements(poses, per_frame_ranges, per_frame_angles)
        pc_grid = point_cloud_grid(
            points,
            grid_size=dataset.grid_size,
            grid_range_m=dataset.grid_range_m,
            grid_spec=dataset.grid_spec,
            sigma_cells=1.0,
        )
        pc_grid_metrics.append(map_metrics(gt, pc_grid))

    return {
        **_mean_metric_dict(scene_metrics_05, prefix="ogm_thr0p5_"),
        **_mean_metric_dict(scene_metrics_04, prefix="ogm_thr0p4_"),
        **_mean_metric_dict(pc_grid_metrics, prefix="point_cloud_grid_"),
        "point_error_mean_m": float(np.mean(loc_err)) if len(loc_err) else float("nan"),
        "point_error_median_m": float(np.median(loc_err)) if len(loc_err) else float("nan"),
        "point_error_p90_m": float(np.percentile(loc_err, 90)) if len(loc_err) else float("nan"),
        "n_static_detections": int(np.sum(eval_mask)),
        "n_scenes": int(n_scenes),
    }


@torch.no_grad()
def evaluate_mapping(model: nn.Module, dataset: MappingDetectionDataset, device: str = "cpu") -> dict:
    true = dataset.angle_deg
    pred_model = _predict_model_angles(model, dataset, device)
    pred_fft, pred_music = _predict_signal_processing_angles(dataset)
    pred_oracle = true.astype(np.float32)

    return {
        "deep_learning_doa": {
            **_method_metrics(true, pred_model, dataset),
            **_angle_velocity_breakdown(true, pred_model, dataset.velocity_mps),
        },
        "signal_processing_angle_fft_doa": _method_metrics(true, pred_fft, dataset),
        "signal_processing_music_doa": _method_metrics(true, pred_music, dataset),
        "oracle_gt_doa": _method_metrics(true, pred_oracle, dataset),
        "map_from_deep_learning_doa": _mapping_metrics_for_angles(dataset, pred_model),
        "map_from_angle_fft_doa": _mapping_metrics_for_angles(dataset, pred_fft),
        "map_from_music_doa": _mapping_metrics_for_angles(dataset, pred_music),
        "map_from_oracle_gt_doa": _mapping_metrics_for_angles(dataset, pred_oracle),
        "data_contract": {
            "input": "moving-ego shared FMCW scene -> range FFT -> Doppler FFT -> per-detection RD antenna vector",
            "evaluation": "same range detections and perfect ego poses are mapped with each method's DoA",
            "mainline_ego_motion": "perfect/error-free",
            "ego_motion_error": "appendix-only; use GT DoA/range then perturb pose",
            "relative_velocity": "simulator closing velocity = dot(v_ego - v_target, line_of_sight)",
            "simulator": "shared.fmcw_simulator.FMCWRadar",
            "up_down_conversion": "excluded_baseband_only",
            "n_test_detections": int(len(dataset)),
            "n_test_scenes": int(dataset.gt_ogm.shape[0]),
            "grid_size": int(dataset.grid_size),
            "grid_nx": int(dataset.grid_nx),
            "grid_ny": int(dataset.grid_ny),
            "grid_range_m": float(dataset.grid_range_m),
            "grid_x_min_m": float(dataset.grid_spec.x_min_m),
            "grid_x_max_m": float(dataset.grid_spec.x_max_m),
            "grid_y_min_m": float(dataset.grid_spec.y_min_m),
            "grid_y_max_m": float(dataset.grid_spec.y_max_m),
            "grid_cell_x_m": float(dataset.grid_cell_x_m),
            "grid_cell_y_m": float(dataset.grid_cell_y_m),
            "radar_max_range_m": float(dataset.radar_max_range_m),
            "wall_spacing_m": float(dataset.wall_spacing_m),
            "radar_bw_hz": float(dataset.radar_bw_hz),
            "radar_n_fast": int(dataset.radar_n_fast),
            "radar_range_res_m": float(dataset.radar_range_res_m),
            "n_steps": int(dataset.n_steps),
        },
    }


class SpectrumLoss(nn.Module):
    """Weighted spectrum loss for sparse Gaussian DoA targets."""

    def __init__(self, pos_weight: float = 8.0, mse_weight: float = 0.25):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        self.mse_weight = mse_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.mse = nn.MSELoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, target) + self.mse_weight * self.mse(torch.sigmoid(logits), target)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        total += criterion(model(xb), yb).item() * xb.size(0)
    return total / len(loader.dataset)


def main() -> None:
    parser = base_parser("P03 Radar Mapping via DoA Spectrum Network")
    parser.add_argument("--mapping", action="store_true", help="Use moving-ego mapping dataset/evaluation path")
    parser.add_argument("--n_train", type=int, default=24000)
    parser.add_argument("--n_val", type=int, default=4000)
    parser.add_argument("--n_test", type=int, default=4000)
    parser.add_argument("--n_train_scenes", type=int, default=80)
    parser.add_argument("--n_val_scenes", type=int, default=16)
    parser.add_argument("--n_test_scenes", type=int, default=16)
    parser.add_argument("--mapping_steps", type=int, default=6)
    parser.add_argument("--mapping_data_dir", type=str, default=None)
    parser.add_argument("--radar_bw_mhz", type=float, default=200.0,
                        help="Mapping radar bandwidth. Results standard: 200 MHz; low-res stress: 50 MHz.")
    parser.add_argument("--radar_n_fast", type=int, default=1024,
                        help="Fast-time samples for mapping radar. Use 1024 with 200 MHz.")
    parser.add_argument("--wall_spacing_m", type=float, default=0.35,
                        help="Dense wall scatterer spacing for mapping scenes. Smoke runs relax this to 1 m.")
    parser.add_argument("--grid_size", type=int, default=128,
                        help="Mapping grid cells per axis for the uniform lecture map.")
    parser.add_argument("--map_x_min_m", type=float, default=-20.0)
    parser.add_argument("--map_x_max_m", type=float, default=20.0)
    parser.add_argument("--map_y_min_m", type=float, default=0.0)
    parser.add_argument("--map_y_max_m", type=float, default=40.0)
    parser.add_argument("--radar_max_range_m", type=float, default=40.0,
                        help="Radar visibility/ISM max range, decoupled from map x half-width.")
    parser.add_argument("--balance_angles", action="store_true",
                        help="Use a weighted sampler that balances DoA angle bins during training.")
    parser.add_argument("--artifact_dir", type=str, default=None,
                        help="Directory for best_model.pt/history.json/metrics.json. Defaults to ./artifacts.")
    parser.add_argument("--include_dynamic", action="store_true", help="Include dynamic detections; GT OGM remains static")
    args = parser.parse_args()

    if args.smoke:
        args.n_train, args.n_val, args.n_test = 256, 64, 64
        args.n_train_scenes, args.n_val_scenes, args.n_test_scenes = 2, 1, 1
        args.mapping_steps = min(args.mapping_steps, 4)
        args.wall_spacing_m = max(args.wall_spacing_m, 1.0)
        args.grid_size = min(args.grid_size, 48)
        if "--radar_bw_mhz" not in sys.argv:
            args.radar_bw_mhz = 50.0
        if "--radar_n_fast" not in sys.argv:
            args.radar_n_fast = 256
        args.epochs = 2
        args.batch_size = 32

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else BASE / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    mapping_data_dir = Path(args.mapping_data_dir) if args.mapping_data_dir else BASE / "data_mapping"
    if args.generate:
        if args.mapping:
            cmd = [
                sys.executable, str(BASE / "generate_mapping_data.py"),
                "--n_train_scenes", str(args.n_train_scenes),
                "--n_val_scenes", str(args.n_val_scenes),
                "--n_test_scenes", str(args.n_test_scenes),
                "--n_steps", str(args.mapping_steps),
                "--radar_bw_mhz", str(args.radar_bw_mhz),
                "--radar_n_fast", str(args.radar_n_fast),
                "--wall_spacing_m", str(args.wall_spacing_m),
                "--grid_size", str(args.grid_size),
                "--map_x_min_m", str(args.map_x_min_m),
                "--map_x_max_m", str(args.map_x_max_m),
                "--map_y_min_m", str(args.map_y_min_m),
                "--map_y_max_m", str(args.map_y_max_m),
                "--radar_max_range_m", str(args.radar_max_range_m),
                "--seed", str(args.seed),
                "--out_dir", str(mapping_data_dir),
            ]
            if args.include_dynamic:
                cmd.append("--include_dynamic")
        else:
            cmd = [
                sys.executable, str(BASE / "generate_data.py"),
                "--n_train", str(args.n_train),
                "--n_val", str(args.n_val),
                "--n_test", str(args.n_test),
                "--seed", str(args.seed),
            ]
        if args.smoke:
            cmd.append("--smoke")
        subprocess.run(cmd, check=True)

    if args.mapping:
        train_ds = MappingDetectionDataset(mapping_data_dir / "train.h5")
        val_ds = MappingDetectionDataset(mapping_data_dir / "val.h5")
        test_ds = MappingDetectionDataset(mapping_data_dir / "test.h5")
    else:
        train_ds = DoAVectorDataset(BASE / "data" / "train.h5")
        val_ds = DoAVectorDataset(BASE / "data" / "val.h5")
        test_ds = DoAVectorDataset(BASE / "data" / "test.h5")
    model = build_model(n_rx=train_ds.x.shape[-1], grid_size=train_ds.y.shape[-1]).to(device)
    print(f"\n[Model] RadarCubeDoANet params={count_parameters(model):,} device={device}")

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[Load] {args.checkpoint}")

    if not args.eval_only:
        sampler = build_angle_balanced_sampler(train_ds) if args.balance_angles else None
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)
        criterion = SpectrumLoss(pos_weight=8.0, mse_weight=0.25).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        history = {"train_loss": [], "val_loss": []}
        best = float("inf")
        print(f"[Train] {args.epochs} epochs angle_balanced={args.balance_angles}")
        for epoch in range(1, args.epochs + 1):
            tr = train_one_epoch(model, train_loader, criterion, optimizer, device)
            va = validate(model, val_loader, criterion, device)
            history["train_loss"].append(tr)
            history["val_loss"].append(va)
            star = ""
            if va < best:
                best = va
                torch.save(model.state_dict(), artifact_dir / "best_model.pt")
                star = " *"
            if epoch <= 3 or epoch % 5 == 0 or epoch == args.epochs:
                print(f"  Epoch {epoch:3d}/{args.epochs} train={tr:.4f} val={va:.4f}{star}")
        with open(artifact_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    ckpt = args.checkpoint or str(artifact_dir / "best_model.pt")
    if Path(ckpt).exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
    elif args.eval_only:
        raise FileNotFoundError(
            f"Evaluation checkpoint not found: {ckpt}. "
            "Pass --checkpoint or run training before --eval_only."
        )

    print("\n[Eval]")
    metrics = evaluate_mapping(model, test_ds, device=device) if args.mapping else evaluate(model, test_ds, device=device)
    for group, vals in metrics.items():
        print(f"  [{group}]")
        for k, v in vals.items():
            if isinstance(v, (int, float)):
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
            else:
                print(f"    {k}: {v}")
    with open(artifact_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[Saved] {artifact_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
