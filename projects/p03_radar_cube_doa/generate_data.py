#!/usr/bin/env python3
"""P03 -- Generate the retained single-target DoA-unit training data.

The active P03 project is mapping-first; see ``generate_mapping_data.py`` for
the moving-ego point-cloud/OGM dataset.  This file is retained as a low-level
antenna-vector DoA contract: it verifies that the shared 77 GHz complex-baseband
FMCW dechirp simulator can produce an RD-selected antenna snapshot consumed by
the neural and signal-processing DoA methods.

1. synthesize target-only raw array beat data with ``shared.fmcw_simulator``,
2. run range FFT and Doppler FFT only,
3. skip angle FFT for the neural input path,
4. select the simulator-known target response at its (R,D) bin,
5. store the complex antenna snapshot vector as the NN/classical-DoA input.

Difficulty is controlled by positive selected-target SNR. Each sample contains
one simulator-known target, so there is no same-RD angular collision or target-RD
overlap case in the active benchmark.

HDF5 schema:
  x_ant        (N, 2, N_rx)  selected complex antenna vector [real, imag]
  y_spectrum   (N, G)        Gaussian DoA spectrum label for selected target
  angle_deg    (N,)          selected target angle label
  range_m      (N,)          selected target range label
  velocity_mps (N,)          selected target radial velocity label
  snr_db       (N,)          realised selected-target SNR label
  r_bin        (N,)          selected target range FFT bin
  d_bin        (N,)          selected target Doppler FFT bin after fftshift
  n_targets    (N,)          always 1 in the active SNR-only benchmark
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.cli import base_parser
from common.hdf5_io import save_hdf5
from common.seed import seed_everything
from shared.fmcw_simulator import FMCWRadar, generate_scene, range_axis, range_doppler_map, velocity_axis

BASE = Path(__file__).parent

# Approved shared FMCW physical simulator configuration for the P03 DoA lane.
# fs is intentionally 4 * bandwidth.  T_chirp equals the sampled fast-time window
# so the dechirped range FFT bin spacing is the physical c/(2*B) resolution.
FC_HZ = 77.0e9
BANDWIDTH_HZ = 50e6
FS_HZ = 4.0 * BANDWIDTH_HZ
N_FAST = 256
T_SWEEP_S = N_FAST / FS_HZ
PRI_S = 50e-6
N_CHIRPS = 32
N_RX = 8


def build_p03_radar(
    bandwidth_hz: float = BANDWIDTH_HZ,
    n_fast: int = N_FAST,
    n_chirps: int = N_CHIRPS,
    n_rx: int = N_RX,
    pri_s: float = PRI_S,
) -> FMCWRadar:
    """Build a P03-compatible FMCW radar while preserving ``fs = 4*bw``.

    The default is the compact 50 MHz smoke/unit radar.  Mapping result runs
    should use the documented 200 MHz / 1024-fast-sample preset so map quality
    is not dominated by an intentionally coarse 3 m range cell.
    """

    fs_hz = 4.0 * float(bandwidth_hz)
    t_sweep_s = int(n_fast) / fs_hz
    return FMCWRadar(
        fc=FC_HZ,
        bw=float(bandwidth_hz),
        T_chirp=t_sweep_s,
        PRI=pri_s,
        N_chirps=int(n_chirps),
        N_rx=int(n_rx),
        N_samples=int(n_fast),
        fs=fs_hz,
        temperature_k=290.0,
        phase_noise_std_rad=0.0,
        reference_range_m=90.0,
    )


P03_RADAR = build_p03_radar()

D_OVER_LAM = P03_RADAR.d_rx / P03_RADAR.lam
ANGLE_GRID = np.linspace(-90.0, 90.0, 181, dtype=np.float32)
LABEL_SIGMA_DEG = 2.0
SCHEMA_VERSION = 4

RANGE_AXIS_M = range_axis(P03_RADAR).astype(np.float32)
VELOCITY_AXIS_MPS = velocity_axis(P03_RADAR).astype(np.float32)
RANGE_RES_M = P03_RADAR.range_res
RANGE_BIN_SPACING_M = P03_RADAR.range_bin_spacing
VELOCITY_BIN_SPACING_MPS = P03_RADAR.vel_res
MAX_CAPTURE_RANGE_M = 0.85 * 0.5 * 299_792_458.0 * P03_RADAR.T_chirp
_MIN_RANGE_BIN = 10
_MAX_RANGE_BIN = max(_MIN_RANGE_BIN + 1, min(P03_RADAR.N_range_bins - 12, int(np.searchsorted(RANGE_AXIS_M, MAX_CAPTURE_RANGE_M))))
_MIN_DOPPLER_BIN = 4
_MAX_DOPPLER_BIN = N_CHIRPS - 5


def steering_vector(angle_deg: float, n_rx: int = N_RX) -> np.ndarray:
    ant = np.arange(n_rx, dtype=np.float64)
    phase = 2.0 * np.pi * D_OVER_LAM * np.sin(np.deg2rad(angle_deg)) * ant
    return np.exp(1j * phase).astype(np.complex64)


def range_bin_to_range_m(r_bin: float) -> float:
    return float(RANGE_AXIS_M[int(round(r_bin))])


def doppler_bin_to_velocity_mps(d_bin_shifted: int) -> float:
    return float(VELOCITY_AXIS_MPS[int(d_bin_shifted)])


def _label_spectrum(angle_deg: float, grid: np.ndarray = ANGLE_GRID) -> np.ndarray:
    y = np.exp(-0.5 * ((grid - angle_deg) / LABEL_SIGMA_DEG) ** 2)
    return y.astype(np.float32)


def _rcs_for_requested_snr(range_m: float, angle_deg: float) -> float:
    """Choose RCS so the shared simulator's reference-SNR convention is local.

    ``generate_scene(..., snr_db=...)`` scales transmit power for the radar's
    reference range/RCS.  P03 wants SNR difficulty to be independent of the
    randomly selected range and angle, so compensate R^-4 path loss and the
    element-pattern gain in the target RCS.  This keeps the dataset SNR label
    close to the requested selected-target value without bypassing the shared
    radar-equation path.
    """

    angle_gain = max(float(np.cos(np.deg2rad(angle_deg)) ** 2), 0.10)
    range_gain = (float(range_m) / P03_RADAR.reference_range_m) ** 4
    return float(P03_RADAR.reference_rcs_m2 * range_gain / angle_gain)


def _simulate_raw_cube(rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Simulate one target with the shared FMCW dechirp core.

    Returned cube has dimensions ``(antenna, doppler, range)`` after range FFT
    and Doppler FFT.  Angle FFT is intentionally skipped; the selected antenna
    vector remains the raw spatial snapshot consumed by the network and by the
    classical angle-FFT/MUSIC baselines.
    """

    r_bin = int(rng.integers(_MIN_RANGE_BIN, _MAX_RANGE_BIN + 1))
    d_bin = int(rng.integers(_MIN_DOPPLER_BIN, _MAX_DOPPLER_BIN + 1))
    angle = float(rng.uniform(-65.0, 65.0))
    requested_snr_db = float(rng.uniform(5.0, 25.0))
    range_m = float(RANGE_AXIS_M[r_bin])
    velocity_mps = float(VELOCITY_AXIS_MPS[d_bin])
    rcs = _rcs_for_requested_snr(range_m, angle)

    target = {
        "range": range_m,
        "velocity": velocity_mps,
        "angle": angle,
        "rcs": rcs,
        "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
    }
    raw, scene_meta = generate_scene(
        P03_RADAR,
        [target],
        snr_db=requested_snr_db,
        seed=int(rng.integers(0, 2**31)),
        return_meta=True,
    )
    rd_cube = range_doppler_map(raw, radar=P03_RADAR, window_range="hann", window_doppler="hann").astype(np.complex64)
    sim_info = scene_meta["target_info"][0]
    selected_r = int(sim_info["range_bin"])
    selected_d = int(sim_info["doppler_bin"])

    meta = {
        "angle_deg": angle,
        "range_m": float(sim_info["range"]),
        "velocity_mps": float(sim_info["velocity"]),
        "snr_db": float(sim_info["actual_snr_db"]),
        "requested_snr_db": requested_snr_db,
        "target_rcs_m2": rcs,
        "r_bin": selected_r,
        "d_bin": selected_d,
        "n_targets": 1,
        "fs_over_bandwidth": float(scene_meta["fs_over_bandwidth"]),
    }
    return rd_cube, meta


def _normalize_antenna_vector(ant_vec: np.ndarray) -> np.ndarray:
    ant_vec = ant_vec / (np.sqrt(np.mean(np.abs(ant_vec) ** 2)) + 1e-12)
    return ant_vec.astype(np.complex64)


def generate_one_sample(rng: np.random.Generator) -> dict[str, np.ndarray | np.float32 | np.int32]:
    rd_cube, meta = _simulate_raw_cube(rng)
    ant_vec = rd_cube[:, meta["d_bin"], meta["r_bin"]]  # complex (N_rx,)
    ant_vec = _normalize_antenna_vector(ant_vec)
    x_ant = np.stack([ant_vec.real, ant_vec.imag], axis=0).astype(np.float32)

    return {
        "x_ant": x_ant,
        "y_spectrum": _label_spectrum(meta["angle_deg"]),
        "angle_deg": np.float32(meta["angle_deg"]),
        "range_m": np.float32(meta["range_m"]),
        "velocity_mps": np.float32(meta["velocity_mps"]),
        "snr_db": np.float32(meta["snr_db"]),
        "requested_snr_db": np.float32(meta["requested_snr_db"]),
        "target_rcs_m2": np.float32(meta["target_rcs_m2"]),
        "r_bin": np.int32(meta["r_bin"]),
        "d_bin": np.int32(meta["d_bin"]),
        "n_targets": np.int32(meta["n_targets"]),
        "fs_over_bandwidth": np.float32(meta["fs_over_bandwidth"]),
    }


def generate_split(n_samples: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = [generate_one_sample(rng) for _ in range(n_samples)]
    keys = rows[0].keys()
    out: dict[str, np.ndarray] = {}
    for key in keys:
        vals = [row[key] for row in rows]
        if key in {"x_ant", "y_spectrum"}:
            out[key] = np.stack(vals, axis=0)
        elif key in {"r_bin", "d_bin", "n_targets"}:
            out[key] = np.asarray(vals, dtype=np.int32)
        else:
            out[key] = np.asarray(vals, dtype=np.float32)

    out.update({
        "angle_grid_deg": ANGLE_GRID.astype(np.float32),
        "range_axis_m": RANGE_AXIS_M.astype(np.float32),
        "velocity_axis_mps": VELOCITY_AXIS_MPS.astype(np.float32),
        "radar_fc_hz": np.array([P03_RADAR.fc], dtype=np.float64),
        "radar_bw_hz": np.array([P03_RADAR.bw], dtype=np.float64),
        "radar_fs_hz": np.array([P03_RADAR.fs], dtype=np.float64),
        "fs_over_bandwidth": np.array([P03_RADAR.fs / P03_RADAR.bw], dtype=np.float32),
        "schema_version": np.array([SCHEMA_VERSION], dtype=np.int32),
    })
    return out


def main() -> None:
    parser = base_parser("Generate P03 radar-cube DoA datasets")
    parser.add_argument("--n_train", type=int, default=24000)
    parser.add_argument("--n_val", type=int, default=4000)
    parser.add_argument("--n_test", type=int, default=4000)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    if args.smoke:
        args.n_train, args.n_val, args.n_test = 256, 64, 64

    seed_everything(args.seed)
    out_dir = Path(args.out_dir) if args.out_dir else BASE / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== P03 Radar-Cube DoA Dataset ===")
    print(f"  shared FMCW: fc={P03_RADAR.fc/1e9:.1f} GHz, B={P03_RADAR.bw/1e6:.1f} MHz, fs/BW={P03_RADAR.fs/P03_RADAR.bw:.1f}")
    print(f"  cube: chirps={P03_RADAR.N_chirps}, fast={P03_RADAR.N_samples}, antennas={P03_RADAR.N_rx}, range_bins={P03_RADAR.N_range_bins}")
    print("  simulator: shared FMCW dechirp/mixing core; RF up/down conversion excluded")
    print("  RD only: range FFT + Doppler FFT; angle FFT skipped for x_ant")
    print("  difficulty: positive selected-target SNR only")
    print(
        f"  range spacing={P03_RADAR.range_bin_spacing:.3f} m, velocity spacing={P03_RADAR.vel_res:.3f} m/s, "
        f"sampled range bins=[{_MIN_RANGE_BIN}, {_MAX_RANGE_BIN}]"
    )

    for name, n, seed in [
        ("train", args.n_train, args.seed),
        ("val", args.n_val, args.seed + 100000),
        ("test", args.n_test, args.seed + 200000),
    ]:
        print(f"\n[{name}] Generating {n} samples...")
        data = generate_split(n, seed)
        print(f"  x_ant: {data['x_ant'].shape}, y_spectrum: {data['y_spectrum'].shape}")
        print(f"  angle range: [{data['angle_deg'].min():.1f}, {data['angle_deg'].max():.1f}] deg")
        print(f"  SNR range: [{data['snr_db'].min():.1f}, {data['snr_db'].max():.1f}] dB")
        save_hdf5(out_dir / f"{name}.h5", **data)

    print("\nDone.")


if __name__ == "__main__":
    main()
