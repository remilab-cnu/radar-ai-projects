# Radar Signal Processing with AI — Student Projects

Graduate-level course projects covering radar signal processing and deep learning.
Each project provides a complete pipeline: **synthetic data generation → model training → quantitative evaluation**, all runnable on CPU.

## Projects

| # | Title | Topic | Params | Simulator |
|---|-------|-------|--------|-----------|
| P1 | U-Net FMCW Detector | Target detection on RDM | 7.7M | FMCW + clutter |
| P2 | ResNet-18 HAR | Human activity classification | 11.1M | micro-Doppler |
| P3 | DeepMUSIC CNN | Direction-of-Arrival estimation | 19.3M | DoA/array |
| P4 | DnCNN-SAR Despeckling | SAR image restoration | 556K | SAR |
| P5 | Neural CFAR | Learned detection threshold | 28K | FMCW + clutter |
| P6 | I/Q Imbalance Correction | RF front-end compensation | 133K | FMCW |
| P7 | Full-Duplex SIC | Self-interference cancellation | 302K | chirp + FIR channel |
| P8 | Jammer Null Steering | Adaptive beamforming | 73K | DoA/array |
| P9 | RD Super-Resolution | Range-Doppler enhancement | 121K | FMCW |
| P10 | Near-Field Localization | Spherical-wave source finding | 6K | DoA/array (near-field) |

> **P1–P4** are lecture-use examples (full-size models, shown in class).
> **P5–P10** are student project templates (CPU-friendly, <302K parameters).

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd radar-ai-projects
pip install -r requirements.txt

# 2. Pick a project (e.g., P5)
cd projects/p05_neural_cfar

# 3. Generate data + train (smoke test first)
python train.py --generate --smoke

# 4. Full training
python train.py --generate --epochs 30

# 5. Evaluate
python train.py --eval_only --checkpoint artifacts/best_model.pt

# 6. Run all projects (smoke)
python scripts/smoke_all.py

# 7. Run all projects (2-epoch full test)
python scripts/smoke_all.py --full
```

---

## Simulator Documentation

All projects use physics-based simulators in `shared/`. No external datasets are needed — everything is generated synthetically from first principles.

### Signal Processing Chain

```
┌─────────────────────────────────────────────────────────────┐
│                   FMCW Radar Full Chain                      │
│                                                              │
│  System Parameters ──→ Beat Signal Gen ──→ Range FFT         │
│  (FMCWRadar class)     (generate_scene)    (range_fft)       │
│                              │                  │            │
│  Clutter Extension ──────────┘                  │            │
│  (clutter_model.py)                             ▼            │
│                                          Range-Doppler Map   │
│                                          (range_doppler_map) │
│                                                │             │
│                              ┌─────────────────┼────────┐   │
│                              ▼                  ▼        ▼   │
│                        Range-Angle Map    CA-CFAR     to_db  │
│                        (range_angle_map)  (1D / 2D)          │
└─────────────────────────────────────────────────────────────┘
```

### `shared/fmcw_simulator.py` — FMCW Radar

Complete FMCW radar signal chain: system parameter definition → analytic beat signal generation → FFT processing → adaptive detection.

#### `FMCWRadar` Class

System parameter container with derived quantities.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fc` | 77 GHz | Carrier frequency |
| `bw` | 1 GHz | Chirp bandwidth |
| `T_chirp` | 50 μs | Chirp duration |
| `N_chirps` | 128 | Chirps per frame (slow-time) |
| `fs` | 10 MHz | ADC sample rate |
| `N_rx` | 1 | Receive antennas (SIMO, no MIMO) |
| `d_rx` | λ/2 | Antenna spacing |

Derived properties:

| Property | Formula | Typical Value |
|----------|---------|---------------|
| `range_res` | c / (2·BW) | 0.15 m |
| `max_range` | fs·c·T / (4·BW) | 37.5 m |
| `vel_res` | λ / (2·N_chirps·T) | 0.015 m/s |
| `max_vel` | λ / (4·T) | 0.97 m/s |
| `N_samples` | T_chirp × fs | 500 |

#### `generate_scene(radar, targets, snr_db, seed)`

Analytic beat signal generation for multiple targets.

**Signal model** (per target, per chirp `m`, per antenna `rx`):

```
beat_signal[rx, m, n] = √(RCS) · exp(j·2π·f_beat·t[n])     ← fast-time (range)
                       · exp(j·2π·f_d·m·T_chirp)             ← slow-time (Doppler)
                       · exp(j·2π·rx·d·sin(θ)/λ)             ← spatial (angle)

where:
  f_beat = μ · τ = (BW/T_chirp) · (2R/c)    [beat frequency]
  f_d    = 2v/λ                                [Doppler frequency]
  t[n]   = n/fs                                [fast-time samples]
```

Noise: complex AWGN scaled to target `snr_db` (referenced to strongest target).

Output shape: `(N_rx, N_chirps, N_samples)` complex128.

#### `range_fft(signal, window='hann')`

1D FFT along fast-time (last axis). Windowing: `hann`, `hamming`, `blackman`, `rect`.

#### `range_doppler_map(signal, window_range, window_doppler)`

2D FFT processing:
1. Range FFT (fast-time axis) with `window_range`
2. Doppler FFT (slow-time axis) with `window_doppler`
3. `fftshift` on Doppler axis

Output shape: same as input, complex. Zero-Doppler is centered.

#### `range_angle_map(signal, radar, window_range, window_angle, N_angle)`

1. Range FFT
2. Coherent integration across chirps (mean)
3. Spatial FFT across antennas (with zero-padding to `N_angle`)

Returns: `(ram, angle_axis)` — Range-Angle Map + angle grid in degrees.

#### `ca_cfar_1d(signal_mag, guard_cells, train_cells, pfa)` / `ca_cfar_2d(...)`

Cell-Averaging CFAR detection.

| Parameter | 1D Default | 2D Default |
|-----------|-----------|-----------|
| `guard_cells` | 2 | (2, 2) |
| `train_cells` | 8 | (4, 4) |
| `pfa` | 1e-4 | 1e-4 |

Threshold: `α = N_train · (Pfa^(-1/N_train) - 1)`, where `N_train` = number of training cells.

---

### `shared/clutter_model.py` — Clutter Extension

Extends `generate_scene()` with realistic clutter environments.

#### `generate_scene_with_clutter(radar, targets, snr_db, clutter_type, ...)`

| Clutter Type | Model | Behavior |
|-------------|-------|----------|
| `zero_doppler` | Static scatterers at v≈0 | Ground clutter, buildings |
| `distributed` | Range-dependent power (closer = stronger) | Volumetric clutter |
| `multipath` | Ghost at 2× target range, 10% RCS | Multi-bounce reflection |
| `mixed` | All of the above combined | Most realistic |

Also generates **ground-truth target mask** (binary, cross-shaped PSF matching Hann window mainlobe) and target bin coordinates.

#### `generate_random_scene(radar, rng, ...)`

Random scenario generator for training data. Outputs 2-channel RDM:
- **ch0**: Noise-floor-referenced log-magnitude (median normalization, clipped [-20, 40] dB → [0, 1])
- **ch1**: Normalized phase [-1, 1]

---

### `shared/doa_utils.py` — Array Signal Processing

Direction-of-Arrival estimation utilities for ULA systems.

#### Signal Generation

`generate_doa_sample(N_rx, n_sources_range, snr_range, ...)` creates:

| Feature | Range | Description |
|---------|-------|-------------|
| Sources | 1–3 | With minimum angle separation |
| SNR | 0–20 dB | Per-sample random |
| Snapshots | 10–200 | Time samples |
| Coherent | 20% prob | Correlated source waveforms |
| Moving | Configurable | Time-varying steering vectors |

**Output**: Sample covariance `R̂` (2-channel real/imag, Frobenius-normalized) + Gaussian pseudo-spectrum label.

#### Classical Algorithms

| Algorithm | Function | Description |
|-----------|----------|-------------|
| CBF | `cbf_spectrum()` | Conventional beamformer: `P(θ) = a^H R a` |
| MUSIC | `music_spectrum()` | Noise subspace orthogonality: `P(θ) = 1/(a^H E_n E_n^H a)` |
| MVDR | `mvdr_spectrum()` | Capon minimum variance: `P(θ) = 1/(a^H R^{-1} a)` |

#### Source Number Estimation

| Method | Function | Criterion |
|--------|----------|-----------|
| MDL | `estimate_n_sources_mdl()` | Minimum Description Length |
| AIC | `estimate_n_sources_aic()` | Akaike Information Criterion |

#### Evaluation

`compute_doa_rmse(est, true)` — Greedy nearest-neighbor matching (10° threshold) with miss/FA counting.

---

### `shared/micro_doppler.py` — Human Activity Recognition

Boulic kinematic body model (10-segment, 11-joint) for micro-Doppler signature generation.

| Segment | Length (m) | RCS |
|---------|-----------|-----|
| torso | 0.50 | 1.00 |
| head | 0.20 | 0.30 |
| upper_arm | 0.28 | 0.15 |
| lower_arm | 0.25 | 0.10 |
| upper_leg | 0.43 | 0.25 |
| lower_leg | 0.43 | 0.15 |

**Activities**: `walk`, `run`, `sit_down`, `fall`, `wave`, `idle` (6 classes).

Pipeline: Joint angles → segment velocities → Doppler shifts → STFT spectrogram → resized to model input.

---

### `shared/sar_simulator.py` — SAR Imaging

Stripmap SAR simulator with simplified Range-Doppler Algorithm (RDA).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fc` | 9.6 GHz | X-band carrier |
| `bw` | 100 MHz | Chirp bandwidth |
| `prf` | 400 Hz | Pulse repetition frequency |
| `V` | 200 m/s | Platform velocity |
| `R0` | 10 km | Scene center range |
| `N_az × N_rg` | 256 × 256 | Image dimensions |

Functions:
- `generate_sar_image()` — Point target → focused SAR image via RDA
- `defocus_image()` — Phase error injection for autofocus studies

---

### `shared/radar_scene.py` — 2D Radar Scene

Ego-vehicle perspective radar environment with static/dynamic targets.

- `RadarScene` — Multi-target 2D scene definition
- `patch_element_pattern()` — Patch antenna cos(θ) model
- `polar_to_cartesian()` / `generate_ogm_gt()` — Occupancy Grid Map generation

---

### `shared/plot_style.py` — Common Plot Style

Consistent matplotlib styling for all projects.

---

## Common Infrastructure

### `common/cli.py` — Shared Argument Parser

`base_parser(description)` returns a parser with standard flags: `--generate`, `--smoke`, `--epochs`, `--batch_size`, `--lr`, `--eval_only`, `--checkpoint`, `--seed`.

All `train.py` and `generate_data.py` files extend this parser.

### `common/hdf5_io.py` — Data I/O

- `save_hdf5(path, **arrays)` — Save numpy arrays to HDF5
- `load_hdf5(path, keys)` — Load specific keys
- `HDF5Dataset(path, x_key, y_key)` — PyTorch Dataset wrapper

### `common/train_utils.py` — Training Loop

- `training_loop(model, train_dl, val_dl, criterion, optimizer, ...)` — Standard training with best-val checkpointing, timing, history JSON
- `count_parameters(model)` — Total trainable parameter count

> Note: P06, P08, P10 use custom training loops due to multi-input/multi-output architectures. See their `train.py` for documented reasons.

### `common/metrics.py` — Evaluation

| Function | Use Case |
|----------|----------|
| `classification_report()` | Accuracy, precision, recall, F1 |
| `pd_at_pfa()` | Pd at fixed Pfa (detection curves) |
| `regression_report()` | MAE, RMSE, R² |
| `psnr()` | Peak signal-to-noise ratio |
| `nmse()` | Normalized mean squared error |

### `common/seed.py` — Reproducibility

`seed_everything(seed)` — Sets `random`, `numpy`, `torch` seeds consistently.

---

## Repository Structure

```
radar-ai-projects/
├── shared/                # Physics-based simulators
│   ├── fmcw_simulator.py      FMCW full chain (beat signal → RDM → CFAR)
│   ├── clutter_model.py       Clutter environments (zero-Doppler, distributed, multipath)
│   ├── doa_utils.py           Array processing (steering, MUSIC, MVDR, MDL/AIC)
│   ├── micro_doppler.py       Boulic body model → STFT spectrogram
│   ├── sar_simulator.py       Stripmap SAR (RDA, phase error)
│   ├── radar_scene.py         2D ego-vehicle scene + OGM
│   └── plot_style.py          Common matplotlib style
├── common/                # Training infrastructure
│   ├── cli.py                 Shared argparse (--generate, --smoke, --epochs, ...)
│   ├── hdf5_io.py             HDF5 save/load + PyTorch Dataset
│   ├── train_utils.py         Training loop + checkpointing
│   ├── metrics.py             Classification, regression, detection metrics
│   └── seed.py                Reproducible seeding
├── projects/
│   ├── p01_unet_detector/     U-Net FMCW target detection
│   ├── p02_resnet18_har/      ResNet-18 human activity recognition
│   ├── p03_deepmusic_cnn/     DeepMUSIC DoA estimation
│   ├── p04_dncnn_sar/         DnCNN SAR despeckling
│   ├── p05_neural_cfar/       Neural CFAR detection
│   ├── p06_iq_imbalance/      I/Q imbalance correction
│   ├── p07_full_duplex_sic/   Full-duplex self-interference cancellation
│   ├── p08_jammer_nulling/    Jammer null steering
│   ├── p09_rd_superres/       Range-Doppler super-resolution
│   └── p10_nearfield_loc/     Near-field source localization
│       ├── README.md              Project description & instructions
│       ├── generate_data.py       Synthetic data generation → data/*.h5
│       ├── model.py               PyTorch model definition
│       └── train.py               Train + evaluate entry point
├── scripts/
│   └── smoke_all.py           Smoke test all 10 projects
└── requirements.txt           Python dependencies
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CPU)
- NumPy, SciPy, h5py, matplotlib, scikit-learn, tqdm

No GPU required. Student projects (P5–P10) train in under 10 minutes on an 8-core CPU.

## Course

REMI Lab, Chungnam National University
Dept. of Information Communication Convergence Engineering
