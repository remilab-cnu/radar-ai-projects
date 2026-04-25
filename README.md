# Radar Signal Processing with AI — Student Projects

Graduate-level course projects covering radar signal processing and deep learning.
Each project provides a complete pipeline: **synthetic data generation → model training → quantitative evaluation**, all runnable on CPU.


## Source-of-truth boundary

This repository is the canonical **technical / runnable project** repository:
project code, data generators, model definitions, training scripts, smoke tests,
and project README files should live here.

The sibling `grad-radar-ai` repository is the canonical **lecture-site** surface:
week pages, course narrative, deployed HTML, and committed lecture figures live
there. P03 and P04 are explicit lecture-canonical exceptions: their assignment
narrative is governed by `grad-radar-ai` Week 12 / Week 13, while runnable code
here must either be aligned to that narrative or clearly marked as archive.
See [`docs/project-source-map.md`](docs/project-source-map.md) before copying or
editing P01–P04 material across repositories.

---

## 🔰 처음 시작하기 (Step-by-Step)

코딩이나 Git을 처음 접하는 학생을 위한 단계별 안내입니다.

### Step 0: 필요한 프로그램 설치

| 프로그램 | Windows 설치 | 설치 확인 (터미널) |
|---------|-------------|-------------------|
| Python 3.10+ | [python.org](https://www.python.org/downloads/)에서 다운로드. **설치 시 "Add Python to PATH" 체크 필수** | `python --version` |
| Git | [git-scm.com](https://git-scm.com/download/win)에서 다운로드. 기본 옵션으로 설치 | `git --version` |
| pip | Python 설치 시 자동 포함 | `pip --version` |

> **Windows 팁:** 설치 후 **명령 프롬프트(cmd)** 또는 **PowerShell**을 열어 확인하세요.
> 시작 메뉴에서 "cmd" 또는 "PowerShell"을 검색하면 됩니다.
> Python을 설치했는데 `python`이 안 되면 `py` 또는 `python3`를 시도하세요.

### Step 1: 레포지토리 다운로드 (git clone)

```bash
# Windows: 명령 프롬프트(cmd) 또는 PowerShell을 열고 아래 명령어를 복사-붙여넣기하세요
cd %USERPROFILE%\Desktop
git clone https://github.com/remilab-cnu/radar-ai-projects.git
cd radar-ai-projects
```

> **Windows 참고:** `cd %USERPROFILE%\Desktop`은 바탕화면으로 이동합니다.
> 원하는 폴더가 있으면 거기서 실행해도 됩니다 (예: `cd C:\Users\내이름\Documents`).

> **`git clone`이란?** GitHub에 있는 코드를 내 컴퓨터로 복사하는 명령어입니다.
> 한 번만 실행하면 됩니다. 이후에는 `cd radar-ai-projects`로 들어가기만 하면 됩니다.

### Step 2: 라이브러리 설치

```bash
pip install -r requirements.txt
```

> 이 명령어는 프로젝트에 필요한 Python 패키지(numpy, torch 등)를 자동으로 설치합니다.
> 처음 한 번만 실행하면 됩니다.

### Step 3: 프로젝트 폴더로 이동

```bash
# 예시: P05 Neural CFAR 프로젝트
cd projects/p05_neural_cfar

# 폴더 내용 확인
ls
# → README.md  generate_data.py  model.py  train.py
```

> 각 프로젝트 폴더에는 항상 같은 4개 파일이 있습니다:
> - `README.md` — 프로젝트 설명
> - `generate_data.py` — 학습 데이터 생성 (시뮬레이터)
> - `model.py` — 신경망 모델 정의
> - `train.py` — 학습 + 평가 실행

### Step 4: 빠른 동작 확인 (Smoke Test)

```bash
# 데이터 생성 + 학습 2 에폭 (1~2분 소요)
python train.py --generate --smoke
```

> `--smoke`는 아주 작은 데이터로 빠르게 동작만 확인하는 모드입니다.
> 에러 없이 끝나면 성공!

### Step 5: 본격 학습

```bash
# 전체 데이터 생성 + 30 에폭 학습 (5~10분)
python train.py --generate --epochs 30
```

### Step 6: 결과 확인

```bash
# 학습 결과는 artifacts/ 폴더에 저장됩니다
ls artifacts/
# → best_model.pt  metrics.json 또는 eval_results.json  history.json

# 결과 수치 확인
cat artifacts/metrics.json
```

### Step 7: 프로젝트 가이드 읽기

각 프로젝트의 상세한 코드 설명과 실험 가이드는 `docs/guides/` 폴더에 있습니다.

```bash
# 브라우저에서 열기 (HTML 파일)
# Windows: 파일 탐색기에서 docs/guides/ 폴더를 열고 HTML 파일을 더블클릭
# Mac/Linux: open docs/guides/p05_guide.html
```

### 자주 하는 실수와 해결법

| 문제 | 원인 | 해결 |
|------|------|------|
| `command not found: git` 또는 `'git'은(는) 인식할 수 없는 명령입니다` | Git 미설치 | [git-scm.com](https://git-scm.com/download/win)에서 설치 후 터미널 재시작 |
| `ModuleNotFoundError: No module named 'torch'` | 패키지 미설치 | `pip install -r requirements.txt` 재실행 |
| `FileNotFoundError: data/train.h5` | 데이터 미생성 | `--generate` 플래그 추가: `python train.py --generate --smoke` |
| `Permission denied` | 권한 문제 | `chmod +x train.py` 또는 `python3 train.py ...`로 실행 |
| `No such file or directory` | 잘못된 디렉토리 | `pwd`로 현재 위치 확인, `cd projects/p05_neural_cfar`로 이동 |

---

## Projects

| # | Title | Topic | Params | Source boundary |
|---|-------|-------|--------|-----------------|
| P1 | U-Net FMCW Detector | Target detection on RDM | 7.7M | Technical canonical here; lecture guide in `grad-radar-ai` |
| P2 | ResNet-18 HAR | Human activity classification | 11.1M | Technical canonical here; lecture guide in `grad-radar-ai` |
| P3 | Deprecated covariance DeepMUSIC archive | Retired DoA-only covariance experiment | 19.3M | Not current P03; current RAM/OGM lecture contract is canonical in `grad-radar-ai` |
| P4 | DnCNN-SAR Despeckling | SAR image restoration | 556K | Runnable companion here; lecture contract canonical in `grad-radar-ai` |
| P5 | Neural CFAR | Learned detection threshold | 28K | Technical canonical here; FMCW + clutter |
| P6 | I/Q Imbalance Correction | RF front-end compensation | 133K | Technical canonical here; FMCW |
| P7 | Full-Duplex SIC | Self-interference cancellation | 302K | Technical canonical here; waveform-consistent chirp echo + FIR SI |
| P8 | Jammer Null Steering | Adaptive beamforming | 73K | Technical canonical here; DoA/array |
| P9 | RD Super-Resolution | Physical LR/HR Range-Doppler mapping | 121K | Technical canonical here; FMCW |

> **P1–P2** are lecture-use examples whose runnable technical source is this repo.
> **P3** covariance-input DeepMUSIC is retained only as a deprecated archive until
> the Week 12 RAM/OGM contract from `grad-radar-ai` is regenerated here.
> **P4** remains a runnable DnCNN-SAR companion, but the lecture/assignment
> narrative is canonical in `grad-radar-ai`.
> **P5–P9** are student project templates (CPU-friendly, <302K parameters).
> **P7 physics contract:** `isr_db` is SI-to-echo power ratio (`P_si/P_echo`),
> `sir_db` is retained as a legacy alias, `snr_db` is echo-to-noise, and the
> target echo is a delayed/Doppler-scaled copy of `tx_ref` rather than an
> independent tone.

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

All projects use synthetic educational simulators or approximations. No external
datasets are needed; the examples are designed for reproducible teaching rather
than high-fidelity sensor validation.

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
| `vel_res` | λ / (2·N_chirps·T) | 0.304 m/s |
| `max_vel` | λ / (4·T) | 19.5 m/s |
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

Most learning datasets keep only the positive range-frequency half
(`N_samples // 2`) after the FFT. With the default 500 ADC samples this gives
250 range bins, so a default single-antenna RDM tensor is typically
`(128 Doppler bins, 250 range bins)`.

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

**Output**: Sample covariance `R̂` (2-channel real/imag, Frobenius-normalized) + Gaussian angle-heatmap label.

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

> Note: P06 and P08 use custom training loops due to multi-input/multi-output architectures. See their `train.py` for documented reasons.

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
│   ├── p03_deepmusic_cnn/     Deprecated covariance DeepMUSIC archive
│   ├── p04_dncnn_sar/         DnCNN SAR despeckling companion
│   ├── p05_neural_cfar/       Neural CFAR detection
│   ├── p06_iq_imbalance/      I/Q imbalance correction
│   ├── p07_full_duplex_sic/   Full-duplex self-interference cancellation
│   ├── p08_jammer_nulling/    Jammer null steering
│   ├── p09_rd_superres/       Range-Doppler super-resolution
│       ├── README.md              Project description & instructions
│       ├── generate_data.py       Synthetic data generation → data/*.h5
│       ├── model.py               PyTorch model definition
│       └── train.py               Train + evaluate entry point
├── scripts/
│   └── smoke_all.py           Smoke test all 9 projects
└── requirements.txt           Python dependencies
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CPU)
- NumPy, SciPy, h5py, matplotlib, scikit-learn, tqdm

No GPU required. Student projects (P5–P9) train in under 10 minutes on an 8-core CPU.

## Course

REMI Lab, Chungnam National University
Dept. of Information Communication Convergence Engineering
