# Radar Signal Processing with AI — Student Projects

Graduate teaching repository for hands-on radar signal processing and AI
projects.  The public course release contains six runnable projects, each with
its own data path, model, training script, and evaluation tools.

## What is included

| Project | Folder | Topic | Main idea |
|---|---|---|---|
| P01 | `projects/p01_unet_detector/` | FMCW Range-Doppler detection | Build MTI-filtered Range-Doppler maps and compare U-Net target masks with CA-CFAR. |
| P02 | `projects/p02_resnet18_har/` | Micro-Doppler HAR | Classify six human activities from target-range micro-Doppler spectrograms. |
| P03 | `projects/p03_radar_cube_doa/` | DoA and mapping | Compare angle FFT, MUSIC, and RadarCubeDoANet by projecting detections into maps. |
| P04 | `projects/p04_dncnn_sar/` | SAR despeckling | Train/evaluate DnCNN-SAR on real Sentinel-1 image patches. |
| P05 | `projects/p05_waveform_classification/` | Lightweight waveform classification | Classify radar waveform families from STFT images using MATLAB-reference-style synthetic examples. |
| P06 | `projects/p06_target_signature_classification/` | Lightweight target signature classification | Classify simple target signatures from angle-dependent RCS-like point-scatterer returns. |

The projects are designed for reproducible classroom experiments.  They are not
claims of operational radar-system performance.

---

## Quick start

### 1. Install tools

| Tool | Check command |
|---|---|
| Python 3.10+ | `python --version` |
| Git | `git --version` |
| pip | `pip --version` |

On Windows, check **Add Python to PATH** during Python installation, then open a
new PowerShell or Command Prompt.

### 2. Clone and install packages

```bash
git clone https://github.com/remilab-cnu/radar-ai-projects.git
cd radar-ai-projects
pip install -r requirements.txt
```

### 3. Run one project smoke test

```bash
cd projects/p01_unet_detector
python train.py --generate --smoke
```

`--smoke` uses a small dataset and short training schedule so students can check
that the code path works before launching longer experiments.

### Common issues

| Symptom | Likely cause | Fix |
|---|---|---|
| `git` or `python` not found | Tool is not installed or PATH is not refreshed | Install the tool and open a new terminal |
| `ModuleNotFoundError` | Python packages are missing | Run `pip install -r requirements.txt` from the repo root |
| `FileNotFoundError: data/*.h5` | Dataset has not been generated | Add `--generate` or run the project data-generation command |
| P04 cannot find SAR source data | Sentinel-1 data is not mounted locally | Set `P04_SAR_DATA_ROOT` or pass `--data_root` |
| P05/P06 checkpoint shape mismatch | Smoke checkpoints use smaller `base_ch` | Add `--base_ch 8` when evaluating a smoke checkpoint |

---

## Smoke-test commands

Run a project smoke test before launching a longer experiment:

```bash
# P01
cd projects/p01_unet_detector
python train.py --generate --smoke

# P02
cd ../p02_resnet18_har
python train.py --generate --smoke

# P03
cd ../p03_radar_cube_doa
python train.py --mapping --generate --smoke

# P04
cd ../p04_dncnn_sar
python train.py --generate --smoke

# P05
cd ../p05_waveform_classification
python train.py --generate --smoke

# P06
cd ../p06_target_signature_classification
python train.py --generate --smoke
```

Return to the repository root with `cd ../..` before switching to a new top-level
folder.

---

## Project summaries

### P01 — U-Net FMCW Detector

Processing chain:

```text
FMCW beat scene -> fixed complex 16-bit I/Q -> MTI/DC notch -> Range-Doppler map
                -> CA-CFAR baseline and U-Net mask prediction
```

The active dataset labels moving targets only when the processed target peak is
clearly above both the global and local Range-Doppler background.  This keeps the
lesson focused on moving-target detection after a standard clutter-suppression
step.

```bash
cd projects/p01_unet_detector
python train.py --generate --epochs 30
python train.py --eval_only --checkpoint artifacts/best_model.pt
```

### P02 — ResNet-18 Micro-Doppler HAR

P02 generates six-class human activity spectrograms from a target-range
micro-Doppler pipeline:

```text
body kinematics -> pedestrian scatterers -> local range-compressed frame
                -> target-range slow-time signal -> STFT spectrogram
```

```bash
cd projects/p02_resnet18_har
python train.py --generate --epochs 30
python evaluate_feature_baseline.py --data_dir data --model rbf_svm --max_train 10000
```

### P03 — Radar Mapping via DoA

P03 compares direction-of-arrival methods by their downstream map quality.  Each
method receives the same selected antenna vector from the same Range-Doppler
cell, then the estimated DoA is projected into point-cloud and occupancy-grid
maps.

```bash
cd projects/p03_radar_cube_doa
python train.py --mapping --generate --smoke
```

### P04 — DnCNN-SAR Despeckling

P04 uses real Sentinel-1 GRD/SLC patches in normalized log/dB magnitude.  The
main learning task is to map speckled SAR patches to pseudo-clean multi-look
references and compare against classical filters.

```bash
cd projects/p04_dncnn_sar
python train.py --generate --smoke
python train.py --eval_only --checkpoint artifacts/best_model.pt
```

Full P04 experiments require instructor-provided Sentinel-1 data and are usually
run in a GPU environment.

### P05 — Lightweight Radar Waveform Classification Example

P05 follows MATLAB radar waveform-classification examples at a compact scale.
It generates rectangular, LFM, Barker-coded, and noise-only baseband observations,
converts them to STFT log-magnitude images, and compares a tiny CNN with
handcrafted waveform descriptors.

```bash
cd projects/p05_waveform_classification
python train.py --generate --smoke
python evaluate_snr_sweep.py --checkpoint artifacts/best_model.pt --base_ch 8
```

### P06 — Lightweight Target Signature Classification Example

P06 follows MATLAB radar target-classification examples at a compact scale.
It generates angle-dependent RCS-like signatures from simple point-scatterer
target geometries, then compares a 1-D CNN with handcrafted return descriptors.
The input is a magnitude/phase sequence from a complex monostatic return, not
a SAR image or range-Doppler map.

```bash
cd projects/p06_target_signature_classification
python train.py --generate --smoke
python evaluate_generalization.py --generate --checkpoint artifacts/best_model.pt --base_ch 8 --smoke
```

---

## Shared simulator modules

| Module | Role |
|---|---|
| `shared/fmcw_simulator.py` | Complex-baseband FMCW dechirp simulator, Range FFT, Doppler FFT, angle map, CFAR helpers, fixed-point I/Q helpers |
| `shared/clutter_model.py` | P01 static-clutter and MTI data-generation helpers |
| `shared/micro_doppler.py` | P02 body/scatterer micro-Doppler generator and handcrafted features |
| `shared/burst_simulator.py` | Clean LFM pulse-burst and matched-filter reference simulator |
| `shared/doa_utils.py` | ULA steering vectors, beamforming, MUSIC, MVDR, and DoA metrics |
| `shared/radar_scene.py` | 2-D radar scene and occupancy-grid utilities |
| `shared/sar_simulator.py` | Simplified stripmap SAR teaching utilities |
| `shared/waveform_library.py` | P05 lightweight waveform-family generation and STFT-image helpers |
| `shared/target_signature.py` | P06 lightweight point-scatterer target-signature generation helpers |

The shared FMCW simulator uses complex-baseband chirps and the mixer output
`rx * conj(tx)`.  Carrier frequency is used for wavelength, phase, Doppler,
array steering, and radar-equation power; explicit RF passband samples are not
created.

---

## Repository structure

```text
radar-ai-projects/
├── common/                  shared CLI, HDF5, metrics, training utilities
├── shared/                  reusable radar simulators and DSP helpers
├── projects/
│   ├── p01_unet_detector/
│   ├── p02_resnet18_har/
│   ├── p03_radar_cube_doa/
│   ├── p04_dncnn_sar/
│   ├── p05_waveform_classification/
│   └── p06_target_signature_classification/
├── docs/                    optional reference notes and result summaries
└── requirements.txt
```

Generated datasets, checkpoints, and local reports are ignored by default unless
a project README explicitly tells you to submit or share them.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy, SciPy, h5py, matplotlib, scikit-learn, tqdm

## Course

REMI Lab, Chungnam National University

Department of Information Communication Convergence Engineering
