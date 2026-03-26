# Radar Signal Processing with AI — Student Projects

Graduate-level course projects covering radar signal processing and deep learning.
Each project provides a complete pipeline: **synthetic data generation → model training → quantitative evaluation**, all runnable on CPU.

## Projects

| # | Title | Topic | Difficulty | Simulator |
|---|-------|-------|-----------|-----------|
| P1 | U-Net FMCW Detector | Target detection | ★★☆ | FMCW + clutter |
| P2 | ResNet-18 HAR | Activity classification | ★★☆ | micro-Doppler |
| P3 | DeepMUSIC CNN | Direction-of-Arrival | ★★☆ | DoA/array |
| P4 | DnCNN-SAR Despeckling | SAR image restoration | ★★☆ | SAR |
| P5 | Neural CFAR | Learned detection threshold | ★☆☆ | FMCW + clutter |
| P6 | I/Q Imbalance Correction | RF front-end compensation | ★☆☆ | FMCW |
| P7 | Full-Duplex SIC | Self-interference cancellation | ★☆☆ | FMCW |
| P8 | Jammer Null Steering | Adaptive beamforming | ★☆☆ | DoA/array |
| P9 | RD Super-Resolution | Range-Doppler enhancement | ★★☆ | FMCW |
| P10 | Near-Field Localization | Spherical-wave source finding | ★★☆ | DoA/array |

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
```

## Repository Structure

```
radar-ai-projects/
├── shared/              # Radar simulators (FMCW, SAR, micro-Doppler, DoA, clutter)
├── common/              # Shared training utilities (CLI, HDF5, training loop, metrics)
├── projects/
│   ├── p01_unet_detector/
│   ├── p02_resnet18_har/
│   ├── p03_deepmusic_cnn/
│   ├── p04_dncnn_sar/
│   ├── p05_neural_cfar/
│   ├── p06_iq_imbalance/
│   ├── p07_full_duplex_sic/
│   ├── p08_jammer_nulling/
│   ├── p09_rd_superres/
│   └── p10_nearfield_loc/
│       ├── README.md         # Project description & instructions
│       ├── generate_data.py  # Synthetic data generation → data/*.h5
│       ├── model.py          # PyTorch model definition
│       └── train.py          # Train + evaluate (--generate --smoke --eval_only)
└── scripts/
    └── smoke_all.py     # Run smoke tests for all projects
```

## Standard CLI

Every `train.py` supports:

| Flag | Description |
|------|-------------|
| `--generate` | Generate HDF5 datasets before training |
| `--smoke` | Quick smoke test (tiny data, 2 epochs) |
| `--epochs N` | Training epochs (default: 30) |
| `--batch_size N` | Batch size (default: 64) |
| `--lr F` | Learning rate (default: 1e-3) |
| `--eval_only` | Skip training, evaluate only |
| `--checkpoint PATH` | Load model checkpoint |
| `--seed N` | Random seed (default: 42) |

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CPU)
- NumPy, SciPy, h5py, matplotlib, scikit-learn, tqdm

No GPU required. All projects train in under 30 minutes on an 8-core CPU.

## Course

REMI Lab, Chungnam National University
Dept. of Information Communication Convergence Engineering
