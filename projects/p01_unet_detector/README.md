# P01: U-Net Radar Detector

Pixel-wise target detection on FMCW Range-Doppler Maps using a U-Net segmentation network.

## Task

Given a 2-channel RDM (log-magnitude + phase), predict a binary target mask.
Evaluated against CA-CFAR as a classical baseline.

## Architecture

- **Model**: UNetDetector — 5-stage encoder/decoder with skip connections
- **Input**: `(B, 2, 128, 128)` — RDM log-magnitude + phase
- **Output**: `(B, 1, 128, 128)` — detection probability map
- **Loss**: FocalDiceLoss (focal: alpha=0.75, gamma=2.0; dice weight=0.5)
- **Parameters**: ~7.7M (base_ch=32)

## Commands

```bash
# Generate data + train (default: 50K train, 30 epochs)
python train.py --generate --epochs 30

# Smoke test (256/64/64 samples, 2 epochs, CPU)
python train.py --generate --smoke

# Eval only
python train.py --eval_only --checkpoint artifacts/best_model.pt

# Custom sizes
python train.py --generate --n_train 10000 --epochs 50 --batch_size 32
```

## Data

Generated via `generate_data.py` using `shared/fmcw_simulator.py` and `shared/clutter_model.py`.

| Split | Filename | Default size |
|-------|----------|-------------|
| Train | `data/det_train.h5` | 50K |
| Val | `data/det_val.h5` | 5K |
| Test | `data/det_test.h5` | 5K |

HDF5 schema: `x (N,2,128,128)`, `y (N,1,128,128)`, `snr_db (N,)`, `n_targets (N,)`

## Metrics

| Metric | Description |
|--------|-------------|
| Pd | Probability of detection (pixel-level recall) |
| Pfa | Probability of false alarm |
| Precision | Pixel-level precision |
| F1 | Harmonic mean of Pd and Precision |

Results saved to `artifacts/eval_results.json`.
