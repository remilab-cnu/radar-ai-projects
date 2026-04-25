# P01: U-Net Radar Detector

Pixel-wise target detection on FMCW Range-Doppler Maps using a U-Net segmentation network.
This project is the runnable source of truth for Week 10 lecture material.

## Task

Given a 2-channel RDM (log-magnitude + phase), predict a binary target mask.
Evaluated against CA-CFAR as a classical baseline.  The baseline must run
on the saved linear RDM magnitude (`rdm_mag_linear`), not the normalized
display channel used as the neural-network input.

## Architecture

- **Model**: UNetDetector — 5-stage encoder/decoder with skip connections
- **Input**: `(B, 2, 128, 250)` by default — RDM log-magnitude + phase
- **Output**: `(B, 1, 128, 250)` by default — detection probability map
- **Loss**: FocalDiceLoss (focal: alpha=0.75, gamma=2.0; dice weight=0.5)
- **Parameters**: ~7.7M (base_ch=32)

## Commands

```bash
# Generate data + train (default: 50K train, 30 epochs)
python train.py --generate --epochs 30

# Smoke test (256/64/64 samples, 2 epochs, CPU)
python train.py --generate --smoke --base_ch 8

# Eval only
python train.py --eval_only --checkpoint artifacts/best_model.pt
# If the checkpoint was created by the CPU smoke command above, add: --base_ch 8

# Verify the data contract and compare operating-point policies
python analyze_data_contract.py --data_dir data --split val --out_dir artifacts/verified_p01
python evaluate_cfar.py --data_dir data --split val --sweep --out artifacts/verified_p01/p01_cfar_sweep_val.json
python evaluate_cfar.py --data_dir data --split test --policy-from artifacts/verified_p01/p01_cfar_sweep_val.json --out artifacts/verified_p01/p01_cfar_selected_test.json
python evaluate_unet.py --data_dir data --checkpoint artifacts/best_model.pt --split val --sweep --out artifacts/verified_p01/p01_unet_threshold_sweep_val.json
python evaluate_unet.py --data_dir data --checkpoint artifacts/best_model.pt --split test --policy-from artifacts/verified_p01/p01_unet_threshold_sweep_val.json --out artifacts/verified_p01/p01_unet_selected_test.json
# Add --base_ch 8 to both evaluate_unet.py commands when replaying a smoke checkpoint.
python make_verified_figures.py --artifacts artifacts/verified_p01

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

HDF5 schema v2:
`x (N,2,128,250)`, `y (N,1,128,250)`, `rdm_mag_linear (N,128,250)`,
`snr_db (N,)`, `n_targets (N,)`, `clutter_power_db (N,)`,
target metadata (`target_*`) and range/velocity axes.

The range dimension is 250 because the simulator uses 500 fast-time ADC
samples and keeps the positive range-frequency half after the FFT.

## Metrics

| Metric | Description |
|--------|-------------|
| Pd | Probability of detection (pixel-level recall) |
| Pfa | Probability of false alarm |
| Precision | Pixel-level precision |
| F1 | Harmonic mean of Pd and Precision |

Results saved to `artifacts/eval_results.json`.
Verified lecture/report artifacts should also include the validation-selected
CFAR policy, validation-selected neural threshold, and held-out test metrics.
