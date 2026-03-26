# P04: DnCNN-SAR Despeckling

SAR image despeckling using DnCNN residual learning in the log-intensity domain.

## Task

Given a speckle-corrupted SAR log-intensity image, predict the clean image.
Evaluated against classical Lee, Frost, and Median filters.

## Architecture

- **Model**: DnCNN-SAR -- 17-layer residual CNN (predict noise, subtract from input)
- **Input**: `(B, 1, 256, 256)` -- log-intensity SAR patch normalized to [0, 1]
- **Output**: `(B, 1, 256, 256)` -- despeckled log-intensity image
- **Loss**: DespecklingLoss (Charbonnier w=0.8 + SSIM w=0.2)
- **Parameters**: ~556K

## Commands

```bash
# Generate data + train (default: 25K train, 30 epochs)
python train.py --generate --epochs 30

# Smoke test (256/64/64 samples, 2 epochs, CPU)
python train.py --generate --smoke

# Eval only
python train.py --eval_only --checkpoint artifacts/best_model.pt

# Custom sizes
python train.py --generate --n_train 5000 --epochs 60 --batch_size 8
```

## Data

Generated via `generate_data.py` using `shared/sar_simulator.py`.
Simulated stripmap SAR scenes with random point targets, 1-5 looks speckle.

| Split | Filename | Default size |
|-------|----------|-------------|
| Train | `data/despeckling_train.h5` | 25K |
| Val | `data/despeckling_val.h5` | 5K |
| Test | `data/despeckling_test.h5` | 5K |

HDF5 schema: `noisy (N,1,256,256)`, `clean (N,1,256,256)`, `n_looks (N,)`, `n_targets (N,)`

## Metrics

| Metric | Description |
|--------|-------------|
| dncnn_psnr | Peak Signal-to-Noise Ratio (dB, higher is better) |
| dncnn_ssim | Structural Similarity Index (higher is better) |
| lee/frost/median_psnr | Classical filter baselines |

Results saved to `artifacts/eval_results.json`.
