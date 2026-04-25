# P03 Archive: Covariance-Input DeepMUSIC CNN

> **Deprecated / archive status.** This covariance-input DoA experiment is not
> the current P03 assignment. Current P03 is lecture-canonical in
> `grad-radar-ai` Week 12 and uses a RAM(range-angle map)-only / OGM-oriented
> contract. Keep this folder only as historical array-processing reference until
> a runnable RAM/OGM P03 is regenerated in this repository.

Direction-of-Arrival (DoA) estimation using a CNN trained on covariance
matrices, compared against classical MUSIC in the provided evaluation.

## Current P03 source of truth

- Canonical lecture/design material: sibling repo `grad-radar-ai/weeks/week12/`
  and `grad-radar-ai/projects/p3-deepmusic/`.
- This folder: retired covariance-input DeepMUSIC baseline.
- Do not describe this folder as the active P03 project in new guides or
  lecture pages.

## Task

Given a 2-channel real/imaginary sample covariance matrix from a ULA, regress an
angle heatmap over the angle grid [-90, 90] degrees.

## Architecture

- **Model**: DeepMUSIC -- 4-layer CNN encoder + FC head
- **Input**: `(B, 2, N_rx, N_rx)` -- real/imag covariance
- **Output**: `(B, 181)` -- Gaussian angle heatmap (sigmoid, values in [0,1])
- **Loss**: BCELoss
- **Parameters**: ~19.3M (N_rx=12)

## Commands

```bash
# Generate data + train (default: 100K train, 30 epochs)
python train.py --generate --epochs 30

# Smoke test (256/64/64 samples, 2 epochs, CPU)
python train.py --generate --smoke

# Eval only
python train.py --eval_only --checkpoint artifacts/best_model.pt

# Smaller dataset for quick iteration
python train.py --generate --n_train 10000 --epochs 50
```

## Data

Generated via `generate_data.py` using `shared/doa_utils.py`.

| Split | Filename | Default size |
|-------|----------|-------------|
| Train | `data/doa_train.h5` | 100K |
| Val | `data/doa_val.h5` | 20K |
| Test | `data/doa_test.h5` | 20K |

HDF5 schema: `covariance (N,2,12,12)`, `spectrum (N,181)`, `snr_db`, `n_sources`, `n_snapshots`, `coherent`, `angles`

## Metrics

| Metric | Description |
|--------|-------------|
| dnn_rmse_mean | Mean RMSE in degrees (DeepMUSIC) |
| dnn_rmse_median | Median RMSE in degrees (DeepMUSIC) |
| music_rmse_mean | Mean RMSE in degrees (classical MUSIC) |

Results saved to `artifacts/eval_results.json`.
