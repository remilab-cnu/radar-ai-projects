# P02: ResNet-18 Micro-Doppler HAR

Human activity recognition from micro-Doppler spectrograms using ResNet-18.

## Task

Classify 6 human activities from 128x128 micro-Doppler time-frequency spectrograms
generated with a 77 GHz FMCW radar simulator.

## Architecture

- **Model**: ResNetHAR — ResNet-18 with single-channel input
- **Input**: `(B, 1, 128, 128)` — micro-Doppler spectrogram
- **Output**: `(B, 6)` — class logits
- **Loss**: CrossEntropyLoss with label smoothing=0.1
- **Parameters**: ~11.2M

## Classes

`walk`, `run`, `sit_down`, `fall`, `wave`, `idle`
(defined in `shared/micro_doppler.py:ACTIVITY_LABELS`)

## Commands

```bash
# Generate data + train (default: 30K train, 30 epochs)
python train.py --generate --epochs 30

# Smoke test (256/64/64 samples, 2 epochs, CPU)
python train.py --generate --smoke

# Eval only
python train.py --eval_only --checkpoint artifacts/best_model.pt

# Custom SNR range
python train.py --generate --snr_lo 5 --snr_hi 25 --epochs 60
```

## Data

Generated via `generate_data.py` using `shared/micro_doppler.py`.
Balanced classes (equal samples per activity).

| Split | Filename | Default size |
|-------|----------|-------------|
| Train | `data/har_train.h5` | 30K |
| Val | `data/har_val.h5` | 3K |
| Test | `data/har_test.h5` | 3K |

HDF5 schema: `x (N,1,128,128)`, `y (N,)` labels, `features (N,F)` handcrafted, `snr_db (N,)`

## Metrics

| Metric | Description |
|--------|-------------|
| accuracy | Overall classification accuracy |
| per_class | Per-activity accuracy breakdown |

Results saved to `artifacts/eval_results.json`.
