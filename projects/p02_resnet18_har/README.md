# P02: ResNet-18 Micro-Doppler HAR

Human activity recognition from micro-Doppler spectrograms using ResNet-18.

## Task

Classify 6 human activities from 128x128 micro-Doppler time-frequency
spectrograms generated from a local range-compressed target response. The
generator expands the Boulic body model into a P02-only,
radar-deconv-inspired pedestrian scatterer model, builds a compact
range-compressed frame around the human target range, extracts the complex
slow-time signal at the simulator-known target range, then computes the
Doppler/STFT spectrogram from that selected range. This supersedes the earlier
range-free shortcut implementation.

This is controlled educational data, not a general pedestrian radar benchmark:
the default aspect angle is sampled as an **absolute** sector `[0°, 60°]`.
The sign of aspect is intentionally not sampled by default because the current
2-D radial-projection model uses `cos(aspect)`, so `+θ` and `-θ` are physically
equivalent in this teaching simulator.  The generator records the slow-time
aperture metadata so the 64-chirp radar configuration is not confused with the
longer HAR observation used for STFT.  The default `run` kinematics are
intentionally bounded so scatterer radial velocities remain below the Doppler
Nyquist limit for the 77 GHz / 10 kHz slow-time PRF teaching setup.

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
python train.py --epochs 30 --model tiny_cnn --artifact_dir artifacts/tiny_cnn_default

# Smoke test (256/64/64 samples, 2 epochs, CPU)
python train.py --generate --smoke

# Eval only
python train.py --eval_only --checkpoint artifacts/best_model.pt

# Handcrafted micro-Doppler descriptor baseline
python evaluate_feature_baseline.py --data_dir data --model logreg
python evaluate_feature_baseline.py --data_dir data --model rbf_svm --max_train 10000

# Custom SNR range
python train.py --generate --snr_lo 5 --snr_hi 25 --epochs 60

# Held-out aspect/range variation set
python generate_data.py --n_train 0 --n_val 600 --n_test 600 \
  --aspect_lo 60 --aspect_hi 80 --range_lo 18 --range_hi 26 \
  --out_dir data_heldout_aspect_range
```

## Data

Generated via `generate_data.py` using `shared/micro_doppler.py` with shared
radar parameters:

```text
body kinematics → P02 pedestrian scatterers
                → local range-compressed frame
                → slow-time signal at target range
                → Doppler/STFT spectrogram
```

Balanced classes (equal samples per activity).

| Split | Filename | Default size |
|-------|----------|-------------|
| Train | `data/har_train.h5` | 30K |
| Val | `data/har_val.h5` | 3K |
| Test | `data/har_test.h5` | 3K |

HDF5 schema: `x (N,1,128,128)`, `y (N,)` labels, `features (N,F)` handcrafted,
`snr_db (N,)`, `range_m (N,)`, `aspect_angle_deg (N,)`,
`target_range_bin (N,)`, `target_range_m (N,)`, slow-time metadata
(`slow_time_prf_hz`, `slow_time_samples`), aspect convention
(`aspect_convention`), Doppler-alias guard metadata
(`max_abs_radial_velocity_mps`, `radar_max_unambiguous_velocity_mps`,
`doppler_alias_margin_mps`), scatterer counts (`n_scatterers`,
`torso_scatterers`, `head_scatterers`, `limb_scatterers`), `range_axis_m`, plus
radar metadata (`radar_*`, `radar_pri_s`, `radar_config_n_chirps`,
`fs_over_bandwidth`, `aspect_angle_range_deg`, `scatter_model`,
`range_processing`, `doppler_source`, `schema_version`).

## Metrics

| Metric | Description |
|--------|-------------|
| accuracy | Overall classification accuracy |
| per_class | Per-activity accuracy breakdown |
| confusion_matrix | Class confusion matrix from ResNet or feature baseline |

ResNet results are saved to `artifacts/eval_results.json`; handcrafted-feature
baseline results are saved to `artifacts/feature_baseline_results.json`.
