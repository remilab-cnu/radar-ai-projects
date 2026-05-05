# P01: U-Net Radar Detector

Pixel-wise target detection on FMCW Range-Doppler Maps using a U-Net segmentation network.
This project is the runnable source of truth for Week 10 lecture material.

## Task

Given a 2-channel RDM (log-magnitude + phase), predict a binary target mask.
Evaluated against CA-CFAR as a classical baseline.  The baseline must run
on the saved linear RDM magnitude (`rdm_mag_linear`), not the normalized
display-normalized channel used as the neural-network input.

## Architecture

- **Model**: UNetDetector — 5-stage encoder/decoder with skip connections
- **Input**: `(B, 2, Nd, Nr)` — RDM log-magnitude + phase
- **Output**: `(B, 1, Nd, Nr)` — detection probability map
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

# Lecture ablations: network capacity and input representation
python train.py --epochs 30 --base_ch 16 --artifact_dir artifacts/unet_base16
python train.py --epochs 30 --input_mode mag_only --artifact_dir artifacts/unet_mag_only

# Verify the data format and compare detector settings
python analyze_data_contract.py --data_dir data --split val --out_dir artifacts/verified_p01
python evaluate_cfar.py --data_dir data --split val --sweep --out artifacts/verified_p01/p01_cfar_sweep_val.json
python evaluate_cfar.py --data_dir data --split test --policy-from artifacts/verified_p01/p01_cfar_sweep_val.json --out artifacts/verified_p01/p01_cfar_selected_test.json
python evaluate_unet.py --data_dir data --checkpoint artifacts/best_model.pt --split val --sweep --out artifacts/verified_p01/p01_unet_threshold_sweep_val.json
python evaluate_unet.py --data_dir data --checkpoint artifacts/best_model.pt --split test --policy-from artifacts/verified_p01/p01_unet_threshold_sweep_val.json --out artifacts/verified_p01/p01_unet_selected_test.json
# Add --base_ch 8 to both evaluate_unet.py commands when replaying a smoke checkpoint.
# Add --input_mode mag_only when evaluating a mag-only ablation checkpoint.
python make_verified_figures.py --artifacts artifacts/verified_p01

# Custom sizes
python train.py --generate --n_train 10000 --epochs 50 --batch_size 32
```

## Data

Generated via `generate_data.py` using `shared/fmcw_simulator.py` and `shared/clutter_model.py`.
The active clutter lane is **static-only**: clutter scatterers have exactly
zero radial velocity, keeping the introductory task focused on moving-target
RDM detection after MTI preprocessing.  Raw beat data is quantized first, then a slow-time
mean-removal MTI/DC notch is applied before RDM generation.  CA-CFAR and U-Net
therefore see the same MTI-filtered RDM.
The balanced schema-v9 generator labels only targets whose realized RDM peak is
at least `min_label_snr_db = 6 dB` above the stricter of the global RD median
noise floor and a local CFAR-like background ring.  Default positive labels also
exclude near-zero-Doppler targets (`|v| >= 2 × velocity_resolution`) and cap the
teaching target-count range to `[1, 8]`, so the Week 10 dataset teaches
moving-target RDM detection before static-clutter cancellation. Sub-threshold
targets and static clutter can still be present in the raw scene as hard
negatives, but they are not counted as positive labels because they are
effectively buried in the RD map background.

Raw beat data is quantized before RDM processing as complex 16-bit I/Q: signed
int16 I plus signed int16 Q components under a fixed P1 full-scale
(`6.0e-5`). The HDF5
dataset does not store raw I/Q by default; it stores processed `x` as `float16`,
`y` masks as `uint8`, and keeps `rdm_mag_linear` as `float32` for CFAR.

| Split | Filename | Default size |
|-------|----------|-------------|
| Train | `data/det_train.h5` | 50K |
| Val | `data/det_val.h5` | 5K |
| Test | `data/det_test.h5` | 5K |

HDF5 schema:
`x (N,2,Nd,Nr)`, `y (N,1,Nd,Nr)`, `rdm_mag_linear (N,Nd,Nr)`,
`snr_db (N,)`, `n_targets (N,)`, `clutter_power_db (N,)`,
`adc_clipped_fraction (N,)`, `min_label_snr_db`, target metadata (`target_*`, including
`target_actual_snr_db`, `target_peak_snr_db`, `target_global_peak_snr_db`,
`target_local_bg_floor`, and `target_effective_bg_floor`) and range/velocity
axes.  Schema metadata also records `mti_applied` and `mti_mode`.
The active schema version is v9.

The exact range/Doppler dimensions follow the approved shared FMCW
dechirp/mixing configuration. Active data includes radar metadata
(`radar_*`, `fs_over_bandwidth`) so regenerated artifacts can be tied back to
the simulator contract.

The compact storage dtypes are recorded in `x_storage_dtype` and
`y_storage_dtype`; the receiver quantization contract is recorded in
`adc_iq_bits`, `adc_iq_full_scale`, `adc_iq_component_dtype`, and
`adc_clipped_fraction`.

## Metrics

| Metric | Description |
|--------|-------------|
| Pd | Probability of detection (pixel-level recall) |
| Pfa | Probability of false alarm |
| Precision | Pixel-level precision |
| F1 | Harmonic mean of Pd and Precision |
| target_recall | Fraction of simulator-labelled targets detected within ±1 RD bin |
| false_alarms_per_rdm | False-positive RD cells per RDM sample |

Results saved to `artifacts/eval_results.json`.
Verified lecture/report artifacts should also include the validation-selected
CFAR settings, validation-selected neural threshold, and final test-split metrics.
