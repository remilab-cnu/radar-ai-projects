# P01 탐지 실험 정의서 — Week 10 기술 자료

P01 is a controlled FMCW Range-Doppler Map (RDM) target-detection experiment.
It is not a full operational radar receiver. Week 10 lecture claims must be made
against the signal model, data format, and evaluation conditions defined here.

## Signal model and processing assumptions

- Radar: `FMCWRadar(...)` in `projects/p01_unet_detector/generate_data.py`, wired to the shared FMCW dechirp/mixing core.
- Signal model target: beat frequency from two-way delay, Doppler phase over chirps, radar-equation amplitude, and kTB thermal noise (`shared/fmcw_simulator.py`). Schema-v9 full-run artifacts are the active evidence for this contract.
- RDM: range FFT followed by Doppler FFT, using the positive range-frequency half.
- MTI input: active P01 applies a slow-time mean-removal DC notch before RDM
  generation.  This suppresses static clutter returns for the U-Net and for the
  CA-CFAR baseline, keeping both detectors on the same moving-target problem.
- Neural input: 2 channels: noise-floor-referenced log magnitude and phase.
- Baseline input: active datasets store `rdm_mag_linear` before display clipping so CA-CFAR is evaluated on the linear RDM magnitude, not on a reconstructed display image.
- Label gate: a target is labeled only when its processed RDM target-bin peak
  (`target_peak_snr_db`) is at least `min_label_snr_db = 6 dB` above the
  stricter of (a) the measured global RD median noise floor and (b) a local
  CFAR-like background ring around the target. This is intentionally based on
  target-bin detectability rather than global scene maximum or only the
  thermal-SNR model. Targets below this processed detectability gate may still
  be present in the simulated raw scene as hard negatives/clutter-like returns,
  but they are excluded from positive masks.  The 6 dB default is the
  schema-v9 educational setting.
- Target velocity: default generated positive labels require
  `|velocity| >= 2 * velocity_resolution`. This avoids making the introductory
  P01 detector learn ambiguous zero-Doppler targets sitting directly on the
  static clutter ridge before clutter cancellation has been taught.
- Clutter: active P1 clutter is static-only. All clutter scatterers have
  exactly zero radial velocity; multipath ghosts and Doppler-tail clutter are
  not generated.  Static clutter is present in the raw simulated scene but is
  not a semantic target label after MTI preprocessing.
- Receiver quantization: raw beat data is quantized before RDM processing as
  complex 16-bit I/Q under fixed P1 full-scale `6.0e-5`. "Complex 16-bit I/Q"
  means two signed int16 components per complex sample (`I`, `Q`), not a scalar
  `int16` complex dtype. The processed neural input is stored as `float16`;
  masks are stored as `uint8`; `rdm_mag_linear` remains `float32` for CFAR.

## Allowed simplifications

| Simplification | Why allowed for Week 10 | Not claimed |
|---|---|---|
| Radar equation is simplified to point scatterers and synthetic clutter | Focuses the lecture on RDM detection conditions and detector settings | Site-calibrated received power or environment-specific propagation |
| Static clutter only + MTI input | Keeps P1 focused on moving-target detection after a standard clutter-suppression preprocessing step | Moving clutter, multipath ghosts, or site-specific environmental clutter |
| Target mask is a 5-pixel cross around quantized range/Doppler bin | Matches the teaching label definition for Hann-windowed point targets | Semantic object segmentation mask |
| Sub-6 dB processed peak/background targets are not labeled | Avoids training on effectively invisible or near-floor positives below global or local RD background | Exhaustive annotation of every simulated scatterer regardless of detectability |
| Near-zero Doppler positives are excluded by default | Keeps the first detector lesson focused on moving-target RDM detection rather than static-clutter cancellation | General zero-Doppler target detection in heavy clutter |
| Single-receiver RDM is used for P01 detection | Keeps Week 10 focused on range/Doppler detection | Full MIMO angle processing |

## Dataset schema

Required HDF5 datasets:

- `x`: neural input `(N, 2, Nd, Nr)`; exact dimensions follow the approved project wiring.
- `y`: binary detection mask `(N, 1, Nd, Nr)`.
- `rdm_mag_linear`: linear RDM magnitude for CFAR.
- `snr_db`, `n_targets`, `clutter_power_db`, `noise_floor`,
  `adc_clipped_fraction`, `min_label_snr_db`.
- `target_range_m`, `target_velocity_mps`, `target_rcs`,
  `target_actual_snr_db`, `target_peak_snr_db`, `target_global_peak_snr_db`,
  `target_local_bg_floor`, `target_effective_bg_floor`, `target_range_bin`,
  `target_doppler_bin`.
- `range_axis_m`, `velocity_axis_mps`, `schema_version`, and radar metadata
  (`radar_fc_hz`, `radar_bw_hz`, `radar_fs_hz`, `fs_over_bandwidth`).
- quantization/storage metadata (`adc_iq_bits`, `adc_iq_full_scale`,
  `adc_iq_component_dtype`, `x_storage_dtype`, `y_storage_dtype`,
  `clutter_type`, `mti_applied`, `mti_mode`).

Schema v9 records local/effective target-background gate metadata,
ADC/storage metadata, static-clutter metadata, `target_min_abs_velocity_mps`,
`mti_mode`, `min_label_snr_db`, `n_targets_range`, and shared-FMCW radar
metadata.

## CFAR / clutter interpretation

Raw CA-CFAR on a non-MTI RDM would naturally alarm on strong static clutter
cells.  That is a valid **raw CFAR detection** lesson, but it is a different
problem from P01's current U-Net detector.  Active P01 uses the MTI-filtered RDM
for both CFAR and U-Net, so clutter cells are not supervised positives.  If a
future project is explicitly called "neural CFAR imitation", its labels should
be generated from CFAR-like detections and should include clutter alarms by
design.

## 16-bit I/Q quantization diagnostic

If a lecture or report claims receiver-like 16-bit FMCW data, it must specify:

- whether full scale is per-frame/adaptive or fixed over a dataset/run,
- clipping fraction,
- quantization step size,
- target peak change after range-Doppler processing.

The active P1 generator records `adc_clipped_fraction` per sample and retries
generation when a fixed full-scale clip is encountered. Current smoke tests
require zero clipping on generated samples. If the target/clutter ranges are
expanded later, the fixed full-scale and clipping tests must be revisited.

## CA-CFAR settings

CA-CFAR is the primary classical baseline. A valid CFAR result must report:

- guard cells,
- training cells,
- design `pfa`,
- split used for selecting detector settings,
- test split used for final reporting.

CFAR must be evaluated from `rdm_mag_linear`. Do not run CFAR from clipped or normalized display channels.

## Neural detector thresholding

A valid U-Net result must report:

- checkpoint path / training command,
- validation threshold-selection rule,
- selected threshold,
- test split metrics,
- whether the run is a quick CPU check or a full GPU training run.

The test split must not be used to choose thresholds.

## Metrics

Report at minimum:

- `Pd`: pixel-level recall on target mask pixels,
- `Pfa`: false detections over non-target pixels,
- `Precision`,
- `F1`,
- `Pd` vs `Pfa` curves for both CFAR and U-Net.

## Week 10 reporting rule

A Week 10 result should point to saved JSON artifacts generated by
`analyze_data_contract.py`, `evaluate_cfar.py`, `evaluate_unet.py`, and
`make_verified_figures.py`. Otherwise it should be presented only as an implementation check, not as a performance conclusion.
