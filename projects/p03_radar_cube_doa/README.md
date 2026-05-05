# P03: Radar Mapping as a DoA Quality Testbed

P03 is a **mapping-first radar AI project**.  The central question is not
only "which method estimates DoA with lower MAE?" but:

> When a moving ego radar projects detections into a world map, how much do
> signal-processing DoA and deep-learning DoA change the resulting point-cloud
> map and probabilistic occupancy map?

The active project lane is moving-ego environmental perception.  A compact
single-target antenna-vector path is retained only for low-level contract tests.

## Active contract

1. Build a Week-12-style world scene with static walls/point reflectors and an
   optional moving object.
2. Move the ego radar along a known, error-free trajectory.
3. Convert each visible world target into radar-relative range, angle, and
   **simulator closing velocity**:
   `v_sim = dot(v_ego - v_target, line_of_sight)`.
4. Simulate a **77 GHz** complex-baseband FMCW array beat cube with the shared
   `shared.fmcw_simulator.FMCWRadar` contract (`fs = 4·BW`).
5. Run range FFT + Doppler FFT; skip angle FFT for the neural input path.
6. Extract the same RD-selected antenna vector for all DoA methods.
7. Compare:
   - angle FFT / Bartlett-style finite-aperture baseline,
   - single-snapshot MUSIC signal-processing baseline,
   - `RadarCubeDoANet` deep-learning DoA.
8. Feed each method's DoA estimates, with the same range detections and perfect
   ego poses, into:
   - a world-frame point-cloud map,
   - a Bayesian/log-odds probabilistic occupancy grid map.
9. Report both DoA metrics and downstream map metrics.

Mainline P03 assumes **ego-motion has no error**.  Ego-motion error is isolated
as an appendix experiment: use GT DoA/range and perturb pose/yaw/velocity only,
so odometry effects are not confounded with DoA estimation.

## Radar/scene standard

- **Results radar:** `BW=200 MHz`, `N_fast=1024`, `fs=800 MHz`,
  `ΔR≈0.75 m`. This is the baseline for P03 map-quality results.
- **Map-grid standard:** use an explicit, physically uniform world grid.  The lecture visualization
  default is `x=[-20,20] m`, `y=[0,40] m`, `128×128`, so
  `cell_x=cell_y=0.3125 m`.  This is a fine rasterization grid for display and
  IoU; continuous point error remains the grid-invariant scientific metric.
  The radar visibility/range limit stays separate at `radar_max_range_m=40 m`.
- **Low-resolution stress/smoke radar:** `BW=50 MHz`, `N_fast=256`,
  `ΔR≈3.0 m`. This remains useful to show map thickening/target merging, but it
  is too coarse to be the headline map-quality standard.
- **Scene:** corridor-like static walls represented by dense point scatterers
  plus point reflectors.  Main results use `wall_spacing≈0.35 m`; smoke tests
  may use `1.0 m` for speed.  Resolution probes are appendix-only.
- **Ego route:** straight known trajectory by default (`8 m/s`, `dt=0.2 s`);
  resolution appendix should use 10–12 frames and optional mild lateral offset
  while keeping perfect ego pose.

## Lecture figures to show

1. **Radar-scatterer corridor scene:** dense point-scatterer side walls, back
   wall, static reflectors, ego route, and first-frame FoV.
2. **Single-frame map:** the same scene from one radar frame, showing wide
   cross-range/angular uncertainty.
3. **Ego-motion accumulated map:** perfect ego-motion combines repeated
   range-bearing observations and sharpens the world map.
4. **DoA estimator comparison:** oracle DoA, MUSIC, angle FFT, and deep-learning
   DoA projected into the same point-cloud/OGM map.
5. **DoA error to map error:** show that a few degrees at range `R` becomes
   lateral error `R·Δθ`.
6. **Radar resolution appendix:** 50/100/200/400/800 MHz maps to show that
   50 MHz is range-limited and 200 MHz is the main P03 result baseline.
7. **Ego-motion error appendix:** with GT DoA/range fixed, perturb yaw/pose to
   show that odometry error is a separate map-degradation mechanism.
8. **Off-grid raster appendix:** with GT DoA/range fixed, shift the continuous
   scene by sub-cell offsets to show that grid IoU is a finite-raster metric,
   while physical point error stays zero.

## Current result status

Current lecture-ready result with an explicit uniform `MapGridSpec`:

- Lecture bundle: https://remilab.cnu.ac.kr/share/576d311ccc45/p03_lecture_bundle.html
- Share: https://remilab.cnu.ac.kr/share/576d311ccc45/p03_main_result_report.html
- Off-grid appendix: https://remilab.cnu.ac.kr/share/576d311ccc45/p03_offgrid_appendix.html
- Tracked summary: `RESULTS.md`
- Reproducible local outputs, after running the commands below:
  `artifacts/main_result_canonical_uniform128/p03_main_result_report.html` and
  `artifacts/runs/doanet_200_canonical_uniform128_eval_20260502/metrics.json`
- Config: 200 MHz, `N_fast=1024`, `ΔR≈0.75 m`, wall spacing `0.35 m`,
  10 ego frames, 4 held-out scenes, 5703 test detections, uniform 128×128
  grid over `x=[-20,20] m`, `y=[0,40] m`, `cell=0.3125 m × 0.3125 m`.

Main canonical DoA/map metrics:

| Method | DoA MAE | ≤2° acc. | Point-grid IoU | Mean point error |
|---|---:|---:|---:|---:|
| Oracle GT DoA | 0.000° | 1.000 | 0.969 | 0.000 m |
| MUSIC | 0.346° | 0.999 | 0.517 | 0.104 m |
| RadarCubeDoANet | 0.434° | 0.992 | 0.525 | 0.126 m |
| Coarse native angle FFT | 3.888° | 0.266 | 0.225 | 1.097 m |

Interpretation:

- MUSIC remains the strongest clean signal-processing reference in this
  selected-vector setup.
- RadarCubeDoANet is sub-degree and close to MUSIC in the map projection, but
  it is not more accurate than MUSIC in the current result.
- The coarse native angle FFT is the failure/weak baseline; its several-degree
  angular error becomes meter-scale lateral map error at lecture ranges.
- OGM IoU is included for continuity, but point-grid IoU and point error are the
  clearer headline metrics because the OGM inverse-sensor model and GT
  rasterization can make thresholded IoU slightly non-monotonic.
- Grid note: point-grid IoU changed after the uniform-grid revision, as
  expected for a raster metric.  Continuous point error and DoA error remain
  the primary grid-invariant scientific signals; the off-grid appendix isolates
  this raster-sensitivity caveat.

Result registry:

- `RESULTS.md` is the tracked result registry summary.
- `artifacts/p03_result_registry.json` is a regenerated local registry output;
  `artifacts/` is ignored and should not be treated as source-controlled
  evidence.

Appendix links:

- DoA-only diagnostics:
  https://remilab.cnu.ac.kr/share/639f797adc4b/p03_doa_diagnostics.html
- Angular/cross-range appendix:
  https://remilab.cnu.ac.kr/share/519e2020fc77/p03_angular_resolution_report.html
- Range-resolution appendix:
  https://remilab.cnu.ac.kr/share/576d311ccc45/p03_resolution_report.html
- Ego-motion-error appendix:
  https://remilab.cnu.ac.kr/share/e470aa251013/p03_ego_motion_error_report.html
- Off-grid raster appendix:
  https://remilab.cnu.ac.kr/share/576d311ccc45/p03_offgrid_appendix.html

## Why this revision

The lecture goal is to show why radar signal processing matters for environment
perception.  A DoA error of a few degrees is easy to underestimate in a table;
after projection to range `r`, it becomes a lateral map error of roughly
`r·Δθ`.  P03 therefore evaluates DoA through maps that students can inspect.

## Code layout

| File | Role |
|---|---|
| `mapping.py` | Ego pose/world target dataclasses, world↔radar transforms, relative velocity, point-cloud and OGM utilities. |
| `generate_mapping_data.py` | Moving-ego scenario generator. Writes per-detection `x_ant` plus scene-level `gt_ogm` and `poses`. |
| `train.py --mapping` | Trains the same DoA spectrum network on scenario-derived detections and evaluates DoA + map quality. |
| `generate_data.py` | Retained single-target RD-selected antenna-vector unit lane for low-level DoA contract tests. |
| `make_offgrid_appendix.py` | Off-grid/raster-alignment appendix with GT DoA/range and perfect ego motion fixed. |
| `model.py` | Residual 1D ConvNet from `(2, N_rx)` antenna vector to 181-bin DoA spectrum. |
| `test_radar_cube_doa_contract.py` | Contract tests for the legacy unit lane and the new mapping/ego-motion lane. |

Technical details live in:

- `docs/technical/p03_fmcw_data_generation_contract.md`

## Commands

```bash
# Mapping-first smoke test: uses 50 MHz / 256 fast samples for runtime
python train.py --mapping --generate --smoke

# Mapping-first full local/GPU run
python train.py --mapping --generate --epochs 30 --batch_size 1024 \
  --n_train_scenes 80 --n_val_scenes 16 --n_test_scenes 16 \
  --mapping_steps 10 --radar_bw_mhz 200 --radar_n_fast 1024 \
  --wall_spacing_m 0.35 --grid_size 128 \
  --map_x_min_m -20 --map_x_max_m 20 --map_y_min_m 0 --map_y_max_m 40 \
  --radar_max_range_m 40

# Canonical uniform-grid evaluation using the current strong checkpoint
python train.py --mapping --generate --eval_only \
  --mapping_data_dir /tmp/p03_mapping_canonical_200_uniform128_eval \
  --checkpoint artifacts/runs/doanet_200_balanced_20260501/best_model.pt \
  --artifact_dir artifacts/runs/doanet_200_canonical_uniform128_eval_20260502 \
  --radar_bw_mhz 200 --radar_n_fast 1024 --wall_spacing_m 0.35 \
  --grid_size 128 --map_x_min_m -20 --map_x_max_m 20 \
  --map_y_min_m 0 --map_y_max_m 40 --radar_max_range_m 40 \
  --mapping_steps 10 --n_train_scenes 1 --n_val_scenes 1 --n_test_scenes 4 \
  --batch_size 256 --seed 4242

# Build the canonical map report
python make_main_result_report.py \
  --dataset /tmp/p03_mapping_canonical_200_uniform128_eval/test.h5 \
  --checkpoint artifacts/runs/doanet_200_balanced_20260501/best_model.pt \
  --metrics artifacts/runs/doanet_200_canonical_uniform128_eval_20260502/metrics.json \
  --out_dir artifacts/main_result_canonical_uniform128 --scene_idx 0 --cpu

# Build the angular/cross-range appendix
python make_angular_resolution_appendix.py \
  --checkpoint artifacts/runs/doanet_200_balanced_20260501/best_model.pt \
  --out_dir artifacts/angular_resolution_appendix --cpu

# Build the ego-motion-error appendix with oracle DoA/range
python make_ego_motion_error_appendix.py \
  --dataset /tmp/p03_mapping_canonical_200_uniform128_eval/test.h5 \
  --out_dir artifacts/ego_motion_appendix --scene_idx 0

# Build the off-grid/raster-alignment appendix
python make_offgrid_appendix.py \
  --out_dir artifacts/offgrid_appendix --grid_size 128 \
  --map_x_min_m -20 --map_x_max_m 20 --map_y_min_m 0 --map_y_max_m 40

# Low-resolution stress condition for the resolution appendix
python train.py --mapping --generate --smoke --radar_bw_mhz 50 --radar_n_fast 256

# Legacy unit DoA lane (kept for low-level antenna-vector checks)
python train.py --generate --smoke
```

## Mapping HDF5 schema

`generate_mapping_data.py` writes `data_mapping/{train,val,test}.h5`:

| Key | Shape | Description |
|---|---:|---|
| `x_ant` | `(N_det, 2, N_rx)` | RD-selected complex antenna vector `[real, imag]` |
| `y_spectrum` | `(N_det, 181)` | Gaussian DoA spectrum label |
| `angle_deg` | `(N_det,)` | GT DoA relative to ego heading |
| `range_m` | `(N_det,)` | GT target range from ego pose |
| `velocity_mps` | `(N_det,)` | simulator closing velocity after ego-motion projection |
| `scene_idx`, `frame_idx` | `(N_det,)` | scene/frame index for map accumulation |
| `target_id`, `is_dynamic` | `(N_det,)` | association metadata; persistent GT map excludes dynamic targets |
| `gt_ogm` | `(N_scene, G, G)` | static world occupancy target |
| `poses` | `(N_scene, T, 4)` | `[x_m, y_m, heading_deg, speed_mps]`; perfect mainline ego motion |
| `grid_nx`, `grid_ny`, `grid_x/y_{min,max}_m`, `grid_cell_x/y_m` | metadata | explicit `MapGridSpec`; canonical grid is square-cell `x=[-20,20]`, `y=[0,40]` |
| `radar_max_range_m` | metadata | radar visibility/ISM range limit, intentionally decoupled from map width |
| `radar_*`, `fs_over_bandwidth`, `schema_version` | metadata | simulator/data contract metadata |

The retained unit generator writes `data/{train,val,test}.h5` with the compact
`x_ant → y_spectrum` schema.

## Metrics

`artifacts/metrics.json` in mapping mode reports:

- `deep_learning_doa`
- `signal_processing_angle_fft_doa`
- `signal_processing_music_doa`
- `oracle_gt_doa`
- `map_from_deep_learning_doa`
- `map_from_angle_fft_doa`
- `map_from_music_doa`
- `map_from_oracle_gt_doa`

Map groups include OGM IoU/F1/precision/recall at strict `0.4` and `0.5`
thresholds, point-cloud-grid IoU/F1, and mean/median/P90 point localization
error.  The oracle map is an upper-bound sanity check for the range/pose/ISM
pipeline.
