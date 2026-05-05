# P03 FMCW Moving-Ego Mapping / DoA Data Contract

Last updated: 2026-04-30.

P03 is the lecture project for **DoA quality as a downstream radar-mapping
problem**.  The neural model still consumes a range/Doppler-selected antenna
vector, but project success is judged by both DoA accuracy and environmental
perception quality: point-cloud maps and probabilistic occupancy maps.

## Lecture-facing objective

Teach the chain:

```text
world scene + known ego trajectory
  -> radar-relative range / DoA / closing velocity
  -> shared FMCW baseband array simulation
  -> range FFT + Doppler FFT
  -> selected RD antenna vector
  -> signal-processing DoA vs deep-learning DoA
  -> point-cloud map + Bayesian OGM
  -> DoA/map metrics
```

The mainline assumes **perfect ego-motion**.  Ego-motion error is appendix-only:
feed GT DoA/range and perturb pose/yaw/velocity so odometry sensitivity is not
confounded with DoA estimator quality.

## Active simulator path

`projects/p03_radar_cube_doa/generate_mapping_data.py` and the retained unit
lane in `generate_data.py` use `shared.fmcw_simulator.FMCWRadar` with the
reviewed complex-baseband dechirp/mixing contract:

```text
raw target scene
  -> shared FMCW tx/rx baseband chirp synthesis
  -> dechirp/mixing: beat(t) = rx(t) * conj(tx(t))
  -> range FFT over fast time
  -> Doppler FFT over slow time
  -> simulator-known selected (range bin, Doppler bin)
  -> complex antenna vector x_ant in C^{N_rx}
```

RF passband up-conversion and receiver down-conversion are intentionally
excluded. Carrier frequency is still used for wavelength, Doppler, carrier
phase, antenna steering phase, and radar-equation power.

## Ego-motion and velocity convention

World coordinates follow the Week-12/P03 convention:

- `x`: lateral, positive right,
- `y`: forward,
- heading `0 deg`: `+y`,
- radar bearing `0 deg`: boresight/forward, positive right.

For ego pose `(x_e, y_e, psi)` and target `(x_t, y_t)`:

```text
dx = x_t - x_e
dy = y_t - y_e
range = sqrt(dx^2 + dy^2)
angle = wrap(atan2(dx, dy) - psi)
```

The shared simulator interprets positive target `velocity` as **closing** range.
Therefore P03 stores:

```text
v_sim = dot(v_ego - v_target, line_of_sight)
```

For a static target and forward ego speed `V`, `v_sim = V*cos(angle)`.  This is
the target-ego direction dependence used by the mapping generator.

## Canonical P03 radar configuration

P03 has two named radar presets.  The moving-ego mapping/result standard is the
200 MHz preset; the 50 MHz preset remains the low-resolution stress and
fast-smoke condition.

| Parameter | Value | Reason |
|---|---:|---|
| `fc` | 77 GHz | automotive/lecture radar carrier; short wavelength makes ULA phase visible |
| `fs` | `4*bw` | active repository sampling contract |
| `N_rx` | 8 | enough aperture for angle FFT/MUSIC/neural comparison |
| `N_chirps` | 32 | compact Doppler axis for moving-ego velocity projection |
| `PRI` | 50 µs | keeps the current Doppler range (`±19.5 m/s` at 77 GHz) |

| Preset | `bw` | `N_samples` | `T_chirp` | `ΔR` | Use |
|---|---:|---:|---:|---:|---|
| Low-resolution stress/smoke | 50 MHz | 256 | 1.28 µs | 3.00 m | fast tests and coarse-map appendix case |
| Results baseline | 200 MHz | 1024 | 1.28 µs | 0.75 m | standard P03 map-quality result runs |

All range/RD helpers must receive `radar=...` so the shared FMCW positive-range
bin convention is preserved.

## Mapping sample generation

For each scene:

1. Build a Week-12-style map: side walls and back wall represented by dense
   point scatterers, static point reflectors, and optional moving target.
2. Generate a straight ego trajectory with exact pose and speed.
3. For each frame, compute visible targets and transform world state to
   radar-relative `(range, angle, closing velocity)`.
4. Simulate each associated detection with the shared FMCW simulator.
5. Run `range_doppler_map(raw, radar=P03_RADAR, ...)`.
6. Select simulator-known `(r_bin, d_bin)` and extract `x_ant`.
7. Store the Gaussian DoA label and scene/frame metadata.
8. Store scene-level persistent GT OGM from static world targets.

The current implementation uses simulator-known association for labels and map
experiments so the main experiment isolates DoA and mapping.  CFAR/detection
errors can be added as a later experiment without changing the DoA comparison
interface.

## Scene standard

The main P03 scene intentionally stays simple: walls are not semantic polygons
or mesh surfaces; they are densely sampled radar point scatterers.

| Component | Main result standard | Appendix use |
|---|---:|---|
| Left wall | `x=-6 m`, `y=6..30 m`, spacing `0.35 m` | same |
| Right wall | `x=6 m`, `y=8..30 m`, spacing `0.35 m` | same |
| Back wall | `y=31.5 m`, `x=-5..5 m`, spacing `0.35 m` | same |
| Static reflectors | four high-RCS points in the corridor | same |
| Dynamic target | optional observations; excluded from persistent GT | optional ghost/fading appendix |
| Resolution probes | disabled | same-bearing radial pairs at `0.5, 1.0, 1.5, 3.0 m` separation |

The dense wall spacing is below the default forward grid cell size
(`40 m / 64 ≈ 0.625 m`), so the GT map reads as connected walls while the radar
simulation still receives physically interpretable point-scatterer returns.
Smoke tests may relax the spacing to `1.0 m` to reduce runtime.

## Mapping HDF5 schema

`generate_mapping_data.py` writes `data_mapping/{train,val,test}.h5` with:

| Key | Shape | Meaning |
|---|---:|---|
| `x_ant` | `(N_det, 2, N_rx)` | real/imag selected antenna vector |
| `y_spectrum` | `(N_det, 181)` | Gaussian DoA target over `[-90, 90]` degrees |
| `angle_deg` | `(N_det,)` | simulator-known DoA relative to ego heading |
| `range_m` | `(N_det,)` | target range from ego pose |
| `velocity_mps` | `(N_det,)` | simulator closing velocity after ego-motion projection |
| `snr_db`, `requested_snr_db` | `(N_det,)` | realised/requested selected-target SNR |
| `target_rcs_m2` | `(N_det,)` | RCS passed through the radar equation |
| `r_bin`, `d_bin` | `(N_det,)` | simulator-known selected RD bin |
| `scene_idx`, `frame_idx`, `target_id`, `is_dynamic` | `(N_det,)` | association/map accumulation metadata |
| `gt_ogm` | `(N_scene, G, G)` | persistent static occupancy target |
| `poses` | `(N_scene, T, 4)` | `[x_m, y_m, heading_deg, speed_mps]`, perfect mainline ego motion |
| `angle_grid_deg` | `(181,)` | DoA spectrum axis |
| `radar_fc_hz`, `radar_bw_hz`, `radar_fs_hz`, `fs_over_bandwidth` | metadata | simulator contract metadata |
| `grid_size`, `grid_range_m`, `n_steps`, `dt_s`, `ego_speed_mps` | metadata | map/trajectory contract |
| `schema_version` | scalar | mapping dataset schema version |

The retained unit generator still writes `data/{train,val,test}.h5` with the
older one-target `x_ant -> y_spectrum` schema for low-level antenna-vector tests.

## Baseline rule

All primary DoA methods consume the same selected antenna vector:

- angle FFT / Bartlett-style weak baseline,
- single-snapshot MUSIC signal-processing baseline,
- `RadarCubeDoANet` deep-learning baseline.

MUSIC may form a rank-1 covariance internally as a classical baseline, but the
neural network must not receive covariance matrices, angle FFT features, ego
pose, or OGM labels as input.  Map evaluation must use the same range detections
and perfect ego poses for every method.

## Metrics

Mapping-mode `artifacts/metrics.json` reports:

| Group | Purpose |
|---|---|
| `deep_learning_doa` | model DoA MAE/RMSE and threshold accuracies |
| `signal_processing_angle_fft_doa` | finite-aperture SP baseline |
| `signal_processing_music_doa` | MUSIC SP baseline |
| `oracle_gt_doa` | upper-bound DoA sanity check |
| `map_from_*_doa` | OGM IoU/F1/precision/recall and point-cloud localization metrics |

Map groups include strict `0.4` and `0.5` occupancy thresholds, point-cloud-grid
IoU/F1, and mean/median/P90 point localization error.

## Appendix: radar resolution vs map quality

P03 also owns the technical standard for showing how radar resolution propagates
to map quality.  This must be a separate appendix from the DL-vs-SP DoA
comparison because it changes the range measurement model.

The appendix should sweep bandwidth/range resolution and map with **measured**
range, not simulator-exact range:

| Bandwidth | Range resolution `c/(2B)` | Role |
|---:|---:|---|
| 50 MHz | about 3.0 m | current compact smoke/runtime baseline; too coarse for sub-metre map claims |
| 100 MHz | about 1.5 m | low-resolution teaching case |
| 200 MHz | about 0.75 m | practical default candidate for the current 0.625 m grid |
| 500 MHz | about 0.30 m | high-resolution lecture figure candidate |
| 1 GHz | about 0.15 m | optional upper-bound / Week-12-style visual |

For each bandwidth, report both the nominal range cell and the cross-range
footprint from DoA:

```text
cross_range_resolution(R) ≈ R * Δθ
```

With ego motion, repeated views can reduce the apparent cross-range uncertainty
through bearing/range intersections, so the accumulated map should approach a
range-resolution-limited wall/point thickness.  The appendix should therefore
plot wall thickness and OGM IoU/F1 against bandwidth using oracle DoA first,
then optionally repeat with MUSIC/DL DoA to show estimator interaction.

## Verification

Minimum contract checks:

```bash
python3 -m pytest -q projects/p03_radar_cube_doa/test_radar_cube_doa_contract.py
python3 projects/p03_radar_cube_doa/generate_mapping_data.py --smoke --out_dir /tmp/p03_mapping_smoke
python3 projects/p03_radar_cube_doa/train.py --mapping --generate --smoke --mapping_data_dir /tmp/p03_mapping_train_smoke
```

Full lecture performance claims require regenerating mapping data and rerunning
the mapping-mode training/evaluation pipeline after simulator changes.
