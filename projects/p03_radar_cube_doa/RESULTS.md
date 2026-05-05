# P03 Result Registry Summary

Date: 2026-05-02
Scope: lecture-ready P03 radar mapping / DoA quality demonstration.

## Main canonical result

- Lecture bundle: https://remilab.cnu.ac.kr/share/576d311ccc45/p03_lecture_bundle.html
- Share: https://remilab.cnu.ac.kr/share/576d311ccc45/p03_main_result_report.html
- Off-grid appendix: https://remilab.cnu.ac.kr/share/576d311ccc45/p03_offgrid_appendix.html
- Reproducible local report: `artifacts/main_result_canonical_uniform128/p03_main_result_report.html`
- Reproducible local metrics: `artifacts/runs/doanet_200_canonical_uniform128_eval_20260502/metrics.json`
- Dataset command output: `/tmp/p03_mapping_canonical_200_uniform128_eval/test.h5`
- Config: 200 MHz, `N_fast=1024`, range resolution ≈ 0.75 m, wall spacing
  0.35 m, 10 ego frames, 4 held-out scenes, 5703 test detections, uniform
  128×128 grid over `x=[-20,20] m`, `y=[0,40] m`,
  `cell=0.3125 m × 0.3125 m`, `radar_max_range_m=40 m`.

| Method | DoA MAE | ≤2° acc. | Point-grid IoU | Mean point error |
|---|---:|---:|---:|---:|
| Oracle GT DoA | 0.000° | 1.000 | 0.969 | 0.000 m |
| MUSIC | 0.346° | 0.999 | 0.517 | 0.104 m |
| RadarCubeDoANet | 0.434° | 0.992 | 0.525 | 0.126 m |
| Coarse native angle FFT | 3.888° | 0.266 | 0.225 | 1.097 m |

Interpretation constraints:

- This is a DoA-isolation result: simulator-exact range and perfect ego pose are used in the main map evaluation.
- MUSIC is currently the strongest clean selected-vector reference. RadarCubeDoANet is close and sub-degree, but not more accurate than MUSIC.
- OGM IoU is secondary because thresholded inverse-sensor-model metrics can be non-monotonic. Point-grid IoU and point error are the clearer teaching metrics.
- Grid note: the canonical report now uses square physical cells.  Point-grid
  IoU remains a raster metric and should be read with the off-grid appendix;
  continuous point error/DoA metrics are the primary grid-invariant signals.
- The data path uses controlled per-detection RD-selected antenna-vector simulation with simulator-known association, so the lecture comparison isolates DoA quality.

## Supporting reports

| Kind | Share URL | Caveat |
|---|---|---|
| DoA-only diagnostics | https://remilab.cnu.ac.kr/share/639f797adc4b/p03_doa_diagnostics.html | Separate diagnostic pool: wall spacing 1.0 m and 4 frames, not the canonical map evaluation. |
| Angular / cross-range projection appendix | https://remilab.cnu.ac.kr/share/519e2020fc77/p03_angular_resolution_report.html | Projection-sensitivity probe; point pairs are simulated independently, so this is not simultaneous same-RD super-resolution. |
| Range-resolution appendix | https://remilab.cnu.ac.kr/share/576d311ccc45/p03_resolution_report.html | Oracle-DoA range-cell/footprint demonstration with first-hit occlusion; bandwidth-vs-IoU is still an ISM/raster metric, but hidden wall scatterers no longer clear visible wall cells. |
| Ego-motion-error appendix | https://remilab.cnu.ac.kr/share/e470aa251013/p03_ego_motion_error_report.html | GT DoA/range fixed; only ego poses are perturbed. Point error is the primary monotonic metric. |
| Off-grid raster appendix | https://remilab.cnu.ac.kr/share/576d311ccc45/p03_offgrid_appendix.html | GT DoA/range fixed; only sub-cell scene alignment changes. Use as a caveat for grid-IoU sensitivity, not as a main DoA benchmark. |

## Reproduction commands

See `README.md` for canonical data/evaluation/report commands. Local `artifacts/` outputs are intentionally ignored by git; this file is the tracked summary of the generated result set.
