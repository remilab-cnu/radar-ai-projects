# Project Source Map

This repository is the canonical runnable technical surface for the Radar AI
student/demo projects. The graduate lecture site in `../grad-radar-ai` is a
course-site authoring surface, not a second implementation source of truth.

## Ownership rules

1. Keep executable project code, data generators, model definitions, training
   entry points, smoke tests, and project README files here unless a project is
   explicitly listed as lecture-canonical below.
2. Keep lecture narrative, week pages, public figures, and course-site deploy
   artifacts in `grad-radar-ai`.
3. If lecture code must remain in `grad-radar-ai`, treat it as a frozen or
   generated lecture snapshot. Do not edit that snapshot as the canonical
   implementation without updating this map.
4. Do not advertise a deprecated implementation as the active student project.
   Archive it with a clear banner and point to the current canonical material.

## Project map

| Project | Canonical technical source | Lecture surface | Status / sync policy |
|---|---|---|---|
| P01 U-Net FMCW Detector | `projects/p01_unet_detector/` | `grad-radar-ai/projects/p1-unet-detector/`, Week 10 | Technical implementation lives here. Grad pages are lecture guides and should link back here for runnable code. |
| P02 ResNet-18 HAR | `projects/p02_resnet18_har/` | `grad-radar-ai/projects/p2-har/`, Week 11 | Technical implementation lives here. Grad pages are lecture guides and should link back here for runnable code. |
| P03 RAM/OGM AI | Current lecture-canonical design is in `grad-radar-ai` | `grad-radar-ai/weeks/week12/`, `grad-radar-ai/projects/p3-deepmusic/` | The old covariance-input `projects/p03_deepmusic_cnn/` code is deprecated archive material, not the active P03. A future runnable P03 should be regenerated here from the Week 12 RAM-only contract. |
| P04 SAR Despeckling | `projects/p04_dncnn_sar/` runnable companion | `grad-radar-ai/weeks/week13/` plus SAR lecture material | Grad P04 lecture/assignment narrative is canonical. Keep this implementation aligned with that lecture contract; do not change P04 objectives here first. |
| P05 Neural CFAR | `projects/p05_neural_cfar/` | Handout guide in `docs/guides/p05_guide.html` | Technical and student handout source lives here. |
| P06 I/Q Imbalance | `projects/p06_iq_imbalance/` | Handout guide in `docs/guides/p06_guide.html` | Technical and student handout source lives here. |
| P07 Full-Duplex SIC | `projects/p07_full_duplex_sic/` | Handout guide in `docs/guides/p07_guide.html` | Technical and student handout source lives here. |
| P08 Jammer Nulling | `projects/p08_jammer_nulling/` | Handout guide in `docs/guides/p08_guide.html` | Technical and student handout source lives here. |
| P09 RD Super-Resolution | `projects/p09_rd_superres/` | Handout guide in `docs/guides/p09_guide.html` | Technical and student handout source lives here. |

## Deprecated / archived material

- `projects/p03_deepmusic_cnn/` is the retired covariance-input DeepMUSIC/DoA
  experiment. It may be useful as an array-processing archive, but it is not the
  current P03 assignment and should not be used for new P03 claims.
- Historical planning documents in `grad-radar-ai/docs/` may mention previous
  P3/P4 designs. Treat those as dated decision records unless they explicitly
  point back to this source map or the current Week 12/Week 13 lecture pages.

## Sync checklist

When changing a project boundary:

1. Update this map first.
2. Update the project README in this repository.
3. Update the corresponding `grad-radar-ai` lecture page only as a consumer or
   canonical lecture exception.
4. Run at least a compile/smoke verification in the repo whose executable code
   changed.
