# Active Project Guide

This guide summarizes the public course project layout.  Students should start
from the root `README.md`, then open the README inside the project folder they
are using.

## Project folders

| Project | Folder | What to run |
|---|---|---|
| P01 U-Net FMCW Detector | `projects/p01_unet_detector/` | `python train.py --generate --smoke` |
| P02 ResNet-18 HAR | `projects/p02_resnet18_har/` | `python train.py --generate --smoke` |
| P03 Radar Mapping via DoA | `projects/p03_radar_cube_doa/` | `python train.py --mapping --generate --smoke` |
| P04 DnCNN-SAR Despeckling | `projects/p04_dncnn_sar/` | `python train.py --generate --smoke` |

## Shared modules

| Module | Used by | Purpose |
|---|---|---|
| `shared/fmcw_simulator.py` | P01, P03 | Complex-baseband FMCW beat simulation and Range-Doppler processing |
| `shared/clutter_model.py` | P01 | Static clutter, MTI preprocessing, and target-mask generation |
| `shared/micro_doppler.py` | P02 | Human body/scatterer model and target-range micro-Doppler spectrograms |
| `shared/burst_simulator.py` | tests / demonstrations | Clean pulse-burst and matched-filter reference |
| `shared/doa_utils.py` | P03 | Array steering, classical DoA algorithms, and DoA metrics |
| `shared/sar_simulator.py` | demonstrations | Simplified SAR image utilities |

## Result reporting

Use smoke tests to check that the code runs.  Use full train/evaluation commands
from each project README when reporting performance numbers.  A valid report
should state the dataset split, model settings, baseline settings, and metric
values used in the comparison.
