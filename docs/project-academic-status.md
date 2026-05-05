# Project Result Summary

This page lists the current classroom interpretation for each active project.
Use the project README files for commands and the technical docs for data-schema
details.

| Project | Classroom contract | Current result to cite |
|---|---|---|
| P01 U-Net FMCW Detector | Detect moving targets on MTI-filtered FMCW Range-Doppler maps and compare U-Net with CA-CFAR on the same processed input. | On the active balanced test setting, U-Net reaches F1 `0.8322` and Pd `0.7740`; CA-CFAR reaches F1 `0.6095`.  State that this is an MTI-filtered moving-target detection task. |
| P02 ResNet-18 Micro-Doppler HAR | Classify six human activities from target-range micro-Doppler spectrograms generated from a controlled pedestrian-scatterer model. | The default split is intentionally easy, so aspect and low-SNR stress sets are more informative for discussing generalization. |
| P03 Radar Mapping via DoA | Compare DoA estimators by projecting equal selected antenna-vector detections into point-cloud and occupancy-grid maps. | In the current 200 MHz map-quality setting, MUSIC and RadarCubeDoANet are sub-degree DoA methods, while coarse angle FFT creates meter-scale map error at lecture ranges. |
| P04 DnCNN-SAR Despeckling | Despeckle real Sentinel-1 GRD/SLC log-magnitude patches and compare DnCNN-SAR with classical filters. | DnCNN-SAR reaches about `31.10 dB / SSIM 0.794` overall; the SLC subset is about `31.00 dB / SSIM 0.792`.  Always mention the pseudo-clean multi-look target caveat. |

## Reporting cautions

- Do not present smoke-test numbers as full experiment results.
- State whether a result comes from synthetic FMCW data, controlled
  micro-Doppler data, moving-ego mapping data, or real Sentinel-1 patches.
- Include enough settings for another student to reproduce the comparison.
