# P04 Sentinel-1 DnCNN-SAR Result Summary

Date: 2026-04-30 KST

This page summarizes the current P04 full-test result for classroom reporting.
Use `projects/p04_dncnn_sar/README.md` for commands and data details.

## Student-facing result

| Method | PSNR mean | SSIM mean | Message |
|---|---:|---:|---|
| Noisy input | 18.80 dB | 0.256 | speckled input reference |
| Median filter | 26.34 dB | 0.621 | strongest simple classical PSNR baseline |
| Lee filter | 24.81 dB | 0.633 | classical adaptive filter baseline |
| Frost filter | 24.73 dB | 0.662 | classical adaptive filter baseline |
| **DnCNN-SAR** | **31.10 dB** | **0.794** | about **+4.76 dB** over Median |

Focused-SLC subset: **31.00 dB / SSIM 0.792**.  GRD patches score higher because
they are already prefiltered and form a small minority subset, so report GRD as
a reference case rather than the main claim.

## Dataset and model

- Data: real Sentinel-1 GRD + focused SLC patches in normalized log/dB magnitude.
- Target: pseudo-clean multi-look target (`smooth_method=multilook`, `look_size=4`).
- Test split: `2,138` patches = `2,118` SLC + `20` GRD.
- Model: DnCNN-SAR, 17 layers, 64 filters, about 556K parameters.
- Training: 100 epochs, batch size 32, Adam learning rate `5e-4`, cosine schedule.
- Full diagnostic evaluation: all test patches.

## Metric caveats

- The target is pseudo-clean multi-look SAR, not true clean SAR ground truth.
- `enl_log_roi_proxy` is computed on normalized log/dB patches, not physical
  linear intensity.
- EPI can vary strongly on heterogeneous SLC scenes; prefer robust summaries and
  qualitative edge inspection over a single EPI number.
- Classical baselines are run in the same normalized log/dB domain as DnCNN for
  same-input comparison.

## Suggested qualitative cases

| Figure | Use |
|---|---|
| `slc_median_idx1794_slc.png` | representative SLC behavior |
| `baseline_wins_idx1622_slc.png` | honest case where a classical baseline is competitive |
| `slc_worst_idx1016_slc.png` | failure/worst-case discussion |
| `grd_easy_idx1589_grd.png` | easy prefiltered GRD contrast |

## Classroom interpretation

P04 demonstrates that a compact residual CNN can remove substantial speckle from
real Sentinel-1 log-magnitude patches under a pseudo-clean multi-look target.
The correct caveat is that the target is a practical teaching reference, not a
true clean SAR measurement.
