# P04: DnCNN-SAR Despeckling on Real Sentinel-1

Runnable P04 implementation for Week 13 SAR despeckling.  The code trains a
DnCNN-style residual CNN on real Sentinel-1 GRD/SLC patches in normalized log/dB
magnitude and compares it with classical filters.

## Student-facing result summary

For students, show the distilled result rather than every diagnostic case.

| Method | PSNR mean | SSIM mean | Student takeaway |
|---|---:|---:|---|
| Median filter | 26.34 dB | 0.621 | strongest simple classical PSNR baseline |
| **DnCNN-SAR** | **31.10 dB** | **0.794** | about **+4.76 dB** over Median on the full test split |

Add one sentence for source context:

> The practically important SLC subset is **31.00 dB / SSIM 0.792**; GRD reaches
> higher scores because it is already prefiltered and is a small minority case.

Recommended qualitative figures for a lecture/handout:

1. `slc_median_idx1794_slc.png` — representative SLC result.
2. `baseline_wins_idx1622_slc.png` or `slc_worst_idx1016_slc.png` — failure case
   showing why worst cases matter.
3. `grd_easy_idx1589_grd.png` — optional contrast case, clearly labeled as easier
   prefiltered GRD.

Use `dncnn_raw` for headline metrics.  Treat `dncnn_clipped` as a separate
post-processing/display diagnostic.

## Data contract

The generator expects Sentinel-1 products under `projects/p04_dncnn_sar/raw_sentinel1/`
by default.  Override with `P04_SAR_DATA_ROOT` or `--data_root` when the data is
mounted elsewhere.

For clone-only environment checks, `python train.py --generate --smoke` remains
runnable without locally mounted Sentinel-1 source data.  When the GRD source
file is missing, smoke mode copies the bundled `sample_data/` real Sentinel-1
smoke subset.  If that bundle is unavailable, it falls back to synthetic
plumbing data with `source=synthetic_smoke`.

`generate_data.py` writes:

| Split | File | Contents |
|---|---|---|
| Train | `data/real_despeckling_train.h5` | training patches |
| Val | `data/real_despeckling_val.h5` | validation patches |
| Test | `data/real_despeckling_test.h5` | test patches |

Each HDF5 file contains `noisy`, `clean`, and `source` arrays plus attrs for data
kind, smoothing method, look size, and source paths.  The default SLC target is
intensity-domain multi-look smoothing (`smooth_method=multilook`, `look_size=4`):
it is pseudo-clean, not true clean SAR ground truth.

## Model and metrics

- Model: 17-layer DnCNN-SAR, 64 filters, about 556K parameters
- Input/output: `(B, 1, 256, 256)` normalized log/dB patches
- Loss: Charbonnier (`w=0.8`) + SSIM (`w=0.2`)
- Metrics: PSNR, SSIM, `enl_log_roi_proxy`, EPI

Metric caveats for instructors:

- `enl_log_roi_proxy` is a smoothness proxy on normalized log/dB images, not
  physical linear-intensity ENL.
- EPI can be high variance on real SLC scenes; prefer robust summaries and case
  images over absolute-value claims.
- Classical baselines are run in the same normalized log/dB domain as DnCNN for
  same-input-contract comparison.

## Commands

```bash
# Generate data and train the full lecture-scale checkpoint.
python train.py --generate --epochs 100 --batch_size 32 --lr 5e-4 --no_amp

# Fast local smoke check. Uses bundled real smoke data if Sentinel-1 is absent.
python train.py --generate --smoke

# Reproduce bundled smoke-sample evaluation with the reference checkpoint.
python train.py --eval_only \
  --data_dir sample_data \
  --checkpoint sample_data/p04_smoke_best_model.pt \
  --eval_samples 0 \
  --ckpt_dir artifacts/sample_eval

# Evaluation only with the default capped first-N setting.
python train.py --eval_only --checkpoint artifacts/best_model.pt

# Full diagnostic evaluation: all test patches + per-sample CSV.
python train.py --eval_only --checkpoint artifacts/best_model.pt \
  --eval_samples 0 --ckpt_dir artifacts/diagnostics_full

# Deterministic qualitative cases from the diagnostic CSV.
python make_case_studies.py \
  --artifact_dir artifacts/diagnostics_full \
  --checkpoint artifacts/best_model.pt \
  --out_dir artifacts/diagnostics_full/case_studies
```

`--eval_samples 0` means full test-set evaluation.  Positive values keep a
faster capped first-N evaluation.

## Diagnostic artifacts

| Artifact | Purpose |
|---|---|
| `artifacts/diagnostics_full/eval_results.json` | aggregate metrics, robust percentiles, and source breakdown |
| `artifacts/diagnostics_full/per_sample_metrics.csv` | per-sample metrics for reproducible case selection |
| `artifacts/diagnostics_full/case_studies/case_manifest.json` | case-study sample indices, metrics, and caveats |
| `artifacts/diagnostics_full/case_studies/*.png` | qualitative panels for lecture/handout drafting |

## Boundaries

- Use real Sentinel-1 GRD/SLC data for Week 13 claims.
- Treat bundled smoke outputs as sample-level checks, not full-test claims.
- Treat synthetic smoke outputs as environment/path checks only if the real
  smoke bundle is missing.
- Do not present smoke-test outputs as full experiment results.
- For new claims, regenerate or read the JSON/CSV artifacts instead of copying
  numbers by hand.
