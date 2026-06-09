# P04 bundled smoke data

This directory contains a small source-stratified P04 smoke subset:

| File | Samples | Purpose |
|---|---:|---|
| `real_despeckling_train.h5` | 24 | quick smoke training |
| `real_despeckling_val.h5` | 6 | quick smoke validation |
| `real_despeckling_test.h5` | 16 | small evaluation check |
| `p04_smoke_best_model.pt` | - | reference full-size DnCNN checkpoint for sample evaluation |
| `p04_smoke_reference_eval_results.json` | 16 evaluated samples | expected reference metrics for the bundled sample |

The HDF5 files contain modified Copernicus Sentinel-1-derived GRD/SLC teaching
patches processed by REMI Lab.  Copernicus Sentinel data access and use is
available on a free, full, and open basis through the Copernicus Data Space
Ecosystem legal notice.  Credit: contains modified Copernicus Sentinel data
processed by REMI Lab for teaching.

Use this bundle to verify installation, evaluation, and approximate P04 behavior:

```bash
python train.py --eval_only \
  --data_dir sample_data \
  --checkpoint sample_data/p04_smoke_best_model.pt \
  --eval_samples 0 \
  --ckpt_dir artifacts/sample_eval
```

Do not report these 16-sample smoke metrics as the full P04 result.  Use the
full real Sentinel-1 dataset and full-test evaluation for lecture claims.
