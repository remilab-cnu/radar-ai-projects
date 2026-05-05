#!/usr/bin/env bash
# Full evaluation pipeline for Week 10 P01 artifacts.
#   0) Validate schema-v9 balanced data contract
#   1) Train UNet freshly unless SKIP_TRAIN=1
#   2) CFAR sweep on val + locked test eval
#   3) UNet threshold sweep on val + locked test eval
#   4) SNR-binned breakdown
#   5) Per-target contrast distribution
#   6) Dump representative scenes
#   7) Optionally regenerate Week 10 figures
#
# Run from repo root or project root:
#   bash projects/p01_unet_detector/run_full_eval_pipeline.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

ART="$ROOT/artifacts/full_eval"
SCN="$ROOT/artifacts/case_studies"
EPOCHS="${EPOCHS:-30}"
BASE_CH="${BASE_CH:-32}"
BATCH="${BATCH:-32}"
LR="${LR:-3e-4}"
GENERATE_DATA="${GENERATE_DATA:-0}"
N_TRAIN="${N_TRAIN:-50000}"
N_VAL="${N_VAL:-5000}"
N_TEST="${N_TEST:-5000}"
CFAR_PARALLEL="${CFAR_PARALLEL:-12}"
CFAR_SWEEP_MAX_SAMPLES="${CFAR_SWEEP_MAX_SAMPLES:-100}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-1000}"
GENERATE_FIGURES="${GENERATE_FIGURES:-1}"
FIGURE_SCRIPT="${FIGURE_SCRIPT:-$ROOT/figures/gen_week10_figures.py}"
CKPT="$ROOT/artifacts/best_model.pt"

echo "=== P01 full pipeline ==="
echo "Root:  $ROOT"
echo "Start: $(date)"
if [[ "$GENERATE_DATA" == "1" ]]; then
    echo ">>> Generating schema-v9 balanced data (train=$N_TRAIN, val=$N_VAL, test=$N_TEST)"
    python3 -u generate_data.py \
        --n_train "$N_TRAIN" \
        --n_val "$N_VAL" \
        --n_test "$N_TEST" \
        --out_dir data \
        --seed "${SEED:-42}"
fi
python3 - "$N_TRAIN" "$N_VAL" "$N_TEST" <<'PY'
import sys
from pathlib import Path
import h5py

expected_counts = {
    'det_train.h5': int(sys.argv[1]),
    'det_val.h5': int(sys.argv[2]),
    'det_test.h5': int(sys.argv[3]),
}

for name, expected in expected_counts.items():
    path = Path('data') / name
    with h5py.File(path, 'r') as f:
        schema = int(f['schema_version'][0])
        shape = tuple(int(v) for v in f['x'].shape)
        clutter = f['clutter_type'][0].decode()
        mti = f['mti_mode'][0].decode()
        clip_max = float(f['adc_clipped_fraction'][:].max())
        fs_over_bw = float(f['fs_over_bandwidth'][0])
        assert schema == 9, (path, schema)
        assert shape[0] == expected, (path, shape)
        assert clutter == 'static', (path, clutter)
        assert mti == 'slow_time_mean_removal_dc_notch', (path, mti)
        assert clip_max == 0.0, (path, clip_max)
        assert fs_over_bw == 4.0, (path, fs_over_bw)
        print(path, 'schema', schema, 'shape', shape, 'clutter', clutter, 'mti', mti, 'clip_max', clip_max, 'fs/BW', fs_over_bw)
PY

rm -rf "$ART" "$SCN"
mkdir -p "$ART" "$SCN"

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
    rm -f artifacts/best_model.pt artifacts/eval_results.json artifacts/history.json
    echo ">>> Training (epochs=$EPOCHS, base_ch=$BASE_CH, batch=$BATCH, lr=$LR)"
    python3 -u train.py --epochs "$EPOCHS" --base_ch "$BASE_CH" --batch_size "$BATCH" --lr "$LR"
else
    echo ">>> Training: SKIPPED via SKIP_TRAIN=1"
    test -f "$CKPT"
fi

echo ">>> CFAR sweep on val (${CFAR_SWEEP_MAX_SAMPLES} samples, 84 configs, parallel=$CFAR_PARALLEL)"
python3 -u evaluate_cfar.py --data_dir data --split val --sweep --parallel "$CFAR_PARALLEL" \
    --max_samples "$CFAR_SWEEP_MAX_SAMPLES" \
    --out "$ART/p01_cfar_sweep_val.json"

echo ">>> CFAR locked policy on test"
python3 -u evaluate_cfar.py --data_dir data --split test \
    --policy_from "$ART/p01_cfar_sweep_val.json" \
    --max_samples "$EVAL_MAX_SAMPLES" \
    --out "$ART/p01_cfar_selected_test.json"

echo ">>> U-Net threshold sweep on val (${EVAL_MAX_SAMPLES} samples)"
python3 -u evaluate_unet.py --data_dir data --split val --sweep \
    --checkpoint "$CKPT" --base_ch "$BASE_CH" \
    --max_samples "$EVAL_MAX_SAMPLES" \
    --out "$ART/p01_unet_threshold_sweep_val.json"

echo ">>> U-Net locked policy on test"
python3 -u evaluate_unet.py --data_dir data --split test \
    --checkpoint "$CKPT" --base_ch "$BASE_CH" \
    --policy_from "$ART/p01_unet_threshold_sweep_val.json" \
    --max_samples "$EVAL_MAX_SAMPLES" \
    --out "$ART/p01_unet_selected_test.json"

echo ">>> SNR-binned breakdown"
python3 -u snr_breakdown.py --data_dir data --split test \
    --cfar_policy "$ART/p01_cfar_selected_test.json" \
    --unet_policy "$ART/p01_unet_selected_test.json" \
    --checkpoint "$CKPT" --base_ch "$BASE_CH" \
    --max_samples "$EVAL_MAX_SAMPLES" \
    --out "$ART/p01_snr_breakdown.json"

echo ">>> Contrast distribution"
python3 -u contrast_distribution.py --data_dir data --split test \
    --max_samples "$EVAL_MAX_SAMPLES" \
    --out "$ART/contrast_distribution.json"

echo ">>> Dumping case studies"
python3 -u dump_case_studies.py --data_dir data --split test \
    --cfar_policy "$ART/p01_cfar_selected_test.json" \
    --unet_policy "$ART/p01_unet_selected_test.json" \
    --checkpoint "$CKPT" --base_ch "$BASE_CH" \
    --out_dir "$SCN" --n_scenes 4

if [[ "$GENERATE_FIGURES" == "1" ]]; then
    echo ">>> Generating Week 10 figures"
    python3 -u "$FIGURE_SCRIPT"
fi

python3 - <<'PY'
import json
from pathlib import Path
for name in ['p01_cfar_selected_test.json', 'p01_unet_selected_test.json']:
    payload = json.loads((Path('artifacts/full_eval') / name).read_text())
    selected = payload['selected_policy']
    print(name, json.dumps({
        key: selected.get(key)
        for key in ['Pd', 'Pfa', 'Precision', 'F1', 'target_recall', 'false_alarms_per_rdm', 'threshold', 'guard', 'train', 'pfa_design']
        if key in selected
    }, indent=2))
PY

echo "=== Done: $(date) ==="
ls -la "$ART" "$SCN"
if [[ "$GENERATE_FIGURES" == "1" ]]; then
    ls -la "$ROOT/figures" | grep fig_p01 || true
fi
