# P10: Near-Field Source Localization

8소자 ULA에서 근거리/원거리 소스를 구분하고 각도·거리를 동시 추정하는 멀티헤드 CNN.

## Task

- **Near-field vs Far-field 분류**: 구면파(근거리) / 평면파(원거리) 판별
- **각도 추정**: sin/cos 표현으로 -60°~60° 범위 추정
- **거리 추정**: near-field 샘플에서만 0.5~5 m 범위 추정

베이스라인: Far-field MUSIC (항상 far-field로 가정)

## Approach / Architecture

NearFieldLocNet (~25K 파라미터):

```
입력: (B, 2, 8, 64)  — real/imag array snapshots

Trunk:
  Conv2d(2→16, 3×3)-BN-ReLU
  Conv2d(16→32, 3×3)-BN-ReLU
  AdaptiveAvgPool2d(1,1) → Flatten → (B, 32)

Head 1 (near/far):  Linear(32→1) → sigmoid
Head 2 (angle):     Linear(32→16)-ReLU → Linear(16→2)  [sin(θ), cos(θ)]
Head 3 (range):     Linear(32→16)-ReLU → Linear(16→1)  [m, near-field only]
```

손실 함수:
```
L = BCE(near_logit, near_label)
  + 0.5 × MSE([sin(θ_pred), cos(θ_pred)], [sin(θ_gt), cos(θ_gt)])
  + 0.5 × SmoothL1(range_pred, range_gt)  [near-field 샘플만]
```

각도를 sin/cos로 표현하면 -180°/+180° 불연속 문제를 방지할 수 있다.

## Data Generation

```bash
python generate_data.py   # 전체 데이터셋 생성
```

- 안테나: 8소자 ULA, d=λ/2
- Near-field: 구면파, range 0.5~5 m
- Far-field: 평면파, range >10 m
- SNR: 0~20 dB, 소스 수: 1~2개
- **분할:** train 20K / val 4K / test 4K (50/50 near/far)

HDF5 키: `x`, `near_label`, `angle_deg`, `range_m`, `snr_db`, `n_sources`

```
data/
  train.h5   # 20K samples
  val.h5     # 4K samples
  test.h5    # 4K samples
```

## Training

```bash
# 전체 데이터 생성 + 학습 (50 에폭)
python train.py --generate --epochs 50

# 데이터는 이미 있고 학습만
python train.py --epochs 50

# 평가만
python train.py --eval_only --checkpoint artifacts/best_model.pt
```

## Expected Results

| 지표 | Far-field MUSIC (baseline) | NearFieldLocNet (목표) |
|------|---------------------------|----------------------|
| Near/Far F1 (macro) | ~0.50 | >0.90 |
| Angle MAE (deg) | ~5 deg | <3 deg |
| Range MAE (m, near-field) | N/A | <0.5 m |
| Joint Loc Acc (5°) | ~30% | >70% |

평가 지표 설명:
- **Near/Far F1 (macro)**: binary 분류 성능
- **Angle MAE (deg)**: 각도 추정 평균 절대 오차
- **Range MAE (m, near-field)**: 근거리 거리 추정 오차
- **Joint Loc Acc (5°)**: 분류 정확 AND 각도 오차 <5°

## Quick Start

```bash
# 데이터 생성 + 학습 (smoke 테스트)
python train.py --generate --smoke
```
