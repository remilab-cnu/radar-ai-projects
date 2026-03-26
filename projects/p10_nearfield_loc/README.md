# P10: Near-Field Source Localization

8소자 ULA에서 근거리/원거리 소스를 구분하고 각도·거리를 동시 추정하는 멀티헤드 CNN.

## 목표

- **Near-field vs Far-field 분류**: 구면파(근거리) / 평면파(원거리) 판별
- **각도 추정**: sin/cos 표현으로 -60°~60° 범위 추정
- **거리 추정**: near-field 샘플에서만 0.5~5 m 범위 추정

## 모델 구조: NearFieldLocNet

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

파라미터 수: ~25K

## 데이터

| 분할  | 샘플 수 | Near/Far |
|-------|---------|---------|
| train | 20,000  | 50/50   |
| val   | 4,000   | 50/50   |
| test  | 4,000   | 50/50   |

- 안테나: 8소자 ULA, d=λ/2
- Near-field: 구면파, range 0.5~5 m
- Far-field: 평면파, range >10 m
- SNR: 0~20 dB, 소스 수: 1~2개
- HDF5 키: `x`, `near_label`, `angle_deg`, `range_m`, `snr_db`, `n_sources`

## 사용법

```bash
# 1. 데이터 생성 + 학습 (smoke 테스트)
python train.py --generate --smoke

# 2. 전체 데이터 생성 + 학습
python train.py --generate --epochs 50

# 3. 데이터 있고 학습만
python train.py --epochs 50

# 4. 평가만
python train.py --eval_only --checkpoint artifacts/best_model.pt
```

## 평가 지표

| 지표 | 설명 |
|------|------|
| Near/Far F1 (macro) | binary 분류 성능 |
| Angle MAE (deg) | 각도 추정 평균 절대 오차 |
| Range MAE (m, near-field) | 근거리 거리 추정 오차 |
| Joint Loc Acc (5°) | 분류 정확 AND 각도 오차 <5° |

베이스라인: Far-field MUSIC (항상 far-field로 가정)

## 파일 구조

```
p10_nearfield_loc/
├── generate_data.py   # HDF5 데이터 생성
├── model.py           # NearFieldLocNet 모델 정의
├── train.py           # 학습 및 평가
├── data/              # 생성된 HDF5 파일
└── artifacts/         # 체크포인트, metrics.json
```

## Loss 함수

```
L = BCE(near_logit, near_label)
  + 0.5 × MSE([sin(θ_pred), cos(θ_pred)], [sin(θ_gt), cos(θ_gt)])
  + 0.5 × SmoothL1(range_pred, range_gt)  [near-field 샘플만]
```

각도를 sin/cos로 표현하면 -180°/+180° 불연속 문제를 방지할 수 있다.
