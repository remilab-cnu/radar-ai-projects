# P09: RD Map Super-Resolution

Range-Doppler Map을 저해상도(32×32)에서 고해상도(64×64)로 복원하는 딥러닝 모델.

## Task

제한된 chirp 수·샘플 수 환경에서 표적 분해능을 향상시키기 위해,
저해상도 RD map을 입력으로 받아 고해상도 RD map을 출력한다.

- **입력**: 저해상도 RD map `(B, 1, 32, 32)` — dB 정규화
- **출력**: 고해상도 RD map `(B, 1, 64, 64)`
- **기준선**: Bicubic interpolation

## Approach / Architecture

SRResNet-lite (~100K 파라미터):

```
Conv(1→32, 3×3) → ReLU
→ 4× ResidualBlock [Conv-BN-ReLU-Conv-BN + skip]
→ Conv(32→32, 3×3)
→ Conv(32→128, 3×3) → PixelShuffle(2)   [32×32 → 64×64]
→ Conv(32→1, 3×3)
```

손실 함수:
```
L_total = L1(pred, HR) + 0.1 × L_grad(pred, HR)
```

- `L1`: 픽셀 수준 절대 오차
- `L_grad`: Sobel gradient 기반 경계 보존 손실

## Data Generation

```bash
python generate_data.py   # 전체 데이터셋 생성
```

- 표적: 1~4개/씬, SNR 5~25 dB
- **분할:** train 12K / val 2K / test 2K
- HDF5 키: `x_lr`, `y_hr`, `peak_mask`, `n_targets`, `snr_db`

```
data/
  train.h5   # 12K scenes
  val.h5     # 2K scenes
  test.h5    # 2K scenes
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

| 지표 | Bicubic (baseline) | SRResNet-lite (목표) |
|------|--------------------|---------------------|
| PSNR (dB) | ~28 dB | >33 dB |
| NMSE | ~-20 dB | <-28 dB |
| Peak Loc Error (px) | ~1.5 px | <0.8 px |

평가 지표 설명:
- **PSNR (dB)**: 픽셀 수준 재구성 품질
- **NMSE**: 정규화 평균 제곱 오차
- **Peak Loc Error (px)**: GT 표적 위치 대비 예측 오차

## Quick Start

```bash
# 데이터 생성 + 학습 (smoke 테스트)
python train.py --generate --smoke
```
