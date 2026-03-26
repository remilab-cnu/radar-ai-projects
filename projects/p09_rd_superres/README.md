# P09: RD Map Super-Resolution

Range-Doppler Map을 저해상도(32×32)에서 고해상도(64×64)로 복원하는 딥러닝 모델.

## 목표

- **입력**: 저해상도 RD map (32×32, dB 정규화)
- **출력**: 고해상도 RD map (64×64)
- **응용**: 제한된 chirp 수·샘플 수 환경에서 표적 분해능 향상

## 모델 구조: SRResNet-lite

```
Conv(1→32, 3×3) → ReLU
→ 4× ResidualBlock [Conv-BN-ReLU-Conv-BN + skip]
→ Conv(32→32, 3×3)
→ Conv(32→128, 3×3) → PixelShuffle(2)   [32×32 → 64×64]
→ Conv(32→1, 3×3)
```

파라미터 수: ~100K

## 데이터

| 분할  | 샘플 수  |
|-------|---------|
| train | 12,000  |
| val   | 2,000   |
| test  | 2,000   |

- 표적: 1~4개/씬, SNR 5~25 dB
- HDF5 키: `x_lr`, `y_hr`, `peak_mask`, `n_targets`, `snr_db`

## 사용법

```bash
# 1. 데이터 생성 + 학습 (smoke 테스트)
python train.py --generate --smoke

# 2. 전체 데이터 생성 + 학습
python train.py --generate --epochs 50

# 3. 데이터는 이미 있고 학습만
python train.py --epochs 50

# 4. 평가만
python train.py --eval_only --checkpoint artifacts/best_model.pt
```

## 평가 지표

| 지표 | 설명 |
|------|------|
| PSNR (dB) | 픽셀 수준 재구성 품질 |
| NMSE | 정규화 평균 제곱 오차 |
| Peak Loc Error (px) | GT 표적 위치 대비 예측 오차 |

베이스라인: Bicubic interpolation

## 파일 구조

```
p09_rd_superres/
├── generate_data.py   # HDF5 데이터 생성
├── model.py           # SRResNet-lite 모델 정의
├── train.py           # 학습 및 평가
├── data/              # 생성된 HDF5 파일
└── artifacts/         # 체크포인트, metrics.json
```

## Loss 함수

```
L_total = L1(pred, HR) + 0.1 × L_grad(pred, HR)
```

- `L1`: 픽셀 수준 절대 오차
- `L_grad`: Sobel gradient 기반 경계 보존 손실
