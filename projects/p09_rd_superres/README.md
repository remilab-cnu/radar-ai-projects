# P09: RD Map Super-Resolution

Range-Doppler Map을 물리적 저해상도 레이다 설정(32×32)에서 고해상도 설정(64×64)으로 매핑하는 딥러닝 데모.

## Task

저해상도 RD map을 입력으로 받아 고해상도 RD map을 출력한다. LR 입력은 HR dB map을
후처리로 이미지 다운샘플링한 것이 아니라, 같은 표적 scene을 **낮은 대역폭/적은 chirp 수**의
FMCW 설정으로 다시 시뮬레이션해 만든다.

- **HR 설정**: BW 1 GHz, 64 chirps → 64×64 RDM
- **LR 설정**: BW 0.5 GHz, 32 chirps → 32×32 RDM
- **입력**: 저해상도 RD map `(B, 1, 32, 32)` — dB 정규화
- **출력**: 고해상도 RD map `(B, 1, 64, 64)`
- **기준선**: Bicubic interpolation, image-domain zero-padding/zero-insertion

### Physics contract / allowed simplification / not claimed

- **Physics contract**: LR/HR 쌍은 동일한 표적 range/velocity/RCS를 서로 다른 FMCW
  radar configs로 관측한다. LR은 HR 대비 range bin spacing과 Doppler bin spacing이 약 2배 크다.
  생성된 HDF5 파일에는 `generation_mode`, LR/HR BW, chirp 수, range/Doppler bin spacing attrs가 저장된다.
- **Allowed simplification**: 두 설정 모두 같은 단순 해석적 FMCW simulator, 단일 RX, per-map dB 정규화를 사용한다.
  LR/HR 잡음은 같은 SNR 범위에서 독립 생성된다.
- **Not claimed**: 이 모델이 실제 센서의 물리적 bandwidth/chirp 부족을 정보 이론적으로 극복한다는 뜻이 아니다.
  학습된 simulator prior로 같은 분포의 LR RDM을 HR grid에 맞춰 추정하는 교육용 실험이다.

## Approach / Architecture

SRResNet-lite (~121K 파라미터):

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
python generate_data.py --smoke  # 빠른 smoke 데이터셋
```

- 표적: 1~4개/씬, SNR 5~25 dB
- **분할:** train 12K / val 2K / test 2K (`--smoke`: 256 / 64 / 64)
- HDF5 키: `x_lr`, `y_hr`, `peak_mask`, `n_targets`, `snr_db`
- HDF5 attrs: `generation_mode`, `hr_bw_hz`, `lr_bw_hz`, `hr_n_chirps`, `lr_n_chirps`,
  `hr_range_bin_spacing_m`, `lr_range_bin_spacing_m`, `hr_doppler_bin_spacing_mps`,
  `lr_doppler_bin_spacing_mps`

```
data/
  train.h5
  val.h5
  test.h5
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

물리 LR/HR 설정이 서로 다르므로 기존 post-FFT image downsample 과제보다 어렵다. 숫자는 CPU smoke가
아니라 충분한 데이터/epoch에서 확인해야 하며, 반드시 `metrics.json`의 세 그룹을 함께 보고한다.

| 지표 | Zero-pad baseline | Bicubic baseline | SRResNet-lite (목표) |
|------|-------------------|------------------|---------------------|
| PSNR (dB) | 낮음 | 물리 LR 보간 기준선 | bicubic보다 높음 |
| NMSE | 높음 | 물리 LR 보간 기준선 | bicubic보다 낮음 |
| Peak Loc Error (px) | peak proxy에서는 강할 수 있음 | 기준선 | 충분한 학습 후 기준선과 비교 |

평가 지표 설명:
- **PSNR (dB)**: 픽셀 수준 재구성 품질
- **NMSE**: 정규화 평균 제곱 오차
- **Peak Loc Error (px)**: GT 표적 위치 대비 예측 오차
- **baseline_zero_pad**: LR bin을 HR 짝수 bin에 복사하고 나머지를 0으로 채우는 image-domain 기준선이며, 물리적 FFT zero-padding 복원 주장으로 해석하지 않는다.
- zero-pad는 bright LR bin을 그대로 복사하므로 peak 위치 proxy에서는 bicubic/초기
  learned model보다 좋아 보일 수 있다. PSNR/NMSE와 peak metric을 함께 해석한다.

## Quick Start

```bash
# 데이터 생성 + 학습 (smoke 테스트)
python train.py --generate --smoke
```
