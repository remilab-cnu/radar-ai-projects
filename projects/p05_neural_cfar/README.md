# P05: Neural CFAR — 신경망 기반 레이다 표적 탐지

## Task

FMCW 레이다의 Range-Doppler Map(RDM)에서 **Cell Under Test(CUT) 주변 15×15 패치**를 입력으로 받아
표적 존재 여부를 이진 분류하는 소형 CNN을 학습한다.

전통적인 CA-CFAR는 선형 전력 RDM의 주변 training cell 평균으로 문턱값을 정한다.
이 데모의 baseline은 원본 RDM이 아니라 **정규화된 log-domain 15×15 패치** 위에서
동일한 guard/training-cell 아이디어를 적용한 교육용 CFAR-like 비교군이다.
Neural CFAR는 클러터 형상을 패치에서 직접 학습하여 이 단순 비교군의 한계를 관찰하게 한다.

| 항목 | 형태 | 설명 |
|------|------|------|
| 입력 | `(N, 2, 15, 15)` | ch0: noise-floor ref dB magnitude, ch1: 로컬 정규화 |
| 출력 | `(N, 1)` | raw logit (sigmoid → 탐지 확률) |
| 레이블 | `(N,)` | 0=비표적, 1=표적 |

## Approach / Architecture

```
Conv(2→16, 3×3)-BN-ReLU
Conv(16→32, 3×3)-BN-ReLU-MaxPool(2)
Conv(32→64, 3×3)-BN-ReLU-MaxPool(2)
GlobalAvgPool
FC(64)-ReLU → FC(1)
총 ~28K 파라미터
```

- **Loss**: Binary cross-entropy with logits
- **Optimizer**: Adam

## Data Generation

```bash
python generate_data.py          # 전체 데이터셋 생성
python generate_data.py --smoke  # 소규모 smoke 데이터셋
```

- **균형:** 50/50 표적/비표적
- **SNR bins:** 0, 5, 10, 15, 20, 25 dB (각 bin에서 균등 샘플링)
- **분할:** train 24K / val 6K / test 6K

```
data/
  train.h5   # 24K patches
  val.h5     # 6K patches
  test.h5    # 6K patches
```

## Training

```bash
# 전체 학습 (30 에폭)
python train.py --generate --epochs 30

# 평가만 (체크포인트 필요)
python train.py --eval_only --checkpoint artifacts/best_model.pt
```

## Expected Results

| 지표 | Patch CFAR-like baseline | Neural CFAR (목표) |
|------|-------------------|-------------------|
| ROC-AUC | ~0.80 | >0.92 |
| Pd @ Pfa=1e-2 | ~0.60 | >0.80 |
| Pd @ Pfa=1e-3 | ~0.40 | >0.65 |
| Balanced Acc. | ~0.65 | >0.85 |

저 SNR(0~5 dB) 구간에서 Neural CFAR의 개선 효과가 가장 뚜렷하게 나타난다.

학습 포인트:
1. **2채널 입력의 의미:** ch0은 절대적 강도(global context), ch1은 패치 내 상대적 강도(local contrast)
2. **CFAR 계열의 한계:** 균일 클러터 가정이 깨지는 경계/다중경로 환경에서 오탐 증가
3. **SNR vs. 탐지 성능:** ROC 곡선을 SNR bin별로 분리하면 저SNR에서의 개선폭이 더 크다
4. **Pfa 제어:** 고정 threshold(0.5)가 아닌 score thresholding으로 Pfa를 조절한다

## Quick Start

```bash
# Smoke test (데이터 생성 + 2 에폭 학습, CPU)
python train.py --generate --smoke
```
