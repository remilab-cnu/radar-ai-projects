# P05: Neural CFAR — 신경망 기반 레이다 표적 탐지

## Task

FMCW 레이다의 Range-Doppler Map(RDM)에서 **Cell Under Test(CUT) 주변 15×15 패치**를 입력으로 받아
표적 존재 여부를 이진 분류하는 소형 CNN을 학습한다.

전통적인 CA-CFAR는 선형 전력 RDM의 주변 training cell 평균으로 문턱값을 정한다.
이 데모의 baseline도 저장된 `patch_power` 선형 전력 15×15 패치에서 CUT(중심 셀), guard cell, training cell을 구분해
목표 Pfa에 맞는 CA-CFAR threshold를 계산한다. Neural CFAR는 같은 CUT geometry에서 생성된 정규화 입력 패치를 학습해
동일한 Pfa 조건의 score threshold와 비교한다.

- **Physics contract:** FMCW beat signal → range-Doppler FFT → CUT 중심 15×15 패치이며, classical CA-CFAR는 `patch_power`의 선형 전력 cell에만 적용한다.
- **Allowed simplification:** 작은 단일수신기 FMCW 장면, 제한된 SNR bin, `mixed` clutter, 15×15 local patch classifier로 CPU handout runtime을 유지한다.
- **Not claimed:** 정규화 log-domain 입력이 classical CFAR의 통계 domain이라는 주장이나, 이 패치 실험이 production-grade full-frame tracker라는 주장은 하지 않는다.

| 항목 | 형태 | 설명 |
|------|------|------|
| 입력 | `(N, 2, 15, 15)` | ch0: noise-floor ref dB magnitude, ch1: 로컬 정규화 |
| CFAR 입력 | `(N, 15, 15)` | `patch_power`: 같은 CUT 주변의 선형 RDM power |
| 메타데이터 | `(N,)` | CUT range/Doppler bin, nearest target-bin distance, clutter type |
| 출력 | `(N, 1)` | raw logit (sigmoid → 탐지 확률) |
| 레이블 | `(N,)` | 0=비표적 CUT, 1=표적 CUT |

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
- **CFAR 메타데이터:** `patch_power`, `cut_range_bin`, `cut_doppler_bin`, `target_distance_bins`, `clutter_type`
- smoke처럼 작은 split에서는 SNR bin 균등 분배 때문에 요청 샘플 수보다 1~2개 적게
  생성될 수 있다. full split은 표의 규모를 기준으로 해석한다.

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

| 지표 | Linear-power patch CA-CFAR baseline | Neural CFAR (목표) |
|------|-------------------|-------------------|
| ROC-AUC | N/A (fixed threshold detector) | >0.92 |
| Pd @ Pfa=1e-2 | ~0.60 | >0.80 |
| Pd @ Pfa=1e-3 | ~0.40 | >0.65 |
| Balanced Acc. | ~0.65 | >0.85 |

위 수치는 full run에서의 기대 범위다. `--smoke`는 표본 수가 작고 쉬운 no-target
CUT가 섞일 수 있어 CA-CFAR empirical Pfa/Pd가 과도하게 좋아지거나 나빠질 수 있다.

저 SNR(0~5 dB) 구간에서 Neural CFAR의 개선 효과가 가장 뚜렷하게 나타난다.

학습 포인트:
1. **2채널 입력의 의미:** ch0은 절대적 강도(global context), ch1은 패치 내 상대적 강도(local contrast)
2. **CFAR 계열의 한계:** 균일 클러터 가정이 깨지는 경계/다중경로 환경에서 오탐 증가
3. **SNR vs. 탐지 성능:** ROC 곡선을 SNR bin별로 분리하면 저SNR에서의 개선폭이 더 크다
4. **Pfa 제어:** CA-CFAR는 선형 전력 threshold로, Neural CFAR는 score thresholding으로 목표 Pfa 조건을 맞춘다

## Quick Start

```bash
# Smoke test (데이터 생성 + 2 에폭 학습, CPU)
python train.py --generate --smoke
```
