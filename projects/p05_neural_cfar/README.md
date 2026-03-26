# P05: Neural CFAR — 신경망 기반 레이다 표적 탐지

## 문제 정의

FMCW 레이다의 Range-Doppler Map(RDM)에서 **Cell Under Test(CUT) 주변 15×15 패치**를 입력으로 받아
표적 존재 여부를 이진 분류하는 소형 CNN을 학습한다.

전통적인 CA-CFAR는 균일한 클러터 분포를 가정하여 비균일 클러터 환경에서 성능이 저하된다.
Neural CFAR는 클러터 형상을 패치에서 직접 학습하여 이러한 한계를 극복한다.

## 입출력

| 항목 | 형태 | 설명 |
|------|------|------|
| 입력 | `(N, 2, 15, 15)` | ch0: noise-floor ref dB magnitude, ch1: 로컬 정규화 |
| 출력 | `(N, 1)` | raw logit (sigmoid → 탐지 확률) |
| 레이블 | `(N,)` | 0=비표적, 1=표적 |

## 데이터셋

- **균형:** 50/50 표적/비표적
- **SNR bins:** 0, 5, 10, 15, 20, 25 dB (각 bin에서 균등 샘플링)
- **분할:** train 24K / val 6K / test 6K

## 모델 구조

```
Conv(2→16, 3×3)-BN-ReLU
Conv(16→32, 3×3)-BN-ReLU-MaxPool(2)
Conv(32→64, 3×3)-BN-ReLU-MaxPool(2)
GlobalAvgPool
FC(64)-ReLU → FC(1)
총 ~28K 파라미터
```

## 실행 방법

```bash
# 1. Smoke test (데이터 생성 + 2 에폭 학습)
python train.py --generate --smoke

# 2. 전체 학습
python train.py --generate --epochs 30

# 3. 평가만 (체크포인트 필요)
python train.py --eval_only --checkpoint artifacts/best_model.pt

# 4. 데이터만 생성
python generate_data.py
python generate_data.py --smoke
```

## 기대 성능

| 지표 | CA-CFAR (baseline) | Neural CFAR (목표) |
|------|-------------------|-------------------|
| ROC-AUC | ~0.80 | >0.92 |
| Pd @ Pfa=1e-2 | ~0.60 | >0.80 |
| Pd @ Pfa=1e-3 | ~0.40 | >0.65 |
| Balanced Acc. | ~0.65 | >0.85 |

저 SNR(0~5 dB) 구간에서 Neural CFAR의 개선 효과가 가장 뚜렷하게 나타난다.

## 파일 구조

```
p05_neural_cfar/
  generate_data.py   # RDM 패치 데이터셋 생성
  model.py           # NeuralCFAR CNN 정의
  train.py           # 학습 + CA-CFAR baseline 비교
  data/              # HDF5 데이터 (train/val/test.h5)
  artifacts/         # 체크포인트, metrics.json, history.json
```

## 학습 포인트

1. **2채널 입력의 의미:** ch0은 절대적 강도(global context), ch1은 패치 내 상대적 강도(local contrast)
2. **CA-CFAR의 한계:** 클러터 경계, 다중경로 환경에서 오탐 증가
3. **SNR vs. 탐지 성능:** ROC 곡선을 SNR bin별로 분리하면 저SNR에서의 개선폭이 더 크다
4. **Pfa 제어:** 고정 threshold(0.5)가 아닌 score thresholding으로 Pfa를 조절한다
