# P06: I/Q Imbalance Correction — 신경망 기반 I/Q 불균형 보정

## Task

실제 레이다/통신 수신기에서 I(동상) 채널과 Q(직교) 채널 사이에는
**이득 불일치(gain mismatch)**, **위상 불일치(phase mismatch)**, **DC 오프셋**이 발생한다.
이를 I/Q 불균형이라 하며, 방치하면 레이다 영상에 허위 표적(image spur)이 생기고
도플러 추정 오차가 커진다.

1D CNN이 손상된 beat signal로부터 불균형 파라미터 4개를 직접 추정하고,
해석적 역변환으로 원신호를 복원한다.

| 항목 | 형태 | 설명 |
|------|------|------|
| 입력 | `(N, 2, 512)` | 손상된 I/Q (ch0: I, ch1: Q) |
| 출력 | `(N, 4)` | `[gain_db, phase_deg, dc_i, dc_q]` |
| 보조 출력 | 해석적 역변환으로 복원된 신호 | |

I/Q Imbalance 모델:
```
I_out(t) = I(t) + dc_i
Q_out(t) = g * (I(t)*sin(φ) + Q(t)*cos(φ)) + dc_q

g   = 10^(gain_db / 20)  [선형 이득]
φ   = phase_deg * π/180  [위상 오차 rad]
```

추정 파라미터: `[gain_db, phase_deg, dc_i, dc_q]`

## Approach / Architecture

```
Conv1d(2→64, k=7)-BN-ReLU
Conv1d(64→128, k=5, stride=2)-BN-ReLU
Conv1d(128→128, k=5, stride=2)-BN-ReLU
AdaptiveAvgPool1d(1)
FC(128→64)-ReLU → FC(64→4)
총 ~133K 파라미터
```

손실 함수:
```
L = SmoothL1(pred_params / scale, true_params / scale)
  + λ * L1(corrected_signal, clean_signal)

scale = [3.0 dB, 15.0 deg, 0.05, 0.05]  (각 파라미터 정규화)
λ = 0.5
```

파라미터 추정 손실과 신호 복원 손실을 동시에 최적화한다.

## Data Generation

```bash
python generate_data.py          # 전체 데이터셋 생성
python generate_data.py --smoke  # 소규모 smoke 데이터셋
```

- **이득 범위:** ±0.5 ~ ±3.0 dB
- **위상 범위:** ±1 ~ ±15 deg
- **DC 오프셋:** ±0.05 (정규화 기준)
- **SNR:** 5 ~ 25 dB
- **분할:** train 18K / val 3K / test 3K

```
data/
  train.h5   # 18K samples
  val.h5     # 3K samples
  test.h5    # 3K samples
```

## Training

```bash
# 전체 학습 (30 에폭)
python train.py --generate --epochs 30

# 평가만 (체크포인트 필요)
python train.py --eval_only --checkpoint artifacts/best_model.pt
```

## Expected Results

| 지표 | Gram-Schmidt (baseline) | CNN (목표) |
|------|------------------------|-----------|
| Gain MAE | — | < 0.3 dB |
| Phase MAE | — | < 1.0 deg |
| NMSE 개선 | ~3 dB | > 8 dB |
| IRR 개선 | ~10 dB | > 20 dB |

이 저장소의 Gram-Schmidt 기준선은 gain/phase를 부분 보정하지만 DC 오프셋 제거
단계를 포함하지 않는다. 실제 고전적 I/Q 보정 파이프라인은 평균 제거 등 DC 보정을
별도로 결합할 수 있다.

평가 지표 설명:
- **Gain/Phase MAE:** 파라미터 추정 정확도
- **Signal NMSE:** 복원 신호와 원신호의 정규화 평균제곱오차
- **IRR (Image Rejection Ratio):** FFT 스펙트럼에서 원하는 성분 대비 허위 성분의 비율 (높을수록 좋음)

학습 포인트:
1. **해석적 역변환:** 모델이 파라미터를 예측하면 역공식으로 신호를 복원한다 — 블랙박스가 아닌 물리 기반 복원
2. **파라미터 스케일 정규화:** gain(dB), phase(deg), dc는 단위가 다르므로 손실에서 정규화 필수
3. **dual loss:** 파라미터 정확도만 최적화하면 신호 복원이 불안정할 수 있다; L1 signal loss가 안정화 역할
4. **IRR 지표:** 스펙트럼 허위 성분 제거 능력을 나타내는 실용적 지표 (레이다 시스템 스펙에 직결)

## Quick Start

```bash
# Smoke test (데이터 생성 + 2 에폭 학습, CPU)
python train.py --generate --smoke
```
