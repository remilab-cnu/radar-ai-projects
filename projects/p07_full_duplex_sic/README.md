# P07: Full-Duplex Self-Interference Cancellation (SIC)

## Task

전이중(Full-Duplex) 통신/레이다 시스템에서 TX 신호가 자체 수신 경로로 유입되어 발생하는
자기 간섭(Self-Interference, SI)을 딥러닝으로 추정·제거한다.

TX 기준 신호를 모델에 입력으로 제공하여, 수신 혼합 신호에서
SI 컴포넌트를 예측하고 잔차(residual)를 표적 신호로 복원한다.

신호 모델:
```
rx_mix = SI_channel(tx) + target_echo + noise
         ─────────────────   ──────────────────
              y_si               y_clean
```

| 텐서 | Shape | 설명 |
|------|-------|------|
| 입력 X | `(N, 4, 512)` | `concat(tx_ref, rx_mix)` |
| 출력 Y | `(N, 2, 512)` | SI 추정값 `si_hat` |
| 클린 복원 | `rx_mix - si_hat` | 후처리 (모델 외부) |

## Approach / Architecture

1D U-Net (`SICUNet`, ~300K 파라미터):

```
Encoder:
  enc1: Conv1d(4→32, k=7, stride=1)    → (B, 32, 512)
  enc2: Conv1d(32→64, k=3, stride=2)   → (B, 64, 256)
  enc3: Conv1d(64→128, k=3, stride=2)  → (B, 128, 128)

Bottleneck:
  Conv1d(128→256→128)                   → (B, 128, 128)

Decoder (skip connections):
  dec2: ConvTranspose1d(128→64) + skip  → (B, 64, 256)
  dec1: ConvTranspose1d(64→32) + skip   → (B, 32, 512)

Head:
  Conv1d(32→2, k=1)                     → (B, 2, 512)
```

손실 함수:
```
L = 0.7 × SmoothL1(si_hat, y_si) + 0.3 × SmoothL1(clean_hat, y_clean)
```

SI 추정 정확도와 클린 신호 복원 품질을 동시에 최적화한다.

## Data Generation

```bash
python generate_data.py   # 전체 데이터셋 생성
```

- **TX 기준 신호** (`tx_ref`): FMCW chirp, 2채널 [real, imag], 512 샘플
- **SI 채널**: 2~5탭 복소 FIR + 30% 확률로 3차 비선형항 추가
- **SIR**: -10 ~ +20 dB (SI가 표적보다 최대 10배 이상 강함)
- **SNR**: 5 ~ 25 dB
- **분할:** train 18K / val 3K / test 3K

HDF5 키:

| 키 | Shape | 설명 |
|----|-------|------|
| `tx_ref` | `(N, 2, 512)` | TX chirp [real, imag] |
| `rx_mix` | `(N, 2, 512)` | 수신 혼합 신호 |
| `y_si` | `(N, 2, 512)` | SI 컴포넌트 (GT) |
| `y_clean` | `(N, 2, 512)` | 표적+잡음 (eval용) |
| `sir_db` | `(N,)` | 샘플별 SIR [dB] |
| `snr_db` | `(N,)` | 샘플별 SNR [dB] |

## Training

```bash
# 전체 학습 (18K 학습 데이터, 30 에폭)
python train.py --generate --epochs 30

# 평가만 실행
python train.py --eval_only --checkpoint artifacts/best_model.pt

# 모델 shape 확인
python model.py
```

## Expected Results

| 지표 | NLMS (baseline) | SICUNet (목표) |
|------|----------------|---------------|
| Cancellation Depth (dB) | ~20 dB (선형) | >35 dB |
| Output SIR Gain (dB) | ~15 dB | >25 dB |
| Clean NMSE | — | <-15 dB |

**기준선**: 32탭 복소 NLMS 적응 필터

평가 지표:
- **Cancellation Depth (dB)**: `10·log10(‖y_si‖² / ‖y_si - si_hat‖²)`
- **Output SIR Gain (dB)**: 출력 SIR - 입력 SIR
- **Clean NMSE**: `‖y_clean - clean_hat‖² / ‖y_clean‖²`

학습 팁:
- SIR이 낮을수록 (SI >> 표적) 문제가 어려움 → SIR별 성능 분석 권장
- 비선형 SI는 선형 NLMS로 완전히 제거 불가 → DNN 장점이 부각되는 시나리오
- Bottleneck 크기(256)를 줄이면 모델 경량화 가능

## Quick Start

```bash
# 스모크 테스트 (빠른 동작 확인)
python train.py --generate --smoke
```
