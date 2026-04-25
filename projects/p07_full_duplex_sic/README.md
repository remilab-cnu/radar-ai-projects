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

- **TX 기준 신호** (`tx_ref`): chirp-like complex reference, 2채널 [real, imag], 512 샘플
- **SI 채널**: 2~5탭 복소 FIR + 30% 확률로 3차 비선형항 추가
- **표적 에코**: TX 기준 chirp를 지연시키고 Doppler phase를 곱한 waveform-consistent
  echo. 전파/beat-chain 전체를 풀지는 않지만, 표적 성분은 더 이상 독립 tone이 아니다.
- **`isr_db` 저장값**: -10 ~ +20 dB의 SI-to-echo power ratio
  (`P_si/P_echo`; +20 dB이면 SI가 표적 에코보다 100배 강하고, -10 dB이면 0.1배)
- **`sir_db` 저장값**: 기존 노트북/스크립트 호환을 위한 legacy alias이며 `isr_db`와 같은 값이다.
- **SNR**: 5 ~ 25 dB, 기준 전력은 표적 에코(`P_echo/P_noise`)이다.
- **분할:** train 18K / val 3K / test 3K

HDF5 키:

| 키 | Shape | 설명 |
|----|-------|------|
| `tx_ref` | `(N, 2, 512)` | TX chirp [real, imag] |
| `rx_mix` | `(N, 2, 512)` | 수신 혼합 신호 |
| `y_si` | `(N, 2, 512)` | SI 컴포넌트 (GT) |
| `y_clean` | `(N, 2, 512)` | 표적+잡음 (eval용) |
| `isr_db` | `(N,)` | 샘플별 SI-to-echo ratio [dB] |
| `sir_db` | `(N,)` | `isr_db`와 동일한 legacy alias |
| `snr_db` | `(N,)` | 샘플별 echo-to-noise SNR [dB] |
| `nonlinear` | `(N,)` | 3차 SI leakage 포함 여부 |
| `si_power`, `target_echo_power`, `noise_power` | `(N,)` | 전력 label 검증용 생성 시점 측정값 |
| `measured_isr_db`, `measured_snr_db` | `(N,)` | `isr_db`, `snr_db`와 비교 가능한 사후 검증 label |

## Physics Contract / Allowed Simplification / Not Claimed

- **Physics contract:** `rx_mix = y_si + target_echo + noise`; `y_si`는 TX 기준 파형의
  짧은 복소 FIR 누설(일부 샘플은 cubic leakage 포함)이고, 표적 에코는 같은 `tx_ref`의
  delayed/Doppler-scaled copy이다. `isr_db`는 측정 전력 기준 `P_si/P_echo`, `snr_db`는
  `P_echo/P_noise`를 의미하며, HDF5에는 이를 재계산할 수 있는 measured power metadata를
  함께 저장한다.
- **Allowed simplification:** 단일 TX chirp, 단일 표적, 짧은 baseband record, 통계적 채널
  랜덤화를 사용한다. 빠른 CPU handout을 위해 안테나 격리, ADC, RF front-end 포화,
  full FMCW range-Doppler processing은 생략한다.
- **Not claimed:** 실제 전이중 RF 하드웨어의 100+ dB cancellation, 다중경로/다중표적
  scene fidelity, calibrated PA/LO/ADC impairment model, 또는 deployment-ready SIC가 아니다.
  이 프로젝트는 waveform-consistent synthetic handout과 NLMS-vs-DNN 비교를 목표로 한다.

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
| Cancellation Depth (dB) | 선형 샘플에서 더 높고 비선형 샘플에서 낮음 | >35 dB |
| Output SIR Gain (dB) | ~15 dB | >25 dB |
| Clean NMSE | — | <-15 dB |

**기준선**: 8탭 복소 NLMS 적응 필터. 생성기의 SI FIR은 2~5탭이므로 선형 샘플은
상대적으로 잘 추적하지만, cubic leakage가 있는 샘플은 `nlms_nonlinear_cancellation_db_mean`
에서 한계를 보인다.

평가 지표:
- **Cancellation Depth (dB)**: `10·log10(‖y_si‖² / ‖y_si - si_hat‖²)`
- **Output SIR Gain (dB)**: 출력 clean-to-residual-SI 비율 − 입력 clean-to-SI 비율
- **Clean NMSE**: `‖y_clean - clean_hat‖² / ‖y_clean‖²`

학습 팁:
- 저장된 `isr_db`/legacy `sir_db`가 높을수록 (SI >> 표적) 문제가 어려움 → ISR 구간별 성능 분석 권장
- 비선형 SI는 선형 NLMS로 완전히 제거 불가 → DNN 장점이 부각되는 시나리오
- Bottleneck 크기(256)를 줄이면 모델 경량화 가능

## Quick Start

```bash
# 스모크 테스트 (빠른 동작 확인)
python train.py --generate --smoke
```
