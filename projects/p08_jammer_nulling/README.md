# P08: Jammer Null Steering

## Task

8-element ULA(Uniform Linear Array) 수신 시스템에서 재머(Jammer) 방향을 신경망으로 추정하고,
LCMV(Linearly Constrained Minimum Variance) 빔포머의 null steering에 활용한다.

수신 공분산 행렬과 원하는 신호의 look angle을 입력으로,
재머 각도를 회귀 추정하여 추정 방향에 null을 배치한다.

신호 모델:
```
x(t) = a(θ_s)·s(t) + Σ a(θ_j)·j(t) + n(t)
        ──────────     ──────────────────────
         원하는신호         재머 1~2개 + 잡음
```

| 텐서 | Shape | 설명 |
|------|-------|------|
| 공분산 행렬 `cov` | `(N, 2, 8, 8)` | 샘플 공분산의 real/imag |
| look angle | `(N, 1)` | 원하는 신호 방향 [deg] |
| 출력 (학습) | `(N, 1)` | sin(jammer_angle) |
| 출력 (추론 후) | scalar | jammer_angle = arcsin(output) × 180/π |

## Approach / Architecture

CovNet (~55K 파라미터):

```
입력: (B, 2, 8, 8) 공분산 행렬

Conv2d(2→16, 3×3) -BN-ReLU  → (B, 16, 6, 6)
Conv2d(16→32, 3×3)-BN-ReLU  → (B, 32, 4, 4)
Conv2d(32→64, 3×3)-BN-ReLU  → (B, 64, 2, 2)
Global Average Pooling        → (B, 64)

Concat with look_angle/90     → (B, 65)
FC(65→128)-ReLU → FC(128→1)  → (B, 1): sin(θ_jammer)
```

각도 복원: `θ_jammer = arcsin(clamp(output, -1, 1)) × 180/π`

손실 함수:
```
L = SmoothL1(sin(θ_pred), sin(θ_true))
```

원시 각도(degree) 대신 `sin(θ)`를 예측하면 범위가 `[-1, 1]`로 제한되어 최적화가 안정적이다.

## Data Generation

```bash
python generate_data.py   # 전체 데이터셋 생성
```

- **ULA**: 8 elements, λ/2 간격
- **재머 수**: 1 또는 2 (랜덤)
- **JNR**: 10 ~ 40 dB
- **SNR**: 0 ~ 20 dB
- **분할:** train 24K / val 4K / test 4K

HDF5 키:

| 키 | Shape | 설명 |
|----|-------|------|
| `cov` | `(N, 2, 8, 8)` | 공분산 real/imag (Frobenius norm 정규화) |
| `look_angle_deg` | `(N,)` | 원하는 신호 방향 [deg] |
| `jammer_angle_deg` | `(N,)` | 강한 재머 방향 (GT) |
| `jnr_db` | `(N,)` | Jammer-to-Noise Ratio [dB] |
| `snr_db` | `(N,)` | Signal-to-Noise Ratio [dB] |
| `n_jammers` | `(N,)` | 재머 수 (1 or 2) |

## Training

```bash
# 전체 학습 (24K 학습 데이터, 30 에폭)
python train.py --generate --epochs 30

# 평가만 실행
python train.py --eval_only --checkpoint artifacts/best_model.pt

# 모델 shape 확인
python model.py
```

## Expected Results

| 지표 | MUSIC (baseline) | CovNet (목표) |
|------|-----------------|--------------|
| Angle MAE (deg) | ~2 deg | <1.5 deg |
| Within ±2 deg Acc | ~60% | >80% |
| LCMV Null Depth (dB) | varies | <-30 dB |

**기준선**: MUSIC 알고리즘으로 재머 방향 추정 → LCMV null steering

학습 팁:
- JNR이 높을수록 재머가 공분산 행렬에 뚜렷이 나타나 추정이 쉬워짐
- 2-재머 케이스에서 약한 재머는 예측 목표가 아님 → 강한 재머가 index 0으로 고정됨
- look_angle 정규화: `/90.0` → `[-1, 1]` 범위로 스케일링

## Quick Start

```bash
# 스모크 테스트
python train.py --generate --smoke
```
