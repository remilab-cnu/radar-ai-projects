# P08: Jammer Null Steering

## 과제 개요

8-element ULA(Uniform Linear Array) 수신 시스템에서 재머(Jammer) 방향을 신경망으로 추정하고,
LCMV(Linearly Constrained Minimum Variance) 빔포머의 null steering에 활용한다.

**핵심 아이디어**: 수신 공분산 행렬과 원하는 신호의 look angle을 입력으로,
재머 각도를 회귀 추정 → 추정 방향에 null 배치.

---

## 신호 모델

```
x(t) = a(θ_s)·s(t) + Σ a(θ_j)·j(t) + n(t)
        ──────────     ──────────────────────
         원하는신호         재머 1~2개 + 잡음
```

- **ULA**: 8 elements, λ/2 간격
- **재머 수**: 1 또는 2 (랜덤)
- **JNR**: 10 ~ 40 dB (재머가 잡음보다 훨씬 강함)
- **SNR**: 0 ~ 20 dB
- **2재머 케이스**: 강한 재머를 예측 목표로 설정

---

## 입출력 형태

| 텐서 | Shape | 설명 |
|------|-------|------|
| 공분산 행렬 `cov` | `(N, 2, 8, 8)` | 샘플 공분산의 real/imag |
| look angle | `(N, 1)` | 원하는 신호 방향 [deg] |
| 출력 (학습) | `(N, 1)` | sin(jammer_angle) |
| 출력 (추론 후) | scalar | jammer_angle = arcsin(output) × 180/π |

---

## 모델: CovNet (~55K 파라미터)

```
입력: (B, 2, 8, 8) 공분산 행렬

Conv2d(2→16, 3×3) -BN-ReLU  → (B, 16, 6, 6)
Conv2d(16→32, 3×3)-BN-ReLU  → (B, 32, 4, 4)
Conv2d(32→64, 3×3)-BN-ReLU  → (B, 64, 2, 2)
Global Average Pooling        → (B, 64)

Concat with look_angle/90     → (B, 65)
FC(65→128)-ReLU → FC(128→1)  → (B, 1): sin(θ_jammer)
```

**각도 복원**: `θ_jammer = arcsin(clamp(output, -1, 1)) × 180/π`

---

## 왜 sin(angle)을 예측하는가?

원시 각도(degree)를 직접 예측하면 `-90°`와 `+90°` 부근에서 gradient가 불안정하다.
`sin(θ)`는 연속적이고 범위가 `[-1, 1]`로 제한되어 최적화가 안정적이다.

---

## 손실 함수

```
L = SmoothL1(sin(θ_pred), sin(θ_true))
```

---

## 평가 지표

| 지표 | 설명 |
|------|------|
| **Angle MAE (deg)** | `|θ_true - θ_pred|` 평균 |
| **Within ±2 deg Acc** | 2도 이내 정확도 |
| **LCMV Null Depth (dB)** | 예측 방향에 null을 둔 경우의 null 깊이 (낮을수록 좋음) |

**기준선**: MUSIC 알고리즘으로 재머 방향 추정 → LCMV null steering

---

## 실행 방법

```bash
cd projects/p08_jammer_nulling

# 1. 스모크 테스트
python train.py --generate --smoke

# 2. 전체 학습 (24K 학습 데이터, 30 에폭)
python train.py --generate --epochs 30

# 3. 평가만 실행
python train.py --eval_only --checkpoint artifacts/best_model.pt

# 4. 데이터만 생성
python generate_data.py

# 5. 모델 shape 확인
python model.py
```

---

## 파일 구조

```
p08_jammer_nulling/
├── generate_data.py   # HDF5 데이터셋 생성
├── model.py           # CovNet
├── train.py           # 학습 + 평가 + MUSIC/LCMV 기준선
├── data/
│   ├── train.h5       # 24K (smoke: 256)
│   ├── val.h5         # 4K  (smoke: 64)
│   └── test.h5        # 4K  (smoke: 64)
└── artifacts/
    ├── best_model.pt
    ├── history.json
    └── metrics.json
```

---

## HDF5 키

| 키 | Shape | 설명 |
|----|-------|------|
| `cov` | `(N, 2, 8, 8)` | 공분산 real/imag (Frobenius norm 정규화) |
| `look_angle_deg` | `(N,)` | 원하는 신호 방향 [deg] |
| `jammer_angle_deg` | `(N,)` | 강한 재머 방향 (GT, 회귀 목표) |
| `jnr_db` | `(N,)` | Jammer-to-Noise Ratio [dB] |
| `snr_db` | `(N,)` | Signal-to-Noise Ratio [dB] |
| `n_jammers` | `(N,)` | 재머 수 (1 or 2) |

---

## 학습 팁

- JNR이 높을수록 재머가 공분산 행렬에 뚜렷이 나타나 추정이 쉬워짐
- 2-재머 케이스에서 약한 재머는 예측 목표가 아님 → 혼동 방지를 위해 data 생성 시 강한 재머가 index 0으로 고정됨
- Conv2d 패딩을 추가하면 모델 크기를 키울 수 있음 (현재 zero-padding 없음)
- look_angle 정규화: `/90.0` → `[-1, 1]` 범위로 스케일링
