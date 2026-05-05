# P1/P2 강의 보강 기술 노트

기준 시점: 2026-05-05 KST

생성 코드:

```bash
python3 scripts/make_p01_p02_lecture_assets.py
```

생성 위치: `docs/lecture-assets/`

| 파일 | 핵심 용도 |
|---|---|
| `p01_signal_chain.png` | P1의 실제 처리 흐름: FMCW beat → 16-bit I/Q → MTI → RDM → label/CFAR/U-Net |
| `p01_label_mask_definition.png` | P1 label이 5-cell cross mask라는 점 설명 |
| `p02_target_range_equations.png` | P2 target-range micro-Doppler 생성식과 alias guard 설명 |
| `p02_stress_interpretation.png` | P2 default 포화와 aspect/SNR/range stress 해석 |

## P1 — U-Net FMCW RDM detector

### 정확한 과제 정의

P1은 static clutter가 포함된 controlled FMCW scene에서 raw beat 신호를 만든 뒤,
fixed-scale complex 16-bit I/Q quantization과 slow-time mean-removal MTI/DC-notch를 거쳐
Range-Doppler map을 만든다. U-Net과 CA-CFAR는 같은 MTI-filtered RDM을 입력으로 받는다.

강의에서 강조할 문장:

> P1은 raw clutter cell을 전부 검출하는 neural CFAR 과제가 아니라, MTI 이후의 moving-target mask를 검출하는 과제다.

### Label gate와 mask

Schema-v9 positive label은 target bin의 processed peak가 다음 배경 기준보다 6 dB 이상 클 때만 생성된다.

```text
background = max(global RD median, local CFAR-like ring median)
target_peak_snr_db = 20 log10(target_peak / background)
positive if target_peak_snr_db >= 6 dB
```

mask는 target 중심 cell과 상하좌우 이웃을 포함한 5-cell cross다.

### 현재 보고 가능한 수치

출처:

- `projects/p01_unet_detector/artifacts/full_eval/p01_cfar_selected_test.json`
- `projects/p01_unet_detector/artifacts/full_eval/p01_unet_selected_test.json`
- `docs/p01-schema9-balanced-full-run-summary.md`

1,000개 held-out test sample에서 validation-locked policy로 평가한 값:

| Detector | Policy | Pd | Pfa | Precision | F1 | Target recall | False alarms/RDM |
|---|---|---:|---:|---:|---:|---:|---:|
| CA-CFAR | guard `(1,1)`, train `(4,4)`, design Pfa `1e-5` | 0.5429 | 1.928e-4 | 0.6947 | 0.6095 | 0.7528 | 2.466 |
| U-Net | threshold `0.2` | 0.7740 | 6.974e-5 | 0.8997 | 0.8322 | 0.8411 | 0.892 |

안전한 해석:

> Schema-v9에서 U-Net은 CA-CFAR보다 높은 Pd/F1을 보이고 false alarm도 적다. 이 결과는 MTI-filtered moving-target detection 조건에서의 비교다.

정확한 표현:

- P1은 MTI 이후 moving-target mask를 검출하는 과제다.
- label은 중심 cell과 상하좌우 이웃으로 구성된 5-cell cross다.
- 현재 schema-v9 결과에서는 U-Net이 CA-CFAR보다 Pd와 F1이 모두 높다.
- 현재 lecture-grade 표는 1,000개 held-out test sample 기준이다.

## P2 — Target-range micro-Doppler HAR

### 정확한 과제 정의

P2는 full raw FMCW cube를 처리하는 과제가 아니다. P02-only pedestrian scatterer model로 사람의 body motion을 scatterer로 펼친 뒤, target 주변의 local range-compressed frame을 만들고 simulator-known target range에서 complex slow-time signal을 추출한다. 이후 STFT로 micro-Doppler spectrogram을 만들고 6개 활동을 분류한다.

강의에서 강조할 문장:

> P2는 target-range micro-Doppler HAR이다. full raw FMCW dechirp cube가 아니라, P02 전용 scatterer model과 target-range extraction을 사용한다.

### 핵심 식

Aspect angle은 현재 2-D radial projection에서 절대값 관례를 사용한다. signed aspect는 `cos(theta)` 때문에 같은 효과를 낸다.

```text
R_k(t, theta) = R_0 + x_k(t) cos(theta) + delta_k(t) cos(theta)

s(t, r_n) = sum_k a_k sinc((r_n - R_k(t, theta)) / Delta R)
            exp(-j 4 pi R_k(t, theta) / lambda)

z(t) = s(t, r_target)
X(tau, f_D) = |STFT{z(t)}|^2

v_max = lambda PRF / 4
```

`doppler_alias_margin_mps`는 각 sample에 저장된다. 기본 데이터는 alias margin이 양수가 되도록 제한한다.

### 현재 보고 가능한 수치

출처:

- `projects/p02_resnet18_har/artifacts/full_eval/p02_default_comparison_summary_compact.json`
- `projects/p02_resnet18_har/artifacts/stress_eval/p02_stress_comparison_summary.md`
- `projects/p02_resnet18_har/artifacts/stress_eval/report/p02_stress_visual_report.html`

Default IID split은 거의 포화되어 있다.

| Method | Default test accuracy |
|---|---:|
| LogReg handcrafted | 98.13% |
| RBF SVM handcrafted | 98.63% |
| TinyCNN | 100.00% |
| ResNet18 | 100.00% |

따라서 강의의 핵심은 stress generalization이다.

| Stress set | Handcrafted RBF SVM | TinyCNN | ResNet18 | 해석 |
|---|---:|---:|---:|---|
| Aspect `[60°,80°]` | 62.40% | 87.30% | 85.77% | radial Doppler가 `cos(theta)`로 줄어들어 class 간 차이가 작아짐 |
| Low SNR `[0,8] dB` | 83.73% | 99.87% | 100.00% | morphology가 noisy해질 때 handcrafted descriptor가 더 크게 흔들림 |
| Far range `[18,26] m` | 98.53% | 100.00% | 100.00% | generator가 SNR을 명시 샘플링하므로 range effect가 작게 보임 |

안전한 해석:

> P2의 default accuracy는 모델 비교를 설명하기엔 너무 쉽다. Aspect와 SNR stress를 보면 handcrafted descriptor와 CNN의 일반화 차이가 드러난다.

정확한 표현:

- P2는 target-range micro-Doppler HAR이다.
- 기본 aspect는 absolute sector `[0°, 60°]` 조건이다.
- far-range stress는 generator가 SNR을 명시 샘플링하는 설계 때문에 영향이 작게 나타난다.
- default accuracy보다 aspect/SNR stress가 일반화 설명에 더 유용하다.

## 산출물별 권장 캡션

### `p01_signal_chain.png`

P1의 schema-v9 처리 흐름. Raw scene에는 static clutter가 있지만, detector 입력은 slow-time mean-removal MTI를 통과한 RDM이다. U-Net과 CA-CFAR는 같은 MTI-filtered RDM에서 validation-locked policy로 비교한다.

### `p01_label_mask_definition.png`

P1 positive label은 target peak가 global/local background보다 6 dB 이상 높을 때만 생성된다. Mask는 중심 cell과 상하좌우 이웃으로 구성된 5-cell cross다.

### `p02_target_range_equations.png`

P2 micro-Doppler는 P02-only scatterer model의 range-compressed target response에서 나온다. Aspect는 radial projection을 통해 Doppler bandwidth를 줄이며, target range의 slow-time signal을 STFT해 spectrogram을 만든다.

### `p02_stress_interpretation.png`

P2 default split은 대부분의 방법이 높은 정확도를 보이는 쉬운 IID 조건이다. Aspect와 low-SNR stress에서 일반화 차이가 드러나며, far-range stress는 SNR을 명시 샘플링하는 generator 설계 때문에 영향이 작게 나타난다.
