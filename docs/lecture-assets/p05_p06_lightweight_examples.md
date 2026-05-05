# P05/P06 MATLAB reference 기반 lightweight examples

기준 시점: 2026-05-06 KST

이 문서는 P05와 P06의 기술적 기준을 정리한다. 두 프로젝트는 MATLAB Radar/AI 예제를
Python으로 작게 옮긴 lightweight example이며, 외부 데이터 없이 CPU에서 실행할 수 있는
형태로 구성한다.

| 프로젝트 | 핵심 용도 | MATLAB reference | Python 구현 위치 |
|---|---|---|---|
| P05 | Radar waveform classification | Radar and Communications Waveform Classification | `projects/p05_waveform_classification/` |
| P06 | Radar target signature classification | Radar Target Classification Using Machine Learning and Deep Learning | `projects/p06_target_signature_classification/` |

## P05 — Lightweight radar waveform classification example

### 정확한 과제 정의

P05는 synthetic complex-baseband waveform을 만든 뒤, STFT log-magnitude image로 변환하고
waveform family를 분류하는 lightweight example이다. 기본 class는 `rect`, `lfm`,
`barker`, `noise_only`이다.

강조할 문장:

> P05는 target detection이 아니라 waveform family classification이다. 입력은 하나의
> baseband observation이고, 모델은 time-frequency structure를 보고 waveform 계열을 구분한다.

### MATLAB reference와 Python equivalent

| MATLAB 예제 흐름 | Python equivalent |
|---|---|
| Radar/communication waveform 생성 | `shared/waveform_library.py`의 rectangular/LFM/Barker/noise generator |
| time-frequency representation 생성 | `stft_log_image()` |
| CNN classification | `projects/p05_waveform_classification/model.py`의 `TinyWaveformCNN` |
| SNR별 성능 확인 | `evaluate_snr_sweep.py` |
| classical comparison | `evaluate_feature_baseline.py` |

Reference:

- https://www.mathworks.com/help/radar/ug/radar-and-communications-waveform-classification-using-deep-learning.html
- https://www.mathworks.com/help/radar/ug/LPI-radar-waveform-classification-using-time-frequency-CNN.html

### 안전한 해석

> P05 결과는 lightweight synthetic waveform 조건에서의 분류 성능이다. 실제 수신기 환경이나
> LPI waveform library 전체에 대한 성능 주장으로 해석하지 않는다.

정확한 표현:

- P05는 rectangular/LFM/Barker/noise-only 계열을 구분하는 small-scale waveform example이다.
- STFT window, SNR, frequency offset 조건이 confusion matrix에 직접 영향을 준다.
- `noise_only` class는 unknown/no-signal condition을 다루기 위한 간단한 negative class이다.
- full operational waveform recognition 성능을 주장하지 않는다.

## P06 — Lightweight target signature classification example

### 정확한 과제 정의

P06는 simple target geometry를 point scatterer로 표현하고, aspect angle 변화에 따른 complex
monostatic response를 생성한 뒤 target signature family를 분류하는 lightweight example이다.
기본 class는 `cylinder`, `cone`, `plate`이다.

강조할 문장:

> P06는 SAR image classification이 아니라 target signature classification이다. 모델은 aspect와
> SNR이 바뀔 때 magnitude/phase return sequence가 어떻게 달라지는지를 이용해 class를 구분한다.

### MATLAB reference와 Python equivalent

| MATLAB 예제 흐름 | Python equivalent |
|---|---|
| 단순 target geometry 설정 | `shared/target_signature.py`의 cylinder/cone/plate scatterer generator |
| aspect-dependent radar return 생성 | `complex_aspect_response()` |
| ML/DL classifier 비교 | `evaluate_feature_baseline.py`, `TinySignatureCNN` |
| aspect/SNR 일반화 확인 | `evaluate_generalization.py` |

Reference:

- https://www.mathworks.com/help/radar/ug/radar-target-classification-using-machine-learning-and-deep-learning.html

### 안전한 해석

> P06 결과는 point-scatterer teaching model에서의 target signature 분류 결과다. 실제 CAD/EM
> solver나 측정 RCS database를 사용한 ATR 성능으로 해석하지 않는다.

정확한 표현:

- P06는 simple geometry 기반의 aspect-varying signature example이다.
- `plate` class는 cylinder/cone reference 흐름을 3-class 교육 예제로 확장하기 위한 lightweight class이다.
- held-out aspect와 low-SNR 조건에서는 기본 IID test보다 성능이 다르게 보일 수 있다.
- P04 SAR despeckling과 달리 P06는 image 복원이 아니라 1-D signature classification이다.

## 공통 실행 확인

```bash
# P05
cd projects/p05_waveform_classification
python train.py --generate --smoke
python evaluate_snr_sweep.py --checkpoint artifacts/best_model.pt --base_ch 8 --smoke
cd ../..

# P06
cd projects/p06_target_signature_classification
python train.py --generate --smoke
python evaluate_generalization.py --generate --checkpoint artifacts/best_model.pt --base_ch 8 --smoke
cd ../..
```

## 공통 경계

- 두 프로젝트는 lightweight example이다.
- MATLAB 예제의 교육적 구조를 따르지만, toolbox 내부 구현을 그대로 복제하지 않는다.
- 결과는 generator, SNR range, aspect range, STFT/signature representation 조건과 함께 해석한다.
- 최고 정확도 하나보다 confusion matrix와 조건별 metric이 더 중요한 기술 자료다.
