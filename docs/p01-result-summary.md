# P01 Balanced Dataset Result Summary

Date: 2026-05-02 KST

This page summarizes the current full-data P01 result for classroom reporting.
Use the project README for commands and dataset details.

## Dataset contract

| Split | Samples | Shape | Schema | Clutter | Preprocessing | ADC clipping | fs/BW |
|---|---:|---|---:|---|---|---:|---:|
| train | 50,000 | `(50000, 2, 64, 200)` | 9 | static | slow-time mean-removal MTI/DC notch | 0.0 | 4.0 |
| val | 5,000 | `(5000, 2, 64, 200)` | 9 | static | slow-time mean-removal MTI/DC notch | 0.0 | 4.0 |
| test | 5,000 | `(5000, 2, 64, 200)` | 9 | static | slow-time mean-removal MTI/DC notch | 0.0 | 4.0 |

The target mask is a five-cell cross around the quantized target
range/Doppler bin.  Positive labels require the processed target-bin peak to be
at least 6 dB above both the global Range-Doppler median and a local
CFAR-like background ring.

Test-split label distribution:

- mean labelled targets per sample: `2.0598`;
- labelled target count range: `1` to `7`;
- `target_peak_snr_db` median: `21.398 dB`;
- labelled target fractions: `>=10 dB` `79.6%`, `>=15 dB` `66.1%`, `>=20 dB` `53.2%`;
- storage: `x=float16`, `y=uint8`, `rdm_mag_linear=float32`.

## Model

- Model: `UNetDetector(base_ch=32, input_mode=mag_phase)`
- Parameters: `7,762,753`
- Epochs: `30`
- Batch size: `32`
- Learning rate: `3e-4`
- Best validation loss: `0.0896`

## Validation-locked test metrics

Detector settings were selected on validation data and then applied to held-out
test samples.

| Detector | Policy | Pd | Pfa | Precision | F1 | Target recall | False alarms / RDM |
|---|---|---:|---:|---:|---:|---:|---:|
| CA-CFAR | guard `(1,1)`, train `(4,4)`, design Pfa `1e-5` | 0.5429 | 1.928e-4 | 0.6947 | 0.6095 | 0.7528 | 2.466 |
| U-Net | threshold `0.2` | 0.7740 | 6.974e-5 | 0.8997 | 0.8322 | 0.8411 | 0.892 |

## SNR-binned metrics

| SNR bin (dB) | Samples | CFAR Pd | CFAR F1 | U-Net Pd | U-Net F1 |
|---|---:|---:|---:|---:|---:|
| 5--10 | 243 | 0.472 | 0.567 | 0.696 | 0.787 |
| 10--15 | 230 | 0.520 | 0.603 | 0.765 | 0.829 |
| 15--20 | 244 | 0.560 | 0.623 | 0.807 | 0.851 |
| 20--25 | 283 | 0.596 | 0.629 | 0.811 | 0.850 |

## Classroom interpretation

P01 is a controlled moving-target detector after MTI/DC-notch preprocessing.
The result supports the lesson that a U-Net can use spatial context in the
Range-Doppler map to improve Pd/F1 and reduce false alarms compared with a local
CA-CFAR policy under the same processed-input contract.
