# Shared FMCW / Clean Burst Simulator Technical Contract

This document defines the simulator contract used by the active radar AI
projects.  It is intended for students and instructors who need the exact signal
and data-processing assumptions.

## 1. FMCW simulator identity

`shared/fmcw_simulator.py` is a complex-baseband FMCW dechirp simulator:

```text
carrier       : wavelength, carrier phase, Doppler, antenna phase, radar equation
bandwidth     : active FMCW signal bandwidth
sampling      : fs = 4 * bandwidth for active configurations
waveform      : baseband FMCW up-chirp
receive echo  : delayed, Doppler-shifted, scaled chirp copies
mixer         : beat(t) = rx(t) * conj(tx(t))
range process : range FFT on dechirped beat signal
Doppler       : FFT over coherent chirps
RF passband   : not explicitly sampled
```

The carrier is therefore physically meaningful even though the simulator does
not create RF passband samples.

## 2. FMCW signal model

For target range `R0`, closing velocity `v`, chirp index `m`, and fast-time `t`:

```text
slope = bandwidth / T_chirp
R_m   = R0 - v * m * PRI
τ_m   = 2 R_m / c
f_d   = 2 v / λ

tx(t) = exp(j π slope t²)
rx(t) += sqrt(Pr(R_m)) * tx(t - τ_m)
          * exp(-j 2π fc τ_m)
          * exp(j 2π f_d t)
          * exp(j array_phase)
          * exp(j scatter_phase)
beat(t) = rx(t) * conj(tx(t))
```

Received power and thermal noise use:

```text
Pr = Pt Gt Gr λ² σ / ((4π)³ R⁴ L)
Pn = k T B NF
```

For this up-chirp and mixer convention, positive target ranges appear on the
negative beat-frequency side.  `range_fft(..., radar=radar)` returns zero range
followed by increasing positive range bins in the correct order.

## 3. Sampling and timing

Active configurations use:

```text
fs = 4 * bandwidth
```

`T_chirp` is the FMCW sweep duration.  `PRI` is the slow-time repetition interval
and may be longer than the sweep duration.  `FMCWRadar` checks the sampling rule
by default.

## 4. Public API surface

| Module | Public surface | Contract |
|---|---|---|
| `shared/fmcw_simulator.py` | `FMCWRadar`, `generate_scene`, `range_fft`, `pulse_compress`, `range_doppler_map`, `range_angle_map`, `target_rd_bins`, `range_axis`, `velocity_axis`, `ca_cfar_1d`, `ca_cfar_2d`, `to_db` | Complex-baseband FMCW teaching core with radar metadata and common DSP helpers. |
| `shared/fmcw_simulator.py` | `encode_complex_iq_signed`, `decode_complex_iq_signed`, `quantize_complex_iq` | Signed fixed-point I/Q helpers.  Complex 16-bit I/Q means int16 I plus int16 Q components. |
| `shared/clutter_model.py` | `generate_scene_with_clutter`, `generate_random_scene`, `apply_slow_time_mti` | P01 static-clutter generation, MTI preprocessing, target masks, and compact RDM storage. |
| `shared/micro_doppler.py` | `generate_har_sample`, `generate_range_compressed_micro_doppler_frame`, `extract_target_range_signal`, `build_p02_pedestrian_scatterers`, `extract_handcrafted_features` | P02 target-range micro-Doppler generator and features. |
| `shared/burst_simulator.py` | `BurstRadar`, `CleanBurstSimulator`, `lfm_pulse`, `range_axis`, `velocity_axis` | Clean coherent pulse-burst and matched-filter reference simulator. |

## 5. Project boundaries

| Project | Simulator boundary |
|---|---|
| P01 | Uses shared FMCW beat simulation plus static clutter.  Raw beat data is quantized as complex 16-bit I/Q, passed through slow-time MTI/DC-notch preprocessing, converted to a Range-Doppler map, then used by both CA-CFAR and U-Net. |
| P02 | Uses body kinematics and a project-specific pedestrian scatterer model to form a local range-compressed frame.  The spectrogram comes from the simulator-known target range. |
| P03 | Uses shared FMCW Range-Doppler processing to extract antenna vectors for DoA estimation, then evaluates map quality after projection into world coordinates. |
| P04 | Uses real Sentinel-1 patch extraction in the project folder.  It does not use the FMCW simulator for its main despeckling results. |

## 6. Verification

Run the contract tests from the repository root:

```bash
pytest -q tests/test_fmcw_simulator_contract.py tests/test_burst_simulator_contract.py tests/test_project_simulator_wiring.py
pytest -q tests/test_p1_detection_quality_contract.py tests/test_p2_scatter_model_contract.py
```

The tests cover sampling, target range/Doppler placement, array phase,
matched-filter peaks, thermal noise scaling, P01 label/quantization metadata,
and P02 scatterer/target-range metadata.
