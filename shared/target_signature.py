"""Lightweight aspect-dependent radar target signatures for P06.

This module provides a compact Python equivalent of the target-return workflow
used in MATLAB radar target-classification examples: simple target geometry,
aspect-varying complex returns, handcrafted features, and small neural-network
inputs.  It is not a full electromagnetic solver; it is a deterministic teaching
model that preserves the link between scatterer geometry, aspect angle, SNR, and
classification difficulty.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fmcw_simulator import C, add_complex_awgn

TARGET_CLASSES = ("cylinder", "cone", "plate")
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TargetSignatureConfig:
    """Configuration for P06 target-signature examples."""

    fc_hz: float = 850.0e6
    n_samples: int = 128
    aspect_range_deg: tuple[float, float] = (-45.0, 45.0)
    snr_range_db: tuple[float, float] = (6.0, 24.0)
    vibration_range_deg: tuple[float, float] = (0.5, 4.0)
    aspect_jitter_std_deg: float = 0.25
    n_scatterers: int = 48

    @property
    def wavelength_m(self) -> float:
        return C / self.fc_hz


def _rng(rng: np.random.Generator | None) -> np.random.Generator:
    return np.random.default_rng() if rng is None else rng


def make_cylinder_scatterers(rng: np.random.Generator | None = None, n: int = 48) -> np.ndarray:
    """Return cylinder-like point scatterers `(x, y, rcs)` in meters."""

    rng = _rng(rng)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radius = rng.uniform(0.85, 1.15)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    rcs = rng.lognormal(mean=0.0, sigma=0.15, size=n)
    return np.column_stack([x, y, rcs]).astype(np.float64)


def make_cone_scatterers(rng: np.random.Generator | None = None, n: int = 48) -> np.ndarray:
    """Return cone-like asymmetric point scatterers `(x, y, rcs)` in meters."""

    rng = _rng(rng)
    axial = np.linspace(-1.2, 1.2, n)
    taper = np.clip((1.2 - axial) / 2.4, 0.05, 1.0)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    radius = taper * rng.uniform(0.5, 1.0, size=n)
    x = radius * np.cos(phi)
    y = axial + 0.2 * radius * np.sin(phi)
    rcs = 0.5 + 1.5 * taper + rng.lognormal(mean=-0.3, sigma=0.2, size=n)
    return np.column_stack([x, y, rcs]).astype(np.float64)


def make_plate_scatterers(rng: np.random.Generator | None = None, n: int = 48) -> np.ndarray:
    """Return flat-plate-like scatterers `(x, y, rcs)` in meters."""

    rng = _rng(rng)
    side = int(np.ceil(np.sqrt(n)))
    xs = np.linspace(-1.8, 1.8, side)
    ys = np.linspace(-0.25, 0.25, side)
    xv, yv = np.meshgrid(xs, ys)
    pts = np.column_stack([xv.ravel(), yv.ravel()])[:n]
    pts += rng.normal(0.0, 0.03, size=pts.shape)
    edge_boost = 1.0 + 0.8 * (np.abs(pts[:, 0]) / max(np.max(np.abs(pts[:, 0])), 1e-6))
    rcs = edge_boost * rng.lognormal(mean=0.1, sigma=0.15, size=len(pts))
    return np.column_stack([pts[:, 0], pts[:, 1], rcs]).astype(np.float64)


def make_target_scatterers(class_name: str, rng: np.random.Generator | None = None, n: int = 48) -> np.ndarray:
    if class_name == "cylinder":
        return make_cylinder_scatterers(rng, n)
    if class_name == "cone":
        return make_cone_scatterers(rng, n)
    if class_name == "plate":
        return make_plate_scatterers(rng, n)
    raise ValueError(f"unknown target class: {class_name!r}")


def rotate_scatterers(scatterers: np.ndarray, aspect_deg: float) -> np.ndarray:
    """Rotate `(x, y, rcs)` scatterers by an aspect angle in the horizontal plane."""

    sc = np.asarray(scatterers, dtype=np.float64)
    theta = np.deg2rad(aspect_deg)
    c, s = np.cos(theta), np.sin(theta)
    x = sc[:, 0] * c - sc[:, 1] * s
    y = sc[:, 0] * s + sc[:, 1] * c
    return np.column_stack([x, y, sc[:, 2]])


def complex_aspect_response(
    scatterers: np.ndarray,
    aspect_deg: float | np.ndarray,
    wavelength_m: float,
    *,
    class_name: str | None = None,
) -> np.ndarray:
    """Compute a far-field monostatic complex response versus aspect angle."""

    sc = np.asarray(scatterers, dtype=np.float64)
    aspect = np.atleast_1d(np.asarray(aspect_deg, dtype=np.float64))
    theta = np.deg2rad(aspect)
    x = sc[:, 0][None, :]
    y = sc[:, 1][None, :]
    amp = np.sqrt(np.maximum(sc[:, 2], 1e-12))[None, :]
    projection = np.sin(theta)[:, None] * x + np.cos(theta)[:, None] * y
    phase = -4.0 * np.pi * projection / wavelength_m

    if class_name == "plate":
        # A flat plate has a stronger broadside response in this lightweight
        # model; this deliberately creates an interpretable aspect dependency.
        pattern = np.clip(np.abs(np.cos(theta)), 0.08, 1.0)[:, None]
    elif class_name == "cone":
        pattern = (0.75 + 0.25 * np.cos(theta - 0.4))[:, None]
    else:
        pattern = 1.0

    resp = np.sum(pattern * amp * np.exp(1j * phase), axis=1)
    return resp if np.ndim(aspect_deg) else resp[0]


def signature_to_tensor(response: np.ndarray) -> np.ndarray:
    """Convert a complex signature to `(2, L)` magnitude/phase channels."""

    x = np.asarray(response, dtype=np.complex128)
    mag_db = 20.0 * np.log10(np.abs(x) / (np.max(np.abs(x)) + 1e-12) + 1e-12)
    mag = np.clip((mag_db + 50.0) / 50.0, 0.0, 1.0)
    phase = np.unwrap(np.angle(x))
    phase = phase - np.mean(phase)
    phase = phase / (np.max(np.abs(phase)) + 1e-6)
    return np.stack([mag, phase], axis=0).astype(np.float32)


def extract_signature_features(response: np.ndarray, aspect_deg: np.ndarray | None = None) -> np.ndarray:
    """Handcrafted descriptor for P06 classical baselines."""

    x = np.asarray(response, dtype=np.complex128)
    mag = np.abs(x)
    mag_db = 20.0 * np.log10(mag / (np.max(mag) + 1e-12) + 1e-12)
    phase = np.unwrap(np.angle(x + 1e-12))
    dmag = np.diff(mag_db)
    dphase = np.diff(phase)
    spec = np.abs(np.fft.rfft(mag_db - np.mean(mag_db)))
    spec_total = float(np.sum(spec) + 1e-12)
    idx = np.arange(spec.size, dtype=np.float64)
    centroid = float(np.sum(idx * spec) / spec_total) / max(spec.size - 1, 1)
    bandwidth = float(np.sqrt(np.sum(((idx / max(spec.size - 1, 1) - centroid) ** 2) * spec) / spec_total))
    vals = [
        float(np.mean(mag_db)),
        float(np.std(mag_db)),
        float(np.min(mag_db)),
        float(np.percentile(mag_db, 10)),
        float(np.percentile(mag_db, 90)),
        float(np.mean(np.abs(dmag))) if dmag.size else 0.0,
        float(np.std(dmag)) if dmag.size else 0.0,
        float(np.mean(np.abs(dphase))) if dphase.size else 0.0,
        float(np.std(dphase)) if dphase.size else 0.0,
        centroid,
        bandwidth,
    ]
    if aspect_deg is not None:
        a = np.asarray(aspect_deg, dtype=np.float64)
        vals.extend([float(np.mean(a)), float(np.std(a)), float(np.ptp(a))])
    else:
        vals.extend([0.0, 0.0, 0.0])
    return np.asarray(vals, dtype=np.float32)


def generate_target_signature_sample(
    class_name: str,
    rng: np.random.Generator | None = None,
    config: TargetSignatureConfig | None = None,
    *,
    snr_db: float | None = None,
    center_aspect_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int | str]]:
    """Return `(tensor, label, features, meta)` for one P06 observation."""

    rng = _rng(rng)
    config = TargetSignatureConfig() if config is None else config
    if class_name not in TARGET_CLASSES:
        raise ValueError(f"unknown target class: {class_name!r}")

    scatterers = make_target_scatterers(class_name, rng, config.n_scatterers)
    label = TARGET_CLASSES.index(class_name)
    center = float(rng.uniform(*config.aspect_range_deg) if center_aspect_deg is None else center_aspect_deg)
    vib = float(rng.uniform(*config.vibration_range_deg))
    cycles = float(rng.uniform(0.7, 2.0))
    phase0 = float(rng.uniform(0.0, 2.0 * np.pi))
    t = np.linspace(0.0, 1.0, config.n_samples, endpoint=False)
    aspect = center + vib * np.sin(2.0 * np.pi * cycles * t + phase0)
    aspect += rng.normal(0.0, config.aspect_jitter_std_deg, size=config.n_samples)
    clean = complex_aspect_response(scatterers, aspect, config.wavelength_m, class_name=class_name)
    clean = clean / (np.sqrt(np.mean(np.abs(clean) ** 2)) + 1e-12)
    snr = float(rng.uniform(*config.snr_range_db) if snr_db is None else snr_db)
    observed = add_complex_awgn(clean, snr, rng, reference_power=1.0)
    tensor = signature_to_tensor(observed)
    features = extract_signature_features(observed, aspect)
    meta = {
        "class_name": class_name,
        "class_index": int(label),
        "snr_db": snr,
        "center_aspect_deg": center,
        "aspect_min_deg": float(np.min(aspect)),
        "aspect_max_deg": float(np.max(aspect)),
        "vibration_deg": vib,
        "n_scatterers": int(len(scatterers)),
        "fc_hz": float(config.fc_hz),
    }
    return tensor, np.asarray(label, dtype=np.int64), features, meta


def class_name_bytes() -> np.ndarray:
    return np.asarray([name.encode("utf-8") for name in TARGET_CLASSES], dtype="S32")
