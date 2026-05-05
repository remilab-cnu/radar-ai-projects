"""Evaluation helpers for P01 verified detector experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


SPLIT_FILES = {
    "train": "det_train.h5",
    "val": "det_val.h5",
    "test": "det_test.h5",
}
EXPECTED_SCHEMA_VERSION = 9


def split_path(data_dir: str | Path, split: str) -> Path:
    if split not in SPLIT_FILES:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLIT_FILES)}")
    return Path(data_dir) / SPLIT_FILES[split]


def confusion_counts(pred: np.ndarray, gt: np.ndarray) -> dict[str, int]:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    return {
        "tp": int(np.sum(pred & gt)),
        "fp": int(np.sum(pred & ~gt)),
        "fn": int(np.sum(~pred & gt)),
        "tn": int(np.sum(~pred & ~gt)),
    }


def add_counts(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    for key in ("tp", "fp", "fn", "tn"):
        a[key] = int(a.get(key, 0) + b.get(key, 0))
    return a


def target_detection_counts(
    pred: np.ndarray,
    target_range_bins: np.ndarray,
    target_doppler_bins: np.ndarray,
    tolerance: tuple[int, int] = (1, 1),
) -> dict[str, int]:
    """Count labelled targets detected within a small RD-bin tolerance.

    Pixel F1 can punish harmless mask-shape differences around a Hann-windowed
    mainlobe.  For P1 lectures we also report target-level recall: each labelled
    target is counted once if any predicted positive falls close to its simulator
    bin.
    """
    pred = np.asarray(pred, dtype=bool)
    td, tr = (int(tolerance[0]), int(tolerance[1]))
    detected = 0
    total = 0
    n_doppler, n_range = pred.shape
    for r_bin, d_bin in zip(np.asarray(target_range_bins), np.asarray(target_doppler_bins)):
        r = int(r_bin)
        d = int(d_bin)
        if r < 0 or d < 0:
            continue
        if not (0 <= r < n_range and 0 <= d < n_doppler):
            continue
        total += 1
        d0 = max(0, d - td)
        d1 = min(n_doppler, d + td + 1)
        r0 = max(0, r - tr)
        r1 = min(n_range, r + tr + 1)
        detected += int(np.any(pred[d0:d1, r0:r1]))
    return {"target_detected": int(detected), "target_total": int(total)}


def add_target_counts(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    for key in ("target_detected", "target_total"):
        a[key] = int(a.get(key, 0) + b.get(key, 0))
    return a


def metrics_from_counts(c: dict[str, int]) -> dict[str, float | int]:
    tp, fp, fn, tn = (int(c.get(k, 0)) for k in ("tp", "fp", "fn", "tn"))
    pd = tp / (tp + fn + 1e-10)
    pfa = fp / (fp + tn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    f1 = 2 * precision * pd / (precision + pd + 1e-10)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "Pd": float(pd),
        "Pfa": float(pfa),
        "Precision": float(precision),
        "F1": float(f1),
    }


def add_target_metrics(metrics: dict, target_counts: dict[str, int]) -> dict:
    detected = int(target_counts.get("target_detected", 0))
    total = int(target_counts.get("target_total", 0))
    metrics.update({
        "target_detected": detected,
        "target_total": total,
        "target_recall": float(detected / (total + 1e-10)),
    })
    return metrics


def choose_by_max_f1(results: list[dict]) -> dict:
    if not results:
        raise ValueError("cannot choose policy from empty results")
    return max(results, key=lambda row: (row.get("F1", -1.0), row.get("Pd", -1.0), -row.get("Pfa", 1.0)))


def choose_nearest_pfa(results: list[dict], target_pfa: float) -> dict:
    if not results:
        raise ValueError("cannot choose policy from empty results")
    return min(results, key=lambda row: (abs(row.get("Pfa", 0.0) - target_pfa), -row.get("F1", 0.0)))


def load_policy(path: str | Path) -> dict:
    import json
    data = json.loads(Path(path).read_text())
    if "selected_policy" in data:
        return data["selected_policy"]
    if "selected" in data:
        return data["selected"]
    if isinstance(data.get("results"), list):
        return choose_by_max_f1(data["results"])
    raise ValueError(f"no selected policy in {path}")


def assert_schema_current(h5: h5py.File) -> None:
    required = [
        "x", "y", "rdm_mag_linear", "snr_db", "n_targets", "clutter_power_db",
        "adc_clipped_fraction", "mti_applied", "mti_mode",
        "target_peak_snr_db", "target_local_bg_floor", "target_effective_bg_floor",
        "radar_fs_hz", "fs_over_bandwidth",
        "target_range_bin", "target_doppler_bin", "range_axis_m", "velocity_axis_mps",
    ]
    missing = [key for key in required if key not in h5]
    if missing:
        raise KeyError(
            f"P01 schema-v{EXPECTED_SCHEMA_VERSION} data required for verified baselines; missing "
            + ", ".join(missing)
        )
    version = int(h5["schema_version"][0]) if "schema_version" in h5 else -1
    if version != EXPECTED_SCHEMA_VERSION:
        raise ValueError(
            f"P01 data schema_version={version}; expected {EXPECTED_SCHEMA_VERSION}. "
            "Regenerate data with projects/p01_unet_detector/generate_data.py."
        )


# Compatibility name used by older evaluation scripts in this repo.
assert_schema_v2 = assert_schema_current


def iter_samples(path: str | Path, max_samples: int | None = None):
    with h5py.File(path, "r") as f:
        assert_schema_current(f)
        n = len(f["x"])
        if max_samples is not None:
            n = min(n, int(max_samples))
        for i in range(n):
            yield {
                "idx": i,
                "x": f["x"][i],
                "gt": f["y"][i, 0] > 0.5,
                "rdm_mag_linear": f["rdm_mag_linear"][i],
                "snr_db": float(f["snr_db"][i]),
                "n_targets": int(f["n_targets"][i]),
                "clutter_power_db": float(f["clutter_power_db"][i]),
                "target_range_bin": f["target_range_bin"][i],
                "target_doppler_bin": f["target_doppler_bin"][i],
            }
