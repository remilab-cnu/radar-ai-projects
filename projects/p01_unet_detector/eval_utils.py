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


def assert_schema_v2(h5: h5py.File) -> None:
    required = [
        "x", "y", "rdm_mag_linear", "snr_db", "n_targets", "clutter_power_db",
        "target_range_bin", "target_doppler_bin", "range_axis_m", "velocity_axis_mps",
    ]
    missing = [key for key in required if key not in h5]
    if missing:
        raise KeyError(
            "P01 schema-v2 data required for verified baselines; missing " + ", ".join(missing)
        )


def iter_samples(path: str | Path, max_samples: int | None = None):
    with h5py.File(path, "r") as f:
        assert_schema_v2(f)
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
            }
