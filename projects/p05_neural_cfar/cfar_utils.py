"""Linear-power patch CA-CFAR utilities for P05.

The neural model still receives normalized 15x15 image-like patches, but the
classical baseline in this module operates only on linear RDM power cells.  This
keeps the CFAR false-alarm claim tied to the detector's native statistical
model instead of to normalized log-magnitude display values.
"""
from __future__ import annotations

import numpy as np


PATCH_SIZE = 15
DEFAULT_GUARD_CELLS = 1
DEFAULT_TRAINING_CELLS = 3


def cfar_training_mask(
    patch_shape: tuple[int, int] = (PATCH_SIZE, PATCH_SIZE),
    guard_cells: int = DEFAULT_GUARD_CELLS,
    training_cells: int = DEFAULT_TRAINING_CELLS,
) -> np.ndarray:
    """Return a boolean mask selecting CA-CFAR training cells around the CUT.

    The CUT is the exact center of an odd-sized patch.  The selected training
    ring is the square window with half-width ``guard_cells + training_cells``
    minus the protected guard square with half-width ``guard_cells``.
    """
    rows, cols = patch_shape
    if rows % 2 == 0 or cols % 2 == 0:
        raise ValueError("CA-CFAR patch dimensions must be odd so the CUT is unambiguous")
    if guard_cells < 0 or training_cells <= 0:
        raise ValueError("guard_cells must be non-negative and training_cells must be positive")

    center_r, center_c = rows // 2, cols // 2
    outer = guard_cells + training_cells
    if center_r - outer < 0 or center_c - outer < 0 or center_r + outer >= rows or center_c + outer >= cols:
        raise ValueError("patch is too small for the requested guard/training cells")

    mask = np.zeros(patch_shape, dtype=bool)
    mask[center_r - outer:center_r + outer + 1, center_c - outer:center_c + outer + 1] = True
    mask[center_r - guard_cells:center_r + guard_cells + 1, center_c - guard_cells:center_c + guard_cells + 1] = False
    return mask


def ca_cfar_alpha(n_train_cells: int, pfa: float) -> float:
    """Scale factor for cell-averaging CFAR with exponential noise power."""
    if n_train_cells <= 0:
        raise ValueError("n_train_cells must be positive")
    if not 0.0 < pfa < 1.0:
        raise ValueError("pfa must be between 0 and 1")
    return float(n_train_cells * (pfa ** (-1.0 / n_train_cells) - 1.0))


def ca_cfar_threshold(
    patch_power: np.ndarray,
    pfa: float = 1e-2,
    guard_cells: int = DEFAULT_GUARD_CELLS,
    training_cells: int = DEFAULT_TRAINING_CELLS,
) -> float:
    """Compute the linear-power CA-CFAR threshold for one patch."""
    patch_power = np.asarray(patch_power, dtype=np.float64)
    mask = cfar_training_mask(patch_power.shape, guard_cells, training_cells)
    noise_estimate = float(np.mean(patch_power[mask]))
    return ca_cfar_alpha(int(mask.sum()), pfa) * noise_estimate


def ca_cfar_detect(
    patch_power: np.ndarray,
    pfa: float = 1e-2,
    guard_cells: int = DEFAULT_GUARD_CELLS,
    training_cells: int = DEFAULT_TRAINING_CELLS,
) -> tuple[bool, float, float]:
    """Detect the center CUT of one linear-power patch.

    Returns ``(detected, threshold, cut_power)``.
    """
    patch_power = np.asarray(patch_power, dtype=np.float64)
    cut_power = float(patch_power[patch_power.shape[0] // 2, patch_power.shape[1] // 2])
    threshold = ca_cfar_threshold(patch_power, pfa, guard_cells, training_cells)
    return bool(cut_power > threshold), threshold, cut_power


def evaluate_patch_ca_cfar(
    patch_power: np.ndarray,
    y_true: np.ndarray,
    pfa: float = 1e-2,
    guard_cells: int = DEFAULT_GUARD_CELLS,
    training_cells: int = DEFAULT_TRAINING_CELLS,
) -> dict[str, float]:
    """Evaluate patch CA-CFAR on a batch of linear-power patches."""
    patch_power = np.asarray(patch_power)
    y_true = np.asarray(y_true).astype(int)
    if patch_power.ndim != 3:
        raise ValueError("patch_power must have shape (N, H, W)")
    if len(patch_power) != len(y_true):
        raise ValueError("patch_power and y_true length mismatch")

    detections = np.zeros(len(patch_power), dtype=int)
    thresholds = np.zeros(len(patch_power), dtype=np.float64)
    cut_powers = np.zeros(len(patch_power), dtype=np.float64)
    for idx, patch in enumerate(patch_power):
        detected, threshold, cut_power = ca_cfar_detect(patch, pfa, guard_cells, training_cells)
        detections[idx] = int(detected)
        thresholds[idx] = threshold
        cut_powers[idx] = cut_power

    tp = int(np.sum((detections == 1) & (y_true == 1)))
    fp = int(np.sum((detections == 1) & (y_true == 0)))
    fn = int(np.sum((detections == 0) & (y_true == 1)))
    tn = int(np.sum((detections == 0) & (y_true == 0)))

    pd = tp / (tp + fn + 1e-12)
    pfa_actual = fp / (fp + tn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    return {
        "pd": float(pd),
        "pfa": float(pfa_actual),
        "balanced_accuracy": float(0.5 * (pd + specificity)),
        "mean_threshold": float(np.mean(thresholds)) if len(thresholds) else 0.0,
        "mean_cut_power": float(np.mean(cut_powers)) if len(cut_powers) else 0.0,
        "n_train_cells": int(cfar_training_mask(patch_power.shape[1:], guard_cells, training_cells).sum()),
        "target_pfa": float(pfa),
    }
