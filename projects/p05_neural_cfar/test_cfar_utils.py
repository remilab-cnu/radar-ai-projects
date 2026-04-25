"""Verification tests for the P05 linear-power CA-CFAR baseline."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cfar_utils import (  # noqa: E402
    PATCH_SIZE,
    ca_cfar_alpha,
    ca_cfar_detect,
    cfar_training_mask,
    evaluate_patch_ca_cfar,
)


class PatchCaCfarTests(unittest.TestCase):
    def test_training_mask_has_expected_guard_and_training_bins(self):
        mask = cfar_training_mask()
        center = PATCH_SIZE // 2
        self.assertFalse(mask[center, center], "CUT must not be a training cell")
        self.assertFalse(mask[center - 1:center + 2, center - 1:center + 2].any())
        self.assertEqual(int(mask.sum()), 72)

    def test_detector_uses_cut_bin_not_patch_maximum(self):
        patch = np.ones((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        patch[2, 2] = 10_000.0  # bright off-CUT reflector must not be declared as CUT detection
        detected, threshold, cut_power = ca_cfar_detect(patch, pfa=1e-2)
        self.assertFalse(detected)
        self.assertEqual(cut_power, 1.0)
        self.assertGreater(threshold, 1.0)

        patch[PATCH_SIZE // 2, PATCH_SIZE // 2] = threshold * 1.1
        detected, _, _ = ca_cfar_detect(patch, pfa=1e-2)
        self.assertTrue(detected)

    def test_ca_cfar_empirical_pfa_matches_exponential_noise_model(self):
        rng = np.random.default_rng(123)
        n = 40_000
        pfa = 1e-2
        patches = rng.exponential(scale=1.0, size=(n, PATCH_SIZE, PATCH_SIZE)).astype(np.float32)
        y = np.zeros(n, dtype=np.float32)

        metrics = evaluate_patch_ca_cfar(patches, y, pfa=pfa)
        self.assertGreater(metrics["pfa"], 0.007)
        self.assertLess(metrics["pfa"], 0.013)
        self.assertAlmostEqual(metrics["target_pfa"], pfa)
        self.assertAlmostEqual(
            ca_cfar_alpha(metrics["n_train_cells"], pfa),
            metrics["mean_threshold"],
            delta=0.08,  # mean training power is approximately 1.0 for this synthetic set
        )


if __name__ == "__main__":
    unittest.main()
