"""Regression tests for P08 LCMV suppression metrics."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import lcmv_null_depth  # noqa: E402


class LcmvMetricTests(unittest.TestCase):
    def test_true_jammer_evaluation_penalizes_wrong_null_angle(self):
        """A wrong predicted null must not look like a perfect suppression."""
        R = np.eye(8, dtype=np.complex64)
        look_angle = 0.0
        true_jammer = 20.0

        correct = lcmv_null_depth(
            R, look_angle, true_jammer, eval_angle_deg=true_jammer
        )
        wrong = lcmv_null_depth(
            R, look_angle, -50.0, eval_angle_deg=true_jammer
        )

        self.assertLess(correct, -120.0)
        self.assertGreater(wrong, -80.0)

    def test_constraint_residual_mode_is_not_the_evaluation_metric(self):
        """Without eval_angle, the function only checks the imposed constraint."""
        R = np.eye(8, dtype=np.complex64)
        residual = lcmv_null_depth(R, 0.0, -50.0)
        true_response = lcmv_null_depth(R, 0.0, -50.0, eval_angle_deg=20.0)

        self.assertLess(residual, -120.0)
        self.assertGreater(true_response, -80.0)


if __name__ == "__main__":
    unittest.main()
