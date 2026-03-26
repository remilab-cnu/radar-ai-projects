"""P06 I/Q Imbalance Correction — 모델 정의

1D CNN regressor: corrupted I/Q signal → imbalance parameters [gain_db, phase_deg, dc_i, dc_q]

Architecture:
  Conv1d(2, 64, 7)-BN-ReLU →
  Conv1d(64, 128, 5, stride=2)-BN-ReLU →
  Conv1d(128, 128, 5, stride=2)-BN-ReLU →
  AdaptiveAvgPool1d(1) →
  FC(128, 64)-ReLU → FC(64, 4)

Input:  (B, 2, 512)
Output: (B, 4)  — [gain_db, phase_deg, dc_i, dc_q]
~133K parameters
"""
from __future__ import annotations

import torch
import torch.nn as nn


class IQImbalanceCNN(nn.Module):
    """1D CNN for I/Q imbalance parameter estimation."""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: (B, 2, 512) → (B, 64, 506)
            nn.Conv1d(2, 64, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            # Block 2: (B, 64, 506) → (B, 128, 251)
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            # Block 3: (B, 128, 251) → (B, 128, 124)
            nn.Conv1d(128, 128, kernel_size=5, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # Adaptive pool → (B, 128, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 2, 512)  — corrupted I/Q signal

        Returns
        -------
        params : (B, 4)  — [gain_db, phase_deg, dc_i, dc_q]
        """
        x = self.features(x)
        x = self.pool(x)
        x = self.regressor(x)
        return x


def build_model() -> IQImbalanceCNN:
    return IQImbalanceCNN()


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from common.train_utils import count_parameters

    model = build_model()
    n = count_parameters(model)
    print(f"IQImbalanceCNN parameters: {n:,}")

    x = torch.randn(4, 2, 512)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
