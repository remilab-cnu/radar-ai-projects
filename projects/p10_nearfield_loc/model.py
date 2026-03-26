"""P10 Near-Field Source Localization — Multi-head CNN 모델

구조:
  Trunk: Conv2d(2,16,3)-BN-ReLU → Conv2d(16,32,3)-BN-ReLU
         → AdaptiveAvgPool2d(1,1) → Flatten → (32,)
  Head 1 (near/far):  FC(32,1) → sigmoid
  Head 2 (angle):     FC(32,16)-ReLU → FC(16,2)  [sin(θ), cos(θ)]
  Head 3 (range):     FC(32,16)-ReLU → FC(16,1)  [near-field only]

약 ~25K 파라미터.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

from common.train_utils import count_parameters


class NearFieldLocNet(nn.Module):
    """Multi-head CNN for near-field source localization.

    Parameters
    ----------
    n_rx : int
        Number of antenna elements (default: 8)
    n_snapshots : int
        Number of time snapshots (default: 64)
    trunk_channels : int
        Feature dimension at trunk output (default: 32)
    """

    def __init__(self, n_rx: int = 8, n_snapshots: int = 64, trunk_channels: int = 32):
        super().__init__()

        self.trunk = nn.Sequential(
            # (B, 2, 8, 64) → (B, 16, 6, 62)
            nn.Conv2d(2, 16, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # (B, 16, 6, 62) → (B, 32, 4, 60)
            nn.Conv2d(16, trunk_channels, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(trunk_channels),
            nn.ReLU(inplace=True),
            # Global average pool → (B, 32, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),   # (B, 32)
        )

        # Head 1: near/far binary classification
        self.head_near = nn.Linear(trunk_channels, 1)

        # Head 2: angle estimation via sin/cos representation
        self.head_angle = nn.Sequential(
            nn.Linear(trunk_channels, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2),   # [sin(θ), cos(θ)]
        )

        # Head 3: range estimation (near-field only)
        self.head_range = nn.Sequential(
            nn.Linear(trunk_channels, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, 2, N_RX, N_SNAPSHOTS) float

        Returns
        -------
        near_logit : (B, 1) — raw logit (apply sigmoid for probability)
        angle_sincos : (B, 2) — [sin(θ), cos(θ)]
        range_out : (B, 1) — range in [m], only meaningful for near-field
        """
        feat = self.trunk(x)                      # (B, 32)
        near_logit = self.head_near(feat)          # (B, 1)
        angle_sincos = self.head_angle(feat)       # (B, 2)
        range_out = self.head_range(feat)          # (B, 1)
        return near_logit, angle_sincos, range_out


def build_model() -> NearFieldLocNet:
    """기본 설정 모델 생성."""
    return NearFieldLocNet(n_rx=8, n_snapshots=64, trunk_channels=32)


if __name__ == "__main__":
    model = build_model()
    n_params = count_parameters(model)
    print(f"NearFieldLocNet parameters: {n_params:,}")

    x = torch.randn(4, 2, 8, 64)
    near_logit, angle_sc, range_out = model(x)
    print(f"Input:        {x.shape}")
    print(f"near_logit:   {near_logit.shape}")
    print(f"angle_sincos: {angle_sc.shape}")
    print(f"range_out:    {range_out.shape}")

    assert near_logit.shape == (4, 1)
    assert angle_sc.shape == (4, 2)
    assert range_out.shape == (4, 1)
    print("Shape check passed.")
