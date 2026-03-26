"""P05 Neural CFAR — 모델 정의

Small 2D CNN for binary target detection from 15x15 RD-map patches.

Architecture:
  Conv(2,16,3)-BN-ReLU →
  Conv(16,32,3)-BN-ReLU-MaxPool(2) →
  Conv(32,64,3)-BN-ReLU-MaxPool(2) →
  GlobalAvgPool →
  FC(64) → ReLU → FC(1)

Input:  (B, 2, 15, 15)
Output: (B, 1)  — raw logit (apply sigmoid for probability)
~72K parameters
"""
from __future__ import annotations

import torch
import torch.nn as nn


class NeuralCFAR(nn.Module):
    """2D CNN binary classifier for CFAR patch detection."""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: (B, 2, 15, 15) → (B, 16, 13, 13)
            nn.Conv2d(2, 16, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # Block 2: (B, 16, 13, 13) → (B, 32, 5, 5)
            nn.Conv2d(16, 32, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 11→5

            # Block 3: (B, 32, 5, 5) → (B, 64, 1, 1)
            nn.Conv2d(32, 64, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 3→1
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 2, 15, 15)

        Returns
        -------
        logits : (B, 1)
        """
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


def build_model() -> NeuralCFAR:
    return NeuralCFAR()


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from common.train_utils import count_parameters

    model = build_model()
    n = count_parameters(model)
    print(f"NeuralCFAR parameters: {n:,}")

    x = torch.randn(4, 2, 15, 15)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
