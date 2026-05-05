"""P03 radar-cube DoA spectrum network.

Input is the complex antenna vector selected from a range-Doppler cube after
range FFT and Doppler FFT.  Angle FFT is not part of the neural input pipeline.
In the active mapping-first P03 lane, this per-detection DoA model is evaluated
through downstream point-cloud and probabilistic-map quality.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.train_utils import count_parameters


class ResidualConvBlock(nn.Module):
    """Lecture-scale 1D residual block over the antenna aperture."""

    def __init__(self, channels: int, dilation: int = 1, dropout: float = 0.05):
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class RadarCubeDoANet(nn.Module):
    """End-to-end antenna-vector to DoA-spectrum network.

    The model is intentionally deeper than a compact toy because P03 is lecture
    material.  It still consumes only the RD-selected antenna vector; it does
    not receive covariance matrices, angle-FFT features, ego pose, or map labels.
    Mapping evaluation happens after DoA estimation through shared projection
    and OGM utilities.
    """

    def __init__(self, n_rx: int = 8, grid_size: int = 181, width: int = 128, dropout: float = 0.20):
        super().__init__()
        self.n_rx = n_rx
        self.grid_size = grid_size
        self.stem = nn.Sequential(
            nn.Conv1d(2, width, kernel_size=1),
            nn.BatchNorm1d(width),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualConvBlock(width, dilation=1, dropout=0.05),
            ResidualConvBlock(width, dilation=2, dropout=0.05),
            ResidualConvBlock(width, dilation=1, dropout=0.05),
            ResidualConvBlock(width, dilation=2, dropout=0.05),
            ResidualConvBlock(width, dilation=1, dropout=0.05),
            ResidualConvBlock(width, dilation=2, dropout=0.05),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * n_rx, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, grid_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return spectrum logits with shape (B, grid_size)."""
        z = self.stem(x)
        z = self.blocks(z)
        return self.head(z)


def build_model(n_rx: int = 8, grid_size: int = 181, dropout: float = 0.20) -> RadarCubeDoANet:
    return RadarCubeDoANet(n_rx=n_rx, grid_size=grid_size, dropout=dropout)


if __name__ == "__main__":
    model = build_model()
    x = torch.randn(4, 2, 8)
    y = model(x)
    print(f"RadarCubeDoANet: input {x.shape} -> logits {y.shape}")
    print(f"Parameters: {count_parameters(model):,}")
