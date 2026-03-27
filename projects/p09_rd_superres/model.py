"""P09 RD Super-Resolution — SRResNet-lite 모델

구조:
  Conv(1,32,3)-ReLU
  → 4x Residual Block (Conv-BN-ReLU-Conv-BN + skip) at 32ch
  → Conv(32,4,3)
  → PixelShuffle(2)   [32x32 → 64x64]
  → Conv(1,1,3)

약 ~121K 파라미터.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

from common.train_utils import count_parameters


class ResidualBlock(nn.Module):
    """Conv-BN-ReLU-Conv-BN with skip connection."""

    def __init__(self, channels: int = 32):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class SRResNetLite(nn.Module):
    """SRResNet-lite: 32x32 LR RD map → 64x64 HR RD map.

    Parameters
    ----------
    n_res_blocks : int
        잔차 블록 수 (기본 4)
    channels : int
        내부 채널 수 (기본 32)
    upscale : int
        업스케일 배율 (기본 2 → PixelShuffle(2))
    """

    def __init__(self, n_res_blocks: int = 4, channels: int = 32, upscale: int = 2):
        super().__init__()
        self.upscale = upscale

        # Initial feature extraction
        self.head = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(n_res_blocks)]
        )

        # Post-residual conv (no activation before PixelShuffle)
        self.post_res = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

        # Upsample: channels → upscale^2 channels, then PixelShuffle
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * upscale * upscale, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale),      # (ch * 4, H, W) → (ch, 2H, 2W)
        )

        # Output conv
        self.tail = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, 32, 32) float — normalized dB LR map

        Returns
        -------
        (B, 1, 64, 64) float — normalized dB HR map
        """
        feat = self.head(x)                    # (B, 32, 32, 32)
        res = self.res_blocks(feat)             # (B, 32, 32, 32)
        res = self.post_res(res) + feat         # skip over all res blocks
        up = self.upsample(res)                 # (B, 32, 64, 64)
        out = self.tail(up)                     # (B, 1, 64, 64)
        return out


def build_model() -> SRResNetLite:
    """기본 설정 모델 생성."""
    return SRResNetLite(n_res_blocks=4, channels=32, upscale=2)


if __name__ == "__main__":
    model = build_model()
    n_params = count_parameters(model)
    print(f"SRResNetLite parameters: {n_params:,}")

    x = torch.randn(2, 1, 32, 32)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    assert y.shape == (2, 1, 64, 64), f"Unexpected output shape: {y.shape}"
    print("Shape check passed.")
