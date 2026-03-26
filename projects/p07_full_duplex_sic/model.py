"""P07 Full-Duplex SIC — 1D U-Net 모델

입력: (B, 4, 512) — tx_ref (2ch) + rx_mix (2ch) concatenated
출력: (B, 2, 512) — SI estimate (real/imag)

Clean estimate: rx_mix - si_hat (post-processing, not in model)

파라미터 수: ~300K
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from common.train_utils import count_parameters


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock1d(nn.Module):
    """Conv1d → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock1d(nn.Module):
    """ConvTranspose1d upsample → Conv1d → BN → ReLU (skip connection 포함)."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 kernel_size: int = 4, stride: int = 2, padding: int = 1):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size,
                                     stride=stride, padding=padding)
        self.conv = nn.Sequential(
            nn.Conv1d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # 길이 불일치 보정 (ConvTranspose1d의 output_padding 이슈)
        if x.shape[-1] != skip.shape[-1]:
            x = x[..., :skip.shape[-1]]
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# U-Net 모델
# ---------------------------------------------------------------------------

class SICUNet(nn.Module):
    """1D U-Net for Self-Interference Cancellation.

    Encoder: 4 → 32 → 64 → 128 (stride-2 downsampling)
    Decoder: 128 → 64 → 32 → 2 (ConvTranspose1d + skip connections)

    Input shape : (B, 4, 512)  — [tx_ref_real, tx_ref_imag, rx_mix_real, rx_mix_imag]
    Output shape: (B, 2, 512)  — SI estimate [si_hat_real, si_hat_imag]
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock1d(4, 32, kernel_size=7, padding=3)          # (B,32,512)
        self.enc2 = ConvBlock1d(32, 64, kernel_size=3, stride=2, padding=1)  # (B,64,256)
        self.enc3 = ConvBlock1d(64, 128, kernel_size=3, stride=2, padding=1) # (B,128,128)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock1d(128, 256, kernel_size=3, padding=1),
            ConvBlock1d(256, 128, kernel_size=3, padding=1),
        )

        # Decoder (with skip connections)
        self.dec2 = UpBlock1d(in_ch=128, skip_ch=64, out_ch=64)   # (B,64,256)
        self.dec1 = UpBlock1d(in_ch=64, skip_ch=32, out_ch=32)    # (B,32,512)

        # Output head
        self.head = nn.Conv1d(32, 2, kernel_size=1)                # (B,2,512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 4, 512) — concatenated [tx_ref, rx_mix]

        Returns
        -------
        si_hat : (B, 2, 512) — SI estimate
        """
        # Encoder
        s1 = self.enc1(x)       # (B, 32, 512)
        s2 = self.enc2(s1)      # (B, 64, 256)
        s3 = self.enc3(s2)      # (B, 128, 128)

        # Bottleneck
        b = self.bottleneck(s3) # (B, 128, 128)

        # Decoder
        d2 = self.dec2(b, s2)   # (B, 64, 256)
        d1 = self.dec1(d2, s1)  # (B, 32, 512)

        return self.head(d1)    # (B, 2, 512)


def build_model() -> SICUNet:
    """모델 생성 및 파라미터 수 출력."""
    model = SICUNet()
    n_params = count_parameters(model)
    print(f"SICUNet: {n_params:,} parameters")
    return model


if __name__ == "__main__":
    model = build_model()
    x = torch.randn(2, 4, 512)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
    assert out.shape == (2, 2, 512), f"Unexpected output shape: {out.shape}"
    print("Shape check passed.")
