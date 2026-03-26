"""P08 Jammer Null Steering — CovNet 모델

입력: (B, 2, 8, 8) 공분산 행렬 (real/imag) + look_angle (B, 1)
출력: (B, 1) — sin(jammer_angle) 예측 (후처리: arcsin으로 각도 복원)

파라미터 수: ~55K
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from common.train_utils import count_parameters


class CovNet(nn.Module):
    """Covariance matrix CNN + look-angle 융합 네트워크.

    Architecture:
        Conv2d(2→16, 3×3, pad=1)-BN-ReLU
        Conv2d(16→32, 3×3, pad=1)-BN-ReLU
        Conv2d(32→64, 3×3, pad=1)-BN-ReLU
        Global Average Pooling → (B, 64)
        Concat with look_angle (B, 1) → (B, 65)
        FC(65→256)-ReLU → FC(256→128)-ReLU → FC(128→1) → sin(jammer_angle)

    Notes
    -----
    - 8×8 입력에 padding=1인 3×3 conv 3번 → spatial 8×8 유지 → GAP → scalar
    - 출력: sin(θ_jammer) ∈ [-1, 1], 각도 복원: arcsin(out) * 180/π
    """

    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            # (B, 2, 8, 8) → (B, 16, 8, 8)
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # (B, 16, 8, 8) → (B, 32, 8, 8)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # (B, 32, 8, 8) → (B, 64, 8, 8)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Global Average Pooling: (B, 64, 8, 8) → (B, 64)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Fusion + regression head
        self.head = nn.Sequential(
            nn.Linear(64 + 1, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, cov: torch.Tensor, look_angle: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        cov        : (B, 2, 8, 8)  — real/imag covariance matrix
        look_angle : (B, 1)        — look angle in degrees (normalized)

        Returns
        -------
        sin_jammer : (B, 1)  — sin(jammer_angle)
        """
        feat = self.cnn(cov)            # (B, 64, 2, 2)
        feat = self.gap(feat)           # (B, 64, 1, 1)
        feat = feat.flatten(1)          # (B, 64)

        # look_angle 정규화: [-90, 90] → [-1, 1]
        look_norm = look_angle / 90.0   # (B, 1)

        x = torch.cat([feat, look_norm], dim=1)  # (B, 65)
        return self.head(x)             # (B, 1): sin(jammer_angle)


def build_model() -> CovNet:
    """모델 생성 및 파라미터 수 출력."""
    model = CovNet()
    n_params = count_parameters(model)
    print(f"CovNet: {n_params:,} parameters")
    return model


if __name__ == "__main__":
    model = build_model()
    cov = torch.randn(4, 2, 8, 8)
    look = torch.randn(4, 1)
    out = model(cov, look)
    print(f"cov: {cov.shape}, look: {look.shape} → output: {out.shape}")
    assert out.shape == (4, 1), f"Unexpected output shape: {out.shape}"
    print("Shape check passed.")
