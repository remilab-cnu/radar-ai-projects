"""P06 lightweight 1-D CNN for target signature classification."""
from __future__ import annotations

import torch
import torch.nn as nn


class TinySignatureCNN(nn.Module):
    """Small magnitude/phase sequence classifier."""

    def __init__(self, n_classes: int = 3, in_channels: int = 2, base_ch: int = 16, dropout: float = 0.15):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, base_ch, 5, padding=2),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(base_ch, base_ch * 2, 5, padding=2),
            nn.BatchNorm1d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(base_ch * 2, base_ch * 4, 3, padding=1),
            nn.BatchNorm1d(base_ch * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_ch * 4, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def make_signature_model(n_classes: int = 3, in_channels: int = 2, base_ch: int = 16, dropout: float = 0.15) -> nn.Module:
    return TinySignatureCNN(n_classes=n_classes, in_channels=in_channels, base_ch=base_ch, dropout=dropout)
