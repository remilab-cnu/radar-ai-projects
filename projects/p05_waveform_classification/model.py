"""P05 lightweight CNN models for radar waveform classification."""
from __future__ import annotations

import torch
import torch.nn as nn


class TinyWaveformCNN(nn.Module):
    """Small STFT-image classifier for CPU-friendly classroom runs."""

    def __init__(self, n_classes: int = 4, base_ch: int = 16, dropout: float = 0.15):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_ch, base_ch * 2, 3, padding=1),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, padding=1),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_ch * 4, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def make_waveform_model(n_classes: int = 4, base_ch: int = 16, dropout: float = 0.15) -> nn.Module:
    return TinyWaveformCNN(n_classes=n_classes, base_ch=base_ch, dropout=dropout)
