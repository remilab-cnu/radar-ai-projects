"""DnCNN-SAR model for SAR speckle removal via residual learning in log domain."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.train_utils import count_parameters


class DnCNNSAR(nn.Module):
    """DnCNN-SAR: 17-layer residual CNN for SAR despeckling in log domain.

    Concept:
        Multiplicative speckle in linear domain -> additive noise in log domain.
        The network predicts the noise residual; the clean image is recovered as:
            clean = input - predicted_residual

    Input:  (B, 1, H, W) -- log-intensity SAR patch
    Output: (B, 1, H, W) -- despeckled log-intensity image
    """

    def __init__(self, n_channels: int = 1, n_filters: int = 64, n_layers: int = 17) -> None:
        super().__init__()
        self.n_layers = n_layers

        layers: list[nn.Module] = []

        # Layer 1 -- Conv + ReLU (no BN)
        layers.append(nn.Conv2d(n_channels, n_filters, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))

        # Layers 2-(n_layers-1) -- Conv + BN + ReLU
        for _ in range(n_layers - 2):
            layers.append(nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(n_filters))
            layers.append(nn.ReLU(inplace=True))

        # Layer n_layers -- Conv only (no BN, no activation)
        layers.append(nn.Conv2d(n_filters, n_channels, kernel_size=3, padding=1))

        self.layers = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.layers(x)
        return x - residual


class DespecklingLoss(nn.Module):
    """Charbonnier + SSIM loss for SAR despeckling.

    Total = w_char * Charbonnier(pred, target) + w_ssim * (1 - SSIM(pred, target))
    """

    def __init__(self, w_char: float = 0.8, w_ssim: float = 0.2,
                 eps: float = 1e-3, window_size: int = 11) -> None:
        super().__init__()
        self.w_char = w_char
        self.w_ssim = w_ssim
        self.eps = eps
        self.window_size = window_size
        kernel = self._gaussian_kernel(window_size)
        self.register_buffer("kernel", kernel)

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        g1d = g1d / g1d.sum()
        g2d = g1d.unsqueeze(0) * g1d.unsqueeze(1)
        return g2d.unsqueeze(0).unsqueeze(0)

    def charbonnier(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))

    def ssim_loss(self, pred, target):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        pad = self.window_size // 2
        kernel = self.kernel
        channels = pred.shape[1]
        if channels > 1:
            kernel = kernel.repeat(channels, 1, 1, 1)

        def _conv(t):
            return F.conv2d(t, kernel.to(t.dtype), padding=pad, groups=channels)

        mu_p = _conv(pred)
        mu_t = _conv(target)
        sigma_p2 = torch.clamp(_conv(pred * pred) - mu_p * mu_p, min=0.0)
        sigma_t2 = torch.clamp(_conv(target * target) - mu_t * mu_t, min=0.0)
        sigma_pt = _conv(pred * target) - mu_p * mu_t
        num = (2.0 * mu_p * mu_t + C1) * (2.0 * sigma_pt + C2)
        den = (mu_p * mu_p + mu_t * mu_t + C1) * (sigma_p2 + sigma_t2 + C2)
        return 1.0 - (num / den).mean()

    def forward(self, pred, target):
        return self.w_char * self.charbonnier(pred, target) + self.w_ssim * self.ssim_loss(pred, target)


# ---------------------------------------------------------------------------
# Classical baselines (used in evaluation)
# ---------------------------------------------------------------------------

def lee_filter(img: np.ndarray, window_size: int = 7) -> np.ndarray:
    """Lee adaptive speckle filter."""
    from scipy.ndimage import uniform_filter
    img = img.astype(np.float64)
    mean_local = uniform_filter(img, size=window_size)
    mean_sq_local = uniform_filter(img ** 2, size=window_size)
    var_local = mean_sq_local - mean_local ** 2
    mean_global = img.mean()
    var_noise = (mean_global ** 2) / 1.0
    K = var_local / (var_local + var_noise + 1e-10)
    return (mean_local + K * (img - mean_local)).astype(np.float32)


def frost_filter(img: np.ndarray, window_size: int = 7, damping: float = 2.0) -> np.ndarray:
    """Frost adaptive speckle filter."""
    from scipy.ndimage import uniform_filter
    img = img.astype(np.float64)
    half = window_size // 2
    mean_local = uniform_filter(img, size=window_size)
    mean_sq = uniform_filter(img ** 2, size=window_size)
    var_local = np.maximum(mean_sq - mean_local ** 2, 0.0)
    cov = np.sqrt(var_local) / (np.abs(mean_local) + 1e-10)
    padded = np.pad(img, half, mode='reflect')
    H, W = img.shape
    weighted_sum = np.zeros_like(img)
    weight_sum = np.zeros_like(img)
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            dist = np.sqrt(dy ** 2 + dx ** 2)
            w = np.exp(-damping * cov * dist)
            neighbour = padded[half + dy:half + dy + H, half + dx:half + dx + W]
            weighted_sum += w * neighbour
            weight_sum += w
    return (weighted_sum / (weight_sum + 1e-10)).astype(np.float32)


def median_filter(img: np.ndarray, window_size: int = 7) -> np.ndarray:
    """Median filter for speckle suppression."""
    from scipy.ndimage import median_filter as _scipy_median
    return _scipy_median(img.astype(np.float32), size=window_size)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-12:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def compute_ssim(pred: np.ndarray, target: np.ndarray, window_size: int = 11) -> float:
    from scipy.ndimage import uniform_filter
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    mu_p = uniform_filter(pred, size=window_size)
    mu_t = uniform_filter(target, size=window_size)
    sigma_p2 = uniform_filter(pred ** 2, size=window_size) - mu_p ** 2
    sigma_t2 = uniform_filter(target ** 2, size=window_size) - mu_t ** 2
    sigma_pt = uniform_filter(pred * target, size=window_size) - mu_p * mu_t
    num = (2.0 * mu_p * mu_t + C1) * (2.0 * sigma_pt + C2)
    den = (mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p2 + sigma_t2 + C2)
    return float((num / (den + 1e-10)).mean())


def compute_enl(img: np.ndarray, roi=None) -> float:
    region = img[roi[0]:roi[1], roi[2]:roi[3]] if roi else img
    region = region.astype(np.float64)
    std = region.std()
    if std < 1e-10:
        return float("inf")
    return float((region.mean() / std) ** 2)


if __name__ == "__main__":
    model = DnCNNSAR(n_channels=1, n_filters=64, n_layers=17)
    x = torch.randn(4, 1, 64, 64)
    with torch.no_grad():
        y = model(x)
    print(f"DnCNN-SAR parameters: {count_parameters(model):,}")
    print(f"input shape: {tuple(x.shape)}, output shape: {tuple(y.shape)}")
