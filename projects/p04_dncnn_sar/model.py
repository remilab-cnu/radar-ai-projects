"""DnCNN-SAR model for speckle removal in normalized Sentinel-1 log/dB magnitude."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class DnCNNSAR(nn.Module):
    """DnCNN-SAR: 17-layer residual CNN for SAR despeckling in log/dB magnitude.

    Concept:
        Multiplicative speckle in linear domain → approximately additive noise
        in the log/dB-magnitude domain.
        The network predicts the noise residual; the clean image is recovered as:
            clean = input - predicted_residual

    Input:
        (B, 1, H, W) — normalized log/dB SAR magnitude patch
    Output:
        (B, 1, H, W) — despeckled normalized log/dB image
    """

    def __init__(self, n_channels: int = 1, n_filters: int = 64, n_layers: int = 17) -> None:
        """Initialise DnCNN-SAR.

        Args:
            n_channels: Input/output channel count (1 for single-pol SAR).
            n_filters:  Number of feature maps in intermediate layers.
            n_layers:   Total depth (17 matches the original DnCNN paper).
        """
        super().__init__()
        self.n_layers = n_layers

        layers: list[nn.Module] = []

        # Layer 1 — Conv + ReLU (no BN)
        layers.append(nn.Conv2d(n_channels, n_filters, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))

        # Layers 2–(n_layers-1) — Conv + BN + ReLU
        for _ in range(n_layers - 2):
            layers.append(nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(n_filters))
            layers.append(nn.ReLU(inplace=True))

        # Layer n_layers — Conv only (no BN, no activation)
        layers.append(nn.Conv2d(n_filters, n_channels, kernel_size=3, padding=1))

        self.layers = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming init for Conv layers; constant init for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return despeckled image (residual subtracted from input).

        Args:
            x: Log-intensity SAR image, shape (B, 1, H, W).

        Returns:
            Despeckled image of the same shape.
        """
        residual = self.layers(x)
        return x - residual


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


class DespecklingLoss(nn.Module):
    """Weighted combination of Charbonnier loss and SSIM loss.

    Total loss = w_char * Charbonnier(pred, target) + w_ssim * (1 - SSIM(pred, target))

    Args:
        w_char:      Weight for Charbonnier term.
        w_ssim:      Weight for structural-similarity term.
        eps:         Charbonnier smoothing constant.
        window_size: Gaussian window size for SSIM computation.
    """

    def __init__(
        self,
        w_char: float = 0.8,
        w_ssim: float = 0.2,
        eps: float = 1e-3,
        window_size: int = 11,
    ) -> None:
        super().__init__()
        self.w_char = w_char
        self.w_ssim = w_ssim
        self.eps = eps
        self.window_size = window_size

        # Pre-build Gaussian kernel (fixed, not learned)
        kernel = self._gaussian_kernel(window_size)
        # Shape: (1, 1, window_size, window_size)
        self.register_buffer("kernel", kernel)

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float = 1.5) -> torch.Tensor:
        """Create a 2-D Gaussian kernel normalised to sum 1."""
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        g1d = g1d / g1d.sum()
        g2d = g1d.unsqueeze(0) * g1d.unsqueeze(1)  # outer product
        return g2d.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    def charbonnier(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Charbonnier (pseudo-Huber) loss: mean(sqrt((pred-target)^2 + eps^2))."""
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))

    def ssim_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return 1 - SSIM so that minimising the loss maximises structural similarity.

        Implements the standard Wang et al. 2004 SSIM using a Gaussian window.
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        pad = self.window_size // 2
        kernel = self.kernel  # (1, 1, W, W)

        channels = pred.shape[1]
        if channels > 1:
            # Repeat kernel for multi-channel inputs (safety; normally channels=1)
            kernel = kernel.repeat(channels, 1, 1, 1)

        def _conv(t: torch.Tensor) -> torch.Tensor:
            return F.conv2d(t, kernel.to(t.dtype), padding=pad, groups=channels)

        mu_p = _conv(pred)
        mu_t = _conv(target)
        mu_p2 = mu_p * mu_p
        mu_t2 = mu_t * mu_t
        mu_pt = mu_p * mu_t

        sigma_p2 = torch.clamp(_conv(pred * pred) - mu_p2, min=0.0)
        sigma_t2 = torch.clamp(_conv(target * target) - mu_t2, min=0.0)
        sigma_pt = _conv(pred * target) - mu_pt

        num = (2.0 * mu_pt + C1) * (2.0 * sigma_pt + C2)
        den = (mu_p2 + mu_t2 + C1) * (sigma_p2 + sigma_t2 + C2)
        ssim_map = num / den
        return 1.0 - ssim_map.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined despeckling loss.

        Args:
            pred:   Model output, shape (B, 1, H, W).
            target: Clean reference, shape (B, 1, H, W).

        Returns:
            Scalar loss tensor.
        """
        loss_char = self.charbonnier(pred, target)
        loss_ssim = self.ssim_loss(pred, target)
        return self.w_char * loss_char + self.w_ssim * loss_ssim


# ---------------------------------------------------------------------------
# Classical baselines
# ---------------------------------------------------------------------------


def lee_filter(img: np.ndarray, window_size: int = 7) -> np.ndarray:
    """Lee adaptive speckle filter.

    In each local window:
        K          = var_local / (var_local + var_noise)
        filtered   = mean + K * (pixel - mean)

    var_noise is estimated from the global ENL assumption:
        var_noise  = (mean_global / ENL)^2   with ENL = 1 (fully developed speckle)

    Args:
        img:         2-D intensity image (linear scale, non-negative).
        window_size: Side length of the square analysis window (odd preferred).

    Returns:
        Filtered image with the same shape as ``img``.
    """
    from scipy.ndimage import uniform_filter

    img = img.astype(np.float64)
    mean_local = uniform_filter(img, size=window_size)
    mean_sq_local = uniform_filter(img ** 2, size=window_size)
    var_local = mean_sq_local - mean_local ** 2

    # Global noise variance estimated from image statistics
    mean_global = img.mean()
    var_noise = (mean_global ** 2) / 1.0  # ENL = 1 for fully developed speckle

    K = var_local / (var_local + var_noise + 1e-10)
    filtered = mean_local + K * (img - mean_local)
    return filtered.astype(np.float32)


def frost_filter(img: np.ndarray, window_size: int = 7, damping: float = 2.0) -> np.ndarray:
    """Frost filter — exponentially weighted by local coefficient of variation.

    Each output pixel is a weighted average of neighbours where weights decay
    exponentially with distance, modulated by the local coefficient of variation:
        w(d) = exp(-damping * CoV * d)

    Vectorized implementation: iterates over window offsets (W²) instead of
    pixels (H×W), giving ~1000x speedup over per-pixel generic_filter.

    Args:
        img:         2-D intensity image.
        window_size: Side length of the square analysis window (odd preferred).
        damping:     Controls how strongly variation drives the weighting.

    Returns:
        Filtered image with the same shape as ``img``.
    """
    from scipy.ndimage import uniform_filter

    img = img.astype(np.float64)
    half = window_size // 2

    # Local mean and CoV via uniform_filter (C-level, fast)
    mean_local = uniform_filter(img, size=window_size)
    mean_sq = uniform_filter(img ** 2, size=window_size)
    var_local = np.maximum(mean_sq - mean_local ** 2, 0.0)
    std_local = np.sqrt(var_local)
    cov = std_local / (np.abs(mean_local) + 1e-10)  # (H, W)

    # Pad image for sliding window
    padded = np.pad(img, half, mode='reflect')

    H, W = img.shape
    weighted_sum = np.zeros_like(img)
    weight_sum = np.zeros_like(img)

    # Iterate over offsets in the window (W² iterations, not H×W)
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            dist = np.sqrt(dy ** 2 + dx ** 2)
            w = np.exp(-damping * cov * dist)  # (H, W) per-pixel weight
            neighbour = padded[half + dy:half + dy + H,
                               half + dx:half + dx + W]
            weighted_sum += w * neighbour
            weight_sum += w

    result = weighted_sum / (weight_sum + 1e-10)
    return result.astype(np.float32)


def median_filter(img: np.ndarray, window_size: int = 7) -> np.ndarray:
    """Simple median filter for speckle suppression.

    Args:
        img:         2-D intensity image.
        window_size: Side length of the square filter window.

    Returns:
        Median-filtered image with the same shape as ``img``.
    """
    from scipy.ndimage import median_filter as _scipy_median

    return _scipy_median(img.astype(np.float32), size=window_size)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def compute_psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio (dB).

    Args:
        pred:       Predicted image (any scale).
        target:     Reference image.
        data_range: Dynamic range of the data (default 1.0 for normalised images).

    Returns:
        PSNR in dB (higher is better).
    """
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-12:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def compute_ssim(pred: np.ndarray, target: np.ndarray, window_size: int = 11) -> float:
    """Structural Similarity Index (SSIM) using a Gaussian window.

    Args:
        pred:        Predicted image, 2-D float array.
        target:      Reference image, 2-D float array.
        window_size: Gaussian kernel size.

    Returns:
        SSIM value in [−1, 1] (higher is better, 1 = perfect).
    """
    from scipy.ndimage import uniform_filter

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    pred = pred.astype(np.float64)
    target = target.astype(np.float64)

    mu_p = uniform_filter(pred, size=window_size)
    mu_t = uniform_filter(target, size=window_size)
    mu_p2 = mu_p ** 2
    mu_t2 = mu_t ** 2
    mu_pt = mu_p * mu_t

    sigma_p2 = uniform_filter(pred ** 2, size=window_size) - mu_p2
    sigma_t2 = uniform_filter(target ** 2, size=window_size) - mu_t2
    sigma_pt = uniform_filter(pred * target, size=window_size) - mu_pt

    num = (2.0 * mu_pt + C1) * (2.0 * sigma_pt + C2)
    den = (mu_p2 + mu_t2 + C1) * (sigma_p2 + sigma_t2 + C2)
    ssim_map = num / (den + 1e-10)
    return float(ssim_map.mean())


def compute_enl(img: np.ndarray, roi: tuple | None = None) -> float:
    """Equivalent Number of Looks (ENL) in a homogeneous region.

    ENL = (mean / std)^2.  A higher ENL indicates smoother (less speckled) output.

    Args:
        img: 2-D intensity image (linear scale).
        roi: Optional bounding box ``(row_start, row_end, col_start, col_end)``
             selecting a homogeneous patch.  If ``None``, the full image is used.

    Returns:
        ENL value (higher is better for filtered images).
    """
    if roi is not None:
        r0, r1, c0, c1 = roi
        region = img[r0:r1, c0:c1]
    else:
        region = img

    region = region.astype(np.float64)
    mean = region.mean()
    std = region.std()
    if std < 1e-10:
        return float("inf")
    return float((mean / std) ** 2)


def compute_epi(
    filtered: np.ndarray,
    original: np.ndarray,
    reference: np.ndarray,
) -> float:
    """Edge Preservation Index (EPI).

    Measures how well the filter retains edges relative to the clean reference:
        EPI = corr(Laplacian(filtered), Laplacian(reference))
            / corr(Laplacian(original),  Laplacian(reference))

    A value close to 1 means edges are well preserved.

    Args:
        filtered:  Filter output image, 2-D float array.
        original:  Noisy input image (used as denominator baseline).
        reference: Clean reference image.

    Returns:
        EPI (higher is better, ideally near 1).
    """
    from scipy.ndimage import laplace

    def _edge(x: np.ndarray) -> np.ndarray:
        return np.abs(laplace(x.astype(np.float64)))

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        a_flat = a.ravel()
        b_flat = b.ravel()
        denom = np.std(a_flat) * np.std(b_flat)
        if denom < 1e-10:
            return 0.0
        return float(np.corrcoef(a_flat, b_flat)[0, 1])

    e_filt = _edge(filtered)
    e_orig = _edge(original)
    e_ref = _edge(reference)

    num = _corr(e_filt, e_ref)
    den = _corr(e_orig, e_ref)
    if abs(den) < 1e-10:
        return 0.0
    return num / den


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- DnCNN-SAR forward pass ---
    model = DnCNNSAR(n_channels=1, n_filters=64, n_layers=17)
    model.eval()

    x = torch.randn(4, 1, 64, 64)
    with torch.no_grad():
        y = model(x)

    print(f"DnCNN-SAR parameters: {count_parameters(model):,}")
    print(f"input shape:          {tuple(x.shape)}")
    print(f"output shape:         {tuple(y.shape)}")
    print(f"residual learning OK: {not torch.allclose(x, y)}")

    # --- Lee filter quick test ---
    rng = np.random.default_rng(0)
    noisy = rng.exponential(scale=1.0, size=(128, 128)).astype(np.float32)
    filtered = lee_filter(noisy, window_size=7)
    print(f"\nLee filter test")
    print(f"  input  ENL: {compute_enl(noisy):.2f}")
    print(f"  output ENL: {compute_enl(filtered):.2f}  (should be higher)")
