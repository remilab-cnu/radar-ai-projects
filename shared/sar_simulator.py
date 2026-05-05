"""SAR 시뮬레이터 — 교육용

Stripmap SAR 시뮬레이션: 점 표적 기반 영상 형성, 위상 오차 주입/보정, PGA.
P4 SAR despeckling/autofocus support module.

사용법:
    from shared.sar_simulator import StripmapSAR, generate_sar_image, defocus_image
"""

import numpy as np

C = 299_792_458.0


class StripmapSAR:
    """Stripmap SAR 시스템 파라미터.

    Parameters
    ----------
    fc : float — carrier frequency [Hz] (default: 9.6 GHz, X-band)
    bw : float — chirp bandwidth [Hz]
    prf : float — pulse repetition frequency [Hz]
    V : float — platform velocity [m/s]
    R0 : float — scene center range [m]
    N_az, N_rg : int — azimuth / range samples
    """

    def __init__(self, fc=9.6e9, bw=100e6, prf=400, V=200, R0=10000,
                 N_az=256, N_rg=256):
        self.fc = fc
        self.bw = bw
        self.prf = prf
        self.V = V
        self.R0 = R0
        self.N_az = N_az
        self.N_rg = N_rg

        self.lam = C / fc
        self.range_res = C / (2 * bw)
        self.az_extent = V * N_az / prf
        self.rg_extent = N_rg * self.range_res


# ─── Image Formation ──────────────────────────────────────────────────────────

def generate_sar_image(sar, targets, snr_db=20, rng=None):
    """Generate focused SAR image from point targets via simplified RDA.

    Parameters
    ----------
    sar : StripmapSAR
    targets : list of dict — {'x': cross-range [m], 'r': range offset [m], 'rcs': float}
    snr_db : float
    rng : np.random.Generator

    Returns
    -------
    image : ndarray (N_az, N_rg) complex — focused SAR image
    """
    N_az, N_rg = sar.N_az, sar.N_rg

    # Platform azimuth positions (centered)
    eta = np.arange(N_az) / sar.prf
    x_p = sar.V * (eta - eta[-1] / 2)

    # Range bins
    r_start = sar.R0 - (N_rg / 2) * sar.range_res
    r_bins = r_start + np.arange(N_rg) * sar.range_res

    # Generate range-compressed raw data (vectorized over all targets)
    raw = np.zeros((N_az, N_rg), dtype=np.complex128)

    if targets:
        x_t = np.array([t['x'] for t in targets])
        r_t = sar.R0 + np.array([t['r'] for t in targets])
        sigma = np.sqrt(np.maximum(
            np.array([t.get('rcs', 1.0) for t in targets]), 0.01))

        # R: (K, N_az) — range from each target to each azimuth position
        R = np.sqrt(r_t[:, None] ** 2 + (x_p[None, :] - x_t[:, None]) ** 2)
        r_idx = np.round((R - r_start) / sar.range_res).astype(int)
        phase = -4 * np.pi / sar.lam * R

        valid = (r_idx >= 0) & (r_idx < N_rg)
        n_idx = np.broadcast_to(np.arange(N_az)[None, :], R.shape)
        vals = sigma[:, None] * np.exp(1j * phase)

        np.add.at(raw, (n_idx[valid], r_idx[valid]), vals[valid])

    # Noise
    if rng is not None:
        sig_power = np.mean(np.abs(raw) ** 2) + 1e-30
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (
            rng.standard_normal(raw.shape) + 1j * rng.standard_normal(raw.shape))
        raw += noise

    # Azimuth compression (range-dependent matched filter)
    image = _azimuth_compress(raw, sar, r_bins)
    return image


def _azimuth_compress(raw, sar, r_bins):
    """RDA azimuth compression with range-dependent matched filter (vectorized)."""
    N_az = raw.shape[0]
    f_az = np.fft.fftfreq(N_az, d=1 / sar.prf)
    Raw_f = np.fft.fft(raw, axis=0)

    K_a = 2 * sar.V ** 2 / (sar.lam * r_bins)  # (N_rg,)
    H = np.exp(-1j * np.pi * f_az[:, None] ** 2 / K_a[None, :])  # (N_az, N_rg)
    Raw_f *= H

    return np.fft.fftshift(np.fft.ifft(Raw_f, axis=0), axes=0)


# ─── Phase Error ──────────────────────────────────────────────────────────────

def defocus_image(image, coefficients):
    """Apply polynomial phase error in azimuth Doppler domain.

    Phase error: φ(f) = Σ c_k * f_norm^(k+2) for k=0..K-1
    where f_norm ∈ [-1, 1].

    Parameters
    ----------
    image : ndarray (N_az, N_rg) complex
    coefficients : array (K,) — c_2, c_3, ..., c_{K+1}
    """
    N_az = image.shape[0]
    f_norm = np.linspace(-1, 1, N_az)
    phi = sum(c * f_norm ** (k + 2) for k, c in enumerate(coefficients))

    S = np.fft.fft(image, axis=0)
    S *= np.exp(1j * phi)[:, None]
    return np.fft.ifft(S, axis=0)


def refocus_image(defocused, coefficients):
    """Remove phase error with known coefficients."""
    return defocus_image(defocused, -np.asarray(coefficients))


def pga_autofocus(defocused_image, n_iter=5):
    """Simplified Phase Gradient Autofocus.

    Returns
    -------
    focused : ndarray (N_az, N_rg) complex
    phi_est : ndarray (N_az,) — estimated phase error
    """
    N_az, N_rg = defocused_image.shape
    S = np.fft.fft(defocused_image, axis=0)
    phi_total = np.zeros(N_az)

    for _ in range(n_iter):
        image = np.fft.ifft(S, axis=0)
        mag = np.abs(image)

        # Select bright range bins (top 30%)
        col_max = mag.max(axis=0)
        thresh = np.percentile(col_max, 70)
        sel = np.where(col_max > thresh)[0]
        if len(sel) == 0:
            break

        # Center each column on peak, apply Hann window, transform to Doppler
        win_len = max(int(N_az * 0.4), 8)  # ~40% of azimuth extent
        hann = np.hanning(win_len)
        centered_S = np.zeros((N_az, len(sel)), dtype=complex)
        for i, j in enumerate(sel):
            peak = np.argmax(mag[:, j])
            shifted = np.roll(image[:, j], N_az // 2 - peak)
            # Apply Hann window centered on the (now centered) peak
            windowed = np.zeros(N_az, dtype=complex)
            start = N_az // 2 - win_len // 2
            windowed[start:start + win_len] = shifted[start:start + win_len] * hann
            centered_S[:, i] = np.fft.fft(windowed)

        # Phase gradient estimation
        dphi = np.angle(np.sum(
            centered_S[1:] * np.conj(centered_S[:-1]), axis=1))

        phi_est = np.zeros(N_az)
        phi_est[1:] = np.cumsum(dphi)
        phi_est -= phi_est.mean()

        S *= np.exp(-1j * phi_est)[:, None]
        phi_total += phi_est

    focused = np.fft.ifft(S, axis=0)
    return focused, phi_total


# ─── Image Quality Metrics ───────────────────────────────────────────────────

def image_entropy(image):
    """Image entropy based on intensity (lower = better focused).

    Uses |I|^2 (intensity) normalization, the standard SAR autofocus metric.
    Magnitude-based (|I|) normalization inverts the metric for SAR images
    because defocusing redistributes energy in a way that lowers magnitude
    entropy while increasing intensity entropy.
    """
    intensity = np.abs(image).ravel() ** 2
    intensity = intensity / (intensity.sum() + 1e-30)
    intensity = intensity[intensity > 0]
    return -np.sum(intensity * np.log(intensity))


def image_contrast(image):
    """Image contrast ratio (higher = better focused)."""
    mag = np.abs(image)
    return float(mag.std() / (mag.mean() + 1e-30))


def sar_to_db(image, clip_db=50, ref_max=None):
    """Complex SAR → dB magnitude normalized to [0, 1].

    Args:
        image:   Complex SAR image.
        clip_db: Dynamic range in dB (default 50).
        ref_max: Reference magnitude for 0 dB normalization.
                 If None, uses the image's own max.
                 Pass the clean image's max when converting noisy images
                 to ensure consistent normalization.
    """
    mag = np.abs(image)
    if ref_max is None:
        ref_max = mag.max()
    mag_db = 20 * np.log10(mag / (ref_max + 1e-30) + 1e-30)
    mag_db = np.clip(mag_db, -clip_db, 0)
    return ((mag_db + clip_db) / clip_db).astype(np.float32)


# ─── Speckle Noise ────────────────────────────────────────────────────────────

def add_speckle(image, n_looks=1, rng=None):
    """Multiplicative speckle noise (gamma distributed)."""
    if rng is None:
        rng = np.random.default_rng()
    speckle = np.sqrt(rng.gamma(n_looks, 1.0 / n_looks, image.shape))
    return image * speckle


# ─── Target Scatterer Models ─────────────────────────────────────────────────

TARGET_CLASSES = ['sedan', 'truck', 'tank', 'apc',
                  'building', 'tower', 'fence', 'background']
TARGET_TO_IDX = {t: i for i, t in enumerate(TARGET_CLASSES)}
N_TARGET_CLASSES = len(TARGET_CLASSES)


def _sedan():
    # x(cross-range), y(range), z(height), rcs
    return np.array([
        [-2.0, -0.8, 0.3, 1.5], [-2.0, 0.8, 0.3, 1.5],
        [2.0, -0.8, 0.3, 1.5], [2.0, 0.8, 0.3, 1.5],
        [-1.5, 0.0, 1.0, 2.0], [1.5, 0.0, 1.0, 1.5],
        [-2.2, -0.9, 0.5, 1.0], [-2.2, 0.9, 0.5, 1.0],
        [2.2, -0.9, 0.5, 1.0], [2.2, 0.9, 0.5, 1.0],
        [0.0, -0.9, 0.7, 0.8], [0.0, 0.9, 0.7, 0.8],
    ])


def _truck():
    return np.array([
        [-3.5, -1.2, 1.5, 2.5], [-3.5, 1.2, 1.5, 2.5],
        [-3.5, 0.0, 2.5, 3.0],
        [-1.0, -1.2, 1.0, 1.5], [-1.0, 1.2, 1.0, 1.5],
        [4.0, -1.2, 1.0, 1.5], [4.0, 1.2, 1.0, 1.5],
        [1.5, -1.2, 2.0, 1.0], [1.5, 1.2, 2.0, 1.0],
        [-2.5, -1.2, 0.3, 1.2], [-2.5, 1.2, 0.3, 1.2],
        [0.5, -1.2, 0.3, 1.2], [0.5, 1.2, 0.3, 1.2],
        [3.0, -1.2, 0.3, 1.2], [3.0, 1.2, 0.3, 1.2],
        [-3.5, -1.4, 1.8, 0.5], [-3.5, 1.4, 1.8, 0.5],
        [4.0, 0.0, 0.5, 0.8],
    ])


def _tank():
    return np.array([
        [-3.5, -1.7, 0.5, 2.0], [-3.5, 1.7, 0.5, 2.0],
        [3.5, -1.7, 0.5, 2.0], [3.5, 1.7, 0.5, 2.0],
        [0.0, -1.7, 0.5, 1.5], [0.0, 1.7, 0.5, 1.5],
        [0.5, -1.0, 1.5, 3.0], [0.5, 1.0, 1.5, 3.0],
        [-0.5, -1.0, 1.5, 2.5], [-0.5, 1.0, 1.5, 2.5],
        [0.0, 0.0, 2.0, 3.5],
        [-3.0, 0.0, 1.8, 1.0], [-4.5, 0.0, 1.8, 0.8],
        [-6.0, 0.0, 1.8, 0.5],
        [-2.0, -1.8, 0.3, 1.0], [2.0, -1.8, 0.3, 1.0],
        [-2.0, 1.8, 0.3, 1.0], [2.0, 1.8, 0.3, 1.0],
        [1.0, 0.5, 2.2, 0.8], [-1.0, -0.5, 2.2, 0.8],
    ])


def _apc():
    return np.array([
        [-3.0, -1.4, 0.5, 1.8], [-3.0, 1.4, 0.5, 1.8],
        [3.0, -1.4, 0.5, 1.8], [3.0, 1.4, 0.5, 1.8],
        [0.0, -1.4, 1.0, 2.0], [0.0, 1.4, 1.0, 2.0],
        [0.0, 0.0, 1.8, 1.5],
        [-0.5, 0.0, 2.0, 2.5], [-1.5, 0.0, 2.0, 1.0],
        [-2.5, -1.4, 0.3, 1.0], [-2.5, 1.4, 0.3, 1.0],
        [-0.5, -1.4, 0.3, 1.0], [-0.5, 1.4, 0.3, 1.0],
        [1.5, -1.4, 0.3, 1.0], [1.5, 1.4, 0.3, 1.0],
    ])


def _building():
    return np.array([
        [-5, -5, 0, 4.0], [-5, 5, 0, 4.0],
        [5, -5, 0, 4.0], [5, 5, 0, 4.0],
        [-5, 0, 0, 2.0], [5, 0, 0, 2.0],
        [0, -5, 0, 2.0], [0, 5, 0, 2.0],
        [-5, -5, 6, 1.5], [-5, 5, 6, 1.5],
        [5, -5, 6, 1.5], [5, 5, 6, 1.5],
        [0, -5, 6, 1.0], [0, 5, 6, 1.0],
        [-5, 0, 6, 1.0], [5, 0, 6, 1.0],
        [-5, -2.5, 3, 1.5], [-5, 2.5, 3, 1.5],
        [5, -2.5, 3, 1.5], [5, 2.5, 3, 1.5],
        [-2.5, -5, 3, 1.5], [2.5, -5, 3, 1.5],
        [-2.5, 5, 3, 1.5], [2.5, 5, 3, 1.5],
        [0, 0, 3, 0.5],
    ])


def _tower():
    return np.array([
        [-1, -1, 0, 2.0], [-1, 1, 0, 2.0],
        [1, -1, 0, 2.0], [1, 1, 0, 2.0],
        [-0.8, -0.8, 7, 1.5], [-0.8, 0.8, 7, 1.5],
        [0.8, -0.8, 7, 1.5], [0.8, 0.8, 7, 1.5],
        [0, 0, 15, 3.0], [0, 0, 12, 2.0],
    ])


def _fence():
    return np.array([
        [-7, 0, 0, 1.0], [-5, 0, 0, 1.0], [-3, 0, 0, 1.0],
        [-1, 0, 0, 1.0], [1, 0, 0, 1.0], [3, 0, 0, 1.0],
        [5, 0, 0, 1.0], [7, 0, 0, 1.0],
        [-7, 0, 1.5, 0.5], [-5, 0, 1.5, 0.5], [-3, 0, 1.5, 0.5],
        [-1, 0, 1.5, 0.5], [1, 0, 1.5, 0.5], [3, 0, 1.5, 0.5],
        [5, 0, 1.5, 0.5],
    ])


_TARGET_FN = {
    'sedan': _sedan, 'truck': _truck, 'tank': _tank, 'apc': _apc,
    'building': _building, 'tower': _tower, 'fence': _fence,
}


def get_target_scatterers(target_type, rng=None):
    """Get scatterer array (N, 4): [x, y, z, rcs]."""
    if target_type == 'background':
        if rng is None:
            rng = np.random.default_rng()
        n = rng.integers(20, 40)
        return np.column_stack([
            rng.uniform(-5, 5, n), rng.uniform(-5, 5, n),
            np.zeros(n), rng.exponential(0.3, n)])
    return _TARGET_FN[target_type]().copy()


def generate_target_patch(sar, target_type, aspect_deg, depression_deg=15,
                           rng=None, patch_size=88, speckle_looks=3):
    """Generate SAR image patch of a target at given aspect angle.

    Returns
    -------
    patch : ndarray (patch_size, patch_size) float32 — dB normalized [0,1]
    """
    if rng is None:
        rng = np.random.default_rng()

    scatterers = get_target_scatterers(target_type, rng)

    # Rotate by aspect angle
    theta = np.radians(aspect_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x_rot = scatterers[:, 0] * cos_t - scatterers[:, 1] * sin_t
    y_rot = scatterers[:, 0] * sin_t + scatterers[:, 1] * cos_t
    z = scatterers[:, 2]
    rcs = scatterers[:, 3]

    # Depression: height → range offset (layover)
    dep_rad = np.radians(max(depression_deg, 1))
    y_proj = y_rot - z / np.tan(dep_rad)

    targets = [{'x': float(x_rot[i]), 'r': float(y_proj[i]), 'rcs': float(rcs[i])}
               for i in range(len(scatterers))]

    # High-resolution SAR for ATR patches
    # Override bw and prf for sub-meter resolution so scatterer models
    # (spanning ~10 m) produce realistic-looking target images.
    #   bw=600 MHz → range_res ≈ 0.25 m → 10 m target ≈ 40 pixels
    #   prf=1200   → az_extent ≈ 29 m   → 10 m target ≈ 60 pixels
    patch_sar = StripmapSAR(
        fc=sar.fc, bw=600e6, prf=1200, V=sar.V, R0=sar.R0,
        N_az=patch_size * 2, N_rg=patch_size * 2)

    image = generate_sar_image(patch_sar, targets, snr_db=25, rng=rng)
    image = add_speckle(image, n_looks=speckle_looks, rng=rng)

    # Extract center patch
    ca, cr = image.shape[0] // 2, image.shape[1] // 2
    h = patch_size // 2
    patch = image[ca - h:ca + h, cr - h:cr + h]

    return sar_to_db(patch)


# ─── Random Scene Generation (P4: Autofocus) ─────────────────────────────────

def generate_random_autofocus_sample(sar, rng, n_targets_range=(10, 50),
                                      severity_range=(0.5, 2.0),
                                      return_complex=False):
    """Generate one autofocus training sample.

    Returns
    -------
    defocused_db : ndarray (N_az, N_rg) float32 — input image [0,1]
    coefficients : ndarray (5,) float32 — GT phase error coefficients
    focused_db : ndarray (N_az, N_rg) float32 — clean reference
    defocused_complex : ndarray (N_az, N_rg) complex128 — only if return_complex=True
    focused_complex : ndarray (N_az, N_rg) complex128 — only if return_complex=True
    """
    K = rng.integers(n_targets_range[0], n_targets_range[1] + 1)
    max_x = sar.az_extent * 0.3
    max_r = sar.rg_extent * 0.3

    targets = []
    for _ in range(K):
        targets.append({
            'x': rng.uniform(-max_x, max_x),
            'r': rng.uniform(-max_r, max_r),
            'rcs': 10 ** rng.uniform(-1, 1),
        })

    # Add distributed clutter
    n_clutter = rng.integers(30, 80)
    for _ in range(n_clutter):
        targets.append({
            'x': rng.uniform(-max_x * 1.2, max_x * 1.2),
            'r': rng.uniform(-max_r * 1.2, max_r * 1.2),
            'rcs': rng.exponential(0.05),
        })

    snr_db = rng.uniform(10, 25)
    focused = generate_sar_image(sar, targets, snr_db=snr_db, rng=rng)

    # Random polynomial phase error (5 coefficients for orders 2..6)
    # Per-coefficient scaling: each coefficient independently bounded by severity.
    # Old global scaling caused heavy tails when polynomial terms partially cancel.
    severity = rng.uniform(severity_range[0], severity_range[1])
    coefficients = (rng.standard_normal(5) * severity * np.pi / 3).astype(np.float32)

    defocused = defocus_image(focused, coefficients)

    if return_complex:
        return sar_to_db(defocused), coefficients, sar_to_db(focused), defocused, focused
    return sar_to_db(defocused), coefficients, sar_to_db(focused)
