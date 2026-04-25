"""DoA (Direction of Arrival) 유틸리티 — 교육용

데이터 생성, 고전 알고리즘 (CBF, MUSIC, MVDR), 평가 함수.
DeepMUSIC 프로젝트와 gen_figures.py 양쪽에서 사용.

사용법:
    from shared.doa_utils import (
        steering_vector, generate_doa_sample, generate_doa_dataset,
        cbf_spectrum, music_spectrum, mvdr_spectrum,
        find_spectrum_peaks, compute_doa_rmse,
    )
"""

import numpy as np

C = 299_792_458.0


def steering_vector(angles_deg, N_rx, d_over_lam=0.5):
    """ULA steering vector(s).

    Parameters
    ----------
    angles_deg : float or array-like
        Source angle(s) in degrees.
    N_rx : int
        Number of antenna elements.
    d_over_lam : float
        Element spacing / wavelength (default: 0.5 = λ/2).

    Returns
    -------
    A : ndarray, shape (N_rx,) or (N_rx, K)
        Steering vector(s).
    """
    angles = np.atleast_1d(np.asarray(angles_deg, dtype=np.float64))
    n = np.arange(N_rx)[:, None]  # (N_rx, 1)
    phase = 2 * np.pi * d_over_lam * np.sin(np.radians(angles))[None, :]  # (1, K)
    A = np.exp(1j * n * phase)  # (N_rx, K)
    return A.squeeze()


def generate_doa_sample(
    N_rx=12,
    d_over_lam=0.5,
    n_sources_range=(1, 3),
    angle_range=(-60.0, 60.0),
    min_angle_sep=3.0,
    snr_range=(0.0, 20.0),
    n_snapshots_range=(10, 200),
    coherent_prob=0.2,
    moving_prob=0.0,
    max_drift_deg=5.0,
    grid_size=181,
    grid_range=(-90.0, 90.0),
    label_sigma=1.0,
    rng=None,
):
    """단일 DoA 시나리오 생성.

    Returns
    -------
    cov_real : ndarray (2, N_rx, N_rx)
        Sample covariance의 real/imag 스택.
    spectrum_label : ndarray (grid_size,)
        GT Gaussian angle heatmap (peaks at true angles).
    angles_true : list of float
        True source angles (degrees).
    meta : dict
        시나리오 메타데이터 (snr, n_snapshots, coherent, n_sources).
    """
    if rng is None:
        rng = np.random.default_rng()

    # --- 시나리오 파라미터 ---
    K = rng.integers(n_sources_range[0], n_sources_range[1] + 1)
    snr_db = rng.uniform(snr_range[0], snr_range[1])
    T = rng.integers(n_snapshots_range[0], n_snapshots_range[1] + 1)
    is_coherent = (K >= 2) and (rng.random() < coherent_prob)

    # --- 소스 각도 (최소 간격 보장) ---
    angles = _sample_angles(K, angle_range, min_angle_sep, rng)

    # --- 신호 생성 ---
    A = steering_vector(angles, N_rx, d_over_lam)  # (N_rx, K)
    if K == 1:
        A = A[:, None]

    snr_lin = 10 ** (snr_db / 10.0)
    sig_power = snr_lin  # noise power = 1

    if is_coherent and K >= 2:
        # Coherent: 모든 소스가 동일 waveform의 복소 스케일 버전
        s0 = np.sqrt(sig_power) * (rng.standard_normal((1, T)) +
                                    1j * rng.standard_normal((1, T))) / np.sqrt(2)
        # 각 소스에 랜덤 복소 계수
        coeffs = rng.standard_normal(K) + 1j * rng.standard_normal(K)
        coeffs = coeffs / np.abs(coeffs)  # unit magnitude
        S = coeffs[:, None] * s0  # (K, T)
    else:
        # Uncorrelated sources
        S = np.sqrt(sig_power) * (rng.standard_normal((K, T)) +
                                   1j * rng.standard_normal((K, T))) / np.sqrt(2)

    # Noise
    N_noise = (rng.standard_normal((N_rx, T)) +
               1j * rng.standard_normal((N_rx, T))) / np.sqrt(2)

    # Moving targets: time-varying steering vectors
    is_moving = (moving_prob > 0) and (rng.random() < moving_prob)
    if is_moving:
        drifts = rng.uniform(-max_drift_deg, max_drift_deg, size=K)
        X = np.zeros((N_rx, T), dtype=complex)
        for t in range(T):
            frac = t / max(T - 1, 1)
            angles_t = angles + drifts * frac
            A_t = steering_vector(angles_t, N_rx, d_over_lam)
            if K == 1:
                A_t = A_t[:, None]
            X[:, t] = (A_t @ S[:, t]) + N_noise[:, t]
    else:
        drifts = None
        X = A @ S + N_noise  # (N_rx, T)

    # --- Sample covariance ---
    R = (X @ X.conj().T) / T  # (N_rx, N_rx)

    # Normalize (divide by Frobenius norm for numerical stability)
    R_norm = R / (np.linalg.norm(R, 'fro') + 1e-10)

    cov_real = np.stack([R_norm.real, R_norm.imag], axis=0)  # (2, N_rx, N_rx)

    # --- GT Gaussian angle heatmap ---
    grid = np.linspace(grid_range[0], grid_range[1], grid_size)
    spectrum_label = np.zeros(grid_size, dtype=np.float32)
    # For moving targets, use midpoint angles as ground truth
    angles_gt = angles + drifts * 0.5 if is_moving else angles
    for a in angles_gt:
        spectrum_label += np.exp(-0.5 * ((grid - a) / label_sigma) ** 2)
    # Clip to [0, 1]
    spectrum_label = np.clip(spectrum_label, 0.0, 1.0)

    meta = {
        'snr_db': float(snr_db),
        'n_snapshots': int(T),
        'n_sources': int(K),
        'coherent': bool(is_coherent),
        'moving': bool(is_moving),
    }

    return cov_real.astype(np.float32), spectrum_label, angles_gt.tolist(), meta


def generate_doa_dataset(n_samples, seed=42, **kwargs):
    """다수의 DoA 샘플 생성 (generator).

    Yields
    ------
    (cov_real, spectrum_label, angles_true, meta)
    """
    rng = np.random.default_rng(seed)
    for _ in range(n_samples):
        yield generate_doa_sample(rng=rng, **kwargs)


def _sample_angles(K, angle_range, min_sep, rng, max_attempts=100):
    """최소 간격을 보장하며 K개 각도 샘플링."""
    for _ in range(max_attempts):
        angles = rng.uniform(angle_range[0], angle_range[1], size=K)
        angles = np.sort(angles)
        if K == 1:
            return angles
        diffs = np.diff(angles)
        if np.all(diffs >= min_sep):
            return angles
    # Fallback: evenly spaced
    return np.linspace(angle_range[0] + 5, angle_range[1] - 5, K)


# ─── Classical DoA Algorithms ──────────────────────────────────────────────────

def cbf_spectrum(R, N_rx, angles_deg, d_over_lam=0.5):
    """Conventional Beamformer (CBF) spatial spectrum.

    Parameters
    ----------
    R : ndarray (N_rx, N_rx)
        Sample covariance matrix.
    N_rx : int
    angles_deg : ndarray
    d_over_lam : float

    Returns
    -------
    P : ndarray, shape (len(angles_deg),)
    """
    P = np.zeros(len(angles_deg))
    for i, theta in enumerate(angles_deg):
        a = steering_vector(theta, N_rx, d_over_lam)
        P[i] = np.real(a.conj() @ R @ a)
    return P


def music_spectrum(R, N_rx, angles_deg, n_sources, d_over_lam=0.5):
    """MUSIC spatial spectrum.

    Parameters
    ----------
    R : ndarray (N_rx, N_rx)
    n_sources : int
        Assumed number of sources.

    Returns
    -------
    P : ndarray
    """
    eigvals, eigvecs = np.linalg.eigh(R)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    En = eigvecs[:, n_sources:]  # Noise subspace

    P = np.zeros(len(angles_deg))
    for i, theta in enumerate(angles_deg):
        a = steering_vector(theta, N_rx, d_over_lam)
        denom = np.real(a.conj() @ En @ En.conj().T @ a)
        P[i] = 1.0 / (denom + 1e-20)
    return P


def mvdr_spectrum(R, N_rx, angles_deg, d_over_lam=0.5):
    """MVDR (Capon) spatial spectrum.

    Parameters
    ----------
    R : ndarray (N_rx, N_rx)

    Returns
    -------
    P : ndarray
    """
    try:
        R_inv = np.linalg.inv(R + 1e-8 * np.eye(N_rx))
    except np.linalg.LinAlgError:
        R_inv = np.linalg.pinv(R)

    P = np.zeros(len(angles_deg))
    for i, theta in enumerate(angles_deg):
        a = steering_vector(theta, N_rx, d_over_lam)
        P[i] = 1.0 / (np.real(a.conj() @ R_inv @ a) + 1e-20)
    return P


# ─── Evaluation Helpers ────────────────────────────────────────────────────────

def find_spectrum_peaks(P, angles_deg, n_sources, min_height_ratio=0.1):
    """스펙트럼에서 n_sources개 피크 추출.

    Parameters
    ----------
    P : ndarray
        Spatial spectrum.
    angles_deg : ndarray
        Angle grid.
    n_sources : int

    Returns
    -------
    peak_angles : ndarray
    """
    from scipy.signal import find_peaks

    P_norm = P / (P.max() + 1e-20)
    peaks, props = find_peaks(P_norm, height=min_height_ratio, distance=3)

    if len(peaks) == 0:
        idx = np.argsort(P)[::-1][:n_sources]
        return np.sort(angles_deg[idx])

    heights = P[peaks]
    top_idx = np.argsort(heights)[::-1][:n_sources]
    top_peaks = peaks[top_idx]
    return np.sort(angles_deg[top_peaks])


def compute_doa_rmse(est_angles, true_angles, detailed=False):
    """Greedy nearest-neighbor matching으로 RMSE 계산.

    Parameters
    ----------
    est_angles : array-like
    true_angles : array-like
    detailed : bool
        True면 dict 반환 (RMSE + miss/FA counts), False면 float(RMSE)만 반환.

    Returns
    -------
    float (if detailed=False) or dict (if detailed=True)
        dict keys: 'rmse', 'n_true', 'n_est', 'n_matched', 'n_miss', 'n_fa'
    """
    true = np.atleast_1d(np.array(true_angles, dtype=float))
    est = np.atleast_1d(np.array(est_angles, dtype=float))

    MATCH_THRESHOLD = 10.0  # degrees

    if len(est) == 0:
        if detailed:
            return {'rmse': 90.0, 'n_true': len(true), 'n_est': 0,
                    'n_matched': 0, 'n_miss': len(true), 'n_fa': 0}
        return 90.0

    used = [False] * len(est)
    errors = []
    n_matched = 0
    for t in true:
        dists = [abs(t - e) if not used[j] else 1e9 for j, e in enumerate(est)]
        best = int(np.argmin(dists))
        if dists[best] <= MATCH_THRESHOLD:
            errors.append((t - est[best]) ** 2)
            used[best] = True
            n_matched += 1

    n_miss = len(true) - n_matched
    n_fa = len(est) - n_matched
    rmse = float(np.sqrt(np.mean(errors))) if errors else 90.0

    if detailed:
        return {'rmse': rmse, 'n_true': len(true), 'n_est': len(est),
                'n_matched': n_matched, 'n_miss': n_miss, 'n_fa': n_fa}
    return rmse


# ─── Source Number Estimation ─────────────────────────────────────────────────

def estimate_n_sources_mdl(R, T):
    """MDL criterion for source number estimation.

    Parameters
    ----------
    R : ndarray (N, N)
        Sample covariance matrix.
    T : int
        Number of snapshots.

    Returns
    -------
    K_hat : int
        Estimated number of sources.
    """
    N = R.shape[0]
    eigvals = np.sort(np.real(np.linalg.eigvalsh(R)))[::-1]
    eigvals = np.maximum(eigvals, 1e-20)

    best_k, best_mdl = 0, np.inf
    for k in range(N):
        noise_eigs = eigvals[k:]
        m = len(noise_eigs)
        if m <= 0:
            break
        geo_mean = np.exp(np.mean(np.log(noise_eigs)))
        ari_mean = np.mean(noise_eigs)
        log_lik = T * m * np.log(ari_mean / (geo_mean + 1e-20))
        penalty = 0.5 * k * (2 * N - k) * np.log(T)
        mdl = log_lik + penalty
        if mdl < best_mdl:
            best_mdl = mdl
            best_k = k

    return best_k


def estimate_n_sources_aic(R, T):
    """AIC criterion for source number estimation.

    Parameters
    ----------
    R : ndarray (N, N)
        Sample covariance matrix.
    T : int
        Number of snapshots.

    Returns
    -------
    K_hat : int
        Estimated number of sources.
    """
    N = R.shape[0]
    eigvals = np.sort(np.real(np.linalg.eigvalsh(R)))[::-1]
    eigvals = np.maximum(eigvals, 1e-20)

    best_k, best_aic = 0, np.inf
    for k in range(N):
        noise_eigs = eigvals[k:]
        m = len(noise_eigs)
        if m <= 0:
            break
        geo_mean = np.exp(np.mean(np.log(noise_eigs)))
        ari_mean = np.mean(noise_eigs)
        log_lik = T * m * np.log(ari_mean / (geo_mean + 1e-20))
        penalty = k * (2 * N - k)
        aic = log_lik + penalty
        if aic < best_aic:
            best_aic = aic
            best_k = k

    return best_k


def find_spectrum_peaks_auto(P, angles_deg, min_height_ratio=0.3, min_distance=5):
    """Find peaks adaptively without knowing K.

    Parameters
    ----------
    P : ndarray
        Spatial spectrum (e.g., DNN output).
    angles_deg : ndarray
        Angle grid.
    min_height_ratio : float
        Minimum peak height relative to max.
    min_distance : int
        Minimum distance between peaks (grid points).

    Returns
    -------
    peak_angles : ndarray
    """
    from scipy.signal import find_peaks

    P_norm = P / (P.max() + 1e-20)
    peaks, _ = find_peaks(P_norm, height=min_height_ratio, distance=min_distance)

    if len(peaks) == 0:
        return np.array([angles_deg[np.argmax(P)]])

    return np.sort(angles_deg[peaks])
