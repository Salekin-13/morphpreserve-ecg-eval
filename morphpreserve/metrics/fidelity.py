"""Signal-fidelity metrics between a clean reference and a denoised estimate.

These are the classical waveform-distance metrics used in the ECG denoising
literature and in DeepFilter (Romero et al., 2021): SSD, MAD, PRD and cosine
similarity, plus RMSE, correlation and SNR improvement.  All operate on 1-D
arrays; a batched helper is provided for 2-D (n_segments, n_samples) input to
mirror DeepFilter's axis-1 convention.
"""

from __future__ import annotations

import numpy as np


def _check(y: np.ndarray, y_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    if y.shape != y_hat.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {y_hat.shape}")
    return y, y_hat


def ssd(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Sum of Squared Distance.  Lower is better; 0 for identical signals."""
    y, y_hat = _check(y, y_hat)
    return float(np.sum((y - y_hat) ** 2))


def mad(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Maximum Absolute Distance.  Lower is better."""
    y, y_hat = _check(y, y_hat)
    return float(np.max(np.abs(y - y_hat)))


def prd(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Percentage Root-mean-square Difference (mean-corrected), in %.

    Matches DeepFilter's definition:
        PRD = 100 * sqrt( sum((y_hat - y)^2) / sum((y_hat - mean(y))^2) )
    Lower is better.
    """
    y, y_hat = _check(y, y_hat)
    num = np.sum((y_hat - y) ** 2)
    den = np.sum((y_hat - np.mean(y)) ** 2)
    if den < 1e-20:
        return float("inf") if num > 0 else 0.0
    return float(np.sqrt(num / den) * 100.0)


def cosine_similarity(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Cosine similarity in [-1, 1].  Higher is better; 1 for identical shape."""
    y, y_hat = _check(y, y_hat)
    ny = np.linalg.norm(y)
    nh = np.linalg.norm(y_hat)
    if ny < 1e-20 or nh < 1e-20:
        return 0.0
    return float(np.dot(y, y_hat) / (ny * nh))


def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Root mean squared error.  Lower is better."""
    y, y_hat = _check(y, y_hat)
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def correlation(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Pearson correlation coefficient.  Higher is better."""
    y, y_hat = _check(y, y_hat)
    if np.std(y) < 1e-20 or np.std(y_hat) < 1e-20:
        return 0.0
    return float(np.corrcoef(y, y_hat)[0, 1])


def snr_db(clean: np.ndarray, estimate: np.ndarray) -> float:
    """Output SNR (dB): signal power over residual (clean - estimate) power."""
    clean, estimate = _check(clean, estimate)
    resid = clean - estimate
    pr = np.mean(resid**2)
    if pr < 1e-20:
        return float("inf")
    return float(10.0 * np.log10(np.mean(clean**2) / pr))


def snr_improvement(clean: np.ndarray, noisy: np.ndarray,
                    denoised: np.ndarray) -> float:
    """SNR improvement (dB) = output SNR - input SNR.

    Positive means the denoiser reduced the residual relative to the clean
    reference.  This is the single most informative fidelity number for
    ranking denoisers.
    """
    return snr_db(clean, denoised) - snr_db(clean, noisy)


def all_fidelity_metrics(clean: np.ndarray, noisy: np.ndarray,
                         denoised: np.ndarray) -> dict:
    """Compute the full fidelity metric bundle for one signal triple."""
    return {
        "ssd": ssd(clean, denoised),
        "mad": mad(clean, denoised),
        "prd": prd(clean, denoised),
        "cosine_sim": cosine_similarity(clean, denoised),
        "rmse": rmse(clean, denoised),
        "correlation": correlation(clean, denoised),
        "snr_in_db": snr_db(clean, noisy),
        "snr_out_db": snr_db(clean, denoised),
        "snr_improvement_db": snr_improvement(clean, noisy, denoised),
    }
