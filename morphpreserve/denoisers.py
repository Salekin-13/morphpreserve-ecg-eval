"""Baseline ECG denoisers and a pluggable denoiser interface.

Every denoiser is a callable ``denoise(signal, fs) -> signal`` exposed as a
small object with a ``.name``.  This is the extension point of the whole
pipeline: to evaluate *your* method (e.g. the DeepFilter deep model, a wavelet
scheme, an adaptive filter), wrap it in :class:`FunctionDenoiser` or subclass
:class:`Denoiser` and register it — the metrics and reporting are method-agnostic.

The built-in baselines are deliberately classical so they form an honest
reference floor:

- ``identity``      : passthrough (lower bound / sanity check).
- ``bandpass_iir``  : Butterworth 0.5-40 Hz (removes BW + high-freq MA).
- ``bandpass_fir``  : linear-phase FIR band-pass (no phase distortion of QRS).
- ``notch``         : IIR notch at the mains frequency (targets PLI).
- ``bandpass_notch``: band-pass followed by mains notch (BW+MA+PLI).
- ``median_baseline``: two-stage median filter baseline removal (BW).
- ``wavelet``       : discrete wavelet soft-thresholding (broadband, morphology
                      friendly) -- requires PyWavelets.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from scipy import signal as sp_signal


class Denoiser:
    """Base class for denoisers.  Subclasses implement :meth:`__call__`."""

    name: str = "denoiser"

    def __call__(self, x: np.ndarray, fs: float) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class FunctionDenoiser(Denoiser):
    """Adapt any ``f(signal, fs) -> signal`` callable into a Denoiser."""

    def __init__(self, name: str, fn: Callable[[np.ndarray, float], np.ndarray]):
        self.name = name
        self._fn = fn

    def __call__(self, x: np.ndarray, fs: float) -> np.ndarray:
        return np.asarray(self._fn(np.asarray(x, dtype=float), fs), dtype=float)


# --------------------------------------------------------------------------- #
# Baseline implementations
# --------------------------------------------------------------------------- #
def _identity(x: np.ndarray, fs: float) -> np.ndarray:
    return x.copy()


def _bandpass_iir(x: np.ndarray, fs: float, low: float = 0.5,
                  high: float = 40.0) -> np.ndarray:
    ny = fs / 2.0
    high = min(high, ny * 0.99)
    b, a = sp_signal.butter(4, [low / ny, high / ny], btype="band")
    return sp_signal.filtfilt(b, a, x)


def _bandpass_fir(x: np.ndarray, fs: float, low: float = 0.5,
                  high: float = 40.0) -> np.ndarray:
    ny = fs / 2.0
    high = min(high, ny * 0.99)
    numtaps = int(fs) | 1  # odd length, ~1 s
    numtaps = max(31, min(numtaps, len(x) // 3 | 1))
    taps = sp_signal.firwin(numtaps, [low, high], pass_zero=False, fs=fs)
    return sp_signal.filtfilt(taps, [1.0], x)


def _notch(x: np.ndarray, fs: float, mains_hz: float = 50.0,
           q: float = 30.0) -> np.ndarray:
    ny = fs / 2.0
    out = x
    # Notch the fundamental and harmonics that sit below Nyquist.
    for k in (1, 2, 3):
        f = mains_hz * k
        if f >= ny * 0.98:
            break
        b, a = sp_signal.iirnotch(f / ny, q)
        out = sp_signal.filtfilt(b, a, out)
    return out


def _bandpass_notch(x: np.ndarray, fs: float, mains_hz: float = 50.0) -> np.ndarray:
    return _notch(_bandpass_iir(x, fs), fs, mains_hz=mains_hz)


def _median_baseline(x: np.ndarray, fs: float) -> np.ndarray:
    """Remove baseline wander via two cascaded median filters (200 ms + 600 ms)."""
    def _odd(k: int) -> int:
        return k + 1 if k % 2 == 0 else k
    k1 = _odd(max(3, int(0.2 * fs)))
    k2 = _odd(max(3, int(0.6 * fs)))
    baseline = sp_signal.medfilt(sp_signal.medfilt(x, k1), k2)
    return x - baseline


def _wavelet(x: np.ndarray, fs: float, wavelet: str = "sym8",
             level: int | None = None) -> np.ndarray:
    """Soft-threshold wavelet denoising (universal threshold, MAD noise est.)."""
    try:
        import pywt  # noqa: WPS433 (optional dependency)
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyWavelets is required for the wavelet denoiser") from exc

    if level is None:
        level = min(6, pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len))
        level = max(1, level)
    coeffs = pywt.wavedec(x, wavelet, level=level)
    # Noise sigma from the finest detail band (robust MAD estimator).
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs[-1]) else 0.0
    thr = sigma * np.sqrt(2 * np.log(len(x))) if sigma > 0 else 0.0
    new = [coeffs[0]] + [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
    rec = pywt.waverec(new, wavelet)
    return rec[: len(x)]


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def build_default_denoisers(mains_hz: float = 50.0) -> Dict[str, Denoiser]:
    """Return the default set of baseline denoisers keyed by name."""
    denoisers: Dict[str, Denoiser] = {
        "identity": FunctionDenoiser("identity", _identity),
        "bandpass_iir": FunctionDenoiser("bandpass_iir", _bandpass_iir),
        "bandpass_fir": FunctionDenoiser("bandpass_fir", _bandpass_fir),
        "notch": FunctionDenoiser(
            "notch", lambda x, fs: _notch(x, fs, mains_hz=mains_hz)),
        "bandpass_notch": FunctionDenoiser(
            "bandpass_notch", lambda x, fs: _bandpass_notch(x, fs, mains_hz=mains_hz)),
        "median_baseline": FunctionDenoiser("median_baseline", _median_baseline),
    }
    try:
        import pywt  # noqa: F401
        denoisers["wavelet"] = FunctionDenoiser("wavelet", _wavelet)
    except ImportError:  # pragma: no cover
        pass
    return denoisers
