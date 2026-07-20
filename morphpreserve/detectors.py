"""QRS / fiducial detection used by the morphology-preservation metrics.

A lightweight Pan-Tompkins-style R-peak detector plus a best-effort per-beat
delineator (P, Q, S, T) implemented on numpy/scipy only.  The detector is not
meant to compete with clinical delineators; it needs only to be *consistent* so
that comparing its output on the clean reference vs. the denoised signal yields
a meaningful measure of morphological distortion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy import signal as sp_signal


def _bandpass(x: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    ny = fs / 2.0
    low = max(low, 0.01)
    high = min(high, ny * 0.99)
    b, a = sp_signal.butter(2, [low / ny, high / ny], btype="band")
    return sp_signal.filtfilt(b, a, x)


def detect_rpeaks(x: np.ndarray, fs: float) -> np.ndarray:
    """Detect R-peaks with a Pan-Tompkins-style pipeline.

    Steps: 5-15 Hz bandpass -> derivative -> square -> moving-window
    integration -> adaptive-threshold peak picking, then refine each detection
    to the local maximum of the (bandpassed) signal.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < int(0.5 * fs):
        return np.asarray([], dtype=int)

    filtered = _bandpass(x, fs, 5.0, 15.0)
    deriv = np.gradient(filtered)
    squared = deriv**2
    win = max(1, int(0.150 * fs))  # 150 ms integration window
    integrated = np.convolve(squared, np.ones(win) / win, mode="same")

    # Refractory period ~200 ms; adaptive prominence threshold.
    min_dist = int(0.2 * fs)
    thr = 0.3 * np.mean(integrated) + 0.2 * np.std(integrated)
    peaks, _ = sp_signal.find_peaks(integrated, distance=min_dist, height=thr)

    # Refine to the true R-peak (max |signal|) within a small search window.
    search = max(1, int(0.05 * fs))
    refined = []
    for p in peaks:
        a = max(0, p - search)
        b = min(len(x), p + search)
        local = np.argmax(np.abs(x[a:b] - np.mean(x[a:b])))
        refined.append(a + local)
    refined = np.unique(refined)
    return refined.astype(int)


@dataclass
class Delineation:
    """Per-record fiducial estimates (sample indices; -1 where undetected)."""

    r_peaks: np.ndarray
    q_points: np.ndarray
    s_points: np.ndarray
    p_peaks: np.ndarray
    t_peaks: np.ndarray

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "R": self.r_peaks, "Q": self.q_points, "S": self.s_points,
            "P": self.p_peaks, "T": self.t_peaks,
        }


def delineate(x: np.ndarray, fs: float, r_peaks: np.ndarray | None = None) -> Delineation:
    """Best-effort delineation of Q, S, P and T around each detected R-peak."""
    x = np.asarray(x, dtype=float)
    if r_peaks is None:
        r_peaks = detect_rpeaks(x, fs)

    def _samp(ms: float) -> int:
        return max(1, int(round(ms * fs / 1000.0)))

    q_pts, s_pts, p_pks, t_pks = [], [], [], []
    n = len(x)
    for r in r_peaks:
        # Q: minimum in the 50 ms before R.
        a = max(0, r - _samp(50)); q = a + int(np.argmin(x[a:r + 1])) if r > a else r
        # S: minimum in the 50 ms after R.
        b = min(n, r + _samp(50)); s = r + int(np.argmin(x[r:b])) if b > r else r
        # P: maximum in [R-250ms, R-80ms].
        pa = max(0, r - _samp(250)); pb = max(pa + 1, r - _samp(80))
        p = pa + int(np.argmax(x[pa:pb])) if pb > pa else -1
        # T: maximum in [R+120ms, R+400ms].
        ta = min(n - 1, r + _samp(120)); tb = min(n, r + _samp(400))
        t = ta + int(np.argmax(x[ta:tb])) if tb > ta else -1
        q_pts.append(q); s_pts.append(s); p_pks.append(p); t_pks.append(t)

    return Delineation(
        r_peaks=np.asarray(r_peaks, dtype=int),
        q_points=np.asarray(q_pts, dtype=int),
        s_points=np.asarray(s_pts, dtype=int),
        p_peaks=np.asarray(p_pks, dtype=int),
        t_peaks=np.asarray(t_pks, dtype=int),
    )
