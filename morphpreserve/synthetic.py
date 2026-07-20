"""Synthetic ECG generator with ground-truth fiducial annotations.

Real PhysioNet / WEAR downloads are not always available (offline CI, blocked
egress), and even when they are, delineation ground truth is limited.  This
module generates physiologically plausible ECG using a sum-of-Gaussians PQRST
model (a simplified, discrete-time cousin of McSharry's ECGSYN) and, crucially,
returns the exact sample locations of every P, Q, R, S and T feature.

Those ground-truth fiducials are what let us *validate the morphology metrics
themselves*: on synthetic data the "clean reference" is a known template, so we
can check that the R-peak detector, RR/QT/PR estimators and correlation scores
behave as expected before trusting them on real recordings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


# Wave parameters relative to the R-peak (t = 0), in seconds and mV.
# (center_offset_s, amplitude_mV, width_s)  -- width is the Gaussian sigma.
_WAVE_TEMPLATE = {
    "P": (-0.190, 0.12, 0.028),
    "Q": (-0.028, -0.14, 0.0075),
    "R": (0.000, 1.00, 0.0090),
    "S": (0.028, -0.22, 0.0085),
    "T": (0.290, 0.30, 0.045),
}


@dataclass
class ECGRecord:
    """A synthetic ECG record plus its ground-truth annotations."""

    signal: np.ndarray               # clean ECG, shape (N,)
    fs: float                        # sampling rate (Hz)
    r_peaks: np.ndarray              # sample indices of R-peaks
    fiducials: Dict[str, np.ndarray] = field(default_factory=dict)  # wave->indices
    name: str = "synthetic"

    @property
    def duration_s(self) -> float:
        return len(self.signal) / self.fs


def _gaussian(t: np.ndarray, center: float, amp: float, sigma: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((t - center) / sigma) ** 2)


def generate_ecg(
    duration_s: float = 30.0,
    fs: float = 360.0,
    heart_rate_bpm: float = 72.0,
    hrv_std_bpm: float = 3.0,
    amplitude_jitter: float = 0.05,
    seed: int | None = 0,
) -> ECGRecord:
    """Generate a clean synthetic ECG record with fiducial ground truth.

    Parameters
    ----------
    duration_s:
        Length of the record in seconds.
    fs:
        Sampling frequency (Hz).
    heart_rate_bpm:
        Mean heart rate.
    hrv_std_bpm:
        Standard deviation of the beat-to-beat heart rate (creates realistic
        RR-interval variability / HRV).
    amplitude_jitter:
        Fractional beat-to-beat amplitude variation.
    seed:
        RNG seed (``None`` for nondeterministic).
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    t = np.arange(n) / fs
    signal = np.zeros(n, dtype=float)

    fiducials: Dict[str, List[int]] = {k: [] for k in _WAVE_TEMPLATE}
    r_peaks: List[int] = []

    # Lay down R-peak times using a jittered RR sequence.
    mean_rr = 60.0 / heart_rate_bpm
    rr_std = mean_rr * (hrv_std_bpm / heart_rate_bpm)

    # Start a little after t=0 so the first P-wave fits in the window.
    r_time = 0.5
    while r_time < duration_s - 0.5:
        beat_scale = 1.0 + amplitude_jitter * rng.standard_normal()
        beat_scale = max(0.5, beat_scale)
        for wave, (offset, amp, sigma) in _WAVE_TEMPLATE.items():
            center = r_time + offset
            signal += _gaussian(t, center, amp * beat_scale, sigma)
            idx = int(round(center * fs))
            if 0 <= idx < n:
                fiducials[wave].append(idx)
        idx_r = int(round(r_time * fs))
        if 0 <= idx_r < n:
            r_peaks.append(idx_r)

        rr = mean_rr + rr_std * rng.standard_normal()
        rr = float(np.clip(rr, 0.4, 1.5))  # 40-150 bpm physiological bound
        r_time += rr

    # Snap ground-truth R-peaks to the local maximum of the *composed* signal:
    # neighbouring Q/S Gaussians shift the true peak by up to a sample, so this
    # keeps the annotation self-consistent with the waveform it labels.
    r_arr = np.asarray(r_peaks, dtype=int)
    win = max(1, int(round(0.020 * fs)))  # +-20 ms search
    for i, r in enumerate(r_arr):
        a, b = max(0, r - win), min(n, r + win + 1)
        r_arr[i] = a + int(np.argmax(signal[a:b]))
    r_peaks = list(r_arr)

    return ECGRecord(
        signal=signal,
        fs=fs,
        r_peaks=np.asarray(r_peaks, dtype=int),
        fiducials={k: np.asarray(v, dtype=int) for k, v in fiducials.items()},
        name="synthetic",
    )
