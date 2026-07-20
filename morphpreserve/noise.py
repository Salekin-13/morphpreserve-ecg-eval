"""Noise models for ECG corruption and SNR-controlled mixing.

Two ways to obtain noise are supported:

1. **Synthetic generators** (this module) that reproduce the spectral / temporal
   character of each real corruption source.  These have no external
   dependencies, so the whole pipeline can run and be validated offline.

2. **Real NSTDB noise** (:func:`load_nstdb_noise`) pulled from PhysioNet's Noise
   Stress Test Database via ``wfdb`` when network access is available.  This is
   the same noise DeepFilter and most of the literature use.

Corruption sources modelled:

- ``bw``  baseline wander      : very-low-frequency drift (< ~0.7 Hz).
- ``em``  electrode motion     : low-frequency colored noise + sparse contact
                                 transients (the hardest artefact to remove).
- ``ma``  muscle artefact/EMG  : broadband band-limited noise (~15-100 Hz).
- ``pli`` power-line interfer. : 50/60 Hz sinusoid + harmonics, amplitude-modulated.

Mixing uses the standard definition ``SNR = 10*log10(P_signal / P_noise)`` and
scales the noise to hit a requested input SNR, which is more controllable and
reproducible than the max-ratio ``alpha`` scheme.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import signal as sp_signal


# --------------------------------------------------------------------------- #
# Individual synthetic noise generators.  Each returns a zero-mean, unit-RMS
# waveform of length ``n``; the caller rescales for the requested SNR.
# --------------------------------------------------------------------------- #
def _normalize(x: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(x**2))
    if rms < 1e-12:
        return x
    return x / rms


def baseline_wander(n: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency drift: a few sinusoids in the 0.05-0.7 Hz band."""
    t = np.arange(n) / fs
    x = np.zeros(n)
    for _ in range(4):
        f = rng.uniform(0.05, 0.7)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.5, 1.5)
        x += amp * np.sin(2 * np.pi * f * t + phase)
    # Respiration-like slow amplitude modulation.
    resp = 1.0 + 0.3 * np.sin(2 * np.pi * rng.uniform(0.2, 0.35) * t)
    return _normalize(x * resp)


def _bandlimited_noise(n: int, fs: float, low: float, high: float,
                       rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    ny = fs / 2.0
    low = max(low, 0.01)
    high = min(high, ny * 0.99)
    if low >= high:
        return _normalize(white)
    b, a = sp_signal.butter(4, [low / ny, high / ny], btype="band")
    return _normalize(sp_signal.filtfilt(b, a, white))


def muscle_artifact(n: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    """EMG-like broadband noise (~15-100 Hz), bursty in amplitude."""
    base = _bandlimited_noise(n, fs, 15.0, min(100.0, fs / 2 * 0.9), rng)
    # Random bursts of muscle activity.
    env = np.ones(n)
    n_bursts = max(1, int(n / fs / 3))
    for _ in range(n_bursts):
        start = rng.integers(0, n)
        length = int(rng.uniform(0.2, 1.0) * fs)
        end = min(n, start + length)
        env[start:end] *= rng.uniform(1.5, 3.0)
    return _normalize(base * env)


def electrode_motion(n: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    """Electrode-motion artefact: low-frequency colored noise plus sparse,
    high-amplitude contact transients (step-like)."""
    colored = _bandlimited_noise(n, fs, 0.5, 15.0, rng)
    transients = np.zeros(n)
    n_events = max(1, int(n / fs / 4))
    for _ in range(n_events):
        loc = rng.integers(0, n)
        amp = rng.uniform(2.0, 5.0) * rng.choice([-1, 1])
        width = max(1, int(rng.uniform(0.05, 0.25) * fs))
        end = min(n, loc + width)
        # Exponential decay transient (electrode re-settling).
        tau = np.arange(end - loc) / (0.1 * fs + 1e-9)
        transients[loc:end] += amp * np.exp(-tau)
    return _normalize(colored + transients)


def powerline_interference(n: int, fs: float, rng: np.random.Generator,
                           mains_hz: float = 50.0) -> np.ndarray:
    """Mains hum: fundamental + 2nd/3rd harmonics with slow amplitude drift."""
    t = np.arange(n) / fs
    x = np.zeros(n)
    for k, amp in ((1, 1.0), (2, 0.25), (3, 0.12)):
        f = mains_hz * k
        if f >= fs / 2:  # avoid aliasing above Nyquist
            continue
        phase = rng.uniform(0, 2 * np.pi)
        x += amp * np.sin(2 * np.pi * f * t + phase)
    # Slow amplitude modulation of the mains coupling.
    am = 1.0 + 0.2 * np.sin(2 * np.pi * rng.uniform(0.1, 0.5) * t)
    return _normalize(x * am)


_SYNTH = {
    "bw": baseline_wander,
    "em": electrode_motion,
    "ma": muscle_artifact,
    "pli": powerline_interference,
}


def make_noise(noise_type: str, n: int, fs: float,
               rng: np.random.Generator, mains_hz: float = 50.0) -> np.ndarray:
    """Return a unit-RMS synthetic noise waveform of the requested type."""
    if noise_type not in _SYNTH:
        raise ValueError(f"unknown noise type {noise_type!r}")
    if noise_type == "pli":
        return powerline_interference(n, fs, rng, mains_hz=mains_hz)
    return _SYNTH[noise_type](n, fs, rng)


# --------------------------------------------------------------------------- #
# SNR-controlled mixing.
# --------------------------------------------------------------------------- #
def _power(x: np.ndarray) -> float:
    return float(np.mean(x**2))


def snr_db(clean: np.ndarray, noise: np.ndarray) -> float:
    """Signal-to-noise ratio (dB) of ``clean`` relative to ``noise``."""
    pn = _power(noise)
    if pn < 1e-20:
        return float("inf")
    return 10.0 * np.log10(_power(clean) / pn)


def add_noise_at_snr(clean: np.ndarray, noise: np.ndarray,
                     target_snr_db: float) -> tuple[np.ndarray, np.ndarray]:
    """Scale ``noise`` so that ``clean + noise`` has ``target_snr_db`` input SNR.

    Returns ``(noisy, scaled_noise)``.
    """
    ps = _power(clean)
    pn = _power(noise)
    if pn < 1e-20 or ps < 1e-20:
        return clean.copy(), np.zeros_like(clean)
    target_pn = ps / (10.0 ** (target_snr_db / 10.0))
    scale = np.sqrt(target_pn / pn)
    scaled = noise * scale
    return clean + scaled, scaled


def make_composite_noise(noise_types, n: int, fs: float,
                         rng: np.random.Generator, mains_hz: float = 50.0,
                         weights: Optional[dict] = None) -> np.ndarray:
    """Sum several unit-RMS noise types (optionally weighted) into one waveform.

    Useful for the realistic "everything at once" corruption a wearable sees.
    """
    total = np.zeros(n)
    for nt in noise_types:
        w = 1.0 if weights is None else float(weights.get(nt, 1.0))
        total += w * make_noise(nt, n, fs, rng, mains_hz=mains_hz)
    return _normalize(total)


# --------------------------------------------------------------------------- #
# Real NSTDB noise (optional, requires network + wfdb).
# --------------------------------------------------------------------------- #
def load_nstdb_noise(noise_type: str, fs_target: Optional[float] = None):
    """Load a real NSTDB noise channel ('bw', 'em', or 'ma') from PhysioNet.

    Requires ``wfdb`` and network access to physionet.org.  Returns a 1-D numpy
    array (unit-RMS normalised).  PLI is not an NSTDB record; use the synthetic
    generator for it.  Raises a clear error if unavailable so callers can fall
    back to synthetic noise.
    """
    if noise_type not in ("bw", "em", "ma"):
        raise ValueError("NSTDB provides only 'bw', 'em', 'ma'; use synthetic 'pli'")
    try:
        import wfdb  # noqa: WPS433 (optional dependency)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("wfdb is required to load NSTDB noise") from exc

    rec = wfdb.rdrecord(noise_type, pn_dir="nstdb")
    sig = np.asarray(rec.p_signal[:, 0], dtype=float)
    fs_src = float(rec.fs)
    if fs_target is not None and abs(fs_target - fs_src) > 1e-6:
        n_new = int(round(len(sig) * fs_target / fs_src))
        sig = sp_signal.resample(sig, n_new)
    sig = sig - np.mean(sig)
    return _normalize(sig)
