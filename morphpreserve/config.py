"""Configuration objects for the ECG denoising evaluation pipeline.

The evaluation is driven by an :class:`EvalConfig` dataclass so that a whole
experiment (sampling rate, the noise types to inject, the SNR sweep, which
metrics to compute, random seed) is described by a single, serialisable object.
This keeps runs reproducible and easy to log.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List


# Canonical noise labels used throughout the package.  These mirror the
# real-world corruption sources documented for wearable ECG (WEAR / BioCAS 2015)
# and the PhysioNet Noise Stress Test Database (NSTDB).
NOISE_BW = "bw"    # baseline wander (respiration, electrode half-cell drift)
NOISE_EM = "em"    # electrode motion (contact transients, worst-case artefact)
NOISE_MA = "ma"    # muscle artefact / EMG
NOISE_PLI = "pli"  # power-line interference (50/60 Hz + harmonics)

ALL_NOISE_TYPES = (NOISE_BW, NOISE_EM, NOISE_MA, NOISE_PLI)


@dataclass
class EvalConfig:
    """Top-level configuration for an evaluation run.

    Attributes
    ----------
    fs:
        Sampling frequency in Hz.  Defaults to 360 Hz (MIT-BIH / NSTDB).  The
        WEAR wearable dataset is sampled at 488 Hz; set ``fs=488`` for it.
    powerline_hz:
        Mains frequency for the PLI model and notch denoisers (50 Hz in
        most of the world, 60 Hz in North America).
    noise_types:
        Which corruption sources to inject and evaluate.
    snr_levels_db:
        Input SNR sweep (dB).  Each clean record is corrupted at every level.
    seed:
        Master RNG seed for reproducibility.
    qrs_window_ms:
        Half-width (ms) of the window extracted around each R-peak for the
        QRS-correlation morphology metric.
    beat_window_ms:
        Half-width (ms) of the window used for whole-beat correlation.
    rpeak_match_tol_ms:
        Tolerance (ms) for matching detected R-peaks between the reference and
        the denoised signal when scoring detection F1 / interval errors.
    """

    fs: int = 360
    powerline_hz: int = 50
    noise_types: List[str] = field(default_factory=lambda: list(ALL_NOISE_TYPES))
    snr_levels_db: List[float] = field(default_factory=lambda: [-6.0, 0.0, 6.0, 12.0])
    seed: int = 1234

    # Morphology-metric windows (clinically motivated defaults).
    qrs_window_ms: float = 60.0
    beat_window_ms: float = 250.0
    rpeak_match_tol_ms: float = 50.0

    def __post_init__(self) -> None:
        if self.fs <= 0:
            raise ValueError("fs must be positive")
        unknown = set(self.noise_types) - set(ALL_NOISE_TYPES)
        if unknown:
            raise ValueError(
                f"unknown noise types {sorted(unknown)}; "
                f"valid options are {ALL_NOISE_TYPES}"
            )
        if self.powerline_hz not in (50, 60):
            # Not fatal, but almost always a mistake.
            raise ValueError("powerline_hz should be 50 or 60")

    # --- convenience ------------------------------------------------------
    def ms_to_samples(self, ms: float) -> int:
        """Convert a duration in milliseconds to an integer number of samples."""
        return max(1, int(round(ms * self.fs / 1000.0)))

    def to_dict(self) -> dict:
        return asdict(self)
