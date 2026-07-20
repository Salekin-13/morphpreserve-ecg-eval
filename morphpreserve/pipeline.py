"""Evaluation pipeline orchestration.

Ties the pieces together:

    clean record  --(inject noise at SNR)-->  noisy  --(denoiser)-->  denoised
                                    |                        |
                                    +----> fidelity + morphology metrics <----+

For every (record, noise_type, snr_level, denoiser) combination it produces one
:class:`EvalResult` row.  Results are returned as a list of dicts that
:mod:`morphpreserve.report` aggregates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np

from .config import EvalConfig
from .denoisers import Denoiser, build_default_denoisers
from .metrics.fidelity import all_fidelity_metrics
from .metrics.morphology import morphology_metrics
from .noise import add_noise_at_snr, load_nstdb_noise, make_noise
from .synthetic import ECGRecord


@dataclass
class EvalResult:
    record: str
    denoiser: str
    noise_type: str
    snr_db: float
    metrics: Dict[str, float] = field(default_factory=dict)

    def as_row(self) -> Dict[str, object]:
        row: Dict[str, object] = {
            "record": self.record,
            "denoiser": self.denoiser,
            "noise_type": self.noise_type,
            "target_snr_db": self.snr_db,
        }
        row.update(self.metrics)
        return row


class Evaluator:
    """Run a denoising evaluation over records, noise types and SNR levels."""

    def __init__(self, config: Optional[EvalConfig] = None,
                 denoisers: Optional[Dict[str, Denoiser]] = None,
                 use_real_noise: bool = False):
        self.config = config or EvalConfig()
        self.denoisers = denoisers or build_default_denoisers(
            mains_hz=self.config.powerline_hz)
        self.use_real_noise = use_real_noise
        self._rng = np.random.default_rng(self.config.seed)
        self._noise_cache: Dict[str, np.ndarray] = {}

    # -- noise handling ----------------------------------------------------
    def _get_noise(self, noise_type: str, n: int, fs: float) -> np.ndarray:
        """Return a unit-RMS noise waveform of length ``n``.

        Uses real NSTDB noise when requested and available, otherwise the
        synthetic generator.  Real records are cached and tiled/cropped to n.
        """
        if self.use_real_noise and noise_type in ("bw", "em", "ma"):
            key = f"{noise_type}@{fs}"
            if key not in self._noise_cache:
                try:
                    self._noise_cache[key] = load_nstdb_noise(noise_type, fs_target=fs)
                except Exception:
                    self._noise_cache[key] = None  # fall back below
            base = self._noise_cache[key]
            if base is not None and len(base) > 0:
                start = int(self._rng.integers(0, max(1, len(base))))
                tiled = np.roll(base, -start)
                if len(tiled) < n:
                    reps = int(np.ceil(n / len(tiled)))
                    tiled = np.tile(tiled, reps)
                seg = tiled[:n]
                rms = np.sqrt(np.mean(seg**2))
                return seg / rms if rms > 1e-12 else seg
        return make_noise(noise_type, n, fs, self._rng,
                          mains_hz=self.config.powerline_hz)

    # -- single evaluation -------------------------------------------------
    def evaluate_one(self, record: ECGRecord, denoiser: Denoiser,
                     noise_type: str, target_snr: float) -> EvalResult:
        clean = np.asarray(record.signal, dtype=float)
        fs = record.fs
        noise = self._get_noise(noise_type, len(clean), fs)
        noisy, _ = add_noise_at_snr(clean, noise, target_snr)
        denoised = denoiser(noisy, fs)
        if len(denoised) != len(clean):  # guard against length drift
            m = min(len(denoised), len(clean))
            clean_c, noisy_c, denoised_c = clean[:m], noisy[:m], denoised[:m]
        else:
            clean_c, noisy_c, denoised_c = clean, noisy, denoised

        metrics = all_fidelity_metrics(clean_c, noisy_c, denoised_c)
        ref_r = record.r_peaks if record.r_peaks is not None and len(record.r_peaks) else None
        metrics.update(morphology_metrics(
            clean_c, denoised_c, fs,
            ref_rpeaks=ref_r,
            qrs_half=self.config.ms_to_samples(self.config.qrs_window_ms),
            beat_half=self.config.ms_to_samples(self.config.beat_window_ms),
            match_tol=self.config.ms_to_samples(self.config.rpeak_match_tol_ms),
        ))
        return EvalResult(record=record.name, denoiser=denoiser.name,
                          noise_type=noise_type, snr_db=target_snr, metrics=metrics)

    # -- full sweep --------------------------------------------------------
    def run(self, records: Iterable[ECGRecord]) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for record in records:
            for noise_type in self.config.noise_types:
                for snr in self.config.snr_levels_db:
                    for denoiser in self.denoisers.values():
                        res = self.evaluate_one(record, denoiser, noise_type, snr)
                        rows.append(res.as_row())
        return rows
