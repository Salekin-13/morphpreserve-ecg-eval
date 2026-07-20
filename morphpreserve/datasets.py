"""Dataset loaders that yield clean ECG :class:`~morphpreserve.synthetic.ECGRecord`s.

A "record source" is any iterable of ``ECGRecord`` objects.  The pipeline adds
noise itself, so a source only needs to provide *clean* reference signals (plus
R-peak annotations when available).

Supported sources
------------------
- :func:`synthetic_source`  -- fully offline; ground-truth fiducials included.
- :func:`nstdb_qt_source`   -- PhysioNet QT DB clean beats (needs wfdb + network).
- :func:`wear_source`       -- adapter for the WEAR / BioCAS-2015 wearable ECG
                               dataset (Jafari Lab, TAMU / UCSD, 488 Hz).

WEAR notes (verified against the BioCAS 2015 paper, DOI 10.1109/BioCAS.2015.7348384
and jafari.tamu.edu/wear): 260+ recordings of 90-210 s at **488 Hz**, wrist and
chest electrodes, 10 activities where activity 1 isolates baseline wander,
activities 2-8 induce motion artefact, and activities 9-10 capture power-line
interference at two distances.  The dataset ships as per-recording data files;
``wear_source`` reads whatever numeric column layout is present (CSV/TSV/whitespace)
so it adapts to the released format without hard-coding a schema that may differ
across mirrors.
"""

from __future__ import annotations

import glob
import os
from typing import Iterable, Iterator, List, Optional

import numpy as np

from .synthetic import ECGRecord, generate_ecg


def synthetic_source(n_records: int = 5, duration_s: float = 30.0,
                     fs: float = 360.0, seed: int = 1234) -> Iterator[ECGRecord]:
    """Yield ``n_records`` synthetic ECGs with varied HR (offline, ground truth)."""
    rng = np.random.default_rng(seed)
    for i in range(n_records):
        hr = float(rng.uniform(55, 95))
        rec = generate_ecg(duration_s=duration_s, fs=fs, heart_rate_bpm=hr,
                           hrv_std_bpm=float(rng.uniform(2, 6)), seed=int(rng.integers(1e9)))
        rec.name = f"synthetic_{i:02d}_hr{int(hr)}"
        yield rec


def nstdb_qt_source(records: Optional[List[str]] = None,
                    fs_target: Optional[float] = None,
                    max_records: int = 10) -> Iterator[ECGRecord]:
    """Yield clean beats from the PhysioNet QT Database (needs wfdb + network).

    QT DB provides clean ECG with beat annotations; DeepFilter uses it as the
    clean reference and NSTDB as the noise source.  R-peak annotations are taken
    from the reference annotator when present.
    """
    import wfdb  # optional dependency; raises clearly if missing

    if records is None:
        # A small default subset of QT DB record names.
        records = ["sel100", "sel103", "sel114", "sel116", "sel117"][:max_records]

    for name in records:
        rec = wfdb.rdrecord(name, pn_dir="qtdb")
        sig = np.asarray(rec.p_signal[:, 0], dtype=float)
        fs = float(rec.fs)
        try:
            ann = wfdb.rdann(name, "atr", pn_dir="qtdb")
            r_peaks = np.asarray(ann.sample, dtype=int)
        except Exception:
            r_peaks = np.asarray([], dtype=int)
        if fs_target is not None and abs(fs_target - fs) > 1e-6:
            from scipy.signal import resample
            n_new = int(round(len(sig) * fs_target / fs))
            factor = fs_target / fs
            sig = resample(sig, n_new)
            r_peaks = np.round(r_peaks * factor).astype(int)
            fs = fs_target
        yield ECGRecord(signal=sig - np.mean(sig), fs=fs,
                        r_peaks=r_peaks, name=f"qtdb_{name}")


def _read_numeric_columns(path: str) -> Optional[np.ndarray]:
    """Read a whitespace/comma/tab-delimited numeric file into a 2-D array."""
    for delim in (None, ",", "\t", ";"):
        try:
            arr = np.genfromtxt(path, delimiter=delim, comments="#")
        except Exception:
            continue
        arr = np.atleast_2d(arr)
        if arr.ndim == 1:
            arr = arr[:, None]
        if np.isfinite(arr).mean() > 0.5 and arr.size > 10:
            # Drop all-NaN columns (headers / labels).
            keep = np.isfinite(arr).mean(axis=0) > 0.5
            arr = arr[:, keep]
            if arr.size:
                return arr[np.isfinite(arr).all(axis=1)]
    return None


def wear_source(root: str, fs: float = 488.0, ecg_column: int = -1,
                pattern: str = "**/*", max_records: int = 50) -> Iterator[ECGRecord]:
    """Adapter for the WEAR / BioCAS-2015 wearable ECG dataset.

    Parameters
    ----------
    root:
        Directory containing the downloaded WEAR recordings.
    fs:
        Sampling rate (WEAR default 488 Hz).
    ecg_column:
        Which numeric column holds the ECG channel (``-1`` = last column).
    pattern:
        Glob (relative to ``root``) selecting recording files.

    R-peaks are not annotated in WEAR, so ``r_peaks`` is left empty and the
    morphology metrics detect them on the (clean/lightly-filtered) reference.
    For denoising evaluation on WEAR, treat activity-1 (clean-ish) segments as
    reference and the motion / PLI activities as the corrupted signals, or inject
    controlled synthetic noise onto the cleaner segments.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"WEAR root not found: {root}")
    files = sorted(
        p for p in glob.glob(os.path.join(root, pattern), recursive=True)
        if os.path.isfile(p) and not p.lower().endswith(
            (".pdf", ".md", ".txt", ".png", ".jpg"))
    )
    count = 0
    for path in files:
        arr = _read_numeric_columns(path)
        if arr is None:
            continue
        col = ecg_column if ecg_column >= 0 else arr.shape[1] - 1
        col = min(col, arr.shape[1] - 1)
        sig = arr[:, col].astype(float)
        if len(sig) < int(2 * fs):  # need at least a couple of seconds
            continue
        sig = sig - np.mean(sig)
        yield ECGRecord(signal=sig, fs=fs,
                        r_peaks=np.asarray([], dtype=int),
                        name=f"wear_{os.path.basename(path)}")
        count += 1
        if count >= max_records:
            break


def load_source(name: str, **kwargs) -> Iterable[ECGRecord]:
    """Dispatch a source by name: 'synthetic' | 'qtdb' | 'wear'."""
    if name == "synthetic":
        return synthetic_source(**kwargs)
    if name == "qtdb":
        return nstdb_qt_source(**kwargs)
    if name == "wear":
        return wear_source(**kwargs)
    raise ValueError(f"unknown source {name!r}")
