"""Morphology-preservation metrics.

Fidelity metrics (SSD/PRD/SNR) tell you how close two waveforms are *sample by
sample*, but a denoiser can score well on them while still smearing the P-wave,
clipping the R-peak, or shifting the T-wave — exactly the features a downstream
**preeclampsia prediction model** depends on.  Preeclampsia has documented ECG
correlates: QT / QTc prolongation and QT dispersion, P-wave duration changes,
altered T-wave morphology, PR changes and reduced heart-rate variability.

This module quantifies how well those clinically meaningful features survive
denoising by comparing fiducials/intervals detected on the clean reference with
those detected on the denoised signal:

- R-peak detection F1 (were beats preserved / no spurious beats?)
- RR-interval error (HRV preservation)
- QT / PR interval error
- P / R / T amplitude preservation ratios
- QRS-complex cross-correlation (shape of the depolarisation wave)
- whole-beat cross-correlation

Everything is computed relative to a *reference* delineation, so on synthetic
data (reference == ground truth) the metrics are directly validated.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..detectors import delineate, detect_rpeaks


def match_peaks(ref: np.ndarray, test: np.ndarray, tol: int):
    """Greedy nearest-neighbour matching of two peak-index sequences.

    Returns ``(tp, fp, fn, matched_pairs)`` where matched_pairs is a list of
    ``(ref_idx, test_idx)`` within ``tol`` samples.
    """
    ref = np.sort(np.asarray(ref, dtype=int))
    test = np.sort(np.asarray(test, dtype=int))
    used = np.zeros(len(test), dtype=bool)
    pairs = []
    for r in ref:
        if len(test) == 0:
            break
        d = np.abs(test - r)
        d[used] = tol + 1
        j = int(np.argmin(d)) if len(d) else -1
        if j >= 0 and d[j] <= tol:
            used[j] = True
            pairs.append((int(r), int(test[j])))
    tp = len(pairs)
    fn = len(ref) - tp
    fp = len(test) - tp
    return tp, fp, fn, pairs


def rpeak_detection_scores(ref_rpeaks: np.ndarray, test_rpeaks: np.ndarray,
                           tol: int) -> Dict[str, float]:
    tp, fp, fn, _ = match_peaks(ref_rpeaks, test_rpeaks, tol)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"rpeak_precision": precision, "rpeak_recall": recall, "rpeak_f1": f1}


def _interval_error(ref_pts_a, ref_pts_b, test_pts_a, test_pts_b, fs, pairs_idx):
    """Mean absolute error (ms) of an interval (b - a) over matched beats."""
    errs = []
    for i, j in pairs_idx:
        ra, rb = ref_pts_a[i], ref_pts_b[i]
        ta, tb = test_pts_a[j], test_pts_b[j]
        if min(ra, rb, ta, tb) < 0:
            continue
        ref_int = (rb - ra) / fs
        test_int = (tb - ta) / fs
        errs.append(abs(ref_int - test_int) * 1000.0)
    return float(np.mean(errs)) if errs else float("nan")


def _windowed_xcorr(ref: np.ndarray, test: np.ndarray, centers: np.ndarray,
                    half: int) -> float:
    """Mean Pearson correlation of windows centred on ``centers``."""
    cors = []
    n = len(ref)
    for c in centers:
        a = max(0, c - half); b = min(n, c + half)
        if b - a < 3:
            continue
        rw = ref[a:b]; tw = test[a:b]
        if np.std(rw) < 1e-12 or np.std(tw) < 1e-12:
            continue
        cors.append(np.corrcoef(rw, tw)[0, 1])
    return float(np.mean(cors)) if cors else float("nan")


def _amplitude_ratio(ref: np.ndarray, test: np.ndarray, idx_ref, idx_test, pairs_idx):
    """Median amplitude-preservation ratio (test/ref) over matched fiducials."""
    ratios = []
    for i, j in pairs_idx:
        ri, ti = idx_ref[i], idx_test[j]
        if ri < 0 or ti < 0:
            continue
        rv = ref[ri]
        if abs(rv) < 1e-9:
            continue
        ratios.append(test[ti] / rv)
    return float(np.median(ratios)) if ratios else float("nan")


def morphology_metrics(clean: np.ndarray, denoised: np.ndarray, fs: float,
                       ref_rpeaks: np.ndarray | None = None,
                       qrs_half: int | None = None,
                       beat_half: int | None = None,
                       match_tol: int | None = None) -> Dict[str, float]:
    """Full morphology-preservation bundle for one (clean, denoised) pair.

    ``ref_rpeaks`` may be supplied as ground truth (synthetic data); otherwise
    R-peaks are detected on the clean reference.
    """
    clean = np.asarray(clean, dtype=float)
    denoised = np.asarray(denoised, dtype=float)

    qrs_half = qrs_half if qrs_half is not None else int(0.06 * fs)
    beat_half = beat_half if beat_half is not None else int(0.25 * fs)
    match_tol = match_tol if match_tol is not None else int(0.05 * fs)

    ref_delin = delineate(clean, fs, r_peaks=ref_rpeaks)
    test_rpeaks = detect_rpeaks(denoised, fs)
    test_delin = delineate(denoised, fs, r_peaks=test_rpeaks)

    scores = rpeak_detection_scores(ref_delin.r_peaks, test_delin.r_peaks, match_tol)

    # Matched beat pairs (indices into each delineation's arrays).
    _, _, _, pairs = match_peaks(ref_delin.r_peaks, test_delin.r_peaks, match_tol)
    ref_pos = {int(v): i for i, v in enumerate(ref_delin.r_peaks)}
    test_pos = {int(v): i for i, v in enumerate(test_delin.r_peaks)}
    pairs_idx = [(ref_pos[r], test_pos[t]) for r, t in pairs]

    # RR-interval error (HRV preservation).
    if len(ref_delin.r_peaks) > 1 and len(test_delin.r_peaks) > 1:
        rr_ref = np.diff(ref_delin.r_peaks) / fs
        rr_test = np.diff(test_delin.r_peaks) / fs
        m = min(len(rr_ref), len(rr_test))
        rr_err = float(np.mean(np.abs(rr_ref[:m] - rr_test[:m])) * 1000.0)
    else:
        rr_err = float("nan")

    # QT interval (Q onset -> T peak) and PR interval (P peak -> R peak) errors.
    qt_err = _interval_error(ref_delin.q_points, ref_delin.t_peaks,
                             test_delin.q_points, test_delin.t_peaks, fs, pairs_idx)
    pr_err = _interval_error(ref_delin.p_peaks, ref_delin.r_peaks,
                             test_delin.p_peaks, test_delin.r_peaks, fs, pairs_idx)

    # Amplitude preservation ratios (ideal == 1.0).
    r_amp = _amplitude_ratio(clean, denoised, ref_delin.r_peaks, test_delin.r_peaks, pairs_idx)
    p_amp = _amplitude_ratio(clean, denoised, ref_delin.p_peaks, test_delin.p_peaks, pairs_idx)
    t_amp = _amplitude_ratio(clean, denoised, ref_delin.t_peaks, test_delin.t_peaks, pairs_idx)

    # Shape correlations centred on the reference R-peaks.
    qrs_xcorr = _windowed_xcorr(clean, denoised, ref_delin.r_peaks, qrs_half)
    beat_xcorr = _windowed_xcorr(clean, denoised, ref_delin.r_peaks, beat_half)

    return {
        **scores,
        "rr_interval_error_ms": rr_err,
        "qt_interval_error_ms": qt_err,
        "pr_interval_error_ms": pr_err,
        "r_amp_preservation": r_amp,
        "p_amp_preservation": p_amp,
        "t_amp_preservation": t_amp,
        "qrs_xcorr": qrs_xcorr,
        "beat_xcorr": beat_xcorr,
    }
