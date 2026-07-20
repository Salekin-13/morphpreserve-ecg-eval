"""Metric sub-package: signal fidelity and morphology preservation."""

from .fidelity import (
    ssd, mad, prd, cosine_similarity, rmse, correlation,
    snr_db, snr_improvement, all_fidelity_metrics,
)
from .morphology import (
    morphology_metrics, rpeak_detection_scores, match_peaks,
)

# "Higher is better" for these keys; everything else is "lower is better".
HIGHER_IS_BETTER = {
    "cosine_sim", "correlation", "snr_in_db", "snr_out_db",
    "snr_improvement_db", "rpeak_precision", "rpeak_recall", "rpeak_f1",
    "qrs_xcorr", "beat_xcorr",
}

# Amplitude-preservation ratios are best when closest to 1.0.
BEST_AT_ONE = {"r_amp_preservation", "p_amp_preservation", "t_amp_preservation"}

__all__ = [
    "ssd", "mad", "prd", "cosine_similarity", "rmse", "correlation",
    "snr_db", "snr_improvement", "all_fidelity_metrics",
    "morphology_metrics", "rpeak_detection_scores", "match_peaks",
    "HIGHER_IS_BETTER", "BEST_AT_ONE",
]
