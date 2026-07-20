import numpy as np

from morphpreserve.metrics import fidelity as F
from morphpreserve.metrics.morphology import match_peaks, morphology_metrics
from morphpreserve.synthetic import generate_ecg


def test_fidelity_identical_signals():
    y = np.random.default_rng(0).standard_normal(500)
    assert F.ssd(y, y) == 0.0
    assert F.mad(y, y) == 0.0
    assert F.prd(y, y) == 0.0
    assert abs(F.cosine_similarity(y, y) - 1.0) < 1e-9
    assert abs(F.correlation(y, y) - 1.0) < 1e-9
    assert F.rmse(y, y) == 0.0


def test_prd_matches_reference_formula():
    rng = np.random.default_rng(1)
    y = rng.standard_normal(300)
    y_hat = y + 0.1 * rng.standard_normal(300)
    num = np.sum((y_hat - y) ** 2)
    den = np.sum((y_hat - np.mean(y)) ** 2)
    expected = np.sqrt(num / den) * 100.0
    assert abs(F.prd(y, y_hat) - expected) < 1e-9


def test_snr_improvement_positive_when_denoised_closer():
    rng = np.random.default_rng(2)
    clean = np.sin(np.linspace(0, 20 * np.pi, 1000))
    noise = 0.5 * rng.standard_normal(1000)
    noisy = clean + noise
    denoised = clean + 0.1 * rng.standard_normal(1000)  # closer to clean
    imp = F.snr_improvement(clean, noisy, denoised)
    assert imp > 0


def test_match_peaks_basic():
    ref = np.array([100, 200, 300, 400])
    test = np.array([102, 199, 405])  # 300 missed, all within tol except none extra
    tp, fp, fn, pairs = match_peaks(ref, test, tol=10)
    assert tp == 3 and fn == 1 and fp == 0


def test_morphology_perfect_on_identical():
    rec = generate_ecg(duration_s=20.0, fs=360.0, seed=3)
    m = morphology_metrics(rec.signal, rec.signal, rec.fs, ref_rpeaks=rec.r_peaks)
    assert m["rpeak_f1"] == 1.0
    assert m["qrs_xcorr"] > 0.999
    assert m["beat_xcorr"] > 0.999
    assert abs(m["r_amp_preservation"] - 1.0) < 1e-6
    assert m["rr_interval_error_ms"] < 1e-6


def test_morphology_degrades_with_distortion():
    rec = generate_ecg(duration_s=20.0, fs=360.0, seed=4)
    rng = np.random.default_rng(5)
    distorted = rec.signal + 0.3 * rng.standard_normal(len(rec.signal))
    m = morphology_metrics(rec.signal, distorted, rec.fs, ref_rpeaks=rec.r_peaks)
    # Correlation should drop below the perfect-case threshold.
    assert m["qrs_xcorr"] < 0.999
