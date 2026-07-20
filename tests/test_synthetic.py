import numpy as np

from morphpreserve.synthetic import generate_ecg
from morphpreserve.detectors import detect_rpeaks


def test_generate_shape_and_fs():
    rec = generate_ecg(duration_s=10.0, fs=360.0, heart_rate_bpm=60, seed=0)
    assert rec.signal.shape == (3600,)
    assert rec.fs == 360.0
    assert abs(rec.duration_s - 10.0) < 1e-6


def test_rpeak_count_matches_heart_rate():
    # 60 bpm over 20 s ~ 20 beats (allow small border effects).
    rec = generate_ecg(duration_s=20.0, fs=360.0, heart_rate_bpm=60,
                       hrv_std_bpm=0.0, seed=1)
    assert 17 <= len(rec.r_peaks) <= 21


def test_ground_truth_rpeaks_are_signal_maxima():
    rec = generate_ecg(duration_s=15.0, fs=360.0, seed=2)
    for r in rec.r_peaks:
        a, b = max(0, r - 5), min(len(rec.signal), r + 6)
        # R-peak should be the local maximum within +-5 samples.
        assert rec.signal[r] == np.max(rec.signal[a:b])


def test_detector_recovers_ground_truth():
    rec = generate_ecg(duration_s=20.0, fs=360.0, heart_rate_bpm=72, seed=3)
    det = detect_rpeaks(rec.signal, rec.fs)
    # Detected count close to ground truth on clean signal.
    assert abs(len(det) - len(rec.r_peaks)) <= 2
