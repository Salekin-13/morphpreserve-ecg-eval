import numpy as np

from morphpreserve.denoisers import build_default_denoisers, FunctionDenoiser
from morphpreserve.noise import make_noise, add_noise_at_snr
from morphpreserve.metrics.fidelity import snr_improvement
from morphpreserve.synthetic import generate_ecg


def test_all_denoisers_preserve_length_and_finite():
    rec = generate_ecg(duration_s=15.0, fs=360.0, seed=0)
    rng = np.random.default_rng(0)
    noisy = rec.signal + 0.2 * make_noise("ma", len(rec.signal), rec.fs, rng)
    for name, d in build_default_denoisers().items():
        out = d(noisy, rec.fs)
        assert len(out) == len(noisy), name
        assert np.all(np.isfinite(out)), name


def test_notch_removes_powerline():
    rec = generate_ecg(duration_s=20.0, fs=360.0, seed=1)
    rng = np.random.default_rng(1)
    pli = make_noise("pli", len(rec.signal), rec.fs, rng, mains_hz=50)
    noisy, _ = add_noise_at_snr(rec.signal, pli, target_snr_db=0.0)
    notch = build_default_denoisers(mains_hz=50)["notch"]
    imp = snr_improvement(rec.signal, noisy, notch(noisy, rec.fs))
    assert imp > 10  # notch should clean 50 Hz strongly


def test_median_removes_baseline_wander():
    rec = generate_ecg(duration_s=20.0, fs=360.0, seed=2)
    rng = np.random.default_rng(2)
    bw = make_noise("bw", len(rec.signal), rec.fs, rng)
    noisy, _ = add_noise_at_snr(rec.signal, bw, target_snr_db=-3.0)
    med = build_default_denoisers()["median_baseline"]
    imp = snr_improvement(rec.signal, noisy, med(noisy, rec.fs))
    assert imp > 3


def test_identity_is_zero_improvement():
    rec = generate_ecg(duration_s=10.0, fs=360.0, seed=3)
    rng = np.random.default_rng(3)
    noisy, _ = add_noise_at_snr(
        rec.signal, make_noise("ma", len(rec.signal), rec.fs, rng), 0.0)
    ident = build_default_denoisers()["identity"]
    assert abs(snr_improvement(rec.signal, noisy, ident(noisy, rec.fs))) < 1e-9


def test_custom_function_denoiser_pluggable():
    d = FunctionDenoiser("scale", lambda x, fs: x * 1.0)
    x = np.arange(100, dtype=float)
    assert np.allclose(d(x, 360.0), x)
    assert d.name == "scale"
