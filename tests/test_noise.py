import numpy as np
import pytest

from morphpreserve import noise as N


@pytest.mark.parametrize("nt", ["bw", "em", "ma", "pli"])
def test_noise_unit_rms_and_length(nt):
    rng = np.random.default_rng(0)
    x = N.make_noise(nt, 3600, 360.0, rng, mains_hz=50)
    assert x.shape == (3600,)
    assert abs(np.sqrt(np.mean(x**2)) - 1.0) < 1e-6


def test_add_noise_hits_target_snr():
    rng = np.random.default_rng(1)
    clean = np.sin(2 * np.pi * 1.0 * np.arange(3600) / 360.0)
    noise = N.make_noise("ma", 3600, 360.0, rng)
    for target in (-6.0, 0.0, 6.0, 12.0):
        noisy, scaled = N.add_noise_at_snr(clean, noise, target)
        measured = N.snr_db(clean, scaled)
        assert abs(measured - target) < 1e-6
        assert np.allclose(noisy, clean + scaled)


def test_pli_energy_at_mains_frequency():
    rng = np.random.default_rng(2)
    fs = 360.0
    x = N.make_noise("pli", 3600, fs, rng, mains_hz=50)
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    mag = np.abs(np.fft.rfft(x))
    peak_f = freqs[np.argmax(mag)]
    assert abs(peak_f - 50.0) < 1.0  # dominant energy at the mains fundamental


def test_baseline_wander_is_low_frequency():
    rng = np.random.default_rng(3)
    fs = 360.0
    x = N.make_noise("bw", 3600, fs, rng)
    # Use a window to suppress spectral leakage, then check that essentially all
    # energy sits below ~2 Hz (the baseline-wander band).
    win = np.hanning(len(x))
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    power = np.abs(np.fft.rfft(x * win)) ** 2
    frac_below = power[freqs < 2.0].sum() / power.sum()
    assert frac_below > 0.95


def test_composite_noise_unit_rms():
    rng = np.random.default_rng(4)
    x = N.make_composite_noise(["bw", "ma", "pli"], 3600, 360.0, rng)
    assert abs(np.sqrt(np.mean(x**2)) - 1.0) < 1e-6
