"""Noise reduction: the estimators must actually raise the SNR."""

from __future__ import annotations

import numpy as np
import pytest

from speechdsp.enhance import log_mmse, spectral_subtraction

from .conftest import segmental_snr_db

SR = 16000
LEAD_SILENCE_S = 0.25


def make_noisy(sr: int = SR, noise_std: float = 0.1, seed: int = 11):
    """Tone preceded by noise-only silence, so the noise estimate is valid."""
    gen = np.random.default_rng(seed)
    n_lead = round(float(LEAD_SILENCE_S * sr))
    t = np.arange(sr, dtype=np.float64) / sr
    tone = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    clean = np.concatenate([np.zeros(n_lead), tone])
    noisy = clean + gen.normal(0.0, noise_std, clean.size)
    return clean, noisy, n_lead


@pytest.mark.parametrize("noise_std", [0.05, 0.1, 0.2])
def test_log_mmse_improves_the_snr(noise_std):
    clean, noisy, lead = make_noisy(noise_std=noise_std)
    enhanced = log_mmse(noisy, SR)
    before = segmental_snr_db(clean, noisy, start=lead)
    after = segmental_snr_db(clean, enhanced, start=lead)
    assert after > before + 3.0


def test_log_mmse_attenuates_the_noise_only_region():
    _, noisy, lead = make_noisy()
    enhanced = log_mmse(noisy, SR)
    noisy_power = float(np.mean(noisy[: lead // 2] ** 2))
    clean_power = float(np.mean(enhanced[: lead // 2] ** 2))
    assert clean_power < 0.5 * noisy_power


def test_log_mmse_preserves_the_signal_length_and_stays_finite():
    _, noisy, _ = make_noisy()
    enhanced = log_mmse(noisy, SR)
    assert enhanced.shape == noisy.shape
    assert np.all(np.isfinite(enhanced))


def test_log_mmse_is_deterministic():
    _, noisy, _ = make_noisy()
    np.testing.assert_array_equal(log_mmse(noisy, SR), log_mmse(noisy, SR))


def test_log_mmse_barely_touches_an_already_clean_signal():
    clean, _, lead = make_noisy(noise_std=0.0)
    enhanced = log_mmse(clean, SR)
    assert segmental_snr_db(clean, enhanced, start=lead) > 10.0


@pytest.mark.parametrize("alpha", [-0.1, 1.0, 1.5])
def test_log_mmse_rejects_an_invalid_smoothing_factor(alpha):
    _, noisy, _ = make_noisy()
    with pytest.raises(ValueError, match="alpha"):
        log_mmse(noisy, SR, alpha=alpha)


def test_log_mmse_returns_a_short_signal_unchanged(caplog):
    x = np.arange(64, dtype=np.float64)
    with caplog.at_level("WARNING"):
        np.testing.assert_array_equal(log_mmse(x, SR), x)
    assert "shorter" in caplog.text


@pytest.mark.parametrize("noise_std", [0.05, 0.1, 0.2])
def test_spectral_subtraction_improves_the_snr(noise_std):
    clean, noisy, lead = make_noisy(noise_std=noise_std)
    enhanced = spectral_subtraction(noisy, SR)
    before = segmental_snr_db(clean, noisy, start=lead)
    after = segmental_snr_db(clean, enhanced, start=lead)
    assert after > before + 3.0


def test_spectral_subtraction_preserves_the_signal_length():
    _, noisy, _ = make_noisy()
    assert spectral_subtraction(noisy, SR).shape == noisy.shape


def test_stronger_over_subtraction_removes_more_noise_floor():
    _, noisy, lead = make_noisy()
    mild = spectral_subtraction(noisy, SR, over_sub=1.0)
    strong = spectral_subtraction(noisy, SR, over_sub=4.0)
    region = slice(0, lead // 2)
    assert np.mean(strong[region] ** 2) < np.mean(mild[region] ** 2)


@pytest.mark.parametrize("floor", [-0.1, 1.0, 2.0])
def test_spectral_subtraction_rejects_an_invalid_floor(floor):
    _, noisy, _ = make_noisy()
    with pytest.raises(ValueError, match="floor"):
        spectral_subtraction(noisy, SR, floor=floor)


def test_enhancers_warn_when_there_are_too_few_noise_frames(caplog):
    _, noisy, _ = make_noisy()
    with caplog.at_level("WARNING"):
        log_mmse(noisy, SR, noise_frames=100000)
    assert "noise estimate" in caplog.text
