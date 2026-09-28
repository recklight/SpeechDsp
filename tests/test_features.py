"""Numerical checks on the MFCC front end and its building blocks."""

from __future__ import annotations

import numpy as np
import pytest

from speechdsp.features import (
    cmn,
    cmvn,
    cvn,
    delta,
    hz_to_mel,
    mel_filterbank,
    mel_to_hz,
    mfcc,
    mfcc_with_deltas,
    preemphasis,
)

from .conftest import make_tone


def test_preemphasis_matches_the_difference_equation():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(preemphasis(x, 0.5), [1.0, 1.5, 2.0, 2.5])


def test_preemphasis_with_zero_coefficient_is_a_pass_through(rng):
    x = rng.standard_normal(32)
    np.testing.assert_allclose(preemphasis(x, 0.0), x)


def test_mel_scale_conversions_are_mutual_inverses():
    f = np.array([0.0, 100.0, 700.0, 1000.0, 4000.0, 8000.0])
    np.testing.assert_allclose(mel_to_hz(hz_to_mel(f)), f, atol=1e-9)
    assert hz_to_mel(700.0) == pytest.approx(2595.0 * np.log10(2.0))


def test_mel_filterbank_shape_and_non_negativity(sr):
    fb = mel_filterbank(sr, n_fft=512, n_mels=26)
    assert fb.shape == (26, 257)
    assert np.all(fb >= 0.0)


def test_no_mel_filter_exceeds_unit_gain(sr):
    # The triangles are defined on continuous frequency and peak at exactly 1 at
    # their centre; the sampled peak is slightly lower whenever the centre falls
    # between two FFT bins, which is the normal case.
    fb = mel_filterbank(sr, n_fft=1024, n_mels=26)
    peaks = fb.max(axis=1)
    assert np.all(peaks <= 1.0 + 1e-12)
    assert np.all(peaks > 0.8)


def test_a_mel_filter_centred_on_a_bin_peaks_at_exactly_one(sr):
    fb = mel_filterbank(sr, n_fft=8192, n_mels=26)
    # With a fine frequency grid at least one centre lands on a bin.
    assert fb.max() == pytest.approx(1.0, abs=1e-3)


def test_mel_filter_centres_increase_monotonically(sr):
    fb = mel_filterbank(sr, n_fft=1024, n_mels=26)
    centres = np.argmax(fb, axis=1)
    assert np.all(np.diff(centres) > 0)


def test_mel_filters_widen_towards_high_frequencies(sr):
    fb = mel_filterbank(sr, n_fft=1024, n_mels=26)
    widths = np.count_nonzero(fb > 0.0, axis=1)
    assert widths[-1] > widths[0]


def test_mel_filterbank_covers_the_whole_band_between_its_edges(sr):
    n_fft, n_mels = 512, 26
    fb = mel_filterbank(sr, n_fft, n_mels=n_mels)
    total = fb.sum(axis=0)
    active = np.flatnonzero(total > 0.0)
    # Only the first and last bins, which lie outside the outermost triangles,
    # may be left uncovered.
    assert active[0] <= 1
    assert active[-1] >= n_fft // 2 - 1
    assert np.all(total[active[0] : active[-1] + 1] > 0.0)


def test_mel_filterbank_respects_an_explicit_frequency_range(sr):
    fb = mel_filterbank(sr, n_fft=512, n_mels=20, fmin=300.0, fmax=3400.0)
    bin_hz = np.linspace(0.0, sr / 2, fb.shape[1])
    active = bin_hz[fb.sum(axis=0) > 0.0]
    assert active.min() >= 300.0 - sr / 512
    assert active.max() <= 3400.0 + sr / 512


def test_mel_filterbank_rejects_an_empty_frequency_range(sr):
    with pytest.raises(ValueError, match="fmin"):
        mel_filterbank(sr, n_fft=512, n_mels=10, fmin=4000.0, fmax=1000.0)


def test_mfcc_shape_follows_the_frame_rate(sr):
    x = make_tone(440.0, 1.0, sr)
    feats = mfcc(x, sr, n_mfcc=13, frame_ms=25.0, hop_ms=10.0)
    expected = 1 + (x.size - int(0.025 * sr)) // int(0.010 * sr)
    assert feats.shape == (expected, 13)
    assert np.all(np.isfinite(feats))


def test_mfcc_of_a_constant_signal_is_identical_in_every_frame(sr):
    feats = mfcc(np.full(sr, 0.3), sr, preemph=0.0)
    assert np.all(np.isfinite(feats))
    assert np.max(np.abs(feats - feats[0])) < 1e-12


def test_mfcc_of_digital_silence_stays_finite(sr):
    feats = mfcc(np.zeros(sr // 2), sr)
    assert np.all(np.isfinite(feats))
    assert feats[:, 0].max() < 0.0


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 44100])
def test_mfcc_works_at_every_sampling_rate(rate):
    feats = mfcc(make_tone(440.0, 0.5, rate), rate)
    assert feats.shape[1] == 13
    assert feats.shape[0] > 10
    assert np.all(np.isfinite(feats))


def test_mfcc_frame_count_is_rate_independent_for_a_fixed_duration():
    counts = {rate: mfcc(make_tone(440.0, 1.0, rate), rate).shape[0] for rate in (8000, 16000)}
    assert abs(counts[8000] - counts[16000]) <= 1


def test_mfcc_tracks_the_spectral_centre_of_gravity(sr):
    low = mfcc(make_tone(300.0, 0.5, sr), sr, append_energy=False)
    high = mfcc(make_tone(3000.0, 0.5, sr), sr, append_energy=False)
    # c1 carries the spectral balance: negative for a high tone, positive for a
    # low one, because the first DCT basis function falls with frequency.
    assert low[:, 1].mean() > high[:, 1].mean()


def test_mfcc_energy_coefficient_follows_the_signal_level(sr):
    quiet = mfcc(make_tone(440.0, 0.5, sr, amp=0.01), sr)
    loud = mfcc(make_tone(440.0, 0.5, sr, amp=0.5), sr)
    assert loud[:, 0].mean() > quiet[:, 0].mean()
    # Doubling the amplitude adds 2 * ln(2) to the log energy.
    louder = mfcc(make_tone(440.0, 0.5, sr, amp=1.0), sr)
    assert (louder[:, 0] - loud[:, 0]).mean() == pytest.approx(2 * np.log(2.0), abs=1e-6)


def test_mfcc_rejects_more_coefficients_than_filters(sr):
    with pytest.raises(ValueError, match="cannot exceed"):
        mfcc(make_tone(440.0, 0.2, sr), sr, n_mfcc=30, n_mels=26)


def test_mfcc_returns_an_empty_block_for_a_too_short_signal(sr):
    assert mfcc(np.zeros(100), sr).shape == (0, 13)


def test_mfcc_raises_the_fft_size_when_the_frame_does_not_fit(caplog):
    with caplog.at_level("WARNING"):
        feats = mfcc(make_tone(440.0, 0.3, 44100), 44100, n_fft=512)
    assert feats.shape[1] == 13
    assert "n_fft" in caplog.text


def test_delta_of_a_linear_ramp_is_the_slope():
    slope = 0.25
    ramp = (slope * np.arange(40.0))[:, None]
    d = delta(ramp, width=9)
    np.testing.assert_allclose(d[4:-4, 0], slope, atol=1e-12)


def test_delta_of_a_constant_sequence_is_zero():
    d = delta(np.full((20, 3), 2.5), width=5)
    np.testing.assert_allclose(d, 0.0, atol=1e-12)


def test_delta_preserves_the_matrix_shape(rng):
    feats = rng.standard_normal((25, 13))
    assert delta(feats).shape == feats.shape


@pytest.mark.parametrize("width", [2, 4, 1])
def test_delta_rejects_an_invalid_window(width):
    with pytest.raises(ValueError, match="odd"):
        delta(np.zeros((10, 2)), width=width)


def test_mfcc_with_deltas_stacks_three_blocks(sr):
    x = make_tone(440.0, 0.6, sr)
    base = mfcc(x, sr, n_mfcc=13)
    stacked = mfcc_with_deltas(x, sr, n_mfcc=13)
    assert stacked.shape == (base.shape[0], 39)
    np.testing.assert_allclose(stacked[:, :13], base)
    np.testing.assert_allclose(stacked[:, 13:26], delta(base))


def test_cmn_removes_the_mean(rng):
    X = rng.standard_normal((50, 8)) + 3.0
    np.testing.assert_allclose(cmn(X).mean(axis=0), 0.0, atol=1e-12)


def test_cvn_normalises_the_variance(rng):
    X = rng.standard_normal((200, 4)) * 5.0
    np.testing.assert_allclose(cvn(X).std(axis=0), 1.0, atol=1e-6)


def test_cmvn_standardises_both_moments(rng):
    X = rng.standard_normal((200, 4)) * 5.0 + 2.0
    Z = cmvn(X)
    np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(Z.std(axis=0), 1.0, atol=1e-6)


def test_cmvn_keeps_a_constant_dimension_finite():
    X = np.hstack([np.ones((10, 1)), np.arange(10.0)[:, None]])
    assert np.all(np.isfinite(cmvn(X)))
