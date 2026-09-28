"""STFT/ISTFT invertibility, peak placement and spectrogram rendering."""

from __future__ import annotations

import numpy as np
import pytest

from speechdsp.spectral import (
    istft,
    power_spectrum,
    spectrogram_db,
    spectrogram_image,
    stft,
)

from .conftest import make_tone


def test_stft_peaks_at_the_bin_of_a_bin_centred_tone(sr):
    n_fft, hop = 512, 128
    bin_index = 40
    freq = bin_index * sr / n_fft
    S = stft(make_tone(freq, 1.0, sr), n_fft, hop)
    mid = np.abs(S[S.shape[0] // 2])
    assert int(np.argmax(mid)) == bin_index


def test_stft_shape_follows_the_documented_convention(sr):
    n_fft, hop = 256, 64
    S = stft(make_tone(500.0, 0.5, sr), n_fft, hop)
    assert S.shape[1] == n_fft // 2 + 1
    assert np.iscomplexobj(S)


@pytest.mark.parametrize("hop_div", [2, 4, 8])
@pytest.mark.parametrize("window", ["hann", "hamming"])
def test_istft_reconstructs_the_signal_exactly(rng, hop_div, window):
    n_fft = 256
    hop = n_fft // hop_div
    x = rng.standard_normal(hop * 60)
    y = istft(stft(x, n_fft, hop, window=window), n_fft, hop, window=window)
    assert y.size >= x.size
    assert np.max(np.abs(y[: x.size] - x)) < 1e-10


def test_istft_round_trip_also_works_without_centring(rng):
    # Without centring the very first sample sits where the periodic Hann window
    # is exactly zero and no other frame overlaps it, so it cannot be recovered;
    # everything from sample 1 onwards is exact.
    n_fft, hop = 128, 32
    x = rng.standard_normal(hop * 40)
    y = istft(stft(x, n_fft, hop, center=False), n_fft, hop, center=False)
    assert np.max(np.abs(y[1 : x.size] - x[1:])) < 1e-10


def test_istft_rejects_a_spectrogram_with_the_wrong_number_of_bins():
    with pytest.raises(ValueError, match="bins"):
        istft(np.zeros((4, 100), dtype=complex), n_fft=512, hop=128)


def test_power_spectrum_is_real_non_negative_and_peaks_correctly(sr):
    n_fft, hop = 512, 160
    P = power_spectrum(make_tone(1000.0, 0.5, sr), n_fft, hop)
    assert P.shape[1] == n_fft // 2 + 1
    assert np.all(P >= 0.0)
    peak_bin = int(np.argmax(P[P.shape[0] // 2]))
    assert abs(peak_bin * sr / n_fft - 1000.0) < sr / n_fft


def test_spectrogram_db_is_bounded_by_the_dynamic_range(sr):
    db = spectrogram_db(make_tone(440.0, 0.4, sr), sr, n_fft=512, hop=160, top_db=80.0)
    assert db.max() == pytest.approx(0.0, abs=1e-9)
    assert db.min() >= -80.0


def test_spectrogram_db_rejects_a_non_positive_dynamic_range(sr):
    with pytest.raises(ValueError, match="top_db"):
        spectrogram_db(make_tone(440.0, 0.2, sr), sr, n_fft=256, hop=64, top_db=0.0)


@pytest.mark.parametrize("duration", [0.2, 1.0, 3.0])
def test_spectrogram_image_has_a_fixed_shape_whatever_the_duration(sr, duration):
    img = spectrogram_image(make_tone(600.0, duration, sr), sr)
    assert img.shape == (40, 98)
    assert img.dtype == np.uint8


def test_spectrogram_image_honours_a_custom_shape(sr):
    img = spectrogram_image(make_tone(600.0, 0.8, sr), sr, shape=(64, 120))
    assert img.shape == (64, 120)
    assert img.max() == 255


def test_spectrogram_image_puts_a_high_tone_in_the_upper_bands(sr):
    low = spectrogram_image(make_tone(200.0, 1.0, sr), sr)
    high = spectrogram_image(make_tone(6000.0, 1.0, sr), sr)
    assert int(np.argmax(low.mean(axis=1))) < int(np.argmax(high.mean(axis=1)))


def test_spectrogram_image_survives_a_signal_shorter_than_one_frame(sr):
    img = spectrogram_image(np.zeros(100), sr)
    assert img.shape == (40, 98)
