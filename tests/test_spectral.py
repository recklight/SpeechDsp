"""STFT/ISTFT invertibility, peak placement and spectrogram rendering."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import get_window

from speechdsp.framing import enframe
from speechdsp.spectral import (
    istft,
    power_spectrum,
    power_to_db,
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


# --- dynamic range: the floor is relative to the peak --------------------------


def test_power_to_db_clamps_relative_to_the_peak():
    db = power_to_db(np.array([1e-6, 1e-8, 1e-20]), top_db=80.0)
    np.testing.assert_allclose(db, [0.0, -20.0, -80.0], atol=1e-9)


def test_power_to_db_maps_a_silent_input_to_the_bottom_of_the_range():
    assert np.all(power_to_db(np.zeros((3, 4)), top_db=60.0) == -60.0)
    assert power_to_db(np.zeros(0)).shape == (0,)


@pytest.mark.parametrize("bad", [0.0, -10.0])
def test_power_to_db_rejects_a_non_positive_dynamic_range(bad):
    with pytest.raises(ValueError, match="top_db"):
        power_to_db(np.ones(3), top_db=bad)


def test_power_to_db_rejects_negative_power():
    with pytest.raises(ValueError, match="non-negative"):
        power_to_db(np.array([1.0, -1.0]))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_power_to_db_rejects_non_finite_power(bad):
    # Mapping NaN or inf to the floor would pass the input off as silence.
    with pytest.raises(ValueError, match="finite"):
        power_to_db(np.array([1.0, bad]))


def test_a_nan_in_the_signal_is_reported_rather_than_drawn_as_silence(sr):
    x = np.zeros(sr)
    x[sr // 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        spectrogram_db(x, sr, n_fft=512, hop=160)


def test_a_quiet_recording_keeps_the_full_dynamic_range(rng, sr):
    # A tone followed by silence spans more than 80 dB.  Scaled by 1e-5
    # (-100 dB) its peak power is about 1e-9, so an absolute 1e-12 floor would
    # stop the silent half near -30 dB instead of at the bottom of the range.
    x = np.concatenate(
        [make_tone(440.0, 0.5, sr) + 1e-3 * rng.standard_normal(sr // 2), np.zeros(sr // 2)]
    )
    loud = spectrogram_db(x, sr, n_fft=512, hop=160)
    quiet = spectrogram_db(1e-5 * x, sr, n_fft=512, hop=160)
    assert loud.min() == pytest.approx(-80.0)
    assert quiet.min() == pytest.approx(-80.0)
    np.testing.assert_allclose(quiet, loud, atol=1e-6)


def test_spectrogram_image_does_not_depend_on_the_recording_level(sr):
    tone = make_tone(600.0, 1.0, sr)
    np.testing.assert_array_equal(spectrogram_image(1e-5 * tone, sr), spectrogram_image(tone, sr))


def test_a_silent_recording_renders_as_a_black_image(sr):
    img = spectrogram_image(np.zeros(sr), sr)
    assert img.shape == (40, 98)
    assert np.all(img == 0)
    assert np.all(spectrogram_db(np.zeros(sr), sr, n_fft=512, hop=160) == -80.0)


# --- frames shorter than the FFT ------------------------------------------------


def test_a_short_window_equals_zero_padded_frames(rng):
    n_fft, win, hop = 512, 400, 160
    x = rng.standard_normal(4000)
    S = stft(x, n_fft, hop, center=False, win_length=win)
    # The window sits in the middle of the n_fft span, so frame t starts its
    # non-zero part (n_fft - win) // 2 samples after t * hop.
    offset = (n_fft - win) // 2
    frames = enframe(x[offset:], win, hop, get_window("hann", win, fftbins=True))
    reference = np.abs(np.fft.rfft(frames, n=n_fft, axis=-1))
    count = frames.shape[0]
    np.testing.assert_allclose(np.abs(S[:count]), reference, atol=1e-9)


@pytest.mark.parametrize("window", ["hann", "hamming", None])
def test_istft_reconstructs_the_signal_with_a_short_window(rng, window):
    n_fft, win, hop = 512, 400, 100
    x = rng.standard_normal(hop * 60)
    S = stft(x, n_fft, hop, window=window, win_length=win)
    y = istft(S, n_fft, hop, window=window, win_length=win)
    assert np.max(np.abs(y[: x.size] - x)) < 1e-10


@pytest.mark.parametrize(("n_fft", "win", "hop"), [(512, 400, 100), (512, 401, 160), (16, 7, 3)])
def test_an_uncentred_short_window_loses_only_the_documented_ends(rng, n_fft, win, hop):
    x = rng.standard_normal(hop * 40 + 7)
    S = stft(x, n_fft, hop, window="hamming", center=False, win_length=win)
    y = istft(S, n_fft, hop, window="hamming", center=False, win_length=win)[: x.size]
    head = (n_fft - win) // 2
    tail = n_fft - win - head
    np.testing.assert_allclose(y[head : x.size - tail], x[head : x.size - tail], atol=1e-10)
    lost = np.flatnonzero(np.abs(y - x) > 1e-10)
    # Exactly the samples before the first window are lost at the start; at the
    # end the loss stays within the bound, however the last frame falls.
    np.testing.assert_array_equal(lost[lost < head], np.arange(head))
    assert np.all((lost < head) | (lost >= x.size - tail))


def test_win_length_equal_to_n_fft_changes_nothing(rng):
    x = rng.standard_normal(3000)
    np.testing.assert_array_equal(stft(x, 256, 64, win_length=256), stft(x, 256, 64))


def test_an_explicit_window_array_has_win_length_taps(rng):
    x = rng.standard_normal(2000)
    taps = get_window("hamming", 300, fftbins=True)
    np.testing.assert_allclose(
        stft(x, 512, 128, window=taps, win_length=300),
        stft(x, 512, 128, window="hamming", win_length=300),
    )
    with pytest.raises(ValueError, match="taps"):
        stft(x, 512, 128, window=taps)


@pytest.mark.parametrize("bad", [0, -1, 513])
def test_win_length_must_fit_inside_the_fft(bad):
    with pytest.raises(ValueError, match="win_length"):
        stft(np.zeros(1000), 512, 128, win_length=bad)


def test_the_spectrogram_helpers_accept_a_short_window(sr):
    tone = make_tone(1000.0, 0.5, sr)
    P = power_spectrum(tone, 512, 160, win_length=400)
    assert P.shape[1] == 257
    peak_bin = int(np.argmax(P[P.shape[0] // 2]))
    assert abs(peak_bin * sr / 512 - 1000.0) < sr / 512
    assert spectrogram_db(tone, sr, 512, 160, win_length=400).max() == pytest.approx(0.0)
    assert spectrogram_image(tone, sr, win_length=400).shape == (40, 98)
