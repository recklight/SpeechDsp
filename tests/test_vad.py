"""Energy/zero-crossing measures and endpoint detection."""

from __future__ import annotations

import numpy as np
import pytest

from speechdsp.framing import enframe
from speechdsp.vad import endpoint_detect, frame_energy, trim_silence, zero_crossing_rate

from .conftest import make_silence_speech_silence, make_tone

TOLERANCE_MS = 60.0


def test_frame_energy_sums_the_squares():
    frames = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 2.0]])
    np.testing.assert_allclose(frame_energy(frames), [4.0, 4.0])


def test_frame_energy_is_zero_for_silence():
    np.testing.assert_allclose(frame_energy(np.zeros((5, 16))), 0.0)


def test_frame_energy_rejects_a_flat_array():
    with pytest.raises(ValueError, match="2-D"):
        frame_energy(np.ones(16))


def test_zero_crossing_rate_of_an_alternating_frame_is_one():
    frames = np.array([[1.0, -1.0, 1.0, -1.0, 1.0]])
    np.testing.assert_allclose(zero_crossing_rate(frames), 1.0)


def test_zero_crossing_rate_of_a_positive_frame_is_zero():
    np.testing.assert_allclose(zero_crossing_rate(np.ones((3, 8))), 0.0)


def test_zero_crossing_rate_increases_with_frequency(sr):
    frame_len = 400
    low = enframe(make_tone(100.0, 0.2, sr), frame_len, frame_len)
    high = enframe(make_tone(4000.0, 0.2, sr), frame_len, frame_len)
    assert zero_crossing_rate(low).mean() < zero_crossing_rate(high).mean()


def test_zero_crossing_rate_matches_the_analytic_value_for_a_tone(sr):
    freq, frame_len = 1000.0, 1600
    frames = enframe(make_tone(freq, 0.5, sr), frame_len, frame_len)
    expected = 2.0 * freq / sr
    np.testing.assert_allclose(zero_crossing_rate(frames).mean(), expected, rtol=0.02)


def test_endpoint_detect_finds_the_speech_boundaries(sr):
    x, true_start, true_end = make_silence_speech_silence(sr)
    start, end = endpoint_detect(x, sr)
    tolerance = TOLERANCE_MS / 1000.0 * sr
    assert abs(start - true_start) < tolerance
    assert abs(end - true_end) < tolerance
    assert start < end


def test_endpoint_detect_never_clips_into_the_speech(sr):
    x, true_start, true_end = make_silence_speech_silence(sr)
    start, end = endpoint_detect(x, sr)
    # The detector may open slightly early or close slightly late, but it must
    # not swallow the loud part of the utterance.
    assert start <= true_start + 0.03 * sr
    assert end >= true_end - 0.03 * sr


def test_endpoint_detect_returns_the_whole_signal_for_pure_silence(sr):
    x = np.zeros(sr)
    assert endpoint_detect(x, sr) == (0, x.size)


def test_endpoint_detect_returns_the_whole_signal_when_it_is_too_short(sr):
    x = make_tone(440.0, 0.01, sr)
    assert endpoint_detect(x, sr) == (0, x.size)


def test_endpoint_detect_is_level_invariant(sr):
    x, _, _ = make_silence_speech_silence(sr)
    quiet = endpoint_detect(x * 0.01, sr)
    loud = endpoint_detect(x * 1.0, sr)
    assert abs(quiet[0] - loud[0]) <= 0.02 * sr
    assert abs(quiet[1] - loud[1]) <= 0.02 * sr


def test_trim_silence_shortens_the_signal_but_keeps_the_energy(sr):
    x, true_start, true_end = make_silence_speech_silence(sr)
    trimmed = trim_silence(x, sr)
    assert trimmed.size < x.size
    kept = float(np.sum(trimmed**2))
    total = float(np.sum(x[true_start:true_end] ** 2))
    assert kept > 0.95 * total


def test_trim_silence_does_not_alias_the_input(sr):
    x, _, _ = make_silence_speech_silence(sr)
    trimmed = trim_silence(x, sr)
    trimmed[0] = 123.0
    assert 123.0 not in x


def test_endpoint_detect_rejects_an_invalid_rate():
    with pytest.raises(ValueError, match="sr"):
        endpoint_detect(np.zeros(1000), 0)
