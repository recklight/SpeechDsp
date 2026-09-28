"""Frame blocking, overlap-add and frame/sample bookkeeping."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import get_window

from speechdsp.framing import (
    enframe,
    frame_time,
    frame_to_sample,
    num_frames,
    overlap_add,
    resolve_window,
    window_sumsquare,
)


def test_num_frames_matches_the_documented_formula():
    assert num_frames(10, 4, 2) == 4
    assert num_frames(11, 4, 2) == 4
    assert num_frames(12, 4, 2) == 5
    assert num_frames(3, 4, 2) == 0


def test_enframe_lays_out_the_expected_windows():
    frames = enframe(np.arange(6.0), frame_len=3, hop=2)
    expected = np.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]])
    np.testing.assert_array_equal(frames, expected)


def test_enframe_returns_an_empty_block_for_a_too_short_signal():
    frames = enframe(np.arange(3.0), frame_len=8, hop=4)
    assert frames.shape == (0, 8)


def test_enframe_applies_the_window():
    taps = np.array([0.0, 1.0, 2.0])
    frames = enframe(np.ones(5), frame_len=3, hop=2, window=taps)
    np.testing.assert_allclose(frames, np.tile(taps, (2, 1)))


def test_enframe_rejects_a_window_of_the_wrong_length():
    with pytest.raises(ValueError, match="taps"):
        enframe(np.ones(10), frame_len=4, hop=2, window=np.ones(5))


def test_enframe_does_not_alias_the_input_buffer():
    x = np.arange(8.0)
    frames = enframe(x, frame_len=4, hop=4)
    frames[0, 0] = -99.0
    assert x[0] == 0.0


def test_enframe_overlap_add_round_trip_without_overlap(rng):
    x = rng.standard_normal(64)
    frames = enframe(x, frame_len=8, hop=8)
    np.testing.assert_allclose(overlap_add(frames, hop=8), x, atol=1e-12)


def test_enframe_overlap_add_round_trip_with_weighted_overlap(rng):
    x = rng.standard_normal(512)
    n_fft, hop = 64, 16
    frames = enframe(x, n_fft, hop, "hann")
    y = overlap_add(frames, hop, "hann")
    env = window_sumsquare("hann", frames.shape[0], n_fft, hop)
    covered = env > 1e-10
    np.testing.assert_allclose(y[covered] / env[covered], x[: y.size][covered], atol=1e-12)


def test_hann_at_half_overlap_satisfies_cola():
    n_fft, hop, n_frames = 64, 32, 10
    env = overlap_add(np.ones((n_frames, n_fft)), hop, "hann")
    interior = env[n_fft : (n_frames - 1) * hop]
    np.testing.assert_allclose(interior, 1.0, atol=1e-12)


def test_overlap_add_rejects_non_matrix_input():
    with pytest.raises(ValueError, match="2-D"):
        overlap_add(np.ones(10), hop=4)


def test_resolve_window_returns_the_periodic_window():
    np.testing.assert_allclose(
        resolve_window("hamming", 16), get_window("hamming", 16, fftbins=True)
    )
    assert resolve_window(None, 16) is None


def test_frame_to_sample_and_frame_time_agree_with_the_frame_layout():
    frame_len, hop, sr = 400, 160, 16000
    assert frame_to_sample(0, frame_len, hop) == 0
    assert frame_to_sample(5, frame_len, hop) == 800
    times = frame_time(3, frame_len, hop, sr)
    np.testing.assert_allclose(times, [0.0125, 0.0225, 0.0325], atol=1e-12)


@pytest.mark.parametrize("bad", [0, -3])
def test_invalid_sizes_are_rejected(bad):
    with pytest.raises(ValueError):
        num_frames(100, bad, 2)
    with pytest.raises(ValueError):
        num_frames(100, 4, bad)
