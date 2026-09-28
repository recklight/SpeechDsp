"""WAV and HTK round trips."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.io import wavfile

from speechdsp.io import HTK_HEADER_SIZE, read_htk, read_wav, write_htk, write_wav

from .conftest import make_tone


def test_wav_round_trip_preserves_the_waveform(tmp_path, sr):
    x = make_tone(440.0, 0.25, sr, amp=0.9)
    path = tmp_path / "tone.wav"
    write_wav(path, x, sr)
    y, sr_read = read_wav(path)
    assert sr_read == sr
    assert y.shape == x.shape
    assert y.dtype == np.float64
    np.testing.assert_allclose(y, x, atol=2e-4)


def test_write_wav_creates_missing_directories(tmp_path, sr):
    path = tmp_path / "nested" / "deeper" / "tone.wav"
    write_wav(path, make_tone(300.0, 0.05, sr), sr)
    assert path.is_file()


def test_write_wav_clips_and_warns_about_overload(tmp_path, sr, caplog):
    with caplog.at_level("WARNING"):
        write_wav(tmp_path / "loud.wav", np.full(sr // 10, 2.0), sr)
    y, _ = read_wav(tmp_path / "loud.wav")
    assert y.max() <= 1.0
    assert "clipped" in caplog.text


def test_read_wav_downmixes_stereo_to_mono(tmp_path, sr):
    left = np.full(100, 0.5)
    right = np.full(100, -0.1)
    stereo = np.stack([left, right], axis=1)
    path = tmp_path / "stereo.wav"
    wavfile.write(path, sr, np.round(stereo * 32767).astype(np.int16))
    y, _ = read_wav(path)
    assert y.ndim == 1
    np.testing.assert_allclose(y, 0.2, atol=1e-3)


def test_read_wav_accepts_float32_files(tmp_path, sr):
    x = make_tone(500.0, 0.05, sr).astype(np.float32)
    path = tmp_path / "float.wav"
    wavfile.write(path, sr, x)
    y, _ = read_wav(path)
    np.testing.assert_allclose(y, x, atol=1e-7)


def test_read_wav_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_wav(tmp_path / "absent.wav")


def test_htk_round_trip_is_bit_exact(tmp_path, rng):
    feats = rng.standard_normal((37, 13)).astype(np.float32).astype(np.float64)
    path = tmp_path / "feats.mfc"
    write_htk(path, feats, period_100ns=100000, kind=6)
    back, header = read_htk(path)
    np.testing.assert_array_equal(back, feats)
    assert header["n_samples"] == 37
    assert header["n_dim"] == 13
    assert header["samp_size"] == 52
    assert header["samp_period"] == 100000
    assert header["parm_kind"] == 6
    assert header["format"] == "binary"


def test_htk_file_uses_big_endian_fields(tmp_path):
    feats = np.zeros((4, 3))
    path = tmp_path / "be.mfc"
    write_htk(path, feats, period_100ns=100000, kind=6)
    raw = path.read_bytes()
    assert raw[:4] == b"\x00\x00\x00\x04"  # nSamples = 4, big endian
    assert raw[8:10] == b"\x00\x0c"  # sampSize = 12 bytes
    assert raw[10:12] == b"\x00\x06"  # parmKind = MFCC
    assert len(raw) == HTK_HEADER_SIZE + 4 * 12


def test_htk_round_trip_for_a_single_frame(tmp_path):
    feats = np.arange(13.0)[None, :]
    path = tmp_path / "one.mfc"
    write_htk(path, feats)
    back, header = read_htk(path)
    np.testing.assert_allclose(back, feats, atol=1e-6)
    assert header["n_samples"] == 1


def test_read_htk_detects_a_plain_text_matrix(tmp_path):
    feats = np.arange(12.0).reshape(4, 3)
    path = tmp_path / "text.mfc"
    np.savetxt(path, feats, fmt="%.6f")
    back, header = read_htk(path)
    np.testing.assert_allclose(back, feats, atol=1e-6)
    assert header["format"] == "text"
    assert header["n_samples"] == 4
    assert header["n_dim"] == 3


def test_read_htk_rejects_a_truncated_file(tmp_path):
    path = tmp_path / "broken.mfc"
    write_htk(path, np.zeros((10, 4)))
    path.write_bytes(path.read_bytes()[:-8])
    with pytest.raises(ValueError, match="bytes"):
        read_htk(path)


def test_read_htk_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_htk(tmp_path / "absent.mfc")


def test_write_htk_rejects_a_non_matrix():
    with pytest.raises(ValueError, match="2-D"):
        write_htk("unused.mfc", np.zeros(10))
