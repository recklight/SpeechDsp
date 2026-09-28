"""Shared synthetic fixtures.

Every signal used by the test suite is generated in code: the package must be
testable without any recorded audio, and the expected values below are derived
analytically rather than captured from a previous run.
"""

from __future__ import annotations

import numpy as np
import pytest

SR = 16000


@pytest.fixture
def sr() -> int:
    """Sampling rate used by most tests."""
    return SR


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator, so every test is reproducible."""
    return np.random.default_rng(20240101)


def make_tone(freq: float, duration: float, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    """Pure sine of the given frequency, duration in seconds."""
    t = np.arange(round(float(duration * sr)), dtype=np.float64) / sr
    return amp * np.sin(2.0 * np.pi * freq * t)


def make_silence_speech_silence(
    sr: int = SR,
    silence_s: float = 0.3,
    speech_s: float = 0.5,
    noise_std: float = 1e-4,
    seed: int = 7,
) -> tuple[np.ndarray, int, int]:
    """Amplitude-modulated tone framed by two silent (very quiet) stretches.

    Returns the waveform together with the sample indices where the loud part
    starts and ends, which is what an endpoint detector has to recover.
    """
    gen = np.random.default_rng(seed)
    n_sil = round(float(silence_s * sr))
    n_speech = round(float(speech_s * sr))
    t = np.arange(n_speech, dtype=np.float64) / sr
    voiced = 0.5 * np.sin(2.0 * np.pi * 300.0 * t) * (1.0 + 0.5 * np.sin(2.0 * np.pi * 5.0 * t))
    x = np.concatenate([np.zeros(n_sil), voiced, np.zeros(n_sil)])
    x += gen.normal(0.0, noise_std, x.size)
    return x, n_sil, n_sil + n_speech


def segmental_snr_db(clean: np.ndarray, estimate: np.ndarray, start: int = 0) -> float:
    """Signal-to-noise ratio of ``estimate`` against ``clean``, in decibels."""
    ref = clean[start:]
    err = estimate[start:] - ref
    return float(10.0 * np.log10(np.sum(ref**2) / np.sum(err**2)))
