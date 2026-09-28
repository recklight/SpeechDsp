"""Cepstral feature extraction: pre-emphasis, mel filterbanks, MFCC and deltas.

The MFCC pipeline implemented here follows the classical formulation: a
first-order pre-emphasis filter, frame blocking with a Hamming window, a
periodogram power spectrum, a bank of triangular filters equally spaced on the
mel scale, a logarithm, a type-II DCT and finally a sinusoidal lifter.

References
----------
.. [1] S. B. Davis and P. Mermelstein, "Comparison of parametric representations
       for monosyllabic word recognition in continuously spoken sentences",
       *IEEE Trans. ASSP*, 28(4):357-366, 1980.
.. [2] S. S. Stevens, J. Volkmann and E. B. Newman, "A scale for the measurement
       of the psychological magnitude pitch", *J. Acoust. Soc. Am.*,
       8(3):185-190, 1937.
.. [3] S. Young et al., *The HTK Book (version 3.4)*, Cambridge University
       Engineering Department, 2006.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.fft import dct

from .framing import enframe, num_frames

__all__ = [
    "cmn",
    "cmvn",
    "cvn",
    "delta",
    "hz_to_mel",
    "mel_filterbank",
    "mel_to_hz",
    "mfcc",
    "mfcc_with_deltas",
    "preemphasis",
]

_LOGGER = logging.getLogger(__name__)

#: Floor applied to mel energies before the logarithm, so that digital silence
#: yields a finite (very negative) value instead of ``-inf``.
_ENERGY_FLOOR = 1e-10


def hz_to_mel(f: np.ndarray | float) -> np.ndarray:
    """Convert a frequency in hertz to the mel scale.

    Parameters
    ----------
    f : numpy.ndarray or float
        Frequency in Hz.

    Returns
    -------
    numpy.ndarray
        ``2595 * log10(1 + f / 700)``.

    References
    ----------
    .. [1] S. S. Stevens, J. Volkmann and E. B. Newman, "A scale for the
           measurement of the psychological magnitude pitch", *J. Acoust. Soc.
           Am.*, 8(3):185-190, 1937.
    """
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m: np.ndarray | float) -> np.ndarray:
    """Convert a mel value back to hertz.

    Parameters
    ----------
    m : numpy.ndarray or float
        Value on the mel scale.

    Returns
    -------
    numpy.ndarray
        ``700 * (10 ** (m / 2595) - 1)``, the inverse of :func:`hz_to_mel`.
    """
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def preemphasis(x: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """Apply the first-order pre-emphasis filter ``y[n] = x[n] - a * x[n-1]``.

    The filter boosts the high band by roughly 6 dB/octave, compensating the
    spectral tilt of voiced speech before short-time analysis.

    Parameters
    ----------
    x : numpy.ndarray
        Input signal, flattened to 1-D.
    coeff : float, optional
        Pre-emphasis coefficient ``a``, default 0.97.  ``0.0`` disables the
        filter.

    Returns
    -------
    numpy.ndarray
        Filtered signal of the same length; the first sample is passed through
        unchanged.

    Examples
    --------
    >>> import numpy as np
    >>> preemphasis(np.array([1.0, 1.0, 1.0]), coeff=0.5)
    array([1. , 0.5, 0.5])
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return x.copy()
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - float(coeff) * x[:-1]
    return y


def mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int = 26,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Build a bank of triangular filters equally spaced on the mel scale.

    Filter ``m`` rises linearly from ``hz[m]`` to its centre ``hz[m + 1]`` and
    falls back to zero at ``hz[m + 2]``, where the ``n_mels + 2`` edge
    frequencies are equally spaced between ``fmin`` and ``fmax`` in the mel
    domain.  Each triangle peaks at exactly 1.0 (HTK convention), so the
    filterbank energies keep the units of the input power spectrum.

    Parameters
    ----------
    sr : int
        Sampling rate in Hz.
    n_fft : int
        FFT size the filterbank will be applied to.
    n_mels : int, optional
        Number of filters, default 26.
    fmin : float, optional
        Lowest edge frequency in Hz, default 0.
    fmax : float or None, optional
        Highest edge frequency in Hz.  ``None`` (default) means the Nyquist
        frequency ``sr / 2``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_mels, n_fft // 2 + 1)``.  Multiplying a power
        spectrum by its transpose gives the mel energies.

    Raises
    ------
    ValueError
        If the frequency range is empty or the sizes are not positive.

    References
    ----------
    .. [1] S. B. Davis and P. Mermelstein, "Comparison of parametric
           representations for monosyllabic word recognition in continuously
           spoken sentences", *IEEE Trans. ASSP*, 28(4):357-366, 1980.
    """
    if sr <= 0:
        raise ValueError("sr must be positive")
    if n_fft <= 0:
        raise ValueError("n_fft must be positive")
    if n_mels <= 0:
        raise ValueError("n_mels must be positive")
    nyquist = sr / 2.0
    if fmax is None:
        fmax = nyquist
    fmax = float(min(float(fmax), nyquist))
    fmin = float(fmin)
    if not 0.0 <= fmin < fmax:
        raise ValueError(f"require 0 <= fmin < fmax <= sr/2, got fmin={fmin}, fmax={fmax}")

    n_bins = n_fft // 2 + 1
    bin_hz = np.linspace(0.0, nyquist, n_bins, dtype=np.float64)
    edges_hz = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))

    lower, centre, upper = edges_hz[:-2, None], edges_hz[1:-1, None], edges_hz[2:, None]
    rise = (bin_hz[None, :] - lower) / (centre - lower)
    fall = (upper - bin_hz[None, :]) / (upper - centre)
    return np.maximum(0.0, np.minimum(rise, fall))


def _frame_sizes(sr: int, frame_ms: float, hop_ms: float) -> tuple[int, int]:
    """Convert frame/hop durations in milliseconds to a whole number of samples."""
    frame_len = round(float(sr * frame_ms / 1000.0))
    hop = round(float(sr * hop_ms / 1000.0))
    if frame_len <= 0 or hop <= 0:
        raise ValueError("frame_ms and hop_ms are too short for this sampling rate")
    return frame_len, hop


def _lifter_weights(n_coeff: int, lifter: int) -> np.ndarray | None:
    """Sinusoidal cepstral lifter ``1 + (L / 2) * sin(pi * n / L)``."""
    if lifter is None or lifter <= 0:
        return None
    n = np.arange(n_coeff, dtype=np.float64)
    return 1.0 + (lifter / 2.0) * np.sin(np.pi * n / float(lifter))


def mfcc(
    x: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_mels: int = 26,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    n_fft: int = 512,
    preemph: float = 0.97,
    lifter: int = 22,
    append_energy: bool = True,
) -> np.ndarray:
    """Mel-frequency cepstral coefficients.

    Pipeline: pre-emphasis, frame blocking with a Hamming window, periodogram
    power spectrum, mel filterbank, natural logarithm, orthonormal type-II DCT,
    sinusoidal liftering.

    Parameters
    ----------
    x : numpy.ndarray
        Input waveform, flattened to 1-D.
    sr : int
        Sampling rate in Hz.
    n_mfcc : int, optional
        Number of cepstral coefficients kept, including ``c0``.  Default 13.
    n_mels : int, optional
        Number of mel filters, default 26.
    frame_ms : float, optional
        Frame length in milliseconds, default 25.
    hop_ms : float, optional
        Frame advance in milliseconds, default 10.
    n_fft : int, optional
        FFT size, default 512.  If the frame is longer than ``n_fft`` the FFT
        size is raised to the next power of two and a warning is logged, so that
        no samples are discarded at high sampling rates.
    preemph : float, optional
        Pre-emphasis coefficient, default 0.97; ``0.0`` disables it.
    lifter : int, optional
        Sinusoidal lifter parameter ``L``, default 22; ``0`` disables liftering.
    append_energy : bool, optional
        If ``True`` (default) coefficient ``c0`` is replaced by the logarithm of
        the total frame energy, which is more robust than the DCT offset term.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_frames, n_mfcc)``; ``(0, n_mfcc)`` when the signal is
        shorter than one frame.

    Raises
    ------
    ValueError
        If ``n_mfcc`` exceeds ``n_mels`` or the sizes are not positive.

    Notes
    -----
    Frames are *not* centred: frame ``t`` covers
    ``[t * hop, t * hop + frame_len)`` of the pre-emphasised signal, matching
    HTK and most ASR front ends.

    References
    ----------
    .. [1] S. B. Davis and P. Mermelstein, "Comparison of parametric
           representations for monosyllabic word recognition in continuously
           spoken sentences", *IEEE Trans. ASSP*, 28(4):357-366, 1980.
    .. [2] S. Young et al., *The HTK Book (version 3.4)*, Cambridge University
           Engineering Department, 2006.
    """
    if sr <= 0:
        raise ValueError("sr must be positive")
    if n_mfcc <= 0:
        raise ValueError("n_mfcc must be positive")
    if n_mfcc > n_mels:
        raise ValueError(f"n_mfcc={n_mfcc} cannot exceed n_mels={n_mels}")

    x = np.asarray(x, dtype=np.float64).ravel()
    frame_len, hop = _frame_sizes(sr, frame_ms, hop_ms)
    if frame_len > n_fft:
        bumped = int(2 ** np.ceil(np.log2(frame_len)))
        _LOGGER.warning(
            "frame of %d samples does not fit n_fft=%d; using n_fft=%d instead",
            frame_len,
            n_fft,
            bumped,
        )
        n_fft = bumped

    if num_frames(x.size, frame_len, hop) == 0:
        return np.zeros((0, n_mfcc), dtype=np.float64)

    frames = enframe(preemphasis(x, preemph), frame_len, hop, "hamming")
    spectrum = np.fft.rfft(frames, n=n_fft, axis=-1)
    power = (np.abs(spectrum) ** 2) / float(n_fft)

    fb = mel_filterbank(sr, n_fft, n_mels=n_mels)
    mel_energy = power @ fb.T
    log_mel = np.log(np.maximum(mel_energy, _ENERGY_FLOOR))

    coeffs = dct(log_mel, type=2, axis=-1, norm="ortho")[:, :n_mfcc]
    weights = _lifter_weights(n_mfcc, lifter)
    if weights is not None:
        coeffs = coeffs * weights
    if append_energy:
        total = np.maximum(power.sum(axis=1), _ENERGY_FLOOR)
        coeffs[:, 0] = np.log(total)
    return coeffs


def delta(feats: np.ndarray, width: int = 9) -> np.ndarray:
    """Regression-based delta (velocity) coefficients.

    Implements ``d[t] = sum_{n=1..N} n * (c[t+n] - c[t-n]) / (2 * sum_n n^2)``
    with ``N = (width - 1) // 2``.  The sequence is padded by edge replication,
    so the output has the same number of frames as the input.

    Parameters
    ----------
    feats : numpy.ndarray
        Feature matrix of shape ``(n_frames, n_dim)``.
    width : int, optional
        Regression window length in frames; must be an odd number ``>= 3``.
        Default 9, i.e. ``N = 4``.

    Returns
    -------
    numpy.ndarray
        Delta coefficients of shape ``(n_frames, n_dim)``.

    Raises
    ------
    ValueError
        If ``width`` is even or smaller than 3.

    Notes
    -----
    For a feature that grows linearly with the frame index the regression
    returns the slope exactly, which is the standard sanity check for this
    formula.

    References
    ----------
    .. [1] S. Young et al., *The HTK Book (version 3.4)*, Cambridge University
           Engineering Department, 2006, section 5.9.
    """
    feats = np.asarray(feats, dtype=np.float64)
    if feats.ndim != 2:
        raise ValueError("feats must be a 2-D array of shape (n_frames, n_dim)")
    if width < 3 or width % 2 == 0:
        raise ValueError(f"width must be an odd integer >= 3, got {width}")
    if feats.shape[0] == 0:
        return feats.copy()

    half = (width - 1) // 2
    padded = np.pad(feats, ((half, half), (0, 0)), mode="edge")
    n_frames = feats.shape[0]
    out = np.zeros_like(feats)
    for n in range(1, half + 1):
        ahead = padded[half + n : half + n + n_frames]
        behind = padded[half - n : half - n + n_frames]
        out += n * (ahead - behind)
    denom = 2.0 * sum(n * n for n in range(1, half + 1))
    return out / denom


def mfcc_with_deltas(x: np.ndarray, sr: int, **kw: Any) -> np.ndarray:
    """MFCCs concatenated with their first and second order deltas.

    Parameters
    ----------
    x : numpy.ndarray
        Input waveform.
    sr : int
        Sampling rate in Hz.
    **kw
        Forwarded to :func:`mfcc`, except ``delta_width`` which is forwarded to
        :func:`delta` (default 9).

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_frames, 3 * n_mfcc)`` laid out as
        ``[mfcc | delta | delta-delta]``.

    Examples
    --------
    >>> import numpy as np
    >>> sr = 16000
    >>> x = np.sin(2 * np.pi * 220 * np.arange(sr // 2) / sr)
    >>> mfcc_with_deltas(x, sr).shape[1]
    39
    """
    width = int(kw.pop("delta_width", 9))
    base = mfcc(x, sr, **kw)
    d1 = delta(base, width=width)
    d2 = delta(d1, width=width)
    return np.hstack([base, d1, d2])


def cmn(X: np.ndarray) -> np.ndarray:
    """Cepstral mean normalisation: subtract the per-dimension mean.

    Removing the utterance mean cancels any constant convolutional channel
    (microphone, room transfer function) in the cepstral domain.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix of shape ``(n_frames, n_dim)``.

    Returns
    -------
    numpy.ndarray
        Mean-normalised copy of ``X``; an all-zero result for a single frame.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array of shape (n_frames, n_dim)")
    if X.shape[0] == 0:
        return X.copy()
    return X - X.mean(axis=0, keepdims=True)


def cvn(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Cepstral variance normalisation: scale each dimension to unit variance.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix of shape ``(n_frames, n_dim)``.
    eps : float, optional
        Added to the standard deviation to keep constant dimensions finite.

    Returns
    -------
    numpy.ndarray
        Variance-normalised copy of ``X``.  The mean is left untouched; use
        :func:`cmvn` to remove it as well.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array of shape (n_frames, n_dim)")
    if X.shape[0] == 0:
        return X.copy()
    return X / (X.std(axis=0, keepdims=True) + eps)


def cmvn(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Cepstral mean and variance normalisation.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix of shape ``(n_frames, n_dim)``.
    eps : float, optional
        Added to the standard deviation to keep constant dimensions finite.

    Returns
    -------
    numpy.ndarray
        Copy of ``X`` with zero mean and (approximately) unit variance along the
        frame axis.
    """
    return cvn(cmn(X), eps=eps)
