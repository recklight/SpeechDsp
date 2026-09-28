"""Short-time Fourier analysis, synthesis and spectrogram rendering.

The STFT/ISTFT pair implemented here is exactly invertible: with a window that
satisfies the NOLA condition (the default periodic Hann window does, for any
``hop <= n_fft // 2``), ``istft(stft(x))`` reproduces ``x`` to floating point
accuracy, because synthesis divides the weighted overlap-add result by the
overlap-add envelope of the squared window rather than assuming a constant.

References
----------
.. [1] J. B. Allen and L. R. Rabiner, "A unified approach to short-time Fourier
       analysis and synthesis", *Proceedings of the IEEE*, 65(11):1558-1564, 1977.
.. [2] D. W. Griffin and J. S. Lim, "Signal estimation from modified short-time
       Fourier transform", *IEEE Trans. ASSP*, 32(2):236-243, 1984.
"""

from __future__ import annotations

import numpy as np

from .features import mel_filterbank
from .framing import enframe, overlap_add, resolve_window, window_sumsquare

__all__ = [
    "istft",
    "power_spectrum",
    "spectrogram_db",
    "spectrogram_image",
    "stft",
]

#: Floor applied before taking a logarithm of a power spectrum.
_POWER_FLOOR = 1e-12


def _center_pad(x: np.ndarray, pad: int) -> np.ndarray:
    """Pad both ends of ``x`` so that frame ``t`` is centred on sample ``t * hop``."""
    if pad <= 0:
        return x
    mode = "reflect" if x.size > 1 else "constant"
    return np.pad(x, pad, mode=mode)


def stft(
    x: np.ndarray,
    n_fft: int,
    hop: int,
    window: str | np.ndarray | None = "hann",
    center: bool = True,
) -> np.ndarray:
    """Short-time Fourier transform.

    Parameters
    ----------
    x : numpy.ndarray
        Real input signal, flattened to 1-D.
    n_fft : int
        FFT size, also used as the frame length.
    hop : int
        Hop size in samples.
    window : str or numpy.ndarray or None, optional
        Analysis window, default ``'hann'`` (periodic).
    center : bool, optional
        If ``True`` (default) the signal is reflection-padded by ``n_fft // 2``
        on both sides, so frame ``t`` is centred on sample ``t * hop``.

    Returns
    -------
    numpy.ndarray
        Complex array of shape ``(n_frames, n_fft // 2 + 1)``.

    Notes
    -----
    The signal is zero-padded on the right by up to ``hop - 1`` samples so that
    every input sample is covered by at least one complete frame; this is what
    makes :func:`istft` able to return the whole signal.
    """
    if n_fft <= 0:
        raise ValueError("n_fft must be positive")
    if hop <= 0:
        raise ValueError("hop must be positive")
    x = np.asarray(x, dtype=np.float64).ravel()
    taps = resolve_window(window, n_fft)
    if center:
        x = _center_pad(x, n_fft // 2)
    n = 1 if x.size <= n_fft else 1 + int(np.ceil((x.size - n_fft) / hop))
    total = (n - 1) * hop + n_fft
    if total > x.size:
        x = np.pad(x, (0, total - x.size))
    frames = enframe(x, n_fft, hop, taps)
    return np.fft.rfft(frames, n=n_fft, axis=-1)


def istft(
    S: np.ndarray,
    n_fft: int,
    hop: int,
    window: str | np.ndarray | None = "hann",
    center: bool = True,
) -> np.ndarray:
    """Inverse short-time Fourier transform (weighted overlap-add).

    Parameters
    ----------
    S : numpy.ndarray
        Complex spectrogram of shape ``(n_frames, n_fft // 2 + 1)``, as returned
        by :func:`stft`.
    n_fft : int
        FFT size used for the analysis.
    hop : int
        Hop size used for the analysis.
    window : str or numpy.ndarray or None, optional
        Window used for the analysis; the same window is applied at synthesis.
    center : bool, optional
        Must match the value used for the analysis; the centring pad is removed.

    Returns
    -------
    numpy.ndarray
        Real signal.  For an unmodified spectrogram the first ``len(x)`` samples
        equal the analysed signal to within floating point accuracy.

    Notes
    -----
    Samples where the squared-window envelope is numerically zero cannot be
    recovered and are set to zero.  With ``center=False`` and a window whose
    first tap is zero (the periodic Hann, for instance) this affects sample 0
    only; ``center=True`` hides that dead sample inside the padding, so the
    whole signal comes back exactly.
    """
    S = np.asarray(S)
    if S.ndim != 2:
        raise ValueError("S must be a 2-D array of shape (n_frames, n_fft // 2 + 1)")
    if S.shape[1] != n_fft // 2 + 1:
        raise ValueError(f"S has {S.shape[1]} bins but n_fft={n_fft} implies {n_fft // 2 + 1}")
    n_frames = S.shape[0]
    if n_frames == 0:
        return np.zeros(0, dtype=np.float64)
    taps = resolve_window(window, n_fft)
    frames = np.fft.irfft(S, n=n_fft, axis=-1)
    y = overlap_add(frames, hop, taps)
    env = window_sumsquare(taps, n_frames, n_fft, hop)
    good = env > 1e-10
    y[good] /= env[good]
    y[~good] = 0.0
    if center:
        pad = n_fft // 2
        y = y[pad : y.size - pad] if y.size > 2 * pad else np.zeros(0, dtype=np.float64)
    return y


def power_spectrum(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Short-time power spectrum (periodogram per frame).

    Parameters
    ----------
    x : numpy.ndarray
        Real input signal.
    n_fft : int
        FFT size / frame length.
    hop : int
        Hop size in samples.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_frames, n_fft // 2 + 1)`` holding
        ``abs(STFT) ** 2 / n_fft``.  A periodic Hann window is used and the
        frames are *not* centred, so frame ``t`` covers
        ``[t * hop, t * hop + n_fft)`` of the input.
    """
    S = stft(x, n_fft, hop, window="hann", center=False)
    return (np.abs(S) ** 2) / float(n_fft)


def spectrogram_db(
    x: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
    top_db: float = 80.0,
) -> np.ndarray:
    """Log-power spectrogram in decibels relative to its peak.

    Parameters
    ----------
    x : numpy.ndarray
        Real input signal.
    sr : int
        Sampling rate in Hz (validated; the returned values are level ratios and
        do not depend on it, but the argument keeps the spectral API uniform).
    n_fft : int
        FFT size / frame length.
    hop : int
        Hop size in samples.
    top_db : float, optional
        Dynamic range.  Values more than ``top_db`` below the peak are clamped.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_frames, n_fft // 2 + 1)`` with values in
        ``[-top_db, 0]``.
    """
    if sr <= 0:
        raise ValueError("sr must be positive")
    if top_db <= 0:
        raise ValueError("top_db must be positive")
    power = power_spectrum(x, n_fft, hop)
    ref = float(power.max()) if power.size else 0.0
    ref = max(ref, _POWER_FLOOR)
    db = 10.0 * np.log10(np.maximum(power, _POWER_FLOOR) / ref)
    return np.maximum(db, -float(top_db))


def spectrogram_image(
    x: np.ndarray,
    sr: int,
    shape: tuple[int, int] = (40, 98),
    n_fft: int = 512,
    hop: int | None = None,
) -> np.ndarray:
    """Render a fixed-size mel spectrogram as an 8-bit grey-scale image.

    Convolutional models expect a constant input size regardless of utterance
    length; the time axis is therefore resampled by linear interpolation to
    exactly ``shape[1]`` columns.

    Parameters
    ----------
    x : numpy.ndarray
        Real input signal.
    sr : int
        Sampling rate in Hz.
    shape : tuple of int, optional
        ``(n_mels, n_frames)`` of the output image, default ``(40, 98)``.
    n_fft : int, optional
        FFT size, default 512.
    hop : int or None, optional
        Hop size.  When ``None`` (default) it is derived from the signal length
        so that the analysis produces roughly ``shape[1]`` frames.

    Returns
    -------
    numpy.ndarray
        ``uint8`` array of shape ``shape``; row 0 is the lowest mel band and
        column 0 the earliest frame.  Level 0 corresponds to ``-80`` dB relative
        to the peak of the utterance and level 255 to the peak itself.

    References
    ----------
    .. [1] S. B. Davis and P. Mermelstein, "Comparison of parametric
           representations for monosyllabic word recognition in continuously
           spoken sentences", *IEEE Trans. ASSP*, 28(4):357-366, 1980.
    """
    if sr <= 0:
        raise ValueError("sr must be positive")
    n_mels, n_cols = int(shape[0]), int(shape[1])
    if n_mels <= 0 or n_cols <= 0:
        raise ValueError("shape entries must be positive")
    x = np.asarray(x, dtype=np.float64).ravel()
    if hop is None:
        hop = max(1, round(float(max(x.size - n_fft, n_fft)) / float(n_cols)))
    top_db = 80.0
    power = power_spectrum(x, n_fft, hop)
    fb = mel_filterbank(sr, n_fft, n_mels=n_mels)
    mel = power @ fb.T
    ref = max(float(mel.max()) if mel.size else 0.0, _POWER_FLOOR)
    db = 10.0 * np.log10(np.maximum(mel, _POWER_FLOOR) / ref)
    db = np.maximum(db, -top_db).T  # (n_mels, n_frames)

    n_in = db.shape[1]
    if n_in == 0:
        db = np.full((n_mels, 1), -top_db)
        n_in = 1
    src = np.linspace(0.0, n_in - 1, n_cols)
    lo = np.floor(src).astype(np.intp)
    hi = np.minimum(lo + 1, n_in - 1)
    frac = src - lo
    resampled = db[:, lo] * (1.0 - frac) + db[:, hi] * frac

    scaled = (resampled + top_db) * (255.0 / top_db)
    return np.round(np.clip(scaled, 0.0, 255.0)).astype(np.uint8)
