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
    "power_to_db",
    "spectrogram_db",
    "spectrogram_image",
    "stft",
]


def _center_pad(x: np.ndarray, pad: int) -> np.ndarray:
    """Pad both ends of ``x`` so that frame ``t`` is centred on sample ``t * hop``."""
    if pad <= 0:
        return x
    mode = "reflect" if x.size > 1 else "constant"
    return np.pad(x, pad, mode=mode)


def _analysis_window(
    window: str | np.ndarray | None, n_fft: int, win_length: int | None
) -> np.ndarray | None:
    """Window taps of length ``n_fft``, zero-padded on both sides when shorter.

    ``win_length`` taps are built (``None`` keeps a rectangular window) and
    centred inside the ``n_fft``-sample frame, so analysing with a short window
    and a long FFT is the same as zero-padding each windowed frame.
    """
    if win_length is None or win_length == n_fft:
        return resolve_window(window, n_fft)
    if not 0 < win_length <= n_fft:
        raise ValueError(f"win_length must be in [1, n_fft={n_fft}], got {win_length}")
    taps = resolve_window(window, win_length)
    if taps is None:
        taps = np.ones(win_length, dtype=np.float64)
    left = (n_fft - win_length) // 2
    return np.pad(taps, (left, n_fft - win_length - left))


def stft(
    x: np.ndarray,
    n_fft: int,
    hop: int,
    window: str | np.ndarray | None = "hann",
    center: bool = True,
    win_length: int | None = None,
) -> np.ndarray:
    """Short-time Fourier transform.

    Parameters
    ----------
    x : numpy.ndarray
        Real input signal, flattened to 1-D.
    n_fft : int
        FFT size.  Also the frame length unless ``win_length`` is given.
    hop : int
        Hop size in samples.
    window : str or numpy.ndarray or None, optional
        Analysis window, default ``'hann'`` (periodic).  An explicit array must
        have ``win_length`` taps (``n_fft`` when ``win_length`` is ``None``).
    center : bool, optional
        If ``True`` (default) the signal is reflection-padded by ``n_fft // 2``
        on both sides, so frame ``t`` is centred on sample ``t * hop``.
    win_length : int or None, optional
        Window length in samples, ``1 <= win_length <= n_fft``.  The window is
        centred inside the ``n_fft``-sample frame and zero elsewhere, which is
        the same as zero-padding every windowed frame to ``n_fft`` points (for
        example 25 ms frames analysed with a 512-point FFT).  ``None`` (default)
        uses ``n_fft``.

    Returns
    -------
    numpy.ndarray
        Complex array of shape ``(n_frames, n_fft // 2 + 1)``.

    Notes
    -----
    The signal is zero-padded on the right by up to ``hop - 1`` samples so that
    every input sample is covered by at least one complete frame; this is what
    makes :func:`istft` able to return the whole signal.

    Examples
    --------
    >>> import numpy as np
    >>> stft(np.zeros(1600), n_fft=512, hop=160, win_length=400).shape
    (11, 257)
    """
    if n_fft <= 0:
        raise ValueError("n_fft must be positive")
    if hop <= 0:
        raise ValueError("hop must be positive")
    x = np.asarray(x, dtype=np.float64).ravel()
    taps = _analysis_window(window, n_fft, win_length)
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
    win_length: int | None = None,
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
    win_length : int or None, optional
        Must match the value used for the analysis.  With a ``hop`` larger than
        ``win_length`` the windows no longer overlap and the samples between
        them are lost.

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
    whole signal comes back exactly.  A ``win_length`` shorter than ``n_fft``
    also leaves samples outside every window in an uncentred analysis: the
    first ``(n_fft - win_length) // 2`` (plus the dead sample of a window that
    starts at zero), and at the end at most
    ``n_fft - win_length - (n_fft - win_length) // 2``, depending on how far the
    last frame runs past the signal.
    """
    S = np.asarray(S)
    if S.ndim != 2:
        raise ValueError("S must be a 2-D array of shape (n_frames, n_fft // 2 + 1)")
    if S.shape[1] != n_fft // 2 + 1:
        raise ValueError(f"S has {S.shape[1]} bins but n_fft={n_fft} implies {n_fft // 2 + 1}")
    n_frames = S.shape[0]
    if n_frames == 0:
        return np.zeros(0, dtype=np.float64)
    taps = _analysis_window(window, n_fft, win_length)
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


def power_spectrum(
    x: np.ndarray, n_fft: int, hop: int, win_length: int | None = None
) -> np.ndarray:
    """Short-time power spectrum (periodogram per frame).

    Parameters
    ----------
    x : numpy.ndarray
        Real input signal.
    n_fft : int
        FFT size.  Also the frame length unless ``win_length`` is given.
    hop : int
        Hop size in samples.
    win_length : int or None, optional
        Window length, see :func:`stft`.  ``None`` (default) uses ``n_fft``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_frames, n_fft // 2 + 1)`` holding
        ``abs(STFT) ** 2 / n_fft``.  A periodic Hann window is used and the
        frames are *not* centred, so frame ``t`` covers
        ``[t * hop, t * hop + n_fft)`` of the input; with a shorter
        ``win_length`` the window occupies the middle of that span.
    """
    S = stft(x, n_fft, hop, window="hann", center=False, win_length=win_length)
    return (np.abs(S) ** 2) / float(n_fft)


def power_to_db(power: np.ndarray, top_db: float = 80.0) -> np.ndarray:
    """Convert power values to decibels relative to their peak.

    The floor is relative: every value more than ``top_db`` below the peak is
    clamped to ``-top_db``, however quiet the input is, so the full dynamic
    range is kept for low-level recordings.  Input without any energy (all
    zeros) has no peak to refer to and maps to ``-top_db`` everywhere.

    Parameters
    ----------
    power : numpy.ndarray
        Non-negative power values of any shape.
    top_db : float, optional
        Dynamic range in dB, default 80.

    Returns
    -------
    numpy.ndarray
        ``float64`` array of the same shape with values in ``[-top_db, 0]``.

    Raises
    ------
    ValueError
        If ``top_db`` is not positive, or ``power`` holds a negative, NaN or
        infinite value.  Such input has no meaningful level, and mapping it to
        the floor would make it look like silence.

    Examples
    --------
    >>> import numpy as np
    >>> power_to_db(np.array([1e-6, 1e-8, 1e-20]))
    array([  0., -20., -80.])
    >>> power_to_db(np.zeros(3))
    array([-80., -80., -80.])
    """
    if top_db <= 0:
        raise ValueError("top_db must be positive")
    power = np.asarray(power, dtype=np.float64)
    if not np.all(np.isfinite(power)):
        raise ValueError("power must be finite, but it holds NaN or infinite values")
    if power.size and np.any(power < 0):
        raise ValueError("power must be non-negative")
    peak = float(power.max()) if power.size else 0.0
    if peak <= 0.0:
        return np.full(power.shape, -float(top_db))
    floor = peak * 10.0 ** (-float(top_db) / 10.0)
    return 10.0 * np.log10(np.maximum(power, floor) / peak)


def spectrogram_db(
    x: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
    top_db: float = 80.0,
    win_length: int | None = None,
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
        FFT size.  Also the frame length unless ``win_length`` is given.
    hop : int
        Hop size in samples.
    top_db : float, optional
        Dynamic range.  Values more than ``top_db`` below the peak are clamped.
    win_length : int or None, optional
        Window length, see :func:`stft`.  ``None`` (default) uses ``n_fft``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_frames, n_fft // 2 + 1)`` with values in
        ``[-top_db, 0]``, computed by :func:`power_to_db`; a silent input is
        ``-top_db`` everywhere.
    """
    if sr <= 0:
        raise ValueError("sr must be positive")
    if top_db <= 0:
        raise ValueError("top_db must be positive")
    return power_to_db(power_spectrum(x, n_fft, hop, win_length=win_length), top_db)


def spectrogram_image(
    x: np.ndarray,
    sr: int,
    shape: tuple[int, int] = (40, 98),
    n_fft: int = 512,
    hop: int | None = None,
    win_length: int | None = None,
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
    win_length : int or None, optional
        Window length, see :func:`stft`.  ``None`` (default) uses ``n_fft``.

    Returns
    -------
    numpy.ndarray
        ``uint8`` array of shape ``shape``; row 0 is the lowest mel band and
        column 0 the earliest frame.  Level 0 corresponds to ``-80`` dB relative
        to the peak of the utterance and level 255 to the peak itself.  A silent
        input renders as an all-zero (black) image.

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
    power = power_spectrum(x, n_fft, hop, win_length=win_length)
    fb = mel_filterbank(sr, n_fft, n_mels=n_mels)
    db = power_to_db(power @ fb.T, top_db).T  # (n_mels, n_frames)

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
