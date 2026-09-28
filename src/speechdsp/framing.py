"""Frame blocking and overlap-add reconstruction.

Short-time analysis splits a waveform into (usually overlapping) frames that are
short enough for the signal to be considered quasi-stationary.  Every
higher-level module in :mod:`speechdsp` is built on the primitives defined here, so
the frame/hop conventions are fixed in one place:

* frame ``t`` starts at sample ``t * hop`` and spans ``frame_len`` samples;
* only complete frames are produced -- a trailing partial frame is dropped;
* the analysis window, when given, is applied by :func:`enframe` and may be
  applied a second time by :func:`overlap_add` (weighted overlap-add).

References
----------
.. [1] L. R. Rabiner and R. W. Schafer, *Digital Processing of Speech Signals*,
       Prentice-Hall, 1978.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import get_window as _scipy_get_window

__all__ = [
    "enframe",
    "frame_time",
    "frame_to_sample",
    "num_frames",
    "overlap_add",
    "resolve_window",
    "window_sumsquare",
]


def resolve_window(window: str | np.ndarray | None, length: int) -> np.ndarray | None:
    """Normalise a window specification into a 1-D array.

    Parameters
    ----------
    window : str or numpy.ndarray or None
        Either the name of a window understood by :func:`scipy.signal.get_window`
        (a periodic window is requested, i.e. ``fftbins=True``), an explicit
        array of taps, or ``None`` for a rectangular window.
    length : int
        Required number of taps.

    Returns
    -------
    numpy.ndarray or None
        Window taps as ``float64``, or ``None`` when ``window`` is ``None``.

    Raises
    ------
    ValueError
        If an explicit window array does not have ``length`` taps.
    """
    if window is None:
        return None
    if isinstance(window, str):
        return np.asarray(_scipy_get_window(window, length, fftbins=True), dtype=np.float64)
    taps = np.asarray(window, dtype=np.float64).ravel()
    if taps.size != length:
        raise ValueError(f"window has {taps.size} taps but {length} are required")
    return taps


def num_frames(n_samples: int, frame_len: int, hop: int) -> int:
    """Number of complete frames obtainable from a signal.

    Parameters
    ----------
    n_samples : int
        Length of the signal in samples.
    frame_len : int
        Frame length in samples.
    hop : int
        Hop (frame advance) in samples.

    Returns
    -------
    int
        ``1 + (n_samples - frame_len) // hop``, or ``0`` when the signal is
        shorter than a single frame.
    """
    if frame_len <= 0:
        raise ValueError("frame_len must be positive")
    if hop <= 0:
        raise ValueError("hop must be positive")
    if n_samples < frame_len:
        return 0
    return 1 + (n_samples - frame_len) // hop


def enframe(
    x: np.ndarray,
    frame_len: int,
    hop: int,
    window: str | np.ndarray | None = None,
) -> np.ndarray:
    """Split a signal into overlapping frames.

    The frames are produced with a strided view and copied once, so the cost is
    a single contiguous allocation instead of a Python loop.

    Parameters
    ----------
    x : numpy.ndarray
        Input signal; flattened to 1-D.
    frame_len : int
        Frame length in samples.
    hop : int
        Hop size in samples.  ``hop == frame_len`` yields non-overlapping frames.
    window : str or numpy.ndarray or None, optional
        Analysis window applied to every frame.  ``None`` (default) means a
        rectangular window.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_frames, frame_len)`` and dtype ``float64``.  Samples
        that cannot fill a complete frame are discarded; if the signal is
        shorter than one frame the result has shape ``(0, frame_len)``.

    Examples
    --------
    >>> import numpy as np
    >>> enframe(np.arange(6.0), frame_len=3, hop=2)
    array([[0., 1., 2.],
           [2., 3., 4.]])
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = num_frames(x.size, frame_len, hop)
    if n == 0:
        return np.zeros((0, frame_len), dtype=np.float64)
    view = np.lib.stride_tricks.sliding_window_view(x, frame_len)[::hop]
    frames = np.array(view[:n], dtype=np.float64)
    taps = resolve_window(window, frame_len)
    if taps is not None:
        frames *= taps
    return frames


def overlap_add(
    frames: np.ndarray,
    hop: int,
    window: str | np.ndarray | None = None,
) -> np.ndarray:
    """Reassemble frames into a signal by (weighted) overlap-add.

    Parameters
    ----------
    frames : numpy.ndarray
        Array of shape ``(n_frames, frame_len)``.
    hop : int
        Hop size in samples, matching the one used for analysis.
    window : str or numpy.ndarray or None, optional
        Synthesis window applied before accumulation.  Passing the same window
        to :func:`enframe` and to this function implements weighted overlap-add,
        for which :func:`window_sumsquare` gives the normalisation envelope.

    Returns
    -------
    numpy.ndarray
        Signal of length ``(n_frames - 1) * hop + frame_len``; an empty array
        when ``frames`` holds no rows.

    Notes
    -----
    The accumulation uses :func:`numpy.bincount`, which is vectorised and
    handles the duplicated sample indices created by overlapping frames.
    """
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 2:
        raise ValueError("frames must be a 2-D array of shape (n_frames, frame_len)")
    if hop <= 0:
        raise ValueError("hop must be positive")
    n_frames, frame_len = frames.shape
    if n_frames == 0:
        return np.zeros(0, dtype=np.float64)
    taps = resolve_window(window, frame_len)
    if taps is not None:
        frames = frames * taps
    out_len = (n_frames - 1) * hop + frame_len
    idx = np.arange(n_frames)[:, None] * hop + np.arange(frame_len)[None, :]
    return np.bincount(idx.ravel(), weights=frames.ravel(), minlength=out_len)


def window_sumsquare(
    window: str | np.ndarray | None,
    n_frames: int,
    frame_len: int,
    hop: int,
) -> np.ndarray:
    """Overlap-add envelope of the squared analysis window.

    Parameters
    ----------
    window : str or numpy.ndarray or None
        Window specification, see :func:`resolve_window`.
    n_frames : int
        Number of overlapped frames.
    frame_len : int
        Frame length in samples.
    hop : int
        Hop size in samples.

    Returns
    -------
    numpy.ndarray
        ``sum_t w[n - t * hop] ** 2``, of length
        ``(n_frames - 1) * hop + frame_len``.  Dividing a weighted overlap-add
        signal by this envelope undoes the analysis/synthesis windowing wherever
        the envelope is non-zero (the NOLA condition).
    """
    taps = resolve_window(window, frame_len)
    sq = np.ones(frame_len, dtype=np.float64) if taps is None else taps**2
    return overlap_add(np.tile(sq, (max(int(n_frames), 0), 1)), hop)


def frame_to_sample(frame_idx: int, frame_len: int, hop: int) -> int:
    """Sample index at which a frame starts.

    Parameters
    ----------
    frame_idx : int
        Zero-based frame index.
    frame_len : int
        Frame length in samples (validated, and part of the frame convention:
        the frame spans ``[frame_idx * hop, frame_idx * hop + frame_len)``).
    hop : int
        Hop size in samples.

    Returns
    -------
    int
        ``frame_idx * hop``.
    """
    if frame_len <= 0:
        raise ValueError("frame_len must be positive")
    if hop <= 0:
        raise ValueError("hop must be positive")
    return int(frame_idx) * int(hop)


def frame_time(n_frames: int, frame_len: int, hop: int, sr: int) -> np.ndarray:
    """Centre time of each frame, in seconds.

    Parameters
    ----------
    n_frames : int
        Number of frames.
    frame_len : int
        Frame length in samples.
    hop : int
        Hop size in samples.
    sr : int
        Sampling rate in Hz.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_frames,)`` holding ``(t * hop + frame_len / 2) / sr``.
    """
    if sr <= 0:
        raise ValueError("sr must be positive")
    if frame_len <= 0 or hop <= 0:
        raise ValueError("frame_len and hop must be positive")
    return (np.arange(int(n_frames), dtype=np.float64) * hop + frame_len / 2.0) / float(sr)
