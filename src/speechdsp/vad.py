"""Energy and zero-crossing based voice activity detection.

:func:`endpoint_detect` implements the classical two-threshold endpoint
detector: a coarse search on short-time energy locates the speech interval, and
the boundaries are then refined outwards using the zero-crossing rate, which
catches low-energy unvoiced onsets such as ``/s/`` or ``/f/``.

References
----------
.. [1] L. R. Rabiner and M. R. Sambur, "An algorithm for determining the
       endpoints of isolated utterances", *Bell System Technical Journal*,
       54(2):297-315, 1975.
.. [2] L. R. Rabiner and R. W. Schafer, *Digital Processing of Speech Signals*,
       Prentice-Hall, 1978.
"""

from __future__ import annotations

import logging

import numpy as np

from .framing import enframe, num_frames

__all__ = [
    "endpoint_detect",
    "frame_energy",
    "trim_silence",
    "zero_crossing_rate",
]

_LOGGER = logging.getLogger(__name__)

#: Number of leading frames assumed to contain background noise only.
_NOISE_FRAMES = 10

#: Zero-crossing rate above which a frame looks fricative-like.  Rabiner &
#: Sambur quote 25 crossings per 10 ms frame at 10 kHz, i.e. a rate of 0.25.
_ZCR_FRICATIVE = 0.25

#: A frame may only extend a boundary if its energy exceeds the background
#: estimate by this factor, which stops broadband background noise (whose
#: zero-crossing rate is close to 0.5) from dragging the boundaries outwards.
_ZCR_ENERGY_GATE = 2.0

#: How far the zero-crossing refinement may extend the boundaries, in frames.
_ZCR_SEARCH_FRAMES = 25

#: How many high-ZCR frames must be found before a boundary is moved.
_ZCR_MIN_HITS = 3


def frame_energy(frames: np.ndarray) -> np.ndarray:
    """Short-time energy of each frame.

    Parameters
    ----------
    frames : numpy.ndarray
        Array of shape ``(n_frames, frame_len)``, e.g. from
        :func:`speechdsp.framing.enframe`.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_frames,)`` holding ``sum(frame ** 2)``.

    Examples
    --------
    >>> import numpy as np
    >>> frame_energy(np.ones((2, 4)))
    array([4., 4.])
    """
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 2:
        raise ValueError("frames must be a 2-D array of shape (n_frames, frame_len)")
    return np.sum(frames**2, axis=1)


def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    """Fraction of adjacent sample pairs in each frame that change sign.

    Parameters
    ----------
    frames : numpy.ndarray
        Array of shape ``(n_frames, frame_len)``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_frames,)`` with values in ``[0, 1]``.

    Notes
    -----
    The sign is taken with :func:`numpy.signbit`, so exact zeros are treated as
    belonging to the preceding sign rather than counting as two crossings.
    """
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 2:
        raise ValueError("frames must be a 2-D array of shape (n_frames, frame_len)")
    if frames.shape[1] < 2:
        return np.zeros(frames.shape[0], dtype=np.float64)
    crossings = np.count_nonzero(np.diff(np.signbit(frames), axis=1), axis=1)
    return crossings / float(frames.shape[1] - 1)


def _refine_with_zcr(
    boundary: int,
    zcr: np.ndarray,
    energy: np.ndarray,
    zcr_threshold: float,
    energy_gate: float,
    backwards: bool,
) -> int:
    """Move a boundary outwards over unvoiced frames that energy alone missed.

    Searching away from the detected speech region, a frame counts as a hit when
    its zero-crossing rate exceeds ``zcr_threshold`` *and* its energy exceeds
    ``energy_gate``.  If at least ``_ZCR_MIN_HITS`` hits are found within
    ``_ZCR_SEARCH_FRAMES``, the boundary moves to the outermost one.
    """
    step = -1 if backwards else 1
    hits: list[int] = []
    for offset in range(1, _ZCR_SEARCH_FRAMES + 1):
        idx = boundary + step * offset
        if not 0 <= idx < zcr.size:
            break
        if zcr[idx] > zcr_threshold and energy[idx] > energy_gate:
            hits.append(idx)
    if len(hits) >= _ZCR_MIN_HITS:
        return hits[-1]
    return boundary


def endpoint_detect(
    x: np.ndarray,
    sr: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> tuple[int, int]:
    """Locate the first and last speech sample with the two-threshold algorithm.

    The upper energy threshold ``ITU`` must be exceeded for a frame to be
    declared speech; from there the boundary is walked back to the last frame
    above the lower threshold ``ITL``.  Both thresholds are derived from the
    energy of the leading (noise-only) frames and the peak energy, so the
    detector is independent of absolute recording level.

    Parameters
    ----------
    x : numpy.ndarray
        Input waveform, flattened to 1-D.
    sr : int
        Sampling rate in Hz.
    frame_ms : float, optional
        Frame length in milliseconds, default 25.
    hop_ms : float, optional
        Frame advance in milliseconds, default 10.

    Returns
    -------
    start : int
        Index of the first speech sample.
    end : int
        Index one past the last speech sample, so ``x[start:end]`` is the speech
        segment.  When no speech is found the whole signal is returned, i.e.
        ``(0, len(x))``.

    References
    ----------
    .. [1] L. R. Rabiner and M. R. Sambur, "An algorithm for determining the
           endpoints of isolated utterances", *Bell System Technical Journal*,
           54(2):297-315, 1975.
    """
    if sr <= 0:
        raise ValueError("sr must be positive")
    x = np.asarray(x, dtype=np.float64).ravel()
    frame_len = round(float(sr * frame_ms / 1000.0))
    hop = round(float(sr * hop_ms / 1000.0))
    if frame_len <= 0 or hop <= 0:
        raise ValueError("frame_ms and hop_ms are too short for this sampling rate")
    if num_frames(x.size, frame_len, hop) < 3:
        _LOGGER.debug("signal too short for endpoint detection; returning the whole signal")
        return 0, int(x.size)

    frames = enframe(x, frame_len, hop)
    energy = frame_energy(frames)
    zcr = zero_crossing_rate(frames)
    n_frames = energy.size

    n_noise = min(_NOISE_FRAMES, max(1, n_frames // 10))
    noise_energy = float(np.mean(energy[:n_noise]))
    peak_energy = float(energy.max())
    if peak_energy <= 0.0:
        return 0, int(x.size)

    # Rabiner & Sambur: ITL is the smaller of a peak-relative and a
    # noise-relative candidate; ITU is five times ITL.
    itl = min(0.03 * (peak_energy - noise_energy) + noise_energy, 4.0 * noise_energy)
    itl = max(itl, 1e-12 * peak_energy)
    itu = 5.0 * itl

    above_itu = np.flatnonzero(energy > itu)
    if above_itu.size == 0:
        _LOGGER.debug("no frame exceeds the upper energy threshold; returning the whole signal")
        return 0, int(x.size)
    first, last = int(above_itu[0]), int(above_itu[-1])

    while first > 0 and energy[first - 1] > itl:
        first -= 1
    while last < n_frames - 1 and energy[last + 1] > itl:
        last += 1

    noise_zcr = zcr[:n_noise]
    zcr_threshold = max(
        float(np.mean(noise_zcr) + 2.0 * np.std(noise_zcr)),
        _ZCR_FRICATIVE,
    )
    energy_gate = max(_ZCR_ENERGY_GATE * noise_energy, 1e-12 * peak_energy)
    first = _refine_with_zcr(first, zcr, energy, zcr_threshold, energy_gate, backwards=True)
    last = _refine_with_zcr(last, zcr, energy, zcr_threshold, energy_gate, backwards=False)

    start = max(0, first * hop)
    end = min(int(x.size), last * hop + frame_len)
    if end <= start:
        return 0, int(x.size)
    return int(start), int(end)


def trim_silence(x: np.ndarray, sr: int, **kw: float) -> np.ndarray:
    """Return the speech segment of a waveform, with leading/trailing silence cut.

    Parameters
    ----------
    x : numpy.ndarray
        Input waveform.
    sr : int
        Sampling rate in Hz.
    **kw
        Forwarded to :func:`endpoint_detect` (``frame_ms``, ``hop_ms``).

    Returns
    -------
    numpy.ndarray
        Copy of ``x[start:end]``.  If no speech is detected the whole signal is
        returned unchanged.

    See Also
    --------
    endpoint_detect : the underlying detector.
    """
    start, end = endpoint_detect(x, sr, **kw)
    return np.asarray(x, dtype=np.float64).ravel()[start:end].copy()
