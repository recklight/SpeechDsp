"""Single-channel speech enhancement in the short-time spectral domain.

Both estimators share the same skeleton: analyse the noisy signal with the STFT,
estimate the noise power from the leading frames (which are assumed to contain
background noise only), apply a real-valued gain to each spectral magnitude,
keep the noisy phase, and resynthesise by weighted overlap-add.

References
----------
.. [1] Y. Ephraim and D. Malah, "Speech enhancement using a minimum mean-square
       error log-spectral amplitude estimator", *IEEE Trans. ASSP*,
       33(2):443-445, 1985.
.. [2] Y. Ephraim and D. Malah, "Speech enhancement using a minimum mean-square
       error short-time spectral amplitude estimator", *IEEE Trans. ASSP*,
       32(6):1109-1121, 1984.
.. [3] S. F. Boll, "Suppression of acoustic noise in speech using spectral
       subtraction", *IEEE Trans. ASSP*, 27(2):113-120, 1979.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.special import exp1

from .spectral import istft, stft

__all__ = [
    "log_mmse",
    "spectral_subtraction",
]

_LOGGER = logging.getLogger(__name__)

#: Target analysis window length in seconds.
_WINDOW_SECONDS = 0.032

#: Smallest positive power used as a denominator floor.
_POWER_FLOOR = 1e-12

#: Clipping range for the a priori SNR, in linear power units (about -25..+40 dB).
_XI_MIN = 10.0**-2.5
_XI_MAX = 1.0e4

#: ``exp1`` underflows to zero well before this; clipping keeps it warning-free.
_VK_MAX = 500.0


def _analysis_sizes(sr: int) -> tuple[int, int]:
    """Pick a power-of-two FFT size near 32 ms and a 75 %-overlap hop."""
    n_fft = int(2 ** max(6, round(np.log2(_WINDOW_SECONDS * sr))))
    return n_fft, n_fft // 4


def _noise_power(mag: np.ndarray, noise_frames: int, tag: str) -> np.ndarray:
    """Average power spectrum of the leading frames, used as the noise estimate."""
    usable = int(min(max(noise_frames, 1), mag.shape[0]))
    if usable < noise_frames:
        _LOGGER.warning(
            "%s: only %d frames available for the noise estimate (requested %d)",
            tag,
            usable,
            noise_frames,
        )
    return np.maximum(np.mean(mag[:usable] ** 2, axis=0), _POWER_FLOOR)


def log_mmse(
    x: np.ndarray,
    sr: int,
    noise_frames: int = 6,
    alpha: float = 0.98,
) -> np.ndarray:
    """Log-MMSE short-time spectral amplitude estimator.

    The gain applied to bin ``k`` of frame ``t`` is

    ``G = xi / (1 + xi) * exp(0.5 * E1(v))``,  ``v = xi / (1 + xi) * gamma``,

    where ``gamma`` is the a posteriori SNR, ``xi`` the a priori SNR tracked by
    the decision-directed rule, and ``E1`` the exponential integral.  Compared
    with the plain MMSE-STSA estimator this minimises the error of the *log*
    spectral amplitude, which matches perceptual loudness far better and leaves
    much less musical noise.

    Parameters
    ----------
    x : numpy.ndarray
        Noisy waveform, flattened to 1-D.
    sr : int
        Sampling rate in Hz.
    noise_frames : int, optional
        Number of leading frames used to estimate the noise power spectrum,
        default 6.  They must contain background noise only.
    alpha : float, optional
        Smoothing factor of the decision-directed a priori SNR estimate,
        default 0.98.  Must lie in ``[0, 1)``.

    Returns
    -------
    numpy.ndarray
        Enhanced waveform with the same length as ``x``.

    Raises
    ------
    ValueError
        If ``alpha`` is outside ``[0, 1)`` or ``sr`` is not positive.

    Notes
    -----
    The recursion over frames is inherently sequential, but every frame is
    processed as a whole vector over frequency, so the Python loop runs once per
    frame rather than once per bin.

    References
    ----------
    .. [1] Y. Ephraim and D. Malah, "Speech enhancement using a minimum
           mean-square error log-spectral amplitude estimator", *IEEE Trans.
           ASSP*, 33(2):443-445, 1985.
    """
    if sr <= 0:
        raise ValueError("sr must be positive")
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"alpha must lie in [0, 1), got {alpha}")
    x = np.asarray(x, dtype=np.float64).ravel()
    n_fft, hop = _analysis_sizes(sr)
    if x.size < n_fft:
        _LOGGER.warning("signal shorter than one analysis window; returned unchanged")
        return x.copy()

    spec = stft(x, n_fft, hop, window="hann", center=True)
    mag = np.abs(spec)
    phase = np.angle(spec)
    noise_pow = _noise_power(mag, noise_frames, "log_mmse")

    gains = np.empty_like(mag)
    prev_clean_pow = noise_pow.copy()
    for t in range(mag.shape[0]):
        obs_pow = mag[t] ** 2
        gamma = np.minimum(obs_pow / noise_pow, _XI_MAX)
        xi = alpha * (prev_clean_pow / noise_pow) + (1.0 - alpha) * np.maximum(gamma - 1.0, 0.0)
        xi = np.clip(xi, _XI_MIN, _XI_MAX)

        ratio = xi / (1.0 + xi)
        vk = np.clip(ratio * gamma, _POWER_FLOOR, _VK_MAX)
        gain = ratio * np.exp(0.5 * exp1(vk))
        gain = np.clip(gain, 0.0, 1.0)

        gains[t] = gain
        prev_clean_pow = np.maximum((gain * mag[t]) ** 2, _POWER_FLOOR)

    enhanced = (gains * mag) * np.exp(1j * phase)
    y = istft(enhanced, n_fft, hop, window="hann", center=True)
    return _match_length(y, x.size)


def spectral_subtraction(
    x: np.ndarray,
    sr: int,
    noise_frames: int = 6,
    over_sub: float = 2.0,
    floor: float = 0.002,
) -> np.ndarray:
    """Power spectral subtraction with over-subtraction and a spectral floor.

    Parameters
    ----------
    x : numpy.ndarray
        Noisy waveform, flattened to 1-D.
    sr : int
        Sampling rate in Hz.
    noise_frames : int, optional
        Number of leading noise-only frames used for the noise estimate,
        default 6.
    over_sub : float, optional
        Over-subtraction factor, default 2.0.  Values above 1 remove more noise
        at the cost of more speech distortion.
    floor : float, optional
        Spectral floor as a fraction of the noisy power, default 0.002.  It
        keeps the residual from collapsing to zero, which is what produces
        "musical noise".

    Returns
    -------
    numpy.ndarray
        Enhanced waveform with the same length as ``x``.

    References
    ----------
    .. [1] S. F. Boll, "Suppression of acoustic noise in speech using spectral
           subtraction", *IEEE Trans. ASSP*, 27(2):113-120, 1979.
    .. [2] M. Berouti, R. Schwartz and J. Makhoul, "Enhancement of speech
           corrupted by acoustic noise", *ICASSP*, 4:208-211, 1979.
    """
    if sr <= 0:
        raise ValueError("sr must be positive")
    if over_sub < 0.0:
        raise ValueError("over_sub must be non-negative")
    if not 0.0 <= floor < 1.0:
        raise ValueError(f"floor must lie in [0, 1), got {floor}")
    x = np.asarray(x, dtype=np.float64).ravel()
    n_fft, hop = _analysis_sizes(sr)
    if x.size < n_fft:
        _LOGGER.warning("signal shorter than one analysis window; returned unchanged")
        return x.copy()

    spec = stft(x, n_fft, hop, window="hann", center=True)
    mag = np.abs(spec)
    noise_pow = _noise_power(mag, noise_frames, "spectral_subtraction")

    obs_pow = mag**2
    clean_pow = np.maximum(obs_pow - over_sub * noise_pow[None, :], floor * obs_pow)
    enhanced = np.sqrt(clean_pow) * np.exp(1j * np.angle(spec))
    y = istft(enhanced, n_fft, hop, window="hann", center=True)
    return _match_length(y, x.size)


def _match_length(y: np.ndarray, n: int) -> np.ndarray:
    """Trim or zero-pad a resynthesised signal to exactly ``n`` samples."""
    if y.size >= n:
        return np.ascontiguousarray(y[:n])
    return np.pad(y, (0, n - y.size))
