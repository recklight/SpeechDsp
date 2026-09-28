"""Reading and writing waveforms and HTK parameter files.

Two container formats are supported:

* RIFF/WAVE, through :mod:`scipy.io.wavfile`, always presented to the caller as
  mono ``float64`` in ``[-1, 1]`` regardless of the on-disk sample format;
* HTK parameter files (``.mfc``, ``.fbank``, ...), a 12-byte big-endian header
  followed by big-endian ``float32`` data.  Some toolchains emit the same
  feature matrices as plain ASCII text, so :func:`read_htk` detects and accepts
  that variant too.

References
----------
.. [1] S. Young et al., *The HTK Book (version 3.4)*, Cambridge University
       Engineering Department, 2006, chapter 5.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile

__all__ = [
    "HTK_HEADER_SIZE",
    "read_htk",
    "read_wav",
    "write_htk",
    "write_wav",
]

_LOGGER = logging.getLogger(__name__)

#: Size in bytes of the fixed HTK parameter-file header.
HTK_HEADER_SIZE = 12

#: Big-endian header layout: nSamples, sampPeriod, sampSize, parmKind.
_HTK_HEADER_DTYPE = np.dtype(
    [
        ("n_samples", ">i4"),
        ("samp_period", ">i4"),
        ("samp_size", ">i2"),
        ("parm_kind", ">i2"),
    ]
)

#: Scaling factors used to map integer PCM to ``[-1, 1)``.
_PCM_SCALE: dict[str, float] = {
    "int16": 32768.0,
    "int32": 2147483648.0,
}

#: Number of bytes inspected when guessing whether a file is ASCII text.
_SNIFF_BYTES = 512


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a WAV file as a mono ``float64`` waveform.

    Integer PCM is divided by the full-scale value of its width, unsigned 8-bit
    PCM is re-centred first, and multi-channel files are averaged down to mono.

    Parameters
    ----------
    path : str or pathlib.Path
        Path of the WAV file.

    Returns
    -------
    x : numpy.ndarray
        1-D ``float64`` signal, nominally in ``[-1, 1]``.
    sr : int
        Sampling rate in Hz.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the sample format is not supported.

    See Also
    --------
    write_wav : the matching writer.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such WAV file: {path}")
    sr, raw = wavfile.read(path)
    data = np.asarray(raw)
    # The on-disk sample format decides the scaling, so it has to be read before
    # the channel mix-down promotes the array to float.
    kind = data.dtype.name
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.ravel()

    if kind == "uint8":
        x = (data.astype(np.float64) - 128.0) / 128.0
    elif kind in _PCM_SCALE:
        x = data.astype(np.float64) / _PCM_SCALE[kind]
    elif kind in ("float32", "float64"):
        x = data.astype(np.float64)
    else:
        raise ValueError(f"unsupported WAV sample format: {kind}")
    return np.ascontiguousarray(x), int(sr)


def write_wav(path: str | Path, x: np.ndarray, sr: int) -> None:
    """Write a waveform as 16-bit PCM WAV.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination path; parent directories are created if needed.
    x : numpy.ndarray
        Mono signal in ``[-1, 1]``.  Samples outside the range are clipped and a
        warning is logged.
    sr : int
        Sampling rate in Hz.

    Returns
    -------
    None

    Notes
    -----
    16-bit PCM is the most portable representation and the one every downstream
    tool in this project expects; round-tripping through :func:`read_wav`
    therefore introduces a quantisation error of about ``3e-5``.
    """
    if sr <= 0:
        raise ValueError("sr must be positive")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=np.float64).ravel()
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.0:
        _LOGGER.warning("signal peak %.3f exceeds full scale and will be clipped", peak)
    pcm = np.round(np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
    wavfile.write(path, int(sr), pcm)


def _looks_like_text(raw: bytes) -> bool:
    """Heuristically decide whether a parameter file holds ASCII numbers."""
    head = raw[:_SNIFF_BYTES]
    if not head:
        return False
    try:
        text = head.decode("ascii")
    except UnicodeDecodeError:
        return False
    return all(ch.isprintable() or ch in "\r\n\t" for ch in text)


def _read_htk_text(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a whitespace-delimited ASCII feature matrix."""
    feats = np.loadtxt(path, dtype=np.float64, ndmin=2)
    n_frames, n_dim = feats.shape
    header: dict[str, Any] = {
        "n_samples": int(n_frames),
        "samp_period": 100000,
        "samp_size": int(n_dim * 4),
        "parm_kind": 9,  # USER
        "n_dim": int(n_dim),
        "format": "text",
    }
    return feats, header


def read_htk(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read an HTK parameter file.

    Binary files start with a 12-byte big-endian header (``nSamples`` int32,
    ``sampPeriod`` int32, ``sampSize`` int16, ``parmKind`` int16) followed by
    ``nSamples * sampSize / 4`` big-endian ``float32`` values.  Files that
    contain only printable ASCII are parsed as a plain numeric matrix instead.

    Parameters
    ----------
    path : str or pathlib.Path
        Path of the parameter file.

    Returns
    -------
    feats : numpy.ndarray
        ``float64`` array of shape ``(n_frames, n_dim)``.
    header : dict
        Keys ``n_samples``, ``samp_period`` (in 100 ns units), ``samp_size``
        (bytes per frame), ``parm_kind``, ``n_dim`` and ``format``
        (``'binary'`` or ``'text'``).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the header is inconsistent with the file size.

    See Also
    --------
    write_htk : the matching writer.

    References
    ----------
    .. [1] S. Young et al., *The HTK Book (version 3.4)*, Cambridge University
           Engineering Department, 2006, chapter 5.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such HTK file: {path}")
    raw = path.read_bytes()
    if _looks_like_text(raw):
        return _read_htk_text(path)
    if len(raw) < HTK_HEADER_SIZE:
        raise ValueError(f"{path} is too short to contain an HTK header")

    fields = np.frombuffer(raw[:HTK_HEADER_SIZE], dtype=_HTK_HEADER_DTYPE)[0]
    n_samples = int(fields["n_samples"])
    samp_period = int(fields["samp_period"])
    samp_size = int(fields["samp_size"])
    parm_kind = int(fields["parm_kind"])
    if n_samples < 0 or samp_size <= 0 or samp_size % 4 != 0:
        raise ValueError(
            f"{path}: implausible HTK header "
            f"(n_samples={n_samples}, samp_size={samp_size}); is the byte order correct?"
        )

    n_dim = samp_size // 4
    expected = HTK_HEADER_SIZE + n_samples * samp_size
    if len(raw) != expected:
        raise ValueError(f"{path}: header announces {expected} bytes but the file holds {len(raw)}")

    body = np.frombuffer(raw, dtype=">f4", count=n_samples * n_dim, offset=HTK_HEADER_SIZE)
    feats = body.astype(np.float64).reshape(n_samples, n_dim)
    header: dict[str, Any] = {
        "n_samples": n_samples,
        "samp_period": samp_period,
        "samp_size": samp_size,
        "parm_kind": parm_kind,
        "n_dim": n_dim,
        "format": "binary",
    }
    return feats, header


def write_htk(
    path: str | Path,
    feats: np.ndarray,
    period_100ns: int = 100000,
    kind: int = 6,
) -> None:
    """Write a feature matrix as a binary HTK parameter file.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination path; parent directories are created if needed.
    feats : numpy.ndarray
        Feature matrix of shape ``(n_frames, n_dim)``.
    period_100ns : int, optional
        Frame period in 100 ns units, default 100000 (10 ms).
    kind : int, optional
        HTK ``parmKind`` code, default 6 (``MFCC``).  Qualifiers are added as
        bit flags, e.g. ``6 + 0o100`` for ``MFCC_E``.

    Returns
    -------
    None

    Notes
    -----
    Values are stored as big-endian ``float32``, so a round trip through
    :func:`read_htk` is exact only to single precision.

    References
    ----------
    .. [1] S. Young et al., *The HTK Book (version 3.4)*, Cambridge University
           Engineering Department, 2006, chapter 5.
    """
    feats = np.asarray(feats, dtype=np.float64)
    if feats.ndim != 2:
        raise ValueError("feats must be a 2-D array of shape (n_frames, n_dim)")
    n_samples, n_dim = feats.shape
    samp_size = n_dim * 4
    if samp_size > np.iinfo(np.int16).max:
        raise ValueError(f"n_dim={n_dim} exceeds what the HTK sampSize field can hold")

    header = np.zeros(1, dtype=_HTK_HEADER_DTYPE)
    header["n_samples"] = n_samples
    header["samp_period"] = int(period_100ns)
    header["samp_size"] = samp_size
    header["parm_kind"] = int(kind)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(header.tobytes())
        fh.write(feats.astype(">f4").tobytes())
