"""speechdsp -- signal processing, speech features and evaluation utilities.

The package is organised as seven small modules that build on one another:

==========================  ======================================================
:mod:`speechdsp.io`         WAV and HTK parameter file readers/writers
:mod:`speechdsp.framing`    frame blocking, overlap-add, frame/sample conversion
:mod:`speechdsp.spectral`   STFT, inverse STFT, spectrograms, CNN-ready images
:mod:`speechdsp.features`   pre-emphasis, mel filterbank, MFCC, deltas, CMVN
:mod:`speechdsp.vad`        short-time energy, zero-crossing rate, endpointing
:mod:`speechdsp.enhance`    log-MMSE and spectral-subtraction noise reduction
:mod:`speechdsp.metrics`    UAR, sensitivity/specificity, cross-validation reports
==========================  ======================================================

Everything is implemented on top of NumPy, SciPy and scikit-learn only; deep
learning back ends are optional extras and are never imported at package import
time.

Examples
--------
>>> import numpy as np
>>> import speechdsp
>>> sr = 16000
>>> x = np.sin(2 * np.pi * 440 * np.arange(sr) / sr)
>>> speechdsp.mfcc(x, sr).shape[1]
13
"""

from __future__ import annotations

from .enhance import log_mmse, spectral_subtraction
from .features import (
    cmn,
    cmvn,
    cvn,
    delta,
    hz_to_mel,
    mel_filterbank,
    mel_to_hz,
    mfcc,
    mfcc_with_deltas,
    preemphasis,
)
from .framing import enframe, frame_time, frame_to_sample, num_frames, overlap_add
from .io import read_htk, read_wav, write_htk, write_wav
from .metrics import confusion_report, cross_val_report, sensitivity_specificity, uar
from .spectral import istft, power_spectrum, spectrogram_db, spectrogram_image, stft
from .vad import endpoint_detect, frame_energy, trim_silence, zero_crossing_rate

__version__ = "0.1.0"
__author__ = "RL"

__all__ = [
    "__version__",
    "cmn",
    "cmvn",
    "confusion_report",
    "cross_val_report",
    "cvn",
    "delta",
    "endpoint_detect",
    "enframe",
    "frame_energy",
    "frame_time",
    "frame_to_sample",
    "hz_to_mel",
    "istft",
    "log_mmse",
    "mel_filterbank",
    "mel_to_hz",
    "mfcc",
    "mfcc_with_deltas",
    "num_frames",
    "overlap_add",
    "power_spectrum",
    "preemphasis",
    "read_htk",
    "read_wav",
    "sensitivity_specificity",
    "spectral_subtraction",
    "spectrogram_db",
    "spectrogram_image",
    "stft",
    "trim_silence",
    "uar",
    "write_htk",
    "write_wav",
    "zero_crossing_rate",
]
