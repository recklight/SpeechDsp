"""Package-level contract: the public API other projects import."""

from __future__ import annotations

import importlib

import pytest

import speechdsp

PUBLIC_API = {
    "speechdsp.io": ["read_wav", "write_wav", "read_htk", "write_htk"],
    "speechdsp.framing": ["enframe", "overlap_add", "frame_to_sample", "frame_time"],
    "speechdsp.spectral": [
        "stft",
        "istft",
        "power_spectrum",
        "power_to_db",
        "spectrogram_db",
        "spectrogram_image",
    ],
    "speechdsp.features": [
        "preemphasis",
        "mel_filterbank",
        "mfcc",
        "delta",
        "mfcc_with_deltas",
        "cmn",
        "cvn",
        "cmvn",
    ],
    "speechdsp.vad": [
        "frame_energy",
        "zero_crossing_rate",
        "endpoint_detect",
        "trim_silence",
    ],
    "speechdsp.enhance": ["log_mmse", "spectral_subtraction"],
    "speechdsp.metrics": [
        "uar",
        "sensitivity_specificity",
        "confusion_report",
        "cross_val_report",
    ],
}


@pytest.mark.parametrize("module_name", sorted(PUBLIC_API))
def test_every_module_exposes_its_contract(module_name):
    module = importlib.import_module(module_name)
    for name in PUBLIC_API[module_name]:
        assert callable(getattr(module, name)), f"{module_name}.{name} is missing"


@pytest.mark.parametrize("name", sorted({n for names in PUBLIC_API.values() for n in names}))
def test_the_top_level_package_re_exports_everything(name):
    assert callable(getattr(speechdsp, name))
    assert name in speechdsp.__all__


def test_metadata():
    assert speechdsp.__version__ == "0.2.0"
    assert speechdsp.__author__ == "RL"


def test_importing_the_package_does_not_require_a_deep_learning_backend():
    import sys

    assert "torch" not in sys.modules


def test_every_public_callable_is_documented():
    for name in speechdsp.__all__:
        obj = getattr(speechdsp, name)
        if callable(obj):
            assert obj.__doc__, f"{name} has no docstring"
