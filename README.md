# SpeechDsp

[![PyPI](https://img.shields.io/pypi/v/speechdsp.svg)](https://pypi.org/project/speechdsp/)
[![CI](https://github.com/recklight/SpeechDsp/actions/workflows/ci.yml/badge.svg)](https://github.com/recklight/SpeechDsp/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

English | [繁體中文](README.zh-TW.md)

> The toolbox my speech projects share: signal processing, feature extraction and model evaluation

`speechdsp` is a Python package I wrote from scratch on top of NumPy, SciPy and
scikit-learn. It covers the front-end steps that come up day to day in speech and
acoustics research: reading and writing audio and feature files, framing, the
short-time Fourier transform, MFCCs and their deltas, endpoint detection, speech
enhancement, and evaluation metrics suited to class-imbalanced data.

I kept it free of `librosa`, `soundfile` and any deep learning framework on purpose,
and implemented every algorithm myself with `numpy` + `scipy`, so it is easy to deploy
where nothing else is installed, such as a cluster node or a clean conda environment.

- Author: RL
- License: MIT
- Python: 3.10 or newer

---

## Contents

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [Modules](#modules)
   - [speechdsp.io — audio and feature file I/O](#speechdspio--audio-and-feature-file-io)
   - [speechdsp.framing — framing and overlap-add](#speechdspframing--framing-and-overlap-add)
   - [speechdsp.spectral — short-time spectral analysis](#speechdspspectral--short-time-spectral-analysis)
   - [speechdsp.features — cepstral features](#speechdspfeatures--cepstral-features)
   - [speechdsp.vad — endpoint detection](#speechdspvad--endpoint-detection)
   - [speechdsp.enhance — speech enhancement](#speechdspenhance--speech-enhancement)
   - [speechdsp.metrics — evaluation metrics](#speechdspmetrics--evaluation-metrics)
4. [API quick reference](#api-quick-reference)
5. [Data paths](#data-paths)
6. [Tests and linting](#tests-and-linting)
7. [Design principles](#design-principles)
8. [References](#references)
9. [Contributing and citing](#contributing-and-citing)

---

## Installation

From PyPI:

```bash
pip install speechdsp
```

To work on the source, install it in editable mode, preferably inside a virtual
environment:

```bash
git clone https://github.com/recklight/SpeechDsp.git
cd SpeechDsp
python -m pip install -e .
```

With the development and test tools as well:

```bash
python -m pip install -e ".[dev]"
```

### Dependencies

| Kind | Package | Minimum version |
| --- | --- | --- |
| Core | `numpy` | 1.24 |
| Core | `scipy` | 1.10 |
| Core | `scikit-learn` | 1.2 |
| Optional `[plot]` | `matplotlib` | 3.6 |
| Optional `[dl]` | `torch` | 2.2 |
| Optional `[dev]` | `pytest`, `ruff` | — |

The core needs just these three scientific packages. `torch` is an optional extra, and
`speechdsp` never imports it at load time, installed or not.

### Using it from another project

Another project can point straight at this folder as a path dependency, for example in
that project's `pyproject.toml`:

```toml
[project]
dependencies = ["speechdsp"]

[tool.uv.sources]
speechdsp = { path = "../SpeechDsp", editable = true }
```

Or just install it into the same environment first:

```bash
python -m pip install -e /path/to/SpeechDsp
```

---

## Quick start

```python
import numpy as np
import speechdsp

# Read a WAV file: a float64 mono signal (range -1 to 1) and its sample rate
x, sr = speechdsp.read_wav("example.wav")

# Trim leading and trailing silence
x = speechdsp.trim_silence(x, sr)

# Speech enhancement (log-MMSE)
x = speechdsp.log_mmse(x, sr, noise_frames=6)

# 39-dimensional MFCCs (13 static + first-order deltas + second-order deltas)
feats = speechdsp.mfcc_with_deltas(x, sr, n_mfcc=13)

# Cepstral mean and variance normalization
feats = speechdsp.cmvn(feats)

print(feats.shape)   # (frames, 39)
```

---

## Modules

### speechdsp.io — audio and feature file I/O

Reads and writes WAV files and HTK feature files. Whether the file on disk holds 16-bit
PCM, 32-bit PCM or floating point, `read_wav` always returns a `float64` mono signal;
multichannel audio is averaged down to mono.

```python
from speechdsp.io import read_wav, write_wav, read_htk, write_htk

x, sr = read_wav("input.wav")        # x: float64, range -1 to 1
write_wav("output.wav", x, sr)       # written as 16-bit PCM

# HTK feature file (.mfc)
feats, header = read_htk("utt001.mfc")
print(header)
# {'n_samples': 312, 'samp_period': 100000, 'samp_size': 52,
#  'parm_kind': 6, 'n_dim': 13, 'format': 'binary'}

write_htk("utt001.mfc", feats, period_100ns=100000, kind=6)
```

The binary HTK header is 12 bytes, every field **big-endian**:

| Field | Type | Meaning |
| --- | --- | --- |
| `nSamples` | int32 | Number of frames |
| `sampPeriod` | int32 | Frame period in units of 100 ns (10 ms is 100000) |
| `sampSize` | int16 | Bytes per frame (dimensions × 4) |
| `parmKind` | int16 | Feature kind code; `6` means MFCC |

Big-endian `float32` data follows the header. Some tools write features out as a
plain-text matrix of numbers instead. `read_htk` detects that and parses the file as
text, in which case the returned `header['format']` is `'text'`.

### speechdsp.framing — framing and overlap-add

The base that every short-time analysis in the package builds on. The framing
convention lives here: frame `t` starts at sample `t * hop` and is `frame_len` samples
long, and trailing samples that do not fill a whole frame are dropped.

```python
from speechdsp.framing import enframe, overlap_add, frame_to_sample, frame_time

frames = enframe(x, frame_len=400, hop=160, window="hamming")  # (frames, 400)
y = overlap_add(frames, hop=160)                                # back to a 1-D signal

frame_to_sample(10, frame_len=400, hop=160)   # 1600, the first sample of frame 10
frame_time(100, 400, 160, sr=16000)           # center time of each frame (s)
```

`enframe` builds a strided view with `sliding_window_view` and copies it once, with no
Python loop, so it stays fast on long signals.

### speechdsp.spectral — short-time spectral analysis

The STFT and its inverse. Synthesis divides by the overlap-added envelope of the
squared window instead of assuming that envelope is constant. As long as the window
meets the NOLA condition (the default periodic Hann window does whenever
`hop <= n_fft // 2`), `istft(stft(x))` gives the signal back to within floating-point
precision.

```python
from speechdsp.spectral import stft, istft, spectrogram_db, spectrogram_image

S = stft(x, n_fft=512, hop=128)            # (frames, 257) complex array
y = istft(S, n_fft=512, hop=128)           # perfect reconstruction, error < 1e-10

# 25 ms frames at 16 kHz (400 samples), zero-padded to a 512-point FFT
S = stft(x, n_fft=512, hop=160, win_length=400)

db = spectrogram_db(x, sr, n_fft=512, hop=160, top_db=80.0)   # range -80 to 0 dB

img = spectrogram_image(x, sr, shape=(40, 98))   # uint8 grayscale image, always 40x98
```

`win_length` sets the window length (the frame length) and must lie between 1 and
`n_fft`. The window sits in the middle of the `n_fft`-point frame and the rest is
zero-padded, which is the same as padding the windowed frame with zeros to `n_fft`
points before the FFT. If `win_length` is not given, the window is `n_fft` samples long.
`stft`, `istft`, `power_spectrum`, `spectrogram_db` and `spectrogram_image` all take
this parameter; `istft` needs the same value the analysis used.

`power_to_db` converts to decibels with a floor **relative to the peak**: anything more
than `top_db` below the peak is clipped to `-top_db`. The recording level therefore has
no effect on the result, and a very quiet recording keeps its full dynamic range. An
input with no energy at all (all zeros) has no peak to refer to, so every value comes
out as `-top_db` and `spectrogram_image` returns an all-black (0) image. A NaN or inf
anywhere in the signal raises `ValueError` at once instead of being drawn as an image
that looks like silence.

`spectrogram_image` resamples the time axis to a fixed width by linear interpolation,
so utterances of any length come out the same size and can go straight into a CNN.
Row 0 is the lowest mel band and column 0 the earliest frame.

### speechdsp.features — cepstral features

The full MFCC pipeline: pre-emphasis → framing with a Hamming window → FFT power
spectrum → triangular mel filterbank → log → DCT-II → cepstral liftering.

```python
from speechdsp.features import mfcc, mfcc_with_deltas, delta, mel_filterbank, cmvn

feats = mfcc(x, sr, n_mfcc=13, n_mels=26,
             frame_ms=25.0, hop_ms=10.0, n_fft=512,
             preemph=0.97, lifter=22, append_energy=True)

feats39 = mfcc_with_deltas(x, sr, n_mfcc=13)   # [static | delta | delta-delta]
feats39 = cmvn(feats39)                         # mean and variance normalization

fb = mel_filterbank(sr, n_fft=512, n_mels=26)   # (26, 257)
```

A few implementation details:

- The mel conversion uses the standard formula `m = 2595 * log10(1 + f / 700)`. Each
  triangular filter peaks at 1.0 (the HTK convention), so the filterbank output is in
  the same units as the input power spectrum.
- With `append_energy=True`, coefficient 0 is replaced by the natural log of the
  frame's total energy, which is more stable than the DCT's DC term; most ASR front
  ends do the same by default.
- `n_fft` grows when it has to. If a high sample rate makes one frame longer than
  `n_fft` (a 25 ms frame at 44.1 kHz is 1102 samples), `n_fft` is raised to the next
  power of two and a warning is logged, so no samples are thrown away.
- `delta` uses the standard regression formula with the denominator `2 * sum(n^2)`,
  and pads the edges by repeating frames. On a linear ramp the interior frames come
  out at exactly the slope.

### speechdsp.vad — endpoint detection

Dual-threshold endpoint detection based on short-time energy and zero-crossing rate.

```python
from speechdsp.vad import endpoint_detect, trim_silence, frame_energy, zero_crossing_rate

start, end = endpoint_detect(x, sr, frame_ms=25.0, hop_ms=10.0)
speech = x[start:end]

speech = trim_silence(x, sr)     # same as the two lines above
```

The first few frames give an estimate of the background noise energy, and the upper
threshold `ITU` and lower threshold `ITL` are derived from it. Once the scan finds the
first frame above `ITU`, it steps back to the last frame still above `ITL`; the end of
the utterance is found the same way. Finally the zero-crossing rate pushes the
boundaries outwards to take in unvoiced fricatives such as `/s/` and `/f/`, which have
low energy but a high zero-crossing rate.

Every threshold comes from the signal itself, so the absolute recording level does not
change what is detected. The zero-crossing extension also has an energy gate, so
broadband background noise (whose zero-crossing rate is close to 0.5 anyway) cannot
drag the boundaries all the way out.

### speechdsp.enhance — speech enhancement

Two single-channel, frequency-domain methods. Both assume that the first few frames of
the signal contain only background noise.

```python
from speechdsp.enhance import log_mmse, spectral_subtraction

y1 = log_mmse(x, sr, noise_frames=6, alpha=0.98)
y2 = spectral_subtraction(x, sr, noise_frames=6, over_sub=2.0, floor=0.002)
```

- `log_mmse`: the log-MMSE short-time spectral amplitude estimator. The gain is
  `G = xi / (1 + xi) * exp(0.5 * E1(v))`, where `xi` is the a priori SNR, estimated
  recursively with the decision-directed approach, and `E1` is the exponential integral
  (`scipy.special.exp1`). Unlike plain MMSE-STSA, it minimizes the error in the **log**
  spectral amplitude, which is closer to how we hear, and it leaves noticeably less
  musical noise behind.
- `spectral_subtraction`: power spectral subtraction with an over-subtraction factor
  and a spectral floor.

Both keep the phase of the input signal, and the output is as long as the input.

> **Important:** `noise_frames` is the number of leading frames used to estimate the
> noise, and those frames must contain background noise and nothing else. If a file
> does not start with silence, add some yourself first, or estimate the noise another
> way.

### speechdsp.metrics — evaluation metrics

Clinical and pathological speech data are almost always class-imbalanced, and accuracy
on its own is misleading there: a model that only ever guesses the majority class can
still post a very good number. This module is therefore built around
**UAR (unweighted average recall, the arithmetic mean of the per-class recalls)**
and **sensitivity/specificity**.

```python
from speechdsp.metrics import uar, sensitivity_specificity, confusion_report, cross_val_report
from sklearn.svm import SVC

uar(y_true, y_pred)                                    # 0.625
sensitivity_specificity(y_true, y_pred, pos_label=1)   # (0.5, 0.75)

report = confusion_report(y_true, y_pred)
print(report["confusion_matrix"], report["uar"], report["macro_f1"])

# Five-fold cross-validation with a fixed random seed, so the results reproduce exactly
cv = cross_val_report(SVC(kernel="rbf"), X, y, n_splits=5, seed=0)
print(cv["mean"]["uar"], cv["std"]["uar"])
print(cv["pooled"]["confusion_matrix"])
```

`uar` averages only over the classes present in `y_true`, the same definition as
scikit-learn's `balanced_accuracy_score`. If the model predicts a class that never
occurs in `y_true`, the mistake only lowers the recall of the class that was
misclassified; the absent class is not added to the average as an extra class with a
recall of 0. This matters most when a fold is missing one of the classes. The `uar` in
`confusion_report` uses the same definition.

The dictionary returned by `cross_val_report` holds:

| Key | Contents |
| --- | --- |
| `folds` | The full report for each fold (confusion matrix, UAR, accuracy, sensitivity, specificity, training/test sample counts) |
| `mean` / `std` | Mean and standard deviation of each metric across the folds |
| `pooled` | One report over the out-of-fold predictions of all folds put together |
| `labels` | Class order |

If a subject has more than one recording, pass the `groups` argument. The split then
uses `StratifiedGroupKFold`, so no group ever has samples in both the training set and
the test set. Without it the model may learn to recognize the person rather than the
symptoms, and the scores come out too optimistic.

Every fold gets a fresh copy of the estimator from `sklearn.base.clone`, so the object
you pass in is never modified.

---

## API quick reference

### `speechdsp.io`

| Function | Description |
| --- | --- |
| `read_wav(path) -> (np.ndarray, int)` | Read a WAV file; returns a float64 mono signal and the sample rate |
| `write_wav(path, x, sr) -> None` | Write a WAV file as 16-bit PCM |
| `read_htk(path) -> (np.ndarray, dict)` | Read an HTK feature file, detecting binary or text format |
| `write_htk(path, feats, period_100ns=100000, kind=6) -> None` | Write a binary HTK feature file |

### `speechdsp.framing`

| Function | Description |
| --- | --- |
| `enframe(x, frame_len, hop, window=None)` | Split into frames; returns `(frames, frame_len)` |
| `overlap_add(frames, hop, window=None)` | Overlap-add back into a 1-D signal |
| `num_frames(n_samples, frame_len, hop)` | Number of frames under the framing convention |
| `frame_to_sample(frame_idx, frame_len, hop)` | Frame index → first sample |
| `frame_time(n_frames, frame_len, hop, sr)` | Center time of each frame (s) |

### `speechdsp.spectral`

| Function | Description |
| --- | --- |
| `stft(x, n_fft, hop, window='hann', center=True, win_length=None)` | Short-time Fourier transform; returns a complex matrix |
| `istft(S, n_fft, hop, window='hann', center=True, win_length=None)` | Inverse STFT with perfect reconstruction |
| `power_spectrum(x, n_fft, hop, win_length=None)` | Power spectrum of each frame |
| `power_to_db(power, top_db=80.0)` | Power to dB relative to the peak, with a relative floor |
| `spectrogram_db(x, sr, n_fft, hop, top_db=80.0, win_length=None)` | Spectrogram in dB relative to the peak |
| `spectrogram_image(x, sr, shape=(40, 98), n_fft=512, hop=None, win_length=None)` | Fixed-size uint8 grayscale spectrogram |

### `speechdsp.features`

| Function | Description |
| --- | --- |
| `preemphasis(x, coeff=0.97)` | First-order pre-emphasis filter |
| `hz_to_mel(f)` / `mel_to_hz(m)` | Convert between Hz and mel |
| `mel_filterbank(sr, n_fft, n_mels=26, fmin=0.0, fmax=None)` | Triangular mel filterbank |
| `mfcc(x, sr, n_mfcc=13, ...)` | MFCC features |
| `delta(feats, width=9)` | Regression delta coefficients |
| `mfcc_with_deltas(x, sr, **kw)` | `[static \| delta \| delta-delta]` concatenated |
| `cmn(X)` / `cvn(X)` / `cmvn(X)` | Cepstral mean, variance, or mean and variance normalization |

### `speechdsp.vad`

| Function | Description |
| --- | --- |
| `frame_energy(frames)` | Short-time energy of each frame |
| `zero_crossing_rate(frames)` | Zero-crossing rate of each frame (0 to 1) |
| `endpoint_detect(x, sr, frame_ms=25.0, hop_ms=10.0)` | Returns `(start_sample, end_sample)` |
| `trim_silence(x, sr, **kw)` | Cut off leading and trailing silence |

### `speechdsp.enhance`

| Function | Description |
| --- | --- |
| `log_mmse(x, sr, noise_frames=6, alpha=0.98)` | log-MMSE spectral amplitude estimator |
| `spectral_subtraction(x, sr, noise_frames=6, over_sub=2.0, floor=0.002)` | Spectral subtraction |

### `speechdsp.metrics`

| Function | Description |
| --- | --- |
| `uar(y_true, y_pred)` | Mean recall over the classes in `y_true` (same as balanced accuracy) |
| `sensitivity_specificity(y_true, y_pred, pos_label=1)` | Sensitivity and specificity |
| `confusion_report(y_true, y_pred, labels=None)` | Dictionary with the confusion matrix and every metric |
| `cross_val_report(estimator, X, y, groups=None, n_splits=5, seed=0)` | Stratified cross-validation report |

---

## Data paths

**`speechdsp` ships no corpora or audio files and never copies any.** Every data path
should come from your own configuration; the functions only take arrays already in
memory or file paths you pass in explicitly.

In your own project, keep the data location in one place, for example:

```python
from pathlib import Path

DATA_ROOT = Path("replace with your own data folder")   # the default is only a placeholder

for wav_path in sorted(DATA_ROOT.glob("*.wav")):
    x, sr = speechdsp.read_wav(wav_path)
    ...
```

The tests under `tests/` use only small synthetic signals (sine waves, white noise,
linear ramps) and run without any external files.

---

## Tests and linting

```bash
# Run the tests (pyproject.toml already sets pythonpath, no environment variables needed)
python -m pytest -q

# Code style and static checks
python -m ruff check .
python -m ruff format --check .
```

CI runs these checks on Ubuntu and Windows with Python 3.10 to 3.13.

The numerical checks in the tests include:

- the STFT peak of a sine wave of known frequency lands in the right frequency bin
- `istft(stft(x))` reconstructs with an error below `1e-10` (several window and hop
  combinations, including windows shorter than `n_fft`)
- an STFT whose window is shorter than `n_fft` has exactly the same magnitudes as
  windowing, zero-padding and then taking the FFT
- the dB spectrogram and the spectrogram image do not depend on the recording level;
  all-zero input gives `-top_db` and an all-black image
- `enframe` followed by `overlap_add` gives the signal back
- the mel filterbank's peak gain, monotonically increasing center frequencies, and
  coverage of the full band
- MFCC behavior on a constant signal, on digital silence and at different sample rates
- `delta` of a linear ramp equals its slope
- an HTK file reads back exactly as written, and its header really is big-endian
- `log_mmse` and `spectral_subtraction` raise the SNR of a noisy signal
- `endpoint_detect` finds the right boundaries in a synthetic silence-speech-silence
  signal
- `uar` and `sensitivity_specificity` match small examples worked by hand, and `uar`
  agrees with scikit-learn's `balanced_accuracy_score`

### Compatibility

The code targets **Python 3.10** (the default on Ubuntu 22.04, Google Colab and most
conda environments) and works with both NumPy 1.x and 2.x. To check that the syntax
uses nothing added in 3.11 or later:

```bash
python -c "import ast,pathlib; [ast.parse(f.read_text(encoding='utf-8'), filename=str(f), feature_version=(3,10)) for d in ('src','tests') for f in pathlib.Path(d).rglob('*.py')]"
```

---

## Design principles

1. Vectorize first. If NumPy broadcasting can do it, it does not get a Python loop.
   The one loop left is the decision-directed recursion in log-MMSE, which by its
   nature has to run frame by frame, and even there each iteration covers the whole
   frequency axis at once.
2. Keep conventions in one place. Framing is defined only in `speechdsp.framing` and
   every other module uses it, so no two modules can disagree about where a given
   frame starts.
3. Fail with a clear message. A shape mismatch or an out-of-range parameter raises a
   `ValueError` straight away, with the actual values in the message, instead of
   surfacing a few layers further down as some unrelated-looking error.
4. Full type hints and numpy-style docstrings. Every public function documents its
   Parameters / Returns / References, so IDEs and documentation tools show them
   properly.
5. No deep learning framework required. `torch` is an optional extra that the code
   never imports; without it, `import speechdsp` still works and the tests still run
   (`tests/test_package.py` checks this).

---

## References

The algorithms follow these references:

1. S. B. Davis and P. Mermelstein, "Comparison of parametric representations for
   monosyllabic word recognition in continuously spoken sentences," *IEEE
   Transactions on Acoustics, Speech, and Signal Processing*, vol. 28, no. 4,
   pp. 357–366, 1980.
2. S. S. Stevens, J. Volkmann, and E. B. Newman, "A scale for the measurement of
   the psychological magnitude pitch," *The Journal of the Acoustical Society of
   America*, vol. 8, no. 3, pp. 185–190, 1937.
3. Y. Ephraim and D. Malah, "Speech enhancement using a minimum mean-square error
   short-time spectral amplitude estimator," *IEEE Transactions on Acoustics,
   Speech, and Signal Processing*, vol. 32, no. 6, pp. 1109–1121, 1984.
4. Y. Ephraim and D. Malah, "Speech enhancement using a minimum mean-square error
   log-spectral amplitude estimator," *IEEE Transactions on Acoustics, Speech, and
   Signal Processing*, vol. 33, no. 2, pp. 443–445, 1985.
5. S. F. Boll, "Suppression of acoustic noise in speech using spectral
   subtraction," *IEEE Transactions on Acoustics, Speech, and Signal Processing*,
   vol. 27, no. 2, pp. 113–120, 1979.
6. M. Berouti, R. Schwartz, and J. Makhoul, "Enhancement of speech corrupted by
   acoustic noise," *ICASSP*, vol. 4, pp. 208–211, 1979.
7. L. R. Rabiner and M. R. Sambur, "An algorithm for determining the endpoints of
   isolated utterances," *Bell System Technical Journal*, vol. 54, no. 2,
   pp. 297–315, 1975.
8. L. R. Rabiner and R. W. Schafer, *Digital Processing of Speech Signals*,
   Prentice-Hall, 1978.
9. J. B. Allen and L. R. Rabiner, "A unified approach to short-time Fourier
   analysis and synthesis," *Proceedings of the IEEE*, vol. 65, no. 11,
   pp. 1558–1564, 1977.
10. D. W. Griffin and J. S. Lim, "Signal estimation from modified short-time
    Fourier transform," *IEEE Transactions on Acoustics, Speech, and Signal
    Processing*, vol. 32, no. 2, pp. 236–243, 1984.
11. S. Young et al., *The HTK Book (version 3.4)*, Cambridge University
    Engineering Department, 2006.
12. A. Rosenberg, "Classifying skewed data: Importance weighting to optimize
    average recall," *Interspeech*, 2012.
13. T. Fawcett, "An introduction to ROC analysis," *Pattern Recognition Letters*,
    vol. 27, no. 8, pp. 861–874, 2006.

---

## Contributing and citing

- Development setup, checks before committing, and code style: [CONTRIBUTING.md](CONTRIBUTING.md)
- Changes in each release: [CHANGELOG.md](CHANGELOG.md)
- If you use the package in research, the citation details are in
  [CITATION.cff](CITATION.cff) (GitHub's "Cite this repository" link on the right of
  the page reads the same file)
- Bug reports and feature requests go to [Issues](https://github.com/recklight/SpeechDsp/issues)

CONTRIBUTING.md and CHANGELOG.md are written in Traditional Chinese.

---

## License

MIT License, Copyright (c) RL. See [LICENSE](LICENSE).
