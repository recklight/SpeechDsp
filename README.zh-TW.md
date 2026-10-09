# SpeechDsp

[![PyPI](https://img.shields.io/pypi/v/speechdsp.svg)](https://pypi.org/project/speechdsp/)
[![CI](https://github.com/recklight/SpeechDsp/actions/workflows/ci.yml/badge.svg)](https://github.com/recklight/SpeechDsp/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[English](README.md) | 繁體中文

> 語音訊號處理、特徵擷取與模型評估的共用工具箱

`speechdsp` 是一套以 NumPy / SciPy / scikit-learn 為基礎、從零撰寫的 Python 套件，
提供語音與聲學研究常用的前端處理流程：音檔與特徵檔讀寫、分幀、短時傅立葉轉換、
MFCC 與差量特徵、端點偵測、語音增強，以及適合類別不平衡資料的評估指標。

本套件刻意不依賴 `librosa`、`soundfile` 或任何深度學習框架，所有演算法都以
`numpy` + `scipy` 自行實作，方便部署在沒有額外套件的運算環境（例如叢集節點或
乾淨的 conda 環境）。

- **作者**：RL
- **授權**：MIT
- **支援版本**：Python 3.10 以上

---

## 目錄

1. [安裝](#安裝)
2. [快速開始](#快速開始)
3. [模組說明](#模組說明)
   - [speechdsp.io — 音檔與特徵檔存取](#speechdspio--音檔與特徵檔存取)
   - [speechdsp.framing — 分幀與重疊相加](#speechdspframing--分幀與重疊相加)
   - [speechdsp.spectral — 短時頻譜分析](#speechdspspectral--短時頻譜分析)
   - [speechdsp.features — 倒頻譜特徵](#speechdspfeatures--倒頻譜特徵)
   - [speechdsp.vad — 端點偵測](#speechdspvad--端點偵測)
   - [speechdsp.enhance — 語音增強](#speechdspenhance--語音增強)
   - [speechdsp.metrics — 評估指標](#speechdspmetrics--評估指標)
4. [API 速查表](#api-速查表)
5. [資料路徑說明](#資料路徑說明)
6. [測試與程式碼檢查](#測試與程式碼檢查)
7. [設計原則](#設計原則)
8. [References](#references)
9. [參與開發與引用](#參與開發與引用)

---

## 安裝

從 PyPI 安裝：

```bash
pip install speechdsp
```

若要修改原始碼，建議在虛擬環境中以可編輯模式安裝：

```bash
git clone https://github.com/recklight/SpeechDsp.git
cd SpeechDsp
python -m pip install -e .
```

若要一併安裝開發與測試工具：

```bash
python -m pip install -e ".[dev]"
```

### 相依套件

| 類別 | 套件 | 版本下限 |
| --- | --- | --- |
| 核心 | `numpy` | 1.24 |
| 核心 | `scipy` | 1.10 |
| 核心 | `scikit-learn` | 1.2 |
| 選用 `[plot]` | `matplotlib` | 3.6 |
| 選用 `[dl]` | `torch` | 2.2 |
| 選用 `[dev]` | `pytest`、`ruff` | — |

核心相依只有三個科學運算套件。`torch` 只是選用的額外項目，套件本身在任何情況下
都不會在載入時 import 它。

### 在其他專案中引用

其他專案可以用 path dependency 的方式直接指向本資料夾，例如在該專案的
`pyproject.toml` 中：

```toml
[project]
dependencies = ["speechdsp"]

[tool.uv.sources]
speechdsp = { path = "../SpeechDsp", editable = true }
```

或者最簡單的做法，先把本套件安裝進同一個環境即可：

```bash
python -m pip install -e /path/to/SpeechDsp
```

---

## 快速開始

```python
import numpy as np
import speechdsp

# 讀取音檔：回傳 float64 單聲道訊號（範圍 -1 ~ 1）與取樣率
x, sr = speechdsp.read_wav("example.wav")

# 去除頭尾靜音
x = speechdsp.trim_silence(x, sr)

# 語音增強（log-MMSE）
x = speechdsp.log_mmse(x, sr, noise_frames=6)

# 擷取 39 維 MFCC（13 維靜態 + 一階差量 + 二階差量）
feats = speechdsp.mfcc_with_deltas(x, sr, n_mfcc=13)

# 倒頻譜平均變異數正規化
feats = speechdsp.cmvn(feats)

print(feats.shape)   # (幀數, 39)
```

---

## 模組說明

### speechdsp.io — 音檔與特徵檔存取

處理 WAV 與 HTK 特徵檔的讀寫。不論磁碟上是 16-bit PCM、32-bit PCM 或浮點格式，
`read_wav` 一律回傳 `float64` 的單聲道訊號，多聲道會自動平均混成單聲道。

```python
from speechdsp.io import read_wav, write_wav, read_htk, write_htk

x, sr = read_wav("input.wav")        # x: float64, 範圍 -1 ~ 1
write_wav("output.wav", x, sr)       # 以 16-bit PCM 寫出

# HTK 特徵檔（.mfc）
feats, header = read_htk("utt001.mfc")
print(header)
# {'n_samples': 312, 'samp_period': 100000, 'samp_size': 52,
#  'parm_kind': 6, 'n_dim': 13, 'format': 'binary'}

write_htk("utt001.mfc", feats, period_100ns=100000, kind=6)
```

HTK 二進位格式的表頭共 12 個位元組，全部使用**大端序（big-endian）**：

| 欄位 | 型別 | 說明 |
| --- | --- | --- |
| `nSamples` | int32 | 幀數 |
| `sampPeriod` | int32 | 幀週期，單位 100 奈秒（10 ms 即 100000） |
| `sampSize` | int16 | 每幀的位元組數（維度 × 4） |
| `parmKind` | int16 | 特徵種類代碼，`6` 代表 MFCC |

表頭之後是大端序的 `float32` 資料。部分工具會把特徵輸出成純文字的數值矩陣，
`read_htk` 會自動偵測並改用文字模式解析，此時回傳的 `header['format']` 為
`'text'`。

### speechdsp.framing — 分幀與重疊相加

短時分析的共用基礎。全套件的分幀慣例都定在這裡：第 `t` 幀從第 `t * hop` 個取樣點
開始、長度為 `frame_len`，尾端不足一幀的取樣點會被捨棄。

```python
from speechdsp.framing import enframe, overlap_add, frame_to_sample, frame_time

frames = enframe(x, frame_len=400, hop=160, window="hamming")  # (幀數, 400)
y = overlap_add(frames, hop=160)                                # 疊回一維訊號

frame_to_sample(10, frame_len=400, hop=160)   # 1600，第 10 幀的起始取樣點
frame_time(100, 400, 160, sr=16000)           # 每一幀的中心時間（秒）
```

`enframe` 使用 `sliding_window_view` 建立跨步視圖後一次複製，不用 Python 迴圈，
長訊號也很快。

### speechdsp.spectral — 短時頻譜分析

STFT 與其反轉換。合成時會除以「窗函數平方的重疊相加包絡線」，而不是假設包絡線
是常數，因此只要窗函數滿足 NOLA 條件（預設的週期性 Hann 窗在 `hop <= n_fft // 2`
時都滿足），`istft(stft(x))` 就能把訊號完整還原，誤差在浮點精度等級。

```python
from speechdsp.spectral import stft, istft, spectrogram_db, spectrogram_image

S = stft(x, n_fft=512, hop=128)            # (幀數, 257) 複數陣列
y = istft(S, n_fft=512, hop=128)           # 完美重建，誤差 < 1e-10

# 16 kHz 下 25 ms 音框（400 點）補零到 512 點 FFT
S = stft(x, n_fft=512, hop=160, win_length=400)

db = spectrogram_db(x, sr, n_fft=512, hop=160, top_db=80.0)   # 值域 -80 ~ 0 dB

img = spectrogram_image(x, sr, shape=(40, 98))   # uint8 灰階圖，固定 40x98
```

`win_length` 指定窗長（音框長度），必須介於 1 與 `n_fft` 之間。窗函數放在
`n_fft` 點音框的正中央，其餘補零，等同於把加窗後的音框補零到 `n_fft` 點再做 FFT。
不指定時窗長就是 `n_fft`。`stft`、`istft`、`power_spectrum`、`spectrogram_db`
與 `spectrogram_image` 都接受這個參數；`istft` 要帶與分析時相同的值。

分貝換算由 `power_to_db` 負責，下限是**相對於峰值**的：比峰值低超過 `top_db`
的值一律截在 `-top_db`。所以錄音音量大小不影響結果，很小聲的錄音也保有完整的
動態範圍。完全沒有能量的輸入（全零）沒有峰值可以參考，會得到全部 `-top_db`，
`spectrogram_image` 則輸出全黑（0）的影像。訊號裡有 NaN 或 inf 時會直接引發
`ValueError`，不會畫成一張看似靜音的圖。

`spectrogram_image` 會把時間軸用線性內插重新取樣成固定寬度，無論語句長短都輸出
同一個尺寸，可直接餵給 CNN。第 0 列是最低的 mel 頻帶，第 0 行是最早的一幀。

### speechdsp.features — 倒頻譜特徵

MFCC 的完整流程：預強調 → 分幀加 Hamming 窗 → FFT 功率譜 → mel 三角濾波器組 →
取對數 → DCT-II → 倒頻譜提升（liftering）。

```python
from speechdsp.features import mfcc, mfcc_with_deltas, delta, mel_filterbank, cmvn

feats = mfcc(x, sr, n_mfcc=13, n_mels=26,
             frame_ms=25.0, hop_ms=10.0, n_fft=512,
             preemph=0.97, lifter=22, append_energy=True)

feats39 = mfcc_with_deltas(x, sr, n_mfcc=13)   # [靜態 | 一階差量 | 二階差量]
feats39 = cmvn(feats39)                         # 平均與變異數正規化

fb = mel_filterbank(sr, n_fft=512, n_mels=26)   # (26, 257)
```

幾個實作細節：

- **mel 轉換**使用標準公式 `m = 2595 * log10(1 + f / 700)`，三角濾波器的峰值
  正規化為 1.0（HTK 慣例），因此濾波器組輸出的能量與輸入功率譜同單位。
- **`append_energy=True`** 時，第 0 維係數會換成該幀總能量的自然對數，這比 DCT 的
  直流項更穩定，也是多數 ASR 前端的預設作法。
- **`n_fft` 自動調整**：若取樣率較高使得一幀的長度超過 `n_fft`（例如 44.1 kHz 的
  25 ms 幀共 1102 點），會自動提升到下一個 2 的冪次並記錄一則警告，不會丟掉取樣點。
- **`delta`** 使用標準迴歸式，分母為 `2 * sum(n^2)`，邊界以複製方式補幀。對一段
  線性斜坡輸入，內部區域會精確得到該斜率。

### speechdsp.vad — 端點偵測

以短時能量與過零率為基礎的雙門檻端點偵測。

```python
from speechdsp.vad import endpoint_detect, trim_silence, frame_energy, zero_crossing_rate

start, end = endpoint_detect(x, sr, frame_ms=25.0, hop_ms=10.0)
speech = x[start:end]

speech = trim_silence(x, sr)     # 等同於上面兩行
```

演算法流程：先用開頭幾幀估計背景雜訊能量，再據此推出上門檻 `ITU` 與下門檻 `ITL`；
掃描出第一個超過 `ITU` 的幀之後，往回退到最後一個高於 `ITL` 的幀，尾端同理。最後
用過零率往外延伸，把能量偏低但過零率偏高的清擦音（例如 `/s/`、`/f/`）也納進來。

門檻全部由訊號自己推得，所以偵測結果不受錄音音量絕對值影響。過零率的延伸步驟另外
加了能量閘門，避免寬頻背景雜訊（過零率本來就接近 0.5）把邊界一路往外拖。

### speechdsp.enhance — 語音增強

兩種單通道頻域增強方法，都假設訊號開頭幾幀只有背景雜訊。

```python
from speechdsp.enhance import log_mmse, spectral_subtraction

y1 = log_mmse(x, sr, noise_frames=6, alpha=0.98)
y2 = spectral_subtraction(x, sr, noise_frames=6, over_sub=2.0, floor=0.002)
```

- **`log_mmse`**：log-MMSE 短時頻譜振幅估測。增益為
  `G = xi / (1 + xi) * exp(0.5 * E1(v))`，其中 `xi` 是以 decision-directed 方式
  遞迴估計的先驗訊雜比，`E1` 是指數積分（`scipy.special.exp1`）。相較於單純的
  MMSE-STSA，它最小化的是**對數**頻譜振幅的誤差，比較貼近聽覺感受，殘留的
  musical noise 也明顯較少。
- **`spectral_subtraction`**：功率頻譜相減，附帶過度相減係數與頻譜地板。

兩者都保留原始相位，輸出長度與輸入相同。

> **注意**：`noise_frames` 指的是開頭用來估計雜訊的幀數，這幾幀必須確實只有背景
> 雜訊。若音檔開頭沒有靜音段，請先自行補上，或改用其他雜訊估計方式。

### speechdsp.metrics — 評估指標

臨床或病理語音資料幾乎都有類別不平衡的問題，單看正確率（accuracy）會產生誤導：
一個永遠只猜多數類別的模型也可以有很漂亮的數字。因此本模組以
**UAR（unweighted average recall，各類別召回率的算術平均）**
與 **敏感度／特異度** 為主要指標。

```python
from speechdsp.metrics import uar, sensitivity_specificity, confusion_report, cross_val_report
from sklearn.svm import SVC

uar(y_true, y_pred)                                    # 0.625
sensitivity_specificity(y_true, y_pred, pos_label=1)   # (0.5, 0.75)

report = confusion_report(y_true, y_pred)
print(report["confusion_matrix"], report["uar"], report["macro_f1"])

# 五折交叉驗證，固定亂數種子，結果完全可重現
cv = cross_val_report(SVC(kernel="rbf"), X, y, n_splits=5, seed=0)
print(cv["mean"]["uar"], cv["std"]["uar"])
print(cv["pooled"]["confusion_matrix"])
```

`uar` 只對 `y_true` 中出現的類別取平均，定義與 scikit-learn 的
`balanced_accuracy_score` 相同。如果模型預測出一個在 `y_true` 裡不存在的類別，
這個錯誤只會降低被誤判那一類的召回率，不會讓那個不存在的類別以召回率 0
再算進平均一次。這在某一折缺少某個類別時特別重要。`confusion_report` 的 `uar`
採用同一個定義。

`cross_val_report` 回傳的字典包含：

| 鍵值 | 內容 |
| --- | --- |
| `folds` | 每一折的完整報告（含混淆矩陣、UAR、正確率、敏感度、特異度、訓練／測試樣本數） |
| `mean` / `std` | 各折指標的平均與標準差 |
| `pooled` | 把所有折的 out-of-fold 預測合併後的整體報告 |
| `labels` | 類別順序 |

若同一位受試者有多筆錄音，請傳入 `groups` 參數，此時會改用
`StratifiedGroupKFold`，確保同一組樣本不會同時出現在訓練集與測試集，否則模型可能
學到的是「認人」而不是「辨識症狀」，分數會過度樂觀。

每一折都會以 `sklearn.base.clone` 複製一份新的估計器，傳進去的物件不會被修改。

---

## API 速查表

### `speechdsp.io`

| 函式 | 說明 |
| --- | --- |
| `read_wav(path) -> (np.ndarray, int)` | 讀 WAV，回傳 float64 單聲道訊號與取樣率 |
| `write_wav(path, x, sr) -> None` | 以 16-bit PCM 寫出 WAV |
| `read_htk(path) -> (np.ndarray, dict)` | 讀 HTK 特徵檔，自動偵測二進位／文字格式 |
| `write_htk(path, feats, period_100ns=100000, kind=6) -> None` | 寫出二進位 HTK 特徵檔 |

### `speechdsp.framing`

| 函式 | 說明 |
| --- | --- |
| `enframe(x, frame_len, hop, window=None)` | 分幀，回傳 `(幀數, frame_len)` |
| `overlap_add(frames, hop, window=None)` | 重疊相加，疊回一維訊號 |
| `num_frames(n_samples, frame_len, hop)` | 依分幀慣例計算的幀數 |
| `frame_to_sample(frame_idx, frame_len, hop)` | 幀索引 → 起始取樣點 |
| `frame_time(n_frames, frame_len, hop, sr)` | 各幀的中心時間（秒） |

### `speechdsp.spectral`

| 函式 | 說明 |
| --- | --- |
| `stft(x, n_fft, hop, window='hann', center=True, win_length=None)` | 短時傅立葉轉換，回傳複數矩陣 |
| `istft(S, n_fft, hop, window='hann', center=True, win_length=None)` | 反短時傅立葉轉換，可完美重建 |
| `power_spectrum(x, n_fft, hop, win_length=None)` | 每幀的功率譜 |
| `power_to_db(power, top_db=80.0)` | 功率換成相對峰值的分貝，下限為相對值 |
| `spectrogram_db(x, sr, n_fft, hop, top_db=80.0, win_length=None)` | 相對峰值的分貝頻譜圖 |
| `spectrogram_image(x, sr, shape=(40, 98), n_fft=512, hop=None, win_length=None)` | 固定尺寸 uint8 灰階頻譜圖 |

### `speechdsp.features`

| 函式 | 說明 |
| --- | --- |
| `preemphasis(x, coeff=0.97)` | 一階預強調濾波器 |
| `hz_to_mel(f)` / `mel_to_hz(m)` | Hz 與 mel 互換 |
| `mel_filterbank(sr, n_fft, n_mels=26, fmin=0.0, fmax=None)` | mel 三角濾波器組 |
| `mfcc(x, sr, n_mfcc=13, ...)` | MFCC 特徵 |
| `delta(feats, width=9)` | 迴歸式差量係數 |
| `mfcc_with_deltas(x, sr, **kw)` | `[靜態 \| 一階 \| 二階]` 串接 |
| `cmn(X)` / `cvn(X)` / `cmvn(X)` | 倒頻譜平均／變異數／兩者正規化 |

### `speechdsp.vad`

| 函式 | 說明 |
| --- | --- |
| `frame_energy(frames)` | 每幀的短時能量 |
| `zero_crossing_rate(frames)` | 每幀的過零率（0 ~ 1） |
| `endpoint_detect(x, sr, frame_ms=25.0, hop_ms=10.0)` | 回傳 `(起始取樣點, 結束取樣點)` |
| `trim_silence(x, sr, **kw)` | 直接切掉頭尾靜音 |

### `speechdsp.enhance`

| 函式 | 說明 |
| --- | --- |
| `log_mmse(x, sr, noise_frames=6, alpha=0.98)` | log-MMSE 頻譜振幅估測 |
| `spectral_subtraction(x, sr, noise_frames=6, over_sub=2.0, floor=0.002)` | 頻譜相減法 |

### `speechdsp.metrics`

| 函式 | 說明 |
| --- | --- |
| `uar(y_true, y_pred)` | `y_true` 中各類別召回率的平均（同 balanced accuracy） |
| `sensitivity_specificity(y_true, y_pred, pos_label=1)` | 敏感度與特異度 |
| `confusion_report(y_true, y_pred, labels=None)` | 混淆矩陣與各項指標的字典 |
| `cross_val_report(estimator, X, y, groups=None, n_splits=5, seed=0)` | 分層交叉驗證報告 |

---

## 資料路徑說明

**本套件不內含、也不會複製任何語料或音檔。** 所有資料路徑都應該由使用者自己的
設定檔提供，套件本身只接受已經載入記憶體的陣列或明確傳入的檔案路徑。

在你自己的專案中，建議把資料位置集中在一處設定，例如：

```python
from pathlib import Path

DATA_ROOT = Path("請改成你自己的資料夾路徑")   # 預設值只是佔位字串

for wav_path in sorted(DATA_ROOT.glob("*.wav")):
    x, sr = speechdsp.read_wav(wav_path)
    ...
```

`tests/` 目錄下的測試完全使用程式合成的小型訊號（正弦波、白雜訊、線性斜坡），
不需要任何外部檔案就能執行。

---

## 測試與程式碼檢查

```bash
# 執行測試（pyproject.toml 已設定好 pythonpath，不需另外設環境變數）
python -m pytest -q

# 程式碼風格與靜態檢查
python -m ruff check .
python -m ruff format --check .
```

CI 會在 Ubuntu 與 Windows 上，以 Python 3.10 至 3.13 執行上述檢查。

測試涵蓋的數值驗證包括：

- 已知頻率的正弦波，其 STFT 峰值落在正確的頻率 bin
- `istft(stft(x))` 的重建誤差小於 `1e-10`（多種窗函數與 hop 組合，含窗長短於 `n_fft`）
- 窗長短於 `n_fft` 的 STFT 與「加窗後補零再做 FFT」的振幅完全一致
- 分貝頻譜圖與頻譜影像不受錄音音量影響；全零輸入得到 `-top_db` 與全黑影像
- `enframe` / `overlap_add` 往返一致
- mel 濾波器組的峰值增益、中心頻率單調遞增、全頻帶覆蓋
- MFCC 對常數訊號、數位靜音、不同取樣率的行為
- 線性斜坡訊號的 `delta` 應等於斜率
- HTK 檔案寫入再讀回完全一致，且表頭確實為大端序
- `log_mmse` 與 `spectral_subtraction` 對加噪訊號能提升訊雜比
- `endpoint_detect` 對「靜音－語音－靜音」合成訊號抓到正確邊界
- `uar` 與 `sensitivity_specificity` 對照手算的小例子；`uar` 與 scikit-learn 的
  `balanced_accuracy_score` 一致

### 相容性

程式碼以 **Python 3.10** 為目標版本撰寫（Ubuntu 22.04、Google Colab 與多數 conda
環境的預設版本），同時相容 NumPy 1.x 與 2.x。若要確認語法沒有用到 3.11 以後才有的
特性：

```bash
python -c "import ast,pathlib; [ast.parse(f.read_text(encoding='utf-8'), filename=str(f), feature_version=(3,10)) for d in ('src','tests') for f in pathlib.Path(d).rglob('*.py')]"
```

---

## 設計原則

1. **向量化優先。** 能用 NumPy 廣播解決的就不寫 Python 迴圈。唯一保留迴圈的地方是
   log-MMSE 的 decision-directed 遞迴（本質上必須逐幀進行），但即使如此，每一次
   迭代仍然是對整個頻率軸一次算完。
2. **慣例集中管理。** 分幀的定義只寫在 `speechdsp.framing` 一處，其他模組全部沿用，
   避免不同模組對「第幾幀從哪裡開始」有不同解讀。
3. **明確的錯誤訊息。** 尺寸不合、參數超出範圍時直接丟出帶有實際數值的
   `ValueError`，不要讓錯誤在後面幾層才以奇怪的形式爆出來。
4. **完整的型別註解與 numpy-style docstring。** 每個公開函式都標註了
   Parameters / Returns / References，方便 IDE 與文件工具正確呈現。
5. **不強制深度學習框架。** `torch` 只是選用額外項目，套件本身不會 import 它，
   沒有安裝也不影響套件載入與測試（`tests/test_package.py` 有對應檢查）。

---

## References

演算法實作參考下列文獻：

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

## 參與開發與引用

- 開發環境、提交前檢查與程式碼風格：見 [CONTRIBUTING.md](CONTRIBUTING.md)
- 各版本的變更：見 [CHANGELOG.md](CHANGELOG.md)
- 在研究中使用本套件時，引用資訊見 [CITATION.cff](CITATION.cff)
  （GitHub 頁面右側的「Cite this repository」也會讀取這個檔案）
- 錯誤回報與功能建議請使用 [Issues](https://github.com/recklight/SpeechDsp/issues)

---

## 授權

MIT License, Copyright (c) RL。詳見 [LICENSE](LICENSE)。
