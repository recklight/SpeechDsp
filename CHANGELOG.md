# 變更紀錄

本檔案記錄 `speechdsp` 的所有重要變更。

格式依循 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.1.0/)，
版本號依循 [語意化版本](https://semver.org/lang/zh-TW/)。

## [Unreleased]

---

## [0.2.0] - 2026-10-08

`uar` 的定義與 scikit-learn 的 `balanced_accuracy_score` 一致（見「變更」），
某一折缺少某個類別時，分數會和 0.1.0 不同。

### 新增

- `stft`、`istft`、`power_spectrum`、`spectrogram_db`、`spectrogram_image` 新增
  `win_length` 參數：窗長可以短於 `n_fft`，等同把加窗後的音框補零到 `n_fft` 點
  （例如 25 ms 音框配 512 點 FFT）。不指定時結果與 0.1.0 完全相同。
- 公開 `power_to_db(power, top_db=80.0)`：把功率換成相對峰值的分貝，
  `spectrogram_db` 與 `spectrogram_image` 都經由它計算。

### 變更

- `uar` 只對 `y_true` 中出現的類別取平均，與 scikit-learn 的
  `balanced_accuracy_score` 定義相同。0.1.0 會把「只出現在預測結果」的類別以
  召回率 0 算進平均，使缺少某個類別的那一折數值偏低。`confusion_report`（以及
  `cross_val_report` 每一折）的 `uar` 採用同一個定義；`per_class_recall`
  仍列出所有標籤。
- 含 NaN 或 inf 的輸入會讓 `power_to_db`、`spectrogram_db` 與 `spectrogram_image`
  引發 `ValueError`。在 0.1.0，只要一個樣本是 NaN，`spectrogram_db` 就整張變成
  NaN，`spectrogram_image` 則輸出一張全黑、看起來像靜音的影像。

### 修正

- `spectrogram_db` 與 `spectrogram_image` 在 0.1.0 取對數前套用固定的絕對功率下限
  `1e-12`，很小聲的錄音因此用不滿 80 dB 的動態範圍：以 512 點 FFT 分析的弦波，
  振幅低於約 -55 dBFS 就會受影響（FFT 越長，門檻越低）。下限現在相對於峰值，
  結果不受錄音音量影響。
- 全零輸入在 0.1.0 得到 0 dB（`spectrogram_image` 輸出全白 255），現在得到
  `-top_db`（全黑 0）。

---

## [0.1.0] - 2026-09-28

首次釋出。核心只依賴 `numpy`、`scipy` 與 `scikit-learn`，所有演算法自行實作。

### 新增

- `speechdsp.io`：WAV 讀寫（一律回傳 float64 單聲道）與 HTK 特徵檔讀寫，
  自動辨識二進位／文字格式。
- `speechdsp.framing`：分幀、重疊相加與幀／取樣點換算，全套件共用同一套分幀慣例。
- `speechdsp.spectral`：STFT／ISTFT（以窗函數平方包絡線正規化，可完美重建）、
  功率譜、分貝頻譜圖，以及可直接餵給 CNN 的固定尺寸頻譜圖。
- `speechdsp.features`：預強調、mel 濾波器組、MFCC、迴歸式差量與 CMN／CVN／CMVN。
- `speechdsp.vad`：短時能量、過零率與雙門檻端點偵測，門檻由訊號本身推得。
- `speechdsp.enhance`：log-MMSE 與頻譜相減法語音增強。
- `speechdsp.metrics`：UAR、敏感度／特異度、混淆矩陣報告，以及支援受試者分組的
  分層交叉驗證報告。
- 公開函式皆附型別註解與 numpy 風格 docstring，並於 `References` 段標明文獻出處。
- 177 項測試，全部使用程式合成的訊號，不依賴任何外部實驗資料。
- GitHub Actions CI：Ubuntu／Windows × Python 3.10–3.13，含 3.10 語法閘門、
  ruff 與 pytest；推送 `v*` tag 時自動發佈到 PyPI。
- 繁體中文說明文件（README、CONTRIBUTING）與 CITATION.cff。
