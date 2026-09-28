# 變更紀錄

本檔案記錄 `speechdsp` 的所有重要變更。

格式依循 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.1.0/)，
版本號依循 [語意化版本](https://semver.org/lang/zh-TW/)。

## [Unreleased]

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
