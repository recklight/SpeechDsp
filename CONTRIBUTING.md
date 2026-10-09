# 參與開發

感謝你對 `speechdsp` 有興趣。這份文件說明如何在本機建立開發環境、執行測試，以及提交變更前該做的檢查。

---

## 開發環境

本專案支援 **Python 3.10 以上**。建議使用虛擬環境：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 安裝本專案

以可編輯模式安裝，並一併安裝開發工具：

```bash
pip install -e ".[dev]"
```

---

## 提交前的檢查

以下三項都必須通過，CI 會執行相同的檢查：

```bash
ruff check .           # 靜態檢查
ruff format --check .  # 格式（確認沒有未套用的變更；要看差異改用 --diff）
pytest -q              # 測試
```

### Python 3.10 相容性

本專案的開發機可能是較新的 Python，但**程式碼必須能在 3.10 上執行**。
新增程式碼時請避免 3.11 以後才有的語法與標準庫：

| 不要用 | 改用 |
| --- | --- |
| `tomllib` | `tomli` 或避免使用 |
| `typing.Self` | 類別名稱字串或 `TypeVar` |
| `enum.StrEnum` | `class Foo(str, Enum)` |
| `datetime.UTC` | `datetime.timezone.utc` |
| `except*`、`ExceptionGroup` | 一般的 `except` |
| `asyncio.TaskGroup` | `asyncio.gather` |
| PEP 695 泛型（`class Foo[T]`、`type X = ...`） | `TypeVar` 與 `TypeAlias` |
| `itertools.batched` | 自行實作切片 |

CI 有一道語法閘門會用 `ast.parse(..., feature_version=(3, 10))` 逐檔檢查，
用到新語法會直接失敗。本機可以這樣先自己跑一次（只掃 `src/` 與 `tests/`，
避免掃進 `.venv` 裡的第三方套件）：

```bash
python -c "import ast,pathlib; [ast.parse(f.read_text(encoding='utf-8'), filename=str(f), feature_version=(3,10)) for d in ('src','tests') for f in pathlib.Path(d).rglob('*.py')]"
```

---

## 程式碼風格

- 每個模組開頭加 `from __future__ import annotations`。
- 公開函式要有 type hints 與 numpy 風格的 docstring（英文），
  docstring 中若實作自某篇文獻，請在 `References` 段標明出處。
- `README.md` 以英文撰寫，`README.zh-TW.md` 是內容相同的**繁體中文**版，
  改了其中一份，另一份要跟著同步更新。本檔案以繁體中文撰寫。
- 以 `logging` 輸出診斷訊息，不要用 `print`。
- 陣列運算優先向量化，避免逐元素的 Python 迴圈。

---

## 資料

**本 repository 不包含任何實驗資料。** 測試一律使用程式合成的訊號作為 fixture。
請不要把音檔、`.mat`、`.npy`、模型權重或任何含個人資訊的檔案提交進版本庫；
`.gitignore` 已涵蓋常見的副檔名，但最終仍請自行確認。

---

## 提交訊息

建議使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

```
feat: 新增 xxx 功能
fix: 修正 xxx 在 yyy 情況下的錯誤
docs: 更新 xxx 說明
test: 補上 xxx 的測試
refactor: 重構 xxx
```

有功能變動時，請一併更新 `CHANGELOG.md` 的 `[Unreleased]` 區塊。

---

## 發佈新版本

推送到分支只會跑測試，**只有推送 `v*` 格式的 tag 才會發佈到 PyPI**。

1. 更新版本號，以下四處必須一致：
   `pyproject.toml`、`src/speechdsp/__init__.py` 的 `__version__`、
   `tests/test_package.py`、`CITATION.cff`（連同 `date-released`）。
2. 把 `CHANGELOG.md` 的 `[Unreleased]` 改成新版本號與日期。
3. commit 並推送，確認 CI 通過。
4. 打 tag 並推送：

   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

CI 會先跑完整測試，再確認 tag 與 `pyproject.toml` 的版本一致，才打包上傳。
PyPI 上的版本號一經上傳就不能重傳，刪除後也不能再用，打 tag 前請再確認一次。
