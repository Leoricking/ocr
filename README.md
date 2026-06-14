# OCR Engine v4.4.0 — PDF 搜尋化與品質分析系統

## 專案功能

將掃描版 PDF 轉換為可搜尋的文字型 PDF，同時輸出純文字與品質分析報告。

- PaddleOCR（繁體中文）三路預處理候選自動選最佳
- 財經雜誌雙欄閱讀順序自動修正
- Claude AI 批次校對（可選）
- 固定式亂碼修正規則
- 品質分析報告（低置信度、亂碼標記、公式偵測）
- GPU 自動偵測與安全回退 CPU
- Nougat / Surya 可選引擎安全停用

---

## 檔案架構

```
OCR/
├── ocr_core.py          # 核心 OCR 邏輯、OCRConfig、OCRProcessor
├── ocr_engine.py        # CLI 入口（互動模式 + argparse 模式）
├── ocr_gui.py           # tkinter 圖形介面
├── settings_manager.py  # GUI 設定持久化
├── run_gui.bat          # 一鍵啟動 GUI
├── run_cli.bat          # 一鍵啟動互動式 CLI
├── data/
│   └── gui_settings.json   # GUI 設定記憶（自動產生）
└── logs/
    └── ocr_gui_YYYYMMDD_HHMMSS.log   # GUI 執行日誌（自動產生）
```

---

## GUI 使用方法

```powershell
python .\ocr_gui.py
```

或直接雙擊：

```
run_gui.bat
```

GUI 功能：
1. 設定來源 PDF 或資料夾
2. 設定輸出資料夾
3. 調整 OCR 設定（裝置、Zoom、預處理模式、Claude 校對）
4. 點選「開始批次 OCR」
5. 可隨時暫停、繼續、取消

---

## CLI 使用方法

### 互動模式（原版行為）

```powershell
python .\ocr_engine.py
```

或：

```
run_cli.bat
```

依序輸入：
1. 來源路徑（PDF 單檔或資料夾）
2. 輸出路徑
3. 裝置選擇（1=GPU, 2=CPU）
4. 是否啟用 Claude 校對（y/n）

### argparse 模式（腳本/自動化）

```powershell
python .\ocr_engine.py --input "D:\scan" --output "D:\output" --device gpu --claude off
```

所有參數：

| 參數 | 說明 | 預設 |
|------|------|------|
| `--input`, `-i` | 來源 PDF 或資料夾 | 必填 |
| `--output`, `-o` | 輸出資料夾 | 必填 |
| `--device` | `gpu` 或 `cpu` | `gpu` |
| `--claude` | `on` 或 `off` | `off` |
| `--recursive` | 遞迴處理子資料夾 | 停用 |
| `--overwrite` | 覆蓋已有輸出 | 停用 |
| `--skip-existing` | 跳過已有輸出 | 啟用 |
| `--zoom` | `2`、`3`、`4` | `3` |
| `--preprocess` | `auto`、`original`、`clahe`、`binary` | `auto` |
| `--no-pdf` | 不輸出 PDF | 停用 |
| `--no-txt` | 不輸出 TXT | 停用 |
| `--no-analysis` | 不輸出品質分析 | 停用 |
| `--no-verify-log` | 不輸出校正日誌 | 停用 |

查看說明：

```powershell
python .\ocr_engine.py --help
```

---

## 批次處理

- 輸入路徑若為資料夾，自動掃描全部 `.pdf`
- `--recursive` 可遞迴處理子資料夾
- GUI 批次清單支援多選、移除、重試失敗檔案
- 預設啟用 `skip_existing`：輸出 PDF 已存在時跳過

---

## 設定記憶（GUI）

GUI 自動記憶設定至：

```
C:\Users\Rossi\Documents\Claude\OCR\data\gui_settings.json
```

記憶內容：
- 最後使用的來源 / 輸出路徑
- 裝置選擇（GPU/CPU）
- Claude 校對開關與模型名稱
- Zoom、預處理模式
- 輸出選項（PDF/TXT/分析/日誌）
- Overwrite / Skip existing 設定
- 置信度門檻
- 視窗位置與大小

**注意**：API Key 不會儲存在設定檔，請使用環境變數設定。

---

## GPU / CPU 選擇

GPU 模式啟動時會自動：
1. 偵測 cuDNN / cuBLAS DLL 是否存在
2. 在獨立子程序執行 Conv2D 測試
3. 若任一步驟失敗，自動回退 CPU，不會崩潰

GUI 提供「測試 GPU」按鈕可手動觸發測試。

---

## Claude API Key

```powershell
# PowerShell（僅當前 session）
$env:CLAUDE_API_KEY="sk-ant-..."

# 永久設定（Windows 環境變數）
[System.Environment]::SetEnvironmentVariable("CLAUDE_API_KEY","sk-ant-...","User")
```

指定模型：

```powershell
$env:CLAUDE_MODEL="claude-3-5-sonnet-20240620"
```

**API Key 絕不儲存在任何設定檔或程式碼中。**

---

## 輸出檔案用途

每份來源 PDF 會同時產生：

| 檔案 | 說明 |
|------|------|
| `<原檔名>_OCR.pdf` | 含隱藏文字層的可搜尋 PDF |
| `<原檔名>_OCR.txt` | 純文字，含頁碼分隔 |
| `<原檔名>_OCR_analysis.txt` | 每頁品質分析（置信度、疑似亂碼、校正項目） |

輸出資料夾另保留：

| 檔案 | 說明 |
|------|------|
| `verify_log.txt` | 校正前後對照日誌 |

---

## 暫停／取消

- **暫停**：目前頁面完成後停止（不中斷正在執行的 OCR）
- **繼續**：從下一頁繼續
- **取消目前任務**：跳過剩餘頁面，進入下一個檔案
- **全部取消**：停止整個批次

---

## 常見錯誤

| 症狀 | 解決方式 |
|------|---------|
| `Python 已停止運作` | 使用 GPU 模式但 DLL 不完整；程式會自動偵測並回退 CPU |
| `AuthenticationError` | CLAUDE_API_KEY 無效或未設定 |
| `PackageNotFoundError: paddleocr` | 執行安裝腳本 `install_ocr_gpu.ps1` |
| NumPy 2.x 衝突警告 | `pip install "numpy<2.0"` |
| Pillow 11.x 衝突警告 | `pip install "pillow<11.0"` |
| 設定檔損毀 | 自動重新命名為 `gui_settings.broken.json`，使用預設值 |

---

## Windows DLL 說明

程式啟動時自動加入以下 DLL 搜尋路徑：

- `site-packages/nvidia/cublas/bin`
- `site-packages/nvidia/cudnn/bin`
- `site-packages/nvidia/cuda_runtime/bin`
- `site-packages/paddle/base`
- `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`
- `%CUDA_PATH%\bin`（若環境變數存在）

若仍找不到 DLL，改用 CPU 模式。

---

## Nougat / Surya 安全停用

Nougat（物理公式）與 Surya（財經雙欄）為可選引擎，預設**停用**：

```powershell
# 啟用可選引擎測試（僅供測試用）
$env:OCR_ENABLE_OPTIONAL_ENGINES="1"
```

停用時 PaddleOCR 獨立完整運作，不因 PyTorch DLL 問題崩潰。

---

## 固定套件版本

```
Python            3.12.x
paddlepaddle-gpu  2.6.1
paddleocr         2.10.0
nvidia-cudnn-cu11 8.9.5.29
numpy             1.26.4
pillow            10.4.0
opencv-python     4.10.0.84
albumentations    1.4.11
albucore          0.0.16
tifffile          2024.9.20
```

---

## 不可隨意升級的套件

以下套件**禁止**單獨執行 `pip install --upgrade`：

- `numpy`（2.x 與 PaddleOCR 不相容）
- `pillow`（11.x 與 PaddleOCR 不相容）
- `opencv-python`（避免與 headless 版衝突）
- `albumentations`（2.x 參數不相容）
- `albucore`（配合 albumentations）
- `paddleocr`（3.x API 不相容）
- `paddlepaddle-gpu`（需配合 cuDNN 8.x）

若需更新，請使用完整安裝腳本 `install_ocr_gpu.ps1` 重新安裝。
