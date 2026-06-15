# OCR Engine v4.5.5 — PDF 搜尋化與品質分析系統

## 專案功能

將掃描版 PDF 轉換為可搜尋的文字型 PDF，同時輸出純文字與品質分析報告。

- PaddleOCR（繁體中文）三路預處理候選自動選最佳
- 財經雜誌雙欄閱讀順序自動修正
- Claude AI 批次校對（可選）
- 固定式亂碼修正規則
- 品質分析報告（低置信度、亂碼標記、公式偵測）
- GPU 自動偵測與安全回退 CPU
- Nougat / Surya 可選引擎安全停用
- **GUI 使用獨立 OCR worker process（v4.4.2 新增）**
- 原生 CUDA / Paddle 崩潰不再關閉 GUI
- 每頁 checkpoint 寫入，支援斷點續跑
- GPU crash 自動重試，二次失敗自動改 CPU

---

## 檔案架構

```
OCR/
├── ocr_core.py           # 核心 OCR 邏輯、OCRConfig、OCRProcessor
├── ocr_engine.py         # CLI 入口（互動模式 + argparse 模式）
├── ocr_gui.py            # tkinter 圖形介面（v4.4.2 改用 worker subprocess）
├── ocr_worker.py         # 獨立 OCR worker process（JSON Lines 輸出）
├── worker_controller.py  # Worker 生命週期管理、crash log、control file
├── settings_manager.py   # GUI 設定持久化
├── run_gui.bat           # 一鍵啟動 GUI
├── run_cli.bat           # 一鍵啟動互動式 CLI
├── data/
│   ├── gui_settings.json           # GUI 設定記憶（自動產生）
│   ├── checkpoints/                # 每頁斷點續跑狀態（自動產生）
│   └── runtime_<job_id>.control.json  # 暫停／取消控制（自動產生）
└── logs/
    ├── ocr_gui_YYYYMMDD_HHMMSS.log      # GUI 執行日誌
    ├── worker_<job_id>_YYYYMMDD.log     # Worker stderr 日誌
    └── worker_crash_YYYYMMDD_HHMMSS.log # Worker 崩潰記錄
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
6. 顯示 Worker PID、裝置、最後心跳、當前頁進度、自動重試次數
7. Worker 崩潰後 GUI 繼續存在，顯示 Windows 原生退出代碼與說明
8. checkpoint 自動斷點續跑（同一 PDF 下次從上次失敗頁繼續）
9. GPU crash 自動重試 → 二次失敗自動改 CPU

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

- **暫停**：目前頁面完成後停止（不中斷正在執行的 OCR），透過 control file 通知 worker
- **繼續**：從下一頁繼續
- **取消目前任務**：跳過剩餘頁面，進入下一個檔案
- **全部取消**：先設 control file cancel，等待 10 秒，超時才 terminate，再等 5 秒才 kill

---

## Worker Process Safety（v4.4.2）

從 v4.4.2 起，GUI 不再於 threading.Thread 內執行 PaddleOCR。所有 OCR 都在獨立 subprocess（`ocr_worker.py`）中執行。

**架構：**

```
ocr_gui.py          只負責 UI、啟動 worker、讀取 JSON Lines
  └─ subprocess.Popen ──▶ ocr_worker.py  (PaddleOCR 在此執行)
       stdout: JSON Lines  ──▶ WorkerController stdout_reader thread
       stderr: worker log 檔
```

**Worker stdout 協議（每行一筆 JSON）：**

```json
{"type":"worker_started","pid":1234}
{"type":"environment","device":"gpu","pid":1234,"python":"3.12.0"}
{"type":"file_started","path":"...","index":1,"total":3}
{"type":"page_progress","path":"...","page":5,"pages":25,"stage":"OCR","preprocess":"CLAHE","elapsed":12.4}
{"type":"file_completed","path":"...","pdf":"...","txt":"...","analysis":"..."}
{"type":"file_failed","path":"...","error":"...","traceback":"..."}
{"type":"batch_completed","completed":2,"failed":1,"cancelled":0}
{"type":"heartbeat","timestamp":1234567890.0}
```

**Worker crash 處理：**

| Windows 退出代碼 | 說明 |
|------|------|
| `-1073741819` (`0xC0000005`) | 原生程式庫存取違規（CUDA/Paddle/OpenCV/PyMuPDF） |
| `-1073740791` (`0xC0000409`) | Stack buffer overrun |
| `-1073741502` (`0xC0000142`) | DLL 初始化失敗 |
| `-1073741515` (`0xC0000135`) | 找不到必要 DLL |

Worker 崩潰後 GUI 顯示退出代碼說明，並在 `logs/worker_crash_YYYYMMDD_HHMMSS.log` 記錄詳細資訊。

**如何關閉卡住的 worker：**

GUI 按「全部取消」→ 等待 10 秒 → 若未退出自動 terminate → 再等 5 秒 → 自動 kill。

或手動：

```powershell
# 查 worker PID（GUI 進度區有顯示）
taskkill /PID <PID> /F
```

**checkpoint 斷點續跑：**

每頁完成後自動寫入 `data/checkpoints/.ocr_state_<hash>.json`。
若 worker 在第 12 頁崩潰，下次執行同一 PDF 時自動從第 13 頁繼續。
若來源 PDF 的大小或修改時間改變，checkpoint 自動失效，從頭重新處理。

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

---

## v4.5.2 安全校正與前後對照修正版

本版針對《今周刊：摸透主力思維》、《賴樹聲 電磁波》與 Newton 雜誌實測結果，修正以下問題：

1. **禁止重複 OCR**
   - 預設排除 `*_OCR.pdf`、`*_OCR_OCR.pdf`、`*_searchable.pdf`、`.part` 與輸出資料夾。
   - 避免產生 `_OCR_OCR.pdf`、文字層重疊與重複搜尋結果。

2. **安全校正規則**
   - 不再使用會誤傷全文的單字或寬鬆 substring 規則。
   - 校正分成：完整行、完整片語、文件限定、上下文限定。
   - 公式密集行預設不自動猜測或刪字。

3. **同時輸出 OCR 前後版本**
   - `<檔名>_OCR_raw.txt`：PaddleOCR 原始文字。
   - `<檔名>_OCR_corrected.txt`：規則／AI 校正後文字。
   - `<檔名>_OCR.txt`：相容舊流程的最終校正版。

4. **真正的校正日誌**
   - `verify_log.txt` 只記錄實際有變動的行。
   - 每筆包含：頁碼、行號、原始文字、修正文字、規則來源。
   - 未修改的 Raw OCR 不再灌入大量日誌。

5. **分析報告強化**
   - `_OCR_analysis.txt` 顯示校正來源。
   - 公式、英文斷詞、單字碎片與低置信度仍標示人工複核。

### 已內建並核對的安全修正

- `股價漲多就有懼高痘?` → `股價漲多就有懼高症？`
- `日時寺間]是什麼?` → `「時間」是什麼？`
- `T時間]是什麼?` → `「時間」是什麼？`
- `http://ww.inewton.com.tw` → `http://www.inewton.com.tw`
- 《賴樹聲 電磁波》限定：
  - `合大物理系第一名畢業` → `台大物理系第一名畢業`
  - `1977白大物理採第一名畢業` → `1977台大物理系第一名畢業`
  - `記得吃古大物理不時門大於必修籠磁學` → `記得唸台大物理系時，大二必修電磁學`

### 重要限制

數學、物理公式、希臘字母、上下標、向量箭頭及手寫內容不會用字典或 LLM 靜默腦補。系統會保留原始 OCR 並在分析檔標示需人工核對。

## v4.5.3：選擇主要輸出格式

GUI 的「輸出選項」提供互斥選擇：

- **全輸出（PDF + TXT）**：同時建立搜尋化 PDF 與文字檔。
- **只輸出 PDF**：只建立搜尋化 PDF，不建立主要 TXT、raw TXT、corrected TXT。
- **只輸出 TXT**：只建立文字輸出，不建立搜尋化 PDF。

品質分析與校正日誌仍是獨立選項，可依需求開啟或關閉。上次選擇會保存在 `data/gui_settings.json`，下次啟動自動還原。

CLI 可使用 `--output-mode all|pdf|txt`。

---

# 安裝與維護

## GPU 安裝與啟動

## 一鍵安裝

必要檔案放在同一個 OCR 專案資料夾：

- `install_ocr_gpu.ps1`
- `ocr_engine.py`
- `README.md`
- `Install OCR GPU.md`

已確認可正常執行的方式：

```powershell
cd "C:\Users\Rossi\Documents\Claude\OCR"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\install_ocr_gpu.ps1"
```

不要只輸入帶引號的 BAT 路徑；PowerShell 會把它視為文字而非執行命令。

## 啟動

```powershell
python .\ocr_engine.py
```

選擇：

```text
1: GPU[RTX 3060]
```

新版會先在獨立子程序測試 CUDA/cuDNN。若 DLL 缺失或 GPU 原生核心異常，主程式不會直接崩潰，而會自動回退 CPU。

## Claude API Key

只做本地 OCR 可選 `n`，不需 API Key。

啟用 Claude 校對前：

```powershell
$env:CLAUDE_API_KEY="你的 API Key"
```

需要更換模型時：

```powershell
$env:CLAUDE_MODEL="模型名稱"
```

## 注意

不要單獨升級 NumPy、Pillow、OpenCV、Albumentations、Albucore、PaddleOCR，否則可能重新產生版本衝突。


## Nougat / Surya 可選引擎

Nougat 與 Surya 不是主程式必要相依套件。它們會載入 PyTorch，可能因
`torch\lib\shm.dll`、Visual C++ Runtime 或 CUDA 版本不相容而失敗。

v4.2.1 預設安全停用這兩個引擎，不會再因已安裝但損壞的 Torch 而使
`ocr_engine.py` 在啟動前崩潰。

先直接執行：

```powershell
python .\ocr_engine.py
```

若確定 Torch 可正常 import，才測試：

```powershell
$env:OCR_ENABLE_OPTIONAL_ENGINES="1"
python .\ocr_engine.py
```

可選引擎測試失敗時，程式會自動停用並回退 PaddleOCR。

---

# 版本更新紀錄

## v4.5.5 — GUI 排版與欄位對齊


## GUI Layout Polish

### Treeview Column Alignment
- Renamed all Treeview column keys from Chinese to English identifiers
  (`index`, `filename`, `pages`, `status`, `progress`, `avg_sec`,
  `elapsed_time`, `output_pdf`, `output_txt`, `analysis`)
- All heading anchors set to `center`
- Cell anchors: `center` for numeric/status columns; `w` (left-aligned) for
  `filename`, `output_pdf`, `output_txt`, `analysis`
- Explicit base widths (pre-DPI-scale): index=50, filename=270, pages=70,
  status=90, progress=90, avg_sec=100, elapsed_time=100, output_pdf/txt/analysis=180
- `stretch=True` on `filename`, `output_pdf`, `output_txt`, `analysis`;
  `stretch=False` on all numeric/status columns
- Horizontal scrollbar preserved
- `_resize_tree_columns` updated to use new English column keys

### Source Section Layout Rebuild
- Entries and button rows now stacked vertically inside the LabelFrame
  (row 0: label + source entry; row 1: 5-button toolbar; row 2: label +
  output entry; row 3: 2-button toolbar; row 4: output mode frame; row 5:
  recursive checkbutton)
- Source buttons in a single horizontal toolbar:
  "選擇單一 PDF", "選擇多個 PDF", "選擇資料夾", "清除來源", "重新掃描"
- Output buttons in a single horizontal toolbar:
  "選擇輸出資料夾", "開啟輸出資料夾"
- `padx=(0, 6)` between buttons except last in each row
- `columnconfigure(1, weight=1)` on the LabelFrame for responsive entry width
- All existing callbacks preserved; DPI scaling preserved

### Version
- Bumped from v4.5.4 to v4.5.5

### Tests Added
- `tests/test_gui_layout.py`: asserts source and output button callbacks exist
  on `OCRGuiApp`, and that `_var_output_mode` / `_var_recursive` are defined
- `tests/test_treeview_alignment.py`: asserts all 10 column keys are present,
  heading anchors are `center`, cell anchors match the spec, stretch flags
  are correct, and base widths match the spec

## v4.5.4 — 多選 PDF 與批次計時


## New Features

### Multi-PDF Selection (Part A)
- Added **選擇單一 PDF** button (replaces old 選擇 PDF)
- Added **選擇多個 PDF** button using `filedialog.askopenfilenames`
- Kept **選擇資料夾** for directory-based batch loading
- Source buttons reorganised into two rows:
  - Row 0: 選擇單一 PDF | 選擇多個 PDF
  - Row 1: 選擇資料夾 | 清除來源 | 重新掃描

### Shared Dedup Logic (`_add_pdf_paths_to_batch`)
- Normalises paths via `Path.resolve().casefold()` for cross-platform dedup
- Filters out `_OCR.pdf`, `_OCR_OCR.pdf`, `_searchable.pdf`, `.part` files
- Skips files inside the configured output folder
- Skips paths already in the batch list
- Skips duplicates within the same selection call
- Gets page count via fitz/PyMuPDF; marks as `"?"` on failure without aborting
- Persists `last_input_directory` in settings after each selection

### Per-File and Batch Elapsed Time (Part B)
- New **花費時間** column in the Treeview (between 平均秒/頁 and 輸出PDF)
- Displayed as `HH:MM:SS` (`_format_duration` helper)
- File elapsed time starts at `file_started` event, freezes at `file_completed` / `file_failed`
- Batch elapsed timer runs every 500 ms (`TIMER_REFRESH_MS`)
- Only one `root.after` chain active at a time (stored in `_timer_id`)
- Timer cancelled on batch completion, pause, and window close

### Progress Section Labels (Part B4)
- 目前檔案花費：HH:MM:SS
- 目前檔案剩餘：HH:MM:SS (linear estimate from page throughput)
- 整批花費時間：HH:MM:SS
- 整批預估剩餘：HH:MM:SS
- 預計完成：HH:MM:SS (wall-clock ETA)

### Worker Events (Part C — `ocr_worker.py`)
- `file_started` now includes `started_monotonic`
- `page_progress` now includes `elapsed_seconds` (time since file started)
- `file_completed` now includes `elapsed_seconds`
- `file_failed` now includes `elapsed_seconds`
- `batch_completed` now includes `skipped` count

### Completion Log Messages (Part D)
After each file:
```
[完成] <filename>
  頁數：N
  花費時間：HH:MM:SS
  平均速度：X.XX 秒／頁
```
After batch:
```
[批次完成]
  總檔案：N
  完成：N  失敗：N  取消：N  跳過：N
  整批花費時間：HH:MM:SS
  總處理頁數：N
  平均每頁：X.XX 秒
```

## Settings
- `last_input_directory` added to `settings_manager.py` DEFAULTS

## Version
- Version string updated to `v4.5.4`
- GUI title: `OCR Engine v4.5.4 — PDF 搜尋化與品質分析系統`

## Tests Added
- `tests/test_multi_pdf_selection.py` — selection modes and dedup
- `tests/test_batch_add_paths.py` — folder scanning and edge cases
- `tests/test_elapsed_time.py` — `_format_duration` and per-file timing
- `tests/test_gui_elapsed_time.py` — Treeview column and timer management

## Preserved
- CLI interface unchanged
- PaddleOCR GPU/CPU fallback unchanged
- Worker subprocess architecture unchanged
- UTF-8 output unchanged
- Output modes (all/pdf/txt) unchanged
- Proofreading/Claude integration unchanged
- Memory guard and checkpoint/resume unchanged
- Pause/resume/cancel unchanged
- Settings persistence unchanged
- Existing source filtering (`_is_source_pdf`) unchanged

## v4.5.3 — 輸出模式


## 主要輸出模式

GUI 新增三個互斥選項：

- 全輸出（PDF + TXT）
- 只輸出 PDF
- 只輸出 TXT

設定會自動記憶；舊版 `output_pdf` / `output_txt` 設定會自動遷移。品質分析與校正日誌仍可獨立開關。

CLI 新增：

```powershell
python .\ocr_engine.py --input "C:\PDFs" --output "C:\Output" --output-mode all
python .\ocr_engine.py --input "C:\PDFs" --output "C:\Output" --output-mode pdf
python .\ocr_engine.py --input "C:\PDFs" --output "C:\Output" --output-mode txt
```

舊的 `--no-pdf` 與 `--no-txt` 仍保留相容性，但不可同時停用 PDF 與 TXT。
