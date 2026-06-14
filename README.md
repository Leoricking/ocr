# OCR Engine v4.2.0 老舊掃描增強版

OCR Engine 是一套以 **PaddleOCR + PyMuPDF + OpenCV** 為核心的 PDF 搜尋化工具，主要用來把原本無法搜尋、無法複製文字的掃描 PDF，轉換成：

- 可搜尋文字的 OCR PDF
- 對應頁碼的純文字檔
- OCR 校正紀錄
- 可選擇使用 Claude API 進一步修正錯字、術語與公式

本專案特別針對以下文件最佳化：

- 泛黃、低對比、模糊的舊書與講義
- 帶印章、手寫旁註、掃描陰影的 PDF
- 繁體中文書籍與雜誌
- 台大物理、電磁學、電機工程講義
- 財經、投資、基金與週刊類文件
- 含英文、數字、公式與頁碼的混合內容

---

## 專案目的

大量掃描 PDF 通常只有圖片，無法直接搜尋、複製或建立知識庫。

OCR Engine 的目的，是把這些文件轉成可搜尋、可整理、可再利用的資料，適合用於：

- 建立個人電子書搜尋庫
- 將舊講義、雜誌與書籍轉成可全文搜尋 PDF
- 匯出純文字後交給 AI 摘要、分類或建立知識庫
- 比對 OCR 原始結果與校正結果
- 批次處理資料夾內多份 PDF
- 保存原始頁面外觀，同時加入透明文字層

---

## 主要功能

### 1. PDF 批次 OCR

來源路徑可以是：

- 單一 PDF 檔案
- 包含多份 PDF 的資料夾

若輸入資料夾，程式會自動依序處理其中所有 `.pdf` 檔案。

### 2. 三路影像預處理

每一頁會建立三種 OCR 候選：

1. 原始影像
2. CLAHE 對比增強
3. 自適應二值化

程式會依照平均置信度、有效文字量與噪聲比例，自動選擇較佳結果。

這對泛黃紙張、淡字、低對比、掃描陰影與輕微模糊特別有幫助。

### 3. 保留原 PDF 外觀

程式不會重新排版整份 PDF。

原始頁面影像會保留，辨識文字會以透明文字層嵌入原頁面，因此可：

- 在 PDF 閱讀器中搜尋文字
- 複製文字
- 保持原始版面、圖片與頁面尺寸
- 避免 OCR 後版面重排

### 4. 輸出純文字檔

每份 PDF 除了產生 `_OCR.pdf`，還會同步產生 `_OCR.txt`。

文字檔會保留頁碼分隔，例如：

```text
=== PAGE 1 ===
第一頁文字內容

=== PAGE 2 ===
第二頁文字內容
```

可用於：

- 全文搜尋
- AI 摘要
- 知識庫匯入
- 文件分類
- OCR 品質比對

### 5. GPU / CPU 雙模式

啟動後可選擇：

```text
1: GPU[RTX 3060]
2: CPU
```

GPU 模式會先檢查：

- CUDA Runtime
- cuDNN 8
- cuBLAS Lt
- PaddlePaddle GPU
- 實際 Conv2D GPU 運算

GPU 自我測試在獨立子程序中執行。若 CUDA、cuDNN 或 DLL 發生問題，主程式不會直接崩潰，而會自動回退 CPU。

### 6. GPU DLL 自動載入

Windows 啟動時會自動尋找並加入：

- `nvidia/cublas/bin`
- `nvidia/cudnn/bin`
- `nvidia/cuda_runtime/bin`
- Paddle 相關 DLL 路徑
- CUDA 11.8 安裝路徑

主要用來避免：

- 找不到 `cudnn_ops_infer64_8.dll`
- 找不到 `cublasLt64_11.dll`
- Paddle GPU 原生崩潰
- Python 直接停止運作

### 7. 可選 Claude AI 校對

Claude 校對不是必要功能。

只做本地 OCR 時，在啟動畫面選：

```text
是否啟用 Claude 自動校對？ (y/n): n
```

啟用後，程式會根據內容模式使用不同校對提示：

- 一般繁體中文
- 財經與投資內容
- 物理、電磁學與工程內容
- LaTeX 與公式內容

AI 校對會嘗試修正：

- OCR 錯字
- 繁簡轉換問題
- 英文拼字
- 專業術語
- 標點與斷句
- 物理公式與 LaTeX
- 財經術語與數字

無法確定的內容會盡量保留原字，不應憑空生成內容。

### 8. 專業錯字修正

程式內建部分常見錯字，例如：

```text
白大       → 台大
輻樹聲     → 賴樹聲
貨格考     → 資格考
理輪       → 理論
梨力學     → 熱力學
電形學     → 電磁學
Post-Doctoe → Post-Doctor
```

這些規則在不使用 Claude API 時也會套用。

### 9. 專用 OCR 引擎擴充

程式支援偵測以下可選引擎：

- PaddleOCR：預設與主要 OCR 引擎
- Nougat：物理、數學與 LaTeX 文件
- Surya OCR：財經雜誌、多欄版面

未安裝 Nougat 或 Surya 時，程式會自動使用 PaddleOCR，不影響基本功能。

### 10. 校正與驗證日誌

每次處理會產生：

```text
verify_log.txt
```

內容會記錄：

- 檔名
- 頁碼
- 使用引擎
- 原始 OCR
- 修正後文字
- 是否使用 AI 校對

方便後續核對與追蹤 OCR 品質。

---

## 輸出結果

假設來源檔案為：

```text
賴樹聲 電磁波.pdf
```

輸出資料夾會包含：

```text
賴樹聲 電磁波_OCR.pdf
賴樹聲 電磁波_OCR.txt
verify_log.txt
```

其中：

- `_OCR.pdf`：保留原始頁面並加入透明文字層
- `_OCR.txt`：依頁碼輸出的純文字
- `verify_log.txt`：OCR 與校正紀錄

---

## 系統需求

建議環境：

```text
Windows 10 / 11
Python 3.12.x
NVIDIA RTX 3060 或其他 CUDA 11.8 相容 GPU
PaddlePaddle GPU 2.6.1
PaddleOCR 2.10.0
CUDA Runtime 11.8
cuDNN 8.9
```

沒有 NVIDIA GPU 也可以使用 CPU 模式。

---

## 專案檔案

必要檔案：

```text
ocr_engine.py
install_ocr_gpu.ps1
requirements-ocr-gpu.txt
README.md
Install OCR GPU.md
```

建議目錄：

```text
OCR/
├─ ocr_engine.py
├─ install_ocr_gpu.ps1
├─ requirements-ocr-gpu.txt
├─ README.md
├─ Install OCR GPU.md
├─ Demo/
└─ Install/
```

---

## 安裝方式

### 方法一：使用一鍵 GPU 安裝腳本

開啟 PowerShell：

```powershell
cd "C:\Users\Rossi\Documents\Claude\OCR"
```

執行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\install_ocr_gpu.ps1"
```

安裝腳本會處理固定套件、PaddleOCR、PaddlePaddle GPU、CUDA Runtime、cuDNN 與 cuBLAS 相依套件。

安裝完成後檢查：

```powershell
python -m pip check
```

理想結果：

```text
No broken requirements found.
```

檢查 PaddleOCR：

```powershell
python -c "from paddleocr import PaddleOCR; print('PaddleOCR import OK')"
```

檢查 GPU：

```powershell
python -c "import paddle; paddle.set_device('gpu:0'); x=paddle.randn([1,3,64,64]); w=paddle.randn([8,3,3,3]); y=paddle.nn.functional.conv2d(x,w); print('GPU/cuDNN/cuBLAS OK:', list(y.shape))"
```

### 方法二：依 requirements 安裝

```powershell
python -m pip install -r requirements-ocr-gpu.txt
```

PaddlePaddle GPU 的安裝來源可能因平台與 CUDA 版本不同，建議仍優先使用 `install_ocr_gpu.ps1`。

---

## 啟動方式

正式啟動檔名統一為：

```powershell
python .\ocr_engine.py
```

不要再使用：

```text
ocr_engine_v4.1.0.py
ocr_engine_v4.1.1.py
ocr_engine_v4.2.0.py
```

目前正式主程式就是：

```text
ocr_engine.py
```

---

## 操作流程

啟動：

```powershell
cd "C:\Users\Rossi\Documents\Claude\OCR"
python .\ocr_engine.py
```

程式會依序詢問：

### 1. 來源路徑

可輸入單一 PDF：

```text
C:\Books\賴樹聲 電磁波.pdf
```

也可輸入整個資料夾：

```text
C:\Books\Input
```

輸入資料夾時，程式會批次處理資料夾內所有 PDF。

### 2. 輸出路徑

例如：

```text
C:\Books\Output
```

若目錄不存在，程式會自動建立。

### 3. 選擇 GPU 或 CPU

```text
1: GPU[RTX 3060]
2: CPU
```

第一次建議選 GPU `1`。

若 GPU 環境異常，程式會自動回退 CPU。

### 4. 是否啟用 Claude 校對

只測本地 OCR：

```text
n
```

啟用 Claude：

```text
y
```

首次驗證建議先選 `n`，使用 3～10 頁 PDF 測試輸出品質與速度。

---

## Claude API 設定

啟用 Claude 前，在同一個 PowerShell 視窗設定：

```powershell
$env:CLAUDE_API_KEY="你的 API Key"
```

需要指定模型時：

```powershell
$env:CLAUDE_MODEL="模型名稱"
```

再執行：

```powershell
python .\ocr_engine.py
```

若未設定 API Key，程式會自動取消 AI 校對並繼續本地 OCR。

注意：Claude API 會產生額外費用，費用依模型與輸入文字量計算。

---

## 第一次測試建議

先準備 3～10 頁 PDF，執行：

```powershell
python .\ocr_engine.py
```

選擇：

```text
來源：測試 PDF
輸出：空白測試資料夾
裝置：1
Claude：n
```

確認：

- `_OCR.pdf` 可以正常開啟
- PDF 內可以搜尋文字
- `_OCR.txt` 有正確頁碼與文字
- `verify_log.txt` 有處理紀錄
- GPU 測試成功或正確回退 CPU
- 頁面尺寸與原始 PDF 一致

確認後再批次處理大量書籍。

---

## 常見問題

### 找不到 `cudnn_ops_infer64_8.dll`

重新執行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\install_ocr_gpu.ps1"
```

並確認：

```powershell
python -m pip show nvidia-cudnn-cu11
```

### 找不到 `cublasLt64_11.dll`

確認已安裝：

```powershell
python -m pip show nvidia-cublas-cu11
```

### GPU 測試失敗

程式會自動回退 CPU，不會中止整批 OCR。

可以執行：

```powershell
python -c "import paddle; print(paddle.__version__); print(paddle.device.is_compiled_with_cuda()); print(paddle.device.get_device())"
```

### OCR 結果仍有錯字

可依序嘗試：

1. 啟用 Claude 校對
2. 使用更清楚的掃描來源
3. 對特定術語擴充 `HARD_CODED_CORRECTIONS`
4. 將少數問題頁以更高解析度重新掃描
5. 安裝 Nougat 或 Surya 作為專用引擎

### PDF 被其他程式占用

若目標 PDF 正被開啟，程式會改用時間戳檔名輸出，避免整批失敗。

### 沒有 NVIDIA GPU

選擇：

```text
2: CPU
```

CPU 模式仍可完整輸出 OCR PDF 與文字檔，只是速度較慢。

---

## 固定環境注意事項

不要任意單獨升級：

```powershell
pip install --upgrade numpy
pip install --upgrade pillow
pip install --upgrade opencv-python
pip install --upgrade albumentations
pip install --upgrade albucore
pip install --upgrade paddleocr
```

可能造成：

- NumPy 2.x 衝突
- OpenCV 套件互相覆蓋
- Albumentations / Albucore 不相容
- PaddleOCR 3.x 參數不相容
- cuDNN DLL 找不到
- Paddle GPU 原生崩潰

建議使用專案固定的：

```text
requirements-ocr-gpu.txt
```

---

## 使用情境

本工具適合：

- 大量舊書掃描搜尋化
- 學術講義與研究資料整理
- 物理、電磁學與工程教材 OCR
- 財經雜誌與投資資料數位化
- 個人 PDF 知識庫建立
- AI 摘要前的文字抽取
- 文件全文索引
- 檔案批次轉換與備份

---

## 版本摘要

### v4.2.0

- 新增原圖、CLAHE、自適應二值化三路候選
- 自動選擇最佳 OCR 結果
- 強化泛黃、淡字、印章與模糊掃描
- 新增 `_OCR.txt` 純文字輸出
- 保留透明文字層與原始版面
- 保留 GPU DLL 自動載入
- 保留獨立 GPU Conv2D 自我測試
- GPU 失敗時自動回退 CPU
- 擴充物理、電磁學與繁體中文校正
- 可選 Claude AI 批次校對
- 正式啟動檔名統一為 `ocr_engine.py`

---

## 授權與資料注意事項

請只處理你有權使用、備份或數位化的 PDF 文件。

不要將受版權保護的測試 PDF、私人文件、API Key、環境快照或 OCR 輸出直接上傳到公開 GitHub 儲存庫。
