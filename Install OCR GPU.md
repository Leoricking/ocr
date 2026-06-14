# OCR GPU v4.1.0 安裝與使用

## 一鍵安裝

請將這五個檔案放在同一個 OCR 專案資料夾：

- install_ocr_gpu.bat
- install_ocr_gpu.ps1
- requirements-ocr-gpu.txt
- ocr_engine_v4.1.0.py
- INSTALL_OCR_GPU.md

直接雙擊 `install_ocr_gpu.bat`。

安裝程式會移除衝突版本、安裝固定套件、補齊 cuDNN 8、實際執行 GPU Conv2D 測試，並產生 `requirements-working.txt`。

## 啟動

```powershell
python .\ocr_engine_v4.1.0.py
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
