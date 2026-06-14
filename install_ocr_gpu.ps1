$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

function Stop-WithMessage([string]$Message) {
    Write-Host ""
    Write-Host "[失敗] $Message" -ForegroundColor Red
    Read-Host "按 Enter 結束"
    exit 1
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " OCR GPU v4.1.2 單檔一鍵修復安裝" -ForegroundColor Cyan
Write-Host " 不需要 requirements 外部檔案" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

try {
    $PythonVersion = (& python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))").Trim()
    $PythonMinor = (& python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
} catch {
    Stop-WithMessage "找不到 python。請先安裝 Python 3.12 x64。"
}

Write-Host "[1/8] Python：$PythonVersion"
if ($PythonMinor -ne "3.12") {
    Stop-WithMessage "目前是 Python $PythonVersion；本安裝程式固定支援 Python 3.12。"
}

Write-Host "[2/8] 更新 pip / setuptools / wheel..."
python -m pip install --upgrade "pip<27" setuptools wheel
if ($LASTEXITCODE -ne 0) { Stop-WithMessage "更新安裝工具失敗。" }

Write-Host "[3/8] 移除互相衝突的核心套件..."
python -m pip uninstall -y `
    opencv-python `
    opencv-contrib-python `
    opencv-python-headless `
    paddleocr `
    paddlepaddle `
    paddlepaddle-gpu `
    albumentations `
    albucore `
    numpy `
    pillow `
    tifffile `
    nvidia-cudnn-cu11 `
    nvidia-cublas-cu11 `
    nvidia-cuda-runtime-cu11
if ($LASTEXITCODE -ne 0) { Stop-WithMessage "移除衝突套件失敗。" }

Write-Host "[4/8] 安裝 PaddlePaddle GPU 2.6.1（禁止自動升級 NumPy）..."
python -m pip install --no-cache-dir --no-deps `
    "paddlepaddle-gpu==2.6.1" `
    -f "https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html"
if ($LASTEXITCODE -ne 0) { Stop-WithMessage "PaddlePaddle GPU 安裝失敗。" }

Write-Host "[5/8] 安裝完整固定版本環境..."
python -m pip install --no-cache-dir `
    "numpy==1.26.4" `
    "pillow==10.4.0" `
    "opencv-python==4.10.0.84" `
    "opencv-contrib-python==4.10.0.84" `
    "opencv-python-headless==4.10.0.84" `
    "tifffile==2024.9.20" `
    "scipy==1.13.1" `
    "scikit-image==0.24.0" `
    "scikit-learn==1.5.2" `
    "albucore==0.0.16" `
    "albumentations==1.4.11" `
    "paddleocr==2.10.0" `
    "nvidia-cublas-cu11==11.11.3.6" `
    "nvidia-cuda-runtime-cu11==11.8.89" `
    "nvidia-cudnn-cu11==8.9.5.29" `
    "PyMuPDF==1.24.13" `
    "tqdm==4.67.1" `
    "anthropic>=0.40,<1" `
    "packaging>=24,<27"
if ($LASTEXITCODE -ne 0) { Stop-WithMessage "完整 OCR 相依套件安裝失敗。" }

Write-Host "[6/8] 再次覆蓋關鍵版本，避免 pip 改版..."
python -m pip install --no-cache-dir --force-reinstall --no-deps `
    "numpy==1.26.4" `
    "pillow==10.4.0" `
    "opencv-python==4.10.0.84" `
    "opencv-contrib-python==4.10.0.84" `
    "opencv-python-headless==4.10.0.84" `
    "tifffile==2024.9.20" `
    "albucore==0.0.16" `
    "albumentations==1.4.11" `
    "nvidia-cublas-cu11==11.11.3.6" `
    "nvidia-cuda-runtime-cu11==11.8.89" `
    "nvidia-cudnn-cu11==8.9.5.29"
if ($LASTEXITCODE -ne 0) { Stop-WithMessage "核心版本鎖定失敗。" }

Write-Host "[7/8] 檢查 DLL、PaddleOCR 與 GPU 實際運算..."
$PythonRoot = (& python -c "import os,sys; print(os.path.dirname(sys.executable))").Trim()
$CublasDir = Join-Path $PythonRoot "Lib\site-packages\nvidia\cublas\bin"
$CudnnDir = Join-Path $PythonRoot "Lib\site-packages\nvidia\cudnn\bin"
$CudaRuntimeDir = Join-Path $PythonRoot "Lib\site-packages\nvidia\cuda_runtime\bin"

$env:PATH = "$CublasDir;$CudnnDir;$CudaRuntimeDir;$env:PATH"

$CublasDll = Join-Path $CublasDir "cublasLt64_11.dll"
$CudnnDll = Join-Path $CudnnDir "cudnn_ops_infer64_8.dll"

if (-not (Test-Path $CublasDll)) { Stop-WithMessage "找不到 cublasLt64_11.dll。" }
if (-not (Test-Path $CudnnDll)) { Stop-WithMessage "找不到 cudnn_ops_infer64_8.dll。" }

python -c "from paddleocr import PaddleOCR; print('PaddleOCR import OK')"
if ($LASTEXITCODE -ne 0) { Stop-WithMessage "PaddleOCR 匯入失敗。" }

python -c "import paddle; paddle.set_device('gpu:0'); x=paddle.randn([1,3,64,64]); w=paddle.randn([8,3,3,3]); y=paddle.nn.functional.conv2d(x,w); print('GPU/cuDNN/cuBLAS OK:', list(y.shape))"
if ($LASTEXITCODE -ne 0) {
    Stop-WithMessage "GPU/cuDNN/cuBLAS 實際運算失敗。"
}

python -m pip check
if ($LASTEXITCODE -ne 0) {
    Write-Host "[警告] pip check 發現非致命相依問題，請保留畫面。" -ForegroundColor Yellow
}

Write-Host "[8/8] 備份目前環境..."
python -m pip freeze | Out-File -Encoding utf8 ".\requirements-working.txt"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " 安裝與 GPU 測試完成" -ForegroundColor Green
Write-Host " GPU/cuDNN/cuBLAS 已通過實際卷積測試" -ForegroundColor Green
Write-Host " 啟動：python .\ocr_engine_v4.1.1.py" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Read-Host "按 Enter 結束"
