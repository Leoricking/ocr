# ocr_core.py — Shared OCR core for OCR Engine v4.5.7
# All OCR logic, constants, helpers, and the OCRProcessor class live here.
# This module must be imported BEFORE PaddleOCR is imported anywhere else,
# because the GPU DLL search-path setup must happen first.

import os
import sys
import io
import time
import difflib
import subprocess
import importlib.util
import importlib.metadata as _meta
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional, List, Tuple, Dict
from offline_corrector import correct_lines
from layout_analyzer import PageLayoutAnalyzer
from ocr_quality import BlockTextEvaluator, FINANCE_TERMS
from collections import Counter

# ==============================================================
# Windows GPU DLL search-path setup
# MUST run before PaddleOCR / Paddle are imported
# ==============================================================
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")

_DLL_DIR_HANDLES = []


def _add_windows_dll_directory(path: str) -> bool:
    """Add a DLL directory to the current process and keep the handle."""
    if sys.platform != "win32" or not path or not os.path.isdir(path):
        return False

    normalized = os.path.abspath(path)

    current_path = os.environ.get("PATH", "")
    existing = {item.lower() for item in current_path.split(os.pathsep) if item}
    if normalized.lower() not in existing:
        os.environ["PATH"] = normalized + os.pathsep + current_path

    if hasattr(os, "add_dll_directory"):
        try:
            _DLL_DIR_HANDLES.append(os.add_dll_directory(normalized))
        except OSError:
            return False

    return True


def configure_windows_gpu_dll_paths() -> list:
    """Automatically add cuBLAS, cuDNN, CUDA Runtime, Paddle, and CUDA 11.8."""
    if sys.platform != "win32":
        return []

    python_root = os.path.dirname(sys.executable)

    candidates = [
        os.path.join(python_root, "Lib", "site-packages", "nvidia", "cublas", "bin"),
        os.path.join(python_root, "Lib", "site-packages", "nvidia", "cudnn", "bin"),
        os.path.join(python_root, "Lib", "site-packages", "nvidia", "cuda_runtime", "bin"),
        os.path.join(python_root, "Lib", "site-packages", "paddle", "base"),
        os.path.join(python_root, "Lib", "site-packages", "paddle", "libs"),
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
    ]

    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(os.path.join(cuda_path, "bin"))

    added = []
    for candidate in candidates:
        if _add_windows_dll_directory(candidate):
            added.append(os.path.abspath(candidate))

    return added


# Run DLL setup at import time
GPU_DLL_PATHS = configure_windows_gpu_dll_paths()

# ==============================================================
# Heavy imports (after DLL setup)
# ==============================================================
from tqdm import tqdm
import fitz  # PyMuPDF
from paddleocr import PaddleOCR
try:
    import anthropic  # optional legacy dependency
except Exception:
    anthropic = None
import cv2
import numpy as np


# ==============================================================
# Dependency version checks
# ==============================================================
def _check_dep(pkg: str, max_ver: str):
    try:
        from packaging.version import Version
        ver = Version(_meta.version(pkg))
        limit = Version(max_ver)
        if ver >= limit:
            print(f"[警告] {pkg}=={ver} 可能與 PaddleOCR 衝突，建議使用 <{max_ver}")
    except Exception:
        pass


_check_dep("numpy", "2.0.0")
_check_dep("pillow", "11.0.0")


# ==============================================================
# OCRConfig dataclass
# ==============================================================
@dataclass
class OCRConfig:
    input_path: str = ""
    output_path: str = ""
    device: str = "gpu"                     # "gpu" or "cpu"
    enable_claude: bool = False  # deprecated; offline correction is always used
    recursive: bool = False
    overwrite: bool = False
    skip_existing: bool = True
    zoom: int = 3
    preprocess_mode: str = "auto"           # "auto" | "original" | "clahe" | "binary"
    layout_aware: bool = True
    region_retry_threshold: float = 65.0
    filter_chart_noise: bool = True
    output_pdf: bool = True
    output_txt: bool = True
    output_analysis: bool = True
    output_verify_log: bool = True
    output_raw_txt: bool = True
    exclude_ocr_output: bool = True
    confidence_threshold: float = 0.70
    preserve_relative_structure: bool = True
    claude_model: str = ""                  # deprecated legacy field
    batch_failure_continue: bool = True
    max_page_pixels: int = 24_000_000
    memory_retry_enabled: bool = True
    memory_retry_scales: list = field(default_factory=lambda: [1.0, 0.8, 0.65, 0.5])
    strict_page_failure: bool = False
    release_between_candidates: bool = True


# ==============================================================
# OCRProgress dataclass
# ==============================================================
@dataclass
class OCRProgress:
    file_index: int = 0
    file_total: int = 0
    file_path: str = ""
    page_index: int = 0
    page_total: int = 0
    stage: str = ""
    preprocess_mode: str = ""
    elapsed_seconds: float = 0.0
    average_seconds_per_page: float = 0.0
    estimated_remaining_seconds: float = 0.0


# ==============================================================
# Settings
# ==============================================================
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "YOUR_CLAUDE_API_KEY").strip()
CLAUDE_MODEL_DEFAULT = os.environ.get("CLAUDE_MODEL", "claude-3-5-sonnet-20240620").strip()
BATCH_SIZE = 4
API_TIMEOUT_DEFAULT = 30.0
API_TIMEOUT_PHYSICS = 60.0
CONF_SKIP_THRESHOLD = 0.95
PAGE_SEP = "===PAGE_{n}==="

_client_cache = {}


def get_anthropic_client(model: str = ""):
    raise RuntimeError("Claude API support is disabled; offline correction is always used.")


def validate_key(model: str = "") -> bool:
    print("[離線校正] Claude API 已停用；不需要 API Key。")
    return False


# ==============================================================
# Optional engine detection
# ==============================================================
OPTIONAL_ENGINE_TIMEOUT = 45
ENABLE_OPTIONAL_ENGINES = (
    os.environ.get("OCR_ENABLE_OPTIONAL_ENGINES", "0").strip().lower()
    in {"1", "true", "yes", "y"}
)

ENGINES = {"paddle": True, "nougat": False, "surya": False}
ENGINE_ERRORS = {"nougat": "", "surya": ""}

_nougat_model = None
_surya_model = None
_surya_proc = None
_surya_recog = None
NougatModel = None


def _module_installed(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _safe_import_probe(label: str, code: str, timeout: int = OPTIONAL_ENGINE_TIMEOUT):
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=os.environ.copy(),
            creationflags=(
                subprocess.CREATE_NO_WINDOW
                if sys.platform == "win32" and hasattr(subprocess, "CREATE_NO_WINDOW")
                else 0
            ),
        )
    except subprocess.TimeoutExpired:
        return False, f"{label} import 測試逾時"
    except Exception as exc:
        return False, f"{label} import 測試無法執行：{exc}"

    output = "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part.strip()
    )
    if result.returncode == 0:
        return True, output
    return False, output or f"{label} 子程序結束碼：{result.returncode}"


def detect_optional_engines():
    if not ENABLE_OPTIONAL_ENGINES:
        if _module_installed("nougat") or _module_installed("surya"):
            print("[可選引擎] 已偵測到 Nougat / Surya，但目前採安全停用模式。")
            print("           主程式只使用 PaddleOCR，不會因 Torch DLL 問題崩潰。")
        else:
            print("[可選引擎] Nougat / Surya 未啟用；PaddleOCR 可獨立完整運作。")
        print("           需要測試可選引擎時才設定：$env:OCR_ENABLE_OPTIONAL_ENGINES='1'")
        return

    if _module_installed("nougat"):
        ok, detail = _safe_import_probe(
            "Nougat",
            "from nougat import NougatModel; import torch; "
            "print('NOUGAT_OK', torch.__version__)",
        )
        ENGINES["nougat"] = ok
        if not ok:
            ENGINE_ERRORS["nougat"] = detail
    else:
        ENGINE_ERRORS["nougat"] = "尚未安裝 nougat-ocr"

    if _module_installed("surya"):
        ok, detail = _safe_import_probe(
            "Surya",
            "import torch; "
            "from surya.recognition import batch_recognition; "
            "print('SURYA_OK', torch.__version__)",
        )
        ENGINES["surya"] = ok
        if not ok:
            ENGINE_ERRORS["surya"] = detail
    else:
        ENGINE_ERRORS["surya"] = "尚未安裝 surya-ocr"

    print(
        f"[引擎] Paddle=✓  "
        f"Nougat={'✓' if ENGINES['nougat'] else '✗'}  "
        f"Surya={'✓' if ENGINES['surya'] else '✗'}"
    )

    for engine_name in ("nougat", "surya"):
        if ENGINE_ERRORS[engine_name]:
            first_line = ENGINE_ERRORS[engine_name].splitlines()[0][:180]
            print(f"       → {engine_name.title()} 已停用：{first_line}")
    print("       → 可選引擎失敗不影響 PaddleOCR，程式會自動回退。")


detect_optional_engines()


def _load_nougat():
    global _nougat_model, NougatModel
    if not ENGINES["nougat"]:
        raise RuntimeError(
            "Nougat 不可用。請保持 Paddle 模式，或在獨立環境修復 PyTorch 後再啟用。"
        )
    if _nougat_model is None:
        import torch
        from nougat import NougatModel as _NougatModel
        from nougat.utils.checkpoint import get_checkpoint
        NougatModel = _NougatModel
        ckpt = get_checkpoint(model_tag="0.1.0-base")
        _nougat_model = NougatModel.from_pretrained(ckpt)
        _nougat_model.eval()
        if torch.cuda.is_available():
            _nougat_model = _nougat_model.cuda()
        print("[引擎] Nougat 模型載入完成")
    return _nougat_model


def _load_surya():
    global _surya_model, _surya_proc, _surya_recog
    if not ENGINES["surya"]:
        raise RuntimeError(
            "Surya 不可用。請保持 Paddle 模式，或在獨立環境修復 PyTorch 後再啟用。"
        )
    if _surya_model is None:
        from surya.recognition import batch_recognition as _batch_recognition
        from surya.model.recognition.model import load_model
        from surya.model.recognition.processor import load_processor
        _surya_recog = _batch_recognition
        _surya_model = load_model()
        _surya_proc = load_processor()
        print("[引擎] Surya 模型載入完成")
    return _surya_model, _surya_proc


# ==============================================================
# Keywords and correction tables
# ==============================================================
FINANCE_KEYWORDS = ['股市', '理財', '周刊', '週刊', '基金', '投資', '財經']
PHYSICS_KEYWORDS = ['公式', '向量', '電磁', '物理', 'LaTeX', '微積分', '電場', '磁場']

# Safe correction rules.  Rules are deliberately scoped to exact lines,
# exact phrases, or a specific document.  Do not add broad single-character
# replacements here because they can silently corrupt valid names/formulas.
EXACT_LINE_CORRECTIONS: Dict[str, str] = {
    "股價漲多就有懼高痘?": "股價漲多就有懼高症？",
    "日時寺間]是什麼?": "「時間」是什麼？",
    "T時間]是什麼?": "「時間」是什麼？",
    "http://ww.inewton.com.tw": "http://www.inewton.com.tw",
}

EXACT_PHRASE_CORRECTIONS: Dict[str, str] = {
    "懼高痘": "懼高症",
    "http://ww.inewton.com.tw": "http://www.inewton.com.tw",
}

DOCUMENT_SPECIFIC_CORRECTIONS: Dict[str, Dict[str, str]] = {
    "今周刊：摸透主力思維": {
        "股價漲多就有懼高痘?": "股價漲多就有懼高症？",
    },
    "賴樹聲 電磁波": {
        "合大物理系第一名畢業": "台大物理系第一名畢業",
        "1977白大物理採第一名畢業": "1977台大物理系第一名畢業",
        "記得吃古大物理不時門大於必修籠磁學": "記得唸台大物理系時，大二必修電磁學",
    },
    "Newton牛頓科學2009年8月第22期 「時間」是什麼": {
        "日時寺間]是什麼?": "「時間」是什麼？",
        "T時間]是什麼?": "「時間」是什麼？",
        "http://ww.inewton.com.tw": "http://www.inewton.com.tw",
    },
}

CONTEXT_CORRECTIONS = [
    # Only correct this phrase when the surrounding page clearly discusses
    # probability/gambling.  This avoids a dangerous global replacement.
    ({"機率", "賭"}, "王萬要賭後", "千萬不要賭"),
]

OCR_OUTPUT_SUFFIXES = (
    "_ocr.pdf", "_ocr_ocr.pdf", "_searchable.pdf", ".part",
)


# ==============================================================
# Claude prompts
# ==============================================================
_VERIFY = (
    "\n\n【格式規定】回傳時必須嚴格使用 ===PAGE_N=== 分隔每頁，不得省略或更改格式。"
    "\n【自我檢查】完成後請確認：\n"
    "1. 每頁輸出行數需與輸入完全一致；嚴禁合併含物理公式的行。\n"
    "2. 向量符號（$\\vec{E}$、$\\vec{B}$）與財經術語（殖利率等）不得消失。\n"
    "3. 確認已剔除亂碼字元（土、聶、畢、吳、車）。\n"
    "若整批幾乎全是亂碼，請在回覆最開頭輸出 [UNCERTAIN]，後續仍輸出最佳猜測結果。"
)

PROMPT_NOUGAT = (
    "你是一位精通物理與 LaTeX 的技術編輯。"
    "以下文字由 Nougat 引擎辨識自物理教材，可能含有 LaTeX 語法錯誤（缺失 $、錯誤括號）"
    "或繁體中文說明缺失。請修正 LaTeX 語法並潤飾中文解釋，維持行數不變。"
    + _VERIFY
)

PROMPT_SURYA = (
    "你是一位專業財經編輯，擅長處理多欄位雜誌排版。"
    "以下文字由 Surya 版面分析引擎辨識，可能含有多欄位混排的邏輯錯誤或財經數字不準確。"
    "請還原正確的閱讀順序，核對財經數字與術語準確性，維持行數不變。"
    + _VERIFY
)

PROMPT_PADDLE_FINANCE = (
    "你是一位專業財經編輯，擅長處理股市、理財、基金、週刊等金融內容。"
    "請透過語意推理修正 OCR 亂碼（如輻→賴、白→台），保持繁體中文，維持行數與順序。"
    + _VERIFY
)
PROMPT_PADDLE_PHYSICS = (
    "你是一位精通物理、電機工程與繁體中文排版校對的資深編輯。"
    "請透過語意推理修正 OCR 亂碼，並將物理公式還原為正確 LaTeX 格式（如 $F=ma$、$\\vec{E}$）。"
    "請務必逐行對應回傳，不要合併段落。"
    + _VERIFY
)
PROMPT_PADDLE_DEFAULT = (
    "你是一位專業編輯，請透過語意推理修正 OCR 亂碼，"
    "保持繁體中文，維持原始行數與順序，不要合併段落。"
    + _VERIFY
)


# ==============================================================
# Engine selection
# ==============================================================
def select_engine(raw_texts: list, confidences: list) -> str:
    sample = " ".join(raw_texts)[:500]
    if ENGINES["nougat"] and any(kw in sample for kw in PHYSICS_KEYWORDS):
        return "nougat"
    if ENGINES["surya"] and any(kw in sample for kw in FINANCE_KEYWORDS):
        return "surya"
    return "paddle"


# ==============================================================
# Image preprocessing
# ==============================================================
def _page_to_bgr(page, zoom: int):
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False, colorspace=fitz.csRGB)
    image = np.frombuffer(pix.samples, dtype=np.uint8)
    image = image.reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _is_memory_error(exc: BaseException) -> bool:
    """Return True for Python/NumPy/OpenCV/Paddle allocation failures."""
    if isinstance(exc, MemoryError):
        return True
    msg = str(exc).lower()
    tokens = (
        "unable to allocate", "bad allocation", "std::bad_alloc",
        "out of memory", "outofmemory", "resource exhausted",
        "cuda error: out of memory", "cuda out of memory",
    )
    return any(token in msg for token in tokens)


def _release_memory(use_gpu: bool = False):
    """Best-effort release of Python and optional Paddle GPU caches."""
    import gc
    gc.collect()
    if use_gpu:
        try:
            import paddle
            if paddle.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()
        except Exception:
            pass


def _effective_zoom_for_page(page, requested_zoom: float, max_page_pixels: int) -> float:
    """Cap render size without changing the user's persisted zoom setting."""
    requested_zoom = max(float(requested_zoom), 1.0)
    base_pixels = max(float(page.rect.width) * float(page.rect.height), 1.0)
    requested_pixels = base_pixels * requested_zoom * requested_zoom
    if not max_page_pixels or requested_pixels <= max_page_pixels:
        return requested_zoom
    safe = (float(max_page_pixels) / base_pixels) ** 0.5
    return max(1.5, min(requested_zoom, safe))


def _build_preprocess_candidate(image_bgr, mode: str):
    """Build one candidate at a time to avoid holding RAW/CLAHE/BINARY together."""
    mode = mode.upper()
    if mode == "ORIGINAL":
        return image_bgr
    if mode == "UPSCALE":
        return cv2.resize(image_bgr, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    try:
        denoised = cv2.fastNlMeansDenoising(
            gray, None, h=7, templateWindowSize=7, searchWindowSize=21
        )
        clip_limit = 1.45 if mode == "LIGHT_CLAHE" else 2.2
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        contrast = clahe.apply(denoised)
        if mode == "LIGHT_CLAHE":
            return contrast
        blur = cv2.GaussianBlur(contrast, (0, 0), 1.0)
        sharpened = cv2.addWeighted(contrast, 1.45, blur, -0.45, 0)
        if mode == "CLAHE":
            return sharpened
        binary = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 35, 13
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    finally:
        # Local temporaries not returned are released on scope exit.
        pass


def _scale_ocr_lines(lines, scale: float):
    if not lines or abs(scale - 1.0) < 1e-9:
        return lines
    scaled = []
    for line in lines:
        box, payload = line[0], line[1]
        new_box = [[float(p[0]) * scale, float(p[1]) * scale] for p in box]
        scaled.append([new_box, payload])
    return scaled


# ==============================================================
# Reading-order helpers
# ==============================================================
def _line_box(line):
    box = line[0]
    xs = [float(point[0]) for point in box]
    ys = [float(point[1]) for point in box]
    return min(xs), min(ys), max(xs), max(ys)


def _sort_zone_by_columns(zone_lines, page_width):
    if len(zone_lines) < 6:
        return sorted(zone_lines, key=lambda line: (_line_box(line)[1], _line_box(line)[0]))
    centers = []
    for line in zone_lines:
        x0, _, x1, _ = _line_box(line)
        centers.append(((x0 + x1) / 2.0, line))
    centers.sort(key=lambda item: item[0])
    best_gap = 0.0
    best_index = None
    for index in range(2, len(centers) - 2):
        gap = centers[index][0] - centers[index - 1][0]
        if gap > best_gap:
            best_gap = gap
            best_index = index
    if best_index is None or best_gap < page_width * 0.10:
        return sorted(zone_lines, key=lambda line: (_line_box(line)[1], _line_box(line)[0]))
    left = [item[1] for item in centers[:best_index]]
    right = [item[1] for item in centers[best_index:]]
    left_median = sorted(_line_box(line)[0] for line in left)[len(left) // 2]
    right_median = sorted(_line_box(line)[0] for line in right)[len(right) // 2]
    if right_median - left_median < page_width * 0.18:
        return sorted(zone_lines, key=lambda line: (_line_box(line)[1], _line_box(line)[0]))
    left.sort(key=lambda line: (_line_box(line)[1], _line_box(line)[0]))
    right.sort(key=lambda line: (_line_box(line)[1], _line_box(line)[0]))
    return left + right


def _sort_reading_order(lines, page_width):
    if not lines:
        return []
    lines = sorted(lines, key=lambda line: (_line_box(line)[1], _line_box(line)[0]))
    spanning = []
    body = []
    for line in lines:
        x0, y0, x1, y1 = _line_box(line)
        width = x1 - x0
        if width >= page_width * 0.62:
            spanning.append(line)
        else:
            body.append(line)
    if not spanning:
        return _sort_zone_by_columns(body, page_width)
    result = []
    current_top = float("-inf")
    for header in spanning:
        _, header_y0, _, header_y1 = _line_box(header)
        zone = [line for line in body if current_top <= _line_box(line)[1] < header_y0]
        result.extend(_sort_zone_by_columns(zone, page_width))
        result.append(header)
        current_top = header_y1
    tail = [line for line in body if _line_box(line)[1] >= current_top]
    result.extend(_sort_zone_by_columns(tail, page_width))
    deduped = []
    seen = set()
    for line in result:
        identity = id(line)
        if identity not in seen:
            seen.add(identity)
            deduped.append(line)
    return deduped


def _normalize_ocr_result(result, page_width):
    if not result or not result[0]:
        return [], [], []
    lines = _sort_reading_order(result[0], page_width)
    texts = [str(line[1][0]).strip() for line in lines]
    confidences = [float(line[1][1]) for line in lines]
    return lines, texts, confidences


def _candidate_score(texts, confidences):
    if not texts or not confidences:
        return -1.0
    usable = [text for text in texts if text.strip()]
    if not usable:
        return -1.0
    avg_conf = sum(confidences) / len(confidences)
    chars = sum(1 for text in usable for char in text if char.isalnum() or "\u4e00" <= char <= "\u9fff")
    single_fragments = sum(1 for text in usable if len(text.strip()) == 1)
    short_english = sum(1 for text in usable if text.isascii() and text.replace(" ", "").isalpha() and len(text) <= 4)
    return avg_conf * 100.0 + min(chars, 2000) * 0.004 + min(len(usable), 120) * 0.12 - single_fragments * 0.20 - short_english * 0.12


# ==============================================================
# OCR page functions
# ==============================================================
def _offset_ocr_lines(lines, offset_x: float, offset_y: float, scale_back: float = 1.0):
    adjusted = []
    for line in lines or []:
        box, payload = line[0], line[1]
        new_box = [
            [
                (float(point[0]) + offset_x) * scale_back,
                (float(point[1]) + offset_y) * scale_back,
            ]
            for point in box
        ]
        adjusted.append([new_box, payload])
    return adjusted


def _chart_line_is_useful(text: str) -> bool:
    """Keep chart captions/labels while dropping coordinate and status-bar noise."""
    stripped = str(text).strip()
    if not stripped:
        return False
    upper = stripped.upper()
    if "圖" in stripped and any(ch.isalpha() or "\u4e00" <= ch <= "\u9fff" for ch in stripped):
        return True
    if any(key in upper for key in ("MACD", "DIFF", "DEA", "KDJ", "RSI", "BOLL")):
        return len(stripped) <= 36
    if re.fullmatch(r"(?:13|34|55|89|233)?MA", upper):
        return True
    if any(term in stripped for term in ("日線", "週線", "月線", "K線", "成交量", "均量線", "黃金交叉", "死亡交叉")):
        return True
    return False


def _document_vocabulary_from_lines(texts, confidences):
    vocab = Counter()
    joined = "\n".join(texts or [])
    for term in FINANCE_TERMS:
        count = joined.count(term)
        if count:
            vocab[term] += count
    for text, conf in zip(texts or [], confidences or []):
        if conf >= 0.92 and 2 <= len(text.strip()) <= 12:
            vocab[text.strip()] += 1
    return vocab


def _ocr_candidate_for_crop(ocr, crop, mode, evaluator, document_vocabulary):
    candidate = encoded = raw_result = None
    try:
        candidate = _build_preprocess_candidate(crop, mode)
        scale_back = 1.0
        if mode == "UPSCALE":
            scale_back = 1.0 / 1.5
        ok, encoded = cv2.imencode(".png", candidate)
        if not ok:
            return None
        raw_result = ocr.ocr(encoded.tobytes(), cls=True)
        width = candidate.shape[1]
        lines, texts, confidences = _normalize_ocr_result(raw_result, width)
        if scale_back != 1.0:
            lines = _scale_ocr_lines(lines, scale_back)
        quality = evaluator.score(texts, confidences, document_vocabulary)
        return {
            "mode": mode,
            "score": quality.total,
            "quality": quality,
            "lines": lines,
            "texts": texts,
            "confidences": confidences,
        }
    finally:
        raw_result = None
        encoded = None
        candidate = None


def _ocr_page_layout_aware(
    image, ocr, evaluator, requested_zoom, effective_zoom,
    retry_threshold=65.0, filter_chart_noise=True,
):
    """Seed full-page OCR, segment geometry, then rerun only text regions."""
    encoded = None
    try:
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            return None
        seed_result = ocr.ocr(encoded.tobytes(), cls=True)
    finally:
        encoded = None

    page_width = image.shape[1]
    page_height = image.shape[0]
    seed_lines, seed_texts, seed_confidences = _normalize_ocr_result(seed_result, page_width)
    if not seed_lines:
        return None

    analyzer = PageLayoutAnalyzer()
    regions = analyzer.analyze(seed_lines, page_width, page_height)
    if not regions:
        return None
    document_vocabulary = _document_vocabulary_from_lines(seed_texts, seed_confidences)

    output_lines = []
    output_modes = []
    output_scores = []
    for region in regions:
        if region.region_type == "noise":
            continue
        if region.region_type == "chart" and filter_chart_noise:
            useful = [line for line in region.lines if _chart_line_is_useful(str(line[1][0]))]
            output_lines.extend(useful)
            if useful:
                output_modes.append("CHART_FILTER")
                output_scores.append(75.0)
            continue

        x0, y0, x1, y1 = analyzer.padded_bbox(region, page_width, page_height, pad=10)
        if x1 <= x0 or y1 <= y0:
            output_lines.extend(region.lines)
            continue
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            output_lines.extend(region.lines)
            continue

        modes = ["ORIGINAL", "LIGHT_CLAHE"]
        if region.region_type in {"title", "caption"}:
            modes.append("UPSCALE")
        candidates = []
        for mode in modes:
            try:
                item = _ocr_candidate_for_crop(ocr, crop, mode, evaluator, document_vocabulary)
                if item:
                    candidates.append(item)
            except Exception as exc:
                if not _is_memory_error(exc):
                    print(f"  [區塊 OCR] {mode} 失敗：{exc}")
            finally:
                _release_memory(use_gpu=False)

        if candidates:
            best = max(candidates, key=lambda item: item["score"])
        else:
            best = None

        if best is None or best["score"] < retry_threshold:
            for mode in ("CLAHE", "BINARY", "UPSCALE"):
                if any(item["mode"] == mode for item in candidates):
                    continue
                try:
                    item = _ocr_candidate_for_crop(ocr, crop, mode, evaluator, document_vocabulary)
                    if item:
                        candidates.append(item)
                        if best is None or item["score"] > best["score"]:
                            best = item
                except Exception as exc:
                    if not _is_memory_error(exc):
                        print(f"  [區塊重跑] {mode} 失敗：{exc}")
                finally:
                    _release_memory(use_gpu=False)

        if best is None or not best["lines"]:
            output_lines.extend(region.lines)
            output_modes.append("SEED_FALLBACK")
            output_scores.append(0.0)
            continue

        output_lines.extend(_offset_ocr_lines(best["lines"], x0, y0, 1.0))
        output_modes.append(best["mode"])
        output_scores.append(best["score"])

    if not output_lines:
        return None

    ordered = _sort_reading_order(output_lines, page_width)
    texts = [str(line[1][0]).strip() for line in ordered]
    confidences = [float(line[1][1]) for line in ordered]
    scale = requested_zoom / effective_zoom
    scaled_lines = _scale_ocr_lines(ordered, scale)
    avg_score = sum(output_scores) / len(output_scores) if output_scores else 0.0
    unique_modes = sorted(set(output_modes))
    if unique_modes and all(mode == "SEED_FALLBACK" for mode in unique_modes):
        mode_summary = "ORIGINAL"
    else:
        mode_summary = "LAYOUT:" + "+".join(unique_modes or ["ORIGINAL"])
    print(
        f"  [版面分區] 區塊 {len(regions)}，輸出 {len(ordered)} 行，"
        f"區塊品質 {avg_score:.1f}/100，模式 {mode_summary}"
    )
    return scaled_lines, texts, confidences, mode_summary


def ocr_page_paddle(
    page, ocr, zoom: int = 3, preprocess_mode: str = "auto",
    max_page_pixels: int = 24_000_000,
    memory_retry_enabled: bool = True,
    memory_retry_scales=None,
    release_between_candidates: bool = True,
    layout_aware: bool | None = None,
    region_retry_threshold: float = 65.0,
    filter_chart_noise: bool = True,
):
    """Run PaddleOCR with layout segmentation and low-quality region reruns.

    The layout path is enabled by default in auto mode.  It performs a seed
    full-page OCR, groups lines into title/body/chart regions, reruns only text
    regions using competing preprocessing modes, and filters chart-coordinate
    noise from TXT output.  Any failure falls back to the established full-page
    sequential candidate path.
    """
    requested_zoom = float(zoom)
    retry_scales = list(memory_retry_scales or [1.0, 0.8, 0.65, 0.5])
    if not memory_retry_enabled:
        retry_scales = [1.0]

    if layout_aware is None:
        layout_aware = os.environ.get("OCR_LAYOUT_AWARE", "1").strip().lower() not in {"0", "false", "no"}

    base_zoom = _effective_zoom_for_page(page, requested_zoom, int(max_page_pixels or 0))
    attempts = []
    for factor in retry_scales:
        candidate_zoom = max(1.5, min(requested_zoom, base_zoom * float(factor)))
        if not attempts or abs(candidate_zoom - attempts[-1]) > 1e-6:
            attempts.append(candidate_zoom)

    if base_zoom < requested_zoom:
        print(
            f"  [記憶體保護] Zoom {requested_zoom:g} 已調整為 {base_zoom:.2f} "
            f"（頁面像素上限 {int(max_page_pixels):,}）"
        )

    last_memory_error = None
    mode_map = {"original": "ORIGINAL", "clahe": "CLAHE", "binary": "BINARY"}
    modes = [mode_map.get(preprocess_mode.lower(), "ORIGINAL")] if preprocess_mode != "auto" else [
        "ORIGINAL", "CLAHE", "BINARY"
    ]
    evaluator = BlockTextEvaluator()

    for attempt_index, effective_zoom in enumerate(attempts, start=1):
        image = None
        try:
            image = _page_to_bgr(page, effective_zoom)
            page_width = image.shape[1]

            if layout_aware and preprocess_mode == "auto":
                try:
                    layout_result = _ocr_page_layout_aware(
                        image, ocr, evaluator, requested_zoom, effective_zoom,
                        retry_threshold=float(region_retry_threshold),
                        filter_chart_noise=bool(filter_chart_noise),
                    )
                    if layout_result is not None:
                        return layout_result
                except Exception as exc:
                    if _is_memory_error(exc):
                        last_memory_error = exc
                        print(f"  [版面分區] 記憶體不足，回退整頁模式：{exc}")
                    else:
                        print(f"  [版面分區] 分區失敗，回退整頁模式：{exc}")
                    _release_memory(use_gpu=True)

            best = None
            memory_failures = 0
            for mode in modes:
                candidate = encoded = raw_result = None
                try:
                    candidate = _build_preprocess_candidate(image, mode)
                    ok, encoded = cv2.imencode(".png", candidate)
                    if not ok:
                        continue
                    raw_result = ocr.ocr(encoded.tobytes(), cls=True)
                    lines, texts, confidences = _normalize_ocr_result(raw_result, page_width)
                    quality = evaluator.score(texts, confidences)
                    compact = {
                        "mode": mode,
                        "score": quality.total,
                        "lines": lines,
                        "texts": texts,
                        "confidences": confidences,
                    }
                    if best is None or compact["score"] > best["score"]:
                        best = compact
                except Exception as exc:
                    if _is_memory_error(exc):
                        memory_failures += 1
                        last_memory_error = exc
                        print(f"\n  [!] {mode} OCR 記憶體不足：{exc}")
                    else:
                        print(f"\n  [!] {mode} OCR 失敗：{exc}")
                finally:
                    raw_result = None
                    encoded = None
                    if candidate is not image:
                        candidate = None
                    if release_between_candidates:
                        _release_memory(use_gpu=False)

            if best is not None:
                scale = requested_zoom / effective_zoom
                lines = _scale_ocr_lines(best["lines"], scale)
                if effective_zoom < requested_zoom:
                    print(
                        f"  [記憶體保護] Zoom {effective_zoom:.2f} 成功，採用 {best['mode']} 結果"
                    )
                return lines, best["texts"], best["confidences"], best["mode"]

            if memory_failures and attempt_index < len(attempts):
                print(
                    f"  [記憶體保護] Zoom {effective_zoom:.2f} 無可用候選，"
                    f"將以 Zoom {attempts[attempt_index]:.2f} 重試"
                )
                _release_memory(use_gpu=True)
                continue
            break
        except Exception as exc:
            if _is_memory_error(exc):
                last_memory_error = exc
                if attempt_index < len(attempts):
                    print(
                        f"  [記憶體保護] Zoom {effective_zoom:.2f} 記憶體不足，"
                        f"將以 Zoom {attempts[attempt_index]:.2f} 重試：{exc}"
                    )
                    _release_memory(use_gpu=True)
                    continue
            raise
        finally:
            image = None
            _release_memory(use_gpu=False)

    if last_memory_error is not None:
        print(f"  [記憶體保護] 所有降級嘗試均失敗：{last_memory_error}")
        return [], [], [], "FAILED_MEMORY"
    return [], [], [], "FAILED"

def ocr_texts_nougat(page, zoom: int = 3) -> list:
    try:
        import torch
        from PIL import Image
        model = _load_nougat()
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        with torch.no_grad():
            output = model.inference(image=img, early_stopping=False)
        raw_md = output["predictions"][0] if output.get("predictions") else ""
        return [ln for ln in raw_md.split("\n") if ln.strip()]
    except Exception as e:
        print(f"\n  [!] Nougat 辨識失敗 ({e})，回退 PaddleOCR 文字")
        return []


def ocr_texts_surya(page, zoom: int = 3) -> list:
    try:
        from PIL import Image
        model, processor = _load_surya()
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        predictions = _surya_recog([img], [["zh", "en"]], model, processor)
        text_lines = predictions[0].text_lines if predictions else []
        return [tl.text for tl in text_lines if tl.text.strip()]
    except Exception as e:
        print(f"\n  [!] Surya 辨識失敗 ({e})，回退 PaddleOCR 文字")
        return []


# ==============================================================
# Prompt / engine helpers
# ==============================================================
def engine_prompt(engine: str, content_mode: str) -> tuple:
    if engine == "nougat":
        return PROMPT_NOUGAT, API_TIMEOUT_PHYSICS
    if engine == "surya":
        return PROMPT_SURYA, API_TIMEOUT_DEFAULT
    if content_mode == "FINANCE":
        return PROMPT_PADDLE_FINANCE, API_TIMEOUT_DEFAULT
    if content_mode == "PHYSICS":
        return PROMPT_PADDLE_PHYSICS, API_TIMEOUT_PHYSICS
    return PROMPT_PADDLE_DEFAULT, API_TIMEOUT_DEFAULT


def _document_key(file_name: str) -> str:
    stem = os.path.splitext(os.path.basename(file_name or ""))[0]
    while stem.lower().endswith("_ocr"):
        stem = stem[:-4]
    return stem


def _formula_dense(text: str) -> bool:
    symbols = sum(ch in "=+-*/^_()[]{}λβωηΓπ∞∇×·" for ch in text)
    return symbols >= 3 or "\\" in text or "$" in text


def apply_safe_corrections(texts: list, file_name: str = "") -> Tuple[list, list]:
    """Apply offline domain/document-aware corrections.

    This function never performs network requests and preserves one output line
    for every input line so PDF overlay and checkpoint alignment remain stable.
    """
    result = correct_lines([str(t) for t in texts], file_name=file_name)
    return result.corrected_texts, result.sources


def apply_hard_corrections(texts: list, file_name: str = "") -> list:
    """Backward-compatible wrapper for older callers."""
    return apply_safe_corrections(texts, file_name)[0]


def align_lines(original: list, corrected: list) -> list:
    if len(corrected) == len(original):
        return corrected
    if not original:
        return corrected
    if abs(len(corrected) - len(original)) / len(original) > 0.20:
        return original
    matcher = difflib.SequenceMatcher(None, original, corrected, autojunk=False)
    aligned = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            aligned.extend(corrected[j1:j2])
        elif tag == "replace":
            orig_chunk = original[i1:i2]
            corr_chunk = corrected[j1:j2]
            has_formula = any('$' in l or '\\' in l or '=' in l for l in orig_chunk)
            if has_formula and len(corr_chunk) < len(orig_chunk):
                aligned.extend(orig_chunk)
            else:
                aligned.extend(corr_chunk)
                for _ in range(len(orig_chunk) - len(corr_chunk)):
                    aligned.append("")
        elif tag == "delete":
            for _ in range(i2 - i1):
                aligned.append("")
    return aligned if len(aligned) == len(original) else original


def call_claude_batch(batch_texts: list, prompt: str, timeout: float, claude_model: str = ""):
    """Deprecated compatibility shim; external Claude API is disabled."""
    return [None] * len(batch_texts), False


# ==============================================================
# Output helpers
# ==============================================================
def apply_text_overlay(page, lines: list, corrected_lines: list, zoom: int):
    for i, line in enumerate(lines):
        if i >= len(corrected_lines):
            break
        text = corrected_lines[i].strip()
        if not text:
            continue
        rect_pts = line[0]
        x0 = rect_pts[0][0] / zoom
        y0 = rect_pts[0][1] / zoom
        x1 = rect_pts[2][0] / zoom
        y1 = rect_pts[2][1] / zoom
        h = y1 - y0
        fitz_rect = fitz.Rect(x0, y0, x1, y1)
        try:
            page.insert_textbox(
                fitz_rect,
                text,
                fontsize=max(5, h * 0.72),
                fontname="china-s",
                render_mode=3,
                overlay=True,
            )
        except Exception:
            # Fallback to insert_text if textbox fails (e.g., rect too small)
            try:
                page.insert_text(
                    fitz.Point(x0, y0 + h * 0.88),
                    text,
                    fontsize=max(5, h * 0.72),
                    fontname="china-t",
                    render_mode=3,
                )
            except Exception:
                pass


def append_ocr_text(text_path: str, page_num: int, texts: list):
    with open(text_path, "a", encoding="utf-8-sig") as handle:
        handle.write(f"\n=== PAGE {page_num + 1} ===\n")
        for text in texts:
            cleaned = str(text).strip()
            if cleaned:
                handle.write(cleaned + "\n")


def _suspicious_reasons(text: str, confidence: float) -> list:
    reasons = []
    stripped = text.strip()
    if confidence < 0.70:
        reasons.append(f"低置信度 {confidence:.3f}")
    elif confidence < 0.82:
        reasons.append(f"中低置信度 {confidence:.3f}")
    suspicious_chars = set("□■◆◇※〒卅卌")
    if any(char in stripped for char in suspicious_chars):
        reasons.append("含疑似亂碼符號")
    if len(stripped) == 1:
        reasons.append("單字碎片")
    ascii_letters = sum(char.isascii() and char.isalpha() for char in stripped)
    spaces = stripped.count(" ")
    if ascii_letters >= 3 and len(stripped) <= 6:
        reasons.append("疑似英文斷詞")
    elif ascii_letters >= 6 and spaces == 0 and len(stripped) >= 18:
        reasons.append("英文黏字")
    formula_symbols = sum(char in "=+-*/^_()[]{}λβωηΓπ∞∇×·" for char in stripped)
    if formula_symbols >= 3:
        reasons.append("公式／符號密集，需人工核對")
    if any(token in stripped for token in ("二0", "之二", "工", "土", "傈數", "駐波此")):
        reasons.append("疑似形近字或公式誤辨")
    return reasons


def append_quality_analysis(analysis_path, file_name, page_num, texts, confidences, preprocess_mode, corrected_texts=None, correction_sources=None):
    confidences = confidences or []
    corrected_texts = corrected_texts or texts
    correction_sources = correction_sources or [""] * len(texts)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    low_count = sum(1 for value in confidences if value < 0.70)
    medium_count = sum(1 for value in confidences if 0.70 <= value < 0.82)
    issues = []
    for index, text in enumerate(texts):
        confidence = confidences[index] if index < len(confidences) else 0.0
        reasons = _suspicious_reasons(text, confidence)
        corrected = corrected_texts[index] if index < len(corrected_texts) else text
        source = correction_sources[index] if index < len(correction_sources) else ""
        if corrected.strip() != text.strip():
            reasons.append(f"已校正（{source or 'AI/規則'}）")
        if reasons:
            issues.append((index + 1, confidence, text.strip(), corrected.strip(), "；".join(reasons)))
    with open(analysis_path, "a", encoding="utf-8-sig") as handle:
        handle.write(
            f"\n{'=' * 72}\n"
            f"{file_name} - 第 {page_num + 1} 頁\n"
            f"預處理版本：{preprocess_mode}\n"
            f"辨識行數：{len(texts)}\n"
            f"平均置信度：{avg_conf:.3f}\n"
            f"低置信度行數(<0.70)：{low_count}\n"
            f"中低置信度行數(0.70-0.82)：{medium_count}\n"
        )
        if avg_conf < 0.75:
            handle.write("頁面判定：高風險，建議人工核對原圖。\n")
        elif issues:
            handle.write("頁面判定：部分行需校正。\n")
        else:
            handle.write("頁面判定：未發現明顯高風險行。\n")
        if issues:
            handle.write("\n需修改／校正部分：\n")
            for line_no, confidence, original, corrected, reasons in issues:
                handle.write(f"  行 {line_no:03d} | conf={confidence:.3f}\n    原始：{original}\n")
                if corrected != original:
                    handle.write(f"    校正：{corrected}\n")
                handle.write(f"    原因：{reasons}\n")
        else:
            handle.write("\n需修改／校正部分：無明顯項目。\n")


def log_corrections(log_path, file_name, page_num, original, corrected, engine, ai_enabled, correction_sources=None):
    """Write only actual changes; raw OCR belongs in *_OCR_raw.txt."""
    correction_sources = correction_sources or [""] * len(original)
    entries = []
    for idx, (orig, corr) in enumerate(zip(original, corrected), start=1):
        o, c = str(orig).strip(), str(corr).strip()
        if o == c or not o:
            continue
        source = correction_sources[idx - 1] if idx - 1 < len(correction_sources) else ""
        if not source:
            source = "AI" if ai_enabled else "RULE"
        entries.append(
            f"  行 {idx:03d} [{source}]\n"
            f"    原始：{o}\n"
            f"    修正：{c}"
        )
    if not entries:
        return
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] {file_name}"
            f" 第 {page_num + 1} 頁  [{engine}]\n"
        )
        f.write("\n".join(entries) + "\n")


# ==============================================================
# GPU utilities
# ==============================================================
def _find_dll(filename: str):
    for directory in GPU_DLL_PATHS:
        candidate = os.path.join(directory, filename)
        if os.path.isfile(candidate):
            return candidate
    return None


def gpu_runtime_self_test(timeout: int = 60) -> tuple:
    test_code = (
        "import paddle\n"
        "paddle.set_device('gpu:0')\n"
        "x=paddle.randn([1,3,64,64])\n"
        "w=paddle.randn([8,3,3,3])\n"
        "y=paddle.nn.functional.conv2d(x,w)\n"
        "print('GPU_RUNTIME_OK', list(y.shape))\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True, text=True, timeout=timeout,
            env=os.environ.copy(),
            creationflags=(subprocess.CREATE_NO_WINDOW if sys.platform == "win32" and hasattr(subprocess, "CREATE_NO_WINDOW") else 0),
        )
    except subprocess.TimeoutExpired:
        return False, "GPU 實際運算測試逾時"
    except Exception as exc:
        return False, f"無法啟動 GPU 自我測試：{exc}"
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    if result.returncode == 0 and "GPU_RUNTIME_OK" in output:
        return True, output
    return False, output or f"子程序結束碼：{result.returncode}"


def print_environment_summary():
    print("\n[環境檢查]")
    print(f"  Python      : {sys.version.split()[0]}")
    print(f"  執行檔      : {sys.executable}")
    try:
        for package in ("paddlepaddle-gpu", "paddleocr", "numpy", "pillow", "opencv-python",
                        "albumentations", "albucore", "nvidia-cublas-cu11", "nvidia-cudnn-cu11",
                        "nvidia-cuda-runtime-cu11"):
            try:
                print(f"  {package:<22}: {_meta.version(package)}")
            except _meta.PackageNotFoundError:
                pass
    except Exception:
        pass
    print("  cuDNN DLL   : " + (_find_dll("cudnn_ops_infer64_8.dll") or "未找到"))
    print("  cuBLAS DLL  : " + (_find_dll("cublasLt64_11.dll") or "未找到"))


# ==============================================================
# CancelledError for OCRProcessor
# ==============================================================
class OCRCancelledError(Exception):
    """Raised when cancel_event is set during processing."""
    pass


# ==============================================================
# OCRProcessor
# ==============================================================
class OCRProcessor:
    """
    Orchestrates OCR processing for a batch of PDF files.

    Callbacks (all optional, default to stdout print):
        progress_callback(OCRProgress)
        log_callback(message: str)
        file_status_callback(file_index, file_path, status, extra_info)
    Threading:
        pause_event  - threading.Event; when set, processing blocks each page
        cancel_event - threading.Event; when set, raises OCRCancelledError
    """

    def __init__(
        self,
        config: OCRConfig,
        progress_callback: Optional[Callable] = None,
        log_callback: Optional[Callable] = None,
        file_status_callback: Optional[Callable] = None,
        pause_event: Optional[threading.Event] = None,
        cancel_event: Optional[threading.Event] = None,
    ):
        self.config = config
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.file_status_callback = file_status_callback
        self.pause_event = pause_event
        self.cancel_event = cancel_event
        self._ocr = None
        self._claude_model = config.claude_model if config.claude_model else CLAUDE_MODEL_DEFAULT

    # ----------------------------------------------------------
    def _log(self, msg: str):
        if self.log_callback:
            self.log_callback(msg)
        else:
            print(msg)

    def _report_progress(self, **kwargs):
        if self.progress_callback:
            prog = OCRProgress(**kwargs)
            self.progress_callback(prog)

    def _report_file_status(self, file_index: int, file_path: str, status: str, extra_info: str = ""):
        if self.file_status_callback:
            self.file_status_callback(file_index, file_path, status, extra_info)
        else:
            self._log(f"[{status}] {os.path.basename(file_path)} {extra_info}")

    def _check_cancel_pause(self):
        if self.cancel_event and self.cancel_event.is_set():
            raise OCRCancelledError("Processing cancelled by user.")
        if self.pause_event:
            while self.pause_event.is_set():
                time.sleep(0.2)
                if self.cancel_event and self.cancel_event.is_set():
                    raise OCRCancelledError("Processing cancelled while paused.")

    # ----------------------------------------------------------
    def collect_pdf_files(self) -> list:
        """Collect source PDFs while excluding generated OCR outputs."""
        input_path = os.path.abspath(os.path.expanduser(self.config.input_path))
        output_root = os.path.abspath(os.path.expanduser(self.config.output_path or ""))

        def eligible(path: str) -> bool:
            lower = os.path.basename(path).lower()
            if not lower.endswith(".pdf"):
                return False
            if self.config.exclude_ocr_output and (
                lower.endswith("_ocr.pdf")
                or lower.endswith("_ocr_ocr.pdf")
                or lower.endswith("_searchable.pdf")
                or lower.endswith(".part.pdf")
            ):
                return False
            full = os.path.abspath(path)
            if output_root and os.path.isdir(output_root):
                try:
                    if os.path.commonpath([full, output_root]) == output_root:
                        return False
                except ValueError:
                    pass
            return True

        if os.path.isfile(input_path):
            return [input_path] if eligible(input_path) else []
        if os.path.isdir(input_path):
            files = []
            if self.config.recursive:
                for root, dirs, fnames in os.walk(input_path):
                    dirs[:] = [d for d in dirs if d.lower() not in {".ocr_state", "output", "outputs"}]
                    for fname in sorted(fnames):
                        candidate = os.path.join(root, fname)
                        if eligible(candidate):
                            files.append(candidate)
            else:
                for fname in sorted(os.listdir(input_path)):
                    candidate = os.path.join(input_path, fname)
                    if eligible(candidate):
                        files.append(candidate)
            return files
        return []

    # ----------------------------------------------------------
    def initialize_engine(self):
        """Initialize PaddleOCR (call once before process_all)."""
        use_gpu = self.config.device.lower() == "gpu"
        if use_gpu:
            missing_dlls = []
            if not _find_dll("cudnn_ops_infer64_8.dll"):
                missing_dlls.append("cudnn_ops_infer64_8.dll")
            if not _find_dll("cublasLt64_11.dll"):
                missing_dlls.append("cublasLt64_11.dll")
            if missing_dlls:
                self._log("[GPU] 缺少必要 DLL，改用 CPU。")
                for dll_name in missing_dlls:
                    self._log(f"      缺少：{dll_name}")
                use_gpu = False
            else:
                self._log("[GPU] 正在以獨立子程序測試 CUDA/cuDNN/cuBLAS...")
                gpu_ok, gpu_detail = gpu_runtime_self_test()
                if gpu_ok:
                    self._log("[GPU] 實際卷積測試通過，將使用 gpu:0。")
                else:
                    self._log("[GPU] 實際運算測試失敗，已自動回退 CPU。")
                    if gpu_detail:
                        self._log(gpu_detail[-2000:])
                    use_gpu = False

        self._ocr = PaddleOCR(
            lang='chinese_cht',
            use_gpu=use_gpu,
            use_angle_cls=True,
            det_db_thresh=0.2,
            det_db_unclip_ratio=1.8,
            show_log=False,
            enable_mkldnn=not use_gpu,
        )
        self._use_gpu = use_gpu

    # ----------------------------------------------------------
    def _resolve_output_path(self, pdf_path: str) -> str:
        """Determine output directory for a given pdf_path."""
        cfg = self.config
        output_root = os.path.abspath(os.path.expanduser(cfg.output_path))
        if cfg.preserve_relative_structure and os.path.isdir(
            os.path.abspath(os.path.expanduser(cfg.input_path))
        ):
            input_root = os.path.abspath(os.path.expanduser(cfg.input_path))
            rel = os.path.relpath(os.path.dirname(pdf_path), input_root)
            if rel == ".":
                return output_root
            return os.path.join(output_root, rel)
        return output_root

    # ----------------------------------------------------------
    def process_all(self):
        """Process all PDFs in the batch. Continues on per-file failure if batch_failure_continue."""
        if self._ocr is None:
            self.initialize_engine()

        cfg = self.config
        files = self.collect_pdf_files()
        if not files:
            self._log(f"[錯誤] 找不到 PDF：{cfg.input_path}")
            return

        os.makedirs(os.path.abspath(os.path.expanduser(cfg.output_path)), exist_ok=True)

        if cfg.enable_claude:
            self._log("[離線校正] 已忽略 enable_claude；本版本不使用外部 API。")
            cfg.enable_claude = False
        if cfg.enable_claude:
            api_key = CLAUDE_API_KEY
            if not api_key or api_key == "YOUR_CLAUDE_API_KEY":
                self._log("[API] 尚未設定 CLAUDE_API_KEY，已取消 AI 校對。")
                cfg.enable_claude = False
            else:
                if not validate_key(self._claude_model):
                    cfg.enable_claude = False

        total_files = len(files)
        for file_index, pdf_path in enumerate(files):
            self._report_file_status(file_index, pdf_path, "等待")
            self._check_cancel_pause()
            try:
                self._report_file_status(file_index, pdf_path, "初始化")
                self.process_pdf(pdf_path, file_index=file_index, file_total=total_files)
            except OCRCancelledError:
                self._report_file_status(file_index, pdf_path, "已取消")
                raise
            except Exception as exc:
                self._log(f"[錯誤] {os.path.basename(pdf_path)}：{exc}")
                self._report_file_status(file_index, pdf_path, "失敗", str(exc))
                if not cfg.batch_failure_continue:
                    raise

    # ----------------------------------------------------------
    def process_pdf(self, pdf_path: str, file_index: int = 0, file_total: int = 1):
        """Process a single PDF file."""
        cfg = self.config
        if cfg.enable_claude:
            self._log("[離線校正] 已停用 Claude API，改用本機詞庫與上下文校正。")
            cfg.enable_claude = False
        file_name = os.path.basename(pdf_path)
        stem = os.path.splitext(file_name)[0]
        out_dir = self._resolve_output_path(pdf_path)
        os.makedirs(out_dir, exist_ok=True)

        target_pdf = os.path.join(out_dir, stem + "_OCR.pdf")
        target_txt = os.path.join(out_dir, stem + "_OCR.txt")
        raw_txt = os.path.join(out_dir, stem + "_OCR_raw.txt")
        corrected_txt = os.path.join(out_dir, stem + "_OCR_corrected.txt")
        analysis_txt = os.path.join(out_dir, stem + "_OCR_analysis.txt")
        log_path = os.path.join(
            os.path.abspath(os.path.expanduser(cfg.output_path)), "verify_log.txt"
        )

        # Skip-existing logic
        if cfg.skip_existing and not cfg.overwrite:
            if cfg.output_pdf and os.path.exists(target_pdf):
                self._report_file_status(file_index, pdf_path, "跳過", f"輸出已存在：{target_pdf}")
                return

        # Overwrite: remove existing outputs
        if cfg.overwrite:
            for path in (target_txt, raw_txt, corrected_txt, analysis_txt):
                if os.path.exists(path):
                    os.remove(path)
        else:
            for path in (target_txt, raw_txt, corrected_txt, analysis_txt):
                if os.path.exists(path):
                    os.remove(path)

        self._report_file_status(file_index, pdf_path, "OCR 中")

        doc = fitz.open(pdf_path)
        pages = list(doc)
        total_pages = len(pages)
        page_times = []

        if cfg.enable_claude:
            for batch_start in range(0, total_pages, BATCH_SIZE):
                self._check_cancel_pause()
                batch_pages = pages[batch_start: batch_start + BATCH_SIZE]
                ocr_data = []
                for p in batch_pages:
                    self._check_cancel_pause()
                    ln, tx, cf, mode = ocr_page_paddle(
                        p, self._ocr, zoom=cfg.zoom, preprocess_mode=cfg.preprocess_mode,
                        max_page_pixels=cfg.max_page_pixels,
                        memory_retry_enabled=cfg.memory_retry_enabled,
                        memory_retry_scales=cfg.memory_retry_scales,
                        release_between_candidates=cfg.release_between_candidates,
                    )
                    ocr_data.append([ln, tx, cf, cfg.zoom, mode])

                sample = " ".join(" ".join(d[1]) for d in ocr_data)
                content_mode = (
                    "FINANCE" if any(kw in sample for kw in FINANCE_KEYWORDS)
                    else "PHYSICS" if any(kw in sample for kw in PHYSICS_KEYWORDS)
                    else "DEFAULT"
                )
                engine = select_engine([t for d in ocr_data for t in d[1]], [c for d in ocr_data for c in d[2]])
                engine_texts = []
                if engine == "nougat":
                    for p, d in zip(batch_pages, ocr_data):
                        nt = ocr_texts_nougat(p) or d[1]
                        engine_texts.append(nt)
                elif engine == "surya":
                    for p, d in zip(batch_pages, ocr_data):
                        st = ocr_texts_surya(p) or d[1]
                        engine_texts.append(st)
                else:
                    engine_texts = [d[1] for d in ocr_data]

                all_confs = [c for d in ocr_data for c in d[2]]
                avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0
                has_terms = content_mode != "DEFAULT"
                skip_api = avg_conf >= CONF_SKIP_THRESHOLD and not has_terms

                if skip_api:
                    corrected_pages = [None] * len(batch_pages)
                    uncertain = False
                    disp_engine = f"{engine.upper()}+SKIP"
                    elapsed = 0.0
                else:
                    self._report_file_status(file_index, pdf_path, "Claude 校對中")
                    prompt, timeout = engine_prompt(engine, content_mode)
                    preprocessed = [apply_safe_corrections(tx, file_name)[0] for tx in engine_texts]
                    t0 = time.perf_counter()
                    corrected_pages, uncertain = call_claude_batch(
                        preprocessed, prompt, timeout, claude_model=self._claude_model
                    )
                    elapsed = time.perf_counter() - t0
                    disp_engine = engine.upper()

                for j, (page_obj, (lines, raw_texts, confs, zoom, preprocess_mode)) in enumerate(
                    zip(batch_pages, ocr_data)
                ):
                    self._check_cancel_pause()
                    page_idx = batch_start + j
                    corrected = corrected_pages[j]
                    if uncertain:
                        lines, raw_texts, confs, preprocess_mode = ocr_page_paddle(
                            page_obj, self._ocr, zoom=cfg.zoom + 1,
                            preprocess_mode=cfg.preprocess_mode,
                            max_page_pixels=cfg.max_page_pixels,
                            memory_retry_enabled=cfg.memory_retry_enabled,
                            memory_retry_scales=cfg.memory_retry_scales,
                            release_between_candidates=cfg.release_between_candidates,
                        )
                        ocr_data[j] = [lines, raw_texts, confs, cfg.zoom + 1, preprocess_mode]
                        zoom = cfg.zoom + 1
                    if corrected is None:
                        corrected = engine_texts[j]
                    else:
                        # Explicit line-count validation before align
                        if len(corrected) != len(engine_texts[j]):
                            self._log(
                                f"[警告] Claude 回傳行數不符（原始 {len(engine_texts[j])} 行，"
                                f"回傳 {len(corrected)} 行），嘗試對齊..."
                            )
                        corrected = align_lines(engine_texts[j], corrected)
                        # If still wrong after align, fall back to hard corrections only
                        if len(corrected) != len(engine_texts[j]):
                            self._log(
                                f"[警告] 對齊後行數仍不符，回退至規則校正結果。"
                            )
                            corrected = engine_texts[j]
                    rule_corrected, rule_sources = apply_safe_corrections(corrected, file_name)
                    final_texts = rule_corrected
                    sources = [src or ("AI" if a.strip() != b.strip() else "")
                               for src, a, b in zip(rule_sources, raw_texts, final_texts)]

                    if cfg.output_pdf:
                        apply_text_overlay(page_obj, lines, final_texts, zoom)
                    if cfg.output_txt:
                        append_ocr_text(target_txt, page_idx, final_texts)
                        append_ocr_text(corrected_txt, page_idx, final_texts)
                    if cfg.output_raw_txt:
                        append_ocr_text(raw_txt, page_idx, raw_texts)
                    if cfg.output_analysis:
                        append_quality_analysis(analysis_txt, file_name, page_idx, raw_texts, confs, preprocess_mode, final_texts, sources)
                    if cfg.output_verify_log:
                        log_corrections(log_path, file_name, page_idx, raw_texts, final_texts, disp_engine, ai_enabled=True, correction_sources=sources)

                    page_time = elapsed / max(len(batch_pages), 1)
                    page_times.append(page_time)
                    avg = sum(page_times) / len(page_times)
                    remaining = avg * (total_pages - len(page_times))
                    self._report_progress(
                        file_index=file_index,
                        file_total=file_total,
                        file_path=pdf_path,
                        page_index=page_idx,
                        page_total=total_pages,
                        stage=disp_engine,
                        preprocess_mode=preprocess_mode,
                        elapsed_seconds=page_time,
                        average_seconds_per_page=avg,
                        estimated_remaining_seconds=remaining,
                    )
        else:
            for page_num, page_obj in enumerate(pages):
                self._check_cancel_pause()
                t0 = time.perf_counter()
                lines, raw_texts, confidences, preprocess_mode = ocr_page_paddle(
                    page_obj, self._ocr, zoom=cfg.zoom, preprocess_mode=cfg.preprocess_mode,
                    max_page_pixels=cfg.max_page_pixels,
                    memory_retry_enabled=cfg.memory_retry_enabled,
                    memory_retry_scales=cfg.memory_retry_scales,
                    release_between_candidates=cfg.release_between_candidates,
                )
                fixed, sources = apply_safe_corrections(raw_texts, file_name)

                if cfg.output_pdf:
                    apply_text_overlay(page_obj, lines, fixed, cfg.zoom)
                if cfg.output_txt:
                    append_ocr_text(target_txt, page_num, fixed)
                    append_ocr_text(corrected_txt, page_num, fixed)
                if cfg.output_raw_txt:
                    append_ocr_text(raw_txt, page_num, raw_texts)
                if cfg.output_analysis:
                    append_quality_analysis(analysis_txt, file_name, page_num, raw_texts, confidences, preprocess_mode, fixed, sources)
                if cfg.output_verify_log:
                    log_corrections(log_path, file_name, page_num, raw_texts, fixed, "PADDLE", ai_enabled=False, correction_sources=sources)

                elapsed = time.perf_counter() - t0
                page_times.append(elapsed)
                avg = sum(page_times) / len(page_times)
                remaining = avg * (total_pages - len(page_times))
                self._report_progress(
                    file_index=file_index,
                    file_total=file_total,
                    file_path=pdf_path,
                    page_index=page_num,
                    page_total=total_pages,
                    stage="PADDLE+RAW",
                    preprocess_mode=preprocess_mode,
                    elapsed_seconds=elapsed,
                    average_seconds_per_page=avg,
                    estimated_remaining_seconds=remaining,
                )

        # Save PDF
        if cfg.output_pdf:
            self._report_file_status(file_index, pdf_path, "寫入 PDF")
            try:
                doc.save(target_pdf, garbage=4, deflate=True)
            except PermissionError:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fallback = target_pdf.replace(".pdf", f"_{ts}.pdf")
                self._log(f"\n  [!] 檔案被佔用，改存為: {fallback}")
                doc.save(fallback, garbage=4, deflate=True)

        doc.close()
        avg_total = sum(page_times) / len(page_times) if page_times else 0
        self._log(f"  [V] PDF 完成: {target_pdf}")
        if cfg.output_txt:
            self._log(f"  [V] TXT 完成: {target_txt}")
            self._log(f"  [V] 校正版 TXT: {corrected_txt}")
        if cfg.output_raw_txt:
            self._log(f"  [V] 原始 TXT: {raw_txt}")
        if cfg.output_analysis:
            self._log(f"  [V] 分析完成: {analysis_txt}")
        self._log(f"      平均速度: {avg_total:.2f}s/頁")
        self._report_file_status(file_index, pdf_path, "完成", f"平均速度: {avg_total:.2f}s/頁")
