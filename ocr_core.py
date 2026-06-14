# ocr_core.py — Shared OCR core for OCR Engine v4.4.0
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
from typing import Callable, Optional, List

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
import anthropic
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
    enable_claude: bool = False
    recursive: bool = False
    overwrite: bool = False
    skip_existing: bool = True
    zoom: int = 3
    preprocess_mode: str = "auto"           # "auto" | "original" | "clahe" | "binary"
    output_pdf: bool = True
    output_txt: bool = True
    output_analysis: bool = True
    output_verify_log: bool = True
    confidence_threshold: float = 0.70
    preserve_relative_structure: bool = True
    claude_model: str = ""                  # empty → use CLAUDE_MODEL env var
    batch_failure_continue: bool = True


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
    """Lazily create a Claude API client, optionally with a custom model string (unused in client but kept for consistency)."""
    global _client_cache
    key = CLAUDE_API_KEY
    if key not in _client_cache:
        _client_cache[key] = anthropic.Anthropic(api_key=key)
    return _client_cache[key]


def validate_key(model: str = "") -> bool:
    """Validate the Claude API key with a minimal call."""
    use_model = model if model else CLAUDE_MODEL_DEFAULT
    try:
        get_anthropic_client().messages.create(
            model=use_model,
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
        print("[API] 金鑰驗證成功")
        return True
    except anthropic.AuthenticationError:
        print("[API] ❌ 金鑰無效，請檢查 CLAUDE_API_KEY 設定（AuthenticationError）")
        return False
    except Exception as e:
        print(f"[API] ⚠ 驗證時發生非認證錯誤 ({type(e).__name__})，繼續執行")
        return True


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

HARD_CODED_CORRECTIONS = {
    "白大": "台大",
    "輻樹聲": "賴樹聲",
    "貨格考": "資格考",
    "F-ma": "F=ma",
    "貨格": "價格",
    "訊號土": "訊號±",
    "聶": "",
    "畢吳": "",
    "理輪": "理論",
    "梨力學": "熱力學",
    "電形學": "電磁學",
    "衣捲": "交卷",
    "D.Eardley日基研nt": "D. Eardley 研一下",
}

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


def _preprocess_variants(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)
    blur = cv2.GaussianBlur(contrast, (0, 0), 1.0)
    sharpened = cv2.addWeighted(contrast, 1.45, blur, -0.45, 0)
    binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 13)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return {"ORIGINAL": image_bgr, "CLAHE": sharpened, "BINARY": binary}


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
def ocr_page_paddle(page, ocr, zoom: int = 3, preprocess_mode: str = "auto"):
    """Run PaddleOCR on a page. preprocess_mode: auto|original|clahe|binary."""
    image = _page_to_bgr(page, zoom)
    page_width = image.shape[1]
    variants = _preprocess_variants(image)

    if preprocess_mode != "auto":
        key_map = {"original": "ORIGINAL", "clahe": "CLAHE", "binary": "BINARY"}
        key = key_map.get(preprocess_mode.lower(), "ORIGINAL")
        selected = {key: variants.get(key, image)}
    else:
        selected = variants

    best = None
    for mode, candidate in selected.items():
        ok, encoded = cv2.imencode(".png", candidate)
        if not ok:
            continue
        try:
            result = ocr.ocr(encoded.tobytes(), cls=True)
        except Exception as exc:
            print(f"\n  [!] {mode} OCR 失敗：{exc}")
            continue
        lines, texts, confidences = _normalize_ocr_result(result, page_width)
        score = _candidate_score(texts, confidences)
        if best is None or score > best["score"]:
            best = {"mode": mode, "score": score, "lines": lines, "texts": texts, "confidences": confidences}
    if best is None:
        return [], [], [], "FAILED"
    return best["lines"], best["texts"], best["confidences"], best["mode"]


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


def apply_hard_corrections(texts: list) -> list:
    result = []
    for t in texts:
        for wrong, right in HARD_CODED_CORRECTIONS.items():
            t = t.replace(wrong, right)
        result.append(t)
    return result


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
    use_model = claude_model if claude_model else CLAUDE_MODEL_DEFAULT
    parts = [PAGE_SEP.format(n=i + 1) + "\n" + "\n".join(texts) for i, texts in enumerate(batch_texts)]
    combined = "\n\n".join(parts)
    try:
        response = get_anthropic_client().messages.create(
            model=use_model,
            max_tokens=8192,
            system=prompt,
            messages=[{"role": "user", "content": (
                f"以下是 {len(batch_texts)} 頁的 OCR 文字，每頁以 ===PAGE_N=== 分隔。"
                "請依序校對每頁，回傳時必須維持 ===PAGE_N=== 分隔格式，不得省略。\n\n" + combined
            )}],
            timeout=timeout,
        )
        raw = response.content[0].text
        uncertain = raw.lstrip().startswith("[UNCERTAIN]")
        if uncertain:
            raw = raw.lstrip()[len("[UNCERTAIN]"):].lstrip("\n")
        result_pages = []
        for i in range(len(batch_texts)):
            sep = PAGE_SEP.format(n=i + 1)
            next_sep = PAGE_SEP.format(n=i + 2)
            start = raw.find(sep)
            if start == -1:
                result_pages.append(None)
                continue
            start += len(sep)
            end = raw.find(next_sep, start) if (i + 1 < len(batch_texts)) else len(raw)
            result_pages.append(raw[start:end].strip().split("\n"))
        return result_pages, uncertain
    except anthropic.AuthenticationError:
        print("\n  [!] 請檢查 API KEY 設定（AuthenticationError），本批次標記為 [Raw OCR]")
    except anthropic.APITimeoutError:
        print("\n  [!] Claude API 逾時，保留原始 OCR 文字")
    except anthropic.APIConnectionError:
        print("\n  [!] Claude API 連線失敗，保留原始 OCR 文字")
    except Exception as e:
        print(f"\n  [!] Claude API 錯誤 ({type(e).__name__})，保留原始 OCR 文字")
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


def append_quality_analysis(analysis_path, file_name, page_num, texts, confidences, preprocess_mode, corrected_texts=None):
    confidences = confidences or []
    corrected_texts = corrected_texts or texts
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    low_count = sum(1 for value in confidences if value < 0.70)
    medium_count = sum(1 for value in confidences if 0.70 <= value < 0.82)
    issues = []
    for index, text in enumerate(texts):
        confidence = confidences[index] if index < len(confidences) else 0.0
        reasons = _suspicious_reasons(text, confidence)
        corrected = corrected_texts[index] if index < len(corrected_texts) else text
        if corrected.strip() != text.strip():
            reasons.append("已被規則或 AI 校正")
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


def log_corrections(log_path, file_name, page_num, original, corrected, engine, ai_enabled):
    entries = []
    if ai_enabled:
        for orig, corr in zip(original, corrected):
            o, c = orig.strip(), corr.strip()
            if o != c:
                entries.append(f"  原始: {o}\n  修正: {c}")
    else:
        for orig in original:
            o = orig.strip()
            if o:
                entries.append(f"  OCR : {o}  →  [Raw OCR]")
    if entries:
        ai_tag = "AI校對" if ai_enabled else "Raw OCR"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now().strftime('%H:%M:%S')}] {file_name}"
                    f" 第 {page_num + 1} 頁  [{engine}+{ai_tag}]\n")
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
        """Collect PDF file paths from config.input_path."""
        input_path = os.path.abspath(os.path.expanduser(self.config.input_path))
        if os.path.isfile(input_path) and input_path.lower().endswith(".pdf"):
            return [input_path]
        if os.path.isdir(input_path):
            if self.config.recursive:
                files = []
                for root, dirs, fnames in os.walk(input_path):
                    for fname in sorted(fnames):
                        if fname.lower().endswith(".pdf"):
                            files.append(os.path.join(root, fname))
                return files
            else:
                return sorted(
                    os.path.join(input_path, f)
                    for f in os.listdir(input_path)
                    if f.lower().endswith(".pdf")
                )
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
        file_name = os.path.basename(pdf_path)
        stem = os.path.splitext(file_name)[0]
        out_dir = self._resolve_output_path(pdf_path)
        os.makedirs(out_dir, exist_ok=True)

        target_pdf = os.path.join(out_dir, stem + "_OCR.pdf")
        target_txt = os.path.join(out_dir, stem + "_OCR.txt")
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
            for path in (target_txt, analysis_txt):
                if os.path.exists(path):
                    os.remove(path)
        else:
            for path in (target_txt, analysis_txt):
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
                    ln, tx, cf, mode = ocr_page_paddle(p, self._ocr, zoom=cfg.zoom,
                                                        preprocess_mode=cfg.preprocess_mode)
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
                    preprocessed = [apply_hard_corrections(tx) for tx in engine_texts]
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
                            preprocess_mode=cfg.preprocess_mode
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
                    corrected = apply_hard_corrections(corrected)

                    if cfg.output_pdf:
                        apply_text_overlay(page_obj, lines, corrected, zoom)
                    if cfg.output_txt:
                        append_ocr_text(target_txt, page_idx, corrected)
                    if cfg.output_analysis:
                        append_quality_analysis(analysis_txt, file_name, page_idx, raw_texts, confs, preprocess_mode, corrected)
                    if cfg.output_verify_log:
                        log_corrections(log_path, file_name, page_idx, raw_texts, corrected, disp_engine, ai_enabled=True)

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
                    page_obj, self._ocr, zoom=cfg.zoom, preprocess_mode=cfg.preprocess_mode
                )
                fixed = apply_hard_corrections(raw_texts)

                if cfg.output_pdf:
                    apply_text_overlay(page_obj, lines, fixed, cfg.zoom)
                if cfg.output_txt:
                    append_ocr_text(target_txt, page_num, fixed)
                if cfg.output_analysis:
                    append_quality_analysis(analysis_txt, file_name, page_num, raw_texts, confidences, preprocess_mode, fixed)
                if cfg.output_verify_log:
                    log_corrections(log_path, file_name, page_num, raw_texts, fixed, "PADDLE", ai_enabled=False)

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
        if cfg.output_analysis:
            self._log(f"  [V] 分析完成: {analysis_txt}")
        self._log(f"      平均速度: {avg_total:.2f}s/頁")
        self._report_file_status(file_index, pdf_path, "完成", f"平均速度: {avg_total:.2f}s/頁")
