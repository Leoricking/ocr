import os
import sys
import subprocess

# Windows / CUDA / cuDNN / cuBLAS DLL 路徑
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")

_DLL_DIR_HANDLES = []


def _add_dll_dir(path: str) -> bool:
    if sys.platform != "win32" or not path or not os.path.isdir(path):
        return False

    path = os.path.abspath(path)
    current = os.environ.get("PATH", "")
    existing = [p.lower() for p in current.split(os.pathsep) if p]
    if path.lower() not in existing:
        os.environ["PATH"] = path + os.pathsep + current

    if hasattr(os, "add_dll_directory"):
        try:
            _DLL_DIR_HANDLES.append(os.add_dll_directory(path))
        except OSError:
            return False
    return True


def configure_gpu_dll_paths() -> list[str]:
    if sys.platform != "win32":
        return []

    root = os.path.dirname(sys.executable)
    candidates = [
        os.path.join(root, "Lib", "site-packages", "nvidia", "cublas", "bin"),
        os.path.join(root, "Lib", "site-packages", "nvidia", "cudnn", "bin"),
        os.path.join(root, "Lib", "site-packages", "nvidia", "cuda_runtime", "bin"),
        os.path.join(root, "Lib", "site-packages", "paddle", "base"),
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
    ]

    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(os.path.join(cuda_path, "bin"))

    added = []
    for candidate in candidates:
        if _add_dll_dir(candidate):
            added.append(os.path.abspath(candidate))
    return added


GPU_DLL_PATHS = configure_gpu_dll_paths()

import io
import time

import difflib
from datetime import datetime
from tqdm import tqdm
import fitz  # PyMuPDF
from paddleocr import PaddleOCR
import anthropic
import cv2
import numpy as np

# ==============================================================
# --- 相依套件版本檢查 ---
# ==============================================================
import importlib.metadata as _meta

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
# 設定區
# ==============================================================
CLAUDE_API_KEY        = os.environ.get("CLAUDE_API_KEY", "YOUR_CLAUDE_API_KEY").strip()
CLAUDE_MODEL          = os.environ.get("CLAUDE_MODEL", "claude-3-5-sonnet-20240620").strip()
BATCH_SIZE            = 4      # 每次 API 請求合併的頁數（Paddle 模式）
API_TIMEOUT_DEFAULT   = 30.0
API_TIMEOUT_PHYSICS   = 60.0
CONF_SKIP_THRESHOLD   = 0.95   # 高置信度且無術語時跳過 API
PAGE_SEP              = "===PAGE_{n}==="

client = None


def get_anthropic_client():
    """延遲建立 API Client；未啟用 AI 時不初始化。"""
    global client
    if client is None:
        client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    return client


def validate_key() -> bool:
    """啟動時以最小呼叫驗證 API 金鑰是否有效。無效則停止程式。"""
    try:
        get_anthropic_client().messages.create(
            model=CLAUDE_MODEL,
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
        return True   # 非金鑰問題（如網路），不阻止啟動


# ==============================================================
# 引擎可用性偵測（啟動時執行一次）
# ==============================================================
ENGINES = {"paddle": True, "nougat": False, "surya": False}

try:
    from nougat import NougatModel            # pip install nougat-ocr
    ENGINES["nougat"] = True
except ImportError:
    pass

try:
    from surya.recognition import batch_recognition as _surya_recog   # pip install surya-ocr
    ENGINES["surya"] = True
except ImportError:
    pass

print(f"[引擎] Paddle=✓  Nougat={'✓' if ENGINES['nougat'] else '✗'}  "
      f"Surya={'✓' if ENGINES['surya'] else '✗'}")
if not ENGINES["nougat"]:
    print("       → 安裝物理模式引擎：pip install nougat-ocr")
if not ENGINES["surya"]:
    print("       → 安裝財經版面引擎：pip install surya-ocr")

# 延遲載入的模型快取
_nougat_model = None
_surya_model  = None
_surya_proc   = None


def _load_nougat():
    global _nougat_model
    if _nougat_model is None:
        import torch
        from nougat.utils.checkpoint import get_checkpoint
        ckpt = get_checkpoint(model_tag="0.1.0-base")
        _nougat_model = NougatModel.from_pretrained(ckpt)
        _nougat_model.eval()
        if torch.cuda.is_available():
            _nougat_model = _nougat_model.cuda()
        print("[引擎] Nougat 模型載入完成")
    return _nougat_model


def _load_surya():
    global _surya_model, _surya_proc
    if _surya_model is None:
        from surya.model.recognition.model import load_model
        from surya.model.recognition.processor import load_processor
        _surya_model = load_model()
        _surya_proc  = load_processor()
        print("[引擎] Surya 模型載入完成")
    return _surya_model, _surya_proc


# ==============================================================
# 關鍵字與對照表
# ==============================================================
FINANCE_KEYWORDS = ['股市', '理財', '周刊', '週刊', '基金', '投資', '財經']
PHYSICS_KEYWORDS = ['公式', '向量', '電磁', '物理', 'LaTeX', '微積分', '電場', '磁場']

HARD_CODED_CORRECTIONS = {
    "白大": "台大",
    "大物理": "台大物理",
    "輻樹聲": "賴樹聲",
    "賴樹生": "賴樹聲",
    "貨格考": "資格考",
    "貨格": "價格",
    "F-ma": "F=ma",
    "訊號土": "訊號±",
    "理輪": "理論",
    "梨力學": "熱力學",
    "電形學": "電磁學",
    "必修磁學": "必修電磁學",
    "衣捲": "交卷",
    "Harvard tenu了": "Harvard tenure",
    "Post-Doctoe": "Post-Doctor",
    "Post-Doctoor": "Post-Doctor",
    "AssistPtranoftea": "Assistant Professor",
    "D.Eardley日基研nt": "D. Eardley 研一下",
    "CRan1": "目錄",
    "波導管與共振腔125": "波導管與共振腔 125",
    "聶": "",
    "畢吳": "",
}

# ==============================================================
# Claude Prompts（依引擎分類）
# ==============================================================
_VERIFY = (
    "\n\n【格式規定】回傳時必須嚴格使用 ===PAGE_N=== 分隔每頁，不得省略或更改格式。"
    "\n【自我檢查】完成後請確認：\n"
    "1. 每頁輸出行數需與輸入完全一致；嚴禁合併含物理公式的行。\n"
    "2. 向量符號（$\\vec{E}$、$\\vec{B}$）與財經術語（殖利率等）不得消失。\n"
    "3. 確認已剔除亂碼字元（土、聶、畢、吳、車）。\n"
    "若整批幾乎全是亂碼，請在回覆最開頭輸出 [UNCERTAIN]，後續仍輸出最佳猜測結果。"
)

# Nougat 輸出：修正 LaTeX 語法 + 潤飾繁體中文說明
PROMPT_NOUGAT = (
    "你是一位精通物理與 LaTeX 的技術編輯。"
    "以下文字由 Nougat 引擎辨識自物理教材，可能含有 LaTeX 語法錯誤（缺失 $、錯誤括號）"
    "或繁體中文說明缺失。請修正 LaTeX 語法並潤飾中文解釋，維持行數不變。"
    + _VERIFY
)

# Surya 輸出：還原多欄位邏輯 + 核對財經數據
PROMPT_SURYA = (
    "你是一位專業財經編輯，擅長處理多欄位雜誌排版。"
    "以下文字由 Surya 版面分析引擎辨識，可能含有多欄位混排的邏輯錯誤或財經數字不準確。"
    "請還原正確的閱讀順序，核對財經數字與術語準確性，維持行數不變。"
    + _VERIFY
)

# Paddle 輸出：語意推理 + 亂碼過濾
PROMPT_PADDLE_FINANCE = (
    "你是一位專業財經編輯，擅長處理股市、理財、基金、週刊等金融內容。"
    "請透過語意推理修正 OCR 亂碼（如輻→賴、白→台），保持繁體中文，維持行數與順序。"
    + _VERIFY
)
PROMPT_PADDLE_PHYSICS = (
    "你是一位精通物理、電機工程、電磁學與繁體中文排版的資深技術編輯。\n"
    "以下內容來自泛黃、低對比、帶印章與手寫旁註的老舊掃描講義。"
    "OCR 可能把淡字、英文字母、標點、頁碼與物理術語辨識錯誤。\n"
    "請依上下文修正為正確且通順的繁體中文與專業物理術語，"
    "保留人名、校名、年份、頁碼、章節順序及英文專有名詞。\n"
    "公式請在能確定時還原為正確 LaTeX；無法確定時保留原字，不得憑空創造內容。\n"
    "手寫旁註只有在明顯與正文無關、破碎且不可判讀時才移除；"
    "不得刪除印刷正文，也不得把不同欄位或不同章節合併。\n"
    "請逐行對應回傳，維持原始行數與順序。"
    + _VERIFY
)

PROMPT_PADDLE_DEFAULT = (
    "你是一位專業繁體中文 OCR 校對編輯。"
    "以下內容來自老舊、泛黃、低對比或帶手寫註記的掃描文件。"
    "請依上下文修正錯字、英文拼字、標點與斷句，保留年份、頁碼、人名與專有名詞。"
    "無法確定的內容請保留原字，不得猜造；維持原始行數與順序，不要合併段落。"
    + _VERIFY
)


# ==============================================================
# 引擎選擇與 OCR 函數
# ==============================================================

def select_engine(raw_texts: list, confidences: list) -> str:
    """根據初步 OCR 文字內容自動選擇最適合的引擎"""
    sample = " ".join(raw_texts)[:500]
    if ENGINES["nougat"] and any(kw in sample for kw in PHYSICS_KEYWORDS):
        return "nougat"
    if ENGINES["surya"] and any(kw in sample for kw in FINANCE_KEYWORDS):
        return "surya"
    return "paddle"


def _decode_page_to_bgr(page, zoom: int):
    """將 PDF 頁面渲染為 OpenCV BGR 影像。"""
    pix = page.get_pixmap(
        matrix=fitz.Matrix(zoom, zoom),
        alpha=False,
        colorspace=fitz.csRGB,
    )
    image = np.frombuffer(pix.samples, dtype=np.uint8)
    image = image.reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def _build_preprocess_variants(image_bgr):
    """建立原圖、CLAHE 與自適應二值化三種候選。"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(
        gray, None, h=7, templateWindowSize=7, searchWindowSize=21
    )
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)
    blur = cv2.GaussianBlur(contrast, (0, 0), 1.0)
    sharpened = cv2.addWeighted(contrast, 1.45, blur, -0.45, 0)
    binary = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        13,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return {
        "ORIGINAL": image_bgr,
        "CLAHE": sharpened,
        "BINARY": binary_closed,
    }


def _normalize_paddle_result(result):
    """統一 PaddleOCR 回傳並依閱讀順序排序。"""
    if not result or not result[0]:
        return [], [], []
    lines = sorted(
        result[0],
        key=lambda x: (
            min(point[1] for point in x[0]),
            min(point[0] for point in x[0]),
        ),
    )
    texts = [str(line[1][0]).strip() for line in lines]
    confidences = [float(line[1][1]) for line in lines]
    return lines, texts, confidences


def _ocr_candidate_score(texts, confidences):
    """依平均置信度、有效文字量和短噪聲評估候選。"""
    if not texts or not confidences:
        return -1.0
    valid_texts = [t for t in texts if t and not t.isspace()]
    if not valid_texts:
        return -1.0
    avg_conf = sum(confidences) / len(confidences)
    char_count = sum(
        1 for text in valid_texts for ch in text
        if ch.isalnum() or "\u4e00" <= ch <= "\u9fff"
    )
    short_noise = sum(1 for t in valid_texts if len(t) == 1)
    return (
        avg_conf * 100.0
        + min(len(valid_texts), 80) * 0.20
        + min(char_count, 1500) * 0.004
        - short_noise * 0.18
    )


def ocr_page_paddle(page, ocr, zoom: int = 3):
    """
    每頁比較原圖、CLAHE 增強與自適應二值化，
    自動選擇置信度與有效文字量最佳的結果。
    """
    image_bgr = _decode_page_to_bgr(page, zoom)
    variants = _build_preprocess_variants(image_bgr)
    best = None

    for mode, candidate in variants.items():
        ok, encoded = cv2.imencode(".png", candidate)
        if not ok:
            continue
        try:
            result = ocr.ocr(encoded.tobytes(), cls=True)
        except Exception as exc:
            print(f"\n  [!] {mode} 預處理 OCR 失敗：{exc}")
            continue

        lines, texts, confidences = _normalize_paddle_result(result)
        score = _ocr_candidate_score(texts, confidences)
        if best is None or score > best["score"]:
            best = {
                "mode": mode,
                "score": score,
                "lines": lines,
                "texts": texts,
                "confidences": confidences,
            }

    if best is None:
        return [], [], []

    if best["mode"] != "ORIGINAL":
        avg_conf = (
            sum(best["confidences"]) / len(best["confidences"])
            if best["confidences"] else 0.0
        )
        print(
            f"\n  [影像增強] 採用 {best['mode']} "
            f"(行數={len(best['texts'])}, 平均置信度={avg_conf:.3f})"
        )

    return best["lines"], best["texts"], best["confidences"]


def ocr_texts_nougat(page, zoom: int = 3) -> list:
    """
    Nougat 辨識頁面，回傳文字行列表。
    失敗時回傳空列表，呼叫端應回退 Paddle 文字。
    """
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
    """
    Surya 辨識頁面，回傳文字行列表。
    失敗時回傳空列表，呼叫端應回退 Paddle 文字。
    """
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


def engine_prompt(engine: str, content_mode: str) -> tuple:
    """回傳 (prompt, timeout) 依引擎與內容模式"""
    if engine == "nougat":
        return PROMPT_NOUGAT, API_TIMEOUT_PHYSICS
    if engine == "surya":
        return PROMPT_SURYA, API_TIMEOUT_DEFAULT
    # paddle
    if content_mode == "FINANCE":
        return PROMPT_PADDLE_FINANCE, API_TIMEOUT_DEFAULT
    if content_mode == "PHYSICS":
        return PROMPT_PADDLE_PHYSICS, API_TIMEOUT_PHYSICS
    return PROMPT_PADDLE_DEFAULT, API_TIMEOUT_DEFAULT


# ==============================================================
# 通用工具函數
# ==============================================================

def apply_hard_corrections(texts: list) -> list:
    """套用 HARD_CODED_CORRECTIONS，不耗 Token，壓蓋前必呼叫"""
    result = []
    for t in texts:
        for wrong, right in HARD_CODED_CORRECTIONS.items():
            t = t.replace(wrong, right)
        result.append(t)
    return result


def align_lines(original: list, corrected: list) -> list:
    """逐行二次對比：行數不符時用 difflib 對齊，超過 20% 差距則回退原始"""
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
        # "insert" → skip extra lines from corrected

    return aligned if len(aligned) == len(original) else original


def call_claude_batch(batch_texts: list, prompt: str, timeout: float):
    """
    送出多頁文字至 Claude（PAGE_SEP 分隔），回傳：
      (list[list[str] | None], uncertain: bool)
    """
    parts = [PAGE_SEP.format(n=i + 1) + "\n" + "\n".join(texts)
             for i, texts in enumerate(batch_texts)]
    combined = "\n\n".join(parts)

    try:
        response = get_anthropic_client().messages.create(
            model=CLAUDE_MODEL,
            max_tokens=8192,
            system=prompt,
            messages=[{
                "role": "user",
                "content": (
                    f"以下是 {len(batch_texts)} 頁的 OCR 文字，每頁以 ===PAGE_N=== 分隔。"
                    "請依序校對每頁，回傳時必須維持 ===PAGE_N=== 分隔格式，不得省略。\n\n"
                    + combined
                ),
            }],
            timeout=timeout,
        )
        raw = response.content[0].text
        uncertain = raw.lstrip().startswith("[UNCERTAIN]")
        if uncertain:
            raw = raw.lstrip()[len("[UNCERTAIN]"):].lstrip("\n")

        result_pages = []
        for i in range(len(batch_texts)):
            sep      = PAGE_SEP.format(n=i + 1)
            next_sep = PAGE_SEP.format(n=i + 2)
            start    = raw.find(sep)
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


def apply_text_overlay(page, lines: list, corrected_lines: list, zoom: int):
    """
    將修正後的文字以透明層方式嵌入 PDF（bounding box 來自 PaddleOCR）。
    座標計算：
      - x0      : bbox 左邊緣
      - baseline: bbox 頂部 + 88% bbox 高度（PyMuPDF insert_text 以 baseline 為錨點）
      - fontsize : bbox 高度 × 0.72，避免手寫算式頁面文字溢出框外
    """
    for i, line in enumerate(lines):
        if i >= len(corrected_lines):
            break
        text = corrected_lines[i].strip()
        if not text:
            continue
        rect = line[0]                          # [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
        x0 = rect[0][0] / zoom
        y0 = rect[0][1] / zoom                  # bbox 頂部
        h  = (rect[2][1] - rect[0][1]) / zoom   # bbox 高度（縮放後）
        # baseline ≈ 頂部 + 88% 高度，使文字貼合原始行底線
        baseline = y0 + h * 0.88
        page.insert_text(
            fitz.Point(x0, baseline),
            text,
            fontsize=max(5, h * 0.72),          # 保守比例，避免手寫算式頁溢出
            fontname="china-t",
            render_mode=3,                      # 透明層，僅供搜尋
        )



def append_searchable_text(text_path: str, page_num: int, texts: list):
    """同步輸出純文字，方便核對 OCR 品質與建立知識庫。"""
    with open(text_path, "a", encoding="utf-8") as handle:
        handle.write(f"\n=== PAGE {page_num + 1} ===\n")
        for text in texts:
            cleaned = str(text).strip()
            if cleaned:
                handle.write(cleaned + "\n")


def log_corrections(log_path: str, file_name: str, page_num: int,
                    original: list, corrected: list, engine: str, ai_enabled: bool):
    """
    記錄每頁校正結果至 verify_log.txt（追加模式，每頁立即寫入）。
    - ai_enabled=True : 記錄有變動的行
    - ai_enabled=False: 記錄所有非空行，標註 [Raw OCR]
    """
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
# GPU / 環境預檢
# ==============================================================

def find_cudnn8_dll() -> str | None:
    """尋找 PaddlePaddle 2.6.1 需要的 cuDNN 8 DLL。"""
    if sys.platform != "win32":
        return None

    for directory in GPU_DLL_PATHS:
        dll_path = os.path.join(directory, "cudnn_ops_infer64_8.dll")
        if os.path.isfile(dll_path):
            return dll_path
    return None


def find_cublaslt11_dll() -> str | None:
    """尋找 PaddlePaddle CUDA 11 需要的 cuBLAS Lt DLL。"""
    if sys.platform != "win32":
        return None

    for directory in GPU_DLL_PATHS:
        dll_path = os.path.join(directory, "cublasLt64_11.dll")
        if os.path.isfile(dll_path):
            return dll_path
    return None


def gpu_runtime_self_test(timeout: int = 60) -> tuple[bool, str]:
    """
    在獨立子程序執行 Conv2D，實際觸發 CUDA/cuDNN。
    子程序若因原生 DLL 問題崩潰，不會拖垮 OCR 主程序。
    """
    test_code = (
        "import paddle\n"
        "paddle.set_device('gpu:0')\n"
        "x=paddle.randn([1,3,64,64])\n"
        "w=paddle.randn([8,3,3,3])\n"
        "y=paddle.nn.functional.conv2d(x,w)\n"
        "print('GPU_CUDNN_OK', list(y.shape))\n"
    )

    try:
        result = subprocess.run(
            [sys.executable, "-c", test_code],
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
        return False, "GPU/cuDNN 自我測試逾時"
    except Exception as exc:
        return False, f"無法執行 GPU 自我測試：{exc}"

    output = "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part.strip()
    )
    if result.returncode == 0 and "GPU_CUDNN_OK" in output:
        return True, output
    return False, output or f"子程序結束碼：{result.returncode}"


def print_environment_summary():
    """顯示目前核心環境，便於日後除錯。"""
    print("\n[環境檢查]")
    print(f"  Python     : {sys.version.split()[0]}")
    print(f"  執行檔     : {sys.executable}")

    try:
        import importlib.metadata as metadata
        for package in (
            "paddlepaddle-gpu",
            "paddleocr",
            "numpy",
            "pillow",
            "opencv-python",
            "albumentations",
            "albucore",
            "nvidia-cudnn-cu11",
            "nvidia-cublas-cu11",
        ):
            try:
                print(f"  {package:<20}: {metadata.version(package)}")
            except metadata.PackageNotFoundError:
                pass
    except Exception:
        pass

    if sys.platform == "win32":
        cudnn_dll = find_cudnn8_dll()
        cublas_dll = find_cublaslt11_dll()
        print(f"  cuDNN 8 DLL : {cudnn_dll or '未找到'}")
        print(f"  cuBLAS Lt DLL: {cublas_dll or '未找到'}")


# ==============================================================
# 主程式
# ==============================================================

def main():
    print("=== 數千本圖書 PDF 搜尋化系統 (v4.2.0 老舊掃描增強版) ===")

    input_path    = input("\n請輸入【來源路徑】: ").strip().replace('"', '')
    output_root   = input("請輸入【輸出路徑】: ").strip().replace('"', '')
    gpu_choice    = input("1: GPU[RTX 3060], 2: CPU: ").strip()
    use_gpu       = gpu_choice == '1'
    ai_choice     = input("是否啟用 Claude 自動校對？ (y/n): ").strip().lower()
    enable_claude = ai_choice == 'y'

    input_path = os.path.abspath(os.path.expanduser(input_path))
    output_root = os.path.abspath(os.path.expanduser(output_root))

    if not os.path.exists(input_path):
        print(f"\n[錯誤] 來源路徑不存在：{input_path}")
        return

    print_environment_summary()

    if use_gpu:
        cudnn_dll = find_cudnn8_dll()
        cublas_dll = find_cublaslt11_dll()
        if sys.platform == "win32" and (not cudnn_dll or not cublas_dll):
            print("\n[GPU] 缺少必要 CUDA DLL，為避免 Python 原生崩潰，改用 CPU。")
            if not cudnn_dll:
                print("      缺少：cudnn_ops_infer64_8.dll")
            if not cublas_dll:
                print("      缺少：cublasLt64_11.dll")
            print("      請重新執行 v4.1.1 一鍵安裝程式。")
            use_gpu = False
        else:
            print("\n[GPU] 正在以獨立子程序測試 CUDA/cuDNN...")
            gpu_ok, gpu_message = gpu_runtime_self_test()
            if gpu_ok:
                print("[GPU] CUDA/cuDNN 實際運算測試通過，將使用 gpu:0。")
            else:
                print("[GPU] CUDA/cuDNN 測試失敗，為避免主程序崩潰，已自動回退 CPU。")
                if gpu_message:
                    print(gpu_message[-2000:])
                use_gpu = False

    if enable_claude:
        if not CLAUDE_API_KEY or CLAUDE_API_KEY == "YOUR_CLAUDE_API_KEY":
            print("\n[API] 尚未設定 CLAUDE_API_KEY，已取消 AI 校對。")
            enable_claude = False
        elif not validate_key():
            return

    print(f"\n[執行裝置] {'GPU' if use_gpu else 'CPU'}")

    ocr = PaddleOCR(
        lang='chinese_cht',
        use_gpu=use_gpu,
        use_angle_cls=True,
        det_db_thresh=0.2,
        det_db_unclip_ratio=1.8,
        show_log=False,
        enable_mkldnn=not use_gpu
    )
    os.makedirs(output_root, exist_ok=True)
    log_path = os.path.join(output_root, "verify_log.txt")

    files = (
        sorted(
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith(".pdf")
        )
        if os.path.isdir(input_path) else [input_path]
    )

    if not files:
        print(f"\n[錯誤] 找不到 PDF：{input_path}")
        return

    for src_path in files:
        file_name   = os.path.basename(src_path)
        target_pdf  = os.path.join(output_root, os.path.splitext(file_name)[0] + "_OCR.pdf")
        target_txt  = os.path.join(output_root, os.path.splitext(file_name)[0] + "_OCR.txt")
        if os.path.exists(target_txt):
            os.remove(target_txt)
        try:
            doc = fitz.open(src_path)
        except Exception as exc:
            print(f"\n  [!] 無法開啟 PDF，已略過：{src_path}\n      {exc}")
            continue
        pages       = list(doc)
        total_pages = len(pages)

        pbar = tqdm(total=total_pages, desc=f"{file_name[:14]}",
                    ascii=" █", ncols=115, unit="頁")
        page_times = []

        if enable_claude:
            for batch_start in range(0, total_pages, BATCH_SIZE):
                batch_pages = pages[batch_start: batch_start + BATCH_SIZE]

                # Step 1: PaddleOCR（取得 bounding box + 初步文字 + 置信度）
                ocr_data = []   # [lines, raw_texts, confidences, zoom]
                for p in batch_pages:
                    ln, tx, cf = ocr_page_paddle(p, ocr, zoom=3)
                    ocr_data.append([ln, tx, cf, 3])

                # Step 2: 偵測內容模式與引擎
                sample      = " ".join(" ".join(d[1]) for d in ocr_data)
                content_mode = ("FINANCE" if any(kw in sample for kw in FINANCE_KEYWORDS)
                                else "PHYSICS" if any(kw in sample for kw in PHYSICS_KEYWORDS)
                                else "DEFAULT")
                engine      = select_engine(
                    [t for d in ocr_data for t in d[1]],
                    [c for d in ocr_data for c in d[2]]
                )

                # Step 3: 若使用專用引擎，取得更精確的文字（bounding box 仍用 Paddle）
                engine_texts = []   # per-page list of text lines from specialist engine
                if engine == "nougat":
                    for j, (p, d) in enumerate(zip(batch_pages, ocr_data)):
                        nt = ocr_texts_nougat(p) or d[1]   # fallback to paddle
                        engine_texts.append(nt)
                elif engine == "surya":
                    for j, (p, d) in enumerate(zip(batch_pages, ocr_data)):
                        st = ocr_texts_surya(p) or d[1]
                        engine_texts.append(st)
                else:
                    engine_texts = [d[1] for d in ocr_data]

                # Step 4: 決定是否呼叫 Claude
                all_confs  = [c for d in ocr_data for c in d[2]]
                avg_conf   = sum(all_confs) / len(all_confs) if all_confs else 0
                has_terms  = content_mode != "DEFAULT"
                skip_api   = avg_conf >= CONF_SKIP_THRESHOLD and not has_terms

                if skip_api:
                    corrected_pages = [None] * len(batch_pages)
                    uncertain       = False
                    disp_engine     = f"{engine.upper()}+SKIP"
                    elapsed         = 0.0
                else:
                    prompt, timeout = engine_prompt(engine, content_mode)
                    preprocessed    = [apply_hard_corrections(tx) for tx in engine_texts]
                    t0 = time.perf_counter()
                    corrected_pages, uncertain = call_claude_batch(preprocessed, prompt, timeout)
                    elapsed     = time.perf_counter() - t0
                    disp_engine = engine.upper()

                # Step 5: 套用修正、壓蓋、記錄
                for j, (page_obj, (lines, raw_texts, confs, zoom)) in enumerate(
                        zip(batch_pages, ocr_data)):
                    page_idx  = batch_start + j
                    corrected = corrected_pages[j]

                    # [UNCERTAIN] → 以 zoom=4 重新 OCR
                    if uncertain:
                        lines, raw_texts, confs = ocr_page_paddle(page_obj, ocr, zoom=4)
                        ocr_data[j] = [lines, raw_texts, confs, 4]
                        zoom = 4

                    if corrected is None:
                        corrected = engine_texts[j]
                    else:
                        corrected = align_lines(engine_texts[j], corrected)

                    corrected = apply_hard_corrections(corrected)
                    apply_text_overlay(page_obj, lines, corrected, zoom)
                    append_searchable_text(target_txt, page_idx, corrected)
                    log_corrections(log_path, file_name, page_idx,
                                    raw_texts, corrected, disp_engine, ai_enabled=True)

                    page_times.append(elapsed / max(len(batch_pages), 1))
                    avg       = sum(page_times) / len(page_times)
                    remaining = avg * (total_pages - len(page_times))
                    pbar.set_postfix(速度=f"{avg:.2f}s/頁", 引擎=disp_engine,
                                     置信=f"{avg_conf:.2f}", 剩餘=f"{remaining:.0f}s")
                    pbar.update(1)

        else:
            # AI 關閉：PaddleOCR + HARD_CODED_CORRECTIONS，仍產生 Log
            for page_num, page_obj in enumerate(pages):
                t0 = time.perf_counter()
                lines, raw_texts, _ = ocr_page_paddle(page_obj, ocr, zoom=3)
                fixed = apply_hard_corrections(raw_texts)
                apply_text_overlay(page_obj, lines, fixed, 3)
                append_searchable_text(target_txt, page_num, fixed)
                elapsed = time.perf_counter() - t0

                log_corrections(log_path, file_name, page_num,
                                raw_texts, fixed, "PADDLE", ai_enabled=False)

                page_times.append(elapsed)
                avg       = sum(page_times) / len(page_times)
                remaining = avg * (total_pages - len(page_times))
                pbar.set_postfix(速度=f"{avg:.2f}s/頁", 引擎="PADDLE+RAW",
                                 剩餘=f"{remaining:.0f}s")
                pbar.update(1)

        pbar.close()
        try:
            doc.save(target_pdf, garbage=4, deflate=True)
        except PermissionError:
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
            fallback = target_pdf.replace(".pdf", f"_{ts}.pdf")
            print(f"\n  [!] 檔案被佔用，改存為: {fallback}")
            doc.save(fallback, garbage=4, deflate=True)
        doc.close()
        avg_total = sum(page_times) / len(page_times) if page_times else 0
        print(f"  [V] 完成: {target_pdf}  (平均 {avg_total:.2f}s/頁)")

    print(f"\n  校正日誌: {log_path}")


if __name__ == "__main__":
    main()
