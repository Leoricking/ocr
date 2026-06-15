import sys
import types

# Lightweight import stubs for environments without OCR runtime packages.
if "paddleocr" not in sys.modules:
    paddleocr = types.ModuleType("paddleocr")
    paddleocr.PaddleOCR = object
    sys.modules["paddleocr"] = paddleocr
if "anthropic" not in sys.modules:
    anthropic = types.ModuleType("anthropic")
    anthropic.Anthropic = object
    anthropic.AuthenticationError = type("AuthenticationError", (Exception,), {})
    anthropic.APITimeoutError = type("APITimeoutError", (Exception,), {})
    anthropic.APIConnectionError = type("APIConnectionError", (Exception,), {})
    sys.modules["anthropic"] = anthropic

import os
from pathlib import Path

from ocr_core import OCRConfig, OCRProcessor, apply_safe_corrections, log_corrections


def test_verified_exact_corrections():
    corrected, sources = apply_safe_corrections(
        ["股價漲多就有懼高痘?", "日時寺間]是什麼?", "http://ww.inewton.com.tw"],
        "今周刊：摸透主力思維.pdf",
    )
    assert corrected == [
        "股價漲多就有懼高症？",
        "「時間」是什麼？",
        "http://www.inewton.com.tw",
    ]
    assert all(sources)


def test_document_specific_correction_does_not_leak():
    corrected, _ = apply_safe_corrections(
        ["1977白大物理採第一名畢業"], "賴樹聲 電磁波.pdf"
    )
    assert corrected == ["1977台大物理系第一名畢業"]

    other, _ = apply_safe_corrections(
        ["1977白大物理採第一名畢業"], "其他文件.pdf"
    )
    assert other == ["1977白大物理採第一名畢業"]


def test_formula_dense_line_is_not_broadly_rewritten():
    original = r"E=E_0 e^{-j\beta z} + 聶"
    corrected, sources = apply_safe_corrections([original], "賴樹聲 電磁波.pdf")
    assert corrected == [original]
    assert sources == [""]


def test_context_rule_requires_page_context():
    corrected, _ = apply_safe_corrections(
        ["千萬不要只靠運氣", "王萬要賭後", "應計算機率再下注"],
        "今周刊：摸透主力思維.pdf",
    )
    assert corrected[1] == "千萬不要賭"

    unchanged, _ = apply_safe_corrections(["王萬要賭後"], "其他.pdf")
    assert unchanged[0] == "王萬要賭後"


def test_generated_ocr_pdf_is_excluded(tmp_path: Path):
    src = tmp_path / "src"
    out = src / "output"
    src.mkdir()
    out.mkdir()
    (src / "book.pdf").write_bytes(b"%PDF-1.4")
    (src / "book_OCR.pdf").write_bytes(b"%PDF-1.4")
    (src / "book_OCR_OCR.pdf").write_bytes(b"%PDF-1.4")
    (out / "other.pdf").write_bytes(b"%PDF-1.4")

    cfg = OCRConfig(
        input_path=str(src), output_path=str(out), recursive=True,
        exclude_ocr_output=True,
    )
    files = OCRProcessor(cfg).collect_pdf_files()
    assert files == [str(src / "book.pdf")]


def test_verify_log_contains_only_changes(tmp_path: Path):
    log = tmp_path / "verify_log.txt"
    log_corrections(
        str(log), "sample.pdf", 0,
        ["正常文字", "懼高痘"],
        ["正常文字", "懼高症"],
        "PADDLE", False,
        correction_sources=["", "exact-phrase"],
    )
    text = log.read_text(encoding="utf-8")
    assert "正常文字" not in text
    assert "原始：懼高痘" in text
    assert "修正：懼高症" in text
    assert "exact-phrase" in text
