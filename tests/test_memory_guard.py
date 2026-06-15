import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

sys.modules.setdefault("paddleocr", types.SimpleNamespace(PaddleOCR=object))

class _Anthropic:
    def __init__(self, *args, **kwargs):
        pass

sys.modules.setdefault("anthropic", types.SimpleNamespace(Anthropic=_Anthropic))

spec = importlib.util.spec_from_file_location("ocr_core_memory_test", ROOT / "ocr_core.py")
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)


def test_memory_error_detection():
    assert core._is_memory_error(MemoryError("x"))
    assert core._is_memory_error(RuntimeError("Unable to allocate 26.8 MiB"))
    assert core._is_memory_error(RuntimeError("bad allocation"))
    assert not core._is_memory_error(RuntimeError("ordinary failure"))


def test_raw_success_survives_enhanced_candidate_oom(monkeypatch):
    class Rect:
        width = 100
        height = 100

    class Page:
        rect = Rect()

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    monkeypatch.setattr(core, "_page_to_bgr", lambda page, zoom: image)
    monkeypatch.setattr(
        core,
        "_build_preprocess_candidate",
        lambda img, mode: img if mode == "ORIGINAL" else np.zeros((100, 100), dtype=np.uint8),
    )

    box = [[0, 0], [10, 0], [10, 10], [0, 10]]

    class OCR:
        calls = 0

        def ocr(self, data, cls=True):
            self.calls += 1
            if self.calls > 1:
                raise MemoryError("Unable to allocate 7.03 MiB")
            return [[[box, ("測試", 0.99)]]]

    lines, texts, confidences, mode = core.ocr_page_paddle(
        Page(), OCR(), zoom=3, preprocess_mode="auto"
    )
    assert mode == "ORIGINAL"
    assert texts == ["測試"]
    assert confidences == [0.99]
    assert lines


def test_zoom_is_capped_for_large_page():
    class Rect:
        width = 3000
        height = 3000

    class Page:
        rect = Rect()

    value = core._effective_zoom_for_page(Page(), 3.0, 24_000_000)
    assert 1.5 <= value < 3.0
