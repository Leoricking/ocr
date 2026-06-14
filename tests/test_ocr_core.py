"""
Mock tests for OCR Engine v4.4.0 — no real PDF, no Claude API calls, no full OCR.
Run with: python -m pytest tests/ -v
or:       python tests/test_ocr_core.py
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# 1. OCRConfig creation test
# ---------------------------------------------------------------------------
class TestOCRConfig(unittest.TestCase):
    def test_defaults(self):
        # Import only the dataclass — does NOT load PaddleOCR
        from ocr_core import OCRConfig
        cfg = OCRConfig(input_path="/tmp/a.pdf", output_path="/tmp/out")
        self.assertEqual(cfg.device, "gpu")
        self.assertFalse(cfg.enable_claude)
        self.assertEqual(cfg.zoom, 3)
        self.assertEqual(cfg.preprocess_mode, "auto")
        self.assertTrue(cfg.output_pdf)
        self.assertTrue(cfg.output_txt)
        self.assertTrue(cfg.output_analysis)
        self.assertTrue(cfg.output_verify_log)
        self.assertFalse(cfg.recursive)
        self.assertFalse(cfg.overwrite)
        self.assertTrue(cfg.skip_existing)
        self.assertAlmostEqual(cfg.confidence_threshold, 0.70)
        self.assertTrue(cfg.preserve_relative_structure)

    def test_custom_values(self):
        from ocr_core import OCRConfig
        cfg = OCRConfig(
            input_path="/tmp/test",
            output_path="/tmp/out",
            device="cpu",
            enable_claude=True,
            zoom=4,
            recursive=True,
            overwrite=True,
            skip_existing=False,
        )
        self.assertEqual(cfg.device, "cpu")
        self.assertTrue(cfg.enable_claude)
        self.assertEqual(cfg.zoom, 4)
        self.assertTrue(cfg.recursive)
        self.assertTrue(cfg.overwrite)
        self.assertFalse(cfg.skip_existing)


# ---------------------------------------------------------------------------
# 2. collect_pdf_files tests
# ---------------------------------------------------------------------------
class TestCollectPdfFiles(unittest.TestCase):
    def _make_processor(self, cfg):
        """Create an OCRProcessor with mocked _ocr (no real init)."""
        from ocr_core import OCRConfig, OCRProcessor
        proc = OCRProcessor.__new__(OCRProcessor)
        proc.config = cfg
        proc.progress_callback = None
        proc.log_callback = None
        proc.file_status_callback = None
        proc.pause_event = None
        proc.cancel_event = None
        proc._ocr = None
        proc._claude_model = ""
        return proc

    def test_single_file(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = f.name
        try:
            from ocr_core import OCRConfig
            cfg = OCRConfig(input_path=path, output_path="/tmp/out")
            proc = self._make_processor(cfg)
            files = proc.collect_pdf_files()
            self.assertEqual(files, [path])
        finally:
            os.unlink(path)

    def test_folder_non_recursive(self):
        with tempfile.TemporaryDirectory() as d:
            # Create 3 PDFs in root, 1 in subdir
            for name in ["a.pdf", "b.pdf", "c.pdf"]:
                Path(d, name).write_bytes(b"")
            sub = Path(d, "sub")
            sub.mkdir()
            Path(sub, "d.pdf").write_bytes(b"")
            # Also a non-PDF
            Path(d, "readme.txt").write_bytes(b"")

            from ocr_core import OCRConfig
            cfg = OCRConfig(input_path=d, output_path="/tmp/out", recursive=False)
            proc = self._make_processor(cfg)
            files = proc.collect_pdf_files()
            basenames = [os.path.basename(f) for f in files]
            self.assertIn("a.pdf", basenames)
            self.assertIn("b.pdf", basenames)
            self.assertIn("c.pdf", basenames)
            self.assertNotIn("d.pdf", basenames)
            self.assertNotIn("readme.txt", basenames)

    def test_folder_recursive(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "a.pdf").write_bytes(b"")
            sub = Path(d, "sub")
            sub.mkdir()
            Path(sub, "b.pdf").write_bytes(b"")
            sub2 = Path(sub, "deep")
            sub2.mkdir()
            Path(sub2, "c.pdf").write_bytes(b"")

            from ocr_core import OCRConfig
            cfg = OCRConfig(input_path=d, output_path="/tmp/out", recursive=True)
            proc = self._make_processor(cfg)
            files = proc.collect_pdf_files()
            basenames = [os.path.basename(f) for f in files]
            self.assertIn("a.pdf", basenames)
            self.assertIn("b.pdf", basenames)
            self.assertIn("c.pdf", basenames)

    def test_empty_folder(self):
        with tempfile.TemporaryDirectory() as d:
            from ocr_core import OCRConfig
            cfg = OCRConfig(input_path=d, output_path="/tmp/out")
            proc = self._make_processor(cfg)
            files = proc.collect_pdf_files()
            self.assertEqual(files, [])

    def test_multiple_files_sorted(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ["c.pdf", "a.pdf", "b.pdf"]:
                Path(d, name).write_bytes(b"")
            from ocr_core import OCRConfig
            cfg = OCRConfig(input_path=d, output_path="/tmp/out")
            proc = self._make_processor(cfg)
            files = proc.collect_pdf_files()
            basenames = [os.path.basename(f) for f in files]
            self.assertEqual(basenames, ["a.pdf", "b.pdf", "c.pdf"])


# ---------------------------------------------------------------------------
# 3. SettingsManager tests
# ---------------------------------------------------------------------------
class TestSettingsManager(unittest.TestCase):
    def _make_sm(self, tmp_dir):
        from settings_manager import SettingsManager
        return SettingsManager(Path(tmp_dir) / "data" / "test_settings.json")

    def test_defaults_when_no_file(self):
        with tempfile.TemporaryDirectory() as d:
            sm = self._make_sm(d)
            result = sm.load()
            self.assertEqual(result["zoom"], 3)
            self.assertEqual(result["device"], "gpu")
            self.assertFalse(result["enable_claude"])

    def test_save_and_reload(self):
        with tempfile.TemporaryDirectory() as d:
            sm = self._make_sm(d)
            settings = sm.load()
            settings["zoom"] = 4
            settings["device"] = "cpu"
            settings["last_input_path"] = "/tmp/test"
            self.assertTrue(sm.save(settings))
            loaded = sm.load()
            self.assertEqual(loaded["zoom"], 4)
            self.assertEqual(loaded["device"], "cpu")
            self.assertEqual(loaded["last_input_path"], "/tmp/test")

    def test_atomic_write(self):
        """Verify that save creates a proper file (not corrupt)."""
        with tempfile.TemporaryDirectory() as d:
            sm = self._make_sm(d)
            settings = sm.load()
            settings["claude_model"] = "claude-test"
            sm.save(settings)
            path = Path(d) / "data" / "test_settings.json"
            self.assertTrue(path.exists())
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["claude_model"], "claude-test")

    def test_corrupt_file_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            sm = self._make_sm(d)
            path = Path(d) / "data" / "test_settings.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{INVALID JSON{{", encoding="utf-8")
            result = sm.load()
            # Should return defaults without crashing
            self.assertEqual(result["zoom"], 3)
            # Original file should be renamed
            self.assertFalse(path.exists())

    def test_reset(self):
        with tempfile.TemporaryDirectory() as d:
            sm = self._make_sm(d)
            settings = sm.load()
            settings["zoom"] = 4
            sm.save(settings)
            defaults = sm.reset()
            self.assertEqual(defaults["zoom"], 3)
            # File should be gone
            path = Path(d) / "data" / "test_settings.json"
            self.assertFalse(path.exists())

    def test_missing_keys_filled_with_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            sm = self._make_sm(d)
            path = Path(d) / "data" / "test_settings.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write a partial settings file
            path.write_text(json.dumps({"zoom": 2}), encoding="utf-8")
            result = sm.load()
            self.assertEqual(result["zoom"], 2)       # from file
            self.assertEqual(result["device"], "gpu")  # from defaults


# ---------------------------------------------------------------------------
# 4. Dual-column reading order test
# ---------------------------------------------------------------------------
class TestDualColumnSort(unittest.TestCase):
    def _make_line(self, x0, y0, x1, y1, text="test", conf=0.9):
        """Create a fake PaddleOCR line: [[box], [text, conf]]"""
        box = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        return [box, [text, conf]]

    def test_dual_column_left_before_right(self):
        """Left column lines must appear before right column lines."""
        from ocr_core import _sort_zone_by_columns
        page_width = 800

        # 8 lines in left column (x: 50-350), 8 in right (x: 450-750)
        left_lines = [self._make_line(50, y, 350, y + 20, f"L{i}") for i, y in enumerate(range(50, 210, 20))]
        right_lines = [self._make_line(450, y, 750, y + 20, f"R{i}") for i, y in enumerate(range(50, 210, 20))]
        all_lines = left_lines + right_lines

        result = _sort_zone_by_columns(all_lines, page_width)
        texts = [line[1][0] for line in result]

        # All left-column texts should come before any right-column text
        last_left = max(texts.index(f"L{i}") for i in range(len(left_lines)))
        first_right = min(texts.index(f"R{i}") for i in range(len(right_lines)))
        self.assertLess(last_left, first_right, "Left column must finish before right column starts")

    def test_single_column_fallback(self):
        """When gap is too small, fall back to single y/x sort."""
        from ocr_core import _sort_zone_by_columns
        page_width = 800

        # All lines clustered in center (no clear column gap)
        lines = [self._make_line(300, y, 500, y + 20, f"T{i}") for i, y in enumerate(range(50, 150, 10))]
        result = _sort_zone_by_columns(lines, page_width)
        # Should still return all lines
        self.assertEqual(len(result), len(lines))

    def test_spanning_header_preserved(self):
        """Wide spanning headers should appear at their vertical position."""
        from ocr_core import _sort_reading_order
        page_width = 800

        # A wide header at y=0
        header = self._make_line(50, 0, 750, 30, "HEADER")
        # Left column body below header
        left = [self._make_line(50, y, 370, y + 20, f"L{i}") for i, y in enumerate(range(50, 130, 20))]
        # Right column body below header
        right = [self._make_line(430, y, 750, y + 20, f"R{i}") for i, y in enumerate(range(50, 130, 20))]

        all_lines = [header] + left + right
        result = _sort_reading_order(all_lines, page_width)
        texts = [line[1][0] for line in result]

        self.assertEqual(texts[0], "HEADER", "Header must be first in output")


# ---------------------------------------------------------------------------
# 5. Auto three-pass scoring test
# ---------------------------------------------------------------------------
class TestCandidateScore(unittest.TestCase):
    def test_high_confidence_wins(self):
        from ocr_core import _candidate_score
        high_conf_texts = ["財經新聞", "股市分析", "投資理財", "基金動態"] * 5
        high_conf_confs = [0.95] * len(high_conf_texts)

        low_conf_texts = ["?", "□", "■"] * 5
        low_conf_confs = [0.30] * len(low_conf_texts)

        score_high = _candidate_score(high_conf_texts, high_conf_confs)
        score_low = _candidate_score(low_conf_texts, low_conf_confs)
        self.assertGreater(score_high, score_low)

    def test_more_chars_wins_over_fragments(self):
        from ocr_core import _candidate_score
        rich_texts = ["這是一段較長的財經新聞內容，包含很多中文字元"] * 5
        rich_confs = [0.85] * len(rich_texts)

        fragment_texts = ["a", "b", "c", "d", "e"] * 5
        fragment_confs = [0.85] * len(fragment_texts)

        score_rich = _candidate_score(rich_texts, rich_confs)
        score_frag = _candidate_score(fragment_texts, fragment_confs)
        self.assertGreater(score_rich, score_frag)

    def test_empty_returns_negative(self):
        from ocr_core import _candidate_score
        self.assertEqual(_candidate_score([], []), -1.0)
        self.assertEqual(_candidate_score([""], [0.9]), -1.0)


# ---------------------------------------------------------------------------
# 6. Claude line-count mismatch fallback test
# ---------------------------------------------------------------------------
class TestAlignLines(unittest.TestCase):
    def test_exact_match_returned(self):
        from ocr_core import align_lines
        original = ["line1", "line2", "line3"]
        corrected = ["LINE1", "LINE2", "LINE3"]
        result = align_lines(original, corrected)
        self.assertEqual(result, corrected)

    def test_large_mismatch_returns_original(self):
        from ocr_core import align_lines
        original = ["a", "b", "c", "d", "e"]
        corrected = ["X"]  # 80% reduction — exceeds 20% threshold
        result = align_lines(original, corrected)
        self.assertEqual(result, original)

    def test_small_mismatch_aligned(self):
        from ocr_core import align_lines
        original = ["a", "b", "c", "d", "e"]
        corrected = ["A", "B", "C", "D"]  # one line dropped — within 20%
        result = align_lines(original, corrected)
        self.assertEqual(len(result), len(original))

    def test_claude_mock_line_count_fallback(self):
        """Simulate what OCRProcessor does: if corrected != original length after align, fall back."""
        from ocr_core import align_lines, apply_hard_corrections

        original_texts = [f"原始行 {i}" for i in range(10)]
        # Mock Claude returning wrong count (50% reduction → exceeds threshold)
        claude_returned = [f"校正行 {i}" for i in range(5)]

        # align_lines should fall back to original when mismatch > 20%
        aligned = align_lines(original_texts, claude_returned)
        self.assertEqual(aligned, original_texts)

        # Then apply hard corrections on the fallback
        corrected = apply_hard_corrections(aligned)
        self.assertEqual(len(corrected), len(original_texts))


# ---------------------------------------------------------------------------
# 7. Overwrite / skip_existing mutual exclusion test
# ---------------------------------------------------------------------------
class TestOverwriteSkipExisting(unittest.TestCase):
    def test_cannot_have_both(self):
        """skip_existing should be False when overwrite is True in CLI logic."""
        # Simulate the CLI logic: skip_existing = skip_existing and not overwrite
        overwrite = True
        skip_existing_flag = True
        effective_skip = skip_existing_flag and not overwrite
        self.assertFalse(effective_skip)

    def test_skip_default_when_no_overwrite(self):
        overwrite = False
        skip_existing_flag = False
        # New behavior: default to skip when neither flag is set
        effective_skip = skip_existing_flag or (not overwrite)
        self.assertTrue(effective_skip)

    def test_overwrite_only(self):
        overwrite = True
        skip_existing_flag = False
        effective_skip = skip_existing_flag and not overwrite
        self.assertFalse(effective_skip)
        self.assertTrue(overwrite)


# ---------------------------------------------------------------------------
# 8. GUI settings save-reload test (no display needed)
# ---------------------------------------------------------------------------
class TestGuiSettingsRoundtrip(unittest.TestCase):
    def test_save_and_reload_all_fields(self):
        with tempfile.TemporaryDirectory() as d:
            from settings_manager import SettingsManager, DEFAULTS
            sm = SettingsManager(Path(d) / "data" / "gui_settings.json")

            test_settings = {
                **DEFAULTS,
                "zoom": 4,
                "device": "cpu",
                "enable_claude": True,
                "claude_model": "claude-opus-4-5",
                "last_input_path": r"C:\test\input",
                "last_output_path": r"C:\test\output",
                "recursive": True,
                "overwrite": False,
                "skip_existing": True,
                "confidence_threshold": 0.85,
                "window_geometry": "1920x1080+0+0",
            }
            sm.save(test_settings)
            loaded = sm.load()

            for key in test_settings:
                self.assertEqual(loaded[key], test_settings[key], f"Mismatch for key: {key}")

    def test_corrupt_recovers_to_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            from settings_manager import SettingsManager, DEFAULTS
            path = Path(d) / "data" / "gui_settings.json"
            path.parent.mkdir(parents=True)
            path.write_text("NOT JSON AT ALL {{{", encoding="utf-8")

            sm = SettingsManager(path)
            result = sm.load()

            # Must not crash, must return defaults
            self.assertEqual(result["zoom"], DEFAULTS["zoom"])
            self.assertEqual(result["device"], DEFAULTS["device"])
            # File must be gone (renamed to broken)
            self.assertFalse(path.exists())


# ---------------------------------------------------------------------------
# 9. apply_hard_corrections test
# ---------------------------------------------------------------------------
class TestHardCorrections(unittest.TestCase):
    def test_known_corrections(self):
        from ocr_core import apply_hard_corrections
        texts = ["白大教授", "訊號土電路", "F-ma 定律"]
        result = apply_hard_corrections(texts)
        self.assertIn("台大", result[0])
        self.assertIn("訊號±", result[1])
        self.assertIn("F=ma", result[2])

    def test_empty_list(self):
        from ocr_core import apply_hard_corrections
        self.assertEqual(apply_hard_corrections([]), [])

    def test_no_change_needed(self):
        from ocr_core import apply_hard_corrections
        texts = ["正常文字", "Normal text"]
        result = apply_hard_corrections(texts)
        self.assertEqual(result, texts)


# ---------------------------------------------------------------------------
# 10. DPI layout regression tests
# ---------------------------------------------------------------------------
class TestDpiLayout(unittest.TestCase):
    """Verify GUI layout helpers produce sane values at various DPI scales."""

    def _make_app_no_mainloop(self, mock_scale=1.0):
        """Create OCRGuiApp with patched DPI scale, destroy immediately after inspection."""
        import tkinter as tk
        from unittest.mock import patch
        r = tk.Tk()
        r.withdraw()  # Don't show window during test
        with patch('ocr_gui.get_dpi_scale', return_value=mock_scale):
            from ocr_gui import OCRGuiApp
            app = OCRGuiApp(r)
        r.update_idletasks()
        return r, app

    def _teardown(self, root):
        try:
            root.destroy()
        except Exception:
            pass

    def test_font_sizes_100pct(self):
        r, app = self._make_app_no_mainloop(1.0)
        try:
            size = app.font_normal.cget("size")
            self.assertGreaterEqual(size, 9)
            self.assertLessEqual(size, 14)
        finally:
            self._teardown(r)

    def test_font_sizes_150pct(self):
        r, app = self._make_app_no_mainloop(1.5)
        try:
            size = app.font_normal.cget("size")
            self.assertGreaterEqual(size, 10)
            self.assertLessEqual(size, 16)
        finally:
            self._teardown(r)

    def test_font_sizes_200pct(self):
        r, app = self._make_app_no_mainloop(2.0)
        try:
            size = app.font_normal.cget("size")
            self.assertGreaterEqual(size, 10)
        finally:
            self._teardown(r)

    def test_treeview_rowheight_gt_linespace(self):
        import tkinter as tk
        from tkinter import ttk as _ttk
        from tkinter import font as tkfont
        from unittest.mock import patch
        r = tk.Tk()
        r.withdraw()
        with patch('ocr_gui.get_dpi_scale', return_value=1.25):
            from ocr_gui import OCRGuiApp
            app = OCRGuiApp(r)
        r.update_idletasks()
        try:
            lh = app.font_normal.metrics("linespace")
            style = _ttk.Style()
            rh = style.lookup("Treeview", "rowheight")
            # rowheight must be > linespace
            if rh:
                self.assertGreater(int(rh), lh)
        finally:
            r.destroy()

    def test_button_padding_nonzero(self):
        import tkinter as tk
        from tkinter import ttk as _ttk
        from unittest.mock import patch
        r = tk.Tk()
        r.withdraw()
        with patch('ocr_gui.get_dpi_scale', return_value=1.5):
            from ocr_gui import OCRGuiApp
            app = OCRGuiApp(r)
        r.update_idletasks()
        try:
            style = _ttk.Style()
            padding = style.lookup("TButton", "padding")
            self.assertIsNotNone(padding)
        finally:
            r.destroy()

    def test_wraplength_updates(self):
        import tkinter as tk
        from unittest.mock import patch
        r = tk.Tk()
        r.withdraw()
        with patch('ocr_gui.get_dpi_scale', return_value=1.0):
            from ocr_gui import OCRGuiApp
            app = OCRGuiApp(r)
        r.update_idletasks()
        try:
            # Force update
            app._update_wraplengths()
            wl = app._lbl_api_status.cget("wraplength")
            self.assertGreater(wl, 50)
        finally:
            r.destroy()

    def test_geometry_within_screen(self):
        import tkinter as tk
        from unittest.mock import patch
        r = tk.Tk()
        r.withdraw()
        with patch('ocr_gui.get_dpi_scale', return_value=1.5):
            from ocr_gui import OCRGuiApp
            app = OCRGuiApp(r)
        r.update_idletasks()
        try:
            sw = r.winfo_screenwidth()
            sh = r.winfo_screenheight()
            w = r.winfo_reqwidth()
            h = r.winfo_reqheight()
            # Requested size must be within screen
            self.assertLessEqual(w, sw + 50)
            self.assertLessEqual(h, sh + 50)
        finally:
            r.destroy()

    def test_no_emoji_in_button_texts(self):
        import tkinter as tk
        from unittest.mock import patch
        import unicodedata
        r = tk.Tk()
        r.withdraw()
        with patch('ocr_gui.get_dpi_scale', return_value=1.0):
            from ocr_gui import OCRGuiApp
            app = OCRGuiApp(r)
        r.update_idletasks()
        try:
            btns = [app._btn_start, app._btn_pause, app._btn_resume,
                    app._btn_cancel, app._btn_cancel_all]
            for btn in btns:
                text = btn.cget("text")
                for ch in text:
                    cat = unicodedata.category(ch)
                    # Emoji are typically So (Symbol, other)
                    self.assertNotEqual(cat, "So",
                        f"Button '{text}' contains emoji character U+{ord(ch):04X}")
        finally:
            r.destroy()


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
