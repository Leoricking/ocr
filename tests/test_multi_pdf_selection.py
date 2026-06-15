"""
tests/test_multi_pdf_selection.py
Tests for multi-PDF selection modes (Part A — _select_single_pdf,
_select_multiple_pdfs, _add_pdf_paths_to_batch).
"""
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# ---------------------------------------------------------------------------
# Minimal stubs so we can import ocr_gui without real tkinter/paddle
# ---------------------------------------------------------------------------

# Stub tkinter
tk_mod = types.ModuleType("tkinter")
tk_mod.Tk = MagicMock
tk_mod.StringVar = MagicMock
tk_mod.BooleanVar = MagicMock
tk_mod.DoubleVar = MagicMock
tk_mod.Canvas = MagicMock
tk_mod.Menu = MagicMock
tk_mod.LEFT = "left"
tk_mod.RIGHT = "right"
tk_mod.HORIZONTAL = "horizontal"
tk_mod.VERTICAL = "vertical"
sys.modules.setdefault("tkinter", tk_mod)

ttk_mod = types.ModuleType("tkinter.ttk")
for _cls in ("Frame", "LabelFrame", "Label", "Button", "Entry", "Combobox",
             "Checkbutton", "Radiobutton", "Scrollbar", "Scale",
             "Treeview", "Progressbar", "PanedWindow", "Style"):
    setattr(ttk_mod, _cls, MagicMock)
sys.modules.setdefault("tkinter.ttk", ttk_mod)
tk_mod.ttk = ttk_mod

fd_mod = types.ModuleType("tkinter.filedialog")
fd_mod.askopenfilename = MagicMock(return_value="")
fd_mod.askopenfilenames = MagicMock(return_value=())
fd_mod.askdirectory = MagicMock(return_value="")
fd_mod.asksaveasfilename = MagicMock(return_value="")
sys.modules.setdefault("tkinter.filedialog", fd_mod)
tk_mod.filedialog = fd_mod

mb_mod = types.ModuleType("tkinter.messagebox")
mb_mod.showinfo = MagicMock()
mb_mod.showerror = MagicMock()
mb_mod.showwarning = MagicMock()
mb_mod.askyesno = MagicMock(return_value=False)
sys.modules.setdefault("tkinter.messagebox", mb_mod)
tk_mod.messagebox = mb_mod

st_mod = types.ModuleType("tkinter.scrolledtext")
st_mod.ScrolledText = MagicMock()
sys.modules.setdefault("tkinter.scrolledtext", st_mod)
tk_mod.scrolledtext = st_mod

font_mod = types.ModuleType("tkinter.font")
font_mod.families = MagicMock(return_value=[])
mock_font = MagicMock()
mock_font.metrics = MagicMock(return_value=14)
mock_font.cget = MagicMock(return_value=10)
font_mod.Font = MagicMock(return_value=mock_font)
sys.modules.setdefault("tkinter.font", font_mod)
tk_mod.font = font_mod

# worker_controller imports cleanly without heavy deps — let it load naturally
# (do not stub it; test_worker_controller.py needs the real one)

# Stub fitz
fitz_mod = types.ModuleType("fitz")
class _FakeDoc:
    def __len__(self): return 3
    def close(self): pass
fitz_mod.open = MagicMock(return_value=_FakeDoc())
sys.modules["fitz"] = fitz_mod

# Stub settings_manager
sm_mod = types.ModuleType("settings_manager")
class _FakeSM:
    def load(self): return {}
    def save(self, s): pass
    def reset(self): return {}
sm_mod.SettingsManager = _FakeSM
sm_mod.DEFAULTS = {
    "version": 1, "last_input_path": "", "last_output_path": "",
    "last_input_directory": "", "output_mode": "all",
}
sys.modules.setdefault("settings_manager", sm_mod)

# Now import
sys.path.insert(0, str(Path(__file__).parent.parent))
import ocr_gui


# ---------------------------------------------------------------------------
# Helper: build a lightweight OCRGuiApp stub (no real Tk)
# ---------------------------------------------------------------------------

def _make_app():
    """Return a minimal OCRGuiApp-like object with just the batch structures."""
    app = object.__new__(ocr_gui.OCRGuiApp)
    app._batch_items = {}
    app._batch_order = []
    app._log_lines = []
    app._settings = {"last_input_directory": "", "exclude_ocr_output": True}
    app._log_file_handle = None

    # Mock _var_output
    var_out = MagicMock()
    var_out.get = MagicMock(return_value="")
    app._var_output = var_out

    # Mock tree
    tree = MagicMock()
    tree.insert = MagicMock()
    app._tree = tree

    # Patch _append_log to real list append (no tkinter)
    def _log(msg):
        app._log_lines.append(msg)
    app._append_log = _log
    app._update_button_states = MagicMock()

    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAddPdfPathsToBatch(unittest.TestCase):

    def setUp(self):
        self.app = _make_app()

    def test_single_pdf_added(self):
        """A valid PDF is added to the batch."""
        path = "/tmp/document.pdf"
        self.app._add_pdf_paths_to_batch([path], source_kind="single")
        self.assertIn(str(Path(path).resolve()), self.app._batch_order)

    def test_multiple_pdfs_added(self):
        """Three valid PDFs are all added."""
        paths = ["/tmp/a.pdf", "/tmp/b.pdf", "/tmp/c.pdf"]
        self.app._add_pdf_paths_to_batch(paths, source_kind="multiple")
        self.assertEqual(len(self.app._batch_order), 3)

    def test_cancel_empty_tuple_does_not_change_list(self):
        """Empty tuple (user cancelled) leaves batch unchanged."""
        self.app._add_pdf_paths_to_batch([], source_kind="multiple")
        self.assertEqual(len(self.app._batch_order), 0)

    def test_askopenfilenames_tuple_handled(self):
        """Tuple input (from askopenfilenames) is accepted."""
        paths = ("/tmp/x.pdf", "/tmp/y.pdf")
        self.app._add_pdf_paths_to_batch(list(paths), source_kind="multiple")
        self.assertEqual(len(self.app._batch_order), 2)

    def test_same_absolute_path_not_added_twice(self):
        """Adding the same path twice only inserts one entry."""
        path = "/tmp/dup.pdf"
        self.app._add_pdf_paths_to_batch([path], source_kind="single")
        self.app._add_pdf_paths_to_batch([path], source_kind="single")
        self.assertEqual(len(self.app._batch_order), 1)

    def test_same_filename_different_folders_both_added(self):
        """Same filename in different directories → both accepted."""
        p1 = "/tmp/folder1/report.pdf"
        p2 = "/tmp/folder2/report.pdf"
        self.app._add_pdf_paths_to_batch([p1, p2], source_kind="multiple")
        self.assertEqual(len(self.app._batch_order), 2)

    def test_ocr_pdf_excluded(self):
        """Files ending in _OCR.pdf are filtered out."""
        path = "/tmp/document_OCR.pdf"
        self.app._add_pdf_paths_to_batch([path], source_kind="single")
        self.assertEqual(len(self.app._batch_order), 0)

    def test_part_excluded(self):
        """Files ending in .part are filtered out."""
        path = "/tmp/document.part"
        self.app._add_pdf_paths_to_batch([path], source_kind="single")
        self.assertEqual(len(self.app._batch_order), 0)

    def test_output_folder_path_excluded(self):
        """Files inside the current output folder are excluded."""
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            self.app._var_output.get = MagicMock(return_value=tmpdir)
            test_pdf = os.path.join(tmpdir, "inside.pdf")
            # Normalize to casefold for comparison
            self.app._add_pdf_paths_to_batch([test_pdf], source_kind="single")
            self.assertEqual(len(self.app._batch_order), 0)

    def test_non_pdf_excluded(self):
        """Non-.pdf files are filtered out."""
        path = "/tmp/document.docx"
        self.app._add_pdf_paths_to_batch([path], source_kind="single")
        self.assertEqual(len(self.app._batch_order), 0)

    def test_ocr_ocr_pdf_excluded(self):
        """Files ending in _OCR_OCR.pdf are filtered out."""
        path = "/tmp/doc_OCR_OCR.pdf"
        self.app._add_pdf_paths_to_batch([path], source_kind="single")
        self.assertEqual(len(self.app._batch_order), 0)

    def test_searchable_pdf_excluded(self):
        """Files ending in _searchable.pdf are filtered out."""
        path = "/tmp/doc_searchable.pdf"
        self.app._add_pdf_paths_to_batch([path], source_kind="single")
        self.assertEqual(len(self.app._batch_order), 0)

    def test_dedup_within_same_selection(self):
        """Duplicate paths in the same call are only added once."""
        path = "/tmp/once.pdf"
        self.app._add_pdf_paths_to_batch([path, path], source_kind="multiple")
        self.assertEqual(len(self.app._batch_order), 1)

    def test_item_has_elapsed_seconds_field(self):
        """Batch item dict contains elapsed_seconds initialized to 0."""
        path = "/tmp/timing.pdf"
        self.app._add_pdf_paths_to_batch([path], source_kind="single")
        norm = str(Path(path).resolve())
        item = self.app._batch_items.get(norm)
        self.assertIsNotNone(item)
        self.assertEqual(item.get("elapsed_seconds"), 0.0)


class TestSelectMultiplePdfs(unittest.TestCase):
    """Test _select_multiple_pdfs behavior."""

    def setUp(self):
        self.app = _make_app()
        # Mock _add_pdf_paths_to_batch and _var_input
        self.app._add_pdf_paths_to_batch = MagicMock()
        var_input = MagicMock()
        var_input.set = MagicMock()
        self.app._var_input = var_input

    def test_cancel_no_change(self):
        """Empty tuple from dialog → _add_pdf_paths_to_batch not called."""
        with patch("ocr_gui.filedialog.askopenfilenames", return_value=()):
            self.app._select_multiple_pdfs()
        self.app._add_pdf_paths_to_batch.assert_not_called()

    def test_paths_forwarded(self):
        """Paths returned by dialog are forwarded to batch adder."""
        paths = ("/tmp/a.pdf", "/tmp/b.pdf", "/tmp/c.pdf")
        with patch("ocr_gui.filedialog.askopenfilenames", return_value=paths):
            self.app._select_multiple_pdfs()
        self.app._add_pdf_paths_to_batch.assert_called_once()
        call_paths = self.app._add_pdf_paths_to_batch.call_args[0][0]
        self.assertEqual(len(call_paths), 3)

    def test_input_label_shows_count(self):
        """_var_input is set to show the count of selected files."""
        paths = ("/tmp/a.pdf", "/tmp/b.pdf")
        with patch("ocr_gui.filedialog.askopenfilenames", return_value=paths):
            self.app._select_multiple_pdfs()
        set_calls = self.app._var_input.set.call_args_list
        self.assertTrue(any("2" in str(c) for c in set_calls))


if __name__ == "__main__":
    unittest.main()
