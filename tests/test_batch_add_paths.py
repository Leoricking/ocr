"""
tests/test_batch_add_paths.py
Tests for folder scanning and _add_pdf_paths_to_batch edge cases.
"""
import os
import sys
import types
import unittest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Reuse the same stub setup as test_multi_pdf_selection
# ---------------------------------------------------------------------------

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

for mod_name, attr_name in [
    ("tkinter.filedialog", "filedialog"),
    ("tkinter.messagebox", "messagebox"),
    ("tkinter.scrolledtext", "scrolledtext"),
]:
    m = types.ModuleType(mod_name)
    for fn in ("askopenfilename", "askopenfilenames", "askdirectory",
               "asksaveasfilename", "showinfo", "showerror", "showwarning"):
        setattr(m, fn, MagicMock(return_value=""))
    m.askyesno = MagicMock(return_value=False)
    sys.modules.setdefault(mod_name, m)
    setattr(tk_mod, attr_name, m)

st_m = types.ModuleType("tkinter.scrolledtext")
st_m.ScrolledText = MagicMock()
sys.modules["tkinter.scrolledtext"] = st_m

font_mod = types.ModuleType("tkinter.font")
font_mod.families = MagicMock(return_value=[])
_mf = MagicMock()
_mf.metrics = MagicMock(return_value=14)
_mf.cget = MagicMock(return_value=10)
font_mod.Font = MagicMock(return_value=_mf)
sys.modules.setdefault("tkinter.font", font_mod)
tk_mod.font = font_mod

# worker_controller imports cleanly — let it load naturally

fitz_mod = types.ModuleType("fitz")
class _FakeDoc:
    def __len__(self): return 5
    def close(self): pass
fitz_mod.open = MagicMock(return_value=_FakeDoc())
sys.modules["fitz"] = fitz_mod

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

sys.path.insert(0, str(Path(__file__).parent.parent))
import ocr_gui


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_app():
    app = object.__new__(ocr_gui.OCRGuiApp)
    app._batch_items = {}
    app._batch_order = []
    app._log_lines = []
    app._settings = {"last_input_directory": "", "exclude_ocr_output": True}
    app._log_file_handle = None

    var_out = MagicMock()
    var_out.get = MagicMock(return_value="")
    app._var_output = var_out

    var_in = MagicMock()
    var_in.get = MagicMock(return_value="")
    var_in.set = MagicMock()
    app._var_input = var_in

    var_rec = MagicMock()
    var_rec.get = MagicMock(return_value=False)
    app._var_recursive = var_rec

    tree = MagicMock()
    tree.insert = MagicMock()
    app._tree = tree

    def _log(msg):
        app._log_lines.append(msg)
    app._append_log = _log
    app._update_button_states = MagicMock()
    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFolderScan(unittest.TestCase):

    def test_folder_scan_non_recursive(self):
        """Non-recursive scan picks up only top-level PDFs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files
            top_pdf = os.path.join(tmpdir, "top.pdf")
            sub_dir = os.path.join(tmpdir, "sub")
            os.makedirs(sub_dir)
            sub_pdf = os.path.join(sub_dir, "nested.pdf")
            Path(top_pdf).write_bytes(b"%PDF-1")
            Path(sub_pdf).write_bytes(b"%PDF-1")

            app = _make_app()
            app._var_recursive.get = MagicMock(return_value=False)
            app._scan_folder(tmpdir)

        # Should have 1 file (only top-level)
        self.assertEqual(len(app._batch_order), 1)
        self.assertTrue(any("top.pdf" in p for p in app._batch_order))

    def test_folder_scan_recursive(self):
        """Recursive scan picks up all PDFs including sub-folders."""
        with tempfile.TemporaryDirectory() as tmpdir:
            top_pdf = os.path.join(tmpdir, "top.pdf")
            sub_dir = os.path.join(tmpdir, "sub")
            os.makedirs(sub_dir)
            sub_pdf = os.path.join(sub_dir, "nested.pdf")
            Path(top_pdf).write_bytes(b"%PDF-1")
            Path(sub_pdf).write_bytes(b"%PDF-1")

            app = _make_app()
            app._var_recursive.get = MagicMock(return_value=True)
            app._scan_folder(tmpdir)

        self.assertEqual(len(app._batch_order), 2)

    def test_new_files_dont_clear_existing_batch(self):
        """Adding new files to an existing batch preserves prior entries."""
        app = _make_app()
        app._add_pdf_paths_to_batch(["/tmp/existing.pdf"], source_kind="single")
        first_count = len(app._batch_order)

        app._add_pdf_paths_to_batch(["/tmp/new.pdf"], source_kind="single")
        self.assertGreater(len(app._batch_order), first_count)
        self.assertTrue(any("existing.pdf" in p for p in app._batch_order))
        self.assertTrue(any("new.pdf" in p for p in app._batch_order))

    def test_page_read_failure_does_not_abort_other_files(self):
        """If fitz.open fails for one file, others are still processed."""
        app = _make_app()

        call_count = [0]
        def _fitz_open(path):
            call_count[0] += 1
            if "bad" in path:
                raise RuntimeError("corrupt PDF")
            doc = _FakeDoc()
            return doc

        import fitz as fitz_m
        orig = fitz_m.open
        fitz_m.open = _fitz_open
        try:
            paths = ["/tmp/good1.pdf", "/tmp/bad.pdf", "/tmp/good2.pdf"]
            app._add_pdf_paths_to_batch(paths, source_kind="multiple")
        finally:
            fitz_m.open = orig

        # All three paths should have been attempted; bad one gets "?" pages
        self.assertEqual(len(app._batch_order), 3)
        bad_norm = str(Path("/tmp/bad.pdf").resolve())
        self.assertEqual(app._batch_items[bad_norm]["pages"], "?")

    def test_last_input_directory_saved(self):
        """_select_single_pdf saves last_input_directory in settings."""
        app = _make_app()
        app._add_pdf_paths_to_batch = MagicMock()

        test_path = "/tmp/myfolder/document.pdf"
        with patch("ocr_gui.filedialog.askopenfilename", return_value=test_path):
            app._select_single_pdf()

        self.assertEqual(
            app._settings.get("last_input_directory"),
            str(Path(test_path).parent),
        )

    def test_last_input_directory_saved_multi(self):
        """_select_multiple_pdfs saves last_input_directory."""
        app = _make_app()
        app._add_pdf_paths_to_batch = MagicMock()

        paths = ("/tmp/myfolder/a.pdf", "/tmp/myfolder/b.pdf")
        with patch("ocr_gui.filedialog.askopenfilenames", return_value=paths):
            app._select_multiple_pdfs()

        self.assertEqual(
            app._settings.get("last_input_directory"),
            str(Path(paths[0]).parent),
        )

    def test_ocr_output_excluded_in_folder_scan(self):
        """Folder scan excludes _OCR.pdf files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            good = os.path.join(tmpdir, "source.pdf")
            bad = os.path.join(tmpdir, "source_OCR.pdf")
            Path(good).write_bytes(b"%PDF-1")
            Path(bad).write_bytes(b"%PDF-1")

            app = _make_app()
            app._scan_folder(tmpdir)

        self.assertEqual(len(app._batch_order), 1)
        self.assertTrue(any("source.pdf" in p and "_OCR" not in p
                            for p in app._batch_order))


if __name__ == "__main__":
    unittest.main()
