"""
tests/test_treeview_alignment.py
Verify Treeview column anchors, widths, and stretch flags for the batch list.
Inspects source code rather than executing headless tkinter.
"""
import sys
import inspect
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Minimal tkinter stubs so ocr_gui imports cleanly without a display
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

for mod_name, attr in [("tkinter.filedialog", "filedialog"),
                        ("tkinter.messagebox", "messagebox"),
                        ("tkinter.scrolledtext", "scrolledtext")]:
    m = types.ModuleType(mod_name)
    for fn in ("askopenfilename", "askopenfilenames", "askdirectory",
               "asksaveasfilename", "showinfo", "showerror", "showwarning"):
        setattr(m, fn, MagicMock(return_value=""))
    m.askyesno = MagicMock(return_value=False)
    m.ScrolledText = MagicMock()
    sys.modules.setdefault(mod_name, m)
    setattr(tk_mod, attr, m)

font_mod = types.ModuleType("tkinter.font")
font_mod.families = MagicMock(return_value=[])
_mf = MagicMock()
_mf.metrics = MagicMock(return_value=14)
_mf.cget = MagicMock(return_value=10)
font_mod.Font = MagicMock(return_value=_mf)
sys.modules.setdefault("tkinter.font", font_mod)
tk_mod.font = font_mod

fitz_mod = types.ModuleType("fitz")
class _FakeDoc:
    def __len__(self): return 3
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
# Expected spec
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = (
    "index", "filename", "pages", "status", "progress",
    "avg_sec", "elapsed_time", "output_pdf", "output_txt", "analysis",
)

# (heading_anchor, column_anchor)
EXPECTED_ANCHORS = {
    "index":        ("center", "center"),
    "filename":     ("center", "w"),
    "pages":        ("center", "center"),
    "status":       ("center", "center"),
    "progress":     ("center", "center"),
    "avg_sec":      ("center", "center"),
    "elapsed_time": ("center", "center"),
    "output_pdf":   ("center", "w"),
    "output_txt":   ("center", "w"),
    "analysis":     ("center", "w"),
}

STRETCH_TRUE = {"filename", "output_pdf", "output_txt", "analysis"}


# ---------------------------------------------------------------------------
# Source-code based checks
# ---------------------------------------------------------------------------

def _get_col_specs(source: str) -> dict:
    """
    Parse col_specs dict literal from the source of _build_batch_panel.
    Returns a dict of col -> (h_anchor, c_anchor, width, stretch).
    """
    # We exec the col_specs block from source to get the actual values
    # Find the col_specs assignment
    start = source.find("col_specs = {")
    if start == -1:
        return {}
    end = source.find("}", start)
    # find the closing brace that ends the dict (possibly nested tuples)
    depth = 0
    i = start + len("col_specs = ")
    while i < len(source):
        ch = source[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
        i += 1
    block = source[start:end]
    local_ns = {}
    try:
        exec(block, {}, local_ns)
    except Exception:
        return {}
    return local_ns.get("col_specs", {})


class TestTreeviewColumnExistence(unittest.TestCase):

    def setUp(self):
        self.src = inspect.getsource(ocr_gui.OCRGuiApp._build_batch_panel)

    def test_all_ten_columns_defined(self):
        """All 10 English column keys must appear in _build_batch_panel."""
        for col in EXPECTED_COLUMNS:
            self.assertIn(f'"{col}"', self.src,
                          f'Column key "{col}" not found in _build_batch_panel source')

    def test_column_tuple_contains_all_keys(self):
        """The cols tuple must contain all 10 column keys."""
        for col in EXPECTED_COLUMNS:
            self.assertIn(col, self.src,
                          f'Column "{col}" missing from source')


class TestTreeviewAnchors(unittest.TestCase):

    def setUp(self):
        self.src = inspect.getsource(ocr_gui.OCRGuiApp._build_batch_panel)
        self.col_specs = _get_col_specs(self.src)

    def test_col_specs_parsed(self):
        """col_specs dict must be parseable and contain all 10 columns."""
        self.assertTrue(len(self.col_specs) >= 10,
                        f"col_specs only has {len(self.col_specs)} entries: {list(self.col_specs)}")

    def test_heading_anchors(self):
        """Every column heading must have anchor='center'."""
        for col in EXPECTED_COLUMNS:
            with self.subTest(col=col):
                self.assertIn(col, self.col_specs,
                              f'col_specs missing column "{col}"')
                h_anchor = self.col_specs[col][0]
                self.assertEqual(h_anchor, "center",
                                 f'Column "{col}" heading anchor: expected "center", got "{h_anchor}"')

    def test_column_anchors(self):
        """Columns must have the correct cell anchor (center or w)."""
        for col in EXPECTED_COLUMNS:
            with self.subTest(col=col):
                self.assertIn(col, self.col_specs,
                              f'col_specs missing column "{col}"')
                c_anchor = self.col_specs[col][1]
                expected = EXPECTED_ANCHORS[col][1]
                self.assertEqual(c_anchor, expected,
                                 f'Column "{col}" cell anchor: expected "{expected}", got "{c_anchor}"')

    def test_stretch_flags(self):
        """filename, output_pdf, output_txt, analysis must have stretch=True; others False."""
        for col in EXPECTED_COLUMNS:
            with self.subTest(col=col):
                self.assertIn(col, self.col_specs,
                              f'col_specs missing column "{col}"')
                stretch = self.col_specs[col][3]
                if col in STRETCH_TRUE:
                    self.assertTrue(stretch,
                                    f'Column "{col}" should have stretch=True, got {stretch}')
                else:
                    self.assertFalse(stretch,
                                     f'Column "{col}" should have stretch=False, got {stretch}')

    def test_column_widths(self):
        """Column base widths must match the spec."""
        expected_widths = {
            "index": 50, "filename": 270, "pages": 70,
            "status": 90, "progress": 90, "avg_sec": 100,
            "elapsed_time": 100, "output_pdf": 180,
            "output_txt": 180, "analysis": 180,
        }
        for col, exp_w in expected_widths.items():
            with self.subTest(col=col):
                self.assertIn(col, self.col_specs,
                              f'col_specs missing column "{col}"')
                actual_w = self.col_specs[col][2]
                self.assertEqual(actual_w, exp_w,
                                 f'Column "{col}" width: expected {exp_w}, got {actual_w}')


if __name__ == "__main__":
    unittest.main()
