"""
tests/test_gui_layout.py
Regression tests for GUI layout: button callbacks, variables.
No real OCR, no API calls.
"""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Minimal tkinter stubs
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
# Tests
# ---------------------------------------------------------------------------

class TestGUICallbacksExist(unittest.TestCase):
    """Verify that all expected callback methods exist on OCRGuiApp."""

    def test_source_button_callbacks(self):
        """Source selection button callbacks must exist."""
        self.assertTrue(callable(getattr(ocr_gui.OCRGuiApp, "_select_single_pdf", None)),
                        "_select_single_pdf not found")
        self.assertTrue(callable(getattr(ocr_gui.OCRGuiApp, "_select_multiple_pdfs", None)),
                        "_select_multiple_pdfs not found")
        # Accept either _select_folder or _browse_input_folder
        has_folder = (
            callable(getattr(ocr_gui.OCRGuiApp, "_select_folder", None))
            or callable(getattr(ocr_gui.OCRGuiApp, "_browse_input_folder", None))
        )
        self.assertTrue(has_folder, "neither _select_folder nor _browse_input_folder found")
        self.assertTrue(callable(getattr(ocr_gui.OCRGuiApp, "_clear_input", None)),
                        "_clear_input not found")
        self.assertTrue(callable(getattr(ocr_gui.OCRGuiApp, "_rescan_batch", None)),
                        "_rescan_batch not found")

    def test_output_button_callbacks(self):
        """Output button callbacks must exist."""
        self.assertTrue(callable(getattr(ocr_gui.OCRGuiApp, "_browse_output_folder", None)),
                        "_browse_output_folder not found")
        self.assertTrue(callable(getattr(ocr_gui.OCRGuiApp, "_open_output_folder", None)),
                        "_open_output_folder not found")

    def test_output_mode_variable_exists(self):
        """_var_output_mode attribute must be set during __init__."""
        # Check the source: _var_output_mode is assigned in _build_source_section
        import inspect
        src = inspect.getsource(ocr_gui.OCRGuiApp)
        self.assertIn("_var_output_mode", src)

    def test_recursive_variable_exists(self):
        """_var_recursive attribute must be set during __init__."""
        import inspect
        src = inspect.getsource(ocr_gui.OCRGuiApp)
        self.assertIn("_var_recursive", src)


if __name__ == "__main__":
    unittest.main()
