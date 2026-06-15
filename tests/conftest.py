"""
conftest.py — Pytest configuration for the OCR test suite.

Our new GUI tests stub tkinter/settings_manager at module level using
sys.modules.setdefault.  conftest.py is imported FIRST by pytest, so we
pre-load the real modules here.  Because the new tests use setdefault (not
direct assignment), the real modules loaded here remain in place and the
stubs are silently ignored.
"""
import sys
import importlib

# Pre-load these real modules before any test file installs stubs.
# Using setdefault in test files means a pre-existing entry is preserved.
_PRELOAD = [
    "tkinter",
    "tkinter.ttk",
    "tkinter.font",
    "tkinter.filedialog",
    "tkinter.messagebox",
    "tkinter.scrolledtext",
    "worker_controller",
    "settings_manager",
]

for _mod_name in _PRELOAD:
    if _mod_name not in sys.modules:
        try:
            importlib.import_module(_mod_name)
        except ImportError:
            pass  # Not available — let individual tests handle it
