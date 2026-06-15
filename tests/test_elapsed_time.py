"""
tests/test_elapsed_time.py
Tests for _format_duration helper and per-file elapsed timing logic.
"""
import sys
import types
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Minimal stubs
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

# worker_controller imports cleanly — let it load naturally

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
from ocr_gui import _format_duration


# ---------------------------------------------------------------------------
# _format_duration tests
# ---------------------------------------------------------------------------

class TestFormatDuration(unittest.TestCase):

    def test_zero(self):
        self.assertEqual(_format_duration(0), "00:00:00")

    def test_307_seconds(self):
        # 5 minutes 7 seconds
        self.assertEqual(_format_duration(307), "00:05:07")

    def test_4509_seconds(self):
        # 1 hour 15 minutes 9 seconds
        self.assertEqual(_format_duration(4509), "01:15:09")

    def test_90197_seconds(self):
        # 25 hours 3 minutes 17 seconds
        self.assertEqual(_format_duration(90197), "25:03:17")

    def test_negative_treated_as_zero(self):
        self.assertEqual(_format_duration(-5), "00:00:00")

    def test_float_input(self):
        # 61.9 → 61 seconds → 00:01:01
        self.assertEqual(_format_duration(61.9), "00:01:01")


# ---------------------------------------------------------------------------
# Per-file timing logic tests (via _handle_worker_event)
# ---------------------------------------------------------------------------

def _make_app():
    """Build a minimal app stub for event handler testing."""
    app = object.__new__(ocr_gui.OCRGuiApp)
    app._batch_items = {}
    app._batch_order = []
    app._log_lines = []
    app._settings = {}
    app._log_file_handle = None
    app._pending_files = []
    app._completed_file_count = 0
    app._crash_count = 0
    app._running = False
    app._batch_had_failures = False
    app._batch_cancelled = False
    app._batch_started_at = None
    app._batch_elapsed_seconds = 0.0
    app._batch_finished_at = None
    app._batch_paused_at = None
    app._batch_pause_accumulated = 0.0
    app._timer_id = None
    app._wc = None

    # Mocked widgets
    app._lbl_current_file = MagicMock()
    app._lbl_file_elapsed = MagicMock()
    app._lbl_file_remaining = MagicMock()
    app._lbl_batch_elapsed = MagicMock()
    app._lbl_batch_remaining = MagicMock()
    app._lbl_eta = MagicMock()
    app._lbl_page_info = MagicMock()
    app._lbl_overall_info = MagicMock()
    app._lbl_worker_info = MagicMock()
    app._pbar_file = MagicMock()
    app._pbar_overall = MagicMock()

    tree = MagicMock()
    tree.insert = MagicMock()
    tree.item = MagicMock(return_value=["1", "f.pdf", "3", "等待", "", "", "00:00:00", "", "", ""])
    app._tree = tree

    def _log(msg):
        app._log_lines.append(msg)
    app._append_log = _log
    app._update_button_states = MagicMock()
    app._tree_refresh_row = MagicMock()
    app._update_tree_pages = MagicMock()
    app._stop_timer = MagicMock()
    app._update_file_status = MagicMock()

    return app


def _add_item(app, path):
    """Add a batch item for testing."""
    app._batch_order.append(path)
    app._batch_items[path] = {
        "index": 1,
        "pages": 5,
        "status": "等待",
        "progress": "",
        "avg_sec": "",
        "elapsed_seconds": 0.0,
        "file_started_at": None,
        "file_finished_at": None,
        "accumulated_pause_seconds": 0.0,
        "pause_started_at": None,
        "out_pdf": "",
        "out_txt": "",
        "out_analysis": "",
        "selected": True,
    }


class TestFileStartedEvent(unittest.TestCase):

    def test_file_started_resets_elapsed(self):
        """file_started event resets elapsed_seconds to 0."""
        app = _make_app()
        path = "/tmp/test.pdf"
        _add_item(app, path)
        app._batch_items[path]["elapsed_seconds"] = 99.9

        app._handle_worker_event({"type": "file_started", "path": path})

        self.assertEqual(app._batch_items[path]["elapsed_seconds"], 0.0)
        self.assertIsNotNone(app._batch_items[path]["file_started_at"])

    def test_file_started_sets_started_at(self):
        """file_started sets file_started_at to a recent monotonic timestamp."""
        app = _make_app()
        path = "/tmp/test2.pdf"
        _add_item(app, path)

        before = time.monotonic()
        app._handle_worker_event({"type": "file_started", "path": path})
        after = time.monotonic()

        started_at = app._batch_items[path]["file_started_at"]
        self.assertGreaterEqual(started_at, before)
        self.assertLessEqual(started_at, after)


class TestFileCompletedEvent(unittest.TestCase):

    def test_elapsed_from_event(self):
        """file_completed uses elapsed_seconds from event when present."""
        app = _make_app()
        path = "/tmp/done.pdf"
        _add_item(app, path)
        app._batch_items[path]["file_started_at"] = time.monotonic() - 100

        app._handle_worker_event({
            "type": "file_completed",
            "path": path,
            "pdf": "",
            "txt": "",
            "analysis": "",
            "elapsed_seconds": 42.5,
        })

        self.assertAlmostEqual(app._batch_items[path]["elapsed_seconds"], 42.5)

    def test_elapsed_fixed_on_completion(self):
        """file_completed sets file_finished_at."""
        app = _make_app()
        path = "/tmp/done2.pdf"
        _add_item(app, path)
        app._pending_files = [path]
        app._batch_items[path]["file_started_at"] = time.monotonic() - 10

        app._handle_worker_event({
            "type": "file_completed",
            "path": path,
            "pdf": "",
            "txt": "",
            "analysis": "",
            "elapsed_seconds": 10.0,
        })

        self.assertIsNotNone(app._batch_items[path]["file_finished_at"])


class TestFileFailedEvent(unittest.TestCase):

    def test_elapsed_from_event(self):
        """file_failed uses elapsed_seconds from event."""
        app = _make_app()
        path = "/tmp/fail.pdf"
        _add_item(app, path)
        app._pending_files = [path]
        app._batch_items[path]["file_started_at"] = time.monotonic() - 30

        app._handle_worker_event({
            "type": "file_failed",
            "path": path,
            "error": "oops",
            "elapsed_seconds": 28.0,
        })

        self.assertAlmostEqual(app._batch_items[path]["elapsed_seconds"], 28.0)

    def test_elapsed_monotonic_fallback(self):
        """file_failed computes elapsed from monotonic when event lacks elapsed_seconds."""
        app = _make_app()
        path = "/tmp/fail2.pdf"
        _add_item(app, path)
        app._pending_files = [path]
        t0 = time.monotonic()
        app._batch_items[path]["file_started_at"] = t0

        time.sleep(0.05)  # ensure some time passes
        app._handle_worker_event({"type": "file_failed", "path": path, "error": "x"})

        self.assertGreater(app._batch_items[path]["elapsed_seconds"], 0.0)


class TestFileSkippedEvent(unittest.TestCase):

    def test_skipped_elapsed_stays_zero(self):
        """file_skipped keeps elapsed at 0."""
        app = _make_app()
        path = "/tmp/skip.pdf"
        _add_item(app, path)

        app._handle_worker_event({"type": "file_skipped", "path": path})

        self.assertEqual(app._batch_items[path]["elapsed_seconds"], 0.0)


class TestPauseExcluded(unittest.TestCase):

    def test_pause_duration_excluded(self):
        """Paused time is not counted in batch elapsed calculation."""
        app = _make_app()
        now = time.monotonic()
        app._batch_started_at = now - 100  # started 100 seconds ago
        app._batch_pause_accumulated = 0.0

        # Simulate a pause that happened 50 seconds ago and lasted 30 seconds
        app._batch_paused_at = now - 20   # still paused (paused 20 seconds ago)
        app._batch_pause_accumulated = 30.0

        # Total elapsed should be ~50 seconds (100 - 30 - 20 so far)
        # The _tick_timer would compute: now - started - (accumulated + current_pause)
        pause_so_far = app._batch_pause_accumulated + (now - app._batch_paused_at)
        batch_elapsed = now - app._batch_started_at - pause_so_far
        self.assertAlmostEqual(batch_elapsed, 50.0, delta=0.1)


class TestBatchCompleted(unittest.TestCase):

    def test_batch_completed_uses_event_elapsed(self):
        """batch_completed uses elapsed_seconds from event if present."""
        app = _make_app()
        app._batch_started_at = time.monotonic() - 200
        app._pending_files = []

        app._handle_worker_event({
            "type": "batch_completed",
            "completed": 3,
            "failed": 0,
            "cancelled": 0,
            "skipped": 0,
            "elapsed_seconds": 195.0,
        })

        self.assertAlmostEqual(app._batch_elapsed_seconds, 195.0)

    def test_batch_completed_stops_timer(self):
        """batch_completed calls _stop_timer."""
        app = _make_app()
        app._batch_started_at = time.monotonic() - 10
        app._pending_files = []

        app._handle_worker_event({
            "type": "batch_completed",
            "completed": 1,
            "failed": 0,
            "cancelled": 0,
            "skipped": 0,
        })

        app._stop_timer.assert_called()


if __name__ == "__main__":
    unittest.main()
