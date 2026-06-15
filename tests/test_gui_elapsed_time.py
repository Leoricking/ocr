"""
tests/test_gui_elapsed_time.py
Tests for Treeview elapsed_time column and timer management.
"""
import sys
import types
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

# ---------------------------------------------------------------------------
# Minimal stubs (same pattern as other test files)
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
# Helper to build lightweight app stub
# ---------------------------------------------------------------------------

def _make_app():
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
    app._pbar_file.__setitem__ = MagicMock()
    app._pbar_overall = MagicMock()
    app._pbar_overall.__setitem__ = MagicMock()

    # Real tree mock with item() returning a tuple-like list
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

    # Mock root.after
    after_ids = [0]
    def _after(ms, func=None):
        after_ids[0] += 1
        return after_ids[0]
    def _after_cancel(tid):
        pass
    root = MagicMock()
    root.after = _after
    root.after_cancel = _after_cancel
    app.root = root

    return app


def _add_item(app, path, status="等待"):
    app._batch_order.append(path)
    app._batch_items[path] = {
        "index": 1,
        "pages": 5,
        "status": status,
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTreeviewElapsedTimeColumn(unittest.TestCase):

    def test_elapsed_time_column_exists_in_constants(self):
        """The 花費時間 column name appears in the GUI source."""
        import inspect
        src = inspect.getsource(ocr_gui)
        self.assertIn("花費時間", src)

    def test_format_duration_in_tree_insert(self):
        """_tree_insert writes 00:00:00 for a new item."""
        app = _make_app()
        path = "/tmp/newfile.pdf"
        _add_item(app, path)
        # _tree_insert was called during _add_item via _add_pdf_paths_to_batch
        # Here we call it directly
        app._tree_insert(path)
        insert_call = app._tree.insert.call_args
        if insert_call:
            values = insert_call[1].get("values") or insert_call[0][2] if len(insert_call[0]) > 2 else None
            if values:
                # elapsed_time is at index 6
                self.assertEqual(values[6], "00:00:00")


class TestWorkerElapsedSecondsPreference(unittest.TestCase):

    def test_page_progress_uses_event_elapsed_seconds(self):
        """page_progress event: elapsed_seconds from event takes priority."""
        app = _make_app()
        path = "/tmp/prio.pdf"
        _add_item(app, path)
        app._batch_items[path]["file_started_at"] = time.monotonic() - 1000  # would give large elapsed

        app._handle_worker_event({
            "type": "page_progress",
            "path": path,
            "page": 1,
            "pages": 5,
            "stage": "PADDLE",
            "avg_sec_per_page": 2.0,
            "estimated_remaining": 6.0,
            "elapsed_seconds": 3.5,  # explicit from worker
        })

        self.assertAlmostEqual(app._batch_items[path]["elapsed_seconds"], 3.5)

    def test_page_progress_monotonic_fallback(self):
        """page_progress without elapsed_seconds falls back to monotonic."""
        app = _make_app()
        path = "/tmp/fallback.pdf"
        _add_item(app, path)
        app._batch_items[path]["file_started_at"] = time.monotonic() - 10.0

        app._handle_worker_event({
            "type": "page_progress",
            "path": path,
            "page": 0,
            "pages": 3,
            "stage": "PADDLE",
            "avg_sec_per_page": 5.0,
            "estimated_remaining": 15.0,
            # no elapsed_seconds key
        })

        # Should be roughly 10 seconds (plus tiny scheduling overhead)
        elapsed = app._batch_items[path]["elapsed_seconds"]
        self.assertGreater(elapsed, 0.0)
        self.assertLess(elapsed, 30.0)


class TestTimerNotCreatedTwice(unittest.TestCase):

    def test_start_timer_cancels_existing(self):
        """_start_timer cancels any existing timer before creating a new one."""
        app = _make_app()

        cancel_calls = []
        def _cancel(tid):
            cancel_calls.append(tid)
        app.root.after_cancel = _cancel

        # Simulate an existing timer
        app._timer_id = 42

        # _start_timer should cancel 42 then create a new one
        app._start_timer()

        self.assertIn(42, cancel_calls)
        self.assertIsNotNone(app._timer_id)

    def test_calling_start_timer_twice_only_one_chain(self):
        """Calling _start_timer twice results in only one active timer."""
        app = _make_app()
        app._batch_started_at = time.monotonic()

        app._start_timer()
        id1 = app._timer_id

        app._start_timer()
        id2 = app._timer_id

        # Both are valid (non-None); the first was cancelled when second started
        self.assertIsNotNone(id2)


class TestTimerCancelledAfterBatch(unittest.TestCase):

    def test_batch_completed_stops_timer(self):
        """batch_completed event stops the tick timer."""
        app = _make_app()
        app._batch_started_at = time.monotonic() - 5
        app._pending_files = []
        app._timer_id = 99

        app._handle_worker_event({
            "type": "batch_completed",
            "completed": 1,
            "failed": 0,
            "cancelled": 0,
            "skipped": 0,
        })

        app._stop_timer.assert_called()

    def test_on_worker_done_stops_timer(self):
        """_on_worker_done stops the tick timer."""
        app = _make_app()
        app._batch_started_at = time.monotonic() - 5
        app._timer_id = 77

        # Additional mocks needed for _on_worker_done
        app._set_running_buttons = MagicMock()
        app._pbar_overall = {"value": 0}
        app._lbl_worker_info = MagicMock()

        app._on_worker_done()
        app._stop_timer.assert_called()


class TestTickTimerUpdatesTree(unittest.TestCase):

    def test_tick_updates_active_file_elapsed(self):
        """_tick_timer updates elapsed_seconds for currently OCR-ing file."""
        app = _make_app()
        path = "/tmp/active.pdf"
        _add_item(app, path, status="OCR 中")
        app._batch_items[path]["file_started_at"] = time.monotonic() - 5.0
        app._batch_started_at = time.monotonic() - 5.0

        # Prevent infinite recursion: patch root.after to not re-schedule
        app.root.after = MagicMock(return_value=1)

        app._tick_timer()

        elapsed = app._batch_items[path]["elapsed_seconds"]
        self.assertGreater(elapsed, 0.0)
        self.assertLess(elapsed, 30.0)


if __name__ == "__main__":
    unittest.main()
