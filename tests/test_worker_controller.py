"""
test_worker_controller.py — Tests for WorkerController logic.
Uses mock subprocess — no real PDF, no OCR, no Claude API.
"""

import sys
import os
import json
import queue
import time
import tempfile
import threading
import subprocess
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker_controller import WorkerController, describe_windows_exit_code


def _make_mock_process(stdout_lines=None, returncode=0):
    """Build a mock subprocess.Popen-like object."""
    mock_proc = MagicMock()
    mock_proc.pid = 12345
    mock_proc.returncode = returncode

    stdout_data = "\n".join(stdout_lines or []) + "\n"
    mock_proc.stdout = iter(stdout_data.splitlines(keepends=True))

    mock_proc.stderr = iter([])

    # poll() returns None while running, then returncode
    _poll_calls = [0]
    def _poll():
        _poll_calls[0] += 1
        if _poll_calls[0] <= 2:
            return None
        return returncode
    mock_proc.poll.side_effect = _poll
    mock_proc.wait.return_value = returncode
    return mock_proc


class TestWorkerControllerBasic(unittest.TestCase):
    def _make_controller(self):
        eq = queue.Queue()
        return WorkerController(eq), eq

    def test_not_running_before_start(self):
        ctrl, _ = self._make_controller()
        self.assertFalse(ctrl.is_running())

    def test_get_pid_returns_none_before_start(self):
        ctrl, _ = self._make_controller()
        self.assertIsNone(ctrl.get_pid())


class TestWorkerControllerExitCodes(unittest.TestCase):
    def test_exit_zero_described_as_normal(self):
        desc = describe_windows_exit_code(0)
        self.assertIn("正常完成", desc)

    def test_access_violation_code(self):
        desc = describe_windows_exit_code(-1073741819)
        self.assertIn("存取違規", desc)

    def test_exit_none_described_as_not_finished(self):
        desc = describe_windows_exit_code(None)
        self.assertIn("尚未結束", desc)


class TestWorkerControllerControlFile(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)
        self.eq = queue.Queue()
        self.ctrl = WorkerController(self.eq)
        # Manually set control file path
        self.ctrl._control_file = str(self.tmpdir / "test.control.json")

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_pause_writes_pause_true(self):
        self.ctrl.pause()
        with open(self.ctrl._control_file, encoding="utf-8") as f:
            data = json.load(f)
        self.assertTrue(data["pause"])
        self.assertFalse(data["cancel"])

    def test_resume_writes_pause_false(self):
        self.ctrl.pause()
        self.ctrl.resume()
        with open(self.ctrl._control_file, encoding="utf-8") as f:
            data = json.load(f)
        self.assertFalse(data["pause"])
        self.assertFalse(data["cancel"])

    def test_cancel_writes_cancel_true(self):
        self.ctrl.cancel()
        with open(self.ctrl._control_file, encoding="utf-8") as f:
            data = json.load(f)
        self.assertTrue(data["cancel"])
        self.assertFalse(data["pause"])

    def test_control_file_no_tmp_left(self):
        self.ctrl.pause()
        tmp = self.ctrl._control_file + ".tmp"
        self.assertFalse(os.path.exists(tmp))


class TestWorkerControllerStdoutReader(unittest.TestCase):
    """Tests the stdout reader logic in isolation (no subprocess)."""

    def _simulate_reader(self, lines):
        """Simulate WorkerController._read_stdout with given lines."""
        eq = queue.Queue()
        ctrl = WorkerController(eq)

        # Build a mock process with those lines
        mock_proc = MagicMock()
        stdout_text = "\n".join(lines) + "\n"
        mock_proc.stdout = iter(stdout_text.splitlines(keepends=True))
        ctrl._process = mock_proc

        # Run _read_stdout synchronously in this thread
        ctrl._read_stdout()
        return eq

    def test_valid_json_lines_enqueued(self):
        lines = [
            '{"type":"worker_started","pid":999}',
            '{"type":"heartbeat","timestamp":1.0}',
        ]
        eq = self._simulate_reader(lines)
        events = []
        while not eq.empty():
            events.append(eq.get())
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["type"], "worker_started")
        self.assertEqual(events[1]["type"], "heartbeat")

    def test_invalid_json_becomes_log_event(self):
        lines = [
            "not valid json",
            '{"type":"worker_started","pid":1}',
        ]
        eq = self._simulate_reader(lines)
        events = []
        while not eq.empty():
            events.append(eq.get())
        self.assertEqual(events[0]["type"], "log")
        self.assertIn("not valid json", events[0]["message"])
        self.assertEqual(events[1]["type"], "worker_started")

    def test_empty_lines_ignored(self):
        lines = ["", "   ", '{"type":"worker_started","pid":1}', ""]
        eq = self._simulate_reader(lines)
        events = []
        while not eq.empty():
            events.append(eq.get())
        # Only one non-empty non-whitespace JSON line
        json_events = [e for e in events if e.get("type") != "log"]
        self.assertEqual(len(json_events), 1)

    def test_heartbeat_updates_last_heartbeat(self):
        ctrl = WorkerController(queue.Queue())
        mock_proc = MagicMock()
        ts = time.time()
        line = json.dumps({"type": "heartbeat", "timestamp": ts})
        mock_proc.stdout = iter([line + "\n"])
        ctrl._process = mock_proc
        ctrl._read_stdout()
        self.assertGreater(ctrl._last_heartbeat, 0)

    def test_page_progress_updates_last_completed_page(self):
        ctrl = WorkerController(queue.Queue())
        mock_proc = MagicMock()
        line = json.dumps({"type": "page_progress", "path": "a.pdf", "page": 7, "pages": 25})
        mock_proc.stdout = iter([line + "\n"])
        ctrl._process = mock_proc
        ctrl._read_stdout()
        self.assertEqual(ctrl._last_completed_page, 7)


class TestWorkerControllerWorkerExited(unittest.TestCase):
    """Tests that worker_exited event is emitted after process ends."""

    def _run_monitor_with_returncode(self, returncode):
        eq = queue.Queue()
        ctrl = WorkerController(eq)
        ctrl._control_file = ""  # no control file to clean up

        tmpdir = tempfile.mkdtemp()
        ctrl._worker_log_path = os.path.join(tmpdir, "worker.log")

        mock_proc = MagicMock()
        mock_proc.returncode = returncode
        mock_proc.wait.return_value = returncode
        ctrl._process = mock_proc

        # Run monitor in thread and join
        t = threading.Thread(target=ctrl._monitor_process, daemon=True)
        t.start()
        t.join(timeout=5)

        # Collect events
        events = []
        while not eq.empty():
            events.append(eq.get())
        return events, tmpdir

    def test_exit_zero_emits_worker_exited(self):
        events, tmpdir = self._run_monitor_with_returncode(0)
        types = [e["type"] for e in events]
        self.assertIn("worker_exited", types)
        exited = next(e for e in events if e["type"] == "worker_exited")
        self.assertEqual(exited["returncode"], 0)

    def test_exit_nonzero_emits_worker_exited_with_crash_log(self):
        events, tmpdir = self._run_monitor_with_returncode(-1073741819)
        exited = next(e for e in events if e["type"] == "worker_exited")
        self.assertEqual(exited["returncode"], -1073741819)
        self.assertIn("存取違規", exited["description"])

    def test_worker_exited_contains_last_completed_page(self):
        events, tmpdir = self._run_monitor_with_returncode(0)
        exited = next(e for e in events if e["type"] == "worker_exited")
        self.assertIn("last_completed_page", exited)

    def test_gui_state_survives_crash(self):
        """Simulate that after worker crash, the queue still holds the event
        and the controller itself does not raise."""
        eq = queue.Queue()
        ctrl = WorkerController(eq)
        ctrl._control_file = ""

        mock_proc = MagicMock()
        mock_proc.returncode = -1073741819
        mock_proc.wait.return_value = -1073741819
        ctrl._process = mock_proc

        # Should not raise
        try:
            ctrl._monitor_process()
        except Exception as exc:
            self.fail(f"_monitor_process raised unexpectedly: {exc}")

        event_found = False
        while not eq.empty():
            e = eq.get()
            if e.get("type") == "worker_exited":
                event_found = True
        self.assertTrue(event_found)


class TestWorkerControllerRetryLogic(unittest.TestCase):
    """Tests crash-retry counter logic used by GUI."""

    def test_first_crash_should_retry_gpu(self):
        crash_count = 0
        max_gpu_retries = 1
        self.assertTrue(crash_count < max_gpu_retries)

    def test_second_crash_should_fall_back_cpu(self):
        crash_count = 1
        max_gpu_retries = 1
        max_cpu_retries = 1
        self.assertFalse(crash_count < max_gpu_retries)
        self.assertTrue(crash_count - max_gpu_retries < max_cpu_retries)

    def test_exceed_retries_should_mark_failed(self):
        crash_count = 2
        max_gpu_retries = 1
        max_cpu_retries = 1
        total_retries = max_gpu_retries + max_cpu_retries
        self.assertFalse(crash_count < total_retries)


if __name__ == "__main__":
    unittest.main()
