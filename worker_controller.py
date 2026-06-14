"""
worker_controller.py — Manages ocr_worker.py subprocess lifecycle.
Used by ocr_gui.py to run OCR in a crash-safe separate process.
"""

import os
import sys
import json
import uuid
import queue
import threading
import subprocess
import time
import logging
from datetime import datetime
from pathlib import Path

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")

KNOWN_EXIT_CODES = {
    0:            "正常完成",
    1:            "一般錯誤",
    -1073741819:  "原生程式庫存取違規（CUDA/Paddle/OpenCV/PyMuPDF），Access Violation",
    -1073740791:  "Stack buffer overrun",
    -1073741502:  "DLL 初始化失敗",
    -1073741515:  "找不到必要 DLL",
    -1073741676:  "堆積損毀",
}


def describe_windows_exit_code(returncode) -> str:
    if returncode is None:
        return "尚未結束"
    desc = KNOWN_EXIT_CODES.get(returncode)
    if desc:
        return f"{returncode} ({hex(returncode & 0xFFFFFFFF)}): {desc}"
    return f"{returncode} ({hex(returncode & 0xFFFFFFFF)}): 未知退出代碼"


class WorkerController:
    """
    Launches and monitors ocr_worker.py subprocess.
    Thread-safe: GUI can call pause/resume/cancel from main thread.
    Events are placed in event_queue for GUI to consume via root.after().
    """

    def __init__(self, event_queue: queue.Queue):
        self.event_queue = event_queue
        self._process = None
        self._job_id = ""
        self._control_file = ""
        self._stdout_thread = None
        self._stderr_thread = None
        self._monitor_thread = None
        self._last_heartbeat = 0.0
        self._last_completed_page = -1
        self._last_event = {}
        self._worker_log_path = ""
        self._crash_log_path = ""

    # ------------------------------------------------------------------ #
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def get_pid(self):
        return self._process.pid if self._process else None

    # ------------------------------------------------------------------ #
    def start(self, job: dict) -> bool:
        """Launch the worker. Returns True if started successfully."""
        if self.is_running():
            return False

        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(LOGS_DIR, exist_ok=True)
        Path(os.path.join(DATA_DIR, "checkpoints")).mkdir(parents=True, exist_ok=True)

        self._job_id = job.get("job_id", str(uuid.uuid4())[:8])
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Control file
        self._control_file = os.path.join(DATA_DIR, f"runtime_{self._job_id}.control.json")
        self._write_control(pause=False, cancel=False)

        # Worker log
        self._worker_log_path = os.path.join(LOGS_DIR, f"worker_{self._job_id}_{ts}.log")

        # Job file
        job["control_file"] = self._control_file
        job["checkpoint_dir"] = os.path.join(DATA_DIR, "checkpoints")
        job_file = os.path.join(DATA_DIR, f"job_{self._job_id}.json")
        with open(job_file, "w", encoding="utf-8") as f:
            json.dump(job, f, ensure_ascii=False, indent=2)

        cmd = [
            sys.executable,
            os.path.join(PROJECT_DIR, "ocr_worker.py"),
            "--job-file", job_file,
            "--worker-log", self._worker_log_path,
        ]

        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=PROJECT_DIR,
                env=os.environ.copy(),
                creationflags=creationflags,
            )
        except Exception as exc:
            self.event_queue.put({"type": "controller_error", "error": str(exc)})
            return False

        self._last_heartbeat = time.time()
        self._last_completed_page = -1
        self._last_event = {}
        self._crash_log_path = ""

        # Start reader threads
        self._stdout_thread = threading.Thread(
            target=self._read_stdout, daemon=True
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr, daemon=True
        )
        self._monitor_thread = threading.Thread(
            target=self._monitor_process, daemon=True
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        self._monitor_thread.start()

        return True

    # ------------------------------------------------------------------ #
    def _write_control(self, pause: bool, cancel: bool):
        tmp = self._control_file + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"pause": pause, "cancel": cancel}, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._control_file)
        except Exception:
            pass

    def pause(self):
        self._write_control(pause=True, cancel=False)

    def resume(self):
        self._write_control(pause=False, cancel=False)

    def cancel(self):
        self._write_control(pause=False, cancel=True)

    def force_stop(self, timeout_graceful=10, timeout_kill=5):
        """Send cancel, wait, then terminate, then kill."""
        if not self._process:
            return
        self.cancel()
        try:
            self._process.wait(timeout=timeout_graceful)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            self._process.terminate()
            self._process.wait(timeout=timeout_kill)
        except Exception:
            pass
        try:
            self._process.kill()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def _read_stdout(self):
        try:
            for line in self._process.stdout:
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    # Track heartbeat time
                    if event.get("type") == "heartbeat":
                        self._last_heartbeat = time.time()
                    elif event.get("type") == "page_progress":
                        self._last_completed_page = event.get("page", -1)
                    self._last_event = event
                    self.event_queue.put(event)
                except json.JSONDecodeError:
                    # Non-JSON output: treat as a log message
                    self.event_queue.put({"type": "log", "message": f"[worker] {line}"})
        except Exception:
            pass

    def _read_stderr(self):
        try:
            for line in self._process.stderr:
                line = line.rstrip("\n")
                if line:
                    self.event_queue.put({"type": "log", "message": f"[worker stderr] {line}"})
        except Exception:
            pass

    def _monitor_process(self):
        """Watch process exit and emit worker_exited event."""
        self._process.wait()
        returncode = self._process.returncode
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Write crash log if non-zero
        if returncode != 0:
            crash_log = os.path.join(LOGS_DIR, f"worker_crash_{ts}.log")
            self._crash_log_path = crash_log
            try:
                with open(crash_log, "w", encoding="utf-8") as f:
                    f.write(f"Worker exit code: {returncode}\n")
                    f.write(f"Description: {describe_windows_exit_code(returncode)}\n")
                    f.write(f"Last completed page: {self._last_completed_page}\n")
                    f.write(f"Last event: {json.dumps(self._last_event, ensure_ascii=False)}\n")
                    f.write(f"Worker log: {self._worker_log_path}\n")
            except Exception:
                pass

        # Clean up control file
        try:
            if os.path.exists(self._control_file):
                os.remove(self._control_file)
        except Exception:
            pass

        self.event_queue.put({
            "type": "worker_exited",
            "returncode": returncode,
            "description": describe_windows_exit_code(returncode),
            "last_completed_page": self._last_completed_page,
            "crash_log": self._crash_log_path if returncode != 0 else "",
        })
