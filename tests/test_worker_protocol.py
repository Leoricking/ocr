"""
test_worker_protocol.py — Tests for worker JSON Lines protocol parsing.
No real PDF, no OCR, no Claude API.
"""

import sys
import os
import json
import queue
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestWorkerProtocol(unittest.TestCase):
    """Tests that worker JSON events are parseable and well-formed."""

    def test_worker_started_event(self):
        event = {"type": "worker_started", "pid": 1234}
        self.assertEqual(event["type"], "worker_started")
        self.assertIsInstance(event["pid"], int)

    def test_environment_event(self):
        event = {"type": "environment", "device": "gpu", "pid": 1234, "python": "3.9.0"}
        parsed = json.loads(json.dumps(event, ensure_ascii=False))
        self.assertEqual(parsed["type"], "environment")
        self.assertIn(parsed["device"], ("gpu", "cpu"))

    def test_file_started_event(self):
        event = {"type": "file_started", "path": "/tmp/test.pdf", "index": 0, "total": 3}
        parsed = json.loads(json.dumps(event, ensure_ascii=False))
        self.assertEqual(parsed["index"], 0)
        self.assertEqual(parsed["total"], 3)

    def test_page_progress_event(self):
        event = {
            "type": "page_progress",
            "path": "/tmp/test.pdf",
            "page": 5,
            "pages": 25,
            "stage": "OCR",
            "preprocess": "CLAHE",
            "elapsed": 12.4,
        }
        parsed = json.loads(json.dumps(event, ensure_ascii=False))
        self.assertEqual(parsed["page"], 5)
        self.assertEqual(parsed["pages"], 25)

    def test_file_completed_event(self):
        event = {
            "type": "file_completed",
            "path": "/tmp/test.pdf",
            "pdf": "/out/test_OCR.pdf",
            "txt": "/out/test_OCR.txt",
            "analysis": "/out/test_OCR_analysis.txt",
        }
        parsed = json.loads(json.dumps(event, ensure_ascii=False))
        self.assertEqual(parsed["type"], "file_completed")

    def test_file_failed_event(self):
        event = {
            "type": "file_failed",
            "path": "/tmp/test.pdf",
            "error": "some error",
            "traceback": "Traceback...",
        }
        parsed = json.loads(json.dumps(event, ensure_ascii=False))
        self.assertEqual(parsed["type"], "file_failed")
        self.assertIn("error", parsed)

    def test_batch_completed_event(self):
        event = {"type": "batch_completed", "completed": 2, "failed": 1, "cancelled": 0}
        parsed = json.loads(json.dumps(event, ensure_ascii=False))
        self.assertEqual(parsed["completed"] + parsed["failed"] + parsed["cancelled"], 3)

    def test_heartbeat_event(self):
        import time
        event = {"type": "heartbeat", "timestamp": time.time()}
        parsed = json.loads(json.dumps(event, ensure_ascii=False))
        self.assertIsInstance(parsed["timestamp"], float)

    def test_chinese_characters_in_json(self):
        event = {"type": "log", "message": "處理中 — 第 1/25 頁"}
        serialized = json.dumps(event, ensure_ascii=False)
        parsed = json.loads(serialized)
        self.assertIn("處理", parsed["message"])

    def test_non_json_stdout_does_not_raise(self):
        """Simulates WorkerController._read_stdout encountering invalid JSON."""
        eq = queue.Queue()

        def simulate_read_line(line):
            line = line.rstrip("\n")
            if not line:
                return
            try:
                event = json.loads(line)
                eq.put(event)
            except json.JSONDecodeError:
                eq.put({"type": "log", "message": f"[worker] {line}"})

        simulate_read_line('{"type": "worker_started", "pid": 999}')
        simulate_read_line("Not valid JSON at all!")
        simulate_read_line("Another bad line [[[")
        simulate_read_line('{"type": "heartbeat", "timestamp": 1.0}')

        events = []
        while not eq.empty():
            events.append(eq.get())

        self.assertEqual(len(events), 4)
        self.assertEqual(events[0]["type"], "worker_started")
        self.assertEqual(events[1]["type"], "log")
        self.assertIn("Not valid JSON", events[1]["message"])
        self.assertEqual(events[2]["type"], "log")
        self.assertEqual(events[3]["type"], "heartbeat")

    def test_all_required_event_types_parseable(self):
        event_lines = [
            '{"type":"worker_started","pid":1234}',
            '{"type":"environment","device":"gpu"}',
            '{"type":"file_started","path":"a.pdf","index":1,"total":3}',
            '{"type":"page_progress","path":"a.pdf","page":5,"pages":25,"stage":"OCR","preprocess":"CLAHE","elapsed":12.4}',
            '{"type":"file_completed","path":"a.pdf","pdf":"a_OCR.pdf","txt":"a_OCR.txt","analysis":"a_OCR_analysis.txt"}',
            '{"type":"file_failed","path":"a.pdf","error":"err","traceback":"tb"}',
            '{"type":"batch_completed","completed":2,"failed":1,"cancelled":0}',
            '{"type":"heartbeat","timestamp":1.0}',
        ]
        for line in event_lines:
            with self.subTest(line=line):
                parsed = json.loads(line)
                self.assertIn("type", parsed)


if __name__ == "__main__":
    unittest.main()
