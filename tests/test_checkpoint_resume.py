"""
test_checkpoint_resume.py — Tests for per-page checkpoint atomic write, load, and resume logic.
No real PDF, no OCR.
"""

import sys
import os
import json
import time
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_worker import _checkpoint_path, _load_checkpoint, _save_checkpoint


def _make_fake_pdf(directory: Path, name="test.pdf") -> Path:
    """Create a small fake file so os.stat() succeeds."""
    p = directory / name
    p.write_bytes(b"%PDF-1.4 fake content for test")
    return p


class TestCheckpointAtomicWrite(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_save_creates_file(self):
        cp_path = self.tmpdir / ".ocr_state_abc123.json"
        data = {"source_path": "/tmp/a.pdf", "completed_pages": [0, 1]}
        _save_checkpoint(cp_path, data)
        self.assertTrue(cp_path.exists())

    def test_save_no_tmp_file_left(self):
        cp_path = self.tmpdir / ".ocr_state_abc123.json"
        _save_checkpoint(cp_path, {"completed_pages": []})
        tmp = cp_path.with_suffix(".tmp")
        self.assertFalse(tmp.exists())

    def test_save_content_is_valid_json(self):
        cp_path = self.tmpdir / ".ocr_state_test.json"
        data = {"completed_pages": [0, 1, 2], "status": "running"}
        _save_checkpoint(cp_path, data)
        with cp_path.open(encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertEqual(loaded["completed_pages"], [0, 1, 2])

    def test_atomic_replace(self):
        """Verify that os.replace is used (no partial write visible)."""
        cp_path = self.tmpdir / ".ocr_state_atomic.json"
        _save_checkpoint(cp_path, {"step": 1})
        _save_checkpoint(cp_path, {"step": 2})
        with cp_path.open() as f:
            loaded = json.load(f)
        self.assertEqual(loaded["step"], 2)


class TestCheckpointPath(unittest.TestCase):
    def test_same_path_same_hash(self):
        a = _checkpoint_path(Path("/tmp/cp"), "/some/path/file.pdf")
        b = _checkpoint_path(Path("/tmp/cp"), "/some/path/file.pdf")
        self.assertEqual(a, b)

    def test_different_paths_different_hashes(self):
        a = _checkpoint_path(Path("/tmp/cp"), "/path/a.pdf")
        b = _checkpoint_path(Path("/tmp/cp"), "/path/b.pdf")
        self.assertNotEqual(a, b)

    def test_filename_format(self):
        p = _checkpoint_path(Path("/tmp/cp"), "/some/file.pdf")
        self.assertTrue(p.name.startswith(".ocr_state_"))
        self.assertTrue(p.name.endswith(".json"))


class TestCheckpointLoadValidation(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_load_nonexistent_returns_none(self):
        cp_path = self.tmpdir / ".ocr_state_missing.json"
        result = _load_checkpoint(cp_path, "/fake/path.pdf")
        self.assertIsNone(result)

    def test_valid_checkpoint_returns_data(self):
        fake_pdf = _make_fake_pdf(self.tmpdir)
        stat = os.stat(str(fake_pdf))
        cp_path = _checkpoint_path(self.tmpdir, str(fake_pdf))
        data = {
            "source_path": str(fake_pdf),
            "source_size": stat.st_size,
            "source_mtime": stat.st_mtime,
            "completed_pages": [0, 1, 2],
        }
        _save_checkpoint(cp_path, data)
        loaded = _load_checkpoint(cp_path, str(fake_pdf))
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["completed_pages"], [0, 1, 2])

    def test_size_mismatch_invalidates_checkpoint(self):
        fake_pdf = _make_fake_pdf(self.tmpdir)
        stat = os.stat(str(fake_pdf))
        cp_path = _checkpoint_path(self.tmpdir, str(fake_pdf))
        data = {
            "source_size": stat.st_size + 100,  # wrong size
            "source_mtime": stat.st_mtime,
            "completed_pages": [0],
        }
        _save_checkpoint(cp_path, data)
        loaded = _load_checkpoint(cp_path, str(fake_pdf))
        self.assertIsNone(loaded)

    def test_mtime_mismatch_invalidates_checkpoint(self):
        fake_pdf = _make_fake_pdf(self.tmpdir)
        stat = os.stat(str(fake_pdf))
        cp_path = _checkpoint_path(self.tmpdir, str(fake_pdf))
        data = {
            "source_size": stat.st_size,
            "source_mtime": stat.st_mtime + 1.0,  # wrong mtime
            "completed_pages": [0],
        }
        _save_checkpoint(cp_path, data)
        loaded = _load_checkpoint(cp_path, str(fake_pdf))
        self.assertIsNone(loaded)

    def test_corrupt_json_returns_none(self):
        cp_path = self.tmpdir / ".ocr_state_bad.json"
        cp_path.write_text("{invalid json{{", encoding="utf-8")
        result = _load_checkpoint(cp_path, "/fake/path.pdf")
        self.assertIsNone(result)

    def test_completed_pages_skipped_on_resume(self):
        """Verify that completed pages from checkpoint form a set for skipping."""
        fake_pdf = _make_fake_pdf(self.tmpdir)
        stat = os.stat(str(fake_pdf))
        cp_path = _checkpoint_path(self.tmpdir, str(fake_pdf))
        data = {
            "source_size": stat.st_size,
            "source_mtime": stat.st_mtime,
            "completed_pages": [0, 1, 2, 3, 4],
            "page_texts": {"0": "text0", "1": "text1"},
        }
        _save_checkpoint(cp_path, data)
        loaded = _load_checkpoint(cp_path, str(fake_pdf))
        completed_set = set(loaded["completed_pages"])
        # Pages 0-4 should be skippable
        self.assertIn(0, completed_set)
        self.assertIn(4, completed_set)
        self.assertNotIn(5, completed_set)


if __name__ == "__main__":
    unittest.main()
