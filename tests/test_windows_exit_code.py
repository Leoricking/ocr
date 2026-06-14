"""
test_windows_exit_code.py — Tests for describe_windows_exit_code().
No real PDF, no OCR.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker_controller import describe_windows_exit_code


class TestDescribeWindowsExitCode(unittest.TestCase):
    def test_none_returns_not_finished(self):
        result = describe_windows_exit_code(None)
        self.assertIn("尚未結束", result)

    def test_zero_normal_completion(self):
        result = describe_windows_exit_code(0)
        self.assertIn("正常完成", result)
        self.assertIn("0", result)

    def test_access_violation(self):
        result = describe_windows_exit_code(-1073741819)
        self.assertIn("存取違規", result)
        self.assertIn("0xc0000005", result.lower())

    def test_stack_buffer_overrun(self):
        result = describe_windows_exit_code(-1073740791)
        self.assertIn("overrun", result.lower())

    def test_dll_init_failure(self):
        result = describe_windows_exit_code(-1073741502)
        self.assertIn("DLL", result)

    def test_unknown_code(self):
        result = describe_windows_exit_code(99999)
        self.assertIn("未知", result)
        self.assertIn("99999", result)

    def test_general_error(self):
        result = describe_windows_exit_code(1)
        self.assertIn("一般錯誤", result)

    def test_hex_included(self):
        # Verify hex representation is included
        result = describe_windows_exit_code(-1073741819)
        self.assertIn("0x", result.lower())


if __name__ == "__main__":
    unittest.main()
