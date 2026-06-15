import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("worker_controller_test", ROOT / "worker_controller.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_controlled_worker_exit_descriptions():
    assert "檔案失敗" in mod.describe_windows_exit_code(2)
    assert "使用者取消" in mod.describe_windows_exit_code(3)
