# settings_manager.py — GUI settings persistence for OCR Engine v4.4.0
# Settings stored at: C:\Users\Rossi\Documents\Claude\OCR\data\gui_settings.json

import json
import os
from pathlib import Path
from datetime import datetime

SETTINGS_PATH = Path(__file__).parent / "data" / "gui_settings.json"

DEFAULTS = {
    "version": 1,
    "last_input_path": "",
    "last_output_path": "",
    "input_mode": "folder",          # "file" or "folder"
    "recursive": False,
    "device": "gpu",
    "enable_claude": False,
    "claude_model": "",
    "zoom": 3,
    "preprocess_mode": "auto",
    "output_pdf": True,
    "output_txt": True,
    "output_analysis": True,
    "output_verify_log": True,
    "overwrite": False,
    "skip_existing": True,
    "confidence_threshold": 0.7,
    "preserve_relative_structure": True,
    "window_geometry": "1280x820+100+80",
    "batch_failure_strategy": "continue",  # "continue" or "stop"
}


class SettingsManager:
    """Load, save, and reset GUI settings with atomic writes and corrupt-file recovery."""

    def __init__(self, path: Path = SETTINGS_PATH):
        self._path = Path(path)

    # ----------------------------------------------------------
    def load(self) -> dict:
        """Return settings dict with all DEFAULTS filled in for missing keys."""
        result = dict(DEFAULTS)
        if not self._path.exists():
            return result
        try:
            with self._path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError("Settings file root is not a JSON object.")
            result.update(data)
            return result
        except (json.JSONDecodeError, ValueError) as exc:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            broken = self._path.with_name(f"gui_settings.broken.{ts}.json")
            try:
                self._path.rename(broken)
                print(f"[設定] 設定檔損毀，已重新命名為 {broken.name}，使用預設值。({exc})")
            except OSError:
                print(f"[設定] 設定檔損毀且無法重新命名，使用預設值。({exc})")
            return dict(DEFAULTS)
        except OSError as exc:
            print(f"[設定] 無法讀取設定檔，使用預設值。({exc})")
            return dict(DEFAULTS)

    # ----------------------------------------------------------
    def save(self, settings: dict) -> bool:
        """Atomically write settings to disk. Returns True on success."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(settings, fh, ensure_ascii=False, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(str(tmp), str(self._path))
            return True
        except OSError as exc:
            print(f"[設定] 儲存設定失敗：{exc}")
            return False

    # ----------------------------------------------------------
    def reset(self) -> dict:
        """Delete settings file and return defaults."""
        try:
            if self._path.exists():
                self._path.unlink()
        except OSError as exc:
            print(f"[設定] 無法刪除設定檔：{exc}")
        return dict(DEFAULTS)
