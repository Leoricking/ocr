# ocr_engine.py — CLI wrapper for OCR Engine v4.4.0
# Supports both interactive mode and argparse mode.
# The heavy GPU DLL setup is handled by ocr_core at import time.

import os
import sys
import time

# Reconfigure stdout/stderr to UTF-8 to avoid Windows encoding issues
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Keep env var defaults here too (safe to call twice)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")

# Import core — this triggers GPU DLL setup before PaddleOCR loads
from ocr_core import (
    OCRConfig,
    OCRProcessor,
    OCRProgress,
    OCRCancelledError,
    print_environment_summary,
    validate_key,
    CLAUDE_API_KEY,
    CLAUDE_MODEL_DEFAULT,
    gpu_runtime_self_test,
    _find_dll,
    PaddleOCR,
)

from tqdm import tqdm


VERSION = "v4.4.0"
HEADER = f"=== 數千本圖書 PDF 搜尋化系統 ({VERSION} PDF+TXT 品質分析版) ==="


# ==============================================================
# tqdm progress adapter
# ==============================================================
class _TqdmProgressAdapter:
    """Bridges OCRProcessor progress callbacks to tqdm bars."""

    def __init__(self):
        self._pbar = None
        self._current_file = ""
        self._last_page = -1

    def on_progress(self, prog: OCRProgress):
        # Create or recreate tqdm bar when file changes
        if prog.file_path != self._current_file:
            if self._pbar is not None:
                self._pbar.close()
            file_label = os.path.basename(prog.file_path)[:14]
            self._pbar = tqdm(
                total=prog.page_total,
                desc=file_label,
                ascii=" █",
                ncols=115,
                unit="頁",
            )
            self._current_file = prog.file_path
            self._last_page = -1

        if self._pbar is not None and prog.page_index > self._last_page:
            self._pbar.set_postfix(
                速度=f"{prog.average_seconds_per_page:.2f}s/頁",
                引擎=prog.stage,
                剩餘=f"{prog.estimated_remaining_seconds:.0f}s",
            )
            self._pbar.update(prog.page_index - self._last_page)
            self._last_page = prog.page_index

    def on_file_status(self, file_index, file_path, status, extra_info):
        if status in ("完成", "跳過", "失敗", "已取消"):
            if self._pbar is not None:
                self._pbar.close()
                self._pbar = None
            self._current_file = ""
            self._last_page = -1
        if status not in ("OCR 中", "初始化", "等待"):
            msg = f"[{status}] {os.path.basename(file_path)}"
            if extra_info:
                msg += f"  {extra_info}"
            print(msg)

    def close(self):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None


# ==============================================================
# Interactive CLI (original behavior)
# ==============================================================
def _run_interactive():
    print(HEADER)
    input_path = input("\n請輸入【來源路徑】: ").strip().replace('"', '')
    output_root = input("請輸入【輸出路徑】: ").strip().replace('"', '')
    gpu_choice = input("1: GPU[RTX 3060], 2: CPU: ").strip()
    use_gpu = gpu_choice == '1'
    ai_choice = input("是否啟用 Claude 自動校對？ (y/n): ").strip().lower()
    enable_claude = ai_choice == 'y'

    input_path = os.path.abspath(os.path.expanduser(input_path))
    output_root = os.path.abspath(os.path.expanduser(output_root))

    if not os.path.exists(input_path):
        print(f"\n[錯誤] 來源路徑不存在：{input_path}")
        return

    print_environment_summary()

    if use_gpu:
        missing_dlls = []
        if not _find_dll("cudnn_ops_infer64_8.dll"):
            missing_dlls.append("cudnn_ops_infer64_8.dll")
        if not _find_dll("cublasLt64_11.dll"):
            missing_dlls.append("cublasLt64_11.dll")
        if missing_dlls:
            print("\n[GPU] 缺少必要 DLL，為避免 Python 崩潰，改用 CPU。")
            for dll_name in missing_dlls:
                print(f"      缺少：{dll_name}")
            use_gpu = False
        else:
            print("\n[GPU] 正在以獨立子程序測試 CUDA/cuDNN/cuBLAS...")
            gpu_ok, gpu_detail = gpu_runtime_self_test()
            if gpu_ok:
                print("[GPU] 實際卷積測試通過，將使用 gpu:0。")
            else:
                print("[GPU] 實際運算測試失敗，已自動回退 CPU。")
                if gpu_detail:
                    print(gpu_detail[-2000:])
                use_gpu = False

    if enable_claude:
        if not CLAUDE_API_KEY or CLAUDE_API_KEY == "YOUR_CLAUDE_API_KEY":
            print("\n[API] 尚未設定 CLAUDE_API_KEY，已取消 AI 校對。")
            enable_claude = False
        elif not validate_key():
            return

    print(f"\n[執行裝置] {'GPU' if use_gpu else 'CPU'}")

    config = OCRConfig(
        input_path=input_path,
        output_path=output_root,
        device="gpu" if use_gpu else "cpu",
        enable_claude=enable_claude,
    )

    adapter = _TqdmProgressAdapter()
    processor = OCRProcessor(
        config=config,
        progress_callback=adapter.on_progress,
        log_callback=print,
        file_status_callback=adapter.on_file_status,
    )
    # GPU was already tested above — directly initialize PaddleOCR
    processor._ocr = PaddleOCR(
        lang='chinese_cht',
        use_gpu=use_gpu,
        use_angle_cls=True,
        det_db_thresh=0.2,
        det_db_unclip_ratio=1.8,
        show_log=False,
        enable_mkldnn=not use_gpu,
    )
    processor._use_gpu = use_gpu

    try:
        processor.process_all()
    finally:
        adapter.close()

    output_root_abs = os.path.abspath(os.path.expanduser(output_root))
    print(f"\n  校正日誌: {os.path.join(output_root_abs, 'verify_log.txt')}")


# ==============================================================
# Argparse CLI
# ==============================================================
def _run_argparse(args):
    import argparse
    parser = argparse.ArgumentParser(
        prog="ocr_engine.py",
        description=f"OCR Engine {VERSION} - PDF 搜尋化與品質分析系統",
    )
    parser.add_argument("--input", "-i", metavar="PATH", nargs="+", help="來源 PDF 或資料夾（可多個）")
    parser.add_argument("--output", "-o", metavar="PATH", help="輸出資料夾")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu", help="運算裝置（預設 gpu）")
    parser.add_argument("--claude", choices=["on", "off"], default="off", help="Claude 自動校對（預設 off）")
    parser.add_argument("--recursive", action="store_true", help="遞迴處理子資料夾")

    # Mutually exclusive overwrite / skip-existing
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument("--overwrite", action="store_true", help="覆蓋已有輸出")
    overwrite_group.add_argument("--skip-existing", action="store_true", default=False,
                                  help="跳過已有輸出（與 --overwrite 互斥）")

    parser.add_argument("--zoom", type=int, choices=[2, 3, 4], default=3, help="縮放倍率（預設 3）")
    parser.add_argument(
        "--preprocess", choices=["auto", "original", "clahe", "binary"], default="auto",
        help="預處理模式（預設 auto）",
    )
    parser.add_argument("--no-pdf", action="store_true", help="不輸出 PDF")
    parser.add_argument("--no-txt", action="store_true", help="不輸出 TXT")
    parser.add_argument("--no-analysis", action="store_true", help="不輸出品質分析")
    parser.add_argument("--no-verify-log", action="store_true", help="不輸出校正日誌")
    parser.add_argument("--flat-output", action="store_true",
                        help="全部輸出到同一資料夾（不保留相對目錄結構）")

    parsed = parser.parse_args(args)

    print(HEADER)

    if not parsed.input:
        parser.error("請提供 --input 路徑")
    if not parsed.output:
        parser.error("請提供 --output 路徑")

    output_path = os.path.abspath(os.path.expanduser(parsed.output))

    # Resolve skip_existing: default True unless --overwrite is set
    skip_existing = parsed.skip_existing or (not parsed.overwrite)

    # Handle multiple input paths
    input_paths = [os.path.abspath(os.path.expanduser(p)) for p in parsed.input]
    for p in input_paths:
        if not os.path.exists(p):
            print(f"[錯誤] 來源路徑不存在：{p}")
            sys.exit(1)

    print_environment_summary()

    adapter = _TqdmProgressAdapter()

    for input_path in input_paths:
        config = OCRConfig(
            input_path=input_path,
            output_path=output_path,
            device=parsed.device,
            enable_claude=(parsed.claude == "on"),
            recursive=parsed.recursive,
            overwrite=parsed.overwrite,
            skip_existing=skip_existing and not parsed.overwrite,
            zoom=parsed.zoom,
            preprocess_mode=parsed.preprocess,
            output_pdf=not parsed.no_pdf,
            output_txt=not parsed.no_txt,
            output_analysis=not parsed.no_analysis,
            output_verify_log=not parsed.no_verify_log,
            preserve_relative_structure=not parsed.flat_output,
        )

        processor = OCRProcessor(
            config=config,
            progress_callback=adapter.on_progress,
            log_callback=print,
            file_status_callback=adapter.on_file_status,
        )

        try:
            processor.process_all()
        except OCRCancelledError:
            print("\n[取消] 使用者中止。")
            break

    adapter.close()
    print(f"\n  校正日誌: {os.path.join(output_path, 'verify_log.txt')}")


# ==============================================================
# Entry point
# ==============================================================
def main():
    # Detect whether any -- flags were passed
    has_flags = any(
        arg.startswith("--") or (arg.startswith("-") and len(arg) == 2)
        for arg in sys.argv[1:]
    )
    if has_flags:
        _run_argparse(sys.argv[1:])
    else:
        _run_interactive()


if __name__ == "__main__":
    main()
