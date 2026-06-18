#!/usr/bin/env python3
"""
ocr_worker.py — OCR Engine v4.5.0 standalone worker process.
Run by ocr_gui.py via subprocess.Popen. Outputs JSON Lines to stdout.
"""

import os as _os, sys as _sys
_os.environ["PYTHONUTF8"] = "1"
_os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace", line_buffering=True)
if hasattr(_sys.stderr, "reconfigure"):
    _sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace", line_buffering=True)

import os
import sys
import json
import gc
import time
import argparse
import traceback
import contextlib
from datetime import datetime
from pathlib import Path

JSON_STDOUT = sys.stdout   # save reference before any redirect


# ---------------------------------------------------------------------------
# JSON Lines output helpers — stdout ONLY
# ---------------------------------------------------------------------------

def emit(event: dict):
    """Output a single JSON event to stdout."""
    print(json.dumps(event, ensure_ascii=False), file=JSON_STDOUT, flush=True)


def _check_encoding(event: dict):
    """Warn if any string value contains UTF-8 replacement characters."""
    for v in event.values():
        if isinstance(v, str) and "\ufffd" in v:
            emit({"type": "log", "message": f"[encoding] replacement char in {list(event.keys())}"})
            break


def emit_log(message: str):
    ev = {"type": "log", "message": message}
    _check_encoding(ev)
    emit(ev)


def emit_heartbeat():
    emit({"type": "heartbeat", "timestamp": time.time()})


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _checkpoint_path(checkpoint_dir: Path, pdf_path: str) -> Path:
    import hashlib
    safe = hashlib.md5(pdf_path.encode()).hexdigest()[:12]
    return checkpoint_dir / f".ocr_state_{safe}.json"


def _load_checkpoint(path: Path, pdf_path: str):
    """Return checkpoint dict if valid, else None."""
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            cp = json.load(f)
        stat = os.stat(pdf_path)
        if cp.get("source_size") != stat.st_size or cp.get("source_mtime") != stat.st_mtime:
            return None
        return cp
    except Exception:
        return None


def _save_checkpoint(path: Path, data: dict):
    """Atomic checkpoint write."""
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


# ---------------------------------------------------------------------------
# Control file helpers
# ---------------------------------------------------------------------------

_was_paused = False


def _read_control(control_file: str) -> dict:
    try:
        with open(control_file, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"pause": False, "cancel": False}


def _check_control(control_file: str, OCRCancelledError):
    """Check for pause/cancel. Blocks while paused. Raises if cancelled."""
    global _was_paused
    while True:
        ctrl = _read_control(control_file)
        if ctrl.get("cancel"):
            raise OCRCancelledError("Cancelled by control file.")
        if not ctrl.get("pause"):
            break
        if not _was_paused:
            emit({"type": "paused"})
            _was_paused = True
        time.sleep(0.5)
    if _was_paused:
        emit({"type": "resumed"})
        _was_paused = False


# ---------------------------------------------------------------------------
# Per-PDF processing with checkpoint resume
# ---------------------------------------------------------------------------

def process_pdf_with_checkpoint(
    pdf_path, config, ocr_instance, checkpoint_dir, control_file,
    file_index, file_total, OCRCancelledError
):
    import fitz
    from ocr_core import (
        ocr_page_paddle, apply_safe_corrections, align_lines,
        apply_text_overlay, append_ocr_text, append_quality_analysis,
        log_corrections, FINANCE_KEYWORDS, PHYSICS_KEYWORDS,
        BATCH_SIZE, CONF_SKIP_THRESHOLD, select_engine, engine_prompt,
        call_claude_batch, ocr_texts_nougat, ocr_texts_surya,
        CLAUDE_API_KEY, CLAUDE_MODEL_DEFAULT,
    )

    file_name = os.path.basename(pdf_path)
    stem = os.path.splitext(file_name)[0]

    out_dir = os.path.abspath(config.output_path)
    os.makedirs(out_dir, exist_ok=True)

    target_pdf = os.path.join(out_dir, stem + "_OCR.pdf")
    target_txt = os.path.join(out_dir, stem + "_OCR.txt")
    raw_txt = os.path.join(out_dir, stem + "_OCR_raw.txt")
    corrected_txt = os.path.join(out_dir, stem + "_OCR_corrected.txt")
    analysis_txt = os.path.join(out_dir, stem + "_OCR_analysis.txt")
    log_path = os.path.join(out_dir, "verify_log.txt")

    # Load or init checkpoint
    cp_path = _checkpoint_path(checkpoint_dir, pdf_path)
    stat = os.stat(pdf_path)
    cp = _load_checkpoint(cp_path, pdf_path) if config.skip_existing else None
    if cp is None:
        cp = {
            "source_path": pdf_path,
            "source_size": stat.st_size,
            "source_mtime": stat.st_mtime,
            "completed_pages": [],
            "page_texts": {},
            "page_confidences": {},
            "page_preprocess_modes": {},
            "status": "running",
            "updated_at": "",
        }

    completed_set = set(cp["completed_pages"])

    ev_started = {
        "type": "file_started",
        "path": pdf_path,
        "index": file_index,
        "total": file_total,
        "started_monotonic": time.monotonic(),
    }
    _check_encoding(ev_started)
    emit(ev_started)

    doc = fitz.open(pdf_path)
    pages = list(doc)
    total_pages = len(pages)

    ev_pages = {"type": "file_pages", "path": pdf_path, "pages": total_pages}
    _check_encoding(ev_pages)
    emit(ev_pages)

    page_times = []
    t_file_start = time.perf_counter()

    # Clear old temp outputs (not checkpoint) if not resuming
    if not completed_set:
        for p in (target_txt, raw_txt, corrected_txt, analysis_txt):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    try:
        if config.enable_claude:
            emit_log("[離線校正] 已忽略 enable_claude；不使用 Claude API。")
            config.enable_claude = False
        if config.enable_claude:
            for batch_start in range(0, total_pages, BATCH_SIZE):
                _check_control(control_file, OCRCancelledError)
                batch_indices = range(batch_start, min(batch_start + BATCH_SIZE, total_pages))

                pending_in_batch = [i for i in batch_indices if i not in completed_set]
                if not pending_in_batch:
                    continue

                batch_pages = [pages[i] for i in pending_in_batch]
                ocr_data = []
                for i, p in zip(pending_in_batch, batch_pages):
                    _check_control(control_file, OCRCancelledError)
                    t0 = time.perf_counter()
                    ln, tx, cf, mode = ocr_page_paddle(
                        p, ocr_instance, zoom=config.zoom,
                        preprocess_mode=config.preprocess_mode,
                        max_page_pixels=config.max_page_pixels,
                        memory_retry_enabled=config.memory_retry_enabled,
                        memory_retry_scales=config.memory_retry_scales,
                        release_between_candidates=config.release_between_candidates,
                    )
                    if mode == "FAILED_MEMORY":
                        emit_log(
                            f"[記憶體保護] 第 {i + 1} 頁所有候選均失敗，保留原始頁面。"
                        )
                        if config.strict_page_failure:
                            raise MemoryError(f"第 {i + 1} 頁 OCR 記憶體不足")
                    ocr_data.append([ln, tx, cf, config.zoom, mode])

                    elapsed = time.perf_counter() - t0
                    page_times.append(elapsed)
                    avg = sum(page_times) / len(page_times)
                    remaining = avg * (total_pages - len(page_times))
                    file_elapsed = time.perf_counter() - t_file_start
                    emit({
                        "type": "page_progress",
                        "path": pdf_path,
                        "page": i,
                        "pages": total_pages,
                        "stage": "OCR",
                        "preprocess": mode,
                        "elapsed": elapsed,
                        "elapsed_seconds": file_elapsed,
                        "avg_sec_per_page": avg,
                        "estimated_remaining": remaining,
                    })
                    emit_heartbeat()

                sample = " ".join(" ".join(d[1]) for d in ocr_data)
                content_mode = (
                    "FINANCE" if any(kw in sample for kw in FINANCE_KEYWORDS)
                    else "PHYSICS" if any(kw in sample for kw in PHYSICS_KEYWORDS)
                    else "DEFAULT"
                )
                engine = select_engine(
                    [t for d in ocr_data for t in d[1]],
                    [c for d in ocr_data for c in d[2]]
                )

                engine_texts = []
                if engine == "nougat":
                    for p, d in zip(batch_pages, ocr_data):
                        nt = ocr_texts_nougat(p) or d[1]
                        engine_texts.append(nt)
                elif engine == "surya":
                    for p, d in zip(batch_pages, ocr_data):
                        st = ocr_texts_surya(p) or d[1]
                        engine_texts.append(st)
                else:
                    engine_texts = [d[1] for d in ocr_data]

                all_confs = [c for d in ocr_data for c in d[2]]
                avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0
                has_terms = content_mode != "DEFAULT"
                skip_api = avg_conf >= CONF_SKIP_THRESHOLD and not has_terms

                if skip_api:
                    corrected_pages = [None] * len(batch_pages)
                    disp_engine = f"{engine.upper()}+SKIP"
                else:
                    emit_log(f"Claude 校對中... {len(batch_pages)} 頁")
                    prompt, timeout = engine_prompt(engine, content_mode)
                    preprocessed = [apply_safe_corrections(tx, file_name)[0] for tx in engine_texts]
                    model = config.claude_model if config.claude_model else CLAUDE_MODEL_DEFAULT
                    corrected_pages, uncertain = call_claude_batch(
                        preprocessed, prompt, timeout, claude_model=model
                    )
                    disp_engine = engine.upper()

                for j, (page_idx, page_obj, ocr_item) in enumerate(
                    zip(pending_in_batch, batch_pages, ocr_data)
                ):
                    lines, raw_texts, confs, zoom, preprocess_mode = ocr_item
                    corrected = corrected_pages[j]
                    if corrected is None:
                        corrected = engine_texts[j]
                    else:
                        corrected = align_lines(engine_texts[j], corrected)
                    final_texts, rule_sources = apply_safe_corrections(corrected, file_name)
                    sources = [src or ("AI" if a.strip() != b.strip() else "")
                               for src, a, b in zip(rule_sources, raw_texts, final_texts)]

                    if config.output_pdf:
                        apply_text_overlay(page_obj, lines, final_texts, zoom)
                    if config.output_txt:
                        append_ocr_text(target_txt, page_idx, final_texts)
                        append_ocr_text(corrected_txt, page_idx, final_texts)
                    if getattr(config, "output_raw_txt", True):
                        append_ocr_text(raw_txt, page_idx, raw_texts)
                    if config.output_analysis:
                        append_quality_analysis(
                            analysis_txt, file_name, page_idx,
                            raw_texts, confs, preprocess_mode, final_texts, sources
                        )
                    if config.output_verify_log:
                        log_corrections(
                            log_path, file_name, page_idx,
                            raw_texts, final_texts, disp_engine, ai_enabled=True,
                            correction_sources=sources
                        )
                    corrected = final_texts

                    cp["completed_pages"].append(page_idx)
                    cp["page_texts"][str(page_idx)] = corrected
                    cp["page_confidences"][str(page_idx)] = confs
                    cp["page_preprocess_modes"][str(page_idx)] = preprocess_mode
                    cp["updated_at"] = datetime.now().isoformat()
                    _save_checkpoint(cp_path, cp)
                    completed_set.add(page_idx)

        else:
            # No Claude — simple page loop
            for page_num in range(total_pages):
                _check_control(control_file, OCRCancelledError)
                if page_num in completed_set:
                    continue

                t0 = time.perf_counter()
                lines, raw_texts, confidences, preprocess_mode = ocr_page_paddle(
                    pages[page_num], ocr_instance, zoom=config.zoom,
                    preprocess_mode=config.preprocess_mode,
                    max_page_pixels=config.max_page_pixels,
                    memory_retry_enabled=config.memory_retry_enabled,
                    memory_retry_scales=config.memory_retry_scales,
                    release_between_candidates=config.release_between_candidates,
                )
                if preprocess_mode == "FAILED_MEMORY":
                    emit_log(
                        f"[記憶體保護] 第 {page_num + 1} 頁所有候選均失敗，保留原始頁面。"
                    )
                    if config.strict_page_failure:
                        raise MemoryError(f"第 {page_num + 1} 頁 OCR 記憶體不足")
                    # Do not mark this page completed in the checkpoint; a later rerun may retry it.
                    continue
                fixed, sources = apply_safe_corrections(raw_texts, file_name)

                if config.output_pdf:
                    apply_text_overlay(pages[page_num], lines, fixed, config.zoom)
                if config.output_txt:
                    append_ocr_text(target_txt, page_num, fixed)
                    append_ocr_text(corrected_txt, page_num, fixed)
                if getattr(config, "output_raw_txt", True):
                    append_ocr_text(raw_txt, page_num, raw_texts)
                if config.output_analysis:
                    append_quality_analysis(
                        analysis_txt, file_name, page_num,
                        raw_texts, confidences, preprocess_mode, fixed, sources
                    )
                if config.output_verify_log:
                    log_corrections(
                        log_path, file_name, page_num,
                        raw_texts, fixed, "PADDLE", ai_enabled=False,
                        correction_sources=sources
                    )

                elapsed = time.perf_counter() - t0
                page_times.append(elapsed)
                avg = sum(page_times) / len(page_times)
                remaining = avg * (total_pages - len(page_times))
                file_elapsed = time.perf_counter() - t_file_start

                emit({
                    "type": "page_progress",
                    "path": pdf_path,
                    "page": page_num,
                    "pages": total_pages,
                    "stage": "PADDLE+RAW",
                    "preprocess": preprocess_mode,
                    "elapsed": elapsed,
                    "elapsed_seconds": file_elapsed,
                    "avg_sec_per_page": avg,
                    "estimated_remaining": remaining,
                })
                emit_heartbeat()

                cp["completed_pages"].append(page_num)
                cp["page_texts"][str(page_num)] = fixed
                cp["page_confidences"][str(page_num)] = confidences
                cp["page_preprocess_modes"][str(page_num)] = preprocess_mode
                cp["updated_at"] = datetime.now().isoformat()
                _save_checkpoint(cp_path, cp)
                completed_set.add(page_num)

        # Save final PDF
        if config.output_pdf:
            try:
                doc.save(target_pdf, garbage=4, deflate=True)
            except PermissionError:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fallback = target_pdf.replace(".pdf", f"_{ts}.pdf")
                emit_log(f"檔案被佔用，改存為: {fallback}")
                doc.save(fallback, garbage=4, deflate=True)
                target_pdf = fallback

        doc.close()

        # Clean up checkpoint on success
        try:
            cp_path.unlink()
        except Exception:
            pass

        avg_total = sum(page_times) / len(page_times) if page_times else 0
        total_file_elapsed = time.perf_counter() - t_file_start
        emit({
            "type": "file_completed",
            "path": pdf_path,
            "pdf": target_pdf if config.output_pdf else "",
            "txt": target_txt if config.output_txt else "",
            "raw_txt": raw_txt if getattr(config, "output_raw_txt", True) else "",
            "corrected_txt": corrected_txt if config.output_txt else "",
            "analysis": analysis_txt if config.output_analysis else "",
            "avg_sec_per_page": avg_total,
            "total_pages": total_pages,
            "elapsed_seconds": total_file_elapsed,
        })
        return True

    except Exception as exc:
        # Check if it's an OCRCancelledError
        if type(exc).__name__ == "OCRCancelledError":
            try:
                doc.close()
            except Exception:
                pass
            emit({"type": "file_cancelled", "path": pdf_path})
            raise
        else:
            try:
                doc.close()
            except Exception:
                pass
            fail_elapsed = time.perf_counter() - t_file_start
            emit({
                "type": "file_failed",
                "path": pdf_path,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": fail_elapsed,
            })
            return False
    finally:
        gc.collect()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OCR Engine v4.4.2 worker process — outputs JSON Lines to stdout."
    )
    parser.add_argument("--job-file", required=True, help="Path to job JSON file")
    parser.add_argument("--worker-log", default=None, help="Path for worker stderr log")
    args = parser.parse_args()

    # Redirect stderr to worker log file early
    if args.worker_log:
        try:
            log_fh = open(args.worker_log, "w", encoding="utf-8", errors="replace")
            sys.stderr = log_fh
        except Exception:
            pass

    # Load job
    try:
        with open(args.job_file, encoding="utf-8") as f:
            job = json.load(f)
    except Exception as exc:
        emit({"type": "log", "message": f"[Worker] 無法讀取 job 檔案: {exc}"})
        emit({"type": "batch_completed", "completed": 0, "failed": 0, "cancelled": 0, "skipped": 0})
        sys.exit(1)

    job_id = job.get("job_id", "unknown")
    control_file = job["control_file"]
    checkpoint_dir = Path(job.get("checkpoint_dir", "data/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Emit startup signal BEFORE heavy imports so GUI knows worker is alive
    emit({"type": "worker_started", "pid": os.getpid()})

    # Heavy imports (loads PaddleOCR, CUDA, etc.)
    try:
        from ocr_core import (
            OCRConfig, OCRCancelledError, configure_windows_gpu_dll_paths,
            _find_dll, gpu_runtime_self_test,
        )
        from paddleocr import PaddleOCR
    except Exception as exc:
        emit({"type": "log", "message": f"[Worker] 無法匯入 OCR 核心: {exc}"})
        emit({"type": "batch_completed", "completed": 0, "failed": 0, "cancelled": 0, "skipped": 0})
        sys.exit(1)

    # Build config
    cfg_dict = job["config"]
    try:
        config = OCRConfig(**cfg_dict)
    except Exception as exc:
        emit({"type": "log", "message": f"[Worker] OCRConfig 建立失敗: {exc}"})
        emit({"type": "batch_completed", "completed": 0, "failed": 0, "cancelled": 0, "skipped": 0})
        sys.exit(1)

    # GPU availability check
    use_gpu = config.device.lower() == "gpu"
    if use_gpu:
        try:
            if not _find_dll("cudnn_ops_infer64_8.dll") or not _find_dll("cublasLt64_11.dll"):
                emit_log("[GPU] 缺少 DLL，改用 CPU")
                use_gpu = False
            else:
                ok, detail = gpu_runtime_self_test()
                if not ok:
                    emit_log(f"[GPU] 測試失敗，改用 CPU: {detail[:200]}")
                    use_gpu = False
        except Exception as exc:
            emit_log(f"[GPU] 檢查失敗，改用 CPU: {exc}")
            use_gpu = False

    emit({
        "type": "environment",
        "pid": os.getpid(),
        "python": sys.version.split()[0],
        "device": "gpu" if use_gpu else "cpu",
    })

    # Initialize PaddleOCR once — redirect stdout so Paddle warnings don't pollute JSON
    try:
        with contextlib.redirect_stdout(sys.stderr):
            ocr_instance = PaddleOCR(
                lang="chinese_cht",
                use_gpu=use_gpu,
                use_angle_cls=True,
                det_db_thresh=0.2,
                det_db_unclip_ratio=1.8,
                show_log=False,
                enable_mkldnn=not use_gpu,
            )
    except Exception as exc:
        emit({"type": "log", "message": f"[Worker] PaddleOCR 初始化失敗: {exc}"})
        emit({"type": "batch_completed", "completed": 0, "failed": 0, "cancelled": 0, "skipped": 0})
        sys.exit(1)

    pdf_files = job["pdf_files"]
    resume = job.get("resume_from_checkpoint", True)

    completed = 0
    failed = 0
    cancelled = 0

    try:
        for file_index, pdf_path in enumerate(pdf_files):
            try:
                ok = process_pdf_with_checkpoint(
                    pdf_path, config, ocr_instance, checkpoint_dir,
                    control_file, file_index, len(pdf_files), OCRCancelledError
                )
                if ok:
                    completed += 1
                else:
                    failed += 1
                    if not config.batch_failure_continue:
                        break
            except OCRCancelledError:
                cancelled += 1
                break
            except Exception as exc:
                emit_log(f"[錯誤] {os.path.basename(pdf_path)}: {exc}")
                failed += 1
                if not config.batch_failure_continue:
                    break
    except OCRCancelledError:
        cancelled = len(pdf_files) - completed - failed

    skipped = 0
    emit({
        "type": "batch_completed",
        "completed": completed,
        "failed": failed,
        "cancelled": cancelled,
        "skipped": skipped,
    })

    gc.collect()
    if cancelled > 0:
        sys.exit(3)
    if failed > 0:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
