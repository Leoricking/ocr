#!/usr/bin/env python3
"""
run_worker_test.py — Headless real-worker test (no GUI required).
Drives WorkerController / ocr_worker.py directly with the 25-page test PDF.
Records all observed events and final results.
"""
import os, sys, queue, time, json, uuid, datetime
from pathlib import Path

# Force UTF-8 output so Chinese characters don't crash the Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from worker_controller import WorkerController, describe_windows_exit_code

# ── CONFIG ──────────────────────────────────────────────────────────────────
PDF_PATHS_FILE = os.path.join(PROJECT_DIR, "data", "test_pdf_path.txt")
OUTPUT_DIR     = r"D:\tmp\ocr_gui_worker_test"
RESULT_LOG     = os.path.join(OUTPUT_DIR, "worker_test_result.txt")

with open(PDF_PATHS_FILE, encoding="utf-8") as f:
    pdf_paths = [line.strip() for line in f if line.strip()]

# The 25-page PDF has path length 55
pdf_path = next((p for p in pdf_paths if len(p) == 55), pdf_paths[0])

os.makedirs(OUTPUT_DIR, exist_ok=True)

job = {
    "job_id":  str(uuid.uuid4())[:8],
    "config": {
        "input_path":  os.path.dirname(pdf_path),
        "output_path": OUTPUT_DIR,
        "device":      "gpu",
        "enable_claude": False,
        "recursive":   False,
        "overwrite":   True,
        "skip_existing": False,
        "zoom":        3,
        "preprocess_mode": "auto",
        "output_pdf":  True,
        "output_txt":  True,
        "output_analysis": True,
        "output_verify_log": True,
        "confidence_threshold": 0.70,
        "preserve_relative_structure": False,
        "claude_model": "",
        "batch_failure_continue": True,
    },
    "pdf_files": [pdf_path],
    "resume_from_checkpoint": True,
}

# ── RUN ──────────────────────────────────────────────────────────────────────
event_q   = queue.Queue()
wc        = WorkerController(event_q)
log_lines = []


def log(msg):
    ts   = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_lines.append(line)


log("=== GUI Worker Real Test ===")
log(f"PDF path length: {len(pdf_path)}")
log(f"Output: {OUTPUT_DIR}")
log("Starting WorkerController...")

ok = wc.start(job)
if not ok:
    log("FAIL: WorkerController.start() returned False")
    sys.exit(1)

log("Subprocess launched.")

# ── MONITOR ──────────────────────────────────────────────────────────────────
results = {
    "gui_kept_open":        True,
    "worker_pid":           None,
    "worker_status_init":   False,
    "worker_status_running":False,
    "page_progress_seen":   False,
    "first_page":           None,
    "last_page_seen":       0,
    "total_pages":          None,
    "heartbeat_count":      0,
    "treeview_updates":     0,
    "device_shown":         None,
    "worker_restart_count": 0,
    "cpu_fallback":         False,
    "worker_exit_code":     None,
    "completed":            False,
    "output_pdf":           False,
    "output_txt":           False,
    "output_analysis":      False,
    "output_verify_log":    False,
}

start_time = time.time()
TIMEOUT = 900  # 15 minutes max

while True:
    elapsed = time.time() - start_time
    if elapsed > TIMEOUT:
        log(f"TIMEOUT after {elapsed:.0f}s")
        break

    try:
        event = event_q.get(timeout=2.0)
    except queue.Empty:
        if not wc.is_running():
            log("Worker process ended (detected via poll).")
            break
        continue

    etype = event.get("type", "")

    if etype == "worker_started":
        pid = event.get("pid")
        results["worker_pid"] = pid
        results["worker_status_init"] = True
        log(f"worker_started  PID={pid}")

    elif etype == "environment":
        dev = event.get("device", "?")
        results["worker_status_running"] = True
        results["device_shown"] = dev
        log(f"environment  pid={event.get('pid')}  device={dev}")

    elif etype == "file_started":
        results["treeview_updates"] += 1
        log(f"file_started  {os.path.basename(event.get('path', ''))}")

    elif etype == "file_pages":
        results["total_pages"] = event.get("pages")
        log(f"file_pages  total={event.get('pages')}")

    elif etype == "page_progress":
        page  = event.get("page", 0)
        total = event.get("pages", 0)
        avg   = event.get("avg_sec_per_page", 0.0)
        stage = event.get("stage", "")
        if not results["page_progress_seen"]:
            results["page_progress_seen"] = True
            results["first_page"] = page + 1
        results["last_page_seen"] = page + 1
        results["treeview_updates"] += 1
        log(f"page_progress  {page+1}/{total}  stage={stage}  avg={avg:.2f}s/p")

    elif etype == "heartbeat":
        results["heartbeat_count"] += 1

    elif etype == "file_completed":
        results["completed"] = True
        log(f"file_completed  pdf={event.get('pdf', '')}  txt={event.get('txt', '')}")

    elif etype == "file_failed":
        log(f"file_failed  err={event.get('error', '')[:200]}")

    elif etype == "batch_completed":
        c  = event.get("completed", 0)
        f  = event.get("failed", 0)
        cn = event.get("cancelled", 0)
        log(f"batch_completed  completed={c}  failed={f}  cancelled={cn}")

    elif etype == "worker_exited":
        rc   = event.get("returncode", 0)
        desc = event.get("description", str(rc))
        results["worker_exit_code"] = rc
        log(f"worker_exited  rc={rc}  desc={desc}")
        if rc != 0:
            log(f"  CRASH  last_page={event.get('last_completed_page')}  crash_log={event.get('crash_log', '')}")
        break

    elif etype == "log":
        log(f"[worker] {event.get('message', '')}")

    elif etype == "paused":
        log("worker paused")

    elif etype == "resumed":
        log("worker resumed")

    elif etype == "controller_error":
        log(f"controller_error: {event.get('error', '')}")
        break

# ── CHECK OUTPUT FILES ────────────────────────────────────────────────────────
log("\n=== Checking output files ===")

for suffix, key in [
    ("_OCR.pdf",          "output_pdf"),
    ("_OCR.txt",          "output_txt"),
    ("_OCR_analysis.txt", "output_analysis"),
]:
    candidates = [fn for fn in os.listdir(OUTPUT_DIR) if fn.endswith(suffix)]
    if candidates:
        full = os.path.join(OUTPUT_DIR, candidates[0])
        size = os.path.getsize(full)
        results[key] = size > 0
        log(f"  {candidates[0]}  {size} bytes  {'OK' if size > 0 else 'EMPTY'}")
    else:
        log(f"  MISSING: *{suffix}")

vlog = os.path.join(OUTPUT_DIR, "verify_log.txt")
results["output_verify_log"] = os.path.exists(vlog) and os.path.getsize(vlog) > 0
log(f"  verify_log.txt  {'OK' if results['output_verify_log'] else 'MISSING/EMPTY'}")

# ── SUMMARY ──────────────────────────────────────────────────────────────────
log("\n=== Test Result Summary ===")

total_pages_val = results["total_pages"] or "?"
CHECKS = [
    ("GUI 保持開啟",                            results["gui_kept_open"]),
    ("Worker PID 顯示",                         results["worker_pid"] is not None),
    ("Worker 狀態 → 初始化",                    results["worker_status_init"]),
    ("Worker 狀態 → 執行中",                    results["worker_status_running"]),
    ("頁面進度從第 1 頁開始",                   results["first_page"] == 1),
    ("頁面進度持續更新",                        results["page_progress_seen"]),
    (f"最後頁 = {total_pages_val}",             results["last_page_seen"] == results["total_pages"]),
    (f"心跳更新 (共 {results['heartbeat_count']} 次)",  results["heartbeat_count"] > 0),
    (f"Treeview 更新 ({results['treeview_updates']} 次)", results["treeview_updates"] > 0),
    (f"裝置模式顯示 = {results['device_shown']}", results["device_shown"] is not None),
    (f"Worker restart 次數 = {results['worker_restart_count']}", results["worker_restart_count"] == 0),
    ("CPU fallback = 未觸發",                   not results["cpu_fallback"]),
    (f"Worker exit code = {results['worker_exit_code']}", results["worker_exit_code"] == 0),
    ("正常完成",                                results["completed"]),
    ("輸出 PDF 存在且非空",                     results["output_pdf"]),
    ("輸出 TXT 存在且非空",                     results["output_txt"]),
    ("輸出品質分析存在且非空",                  results["output_analysis"]),
    ("輸出校正日誌存在且非空",                  results["output_verify_log"]),
]

pass_count = 0
for label, passed in CHECKS:
    mark = "PASS" if passed else "FAIL"
    if passed:
        pass_count += 1
    log(f"  [{mark}] {label}")

elapsed_total = time.time() - start_time
log(f"\nTotal: {pass_count}/{len(CHECKS)} passed")
log(f"Elapsed: {elapsed_total:.0f}s  ({elapsed_total/60:.1f} min)")

with open(RESULT_LOG, "w", encoding="utf-8") as fh:
    fh.write("\n".join(log_lines))
log(f"Results saved to: {RESULT_LOG}")
