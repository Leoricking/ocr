# Changelog — OCR Engine v4.5.4

## New Features

### Multi-PDF Selection (Part A)
- Added **選擇單一 PDF** button (replaces old 選擇 PDF)
- Added **選擇多個 PDF** button using `filedialog.askopenfilenames`
- Kept **選擇資料夾** for directory-based batch loading
- Source buttons reorganised into two rows:
  - Row 0: 選擇單一 PDF | 選擇多個 PDF
  - Row 1: 選擇資料夾 | 清除來源 | 重新掃描

### Shared Dedup Logic (`_add_pdf_paths_to_batch`)
- Normalises paths via `Path.resolve().casefold()` for cross-platform dedup
- Filters out `_OCR.pdf`, `_OCR_OCR.pdf`, `_searchable.pdf`, `.part` files
- Skips files inside the configured output folder
- Skips paths already in the batch list
- Skips duplicates within the same selection call
- Gets page count via fitz/PyMuPDF; marks as `"?"` on failure without aborting
- Persists `last_input_directory` in settings after each selection

### Per-File and Batch Elapsed Time (Part B)
- New **花費時間** column in the Treeview (between 平均秒/頁 and 輸出PDF)
- Displayed as `HH:MM:SS` (`_format_duration` helper)
- File elapsed time starts at `file_started` event, freezes at `file_completed` / `file_failed`
- Batch elapsed timer runs every 500 ms (`TIMER_REFRESH_MS`)
- Only one `root.after` chain active at a time (stored in `_timer_id`)
- Timer cancelled on batch completion, pause, and window close

### Progress Section Labels (Part B4)
- 目前檔案花費：HH:MM:SS
- 目前檔案剩餘：HH:MM:SS (linear estimate from page throughput)
- 整批花費時間：HH:MM:SS
- 整批預估剩餘：HH:MM:SS
- 預計完成：HH:MM:SS (wall-clock ETA)

### Worker Events (Part C — `ocr_worker.py`)
- `file_started` now includes `started_monotonic`
- `page_progress` now includes `elapsed_seconds` (time since file started)
- `file_completed` now includes `elapsed_seconds`
- `file_failed` now includes `elapsed_seconds`
- `batch_completed` now includes `skipped` count

### Completion Log Messages (Part D)
After each file:
```
[完成] <filename>
  頁數：N
  花費時間：HH:MM:SS
  平均速度：X.XX 秒／頁
```
After batch:
```
[批次完成]
  總檔案：N
  完成：N  失敗：N  取消：N  跳過：N
  整批花費時間：HH:MM:SS
  總處理頁數：N
  平均每頁：X.XX 秒
```

## Settings
- `last_input_directory` added to `settings_manager.py` DEFAULTS

## Version
- Version string updated to `v4.5.4`
- GUI title: `OCR Engine v4.5.4 — PDF 搜尋化與品質分析系統`

## Tests Added
- `tests/test_multi_pdf_selection.py` — selection modes and dedup
- `tests/test_batch_add_paths.py` — folder scanning and edge cases
- `tests/test_elapsed_time.py` — `_format_duration` and per-file timing
- `tests/test_gui_elapsed_time.py` — Treeview column and timer management

## Preserved
- CLI interface unchanged
- PaddleOCR GPU/CPU fallback unchanged
- Worker subprocess architecture unchanged
- UTF-8 output unchanged
- Output modes (all/pdf/txt) unchanged
- Proofreading/Claude integration unchanged
- Memory guard and checkpoint/resume unchanged
- Pause/resume/cancel unchanged
- Settings persistence unchanged
- Existing source filtering (`_is_source_pdf`) unchanged
