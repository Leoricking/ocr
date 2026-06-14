# ocr_gui.py — Full tkinter GUI for OCR Engine v4.4.0
# Window title: OCR Engine v4.4.0 — PDF 搜尋化與品質分析系統

import os
import sys
import queue
import threading
import time
import subprocess
import datetime
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# DPI awareness (Windows only, must be before Tk())
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Lazy import of OCRProcessor so GUI starts instantly without loading Paddle
# ---------------------------------------------------------------------------
_ocr_core_loaded = False
_OCRConfig = None
_OCRProcessor = None
_OCRCancelledError = None


def _ensure_ocr_core():
    global _ocr_core_loaded, _OCRConfig, _OCRProcessor, _OCRCancelledError
    if not _ocr_core_loaded:
        from ocr_core import OCRConfig, OCRProcessor, OCRCancelledError
        _OCRConfig = OCRConfig
        _OCRProcessor = OCRProcessor
        _OCRCancelledError = OCRCancelledError
        _ocr_core_loaded = True


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VERSION = "v4.4.0"
TITLE = f"OCR Engine {VERSION} — PDF 搜尋化與品質分析系統"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")

STATUS_COLORS = {
    "等待": "#888888",
    "初始化": "#1976D2",
    "OCR 中": "#0288D1",
    "Claude 校對中": "#7B1FA2",
    "寫入 PDF": "#00796B",
    "完成": "#2E7D32",
    "跳過": "#F57F17",
    "已取消": "#E65100",
    "失敗": "#B71C1C",
}

PREPROCESS_LABELS = {
    "自動選最佳": "auto",
    "原圖": "original",
    "CLAHE": "clahe",
    "BINARY": "binary",
}
PREPROCESS_LABELS_REV = {v: k for k, v in PREPROCESS_LABELS.items()}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _open_path(path: str):
    """Open a file or directory with the OS default application."""
    if not os.path.exists(path):
        return
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def _count_pdf_pages(path: str) -> int:
    try:
        import fitz
        doc = fitz.open(path)
        n = len(doc)
        doc.close()
        return n
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# OCRGuiApp
# ---------------------------------------------------------------------------
class OCRGuiApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(TITLE)
        self.root.minsize(1100, 720)

        # Load settings
        from settings_manager import SettingsManager
        self._sm = SettingsManager()
        self._settings = self._sm.load()

        # Apply saved geometry
        geom = self._settings.get("window_geometry", "1280x820+100+80")
        try:
            self.root.geometry(geom)
        except Exception:
            self.root.geometry("1280x820+100+80")

        # Threading primitives
        self._pause_event = threading.Event()   # set = paused
        self._cancel_event = threading.Event()
        self._worker_thread = None
        self._task_queue = queue.Queue()

        # State
        self._batch_items = {}   # file_path -> {index, pages, status, progress, avg_sec, out_pdf, out_txt, out_analysis}
        self._batch_order = []   # ordered list of paths
        self._log_lines = []
        self._log_file_handle = None
        self._running = False

        os.makedirs(LOGS_DIR, exist_ok=True)
        self._open_log_file()

        self._build_ui()
        self._apply_settings_to_ui()
        self._start_queue_poll()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -----------------------------------------------------------------------
    # Log file
    # -----------------------------------------------------------------------
    def _open_log_file(self):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(LOGS_DIR, f"ocr_gui_{ts}.log")
        try:
            self._log_file_handle = open(log_path, "w", encoding="utf-8")
        except Exception:
            self._log_file_handle = None

    def _write_log_file(self, msg: str):
        if self._log_file_handle:
            try:
                self._log_file_handle.write(msg + "\n")
                self._log_file_handle.flush()
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # UI Build
    # -----------------------------------------------------------------------
    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=0)
        self.root.rowconfigure(3, weight=0)
        self.root.rowconfigure(4, weight=0)

        self._build_source_section()
        self._build_main_area()
        self._build_progress_section()
        self._build_log_section()
        self._build_bottom_bar()

    # ---- Source/Output section ----
    def _build_source_section(self):
        frm = ttk.LabelFrame(self.root, text="來源與輸出", padding=6)
        frm.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 2))
        frm.columnconfigure(1, weight=1)

        # Row 1: Source
        ttk.Label(frm, text="來源：").grid(row=0, column=0, sticky="w", padx=(0, 4))
        self._var_input = tk.StringVar()
        ttk.Entry(frm, textvariable=self._var_input).grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Button(frm, text="選擇 PDF", command=self._browse_pdf).grid(row=0, column=2, padx=2)
        ttk.Button(frm, text="選擇資料夾", command=self._browse_input_folder).grid(row=0, column=3, padx=2)
        ttk.Button(frm, text="清除來源", command=self._clear_input).grid(row=0, column=4, padx=2)

        # Row 2: Output
        ttk.Label(frm, text="輸出：").grid(row=1, column=0, sticky="w", padx=(0, 4), pady=(4, 0))
        self._var_output = tk.StringVar()
        ttk.Entry(frm, textvariable=self._var_output).grid(row=1, column=1, sticky="ew", padx=2, pady=(4, 0))
        ttk.Button(frm, text="選擇輸出資料夾", command=self._browse_output_folder).grid(row=1, column=2, padx=2, pady=(4, 0))
        ttk.Button(frm, text="開啟輸出資料夾", command=self._open_output_folder).grid(row=1, column=3, padx=2, pady=(4, 0))

        # Row 3: Recursive
        self._var_recursive = tk.BooleanVar()
        ttk.Checkbutton(frm, text="包含子資料夾", variable=self._var_recursive).grid(
            row=2, column=1, sticky="w", pady=(2, 0)
        )

    # ---- Main area (settings left, batch list center) ----
    def _build_main_area(self):
        pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pane.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)

        # Left settings panel
        left = ttk.Frame(pane, width=280)
        left.pack_propagate(False)
        pane.add(left, weight=0)
        self._build_settings_panel(left)

        # Center batch list
        center = ttk.Frame(pane)
        pane.add(center, weight=1)
        self._build_batch_panel(center)

    def _build_settings_panel(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        canvas = tk.Canvas(parent, borderwidth=0, highlightthickness=0)
        scroll = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

        inner = ttk.Frame(canvas)
        inner_window = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(inner_window, width=canvas.winfo_width())

        inner.bind("<Configure>", _on_configure)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(inner_window, width=e.width))

        self._build_settings_inner(inner)

    def _build_settings_inner(self, parent):
        row = 0

        # OCR Settings LabelFrame
        ocr_frm = ttk.LabelFrame(parent, text="OCR 設定", padding=8)
        ocr_frm.grid(row=row, column=0, sticky="ew", padx=4, pady=4)
        ocr_frm.columnconfigure(1, weight=1)
        row += 1

        r = 0
        # Device
        ttk.Label(ocr_frm, text="運算裝置：").grid(row=r, column=0, sticky="w")
        dev_frm = ttk.Frame(ocr_frm)
        dev_frm.grid(row=r, column=1, sticky="w")
        self._var_device = tk.StringVar(value="gpu")
        ttk.Radiobutton(dev_frm, text="GPU", variable=self._var_device, value="gpu").pack(side="left")
        ttk.Radiobutton(dev_frm, text="CPU", variable=self._var_device, value="cpu").pack(side="left", padx=(8, 0))
        r += 1

        # GPU status label
        self._lbl_gpu_status = ttk.Label(ocr_frm, text="GPU 狀態：尚未測試", foreground="#666666", wraplength=200)
        self._lbl_gpu_status.grid(row=r, column=0, columnspan=2, sticky="w", pady=(2, 0))
        r += 1

        # Test GPU button
        ttk.Button(ocr_frm, text="測試 GPU", command=self._test_gpu).grid(
            row=r, column=0, columnspan=2, sticky="w", pady=(2, 4)
        )
        r += 1

        # Separator
        ttk.Separator(ocr_frm, orient="horizontal").grid(row=r, column=0, columnspan=2, sticky="ew", pady=4)
        r += 1

        # Claude
        self._var_claude = tk.BooleanVar()
        ttk.Checkbutton(ocr_frm, text="啟用 Claude 校對", variable=self._var_claude).grid(
            row=r, column=0, columnspan=2, sticky="w"
        )
        r += 1

        self._lbl_api_status = ttk.Label(ocr_frm, text="API Key：未設定", foreground="#B71C1C", wraplength=200)
        self._lbl_api_status.grid(row=r, column=0, columnspan=2, sticky="w")
        self._update_api_key_label()
        r += 1

        ttk.Label(ocr_frm, text="Claude 模型：").grid(row=r, column=0, sticky="w")
        self._var_claude_model = tk.StringVar()
        ttk.Entry(ocr_frm, textvariable=self._var_claude_model, width=22).grid(row=r, column=1, sticky="ew")
        r += 1

        ttk.Button(ocr_frm, text="測試 Claude API", command=self._test_claude_api).grid(
            row=r, column=0, columnspan=2, sticky="w", pady=(2, 4)
        )
        r += 1

        ttk.Separator(ocr_frm, orient="horizontal").grid(row=r, column=0, columnspan=2, sticky="ew", pady=4)
        r += 1

        # Zoom
        ttk.Label(ocr_frm, text="縮放倍率：").grid(row=r, column=0, sticky="w")
        self._var_zoom = tk.StringVar(value="3")
        ttk.Combobox(ocr_frm, textvariable=self._var_zoom, values=["2", "3", "4"],
                     state="readonly", width=6).grid(row=r, column=1, sticky="w")
        r += 1

        # Preprocess
        ttk.Label(ocr_frm, text="預處理模式：").grid(row=r, column=0, sticky="w")
        self._var_preprocess = tk.StringVar(value="自動選最佳")
        ttk.Combobox(
            ocr_frm, textvariable=self._var_preprocess,
            values=list(PREPROCESS_LABELS.keys()), state="readonly", width=12
        ).grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Separator(ocr_frm, orient="horizontal").grid(row=r, column=0, columnspan=2, sticky="ew", pady=4)
        r += 1

        # Output options
        ttk.Label(ocr_frm, text="輸出選項：").grid(row=r, column=0, sticky="w")
        r += 1
        self._var_out_pdf = tk.BooleanVar(value=True)
        self._var_out_txt = tk.BooleanVar(value=True)
        self._var_out_analysis = tk.BooleanVar(value=True)
        self._var_out_verify = tk.BooleanVar(value=True)
        self._var_preserve = tk.BooleanVar(value=True)

        for var, label in [
            (self._var_out_pdf, "輸出 PDF"),
            (self._var_out_txt, "輸出 TXT"),
            (self._var_out_analysis, "輸出品質分析"),
            (self._var_out_verify, "輸出校正日誌"),
            (self._var_preserve, "保留相對資料夾結構"),
        ]:
            ttk.Checkbutton(ocr_frm, text=label, variable=var).grid(
                row=r, column=0, columnspan=2, sticky="w"
            )
            r += 1

        ttk.Separator(ocr_frm, orient="horizontal").grid(row=r, column=0, columnspan=2, sticky="ew", pady=4)
        r += 1

        # Overwrite / skip existing (mutually exclusive)
        self._var_overwrite = tk.BooleanVar()
        self._var_skip_existing = tk.BooleanVar(value=True)
        cb_overwrite = ttk.Checkbutton(
            ocr_frm, text="覆蓋已有輸出", variable=self._var_overwrite,
            command=self._on_overwrite_changed
        )
        cb_overwrite.grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1
        cb_skip = ttk.Checkbutton(
            ocr_frm, text="跳過已有輸出", variable=self._var_skip_existing,
            command=self._on_skip_existing_changed
        )
        cb_skip.grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1

        ttk.Separator(ocr_frm, orient="horizontal").grid(row=r, column=0, columnspan=2, sticky="ew", pady=4)
        r += 1

        # Confidence threshold
        ttk.Label(ocr_frm, text="置信度門檻：").grid(row=r, column=0, sticky="w")
        self._var_conf = tk.DoubleVar(value=0.70)
        self._lbl_conf_val = ttk.Label(ocr_frm, text="0.70")
        self._lbl_conf_val.grid(row=r, column=1, sticky="w")
        r += 1
        conf_scale = ttk.Scale(
            ocr_frm, from_=0.50, to=0.95, variable=self._var_conf, orient="horizontal",
            command=lambda v: self._lbl_conf_val.config(text=f"{float(v):.2f}")
        )
        conf_scale.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        r += 1

        # Reset defaults button
        ttk.Button(parent, text="恢復預設設定", command=self._reset_defaults).grid(
            row=row, column=0, sticky="ew", padx=4, pady=(0, 4)
        )
        row += 1

    def _build_batch_panel(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=0)
        parent.rowconfigure(1, weight=1)

        # Toolbar
        toolbar = ttk.Frame(parent)
        toolbar.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 2))
        for text, cmd in [
            ("全選", self._select_all),
            ("取消全選", self._deselect_all),
            ("移除選取", self._remove_selected),
            ("清空清單", self._clear_batch),
            ("重新掃描", self._rescan_batch),
            ("重試失敗", self._retry_failed),
        ]:
            ttk.Button(toolbar, text=text, command=cmd).pack(side="left", padx=2)

        # Treeview
        batch_frm = ttk.LabelFrame(parent, text="批次清單", padding=4)
        batch_frm.grid(row=1, column=0, sticky="nsew", padx=4, pady=2)
        batch_frm.columnconfigure(0, weight=1)
        batch_frm.rowconfigure(0, weight=1)

        cols = ("編號", "檔名", "頁數", "狀態", "進度", "平均秒/頁", "輸出PDF", "輸出TXT", "分析檔")
        self._tree = ttk.Treeview(batch_frm, columns=cols, show="headings", selectmode="extended")
        col_widths = [40, 240, 50, 80, 80, 80, 80, 80, 80]
        for col, w in zip(cols, col_widths):
            self._tree.heading(col, text=col)
            self._tree.column(col, width=w, minwidth=30)

        vsb = ttk.Scrollbar(batch_frm, orient="vertical", command=self._tree.yview)
        hsb = ttk.Scrollbar(batch_frm, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        self._tree.bind("<Double-1>", self._on_tree_double_click)
        self._tree.bind("<Button-3>", self._on_tree_right_click)

        # Context menu
        self._ctx_menu = tk.Menu(self.root, tearoff=0)
        self._ctx_menu.add_command(label="開啟來源 PDF", command=self._open_selected_source)
        self._ctx_menu.add_command(label="開啟輸出 PDF", command=self._open_selected_output_pdf)
        self._ctx_menu.add_command(label="開啟 TXT", command=self._open_selected_output_txt)
        self._ctx_menu.add_separator()
        self._ctx_menu.add_command(label="移除", command=self._remove_selected)
        self._ctx_menu.add_command(label="重試", command=self._retry_selected)

    def _build_progress_section(self):
        frm = ttk.LabelFrame(self.root, text="進度", padding=6)
        frm.grid(row=2, column=0, sticky="ew", padx=8, pady=2)
        frm.columnconfigure(1, weight=1)

        # Current file
        ttk.Label(frm, text="目前檔案：").grid(row=0, column=0, sticky="w")
        self._lbl_current_file = ttk.Label(frm, text="—")
        self._lbl_current_file.grid(row=0, column=1, sticky="w")

        self._pbar_file = ttk.Progressbar(frm, mode="determinate", length=400)
        self._pbar_file.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(2, 0))

        self._lbl_page_info = ttk.Label(frm, text="")
        self._lbl_page_info.grid(row=2, column=0, columnspan=3, sticky="w")

        # Overall
        ttk.Label(frm, text="整體進度：").grid(row=3, column=0, sticky="w", pady=(4, 0))
        self._pbar_overall = ttk.Progressbar(frm, mode="determinate", length=400)
        self._pbar_overall.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(2, 0))

        self._lbl_overall_info = ttk.Label(frm, text="")
        self._lbl_overall_info.grid(row=5, column=0, columnspan=3, sticky="w")

    def _build_log_section(self):
        log_frm = ttk.LabelFrame(self.root, text="執行日誌", padding=4)
        log_frm.grid(row=3, column=0, sticky="ew", padx=8, pady=2)
        log_frm.columnconfigure(0, weight=1)
        log_frm.rowconfigure(0, weight=1)

        self._log_text = scrolledtext.ScrolledText(
            log_frm, height=8, state="disabled",
            font=("Consolas", 9), wrap="word"
        )
        self._log_text.grid(row=0, column=0, sticky="nsew")

        btn_frm = ttk.Frame(log_frm)
        btn_frm.grid(row=1, column=0, sticky="w", pady=(4, 0))
        for text, cmd in [
            ("清除日誌", self._clear_log),
            ("複製日誌", self._copy_log),
            ("儲存日誌", self._save_log),
            ("開啟log資料夾", lambda: _open_path(LOGS_DIR)),
        ]:
            ttk.Button(btn_frm, text=text, command=cmd).pack(side="left", padx=2)

    def _build_bottom_bar(self):
        bar = ttk.Frame(self.root, relief="groove", padding=4)
        bar.grid(row=4, column=0, sticky="ew", padx=8, pady=(2, 8))

        self._btn_start = ttk.Button(bar, text="開始批次 OCR", command=self._start_batch)
        self._btn_start.pack(side="left", padx=4)

        self._btn_pause = ttk.Button(bar, text="暫停", command=self._pause_batch, state="disabled")
        self._btn_pause.pack(side="left", padx=2)

        self._btn_resume = ttk.Button(bar, text="繼續", command=self._resume_batch, state="disabled")
        self._btn_resume.pack(side="left", padx=2)

        self._btn_cancel = ttk.Button(bar, text="取消目前任務", command=self._cancel_current, state="disabled")
        self._btn_cancel.pack(side="left", padx=2)

        self._btn_cancel_all = ttk.Button(bar, text="全部取消", command=self._cancel_all, state="disabled")
        self._btn_cancel_all.pack(side="left", padx=2)

        self._btn_retry = ttk.Button(bar, text="重試失敗", command=self._retry_failed)
        self._btn_retry.pack(side="left", padx=2)

        ttk.Button(bar, text="開啟輸出資料夾", command=self._open_output_folder).pack(side="left", padx=2)
        ttk.Button(bar, text="關閉", command=self._on_close).pack(side="right", padx=4)

    # -----------------------------------------------------------------------
    # Settings apply / collect
    # -----------------------------------------------------------------------
    def _apply_settings_to_ui(self):
        s = self._settings
        self._var_input.set(s.get("last_input_path", ""))
        self._var_output.set(s.get("last_output_path", ""))
        self._var_recursive.set(s.get("recursive", False))
        self._var_device.set(s.get("device", "gpu"))
        self._var_claude.set(s.get("enable_claude", False))
        self._var_claude_model.set(s.get("claude_model", ""))
        self._var_zoom.set(str(s.get("zoom", 3)))
        preprocess_val = s.get("preprocess_mode", "auto")
        self._var_preprocess.set(PREPROCESS_LABELS_REV.get(preprocess_val, "自動選最佳"))
        self._var_out_pdf.set(s.get("output_pdf", True))
        self._var_out_txt.set(s.get("output_txt", True))
        self._var_out_analysis.set(s.get("output_analysis", True))
        self._var_out_verify.set(s.get("output_verify_log", True))
        self._var_preserve.set(s.get("preserve_relative_structure", True))
        self._var_overwrite.set(s.get("overwrite", False))
        self._var_skip_existing.set(s.get("skip_existing", True))
        conf = s.get("confidence_threshold", 0.70)
        self._var_conf.set(conf)
        self._lbl_conf_val.config(text=f"{conf:.2f}")

    def _collect_settings(self) -> dict:
        s = dict(self._settings)
        s["last_input_path"] = self._var_input.get()
        s["last_output_path"] = self._var_output.get()
        s["recursive"] = self._var_recursive.get()
        s["device"] = self._var_device.get()
        s["enable_claude"] = self._var_claude.get()
        s["claude_model"] = self._var_claude_model.get()
        s["zoom"] = int(self._var_zoom.get())
        s["preprocess_mode"] = PREPROCESS_LABELS.get(self._var_preprocess.get(), "auto")
        s["output_pdf"] = self._var_out_pdf.get()
        s["output_txt"] = self._var_out_txt.get()
        s["output_analysis"] = self._var_out_analysis.get()
        s["output_verify_log"] = self._var_out_verify.get()
        s["preserve_relative_structure"] = self._var_preserve.get()
        s["overwrite"] = self._var_overwrite.get()
        s["skip_existing"] = self._var_skip_existing.get()
        s["confidence_threshold"] = round(self._var_conf.get(), 2)
        try:
            s["window_geometry"] = self.root.geometry()
        except Exception:
            pass
        return s

    def _reset_defaults(self):
        self._settings = self._sm.reset()
        self._apply_settings_to_ui()
        self._append_log("[設定] 已恢復預設設定。")

    # -----------------------------------------------------------------------
    # API Key label
    # -----------------------------------------------------------------------
    def _update_api_key_label(self):
        key = os.environ.get("CLAUDE_API_KEY", "")
        if key and key != "YOUR_CLAUDE_API_KEY":
            short = key[:8] + "..." if len(key) > 8 else key
            self._lbl_api_status.config(text=f"API Key：已設定 ({short})", foreground="#2E7D32")
        else:
            self._lbl_api_status.config(text="API Key：未設定 (請設定 CLAUDE_API_KEY 環境變數)", foreground="#B71C1C")

    # -----------------------------------------------------------------------
    # Browse helpers
    # -----------------------------------------------------------------------
    def _browse_pdf(self):
        path = filedialog.askopenfilename(
            title="選擇 PDF 檔案",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if path:
            self._var_input.set(path)
            self._add_files_to_batch([path])

    def _browse_input_folder(self):
        path = filedialog.askdirectory(title="選擇來源資料夾")
        if path:
            self._var_input.set(path)
            self._scan_folder(path)

    def _browse_output_folder(self):
        path = filedialog.askdirectory(title="選擇輸出資料夾")
        if path:
            self._var_output.set(path)

    def _open_output_folder(self):
        path = self._var_output.get().strip()
        if path and os.path.isdir(path):
            _open_path(path)
        else:
            messagebox.showinfo("提示", "輸出資料夾尚未設定或不存在。")

    def _clear_input(self):
        self._var_input.set("")

    # -----------------------------------------------------------------------
    # Batch list management
    # -----------------------------------------------------------------------
    def _scan_folder(self, folder: str):
        recursive = self._var_recursive.get()
        files = []
        if recursive:
            for root, dirs, fnames in os.walk(folder):
                for fname in sorted(fnames):
                    if fname.lower().endswith(".pdf"):
                        files.append(os.path.join(root, fname))
        else:
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith(".pdf"):
                    files.append(os.path.join(folder, fname))
        self._add_files_to_batch(files)

    def _add_files_to_batch(self, files: list):
        for path in files:
            if path not in self._batch_items:
                idx = len(self._batch_order) + 1
                self._batch_order.append(path)
                self._batch_items[path] = {
                    "index": idx,
                    "pages": 0,
                    "status": "等待",
                    "progress": "",
                    "avg_sec": "",
                    "out_pdf": "",
                    "out_txt": "",
                    "out_analysis": "",
                }
                self._tree_insert(path)
                # Count pages in background
                t = threading.Thread(target=self._count_pages_bg, args=(path,), daemon=True)
                t.start()

    def _count_pages_bg(self, path: str):
        n = _count_pdf_pages(path)
        self.root.after(0, lambda: self._update_tree_pages(path, n))

    def _update_tree_pages(self, path: str, n: int):
        if path in self._batch_items:
            self._batch_items[path]["pages"] = n
            self._tree_refresh_row(path)

    def _tree_insert(self, path: str):
        item = self._batch_items[path]
        self._tree.insert(
            "", "end", iid=path,
            values=(
                item["index"],
                os.path.basename(path),
                item["pages"] or "",
                item["status"],
                item["progress"],
                item["avg_sec"],
                item["out_pdf"],
                item["out_txt"],
                item["out_analysis"],
            )
        )

    def _tree_refresh_row(self, path: str):
        if path not in self._batch_items:
            return
        item = self._batch_items[path]
        try:
            self._tree.item(path, values=(
                item["index"],
                os.path.basename(path),
                item["pages"] or "",
                item["status"],
                item["progress"],
                item["avg_sec"],
                item["out_pdf"],
                item["out_txt"],
                item["out_analysis"],
            ))
            # Color by status
            color = STATUS_COLORS.get(item["status"], "")
            if color:
                self._tree.tag_configure(item["status"], foreground=color)
                self._tree.item(path, tags=(item["status"],))
        except Exception:
            pass

    def _select_all(self):
        self._tree.selection_set(self._batch_order)

    def _deselect_all(self):
        self._tree.selection_remove(self._tree.selection())

    def _remove_selected(self):
        for path in list(self._tree.selection()):
            self._remove_from_batch(path)

    def _remove_from_batch(self, path: str):
        try:
            self._tree.delete(path)
        except Exception:
            pass
        self._batch_items.pop(path, None)
        if path in self._batch_order:
            self._batch_order.remove(path)

    def _clear_batch(self):
        for path in list(self._batch_order):
            try:
                self._tree.delete(path)
            except Exception:
                pass
        self._batch_items.clear()
        self._batch_order.clear()

    def _rescan_batch(self):
        folder = self._var_input.get().strip()
        if folder and os.path.isdir(folder):
            self._clear_batch()
            self._scan_folder(folder)

    def _retry_failed(self):
        for path in self._batch_order:
            if self._batch_items[path]["status"] in ("失敗", "已取消"):
                self._batch_items[path]["status"] = "等待"
                self._tree_refresh_row(path)

    def _retry_selected(self):
        for path in self._tree.selection():
            if path in self._batch_items:
                self._batch_items[path]["status"] = "等待"
                self._tree_refresh_row(path)

    # -----------------------------------------------------------------------
    # Tree double-click / right-click
    # -----------------------------------------------------------------------
    def _on_tree_double_click(self, event):
        region = self._tree.identify("region", event.x, event.y)
        col = self._tree.identify_column(event.x)
        item = self._tree.identify_row(event.y)
        if not item:
            return
        col_idx = int(col.replace("#", "")) - 1
        cols = ("編號", "檔名", "頁數", "狀態", "進度", "平均秒/頁", "輸出PDF", "輸出TXT", "分析檔")
        col_name = cols[col_idx] if col_idx < len(cols) else ""
        path = item  # iid is path
        if col_name in ("輸出PDF",):
            val = self._batch_items.get(path, {}).get("out_pdf", "")
            if val and os.path.exists(val):
                _open_path(val)
        elif col_name in ("輸出TXT",):
            val = self._batch_items.get(path, {}).get("out_txt", "")
            if val and os.path.exists(val):
                _open_path(val)
        elif col_name in ("分析檔",):
            val = self._batch_items.get(path, {}).get("out_analysis", "")
            if val and os.path.exists(val):
                _open_path(val)
        else:
            if os.path.exists(path):
                _open_path(path)

    def _on_tree_right_click(self, event):
        item = self._tree.identify_row(event.y)
        if item:
            self._tree.selection_set(item)
            self._ctx_menu.post(event.x_root, event.y_root)

    def _open_selected_source(self):
        for path in self._tree.selection():
            if os.path.exists(path):
                _open_path(path)

    def _open_selected_output_pdf(self):
        for path in self._tree.selection():
            val = self._batch_items.get(path, {}).get("out_pdf", "")
            if val and os.path.exists(val):
                _open_path(val)

    def _open_selected_output_txt(self):
        for path in self._tree.selection():
            val = self._batch_items.get(path, {}).get("out_txt", "")
            if val and os.path.exists(val):
                _open_path(val)

    # -----------------------------------------------------------------------
    # Overwrite / skip_existing mutual exclusion
    # -----------------------------------------------------------------------
    def _on_overwrite_changed(self):
        if self._var_overwrite.get():
            self._var_skip_existing.set(False)

    def _on_skip_existing_changed(self):
        if self._var_skip_existing.get():
            self._var_overwrite.set(False)

    # -----------------------------------------------------------------------
    # GPU / Claude tests (run in threads so UI doesn't block)
    # -----------------------------------------------------------------------
    def _test_gpu(self):
        self._lbl_gpu_status.config(text="GPU 狀態：測試中...", foreground="#1976D2")
        self.root.update_idletasks()

        def _run():
            try:
                from ocr_core import gpu_runtime_self_test, _find_dll
                cudnn = _find_dll("cudnn_ops_infer64_8.dll")
                cublas = _find_dll("cublasLt64_11.dll")
                if not cudnn or not cublas:
                    missing = []
                    if not cudnn:
                        missing.append("cudnn_ops_infer64_8.dll")
                    if not cublas:
                        missing.append("cublasLt64_11.dll")
                    msg = "DLL 缺少：" + ", ".join(missing)
                    self._task_queue.put(("gpu_test_result", False, msg))
                    return
                ok, detail = gpu_runtime_self_test()
                self._task_queue.put(("gpu_test_result", ok, detail[:200] if detail else ""))
            except Exception as exc:
                self._task_queue.put(("gpu_test_result", False, str(exc)))

        threading.Thread(target=_run, daemon=True).start()

    def _test_claude_api(self):
        self._append_log("[API] 正在測試 Claude API Key...")

        def _run():
            try:
                from ocr_core import validate_key
                model = self._var_claude_model.get().strip()
                ok = validate_key(model)
                self._task_queue.put(("claude_test_result", ok))
            except Exception as exc:
                self._task_queue.put(("claude_test_result", False))
                self._task_queue.put(("log", f"[API] 測試失敗：{exc}"))

        threading.Thread(target=_run, daemon=True).start()

    # -----------------------------------------------------------------------
    # Main OCR batch
    # -----------------------------------------------------------------------
    def _build_ocr_config(self):
        _ensure_ocr_core()
        s = self._collect_settings()
        return _OCRConfig(
            input_path=self._var_input.get().strip(),
            output_path=self._var_output.get().strip(),
            device=self._var_device.get(),
            enable_claude=self._var_claude.get(),
            recursive=self._var_recursive.get(),
            overwrite=self._var_overwrite.get(),
            skip_existing=self._var_skip_existing.get() and not self._var_overwrite.get(),
            zoom=int(self._var_zoom.get()),
            preprocess_mode=PREPROCESS_LABELS.get(self._var_preprocess.get(), "auto"),
            output_pdf=self._var_out_pdf.get(),
            output_txt=self._var_out_txt.get(),
            output_analysis=self._var_out_analysis.get(),
            output_verify_log=self._var_out_verify.get(),
            confidence_threshold=round(self._var_conf.get(), 2),
            preserve_relative_structure=self._var_preserve.get(),
            claude_model=self._var_claude_model.get().strip(),
            batch_failure_continue=True,
        )

    def _start_batch(self):
        if self._running:
            messagebox.showinfo("提示", "批次作業正在執行中。")
            return

        input_path = self._var_input.get().strip()
        output_path = self._var_output.get().strip()

        if not input_path:
            messagebox.showerror("錯誤", "請設定來源路徑。")
            return
        if not output_path:
            messagebox.showerror("錯誤", "請設定輸出路徑。")
            return
        if not os.path.exists(input_path):
            messagebox.showerror("錯誤", f"來源路徑不存在：\n{input_path}")
            return

        # Collect pending files
        pending = [
            p for p in self._batch_order
            if self._batch_items[p]["status"] in ("等待", "失敗", "已取消")
        ]
        if not pending:
            # If batch list is empty, populate from input path
            if not self._batch_order:
                if os.path.isfile(input_path):
                    self._add_files_to_batch([input_path])
                elif os.path.isdir(input_path):
                    self._scan_folder(input_path)
                pending = [
                    p for p in self._batch_order
                    if self._batch_items[p]["status"] in ("等待", "失敗", "已取消")
                ]
            if not pending:
                messagebox.showinfo("提示", "沒有待處理的 PDF。")
                return

        self._cancel_event.clear()
        self._pause_event.clear()
        self._running = True
        self._set_running_buttons(True)

        self._worker_thread = threading.Thread(target=self._worker_main, daemon=True)
        self._worker_thread.start()

    def _worker_main(self):
        try:
            _ensure_ocr_core()
            config = self._build_ocr_config()

            pending_files = [
                p for p in self._batch_order
                if self._batch_items[p]["status"] in ("等待", "失敗", "已取消")
            ]
            file_total = len(pending_files)

            self._task_queue.put(("log", f"[批次] 開始處理 {file_total} 個 PDF..."))

            processor = _OCRProcessor(
                config=config,
                progress_callback=self._on_progress,
                log_callback=lambda msg: self._task_queue.put(("log", msg)),
                file_status_callback=self._on_file_status,
                pause_event=self._pause_event,
                cancel_event=self._cancel_event,
            )
            processor.initialize_engine()

            for file_index, pdf_path in enumerate(pending_files):
                if self._cancel_event.is_set():
                    self._task_queue.put(("file_status", file_index, pdf_path, "已取消", ""))
                    break
                # Update overall progress
                pct = int(file_index / file_total * 100) if file_total else 0
                self._task_queue.put(("overall_progress", file_index, file_total, pct))
                try:
                    processor.process_pdf(pdf_path, file_index=file_index, file_total=file_total)
                    # Record output paths
                    out_dir = processor._resolve_output_path(pdf_path)
                    stem = os.path.splitext(os.path.basename(pdf_path))[0]
                    self._task_queue.put(("set_outputs", pdf_path, out_dir, stem))
                except _OCRCancelledError:
                    self._task_queue.put(("file_status", file_index, pdf_path, "已取消", ""))
                    break
                except Exception as exc:
                    self._task_queue.put(("log", f"[錯誤] {os.path.basename(pdf_path)}：{exc}"))
                    self._task_queue.put(("file_status", file_index, pdf_path, "失敗", str(exc)))
                    if not config.batch_failure_continue:
                        break

            self._task_queue.put(("overall_progress", file_total, file_total, 100))
            self._task_queue.put(("log", "[批次] 全部完成。"))
        except Exception as exc:
            self._task_queue.put(("log", f"[錯誤] 工作執行緒異常：{exc}\n{traceback.format_exc()}"))
        finally:
            self._task_queue.put(("done",))

    def _on_progress(self, prog):
        self._task_queue.put(("progress", prog))

    def _on_file_status(self, file_index, file_path, status, extra_info):
        self._task_queue.put(("file_status", file_index, file_path, status, extra_info))

    # -----------------------------------------------------------------------
    # Pause / Cancel
    # -----------------------------------------------------------------------
    def _pause_batch(self):
        if self._running and not self._pause_event.is_set():
            self._pause_event.set()
            self._btn_pause.config(state="disabled")
            self._btn_resume.config(state="normal")
            self._append_log("[暫停] 已暫停，當前頁面完成後停止。")

    def _resume_batch(self):
        if self._pause_event.is_set():
            self._pause_event.clear()
            self._btn_pause.config(state="normal")
            self._btn_resume.config(state="disabled")
            self._append_log("[繼續] 已繼續。")

    def _cancel_current(self):
        if self._running:
            self._cancel_event.set()
            self._append_log("[取消] 已傳送取消訊號。")

    def _cancel_all(self):
        self._cancel_event.set()
        self._pause_event.clear()
        self._append_log("[取消] 已傳送全部取消訊號。")

    def _set_running_buttons(self, running: bool):
        state_run = "disabled" if running else "normal"
        state_ctrl = "normal" if running else "disabled"
        self._btn_start.config(state=state_run)
        self._btn_pause.config(state=state_ctrl)
        self._btn_cancel.config(state=state_ctrl)
        self._btn_cancel_all.config(state=state_ctrl)
        if not running:
            self._btn_resume.config(state="disabled")

    # -----------------------------------------------------------------------
    # Queue polling
    # -----------------------------------------------------------------------
    def _start_queue_poll(self):
        self.root.after(100, self._poll_queue)

    def _poll_queue(self):
        try:
            while True:
                msg = self._task_queue.get_nowait()
                self._handle_message(msg)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _handle_message(self, msg):
        kind = msg[0]
        if kind == "log":
            self._append_log(msg[1])
        elif kind == "progress":
            self._handle_progress(msg[1])
        elif kind == "file_status":
            _, file_index, file_path, status, extra_info = msg
            self._update_file_status(file_path, status, extra_info)
        elif kind == "overall_progress":
            _, done, total, pct = msg
            self._pbar_overall["value"] = pct
            self._lbl_overall_info.config(text=f"{done}/{total} 個檔案  ({pct}%)")
        elif kind == "set_outputs":
            _, pdf_path, out_dir, stem = msg
            item = self._batch_items.get(pdf_path)
            if item:
                item["out_pdf"] = os.path.join(out_dir, stem + "_OCR.pdf")
                item["out_txt"] = os.path.join(out_dir, stem + "_OCR.txt")
                item["out_analysis"] = os.path.join(out_dir, stem + "_OCR_analysis.txt")
                self._tree_refresh_row(pdf_path)
        elif kind == "gpu_test_result":
            _, ok, detail = msg
            if ok:
                self._lbl_gpu_status.config(text="GPU 狀態：測試通過 ✓", foreground="#2E7D32")
                self._append_log(f"[GPU] 測試通過。{detail}")
            else:
                self._lbl_gpu_status.config(text=f"GPU 狀態：失敗 — {detail[:60]}", foreground="#B71C1C")
                self._append_log(f"[GPU] 測試失敗：{detail}")
        elif kind == "claude_test_result":
            ok = msg[1]
            if ok:
                self._append_log("[API] Claude API Key 驗證成功。")
            else:
                self._append_log("[API] Claude API Key 驗證失敗，請檢查 CLAUDE_API_KEY 環境變數。")
        elif kind == "done":
            self._running = False
            self._pause_event.clear()
            self._cancel_event.clear()
            self._set_running_buttons(False)
            self._append_log("[批次] 工作執行緒結束。")

    def _handle_progress(self, prog):
        from ocr_core import OCRProgress
        fname = os.path.basename(prog.file_path)
        self._lbl_current_file.config(text=fname)
        if prog.page_total > 0:
            pct = int((prog.page_index + 1) / prog.page_total * 100)
            self._pbar_file["value"] = pct
        self._lbl_page_info.config(
            text=(
                f"第 {prog.page_index + 1}/{prog.page_total} 頁  "
                f"階段：{prog.stage}  "
                f"平均：{prog.average_seconds_per_page:.2f}s/頁  "
                f"剩餘：{prog.estimated_remaining_seconds:.0f}s"
            )
        )
        item = self._batch_items.get(prog.file_path)
        if item:
            item["progress"] = f"{prog.page_index + 1}/{prog.page_total}"
            item["avg_sec"] = f"{prog.average_seconds_per_page:.2f}"
            self._tree_refresh_row(prog.file_path)

    def _update_file_status(self, file_path: str, status: str, extra_info: str):
        item = self._batch_items.get(file_path)
        if item:
            item["status"] = status
            self._tree_refresh_row(file_path)
        log_msg = f"[{status}] {os.path.basename(file_path)}"
        if extra_info:
            log_msg += f"  {extra_info}"
        self._append_log(log_msg)

    # -----------------------------------------------------------------------
    # Log helpers
    # -----------------------------------------------------------------------
    def _append_log(self, msg: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self._log_lines.append(line)
        self._write_log_file(line)
        try:
            self._log_text.config(state="normal")
            self._log_text.insert("end", line + "\n")
            self._log_text.see("end")
            self._log_text.config(state="disabled")
        except Exception:
            pass

    def _clear_log(self):
        self._log_text.config(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.config(state="disabled")
        self._log_lines.clear()

    def _copy_log(self):
        text = "\n".join(self._log_lines)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def _save_log(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="儲存日誌",
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write("\n".join(self._log_lines))
                self._append_log(f"[日誌] 已儲存至 {path}")
            except Exception as exc:
                messagebox.showerror("錯誤", f"無法儲存日誌：{exc}")

    # -----------------------------------------------------------------------
    # Close
    # -----------------------------------------------------------------------
    def _on_close(self):
        if self._running:
            if not messagebox.askyesno(
                "確認關閉",
                "批次作業正在執行中，確定要關閉嗎？\n（OCR 工作將被取消）"
            ):
                return
            self._cancel_event.set()
            self._pause_event.clear()
            time.sleep(0.5)

        # Save settings
        try:
            settings = self._collect_settings()
            self._sm.save(settings)
        except Exception:
            pass

        # Close log file
        if self._log_file_handle:
            try:
                self._log_file_handle.close()
            except Exception:
                pass

        self.root.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    root = tk.Tk()
    app = OCRGuiApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
