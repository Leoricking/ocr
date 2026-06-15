# Changelog — OCR Engine v4.5.5

## GUI Layout Polish

### Treeview Column Alignment
- Renamed all Treeview column keys from Chinese to English identifiers
  (`index`, `filename`, `pages`, `status`, `progress`, `avg_sec`,
  `elapsed_time`, `output_pdf`, `output_txt`, `analysis`)
- All heading anchors set to `center`
- Cell anchors: `center` for numeric/status columns; `w` (left-aligned) for
  `filename`, `output_pdf`, `output_txt`, `analysis`
- Explicit base widths (pre-DPI-scale): index=50, filename=270, pages=70,
  status=90, progress=90, avg_sec=100, elapsed_time=100, output_pdf/txt/analysis=180
- `stretch=True` on `filename`, `output_pdf`, `output_txt`, `analysis`;
  `stretch=False` on all numeric/status columns
- Horizontal scrollbar preserved
- `_resize_tree_columns` updated to use new English column keys

### Source Section Layout Rebuild
- Entries and button rows now stacked vertically inside the LabelFrame
  (row 0: label + source entry; row 1: 5-button toolbar; row 2: label +
  output entry; row 3: 2-button toolbar; row 4: output mode frame; row 5:
  recursive checkbutton)
- Source buttons in a single horizontal toolbar:
  "選擇單一 PDF", "選擇多個 PDF", "選擇資料夾", "清除來源", "重新掃描"
- Output buttons in a single horizontal toolbar:
  "選擇輸出資料夾", "開啟輸出資料夾"
- `padx=(0, 6)` between buttons except last in each row
- `columnconfigure(1, weight=1)` on the LabelFrame for responsive entry width
- All existing callbacks preserved; DPI scaling preserved

### Version
- Bumped from v4.5.4 to v4.5.5

### Tests Added
- `tests/test_gui_layout.py`: asserts source and output button callbacks exist
  on `OCRGuiApp`, and that `_var_output_mode` / `_var_recursive` are defined
- `tests/test_treeview_alignment.py`: asserts all 10 column keys are present,
  heading anchors are `center`, cell anchors match the spec, stretch flags
  are correct, and base widths match the spec
