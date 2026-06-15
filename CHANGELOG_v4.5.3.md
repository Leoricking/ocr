# OCR Engine v4.5.3

## 主要輸出模式

GUI 新增三個互斥選項：

- 全輸出（PDF + TXT）
- 只輸出 PDF
- 只輸出 TXT

設定會自動記憶；舊版 `output_pdf` / `output_txt` 設定會自動遷移。品質分析與校正日誌仍可獨立開關。

CLI 新增：

```powershell
python .\ocr_engine.py --input "C:\PDFs" --output "C:\Output" --output-mode all
python .\ocr_engine.py --input "C:\PDFs" --output "C:\Output" --output-mode pdf
python .\ocr_engine.py --input "C:\PDFs" --output "C:\Output" --output-mode txt
```

舊的 `--no-pdf` 與 `--no-txt` 仍保留相容性，但不可同時停用 PDF 與 TXT。
