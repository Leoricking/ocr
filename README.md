# OCR Engine v4.2.0 老舊掃描增強版

- 原圖、CLAHE、自適應二值化三路辨識，自動挑選最佳結果。
- 保持原尺寸，透明文字層座標不偏移。
- 強化泛黃紙張、淡字、印章與輕微模糊。
- 擴充台大物理／電磁學常見錯字。
- 每份 PDF 同步輸出 `_OCR.txt`。
- 保留 GPU DLL 自動載入、GPU 自我測試與 CPU 回退。

啟動：

```powershell
python .\ocr_engine_v4.2.0.py
```

第一次先選 GPU `1`、Claude 校對 `n`，用 3～10 頁 PDF 驗證。
