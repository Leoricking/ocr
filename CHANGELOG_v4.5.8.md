# OCR Engine v4.5.8

## Offline finance-aware correction

- No Claude API or API key is required.
- Expanded finance phrase, context and document-scoped corrections.
- Added safe correction for daily/weekly/K-line terminology and crossover/trend phrases.
- Added conservative punctuation-led line continuation merging while preserving OCR line count.
- Preserved `_OCR_raw.txt`; corrected output is written to `_OCR_corrected.txt` and `_OCR.txt`.
- Added regression coverage for the page 66 sample from 《135均線技術分析》.
- Updated GUI and CLI version to v4.5.8.
