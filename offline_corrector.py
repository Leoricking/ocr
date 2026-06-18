"""Offline, domain-aware OCR correction for Traditional Chinese documents.

The module is deliberately network-free.  It uses phrase, context and
file-scoped rules only; dangerous single-character global substitutions are
not allowed.  Line count is preserved so corrected text stays aligned with
PaddleOCR boxes used by the searchable-PDF overlay.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class CorrectionChange:
    source: str
    target: str
    rule_id: str
    rule_type: str
    line_index: int = -1


@dataclass
class CorrectionResult:
    corrected_texts: list[str]
    sources: list[str]
    changes: list[CorrectionChange]
    domain: str
    confidence: float = 1.0
    unresolved_tokens: list[str] | None = None


FINANCE_MARKERS = {
    "股市", "股票", "股價", "K線", "均線", "均綜", "MA", "MACD",
    "成交量", "主力", "籌碼", "買進", "賣出", "多頭", "空頭",
    "停損", "停利", "日綜", "涸綜", "黃金交又",
}
SCIENCE_MARKERS = {
    "牛頓", "物理", "宇宙", "相對論", "量子", "電磁", "光速", "粒子",
}

# Phrase-scoped rules.  Never add one-character rules such as 綜->線.
FINANCE_RULES: tuple[tuple[str, str, str], ...] = (
    # Moving-average family
    ("135均總", "135均線", "finance:135-line"),
    ("135均或", "135均線", "finance:135-line"),
    ("135均綜", "135均線", "finance:135-line"),
    ("135均鎳", "135均線", "finance:135-line"),
    ("135均貌", "135均線", "finance:135-line"),
    ("135均綠", "135均線", "finance:135-line"),
    ("均綜交易法", "均線交易法", "finance:ma-phrase"),
    ("均綜系統", "均線系統", "finance:ma-phrase"),
    ("均綜參數", "均線參數", "finance:ma-phrase"),
    ("均綜排列", "均線排列", "finance:ma-phrase"),
    ("均綜形態", "均線形態", "finance:ma-phrase"),
    ("均綜纏繞", "均線纏繞", "finance:ma-phrase"),
    ("均綜背離", "均線背離", "finance:ma-phrase"),
    ("均綜變化", "均線變化", "finance:ma-phrase"),
    ("均綜走勢", "均線走勢", "finance:ma-phrase"),
    ("均綜週期", "均線週期", "finance:ma-phrase"),
    ("均綜涸期", "均線週期", "finance:ma-period"),
    ("均綜調期", "均線週期", "finance:ma-period"),
    ("均綜安數", "均線參數", "finance:ma-parameter"),
    ("均量綜", "均量線", "finance:volume-line"),
    ("均量鎳", "均量線", "finance:volume-line"),
    ("DIFF綜", "DIFF線", "finance:indicator-line"),
    ("DEA綜", "DEA線", "finance:indicator-line"),
    ("雙綜", "雙線", "finance:indicator-line"),

    # Daily / weekly / monthly / intraday lines
    ("日綜圖", "日線圖", "finance:daily-line"),
    ("日綜初選", "日線初選", "finance:daily-line"),
    ("日綜顯示", "日線顯示", "finance:daily-line"),
    ("日綜反映", "日線反映", "finance:daily-line"),
    ("日綜觀察", "日線觀察", "finance:daily-line"),
    ("日綜複盤", "日線複盤", "finance:daily-line"),
    ("日鎳圖", "日線圖", "finance:daily-line"),
    ("日貌圖", "日線圖", "finance:daily-line"),
    ("涸綜圖", "週線圖", "finance:weekly-line"),
    ("迥綜圖", "週線圖", "finance:weekly-line"),
    ("過綜圖", "週線圖", "finance:weekly-line"),
    ("遞綜圖", "週線圖", "finance:weekly-line"),
    ("酒綜圖", "週線圖", "finance:weekly-line"),
    ("涸綜複查", "週線複查", "finance:weekly-line"),
    ("迥綜複查", "週線複查", "finance:weekly-line"),
    ("過綜複查", "週線複查", "finance:weekly-line"),
    ("涸綜顯示", "週線顯示", "finance:weekly-line"),
    ("月綜圖", "月線圖", "finance:monthly-line"),
    ("月綜", "月線", "finance:monthly-line"),
    ("分時綜圖", "分時線圖", "finance:intraday-line"),
    ("分時K綜圖", "分時K線圖", "finance:intraday-kline"),
    ("短涸期", "短週期", "finance:period"),
    ("長涸期", "長週期", "finance:period"),

    # K-line and indicators
    ("K綜形態", "K線形態", "finance:kline"),
    ("K鎳形態", "K線形態", "finance:kline"),
    ("K貌形態", "K線形態", "finance:kline"),
    ("K綜圖", "K線圖", "finance:kline"),
    ("K鎳圖", "K線圖", "finance:kline"),
    ("K貌圖", "K線圖", "finance:kline"),
    ("日K德", "日K線", "finance:kline"),
    ("周K德", "週K線", "finance:kline"),
    ("黃金交又", "黃金交叉", "finance:crossover"),
    ("金又十字", "黃金交叉", "finance:crossover"),
    ("金文十字", "黃金交叉", "finance:crossover"),
    ("死亡交又", "死亡交叉", "finance:crossover"),
    ("死心交叉", "死亡交叉", "finance:crossover"),
    ("死亡文叉", "死亡交叉", "finance:crossover"),
    ("MAGD", "MACD", "finance:macd"),
    ("mAGD", "MACD", "finance:macd"),

    # Trend / prose phrases that are safe as complete phrases
    ("上張趨勢", "上漲趨勢", "finance:trend"),
    ("上張行情", "上漲行情", "finance:trend"),
    ("上張過程", "上漲過程", "finance:trend"),
    ("上張形態", "上漲形態", "finance:trend"),
    ("上張訊號", "上漲訊號", "finance:trend"),
    ("震邊行情", "震盪行情", "finance:trend"),
    ("震渥行情", "震盪行情", "finance:trend"),
    ("震漫行情", "震盪行情", "finance:trend"),
    ("震燙行情", "震盪行情", "finance:trend"),
    ("震盜行情", "震盪行情", "finance:trend"),
    ("低位震邊", "低位震盪", "finance:trend"),
    ("低位震渥", "低位震盪", "finance:trend"),
    ("高位震邊", "高位震盪", "finance:trend"),
    ("高位震渥", "高位震盪", "finance:trend"),
    ("橫盤震邊", "橫盤震盪", "finance:trend"),
    ("橫盤震渥", "橫盤震盪", "finance:trend"),
    ("趨對", "趨勢", "finance:trend"),
    ("趨剪", "趨勢", "finance:trend"),
    ("趨藝", "趨勢", "finance:trend"),
    ("超勢", "趨勢", "finance:trend"),
    ("處放上漲", "處於上漲", "finance:position"),
    ("處放下跌", "處於下跌", "finance:position"),
    ("處放低位", "處於低位", "finance:position"),
    ("處放高位", "處於高位", "finance:position"),
    ("位放股價", "位於股價", "finance:position"),
    ("位放13MA", "位於13MA", "finance:position"),
    ("基放這", "基於這", "finance:position"),
    ("由放135", "由於135", "finance:position"),
    ("因塢", "因為", "finance:phrase"),
    ("作塢", "作為", "finance:phrase"),
    ("改塢", "改為", "finance:phrase"),
    ("塢準", "為準", "finance:phrase"),
    ("尺要", "只要", "finance:phrase"),
    ("尺有", "只有", "finance:phrase"),
    ("尺是", "只是", "finance:phrase"),
    ("貝是", "只是", "finance:phrase"),
    ("化繁塢簡", "化繁為簡", "finance:phrase"),
    ("圖文立茂", "圖文並茂", "finance:phrase"),
    ("功虧一簧", "功虧一簣", "finance:phrase"),
    ("精隨", "精髓", "finance:phrase"),
    ("與罘不同", "與眾不同", "finance:phrase"),
    ("優翼", "優異", "finance:phrase"),
    ("關註", "關注", "finance:phrase"),
    ("探取", "採取", "finance:phrase"),
    ("探用", "採用", "finance:phrase"),
    ("觀禁", "觀察", "finance:phrase"),
    ("謨導", "誤導", "finance:phrase"),
    ("放案進場", "放棄進場", "finance:phrase"),
    ("買得最刻算", "買得最划算", "finance:phrase"),
    ("獲利叮結", "獲利了結", "finance:phrase"),
    ("落袋塢安", "落袋為安", "finance:phrase"),
    ("懼高痘", "懼高症", "finance:businessweekly"),
    ("取決放", "取決於", "finance:businessweekly"),
    ("籌碼乾浮", "籌碼乾淨", "finance:businessweekly"),
    ("分辦是洗盤", "分辨是洗盤", "finance:businessweekly"),
    ("進出跳象", "進出跡象", "finance:businessweekly"),
    ("阪權所有", "版權所有", "finance:businessweekly"),
    ("下得轉載", "不得轉載", "finance:businessweekly"),
    ("成交最", "成交量", "finance:volume"),
    ("買進訊德", "買進訊號", "finance:signal"),
    ("賣出訊德", "賣出訊號", "finance:signal"),
)

SCIENCE_RULES: tuple[tuple[str, str, str], ...] = (
    ("日時寺間]是什麼?", "「時間」是什麼？", "science:time-title"),
    ("日時寺間", "時間", "science:time"),
    ("日時寸間", "時間", "science:time"),
    ("宇審", "宇宙", "science:universe"),
    ("字宙", "宇宙", "science:universe"),
    ("相斜對論", "相對論", "science:relativity"),
    ("相對制", "相對論", "science:relativity"),
    ("量力學", "量子力學", "science:quantum"),
    ("雜陡", "雜誌", "science:magazine"),
    ("東印唐公置", "東印度公司", "science:proper-noun"),
)

DOCUMENT_RULES: tuple[tuple[tuple[str, ...], tuple[tuple[str, str, str], ...]], ...] = (
    (("135均線技術分析", "135均綜技術分析"), (
        ("北汁星通", "北斗星通", "document:135-company"),
        ("北洋星通", "北斗星通", "document:135-company"),
        ("夕北才星通", "北斗星通", "document:135-company"),
        ("車映科技", "華映科技", "document:135-company"),
        ("貴州茅合", "貴州茅台", "document:135-company"),
        ("格基軟體", "榕基軟體", "document:135-company"),
        ("咸明星辰", "啟明星辰", "document:135-company"),
        ("均綜", "均線", "document:135-ma-term"),
        ("均鎳", "均線", "document:135-ma-term"),
        ("均貌", "均線", "document:135-ma-term"),
        ("均綠", "均線", "document:135-ma-term"),
        ("日綜", "日線", "document:135-daily"),
        ("日鎳", "日線", "document:135-daily"),
        ("日貌", "日線", "document:135-daily"),
        ("涸綜", "週線", "document:135-weekly"),
        ("迥綜", "週線", "document:135-weekly"),
        ("過綜", "週線", "document:135-weekly"),
        ("遞綜", "週線", "document:135-weekly"),
        ("酒綜", "週線", "document:135-weekly"),
        ("月綜", "月線", "document:135-monthly"),
        ("K綜", "K線", "document:135-kline"),
        ("K鎳", "K線", "document:135-kline"),
        ("K貌", "K線", "document:135-kline"),
    )),
    (("K線圖一看就懂",), (
        ("K生圈", "K線圖", "document:kline-title"),
        ("K鞋圈", "K線圖", "document:kline-title"),
        ("K載圈", "K線圖", "document:kline-title"),
        ("政線因", "K線圖", "document:kline-title"),
        ("刻元吉", "劉元吉", "document:kline-author"),
        ("利元青", "劉元吉", "document:kline-author"),
        ("人民郎車出版社", "人民郵電出版社", "document:kline-publisher"),
        ("人民部車出版社", "人民郵電出版社", "document:kline-publisher"),
        ("人居郎車出版社", "人民郵電出版社", "document:kline-publisher"),
        ("正券交易", "證券交易", "document:kline-term"),
    )),
    (("今周刊", "摸透主力思維"), (
        ("股價漲多就有懼高痘?", "股價漲多就有懼高症？", "document:businessweekly-title"),
    )),
    (("賴樹聲", "電磁波"), (
        ("輻樹聲", "賴樹聲", "document:em-name"),
        ("白大物理", "台大物理", "document:em-school"),
        ("台台大", "台大", "document:em-school"),
        ("證老師", "賴老師", "document:em-name"),
    )),
)

EXACT_RULES = (
    ("http://ww.inewton.com.tw", "http://www.inewton.com.tw", "exact:url"),
    ("白大教授", "台大教授", "exact:school-title"),
    ("訊號土電路", "訊號±電路", "exact:signal-plus-minus"),
    ("F-ma 定律", "F=ma 定律", "exact:fma"),
)

_CONTEXT_PATTERNS: tuple[tuple[str, str, str], ...] = (
    (r"(?<!綜合)日(?:綜|鎳|貌)(?=(?:、|，|。|與|和|上|下|是|的|圖|初選|顯示|複盤|觀察|反映|\s|$))", "日線", "finance:daily-context"),
    (r"(?:涸|迥|過|遞|酒)(?:綜|鎳|貌)?(?=(?:、|，|。|與|和|上|下|是|的|圖|複查|顯示|觀察|\s|$))", "週線", "finance:weekly-context"),
    (r"K(?:綜|鎳|貌|總|繞)(?=(?:圖|形態|走勢|、|，|。|與|和|上|下|\s|$))", "K線", "finance:kline-context"),
    (r"(?<!一)上張(?=(?:趨勢|行情|形態|訊號|過程|、|，|。|\s|$))", "上漲", "finance:rise-context"),
    (r"震(?:邊|渥|漫|燙|盜)(?=(?:行情|走勢|盤整|、|，|。|中|後|\s|$))", "震盪", "finance:sideways-context"),
)

UNRESOLVED_PATTERNS = (
    "均綜", "日綜", "涸綜", "迥綜", "K綜", "交又", "上張", "震邊", "因塢", "尺要",
)


def _document_stem(file_name: str) -> str:
    stem = os.path.splitext(os.path.basename(file_name or ""))[0]
    while stem.lower().endswith("_ocr"):
        stem = stem[:-4]
    return stem


def detect_domain(texts: Iterable[str], file_name: str = "") -> str:
    blob = f"{file_name} {' '.join(map(str, texts))}"
    finance = sum(1 for marker in FINANCE_MARKERS if marker in blob)
    science = sum(1 for marker in SCIENCE_MARKERS if marker in blob)
    if finance >= 2 or any(k in file_name for k in ("均線", "均綜", "K線", "今周刊", "主力思維")):
        return "finance"
    if science >= 2 or any(k in file_name for k in ("牛頓", "電磁波", "時間")):
        return "science"
    return "general"


def _formula_dense(text: str) -> bool:
    symbols = sum(ch in "=+-*/^_()[]{}λβωηΓπ∞∇×" for ch in text)
    return symbols >= 4 or "\\" in text or "$" in text


def _apply_rules(text: str, rules: Sequence[tuple[str, str, str]], rule_type: str,
                 changes: list[CorrectionChange], line_index: int) -> tuple[str, str]:
    source_tag = ""
    for wrong, right, rule_id in rules:
        if wrong in text:
            before = text
            text = text.replace(wrong, right)
            if text != before:
                changes.append(CorrectionChange(wrong, right, rule_id, rule_type, line_index))
                source_tag = rule_id
    return text, source_tag


def _normalize_punctuation(text: str) -> str:
    # Preserve URLs and formulas; only normalize unambiguous punctuation.
    if re.search(r"https?://|www\.", text, re.IGNORECASE):
        protected_colons = True
    else:
        protected_colons = False
    text = text.replace("丶", "、")
    text = text.replace("，，", "，").replace("。。", "。").replace("……。。", "……")
    if not protected_colons:
        text = re.sub(r":(?=\D)", "：", text)
    text = re.sub(r"\?(?=\s|$)", "？", text)
    text = re.sub(r"!(?=\s|$)", "！", text)
    return text


def _merge_split_lines(lines: list[str], sources: list[str], changes: list[CorrectionChange]) -> None:
    """Merge only obvious punctuation-led continuations while preserving line count.

    The continuation line is blanked after its text is appended to the previous
    line.  This improves TXT readability without invalidating OCR box indexes.
    """
    for i in range(1, len(lines)):
        prev = lines[i - 1].rstrip()
        cur = lines[i].lstrip()
        if not prev or not cur:
            continue
        if cur[0] not in "、，；：！？）】」』":
            continue
        if re.search(r"[。！？；：]$", prev):
            continue
        # Avoid merging table/chart fragments and codes.
        if re.fullmatch(r"[\dA-Za-z.+%:/_-]+", prev) or re.fullmatch(r"[\dA-Za-z.+%:/_-]+", cur):
            continue
        merged = prev + cur
        if merged != lines[i - 1]:
            changes.append(CorrectionChange(lines[i - 1] + "\n" + lines[i], merged,
                                            "layout:punctuation-continuation", "layout", i - 1))
            lines[i - 1] = merged
            lines[i] = ""
            sources[i - 1] = sources[i - 1] or "layout:punctuation-continuation"
            sources[i] = "layout:merged"


class OfflineTextCorrector:
    """Reusable offline corrector with a stable API for GUI/CLI/tests."""

    def correct_page(self, texts: list[str], *, file_name: str = "",
                     page_num: int | None = None, domain: str = "auto") -> CorrectionResult:
        del page_num  # retained for API compatibility/audit callers
        return correct_lines(texts, file_name=file_name, domain=domain)


def correct_lines(texts: list[str], file_name: str = "", domain: str = "auto") -> CorrectionResult:
    selected_domain = detect_domain(texts, file_name) if domain == "auto" else domain
    doc = _document_stem(file_name)
    page_context = " ".join(str(x) for x in texts)
    corrected: list[str] = []
    sources: list[str] = []
    all_changes: list[CorrectionChange] = []

    doc_rules: list[tuple[str, str, str]] = []
    for markers, rules in DOCUMENT_RULES:
        if any(marker in doc for marker in markers):
            doc_rules.extend(rules)

    for line_index, original in enumerate(texts):
        text = str(original)
        line_changes: list[CorrectionChange] = []
        source = ""

        text, tag = _apply_rules(text, EXACT_RULES, "exact", line_changes, line_index)
        source = tag or source
        text, tag = _apply_rules(text, doc_rules, "document", line_changes, line_index)
        source = tag or source

        if not _formula_dense(text):
            if selected_domain == "finance":
                safe_rules = []
                for rule in FINANCE_RULES:
                    wrong = rule[0]
                    if wrong == "成交最" and not any(k in page_context for k in ("股價", "均線", "K線", "VOL", "放量", "縮量", "成交")):
                        continue
                    if wrong == "買進訊德" and not any(k in page_context for k in ("買進", "進場", "訊號", "指標", "K線")):
                        continue
                    if wrong == "賣出訊德" and not any(k in page_context for k in ("賣出", "出場", "停利", "停損", "訊號")):
                        continue
                    safe_rules.append(rule)
                text, tag = _apply_rules(text, safe_rules, "domain", line_changes, line_index)
                source = tag or source
                for pattern, repl, rule_id in _CONTEXT_PATTERNS:
                    before = text
                    text = re.sub(pattern, repl, text)
                    if text != before:
                        line_changes.append(CorrectionChange(before, text, rule_id, "context", line_index))
                        source = rule_id
            elif selected_domain == "science":
                safe_rules = []
                for rule in SCIENCE_RULES:
                    if rule[0] == "量力學" and not (any(k in page_context for k in ("物理", "相對論", "粒子", "原子", "宇宙", "科學")) or any(k in file_name for k in ("牛頓", "科學", "物理"))):
                        continue
                    safe_rules.append(rule)
                text, tag = _apply_rules(text, safe_rules, "domain", line_changes, line_index)
                source = tag or source

        normalized = _normalize_punctuation(text)
        if normalized != text:
            line_changes.append(CorrectionChange(text, normalized, "punctuation:normalize", "punctuation", line_index))
            source = source or "punctuation:normalize"
            text = normalized

        corrected.append(text)
        sources.append(source)
        all_changes.extend(line_changes)

    _merge_split_lines(corrected, sources, all_changes)
    unresolved = sorted({token for line in corrected for token in UNRESOLVED_PATTERNS if token in line})
    confidence = max(0.0, 1.0 - min(len(unresolved), 10) * 0.05)
    return CorrectionResult(corrected, sources, all_changes, selected_domain, confidence, unresolved)
