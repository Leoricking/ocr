from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

FINANCE_TERMS = {
    "K線", "K線圖", "均線", "移動平均線", "日線", "週線", "月線", "分時線",
    "成交量", "均量線", "黃金交叉", "死亡交叉", "MACD", "DIFF", "DEA",
    "KDJ", "RSI", "BOLL", "MA", "VOL", "多頭", "空頭", "主力", "籌碼",
    "洗盤", "拉升", "出貨", "支撐", "壓力", "趨勢", "震盪", "突破", "跌破",
    "停利", "停損", "買進", "賣出", "上漲", "下跌",
}

KNOWN_BAD_TERMS = {
    "均綜", "均鎳", "均貌", "均綠", "日綜", "涸綜", "迥綜", "過綜",
    "K綜", "K貌", "K鎳", "黃金交又", "死亡交又", "上張", "震邊",
    "震渥", "處放", "因塢", "尺要", "尺有", "取決放", "K生圈",
}

ALLOWED_ASCII_TERMS = {
    "BUY", "SELL", "MACD", "KDJ", "RSI", "BOLL", "MA", "EMA", "SMA",
    "VOL", "DIFF", "DEA", "STOCK", "ANALYSIS", "MOVING", "AVERAGE",
}

_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_ASCII_RE = re.compile(r"[A-Za-z]")
_DIGIT_RE = re.compile(r"\d")
_MIXED_GARBAGE_RE = re.compile(r"(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d:+\-_/]{7,}")


@dataclass(frozen=True)
class QualityBreakdown:
    total: float
    confidence: float
    readability: float
    dictionary: float
    domain: float
    consistency: float
    punctuation: float
    penalties: float


class BlockTextEvaluator:
    """Score OCR candidates using OCR confidence plus language quality.

    The evaluator intentionally does not trust Paddle confidence alone.  It is
    deterministic, offline, and cheap enough to run for every region.
    """

    def __init__(
        self,
        finance_lexicon: Iterable[str] | None = None,
        common_words: Iterable[str] | None = None,
    ) -> None:
        self.finance_lexicon = set(finance_lexicon or FINANCE_TERMS)
        self.common_words = set(common_words or {
            "投資", "股票", "股價", "交易", "市場", "技術", "分析", "系統", "參數",
            "圖表", "指標", "行情", "價格", "買點", "賣點", "形態", "觀察", "操作",
            "時間", "宇宙", "相對論", "量子力學", "科學", "物理", "電磁波",
        })

    @staticmethod
    def _line_penalties(lines: Sequence[str]) -> float:
        penalty = 0.0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            cjk = len(_CJK_RE.findall(stripped))
            ascii_count = len(_ASCII_RE.findall(stripped))
            digits = len(_DIGIT_RE.findall(stripped))
            if len(stripped) <= 2 and cjk:
                penalty += 5.0
            if _MIXED_GARBAGE_RE.fullmatch(stripped) and cjk == 0:
                penalty += 12.0
            if ascii_count and digits and cjk == 0 and len(stripped) >= 8:
                penalty += 8.0
            if sum(ch in "|_[]{}<>^~`" for ch in stripped) >= 2:
                penalty += 4.0
        return penalty

    def score(
        self,
        texts: Sequence[str] | str,
        confidences: Sequence[float] | float,
        document_vocabulary: Mapping[str, int] | None = None,
    ) -> QualityBreakdown:
        lines = [texts] if isinstance(texts, str) else list(texts)
        text = "\n".join(line.strip() for line in lines if line and line.strip())
        if isinstance(confidences, (int, float)):
            conf_values = [float(confidences)]
        else:
            conf_values = [float(value) for value in confidences]
        if not text:
            return QualityBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        avg_conf = sum(conf_values) / len(conf_values) if conf_values else 0.0
        total_chars = max(len(text), 1)
        cjk_chars = len(_CJK_RE.findall(text))
        ascii_chars = len(_ASCII_RE.findall(text))
        digits = len(_DIGIT_RE.findall(text))
        useful_chars = cjk_chars + ascii_chars + digits
        readability_ratio = useful_chars / total_chars

        common_hits = sum(1 for word in self.common_words if word in text)
        finance_hits = sum(1 for word in self.finance_lexicon if word in text)
        consistency_hits = 0
        if document_vocabulary:
            consistency_hits = sum(
                1 for word, count in document_vocabulary.items()
                if count >= 3 and word in text
            )

        confidence_score = avg_conf * 25.0
        readability_score = min(readability_ratio, 1.0) * 25.0
        dictionary_score = min(common_hits * 9.0, 15.0)
        domain_score = min(finance_hits * 6.0, 15.0)
        consistency_score = min(consistency_hits * 4.0, 10.0)

        punctuation_count = sum(text.count(ch) for ch in "，。！？；：、,.!?;:")
        punctuation_score = min(10.0, 4.0 + punctuation_count * 0.4)

        penalties = self._line_penalties(lines)
        penalties += sum(7.0 for bad in KNOWN_BAD_TERMS if bad in text)
        rare_ratio = sum(1 for ch in text if "\u3400" <= ch <= "\u4dbf") / total_chars
        if rare_ratio > 0.02:
            penalties += min(12.0, rare_ratio * 180.0)

        total = (
            confidence_score + readability_score + dictionary_score + domain_score
            + consistency_score + punctuation_score - penalties
        )
        total = max(0.0, min(100.0, total))
        return QualityBreakdown(
            total=total,
            confidence=confidence_score,
            readability=readability_score,
            dictionary=dictionary_score,
            domain=domain_score,
            consistency=consistency_score,
            punctuation=punctuation_score,
            penalties=penalties,
        )

    def calculate_score(
        self,
        text: str,
        ocr_conf: float,
        doc_vocabulary: Mapping[str, int] | None = None,
    ) -> float:
        return self.score(text, ocr_conf, doc_vocabulary).total
