from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

BBox = Tuple[float, float, float, float]


@dataclass
class LayoutRegion:
    bbox: BBox
    region_type: str
    lines: list = field(default_factory=list)
    column_index: int = 0
    reading_order: int = 0
    score: float = 0.0

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])


class PageLayoutAnalyzer:
    """Lightweight offline layout analysis based on OCR geometry.

    This avoids adding another heavy model.  It first obtains a conservative
    full-page OCR seed, groups nearby lines into regions, detects columns, and
    classifies chart/noise-heavy blocks before region-level OCR reruns.
    """

    CHART_KEYS = ("MA", "MACD", "VOL", "KDJ", "RSI", "BOLL", "DIFF", "DEA")

    @staticmethod
    def line_box(line) -> BBox:
        box = line[0]
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        return min(xs), min(ys), max(xs), max(ys)

    @staticmethod
    def line_text(line) -> str:
        return str(line[1][0]).strip()

    @staticmethod
    def _union(boxes: Sequence[BBox]) -> BBox:
        return (
            min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes),
        )

    @staticmethod
    def _vertical_gap(a: BBox, b: BBox) -> float:
        return b[1] - a[3]

    @staticmethod
    def _horizontal_overlap(a: BBox, b: BBox) -> float:
        overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        denom = max(1.0, min(a[2] - a[0], b[2] - b[0]))
        return overlap / denom

    def _classify(self, lines: Sequence, page_width: float, page_height: float) -> str:
        texts = [self.line_text(line) for line in lines if self.line_text(line)]
        if not texts:
            return "noise"
        joined = " ".join(texts)
        total = max(len(joined), 1)
        cjk = len(re.findall(r"[\u3400-\u9fff]", joined))
        digits = len(re.findall(r"\d", joined))
        ascii_chars = len(re.findall(r"[A-Za-z]", joined))
        short_ratio = sum(1 for text in texts if len(text) <= 5) / max(len(texts), 1)
        chart_hits = sum(joined.upper().count(key) for key in self.CHART_KEYS)
        mixed_ratio = (digits + ascii_chars) / total
        boxes = [self.line_box(line) for line in lines]
        bbox = self._union(boxes)
        top_ratio = bbox[1] / max(page_height, 1.0)
        avg_height = sum(box[3] - box[1] for box in boxes) / len(boxes)

        if mixed_ratio > 0.48 and short_ratio > 0.45:
            return "chart"
        if chart_hits >= 2 and (mixed_ratio > 0.25 or short_ratio > 0.55):
            return "chart"
        if cjk == 0 and mixed_ratio > 0.65:
            return "noise"
        if top_ratio < 0.18 and len(texts) <= 4 and avg_height > page_height * 0.015:
            return "title"
        if len(texts) <= 3 and any("圖" in text for text in texts):
            return "caption"
        return "text"

    def _detect_columns(self, lines: Sequence, page_width: float) -> dict[int, int]:
        centers = []
        for idx, line in enumerate(lines):
            box = self.line_box(line)
            centers.append(((box[0] + box[2]) / 2.0, idx))
        centers.sort()
        if len(centers) < 8:
            return {idx: 0 for _, idx in centers}
        best_gap = 0.0
        split = None
        for pos in range(3, len(centers) - 3):
            gap = centers[pos][0] - centers[pos - 1][0]
            if gap > best_gap:
                best_gap = gap
                split = pos
        if split is None or best_gap < page_width * 0.11:
            return {idx: 0 for _, idx in centers}
        left = {idx for _, idx in centers[:split]}
        right = {idx for _, idx in centers[split:]}
        return {idx: (0 if idx in left else 1) for _, idx in centers}

    def analyze(self, lines: Sequence, page_width: float, page_height: float) -> List[LayoutRegion]:
        if not lines:
            return []
        indexed = list(enumerate(lines))
        columns = self._detect_columns(lines, page_width)
        regions: list[LayoutRegion] = []

        for column in sorted(set(columns.values())):
            column_items = [(idx, line) for idx, line in indexed if columns[idx] == column]
            column_items.sort(key=lambda item: (self.line_box(item[1])[1], self.line_box(item[1])[0]))
            current: list = []
            current_box: BBox | None = None
            median_height = 0.0
            heights = [self.line_box(line)[3] - self.line_box(line)[1] for _, line in column_items]
            if heights:
                median_height = sorted(heights)[len(heights) // 2]
            gap_limit = max(12.0, median_height * 1.8)

            for _, line in column_items:
                box = self.line_box(line)
                if not current:
                    current = [line]
                    current_box = box
                    continue
                assert current_box is not None
                gap = self._vertical_gap(current_box, box)
                overlap = self._horizontal_overlap(current_box, box)
                if gap <= gap_limit and (overlap >= 0.08 or abs(box[0] - current_box[0]) < page_width * 0.08):
                    current.append(line)
                    current_box = self._union([current_box, box])
                else:
                    region_type = self._classify(current, page_width, page_height)
                    regions.append(LayoutRegion(current_box, region_type, list(current), column))
                    current = [line]
                    current_box = box
            if current and current_box is not None:
                region_type = self._classify(current, page_width, page_height)
                regions.append(LayoutRegion(current_box, region_type, list(current), column))

        spanning = [r for r in regions if r.width >= page_width * 0.62]
        normal = [r for r in regions if r not in spanning]
        spanning.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        normal.sort(key=lambda r: (r.column_index, r.bbox[1], r.bbox[0]))
        ordered = spanning + normal
        for idx, region in enumerate(ordered):
            region.reading_order = idx
        return ordered

    @staticmethod
    def padded_bbox(region: LayoutRegion, image_width: int, image_height: int, pad: int = 8) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = region.bbox
        return (
            max(0, int(x0) - pad), max(0, int(y0) - pad),
            min(image_width, int(x1) + pad), min(image_height, int(y1) + pad),
        )
