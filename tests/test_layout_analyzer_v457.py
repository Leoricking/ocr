from layout_analyzer import PageLayoutAnalyzer


def line(x0, y0, x1, y1, text, conf=0.9):
    return [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]], [text, conf]]


def test_detects_two_columns_and_keeps_left_before_right():
    lines = []
    for i in range(5):
        lines.append(line(50, 100 + i * 35, 420, 125 + i * 35, f"左欄正文{i}"))
        lines.append(line(600, 100 + i * 35, 970, 125 + i * 35, f"右欄正文{i}"))
    analyzer = PageLayoutAnalyzer()
    regions = analyzer.analyze(lines, 1024, 1400)
    text_regions = [r for r in regions if r.region_type == "text"]
    assert {r.column_index for r in text_regions} == {0, 1}
    left_orders = [r.reading_order for r in text_regions if r.column_index == 0]
    right_orders = [r.reading_order for r in text_regions if r.column_index == 1]
    assert max(left_orders) < min(right_orders)


def test_chart_region_classification():
    lines = [
        line(50, 100, 240, 120, "5230371MAW0L10:697164"),
        line(50, 130, 180, 150, "13MA"),
        line(50, 160, 180, 180, "MACD"),
        line(50, 190, 180, 210, "-17.94"),
    ]
    analyzer = PageLayoutAnalyzer()
    regions = analyzer.analyze(lines, 1000, 1400)
    assert any(r.region_type == "chart" for r in regions)
