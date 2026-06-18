from offline_corrector import correct_lines


def test_finance_document_rules():
    lines = ["135均綜交易法", "日綜圖與涸綜圖", "低位黃金交又", "上張趨勢"]
    out = correct_lines(lines, "135均線技術分析.pdf").corrected_texts
    assert out == ["135均線交易法", "日線圖與週線圖", "低位黃金交叉", "上漲趨勢"]


def test_kline_document_rules():
    lines = ["K生圈一看就懂", "刻元吉 編著", "人民郎車出版社"]
    out = correct_lines(lines, "K線圖一看就懂.pdf").corrected_texts
    assert out == ["K線圖一看就懂", "劉元吉 編著", "人民郵電出版社"]


def test_science_rules_and_scope():
    out = correct_lines(["日時寺間是什麼", "相斜對論", "量力學"], "Newton牛頓科學 時間.pdf").corrected_texts
    assert out == ["時間是什麼", "相對論", "量子力學"]
    assert correct_lines(["能量力學"], "一般文件.pdf").corrected_texts == ["能量力學"]


def test_no_overcorrection():
    lines = ["綜合分析", "放量上漲", "一張圖", "圓圈", "張先生"]
    assert correct_lines(lines, "一般文件.pdf").corrected_texts == lines
