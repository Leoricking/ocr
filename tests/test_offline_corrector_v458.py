from offline_corrector import OfflineTextCorrector, correct_lines


def test_page_66_finance_acceptance():
    lines = [
        "135均綜技術分析",
        "即使日綜",
        "丶涸綜顯示可以進場",
        "也要復盤觀察",
        "135均綜交易法有獨到的買進技巧",
        "投資者應嚴格按照日綜初選",
        "·涸綜複查的原則",
        "代表主力洗盤結束丶什麼形態暗示股價快速上張",
        "投資者才能真正掌握135均綜交易法的買股精隨",
        "因塢即使出現再好的形態",
        "MACD和均量綜都形成黃金交又",
        "北汁星通",
    ]
    out = correct_lines(lines, "135均線技術分析.pdf")
    assert out.corrected_texts[0] == "135均線技術分析"
    assert out.corrected_texts[1] == "即使日線、週線顯示可以進場"
    assert out.corrected_texts[2] == ""
    joined = "\n".join(out.corrected_texts)
    assert "135均線交易法" in joined
    assert "日線初選" in joined
    assert "週線複查" in joined
    assert "上漲" in joined
    assert "精髓" in joined
    assert "因為" in joined
    assert "均量線" in joined
    assert "黃金交叉" in joined
    assert "北斗星通" in joined


def test_no_dangerous_global_replacements():
    lines = ["綜合分析", "放量上漲", "一張圖片", "張先生", "網址 http://example.com/a:b"]
    assert correct_lines(lines, "一般文件.pdf").corrected_texts == lines


def test_network_free_corrector_api():
    result = OfflineTextCorrector().correct_page(["135均綜交易法"], file_name="135均線技術分析.pdf")
    assert result.corrected_texts == ["135均線交易法"]
    assert result.domain == "finance"
