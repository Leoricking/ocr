from ocr_quality import BlockTextEvaluator


def test_finance_candidate_beats_high_confidence_bad_term():
    evaluator = BlockTextEvaluator()
    bad = evaluator.score(["135均綜交易法"], [0.98]).total
    good = evaluator.score(["135均線交易法"], [0.86]).total
    assert good > bad


def test_mixed_chart_noise_is_penalized():
    evaluator = BlockTextEvaluator()
    noise = evaluator.score(["5230371MAW0L10:697164"], [0.95]).total
    text = evaluator.score(["股價突破13MA並形成黃金交叉"], [0.85]).total
    assert text > noise
