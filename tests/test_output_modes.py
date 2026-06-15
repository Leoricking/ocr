"""Regression tests for v4.5.3 primary output selection."""
from settings_manager import DEFAULTS


def resolve_mode(mode):
    if mode not in {"all", "pdf", "txt"}:
        mode = "all"
    return mode in {"all", "pdf"}, mode in {"all", "txt"}


def test_default_mode_is_all():
    assert DEFAULTS["output_mode"] == "all"
    assert DEFAULTS["output_pdf"] is True
    assert DEFAULTS["output_txt"] is True


def test_all_outputs_pdf_and_txt():
    assert resolve_mode("all") == (True, True)


def test_pdf_only():
    assert resolve_mode("pdf") == (True, False)


def test_txt_only():
    assert resolve_mode("txt") == (False, True)


def test_invalid_mode_falls_back_to_all():
    assert resolve_mode("invalid") == (True, True)
