"""Unit tests for the chip-amount string parser.

The parser sees raw Tesseract output and converts it to integer cents.
Covers the comma-vs-period misread heuristic that resolves stacks like
``1,090`` (incorrectly emitted as ``1.090``) back to $1090 instead of
collapsing to $1.09.
"""

from __future__ import annotations

from plo5bp.ocr.text import _parse_chip_text


def test_plain_integer_dollars():
    assert _parse_chip_text("180") == 18000
    assert _parse_chip_text("1090") == 109000


def test_decimal_two_digit_cents():
    assert _parse_chip_text("580.03") == 58003
    assert _parse_chip_text("1.09") == 109
    assert _parse_chip_text("20.59") == 2059


def test_one_digit_cents_padded_to_two():
    # ``"5.5"`` is ambiguous Tesseract output but the parser pads
    # missing cents digits with trailing zero so $5.50 lands cleanly.
    assert _parse_chip_text("5.5") == 550


def test_comma_thousands_separator_stripped():
    assert _parse_chip_text("1,090") == 109000
    assert _parse_chip_text("1,755.59") == 175559
    assert _parse_chip_text("12,345") == 1234500


def test_comma_misread_as_decimal_three_digit_cents():
    # The bug case: Tesseract reads "1,090" as "1.090". Without the
    # heuristic we'd take cents="09" and return $1.09; with it, three-
    # digit cents flips us to comma interpretation → $1090.
    assert _parse_chip_text("1.090") == 109000
    assert _parse_chip_text("1.030") == 103000  # ShoveDeez post-ante
    assert _parse_chip_text("12.345") == 1234500


def test_two_dots_uses_last_as_decimal():
    # "1.234.56" — Tesseract emitted both a thousands separator and
    # a real decimal. Existing logic strips the leading dot.
    assert _parse_chip_text("1.234.56") == 123456


def test_two_dots_with_three_digit_cents():
    # "1.234.567" — last fragment is 3 digits, so even after the
    # multi-dot collapse the heuristic still flips to comma view.
    assert _parse_chip_text("1.234.567") == 123456700


def test_unparseable_returns_none():
    assert _parse_chip_text("") is None
    assert _parse_chip_text("abc") is None
    assert _parse_chip_text("$") is None


def test_strips_leading_currency_symbol():
    assert _parse_chip_text("$1,090") == 109000
    assert _parse_chip_text("$1.090") == 109000
    assert _parse_chip_text("$580.03") == 58003


def test_whitespace_tolerated():
    assert _parse_chip_text("  1,090  ") == 109000
    assert _parse_chip_text("1 090") == 109000
