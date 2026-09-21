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


# --- review 2026-09-20 I11: trailing separators / leading dot -----------

import pytest  # noqa: E402  (kept next to the table it serves)

# (raw tesseract text, expected cents). The "was" column in the comments is
# what the pre-fix parser returned.
_I11_CASES = [
    # A trailing "." / "," is punctuation noise, NOT a decimal point. The old
    # multi-dot collapse kept the LAST dot: "450.5." -> "4505." -> $4505.
    ("450.5.", 45_050),         # was 450_500  (the HANDOFF 450.5 -> 4505 drop)
    ("70.05.", 7_005),          # was 700_500
    ("1,755.59.", 175_559),     # was 17_555_900
    ("580.03,", 58_003),
    ("12.", 1_200),
    ("180 .", 18_000),
    # Last separator + 1-2 digits = decimal point.
    ("580.03", 58_003),
    ("1,755.59", 175_559),
    ("1.755.59", 175_559),
    ("1455.1", 145_510),
    ("0.50", 50),
    (".50", 50),                # was 5_000 (regex started at the "5")
    ("$.5", 50),
    # ... and the same for a comma misread of the decimal point.
    ("450,5", 45_050),          # was 450_500
    ("70,05", 7_005),           # was 700_500
    # 3+ digits after the last separator = a (mis)read thousands comma.
    ("1.090", 109_000),
    ("1,090", 109_000),
    ("12.345", 1_234_500),
    ("1.234.567", 123_456_700),
    # Plain integers.
    ("74", 7_400),
    ("1455", 145_500),
    # None-vs-0: no digit at all -> None; a real zero -> 0.
    ("", None),
    ("$", None),
    (",", None),
    (".", None),
    ("All In", None),
    ("0", 0),
    ("$0", 0),
    ("0.00", 0),
]


@pytest.mark.parametrize("raw, expected", _I11_CASES)
def test_parse_chip_text_table(raw, expected):
    got = _parse_chip_text(raw)
    assert got == expected
    # `0 == False`-style accidents: None and 0 must stay distinguishable.
    assert (got is None) == (expected is None)


def test_trailing_separator_never_inflates_the_amount():
    """Property form of I11: appending "."/"," to any clean amount must not
    change its value (it used to multiply it by 10-100)."""
    for clean in ("5", "74", "180", "450.5", "70.05", "580.03", "1,090",
                  "1,755.59", "12,345.6", "0", "0.5"):
        want = _parse_chip_text(clean)
        for junk in (".", ",", "..", ".,", " ."):
            assert _parse_chip_text(clean + junk) == want, (clean, junk)
