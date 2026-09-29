"""(TOOL-033) The digit-template reader, on synthetic digits (numpy only).

A 5x7 bitmap font stands in for ClubGG's fixed amount font: templates are
harvested from labelled renders, then other amounts — at other scales — must
read back exactly, and anything the reader is unsure of must come back None
(Tesseract reads those), never as a wrong number.
"""

from __future__ import annotations

import random
import time

import numpy as np
import pytest

from plo5bp.ocr import digits as dg
from plo5bp.ocr import text as text_mod

FONT = {
    "0": [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."],
    "1": ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
    "2": [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
    "3": ["#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."],
    "4": ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
    "5": ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
    "6": ["..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."],
    "7": ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
    "8": [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
    "9": [".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."],
    ".": ["..", "..", "..", "..", "..", "##", "##"],
    ",": ["..", "..", "..", "..", "..", "##", "##", ".#", "#."],
    "$": ["..#..", ".####", "#.#..", "#.#..", ".###.", "..#.#", "..#.#", "####.", "..#.."],
}
TOP = {"$": -1}  # rows above the digit top


def render(text: str, scale: int = 3, gap: int = 1, touching: frozenset[int] = frozenset()) -> np.ndarray:
    """Ink mask of ``text`` in the bitmap font; glyph ``i`` in ``touching``
    has no gap after it (it touches the next one)."""
    cols, h = [], 11
    for i, ch in enumerate(text):
        rows = FONT[ch]
        g = np.zeros((h, len(rows[0])), dtype=bool)
        top = 1 + TOP.get(ch, 0)
        for r, line in enumerate(rows):
            g[top + r] = [c == "#" for c in line]
        cols += [g, np.zeros((h, 0 if i in touching else gap), dtype=bool)]
    img = np.concatenate([np.zeros((h, 2), dtype=bool)] + cols + [np.zeros((h, 2), dtype=bool)], axis=1)
    return np.kron(img, np.ones((scale, scale), dtype=bool))


TRAIN = ["0.12", "3,456.78", "9,012.34", "$5.67", "89", "1,070.00", "$2,468.13"]


@pytest.fixture(scope="module")
def reader() -> dg.DigitReader:
    res = dg.harvest(dg.Sample(render(t), text=t, source=t) for t in TRAIN)
    assert res.templates.complete and not res.problems, res.report()
    assert res.self_check == {"right": len(TRAIN), "unsure": 0, "wrong": 0, "total": len(TRAIN)}
    return dg.DigitReader(res.templates)


def _amount(rng: random.Random) -> str:
    d, c = rng.randrange(0, 250_000), rng.randrange(100)
    return rng.choice([f"{d:,}.{c:02d}", f"{d}.{c:02d}", f"${d:,}.{c:02d}", f"{d:,}"])


def test_segmentation_finds_glyphs_and_punctuation():
    glyphs, line = dg.segment(render("1,234.56"))
    assert len(glyphs) == 8 and line is not None
    punct = [i for i, g in enumerate(glyphs) if dg.is_punct(g, line)]
    assert punct == [1, 5]
    assert [dg.punct_by_shape(glyphs[i], line) for i in punct] == [",", "."]


@pytest.mark.parametrize("scale", [3, 4, 5])
def test_reads_unseen_amounts_exactly_at_any_scale(reader, scale):
    rng = random.Random(scale)
    for _ in range(40):
        text = _amount(rng)
        assert reader.read_text(render(text, scale=scale)) == text
        assert text_mod._parse_chip_text(text) is not None


def test_touching_digits_are_split_or_left_to_tesseract(reader):
    rng = random.Random(7)
    read = tried = 0
    for _ in range(60):
        text = _amount(rng)
        pairs = [i for i in range(len(text) - 1) if text[i].isdigit() and text[i + 1].isdigit()]
        if not pairs:
            continue
        tried += 1
        got = reader.read_text(render(text, touching=frozenset({rng.choice(pairs)})))
        assert got in (text, None), (text, got)  # never a wrong number
        read += got == text
    assert read > tried // 2, (read, tried)  # most touching pairs are cut apart
    # a whole amount run together is refused, not guessed
    assert reader.read_text(render("12,345.67", gap=0)) in ("12,345.67", None)


def test_noise_and_foreign_shapes_are_never_misread(reader):
    rng = np.random.default_rng(3)
    blob = rng.random((33, 90)) < 0.5
    assert reader.read_text(blob) is None
    letters = np.kron(np.array([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1]] * 3, dtype=bool), np.ones((3, 3), bool))
    assert reader.read_text(letters) is None
    assert reader.read_text(np.zeros((30, 60), dtype=bool)) is None
    # speckle on real digits: exact or refused, never a different number
    for seed in range(10):
        m = render("4,321.09").copy()
        noise = np.random.default_rng(seed).random(m.shape) < 0.01
        m ^= noise
        assert reader.read_text(m) in ("4,321.09", None)


def test_harvest_aligns_labels_given_as_cents():
    samples = [dg.Sample(render(t), cents=text_mod._parse_chip_text(t), source=t) for t in TRAIN]
    res = dg.harvest(samples)
    assert res.used == len(TRAIN) and res.templates.complete, res.report()
    # a label that cannot line up is reported, not guessed
    bad = dg.harvest([dg.Sample(render("123.45"), text="12.345", source="mislabelled")])
    assert bad.used == 0 and "do not line up" in bad.problems[0]
    assert "$1,755.59" in dg.label_variants(175559) and "1755.59" in dg.label_variants(175559)
    assert "1,070" in dg.label_variants(107000)


def test_incomplete_templates_are_not_used(tmp_path):
    res = dg.harvest(dg.Sample(render(t), text=t) for t in ["0.12", "3,456.8", "$5.6"])  # no 7 or 9
    assert not res.templates.complete and "MISSING digits: 79" in res.report()
    path = res.templates.save(tmp_path / "digits.npz")
    assert dg.DigitReader.load(path) is None  # Tesseract keeps reading


def test_template_file_round_trip(tmp_path, reader):
    path = reader.templates.save(tmp_path / "t" / "digits.npz")
    back = dg.Templates.load(path)
    assert back.chars == reader.templates.chars
    assert np.array_equal(back.glyphs, reader.templates.glyphs.astype(np.float32))
    loaded = dg.DigitReader.load(path)
    assert loaded is not None and loaded.read_text(render("2,580.47")) == "2,580.47"
    assert not list((tmp_path / "t").glob("*.tmp*"))  # atomic write


def test_text_reads_with_templates_first_and_tesseract_second(monkeypatch, reader):
    monkeypatch.setattr(text_mod, "_DIGIT_READER", reader)
    tess_calls: list[int] = []
    monkeypatch.setattr(text_mod, "_tesseract_chip_token", lambda prep: tess_calls.append(1) or 777)
    before = text_mod.reader_stats()
    prep = np.where(render("1,755.59"), 0, 255).astype(np.uint8)  # black text on white
    assert text_mod._ocr_chip_token(prep) == 175559 and not tess_calls
    garbage = np.where(np.random.default_rng(0).random((30, 80)) < 0.5, 0, 255).astype(np.uint8)
    assert text_mod._ocr_chip_token(garbage) == 777 and tess_calls == [1]
    after = text_mod.reader_stats()
    assert after["template"] == before["template"] + 1 and after["tesseract"] == before["tesseract"] + 1


def test_without_templates_nothing_changes(monkeypatch, tmp_path):
    monkeypatch.setattr(text_mod, "_DIGIT_READER", text_mod._UNSET)
    monkeypatch.setattr(dg, "TEMPLATE_PATH", tmp_path / "none.npz")
    assert text_mod._digit_reader() is None
    monkeypatch.setattr(text_mod, "_DIGIT_READER", text_mod._UNSET)
    monkeypatch.setenv("PLO5BP_OCR_DIGITS", "0")
    assert text_mod._digit_reader() is None


def test_reading_is_far_cheaper_than_a_tesseract_call(reader):
    crops = [render(_amount(random.Random(i))) for i in range(60)]
    t0 = time.perf_counter()
    for c in crops:
        reader.read_text(c)
    per_read = (time.perf_counter() - t0) / len(crops)
    assert per_read < 0.02, f"{per_read * 1000:.1f} ms per read"  # Tesseract: ~30-50 ms


def test_harvest_tool_says_what_it_needs(tmp_path, capsys):
    from plo5bp.ocr.tools import harvest_digits

    assert harvest_digits.main(["--frames", str(tmp_path), "--dry-run"]) == 1
    assert "no labelled crops" in capsys.readouterr().err
    assert harvest_digits.main(["--frames", str(tmp_path), "--crops", str(tmp_path / "nope")]) == 2
