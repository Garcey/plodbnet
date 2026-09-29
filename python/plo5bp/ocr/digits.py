"""Digit-template reader for ClubGG chip amounts (TOOL-033) — numpy only.

ClubGG draws every stack, bet and pot in one fixed font, so reading an amount
does not need a general OCR engine: split the preprocessed crop into glyphs
(connected ink components), scale each to a fixed box and compare it with
templates harvested from labelled crops of that same font. The Tesseract path
(`text.py`) stays as the fallback — this reader answers only when every glyph
matches confidently and returns None otherwise, never a guess.

Why: every changed crop used to start a Tesseract process (~30–50 ms on
Windows) and needed the external binary, and Tesseract's ``,``/``.`` confusion
needed heuristics. Template matching on a fixed font is exact and in-process.

- Templates live in ``templates/digits.npz`` (plain arrays, no pickle), built
  by ``python -m plo5bp.ocr.tools.harvest_digits`` from labelled frames or
  crops. Until a file covering all ten digits exists, `DigitReader.load`
  returns None and nothing changes.
- ``.`` and ``,`` are told apart by where they sit on the text line (a comma
  hangs below the baseline) unless harvested templates say otherwise.
- The input is an INK MASK (True = text pixel) of the same preprocessing the
  templates were harvested from (`text.prep_chip_crop` /
  `text.prep_commit_crop`).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

GLYPH_H = 24
GLYPH_W = 18
DIGITS = "0123456789"
CHARSET = DIGITS + ".,$"
TEMPLATE_PATH = Path(__file__).parent / "templates" / "digits.npz"
TEMPLATE_VERSION = 1
# A glyph is read only when its best template correlates at least this well…
MIN_SCORE = 0.80
# …and beats the best OTHER character by this margin.
MIN_MARGIN = 0.04
# Glyph aspect (w/h) may differ from its character's harvested aspect by at
# most this factor (a merged "10" is twice as wide as a "0").
MAX_ASPECT_RATIO = 1.45


# ---------------------------------------------------------------------------
# Pixels → glyphs
# ---------------------------------------------------------------------------


def ink_mask(img: np.ndarray, *, ink_dark: bool | None = None) -> np.ndarray:
    """Boolean ink mask of a (preprocessed) crop. ``ink_dark`` = the text is the
    dark colour (Tesseract-style black on white); None = the minority colour."""
    a = np.asarray(img)
    if a.ndim == 3:
        a = a[..., 0] * 0.114 + a[..., 1] * 0.587 + a[..., 2] * 0.299  # BGR → gray
    dark = a < 128
    if ink_dark is None:
        ink_dark = dark.mean() < 0.5
    return dark if ink_dark else ~dark


@dataclass
class Glyph:
    x0: int
    y0: int
    x1: int  # exclusive
    y1: int  # exclusive
    mask: np.ndarray  # bool, (y1 - y0, x1 - x0)

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0


def components(mask: np.ndarray) -> list[Glyph]:
    """8-connected ink components (run-length union-find: crops are small)."""
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    parent: list[int] = []

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    runs: list[tuple[int, int, int, int]] = []  # (row, start, end, label)
    prev: list[tuple[int, int, int]] = []
    for y in range(h):
        d = np.diff(np.concatenate(([0], m[y].view(np.int8), [0])))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        cur = []
        for s, e in zip(starts.tolist(), ends.tolist()):
            lab = len(parent)
            parent.append(lab)
            for ps, pe, pl in prev:  # 8-connected: touching diagonally counts
                if ps <= e and pe >= s:
                    ra, rb = find(lab), find(pl)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
            cur.append((s, e, lab))
            runs.append((y, s, e, lab))
        prev = cur
    boxes: dict[int, list[int]] = {}
    for y, s, e, lab in runs:
        r = find(lab)
        b = boxes.get(r)
        if b is None:
            boxes[r] = [s, y, e, y + 1]
        else:
            b[0], b[1], b[2], b[3] = min(b[0], s), min(b[1], y), max(b[2], e), max(b[3], y + 1)
    out = []
    for r, (x0, y0, x1, y1) in boxes.items():
        sub = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        out.append(Glyph(x0, y0, x1, y1, sub))
    # Paint each component's own runs (a box may overlap a neighbour's pixels).
    index = {r: g for r, g in zip(boxes, out)}
    for y, s, e, lab in runs:
        g = index[find(lab)]
        g.mask[y - g.y0, s - g.x0 : e - g.x0] = True
    return sorted(out, key=lambda g: (g.x0, g.y0))


def _merge(a: Glyph, b: Glyph) -> Glyph:
    x0, y0, x1, y1 = min(a.x0, b.x0), min(a.y0, b.y0), max(a.x1, b.x1), max(a.y1, b.y1)
    m = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    for g in (a, b):
        m[g.y0 - y0 : g.y1 - y0, g.x0 - x0 : g.x1 - x0] |= g.mask
    return Glyph(x0, y0, x1, y1, m)


@dataclass
class Line:
    """The text line's metrics, from its digit-sized glyphs."""

    top: float
    baseline: float

    @property
    def height(self) -> float:
        return max(1.0, self.baseline - self.top)


def segment(mask: np.ndarray) -> tuple[list[Glyph], Line | None]:
    """Glyphs left to right (noise dropped, pieces of one glyph merged) and the
    line metrics; ``([], None)`` when there is no text."""
    comps = components(mask)
    if not comps:
        return [], None
    tallest = max(c.h for c in comps)
    # Noise: specks far below a dot's size.
    comps = [c for c in comps if c.h >= 0.08 * tallest and c.mask.sum() >= max(2, 0.004 * tallest * tallest)]
    # Pieces of one glyph (a stroke broken by the threshold) overlap in x.
    merged: list[Glyph] = []
    for c in comps:
        if merged:
            p = merged[-1]
            overlap = min(p.x1, c.x1) - max(p.x0, c.x0)
            if overlap > 0.5 * min(p.w, c.w):
                merged[-1] = _merge(p, c)
                continue
        merged.append(c)
    big = [g for g in merged if g.h >= 0.6 * tallest]
    if not big:
        return merged, None
    line = Line(top=float(np.median([g.y0 for g in big])), baseline=float(np.median([g.y1 for g in big])))
    return merged, line


def is_punct(g: Glyph, line: Line) -> bool:
    """A glyph that starts in the lower half of the line: ``.`` or ``,``
    (digits start at the line's top, ``$`` above it)."""
    return g.y0 >= line.top + 0.45 * line.height and g.h < 0.75 * line.height


def punct_by_shape(g: Glyph, line: Line) -> str:
    """``,`` hangs below the baseline or is clearly taller than wide; ``.`` sits on it."""
    below = g.y1 - line.baseline
    if below > 0.08 * line.height or g.h > 1.35 * g.w:
        return ","
    return "."


def normalize(g: Glyph) -> np.ndarray:
    """The glyph area-averaged into a GLYPH_H × GLYPH_W box: height fills the
    box, the aspect ratio is kept, centred horizontally. Float in [0, 1]."""
    h, w = g.h, g.w
    scale = GLYPH_H / h
    if w * scale > GLYPH_W:
        scale = GLYPH_W / w
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    # Exact box averaging: the integral image of piecewise-constant pixels is
    # bilinear inside each pixel, so interpolating it integrates exactly.
    integ = np.zeros((h + 1, w + 1))
    integ[1:, 1:] = np.cumsum(np.cumsum(g.mask.astype(np.float64), axis=0), axis=1)

    def at(ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
        y0 = np.clip(np.floor(ys).astype(int), 0, h - 1)
        x0 = np.clip(np.floor(xs).astype(int), 0, w - 1)
        fy, fx = ys - y0, xs - x0
        y0g, x0g = y0[:, None], x0[None, :]
        fy, fx = fy[:, None], fx[None, :]
        return (
            integ[y0g, x0g] * (1 - fy) * (1 - fx)
            + integ[y0g + 1, x0g] * fy * (1 - fx)
            + integ[y0g, x0g + 1] * (1 - fy) * fx
            + integ[y0g + 1, x0g + 1] * fy * fx
        )

    ys = np.linspace(0, h, nh + 1)
    xs = np.linspace(0, w, nw + 1)
    grid = at(ys, xs)
    cell = (h / nh) * (w / nw)
    small = (grid[1:, 1:] - grid[:-1, 1:] - grid[1:, :-1] + grid[:-1, :-1]) / cell
    out = np.zeros((GLYPH_H, GLYPH_W))
    top = (GLYPH_H - nh) // 2
    left = (GLYPH_W - nw) // 2
    out[top : top + nh, left : left + nw] = np.clip(small, 0.0, 1.0)
    return out


def _zscore(a: np.ndarray) -> np.ndarray:
    v = a.reshape(-1).astype(np.float64)
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ---------------------------------------------------------------------------
# Templates + reader
# ---------------------------------------------------------------------------


@dataclass
class Templates:
    """Harvested exemplars: ``glyphs[i]`` (normalized) is character ``chars[i]``."""

    chars: list[str]
    glyphs: np.ndarray  # (n, GLYPH_H, GLYPH_W) float32
    aspects: np.ndarray  # (n,) w / h of the source glyph
    meta: dict = field(default_factory=dict)

    def covers(self) -> set[str]:
        return set(self.chars)

    @property
    def complete(self) -> bool:
        """All ten digits present — the reader's precondition."""
        return set(DIGITS) <= self.covers()

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(self.meta, version=TEMPLATE_VERSION, glyph_h=GLYPH_H, glyph_w=GLYPH_W)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(
            tmp,
            chars=np.array(self.chars, dtype="<U1"),
            glyphs=self.glyphs.astype(np.float32),
            aspects=self.aspects.astype(np.float32),
            meta=np.array(json.dumps(meta)),
        )
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "Templates":
        with np.load(Path(path), allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if meta.get("glyph_h") != GLYPH_H or meta.get("glyph_w") != GLYPH_W:
                raise ValueError(f"{path}: glyph size {meta.get('glyph_h')}x{meta.get('glyph_w')} is not {GLYPH_H}x{GLYPH_W}")
            return cls(
                chars=[str(c) for c in z["chars"]],
                glyphs=np.asarray(z["glyphs"], dtype=np.float32),
                aspects=np.asarray(z["aspects"], dtype=np.float32),
                meta=meta,
            )


class DigitReader:
    """Reads an ink mask to text (``"1,755.59"``) or None when unsure."""

    def __init__(self, templates: Templates):
        self.templates = templates
        self._z = np.stack([_zscore(g) for g in templates.glyphs]) if len(templates.chars) else np.zeros((0, GLYPH_H * GLYPH_W))
        self._chars = np.array(templates.chars)
        self._aspect = {c: float(np.median(templates.aspects[self._chars == c])) for c in set(templates.chars)}
        self._has_punct = {".", ","} <= templates.covers()

    @classmethod
    def load(cls, path: Path | str | None = None) -> "DigitReader | None":
        """The reader for the harvested templates, or None while there is no
        complete template file (then Tesseract does the reading, as before)."""
        p = Path(path) if path is not None else TEMPLATE_PATH
        if not p.is_file():
            return None
        t = Templates.load(p)
        return cls(t) if t.complete else None

    def classify(self, g: Glyph, *, allowed: str = CHARSET) -> tuple[str, float] | None:
        """(char, score) for a confidently matched glyph, else None."""
        if not len(self._chars):
            return None
        z = _zscore(normalize(g))
        scores = self._z @ z
        best: dict[str, float] = {}
        for c, s in zip(self._chars.tolist(), scores.tolist()):
            if c in allowed and s > best.get(c, -2.0):
                best[c] = s
        if not best:
            return None
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        char, score = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else -1.0
        if score < MIN_SCORE or score - second < MIN_MARGIN:
            return None
        aspect = g.w / max(1, g.h)
        want = self._aspect.get(char, aspect)
        if max(aspect, want) > MAX_ASPECT_RATIO * max(1e-6, min(aspect, want)):
            return None
        return char, score

    def _split(self, g: Glyph, line: Line) -> list[Glyph] | None:
        """Touching digits: cut a too-wide glyph at its thinnest columns."""
        digit_w = np.median([self._aspect[c] for c in DIGITS if c in self._aspect]) * line.height
        k = int(round(g.w / max(1.0, digit_w)))
        if k < 2 or k > 4:
            return None
        ink = g.mask.sum(axis=0).astype(float)
        cuts = []
        for i in range(1, k):
            centre = i * g.w / k
            lo, hi = int(max(1, centre - 0.25 * digit_w)), int(min(g.w - 1, centre + 0.25 * digit_w))
            if hi <= lo:
                return None
            cuts.append(lo + int(np.argmin(ink[lo:hi])))
        parts, start = [], 0
        for cut in cuts + [g.w]:
            sub = g.mask[:, start:cut]
            rows = np.flatnonzero(sub.any(axis=1))
            cols = np.flatnonzero(sub.any(axis=0))
            if rows.size == 0:
                return None
            sub = sub[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
            x0 = g.x0 + start + int(cols[0])
            parts.append(Glyph(x0, g.y0 + int(rows[0]), x0 + sub.shape[1], g.y0 + int(rows[-1]) + 1, sub))
            start = cut
        return parts

    def read_text(self, mask: np.ndarray) -> str | None:
        glyphs, line = segment(mask)
        if not glyphs or line is None:
            return None
        out: list[str] = []
        for g in glyphs:
            if is_punct(g, line):
                if g.w > 0.6 * line.height:
                    return None  # a flat streak (a pill edge), not a point
                if self._has_punct:
                    hit = self.classify(g, allowed=".,")
                    out.append(hit[0] if hit else punct_by_shape(g, line))
                else:
                    out.append(punct_by_shape(g, line))
                continue
            hit = self.classify(g, allowed=DIGITS + "$")
            if hit is None:
                parts = self._split(g, line)
                if parts is None:
                    return None
                hits = [self.classify(p, allowed=DIGITS) for p in parts]
                if any(h is None for h in hits):
                    return None
                out.extend(h[0] for h in hits)
                continue
            out.append(hit[0])
        text = "".join(out)
        return text if any(ch.isdigit() for ch in text) else None


# ---------------------------------------------------------------------------
# Harvesting (the tool: plo5bp.ocr.tools.harvest_digits)
# ---------------------------------------------------------------------------


def label_variants(cents: int) -> list[str]:
    """How ClubGG may print ``cents`` (with / without grouping, cents, ``$``)."""
    d, c = divmod(int(cents), 100)
    bodies = {f"{d:,}.{c:02d}", f"{d}.{c:02d}"}
    if c == 0:
        bodies |= {f"{d:,}", f"{d}"}
    return sorted(bodies | {"$" + b for b in bodies})


@dataclass
class Sample:
    """One labelled crop: its ink mask and the text (or cents) it shows."""

    mask: np.ndarray
    text: str | None = None
    cents: int | None = None
    kind: str = "any"
    source: str = ""


@dataclass
class HarvestResult:
    templates: Templates
    used: int
    problems: list[str]
    counts: dict[str, int]
    self_check: dict[str, int]

    def report(self) -> str:
        lines = [f"samples used: {self.used}; problems: {len(self.problems)}"]
        lines.append("glyphs per char: " + ", ".join(f"{c}={self.counts.get(c, 0)}" for c in CHARSET))
        missing = [c for c in DIGITS if not self.counts.get(c)]
        lines.append("complete (all ten digits)" if not missing else f"MISSING digits: {''.join(missing)}")
        sc = self.self_check
        if sc:
            lines.append(f"self-check: {sc['right']} read right, {sc['unsure']} left to Tesseract, {sc['wrong']} WRONG of {sc['total']}")
        lines += [f"  - {p}" for p in self.problems[:40]]
        if len(self.problems) > 40:
            lines.append(f"  … {len(self.problems) - 40} more")
        return "\n".join(lines)


def _align(glyphs: Sequence[Glyph], line: Line, candidates: Iterable[str]) -> str | None:
    """The one candidate text whose characters line up with the glyphs: same
    count, punctuation exactly where the small glyphs are."""
    punct = [is_punct(g, line) for g in glyphs]
    fits = [t for t in candidates if len(t) == len(glyphs) and all((ch in ".,") == p for ch, p in zip(t, punct))]
    return fits[0] if len(set(fits)) == 1 else None


def harvest(samples: Iterable[Sample], *, max_per_char: int = 12) -> HarvestResult:
    """Build templates from labelled crops: each crop's glyphs are paired with
    its text's characters; per character, the mean glyph plus up to
    ``max_per_char`` exemplars are kept. A self-check re-reads every sample."""
    per_char: dict[str, list[tuple[np.ndarray, float]]] = {}
    problems: list[str] = []
    kept: list[tuple[Sample, str]] = []
    for s in samples:
        glyphs, line = segment(s.mask)
        if not glyphs or line is None:
            problems.append(f"{s.source or '?'}: no text found")
            continue
        if s.text is not None:
            cands = [s.text.replace(" ", "")]
        elif s.cents is not None:
            cands = label_variants(s.cents)
        else:
            problems.append(f"{s.source or '?'}: no label")
            continue
        text = _align(glyphs, line, cands)
        if text is None:
            problems.append(f"{s.source or '?'}: {len(glyphs)} glyphs do not line up with {cands[0]!r}")
            continue
        for g, ch in zip(glyphs, text):
            per_char.setdefault(ch, []).append((normalize(g), g.w / max(1, g.h)))
        kept.append((s, text))
    chars, glyphs_out, aspects = [], [], []
    for ch in sorted(per_char):
        items = per_char[ch]
        stack = np.stack([a for a, _ in items])
        chars.append(ch)
        glyphs_out.append(stack.mean(axis=0))
        aspects.append(float(np.median([r for _, r in items])))
        # Exemplars nearest to the mean first: typical shapes, not outliers.
        mean_z = _zscore(stack.mean(axis=0))
        order = np.argsort([-float(_zscore(a) @ mean_z) for a, _ in items])
        for i in order[: max(0, max_per_char - 1)]:
            chars.append(ch)
            glyphs_out.append(items[i][0])
            aspects.append(items[i][1])
    counts = {ch: len(v) for ch, v in per_char.items()}
    t = Templates(
        chars=chars,
        glyphs=np.stack(glyphs_out).astype(np.float32) if glyphs_out else np.zeros((0, GLYPH_H, GLYPH_W), np.float32),
        aspects=np.array(aspects, dtype=np.float32),
        meta={"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "samples": len(kept), "counts": counts},
    )
    check = {"right": 0, "unsure": 0, "wrong": 0, "total": len(kept)}
    if t.complete:
        reader = DigitReader(t)
        for s, text in kept:
            got = reader.read_text(s.mask)
            if got is None:
                check["unsure"] += 1
            elif got == text:
                check["right"] += 1
            else:
                check["wrong"] += 1
                problems.append(f"{s.source or '?'}: self-check read {got!r}, label {text!r}")
    return HarvestResult(t, len(kept), problems, counts, check if t.complete else {})
