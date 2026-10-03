"""The Trainer's side rail (2026-10-03; owner: "As the action list fills up, the stuff below
it moves down to make room. I don't want it to move around like that" — then: "pin the
session and lifetime stats to the bottom of the right panel. Put a little tab at the bottom
… 'Previous hands' … pressing that little tab or scrolling down pulls the hand history up
into the full right panel").

Measured in the preview (1500x920 and 2000x1028): the stats keep ONE position through whole
hands, the review included, and the sheet opens from the tab, from scrolling down over the
stats (never from the action list) and closes with Back, Esc or scrolling up at its top.
This pins the structure that does it."""

from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[3] / "python" / "plo5bp" / "ui" / "static"


def test_the_previous_hands_sheet_and_its_tab():
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    stats = page.split('<aside id="trainer-stats-panel"', 1)[1].split("</aside>", 1)[0]
    assert 'id="ph-tab"' in stats and 'id="recent-hands"' not in stats  # (the tab under the stats)
    sheet = page.split('<section id="prev-hands"', 1)[1].split("</section>", 1)[0]
    assert 'id="recent-hands"' in sheet and 'id="ph-back"' in sheet
    # a child of <main> after the rail: it lays itself over the rail's own grid cell
    assert page.index('<div id="history-panel"') < page.index('<section id="prev-hands"') < page.index("</main>")


def test_the_stats_are_pinned_and_the_actions_scroll_in_between():
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    assert "body.trainer-mode #side-rail > #history-panel { flex: 1 1 auto;" in css
    assert re.search(r"body\.trainer-mode #side-rail > #trainer-stats-panel \{\s*position: sticky; bottom: 0;", css)
    assert "body.trainer-mode #prev-hands {\n    display: block; grid-column: 2; grid-row: 1;" in css.replace("\r\n", "\n")
    assert "body.hands-open #prev-hands .ph-sheet { transform: none;" in css
    js = (STATIC / "app.trainer.js").read_text(encoding="utf-8")
    assert "function setupPreviousHands()" in js
    assert 'if (e.target.closest("#history")) return;' in js  # (reading the actions never flips the rail)
    assert "setupPreviousHands();" in (STATIC / "app.js").read_text(encoding="utf-8")
    assert 'document.body.classList.remove("hands-open")' in (STATIC / "app.topbar.js").read_text(encoding="utf-8")
