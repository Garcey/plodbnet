"""Home-games felt and chrome layout, 2026-09-28 second pass (the numbers were re-measured in
the preview at 360-430 px phones, 375x667, 812x375, 768x1024 and desktop):

- HGT-003: a phone's top bar keeps room for the table's name — the seat menu and the
  shuffle's shield move into ≡ (a FAILED check stays in the bar), start / pause is its icon;
  a phone on its side gets the phone's top bar too;
- MOB-003: on touch screens every small control has a 44 px target (an invisible margin, so
  nothing moves) and the pickers / pre-actions / sizing panel / menus grow where there is room;
- HGT-035: on a desktop the sizing panel is one row, so the dock keeps 114 px, not 184;
- MOB-004: a phone on its side shows what you hold over your cards (#hero-tag);
- FE-010: games.table.js is the one source of the felt's numbers (it publishes them as
  CSS variables); games.css no longer keeps copies.
- FE-009: games.css is laid out by component (an index, numbered sections, each with its own
  phone / landscape blocks) instead of dated blocks appended at the end; the table itself is
  games.felt.css (2026-10-01, shared with Study / Trainer), laid out the same way.
Node half skipped when Node is not installed."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hg_client_tools import STATIC, UI_BOOT, UI_FILES, computed, el, games_css, node_exe, parse_css, root, run_node  # noqa: E402

RULES = parse_css(games_css())
CSS = games_css()
PHONE = "(max-width: 560px)"
SMALL = "(max-width: 420px)"
SIDEWAYS = "(max-height: 480px) and (orientation: landscape)"
TOUCH = "(pointer: coarse)"
DESK = "(min-width: 761px) and (min-height: 701px)"


def _in_bar(spec):
    return [root(), el("body"), el("div#table-view"), el("header#tbar"), el(spec)]


# ------------------------------------------------------------------ HGT-003
def test_a_phone_top_bar_leaves_room_for_the_table_name():
    for spec in ("button#tb-seat", "button#tb-fair.pill.fair.off", "button#tb-fair.pill.fair.ok"):
        assert computed(RULES, _in_bar(spec), "display", (PHONE,)) == "none", spec
    # a failed shuffle check stays where everyone sees it
    assert computed(RULES, _in_bar("button#tb-fair.pill.fair.bad"), "display", (PHONE,)) != "none"
    # start / pause as its icon on a small phone (the accent ▶ is Start)
    run = [root(), el("body"), el("header#tbar"), el("button#tb-run.btn.sm.start"), el("span.lbl-s")]
    assert computed(RULES, run, "display", (SMALL,)) == "none"
    # a phone on its side: the phone's set, not the desktop's
    for spec in ("button#tb-invite", "button#tb-info", "button#tb-sound", "button#tb-prefs", "span#conn"):
        assert computed(RULES, _in_bar(spec), "display", (SIDEWAYS,)) == "none", spec
    assert computed(RULES, _in_bar("button#tb-more.icon-btn"), "display", (SIDEWAYS,)) == "inline-flex"
    # a long ≡ on a short phone scrolls
    menu = [root(), el("body"), el("div.menu")]
    assert computed(RULES, menu, "overflow-y") == "auto" and "100dvh" in computed(RULES, menu, "max-height")


def test_the_phone_menu_carries_the_seat_menu_and_the_shuffle(node, tmp_path):
    got = run_node(UI_BOOT + r"""
(async () => {
  const B = boot(() => ({}));
  B.ctx.matchMedia = (q) => ({ matches: q.includes("max-width: 560px") });
  const s = table();
  B.HG.core.G.state = s; B.HG.core.G.gameId = "T1";
  const fair = B.$("tb-fair"); fair.hidden = false; fair.lastElementChild.textContent = "Deck sealed";
  B.HG.fair = { openPanel() {} };
  B.HG.core.G.me = { name: "Host", email: "h@x" };
  B.HG.ui.init();  // (wires the top bar)
  let err = null;
  try { B.$("tb-more").click(); } catch (e) { err = String(e && e.stack || e); }
  const menu = B.W.doc.querySelector(".menu");
  const items = menu ? menu.children.map((x) => (x.tagName === "HR" ? "-" : x.textContent)) : [];
  console.log(JSON.stringify({ items, err }));
})();
""", STATIC, json.dumps(UI_FILES), tmp=tmp_path)
    items = got["items"]
    assert items[0].startswith("Host ·") and "Add chips" in items and "Leave seat" in items
    assert items.index("Leave seat") < items.index("Friday")  # (your seat first, then the table)
    assert "Shuffle: Deck sealed" in items


# ------------------------------------------------------------------ MOB-003
def test_touch_screens_get_fingertip_sized_targets_without_moving_the_table():
    rules_touch = [r for r in RULES if r.media == TOUCH]
    sel = {r.selector: r.decls for r in rules_touch}
    # an invisible 44 px target around the small ones (the visible control keeps its size)
    assert sel[".btn.sm::after"]["inset"][0] == sel[".chk::after"]["inset"][0] == "-7px -4px"  # 30 + 2 x 7 = 44
    assert sel[".seg button::after"]["inset"][0] == "-5px 0"  # 34 + 2 x 5 = 44
    assert sel[".icon-btn::after"]["inset"][0] == "-4px 0"  # 36 + 2 x 4 = 44
    assert sel["#tbar .btn.sm::after"]["inset"][0] == "-7px 0"  # (2 px apart in the top bar)
    # and the controls themselves grow where there is room
    seg = [root(), el("body"), el("div.seg"), el("button")]
    assert computed(RULES, seg, "height", (TOUCH,)) == "34px" and computed(RULES, seg, "height") == "30px"
    menu = [root(), el("body"), el("div.menu"), el("button")]
    assert computed(RULES, menu, "height", (TOUCH,)) == "44px"
    tag = [root(), el("body"), el("div.tagrow"), el("button")]
    assert computed(RULES, tag, "width", (TOUCH,)) == "34px"
    pre = [root(), el("body"), el("div#pre-row"), el("label.chk")]
    assert computed(RULES, pre, "height", (TOUCH, "(pointer: coarse) and (min-height: 481px)")) == "40px"
    assert computed(RULES, pre, "height", (TOUCH,)) == "34px"  # (sideways it floats over the felt: it keeps its size)
    step = [root(), el("body"), el("div#sizing"), el("button.sz-step")]
    assert computed(RULES, step, "height", (TOUCH,)) == "40px"


# ------------------------------------------------------------------ HGT-035
def test_the_desktop_dock_keeps_only_what_a_one_row_sizing_panel_needs():
    dock = [root(), el("body"), el("div#dock")]
    assert computed(RULES, dock, "--sz-h") == "50px"
    # 114 px: the one-row panel, a gap, the buttons (the action slot holds it — 2026-10-02)
    assert computed(RULES, dock, "--slot-h") == "calc(var(--sz-h) + 8px + 56px)"
    slot = [*dock, el("div#actbar"), el("div#act-slot")]
    assert computed(RULES, slot, "min-height") == "var(--slot-h)"
    sizing = [*slot, el("div#sizing")]
    assert computed(RULES, sizing, "height", (DESK,)) == "var(--sz-h)"  # (exactly: the slot is ONE height)
    assert computed(RULES, sizing, "flex-direction", (DESK,)) == "row"
    assert computed(RULES, sizing, "flex-direction") == "column"  # (the floating panel of phones / short windows)
    for spec in ("button.sz-step", "div.sz-row.sz-foot"):
        assert computed(RULES, [*sizing, el(spec)], "display", (DESK,)) == "none", spec
    play = (STATIC / "games.play.js").read_text(encoding="utf-8")
    assert '$("sz-slider").title =' in play  # (the range and the pot, where the foot line was)


# ------------------------------------------------------------------ 2026-10-02
COMPACT = "(max-width: 760px), (max-height: 700px)"
NARROW = "(max-width: 760px)"
SHORT_UPRIGHT = "(max-width: 760px) and (max-height: 700px) and (orientation: portrait)"
LOW_SIDEWAYS = "(max-height: 560px) and (orientation: landscape)"
#: Study / Trainer's page: the shared table's stylesheet, then its own (index.html's order)
STUDY_RULES = parse_css("\n".join((STATIC / n).read_text(encoding="utf-8") for n in ("games.felt.css", "style.css")))


def test_the_table_keeps_its_size_whatever_the_dock_shows():
    """(owner, 2026-10-02) "when it's your turn vs not your turn, the addition and removal
    of the betting options slightly resizes the whole table": the felt gets what the dock
    leaves, so every part of the dock is ONE height per layout whatever it shows — the
    action slot (sizing + buttons, the pre-actions or the status line) the buttons' height,
    the hand labels one line, and on Study / Trainer the actor line and the Trainer's "last
    move" line keep their place when empty. Measured on all three pages at desktop, short
    window, tablet and phone sizes with tools/games_preview/measure_dock.js."""
    dock = [root(), el("body"), el("div#dock")]
    act = [*dock, el("div#actbar"), el("div#act-slot"), el("div#act-btns"), el("button.act")]
    for media, h in (((COMPACT,), "56px"), ((COMPACT, NARROW), "52px"),
                     ((COMPACT, NARROW, SHORT_UPRIGHT), "46px"), ((COMPACT, LOW_SIDEWAYS), "42px")):
        assert computed(RULES, dock, "--slot-h", media) == h, media
        assert computed(RULES, act, "height", media) == h, media  # (the buttons fill it)
    # a phone on its side floats the dock over the felt: it moves nothing, so it holds nothing
    assert computed(RULES, dock, "--slot-h", (COMPACT, LOW_SIDEWAYS, SIDEWAYS)) == "0px"
    # the status line stands where the buttons stand: its words wrap BESIDE its buttons
    strip = [*dock, el("div#actbar"), el("div#act-slot"), el("div#status-strip")]
    assert computed(RULES, strip, "flex-wrap", (COMPACT,)) == "nowrap"
    assert computed(RULES, strip, "min-height", (COMPACT,)) == "var(--slot-h)"
    assert computed(RULES, [*strip, el("button.btn.sm")], "flex", (COMPACT,)) == "none"
    # one line of hand labels: a long one ends in "…"
    labels = [root(), el("body"), el("div#stage-wrap"), el("div#hero-hand-labels")]
    assert computed(RULES, labels, "flex-wrap") == "nowrap"
    assert computed(RULES, [*labels, el("span.hero-hand-label"), el("span.hhl-txt")], "text-overflow") == "ellipsis"
    # both pages build the slot the same way, and the labels' words in their own span
    play = (STATIC / "games.play.js").read_text(encoding="utf-8")
    assert '<div id="act-slot"><div id="sizing" hidden>' in play
    assert '<div id="status-strip" hidden></div></div>`' in play and 'class="hhl-txt"' in play
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    slot = page.split('<div id="act-slot">', 1)[1].split('<div class="dock-side right"', 1)[0]
    assert all(f'id="{x}"' in slot for x in ("sizing", "act-btns", "status-strip"))
    assert 'class="hhl-txt"' in (STATIC / "app.table.js").read_text(encoding="utf-8")

    # Study / Trainer: the actor line and the Trainer's "last move" line keep their place
    # (the page hides with `[hidden] { display: none !important }`: these win over it)
    bar = [root(), el("body.trainer-mode"), el("div#stage-wrap.hg-felt"), el("div#dock"), el("div#actbar")]
    gone = [*bar, el("div#actor-banner.actor-banner", hidden="")]
    assert computed(STUDY_RULES, gone, "display") == "block" and computed(STUDY_RULES, gone, "visibility") == "hidden"
    line = [*bar, el("div#actor-banner.actor-banner")]
    assert computed(STUDY_RULES, line, "white-space") == "nowrap" and computed(STUDY_RULES, line, "height") == "22px"
    verdict = [*bar, el("div#last-verdict.last-verdict", hidden="")]
    assert computed(STUDY_RULES, verdict, "display") == "flex" and computed(STUDY_RULES, verdict, "visibility") == "hidden"
    assert computed(STUDY_RULES, [*bar[:-1], el("div#actbar"), el("div#last-verdict.last-verdict")], "height") == "29px"
    study_verdict = [root(), el("body"), el("div#dock"), el("div#actbar"), el("div#last-verdict.last-verdict", hidden="")]
    assert computed(STUDY_RULES, study_verdict, "display") == "none"  # (Study grades nothing)
    # floating over a sideways phone's felt they move nothing: gone when empty
    assert computed(STUDY_RULES, gone, "display", (COMPACT, LOW_SIDEWAYS, SIDEWAYS, "(max-width: 860px) and (max-height: 480px) and (orientation: landscape)")) == "none"
    # no-limit's sizing panel is ALWAYS two rows on a desktop, pot-limit's one
    assert computed(STUDY_RULES, [root(), el("body.fmt-nl"), el("div#dock")], "--sz-h", (DESK,)) == "92px"
    assert computed(STUDY_RULES, [root(), el("body"), el("div#dock")], "--sz-h", (DESK,)) == "50px"
    assert 'document.body.classList.toggle("fmt-nl", !isPotLimit(s));' in (STATIC / "app.play.js").read_text(encoding="utf-8")
    # "Next hand" is a small button like the others (the old .primary min-height made it 40 px)
    nxt = [*bar[:-1], el("div#actbar"), el("div#act-slot"), el("div#status-strip"), el("button.btn.sm.primary")]
    assert computed(STUDY_RULES, nxt, "min-height") == "0" and computed(STUDY_RULES, nxt, "height") == "30px"


# ------------------------------------------------------------------ MOB-004
def test_a_phone_on_its_side_shows_what_you_hold_over_your_cards():
    wide = [root(), el("body"), el("div#stage.wide"), el("div#hero-zone"), el("div#hero-tag")]
    upright = [root(), el("body"), el("div#stage.portrait"), el("div#hero-zone"), el("div#hero-tag")]
    assert computed(RULES, wide, "display") == "flex"
    assert computed(RULES, upright, "display") == "none"  # (the row under the felt says it there)
    table = (STATIC / "games.table.js").read_text(encoding="utf-8")
    assert '$("hero-tag")' in table and "shortHand(d)" in table.split("function updateHero", 1)[1][:1600]
    # the hero's bet (up from the cards) and the award caption land in its band: it steps aside
    # (kept free of the bet instead, the bet went beside the cards — under the action bar)
    assert 'classList.toggle("hero-bet",' in table and 'classList.toggle("has-cap", !!line)' in table
    assert "fixed.push([hx - half - u * 2" not in table
    for cls in ("has-cap", "hero-bet"):
        tag = [root(), el("body"), el(f"div#stage.wide.{cls}"), el("div#hero-zone"), el("div#hero-tag")]
        assert computed(RULES, tag, "visibility") == "hidden", cls
    assert '<div id="hero-tag" hidden></div>' in (STATIC / "games.html").read_text(encoding="utf-8")


# ------------------------------------------------------------------ FE-010
def test_the_felt_numbers_have_one_source():
    table = (STATIC / "games.table.js").read_text(encoding="utf-8")
    for var in ("--felt-inset", "--hero-cw", "--hero-ov", "--open-cw"):
        assert f'stage.style.setProperty("{var}"' in table, var
        assert f"var({var}" in CSS, var
    assert "felt.style.inset" not in table
    # the copies are gone: no hand-kept hero width, overlap, felt insets or tabled width
    assert "--hk" not in CSS and "--hov" not in CSS
    assert not re.search(r"#hero-hole \{[^}]*--cw: calc\(var\(--u\) \* 6", CSS)
    assert ".portrait #felt" not in CSS
    assert "fitOpenRows" not in CSS and "keep in step" not in table.lower()


@pytest.fixture(scope="module")
def node():
    if node_exe() is None:
        pytest.skip("node is not installed")
    return node_exe()


# ------------------------------------------------------------------ HGT-005 / HGT-007
def test_the_dealer_disc_tucks_onto_its_own_plate_before_it_covers_the_total():
    """HGT-005: with no free spot beside a seat (7 seats on a small phone) the disc used to
    land on the pot row's "Total"; now a half-tucked place on its OWN plate costs little
    (it sits under the seats) and anything else costs full. Measured with measure_bets.js
    (it now checks the disc against "total" too) — clear at 360x740 / 375x667 / phones."""
    table = (STATIC / "games.table.js").read_text(encoding="utf-8")
    body = table.split("// dealer button: beside the seat", 1)[1][:2600]
    assert "for (const tuck of [0, 1, 2, 3])" in body and "tuck * disc.hw * 1.1" in body
    assert "over += rOver(rc, own, 0) * 0.15;" in body
    bets = (Path(__file__).resolve().parents[3] / "tools" / "games_preview" / "measure_bets.js").read_text(encoding="utf-8")
    assert '["pot", "total", "boards", "heroCards", "dockL", "dockR"]' in bets
    # a phone on its side: the dock's floating corners are obstacles for the disc and the bets
    assert 'if (g.wide) for (const id of ["dock-left", "actbar"])' in table


def test_a_runouts_pots_slide_below_the_top_seats_badges():
    """HGT-007: a 6-pot runout at 375x667 (7 seats) had the top seats' equity badges 2 px
    into the row of pots; the row now slides down by that much (never onto the boards)
    before it is fitted between the seats beside it. Measured (tools/games_preview, the
    felt's transitions finished first) on PLO6 x 7 seats, PLO5 x 6 and PLO67 x 5, 1-6 pots,
    every award step, long captions: 360-430 px phones, 375x667, 667/812x375, 768x1024,
    1366x768, 1920x1080."""
    table = (STATIC / "games.table.js").read_text(encoding="utf-8")
    body = table.split("function fitPots(host) {", 1)[1].split("function placeRabbit", 1)[0]
    assert 'if (host.id === "pots")' in body and "dy = Math.max(dy, r.bottom - pr.top + 2);" in body
    assert 'const free = $("boards").getBoundingClientRect().top - pr.bottom - 1' in body
    assert "const top = pr.top + dy, bot = pr.bottom + dy" in body and "if (r.bottom <= top || r.top >= bot) continue;" in body
    assert "`${shift} scale(" in body
    # a badge measured where it RESTS, not where its pop-in animation has it (~9 px higher)
    assert "function restBox(node, seatEl)" in table and body.count("restBox(node, sv.el)") == 2
    # refitted when what the seats beside it show changes (an award step's badge, a stack)
    assert "const beside = T.seats.map((sv) => `${sv.badge.dataset.k" in body
    # a top seat's showdown label counts when the row can clear it; the single pot pill too
    assert "if (lab - 2 < free) dy = Math.max(dy, lab);" in body
    nudge = table.split("function nudgePot(s) {", 1)[1][:1400]
    assert "pot.style.translate = `0 ${dy}px`" in nudge and 'nudgePot(s);' in table and "nudgePot(T.lastS);" in table


def test_six_pots_the_side_rows_and_the_caption_keep_clear():
    """HGT-007 (continued): six pots are wider than the gap between a 7-seat table's upper
    side seats' rows even at their smallest: those rows drop onto their own avatars (never
    their name and stack). The award caption stays clear of the seats beside it (a narrower
    wrap upright, a slide right on a phone on its side), rises over the street tag rather
    than onto the hero's cards, and the pot being paid grows inward at the row's ends."""
    table = (STATIC / "games.table.js").read_text(encoding="utf-8")
    fit = table.split("function fitSeats() {", 1)[1].split("function fitCaption", 1)[0]
    assert "sv.openLift = lift" in fit and "restBox(plate, sv.el).top - st.top - 2 - bot" in fit
    assert "calc(var(--u) * ${T.geom && T.geom.wide ? -0.4 : -5.5}${lift})" in table
    cap = table.split("function fitCaption(g, st) {", 1)[1].split("function captionLanding", 1)[0]
    for var in ("--cap-room", "--cap-dx", "--cap-up"):
        assert var in cap, var
    assert 'cap.style.setProperty("--cap-room", "100vw")' in table  # (captionHeight: the full 35 u)
    land = table.split("function captionLanding(g, st) {", 1)[1][:900]
    assert '"--cap-up"' in land and '"--cap-dx"' in land
    # a label sliding off a block may use the stage's own margin, 2 px from the screen edge
    assert "const mL = Math.min(4, 2 - Math.max(0, st.left - sb.left))" in fit
    base = [root(), el("body"), el("div#stage"), el("div#center"), el("div#award-caption")]
    assert "var(--cap-dx, 0px)" in computed(RULES, base, "transform") and "var(--cap-up, 0px)" in computed(RULES, base, "transform")
    up = [root(), el("body"), el("div#stage.portrait.cap-up"), el("div#center"), el("div#street-tag")]
    assert computed(RULES, up, "visibility") == "hidden"
    portrait = [root(), el("body"), el("div#stage.portrait"), el("div#center"), el("div#award-caption")]
    assert "var(--cap-room" in computed(RULES, portrait, "max-width")
    wide = [root(), el("body"), el("div#stage.wide"), el("div#center"), el("div#award-caption")]
    assert computed(RULES, wide, "--cap-dy") == "0.6" and "var(--cap-room" in computed(RULES, wide, "max-width")
    pots = [root(), el("body"), el("div#pots.tight"), el("div.potc.on")]
    assert computed(RULES, pots, "transform") == "none"
    assert ".potc.on:first-child { transform-origin: 0 50%; }" in CSS and ".potc.on:last-child { transform-origin: 100% 50%; }" in CSS
    # the measuring tools finish the felt's transitions first (a background tab's clock stands still)
    for tool in ("measure_showdown.js", "measure_bets.js"):
        src = (Path(__file__).resolve().parents[3] / "tools" / "games_preview" / tool).read_text(encoding="utf-8")
        assert "getAnimations()" in src and "a.finish()" in src, tool


# ------------------------------------------------------------------ FE-009
@pytest.mark.parametrize("name,sections", [("games.felt.css", 8), ("games.css", 13)])
def test_the_stylesheet_has_one_section_per_component(name, sections):
    """FE-009: games.css grew by appending dated blocks that overrode earlier rules, so how a
    seat or a button looked depended on the whole file. It is laid out by component now: a
    CONTENTS index, the numbered sections in that order, each component's phone / landscape
    rules in its own section, touch screens last (it wins over every size it enlarges); the
    rules the audit found set twice are one rule each. (The reorder was checked in the browser:
    every element's computed style is the same as before, at 12 window sizes.) Since
    2026-10-01 the TABLE is its own file, games.felt.css (Study / Trainer load it too), laid
    out the same way; the move was checked the same way (in-hand, your turn, showdown, desktop
    and both phone orientations: every computed style the same)."""
    css = (STATIC / name).read_text(encoding="utf-8")
    heads = re.findall(r"^/\* ========== (\d+)\. ([^=]+?) ========== \*/", css, re.M)
    assert [int(n) for n, _ in heads] == list(range(1, sections + 1))
    contents = css[css.index("CONTENTS."):css.index("/* ========== 1.")]
    for n in range(1, sections + 1):
        assert re.search(rf"^\s+{n}\. \S", contents, re.M), n
    # the touch section is the last one, and nothing but touch rules follow its header
    tail = parse_css(css[css.index(f"/* ========== {sections}."):])
    assert tail and all(r.media and "pointer: coarse" in r.media for r in tail)
    # every banner is a numbered section header: no "/* ====== Table UX round 2 (2026-09-22)"
    # block appended at the end any more
    banners = re.findall(r"^/\* ={6,}.*$", css, re.M)
    assert len(banners) == sections and all(re.match(r"/\* ========== \d+\. .+ ========== \*/$", b.rstrip()) for b in banners)


def test_the_rules_set_twice_are_one_rule_and_the_page_loads_the_table_first():
    for sel in ("#tbar .ttl", "#boards", ".av", "#stage.portrait #burns", "#stage.wide #burns"):
        assert sum(1 for r in RULES if r.media is None and r.selector == sel) == 1, sel
    assert computed(RULES, [root(), el("body"), el("div#boards")], "position") == "relative"  # (the rabbit is placed in it)
    page = (STATIC / "games.html").read_text(encoding="utf-8")
    assert page.index("/games/static/games.felt.css") < page.index("/games/static/games.css")
