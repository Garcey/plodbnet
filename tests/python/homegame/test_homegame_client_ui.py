"""Home-games web client (static/games.*): CSS and client-logic regressions from the
2026-09-28 improvements pass. CSS is checked with the small cascade in
hg_client_tools; client logic runs in Node (skipped when Node is not installed)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hg_client_tools import (  # noqa: E402
    STATIC, computed, el, games_css, keyframes, node_exe, parse_css, root, run_node,
)

RULES = parse_css(games_css())
REDUCE = "(prefers-reduced-motion: reduce)"


def _seat_button(*seat_classes, host=False):
    seat = el("div.seat" + "".join("." + c for c in seat_classes))
    return [root(), el("body"), el("div#table-view"), el("div#stage"), el("div#seats"), seat, el("button.seat-sit")]


# --------------------------------------------------------------- HGT-008
def test_the_host_can_tap_a_reserved_seat_to_review_the_request():
    """games.table.js opens the request dialog when the HOST taps a reserved seat (and
    the seat pulses to invite that tap): the button must receive the click."""
    host = _seat_button("is-empty", "reserved", "host-review")
    assert computed(RULES, host, "pointer-events") == "auto"
    assert computed(RULES, host, "cursor") == "pointer"
    # a player waiting on the host has nothing to tap (the button is disabled too)
    assert computed(RULES, _seat_button("is-empty", "reserved"), "pointer-events") == "none"
    # an ordinary empty seat and a taken one stay tappable
    assert computed(RULES, _seat_button("is-empty"), "pointer-events") == "auto"
    assert computed(RULES, _seat_button("is-empty", "locked"), "pointer-events") == "none"


def test_the_reserved_seat_click_goes_to_the_request_dialog_for_the_host():
    js = (STATIC / "games.table.js").read_text(encoding="utf-8")
    i = js.index('const sit = el("button", "seat-sit"')
    handler = js[i:js.index("});", i)]
    assert "st.is_host && HG.ui && HG.ui.openRequest" in handler and "openRequest(i)" in handler
    assert "sv.sit.disabled = held && !s.is_host" in js


# --------------------------------------------------------------- A11Y-001
FADES_OUT = []  # (selector, keyframes) of every animation that ends invisible and holds it
_KF = keyframes(games_css())
for r in RULES:
    v = r.decls.get("animation", ("", False))[0]
    m = re.match(r"([\w-]+)\s.*\bforwards\b", v)
    if m and m.group(1) in _KF:
        last = re.findall(r"(?:100%|to)\s*\{([^}]*)\}", _KF[m.group(1)])
        if last and re.search(r"opacity:\s*0(?:\.0*)?\s*(;|$)", last[-1].strip()):
            FADES_OUT.append((r.selector, m.group(1)))
# exit animations: the element is removed as soon as they end (games.table.js muck, toasts)
EXITS = {"muck", "toastout"}


def _el_for(selector: str):
    """An element path the selector matches (its compound classes, under #stage)."""
    last = selector.split()[-1]
    return [root(), el("body"), el("div#stage"), el("div#fx"), el("div" + last)]


def test_every_fade_out_animation_is_known():
    names = {k for _, k in FADES_OUT}
    assert {"bubble", "emote", "floatup"} <= names, names
    assert names <= {"bubble", "emote", "floatup"} | EXITS, names


@pytest.mark.parametrize("anim_attr,media", [("off", ()), (None, (REDUCE,))])
def test_with_motion_off_bubbles_emotes_and_win_amounts_stay_visible(anim_attr, media):
    """Squeezed to 0.01 ms, an animation that fades in AND out jumps to its invisible last
    frame: with Animations off (or the device's Reduce Motion before the client runs)
    those must not animate at all — they just show until the client removes them."""
    attrs = {"data_anim": anim_attr} if anim_attr else {}
    for sel, kf in FADES_OUT:
        if kf in EXITS:
            continue
        path = _el_for(sel)
        path[0] = root(**attrs)
        assert computed(RULES, path, "animation", media) == "none", (sel, anim_attr)
        # ...and with motion on they still animate
        on = _el_for(sel)
        on[0] = root(data_anim="full")
        assert computed(RULES, on, "animation", media) != "none", sel


def test_reduced_motion_stops_endless_animations_instead_of_strobing():
    for attrs, media in (({"data_anim": "off"}, ()), ({}, (REDUCE,))):
        path = [root(**attrs), el("body"), el("div#seats"), el("div.seat.is-actor"), el("div.seat-av")]
        assert computed(RULES, path, "animation-iteration-count", media) == "1"
        assert computed(RULES, path, "animation-duration", media) == "0.01ms"
    # an explicit "Full" wins over the device setting
    path = [root(data_anim="full"), el("body"), el("div#seats"), el("div.seat.is-actor"), el("div.seat-av")]
    assert computed(RULES, path, "animation-duration", (REDUCE,)) is None


# --------------------------------------------------------------- A11Y-002 (games.js motion)
PREFS_HARNESS = r"""
const fs = require("fs"), vm = require("vm");
const cases = JSON.parse(process.argv[3]);
const out = [];
for (const c of cases) {
  const store = {}; if (c.stored) store["hg.prefs.v1"] = JSON.stringify(c.stored);
  const mq = { matches: !!c.reduce, addEventListener() {} };
  const root = { dataset: {} };
  const ctx = {
    console, JSON, Math, Number, String, Object, Array, Set, Map, Promise,
    localStorage: { getItem: (k) => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = v; } },
    matchMedia: () => mq,
    document: { documentElement: root, getElementById: () => null, hidden: false },
    location: { pathname: "/games" }, history: { replaceState() {}, pushState() {} },
    setInterval() {}, clearInterval() {}, setTimeout() {}, clearTimeout() {},
  };
  ctx.globalThis = ctx;
  vm.createContext(ctx);
  let src = fs.readFileSync(process.argv[2], "utf8").replace(/\r\n/g, "\n").replace(/\ninit\(\);\s*$/, "\n");
  src += "\n;globalThis.__t = { G, loadPrefs, applyPrefs, motionOn, savePrefs };";
  vm.runInContext(src, ctx);
  const t = ctx.__t;
  t.loadPrefs(); t.applyPrefs();
  const r = { anim: t.G.prefs.anim, on: t.motionOn(), attr: root.dataset.anim };
  if (c.pick) { t.savePrefs({ anim: c.pick }); r.afterPick = { on: t.motionOn(), attr: root.dataset.anim, stored: JSON.parse(store["hg.prefs.v1"]) }; }
  out.push(r);
}
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def node():
    if node_exe() is None:
        pytest.skip("node is not installed")
    return node_exe()


def test_motion_follows_the_device_unless_the_player_chose(node, tmp_path):
    import json
    cases = [
        {"stored": None, "reduce": False},                       # new player
        {"stored": None, "reduce": True},                        # new player, Reduce Motion on
        {"stored": {"anim": "full", "felt": "royal"}, "reduce": True},  # v1 default "full" = not a choice
        {"stored": {"anim": "off"}, "reduce": False},            # explicit Off stays
        {"stored": {"anim": "full", "pv": 2}, "reduce": True},   # explicit Full (v2) wins
        {"stored": None, "reduce": True, "pick": "full"},        # picking Full in the dialog
    ]
    got = run_node(PREFS_HARNESS, STATIC / "games.js", json.dumps(cases), tmp=tmp_path)
    assert [(g["anim"], g["on"], g["attr"]) for g in got] == [
        ("auto", True, "full"),
        ("auto", False, "off"),
        ("auto", False, "off"),
        ("off", False, "off"),
        ("full", True, "full"),
        ("auto", False, "off"),
    ]
    pick = got[5]["afterPick"]
    assert pick["on"] is True and pick["attr"] == "full"
    assert pick["stored"]["anim"] == "full" and pick["stored"]["pv"] == 2  # (so it stays a choice)


def test_the_felt_reads_the_effective_motion():
    js = (STATIC / "games.table.js").read_text(encoding="utf-8")
    line = next(x for x in js.splitlines() if "const anim = () =>" in x)
    assert "motionOn" in line
    ui = (STATIC / "games.ui.js").read_text(encoding="utf-8")
    assert '["auto", "Auto"], ["full", "Full"], ["off", "Off"]' in ui


# --------------------------------------------------------------- FE-005 (the UI modules)
UI_MODULES = ["games.ui.js", "games.lobby.js", "games.history.js", "games.review.js", "games.seat.js", "games.manage.js"]

LOAD_HARNESS = r"""
const fs = require("fs"), vm = require("vm");
const ctx = { console, JSON, Math, Number, String, Object, Array, Set, Map, Promise };
ctx.globalThis = ctx;
vm.createContext(ctx);
for (const f of JSON.parse(process.argv[2])) vm.runInContext(fs.readFileSync(f, "utf8"), ctx, { filename: f });
console.log(JSON.stringify({ ui: Object.keys(ctx.HG.ui).filter((k) => typeof ctx.HG.ui[k] === "function" || k === "TAGS"),
  kit: Object.keys(ctx.HG.uikit), inits: ctx.HG.ui.onInit.length, state: typeof ctx.HG.uiState }));
"""


@pytest.fixture(scope="module")
def public_server(boot_public_server):
    return boot_public_server(PLO5BP_ADMIN_EMAILS="admin@clientui.example")


def test_every_client_file_is_served_gated_and_loaded_in_order(public_server):
    from starlette.testclient import TestClient
    hg = sys.modules["plo5bp.ui.homegame"]
    on_disk = sorted(p.name for p in STATIC.glob("games.*js")) + sorted(p.name for p in STATIC.glob("games*.css"))
    assert sorted(hg.GAMES_ASSETS) == sorted(on_disk)
    signed = TestClient(public_server.app, raise_server_exceptions=False)
    assert signed.get("/auth/dev", params={"email": "admin@clientui.example"}).status_code == 200
    anon = TestClient(public_server.app, raise_server_exceptions=False)
    for name in hg.GAMES_ASSETS:  # gated under /games/static, never on the public /static mount
        assert signed.get(f"/games/static/{name}").status_code == 200, name
        assert anon.get(f"/games/static/{name}").status_code == 404, name
        assert signed.get(f"/static/{name}").status_code == 404, name
    page = (STATIC / "games.html").read_text(encoding="utf-8")
    order = re.findall(r'<script src="/games/static/([\w.]+\.js)', page)
    assert sorted(order) == sorted(n for n in on_disk if n.endswith(".js"))
    assert order[-1] == "games.js"  # the core runs init() once everything is defined
    base = order.index("games.ui.js")
    assert all(order.index(m) > base for m in UI_MODULES[1:])  # (the modules read HG.uikit at load)
    assert order.index("games.play.js") > base


def test_the_ui_modules_provide_every_hg_ui_function_the_other_files_call(node, tmp_path):
    import json
    got = run_node(LOAD_HARNESS, json.dumps([str(STATIC / m) for m in UI_MODULES]), tmp=tmp_path)
    called = set()
    for p in STATIC.glob("games*.js"):
        called |= set(re.findall(r"HG\.ui\.(\w+)", p.read_text(encoding="utf-8")))
        called |= set(re.findall(r"\bUI\.(\w+)\(", p.read_text(encoding="utf-8")))
    called -= {"onInit"}
    missing = sorted(called - set(got["ui"]))
    assert not missing, missing
    assert got["inits"] >= 1 and got["state"] == "object"


# --------------------------------------------------------------- MOB-002 / A11Y-003 / A11Y-006
def _rules_raw():
    css = re.sub(r"/\*.*?\*/", "", games_css(), flags=re.S)
    return re.findall(r"([^{}]+)\{([^{}]*)\}", css)


def test_older_safari_gets_its_fallbacks():
    for sel, body in _rules_raw():
        if re.search(r"(?<![-\w])backdrop-filter:", body):
            assert "-webkit-backdrop-filter:" in body, sel.strip()  # (Safari < 18)
        if "color-mix(" in body:
            bgs = re.findall(r"(?<![-\w])background:\s*([^;]+)", body)
            assert len(bgs) >= 2 and "color-mix(" not in bgs[0], sel.strip()  # (a plain one first: Safari < 16.2)


def _lum(hexc):
    r, g, b = (int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def test_small_grey_text_is_readable_and_switches_show_focus():
    tx3 = re.search(r"--tx-3:\s*(#[0-9a-fA-F]{6})", games_css()).group(1)
    for panel in ("#131c2a", "#111926", "#0c121b"):  # (dialogs, the drawer, the rail)
        hi, lo = sorted((_lum(tx3), _lum(panel)), reverse=True)
        assert (hi + 0.05) / (lo + 0.05) >= 4.5, (tx3, panel)
    path = [root(), el("body"), el("label.switch"), el("input", ("focus-visible",)), ]
    rules = parse_css(games_css())
    # the drawn switch (the <i> after the hidden checkbox) carries the focus ring
    sel = [r for r in rules if "focus-visible + i" in r.selector and r.decls.get("outline")]
    assert sel and "var(--accent)" in sel[0].decls["outline"][0]
