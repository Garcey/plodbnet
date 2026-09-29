"""Home games: the client build (2026-09-28, FE-012 + OPS-039).

The page links every client file by a versioned address (`?v=<content hash>`) that
the browser keeps for a year (private — a shared cache must never serve the gated
files); any other address is never cached. The page names the build it was served
with and the lobby / table views name the server's current build: an open table
still running an older page offers a refresh between hands, and a background tab
with nothing to lose reloads itself once."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hg_client_tools import STATIC, node_exe, run_node  # noqa: E402

ADMIN_EMAIL = "admin@build.example"


@pytest.fixture(scope="module")
def server(boot_public_server):
    return boot_public_server(PLO5BP_ADMIN_EMAILS=ADMIN_EMAIL)


@pytest.fixture(scope="module")
def host(server):
    c = TestClient(server.app, raise_server_exceptions=False)
    assert c.get("/auth/dev", params={"email": ADMIN_EMAIL}).status_code == 200
    return c


def _hg():
    return sys.modules["plo5bp.ui.homegame"]


def test_the_page_links_every_client_file_by_its_content_hash(host):
    hg = _hg()
    page = host.get("/games").text
    links = dict(re.findall(r'/games/static/([\w.]+)\?v=([0-9a-f]{12})"', page))
    assert set(links) == set(hg.GAMES_ASSETS)  # every script and the stylesheet
    for name, v in links.items():
        assert v == hg._asset_hash(STATIC, name)
        r = host.get(f"/games/static/{name}?v={v}")
        assert r.status_code == 200
        cc = r.headers["cache-control"]
        assert "immutable" in cc and "max-age=31536000" in cc and "private" in cc, (name, cc)
        assert r.headers["x-content-type-options"] == "nosniff"
    # any other address is never cached: an old ?v= must not pin old code
    for url in ("/games/static/games.js", "/games/static/games.js?v=000000000000"):
        assert "no-store" in host.get(url).headers["cache-control"]
    # the page itself never is either
    assert "no-store" in host.get("/games").headers["cache-control"]


def test_the_page_and_the_views_name_the_same_build(host):
    page = host.get("/games").text
    m = re.search(r'<meta name="hg-build" content="([0-9a-f]{12})" />', page)
    assert m, "the page carries its build"
    build = m.group(1)
    lobby = host.get("/games/api/tables").json()
    assert lobby["client_build"] == build
    club = host.post("/games/api/clubs", json={"name": "Build club"}).json()
    t = host.post("/games/api/tables", json={"name": "b", "club_id": club["id"]}).json()
    assert t["client_build"] == build
    assert host.get(f"/games/api/tables/{t['id']}").json()["client_build"] == build


def test_any_client_file_change_is_a_new_build(server, tmp_path):
    hg = _hg()
    for name in [*hg.GAMES_ASSETS, "games.html"]:
        (tmp_path / name).write_text("x", encoding="utf-8")
    before = hg.client_build(tmp_path, fresh=True)
    (tmp_path / "games.table.js").write_text("changed", encoding="utf-8")
    after = hg.client_build(tmp_path, fresh=True)
    assert before and after and before != after
    hg.client_build(STATIC, fresh=True)  # (put the cache back on the real files)


# ------------------------------------------------------------------ the client half
HARNESS = r"""
const fs = require("fs"), vm = require("vm");
const cases = JSON.parse(process.argv[3]);
const out = [];
for (const c of cases) {
  const store = {}, shown = [];
  let reloads = 0;
  if (c.reloaded) store["hg.reloadedFor"] = c.reloaded;
  const ctx = {
    console, JSON, Math, Number, String, Object, Array, Set, Map, Promise,
    sessionStorage: { getItem: (k) => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = v; } },
    document: { querySelector: (q) => (q === 'meta[name="hg-build"]' ? { content: c.page } : null), hidden: !!c.hidden,
                activeElement: null, getElementById: () => null, documentElement: { dataset: {} } },
    location: { pathname: "/games", reload: () => { reloads++; } }, history: { replaceState() {}, pushState() {} },
    setInterval() {}, clearInterval() {}, setTimeout() {}, clearTimeout() {},
  };
  ctx.globalThis = ctx;
  vm.createContext(ctx);
  let src = fs.readFileSync(process.argv[2], "utf8").replace(/\r\n/g, "\n").replace(/\ninit\(\);\s*$/, "\n");
  src += "\n;globalThis.__t = { G, noteBuild };";
  vm.runInContext(src, ctx);
  ctx.HG.uiState = { modals: c.modal ? [1] : [], drawer: null };
  ctx.HG.ui = { showUpdate: (idle) => shown.push(idle) };
  const t = ctx.__t;
  t.G.gameId = c.state ? "T1" : null;
  t.G.state = c.state || null;
  t.noteBuild(c.server);
  out.push({ newBuild: t.G.newBuild || null, reloads, shown, stored: store["hg.reloadedFor"] || null });
}
console.log(JSON.stringify(out));
"""


def _seat(in_hand, folded=False):
    return {"seat": 0, "empty": False, "in_hand": in_hand, "folded": folded}


@pytest.fixture(scope="module")
def node():
    if node_exe() is None:
        pytest.skip("node is not installed")
    return node_exe()


def test_an_old_page_offers_the_refresh_only_when_it_costs_nothing(node, tmp_path):
    playing = {"my_seat": 0, "phase": "in_hand", "runout": {"blocking": False}, "seats": [_seat(True)]}
    between = {"my_seat": 0, "phase": "waiting", "runout": {"blocking": False}, "seats": [_seat(False)]}
    folded = {"my_seat": 0, "phase": "in_hand", "runout": {"blocking": False}, "seats": [_seat(True, folded=True)]}
    cases = [
        {"page": "aaaaaaaaaaaa", "server": "aaaaaaaaaaaa", "state": between},               # same build
        {"page": "aaaaaaaaaaaa", "server": "bbbbbbbbbbbb", "state": between},               # visible: ask
        {"page": "aaaaaaaaaaaa", "server": "bbbbbbbbbbbb", "state": playing},               # in a hand: wait
        {"page": "aaaaaaaaaaaa", "server": "bbbbbbbbbbbb", "state": playing, "hidden": True},  # hidden, but in a hand
        {"page": "aaaaaaaaaaaa", "server": "bbbbbbbbbbbb", "state": folded, "hidden": True},   # hidden + folded: reload
        {"page": "aaaaaaaaaaaa", "server": "bbbbbbbbbbbb", "state": None, "hidden": True},     # hidden lobby: reload
        {"page": "aaaaaaaaaaaa", "server": "bbbbbbbbbbbb", "state": None, "hidden": True, "modal": True},  # a dialog open
        {"page": "aaaaaaaaaaaa", "server": "bbbbbbbbbbbb", "state": None, "hidden": True, "reloaded": "bbbbbbbbbbbb"},  # once
        {"page": "", "server": "bbbbbbbbbbbb", "state": between},                            # an old server's page
    ]
    got = run_node(HARNESS, STATIC / "games.js", json.dumps(cases), tmp=tmp_path)
    assert [g["newBuild"] for g in got] == [None] + ["bbbbbbbbbbbb"] * 7 + [None]
    assert [g["reloads"] for g in got] == [0, 0, 0, 0, 1, 1, 0, 0, 0]
    assert got[1]["shown"] == [True] and got[2]["shown"] == [False] and got[3]["shown"] == [False]
    assert got[4]["stored"] == "bbbbbbbbbbbb"  # (so the reloaded page never loops)
    assert got[6]["shown"] == [True] and got[7]["shown"] == [True]


def test_the_table_offers_the_refresh_in_the_dock_between_hands():
    play = (STATIC / "games.play.js").read_text(encoding="utf-8")
    assert 'if (C().G.newBuild) btns.push(["update"' in play
    assert 'else if (k === "update") C().reloadForUpdate();' in play


def test_the_lobby_names_the_games_this_server_deals(host):
    """FE-004: the create dialog is built from the server's list (one source of truth)."""
    hg = _hg()
    games = host.get("/games/api/tables").json()["games"]
    assert [g["code"] for g in games] == list(hg.GAMES)
    for g in games:
        src = hg.GAMES[g["code"]]
        assert g["label"] == src["label"] and g["max_seats"] == src["max_seats"] and g["graded"] == src["graded"]
        assert g["available"] == bool(not src["burns"] or hg.PLO67_ON)


def test_chat_rows_name_their_user(host):
    """HGT-027: the client matches chat to people by user id, never by display name."""
    club = host.post("/games/api/clubs", json={"name": "Chat club"}).json()
    t = host.post("/games/api/tables", json={"name": "chat", "club_id": club["id"]}).json()
    assert host.post(f"/games/api/tables/{t['id']}/chat", json={"text": "hi"}).status_code == 200
    rows = host.get(f"/games/api/tables/{t['id']}").json()["chat"]
    me = host.get("/me").json()
    assert rows and all(isinstance(r["user_id"], int) for r in rows)
    assert rows[-1]["text"] == "hi" and rows[-1]["user_id"] == me.get("id", rows[-1]["user_id"])
