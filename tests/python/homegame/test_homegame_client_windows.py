"""Home-games windows (games.ui.js + its feature modules), driven in Node with the real
games.js core and a tiny DOM (hg_mini_dom.js). 2026-09-28 improvements:

- Manage: every control saves on its own, with a "Saved" tick, and the form is never
  rebuilt under the host's pointer while stacks move — they update in place (HGT-015,
  HGT-009); tabs are named Settings / … / Controls with tab roles (HGT-018, A11Y-009);
- the lobby keeps its table cards by id — a new hand only updates "#N" (HGL-008), and
  says "1 live · 1 paused" (HGL-003);
- Chips is one dialog with an Add / Take off switch (HGT-016); on a table with no maximum
  a typed amount above the slider is kept (HGT-010);
- "Host a table" offers what the server deals and every seat count (FE-004, HGT-029);
- a private note is saved when the card closes, typed or not (HGT-028).
Skipped when Node is not installed."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hg_client_tools import STATIC, UI_BOOT, UI_FILES, node_exe, run_node  # noqa: E402

FILES = UI_FILES


@pytest.fixture(scope="module")
def node():
    if node_exe() is None:
        pytest.skip("node is not installed")
    return node_exe()


PRELUDE = UI_BOOT


def _run(tmp_path, body):
    return run_node(PRELUDE + body, STATIC, json.dumps(FILES), tmp=tmp_path)


def test_manage_saves_each_control_on_its_own_and_keeps_its_form(node, tmp_path):
    got = _run(tmp_path, r"""
(async () => {
  const B = boot(() => table()), out = {};
  let s = table();
  B.HG.core.G.state = s; B.HG.core.G.gameId = "T1";
  B.HG.ui.openDrawer("game");
  const dr = () => B.W.doc.querySelector(".drawer");
  out.tabs = dr().querySelectorAll(".dr-tabs button").map((b) => [b.textContent, b.getAttribute("role"), b.getAttribute("aria-selected")]);
  out.footer = !!dr().querySelector(".dr-foot") || !!dr().querySelector("#m-save");
  const rabbit = dr().querySelector("#m-rabbit");
  rabbit.checked = false; rabbit.dispatch("change"); await flush();
  const name = dr().querySelector("#m-name");
  name.value = "Friday late"; name.dispatch("change"); await flush();
  const min = dr().querySelector("#m-min");
  min.value = "20"; min.dispatch("change"); await flush();
  out.posts = B.posts.map((p) => [p.url.split("/").pop(), p.body]);
  out.saved = rabbit.closest(".setrow").classList.contains("saved");
  // the Players tab: a bet moves a stack — the form stays, the number changes in place
  B.HG.ui.openDrawer("players");
  const btn = dr().querySelector("[data-away]");
  s = table({ rev: 2, seats: table().seats.map((x, i) => (i === 1 ? Object.assign({}, x, { stack_cents: 4700 }) : x)) });
  B.HG.core.G.state = s;
  B.HG.ui.paintDrawer(s, false);
  out.same = dr().querySelector("[data-away]") === btn;
  out.stack = dr().querySelector('[data-stk="2"]').textContent;
  console.log(JSON.stringify(out));
})();
""")
    assert [t[0] for t in got["tabs"]] == ["Settings", "Chips", "Pace", "Players", "Controls"]
    assert all(t[1] == "tab" for t in got["tabs"]) and got["tabs"][0][2] == "true"
    assert got["footer"] is False
    assert got["posts"] == [
        ["settings", {"allow_rabbit": False}],
        ["settings", {"name": "Friday late"}],
        ["settings", {"min_buyin_cents": 2000, "default_buyin_cents": 4000, "max_buyin_cents": 0}],
    ]
    assert got["saved"] is True
    assert got["same"] is True and got["stack"] == "$47.00"


def test_the_lobby_keeps_its_cards_by_table(node, tmp_path):
    got = _run(tmp_path, r"""
const B = boot(), out = {};
const card = (id, running, hand) => ({ id, name: "T " + id, host_name: "Host", is_host: false, is_seated: false, running, hand_no: hand, num_seats: 6, seated: 2,
  bb_cents: 100, ante_cents: 300, variant: "plo5", listed: true, players: [], club_id: "C1" });
const data = (h) => ({ club: "C1", clubs: [{ id: "C1", name: "Club", members: 3, role: "member", requests: 0 }], tables: [card("A", true, h), card("B", false, 2)], sessions: [], games: [] });
B.HG.core.G.clubId = "C1";
B.HG.ui.renderLobby(data(7));
const a = B.$("lb-open").querySelector('[data-id="A"]');
B.HG.ui.renderLobby(data(8));
const a2 = B.$("lb-open").querySelector('[data-id="A"]');
out.same = a === a2;
out.pill = a2.querySelector(".tcard-hand").textContent;
out.count = B.$("lb-open-count").textContent;
out.kicker = B.$("club-kicker").textContent;
out.meta = B.$("club-meta").textContent;
console.log(JSON.stringify(out));
""")
    assert got["same"] is True and got["pill"] == "#8"
    assert got["count"] == "1 live · 1 paused"
    assert got["meta"] == "3 members · you're a member"


def test_chips_is_one_dialog_and_a_no_limit_amount_is_kept(node, tmp_path):
    got = _run(tmp_path, r"""
(async () => {
  const B = boot(() => table()), out = {};
  const s = table({ settings: Object.assign({}, table().settings, { allow_rathole: true }) });
  B.HG.core.G.state = s; B.HG.core.G.gameId = "T1";
  B.HG.ui.openTopUp();
  const m = () => B.W.doc.querySelectorAll("#modal-root .modal").slice(-1)[0];
  out.dialogs = B.W.doc.querySelectorAll("#modal-root .modal").length;
  out.seg = m().querySelectorAll("#ch-mode button").map((b) => b.textContent);
  out.ok = m().querySelectorAll(".m-foot button").map((b) => b.textContent);
  const inp = m().querySelector(".money input");
  inp.value = "900"; inp.dispatch("input"); inp.dispatch("change");
  out.big = m().querySelector(".md-big").textContent;
  out.top = m().querySelectorAll(".sz-presets button").map((b) => b.textContent).slice(-1)[0];
  m().querySelector('#ch-mode button[data-v="remove"]').click();
  out.removeOk = m().querySelectorAll(".m-foot button").map((b) => b.textContent);
  m().querySelector('#ch-mode button[data-v="add"]').click();
  const inp2 = m().querySelector(".money input");
  inp2.value = "900"; inp2.dispatch("input"); inp2.dispatch("change");
  m().querySelectorAll(".m-foot button").slice(-1)[0].click(); await flush();
  out.posted = B.posts.map((p) => [p.url.split("/").pop(), p.body]);
  console.log(JSON.stringify(out));
})();
""")
    assert got["dialogs"] == 1
    assert got["seg"] == ["Add chips", "Take chips off"]
    assert got["ok"] == ["Cancel", "Add chips"] and got["removeOk"] == ["Cancel", "Take off"]
    assert got["big"] == "$900.00"  # (above the slider's $200 end: no table maximum)
    assert got["top"] == "$200"  # (the slider's end is an amount, not "Max")
    assert got["posted"] == [["rebuy", {"amount_cents": 90000, "queue": True}]]


def test_host_a_table_offers_what_the_server_deals(node, tmp_path):
    got = _run(tmp_path, r"""
(async () => {
  const B = boot((url) => (url.includes("host_prefs") ? { prefs: null } : {}));
  B.HG.core.G.me = { name: "" };
  B.HG.core.G.clubId = "C1";
  B.HG.table.EMOTES = {};
  B.HG.ui.init();  // (the lobby wires its buttons from UI.onInit)
  const g = (code, label, max, avail) => ({ code, label, name: label + " bomb pot", hole: 5, dealt: code === "plo67" ? 4 : 5, burns: code === "plo67" ? 3 : 0, max_seats: max, graded: code === "plo5", available: avail });
  B.HG.ui.renderLobby({ club: "C1", clubs: [{ id: "C1", name: "Club", members: 2, role: "owner", requests: 0 }], tables: [], sessions: [],
    games: [g("plo5", "PLO5", 8, true), g("plo6", "PLO6", 7, true), g("plo67", "PLO67", 5, false)] });
  B.$("c-open").click();
  await flush(); B.W.advance(10); await flush();
  const m = B.W.doc.querySelectorAll("#modal-root .modal").slice(-1)[0];
  console.log(JSON.stringify({
    games: m.querySelectorAll("#c-game button").map((b) => b.textContent),
    seats: m.querySelectorAll("#c-seats button").map((b) => b.textContent),
    clock: m.querySelectorAll("#c-clock button").map((b) => b.textContent),
    name: m.querySelector("#c-name").value,
  }));
})();
""")
    assert got["games"] == ["PLO5", "PLO6"]  # (this server can't deal PLO67)
    assert got["seats"] == ["2", "3", "4", "5", "6", "7", "8"]
    assert "10s" in got["clock"]
    assert got["name"] == "Home game"  # (no name on the account: never "My's game")


def test_a_private_note_is_saved_when_the_card_closes(node, tmp_path):
    got = _run(tmp_path, r"""
const B = boot();
const s = table();
B.HG.core.G.state = s; B.HG.core.G.gameId = "T1";
B.HG.ui.openPlayer(1);
const m = B.W.doc.querySelectorAll("#modal-root .modal").slice(-1)[0];
const box = m.querySelector("#pc-note");
box.value = "calls everything"; box.dispatch("input");
B.HG.ui.closeTop();  // (Escape: no change event)
console.log(JSON.stringify({ note: B.HG.ui.noteFor(2), stored: JSON.parse(B.store["hg.notes.v1"] || "{}") }));
""")
    assert got["note"] == {"tag": "none", "text": "calls everything"}
    assert got["stored"]["2"]["text"] == "calls everything"


def test_the_history_tab_loads_older_hands_and_keeps_them(node, tmp_path):
    """HGH-001 / HGH-005: "Load older hands" pages back with before=, a refresh after the
    next hand keeps what was loaded, and every row shows both boards."""
    got = _run(tmp_path, r"""
(async () => {
  const mk = (a, b) => Array.from({ length: a - b + 1 }, (_, i) => ({ hand_no: a - i, board_a: [1, 2, 3, 4, 5], board_b: [6, 7, 8, 9, 10], my_hole: [11, 12, 13, 14, 15], winners: [], pot_cents: 900 }));
  let newest = 60;
  const B = boot((url) => {
    if (!url.includes("/hands")) return table();
    const m = url.match(/before=(\d+)/);
    if (m) return { hands: mk(Number(m[1]) - 1, 1), more: false, stats: [] };
    return { hands: mk(newest, newest - 39), more: true, stats: [] };
  });
  B.HG.table.EMOTES = {};
  B.HG.cards.cardEl = () => B.W.doc.createElement("span");
  let s = table({ last_hand_no: 60 });
  B.HG.core.G.state = s; B.HG.core.G.gameId = "T1"; B.HG.core.G.prefs.rail = true;
  B.HG.ui.setRail(true, "hands"); await flush(); B.W.advance(50); await flush();
  const rows = () => B.$("hands-body").querySelectorAll(".hand-row").length;
  const out = { first: rows(), boards: B.$("hands-body").querySelector(".hand-row").querySelectorAll(".boards2 .mini-cards").length };
  B.$("hands-body").querySelectorAll("button.btn").find((b) => /older/.test(b.textContent)).click();
  await flush(); B.W.advance(50); await flush();
  out.paged = rows();
  out.more = B.$("hands-body").querySelectorAll("button.btn").some((b) => /older/.test(b.textContent));
  newest = 61; s = table({ last_hand_no: 61 }); B.HG.core.G.state = s;
  B.HG.ui.render(s, s); await flush(); B.W.advance(50); await flush();
  out.refreshed = rows();
  console.log(JSON.stringify(out));
})();
""")
    assert got["first"] == 40 and got["boards"] == 2
    assert got["paged"] == 60 and got["more"] is False
    assert got["refreshed"] == 61  # (the newest page came back, the older ones stayed)


def test_a_hand_link_opens_its_table_and_the_replayer(node, tmp_path):
    """FEAT-010: /games/t/<id>?hand=N opens the table, then hand N on top."""
    got = _run(tmp_path, r"""
(async () => {
  const B = boot(() => table());
  B.ctx.location.pathname = "/games/t/T1"; B.ctx.location.search = "?hand=7";
  const opened = [];
  B.HG.ui.openHand = (g, n) => opened.push([g, n]);
  B.HG.ui.showTable = () => {}; B.HG.ui.render = () => {};
  B.ctx.EventSource = undefined;
  await vm.runInContext("route()", B.ctx);
  console.log(JSON.stringify({ opened }));
})();
""")
    assert got["opened"] == [["T1", 7]]


def test_the_replayer_runs_an_all_in_out_street_by_street_with_its_equities(node, tmp_path):
    """(owner, 2026-09-29) "I would like all in equities to be shown in the hand
    histories": after the last action the replayer steps through the runout — each
    street's equities under the players' plates, then the result — and the action list
    has a row per runout street. (2026-10-05: the result's place is kept from the start,
    "pending" until the hand is over, and the network's card is always there -- drawn empty
    where it has no read -- so nothing in the dialog changes size while stepping.)"""
    got = _run(tmp_path, r"""
(async () => {
  const seat = (i, name, delta) => ({ seat: i, name, is_me: i === 0, start_cents: 20000, delta_cents: delta,
    hole: [20 + 5 * i, 21 + 5 * i, 22 + 5 * i, 23 + 5 * i, 24 + 5 * i], shown: true, folded: false });
  const rec = { v: 2, hand_no: 9, variant: "plo5", hole_count: 5, button: 0, num_seats: 6, bb_cents: 100, ante_cents: 300,
    pot_cents: 40000, showdown: true, board_a: [0, 4, 8, 12, 16], board_b: [1, 5, 9, 13, 17], burns: [],
    equities: { "3": { "0": [0.62, 0.41], "1": [0.38, 0.59] }, "4": { "0": [0.9, 0.2], "1": [0.1, 0.8] } }, runout_from: 3,
    actions: [{ seat: 1, street: "flop", action: 2, label: "Raise to $12.00", cents: 1200 },
              { seat: 0, street: "flop", action: 9, label: "All-in $197.00", cents: 19700 },
              { seat: 1, street: "flop", action: 1, label: "Call $185.00", cents: 18500 }],
    seats: [seat(0, "Host", 19700), seat(1, "Dana", -19700)], awards: [], flows: [], grades: [] };
  const B = boot((url) => (url.includes("/hands/9") ? rec : table()));
  B.HG.cards.cardEl = () => B.W.doc.createElement("span");
  B.HG.core.G.state = table(); B.HG.core.G.gameId = "T1";
  await B.HG.ui.openHand("T1", 9); await flush();
  const m = () => B.W.doc.querySelectorAll("#modal-root .modal").slice(-1)[0];
  const look = () => [m().querySelector("#rp-step").textContent, m().querySelectorAll(".rp-eq").map((e) => e.textContent),
    m().querySelector("#rp-banner").textContent, m().querySelector("#rp-result").classList.contains("pending")];
  const out = { start: look() };
  out.net = [m().querySelector("#rp-net").hidden, m().querySelector("#rp-net").classList.contains("empty")];
  out.rows = m().querySelectorAll("#rp-list .log-row.k-runout").map((r) => r.textContent);
  m().querySelectorAll("#rp-list .log-row")[2].click();  // just after the call: all in on the flop
  out.allin = look();
  m().querySelectorAll("#rp-list .log-row.k-runout")[0].click();
  out.turn = look();
  m().querySelector("#rp-last").click();
  out.end = look();
  out.nextOff = m().querySelector("#rp-next").disabled;
  console.log(JSON.stringify(out));
})();
""")
    assert got["start"][0] == "0 / 5" and got["start"][1] == [] and got["start"][3] is True
    assert got["net"] == [False, True]  # (there from the start, empty before any decision)
    assert got["rows"] == ["All inrun out — the equities", "All inrun out — the result"]
    step, eqs, banner, hidden = got["allin"]
    assert step == "3 / 5" and eqs == ["62%41%", "38%59%"]
    assert "Call $185.00" in banner and "running it out" in banner and hidden is True
    step, eqs, banner, hidden = got["turn"]
    assert step == "4 / 5" and eqs == ["90%20%", "10%80%"] and "Turn dealt" in banner
    step, eqs, banner, hidden = got["end"]
    assert step == "5 / 5" and eqs == [] and "River dealt" in banner and "Hand over" in banner
    assert hidden is False and got["nextOff"] is True


def test_the_replayer_shows_each_action_where_it_happened(node, tmp_path):
    """(owner, 2026-10-05) "The hand history is always like a step ahead on the table": a
    step shows ITS action -- that player in the light, that street's chips and boards (the
    river comes with the river's first action) -- the step after the last one is the
    result, the dealer button is a disc on the felt (not a "D" in a nameplate), and a
    shown-down player's graded decision carries its mark like yours."""
    got = _run(tmp_path, r"""
(async () => {
  const seat = (i, name, delta) => ({ seat: i, name, is_me: i === 0, start_cents: 20000, delta_cents: delta,
    hole: [20 + 5 * i, 21 + 5 * i, 22 + 5 * i, 23 + 5 * i, 24 + 5 * i], shown: true, folded: i === 1 });
  const rec = { v: 3, hand_no: 6, variant: "plo5", hole_count: 5, button: 1, num_seats: 3, bb_cents: 100, ante_cents: 300,
    pot_cents: 2700, showdown: true, board_a: [0, 4, 8, 12, 16], board_b: [1, 5, 9, 13, 17], burns: [],
    actions: [{ seat: 1, street: "flop", action: 1, label: "Check", cents: 0 },
              { seat: 2, street: "flop", action: 2, label: "Bet $3.00", cents: 300 },
              { seat: 0, street: "flop", action: 1, label: "Call $3.00", cents: 300 },
              { seat: 1, street: "flop", action: 0, label: "Fold", cents: 0 },
              { seat: 2, street: "turn", action: 2, label: "Bet $6.00", cents: 600 },
              { seat: 0, street: "turn", action: 1, label: "Call $6.00", cents: 600 },
              { seat: 2, street: "river", action: 1, label: "Check", cents: 0 },
              { seat: 0, street: "river", action: 1, label: "Check", cents: 0 }],
    seats: [seat(0, "Host", 1200), seat(1, "Dana", -300), seat(2, "Rico", -900)], awards: [], flows: [],
    grades: [{ i: 4, seat: 2, score: 96, cat: "best" }, { i: 5, seat: 0, score: 3, cat: "blunder" }] };
  const B = boot((url) => (url.includes("/hands/6") ? rec : table()));
  B.HG.cards.cardEl = () => B.W.doc.createElement("span");
  B.HG.core.G.state = table(); B.HG.core.G.gameId = "T1";
  await B.HG.ui.openHand("T1", 6); await flush();
  const m = () => B.W.doc.querySelectorAll("#modal-root .modal").slice(-1)[0];
  const look = () => ({ step: m().querySelector("#rp-step").textContent,
    boards: m().querySelector(".rp-center").querySelectorAll(".mini-cards").map((e) => e.getAttribute("data-cards")),
    lit: m().querySelectorAll(".rp-seat.acting .rp-plate b").map((e) => e.textContent),
    bets: m().querySelectorAll(".rp-bet").map((e) => e.textContent),
    banner: m().querySelector("#rp-banner").textContent,
    on: m().querySelectorAll("#rp-list .log-row.on").map((r) => r.textContent),
    pending: m().querySelector("#rp-result").classList.contains("pending") });
  const out = { start: look() };
  const row = (i) => m().querySelectorAll("#rp-list .log-row")[i];
  row(5).click(); out.turnCall = look();
  m().querySelector("#rp-next").click(); out.riverFirst = look();
  m().querySelector("#rp-last").click(); out.end = look();
  out.rows = m().querySelectorAll("#rp-list .log-row").length;
  out.marks = m().querySelectorAll("#rp-list .log-row").map((r) => (r.querySelector(".grade") ? r.querySelector(".grade").textContent : ""));
  const disc = m().querySelector(".rp-felt .rp-dbtn");
  out.disc = disc ? [disc.textContent, disc.getAttribute("title")] : null;
  out.plateD = m().querySelectorAll(".rp-plate .rp-d").length;
  console.log(JSON.stringify(out));
})();
""")
    st = got["start"]
    assert st["step"] == "0 / 9" and st["lit"] == [] and "Rico acts first" not in st["banner"]
    assert "Dana acts first" in st["banner"] and st["boards"] == ["0,4,8", "1,5,9"]
    tc = got["turnCall"]  # the turn's last call: the turn, its chips, its caller
    assert tc["step"] == "6 / 9" and tc["boards"] == ["0,4,8,12", "1,5,9,13"]
    assert tc["lit"] == ["You"] and sorted(tc["bets"]) == ["$6.00", "$6.00"]
    assert "Call $6.00" in tc["banner"] and "TURN" in tc["banner"] and "RIVER" not in tc["banner"]
    assert tc["on"] == ["YouCall $6.00✗✗"]
    rf = got["riverFirst"]  # the river's first action brings the river
    assert rf["step"] == "7 / 9" and rf["boards"] == ["0,4,8,12,16", "1,5,9,13,17"]
    assert rf["lit"] == ["Rico"] and rf["bets"] == [] and "RIVER" in rf["banner"]
    end = got["end"]  # one step past the last action: the result
    assert end["step"] == "9 / 9" and end["lit"] == [] and "Hand over" in end["banner"] and end["pending"] is False
    assert end["on"] == ["Showdownthe pots paid"] and got["rows"] == 9
    assert got["marks"][4] == "✓✓" and got["marks"][5] == "✗✗"  # (Rico's shown-down bet is marked too)
    assert got["disc"] == ["D", "Dana has the button"] and got["plateD"] == 0


def test_the_replayer_gives_back_the_bet_nobody_matched(node, tmp_path):
    """(owner, 2026-10-02) $200 bets the $180 pot, $150 folds, $100 calls all in: the
    $80 nobody matched goes back to the bettor when the betting closes — the replay
    ends with a $380 pot, never a $460 one the bettor partly "wins" from themselves."""
    got = _run(tmp_path, r"""
(async () => {
  const seat = (i, name, start, delta, folded) => ({ seat: i, name, is_me: i === 1, start_cents: start, delta_cents: delta,
    hole: [20 + 5 * i, 21 + 5 * i, 22 + 5 * i, 23 + 5 * i, 24 + 5 * i], shown: !folded, folded });
  const rec = { v: 3, hand_no: 4, variant: "plo5", hole_count: 5, button: 2, num_seats: 3, bb_cents: 100, ante_cents: 6000,
    pot_cents: 38000, uncalled: { seat: 0, cents: 8000 }, showdown: true,
    board_a: [0, 4, 8, 12, 16], board_b: [1, 5, 9, 13, 17], burns: [],
    actions: [{ seat: 0, street: "flop", action: 2, label: "Bet $180.00", cents: 18000 },
              { seat: 1, street: "flop", action: 0, label: "Fold", cents: 0 },
              { seat: 2, street: "flop", action: 1, label: "Call $100.00", cents: 10000 }],
    seats: [seat(0, "Dana", 26000, 22000, false), seat(1, "Host", 21000, -6000, true), seat(2, "Rico", 16000, -16000, false)],
    awards: [], flows: [], grades: [] };
  const B = boot((url) => (url.includes("/hands/4") ? rec : table()));
  B.HG.cards.cardEl = () => B.W.doc.createElement("span");
  B.HG.core.G.state = table(); B.HG.core.G.gameId = "T1";
  await B.HG.ui.openHand("T1", 4); await flush();
  const m = () => B.W.doc.querySelectorAll("#modal-root .modal").slice(-1)[0];
  const look = () => ({ pot: m().querySelector(".rp-pot").textContent,
    stacks: m().querySelectorAll(".rp-plate .num").map((e) => e.textContent),
    bets: m().querySelectorAll(".rp-bet").map((e) => e.textContent) });
  m().querySelectorAll("#rp-list .log-row")[1].click();  // Dana bet, Host folded
  const out = { before: look() };
  m().querySelector("#rp-last").click();
  out.end = look();
  console.log(JSON.stringify(out));
})();
""")
    assert got["before"] == {"pot": "Pot $360.00", "stacks": ["$20.00", "$150.00", "$100.00"], "bets": ["$180.00"]}
    assert got["end"] == {"pot": "Pot $380.00", "stacks": ["$100.00", "$150.00", "$0.00"], "bets": []}


def test_private_notes_back_up_and_come_back(node, tmp_path):
    """FEAT-014: Preferences saves the notes to a file and restores them (a backup's note
    replaces this browser's for the same player; anything else is refused)."""
    got = _run(tmp_path, r"""
(async () => {
  const B = boot();
  B.ctx.FileReader = class { readAsText(f) { this.result = f.text; setTimeout(() => this.onload(), 0); } };
  B.ctx.Blob = class { constructor(parts) { B.ctx.__blob = parts.join(""); } };
  const toasts = []; B.HG.ui.toast = (m) => toasts.push(m);
  B.HG.core.G.state = null;
  const good = JSON.stringify({ kind: "wrapgto-home-games-notes", v: 1, notes: { "7": { tag: "red", text: "bluffs rivers" }, "x": { tag: "red" }, "8": { tag: "none", text: "" } } });
  let p = B.HG.ui.importNotes({ text: good }); B.W.advance(5); const ok = await p;
  const restored = B.HG.ui.noteFor(7);
  p = B.HG.ui.importNotes({ text: '{"notes": {}}' }); B.W.advance(5); const bad = await p;
  B.HG.ui.exportNotes();
  const saved = JSON.parse(B.ctx.__blob || "{}");
  const shown = B.$("toast-root").children.map((t) => t.textContent);
  console.log(JSON.stringify({ ok, restored, bad, saved: saved.notes, kind: saved.kind, toasts: shown }));
})();
""")
    assert got["ok"] is True and got["restored"] == {"tag": "red", "text": "bluffs rivers"}
    assert got["bad"] is False
    assert got["kind"] == "wrapgto-home-games-notes" and got["saved"] == {"7": {"tag": "red", "text": "bluffs rivers"}}
    assert "1 note restored" in got["toasts"] and "That isn't a notes backup" in got["toasts"]


def test_the_shuffle_pill_says_its_state_in_words_and_icon(node, tmp_path):
    """CPY-009 / HGT-012: idle = "Deck sealed" (a lock), a device that missed confirming
    a shuffle sits out with a reason, and the icon changes with the words (a phone shows
    only the icon)."""
    got = run_node(PRELUDE + r"""
const B = boot();
const fs2 = require("fs");
vm.runInContext(fs2.readFileSync(process.argv[2] + "/games.fair.js", "utf8"), B.ctx, { filename: "games.fair.js" });
const pill = () => { const b = B.$("tb-fair"); return [b.hidden, b.lastElementChild.textContent, b.querySelector("use").getAttribute("href"), b.title]; };
const base = { id: "T1", my_seat: 0, seats: [{ seat: 0 }], hand_no: 3 };
B.HG.fair.onState(Object.assign({}, base, { my_seat: null, fair: { supported: true, next: { hand_no: 4, stage: "commit", pending: false, hand_id: "h", seal: "s" } } }));
const idle = pill();
B.HG.fair.onState(Object.assign({}, base, { fair: { supported: true, next: { hand_no: 4, stage: "commit", pending: false, hand_id: "h", seal: "s", you: { seat: 0, barred: true, benched_hands: 3 } } } }));
const benched = pill();
console.log(JSON.stringify({ idle, benched }));
""", STATIC, json.dumps(["games.js", "games.table.js", "games.ui.js"]), tmp=tmp_path)
    assert got["idle"][:3] == [False, "Deck sealed", "#i-lock"]
    assert got["benched"][1:3] == ["Shuffle: sitting out", "#i-shield0"] and "next 3 hands" in got["benched"][3]


def test_menus_work_from_the_keyboard(node, tmp_path):
    """A11Y-013: a menu takes focus, arrows move through its items, Escape (closeTop)
    closes it and gives focus back to its button; the dock's hotkeys pause meanwhile."""
    got = _run(tmp_path, r"""
const B = boot();
const anchor = B.$("tb-more");
let picked = null;
B.HG.ui.openMenu(anchor, [{ header: "Table" }, { label: "One", onClick: () => { picked = 1; } }, "-", { label: "Two", onClick: () => { picked = 2; } }, { label: "Off", disabled: true, onClick() {} }]);
const menu = B.W.doc.querySelector(".menu");
const items = menu.querySelectorAll("[role=menuitem]");
const out = { role: menu.getAttribute("role"), focus0: B.W.doc.activeElement === items[0], expanded: anchor.getAttribute("aria-expanded") };
menu.dispatch("keydown", { key: "ArrowDown" }); out.down = B.W.doc.activeElement === items[1];
menu.dispatch("keydown", { key: "ArrowDown" }); out.wraps = B.W.doc.activeElement === items[0];
B.HG.ui.closeTop();
out.closed = !B.W.doc.querySelector(".menu"); out.back = B.W.doc.activeElement === anchor; out.expanded2 = anchor.getAttribute("aria-expanded");
console.log(JSON.stringify(out));
""")
    assert got["role"] == "menu" and got["focus0"] and got["expanded"] == "true"
    assert got["down"] and got["wraps"]  # (the disabled item is skipped)
    assert got["closed"] and got["back"] and got["expanded2"] == "false"


def test_the_chat_adds_new_lines_instead_of_redrawing(node, tmp_path):
    """FE-008: a new chat line is appended (earlier line nodes stay the same objects), a
    line the server's window dropped is removed, and a forced redraw still works."""
    got = _run(tmp_path, r"""
const B = boot();
B.HG.table.EMOTES = {};
const msg = (id) => ({ id, name: "Dana", user_id: 2, text: "hi " + id, created_at: new Date(1000 * id).toISOString() });
let s = table({ chat: [msg(1), msg(2)] });
B.HG.core.G.state = s; B.HG.core.G.gameId = "T1";
B.HG.ui.setRail(true, "chat");
const log = B.$("chat-log");
const first = log.firstElementChild;
s = table({ rev: 2, chat: [msg(1), msg(2), msg(3)] }); B.HG.core.G.state = s; B.HG.ui.render(s, s);
const out = { appended: log.childElementCount, same: log.firstElementChild === first, last: log.lastElementChild.textContent.includes("hi 3") };
s = table({ rev: 3, chat: [msg(2), msg(3), msg(4)] }); B.HG.core.G.state = s; B.HG.ui.render(s, s);
out.window = log.children.map((x) => x.querySelector(".txt").textContent);
console.log(JSON.stringify(out));
""")
    assert got["appended"] == 3 and got["same"] is True and got["last"] is True
    assert got["window"] == ["hi 2", "hi 3", "hi 4"]
