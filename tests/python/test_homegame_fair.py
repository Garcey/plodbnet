"""Home games — the verifiable shuffle (sealed deck + players' cut).

``plo5bp.ui.fairdeal`` is the spec; ``verify_transcript`` / ``verify_opening`` are
the independent checker. The API tests play the part of the players' browsers:
commit -> (lock) -> reveal -> deal, then check every card they are shown.
"""
from __future__ import annotations

import collections
import secrets
import sys

import pytest
from starlette.testclient import TestClient

from plo5bp.ui import fairdeal as fd

ADMIN_EMAIL = "themilesgarcia@icloud.com"
NAMES = ["ann", "ben", "cat", "dov"]
# fd.permutation("00" * 32): pinned — a change here is a change of the PUBLIC spec, and
# games.fair.js (checked against the same numbers in the Node harness) must follow.
KAT_ZERO_CUT = [1, 47, 26, 17, 9, 51, 29, 23, 0, 19, 21, 20, 2, 5, 44, 34, 31, 37, 49, 27, 24, 38, 7, 16, 4, 12, 3, 39, 11, 6, 13, 48, 28, 8, 42, 15, 33, 30, 36, 35, 32, 14, 46, 25, 43, 10, 18, 50, 45, 41, 40, 22]


# --- the spec, on its own --------------------------------------------------------------


def test_the_cut_is_a_deterministic_unbiased_looking_permutation():
    cut = fd.sha("any cut")
    p = fd.permutation(cut)
    assert sorted(p) == list(range(52)) and p == fd.permutation(cut)
    assert p != fd.permutation(fd.sha("another cut"))
    # KNOWN ANSWER (pins the algorithm — games.fair.js must produce the same)
    assert fd.permutation("00" * 32) == KAT_ZERO_CUT


def test_a_stacked_deck_is_still_dealt_at_random():
    """The whole point: the server may seal ANY order — the players' numbers,
    unknown to it at that moment, decide where every card goes."""
    stacked = list(range(52))  # aces on top, as rigged as it gets
    first_slot = collections.Counter()
    for k in range(2600):
        sd = fd.SealedDeck.create(f"t:{k}:1", 6, deck=stacked)
        n = secrets.token_hex(32)
        sd.set_lock({2: fd.nonce_commitment(sd.hand_id, sd.seal, 2, n)})
        first_slot[sd.finish({2: n})[0]] += 1
    assert len(first_slot) == 52, "every card reaches the first slot"
    assert max(first_slot.values()) < 100 and min(first_slot.values()) > 15  # mean 50


def _sealed(n_seats=6, contributors=(0, 3)):
    sd = fd.SealedDeck.create("game:7:1", n_seats)
    nonces = {s: secrets.token_hex(32) for s in contributors}
    sd.set_lock({s: fd.nonce_commitment(sd.hand_id, sd.seal, s, n) for s, n in nonces.items()})
    deck = sd.finish(nonces)
    return sd, nonces, deck


def test_a_transcript_checks_out_and_every_tampering_is_caught():
    sd, nonces, deck = _sealed()
    tr = sd.public()
    assert sorted(deck) == list(range(52))
    perm = fd.verify_transcript(tr, my_seat=3, my_nonce=nonces[3], seen_seal=sd.seal, seen_lock=sd.lock)
    assert [sd.deck[perm[j]] for j in range(52)] == deck
    # seat 3's second hole card sits in slot 5*3+1 and opens
    card = deck[fd.hole_slot(3, 1)]
    op = sd.opening(card)
    fd.verify_opening(tr, perm, card, op, expect_slots=range(15, 20))
    assert op["slot"] == 16
    # the store round-trips (history / later audits)
    again = fd.SealedDeck.from_store(sd.to_store())
    assert again.public() == tr and again.opening(card) == op

    def broken(**kw):
        bad = dict(tr)
        bad.update(kw)
        return bad

    with pytest.raises(fd.FairError):  # a different card under the same commitment
        fd.verify_opening(tr, perm, (card + 1) % 52, op)
    with pytest.raises(fd.FairError):  # dealt from the wrong place
        fd.verify_opening(tr, perm, card, op, expect_slots=range(0, 5))
    with pytest.raises(fd.FairError):  # a swapped position
        fd.verify_opening(tr, perm, card, dict(op, pos=(op["pos"] + 1) % 52))
    swapped = list(tr["commitments"])
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(fd.FairError):  # the sealed deck was changed
        fd.verify_transcript(broken(commitments=swapped))
    with pytest.raises(fd.FairError):  # a number that does not open its commitment
        fd.verify_transcript(broken(reveals=[[0, "ab" * 32], [3, nonces[3]]]))
    with pytest.raises(fd.FairError):  # a contributor dropped from the cut
        fd.verify_transcript(broken(locked=tr["locked"][:1], reveals=tr["reveals"][:1]))
    with pytest.raises(fd.FairError):  # "your number was used" — it was not
        fd.verify_transcript(tr, my_seat=3, my_nonce="cd" * 32)
    with pytest.raises(fd.FairError):  # the seal this device committed to was replaced
        fd.verify_transcript(tr, seen_seal="ef" * 32)
    with pytest.raises(fd.FairError):  # the cut was not derived from the numbers
        fd.verify_transcript(broken(cut=fd.sha("chosen by the server")))


def test_an_unopened_commitment_gives_nothing_away():
    sd, _, _ = _sealed()
    # without the salt, all 52 candidate cards are equally (in)consistent
    assert not any(fd.card_commitment(sd.hand_id, 0, "00" * 32, c) == sd.commitments[0] for c in range(52))
    assert len({sd.salt(i) for i in range(52)}) == 52


# --- the table -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server(boot_public_server):
    return boot_public_server(PLO5BP_HOMEGAME_GRADING="1")


@pytest.fixture(scope="module")
def hg(server):
    return sys.modules["plo5bp.ui.homegame"]


@pytest.fixture(scope="module")
def cast(server):
    def login(email):
        cl = TestClient(server.app, raise_server_exceptions=False)
        assert cl.get("/auth/dev", params={"email": email}).status_code == 200
        return cl

    adm = login(ADMIN_EMAIL)
    players = [login(f"{n}@fair.example") for n in NAMES]
    ids = {u["email"]: u["id"] for u in adm.get("/admin/api/users").json()["users"]}
    for n in NAMES:
        adm.post("/admin/api/games_access", json={"user_id": ids[f"{n}@fair.example"], "action": "grant"})
    return {"p": players}


def _post(cl, gid, what, body=None):
    return cl.post(f"/games/api/tables/{gid}/{what}", json=body or {})


def _state(cl, gid):
    r = cl.get(f"/games/api/tables/{gid}")
    assert r.status_code == 200, r.text
    return r.json()


def _table(cast, n, **kw):
    body = {"name": "fair", "sb_cents": 50, "bb_cents": 100, "ante_cents": 300,
            "default_buyin_cents": 20000, "num_seats": 6, **kw}
    gid = cast["p"][0].post("/games/api/tables", json=body).json()["id"]
    for i in range(1, n):
        assert _post(cast["p"][i], gid, "sit", {"seat": i, "buyin_cents": 20000}).status_code == 200
    return gid


class Device:
    """What games.fair.js does, in Python."""

    def __init__(self, cl, gid):
        self.cl, self.gid, self.mem = cl, gid, {}

    def commit(self):
        s = _state(self.cl, self.gid)
        nx = s["fair"]["next"]
        nonce = secrets.token_hex(32)
        c = fd.nonce_commitment(nx["hand_id"], nx["seal"], s["my_seat"], nonce)
        r = _post(self.cl, self.gid, "fair/commit", {"hand_id": nx["hand_id"], "commit": c})
        if r.status_code == 200:
            self.mem[nx["hand_id"]] = {"nonce": nonce, "seal": nx["seal"], "seat": s["my_seat"], "commit": c}
        return r

    def reveal(self):
        s = _state(self.cl, self.gid)
        nx = s["fair"]["next"]
        m = self.mem[nx["hand_id"]]
        assert nx["stage"] == "reveal" and nx["seal"] == m["seal"], "never reveal under a different seal"
        assert [m["seat"], m["commit"]] in nx["locked"], "never reveal unless you are in the lock list"
        assert fd.lock_of(nx["hand_id"], nx["seal"], [(a, b) for a, b in nx["locked"]]) == nx["lock"]
        m["lock"] = nx["lock"]
        return _post(self.cl, self.gid, "fair/reveal", {"hand_id": nx["hand_id"], "nonce": m["nonce"]})

    def check_my_cards(self):
        """Verify the transcript and EVERY card on this player's screen."""
        s = _state(self.cl, self.gid)
        h = s["fair"]["hand"]
        r = self.cl.get(f"/games/api/tables/{self.gid}/fair/{h['hand_no']}")
        assert r.status_code == 200, r.text
        tr = r.json()
        m = self.mem.get(h["hand_id"])
        perm = fd.verify_transcript(
            tr, my_seat=m and m["seat"], my_nonce=m and m["nonce"],
            seen_seal=m and m["seal"], seen_lock=m and m.get("lock"))
        n, seen = s["num_seats"], []
        for i, seat in enumerate(s["seats"]):
            for c in seat.get("hole") or []:
                if c >= 0:
                    fd.verify_opening(tr, perm, c, h["open"][str(c)], expect_slots=range(5 * i, 5 * i + 5))
                    seen.append(c)
        for b, off in (("a", 0), ("b", 5)):
            bd = s["board"][b]
            for mth, c in enumerate(list(bd["flop"]) + [bd["turn"], bd["river"]]):
                if c is not None:
                    fd.verify_opening(tr, perm, c, h["open"][str(c)], expect_slots=[5 * n + off + mth])
                    seen.append(c)
        assert sorted(int(k) for k in h["open"]) == sorted(seen), "proofs for exactly what is on screen"
        return s, tr, seen


def test_a_table_nobodys_browser_takes_part_in_deals_at_once_and_is_still_sealed(cast, hg):
    p = cast["p"]
    gid = _table(cast, 2)
    before = _state(p[0], gid)["fair"]
    assert before["supported"] and before["next"]["stage"] == "commit" and before["next"]["hand_no"] == 1
    assert _post(p[0], gid, "run", {"running": True}).json()["phase"] == "in_hand"
    s, tr, seen = Device(p[0], gid).check_my_cards()
    assert s["fair"]["hand"]["seal"] == before["next"]["seal"], "the deck dealt is the deck that was sealed"
    assert s["fair"]["hand"]["contributors"] == [] and tr["locked"] == []
    assert len(seen) == 11, "my five cards and the two flops — never the other player's"


def test_devices_cut_the_deck_and_each_player_can_prove_their_own_cards(cast, hg):
    p = cast["p"]
    gid = _table(cast, 3)
    devs = [Device(p[i], gid) for i in range(3)]
    assert devs[0].commit().status_code == 200 and devs[2].commit().status_code == 200
    lost_tab = Device(p[0], gid)  # the same player in a tab that is then closed …
    assert lost_tab.commit().status_code == 200
    assert devs[0].commit().status_code == 200, "… and the tab that is still open: the newest commitment counts"
    r = _post(p[0], gid, "run", {"running": True}).json()
    assert r["phase"] != "in_hand", "the deal waits for the devices' numbers"
    nx = r["fair"]["next"]
    assert nx["stage"] == "reveal" and nx["pending"] and [x[0] for x in nx["locked"]] == [0, 2]
    assert devs[1].commit().status_code == 409, "the list is frozen at the lock"
    assert _post(p[0], gid, "deal", {}).status_code == 200, "an impatient second click is not an error"
    bad = _post(p[0], gid, "fair/reveal", {"hand_id": nx["hand_id"], "nonce": "ab" * 32})
    assert bad.status_code == 400
    assert devs[0].reveal().status_code == 200
    assert _state(p[0], gid)["phase"] != "in_hand", "still waiting for seat 2"
    assert devs[2].reveal().status_code == 200
    states = [d.check_my_cards() for d in devs]
    assert all(s["phase"] == "in_hand" for s, _, _ in states)
    assert states[0][0]["fair"]["hand"]["contributors"] == [0, 2]
    # three players, three different sets of proven cards; the flops are common
    holes = [set(seen) - set(states[0][2][-6:]) for _, _, seen in states]
    assert all(len(h) == 5 for h in holes) and not (holes[0] & holes[1]) and not (holes[1] & holes[2])
    # the transcript is the same for everybody except for what each may open
    assert states[0][1]["cut"] == states[1][1]["cut"] == states[2][1]["cut"]


def test_a_device_that_withholds_its_number_voids_the_shuffle_in_public(cast, hg):
    p = cast["p"]
    gid = _table(cast, 3)
    devs = [Device(p[i], gid) for i in range(3)]
    for d in devs:
        assert d.commit().status_code == 200
    _post(p[0], gid, "run", {"running": True})
    first = _state(p[0], gid)["fair"]["next"]
    assert devs[0].reveal().status_code == 200 and devs[1].reveal().status_code == 200
    t = hg.HUB.get(gid)
    with t.lock:  # seat 2 (cat) never reveals: run the clock out
        t.fair_next.deadline_mono -= 60.0
        hg._fair_tick_locked(t)
    s = _state(p[0], gid)
    nx = s["fair"]["next"]
    assert s["phase"] != "in_hand" and nx["attempt"] == 2 and nx["seal"] != first["seal"]
    assert nx["stage"] == "commit" and nx["pending"], "a NEW deck is sealed; the deal still wants to happen"
    assert any(e["kind"] == "fair" and "cat" in e["text"] for e in s["events"]), "announced, with the name"
    assert s["fair"]["void_counts"] == {"cat": 1}
    assert devs[2].commit().status_code == 409, "the absentee sits this hand's shuffle out"
    assert devs[0].commit().status_code == 200 and devs[1].commit().status_code == 200
    assert _state(p[0], gid)["fair"]["next"]["stage"] == "reveal", "everyone expected is in: locked at once"
    assert devs[0].reveal().status_code == 200 and devs[1].reveal().status_code == 200
    s, tr, _ = devs[0].check_my_cards()
    assert s["phase"] == "in_hand" and s["fair"]["hand"]["contributors"] == [0, 1]
    assert tr["voids"] and tr["voids"][0]["names"] == ["cat"] and tr["voids"][0]["seal"] == first["seal"]
    devs[2].check_my_cards()  # cat is still dealt in and can still check every card they see


def test_pausing_calls_a_waiting_deal_off_and_throws_the_cut_deck_away(cast, hg):
    p = cast["p"]
    gid = _table(cast, 2)
    devs = [Device(p[i], gid) for i in range(2)]
    for d in devs:
        d.commit()
    _post(p[0], gid, "run", {"running": True})
    then = _state(p[0], gid)["fair"]["next"]
    sealed_then = then["seal"]
    assert devs[0].reveal().status_code == 200
    assert _post(p[0], gid, "run", {"running": False}).status_code == 200
    s = _state(p[0], gid)
    assert s["phase"] != "in_hand" and not s["fair"]["next"]["pending"]
    assert s["fair"]["next"]["seal"] != sealed_then and s["fair"]["next"]["attempt"] == 2
    stale = _post(p[1], gid, "fair/reveal", {"hand_id": then["hand_id"],
                                             "nonce": devs[1].mem[then["hand_id"]]["nonce"]})
    assert stale.status_code == 409, "that shuffle is gone"


def test_showdown_opens_the_tabled_hands_and_history_follows_the_reveal_rule(cast, hg):
    p = cast["p"]
    gid = _table(cast, 3)
    devs = [Device(p[i], gid) for i in range(3)]
    for d in devs:
        d.commit()
    _post(p[0], gid, "run", {"running": True})
    for d in devs:
        assert d.reveal().status_code == 200
    # seat order of action is the engine's business: seat 2 folds, the others check it down
    for _ in range(40):
        s = _state(p[0], gid)
        if s["phase"] != "in_hand":
            break
        a = s["actor"]
        _post(p[a], gid, "act", {"gate": "fold" if (a == 2 and _state(p[a], gid)["legal"]["fold"]) else "check_call",
                                 "hand_no": s["hand_no"], "action_seq": s["action_seq"]})
    t = hg.HUB.get(gid)
    with t.lock:  # skip the award animation
        if t.runout_active and t.runout_started_mono is not None:
            t.runout_started_mono -= 600.0
    s0, tr0, seen0 = devs[0].check_my_cards()
    assert s0["phase"] == "showdown"
    tabled = [i for i, seat in enumerate(s0["seats"]) if (seat.get("hole") or [-1])[0] >= 0]
    assert 0 in tabled and 1 in tabled, "both live hands are tabled — and both come with proofs"
    assert len(seen0) == 5 * len(tabled) + 10
    # history: the same transcript; a folded player's cards are opened to THEM only
    assert hg.wait_for_grading(30.0), "a verified hand is graded like any other (replayed from the dealt deck)"
    hist0 = p[0].get(f"/games/api/tables/{gid}/fair/1").json()
    hist2 = p[2].get(f"/games/api/tables/{gid}/fair/1").json()
    perm = fd.verify_transcript(hist2)
    folded_cards = {c for c, op in hist2["open"].items() if 10 <= op["slot"] < 15}
    assert hist0["cut"] == hist2["cut"]
    if 2 not in tabled:
        assert len(folded_cards) == 5 and not any(10 <= op["slot"] < 15 for op in hist0["open"].values())
    for c, op in hist2["open"].items():
        fd.verify_opening(hist2, perm, int(c), op)
    det = p[0].get(f"/games/api/tables/{gid}/hands/1").json()
    assert det["fair"] == {"hand_id": hist0["hand_id"], "contributors": [0, 1, 2], "voids": 0}
    assert det["grades"], "graded"
    outsider = p[3].get(f"/games/api/tables/{gid}/fair/1")
    assert outsider.status_code in (400, 403, 404), "history is for the table's members"


def test_a_device_that_went_to_sleep_does_not_hold_the_table_up(cast, hg):
    """A phone that committed and was then locked: it is not asked for its number
    (nobody waits three seconds on it, nobody is named and shamed for it)."""
    p = cast["p"]
    gid = _table(cast, 2)
    devs = [Device(p[i], gid) for i in range(2)]
    for d in devs:
        assert d.commit().status_code == 200
    t = hg.HUB.get(gid)
    with t.lock:
        uid1 = t.seats[1].user_id
        t.seen[uid1] -= 600.0  # ben's browser has not been heard from for ten minutes
    r = _post(p[0], gid, "run", {"running": True}).json()
    assert [x[0] for x in r["fair"]["next"]["locked"]] == [0], "only the device that is here is locked"
    assert devs[0].reveal().status_code == 200
    s, tr, _ = devs[0].check_my_cards()
    assert s["phase"] == "in_hand" and s["fair"]["hand"]["contributors"] == [0] and not tr["voids"]
    assert not s["fair"]["void_counts"]


def test_only_a_seated_player_takes_part(cast, hg):
    p = cast["p"]
    gid = _table(cast, 2)
    nx = _state(p[3], gid)["fair"]["next"]
    r = _post(p[3], gid, "fair/commit", {"hand_id": nx["hand_id"], "commit": "ab" * 32})
    assert r.status_code == 409
    assert _post(p[0], gid, "fair/commit", {"hand_id": nx["hand_id"], "commit": "nope"}).status_code == 400
    assert _post(p[0], gid, "fair/commit", {"hand_id": "x:1:1", "commit": "ab" * 32}).status_code == 409
    assert p[0].get(f"/games/api/tables/{gid}/fair/1").status_code == 404, "no hand dealt yet"
