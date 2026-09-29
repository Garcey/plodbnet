"""Drive EVERY seat of a preview table (host included) - big showdowns and all-in
runouts for the felt checks (measure_showdown.js). Preview server, throwaway DB only.
Written for the PLO6 7-seat check (2026-09-26).

usage:
  drive.py seat <table> [n]          seat n guests (default 6) next to the host
  drive.py run  <table> [minutes] [style]
        every seat acts (host included) until the time is up.
        style: calls (check/call down: big showdowns) | shove (all-ins: runouts) | mix
"""
from __future__ import annotations

import random
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import bot  # noqa: E402

GUESTS = bot.GUESTS + [("zoe@example.com", "Zoe"), ("nate@example.com", "Nate"), ("jess@example.com", "Jess")]


def seat(tid: str, n: int) -> None:
    host = bot.Client(*bot.HOST)
    view = host.req("GET", f"/games/api/tables/{tid}")
    free = [s["seat"] for s in view["seats"] if s["empty"]]
    for (email, name), seat_no in zip(GUESTS[:n], free):
        g = bot.Client(email, name)
        out = g.req("POST", f"/games/api/tables/{tid}/sit", {"seat": seat_no, "buyin_cents": 20000})
        print("sit", name, seat_no, out.get("_detail", "ok"))


def run(tid: str, minutes: float, style: str) -> None:
    people = [bot.Client(*bot.HOST)] + [bot.Client(e, n) for e, n in GUESTS]
    path = f"/games/api/tables/{tid}"
    end = time.time() + minutes * 60
    host = people[0]
    v = host.req("GET", path)
    if not v.get("running"):
        print("start", host.req("POST", path + "/run", {"running": True}).get("_detail", "ok"))
    hand_style = {}
    while time.time() < end:
        acted = False
        for c in people:
            v = c.req("GET", path)
            if "_error" in v or v.get("my_seat") is None:
                continue
            me = v["seats"][v["my_seat"]]
            if not me["in_hand"] and me["stack_cents"] <= 3 * v["stakes"]["ante_cents"]:
                c.req("POST", path + "/rebuy", {"amount_cents": 20000})
            if v.get("actor") != v["my_seat"] or v.get("phase") != "in_hand":
                continue
            hs = hand_style.setdefault(v["hand_no"], style if style != "mix" else random.choice(["calls", "shove", "calls"]))
            legal, rb = v["legal"], v["raise_bounds"]
            body = {"hand_no": v["hand_no"], "action_seq": v["action_seq"]}
            if hs == "shove" and legal["raise"] and random.random() < 0.6:
                body.update(gate="raise", chips=rb["max_chips"])
            elif legal["check_call"]:
                body.update(gate="check_call", chips=0)
            else:
                body.update(gate="fold", chips=0)
            time.sleep(0.25)
            out = c.req("POST", path + "/act", body)
            if "_error" in out:
                print("act", c.name, body, out.get("_detail"))
            acted = True
            break
        if not acted:
            time.sleep(0.4)


if __name__ == "__main__":
    cmd, tid = sys.argv[1], sys.argv[2]
    if cmd == "seat":
        seat(tid, int(sys.argv[3]) if len(sys.argv) > 3 else 6)
    elif cmd == "run":
        run(tid, float(sys.argv[3]) if len(sys.argv) > 3 else 5, sys.argv[4] if len(sys.argv) > 4 else "mix")
