"""Scripted guests for the home-games preview server (throwaway DB only).

usage:
  bot.py setup <table_id> [n_guests]      log guests in, grant access, seat them
  bot.py play  <table_id> <style> [max]   act for guests until the HOST must act
                                          (or the hand ends). style: passive|bet|raise|fold
  bot.py chat  <table_id> <text>          guest 1 says something
  bot.py view  <table_id>                 dump a compact host-side view
  bot.py loop  <table_id> [minutes]       guests keep playing on their own (default 30 min)
  bot.py reload <table_id>                guests with less than two antes rebuy $40
"""
from __future__ import annotations

import http.cookiejar
import json
import os
import sys
import time
import urllib.error
import urllib.request

# The preview server's port: $PREVIEW_PORT (default 8772), like run_games_preview.py.
BASE = f"http://127.0.0.1:{os.environ.get('PREVIEW_PORT', '8772')}"
GUESTS = [
    ("dana@example.com", "Dana"),
    ("rico@example.com", "Rico"),
    ("sam@example.com", "Sam K"),
    ("priya@example.com", "Priya"),
    ("walt@example.com", "Walt"),
]
HOST = ("host@example.com", "Miles")


class Client:
    def __init__(self, email: str, name: str):
        self.email, self.name = email, name
        self.jar = http.cookiejar.CookieJar()
        self.op = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.jar)
        )
        self.req("GET", f"/auth/dev?email={email}&name={name.replace(' ', '%20')}")

    def req(self, method: str, path: str, body: dict | None = None):
        data = None if body is None else json.dumps(body).encode()
        r = urllib.request.Request(BASE + path, data=data, method=method)
        if data is not None:
            r.add_header("Content-Type", "application/json")
        try:
            with self.op.open(r, timeout=20) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            return {"_error": e.code, "_detail": e.read().decode(errors="replace")}
        except (urllib.error.URLError, OSError) as e:  # server restarting: try again later
            return {"_error": 0, "_detail": str(e)}
        try:
            return json.loads(raw)
        except Exception:
            return {"_raw": raw[:80].decode(errors="replace")}


def guests(n: int) -> list[Client]:
    return [Client(e, nm) for e, nm in GUESTS[:n]]


def setup(tid: str, n: int) -> None:
    gs = guests(n)
    host = Client(*HOST)
    users = host.req("GET", "/admin/api/users")
    rows = users.get("users", users) if isinstance(users, dict) else users
    by_email = {u["email"]: u["id"] for u in rows}
    for g in gs:
        out = host.req("POST", "/admin/api/games_access",
                       {"user_id": by_email[g.email], "action": "grant"})
        print("grant", g.email, out.get("ok"), out.get("_detail", ""))
    view = host.req("GET", f"/games/api/tables/{tid}")
    free = [s["seat"] for s in view["seats"] if s["empty"]]
    buyins = [4000, 6500, 2500, 9000, 4000]
    for g, seat, b in zip(gs, free, buyins):
        out = g.req("POST", f"/games/api/tables/{tid}/sit",
                    {"seat": seat, "buyin_cents": b})
        print("sit", g.name, seat, out.get("_detail", "ok"))


def play(tid: str, style: str, max_actions: int, with_host: bool = False) -> None:
    gs = guests(len(GUESTS)) + ([Client(*HOST)] if with_host else [])
    done = 0
    for _ in range(400):
        acted = False
        for g in gs:
            v = g.req("GET", f"/games/api/tables/{tid}")
            if "_error" in v or v.get("my_seat") is None:
                continue
            if v.get("actor") is None or v["actor"] != v["my_seat"]:
                continue
            legal = v.get("legal") or {}
            body = {"hand_no": v["hand_no"], "action_seq": v["action_seq"]}
            rb = v.get("raise_bounds") or {}
            if style in ("bet", "raise") and legal.get("raise") and done == 0:
                lo, hi = rb.get("min_chips", 0), rb.get("max_chips", 0)
                body.update(gate="raise", chips=min(hi, max(lo, (lo + hi) // 3)))
            elif style == "fold" and legal.get("fold"):
                body.update(gate="fold", chips=0)
            elif legal.get("check_call"):
                body.update(gate="check_call", chips=0)
            else:
                body.update(gate="fold", chips=0)
            out = g.req("POST", f"/games/api/tables/{tid}/act", body)
            print(g.name, body["gate"], body.get("chips"), out.get("_detail", "ok"))
            acted = True
            done += 1
            break
        hv = Client(*HOST).req("GET", f"/games/api/tables/{tid}") if not acted else None
        if done >= max_actions:
            return
        if not acted:
            if hv and (hv.get("actor") is None or hv.get("actor") == hv.get("my_seat")):
                print("stop: actor=", hv.get("actor"), "phase=", hv.get("phase"))
                return
            time.sleep(0.2)


def reload_all(tid: str) -> None:
    for g in guests(len(GUESTS)):
        v = g.req("GET", f"/games/api/tables/{tid}")
        if "_error" in v or v.get("my_seat") is None:
            continue
        me = v["seats"][v["my_seat"]]
        if me["stack_cents"] <= 2 * v["stakes"]["ante_cents"]:
            out = g.req("POST", f"/games/api/tables/{tid}/rebuy", {"amount_cents": 4000})
            print("reload", g.name, out.get("_detail", "ok"))


def loop(tid: str, minutes: float) -> None:
    """Keep the guests playing (loosely) so a human can sit at the table and play
    against them: mostly check/call, sometimes bet or fold, reload when busted."""
    import random

    gs = guests(len(GUESTS))
    end = time.time() + minutes * 60
    path = f"/games/api/tables/{tid}"
    while time.time() < end:
        for g in gs:
            v = g.req("GET", path)
            if "_error" in v or v.get("my_seat") is None or v.get("status") != "open":
                continue
            me = v["seats"][v["my_seat"]]
            if me["sitting_out"]:
                g.req("POST", path + "/sit_out", {"on": False})
            if not me["in_hand"] and me["stack_cents"] <= 2 * v["stakes"]["ante_cents"]:
                g.req("POST", path + "/rebuy", {"amount_cents": 4000})
            if v.get("actor") != v["my_seat"]:
                continue
            time.sleep(random.uniform(0.8, 2.2))  # "thinking"
            legal, rb = v["legal"], v["raise_bounds"]
            body = {"hand_no": v["hand_no"], "action_seq": v["action_seq"]}
            r = random.random()
            facing = v["to_call_cents"] > 0
            if legal["raise"] and random.random() < float(__import__("os").environ.get("BOT_SHOVE", "0.05")):  # a shove now and then: side pots
                body.update(gate="raise", chips=rb["max_chips"])
            elif legal["raise"] and r < (0.12 if facing else 0.3):
                lo, hi = rb["min_chips"], rb["max_chips"]
                body.update(gate="raise", chips=min(hi, max(lo, int(lo + (hi - lo) * random.choice([0, 0.25, 0.5])))))
            elif facing and legal["fold"] and r > 0.72:
                body.update(gate="fold", chips=0)
            else:
                body.update(gate="check_call", chips=0)
            g.req("POST", path + "/act", body)
            if random.random() < 0.06:
                g.req("POST", path + "/react", {"emote": random.choice(["gg", "nh", "lol", "wow", "fire"])})
        time.sleep(0.4)


def main() -> None:
    cmd, tid = sys.argv[1], sys.argv[2]
    if cmd == "loop":
        return loop(tid, float(sys.argv[3]) if len(sys.argv) > 3 else 30)
    if cmd == "reload":
        return reload_all(tid)
    if cmd == "setup":
        setup(tid, int(sys.argv[3]) if len(sys.argv) > 3 else 3)
    elif cmd == "play":
        play(tid, sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 99)
    elif cmd == "playall":  # the host seat is scripted too: runs to the end of the hand
        play(tid, sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 99, with_host=True)
    elif cmd == "chat":
        g = guests(1)[0]
        print(g.req("POST", f"/games/api/tables/{tid}/chat", {"text": sys.argv[3]}).get("_detail", "ok"))
    elif cmd == "view":
        v = Client(*HOST).req("GET", f"/games/api/tables/{tid}")
        keep = {k: v.get(k) for k in ("phase", "hand_no", "actor", "my_seat", "street",
                                      "legal", "raise_bounds", "to_call_cents", "running",
                                      "can_deal", "pot_cents")}
        print(json.dumps(keep, indent=1))


if __name__ == "__main__":
    main()
