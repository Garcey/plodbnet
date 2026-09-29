"""Test helpers for the home-games web client (static/games.*): a small CSS cascade
and a Node runner. Not a test module (no test_ prefix); the client tests import it.

`computed(rules, path, prop)` answers "what value does this property end up with on
this element" for the handful of layout-critical rules the tests pin — enough of CSS
for games.css: selector lists, descendant / child combinators, classes, ids,
attributes, :root / :not / :is / :where and state pseudo-classes, !important,
specificity, source order, one level of @media (only the conditions a test passes
are active; @supports always is) and inheritance. Sibling combinators and
pseudo-elements never match (nothing here needs them)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

STATIC = Path(__file__).resolve().parents[2] / "python" / "plo5bp" / "ui" / "static"

INHERITED = {"pointer-events", "color", "font-size", "font-weight", "visibility", "cursor", "line-height"}


# ------------------------------------------------------------------ elements
@dataclass
class El:
    tag: str = "div"
    id: str | None = None
    classes: frozenset = frozenset()
    attrs: dict = field(default_factory=dict)
    states: frozenset = frozenset()  # "root", "hover", "disabled", "first-child", ...


def el(spec: str, states=(), **attrs) -> El:
    """`el("button.seat-sit")`, `el("div#stage")`, `el("html", ("root",), data_anim="off")`."""
    m = re.match(r"^([a-z0-9]*)", spec)
    tag = m.group(1) or "div"
    ids = re.findall(r"#([\w-]+)", spec)
    classes = frozenset(re.findall(r"\.([\w-]+)", spec))
    return El(tag, ids[0] if ids else None, classes, {k.replace("_", "-"): v for k, v in attrs.items()}, frozenset(states))


def root(**attrs) -> El:
    return el("html", ("root",), **attrs)


# ------------------------------------------------------------------ parsing
@dataclass
class Rule:
    media: str | None
    selector: str
    decls: dict  # prop -> (value, important)
    order: int


def _split_top(text: str, sep: str) -> list[str]:
    out, depth, cur = [], 0, []
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return [x.strip() for x in out if x.strip()]


def parse_css(css: str) -> list[Rule]:
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    rules: list[Rule] = []

    def block(text: str, media: str | None) -> None:
        pos = 0
        while True:
            j = text.find("{", pos)
            if j < 0:
                return
            head = text[pos:j].strip()
            depth, k = 1, j + 1
            while depth and k < len(text):
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                k += 1
            body, pos = text[j + 1:k - 1], k
            if head.startswith("@media"):
                block(body, " ".join(head[len("@media"):].split()))
            elif head.startswith("@supports"):
                block(body, media)
            elif head.startswith("@"):
                continue  # @keyframes, @font-face ...
            else:
                decls = {}
                for d in _split_top(body, ";"):
                    if ":" not in d:
                        continue
                    p, v = d.split(":", 1)
                    v = v.strip()
                    imp = v.endswith("!important")
                    if imp:
                        v = v[: -len("!important")].strip()
                    decls[p.strip().lower()] = (v, imp)
                for sel in _split_top(head, ","):
                    rules.append(Rule(media, sel, decls, len(rules)))

    block(css, None)
    return rules


def keyframes(css: str) -> dict[str, str]:
    """name -> the @keyframes body."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    out = {}
    for m in re.finditer(r"@keyframes\s+([\w-]+)\s*\{", css):
        depth, k = 1, m.end()
        while depth and k < len(css):
            if css[k] == "{":
                depth += 1
            elif css[k] == "}":
                depth -= 1
            k += 1
        out[m.group(1)] = css[m.end():k - 1]
    return out


_TOKEN = re.compile(
    r"""(?P<star>\*)|(?P<tag>[a-zA-Z][\w-]*)|\#(?P<id>[\w-]+)|\.(?P<cls>[\w-]+)
    |\[(?P<attr>[\w-]+)(?:(?P<op>[~|^$*]?=)["']?(?P<val>[^"'\]]*)["']?)?\]
    |::(?P<pel>[\w-]+)|:(?P<pcl>[\w-]+)(?:\((?P<arg>(?:[^()]|\([^()]*\))*)\))?""",
    re.X,
)


def _compound(text: str):
    """[(kind, value, extra)] for one compound selector, or None if it can't be read."""
    out, pos = [], 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            return None
        pos = m.end()
        if m.group("star"):
            continue
        if m.group("tag"):
            out.append(("tag", m.group("tag").lower(), None))
        elif m.group("id"):
            out.append(("id", m.group("id"), None))
        elif m.group("cls"):
            out.append(("cls", m.group("cls"), None))
        elif m.group("attr"):
            out.append(("attr", m.group("attr"), (m.group("op"), m.group("val"))))
        elif m.group("pel"):
            out.append(("pel", m.group("pel"), None))
        else:
            name, arg = m.group("pcl"), m.group("arg")
            if name in ("not", "is", "where"):
                subs = [_complex(a) for a in _split_top(arg or "", ",")]
                if any(s is None for s in subs):
                    return None
                out.append((name, subs, None))
            elif name in ("before", "after"):
                out.append(("pel", name, None))
            else:
                out.append(("pcl", name, arg))
    return out


def _complex(sel: str):
    """[(combinator-to-the-left, compound)] or None (sibling combinators, unreadable)."""
    sel = re.sub(r"\s*>\s*", " > ", sel.strip())
    if re.search(r"[+~](?![^\[]*\])", sel):
        return None
    parts, comb = [], " "
    for tok in sel.split():
        if tok == ">":
            comb = ">"
            continue
        c = _compound(tok)
        if c is None:
            return None
        parts.append((comb, c))
        comb = " "
    return parts


def _spec_compound(c) -> tuple[int, int, int]:
    a = b = t = 0
    for kind, v, _ in c:
        if kind == "id":
            a += 1
        elif kind in ("cls", "attr", "pcl"):
            b += 1
        elif kind in ("tag", "pel"):
            t += 1
        elif kind in ("not", "is"):
            best = max((specificity(s) for s in v), default=(0, 0, 0))
            a, b, t = a + best[0], b + best[1], t + best[2]
    return a, b, t


def specificity(parts) -> tuple[int, int, int]:
    tot = [0, 0, 0]
    for _, c in parts:
        for i, x in enumerate(_spec_compound(c)):
            tot[i] += x
    return tuple(tot)


def _match_compound(c, e: El, path, idx) -> bool:
    for kind, v, extra in c:
        if kind == "tag" and e.tag != v:
            return False
        if kind == "id" and e.id != v:
            return False
        if kind == "cls" and v not in e.classes:
            return False
        if kind == "attr":
            op, val = extra
            have = e.attrs.get(v)
            if have is None:
                return False
            if op == "=" and have != val:
                return False
            if op == "~=" and val not in have.split():
                return False
            if op == "^=" and not have.startswith(val):
                return False
            if op == "*=" and val not in have:
                return False
        if kind == "pel":
            return False
        if kind == "pcl" and v not in e.states:
            return False
        if kind == "not" and any(_match(s, path[: idx + 1]) for s in v):
            return False
        if kind in ("is", "where") and not any(_match(s, path[: idx + 1]) for s in v):
            return False
    return True


def _match(parts, path) -> bool:
    def rec(pi: int, ei: int) -> bool:
        comb, comp = parts[pi]
        if not _match_compound(comp, path[ei], path, ei):
            return False
        if pi == 0:
            return True
        if parts[pi][0] == ">":
            return ei > 0 and rec(pi - 1, ei - 1)
        return any(rec(pi - 1, k) for k in range(ei - 1, -1, -1))

    return bool(parts) and rec(len(parts) - 1, len(path) - 1)


def computed(rules: list[Rule], path: list[El], prop: str, media=()) -> str | None:
    """The cascaded value of `prop` on path[-1] (path = root ... element); inherited
    properties fall back to the parent's value."""
    best = None
    for r in rules:
        if prop not in r.decls:
            continue
        if r.media is not None and r.media not in media:
            continue
        parts = _complex(r.selector)
        if parts is None or not _match(parts, path):
            continue
        val, imp = r.decls[prop]
        key = (imp, specificity(parts), r.order)
        if best is None or key > best[0]:
            best = (key, val)
    if best is not None:
        return best[1]
    if prop in INHERITED and len(path) > 1:
        return computed(rules, path[:-1], prop, media)
    return None


def matching(rules: list[Rule], path: list[El], media=()) -> list[Rule]:
    out = []
    for r in rules:
        if r.media is not None and r.media not in media:
            continue
        parts = _complex(r.selector)
        if parts is not None and _match(parts, path):
            out.append(r)
    return out


def games_css() -> str:
    return (STATIC / "games.css").read_text(encoding="utf-8")


# ------------------------------------------------------------------ node
def node_exe() -> str | None:
    return shutil.which("node")


def run_node(script: str, *args: str, tmp: Path, timeout: int = 60):
    """Run a Node script (written to tmp) and parse the JSON on its last stdout line."""
    f = tmp / "harness.js"
    f.write_text(script, encoding="utf-8")
    r = subprocess.run([node_exe(), str(f), *map(str, args)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    assert r.returncode == 0, r.stderr[-4000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


# ------------------------------------------------------------------ a tiny DOM for Node
# hg_mini_dom.js: enough of the DOM for the UI modules' logic — elements parsed from
# innerHTML, ids, classes, data-*, simple selectors (querySelector / closest), bubbling
# events (click() checks checkboxes and radios first), plus fake timers (`advance(ms)`
# runs what is due). A harness starts with MINI_DOM and calls makeWorld().
MINI_DOM = (Path(__file__).resolve().parent / "hg_mini_dom.js").read_text(encoding="utf-8")

# The whole UI in Node: the real games.js core, the felt module and every window module on
# the mini DOM, with games.html's body, a fetch stub (`replies(url, body)` answers; POSTs are
# recorded) and fake timers. `boot(replies)` returns {W, HG, $, posts, store, ctx};
# `table(over)` is a small table view; `flush()` settles pending promises. A test runs
# `UI_BOOT + body` with run_node(script, STATIC, json.dumps(UI_FILES), tmp=...).
UI_FILES = ["games.js", "games.table.js", "games.ui.js", "games.lobby.js", "games.history.js", "games.seat.js", "games.manage.js"]
UI_BOOT = MINI_DOM + r"""
const fs = require("fs"), vm = require("vm");
function boot(replies) {
  const W = makeWorld();
  const html = fs.readFileSync(process.argv[2] + "/games.html", "utf8");
  const bodyHtml = html.slice(html.indexOf("<body>") + 6, html.lastIndexOf("</body>")).replace(/<script[^>]*><\/script>/g, "");
  W.doc.body.innerHTML = bodyHtml;
  const posts = [];
  const store = {};
  const ctx = { console, JSON, Math, Number, String, Object, Array, Set, Map, Promise, Error, Date, Proxy, URL: { createObjectURL: () => "blob:x", revokeObjectURL() {} },
    document: W.doc, location: { pathname: "/games", origin: "http://x" }, history: { replaceState() {}, pushState() {} },
    setTimeout: W.setTimeout, clearTimeout: W.clearTimeout, setInterval: () => 1, clearInterval() {},
    requestAnimationFrame: (f) => W.setTimeout(f, 0), performance: { now: () => W.now() }, matchMedia: () => ({ matches: false }),
    getComputedStyle: () => ({ position: "static" }), innerWidth: 1280, innerHeight: 800, CustomEvent: class { constructor(t, o) { this.type = t; this.detail = o && o.detail; } },
    Event: class { constructor(t) { this.type = t; } },
    localStorage: { getItem: (k) => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = String(v); }, removeItem: (k) => { delete store[k]; } },
    fetch: async (url, o) => {
      const body = o && o.body ? JSON.parse(o.body) : null;
      if (o && o.method === "POST") posts.push({ url, body });
      const r = (replies && replies(url, body)) || {};
      return { ok: true, status: 200, headers: { get: () => "application/json" }, json: async () => r, text: async () => "{}" };
    } };
  ctx.globalThis = ctx;
  ctx.addEventListener = () => {};
  vm.createContext(ctx);
  for (const f of JSON.parse(process.argv[3])) {
    let src = fs.readFileSync(process.argv[2] + "/" + f, "utf8").replace(/\r\n/g, "\n");
    if (f === "games.js") src = src.replace(/\ninit\(\);\s*$/, "\n");
    vm.runInContext(src, ctx, { filename: f });
  }
  const HG = ctx.HG;
  HG.table.render = () => {}; HG.table.layout = () => {};
  HG.play = { render() {}, init() {}, onClock() {} };
  HG.fair = null;
  const $ = (id) => W.doc.getElementById(id);
  return { W, HG, $, posts, store, ctx };
}
const flush = async () => { for (let i = 0; i < 30; i++) await Promise.resolve(); };
function table(over) {
  const seat = (i, name, stack) => ({ seat: i, empty: false, name, user_id: i + 1, stack_cents: stack, in_hand: false, folded: false, sitting_out: false,
    pending_remove: false, auto_stack_cents: 0, trusted: false, present: true, is_host: i === 0, avatar: null, topup_target_cents: 0, topup_below_cents: 0 });
  return Object.assign({ id: "T1", epoch: "e", rev: 1, name: "Friday", status: "open", variant: "plo5", phase: "waiting", hand_no: 4, last_hand_no: 3,
    my_seat: 0, my_user_id: 1, is_host: true, is_member: true, running: true, num_seats: 6, actor: null, can_deal: false, eligible_count: 2,
    decision_secs: 30, street_pause_secs: 1, runout: { blocking: false }, requests: [], spectators: [], ledger: [], events: [], chat: [], history: [],
    stakes: { bb_cents: 100, bb_chips: 10000, ante_cents: 300, default_buyin_cents: 4000 },
    settings: { min_buyin_cents: 0, max_buyin_cents: 0, listed: true, show_grades: true, allow_rabbit: true, allow_rathole: false, approve_buyins: false, time_bank_secs: 30, deal_delay_secs: 5 },
    auto_topup: { mode: "off" }, auto_stack: { mode: "off" },
    seats: [seat(0, "Host", 4000), seat(1, "Dana", 5000), { seat: 2, empty: true }, { seat: 3, empty: true }, { seat: 4, empty: true }, { seat: 5, empty: true }] }, over || {});
}
"""
