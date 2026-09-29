"""Home-games client: markup is safe by construction, and styling lives in the CSS
(2026-09-28, FE-003 / FE-011).

- Every piece of markup in the client is written in an html`` template, which escapes
  every value it is given (a name, a chat line, a club name, a code the server sent);
  put() / h() take markup as markup and anything else as TEXT. A static scan of every
  client file holds that line: no quoted string or untagged template builds markup, and
  innerHTML only ever receives "" or an html`` template.
- No inline style anywhere (the page's CSP no longer allows 'unsafe-inline' styles): a
  value the CSS needs rides in data-vars and is set through the CSSOM.
- In Node, with the real modules on the mini DOM: html`` escapes / nests / takes lists,
  `markup + text` fails loudly, and a player named like a tag, a chat line with markup
  and a server value in a class all arrive as text.
Skips its Node half when Node is not installed."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hg_client_tools import MINI_DOM, STATIC, UI_BOOT, UI_FILES, node_exe, run_node  # noqa: E402

CLIENT = sorted(p for p in STATIC.glob("games*.js"))


# ------------------------------------------------------------------ a JS scanner
_KEYWORDS_BEFORE_REGEX = {"return", "typeof", "case", "do", "else", "in", "of", "new", "delete", "void",
                          "throw", "instanceof", "yield", "await"}


def scan_js(src: str):
    """Every string and template literal in `src`, comments and regex literals skipped.
    Returns (literals, mask): each literal is {kind: "str"|"tpl", text: its static text
    (a template's parts outside ${}), tag: the expression right before a template's
    backtick ("html", "HG.core.html", ""), start, end}; `mask` is `src` with comments and
    literal bodies blanked (code only, same offsets)."""
    lits: list[dict] = []
    mask = list(src)
    n = len(src)

    def blank(a: int, b: int) -> None:
        for k in range(a, b):
            if mask[k] != "\n":
                mask[k] = " "

    def prev_significant(i: int) -> str:
        j = i - 1
        while j >= 0 and mask[j].isspace():
            j -= 1
        if j < 0:
            return ""
        if mask[j].isalnum() or mask[j] in "_$":
            k = j
            while k >= 0 and (mask[k].isalnum() or mask[k] in "_$"):
                k -= 1
            return src[k + 1 : j + 1]
        return mask[j]

    def regex_allowed(i: int) -> bool:
        p = prev_significant(i)
        if p == "":
            return True
        if p[0].isalnum() or p[0] in "_$":
            return p in _KEYWORDS_BEFORE_REGEX
        return p in "(,=:[!&|?{};+-*%<>~^"

    def tag_before(i: int) -> str:
        j = i - 1
        k = j
        while k >= 0 and (src[k].isalnum() or src[k] in "_$."):
            k -= 1
        return src[k + 1 : j + 1]

    def code(i: int, stop_at_brace: bool) -> int:
        depth = 0
        while i < n:
            c = src[i]
            if c in "\"'":
                j = i + 1
                while j < n and src[j] != c:
                    j += 2 if src[j] == "\\" else 1
                lits.append({"kind": "str", "text": src[i + 1 : j], "tag": "", "start": i, "end": j + 1})
                blank(i + 1, j)
                i = j + 1
            elif c == "`":
                i = template(i)
            elif c == "/" and i + 1 < n and src[i + 1] == "/":
                j = src.find("\n", i)
                j = n if j < 0 else j
                blank(i, j)
                i = j
            elif c == "/" and i + 1 < n and src[i + 1] == "*":
                j = src.find("*/", i + 2)
                j = n if j < 0 else j + 2
                blank(i, j)
                i = j
            elif c == "/" and regex_allowed(i):
                j, in_class = i + 1, False
                while j < n and (in_class or src[j] != "/"):
                    if src[j] == "\\":
                        j += 1
                    elif src[j] == "[":
                        in_class = True
                    elif src[j] == "]":
                        in_class = False
                    j += 1
                blank(i + 1, j)
                i = j + 1
            elif c == "{":
                depth += 1
                i += 1
            elif c == "}":
                if stop_at_brace and depth == 0:
                    return i
                depth -= 1
                i += 1
            else:
                i += 1
        return i

    def template(i: int) -> int:
        start, tag = i, tag_before(i)
        parts, j, chunk = [], i + 1, i + 1
        while j < n and src[j] != "`":
            if src[j] == "\\":
                j += 2
            elif src[j] == "$" and j + 1 < n and src[j + 1] == "{":
                parts.append(src[chunk:j])
                blank(chunk, j)
                j = code(j + 2, True) + 1
                chunk = j
            else:
                j += 1
        parts.append(src[chunk:j])
        blank(chunk, j)
        lits.append({"kind": "tpl", "text": "".join(parts), "tag": tag, "start": start, "end": j + 1})
        return j + 1

    code(0, False)
    return lits, "".join(mask)


MARKUP = re.compile(r"<[a-zA-Z!/]")


def _line(src: str, pos: int) -> int:
    return src.count("\n", 0, pos) + 1


def test_the_scanner_itself():
    src = ('const a = "x<b>"; // a comment with `<i>`\n'
           'const b = html`<p>${name}</p>`, c = /<[a-z]+>/g.test(s) ? 1 : 2;\n'
           "const d = `plain ${`<em>${x}</em>`}`; x = y / 2 / z;")
    lits, mask = scan_js(src)
    got = [(x["kind"], x["text"], x["tag"]) for x in lits]
    assert ("str", "x<b>", "") in got
    assert ("tpl", "<p></p>", "html") in got
    assert ("tpl", "<em></em>", "") in got  # (the nested one: untagged — flagged below)
    assert "/<[a-z]+>/" not in mask and "a comment" not in mask and "y / 2 / z" in mask


def test_markup_is_only_ever_written_in_html_templates():
    bad = []
    for f in CLIENT:
        src = f.read_text(encoding="utf-8")
        lits, mask = scan_js(src)
        for x in lits:
            if not MARKUP.search(x["text"]):
                continue
            if x["kind"] == "tpl" and re.search(r"(^|\.)html$", x["tag"]):
                continue
            bad.append(f"{f.name}:{_line(src, x['start'])}: {src[x['start']:x['start'] + 70]!r}")
    assert not bad, "markup outside an html`` template (FE-003):\n" + "\n".join(bad)


def test_innerhtml_only_ever_gets_nothing_or_an_html_template():
    bad = []
    for f in CLIENT:
        src = f.read_text(encoding="utf-8")
        _, mask = scan_js(src)
        for m in re.finditer(r"\.innerHTML\s*=(?!=)", mask):
            rest = src[m.end():m.end() + 40].lstrip()
            if re.match(r'(""|\'\'|(?:[\w$]+\.)*html`)', rest):
                continue
            if f.name == "games.js" and rest.startswith("content.s;"):
                continue  # (the one sink: put() with a SafeHTML — games.js)
            bad.append(f"{f.name}:{_line(src, m.start())}: {src[m.start():m.end() + 50]!r}")
        for m in re.finditer(r"insertAdjacentHTML|outerHTML\s*=(?!=)|document\.write|setAttribute\(\s*[\"']style", mask):
            bad.append(f"{f.name}:{_line(src, m.start())}: {m.group(0)}")
    assert not bad, "markup must go in through put() / h() or an html`` template:\n" + "\n".join(bad)


def test_no_inline_styles_and_the_csp_allows_none():
    """FE-011: static styling is CSS classes; a value (a hue, a position, a width) rides
    in data-vars and is set through the CSSOM — so the page's CSP drops 'unsafe-inline'."""
    bad = []
    for f in CLIENT:
        src = f.read_text(encoding="utf-8")
        lits, mask = scan_js(src)
        bad += [f"{f.name}:{_line(src, x['start'])}: style= in markup" for x in lits if re.search(r"\sstyle\s*=", x["text"])]
        bad += [f"{f.name}:{_line(src, m.start())}: a style attribute object" for m in re.finditer(r"[{,]\s*style\s*:", mask)]
    page = (STATIC / "games.html").read_text(encoding="utf-8")
    assert not re.search(r"\sstyle\s*=|<style", page), "games.html carries an inline style"
    assert not bad, "\n".join(bad)
    from plo5bp.ui import homegame as hg
    directives = dict(d.strip().split(" ", 1) for d in hg.PAGE_CSP.split(";"))
    assert "unsafe-inline" not in directives["style-src"]
    assert directives["style-src"].split()[0] == "'self'"


# ------------------------------------------------------------------ in Node
@pytest.fixture(scope="module")
def node():
    if node_exe() is None:
        pytest.skip("node is not installed")
    return node_exe()


CORE = MINI_DOM + r"""
const fs = require("fs"), vm = require("vm");
const W = makeWorld();
const ctx = { console, document: W.doc, setTimeout: W.setTimeout, clearTimeout: W.clearTimeout, location: { pathname: "/games" }, history: {} };
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[2], "utf8").replace(/\r\n/g, "\n").replace(/\ninit\(\);\s*$/, "\n"), ctx);
const { html, put, raw, isHTML } = ctx.HG.core;
"""


def test_html_escapes_every_value_and_composes(node, tmp_path):
    got = run_node(CORE + r"""
const out = {};
const name = `<img src=x onerror="alert(1)">&'`;
out.text = String(html`<b title="${name}">${name}</b>`);
out.nested = String(html`<ul>${["a<", "b"].map((x) => html`<li>${x}</li>`)}</ul>`);
out.empty = String(html`[${null}${undefined}${0}${false}${""}]`);
out.raw = String(html`${raw("<i>ok</i>")}`);
out.safe = isHTML(html`x`) && !isHTML("x");
try { const joined = html`<b>a</b>` + "c"; out.plus = "allowed: " + joined; } catch (e) { out.plus = e.constructor.name; }
out.stringified = `${html`<b>ok</b>`}` === "<b>ok</b>";  // (a template literal is a string, not an addition)
const box = W.doc.createElement("div");
put(box, "<b>text</b>"); out.putText = [box.childElementCount, box.textContent];
put(box, html`<b>${"x"}</b>`); out.putMarkup = [box.childElementCount, box.firstElementChild.tagName];
const node = W.doc.createElement("i"); put(box, node); out.putNode = box.firstElementChild === node;
put(box, html`<span class="av" data-vars="h:212;x:40.5%"><em data-vars="pct:7%"></em></span>`);
out.vars = [box.firstElementChild.style["--h"], box.firstElementChild.style["--x"], box.querySelector("em").style["--pct"]];
console.log(JSON.stringify(out));
""", STATIC / "games.js", tmp=tmp_path)
    assert got["text"] == ('<b title="&lt;img src=x onerror=&quot;alert(1)&quot;&gt;&amp;&#39;">'
                           "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;&amp;&#39;</b>")
    assert got["nested"] == "<ul><li>a&lt;</li><li>b</li></ul>"
    assert got["empty"] == "[0false]"
    assert got["raw"] == "<i>ok</i>"
    assert got["safe"] is True
    assert got["plus"] == "TypeError"  # (markup + text would quietly become escaped text)
    assert got["stringified"] is True
    assert got["putText"] == [0, "<b>text</b>"]
    assert got["putMarkup"] == [1, "B"]
    assert got["putNode"] is True
    assert got["vars"] == ["212", "40.5%", "7%"]


def test_names_chat_and_server_values_arrive_as_text(node, tmp_path):
    """A player named like a tag, a chat line with markup, a club name and a member role
    (a server value) in a class: text in the page, never elements."""
    evil = '<img src=x onerror="alert(1)">'
    got = run_node(UI_BOOT + r"""
(async () => {
  const evil = process.argv[4];
  const B = boot((url) => url.startsWith("/games/api/clubs/") ? { id: "c1", name: evil, role: "owner", invite_code: "abc", approve_joins: false,
    members: [{ user_id: 1, name: evil, role: evil, is_me: false, avatar: null }], requests: [] } : {});
  const s = table({ name: evil, chat: [{ id: 1, user_id: 2, name: evil, text: evil + " <b>hi</b>", created_at: "2026-09-28T12:00:00Z" }],
    events: [{ id: 1, ts: 1, kind: "win", text: evil + " wins" }],
    ledger: [{ user_id: 2, name: evil, buyin_cents: 100, stack_cents: 50, leftover_cents: 0, net_cents: -50, seated: true }] });
  s.seats[1].name = evil;
  B.HG.core.G.state = s; B.HG.core.G.gameId = "T1"; B.HG.core.G.prefs.rail = true;
  B.HG.ui.render(s, null);
  B.HG.ui.setRail(true, "ledger");
  const out = {};
  const imgs = () => B.W.doc.querySelectorAll("img").filter((i) => i.getAttribute("src") === "x").length;
  out.chat = B.$("chat-log").textContent.includes(evil + " <b>hi</b>");
  out.ledger = B.$("ledger-body").textContent.includes(evil);
  out.title = B.$("table-title").textContent === evil;
  B.HG.ui.openPlayer(1); await flush();
  out.card = B.W.doc.querySelector(".pcard-head b").textContent.startsWith(evil);
  B.HG.ui.closeTop(); B.W.advance(500);
  B.HG.core.G.clubId = "c1";
  B.HG.ui.openClubSettings(); await flush(); B.W.advance(100); await flush();
  const pill = B.W.doc.querySelector(".mem .pill");
  out.role = pill ? [pill.textContent, pill.childElementCount, pill.getAttribute("onerror")] : null;
  out.imgs = imgs();
  console.log(JSON.stringify(out));
})();
""", STATIC, json.dumps(UI_FILES), evil, tmp=tmp_path)
    assert got["chat"] and got["ledger"] and got["title"] and got["card"]
    assert got["role"] == [evil, 0, None]  # (the role's text; in its class it stays inside the attribute)
    assert got["imgs"] == 0  # nothing a name or a line said ever became an element
