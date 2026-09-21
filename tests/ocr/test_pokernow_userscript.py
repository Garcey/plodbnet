"""Behavioural tests for the PokerNow Tampermonkey collector (review 2026-09-20).

`tools/pokernow/pokernow.user.js` is the only producer of the live PokerNow
stream, and it had no tests at all. The reconstructor rebuilds a hand from the
ORDERED stream of distinct frames, so a delivery bug in the browser is a lost
action on the server. `pokernow_userscript_harness.js` runs the real script in a
node `vm` sandbox (fake DOM / timers / MutationObserver / GM_xmlhttpRequest) and
checks:

- `onerror` / `ontimeout` requeue the in-flight frame (front of the queue, order
  preserved) and retry after a backoff — they used to drop it and never drain;
- the "nothing changed" early return no longer skips `drain()`;
- an HTTP error status drops only that frame and keeps draining;
- the heartbeat sends a FRESH snapshot, re-attaches an observer whose `.table`
  root was replaced, and sends nothing when no table is on screen — it used to
  re-send a remembered payload forever;
- `allIn` comes from the DOM ("All In" label / class), never from a missing
  stack number;
- the payload still has the `pokernow.v1` shape `map_payload` validates, over
  `GM_xmlhttpRequest` to 127.0.0.1:8765.

Skipped when node is not installed (it is a dev-only dependency).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
USERSCRIPT = REPO_ROOT / "tools" / "pokernow" / "pokernow.user.js"
HARNESS = Path(__file__).with_name("pokernow_userscript_harness.js")

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


@needs_node
def test_userscript_is_valid_javascript():
    proc = subprocess.run(
        [NODE, "--check", str(USERSCRIPT)], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr


@needs_node
def test_userscript_delivery_and_all_in_scenarios():
    proc = subprocess.run(
        [NODE, str(HARNESS), str(USERSCRIPT)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert proc.stdout.strip().startswith("OK "), proc.stdout


def test_userscript_keeps_privileged_transport_and_loopback_target():
    """Static pins that need no node: the transport must stay
    `GM_xmlhttpRequest` (page-context fetch/WebSocket cannot reach 127.0.0.1 from
    an https tab) and the target must stay the local server."""
    src = USERSCRIPT.read_text(encoding="utf-8")
    assert "// @grant        GM_xmlhttpRequest" in src
    assert "// @connect      127.0.0.1" in src
    assert "const INGEST_URL = 'http://127.0.0.1:8765/pokernow/ingest';" in src
    assert "GM_xmlhttpRequest({" in src
    # No page-context transports, and nothing that writes to the page.
    body = src.split("==/UserScript==", 1)[1]
    code = "\n".join(
        line for line in body.splitlines() if not line.strip().startswith(("//", "*", "/*"))
    )
    for forbidden in ("fetch(", "new WebSocket", "XMLHttpRequest(", ".click(", "dispatchEvent("):
        assert forbidden not in code, forbidden
