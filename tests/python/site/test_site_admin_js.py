"""admin.js under Node (site ACC-029 / ACC-031 / ACC-030).

- an earlier error is cleared once a refresh succeeds;
- admin rows offer no "Grant comp";
- free mode is shown as a Mode card with the paywall numbers greyed out;
- the users table can be searched and sorted; online players are listed.
Skipped without Node.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ADMIN_JS = Path(__file__).resolve().parents[3] / "python" / "plo5bp" / "ui" / "static" / "admin.js"

HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
const els = new Map();
class El {
  constructor(id) { this.id = id; this.innerHTML = ""; this.textContent = ""; this.style = {}; this.value = ""; this.disabled = false; this.listeners = {}; }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
}
const $ = (id) => { if (!els.has(id)) els.set(id, new El(id)); return els.get(id); };
const fixtures = {
  "/admin/api/metrics": { users: 3, active_stripe_subs: 0, active_comp_subs: 1, mrr_cents: 0, mrr_is_estimate: true,
    revenue_cents_total: 0, price_cents: 1000, free_hands_per_day: 5, free_for_all: true,
    hands_by_day: [{ day: "2026-09-28", n: 4 }], recent_payments: [] },
  "/admin/api/users": { users: [
    { id: 1, email: "boss@x.com", name: "Boss", is_admin: true, sub_status: "none", created_at: "2026-09-01", last_login_at: "2026-09-28", hands_today: 0, hands_total: 9, active_now: true },
    { id: 2, email: "amy@x.com", name: "Amy", is_admin: false, sub_status: "none", created_at: "2026-09-10", last_login_at: "2026-09-20", hands_today: 1, hands_total: 30, active_now: false },
    { id: 3, email: "zed@x.com", name: "Zed", is_admin: false, sub_status: "active", sub_source: "comp", created_at: "2026-09-20", last_login_at: "2026-09-27", hands_today: 0, hands_total: 2, active_now: true },
  ] },
  "/admin/api/system": { health: { status: "ok", problems: [] }, formats: [], http: { routes: [], recent_errors: [] }, config: {}, db: {} },
  "/admin/api/audit?limit=40": { entries: [] },
};
let failNext = false;
const fetchStub = async (url) => {
  if (failNext) { failNext = false; return { ok: false, status: 500, json: async () => ({ detail: "boom" }), text: async () => "boom" }; }
  const body = fixtures[url];
  if (!body) return { ok: false, status: 404, json: async () => ({ detail: "nope" }), text: async () => "" };
  return { ok: true, status: 200, json: async () => body };
};
const ctx = vm.createContext({
  document: { getElementById: $, addEventListener() {} }, fetch: fetchStub, console, window: {},
  JSON, Math, Number, String, Object, Array, Promise, Error, parseInt, Date, setTimeout,
});
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), ctx, { filename: "admin.js" });
const flush = () => new Promise((r) => setTimeout(r, 20));
(async () => {
  const out = {};
  await flush(); await flush();
  out.metrics = $("metric-cards").innerHTML;
  out.rows = $("user-rows").innerHTML;
  out.active = $("active-list").innerHTML;
  out.count = $("user-count").textContent;
  // an error, then a successful refresh clears it
  $("err").textContent = "old failure"; $("err").style.display = "block";
  await vm.runInContext("refreshAll()", ctx);
  out.errAfter = [$("err").textContent, $("err").style.display];
  // search + sort
  $("user-search").listeners.input[0]({ target: { value: "amy" } });
  out.searched = $("user-rows").innerHTML;
  out.searchedCount = $("user-count").textContent;
  $("user-search").listeners.input[0]({ target: { value: "" } });
  $("user-sort").listeners.change[0]({ target: { value: "hands" } });
  out.sortedFirst = $("user-rows").innerHTML.indexOf("amy@x.com") < $("user-rows").innerHTML.indexOf("boss@x.com");
  process.stdout.write(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
"""


@pytest.fixture(scope="module")
def js():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    proc = subprocess.run([node, "-", str(ADMIN_JS)], input=HARNESS, capture_output=True,
                          text=True, encoding="utf-8", timeout=60)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_free_mode_card_and_dormant_paywall_numbers(js):
    assert "Free for all" in js["metrics"]
    assert 'class="card off"><div class="k">Price' in js["metrics"]
    assert 'class="card off"><div class="k">Free hands/day' in js["metrics"]


def test_admin_rows_offer_no_comp(js):
    boss_row = js["rows"].split("boss@x.com", 1)[1].split("</tr>", 1)[0]
    assert "Grant comp" not in boss_row
    assert "Grant comp" in js["rows"]          # a normal free user still gets it


def test_errors_clear_after_a_good_refresh(js):
    assert js["errAfter"] == ["", "none"]


def test_search_sort_and_online_list(js):
    assert "Boss" in js["active"] and "Zed" in js["active"] and "Amy" not in js["active"]
    assert js["count"] == "3"
    assert "amy@x.com" in js["searched"] and "boss@x.com" not in js["searched"]
    assert js["searchedCount"] == "1 of 3"
    assert js["sortedFirst"] is True
