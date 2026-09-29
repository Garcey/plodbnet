// Admin dashboard script. Served ONLY under the admin-gated /admin prefix
// (`/admin/admin.js`); the page's Content-Security-Policy allows scripts from
// this origin and never inline ones, so none may be added to admin.html.
const $ = (id) => document.getElementById(id);
const fmtUsd = (c) => `$${(c / 100).toFixed(2)}`;
const fmtDay = (iso) => (iso ? iso.slice(0, 10) : "—");
const fmtTime = (iso) => (iso ? String(iso).replace("T", " ").slice(0, 16) : "—");
// Escape every DB/user-derived string before it goes into innerHTML.
// `name`/`email` originate from the Google profile (fully user-controlled)
// and `ref`/`sub_source` from Stripe/DB — none may inject markup here or a
// crafted display name becomes stored XSS in the admin's own session.
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );

async function j(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let msg = "";
    try {
      const body = await res.json();
      msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (_) {
      msg = (await res.text()).slice(0, 200);
    }
    throw new Error(`${res.status}: ${msg}`);
  }
  return res.json();
}

const post = (url, body) =>
  j(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });

function showErr(e) {
  $("err").textContent = e.message || String(e);
  $("err").style.display = "block";
}
// (site ACC-029) an old error must not linger once things work again.
function clearErr() {
  $("err").textContent = "";
  $("err").style.display = "none";
}

function metricCard(k, v, sub, cls) {
  return `<div class="card${cls ? ` ${cls}` : ""}"><div class="k">${k}</div><div class="v num">${v}${sub ? ` <small>${sub}</small>` : ""}</div></div>`;
}

function fmtUptime(s) {
  s = Math.max(0, Math.floor(Number(s) || 0));
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
  return d ? `${d}d ${h}h` : h ? `${h}h ${m}m` : `${m}m`;
}

// --- System (FEAT-024) -----------------------------------------------------------

function statusPill(status) {
  const cls = status === "ok" ? "active-comp" : status === "degraded" ? "admin" : "bad";
  return `<span class="pill ${cls}">${esc(String(status || "?").toUpperCase())}</span>`;
}

async function loadSystem() {
  const s = await j("/admin/api/system");
  const h = s.health || {};
  const build = h.build || {};
  $("sys-cards").innerHTML =
    metricCard("Health", statusPill(h.status)) +
    metricCard("Build", esc(build.commit_short || "unknown"), build.built_at ? esc(fmtDay(build.built_at)) : "") +
    metricCard("Up", fmtUptime(h.uptime_s)) +
    metricCard("Active now", s.active_users ?? 0) +
    metricCard("Sessions in memory", `${s.runtimes?.in_memory ?? 0}`, `/ ${s.runtimes?.capacity ?? "?"}`) +
    metricCard("Model work", `${s.work_gate?.busy ?? 0}`, `busy / ${s.work_gate?.slots ?? "?"} · ${s.work_gate?.waiting ?? 0} waiting`) +
    metricCard("Requests", s.http?.requests ?? 0);
  $("sys-problems").innerHTML = (h.problems || []).length
    ? `<ul>${h.problems.map((p) => `<li>${esc(p)}</li>`).join("")}</ul>`
    : '<span class="muted">No problems reported.</span>';

  $("model-rows").innerHTML = (s.formats || []).map((f) => {
    const avail = f.available !== false;
    const sha = f.sha256 ? f.sha256.slice(0, 12) : "—";
    const actions = avail || f.format === "experimental"
      ? ["reload", "promote", "rollback"].map((a) =>
          `<button data-model="${esc(f.format)}" data-maction="${a}">${a === "promote" ? "Promote .new" : a === "rollback" ? "Roll back" : "Reload"}</button>`
        ).join(" ")
      : "";
    return `<tr>
      <td>${esc(f.label || f.format)}<div class="muted small">${esc(f.format)}</div></td>
      <td>${esc(f.checkpoint || "— (placeholder)")}<div class="muted small num">${esc(sha)}</div></td>
      <td>${f.model_loaded ? '<span class="pill active-comp">LOADED</span>' : '<span class="pill bad">PLACEHOLDER</span>'}</td>
      <td>${f.critic_loaded ? "yes" : '<span class="muted">no</span>'}</td>
      <td class="num">${esc(f.obs_rev)}${f.obs_rev_mismatch ? ' <span class="pill bad">MISMATCH</span>' : ""}</td>
      <td class="num">v${esc(f.version)}<div class="muted small">${f.loaded_at ? esc(new Date(f.loaded_at * 1000).toISOString().slice(0, 16).replace("T", " ")) : ""}</div></td>
      <td>${actions}</td>
    </tr>`;
  }).join("");

  const routes = (s.http?.routes || []).slice(0, 12);
  $("route-rows").innerHTML = routes.length
    ? routes.map((r) => `<tr>
        <td class="num">${esc(r.route)}</td><td class="num">${r.count}</td>
        <td class="num">${r.p50_ms}</td><td class="num">${r.p95_ms}</td><td class="num">${r.max_ms}</td>
        <td class="num">${r.slow}</td><td class="num">${r.errors ? `<b class="bad-text">${r.errors}</b>` : 0}</td>
      </tr>`).join("")
    : '<tr><td colspan="7" class="muted">No requests yet.</td></tr>';

  const errs = s.http?.recent_errors || [];
  $("error-rows").innerHTML = errs.length
    ? errs.map((e) => `<tr>
        <td class="num">${esc(fmtTime(e.at))}</td><td class="num">${esc(e.id)}</td>
        <td class="num">${esc(e.method)} ${esc(e.path)}</td><td class="num">${esc(e.status)}</td>
        <td class="num">${esc(e.user ?? "—")}</td><td class="muted">${esc(e.error)}</td>
      </tr>`).join("")
    : '<tr><td colspan="6" class="muted">No errors since the last restart.</td></tr>';

  const cfg = s.config || {};
  $("sys-config").innerHTML = [
    ["Free for everyone", cfg.free_for_all ? "yes" : "no"],
    ["Google sign-in", cfg.google_sign_in ? "on" : "off"],
    ["Email sign-in", cfg.email_sign_in ? "on" : "off"],
    ["Stripe", cfg.stripe],
    ["Session key", cfg.session_key],
    ["Admin pinned to Google id", cfg.admin_pinned_to_google_id ? "yes" : "no"],
    ["Your Google id", s.you?.google_id || "—"],
    ["Database", `${s.db?.path} · ${((s.db?.bytes || 0) / 1e6).toFixed(1)} MB · schema ${JSON.stringify(s.db?.schema || {})}`],
    ["Device", `${s.device} · ${s.torch_threads} thread(s)`],
  // every value is plain text, escaped exactly once here
  ].map(([k, v]) => `<tr><th>${esc(k)}</th><td class="num">${v === undefined ? "—" : esc(v)}</td></tr>`).join("");

  const m = s.maintenance;
  $("maint-current").innerHTML = m
    ? `<b>Showing:</b> ${esc(m.message)}${m.at ? ` <span class="muted">(at ${esc(fmtTime(m.at))} UTC)</span>` : ""}`
    : '<span class="muted">No notice is showing.</span>';
  $("maint-clear").disabled = !m;
}

async function modelAction(fmt, action) {
  const verb = { reload: "Reload", promote: "Promote the .new file for", rollback: "Roll back" }[action];
  if (!window.confirm(`${verb} ${fmt}? It is loaded and test-run first; players keep their spots and hands.`)) return;
  const r = await post("/admin/api/models", { action, format: fmt });
  window.alert(`${fmt} now serves ${r.checkpoint || "?"} (sha256 ${String(r.sha256 || "").slice(0, 12)}, v${r.version}).`);
}

// --- Audit (SEC-022) ----------------------------------------------------------------

async function loadAudit() {
  const a = await j("/admin/api/audit?limit=40");
  $("audit-rows").innerHTML = a.entries.length
    ? a.entries.map((e) => `<tr>
        <td class="num">${esc(fmtTime(e.at))}</td><td>${esc(e.admin || "—")}</td>
        <td class="num">${esc(e.action)}</td><td>${esc(e.target || "")}</td>
        <td class="muted">${esc(Object.keys(e.detail || {}).length ? JSON.stringify(e.detail) : "")}</td>
      </tr>`).join("")
    : '<tr><td colspan="5" class="muted">No admin actions recorded yet.</td></tr>';
}

// --- Users + revenue ------------------------------------------------------------------

// Users table: search + sort (site ACC-031) over the last loaded list.
const USERS = { list: [], q: "", sort: "joined" };
const USER_SORTS = {
  joined: (a, b) => String(b.created_at || "").localeCompare(String(a.created_at || "")),
  login: (a, b) => String(b.last_login_at || "").localeCompare(String(a.last_login_at || "")),
  hands: (a, b) => (b.hands_total || 0) - (a.hands_total || 0),
  online: (a, b) => (b.active_now ? 1 : 0) - (a.active_now ? 1 : 0)
    || String(b.last_login_at || "").localeCompare(String(a.last_login_at || "")),
};

async function load() {
  const [m, u] = await Promise.all([j("/admin/api/metrics"), j("/admin/api/users")]);
  // (site ACC-031) In free mode the paywall numbers describe rules that aren't
  // in force: say so, and grey them out.
  const free = m.free_for_all === true;
  const pay = free ? "off" : "";
  $("metric-cards").innerHTML =
    metricCard("Mode", free ? "Free for all" : "Paywall on", free ? "paywall numbers below are dormant" : "", free ? "mode-free" : "") +
    metricCard("Users", m.users) +
    metricCard("Paying subs", m.active_stripe_subs, "", pay) +
    metricCard("Comp subs", m.active_comp_subs, "", pay) +
    metricCard(m.mrr_is_estimate ? "MRR (est.)" : "MRR", fmtUsd(m.mrr_cents), "", pay) +
    metricCard("Revenue (recorded)", fmtUsd(m.revenue_cents_total)) +
    metricCard("Price", fmtUsd(m.price_cents), "/mo", pay) +
    metricCard("Free hands/day", m.free_hands_per_day, "", pay) +
    metricCard("Hands (30d)", m.hands_by_day.reduce((a, r) => a + r.n, 0));

  USERS.list = u.users || [];
  const online = USERS.list.filter((x) => x.active_now);
  $("active-list").innerHTML = online.length
    ? `<b>${online.length} online now:</b> ${online.map((x) => esc(x.name || x.email)).join(", ")}`
    : '<span class="muted">Nobody is online right now.</span>';
  renderUsers();

  $("payment-rows").innerHTML = m.recent_payments.length
    ? m.recent_payments.map((p) => `<tr>
        <td class="num">${esc(p.created_at).replace("T", " ").slice(0, 16)}</td>
        <td>${esc(p.email)}</td>
        <td class="num">${fmtUsd(p.amount_cents)}</td>
        <td class="muted">${esc(p.ref)}</td>
      </tr>`).join("")
    : '<tr><td colspan="4" class="muted">No payments recorded yet.</td></tr>';
}

function renderUsers() {
  const q = USERS.q.trim().toLowerCase();
  const rows = USERS.list
    .filter((x) => !q || String(x.email || "").toLowerCase().includes(q) || String(x.name || "").toLowerCase().includes(q))
    .sort(USER_SORTS[USERS.sort] || USER_SORTS.joined);
  $("user-count").textContent = q ? `${rows.length} of ${USERS.list.length}` : `${USERS.list.length}`;
  $("user-rows").innerHTML = rows.length ? rows.map((x) => {
    const sub = x.is_admin
      ? '<span class="pill admin">ADMIN</span>'
      : x.sub_status === "active"
        ? `<span class="pill active-${esc(x.sub_source)}">${esc(x.sub_source).toUpperCase()}</span>` +
          (x.period_end ? ` <span class="muted">→ ${fmtDay(x.period_end)}</span>` : "")
        : '<span class="pill none">FREE</span>';
    // (site ACC-029) an admin already has everything: no "Grant comp".
    const btn = x.is_admin
      ? '<span class="muted">—</span>'
      : x.sub_status === "active" && x.sub_source === "comp"
        ? `<button class="revoke" data-uid="${x.id}" data-action="revoke">Revoke comp</button>`
        : x.sub_status !== "active"
          ? `<button class="grant" data-uid="${x.id}" data-action="grant">Grant comp</button>`
          : '<span class="muted">via Stripe</span>';
    const games = x.is_admin
      ? '<span class="pill admin">ADMIN</span>'
      : x.homegame_access
        ? `<span class="pill active-comp">MEMBER</span> <button class="revoke" data-uid="${x.id}" data-gaction="revoke">Remove</button>`
        : `<span class="pill none">—</span> <button class="grant" data-uid="${x.id}" data-gaction="grant">Add to club</button>`;
    const acct = x.is_admin
      ? ""
      : [
          x.disabled
            ? `<button class="grant" data-uid="${x.id}" data-uaction="enable">Enable</button>`
            : `<button class="revoke" data-uid="${x.id}" data-uaction="disable">Disable</button>`,
          `<button data-uid="${x.id}" data-uaction="signout" title="End every session of this account">Sign out everywhere</button>`,
          `<a class="btn-link" href="/admin/api/users/${x.id}/export" download>Export</a>`,
          `<button class="revoke" data-uid="${x.id}" data-email="${esc(x.email)}" data-uaction="delete">Delete</button>`,
        ].join(" ");
    const who = `${esc(x.email)}${x.disabled ? ' <span class="pill bad">DISABLED</span>' : ""}${x.active_now ? ' <span class="pill active-stripe">ONLINE</span>' : ""}`;
    return `<tr>
      <td>${who}</td><td class="col-name">${esc(x.name) || "—"}</td>
      <td class="num">${fmtDay(x.created_at)}</td>
      <td class="num">${fmtDay(x.last_login_at)}</td>
      <td>${sub}</td>
      <td>${games}</td>
      <td class="num">${x.hands_today}</td>
      <td class="num col-total">${x.hands_total}</td>
      <td>${btn}</td>
      <td class="acct">${acct}</td>
    </tr>`;
  }).join("") : '<tr><td colspan="10" class="muted">No user matches that search.</td></tr>';
}

async function userAction(b) {
  const uid = parseInt(b.dataset.uid, 10);
  const action = b.dataset.uaction;
  if (action === "delete") {
    const typed = window.prompt(
      `Delete ${b.dataset.email}? Their personal data is removed; home-game ledgers and other players' histories stay (as "Deleted player"). Type the email to confirm:`
    );
    if (!typed) return;
    await post("/admin/api/users/delete", { user_id: uid, confirm: typed });
    return;
  }
  if (action === "disable" && !window.confirm("Disable this account? They are signed out at once and can't sign in.")) return;
  await post("/admin/api/users/action", { user_id: uid, action });
}

async function refreshAll() {
  const results = await Promise.allSettled([load(), loadSystem(), loadAudit()]);
  const failed = results.find((r) => r.status === "rejected");
  if (failed) showErr(new Error(`${failed.reason.message} — are you signed in as an admin?`));
  else clearErr();
}

$("user-search").addEventListener("input", (e) => { USERS.q = e.target.value; renderUsers(); });
$("user-sort").addEventListener("change", (e) => { USERS.sort = e.target.value; renderUsers(); });

// One delegated listener for every button the tables render.
document.addEventListener("click", async (ev) => {
  const b = ev.target.closest("button");
  if (!b || b.disabled) return;
  const run = async (fn) => {
    b.disabled = true;
    try { await fn(); await refreshAll(); } catch (e) { showErr(e); } finally { b.disabled = false; }
  };
  if (b.dataset.model) return run(() => modelAction(b.dataset.model, b.dataset.maction));
  if (b.dataset.uaction) return run(() => userAction(b));
  if (b.dataset.uid) {
    return run(() => {
      const games = b.dataset.gaction;
      const url = games ? "/admin/api/games_access" : "/admin/api/grant";
      return post(url, { user_id: parseInt(b.dataset.uid, 10), action: games || b.dataset.action });
    });
  }
  if (b.id === "maint-clear") return run(() => post("/admin/api/maintenance", { clear: true }));
  if (b.id === "sys-refresh") return run(async () => {});
});

$("maint-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const message = $("maint-message").value.trim();
  const minutes = $("maint-minutes").value.trim();
  if (!message) return;
  try {
    await post("/admin/api/maintenance", minutes ? { message, in_minutes: parseInt(minutes, 10) } : { message });
    $("maint-message").value = "";
    await refreshAll();
  } catch (e) { showErr(e); }
});

refreshAll();
