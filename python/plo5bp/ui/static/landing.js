"use strict";
// Landing-page extras for signed-out visitors (ACC-016 / PERF-002). The page
// itself needs no script: it is visible as served, and every "Sign in" is a
// plain link to /auth/login. This file only adds the optional bits:
//   - why a sign-in didn't complete (/?login=failed&why=…, ACC-025);
//   - the loopback-only dev sign-in row (local previews).
// It does nothing on a signed-in page (the app is there instead).
(function () {
  if (document.getElementById("top-bar")) return;

  const REASONS = {
    cancelled: "Sign-in was cancelled — no problem, try again whenever you like.",
    expired: "That sign-in page had expired. Please try again.",
    unverified: "Google didn't confirm an email address for that account. Please use an account with a verified email.",
    conflict: "That email address belongs to an account linked to a different Google login. Sign in with that Google account, or with an email link.",
    disabled: "This account has been disabled. Contact support@wrapgto.com if you think that's a mistake.",
  };
  const params = new URLSearchParams(location.search);
  if (params.get("login") === "failed") {
    const note = document.getElementById("login-error");
    if (note) {
      note.textContent = REASONS[params.get("why")] || "Sign-in didn't complete. Please try again.";
      note.hidden = false;
    }
    params.delete("login");
    params.delete("why");
    const qs = params.toString();
    try { history.replaceState(null, "", location.pathname + (qs ? `?${qs}` : "") + location.hash); } catch (_) { /* fine */ }
  }

  // Only a request that could actually use the dev login learns it exists
  // (/me says so); production never has it.
  fetch("/me", { credentials: "same-origin" })
    .then((r) => (r.ok ? r.json() : null))
    .then((me) => {
      // Email sign-in link (ACC-008): offered only when the server can send it.
      if (me && me.email_login && !document.querySelector(".gate-email")) {
        const google = document.querySelector(".gate-google");
        if (google) {
          const a = document.createElement("a");
          a.className = "gate-email";
          a.href = "/auth/email";
          a.textContent = "No Google account? Email me a sign-in link";
          google.insertAdjacentElement("afterend", a);
        }
      }
      if (!me || !me.dev_login) return;
      const row = document.getElementById("dev-login-row");
      const btn = document.getElementById("dev-login-btn");
      const input = document.getElementById("dev-login-email");
      if (!row || !btn || !input) return;
      row.hidden = false;
      const go = () => {
        const email = input.value.trim();
        if (email) window.location.href = `/auth/dev?email=${encodeURIComponent(email)}`;
      };
      btn.addEventListener("click", go);
      input.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
    })
    .catch(() => { /* the page works without it */ });
})();
