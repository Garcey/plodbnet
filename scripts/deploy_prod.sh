#!/usr/bin/env bash
# Ship this working tree to production (wrapgto.com).
#
#   scripts/deploy_prod.sh check    read-only: login, service, toolchain, open tables
#   scripts/deploy_prod.sh pack     local only: what would ship (no network)
#   scripts/deploy_prod.sh stage    upload, build the engine and START-TEST the new code
#                                   in a staging folder on the server — the live site
#                                   is not touched
#   scripts/deploy_prod.sh          stage, then switch the live site over: one ~5 s
#                                   restart, health check, automatic rollback if the
#                                   new code does not come up
#   SKIP_ENGINE=1 scripts/deploy_prod.sh    same, reusing the engine that is live
#
# WHERE it ships to is deliberately not in this file: pass WRAPGTO_HOST=root@<server>
# or (better) define `Host wrapgto-prod` in ~/.ssh/config — see PUBLIC_SETUP.md,
# "Deploying from a new machine".
#
# Order of events on the server, and why:
#   1. database backup (the existing daily job, run now)
#   2. unpack into /opt/wrapgto/ship-staging — never over the live code
#   3. build the Rust engine THERE, as root (rustup lives in /root; `wrapgto` is a
#      no-login service account and stays one). The home games' verifiable shuffle
#      needs the engine's `reset_with_deck`; Python and engine must go live together.
#   4. pre-flight: import the whole new app from staging with a throwaway database.
#      Anything wrong — a build error, a missing package, a Python-version problem —
#      stops HERE, with the live site exactly as it was.
#   5. keep a copy of the live code, copy staging over it, restart once, health check;
#      if the app is not healthy, put the copy back and restart again.
# The server's checkpoints/ and data/ are never in the archive and are never deleted.
set -euo pipefail

MODE="${1:-deploy}"
HOST="${WRAPGTO_HOST:-wrapgto-prod}"
APP="${WRAPGTO_APP:-/opt/wrapgto/app}"
cd "$(dirname "$0")/.."

# On Windows (Git Bash) use Windows' own OpenSSH: it talks to the Windows ssh-agent
# service, so a passphrase-protected key is unlocked once, not once per call.
SSH="${WRAPGTO_SSH:-ssh}"
if [ -z "${WRAPGTO_SSH:-}" ] && [ -x /c/Windows/System32/OpenSSH/ssh.exe ]; then
  SSH=/c/Windows/System32/OpenSSH/ssh.exe
  export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'  # never rewrite remote paths or commands
fi
# Ask for a passphrase only when a person is there to type it.
if [ -t 0 ]; then BATCH=no; else BATCH=yes; fi
remote() { "$SSH" -o BatchMode=$BATCH -o ConnectTimeout=15 "$HOST" "$@"; }

pack() {  # -> $TARBALL. Never shipped: VCS + local envs, weights and user data (they
          # live on the server), build output, other platforms' engines, anything secret.
  TMPDIR_SHIP="$(mktemp -d)"
  trap 'rm -rf "$TMPDIR_SHIP"' EXIT
  TARBALL="$TMPDIR_SHIP/ship.tgz"
  tar --exclude=./.git --exclude=./.venv --exclude=./checkpoints --exclude=./data \
      --exclude=./target --exclude=./rust_engine/target --exclude=node_modules \
      --exclude=__pycache__ --exclude='*.pyc' --exclude=.pytest_cache \
      --exclude='*.pyd' --exclude='*.so' --exclude=./.claude --exclude=./.grok \
      --exclude=./screenrecords --exclude=./runs --exclude=./ssh --exclude='.env*' \
      --exclude='*.pem' --exclude='*.key' --exclude='*.pfx' --exclude='*.p12' \
      --exclude='id_rsa*' --exclude='id_ed25519*' --exclude='client_secret*' \
      --exclude='*.db' --exclude='*.db-wal' --exclude='*.db-shm' \
      -czf "$TARBALL" .
}

case "$MODE" in
  pack)
    pack
    echo "== $(tar -tzf "$TARBALL" | wc -l | tr -d ' ') entries, $(du -h "$TARBALL" | cut -f1) — top level:"
    tar -tzf "$TARBALL" | cut -d/ -f2 | sort -u | tr '\n' ' '; echo
    echo "== anything that looks like a secret or user data (should be empty):"
    tar -tzf "$TARBALL" | grep -iE '(^|/)\.env|\.pem$|\.key$|(^|/)id_(rsa|ed25519)|(^|/)ssh/|\.db$|(^|/)data/|(^|/)checkpoints/|client_secret' || echo "   (nothing)"
    exit 0 ;;
  check)
    echo "== logging in to $HOST"
    remote 'echo "   connected as $(whoami) on $(hostname)"' || {
      echo "!! cannot log in. Is this machine's PUBLIC key in the server's ~/.ssh/authorized_keys,"
      echo "!! and does ~/.ssh/config define Host wrapgto-prod? (PUBLIC_SETUP.md)"; exit 1; }
    remote "printf '   app service: '; systemctl is-active wrapgto || true; test -d '$APP' && echo '   app dir: ok' || echo '   app dir: MISSING ($APP)'; printf '   disk: '; df -h '$APP' | tail -1"
    remote "bash -lc 'command -v cargo >/dev/null && echo \"   cargo (root): ok — \$(cargo --version)\" || echo \"   cargo (root): MISSING — the engine cannot be rebuilt on this server yet\"'; test -x '$APP/.venv/bin/maturin' && echo '   maturin: ok' || echo '   maturin: MISSING (.venv/bin/pip install maturin)'; ls -l --time-style=long-iso '$APP'/python/plo5bp/_engine*.so 2>/dev/null | awk '{print \"   engine on the server: built \" \$6 \" \" \$7}'" || true
    remote "cd '$APP' && sudo -u wrapgto -H .venv/bin/python -c 'import sqlite3; c = sqlite3.connect(\"file:data/public.db?mode=ro\", uri=True); n = c.execute(\"select count(*) from homegames where status=\\\"open\\\"\").fetchone()[0]; print(\"   open home-game tables:\", n, \"(the switch restarts the app: a hand in progress ends)\" if n else \"\")'" 2>/dev/null || true
    echo "== nothing was changed"
    exit 0 ;;
  stage|deploy) ;;
  *) echo "usage: $0 [check|pack|stage|deploy]"; exit 2 ;;
esac

dirty=$(git status --porcelain --untracked-files=no | grep -v '^ M .grok/' | wc -l | tr -d ' ')
echo "== $MODE: $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD) -> $HOST:$APP"
[ "$dirty" = "0" ] || echo "   note: $dirty tracked file(s) have uncommitted changes — they ship too"
pack
SUM=$(sha256sum "$TARBALL" | cut -d' ' -f1)
echo "   archive: $(du -h "$TARBALL" | cut -f1)  sha256 ${SUM:0:12}…"
remote "cat > /tmp/wrapgto-ship.tgz" < "$TARBALL"

# The server-side half travels as a FILE, not on stdin: a script fed on stdin gets
# eaten by the first child process that reads stdin.
cat > "$TMPDIR_SHIP/remote.sh" <<'REMOTE'
set -euo pipefail
STAGE=/opt/wrapgto/ship-staging
KEEP=/opt/wrapgto/backups
TS=$(date +%Y%m%d-%H%M%S)
PRE=/tmp/wrapgto-preflight-$TS
cleanup() { rm -rf "$PRE" /tmp/wrapgto-wheels /tmp/wrapgto-whl-x /tmp/wrapgto-ship.tgz; }
trap cleanup EXIT

echo "$SUM  /tmp/wrapgto-ship.tgz" | sha256sum -c --quiet || { echo "!! the upload is corrupted (checksum mismatch). Nothing was changed."; exit 2; }
echo "== [server] upload verified"

echo "== [server] database backup"
if [ -x /etc/cron.daily/wrapgto-backup ]; then
  /etc/cron.daily/wrapgto-backup </dev/null || { echo "!! the database backup failed — stopping. Nothing was changed."; exit 2; }
  echo "   $(ls -t "$KEEP" | grep -v '^app-before' | head -1 || true)"
else
  echo "   (no backup job found — skipped)"
fi

echo "== [server] unpacking into staging (the live code is not touched)"
rm -rf "${STAGE:?}"; mkdir -p "$STAGE"
tar -xzf /tmp/wrapgto-ship.tgz -C "$STAGE"

if [ "$SKIP_ENGINE" = "1" ]; then
  echo "== [server] reusing the live engine"
  cp -p "$APP"/python/plo5bp/_engine*.so "$STAGE/python/plo5bp/"
else
  echo "== [server] building the Rust engine in staging (a few minutes the first time)"
  bash -lc 'command -v cargo >/dev/null' || { echo "!! root has no cargo — cannot build. Nothing was changed."; exit 3; }
  rm -rf /tmp/wrapgto-wheels /tmp/wrapgto-whl-x; mkdir -p /tmp/wrapgto-wheels /tmp/wrapgto-whl-x
  ( cd "$STAGE" && bash -lc "CARGO_TARGET_DIR=/opt/wrapgto/cargo-target '$APP/.venv/bin/maturin' build --release --compatibility linux -i '$APP/.venv/bin/python' -o /tmp/wrapgto-wheels" </dev/null ) 2>&1 | tail -4     || { echo "!! the engine build failed. Nothing was changed."; exit 3; }
  "$APP/.venv/bin/python" -m zipfile -e "$(ls -t /tmp/wrapgto-wheels/*.whl | head -1)" /tmp/wrapgto-whl-x
  so=$(find /tmp/wrapgto-whl-x -name '_engine*.so' | head -1)
  [ -n "$so" ] || { echo "!! the build produced no engine binary. Nothing was changed."; exit 3; }
  cp "$so" "$STAGE/python/plo5bp/"
  echo "   built $(basename "$so")"
fi

echo "== [server] pre-flight: does the NEW code start? (throwaway database)"
chown -R wrapgto:wrapgto "$STAGE"
mkdir -p "$PRE"; chown wrapgto:wrapgto "$PRE"
( cd "$STAGE" && sudo -u wrapgto -H env PYTHONPATH="$STAGE/python" PLO5BP_PUBLIC=1 \
    PLO5BP_DB="$PRE/preflight.db" PLO5BP_TRAINER_STATS="$PRE/stats.json" PLO5BP_HOMEGAME_GRADING=0 \
    "$APP/.venv/bin/python" - <<'PY'
import sys
import plo5bp
assert "ship-staging" in plo5bp.__file__, f"pre-flight imported the wrong tree: {plo5bp.__file__}"
from plo5bp import env
from plo5bp.ui import homegame, server  # noqa: F401 — the whole app, public layer included
print("   imports ok · python", sys.version.split()[0],
      "· engine has reset_with_deck:", hasattr(env._RustGameState, "reset_with_deck"),
      "· verifiable shuffle on:", homegame.FAIR_ON)
PY
) 2>&1 | grep -v "not found .* using random-init" || { echo "!! the new code does not start. Nothing was changed."; exit 4; }

if [ "$MODE" = "stage" ]; then
  echo "== [server] staged and start-tested in $STAGE — the live site was not touched"
  exit 0
fi

echo "== [server] keeping a copy of the live code: $KEEP/app-before-$TS.tgz"
mkdir -p "$KEEP"
tar -czf "$KEEP/app-before-$TS.tgz" --exclude=./.venv --exclude=./data --exclude=./checkpoints \
    --exclude=./target --exclude=__pycache__ -C "$APP" .
ls -t "$KEEP"/app-before-*.tgz | tail -n +6 | xargs -r rm -f   # this script's own copies: keep five

healthy() {  # the app loads torch + its models before it listens: poll for up to ~2 minutes
  for _ in $(seq 1 40); do
    sleep 3
    if systemctl is-active --quiet wrapgto && curl -fsS -m 5 http://127.0.0.1:8770/health >/dev/null 2>&1; then return 0; fi
  done
  return 1
}

echo "== [server] switching the live site over"
if command -v rsync >/dev/null; then rsync -a "$STAGE"/ "$APP"/; else cp -a "$STAGE"/. "$APP"/; fi
chown -R wrapgto:wrapgto "$APP"
systemctl restart wrapgto
if healthy; then
  echo "== [server] LIVE and healthy"
else
  echo "!! the app did not come up healthy — rolling back to $KEEP/app-before-$TS.tgz"
  journalctl -u wrapgto -n 25 --no-pager | sed 's/^/   | /' || true
  tar -xzf "$KEEP/app-before-$TS.tgz" -C "$APP"
  chown -R wrapgto:wrapgto "$APP"
  systemctl restart wrapgto
  if healthy; then echo "!! rolled back: the PREVIOUS version is live and healthy"; else echo "!! STILL unhealthy after the rollback — look at: journalctl -u wrapgto -n 80"; fi
  exit 5
fi
REMOTE
remote "cat > /tmp/wrapgto-deploy.sh" < "$TMPDIR_SHIP/remote.sh"
remote "APP='$APP' MODE='$MODE' SKIP_ENGINE='${SKIP_ENGINE:-0}' SUM='$SUM' bash /tmp/wrapgto-deploy.sh; rc=\$?; rm -f /tmp/wrapgto-deploy.sh; exit \$rc"

if [ "$MODE" = "deploy" ]; then
  remote "cd '$APP' && sudo -u wrapgto -H .venv/bin/python -c 'from plo5bp import env; print(\"   verifiable shuffle — engine support on the live site:\", hasattr(env._RustGameState, \"reset_with_deck\"))'" || true
fi
echo "== done"
