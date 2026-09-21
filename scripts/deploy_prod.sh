#!/usr/bin/env bash
# Ship this working tree to production (wrapgto.com) and restart the app.
#
#   scripts/deploy_prod.sh check     read-only: can I log in? is the app up? can the
#                                    server rebuild the Rust engine? (changes nothing)
#   scripts/deploy_prod.sh pack      local only: build the archive and show what would
#                                    ship (no network)
#   scripts/deploy_prod.sh           ship + rebuild the engine + restart + health check
#   SKIP_ENGINE=1 scripts/deploy_prod.sh    ship + restart only
#
# WHERE it ships to is deliberately not in this file: pass WRAPGTO_HOST=root@<server>
# or (better) define `Host wrapgto-prod` in ~/.ssh/config — see PUBLIC_SETUP.md,
# "Deploying from a new machine".
#
# It is the documented "tar over ssh" ship plus the step a code-only ship misses:
# REBUILDING THE RUST ENGINE on the server. The home games' verifiable shuffle needs
# the engine's `reset_with_deck`; an engine built before it still runs (the tables
# deal the old way and say "Unverified shuffle"), so a failed rebuild is a warning
# here, never a broken site. The server's checkpoints/ and data/ are never touched:
# they are not in the archive, and extracting only adds or overwrites.
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
    remote "sudo -u wrapgto -H bash -lc 'command -v cargo >/dev/null && echo \"   cargo (user wrapgto): ok\" || echo \"   cargo (user wrapgto): MISSING — the engine cannot be rebuilt here yet\"; test -x $APP/.venv/bin/maturin && echo \"   maturin: ok\" || echo \"   maturin: MISSING (.venv/bin/pip install maturin)\"'" || true
    echo "== nothing was changed"
    exit 0 ;;
  deploy) ;;
  *) echo "usage: $0 [check|pack|deploy]"; exit 2 ;;
esac

dirty=$(git status --porcelain --untracked-files=no | grep -v '^ M .grok/' | wc -l | tr -d ' ')
echo "== shipping $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD) to $HOST:$APP"
[ "$dirty" = "0" ] || echo "   note: $dirty tracked file(s) have uncommitted changes — they ship too"
pack
echo "   archive: $(du -h "$TARBALL" | cut -f1)"
remote "cat > /tmp/wrapgto-ship.tgz" < "$TARBALL"
remote "tar -xzf /tmp/wrapgto-ship.tgz -C '$APP' && rm -f /tmp/wrapgto-ship.tgz && chown -R wrapgto:wrapgto '$APP'"
echo "== shipped"

if [ "${SKIP_ENGINE:-0}" != "1" ]; then
  echo "== rebuilding the Rust engine on the server (a few minutes the first time)"
  if remote "cd '$APP' && sudo -u wrapgto -H bash -lc 'command -v cargo >/dev/null && .venv/bin/maturin develop --release'"; then
    echo "== engine rebuilt"
  else
    echo "!! engine NOT rebuilt (no cargo/maturin for user wrapgto, or the build failed)."
    echo "!! The site still runs; home-game hands are dealt the old way and show 'Unverified shuffle'."
    echo "!! Run '$0 check' to see what is missing."
  fi
fi

echo "== restarting"
remote "systemctl restart wrapgto && sleep 4 && systemctl is-active wrapgto && curl -fsS http://127.0.0.1:8770/health && echo"
remote "cd '$APP' && sudo -u wrapgto -H .venv/bin/python -c 'from plo5bp import env; print(\"verifiable shuffle — engine support:\", hasattr(env._RustGameState, \"reset_with_deck\"))'" || true
echo "== done"
