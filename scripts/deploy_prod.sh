#!/usr/bin/env bash
# Ship this working tree to production (wrapgto.com) and restart the app.
#
# Run it from a machine whose SSH key the server accepts (the laptop — see
# PUBLIC_SETUP.md). It is the documented "tar-over-ssh ship" plus the one step
# a code-only ship misses: REBUILDING THE RUST ENGINE on the server. The home
# games' verifiable shuffle needs the engine's `reset_with_deck` entry point; an
# engine built before it still runs fine (homegame.FAIR_ON turns itself off and
# the tables deal the old way, shown as "unverified") — so a failed rebuild is
# a warning here, never a broken site.
#
#   scripts/deploy_prod.sh            ship + rebuild + restart + health check
#   SKIP_ENGINE=1 scripts/deploy_prod.sh   ship + restart only
#   WRAPGTO_HOST=user@host scripts/deploy_prod.sh
set -euo pipefail

HOST="${WRAPGTO_HOST:-root@87.99.132.209}"
APP="${WRAPGTO_APP:-/opt/wrapgto/app}"
cd "$(dirname "$0")/.."

dirty=$(git status --porcelain --untracked-files=no | grep -v '^ M .grok/' | wc -l | tr -d ' ')
echo "== shipping $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD) to $HOST:$APP"
[ "$dirty" = "0" ] || echo "   note: $dirty tracked file(s) have uncommitted changes — they ship too"

# Never shipped: VCS + local envs, weights and user data (they live on the
# server), build output, compiled engines for other platforms, secrets.
tar --exclude=.git --exclude=.venv --exclude=checkpoints --exclude=data \
    --exclude=target --exclude=node_modules --exclude=__pycache__ --exclude='*.pyc' \
    --exclude='*.pyd' --exclude='*.so' --exclude=.claude --exclude=.grok \
    --exclude=screenrecords --exclude=runs --exclude=ssh --exclude='.env*' \
    -czf - . | ssh "$HOST" "tar -xzf - -C '$APP' && chown -R wrapgto:wrapgto '$APP'"
echo "== shipped"

if [ "${SKIP_ENGINE:-0}" != "1" ]; then
  echo "== rebuilding the Rust engine on the server (a few minutes the first time)"
  if ssh "$HOST" "cd '$APP' && sudo -u wrapgto -H bash -lc 'command -v cargo >/dev/null && .venv/bin/maturin develop --release'"; then
    echo "== engine rebuilt"
  else
    echo "!! engine NOT rebuilt (no cargo/maturin for user wrapgto, or the build failed)."
    echo "!! The site still runs; home-game hands are dealt the old way and show 'Unverified shuffle'."
    echo "!! Fix: install rustup for user wrapgto, 'pip install maturin' in $APP/.venv, run this again."
  fi
fi

echo "== restarting"
ssh "$HOST" "systemctl restart wrapgto && sleep 4 && systemctl is-active wrapgto && curl -fsS http://127.0.0.1:8770/health && echo"
ssh "$HOST" "cd '$APP' && sudo -u wrapgto -H .venv/bin/python -c \"
import os; os.environ.setdefault('PLO5BP_PUBLIC','0')
from plo5bp.env import BombPotEnv
print('verifiable shuffle engine support:', hasattr(BombPotEnv, 'reset_with_deck') and hasattr(__import__('plo5bp.env', fromlist=['x'])._RustGameState, 'reset_with_deck'))
\"" || true
echo "== done — https://wrapgto.com"
