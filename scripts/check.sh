#!/usr/bin/env bash
# Run CI's checks on this machine (Git Bash on Windows, or any bash).
#
#   bash scripts/check.sh            everything: shell scripts, secrets, Python + Rust tests
#   bash scripts/check.sh site       just what a website deploy needs (home games,
#                                    public, study/trainer, deploy tooling) — minutes
#   bash scripts/check.sh training   the engine, encoders, rollout, PPO
#   bash scripts/check.sh quick      everything but the slow files
#
# It rebuilds nothing: if the Rust engine is older than its sources the test run
# says STALE ENGINE at the end — rebuild with  .venv/Scripts/maturin develop --release
set -uo pipefail
cd "$(dirname "$0")/.."

WHAT="${1:-all}"
PY=.venv/Scripts/python
[ -x "$PY" ] || PY=.venv/bin/python
fails=()
step() { printf '\n== %s\n' "$*"; }

step "shell scripts (bash -n)"
for f in $(git ls-files '*.sh') ops/bin/* scripts/hooks/*; do
  [ -f "$f" ] || continue
  bash -n "$f" || fails+=("bash -n $f")
done
if command -v shellcheck >/dev/null 2>&1; then
  shellcheck -S error scripts/deploy_prod.sh ops/deploy-remote.sh ops/bin/* scripts/hooks/* scripts/check.sh \
    || fails+=("shellcheck")
else
  echo "   (shellcheck not installed — CI runs it)"
fi

step "secrets"
if command -v gitleaks >/dev/null 2>&1; then
  gitleaks detect --redact --config .gitleaks.toml --no-banner || fails+=("gitleaks")
else
  echo "   (gitleaks not installed — CI runs it; winget install gitleaks.gitleaks)"
fi

step "python tests ($WHAT)"
case "$WHAT" in
  site) sel=(-m "homegame or public or ui or ops") ;;
  training) sel=(-m training) ;;
  quick) sel=(-m "not slow") ;;
  *) sel=() ;;
esac
"$PY" -m pytest -q "${sel[@]}" || fails+=("pytest $WHAT")

if [ "$WHAT" = "all" ] || [ "$WHAT" = "training" ]; then
  step "rust: format, lints, tests (scripts/rust_check.sh)"
  if command -v cargo >/dev/null 2>&1; then
    bash scripts/rust_check.sh || fails+=("rust_check")
  else
    echo "   (cargo not installed)"
  fi
fi

echo
if [ "${#fails[@]}" -eq 0 ]; then
  echo "== all checks passed"
else
  echo "!! failed: ${fails[*]}"
  exit 1
fi
