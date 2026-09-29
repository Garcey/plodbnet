#!/usr/bin/env bash
# Ship code — or a model — to production (wrapgto.com), and undo either.
# Live changes need the owner's explicit OK, every time.
#
#   bash scripts/deploy_prod.sh check           read-only: service, live commit + model, games
#                                               in progress, backups, rollback targets, the unit
#   bash scripts/deploy_prod.sh pack            offline: exactly what would ship
#   bash scripts/deploy_prod.sh stage           upload, build the engine, start-test the new code
#                                               with the live model and a COPY of the live data
#                                               — the live site is not touched
#   bash scripts/deploy_prod.sh                 stage, then switch the live site over: one
#                                               restart, health check, automatic rollback
#   bash scripts/deploy_prod.sh rollback [NAME] put an earlier version back (lists them first)
#   bash scripts/deploy_prod.sh restore-db [NAME]  put a database backup back (no NAME = list)
#   bash scripts/deploy_prod.sh promote FILE    verify a checkpoint with the live code and stage it;
#                                               then Promote it in /admin — NO restart
#                                               (RESTART=1: swap it in now, with a restart)
#   bash scripts/deploy_prod.sh promote-undo    put the previous model back (with a restart)
#   bash scripts/deploy_prod.sh watch           the latest change's log (followed while it runs)
#   bash scripts/deploy_prod.sh recover         finish or undo a change whose process died
#                                               half-way (a crash or a reboot of the server)
#   bash scripts/deploy_prod.sh install-unit    install ops/systemd/wrapgto.service: checked,
#                                               guarded, backed up, reverted if unhealthy
#   bash scripts/deploy_prod.sh freeze          save the server's exact Python packages to
#                                               requirements/server.txt
#   bash scripts/deploy_prod.sh lock            re-resolve requirements/server.in ON the server
#                                               into requirements/server.txt (after adding a package)
#
# Once a change starts, the SERVER finishes it by itself: closing this window or losing
# the connection never stops it half-way (`watch` shows how it goes).
#
# Settings (environment variables, e.g.  DRY_RUN=1 bash scripts/deploy_prod.sh):
#   DRY_RUN=1          every local step, then print what the server would do — no network
#   LIST=1             pack / deploy: print every file that ships
#   ALLOW_DIRTY=1      ship uncommitted changes (default: refuse — the live site is a commit)
#   FORCE=1            skip ONLY the "is anyone playing?" check: a hand being played is void
#                      (chips go back) and running tables come back paused until Start
#   ROLLBACK_ANYWAY=1  rollback: switch although that version's pre-flight FAILED
#   RESTART=1          promote: swap the model in with a restart instead of staging it
#   OBS_REV=1|2        promote / promote-undo: the model was trained on the other observation
#                      revision — PLO5BP_OBS_REV changes with it (and back, if unhealthy)
#   SKIP_ENGINE=1      reuse the live engine instead of building it (only when it was built
#                      from this commit's Rust sources — the pre-flight checks)
#   ENGINE_WHEEL=FILE  ship a prebuilt Linux wheel (e.g. CI's) instead of building on the server
#   ALLOW_ENGINE_MISMATCH=1  accept an engine NOT built from this commit's Rust sources
#   ALLOW_NO_CRITIC=1  accept a model whose critic does not load
#   ALLOW_RED=1        deploy although CI failed on this commit
#   CONFIRM=WORD       answer the typed confirmation (rollback, restore-db, promote, recover…)
#   WRAPGTO_HOST       ssh destination (default: the `wrapgto-prod` alias in ~/.ssh/config)
#
# WHERE it ships to is deliberately not in this file: define `Host wrapgto-prod` in
# ~/.ssh/config (docs/ops/PRODUCTION.md, "A new machine").
# What ships: the COMMITTED files under python/ rust_engine/ requirements/ ops/ plus
# Cargo.toml Cargo.lock rust-toolchain.toml pyproject.toml — an allowlist, exported with `git archive`
# (LF line endings, nothing untracked or ignored), content-scanned for secrets.
# The server-side half is ops/deploy-remote.sh (+ ops/deploytool.py): read it for the
# exact order of events on the server.
set -euo pipefail

MODE="${1:-deploy}"
ARG="${2:-}"
HOST="${WRAPGTO_HOST:-wrapgto-prod}"
APP="${WRAPGTO_APP:-/opt/wrapgto/app}"
DRY_RUN="${DRY_RUN:-0}"
cd "$(dirname "$0")/.."

SHIP=(python rust_engine requirements ops Cargo.toml Cargo.lock rust-toolchain.toml pyproject.toml)
# Key-shaped secrets (a word like "sk_live_" alone does not match).
SECRET_RE='(sk|rk)_live_[0-9A-Za-z]{16,}|whsec_[0-9A-Za-z]{20,}|GOCSPX-[0-9A-Za-z_-]{20,}|-----BEGIN ([A-Z]+ )?PRIVATE KEY-----|AKIA[0-9A-Z]{16}|gh[pousr]_[0-9A-Za-z]{30,}|github_pat_[0-9A-Za-z_]{40,}|xox[baprs]-[0-9A-Za-z-]{10,}'

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }

# On Windows (Git Bash) use Windows' own OpenSSH: it talks to the Windows ssh-agent
# service, so a passphrase-protected key is unlocked once, not once per call.
SSH="${WRAPGTO_SSH:-ssh}"
if [ -z "${WRAPGTO_SSH:-}" ] && [ -x /c/Windows/System32/OpenSSH/ssh.exe ]; then
  SSH=/c/Windows/System32/OpenSSH/ssh.exe
fi
# Ask for a passphrase only when a person is there to type it.
if [ -t 0 ]; then BATCH=no; else BATCH=yes; fi
# MSYS_NO_PATHCONV: never let Git Bash rewrite the remote command's paths.
remote() { MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' "$SSH" -o BatchMode=$BATCH -o ConnectTimeout=15 -o ServerAliveInterval=30 "$HOST" "$@"; }

W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT

say() { printf '%s\n' "$*"; }
die() { printf '!! %s\n' "$*"; exit 1; }
safe() { printf '%s' "$1" | tr -c 'A-Za-z0-9._:+@=/-' '_'; }  # values sent to the server's shell

confirm() {  # $1 = the word to type
  local word=$1 reply=""
  [ "$DRY_RUN" = "1" ] && { say "   (DRY_RUN: would ask you to type $word)"; return 0; }
  [ "${CONFIRM:-}" = "$word" ] && return 0
  [ -t 0 ] || die "no terminal to confirm on — set CONFIRM=$word to go ahead"
  read -r -p "   type $word to go ahead (anything else cancels): " reply
  [ "$reply" = "$word" ] || { say "== cancelled — nothing was changed"; exit 1; }
}

# --- what ships -------------------------------------------------------------

build_archive() {  # -> $W/app.tgz; sets COMMIT BRANCH DIRTY DIFF_SHA TREE
  COMMIT="$(git rev-parse HEAD)"
  BRANCH="$(git rev-parse --abbrev-ref HEAD)"
  DIRTY=0; DIFF_SHA=""; TREE="$COMMIT"
  local changes n
  changes="$(git status --porcelain --untracked-files=all -- "${SHIP[@]}")"
  if [ -n "$changes" ]; then
    n="$(printf '%s\n' "$changes" | wc -l | tr -d ' ')"
    say "   $n uncommitted change(s) in what ships:"
    printf '%s\n' "$changes" | head -15 | sed 's/^/     /'
    [ "$n" -gt 15 ] && say "     … and $((n - 15)) more"
    if [ "${ALLOW_DIRTY:-0}" = "1" ]; then
      DIRTY=1
      # A throwaway index: the working tree as it is, without touching the real index.
      local p present=()
      for p in "${SHIP[@]}"; do
        if [ -e "$p" ] || [ -n "$(git ls-tree HEAD -- "$p")" ]; then present+=("$p"); fi
      done
      GIT_INDEX_FILE="$W/index" git read-tree HEAD
      GIT_INDEX_FILE="$W/index" git add -A -- "${present[@]}"
      TREE="$(GIT_INDEX_FILE="$W/index" git write-tree)"
      DIFF_SHA="$(git diff "$COMMIT" "$TREE" | sha256sum | cut -d' ' -f1)"
      say "   ALLOW_DIRTY=1: shipping them as they are (recorded as ${COMMIT:0:7} + changes ${DIFF_SHA:0:12})"
    elif [ "$MODE" = "pack" ] || [ "$DRY_RUN" = "1" ]; then
      say "   (this shows the COMMITTED version; a real deploy refuses until they are committed, or ALLOW_DIRTY=1)"
    else
      die "commit them first — the live site should always be a commit — or run with ALLOW_DIRTY=1"
    fi
  fi
  local paths=()
  mapfile -t paths < <(git ls-tree --name-only "$TREE" -- "${SHIP[@]}")
  [ "${#paths[@]}" -gt 0 ] || die "nothing to ship"
  git -c core.autocrlf=false -c core.eol=lf archive --format=tar "$TREE" -- "${paths[@]}" | gzip -9n > "$W/app.tgz"
}

scan_archive() {  # refuse anything that looks like a secret or user data
  rm -rf "$W/scan"; mkdir -p "$W/scan"
  tar -xzf "$W/app.tgz" -C "$W/scan"
  local names content
  names="$(cd "$W/scan" && find . -type f | sed 's#^\./##' | grep -iE '(^|/)\.env|\.pem$|\.key$|\.p12$|\.pfx$|(^|/)id_(rsa|ed25519)|\.db$|\.sqlite3?$|client_secret|(^|/)(data|checkpoints)/' || true)"
  content="$( (cd "$W/scan" && grep -rlIE "$SECRET_RE" . 2>/dev/null) | sed 's#^\./##' || true)"
  if [ -n "$names$content" ]; then
    say "!! refusing to ship — these look like secrets or user data (names only, contents not shown):"
    printf '%s\n' $names $content | sort -u | sed 's/^/     /'
    exit 1
  fi
}

archive_summary() {
  say "   archive: $(tar -tzf "$W/app.tgz" | grep -vc '/$') files, $(du -h "$W/app.tgz" | cut -f1), sha256 $(sha256sum "$W/app.tgz" | cut -c1-12)"
  say "   top level: $(tar -tzf "$W/app.tgz" | cut -d/ -f1 | sort -u | tr '\n' ' ')"
  if [ "${LIST:-0}" = "1" ]; then tar -tzf "$W/app.tgz" | grep -v '/$' | sed 's/^/     /'; else say "   (LIST=1 prints every file)"; fi
}

dry_plan() {  # what the server half does, in order (ops/deploy-remote.sh)
  local engine="cargo build on the server"
  if [ -n "${ENGINE_WHEEL:-}" ]; then engine="from $(basename "$ENGINE_WHEEL")"; elif [ "${SKIP_ENGINE:-0}" = "1" ]; then engine="reuse the live one"; fi
  say "   on the server — detached from this session, one change at a time (a lock) — in order:"
  say "   unpack into $(dirname "$APP")/ship-staging · Python packages (reuse the live environment, or"
  say "   build venvs/<hash> from requirements/server.txt) · engine: $engine · precompile, make it"
  say "   root-owned and read-only · pre-flight as the service user (production settings, the live"
  say "   model, a COPY of the live database; the engine must match this commit's Rust sources) ·"
  say "   BUILD_INFO.json"
  if [ "$MODE" = "deploy" ]; then
    say "   · home-games guard · fresh backup · switch: stop, app -> releases/<time>, staging -> app,"
    say "   move data/ checkpoints/ .venv across, start · health check · automatic rollback if unhealthy"
  fi
}

# --- the upload ----------------------------------------------------------------

make_bundle() {  # $@ = extra files already in $W
  tr -d '\r' < ops/deploy-remote.sh > "$W/remote.sh"
  tr -d '\r' < ops/deploytool.py > "$W/deploytool.py"
  # The backup every change takes first, and the unit `check` / `install-unit` compare:
  # always the committed ones, never whatever older copies the server has.
  tr -d '\r' < ops/bin/wrapgto-backup > "$W/wrapgto-backup"
  tr -d '\r' < ops/systemd/wrapgto.service > "$W/wrapgto.service"
  local files=(remote.sh deploytool.py wrapgto-backup wrapgto.service)
  if [ -f requirements/server.txt ]; then tr -d '\r' < requirements/server.txt > "$W/lock.txt"; files+=(lock.txt); fi
  files+=("$@")
  bash -n "$W/remote.sh" || die "ops/deploy-remote.sh has a syntax error"
  bash -n "$W/wrapgto-backup" || die "ops/bin/wrapgto-backup has a syntax error"
  (cd "$W" && tar -czf bundle.tgz "${files[@]}")
  BSUM="$(sha256sum "$W/bundle.tgz" | cut -d' ' -f1)"
}

run_bundle() {  # $@ = NAME=value settings for the server side (values are sanitised)
  local envs="" kv base
  for kv in "$@"; do envs="$envs ${kv%%=*}=$(safe "${kv#*=}")"; done
  base="$(safe "$(dirname "$APP")")"
  if [ "$DRY_RUN" = "1" ]; then
    say "== DRY_RUN — nothing was sent. On $HOST this would upload $(du -h "$W/bundle.tgz" | cut -f1) (sha256 ${BSUM:0:12}) into a"
    say "   root-only folder under $base/deploys and run:  env$envs bash remote.sh   (ops/deploy-remote.sh)"
    return 0
  fi
  # One connection: upload, verify, unpack, run. The bundle arrives on stdin, so the
  # script itself gets /dev/null (a script fed on stdin gets eaten by the first child
  # process that reads stdin). A change hands the folder to its detached run
  # (.detached), which removes it when done; otherwise it goes when this ends.
  remote "set -eu; umask 077; b='$base'; if [ -d \"\$b\" ]; then mkdir -p \"\$b/deploys\"; chmod 700 \"\$b/deploys\"; p=\"\$b/deploys\"; else p=/tmp; fi; d=\$(mktemp -d \"\$p/upload.XXXXXXXX\"); trap '[ -e \"\$d/.detached\" ] || rm -rf \"\$d\"' EXIT; cat > \"\$d/bundle.tgz\"; echo '$BSUM  '\"\$d/bundle.tgz\" | sha256sum -c --quiet || { echo '!! the upload is corrupted (checksum mismatch). Nothing was changed.'; exit 2; }; tar -xzf \"\$d/bundle.tgz\" -C \"\$d\"; rm -f \"\$d/bundle.tgz\"; env$envs bash \"\$d/remote.sh\" </dev/null" < "$W/bundle.tgz"
}

change() {  # run_bundle for a change to the live site: what a dropped connection means
  local rc=0
  run_bundle "$@" || rc=$?
  if [ "$rc" = 255 ]; then
    say "!! the connection to $HOST dropped (or could not be made)."
    say "!! If the change had started, the SERVER carries on with it by itself — see how it ends:"
    say "!!   bash scripts/deploy_prod.sh watch"
  fi
  return "$rc"
}

rerun_hint() {  # $1 = output log, $2 = the command to repeat: the server's exact advice
  local with
  with="$(sed -n 's/^RERUN-WITH //p' "$1" | tail -1 | tr -cd 'A-Z0-9_= ')" || true
  if [ -n "$with" ]; then
    say "== to do that, run exactly:"
    say "   $with bash scripts/deploy_prod.sh $2"
  fi
}

ci_gate() {  # soft: only when the GitHub CLI can tell
  [ "$DRY_RUN" = "1" ] && return 0
  command -v gh >/dev/null 2>&1 || { say "   CI: status unknown (no GitHub CLI) — make sure the tests pass"; return 0; }
  local c
  c="$(gh run list --commit "$COMMIT" --json conclusion -q '.[].conclusion' 2>/dev/null | head -5 | tr '\n' ' ' || true)"
  case "$c" in
    *failure*|*cancelled*|*timed_out*)
      [ "${ALLOW_RED:-0}" = "1" ] || die "CI FAILED on ${COMMIT:0:7} ($c) — fix it first, or ALLOW_RED=1"
      say "   CI failed on this commit — ALLOW_RED=1, going ahead" ;;
    *success*) say "   CI: green on ${COMMIT:0:7}" ;;
    *) say "   CI: no finished run for ${COMMIT:0:7} yet" ;;
  esac
}

capture_lock() {  # $1 = output log -> requirements/server.txt
  awk '/^--- LOCK END/{f=0} f{print} /^--- LOCK BEGIN/{f=1}' "$1" > "$W/server.txt"
  [ -s "$W/server.txt" ] || die "the server sent no package list"
  mkdir -p requirements
  cp "$W/server.txt" requirements/server.txt
  say "== wrote requirements/server.txt ($(grep -c '==' requirements/server.txt) pinned packages) — commit it"
}

local_python() {
  for p in .venv/Scripts/python.exe .venv/bin/python python3 python; do
    if command -v "$p" >/dev/null 2>&1; then printf '%s' "$p"; return 0; fi
  done
  return 1
}

# --- modes ---------------------------------------------------------------------

case "$MODE" in
  -h|--help|help)
    usage; exit 0 ;;

  pack)
    say "== pack (offline): what a deploy of $(git rev-parse --short HEAD) would ship"
    build_archive; scan_archive; archive_summary
    say "== nothing looks like a secret or user data"
    ;;

  check)
    say "== check (read-only) — logging in to $HOST"
    make_bundle
    run_bundle MODE=check APP="$APP" || {
      say "!! could not log in or the check failed. Is this machine's PUBLIC key in the server's"
      say "!! /root/.ssh/authorized_keys, and does ~/.ssh/config define Host wrapgto-prod?"
      say "!! (docs/ops/PRODUCTION.md, 'A new machine')"; exit 1; }
    ;;

  watch)
    make_bundle
    run_bundle MODE=watch APP="$APP"
    ;;

  stage|deploy)
    say "== $MODE: $(git rev-parse --short HEAD) \"$(git log -1 --format=%s | cut -c1-60)\" on $(git rev-parse --abbrev-ref HEAD) -> $HOST:$APP"
    build_archive; scan_archive; archive_summary
    ci_gate
    extra=(app.tgz)
    if [ -n "${ENGINE_WHEEL:-}" ]; then
      [ -f "$ENGINE_WHEEL" ] || die "ENGINE_WHEEL=$ENGINE_WHEEL does not exist"
      cp "$ENGINE_WHEEL" "$W/engine.whl"; extra+=(engine.whl)
    fi
    make_bundle "${extra[@]}"
    if [ "$DRY_RUN" = "1" ]; then dry_plan; fi
    change MODE="$MODE" APP="$APP" COMMIT="$COMMIT" DIRTY="$DIRTY" DIFF_SHA="$DIFF_SHA" \
      BRANCH="$BRANCH" DEPLOYER="$(hostname 2>/dev/null || echo unknown)" SKIP_ENGINE="${SKIP_ENGINE:-0}" \
      FORCE="${FORCE:-0}" ALLOW_NO_CRITIC="${ALLOW_NO_CRITIC:-0}" ENGINE_WHEEL="$(basename "${ENGINE_WHEEL:-}")" \
      ALLOW_ENGINE_MISMATCH="${ALLOW_ENGINE_MISMATCH:-0}"
    if [ "$MODE" = "deploy" ] && [ "$DRY_RUN" != "1" ] && [ "$DIRTY" = "0" ]; then
      tag="prod-$(date -u +%Y%m%d-%H%M)"
      if git tag "$tag" "$COMMIT" 2>/dev/null; then
        say "   tagged $tag (share it: git push origin $tag)"
      fi
    fi
    ;;

  rollback)
    make_bundle
    say "== rollback — what the server has:"
    run_bundle MODE=list APP="$APP"
    if [ -n "$ARG" ]; then say "   target: $ARG"; else say "   target: the newest one above (the version before the live one)"; fi
    confirm ROLLBACK
    change MODE=rollback APP="$APP" TARGET="$ARG" FORCE="${FORCE:-0}" ROLLBACK_ANYWAY="${ROLLBACK_ANYWAY:-0}" \
      ALLOW_NO_CRITIC="${ALLOW_NO_CRITIC:-0}" ALLOW_ENGINE_MISMATCH="${ALLOW_ENGINE_MISMATCH:-0}"
    ;;

  restore-db)
    make_bundle
    if [ -z "$ARG" ]; then
      run_bundle MODE=list APP="$APP"
      say "   restore one with: bash scripts/deploy_prod.sh restore-db NAME"
      exit 0
    fi
    say "== restore-db $ARG: the live database is replaced by that backup (the current one is kept)"
    confirm RESTORE
    change MODE=restore-db APP="$APP" TARGET="$ARG" FORCE="${FORCE:-0}"
    ;;

  promote)
    [ -n "$ARG" ] && [ -f "$ARG" ] || die "usage: bash scripts/deploy_prod.sh promote checkpoints/<file>.pt"
    SHA="$(sha256sum "$ARG" | cut -d' ' -f1)"
    say "== promote $(basename "$ARG") ($(du -h "$ARG" | cut -f1), sha256 ${SHA:0:12}) -> the LIVE site's model"
    if PY="$(local_python)"; then
      "$PY" - "$ARG" <<'PYINFO' 2>/dev/null || true
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if isinstance(ck, dict):
    keys = ("update", "variant", "obs_rev", "obs_mode", "hidden_dim", "num_layers")
    print("   checkpoint:", " · ".join(f"{k} {ck.get(k)}" for k in keys if k in ck),
          "· critic", "yes" if ck.get("critic") else "NO")
PYINFO
    fi
    if [ -n "${OBS_REV:-}" ]; then say "   OBS_REV=$OBS_REV: PLO5BP_OBS_REV on the server changes with the model (and back, if unhealthy)"; fi
    say "   Only with the owner's explicit OK: this is what every player sees next."
    confirm PROMOTE
    cp "$ARG" "$W/candidate.pt"
    make_bundle candidate.pt
    rc=0
    change MODE=promote APP="$APP" CKPT_SHA="$SHA" CKPT_NAME="$(basename "$ARG")" RESTART="${RESTART:-0}" \
      OBS_REV="${OBS_REV:-}" FORCE="${FORCE:-0}" ALLOW_NO_CRITIC="${ALLOW_NO_CRITIC:-0}" | tee "$W/out.log" || rc=$?
    rerun_hint "$W/out.log" "promote $ARG"
    [ "$rc" = 0 ] || exit "$rc"
    if [ "$DRY_RUN" != "1" ] && [ -f docs/models.md ]; then
      how=""
      if grep -q '^MODEL-LOG ' "$W/out.log"; then how="live (restart${OBS_REV:+, obs rev $OBS_REV})"; fi
      if grep -q '^MODEL-STAGED ' "$W/out.log"; then how="staged, then Promote in /admin"; fi
      if [ -n "$how" ]; then
        printf '| %s | %s | `%s` | %s | (why: …) |\n' "$(date -u '+%Y-%m-%d %H:%M')" \
          "$(basename "$ARG")" "${SHA:0:16}" "$how" >> docs/models.md
        say "   added a row to docs/models.md — fill in the 'why' (the h2h evidence) and commit it"
      fi
    fi
    ;;

  promote-undo)
    say "== promote-undo: put the previous model back on the LIVE site (with a restart)"
    confirm UNDO
    make_bundle
    rc=0
    change MODE=promote-undo APP="$APP" OBS_REV="${OBS_REV:-}" FORCE="${FORCE:-0}" \
      ALLOW_NO_CRITIC="${ALLOW_NO_CRITIC:-0}" | tee "$W/out.log" || rc=$?
    rerun_hint "$W/out.log" "promote-undo"
    [ "$rc" = 0 ] || exit "$rc"
    ;;

  recover)
    say "== recover: finish — or undo — a change whose process on the server died half-way"
    say "   (it keeps the change if the site is healthy with it, otherwise puts the previous state back)"
    confirm RECOVER
    make_bundle
    change MODE=recover APP="$APP"
    ;;

  install-unit)
    say "== install-unit: put ops/systemd/wrapgto.service in place of the live service definition"
    say "   (checked first — settings, drop-ins, paths — then guarded, backed up, one restart,"
    say "   health check, and the old definition back if the site is not healthy)"
    confirm INSTALL
    make_bundle
    change MODE=install-unit APP="$APP" FORCE="${FORCE:-0}"
    ;;

  freeze)
    say "== freeze (read-only on the server): the live environment's exact packages"
    make_bundle
    run_bundle MODE=freeze APP="$APP" | tee "$W/out.log"
    [ "$DRY_RUN" = "1" ] || capture_lock "$W/out.log"
    ;;

  lock)
    [ -f requirements/server.in ] || die "requirements/server.in is missing"
    tr -d '\r' < requirements/server.in > "$W/server.in"
    make_bundle server.in
    run_bundle MODE=lock APP="$APP" | tee "$W/out.log"
    [ "$DRY_RUN" = "1" ] || capture_lock "$W/out.log"
    ;;

  *)
    say "unknown mode: $MODE"; usage; exit 2 ;;
esac
say "== done"
