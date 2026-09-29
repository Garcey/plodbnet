#!/usr/bin/env bash
# The server-side half of scripts/deploy_prod.sh. Runs as ROOT on the production
# server, from a root-only folder /opt/wrapgto/deploys/upload.XXXXXXXX that holds the
# uploaded bundle:
#   remote.sh (this file) · deploytool.py · wrapgto-backup · wrapgto.service ·
#   app.tgz (stage/deploy) · lock.txt and server.in (check/lock) · candidate.pt
#   (promote) · engine.whl (optional)
# Never run it by hand: the deploy script uploads it, checks its checksum and
# removes the folder afterwards. tests/python/ops/test_deploy_remote.py runs it
# against a fake server (the WRAPGTO_* settings below exist for that).
#
# Every mode that CHANGES something (stage, deploy, rollback, restore-db, promote,
# promote-undo, recover, install-unit) runs DETACHED from the ssh session: this
# script starts a second copy of itself as a transient systemd service
# (systemd-run; setsid/nohup where there is no systemd) that writes everything to
# /opt/wrapgto/deploys/<time>-<mode>.log, and the ssh session only FOLLOWS that log.
# A closed window, Ctrl+C or a dropped connection can never stop a switch half-way,
# and `bash scripts/deploy_prod.sh watch` shows the log again. The detached copy
# holds a lock (one change at a time) and journals each step of a live change in
# /opt/wrapgto/deploys/state: if its process dies anyway (a crash, a reboot),
# `recover` finishes the change when the site is healthy with it, or puts the
# previous state back; every other change refuses to start until then.
#
# Layout on the server (all under /opt/wrapgto, one filesystem):
#   app/            the LIVE site: code + .venv + data/ + checkpoints/ (systemd's
#                   WorkingDirectory). Release code is root-owned and read-only
#                   to the service account; data/ and checkpoints/ keep their owner.
#   ship-staging/   the next release while it is built and tested
#   releases/<ts>/  the code that was live until <ts> (rollback targets; newest 5,
#                   plus the newest 2 that failed their health check)
#   venvs/<hash>/   Python environments built from requirements/server.txt
#   backups/        database backups (wrapgto-backup) + app-before-*.tgz (old deploys)
#   deploys/        logs of every change, the lock, the journal (root-only)
# A switch is: stop · rename app -> releases/<ts> · rename staging -> app · move
# data/, checkpoints/ (and the .venv, unless the release brought its own) across ·
# start · health check. A failed health check reverses every rename.
set -euo pipefail
umask 077
# The detached copy gets its settings from a file: systemd-run starts it with a
# clean environment.
if [ -n "${WRAPGTO_SETTINGS:-}" ] && [ -r "$WRAPGTO_SETTINGS" ]; then
  # shellcheck source=/dev/null
  . "$WRAPGTO_SETTINGS"
fi

MODE="${MODE:?MODE is required}"
APP="${APP:-/opt/wrapgto/app}"
BASE="$(dirname "$APP")"
STAGE="$BASE/ship-staging"
REL="$BASE/releases"
VENVS="$BASE/venvs"
DEPLOYS="$BASE/deploys"
STATE="$DEPLOYS/state"
KEEP="${WRAPGTO_BACKUPS:-$BASE/backups}"
UNIT="${WRAPGTO_UNIT:-wrapgto}"
UNITFILE="${WRAPGTO_UNITFILE:-/etc/systemd/system/$UNIT.service}"
SVC="${WRAPGTO_SVC_USER:-wrapgto}"
PORT="${WRAPGTO_PORT:-8770}"
ENVFILE="${WRAPGTO_ENVFILE:-/etc/wrapgto/env}"
CARGO_TARGET="${WRAPGTO_CARGO_TARGET:-$BASE/cargo-target}"
HEALTH_TIMEOUT="${WRAPGTO_HEALTH_TIMEOUT:-180}"
KEEP_RELEASES=5
KEEP_FAILED=2
# Top-level entries that belong to the SERVER, never to a release.
CARRY="data checkpoints runs logs screenrecords"

W="$(cd "$(dirname "$0")" && pwd)"
TS="${WRAPGTO_TS:-$(date -u +%Y%m%d-%H%M%S)}"
FORCE="${FORCE:-0}"                      # skip the "is anyone playing?" guard — nothing else
ROLLBACK_ANYWAY="${ROLLBACK_ANYWAY:-0}"  # rollback: switch although the pre-flight failed
ALLOW_ENGINE_MISMATCH="${ALLOW_ENGINE_MISMATCH:-0}"
OBS_REV="${OBS_REV:-}"
TARGET="${TARGET:-}"
COMMIT="${COMMIT:-}"
NC_FLAG=""; [ "${ALLOW_NO_CRITIC:-0}" = "1" ] && NC_FLAG="--allow-no-critic"
LIVEPY="$APP/.venv/bin/python"
# Plain python for the stdlib-only helper calls (health, status, …).
SYSPY="${WRAPGTO_PY:-$(command -v python3 || true)}"
[ -n "$SYSPY" ] || SYSPY="$LIVEPY"
LOG="${WRAPGTO_LOG:-}"
PRE=""
PREFLIGHT_JSON=""
EXTRA_CLEAN=""    # a half-placed file to remove if the script stops early
HOLDING_LOCK=0
FORCED_GUARD=0
ENGINE_SO=""; ENGINE_SOURCE=""; LOCK_SHA=""; TREE_PY=""
DB=""; CKPT=""; ENV_REV=""

say() { printf '%s\n' "$*" 2>/dev/null || true; }   # never the reason a switch stops
die() { local rc=$1; shift; say "!! $*"; exit "$rc"; }
tool() { "$SYSPY" "$W/deploytool.py" "$@"; }
exists() { [ -e "$1" ] || [ -L "$1" ]; }
sha_of() { if [ -f "$1" ]; then sha256sum "$1" | cut -d' ' -f1; fi; }
log_event() { printf '%s %s %s %s\n' "$TS" "$MODE" "${COMMIT:-?}" "$*" >> "$BASE/deploys.log" 2>/dev/null || true; }

on_exit() {
  local rc=$?
  if [ -n "$PRE" ]; then rm -rf "$PRE"; fi
  if [ -n "$EXTRA_CLEAN" ]; then rm -f "$EXTRA_CLEAN"; fi
  if [ "$HOLDING_LOCK" = 1 ]; then
    rm -f "$DEPLOYS/.current"
    flock -u 9 2>/dev/null || true   # free for the next change before the result appears
    exec 9>&-
  fi
  if [ -n "${WRAPGTO_DETACHED:-}" ] && [ -n "$LOG" ]; then
    case "$W" in "$DEPLOYS"/upload.*|/tmp/upload.*) rm -rf "$W" ;; esac   # the upload was handed to this run
    rm -f "$LOG.pid"
    # the result, LAST: the follower (and `watch`) stop when it appears
    { printf '%s\n' "$rc" > "$LOG.rc.tmp" && mv -f "$LOG.rc.tmp" "$LOG.rc"; } 2>/dev/null || true
  fi
  return 0
}
trap on_exit EXIT

# ---------------------------------------------------------------------------
# running detached: launch · follow · lock · journal
# ---------------------------------------------------------------------------

settings_file() {  # what the detached copy needs, shell-quoted (systemd-run passes no environment)
  local v
  for v in MODE APP COMMIT DIRTY DIFF_SHA BRANCH DEPLOYER SKIP_ENGINE ENGINE_WHEEL FORCE ROLLBACK_ANYWAY \
           ALLOW_ENGINE_MISMATCH ALLOW_NO_CRITIC TARGET CKPT_SHA CKPT_NAME RESTART OBS_REV \
           PATH HOME USER LOGNAME LANG LC_ALL TMPDIR $(compgen -v | grep '^WRAPGTO_' || true); do
    case "$v" in WRAPGTO_DETACHED|WRAPGTO_LOG|WRAPGTO_SETTINGS|WRAPGTO_TS) continue ;; esac
    if [ -n "${!v+x}" ]; then printf 'export %s=%q\n' "$v" "${!v}"; fi
  done
}

detach_method() {
  case "${WRAPGTO_DETACH:-auto}" in
    auto)
      if command -v systemd-run >/dev/null 2>&1 && [ -d /run/systemd/system ]; then echo systemd-run
      elif command -v setsid >/dev/null 2>&1; then echo setsid
      else echo nohup; fi ;;
    *) echo "$WRAPGTO_DETACH" ;;
  esac
}

launch_detached() {  # start the real work as its own service, then only follow its log
  local how log
  mkdir -p "$DEPLOYS"; chmod 700 "$DEPLOYS"
  log="$DEPLOYS/$TS-$MODE.log"
  if [ -e "$log" ]; then log="$DEPLOYS/$TS-$MODE-$$.log"; fi
  : > "$log"
  { settings_file; printf 'export WRAPGTO_DETACHED=1 WRAPGTO_LOG=%q WRAPGTO_TS=%q\n' "$log" "$TS"; } > "$W/settings.sh"
  : > "$W/.detached"   # the upload's cleanup now leaves this folder to the detached run
  how="$(detach_method)"
  if [ "$how" = systemd-run ]; then
    if ! systemd-run --unit="wrapgto-$MODE-$TS-$$" --description="WrapGTO $MODE ($TS)" --collect --quiet \
         --setenv=WRAPGTO_SETTINGS="$W/settings.sh" /bin/bash "$W/remote.sh" >"$W/launch.err" 2>&1; then
      say "   (systemd-run failed: $(head -c 300 "$W/launch.err" 2>/dev/null | tr '\n' ' ') — using setsid)"
      how=setsid
    fi
  fi
  if [ "$how" = setsid ] && ! command -v setsid >/dev/null 2>&1; then how=nohup; fi
  case "$how" in
    systemd-run) ;;
    setsid) WRAPGTO_SETTINGS="$W/settings.sh" setsid nohup bash "$W/remote.sh" </dev/null >/dev/null 2>&1 & ;;
    nohup) WRAPGTO_SETTINGS="$W/settings.sh" nohup bash "$W/remote.sh" </dev/null >/dev/null 2>&1 & ;;
    *) rm -f "$W/.detached"; die 2 "unknown WRAPGTO_DETACH=$how" ;;
  esac
  say "== [server] from here the server does this BY ITSELF (log: $log)."
  say "   Closing this window or losing the connection does NOT stop it — it finishes, or puts"
  say "   everything back, on its own. Watch it again any time: bash scripts/deploy_prod.sh watch"
  follow "$log"
}

follow() {  # print $1 as it grows until the run writing it finishes; exit with its code
  local log=$1 off=0 size pid="" rc start=$SECONDS poll="${WRAPGTO_FOLLOW_POLL:-1}"
  while :; do
    size="$(stat -c %s "$log" 2>/dev/null || echo 0)"
    if [ "$size" -gt "$off" ]; then
      tail -c +"$((off + 1))" "$log" 2>/dev/null | head -c "$((size - off))" || true
      off=$size
    fi
    if [ -f "$log.rc" ]; then
      size="$(stat -c %s "$log" 2>/dev/null || echo 0)"
      if [ "$size" -gt "$off" ]; then tail -c +"$((off + 1))" "$log" 2>/dev/null || true; fi
      rc="$(tr -dc '0-9' < "$log.rc" 2>/dev/null || true)"
      exit "${rc:-1}"
    fi
    if [ -z "$pid" ]; then pid="$(tr -dc '0-9' < "$log.pid" 2>/dev/null || true)"; fi
    if [ -n "$pid" ]; then
      if ! kill -0 "$pid" 2>/dev/null; then
        sleep 1
        if [ -f "$log.rc" ]; then continue; fi
        say "!! the process doing this on the server stopped before it finished (killed, or the server"
        say "!! restarted). Run  bash scripts/deploy_prod.sh recover  — it finishes the change if the site"
        say "!! is healthy with it, or puts the previous state back."
        exit 7
      fi
    elif [ $((SECONDS - start)) -gt 60 ]; then
      say "!! it did not start on the server (no $log.pid after 60 s) — see: journalctl -n 50"
      exit 7
    fi
    sleep "$poll"
  done
}

take_lock() {  # one change at a time; the kernel drops the lock with this process
  mkdir -p "$DEPLOYS"; chmod 700 "$DEPLOYS"
  command -v flock >/dev/null 2>&1 \
    || die 2 "flock (util-linux) is missing on the server — cannot take the deploy lock. Nothing was changed."
  exec 9>>"$DEPLOYS/.lock"
  if ! flock -n 9; then
    die 8 "another change is running on the server right now: $(cat "$DEPLOYS/.current" 2>/dev/null || echo '(unknown)'). Nothing was changed. Watch it: bash scripts/deploy_prod.sh watch"
  fi
  HOLDING_LOCK=1
  printf '%s since %s UTC · pid %s · log %s\n' "$MODE" "$TS" "$$" "${LOG:-(none)}" > "$DEPLOYS/.current"
}

# The journal of a live change (kind: switch | model | db | unit), for `recover`.
S_VARS="S_KIND S_MODE S_TS S_PHASE S_NEW S_RETIRED S_WANT S_CARRIED S_CKPT S_OLDSHA S_NEWSHA S_NEWREV S_ENV_BACKUP S_DB S_REPLACED S_TARGET S_UNIT_BACKUP"
for _v in $S_VARS; do printf -v "$_v" '%s' ""; done
state_save() {
  local v
  { for v in $S_VARS; do printf '%s=%q\n' "$v" "${!v:-}"; done; } > "$STATE.tmp" && mv -f "$STATE.tmp" "$STATE"
}
state_begin() {  # $1 = kind, $2 = first phase: before anything live changes, so it must be written
  S_KIND=$1; S_MODE=$MODE; S_TS=$TS; S_PHASE=$2
  state_save || die 2 "cannot write $STATE (the journal of this change — is the disk full?). Nothing was changed."
}
phase() { S_PHASE=$1; state_save || say "!! could not update $STATE"; }
state_clear() { rm -f "$STATE" "$STATE.tmp"; }
state_summary() { ( . "$STATE" 2>/dev/null; printf '%s of %s UTC stopped at "%s"' "${S_MODE:-?}" "${S_TS:-?}" "${S_PHASE:-?}" ) || true; }
refuse_if_interrupted() {
  [ -f "$STATE" ] || return 0
  say "!! a previous change was interrupted half-way: $(state_summary)"
  die 9 "Nothing was changed. First run: bash scripts/deploy_prod.sh recover — it finishes that change if the site is healthy with it, or puts the previous state back."
}

begin_run() {  # the detached copy (or WRAPGTO_DETACH=none): log, lock, housekeeping
  trap '' HUP PIPE   # nothing may stop a switch half-way
  if [ -n "${WRAPGTO_DETACHED:-}" ] && [ -n "$LOG" ]; then
    exec >>"$LOG" 2>&1 </dev/null
    echo "$$" > "$LOG.pid"
  fi
  say "== [server] $MODE · started $(date -u '+%Y-%m-%d %H:%M:%S') UTC on $(hostname)"
  take_lock
  prune_deploys
}

prune_deploys() {  # old logs, uploads a crash left behind
  find "$DEPLOYS" -maxdepth 1 -type f \( -name '*.log' -o -name '*.log.rc' -o -name '*.log.pid' \) \
    -mtime +120 -delete 2>/dev/null || true
  find "$DEPLOYS" -maxdepth 1 -mindepth 1 -type d -name 'upload.*' -mtime +2 ! -path "$W" \
    -exec rm -rf {} + 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# A folder the service account can use, holding a copy of the helper (the bundle
# folder itself is root-only).
make_pre() {
  PRE="$(mktemp -d "${TMPDIR:-/tmp}/wrapgto-preflight.XXXXXXXX")"
  cp "$W/deploytool.py" "$PRE/"
  chmod 700 "$PRE"; chmod 644 "$PRE/deploytool.py"   # the service's own: it holds the DB copy
  chown "$SVC:$SVC" "$PRE"
}
as_svc() { sudo -u "$SVC" -H "$@"; }

load_env() {  # -> $W/env.json, DB, CKPT, ENV_REV: what the service REALLY runs with (the
              # unit's Environment= and every EnvironmentFile=, not only $ENVFILE)
  systemctl show "$UNIT" -p Environment -p EnvironmentFiles > "$W/unit-env.txt" 2>/dev/null || : > "$W/unit-env.txt"
  tool envfile "$ENVFILE" --unit-show "$W/unit-env.txt" > "$W/env.json" \
    || die 2 "could not read the service's settings ($ENVFILE). Nothing was changed."
  tool paths --env-json "$W/env.json" --app "$APP" --key db,checkpoint,obs_rev > "$W/paths.txt" \
    || die 2 "could not work out the app's files from its settings. Nothing was changed."
  { IFS= read -r DB; IFS= read -r CKPT; IFS= read -r ENV_REV; } < "$W/paths.txt" || true
}

refuse_build_commit_override() {  # /health must report the commit BUILD_INFO.json names
  local v
  v="$(tool jsonget PLO5BP_BUILD_COMMIT < "$W/env.json" 2>/dev/null || true)"
  [ -z "$v" ] || die 2 "PLO5BP_BUILD_COMMIT is set in the service's settings: /health reports it instead of the deployed commit, so no deploy can confirm the new code is the one answering. Remove that line from $ENVFILE. Nothing was changed."
}

commit_of() {  # the commit recorded in a tree's BUILD_INFO.json ("" when unknown)
  [ -f "$1/BUILD_INFO.json" ] || return 0
  "$SYSPY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("commit") or "")' "$1/BUILD_INFO.json" 2>/dev/null || true
}

describe_tree() {  # one line: what code a tree holds
  if [ -f "$1/BUILD_INFO.json" ]; then
    "$SYSPY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("commit %s%s · deployed %s by %s" % (d.get("short") or "?", " (+uncommitted changes)" if d.get("dirty") else "", d.get("deployed_at") or "?", d.get("deployed_by") or "?"))' "$1/BUILD_INFO.json" 2>/dev/null || echo "unreadable BUILD_INFO.json"
  else
    echo "deployed before BUILD_INFO.json existed"
  fi
}

db_mb() {  # the database (+ its -wal) in MB, rounded up
  local b=0 f s
  for f in "$DB" "$DB-wal"; do s="$(stat -c %s "$f" 2>/dev/null || echo 0)"; b=$((b + s)); done
  echo $(( (b + 1048575) / 1048576 ))
}

need_space() {  # $1 = folder (or where it will be), $2 = MB needed, $3 = what for
  local d=$1 free=""
  while [ ! -d "$d" ] && [ "$d" != / ] && [ "$d" != . ]; do d="$(dirname "$d")"; done
  free="$(df -Pk "$d" 2>/dev/null | awk 'NR==2 {print int($4 / 1024)}')" || true
  [ -n "$free" ] || return 0
  [ "$free" -ge "$2" ] || die 2 "not enough free disk space for $3: $free MB free in $d, about $2 MB needed. Nothing was changed. (bash scripts/deploy_prod.sh check shows the disk; old releases in $REL and backups in $KEEP can be trimmed.)"
}

status_guard() {  # stop unless no hand would be interrupted (FORCE=1 skips THIS check only).
                  # $1 = quiet: the last look right before the stop, silent unless it trips
  local rc=0 running=0 q=""
  if [ "${1:-}" = quiet ]; then q=--quiet; else say "== [server] home games: is anyone playing?"; fi
  if [ ! -e "$DB" ]; then [ -n "$q" ] || say "   (no database yet)"; return 0; fi
  [ -n "$PRE" ] || make_pre
  if systemctl is-active --quiet "$UNIT"; then running=1; fi
  as_svc "$SYSPY" "$PRE/deploytool.py" status --db "$DB" --guard $q \
    --health-url "http://127.0.0.1:$PORT/health?deploy=1" --app-running "$running" || rc=$?
  [ "$rc" = 0 ] && return 0
  if [ "$rc" = 3 ] && [ "$FORCE" = "1" ]; then
    [ "$FORCED_GUARD" = 1 ] || say "   FORCE=1 — going ahead anyway: those hands are void (everyone keeps the chips they had before them), and running tables come back PAUSED until their host presses Start"
    FORCED_GUARD=1
    return 0
  fi
  [ "$rc" = 3 ] && die 6 "Nothing was changed. Try again when those games are over (a host can pause a table from its menu), or run with FORCE=1 to cut them short."
  die 6 "could not read the home-games state (exit $rc). Nothing was changed. FORCE=1 skips this check."
}

backup_now() {  # a fresh local database backup by the SHIPPED ops/bin/wrapgto-backup, verified
  local mb
  say "== [server] database backup (ops/bin/wrapgto-backup from this upload, local copy)"
  if [ ! -e "$DB" ]; then say "   (no database yet — nothing to back up)"; return 0; fi
  [ -f "$W/wrapgto-backup" ] || die 2 "the upload has no wrapgto-backup script. Nothing was changed."
  mb="$(db_mb)"
  need_space "$KEEP" $((mb / 2 + 100)) "the database backup"
  need_space "${TMPDIR:-/tmp}" $((mb + 100)) "the backup's temporary copy of the database"
  # everything the job writes from now on is newer than this (a second back: mtimes are
  # compared whole-second on some filesystems)
  touch -d "@$(( $(date +%s) - 1 ))" "$W/backup-marker"
  WRAPGTO_BACKUP_LOCAL_ONLY=1 WRAPGTO_APP="$APP" WRAPGTO_BACKUPS="$KEEP" WRAPGTO_ENVFILE="$ENVFILE" \
    WRAPGTO_DB="$DB" WRAPGTO_PY="$SYSPY" bash "$W/wrapgto-backup" </dev/null >"$W/backup.log" 2>&1 \
    || { tail -n 5 "$W/backup.log" | sed 's/^/   | /'; die 2 "the database backup failed — stopping. Nothing was changed."; }
  sed 's/^/   /' "$W/backup.log" || true
  tool backup-check --dir "$KEEP" --newer-than "$W/backup-marker" --max-age-hours 1 \
    || die 2 "the backup did not leave a fresh, readable backup — stopping. Nothing was changed."
}

healthy() {  # $1 = expected commit ("" = none recorded), [$2 = expected obs rev]. The app
             # loads torch + its models before it listens; progress lines every 20 s.
  local want="${1:-}" rev="${2:-}" rc=0
  local args=(health --url "http://127.0.0.1:$PORT/health" --timeout "$HEALTH_TIMEOUT" --interval 3 --progress 20)
  if [ -n "$want" ]; then args+=(--expect-commit "$want"); fi
  if [ -n "$rev" ]; then args+=(--expect-obs-rev "$rev"); fi
  if [ -n "$NC_FLAG" ]; then args+=("$NC_FLAG"); fi
  tool "${args[@]}" || rc=$?
  [ "$rc" = 0 ] && systemctl is-active --quiet "$UNIT"
}

stop_app() {
  local t
  t="$(systemctl show -p TimeoutStopUSec --value "$UNIT" 2>/dev/null || true)"
  say "   stopping the app (open live connections get up to ${t:-30s} to close)…"
  systemctl stop "$UNIT" || true
}
start_app() {
  say "   starting the app — it loads torch and the models before it answers (up to $HEALTH_TIMEOUT s)…"
  systemctl start "$UNIT" || true
}
journal_tail() { journalctl -u "$UNIT" -n 25 --no-pager 2>/dev/null | sed 's/^/   | /' || true; }

same_fs() {  # releases/ must be on the app's filesystem: the switch renames, never copies
  # (This script runs under umask 077: open up what the service account must read —
  # a rollback target is pre-flighted, and later served, from releases/.)
  mkdir -p "$REL"; chmod 755 "$REL"
  [ "$(stat -c %d "$APP")" = "$(stat -c %d "$REL")" ] \
    || die 2 "$REL is on a different filesystem than $APP — a switch would copy instead of rename. Nothing was changed."
}

preflight() {  # $1 = code root, $2 = python, then deploytool preflight options
  local root=$1 py=$2 rc=0
  shift 2
  say "== [server] pre-flight: does this code start with the live model and data? (nothing live is touched)"
  [ -n "$PRE" ] || make_pre
  local args=(preflight --root "$root" --work "$PRE" --env-json - "$@")
  if [ -n "$NC_FLAG" ]; then args+=("$NC_FLAG"); fi
  if [ "$ALLOW_ENGINE_MISMATCH" = "1" ]; then args+=(--allow-engine-mismatch); fi
  ( cd "$APP" && timeout "${WRAPGTO_PREFLIGHT_TIMEOUT:-600}" sudo -u "$SVC" -H env PYTHONPATH="$root/python" \
      "$py" "$PRE/deploytool.py" "${args[@]}" < "$W/env.json" ) > "$W/preflight.log" 2>&1 || rc=$?
  if [ "$rc" = 124 ]; then echo "!! the pre-flight did not finish within ${WRAPGTO_PREFLIGHT_TIMEOUT:-600} s" >> "$W/preflight.log"; fi
  PREFLIGHT_JSON="$(grep '^PREFLIGHT ' "$W/preflight.log" | tail -1 | cut -d' ' -f2- || true)"
  if [ "$rc" = 0 ]; then
    grep -E '^(   |!! )' "$W/preflight.log" || true
  else
    grep -v '^PREFLIGHT ' "$W/preflight.log" | tail -n 30 | sed 's/^/   | /' || true
  fi
  return "$rc"
}

pf_get() { printf '%s' "$PREFLIGHT_JSON" | tool jsonget "$1" 2>/dev/null || true; }

prepare_tree() {  # $1 = tree, $2 = python: precompile, then root-owned + read-only to the service
  "$2" -m compileall -q "$1/python" >/dev/null 2>&1 || true
  chown -R root:root "$1"
  chmod -R u=rwX,go=rX "$1"
}

check_carry_free() {  # $1 = a tree about to go live: nothing in it may belong to the server
  local d bad=""
  for d in $CARRY; do if exists "$1/$d"; then bad="$bad $d/"; fi; done
  [ -z "$bad" ] || die 5 "$1 contains$bad, which belongs to the server — refusing before anything was stopped. Nothing was changed."
}

rust_toolchain() {  # $1 = tree. Installs the compiler the release pins (rust-toolchain.toml,
                    # ENG-008) up front: some rustup releases do not fetch it on first use, and
                    # the engine must never be quietly built by another compiler.
  local chan=""
  if [ -f "$1/rust-toolchain.toml" ]; then
    chan="$(sed -n 's/^channel *= *"\([^"]*\)".*/\1/p' "$1/rust-toolchain.toml" | head -n 1)" || chan=""
  fi
  if [ -z "$chan" ]; then
    say "   rust toolchain: this release pins none (no rust-toolchain.toml) — root's default compiler"
    return 0
  fi
  case "$chan" in *[!0-9A-Za-z._-]*) die 3 "rust-toolchain.toml names an odd toolchain ($chan). Nothing was changed." ;; esac
  if bash -lc 'command -v rustup >/dev/null' </dev/null; then
    say "   rust toolchain: $chan (rust-toolchain.toml)"
    bash -lc "rustup toolchain install '$chan' --profile minimal -c rustfmt -c clippy --no-self-update" </dev/null \
      > "$W/build.log" 2>&1 || { tail -n 20 "$W/build.log" | sed 's/^/   | /'; die 3 "could not install the pinned Rust toolchain $chan. Nothing was changed."; }
  else
    say "   !! root has cargo but no rustup: building with $(bash -lc 'rustc --version' </dev/null 2>/dev/null || echo 'an unknown rustc'), not the pinned $chan"
  fi
}

engine_into() {  # $1 = tree, $2 = python for the build. Sets ENGINE_SO, ENGINE_SOURCE.
  local tree=$1 py=$2 so maturin whl
  rm -f "$tree"/python/plo5bp/_engine*.so
  if [ -n "${ENGINE_WHEEL:-}" ]; then
    say "== [server] engine from the uploaded wheel ($ENGINE_WHEEL)"
    rm -rf "$W/whl-x"; mkdir -p "$W/whl-x"
    "$py" -m zipfile -e "$W/engine.whl" "$W/whl-x"
    so="$(find "$W/whl-x" -name '_engine*.so' | head -1)" || true
    [ -n "$so" ] || die 3 "the wheel has no engine binary. Nothing was changed."
    cp "$so" "$tree/python/plo5bp/"; ENGINE_SOURCE="wheel:$ENGINE_WHEEL"
  elif [ "${SKIP_ENGINE:-0}" = "1" ]; then
    say "== [server] reusing the live engine (the pre-flight checks it was built from this release's Rust sources)"
    cp -p "$APP"/python/plo5bp/_engine*.so "$tree/python/plo5bp/" || die 3 "there is no live engine to reuse. Nothing was changed."
    ENGINE_SOURCE="reused"
  else
    say "== [server] building the Rust engine in staging (a few minutes the first time)"
    bash -lc 'command -v cargo >/dev/null' </dev/null || die 3 "root has no cargo — cannot build (ENGINE_WHEEL=… ships a prebuilt one). Nothing was changed."
    maturin="$(dirname "$py")/maturin"
    [ -x "$maturin" ] || maturin="$APP/.venv/bin/maturin"
    [ -x "$maturin" ] || die 3 "no maturin in the server's environment. Nothing was changed."
    need_space "$CARGO_TARGET" 3000 "building the engine (the pinned Rust toolchain + an optimised build)"
    rm -rf "$W/wheels" "$W/whl-x"; mkdir -p "$W/wheels" "$W/whl-x"
    : > "$W/build.log"
    rust_toolchain "$tree"
    # --locked: exactly the crate versions in Cargo.lock, or stop (never a silent re-resolve).
    ( cd "$tree" && bash -lc "CARGO_TARGET_DIR='$CARGO_TARGET' '$maturin' build --release --locked --compatibility linux -i '$py' -o '$W/wheels'" </dev/null ) \
      >> "$W/build.log" 2>&1 || { tail -n 20 "$W/build.log" | sed 's/^/   | /'; die 3 "the engine build failed. Nothing was changed."; }
    whl="$(ls -t "$W"/wheels/*.whl 2>/dev/null | head -1)" || true
    [ -n "$whl" ] || die 3 "the build produced no wheel (the log above). Nothing was changed."
    "$py" -m zipfile -e "$whl" "$W/whl-x"
    so="$(find "$W/whl-x" -name '_engine*.so' | head -1)" || true
    [ -n "$so" ] || die 3 "the build produced no engine binary. Nothing was changed."
    cp "$so" "$tree/python/plo5bp/"; ENGINE_SOURCE="built"
  fi
  ENGINE_SO="$(ls "$tree"/python/plo5bp/_engine*.so | head -1)"
  say "   engine: $(basename "$ENGINE_SO") ($ENGINE_SOURCE)"
}

venv_for() {  # $1 = tree. Sets TREE_PY, LOCK_SHA. Reuses the live environment unless the
              # release's requirements/server.txt asks for packages it lacks — then builds
              # venvs/<hash> where it will stay (a venv cannot be moved) and links it in.
  local tree=$1 lock="$1/requirements/server.txt" v purelib basepy
  TREE_PY="$LIVEPY"; LOCK_SHA=""
  if [ ! -f "$lock" ]; then
    say "   python packages: this release has no requirements/server.txt — reusing the live environment"
    return 0
  fi
  LOCK_SHA="$(sha256sum "$lock" | cut -d' ' -f1)"
  "$LIVEPY" -m pip freeze > "$W/freeze.txt" 2>/dev/null \
    || die 3 "could not list the live environment's packages (pip freeze). Nothing was changed."
  if tool deps --lock "$lock" < "$W/freeze.txt" > "$W/deps.txt"; then
    say "   python packages: the live environment matches requirements/server.txt"
    return 0
  fi
  say "== [server] this release needs different Python packages:"
  sed 's/^/ /' "$W/deps.txt"
  v="$VENVS/${LOCK_SHA:0:12}"
  if [ ! -f "$v/.complete" ]; then
    need_space "$VENVS" 3000 "a new Python environment"
    say "   building a new environment in $v (the live one is not touched)"
    rm -rf "${v:?}"; mkdir -p "$VENVS"; chmod 755 "$VENVS"
    basepy="$("$LIVEPY" -c 'import sys; print(getattr(sys, "_base_executable", sys.executable))')"
    "$basepy" -m venv "$v" || die 3 "could not create $v. Nothing was changed."
    "$v/bin/python" -m pip install -q --no-input --disable-pip-version-check -r "$lock" > "$W/pip.log" 2>&1 \
      || { tail -n 20 "$W/pip.log" | sed 's/^/   | /'; die 3 "installing the new Python packages failed. Nothing was changed."; }
    purelib="$("$v/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    echo "$APP/python" > "$purelib/wrapgto-app.pth"   # what the editable install did in the old one
    touch "$v/.complete"
  fi
  chown -R root:root "$v"; chmod -R u=rwX,go=rX "$v"   # read-only to the service
  ln -s "$v" "$tree/.venv"
  TREE_PY="$tree/.venv/bin/python"
}

write_build_info() {  # $1 = tree: the commit /health must report + what the engine was built from
  tool deployinfo --out "$1/BUILD_INFO.json" --commit "${COMMIT:-unknown}" --dirty "${DIRTY:-0}" \
    --diff-sha256 "${DIFF_SHA:-}" --branch "${BRANCH:-}" --deployed-by "${DEPLOYER:-}" \
    --engine-so "$ENGINE_SO" --engine-source "$ENGINE_SOURCE" --lock-sha256 "$LOCK_SHA" \
    --preflight-json "$PREFLIGHT_JSON"
  chown root:root "$1/BUILD_INFO.json"; chmod 644 "$1/BUILD_INFO.json"   # the service reads it
}

after_switch_notes() {
  say "   (a shell that was open in $APP on the server now sits in the retired copy: cd $APP again)"
  if [ "$FORCED_GUARD" = 1 ]; then
    say "   home-game tables that were playing come back PAUSED: each host presses Start"
  fi
}

# ---------------------------------------------------------------------------
# the switch
# ---------------------------------------------------------------------------
RETIRED=""

undo_switch() {  # put the previous tree back exactly as it was. Moves back every server-owned
                 # entry the new tree holds (never trusts a list a crash could have cut short);
                 # if one cannot move, it stops with the site DOWN rather than start the old
                 # code without its data (an empty database would be created).
  local d failed=""
  phase undoing
  stop_app
  for d in $CARRY .venv; do
    if exists "$APP/$d" && ! exists "$RETIRED/$d"; then
      mv -T "$APP/$d" "$RETIRED/$d" || failed="$failed $d"
    fi
  done
  if [ -n "$failed" ]; then
    say "!! COULD NOT MOVE$failed BACK into the previous version ($RETIRED)."
    say "!! Stopped with the site DOWN — starting it without its data would be worse. Fix it by hand:"
    for d in $failed; do say "     mv -T $APP/$d $RETIRED/$d"; done
    say "     mv -T $APP $REL/$TS-failed && mv -T $RETIRED $APP && systemctl start $UNIT"
    say "!! or run: bash scripts/deploy_prod.sh recover (it retries these moves)"
    log_event "undo-stuck: could not move$failed back"
    exit 7
  fi
  phase undo-moving
  mv -T "$APP" "$REL/$TS-failed" \
    || die 7 "could not move the failed version aside. By hand: mv -T $APP $REL/$TS-failed && mv -T $RETIRED $APP && systemctl start $UNIT"
  mv -T "$RETIRED" "$APP" \
    || die 7 "COULD NOT RESTORE $RETIRED -> $APP. By hand: mv -T $RETIRED $APP && systemctl start $UNIT"
  phase undo-start
  start_app
  if healthy ""; then
    say "!! rolled back: the PREVIOUS version is live and healthy (the failed one is in $REL/$TS-failed)"
  else
    say "!! STILL unhealthy after the rollback — look at: journalctl -u $UNIT -n 80"
  fi
  state_clear
}

switch_to() {  # $1 = prepared tree, $2 = expected commit of it (optional). Every check runs
               # BEFORE the site stops; after that each step is journaled for `recover`.
  local new=$1 want="${2:-}" d
  RETIRED="$REL/$TS"
  ! exists "$RETIRED" || die 5 "$RETIRED already exists. Nothing was changed."
  check_carry_free "$new"
  status_guard quiet   # the last look: seconds before the stop
  S_NEW=$new; S_RETIRED=$RETIRED; S_WANT=$want; S_CARRIED=""
  state_begin switch stopping
  say "== [server] switching the live site over"
  stop_app
  phase moving-live
  if ! mv -T "$APP" "$RETIRED"; then
    start_app; state_clear
    die 5 "could not move the live code aside. Nothing was changed (the app is starting again)."
  fi
  phase moving-new
  if ! mv -T "$new" "$APP"; then
    if mv -T "$RETIRED" "$APP"; then
      start_app; state_clear
      die 5 "could not move the new code into place — the old code is back, unchanged."
    fi
    die 7 "could not move the new code into place, NOR the live code back: it is in $RETIRED. Run: bash scripts/deploy_prod.sh recover"
  fi
  phase carrying
  for d in $CARRY .venv; do
    exists "$RETIRED/$d" || continue
    if exists "$APP/$d"; then continue; fi   # only .venv: the release brought its own environment
    if ! mv -T "$RETIRED/$d" "$APP/$d"; then
      say "!! could not move $d/ across — rolling back"
      undo_switch; return 5
    fi
    S_CARRIED="$S_CARRIED $d"; phase carrying
  done
  phase starting
  start_app
  phase health
  if healthy "$want"; then
    phase done
    say "== [server] LIVE and healthy"
    state_clear
    return 0
  fi
  say "!! the app did not come up healthy — rolling back"
  journal_tail
  undo_switch
  return 5
}

prune_releases() {  # the newest KEEP_RELEASES releases, and the newest KEEP_FAILED failed ones
  local n=0 f=0 d
  for d in $(ls -1t "$REL" 2>/dev/null); do
    case "$d" in
      *-failed) f=$((f + 1)); if [ "$f" -gt "$KEEP_FAILED" ]; then rm -rf "${REL:?}/$d"; fi ;;
      *) n=$((n + 1)); if [ "$n" -gt "$KEEP_RELEASES" ]; then rm -rf "${REL:?}/$d"; fi ;;
    esac
  done
  return 0
}

# ---------------------------------------------------------------------------
# models (promote / promote-undo): the checkpoint and, for an obs-revision change,
# PLO5BP_OBS_REV in the env file — written together, reverted together
# ---------------------------------------------------------------------------

env_revert() {
  if [ -n "${S_ENV_BACKUP:-}" ] && [ -f "$S_ENV_BACKUP" ]; then
    if cp -p "$S_ENV_BACKUP" "$ENVFILE.wrapgto-old" && mv -f "$ENVFILE.wrapgto-old" "$ENVFILE"; then
      rm -f "$S_ENV_BACKUP"
      say "   $ENVFILE is back as it was"
    else
      say "!! could not put $ENVFILE back — the previous one is $S_ENV_BACKUP: cp -p $S_ENV_BACKUP $ENVFILE"
    fi
  fi
}

set_env_rev() {  # $1 = the new PLO5BP_OBS_REV: rewritten in $ENVFILE (backup kept), verified
  local rev=$1 tmp="$ENVFILE.wrapgto-new"
  [ -f "$ENVFILE" ] || die 2 "$ENVFILE does not exist — cannot change PLO5BP_OBS_REV. Nothing was changed."
  S_ENV_BACKUP="$ENVFILE.before-$TS"
  cp -p "$ENVFILE" "$S_ENV_BACKUP" || die 2 "could not back up $ENVFILE. Nothing was changed."
  phase env
  if ! tool envset --file "$ENVFILE" --set "PLO5BP_OBS_REV=$rev" --out "$tmp"; then
    rm -f "$tmp"; rm -f "$S_ENV_BACKUP"; state_clear
    die 2 "could not rewrite $ENVFILE. Nothing was changed."
  fi
  chown --reference="$ENVFILE" "$tmp"; chmod --reference="$ENVFILE" "$tmp"
  mv -f "$tmp" "$ENVFILE"
  load_env   # what the SERVICE will get: another source could still override the file
  if [ "$ENV_REV" != "$rev" ]; then
    env_revert; state_clear
    die 2 "PLO5BP_OBS_REV is also set somewhere the service reads after $ENVFILE (bash scripts/deploy_prod.sh check lists where its settings come from) — change it there. Nothing was changed."
  fi
  say "   $ENVFILE: PLO5BP_OBS_REV=$rev (the previous file is kept until the site is healthy)"
}

model_revert() {  # put the model that was live (and the old settings) back, restart, check
  local cur
  phase reverting
  say "!! the app did not come up healthy with that model — putting the previous one back"
  journal_tail
  cur="$(sha_of "$S_CKPT")"
  if [ -n "$S_OLDSHA" ]; then
    if [ "$(sha_of "$S_CKPT.prev")" != "$S_OLDSHA" ]; then
      die 7 "$S_CKPT.prev is not the model that was live (its sha256 differs) — not touching anything else. The model that was live had sha256 ${S_OLDSHA:0:16}; journalctl -u $UNIT -n 80"
    fi
    cp -p "$S_CKPT.prev" "$S_CKPT.old.tmp" \
      || die 7 "could not copy $S_CKPT.prev back (disk full?). By hand: cp -p $S_CKPT.prev $S_CKPT && systemctl restart $UNIT"
    if [ "$S_MODE" = promote-undo ] && [ "$cur" = "$S_NEWSHA" ]; then cp -p "$S_CKPT" "$S_CKPT.rejected.tmp" || true; fi
    mv -f "$S_CKPT.old.tmp" "$S_CKPT" \
      || die 7 "could not put the previous model back. By hand: cp -p $S_CKPT.prev $S_CKPT && systemctl restart $UNIT"
    # the undo's target goes back to being the previous model, exactly as before
    if [ -f "$S_CKPT.rejected.tmp" ]; then mv -f "$S_CKPT.rejected.tmp" "$S_CKPT.prev"; fi
  else
    rm -f "$S_CKPT"   # there was no model before this one
  fi
  env_revert
  stop_app; start_app
  if healthy ""; then say "!! the previous model is back and the site is healthy"; else say "!! STILL unhealthy — journalctl -u $UNIT -n 80"; fi
  state_clear
  log_event "$S_MODE-failed-reverted sha256=$S_NEWSHA"
}

# ---------------------------------------------------------------------------
# restore-db: move the database aside (companions FIRST, the .db last), put the
# backup in place, check; a failed check puts the original set back
# ---------------------------------------------------------------------------

db_revert() {
  local f base
  phase reverting
  say "!! putting the previous database back"
  stop_app
  base="$(basename "$S_DB")"
  if [ -e "$S_REPLACED/raw/$base" ]; then
    # the original set is complete in raw/ (the .db moves last): everything at the live
    # path belongs to the restored copy — a -wal of it must never meet the original
    rm -f "$S_DB" "$S_DB-wal" "$S_DB-shm" "$S_DB-journal" "$S_DB.restore.tmp"
  fi
  for f in "$S_REPLACED/raw"/*; do
    if [ -e "$f" ]; then mv "$f" "$(dirname "$S_DB")/" || say "!! could not move $f back"; fi
  done
  start_app
  if healthy ""; then say "!! the previous database is back and the site is healthy"; else say "!! STILL unhealthy — journalctl -u $UNIT -n 80"; fi
  state_clear
  log_event "restore-db-failed ${S_TARGET:-?}"
}

prune_replaced() {  # restore-db's replaced-<time> folders: the newest 3
  local n=0 d
  for d in $(ls -1td "$KEEP"/replaced-* 2>/dev/null); do
    n=$((n + 1))
    if [ "$n" -gt 3 ]; then rm -rf "${d:?}"; fi
  done
  return 0
}

unit_revert() {
  phase reverting
  say "!! putting the previous service definition back"
  if [ -n "${S_UNIT_BACKUP:-}" ] && [ -f "$S_UNIT_BACKUP" ]; then
    cp -p "$S_UNIT_BACKUP" "$UNITFILE.wrapgto-old" && mv -f "$UNITFILE.wrapgto-old" "$UNITFILE"
  else
    rm -f "$UNITFILE"
  fi
  systemctl daemon-reload || true
  stop_app; start_app
  if healthy ""; then say "!! the previous service definition is back and the site is healthy"; else say "!! STILL unhealthy — journalctl -u $UNIT -n 80"; fi
  state_clear
  log_event "install-unit-failed-reverted"
}

unit_snapshot() {  # -> $W/unit-show.txt, $W/unit-cat.txt (what the live unit is)
  systemctl show "$UNIT" -p FragmentPath -p DropInPaths -p ExecStart -p Environment -p EnvironmentFiles \
    -p User -p WorkingDirectory > "$W/unit-show.txt" 2>/dev/null || : > "$W/unit-show.txt"
  systemctl cat "$UNIT" > "$W/unit-cat.txt" 2>/dev/null || : > "$W/unit-cat.txt"
}

# ---------------------------------------------------------------------------
# recover: a change whose process died half-way
# ---------------------------------------------------------------------------

recover_switch() {
  local d
  TS="$S_TS"; RETIRED="$S_RETIRED"
  if [ "${S_PHASE:-}" = done ]; then
    say "   it had finished and was healthy — only the bookkeeping was missing"
    state_clear; log_event "recovered: $S_MODE had finished"; return 0
  fi
  if ! exists "$APP"; then
    exists "$RETIRED" || die 7 "$APP is missing and so is $RETIRED — look in $REL by hand (ls -lt $REL)"
    say "   the live code had been moved aside and nothing is in its place — putting it back"
    mv -T "$RETIRED" "$APP" || die 7 "could not move $RETIRED back. By hand: mv -T $RETIRED $APP && systemctl start $UNIT"
    start_app
    if healthy ""; then say "== [server] recovered: the previous version is live and healthy"
    else say "!! the previous version does not come up healthy — journalctl -u $UNIT -n 80"; fi
    state_clear; log_event "recovered: previous version back"; return 0
  fi
  if ! exists "$RETIRED"; then
    say "   nothing had been moved (or the rollback had finished) — making sure the app runs"
    systemctl is-active --quiet "$UNIT" || start_app
    if healthy ""; then say "== [server] the site is live and healthy"; else say "!! not healthy — journalctl -u $UNIT -n 80"; fi
    state_clear; log_event "recovered: nothing had moved"; return 0
  fi
  if exists "$S_NEW"; then
    die 7 "$APP, $RETIRED and $S_NEW all exist — not guessing. Look at them by hand (BUILD_INFO.json in each says what it is)."
  fi
  case "${S_PHASE:-}" in
    undo*)
      say "   it was rolling back — finishing that"
      undo_switch; log_event "recovered: rolled back"; return 0 ;;
  esac
  say "   the new code is in place — finishing the switch; the health check decides"
  for d in $CARRY .venv; do
    exists "$RETIRED/$d" || continue
    if exists "$APP/$d"; then continue; fi
    mv -T "$RETIRED/$d" "$APP/$d" || { say "!! could not move $d/ across — rolling back"; undo_switch; exit 5; }
  done
  phase starting
  systemctl stop "$UNIT" 2>/dev/null || true   # a clean start: every server folder is in place now
  start_app
  phase health
  if healthy "$S_WANT"; then
    say "== [server] recovered: the new version is LIVE and healthy"
    state_clear; prune_releases; log_event "recovered: $S_MODE finished ok"
  else
    say "!! not healthy — rolling back"; journal_tail
    undo_switch; log_event "recovered: $S_MODE failed, rolled back"; exit 5
  fi
}

recover_model() {
  local cur
  TS="$S_TS"
  cur="$(sha_of "$S_CKPT")"
  if [ "$cur" = "$S_OLDSHA" ] && [ -f "$S_CKPT.new.tmp" ] && [ "$(sha_of "$S_CKPT.new.tmp")" = "$S_NEWSHA" ] \
     && [ "${S_PHASE:-}" != prepared ] && [ "${S_PHASE:-}" != reverting ]; then
    say "   the checked model was not swapped in yet — finishing that"
    cp -p "$S_CKPT" "$S_CKPT.prev.tmp" && mv -f "$S_CKPT.prev.tmp" "$S_CKPT.prev"
    mv -f "$S_CKPT.new.tmp" "$S_CKPT"; cur="$S_NEWSHA"
  fi
  rm -f "$S_CKPT.new.tmp" "$S_CKPT.prev.tmp" "$S_CKPT.old.tmp" "$S_CKPT.rejected.tmp"
  if [ "$cur" = "$S_OLDSHA" ] && [ -n "$cur" ]; then
    say "   the live model was never swapped"
    env_revert
    systemctl is-active --quiet "$UNIT" || start_app
    state_clear; log_event "recovered: $S_MODE had not swapped"; return 0
  fi
  if [ "$cur" = "$S_NEWSHA" ] && [ "${S_PHASE:-}" != reverting ]; then
    say "   the new model is in place — restarting so the app serves it; the health check decides"
    stop_app; start_app
    if healthy "$(commit_of "$APP")" "${S_NEWREV:-}"; then
      if [ -n "${S_ENV_BACKUP:-}" ]; then rm -f "$S_ENV_BACKUP"; fi
      say "== [server] recovered: the new model is LIVE and healthy"
      state_clear; log_event "recovered: $S_MODE ok sha256=$S_NEWSHA"; return 0
    fi
  fi
  model_revert
}

recover_db() {
  TS="$S_TS"
  case "${S_PHASE:-}" in
    prepared|stopping)
      say "   the database had not been touched — making sure the app runs"
      systemctl is-active --quiet "$UNIT" || start_app
      state_clear; log_event "recovered: restore-db had not started"; return 0 ;;
    starting|health)
      systemctl is-active --quiet "$UNIT" || start_app
      if healthy "$(commit_of "$APP")"; then
        say "== [server] recovered: the restored database is live and healthy"
        state_clear; log_event "recovered: restore-db ${S_TARGET:-?} ok"; return 0
      fi ;;
  esac
  db_revert
}

recover_unit() {
  TS="$S_TS"
  rm -f "$UNITFILE.wrapgto-new" "$UNITFILE.wrapgto-old"
  if [ "${S_PHASE:-}" = prepared ] \
     || { [ -n "${S_UNIT_BACKUP:-}" ] && cmp -s "$UNITFILE" "$S_UNIT_BACKUP"; } \
     || { [ -z "${S_UNIT_BACKUP:-}" ] && [ ! -e "$UNITFILE" ]; }; then
    say "   the service definition had not been touched"
    systemctl is-active --quiet "$UNIT" || start_app
    state_clear; return 0
  fi
  if [ "${S_PHASE:-}" != reverting ]; then
    systemctl daemon-reload || true
    stop_app; start_app
    if healthy "$(commit_of "$APP")"; then
      say "== [server] recovered: the new service definition is live and healthy"
      state_clear; log_event "recovered: install-unit ok"; return 0
    fi
  fi
  unit_revert
}

# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------
case "$MODE" in
  check|list|watch|freeze|lock) ;;   # read-only on the live site: here, in the foreground
  stage|deploy|rollback|restore-db|promote|promote-undo|recover|install-unit)
    if [ -z "${WRAPGTO_DETACHED:-}" ] && [ "${WRAPGTO_DETACH:-auto}" != none ]; then launch_detached; fi
    begin_run ;;
  *) die 2 "unknown mode $MODE" ;;
esac

case "$MODE" in
check)
  say "   connected as $(whoami) on $(hostname)"
  if [ -f "$STATE" ]; then
    say "!! A CHANGE WAS INTERRUPTED HALF-WAY: $(state_summary)"
    say "!! Run: bash scripts/deploy_prod.sh recover"
  fi
  if [ -f "$DEPLOYS/.current" ]; then
    cur_pid="$(sed -n 's/.* pid \([0-9]*\) .*/\1/p' "$DEPLOYS/.current" 2>/dev/null || true)"
    if [ -n "$cur_pid" ] && kill -0 "$cur_pid" 2>/dev/null; then
      say "   A CHANGE IS RUNNING: $(cat "$DEPLOYS/.current") — watch it: bash scripts/deploy_prod.sh watch"
    fi
  fi
  printf '   app service: '; systemctl is-active "$UNIT" || true
  [ -d "$APP" ] || die 2 "app dir MISSING ($APP)"
  say "   disk: $(df -h "$BASE" | tail -1 | awk '{print $4 " free of " $2 " (" $5 " used)"}')"
  say "   live code: $(describe_tree "$APP")"
  ls -l --time-style=long-iso "$APP"/python/plo5bp/_engine*.so 2>/dev/null | awk '{print "   engine: built " $6 " " $7}' || true
  if bash -lc 'command -v cargo >/dev/null' </dev/null; then say "   cargo (root): ok"; else say "   cargo (root): missing — deploy with ENGINE_WHEEL=… (or SKIP_ENGINE=1 while the Rust code is unchanged)"; fi
  load_env
  if [ -f "$CKPT" ]; then
    say "   model: $CKPT · sha256 $(sha256sum "$CKPT" | cut -c1-12) · $(du -h "$CKPT" | cut -f1) · $(date -u -r "$CKPT" '+%Y-%m-%d %H:%M UTC')"
  else
    say "   model: $CKPT is MISSING — the site serves a random network"
  fi
  if [ -f "$CKPT.prev" ]; then say "   previous model kept: $(basename "$CKPT").prev (promote-undo puts it back)"; fi
  if [ -f "$CKPT.new" ]; then say "   staged, not live yet: $(basename "$CKPT").new (sha256 $(sha256sum "$CKPT.new" | cut -c1-12)) — /admin → System → Promote"; fi
  say "   settings: obs rev ${ENV_REV:-INVALID} from the service's settings (PLO5BP_OBS_REV)"
  tool health --once --url "http://127.0.0.1:$PORT/health" || true
  make_pre
  if [ -e "$DB" ]; then
    run_now=0; if systemctl is-active --quiet "$UNIT"; then run_now=1; fi
    as_svc "$SYSPY" "$PRE/deploytool.py" status --db "$DB" \
      --health-url "http://127.0.0.1:$PORT/health?deploy=1" --app-running "$run_now" || true
  else
    say "   (home games: no database)"
  fi
  say "== backups"
  tool backup-check --dir "$KEEP" || true
  if [ -f "$KEEP/.offsite-ok" ]; then
    say "   off-site copy: last success $(date -u -r "$KEEP/.offsite-ok" '+%Y-%m-%d %H:%M UTC')"
  else
    say "   off-site copy: NOT set up (ops/SERVER_SETUP.md, 'Backups')"
  fi
  say "== rollback targets (newest first)"
  if [ -d "$REL" ] && [ -n "$(ls -A "$REL" 2>/dev/null)" ]; then
    for d in $(ls -1t "$REL" | head -6); do say "   $d · $(describe_tree "$REL/$d")"; done
  else
    say "   none yet (the first deploy with this script creates one)"
  fi
  say "== recent changes ($BASE/deploys.log)"
  if [ -f "$BASE/deploys.log" ]; then tail -n 5 "$BASE/deploys.log" | sed 's/^/   /'; else say "   none yet"; fi
  say "== python packages"
  if [ -f "$W/lock.txt" ]; then
    if "$LIVEPY" -m pip freeze 2>/dev/null | tool deps --lock "$W/lock.txt"; then
      say "   the live environment matches requirements/server.txt"
    else
      say "   (the next deploy builds a new environment with these)"
    fi
  else
    say "   no requirements/server.txt yet — run: bash scripts/deploy_prod.sh freeze"
  fi
  extra=""
  for d in $(ls -A "$APP"); do
    case " python rust_engine Cargo.toml Cargo.lock rust-toolchain.toml pyproject.toml requirements ops BUILD_INFO.json .venv $CARRY " in
      *" $d "*) ;;
      *) extra="$extra $d" ;;
    esac
  done
  if [ -n "$extra" ]; then say "   also in the live folder (a deploy leaves these in the retired copy):$extra"; fi
  say "== service ($UNIT)"
  unit_snapshot
  tool unitcheck --show "$W/unit-show.txt" --cat "$W/unit-cat.txt" --app "$APP" --envfile "$ENVFILE" \
    --venv-python "$LIVEPY" || true
  if [ -f "$W/wrapgto.service" ]; then
    frag="$(sed -n 's/^FragmentPath=//p' "$W/unit-show.txt" | head -1)" || true
    if [ -n "$frag" ] && cmp -s "$W/wrapgto.service" "$frag"; then
      say "   the installed unit is the repo's ops/systemd/wrapgto.service"
    else
      say "   the installed unit differs from ops/systemd/wrapgto.service — install it with:"
      say "   bash scripts/deploy_prod.sh install-unit   (checked, guarded, backed up, reverted if unhealthy)"
    fi
  fi
  say "== nothing was changed"
  ;;

list)
  say "== rollback targets (newest first) — bash scripts/deploy_prod.sh rollback NAME"
  if [ -d "$REL" ]; then
    for d in $(ls -1t "$REL" | grep -v -- '-failed$' || true); do say "   $d · $(describe_tree "$REL/$d")"; done
  fi
  for f in $(ls -1t "$KEEP"/app-before-*.tgz 2>/dev/null | head -5); do say "   $(basename "$f") · a copy an older deploy kept"; done
  say "== database backups (newest first) — bash scripts/deploy_prod.sh restore-db NAME"
  ls -1t "$KEEP" 2>/dev/null | grep -Ev '^(app-before-|replaced-|\.)' | head -15 | sed 's/^/   /' || true
  for d in $(ls -1td "$KEEP"/replaced-* 2>/dev/null | head -3); do
    for f in "$d"/*.db; do
      if [ -f "$f" ]; then say "   $(basename "$d")/$(basename "$f") · the database a restore-db replaced"; fi
    done
  done
  ;;

watch)
  newest="$(ls -1t "$DEPLOYS"/*.log 2>/dev/null | head -1)" || true
  if [ -z "$newest" ]; then say "== no change has run on this server with this script yet"; exit 0; fi
  say "== $(basename "$newest")"
  if [ -f "$newest.rc" ]; then
    cat "$newest"
    rc="$(tr -dc '0-9' < "$newest.rc" || true)"
    say "== that run finished (exit ${rc:-?}: $([ "${rc:-1}" = 0 ] && echo ok || echo 'see above'))"
    exit "${rc:-1}"
  fi
  follow "$newest"
  ;;

stage|deploy)
  refuse_if_interrupted
  [ -d "$APP" ] || die 2 "$APP does not exist. Nothing was changed."
  [ -x "$LIVEPY" ] || die 2 "$LIVEPY is missing. Nothing was changed."
  same_fs
  load_env
  refuse_build_commit_override
  need_space "$BASE" 1500 "unpacking and pre-compiling the release"
  if [ -e "$DB" ]; then need_space "${TMPDIR:-/tmp}" $(( $(db_mb) + 200 )) "the pre-flight's copy of the database"; fi
  say "== [server] unpacking into staging (the live code is not touched)"
  rm -rf "${STAGE:?}"; mkdir -p "$STAGE"
  tar -xzf "$W/app.tgz" -C "$STAGE"
  check_carry_free "$STAGE"
  venv_for "$STAGE"
  engine_into "$STAGE" "$TREE_PY"
  prepare_tree "$STAGE" "$TREE_PY"
  pf=(--engine-check strict)
  if [ -e "$DB" ]; then pf+=(--db-copy-from "$DB"); fi
  preflight "$STAGE" "$TREE_PY" "${pf[@]}" || die 4 "the new code does not start (above). Nothing was changed."
  write_build_info "$STAGE"
  if [ "$MODE" = "stage" ]; then
    say "== [server] staged and start-tested in $STAGE — the live site was not touched"
    exit 0
  fi
  status_guard
  backup_now
  if switch_to "$STAGE" "$COMMIT"; then
    prune_releases; log_event "ok"; after_switch_notes
  else
    log_event "failed-rolled-back"; exit 5
  fi
  ;;

rollback)
  refuse_if_interrupted
  [ -d "$APP" ] || die 2 "$APP does not exist. Nothing was changed."
  if [ -z "$TARGET" ]; then TARGET="$(ls -1t "$REL" 2>/dev/null | grep -v -- '-failed$' | head -1)" || true; fi
  [ -n "$TARGET" ] || die 2 "there is nothing to roll back to. Nothing was changed."
  case "$TARGET" in */*|.*) die 2 "give the NAME of a rollback target (bash scripts/deploy_prod.sh rollback lists them)";; esac
  same_fs
  load_env
  refuse_build_commit_override
  if [ -d "$REL/$TARGET" ]; then
    T="$REL/$TARGET"
  elif [ -f "$KEEP/$TARGET" ]; then
    say "== [server] unpacking $TARGET"
    rm -rf "${STAGE:?}"; mkdir -p "$STAGE"
    tar -xzf "$KEEP/$TARGET" -C "$STAGE"
    stripped=""
    for d in $CARRY .venv; do   # an old deploy's copy may hold what belongs to the server
      if exists "$STAGE/$d"; then rm -rf "${STAGE:?}/$d"; stripped="$stripped $d/"; fi
    done
    if [ -n "$stripped" ]; then say "   left out of the old copy (they belong to the server):$stripped"; fi
    T="$STAGE"
  else
    die 2 "no rollback target called $TARGET. Nothing was changed."
  fi
  check_carry_free "$T"
  ls "$T"/python/plo5bp/_engine*.so >/dev/null 2>&1 || die 2 "$TARGET has no engine binary. Nothing was changed."
  say "== rolling back to $TARGET · $(describe_tree "$T")"
  TPY="$LIVEPY"; if [ -x "$T/.venv/bin/python" ]; then TPY="$T/.venv/bin/python"; fi
  prepare_tree "$T" "$TPY"
  pf=(--engine-check lenient)
  if [ -e "$DB" ]; then pf+=(--db-copy-from "$DB"); fi
  if ! preflight "$T" "$TPY" "${pf[@]}"; then
    if [ "$ROLLBACK_ANYWAY" != "1" ]; then
      die 4 "that version does not start with today's model and data (above). Nothing was changed. (ROLLBACK_ANYWAY=1 switches to it anyway — only when you know why the pre-flight failed and that it does not matter. FORCE=1 does NOT: it only skips the home-games check.)"
    fi
    say "!! ROLLBACK_ANYWAY=1: switching although the pre-flight FAILED (above). The health check after the"
    say "!! switch is the only safety net left — and on an older version it cannot see everything (below)."
  fi
  if [ ! -f "$T/BUILD_INFO.json" ]; then
    say "   note: this version predates BUILD_INFO.json. After the switch the health check can only see"
    say "   that the app answers (and whatever its older /health reports) — not which code is running,"
    say "   and maybe not the model, the critic or the obs revision. The pre-flight above checked those."
    say "   Afterwards: bash scripts/deploy_prod.sh check, and open Study for a recommendation."
  fi
  status_guard
  backup_now
  if switch_to "$T" "$(commit_of "$T")"; then
    prune_releases; log_event "rollback-to $TARGET"; after_switch_notes
  else
    log_event "rollback-failed $TARGET"; exit 5
  fi
  ;;

restore-db)
  refuse_if_interrupted
  [ -n "$TARGET" ] || die 2 "which backup? (bash scripts/deploy_prod.sh restore-db lists them)"
  case "$TARGET" in /*|*..*) die 2 "give a backup NAME from the list";; esac
  B="$KEEP/$TARGET"; [ -f "$B" ] || die 2 "no backup called $TARGET. Nothing was changed."
  [ -d "$APP" ] || die 2 "$APP does not exist. Nothing was changed."
  load_env
  need_space "$W" $(( $(stat -c %s "$B") * 6 / 1048576 + 100 )) "unpacking the backup"
  say "== [server] checking $TARGET"
  case "$B" in
    *.gz) gzip -dc "$B" > "$W/restore.db" ;;
    *) cp "$B" "$W/restore.db" ;;
  esac
  tool dbcheck "$W/restore.db" || die 2 "that backup is not a healthy database. Nothing was changed."
  status_guard
  backup_now   # the database as it is now — so this restore can itself be undone
  owner="$(stat -c %U:%G "$(dirname "$DB")")"
  REPL="$KEEP/replaced-$TS"
  status_guard quiet
  S_DB=$DB; S_REPLACED=$REPL; S_TARGET=$TARGET
  state_begin db prepared
  phase stopping
  stop_app
  if [ -e "$DB" ]; then
    # one self-contained file of the database being replaced (its -wal included), made by
    # its owner so SQLite never leaves a root-owned -shm next to the live path
    [ -n "$PRE" ] || make_pre
    as_svc "$SYSPY" "$PRE/deploytool.py" dbcopy "$DB" "$PRE/replaced.db" > "$W/replaced.log" 2>&1 \
      || say "!! could not make a single-file copy of the current database ($(tail -n 1 "$W/replaced.log")) — the raw files are kept"
  fi
  mkdir -p "$REPL/raw"
  phase moving-aside
  for f in "$DB-journal" "$DB-wal" "$DB-shm" "$DB"; do   # the .db LAST: its presence in raw/ = the set is complete
    if exists "$f"; then mv "$f" "$REPL/raw/" || { say "!! could not move $f aside"; db_revert; exit 5; }; fi
  done
  if [ -n "$PRE" ] && [ -f "$PRE/replaced.db" ]; then mv "$PRE/replaced.db" "$REPL/$(basename "$DB")"; fi
  cat > "$REPL/README.txt" <<EOF
The database that "restore-db $TARGET" replaced on $TS UTC.
  $(basename "$DB")   one self-contained copy of it (everything included) — the one to use.
  raw/        the files exactly as they were: a .db-wal belongs with its .db; never copy one without the other.
Put it back with:  bash scripts/deploy_prod.sh restore-db replaced-$TS/$(basename "$DB")
By hand (only if that cannot run): systemctl stop $UNIT; rm -f $DB-wal $DB-shm $DB-journal;
  cp $REPL/$(basename "$DB") $DB; chown $owner $DB; chmod 600 $DB; systemctl start $UNIT
EOF
  phase copying
  cp "$W/restore.db" "$DB.restore.tmp"; chown "$owner" "$DB.restore.tmp"; chmod 600 "$DB.restore.tmp"
  mv -f "$DB.restore.tmp" "$DB"
  phase starting
  start_app
  phase health
  if healthy "$(commit_of "$APP")"; then
    say "== [server] restored $TARGET — the database it replaced is in $REPL/ (README.txt says how to put it back)"
    state_clear; log_event "restore-db $TARGET"; prune_replaced
  else
    say "!! the app did not come up healthy with that database"
    journal_tail
    db_revert; exit 5
  fi
  ;;

promote|promote-undo)
  refuse_if_interrupted
  [ -d "$APP" ] || die 2 "$APP does not exist. Nothing was changed."
  load_env
  [ -d "$(dirname "$CKPT")" ] || die 2 "$(dirname "$CKPT") does not exist. Nothing was changed."
  case "$OBS_REV" in ""|1|2) ;; *) die 2 "OBS_REV must be 1 or 2. Nothing was changed.";; esac
  [ -n "$ENV_REV" ] || die 2 "PLO5BP_OBS_REV in the service's settings is not 1 or 2 — fix $ENVFILE first. Nothing was changed."
  NEW_REV="${OBS_REV:-$ENV_REV}"
  if [ "$MODE" = "promote" ]; then
    [ -f "$W/candidate.pt" ] || die 2 "no checkpoint was uploaded"
    [ "$(sha_of "$W/candidate.pt")" = "${CKPT_SHA:-}" ] \
      || die 2 "the uploaded checkpoint's sha256 does not match. Nothing was changed."
    SRC="$W/candidate.pt"; SRCNAME="${CKPT_NAME:-the checkpoint}"
    RESTART_NOW="${RESTART:-0}"
  else
    [ -f "$CKPT.prev" ] || die 2 "there is no previous model ($CKPT.prev) to go back to. Nothing was changed."
    [ -f "$CKPT" ] || die 2 "there is no live model ($CKPT) to swap with. Nothing was changed."
    SRC="$CKPT.prev"; SRCNAME="$(basename "$CKPT").prev"
    RESTART_NOW=1   # (/admin → System → Roll back is the no-restart undo)
  fi
  # 1. Can the live code serve it, on the revision it will be served at? (read-only)
  NEWCK_TMP="$CKPT.new.tmp"
  EXTRA_CLEAN="$NEWCK_TMP"
  cp -p "$SRC" "$NEWCK_TMP"; chmod 644 "$NEWCK_TMP"
  if [ -e "$CKPT" ]; then chown --reference="$CKPT" "$NEWCK_TMP"; else chown "$SVC:$SVC" "$NEWCK_TMP"; fi
  NEWSHA="$(sha_of "$NEWCK_TMP")"
  if ! preflight "$APP" "$LIVEPY" --checkpoint "$NEWCK_TMP" --engine-check info --env-set "PLO5BP_OBS_REV=$NEW_REV"; then
    ck_rev="$(pf_get obs_rev)"
    if [ "$(pf_get obs_rev_mismatch)" = "True" ] && [ -n "$ck_rev" ] && [ "$ck_rev" != "$NEW_REV" ]; then
      say "!! $SRCNAME was trained on obs rev $ck_rev; the server serves rev $NEW_REV. Switch both together"
      say "!! (the model and PLO5BP_OBS_REV in $ENVFILE — put back together if the site is not healthy):"
      say "!! run the same command again with OBS_REV=$ck_rev$([ "$MODE" = promote ] && echo ' RESTART=1') in front."
      if [ "$MODE" = promote ]; then say "RERUN-WITH OBS_REV=$ck_rev RESTART=1"; else say "RERUN-WITH OBS_REV=$ck_rev"; fi
    fi
    die 4 "the live code cannot serve that model as things stand (above). Nothing was changed."
  fi
  # 2. The no-restart path: the RUNNING app's model manager swaps <model>.new in (/admin),
  #    so it needs that manager, and a process already on the revision the model wants.
  if [ "$RESTART_NOW" != "1" ]; then
    if [ "$NEW_REV" != "$ENV_REV" ]; then
      say "RERUN-WITH OBS_REV=$NEW_REV RESTART=1"
      die 2 "an obs-revision change needs a restart (the app reads PLO5BP_OBS_REV once, when it starts): run the same promote again with RESTART=1 in front. Nothing was changed."
    fi
    if ! grep -qs '/admin/api/models' "$APP/python/plo5bp/ui/public.py"; then
      say "RERUN-WITH RESTART=1"
      die 2 "the live app is too old to swap models without a restart: run the same promote again with RESTART=1 in front (a restart, with the home-games check and an automatic revert). Nothing was changed."
    fi
    PROC_REV="$(tool health --url "http://127.0.0.1:$PORT/health" --field process_obs_rev 2>/dev/null || true)"
    [ -n "$PROC_REV" ] || die 2 "the app does not answer /health with its obs revision (is it running? bash scripts/deploy_prod.sh check) — the /admin swap needs it. Nothing was changed."
    if [ "$PROC_REV" != "$ENV_REV" ]; then
      say "RERUN-WITH RESTART=1"
      die 2 "the running app encodes obs rev $PROC_REV, but its settings now say $ENV_REV ($ENVFILE was changed without a restart): the /admin swap would serve this model on the wrong revision. Run it with RESTART=1 in front — the restart applies the settings. Nothing was changed."
    fi
    if [ -e "$CKPT.new" ]; then say "   (replacing a previously staged $(basename "$CKPT").new)"; fi
    mv -f "$NEWCK_TMP" "$CKPT.new"; EXTRA_CLEAN=""
    say "== [server] verified and staged as $(basename "$CKPT").new (sha256 ${NEWSHA:0:12}) — the live model is unchanged"
    say "   Switch it with NO restart (no hand, Study spot or trainer hand is interrupted):"
    say "   wrapgto.com/admin → System → the PLO5 model → Promote   (the current one is kept as .prev)"
    say "   Or run again with RESTART=1 to swap it in with a restart now."
    printf '%s staged sha256=%s from=%s\n' "$TS" "$NEWSHA" "${CKPT_NAME:-?}" >> "$BASE/models.log"
    say "MODEL-STAGED $TS $NEWSHA"
    exit 0
  fi
  # --- the restart path: the guard runs FIRST — a refusal leaves nothing staged or changed
  status_guard
  OLDSHA="$(sha_of "$CKPT")"
  S_CKPT=$CKPT; S_OLDSHA=$OLDSHA; S_NEWSHA=$NEWSHA; S_NEWREV=$NEW_REV; S_ENV_BACKUP=""
  state_begin model prepared
  if [ "$NEW_REV" != "$ENV_REV" ]; then set_env_rev "$NEW_REV"; fi
  EXTRA_CLEAN=""   # from here the journal owns $NEWCK_TMP (recover finishes or removes it)
  phase model
  if [ -f "$CKPT" ]; then cp -p "$CKPT" "$CKPT.prev.tmp" && mv -f "$CKPT.prev.tmp" "$CKPT.prev"; fi
  mv -f "$NEWCK_TMP" "$CKPT"
  phase restart
  say "== [server] swapping the model (${OLDSHA:0:12} -> ${NEWSHA:0:12})$([ -n "$S_ENV_BACKUP" ] && echo " and PLO5BP_OBS_REV -> $NEW_REV") and restarting"
  stop_app; start_app
  phase health
  if healthy "$(commit_of "$APP")" "$NEW_REV"; then
    if [ -n "$S_ENV_BACKUP" ]; then rm -f "$S_ENV_BACKUP"; fi
    if [ "$MODE" = promote ] && [ -f "$CKPT.new" ]; then
      rm -f "$CKPT.new"; say "   (removed the previously staged $(basename "$CKPT").new — this promote supersedes it)"
    fi
    state_clear
    say "== [server] the new model is LIVE and healthy (the one it replaced is $(basename "$CKPT").prev)"
    PJ="$PREFLIGHT_JSON"; [ -n "$PJ" ] || PJ='{}'
    printf '%s %s sha256=%s from=%s replaced=%s obs_rev=%s preflight=%s\n' "$TS" "$MODE" "$NEWSHA" \
      "${CKPT_NAME:-$SRCNAME}" "${OLDSHA:-none}" "$NEW_REV" "$PJ" >> "$BASE/models.log"
    say "MODEL-LOG $TS $NEWSHA $PJ"
    log_event "$MODE sha256=$NEWSHA obs_rev=$NEW_REV"
  else
    model_revert
    exit 5
  fi
  ;;

install-unit)
  refuse_if_interrupted
  [ -f "$W/wrapgto.service" ] || die 2 "the upload has no wrapgto.service"
  [ -d "$APP" ] || die 2 "$APP does not exist. Nothing was changed."
  load_env
  say "== [server] the service now, and what installing ops/systemd/wrapgto.service would change"
  unit_snapshot
  tool unitcheck --show "$W/unit-show.txt" --cat "$W/unit-cat.txt" --app "$APP" --envfile "$ENVFILE" \
    --venv-python "$LIVEPY" --new-unit "$W/wrapgto.service" \
    || die 2 "installing it now would break the site (the !! lines above say what to fix first). Nothing was changed."
  if [ -f "$UNITFILE" ] && cmp -s "$W/wrapgto.service" "$UNITFILE"; then
    say "== the installed unit already is ops/systemd/wrapgto.service — nothing to do"
    exit 0
  fi
  if command -v systemd-analyze >/dev/null 2>&1; then
    systemd-analyze verify "$W/wrapgto.service" > "$W/verify.log" 2>&1 \
      || { sed 's/^/   | /' "$W/verify.log"; die 2 "systemd-analyze rejects the new unit (above). Nothing was changed."; }
  fi
  status_guard
  backup_now
  status_guard quiet
  S_UNIT_BACKUP=""
  if [ -f "$UNITFILE" ]; then
    S_UNIT_BACKUP="$DEPLOYS/unit-before-$TS.service"
    cp -p "$UNITFILE" "$S_UNIT_BACKUP" || die 2 "could not keep a copy of $UNITFILE. Nothing was changed."
  fi
  state_begin unit prepared
  phase installing
  cp "$W/wrapgto.service" "$UNITFILE.wrapgto-new"; chown root:root "$UNITFILE.wrapgto-new"; chmod 644 "$UNITFILE.wrapgto-new"
  mv -f "$UNITFILE.wrapgto-new" "$UNITFILE"
  systemctl daemon-reload || true
  phase restart
  say "== [server] installed $UNITFILE — restarting the app with it"
  stop_app; start_app
  phase health
  if healthy "$(commit_of "$APP")"; then
    state_clear
    say "== [server] the new service definition is LIVE and healthy"
    if [ -n "$S_UNIT_BACKUP" ]; then say "   (the previous one is kept in $S_UNIT_BACKUP)"; fi
    log_event "install-unit ok"
  else
    journal_tail
    unit_revert; exit 5
  fi
  ;;

recover)
  if [ ! -f "$STATE" ]; then
    say "== nothing to recover: no change was interrupted"
    if ! exists "$APP"; then say "!! but $APP is MISSING — look at $REL (ls -lt $REL)"; fi
    if ! systemctl is-active --quiet "$UNIT"; then say "!! and the app is not running: systemctl start $UNIT"; fi
    exit 0
  fi
  # shellcheck source=/dev/null
  . "$STATE"
  say "== [server] recovering: $(state_summary)"
  case "${S_KIND:-}" in
    switch) recover_switch ;;
    model) recover_model ;;
    db) recover_db ;;
    unit) recover_unit ;;
    *) die 2 "the journal $STATE is unreadable (kind '${S_KIND:-}'). Look at it and $APP by hand." ;;
  esac
  ;;

freeze)
  "$LIVEPY" -m pip freeze > "$W/freeze.txt" 2>/dev/null || die 2 "pip freeze failed in the live environment"
  say "--- LOCK BEGIN"
  tool lockfile --header "requirements/server.txt — the production server's exact Python packages.
Written by: bash scripts/deploy_prod.sh freeze   (from the live environment, $TS UTC)
Change it: edit requirements/server.in, then run: bash scripts/deploy_prod.sh lock" < "$W/freeze.txt"
  say "--- LOCK END"
  ;;

lock)
  [ -f "$W/server.in" ] || die 2 "requirements/server.in was not uploaded"
  basepy="$("$LIVEPY" -c 'import sys; print(getattr(sys, "_base_executable", sys.executable))')"
  say "== [server] resolving requirements/server.in in a throwaway environment (a few minutes; nothing live is touched)"
  "$basepy" -m venv "$W/lockenv" || die 2 "could not create a throwaway environment"
  pipi=("$W/lockenv/bin/python" -m pip install -q --no-input --disable-pip-version-check)
  ok=0
  if [ -f "$W/lock.txt" ]; then
    # Keep every current pin that still fits, so a new package moves nothing else.
    grep -v '^-' "$W/lock.txt" > "$W/constraints.txt" || true
    "${pipi[@]}" -c "$W/constraints.txt" -r "$W/server.in" > "$W/pip.log" 2>&1 && ok=1
    [ "$ok" = 1 ] || say "   (the current pins conflict with server.in — resolving from scratch)"
  fi
  if [ "$ok" = 0 ]; then
    "${pipi[@]}" -r "$W/server.in" > "$W/pip.log" 2>&1 || { tail -n 20 "$W/pip.log"; die 2 "pip could not resolve requirements/server.in"; }
  fi
  "$W/lockenv/bin/python" -m pip freeze > "$W/freeze.txt" 2>/dev/null
  say "--- LOCK BEGIN"
  tool lockfile --header "requirements/server.txt — the production server's exact Python packages.
Written by: bash scripts/deploy_prod.sh lock   (resolved on the server, $TS UTC)
Change it: edit requirements/server.in, then run: bash scripts/deploy_prod.sh lock" < "$W/freeze.txt"
  say "--- LOCK END"
  ;;
esac
