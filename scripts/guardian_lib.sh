# shellcheck shell=bash
# Shared machinery of the pod's training guardians (2026-09-28, ML-024 / ML-035 /
# ML-057 / ML-059). A guardian sets STEM (checkpoint stem), GLOG (its log) and
# PY (the venv python), defines its own launch() -- the recipe -- and sources
# this file for everything else:
#
#   gl_numa_prefix              "taskset -c <cpus of the GPU's NUMA node>" or ""
#   gl_pick_warm [--need k=v]... [--need-own k=v]... [--also FILE]
#                               the checkpoint to resume from: the HIGHEST numbered
#                               <STEM>_<N>.pt (or the rolling <STEM>.pt when its
#                               counter says it is newer) that satisfies the
#                               requirements, else --also; not the newest by mtime
#                               (a cp / rsync / touch changes that). Exit 1 = none.
#   gl_inductor_cache           this stem's own TORCHINDUCTOR_CACHE_DIR (Inductor
#                               autotunes by timing and caches the choice: a cache
#                               shared with evaluators / sweeps changes numerics)
#   gl_check_heartbeat PID      kill PID when its heartbeat (runs/<STEM>.heartbeat,
#                               rewritten after every update) is older than
#                               STALE_SECS: a hung trainer looked "alive" to a
#                               PID-only guardian. Returns 0 when it killed. A
#                               heartbeat that is missing, another process's or
#                               older than PID's start (a train.py from before
#                               2026-09-28 writes none) turns the check OFF for
#                               that PID -- logged once -- and the PID watch rules.
#
# The decisions live in python/plo5bp/train/guardian.py (tested); this file only
# wires them into bash. SHIP TOGETHER: the guardians, this file and
# python/plo5bp/train/ (guardian.py, and the train.py that writes heartbeats) go
# to the pod in one pull -- sourcing refuses to run without the helper module.

if ! "$PY" -c "import plo5bp.train.guardian" >/dev/null 2>&1; then
  echo "guardian_lib.sh: python/plo5bp/train/guardian.py is missing or broken -- the guardians," \
       "scripts/guardian_lib.sh and python/plo5bp/train/ ship together: pull the whole repo" >&2
  exit 1
fi
GL_HB_OFF_PID=""

gl_numa_prefix(){
  command -v taskset >/dev/null 2>&1 || return 0
  local bus node
  bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | head -1 | tr 'A-F' 'a-f')
  node=$(cat "/sys/bus/pci/devices/${bus: -12}/numa_node" 2>/dev/null || echo -1)
  if [ "$node" -ge 0 ] 2>/dev/null && [ -r "/sys/devices/system/node/node${node}/cpulist" ]; then
    echo "taskset -c $(cat "/sys/devices/system/node/node${node}/cpulist")"
  fi
}

gl_pick_warm(){
  "$PY" -m plo5bp.train.guardian pick-warm --dir checkpoints --stem "$STEM" "$@" 2>>"$GLOG"
}

gl_inductor_cache(){
  echo "${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/plo5bp-inductor/$STEM}"
}

gl_check_heartbeat(){
  local pid="$1" age
  [ -n "$pid" ] || return 1
  age=$("$PY" -m plo5bp.train.guardian heartbeat-age "runs/${STEM}.heartbeat" --pid "$pid" 2>/dev/null || echo -1)
  if ! [ "${age:--1}" -ge 0 ] 2>/dev/null; then
    # Unknown: no heartbeat written by THIS trainer (e.g. an older train.py).
    if [ "$GL_HB_OFF_PID" != "$pid" ]; then
      GL_HB_OFF_PID="$pid"
      echo "[${STEM}-guardian $(date -u '+%m-%d %H:%M:%S')] heartbeat check OFF for pid $pid (no heartbeat from this trainer yet -- an older train.py writes none): PID watch only" >> "$GLOG"
    fi
    return 1
  fi
  GL_HB_OFF_PID=""
  [ "$age" -gt "${STALE_SECS:-10800}" ] || return 1
  echo "[${STEM}-guardian $(date -u '+%m-%d %H:%M:%S')] heartbeat ${age}s old (> ${STALE_SECS:-10800}s): pid $pid is HUNG -> SIGTERM, then SIGKILL" >> "$GLOG"
  kill -TERM "$pid" 2>/dev/null
  sleep 120
  kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null
  return 0
}
