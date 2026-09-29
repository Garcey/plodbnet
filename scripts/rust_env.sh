# Sourced by scripts/rust_check.sh and scripts/rust_bench.sh (not run on its own):
# the environment cargo needs to build and run the engine's Rust tests and
# benchmarks. pyo3's build step needs a Python (the repo's venv when there is
# one), and a test / bench binary needs libpython at run time: on Windows the
# base installation's folder (python3.dll) on PATH, on Linux its LIBDIR on
# LD_LIBRARY_PATH. A PYO3_PYTHON already set is kept.
if [ -z "${PYO3_PYTHON:-}" ]; then
  for p in .venv/Scripts/python.exe .venv/bin/python; do
    if [ -x "$p" ]; then PYO3_PYTHON="$(pwd)/$p"; break; fi
  done
fi
: "${PYO3_PYTHON:=$(command -v python3 || command -v python)}"
export PYO3_PYTHON
rust_env_libdir="$("$PYO3_PYTHON" -c 'import sys, sysconfig; print(sys.base_prefix if sys.platform == "win32" else (sysconfig.get_config_var("LIBDIR") or ""))')"
if [ -n "$rust_env_libdir" ]; then
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) PATH="$(cygpath -u "$rust_env_libdir"):$PATH"; export PATH ;;
    *) export LD_LIBRARY_PATH="$rust_env_libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
fi
unset rust_env_libdir

# rust_engine/bench is a workspace of its own (so no engine build ever resolves
# Criterion). Its Cargo.lock (git-ignored) starts as a copy of the root one and is
# re-derived whenever the root one changes: cargo then only ADDS Criterion's
# packages, so the benchmarks build the engine against exactly the dependency
# versions production uses.
rust_bench_lock() {
  local sum
  sum="$(cksum < Cargo.lock)"
  if [ ! -f rust_engine/bench/Cargo.lock ] || [ "$(cat rust_engine/bench/Cargo.lock.root 2>/dev/null)" != "$sum" ]; then
    cp Cargo.lock rust_engine/bench/Cargo.lock
    printf '%s\n' "$sum" > rust_engine/bench/Cargo.lock.root
  fi
}
