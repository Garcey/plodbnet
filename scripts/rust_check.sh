#!/usr/bin/env bash
# The Rust engine's gate (ENG-032): formatting, clippy's lints as errors, then the
# Rust tests (golden digests included) — exactly what CI's "Rust tests" job runs.
# Run it before committing a Rust change.
#
#   bash scripts/rust_check.sh          check only (changes nothing)
#   bash scripts/rust_check.sh --fix    format the sources first (cargo fmt), then check
#
# The compiler is the one rust-toolchain.toml pins (rustup fetches it on first use);
# the lint policy is the [lints.clippy] table in rust_engine/Cargo.toml.
set -euo pipefail
cd "$(dirname "$0")/.."

FIX=0
case "${1:-}" in
  --fix) FIX=1 ;;
  "") ;;
  *) echo "usage: bash scripts/rust_check.sh [--fix]" >&2; exit 2 ;;
esac

# shellcheck source=scripts/rust_env.sh
. scripts/rust_env.sh

M=(--manifest-path rust_engine/Cargo.toml)
B=(--manifest-path rust_engine/bench/Cargo.toml)
fails=()
step() { printf '\n== %s\n' "$*"; }

step "format (cargo fmt: the engine and its benchmarks)"
if [ "$FIX" = 1 ]; then cargo fmt "${M[@]}"; cargo fmt "${B[@]}"; fi
cargo fmt "${M[@]}" --check || fails+=("format (fix: bash scripts/rust_check.sh --fix)")
cargo fmt "${B[@]}" --check || fails+=("format of rust_engine/bench (fix: bash scripts/rust_check.sh --fix)")

step "lints (cargo clippy, warnings are errors)"
cargo clippy --locked "${M[@]}" --profile fasttest --lib --tests -- -D warnings || fails+=("clippy")

step "tests (cargo test, profile fasttest)"
cargo test --locked "${M[@]}" --profile fasttest --lib || fails+=("cargo test")

step "benchmarks still build (cargo clippy on rust_engine/bench)"
rust_bench_lock
cargo clippy "${B[@]}" --all-targets -- -D warnings || fails+=("clippy rust_engine/bench")

echo
if [ "${#fails[@]}" -eq 0 ]; then
  echo "== rust checks passed"
else
  echo "!! failed: ${fails[*]}"
  exit 1
fi
