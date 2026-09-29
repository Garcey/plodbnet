#!/usr/bin/env bash
# Criterion benchmarks of the engine's hot kernels (TEST-026):
# rust_engine/bench/benches/kernels.rs says what each group times.
#
#   bash scripts/rust_bench.sh                            every benchmark (a few minutes)
#   bash scripts/rust_bench.sh encode                     only names matching a regex
#   bash scripts/rust_bench.sh eval -- --save-baseline before
#   bash scripts/rust_bench.sh eval -- --baseline before  after a change: the difference
#
# One rayon thread, production's build profile (fat LTO); results and saved
# baselines live under rust_engine/bench/target/criterion/. Close other heavy
# programs first: the numbers are only as quiet as the machine.
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck source=scripts/rust_env.sh
. scripts/rust_env.sh
rust_bench_lock

filter=()
if [ $# -gt 0 ] && [ "$1" != "--" ]; then filter=("$1"); shift; fi
if [ "${1:-}" = "--" ]; then shift; fi
exec cargo bench --manifest-path rust_engine/bench/Cargo.toml --bench kernels -- "${filter[@]}" "$@"
