#!/usr/bin/env bash
#
# The whole pipeline. CI runs exactly this script, so "it passes locally" and
# "it passes in CI" cannot diverge — the same reasoning that keeps the Python
# side shelling out to the engine binary that ships rather than binding it.
#
#   ci/run.sh          everything, including sanitizers
#   ci/run.sh quick    skip the sanitizer build, which dominates the wall clock
#
# Every stage prints its own timing. A suite that quietly grows from four
# seconds to ninety is a suite people start skipping.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-full}"
PY="${PY:-$ROOT/.venv/bin/python}"

FAILED=()
PASSED=0

run() {
  local name="$1"; shift
  local started
  started=$(date +%s)

  printf '\n\033[1m▸ %s\033[0m\n' "$name"
  if "$@"; then
    local elapsed=$(( $(date +%s) - started ))
    printf '  \033[32m✓\033[0m %s (%ds)\n' "$name" "$elapsed"
    PASSED=$((PASSED + 1))
  else
    local elapsed=$(( $(date +%s) - started ))
    printf '  \033[31m✗ FAILED\033[0m %s (%ds)\n' "$name" "$elapsed"
    FAILED+=("$name")
  fi
}

# Everything runs even after a failure. Stopping at the first one means a commit
# that breaks three things gets fixed three times, once per CI round trip.

printf '\033[1mAudionova CI\033[0m — mode: %s\n' "$MODE"

# --- gates first: cheap, and they fail for reasons no test would catch -------
run "secret scan"         "$PY" ci/check_secrets.py
run "licence gate"        "$PY" ci/check_licences.py
run "structural gates"    "$PY" ci/check_structure.py

# --- engine ------------------------------------------------------------------
run "engine: build"       make -C engine --no-print-directory all
run "engine: contract"    make -C engine --no-print-directory test
run "engine: I/O"         make -C engine --no-print-directory iotest
# THE ONLY CROSS-ARCHITECTURE CHECK IN THE PIPELINE. Every other engine test
# compares this machine's output against this machine's expectations, which says
# nothing about whether two machines agree. Both architectures comparing against
# one committed render establishes that transitively, without CI having to ship
# artifacts between jobs.
run "engine: golden"      "$PY" ci/check_golden.py

if [ "$MODE" != "quick" ]; then
  # Not optional discipline. Mutation testing found a partition-indexing bug
  # that read out of bounds, produced benign-looking garbage, and passed every
  # functional assertion. Memory errors do not reliably show up as wrong numbers.
  run "engine: sanitizers" make -C engine --no-print-directory san
fi

# --- backend -----------------------------------------------------------------
for suite in loudness measure schema planner rules; do
  run "backend: $suite"   make -C backend --no-print-directory "$suite"
done

# These render audio, so they are slower and are grouped after the fast ones.
for suite in bridge refine dynamics verify sensitivity recovery api; do
  run "backend: $suite"   make -C backend --no-print-directory "$suite"
done

# --- report ------------------------------------------------------------------
printf '\n────────────────────────────────────────\n'
if [ ${#FAILED[@]} -eq 0 ]; then
  printf '\033[32mall %d stages passed\033[0m\n' "$PASSED"
  exit 0
fi

printf '\033[31m%d of %d stages failed:\033[0m\n' \
  "${#FAILED[@]}" "$((PASSED + ${#FAILED[@]}))"
for name in "${FAILED[@]}"; do
  printf '  ✗ %s\n' "$name"
done
exit 1
