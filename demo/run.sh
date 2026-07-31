#!/usr/bin/env bash
#
# Hear it. Processes an audio file through a hand-written chain, writes the
# result, and reports what changed.
#
#   demo/run.sh audio/stems/vocals.wav
#   demo/run.sh mytrack.wav demo/vocal-chain.json
#
# THE CHAIN IS HAND-WRITTEN, NOT PLANNED. There is no model call in this
# repository yet, so nothing here chose these settings from the audio — a person
# did, once, and they are applied to whatever you point at. That makes this a
# test of the ENGINE, not of the product. The part that reads your sentence and
# decides the chain is the piece that does not exist.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

INPUT="${1:-}"
RECIPE="${2:-demo/vocal-chain.json}"
PY="${PY:-$ROOT/.venv/bin/python}"

if [ -z "$INPUT" ]; then
  echo "usage: demo/run.sh <input.wav> [recipe.json]"
  echo
  echo "Your stems:"
  ls audio/stems/*.wav 2>/dev/null | sed 's|^|  |' || echo "  (none in audio/stems)"
  exit 2
fi

if [ ! -f "$INPUT" ]; then
  echo "no such file: $INPUT"; exit 2
fi

if [ ! -x engine/build/anv_render ]; then
  echo "building the engine..."
  make -C engine --no-print-directory render
fi

BASE="$(basename "${INPUT%.*}")"
OUT="demo/out/${BASE}-processed.wav"
MATCHED="demo/out/${BASE}-matched.wav"
mkdir -p demo/out

printf '\n\033[1mrendering\033[0m %s\n  chain: %s\n\n' "$INPUT" "$RECIPE"
engine/build/anv_render --in "$INPUT" --out "$OUT" --recipe "$RECIPE"

"$PY" demo/report.py "$INPUT" "$OUT" "$MATCHED" "$RECIPE"

printf '\n\033[1mwrote\033[0m\n'
printf '  %s\n' "$OUT"
printf '  %s   \033[2m<- use this one to compare\033[0m\n' "$MATCHED"
printf '\ncompare:\n'
printf '  afplay %s\n' "$INPUT"
printf '  afplay %s\n\n' "$MATCHED"
