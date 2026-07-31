"""What the demo render changed, and whether it would ship.

Split out of run.sh because a shell script containing three heredocs of Python
is a shell script nobody will edit — and because the loudness matching below is
the part that decides whether a listening comparison means anything.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.audio import wavio
from backend.measure import loudness
from backend.measure.report import measure_file
from backend.verify import gate

GREEN, RED, DIM, BOLD, OFF = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"


class _Node:
    def __init__(self, data: dict) -> None:
        self.type = data["type"]
        self.params = data.get("params", {})
        self.bypass = data.get("bypass", False)


class _Graph:
    """Just enough of an EffectGraph for the gate to read the ceiling."""

    def __init__(self, recipe: dict) -> None:
        self.nodes = [_Node(n) for n in recipe.get("nodes", [])]
        self.master = [_Node(n) for n in recipe.get("master", [])]


def dig(report: dict, path: str):
    node = report
    for key in path.split("."):
        node = (node or {}).get(key)
    return node


def main(argv: list[str]) -> int:
    source_path, out_path, matched_path, recipe_path = argv[1:5]

    before = measure_file(source_path).to_dict()
    after = measure_file(out_path).to_dict()

    print(f"\n{BOLD}what changed{OFF}")
    rows = (
        ("integrated loudness", "loudness.integrated_lufs", "LUFS"),
        ("true peak", "loudness.true_peak_dbtp", "dBTP"),
        ("loudness range", "loudness.loudness_range_lu", "LU"),
        ("crest factor", "dynamics.crest_factor_db", "dB"),
        ("peak-to-loudness", "dynamics.peak_to_loudness_ratio_db", "dB"),
        ("spectral tilt", "spectrum.tilt_db_per_octave_re_pink", "dB/oct"),
        ("HF rolloff", "spectrum.hf_rolloff_db_per_octave", "dB/oct"),
    )
    for label, path, unit in rows:
        old, new = dig(before, path), dig(after, path)
        if old is None or new is None:
            continue
        delta = new - old
        colour = GREEN if delta > 0 else (RED if delta < 0 else "")
        print(f"  {label:22s} {old:8.2f} -> {new:8.2f} {unit:7s} "
              f"{colour}{delta:+.2f}{OFF}")

    warnings = after.get("warnings") or []
    if warnings:
        print(f"\n  {DIM}warnings on the output:{OFF}")
        for warning in warnings:
            print(f"    {warning['severity']:8s} {warning['code']}")

    # --- the safety gate ---------------------------------------------------
    with open(recipe_path) as handle:
        recipe = json.load(handle)

    processed = wavio.read_wav(out_path)
    result = gate.verify(processed.samples, processed.sample_rate,
                         graph=_Graph(recipe), source_report=before)

    print(f"\n{BOLD}safety gate{OFF}")
    if result.passed:
        print(f"  {GREEN}passed{OFF} — {len(result.checked)} checks: "
              f"{', '.join(result.checked)}")
    else:
        print(f"  {RED}FAILED — this render would not ship{OFF}")
        for failure in result.failures:
            print(f"    {failure.code}: {failure.message}")

    # --- loudness matching -------------------------------------------------
    #
    # AN UNMATCHED COMPARISON IS NOT A COMPARISON. Louder reliably wins a blind
    # A/B whether or not it is better, and this chain lands several dB below the
    # source. Matching the levels is the only way to hear what the PROCESSING
    # did rather than what the level did — and it is the same reason the plan
    # specifies a loudness-matched A/B transport in the product itself.
    source_audio = wavio.read_wav(source_path)
    source_lufs = loudness.integrated_lufs(source_audio.samples,
                                           source_audio.sample_rate)
    processed_lufs = loudness.integrated_lufs(processed.samples,
                                              processed.sample_rate)

    print(f"\n{BOLD}loudness-matched copy{OFF}")
    if not (np.isfinite(source_lufs) and np.isfinite(processed_lufs)):
        print("  one of the files has no measurable loudness; skipping")
        return 0

    offset = source_lufs - processed_lufs
    matched = processed.samples * (10.0 ** (offset / 20.0))

    # Matching must never push the file into clipping. If it would, both come
    # down together so the comparison stays fair and the file stays clean.
    peak = float(np.max(np.abs(matched)))
    trim = 0.0
    if peak > 0.999:
        trim = 20.0 * float(np.log10(0.999 / peak))
        matched *= 10.0 ** (trim / 20.0)

    wavio.write_wav(matched_path, matched, processed.sample_rate, 24)
    print(f"  applied {offset:+.2f} dB, so both files sit at {source_lufs:.2f} LUFS")
    if trim:
        print(f"  then {trim:+.2f} dB on both to stay below full scale")
    print(f"  {DIM}compare against this one, not the raw output{OFF}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
