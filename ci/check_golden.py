"""Compare this machine's render against the committed golden.

    ./.venv/bin/python ci/check_golden.py

THIS IS THE ONLY THING THAT TESTS THE CROSS-ARCHITECTURE CLAIM. The contract
tests run on x86-64 and on arm64 and each machine checks its own output against
its own expectations, which proves nothing about whether the two agree. Both
comparing against one committed file does, transitively, and without CI shipping
artifacts between jobs.

TWO THRESHOLDS, AND THE DIFFERENCE IS THE POINT.

  bit-exact       within one (engine_version, arch). Any deviation is a bug in
                  something that promised to be deterministic.
  -140 dBFS       across architectures. libm differs by ULPs between platforms,
                  SIMD changes summation order, and FMA contraction is a
                  compiler decision. Chasing bit-exactness across arch is a
                  losing fight that produces flaky CI rather than correct audio.

The script does not know which case it is in — it cannot tell whether the golden
was generated here — so it always reports the exact error and applies the looser
bound. A bit-exact match is stated when it happens, because losing it on the
machine that generated the file would be a real regression worth seeing.

LAG IS CHECKED SEPARATELY AND MUST BE EXACTLY ZERO. A latency error shows up as
an enormous sample-wise error that says nothing about its cause, and every node
reporting exact integer latency is precisely the property that makes delay
compensation trustworthy. Cross-correlating and asserting the peak sits at zero
turns "the audio is wrong" into "the audio is late", which are different bugs.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.audio import wavio  # noqa: E402
from backend.engine.render import render_array  # noqa: E402

GOLDEN_DIR = os.path.join(ROOT, "engine", "tests", "golden")

# -140 dBFS as an amplitude. The plan's number, and now testable rather than
# asserted: if both architectures pass with room to spare it was generous, and
# if one sits just under it the threshold is doing real work.
TOLERANCE_DB = -140.0
TOLERANCE = 10.0 ** (TOLERANCE_DB / 20.0)

GREEN, RED, DIM, BOLD, OFF = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"


def db(x: float) -> str:
    """dBFS, or the word for it when the error is exactly nothing.

    log10(0) has no useful rendering, and a clamped stand-in like -6000 dBFS
    reads as a measurement when it is an artefact of the clamp.
    """
    if float(x) == 0.0:
        return "exactly zero"
    return f"{20 * np.log10(float(x)):+.1f} dBFS"


def lag_of(a: np.ndarray, b: np.ndarray) -> int:
    """Where the cross-correlation peaks, in samples.

    Computed on the first channel over a window around the loudest part, which
    is where a shift is actually visible — correlating silence against silence
    peaks at zero regardless and would report success for a broken render.
    """
    x, y = a[0], b[0]
    centre = int(np.argmax(np.abs(y)))
    half = min(4096, len(y) // 2)
    lo, hi = max(0, centre - half), min(len(y), centre + half)
    x, y = x[lo:hi], y[lo:hi]
    if not np.any(x) or not np.any(y):
        return 0
    correlation = np.correlate(x - x.mean(), y - y.mean(), mode="full")
    return int(np.argmax(np.abs(correlation)) - (len(y) - 1))


def main() -> int:
    source_path = os.path.join(GOLDEN_DIR, "source.wav")
    expected_path = os.path.join(GOLDEN_DIR, "expected.wav")
    recipe_path = os.path.join(GOLDEN_DIR, "recipe.json")

    for path in (source_path, expected_path, recipe_path):
        if not os.path.exists(path):
            print(f"{RED}missing {path}{OFF} — run ci/make_golden.py")
            return 2

    source = wavio.read_wav(source_path)
    expected = wavio.read_wav(expected_path).samples
    with open(recipe_path) as handle:
        recipe = json.load(handle)

    rendered = render_array(source.samples, source.sample_rate, recipe)

    if rendered.shape != expected.shape:
        print(f"{RED}FAIL{OFF} shape {rendered.shape} against golden "
              f"{expected.shape} — the render changed length, which is a "
              f"latency or tail bug rather than a precision one")
        return 1

    error = np.abs(rendered - expected)
    worst = float(np.max(error))
    rms = float(np.sqrt(np.mean(error**2)))
    lag = lag_of(rendered, expected)

    print(f"{BOLD}golden render{OFF}  {expected.shape[1]} frames x "
          f"{expected.shape[0]} ch")
    print(f"  peak error   {db(worst)}   (tolerance {TOLERANCE_DB:.0f} dBFS)")
    print(f"  rms  error   {db(rms)}")
    print(f"  lag          {lag} samples")

    failed = False
    if worst == 0.0:
        print(f"  {GREEN}bit-exact{OFF} — identical to the committed golden")
    elif worst <= TOLERANCE:
        headroom = TOLERANCE_DB - 20 * np.log10(max(worst, 1e-300))
        print(f"  {GREEN}within tolerance{OFF} by {headroom:.1f} dB "
              f"{DIM}(expected across architectures){OFF}")
    else:
        print(f"  {RED}FAIL{OFF} peak error exceeds {TOLERANCE_DB:.0f} dBFS")
        failed = True

    if lag != 0:
        print(f"  {RED}FAIL{OFF} cross-correlation peaks at {lag} samples, not 0 "
              f"— this is a latency compensation bug, not a precision one")
        failed = True

    if failed:
        print(f"\n{DIM}If the output was MEANT to change, regenerate with "
              f"ci/make_golden.py and say so in the commit message. "
              f"Regenerating to turn a build green deletes the test.{OFF}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
