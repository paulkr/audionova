"""Regenerate the cross-architecture golden render.

    ./.venv/bin/python ci/make_golden.py

WHAT THIS IS FOR. The engine's determinism contract is bit-exact within one
(engine_version, arch) and -140 dBFS across architectures. Both halves of that
were unevidenced: CI ran the contract tests on x86-64 and on arm64, each machine
checked its own output against its own expectations, and nothing ever compared
the two to each other. A test that passes independently on both machines says
nothing about whether they agree.

Committing one golden render fixes that transitively. Each architecture compares
against the same bytes, so if both pass, they agree with each other to within the
threshold — without CI needing to ship artifacts between jobs.

RUN THIS ONLY WHEN THE OUTPUT IS MEANT TO CHANGE, and say so in the commit
message. A golden regenerated to make a red build green is a deleted test.

WHY THE SIGNAL LOOKS LIKE THIS. Determinism bugs do not show up on well-behaved
input. Each component targets a specific way arithmetic diverges across
platforms:

  a log sweep          every filter coefficient path, across the whole band
  an impulse train     transient handling, lookahead, exact latency alignment
  correlated noise     the mid path
  decorrelated noise   the side path, so stereo width has something to widen
  a DC-ish tail        denormal handling, which is where FTZ/DAZ differences bite

and the chain runs the nonlinear, oversampled and randomised nodes, because
those are the ones that can differ: ADAA polynomials evaluate transcendentals,
oversampling filters sum long taps in an order SIMD may change, and dither draws
from the counter-based PRNG whose whole promise is that it does not care.
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
SOURCE = os.path.join(GOLDEN_DIR, "source.wav")
EXPECTED = os.path.join(GOLDEN_DIR, "expected.wav")
RECIPE = os.path.join(GOLDEN_DIR, "recipe.json")

RATE = 48000.0
SECONDS = 1.0
SEED = 20260804


def build_source() -> np.ndarray:
    """One second of stereo deliberately chosen to be hard to reproduce.

    Every constant here is fixed. The PRNG is seeded explicitly and uses PCG64
    rather than the global state, so this file regenerates byte-identically on
    any machine — which it must, or the golden would drift every time somebody
    ran it.
    """
    n = int(RATE * SECONDS)
    t = np.arange(n, dtype=np.float64) / RATE

    # Log sweep, 20 Hz to 20 kHz. Exercises every filter coefficient path.
    f0, f1 = 20.0, 20000.0
    k = (f1 / f0) ** (1.0 / SECONDS)
    sweep = 0.25 * np.sin(2 * np.pi * f0 * (k**t - 1.0) / np.log(k))

    # Impulses on an irregular grid. Irregular so no node's internal period can
    # accidentally align with them and hide a timing error.
    impulses = np.zeros(n)
    for position in (0.07, 0.19, 0.31, 0.55, 0.83):
        impulses[int(position * n)] = 0.9

    rng = np.random.Generator(np.random.PCG64(SEED))
    common = 0.08 * rng.standard_normal(n)
    left_only = 0.05 * rng.standard_normal(n)
    right_only = 0.05 * rng.standard_normal(n)

    # A slow ramp toward zero. The tail spends its last milliseconds in the
    # denormal range, which is exactly where a missing FTZ/DAZ scope shows up as
    # a numeric difference rather than only as a speed difference.
    decay = np.exp(-t * 30.0) * 1e-20

    left = sweep + impulses + common + left_only + decay
    right = sweep + impulses + common + right_only + decay
    return np.ascontiguousarray(np.stack([left, right]))


def build_recipe() -> dict:
    """A chain over the nodes most likely to diverge across architectures."""
    return {
        "sample_rate": RATE,
        "seed": 11,
        "output_sample_rate": RATE,
        "output_format": "float64",
        "nodes": [
            {"id": "eq", "type": "parametric_eq", "mix": 1.0, "bypass": False,
             "params": {"bands": [
                 {"filter_type": "highpass", "center_freq_hz": 40.0,
                  "gain_db": 0.0, "q_factor": 0.707},
                 {"filter_type": "bell", "center_freq_hz": 900.0,
                  "gain_db": -3.5, "q_factor": 1.4},
                 {"filter_type": "highshelf", "center_freq_hz": 8000.0,
                  "gain_db": 2.5, "q_factor": 0.707},
             ]}},
            {"id": "comp", "type": "compressor", "mix": 1.0, "bypass": False,
             "params": {"threshold_dbfs": -22.0, "ratio_to_1": 3.0,
                        "attack_ms": 8.0, "release_ms": 120.0,
                        "knee_db": 6.0, "lookahead_ms": 5.0}},
            # Nonlinear and oversampled: ADAA polynomials plus up/down FIRs.
            {"id": "drive", "type": "drive", "mix": 1.0, "bypass": False,
             "params": {"drive_db": 9.0, "output_trim_db": -4.5, "bias": 0.2}},
            {"id": "delay", "type": "delay", "mix": 0.25, "bypass": False,
             "params": {"time_ms": 137.0, "feedback_percent": 35.0,
                        "hf_damp_hz": 6000.0}},
            {"id": "verb", "type": "plate_reverb", "mix": 0.18, "bypass": False,
             "params": {"decay_s": 1.6, "pre_delay_ms": 20.0,
                        "modulation_depth": 0.6}},
            {"id": "width", "type": "stereo_width", "mix": 1.0, "bypass": False,
             "params": {"width_percent": 140.0, "bass_mono_below_hz": 150.0}},
        ],
        "master": [
            {"id": "limit", "type": "true_peak_limiter", "mix": 1.0,
             "bypass": False,
             "params": {"ceiling_dbtp": -1.0, "release_ms": 90.0}},
            # Draws from the counter-based PRNG. Its entire promise is that the
            # draw depends on absolute frame index and not on block layout or
            # thread count, so it belongs in a determinism golden.
            {"id": "dither", "type": "dither", "mix": 1.0, "bypass": False,
             "params": {}},
        ],
    }


def main() -> int:
    os.makedirs(GOLDEN_DIR, exist_ok=True)

    source = build_source()
    recipe = build_recipe()

    # 32-bit float, not 24-bit PCM. The comparison threshold is -140 dBFS and
    # 24-bit quantisation lands at about -144: storing the source as PCM would
    # put the format's own noise floor within a few dB of the thing being
    # measured, which is not a margin worth defending.
    wavio.write_wav(SOURCE, source, RATE, 32, is_float=True)
    with open(RECIPE, "w") as handle:
        json.dump(recipe, handle, indent=2, sort_keys=True)
        handle.write("\n")

    # RENDER FROM THE FILE, NOT FROM MEMORY. Writing float32 rounds the source
    # at about -145 dB relative to each sample, and the checker reads that
    # rounded file. Rendering the unrounded array here would bake a difference
    # into the golden that no machine could ever reproduce.
    #
    # Caught by the checker failing on the machine that had just written the
    # golden, at -138.5 dBFS peak error — a fifth of a dB of input rounding
    # amplified through a compressor, a saturator and a feedback delay into
    # something above the -140 dB threshold. Worth keeping in mind before
    # reading any future failure here as a portability problem.
    rendered = render_array(wavio.read_wav(SOURCE).samples, RATE, recipe)
    wavio.write_wav(EXPECTED, rendered, RATE, 32, is_float=True)

    peak = float(np.max(np.abs(rendered)))
    print(f"source    {source.shape[1]} frames x {source.shape[0]} ch")
    print(f"expected  {rendered.shape[1]} frames, peak "
          f"{20 * np.log10(max(peak, 1e-30)):+.2f} dBFS")
    print(f"written   {SOURCE}\n          {EXPECTED}\n          {RECIPE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
