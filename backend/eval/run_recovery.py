"""Run a recogniser over the generated corpus and print the recovery floor.

    python -m backend.eval.run_recovery audio/stems/*.wav
    python -m backend.eval.run_recovery --synthetic          # no stems needed

The number at the bottom is the point of the whole exercise: what a handful of
thresholds achieves on this corpus. Any future planner has to beat it to have
demonstrated anything, and without it a planner scoring 0.6 is unreadable —
there is no way to tell reasoning from having learned which nodes are common.

FOUR THINGS ARE REPORTED, AND THEY ARE NOT ONE NUMBER.

  THE HEADLINE covers only the nodes the sensitivity sweep demonstrated are
  visible. Chains deliberately containing a 0.2 dB shelf are in the corpus, and
  those nodes are counted on their own line rather than folded into the
  headline as misses — a harness whose ceiling is below 1.0 for reasons no
  planner can address produces a number that cannot improve.

  THE PER-TYPE TABLE, because "F1 0.58" hides whether the planner reliably
  finds compression and never finds stereo width. Those have different fixes.

  ORDERING, restricted to the chains whose ground-truth order has an exact
  mirror in the corpus, and printed beside an ORDER-BLIND CONTROL: the same
  predicted multiset re-emitted in a fixed conventional sequence. The previous
  ordering figure of +0.498 was a property of the corpus rather than of the
  recogniser, and the excess over this control is what that number should have
  been all along.

  PARAMETERS, on the chains where the type was recovered, in octaves and in dB,
  each beside a control that says how much of it is luck.

REPRODUCIBILITY. The corpus comes from a committed manifest and a fixed seed,
so a change in the score is attributable to the recogniser rather than to new
random chains. A manifest that does not regenerate from its own seed is a hard
error.

NO MODEL CALLS. `--planner` resolves a name from `recover.RECOGNISERS` and no
other way; nothing in this file constructs a client or reads a key.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import sys
import time
from typing import Any

import numpy as np

from backend.engine.render import DEFAULT_BINARY, render_array
from backend.eval import chains, corpus, generate, recover, sensitivity
from backend.measure import report as report_mod

SYNTHETIC_RATE = 48000.0

# The control's proposals are drawn across the range the shape is read over, so
# it is guessing on the same board rather than a wider one.
CONTROL_LOW_HZ = float(min(recover.SHAPE_BANDS))
CONTROL_HIGH_HZ = float(max(recover.SHAPE_BANDS))


def synthetic_sources(seconds: float, seed: int) -> dict[str, tuple[np.ndarray, float]]:
    """Deterministic stand-ins for the stems, for CI, which has none.

    `audio/stems/` is gitignored — its licence status is the user's to decide —
    so every suite here has to work without it. These are not a substitute for
    real material and the numbers they produce are not the numbers in the docs:
    real stems are what `make recover` is for. What they give is a run that
    exercises the whole harness on every machine.

    Three sources, because the corpus needs a stereo one for width to apply at
    all and a dynamic one for a compressor to have something to reduce. Steady
    noise has no dynamics.
    """
    rng = np.random.default_rng(seed)
    n = int(SYNTHETIC_RATE * seconds)
    freqs = np.fft.rfftfreq(n, 1.0 / SYNTHETIC_RATE)
    shape = np.ones_like(freqs)
    shape[1:] = 1.0 / np.sqrt(freqs[1:])
    shape[0] = 0.0

    def noise(channels: int) -> np.ndarray:
        spec = (rng.normal(size=(channels, freqs.size))
                + 1j * rng.normal(size=(channels, freqs.size)))
        out = np.fft.irfft(spec * shape, n=n, axis=-1)
        return out / np.max(np.abs(out)) * 0.35

    t = np.arange(n) / SYNTHETIC_RATE
    envelope = 0.25 + 0.75 * np.abs(np.sin(2.0 * np.pi * 0.7 * t)) ** 3

    stereo = noise(2)
    mono = noise(1) * envelope
    return {
        "synthetic-stereo": (stereo * envelope, SYNTHETIC_RATE),
        "synthetic-mono": (mono, SYNTHETIC_RATE),
        # Dual mono: two identical channels. Width cannot act on it, which is a
        # corpus gap rather than a detection failure, and the runner has to say
        # so rather than count it as a miss.
        "synthetic-dual-mono": (np.repeat(mono, 2, axis=0), SYNTHETIC_RATE),
    }


def _control_bands(rng: random.Random, recovered: list[recover.RecoveredBand]
                   ) -> list[tuple[float, float]]:
    """The same proposals with their frequencies replaced by guesses.

    NEAREST-NEIGHBOUR MATCHING FLATTERS. A proposal placed anywhere lands near
    something, and on a corpus whose EQ moves span three octaves "somewhere in
    the midrange" scores better than it deserves. This says how much better.
    """
    low, high = math.log(CONTROL_LOW_HZ), math.log(CONTROL_HIGH_HZ)
    return [(math.exp(low + (high - low) * rng.random()), band.gain_db)
            for band in recovered]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stems", nargs="*", help="stem WAV files")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--synthetic", action="store_true",
                        help="ignore the stems and use synthesised sources")
    parser.add_argument("--planner", default="rules",
                        help="a name registered in recover.RECOGNISERS")
    parser.add_argument("--manifest", default=generate.DEFAULT_MANIFEST)
    parser.add_argument("--control-seed", type=int, default=generate.DEFAULT_SEED)
    parser.add_argument("--quiet", action="store_true",
                        help="print only the aggregate scores")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="write the full result to this path")
    args = parser.parse_args(argv)

    if not os.path.exists(DEFAULT_BINARY):
        print(f"engine binary missing at {DEFAULT_BINARY}; "
              "run `make render` in engine/ — skipping", file=sys.stderr)
        return 0

    # HARD ERROR ON A MANIFEST THAT DOES NOT REGENERATE. Silently accepting a
    # different corpus makes a score change unattributable, which is the one
    # thing the manifest exists to prevent.
    specs = generate.verify_manifest(args.manifest)
    recognise = recover.resolve(args.planner)

    stems = [p for p in args.stems if os.path.exists(p)]
    if args.synthetic or not stems:
        if not args.synthetic:
            print("no stems found (audio/stems/ is gitignored) — "
                  "running on synthesised sources; the numbers below are NOT "
                  "the documented floor, which needs real material\n")
        sources = synthetic_sources(args.seconds, args.control_seed)
    else:
        sources = {}
        for path in stems:
            samples, rate = corpus.load_excerpt(path, args.seconds)
            sources[os.path.splitext(os.path.basename(path))[0]] = (samples, rate)

    scores: list[chains.RecoveryScore] = []
    by_family: dict[str, list[chains.RecoveryScore]] = collections.defaultdict(list)
    matches: list[chains.BandMatch] = []
    control_matches: list[chains.BandMatch] = []
    corpus_gaps: collections.Counter[str] = collections.Counter()
    control_rng = random.Random(args.control_seed)
    started = time.time()

    for source_name, (samples, rate) in sources.items():
        context = corpus.SourceContext.measure(samples, rate)

        # The null render, once per source: the only valid baseline for a delta,
        # since measuring against the raw source folds the sample-rate
        # conversion's own change into every result.
        reference = sensitivity.render_reference(samples, rate)
        before = sensitivity.flatten(report_mod.measure(reference, rate).to_dict())

        for spec in specs:
            case = chains.build_generated(spec, context)
            corpus_gaps.update(case.inapplicable)
            if not case.recipe.nodes:
                continue

            processed = render_array(samples, rate, case.recipe)
            after = sensitivity.flatten(report_mod.measure(processed, rate).to_dict())
            deltas = {k: after[k] - before[k] for k in before.keys() & after.keys()}

            prediction = recognise(deltas, before=before, after=after,
                                   context=context)
            score = chains.score_recovery(
                prediction.node_types, case.node_types,
                ignored=case.ignored,
                control_order=recover.CONVENTIONAL_ORDER,
                ordering_scored=case.ordering_eligible,
                name=f"{source_name}/{case.name}")
            scores.append(score)
            by_family[case.family].append(score)

            # PARAMETERS ONLY WHERE THE TYPE WAS RECOVERED. Scoring the
            # frequency of a band the recogniser never claimed existed measures
            # the type failure twice.
            if case.eq_truth and "parametric_eq" in score.predicted:
                recovered = [(b.center_freq_hz, b.gain_db)
                             for b in prediction.eq_bands]
                matches.extend(chains.match_bands(recovered, case.eq_truth))
                control_matches.extend(chains.match_bands(
                    _control_bands(control_rng, prediction.eq_bands),
                    case.eq_truth))

            if not args.quiet:
                print(f"  {source_name[:20]:22s} {case.name:16s} "
                      f"P{score.precision:.2f} R{score.recall:.2f} F{score.f1:.2f}  "
                      f"pred={score.predicted} actual={score.actual}"
                      + (f" ignored={score.ignored}" if score.ignored else ""))

    total = chains.aggregate(scores)
    result = report(scores, by_family, total, matches, control_matches,
                    corpus_gaps, time.time() - started, args)

    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(result, handle, indent=1, sort_keys=True)
            handle.write("\n")
    return 0


def _fmt(value: float, places: int = 2, sign: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "   -  "
    return f"{value:{'+' if sign else ''}.{places}f}"


def report(scores, by_family, total, matches, control_matches, corpus_gaps,
           elapsed, args) -> dict[str, Any]:
    print(f"\n{len(scores)} cases in {elapsed:.0f}s "
          f"(recogniser: {args.planner}, seed {args.control_seed})\n")

    print("  family              cases    P     R     F1")
    for name, group in sorted(by_family.items()):
        summary = chains.aggregate(group)
        print(f"  {name:20s} {summary['cases']:4d}  "
              f"{_fmt(summary['precision'])} {_fmt(summary['recall'])} "
              f"{_fmt(summary['f1'])}")

    # --- per node type ------------------------------------------------------
    print("\n  PER NODE TYPE  (in-evidence instances only)")
    print("  type              support  found  recall   proposed  precision  "
          "F1    below-floor  named")
    breakdown = chains.per_type(scores)
    for row in breakdown:
        print(f"  {row.node_type:18s} {row.support:5d}  {row.hits:5d}  "
              f"{_fmt(row.recall):7s}  {row.proposed:7d}  {_fmt(row.precision):8s}  "
              f"{_fmt(row.f1):5s}  {row.ignored_support:9d}  {row.named_ignored:5d}")

    if corpus_gaps:
        print("\n  CORPUS GAPS — nodes that cannot act on the material and were "
              "not applied.\n  Not a detection failure; the material is the "
              "limitation.")
        for node_type, count in sorted(corpus_gaps.items()):
            print(f"    {node_type:18s} {count} chain instances dropped")

    # --- ordering -----------------------------------------------------------
    print("\n  ORDERING  (mirrored chains only; every scored chain has its exact "
          "reverse in the corpus)")
    print(f"    scored on            {total['ordering_cases']} cases")
    print(f"    raw                  {_fmt(total['ordering'], 3, sign=True)}")
    print(f"    order-blind control  {_fmt(total['ordering_control'], 3, sign=True)}"
          "   same multiset, fixed conventional sequence")
    print(f"    shuffled control     {_fmt(total['ordering_shuffled'], 3, sign=True)}"
          "   exactly 0 by symmetry; it cannot correct the artifact")
    print(f"    EXCESS OVER CONTROL  {_fmt(total['ordering_excess'], 3, sign=True)}"
          "   <- the ordering figure. Anything else is the corpus.")

    # --- parameters ---------------------------------------------------------
    band = chains.aggregate_bands(matches)
    control = chains.aggregate_bands(control_matches)
    zero_gain = [abs(m.truth.gain_db) for m in matches if m.matched]
    print("\n  PARAMETERS  (only where parametric_eq was recovered)")
    print(f"    applied bands        {band['bands']}  "
          f"({band['matched']} matched to a proposal)")
    print(f"    median freq error    {_fmt(band['median_octaves'])} octaves"
          f"   control {_fmt(control['median_octaves'])}")
    print(f"    within 1/3 octave    {_fmt(band['within_third_octave'])}"
          f"          control {_fmt(control['within_third_octave'])}")
    print(f"    median gain error    {_fmt(band['median_gain_db'])} dB"
          f"       control {_fmt(float(np.median(zero_gain)) if zero_gain else float('nan'))}"
          "  (a control predicting 0 dB)")
    print(f"    within {chains.GAIN_TOLERANCE_DB:.0f} dB          "
          f"{_fmt(band['within_gain_tolerance'])}")

    print(f"\nBASELINE FLOOR  precision {total['precision']:.3f}  "
          f"recall {total['recall']:.3f}  f1 {total['f1']:.3f}  "
          f"({total['cases']} cases, {len(scores)} chains scored)")
    print(f"ORDERING FLOOR  {_fmt(total['ordering_excess'], 3, sign=True)} excess "
          f"over the order-blind control on {total['ordering_cases']} mirrored "
          "chains.\nA fixed-sequence recogniser scores exactly 0 here by "
          "construction, which is\nwhat the old +0.498 should have read.")

    return {
        "recogniser": args.planner,
        "seed": args.control_seed,
        "manifest_digest": generate.digest(generate.verify_manifest(args.manifest)),
        "total": total,
        "per_family": {name: chains.aggregate(group)
                       for name, group in by_family.items()},
        "per_type": [vars(row) | {"recall": row.recall, "precision": row.precision,
                                  "f1": row.f1} for row in chains.per_type(scores)],
        "parameters": band,
        "parameter_control": control,
        "corpus_gaps": dict(corpus_gaps),
    }


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
