"""Run the sensitivity sweep and print the detectability table.

    python -m backend.eval.run_sensitivity audio/stems/*.wav

The output is the answer to "what can the planner actually see", and it is
meant to be read rather than asserted on. The pass/fail version lives in
backend/tests/test_sensitivity.py, which checks only the hypotheses that were
written down in advance.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import time

from backend.eval import corpus, sensitivity


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stems", nargs="+", help="stem WAV files")
    parser.add_argument("--seconds", type=float, default=25.0)
    parser.add_argument("--only", nargs="*", default=None,
                        help="restrict to named perturbations")
    parser.add_argument("--top", type=int, default=6,
                        help="report fields to show per perturbation")
    args = parser.parse_args(argv)

    stems = [p for p in args.stems if os.path.exists(p)]
    if not stems:
        print("no stems found", file=sys.stderr)
        return 2

    only = tuple(args.only) if args.only else None
    print(f"{len(stems)} stems, {args.seconds:.0f}s excerpts\n")

    # Grouped by perturbation, then by source: the analysis correlates within
    # a source and averages across them, so the grouping has to survive.
    by_perturbation: dict[str, dict[str, list]] = collections.defaultdict(dict)
    started = time.time()
    total_sweeps = len(stems) * len([
        p for p in corpus.PERTURBATIONS if not only or p.name in only
    ])
    done = 0
    skipped: dict[str, list[str]] = collections.defaultdict(list)
    for path in stems:
        stem = os.path.splitext(os.path.basename(path))[0]
        source, rate = corpus.load_excerpt(path, args.seconds)
        context = corpus.SourceContext.measure(source, rate)
        for perturbation in corpus.PERTURBATIONS:
            if only and perturbation.name not in only:
                continue
            # Inapplicable is not undetected. Stereo width on a mono stem
            # cannot move anything, and scoring it as a miss would report a
            # hole in the measurement layer that does not exist.
            if not perturbation.applies_to(context):
                skipped[perturbation.name].append(stem)
                continue
            by_perturbation[perturbation.name][stem] = sensitivity.sweep_source(
                source, rate, perturbation
            )
            done += 1
            print(f"\r  {done}/{total_sweeps} sweeps ({stem}: {perturbation.name})"
                  + " " * 20, end="", flush=True)
    print(f"\r  {done} sweeps in {time.time() - started:.0f}s" + " " * 40)

    confirmed = 0
    inapplicable = []
    for perturbation in corpus.PERTURBATIONS:
        if perturbation.name not in by_perturbation:
            if perturbation.name in skipped:
                inapplicable.append(perturbation.name)
            continue
        effects = sensitivity.analyse(perturbation, by_perturbation[perturbation.name])
        ok, missing = sensitivity.hypothesis_verdict(perturbation, effects)
        confirmed += int(ok)

        header = f"{perturbation.name}  ({perturbation.node_type}.{perturbation.parameter})"
        print(f"\n{header}\n{'-' * len(header)}")
        print(f"  {'field':44s} {'slope':>10s} {'r':>6s} {'r|v|':>6s} {'span':>8s}  kind")
        for effect in effects[: args.top]:
            mark = "*" if effect.field in perturbation.expected_fields else " "
            print(f" {mark}{effect.field:44s} {effect.slope:+10.4f} "
                  f"{effect.r:+6.2f} {effect.r_abs:+6.2f} {effect.span:8.3f}  {effect.kind}")
        if missing:
            print(f"  PREDICTED BUT NOT DETECTED: {', '.join(missing)}")

    total = len([p for p in corpus.PERTURBATIONS if p.name in by_perturbation])
    print(f"\n{confirmed}/{total} perturbations detected on the fields predicted "
          f"in advance (* marks a predicted field)")

    if inapplicable:
        print(f"\n{len(inapplicable)} not applicable to any stem in this corpus: "
              f"{', '.join(inapplicable)}")
        print("  These are gaps in the CORPUS, not in the measurement layer — a "
              "node that\n  cannot act on the available material tells us nothing "
              "either way.")
    for name, stems_skipped in sorted(skipped.items()):
        if name in by_perturbation:
            print(f"  {name}: skipped on {', '.join(stems_skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
