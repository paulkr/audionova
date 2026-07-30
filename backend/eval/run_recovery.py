"""Run the rule-based baseline over real stems and print the recovery floor.

    python -m backend.eval.run_recovery audio/stems/*.wav

The number at the bottom is the point of the whole exercise: what a handful of
thresholds achieves on this corpus. Any future planner has to beat it to have
demonstrated anything, and without it a planner scoring 0.6 is unreadable —
there is no way to tell reasoning from having learned which nodes are common.

Read the ordering figure with suspicion. The baseline emits a fixed
conventional sequence and therefore has NO ordering information at all; its
score is decided entirely by how often the chains happen to follow convention.
See test_recovery.py, which pins that down so the number cannot be mistaken for
ability.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import time

from backend.engine.render import render_array
from backend.eval import chains, corpus, recover, sensitivity
from backend.measure import report as report_mod


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stems", nargs="+", help="stem WAV files")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--quiet", action="store_true",
                        help="print only the aggregate scores")
    args = parser.parse_args(argv)

    stems = [p for p in args.stems if os.path.exists(p)]
    if not stems:
        print("no stems found", file=sys.stderr)
        return 2

    scores: list[chains.RecoveryScore] = []
    by_chain: dict[str, list[chains.RecoveryScore]] = collections.defaultdict(list)
    started = time.time()

    for path in stems:
        source, rate = corpus.load_excerpt(path, args.seconds)
        context = corpus.SourceContext.measure(source, rate)

        # The null render, once per stem: the only valid baseline for a delta,
        # since measuring against the raw source folds the sample-rate
        # conversion's own change into every result.
        reference = sensitivity.render_reference(source, rate)
        before = sensitivity.flatten(report_mod.measure(reference, rate).to_dict())

        for template in chains.CHAIN_TEMPLATES:
            case = chains.build_chain(template, context)
            processed = render_array(source, rate, case.recipe)
            after = sensitivity.flatten(report_mod.measure(processed, rate).to_dict())
            deltas = {k: after[k] - before[k] for k in before.keys() & after.keys()}

            predicted = recover.recover_deterministic(deltas)
            score = chains.score_recovery(predicted, case.node_types)
            scores.append(score)
            by_chain[template["name"]].append(score)

            if not args.quiet:
                print(f"  {os.path.basename(path)[:20]:22s} {template['name']:18s} "
                      f"P{score.precision:.2f} R{score.recall:.2f} F{score.f1:.2f}  "
                      f"pred={predicted} actual={case.node_types}")

    print(f"\n{len(scores)} cases in {time.time() - started:.0f}s\n")
    for name, group in by_chain.items():
        summary = chains.aggregate(group)
        print(f"  {name:20s} P{summary['precision']:.2f} R{summary['recall']:.2f} "
              f"F{summary['f1']:.2f}  ordering {summary['ordering']:+.2f}")

    total = chains.aggregate(scores)
    print(f"\nBASELINE FLOOR  precision {total['precision']:.3f}  "
          f"recall {total['recall']:.3f}  f1 {total['f1']:.3f}  "
          f"ordering {total['ordering']:+.3f}  ({total['cases']} cases)")
    print("\nThe ordering figure is a property of the corpus, not of the "
          "recogniser:\nit emits a fixed sequence, so it scores +1 on "
          "conventionally-ordered chains\nand -1 on reversed ones. Treat it as "
          "the zero mark, not as partial skill.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
