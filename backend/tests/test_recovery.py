"""Chain recovery scoring, and the rule-based floor a planner must clear.

Hermetic: the scoring functions are exercised on hand-written lists and the
recogniser on hand-written delta dictionaries, so nothing here renders audio.
The real numbers come from `make recover`, which runs the same code over the
stems.

The most important test in this file is the one asserting that the baseline's
ORDERING score is an artifact. It comes out at about +0.5, which looks like
half-decent ordering ability and is nothing of the kind — the recogniser always
emits conventional order, so it scores +1.00 on chains that happen to follow
convention and -1.00 on the one that does not. A number that looks like skill
and is really a property of the corpus is exactly the kind of proxy the plan
warns about, so it is pinned down here rather than left to be discovered.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.eval import chains, recover  # noqa: E402
from backend.eval.corpus import SourceContext  # noqa: E402


class Results:
    def __init__(self) -> None:
        self.checks = 0
        self.failures = 0

    def check(self, ok: bool, what: str) -> None:
        self.checks += 1
        if not ok:
            self.failures += 1
            print(f"\n  FAIL  {what}")

    def near(self, got: float, want: float, tol: float, what: str) -> None:
        self.check(got is not None and math.isfinite(got) and abs(got - want) <= tol,
                   f"{what}: got {got}, want {want} +/- {tol}")


R = Results()


def test_scoring() -> None:
    print("[1] recovery scoring")

    perfect = chains.score_recovery(["parametric_eq", "compressor"],
                                    ["parametric_eq", "compressor"])
    R.near(perfect.precision, 1.0, 1e-9, "exact recovery has precision 1")
    R.near(perfect.recall, 1.0, 1e-9, "exact recovery has recall 1")
    R.near(perfect.ordering, 1.0, 1e-9, "exact recovery has ordering +1")

    # MULTISET, not set. A chain with two EQ nodes is not recovered by
    # proposing one — and "EQ, dynamics, EQ" is the commonest structure there
    # is, so treating these as sets would score the shape that ordering exists
    # to capture as fully recovered.
    partial = chains.score_recovery(
        ["parametric_eq"], ["parametric_eq", "drive", "parametric_eq"])
    R.near(partial.precision, 1.0, 1e-9, "one correct guess is fully precise")
    R.near(partial.recall, 1.0 / 3.0, 1e-9,
           "but recovers only one of three nodes, counting duplicates")

    over = chains.score_recovery(
        ["parametric_eq", "compressor", "delay", "plate_reverb"], ["parametric_eq"])
    R.near(over.recall, 1.0, 1e-9, "proposing everything recovers everything")
    R.near(over.precision, 0.25, 1e-9, "but precision punishes the guessing")
    R.check(over.f1 < 0.5,
            f"so F1 stays low for a scattergun answer, {over.f1:.2f}")

    # Reversed order, same nodes: the module score must be blind to order and
    # the ordering score must catch it. If one metric moved with the other they
    # would not be measuring two different things.
    reversed_case = chains.score_recovery(
        ["compressor", "parametric_eq"], ["parametric_eq", "compressor"])
    R.near(reversed_case.f1, 1.0, 1e-9, "reversing the order leaves F1 at 1")
    R.near(reversed_case.ordering, -1.0, 1e-9, "and drives ordering to -1")

    R.check(math.isnan(chains.score_recovery(["gain"], ["gain"]).ordering),
            "a single-node chain has no defined ordering")

    empty = chains.score_recovery([], ["parametric_eq"])
    R.near(empty.precision, 0.0, 1e-9, "an empty prediction scores zero")
    R.near(empty.recall, 0.0, 1e-9, "and recovers nothing")

    summary = chains.aggregate([perfect, reversed_case])
    R.near(summary["f1"], 1.0, 1e-9, "aggregate averages F1")
    R.near(summary["ordering"], 0.0, 1e-9, "and averages ordering, +1 with -1")
    R.check(chains.aggregate([])["cases"] == 0, "aggregating nothing is safe")


def test_baseline_detects() -> None:
    print("[2] baseline recogniser")

    def bands(**moves: float) -> dict[str, float]:
        """A delta dict with every third-octave band at 0 except those named."""
        out = {f"spectrum.third_octave_db.{hz}hz.level_db": 0.0
               for hz in (50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500,
                          630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
                          5000, 6300, 8000, 10000, 12500, 16000)}
        for key, value in moves.items():
            out[f"spectrum.third_octave_db.{key.removeprefix('hz')}hz.level_db"] = value
        return out

    flat = bands()
    R.check(recover.recover_deterministic(flat) == [],
            "an unprocessed signal yields no nodes")

    # A bell.
    eq = bands(hz1000=6.0)
    R.check("parametric_eq" in recover.recover_deterministic(eq),
            "a moved band is recognised as EQ")

    # A BROADBAND LEVEL CHANGE IS NOT EQ. Every band moves together, so the
    # shape is unchanged; without removing the median first, every makeup-gain
    # change in every chain would be reported as a broadband EQ move.
    louder = {k: 6.0 for k in flat}
    R.check("parametric_eq" not in recover.recover_deterministic(louder),
            "a uniform level change is not mistaken for EQ")

    # Compression, via the envelope modulation spectrum rather than crest
    # factor — crest passed on synthetic material and failed on all six stems.
    comp = dict(flat)
    comp["dynamics.envelope_modulation.centroid_hz"] = 0.6
    R.check("compressor" in recover.recover_deterministic(comp),
            "a faster envelope is recognised as compression")

    # Drive: harmonics up top AND flattened peaks. Either alone is something
    # else — a high shelf lifts the top without touching dynamics.
    driven = bands(hz8000=9.0, hz10000=9.0, hz12500=9.0, hz16000=9.0)
    driven["dynamics.crest_factor_db"] = -3.0
    R.check("drive" in recover.recover_deterministic(driven),
            "top-end harmonics plus flattened peaks is drive")

    shelf = bands(hz8000=9.0, hz10000=9.0, hz12500=9.0, hz16000=9.0)
    shelf["dynamics.crest_factor_db"] = 0.0
    R.check("drive" not in recover.recover_deterministic(shelf),
            "the same top end without dynamics is a shelf, not drive")

    # Reverb and delay are detected INDEPENDENTLY. An earlier version chose
    # between them with an if/else, so a chain containing both could never be
    # recovered — the two-node space_only template scored 0.12 because the
    # recogniser was structurally unable to name more than one of them.
    both = dict(flat)
    both["dynamics.short_term_rms_percentiles.p10_dbfs"] = 14.0
    both["dynamics.short_term_rms_percentiles.p90_dbfs"] = -4.0
    both["hygiene.noise_floor_dbfs"] = 18.0
    detected = recover.recover_deterministic(both)
    R.check("plate_reverb" in detected and "delay" in detected,
            f"reverb and delay can both be detected at once, got {detected}")

    width = dict(flat)
    width["stereo.correlation"] = -0.3
    R.check("stereo_width" in recover.recover_deterministic(width),
            "a correlation change is recognised as width")

    # Output is always in conventional order, whatever order detection happened
    # in, so the ordering metric measures the baseline's guess rather than set
    # iteration order — which would otherwise vary between runs.
    everything = bands(hz1000=6.0, hz8000=9.0, hz10000=9.0, hz12500=9.0, hz16000=9.0)
    everything.update({
        "dynamics.crest_factor_db": -3.0,
        "dynamics.envelope_modulation.centroid_hz": 0.6,
        "stereo.correlation": -0.3,
    })
    out = recover.recover_deterministic(everything)
    R.check(out == [n for n in recover.CONVENTIONAL_ORDER if n in out],
            f"output follows the conventional order, got {out}")


def test_ordering_score_is_an_artifact() -> None:
    print("[3] the ordering number is a property of the corpus")

    # THE BASELINE HAS NO ORDERING INFORMATION AT ALL. It emits a fixed
    # sequence, so its ordering score is decided entirely by how often the
    # chains being scored happen to follow convention. On the six templates
    # that works out at about +0.5, which reads like partial ability and is
    # nothing of the kind.
    #
    # Pinned here because an unanchored metric that looks like skill is the
    # easiest possible thing to optimise against, and the plan lists exactly
    # that as a top risk.
    conventional = chains.score_recovery(
        recover.CONVENTIONAL_ORDER[:2], ["parametric_eq", "compressor"])
    against = chains.score_recovery(
        recover.CONVENTIONAL_ORDER[:2], ["compressor", "parametric_eq"])

    R.near(conventional.ordering, 1.0, 1e-9,
           "a conventionally-ordered chain scores +1 for free")
    R.near(against.ordering, -1.0, 1e-9,
           "and the same guess scores -1 when the chain is reversed")

    R.near(chains.aggregate([conventional, against])["ordering"], 0.0, 1e-9,
           "so a corpus with equal numbers of both scores 0, the true baseline")

    # Which means the corpus must contain both directions, or the metric
    # rewards a recogniser for knowing nothing.
    names = [t["name"] for t in chains.CHAIN_TEMPLATES]
    R.check("eq_then_compress" in names and "compress_then_eq" in names,
            "the corpus contains a chain and its reverse")


def test_templates_are_well_formed() -> None:
    print("[4] chain templates")

    context = SourceContext(channels=2, sample_rate=48000.0,
                            integrated_lufs=-18.0, correlation=0.4)
    seen = set()
    for template in chains.CHAIN_TEMPLATES:
        R.check(template["name"] not in seen, f"{template['name']} is unique")
        seen.add(template["name"])

        case = chains.build_chain(template, context)
        R.check(len(case.node_types) == len(template["nodes"]),
                f"{template['name']} records one type per node")
        R.check(all("type" in n for n in case.recipe.nodes),
                f"{template['name']} builds well-formed nodes")

        # Dynamics thresholds must be source-relative, for the same reason as
        # in the perturbation catalogue: a fixed threshold makes one chain mean
        # different things on a bass and an acoustic guitar.
        for node in case.recipe.nodes:
            if node["type"] == "compressor":
                threshold = node["params"]["threshold_dbfs"]
                R.near(threshold - context.integrated_lufs, -6.0, 1e-9,
                       f"{template['name']} sets the threshold below the source level")

    R.check(len(chains.CHAIN_TEMPLATES) >= 5,
            "the corpus has enough chains for the averages to mean something")


if __name__ == "__main__":
    print("Audionova chain recovery\n")
    test_scoring()
    test_baseline_detects()
    test_ordering_score_is_an_artifact()
    test_templates_are_well_formed()
    print(f"\n{R.checks} checks, {R.failures} failure(s)")
    sys.exit(1 if R.failures else 0)
