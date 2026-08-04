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

The fix is asserted here too, in two halves, because either alone is not enough:

  THE CORPUS IS MIRRORED. Every chain scored for ordering has its exact reverse
  in the corpus, so a fixed-sequence predictor averages zero by construction.

  THE SCORE CARRIES AN ORDER-BLIND CONTROL. The same predicted multiset
  re-emitted in a fixed conventional sequence has identical precision and
  recall, so the excess over it is ordering information and nothing else.

A shuffled control is asserted to be EXACTLY ZERO, which is the reason it is not
the fix: excess over a control whose expectation is identically zero is the raw
number again.

THRESHOLDS ARE PROBED FROM BOTH SIDES AT A PLAUSIBLE DISTANCE. Mutation-testing
the Python stages found forty-two of seventy-two mutations surviving, and the
pattern was always the same: the check existed, the check fired, and the test
sat so far past the bar that the constant could move by half its value and
nothing noticed.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.eval import chains, generate, recover  # noqa: E402
from backend.eval.corpus import SourceContext  # noqa: E402
from backend.schemas.graph import PARAM_MODELS  # noqa: E402


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


def _bands(**moves: float) -> dict[str, float]:
    """A delta dict with every third-octave band at 0 except those named."""
    out = {f"spectrum.third_octave_db.{hz}hz.level_db": 0.0
           for hz in recover.SHAPE_BANDS}
    for key, value in moves.items():
        out[f"spectrum.third_octave_db.{key.removeprefix('hz')}hz.level_db"] = value
    return out


def test_evidence_floors_come_from_the_sweep() -> None:
    print("[5] detection floors are derived, not invented")

    # THE FLOOR IS THE SMALLEST AMOUNT THE SENSITIVITY SWEEP EVER APPLIED. Any
    # other number would be an assertion about detectability that nothing in
    # this project has measured, and the harness would then be scoring a
    # planner against a threshold somebody made up.
    R.near(generate.evidence_floor("mid_bell"), 5.0, 1e-9,
           "the bell floor is the smallest bell the sweep applied")
    R.near(generate.evidence_floor("high_shelf"), 4.0, 1e-9,
           "and the shelf floor is the smallest shelf")
    R.near(generate.evidence_floor("drive"), 6.0, 1e-9, "and 6 dB for drive")

    # Amounts measured from the neutral value, not from zero: 1:1 is the
    # compressor's no-op and 100% is width's, so scaling the raw parameter
    # would make the sub-threshold tier ask for a 0.06:1 ratio, which is not a
    # weak compressor but an illegal one.
    R.near(generate.evidence_floor("compression", relative_to=1.0), 1.0, 1e-9,
           "the compression floor is ratio minus one, not the ratio")
    R.near(generate.evidence_floor("stereo_width", relative_to=100.0), 50.0, 1e-9,
           "and width is measured from 100%, which is no change at all")

    # A name that is not in the catalogue is a HARD ERROR. Returning a default
    # would put an invented floor into the scoring silently, which is the exact
    # thing deriving it was meant to prevent.
    #
    # THE MESSAGE IS PART OF THE CONTRACT, not decoration, and asserting the
    # exception type alone does not test the check: the dictionary lookup two
    # lines later raises the same KeyError with a message that says only that a
    # string was missing. What the caller needs to be told is that there is no
    # evidence for a floor, which is a different fact.
    try:
        generate.evidence_floor("no_such_perturbation")
        R.check(False, "an unknown perturbation must not yield a floor")
    except KeyError as error:
        R.check("catalogue" in str(error),
                f"an unknown perturbation says why it cannot have a floor, "
                f"got {str(error)!r}")

    rng = random.Random(7)

    def amounts(tier: str) -> list[float]:
        return [generate._amount(rng, tier, 4.0) for _ in range(200)]

    strong, shown = amounts("strong"), amounts("demonstrated")
    borderline, faint = amounts("borderline"), amounts("sub_threshold")

    # THE TIERS MUST NOT OVERLAP, or "below the floor" and "above it" stop
    # meaning anything and the headline quietly includes nodes it excludes.
    R.check(min(strong) > max(shown), "strong sits entirely above demonstrated")
    R.check(min(shown) >= 4.0 - 1e-9,
            f"demonstrated never drops below the floor, min {min(shown):.3f}")
    R.check(max(borderline) < 4.0,
            f"borderline is entirely below it, max {max(borderline):.3f}")
    R.check(min(borderline) > max(faint),
            "and sub-threshold is below borderline")

    # The case the whole tier system exists for. A 0.2 dB shelf is not
    # recoverable by anyone; if this tier could produce 2 dB the corpus would be
    # excluding moves a planner could actually have found.
    R.check(max(faint) < 0.5,
            f"a sub-threshold shelf is under half a dB, max {max(faint):.3f}")
    R.check(min(faint) > 0.0, "but is still a real move, not a no-op")

    R.check(generate.IN_EVIDENCE_TIERS == ("strong", "demonstrated"),
            "only the tiers the sweep demonstrated count toward the headline")


def test_out_of_evidence_nodes_are_excluded_not_missed() -> None:
    print("[6] nodes below the floor are excluded from the headline")

    # NOT DETECTED AND NOT DETECTABLE ARE DIFFERENT FAILURES. Counting a 0.2 dB
    # shelf as a miss builds a ceiling into the harness that no planner can
    # raise, and a number that cannot improve is a number nobody acts on.
    missed = chains.score_recovery(
        ["compressor"], ["parametric_eq", "compressor"], ignored=["parametric_eq"])
    R.near(missed.recall, 1.0, 1e-9,
           "an undetectable node is not counted as a miss")
    R.near(missed.precision, 1.0, 1e-9, "and the rest is scored normally")
    R.check(missed.actual == ["compressor"],
            f"the truth it is scored against drops it, got {missed.actual}")

    # THE SAME IN THE OTHER DIRECTION. The planner cannot know the shelf was
    # there, so naming it is neither a success nor a false positive.
    named = chains.score_recovery(
        ["parametric_eq", "compressor"], ["parametric_eq", "compressor"],
        ignored=["parametric_eq"])
    R.near(named.precision, 1.0, 1e-9,
           "naming an undetectable node is not punished")
    R.near(named.recall, 1.0, 1e-9, "nor credited")
    R.check(named.named_ignored == ["parametric_eq"],
            f"but it is recorded, got {named.named_ignored}")

    # ONE PREDICTION CANCELS ONE NODE, not every prediction of that type. A
    # chain with a loud EQ and a faint one is still an EQ chain, and a planner
    # naming a single EQ has recovered the loud one.
    two_eqs = chains.score_recovery(
        ["parametric_eq"], ["parametric_eq", "parametric_eq"],
        ignored=["parametric_eq"])
    R.near(two_eqs.recall, 1.0, 1e-9,
           "one faint EQ beside a loud one leaves one node to recover")
    R.near(two_eqs.precision, 1.0, 1e-9, "and one prediction recovers it")

    # A false positive is still a false positive when nothing was ignored.
    wrong = chains.score_recovery(["drive"], ["compressor"])
    R.near(wrong.precision, 0.0, 1e-9, "an unrelated guess scores zero")


def test_ordering_control_and_corpus_balance() -> None:
    print("[7] the ordering artifact, and the two things that remove it")

    order = recover.CONVENTIONAL_ORDER

    # THE ORDER-BLIND CONTROL IS THE SAME MULTISET, RE-SORTED. Its precision and
    # recall are identical by construction, so anything the real prediction
    # beats it by is ordering information and nothing else.
    forward = chains.score_recovery(
        ["parametric_eq", "compressor"], ["parametric_eq", "compressor"],
        control_order=order)
    R.near(forward.ordering, 1.0, 1e-9, "a conventional chain scores +1")
    R.near(forward.ordering_control, 1.0, 1e-9,
           "and the order-blind control scores +1 too — for free")

    reverse = chains.score_recovery(
        ["parametric_eq", "compressor"], ["compressor", "parametric_eq"],
        control_order=order)
    R.near(reverse.ordering, -1.0, 1e-9, "the reversed chain scores -1")
    R.near(reverse.ordering_control, -1.0, 1e-9, "and so does the control")

    both = chains.aggregate([forward, reverse])
    R.near(both["ordering_excess"], 0.0, 1e-9,
           "so a fixed-sequence recogniser has EXACTLY zero excess, which is "
           "what +0.498 should have read")

    # A recogniser that actually reads order beats the control, and the excess
    # is what shows it. Both cases predict the truth's own order.
    reads_order = chains.aggregate([
        chains.score_recovery(["parametric_eq", "compressor"],
                              ["parametric_eq", "compressor"], control_order=order),
        chains.score_recovery(["compressor", "parametric_eq"],
                              ["compressor", "parametric_eq"], control_order=order),
    ])
    R.near(reads_order["ordering_excess"], 1.0, 1e-9,
           "a recogniser that reads order scores +1 excess over the control")
    R.near(reads_order["ordering"], 1.0, 1e-9, "on a raw score of +1")
    R.near(reads_order["ordering_control"], 0.0, 1e-9,
           "against a control that averages 0 on the mirrored pair")

    # A SHUFFLED CONTROL CANNOT FIX THIS, and the number says why rather than
    # leaving the next person to assume it can. Permutations come in reversing
    # pairs and reversing a permutation negates its rank correlation, so the
    # mean is identically zero and excess over it is the raw score again.
    R.near(reverse.ordering_shuffled, 0.0, 1e-12,
           "the shuffled control is exactly zero, not approximately")
    R.near(chains._shuffled_control(
        ["parametric_eq", "compressor", "drive"],
        ["drive", "parametric_eq", "compressor"]), 0.0, 1e-12,
        "still exactly zero at length three")

    # AND THE EXCESS MUST BE TAKEN OVER THE ORDER-BLIND CONTROL, NOT THAT ONE.
    # On a balanced corpus the two agree, so the difference only shows on an
    # UNBALANCED one — which is the case the whole metric exists for. A single
    # conventional chain is the smallest unbalanced corpus there is: raw +1,
    # order-blind control +1, shuffled 0. The honest excess is 0.
    lopsided = chains.aggregate([forward])
    R.near(lopsided["ordering"], 1.0, 1e-9,
           "one conventional chain scores +1 raw")
    R.near(lopsided["ordering_shuffled"], 0.0, 1e-12, "against a shuffled control of 0")
    R.near(lopsided["ordering_excess"], 0.0, 1e-9,
           "but zero excess, because the order-blind control scored +1 too — "
           "excess over the shuffled control would read +1.00 and be the "
           "+0.498 artifact all over again")

    # `aggregate` enforces the mirrored-only rule on its own, rather than
    # relying on `score_recovery` having already blanked the ordering. Two
    # places must agree for the metric to be safe, and a directly-constructed
    # score is how a future caller will reach this.
    smuggled = chains.RecoveryScore(
        1.0, 1.0, 1.0, 1.0, ["parametric_eq", "compressor"],
        ["parametric_eq", "compressor"], ordering_control=0.0,
        ordering_shuffled=0.0, ordering_scored=False)
    R.check(chains.aggregate([smuggled])["ordering_cases"] == 0,
            "an unmirrored score cannot smuggle an ordering figure into the mean")
    R.check(math.isnan(chains.aggregate([smuggled])["ordering"]),
            "and contributes no ordering at all")

    # Chains whose order has no mirror in the corpus are not scored at all.
    unmirrored = chains.score_recovery(
        ["parametric_eq", "compressor"], ["parametric_eq", "compressor"],
        control_order=order, ordering_scored=False)
    R.check(math.isnan(unmirrored.ordering),
            "an unmirrored chain contributes no ordering score")
    R.check(chains.aggregate([unmirrored])["ordering_cases"] == 0,
            "and is not counted in the ordering case count")
    R.near(unmirrored.f1, 1.0, 1e-9,
           "while still being scored for module recovery")

    # THE CONTROL AND THE SCORE ARE AVERAGED OVER THE SAME CASES. Two means over
    # different subsets are not comparable and their difference is not a
    # difference in ability.
    mixed = chains.aggregate([forward, reverse, unmirrored])
    R.check(mixed["ordering_cases"] == 2,
            f"only mirrored cases enter the ordering mean, got {mixed['ordering_cases']}")


def test_per_type_breakdown() -> None:
    print("[8] the per-node-type table")

    # "F1 0.58" HIDES WHETHER THE PLANNER FINDS COMPRESSION EVERY TIME AND
    # STEREO WIDTH NEVER. Those are different problems with different fixes.
    scores = [
        chains.score_recovery(["parametric_eq", "compressor"],
                              ["parametric_eq", "compressor"]),
        chains.score_recovery(["parametric_eq"], ["parametric_eq", "drive"]),
        chains.score_recovery(["parametric_eq", "delay"], ["parametric_eq"]),
    ]
    rows = {row.node_type: row for row in chains.per_type(scores)}

    R.check(rows["parametric_eq"].support == 3, "support counts every instance")
    R.near(rows["parametric_eq"].recall, 1.0, 1e-9, "EQ is found every time")
    R.near(rows["parametric_eq"].precision, 1.0, 1e-9, "and never over-proposed")

    R.check(rows["drive"].support == 1 and rows["drive"].hits == 0,
            "drive was present once and never found")
    R.near(rows["drive"].recall, 0.0, 1e-9, "so its recall is zero")
    R.check(math.isnan(rows["drive"].precision),
            "and its precision is undefined rather than zero — it was never "
            "proposed, which is a different failure from proposing it wrongly")

    R.check(rows["delay"].proposed == 1 and rows["delay"].support == 0,
            "delay was proposed once and never present")
    R.near(rows["delay"].precision, 0.0, 1e-9, "so its precision is zero")
    R.check(math.isnan(rows["delay"].recall), "and its recall is undefined")

    # MULTISET WITHIN A TYPE. Proposing one EQ against two does not recover both.
    doubled = chains.per_type([chains.score_recovery(
        ["parametric_eq"], ["parametric_eq", "parametric_eq"])])[0]
    R.check(doubled.support == 2 and doubled.hits == 1,
            f"two EQ nodes, one recovered, got {doubled.hits} of {doubled.support}")
    R.near(doubled.recall, 0.5, 1e-9, "which is half the recall")
    R.near(doubled.f1, 2 / 3, 1e-9, "and an F1 of 2/3")

    # Out-of-evidence instances are counted on their own column, never folded
    # into support.
    faint = chains.per_type([chains.score_recovery(
        ["compressor"], ["parametric_eq", "compressor"], ignored=["parametric_eq"])])
    eq_row = {r.node_type: r for r in faint}["parametric_eq"]
    R.check(eq_row.support == 0 and eq_row.ignored_support == 1,
            "a below-floor node counts as below-floor, not as support")

    R.check(chains.per_type([]) == [], "an empty run produces an empty table")


def test_parameter_recovery() -> None:
    print("[9] parameter recovery: where and how much")

    # A single moved band is read back at its own frequency and its own gain.
    single = recover.recover_eq_bands(_bands(hz1000=6.0))
    R.check(len(single) == 1, f"one move gives one band, got {len(single)}")
    R.near(single[0].center_freq_hz, 1000.0, 1.0, "at the band that moved")
    R.near(single[0].gain_db, 6.0, 1e-9, "with the gain that moved it")

    # A bell's skirt spreads over its neighbours, and the centroid must land in
    # the middle of it rather than on whichever band happened to be largest.
    bell = recover.recover_eq_bands(
        _bands(hz1600=2.0, hz2000=5.0, hz2500=6.0, hz3150=5.0, hz4000=2.0))
    R.check(len(bell) == 1, f"a skirt is one move, not five, got {len(bell)}")
    R.check(abs(math.log2(bell[0].center_freq_hz / 2500.0)) < 1 / 6,
            f"and is centred on the peak within a sixth of an octave, "
            f"got {bell[0].center_freq_hz:.0f} Hz")
    # THE GAIN IS THE PEAK OF THE RUN, NOT ITS MEAN. A bell's skirt drags the
    # mean well below the move that was actually made — this one averages 4.0
    # against a 6 dB peak — and reporting that as the recovered gain would
    # understate every EQ move by most of the skirt.
    R.near(bell[0].gain_db, 6.0, 0.2,
           "and reports the peak of the run rather than its mean")

    # THE CENTROID IS IN LOG FREQUENCY. The bands are logarithmically spaced, so
    # a linear centroid over a run from 1 kHz to 4 kHz lands at 2214 Hz where
    # the band actually halfway between them is 2000.
    flat_run = recover.recover_eq_bands(
        _bands(hz1000=4.0, hz1250=4.0, hz1600=4.0, hz2000=4.0, hz2500=4.0,
               hz3150=4.0, hz4000=4.0))
    R.near(flat_run[0].center_freq_hz, 2000.0, 30.0,
           "an even run is centred on its geometric middle, not its arithmetic one")

    # TWO MOVES OF OPPOSITE SIGN ARE TWO MOVES. Merging them by absolute value
    # would report one band halfway between at a gain neither of them has.
    two = recover.recover_eq_bands(_bands(hz200=5.0, hz5000=-7.0))
    R.check(len(two) == 2, f"a boost and a cut are two bands, got {len(two)}")
    R.check(two[0].gain_db < 0 and two[1].gain_db > 0,
            "largest first, and each keeps its own sign")
    R.near(two[0].center_freq_hz, 5000.0, 5.0, "the cut is at 5 kHz")
    R.near(two[1].center_freq_hz, 200.0, 1.0, "and the boost at 200 Hz")

    # THE HARD CASE FOR THE SIGN RULE: adjacent bands moving opposite ways. The
    # zeros between two distant moves separate them whatever the rule is, so
    # the case above passes even when sign is ignored entirely. Two neighbours
    # do not — merged, they report one band between them at a gain neither of
    # them has.
    adjacent = recover.recover_eq_bands(_bands(hz1000=5.0, hz1250=-5.0))
    R.check(len(adjacent) == 2,
            f"adjacent opposite moves stay two moves, got {len(adjacent)}")
    R.check(sorted(round(b.gain_db, 6) for b in adjacent) == [-5.0, 5.0],
            f"each keeping its own gain, got "
            f"{[round(b.gain_db, 2) for b in adjacent]}")

    # A UNIFORM LEVEL CHANGE IS NOT AN EQ MOVE, for the same reason it is not
    # detected as one: the shape is unchanged and only the level moved.
    louder = {k: 6.0 for k in _bands()}
    R.check(recover.recover_eq_bands(louder) == [],
            "a broadband level change yields no bands")

    # The threshold is the same one the type detector uses. Two thresholds for
    # one question is how a node gets detected on one band and described with
    # another.
    below = recover.recover_eq_bands(_bands(hz1000=1.4))
    R.check(below == [], "a move under the bar yields nothing")
    above = recover.recover_eq_bands(_bands(hz1000=1.6))
    R.check(len(above) == 1, "and a move just over it yields a band")
    R.check(("parametric_eq" in recover.recover_deterministic(_bands(hz1000=1.6)))
            and ("parametric_eq" not in recover.recover_deterministic(_bands(hz1000=1.4))),
            "which is exactly where the type detector's bar sits too")

    R.check(len(recover.recover_eq_bands(
        _bands(hz80=6.0, hz315=-6.0, hz1250=7.0, hz5000=-8.0), limit=2)) == 2,
        "the band count is capped: the report cannot resolve more moves than "
        "it has independent evidence for")


def test_parameter_scoring() -> None:
    print("[10] parameter scoring, in octaves and dB")

    truth = [chains.BandTruth(center_freq_hz=900.0, gain_db=-3.0, filter_type="bell")]

    exact = chains.match_bands([(900.0, -3.0)], truth)[0]
    R.near(exact.freq_error_octaves, 0.0, 1e-12, "an exact hit is zero octaves")
    R.near(exact.gain_error_db, 0.0, 1e-12, "and zero dB")

    # OCTAVES, NOT HERTZ. A 100 Hz error at 200 Hz and a 100 Hz error at 8 kHz
    # are not the same mistake, and a linear error would call the second one
    # small.
    octave_up = chains.match_bands([(1800.0, -3.0)], truth)[0]
    R.near(octave_up.freq_error_octaves, 1.0, 1e-12, "double the frequency is one octave")
    octave_down = chains.match_bands([(450.0, -3.0)], truth)[0]
    R.near(octave_down.freq_error_octaves, 1.0, 1e-12, "and so is half of it")

    R.near(chains.match_bands([(900.0, 1.0)], truth)[0].gain_error_db, 4.0, 1e-12,
           "gain error is signed distance in dB: +1 against -3 is 4 dB out")

    # MATCHING IS IN LOG FREQUENCY TOO, not only the error it reports. Against a
    # 1 kHz band, 400 Hz is 1.32 octaves away and 1900 Hz is 0.93 — but in
    # hertz 400 is the nearer of the two, so a linear match pairs the wrong
    # proposal and then reports a smaller error for it than the truth deserves.
    pair = chains.match_bands([(400.0, -3.0), (1900.0, -3.0)],
                              [chains.BandTruth(1000.0, -3.0, "bell")])[0]
    R.near(pair.recovered_hz, 1900.0, 1e-9,
           "the nearer proposal in octaves wins, not the nearer one in hertz")

    # ONE TO ONE. A recogniser proposing a single band cannot be credited with
    # recovering three.
    three = [chains.BandTruth(200.0, 6.0, "bell"),
             chains.BandTruth(1000.0, -5.0, "bell"),
             chains.BandTruth(8000.0, 4.0, "highshelf")]
    one_proposal = chains.match_bands([(1000.0, -5.0)], three)
    R.check(sum(1 for m in one_proposal if m.matched) == 1,
            "one proposal matches one applied band")
    R.check(len(one_proposal) == 3, "and the other two are recorded as unmatched")
    # The largest applied move is matched first, so a lone proposal is spent on
    # the band the report most clearly carries.
    R.check(one_proposal[0].truth.center_freq_hz == 200.0,
            "matching starts from the largest applied move")

    summary = chains.aggregate_bands(chains.match_bands(
        [(200.0, 6.0), (1000.0, -5.0), (8000.0, 4.0)], three))
    R.check(summary["bands"] == 3 and summary["matched"] == 3, "all three matched")
    R.near(summary["median_octaves"], 0.0, 1e-12, "with no frequency error")
    R.near(summary["within_third_octave"], 1.0, 1e-12, "all inside a third of an octave")

    # A third of an octave is one band of the report's own analysis, so the
    # tolerance must be probed either side of it rather than at a hit and a
    # miss an octave apart.
    inside = chains.match_bands([(1000.0 * 2 ** 0.30, -5.0)],
                                [chains.BandTruth(1000.0, -5.0, "bell")])
    outside = chains.match_bands([(1000.0 * 2 ** 0.36, -5.0)],
                                 [chains.BandTruth(1000.0, -5.0, "bell")])
    R.near(chains.aggregate_bands(inside)["within_third_octave"], 1.0, 1e-12,
           "0.30 octaves out is inside the tolerance")
    R.near(chains.aggregate_bands(outside)["within_third_octave"], 0.0, 1e-12,
           "0.36 octaves out is not")

    # The gain tolerance is probed the same way. Set at 12 dB it would pass
    # every recogniser in this file, including one that reported the sign
    # backwards on a 5 dB move.
    near_gain = chains.match_bands([(1000.0, -5.0 + 2.9)],
                                   [chains.BandTruth(1000.0, -5.0, "bell")])
    far_gain = chains.match_bands([(1000.0, -5.0 + 3.1)],
                                  [chains.BandTruth(1000.0, -5.0, "bell")])
    R.near(chains.aggregate_bands(near_gain)["within_gain_tolerance"], 1.0, 1e-12,
           "2.9 dB out is inside the gain tolerance")
    R.near(chains.aggregate_bands(far_gain)["within_gain_tolerance"], 0.0, 1e-12,
           "3.1 dB out is not")

    empty = chains.aggregate_bands([])
    R.check(math.isnan(empty["median_octaves"]),
            "no bands gives no median rather than a zero that reads as perfect")


def test_manifest_is_reproducible() -> None:
    print("[11] the committed manifest")

    # A SEED AND A COMMITTED MANIFEST, so a change in the headline is
    # attributable to the planner rather than to new random chains.
    first = generate.generate(generate.DEFAULT_SEED)
    second = generate.generate(generate.DEFAULT_SEED)
    R.check(generate.digest(first) == generate.digest(second),
            "the generator is deterministic in its seed")

    other = generate.generate(generate.DEFAULT_SEED + 1)
    R.check(generate.digest(other) != generate.digest(first),
            "and a different seed really does give a different corpus")

    committed = generate.verify_manifest()
    R.check(generate.digest(committed) == generate.digest(first),
            "the committed manifest is what this seed produces")
    R.check(len(committed) == len(first) and len(committed) >= 60,
            f"and the corpus is large enough to average over, {len(committed)} chains")

    # TWO CHECKS, AND EACH NEEDS ITS OWN CASE. The recorded digest catches a
    # manifest edited by hand; regenerating from the seed catches a generator
    # that changed underneath it. Testing only a chain edit exercises the second
    # and lets the first be deleted, because a tampered chain fails both.
    #
    # This one fails ONLY the digest check: the chains still regenerate exactly.
    only_digest = generate.load_manifest()
    only_digest["digest"] = "0" * 64
    with tempfile.TemporaryDirectory() as work:
        path = os.path.join(work, "wrong_digest.json")
        with open(path, "w") as handle:
            json.dump(only_digest, handle)
        try:
            generate.verify_manifest(path)
            R.check(False, "a manifest whose recorded digest is wrong must not verify")
        except generate.ManifestMismatch:
            R.check(True, "the manifest carries and checks its own digest")

    # And this one fails only the version check.
    wrong_version = generate.load_manifest()
    wrong_version["generator_version"] = "recovery_v0"
    with tempfile.TemporaryDirectory() as work:
        path = os.path.join(work, "old_version.json")
        with open(path, "w") as handle:
            json.dump(wrong_version, handle)
        try:
            generate.verify_manifest(path)
            R.check(False, "a manifest from another generator must not verify")
        except generate.ManifestMismatch:
            R.check(True, "the generator version is checked")

    # A MANIFEST THAT DOES NOT REGENERATE IS A HARD ERROR, never a silent
    # regeneration: a corpus that quietly reshapes itself is indistinguishable
    # from a planner that got worse.
    body = generate.load_manifest()
    body["chains"][0]["nodes"][0]["tier"] = "borderline"
    with tempfile.TemporaryDirectory() as work:
        path = os.path.join(work, "tampered.json")
        with open(path, "w") as handle:
            json.dump(body, handle)
        try:
            generate.verify_manifest(path)
            R.check(False, "a tampered manifest must not verify")
        except generate.ManifestMismatch:
            R.check(True, "a tampered manifest raises")

    # And a manifest whose digest was updated to match its tampered contents
    # still fails, because the seed no longer produces it.
    body["digest"] = generate.digest(
        [generate.ChainSpec.from_dict(c) for c in body["chains"]])
    with tempfile.TemporaryDirectory() as work:
        path = os.path.join(work, "resigned.json")
        with open(path, "w") as handle:
            json.dump(body, handle)
        try:
            generate.verify_manifest(path)
            R.check(False, "a re-signed tampered manifest must not verify either")
        except generate.ManifestMismatch:
            R.check(True, "re-signing does not help: the seed is checked too")


def test_generated_corpus_shape() -> None:
    print("[12] what the generated corpus covers")

    specs = generate.verify_manifest()
    by_name = {s.name: s for s in specs}

    # ORDERING CHAINS COME IN EXACT MIRRORED PAIRS. This is the property that
    # makes a fixed-sequence predictor score zero, and without it the ordering
    # figure measures how conventional the corpus is.
    eligible = [s for s in specs if s.ordering_eligible]
    R.check(len(eligible) >= 20,
            f"there are enough mirrored chains to average over, {len(eligible)}")
    for spec in eligible:
        R.check(spec.name.endswith("_fwd") or spec.name.endswith("_rev"),
                f"{spec.name} is half of a pair")
        twin = by_name.get(spec.name[:-4] + ("_rev" if spec.name.endswith("_fwd")
                                             else "_fwd"))
        R.check(twin is not None and twin.node_types == list(reversed(spec.node_types)),
                f"{spec.name} has its exact reverse in the corpus")
        R.check(twin is not None and spec.ordering_eligible == twin.ordering_eligible,
                f"{spec.name} and its twin are both scored, or neither is")

        # REPEATED TYPES BREAK THE MIRROR. Ordering is scored over the first
        # occurrence of each shared type, so reversing "eq, comp, eq" gives
        # "eq, comp, eq" back and the pair is not a pair at all.
        R.check(len(set(spec.node_types)) == len(spec.node_types),
                f"{spec.name} has no repeated node type")
        R.check(len(spec.nodes) >= 2, f"{spec.name} has an order to recover")

    # And nothing outside that family is scored for ordering.
    R.check(all(not s.ordering_eligible for s in specs if s.family != "ordering_pair"),
            "only the mirrored family is scored for ordering")

    lengths = {len(s.nodes) for s in specs}
    R.check(len(lengths) >= 4 and max(lengths) >= 5,
            f"chain length varies, {sorted(lengths)}")

    tiers = {n.tier for s in specs for n in s.nodes}
    R.check(tiers == set(generate.TIER_SCALE),
            f"every extremity tier appears, got {sorted(tiers)}")

    types = {n.node_type for s in specs for n in s.nodes}
    R.check(types == {m.node_type for m in generate.MOVES},
            f"every move in the catalogue appears, missing "
            f"{ {m.node_type for m in generate.MOVES} - types}")

    # A chain with two EQ nodes is the commonest structure there is, and a
    # set-based recogniser must not get it for free.
    R.check(any(s.node_types.count("parametric_eq") >= 2 for s in specs),
            "at least one chain repeats a node type")

    faint = [s for s in specs if s.family == "faint"]
    R.check(len(faint) >= 10, f"there are enough faint chains, {len(faint)}")
    R.check(any(any(n.in_evidence for n in s.nodes) for s in faint),
            "some faint chains also contain a clearly detectable node, which is "
            "what separates 'found nothing' from 'found what was there'")
    R.check(all(any(not n.in_evidence for n in s.nodes) for s in faint),
            "and every one of them contains something below the floor")

    # EVERY GENERATED NODE MUST BE SOMETHING THE ENGINE ACCEPTS. A corpus that
    # names a node type the schema does not have fails at render time, hundreds
    # of cases into a run.
    for spec in specs:
        for node in spec.nodes:
            R.check(node.node_type in PARAM_MODELS,
                    f"{spec.name}: {node.node_type} is a real node type")
            R.check(0.0 < node.mix <= 1.0, f"{spec.name}: mix is a fraction")


def test_generated_chains_build() -> None:
    print("[13] building a generated chain against a source")

    specs = generate.verify_manifest()
    stereo = SourceContext(channels=2, sample_rate=48000.0,
                           integrated_lufs=-18.0, correlation=0.4)
    mono = SourceContext(channels=1, sample_rate=48000.0,
                         integrated_lufs=-11.5, correlation=1.0)

    width_spec = next(s for s in specs if s.name == "single_stereo_width")

    # INAPPLICABLE IS NOT UNDETECTED. Width on a mono stem has no side signal to
    # act on and changes nothing, correctly. Scored as a miss it reports a hole
    # in the measurement layer that does not exist; the honest answer is that
    # the corpus has a gap.
    on_mono = chains.build_generated(width_spec, mono)
    R.check(on_mono.node_types == [], "width is not applied to a mono source")
    R.check(on_mono.inapplicable == ["stereo_width"], "and is reported as a gap")
    R.check(on_mono.ignored == [],
            "which is not the same as being below the floor")

    on_stereo = chains.build_generated(width_spec, stereo)
    R.check(on_stereo.node_types == ["stereo_width"], "and is applied to real stereo")

    # THRESHOLDS ARE SOURCE-RELATIVE, for the same reason as in the perturbation
    # catalogue: a fixed threshold makes one chain mean different things on a
    # bass and an acoustic guitar.
    compressed = next(s for s in specs
                      if any(n.move == "compression" for n in s.nodes))
    for context in (stereo, mono):
        case = chains.build_generated(compressed, context)
        for node in case.recipe.nodes:
            if node["type"] == "compressor":
                R.near(node["params"]["threshold_dbfs"] - context.integrated_lufs,
                       -6.0, 0.01,
                       "the threshold sits 6 dB below the source's own level")
                R.check("threshold_offset_db" not in node["params"],
                        "and the relative form does not reach the engine")

    # THE SPEC IS NOT MUTATED by building against a source. The forward and
    # reverse halves of an ordering pair share their node objects, so a build
    # that wrote into them would make the second source see the first source's
    # thresholds.
    R.check(all("threshold_dbfs" not in n.params for s in specs for n in s.nodes),
            "building leaves the manifest's parameters untouched")

    faint_spec = next(s for s in specs
                      if s.family == "faint" and len(s.nodes) == 1)
    faint_case = chains.build_generated(faint_spec, stereo)
    R.check(faint_case.ignored == faint_case.node_types,
            "a chain of nothing but faint nodes has nothing scorable in it")
    R.check(faint_case.eq_truth == [],
            "and contributes no parameter ground truth")

    # Parameter ground truth covers only the bands that HAVE a gain. A high-pass
    # has none, so there is nothing to compare a recovered bell against.
    highpass = next((s for s in specs
                     if any(n.move == "highpass" and n.in_evidence for n in s.nodes)),
                    None)
    if highpass is not None:
        case = chains.build_generated(highpass, stereo)
        R.check(all(b.filter_type != "highpass" for b in case.eq_truth),
                "a high-pass contributes no parameter ground truth")


def test_no_model_calls_are_possible() -> None:
    print("[14] the harness cannot make a model call")

    # THE CONSTRAINT IS STRUCTURAL, not a convention. There is no credit, and a
    # harness that can reach the API is a harness that will, on the run nobody
    # was watching.
    eval_dir = os.path.dirname(os.path.abspath(chains.__file__))
    for name in sorted(os.listdir(eval_dir)):
        if not name.endswith(".py"):
            continue
        with open(os.path.join(eval_dir, name)) as handle:
            source = handle.read()
        R.check("anthropic" not in source and "backend.plan.pipeline" not in source,
                f"{name} does not reach the model client")

    R.check(set(recover.RECOGNISERS) == {"rules"},
            f"only the rule-based recogniser is registered, got "
            f"{sorted(recover.RECOGNISERS)}")
    R.check(recover.resolve("rules") is recover.rule_based,
            "and it resolves to the deterministic floor")

    # AN UNKNOWN NAME IS A HARD ERROR, never a fall back to the rules. The
    # caller is a script whose whole output is a number; silently scoring a
    # different recogniser than the one asked for attributes the floor to the
    # wrong thing.
    try:
        recover.resolve("claude")
        R.check(False, "an unknown recogniser must not resolve")
    except recover.UnknownRecogniser:
        R.check(True, "an unknown recogniser is a hard error")

    # The seam a model planner drops into: a callable returning types and bands.
    prediction = recover.rule_based(_bands(hz1000=6.0))
    R.check(prediction.node_types == ["parametric_eq"],
            f"the seam returns node types, got {prediction.node_types}")
    R.check(len(prediction.eq_bands) == 1,
            "and the parameters that go with them")


if __name__ == "__main__":
    print("Audionova chain recovery\n")
    test_scoring()
    test_baseline_detects()
    test_ordering_score_is_an_artifact()
    test_templates_are_well_formed()
    test_evidence_floors_come_from_the_sweep()
    test_out_of_evidence_nodes_are_excluded_not_missed()
    test_ordering_control_and_corpus_balance()
    test_per_type_breakdown()
    test_parameter_recovery()
    test_parameter_scoring()
    test_manifest_is_reproducible()
    test_generated_corpus_shape()
    test_generated_chains_build()
    test_no_model_calls_are_possible()
    print(f"\n{R.checks} checks, {R.failures} failure(s)")
    sys.exit(1 if R.failures else 0)
