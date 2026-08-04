"""Multi-node chains with known ground truth, and how to score a recovery.

A SINGLE PERTURBATION IS THE EASY CASE. The sensitivity harness showed that one
effect at a time is readable off the report at r = 1.00, which is necessary but
says nothing about the actual problem: real chains have four or five nodes whose
effects overlap, mask each other, and in some orders cancel. A compressor after
a boost is not the same as a compressor before it, and a report that can see
each in isolation may not be able to see either in company.

WHAT IS SCORED, and why these two things.

  MODULE RECOVERY — which node types were applied. This is the question the
  planner's first stage answers, and it is scored as precision and recall
  rather than accuracy, because a recogniser that proposes every node type on
  every input would score 100% accuracy on the ones that were there.

  ORDERING — the sequence. LLM2Fx-Tools found two systems with near-identical
  parameter accuracy and ordering correlation 0.00 versus 0.56, and the ordered
  one won perceptually by a wide margin. Order is not an implementation detail
  and is scored separately.

  PARAMETERS, but only where the type was already recovered, and reported on
  their own line rather than folded into F1. "There was an EQ" and "there was a
  -3 dB bell near 900 Hz" are not the same claim and averaging them into one
  number hides which one was made. Frequency error is in OCTAVES because a
  100 Hz error at 200 Hz is a different mistake from a 100 Hz error at 8 kHz,
  and gain error is in dB because that is the unit the move was written in.

  Parameter error remains a BAD QUALITY PROXY and is not part of the headline.
  The regression baseline in LLM2Fx had the best parameter MAE and nearly the
  worst perceptual score, so a harness that ranked planners on it would rank
  them wrongly. What it is good for is diagnosis: a planner recovering every
  node type at random frequencies has not recovered the chain, and the type
  score alone cannot say so.

THE BASELINE EXISTS TO BE BEATEN. `recover_deterministic` is a plain rule-based
recogniser built from the sensitivity table. It is not meant to be good; it is
meant to make any future planner's score MEAN something. A model that scores
0.6 is only impressive against a floor, and without one there is no way to tell
a planner that reasons from a planner that guesses the common nodes.

CONTROLS, BECAUSE TWO OF THESE METRICS FLATTER A PREDICTOR THAT KNOWS NOTHING.

  A fixed-sequence predictor scored +0.498 on ordering, which read as partial
  ability and measured how conventional the corpus was. Every ordering figure
  here is therefore reported beside an ORDER-BLIND CONTROL: the same predicted
  multiset, re-emitted in a fixed conventional order. It has identical
  precision and recall by construction, so the difference between the two is
  ordering information and nothing else.

  A frequency-matching score flatters in the same way, because matching a
  proposal to its nearest ground-truth band means a proposal placed at random
  still lands somewhere. So the parameter score is reported beside a
  RANDOM-FREQUENCY CONTROL drawn from the same range.

  A SHUFFLED control is also computed, and it is worth being precise about what
  it does: the expected rank correlation of a random permutation is exactly
  zero by symmetry, so excess-over-shuffled is algebraically identical to the
  raw score. It cannot correct the artifact. It is reported anyway, as evidence
  that it cannot — the two fixes that do work are the order-blind control above
  and building the corpus so that every ordering-scored chain has an exact
  mirror in it.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from backend.engine.render import Recipe
from backend.eval.corpus import SourceContext

# Chains a planner might plausibly emit, in orders an engineer would accept.
# Hand-written rather than sampled: a random permutation of node types produces
# chains nobody would ever build (a limiter before an EQ, a reverb before a
# high-pass), and scoring recovery on those measures nothing useful.
CHAIN_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "name": "corrective_eq",
        "nodes": [
            ("parametric_eq", {"bands": [
                {"filter_type": "highpass", "center_freq_hz": 90.0, "gain_db": 0.0,
                 "q_factor": 0.707},
                {"filter_type": "bell", "center_freq_hz": 400.0, "gain_db": -4.0,
                 "q_factor": 1.6},
            ]}),
        ],
    },
    {
        "name": "eq_then_compress",
        "nodes": [
            ("parametric_eq", {"bands": [
                {"filter_type": "highshelf", "center_freq_hz": 5000.0,
                 "gain_db": 4.0, "q_factor": 0.707},
            ]}),
            ("compressor", {"ratio_to_1": 3.0, "knee_db": 6.0,
                            "attack_ms": 15.0, "release_ms": 140.0}),
        ],
    },
    {
        "name": "compress_then_eq",
        # The same two nodes in the other order. If the report cannot separate
        # these, ordering recovery is impossible in principle and the metric is
        # worth abandoning rather than optimising.
        "nodes": [
            ("compressor", {"ratio_to_1": 3.0, "knee_db": 6.0,
                            "attack_ms": 15.0, "release_ms": 140.0}),
            ("parametric_eq", {"bands": [
                {"filter_type": "highshelf", "center_freq_hz": 5000.0,
                 "gain_db": 4.0, "q_factor": 0.707},
            ]}),
        ],
    },
    {
        "name": "drive_and_shape",
        "nodes": [
            ("parametric_eq", {"bands": [
                {"filter_type": "highpass", "center_freq_hz": 80.0, "gain_db": 0.0,
                 "q_factor": 0.707},
            ]}),
            ("drive", {"drive_db": 10.0, "output_trim_db": -5.0}),
            ("parametric_eq", {"bands": [
                {"filter_type": "bell", "center_freq_hz": 2500.0, "gain_db": -3.0,
                 "q_factor": 1.2},
            ]}),
        ],
    },
    {
        "name": "vocal_chain",
        "nodes": [
            ("parametric_eq", {"bands": [
                {"filter_type": "highpass", "center_freq_hz": 110.0, "gain_db": 0.0,
                 "q_factor": 0.707},
                {"filter_type": "highshelf", "center_freq_hz": 8000.0,
                 "gain_db": 3.0, "q_factor": 0.707},
            ]}),
            ("compressor", {"ratio_to_1": 4.0, "knee_db": 8.0,
                            "attack_ms": 8.0, "release_ms": 110.0}),
            ("plate_reverb", {"decay_s": 1.6, "pre_delay_ms": 25.0,
                              "damping_hz": 7000.0}, 0.18),
        ],
    },
    {
        "name": "space_only",
        "nodes": [
            ("delay", {"time_ms": 280.0, "feedback_percent": 30.0}, 0.2),
            ("plate_reverb", {"decay_s": 2.2, "pre_delay_ms": 15.0,
                              "damping_hz": 5000.0}, 0.22),
        ],
    },
)


@dataclass
class ChainCase:
    name: str
    recipe: Recipe
    node_types: list[str]
    source: np.ndarray = field(repr=False, default=None)
    sample_rate: float = 48000.0
    family: str = "template"
    # Types present in the chain but excluded from the headline score: below
    # the sensitivity sweep's floor, or inapplicable to this source.
    ignored: list[str] = field(default_factory=list)
    # Node types dropped from the recipe entirely because they cannot act on
    # this material — stereo width on a mono stem. A CORPUS GAP, not a
    # measurement failure, and reported as one.
    inapplicable: list[str] = field(default_factory=list)
    eq_truth: list["BandTruth"] = field(default_factory=list)
    ordering_eligible: bool = False


def build_chain(
    template: dict[str, Any], context: SourceContext, *, seed: int = 3
) -> ChainCase:
    recipe = Recipe(seed=seed, output_sample_rate=context.sample_rate,
                    output_format="float64")
    node_types: list[str] = []

    for entry in template["nodes"]:
        node_type, params = entry[0], dict(entry[1])
        mix = entry[2] if len(entry) > 2 else 1.0

        # Dynamics thresholds are relative to the material, for the same reason
        # as in the perturbation catalogue: a fixed threshold makes the same
        # chain mean different things on a bass and an acoustic guitar.
        if node_type == "compressor":
            params.setdefault("threshold_dbfs", context.integrated_lufs - 6.0)

        node: dict[str, Any] = {"type": node_type, "params": params}
        if mix != 1.0:
            node["mix"] = mix
        recipe.nodes.append(node)
        node_types.append(node_type)

    return ChainCase(name=template["name"], recipe=recipe, node_types=node_types)


def build_generated(spec: Any, context: SourceContext, *, seed: int = 3) -> ChainCase:
    """Turn a generated `ChainSpec` into a renderable case with its ground truth.

    Two things happen here that cannot happen at generation time, because both
    depend on the material rather than on the chain:

      THRESHOLDS BECOME ABSOLUTE. The spec carries `threshold_offset_db`; a
      compressor at a fixed -28 dBFS crushes a bass at -11.5 LUFS and barely
      touches a guitar at -22.6, so the same spec on two sources would be two
      different experiments.

      INAPPLICABLE NODES ARE DROPPED. Stereo width on a mono stem has no side
      signal to act on and changes nothing — correctly. Leaving it in the truth
      reports a hole in the measurement layer that does not exist; the right
      answer is that the CORPUS has a gap, and it is counted as one.
    """
    # Imported here rather than at module scope: the scorer is the lower layer
    # and must not depend on the generator, so that a corpus written by hand,
    # by the generator or by a future one all score through the same code.
    from backend.eval.generate import MOVES_BY_NAME

    recipe = Recipe(seed=seed, output_sample_rate=context.sample_rate,
                    output_format="float64")
    node_types: list[str] = []
    ignored: list[str] = []
    inapplicable: list[str] = []
    eq_truth: list[BandTruth] = []

    for node_spec in spec.nodes:
        move = MOVES_BY_NAME[node_spec.move]
        if move.needs_stereo and not (context.channels >= 2
                                      and context.correlation < 0.99):
            inapplicable.append(node_spec.node_type)
            continue

        params = {k: v for k, v in node_spec.params.items()}
        offset = params.pop("threshold_offset_db", None)
        if offset is not None:
            params["threshold_dbfs"] = round(context.integrated_lufs + offset, 2)

        node: dict[str, Any] = {"type": node_spec.node_type, "params": params}
        if node_spec.mix != 1.0:
            node["mix"] = node_spec.mix
        recipe.nodes.append(node)
        node_types.append(node_spec.node_type)

        if not node_spec.in_evidence:
            ignored.append(node_spec.node_type)
            continue

        for band in params.get("bands", []):
            # High-pass and low-pass have no gain, so there is no gain error to
            # report and no centre to compare a recovered bell against. Counted
            # in the type score, absent from the parameter score.
            if band["filter_type"] in ("bell", "lowshelf", "highshelf"):
                eq_truth.append(BandTruth(
                    center_freq_hz=float(band["center_freq_hz"]),
                    gain_db=float(band["gain_db"]),
                    filter_type=band["filter_type"]))

    return ChainCase(name=spec.name, recipe=recipe, node_types=node_types,
                     sample_rate=context.sample_rate, family=spec.family,
                     ignored=ignored, inapplicable=inapplicable,
                     eq_truth=eq_truth,
                     ordering_eligible=bool(spec.ordering_eligible))


# ---------------------------------------------------------------------------
# Scoring


@dataclass
class RecoveryScore:
    precision: float
    recall: float
    f1: float
    ordering: float          # Spearman rho, or nan when undefined
    predicted: list[str]
    actual: list[str]
    # Node types that were in the chain but are NOT scored: either below the
    # sensitivity sweep's floor, or inapplicable to this source. Removed from
    # the truth, and one matching prediction removed with them, so naming them
    # is neither credited nor punished.
    ignored: list[str] = field(default_factory=list)
    # Predictions cancelled against an ignored node — the planner named a type
    # that was present only sub-threshold. Recorded without a verdict, because
    # it is indistinguishable from a false positive that happened to land.
    named_ignored: list[str] = field(default_factory=list)
    # The same predicted multiset re-emitted in a fixed conventional order.
    # Identical precision and recall by construction, so the gap between this
    # and `ordering` is ordering information and nothing else.
    ordering_control: float = float("nan")
    ordering_shuffled: float = float("nan")
    ordering_scored: bool = True
    name: str = ""


def _spearman(a: list[int], b: list[int]) -> float:
    """Rank correlation over the positions of the nodes both lists contain.

    Computed on the INTERSECTION. Scoring order over nodes the recogniser
    never proposed would conflate two different failures — missing a node and
    misplacing it — and the module score already covers the first.
    """
    if len(a) < 2:
        return float("nan")
    x = np.array(a, dtype=float)
    y = np.array(b, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= 1e-30:
        return float("nan")
    return float(np.clip(np.sum(x * y) / denom, -1.0, 1.0))


def _ordering_of(predicted: list[str], actual: list[str]) -> float:
    """Rank correlation over the first occurrence of each shared type.

    THE ORDER OF `shared` IS IRRELEVANT and deliberately not relied on: the
    correlation is taken over paired index lists, and permuting both lists the
    same way leaves it unchanged. Mutating this line to a sorted set produces
    identical output on every input, which is worth writing down so the next
    person to notice does not go looking for the test that should have caught
    it. What `dict.fromkeys` buys is a deterministic intermediate — set
    iteration over strings varies with the hash seed — not a different answer.
    """
    shared = [n for n in dict.fromkeys(predicted) if n in actual]
    return _spearman(
        [predicted.index(n) for n in shared],
        [actual.index(n) for n in shared],
    )


# How many permutations the shuffled control samples before it stops
# enumerating. Six distinct types is 720 orderings, which is instant; beyond
# that the exact answer and the sampled one agree to well past the precision
# anything is reported at.
_MAX_ENUMERATED_PERMUTATIONS = 720


def _shuffled_control(predicted: list[str], actual: list[str]) -> float:
    """Mean ordering score over every permutation of the same prediction.

    THIS IS EXACTLY ZERO whenever it is defined, and that is the finding rather
    than a bug: permutations come in reversing pairs, and reversing a
    permutation negates its rank correlation. So a shuffled control cannot
    rescue the ordering metric — excess over it is the raw score again. It is
    computed and printed so that the next person to propose it as the fix can
    see the number instead of assuming.
    """
    if len(predicted) < 2:
        return float("nan")
    orderings: list[float] = []
    for count, permutation in enumerate(itertools.permutations(predicted)):
        if count >= _MAX_ENUMERATED_PERMUTATIONS:
            break
        value = _ordering_of(list(permutation), actual)
        if not math.isnan(value):
            orderings.append(value)
    return float(np.mean(orderings)) if orderings else float("nan")


def _cancel_ignored(
    predicted: list[str], actual: list[str], ignored: Sequence[str]
) -> tuple[list[str], list[str], list[str]]:
    """Remove out-of-evidence nodes from the truth, and one prediction each.

    NOT APPLICABLE IS NOT NOT DETECTED, and neither is BELOW THE FLOOR. A 0.2 dB
    shelf is not recoverable by anyone; counting it as a miss builds a ceiling
    into the harness that no planner can raise, and a number that cannot improve
    is a number nobody acts on. Counting a prediction of it as a false positive
    is the same mistake in the other direction — the planner cannot know the
    shelf was there, so it cannot be blamed either way.

    So the node leaves the truth, and a prediction of that type leaves with it —
    but ONLY A SURPLUS ONE. A chain with a loud EQ and a faint one is still an
    EQ chain, and a planner naming a single EQ has recovered the loud one.
    Cancelling that prediction against the faint node instead would score a
    correct answer as zero, which is how a fairness adjustment turns into a new
    unfairness pointing the other way.
    """
    remaining_actual = list(actual)
    for node in ignored:
        if node in remaining_actual:
            remaining_actual.remove(node)

    scorable = Counter(remaining_actual)
    proposals = Counter(predicted)
    budget = Counter(ignored)

    named: list[str] = []
    remaining_predicted = list(predicted)
    for node_type, count in proposals.items():
        surplus = min(count - scorable.get(node_type, 0), budget.get(node_type, 0))
        for _ in range(max(surplus, 0)):
            remaining_predicted.remove(node_type)
            named.append(node_type)

    return remaining_predicted, remaining_actual, named


def score_recovery(
    predicted: list[str],
    actual: list[str],
    *,
    ignored: Sequence[str] = (),
    control_order: Sequence[str] | None = None,
    ordering_scored: bool = True,
    name: str = "",
) -> RecoveryScore:
    """Precision, recall, ordering and its controls for one recovered chain.

    MULTISET semantics: a chain with two parametric_eq nodes is not recovered
    by proposing one. Treating the lists as sets would score the common
    "EQ, dynamics, EQ" sandwich as fully recovered by a single EQ, which is
    exactly the structure ordering is supposed to capture.

    `ignored` names the nodes that are in the chain but out of evidence — below
    the sweep's floor, or inapplicable to this source. `control_order` supplies
    the fixed sequence the order-blind control emits.

    `ordering_scored` is False for chains whose ground-truth order has no mirror
    in the corpus. Scoring those rewards a fixed-sequence predictor for the
    corpus's own habits, which is precisely the +0.498 artifact.
    """
    scored_predicted, scored_actual, named = _cancel_ignored(
        predicted, actual, ignored)

    remaining = list(scored_actual)
    hits = 0
    for node in scored_predicted:
        if node in remaining:
            remaining.remove(node)
            hits += 1

    precision = hits / len(scored_predicted) if scored_predicted else 0.0
    recall = hits / len(scored_actual) if scored_actual else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    if ordering_scored:
        ordering = _ordering_of(scored_predicted, scored_actual)
        shuffled = _shuffled_control(scored_predicted, scored_actual)
        if control_order is None:
            control = float("nan")
        else:
            # THE SAME MULTISET, RE-SORTED. Identical precision and recall by
            # construction, so anything by which the real prediction beats it
            # is ordering information rather than detection ability.
            blind = sorted(scored_predicted,
                           key=lambda n: (control_order.index(n)
                                          if n in control_order else len(control_order)))
            control = _ordering_of(blind, scored_actual)
    else:
        ordering = control = shuffled = float("nan")

    return RecoveryScore(
        precision, recall, f1, ordering,
        list(scored_predicted), list(scored_actual),
        ignored=list(ignored), named_ignored=named,
        ordering_control=control, ordering_shuffled=shuffled,
        ordering_scored=ordering_scored, name=name,
    )


def aggregate(scores: list[RecoveryScore]) -> dict[str, float]:
    if not scores:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "ordering": float("nan"), "ordering_control": float("nan"),
                "ordering_shuffled": float("nan"), "ordering_excess": float("nan"),
                "ordering_cases": 0, "cases": 0}

    def mean_defined(values: list[float]) -> float:
        kept = [v for v in values if not np.isnan(v)]
        return float(np.mean(kept)) if kept else float("nan")

    defined = [s for s in scores if s.ordering_scored and not np.isnan(s.ordering)]

    # THE EXCESS IS A PER-CASE DIFFERENCE, not the gap between two means taken
    # over whatever subset each happened to be defined on. Those two subsets are
    # not the same set, and the difference between means over different sets is
    # not a difference in ability.
    paired = [s for s in defined if not np.isnan(s.ordering_control)]

    return {
        "precision": float(np.mean([s.precision for s in scores])),
        "recall": float(np.mean([s.recall for s in scores])),
        "f1": float(np.mean([s.f1 for s in scores])),
        "ordering": mean_defined([s.ordering for s in defined]),
        "ordering_control": mean_defined([s.ordering_control for s in paired]),
        "ordering_shuffled": mean_defined([s.ordering_shuffled for s in paired]),
        "ordering_excess": mean_defined(
            [s.ordering - s.ordering_control for s in paired]),
        "ordering_cases": len(defined),
        "cases": len(scores),
    }


# ---------------------------------------------------------------------------
# Per-node-type breakdown
#
# "F1 0.58" hides whether the planner reliably finds compression and never finds
# stereo width. Those are different problems with different fixes — one is a
# threshold, the other is a corpus gap — and a single number cannot distinguish
# them. This is the table that makes the headline actionable.


@dataclass
class TypeBreakdown:
    node_type: str
    support: int          # in-evidence instances in the ground truth
    proposed: int         # instances the recogniser named
    hits: int
    ignored_support: int  # instances present but out of evidence
    named_ignored: int    # times it named a type that was only out of evidence

    @property
    def recall(self) -> float:
        return self.hits / self.support if self.support else float("nan")

    @property
    def precision(self) -> float:
        return self.hits / self.proposed if self.proposed else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if np.isnan(p) or np.isnan(r) or (p + r) == 0.0:
            return float("nan")
        return 2 * p * r / (p + r)


def per_type(scores: list[RecoveryScore]) -> list[TypeBreakdown]:
    support: Counter[str] = Counter()
    proposed: Counter[str] = Counter()
    hits: Counter[str] = Counter()
    ignored_support: Counter[str] = Counter()
    named_ignored: Counter[str] = Counter()

    for score in scores:
        actual = Counter(score.actual)
        predicted = Counter(score.predicted)
        support.update(actual)
        proposed.update(predicted)
        for node_type in actual.keys() | predicted.keys():
            hits[node_type] += min(actual[node_type], predicted[node_type])
        ignored_support.update(score.ignored)
        named_ignored.update(score.named_ignored)

    types = (set(support) | set(proposed) | set(ignored_support))
    return sorted(
        (TypeBreakdown(node_type=t, support=support[t], proposed=proposed[t],
                       hits=hits[t], ignored_support=ignored_support[t],
                       named_ignored=named_ignored[t])
         for t in types),
        key=lambda b: (-b.support, b.node_type),
    )


# ---------------------------------------------------------------------------
# Parameter-level scoring


@dataclass
class BandTruth:
    """One EQ band that was actually applied, and is in evidence."""

    center_freq_hz: float
    gain_db: float
    filter_type: str


@dataclass
class BandMatch:
    truth: BandTruth
    recovered_hz: float | None
    recovered_gain_db: float | None

    @property
    def matched(self) -> bool:
        return self.recovered_hz is not None

    @property
    def freq_error_octaves(self) -> float:
        if self.recovered_hz is None or self.recovered_hz <= 0.0:
            return float("nan")
        return abs(math.log2(self.recovered_hz / self.truth.center_freq_hz))

    @property
    def gain_error_db(self) -> float:
        if self.recovered_gain_db is None:
            return float("nan")
        return abs(self.recovered_gain_db - self.truth.gain_db)


def match_bands(recovered: list[tuple[float, float]],
                truth: list[BandTruth]) -> list[BandMatch]:
    """Pair recovered bands to applied ones, largest applied move first.

    Greedy on |gain| rather than on distance, because the largest move is the
    one the report most clearly carries and the one a planner is most likely to
    have been looking at. One-to-one: a recogniser proposing a single band
    cannot be credited with recovering three.

    NEAREST-NEIGHBOUR MATCHING FLATTERS, which is why the runner scores a
    random-frequency control through this same function. A proposal placed
    anywhere lands near something; the control says how near that is by luck.
    """
    available = list(recovered)
    matches: list[BandMatch] = []
    for band in sorted(truth, key=lambda b: -abs(b.gain_db)):
        if not available:
            matches.append(BandMatch(band, None, None))
            continue
        best = min(range(len(available)),
                   key=lambda i: abs(math.log2(max(available[i][0], 1e-9)
                                               / band.center_freq_hz)))
        hz, gain = available.pop(best)
        matches.append(BandMatch(band, hz, gain))
    return matches


# A third of an octave is one band of the report's own third-octave analysis, so
# a proposal inside it is indistinguishable from the truth in the document the
# planner was handed. Tighter would be scoring below the resolution of the
# evidence.
FREQ_TOLERANCE_OCTAVES = 1.0 / 3.0

# The smallest EQ move the sensitivity sweep ever applied was 4 dB, and the
# recogniser reads gain off third-octave band deltas whose neighbours overlap.
# 3 dB is deliberately generous; the median error is the number to read.
GAIN_TOLERANCE_DB = 3.0


def aggregate_bands(matches: list[BandMatch]) -> dict[str, float]:
    freq = [m.freq_error_octaves for m in matches
            if m.matched and not np.isnan(m.freq_error_octaves)]
    gain = [m.gain_error_db for m in matches
            if m.matched and not np.isnan(m.gain_error_db)]
    return {
        "bands": len(matches),
        "matched": sum(1 for m in matches if m.matched),
        "median_octaves": float(np.median(freq)) if freq else float("nan"),
        "median_gain_db": float(np.median(gain)) if gain else float("nan"),
        "within_third_octave": (float(np.mean([f <= FREQ_TOLERANCE_OCTAVES
                                               for f in freq])) if freq
                                else float("nan")),
        "within_gain_tolerance": (float(np.mean([g <= GAIN_TOLERANCE_DB
                                                 for g in gain])) if gain
                                  else float("nan")),
    }
