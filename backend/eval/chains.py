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

Parameter values are deliberately NOT scored here. The research is unambiguous
that parameter error is a bad quality proxy — the regression baseline in
LLM2Fx had the best parameter MAE and nearly the worst perceptual score — so
putting a number on it would invite optimising against it.

THE BASELINE EXISTS TO BE BEATEN. `recover_deterministic` is a plain rule-based
recogniser built from the sensitivity table. It is not meant to be good; it is
meant to make any future planner's score MEAN something. A model that scores
0.6 is only impressive against a floor, and without one there is no way to tell
a planner that reasons from a planner that guesses the common nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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


def score_recovery(predicted: list[str], actual: list[str]) -> RecoveryScore:
    """Precision, recall and ordering for one recovered chain.

    MULTISET semantics: a chain with two parametric_eq nodes is not recovered
    by proposing one. Treating the lists as sets would score the common
    "EQ, dynamics, EQ" sandwich as fully recovered by a single EQ, which is
    exactly the structure ordering is supposed to capture.
    """
    remaining = list(actual)
    hits = 0
    for node in predicted:
        if node in remaining:
            remaining.remove(node)
            hits += 1

    precision = hits / len(predicted) if predicted else 0.0
    recall = hits / len(actual) if actual else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # Ordering over the first occurrence of each type present in both.
    shared = [n for n in dict.fromkeys(predicted) if n in actual]
    ordering = _spearman(
        [predicted.index(n) for n in shared],
        [actual.index(n) for n in shared],
    )

    return RecoveryScore(precision, recall, f1, ordering, list(predicted), list(actual))


def aggregate(scores: list[RecoveryScore]) -> dict[str, float]:
    if not scores:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "ordering": float("nan"), "cases": 0}
    orderings = [s.ordering for s in scores if not np.isnan(s.ordering)]
    return {
        "precision": float(np.mean([s.precision for s in scores])),
        "recall": float(np.mean([s.recall for s in scores])),
        "f1": float(np.mean([s.f1 for s in scores])),
        "ordering": float(np.mean(orderings)) if orderings else float("nan"),
        "cases": len(scores),
    }
