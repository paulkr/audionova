"""The quantifier ladder and the patch operations for multi-turn refinement.

THE ARITHMETIC IN THIS FILE MUST NEVER BE DONE BY A MODEL. When a user says
"slightly more delay", the size of "slightly" has to be the same number every
time — on Tuesday, on a different stem, in a different session, at a different
temperature. A model asked to apply "slightly" will produce 1.2 dB one turn and
4 dB the next, and the user will conclude the tool is unpredictable rather than
that they were unclear.

So the model's only job is to name a target and a DIRECTION AND MAGNITUDE WORD
from a closed set. Python decides what that word means.

MULTI-TURN IS A PATCH, NOT A REGENERATION, and this is the more important half
of the design. Asking the planner to produce a whole new graph for "add a bit
more air" discards every decision the user implicitly accepted by not
complaining about it, and churns unrelated parameters. To the user that reads as
a bug: they asked for one change and six things moved.

So a refinement is a list of operations against stable node ids, applied
deterministically. The graph the user was listening to is the graph they keep,
minus exactly what they asked to change.

STEP SIZES ARE PERCEPTUAL, NOT ARITHMETIC. A step is "the smallest change a
competent listener reliably notices on this parameter", which is why they are not
all the same shape:

  - EQ gain steps ADD 1.5 dB, because dB is already a perceptual scale
  - frequency steps MULTIPLY by 1.26, one third of an octave, because pitch is
    logarithmic and "move it up a bit" means a ratio
  - mix and feedback steps MULTIPLY, because doubling a small mix is audible and
    adding 0.1 to a large one is not
  - a threshold step of -2.5 dB is larger than an EQ step, because a compressor
    threshold has to move further before the gain reduction changes audibly
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from backend.schemas.graph import EffectGraph, GraphNode

# How many steps each magnitude word is worth. "slightly" and "a bit" are
# deliberately the same: users treat them as synonyms and pretending to
# distinguish them would be inventing precision the language does not carry.
QUANTIFIERS: dict[str, int] = {
    "slightly": 1,
    "a_bit": 1,
    "somewhat": 1,
    "noticeably": 2,
    "moderately": 2,
    "much": 3,
    "a_lot": 3,
    "significantly": 3,
    "dramatically": 4,
}

Quantifier = Literal[
    "slightly", "a_bit", "somewhat", "noticeably", "moderately",
    "much", "a_lot", "significantly", "dramatically",
]


@dataclass(frozen=True)
class Step:
    """One perceptual step on a parameter.

    Either additive (`delta`) or multiplicative (`factor`), never both. The two
    forms exist because parameters live on different scales, not for convenience.
    """

    delta: float = 0.0
    factor: float = 1.0

    def apply(self, value: float, steps: int) -> float:
        if self.factor != 1.0:
            return value * (self.factor ** steps)
        return value + self.delta * steps


# One step per parameter. These are the numbers the whole refinement loop rests
# on, so each is a judgement about audibility rather than a round number.
STEPS: dict[str, Step] = {
    # Tonal. dB is already perceptual, so these add.
    "gain_db": Step(delta=1.5),
    "static_gain_db": Step(delta=1.5),
    "makeup_gain_db": Step(delta=1.5),
    "output_trim_db": Step(delta=1.5),
    "drive_db": Step(delta=2.0),

    # Frequency. Pitch is logarithmic, so these multiply. 1.26 is one third of
    # an octave — the smallest move that reads as "a different frequency"
    # rather than as the same band slightly mis-set.
    "center_freq_hz": Step(factor=1.26),
    "cutoff_hz": Step(factor=1.26),
    "hf_damp_hz": Step(factor=1.26),
    "lf_damp_hz": Step(factor=1.26),
    "damping_hz": Step(factor=1.26),
    "bandwidth_hz": Step(factor=1.26),
    "lf_cut_hz": Step(factor=1.26),
    "sidechain_hp_hz": Step(factor=1.26),
    "bass_mono_below_hz": Step(factor=1.26),

    # Q. Multiplicative, because going from 1 to 2 is a large change and from
    # 8 to 9 is not.
    "q_factor": Step(factor=1.4),

    # Dynamics. A threshold has to move further than an EQ band before the
    # gain reduction changes audibly.
    "threshold_dbfs": Step(delta=-2.5),
    "ratio_to_1": Step(factor=1.35),
    "attack_ms": Step(factor=1.6),
    "release_ms": Step(factor=1.5),
    "knee_db": Step(delta=3.0),
    "range_db": Step(delta=6.0),
    "hold_ms": Step(factor=1.5),

    # Time and space.
    "time_ms": Step(factor=1.35),
    "feedback_percent": Step(factor=1.4),
    "decay_s": Step(factor=1.25),
    "pre_delay_ms": Step(delta=10.0),
    "spread_ms": Step(delta=8.0),
    "modulation_depth": Step(delta=0.15),
    "size": Step(factor=1.15),

    # Modulation.
    "rate_hz": Step(factor=1.4),
    "depth_ms": Step(factor=1.4),
    "voices": Step(delta=1.0),

    # Stereo.
    "width_percent": Step(delta=25.0),
    "bass_width_percent": Step(delta=25.0),
    "stereo_link": Step(delta=0.2),

    # Master.
    "ceiling_dbtp": Step(delta=-0.5),
    "safety_margin_db": Step(delta=0.5),

    # Wet/dry. Multiplicative: doubling a 0.1 mix is obvious, adding 0.1 to a
    # 0.9 mix is not.
    "mix": Step(factor=1.4),
}


# WHAT "MORE X" MEANS, resolved here rather than by the model.
#
# These are the ambiguities that would otherwise be settled differently on
# different turns. "More delay" is the clearest case: it almost always means a
# louder delay, not a longer one, and a planner that guessed "longer" would move
# the repeats off the beat.
INTENT_TO_PARAM: dict[str, tuple[str, str]] = {
    # phrase -> (node type, parameter)
    "more_delay": ("delay", "mix"),
    "more_repeats": ("delay", "feedback_percent"),
    "longer_delay": ("delay", "time_ms"),
    "darker_delay": ("delay", "hf_damp_hz"),
    "more_reverb": ("plate_reverb", "mix"),
    "longer_reverb": ("plate_reverb", "decay_s"),
    "darker_reverb": ("plate_reverb", "damping_hz"),
    "more_compression": ("compressor", "ratio_to_1"),
    "harder_compression": ("compressor", "threshold_dbfs"),
    "faster_compression": ("compressor", "attack_ms"),
    "more_drive": ("drive", "drive_db"),
    "wider": ("stereo_width", "width_percent"),
    "louder": ("gain", "gain_db"),
}

# Which direction a "more" means for each. Not all of them increase the number:
# a lower threshold means MORE compression, and a lower damping frequency means
# a DARKER reverb.
INTENT_SIGN: dict[str, int] = {
    "more_delay": +1,
    "more_repeats": +1,
    "longer_delay": +1,
    "darker_delay": -1,
    "more_reverb": +1,
    "longer_reverb": +1,
    "darker_reverb": -1,
    "more_compression": +1,
    "harder_compression": +1,   # the step itself is negative
    "faster_compression": -1,
    "more_drive": +1,
    "wider": +1,
    "louder": +1,
}


@dataclass
class Patch:
    """One operation against a graph. Eight kinds, closed set.

    Closed because an open-ended "edit" operation would put the model back in
    charge of arithmetic, which is the thing this file exists to prevent.
    """

    op: Literal[
        "adjust_param", "set_param", "add_node", "remove_node",
        "bypass_node", "enable_node", "add_band", "remove_band",
    ]
    node_id: str = ""
    param: str = ""
    quantifier: Quantifier | None = None
    # An amount the user named, as an alternative to a ladder word. Exactly one
    # of quantifier / percent / delta is meaningful; the model picks which the
    # sentence contained and never computes the result.
    percent: float | None = None
    delta: float | None = None
    direction: Literal["increase", "decrease"] = "increase"
    value: Any = None
    band_index: int | None = None
    node_type: str = ""
    after: str = ""              # node id to insert after; empty means append
    reason: str = ""


@dataclass
class PatchResult:
    graph: EffectGraph
    applied: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.rejected


def steps_for(quantifier: Quantifier | None) -> int:
    """How many steps a magnitude word is worth. Unknown words get one.

    One rather than zero, because a user who typed an unrecognised intensifier
    still asked for a change, and doing nothing looks broken while doing the
    smallest thing looks conservative.
    """
    if quantifier is None:
        return 1
    return QUANTIFIERS.get(quantifier, 1)


def adjust(param: str, value: float, quantifier: Quantifier | None,
           direction: str) -> float:
    """Move one parameter by a number of perceptual steps.

    Raises for a parameter with no declared step, rather than guessing. A step
    size is a judgement about audibility and inventing one at call time would
    make the same word mean different things on different parameters.
    """
    step = STEPS.get(param)
    if step is None:
        raise KeyError(
            f"no perceptual step declared for {param!r} — add one to STEPS "
            "rather than letting the caller invent a size"
        )

    count = steps_for(quantifier)
    if direction == "decrease":
        count = -count

    return step.apply(value, count)


class AmountError(ValueError):
    """An amount that cannot be applied to this parameter, with the reason."""


def adjust_by_amount(param: str, value: float, *,
                     percent: float | None = None,
                     delta: float | None = None,
                     direction: str = "increase") -> float:
    """Move one parameter by an amount the USER named, not a ladder step.

    "Decrease the delay by 15%" is a different request from "shorten the delay
    a bit", and collapsing them loses what the user actually said. The ladder
    answers the second; this answers the first.

    THE MODEL STILL DOES NO ARITHMETIC. It extracts the number 15 and the word
    "percent" from the sentence; every multiplication happens here. That is the
    same invariant the ladder rests on, and the reason a percentage is not
    simply handed to the model to resolve.

    A PERCENTAGE OF A DECIBEL IS NOT A THING, and this refuses rather than
    guessing. dB is already logarithmic — "15% more" of a 4 dB boost could mean
    4.6 dB (a ratio of the number) or a 15% louder signal (1.2 dB more), and
    those are different requests. Worse, at 0 dB every ratio is 0, so the
    obvious reading silently does nothing on the commonest starting value.
    Which form applies is decided by the parameter's own step: multiplicative
    parameters take percentages, additive ones take amounts in their unit.
    """
    step = STEPS.get(param)
    if step is None:
        raise KeyError(
            f"no perceptual step declared for {param!r} — add one to STEPS "
            "rather than letting the caller invent a size")

    sign = -1.0 if direction == "decrease" else 1.0

    if percent is not None:
        if step.factor == 1.0:
            raise AmountError(
                f"{param} is measured on an additive scale, so a percentage of "
                f"it is ambiguous — say how many units instead, or use "
                f"'slightly', 'noticeably' or 'a lot'")
        return value * (1.0 + sign * percent / 100.0)

    if delta is not None:
        if step.factor != 1.0:
            raise AmountError(
                f"{param} is measured on a multiplicative scale, so a fixed "
                f"amount does not mean the same thing at every setting — say a "
                f"percentage instead, or use 'slightly', 'noticeably' or "
                f"'a lot'")
        return value + sign * delta

    raise AmountError("no amount was given")


def _find(graph: EffectGraph, node_id: str) -> tuple[list[GraphNode], int] | None:
    for chain in (graph.nodes, graph.master):
        for index, node in enumerate(chain):
            if node.id == node_id:
                return chain, index
    return None


def _unique_id(graph: EffectGraph, base: str) -> str:
    existing = {n.id for n in list(graph.nodes) + list(graph.master)}
    if base not in existing:
        return base
    for suffix in range(2, 100):
        candidate = f"{base}_{suffix}"
        if candidate not in existing:
            return candidate
    raise ValueError("cannot allocate a unique node id")


def apply_patches(graph: EffectGraph, patches: list[Patch]) -> PatchResult:
    """Apply a refinement to a graph, in order.

    Every operation names an existing node by id, so the result is the graph the
    user was listening to with exactly the requested change — not a new graph
    that happens to be similar. An operation naming a node that does not exist is
    REJECTED rather than ignored: a silently dropped edit means the user asked for
    something, saw no change, and has no idea why.
    """
    working = graph.model_copy(deep=True)
    result = PatchResult(graph=working)

    for patch in patches:
        try:
            _apply_one(working, patch, result)
        except (KeyError, ValueError, IndexError) as error:
            result.rejected.append(f"{patch.op} on {patch.node_id or '?'}: {error}")

    # Re-validate through Pydantic so a patched graph is held to the same
    # structural rules as a freshly emitted one. Bounds are clamped by the
    # semantic validator afterwards; this catches shape errors.
    result.graph = EffectGraph.model_validate(working.model_dump())
    return result


def _apply_one(graph: EffectGraph, patch: Patch, result: PatchResult) -> None:
    if patch.op in ("add_node",):
        if not patch.node_type:
            raise ValueError("add_node needs a node_type")
        node = GraphNode(
            id=_unique_id(graph, patch.node_id or patch.node_type),
            type=patch.node_type,  # type: ignore[arg-type]
            params=dict(patch.value or {}),
            rationale=patch.reason,
        )
        if patch.after:
            found = _find(graph, patch.after)
            if found is None:
                raise KeyError(f"no node {patch.after!r} to insert after")
            chain, index = found
            chain.insert(index + 1, node)
        else:
            graph.nodes.append(node)
        result.applied.append(f"added {node.type} as {node.id}")
        return

    found = _find(graph, patch.node_id)
    if found is None:
        raise KeyError(f"no node with id {patch.node_id!r}")
    chain, index = found
    node = chain[index]

    if patch.op == "remove_node":
        chain.pop(index)
        result.applied.append(f"removed {node.id}")

    elif patch.op == "bypass_node":
        object.__setattr__(node, "bypass", True)
        result.applied.append(f"bypassed {node.id}")

    elif patch.op == "enable_node":
        object.__setattr__(node, "bypass", False)
        result.applied.append(f"enabled {node.id}")

    elif patch.op == "set_param":
        # An absolute value, for the case where the user names one: "set the
        # delay to 300 ms". No ladder involved.
        if patch.param == "mix":
            object.__setattr__(node, "mix", float(patch.value))
        else:
            node.params[patch.param] = patch.value
        result.applied.append(f"{node.id}.{patch.param} set to {patch.value}")

    elif patch.op == "adjust_param":
        target, current = _read(node, patch)
        # An amount the user NAMED takes precedence over a ladder word. Both
        # routes end in arithmetic that lives in this file; the difference is
        # only whether the size came from the sentence or from STEPS.
        if patch.percent is not None or patch.delta is not None:
            try:
                moved = adjust_by_amount(
                    patch.param, current, percent=patch.percent,
                    delta=patch.delta, direction=patch.direction)
            except AmountError as error:
                # REFUSED WITH THE REASON, not clamped into something that
                # would apply. "15% more" of a dB value has no correct answer,
                # and picking one would teach the user a vocabulary that does
                # not mean what they think.
                result.rejected.append(f"{node.id}.{patch.param}: {error}")
                return
            size = (f"by {patch.percent:g}%" if patch.percent is not None
                    else f"by {patch.delta:g}")
        else:
            moved = adjust(patch.param, current, patch.quantifier,
                           patch.direction)
            size = patch.quantifier or "slightly"
        _write(node, patch, target, moved)
        result.applied.append(
            f"{node.id}.{patch.param} {patch.direction}d "
            f"{size}: {current:g} -> {moved:g}"
        )

    elif patch.op == "add_band":
        bands = node.params.setdefault("bands", [])
        bands.append(dict(patch.value or {}))
        result.applied.append(f"{node.id}: added a band")

    elif patch.op == "remove_band":
        bands = node.params.get("bands") or []
        if patch.band_index is None or not (0 <= patch.band_index < len(bands)):
            raise IndexError(f"no band at index {patch.band_index}")
        removed = bands.pop(patch.band_index)
        result.applied.append(
            f"{node.id}: removed the band at {removed.get('center_freq_hz', '?')} Hz")

    else:
        raise ValueError(f"unknown operation {patch.op!r}")


def _read(node: GraphNode, patch: Patch) -> tuple[dict[str, Any] | None, float]:
    """Locate the parameter and return (container, current value).

    `mix` lives on the node rather than in params, and a band parameter lives
    inside the band list, so three cases.
    """
    if patch.param == "mix":
        return None, float(node.mix)

    if patch.band_index is not None:
        bands = node.params.get("bands") or []
        if not (0 <= patch.band_index < len(bands)):
            raise IndexError(f"no band at index {patch.band_index}")
        band = bands[patch.band_index]
        if patch.param not in band:
            raise KeyError(f"band has no parameter {patch.param!r}")
        return band, float(band[patch.param])

    if patch.param not in node.params:
        raise KeyError(f"{node.type} has no parameter {patch.param!r} set")
    return node.params, float(node.params[patch.param])


def _write(node: GraphNode, patch: Patch, container: dict[str, Any] | None,
           value: float) -> None:
    if container is None:
        object.__setattr__(node, "mix", min(1.0, max(0.0, value)))
    else:
        # Rounded to something a human would type. An engineer reading
        # "-3.2 dB at 340 Hz" trusts it; "-3.1999999 dB at 339.87 Hz" reads as
        # machine output and invites the question of whether it was chosen at all.
        container[patch.param] = _tidy(patch.param, value)


def _tidy(param: str, value: float) -> float:
    """Round to a value a human would write down."""
    if param.endswith("_hz"):
        if value >= 1000.0:
            return float(round(value / 10.0) * 10)
        return float(round(value))
    if param.endswith(("_ms", "_percent", "_deg")):
        return float(round(value))
    if param.endswith(("_db", "_dbfs", "_dbtp")):
        return round(value, 1)
    return round(value, 3)


def patch_from_intent(
    graph: EffectGraph, intent: str, quantifier: Quantifier | None = None
) -> list[Patch]:
    """Turn a resolved intent phrase into patches against the actual graph.

    The intent vocabulary is closed and its meaning is fixed here — "more delay"
    means mix, "more repeats" means feedback, "longer delay" means time. Letting
    a model choose between those on each turn is how "more delay" comes to mean
    something different on Tuesday.

    Returns an empty list when the graph has no node of the required type. The
    caller decides whether to add one; guessing here would insert a reverb into a
    chain the user never asked to have one in.
    """
    if intent not in INTENT_TO_PARAM:
        raise KeyError(f"unknown intent {intent!r}; the vocabulary is closed")

    node_type, param = INTENT_TO_PARAM[intent]
    sign = INTENT_SIGN[intent]
    direction = "increase" if sign > 0 else "decrease"

    return [
        Patch(op="adjust_param", node_id=node.id, param=param,
              quantifier=quantifier, direction=direction,
              reason=f"user asked for {intent.replace('_', ' ')}")
        for node in graph.nodes if node.type == node_type
    ]


def describe(patches: list[Patch]) -> list[str]:
    """Human-readable description of what a patch set will do, BEFORE rendering.

    The plan's definition of done for the refinement loop is that the user reads
    the diff before hearing it. That requires the description to come from the
    patch itself rather than from a model's account of what it intended.
    """
    out = []
    for patch in patches:
        if patch.op == "adjust_param":
            word = (patch.quantifier or "slightly").replace("_", " ")
            arrow = "more" if patch.direction == "increase" else "less"
            out.append(f"{patch.node_id}: {word} {arrow} {patch.param}")
        elif patch.op == "set_param":
            out.append(f"{patch.node_id}: {patch.param} to {patch.value}")
        elif patch.op == "add_node":
            out.append(f"add {patch.node_type}")
        elif patch.op == "remove_node":
            out.append(f"remove {patch.node_id}")
        elif patch.op == "bypass_node":
            out.append(f"bypass {patch.node_id}")
        elif patch.op == "enable_node":
            out.append(f"re-enable {patch.node_id}")
        elif patch.op == "add_band":
            freq = (patch.value or {}).get("center_freq_hz", "?")
            out.append(f"{patch.node_id}: add a band at {freq} Hz")
        elif patch.op == "remove_band":
            out.append(f"{patch.node_id}: remove band {patch.band_index}")
    return out
