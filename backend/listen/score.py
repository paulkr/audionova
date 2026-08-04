"""Scoring a blinded session: two scales, honest intervals, and inconsistency.

TWO SCALES, KEPT SEPARATE, because they dissociate and the plan says so
explicitly. "Does it match the request?" and "does it sound professional?" are
different questions with different failure modes: a render can execute the
request precisely and sound amateur, and it can sound beautiful while ignoring
what was asked. Averaging them produces one number that moves for two reasons
and tells you neither.

THE INTERVAL IS CLUSTERED BY LISTENER, and this is the choice most likely to be
argued with, so the reasoning is here rather than in a commit message. Ratings
are not independent observations: one listener contributes many of them, and
listeners differ enormously in how they use a scale. A t-interval over the
pooled ratings treats thirty ratings from three people as thirty independent
draws and comes out roughly sqrt(10) times too narrow. Averaging within a
listener first and taking the interval over the listener means gives an
interval whose n is the number of PEOPLE, which is the sample the claim is
actually about. On a three-person panel that interval is wide. It is supposed
to be — three people is a small sample, and an interval that hides it would be
the proxy-capture failure in its purest form.

REPEAT TRIALS ARE EXCLUDED FROM THE MEANS. They exist to measure whether a
listener is self-consistent, not to give some comparisons twice the weight of
others. The first occurrence counts; the second scores only consistency.

INCONSISTENCY IS REPORTED TWO WAYS. A listener who says A then B has
contradicted themselves. A listener who says A then "tie" has not — they have
said the difference is small, which on a genuinely close pair is the correct
answer. `reversal_rate` counts only true contradictions; `unstable_rate`
counts any change. Reporting only the second would punish honesty about ties.

THE PHASE 0 GATE IS PER ITEM, AND THIS MODULE COMPUTES IT. The plan's
stop-or-go point is "three working engineers blind-A/B our render against their
own chain on ten files, loudness-matched, and prefer-or-tie ours on at least 6
of 10". That is a count of ITEMS, and the pooled prefer-or-tie rate this module
used to print is a different and more flattering statistic: a landslide on three
items carries seven losses, and the pooled rate clears 60% while the gate fails
4-6. The number the whole phase rests on should not be the one nobody computed.

THE SIDE TALLY IS PRINTED WHETHER OR NOT ANYTHING IS WRONG WITH IT. A/B is
re-randomised per trial, so a listener answering by what they hear splits the
sides about evenly however strong their preference — and a listener pressing A
thirty times does not. `consistency_of` already catches that probabilistically,
at 2^-(repeat pairs), and even when it fires the resulting reversal rate is
indistinguishable from an honest listener at chance. The count that separates
them is over data already in hand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import stats

from backend.listen import correlate
from backend.listen.session import PREFERENCES, SCALES, SCALE_KEYS

SIDES = ("A", "B")
CONFIDENCE = 0.95

# The plan's Phase 0 gate, verbatim: prefer-or-tie on at least 6 of 10 items.
# Both halves are here because both are arguable and neither may drift silently:
# the bar is a COUNT of items, not a rate, and the shape it was written for is
# ten items.
PHASE0_REQUIRED_ITEMS = 6
PHASE0_ITEMS = 10

# How lopsided a listener's side counts have to be before the report says the
# word. The count and the probability are printed unconditionally; this decides
# only when the sentence "this listener answered by position" appears beside
# them.
#
# 0.01 rather than 0.05, because the line is printed for every listener on every
# panel and what it says is "you were not listening". On a 30-trial session with
# no ties, an honest listener trips 0.05 with probability 0.0428 — one
# three-person panel in eight has somebody accused — against 0.0052 at 0.01,
# which is one panel in sixty-four. The cost of the stricter bar is one step of
# lopsidedness: 21-9 fires at 0.05 and 23-7 is needed at 0.01, and a real
# side-presser lands 30-0.
SIDE_BIAS_ALPHA = 0.01

_BOUNDS = {scale["key"]: (float(scale["min"]), float(scale["max"]))
           for scale in SCALES}


class ResponseError(ValueError):
    """A response cannot be scored, and guessing what was meant is worse."""


@dataclass(frozen=True)
class Response:
    listener: str
    trial_id: str
    preference: str
    ratings: dict[str, dict[str, float]]


def parse_responses(payload: Any) -> list[Response]:
    """Turn the interface's JSON into responses, refusing anything malformed.

    NO SILENT COERCION ANYWHERE IN HERE. A rating of "4" that becomes 4.0, a
    preference of "a" that becomes "A", a scale name that is nearly right — each
    of those turns a broken interface into plausible data, and plausible data
    is the thing this whole harness exists to avoid producing.
    """
    if isinstance(payload, dict):
        items = payload.get("responses")
        if items is None:
            raise ResponseError("response payload has no 'responses' list")
    else:
        items = payload
    if not isinstance(items, list):
        raise ResponseError(f"expected a list of responses, got {type(items).__name__}")

    out: list[Response] = []
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            raise ResponseError(f"response {position} is not an object")
        for required in ("listener", "trial_id", "preference", "ratings"):
            if required not in item:
                raise ResponseError(f"response {position} has no {required!r}")

        preference = item["preference"]
        if preference not in PREFERENCES:
            raise ResponseError(
                f"response {position}: preference {preference!r} is not one of "
                f"{list(PREFERENCES)}")

        raw = item["ratings"]
        if not isinstance(raw, dict):
            raise ResponseError(f"response {position}: ratings is not an object")
        if set(raw) != set(SIDES):
            raise ResponseError(
                f"response {position}: ratings cover {sorted(raw)}, and both "
                f"{list(SIDES)} must be rated — a missing side is a hole in the "
                "data, not a zero")

        ratings: dict[str, dict[str, float]] = {}
        for side in SIDES:
            side_ratings = raw[side]
            if not isinstance(side_ratings, dict):
                raise ResponseError(
                    f"response {position}: ratings for side {side} is not an object")
            unknown = sorted(set(side_ratings) - set(SCALE_KEYS))
            if unknown:
                raise ResponseError(
                    f"response {position}: unknown scale(s) {unknown}; the "
                    f"scales are {list(SCALE_KEYS)} and an unrecognised name is "
                    "a hard error rather than a discarded rating")
            missing = sorted(set(SCALE_KEYS) - set(side_ratings))
            if missing:
                raise ResponseError(
                    f"response {position}: side {side} is missing {missing}")
            values: dict[str, float] = {}
            for scale, value in side_ratings.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ResponseError(
                        f"response {position}: {scale} on side {side} is "
                        f"{value!r}, which is not a number")
                low, high = _BOUNDS[scale]
                if not (low <= float(value) <= high):
                    raise ResponseError(
                        f"response {position}: {scale} on side {side} is "
                        f"{value}, outside the scale's {low}-{high}. Clamping "
                        "it would put a number nobody gave into the result.")
                values[scale] = float(value)
            ratings[side] = values

        out.append(Response(
            listener=str(item["listener"]),
            trial_id=str(item["trial_id"]),
            preference=preference,
            ratings=ratings,
        ))
    return out


@dataclass(frozen=True)
class ConditionScore:
    condition: str
    scale: str
    mean: float
    ci_low: float | None
    ci_high: float | None
    n_listeners: int
    n_ratings: int
    per_listener: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        if self.ci_low is None:
            return (f"{self.condition}/{self.scale}: {self.mean:.2f} "
                    f"(no interval — {self.n_listeners} listener(s))")
        return (f"{self.condition}/{self.scale}: {self.mean:.2f} "
                f"[{self.ci_low:.2f}, {self.ci_high:.2f}] "
                f"n={self.n_listeners} listeners, {self.n_ratings} ratings")


@dataclass(frozen=True)
class PreferenceTally:
    condition: str
    wins: int
    losses: int
    ties: int

    @property
    def trials(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def prefer_or_tie(self) -> float | None:
        """The Phase 0 gate's own criterion, as a rate."""
        return (self.wins + self.ties) / self.trials if self.trials else None


@dataclass(frozen=True)
class ItemPreference:
    """One item's verdict for one condition, as the Phase 0 gate counts it.

    ONE VERDICT PER LISTENER PER ITEM, from the FIRST occurrence. A repeat is a
    second judgement of the same item by the same person; counting both would
    give that listener two votes on that item and one on every other, which is
    the pooled rate's failure mode in miniature. What the repeat is for is
    `Consistency`, which says how much the vote is worth.
    """

    item: str
    condition: str
    prefer: int
    tie: int
    against: int
    per_listener: dict[str, str] = field(default_factory=dict)

    @property
    def listeners(self) -> int:
        return self.prefer + self.tie + self.against

    @property
    def prefer_or_tie(self) -> int:
        return self.prefer + self.tie

    @property
    def passed(self) -> bool:
        """STRICTLY more than half the listeners who judged this item.

        The plan says three engineers and does not say what two of them
        disagreeing means, so this is a decision rather than a reading: an item
        counts for a condition when a majority prefers-or-ties it. On three
        listeners that is 2 of 3. On an even panel a 1-1 split does NOT count —
        half is not a majority, and the alternative (counting a tied item for
        both conditions) would let a panel that agreed on nothing pass a gate
        designed to need agreement.
        """
        return self.prefer_or_tie * 2 > self.listeners

    def summary(self) -> str:
        verdict = "counts" if self.passed else "does not count"
        return (f"{self.item}: {self.prefer} prefer, {self.tie} tie, "
                f"{self.against} against — {verdict} for {self.condition}")


@dataclass(frozen=True)
class Phase0Gate:
    """The plan's stop-or-go point, computed rather than eyeballed."""

    condition: str
    required_items: int
    items: tuple[ItemPreference, ...]

    @property
    def items_passed(self) -> int:
        return sum(1 for item in self.items if item.passed)

    @property
    def items_total(self) -> int:
        return len(self.items)

    @property
    def passed(self) -> bool:
        return self.items_passed >= self.required_items

    def report(self) -> str:
        lines = [item.summary() for item in self.items]
        verdict = "PASSED" if self.passed else "FAILED"
        lines.append(
            f"PHASE 0 GATE ({self.condition}): {self.items_passed} of "
            f"{self.items_total} items prefer-or-tie, bar is "
            f"{self.required_items} — {verdict}")
        if self.items_total != PHASE0_ITEMS:
            # SAID PLAINLY RATHER THAN RESCALED. The plan's criterion is 6 of
            # 10; a proportional bar on a different number of items is a
            # different criterion, and inventing one here would put a threshold
            # nobody agreed to behind the word "gate".
            lines.append(
                f"  note: the plan's gate is {PHASE0_REQUIRED_ITEMS} of "
                f"{PHASE0_ITEMS} items and this session has {self.items_total}. "
                f"The verdict above is against the bar as given "
                f"({self.required_items}), not a rescaled one.")
        return "\n".join(lines)


@dataclass(frozen=True)
class Consistency:
    listener: str
    repeat_pairs: int
    reversals: int
    unstable: int
    reversal_rate: float | None
    unstable_rate: float | None
    rating_drift: dict[str, float | None] = field(default_factory=dict)
    chose_a: int = 0
    chose_b: int = 0
    chose_tie: int = 0
    # P(side counts this lopsided | the listener has no side preference), two
    # sided, over the non-tie trials. None when they tied everything.
    side_bias_p: float | None = None

    @property
    def side_biased(self) -> bool:
        return self.side_bias_p is not None and self.side_bias_p <= SIDE_BIAS_ALPHA

    def summary(self) -> str:
        if self.repeat_pairs == 0:
            return (f"{self.listener}: no repeat trials, so nothing is known "
                    "about how much to trust their other answers")
        drift = ", ".join(
            f"{scale} {value:.2f}" if value is not None else f"{scale} n/a"
            for scale, value in sorted(self.rating_drift.items()))
        return (f"{self.listener}: {self.reversals}/{self.repeat_pairs} "
                f"reversed, {self.unstable}/{self.repeat_pairs} changed at all; "
                f"mean |change| {drift}")

    def side_summary(self) -> str:
        """The count that separates a side-presser from a listener at chance.

        Both look like a reversal rate near a half. This does not: the sides
        are re-randomised per trial, so an honest listener's A/B split is a
        coin toss regardless of how strongly they prefer one condition, and a
        listener who pressed one button all afternoon is 30-0.
        """
        total = self.chose_a + self.chose_b + self.chose_tie
        line = (f"{self.listener}: chose A {self.chose_a}, B {self.chose_b}, "
                f"no preference {self.chose_tie}, of {total} trial(s)")
        if self.side_bias_p is None:
            return line + " — no side was chosen, so there is nothing to test"
        line += f"; P(this lopsided | no side preference) = {self.side_bias_p:.4f}"
        if self.side_biased:
            return (line + " — SIDE BIAS. A and B are re-randomised per trial, "
                    "so this listener was answering by position rather than by "
                    "what they heard, and their ratings measure nothing. "
                    "Their file belongs out of responses/ before scoring.")
        return line


@dataclass(frozen=True)
class SessionScores:
    conditions: dict[tuple[str, str], ConditionScore]
    preferences: dict[str, PreferenceTally]
    consistency: dict[str, Consistency]
    scored_trials: int
    repeat_trials: int
    items: dict[tuple[str, str], ItemPreference] = field(default_factory=dict)

    def condition_names(self) -> list[str]:
        return sorted({condition for condition, _ in self.conditions})

    def item_names(self) -> list[str]:
        return sorted({item for item, _ in self.items})

    def mean_for(self, condition: str, scale: str) -> float:
        return self.conditions[(condition, scale)].mean

    def gate(self, condition: str,
             required_items: int = PHASE0_REQUIRED_ITEMS) -> Phase0Gate:
        """The Phase 0 gate for one named condition.

        The condition has to be named. Nothing in a blinded session says which
        of the two is ours — the whole point is that the session does not know
        — and picking one alphabetically would put the stop-or-go decision on a
        coin toss between `audionova` and `engineer`.
        """
        known = self.condition_names()
        if condition not in known:
            raise ResponseError(
                f"no condition {condition!r} in this session; the conditions "
                f"are {known}. The gate is computed for a named condition "
                "because the session does not know which one is ours.")
        return Phase0Gate(
            condition=condition,
            required_items=required_items,
            items=tuple(self.items[key] for key in sorted(self.items)
                        if key[1] == condition),
        )

    def report(self) -> str:
        lines = [f"{self.scored_trials} scored trial(s), "
                 f"{self.repeat_trials} repeat trial(s) held back for consistency"]
        for key in sorted(self.conditions):
            lines.append("  " + self.conditions[key].summary())

        for condition in sorted(self.preferences):
            tally = self.preferences[condition]
            rate = tally.prefer_or_tie
            lines.append(f"  {condition}: preferred {tally.wins}, tied "
                         f"{tally.ties}, lost {tally.losses}"
                         + (f", prefer-or-tie {rate:.0%} POOLED over trials"
                            if rate is not None else ""))

        # THE POOLED RATE ABOVE IS NOT THE GATE, and the two are printed
        # together so nobody has to remember that. The gate counts items.
        if self.items:
            lines.append("  per item, first occurrence only, one verdict per "
                         "listener (the Phase 0 gate's unit):")
            for key in sorted(self.items):
                lines.append("    " + self.items[key].summary())

        for listener in sorted(self.consistency):
            lines.append("  " + self.consistency[listener].summary())
        for listener in sorted(self.consistency):
            lines.append("  " + self.consistency[listener].side_summary())
        return "\n".join(lines)


def _interval(values: Sequence[float],
              confidence: float = CONFIDENCE) -> tuple[float | None, float | None]:
    """A t-interval, or nothing.

    Nothing, rather than a plausible width, when there is one observation. A
    single listener has no measurable spread, and any interval printed for one
    is a statement about an assumption rather than about the data.
    """
    n = len(values)
    if n < 2:
        return (None, None)
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    half = float(stats.t.ppf(0.5 + confidence / 2.0, n - 1)) * sd / math.sqrt(n)
    return (mean - half, mean + half)


def score_session(responses: Iterable[Response], key: dict[str, Any],
                  *, confidence: float = CONFIDENCE) -> SessionScores:
    """Score a whole session against its private key."""
    trials = key.get("trials")
    if not isinstance(trials, dict) or not trials:
        raise ResponseError("the session key has no trials in it")

    responses = list(responses)
    seen: set[tuple[str, str]] = set()
    for response in responses:
        if response.trial_id not in trials:
            raise ResponseError(
                f"response names trial {response.trial_id!r}, which is not in "
                "the key — either the wrong key or a fabricated response")
        expected = trials[response.trial_id]["listener"]
        if response.listener != expected:
            raise ResponseError(
                f"trial {response.trial_id} belongs to listener {expected!r} "
                f"and was answered by {response.listener!r}; presentation order "
                "is per listener, so answers cannot be pooled across them")
        pair = (response.listener, response.trial_id)
        if pair in seen:
            raise ResponseError(
                f"{response.listener} answered trial {response.trial_id} twice; "
                "a repeat trial has its own id, so this is a duplicate rather "
                "than a repeat")
        seen.add(pair)

    # -- ratings, excluding repeat occurrences ----------------------------
    gathered: dict[tuple[str, str], dict[str, list[float]]] = {}
    tallies: dict[str, dict[str, int]] = {}
    votes: dict[tuple[str, str], dict[str, str]] = {}
    scored = 0
    repeats = 0

    for response in responses:
        record = trials[response.trial_id]
        if record.get("repeat_of"):
            repeats += 1
            continue
        scored += 1
        sides = {"A": record["condition_a"], "B": record["condition_b"]}

        item = record["item"]
        for side, condition in sides.items():
            if response.preference == "tie":
                verdict = "tie"
            elif response.preference == side:
                verdict = "prefer"
            else:
                verdict = "against"
            per_listener = votes.setdefault((item, condition), {})
            if response.listener in per_listener:
                raise ResponseError(
                    f"{response.listener} judged item {item!r} twice outside a "
                    "repeat trial, so the per-item gate cannot say what their "
                    "verdict on it was. Two comparisons sharing an item name "
                    "is the usual cause; give them distinct names.")
            per_listener[response.listener] = verdict
        for side, condition in sides.items():
            for scale, value in response.ratings[side].items():
                gathered.setdefault((condition, scale), {}) \
                         .setdefault(response.listener, []).append(value)
        for side, condition in sides.items():
            tally = tallies.setdefault(condition, {"wins": 0, "losses": 0, "ties": 0})
            if response.preference == "tie":
                tally["ties"] += 1
            elif response.preference == side:
                tally["wins"] += 1
            else:
                tally["losses"] += 1

    conditions: dict[tuple[str, str], ConditionScore] = {}
    for key_pair, by_listener in gathered.items():
        condition, scale = key_pair
        listener_means = {listener: float(np.mean(values))
                          for listener, values in by_listener.items()}
        means = [listener_means[listener] for listener in sorted(listener_means)]
        low, high = _interval(means, confidence)
        conditions[key_pair] = ConditionScore(
            condition=condition,
            scale=scale,
            mean=float(np.mean(means)),
            ci_low=low,
            ci_high=high,
            n_listeners=len(means),
            n_ratings=sum(len(values) for values in by_listener.values()),
            per_listener=listener_means,
        )

    preferences = {condition: PreferenceTally(condition=condition, **counts)
                   for condition, counts in tallies.items()}

    items = {
        key_pair: ItemPreference(
            item=key_pair[0],
            condition=key_pair[1],
            prefer=sum(1 for v in per_listener.values() if v == "prefer"),
            tie=sum(1 for v in per_listener.values() if v == "tie"),
            against=sum(1 for v in per_listener.values() if v == "against"),
            per_listener=dict(per_listener),
        )
        for key_pair, per_listener in votes.items()
    }

    return SessionScores(
        conditions=conditions,
        preferences=preferences,
        consistency=consistency_of(responses, key),
        scored_trials=scored,
        repeat_trials=repeats,
        items=items,
    )


def consistency_of(responses: Iterable[Response],
                   key: dict[str, Any]) -> dict[str, Consistency]:
    """How much to trust a listener's other answers, from the repeat trials.

    A repeat is scored against the trial it repeats AFTER MAPPING BOTH BACK TO
    CONDITIONS. The two occurrences are independently randomised, so "they said
    A both times" is not consistency — it may be a side bias, which is the other
    thing this catches.
    """
    trials = key["trials"]
    by_id = {response.trial_id: response for response in responses}

    listeners = sorted({trials[trial_id]["listener"] for trial_id in by_id})
    out: dict[str, Consistency] = {}

    for listener in listeners:
        pairs = 0
        reversals = 0
        unstable = 0
        drift: dict[str, list[float]] = {scale: [] for scale in SCALE_KEYS}
        chose = {"A": 0, "B": 0, "tie": 0}

        for trial_id, response in by_id.items():
            record = trials[trial_id]
            if record["listener"] != listener:
                continue
            # EVERY TRIAL, REPEATS INCLUDED. The side tally is a count of
            # button presses, not a measurement of a comparison, and a
            # side-presser presses on the repeats too. Excluding them here for
            # symmetry with the ratings would throw away a fifth of the
            # evidence about the one thing this count exists to detect.
            chose[response.preference] += 1

            original_id = record.get("repeat_of")
            if not original_id:
                continue
            original = by_id.get(original_id)
            if original is None:
                # The first occurrence was never answered. Counting the pair
                # would need an answer that does not exist.
                continue
            pairs += 1

            first = _preferred_condition(trials[original_id], original.preference)
            second = _preferred_condition(record, response.preference)
            if first != second:
                unstable += 1
                if first is not None and second is not None:
                    reversals += 1

            first_by_condition = _by_condition(trials[original_id], original)
            second_by_condition = _by_condition(record, response)
            for condition, scales in first_by_condition.items():
                for scale, value in scales.items():
                    other = second_by_condition[condition][scale]
                    drift[scale].append(abs(value - other))

        sided = chose["A"] + chose["B"]
        out[listener] = Consistency(
            listener=listener,
            repeat_pairs=pairs,
            reversals=reversals,
            unstable=unstable,
            reversal_rate=(reversals / pairs) if pairs else None,
            unstable_rate=(unstable / pairs) if pairs else None,
            rating_drift={scale: (float(np.mean(values)) if values else None)
                          for scale, values in drift.items()},
            chose_a=chose["A"],
            chose_b=chose["B"],
            chose_tie=chose["tie"],
            side_bias_p=(two_sided_binomial(max(chose["A"], chose["B"]), sided)
                         if sided else None),
        )
    return out


def two_sided_binomial(successes: int, trials: int, p: float = 0.5) -> float:
    """P(a split at least this lopsided | each side equally likely).

    Two sided and computed as twice the upper tail from the LARGER count. The
    question is "did this listener favour a side", not "did they favour A", so
    a 0-30 split has to be as damning as 30-0 — a one-sided test would clear a
    listener who pressed B thirty times.

    Exact rather than normal-approximated: thirty trials is small, the
    approximation is poor in exactly the tail this reads, and `math.comb` is
    free at this size.
    """
    if trials < 0 or successes < 0 or successes > trials:
        raise ResponseError(f"{successes} of {trials} is not a split")
    if trials == 0:
        raise ResponseError("no trials to test for a side preference")
    larger = max(successes, trials - successes)
    tail = sum(math.comb(trials, k) * (p ** k) * ((1.0 - p) ** (trials - k))
               for k in range(larger, trials + 1))
    return min(1.0, 2.0 * tail)


def item_points(responses: Iterable[Response], key: dict[str, Any],
                scale: str) -> dict[str, correlate.HumanPoint]:
    """Per-rendered-clip human scores, keyed `item::condition`.

    THE UNIT FOR A CORRELATION IS THE RENDERED CLIP, not the condition. An
    automated metric produces a number per file; correlating against two
    condition means would be a correlation over two points, which is not a
    correlation. Repeat occurrences are folded into their original's mean
    rather than dropped, because here they are extra judgements of the same
    file and not extra weight on a comparison.
    """
    if scale not in SCALE_KEYS:
        raise ResponseError(
            f"unknown scale {scale!r}; the scales are {list(SCALE_KEYS)}")
    trials = key["trials"]
    gathered: dict[str, list[float]] = {}
    for response in responses:
        record = trials.get(response.trial_id)
        if record is None:
            raise ResponseError(
                f"response names trial {response.trial_id!r}, not in the key")
        for side, condition in (("A", record["condition_a"]),
                                ("B", record["condition_b"])):
            name = f"{record['item']}::{condition}"
            gathered.setdefault(name, []).append(response.ratings[side][scale])
    return {name: correlate.HumanPoint(mean=float(np.mean(values)),
                                       n_ratings=len(values))
            for name, values in gathered.items()}


def _preferred_condition(record: dict[str, Any], preference: str) -> str | None:
    if preference == "tie":
        return None
    if preference == "A":
        return record["condition_a"]
    if preference == "B":
        return record["condition_b"]
    raise ResponseError(f"preference {preference!r} is not one of {list(PREFERENCES)}")


def _by_condition(record: dict[str, Any],
                  response: Response) -> dict[str, dict[str, float]]:
    return {
        record["condition_a"]: dict(response.ratings["A"]),
        record["condition_b"]: dict(response.ratings["B"]),
    }
