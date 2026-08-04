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
class Consistency:
    listener: str
    repeat_pairs: int
    reversals: int
    unstable: int
    reversal_rate: float | None
    unstable_rate: float | None
    rating_drift: dict[str, float | None] = field(default_factory=dict)

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


@dataclass(frozen=True)
class SessionScores:
    conditions: dict[tuple[str, str], ConditionScore]
    preferences: dict[str, PreferenceTally]
    consistency: dict[str, Consistency]
    scored_trials: int
    repeat_trials: int

    def condition_names(self) -> list[str]:
        return sorted({condition for condition, _ in self.conditions})

    def mean_for(self, condition: str, scale: str) -> float:
        return self.conditions[(condition, scale)].mean

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
                         + (f", prefer-or-tie {rate:.0%}" if rate is not None else ""))
        for listener in sorted(self.consistency):
            lines.append("  " + self.consistency[listener].summary())
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
    scored = 0
    repeats = 0

    for response in responses:
        record = trials[response.trial_id]
        if record.get("repeat_of"):
            repeats += 1
            continue
        scored += 1
        sides = {"A": record["condition_a"], "B": record["condition_b"]}
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

    return SessionScores(
        conditions=conditions,
        preferences=preferences,
        consistency=consistency_of(responses, key),
        scored_trials=scored,
        repeat_trials=repeats,
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

        for trial_id, response in by_id.items():
            record = trials[trial_id]
            if record["listener"] != listener:
                continue
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

        out[listener] = Consistency(
            listener=listener,
            repeat_pairs=pairs,
            reversals=reversals,
            unstable=unstable,
            reversal_rate=(reversals / pairs) if pairs else None,
            unstable_rate=(unstable / pairs) if pairs else None,
            rating_drift={scale: (float(np.mean(values)) if values else None)
                          for scale, values in drift.items()},
        )
    return out


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
