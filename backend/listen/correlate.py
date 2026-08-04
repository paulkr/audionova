"""Spearman rho between a listening result and the automated proxies.

THIS IS THE NUMBER THAT DECIDES WHICH PROXIES ARE ALLOWED TO GATE CI. The plan
is firm: no automated metric gates anything until it correlates with human MOS
at rho > 0.6. Every figure in `evaluation.md` today should be read as "this
measures what it says it measures" and not as "this predicts quality", because
none of them has been through here.

SPEARMAN, NOT PEARSON, and the reason is not fashion. A proxy does not have to
be linear in perceived quality to be useful — it has to be MONOTONE in it. A
metric that tracks quality perfectly through a cube is worthless to Pearson and
perfect to Spearman, and it is perfect to a gate too, because a gate only ever
asks "did this get better or worse". A rating scale is ordinal anyway; the
distance between 3 and 4 is not the distance between 4 and 5.

WHY THIS REFUSES TO PRINT A BARE RHO. Spearman on a handful of points is
enormously noisy: with n = 8, a sample rho of 0.62 has a 95% interval running
from roughly 0.0 to 0.92. Printing "rho = 0.62, over the bar" from that data is
how a proxy gets blessed on noise, and blessing a proxy is exactly the
proxy-metric capture the plan names as its top risk. So:

  * every result carries its interval and its n,
  * `usable` is false below `MIN_N`, and the reason says which n was short,
  * `__str__` cannot be made to print a lone number,
  * `may_gate_ci` tests the interval's LOWER BOUND against the bar, not the
    point estimate — "we are confident it is over 0.6", not "our best guess is
    over 0.6". The bar is the plan's; the lower bound is the honest reading of
    it, and the difference is entire on small panels.

An exactly degenerate result reports no interval rather than a zero-width one.
A perfect rho on five points is not certainty, and the Fisher transform says it
is, which makes it the wrong tool there rather than a tool to be trusted
harder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

# The plan's threshold.
GATE_RHO = 0.6

# Below this, an estimate is reported and refused. Fifteen is the plan's
# figure; it is not a magic number so much as the point where a Spearman
# interval stops spanning most of the usable range.
MIN_N = 15

CONFIDENCE = 0.95

# |rho| this close to 1 makes the Fisher transform degenerate.
_DEGENERATE = 1.0 - 1e-12


class CorrelationError(ValueError):
    """The data cannot support a correlation, and inventing one is worse."""


def ranks(values: Sequence[float]) -> np.ndarray:
    """Ranks with ties averaged.

    Averaged rather than ordered-by-appearance: two clips a metric cannot tell
    apart must not be given an order the metric never expressed. Assigning them
    consecutive ranks invents information and biases rho toward whatever order
    the input happened to be in.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise CorrelationError(f"expected a 1-D sequence, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise CorrelationError(
            "cannot rank a series containing a non-finite value — a missing "
            "measurement is null, and null has no rank")

    order = np.argsort(array, kind="mergesort")
    out = np.empty(array.size, dtype=float)
    out[order] = np.arange(1, array.size + 1, dtype=float)

    ordered = array[order]
    start = 0
    while start < array.size:
        stop = start
        while stop + 1 < array.size and ordered[stop + 1] == ordered[start]:
            stop += 1
        if stop > start:
            out[order[start:stop + 1]] = (start + stop + 2) / 2.0
        start = stop + 1
    return out


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson's r on the ranks. NaN when either series is constant.

    NaN, not zero. A metric with the same value on every item has expressed no
    ordering at all, and reporting "no relationship" for that is a different
    and much more reassuring statement than "this metric said nothing".
    """
    if len(x) != len(y):
        raise CorrelationError(
            f"paired series must be the same length, got {len(x)} and {len(y)}")
    if len(x) < 2:
        raise CorrelationError(
            f"a correlation needs at least two paired points, got {len(x)}")

    rx = ranks(x)
    ry = ranks(y)
    dx = rx - rx.mean()
    dy = ry - ry.mean()
    denominator = math.sqrt(float(np.dot(dx, dx)) * float(np.dot(dy, dy)))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(dx, dy) / denominator)


def _z(confidence: float) -> float:
    # Inverse normal CDF without pulling in a distribution object for one call.
    from scipy import stats
    return float(stats.norm.ppf(0.5 + confidence / 2.0))


def rho_interval(rho: float, n: int,
                 confidence: float = CONFIDENCE) -> tuple[float | None, float | None]:
    """Bonett-Wright interval for Spearman's rho, or nothing.

    The standard error of the Fisher-transformed Spearman coefficient is
    sqrt((1 + rho^2 / 2) / (n - 3)), which needs n > 3. Below that, and at
    |rho| = 1 where the transform runs to infinity, there is no interval to
    report and none is reported.
    """
    if not math.isfinite(rho) or n <= 3:
        return (None, None)
    if abs(rho) >= _DEGENERATE:
        return (None, None)
    se = math.sqrt((1.0 + rho * rho / 2.0) / (n - 3))
    half = _z(confidence) * se
    centre = math.atanh(rho)
    return (math.tanh(centre - half), math.tanh(centre + half))


@dataclass(frozen=True)
class Correlation:
    metric: str
    rho: float
    n_pairs: int
    ci_low: float | None
    ci_high: float | None
    min_ratings: int | None
    usable: bool
    reason: str

    def __str__(self) -> str:
        """Always the interval, always the n, always the verdict.

        There is deliberately no formatting path that yields a lone rho. The
        number's whole failure mode is being quoted without its uncertainty.
        """
        if math.isnan(self.rho):
            body = "rho undefined (a series is constant)"
        elif self.ci_low is None:
            body = f"rho {self.rho:+.3f}, interval unavailable"
        else:
            body = f"rho {self.rho:+.3f} [{self.ci_low:+.3f}, {self.ci_high:+.3f}]"
        verdict = "usable" if self.usable else "NOT USABLE"
        return f"{self.metric}: {body}, n={self.n_pairs} — {verdict}: {self.reason}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "rho": None if math.isnan(self.rho) else round(self.rho, 6),
            "n_pairs": self.n_pairs,
            "ci_low": None if self.ci_low is None else round(self.ci_low, 6),
            "ci_high": None if self.ci_high is None else round(self.ci_high, 6),
            "min_ratings_per_point": self.min_ratings,
            "usable": self.usable,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class HumanPoint:
    """One item's human score and how many judgements produced it."""

    mean: float
    n_ratings: int


def correlate(human: Sequence[float], metric: Sequence[float], *,
              name: str = "metric", min_ratings: int | None = None,
              min_n: int = MIN_N, confidence: float = CONFIDENCE) -> Correlation:
    """One metric against one set of human scores, with its interval."""
    rho = spearman_rho(human, metric)
    n = len(human)
    low, high = rho_interval(rho, n, confidence)

    reasons: list[str] = []
    if math.isnan(rho):
        reasons.append("one of the series has no variation, so no ordering "
                       "was expressed to correlate with")
    if n < min_n:
        reasons.append(f"{n} paired points is under the {min_n} this project "
                       "requires before a rho is treated as an estimate")
    if min_ratings is not None and min_ratings < min_n:
        reasons.append(f"the thinnest point rests on {min_ratings} human "
                       f"rating(s), under {min_n}")
    if low is None and not math.isnan(rho):
        reasons.append("no interval could be computed, so the uncertainty is "
                       "unstated rather than small")

    usable = not reasons
    return Correlation(
        metric=name,
        rho=rho,
        n_pairs=n,
        ci_low=low,
        ci_high=high,
        min_ratings=min_ratings,
        usable=usable,
        reason="; ".join(reasons) if reasons else "n and interval both adequate",
    )


def correlate_proxies(
    human: Mapping[str, HumanPoint | float],
    metrics: Mapping[str, Mapping[str, float]],
    *,
    min_n: int = MIN_N,
    confidence: float = CONFIDENCE,
) -> dict[str, Correlation]:
    """Every automated metric against the same human scores.

    Items are matched by name, and a metric that does not cover every item is
    refused rather than correlated on the overlap. Silently dropping items
    would let a metric improve its own rho by declining to measure the cases it
    gets wrong.
    """
    if not human:
        raise CorrelationError("no human scores to correlate against")

    names = sorted(human)
    points = [human[name] for name in names]
    values = [point.mean if isinstance(point, HumanPoint) else float(point)
              for point in points]
    counts = [point.n_ratings for point in points if isinstance(point, HumanPoint)]
    min_ratings = min(counts) if len(counts) == len(points) else None

    out: dict[str, Correlation] = {}
    for metric_name in sorted(metrics):
        series = metrics[metric_name]
        missing = [name for name in names if name not in series]
        if missing:
            raise CorrelationError(
                f"metric {metric_name!r} has no value for {missing[:3]} "
                f"({len(missing)} item(s)). Correlating on the overlap would "
                "let a metric raise its own rho by declining to measure the "
                "items it does badly on.")
        extra = sorted(set(series) - set(names))
        if extra:
            raise CorrelationError(
                f"metric {metric_name!r} covers items with no human score: "
                f"{extra[:3]}")
        out[metric_name] = correlate(
            values, [float(series[name]) for name in names],
            name=metric_name, min_ratings=min_ratings, min_n=min_n,
            confidence=confidence)
    return out


def may_gate_ci(result: Correlation, *, threshold: float = GATE_RHO,
                use_lower_bound: bool = True) -> bool:
    """Is this proxy allowed to gate CI?

    THE LOWER BOUND, not the point estimate. The plan says rho > 0.6; on a
    sample small enough for a point estimate of 0.62 to be consistent with
    zero, gating on the point estimate is gating on noise. Requiring the
    interval to clear the bar is the same rule read as a claim about the
    population rather than about one sample.
    """
    if not result.usable:
        return False
    if math.isnan(result.rho):
        return False
    if not use_lower_bound:
        return result.rho > threshold
    return result.ci_low is not None and result.ci_low > threshold


def report(results: Mapping[str, Correlation], *,
           threshold: float = GATE_RHO) -> str:
    lines = [f"Spearman against human judgement (gate: rho > {threshold}, "
             f"on the interval's lower bound)"]
    for name in sorted(results):
        result = results[name]
        allowed = "MAY GATE" if may_gate_ci(result, threshold=threshold) else "may not gate"
        lines.append(f"  {result}  [{allowed}]")
    return "\n".join(lines)
