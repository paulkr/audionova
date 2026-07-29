"""Which measurements can actually see which effects.

THE QUESTION THIS ANSWERS IS PRIOR TO EVERYTHING ELSE. The planner is going to
be handed a measurement report and asked to reason about audio it cannot hear.
If a report field does not move when an effect is applied, then no prompt, no
model and no amount of chain-of-thought will recover that effect — the
information is not there. Finding that out from a table is cheap; finding it
out from a listening panel is not.

THE METRIC IS "CAN YOU READ THE PARAMETER OFF THIS FIELD", not "did this field
move". The first version scored a field by its mean shift over its own spread,
pooling every value of the parameter as though they were repeats of one
experiment. They are not — they are different treatments — and the consequence
was that GAIN scored as undetectable: sweeping -6, -3, +3 and +6 dB moves
loudness down twice and up twice, so the "consistent sign" test failed on the
single most obviously detectable effect in the catalogue.

What matters is whether the field TRACKS the parameter. So each field is scored
by the correlation between the parameter value and the field's delta, computed
WITHIN each source and then averaged. Within-source removes the offset between
different material; averaging across sources is what makes the result about the
measurement rather than about one recording.

Two forms count, because two kinds of field are useful:

  MONOTONIC — the delta follows the parameter, so the parameter can in
  principle be inverted out of it. Gain against loudness.

  MAGNITUDE — the delta follows the ABSOLUTE value, so the field reports that
  something happened and how much, but not which direction. A bell's residual
  rises whether the bell cuts or boosts; that still localises the move, it just
  needs a second field to resolve the sign.

Reporting only the first would discard genuinely useful evidence and call
symmetric effects invisible.

EVERY COMPARISON IS AGAINST A NULL RENDER. Measuring against the raw source
folds the 48 -> 96 -> 48 kHz conversion's own 0.12 dB into every result, which
is larger than several of the effects being measured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.engine.render import Recipe, render_array
from backend.eval.corpus import Perturbation, SourceContext
from backend.measure import report as report_mod

# "Twice the noise" is the usual convention for an effect worth believing, and
# 0.7 is the usual bar for a correlation that is describing a relationship
# rather than a coincidence. Neither is tuned to make a particular field pass.
MIN_CORRELATION = 0.7

# The report rounds every number to two decimals, so a field whose whole swing
# across the parameter range is smaller than a few rounding quanta is not
# merely weak — it is INVISIBLE in the document the planner is handed.
#
# This exists because correlation alone is not enough. A 1 kHz bell moves the
# high-frequency rolloff at r = -1.00, a flawless correlation, with a slope of
# 0.0085 dB/octave per dB of bell gain: the entire sweep moves it by less than
# a tenth of a dB. Reporting that as "read the bell gain off the HF rolloff" is
# true and useless.
MIN_SPAN = 0.05


def flatten(report: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Report to ``{dotted.path: number}``, including per-band entries.

    BAND LISTS ARE EXPANDED, keyed by their centre frequency. Dropping them was
    a flaw in the analysis rather than a simplification: the report genuinely
    contains ``third_octave_db`` and ``erb_db``, so the planner genuinely sees
    them, and an analysis that discards them measures a report nobody is given.

    It mattered immediately. A 1 kHz bell appeared undetectable except through
    summary statistics that a plain gain change moves identically — the
    conclusion would have been "the report cannot localise mid-band EQ", which
    is the bread and butter of mixing. The band levels were there the whole
    time.
    """
    out: dict[str, float] = {}
    for key, value in report.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            if math.isfinite(float(value)):
                out[path] = float(value)
        elif isinstance(value, dict):
            out.update(flatten(value, path))
        elif isinstance(value, list):
            for item in value:
                if not isinstance(item, dict):
                    continue
                centre = item.get("centre_hz")
                if centre is None:
                    continue
                # Keyed by frequency rather than index, so the field name stays
                # stable if the band layout ever changes.
                for name, entry in item.items():
                    if name == "centre_hz" or not isinstance(entry, (int, float)):
                        continue
                    if isinstance(entry, bool) or not math.isfinite(float(entry)):
                        continue
                    out[f"{path}.{int(round(float(centre)))}hz.{name}"] = float(entry)
    return out


@dataclass
class FieldEffect:
    field: str
    slope: float        # field units per unit of parameter
    r: float            # mean within-source correlation with the value
    r_abs: float        # mean within-source correlation with |value|
    span: float         # mean absolute delta, for a sense of scale

    @property
    def monotonic(self) -> bool:
        return abs(self.r) >= MIN_CORRELATION

    @property
    def magnitude(self) -> bool:
        return abs(self.r_abs) >= MIN_CORRELATION

    @property
    def visible(self) -> bool:
        """Does it move enough to survive the report's own rounding?"""
        return self.span >= MIN_SPAN

    @property
    def diagnostic(self) -> bool:
        return (self.monotonic or self.magnitude) and self.visible

    @property
    def kind(self) -> str:
        if not self.visible:
            return "too small"
        if self.monotonic:
            return "monotonic"
        if self.magnitude:
            return "magnitude"
        return "-"


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Correlation, with the degenerate cases returning 0 rather than NaN.

    A field that does not move has zero variance, and a NaN propagated from
    here would sort to the top of the table or crash the comparison. Zero is
    the truthful answer: no relationship was observed.
    """
    if x.size < 3:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= 1e-30:
        return 0.0
    return float(np.clip(np.sum(x * y) / denom, -1.0, 1.0))


def render_reference(source: np.ndarray, sample_rate: float, **kwargs: Any) -> np.ndarray:
    """The source through an empty recipe: the only valid baseline for a delta."""
    return render_array(
        source, sample_rate,
        Recipe(output_sample_rate=sample_rate, output_format="float64"),
        **kwargs,
    )


def sweep_source(
    source: np.ndarray,
    sample_rate: float,
    perturbation: Perturbation,
    *,
    seed: int = 1,
    **kwargs: Any,
) -> list[tuple[float, dict[str, float]]]:
    """Render every value of one perturbation over one source.

    The null reference is rendered ONCE per source rather than once per case.
    It depends only on the audio, and rendering it per case doubled the sweep's
    cost for no additional information.
    """
    context = SourceContext.measure(source, sample_rate)
    reference = render_reference(source, sample_rate, **kwargs)
    before = flatten(report_mod.measure(reference, sample_rate).to_dict())

    out: list[tuple[float, dict[str, float]]] = []
    for value in perturbation.values:
        recipe = Recipe(seed=seed, output_sample_rate=sample_rate,
                        output_format="float64")
        nodes = perturbation.build(value, context)
        (recipe.master if perturbation.master else recipe.nodes).extend(nodes)

        processed = render_array(source, sample_rate, recipe, **kwargs)
        after = flatten(report_mod.measure(processed, sample_rate).to_dict())

        out.append((value, {
            key: after[key] - before[key] for key in before.keys() & after.keys()
        }))
    return out


def analyse(
    perturbation: Perturbation,
    by_source: dict[str, list[tuple[float, dict[str, float]]]],
) -> list[FieldEffect]:
    """Rank report fields by how well the parameter can be read off them."""
    fields: set[str] = set()
    for sweep in by_source.values():
        for _, deltas in sweep:
            fields |= deltas.keys()

    effects: list[FieldEffect] = []
    for field in sorted(fields):
        correlations: list[float] = []
        abs_correlations: list[float] = []
        slopes: list[float] = []
        spans: list[float] = []

        for sweep in by_source.values():
            pairs = [(v, d[field]) for v, d in sweep if field in d]
            if len(pairs) < 3:
                continue
            values = np.array([p[0] for p in pairs])
            deltas = np.array([p[1] for p in pairs])

            correlations.append(_pearson(values, deltas))
            abs_correlations.append(_pearson(np.abs(values), np.abs(deltas)))
            spans.append(float(np.mean(np.abs(deltas))))

            spread = float(np.var(values))
            if spread > 1e-30:
                slopes.append(float(np.cov(values, deltas, bias=True)[0, 1] / spread))

        if not correlations:
            continue

        effects.append(FieldEffect(
            field=field,
            slope=float(np.mean(slopes)) if slopes else 0.0,
            # Averaging the correlations rather than pooling the points: a
            # source whose deltas happen to be large would otherwise dominate,
            # and the question is whether the RELATIONSHIP holds everywhere.
            r=float(np.mean(correlations)),
            r_abs=float(np.mean(abs_correlations)),
            span=float(np.mean(spans)) if spans else 0.0,
        ))

    # Sorted by correlation, but anything too small to appear in the report is
    # pushed below everything that is visible.
    effects.sort(key=lambda e: (e.visible, max(abs(e.r), abs(e.r_abs))), reverse=True)
    return effects


def hypothesis_verdict(
    perturbation: Perturbation, effects: list[FieldEffect]
) -> tuple[bool, list[str]]:
    """Did the fields PREDICTED to move actually move?

    The prediction is written in the perturbation catalogue before anything
    runs. Without it this analysis degenerates into reading whatever moved and
    calling it the answer — the failure mode the whole harness exists to avoid,
    and the one that let CLAP-guided EQ optimisation be published as an
    improvement while scoring worse than random parameters.
    """
    by_field = {e.field: e for e in effects}
    missing = [
        field for field in perturbation.expected_fields
        if field not in by_field or not by_field[field].diagnostic
    ]
    return (not missing), missing
