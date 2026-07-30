"""Semantic validation of an emitted effect graph.

The schema catches graphs that are malformed. This catches graphs that are
well-formed and WRONG — and that is the harder and more valuable half, because a
malformed graph fails to render while a wrong one renders beautifully and sounds
bad.

Everything here is deterministic. None of it asks a model to check its own work.

TWO OUTCOMES, AND THE DISTINCTION MATTERS FOR COST. A violation is either:

  REPAIRABLE — there is exactly one sensible fix, so Python applies it and logs
  what it did. Clamping an out-of-range value, sorting the master section,
  removing a no-op band. Burning a model retry on these would cost a second
  round-trip to arrive at the answer we already have.

  FATAL — the graph expresses an intent that cannot be salvaged without
  guessing what was meant. These trigger a bounded repair loop: at most two
  retries, then fall back to the previous accepted graph.

A rising auto-repair rate is a prompt-quality signal, which is why repairs are
logged rather than applied silently. The service-level indicators from the plan:
repair rate under 15%, retry rate under 5%, fallback rate under 0.5%.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from backend.schemas.graph import (
    MASTER_ONLY,
    PARAM_MODELS,
    DynamicEqBand,
    EffectGraph,
    EqBand,
    GraphNode,
)

# The single most important number in this file.
#
# Total additive EQ boost across the whole graph. Exceeding it is the commonest
# cause of a chain that "sounds like an EQ" rather than like a record: each move
# is defensible on its own, and the sum is a bright, thin, obviously-processed
# result. 12 dB is generous — most good mixes stay under 8.
TOTAL_BOOST_BUDGET_DB = 12.0

# A biquad whose poles sit closer than this to the unit circle rings audibly and
# is numerically fragile at float32 in any downstream host.
MAX_POLE_RADIUS = 0.9995

# Above this, a shelf or bell has nothing to act on if the source stops below it.
AIR_BAND_HZ = 10000.0


@dataclass
class Finding:
    code: str
    severity: str            # "fatal" | "repairable"
    message: str
    node_id: str = ""

    def __str__(self) -> str:
        where = f" [{self.node_id}]" if self.node_id else ""
        return f"{self.severity}:{self.code}{where}: {self.message}"


@dataclass
class ValidationResult:
    graph: EffectGraph
    findings: list[Finding] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)

    @property
    def fatal(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "fatal"]

    @property
    def ok(self) -> bool:
        return not self.fatal

    def summary(self) -> str:
        if self.ok and not self.repairs:
            return "graph accepted with no changes"
        parts = []
        if self.repairs:
            parts.append(f"{len(self.repairs)} repair(s)")
        if self.fatal:
            parts.append(f"{len(self.fatal)} fatal")
        return "; ".join(parts)


# ---------------------------------------------------------------------------
# 1. Ranges
#
# The schema's Pydantic layer already enforces these on construction, so this
# exists for graphs assembled in code or patched in place — and as the last line
# of defence if a future path ever builds a graph without going through Pydantic.


def _bounds(model: Any) -> dict[str, tuple[float, float, Any]]:
    """Extract (min, max, default) per field from a Pydantic model.

    READ FROM THE MODEL, never from a table kept alongside it. This function
    replaced a hand-maintained name-to-range map, which had already diverged: a
    band's `gain_db` is bounded at plus or minus 18 while a gain node's is plus
    or minus 24, and a flat map keyed on the NAME cannot tell them apart. It
    silently applied the wider bound to bands, so a 22 dB band gain passed the
    clamp that the schema itself would have rejected.

    Names are not unique across contexts, so any lookup that ignores context is
    wrong the moment two nodes share a parameter name — which they already do.
    """
    out: dict[str, tuple[float, float, Any]] = {}
    for name, info in model.model_fields.items():
        low = next((m.ge for m in info.metadata if getattr(m, "ge", None) is not None), None)
        high = next((m.le for m in info.metadata if getattr(m, "le", None) is not None), None)
        if low is None or high is None:
            continue
        out[name] = (float(low), float(high), info.default)
    return out


def _check_ranges(graph: EffectGraph, result: ValidationResult) -> None:
    def clamp(params: dict[str, Any], bounds: dict[str, tuple[float, float, Any]],
              where: str) -> None:
        for key, value in list(params.items()):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            if key not in bounds:
                continue
            low, high, default = bounds[key]

            if not math.isfinite(value):
                params[key] = default
                result.repairs.append(
                    f"{where}.{key}: non-finite value replaced with the default")
                continue
            if value < low or value > high:
                params[key] = min(max(float(value), low), high)
                result.repairs.append(
                    f"{where}.{key}: {value:g} clamped to {params[key]:g} "
                    f"(range {low:g} to {high:g})")

    for node in list(graph.nodes) + list(graph.master):
        model = PARAM_MODELS.get(node.type)
        if model is None:
            continue
        clamp(node.params, _bounds(model), node.id)

        band_model = DynamicEqBand if node.type == "dynamic_eq" else EqBand
        band_bounds = _bounds(band_model)
        for index, band in enumerate(node.params.get("bands", []) or []):
            if isinstance(band, dict):
                clamp(band, band_bounds, f"{node.id}.bands[{index}]")


# ---------------------------------------------------------------------------
# 2. Filter stability


def _check_filter_stability(graph: EffectGraph, result: ValidationResult) -> None:
    """Reject filters whose poles sit on the unit circle.

    A very high Q at a very low frequency is the dangerous combination: the pole
    radius approaches 1 as the corner approaches DC, so a Q of 12 at 20 Hz rings
    for seconds and is numerically fragile in any float32 host downstream.

    The engine's TPT/ZDF topology is unconditionally stable, so this is not
    protecting the engine — it is protecting the RESULT. A filter that does not
    blow up can still ring audibly.
    """
    for node in list(graph.nodes) + list(graph.master):
        for index, band in enumerate(node.params.get("bands", []) or []):
            if not isinstance(band, dict):
                continue
            freq = float(band.get("center_freq_hz", 1000.0))
            q = float(band.get("q_factor", 0.707))

            # Bilinear-transform pole radius at the engine's internal rate.
            omega = 2.0 * math.pi * freq / 96000.0
            alpha = math.sin(omega) / (2.0 * max(q, 1e-6))
            radius = math.sqrt(max(0.0, (1.0 - alpha) / (1.0 + alpha)))

            if radius > MAX_POLE_RADIUS:
                suggested = max(0.5, q / 4.0)
                band["q_factor"] = suggested
                result.repairs.append(
                    f"{node.id}.bands[{index}]: Q {q:g} at {freq:g} Hz puts the "
                    f"pole at radius {radius:.5f}, which rings audibly; reduced "
                    f"to Q {suggested:g}")


# ---------------------------------------------------------------------------
# 3. Total boost budget


def _check_boost_budget(graph: EffectGraph, result: ValidationResult) -> None:
    """Sum every positive EQ gain in the graph and cap the total.

    ONLY BOOSTS COUNT. Cuts do not consume the budget: three 4 dB cuts leave a
    result that still sounds like the source, while three 4 dB boosts leave one
    that sounds like an EQ. The asymmetry is the point — it is not an oversight
    that this is not `sum(abs(gain))`.

    Scaling proportionally rather than clipping the largest, so the SHAPE the
    planner chose survives and only its depth changes. Clipping one band would
    silently change the tonal balance it had reasoned about.
    """
    boosts: list[tuple[dict[str, Any], str, float]] = []
    for node in graph.nodes:
        for index, band in enumerate(node.params.get("bands", []) or []):
            if not isinstance(band, dict):
                continue
            for key in ("gain_db", "static_gain_db"):
                gain = float(band.get(key, 0.0) or 0.0)
                if gain > 0.0:
                    boosts.append((band, key, gain))

    total = sum(gain for _, _, gain in boosts)
    if total <= TOTAL_BOOST_BUDGET_DB or not boosts:
        return

    scale = TOTAL_BOOST_BUDGET_DB / total
    for band, key, gain in boosts:
        band[key] = round(gain * scale, 2)
    result.repairs.append(
        f"total EQ boost was {total:.1f} dB against a {TOTAL_BOOST_BUDGET_DB:g} dB "
        f"budget; all boosts scaled by {scale:.2f} to preserve the shape"
    )


# ---------------------------------------------------------------------------
# 4. Gain staging


def _check_gain_staging(
    graph: EffectGraph, report: dict[str, Any] | None, result: ValidationResult
) -> None:
    """Estimate the output level and refuse to hand the limiter an impossible job.

    Crude by design: sum the static gains, the EQ boosts and the makeup, and add
    them to the measured 95th-percentile level. It cannot model compression or
    saturation, so it is a bound rather than a prediction — but it catches the
    case where a chain arrives at the limiter 15 dB hot, where the limiter's only
    option is to pull down so far that nothing of the dynamics survives.

    Needs the measurement report. Without it the check is skipped rather than
    guessed at, because an invented input level would produce invented warnings.
    """
    if not report:
        return

    percentiles = (report.get("dynamics") or {}).get("short_term_rms_percentiles") or {}
    start = percentiles.get("p95_dbfs")
    if start is None:
        return

    projected = float(start)
    for node in graph.nodes:
        if node.bypass:
            continue
        if node.type == "gain":
            projected += float(node.params.get("gain_db", 0.0))
        elif node.type == "compressor":
            projected += float(node.params.get("makeup_gain_db", 0.0))
        elif node.type == "drive":
            projected += float(node.params.get("output_trim_db", 0.0))
        elif node.type in ("parametric_eq", "dynamic_eq"):
            for band in node.params.get("bands", []) or []:
                if not isinstance(band, dict):
                    continue
                gain = max(float(band.get("gain_db", 0.0) or 0.0),
                           float(band.get("static_gain_db", 0.0) or 0.0))
                # Only the largest boost, not the sum: bands at different
                # frequencies do not add at any single frequency.
                projected = max(projected, projected + gain) if gain > 0 else projected

    headroom = 0.0 - projected
    if headroom < -12.0:
        result.findings.append(Finding(
            "gain_staging", "fatal",
            f"the chain projects to about {projected:+.1f} dBFS before the "
            f"limiter, which is {-headroom:.1f} dB over. The limiter would have "
            "to remove so much that the dynamics would not survive; reduce the "
            "boosts or the makeup gain",
        ))
    elif headroom < -3.0:
        result.findings.append(Finding(
            "gain_staging_tight", "repairable",
            f"the chain projects to about {projected:+.1f} dBFS before the limiter",
        ))


# ---------------------------------------------------------------------------
# 5. Chain-order legality


# Ordering rules as (earlier, later, reason). Each is a case where the reverse
# order produces a result that sounds finished and is wrong.
ORDER_RULES: tuple[tuple[str, str, str], ...] = (
    ("dc_block", "drive",
     "a DC offset biases a saturator asymmetrically, so it must be removed first"),
    ("drive", "convolution_reverb",
     "a cabinet or room should filter the saturation, not be saturated by it"),
    ("mono_to_stereo", "stereo_width",
     "width has no side signal to act on until the source is stereo"),
    ("mid_side", "stereo_width",
     "mid/side encoding must happen before width operates on the side signal"),
    ("delay", "plate_reverb",
     "reverb should sit on the repeats; a delay after reverb smears the tail"),
)


def _check_order(graph: EffectGraph, result: ValidationResult) -> None:
    positions: dict[str, int] = {}
    for index, node in enumerate(graph.nodes):
        positions.setdefault(node.type, index)

    for earlier, later, reason in ORDER_RULES:
        if earlier in positions and later in positions:
            if positions[earlier] > positions[later]:
                result.findings.append(Finding(
                    "chain_order", "fatal",
                    f"{earlier} must come before {later}: {reason}",
                ))

    # The master section's own order is fully determined, so it is sorted rather
    # than rejected — there is exactly one right answer and no intent to guess at.
    order = {name: i for i, name in enumerate(MASTER_ONLY)}
    current = [n.type for n in graph.master]
    wanted = sorted(current, key=lambda t: order.get(t, 99))
    if current != wanted:
        graph.master.sort(key=lambda n: order.get(n.type, 99))
        result.repairs.append(
            f"master reordered from {current} to {wanted}: the limiter must run "
            "before dither, and nothing may follow dither"
        )


# ---------------------------------------------------------------------------
# 6. No-ops
#
# A move that cannot be heard is worse than no move: it consumes a band, it
# appears in the explanation shown to the user, and the user hears no change and
# concludes the tool does not work.


def _check_noops(
    graph: EffectGraph, report: dict[str, Any] | None, result: ValidationResult
) -> None:
    spectrum = (report or {}).get("spectrum") or {}
    limit = spectrum.get("bandwidth_limit_hz")
    limit_confidence = spectrum.get("bandwidth_limit_confidence") or 0.0
    stereo = (report or {}).get("stereo") or {}
    channels = ((report or {}).get("source") or {}).get("channels", 2)

    for node in list(graph.nodes):
        # An air shelf above a codec wall does nothing audible at all.
        if node.type in ("parametric_eq", "dynamic_eq") and limit and limit_confidence >= 0.4:
            kept = []
            for band in node.params.get("bands", []) or []:
                if not isinstance(band, dict):
                    kept.append(band)
                    continue
                freq = float(band.get("center_freq_hz", 1000.0))
                gain = float(band.get("gain_db", 0.0) or 0.0)
                if freq >= float(limit) and abs(gain) > 0.0:
                    result.repairs.append(
                        f"{node.id}: dropped a band at {freq:g} Hz — the source "
                        f"stops at {float(limit):g} Hz, so it would be inaudible"
                    )
                    continue
                kept.append(band)
            node.params["bands"] = kept

        # Width on material that has no side signal.
        if node.type == "stereo_width":
            correlation = stereo.get("correlation")
            dual_mono = stereo.get("type") == "dual_mono"
            if channels < 2 or dual_mono or (correlation is not None and correlation > 0.999):
                if float(node.params.get("width_percent", 100.0)) != 100.0:
                    result.findings.append(Finding(
                        "noop_width", "repairable",
                        "the source has no side signal, so width cannot change "
                        "anything; add mono_to_stereo first or drop this node",
                        node.id,
                    ))

        # A bypassed node is a no-op by definition. Legal, but if the planner
        # emitted one it probably meant to omit it.
        if node.bypass:
            result.findings.append(Finding(
                "bypassed_node", "repairable",
                "emitted bypassed, which has no effect; omit it instead",
                node.id,
            ))


# ---------------------------------------------------------------------------
# 7. Source-condition constraints
#
# Things the measurement report says the graph is not allowed to do.


def _check_source_constraints(
    graph: EffectGraph, report: dict[str, Any] | None, result: ValidationResult
) -> None:
    if not report:
        return

    codes = {w.get("code") for w in (report.get("warnings") or [])}
    hygiene = report.get("hygiene") or {}

    if "clipped_source" in codes:
        for node in graph.nodes:
            if node.type == "drive" and float(node.params.get("drive_db", 0.0)) > 0.0:
                result.findings.append(Finding(
                    "drive_on_clipped_source", "fatal",
                    "the source is already clipped, so saturation will amplify "
                    "existing distortion rather than add character",
                    node.id,
                ))

    floor = hygiene.get("noise_floor_dbfs")
    floor_confidence = hygiene.get("noise_floor_confidence") or 0.0
    if floor is not None and floor > -55.0 and floor_confidence >= 0.4:
        for node in graph.nodes:
            if node.type == "gate_expander":
                continue
            for band in node.params.get("bands", []) or []:
                if not isinstance(band, dict):
                    continue
                freq = float(band.get("center_freq_hz", 1000.0))
                gain = float(band.get("gain_db", 0.0) or 0.0)
                if freq >= AIR_BAND_HZ and gain > 3.0:
                    result.findings.append(Finding(
                        "boost_over_noise", "repairable",
                        f"a {gain:+.1f} dB boost at {freq:g} Hz will lift a "
                        f"{floor:.1f} dBFS noise floor into the foreground",
                        node.id,
                    ))

    if "out_of_phase" in codes:
        has_fix = any(n.type in ("polarity", "mid_side") for n in graph.nodes)
        if not has_fix:
            result.findings.append(Finding(
                "unaddressed_phase", "fatal",
                "the source's channels are strongly negatively correlated and "
                "the graph does nothing about it; the mono sum currently "
                "cancels, so fix that before anything else",
            ))

    # Dither must match the delivery format. Getting this wrong is silent: a
    # dithered float export just has added noise, and an undithered 16-bit
    # export has correlated quantisation distortion on every fade.
    for node in graph.master:
        if node.type == "dither":
            depth = float(node.params.get("bit_depth", 24.0))
            if depth not in (0.0, 16.0, 24.0, 32.0):
                node.params["bit_depth"] = 24.0
                result.repairs.append(
                    f"{node.id}: bit_depth {depth:g} is not a delivery word "
                    "length; set to 24"
                )


# ---------------------------------------------------------------------------
# Entry point


def validate(
    graph: EffectGraph, report: dict[str, Any] | None = None
) -> ValidationResult:
    """Run every check. Repairs are applied in place; fatals are reported.

    Order matters: ranges are clamped before anything reads a parameter, and the
    boost budget runs after no-op removal so a band that was going to be dropped
    does not consume budget on its way out.
    """
    graph = graph.model_copy(deep=True)
    result = ValidationResult(graph=graph)

    _check_ranges(graph, result)
    _check_filter_stability(graph, result)
    _check_noops(graph, report, result)
    _check_boost_budget(graph, result)
    _check_order(graph, result)
    _check_gain_staging(graph, report, result)
    _check_source_constraints(graph, report, result)

    return result
