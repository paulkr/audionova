"""Dynamics matching: fit compressor settings by rendering, not by inverting.

A compressor's effect on any dynamics statistic depends on the material's own
distribution, on the attack and release interacting with its envelope, and on the
knee. There is no closed-form inversion, so every parameter that cannot be
derived directly is found by rendering and measuring.

THE CONTROL VARIABLE IS GAIN REDUCTION, NOT LOUDNESS RANGE, and this is a
correction to the obvious design. The plan specified bisecting on rendered LRA.
Measured across three real stems, LRA is not monotonic in ratio and sometimes not
even the right sign:

    vocals   LRA falls 2.95 -> 0.79 up to ratio 4-6, then RISES to 1.04 at 20
    guitar   LRA RISES throughout, 0.84 -> 1.15
    bass     LRA falls 1.20 -> 0.47, monotonic

Bisection requires monotonicity, so on two of those three it would either fail to
converge or converge on the wrong side. The cause is real rather than
measurement noise: past a certain ratio the gain reduction itself becomes a
modulation. Heavy compression with a slowish release is pumping, and pumping is
variance — it ADDS to short-term loudness spread. On material that is already
flat, like the guitar at 7.2 dB crest, that is the only thing compression can do.

Gain reduction at the loud percentile is monotonic on all three, with no
violations anywhere in the sweep, because it follows the static curve directly.
It is also what an engineer already thinks in: "three dB of gain reduction on the
peaks" is a sentence people say.

TIMING COMES FROM THE SOURCE, NOT FROM THE REFERENCE. The plan specified
deriving attack and release from where the source's and reference's envelope
modulation spectra diverge by 3 dB. Measurement rejected that: compression ADDS
modulation energy at the rates in question rather than removing it, so there is
no corner to find. See `timing_from_source_rate` for the numbers.

THE ORDER IS THRESHOLD, TIMING, RATIO, MAKEUP. Threshold and timing are derived
directly from measurements and cost nothing. Only the ratio needs the loop, and
makeup is computed once at the end from what the loop produced — putting makeup
inside the loop would change the level the next iteration measures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from backend.measure import dynamics, loudness

# Ratio search bounds. Above 20:1 a compressor is a limiter and the extra range
# buys nothing a limiter would not do better.
MIN_RATIO = 1.0
MAX_RATIO = 20.0

# Enough to land within about 0.1 dB of gain reduction, which is finer than
# anybody hears and finer than the measurement repeats.
DEFAULT_MAX_RENDERS = 9


@dataclass
class DynamicsSettings:
    threshold_dbfs: float
    ratio_to_1: float
    attack_ms: float
    release_ms: float
    knee_db: float = 6.0
    makeup_gain_db: float = 0.0

    def as_params(self) -> dict[str, float]:
        return {
            "threshold_dbfs": round(self.threshold_dbfs, 1),
            "ratio_to_1": round(self.ratio_to_1, 2),
            "attack_ms": round(self.attack_ms, 1),
            "release_ms": round(self.release_ms, 0),
            "knee_db": round(self.knee_db, 1),
            "makeup_gain_db": round(self.makeup_gain_db, 1),
        }


@dataclass
class DynamicsResult:
    settings: DynamicsSettings
    target_reduction_db: float = 0.0
    achieved_reduction_db: float = 0.0
    achievable_reduction_db: float = 0.0
    renders: int = 0
    # True when the target was beyond what this threshold can deliver at any
    # ratio. Reported rather than silently maxed out: "the material is already
    # too flat for this" is the useful answer, and quietly returning 20:1
    # would hide it.
    unachievable: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def error_db(self) -> float:
        return abs(self.achieved_reduction_db - self.target_reduction_db)


def _loud_level_db(samples: np.ndarray, sample_rate: float,
                   percentile: float = 90.0) -> float:
    """Short-term RMS at a high percentile: where the compressor is working."""
    rms = dynamics.short_term_rms_db(samples, sample_rate)
    finite = rms[np.isfinite(rms)]
    if finite.size == 0:
        return float("-inf")
    return float(np.percentile(finite, percentile))


def threshold_for_duty_cycle(
    samples: np.ndarray, sample_rate: float, duty_cycle: float = 0.3
) -> float:
    """Threshold that puts the compressor to work on the loudest `duty_cycle`.

    DERIVED, NOT SEARCHED. "Compress the loudest 30% of the material" is a
    decision about intent, and once made the threshold follows from the measured
    percentile with no rendering at all. Searching for it would be spending
    renders on arithmetic.

    Duty cycle is the honest parameter here because it is the thing an engineer
    actually decides — a vocal that needs riding wants a low threshold and a high
    duty cycle; a drum bus catching only peaks wants the opposite.
    """
    rms = dynamics.short_term_rms_db(samples, sample_rate)
    finite = rms[np.isfinite(rms)]
    if finite.size == 0:
        return -18.0
    return float(np.percentile(finite, (1.0 - duty_cycle) * 100.0))


def timing_from_source_rate(
    source: np.ndarray,
    sample_rate: float,
    *,
    fallback_attack_ms: float = 15.0,
    fallback_release_ms: float = 150.0,
) -> tuple[float, float, float]:
    """Attack and release from how fast the SOURCE's own level moves.

    NOT from comparing the source's modulation spectrum against a reference's,
    which is what the plan specified and what measurement rejected. The
    reasoning was that a compressor low-passes the amplitude envelope, so the
    frequency where the reference's modulation falls 3 dB below the source's is
    the corner. Measured against the engine on a source with envelope energy
    spread across 0.3 to 25 Hz, at 8:1:

        attack 1 ms / release 30 ms     -1.3 dB at 2 Hz, +1.0 dB at 30 Hz
        attack 15 ms / release 150 ms   +3.8 dB at 4 Hz, +3.2 dB at 30 Hz
        attack 80 ms / release 800 ms   +2.3 dB at 4 Hz, +1.3 dB at 30 Hz

    Compression ADDS modulation at those rates rather than removing it, because
    the gain movement is itself a modulation — the same mechanism that makes
    loudness range rise under heavy compression. There is no 3 dB corner to
    find, and a first version of this function duly returned 582 ms for the fast
    compressor and 43 ms for the slow one: reading a time constant out of noise,
    inverted.

    What IS measurable is the source's own modulation rate, and it is the more
    useful quantity anyway. A compressor should move at the speed of the
    material: a source whose level changes several times a second wants a
    release short enough to recover between events, and one that swells slowly
    wants a release that does not chatter. So the time constants come from the
    source alone and the reference is not consulted for them at all.

    Returns `(attack_ms, release_ms, confidence)`.
    """
    freqs, power = dynamics.envelope_modulation_spectrum(source, sample_rate)
    summary = dynamics.modulation_summary(freqs, power)
    centroid = float(summary.get("centroid_hz", 0.0))

    if freqs.size < 8 or centroid <= 0.05:
        return fallback_attack_ms, fallback_release_ms, 0.0

    # Release long enough to span one cycle of the source's typical level
    # movement, so the compressor recovers between events rather than during
    # them. Attack a fraction of that, which is the conventional asymmetry and
    # what the ear expects.
    period_ms = 1000.0 / centroid
    release_ms = float(np.clip(period_ms * 0.5, 30.0, 800.0))
    attack_ms = float(np.clip(release_ms * 0.08, 1.0, 60.0))

    return attack_ms, release_ms, float(summary.get("confidence", 0.0))


def target_reduction_from_reference(
    source: np.ndarray,
    reference: np.ndarray,
    sample_rate: float,
    *,
    percentile: float = 90.0,
) -> float:
    """How much gain reduction would bring the source's dynamics to the reference's.

    Measured as the change in the gap between loud and quiet material, so a pure
    level difference between the two contributes nothing. The reference being
    6 dB louder is a gain decision, not a dynamics one.
    """
    def spread(samples: np.ndarray) -> float:
        loud = _loud_level_db(samples, sample_rate, percentile)
        quiet = _loud_level_db(samples, sample_rate, 100.0 - percentile)
        return loud - quiet if math.isfinite(loud) and math.isfinite(quiet) else 0.0

    return max(0.0, spread(source) - spread(reference))


def match(
    source: np.ndarray,
    sample_rate: float,
    render: Callable[[dict[str, float]], np.ndarray],
    *,
    target_reduction_db: float | None = None,
    reference: np.ndarray | None = None,
    duty_cycle: float = 0.3,
    knee_db: float = 6.0,
    attack_ms: float | None = None,
    release_ms: float | None = None,
    percentile: float = 90.0,
    max_renders: int = DEFAULT_MAX_RENDERS,
    tolerance_db: float = 0.15,
) -> DynamicsResult:
    """Find compressor settings by rendering.

    `render` takes a compressor parameter dict and returns the processed audio.
    Injected rather than imported so the loop can be driven by the real engine in
    production and by anything else in a test — and so this module has no opinion
    about how a render happens.
    """
    notes: list[str] = []

    # --- threshold: derived, no rendering ----------------------------------
    threshold = threshold_for_duty_cycle(source, sample_rate, duty_cycle)

    # --- timing: derived from the reference if there is one -----------------
    if attack_ms is None or release_ms is None:
        derived_attack, derived_release, confidence = timing_from_source_rate(
            source, sample_rate)
        if confidence < 0.4:
            notes.append(
                f"the source's modulation rate is not measurable with confidence "
                f"({confidence:.2f}); using default time constants")
            derived_attack, derived_release = 15.0, 150.0
        attack_ms = attack_ms if attack_ms is not None else derived_attack
        release_ms = release_ms if release_ms is not None else derived_release

    # --- target -------------------------------------------------------------
    if target_reduction_db is None:
        if reference is None:
            raise ValueError(
                "dynamics matching needs either a target reduction or a reference")
        target_reduction_db = target_reduction_from_reference(
            source, reference, sample_rate, percentile=percentile)

    source_loud = _loud_level_db(source, sample_rate, percentile)
    renders = 0

    def reduction_at(ratio: float) -> tuple[float, np.ndarray]:
        nonlocal renders
        renders += 1
        settings = DynamicsSettings(threshold, ratio, attack_ms, release_ms, knee_db)
        rendered = render(settings.as_params())
        return source_loud - _loud_level_db(rendered, sample_rate, percentile), rendered

    # --- is the target even reachable? -------------------------------------
    #
    # Asked FIRST, and it costs one render. Bisecting toward something
    # unreachable spends the whole budget converging on the bound and then
    # reports 20:1 as though it were a decision.
    max_reduction, _ = reduction_at(MAX_RATIO)
    unachievable = target_reduction_db > max_reduction + tolerance_db
    if unachievable:
        notes.append(
            f"the material only yields {max_reduction:.2f} dB of reduction at this "
            f"threshold even at {MAX_RATIO:g}:1, against a target of "
            f"{target_reduction_db:.2f} dB — it is already too flat, or the "
            "threshold is too high"
        )

    # --- ratio by bisection on GAIN REDUCTION -------------------------------
    #
    # Monotonic in ratio on every stem measured, because it follows the static
    # curve. Loudness range is not, which is why it is not the control variable.
    low, high = MIN_RATIO, MAX_RATIO
    best_ratio, best_reduction = MAX_RATIO, max_reduction

    if not unachievable and target_reduction_db > tolerance_db:
        for _ in range(max(0, max_renders - renders)):
            middle = 0.5 * (low + high)
            reduction, _ = reduction_at(middle)

            if abs(reduction - target_reduction_db) < abs(best_reduction - target_reduction_db):
                best_ratio, best_reduction = middle, reduction

            if abs(reduction - target_reduction_db) <= tolerance_db:
                break
            if reduction < target_reduction_db:
                low = middle
            else:
                high = middle
    elif target_reduction_db <= tolerance_db:
        best_ratio, best_reduction = MIN_RATIO, 0.0
        notes.append("no meaningful reduction was asked for; ratio left at 1:1")

    # --- makeup: computed once, at the end ----------------------------------
    #
    # Inside the loop it would change the level the next iteration measures, and
    # the loop would be chasing its own tail.
    settings = DynamicsSettings(threshold, best_ratio, attack_ms, release_ms, knee_db)
    rendered = render(settings.as_params())
    renders += 1

    source_loudness = loudness.integrated_lufs(source, sample_rate)
    output_loudness = loudness.integrated_lufs(rendered, sample_rate)
    if math.isfinite(source_loudness) and math.isfinite(output_loudness):
        settings.makeup_gain_db = float(np.clip(
            source_loudness - output_loudness, -24.0, 24.0))

    return DynamicsResult(
        settings=settings,
        target_reduction_db=float(target_reduction_db),
        achieved_reduction_db=float(best_reduction),
        achievable_reduction_db=float(max_reduction),
        renders=renders,
        unachievable=unachievable,
        notes=notes,
    )


def engine_renderer(
    source: np.ndarray, sample_rate: float
) -> Callable[[dict[str, float]], np.ndarray]:
    """A `render` callable backed by the real engine.

    Imported lazily so this module can be exercised without a built binary.
    """
    from backend.engine.render import Recipe, render_array

    def render(params: dict[str, float]) -> np.ndarray:
        recipe = Recipe(output_sample_rate=sample_rate, output_format="float64")
        recipe.add("compressor", **params)
        return render_array(source, sample_rate, recipe)

    return render
