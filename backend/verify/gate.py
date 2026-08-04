"""The post-render safety gate.

THE LAST THING THAT RUNS, AND THE ONLY ONE THAT SEES THE ACTUAL AUDIO. Every
other check in this system reasons about a graph: the schema checks its shape,
the validators check its intent against the measurements, and both run before
anything is rendered. This one measures what came out.

That distinction is the point. A graph can be structurally perfect, semantically
justified, within budget, correctly ordered — and still produce a render with a
resonance nobody asked for, because two gentle boosts landed on the same
frequency and a compressor's makeup pushed the sum into a limiter that was
already working. No amount of pre-render reasoning catches that. Re-measuring
does.

ANY FAILURE MEANS THE RENDER DOES NOT SHIP. Not a warning, not a note in the
explanation — the render is discarded and the previous accepted graph is used
instead. This is what makes it safe to let a language model near a paying
engineer's audio: whatever it proposes, something deterministic listens to the
result before the user does.

THE CHECKS ARE NOT SYMMETRICAL WITH THE PRE-RENDER ONES, and they should not be.
A pre-render validator asks "is this a sensible thing to do?" and can afford to
be opinionated. This one asks "did something go wrong?" and has to be nearly
impossible to trip by accident, because a false failure means a user's perfectly
good render is thrown away and replaced with an older one. Every threshold here
is set where a competent engineer would agree something is broken, not where they
would have made a different choice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backend.measure import report as report_mod
from backend.measure import spectrum

# Loudness has to land close enough that a listener would not reach for the
# volume, but the tolerance is not tight: LUFS is a gated average and a render
# whose tail differs slightly from the excerpt it was planned on will move by a
# few tenths for entirely correct reasons.
LOUDNESS_TOLERANCE_LU = 0.5

# Below this the mono sum is cancelling, whatever the stereo image sounds like.
MIN_CORRELATION = -0.2

# Beyond the arithmetic 3 dB. One dB of real cancellation is audible on a phone.
#
# MIN_CORRELATION above is the same bar in different units and is kept only for
# the message text: for equal-power channels a correlation of -0.2 IS a mono
# loss of 0.969 dB. Do not add a second check on it.
MAX_MONO_LOSS_DB = 1.0

# Per band, deliberately looser. Any stereo widening costs a band a decibel or
# two on mono; six is the band disappearing rather than thinning.
MAX_BAND_MONO_LOSS_DB = 6.0

# A band standing this far above BOTH its neighbours, in the output and not in
# the source, is a resonance rather than a tonal choice. Deliberately high: a
# deliberate 6 dB bell with a moderate Q does not reach it, because a bell's
# skirts lift its neighbours too.
RESONANCE_DB = 9.0

# If the render is quieter than the source by more than this and nothing in the
# graph asked for it, something collapsed.
MAX_UNEXPLAINED_DROP_DB = 12.0


@dataclass
class Failure:
    code: str
    message: str
    # What the user should be told. The engineering detail goes in `message`;
    # this is what appears in the interface, because "the render did not pass
    # its safety checks and the previous version was kept" is more useful to a
    # musician than a correlation coefficient.
    user_message: str = ""


@dataclass
class GateResult:
    passed: bool
    failures: list[Failure] = field(default_factory=list)
    measurements: dict[str, Any] = field(default_factory=dict)
    # Checks that ran and found nothing, recorded so a passing gate is
    # distinguishable from a gate that silently skipped everything.
    checked: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.passed:
            return f"passed {len(self.checked)} checks"
        return "; ".join(f.code for f in self.failures)


def _limiter_ceiling(graph: Any) -> float | None:
    """The ceiling the graph asked for, if it asked for one."""
    if graph is None:
        return None
    for node in getattr(graph, "master", []) or []:
        if node.type == "true_peak_limiter" and not node.bypass:
            return float(node.params.get("ceiling_dbtp", -1.0))
    return None


def _source_band_loss(source_report: dict[str, Any] | None,
                      centre_hz: float) -> float | None:
    """The source's own mono loss in the band nearest `centre_hz`.

    Matched by nearest centre rather than by index, because the two reports are
    only guaranteed to use the same band layout as long as nobody changes it,
    and an index match would silently compare different frequencies the day
    somebody does.
    """
    bands = ((source_report or {}).get("stereo") or {}).get("per_band") or []
    best, best_gap = None, float("inf")
    for band in bands:
        their_centre = band.get("centre_hz")
        if their_centre is None or band.get("mono_loss_db") is None:
            continue
        gap = abs(their_centre - centre_hz)
        if gap < best_gap:
            best, best_gap = band, gap
    # A quarter octave. Further apart than that and it is not the same band.
    if best is None or best_gap > centre_hz * 0.19:
        return None
    return float(best["mono_loss_db"])


def _expected_gain_db(graph: Any) -> float:
    """Roughly how much level the graph intends to add.

    Used only to decide whether a large level DROP is explained. Deliberately
    crude and deliberately one-directional: it exists to avoid failing a render
    that was supposed to get quieter, not to predict the output level.
    """
    if graph is None:
        return 0.0
    total = 0.0
    for node in getattr(graph, "nodes", []) or []:
        if node.bypass:
            continue
        if node.type == "gain":
            total += float(node.params.get("gain_db", 0.0))
        elif node.type == "compressor":
            total += float(node.params.get("makeup_gain_db", 0.0))
        elif node.type == "drive":
            total += float(node.params.get("output_trim_db", 0.0))
    return total


def _narrow_resonances(
    source_report: dict[str, Any] | None, output_report: dict[str, Any]
) -> list[tuple[float, float]]:
    """Bands standing far above both neighbours in the output but not the source.

    COMPARED AGAINST THE SOURCE, not judged in isolation. Plenty of instruments
    have a strong resonance of their own — a snare, a tom, a resonant acoustic
    guitar body — and failing a render because the source had a peak would throw
    away good work for a property of the recording.

    What this looks for is a peak the processing INTRODUCED: a filter oscillating,
    two boosts landing on the same frequency, or a feedback path that is not quite
    stable.
    """
    output_bands = (output_report.get("spectrum") or {}).get("third_octave_db") or []
    if len(output_bands) < 3:
        return []

    source_lookup: dict[float, float] = {}
    for band in ((source_report or {}).get("spectrum") or {}).get("third_octave_db") or []:
        if band.get("level_db") is not None:
            source_lookup[float(band["centre_hz"])] = float(band["level_db"])

    found: list[tuple[float, float]] = []
    for index in range(1, len(output_bands) - 1):
        centre = output_bands[index]
        below, above = output_bands[index - 1], output_bands[index + 1]
        levels = [b.get("level_db") for b in (below, centre, above)]
        if any(level is None for level in levels):
            continue

        prominence = min(float(levels[1]) - float(levels[0]),
                         float(levels[1]) - float(levels[2]))
        if prominence < RESONANCE_DB:
            continue

        # Was it already there? Compare the same prominence in the source.
        freqs = [float(b["centre_hz"]) for b in (below, centre, above)]
        if all(f in source_lookup for f in freqs):
            source_prominence = min(source_lookup[freqs[1]] - source_lookup[freqs[0]],
                                    source_lookup[freqs[1]] - source_lookup[freqs[2]])
            if source_prominence > prominence - 3.0:
                continue  # the source had it; not ours

        found.append((float(centre["centre_hz"]), prominence))
    return found


def verify(
    rendered: np.ndarray,
    sample_rate: float,
    *,
    graph: Any = None,
    source_report: dict[str, Any] | None = None,
    target_lufs: float | None = None,
    output_report: dict[str, Any] | None = None,
) -> GateResult:
    """Measure the render and decide whether it may ship.

    `graph` and `source_report` are optional, and every check that needs one is
    skipped without it rather than guessed at — but the checks that need
    neither, which are the ones that catch genuine corruption, always run.
    """
    rendered = np.atleast_2d(np.asarray(rendered, dtype=np.float64))
    report = output_report or report_mod.measure(rendered, sample_rate).to_dict()

    result = GateResult(passed=True, measurements=report)
    failures: list[Failure] = []

    def fail(code: str, message: str, user_message: str) -> None:
        failures.append(Failure(code, message, user_message))

    # --- corruption: always checked, never skipped -------------------------
    result.checked.append("finite_samples")
    non_finite = int((report.get("hygiene") or {}).get("non_finite_samples", 0) or 0)
    if non_finite:
        fail("non_finite",
             f"{non_finite} samples are NaN or infinite",
             "The processed audio came out corrupted, so it was not used.")

    result.checked.append("not_silent")
    peak = (report.get("loudness") or {}).get("sample_peak_dbfs")
    if peak is None or peak < -80.0:
        fail("silent",
             f"the render peaks at {peak if peak is not None else float('-inf')} dBFS",
             "The processed audio came out silent, so it was not used.")

    # --- true peak against the ceiling the graph asked for -----------------
    ceiling = _limiter_ceiling(graph)
    true_peak = (report.get("loudness") or {}).get("true_peak_dbtp")
    if true_peak is not None:
        result.checked.append("true_peak")
        # Without a limiter the bar is 0 dBTP: anything above that clips on
        # conversion no matter what the graph intended.
        limit = ceiling if ceiling is not None else 0.0
        # A tenth of a dB of slack, because the limiter and this meter are
        # separate implementations and agreeing to the last decimal is not a
        # requirement either of them was built to meet.
        if true_peak > limit + 0.1:
            fail("true_peak",
                 f"true peak is {true_peak:+.2f} dBTP against a {limit:+.2f} dBTP "
                 "ceiling",
                 "The processed audio would clip on playback, so it was not used.")

    # --- loudness against the target, if one was set -----------------------
    integrated = (report.get("loudness") or {}).get("integrated_lufs")
    if target_lufs is not None:
        result.checked.append("loudness_target")
        if integrated is None:
            fail("no_loudness",
                 "the render has no measurable integrated loudness",
                 "The processed audio came out too quiet to measure.")
        elif abs(integrated - target_lufs) > LOUDNESS_TOLERANCE_LU:
            fail("loudness_target",
                 f"integrated loudness is {integrated:.2f} LUFS against a target "
                 f"of {target_lufs:.2f}",
                 "The processed audio did not reach the requested loudness.")

    # --- an unexplained collapse in level ----------------------------------
    source_loudness = ((source_report or {}).get("loudness") or {}).get("integrated_lufs")
    if source_loudness is not None and integrated is not None:
        result.checked.append("level_collapse")
        # Compared against what the graph MEANT to do. A chain with -10 dB of
        # gain in it is supposed to come out quieter.
        drop = source_loudness + _expected_gain_db(graph) - integrated
        if drop > MAX_UNEXPLAINED_DROP_DB:
            fail("level_collapse",
                 f"the render is {drop:.1f} dB quieter than the chain accounts for",
                 "The processed audio lost most of its level unexpectedly.")

    # --- mono compatibility -------------------------------------------------
    stereo = report.get("stereo") or {}
    if rendered.shape[0] >= 2 and stereo.get("type") not in (None, "mono"):
        result.checked.append("mono_compatibility")
        correlation = stereo.get("correlation")
        mono_loss = stereo.get("mono_loss_db")

        # ONE CHECK, NOT TWO. These used to be a correlation test and a mono-loss
        # test in an if/elif, which read as two independent safeguards and was
        # not: for equal-power channels the two quantities are the same number
        # in different units. At exactly the correlation bar of -0.2 the mono
        # loss is 0.969 dB, against a bar of 1.0 — so the second branch could
        # never run. Verified by sweep as well as algebraically, and unequal
        # channel powers only shrink the cross term, which makes it less
        # reachable rather than more.
        #
        # Two checks that are secretly one are worse than one check, because
        # they make a coverage list look longer than it is.
        #
        # The surviving test is expressed in mono loss, because that is the
        # audible consequence — decibels missing on a phone — and correlation is
        # reported alongside as the reason.
        if mono_loss is not None and mono_loss > MAX_MONO_LOSS_DB:
            detail = (f"summing to mono loses {mono_loss:.1f} dB beyond the "
                      f"arithmetic 3")
            if correlation is not None:
                detail += f", with channel correlation {correlation:+.2f}"
            fail("out_of_phase", detail,
                 "The processed audio would partly cancel on mono playback.")

    # --- a band that cancels while the whole file looks fine ----------------
    #
    # GENUINELY A DIFFERENT CHECK, unlike the one above. Wideband correlation is
    # an average, and an average hides a band: a widened low-mid can vanish on a
    # phone while the file as a whole correlates perfectly well. This is the
    # case a mastering engineer actually worries about, and nothing was looking
    # for it.
    #
    # The bar is deliberately looser than the wideband one. A single narrow band
    # losing 2 dB is a normal consequence of any stereo widening; losing 6 dB is
    # the band disappearing.
    bands = (report.get("stereo") or {}).get("per_band") or []
    if bands:
        result.checked.append("per_band_mono")
        # THE WORST BAND, not every band. Material that is broadly out of phase
        # trips all of them, and eight near-identical failures describe one
        # problem worse than one failure does.
        worst: tuple[float, float, float | None] | None = None
        for band in bands:
            loss = band.get("mono_loss_db")
            centre = band.get("centre_hz")
            if loss is None or centre is None or loss <= MAX_BAND_MONO_LOSS_DB:
                continue
            # Only if the SOURCE did not already have it. Cancellation that
            # arrived with the file is not something this render did, and
            # refusing to process already-phasey material would be refusing the
            # material most in need of help.
            source_loss = _source_band_loss(source_report, centre)
            if source_loss is not None and loss <= source_loss + 1.0:
                continue
            if worst is None or loss > worst[1]:
                worst = (float(centre), float(loss), source_loss)
        if worst is not None:
            centre, loss, source_loss = worst
            fail("band_mono_loss",
                 f"the {centre:.0f} Hz band loses {loss:.1f} dB summing to mono"
                 + (f", against {source_loss:.1f} dB in the source"
                    if source_loss is not None else ""),
                 "Part of the processed audio would disappear on mono playback.")

    # --- a resonance the processing introduced ------------------------------
    result.checked.append("resonance")
    for centre, prominence in _narrow_resonances(source_report, report):
        fail("resonance",
             f"a {prominence:.1f} dB peak at {centre:.0f} Hz that the source did "
             "not have",
             "The processing produced a ringing tone, so it was not used.")

    # --- bandwidth that the processing destroyed ----------------------------
    if source_report:
        result.checked.append("bandwidth")
        source_rolloff = (source_report.get("spectrum") or {}).get(
            "hf_rolloff_db_per_octave")
        output_rolloff = (report.get("spectrum") or {}).get("hf_rolloff_db_per_octave")
        if source_rolloff is not None and output_rolloff is not None:
            # Only a dramatic darkening counts. A chain is often supposed to
            # remove top end, so this is set where an engineer would say the
            # source had been destroyed rather than shaped.
            if output_rolloff < source_rolloff - 18.0:
                fail("bandwidth_collapse",
                     f"the high-frequency rolloff went from {source_rolloff:.1f} to "
                     f"{output_rolloff:.1f} dB/octave",
                     "The processing removed most of the high frequencies.")

    result.failures = failures
    result.passed = not failures
    return result


def verify_or_fall_back(
    rendered: np.ndarray,
    sample_rate: float,
    *,
    previous_render: np.ndarray | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray | None, GateResult]:
    """Gate a render, returning the previous one if it fails.

    Returns `(audio_to_ship, result)`. The audio is `None` when the render fails
    and there is nothing to fall back to — which the caller must treat as an
    error rather than shipping silence, since a gate that quietly returns
    nothing is worse than one that fails loudly.
    """
    result = verify(rendered, sample_rate, **kwargs)
    if result.passed:
        return rendered, result
    return previous_render, result
