"""A deterministic planner: measurement report in, effect graph out.

WHAT THIS IS FOR. On 31 July a listener preferred a chain written from a
guitar's measured spectrum over a generic one. I wrote that chain by hand,
reading the report and deciding. This encodes the same reasoning so it runs
without me, and it exists for two reasons.

  IT IS THE FLOOR THE MODEL HAS TO BEAT. Without it, a planner that scores well
  is unreadable: we cannot tell whether it reasoned from the measurements or
  learned that acoustic guitars want a cut around 250 Hz. The recovery harness
  already established this pattern with a rule-based recogniser at F1 0.580, and
  the same argument applies harder here, because the output is audio somebody
  pays for.

  IT IS A FALLBACK THAT SHIPS. A model call can fail, time out, or cost more
  than a turn is worth. A deterministic chain that is defensible on every move
  is a better answer than an error.

WHAT IT IS NOT. It has no world knowledge, no sense of genre, and no idea what a
1965 British combo sounds like. It cannot read a sentence. Everything it knows is
in this file, and the interesting half of the product — "make it sound like that
record" — is precisely what it cannot do. If the model does not comfortably beat
this, the model is not earning its cost.

EVERY MOVE NAMES ITS NUMBER. Each node carries a rationale citing the
measurement that triggered it, in the same field the model is asked to fill.
That is not decoration: a rule that cannot say why it fired is a rule nobody can
argue with, and this file exists to be argued with.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backend.schemas.graph import EffectGraph, GraphNode

# Below this, a source has essentially no content and a high-pass is cleanup
# rather than surgery.
QUIET_BAND_DB = 8.0

# How far a band must peak above its IMMEDIATE NEIGHBOURS before it counts as a
# resonance. Measured against the distant midrange instead, this rule proposed
# cutting 100 Hz on a bass guitar by 5 dB — see choose_resonance.
RESONANCE_THRESHOLD_DB = 4.0

# Never place a high-pass above this, whatever the spectrum says. A male vocal
# fundamental reaches 85 Hz and a guitar's low E is 82 Hz, so a corner much
# above 120 removes fundamentals on most sources — and the step-detection below
# will happily suggest 200 Hz on a vocal whose energy starts there.
MAX_HIGHPASS_HZ = 120.0

# Peak-to-loudness thresholds. Above the first the material is unprocessed and
# genuinely dynamic; below the second it has already been squashed and more
# compression removes what is left rather than controlling anything.
VERY_DYNAMIC_PLR_DB = 18.0
ALREADY_COMPRESSED_PLR_DB = 10.0

# A source has to be darker than this before brightening is a measurement-backed
# move rather than a taste one.
DARK_ROLLOFF_DB_PER_OCTAVE = -12.0


@dataclass
class PlanExplanation:
    """Why the planner did what it did, and what it deliberately did not do.

    The omissions matter as much as the moves. A chain of four well-chosen
    things beats a chain of nine defensible ones, and a planner that cannot say
    what it declined is a planner nobody can audit.
    """

    moves: list[str] = field(default_factory=list)
    declined: list[str] = field(default_factory=list)

    def text(self) -> str:
        parts = list(self.moves)
        if self.declined:
            parts.append("Deliberately left alone: " + "; ".join(self.declined) + ".")
        return " ".join(parts)


def _bands(report: dict[str, Any]) -> dict[float, float]:
    out: dict[float, float] = {}
    for band in (report.get("spectrum") or {}).get("third_octave_db") or []:
        if band.get("level_db") is not None:
            out[float(band["centre_hz"])] = float(band["level_db"])
    return out


def _midrange_reference(bands: dict[float, float]) -> float | None:
    inside = [v for k, v in bands.items() if 500.0 <= k <= 4000.0]
    return float(np.mean(inside)) if inside else None


def choose_highpass(bands: dict[float, float], reference: float) -> tuple[float, str] | None:
    """Where the source's own low end actually begins.

    NOT A FIXED FREQUENCY, which is the mistake that made a vocal chain wrong on
    a guitar: 90 Hz is cleanup on a voice and surgery on an instrument whose low
    E fundamental is 82 Hz.

    The corner goes below the largest upward step in the low bands, which is
    where the source's content starts. Measured across four stems that lands at
    50 Hz on acoustic guitar, 25 Hz on bass, and 100 Hz on a vocal — the right
    answer in each case, from one rule.
    """
    low = sorted(k for k in bands if k <= 400.0)
    if len(low) < 4:
        return None

    relative = [(k, bands[k] - reference) for k in low]
    steps = [(relative[i + 1][0], relative[i + 1][1] - relative[i][1], i)
             for i in range(len(relative) - 1)]
    frequency, rise, index = max(steps, key=lambda s: s[1])

    if rise < 6.0:
        return None  # no clear onset; the low end fades in rather than starting

    below = relative[index]

    # Only worth doing if there is little down there to lose. If the bands below
    # the step are within a few dB of the source's own low-band peak, the "step"
    # is a feature of the instrument rather than the edge of its range.
    peak_low = max(value for _, value in relative)
    if below[1] > peak_low - QUIET_BAND_DB:
        return None

    corner = min(below[0], MAX_HIGHPASS_HZ)
    return corner, (
        f"content starts at {frequency:.0f} Hz, rising {rise:.0f} dB from the "
        f"band below it, so everything under {corner:.0f} Hz is rumble rather "
        "than signal"
    )


def choose_resonance(bands: dict[float, float], reference: float) -> tuple[float, float, str] | None:
    """A band that peaks above its IMMEDIATE NEIGHBOURS, not above the midrange.

    THE DISTINCTION IS THE WHOLE RULE, and getting it wrong produced the worst
    output this planner has made. Scoring a band against the distant 500 Hz to
    4 kHz average, it proposed a 5 dB cut at 100 Hz on a bass guitar, on the
    grounds that 100 Hz sat 41 dB above the midrange. It does. That is what a
    bass is.

    A resonance is LOCALLY peaked — prominent against the bands either side of
    it. A broad elevation is the instrument's range, and cutting it is a
    judgement about what that instrument should sound like rather than a
    correction. The glossary already says this under "boxy": look for a peak
    rather than a broad lift, because a broad lift there is body, not box.

    THIS RULE CANNOT REPRODUCE THE MOVE THAT WON THE LISTENING TEST, and that is
    the honest outcome rather than a reason to loosen it. The 4 dB I took out of
    the acoustic guitar's 250 Hz by hand was a broad plateau — only +2.5 dB
    above its neighbours at its peakiest — so it was a tonal balance decision,
    not a resonance fix. The measurement told me where; knowing what an acoustic
    guitar should sound like told me whether. That second half is instrument
    knowledge, and this planner has none.

    Which is exactly what CLASSIFY and the preset knowledge base are for, and a
    good reason to expect a model with world knowledge to beat this floor.
    """
    ordered = sorted(bands)
    candidates: list[tuple[float, float]] = []
    for index in range(1, len(ordered) - 1):
        frequency = ordered[index]
        if not (100.0 <= frequency <= 800.0):
            continue
        neighbours = 0.5 * (bands[ordered[index - 1]] + bands[ordered[index + 1]])
        candidates.append((frequency, bands[frequency] - neighbours))

    if not candidates:
        return None

    frequency, prominence = max(candidates, key=lambda c: c[1])
    if prominence < RESONANCE_THRESHOLD_DB:
        return None

    # Half the prominence, and never more than 4 dB. A resonance is part of the
    # instrument even when it is too strong, and flattening it entirely removes
    # what makes the source sound like itself.
    gain = -min(prominence * 0.5, 4.0)
    return frequency, round(gain, 1), (
        f"{frequency:.0f} Hz peaks {prominence:+.1f} dB above the bands either "
        f"side of it, which is a resonance rather than the instrument's range"
    )


def choose_compressor(report: dict[str, Any]) -> tuple[dict[str, float], str] | None:
    """Threshold from the measured percentiles, timing from how dynamic it is.

    Peak-to-loudness ratio decides both whether to compress and how. Above
    18 dB the material is unprocessed and wants a slow attack so transients
    survive; below 10 it has already been squashed, and more compression removes
    what is left rather than controlling anything.
    """
    dynamics = report.get("dynamics") or {}
    plr = dynamics.get("peak_to_loudness_ratio_db")
    percentiles = dynamics.get("short_term_rms_percentiles") or {}
    threshold = percentiles.get("p75_dbfs")

    if plr is None or threshold is None:
        return None

    if plr < ALREADY_COMPRESSED_PLR_DB:
        return None

    if plr >= VERY_DYNAMIC_PLR_DB:
        ratio, attack, release = 2.2, 30.0, 180.0
        why = (f"peak-to-loudness is {plr:.1f} dB, so the material is very "
               "dynamic; a slow attack keeps the transients")
    else:
        ratio, attack, release = 2.5, 15.0, 150.0
        why = (f"peak-to-loudness is {plr:.1f} dB, which is moderately dynamic")

    return {
        "threshold_dbfs": round(float(threshold), 1),
        "ratio_to_1": ratio,
        "knee_db": 9.0,
        "attack_ms": attack,
        "release_ms": release,
        # Roughly what the reduction will cost, so the chain does not arrive at
        # the limiter far below where it started. Refine can measure it exactly;
        # this is a planner, and its job is to be close and defensible.
        "makeup_gain_db": round(min(4.0, (plr - 8.0) * 0.25), 1),
    }, why + f", threshold at the 75th percentile of short-term RMS ({threshold:.1f} dBFS)"


def choose_air(report: dict[str, Any]) -> tuple[float, float, str] | None:
    """Brighten only a source that measurement says is dark.

    Three conditions, and the second is the one the codec detector exists for.
    Boosting above a lossy wall does nothing audible at all, so the user hears
    no change and concludes the tool does not work.
    """
    spectrum = report.get("spectrum") or {}
    hygiene = report.get("hygiene") or {}

    rolloff = spectrum.get("hf_rolloff_db_per_octave")
    if rolloff is None or rolloff > DARK_ROLLOFF_DB_PER_OCTAVE:
        return None

    limit = spectrum.get("bandwidth_limit_hz")
    confidence = spectrum.get("bandwidth_limit_confidence") or 0.0
    if limit is not None and confidence >= 0.4:
        return None

    floor = hygiene.get("noise_floor_dbfs")
    floor_confidence = hygiene.get("noise_floor_confidence") or 0.0
    if floor is not None and floor > -55.0 and floor_confidence >= 0.4:
        return None

    # Scaled to how dark it is, and capped: this is a corrective nudge, not a
    # tonal opinion.
    gain = min(3.0, (DARK_ROLLOFF_DB_PER_OCTAVE - rolloff) * 0.12)
    if gain < 1.0:
        return None

    return 7000.0, round(gain, 1), (
        f"the top end rolls off at {rolloff:.1f} dB/octave with no codec wall "
        "and a quiet noise floor, so there is room to open it"
    )


def plan(
    report: dict[str, Any],
    *,
    reverb_mix: float = 0.0,
    output_bit_depth: int | None = 24,
    ceiling_dbtp: float = -1.0,
) -> tuple[EffectGraph, PlanExplanation]:
    """Build a graph from a measurement report.

    `reverb_mix` is opt-in rather than inferred. Adding space nobody asked for
    is a strong opinion, and there is no measurement that says a dry source
    wants to be wet — only a listener.
    """
    explanation = PlanExplanation()
    nodes: list[GraphNode] = []
    master: list[GraphNode] = []

    bands = _bands(report)
    reference = _midrange_reference(bands)
    warnings = {w.get("code") for w in (report.get("warnings") or [])}

    # --- things the report says must be fixed first ------------------------
    if "out_of_phase" in warnings:
        nodes.append(GraphNode(
            id="polarity", type="polarity", params={"inverted": True},
            rationale="the channels are strongly negatively correlated, so the "
                      "mono sum currently cancels",
        ))
        explanation.moves.append(
            "The two channels are out of phase, which makes the mono sum cancel, "
            "so one is inverted before anything else.")

    if "dc_offset" in warnings:
        nodes.append(GraphNode(
            id="dc", type="dc_block", params={"cutoff_hz": 20.0},
            rationale="measurable DC offset, which consumes headroom and biases "
                      "every level decision downstream",
        ))
        explanation.moves.append("A DC offset is removed first.")

    # --- corrective EQ ------------------------------------------------------
    eq_bands: list[dict[str, Any]] = []

    if reference is not None:
        highpass = choose_highpass(bands, reference)
        if highpass:
            corner, why = highpass
            eq_bands.append({"filter_type": "highpass", "center_freq_hz": round(corner),
                             "gain_db": 0.0, "q_factor": 0.707})
            explanation.moves.append(f"High-pass at {corner:.0f} Hz because {why}.")
        else:
            explanation.declined.append(
                "the low end, which has no clear rumble to remove")

        resonance = choose_resonance(bands, reference)
        if resonance:
            frequency, gain, why = resonance
            eq_bands.append({"filter_type": "bell", "center_freq_hz": round(frequency),
                             "gain_db": gain, "q_factor": 1.0})
            explanation.moves.append(
                f"A {gain:+.1f} dB bell at {frequency:.0f} Hz because {why}.")
        else:
            explanation.declined.append(
                "the low mids, where no band peaks above its neighbours enough "
                "to call a resonance — a broad lift there is the instrument's "
                "range, and reshaping it needs to know what the instrument is")

    air = choose_air(report)
    if air:
        frequency, gain, why = air
        eq_bands.append({"filter_type": "highshelf", "center_freq_hz": frequency,
                         "gain_db": gain, "q_factor": 0.707})
        explanation.moves.append(f"A {gain:+.1f} dB shelf at {frequency:.0f} Hz "
                                 f"because {why}.")
    else:
        explanation.declined.append(
            "the top end, which is not dark enough to justify brightening")

    if eq_bands:
        nodes.append(GraphNode(
            id="tone", type="parametric_eq", params={"bands": eq_bands},
            rationale="corrective moves, each derived from this source's own "
                      "band levels rather than from a genre convention",
        ))

    # --- dynamics -----------------------------------------------------------
    compressor = choose_compressor(report)
    if compressor:
        params, why = compressor
        nodes.append(GraphNode(id="level", type="compressor", params=params,
                               rationale=why))
        explanation.moves.append(f"Compression at {params['ratio_to_1']}:1 because {why}.")
    else:
        plr = (report.get("dynamics") or {}).get("peak_to_loudness_ratio_db")
        if plr is not None and plr < ALREADY_COMPRESSED_PLR_DB:
            explanation.declined.append(
                f"the dynamics — at {plr:.1f} dB peak-to-loudness this has already "
                "been compressed, and more would flatten it")
        else:
            explanation.declined.append("the dynamics")

    # --- space, only if asked for ------------------------------------------
    if reverb_mix > 0.0:
        nodes.append(GraphNode(
            id="space", type="plate_reverb", mix=min(reverb_mix, 1.0),
            params={"decay_s": 1.3, "pre_delay_ms": 25.0, "damping_hz": 6000.0},
            rationale="requested; pre-delay keeps the dry source in front of it",
        ))
        explanation.moves.append(
            f"A plate at {reverb_mix:.0%} wet, as requested.")

    # --- master -------------------------------------------------------------
    master.append(GraphNode(
        id="ceiling", type="true_peak_limiter",
        params={"ceiling_dbtp": ceiling_dbtp, "lookahead_ms": 5.0,
                "release_ms": 90.0},
        rationale="true peak is defined at the delivery rate, so the ceiling is "
                  "enforced after the output conversion",
    ))

    true_peak = (report.get("loudness") or {}).get("true_peak_dbtp")
    if true_peak is not None and true_peak > ceiling_dbtp:
        explanation.moves.append(
            f"The source already true-peaks at {true_peak:+.2f} dBTP, which clips "
            f"on conversion, so a limiter holds it at {ceiling_dbtp:.1f} dBTP.")

    if output_bit_depth is not None and 0 < output_bit_depth < 32:
        master.append(GraphNode(
            id="word_length", type="dither",
            params={"bit_depth": float(output_bit_depth), "amount_lsb": 2.0},
            rationale=f"reducing to {output_bit_depth} bits, so the quantisation "
                      "error is decorrelated from the signal",
        ))

    graph = EffectGraph(nodes=nodes, master=master, explanation=explanation.text())
    return graph, explanation
