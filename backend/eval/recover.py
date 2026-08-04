"""A rule-based baseline for recovering a chain from measurement deltas.

THIS IS NOT MEANT TO BE GOOD. It exists so that a future planner's score means
something. A model that recovers 60% of nodes is either impressive or
embarrassing depending entirely on what a handful of thresholds achieves on the
same data, and without that number there is no way to distinguish a planner
that reasons from one that has learned which nodes are common.

The plan's top risks include proxy-metric capture — the team optimising
whichever number moves easiest — and an unanchored score is the easiest number
of all to move.

EVERY RULE HERE COMES FROM THE SENSITIVITY TABLE, not from intuition. The
thresholds are the fields that were shown to track their parameter at r near 1
across six real stems, and nothing else is consulted. Where the table found no
diagnostic field, this recogniser correctly cannot detect the node — which is
the honest outcome and is exactly what it should report.

ORDER IS GUESSED FROM CONVENTION, and this is the baseline's weakest part by
design. Measurement deltas carry very little ordering information: a shelf
before a compressor and a shelf after it produce similar spectra, differing
mainly in how the compressor's gain reduction tracked the boosted band. The
baseline emits the conventional engineering order and nothing more, so any
planner that reasons about interaction should beat it. If none can, the
ordering metric is measuring nothing and should be retired rather than chased.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# The conventional order an engineer would place these in, used only to
# sequence whatever was detected. Corrective filtering first, then level, then
# character, then space.
CONVENTIONAL_ORDER = (
    "parametric_eq",
    "compressor",
    "drive",
    "stereo_width",
    "delay",
    "plate_reverb",
)

# How far a band must sit from the rest of the spectrum before it counts as an
# EQ move. Named rather than inline because the parameter recogniser below
# reads the same shape and must use the same bar — two thresholds for one
# question is how a node gets detected and then described with a band the
# detector never saw.
EQ_SHAPE_THRESHOLD_DB = 1.5

# The bands the shape is computed over. 25 Hz and 20 kHz are excluded: both sit
# at the edge of the analysis and move for reasons that have nothing to do with
# an EQ move — the sample-rate conversion's own passband edge is at 20 kHz.
SHAPE_BANDS = (50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
               1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
               10000, 12500, 16000)


def _band(deltas: dict[str, float], hz: int) -> float:
    return deltas.get(f"spectrum.third_octave_db.{hz}hz.level_db", 0.0)


def _band_range(deltas: dict[str, float], lo: int, hi: int) -> list[float]:
    out = []
    for key, value in deltas.items():
        if not key.startswith("spectrum.third_octave_db."):
            continue
        try:
            hz = int(key.split(".")[2].removesuffix("hz"))
        except (IndexError, ValueError):
            continue
        if lo <= hz <= hi:
            out.append(value)
    return out


def spectral_shape(deltas: dict[str, float]) -> dict[int, float] | None:
    """Band deltas with the chain's overall level change removed.

    A chain's overall level change is not evidence of anything in particular,
    so the spectrum is examined after removing it. Without this every
    makeup-gain change reads as a broadband EQ move.

    Extracted so the type recogniser and the parameter recogniser read exactly
    the same numbers. Two independent readings of the same spectrum is how a
    node gets detected on one band and then described with another.
    """
    bands = _band_range(deltas, 25, 20000)
    if not bands:
        return None
    level = float(np.median(bands))
    return {hz: _band(deltas, hz) - level for hz in SHAPE_BANDS}


def recover_deterministic(deltas: dict[str, float]) -> list[str]:
    """Propose the node types that best explain a set of measurement deltas."""
    detected: set[str] = set()

    shape = spectral_shape(deltas)
    if shape is None:
        return []

    # --- EQ: any band that moved substantially relative to the rest ---------
    if max(abs(v) for v in shape.values()) > EQ_SHAPE_THRESHOLD_DB:
        detected.add("parametric_eq")

    # --- Drive: harmonics added well above where the source had content -----
    #
    # Distinguished from a high shelf by WHERE it lands. Saturation lifts the
    # top progressively and heavily — 12 dB of drive moved the 8 kHz band by
    # +33 dB on a bass — while also flattening peaks. A shelf raises the top
    # by its own gain and leaves dynamics alone.
    top = float(np.mean([shape[hz] for hz in (8000, 10000, 12500, 16000)]))
    crest = deltas.get("dynamics.crest_factor_db", 0.0)
    plr = deltas.get("dynamics.peak_to_loudness_ratio_db", 0.0)
    if top > 6.0 and crest < -1.0:
        detected.add("drive")

    # --- Compression: the envelope speeds up --------------------------------
    #
    # Crest factor is deliberately NOT the trigger. It passed on synthetic
    # material and failed on all six real stems: peak is a single sample and a
    # 10 ms attack passes the initial transient, so crest rises on percussive
    # sources and falls on sustained ones. The envelope modulation spectrum
    # tracked ratio at r = 0.94 on every stem.
    modulation = deltas.get("dynamics.envelope_modulation.centroid_hz", 0.0)
    lra = deltas.get("loudness.loudness_range_lu", 0.0)
    if modulation > 0.3 or (lra < -0.8 and plr < -0.5):
        detected.add("compressor")

    # --- Reverb and delay ---------------------------------------------------
    #
    # Both add energy between the notes, which raises the quiet percentiles far
    # more than the loud ones. They are separated by how far the effect
    # reaches: a plate fills continuously and lifts the noise floor, while a
    # discrete delay leaves gaps between repeats.
    quiet = deltas.get("dynamics.short_term_rms_percentiles.p10_dbfs", 0.0)
    loud = deltas.get("dynamics.short_term_rms_percentiles.p90_dbfs", 0.0)
    floor = deltas.get("hygiene.noise_floor_dbfs", 0.0)
    fill = quiet - loud

    # DETECTED INDEPENDENTLY, not as an either/or. An earlier version chose
    # between them with an if/else, which made a chain containing both
    # impossible to recover: the space_only template scored 0.12 because the
    # recogniser could physically only name one of its two nodes.
    if floor > 2.0:
        detected.add("plate_reverb")
    if fill > 1.0:
        detected.add("delay")

    # A LIMITATION WORTH STATING RATHER THAN PAPERING OVER: on continuously
    # dense material neither fires. A bass or a busy drum loop never leaves
    # gaps, so added space has nowhere to show up in the level percentiles —
    # measured on the bass stem, a delay and a plate together moved the
    # p10-to-p90 fill by -0.42 dB, which is to say not at all. The report may
    # simply not carry this information for dense sources; a planner that
    # recovers it would be demonstrating something real.

    # --- Stereo width -------------------------------------------------------
    correlation = deltas.get("stereo.correlation", 0.0)
    if abs(correlation) > 0.02:
        detected.add("stereo_width")

    return [node for node in CONVENTIONAL_ORDER if node in detected]


# ---------------------------------------------------------------------------
# Parameter recovery
#
# RECOVERING "THERE WAS AN EQ" IS A MUCH WEAKER CLAIM than recovering "there was
# a -3 dB bell near 900 Hz", and the type score cannot tell the two apart. This
# reads the same shape the detector reads and says where and how much.
#
# It is a baseline in exactly the sense the type recogniser is: a run of
# contiguous same-sign bands, an energy-weighted centroid, and the peak. It has
# no model of a filter's skirt, so a broad shelf is reported at the middle of
# what it lifted rather than at its corner, and that error is real and is
# reported rather than corrected for. A planner that fits an actual filter
# response should beat it — `backend/refine/match_eq.py` already does exactly
# that, against the rendered audio rather than the report.


def _runs(shape: dict[int, float], threshold: float) -> list[list[int]]:
    """Contiguous same-sign stretches containing at least one band over the bar.

    SIGN MATTERS AND ADJACENCY MATTERS. A boost at 200 Hz and a cut at 5 kHz are
    two moves, and merging them by absolute value would report one band halfway
    between at a gain neither of them has.
    """
    ordered = sorted(shape)
    runs: list[list[int]] = []
    current: list[int] = []
    current_sign = 0

    for hz in ordered:
        sign = 1 if shape[hz] > 0 else (-1 if shape[hz] < 0 else 0)
        if sign != 0 and sign == current_sign:
            current.append(hz)
        else:
            if current:
                runs.append(current)
            current = [hz] if sign != 0 else []
            current_sign = sign
    if current:
        runs.append(current)

    return [r for r in runs if max(abs(shape[hz]) for hz in r) > threshold]


@dataclass
class RecoveredBand:
    center_freq_hz: float
    gain_db: float


def recover_eq_bands(deltas: dict[str, float], *, limit: int = 3) -> list[RecoveredBand]:
    """Where the EQ moves were and how big, read off the band deltas.

    Returns at most `limit` bands, largest first. The limit exists because the
    skirt of one bell spreads over several third-octave bands and the report
    cannot resolve more moves than it has independent evidence for; proposing
    one band per band that moved would be describing the analysis rather than
    the chain.
    """
    shape = spectral_shape(deltas)
    if shape is None:
        return []

    bands: list[RecoveredBand] = []
    for run in _runs(shape, EQ_SHAPE_THRESHOLD_DB):
        weights = [abs(shape[hz]) for hz in run]
        total = sum(weights)
        if total <= 0.0:
            continue
        # CENTROID IN LOG FREQUENCY, because the bands are logarithmically
        # spaced and a linear centroid over them lands high by most of an
        # octave — an arithmetic mean of 1000 and 4000 is 2500, and the band
        # halfway between them is 2000.
        centre = math.exp(sum(w * math.log(hz) for hz, w in zip(run, weights)) / total)
        peak = max(run, key=lambda hz: abs(shape[hz]))
        bands.append(RecoveredBand(center_freq_hz=centre, gain_db=shape[peak]))

    bands.sort(key=lambda b: -abs(b.gain_db))
    return bands[:limit]


# ---------------------------------------------------------------------------
# The seam a model planner drops into
#
# THE HARNESS MUST NOT KNOW WHICH RECOGNISER IT IS SCORING. A model planner
# arrives as another entry in `RECOGNISERS` and nothing above this line changes.
# That matters for a reason beyond tidiness: a harness with a model-shaped hole
# in it tends to grow model-shaped conveniences, and then the floor and the
# model are no longer being measured by the same instrument.
#
# Nothing here constructs a client or reads an API key. `run_recovery` resolves
# a name from this table and no other way, so no argument to the runner can
# cause a network call.


@dataclass
class Prediction:
    """What a recogniser returns: the chain it thinks was applied."""

    node_types: list[str] = field(default_factory=list)
    eq_bands: list[RecoveredBand] = field(default_factory=list)


def rule_based(deltas: dict[str, float], **_: object) -> Prediction:
    """The deterministic floor. Thresholds only, straight from the sweep."""
    return Prediction(node_types=recover_deterministic(deltas),
                      eq_bands=recover_eq_bands(deltas))


RECOGNISERS = {"rules": rule_based}


class UnknownRecogniser(KeyError):
    """Named a recogniser that does not exist.

    A hard error rather than a fall back to the rules, because the caller is a
    script whose whole output is a number: silently scoring a different
    recogniser than the one asked for produces a floor attributed to the wrong
    thing, which is worse than no floor.
    """


def resolve(name: str):
    if name not in RECOGNISERS:
        raise UnknownRecogniser(
            f"{name!r} is not a registered recogniser; known: "
            f"{', '.join(sorted(RECOGNISERS))}")
    return RECOGNISERS[name]
