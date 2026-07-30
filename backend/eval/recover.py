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


def recover_deterministic(deltas: dict[str, float]) -> list[str]:
    """Propose the node types that best explain a set of measurement deltas."""
    detected: set[str] = set()

    bands = _band_range(deltas, 25, 20000)
    if not bands:
        return []

    # A chain's overall level change is not evidence of anything in
    # particular, so the spectrum is examined after removing it. Without this
    # every makeup-gain change reads as a broadband EQ move.
    level = float(np.median(bands))
    shape = {hz: _band(deltas, hz) - level
             for hz in (50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500,
                        630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
                        5000, 6300, 8000, 10000, 12500, 16000)}

    # --- EQ: any band that moved substantially relative to the rest ---------
    if max(abs(v) for v in shape.values()) > 1.5:
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
