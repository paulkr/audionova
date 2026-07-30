"""The descriptor glossary: what engineers say, and where it lives.

Users describe audio in adjectives. "Boxy", "harsh", "thin", "muddy" — these are
precise and shared among engineers, and each corresponds to a fairly narrow
frequency region and a conventional move.

THE GLOSSARY IS EVIDENCE, NOT INSTRUCTION. Every entry carries a `check_first`
condition that has to be true of the measured audio before the default move
applies. That field is the whole point of the file, because the alternative —
"muddy means cut 250 Hz" — is a stock move applied to a source that may not have
a problem at 250 Hz, and a chain built from stock moves sounds like a chain of
stock moves.

So a mud complaint on a source whose 200–400 Hz region is already 6 dB below
its own average is not a mud problem: something else is making it feel that way,
most often a lack of definition higher up. The planner is expected to say so
rather than cut a band that is already quiet.

FREQUENCY RANGES ARE WIDE ON PURPOSE. "Boxy" is not 400 Hz, it is roughly
250–500 Hz and the exact centre depends on the instrument — a boxy kick and a
boxy acoustic guitar are not the same frequency. Narrow ranges would encode a
precision nobody has.

This is a starting set of forty terms, not the sixty the plan calls for. The
missing ones are the genre-specific and instrument-specific vocabulary, which
should be authored alongside the presets by someone who mixes for a living
rather than guessed at here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Axis = Literal["tonal", "dynamic", "spatial", "character"]


@dataclass(frozen=True)
class Descriptor:
    term: str
    axis: Axis
    # Region the term refers to, in Hz. Wide, because the exact centre depends
    # on the instrument.
    low_hz: float
    high_hz: float
    # Sign of the conventional move. Negative means cut, positive means boost.
    # Zero means the term does not map to an EQ move at all.
    direction: int
    # What must be true of the measurement before the move is justified.
    check_first: str
    note: str = ""
    # Terms a user is likely to use interchangeably.
    aliases: tuple[str, ...] = ()


GLOSSARY: tuple[Descriptor, ...] = (
    # --- low end -----------------------------------------------------------
    Descriptor(
        "muddy", "tonal", 150.0, 400.0, -1,
        "the 150-400 Hz bands sit at or above the source's own band average",
        "If that region is already below average the problem is usually a lack "
        "of definition higher up, not excess here. Say so rather than cutting.",
        aliases=("mud", "murky", "cloudy"),
    ),
    Descriptor(
        "boomy", "tonal", 60.0, 150.0, -1,
        "the 60-150 Hz bands are elevated and the crest factor there is low",
        "Low crest in the low bands means it is sustained resonance rather than "
        "note fundamentals, which is what makes it boom rather than support.",
        aliases=("boom", "woofy"),
    ),
    Descriptor(
        "thin", "tonal", 80.0, 250.0, +1,
        "the low bands are below the source's own average AND the noise floor "
        "there is low enough to boost without lifting rumble",
        aliases=("weak", "lightweight", "no_body"),
    ),
    Descriptor(
        "rumbly", "tonal", 20.0, 60.0, -1,
        "there is measurable energy below 60 Hz",
        "Almost always a high-pass rather than a bell. Very little musical "
        "content lives below 40 Hz outside kick, bass and synth.",
        aliases=("rumble", "subby"),
    ),
    Descriptor(
        "fat", "tonal", 80.0, 200.0, +1,
        "the low-mids are not already elevated",
        aliases=("full", "warm_low"),
    ),

    # --- low mids ----------------------------------------------------------
    Descriptor(
        "boxy", "tonal", 250.0, 500.0, -1,
        "a narrow peak exists somewhere in 250-500 Hz relative to neighbours",
        "Boxiness is a resonance, so look for a peak rather than a broad lift. "
        "A broad lift there is body, not box.",
        aliases=("box", "cardboardy"),
    ),
    Descriptor(
        "honky", "tonal", 500.0, 900.0, -1,
        "a peak exists in 500-900 Hz",
        aliases=("honk", "nasal_low"),
    ),
    Descriptor(
        "woody", "tonal", 300.0, 600.0, 0,
        "instrument is acoustic and the region is not already scooped",
        "Desirable on acoustic guitar and double bass; unwanted on vocals. "
        "The sign depends on the instrument, so this term has no default move.",
    ),
    Descriptor(
        "hollow", "tonal", 300.0, 800.0, +1,
        "the 300-800 Hz region is scooped relative to its neighbours",
        "Usually the result of over-cutting the low mids on a previous pass.",
        aliases=("scooped", "phasey_mid"),
    ),

    # --- mids --------------------------------------------------------------
    Descriptor(
        "nasal", "tonal", 900.0, 1600.0, -1,
        "a peak exists in 900-1600 Hz",
        aliases=("nasally", "pinched"),
    ),
    Descriptor(
        "forward", "tonal", 1000.0, 3000.0, +1,
        "the presence region is not already elevated and the source is not "
        "already flagged as harsh",
        aliases=("present", "upfront", "in_your_face"),
    ),
    Descriptor(
        "buried", "tonal", 1000.0, 4000.0, +1,
        "the presence region is below the source's own average",
        "Often better solved by cutting the masking source than by boosting "
        "this one, but that needs a mix rather than a stem. NOT an alias of the "
        "spatial 'distant': a source can lack presence with no reverb at all, "
        "and the two want different answers. Which one applies is decided by "
        "whether a reverb is present, not by the word.",
        aliases=("recessed", "behind"),
    ),
    Descriptor(
        "honest", "tonal", 0.0, 0.0, 0,
        "never — this is a preference, not a measurement",
        "Recorded so the planner recognises it as unactionable rather than "
        "inventing a frequency for it.",
    ),

    # --- upper mids --------------------------------------------------------
    Descriptor(
        "harsh", "tonal", 2000.0, 5000.0, -1,
        "the 2-5 kHz bands are elevated relative to the source's own average",
        "The commonest complaint about anything digital or distorted. Check for "
        "clipping first: harshness from clipping cannot be EQ'd out.",
        aliases=("harshness", "brittle", "aggressive", "abrasive"),
    ),
    Descriptor(
        "edgy", "tonal", 2500.0, 4000.0, -1,
        "a peak exists in 2.5-4 kHz",
        aliases=("edge", "gritty_mid"),
    ),
    Descriptor(
        "bright", "tonal", 4000.0, 12000.0, +1,
        "the high bands are not already elevated and the noise floor is low",
        aliases=("brighter", "open", "crisp"),
    ),
    Descriptor(
        "dull", "tonal", 4000.0, 12000.0, +1,
        "the HF rolloff is steeper than about -12 dB/octave AND there is no "
        "codec wall — above a wall a boost is inaudible",
        "This is the term the bandwidth-limit check exists for. A dull MP3 and "
        "a dull performance need different answers.",
        aliases=("dark", "muffled", "closed", "veiled"),
    ),
    Descriptor(
        "sibilant", "tonal", 5000.0, 9000.0, -1,
        "the source is a vocal and there are short high-band peaks",
        "Wants a dynamic EQ or de-esser rather than a static cut: a static cut "
        "dulls the whole performance to fix a few consonants.",
        aliases=("essy", "spitty", "sss"),
    ),

    # --- top ---------------------------------------------------------------
    Descriptor(
        "airy", "tonal", 10000.0, 18000.0, +1,
        "there is content above 10 kHz to lift AND no codec wall below it",
        "The single most commonly requested inaudible move. Check the "
        "bandwidth limit before spending a band.",
        aliases=("air", "sheen", "sparkle", "open_top"),
    ),
    Descriptor(
        "fizzy", "tonal", 6000.0, 12000.0, -1,
        "the 6-12 kHz bands are elevated",
        "Characteristic of a DI guitar with no cabinet simulation, where the "
        "answer is a cabinet rather than an EQ cut.",
        aliases=("fizz", "buzzy", "raspy"),
    ),
    Descriptor(
        "hissy", "tonal", 4000.0, 20000.0, -1,
        "the measured noise floor is above about -55 dBFS with confidence",
        "A noise problem, not a tonal one. Cutting the top to hide hiss dulls "
        "the source; a gate or expander addresses the cause.",
        aliases=("noisy", "hiss"),
    ),

    # --- dynamics ----------------------------------------------------------
    Descriptor(
        "punchy", "dynamic", 0.0, 0.0, 0,
        "the crest factor is not already low and the peak-to-loudness ratio is "
        "above about 10 dB",
        "Wants a slower attack so transients pass, not more compression. More "
        "compression on already-squashed material removes punch.",
        aliases=("punch", "snappy", "impactful"),
    ),
    Descriptor(
        "squashed", "dynamic", 0.0, 0.0, 0,
        "peak-to-loudness ratio is below about 8 dB",
        "Mastered-record territory. Nothing here can restore what was removed; "
        "say so rather than adding an expander and hoping.",
        aliases=("crushed", "flat", "lifeless", "over_compressed"),
    ),
    Descriptor(
        "uneven", "dynamic", 0.0, 0.0, 0,
        "the short-term RMS percentile spread is wide, p90 minus p10 above "
        "about 12 dB",
        "The one complaint compression genuinely solves. Set the threshold from "
        "the measured percentiles rather than from habit.",
        aliases=("inconsistent", "jumpy", "wandering"),
    ),
    Descriptor(
        "pumping", "dynamic", 0.0, 0.0, 0,
        "an existing compressor has a release comparable to the material's own "
        "modulation rate",
        "Derive the release from the envelope modulation spectrum, not by ear "
        "on a static loop.",
        aliases=("breathing", "pump"),
    ),
    Descriptor(
        "sustained", "dynamic", 0.0, 0.0, 0,
        "the envelope modulation centroid is low",
        "Wants upward compression or a slower release, not a lower threshold.",
        aliases=("long", "singing"),
    ),

    # --- spatial -----------------------------------------------------------
    Descriptor(
        "narrow", "spatial", 0.0, 0.0, 0,
        "the source has two channels AND a side signal to widen — dual mono "
        "cannot be widened, only decorrelated",
        "This is what the no-op width check exists for.",
        aliases=("mono_ish", "collapsed", "small"),
    ),
    Descriptor(
        "wide", "spatial", 0.0, 0.0, 0,
        "the mono-compatibility loss is under about 1 dB, so there is room to "
        "widen without breaking the mono sum",
        aliases=("spacious", "big", "expansive"),
    ),
    Descriptor(
        "dry", "spatial", 0.0, 0.0, 0,
        "no reverb or delay is present, or their mix is low",
        aliases=("close", "intimate", "upfront_space"),
    ),
    Descriptor(
        "washy", "spatial", 0.0, 0.0, 0,
        "reverb decay is long relative to the material's own note rate",
        "Shorten the decay or add pre-delay before reducing the mix — the mix "
        "is usually not the problem.",
        aliases=("swimmy", "cluttered_space", "smeared"),
    ),
    Descriptor(
        "distant", "spatial", 0.0, 0.0, 0,
        "a reverb or delay is present; if none is, the user means the tonal "
        "'buried' instead and the answer is presence rather than space",
        "AMBIGUOUS BY NATURE, and the duplicate-alias guard in this file caught "
        "it claiming to be both this and an alias of 'buried'. Pre-delay is what "
        "separates a source from its space: a short pre-delay glues them "
        "together and reads as distance. But a dry source with no presence reads "
        "as distant too, and cutting reverb there does nothing. Check which "
        "before acting.",
        aliases=("far", "back_of_the_room"),
    ),
    Descriptor(
        "phasey", "spatial", 0.0, 0.0, 0,
        "per-band correlation is low in the low bands, or the mono loss is high",
        "A real problem rather than a taste one. Check polarity before "
        "anything else.",
        aliases=("swirly", "out_of_phase_ish"),
    ),

    # --- character ---------------------------------------------------------
    Descriptor(
        "warm", "character", 100.0, 400.0, +1,
        "the low-mids are not already elevated and the source is not muddy",
        "Warmth is a low-mid lift plus low-order harmonic content. A boost "
        "alone reads as muddy; a little even-order saturation reads as warm.",
        aliases=("warmer", "round", "smooth_low"),
    ),
    Descriptor(
        "crunchy", "character", 0.0, 0.0, 0,
        "the source is not already clipped — saturating clipped material "
        "amplifies existing distortion rather than adding character",
        aliases=("crunch", "dirty", "driven", "gritty"),
    ),
    Descriptor(
        "saturated", "character", 0.0, 0.0, 0,
        "the source is not already clipped",
        aliases=("hot", "cooked", "pushed"),
    ),
    Descriptor(
        "clean", "character", 0.0, 0.0, 0,
        "always — this is a request to remove character, not add it",
        "Reduce or remove drive; do not add a compensating EQ.",
        aliases=("pristine", "transparent", "hifi"),
    ),
    Descriptor(
        "vintage", "character", 0.0, 0.0, 0,
        "requires a preset with citations; there is no generic vintage move",
        "Routed to the preset knowledge base rather than answered directly. "
        "A bandwidth limit, a tilt and some saturation is a caricature.",
        aliases=("retro", "old_school", "analog"),
    ),
    Descriptor(
        "modern", "character", 0.0, 0.0, 0,
        "requires a preset; too broad to act on alone",
        aliases=("contemporary", "polished"),
    ),
    Descriptor(
        "lofi", "character", 0.0, 0.0, 0,
        "requires a preset; the move is deliberate degradation and must be "
        "explicit rather than inferred",
        aliases=("lo_fi", "crushed_vintage"),
    ),
)


# Alias to canonical term. Built once; a duplicate alias is a programming error
# rather than something to resolve at lookup time, so it raises here.
def _build_index() -> dict[str, Descriptor]:
    index: dict[str, Descriptor] = {}
    for entry in GLOSSARY:
        for name in (entry.term, *entry.aliases):
            if name in index:
                raise ValueError(
                    f"glossary alias {name!r} is claimed by both "
                    f"{index[name].term!r} and {entry.term!r}"
                )
            index[name] = entry
    return index


INDEX: dict[str, Descriptor] = _build_index()


def lookup(term: str) -> Descriptor | None:
    """Resolve a term or alias. Case- and separator-insensitive."""
    key = term.strip().lower().replace(" ", "_").replace("-", "_")
    return INDEX.get(key)


def terms_by_axis(axis: Axis) -> list[Descriptor]:
    return [d for d in GLOSSARY if d.axis == axis]


def as_prompt_table() -> str:
    """The glossary as it appears in the cached prompt prefix.

    One line per term, because the model reads this as reference rather than
    prose and a table costs fewer tokens than sentences. `check_first` is
    included for every entry — it is the field that stops the glossary becoming
    a list of stock moves.
    """
    lines = ["term | axis | region | move | check first"]
    for entry in GLOSSARY:
        if entry.low_hz or entry.high_hz:
            region = f"{entry.low_hz:.0f}-{entry.high_hz:.0f} Hz"
        else:
            region = "not tonal"
        move = {1: "boost", -1: "cut", 0: "no default"}[entry.direction]
        lines.append(f"{entry.term} | {entry.axis} | {region} | {move} | "
                     f"{entry.check_first}")
    return "\n".join(lines)
