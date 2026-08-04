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

CHECK_FIRST NAMES A MEASUREMENT THAT EXISTS. Every condition added after the
first forty terms is phrased against a field the measurement report actually
produces, and `test_planner` enforces that against a real report rather than
against a list of field names someone typed. A condition like "reverb decay is
long" reads as rigour and is not: there is no blind reverb time in the report,
so nothing can evaluate it. The terms that predate this rule are listed in the
test as exemptions rather than quietly excluded, because the exemption list is
the honest measure of how much of this file is still prose.

Sixty terms. The genre-specific and artist-specific vocabulary is still absent
and still belongs with the presets, authored by someone who mixes for a living
rather than guessed at here — 'Motown bass' names a record, not a region.
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
    Descriptor(
        "bassy", "tonal", 40.0, 120.0, +1,
        "the 40-120 Hz bands sit at or below the source's own band average AND "
        "there is measurable energy there to lift — below a source's own "
        "fundamental a boost raises the noise floor and nothing else",
        "'Not enough bass' is about 80-120 Hz on anything that is not a kick, a "
        "bass or a synth. Boosting 40 Hz on a guitar adds weight nobody asked "
        "for to a region the instrument does not occupy.",
        aliases=("bassier", "bass_heavy", "more_bass"),
    ),
    Descriptor(
        "plosive", "tonal", 20.0, 120.0, -1,
        "the per-ERB crest in the lowest bands is far ABOVE the broadband crest "
        "factor — plosives are short bursts, not the sustained energy that makes "
        "a source rumbly",
        "The same high-pass answers this and 'rumbly', but for opposite reasons, "
        "and the crest is what tells them apart. Worth checking on any voice "
        "even when the user did not complain: it costs nothing to fix and it "
        "biases every level decision downstream while it is there.",
        aliases=("plosives", "popping", "p_pops"),
    ),
    Descriptor(
        "chesty", "tonal", 150.0, 350.0, -1,
        "the 150-350 Hz bands sit above the source's own band average AND the "
        "per-ERB crest there is below the broadband crest factor, i.e. it is "
        "sustained resonance rather than note attacks",
        "Vocal-specific and narrower than 'muddy'. Chest sits just above the "
        "fundamental of most voices, so a cut placed by ear at 250 Hz often "
        "lands on the note being held. Place it from the band levels.",
        aliases=("chest", "chestiness", "proximity"),
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
        aliases=("present", "presence", "upfront", "in_your_face"),
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
        "midrangey", "tonal", 400.0, 2000.0, -1,
        "the 400-2000 Hz bands sit above the source's own band average as a "
        "GROUP rather than at one peak — a single peak is honky or nasal and "
        "wants a narrower fix",
        "Broad and narrow excesses in the same region are different problems. "
        "The broad one is usually the instrument's own range being too loud, "
        "which a fader answers better than an EQ.",
        aliases=("middy", "mid_heavy", "too_much_midrange"),
    ),
    Descriptor(
        "ringy", "tonal", 100.0, 8000.0, -1,
        "a narrow peak exists in the band levels AND the per-ERB crest at that "
        "band is below the broadband crest factor — a ring outlasts the note "
        "that excited it, so it shows up as sustain, not as attack",
        "The region is wide because a ring can be anywhere: a snare shell at "
        "400 Hz and a cymbal at 6 kHz are both ringy. The measurement locates "
        "it; the word does not, and a centre guessed from the word is a "
        "resonance left in place plus a hole somewhere else.",
        aliases=("ringing", "resonant", "rings_out"),
    ),
    Descriptor(
        "tinny", "tonal", 200.0, 4000.0, 0,
        "two things at once — the bands under 250 Hz sit beneath the source's "
        "own band average AND a peak exists in 1-4 kHz; which of the two is "
        "measurably true decides the move, so this term has no default one",
        "Tinny is the sound of a small speaker: no bottom and a midrange "
        "resonance. Fixing only the peak leaves it thin and fixing only the "
        "bottom leaves it honky. If just one of the two holds, the answer is "
        "that term instead.",
        aliases=("tin", "cheap_sounding", "small_speaker"),
    ),
    Descriptor(
        "clear", "tonal", 150.0, 6000.0, 0,
        "which end is at fault: if the 150-400 Hz bands sit above the source's "
        "own band average the answer is a low-mid cut, and if the 2-5 kHz bands "
        "sit beneath it the answer is a presence lift",
        "Clarity names an outcome, not a region, and the two moves that produce "
        "it are opposite. Doing both because the word is vague is how a source "
        "ends up hollow. Pick the one the band levels support, and if neither "
        "holds, the source is already clear and something else is masking it.",
        aliases=("clarity", "defined", "articulate", "intelligible"),
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
        "shrill", "tonal", 3000.0, 8000.0, -1,
        "a peak exists in the 3-8 kHz bands AND the per-ERB crest there is not "
        "far above the broadband crest factor — if it is, the peaks are short "
        "and the term is 'sibilant', which wants a de-esser instead",
        "The third of three neighbouring complaints. Harsh is broad and lower, "
        "shrill is a peak higher up, sibilant is short bursts. They are told "
        "apart by measurement, not by how strongly the user said it.",
        aliases=("piercing", "screechy", "ear_splitting"),
    ),
    Descriptor(
        "smooth", "tonal", 2000.0, 6000.0, -1,
        "a peak exists in 2-6 kHz to take down; if the region is flat and the "
        "per-ERB crest there is high, the roughness is transient rather than "
        "tonal and an EQ will not reach it",
        "The request form of the 'harsh' complaint. A broad shelf here reads as "
        "dull rather than smooth — the ear hears the peak, not the average, so "
        "removing the average removes the wrong thing.",
        aliases=("smoother", "silky", "less_harsh"),
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
    Descriptor(
        "lispy", "tonal", 5000.0, 9000.0, +1,
        "the 5-9 kHz bands sit beneath the source's own band average — usually "
        "the mark of a de-esser already working too hard rather than of the "
        "recording",
        "The mirror of 'sibilant', and the reason a static cut is the wrong "
        "tool for sibilance: it takes the whole band down, consonants and air "
        "together, and this is what that sounds like afterwards.",
        aliases=("lisping", "over_de_essed", "dull_esses"),
    ),
    Descriptor(
        "breathy", "tonal", 6000.0, 12000.0, +1,
        "the noise floor is low enough that lifting breath does not lift the "
        "room with it, AND there is no codec wall below 12 kHz",
        "Breath and hiss share an octave, so this boost is only available on a "
        "quiet recording. On a noisy one the honest answer is that an EQ cannot "
        "separate them.",
        aliases=("breath", "breathiness", "intimate_vocal"),
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
    Descriptor(
        "peaky", "dynamic", 0.0, 0.0, 0,
        "the crest factor is high WHILE the short-term RMS percentile spread, "
        "p90 minus p10, is narrow — isolated peaks over a steady body",
        "The opposite pair to 'uneven', which is a wide percentile spread at an "
        "ordinary crest. Peaky wants a limiter or a clipper on the peaks; "
        "uneven wants a compressor across the body. Reading the crest alone "
        "confuses the two and gets a fast compressor on both.",
        aliases=("peaks", "spiky", "transient_heavy"),
    ),
    Descriptor(
        "tight", "dynamic", 40.0, 250.0, 0,
        "the per-ERB crest in the 40-250 Hz bands is well below the broadband "
        "crest factor, which is what a low end that rings rather than stops "
        "looks like",
        "Not an EQ move, which is why the direction is zero: the complaint word "
        "is 'flabby', the request word is 'tight', and both name the same "
        "measurement. The answer is faster envelope control in the low bands. A "
        "cut makes it smaller, not tighter, and the user will say so.",
        aliases=("tighter", "flabby", "loose", "undamped"),
    ),
    Descriptor(
        "clicky", "dynamic", 1500.0, 6000.0, -1,
        "the per-ERB crest in the 1.5-6 kHz bands is well above the broadband "
        "crest factor — the click is a transient, so a static cut also removes "
        "the sustain that was not the problem",
        "Beater, pick and key noise all live here. A transient shaper or a "
        "dynamic EQ keeps the body; a bell does not.",
        aliases=("click", "clacky", "beater_heavy"),
    ),
    Descriptor(
        "glued", "dynamic", 0.0, 0.0, 0,
        "there is something to glue: the peak-to-loudness ratio is above about "
        "10 dB and the loudness range is above about 4 LU",
        "Glue is slow, low-ratio bus compression, and on material that has "
        "already been levelled it does nothing audible except cost headroom. "
        "Say so rather than adding a node that measures as a no-op.",
        aliases=("glue", "cohesive", "gel"),
    ),
    Descriptor(
        "loud", "dynamic", 0.0, 0.0, 0,
        "there is headroom to use: the true peak sits below the ceiling and the "
        "peak-to-loudness ratio is above about 10 dB, so level can come up "
        "without a limiter doing the work",
        "Below that, loudness is bought with dynamics, and the honest answer is "
        "what it costs rather than a louder file with no comment. Note also "
        "that the integrated loudness here is the excerpt's, not the file's — "
        "an excerpt can sit well below the loudest moment, and a chain that "
        "clears the ceiling on one can clip on the other.",
        aliases=("louder", "turn_it_up", "more_level"),
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
        "wet", "spatial", 0.0, 0.0, 0,
        "the envelope modulation depth is deep enough that the gaps between "
        "notes are still open — a shallow depth means the source is already "
        "sitting in a space and a second one stacks on it",
        "The antonym of 'dry', and the weakest evidence in this file: the "
        "report carries no blind reverb time, so modulation depth is all there "
        "is and it is indirect. A request for 'wetter' on a source that already "
        "measures shallow is worth questioning rather than executing.",
        aliases=("wetter", "reverby", "more_space"),
    ),
    Descriptor(
        "roomy", "spatial", 0.0, 0.0, 0,
        "the envelope modulation depth is shallow — the gaps between notes are "
        "filled — and no reverb node in the chain accounts for it, so the space "
        "is in the recording",
        "A room in the recording cannot be removed, only masked, and gating "
        "trades it for a different artefact. One of the honest 'no' answers, "
        "and much better given before rendering than after.",
        aliases=("room_sound", "live_sounding", "ambient"),
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
        "distorted", "character", 0.0, 0.0, 0,
        "whether the clipping measurement shows flat-topped runs: if it does, "
        "the distortion is in the file and nothing downstream removes it; if it "
        "does not, the user means the drive already in the chain and the answer "
        "is to reduce it",
        "The same sound as 'crunchy', unwanted, and the measurement is the only "
        "thing that separates them. Do not reach for a repair on the strength "
        "of the word alone — on an unclipped source there is nothing to repair.",
        aliases=("distortion", "broken_up", "clipped", "fuzzed"),
    ),
    Descriptor(
        "crackly", "character", 0.0, 0.0, 0,
        "the hygiene measurements report non-finite samples, or clipping runs; "
        "if neither is present the user means 'fizzy' or 'gritty' and the "
        "answer is tonal rather than a repair",
        "A defect, like hiss, not a taste. Nothing in the chain repairs a "
        "damaged file, and processing one carries the damage into every "
        "measurement downstream.",
        aliases=("crackle", "crackling", "glitchy"),
    ),
    Descriptor(
        "sterile", "character", 0.0, 0.0, 0,
        "the source measures clean — no clipping, and a crest factor that has "
        "not already been squashed — so the complaint is a lack of harmonic "
        "content rather than a tonal one",
        "The complaint form of 'clean'. A low-mid boost is the wrong answer and "
        "reads as muddy; low-order saturation adds the harmonics the word is "
        "actually about. If the source already clips, this is not sterility and "
        "adding drive makes it worse.",
        aliases=("cold", "clinical", "characterless", "antiseptic"),
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
