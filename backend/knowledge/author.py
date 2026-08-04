"""The preset authoring tool.

WHY THIS IS A FIRST-CLASS INTERNAL PRODUCT rather than a script. The plan's
risk #7 is that preset authoring is the schedule bottleneck and is not an
engineering task: it needs the books, and a person who can check a claim about
a session. Everything an engineer CAN do to shorten that four-to-six person-week
estimate is tooling — so this is the tooling, and its whole design goal is that
the author never has to read `schema.py` to use it correctly.

Four commands, each answering a question the author would otherwise answer by
hand-writing JSON and hoping:

    new        scaffold a dossier. Every prompt states the rule it enforces, so
               the schema is learned by using it. A claim and its citation are
               asked for in the same breath, because the schema refuses them
               separately and discovering that later is the annoying way.

    check      validate, and say what is wrong in a form an author can act on.
               Pydantic's errors are precise and unreadable; these are
               translated, and every one names the offending claim and the ways
               out.

    audition   build an effect graph from `parameter_priors`, render it with the
               real engine, and write a loudness-matched A/B pair. AN AUTHOR WHO
               CANNOT HEAR THEIR OWN PRIORS IS GUESSING.

    diff       render two dossiers against one source and report what measurably
               differs. Two presets that measure the same are one preset, and
               that is worth knowing before authoring the twelfth.

THE CATALOGUE IS EMPTY ON PURPOSE AND THIS TOOL DOES NOT FILL IT. It writes
whatever the author types and refuses to invent a claim, a citation, or a
number. There is no "suggest a citation" and there never should be.

---

WHAT `audition` IS AND IS NOT. A preset is not an effect graph, and the planner
composes priors against a measured source rather than applying them. So
auditioning has to do the same thing or it would be auditioning something the
product will never render: the source is MEASURED, each target's move is the
DELTA between what the source already measures and what the prior asks for, and
the render is the delta applied.

That is not the planner. It is a deterministic, documented, single-move
interpretation of the priors with no model in it, and its output is labelled as
such. What it is good for is the question the author actually has — "does -1.5
dB/octave of tilt sound like the record I am thinking of?" — and it answers that
by putting the sound in their ears with the level difference removed.

The A side is a NULL RENDER, never the raw source. Anything else measures the
engine's conversion path along with the preset.

Every move prints what was asked, what the source measured, and what the render
achieved. The shelf pair that implements a tilt does NOT deliver its nominal
slope, and — measured — it does not even miss in a consistent direction: on
pink noise it overshoots by about 38%, on an acoustic guitar stem it
under-delivers by about 25%, because the fitted slope depends on the spectrum
being fitted. That is not a defect to be calibrated away with a fudge factor;
it is the reason the achieved figure is printed instead of the requested one.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel, ValidationError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from backend.audio import wavio  # noqa: E402
from backend.engine import render as render_mod  # noqa: E402
from backend.knowledge import loader  # noqa: E402
from backend.knowledge.schema import (  # noqa: E402
    AvoidBand, Citation, Confidence, Evidence, ParameterPriors, Preset,
    ReviewStatus, TargetRange, as_dossier, name_filter_hits,
)
from backend.listen import match  # noqa: E402
from backend.measure import report as report_mod  # noqa: E402
from backend.schemas.graph import (  # noqa: E402
    MASTER_ONLY, PARAM_MODELS, EffectGraph,
)


# ---------------------------------------------------------------------------
# Problems
#
# One shape for everything this tool complains about, whether it came from
# Pydantic or from a rule Pydantic cannot express. An author should not have to
# tell the two apart, and the fix is part of the complaint rather than something
# to be inferred from it.


@dataclass(frozen=True)
class Problem:
    where: str
    what: str
    fix: str
    severity: str = "error"     # "error" blocks; "warning" does not

    def text(self) -> str:
        head = "ERROR" if self.severity == "error" else "warn "
        return f"  {head}  {self.where}\n         {self.what}\n         fix: {self.fix}"


def errors(problems: Sequence[Problem]) -> list[Problem]:
    return [p for p in problems if p.severity == "error"]


def print_problems(problems: Sequence[Problem], where: str) -> None:
    if not problems:
        print(f"{where}: ok")
        return
    print(f"{where}:")
    for problem in problems:
        print(problem.text())
    counts = (len(errors(problems)), len(problems) - len(errors(problems)))
    print(f"  {counts[0]} error(s), {counts[1]} warning(s)")


# ---------------------------------------------------------------------------
# Translating Pydantic
#
# The messages Pydantic produces are exactly right and read like a stack trace.
# `evidence.0.citation` / `Input should be a valid dictionary` is true and does
# not tell an author that the claim they just typed needs a page number.


_MODEL_FIELDS: dict[type[BaseModel], tuple[str, ...]] = {
    model: tuple(model.model_fields)
    for model in (Preset, Evidence, Citation, TargetRange, AvoidBand,
                  ParameterPriors)
}


def _model_at(loc: tuple[Any, ...]) -> type[BaseModel] | None:
    """Which model a Pydantic error location lands in.

    Used only to suggest a near-miss field name for a typo, so an unknown
    location is a `None` rather than an error of its own.
    """
    parts = tuple(p for p in loc if isinstance(p, str))
    if not parts:
        return Preset
    if parts[0] == "evidence":
        if "citation" in parts:
            return Citation
        return Evidence
    if parts[0] == "parameter_priors":
        if "targets" in parts:
            return TargetRange
        if "avoid_bands" in parts:
            return AvoidBand
        return ParameterPriors
    return Preset


def _where(loc: tuple[Any, ...], data: Any) -> str:
    """A location an author can find in their file, with the claim spelled out.

    `evidence.2.citation.page` is findable. `evidence[2] (claim "Vox AC30
    combo amplifier") -> citation.page` is findable WITHOUT counting array
    entries, which is the part that matters when a dossier has forty of them.
    """
    text = ""
    rest = list(loc)
    if rest and rest[0] == "evidence" and len(rest) > 1 and isinstance(rest[1], int):
        index = rest[1]
        claim = _claim_at(data, index)
        text = f"evidence[{index}]"
        if claim:
            text += f' (claim {claim!r})'
        rest = rest[2:]
    if rest:
        joined = ".".join(str(part) for part in rest)
        text = f"{text} -> {joined}" if text else joined
    return text or "(whole dossier)"


def _claim_at(data: Any, index: int) -> str:
    if not isinstance(data, dict):
        return ""
    evidence = data.get("evidence")
    if not isinstance(evidence, list) or index >= len(evidence):
        return ""
    entry = evidence[index]
    if isinstance(entry, dict) and isinstance(entry.get("claim"), str):
        return _elide(entry["claim"])
    return ""


def _elide(text: str, limit: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# The escape hatch, written once. It is the answer to more than one kind of
# citation complaint and an author reading two different phrasings of it would
# reasonably wonder whether they are two different rules.
_TWO_WAYS_OUT = (
    "either (1) add a citation — citation.source naming author and title, plus "
    "citation.page, because a citation to a whole book is not a citation; or "
    "(2) set confidence to 'unverified' or 'disputed' AND write a review_note "
    "saying why it is doubtful. There is no third way, and that is deliberate: "
    "a claim with neither reads as established fact to everyone downstream."
)


def _translate(entry: dict[str, Any], data: Any) -> Problem:
    kind = entry.get("type", "")
    loc = tuple(entry.get("loc", ()))
    where = _where(loc, data)
    message = str(entry.get("msg", "")).replace("Value error, ", "")
    field_name = str(loc[-1]) if loc else ""

    if kind == "value_error":
        # The two model validators in the schema. Their own messages are good;
        # what they lack is the menu of ways out, stated the same way every
        # time so that an author learns it once.
        if "no citation" in message:
            return Problem(where, message, _TWO_WAYS_OUT)
        if "no review_note" in message:
            return Problem(
                where, message,
                "write review_note saying what is doubtful and why — the next "
                "author reads it before deciding whether to promote the claim. "
                "Or cite it and raise the confidence.")
        return Problem(where, message, "see schema.py for the rule this breaks")

    if kind == "extra_forbidden":
        model = _model_at(loc)
        known = _MODEL_FIELDS.get(model, ()) if model else ()
        near = difflib.get_close_matches(field_name, known, n=1, cutoff=0.6)
        suggestion = f"did you mean {near[0]!r}? " if near else ""
        return Problem(
            where, f"unknown field {field_name!r}",
            f"{suggestion}remove it. The schema forbids extra fields so a typo "
            f"cannot be silently ignored on its way to nowhere. Known fields "
            f"here: {', '.join(known) if known else 'see schema.py'}")

    if kind == "missing":
        return Problem(where, f"{field_name!r} is required and absent",
                       f"add {field_name!r}")

    if kind == "string_pattern_mismatch":
        if field_name == "preset_id" or "preset_id" in where:
            return Problem(
                where, "preset_id must be 4-60 characters of lowercase letters, "
                       "digits and underscores",
                "this id is the only thing that crosses into the effect graph "
                "and it is a closed enum, so it must not contain an artist's "
                "name — describe the era and the instrument instead, e.g. "
                "'era_uk_1965_beatgroup_guitar'")
        return Problem(where, message, "see the pattern in schema.py")

    if kind in ("string_too_short", "too_short"):
        return Problem(where, message,
                       "write more. The minimum exists because a one-word "
                       "answer here is not usable by anyone reading the file "
                       "later")

    if kind in ("string_too_long", "too_long"):
        return Problem(where, message, "shorten it, or split it into two entries")

    if kind in ("greater_than", "greater_than_equal", "less_than",
                "less_than_equal"):
        if field_name == "tolerance":
            return Problem(
                where, "tolerance must be greater than zero",
                "a prior with no tolerance is a specification, and the planner "
                "is meant to compose priors against a source rather than hit "
                "them. Give the range you would actually accept")
        if field_name == "page":
            return Problem(where, "page numbers start at 1",
                           "give the page the claim is on, or omit `page` "
                           "entirely if the source genuinely has none")
        return Problem(where, message, "bring the value inside the range")

    if kind == "enum":
        return Problem(where, message,
                       "unknown values are a hard error rather than a silent "
                       "fallback, because the caller downstream is a language "
                       "model and a quietly-defaulted field is invisible to it")

    return Problem(where, message or kind, "see schema.py")


# ---------------------------------------------------------------------------
# The target vocabulary
#
# `ParameterPriors.targets` is `dict[str, TargetRange]` — free-form keys, no
# vocabulary anywhere. Nothing in the repository consumes it yet, so this file
# is the first consumer and therefore the thing that has to decide what a key
# MEANS. It is declared here, closed, and an unknown key is a hard error: a
# target nothing can interpret is a target that will be silently ignored by
# whatever consumes priors next, and silence is the failure mode the whole
# knowledge base is built to avoid.
#
# Every key names a field of the MEASUREMENT REPORT, because that is what a
# prior is for: the planner composes it against what it measured. A key naming
# a device parameter (tape speed, head bump) has no home here — see the report
# at the end of this work for why that is a gap in the schema rather than in
# this table.


@dataclass(frozen=True)
class TargetSpec:
    section: str                # measurement report section
    key: str                    # field within it
    node_type: str              # which node the move needs
    absolute: bool              # False = move is (target - measured)
    unit: str
    meaning: str


TARGETS: dict[str, TargetSpec] = {
    "tilt_db_per_octave_re_pink": TargetSpec(
        "spectrum", "tilt_db_per_octave_re_pink", "parametric_eq", False,
        "dB/octave",
        "Overall spectral slope, 100 Hz to 10 kHz, where 0 is pink (what an "
        "engineer calls neutral) and negative is darker."),
    "hf_rolloff_db_per_octave": TargetSpec(
        "spectrum", "hf_rolloff_db_per_octave", "parametric_eq", False,
        "dB/octave",
        "Slope of the top end above 6.3 kHz. A tonal fact, not a bandwidth "
        "wall — a dark source can be brightened, a codec-limited one cannot."),
    "bandwidth_limit_hz": TargetSpec(
        "spectrum", "bandwidth_limit_hz", "parametric_eq", True,
        "Hz",
        "A hard ceiling on content, as a period machine or medium would "
        "impose. Absolute rather than a delta: you cannot un-limit a source. "
        "NOT VERIFIABLE FROM THE RENDER — the report's bandwidth_limit_hz "
        "detects a codec wall, and a musical low-pass is not one, so this "
        "target reads back as 'not measurable' and that is correct rather "
        "than a failure."),
    "crest_factor_db": TargetSpec(
        "dynamics", "crest_factor_db", "compressor", False,
        "dB",
        "Peak minus RMS over gated material. Lower is more compressed."),
}


def target_help() -> str:
    lines = []
    for name, spec in TARGETS.items():
        kind = "absolute" if spec.absolute else "delta from the source"
        lines.append(f"  {name} ({spec.unit}, {kind}, needs {spec.node_type})\n"
                     f"      {spec.meaning}")
    return "\n".join(lines)


# The tilt is fitted over 100 Hz to 10 kHz and the top-end slope above 6.3 kHz;
# both are least-squares slopes against log2(frequency), so converting a slope
# into a shelf gain is a span in octaves. Read from the measurement layer's own
# fit windows rather than guessed.
TILT_LOW_HZ, TILT_HIGH_HZ = 100.0, 10000.0
TILT_PIVOT_HZ = 1000.0
HF_FIT_LOW_HZ, HF_FIT_HIGH_HZ = 6300.0, 20000.0

MAX_BAND_GAIN_DB = 18.0     # EqBand's own bound; clamping here so the message
                            # names the target rather than the band


# ---------------------------------------------------------------------------
# check


def check_preset(data: Any, *, denied: Sequence[str] | None = None
                 ) -> tuple[Preset | None, list[Problem]]:
    """Validate one dossier and translate everything wrong with it.

    Returns the parsed preset when it parses, so a caller that wants to go on
    and render does not have to validate twice.
    """
    problems: list[Problem] = []

    if not isinstance(data, dict):
        return None, [Problem("(whole dossier)",
                              f"the file contains a {type(data).__name__}, not "
                              f"a JSON object",
                              "a dossier is a single JSON object")]

    try:
        preset = Preset.model_validate(data)
    except ValidationError as error:
        return None, [_translate(entry, data) for entry in error.errors()]

    problems += _check_priors(preset)
    problems += _check_evidence(preset)
    problems += _check_names(preset, denied)
    return preset, problems


def _check_priors(preset: Preset) -> list[Problem]:
    """Rules about `parameter_priors` that Pydantic has no way to express.

    `chain_skeleton` is `list[str]` and `targets` is a free-form dict, so
    everything here is about a value that parses and means nothing.
    """
    problems: list[Problem] = []
    priors = preset.parameter_priors
    skeleton = list(priors.chain_skeleton)

    for index, node_type in enumerate(skeleton):
        if node_type in PARAM_MODELS:
            continue
        near = difflib.get_close_matches(node_type, list(PARAM_MODELS), n=1,
                                         cutoff=0.5)
        problems.append(Problem(
            f"parameter_priors.chain_skeleton[{index}]",
            f"{node_type!r} is not a node type the engine has",
            (f"did you mean {near[0]!r}? " if near else "") +
            f"the closed set is: {', '.join(sorted(PARAM_MODELS))}"))

    for index, node_type in enumerate(skeleton):
        if node_type in MASTER_ONLY:
            problems.append(Problem(
                f"parameter_priors.chain_skeleton[{index}]",
                f"{node_type!r} is only correct after the output sample-rate "
                f"conversion, so it cannot be part of a preset's chain",
                f"remove it. {', '.join(MASTER_ONLY)} are delivery decisions "
                f"the pipeline makes, not era characteristics"))

    duplicates = {t for t in skeleton if skeleton.count(t) > 1}
    for node_type in sorted(duplicates):
        problems.append(Problem(
            "parameter_priors.chain_skeleton",
            f"{node_type!r} appears {skeleton.count(node_type)} times and the "
            f"skeleton has no ids, so nothing can say which one a target moves",
            "list it once. If the era genuinely needs two, the schema cannot "
            "express that yet — say so in review_note and raise it"))

    for name, target in priors.targets.items():
        spec = TARGETS.get(name)
        if spec is None:
            near = difflib.get_close_matches(name, list(TARGETS), n=1, cutoff=0.5)
            problems.append(Problem(
                f"parameter_priors.targets.{name}",
                f"{name!r} is not a target this system can interpret",
                (f"did you mean {near[0]!r}? " if near else "") +
                "the vocabulary is closed, because a target nothing reads is a "
                "target that is silently ignored:\n" + target_help()))
            continue
        if spec.node_type not in skeleton:
            problems.append(Problem(
                f"parameter_priors.targets.{name}",
                f"this target needs a {spec.node_type!r} and chain_skeleton "
                f"does not list one",
                f"add {spec.node_type!r} to chain_skeleton, or drop the "
                f"target. The tool will not add a node you did not ask for"))

    if priors.targets and not skeleton:
        problems.append(Problem(
            "parameter_priors.chain_skeleton",
            "there are targets but no chain at all",
            "list the node types the era needs, in order, e.g. "
            "[\"parametric_eq\", \"compressor\"]"))

    for index, band in enumerate(priors.avoid_bands):
        for name, target in priors.targets.items():
            spec = TARGETS.get(name)
            if spec is None or not spec.absolute:
                continue
            if name == "bandwidth_limit_hz" and band.center_freq_hz > target.value:
                problems.append(Problem(
                    f"parameter_priors.avoid_bands[{index}]",
                    f"this band is at {band.center_freq_hz:.0f} Hz, above the "
                    f"{target.value:.0f} Hz bandwidth limit the same dossier "
                    f"asks for — nothing will reach it",
                    "remove the avoid_band, or raise bandwidth_limit_hz",
                    severity="warning"))

    if not skeleton and not priors.targets and not priors.avoid_bands:
        problems.append(Problem(
            "parameter_priors", "the dossier carries no priors at all",
            "a dossier with evidence and no priors is a history lesson the "
            "planner cannot act on. Add a chain_skeleton and at least one "
            "target once you know what the evidence implies",
            severity="warning"))

    return problems


def _check_evidence(preset: Preset) -> list[Problem]:
    problems: list[Problem] = []

    if not preset.evidence:
        problems.append(Problem(
            "evidence", "no claims yet",
            "a scaffold is allowed to be empty, but a preset with no evidence "
            "can never become usable — `usable` requires at least one high- or "
            "medium-confidence claim",
            severity="warning"))

    for index, item in enumerate(preset.evidence):
        where = f"evidence[{index}] (claim {_elide(item.claim)!r})"
        if not item.dsp_implication.strip():
            problems.append(Problem(
                where, "no dsp_implication",
                "say what this means for processing, in measurable terms. The "
                "schema's own words: a claim with no DSP implication is trivia, "
                "and it reaches the planner either way",
                severity="warning"))
        if item.citation is not None and item.citation.page is None and \
                not item.citation.url:
            problems.append(Problem(
                where, "the citation names a source but gives neither page nor "
                       "url",
                "add citation.page. A citation to a whole book is not a "
                "citation — the next person has to be able to check it",
                severity="warning"))

    if preset.legal_review_status is ReviewStatus.APPROVED and not preset.usable:
        problems.append(Problem(
            "legal_review_status",
            "this dossier is approved but will never be returned by "
            "usable_presets()",
            "approval is not enough: `usable` also needs at least one claim at "
            "high or medium confidence. Right now every claim is disputed, "
            "unverified, or absent, so the preset would be a set of numbers "
            "somebody liked presented as heritage"))

    return problems


def _check_names(preset: Preset, denied: Sequence[str] | None) -> list[Problem]:
    """No routed artist name may appear in what the model is shown.

    THIS IS THE CHECK THE REST OF THE SYSTEM ASSUMES SOMEBODY IS DOING. The
    alias routing table exists so an artist's name is matched in Python and
    never enters a prompt, and `name_filter_hits` exists to catch one that
    escapes into generated text. Neither of them looks at the dossier — and
    `as_dossier` puts `display_name`, every `claim`, every `dsp_implication`
    and every `negative_constraint` straight into the prompt.

    So the one place a name can walk past the guarantee is the file an author
    types by hand, which is exactly this file. Scanned by serialising
    `as_dossier` itself rather than by listing the fields here, so the check
    cannot drift from what the model is actually shown.

    Note what is NOT scanned, because it is deliberate: `citation.source` is
    absent from `as_dossier`, so a book with a band's name in the title is
    fine, and must be, or the citations would be unwritable.
    """
    names = list(denied) if denied is not None else loader.denied_names()
    if not names:
        return []

    problems: list[Problem] = []
    text = json.dumps(as_dossier(preset))
    hits = name_filter_hits(text, names)
    if hits:
        problems.append(Problem(
            "(text shown to the model)",
            f"{', '.join(repr(h) for h in hits)} appears in the dossier's "
            f"display_name, claims, dsp_implications or negative_constraints — "
            f"all of which are sent to the model verbatim",
            "rewrite it to describe the era, the room or the equipment "
            "instead. The routing table matches the name in Python before the "
            "model is called precisely so the model never sees it; a name "
            "typed into a claim walks straight past that"))

    problems += _name_in_preset_id(preset, names)
    return problems


def _name_in_preset_id(preset: Preset, names: Sequence[str]) -> list[Problem]:
    """An artist's name hidden inside the id, where the word filter cannot see.

    FOUND BY TRYING TO WRITE A TEST FOR THE CHECK ABOVE. `name_filter_hits`
    matches on word boundaries, and `_` is a word character — so `\\bBeatles\\b`
    does not match `era_beatles_1965_guitar`, and the preset_id is the ONE
    field of the dossier that reaches the model as a value it is asked to
    choose between. The word-boundary rule is right for prose and blind to the
    only field written in snake_case.

    Matched on underscore-separated tokens instead, plus the joined and
    concatenated forms of a multi-word alias, because `fab_four` and `fabfour`
    are the two ways someone writes "Fab Four" as an identifier.
    """
    identifier = preset.preset_id.lower()
    tokens = set(identifier.split("_"))
    problems: list[Problem] = []
    for name in names:
        parts = name.lower().split()
        if not parts:
            continue
        forms = {"_".join(parts), "".join(parts)}
        hit = any(form in identifier for form in forms) or \
            (len(parts) == 1 and parts[0] in tokens)
        if hit:
            problems.append(Problem(
                "preset_id",
                f"{preset.preset_id!r} contains the routed alias {name!r}",
                "preset_id is the only field of the dossier that crosses into "
                "the effect graph, and it is offered to the model as an enum "
                "value — so a name in it is a name the model can echo. Name "
                "the era and the instrument instead"))
    return problems


def check_path(path: str, *, denied: Sequence[str] | None = None
               ) -> tuple[Preset | None, list[Problem]]:
    try:
        with open(path) as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None, [Problem(path, "no such file", "check the path")]
    except json.JSONDecodeError as error:
        return None, [Problem(
            f"{path}:{error.lineno}",
            f"not valid JSON: {error.msg}",
            "a trailing comma and an unquoted key are the two usual causes")]
    return check_preset(data, denied=denied)


def check_directory(directory: str, *, denied: Sequence[str] | None = None
                    ) -> dict[str, list[Problem]]:
    out: dict[str, list[Problem]] = {}
    seen: dict[str, str] = {}
    if not os.path.isdir(directory):
        return out
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(directory, name)
        preset, problems = check_path(path, denied=denied)
        if preset is not None:
            first = seen.get(preset.preset_id)
            if first is not None:
                problems = list(problems) + [Problem(
                    "preset_id",
                    f"{preset.preset_id!r} is already claimed by {first}",
                    "the id is a closed enum in the graph schema and must "
                    "identify exactly one dossier — rename one of them")]
            else:
                seen[preset.preset_id] = name
        out[path] = problems
    return out


# ---------------------------------------------------------------------------
# Building the audition graph


@dataclass
class Move:
    """One target, and everything about how it was turned into processing."""

    target: str
    requested: float
    tolerance: float
    measured_source: float | None
    delta: float | None
    node_id: str
    how: str
    achieved: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"target": self.target, "requested": self.requested,
                "tolerance": self.tolerance,
                "source": self.measured_source, "delta": self.delta,
                "node_id": self.node_id, "how": self.how,
                "achieved": self.achieved}


@dataclass
class Audition:
    graph: EffectGraph
    moves: list[Move]
    defaults: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


def _clamp_band_gain(gain: float, target: str, notes: list[str]) -> float:
    if abs(gain) <= MAX_BAND_GAIN_DB:
        return gain
    notes.append(f"{target}: the move needs {gain:+.1f} dB of shelf, which is "
                 f"past the schema's {MAX_BAND_GAIN_DB:.0f} dB band limit; "
                 f"clamped")
    return math.copysign(MAX_BAND_GAIN_DB, gain)


def _measured(report: dict[str, Any], spec: TargetSpec) -> float | None:
    value = (report.get(spec.section) or {}).get(spec.key)
    if value is None or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def build_audition(preset: Preset, source_report: dict[str, Any]
                   ) -> tuple[Audition | None, list[Problem]]:
    """Turn priors plus a measured source into a graph, or say why not.

    Refuses rather than guessing, every time. A missing measurement is a
    refusal and never a substituted constant: a fabricated number is
    indistinguishable from a measured one the moment a move is derived from it,
    and the author would be listening to an invention.
    """
    problems: list[Problem] = []
    priors = preset.parameter_priors
    moves: list[Move] = []
    eq_bands: list[dict[str, Any]] = []
    compressor_params: dict[str, Any] | None = None
    notes: list[str] = []

    for name in sorted(priors.targets):
        target = priors.targets[name]
        spec = TARGETS.get(name)
        if spec is None:
            continue        # already an error from check_preset
        measured = _measured(source_report, spec)

        if spec.absolute:
            delta = None
        elif measured is None:
            problems.append(Problem(
                f"parameter_priors.targets.{name}",
                f"this target is a delta from the source, and the source has "
                f"no measurable {spec.section}.{spec.key}",
                "audition against a source where that measurement exists. "
                "There is no default for it, and there must not be: a "
                "substituted number is indistinguishable from a measured one "
                "once a move has been derived from it"))
            continue
        else:
            delta = float(target.value) - measured

        if name == "tilt_db_per_octave_re_pink":
            half_span = math.log2(TILT_HIGH_HZ / TILT_LOW_HZ) / 2.0
            gain = _clamp_band_gain(float(delta) * half_span, name, notes)
            eq_bands.append({"filter_type": "lowshelf",
                             "center_freq_hz": TILT_PIVOT_HZ,
                             "gain_db": -gain, "q_factor": 0.707})
            eq_bands.append({"filter_type": "highshelf",
                             "center_freq_hz": TILT_PIVOT_HZ,
                             "gain_db": gain, "q_factor": 0.707})
            moves.append(Move(
                name, target.value, target.tolerance, measured, delta,
                "priors_eq",
                f"mirrored shelves at {TILT_PIVOT_HZ:.0f} Hz, "
                f"{gain:+.2f} dB high / {-gain:+.2f} dB low. A shelf pair does "
                f"not deliver its nominal slope, and whether it over- or "
                f"under-delivers depends on the source spectrum; the achieved "
                f"figure is the one to read"))

        elif name == "hf_rolloff_db_per_octave":
            span = math.log2(HF_FIT_HIGH_HZ / HF_FIT_LOW_HZ)
            gain = _clamp_band_gain(float(delta) * span, name, notes)
            eq_bands.append({"filter_type": "highshelf",
                             "center_freq_hz": HF_FIT_LOW_HZ,
                             "gain_db": gain, "q_factor": 0.707})
            moves.append(Move(
                name, target.value, target.tolerance, measured, delta,
                "priors_eq",
                f"high shelf at {HF_FIT_LOW_HZ:.0f} Hz, {gain:+.2f} dB"))

        elif name == "bandwidth_limit_hz":
            eq_bands.append({"filter_type": "lowpass",
                             "center_freq_hz": float(target.value),
                             "gain_db": 0.0, "q_factor": 0.707})
            moves.append(Move(
                name, target.value, target.tolerance, measured, None,
                "priors_eq",
                f"low-pass at {target.value:.0f} Hz. Absolute, not a delta — "
                f"the source measures "
                f"{'no limit at all' if measured is None else f'{measured:.0f} Hz'}"
                f". Expect 'not measurable' in the rendered column: the "
                f"report's bandwidth_limit_hz looks for a codec wall, and a "
                f"musical low-pass is not one"))

        elif name == "crest_factor_db":
            params, why, refusal = _compressor_for_crest(
                float(delta), source_report, name)
            if refusal is not None:
                problems.append(refusal)
                continue
            compressor_params = params
            moves.append(Move(name, target.value, target.tolerance, measured,
                              delta, "priors_comp", why))

    if problems:
        return None, problems

    # The skeleton decides the ORDER, and every node in it is rendered. A node
    # no target speaks to is rendered at the schema's defaults and reported as
    # such — loudly, because at defaults `drive` is 6 dB and `tape` has a 2 dB
    # head bump, so "nothing said about it" does not mean "nothing happens".
    nodes: list[dict[str, Any]] = []
    defaults: list[tuple[str, dict[str, Any]]] = []
    for index, node_type in enumerate(priors.chain_skeleton):
        node_id = f"priors_{index}_{node_type}"
        params: dict[str, Any] = {}
        if node_type == "parametric_eq" and eq_bands:
            params = {"bands": eq_bands}
            node_id = "priors_eq"
        elif node_type == "compressor" and compressor_params is not None:
            params = compressor_params
            node_id = "priors_comp"
        else:
            params = PARAM_MODELS[node_type]().model_dump()
            defaults.append((node_type, params))
        nodes.append({"id": node_id, "type": node_type, "params": params})

    try:
        graph = EffectGraph(nodes=nodes, explanation="")
    except ValidationError as error:
        return None, [_translate(entry, {}) for entry in error.errors()]

    conflicts = _avoid_band_conflicts(eq_bands, priors.avoid_bands)
    if conflicts:
        return None, conflicts

    for note in notes:
        problems.append(Problem("parameter_priors.targets", note,
                                "the render is what the schema allows, not "
                                "what the prior asked for", severity="warning"))

    return Audition(graph=graph, moves=moves, defaults=defaults), problems


def _compressor_for_crest(delta: float, source_report: dict[str, Any],
                          name: str
                          ) -> tuple[dict[str, Any] | None, str, Problem | None]:
    """Threshold and ratio that remove a stated number of dB of crest.

    Closed form rather than a solver, and deliberately: `backend/refine` has a
    render-in-the-loop matcher for a gain-reduction target, and an audition is
    not a master. What the author needs is to hear roughly this much
    compression with the level difference removed, and to be told what it
    actually achieved.

    A compressor above threshold T with ratio R attenuates a peak at P by
    (P - T)(1 - 1/R). Setting T at the source's 90th-percentile short-term RMS
    puts the knee where the loud passages live, and R falls out of the crest
    reduction asked for.

    THE THRESHOLD IS DERIVED FROM THE SOURCE AND THE COMPRESSOR SEES THE EQ.
    Anything in the chain before it has already moved the level, so the crest
    the render achieves can miss the target badly and can even move the wrong
    way — a shelf cutting the low-mids of an acoustic guitar removes far more
    RMS than peak. This is not corrected for, and it is not hidden either: the
    achieved figure is measured off the written file and printed beside the
    requested one, so an author sees the miss rather than being told the number
    they asked for.
    """
    where = f"parameter_priors.targets.{name}"
    if delta >= 0.0:
        return None, "", Problem(
            where,
            f"the target is {delta:+.1f} dB ABOVE what the source measures, "
            f"which asks for more crest than the source has",
            "this tool only reduces crest, with a compressor. Restoring "
            "transients needs a transient_designer and there is no defined "
            "mapping from a crest target to one, so it refuses rather than "
            "rendering something that does not answer the target")

    percentiles = (source_report.get("dynamics") or {}).get(
        "short_term_rms_percentiles") or {}
    threshold = percentiles.get("p90_dbfs")
    peak = (source_report.get("loudness") or {}).get("sample_peak_dbfs")
    if threshold is None or peak is None:
        return None, "", Problem(
            where,
            "the source has no measurable 90th-percentile short-term RMS or no "
            "sample peak, so there is nothing to place a threshold against",
            "audition against a source with programme material in it")

    threshold = float(threshold)
    peak = float(peak)
    headroom = peak - threshold
    if headroom <= 1.0:
        return None, "", Problem(
            where,
            f"the source's peak ({peak:.1f} dBFS) is only {headroom:.1f} dB "
            f"above its 90th-percentile RMS ({threshold:.1f} dBFS), so a "
            f"compressor has almost nothing to act on",
            "this source is already flat; audition the crest target against "
            "something dynamic")

    reduction = -delta
    clamped = ""
    if reduction >= headroom:
        ratio = 20.0
        clamped = (f" The {reduction:.1f} dB asked for is more than the "
                   f"{headroom:.1f} dB of headroom above the threshold, so the "
                   f"ratio is pinned at 20:1 and the target is not reachable "
                   f"this way.")
    else:
        ratio = 1.0 / (1.0 - reduction / headroom)
        ratio = min(20.0, max(1.0, ratio))

    params = PARAM_MODELS["compressor"]().model_dump()
    params.update({"threshold_dbfs": max(-60.0, min(0.0, threshold)),
                   "ratio_to_1": ratio, "makeup_gain_db": 0.0})
    why = (f"compressor at {params['threshold_dbfs']:.1f} dBFS, "
           f"{ratio:.1f}:1 — {reduction:.1f} dB of crest to remove from "
           f"{headroom:.1f} dB of headroom above the 90th-percentile RMS."
           f"{clamped} No makeup gain: the A/B is loudness-matched, and makeup "
           f"here would be a level difference in disguise")
    return params, why, None


def _avoid_band_conflicts(eq_bands: Sequence[dict[str, Any]],
                          avoid: Sequence[AvoidBand]) -> list[Problem]:
    """A dossier that boosts where it says not to boost.

    Only checkable at audition time, because the sign of a delta move is a
    property of the SOURCE — the same prior cuts one source and boosts another.
    Only boosts count: an avoid_band reads as "do not put energy here", and the
    schema's own example says so.
    """
    problems: list[Problem] = []

    # A low-pass in the same node takes the band away again, so a shelf that
    # nominally boosts above it is not putting energy there. Without this the
    # commonest era prior of all — dark, band-limited, and told not to boost
    # air — reports a conflict with itself on every source, and an author who
    # sees a guard fire where it should not is an author who stops reading it.
    cutoffs = [float(band["center_freq_hz"]) for band in eq_bands
               if band["filter_type"] == "lowpass"]

    for band in eq_bands:
        gain = float(band["gain_db"])
        if gain < 1.0:
            continue
        for entry in avoid:
            centre = entry.center_freq_hz
            if any(centre >= cutoff for cutoff in cutoffs):
                continue
            corner = float(band["center_freq_hz"])
            kind = band["filter_type"]
            if kind == "highshelf":
                hit = centre >= corner
            elif kind == "lowshelf":
                hit = centre <= corner
            elif kind == "bell":
                hit = abs(math.log2(centre / corner)) <= 0.5
            else:
                hit = False
            if not hit:
                continue
            problems.append(Problem(
                "parameter_priors",
                f"against this source the priors want a {gain:+.1f} dB "
                f"{kind} at {corner:.0f} Hz, which boosts {centre:.0f} Hz — "
                f"where the same dossier's avoid_bands says: {entry.reason!r}",
                "the dossier contradicts itself on this source. Either the "
                "target is wrong, or the avoid_band is, and rendering it would "
                "let you hear something your own preset forbids"))
    return problems


# ---------------------------------------------------------------------------
# Rendering


def load_source(path: str, seconds: float) -> tuple[np.ndarray, float]:
    audio = wavio.read_wav(path)
    samples = np.atleast_2d(audio.samples)
    if seconds > 0:
        keep = int(round(seconds * audio.sample_rate))
        samples = samples[:, :keep]
    return samples, float(audio.sample_rate)


def measure_array(samples: np.ndarray, sample_rate: float) -> dict[str, Any]:
    return report_mod.measure(samples, sample_rate).to_dict()


NULL_GRAPH = EffectGraph(nodes=[], explanation="")


def render_pair(samples: np.ndarray, sample_rate: float, graph: EffectGraph,
                directory: str, name: str, binary: str
                ) -> tuple[str, str, match.MatchedGroup]:
    """Null render, priors render, loudness-matched pair on disk.

    THE A SIDE IS A NULL RENDER, NOT THE RAW SOURCE. The engine's
    sample-rate conversion is transparent in the audible band and not bit-exact
    — a null test against the raw file fails at 22 kHz by 12 dB — so comparing
    a render to the source measures the conversion path along with the preset.
    Compare against a null render, always.

    Intermediates are written as 64-bit float. A render carrying a boost can
    legitimately sit over full scale before the matcher's common ceiling offset
    is applied, and PCM would clamp it there, inside the stimulus.
    """
    os.makedirs(directory, exist_ok=True)
    work = os.path.join(directory, "_intermediate")
    os.makedirs(work, exist_ok=True)

    null = render_mod.render_array(samples, sample_rate,
                                   NULL_GRAPH.to_recipe(output_format="float64"),
                                   binary=binary)
    wet = render_mod.render_array(samples, sample_rate,
                                  graph.to_recipe(output_format="float64"),
                                  binary=binary)

    null_raw = os.path.join(work, "null.wav")
    wet_raw = os.path.join(work, f"{name}.wav")
    match.write_clip(null_raw, null, sample_rate, 64, True)
    match.write_clip(wet_raw, wet, sample_rate, 64, True)

    a_path = os.path.join(directory, "a_null.wav")
    b_path = os.path.join(directory, f"b_{name}.wav")
    group = match.match_group([null_raw, wet_raw], [a_path, b_path],
                              what=f"audition of {name}")
    return a_path, b_path, group


# ---------------------------------------------------------------------------
# diff
#
# The fields two dossiers are compared on, and the difference below which this
# tool will not claim they differ.
#
# THESE ARE REPORTING THRESHOLDS, NOT JUST-NOTICEABLE DIFFERENCES. Nothing here
# has been correlated against human judgement — that is what the listening
# harness is for, and until it has been run, a number presented as a JND would
# be exactly the fabricated measurement this codebase refuses everywhere else.
# They are set at roughly the resolution at which a difference is worth an
# author's attention, and the raw numbers are printed either way so the reader
# can apply their own bar.

DIFF_FIELDS: tuple[tuple[str, str, float], ...] = (
    ("spectrum", "tilt_db_per_octave_re_pink", 0.3),
    ("spectrum", "hf_rolloff_db_per_octave", 0.5),
    ("spectrum", "tilt_residual_db", 0.5),
    ("dynamics", "crest_factor_db", 0.5),
    ("dynamics", "peak_to_loudness_ratio_db", 0.5),
    ("loudness", "loudness_range_lu", 0.5),
    ("stereo", "correlation", 0.05),
    ("stereo", "mono_loss_db", 0.5),
)

# The strongest single signal that two chains are not the same chain: the
# biggest third-octave band level difference anywhere in the spectrum.
BAND_DIFFERENCE_DB = 1.0


@dataclass
class Difference:
    what: str
    left: float | None
    right: float | None
    delta: float | None
    threshold: float
    differs: bool


def compare_reports(left: dict[str, Any], right: dict[str, Any]
                    ) -> list[Difference]:
    out: list[Difference] = []
    for section, key, threshold in DIFF_FIELDS:
        a = (left.get(section) or {}).get(key)
        b = (right.get(section) or {}).get(key)
        if a is None or b is None:
            # One side has no measurement. That is itself a difference worth
            # printing, and it is NOT reported as a delta of zero.
            out.append(Difference(f"{section}.{key}", a, b, None, threshold,
                                  a is not b))
            continue
        delta = float(b) - float(a)
        out.append(Difference(f"{section}.{key}", float(a), float(b), delta,
                              threshold, abs(delta) > threshold))

    a_bands = {entry["centre_hz"]: entry["level_db"]
               for entry in (left.get("spectrum") or {}).get("third_octave_db", [])}
    b_bands = {entry["centre_hz"]: entry["level_db"]
               for entry in (right.get("spectrum") or {}).get("third_octave_db", [])}
    worst_hz, worst = None, 0.0
    for centre, a in a_bands.items():
        b = b_bands.get(centre)
        if a is None or b is None:
            continue
        if abs(b - a) > abs(worst):
            worst_hz, worst = centre, b - a
    out.append(Difference(
        f"third_octave_db (worst band"
        f"{f', {worst_hz:.0f} Hz' if worst_hz else ''})",
        None, None, worst if worst_hz else None, BAND_DIFFERENCE_DB,
        abs(worst) > BAND_DIFFERENCE_DB))
    return out


# ---------------------------------------------------------------------------
# new


def _ask_number(prompt: str, rule: str, cast=float, *, required: bool = False
                ) -> Any:
    """A numeric prompt that re-asks rather than dying on a typo.

    An interactive scaffolder that raises `ValueError` on a mistyped page
    number and loses forty claims is worse than no scaffolder.
    """
    while True:
        answer = _ask(prompt, rule, required=required)
        if not answer:
            return None
        try:
            return cast(answer)
        except ValueError:
            print(f"  {answer!r} is not a number")


def _ask(prompt: str, rule: str, *, required: bool = False,
         default: str = "") -> str:
    """One prompt, with the rule it enforces stated above it.

    THE RULE IS PRINTED EVERY TIME, not once at the top. An author authoring
    their fourth dossier will skim, and the cost of re-reading one line is far
    below the cost of a claim written at the wrong confidence.
    """
    while True:
        print()
        print(f"  {rule}")
        suffix = f" [{default}]" if default else ""
        try:
            answer = input(f"{prompt}{suffix}: ").strip()
        except EOFError:
            raise SystemExit(
                "\nstdin closed. Run `new` interactively, or pass every field "
                "as a flag — see `author new --help`.")
        if not answer and default:
            return default
        if answer or not required:
            return answer
        print("  (required)")


def new_interactive() -> dict[str, Any]:
    print("Scaffolding a preset dossier.\n"
          "A PRESET IS NOT AN EFFECT GRAPH. It is evidence, intent and "
          "parameter priors;\nthe planner composes it against what it measured. "
          "Blank ends a list.\n")

    data: dict[str, Any] = {
        "preset_id": _ask(
            "preset_id", "Lowercase letters, digits and underscores, 4-60 "
            "characters. This is the ONLY thing that crosses into the effect "
            "graph and it is a closed enum, so it must not contain an artist's "
            "name — describe the era and instrument: era_uk_1965_beatgroup_guitar.",
            required=True),
        "display_name": _ask(
            "display_name", "3-80 characters, shown to users. Also sent to the "
            "model, so no artist, band or record title here either.",
            required=True),
        "evidence": [],
        "parameter_priors": {"chain_skeleton": [], "targets": {},
                             "avoid_bands": []},
        "negative_constraints": [],
        "legal_review_status": "draft",
    }

    evidence: list[dict[str, Any]] = []
    while True:
        claim = _ask(
            f"claim #{len(evidence) + 1}",
            "One factual claim about how a record was made, 4-400 characters. "
            "You will be asked for its citation immediately — the schema "
            "refuses a confident claim without one, and finding that out after "
            "typing forty of them is the annoying way.")
        if not claim:
            break

        confidence = ""
        while confidence not in {c.value for c in Confidence}:
            confidence = _ask(
                "confidence",
                "high / medium / low need a citation. disputed / unverified do "
                "not, but need a review_note saying why — that is how a "
                "half-known fact gets recorded without being quietly promoted "
                "to a fact.", default="high").lower()
            if confidence not in {c.value for c in Confidence}:
                print(f"  unknown value. One of: "
                      f"{', '.join(c.value for c in Confidence)}")

        entry: dict[str, Any] = {"claim": claim, "confidence": confidence}

        if confidence in ("high", "medium", "low"):
            entry["citation"] = {
                "source": _ask("citation.source",
                               "Author, title. At least 4 characters — enough "
                               "to find it again.", required=True)}
            page = _ask_number(
                "citation.page",
                "The page this claim is on. A citation to a whole book is not "
                "a citation. Blank only if the source genuinely has no pages, "
                "in which case give a url.", int)
            if page is not None:
                entry["citation"]["page"] = page
            else:
                url = _ask("citation.url", "URL of the source, since it has no "
                                           "page.")
                if url:
                    entry["citation"]["url"] = url
        else:
            entry["review_note"] = _ask(
                "review_note",
                "Why this is doubtful, for the next author. An unverified "
                "claim with no reason to doubt it is indistinguishable from "
                "one nobody checked.", required=True)

        entry["dsp_implication"] = _ask(
            "dsp_implication",
            "What this claim means for processing, in measurable terms. A "
            "claim with no DSP implication is trivia — and it reaches the "
            "planner either way.")
        evidence.append(entry)
    data["evidence"] = evidence

    skeleton: list[str] = []
    print(f"\n  node types: {', '.join(sorted(PARAM_MODELS))}")
    while True:
        node_type = _ask(
            f"chain_skeleton[{len(skeleton)}]",
            "A node type, in the order the era's signal path had it. Not "
            f"parameters — the schema has nowhere to put those. "
            f"{', '.join(MASTER_ONLY)} are delivery decisions and do not belong "
            f"in a preset.")
        if not node_type:
            break
        if node_type not in PARAM_MODELS:
            print(f"  unknown node type {node_type!r} — a typo here would be "
                  f"silently ignored downstream, so it is refused here")
            continue
        skeleton.append(node_type)
    data["parameter_priors"]["chain_skeleton"] = skeleton

    targets: dict[str, Any] = {}
    print(f"\n  targets this system can interpret:\n{target_help()}")
    while True:
        name = _ask("target name",
                    "One of the names above. The vocabulary is closed: a "
                    "target nothing reads is a target that is silently ignored.")
        if not name:
            break
        if name not in TARGETS:
            print(f"  unknown target {name!r}")
            continue
        value = _ask_number(f"{name} value", f"In {TARGETS[name].unit}.",
                            required=True)
        tolerance = _ask_number(
            f"{name} tolerance",
            "Greater than zero, always. A prior with no tolerance is a "
            "specification, and the planner composes priors against a source "
            "rather than hitting them.", required=True)
        targets[name] = {"value": value, "tolerance": tolerance}
    data["parameter_priors"]["targets"] = targets

    avoid: list[dict[str, Any]] = []
    while True:
        freq = _ask_number(f"avoid_bands[{len(avoid)}] centre Hz",
                           "A frequency this era's material will not tolerate "
                           "energy being added at, 20-20000 Hz.")
        if freq is None:
            break
        reason = _ask("reason", "At least 8 characters, and it has to be a real "
                                "reason — the next author decides whether to "
                                "keep this band from what you write here.",
                      required=True)
        avoid.append({"center_freq_hz": freq, "reason": reason})
    data["parameter_priors"]["avoid_bands"] = avoid

    constraints: list[str] = []
    while True:
        line = _ask(f"negative_constraint[{len(constraints)}]",
                    "Something the planner must NOT do for this era, in plain "
                    "language. Sent to the model verbatim, so no artist names.")
        if not line:
            break
        constraints.append(line)
    data["negative_constraints"] = constraints

    data["review_note"] = _ask(
        "review_note",
        "Anything legal or editorial review needs to know. Not shown to the "
        "model — it is about our process, not about the audio.")
    return data


def new_from_flags(arguments: argparse.Namespace) -> dict[str, Any]:
    data: dict[str, Any] = {
        "preset_id": arguments.preset_id,
        "display_name": arguments.display_name,
        "evidence": [],
        "parameter_priors": {
            "chain_skeleton": list(arguments.chain or []),
            "targets": {}, "avoid_bands": []},
        "negative_constraints": list(arguments.negative or []),
        "legal_review_status": arguments.status,
        "review_note": arguments.review_note or "",
    }

    if arguments.claim:
        entry: dict[str, Any] = {"claim": arguments.claim,
                                 "confidence": arguments.confidence,
                                 "dsp_implication": arguments.dsp_implication or ""}
        if arguments.citation_source:
            citation: dict[str, Any] = {"source": arguments.citation_source}
            if arguments.citation_page is not None:
                citation["page"] = arguments.citation_page
            if arguments.citation_url:
                citation["url"] = arguments.citation_url
            entry["citation"] = citation
        if arguments.review_note_claim:
            entry["review_note"] = arguments.review_note_claim
        data["evidence"].append(entry)

    for raw in arguments.target or []:
        # name=value+/-tolerance, e.g. tilt_db_per_octave_re_pink=-1.5+/-1.0
        if "=" not in raw or "+/-" not in raw:
            raise SystemExit(
                f"--target {raw!r} is not name=value+/-tolerance, e.g. "
                f"--target tilt_db_per_octave_re_pink=-1.5+/-1.0. The "
                f"tolerance is not optional: a prior without one is a "
                f"specification.")
        name, rest = raw.split("=", 1)
        value, tolerance = rest.split("+/-", 1)
        try:
            numbers = (float(value), float(tolerance))
        except ValueError:
            raise SystemExit(f"--target {raw!r}: {value!r} and {tolerance!r} "
                             f"must both be numbers")
        data["parameter_priors"]["targets"][name.strip()] = {
            "value": numbers[0], "tolerance": numbers[1]}

    for raw in arguments.avoid or []:
        if "=" not in raw:
            raise SystemExit(f"--avoid {raw!r} is not frequency=reason")
        freq, reason = raw.split("=", 1)
        try:
            centre = float(freq)
        except ValueError:
            raise SystemExit(f"--avoid {raw!r}: {freq!r} is not a frequency")
        data["parameter_priors"]["avoid_bands"].append(
            {"center_freq_hz": centre, "reason": reason.strip()})

    return data


# ---------------------------------------------------------------------------
# Commands


def command_new(arguments: argparse.Namespace) -> int:
    if arguments.preset_id and arguments.display_name:
        data = new_from_flags(arguments)
    elif arguments.preset_id or arguments.display_name:
        print("give both --preset-id and --display-name, or neither for the "
              "interactive walkthrough", file=sys.stderr)
        return 2
    else:
        data = new_interactive()

    # A SCAFFOLD THAT DOES NOT PASS `check` IS NOT A SCAFFOLD. It is the first
    # thing the author has to debug, in a format they have not learned yet.
    preset, problems = check_preset(data, denied=arguments.denied)
    if errors(problems):
        print("\nthis would not pass `check`, so it was not written:\n")
        print_problems(problems, "would-be dossier")
        return 1

    # WRITTEN FROM THE VALIDATED MODEL, not from the dict that was typed. Two
    # reasons: the file is then guaranteed to load, and every field the schema
    # has appears in it — including the ones left empty, which is how the next
    # author finds out they exist without opening schema.py.
    assert preset is not None
    text = json.dumps(preset.model_dump(mode="json"), indent=2) + "\n"
    if arguments.out == "-":
        sys.stdout.write(text)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(arguments.out)), exist_ok=True)
        with open(arguments.out, "w") as handle:
            handle.write(text)
        print(f"\nwrote {arguments.out}")
    if problems:
        print_problems(problems, "warnings")
    if not preset.usable:
        print("\nnote: this dossier is not `usable` yet. usable_presets() "
              "returns only APPROVED dossiers carrying at least one high- or "
              "medium-confidence claim, and review gates USE, not storage — so "
              "it can be worked on freely.")
    return 0


def command_check(arguments: argparse.Namespace) -> int:
    target = arguments.path
    if not os.path.exists(target) and not target.endswith(".json"):
        # An absent catalogue directory is the state this project is in, and
        # the loader returns an empty list for it rather than raising. This
        # says the same thing, so `check` on a fresh clone is not an error.
        print(f"{target}: no catalogue directory. An empty catalogue is a "
              f"valid state, and the one this project is deliberately in.")
        return 0
    if os.path.isdir(target):
        results = check_directory(target, denied=arguments.denied)
        if not results:
            print(f"{target}: no dossiers. An empty catalogue is a valid "
                  f"state, and the one this project is deliberately in.")
            return 0
        failed = 0
        for path, problems in sorted(results.items()):
            print_problems(problems, path)
            failed += 1 if errors(problems) else 0
        print(f"\n{len(results)} dossier(s), {failed} with errors")
        return 1 if failed else 0

    _, problems = check_path(target, denied=arguments.denied)
    print_problems(problems, target)
    return 1 if errors(problems) else 0


def _require_engine(binary: str) -> bool:
    if os.path.exists(binary):
        return True
    print(f"engine binary not found at {binary} — run `make render` in engine/",
          file=sys.stderr)
    return False


def _prepared(path: str, arguments: argparse.Namespace) -> Preset | None:
    """Validate a dossier and refuse before anything is rendered.

    AUDITION REFUSES WHAT WOULD NOT VALIDATE. Rendering an invalid dossier
    would produce audio the author would then judge, and a judgement made on a
    render the pipeline would never produce is worse than no render at all —
    the author cannot tell which of the two they are hearing.
    """
    preset, problems = check_path(path, denied=arguments.denied)
    if errors(problems) or preset is None:
        print_problems(problems, path)
        print("\nrefusing to render: fix the errors above first. A render of a "
              "dossier the pipeline would reject is audio you would judge and "
              "we would never ship.")
        return None
    if problems:
        print_problems(problems, path)
    return preset


def command_audition(arguments: argparse.Namespace) -> int:
    preset = _prepared(arguments.dossier, arguments)
    if preset is None:
        return 1
    if not _require_engine(arguments.binary):
        return 2

    samples, sample_rate = load_source(arguments.source, arguments.seconds)
    source_report = measure_array(samples, sample_rate)

    audition, problems = build_audition(preset, source_report)
    if audition is None:
        print_problems(problems, arguments.dossier)
        print("\nrefusing to render.")
        return 1
    if problems:
        print_problems(problems, arguments.dossier)

    a_path, b_path, group = render_pair(
        samples, sample_rate, audition.graph, arguments.out, preset.preset_id,
        arguments.binary)

    # RE-MEASURED FROM THE WRITTEN FILE, not from the array in memory. The
    # author's judgement is made on the bytes on disk, and so is this table.
    written = wavio.read_wav(b_path)
    rendered = measure_array(written.samples, written.sample_rate)
    for move in audition.moves:
        spec = TARGETS[move.target]
        move.achieved = _measured(rendered, spec)

    print(f"\n{preset.display_name}  ({preset.preset_id})")
    print(f"source: {arguments.source}"
          f"{f', first {arguments.seconds:g} s' if arguments.seconds > 0 else ''}")
    print(f"\nA  {a_path}   null render — the engine's own path, no preset")
    print(f"B  {b_path}   the priors")
    print(f"loudness-matched to {group.spread_lu:.4f} LU "
          f"(bar is {match.TOLERANCE_LU} LU), common ceiling offset "
          f"{group.offset_db:+.2f} dB")

    if audition.moves:
        print(f"\n  {'target':<28}{'asked':>9}{'source':>15}{'rendered':>15}")
        for move in audition.moves:
            achieved = ("not measurable" if move.achieved is None
                        else f"{move.achieved:.2f}")
            source = ("not measurable" if move.measured_source is None
                      else f"{move.measured_source:.2f}")
            hit = ""
            if move.achieved is not None:
                inside = abs(move.achieved - move.requested) <= move.tolerance
                hit = "  within tolerance" if inside else "  OUTSIDE TOLERANCE"
            print(f"  {move.target:<28}{move.requested:9.2f}{source:>15}"
                  f"{achieved:>15}{hit}")
            print(f"      {move.how}")
        print("\n  'rendered' is measured off the written B file, so it is what "
              "you will hear.\n  A miss is information, not a bug: the priors "
              "are what the dossier claims,\n  and this is what they do to this "
              "source.")

    if audition.defaults:
        print("\nRENDERED AT SCHEMA DEFAULTS — your dossier does not say what "
              "these should do,\nand a default is not nothing: what you hear "
              "from them is the schema's choice,\nnot your preset's.")
        for node_type, params in audition.defaults:
            shown = ", ".join(f"{k}={v}" for k, v in params.items()
                              if not isinstance(v, list))
            print(f"  {node_type}: {shown or '(no scalar parameters)'}")

    manifest = {
        "preset_id": preset.preset_id,
        "source": os.path.abspath(arguments.source),
        "seconds": arguments.seconds,
        "a_null": a_path,
        "b_priors": b_path,
        "spread_lu": group.spread_lu,
        "offset_db": group.offset_db,
        "graph": audition.graph.model_dump(mode="json"),
        "moves": [move.to_dict() for move in audition.moves],
        "defaults_rendered": [name for name, _ in audition.defaults],
        "note": ("This is not the planner. It is a deterministic single-move "
                 "interpretation of the priors against one measured source, "
                 "for listening."),
    }
    path = os.path.join(arguments.out, "audition.json")
    with open(path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"\n{path}")
    return 0


def command_diff(arguments: argparse.Namespace) -> int:
    left = _prepared(arguments.left, arguments)
    right = _prepared(arguments.right, arguments)
    if left is None or right is None:
        return 1
    if not _require_engine(arguments.binary):
        return 2

    samples, sample_rate = load_source(arguments.source, arguments.seconds)
    source_report = measure_array(samples, sample_rate)

    renders = []
    for preset in (left, right):
        audition, problems = build_audition(preset, source_report)
        if audition is None:
            print_problems(problems, preset.preset_id)
            print("\nrefusing to render.")
            return 1
        renders.append((preset, audition))

    os.makedirs(arguments.out, exist_ok=True)
    work = os.path.join(arguments.out, "_intermediate")
    os.makedirs(work, exist_ok=True)
    # SIDE-PREFIXED, not named after the preset. Two dossiers may legitimately
    # share a preset_id — that is one of the ways a duplicate gets noticed —
    # and naming the files after the id would have one render overwrite the
    # other and report a difference of exactly zero.
    raw_paths, out_paths = [], []
    for side, (preset, audition) in zip(("left", "right"), renders):
        samples_out = render_mod.render_array(
            samples, sample_rate,
            audition.graph.to_recipe(output_format="float64"),
            binary=arguments.binary)
        raw = os.path.join(work, f"{side}_{preset.preset_id}.wav")
        match.write_clip(raw, samples_out, sample_rate, 64, True)
        raw_paths.append(raw)
        out_paths.append(os.path.join(arguments.out,
                                      f"{side}_{preset.preset_id}.wav"))

    # MATCHED BEFORE COMPARING, and this is the difference between a useful
    # answer and a trivial one. Two presets that differ only in level are the
    # same preset with a fader move, and an unmatched comparison would report a
    # difference in every loudness-derived field and call them distinct.
    group = match.match_group(raw_paths, out_paths,
                              what=f"{renders[0][0].preset_id} vs "
                                   f"{renders[1][0].preset_id}")

    reports = []
    for path in out_paths:
        written = wavio.read_wav(path)
        reports.append(measure_array(written.samples, written.sample_rate))
    differences = compare_reports(reports[0], reports[1])

    print(f"\n{renders[0][0].preset_id}  vs  {renders[1][0].preset_id}")
    print(f"source: {arguments.source}, loudness-matched to "
          f"{group.spread_lu:.4f} LU before measuring\n")
    print(f"{'field':<44}{'left':>10}{'right':>10}{'delta':>10}  ")
    for difference in differences:
        left_text = "n/a" if difference.left is None else f"{difference.left:10.3f}"
        right_text = "n/a" if difference.right is None else f"{difference.right:10.3f}"
        delta_text = "n/a" if difference.delta is None else f"{difference.delta:10.3f}"
        flag = "  DIFFERS" if difference.differs else ""
        print(f"{difference.what:<44}{left_text:>10}{right_text:>10}"
              f"{delta_text:>10}{flag}")

    differing = [d for d in differences if d.differs]
    print()
    if differing:
        print(f"{len(differing)} field(s) differ by more than this tool's "
              f"reporting threshold.")
    else:
        print("NOTHING MEASURABLY DIFFERS between these two dossiers on this "
              "source.\nTwo presets that measure the same are one preset. "
              "Either the priors are\nnearly identical, or this source cannot "
              "show what separates them — try a\nsecond source before merging "
              "them.")
    print("\nThresholds here are this tool's reporting resolution, not "
          "just-noticeable\ndifferences. Nothing in this repository has been "
          "correlated against human\njudgement yet; that is what "
          "backend/listen/ is for.")
    return 0


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="author", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # `action="append"` rather than `nargs="*"`, which would swallow the
    # subcommand name as a denied alias and fail with an error about `new`.
    parser.add_argument("--denied", action="append", default=None,
                        help="override the alias denylist, repeatable "
                             "(default: derived from the routing table)")
    sub = parser.add_subparsers(dest="command", required=True)

    new = sub.add_parser("new", help="scaffold a dossier")
    new.add_argument("--out", "-o", default="-",
                     help="where to write it; - for stdout")
    new.add_argument("--preset-id")
    new.add_argument("--display-name")
    new.add_argument("--claim")
    new.add_argument("--confidence", default="high",
                     choices=[c.value for c in Confidence])
    new.add_argument("--citation-source")
    new.add_argument("--citation-page", type=int)
    new.add_argument("--citation-url")
    new.add_argument("--dsp-implication")
    new.add_argument("--review-note-claim",
                     help="why a disputed or unverified claim is doubtful")
    new.add_argument("--review-note", help="note for legal/editorial review")
    new.add_argument("--chain", nargs="*", help="chain_skeleton node types")
    new.add_argument("--target", action="append",
                     help="name=value+/-tolerance, repeatable")
    new.add_argument("--avoid", action="append", help="frequency=reason")
    new.add_argument("--negative", nargs="*", help="negative constraints")
    new.add_argument("--status", default="draft",
                     choices=[s.value for s in ReviewStatus])
    new.set_defaults(run=command_new)

    check = sub.add_parser("check", help="validate a dossier or a directory")
    check.add_argument("path", nargs="?", default=loader.PRESET_DIR)
    check.set_defaults(run=command_check)

    audition = sub.add_parser(
        "audition", help="render a loudness-matched A/B of the priors")
    audition.add_argument("dossier")
    audition.add_argument("source", help="a WAV file to render through")
    audition.add_argument("--out", "-o", default="audition")
    audition.add_argument("--seconds", type=float, default=20.0,
                          help="excerpt length; 0 for the whole file")
    audition.add_argument("--binary", default=render_mod.DEFAULT_BINARY)
    audition.set_defaults(run=command_audition)

    diff = sub.add_parser("diff", help="what measurably separates two dossiers")
    diff.add_argument("left")
    diff.add_argument("right")
    diff.add_argument("source")
    diff.add_argument("--out", "-o", default="diff")
    diff.add_argument("--seconds", type=float, default=20.0)
    diff.add_argument("--binary", default=render_mod.DEFAULT_BINARY)
    diff.set_defaults(run=command_diff)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    return int(arguments.run(arguments))


if __name__ == "__main__":
    sys.exit(main())
