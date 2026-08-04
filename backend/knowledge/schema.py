"""Preset dossiers: evidence, priors, and the rules that keep them honest.

A PRESET IS NOT AN EFFECT GRAPH. It is evidence, intent and parameter priors,
and the planner composes it against what it measured. A preset that hard-codes
"boost 3 kHz by 4 dB" is wrong for a source that already has a 3 kHz resonance,
and the whole reason this is a dossier rather than a chain is so the model can
decline a move the measurements do not support.

THE SCHEMA'S REAL JOB IS TO MAKE AN UNSOURCED CLAIM UNREPRESENTABLE.

The plan's own first draft preset contained a factual error: it attributed
Artificial Double Tracking to a session recorded 16 October 1965, when ADT was
invented by Ken Townsend in April 1966 for the Revolver sessions. That error was
caught by a person who knew the history. Nothing in a schema catches that.

What a schema CAN do is make the error visible before it ships — by refusing to
represent a claim that has no citation, and by refusing to let a preset be used
until someone has said, in the file, that they checked it. So:

  * every `claim` needs a `citation` with a source and a page, OR an explicit
    confidence of `disputed` or `unverified` with a `review_note` saying why
  * `legal_review_status` gates USE, not storage. An unreviewed preset can exist
    and be worked on; `usable_presets()` will not return it
  * `preset_id` is the only thing that reaches the effect graph, and it is a
    closed enum, so constrained decoding cannot put an artist's name in a chain
    even if the user typed one

WHY THERE ARE NO PRESETS IN THIS DIRECTORY. Authoring them is 4-6 person-weeks
by the plan's own estimate and it is not an engineering task: it needs the
books, and a claim about a 1965 session cannot be verified by the person writing
the loader. Shipping a plausible-sounding dossier would be exactly the failure
this file exists to prevent — a confidently wrong answer, which is worse than no
answer, because the user cannot tell it from a right one.

So the machinery is here and the catalogue is empty, deliberately. A test
asserts the catalogue is either empty or fully cited; it must never become
"mostly cited".
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Confidence(str, Enum):
    """How much weight a claim carries.

    `disputed` and `unverified` are not failure states — they are how a
    half-known fact gets recorded WITHOUT being quietly promoted to a fact. A
    dossier that can only express certainty forces the author to either drop
    what they suspect or assert it, and both lose information.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    DISPUTED = "disputed"
    UNVERIFIED = "unverified"


class ReviewStatus(str, Enum):
    DRAFT = "draft"
    NEEDS_LEGAL = "needs_legal"
    APPROVED = "approved"
    REJECTED = "rejected"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Citation(_Strict):
    source: str = Field(min_length=4, max_length=300,
                        description="Author, title. Enough to find it again.")
    page: int | None = Field(default=None, ge=1,
                             description="Page number. A citation to a whole "
                                         "book is not a citation.")
    url: str | None = Field(default=None, max_length=500)


class Evidence(_Strict):
    """One factual claim about how a record was made, and where it comes from."""

    claim: str = Field(min_length=4, max_length=400)
    citation: Citation | None = None
    confidence: Confidence
    dsp_implication: str = Field(
        default="", max_length=600,
        description="What this claim means for processing, in measurable "
                    "terms. A claim with no DSP implication is trivia.")
    review_note: str = Field(default="", max_length=600)

    @model_validator(mode="after")
    def _sourced_or_declared(self) -> "Evidence":
        """A confident claim needs a citation. An uncited one must say so.

        THIS IS THE WHOLE POINT OF THE FILE. The failure mode being prevented is
        not a missing citation — it is a claim that reads as established fact,
        was actually someone's recollection, and reaches a paying engineer as
        the reason their guitar sounds the way it does.
        """
        confident = self.confidence in (Confidence.HIGH, Confidence.MEDIUM,
                                        Confidence.LOW)
        if confident and self.citation is None:
            raise ValueError(
                f"claim {self.claim!r} has confidence {self.confidence.value} "
                f"and no citation. Either cite it, or mark it 'unverified' or "
                f"'disputed' and say why in review_note.")
        if not confident and not self.review_note.strip():
            raise ValueError(
                f"claim {self.claim!r} is {self.confidence.value} and has no "
                f"review_note. An unverified claim without a reason to doubt it "
                f"is indistinguishable from one nobody checked.")
        return self


class TargetRange(_Strict):
    """A measured target with a tolerance, never a bare number.

    A prior without a tolerance reads as a specification, and the planner is
    supposed to compose these against a source rather than hit them.
    """

    value: float
    tolerance: float = Field(gt=0.0)


class AvoidBand(_Strict):
    center_freq_hz: float = Field(ge=20.0, le=20000.0)
    reason: str = Field(min_length=8, max_length=300)


class ParameterPriors(_Strict):
    chain_skeleton: list[str] = Field(default_factory=list, max_length=16)
    targets: dict[str, TargetRange] = Field(default_factory=dict)
    avoid_bands: list[AvoidBand] = Field(default_factory=list, max_length=12)


class Preset(_Strict):
    preset_id: str = Field(pattern=r"^[a-z0-9_]{4,60}$")
    display_name: str = Field(min_length=3, max_length=80)
    evidence: list[Evidence] = Field(default_factory=list, max_length=40)
    parameter_priors: ParameterPriors = Field(default_factory=ParameterPriors)
    negative_constraints: list[str] = Field(default_factory=list, max_length=20)
    legal_review_status: ReviewStatus = ReviewStatus.DRAFT
    review_note: str = Field(default="", max_length=1000)

    @property
    def usable(self) -> bool:
        """Approved, and carrying at least one claim that survived review.

        Approval alone is not enough. A preset with no evidence is a set of
        numbers somebody liked, which may be perfectly good processing but is
        not what this system is for and must not be presented as heritage.
        """
        if self.legal_review_status is not ReviewStatus.APPROVED:
            return False
        return any(e.confidence in (Confidence.HIGH, Confidence.MEDIUM)
                   for e in self.evidence)

    @property
    def disputed_claims(self) -> list[Evidence]:
        return [e for e in self.evidence
                if e.confidence in (Confidence.DISPUTED, Confidence.UNVERIFIED)]


class AliasRoute(_Strict):
    """What a user typed, and which dossier it reaches.

    USERS MAY TYPE ANY ARTIST NAME. Our catalogue, interface and marketing
    contain none. The alias lives here, in a routing table that is matched
    BEFORE the model is called, and only the `preset_id` crosses into the
    prompt — so an artist's name is never in anything the model can echo back
    into a chain.
    """

    aliases: list[str] = Field(min_length=1, max_length=40)
    preset_id: str = Field(pattern=r"^[a-z0-9_]{4,60}$")


# Names that must never appear in generated free text. Deliberately a filter on
# OUTPUT rather than a request in the prompt: asking a model not to say
# something is a preference, and this needs to be a guarantee.
#
# Seeded from the routing table at load time rather than hard-coded, so the
# denylist cannot drift from the aliases that are actually routed.
def name_filter_hits(text: str, denied: list[str]) -> list[str]:
    """Denied names appearing in free text, matched on word boundaries.

    Word boundaries matter in both directions. Substring matching would flag
    "beatles" inside an ordinary word and, worse, would let an author conclude
    the filter is noisy and loosen it. Matching too loosely is how a guard gets
    switched off.
    """
    hits = []
    for name in denied:
        if not name.strip():
            continue
        if re.search(rf"\b{re.escape(name.strip())}\b", text, re.IGNORECASE):
            hits.append(name)
    return sorted(set(hits))


def usable_presets(presets: list[Preset]) -> list[Preset]:
    """The ones allowed to reach a planner. Everything else exists but waits."""
    return [p for p in presets if p.usable]


def preset_id_enum(presets: list[Preset]) -> list[str]:
    """The closed set for the graph schema.

    Closed on the USABLE presets, not on every file in the directory. A draft
    dossier being worked on must not become selectable because it happened to
    parse.
    """
    return sorted(p.preset_id for p in usable_presets(presets))


def as_dossier(preset: Preset) -> dict[str, Any]:
    """What the planner is shown.

    Disputed and unverified claims are INCLUDED, with their status, rather than
    filtered out. The model is explicitly told elsewhere to disregard estimates
    below 0.4 confidence; the same principle applies here, and hiding a doubtful
    claim would leave the model to rediscover it as a confident one.

    `legal_review_status` and `review_note` are not shown. They are about the
    company's process, not about the audio, and every token in a prompt should
    be something the model can act on.
    """
    return {
        "preset_id": preset.preset_id,
        "display_name": preset.display_name,
        "evidence": [
            {"claim": e.claim, "confidence": e.confidence.value,
             "dsp_implication": e.dsp_implication,
             **({"cited": True} if e.citation else {"cited": False})}
            for e in preset.evidence
        ],
        "parameter_priors": preset.parameter_priors.model_dump(mode="json"),
        "negative_constraints": preset.negative_constraints,
    }
