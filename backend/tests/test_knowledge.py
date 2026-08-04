"""The preset schema, and the guarantees it is supposed to provide.

THE ASSERTION THAT MATTERS MOST is that an unsourced factual claim cannot be
represented at all. Everything else here is ordinary schema testing; that one is
the reason the file exists.

The project's own first draft preset attributed Artificial Double Tracking to a
session recorded five months before ADT was invented. A person caught it. No
schema catches a wrong citation — but a schema can refuse a claim with NO
citation, which is the case where nobody had the chance to catch anything.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.knowledge import loader, schema  # noqa: E402
from backend.knowledge.schema import (  # noqa: E402
    AliasRoute, Citation, Confidence, Evidence, Preset, ReviewStatus,
)


class Results:
    def __init__(self) -> None:
        self.checks = 0
        self.failures = 0

    def check(self, ok: bool, what: str) -> None:
        self.checks += 1
        if not ok:
            self.failures += 1
            print(f"\n  FAIL  {what}")


R = Results()

CITED = Evidence(
    claim="Vox AC30 combo amplifier",
    citation=Citation(source="Ryan & Kehew, Recording The Beatles", page=344),
    confidence=Confidence.HIGH,
    dsp_implication="Class-A EL84: ~2-3 kHz presence peak, soft even-order "
                    "compression, 12-inch Alnico rolling off above ~5 kHz.",
)


def preset(**overrides) -> Preset:
    data = {
        "preset_id": "era_uk_1965_beatgroup_guitar",
        "display_name": "British Invasion Combo '65",
        "evidence": [CITED],
        "legal_review_status": ReviewStatus.APPROVED,
    }
    data.update(overrides)
    return Preset.model_validate(data)


def test_an_unsourced_claim_cannot_be_represented() -> None:
    print("[1] an unsourced claim cannot be represented")

    # A confident claim with no citation must not construct.
    for level in (Confidence.HIGH, Confidence.MEDIUM, Confidence.LOW):
        try:
            Evidence(claim="Recorded through a Fairchild 660",
                     confidence=level, dsp_implication="heavy compression")
            R.check(False, f"a {level.value}-confidence claim needs a citation")
        except Exception:
            R.check(True, f"a {level.value}-confidence claim needs a citation")

    # The escape hatch exists, and it costs a stated reason. This is the ADT
    # case: known to be doubtful, recorded as doubtful, with the reason in the
    # file where the next author will read it.
    disputed = Evidence(
        claim="Artificial Double Tracking",
        confidence=Confidence.DISPUTED,
        review_note="ADT was invented by Ken Townsend in April 1966 for the "
                    "Revolver sessions. This preset covers 1965. Do not enable.",
    )
    R.check(disputed.citation is None, "a disputed claim may be uncited")

    # But it cannot be uncited AND unexplained, which is the state that reads as
    # fact to everyone downstream.
    try:
        Evidence(claim="Recorded at half speed", confidence=Confidence.UNVERIFIED)
        R.check(False, "an unverified claim with no review_note is refused")
    except Exception:
        R.check(True, "an unverified claim with no review_note is refused")

    # A citation to a whole book is not a citation, but a URL-only source is
    # allowed — some sources genuinely have no pages.
    try:
        Citation(source="a book")
        R.check(True, "a source without a page is allowed")
    except Exception:
        R.check(False, "a source without a page is allowed")
    try:
        Citation(source="x", page=0)
        R.check(False, "page 0 is refused")
    except Exception:
        R.check(True, "page 0 is refused")


def test_review_gates_use_not_storage() -> None:
    print("[2] review gates use, not storage")

    draft = preset(legal_review_status=ReviewStatus.DRAFT)
    R.check(not draft.usable, "a draft preset is not usable")
    R.check(draft.preset_id, "but it still exists and can be worked on")

    for status in (ReviewStatus.NEEDS_LEGAL, ReviewStatus.REJECTED):
        R.check(not preset(legal_review_status=status).usable,
                f"a {status.value} preset is not usable")

    R.check(preset().usable, "an approved, cited preset is usable")

    # Approval is not enough on its own. A preset with no surviving evidence is
    # a set of numbers somebody liked — possibly good processing, but it must
    # not be presented as heritage.
    R.check(not preset(evidence=[]).usable,
            "an approved preset with no evidence is still not usable")
    only_doubtful = preset(evidence=[Evidence(
        claim="Artificial Double Tracking", confidence=Confidence.DISPUTED,
        review_note="invented in 1966; this preset is 1965")])
    R.check(not only_doubtful.usable,
            "nor one whose every claim is disputed")

    R.check([p.preset_id for p in schema.usable_presets([draft, preset()])]
            == ["era_uk_1965_beatgroup_guitar"],
            "usable_presets returns only the approved ones")
    R.check(schema.preset_id_enum([draft]) == [],
            "and the graph schema's enum is closed on usable presets only")


def test_artist_names_never_reach_the_model() -> None:
    print("[3] an artist name never reaches the model")

    routes = [AliasRoute(aliases=["Beatles", "The Beatles", "Fab Four"],
                         preset_id="era_uk_1965_beatgroup_guitar")]

    R.check(loader.resolve("the beatles", routes) == "era_uk_1965_beatgroup_guitar",
            "an alias resolves, case- and space-insensitively")
    R.check(loader.resolve("  Fab   Four ", routes) == "era_uk_1965_beatgroup_guitar",
            "and whitespace is normalised")
    R.check(loader.resolve("beatle", routes) is None,
            "a near-miss does NOT resolve: fuzzy-matching an artist name routes "
            "people to a preset about a different band")
    R.check(loader.resolve("", routes) is None, "and an empty query resolves to nothing")

    # THE DOSSIER THE MODEL SEES CARRIES NO ALIAS. This is the property that
    # makes the closed enum worth having: even asked directly, the model has
    # never been shown the name.
    dossier = json.dumps(schema.as_dossier(preset()))
    for name in ("Beatles", "Fab Four"):
        R.check(name.lower() not in dossier.lower(),
                f"the dossier does not contain {name!r}")

    # The denylist is DERIVED from the routes, so an alias added for discovery
    # cannot be missing from the filter that stops it being echoed.
    denied = loader.denied_names(routes)
    R.check(denied == ["Beatles", "Fab Four", "The Beatles"],
            f"the denylist is derived from the routing table, got {denied}")

    # Both "Beatles" and "The Beatles" are denied names and both genuinely
    # occur in this sentence, so the filter reports both. My first version of
    # this assertion demanded exactly ["Beatles"] and was simply wrong about
    # what the text contains — worth keeping, because a filter that reported
    # fewer hits than there are would be the actually dangerous outcome and an
    # over-tight test would have pushed it that way.
    hits = schema.name_filter_hits(
        "Warmed the guitar the way the Beatles did on that record.", denied)
    R.check("Beatles" in hits,
            f"the filter catches a name in free text, got {hits}")
    R.check(set(hits) <= set(denied),
            f"and reports only names that are actually denied, got {hits}")
    R.check(schema.name_filter_hits("The beetles ate my lettuce.", denied) == [],
            "and does not fire on a word that merely looks similar")
    # Word boundaries in the other direction: a substring match would flag this,
    # and an author who saw that would loosen the filter, which is how a guard
    # gets switched off.
    R.check(schema.name_filter_hits("beatlesque phrasing", denied) == [],
            "nor inside a longer word")


def test_the_catalogue_is_empty_or_fully_cited() -> None:
    print("[4] the catalogue is empty or fully cited")

    presets = loader.load_presets()
    R.check(isinstance(presets, list),
            f"the loader handles an absent catalogue, got {len(presets)} presets")

    # THE INVARIANT THAT MUST HOLD FOREVER. Zero presets is fine. Every preset
    # cited is fine. "Most of them cited" is the state where a user cannot tell
    # which answers are grounded, and it must never be reachable by drift —
    # so it is asserted rather than intended.
    for p in presets:
        for e in p.evidence:
            grounded = e.citation is not None or e.review_note.strip()
            R.check(grounded,
                    f"{p.preset_id}: claim {e.claim!r} is neither cited nor "
                    f"explained")
        if p.legal_review_status is ReviewStatus.APPROVED:
            R.check(any(e.citation for e in p.evidence),
                    f"{p.preset_id} is approved and must carry a citation")

    R.check(loader.denied_names() == loader.denied_names(),
            "the denylist is stable across calls")


def test_a_broken_dossier_is_never_skipped() -> None:
    print("[5] a broken dossier is never skipped")

    with tempfile.TemporaryDirectory() as directory:
        with open(os.path.join(directory, "good.json"), "w") as handle:
            json.dump(preset().model_dump(mode="json"), handle)
        R.check(len(loader.load_presets(directory)) == 1, "a good dossier loads")

        with open(os.path.join(directory, "broken.json"), "w") as handle:
            handle.write("{not json")
        try:
            loader.load_presets(directory)
            R.check(False, "a malformed dossier RAISES rather than being skipped")
        except loader.KnowledgeError:
            R.check(True, "a malformed dossier RAISES rather than being skipped")

        os.remove(os.path.join(directory, "broken.json"))
        # A claim that lost its citation must fail the same way, not load with
        # the claim intact and the source gone.
        bad = preset().model_dump(mode="json")
        bad["evidence"][0].pop("citation")
        with open(os.path.join(directory, "uncited.json"), "w") as handle:
            json.dump(bad, handle)
        try:
            loader.load_presets(directory)
            R.check(False, "so does a dossier whose claim lost its citation")
        except loader.KnowledgeError:
            R.check(True, "so does a dossier whose claim lost its citation")

        os.remove(os.path.join(directory, "uncited.json"))
        duplicate = preset().model_dump(mode="json")
        with open(os.path.join(directory, "again.json"), "w") as handle:
            json.dump(duplicate, handle)
        try:
            loader.load_presets(directory)
            R.check(False, "and two dossiers sharing a preset_id are refused")
        except loader.KnowledgeError:
            R.check(True, "and two dossiers sharing a preset_id are refused")


def test_priors_are_ranges_not_specifications() -> None:
    print("[6] priors are ranges, not specifications")

    from backend.knowledge.schema import AvoidBand, ParameterPriors, TargetRange

    priors = ParameterPriors(
        chain_skeleton=["parametric_eq", "drive", "compressor"],
        targets={"tilt_db_per_octave": TargetRange(value=-1.5, tolerance=1.0)},
        avoid_bands=[AvoidBand(center_freq_hz=8000.0,
                               reason="period recordings have little energy "
                                      "above 8 kHz; air boosts break it")],
    )
    R.check(priors.targets["tilt_db_per_octave"].tolerance == 1.0,
            "a target carries a tolerance")
    try:
        TargetRange(value=-1.5, tolerance=0.0)
        R.check(False, "a zero tolerance is refused: that is a specification, "
                       "and the planner is meant to compose priors against a "
                       "source rather than hit them")
    except Exception:
        R.check(True, "a zero tolerance is refused")

    try:
        AvoidBand(center_freq_hz=8000.0, reason="no")
        R.check(False, "an avoid_band needs a real reason")
    except Exception:
        R.check(True, "an avoid_band needs a real reason")

    dossier = schema.as_dossier(preset())
    R.check("legal_review_status" not in json.dumps(dossier),
            "the dossier shown to the model omits the review status, which is "
            "about our process rather than about the audio")
    R.check(dossier["evidence"][0]["cited"] is True,
            "but it does say whether a claim was cited")


if __name__ == "__main__":
    print("Audionova preset knowledge base\n")
    test_an_unsourced_claim_cannot_be_represented()
    test_review_gates_use_not_storage()
    test_artist_names_never_reach_the_model()
    test_the_catalogue_is_empty_or_fully_cited()
    test_a_broken_dossier_is_never_skipped()
    test_priors_are_ranges_not_specifications()
    print(f"\n{R.checks} checks, {R.failures} failure(s)")
    sys.exit(1 if R.failures else 0)
