"""The preset schema and the authoring tool, and what each guarantees.

THE ASSERTION THAT MATTERS MOST is that an unsourced factual claim cannot be
represented at all. Everything else here is ordinary schema testing; that one is
the reason the file exists.

The project's own first draft preset attributed Artificial Double Tracking to a
session recorded five months before ADT was invented. A person caught it. No
schema catches a wrong citation — but a schema can refuse a claim with NO
citation, which is the case where nobody had the chance to catch anything.

Sections 7 onwards cover `backend/knowledge/author.py`, and they are written to
a different standard than "the check exists", because `findings.md` records
three mutation rounds — 42 of 72, 81 of 139, 10 of 95 — whose dominant survivor
was a test probing a threshold from far outside it. So:

  * every `check` assertion names the OFFENDING CLAIM or field in the message,
    not merely that some error was produced. A tool that reported "invalid" for
    everything would pass the weaker version of every one of these.
  * the loudness-match assertion measures the pair BEFORE matching as well as
    after, so it can distinguish "the matcher worked" from "the two clips were
    already the same".
  * the refusal tests assert that NOTHING WAS WRITTEN, not just that the exit
    code was non-zero.

NO DOSSIER IN THIS FILE CONTAINS A FACTUAL CLAIM. The fixture is called
`example_not_a_real_preset`, its claims describe themselves as synthetic, and
its citation names a source that is stated to be fictional. A plausible-looking
example is how a fabricated citation gets copied into a real catalogue.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.audio import wavio  # noqa: E402
from backend.engine.render import DEFAULT_BINARY  # noqa: E402
from backend.knowledge import author, loader, schema  # noqa: E402
from backend.knowledge.schema import (  # noqa: E402
    AliasRoute, Citation, Confidence, Evidence, Preset, ReviewStatus,
)
from backend.listen import match  # noqa: E402
from backend.measure import loudness as loudness_mod  # noqa: E402


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


# ---------------------------------------------------------------------------
# The authoring tool.


DENIED = ["Beatles", "Fab Four"]

# THE ONLY DOSSIER IN THIS REPOSITORY, AND IT IS NOT A PRESET. Its id says so,
# its claim says so, and its citation names a source stated to be fictional. A
# realistic-looking example is how a fabricated citation ends up copied into a
# real catalogue by someone in a hurry.
SYNTHETIC_CITATION = {
    "source": "A Source That Does Not Exist, invented for this test",
    "page": 1,
}


def example_dossier(**overrides) -> dict:
    data = {
        "preset_id": "example_not_a_real_preset",
        "display_name": "Example, Not A Real Preset",
        "evidence": [{
            "claim": "This claim is synthetic and describes no real session",
            "citation": dict(SYNTHETIC_CITATION),
            "confidence": "high",
            "dsp_implication": "None whatsoever. It exists to exercise the "
                               "authoring tool.",
        }],
        "parameter_priors": {
            "chain_skeleton": ["parametric_eq", "compressor"],
            "targets": {
                "tilt_db_per_octave_re_pink": {"value": -1.5, "tolerance": 1.0},
            },
            "avoid_bands": [],
        },
        "negative_constraints": [],
        "legal_review_status": "draft",
        "review_note": "",
    }
    data.update(overrides)
    return data


def priors(**overrides) -> dict:
    base = example_dossier()["parameter_priors"]
    base.update(overrides)
    return base


def problem_with(problems, *needles, severity: str = "error"):
    """The first problem of `severity` whose text mentions everything named.

    SEARCHING THE WHOLE PROBLEM, where + what + fix, on purpose. A message that
    names the offending claim in `where` and the rule in `fix` is exactly as
    useful as one that puts both in `what`, and a test that insisted on the
    layout would fail on a rewording that improved it.
    """
    for problem in problems:
        blob = f"{problem.where} {problem.what} {problem.fix}".lower()
        if problem.severity == severity and all(n.lower() in blob for n in needles):
            return problem
    return None


def run_cli(*argv) -> tuple[int, str]:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        try:
            code = author.main(list(argv))
        except SystemExit as exit_error:      # argparse, and our own refusals
            code = int(exit_error.code or 0)
    return code, stream.getvalue()


def synthetic_source(seconds: float = 4.0, sample_rate: float = 48000.0,
                     seed: int = 20260804) -> tuple[np.ndarray, float]:
    """Pink noise under a plucked envelope, stereo.

    SYNTHESISED RATHER THAN LOADED. `audio/stems/` exists locally and is
    gitignored, so a suite that needed it would only pass where it was written.
    Pink because the tilt target has to have something to move and a white
    source would read as +3 dB/octave; plucked because a crest target needs
    crest, and because a stationary signal would leave the gating coverage at
    100% and hide nothing.
    """
    rng = np.random.default_rng(seed)
    frames = int(seconds * sample_rate)
    noise = np.fft.rfft(rng.standard_normal((2, frames)), axis=1)
    freqs = np.fft.rfftfreq(frames, 1.0 / sample_rate)
    shape = np.ones_like(freqs)
    shape[1:] = 1.0 / np.sqrt(freqs[1:])        # amplitude 1/sqrt(f) = pink
    pink = np.fft.irfft(noise * shape, n=frames, axis=1)
    pink /= np.max(np.abs(pink))

    time = np.arange(frames) / sample_rate
    envelope = np.exp(-((time * 4.0) % 1.0) * 6.0)
    return pink * envelope * 0.5, sample_rate


def write_source(directory: str) -> tuple[str, np.ndarray, float]:
    samples, sample_rate = synthetic_source()
    path = os.path.join(directory, "source.wav")
    wavio.write_wav(path, samples, sample_rate, 24)
    return path, samples, sample_rate


def test_check_translates_every_class_of_violation() -> None:
    print("[7] check names the offending claim, in words an author can act on")

    def look(data):
        return author.check_preset(data, denied=DENIED)

    preset, problems = look(example_dossier())
    R.check(preset is not None and not author.errors(problems),
            f"a clean dossier produces no errors, got "
            f"{[p.what for p in author.errors(problems)]}")

    # THE ONE THAT MATTERS. Not "an error was raised" — the message has to name
    # the claim, and it has to give BOTH ways out, because an author who is
    # told only "add a citation" will invent one rather than mark it disputed.
    uncited = example_dossier()
    uncited["evidence"][0].pop("citation")
    _, problems = look(uncited)
    found = problem_with(problems, "This claim is synthetic", "citation")
    R.check(found is not None, "an uncited confident claim is an error naming "
                               "the claim")
    R.check(found is not None and "unverified" in found.fix
            and "disputed" in found.fix and "citation.source" in found.fix,
            f"and states both ways out, got "
            f"{found.fix[:80] if found else None!r}")
    # THE CLAIM HAS TO BE IN THE LOCATION, not only in the message. Pydantic's
    # own text happens to repeat the claim for this one rule, so an assertion
    # that searched the whole problem would pass with the location reduced to
    # `evidence[0]` — and every OTHER complaint about that claim, which does
    # not repeat it, would then be unattributable in a forty-claim dossier.
    R.check(found is not None and "This claim is synthetic" in found.where,
            f"and the location itself identifies the claim, got "
            f"{found.where if found else None!r}")

    # The other half of the same rule, and it must name the claim too: a
    # dossier with forty claims produces this error with no other way to tell
    # which one it is about.
    unexplained = example_dossier()
    unexplained["evidence"][0] = {
        "claim": "A second synthetic claim, deliberately unexplained",
        "confidence": "unverified",
    }
    _, problems = look(unexplained)
    R.check(problem_with(problems, "A second synthetic claim", "review_note")
            is not None,
            "an unverified claim with no review_note names the claim")

    page_zero = example_dossier()
    page_zero["evidence"][0]["citation"]["page"] = 0
    _, problems = look(page_zero)
    found = problem_with(problems, "page")
    R.check(found is not None, "page 0 is refused with a message about pages")
    R.check(found is not None and "This claim is synthetic" in found.where,
            f"and this one carries the claim only because the location does, "
            f"got {found.where if found else None!r}")

    typo = example_dossier()
    typo["evidence"][0]["citaton"] = dict(SYNTHETIC_CITATION)
    _, problems = look(typo)
    found = problem_with(problems, "citaton")
    # `"citation" in fix` would pass on the list of known field names the fix
    # prints anyway, which is a second implementation of the same words.
    R.check(found is not None and "did you mean 'citation'" in found.fix,
            f"a misspelled field names the typo and suggests the real name, "
            f"got {found.fix[:60] if found else None!r}")

    shouted = example_dossier(preset_id="Era_UK_1965")
    _, problems = look(shouted)
    R.check(problem_with(problems, "preset_id", "lowercase") is not None,
            "a preset_id with capitals explains the pattern in words")

    no_tolerance = example_dossier(parameter_priors=priors(targets={
        "tilt_db_per_octave_re_pink": {"value": -1.5, "tolerance": 0.0}}))
    _, problems = look(no_tolerance)
    found = problem_with(problems, "tolerance")
    R.check(found is not None and "specification" in found.fix,
            "a zero tolerance says why a prior needs one")

    unknown_confidence = example_dossier()
    unknown_confidence["evidence"][0]["confidence"] = "quite_sure"
    _, problems = look(unknown_confidence)
    R.check(author.errors(problems), "an unknown confidence is a hard error")

    # Everything below parses, so it is reachable only because `check` knows
    # things Pydantic cannot express. `chain_skeleton` is `list[str]` and
    # `targets` is a free-form dict: every value here is valid JSON, valid
    # against the schema, and meaningless.
    bad_node = example_dossier(parameter_priors=priors(
        chain_skeleton=["parametric_eqq", "compressor"]))
    _, problems = look(bad_node)
    found = problem_with(problems, "parametric_eqq")
    R.check(found is not None and "parametric_eq'" in found.fix,
            f"an unknown node type names it and suggests the near miss, got "
            f"{found.fix[:60] if found else None!r}")

    master_node = example_dossier(parameter_priors=priors(
        chain_skeleton=["parametric_eq", "compressor", "true_peak_limiter"]))
    _, problems = look(master_node)
    R.check(problem_with(problems, "true_peak_limiter") is not None,
            "a master-only node in a preset's chain is refused")

    twice = example_dossier(parameter_priors=priors(
        chain_skeleton=["parametric_eq", "parametric_eq", "compressor"]))
    _, problems = look(twice)
    R.check(problem_with(problems, "parametric_eq", "twice") is not None
            or problem_with(problems, "parametric_eq", "2 times") is not None,
            "a node listed twice is refused: the skeleton has no ids, so "
            "nothing can say which one a target moves")

    unknown_target = example_dossier(parameter_priors=priors(
        targets={"warmth": {"value": 3.0, "tolerance": 1.0}}))
    _, problems = look(unknown_target)
    found = problem_with(problems, "warmth")
    R.check(found is not None
            and "tilt_db_per_octave_re_pink" in found.fix,
            "an uninterpretable target names it and lists the closed "
            "vocabulary")

    orphan = example_dossier(parameter_priors=priors(
        chain_skeleton=["parametric_eq"],
        targets={"crest_factor_db": {"value": 10.0, "tolerance": 2.0}}))
    _, problems = look(orphan)
    R.check(problem_with(problems, "crest_factor_db", "compressor") is not None,
            "a target whose node is missing from the chain names both")

    approved_thin = example_dossier(legal_review_status="approved", evidence=[{
        "claim": "A synthetic claim recorded as doubtful on purpose",
        "confidence": "disputed",
        "review_note": "It is not a real claim; nothing supports it.",
    }])
    _, problems = look(approved_thin)
    R.check(problem_with(problems, "usable_presets") is not None,
            "an approved dossier that can never be usable says so")

    # THE CHECK NOTHING ELSE IN THE SYSTEM DOES. `as_dossier` puts claims and
    # dsp_implications straight into the prompt, and the whole alias-routing
    # design rests on the model never having seen the name.
    named = example_dossier()
    named["evidence"][0]["dsp_implication"] = "Bright, the way the Beatles did."
    _, problems = look(named)
    R.check(problem_with(problems, "beatles") is not None,
            "a routed artist name in text the model is shown is an error")

    # And the other direction, which has to stay legal or the citations become
    # unwritable: `citation.source` is absent from `as_dossier`.
    in_source = example_dossier()
    in_source["evidence"][0]["citation"] = {
        "source": "A Fictional Book About The Beatles That Does Not Exist",
        "page": 1}
    _, problems = look(in_source)
    R.check(problem_with(problems, "beatles") is None,
            "but a name in a citation source is fine — it is never shown to "
            "the model, and a citation that cannot name its subject is useless")

    # Found by trying to test the check above: `_` is a word character, so the
    # word-boundary filter is blind to exactly the one field written in
    # snake_case — and preset_id is the field the model is offered as an enum.
    hidden = example_dossier(preset_id="era_beatles_1965_guitar")
    _, problems = look(hidden)
    R.check(problem_with(problems, "preset_id", "beatles") is not None,
            "an alias hidden inside the preset_id is caught, which the "
            "word-boundary filter cannot do")
    R.check(schema.name_filter_hits("era_beatles_1965_guitar", DENIED) == [],
            "and it needs its own check, because name_filter_hits genuinely "
            "does not see it")

    # Warnings must not block, or an author fixing them learns to ignore the
    # distinction and then ignores an error.
    thin = example_dossier()
    thin["evidence"][0]["dsp_implication"] = ""
    _, problems = look(thin)
    R.check(not author.errors(problems),
            "a claim with no dsp_implication is a warning, not an error")
    R.check(problem_with(problems, "trivia", severity="warning") is not None,
            "and it says why: a claim with no DSP implication is trivia")

    R.check(author.check_path("/nonexistent/dossier.json")[1],
            "a missing file is reported rather than raising")
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "broken.json")
        with open(path, "w") as handle:
            handle.write('{"preset_id": "x",}')
        _, problems = author.check_path(path)
        R.check(problem_with(problems, "json") is not None,
                "and so is a file that is not JSON at all")

        # Two dossiers, one id. The loader raises on this; `check` has to say
        # which two files, because the loader's message cannot.
        for name in ("one.json", "two.json"):
            with open(os.path.join(directory, name), "w") as handle:
                json.dump(example_dossier(), handle)
        os.remove(path)
        results = author.check_directory(directory, denied=DENIED)
        clash = [p for problems in results.values() for p in problems
                 if "already claimed" in p.what]
        R.check(len(clash) == 1 and "one.json" in clash[0].what,
                f"a duplicated preset_id names the file that claimed it first, "
                f"got {[p.what for p in clash]}")


def test_new_cannot_produce_a_dossier_that_fails_check() -> None:
    print("[8] new cannot produce a dossier that fails check")

    with tempfile.TemporaryDirectory() as directory:
        out = os.path.join(directory, "written.json")
        code, text = run_cli(
            "--denied", "Beatles",
            "new", "-o", out,
            "--preset-id", "example_not_a_real_preset",
            "--display-name", "Example, Not A Real Preset",
            "--claim", "This claim is synthetic and describes no real session",
            "--confidence", "high",
            "--citation-source", SYNTHETIC_CITATION["source"],
            "--citation-page", "1",
            "--dsp-implication", "None whatsoever.",
            "--chain", "parametric_eq", "compressor",
            "--target", "tilt_db_per_octave_re_pink=-1.5+/-1.0",
        )
        R.check(code == 0, f"a complete dossier is written, got {code}: {text}")
        R.check(os.path.exists(out), "and the file exists")

        _, problems = author.check_path(out, denied=DENIED)
        R.check(not author.errors(problems),
                f"what `new` wrote passes `check`, got "
                f"{[p.what for p in author.errors(problems)]}")
        R.check(len(loader.load_presets(directory)) == 1,
                "and the real loader loads it")

        with open(out) as handle:
            written = json.load(handle)
        R.check(Preset.model_validate(written).model_dump(mode="json") == written,
                "and it round-trips exactly: the file is the validated model, "
                "so every field the schema has is visible to the next author")

    # THE REFUSALS. Each of these is a dossier `new` could have written and
    # left the author to debug in a format they have not learned yet, and each
    # asserts that NOTHING WAS WRITTEN rather than only that the code was 1.
    def refuses(what: str, *flags: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = os.path.join(directory, "written.json")
            code, text = run_cli("--denied", "Beatles", "new", "-o", out,
                                 "--preset-id", "example_not_a_real_preset",
                                 "--display-name", "Example, Not A Real Preset",
                                 *flags)
            R.check(code == 1 and not os.path.exists(out),
                    f"{what}: refused and nothing written, got code {code}, "
                    f"exists {os.path.exists(out)}")
            R.check("would not pass `check`" in text,
                    f"{what}: and it says why it refused")

    refuses("a confident claim with no citation",
            "--claim", "A synthetic claim with no source", "--confidence", "high")
    refuses("an unverified claim with no review note",
            "--claim", "A synthetic claim nobody checked",
            "--confidence", "unverified")
    refuses("a target nothing can interpret",
            "--claim", "This claim is synthetic and describes no real session",
            "--citation-source", SYNTHETIC_CITATION["source"],
            "--citation-page", "1",
            "--chain", "parametric_eq",
            "--target", "warmth=3.0+/-1.0")
    refuses("a target whose node is not in the chain",
            "--claim", "This claim is synthetic and describes no real session",
            "--citation-source", SYNTHETIC_CITATION["source"],
            "--citation-page", "1",
            "--chain", "compressor",
            "--target", "tilt_db_per_octave_re_pink=-1.5+/-1.0")

    # And the escape hatch works, at the price the schema charges for it.
    with tempfile.TemporaryDirectory() as directory:
        out = os.path.join(directory, "doubtful.json")
        code, _ = run_cli(
            "new", "-o", out,
            "--preset-id", "example_not_a_real_preset",
            "--display-name", "Example, Not A Real Preset",
            "--claim", "A synthetic claim recorded as doubtful on purpose",
            "--confidence", "disputed",
            "--review-note-claim", "It is not a real claim; nothing supports it.")
        R.check(code == 0 and os.path.exists(out),
                "a disputed claim with a review_note is written")
        _, problems = author.check_path(out)
        R.check(not author.errors(problems), "and passes check")

    # A dossier `new` refuses to write because of a name is the one case where
    # the tool is protecting a guarantee rather than the schema.
    with tempfile.TemporaryDirectory() as directory:
        out = os.path.join(directory, "named.json")
        code, _ = run_cli("--denied", "Beatles", "new", "-o", out,
                          "--preset-id", "example_not_a_real_preset",
                          "--display-name", "Beatles Combo 1965")
        R.check(code == 1 and not os.path.exists(out),
                "a routed artist name in display_name is refused")


def test_audition_refuses_rather_than_rendering_something_unusable() -> None:
    print("[9] audition refuses rather than rendering something unusable")

    with tempfile.TemporaryDirectory() as directory:
        source, samples, sample_rate = write_source(directory)
        out = os.path.join(directory, "audition")

        broken = os.path.join(directory, "broken.json")
        data = example_dossier()
        data["evidence"][0].pop("citation")
        with open(broken, "w") as handle:
            json.dump(data, handle)

        code, text = run_cli("audition", broken, source, "-o", out)
        R.check(code == 1, f"an invalid dossier is refused, got {code}")
        R.check("refusing to render" in text, "and it says so")
        # THE PART THAT MATTERS. A refusal that still wrote audio would leave
        # the author judging a render the pipeline would never produce, and
        # nothing in the files would tell them which they were listening to.
        written = [] if not os.path.isdir(out) else \
            [n for n in os.listdir(out) if n.endswith(".wav")]
        R.check(written == [],
                f"and NOTHING was rendered, got {written}")
        R.check(problem_with(author.check_path(broken)[1],
                             "This claim is synthetic", "citation") is not None,
                "for the reason `check` would have given")

        # AND A DOSSIER THAT PARSES PERFECTLY AND IS STILL WRONG. The case
        # above fails Pydantic, so a refusal keyed only on "did it parse" would
        # catch it — this one is valid JSON, valid against the schema, and
        # names a target nothing can interpret. Without it the assertion above
        # is satisfied by a check that does not exist.
        parses = os.path.join(directory, "parses.json")
        with open(parses, "w") as handle:
            json.dump(example_dossier(parameter_priors=priors(
                targets={"warmth": {"value": 3.0, "tolerance": 1.0}})), handle)
        second = os.path.join(directory, "audition_two")
        code, text = run_cli("audition", parses, source, "-o", second)
        R.check(Preset.model_validate(json.load(open(parses))) is not None,
                "the second dossier is schema-valid")
        R.check(code == 1 and "warmth" in text,
                f"but audition refuses it by name, got {code}")
        rendered = [] if not os.path.isdir(second) else \
            [n for n in os.listdir(second) if n.endswith(".wav")]
        R.check(rendered == [], f"and rendered nothing, got {rendered}")

        # Everything below refuses without needing the engine at all, because
        # the graph is built from the priors and a measurement report before a
        # single sample is rendered.
        report = author.measure_array(samples, sample_rate)
        preset = Preset.model_validate(example_dossier())
        audition, problems = author.build_audition(preset, report)
        R.check(audition is not None and not author.errors(problems),
                f"a valid dossier builds a graph, got "
                f"{[p.what for p in problems]}")
        R.check(audition is not None
                and [n.type for n in audition.graph.nodes]
                == ["parametric_eq", "compressor"],
                "in the order the chain_skeleton gives")
        R.check(audition is not None
                and len(audition.graph.nodes[0].params["bands"]) == 2,
                "with the tilt as a mirrored shelf pair")
        R.check(audition is not None and audition.defaults
                and audition.defaults[0][0] == "compressor",
                "and the node no target speaks to is reported as rendered at "
                "schema defaults, because a default is not nothing")

        # A missing measurement is a refusal, never a substituted constant: a
        # fabricated number is indistinguishable from a measured one the moment
        # a move has been derived from it.
        blind = json.loads(json.dumps(report))
        blind["spectrum"]["tilt_db_per_octave_re_pink"] = None
        audition, problems = author.build_audition(preset, blind)
        R.check(audition is None
                and problem_with(problems, "tilt_db_per_octave_re_pink") is not None,
                "a delta target with no source measurement refuses rather than "
                "assuming a value for it")

        crest = report["dynamics"]["crest_factor_db"]
        upward = Preset.model_validate(example_dossier(
            parameter_priors=priors(targets={
                "crest_factor_db": {"value": crest + 6.0, "tolerance": 1.0}})))
        audition, problems = author.build_audition(upward, report)
        R.check(audition is None
                and problem_with(problems, "transient_designer") is not None,
                "a crest target above the source refuses and names what would "
                "be needed, rather than rendering a compressor that cannot "
                "answer it")

        # Self-contradiction, which only becomes visible against a source: the
        # sign of a delta move is a property of the audio, so the same prior
        # boosts one source and cuts another.
        source_tilt = report["spectrum"]["tilt_db_per_octave_re_pink"]
        contradictory = Preset.model_validate(example_dossier(
            parameter_priors=priors(
                chain_skeleton=["parametric_eq"],
                targets={"tilt_db_per_octave_re_pink":
                         {"value": source_tilt + 2.0, "tolerance": 0.5}},
                avoid_bands=[{"center_freq_hz": 8000.0,
                              "reason": "a synthetic band for this test"}])))
        audition, problems = author.build_audition(contradictory, report)
        R.check(audition is None
                and problem_with(problems, "8000", "avoid_bands") is not None,
                "a dossier that boosts where its own avoid_bands say not to is "
                "refused, naming the frequency")

        # And the same dossier asking for a CUT there is fine, because an
        # avoid_band reads as 'do not put energy here'. A guard that fired on
        # both directions would be one an author learns to ignore.
        allowed = Preset.model_validate(example_dossier(
            parameter_priors=priors(
                chain_skeleton=["parametric_eq"],
                targets={"tilt_db_per_octave_re_pink":
                         {"value": source_tilt - 2.0, "tolerance": 0.5}},
                avoid_bands=[{"center_freq_hz": 8000.0,
                              "reason": "a synthetic band for this test"}])))
        audition, problems = author.build_audition(allowed, report)
        R.check(audition is not None,
                f"but a cut in the same band is allowed, got "
                f"{[p.what for p in problems]}")

        # The compressor derivation, pinned as arithmetic rather than through a
        # render. A ratio derived from the wrong quantity would still produce a
        # compressor, still render, and still sound like compression — the
        # audition would simply be answering a different question than the one
        # the target asked.
        synthetic = {"dynamics": {"short_term_rms_percentiles":
                                  {"p90_dbfs": -20.0}},
                     "loudness": {"sample_peak_dbfs": -2.0}}
        params, why, refusal = author._compressor_for_crest(
            -9.0, synthetic, "crest_factor_db")
        # 18 dB of headroom above the threshold, 9 dB of crest to remove:
        # (P - T)(1 - 1/R) = 9 gives R = 2.
        R.check(refusal is None and params is not None
                and abs(params["ratio_to_1"] - 2.0) < 1e-9,
                f"9 dB of crest out of 18 dB of headroom is 2:1, got "
                f"{params['ratio_to_1'] if params else refusal}")
        R.check(params is not None and params["threshold_dbfs"] == -20.0,
                "with the threshold at the 90th-percentile short-term RMS")
        R.check(params is not None and params["makeup_gain_db"] == 0.0,
                "and no makeup gain, which would be a level difference in "
                "disguise inside a loudness-matched comparison")

        params, why, refusal = author._compressor_for_crest(
            -18.0, synthetic, "crest_factor_db")
        R.check(refusal is None and params is not None
                and params["ratio_to_1"] == 20.0 and "pinned" in why,
                f"asking for more crest than there is headroom pins the ratio "
                f"and says the target is not reachable, got "
                f"{params['ratio_to_1'] if params else refusal}")

        flat = {"dynamics": {"short_term_rms_percentiles": {"p90_dbfs": -3.0}},
                "loudness": {"sample_peak_dbfs": -2.5}}
        _, _, refusal = author._compressor_for_crest(-3.0, flat,
                                                     "crest_factor_db")
        R.check(refusal is not None,
                "and a source whose peak is barely above its RMS is refused "
                "rather than given a meaningless threshold")


def test_the_ab_pair_is_loudness_matched_on_disk() -> None:
    print("[10] the A/B pair is loudness-matched, measured off the written files")

    if not os.path.exists(DEFAULT_BINARY):
        print("  engine binary missing; skipping")
        return

    with tempfile.TemporaryDirectory() as directory:
        source, samples, sample_rate = write_source(directory)
        report = author.measure_array(samples, sample_rate)
        crest = report["dynamics"]["crest_factor_db"]

        # The targets are derived from what this source actually measures, so
        # the test cannot start passing because the synthesiser drifted.
        dossier = os.path.join(directory, "example.json")
        with open(dossier, "w") as handle:
            json.dump(example_dossier(parameter_priors=priors(targets={
                "tilt_db_per_octave_re_pink":
                    {"value": report["spectrum"]["tilt_db_per_octave_re_pink"]
                        - 2.5, "tolerance": 0.5},
                "crest_factor_db": {"value": crest - 4.0, "tolerance": 1.0},
            })), handle)

        out = os.path.join(directory, "audition")
        code, text = run_cli("audition", dossier, source, "-o", out,
                             "--seconds", "0")
        R.check(code == 0, f"the audition renders, got {code}: {text[-400:]}")

        a_path = os.path.join(out, "a_null.wav")
        b_path = os.path.join(out, "b_example_not_a_real_preset.wav")
        R.check(os.path.exists(a_path) and os.path.exists(b_path),
                "and writes both sides")

        a = wavio.read_wav(a_path)
        b = wavio.read_wav(b_path)
        after = [loudness_mod.integrated_lufs(clip.samples, clip.sample_rate)
                 for clip in (a, b)]
        spread = abs(after[0] - after[1])
        R.check(spread <= match.TOLERANCE_LU,
                f"the written pair is within the {match.TOLERANCE_LU} LU bar, "
                f"got {spread:.4f} LU from {[round(v, 3) for v in after]}")

        # THE HALF THAT MAKES THE ASSERTION ABOVE MEAN ANYTHING. Two clips that
        # were already the same loudness would satisfy it while the matcher did
        # nothing at all, so the unmatched intermediates are measured too — the
        # test can only pass because something moved them.
        raw = [os.path.join(out, "_intermediate", "null.wav"),
               os.path.join(out, "_intermediate", "example_not_a_real_preset.wav")]
        R.check(all(os.path.exists(p) for p in raw),
                "the pre-match intermediates were kept")
        if all(os.path.exists(p) for p in raw):
            before = []
            for path in raw:
                clip = wavio.read_wav(path)
                before.append(loudness_mod.integrated_lufs(clip.samples,
                                                           clip.sample_rate))
            gap = abs(before[0] - before[1])
            R.check(gap > match.TOLERANCE_LU,
                    f"and the priors DID change the loudness before matching, "
                    f"by {gap:.3f} LU — otherwise this test would pass with the "
                    f"matcher removed")
            R.check(gap > spread,
                    f"so the matcher closed a real gap: {gap:.3f} LU before, "
                    f"{spread:.4f} LU after")

        # The pair has to be the same excerpt, or an A/B compares two moments.
        R.check(a.frames == b.frames and a.sample_rate == b.sample_rate,
                f"both sides are the same length and rate, got {a.frames} vs "
                f"{b.frames}")
        R.check(a.bit_depth == match.BIT_DEPTH and not a.is_float,
                f"written at the delivery word length, got {a.format_name}")

        # ...and it has to be an actual comparison. A tool that rendered the
        # null twice would be perfectly loudness-matched and useless.
        difference = float(np.max(np.abs(a.samples - b.samples)))
        R.check(difference > 1e-3,
                f"the two sides are audibly different audio, peak difference "
                f"{difference:.5f}")

        for clip, name in ((a, "A"), (b, "B")):
            peak = loudness_mod.true_peak_dbtp(clip.samples, clip.sample_rate)
            R.check(peak <= match.CEILING_DBTP + 0.01,
                    f"{name} sits under the {match.CEILING_DBTP} dBTP ceiling, "
                    f"got {peak:.3f} dBTP")

        # THE A SIDE IS A NULL RENDER, AND THIS IS THE ASSERTION THAT SAYS SO.
        # The label in the printed output is a string; what matters is that the
        # bytes went through the engine. On this source the difference is not
        # subtle — the conversion path is transparent in the audible band and
        # pink noise has plenty of energy above it — so a tool that quietly
        # used the raw file as side A would be caught here and nowhere else.
        independent = author.render_mod.render_array(
            wavio.read_wav(source).samples, sample_rate,
            author.NULL_GRAPH.to_recipe(output_format="float64"))
        kept = wavio.read_wav(raw[0]).samples
        R.check(float(np.max(np.abs(kept - independent))) < 1e-9,
                "side A is exactly what the engine returns for an empty graph")
        raw_source = wavio.read_wav(source).samples
        R.check(float(np.max(np.abs(kept - raw_source))) > 1e-3,
                f"and that is NOT the raw file: they differ by "
                f"{float(np.max(np.abs(kept - raw_source))):.5f}, which is what "
                f"a comparison against the source would have attributed to the "
                f"preset")

        with open(os.path.join(out, "audition.json")) as handle:
            manifest = json.load(handle)

        # The move went the right way and got most of the way there. A sign
        # error or a span constant off by a factor of two would still render,
        # still be loudness-matched, and still print a table.
        tilt_move = [m for m in manifest["moves"]
                     if m["target"] == "tilt_db_per_octave_re_pink"][0]
        asked, began, ended = (tilt_move["requested"], tilt_move["source"],
                               tilt_move["achieved"])
        R.check(ended is not None and (ended - began) * (asked - began) > 0,
                f"the tilt moved towards the target, not away: {began:.2f} -> "
                f"{ended if ended is None else round(ended, 2)}, asked "
                f"{asked:.2f}")
        # THE GAIN OF THE MAPPING, not the distance remaining. "Got more than
        # half way" survives halving the span constant, and that is not a near
        # miss: on this source the shelf pair OVERSHOOTS by 38%, so halving it
        # lands about as far from the target as the correct version does, on
        # the other side. Only the ratio of delivered slope to requested slope
        # separates them — 1.38 correct, 0.69 halved, 2.77 doubled.
        delivered = (ended - began) / (asked - began) if ended is not None else 0.0
        R.check(1.0 <= delivered <= 2.0,
                f"the shelf pair delivers between 1.0 and 2.0 times the slope "
                f"change asked for on this source, got {delivered:.3f}. Whether "
                f"it over- or under-delivers depends on the source spectrum, "
                f"which is why the achieved figure is printed rather than the "
                f"requested one")
        # EXACT EQUALITY, because the real spread is around 2e-8 and any
        # tolerance loose enough to be comfortable is loose enough for a
        # hard-coded zero to pass. Both sides are the same arithmetic on the
        # same written files, so they are the same float or the manifest is
        # reporting something it did not measure.
        R.check(manifest["spread_lu"] <= match.TOLERANCE_LU
                and manifest["spread_lu"] == spread,
                f"the manifest records the spread this test measured "
                f"independently, got {manifest['spread_lu']!r} vs {spread!r}")
        R.check(manifest["defaults_rendered"] == [],
                f"and records that nothing was rendered at defaults, got "
                f"{manifest['defaults_rendered']}")

        # An excerpt is an excerpt. A `--seconds` that was ignored would render
        # the whole file and cost minutes on a real stem without saying so.
        short = os.path.join(directory, "short")
        code, _ = run_cli("audition", dossier, source, "-o", short,
                          "--seconds", "1")
        excerpt = wavio.read_wav(os.path.join(short, "a_null.wav"))
        R.check(code == 0 and abs(excerpt.duration_s - 1.0) < 0.01,
                f"--seconds trims the excerpt, got {excerpt.duration_s:.3f} s")


def test_diff_finds_two_presets_that_are_one() -> None:
    print("[11] diff finds two presets that are one preset")

    # The threshold half needs no engine, and it is the half that would
    # otherwise be probed from far outside: a delta at 99% of the bar must NOT
    # report a difference and one at 101% must.
    left = {"spectrum": {"tilt_db_per_octave_re_pink": 0.0,
                         "hf_rolloff_db_per_octave": -10.0,
                         "tilt_residual_db": 3.0, "third_octave_db": []},
            "dynamics": {"crest_factor_db": 12.0,
                         "peak_to_loudness_ratio_db": 14.0},
            "loudness": {"loudness_range_lu": 5.0},
            "stereo": {"correlation": 0.5, "mono_loss_db": 0.2}}

    def moved(field: str, section: str, by: float) -> dict:
        right = json.loads(json.dumps(left))
        right[section][field] = left[section][field] + by
        return right

    # WRITTEN OUT RATHER THAN READ FROM THE MODULE. Probing at
    # `author.DIFF_FIELDS[i][2] * 0.99` is self-referential: it moves with the
    # constant, so doubling every threshold passes unchanged. `findings.md` has
    # a section on exactly this. These numbers are the contract; if one of them
    # ought to move, this table is the place the decision gets made.
    EXPECTED = {
        ("spectrum", "tilt_db_per_octave_re_pink"): 0.3,
        ("spectrum", "hf_rolloff_db_per_octave"): 0.5,
        ("spectrum", "tilt_residual_db"): 0.5,
        ("dynamics", "crest_factor_db"): 0.5,
        ("dynamics", "peak_to_loudness_ratio_db"): 0.5,
        ("loudness", "loudness_range_lu"): 0.5,
        ("stereo", "correlation"): 0.05,
        ("stereo", "mono_loss_db"): 0.5,
    }
    declared = {(s, f): t for s, f, t in author.DIFF_FIELDS}
    R.check(declared == EXPECTED,
            f"the compared fields and their thresholds are what this test "
            f"pins, got {declared}")
    R.check(author.BAND_DIFFERENCE_DB == 1.0,
            f"and the band threshold is 1.0 dB, got {author.BAND_DIFFERENCE_DB}")

    for (section, field), threshold in EXPECTED.items():
        just_under = author.compare_reports(left, moved(field, section,
                                                        threshold * 0.99))
        just_over = author.compare_reports(left, moved(field, section,
                                                       threshold * 1.01))
        under = [d for d in just_under if d.differs]
        over = [d for d in just_over if d.differs]
        R.check(not under,
                f"{section}.{field}: a delta of {threshold * 0.99:.4g} does not "
                f"report a difference, got {[d.what for d in under]}")
        R.check(len(over) == 1 and field in over[0].what,
                f"{section}.{field}: a delta of {threshold * 1.01:.4g} does, "
                f"got {[d.what for d in over]}")

    # The band comparison is not part of that loop and is the strongest single
    # signal of the lot, so it is probed the same way.
    def banded(offset: float) -> tuple[dict, dict]:
        a = json.loads(json.dumps(left))
        b = json.loads(json.dumps(left))
        a["spectrum"]["third_octave_db"] = [
            {"centre_hz": 100.0, "level_db": -20.0},
            {"centre_hz": 1000.0, "level_db": -18.0}]
        b["spectrum"]["third_octave_db"] = [
            {"centre_hz": 100.0, "level_db": -20.0},
            {"centre_hz": 1000.0, "level_db": -18.0 + offset}]
        return a, b

    under = [d for d in author.compare_reports(*banded(0.99)) if d.differs]
    over = [d for d in author.compare_reports(*banded(1.01)) if d.differs]
    R.check(not under,
            f"a band 0.99 dB apart is not a difference, got "
            f"{[d.what for d in under]}")
    R.check(len(over) == 1 and "1000 Hz" in over[0].what,
            f"one 1.01 dB apart is, and it names the band, got "
            f"{[d.what for d in over]}")

    # A measurement that exists on one side and not the other is a difference,
    # and it must not be reported as a delta of zero — which is what any
    # arithmetic on a missing value produces if a fallback is substituted.
    gone = json.loads(json.dumps(left))
    gone["stereo"]["mono_loss_db"] = None
    differences = author.compare_reports(left, gone)
    missing = [d for d in differences if d.what.endswith("mono_loss_db")]
    R.check(len(missing) == 1 and missing[0].differs
            and missing[0].delta is None,
            f"a measurement present on one side only differs, with no delta, "
            f"got {missing}")

    if not os.path.exists(DEFAULT_BINARY):
        print("  engine binary missing; skipping the rendered half")
        return

    with tempfile.TemporaryDirectory() as directory:
        source, samples, sample_rate = write_source(directory)
        report = author.measure_array(samples, sample_rate)
        tilt = report["spectrum"]["tilt_db_per_octave_re_pink"]

        def dossier(name: str, value: float, filename: str = "") -> str:
            path = os.path.join(directory, f"{filename or name}.json")
            with open(path, "w") as handle:
                json.dump(example_dossier(
                    preset_id=name,
                    parameter_priors=priors(
                        chain_skeleton=["parametric_eq"],
                        targets={"tilt_db_per_octave_re_pink":
                                 {"value": value, "tolerance": 1.0}})), handle)
            return path

        twin_a = dossier("example_not_a_real_preset_one", tilt - 2.0)
        twin_b = dossier("example_not_a_real_preset_two", tilt - 2.0)
        code, text = run_cli("diff", twin_a, twin_b, source,
                             "-o", os.path.join(directory, "same"),
                             "--seconds", "0")
        R.check(code == 0, f"diff runs, got {code}: {text[-400:]}")
        R.check("NOTHING MEASURABLY DIFFERS" in text,
                "two dossiers with the same priors are one preset, and it says "
                "so in those terms")

        other = dossier("example_not_a_real_preset_three", tilt + 2.0)
        code, text = run_cli("diff", twin_a, other, source,
                             "-o", os.path.join(directory, "different"),
                             "--seconds", "0")
        R.check(code == 0 and "NOTHING MEASURABLY DIFFERS" not in text,
                "and two with opposite tilts are not")
        R.check("tilt_db_per_octave_re_pink" in text
                and "DIFFERS" in text.split("tilt_db_per_octave_re_pink")[1]
                .split("\n")[0],
                "with the tilt named as the field that separates them")

        # TWO DOSSIERS SHARING A preset_id, which is one of the ways a
        # duplicate gets noticed and the case where naming the render files
        # after the preset would have one overwrite the other and report a
        # difference of exactly zero — the most confident possible wrong answer
        # from this command.
        clash_a = dossier("example_not_a_real_preset_clash", tilt - 2.0,
                          filename="clash_left")
        clash_b = dossier("example_not_a_real_preset_clash", tilt + 2.0,
                          filename="clash_right")
        code, text = run_cli("diff", clash_a, clash_b, source,
                             "-o", os.path.join(directory, "clash"),
                             "--seconds", "0")
        R.check(code == 0 and "NOTHING MEASURABLY DIFFERS" not in text,
                "two dossiers sharing a preset_id are still compared as two "
                "renders, not silently collapsed into one")

        # Level is not a preset. Both renders are loudness-matched before being
        # measured, so a difference reported here is a difference in the audio
        # rather than in the fader — and the matched files are on disk to check.
        matched = sorted(n for n in os.listdir(
            os.path.join(directory, "different")) if n.endswith(".wav"))
        R.check(len(matched) == 2, f"both renders are kept, got {matched}")
        if len(matched) == 2:
            levels = []
            for name in matched:
                clip = wavio.read_wav(os.path.join(directory, "different", name))
                levels.append(loudness_mod.integrated_lufs(clip.samples,
                                                           clip.sample_rate))
            R.check(abs(levels[0] - levels[1]) <= match.TOLERANCE_LU,
                    f"and they were matched before being compared, got "
                    f"{abs(levels[0] - levels[1]):.4f} LU apart")


if __name__ == "__main__":
    print("Audionova preset knowledge base\n")
    test_an_unsourced_claim_cannot_be_represented()
    test_review_gates_use_not_storage()
    test_artist_names_never_reach_the_model()
    test_the_catalogue_is_empty_or_fully_cited()
    test_a_broken_dossier_is_never_skipped()
    test_priors_are_ranges_not_specifications()
    test_check_translates_every_class_of_violation()
    test_new_cannot_produce_a_dossier_that_fails_check()
    test_audition_refuses_rather_than_rendering_something_unusable()
    test_the_ab_pair_is_loudness_matched_on_disk()
    test_diff_finds_two_presets_that_are_one()
    print(f"\n{R.checks} checks, {R.failures} failure(s)")
    sys.exit(1 if R.failures else 0)
