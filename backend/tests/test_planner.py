"""The planner scaffolding: ladder, glossary, prompt cache boundary, pipeline.

Everything here runs offline. The model is behind a two-method protocol and every
test drives a fake, which means the paths a live test would exercise LEAST are
the ones covered best: malformed JSON, two consecutive budget violations, an
unparseable final retry, an exhausted loop with and without a fallback.

The single most valuable assertion in the file is the cache-boundary one. A
per-request value leaking into the system prefix does not fail — the request
succeeds, the answer is fine, and the bill multiplies silently. It is the kind of
bug that is only ever found by looking for it.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.plan import glossary, ladder, pipeline  # noqa: E402
from backend.plan import prompt as prompts  # noqa: E402
from backend.plan.ladder import Patch  # noqa: E402
from backend.plan.validate import validate  # noqa: E402
from backend.schemas.graph import EffectGraph, GraphNode  # noqa: E402


class Results:
    def __init__(self) -> None:
        self.checks = 0
        self.failures = 0

    def check(self, ok: bool, what: str) -> None:
        self.checks += 1
        if not ok:
            self.failures += 1
            print(f"\n  FAIL  {what}")

    def near(self, got: float, want: float, tol: float, what: str) -> None:
        self.check(got is not None and abs(got - want) <= tol,
                   f"{what}: got {got}, want {want} +/- {tol}")


R = Results()

REPORT = {
    "source": {"channels": 2, "sample_rate_hz": 48000.0, "duration_s": 20.0,
               "gated_coverage_percent": 88.0},
    "loudness": {"integrated_lufs": -19.4, "true_peak_dbtp": -3.2,
                 "loudness_range_lu": 6.1},
    "dynamics": {"crest_factor_db": 14.2, "peak_to_loudness_ratio_db": 16.2,
                 "short_term_rms_percentiles": {"p10_dbfs": -32.0, "p50_dbfs": -24.0,
                                                "p90_dbfs": -19.0, "p95_dbfs": -18.0}},
    "spectrum": {"tilt_db_per_octave_re_pink": -1.8, "tilt_residual_db": 3.1,
                 "bandwidth_limit_hz": None, "bandwidth_limit_confidence": 0.0,
                 "hf_rolloff_db_per_octave": -8.4,
                 "third_octave_db": [{"centre_hz": 1000.0, "level_db": -28.0}],
                 "erb_db": [{"centre_hz": 1050.0, "level_db": -28.0}]},
    "stereo": {"type": "conventional_stereo", "correlation": 0.72,
               "mono_loss_db": 0.1, "per_band": []},
    "hygiene": {"clipping": {"clipped_sample_percent": 0.0},
                "noise_floor_dbfs": -78.0, "noise_floor_confidence": 0.9,
                "dead_channels": [], "non_finite_samples": 0},
    "warnings": [],
}

GOOD_GRAPH = {
    "schema_version": "effect_graph_v1",
    "nodes": [
        {"id": "shape", "type": "parametric_eq", "rationale": "tilt is -1.8",
         "params": {"bands": [
             {"filter_type": "highpass", "center_freq_hz": 85.0,
              "gain_db": 0.0, "q_factor": 0.707},
             {"filter_type": "highshelf", "center_freq_hz": 6000.0,
              "gain_db": 3.0, "q_factor": 0.707},
         ]}},
        {"id": "comp", "type": "compressor", "rationale": "LRA is 6.1 LU",
         "params": {"threshold_dbfs": -24.0, "ratio_to_1": 2.5,
                    "attack_ms": 12.0, "release_ms": 140.0}},
    ],
    "master": [
        {"id": "lim", "type": "true_peak_limiter",
         "params": {"ceiling_dbtp": -1.0}},
    ],
    "explanation": "Gentle shaping and levelling.",
}


def graph_json(**overrides) -> str:
    data = json.loads(json.dumps(GOOD_GRAPH))
    data.update(overrides)
    return json.dumps(data)


# ---------------------------------------------------------------------------


def test_quantifier_ladder() -> None:
    print("[1] quantifier ladder")

    # The arithmetic must be identical every time. That is the entire reason it
    # lives in Python rather than in a prompt.
    R.near(ladder.adjust("gain_db", 0.0, "slightly", "increase"), 1.5, 1e-9,
           "one step of EQ gain adds 1.5 dB")
    R.near(ladder.adjust("gain_db", 0.0, "a_lot", "increase"), 4.5, 1e-9,
           "'a lot' is three steps")
    R.near(ladder.adjust("gain_db", 0.0, "slightly", "decrease"), -1.5, 1e-9,
           "and decreasing reverses it")

    # "slightly" and "a bit" are deliberately identical. Users treat them as
    # synonyms and distinguishing them would invent precision the words lack.
    R.check(ladder.steps_for("slightly") == ladder.steps_for("a_bit"),
            "'slightly' and 'a bit' are the same size")

    # Frequency is logarithmic, so its step multiplies. One third of an octave.
    R.near(ladder.adjust("center_freq_hz", 1000.0, "slightly", "increase"),
           1260.0, 1.0, "a frequency step is a 1.26 ratio, not an addition")
    R.near(ladder.adjust("center_freq_hz", 100.0, "slightly", "increase"),
           126.0, 1.0, "and the same ratio at a tenth of the frequency")

    # A threshold step is deliberately larger than an EQ step, and negative:
    # more compression means a LOWER threshold.
    R.near(ladder.adjust("threshold_dbfs", -20.0, "slightly", "increase"),
           -22.5, 1e-9, "one step of 'more compression' lowers the threshold 2.5 dB")

    # An undeclared parameter raises rather than guessing. A step size is a
    # judgement about audibility, and inventing one at call time would make the
    # same word mean different things on different parameters.
    try:
        ladder.adjust("no_such_param", 1.0, "slightly", "increase")
        R.check(False, "should refuse a parameter with no declared step")
    except KeyError:
        R.check(True, "refuses a parameter with no declared step")

    # An unknown intensifier gets one step, not zero. The user still asked for a
    # change, and doing nothing looks broken.
    R.check(ladder.steps_for("terrifically") == 1,
            "an unrecognised intensifier is one step, not zero")
    R.check(ladder.steps_for(None) == 1, "and so is no intensifier at all")

    # Every parameter in the ladder must exist in the schema, or a patch would
    # move something the engine never reads.
    from backend.schemas.graph import DynamicEqBand, EqBand, PARAM_MODELS

    known = {"mix"}
    for model in list(PARAM_MODELS.values()) + [EqBand, DynamicEqBand]:
        known |= set(model.model_fields)
    unknown = sorted(set(ladder.STEPS) - known)
    R.check(not unknown, f"every ladder parameter exists in the schema; {unknown}")


def test_intent_vocabulary() -> None:
    print("[2] intent vocabulary")

    graph = EffectGraph.model_validate(json.loads(graph_json(
        nodes=GOOD_GRAPH["nodes"] + [
            {"id": "dly", "type": "delay", "mix": 0.2,
             "params": {"time_ms": 300.0, "feedback_percent": 30.0}},
        ])))

    # THE AMBIGUITY RESOLVED IN CODE. "More delay" means louder, not longer. A
    # planner allowed to choose would move the repeats off the beat sometimes.
    patches = ladder.patch_from_intent(graph, "more_delay", "slightly")
    R.check(len(patches) == 1 and patches[0].param == "mix",
            f"'more delay' means mix, got {[p.param for p in patches]}")
    R.check(ladder.patch_from_intent(graph, "more_repeats")[0].param
            == "feedback_percent", "'more repeats' means feedback")
    R.check(ladder.patch_from_intent(graph, "longer_delay")[0].param == "time_ms",
            "'longer delay' means time")

    # Some "more"s decrease a number. A darker delay is a LOWER damping corner.
    R.check(ladder.patch_from_intent(graph, "darker_delay")[0].direction == "decrease",
            "'darker delay' lowers the damping frequency")

    # An intent with no matching node returns nothing rather than inventing one.
    # Adding a reverb to a chain the user never asked to have one in would be a
    # much larger change than they requested.
    R.check(ladder.patch_from_intent(graph, "more_reverb") == [],
            "an intent with no matching node produces no patches")

    try:
        ladder.patch_from_intent(graph, "make_it_sound_expensive")
        R.check(False, "should reject an intent outside the vocabulary")
    except KeyError:
        R.check(True, "rejects an intent outside the closed vocabulary")


def test_patch_application() -> None:
    print("[3] patches are surgical")

    graph = EffectGraph.model_validate(json.loads(graph_json()))
    before = graph.model_dump()

    patches = [Patch(op="adjust_param", node_id="comp", param="ratio_to_1",
                     quantifier="slightly", direction="increase")]
    result = ladder.apply_patches(graph, patches)

    R.check(result.ok, f"the patch applied: {result.rejected}")
    after = result.graph.model_dump()

    # THE PROPERTY THAT MAKES REFINEMENT USABLE. Exactly one value moved.
    # Regenerating instead would churn parameters the user implicitly accepted,
    # and to them that reads as a bug: they asked for one change and six moved.
    changed = []
    for i, (b, a) in enumerate(zip(before["nodes"], after["nodes"])):
        for key in b["params"]:
            if b["params"][key] != a["params"][key]:
                changed.append(f"nodes[{i}].{key}")
    R.check(changed == ["nodes[1].ratio_to_1"],
            f"exactly one parameter moved, got {changed}")
    R.near(after["nodes"][1]["params"]["ratio_to_1"], 2.5 * 1.35, 0.01,
           "and it moved by exactly one multiplicative step")

    # Values are rounded to something a human would type. An engineer reading
    # "-3.2 dB at 340 Hz" trusts it; nine decimal places reads as machine output.
    freq_patch = [Patch(op="adjust_param", node_id="shape", param="center_freq_hz",
                        band_index=1, quantifier="slightly", direction="increase")]
    freq_result = ladder.apply_patches(graph, freq_patch)
    moved = freq_result.graph.nodes[0].params["bands"][1]["center_freq_hz"]
    R.check(moved == round(moved / 10) * 10,
            f"a frequency lands on a round number, got {moved}")

    # A patch naming a node that does not exist is REJECTED, not ignored. A
    # silently dropped edit leaves the user with no change and no explanation.
    missing = ladder.apply_patches(graph, [
        Patch(op="adjust_param", node_id="nope", param="gain_db",
              quantifier="slightly")])
    R.check(not missing.ok and "nope" in missing.rejected[0],
            f"a patch on a missing node is rejected, got {missing.rejected}")

    # Structural ops.
    added = ladder.apply_patches(graph, [
        Patch(op="add_node", node_type="plate_reverb", node_id="plate",
              after="comp", value={"decay_s": 1.6})])
    R.check([n.id for n in added.graph.nodes] == ["shape", "comp", "plate"],
            f"add_node inserts in position, got {[n.id for n in added.graph.nodes]}")

    # A duplicate id is renamed rather than rejected: the user asked for another
    # reverb, and refusing over a name collision would be pedantry.
    twice = ladder.apply_patches(graph, [
        Patch(op="add_node", node_type="compressor", node_id="comp")])
    ids = [n.id for n in twice.graph.nodes]
    R.check(len(set(ids)) == len(ids) and "comp_2" in ids,
            f"a colliding id is made unique, got {ids}")

    bypassed = ladder.apply_patches(graph, [Patch(op="bypass_node", node_id="comp")])
    R.check(bypassed.graph.nodes[1].bypass, "bypass_node works")
    R.check(not ladder.apply_patches(bypassed.graph,
            [Patch(op="enable_node", node_id="comp")]).graph.nodes[1].bypass,
            "and enable_node reverses it")

    removed = ladder.apply_patches(graph, [
        Patch(op="remove_band", node_id="shape", band_index=0)])
    R.check(len(removed.graph.nodes[0].params["bands"]) == 1,
            "remove_band works")
    R.check(not ladder.apply_patches(graph, [
        Patch(op="remove_band", node_id="shape", band_index=9)]).ok,
        "and an out-of-range band index is rejected")

    # The diff must be describable BEFORE rendering — the plan's definition of
    # done for refinement is that the user reads it first.
    described = ladder.describe(patches)
    R.check(len(described) == 1 and "ratio_to_1" in described[0],
            f"patches describe themselves, got {described}")


def test_glossary() -> None:
    print("[4] descriptor glossary")

    R.check(len(glossary.GLOSSARY) >= 35,
            f"the glossary has enough terms to be useful, {len(glossary.GLOSSARY)}")

    muddy = glossary.lookup("muddy")
    R.check(muddy is not None and muddy.direction == -1, "'muddy' resolves to a cut")
    R.check(glossary.lookup("MUD") is muddy, "aliases and case both resolve")
    R.check(glossary.lookup("Murky") is muddy, "and so do spelling variants")
    R.check(glossary.lookup("bananas") is None, "an unknown term returns nothing")

    # EVERY ENTRY MUST CARRY A CONDITION. Without check_first the glossary is a
    # list of stock moves, and a chain of stock moves sounds like one.
    missing = [d.term for d in glossary.GLOSSARY if not d.check_first.strip()]
    R.check(not missing, f"every term states what to check first; {missing}")

    # A tonal term needs a region; a non-tonal one must not claim a fake one.
    for entry in glossary.GLOSSARY:
        if entry.direction != 0:
            R.check(entry.high_hz > entry.low_hz > 0.0,
                    f"{entry.term} has a real frequency region")

    # The terms that exist to be recognised as UNACTIONABLE are the interesting
    # ones: a planner that invented a frequency for "vintage" would produce a
    # caricature.
    for term in ("vintage", "modern", "lofi", "honest"):
        entry = glossary.lookup(term)
        R.check(entry is not None and entry.direction == 0,
                f"{term!r} has no default move")

    # 'dull' and 'airy' both depend on the bandwidth limit, which is the one
    # measurement that decides whether brightening is possible at all.
    for term in ("dull", "airy"):
        entry = glossary.lookup(term)
        R.check(entry is not None and "wall" in entry.check_first,
                f"{term!r} checks for a codec wall first")

    table = glossary.as_prompt_table()
    R.check("check first" in table and "muddy" in table,
            "the prompt table includes the conditions")
    R.check(len(table.splitlines()) == len(glossary.GLOSSARY) + 1,
            "one line per term plus a header")


# Prose in `check_first`, and the report field it can only mean. The glossary is
# written for a reader, so its conditions say "the noise floor" rather than
# `hygiene.noise_floor_dbfs` — but a condition naming a measurement that does not
# exist is unevaluable, and reads exactly like one that does. This table is what
# lets the test tell those apart, and every path in it is checked against a real
# report below so the table cannot itself go stale.
MEASUREMENT_MARKERS = {
    "own band average": "spectrum.third_octave_db",
    "own average": "spectrum.third_octave_db",
    "band levels": "spectrum.third_octave_db",
    "per-erb crest": "dynamics.per_erb_crest_db",
    "crest factor": "dynamics.crest_factor_db",
    "peak-to-loudness": "dynamics.peak_to_loudness_ratio_db",
    "short-term rms percentile": "dynamics.short_term_rms_percentiles",
    "envelope modulation": "dynamics.envelope_modulation",
    "noise floor": "hygiene.noise_floor_dbfs",
    "clipping": "hygiene.clipping",
    "clipped": "hygiene.clipping",
    "non-finite": "hygiene.non_finite_samples",
    "codec wall": "spectrum.bandwidth_limit_hz",
    "bandwidth limit": "spectrum.bandwidth_limit_hz",
    "hf rolloff": "spectrum.hf_rolloff_db_per_octave",
    "tilt": "spectrum.tilt_db_per_octave_re_pink",
    "mono-compatibility": "stereo.mono_loss_db",
    "mono loss": "stereo.mono_loss_db",
    "correlation": "stereo.correlation",
    "loudness range": "loudness.loudness_range_lu",
    "integrated loudness": "loudness.integrated_lufs",
    "true peak": "loudness.true_peak_dbtp",
    "gated coverage": "source.gated_coverage_percent",
    "channels": "source.channels",
}

# The terms whose conditions predate the rule that a condition must name a
# measurement. Named individually rather than skipped by a date or a range, so
# that ANY term added from now on is enrolled in the grounding check without
# anyone remembering to do it, and so that the size of this list is a visible
# measure of how much of the glossary is still prose.
#
# They fall into three kinds, and only the third is a defect:
#   - unactionable on purpose: honest, clean, vintage, modern, lofi, woody
#   - true of a spectrum but not of one named field: fat, boxy, honky, hollow,
#     nasal, forward, edgy, fizzy, rumbly, sibilant, warm
#   - naming a measurement THAT DOES NOT EXIST: dry, washy, distant and pumping
#     all turn on a reverb decay or an existing compressor's release, and the
#     report produces neither. See docs/open-questions.md.
UNGROUNDED_LEGACY_TERMS = frozenset({
    "rumbly", "fat", "boxy", "honky", "woody", "hollow", "nasal", "forward",
    "honest", "edgy", "sibilant", "fizzy", "pumping", "dry", "washy", "distant",
    "warm", "clean", "vintage", "modern", "lofi",
})


def _measure_a_real_report() -> dict:
    """A genuine report, not a hand-written dict of field names.

    A literal is the one thing that cannot verify this property: it would be
    written by reading the same source the glossary was written from, and would
    keep agreeing with a field the analyser had stopped producing.
    """
    import numpy as np

    from backend.measure.report import measure

    rng = np.random.default_rng(7)
    sample_rate = 48000.0
    frames = int(3 * sample_rate)
    t = np.arange(frames) / sample_rate
    # Amplitude-modulated noise: loud enough and varied enough that the gate
    # keeps material and every section of the report is populated.
    envelope = 0.4 + 0.6 * np.abs(np.sin(2 * np.pi * 2.0 * t))
    samples = rng.normal(0.0, 0.05, size=(2, frames)) * envelope
    return measure(samples, sample_rate).to_dict()


def _has_path(report: dict, path: str) -> bool:
    node: object = report
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    return True


def test_glossary_conditions() -> None:
    print("[4b] glossary conditions name real measurements")

    report = _measure_a_real_report()

    # The marker table first. If a path here has gone stale the grounding check
    # below would still pass on the prose alone, which is the failure mode this
    # whole test exists to prevent one level down.
    stale = sorted({path for path in MEASUREMENT_MARKERS.values()
                    if not _has_path(report, path)})
    R.check(not stale, f"every marker names a field a real report produces; {stale}")

    # THE CENTRAL ASSERTION. A condition that names nothing measurable is worse
    # than a missing term: the planner reads it, cannot evaluate it, and falls
    # back on the stock move with a rigorous-looking sentence attached.
    ungrounded = []
    for entry in glossary.GLOSSARY:
        if entry.term in UNGROUNDED_LEGACY_TERMS:
            continue
        text = entry.check_first.lower()
        if not any(marker in text for marker in MEASUREMENT_MARKERS):
            ungrounded.append(entry.term)
    R.check(not ungrounded,
            f"every non-legacy term's condition names a measured field; {ungrounded}")

    # And the exemption list cannot outlive the terms it exempts. Without this a
    # renamed term would silently take its exemption with it.
    unknown = sorted(UNGROUNDED_LEGACY_TERMS
                     - {entry.term for entry in glossary.GLOSSARY})
    R.check(not unknown, f"the exemption list names only real terms; {unknown}")

    # THE EXEMPTION LIST MAY ONLY SHRINK. Found by mutation: adding a new term to
    # it silenced the grounding check for that term and the suite still passed —
    # so the assertion above was, on its own, an assertion the next author could
    # satisfy by editing the list instead of the entry. Twenty-one is what it was
    # when the rule was introduced. Grounding an old term lowers it; nothing
    # raises it.
    R.check(len(UNGROUNDED_LEGACY_TERMS) <= 21,
            f"the exemption list only shrinks, now {len(UNGROUNDED_LEGACY_TERMS)} "
            "of the original 21")

    # THE SIGN MUST FOLLOW THE CONDITION, not the other way round. A term whose
    # condition describes an EXCESS and whose default move is a boost has one of
    # the two written backwards, and the entry will read as authoritative either
    # way. Checked across the whole glossary, legacy terms included: this one
    # needs no field to exist, only the two halves of the entry to agree.
    for entry in glossary.GLOSSARY:
        text = entry.check_first.lower()
        excess = ("above the source's own" in text
                  or "a peak exists" in text
                  or ("elevated" in text and "not already elevated" not in text))
        deficit = ("below the source's own" in text
                   or "beneath the source's own" in text
                   or "scooped" in text)
        if excess and not deficit:
            R.check(entry.direction <= 0,
                    f"{entry.term}: condition describes an excess, so the default "
                    f"move cannot be a boost (got {entry.direction:+d})")
        if deficit and not excess:
            R.check(entry.direction >= 0,
                    f"{entry.term}: condition describes a deficit, so the default "
                    f"move cannot be a cut (got {entry.direction:+d})")

    # The words themselves, independent of what the entry says about itself. A
    # term that means less of something must never carry a positive default, and
    # a term that means more of something must never carry a negative one — a
    # sign error here inverts the move on a source the planner never questions
    # because the condition it checked was satisfied.
    means_less = ("plosive", "chesty", "midrangey", "ringy", "shrill", "smooth",
                  "clicky", "muddy", "boomy", "harsh", "sibilant", "hissy")
    means_more = ("bassy", "breathy", "lispy", "thin", "fat", "bright", "airy",
                  "hollow", "buried")
    for term in means_less:
        entry = glossary.lookup(term)
        R.check(entry is not None and entry.direction <= 0,
                f"{term!r} names an excess, so it cannot default to a boost")
    for term in means_more:
        entry = glossary.lookup(term)
        R.check(entry is not None and entry.direction >= 0,
                f"{term!r} asks for more of a region, so it cannot default to a cut")

    # Every region a term claims has to be a region a filter can be placed in.
    # 20 Hz to 20 kHz is not a style rule: a centre outside it is either a typo
    # or a unit error, and both survive review because the number looks like a
    # frequency.
    for entry in glossary.GLOSSARY:
        if not (entry.low_hz or entry.high_hz):
            continue
        R.check(20.0 <= entry.low_hz < entry.high_hz <= 20000.0,
                f"{entry.term} claims a placeable region, got "
                f"{entry.low_hz}-{entry.high_hz} Hz")

    # THE DUPLICATE-ALIAS GUARD, restated without the exception. The import-time
    # guard in glossary.py raises, which means a collision takes the whole module
    # down and this file never loads — so this check is here to name the offender
    # if the guard is ever softened to a warning, which is the obvious "fix" for
    # a module that will not import.
    seen: dict[str, str] = {}
    collisions = []
    for entry in glossary.GLOSSARY:
        for name in (entry.term, *entry.aliases):
            if name in seen:
                collisions.append(f"{name} ({seen[name]} / {entry.term})")
            seen[name] = entry.term
    R.check(not collisions, f"no alias is claimed twice; {collisions}")
    R.check(len(glossary.INDEX) == sum(1 + len(e.aliases) for e in glossary.GLOSSARY),
            "the index holds every term and alias exactly once")

    # The words from the prompt's own example list all resolve. 'presence' was
    # the one that did not: 'present' was an alias of 'forward' and 'presence'
    # was not, so the commonest spelling of the commonest request returned None.
    for term in ("warm", "boxy", "muddy", "harsh", "presence", "air", "punch"):
        R.check(glossary.lookup(term) is not None, f"{term!r} resolves")

    # Separator and case handling has to reach the new aliases too, since several
    # of them are multi-word phrases a user would type with spaces.
    R.check(glossary.lookup("More Bass") is glossary.lookup("bassy"),
            "a multi-word alias resolves with spaces and capitals")
    R.check(glossary.lookup("over-de-essed") is glossary.lookup("lispy"),
            "and with hyphens")

    # The terms that answer a request with "no". These are the entries most
    # likely to be quietly given a direction later by someone tidying up, and
    # the whole value of them is that they do not have one.
    for term in ("tinny", "clear", "tight", "roomy", "distorted", "sterile"):
        entry = glossary.lookup(term)
        R.check(entry is not None and entry.direction == 0,
                f"{term!r} has no default move, because the measurement decides it")


def test_cache_boundary() -> None:
    print("[5] the prompt cache boundary")

    # THE MOST VALUABLE ASSERTION HERE. A per-request value in the prefix does
    # not fail: the request succeeds, the answer is fine, and the bill multiplies.
    # Only looking for it finds it.
    first = prompts.system_blocks()
    second = prompts.system_blocks()
    R.check(json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True),
            "two independently-built prefixes are byte-identical")

    # And it cannot accidentally take a request-shaped argument.
    import inspect

    R.check(not inspect.signature(prompts.system_blocks).parameters,
            "system_blocks takes no arguments, so nothing per-request can reach it")

    prompt_a = prompts.plan_prompt(report=REPORT, request="warmer, less harsh")
    prompt_b = prompts.plan_prompt(report=REPORT, request="completely different")
    R.check(json.dumps(prompt_a.system) == json.dumps(prompt_b.system),
            "two different requests share an identical prefix")

    # The prefix must be large enough that caching is worth the complexity.
    tokens = prompt_a.estimated_prefix_tokens()
    R.check(tokens > 8000, f"the prefix is worth caching, roughly {tokens} tokens")

    # A cache breakpoint must exist, or nothing is cached at all.
    R.check(any("cache_control" in block for block in prompt_a.system),
            "the prefix carries a cache breakpoint")

    # Nothing about this request may appear in the prefix.
    prefix = prompt_a.prefix_text()
    for leaked in ("warmer, less harsh", "-19.4", "conventional_stereo"):
        R.check(leaked not in prefix,
                f"the prefix does not contain the per-request value {leaked!r}")

    # And the volatile half must contain them, or the model is flying blind.
    volatile = prompt_a.messages[0]["content"]
    for needed in ("warmer, less harsh", "-19.4"):
        R.check(needed in volatile, f"the message carries {needed!r}")

    # The DSP source is the highest-leverage item in the prompt and most of its
    # size, which is exactly why the prefix has to cache.
    # Probes taken from the source files themselves, not from my documentation
    # of them. The first version checked for "TPT", which appears in docs/engine.md
    # and nowhere in the header — a test that would have passed if the DSP
    # reference were dropped entirely, as long as the docs still mentioned it.
    for probe in ("===== ParametricEQ.h =====", "===== Compressor.h =====",
                  "Minimum-phase parametric EQ", "shapeFromString"):
        R.check(probe in prefix,
                f"the prefix includes the DSP source, probe {probe!r}")

    # And the prefix must be close to the size the cache design assumes. Well
    # under and the DSP reference has silently stopped being included; well over
    # and the volatile half is no longer the cheap part.
    # The bracket is around the LIVE API's own count, 30,938 cached tokens, not
    # around a character-count guess. It used to be 14k-26k, from back when the
    # estimator assumed 4 characters per token; the constant was later corrected
    # to 2.4 against that measurement and this bound was not moved with it, so
    # the test failed on a prefix that was in fact exactly the right size.
    R.check(26000 < tokens < 36000,
            f"the prefix is the expected size, roughly {tokens} tokens")

    # The user's own words go last, where instructions carry most weight.
    R.check(volatile.index("warmer, less harsh") > volatile.index("Measurement report"),
            "the request appears after the measurements")

    # Stage 2 reframes the model as a transcriber, which it cannot be in the same
    # turn that still holds the reasoning task.
    emit = prompts.emit_prompt(report=REPORT, request="warmer", reasoning="do X")
    R.check("transcriber" in emit.messages[0]["content"],
            "the emit prompt reframes the model as a transcriber")
    R.check("do X" in emit.messages[0]["content"],
            "and carries the decision to transcribe")
    # THE TWO STAGES MUST NOT SHARE A PREFIX. This assertion used to say the
    # opposite, and the opposite was a bug: the stages run on different models,
    # so one shared prefix is written into two caches and read from neither.
    # The cost was paid twice for a saving collectable once.
    R.check(json.dumps(emit.system) != json.dumps(prompt_a.system),
            "the two stages do not share a prefix, because they do not share a model")

    emit_tokens = emit.estimated_prefix_tokens()
    R.check(emit_tokens < tokens * 0.5,
            f"the emit prefix is far smaller: {emit_tokens} against {tokens}")
    R.check(emit_tokens > 8000,
            f"but still worth caching on its own, {emit_tokens} tokens")

    # What EMIT drops, it drops because a transcriber cannot use it. Probed on
    # content rather than on size, so shrinking the prefix some other way while
    # leaving these in would not pass.
    emit_prefix = emit.prefix_text()
    for gone in ("===== ParametricEQ.h =====", "shapeFromString", "boxy"):
        R.check(gone not in emit_prefix,
                f"the emit prefix drops {gone!r}: the decision is already made")
    for kept in ("effect_graph_v1", "ceiling_dbtp"):
        R.check(kept in emit_prefix,
                f"but keeps {kept!r}, which is what it is serialising into")

    # The transcriber framing moved into the cached prefix. It is static, so
    # paying for it in the volatile half every call was waste.
    R.check("not the engineer" in emit_prefix.lower(),
            "the transcriber framing is in the cached prefix")
    R.check(json.dumps(prompts.emit_system_blocks())
            == json.dumps(prompts.emit_system_blocks()),
            "and the emit prefix is byte-identical across builds too")


def test_json_extraction() -> None:
    print("[6] extracting a graph from a reply")

    payload = '{"schema_version": "effect_graph_v1", "nodes": []}'
    R.check(pipeline.extract_json(payload)["nodes"] == [], "bare JSON parses")
    R.check(pipeline.extract_json(f"```json\n{payload}\n```")["nodes"] == [],
            "a fenced block parses")
    R.check(pipeline.extract_json(f"Here you go:\n{payload}\nHope that helps")
            ["nodes"] == [], "surrounding prose is tolerated")

    # Intolerant of everything else. A parser that tried harder would start
    # succeeding on replies that are not really graphs, which is worse than
    # failing cleanly.
    for bad, what in (("no json here", "prose with no object"),
                      ("[1, 2, 3]", "a JSON array rather than an object"),
                      ("{not json}", "a malformed object")):
        try:
            pipeline.extract_json(bad)
            R.check(False, f"should have refused {what}")
        except ValueError:
            R.check(True, f"refuses {what}")
        except json.JSONDecodeError:
            R.check(True, f"refuses {what}")


def test_pipeline_happy_path() -> None:
    print("[7] pipeline, accepted first time")

    # Deliberately the realistic default: a one-line reasoning is now refused,
    # and the refusal is the point of test [15].
    client = pipeline.ScriptedClient(
                                    emissions=[graph_json()])
    counters = pipeline.Counters()
    outcome = pipeline.plan(client, report=REPORT, request="warmer, less harsh",
                            counters=counters)

    R.check(outcome.ok, f"the graph was accepted: {outcome.fatal}")
    R.check(outcome.attempts == 1, f"in one attempt, got {outcome.attempts}")
    R.check(outcome.graph is not None and len(outcome.graph.nodes) == 2,
            "with both nodes")
    R.check(outcome.reasoning == pipeline.ScriptedClient.DEFAULT_REASONING,
            "the reasoning is retained")
    R.check(not outcome.used_fallback, "and no fallback was needed")

    # Stage 1 must be unconstrained and stage 2 constrained. Getting this
    # backwards would degrade the reasoning that the whole two-call split exists
    # to protect.
    R.check([c["kind"] for c in client.calls] == ["text", "structured"],
            f"one unconstrained call then one constrained, got "
            f"{[c['kind'] for c in client.calls]}")
    R.check(client.calls[1]["schema"]["additionalProperties"] is False,
            "and the constrained call carries the schema")

    # Stage 2 must see the reasoning, or the split achieves nothing.
    R.check("4. PARAMETER PLANNING" in client.calls[1]["messages"][0]["content"],
            "stage 2 receives stage 1's reasoning")

    R.check(counters.report()["turns"] == 1, "the turn was counted")
    R.check(counters.report()["retry_rate"] == 0.0, "with no retries")


def test_pipeline_repairs() -> None:
    print("[8] pipeline, repaired without a retry")

    # Four 5 dB boosts: 20 dB against a 12 dB budget. Repairable, so it must be
    # fixed in Python rather than costing a second round-trip to arrive at an
    # answer we already have.
    over_budget = json.loads(graph_json())
    over_budget["nodes"][0]["params"]["bands"] = [
        {"filter_type": "bell", "center_freq_hz": f, "gain_db": 5.0, "q_factor": 1.0}
        for f in (120.0, 900.0, 3000.0, 9000.0)
    ]

    client = pipeline.ScriptedClient(emissions=[json.dumps(over_budget)])
    counters = pipeline.Counters()
    outcome = pipeline.plan(client, report=REPORT, request="brighter",
                            counters=counters)

    R.check(outcome.ok, f"accepted after repair: {outcome.fatal}")
    R.check(outcome.attempts == 1, "without a retry")
    R.check(any("budget" in r for r in outcome.repairs),
            f"and the repair was logged: {outcome.repairs}")

    total = sum(b["gain_db"] for b in outcome.graph.nodes[0].params["bands"])
    R.near(total, 12.0, 0.1, "the boosts were scaled to the budget")

    R.check(counters.report()["repair_rate"] == 1.0, "counted as a repair")
    R.check(counters.report()["retry_rate"] == 0.0, "not as a retry")


def test_pipeline_retries() -> None:
    print("[9] pipeline, fatal then corrected")

    # Drive on a clipped source is fatal: saturation amplifies existing
    # distortion rather than adding character. Not repairable, because removing
    # the node would be guessing at what the user wanted instead.
    clipped_report = json.loads(json.dumps(REPORT))
    clipped_report["warnings"] = [{"code": "clipped_source", "severity": "high",
                                  "message": "0.4% clipped"}]

    bad = json.loads(graph_json())
    bad["nodes"].append({"id": "drv", "type": "drive",
                         "params": {"drive_db": 9.0}})

    client = pipeline.ScriptedClient(emissions=[json.dumps(bad), graph_json()])
    counters = pipeline.Counters()
    outcome = pipeline.plan(client, report=clipped_report, request="crunchier",
                            counters=counters)

    R.check(outcome.ok, f"accepted on the retry: {outcome.fatal}")
    R.check(outcome.attempts == 2, f"after two attempts, got {outcome.attempts}")
    R.check(counters.report()["retry_rate"] == 1.0, "counted as a retry")

    # The repair prompt must quote the validator verbatim rather than paraphrase.
    # A validator message already names the measurement and the rule; restating
    # it is an opportunity to soften it.
    retry_text = client.calls[2]["messages"][0]["content"]
    R.check("already clipped" in retry_text,
            "the retry quotes the finding verbatim")
    R.check("Change only what the findings require" in retry_text,
            "and asks for a minimal change rather than a rewrite")


def test_pipeline_exhausts() -> None:
    print("[10] pipeline, exhausted")

    clipped_report = json.loads(json.dumps(REPORT))
    clipped_report["warnings"] = [{"code": "clipped_source", "severity": "high",
                                  "message": "0.4% clipped"}]
    bad = json.loads(graph_json())
    bad["nodes"].append({"id": "drv", "type": "drive",
                         "params": {"drive_db": 9.0}})
    stubborn = json.dumps(bad)

    # No fallback available: the outcome must report failure rather than ship
    # something nobody validated.
    client = pipeline.ScriptedClient(emissions=[stubborn, stubborn, stubborn])
    counters = pipeline.Counters()
    outcome = pipeline.plan(client, report=clipped_report, request="crunchier",
                            counters=counters)

    R.check(not outcome.ok, "an unfixable graph is not accepted")
    R.check(outcome.attempts == pipeline.MAX_RETRIES + 1,
            f"after exactly {pipeline.MAX_RETRIES + 1} attempts, got {outcome.attempts}")
    R.check(outcome.graph is None, "and nothing is returned")
    R.check(counters.report()["fallback_rate"] == 1.0, "counted as a fallback")

    # With a previous graph, fall back to THAT. It is something this pipeline
    # already validated for this source, which a default chain would not be.
    previous = EffectGraph.model_validate(json.loads(graph_json()))
    client2 = pipeline.ScriptedClient(emissions=[stubborn, stubborn, stubborn])
    recovered = pipeline.plan(client2, report=clipped_report, request="crunchier",
                             previous_graph=previous)
    R.check(recovered.used_fallback, "the previous graph is used as the fallback")
    R.check(recovered.graph is not None
            and [n.id for n in recovered.graph.nodes] == ["shape", "comp"],
            "and it is the graph that previously passed")


def test_pipeline_unparseable() -> None:
    print("[11] pipeline, unparseable reply")

    client = pipeline.ScriptedClient(
        emissions=["I'm afraid I can't do that", graph_json()])
    counters = pipeline.Counters()
    outcome = pipeline.plan(client, report=REPORT, request="warmer",
                            counters=counters)

    R.check(outcome.ok, "a garbled first reply is retried and recovers")
    R.check(outcome.attempts == 2, "in two attempts")
    R.check(counters.report()["unparseable_rate"] == 1.0, "and is counted")

    # A schema violation is as retryable as a semantic one, for the same reason:
    # the model can be told what was wrong.
    invalid = json.dumps({"schema_version": "effect_graph_v1",
                          "nodes": [{"id": "x", "type": "tape_machine"}]})
    client2 = pipeline.ScriptedClient(emissions=[invalid, graph_json()])
    R.check(pipeline.plan(client2, report=REPORT, request="warmer").ok,
            "an unknown node type is retried and recovers")


def test_counters() -> None:
    print("[12] SLI counters")

    counters = pipeline.Counters()
    R.check(counters.breaches() == [], "no breaches with no data")

    counters.turns = 100
    counters.repaired = 10
    counters.retried = 3
    counters.fell_back = 0
    R.check(counters.breaches() == [], f"healthy rates pass, {counters.report()}")

    counters.repaired = 30
    breaches = counters.breaches()
    R.check(any("repair rate" in b for b in breaches),
            f"a 30% repair rate breaches, got {breaches}")

    # Rates are meaningless on a handful of samples, so they are not reported
    # until there are enough of them.
    small = pipeline.Counters()
    small.turns = 5
    small.fell_back = 5
    R.check(small.breaches() == [],
            "a rate over five turns is not treated as a signal")


def test_end_to_end_renders() -> None:
    print("[13] a planned graph renders")

    from backend.engine.render import DEFAULT_BINARY, render_array

    if not os.path.exists(DEFAULT_BINARY):
        print("  engine binary missing; skipping")
        return

    import numpy as np

    client = pipeline.ScriptedClient(emissions=[graph_json()])
    outcome = pipeline.plan(client, report=REPORT, request="warmer, less harsh")
    R.check(outcome.ok, "the graph was accepted")

    recipe = outcome.graph.to_recipe(seed=11, output_sample_rate=48000.0,
                                     output_format="float64")
    source = np.random.default_rng(3).normal(0.0, 0.2, (2, 48000))
    rendered = render_array(source, 48000.0, recipe)

    R.check(rendered.shape == source.shape, "and it rendered")
    R.check(bool(np.all(np.isfinite(rendered))), "with finite samples throughout")

    # Then refine it the way a user would, and confirm the refined graph also
    # renders — a patch that produces something unrenderable would be worse than
    # a rejected one.
    patches = [Patch(op="adjust_param", node_id="comp", param="ratio_to_1",
                     quantifier="noticeably", direction="increase")]
    patched = ladder.apply_patches(outcome.graph, patches)
    R.check(patched.ok, f"the refinement applied: {patched.rejected}")

    from backend.plan.validate import validate as revalidate

    revalidated = revalidate(patched.graph, REPORT)
    R.check(revalidated.ok, f"and revalidates: {revalidated.fatal}")

    again = render_array(source, 48000.0, revalidated.graph.to_recipe(
        seed=11, output_sample_rate=48000.0, output_format="float64"))
    R.check(bool(np.all(np.isfinite(again))), "and the refined graph renders too")
    R.check(not np.array_equal(again, rendered),
            "and it sounds different, since the refinement changed something")


def test_delivery_ceiling_is_never_optional() -> None:
    """A graph with no limiter must acquire one, not be rejected.

    THE BUG THIS PINS DOWN was invisible on the excerpt. Planning and
    auditioning run on a 20-second window; on a real guitar track that window
    measured -2.53 dBTP while the whole file measured +0.52. A chain that never
    touched level therefore cleared 0 dBTP on the audition and blew through it
    on delivery, and the failure surfaced only at download, which is the worst
    possible moment for it.

    The excerpt is not at fault and no window choice fixes it — 20 seconds
    cannot promise anything about the 230 it excludes. The ceiling is what makes
    the audition predict the delivery.
    """
    print("[14] a delivery ceiling is added, not demanded")

    graph = EffectGraph.model_validate(json.loads(graph_json()))
    graph.master = []
    result = validate(graph, None)

    R.check(result.ok, "a graph with no limiter is repaired rather than rejected")

    limiters = [n for n in result.graph.master if n.type == "true_peak_limiter"]
    R.check(len(limiters) == 1, "exactly one true-peak limiter is present after validation")
    R.check(limiters and limiters[0].params["ceiling_dbtp"] == -1.0,
            "and it holds the conventional -1.0 dBTP delivery ceiling")
    R.check(any("ceiling" in r for r in result.repairs),
            "and the repair is reported rather than applied silently")

    # It must not duplicate one the planner already placed, or two limiters
    # end up in series and the second one meters a signal the first flattened.
    twice = validate(result.graph, None)
    R.check(len([n for n in twice.graph.master if n.type == "true_peak_limiter"]) == 1,
            "re-validating does not add a second limiter")

    # And it must not silently displace a ceiling the planner chose on purpose.
    chosen = EffectGraph.model_validate(json.loads(graph_json()))
    chosen.master = [GraphNode(id="ceiling", type="true_peak_limiter",
                               params={"ceiling_dbtp": -0.3},
                               rationale="the plan asked for -0.3")]
    kept = validate(chosen, None)
    R.check([n.params["ceiling_dbtp"] for n in kept.graph.master
             if n.type == "true_peak_limiter"] == [-0.3],
            "a ceiling the planner chose deliberately is left alone")



def test_stage_one_must_finish() -> None:
    """A truncated or missing decision must never reach stage 2.

    THE BUG THIS PINS DOWN RAN IN PRODUCTION FOR THE WHOLE BUILD. Stage 1 was
    called with max_tokens=8192. Opus spends output tokens on thinking and
    thinking counts against that ceiling, so a real call returned ONE thinking
    block, ZERO text blocks, and stop_reason max_tokens. The client filtered for
    text blocks, found none, and returned "".

    Stage 2 was then handed an empty decision. It could not tell that apart from
    a decision with no prose, so it invented the chain from the measurement
    report — which is exactly what the two-stage split exists to prevent, and
    the reason it exists is that constrained decoding degrades reasoning.

    Nothing downstream noticed. The chains were plausible, the validators passed
    them, the safety gate passed the renders, and the only visible trace was a
    single retry that looked like ordinary schema friction.
    """
    print("[15] stage 1 must finish before stage 2 begins")

    class Truncating(pipeline.ScriptedClient):
        def complete(self, system, messages):
            raise pipeline.TruncatedReply("hit the ceiling")

    client = Truncating(emissions=[graph_json()])
    counters = pipeline.Counters()
    outcome = pipeline.plan(client, report=REPORT, request="warmer",
                            counters=counters)

    R.check(not outcome.ok, "a truncated stage 1 fails the turn")
    R.check(any("ceiling" in f for f in outcome.fatal),
            f"and says why: {outcome.fatal}")
    R.check(not any(c["kind"] == "structured" for c in client.calls),
            "and stage 2 is never called, so it cannot invent a chain")
    R.check(counters.fell_back == 1, "and it counts as a fallback")

    # Text that arrives but is too short is the same failure wearing prose. The
    # four-step task cannot be answered in one line.
    short = pipeline.ScriptedClient(reasoning="Use an EQ.",
                                    emissions=[graph_json()])
    outcome = pipeline.plan(short, report=REPORT, request="warmer")
    R.check(not outcome.ok, "a one-line decision fails too")
    R.check(not any(c["kind"] == "structured" for c in short.calls),
            "and again stage 2 never runs")

    # With a previous graph, the turn falls back to something already accepted
    # rather than failing outright — the same policy as an exhausted repair loop.
    previous = EffectGraph.model_validate(json.loads(graph_json()))
    fell = pipeline.plan(Truncating(emissions=[graph_json()]), report=REPORT,
                         request="warmer", previous_graph=previous)
    R.check(fell.ok and fell.used_fallback,
            "with a previous graph it falls back to one that already passed")

    # And the healthy path must still reach stage 2, or this test would pass by
    # breaking the pipeline entirely.
    healthy = pipeline.ScriptedClient(emissions=[graph_json()])
    R.check(pipeline.plan(healthy, report=REPORT, request="warmer").ok,
            "a complete decision still goes through")
    R.check(any(c["kind"] == "structured" for c in healthy.calls),
            "and does reach stage 2")



def test_envelope_unwrapping() -> None:
    """The model wraps the graph in one key. Unwrap it, but only when it is one.

    Measured against the live API across four trials crossing prefix size with
    reasoning length: every reply nested the graph inside a single key rather
    than returning it directly. The key was "graph" with the small prefix and
    "$PARAMETER_VALUE" or "$PARAMETER_NAME" with the full one.
    """
    print("[16] single-key envelopes are unwrapped, degenerate ones are not")

    graph = json.loads(graph_json())

    for key in ("graph", "$PARAMETER_VALUE", "effect_graph", "input"):
        notes = []
        got = pipeline.extract_json(json.dumps({key: graph}), notes)
        R.check(got == graph, f"a {key!r} envelope is unwrapped")
        R.check(any(key in n for n in notes),
                f"and the {key!r} unwrap is recorded, not silent")

    # A real graph is passed through untouched and nothing is recorded.
    notes = []
    R.check(pipeline.extract_json(json.dumps(graph), notes) == graph,
            "an unwrapped graph is returned as-is")
    R.check(notes == [], "and nothing is recorded for it")

    # THE DEGENERATE REPLY MUST STAY A HARD ERROR. This one really was returned
    # by the API. There is no graph inside it, so there is nothing to recover
    # and anything this function did would be an invention.
    degenerate = {"$PARAMETER_NAME": "schema_version",
                  "$PARAMETER_NAME2": "effect_graph_v1"}
    R.check(pipeline.extract_json(json.dumps(degenerate)) == degenerate,
            "a degenerate reply is not rescued")
    try:
        EffectGraph.model_validate(pipeline.extract_json(json.dumps(degenerate)))
        R.check(False, "and it should fail validation")
    except Exception:
        R.check(True, "and it fails validation, as it must")

    # Two keys is not an envelope even when one of them holds a graph — that is
    # a reply this function does not understand, and guessing which key matters
    # is exactly the overreach the module docstring warns about.
    two = {"graph": graph, "notes": "something else"}
    R.check(pipeline.extract_json(json.dumps(two)) == two,
            "a two-key object is not treated as an envelope")

    # A single key whose value is not a graph is left alone too.
    R.check(pipeline.extract_json(json.dumps({"graph": "not a graph"}))
            == {"graph": "not a graph"},
            "a single key wrapping a non-graph is left alone")



def test_failures_always_state_a_reason() -> None:
    """An exhausted turn must say why, including when nothing ever parsed.

    This was a real hole. The parse-failure branch set the finding for the
    retry prompt but never put it on the outcome, so a turn where all three
    attempts failed to parse returned ok=False with fatal=[] — a failure with
    no stated reason at all. It cost real time to diagnose the first time it
    happened, because the empty list looked like a different bug entirely.
    """
    print("[17] a failed turn always states a reason")

    # Nothing ever parses.
    client = pipeline.ScriptedClient(emissions=["not json", "still not", "nope"])
    outcome = pipeline.plan(client, report=REPORT, request="warmer")
    R.check(not outcome.ok, "a turn where nothing parses fails")
    R.check(outcome.fatal, f"and says why, got {outcome.fatal}")
    R.check(any("parse" in f for f in outcome.fatal),
            f"naming the parse failure: {outcome.fatal}")

    # Parses, but violates a validator every time. This path already worked;
    # it is here so the test covers both ways of exhausting the loop.
    hot = json.loads(graph_json())
    hot["nodes"][0]["params"]["bands"] = [
        {"filter_type": "bell", "center_freq_hz": f, "gain_db": 17.0,
         "q_factor": 1.0} for f in (100.0, 1000.0, 5000.0)]
    hot["explanation"] = "x" * 5000        # over the cap, so it cannot validate
    client = pipeline.ScriptedClient(emissions=[json.dumps(hot)] * 3)
    outcome = pipeline.plan(client, report=REPORT, request="warmer")
    R.check(not outcome.ok and outcome.fatal,
            f"a turn that never validates also states a reason: {outcome.fatal}")



if __name__ == "__main__":
    print("Audionova planner scaffolding\n")
    test_quantifier_ladder()
    test_intent_vocabulary()
    test_patch_application()
    test_glossary()
    test_glossary_conditions()
    test_cache_boundary()
    test_json_extraction()
    test_pipeline_happy_path()
    test_pipeline_repairs()
    test_pipeline_retries()
    test_pipeline_exhausts()
    test_pipeline_unparseable()
    test_counters()
    test_end_to_end_renders()
    test_delivery_ceiling_is_never_optional()
    test_stage_one_must_finish()
    test_envelope_unwrapping()
    test_failures_always_state_a_reason()
    print(f"\n{R.checks} checks, {R.failures} failure(s)")
    sys.exit(1 if R.failures else 0)
