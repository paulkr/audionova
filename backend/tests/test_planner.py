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
    R.check(json.dumps(emit.system) == json.dumps(prompt_a.system),
            "both stages share the same cached prefix")


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

    client = pipeline.ScriptedClient(reasoning="I chose an EQ and a compressor.",
                                    emissions=[graph_json()])
    counters = pipeline.Counters()
    outcome = pipeline.plan(client, report=REPORT, request="warmer, less harsh",
                            counters=counters)

    R.check(outcome.ok, f"the graph was accepted: {outcome.fatal}")
    R.check(outcome.attempts == 1, f"in one attempt, got {outcome.attempts}")
    R.check(outcome.graph is not None and len(outcome.graph.nodes) == 2,
            "with both nodes")
    R.check(outcome.reasoning.startswith("I chose"), "the reasoning is retained")
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
    R.check("I chose an EQ" in client.calls[1]["messages"][0]["content"],
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


if __name__ == "__main__":
    print("Audionova planner scaffolding\n")
    test_quantifier_ladder()
    test_intent_vocabulary()
    test_patch_application()
    test_glossary()
    test_cache_boundary()
    test_json_extraction()
    test_pipeline_happy_path()
    test_pipeline_repairs()
    test_pipeline_retries()
    test_pipeline_exhausts()
    test_pipeline_unparseable()
    test_counters()
    test_end_to_end_renders()
    print(f"\n{R.checks} checks, {R.failures} failure(s)")
    sys.exit(1 if R.failures else 0)
