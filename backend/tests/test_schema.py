"""The effect graph schema, its validators, and their agreement with the engine.

Three things are asserted here that nothing else can catch.

THE SCHEMA MUST AGREE WITH THE ENGINE. Every node type and every parameter name
in the schema has to exist in `engine/include/anv/nodes/`, checked by reading the
headers. A schema that drifts produces graphs which validate and then fail to
render — or worse, render with a silently defaulted parameter, which is the
failure mode this project has spent the most effort eliminating.

BOUNDS MUST APPEAR IN THE DESCRIPTION TEXT. Claude structured outputs honour
`enum` and `additionalProperties` but do NOT enforce `minimum`/`maximum`, so a
bound that lives only in the numeric keywords is a bound the model never sees and
the API never applies. Every ranged field is checked for its range in prose.

A VALID GRAPH MUST ACTUALLY RENDER. The schema is only correct if something that
passes it survives contact with the engine, so the last section builds a graph,
converts it to a recipe, and renders real samples through it.
"""

from __future__ import annotations

import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.plan import validate as validator  # noqa: E402
from backend.schemas import graph as G  # noqa: E402

ENGINE_NODES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "engine", "include", "anv", "nodes",
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


def node(node_type: str, node_id: str = "n", **params) -> G.GraphNode:
    return G.GraphNode(id=node_id, type=node_type, params=params)


def eq(node_id: str = "eq", *bands) -> G.GraphNode:
    return G.GraphNode(id=node_id, type="parametric_eq", params={"bands": list(bands)})


def band(freq: float, gain: float, q: float = 1.0, kind: str = "bell") -> dict:
    return {"filter_type": kind, "center_freq_hz": freq, "gain_db": gain,
            "q_factor": q}


# ---------------------------------------------------------------------------


def test_agrees_with_engine() -> None:
    print("[1] schema agrees with the engine")

    sources = {}
    for name in os.listdir(ENGINE_NODES):
        if name.endswith(".h"):
            with open(os.path.join(ENGINE_NODES, name)) as handle:
                sources[name] = handle.read()
    everything = "\n".join(sources.values())

    # Every node type the schema can emit must exist in the engine's factory.
    recipe_header = os.path.join(os.path.dirname(ENGINE_NODES), "io", "Recipe.h")
    with open(recipe_header) as handle:
        recipe_source = handle.read()

    for node_type in G.PARAM_MODELS:
        R.check(f'"{node_type}"' in recipe_source,
                f"engine's makeNode knows the node type {node_type!r}")

    # And the master-only set must match exactly, in both directions.
    for node_type in G.MASTER_ONLY:
        R.check(f'type == "{node_type}"' in recipe_source,
                f"engine agrees {node_type!r} is master-only")
    engine_master_only = set(
        re.findall(r'return type == "(\w+)" \|\| type == "(\w+)"', recipe_source)[0]
    ) if re.findall(r'return type == "(\w+)" \|\| type == "(\w+)"', recipe_source) else set()
    R.check(engine_master_only == set(G.MASTER_ONLY),
            f"master-only sets match exactly: engine {engine_master_only}, "
            f"schema {set(G.MASTER_ONLY)}")

    # Every parameter name the schema emits must be one the engine reads.
    # A name the engine never looks up is silently ignored — the graph renders,
    # and the parameter the planner reasoned about does nothing.
    engine_params = set(re.findall(r'(?:p|b|band)\.(?:num|str)\("([a-z_0-9]+)"',
                                  everything))

    # A list parameter is read through ParamSet::items() rather than by name, so
    # it will never appear in a num()/str() lookup. That is the engine's one
    # ordered sub-list, and the name is chosen by the recipe layer.
    def is_list_field(model, name) -> bool:
        annotation = model.model_fields[name].annotation
        return getattr(annotation, "__origin__", None) is list

    R.check("items()" in everything,
            "the engine reads its ordered sub-list through items()")

    for node_type, model in G.PARAM_MODELS.items():
        for name in model.model_fields:
            if is_list_field(model, name):
                continue
            R.check(name in engine_params,
                    f"{node_type}.{name} is a parameter the engine reads")

    # Every band model, not just the EQ ones. A list field is skipped above
    # because it is read through items() rather than by name, so without this
    # loop a new band type's parameters would never be checked against the
    # engine at all — which is the drift this whole section exists to catch.
    for band_model in (G.EqBand, G.DynamicEqBand, G.MultibandBand):
        for name in band_model.model_fields:
            R.check(name in engine_params,
                    f"{band_model.__name__}.{name} is read by the engine")

    # Filter shapes must match the engine's closed set exactly. Read from
    # isKnownFilterShape in Recipe.h rather than from the node's dispatch, because
    # the node returns Bell as its fall-through and so never compares against
    # "bell" — reading the dispatch would silently miss the default shape.
    engine_shapes = set(re.findall(
        r'shape == "(\w+)"',
        recipe_source[recipe_source.index("isKnownFilterShape"):][:400]))
    schema_shapes = set(G.FilterType.__args__)
    R.check(schema_shapes == engine_shapes,
            f"filter shapes match: engine {sorted(engine_shapes)}, "
            f"schema {sorted(schema_shapes)}")


def test_bounds_are_visible_to_the_model() -> None:
    print("[2] bounds appear in prose, not just in keywords")

    schema = G.json_schema()

    # THE CENTRAL ASSERTION OF THIS FILE. Numeric keywords are stripped, because
    # leaving them in would imply a guarantee the API does not provide.
    def find_keywords(node, path=""):
        found = []
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
                    found.append(f"{path}.{key}")
                found += find_keywords(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, item in enumerate(node):
                found += find_keywords(item, f"{path}[{i}]")
        return found

    leftover = find_keywords(schema)
    R.check(not leftover,
            f"the emitted schema carries no unenforced numeric keywords, found {leftover}")

    for name, sub in G.param_schemas().items():
        R.check(not find_keywords(sub), f"{name} params carry none either")

    # Every ranged parameter states its range in words, where the model reads it.
    missing = []
    for node_type, model in G.PARAM_MODELS.items():
        for field_name, info in model.model_fields.items():
            has_bound = any(
                getattr(m, "ge", None) is not None or getattr(m, "le", None) is not None
                for m in info.metadata
            )
            if not has_bound:
                continue
            text = info.description or ""
            if not re.search(r"Range -?\d+(?:\.\d+)? to -?\d+(?:\.\d+)?", text):
                missing.append(f"{node_type}.{field_name}")
    R.check(not missing, f"every bounded field states its range in prose; missing {missing}")

    # And the prose must agree with the enforced bound. Two copies of a number is
    # exactly where drift happens, so this asserts they came from one source.
    disagreements = []
    for node_type, model in G.PARAM_MODELS.items():
        for field_name, info in model.model_fields.items():
            stated = re.search(r"Range (-?\d+(?:\.\d+)?) to (-?\d+(?:\.\d+)?)",
                               info.description or "")
            if not stated:
                continue
            low = next((m.ge for m in info.metadata if getattr(m, "ge", None) is not None), None)
            high = next((m.le for m in info.metadata if getattr(m, "le", None) is not None), None)
            if low is None or high is None:
                continue
            if abs(float(stated.group(1)) - low) > 1e-9 or abs(float(stated.group(2)) - high) > 1e-9:
                disagreements.append(
                    f"{node_type}.{field_name}: prose {stated.groups()} vs enforced ({low}, {high})")
    R.check(not disagreements, f"prose matches enforcement; {disagreements}")

    R.check(schema.get("additionalProperties") is False,
            "the graph rejects unknown top-level keys")


def test_schema_rejects() -> None:
    print("[3] schema rejection")

    def rejects(build, what: str) -> None:
        try:
            build()
            R.check(False, f"should have rejected {what}")
        except Exception:
            R.check(True, f"rejects {what}")

    rejects(lambda: G.GraphNode(id="x", type="tape_saturation"), "an unknown node type")
    rejects(lambda: G.GraphNode(id="Bad Id", type="gain"), "an id with spaces or capitals")
    rejects(lambda: G.GraphNode(id="", type="gain"), "an empty id")
    rejects(lambda: node("gain", gain_db=400.0), "a gain far out of range")
    rejects(lambda: node("gain", gain_dB=3.0), "a misspelt parameter name")
    rejects(lambda: eq("e", {"filter_type": "high_shelf", "center_freq_hz": 5000.0,
                             "gain_db": 3.0, "q_factor": 0.7}),
            "a misspelt filter type")
    rejects(lambda: G.EffectGraph(nodes=[node("gain", "a"), node("pan", "a")]),
            "duplicate node ids")
    rejects(lambda: G.EffectGraph(nodes=[node("true_peak_limiter", "l")]),
            "a limiter in the internal chain")
    rejects(lambda: G.EffectGraph(master=[node("gain", "g")]),
            "a gain in the master section")
    rejects(lambda: G.EffectGraph(nodes=[node("gain", f"g{i}") for i in range(13)]),
            "more nodes than the graph allows")

    # A gain on a band type that has no gain is normalised rather than rejected:
    # it signals the model misread the band type, and silently ignoring it would
    # leave a parameter in the graph that the user can see and the engine cannot
    # act on.
    filtered = G.EqBand(filter_type="highpass", center_freq_hz=80.0, gain_db=6.0)
    R.check(filtered.gain_db == 0.0,
            "a gain on a highpass is zeroed rather than carried")


def test_range_repair() -> None:
    print("[4] range clamping")

    # A CUT, so the clamp is isolated. A boost would also trip the boost budget
    # and the final value would reflect both checks — see below, where that
    # composition is asserted deliberately rather than tripped over.
    graph = G.EffectGraph(nodes=[eq("e", band(1000.0, -6.0))])
    # Bypass Pydantic to simulate a graph patched in place, which is the path
    # this check actually protects.
    graph.nodes[0].params["bands"][0]["gain_db"] = -40.0
    graph.nodes[0].params["bands"][0]["center_freq_hz"] = 1e9

    result = validator.validate(graph)
    got = result.graph.nodes[0].params["bands"][0]
    R.check(got["gain_db"] == -18.0, f"gain clamped to -18, got {got['gain_db']}")
    R.check(got["center_freq_hz"] == 20000.0,
            f"frequency clamped to 20 kHz, got {got['center_freq_hz']}")
    R.check(len(result.repairs) >= 2, "and both clamps were logged")
    R.check(result.ok, "clamping is repairable, not fatal")

    # THE CHECKS COMPOSE, and the order is load-bearing. A 40 dB boost is first
    # clamped to the 18 dB parameter limit, and the boost budget then scales that
    # 18 dB down to the 12 dB total. Asserting only the clamp would pass today
    # and break the moment the budget threshold moved, so both steps are named.
    boosted = G.EffectGraph(nodes=[eq("e", band(1000.0, 6.0))])
    boosted.nodes[0].params["bands"][0]["gain_db"] = 40.0
    chained = validator.validate(boosted)
    R.check(any("clamped to 18" in r for r in chained.repairs),
            f"the parameter clamp fired first, repairs {chained.repairs}")
    # A band gain is bounded at 18 while a gain NODE is bounded at 24. The
    # clamp must use the band's bound, which a name-keyed lookup cannot do.
    node_gain = G.EffectGraph(nodes=[node("gain", "g", gain_db=6.0)])
    node_gain.nodes[0].params["gain_db"] = 40.0
    R.check(validator.validate(node_gain).graph.nodes[0].params["gain_db"] == 24.0,
            "a gain NODE clamps to its own wider bound of 24")
    R.check(any("budget" in r for r in chained.repairs),
            "then the boost budget scaled it")
    final = chained.graph.nodes[0].params["bands"][0]["gain_db"]
    R.check(abs(final - validator.TOTAL_BOOST_BUDGET_DB) < 0.1,
            f"leaving {validator.TOTAL_BOOST_BUDGET_DB:g} dB, got {final}")

    nan_graph = G.EffectGraph(nodes=[node("gain", "g", gain_db=3.0)])
    nan_graph.nodes[0].params["gain_db"] = float("nan")
    nan_result = validator.validate(nan_graph)
    R.check(nan_result.graph.nodes[0].params["gain_db"] == 0.0,
            "a non-finite value is replaced with the default")


def test_boost_budget() -> None:
    print("[5] total boost budget")

    # Four 5 dB boosts: each defensible alone, 20 dB together. This is the
    # commonest cause of a chain that sounds like an EQ rather than a record.
    graph = G.EffectGraph(nodes=[eq(
        "e", band(120.0, 5.0), band(900.0, 5.0), band(3000.0, 5.0), band(9000.0, 5.0),
    )])
    result = validator.validate(graph)
    total = sum(b["gain_db"] for b in result.graph.nodes[0].params["bands"])
    R.check(abs(total - validator.TOTAL_BOOST_BUDGET_DB) < 0.1,
            f"20 dB of boosts scaled to the {validator.TOTAL_BOOST_BUDGET_DB:g} dB "
            f"budget, got {total:.1f}")

    # SCALED, NOT CLIPPED. The shape the planner chose has to survive; clipping
    # the largest band would silently change the tonal balance it reasoned about.
    gains = [b["gain_db"] for b in result.graph.nodes[0].params["bands"]]
    R.check(max(gains) - min(gains) < 0.1,
            f"four equal boosts stay equal after scaling, got {gains}")

    # CUTS DO NOT CONSUME THE BUDGET. Three 4 dB cuts still sound like the
    # source; three 4 dB boosts do not. The asymmetry is deliberate.
    cuts = G.EffectGraph(nodes=[eq(
        "e", band(120.0, -6.0), band(900.0, -6.0), band(3000.0, -6.0),
    )])
    # Named rather than "no repairs at all": validation also guarantees a
    # delivery ceiling on every graph, so the list is never empty and asserting
    # that it is would test the wrong thing.
    cut_result = validator.validate(cuts)
    R.check(not budget_repairs(cut_result),
            f"18 dB of cuts is untouched, repairs were {cut_result.repairs}")

    under = G.EffectGraph(nodes=[eq("e", band(1000.0, 4.0), band(5000.0, 3.0))])
    R.check(not budget_repairs(validator.validate(under)), "7 dB of boosts passes")


def budget_repairs(result) -> list[str]:
    """Repairs about the boost budget, ignoring ones about anything else."""
    return [r for r in result.repairs if "budget" in r or "scaled" in r]


def test_filter_stability() -> None:
    print("[6] filter stability")

    # A high Q at a very low frequency puts the pole on the unit circle. It does
    # not blow up — the engine's topology is unconditionally stable — but it
    # rings for seconds, which is the audible problem.
    graph = G.EffectGraph(nodes=[eq("e", band(22.0, 3.0, q=12.0))])
    result = validator.validate(graph)
    R.check(result.graph.nodes[0].params["bands"][0]["q_factor"] < 12.0,
            "an extreme Q at 22 Hz is reduced")
    R.check(any("rings" in r for r in result.repairs), "and the reason is logged")

    sane = G.EffectGraph(nodes=[eq("e", band(1000.0, 3.0, q=2.0))])
    R.check(not any("rings" in r for r in validator.validate(sane).repairs),
            "an ordinary bell is untouched")


def test_chain_order() -> None:
    print("[7] chain order")

    bad = G.EffectGraph(nodes=[
        node("plate_reverb", "rev", decay_s=2.0),
        node("delay", "dly", time_ms=300.0),
    ])
    result = validator.validate(bad)
    R.check(not result.ok, "a delay after reverb is fatal")
    R.check(any(f.code == "chain_order" for f in result.fatal), "and named as such")

    good = G.EffectGraph(nodes=[
        node("delay", "dly", time_ms=300.0),
        node("plate_reverb", "rev", decay_s=2.0),
    ])
    R.check(validator.validate(good).ok, "the right order passes")

    drive_first = G.EffectGraph(nodes=[
        node("drive", "d", drive_db=6.0),
        node("dc_block", "dc"),
    ])
    R.check(not validator.validate(drive_first).ok,
            "saturating before removing DC is fatal")

    # The master section has exactly one right order, so it is sorted rather than
    # rejected — there is no intent to guess at.
    unsorted = G.EffectGraph(master=[
        node("dither", "dth", bit_depth=24.0),
        node("true_peak_limiter", "lim", ceiling_dbtp=-1.0),
    ])
    sorted_result = validator.validate(unsorted)
    R.check([n.type for n in sorted_result.graph.master]
            == ["true_peak_limiter", "dither"],
            "the master section is reordered to limiter then dither")
    R.check(sorted_result.ok, "and that is a repair, not a rejection")


def test_source_aware_checks() -> None:
    print("[8] checks that need the measurement report")

    clipped = {"warnings": [{"code": "clipped_source", "severity": "high",
                             "message": ""}]}
    graph = G.EffectGraph(nodes=[node("drive", "d", drive_db=8.0)])
    R.check(not validator.validate(graph, clipped).ok,
            "drive on a clipped source is fatal")
    R.check(validator.validate(graph, {}).ok,
            "and fine on a source with no such warning")

    # An air shelf above a codec wall is inaudible. The user hears no change and
    # concludes the tool does not work.
    lopped = {"spectrum": {"bandwidth_limit_hz": 15000.0,
                           "bandwidth_limit_confidence": 0.8}}
    air = G.EffectGraph(nodes=[eq("e", band(16000.0, 4.0, kind="highshelf"),
                                 band(2000.0, -2.0))])
    air_result = validator.validate(air, lopped)
    remaining = [b["center_freq_hz"] for b in air_result.graph.nodes[0].params["bands"]]
    R.check(remaining == [2000.0],
            f"the inaudible band is dropped and the audible one kept, got {remaining}")

    # But not when the detector was unsure — a low-confidence limit must never
    # drive a decision.
    unsure = {"spectrum": {"bandwidth_limit_hz": 15000.0,
                           "bandwidth_limit_confidence": 0.2}}
    R.check(len(validator.validate(air, unsure).graph.nodes[0].params["bands"]) == 2,
            "a low-confidence bandwidth limit does not drop bands")

    # Width on dual mono cannot do anything.
    dual = {"source": {"channels": 2}, "stereo": {"type": "dual_mono",
                                                  "correlation": 1.0}}
    width = G.EffectGraph(nodes=[node("stereo_width", "w", width_percent=160.0)])
    R.check(any(f.code == "noop_width" for f in validator.validate(width, dual).findings),
            "width on dual mono is flagged as a no-op")

    real_stereo = {"source": {"channels": 2},
                   "stereo": {"type": "conventional_stereo", "correlation": 0.6}}
    R.check(not any(f.code == "noop_width"
                    for f in validator.validate(width, real_stereo).findings),
            "and not flagged on real stereo")

    # An unaddressed polarity problem is fatal: the mono sum currently cancels.
    phase = {"warnings": [{"code": "out_of_phase", "severity": "critical",
                           "message": ""}]}
    R.check(not validator.validate(G.EffectGraph(nodes=[eq("e", band(1000.0, 2.0))]),
                                   phase).ok,
            "ignoring an out-of-phase source is fatal")
    R.check(validator.validate(
        G.EffectGraph(nodes=[node("polarity", "p", inverted=True)]), phase).ok,
        "addressing it passes")

    # A boost over an audible noise floor lifts hiss into the foreground.
    noisy = {"hygiene": {"noise_floor_dbfs": -42.0, "noise_floor_confidence": 0.9}}
    R.check(any(f.code == "boost_over_noise" for f in validator.validate(
        G.EffectGraph(nodes=[eq("e", band(12000.0, 6.0, kind="highshelf"))]),
        noisy).findings),
        "a high boost over a loud noise floor is flagged")


def test_gain_staging() -> None:
    print("[9] gain staging")

    hot = {"dynamics": {"short_term_rms_percentiles": {"p95_dbfs": -6.0}}}
    graph = G.EffectGraph(nodes=[
        node("gain", "g", gain_db=12.0),
        node("compressor", "c", threshold_dbfs=-20.0, makeup_gain_db=12.0),
    ])
    result = validator.validate(graph, hot)
    R.check(not result.ok, "a chain projecting 18 dB over is fatal")
    R.check(any(f.code == "gain_staging" for f in result.fatal), "and named")

    modest = G.EffectGraph(nodes=[node("gain", "g", gain_db=2.0)])
    R.check(validator.validate(modest, hot).ok, "a modest chain passes")

    # Without a report the check is skipped rather than guessed at: an invented
    # input level would produce invented warnings.
    R.check(validator.validate(graph, None).ok,
            "the check is skipped when there is no measurement report")


def test_renders() -> None:
    print("[10] a valid graph actually renders")

    from backend.engine.render import DEFAULT_BINARY, render_array

    if not os.path.exists(DEFAULT_BINARY):
        print("  engine binary missing; skipping")
        return

    graph = G.EffectGraph(
        nodes=[
            eq("shape", band(90.0, 0.0, q=0.707, kind="highpass"),
               band(320.0, -2.5, q=1.1), band(6000.0, 3.0, q=0.707, kind="highshelf")),
            node("compressor", "comp", threshold_dbfs=-24.0, ratio_to_1=3.0,
                 attack_ms=12.0, release_ms=140.0, makeup_gain_db=2.0),
            G.GraphNode(id="plate", type="plate_reverb", mix=0.18,
                        params={"decay_s": 1.4, "pre_delay_ms": 18.0,
                                "damping_hz": 6000.0}),
        ],
        master=[
            node("true_peak_limiter", "lim", ceiling_dbtp=-1.0, lookahead_ms=5.0),
            node("dither", "dth", bit_depth=24.0),
        ],
    )

    result = validator.validate(graph)
    R.check(result.ok, f"the graph validates: {result.summary()}")

    recipe = result.graph.to_recipe(seed=7, output_sample_rate=48000.0,
                                    output_format="float64")
    R.check(recipe["seed"] == 7, "the recipe carries the seed")
    R.check([n["type"] for n in recipe["master"]] == ["true_peak_limiter", "dither"],
            "and the master order")

    rng = np.random.default_rng(5)
    source = rng.normal(0.0, 0.2, (2, 48000)).astype(np.float64)
    out = render_array(source, 48000.0, recipe)

    R.check(out.shape == source.shape, f"the render came back, shape {out.shape}")
    R.check(np.all(np.isfinite(out)), "with no NaN or infinite samples")
    R.check(float(np.max(np.abs(out))) > 0.0, "and it is not silent")

    # The graph carries no rate or format, so the same graph must render at a
    # different delivery rate without being regenerated.
    other = result.graph.to_recipe(seed=7, output_sample_rate=44100.0,
                                    output_format="pcm24")
    out2 = render_array(source, 48000.0, other)
    R.check(out2.shape[0] == 2, "and the same graph renders at 44.1 kHz too")


if __name__ == "__main__":
    print("Audionova effect graph schema\n")
    test_agrees_with_engine()
    test_bounds_are_visible_to_the_model()
    test_schema_rejects()
    test_range_repair()
    test_boost_budget()
    test_filter_stability()
    test_chain_order()
    test_source_aware_checks()
    test_gain_staging()
    test_renders()
    print(f"\n{R.checks} checks, {R.failures} failure(s)")
    sys.exit(1 if R.failures else 0)
