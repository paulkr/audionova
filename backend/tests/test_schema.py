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


def bands_of(result, node_id: str = "e") -> list[dict]:
    node_out = next(n for n in result.graph.nodes if n.id == node_id)
    return node_out.params.get("bands", []) or []


def master_of(result, node_type: str):
    return next(n for n in result.graph.master if n.type == node_type)


def pole_radius(freq: float, q: float, sample_rate: float = 96000.0) -> float:
    """Bilinear-transform pole radius, written out here rather than imported.

    The validator's own copy is the thing under test, so an assertion that
    called it would agree with any version of it.
    """
    omega = 2.0 * math.pi * freq / sample_rate
    alpha = math.sin(omega) / (2.0 * q)
    return math.sqrt(max(0.0, (1.0 - alpha) / (1.0 + alpha)))


def report_at(p95: float) -> dict:
    return {"dynamics": {"short_term_rms_percentiles": {"p95_dbfs": p95}}}


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

    # THE FIELD'S OWN DEFAULT, not zero — and only a field whose default is not
    # zero can tell the two apart. A gain's default happens to be 0.0, so the
    # check above passes just as well against `params[key] = 0.0`, which for a
    # ratio would write a value the schema itself rejects.
    ratio = G.EffectGraph(nodes=[node("compressor", "c", ratio_to_1=4.0)])
    ratio.nodes[0].params["ratio_to_1"] = float("inf")
    repaired_ratio = validator.validate(ratio).graph.nodes[0].params["ratio_to_1"]
    R.check(repaired_ratio == 3.0,
            f"a non-finite ratio goes back to the ratio's default of 3, got "
            f"{repaired_ratio}")

    # A DYNAMIC EQ'S BANDS ARE BOUNDED BY DynamicEqBand, not by EqBand. The two
    # models do not even share a field list — `range_db` exists on one and not
    # the other — so a clamp reading the wrong model does not clamp it at all.
    dyn = G.GraphNode(id="de", type="dynamic_eq", params={"bands": [
        {"filter_type": "bell", "center_freq_hz": 300.0, "static_gain_db": 2.0,
         "range_db": 6.0}]})
    dyn_graph = G.EffectGraph(nodes=[dyn])
    dyn_graph.nodes[0].params["bands"][0]["range_db"] = 40.0
    got_range = bands_of(validator.validate(dyn_graph), "de")[0]["range_db"]
    R.check(got_range == 18.0,
            f"a dynamic EQ band's range_db clamps to 18, got {got_range}")

    # And the master section is clamped too. Its nodes go through the same
    # renderer as everything else and nothing else bounds them.
    master_hot = G.EffectGraph(master=[node("dither", "dth", bit_depth=24.0)])
    master_hot.master[0].params["amount_lsb"] = 9.0
    got_lsb = master_of(validator.validate(master_hot), "dither").params["amount_lsb"]
    R.check(got_lsb == 4.0, f"a master node's parameter is clamped too, got {got_lsb}")


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

    # THE BAR IS PROBED FROM BOTH SIDES, AND THE NUMBER IS WRITTEN HERE. Every
    # assertion above compares against `validator.TOTAL_BOOST_BUDGET_DB`, so the
    # budget and the expectation move together: it could be 8 dB or 16 dB and
    # all of them would still pass. 11 dB and 13 dB pin it to the dB.
    just_under = G.EffectGraph(nodes=[eq("e", band(200.0, 5.5), band(4000.0, 5.5))])
    R.check(not budget_repairs(validator.validate(just_under)),
            "11 dB of boosts is inside the 12 dB budget and untouched")
    just_over = G.EffectGraph(nodes=[eq("e", band(200.0, 6.5), band(4000.0, 6.5))])
    over_result = validator.validate(just_over)
    R.check(budget_repairs(over_result), "13 dB of boosts is over it and scaled")
    over_total = sum(b["gain_db"] for b in bands_of(over_result))
    R.check(abs(over_total - 12.0) < 0.05,
            f"and lands on 12 dB, got {over_total:.2f}")

    # Exactly at the budget is not over it.
    exact = G.EffectGraph(nodes=[eq("e", band(200.0, 6.0), band(4000.0, 6.0))])
    R.check(not budget_repairs(validator.validate(exact)),
            "exactly 12 dB is at the budget, not past it")

    # Scaling has to LAND on the budget, not near it. Four equal 5 dB boosts
    # scale to 3.0 each whether the result is rounded to 0.01 dB or to whole
    # decibels, so a set that does not divide evenly is what tests the rounding.
    awkward = G.EffectGraph(nodes=[eq(
        "e", band(200.0, 5.0), band(1500.0, 5.0), band(7000.0, 7.0))])
    awkward_total = sum(b["gain_db"] for b in bands_of(validator.validate(awkward)))
    R.check(abs(awkward_total - 12.0) < 0.05,
            f"17 dB of awkward boosts scales onto the budget, got {awkward_total:.2f}")

    # THE BUDGET IS A WHOLE-GRAPH TOTAL. Two EQs of 8 dB each is 16 dB of boost
    # arriving at the listener, however it is spread across nodes.
    two_nodes = G.EffectGraph(nodes=[eq("a", band(200.0, 8.0)),
                                     eq("b", band(3000.0, 8.0))])
    two_result = validator.validate(two_nodes)
    spread_gains = [b["gain_db"] for n in two_result.graph.nodes
                    for b in n.params.get("bands", [])]
    R.check(abs(sum(spread_gains) - 12.0) < 0.05
            and abs(spread_gains[0] - spread_gains[1]) < 0.05,
            f"16 dB across two nodes scales to 12, got {spread_gains}")

    # A cut sitting alongside a boost does not add to it. `sum(abs(gain))` is
    # the plausible-looking version of this check and it is the wrong one.
    mixed = G.EffectGraph(nodes=[eq("e", band(120.0, -8.0), band(900.0, 8.0))])
    R.check(not budget_repairs(validator.validate(mixed)),
            "8 dB of boost beside 8 dB of cut is 8 dB of boost, not 16")

    # THE ORDER AGAINST NO-OP REMOVAL IS LOAD-BEARING, and this is the only
    # thing that asserts it: a band about to be dropped as inaudible must not
    # consume budget on its way out and shrink the bands that survive.
    lopped = {"spectrum": {"bandwidth_limit_hz": 15000.0,
                           "bandwidth_limit_confidence": 0.8}}
    doomed = G.EffectGraph(nodes=[eq("e", band(300.0, 5.0), band(2000.0, 5.0),
                                     band(16000.0, 6.0, kind="highshelf"))])
    survivors = [b["gain_db"] for b in bands_of(validator.validate(doomed, lopped))]
    R.check(survivors == [5.0, 5.0],
            f"an inaudible band does not spend budget before being dropped, "
            f"got {survivors}")


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
    repaired_q = bands_of(result)[0]["q_factor"]
    R.check(repaired_q < 12.0, "an extreme Q at 22 Hz is reduced")
    R.check(any("rings" in r for r in result.repairs), "and the reason is logged")

    # THE REPAIR HAS TO REACH THE BAR IT NAMED, and for a long time it did not.
    # Q 12 at 22 Hz puts the pole at 0.999940; the repair was `q / 4`, which
    # leaves it at 0.999760 — still past the 0.9995 the message calls audible
    # ringing. A repair that logs a fix without delivering one is worse than no
    # repair, because the log now says the graph is safe.
    R.check(pole_radius(22.0, repaired_q) <= validator.MAX_POLE_RADIUS,
            f"and the repaired Q {repaired_q:g} clears the bar: pole at "
            f"{pole_radius(22.0, repaired_q):.6f} against "
            f"{validator.MAX_POLE_RADIUS}")

    # It also has to be the WIDEST Q that clears the bar rather than a token
    # value, or the repair throws away the move the planner asked for. At 22 Hz
    # the pole crosses the bar at Q 1.44.
    R.check(1.0 < repaired_q <= 1.44,
            f"and is the widest Q that does, got {repaired_q:g}")

    # THE BAR PROBED FROM BOTH SIDES. The pole radius crosses 0.9995 at Q 6.54
    # at 100 Hz, so Q 8 must be repaired and Q 5 must not. Q 12 at 22 Hz alone
    # is far past the bar and leaves the constant free to sit anywhere from
    # 0.998 to 0.99999 — and to be computed at the wrong sample rate, which
    # halves the pole's distance from the circle and changes no verdict at all
    # on a probe that extreme.
    hot = G.EffectGraph(nodes=[eq("e", band(100.0, 3.0, q=8.0))])
    R.check(bands_of(validator.validate(hot))[0]["q_factor"] < 8.0,
            "Q 8 at 100 Hz is past the bar and reduced")
    cool_result = validator.validate(G.EffectGraph(nodes=[eq("e", band(100.0, 3.0, q=5.0))]))
    R.check(bands_of(cool_result)[0]["q_factor"] == 5.0,
            "Q 5 at 100 Hz is inside it and untouched")
    # And not merely unchanged — untouched. Solving for the widest safe Q
    # returns the original value whenever the original was already safe, so the
    # value alone cannot tell a check that did not fire from one that fired and
    # had nothing to do. The repair log can.
    R.check(not any("rings" in r for r in cool_result.repairs),
            f"with no repair logged, got {cool_result.repairs}")

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

    # EVERY RULE, not two of the five. A rule nobody probes is a rule that can
    # be deleted, reversed or pointed at the wrong node type without any test
    # noticing — and three of them were in exactly that position.
    #
    # WRITTEN OUT HERE rather than read from `validator.ORDER_RULES`. A loop
    # over the module's own table cannot see a rule being deleted, because the
    # case that would have caught it disappears with it.
    expected_rules = (
        ("dc_block", "drive"),
        ("drive", "convolution_reverb"),
        ("mono_to_stereo", "stereo_width"),
        ("mid_side", "stereo_width"),
        ("delay", "plate_reverb"),
    )
    R.check(tuple((a, b) for a, b, _ in validator.ORDER_RULES) == expected_rules,
            f"the ordering table is the five rules named here, got "
            f"{[(a, b) for a, b, _ in validator.ORDER_RULES]}")

    for earlier, later in expected_rules:
        wrong = G.EffectGraph(nodes=[node(later, "later"), node(earlier, "earlier")])
        R.check(any(f.code == "chain_order" for f in validator.validate(wrong).fatal),
                f"{later} before {earlier} is fatal")
        right = G.EffectGraph(nodes=[node(earlier, "earlier"), node(later, "later")])
        R.check(not any(f.code == "chain_order"
                        for f in validator.validate(right).fatal),
                f"and {earlier} before {later} is not")

    # A type appearing twice is judged by its FIRST position. A drive ahead of
    # the DC blocker is a violation whether or not a second drive follows it,
    # and reading the last position instead silently forgives the first.
    repeated = G.EffectGraph(nodes=[node("drive", "d1", drive_db=6.0),
                                    node("dc_block", "dc"),
                                    node("drive", "d2", drive_db=6.0)])
    R.check(any(f.code == "chain_order" for f in validator.validate(repeated).fatal),
            "the first occurrence of a type is what the ordering rules see")

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


def test_delivery_ceiling() -> None:
    print("[7b] the delivery ceiling")

    # NOTHING HERE ASSERTED THIS EXISTED. The whole excerpt-first design depends
    # on it — a 20 s audition window measured -2.53 dBTP on a file whose peak
    # was +0.52, so the ceiling is what makes the excerpt predict the delivery —
    # and the check could have been deleted outright without a failure.
    bare = validator.validate(G.EffectGraph(nodes=[node("gain", "g", gain_db=1.0)]))
    limiters = [n for n in bare.graph.master if n.type == "true_peak_limiter"]
    R.check(len(limiters) == 1,
            f"a graph with no master section gets a limiter, got "
            f"{[n.type for n in bare.graph.master]}")
    R.check(bare.ok, "and that is a repair, not a rejection")
    R.check(any("ceiling" in r for r in bare.repairs), "logged as a repair")

    # ITS VALUES ARE DECISIONS, so they are asserted rather than left to drift.
    # -1.0 dBTP is the streaming convention; 5 ms of lookahead is what makes the
    # overshoot zero offline.
    R.check(limiters[0].params["ceiling_dbtp"] == -1.0,
            f"at -1.0 dBTP, got {limiters[0].params['ceiling_dbtp']}")
    R.check(limiters[0].params["lookahead_ms"] == 5.0,
            f"with 5 ms of lookahead, got {limiters[0].params['lookahead_ms']}")
    R.check(not limiters[0].bypass, "and not bypassed")

    # A BYPASSED LIMITER IS NOT A CEILING. It renders as nothing at all, so a
    # graph carrying one is exactly as unprotected as a graph carrying none.
    bypassed = G.EffectGraph(master=[G.GraphNode(
        id="lim", type="true_peak_limiter", bypass=True,
        params={"ceiling_dbtp": -1.0})])
    active = [n for n in validator.validate(bypassed).graph.master
              if n.type == "true_peak_limiter" and not n.bypass]
    R.check(len(active) == 1,
            "a bypassed limiter still gets a working one added alongside it")

    # And a graph that already has one keeps its own, unduplicated: the
    # planner's -2.0 dBTP is a decision, not a mistake to be overwritten.
    already = validator.validate(G.EffectGraph(
        master=[node("true_peak_limiter", "lim", ceiling_dbtp=-2.0)]))
    kept = [n for n in already.graph.master if n.type == "true_peak_limiter"]
    R.check(len(kept) == 1, f"an existing ceiling is not duplicated, got {len(kept)}")
    R.check(kept[0].params["ceiling_dbtp"] == -2.0,
            "and its own ceiling is left alone")


def test_source_aware_checks() -> None:
    print("[8] checks that need the measurement report")

    clipped = {"warnings": [{"code": "clipped_source", "severity": "high",
                             "message": ""}]}
    graph = G.EffectGraph(nodes=[node("drive", "d", drive_db=8.0)])
    R.check(not validator.validate(graph, clipped).ok,
            "drive on a clipped source is fatal")
    R.check(validator.validate(graph, {}).ok,
            "and fine on a source with no such warning")

    # ANY drive, not a lot of drive. The bar is "the saturator is doing
    # something at all", so 1 dB has to trip it as surely as 8 does — and with
    # only the 8 dB case the bar could have been anywhere below it.
    R.check(not validator.validate(
        G.EffectGraph(nodes=[node("drive", "d", drive_db=1.0)]), clipped).ok,
        "even 1 dB of drive on a clipped source is fatal")
    R.check(validator.validate(
        G.EffectGraph(nodes=[node("drive", "d", drive_db=0.0)]), clipped).ok,
        "and a drive node set to no drive at all is not")

    # An air shelf above a codec wall is inaudible. The user hears no change and
    # concludes the tool does not work.
    lopped = {"spectrum": {"bandwidth_limit_hz": 15000.0,
                           "bandwidth_limit_confidence": 0.8}}
    air = G.EffectGraph(nodes=[eq("e", band(16000.0, 4.0, kind="highshelf"),
                                 band(2000.0, -2.0))])
    air_result = validator.validate(air, lopped)
    remaining = [b["center_freq_hz"] for b in bands_of(air_result)]
    R.check(remaining == [2000.0],
            f"the inaudible band is dropped and the audible one kept, got {remaining}")

    # But not when the detector was unsure — a low-confidence limit must never
    # drive a decision.
    unsure = {"spectrum": {"bandwidth_limit_hz": 15000.0,
                           "bandwidth_limit_confidence": 0.2}}
    R.check(len(bands_of(validator.validate(air, unsure))) == 2,
            "a low-confidence bandwidth limit does not drop bands")

    # 0.8 against 0.2 is a factor of two either side of the bar, which leaves it
    # free to sit anywhere between them. 0.45 and 0.35 pin it.
    for confidence, wanted in ((0.45, 1), (0.35, 2)):
        probe = {"spectrum": {"bandwidth_limit_hz": 15000.0,
                              "bandwidth_limit_confidence": confidence}}
        kept = len(bands_of(validator.validate(air, probe)))
        R.check(kept == wanted,
                f"bandwidth-limit confidence {confidence}: {kept} band(s) kept, "
                f"wanted {wanted}")

    # And the limit itself is the bar, not some fraction below it: a band a
    # kilohertz under the wall is audible and must survive.
    below = G.EffectGraph(nodes=[eq("e", band(14000.0, 4.0, kind="highshelf"))])
    R.check(len(bands_of(validator.validate(below, lopped))) == 1,
            "a band below the bandwidth limit is kept")
    at_wall = G.EffectGraph(nodes=[eq("e", band(15000.0, 4.0, kind="highshelf"))])
    R.check(len(bands_of(validator.validate(at_wall, lopped))) == 0,
            "and one at the limit is dropped")

    # A CUT above the wall is exactly as inaudible as a boost.
    cut_above = G.EffectGraph(nodes=[eq("e", band(16000.0, -5.0, kind="highshelf"))])
    R.check(len(bands_of(validator.validate(cut_above, lopped))) == 0,
            "a cut above the limit is dropped too")

    # BUT A FILTER WITH NO GAIN IS NOT A GAIN MOVE, and the rule is about gain
    # moves. A highpass at 16 kHz removes everything below 16 kHz — the most
    # audible thing in the graph — and its `gain_db` is zero because a highpass
    # has no gain. Dropping it as inaudible would silently delete the loudest
    # change in the chain.
    steep = G.EffectGraph(nodes=[eq("e", band(16000.0, 0.0, kind="highpass"))])
    R.check(len(bands_of(validator.validate(steep, lopped))) == 1,
            "a highpass above the limit is kept: it has no gain, but it is not "
            "doing nothing")

    # Width on dual mono cannot do anything.
    dual = {"source": {"channels": 2}, "stereo": {"type": "dual_mono",
                                                  "correlation": 1.0}}
    width = G.EffectGraph(nodes=[node("stereo_width", "w", width_percent=160.0)])

    def is_noop_width(report: dict, graph_in=None) -> bool:
        return any(f.code == "noop_width" for f in
                   validator.validate(graph_in or width, report).findings)

    R.check(is_noop_width(dual), "width on dual mono is flagged as a no-op")

    real_stereo = {"source": {"channels": 2},
                   "stereo": {"type": "conventional_stereo", "correlation": 0.6}}
    R.check(not is_noop_width(real_stereo), "and not flagged on real stereo")

    # THE CORRELATION BAR IS 0.999 AND NOTHING WENT NEAR IT: 1.0 against 0.6 is
    # most of the range, and the constant could have been 0.7 or 0.99999.
    R.check(is_noop_width({"source": {"channels": 2},
                           "stereo": {"type": "conventional_stereo",
                                      "correlation": 0.9995}}),
            "correlation 0.9995 is mono in all but name and is flagged")
    R.check(not is_noop_width({"source": {"channels": 2},
                               "stereo": {"type": "conventional_stereo",
                                          "correlation": 0.99}}),
            "correlation 0.99 still has a side signal and is not")

    # A mono file has no side signal by construction.
    R.check(is_noop_width({"source": {"channels": 1}, "stereo": {}}),
            "a mono source is flagged")

    # But width left at 100% is not asking for anything, so there is nothing to
    # warn about — flagging it would put a finding in front of the user for a
    # node that is already doing what they wanted.
    unity = G.EffectGraph(nodes=[node("stereo_width", "w", width_percent=100.0)])
    R.check(not is_noop_width(dual, unity),
            "width left at 100% on dual mono is not a no-op worth reporting")

    # A bypassed node is legal and pointless, and nothing asserted it was
    # reported at all.
    idle = G.EffectGraph(nodes=[G.GraphNode(id="c", type="compressor", bypass=True,
                                            params={"ratio_to_1": 3.0})])
    idle_result = validator.validate(idle)
    R.check(any(f.code == "bypassed_node" for f in idle_result.findings),
            "a node emitted bypassed is reported")
    R.check(idle_result.ok, "as repairable rather than fatal")

    # An unaddressed polarity problem is fatal: the mono sum currently cancels.
    phase = {"warnings": [{"code": "out_of_phase", "severity": "critical",
                           "message": ""}]}
    R.check(not validator.validate(G.EffectGraph(nodes=[eq("e", band(1000.0, 2.0))]),
                                   phase).ok,
            "ignoring an out-of-phase source is fatal")
    R.check(validator.validate(
        G.EffectGraph(nodes=[node("polarity", "p", inverted=True)]), phase).ok,
        "addressing it with polarity passes")
    R.check(validator.validate(
        G.EffectGraph(nodes=[node("mid_side", "ms")]), phase).ok,
        "and addressing it with mid_side passes too")

    # A boost over an audible noise floor lifts hiss into the foreground. Three
    # constants meet here — the floor, the detector's confidence and the air
    # frequency — and one probe sat comfortably past all three at once.
    def lifts_noise(freq: float, gain: float, floor: float = -42.0,
                    confidence: float = 0.9) -> bool:
        report = {"hygiene": {"noise_floor_dbfs": floor,
                              "noise_floor_confidence": confidence}}
        graph_in = G.EffectGraph(nodes=[eq("e", band(freq, gain, kind="highshelf"))])
        return any(f.code == "boost_over_noise"
                   for f in validator.validate(graph_in, report).findings)

    R.check(lifts_noise(12000.0, 6.0), "a high boost over a loud noise floor is flagged")

    R.check(lifts_noise(12000.0, 6.0, floor=-50.0),
            "a -50 dBFS floor is still loud enough to be lifted into the foreground")
    R.check(not lifts_noise(12000.0, 6.0, floor=-60.0),
            "and a -60 dBFS floor is not")

    R.check(lifts_noise(12000.0, 6.0, confidence=0.45),
            "confidence 0.45 is enough to act on")
    R.check(not lifts_noise(12000.0, 6.0, confidence=0.35),
            "and 0.35 is not")

    R.check(lifts_noise(11000.0, 6.0), "11 kHz is in the air band")
    R.check(not lifts_noise(9000.0, 6.0), "and 9 kHz is below it")

    R.check(lifts_noise(12000.0, 3.5), "a 3.5 dB boost is over the bar")
    R.check(not lifts_noise(12000.0, 2.5), "and 2.5 dB is under it")

    # Dither must match the delivery format. Getting this wrong is silent — a
    # dithered float export just has added noise — so nothing downstream will
    # ever complain, and nothing here checked it either.
    odd = G.EffectGraph(master=[node("dither", "dth", bit_depth=24.0)])
    odd.master[0].params["bit_depth"] = 20.0
    fixed = validator.validate(odd)
    R.check(master_of(fixed, "dither").params["bit_depth"] == 24.0,
            f"a 20-bit dither is repaired to 24, got "
            f"{master_of(fixed, 'dither').params['bit_depth']}")
    R.check(any("word length" in r for r in fixed.repairs), "and the repair is logged")

    # WITH AND WITHOUT A REPORT. The repair reads nothing from the measurement,
    # and it lived below the early return that skips this whole function when
    # there is no report — so a graph validated before the measurement landed
    # kept its 20-bit dither.
    with_report = G.EffectGraph(master=[node("dither", "dth", bit_depth=24.0)])
    with_report.master[0].params["bit_depth"] = 20.0
    R.check(master_of(validator.validate(with_report, {"source": {"channels": 2}}),
                      "dither").params["bit_depth"] == 24.0,
            "and repaired the same way when a report is present")

    for depth in (0.0, 16.0, 24.0, 32.0):
        keep = G.EffectGraph(master=[node("dither", "d", bit_depth=depth)])
        got = master_of(validator.validate(keep), "dither").params["bit_depth"]
        R.check(got == depth,
                f"{depth:g} bit is a real delivery word length and is left alone, "
                f"got {got}")


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

    def staging(graph_in, report=hot) -> set[str]:
        return {f.code for f in validator.validate(graph_in, report).findings
                if f.code.startswith("gain_staging")}

    # BOTH BARS, FROM BOTH SIDES. The source is at -6 dBFS, so a gain node's
    # value is the projected level plus six. The suite drove 18 dB past the
    # 12 dB bar and never exercised the -3 dB one at all, so the fatal bar could
    # have been anywhere from 8 to 16 and the tight bar anywhere below it.
    for gain_db, wanted in ((19.0, {"gain_staging"}),          # projects +13
                            (17.0, {"gain_staging_tight"}),    # projects +11
                            (10.0, {"gain_staging_tight"}),    # projects +4
                            (8.0, set())):                     # projects +2
        found = staging(G.EffectGraph(nodes=[node("gain", "g", gain_db=gain_db)]))
        R.check(found == wanted,
                f"a chain projecting {gain_db - 6.0:+.0f} dBFS reports "
                f"{found or 'nothing'}, wanted {wanted or 'nothing'}")

    # EVERY TERM IN THE PROJECTION, probed on its own. Only the gain node and
    # the compressor's makeup were reachable from the old probe; the drive trim
    # could have been dropped entirely.
    trim = G.EffectGraph(nodes=[node("drive", "d", drive_db=6.0,
                                     output_trim_db=19.0)])
    R.check(staging(trim) == {"gain_staging"},
            "a drive node's output trim counts toward the projection")

    # A bypassed node renders as nothing, so it cannot make the limiter's job
    # harder and must not be projected as though it did.
    bypassed = G.EffectGraph(nodes=[G.GraphNode(id="g", type="gain", bypass=True,
                                                params={"gain_db": 24.0})])
    R.check(staging(bypassed) == set(),
            "a bypassed gain contributes nothing to the projection")

    # ONLY THE LARGEST BOOST IN AN EQ, NOT THE SUM. Three 4 dB bands at three
    # different frequencies do not make a 12 dB peak anywhere, and projecting
    # them as one would fail chains that are fine. The code summed them and the
    # comment saying it did not was the only thing asserting otherwise.
    spread = G.EffectGraph(nodes=[eq("e", band(200.0, 4.0), band(1500.0, 4.0),
                                     band(8000.0, 4.0))])
    R.check(staging(spread, report_at(-3.0)) == set(),
            "three 4 dB bands project 4 dB, not 12")

    # An EQ boost does count, at its largest band.
    steep = G.EffectGraph(nodes=[eq("e", band(1000.0, 12.0))])
    R.check(staging(steep) == {"gain_staging_tight"},
            "a single 12 dB band is projected")

    # And a dynamic EQ's static gain is that node's boost, since it has no
    # `gain_db` field at all.
    dyn = G.GraphNode(id="de", type="dynamic_eq", params={"bands": [
        {"filter_type": "bell", "center_freq_hz": 800.0, "static_gain_db": 12.0,
         "range_db": 6.0}]})
    R.check(staging(G.EffectGraph(nodes=[dyn])) == {"gain_staging_tight"},
            "a dynamic EQ's static gain is projected")

    # Boosts DO stack across nodes, though: two EQs lifting the same region are
    # two lifts of the same region.
    stacked = G.EffectGraph(nodes=[eq("a", band(1000.0, 6.0)),
                                   eq("b", band(1000.0, 6.0))])
    R.check(staging(stacked) == {"gain_staging_tight"},
            "two EQs boosting the same region stack")


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
    test_delivery_ceiling()
    test_source_aware_checks()
    test_gain_staging()
    test_renders()
    print(f"\n{R.checks} checks, {R.failures} failure(s)")
    sys.exit(1 if R.failures else 0)
