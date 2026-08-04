"""The local API: what it refuses, and what it must not quietly change.

THE API IS THE ONLY PLACE A PERSON TYPES A NUMBER. Everywhere else in this
project the caller is a model, and a model that overshoots a bound still meant
the direction, so the validator clamps and logs. A person did not mean the
direction; they meant the number. That asymmetry is the reason `edit` exists as
its own endpoint, and it is what most of this file tests.

The single most valuable assertion here is that a bound is read PER MODEL and
never per name. A band's `gain_db` runs to plus or minus 18 and a gain node's to
plus or minus 24, and a lookup keyed on the name alone cannot tell them apart —
it applies the wider one and a 22 dB band gain sails through a check that exists
to stop exactly that. That bug has already been found and fixed once in the
validator (`backend/plan/validate.py`, `_bounds`). The edit endpoint is the
second place it can reappear, so both directions are tested, on two independent
name collisions, plus a sweep over every parameter the fixture graph exposes.

NO MODEL CALLS. Every plan is `planner=rules`, which is deterministic and needs
no key, and the one test that touches the model path removes the key from the
environment first so it cannot reach the network. Audio is synthesised rather
than read from `audio/stems/`, which is gitignored and therefore absent in CI —
a suite that skips its most interesting half on the machine that gates merges is
not doing the job.
"""

from __future__ import annotations

import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi.testclient import TestClient  # noqa: E402

from backend.api import server  # noqa: E402
from backend.audio import wavio  # noqa: E402
from backend.engine.render import DEFAULT_BINARY  # noqa: E402
from backend.plan import ladder  # noqa: E402
from backend.plan.validate import _bounds  # noqa: E402
from backend.schemas.graph import (  # noqa: E402
    PARAM_MODELS, DynamicEqBand, EffectGraph, EqBand,
)

SR = 48000.0
RNG = np.random.default_rng(90210)

# NOT a context manager. Entering one runs the app's lifespan, and this app's
# shutdown hook deletes the working directory every render and upload writes
# into — the second test would then be operating on files that no longer exist.
CLIENT = TestClient(server.app)


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


# ---------------------------------------------------------------------------
# Fixtures


def pink(seconds: float, channels: int = 1, amplitude: float = 0.3) -> np.ndarray:
    n = int(SR * seconds)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    shape = np.ones_like(freqs)
    shape[1:] = 1.0 / np.sqrt(freqs[1:])
    shape[0] = 0.0
    spec = (RNG.normal(size=(channels, freqs.size))
            + 1j * RNG.normal(size=(channels, freqs.size)))
    out = np.fft.irfft(spec * shape, n=n, axis=-1)
    return out / np.max(np.abs(out)) * amplitude


def stereo(seconds: float, amplitude: float) -> np.ndarray:
    """Realistic stereo: a shared mid with a quieter decorrelated side.

    Two independent channels would measure as a pathologically wide source and
    trip the phase constraint in validation, which would make every graph in
    this file fail for a reason that has nothing to do with the API.
    """
    mid = pink(seconds, 1, amplitude)[0]
    side = pink(seconds, 1, amplitude)[0] * 0.4
    return np.vstack([mid + side, mid - side])


def upload(name: str, samples: np.ndarray):
    return CLIENT.post("/api/projects",
                       files={"file": (name, wavio.encode_wav(samples, SR, 24),
                                       "audio/wav")})


def have_engine() -> bool:
    return os.path.exists(DEFAULT_BINARY)


def store(project_id: str, spec: dict) -> str:
    """Put a hand-built graph on a project's tree.

    Built here rather than planned, because the deterministic planner emits
    neither a gain node nor an EQ on synthetic material, and the assertion that
    matters most needs both in one graph so the same parameter NAME is reachable
    under two different models.
    """
    graph = EffectGraph.model_validate(
        {"schema_version": "effect_graph_v1", "master": [],
         "explanation": "fixture", **spec})
    return server._store_graph(server.PROJECTS[project_id], graph, "fixture", None)


def edit(project_id: str, graph_id: str, **body):
    return CLIENT.post(f"/api/projects/{project_id}/graphs/{graph_id}/edit",
                       json=body)


def status(call) -> object:
    """The status code, or the name of whatever the endpoint raised instead.

    A handler that raises where it should return 404 takes the whole suite down
    with it, and a run that stops at the first defect reports one defect per
    round trip. The exception name is compared like any other wrong answer.
    """
    try:
        return call().status_code
    except Exception as error:  # noqa: BLE001
        return type(error).__name__


def snapshot(project_id: str, graph_id: str) -> str:
    """One stored graph, exactly as the API reports it, as a comparable string."""
    body = CLIENT.get(f"/api/projects/{project_id}").json()
    return json.dumps(body["graphs"][graph_id], sort_keys=True)


# A graph reaching for every shape the edit endpoint has to distinguish: the
# same parameter name under two different models (`gain_db` on a band and on a
# gain node, `range_db` on a dynamic EQ band and on a gate), a Literal that is
# not a number at all, and a genuine bool.
BENCH = {"nodes": [
    {"id": "tone", "type": "parametric_eq", "params": {"bands": [
        {"filter_type": "bell", "center_freq_hz": 250.0,
         "gain_db": 3.0, "q_factor": 1.0},
        {"filter_type": "highshelf", "center_freq_hz": 8000.0,
         "gain_db": 2.0, "q_factor": 0.707}]}},
    {"id": "makeup", "type": "gain", "params": {"gain_db": 0.0}},
    {"id": "gate", "type": "gate_expander", "params": {"range_db": 40.0}},
    {"id": "dyn", "type": "dynamic_eq", "params": {"bands": [
        {"filter_type": "bell", "mode": "downward", "center_freq_hz": 3000.0,
         "static_gain_db": 0.0, "range_db": 6.0}]}},
    {"id": "comp", "type": "compressor",
     "params": {"threshold_dbfs": -24.0, "ratio_to_1": 3.0}},
    {"id": "flip", "type": "polarity", "params": {"inverted": True}},
]}

# A chain that is already close to the limiter's limit. One more in-range push
# takes it over, which is the only way an edit that respects its own bound can
# still be refused by whole-graph validation.
HOT = {"nodes": [
    {"id": "level", "type": "compressor", "params": {
        "threshold_dbfs": -24.0, "ratio_to_1": 3.0, "makeup_gain_db": 20.0}},
    {"id": "makeup", "type": "gain", "params": {"gain_db": 0.0}},
]}


class Fixtures:
    """Built once. Uploading measures the file twice, so it is not free."""

    quiet_id = ""     # 24 s, quiet: the excerpt is a strict subset, and a
                      # +24 dB gain node still leaves the limiter a job it can do
    loud_id = ""      # 6 s, loud: enough level that one more boost is fatal
    named_id = ""     # a project whose name is hostile to a Content-Disposition
    plan_id = ""      # a deterministically planned graph, for rendering


F = Fixtures()


def build_fixtures() -> None:
    print("[0] fixtures")

    response = upload("My Song (final)!.wav", stereo(24.0, 0.08))
    R.check(response.status_code == 200, f"a real WAV uploads: {response.text[:200]}")
    F.quiet_id = response.json()["id"]

    F.loud_id = upload("loud.wav", stereo(6.0, 0.5)).json()["id"]
    F.named_id = upload("Café Session ▲.wav", stereo(3.0, 0.1)).json()["id"]

    planned = CLIENT.post(f"/api/projects/{F.quiet_id}/plan",
                          json={"planner": "rules"})
    R.check(planned.status_code == 200, f"the rules planner runs: {planned.text[:200]}")
    F.plan_id = planned.json()["graph_id"]


# ---------------------------------------------------------------------------


def test_health_and_intents() -> None:
    print("[1] health and the intent vocabulary")

    health = CLIENT.get("/api/health")
    body = health.json()
    R.check(health.status_code == 200 and body.get("ok") is True,
            f"health reports ok, got {body}")
    R.check(set(body) == {"ok", "engine", "model", "projects"},
            f"and exactly the documented keys, got {sorted(body)}")
    # `model` is whether a key is present, never whether one works. A health
    # check that made a model call would cost money on every page load.
    R.check(isinstance(body["model"], bool) and isinstance(body["engine"], bool),
            "engine and model are booleans, not URLs or version strings")
    R.check(body["projects"] >= 3, f"and the project count is live, got {body['projects']}")

    intents = CLIENT.get("/api/intents").json()
    R.check(len(intents["intents"]) == len(ladder.INTENT_TO_PARAM),
            "every intent in the ladder is offered")

    # THE VOCABULARY IS CLOSED, and the UI builds buttons from it. An intent
    # naming a node type or a parameter that does not exist would render a
    # button that fails only when pressed.
    for entry in intents["intents"]:
        R.check(set(entry) == {"id", "label", "node_type", "parameter"},
                f"{entry.get('id')} has the documented shape, got {sorted(entry)}")
        model = PARAM_MODELS.get(entry["node_type"])
        R.check(model is not None, f"{entry['id']} names a real node type")
        R.check(entry["parameter"] == "mix"
                or (model is not None and entry["parameter"] in model.model_fields),
                f"{entry['id']} names a parameter {entry['node_type']} actually has")
        R.check("_" not in entry["label"], f"{entry['id']}'s label is prose, not an id")

    # Sorted by strength, because the UI shows them in order and a list that
    # reads "dramatically, slightly, much" tells the user nothing.
    strengths = [ladder.QUANTIFIERS[q] for q in intents["quantifiers"]]
    R.check(strengths == sorted(strengths),
            f"quantifiers come back weakest first, got {intents['quantifiers']}")


def test_non_wav_is_refused() -> None:
    print("[2] anything that is not a WAV")

    before = CLIENT.get("/api/health").json()["projects"]

    # A valid RIFF header with a codec tag this decoder does not implement. The
    # interesting one: everything about it says WAV, so a decoder that guessed
    # would happily read the payload as PCM and return noise.
    adpcm = (b"RIFF" + struct.pack("<I", 36) + b"WAVE"
             + b"fmt " + struct.pack("<IHHIIHH", 16, 0x0011, 1, 44100,
                                     22050, 512, 4)
             + b"data" + struct.pack("<I", 0))

    cases = [
        ("random bytes", "noise.wav", RNG.integers(0, 256, 4096, dtype=np.uint8).tobytes()),
        ("an MP3", "song.mp3", b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\xff\xfb\x90\x00" * 256),
        ("a truncated WAV", "cut.wav", wavio.encode_wav(stereo(1.0, 0.2), SR, 24)[:30]),
        ("an ADPCM WAV", "adpcm.wav", adpcm),
    ]

    for label, name, payload in cases:
        response = CLIENT.post("/api/projects",
                               files={"file": (name, payload, "audio/wav")})
        R.check(response.status_code == 415,
                f"{label} is refused with 415, got {response.status_code}")
        detail = response.json().get("detail", "")
        # THE REFUSAL HAS TO SAY WHY. "415" alone leaves the user converting the
        # file at random; naming the filename and the reason turns it into one
        # action. It is also the difference between a refusal and a guess.
        R.check(name in detail and "WAV" in detail,
                f"{label}: the refusal names the file and the format, got {detail!r}")
        R.check(len(detail) > len(name) + 40,
                f"{label}: the refusal says more than 'no', got {detail!r}")

    R.check(CLIENT.get("/api/health").json()["projects"] == before,
            "and a refused upload leaves no half-made project behind")

    # The direction that matters just as much: a real WAV still gets in. A test
    # suite that only proves things are rejected is satisfied by an API that
    # rejects everything.
    accepted = upload("fine.wav", stereo(2.0, 0.1))
    R.check(accepted.status_code == 200, f"a real WAV is accepted, got {accepted.text[:120]}")
    body = accepted.json()
    R.check(body["channels"] == 2 and abs(body["duration_s"] - 2.0) < 0.05
            and body["sample_rate"] == SR,
            f"and it is described correctly, got {body['channels']}ch "
            f"{body['duration_s']}s {body['sample_rate']}Hz")


def test_bounds_are_read_per_model() -> None:
    print("[3] bounds come from the model, never from the name")

    gid = store(F.quiet_id, BENCH)

    # THE ASSERTION THIS FILE EXISTS FOR.
    #
    # `gain_db` means two different things. On an EQ band it is bounded at plus
    # or minus 18; on a gain node at plus or minus 24. A table keyed on the name
    # cannot represent that, so it silently applies the wider bound to bands and
    # a 22 dB band gain is stored — a value the schema itself would have
    # rejected, now sitting in a graph that will render.
    #
    # The same bug in the validator was found by measurement, not by reading.
    # This is the second place it can appear, so it is checked from both sides:
    # 22 must be refused on the band AND accepted on the node. Only checking the
    # refusal would pass against a table that bounded everything at 18.
    band = edit(F.quiet_id, gid, node_id="tone", param="gain_db",
                value=22.0, band_index=0)
    R.check(band.status_code == 422,
            f"22 dB on an EQ band is refused, got {band.status_code} {band.text[:160]}")
    detail = band.json().get("detail", "")
    R.check("18" in detail and "-18" in detail,
            f"and the refusal names the band's own bound, got {detail!r}")
    R.check("24" not in detail,
            f"and does not quote the gain node's wider bound, got {detail!r}")

    node = edit(F.quiet_id, gid, node_id="makeup", param="gain_db", value=22.0)
    R.check(node.status_code == 200,
            f"22 dB on a gain node is accepted, got {node.status_code} {node.text[:200]}")
    stored = [n for n in node.json()["nodes"] if n["id"] == "makeup"]
    R.check(stored and stored[0]["params"]["gain_db"] == 22.0,
            f"and stored as typed rather than clamped, got {stored}")

    # Both directions of the sign, because a bound applied as `abs(value) <= 18`
    # and one applied as `-18 <= value <= 18` differ only on asymmetric ranges.
    R.check(edit(F.quiet_id, gid, node_id="tone", param="gain_db",
                 value=-22.0, band_index=0).status_code == 422,
            "-22 dB on an EQ band is refused too")
    R.check(edit(F.quiet_id, gid, node_id="makeup", param="gain_db",
                 value=-22.0).status_code == 200,
            "-22 dB on a gain node is accepted")

    # And the gain node is not simply unbounded, which would also pass every
    # check above.
    wide = edit(F.quiet_id, gid, node_id="makeup", param="gain_db", value=25.0)
    R.check(wide.status_code == 422 and "24" in wide.json().get("detail", ""),
            f"25 dB on a gain node is refused, naming 24, got {wide.text[:160]}")

    # A SECOND, INDEPENDENT COLLISION. `range_db` is 0 to 80 on a gate — how far
    # it closes — and -18 to 18 on a dynamic EQ band, where it is signed because
    # a negative value compresses the band and a positive one expands it. 60 is
    # ordinary on one and meaningless on the other.
    R.check(edit(F.quiet_id, gid, node_id="gate", param="range_db",
                 value=60.0).status_code == 200,
            "60 dB of range on a gate is accepted")
    dyn = edit(F.quiet_id, gid, node_id="dyn", param="range_db",
               value=60.0, band_index=0)
    R.check(dyn.status_code == 422,
            f"60 dB of range on a dynamic EQ band is refused, got {dyn.status_code}")
    R.check("18" in dyn.json().get("detail", ""),
            f"naming the band's bound, got {dyn.json().get('detail')!r}")

    # The generalisation. Every numeric parameter the fixture exposes, probed
    # one unit outside its declared range, must be refused with its OWN bound in
    # the message. This is what a name-keyed table fails on the next time two
    # nodes come to share a parameter name — which they already do twice.
    probes = 0
    # Read from the STORED graph rather than the fixture literal, so the sweep
    # covers every parameter the node actually carries — including the defaults
    # nobody typed, which are exactly the ones a hand-maintained table forgets.
    for entry in server.PROJECTS[F.quiet_id].graphs[gid]["graph"].nodes:
        checks: list[tuple[dict, dict, int | None]] = [
            (entry.params, _bounds(PARAM_MODELS[entry.type]), None)]
        band_model = DynamicEqBand if entry.type == "dynamic_eq" else EqBand
        for index, spec in enumerate(entry.params.get("bands", [])):
            checks.append((spec, _bounds(band_model), index))

        for params, bounds, index in checks:
            for param, value in params.items():
                if param not in bounds or isinstance(value, bool):
                    continue
                low, high, _ = bounds[param]
                response = edit(F.quiet_id, gid, node_id=entry.id, param=param,
                                value=high + 1.0, band_index=index)
                where = f"{entry.id}.{param}" + ("" if index is None else f"[{index}]")
                R.check(response.status_code == 422,
                        f"{where} at {high + 1:g} is refused, got {response.status_code}")
                text = response.json().get("detail", "")
                R.check(f"{low:g}" in text and f"{high:g}" in text,
                        f"{where}: the refusal names {low:g} to {high:g}, got {text!r}")
                probes += 1

    R.check(probes >= 30, f"and the sweep actually covered the graph, {probes} parameters")


def test_a_name_is_not_a_number() -> None:
    print("[4] parameters that are names rather than numbers")

    gid = store(F.quiet_id, BENCH)
    before = snapshot(F.quiet_id, gid)

    # `filter_type` and `detector` are closed sets of names. Writing a number
    # into one is accepted by any check that asks "is this within its numeric
    # bound?", because a Literal has no numeric bound to be within. The stored
    # graph then no longer round-trips through the schema that produced it, and
    # the engine — which reads these keys as strings — falls back to its own
    # default. Someone asking for a notch gets a bell and is told nothing, which
    # is the silent enum fallback this project refuses everywhere else.
    for node_id, param, index in (("tone", "filter_type", 0),
                                  ("comp", "detector", None),
                                  ("dyn", "mode", 0)):
        response = edit(F.quiet_id, gid, node_id=node_id, param=param,
                        value=2.0, band_index=index)
        R.check(response.status_code == 422,
                f"a number written to {node_id}.{param} is refused, "
                f"got {response.status_code}")
        detail = response.json().get("detail", "")
        R.check("number" in detail.lower(),
                f"and the refusal explains why, got {detail!r}")

    R.check(snapshot(F.quiet_id, gid) == before,
            "and none of those refusals touched the stored graph")

    # A bool is a number's near neighbour and must NOT be caught by the same
    # rule: `inverted` is a real, editable parameter. What matters is that it
    # stays a bool — `float(False)` is 0.0, which the engine happens to read the
    # same way but which is the wrong type in the graph the inspector renders
    # and the next model turn reads.
    flip = edit(F.quiet_id, gid, node_id="flip", param="inverted", value=False)
    R.check(flip.status_code == 200, f"a bool parameter is editable, got {flip.text[:160]}")
    if flip.status_code == 200:
        value = server.PROJECTS[F.quiet_id].graphs[flip.json()["graph_id"]]
        value = [n for n in value["graph"].nodes if n.id == "flip"][0].params["inverted"]
        R.check(value is False,
                f"and stored as a bool, not a float, got {value!r} ({type(value).__name__})")

    # The property all of this protects: a graph that came out of an edit is
    # still a graph. Anything the endpoint stores must survive the schema that
    # every other stage validates against.
    for graph_id, entry in server.PROJECTS[F.quiet_id].graphs.items():
        try:
            EffectGraph.model_validate(entry["graph"].model_dump())
            ok, why = True, ""
        except Exception as error:  # noqa: BLE001
            ok, why = False, str(error).splitlines()[0]
        R.check(ok, f"graph {graph_id} still round-trips through its own schema: {why}")


def test_unknown_things_404() -> None:
    print("[5] unknown ids")

    gid = store(F.quiet_id, BENCH)

    R.check(CLIENT.get("/api/projects/nosuchproject").status_code == 404,
            "an unknown project is 404")
    R.check(edit("nosuchproject", gid, node_id="makeup", param="gain_db",
                 value=1.0).status_code == 404,
            "editing inside an unknown project is 404")

    missing_graph = edit(F.quiet_id, "nosuchgraph", node_id="makeup",
                         param="gain_db", value=1.0)
    R.check(missing_graph.status_code == 404,
            f"an unknown graph is 404, got {missing_graph.status_code}")

    # THE TEXT IS THE PRODUCT HERE. The caller is a person in a UI or a model
    # composing the next turn, and "404" tells neither of them which of the four
    # ids they got wrong.
    node = edit(F.quiet_id, gid, node_id="nosuchnode", param="gain_db", value=1.0)
    R.check(node.status_code == 404, f"an unknown node is 404, got {node.status_code}")
    R.check("nosuchnode" in node.json().get("detail", ""),
            f"and the message names it, got {node.json().get('detail')!r}")

    param = edit(F.quiet_id, gid, node_id="makeup", param="nosuchparam", value=1.0)
    R.check(param.status_code == 404, f"an unknown parameter is 404, got {param.status_code}")
    detail = param.json().get("detail", "")
    R.check("nosuchparam" in detail and "gain_db" in detail,
            f"and the message lists what the node does have, got {detail!r}")

    band = edit(F.quiet_id, gid, node_id="tone", param="gain_db",
                value=1.0, band_index=7)
    R.check(band.status_code == 404, f"an unknown band index is 404, got {band.status_code}")
    R.check("7" in band.json().get("detail", ""),
            f"and the message names the index, got {band.json().get('detail')!r}")

    # A band index on a node that has no bands at all, which is the same mistake
    # made one level up.
    R.check(edit(F.quiet_id, gid, node_id="makeup", param="gain_db",
                 value=1.0, band_index=0).status_code == 404,
            "a band index on a node with no bands is 404")


def test_a_rejected_edit_changes_nothing() -> None:
    print("[6] a rejected edit leaves the stored graph alone")

    gid = store(F.loud_id, HOT)
    before = snapshot(F.loud_id, gid)
    graphs_before = len(server.PROJECTS[F.loud_id].graphs)

    # First prove the graph is otherwise editable, so the rejection below is
    # attributable to the value rather than to the fixture being broken.
    R.check(edit(F.loud_id, gid, node_id="makeup", param="gain_db",
                 value=0.0).status_code == 200,
            "the fixture accepts an ordinary edit")

    # THE REJECTION THAT MATTERS FOR THIS PROPERTY is one that happens AFTER the
    # value has been written. A bound check refuses before anything is touched,
    # so it would pass even with the deep copy removed; whole-graph validation
    # refuses after, and that is the path where a shared reference corrupts the
    # stored graph. +24 dB on top of 20 dB of makeup leaves the limiter an
    # impossible job.
    hot = edit(F.loud_id, gid, node_id="makeup", param="gain_db", value=24.0)
    R.check(hot.status_code == 422,
            f"an in-range value that breaks the graph is refused, "
            f"got {hot.status_code} {hot.text[:200]}")
    R.check("gain_staging" in hot.json().get("detail", ""),
            f"and the refusal names the check that failed, "
            f"got {hot.json().get('detail')!r}")

    R.check(snapshot(F.loud_id, gid) == before,
            "and the graph it was edited from is byte-for-byte unchanged")
    stored = [n for n in server.PROJECTS[F.loud_id].graphs[gid]["graph"].nodes
              if n.id == "makeup"]
    R.check(stored[0].params["gain_db"] == 0.0,
            f"the parameter still holds its original value, got {stored[0].params}")

    # A refusal must not leave a graph on the tree either. A rejected edit that
    # still branched would show up in the UI as a version the user never made.
    R.check(len(server.PROJECTS[F.loud_id].graphs) == graphs_before + 1,
            "and only the one accepted edit added a graph")

    # The bound-check path, for completeness: refused before anything is written.
    out_of_range = edit(F.loud_id, gid, node_id="makeup", param="gain_db", value=99.0)
    R.check(out_of_range.status_code == 422, "an out-of-range value is refused")
    R.check(snapshot(F.loud_id, gid) == before,
            "and that path leaves the stored graph unchanged as well")


def test_an_accepted_edit_branches() -> None:
    print("[7] an accepted edit lands on the tree as its own graph")

    gid = store(F.quiet_id, BENCH)
    before = snapshot(F.quiet_id, gid)

    response = edit(F.quiet_id, gid, node_id="tone", param="center_freq_hz",
                    value=300.0, band_index=0)
    R.check(response.status_code == 200, f"the edit is accepted: {response.text[:200]}")
    body = response.json()
    new_id = body["graph_id"]

    R.check(new_id != gid, "it produces a NEW graph rather than editing in place")
    R.check(body["parent"] == gid, f"whose parent is the graph it came from, got {body['parent']}")
    R.check(body["origin"] == "user_edit", f"marked user_edit, got {body['origin']}")
    R.check("center_freq_hz" in body["changed"] and "band[0]" in body["changed"],
            f"and says what changed, got {body['changed']!r}")

    # ORIGIN IS NOT DECORATION, and it has to be on the STORED entry rather than
    # only in the response. A later model turn reads the tree, and a correction
    # it cannot see as a correction is one it will re-plan away.
    entry = server.PROJECTS[F.quiet_id].graphs[new_id]
    R.check(entry["origin"] == "user_edit",
            f"the stored entry records the origin, got {entry.get('origin')!r}")
    R.check(entry["parent"] == gid, "and the stored parent")

    R.check(snapshot(F.quiet_id, gid) == before,
            "the graph it branched from is untouched")
    R.check(server.PROJECTS[F.quiet_id].head == new_id,
            "and the edit becomes the head, so the next action follows it")

    project = CLIENT.get(f"/api/projects/{F.quiet_id}").json()
    R.check(gid in project["graphs"] and new_id in project["graphs"],
            "both versions are visible on the tree")
    changed = [n for n in project["graphs"][new_id]["nodes"] if n["id"] == "tone"]
    R.check(changed[0]["params"]["bands"][0]["center_freq_hz"] == 300.0,
            f"with the new value, got {changed[0]['params']['bands'][0]}")

    # Editing the edit, because a tree whose second level cannot be edited is a
    # list with extra steps.
    again = edit(F.quiet_id, new_id, node_id="tone", param="center_freq_hz",
                 value=350.0, band_index=0)
    R.check(again.status_code == 200 and again.json()["parent"] == new_id,
            f"an edit of an edit chains from it, got {again.text[:160]}")

    # mix and bypass live on the node rather than in params, so they take a
    # different path through the endpoint and need their own check.
    mixed = edit(F.quiet_id, gid, node_id="tone", param="mix", value=0.5)
    R.check(mixed.status_code == 200
            and [n for n in mixed.json()["nodes"] if n["id"] == "tone"][0]["mix"] == 0.5,
            f"mix is editable, got {mixed.text[:160]}")
    R.check(edit(F.quiet_id, gid, node_id="tone", param="mix",
                 value=1.5).status_code == 422,
            "and bounded at 1")
    bypassed = edit(F.quiet_id, gid, node_id="tone", param="bypass", value=True)
    R.check(bypassed.status_code == 200
            and [n for n in bypassed.json()["nodes"] if n["id"] == "tone"][0]["bypass"] is True,
            f"bypass is editable and stays a bool, got {bypassed.text[:160]}")


def test_validation_still_runs_on_an_edit() -> None:
    print("[8] the whole graph is still validated after an edit")

    gid = store(F.quiet_id, BENCH)

    # The fixture has an empty master section. Validation's job is to add the
    # delivery ceiling, so its presence in the result is proof the edit went
    # through validation rather than around it — and it is proof that does not
    # depend on the edit being a bad one.
    response = edit(F.quiet_id, gid, node_id="makeup", param="gain_db", value=1.0)
    R.check(response.status_code == 200, f"the edit is accepted: {response.text[:200]}")
    nodes = response.json()["nodes"]
    limiter = [n for n in nodes if n["type"] == "true_peak_limiter"]
    R.check(len(limiter) == 1 and limiter[0]["master"] is True,
            f"validation added the delivery ceiling, got {[n['type'] for n in nodes]}")
    R.check(any("ceiling" in r or "limiter" in r for r in response.json()["repairs"]),
            f"and said so, got {response.json()['repairs']}")

    # A REPAIR IS REPORTED, NOT SILENT. Pushing the band gains past the total
    # boost budget is repairable rather than fatal — the shape the user asked
    # for is preserved and only its depth changes — but the number they typed is
    # not what gets stored, so the response has to say so. This is the one place
    # the endpoint's "we never quietly store a different number" promise is
    # softened, and it is softened deliberately.
    budget = edit(F.quiet_id, gid, node_id="tone", param="gain_db",
                  value=18.0, band_index=0)
    R.check(budget.status_code == 200, f"an in-budget-breaking edit is repaired: "
                                       f"{budget.text[:200]}")
    repairs = budget.json()["repairs"]
    R.check(any("budget" in r for r in repairs),
            f"and the repair names the boost budget, got {repairs}")
    band = [n for n in budget.json()["nodes"] if n["id"] == "tone"][0]
    R.check(band["params"]["bands"][0]["gain_db"] < 18.0,
            f"the stored gain really was scaled, got {band['params']['bands'][0]['gain_db']}")

    # And the fatal direction, on a chain that is already hot: an edit inside
    # its own bound that the graph as a whole cannot absorb.
    hot_gid = store(F.loud_id, HOT)
    fatal = edit(F.loud_id, hot_gid, node_id="makeup", param="gain_db", value=24.0)
    R.check(fatal.status_code == 422,
            f"an edit the graph cannot absorb is refused, got {fatal.status_code}")
    R.check("limiter" in fatal.json().get("detail", ""),
            f"and the refusal is in words, got {fatal.json().get('detail')!r}")


def test_planner_needs_no_model() -> None:
    print("[9] the deterministic planner, and the model path stays shut")

    response = CLIENT.post(f"/api/projects/{F.quiet_id}/plan",
                           json={"planner": "rules"})
    R.check(response.status_code == 200, f"planner=rules works: {response.text[:200]}")
    body = response.json()
    R.check(body["nodes"] and body["explanation"],
            "and returns a chain with an explanation")
    R.check(any(n["type"] == "true_peak_limiter" and n["master"] for n in body["nodes"]),
            "with a delivery ceiling in the master section")

    # BOTH PROBES BELOW RUN WITH THE KEY REMOVED, and that is not incidental.
    # An unknown planner name that stopped being rejected would fall through to
    # the model branch, and a test asserting "this is refused" would then prove
    # it by making a paid API call on every CI run. The key is taken out of the
    # environment so the worst case is a 503 rather than a request.
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        # UNKNOWN ENUM VALUES ARE HARD ERRORS. The caller is a language model,
        # and a planner name that silently fell back to a default would produce
        # a plan from the wrong system with nothing to say which.
        unknown = CLIENT.post(f"/api/projects/{F.quiet_id}/plan",
                              json={"planner": "banana"})
        R.check(unknown.status_code == 422,
                f"an unknown planner name is refused rather than defaulted, "
                f"got {unknown.status_code}")

        # And the model path refuses rather than reaching for the network when
        # there is no key to reach with.
        no_key = CLIENT.post(f"/api/projects/{F.quiet_id}/plan",
                             json={"planner": "model"})
        R.check(no_key.status_code == 503,
                f"planner=model with no key is 503, got {no_key.status_code}")
        R.check("rules" in no_key.json().get("detail", ""),
                f"and points at the alternative, got {no_key.json().get('detail')!r}")
    finally:
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved


def test_render_excerpt_and_full() -> None:
    print("[10] rendering an excerpt and the whole file")

    if not have_engine():
        print("  engine binary missing; skipping")
        return

    excerpt = CLIENT.post(f"/api/projects/{F.quiet_id}/graphs/{F.plan_id}/render",
                          params={"full": "false"}).json()
    R.check(excerpt.get("ok") is True, f"the excerpt renders: {str(excerpt)[:240]}")
    R.check(excerpt["render_key"] == F.plan_id,
            f"keyed on the graph, got {excerpt.get('render_key')}")
    R.check(excerpt["source_key"] == "excerpt" and excerpt["full"] is False,
            f"and says which source it used, got {excerpt.get('source_key')}")

    full = CLIENT.post(f"/api/projects/{F.quiet_id}/graphs/{F.plan_id}/render",
                       params={"full": "true"}).json()
    R.check(full.get("ok") is True, f"the full file renders: {str(full)[:240]}")
    R.check(full["render_key"] == f"{F.plan_id}-full" and full["source_key"] == "full",
            f"under a distinct key, got {full.get('render_key')}")

    # The two must not collide. Sharing a key would mean auditioning an excerpt
    # and downloading it silently returned the other one.
    R.check(full["render_key"] != excerpt["render_key"], "the two renders are distinct")

    frames = {}
    for label, body in (("excerpt", excerpt), ("full", full)):
        audio = CLIENT.get(f"/api/projects/{F.quiet_id}/audio/{body['render_key']}")
        R.check(audio.status_code == 200, f"the {label} render is fetchable")
        frames[label] = wavio.decode_wav(audio.content).frames

    # EXCERPT-FIRST HAS TO ACTUALLY MEAN SOMETHING. If `full` rendered the same
    # 20-second window the whole design would be a no-op that still cost the
    # user the wait.
    R.check(frames["full"] > frames["excerpt"] * 1.1,
            f"the full render really is the whole file, {frames}")
    R.check(abs(frames["excerpt"] / SR - server.EXCERPT_SECONDS) < 0.5,
            f"and the excerpt is the excerpt window, got {frames['excerpt'] / SR:.1f}s")

    R.check("after" in excerpt and excerpt["after"].get("loudness"),
            "and a measurement of the result comes back with it")


def test_the_delivered_file_is_not_loudness_matched() -> None:
    print("[11] what you audition is not what you download")

    if not have_engine():
        print("  engine binary missing; skipping")
        return

    body = CLIENT.post(f"/api/projects/{F.quiet_id}/graphs/{F.plan_id}/render",
                       params={"full": "false"}).json()
    R.check(body.get("ok") is True, f"the render succeeds: {str(body)[:200]}")

    R.check(body["deliver_key"] != body["render_key"],
            f"the delivered file has its own key, got {body['deliver_key']}")

    matched = wavio.decode_wav(
        CLIENT.get(f"/api/projects/{F.quiet_id}/audio/{body['render_key']}").content)
    delivered = wavio.decode_wav(
        CLIENT.get(f"/api/projects/{F.quiet_id}/audio/{body['deliver_key']}").content)

    offset = body["match_offset_db"]
    # A vacuous version of this test is one where the chain happened not to
    # change the level: the two files would be identical and every assertion
    # below would pass against an endpoint that returned the same file twice.
    R.check(abs(offset) > 0.5,
            f"the chain moved the level enough for this to mean anything, {offset} dB")
    # The safety rescale inside the endpoint would break the exact relationship,
    # so the fixture is deliberately quiet enough that it cannot fire.
    R.check(float(np.max(np.abs(matched.samples))) < 0.999,
            "and the matched render did not need the peak safety net")

    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))  # noqa: E731
    measured = 20.0 * np.log10(rms(matched.samples) / rms(delivered.samples))
    # AUDITIONING IS LOUDNESS-MATCHED SO THE A/B COMPARES PROCESSING; DELIVERY
    # IS NOT, because matching hands back audio deliberately pulled to the
    # source's loudness and undoes most of what a limiter was for. The two files
    # therefore differ by exactly the reported offset, and by nothing else.
    R.check(abs(measured - offset) < 0.02,
            f"the two files differ by exactly match_offset_db: "
            f"measured {measured:.3f}, reported {offset}")
    R.check(matched.frames == delivered.frames,
            "they are the same render, at two levels")


def test_download_naming() -> None:
    print("[12] the downloaded file is named for a Downloads folder")

    if not have_engine():
        print("  engine binary missing; skipping")
        return

    body = CLIENT.post(f"/api/projects/{F.quiet_id}/graphs/{F.plan_id}/render",
                       params={"full": "false"}).json()
    response = CLIENT.get(f"/api/projects/{F.quiet_id}/download/{body['deliver_key']}")
    R.check(response.status_code == 200, "the download resolves")

    disposition = response.headers.get("content-disposition", "")
    R.check(disposition.startswith("attachment"),
            f"it is sent as an attachment, got {disposition!r}")
    R.check('filename="' in disposition,
            f"with a plain filename, got {disposition!r}")

    # THE GRAPH ID IS THE POINT OF THE NAME. A file that turns up three weeks
    # later has to be traceable to the exact chain that produced it, and the
    # graph id is the key the recipe is stored under.
    R.check(F.plan_id in disposition,
            f"carrying the graph id, got {disposition!r}")
    R.check("My-Song-final" in disposition,
            f"and something the user will recognise, got {disposition!r}")

    # The same bytes, but /audio streams inline for an <audio> element. If both
    # sent an attachment the player would trigger a download on every audition.
    inline = CLIENT.get(f"/api/projects/{F.quiet_id}/audio/{body['deliver_key']}")
    R.check("attachment" not in inline.headers.get("content-disposition", ""),
            "while /audio does not force a download")

    # ASCII ONLY. Anything outside it comes back as an RFC 5987 `filename*=`
    # header with no plain fallback — correct, unreadable, and dependent on the
    # client decoding it. A filename is not the place to be clever, so a project
    # name full of things a header cannot carry is checked directly.
    project = server.PROJECTS[F.named_id]
    project.renders["abcd1234-deliver"] = project.path
    named = CLIENT.get(f"/api/projects/{F.named_id}/download/abcd1234-deliver")
    header = named.headers.get("content-disposition", "")
    R.check(named.status_code == 200, "a project with a hostile name still downloads")
    try:
        header.encode("ascii")
        ascii_only = True
    except UnicodeEncodeError:
        ascii_only = False
    R.check(ascii_only, f"and its header is pure ASCII, got {header!r}")
    R.check("filename*" not in header,
            f"with no RFC 5987 encoding to decode, got {header!r}")
    R.check("abcd1234" in header, f"still carrying the graph id, got {header!r}")

    # Scope, so an excerpt and a full render of the same graph do not land in
    # the same folder under the same name.
    full_body = CLIENT.post(f"/api/projects/{F.quiet_id}/graphs/{F.plan_id}/render",
                            params={"full": "true"}).json()
    full_header = CLIENT.get(
        f"/api/projects/{F.quiet_id}/download/{full_body['deliver_key']}"
    ).headers.get("content-disposition", "")
    R.check("full" in full_header and "excerpt" not in full_header,
            f"a full render says so in its name, got {full_header!r}")
    R.check(full_header != disposition, "so the two do not overwrite each other")


def test_missing_renders_404() -> None:
    print("[13] renders that do not exist")

    cases = [
        ("rendering an unknown graph",
         lambda: CLIENT.post(f"/api/projects/{F.quiet_id}/graphs/nosuchgraph/render")),
        ("rendering inside an unknown project",
         lambda: CLIENT.post(f"/api/projects/nosuchproject/graphs/{F.plan_id}/render")),
        ("an unknown render key on /audio",
         lambda: CLIENT.get(f"/api/projects/{F.quiet_id}/audio/nosuchkey")),
        ("an unknown render key on /download",
         lambda: CLIENT.get(f"/api/projects/{F.quiet_id}/download/nosuchkey")),
        ("a download from an unknown project",
         lambda: CLIENT.get("/api/projects/nosuchproject/download/anything")),
    ]
    for label, call in cases:
        R.check(status(call) == 404, f"{label} is 404, got {status(call)}")

    # A KEY REGISTERED AGAINST A FILE THAT HAS SINCE GONE. The working directory
    # is temporary, so this is the ordinary state of every render after a
    # restart, not an exotic case — and the file layer's answer to a missing
    # path is an exception thrown while the response is already being sent,
    # which no error handler above it can turn into anything a client can read.
    project = server.PROJECTS[F.quiet_id]
    project.renders["ghost"] = os.path.join(server.WORKDIR, "does-not-exist.wav")
    for endpoint in ("audio", "download"):
        got = status(lambda: CLIENT.get(f"/api/projects/{F.quiet_id}/{endpoint}/ghost"))
        R.check(got == 404,
                f"a key whose file has gone is 404 on /{endpoint}, got {got}")


if __name__ == "__main__":
    print("Audionova local API\n")
    build_fixtures()
    test_health_and_intents()
    test_non_wav_is_refused()
    test_bounds_are_read_per_model()
    test_a_name_is_not_a_number()
    test_unknown_things_404()
    test_a_rejected_edit_changes_nothing()
    test_an_accepted_edit_branches()
    test_validation_still_runs_on_an_edit()
    test_planner_needs_no_model()
    test_render_excerpt_and_full()
    test_the_delivered_file_is_not_loudness_matched()
    test_download_naming()
    test_missing_renders_404()
    print(f"\n{R.checks} checks, {R.failures} failure(s)")
    sys.exit(1 if R.failures else 0)
