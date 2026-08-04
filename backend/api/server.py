"""The local API: upload, measure, plan, render, refine.

    ./.venv/bin/uvicorn backend.api.server:app --reload --port 8000

This is the product's shape without the platform. Everything the plan schedules
for Phase 1 that is genuinely about audio lives here; everything that is about
running a business — auth, object storage, a durable queue, billing — does not,
and the seams are marked so those slot in rather than being retrofitted.

WHAT IS DELIBERATELY NOT HERE, and where it goes:

  Storage is a temp directory keyed by an opaque id. In production that is S3
  with presigned multipart upload direct from the browser, because a
  three-minute 24-bit WAV is about 50 MB and Server Actions cap request bodies
  at 1 MB. The `store` module is the seam.

  Rendering is synchronous. In production it is a Temporal workflow keyed on the
  recipe hash, so a worker crash does not lose the job and a cancel is honoured
  sub-second. `render_project` is the seam.

  Ingest decodes with whatever the file claims to be. In production it is
  sandboxed FFmpeg with no network, because the bytes are entirely
  attacker-controlled. `_decode_upload` is the seam.

  There is no auth, no tenancy and no quota.

EXCERPT-FIRST IS NOT A SHORTCUT. Planning and auditioning run on a 20-second
window chosen by gated coverage. Users iterate five to fifteen times per session
and almost none of those need the full file; rendering it every time would make
the loop feel slow and cost an order of magnitude more. The full render is an
explicit action.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from fastapi import FastAPI, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from backend.audio import wavio  # noqa: E402
from backend.engine.render import ENGINE_VERSION, render_array  # noqa: E402
from backend.eval import corpus  # noqa: E402
from backend.measure import loudness  # noqa: E402
from backend.measure.report import measure  # noqa: E402
from backend.plan import ladder, pipeline, rules  # noqa: E402
from backend.refine import match_eq  # noqa: E402
from backend.plan.canary import load_env  # noqa: E402
from backend.plan import validate as validate_module  # noqa: E402
from backend.plan.validate import validate  # noqa: E402
from backend.schemas.graph import (  # noqa: E402
    PARAM_MODELS, DynamicEqBand, EffectGraph, EqBand, GraphNode,
)
from backend.verify import gate  # noqa: E402

load_env()

EXCERPT_SECONDS = 20.0
MAX_UPLOAD_BYTES = 512 * 1024 * 1024

app = FastAPI(title="Audionova", version="0.1.0")

# The browser talks to this directly rather than through Next.js. Audio bytes
# must never traverse a Next route: Server Actions cap bodies at 1 MB, and a
# proxy silently truncates anything over 10 MB with only a console warning —
# a data-corruption bug that returns success.
# ANY LOCAL PORT, because Next picks 3001 when 3000 is taken and pinning one
# turns an ordinary occupied port into a CORS error the user has to decode.
#
# A regex over loopback only, NOT allow_origins=["*"]. The wildcard would let
# any page on the internet drive this API while the user has it running, and
# "it is only local" is exactly the reasoning that ships a wildcard to
# production. In production this becomes the single deployed origin.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["*"], allow_headers=["*"],
)


@dataclass
class Project:
    """One uploaded file and everything derived from it.

    Recipes form a TREE rather than a list, so undo is a pointer move and
    branching is free. Each entry records its parent, which is what makes a
    refinement a patch against something the user already accepted rather than
    a regeneration.
    """

    id: str
    name: str
    path: str
    sample_rate: float
    channels: int
    duration_s: float
    report: dict[str, Any]
    excerpt: np.ndarray = field(repr=False, default=None)
    excerpt_report: dict[str, Any] = field(default_factory=dict)
    graphs: dict[str, dict[str, Any]] = field(default_factory=dict)
    head: str | None = None
    renders: dict[str, str] = field(default_factory=dict)

    # A reference track reduced to a spectral curve. NO AUDIO, NO FILENAME, AND
    # DELIBERATELY NO CONTENT HASH. A reference is somebody else's master, very
    # often unreleased. A hash would let this service answer "has anyone else
    # uploaded this file", which is a question a company holding unreleased
    # masters should not be able to answer — so it is not that the hash is
    # unused, it is that it is never computed.
    reference_curve: np.ndarray | None = field(repr=False, default=None)
    reference_lufs: float | None = None


PROJECTS: dict[str, Project] = {}
WORKDIR = tempfile.mkdtemp(prefix="audionova_")


def project_or_404(project_id: str) -> Project:
    project = PROJECTS.get(project_id)
    if project is None:
        raise HTTPException(404, "no such project")
    return project


def _decode_upload(raw: bytes, filename: str) -> tuple[np.ndarray, float]:
    """Bytes to samples.

    THE SEAM FOR THE INGEST SANDBOX. In production this is FFmpeg in a
    container with `--network none`, seccomp, a read-only rootfs and a wallclock
    kill, because the input is 100% attacker-controlled and FFmpeg's CVE history
    is long. Here it is our own WAV reader, which is the one decoder we can
    reason about — and which is why the engine links no third-party parser.

    Anything that is not a WAV is rejected rather than guessed at.
    """
    try:
        audio = wavio.decode_wav(raw)
    except wavio.WavError as error:
        raise HTTPException(
            415, f"{filename}: only uncompressed WAV is accepted here. "
                 f"Other formats need the ingest sandbox, which is not built. "
                 f"({error})")
    return audio.samples, audio.sample_rate


class PlanRequest(BaseModel):
    request: str = Field(default="", max_length=2000)
    planner: str = Field(default="model", pattern="^(model|rules)$")
    reverb_mix: float = Field(default=0.0, ge=0.0, le=1.0)


class RefineRequest(BaseModel):
    intent: str
    quantifier: str = "slightly"


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "engine": os.path.exists(
            os.path.join(ROOT, "engine", "build", "anv_render")),
        "model": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "projects": len(PROJECTS),
    }


@app.post("/api/projects")
async def create_project(file: UploadFile) -> dict[str, Any]:
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "file too large")

    samples, rate = _decode_upload(raw, file.filename or "upload.wav")

    project_id = uuid.uuid4().hex[:12]
    path = os.path.join(WORKDIR, f"{project_id}.wav")
    wavio.write_wav(path, samples, rate, 24)

    # The full-file report describes what the user uploaded. The excerpt report
    # is what the planner reasons about, and they are kept separate because a
    # 20-second window of a four-minute file is not the same measurement.
    report = measure(samples, rate).to_dict()
    excerpt, _ = corpus.load_excerpt(path, EXCERPT_SECONDS)

    project = Project(
        id=project_id, name=file.filename or "upload.wav", path=path,
        sample_rate=rate, channels=int(samples.shape[0]),
        duration_s=samples.shape[1] / rate, report=report,
        excerpt=excerpt, excerpt_report=measure(excerpt, rate).to_dict(),
    )
    PROJECTS[project_id] = project

    return {"id": project_id, "name": project.name,
            "sample_rate": rate, "channels": project.channels,
            "duration_s": round(project.duration_s, 2), "report": report}


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict[str, Any]:
    project = project_or_404(project_id)
    return {
        "id": project.id, "name": project.name,
        "sample_rate": project.sample_rate, "channels": project.channels,
        "duration_s": round(project.duration_s, 2),
        "report": project.report, "head": project.head,
        "graphs": {k: {"id": k, "parent": v["parent"], "label": v["label"],
                       "explanation": v["graph"].explanation,
                       "nodes": _describe(v["graph"])}
                   for k, v in project.graphs.items()},
    }


def _describe(graph: EffectGraph) -> list[dict[str, Any]]:
    """The graph as the inspector shows it: every parameter, with units."""
    out = []
    for node in list(graph.nodes) + list(graph.master):
        out.append({
            "id": node.id, "type": node.type, "mix": node.mix,
            "bypass": node.bypass, "rationale": node.rationale,
            "params": node.params,
            "master": node in graph.master,
        })
    return out


def _store_graph(project: Project, graph: EffectGraph, label: str,
                 parent: str | None, origin: str = "planner") -> str:
    """Add a graph to the tree.

    `origin` distinguishes what the planner proposed from what the user typed.
    It is not decoration: a manual edit lands on the SAME tree as a normal
    parent, so the next model turn sees the user's correction as the thing it is
    refining rather than as an anomaly to be re-planned away. Regenerating over
    a correction is the fastest way to lose an engineer's trust.
    """
    graph_id = uuid.uuid4().hex[:8]
    project.graphs[graph_id] = {"graph": graph, "parent": parent,
                                "label": label, "origin": origin}
    project.head = graph_id
    return graph_id


def _find_node(graph: EffectGraph, node_id: str) -> Any:
    for node in list(graph.nodes) + list(graph.master):
        if node.id == node_id:
            return node
    raise HTTPException(404, f"no node called {node_id!r} in this graph")


def _bounds_for(node: Any, param: str, band: bool) -> tuple[float, float] | None:
    """The declared range for one parameter, read from the Pydantic model.

    NEVER FROM A NAME-KEYED TABLE. A band's `gain_db` is bounded at +/-18 and a
    gain node's at +/-24; a map keyed on the name alone cannot tell them apart
    and silently applies the wider one. That exact bug has already been fixed
    once in the validator, and this is the second place that would reintroduce
    it.
    """
    if band:
        model = DynamicEqBand if node.type == "dynamic_eq" else EqBand
    else:
        model = PARAM_MODELS.get(node.type)
    if model is None:
        return None
    found = validate_module._bounds(model).get(param)
    return (found[0], found[1]) if found else None


class EditRequest(BaseModel):
    """One parameter change, typed by a person.

    Deliberately one at a time. A batch edit would have to define what happens
    when the third change is rejected and the first two are not, and every
    answer to that is worse than making the caller send three requests.
    """

    node_id: str
    param: str = Field(max_length=60)
    value: float | bool
    band_index: int | None = Field(default=None, ge=0, le=31)


@app.post("/api/projects/{project_id}/graphs/{graph_id}/edit")
def edit_graph(project_id: str, graph_id: str,
               body: EditRequest) -> dict[str, Any]:
    """Change one parameter by hand, producing a new graph on the same tree.

    OUT OF RANGE IS REFUSED, NOT CLAMPED. The validator clamps, because a model
    that overshoots a bound still meant the direction and silently correcting it
    is better than discarding the whole graph. A person is different: they typed
    a specific number, and quietly storing a different one is the sort of thing
    that gets noticed three renders later. So the bound is checked first and the
    refusal names it.
    """
    project = project_or_404(project_id)
    entry = project.graphs.get(graph_id)
    if entry is None:
        raise HTTPException(404, "no such graph")

    # Deep-copied so a rejected edit cannot leave the stored graph half-changed.
    graph = entry["graph"].model_copy(deep=True)
    node = _find_node(graph, body.node_id)

    if body.param in ("mix", "bypass"):
        if body.param == "bypass":
            node.bypass = bool(body.value)
        else:
            if not 0.0 <= float(body.value) <= 1.0:
                raise HTTPException(422, "mix runs from 0 to 1")
            node.mix = float(body.value)
    else:
        if body.band_index is not None:
            bands = node.params.get("bands")
            if not isinstance(bands, list) or body.band_index >= len(bands):
                raise HTTPException(404, f"{node.id} has no band "
                                         f"{body.band_index}")
            target = bands[body.band_index]
        else:
            target = node.params

        if body.param not in target:
            raise HTTPException(
                404, f"{node.id} has no parameter {body.param!r}. It has "
                     f"{sorted(target)}.")

        # THE TYPE ALREADY IN THE SLOT IS THE CONTRACT. Every params dict got
        # there through the per-type Pydantic model, so what is sitting in the
        # slot says what the slot accepts.
        #
        # Coercing everything to float instead meant a parameter with no numeric
        # bound — every Literal in the schema — was overwritten with a number and
        # accepted. `filter_type` set to 2.0 was stored, the graph stopped
        # round-tripping through its own schema, and the engine, which reads that
        # key as a string, fell back to a bell. Someone asking for a notch got a
        # bell with no error: exactly the silent enum fallback this project
        # refuses everywhere else.
        current = target[body.param]
        if isinstance(current, bool):
            # Before the numeric branch, because a bool IS an int in Python and
            # the other branch would store 0.0 where the schema declares a bool.
            target[body.param] = bool(body.value)
        elif isinstance(current, (int, float)):
            bounds = _bounds_for(node, body.param, body.band_index is not None)
            if bounds and not bounds[0] <= float(body.value) <= bounds[1]:
                raise HTTPException(
                    422, f"{body.param} runs from {bounds[0]:g} to {bounds[1]:g} "
                         f"on a {node.type}; you asked for {float(body.value):g}")
            target[body.param] = float(body.value)
        else:
            raise HTTPException(
                422, f"{body.param} on a {node.type} is a name, not a number — "
                     f"it is currently {current!r}, one of a fixed set. This "
                     f"endpoint edits numbers only.")

    # Still validated. A single in-range parameter can still break the graph —
    # a Q that rings, a boost budget now exceeded, a chain that no longer has a
    # delivery ceiling — and the user's edit does not get to skip that.
    result = validate(graph, project.excerpt_report)
    if not result.ok:
        raise HTTPException(422, "; ".join(str(f) for f in result.fatal))

    where = (f"{body.node_id}.band[{body.band_index}].{body.param}"
             if body.band_index is not None else f"{body.node_id}.{body.param}")
    new_id = _store_graph(project, result.graph, f"you set {where}",
                          parent=graph_id, origin="user_edit")
    return {"graph_id": new_id, "parent": graph_id, "origin": "user_edit",
            "changed": where, "repairs": result.repairs,
            "nodes": _describe(result.graph)}


@app.post("/api/projects/{project_id}/plan")
def plan_project(project_id: str, body: PlanRequest) -> dict[str, Any]:
    project = project_or_404(project_id)
    report = project.excerpt_report

    started = time.time()
    if body.planner == "rules":
        graph, explanation = rules.plan(report, reverb_mix=body.reverb_mix)
        result = validate(graph, report)
        if not result.ok:
            raise HTTPException(500, f"the deterministic plan failed validation: "
                                     f"{[str(f) for f in result.fatal]}")
        graph = result.graph
        notes = {"attempts": 1, "repairs": result.repairs,
                 "declined": explanation.declined}
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise HTTPException(503, "no ANTHROPIC_API_KEY; use planner=rules")
        client = pipeline.AnthropicClient()
        outcome = pipeline.plan(
            client, report=report,
            request=body.request or "Improve this without changing its character.")
        if not outcome.ok:
            raise HTTPException(502, f"the planner could not produce a valid "
                                     f"graph: {outcome.fatal}")
        graph = outcome.graph
        notes = {"attempts": outcome.attempts, "repairs": outcome.repairs,
                 "reasoning": outcome.reasoning,
                 "stage_seconds": outcome.stage_seconds}

    graph_id = _store_graph(project, graph, body.request or "initial plan", None)
    return {"graph_id": graph_id, "elapsed_s": round(time.time() - started, 1),
            "explanation": graph.explanation, "nodes": _describe(graph),
            **notes}


@app.get("/api/projects/{project_id}/plan/stream")
def plan_stream(project_id: str, request: str = "",
                planner: str = "model") -> StreamingResponse:
    """Plan, streaming stage 1's reasoning as it is written.

    THE TEXT WAS ALREADY ARRIVING AND NOBODY WAS SHOWING IT. Stage 1 takes 150
    to 235 seconds and has streamed since the SDK refused a non-streaming call
    at that budget — every token was discarded until the call finished. The user
    watched a spinner for three minutes while the analysis they were waiting for
    came down the wire a sentence at a time.

    This does not make the planner faster and is not a substitute for making it
    faster. What it changes is that three minutes of nothing becomes three
    minutes of watching the reasoning that justifies the chain, which is the
    part a paying engineer is being asked to trust.

    GET rather than POST, because EventSource cannot issue a POST. That puts the
    request text in a query string, which is acceptable here and ONLY here: it
    is the user's own sentence about their own audio, not a credential, and it
    goes to a loopback server. In production this is a POST that returns a
    stream id, and the seam is here.
    """
    project = project_or_404(project_id)
    if planner != "model":
        raise HTTPException(400, "streaming is only meaningful for the model "
                                 "planner; the deterministic one is instant")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "no ANTHROPIC_API_KEY; use planner=rules")

    def events():
        chunks: list[tuple[str, Any]] = []

        def on_text(kind: str, payload: Any) -> None:
            chunks.append((kind, payload))

        # Run the plan on this thread and flush whatever has arrived between
        # yields. A queue and a worker thread would deliver each token sooner;
        # this delivers them in batches and has no thread to leak if the client
        # disconnects mid-render, which for a 200-second call is the likelier
        # event.
        import threading
        outcome: dict[str, Any] = {}

        def run() -> None:
            try:
                outcome["value"] = pipeline.plan(
                    pipeline.AnthropicClient(), report=project.excerpt_report,
                    request=request or "Improve this without changing its "
                                       "character.",
                    on_text=on_text)
            except Exception as error:
                outcome["error"] = str(error)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()

        sent = 0
        while worker.is_alive() or sent < len(chunks):
            while sent < len(chunks):
                kind, payload = chunks[sent]
                sent += 1
                # Two event types, because they are two different things. The
                # thinking figure is progress; the reasoning is the argument the
                # user is being asked to trust, and conflating them would let a
                # progress bar read as an explanation.
                name = "reasoning" if kind == "text" else "started"
                yield f"event: {name}\ndata: {json.dumps(payload)}\n\n"
            if worker.is_alive():
                time.sleep(0.2)

        worker.join()
        if "error" in outcome:
            yield f"event: error\ndata: {json.dumps(outcome['error'])}\n\n"
            return

        result = outcome.get("value")
        if result is None or not result.ok:
            fatal = result.fatal if result else ["the planner produced nothing"]
            yield f"event: error\ndata: {json.dumps('; '.join(fatal))}\n\n"
            return

        graph_id = _store_graph(project, result.graph, request or "initial plan",
                                None)
        payload = {"graph_id": graph_id, "explanation": result.graph.explanation,
                   "nodes": _describe(result.graph), "attempts": result.attempts,
                   "repairs": result.repairs,
                   "stage_seconds": result.stage_seconds}
        yield f"event: done\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(
        events(), media_type="text/event-stream",
        # A proxy that buffers an event stream turns it back into a spinner.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/projects/{project_id}/graphs/{graph_id}/refine")
def refine_graph(project_id: str, graph_id: str,
                 body: RefineRequest) -> dict[str, Any]:
    """Apply a refinement as a PATCH against an accepted graph.

    Never a regeneration. Regenerating discards every decision the user
    implicitly accepted by not complaining, and churns unrelated parameters —
    they asked for one change and six moved, which reads as a bug.

    The quantifier's arithmetic is done in Python so "slightly" means the same
    thing on Tuesday as today.
    """
    project = project_or_404(project_id)
    entry = project.graphs.get(graph_id)
    if entry is None:
        raise HTTPException(404, "no such graph")

    try:
        patches = ladder.patch_from_intent(entry["graph"], body.intent,
                                           body.quantifier)
    except KeyError as error:
        raise HTTPException(400, str(error))

    if not patches:
        return {"graph_id": graph_id, "applied": [], "rejected": [],
                "note": "nothing in this chain responds to that; "
                        "add the effect first"}

    result = ladder.apply_patches(entry["graph"], patches)
    validated = validate(result.graph, project.excerpt_report)

    new_id = _store_graph(project, validated.graph,
                          f"{body.quantifier} {body.intent}".replace("_", " "),
                          graph_id)
    return {"graph_id": new_id, "parent": graph_id,
            "applied": result.applied, "rejected": result.rejected,
            "diff": ladder.describe(patches),
            "repairs": validated.repairs, "nodes": _describe(validated.graph)}


@app.post("/api/projects/{project_id}/graphs/{graph_id}/render")
def render_graph(project_id: str, graph_id: str,
                 full: bool = False) -> dict[str, Any]:
    """Render, gate, and loudness-match.

    THE SEAM FOR THE JOB QUEUE. Synchronous here; in production a Temporal
    workflow keyed on the recipe hash, so a worker crash does not lose the job
    and stage-level caching makes a refinement that touches only the reverb
    cheap to re-render.
    """
    project = project_or_404(project_id)
    entry = project.graphs.get(graph_id)
    if entry is None:
        raise HTTPException(404, "no such graph")

    source = (wavio.read_wav(project.path).samples if full else project.excerpt)
    report = project.report if full else project.excerpt_report
    graph = entry["graph"]

    recipe = graph.to_recipe(seed=RENDER_SEED,
                             output_sample_rate=project.sample_rate,
                             output_format="float64")
    started = time.time()

    # The cache is keyed on the recipe and the source, so a hit is the same
    # bytes by construction rather than by assumption — the engine is
    # deterministic within one (engine_version, arch) and that is exactly the
    # guarantee a render cache needs to be allowed to exist.
    #
    # Undo, redo, re-auditioning a chain and A/B-ing two of them all land here.
    # Those are the commonest actions in a session and every one of them was
    # re-rendering something already rendered.
    # THE WHOLE RESULT IS CACHED, NOT JUST THE AUDIO. Everything downstream of
    # the render — the safety gate, the loudness match, the measurement report —
    # is a pure function of the rendered samples, and the samples are identical
    # by determinism. Re-deriving them would burn 0.5 s to arrive at bytes we
    # already have. Caching only the audio measured 1.3 s -> 0.5 s; caching the
    # result takes it to nothing.
    #
    # The gate is not being skipped. Its verdict is part of what is stored, and
    # a stored verdict is the verdict it reached on exactly these samples.
    cache_key = _recipe_hash(recipe, f"{project.id}:{'full' if full else 'excerpt'}")
    cached = RENDER_CACHE.get(cache_key)
    if cached is not None and all(
            os.path.exists(project.renders.get(k, ""))
            for k in (cached.get("render_key"), cached.get("deliver_key"))
            if k is not None):
        return {**cached, "cached": True,
                "elapsed_s": round(time.time() - started, 2)}

    rendered = render_array(source, project.sample_rate, recipe)

    verdict = gate.verify(rendered, project.sample_rate, graph=graph,
                          source_report=report)
    if not verdict.passed:
        # The gate is not advisory. A render that fails does not ship, and the
        # user is told in words rather than shown a correlation coefficient.
        return {"ok": False, "elapsed_s": round(time.time() - started, 1),
                "failures": [{"code": f.code, "message": f.user_message or f.message}
                             for f in verdict.failures]}

    # Loudness-matched, because an unmatched A/B compares levels rather than
    # processing. Louder wins regardless of whether it is better.
    source_lufs = loudness.integrated_lufs(source, project.sample_rate)
    out_lufs = loudness.integrated_lufs(rendered, project.sample_rate)
    matched = rendered
    offset = 0.0
    if np.isfinite(source_lufs) and np.isfinite(out_lufs):
        offset = float(source_lufs - out_lufs)
        matched = rendered * (10.0 ** (offset / 20.0))
        peak = float(np.max(np.abs(matched)))
        if peak > 0.999:
            matched *= 0.999 / peak

    key = f"{graph_id}{'-full' if full else ''}"
    out_path = os.path.join(WORKDIR, f"{project.id}-{key}.wav")
    wavio.write_wav(out_path, matched, project.sample_rate, 24)
    project.renders[key] = out_path

    # THE DOWNLOAD IS NOT THE FILE YOU AUDITIONED, AND THAT IS DELIBERATE.
    # Everything above this point is loudness-matched so the A/B compares
    # processing instead of level. Delivering that same file would hand back
    # audio deliberately pulled to the source's loudness — undoing whatever the
    # chain did to level, which for anything with a limiter in it is most of the
    # point. So the unmatched render is kept alongside, and it is the one that
    # leaves the building. The two differ by exactly `match_offset_db`.
    deliver_key = f"{key}-deliver"
    deliver_path = os.path.join(WORKDIR, f"{project.id}-{deliver_key}.wav")
    wavio.write_wav(deliver_path, rendered, project.sample_rate, 24)
    project.renders[deliver_key] = deliver_path

    source_key = "full" if full else "excerpt"
    if source_key not in project.renders:
        source_path = os.path.join(WORKDIR, f"{project.id}-{source_key}-src.wav")
        wavio.write_wav(source_path, source, project.sample_rate, 24)
        project.renders[source_key] = source_path

    result = {
        "ok": True, "render_key": key, "source_key": source_key,
        "deliver_key": deliver_key, "full": full,
        "elapsed_s": round(time.time() - started, 1),
        "match_offset_db": round(offset, 2),
        "gate": {"passed": True, "checked": verdict.checked},
        "after": measure(rendered, project.sample_rate).to_dict(),
    }
    # Only successful renders are cached. A gate failure returns before this
    # point, deliberately: caching a refusal would mean a fix to the gate could
    # not be seen without restarting the process.
    RENDER_CACHE[cache_key] = result
    return {**result, "cached": False}


@app.get("/api/projects/{project_id}/audio/{key}")
def get_audio(project_id: str, key: str) -> FileResponse:
    project = project_or_404(project_id)
    path = project.renders.get(key)
    if path is None or not os.path.exists(path):
        raise HTTPException(404, "no such render")
    return FileResponse(path, media_type="audio/wav")


@app.get("/api/projects/{project_id}/download/{key}")
def download_audio(project_id: str, key: str) -> FileResponse:
    """The same bytes as /audio, but named and sent as an attachment.

    A separate endpoint rather than a query flag because the two have different
    contracts: /audio feeds an <audio> element and wants to stream inline, this
    one wants Content-Disposition and a filename a person will recognise three
    weeks later in a Downloads folder full of `render.wav`.

    The name carries the source stem and the graph id. The graph id is the part
    that matters — it is the key the recipe is stored under, so a file that
    turns up later can be traced back to the exact chain that produced it.
    """
    project = project_or_404(project_id)
    path = project.renders.get(key)
    if path is None or not os.path.exists(path):
        raise HTTPException(404, "no such render")

    # ASCII, and no characters that need quoting. Anything outside that set
    # comes back as an RFC 5987 `filename*=utf-8''...` header with no plain
    # fallback, which is correct but unreadable and depends on the client
    # decoding it. A filename is not the place to be clever.
    stem = os.path.splitext(os.path.basename(project.name))[0]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-") or "audio"
    graph_id = key.split("-")[0]
    scope = "full" if "-full" in key else "excerpt"
    return FileResponse(
        path, media_type="audio/wav",
        filename=f"{stem}-audionova-{graph_id}-{scope}.wav")


# The render seed, in one place. Bit-identical re-rendering needs it, so it
# cannot stay an inline literal at each call site — a second copy is a second
# thing to forget when one of them changes.
RENDER_SEED = 7
RECIPE_FORMAT = "audionova_recipe_v1"


def _recipe_hash(recipe: dict[str, Any], source_key: str) -> str:
    """A cache key for one render.

    Canonical JSON — sorted keys, no incidental whitespace — so two graphs that
    differ only in the order their fields were written hash the same. The source
    is in the key because the same chain on a different excerpt is a different
    render, and leaving it out would be the kind of cache bug that returns
    somebody else's audio.
    """
    payload = json.dumps({"source": source_key, "recipe": recipe},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()


# WHY THIS IS A WHOLE-RENDER CACHE AND NOT THE STAGE-LEVEL ONE THE PLAN ASKS FOR.
#
# Measured on a 20-second excerpt, best of three:
#
#     passthrough (process spawn + WAV I/O)      271 ms
#     one EQ node                               293 ms   node cost   21 ms
#     four nodes incl. drive and plate reverb   997 ms   node cost  725 ms
#
# Stage-level caching means rendering the chain in segments, and every extra
# segment pays that 271 ms again. It only pays off when the nodes it SKIPS cost
# more than the process it ADDS — and on the current node set the one expensive
# node is the reverb, which sits at the end of the chain. Changing a late node
# means re-rendering the late node; changing an early one invalidates everything
# after it. Neither case wins.
#
# That changes when the expensive nodes move early — a NAM amp model and a
# cabinet IR are Phase 2, they are roughly 60% of a chain's cost, and they sit
# at the front. Then the plan's 3-5x is real. Building it now would add a cache
# invalidation surface to save nothing measurable.
#
# The whole-render cache below is a smaller idea that pays immediately: undo,
# redo, re-auditioning a graph, and A/B-ing two chains all re-render something
# already rendered, and those are the commonest actions in a session.
RENDER_CACHE: dict[str, dict[str, Any]] = {}


@app.post("/api/projects/{project_id}/reference")
async def set_reference(project_id: str, file: UploadFile) -> dict[str, Any]:
    """Take a reference recording, keep its balance, discard the recording.

    The audio exists in this process for the length of one request and is never
    written to disk. What survives is 256 numbers on a fixed log grid, which is
    not audio, cannot be inverted into audio, and is not a fingerprint — the
    grid is coarse and the curve is level-normalised during the fit, so two
    different masters with the same tonal balance reduce to the same numbers.

    The filename is not stored either. It is used once, in an error message, if
    the file will not decode.
    """
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "file too large")
    project = project_or_404(project_id)

    samples, rate = _decode_upload(raw, file.filename or "reference.wav")

    project.reference_curve = match_eq.reference_curve(samples, rate)
    project.reference_lufs = float(loudness.integrated_lufs(samples, rate))

    # Named `del` rather than left to fall out of scope, because the whole
    # promise of this endpoint is the discard and it should be visible.
    del samples, raw

    return {
        "stored": "spectral balance only",
        "points": int(project.reference_curve.shape[0]),
        "reference_lufs": round(project.reference_lufs, 2),
        "kept": ["a 256-point log-frequency curve", "integrated loudness"],
        "not_kept": ["the audio", "the filename", "any content hash"],
    }


class MatchRequest(BaseModel):
    max_bands: int = Field(default=5, ge=1, le=8)
    clamp_db: float = Field(default=6.0, ge=1.0, le=12.0)


@app.post("/api/projects/{project_id}/graphs/{graph_id}/match")
def match_reference(project_id: str, graph_id: str,
                    body: MatchRequest) -> dict[str, Any]:
    """Fit an EQ that moves this source toward the reference's balance.

    THE MODEL IS NEVER ASKED FOR A MATCH CURVE. This is a deterministic fit
    against a measurement, which is exactly the kind of thing a language model
    is worst at and a least-squares fit is best at. The result is written back
    as an ordinary `parametric_eq` node, so it stays readable and editable
    rather than becoming an opaque 8000-tap FIR an engineer cannot argue with.

    The level offset is reported and NOT folded into the bands. Hiding a
    broadband gain change inside an EQ makes the EQ look like it did something
    it did not.
    """
    project = project_or_404(project_id)
    if project.reference_curve is None:
        raise HTTPException(
            409, "no reference has been set for this project; POST one to "
                 "/api/projects/{id}/reference first")
    entry = project.graphs.get(graph_id)
    if entry is None:
        raise HTTPException(404, "no such graph")

    result = match_eq.match_to_curve(
        project.excerpt, project.reference_curve, project.sample_rate,
        max_bands=body.max_bands, clamp_db=body.clamp_db)

    if not result.bands:
        raise HTTPException(
            422, "the fit produced no bands: this source and the reference "
                 "already have the same balance, or one of them has too little "
                 "energy to compare")

    graph = entry["graph"].model_copy(deep=True)
    # Replace rather than append, so matching twice does not stack two match
    # EQs. The id is fixed for exactly that reason.
    graph.nodes = [n for n in graph.nodes if n.id != "match_eq"]
    graph.nodes.insert(0, GraphNode(
        id="match_eq", type="parametric_eq",
        params={"bands": [b.as_dict() for b in result.bands]},
        rationale=(f"fitted against the reference's spectral balance; "
                   f"error {result.initial_error_db:.2f} -> "
                   f"{result.rounded_error_db:.2f} dB RMS"),
    ))

    validated = validate(graph, project.excerpt_report)
    if not validated.ok:
        raise HTTPException(422, "; ".join(str(f) for f in validated.fatal))

    new_id = _store_graph(project, validated.graph, "matched to reference",
                          parent=graph_id, origin="match")
    return {
        "graph_id": new_id, "parent": graph_id, "origin": "match",
        "bands": [b.as_dict() for b in result.bands],
        "error_db_before": round(result.initial_error_db, 2),
        "error_db_after": round(result.rounded_error_db, 2),
        "level_offset_db": round(result.level_offset_db, 2),
        "clamped": result.clamped,
        "repairs": validated.repairs,
        "nodes": _describe(validated.graph),
    }


@app.get("/api/projects/{project_id}/graphs/{graph_id}/recipe")
def export_recipe(project_id: str, graph_id: str) -> dict[str, Any]:
    """A graph as a portable document.

    WHAT MAKES IT PORTABLE IS THE ENGINE VERSION, not the parameters. The
    determinism contract is bit-exact within one (engine_version, arch), so a
    recipe without a version stamp is a set of numbers that will probably
    reproduce — and "probably" is not what a reproducibility promise means.
    Import states the mismatch rather than silently rendering something else.

    The graph is exported, not the engine recipe. The engine recipe is a lower
    representation this can always regenerate; the graph is the editable one, so
    an imported recipe stays as readable and as adjustable as a planned chain.
    """
    project = project_or_404(project_id)
    entry = project.graphs.get(graph_id)
    if entry is None:
        raise HTTPException(404, "no such graph")

    return {
        "format": RECIPE_FORMAT,
        "engine_version": ENGINE_VERSION,
        "graph": entry["graph"].model_dump(mode="json"),
        "render": {"seed": RENDER_SEED, "output_format": "float64"},
        "provenance": {
            # The file it came from, because a recipe with no idea what it was
            # made for is hard to trust six months later. This is the user's own
            # material and their own export — unlike a reference analysis, which
            # deliberately stores no filename because the file is someone else's.
            "source_name": project.name,
            "source_sample_rate": project.sample_rate,
            "source_channels": project.channels,
            "label": entry.get("label", ""),
            "origin": entry.get("origin", "planner"),
        },
    }


class ImportRequest(BaseModel):
    format: str
    engine_version: str = ""
    graph: dict[str, Any]
    render: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)


@app.post("/api/projects/{project_id}/recipes")
def import_recipe(project_id: str, body: ImportRequest) -> dict[str, Any]:
    """Load an exported recipe onto this project.

    Validated like anything else. An imported recipe is not more trusted than a
    planned one — it may have been exported before a validator existed, or
    edited by hand in a text editor on the way through.
    """
    project = project_or_404(project_id)

    if body.format != RECIPE_FORMAT:
        raise HTTPException(
            415, f"this is not an Audionova recipe: format is {body.format!r}, "
                 f"expected {RECIPE_FORMAT!r}")

    try:
        graph = EffectGraph.model_validate(body.graph)
    except Exception as error:
        raise HTTPException(422, f"the recipe's graph is not valid: {error}")

    result = validate(graph, project.excerpt_report)
    if not result.ok:
        raise HTTPException(422, "; ".join(str(f) for f in result.fatal))

    # STATED, NOT ENFORCED. Refusing to load a recipe from another engine
    # version would make old recipes worthless; loading it silently would make
    # the bit-identical claim false. So it renders, and says what it is.
    warnings = []
    if body.engine_version and body.engine_version != ENGINE_VERSION:
        warnings.append(
            f"this recipe was exported from engine {body.engine_version} and "
            f"this is {ENGINE_VERSION}. It will render, but not necessarily "
            f"bit-identically to the original.")

    label = body.provenance.get("label") or "imported recipe"
    new_id = _store_graph(project, result.graph, f"imported: {label}",
                          parent=None, origin="imported")
    return {"graph_id": new_id, "origin": "imported", "warnings": warnings,
            "repairs": result.repairs, "nodes": _describe(result.graph)}


@app.get("/api/intents")
def intents() -> dict[str, Any]:
    """The closed refinement vocabulary, for the UI to offer as buttons.

    Closed because the arithmetic behind each is fixed in Python. Free text
    would put the model back in charge of what "slightly" means.
    """
    return {
        "intents": [
            {"id": name, "label": name.replace("_", " "),
             "node_type": target[0], "parameter": target[1]}
            for name, target in ladder.INTENT_TO_PARAM.items()
        ],
        "quantifiers": sorted(ladder.QUANTIFIERS, key=ladder.QUANTIFIERS.get),
    }


@app.on_event("shutdown")
def cleanup() -> None:
    shutil.rmtree(WORKDIR, ignore_errors=True)
