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

import os
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
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from backend.audio import wavio  # noqa: E402
from backend.engine.render import render_array  # noqa: E402
from backend.eval import corpus  # noqa: E402
from backend.measure import loudness  # noqa: E402
from backend.measure.report import measure  # noqa: E402
from backend.plan import ladder, pipeline, rules  # noqa: E402
from backend.plan.canary import load_env  # noqa: E402
from backend.plan.validate import validate  # noqa: E402
from backend.schemas.graph import EffectGraph  # noqa: E402
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
                 parent: str | None) -> str:
    graph_id = uuid.uuid4().hex[:8]
    project.graphs[graph_id] = {"graph": graph, "parent": parent, "label": label}
    project.head = graph_id
    return graph_id


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

    recipe = graph.to_recipe(seed=7, output_sample_rate=project.sample_rate,
                             output_format="float64")
    started = time.time()
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

    source_key = "full" if full else "excerpt"
    if source_key not in project.renders:
        source_path = os.path.join(WORKDIR, f"{project.id}-{source_key}-src.wav")
        wavio.write_wav(source_path, source, project.sample_rate, 24)
        project.renders[source_key] = source_path

    return {
        "ok": True, "render_key": key, "source_key": source_key,
        "elapsed_s": round(time.time() - started, 1),
        "match_offset_db": round(offset, 2),
        "gate": {"passed": True, "checked": verdict.checked},
        "after": measure(rendered, project.sample_rate).to_dict(),
    }


@app.get("/api/projects/{project_id}/audio/{key}")
def get_audio(project_id: str, key: str) -> FileResponse:
    project = project_or_404(project_id)
    path = project.renders.get(key)
    if path is None or not os.path.exists(path):
        raise HTTPException(404, "no such render")
    return FileResponse(path, media_type="audio/wav")


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
