"""The two-stage planner: PLAN, EMIT, validate, repair.

The model is behind a `Client` protocol with two methods. That indirection exists
so the whole loop — including every repair path and every failure mode — is
testable with a fake, without an API key and without spending anything. The paths
that matter most here are the ones a live test would exercise least: a model that
emits malformed JSON, one that violates the boost budget twice in a row, one that
returns something unparseable on the final retry.

BOUNDED, WITH A FALLBACK THAT IS ALWAYS A GRAPH THAT ALREADY PASSED. Two retries,
then fall back to the previous accepted graph rather than to a default chain. A
default chain is a guess about audio nobody measured; the previous graph is
something this same pipeline already validated for this same source.

THE COUNTERS ARE THE PRODUCT'S EARLY WARNING SYSTEM. Repair rate is the one to
watch: it rising means the prompt has drifted from what the validators expect, and
it rises long before quality visibly drops. Targets from the plan are repair under
15%, retry under 5%, fallback under 0.5%.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from backend.plan import prompt as prompts
from backend.plan.validate import ValidationResult, validate
from backend.schemas.graph import EffectGraph, json_schema

MAX_RETRIES = 2


class Client(Protocol):
    """What the pipeline needs from a model. Two calls, deliberately narrow."""

    def complete(self, system: list[dict[str, Any]],
                 messages: list[dict[str, Any]]) -> str:
        """Unconstrained text, for stage 1."""
        ...

    def complete_structured(self, system: list[dict[str, Any]],
                            messages: list[dict[str, Any]],
                            schema: dict[str, Any]) -> str:
        """JSON matching the schema, for stage 2. Returns the raw text."""
        ...


@dataclass
class Counters:
    """Service-level indicators, accumulated across requests."""

    turns: int = 0
    repaired: int = 0        # accepted after a deterministic repair
    retried: int = 0         # needed at least one model retry
    fell_back: int = 0       # exhausted retries
    unparseable: int = 0     # stage 2 returned something that was not a graph

    def rate(self, count: int) -> float:
        return count / self.turns if self.turns else 0.0

    def report(self) -> dict[str, float | int]:
        return {
            "turns": self.turns,
            "repair_rate": round(self.rate(self.repaired), 4),
            "retry_rate": round(self.rate(self.retried), 4),
            "fallback_rate": round(self.rate(self.fell_back), 4),
            "unparseable_rate": round(self.rate(self.unparseable), 4),
        }

    def breaches(self) -> list[str]:
        """Which SLI targets are currently missed.

        Reported rather than enforced. A breach means the prompt needs attention,
        not that the request should fail — refusing to serve because a rolling
        rate is high would turn a quality signal into an outage.
        """
        out = []
        if self.turns < 20:
            return out  # too few samples for a rate to mean anything
        if self.rate(self.repaired) > 0.15:
            out.append(f"repair rate {self.rate(self.repaired):.1%} exceeds 15%")
        if self.rate(self.retried) > 0.05:
            out.append(f"retry rate {self.rate(self.retried):.1%} exceeds 5%")
        if self.rate(self.fell_back) > 0.005:
            out.append(f"fallback rate {self.rate(self.fell_back):.1%} exceeds 0.5%")
        return out


@dataclass
class PlanOutcome:
    graph: EffectGraph | None
    reasoning: str = ""
    validation: ValidationResult | None = None
    attempts: int = 0
    repairs: list[str] = field(default_factory=list)
    fatal: list[str] = field(default_factory=list)
    used_fallback: bool = False

    # Seconds per stage. Recorded because the first end-to-end measurement was
    # 128.7 s against a 45 s p50 target, and "the planner is slow" is not
    # actionable — the two stages have different models, different prompts and
    # very different jobs, and only one of them is doing reasoning worth paying
    # for. `emit` accumulates across retries, so a repair loop shows up here as
    # a large emit rather than hiding inside the total.
    stage_seconds: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.graph is not None and not self.fatal


def extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model reply.

    Tolerant of the two things that actually happen — a fenced code block, and a
    sentence before or after the object — and intolerant of everything else. A
    parser that tried harder than this would start succeeding on replies that are
    not really graphs, which is worse than failing.
    """
    stripped = text.strip()

    fenced = re.search(r"```(?:json)?\s*(.+?)```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found in the reply")
        parsed = json.loads(stripped[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def plan(
    client: Client,
    *,
    report: dict[str, Any],
    request: str,
    classification: dict[str, Any] | None = None,
    presets: list[dict[str, Any]] | None = None,
    previous_graph: EffectGraph | None = None,
    counters: Counters | None = None,
) -> PlanOutcome:
    """Run the full pipeline for one turn."""
    counters = counters or Counters()
    counters.turns += 1

    # --- stage 1: unconstrained reasoning ----------------------------------
    stage1 = prompts.plan_prompt(
        report=report, request=request,
        classification=classification, presets=presets,
    )
    started = time.monotonic()
    reasoning = client.complete(stage1.system, stage1.messages)
    plan_seconds = time.monotonic() - started

    outcome = PlanOutcome(graph=None, reasoning=reasoning)
    outcome.stage_seconds = {"plan": round(plan_seconds, 2), "emit": 0.0}

    # --- stage 2: constrained emission, with a bounded repair loop ---------
    emit = prompts.emit_prompt(report=report, request=request, reasoning=reasoning)
    schema = json_schema()
    findings: list[str] = []
    raw = ""

    for attempt in range(MAX_RETRIES + 1):
        outcome.attempts = attempt + 1

        if attempt > 0:
            counters.retried += 1 if attempt == 1 else 0
            emit = prompts.repair_prompt(
                previous=emit, graph_json=raw, findings=findings)

        started = time.monotonic()
        raw = client.complete_structured(emit.system, emit.messages, schema)
        outcome.stage_seconds["emit"] = round(
            outcome.stage_seconds["emit"] + time.monotonic() - started, 2)

        try:
            graph = EffectGraph.model_validate(extract_json(raw))
        except Exception as error:
            # A schema failure is as retryable as a semantic one, and for the
            # same reason: the model can be told what was wrong.
            findings = [f"the reply did not parse as a valid graph: {error}"]
            if attempt == 0:
                counters.unparseable += 1
            continue

        result = validate(graph, report)
        outcome.validation = result
        outcome.repairs = list(result.repairs)

        if result.ok:
            outcome.graph = result.graph
            outcome.fatal = []
            if result.repairs:
                counters.repaired += 1
            return outcome

        findings = [str(f) for f in result.fatal]
        outcome.fatal = findings

    # --- exhausted -------------------------------------------------------
    counters.fell_back += 1
    if previous_graph is not None:
        # The fallback is a graph this pipeline already accepted for this source,
        # not a default chain. A default is a guess about audio nobody measured.
        outcome.graph = previous_graph
        outcome.used_fallback = True
        outcome.fatal = []
    return outcome


# ---------------------------------------------------------------------------
# Clients


class ScriptedClient:
    """A fake that returns prepared replies. For tests.

    Records what it was asked, so a test can assert on the prompt as well as on
    the behaviour — the cache boundary and the absence of per-request data in the
    prefix are properties of the prompt, not of the reply.
    """

    def __init__(self, reasoning: str = "reasoning",
                 emissions: list[str] | None = None) -> None:
        self.reasoning = reasoning
        self.emissions = list(emissions or [])
        self.calls: list[dict[str, Any]] = []

    def complete(self, system, messages) -> str:
        self.calls.append({"kind": "text", "system": system, "messages": messages})
        return self.reasoning

    def complete_structured(self, system, messages, schema) -> str:
        self.calls.append({"kind": "structured", "system": system,
                           "messages": messages, "schema": schema})
        if not self.emissions:
            raise AssertionError("ScriptedClient ran out of prepared emissions")
        return self.emissions.pop(0)


class AnthropicClient:
    """The real client.

    UNTESTED AGAINST THE LIVE API — nothing in this repository has ever made a
    model call. The shape follows the documented API, and the first thing to do
    when a key is available is to check the two things this cannot verify
    offline: that `cache_read_input_tokens` exceeds 15000 on a repeated request,
    and that the structured output actually honours the schema's enums.

    The SDK import is deliberately lazy so that neither the tests nor the
    evaluation harnesses need the dependency installed.
    """

    def __init__(self, *, plan_model: str = "claude-opus-5",
                 emit_model: str = "claude-sonnet-5",
                 max_tokens: int = 8192) -> None:
        self.plan_model = plan_model
        self.emit_model = emit_model
        self.max_tokens = max_tokens
        self._client: Any = None
        self.last_usage: dict[str, Any] = {}

    def _sdk(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as error:
                raise RuntimeError(
                    "the anthropic SDK is not installed; the planner runs "
                    "offline with ScriptedClient and only needs it for live calls"
                ) from error
            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, system, messages) -> str:
        response = self._sdk().messages.create(
            model=self.plan_model, max_tokens=self.max_tokens,
            system=system, messages=messages,
        )
        self.last_usage = dict(response.usage.__dict__ if response.usage else {})
        return "".join(
            block.text for block in response.content if block.type == "text")

    def complete_structured(self, system, messages, schema) -> str:
        # A tool with the schema as its input, which is the documented way to get
        # constrained output. Bounds in the schema are NOT enforced by the API —
        # see backend/schemas/graph.py — so the reply is still validated in Python.
        response = self._sdk().messages.create(
            model=self.emit_model, max_tokens=self.max_tokens,
            system=system, messages=messages,
            tools=[{
                "name": "emit_effect_graph",
                "description": "Serialise the decided effect graph.",
                "input_schema": schema,
            }],
            tool_choice={"type": "tool", "name": "emit_effect_graph"},
        )
        self.last_usage = dict(response.usage.__dict__ if response.usage else {})
        for block in response.content:
            if block.type == "tool_use":
                return json.dumps(block.input)
        raise ValueError("the model did not call the emit tool")

    def cache_hit_tokens(self) -> int:
        """Cached tokens read on the last call.

        The number the CI canary asserts on. If it is near zero on a repeated
        request, something per-request has leaked into the system prefix and the
        bill has quietly multiplied.
        """
        return int(self.last_usage.get("cache_read_input_tokens", 0) or 0)
