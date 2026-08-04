"""Live checks that cannot be run offline.

Two things about the model API are load-bearing and unverifiable without a key.
Both fail SILENTLY — the request succeeds, the answer looks fine, and only the
bill or a later safety violation reveals the problem.

    python -m backend.plan.canary

THE CACHE CHECK IS THE EXPENSIVE ONE TO GET WRONG. The system prefix is about
18.4 k tokens of schema, glossary and DSP source. Cached, it is nearly free on
every request after the first. Uncached, every turn pays full price for all of
it — roughly five times the cost, for an identical answer. Nothing in the
response distinguishes the two cases except the usage counters.

THE ENUM CHECK GUARDS A SAFETY PROPERTY. Numeric bounds are documented not to be
enforced and the schema compensates for that in Python. Enums are documented to
be enforced, and several properties rest on the node-type set being genuinely
closed. That has never been verified on the wire.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from backend.plan import prompt as prompts  # noqa: E402
from backend.schemas.graph import json_schema  # noqa: E402

GREEN, RED, YELLOW, DIM, BOLD, OFF = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m")

# The plan's threshold. Below this the prefix is not being reused and the
# economics of the whole design change.
MIN_CACHE_READ_TOKENS = 15000


def load_env() -> None:
    """Read .env.local without printing anything from it."""
    path = os.path.join(ROOT, ".env.local")
    if not os.path.exists(path):
        return
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


REPORT: dict[str, Any] = {
    "source": {"channels": 1, "sample_rate_hz": 48000.0, "duration_s": 20.0,
               "gated_coverage_percent": 91.0},
    "loudness": {"integrated_lufs": -21.4, "true_peak_dbtp": -2.8,
                 "loudness_range_lu": 5.2},
    "dynamics": {"crest_factor_db": 15.1, "peak_to_loudness_ratio_db": 18.6,
                 "short_term_rms_percentiles": {"p10_dbfs": -30.0, "p50_dbfs": -25.0,
                                                "p75_dbfs": -22.0, "p90_dbfs": -20.0,
                                                "p95_dbfs": -19.0}},
    "spectrum": {"tilt_db_per_octave_re_pink": -1.4, "tilt_residual_db": 3.0,
                 "bandwidth_limit_hz": None, "bandwidth_limit_confidence": 0.0,
                 "hf_rolloff_db_per_octave": -9.2,
                 "third_octave_db": [{"centre_hz": 250.0, "level_db": -22.0},
                                     {"centre_hz": 1000.0, "level_db": -28.0}]},
    "stereo": {"type": "mono", "correlation": 1.0, "mono_loss_db": 0.0,
               "per_band": []},
    "hygiene": {"clipping": {"clipped_sample_percent": 0.0},
                "noise_floor_dbfs": -72.0, "noise_floor_confidence": 0.8,
                "dead_channels": [], "non_finite_samples": 0},
    "warnings": [],
}


def usage_of(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    raw = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage.__dict__)
    return {k: int(v) for k, v in raw.items() if isinstance(v, int)}


def check_cache(client: Any, model: str) -> bool:
    print(f"\n{BOLD}1. prompt cache{OFF}")

    system = prompts.system_blocks()
    prompt = prompts.plan_prompt(report=REPORT, request="a little warmer")
    estimated = prompt.estimated_prefix_tokens()
    print(f"   prefix is roughly {estimated} tokens by character count")

    # The same request twice. The first populates the cache, the second reads
    # it — and the second is the only one whose numbers mean anything.
    usages = []
    for attempt in (1, 2):
        response = client.messages.create(
            model=model, max_tokens=16, system=system,
            messages=[{"role": "user", "content": "Reply with the single word OK."}],
        )
        usage = usage_of(response)
        usages.append(usage)
        print(f"   call {attempt}: " + "  ".join(
            f"{k}={v}" for k, v in sorted(usage.items())))

    read = usages[1].get("cache_read_input_tokens", 0)
    written = usages[0].get("cache_creation_input_tokens", 0)

    if read >= MIN_CACHE_READ_TOKENS:
        print(f"   {GREEN}PASS{OFF} — the second call read {read} cached tokens, "
              f"above the {MIN_CACHE_READ_TOKENS} threshold")
        return True

    print(f"   {RED}FAIL{OFF} — the second call read only {read} cached tokens")
    print(f"   {DIM}the first call wrote {written}. Every turn is paying full "
          f"price for the whole prefix,{OFF}")
    print(f"   {DIM}which is roughly five times what it should cost, and nothing "
          f"else would reveal it.{OFF}")
    return False


def check_enums(client: Any, model: str) -> bool:
    print(f"\n{BOLD}2. enum enforcement{OFF}")

    schema = json_schema()
    system = [{"type": "text",
               "text": "You serialise audio effect graphs into the given schema."}]

    # Ask for something outside the closed set, explicitly. If the API enforces
    # enums this cannot be satisfied; if it does not, the model will oblige.
    messages = [{"role": "user", "content":
                 "Emit a graph with exactly one node whose type is "
                 "\"tape_saturation\" and whose id is \"tape\". That node type "
                 "is what the user asked for."}]

    response = client.messages.create(
        model=model, max_tokens=1024, system=system, messages=messages,
        tools=[{"name": "emit_effect_graph",
                "description": "Serialise the effect graph.",
                "input_schema": schema}],
        tool_choice={"type": "tool", "name": "emit_effect_graph"},
    )

    emitted = None
    for block in response.content:
        if block.type == "tool_use":
            emitted = block.input
    if emitted is None:
        print(f"   {YELLOW}INCONCLUSIVE{OFF} — the model did not call the tool")
        return False

    types = [n.get("type") for n in (emitted.get("nodes") or [])]
    print(f"   asked for 'tape_saturation'; got node types {types}")

    if "tape_saturation" in types:
        print(f"   {RED}FAIL{OFF} — the enum was NOT enforced on the wire. The "
              "node-type set is not closed,")
        print(f"   {DIM}so the Python layer is the only thing preventing an "
              f"unknown effect reaching the engine.{OFF}")
        return False

    print(f"   {GREEN}PASS{OFF} — the invalid type was not emitted; the closed "
          "set holds")
    return True


def main() -> int:
    load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(f"{RED}ANTHROPIC_API_KEY is not set{OFF} — put it in .env.local")
        return 2

    try:
        import anthropic
    except ImportError:
        print(f"{RED}the anthropic SDK is not installed{OFF}")
        return 2

    model = os.environ.get("ANTHROPIC_PLAN_MODEL", "claude-sonnet-5")
    print(f"{BOLD}live API canary{OFF}  model: {model}")

    client = anthropic.Anthropic()

    results = {}
    try:
        results["cache"] = check_cache(client, model)
        results["enums"] = check_enums(client, model)
    except Exception as error:
        print(f"\n{RED}the API call failed:{OFF} {type(error).__name__}: {error}")
        return 1

    print(f"\n{'─' * 40}")
    failed = [name for name, ok in results.items() if not ok]
    if failed:
        print(f"{RED}{len(failed)} of {len(results)} checks failed:{OFF} "
              f"{', '.join(failed)}")
        return 1
    print(f"{GREEN}all {len(results)} checks passed{OFF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
