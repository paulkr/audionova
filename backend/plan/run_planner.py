"""Run the planner on a real file and compare it against the deterministic floor.

    python -m backend.plan.run_planner audio/stems/vocals.wav "a bit warmer"

Measures the source, plans a chain twice — once with the model, once with the
rules — validates both, renders both, and gates both.

THE COMPARISON IS THE POINT. A model chain that merely matches a handful of
thresholds has not earned a cost 200 to 700 times the DSP it configures. What it
has to supply is the half the rules provably cannot: the judgement about what a
source *should* sound like, which is instrument knowledge rather than
measurement. The clearest evidence of that gap so far is the acoustic guitar's
250 Hz plateau — a broad lift the rule cannot justify cutting and a person could.

Both chains are written out as loudness-matched audio, because the numbers below
cannot settle whether either sounds good.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from backend.audio import wavio  # noqa: E402
from backend.engine.render import render_array  # noqa: E402
from backend.eval import corpus  # noqa: E402
from backend.measure import loudness  # noqa: E402
from backend.measure.report import measure  # noqa: E402
from backend.plan import pipeline, rules  # noqa: E402
from backend.plan.canary import load_env  # noqa: E402
from backend.plan.validate import validate  # noqa: E402
from backend.schemas.graph import EffectGraph  # noqa: E402
from backend.verify import gate  # noqa: E402

GREEN, RED, YELLOW, DIM, BOLD, OFF = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m")


def describe(graph: EffectGraph) -> list[str]:
    lines = []
    for node in list(graph.nodes) + list(graph.master):
        if node.type in ("parametric_eq", "dynamic_eq"):
            for band in node.params.get("bands", []):
                lines.append(
                    f"    {node.id:10s} {band['filter_type']:10s} "
                    f"{band['center_freq_hz']:>6.0f} Hz  "
                    f"{band.get('gain_db', 0.0):+5.1f} dB  Q{band.get('q_factor', 0.7)}")
        else:
            interesting = {k: v for k, v in node.params.items()
                           if k in ("threshold_dbfs", "ratio_to_1", "attack_ms",
                                    "release_ms", "drive_db", "decay_s",
                                    "ceiling_dbtp", "width_percent", "bit_depth")}
            detail = " ".join(f"{k}={v}" for k, v in interesting.items())
            mix = f" mix={node.mix}" if node.mix != 1.0 else ""
            lines.append(f"    {node.id:10s} {node.type:20s} {detail}{mix}")
    return lines


def render_and_gate(source: np.ndarray, rate: float, graph: EffectGraph,
                    report: dict[str, Any], label: str, out_dir: str,
                    stem_name: str) -> dict[str, Any]:
    recipe = graph.to_recipe(seed=11, output_sample_rate=rate,
                             output_format="float64")
    rendered = render_array(source, rate, recipe)

    verdict = gate.verify(rendered, rate, graph=graph, source_report=report)

    # Loudness-matched, because an unmatched comparison is a comparison of
    # levels. Louder wins a blind A/B whether or not it is better.
    source_lufs = loudness.integrated_lufs(source, rate)
    out_lufs = loudness.integrated_lufs(rendered, rate)
    matched = rendered
    if np.isfinite(source_lufs) and np.isfinite(out_lufs):
        matched = rendered * (10.0 ** ((source_lufs - out_lufs) / 20.0))
        peak = float(np.max(np.abs(matched)))
        if peak > 0.999:
            matched *= 0.999 / peak

    path = os.path.join(out_dir, f"{stem_name}--{label}-matched.wav")
    wavio.write_wav(path, matched, rate, 24)

    after = measure(rendered, rate).to_dict()
    return {"verdict": verdict, "path": path, "after": after,
            "nodes": len(graph.nodes) + len(graph.master)}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("request", nargs="?",
                        default="Make this sound better, without changing its character.")
    parser.add_argument("--seconds", type=float, default=20.0)
    # Unset means the same models the API runs, including any environment
    # override — see pipeline.configured_models.
    parser.add_argument("--plan-model", default=None)
    parser.add_argument("--emit-model", default=None)
    args = parser.parse_args(argv)

    load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(f"{RED}ANTHROPIC_API_KEY is not set{OFF}")
        return 2

    stem_name = os.path.splitext(os.path.basename(args.source))[0]
    out_dir = os.path.join(ROOT, "demo", "out")
    os.makedirs(out_dir, exist_ok=True)

    print(f"{BOLD}source{OFF}  {args.source}")
    print(f"{BOLD}request{OFF} {args.request!r}\n")

    source, rate = corpus.load_excerpt(args.source, args.seconds)
    report = measure(source, rate).to_dict()

    loud, dyn, spec = report["loudness"], report["dynamics"], report["spectrum"]
    print(f"  {loud['integrated_lufs']} LUFS   true peak {loud['true_peak_dbtp']} dBTP   "
          f"PLR {dyn['peak_to_loudness_ratio_db']} dB   "
          f"rolloff {spec['hf_rolloff_db_per_octave']} dB/oct")
    if report["warnings"]:
        print(f"  warnings: {[w['code'] for w in report['warnings']]}")

    # --- the floor ----------------------------------------------------------
    print(f"\n{BOLD}rule-based floor{OFF}")
    rule_graph, explanation = rules.plan(report)
    rule_result = validate(rule_graph, report)
    for line in describe(rule_result.graph):
        print(line)
    print(f"  {DIM}declined: {'; '.join(explanation.declined) or 'nothing'}{OFF}")

    # --- the model ----------------------------------------------------------
    client = pipeline.AnthropicClient(plan_model=args.plan_model,
                                      emit_model=args.emit_model)
    print(f"\n{BOLD}model planner{OFF}  {client.plan_model} -> {client.emit_model}")
    counters = pipeline.Counters()
    started = time.time()
    outcome = pipeline.plan(client, report=report, request=args.request,
                            counters=counters)
    elapsed = time.time() - started

    if not outcome.ok:
        print(f"  {RED}failed{OFF} after {outcome.attempts} attempts: {outcome.fatal}")
        return 1

    for line in describe(outcome.graph):
        print(line)
    print(f"  {DIM}{outcome.attempts} attempt(s), {elapsed:.0f}s"
          + (f", repairs: {outcome.repairs}" if outcome.repairs else "") + f"{OFF}")

    # --- render, gate, write both ------------------------------------------
    print(f"\n{BOLD}renders{OFF}")
    results = {}
    for label, graph in (("rules", rule_result.graph), ("model", outcome.graph)):
        results[label] = render_and_gate(source, rate, graph, report,
                                         label, out_dir, stem_name)
        verdict = results[label]["verdict"]
        mark = f"{GREEN}passed{OFF}" if verdict.passed else f"{RED}FAILED{OFF}"
        print(f"  {label:6s} {results[label]['nodes']} nodes   gate {mark}"
              + ("" if verdict.passed else f" — {verdict.summary()}"))

    print(f"\n{BOLD}what each did{OFF}")
    print(f"  {'':22s} {'source':>9s} {'rules':>9s} {'model':>9s}")
    for label, path in (("crest factor dB", "dynamics.crest_factor_db"),
                        ("PLR dB", "dynamics.peak_to_loudness_ratio_db"),
                        ("loudness range LU", "loudness.loudness_range_lu"),
                        ("tilt re pink", "spectrum.tilt_db_per_octave_re_pink"),
                        ("HF rolloff dB/oct", "spectrum.hf_rolloff_db_per_octave")):
        def dig(node, p=path):
            for key in p.split("."):
                node = (node or {}).get(key)
            return node
        row = [dig(report), dig(results["rules"]["after"]), dig(results["model"]["after"])]
        cells = "".join(f"{v:>10.2f}" if isinstance(v, (int, float)) else f"{'-':>10s}"
                        for v in row)
        print(f"  {label:22s}{cells}")

    print(f"\n{BOLD}explanation{OFF}")
    print(f"  {outcome.graph.explanation[:400]}")

    print(f"\n{BOLD}listen{OFF}")
    print(f"  afplay {args.source}")
    for label in ("rules", "model"):
        print(f"  afplay {results[label]['path']}")

    usage = client.last_usage
    cached = client.cache_hit_tokens()
    print(f"\n{DIM}cache read {cached} tokens on the final call; "
          f"usage {json.dumps({k: v for k, v in usage.items() if isinstance(v, int)})}{OFF}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
