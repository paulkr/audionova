<h1 align="center">audionova</h1>

<p align="center">
  <strong>Describe the sound you want. Get a processed file, and the argument for every move.</strong>
</p>

<p align="center">
  <img src="docs/images/studio.png" alt="The Audionova studio: measurements before and after, a third-octave spectrum, the planner's reasoning, and the effect chain with every parameter" width="900">
</p>

Audionova turns a plain-language request like *"make this warmer and less
boxy"* into a processed audio file. It does this without asking a model to
imagine what the audio sounds like. Instead it **measures** the track
deterministically, hands those **numbers** to a planner, renders the resulting
effect chain in a **float64 C++ engine**, and **re-measures** the output before
anything is allowed to ship.

Every node in the chain arrives with the sentence that justified it, and every
parameter is editable. You can refine the result by saying what you want, A/B it
loudness-matched against the source, step back through the session, and export
the chain as a portable recipe.

> **Status: research-grade, local only.** The engine, measurement layer,
> planners, refinement, safety gate and studio all work end to end on your
> machine. There is no hosted service, and the listening study that decides
> whether the output beats an engineer's own chain has not been run yet. See
> [What is not built](#what-is-not-built).

---

## Contents

- [Why it is built this way](#why-it-is-built-this-way)
- [The pipeline](#the-pipeline)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Using the engine on its own](#using-the-engine-on-its-own)
- [Tests](#tests)
- [Repository layout](#repository-layout)
- [Documentation](#documentation)
- [What is not built](#what-is-not-built)
- [Contributing](#contributing)
- [License](#license)

## Why it is built this way

The obvious design is one model call that listens to the audio and emits
parameters. The published results say that design does not work. CLAP-guided
optimisation scored *worse than random parameters* on EQ (LLM2Fx,
arXiv:2505.20770), while a zero-shot LLM given measured DSP features did more
than twice as well. Audio embeddings also under-encode EQ detail, and they
cannot tell a 40 ms slap from a 400 ms echo.

So **the model is handed numbers**: integrated loudness, true peak, loudness
range, a gated third-octave spectrum, crest factor, stereo correlation and codec
artefacts. Everything that can be arithmetic is arithmetic, because arithmetic
can be tested and a model cannot. [`docs/architecture.md`](docs/architecture.md)
has the full argument.

Two consequences are worth knowing before you read the code:

- **Odd-looking decisions are usually measured ones.**
  [`docs/findings.md`](docs/findings.md) records every bug found and every
  lesson learned, with evidence. Read it before simplifying anything.
- **Quality claims are gated, not asserted.** The engine has a DSP measurement
  suite (THD, aliasing, intermodulation, swept-sine response, true peak) that
  gates against a committed baseline. The project also has a blinded listening
  harness, because no automated metric is trusted until it correlates with human
  judgement.

## The pipeline

```
MEASURE   deterministic DSP, no ML         loudness, spectrum, dynamics, stereo, hygiene
PLAN      model, unconstrained reasoning   reads the measurements, argues for each move
EMIT      model, constrained               transcribes that decision into a typed graph
REFINE    deterministic fitting            match-EQ, dynamics matched by rendering
RENDER    C++ engine                       float64, 96 kHz internal, exact latency
VERIFY    re-measure, hard gate            refuses corruption, overshoot, collapse
```

PLAN and EMIT are separate calls on purpose. Constrained decoding from the first
token degrades reasoning, so stage 1 thinks in prose and stage 2 only
serialises. A **rule-based planner** sits beside them. It needs no API key, and
it is the floor the model has to beat. Every move it makes names the measurement
that justified it, and it says what it deliberately left alone.

**The engine** is header-only C++20 with no dependencies: 24 DSP nodes, including
parametric, dynamic, linear-phase and zero-phase EQ, a compressor, multiband
compressor, transient designer, gate, drive, tape, console, plate and
convolution reverb, delay, chorus, stereo tools, a true-peak limiter and dither.
Every render resamples exactly twice (in to 96 kHz, out to the delivery rate),
and the limiter and dither run *after* the output conversion, because true peak
is only defined at the rate you ship.

**The studio** is a Next.js app that talks straight to a local FastAPI server.
It shows:

- measurements before and after, with deltas;
- a third-octave spectrum comparison;
- the planner's reasoning, streamed while it is written;
- every node with every parameter, editable in place;
- a gapless, loudness-matched A/B;
- chat refinement, reference matching, session history, recipe import and
  export, and an optional spoken explanation.

## Quick start

**Requirements:** a C++20 compiler (clang or GCC), Python 3.12, Node.js 22, and
macOS or Linux. If `python3.12` isn't on your `PATH`, use its full path (for
Homebrew, `/opt/homebrew/bin/python3.12`).

```sh
git clone https://github.com/paulkr/audionova.git
cd audionova

# Python backend, pinned to the versions the licence gate has reviewed
python3.12 -m venv .venv
.venv/bin/pip install -c backend/constraints.txt pip
.venv/bin/pip install -r backend/requirements.txt -c backend/constraints.txt

# C++ engine (builds engine/build/anv_render)
make -C engine render

# Frontend
npm ci
```

Then run the API and the studio in two terminals:

```sh
.venv/bin/uvicorn backend.api.server:app --port 8787    # terminal 1: the audio API
npm run dev                                             # terminal 2: the studio
```

Open <http://localhost:3000>, drop in a **WAV** file, type what you want, and
press **Plan**. Other formats are rejected on purpose. Decoding untrusted bytes
belongs in a sandboxed ingest stage that has not been built, so the API reads WAV
with its own parser and nothing else.

<p align="center">
  <img src="docs/images/landing.jpg" alt="The Audionova landing page: drop a WAV file to start" width="720">
</p>

Without an API key the studio uses the **deterministic** planner, which is
instant and needs no network. With a key, the **model** planner is the default
and streams its analysis while it reasons, which takes a few minutes. Tick
*deterministic only* to switch back. Planning and auditioning run on a 20-second
excerpt so iteration stays fast, and **Download processed WAV** renders the
whole track on demand.

## Configuration

Copy [`.env.example`](.env.example) to `.env.local`. Every variable is optional,
and real environment variables take precedence.

| Variable | Effect |
|---|---|
| `ANTHROPIC_API_KEY` | Enables the model planner and chat refinement. Without it, those return 503 and the rules planner still works. |
| `ANTHROPIC_PLAN_MODEL`, `ANTHROPIC_EMIT_MODEL` | Override the models for the two planner stages. The defaults are in `backend/plan/pipeline.py`. |
| `ELEVENLABS_API_KEY` | Enables the spoken explanation. The button is hidden without it. |
| `ELEVENLABS_VOICE_ID`, `ELEVENLABS_MODEL_ID` | Voice and TTS model overrides. |
| `NEXT_PUBLIC_API_BASE` | Where the browser finds the API. Defaults to `http://localhost:8787`. |

The API accepts cross-origin requests from loopback origins only, on any port.

## Using the engine on its own

The engine is a standalone CLI that applies a JSON **recipe** to a WAV or RF64
file:

```sh
make -C engine render
engine/build/anv_render --in input.wav --out output.wav --recipe chain.json
```

| Flag | Meaning |
|---|---|
| `--in`, `--out`, `--recipe` | Required. |
| `--block <n>` | Render block size in frames, default 4096. Output is bit-identical at any value. |
| `--quiet` | Suppress the progress report on stderr. |

A recipe is two node lists. `nodes[]` runs at the internal rate, and `master[]`
runs at the delivery rate:

```json
{
  "engine_version": "0.1.0",
  "internal_sample_rate": 96000,
  "output": { "sample_rate": 0, "format": "pcm24" },
  "nodes": [
    { "id": "eq", "type": "parametric_eq", "params": { "bands": [
      { "filter_type": "highpass", "center_freq_hz": 80, "gain_db": 0, "q_factor": 0.707 },
      { "filter_type": "bell", "center_freq_hz": 320, "gain_db": -2.5, "q_factor": 1.1 }
    ] } },
    { "id": "comp", "type": "compressor", "params": {
      "threshold_dbfs": -24, "ratio_to_1": 2.8, "attack_ms": 12, "release_ms": 140 } }
  ],
  "master": [
    { "id": "limit", "type": "true_peak_limiter", "params": { "ceiling_dbtp": -1.0 } },
    { "id": "dither", "type": "dither", "params": {} }
  ]
}
```

`"sample_rate": 0` keeps the source rate. The engine refuses anything it would
otherwise render as something else:

- unknown node types and unknown values for any string parameter
  (`filter_type`, `mode`, `detector` and so on) are errors, never silent
  fallbacks;
- a limiter or dither in `nodes[]` is rejected, as is anything after the
  limiter except dither;
- NaN or infinite input samples, and absurd sample rates or seeds, are refused.

Parameter **ranges** are enforced one layer up, by the Python schema
(`backend/schemas/graph.py`), which is what the planner and the API emit
through. Numeric parameters are clamped by each node, so a hand-written recipe
bypasses those bounds; see `docs/open-questions.md`.

[`docs/recipe-format.md`](docs/recipe-format.md) is the full contract, and
[`docs/engine.md`](docs/engine.md) documents every node.

To hear the engine on your own material through a hand-written chain, with a
loudness-matched copy for a fair A/B:

```sh
make demo IN=path/to/vocal.wav     # writes demo/out/
```

## Tests

```sh
make ci       # everything: gates, engine, sanitizers, backend, frontend (~11 min)
make quick    # the same without the AddressSanitizer/UBSan build (~8 min)
make engine   # engine only
make backend  # backend only
```

[`ci/run.sh`](ci/run.sh) is the single definition of the pipeline, and the
GitHub workflow calls it and nothing else, so "passes locally" and "passes in
CI" cannot diverge. It runs these stages:

- **Gates:** secret scan, a licence allowlist checked against a lockfile, and
  structural rules (for example, audio bytes may never pass through a Next.js
  route, and the engine may not include a header that isn't on its allowlist).
- **Engine:**
  - 1950 node contract checks (determinism, block-size invariance, reset,
    bypass and mix nulls, latency, parameter robustness) and 208 I/O checks
    (WAV/RF64, JSON, resampler, recipe validation);
  - a committed golden render, which is the cross-architecture check;
  - the DSP measurement suite gated against its baseline;
  - an ASan/UBSan build.
- **Backend:** 14 suites covering measurement, the planner
  contract, rules, refinement, the verify gate, the evaluation harnesses, the
  API and the listening harness. They synthesise their own audio and never make
  a model call, so no key is needed.
- **Frontend:** lint and typecheck.

Some assertions run on real stems in `audio/stems/`, which are gitignored. Those
sections skip cleanly without them and say so. The gates need a git checkout,
because the secret scan reads `git ls-files`, so run them from a clone rather
than a downloaded archive.

## Repository layout

```
engine/          C++20 header-only render engine, CLI, contract/IO/measurement tests
  include/anv/     core/  dsp/  graph/  io/  nodes/
backend/
  measure/         BS.1770 loudness, true peak, spectrum, dynamics, stereo, hygiene
  plan/            graph planner: prompt assembly, two-stage driver, rules, chat, validators
  schemas/         the effect-graph contract (JSON Schema generated from pydantic)
  refine/          match-EQ and dynamics matching
  verify/          post-render safety gate
  eval/            sensitivity and chain-recovery harnesses
  listen/          blinded listening-test harness and scoring
  knowledge/       preset schema and authoring tool (catalogue intentionally empty)
  voice/           spoken explanation
  api/             local FastAPI server
app/             Next.js studio
ci/              the pipeline and its gates
demo/            hand-written chains you can listen to
docs/            architecture, findings, open questions, protocols
```

## Documentation

Start at [`docs/README.md`](docs/README.md). The documents carrying the most
weight:

| | |
|---|---|
| [architecture.md](docs/architecture.md) | The whole system, and why each stage exists |
| [findings.md](docs/findings.md) | Every bug found and lesson learned, with evidence |
| [open-questions.md](docs/open-questions.md) | Known risks, gaps and unverified claims, and what would resolve each one |
| [engine.md](docs/engine.md) | The node contract, the invariants, and every node |
| [planner.md](docs/planner.md) | The graph contract, validators, prompt cache, and repair loop |
| [evaluation.md](docs/evaluation.md) | The test harnesses, and how to read their numbers |
| [listening-protocol.md](docs/listening-protocol.md) | How to run a blinded listening panel |
| [licensing.md](docs/licensing.md) | What the project may depend on, and why |

## What is not built

These are stated plainly here, and each has an entry in
[`open-questions.md`](docs/open-questions.md):

- **The listening gate.** The harness exists, and no panel has ever used it. No
  claim that the output beats a human engineer's chain has been tested.
- **CLASSIFY:** instrument, drive and DI-versus-miked detection.
- **Presets.** The schema and authoring tool exist, and the catalogue is empty on
  purpose. A preset that makes a factual claim about a recording must cite a
  source, and none has been sourced yet.
- **A platform.** Storage is a temp directory, rendering is synchronous, and
  there is no auth. Ingest is WAV-only, with no sandboxed decoder.
- **Cross-architecture CI.** The workflow is paused. Run `make ci` locally.

## Contributing

Issues and pull requests are welcome. Read [`AGENTS.md`](AGENTS.md) first. It
is short, and its rules are enforced:

- New nodes must enrol in the contract tests through the registry in
  `engine/tests/contract_tests.cpp`.
- Mutation-test new nodes. Every node so far has had a test that passed while
  the property it named was broken.
- Never substitute a plausible constant for a missing measurement. `null` is the
  honest answer.
- Unknown enum values are hard errors, never silent fallbacks.
- To isolate what a node did, compare against a **null render**, never against
  the raw source.

Put concerns in [`docs/open-questions.md`](docs/open-questions.md), not only in a
commit message. Run `make ci` before opening a PR.

## License

Audionova is licensed under the [Apache License 2.0](LICENSE).

Every dependency is held to a permissive allowlist by a CI gate, so that stays
true of the whole tree. [`docs/licensing.md`](docs/licensing.md) records what
was rejected and why.
