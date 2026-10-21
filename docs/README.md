# Audionova documentation

A subscription tool for audio engineers: drop in a track, describe what you
want in plain language, get back a professionally processed file.

The users are people who will A/B our output against their own plugin chains and
will hear aliasing, phase smear and wrong parameters immediately. **Quality is
the entire product**, and one harsh render loses a subscriber permanently.
Everything in these documents optimises for that.

## Where to start

| Document | What it covers |
|---|---|
| [architecture.md](architecture.md) | The whole system and why each piece exists |
| [engine.md](engine.md) | The C++ render core: node contract, invariants, every node |
| [recipe-format.md](recipe-format.md) | The JSON that describes a render |
| [measurement.md](measurement.md) | Every field of the measurement report and what it means |
| [planner.md](planner.md) | The graph contract, validators, prompt cache, ladder, glossary |
| [evaluation.md](evaluation.md) | The four test harnesses and how to read their numbers |
| [listening-protocol.md](listening-protocol.md) | How to run a listening panel, start to finish |
| [findings.md](findings.md) | Every bug found and lesson learned, with the evidence |
| [open-questions.md](open-questions.md) | Everything currently flagged: risks, gaps, unverified claims |
| [licensing.md](licensing.md) | What we may and may not depend on, and why |
| [progress/](progress/) | Dated notes on milestones: the first model plan, the local app, the silent-stage bug |

`findings.md` is the one to read if you are wondering why something is written
the way it is. Most of the odd-looking decisions in this codebase are there
because the obvious version was measured and found wrong.

`open-questions.md` is the one to read before planning work. It is a living
document — anything raised as a concern gets an entry there rather than living
only in a commit message.

## Current state

Built and verified:

- **Engine** — 24 DSP nodes, float64, 96 kHz internal, exact integer latency.
  1950 contract checks + 208 I/O checks.
- **Render CLI** — `anv_render`, WAV/RF64 in and out, hand-written resampler.
- **Measurement layer** — BS.1770 loudness validated against EBU conformance
  signals, gated spectrum, dynamics, stereo, hygiene. 218 checks (66 loudness, 152 measurement).
- **Cross-validation** — the engine and the measurement layer checked against
  each other on analytically-known quantities. 52 checks.
- **Evaluation harnesses** — sensitivity (can the report see what the engine
  does: 10/10) and chain recovery (a scored floor for any future planner:
  F1 0.580). 716 checks (101 sensitivity, 615 recovery).
- **Effect graph schema and validators** — the planner's output contract, with the
  JSON Schema generated from the enforcing model so the two cannot drift.
  302 checks.
- **Planner scaffolding** — prompt assembly with an 18.4 k cached prefix, the
  two-stage PLAN/EMIT driver, a bounded repair loop with SLI counters, the
  deterministic quantifier ladder, and a sixty-term descriptor glossary whose
  conditions are checked against a real measurement report. 296 checks.
- **Match-EQ** — fits a small editable biquad cascade to a spectral difference.
  Recovers a known filter applied by the engine to within 0.2 dB and the right
  band count. 116 checks.
- **Rule-based planner** — a measurement report in, a validated chain out, with
  every move naming the number that justified it. The floor the model planner
  has to beat. 72 checks.
- **Dynamics matching** — compressor settings found by rendering. Converges
  within 0.3 dB of a gain-reduction target. 66 checks.
- **Post-render verify gate** — re-measures the render and refuses to ship
  corruption, ceiling overshoot, mono incompatibility, an introduced resonance,
  a level collapse or destroyed bandwidth. 75 checks.
- **Blinded listening harness** — the tooling for the Phase 0 panel and for
  calibrating any proxy metric against human judgement. Seeded blinded
  sessions whose mapping is not recoverable from the payload, loudness
  matching verified on the written files at a 0.1 LU bar, the plan's 3 kHz
  screen rendered by the real engine, two rating scales with listener-clustered
  intervals, and Spearman rho that refuses to print without its interval.
  269 checks, and 94 of 95 mutations caught. See `evaluation.md`.

- **Preset authoring tool** — the mitigation for the plan's risk #7, that preset
  authoring is the schedule bottleneck and is not an engineering task.
  `backend/knowledge/author.py`: `new` scaffolds a dossier with every prompt
  stating the rule it enforces, `check` translates the schema's refusals into
  something an author can act on, `audition` renders the priors against a
  measured source and writes a loudness-matched A/B against a **null render** so
  the author can hear them, and `diff` reports what measurably separates two
  dossiers. 146 checks, and 39 of 39 mutations caught after the first pass
  caught 31. `cd backend && make author`.

- **The model planner, live.** PLAN and EMIT run against the Anthropic API,
  with stage 1's reasoning streamed to the studio as it is written. The suites
  still test it only against a scripted client, so CI needs no key;
  `python -m backend.plan.canary` is the live check.
- **The local application.** A FastAPI server (`backend/api/server.py`) and a
  Next.js studio (`app/`): upload, measure, plan, render, a loudness-matched
  A/B, before-and-after measurements taken against a null render of the same
  excerpt, refinement by button or in
  your own words, reference matching, an editable graph inspector, session
  history, recipe import/export, and an optional spoken explanation. The API
  suite is 288 checks; the studio is linted and typechecked in CI.

Not built:

- **CLASSIFY**.
- **The Phase 0 listening gate.** Three engineers, ten guitar DIs, prefer-or-tie
  in at least 6 of 10. This is the plan's explicit stop-or-go point and has not
  been attempted. **The harness for running it now exists** (`backend/listen/`;
  its suite is `cd backend && make listen`); what is missing is three engineers,
  their own chains, and ten guitar DIs. The whole sequence has been driven end
  to end once by a test client — built from real stems, served, answered by four
  synthetic listeners, scored, correlated — so the plumbing is validated.
  **No session has ever been taken by a person.**
  [listening-protocol.md](listening-protocol.md) is the recipe for running one.
- **Presets.** Zero authored, and the catalogue directory does not exist. The
  plan budgets 4–6 person-weeks for twelve and notes this is not an engineering
  task. The tooling now exists (above); what is missing is the source books and
  a person who can verify a claim about a session. **Nothing in this repository
  contains a factual claim about a recording**, including the authoring tool's
  test fixture, which is called `example_not_a_real_preset` and cites a source
  named as fictional.
- **The platform.** The API is local only: no queue, no object storage, no auth,
  and ingest is WAV-only with no sandboxed decoder.
- **CI on a second machine.** The pipeline and its matrix exist; nothing has run
  anywhere but here, so the cross-architecture claim is still untested.

## Running things

```sh
make ci        # everything CI runs: gates, engine, sanitizers, backend, frontend (~11 min)
make quick     # the same without the sanitizer build (~8 min)

make engine    # engine only
make backend   # backend only
```

`ci/run.sh` is the single definition of the pipeline — the GitHub workflow calls
it and nothing else, so "passes locally" and "passes in CI" cannot diverge.

Analyses that need real audio, which is gitignored (see `licensing.md`):

```sh
cd backend && make sweep   STEMS="audio/stems/*.wav"
cd backend && make recover STEMS="audio/stems/*.wav"
```

The audio code on the Python side needs only `numpy` and `scipy`; the schema,
model client and API add `pydantic`, `anthropic`, `fastapi`, `uvicorn`, `httpx`
and `python-multipart`. See `backend/requirements.txt`, which documents the
rejected dependencies as well as the accepted ones.
