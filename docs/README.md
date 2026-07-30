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
| [evaluation.md](evaluation.md) | The four test harnesses and how to read their numbers |
| [findings.md](findings.md) | Every bug found and lesson learned, with the evidence |
| [licensing.md](licensing.md) | What we may and may not depend on, and why |

`findings.md` is the one to read if you are wondering why something is written
the way it is. Most of the odd-looking decisions in this codebase are there
because the obvious version was measured and found wrong.

## Current state

Built and verified:

- **Engine** — 18 DSP nodes, float64, 96 kHz internal, exact integer latency.
  1321 contract checks + 139 I/O checks.
- **Render CLI** — `anv_render`, WAV/RF64 in and out, hand-written resampler.
- **Measurement layer** — BS.1770 loudness validated against EBU conformance
  signals, gated spectrum, dynamics, stereo, hygiene. 148 checks.
- **Cross-validation** — the engine and the measurement layer checked against
  each other on analytically-known quantities. 48 checks.
- **Evaluation harnesses** — sensitivity (can the report see what the engine
  does: 10/10) and chain recovery (a scored floor for any future planner:
  F1 0.580). 152 checks.

Not built:

- **The intelligence layer.** Measure exists; classify, plan, emit and refine do
  not. No LLM call happens anywhere in this codebase yet.
- **The Phase 0 listening gate.** Three engineers, ten guitar DIs, prefer-or-tie
  in at least 6 of 10. This is the plan's explicit stop-or-go point and has not
  been attempted.
- **Presets.** Zero authored. The plan budgets 4–6 person-weeks for twelve and
  notes this is not an engineering task.
- **The platform.** No API, no queue, no storage, no auth.

## Running things

```sh
cd engine  && make check          # contract tests, I/O tests, sanitizers
cd backend && make test           # loudness, measurement, bridge, sensitivity, recovery

# Analyses that need real audio (not in git — see licensing.md)
cd backend && make sweep   STEMS="../audio/stems/*.wav"
cd backend && make recover STEMS="../audio/stems/*.wav"
```

The Python side needs `numpy` and `scipy` and nothing else. See
`backend/requirements.txt`, which documents the rejected dependencies as well as
the accepted ones.
