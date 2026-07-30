# Architecture

## The shape of the problem

A user uploads a track and types a sentence. Something has to turn that sentence
into a chain of DSP with specific parameter values, apply it, and hand back
audio an engineer will accept.

The naive design is one large model call that listens to the audio and emits
parameters. That design does not work, and the research is unusually clear about
why. Three independent results:

- **LLM2Fx** (arXiv:2505.20770) measured CLAP-guided optimisation at 0.55 MMD on
  EQ — *worse than random parameters* at 0.53 — while a zero-shot LLM given
  measured DSP features scored 0.22.
- **FxSearcher** (ICASSP 2026) found Text2FX scored *higher* on CLAP while losing
  on human MOS, 2.28 against 3.48.
- Probing work shows CLAP under-encodes EQ detail specifically, and T-CLAP shows
  it is temporally blind: it cannot distinguish a 40 ms slap from a 400 ms echo,
  or "slightly more" from "much more".

The conclusion that shapes everything below: **the model is never asked to
imagine what the audio sounds like. It is handed numbers.** Audio embeddings are
demoted to an optional re-ranker behind a deterministic safety gate, and deleted
if they cannot beat single-shot on human listening.

## The pipeline

```
MEASURE     deterministic DSP, no ML           → grounds the model in numbers
CLASSIFY    small models                       → instrument, drive, DI-vs-miked
PLAN        LLM, unconstrained reasoning       → where world knowledge earns its keep
EMIT        LLM, constrained typed graph       → safety and editability
REFINE      deterministic closed-loop fitting  → fixes the model's parameter weakness
RENDER      C++ core                           → provable quality
VERIFY      re-measure, hard gate              → nothing ships that fails
```

Five of the seven stages are deterministic and testable. Only PLAN and EMIT are
model calls. That ratio is deliberate: every stage that can be arithmetic is
arithmetic, because arithmetic can be tested and a model cannot.

**Built today:** MEASURE and RENDER, plus the harnesses that check them against
each other. CLASSIFY, PLAN, EMIT and REFINE do not exist.

## Why two model calls and not one

Constrained decoding from token one degrades reasoning (arXiv:2408.02442), and
LLM2Fx-Tools' ablation shows removing chain-of-thought costs 13 points of module
accuracy. So:

- **PLAN** is unconstrained. Four explicit sub-tasks: input analysis, module
  selection, **order determination**, parameter planning. Output is prose and is
  never parsed programmatically.
- **EMIT** is a separate call whose system prompt says, in effect, *the
  engineering decision has already been made; serialise it; do not reconsider
  anything.*

Chain order is a first-class emitted field with a legality checker, not an
implementation detail. LLM2Fx-Tools showed two systems with near-identical
parameter accuracy but ordering correlation 0.00 against 0.56 — and the ordered
one won perceptually by a wide margin.

## The two halves, and the boundary between them

```
       C++  (engine/)                    Python  (backend/)
  ┌──────────────────────┐          ┌──────────────────────────┐
  │ 18 DSP nodes         │          │ measurement report       │
  │ Chain + PDC          │◀── JSON ─│ planner (not built)      │
  │ resampler            │  recipe  │ validators (not built)   │
  │ WAV / RF64           │──▶ WAV ──│ evaluation harnesses     │
  └──────────────────────┘          └──────────────────────────┘
```

The boundary is a **subprocess**, not a binding. `backend/engine/render.py`
shells out to `engine/build/anv_render`. pybind11 is planned and will happen for
throughput, but the subprocess has one property worth keeping as long as
possible: Python exercises *exactly* the binary that ships. An in-process binding
is a second build configuration, and a second build configuration is a second
set of numerical results.

The blunter reason: the engine is where audio quality lives and Python is where
the mistakes will be. A hard process boundary means a bug on the Python side
cannot corrupt a render, only fail to produce one.

## Rate handling, and the master section

Every render is:

```
read → resample in → nodes[] at 96 kHz → resample out → master[] at delivery rate → write
```

Exactly two conversions. Every additional one is another transition band and
another set of passband ripples, and unlike most DSP errors they compound
silently — a chain that converts five times measures fine at every individual
stage.

**96 kHz internal** because sample-rate-dependent processors (notably the
Airwindows family the plan draws character from) behave differently at 44.1 and
96 kHz *by design*, so one canonical rate is the only way to get reproducible
renders. It also halves the oversampling burden on nonlinear nodes.

The recipe has **two node lists**, and the split is load-bearing rather than
stylistic:

- `nodes[]` runs at the internal rate.
- `master[]` runs **after** the output sample-rate conversion.

Two operations are only correct at the delivery rate:

- **True peak is defined at the delivery rate.** A limiter meeting −1.0 dBTP at
  96 kHz can overshoot after conversion to 44.1 kHz, because the resampler
  reconstructs peaks that were never sample-aligned. Limiting before the final
  conversion does not limit the file that ships. This is measured, not argued —
  see `test_engine_bridge.py` section 6.
- **Dither is a function of the delivery word length**, and anything that
  processes after it re-correlates the quantisation error. Resampling after
  dither is exactly such a process.

`validateRecipe` rejects a limiter or dither in `nodes[]`, dither anywhere but
last, and any node after the limiter.

## What the audible-band guarantee actually is

The 48 → 96 → 48 kHz round trip nulls at **−194 dB at 1 kHz, −176 dB at 10 kHz
and −168 dB at 18 kHz**, then falls off deliberately: a 22 kHz tone returns
12 dB down, because the conversion filter's passband edge is at 20 kHz.

This is a stated trade rather than an oversight. Protecting 20–24 kHz would cost
two to fourteen times the conversion compute for content nobody can hear.
Both halves are asserted in the test suite, so moving the passband edge shows up
as an intentional change rather than downstream as a mysterious fraction of a dB.

**Consequence for any measurement work:** to isolate what a node did, compare
against a *null render*, never against the raw source. Measured against the raw
source a −0.5 dB gain reads as −0.62, because the conversion's own 0.12 dB of
K-weighted loudness change is folded in. Against an empty render it reads
−0.5000.

## Safety posture

Two boundaries carry the risk.

**Untrusted bytes.** Everything a user uploads goes through a sandboxed FFmpeg
ingest stage before the engine sees it: separate container, `--network none`,
seccomp, `cap-drop=ALL`, non-root, read-only rootfs, rlimits, wallclock kill.
FFmpeg's CVE history is long and the input is 100% attacker-controlled. The
engine itself links nothing that parses untrusted input — which is why WAV
reading is hand-written rather than libsndfile.

File metadata is attacker-controlled text on a path that ends in a prompt.
**Nothing from ID3 or Vorbis comments reaches the measurement report**, which
describes samples and only samples.

**Model output.** A language model configures DSP that a paying engineer will
listen to. Every structural rule is enforced in code rather than requested in a
prompt: the node type set is closed, filter shapes are a closed set, the master
ordering is structural, and a post-render gate re-measures and refuses to ship
anything that fails. See `engine.md` for what the engine rejects and
`findings.md` for the silent-fallback bugs that motivated each rule.

## Reading order for the code

1. `engine/include/anv/core/Node.h` — the contract every effect implements
2. `engine/include/anv/graph/Chain.h` — serial chain, latency compensation, dry/wet
3. `engine/include/anv/io/Recipe.h` — the JSON contract and its validators
4. `backend/measure/report.py` — what the planner will be handed
5. `backend/eval/sensitivity.py` — whether that report can see anything
