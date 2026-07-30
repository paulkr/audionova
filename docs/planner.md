# The planner

`backend/schemas/graph.py` and `backend/plan/validate.py`.

**Status: the contract and its validators exist; no model call does.** This
document describes what is built and what the built parts are for. The PLAN and
EMIT stages themselves are not implemented.

## The effect graph

The graph is the planner's output contract, the user's editable representation of
what will happen, and the input to the engine's recipe. It deliberately carries
**no sample rate and no output format** — those are delivery decisions, and
keeping them out means the same graph renders as a 25-second excerpt for audition
and as a full-length 24-bit master without being regenerated.

```python
from backend.schemas.graph import EffectGraph
from backend.plan.validate import validate

graph = EffectGraph.model_validate(model_output)   # structural
result = validate(graph, measurement_report)       # semantic
if result.ok:
    recipe = result.graph.to_recipe(seed=7, output_sample_rate=44100.0)
```

## Why bounds exist twice

    CLAUDE STRUCTURED OUTPUTS DO NOT ENFORCE minimum/maximum.

`enum`, `const`, `anyOf` and `additionalProperties: false` are honoured. Numeric
bounds are not. A schema that reads like it constrains `gain_db` to ±18 does
nothing of the kind on the wire — the model may emit 400 and the API passes it
through.

So every bound has to appear **in the description text**, where the model reads
it, and be enforced **separately in Python** afterwards. That is the same number
in two places, which is exactly where drift happens.

The resolution is one declaration with three consumers:

1. a Pydantic model that enforces the range
2. a JSON Schema generated from it, with numeric keywords **stripped** — leaving
   them in would advertise a guarantee that does not exist
3. the range text injected into each field's `description`

A test asserts the prose and the enforced bound agree, and that no unenforced
numeric keyword survives into the emitted schema.

## The schema is narrower than the engine

The engine's compressor accepts twelve parameters; the schema exposes ten.
`os_profile`, `freeze`, `makeup_mode` and similar are omitted — not because they
are broken, but because **every parameter offered to a planner is a parameter it
can get wrong**, and the omitted ones have defaults an engineer would rarely
move.

Parameter names and ranges mirror the engine exactly, and a test reads the engine
headers to prove it: every node type exists in `makeNode`, every parameter name is
one the engine actually reads, the filter shapes match `isKnownFilterShape`
exactly, and the master-only sets agree in both directions.

A drifted schema produces graphs that validate and then fail to render — or worse,
render with a silently defaulted parameter, which is the failure mode this project
has spent the most effort eliminating.

## The validators

Structural rules live in the schema; these catch graphs that are well-formed and
**wrong**. That is the harder half: a malformed graph fails to render, while a
wrong one renders beautifully and sounds bad.

Two outcomes, and the distinction is about cost:

- **repairable** — exactly one sensible fix exists, so Python applies it and logs
  what it did. Burning a model retry to arrive at an answer we already have is a
  wasted round-trip.
- **fatal** — the graph expresses an intent that cannot be salvaged without
  guessing. Triggers the bounded repair loop: at most two retries, then fall back
  to the previous accepted graph.

Repairs are logged rather than applied silently because **a rising repair rate is
a prompt-quality signal**. Target SLIs: repair under 15%, retry under 5%,
fallback under 0.5%.

### 1. Ranges

Clamps every numeric parameter, reading the bounds **from the Pydantic models**
rather than from a table kept beside them.

That indirection is not decoration. The first version used a hand-maintained
name-to-range map and it had already diverged: a band's `gain_db` is bounded at
±18 while a gain *node*'s is ±24, and a map keyed on the name cannot tell them
apart. It silently applied the wider bound to bands, so a 22 dB band gain passed
a clamp the schema itself would have rejected. Names are not unique across
contexts.

### 2. Filter stability

Rejects poles closer than 0.9995 to the unit circle. A high Q at a very low
frequency is the dangerous combination — Q 12 at 20 Hz rings for seconds.

This is not protecting the engine, whose TPT/ZDF topology is unconditionally
stable. It is protecting the **result**: a filter that does not blow up can still
ring audibly.

### 3. Total boost budget — 12 dB

The single most important check here. Sums every **positive** EQ gain across the
graph and caps the total.

Exceeding it is the commonest cause of a chain that "sounds like an EQ" rather
than like a record: each move is defensible alone, and the sum is bright, thin
and obviously processed. Most good mixes stay under 8 dB.

**Only boosts count.** Three 4 dB cuts leave a result that still sounds like the
source; three 4 dB boosts do not. The asymmetry is deliberate, not an oversight
that this is not `sum(abs(gain))`.

**Scaled proportionally, not clipped.** The shape the planner chose survives and
only its depth changes. Clipping the largest band would silently alter the tonal
balance it had reasoned about.

### 4. Gain staging

Projects the output level from the measured 95th-percentile RMS plus the static
gains, largest EQ boost and makeup. Crude by design — it cannot model compression
or saturation, so it is a bound rather than a prediction — but it catches a chain
arriving at the limiter 15 dB hot, where the limiter's only option is to remove so
much that no dynamics survive.

**Skipped entirely without a measurement report**, rather than guessed at. An
invented input level produces invented warnings.

### 5. Chain-order legality

Each rule is a case where the reverse order sounds finished and is wrong:

| Must precede | Reason |
|---|---|
| `dc_block` → `drive` | A DC offset biases a saturator asymmetrically |
| `drive` → `convolution_reverb` | A cabinet should filter the saturation, not be saturated by it |
| `mono_to_stereo` → `stereo_width` | Width has no side signal until the source is stereo |
| `mid_side` → `stereo_width` | Encoding must precede operating on the side signal |
| `delay` → `plate_reverb` | Reverb should sit on the repeats, not be smeared by them |

The master section has exactly one right order, so it is **sorted** rather than
rejected — there is no intent to guess at.

### 6. No-ops

A move that cannot be heard is worse than no move: it consumes a band, it appears
in the explanation shown to the user, and the user hears nothing and concludes the
tool does not work.

- A band above a **codec wall** is dropped — but only when the detector's
  confidence is at least 0.4. A low-confidence limit must never drive a decision.
- **Width on material with no side signal** is flagged.
- A **bypassed node** is flagged: legal, but a planner that emitted one probably
  meant to omit it.

### 7. Source-condition constraints

Things the measurement report says the graph may not do:

- **Drive on a clipped source** is fatal. Saturation amplifies existing
  distortion rather than adding character.
- **An unaddressed out-of-phase source** is fatal. The mono sum currently
  cancels; nothing else matters until that is fixed.
- **A high boost over an audible noise floor** is flagged — it lifts hiss into the
  foreground.
- **A dither bit depth that is not a delivery word length** is corrected. Getting
  this wrong is silent in both directions: a dithered float export just has added
  noise, and an undithered 16-bit export has correlated quantisation distortion on
  every fade.

## Testing

`backend/tests/test_schema.py`, 179 checks, run by `make schema`.

The suite ends by **rendering**. A schema is only correct if something that passes
it survives contact with the engine, so the last section builds a three-node graph
with a master section, validates it, converts it to a recipe and renders real
samples — then renders the same graph at a different delivery rate, since the
graph deliberately carries no rate or format.

## What is not built

The two model calls, and everything around them:

- **PLAN** — unconstrained reasoning over the measurement report, with four
  explicit sub-tasks: input analysis, module selection, order determination,
  parameter planning. Output is prose and is never parsed. Unconstrained because
  constrained decoding from token one degrades reasoning, and removing
  chain-of-thought costs 13 points of module accuracy.
- **EMIT** — a separate constrained call that serialises the decision without
  reconsidering it.
- The **prompt cache layout** (~18.5 k cached prefix, ~7 k volatile) and the CI
  canary asserting `cache_read_input_tokens > 15000`.
- The **repair loop** driver, and the SLI counters.
- **REFINE** — match-EQ and dynamics matching, both deterministic.
- The **preset knowledge base** and the alias routing table.

The scoreboard for all of it already exists: `docs/evaluation.md` describes the
rule-based recovery floor of **F1 0.580** that any planner has to beat.
