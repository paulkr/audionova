# The planner

`backend/schemas/graph.py` and `backend/plan/validate.py`.

**Status: everything except the model calls themselves.** The contract, the
validators, the prompt assembly, the two-stage driver and its repair loop all
exist and are tested offline against a scripted client. Nothing in this
repository has ever made a live model call.

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

## The prompt, and the cache boundary

The economics are inverted from the obvious assumption. DSP compute is about
$0.0007 for a three-minute render; a frontier planning turn is $0.15 to $0.50 —
**the model costs two to seven hundred times the audio processing.** So the thing
worth engineering is the prompt.

The prefix measures **18,409 estimated tokens** and never varies between
requests: role framing, the annotated schema, the per-node parameter schemas, the
glossary, and the DSP source of six effects. The volatile half — measurement
report, classification, preset dossiers, the user's sentence — sits after the
cache breakpoint.

**One timestamp in the prefix destroys the entire saving.** A request id, a user
id, a formatted date — anything per-request in the system block moves the cache
boundary before it and turns every subsequent token into a miss. This fails
silently: the request succeeds, the answer is fine, and the bill is five times
larger.

So `system_blocks()` takes **no arguments**, and a test asserts both that two
independently-built prefixes are byte-identical and that the function's signature
is empty. Further tests assert that no per-request value appears in the prefix
and that the values the model needs *do* appear in the volatile half.

**The DSP source is in the prompt because it is the highest-leverage item in it.**
Putting the implementations in front of the model is what moved LLM2Fx from 0.53
to 0.22 on EQ matching — the model reasons about what a parameter *does* rather
than what its name suggests. It is also most of the prefix's size, which is
exactly why the prefix has to cache.

Six node implementations are included, not eighteen. The utility nodes are
self-explanatory from their names and their source would cost prefix tokens for
nothing.

## Two calls, not one

- **PLAN** is unconstrained, with four explicit sub-tasks: input analysis, module
  selection, order determination, parameter planning. Output is prose and is
  never parsed programmatically — only handed to stage 2. Unconstrained because
  constrained decoding from the first token measurably degrades reasoning, and
  removing chain-of-thought costs 13 points of module accuracy.
- **EMIT** is a separate call whose system prompt reframes the model as a
  transcriber. Separate rather than a continuation, because a model cannot be a
  transcriber in the same turn that still holds the reasoning task — and the
  schema constraint then applies to a turn with no reasoning left to degrade.

Both share the same cached prefix.

## The repair loop

Two retries, then fall back. The distinction that matters for cost:

- **repairable** violations are fixed in Python and logged. Burning a retry to
  arrive at an answer we already have is a wasted round-trip.
- **fatal** violations go back to the model, with the validator's findings quoted
  **verbatim** rather than paraphrased. A validator message already names the
  measurement and the rule; restating it is an opportunity to soften it. The
  retry prompt asks for a minimal change, since every other decision was
  accepted.

**The fallback is always a graph that already passed** — the previous accepted
graph for this source, never a default chain. A default is a guess about audio
nobody measured.

Schema failures are as retryable as semantic ones, for the same reason: the model
can be told what was wrong.

### Counters

`Counters` accumulates the SLIs: repair rate, retry rate, fallback rate, and an
unparseable rate for replies that were not graphs at all.

Rates are **not reported below twenty turns**, because a rate over five samples
is not a signal. Breaches are reported rather than enforced: refusing to serve
because a rolling rate is high would turn a quality signal into an outage.

Repair rate is the one to watch. It rising means the prompt has drifted from what
the validators expect, and it rises long before quality visibly drops.

## Multi-turn refinement

`backend/plan/ladder.py`. **A refinement is a patch, not a regeneration.**

Asking the planner for a whole new graph on "add a bit more air" discards every
decision the user implicitly accepted by not complaining about it, and churns
unrelated parameters. To the user that reads as a bug: they asked for one change
and six things moved. A test asserts that a one-parameter refinement moves
**exactly one** value.

### The arithmetic never touches a model

When a user says "slightly more delay", the size of "slightly" must be the same
number every time — on Tuesday, on a different stem, at a different temperature.
A model asked to apply it produces 1.2 dB one turn and 4 dB the next, and the
user concludes the tool is unpredictable rather than that they were unclear.

So the model's only job is to name a target and a magnitude word from a closed
set. Python decides what the word means.

| word | steps |
|---|---|
| slightly, a bit, somewhat | 1 |
| noticeably, moderately | 2 |
| much, a lot, significantly | 3 |
| dramatically | 4 |

`slightly` and `a bit` are deliberately identical: users treat them as synonyms
and distinguishing them would invent precision the language does not carry. An
unrecognised intensifier gets one step rather than zero — the user still asked for
a change, and doing nothing looks broken.

### Step sizes are perceptual, not arithmetic

A step is "the smallest change a competent listener reliably notices", which is
why they are not all the same shape:

| parameter | step | why |
|---|---|---|
| EQ gain | **+1.5 dB** | dB is already a perceptual scale, so this adds |
| frequency | **×1.26** | one third of an octave; pitch is logarithmic, so this multiplies |
| Q | ×1.4 | 1 → 2 is a large change, 8 → 9 is not |
| threshold | **−2.5 dB** | has to move further than an EQ band before the gain reduction changes audibly |
| reverb decay | ×1.25 | |
| mix | ×1.4 | doubling a 0.1 mix is obvious; adding 0.1 to a 0.9 mix is not |

A parameter with no declared step **raises** rather than being guessed at. A step
size is a judgement about audibility, and inventing one at call time would make
the same word mean different things on different parameters.

### Ambiguity resolved in code

"More delay" almost always means a *louder* delay, not a longer one, and a planner
allowed to choose would sometimes move the repeats off the beat.

| phrase | parameter |
|---|---|
| more delay | `mix` |
| more repeats | `feedback_percent` |
| longer delay | `time_ms` |
| darker delay | `hf_damp_hz`, **decreased** |
| more compression | `ratio_to_1` |
| harder compression | `threshold_dbfs`, decreased |

Not every "more" increases a number: a darker reverb is a *lower* damping corner.

An intent with no matching node returns **no patches** rather than adding one.
Inserting a reverb into a chain the user never asked to have one in is a much
larger change than they requested.

### Eight operations, closed set

`adjust_param`, `set_param`, `add_node`, `remove_node`, `bypass_node`,
`enable_node`, `add_band`, `remove_band`.

Closed because an open-ended "edit" would put the model back in charge of
arithmetic. A patch naming a node that does not exist is **rejected, not
ignored** — a silently dropped edit leaves the user with no change and no
explanation.

Values are rounded to something a human would type. An engineer reading
"−3.2 dB at 340 Hz" trusts it; nine decimal places reads as machine output and
invites the question of whether it was chosen at all.

`describe()` renders a patch set to prose **before** rendering audio, because the
plan's definition of done for refinement is that the user reads the diff first.

## The descriptor glossary

`backend/plan/glossary.py`, forty terms across tonal, dynamic, spatial and
character axes, with aliases.

**The glossary is evidence, not instruction.** Every entry carries a
`check_first` condition that must hold of the measured audio before the
conventional move applies, and that field is the whole point of the file. The
alternative — "muddy means cut 250 Hz" — is a stock move applied to a source that
may not have a problem at 250 Hz, and a chain of stock moves sounds like one.

A mud complaint on a source whose 200–400 Hz region is already 6 dB below its own
average is not a mud problem. The planner is expected to say so rather than cut a
band that is already quiet.

Frequency ranges are **wide on purpose**. "Boxy" is roughly 250–500 Hz and the
exact centre depends on the instrument; a boxy kick and a boxy acoustic guitar are
not the same frequency. Narrow ranges would encode precision nobody has.

Several terms deliberately have **no default move** — `vintage`, `modern`,
`lofi`, `honest`, `woody`. They are recorded so the planner recognises them as
unactionable or preset-routed rather than inventing a frequency. A bandwidth
limit, a tilt and some saturation is a caricature of "vintage", not an answer.

A duplicate-alias guard raises at import. It immediately caught `distant` claiming
to be both a spatial term and an alias of the tonal `buried` — a real ambiguity,
since a source can read as distant either from too much reverb or from no
presence at all, and the two want opposite answers. Both entries now say which
measurement decides.

## Refine: match-EQ

`backend/refine/match_eq.py`. Deterministic, and the stage where the model's
parameter weakness gets erased without inheriting a regression model's
perceptual behaviour — the model chose the modules and the order, this fixes only
the numbers.

**The output is a handful of biquads, not an 8000-tap FIR.** A paying engineer has
to read the result, trust it, disagree with it, and retype it. "−3.2 dB at
340 Hz, Q 1.1" is all four; an impulse response is none of them.

Four steps, three of which are the ones implementations skip:

1. **Level-match first.** Without it a match EQ is a broadband gain change wearing
   a filter costume — the fit spends its bands correcting an offset and the result
   is louder rather than closer. The offset is the median, not the mean, so one
   narrow resonance cannot drag the whole curve. It is reported separately and
   deliberately not folded into the bands: a level change is a gain node's job.
2. **Resample onto a uniform log grid before smoothing.** Fractional-octave
   smoothing on linearly-spaced bins covers more bins above the centre than below,
   biasing the curve upward and increasingly so toward the top (Tylka, Boren and
   Choueiri, JAES 2017). On a linear grid a 1/3-octave smooth of a flat spectrum
   is not flat.
3. **Clamp with a soft knee.** Linear below 60% of the limit, bending above.
   Unbounded match curves are the commonest reason match EQ sounds wrong — the
   difference between two recordings includes performances, rooms and noise
   floors, and following all of it into a 20 dB correction produces something
   that matches on paper and is unusable.
4. **Fit, round, then re-verify.** Rounding changes the response, so the reported
   error is measured after it rather than before.

### The smallest cascade that fits

Every size from one band upward is fitted, and the smallest within 0.15 dB of the
best is chosen. Fitting straight to the budget uses all of it whether the material
needs it or not: recovering a two-band EQ with five bands available returned four,
splitting one bell across two frequencies. Numerically better, worse to hand an
engineer — they cannot tell which bands matter, and the halves of a split band
cannot be adjusted independently without breaking the shape.

Q is capped at 4. Unconstrained, least squares happily returns Q 8.7 notches that
track every wrinkle of a difference curve which is mostly not EQ.

### Measured recovery

A known EQ applied by the real engine, recovered from the two spectra alone:

| applied | recovered | error |
|---|---|---|
| bell 900 Hz +5.0 Q1.30 | bell **900 Hz** +4.9 Q1.38 | 1.45 → 0.11 dB RMS |
| bell 420 Hz −5.0, shelf 6 kHz +4.5 | bell **420 Hz** −4.8, shelf 5860 Hz +4.7 | 2.25 → 0.09 dB RMS |
| three moves | all three, right count | 1.71 → 0.17 dB RMS |

The analytic filter response the optimiser fits against is checked against what
the engine actually renders, within 0.6 dB across five band types. The optimiser
evaluates it thousands of times so it must be arithmetic — but a response that had
drifted from the engine would produce a beautiful fit to the wrong target, and
nothing inside the fit could detect it.

## Refine: dynamics matching

`backend/refine/match_dynamics.py`. A closed loop against the actual renderer,
because a compressor's effect on any statistic depends on the material's own
distribution, on the timing interacting with its envelope, and on the knee. There
is no closed-form inversion.

Order: **threshold, timing, ratio, makeup.** Threshold and timing are derived
directly and cost nothing. Only the ratio needs the loop. Makeup is computed once
at the end — inside the loop it would change the level the next iteration
measures, and the loop would chase its own tail.

### The control variable is gain reduction, not loudness range

A correction to the plan, forced by measurement. Loudness range is not monotonic
in ratio on real material — it falls then rises on vocals, and rises throughout on
an already-compressed guitar — so bisection on it cannot converge. Gain reduction
at the loud percentile is monotonic on every stem, because it follows the static
curve. See `findings.md` for the numbers.

### Timing comes from the source, not from a reference

The plan's method — the 3 dB corner between the source's and reference's
modulation spectra — does not exist to be found. Compression *adds* modulation
energy at those rates rather than removing it. Time constants are derived from the
source's own modulation rate instead: a compressor should move at the speed of
the material.

### An unreachable target is reported, not silently maxed out

The achievable reduction is measured first, in one render. "This material is
already too flat for what you asked" is the useful answer; quietly returning 20:1
would hide it, and the user would hear no change and conclude the tool does not
work.

### Measured

Converges within 0.3 dB of a 2, 4 or 6 dB target inside the render budget, on
synthetic material and on the vocals and bass stems. Matching a compressed
reference recovers its reduction within 0.4 dB. A reference that is merely
*louder* asks for no compression at all — a level difference is not a dynamics
difference.

## Verify: the post-render safety gate

`backend/verify/gate.py`. The last thing that runs, and **the only check that
sees the actual audio.**

Every other check reasons about a graph and runs before anything is rendered. A
graph can be structurally perfect, semantically justified, within budget and
correctly ordered — and still produce a render with a resonance nobody asked for,
because two gentle boosts landed on the same frequency and a compressor's makeup
pushed the sum into a limiter that was already working. No amount of pre-render
reasoning catches that.

**Any failure means the render does not ship.** Not a warning — the render is
discarded and the previous accepted graph is used instead. This is what makes it
safe to let a model near a paying engineer's audio: whatever it proposes,
something deterministic listens to the result before the user does.

| check | fails when |
|---|---|
| `non_finite` | any NaN or Inf sample |
| `silent` | the render peaks below −80 dBFS |
| `true_peak` | above the ceiling the graph claimed, or above 0 dBTP with no limiter |
| `loudness_target` | more than 0.5 LU off a stated target |
| `level_collapse` | more than 12 dB quieter than the chain accounts for |
| `out_of_phase` | correlation below −0.2 |
| `mono_loss` | more than 1 dB beyond the arithmetic 3 |
| `resonance` | a 9 dB peak above both neighbours that the source did not have |
| `bandwidth_collapse` | the HF rolloff steepened by more than 18 dB/octave |

### The thresholds are not symmetric with the pre-render ones

A pre-render validator asks "is this a sensible thing to do?" and can afford to be
opinionated. This asks "did something go wrong?" and has to be **nearly impossible
to trip by accident**, because a false failure throws away work the user was happy
with and replaces it with something older. A miss is a bug they can report; a
false failure is the tool refusing to do its job.

So every threshold sits where a competent engineer would say something is
*broken*, not where they would have made a different choice. A deliberate 6 dB
bell passes the resonance check; an 18 dB Q-10 boost does not. A 6 dB high-shelf
cut passes the bandwidth check; a lowpass at 900 Hz does not.

### Resonances are compared against the source

Plenty of instruments have a strong resonance of their own — a snare, a resonant
guitar body. Failing a render because the *source* had a peak would throw away
good work for a property of the recording. What the check looks for is a peak the
processing **introduced**.

### Checks that need context are skipped, not guessed

`graph` and `source_report` are both optional. Every check needing one is skipped
without it — but the checks that catch genuine corruption need neither and always
run. A gate that only works when fully informed is not a safety net.

## What is still not built

- **The live model calls.** `AnthropicClient` follows the documented API shape and
  has never run. The first two things to check with a key are the ones that cannot
  be verified offline: that `cache_read_input_tokens` exceeds 15000 on a repeated
  request, and that structured output actually honours the schema's enums.
- **CLASSIFY** — instrument family, clean-versus-driven, DI-versus-miked.
- **CLASSIFY** — instrument family, clean-versus-driven, DI-versus-miked.
- The **preset knowledge base** and the artist-alias routing table. The glossary
  routes `vintage` and `lofi` to it; it does not exist.
- The **twenty missing glossary terms**, which are the genre- and
  instrument-specific vocabulary. Those should be authored alongside the presets
  by someone who mixes for a living rather than guessed at.

The scoreboard for all of it already exists: `docs/evaluation.md` describes the
rule-based recovery floor of **F1 0.580** that any planner has to beat.
