# The stage that returned nothing

**4 August 2026.** A day of closing known debts that turned into finding the
worst bug in the project so far — one that had been running the whole time, that
every layer built to catch it reported as success, and that was only visible
because a *latency* investigation went one level deeper than it needed to.

## What happened

The planner is two model calls on purpose. Stage 1 reasons about the
measurements in prose. Stage 2 transcribes that decision into the schema, and is
told in its own system prompt that it is not the engineer. The split exists
because constrained decoding from the first token measurably degrades reasoning,
and because removing chain-of-thought costs 13 points of module accuracy.

Stage 1 was called with `max_tokens=8192`. Reasoning models spend output tokens
on thinking, thinking counts against that ceiling, and a real call returned:

```
stop_reason: max_tokens
content blocks: [('thinking', 0)]
usage: output_tokens=8192
```

One thinking block. **Zero text blocks.** The client filtered `content` for text
blocks, found none, and returned the empty string. On other runs it returned a
fragment stopping mid-sentence at `"LRA is only 1.98 LU,"`.

Stage 2 received an empty decision. It has no way to distinguish that from a
decision that happened to contain no prose, so it did the only thing available:
it invented the chain from the measurement report. The stage explicitly
forbidden from reasoning was doing all of it.

## Why nothing caught it

This is the part worth remembering.

- The chains were **plausible**. A capable model given a measurement report
  writes a defensible chain.
- The **validators passed them** — they check ranges, ordering, budget and
  stability, none of which a plausible chain violates.
- The **safety gate passed the renders**, because the audio really was fine.
- The **explanations read well**, because stage 2 wrote them.

The only visible symptom was one retry per turn, which looked like ordinary
schema friction. Everything downstream was working exactly as designed. The
whole bug was `complete()` returning `""` instead of raising.

**The general shape:** a pipeline stage that silently substitutes for a failed
upstream stage produces output indistinguishable from correct at every layer
below it. No amount of downstream validation finds it, because downstream
validation is checking the substitute.

## What chasing it turned up

Four more, each independent:

**The tool schema was unreadable to the model.** Pydantic factors repeated
definitions into `$defs` and points at them with `$ref` — correct JSON Schema,
and the right thing for a validator. The API does not resolve them for the
model, so it was shown properties whose shapes were references it could not
follow, and it invented field names: literally `$PARAMETER_NAME` and
`$PARAMETER_VALUE`, with the real graph sometimes nested under a made-up key.
Four trials crossing prefix size against reasoning length all wrapped. After
inlining every `$ref`, three trials for three returned
`['explanation', 'nodes', 'master']` directly.

**A 1200-character cap on `explanation`,** unenforced on the wire. Once stage 1
produced a real analysis, the transcriber wrote a fuller summary and two turns in
three lost a whole graph over prose length. `maxLength` is in the same category
as `minimum` and `maximum` — in the schema, not enforced, so the bound has to be
in the description text where the model reads it.

**The parse-failure branch recorded no reason.** A turn where every attempt
failed to parse returned `ok=False` with an empty `fatal` list.

**The prompt cache was paid for twice and collected once.** PLAN and EMIT shared
one 30,680-token prefix and ran on different models, so it was written into two
caches and read from neither. EMIT is a transcriber: it needs the schema, not the
DSP source or the glossary. Its prefix is now 9,868 tokens, measured live.

The canary that was supposed to catch this had tested one model against one
prefix — a configuration the product never runs. **A canary that tests something
you do not ship produces a green tick and a wrong bill.**

## The honest latency picture

The investigation started because a model plan measured 128.7 s against a 45 s
p50 target. That number was never a latency measurement — it was the time to
produce nothing.

A complete decision takes **170–230 s**. The gap is much larger than it looked,
and the task is re-scoped rather than closed. Options are a thinking-budget cap,
streaming stage 1 to the UI so felt latency drops, or accepting that a full
analysis costs minutes and designing the interaction around that.

## Also today

**Cross-architecture determinism is tested for the first time.** Both CI arches
ran their own tests against their own expectations, which says nothing about
whether they agree. A committed golden render establishes it transitively. The
generator caught its own first bug immediately — rendering from memory while
committing a float32 file left a −138.5 dBFS discrepancy no machine could
reproduce.

**The codec detector's "wall" in the demo case was our own resampler.** Reported
at 0.878 confidence on a 4.5 dB margin; now 0.148, below the 0.4 the planner acts
on. Real MP3 and AAC at seven bitrates still report 1.00. The 150 dB/oct slope
gate turns out never to decide anything — the 20 dB drop gate binds first.

**The glossary went 39 → 60 terms,** and revealed that **21 of the original 39**
`check_first` conditions name measurements the report does not produce. Four turn
on a reverb-decay estimator and a compressor detector that do not exist.

**The graph inspector is editable.** Out-of-range is refused with the bound
quoted, not clamped — a model that overshoots still meant the direction, but a
person typed a specific number.

**Recipes export and import,** verified bit-identical over 960,000 samples.

**Reference matching is live,** and the reference audio is discarded inside the
request that carries it. No filename, and deliberately no content hash: a hash
would let this service answer "has anyone else uploaded this master".

**Two new EQ nodes.** ZeroPhaseEQ measures 1.2e−11 degrees of phase deviation.
Its instruction from me said latency is zero by construction; that was wrong, and
it was right to refuse — filtfilt is latency-free only when the whole signal is
one array.

**An API test suite from nothing,** 258 checks, which found a real bug in the
edit endpoint an hour after it was written: bounds were checked, then
`float(value)` stored unconditionally, so parameters that are *names* took
numbers. `filter_type = 2.0` meant someone asking for a notch got a bell and was
told nothing.

## Blocked

The API credit balance ran out mid-verification. Each planner fix is verified by
its own live experiment; the combined three-stem run is not yet done.
