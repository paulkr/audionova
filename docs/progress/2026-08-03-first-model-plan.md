# First model-planned chain

**3 August 2026.** The planner produced a graph for the first time.

Source: `acoustic-guitar-track.wav`, request *"make this sound warmer and less boxy"*.

## What each planner did

| | rules | model |
|---|---|---|
| nodes | 4 | 2 |
| high-pass | 63 Hz | — |
| boxy cut | **declined** | **bell 280 Hz −3.5 dB Q1.4** |
| warmth | — | **bell 150 Hz +2.5 dB Q1.0** |
| compressor | 2.2:1 @ −23.8 | — |
| gate | passed | passed |

## Why this matters

The rule-based planner **cannot** cut the guitar's low-mid plateau. It says so
explicitly in its own explanation:

> declined: the low mids, where no band peaks above its neighbours enough to
> call a resonance — a broad lift there is the instrument's range, and
> reshaping it needs to know what the instrument is

That refusal was written deliberately, after the same rule proposed a −5 dB cut
at 100 Hz on a bass guitar. A broad lift is the instrument; only instrument
knowledge distinguishes "too much body" from "this is what a bass is".

The model supplied exactly that. It cut 280 Hz and *added* 150 Hz — addressing
"less boxy" and "warmer" as two separate moves in adjacent bands, which is a
distinction the measurements alone do not contain.

It also cited its evidence: the 250–315 Hz bands sit at −32.9/−33.9 dB against
400 Hz at −38.5 and 500 Hz at −46.8, and it named the 80–200 Hz region as *not*
elevated before boosting it.

**This is the first evidence the model earns its cost premium.** It is one file
and one request, unblind, and nobody has listened yet.

## A cost finding

The final call reported `cache_read_input_tokens: 0` and
`cache_creation_input_tokens: 32133` — it *wrote* the cache rather than reading
it.

Cause: PLAN runs on Opus and EMIT on Sonnet, and **the prompt cache is per
model**. The 31 k prefix is therefore cached twice per turn, once for each, and
neither reads the other's.

The canary passed because it used one model for both calls. It was measuring a
configuration the pipeline does not use.

Options, none yet taken:
- run both stages on the same model — one cache entry, more expensive EMIT
- keep two models and accept two cache entries — each still amortises across
  requests, so this is a warm-up cost rather than a per-turn one
- shrink the prefix for EMIT, which does not need the DSP source at all

The third is probably right: a transcriber does not need to read the compressor
implementation.
