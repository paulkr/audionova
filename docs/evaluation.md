# Evaluation

Four harnesses, each answering a different question. They exist in this order
because each is meaningless without the one before it.

| Harness | Question | Checks |
|---|---|---|
| Contract tests | Does every node obey the node contract? | 1321 |
| I/O tests | Do WAV, JSON, resampling and recipe validation work? | 139 |
| Cross-validation | Do the engine and the measurement layer agree? | 48 |
| Sensitivity | Can the report see what the engine does? | 101 |
| Chain recovery | Can a chain be read back out of the deltas? | 51 |
| Loudness + measurement | Are the measurements right? | 148 |

```sh
cd engine  && make check
cd backend && make test
```

## Why the order matters

A planner is scored on how well it recovers a chain. That score means nothing
unless the report can see the effects at all, which means nothing unless the
measurements are right, which means nothing unless the engine does what it
claims. Building the harness *after* the planner tends to produce a harness that
agrees with the planner.

The research corpus this project draws on is a case study in the failure. CLAP-
guided EQ optimisation was published as an improvement while scoring worse than
random parameters, because the metric was never validated against human
judgement.

## Contract tests

`engine/tests/contract_tests.cpp`, 22 sections, parameterised over a node
registry so adding a node auto-enrols it.

1. **Bypass null** — bit-exact zero. Because bypass still inserts the
   compensating delay, this exercises the real PDC path. Highest-value test in
   the suite.
2. **Block-size invariance** at `{1, 7, 64, 512, 1023, 4096}`. Odd and prime
   sizes are mandatory; powers of two hide index bugs.
3. **Determinism** across runs and process restarts.
4. **Reset equivalence** — bit-exact with a fresh instance.
5. **Latency correctness** — Dirac cross-correlation must match
   `latencyFrames()` exactly. Measured by *peak*, not onset: a linear-phase FIR
   is symmetric about its group delay, so energy legitimately appears before the
   peak, and onset detection would flag that pre-ringing as an error.
6. Silence → silence · 7. Robustness · 7b. Parameter sweeps · 8. Denormal control
   · 9. Counter RNG · 10. Frequency response · 11. Aliasing (ASR) · 12–22 per-node
   correctness.

`make san` builds under ASan+UBSan. This is not optional discipline: mutation
testing found a partition-indexing bug that read out of bounds, produced
benign-looking garbage, and passed every functional assertion. Memory errors do
not reliably show up as wrong numbers.

### Mutation testing

Every node was mutation-tested — deliberately break the implementation, confirm a
test fails. **Every run on a new node found at least one test that passed while
the property it named was broken.** The categories of failure found are recorded
in `findings.md`; they include self-referential expectations, measuring where the
effect cannot appear, insufficiently independent references, and floored
measurement functions.

`Dither` was the first node where all nine mutations were caught on the first
pass, which took eighteen nodes of practice.

## Cross-validation

`backend/tests/test_engine_bridge.py`. **Structurally the strongest test here.**

Every other suite checks one implementation against a reference written by the
same hand on the same day. Here the C++ DSP and the Python measurement share no
code and must agree on quantities whose values are known analytically in advance.
If the gain node and the loudness meter both had a 0.5 dB error, nothing inside
either half would find it.

What it asserts, all against analytic answers:

- gain moves integrated loudness and true peak by exactly its own value
- makeup gain moves loudness by exactly its own value
- a high shelf raises the bands above it by exactly its gain and leaves the low
  end alone
- a bell moves its own third-octave band and nothing two octaves away
- a 4:1 compressor puts a −10 dBFS tone at −17.5 dBFS, per its static curve
- polarity inverts exactly; width 0% gives correlation 1
- the limiter's ceiling holds, verified by a true-peak meter sharing no code with
  the one inside it

It also **measures the master-section rule** rather than arguing it: limiting
after the conversion holds a −1.0 dBTP ceiling at the delivery rate; limiting
before and converting afterwards overshoots. That rule costs two node lists, a
validator and a second prepare, so confirming the effect is real was worth doing.

**Always compare against a null render.** Against the raw source a −0.5 dB gain
reads −0.62, because the conversion's own K-weighted 0.12 dB is folded in.
Against an empty render it reads −0.5000, with a −306 dB residual against an
exactly scaled copy.

## Sensitivity

`backend/eval/sensitivity.py`, `backend/tests/test_sensitivity.py`.

```sh
cd backend && make sweep STEMS="../audio/stems/*.wav"
```

If a report field does not move when an effect is applied, no prompt and no
amount of chain-of-thought recovers that effect — the information is not there.
Finding that out from a table costs minutes; from a listening panel, weeks.

**Every hypothesis is written into the perturbation catalogue before anything
runs.** An analysis that reads whatever moved and calls it the answer will
validate any measurement suite including a broken one. A prediction made in
advance can fail.

### The metric

Not "did this field move" but "can you read the parameter off it". Each field is
scored by the correlation between parameter value and field delta, computed
*within* each source and averaged across sources. Two forms count:

- **monotonic** — the delta follows the parameter, so it can be inverted out
- **magnitude** — the delta follows |value|; the field says something happened
  and how much, but not the direction. A bell's residual rises whether it cuts or
  boosts.

Plus a **span floor**. Correlation alone is not enough: a 1 kHz bell moves the HF
rolloff at r = −1.00 with a slope of 0.0085 dB/octave per dB — a flawless
relationship whose entire swing is under a tenth of a dB. The report rounds to two
decimals, so a field that small is absent from the document the planner receives.

### Applicability

**Not applicable is not the same as not detected.** Stereo width on a mono stem
moves nothing — correctly, since there is no side signal — and scoring that as a
detection failure reports a hole in the measurement layer that does not exist.
Perturbations declare what they need; the runner reports corpus gaps separately.

### Current result: 10/10

All ten perturbations detected on the fields predicted in advance, most at
r = 1.00, across six real stems. Every EQ move is recovered at near-unity slope
on its own band:

| perturbation | best field | slope | r |
|---|---|---|---|
| high shelf @ 6 kHz | `third_octave_db.16000hz` | +0.98 dB/dB | 1.00 |
| low shelf @ 150 Hz | `third_octave_db.63hz` | +0.96 dB/dB | 1.00 |
| bell @ 1 kHz | `third_octave_db.1000hz` | +0.98 dB/dB | 1.00 |
| highpass | `third_octave_db.100hz` | −0.076 dB/Hz | −1.00 |
| compression | `envelope_modulation.centroid_hz` | +0.40 | 0.94 |
| drive | `third_octave_db.8000hz` | varies by source | 0.99 |
| stereo width | `stereo.correlation` | −0.0035 | −0.99 |

## Chain recovery

`backend/eval/chains.py`, `backend/eval/recover.py`.

```sh
cd backend && make recover STEMS="../audio/stems/*.wav"
```

Single perturbations are the easy case. Real chains have four or five nodes whose
effects overlap, mask each other, and in some orders cancel.

### Current floor: F1 0.580

**precision 0.547 · recall 0.694 · F1 0.580** over 24 cases, from a rule-based
recogniser built entirely from the sensitivity table.

The floor is the point. A planner scoring 0.6 is either impressive or
embarrassing depending on what a handful of thresholds achieves on the same data,
and without the floor there is no way to distinguish a model that reasons from
one that has learned which nodes are common. Proxy-metric capture is a named
risk, and an unanchored score is the easiest number there is to move.

| chain | F1 |
|---|---|
| `eq_then_compress` | 0.73 |
| `corrective_eq` | 0.71 |
| `vocal_chain` | 0.70 |
| `drive_and_shape` | 0.41 |
| `space_only` | 0.20 |

### The ordering number is not what it looks like

It reads **+0.498**, which resembles partial ability and is nothing of the kind.
The baseline emits a *fixed* conventional sequence and carries no ordering
information at all: it scores +1.00 on chains that follow convention and −1.00 on
the one that does not. The figure measures how conventional the corpus is.

The corpus deliberately contains `eq_then_compress` **and its exact reverse**.
Without both directions the metric rewards knowing nothing. If no planner can
separate them, then measurement deltas do not carry ordering and the metric
should be retired rather than chased.

### Scoring choices

- **Precision and recall, not accuracy.** A recogniser proposing every node type
  would score 100% accuracy on the ones present.
- **Multiset, not set.** "EQ, dynamics, EQ" is not recovered by naming one EQ,
  and that sandwich is the commonest structure there is.
- **Parameter values are not scored at all.** The regression baseline in LLM2Fx
  had the best parameter MAE and nearly the worst perceptual score, so putting a
  number on it would invite optimising against it.

### A stated limitation

On continuously dense material neither space effect is detectable. A delay and a
plate together moved the bass stem's p10-to-p90 fill by **−0.42 dB** — nothing.
Dense sources never leave gaps for added space to show up in level percentiles.
A planner that recovers this would be demonstrating something real.

## What none of this measures

**Recovering a chain is not the same as sounding good.** A recipe that recovers
every parameter is by definition perfect; a recipe that recovers none may still
sound better than the original to a paying engineer.

Every harness here is a **necessary condition, not a sufficient one**. They make
the listening panel affordable by catching everything that fails objectively
first. They never replace it.

The gate that decides whether any of this works is still unattempted: three
working engineers, blind A/B against their own chains on ten guitar DIs,
loudness-matched, prefer-or-tie in at least 6 of 10.

## The reference corpus

Six stems in `audio/stems/`, not in git — see `licensing.md`.

| stem | ch | rate | notes |
|---|---|---|---|
| `bass` | 2 (dual mono) | 48 k | LUFS −11.5, rolloff −27 dB/oct |
| `acoustic-guitar-riff` | 2 (dual mono) | 44.1 k | LRA 17.6 |
| `acoustic-guitar-track` | 1 | 48 k | true peak **+0.52 dBTP** |
| `vocals` | 1 | 48 k | floor −84.7 dB at confidence 1.0 |
| `drums` | 2 (real stereo, r 0.70) | 192 k | 192 k in name only — wall at 21.65 kHz |
| `stereo-guitar` | 2 (real stereo, r 0.86) | 48 k | already compressed |

Covers mono, dual-mono and real stereo; 44.1, 48 and 192 kHz; a dark bass through
a bright acoustic. Two things it does not cover: percussive transient material at
a normal rate, and anything genuinely wide at low frequencies.

Source-separated stems were considered and rejected: they carry separator
artifacts (one measured as `out_of_phase` at correlation −0.32, which no real
vocal recording does), they are several lossy generations deep, and v1 does not
do separation, so they are the wrong distribution as well as the wrong licence.
