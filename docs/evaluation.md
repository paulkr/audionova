# Evaluation

Four harnesses, each answering a different question. They exist in this order
because each is meaningless without the one before it.

| Harness | Question | Checks |
|---|---|---|
| Contract tests | Does every node obey the node contract? | 1321 |
| I/O tests | Do WAV, JSON, resampling and recipe validation work? | 139 |
| Cross-validation | Do the engine and the measurement layer agree? | 48 |
| Sensitivity | Can the report see what the engine does? | 101 |
| Chain recovery | Can a chain be read back out of the deltas? | 615 |
| Loudness + measurement | Are the measurements right? | 148 |
| Listening harness | Can human judgement be collected without bias? | 225 |

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

`backend/eval/generate.py`, `backend/eval/chains.py`, `backend/eval/recover.py`.

```sh
cd backend && make recover STEMS="../audio/stems/*.wav"
cd backend && make recover-synthetic     # no stems needed
```

Single perturbations are the easy case. Real chains have four or five nodes whose
effects overlap, mask each other, and in some orders cancel.

**This is the one axis that can grow without a listener or a licence.** Take a
source, apply a known chain, score whether the planner recovers it — the ground
truth is free and objective by construction. So the corpus is generated rather
than hand-written: 67 chains from a fixed seed, varying in length (1 to 5 nodes),
in node type, in parameter extremity, and in whether the move is detectable at
all. Committed as `backend/eval/manifests/recovery_v1.json` with a digest, and a
manifest that does not regenerate from its own seed is a hard error — otherwise a
score change cannot be attributed to the planner rather than to new chains.

### Current floor: F1 0.509

**precision 0.485 · recall 0.596 · F1 0.509** over 394 cases — 67 chains across
six stems — from a rule-based recogniser built entirely from the sensitivity
table. It is lower than the 0.580 the six hand-written templates produced, and
that is the corpus getting harder rather than the recogniser getting worse.

The floor is the point. A planner scoring 0.6 is either impressive or
embarrassing depending on what a handful of thresholds achieves on the same data,
and without the floor there is no way to distinguish a model that reasons from
one that has learned which nodes are common. Proxy-metric capture is a named
risk, and an unanchored score is the easiest number there is to move.

| family | cases | P | R | F1 | what it isolates |
|---|---|---|---|---|---|
| `varied` | 116 | 0.61 | 0.62 | 0.58 | length and type coverage, duplicates included |
| `ordering_pair` | 144 | 0.52 | 0.62 | 0.55 | every chain and its exact reverse |
| `single` | 50 | 0.49 | 0.70 | 0.55 | one node — a failure here is a bug, not a limit |
| `faint` | 84 | 0.25 | 0.46 | 0.32 | at least one node below the sweep's floor |

### Per node type, because F1 0.51 hides which half is broken

| type | support | found | recall | proposed | precision | F1 |
|---|---|---|---|---|---|---|
| `compressor` | 114 | 104 | **0.91** | 197 | 0.53 | 0.67 |
| `parametric_eq` | 270 | 210 | 0.78 | 305 | 0.69 | **0.73** |
| `delay` | 120 | 91 | 0.76 | 211 | 0.43 | 0.55 |
| `stereo_width` | 24 | 24 | **1.00** | 127 | 0.19 | 0.32 |
| `plate_reverb` | 138 | 28 | 0.20 | 96 | 0.29 | 0.24 |
| `drive` | 138 | 19 | **0.14** | 19 | **1.00** | 0.24 |

Three actionable facts the headline could not carry:

- **`drive` is the hole.** Recall 0.14 at precision 1.00 — when it fires it is
  right, and it almost never fires. The rule needs `top > 6 dB` *and*
  `crest < −1 dB`, and on already-bright material the first never happens. This
  is the single largest available gain for a planner with world knowledge.
- **`stereo_width` and `delay` are over-proposed, not missed.** Width is recalled
  perfectly and proposed five times too often; correlation moves for reasons
  other than a width node. Both are precision problems.
- **`plate_reverb` fails in both directions** on real material, which is the
  dense-source limitation below showing up as a number.

### The ordering artifact, and what was done about it

The old figure read **+0.498**, which resembles partial ability and was nothing of
the kind: the baseline emits a *fixed* conventional sequence and carries no
ordering information at all, so the number measured how conventional the corpus
was.

**Both available fixes are applied; the obvious third one does not work.**

1. **Ordering is scored only over genuinely varied chains.** The `ordering_pair`
   family generates every chain together with its exact reverse, sharing the same
   parameter draws, so a fixed-sequence predictor scores +ρ on one and −ρ on the
   other. Chains outside that family are not scored for ordering at all. The
   mirror only works when the node types are distinct — reversing "eq, comp, eq"
   returns "eq, comp, eq" — so eligible chains are built from distinct types.
2. **An order-blind control runs beside the score:** the same predicted multiset
   re-emitted in a fixed conventional sequence. Its precision and recall are
   identical by construction, so the difference is ordering information and
   nothing else.
3. **A shuffled-baseline control does *not* fix it.** Permutations come in
   reversing pairs and reversing a permutation negates its rank correlation, so
   the shuffled control is **identically zero** and excess over it is the raw
   score again. Computed, printed, and asserted to 1e−12 so nobody proposes it a
   second time.

| | over 81 mirrored chains |
|---|---|
| raw | **−0.037** |
| order-blind control | **−0.037** |
| shuffled control | +0.000 (exactly, by symmetry) |
| **excess over control** | **+0.000** |

Corpus balance alone gets the raw figure from +0.498 to −0.037; it does not reach
exactly zero because detection is not identical in the two directions — a node
found in one order and missed in its reverse breaks the antisymmetry. The matched
control absorbs that residue, and the corrected floor is **+0.000**. Any planner
scoring above zero here has read something out of the deltas.

### Parameters, reported separately from type recovery

Recovering "there was an EQ" is a much weaker claim than "there was a −3 dB bell
near 900 Hz". Scored only where `parametric_eq` was recovered, in octaves and dB,
each beside a control that says how much is luck.

| | baseline | control |
|---|---|---|
| applied bands | 234 (229 matched) | |
| median frequency error | **0.27 octaves** | 1.80 (random frequency) |
| within ⅓ octave | **52%** | 11% |
| median gain error | **1.22 dB** | 8.80 (predicting 0 dB) |
| within 3 dB | **81%** | |

Frequency is in octaves because a 100 Hz error at 200 Hz and at 8 kHz are not the
same mistake. The ⅓-octave tolerance is one band of the report's own analysis, so
a proposal inside it is indistinguishable from the truth in the document the
planner was handed.

**This is not part of the headline and must not become one.** The regression
baseline in LLM2Fx had the best parameter MAE and nearly the worst perceptual
score, so ranking planners on it would rank them wrongly. What it is good for is
diagnosis: a planner naming every node type at random frequencies has not
recovered the chain, and the type score alone cannot say so.

### Undetectable moves are excluded, not counted as misses

The corpus deliberately contains a 0.2 dB shelf, a 1.05:1 compressor, a 20 Hz
high-pass. **A harness that marks a planner down for not hearing those has a
ceiling below 1.0 that no planner can raise, and a number that cannot improve is
a number nobody acts on.** They are dropped from the truth, and a *surplus*
prediction of that type is dropped with them, so naming one is neither credited
nor punished. 80 such instances across the corpus; the recogniser named 21.

The floors are read from `corpus.PERTURBATIONS` rather than written down: the
smallest amount of each move the sensitivity sweep ever applied is the smallest
amount anything here has evidence is detectable. A hand-written threshold would
be an invented measurement.

**Inapplicable is a third category again.** Stereo width on a mono stem has no
side signal to act on; 52 instances were dropped and reported as a *corpus* gap,
not a detection failure.

### Scoring choices

- **Precision and recall, not accuracy.** A recogniser proposing every node type
  would score 100% accuracy on the ones present.
- **Multiset, not set.** "EQ, dynamics, EQ" is not recovered by naming one EQ,
  and that sandwich is the commonest structure there is.
- **The recogniser is a name in a table.** `--planner` resolves from
  `recover.RECOGNISERS` and no other way; a model planner drops in as another
  entry, and nothing in `backend/eval/` imports the model client. A test asserts
  it.

### A stated limitation

On continuously dense material neither space effect is detectable. A delay and a
plate together moved the bass stem's p10-to-p90 fill by **−0.42 dB** — nothing.
Dense sources never leave gaps for added space to show up in level percentiles,
and `plate_reverb`'s recall of 0.20 above is that limitation counted. A planner
that recovers this would be demonstrating something real.

Synthetic sources are available (`make recover-synthetic`) so the harness runs
where the gitignored stems are not, but they are **not a substitute**: pink noise
already has energy everywhere, which masks exactly the effect drive detection
depends on, and `drive` recall falls from 0.14 to 0.00 on it.

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

**The harness for running it now exists.** See below. Running a panel needs
people; building the harness did not, and it is what stood between every
number in this document and being calibrated against a person.

## The listening harness

`backend/listen/`, `backend/tests/test_listen.py`, `make listen`. 225 checks.

Nothing here measures audio quality. It measures whether a *person's* judgement
of audio quality can be collected without the four things that would make it
worthless: knowing which render is ours, hearing one of them louder, hearing
them in an order that tracks the conditions, and having no way to tell an
attentive listener from a coin.

### Loudness matching is the load-bearing part

Louder reliably wins a blind comparison whether or not it is better. So every
clip in a comparison is matched with `backend/measure/loudness.py`, and then —
the part that matters — the written files are **read back from disk and
re-measured**, and the session build fails if the pair spans more than 0.1 LU.

Applying `target − measured` dB is arithmetically exact, so it is tempting to
write the gain and move on. Three things can still make the file miss:
BS.1770's absolute gate at −70 LUFS is not relative to the programme, so
attenuating a group across it changes which blocks are integrated; quantising
to the delivery word length clamps peaks; and the gain solver is a caller-supplied
parameter. The test suite exercises the first of those directly — a group
predicted at −73 LUFS whose written files have no measurable loudness at all.

The ceiling is applied as a **common** offset. Anything per-clip is a loudness
difference reintroduced under a safety mechanism's name.

Matching turns out to be load-bearing for the screen too, by a wide margin: the
screening bell alone moves integrated loudness by **0.58 LU**, six times the
bar. Unmatched, a listener with no frequency discrimination at all passes the
screen by picking the louder clip.

### What blinding has to mean

Not "the labels are hidden in the interface" — that is a rendering decision and
a rendering decision can be undone with a right-click. The session payload
contains no condition names, no item names, no source paths, no repeat flags,
and no identifier that is a function of what it points at. The mapping lives in
a key file the server loads and never serves.

The sharpest test of that last property: build the same session with the two
conditions written the other way round and require the two payloads to be
**byte-identical** while the key's assignment is inverted throughout. That fails
the moment any identifier is derived from a condition name or a file path.

Clip identifiers are per trial **occurrence**, not per file, so a repeat trial
cannot be spotted by its filenames. Repeats sit at least three trials from their
original and are independently re-randomised, so noticing a repeat does not tell
a listener what they said last time.

Two things it does not hide are written down in `open-questions.md` rather than
implied away: the request text is shown, because one of the scales asks whether
the render matched it, and two identifiers resolving to one file are served as
byte-identical responses.

### Two scales, and an interval that is honest about three people

"Does it match the request?" and "does it sound professional?" are collected
separately because they dissociate — a render can execute the request precisely
and sound amateur. Averaging them gives one number that moves for two reasons.

**Intervals are clustered by listener.** Ratings are not independent
observations: one listener contributes many, and listeners differ enormously in
how they use a scale. A t-interval over pooled ratings treats thirty ratings
from three people as thirty draws and comes out about a factor of sqrt(10) too
narrow. Averaging within a listener first gives an interval whose n is the
number of *people*. On a three-person panel that interval spans about 2.9 points
of a 5-point scale. It is supposed to.

Repeat trials are excluded from the means and used only for consistency, which
is reported two ways: `reversal_rate` counts a listener naming the opposite
condition the second time, and `unstable_rate` counts any change including a
move to a tie. Only the first is a contradiction — on a genuinely close pair,
moving to a tie is the correct answer, and a single number would punish honesty
about it.

### The correlation, and why it will not print a bare rho

`correlate.py` takes a listening result and any set of automated metrics and
computes Spearman rho per metric. Spearman rather than Pearson because a proxy
does not have to be linear in perceived quality to gate a build — it has to be
monotone in it, and a rating scale is ordinal anyway.

Every result carries its interval (Bonett–Wright) and its n, `usable` is false
below fifteen paired points *or* fifteen ratings per point, and the printed form
has no path that yields a lone number. `may_gate_ci` tests the interval's
**lower bound** against the plan's 0.6, not the point estimate: a sample rho of
0.62 on twenty points has a lower bound of 0.28, and gating on the point
estimate is gating on noise. That is stricter than the plan's literal wording
and is recorded in `open-questions.md` as a decision rather than a reading.

An exactly degenerate result — rho of ±1, or fewer than four points — reports
**no** interval rather than a zero-width one.

### The screen and the interface

The plan's screen is a 2 dB Q=1 bell at 3 kHz identified at ≥ 80%, as a 2AFC.
Both sides are rendered by the real engine and the unprocessed side is a **null
render**, not the raw source, for the same reason every delta in this repository
is: against the raw file the engine's own conversion is inside the difference.
The suite verifies the stimulus rather than trusting it — the boost measures
+1.88 dB in the third-octave band containing 3 kHz (a 2 dB bell seen through a
third-octave filter), +0.90 dB two thirds of an octave up, and under 0.15 dB two
octaves out, which is what pins Q.

`interface.html` is one self-contained file served by a stdlib HTTP server with
no static root and no filename in any URL. A/B switching is gapless: all three
clips are decoded up front, started together on one clock, looped, and switched
by an 8 ms gain ramp. Nothing stops or restarts, so the playhead cannot move —
a gap lets auditory memory fill in the difference, and auditory memory is
generous.

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
