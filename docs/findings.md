# Findings

Every bug found and lesson learned, with the evidence. This is the document to
read when wondering why something is written the way it is — most of the
odd-looking decisions in this codebase are there because the obvious version was
measured and found wrong.

Entries are grouped by what they teach rather than chronologically, because the
patterns matter more than the sequence.

---

## The recurring lesson

**Every mutation-testing run on a new node found at least one test that passed
while the property it named was broken.** Twenty nodes, twenty times. The tests
were not sloppy; they were subtly measuring something adjacent to what they
claimed.

`Dither` was the first node where all nine mutations were caught on the first
pass. It took eighteen nodes of practice to write a test suite that worked
immediately — and the streak did not survive the next node. `ZeroPhaseEQ`
caught all nine of its mutations first time; `LinearPhaseEQ` let two of seven
through, both of them defences whose effect is invisible at the parameter values
the test happened to use. See "the textbook window made the design worse" below,
and the tap-count parity case: forcing an even `taps` down to odd changes
nothing observable through `latencyFrames()`, because `(512-1)/2` and `(511-1)/2`
are both 255. Only the kernel's LENGTH shows it.

Second recurring lesson, arrived at later: **synthetic test material is more
flattering than real audio, twice in a row and in the same direction.** The codec
detector and the drive analysis both passed on synthetic signals and were wrong
on real stems.

---

## Tests that passed while measuring nothing

### Self-referential expectations

The compressor's static-curve test compared the node's output against
`GainComputer` — the production code. A 10% parameter mutation passed all 139
checks. Rewritten with the expected reduction spelled out longhand.

**Rule:** a test's expectation must not import the thing under test.

### Measuring where the effect cannot appear

The dynamic-EQ selectivity test measured at a frequency where the bell had no
effect at all. Rewritten as a two-tone probe: a quiet 8 kHz tone plus a loud
400 Hz interferer, read from an FFT bin.

### Measuring the wrong window

The gate's lookahead test could not isolate `RunningMax`, because delay alone
anticipates a step onset. A dip test *also* failed to isolate it: with lookahead
L, the gain dip lands L samples *before* the audio event. Fixed by scanning the
whole loud region.

Same category, twice over, on the offline EQs. The registry's shared
steady-state gain probe analyses the **last** window of the render, which is
right for every causal node and wrong for a two-sided one. A zero-phase filter
reaches backwards as well as forwards, so the end of a render is the response
decaying into the silence past the signal, not a steady state. A +6 dB bell at
200 Hz read +11.96 dB instead of +12.00 — an error that scaled with the filter's
reach, which is exactly the pattern that makes it look like a filter bug. Moving
the window into the middle of the signal made it read +12.0000.

The same trap one level up: the phase probe for the reach-capped filter used a
65 536-sample analysis window on a filter that rings for ~56 000 samples, and
the window's own truncation contributed 0.08° of the measured error. Widening
the window made the number *worse*, which is how it was caught — a measurement
artifact shrinks when the measurement improves, and this one grew.

### Insufficiently independent references

The limiter was checked against a halfband cascade that under-reported in the
same direction, so the errors cancelled while the limiter overshot by 0.502 dB.
Replaced with an over-specified upsampler validated against an analytic answer.

Same category, later: the Python true-peak meter was checked against scipy's 64×
`resample_poly`, whose default Kaiser window has ~1% passband ripple — **0.086 dB,
larger than the error being measured.** Comparing two approximations tells you
they disagree, not which is right. Replaced with the analytic answer: a sampled
sinusoid's band-limited reconstruction *is* that sinusoid, so a unit sinusoid's
true peak is exactly 0 dBTP at every frequency and phase. Against that the meter
is accurate to **0.0001 dB**.

### Floored measurement functions

A mid/side null test bottomed out at −240 dB — which was `linToDb`'s safety
clamp, not the signal. The real value was −313 dB.

### Vacuous tests

Bypass and latency tests passed before `ReferenceDelay` existed, because nothing
in the suite had any latency to compensate.

### Tooling that reported false negatives

Three plate-reverb mutations were recorded as "not caught" when they had been
detected. Cause: several tests print progress without a trailing newline, so a
failure message was appended mid-line and a grep anchored to `^  FAIL` missed it.
`check()` now emits a leading newline.

Two more in the same script: patterns containing `/` parsed as regex modifiers
and failed silently (switched to `{}` delimiters), and build failures were counted
as "not caught" rather than inconclusive.

**Rule:** verify the harness before trusting a negative result.

---

## Silent fallbacks

Each of these produced plausible-sounding output that was not what was asked for.

### `filter_type` fell back to a bell

An unrecognised filter shape returned `FilterShape::Bell`. A planner writing
`high_shelf` instead of `highshelf` got a bell, a plausible result, and no
indication anything was wrong. Found by the cross-validation suite, where a
high-shelf test measured −1.3 dB instead of −8.0.

Now rejected by name in `validateRecipe`, alongside the already-closed node type
set.

### Band levels substituted −120 dB

`band_levels_db` returned −inf for every third-octave band below about 60 Hz, on
every file: at a 50 ms window the FFT bins are 23 Hz apart while the 50 Hz band is
11.6 Hz wide, so the band contains no bin and sums to zero. The report was
substituting −120 dB.

That substitution is worse than propagating −inf. **A fabricated number is
indistinguishable from a measurement once it is in the report**, and it is exactly
the kind of value a planner will do arithmetic on. Window is now 200 ms, narrow
bands interpolate the density at their centre, and a band with genuinely no energy
reads `null`.

---

## Measurements that were wrong about the world

### `StereoWidth` declared latency it did not have

It reported crossover latency even when running as a mono pass-through, so the
chain trimmed thousands of samples off the front of the render. **A node's
declared latency must describe what it actually does in the configuration it was
prepared for.**

### "Zero phase" was assumed to mean "zero declared latency"

`ZeroPhaseEQ` was specified with `latencyFrames() == 0`, on the reasonable
grounds that filtfilt is latency-free everywhere it appears — MATLAB, SciPy, and
every offline EQ that advertises the property. **It is latency-free only because
those implementations are handed the whole signal as one array.** A zero-phase
filter's impulse response extends backwards in time: output sample `k` depends
on input samples after `k`. Under a block-streaming node contract that is
lookahead, and lookahead is latency.

There is no way around it and no existing mechanism in the engine to exploit:
`Chain::render` is a whole-signal driver, but it drives `processBlock` in order,
so the node still sees the file a block at a time. Declaring 0 would not make
the node faster — it would make the chain trim the wrong number of samples off
the head of the render, which is the `StereoWidth` bug above with the sign
flipped.

The node declares `chunk + reach` exactly and the chain compensates it. Phase
measured after compensation is 1.2e−11 degrees across 20 Hz–20 kHz, which is the
property the name was actually about.

### `ZeroPhaseEQ` is not zero-phase below ~100 Hz at high Q

The reverse pass is started from rest `reach` samples ahead of each output
sample, where `reach` is measured as the point at which the cascade's remaining
impulse-response tail falls below 1e−12 of its L1 norm. For a 20 Hz bell at
Q 12 that point is **506 000 samples — 5.3 seconds.** The cap is 24 000, so the
filter gets 4.7% of the memory it needs and the reverse pass is missing 27% of
the response's L1 norm.

Measured consequence: 6.6° of residual phase below 100 Hz and 0.12° above it,
against 1.2e−11° for any filter that fits inside the cap. The same filter as
`parametric_eq` is 19.3°, so it is still the better of the two — but "zero
phase" is a claim it cannot keep there, and the contract test asserts the
degraded number rather than a wish. See `open-questions.md`.

The general shape of this: **the reach criterion has to be the L1 sum of the
discarded tail, not the amplitude of the last sample kept.** A high-Q filter
sits far below any per-sample threshold while several thousand of those samples
still add up to something audible. Mutating the criterion from a running sum to
a single sample was caught, and it is the only mutation in this pair that a
plausible implementation would have gotten wrong by accident.

### The textbook window made the linear-phase design *worse*

`LinearPhaseEQ` frequency-samples the target magnitude and windows the result,
which is the standard recipe and exists to suppress Gibbs ripple from
truncating an infinite ideal kernel. Removing the Kaiser entirely passed every
assertion in the node's test section — so the window was measured directly.

At 4095 taps the **unwindowed** design is the more accurate one: +6.0000 dB at
a +6 dB bell's centre against +5.9910 windowed, and 0.0000 dB worst deviation
from `parametric_eq` against 0.0034. The reason is that the tap budget already
exceeds what the filter needs, so there is no truncation to suppress and the
window is only tapering coefficients that had converged.

At 255 taps it reverses hard: a −15 dB Q 4 cut at 300 Hz leaks **0.238 dB**
into 2–20 kHz unwindowed and **0.0035 dB** windowed, a factor of 68. The window
is kept, the 0.009 dB is the price, and the test now measures at 255 taps
because that is the only place the window's job exists.

**The general lesson:** a defensive measure whose benefit depends on a parameter
has to be tested at the parameter value where it binds. Tested anywhere else it
looks like pure cost, and the obvious conclusion — delete it — is wrong.

### `DynamicEQ` upward mode pinned at full boost

The detector was fed the instantaneous rectified signal, so it latched onto zero
crossings. Fixed by enveloping the level before the gain computer.

### Plate reverb decay was off by a factor of two in the exponent

A full circuit passes through both tank halves and **each half applies the decay
gain twice**, so round-trip gain is `decay⁴` rather than `decay²`. The corrected
derivation still overshot by ~1.35×, because the tank's Schroeder allpasses have
lossless sub-loops where energy recirculates without crossing a decay gain.
Calibrated to −1.02 with the reasoning recorded, and the topology's ~0.52 s decay
floor documented rather than silently clamped.

### Clipping detection fired on undamaged audio

A sine is quadratic near its peak, so a clean 100 Hz tone at 0 dBFS holds about
five consecutive samples within 0.0005 of full scale purely from geometry. A
run-length test reported **2% clipping** on a perfectly clean file — and that
fires the "do not add drive" warning on material that would take drive well. A run
now qualifies only if it is *flat*: a clamp is a horizontal line, a peak is
curved.

### The codec detector flagged real instruments as MP3

It looked for the point where the spectrum fell 30 dB below the midrange. **Every
dark source crosses that eventually.** On real stems it placed a bass at 4 kHz, a
vocal at 10.5 kHz and an acoustic guitar at 15 kHz, and warned that two of the
three were lossily encoded. All three had continuous energy to 20 kHz. The
instruction it produced — *do not spend a band on air* — was wrong,
high-severity, and aimed at exactly the material an engineer would most want to
brighten.

What separates a codec is not how far the level falls but **how fast**:

| | steepest rolloff |
|---|---|
| bass, vocal, acoustic guitar | 3 – 37 dB/octave |
| MP3 at 128, 96, 64 kbps | 425 – 443 dB/octave |

An order of magnitude with nothing between. Threshold now 150 dB/octave,
validated against real LAME encodes, on a 1/12-octave grid — third-octave bands
give only three points above 10 kHz and cannot localise a wall at all.

A second guard came from the tests: a source with no noise floor keeps falling
toward numerical zero, which on a log scale looks arbitrarily steep. A 6th-order
lowpass at 5 kHz reported a wall at 21 kHz. Cliffs are only considered where the
level is still within 60 dB of the midrange.

### The codec detector's confidence described the cliff, not the decision

The rewrite above got the verdict right and the confidence wrong. Confidence was
`(slope − 150) / 200` — a function of how steep the cliff was, and of nothing
else. Steepness is the strength of the evidence. It is not the certainty of the
conclusion, because the conclusion also rests on the depth guard, and a cliff can
be vertical while sitting a decibel from being ruled out entirely.

Measured on the demo vocal chain, with the engine doing the filtering:

| | slope | depth below midrange | margin inside the 60 dB guard | confidence |
|---|---|---|---|---|
| before | 325.6 dB/oct | 55.5 dB | **4.5 dB** | **0.88** |

Lifting a 3 kHz bell on that output — which does not touch the audio above
15 kHz at all, only the reference the guard is measured against — walks the
margin down a decibel at a time. At a margin of 1.0 dB, one decibel from the
verdict changing, the old code reported **confidence 1.000**. A cascade of engine
lowpasses shows the same thing from the other side: three sections report a wall
at 0.86, four sections report no wall.

The wall it was so confident about was not a codec at all. It was the engine's
own 96 kHz → 48 kHz resampler filter, at 21.2 kHz, in a region already 55 dB down.

Confidence is now the **weaker of two margins**, because a conclusion cannot be
better supported than its shakiest premise:

    slope   vs the threshold that actually binds
    depth   vs max_depth_db

Two things fell out of writing it. First, the slope gate at 150 dB/octave never
decides anything: a drop is a slope times the 0.118-octave window, so the 20 dB
drop gate binds at 169 dB/octave and is always the stricter of the two. Measuring
the margin from the nominal 150 reported 0.09 confidence at the exact frequency
where the verdict changed. Second, the depth term needs normalising by half the
guard rather than all of it — real encodes cluster at margins of 36 to 39 dB and
the unstable readings at 0.3 to 8.8, and half the guard saturates on the first
population without reaching the second.

Verified against real encoded material rather than synthesised walls. Seven
encodes of the acoustic guitar stem — lame at 64, 96, 128 and 192 kbps and
afconvert AAC at 64, 96 and 128 — every one still detected at **1.00**. Clean
stems still `None`. The demo case dropped from 0.88 to **0.15**, below the 0.4
the planner acts on.

**Rule:** a confidence must be a function of distance from the decision boundary.
If it is a function of the measurement instead, it will read high on exactly the
cases where it is a coin flip, and it will do so most confidently at the moment
it is most wrong.

### R-squared was the wrong fit metric

It measures the variance the *slope* explains, so perfectly pink noise — flat,
zero slope, the best case there is — scores near zero and would be flagged
"tilt unreliable" on the most common good input. Replaced with RMS deviation from
the fitted line, which has no such degeneracy and is in dB rather than being a
unitless fraction.

### The tilt sign convention was unnamed

On band-integrated levels white noise reads +3 dB/octave and pink reads 0 — the
opposite of the PSD convention. Both are defensible; a planner assuming the other
one **inverts every tonal move it makes**. The key is now
`tilt_db_per_octave_re_pink`.

---

## Claims I made that were wrong

### "Drive does not brighten"

Stated confidently in a commit message and a code comment, on the basis that
measured across 0–24 dB of drive the HF rolloff moved only −0.59 → −0.43
dB/octave. I called it a compliment to the engine's anti-aliasing.

Both halves of the measurement were wrong: full-band pink noise as the source,
and a summary statistic as the readout. On real stems with band levels visible,
12 dB of drive moves the 8 kHz band by **+33.5 dB on a bass** and +6.6 dB on an
already-bright acoustic guitar. Saturation generates harmonics of the
fundamentals, so a source with nothing above 5 kHz gains enormously up top.

The rolloff missed it because on full-band material every band rises together and
the *shape* barely changes. Pink noise misled because it already has energy
everywhere, masking exactly the effect being measured.

### "The report cannot localise mid-band EQ"

Nearly stated, on the basis that a 1 kHz bell moved no summary statistic that a
plain gain change did not move identically. The cause was that
`sensitivity.flatten()` **dropped lists**, so the analysis never saw
`third_octave_db` — which is in the report, and which the planner therefore does
see. With bands expanded the bell is recovered at +0.98 dB per dB on its own band
at r = 1.00. **The analysis was blind, not the report.**

### "Gating removes room tone from the spectrum"

Wrong framing. Gating cannot remove noise that coexists with the signal;
expecting it to is expecting noise reduction from a level threshold. What it fixes
is the *shape*: averaging over a file that is 75% silence divides the programme
bands by four while leaving a continuous floor untouched, tilting the spectrum
about 6 dB toward the bottom.

### "The empty recipe is bit-exact"

Asserted, and it failed at −32 dB, which looked catastrophic. Per frequency the
round trip nulls at −194 dB at 1 kHz, −176 at 10 kHz and −168 at 18 kHz, then
falls off a cliff: 22 kHz returns 12 dB down. The test signal simply had energy
above the conversion filter's 20 kHz passband edge. The guarantee is transparency
*in the audible band*, and both halves are now asserted.

---

## Metric design mistakes

### Effect size pooled different treatments

The first sensitivity metric scored a field by mean shift over its own spread,
pooling every parameter value as though they were repeats of one experiment. They
are different treatments — and the consequence was that **gain, the most obviously
visible effect there is, scored as undetectable**, because sweeping −6, −3, +3,
+6 dB moves loudness down twice and up twice and the consistent-sign test failed.

### Correlation without magnitude

A 1 kHz bell moves the HF rolloff at r = −1.00 with a slope of 0.0085 dB/octave
per dB. A flawless relationship whose entire swing is under a tenth of a dB —
below the report's own two-decimal rounding, so **absent from the document the
planner receives**. Correlation now needs a span floor alongside it.

### Fixed thresholds across sources of different loudness

A compressor at a fixed −28 dBFS crushed a bass at −11.5 LUFS while barely
touching a guitar at −22.6, making a sweep of the *ratio* into a sweep of two
variables. It cost crest factor as a detectable field. Perturbations now receive a
source context and set thresholds relative to the material's own loudness — which
is also how a planner will emit them.

### "Not applicable" scored as "not detected"

All four original stems were mono or dual-mono, so stereo width moved every field
by exactly 0.00 — correctly, since there is no side signal — and the harness
reported a measurement failure for a node behaving exactly as designed.
Perturbations now declare what they need, and corpus gaps are reported separately
from measurement gaps.

### A name-keyed range table that had already diverged

The graph validator clamped parameters using a hand-maintained name-to-range map.
A band's `gain_db` is bounded at plus or minus 18 while a gain *node*'s is plus or
minus 24, and **a map keyed on the name cannot tell them apart** — it silently
applied the wider bound to bands, so a 22 dB band gain passed a clamp the schema
itself would have rejected.

Bounds are now read from the Pydantic models, removing the second source of truth
entirely. Names are not unique across contexts, and any lookup that ignores
context is wrong the moment two nodes share a parameter name — which they already
did.

### An ordering metric that measures the corpus

The recovery baseline's ordering score reads **+0.498**, which resembles partial
ability. The baseline emits a *fixed* sequence and has no ordering information at
all; it scores +1.00 on conventionally-ordered chains and −1.00 on reversed ones,
so the number measures how conventional the corpus is. Pinned by a test so it
cannot be mistaken for skill, and the corpus deliberately contains a chain and its
exact reverse.

### A recogniser that could not name two nodes at once

Reverb and delay were selected with an if/else, so a chain containing both was
structurally impossible to recover. The two-node `space_only` template scored 0.12
for that reason alone.

---

## Assertions that were arithmetically wrong

Cases where the code was right and my expectation was not. Worth recording
because each one cost time chasing a phantom bug.

| Assertion | Reality |
|---|---|
| 16-bit round trip preserves +1.0 | Two's complement runs to +2^(n−1)−1. −1.0 is an exact code; the largest positive value is 1.0 − step. |
| DC gain of a windowed-sinc is 1 to 1e−9 | Bounded by the design's passband ripple. Tightening past that asserts the ripple of one rate pair. |
| The compressor overshoots by X | Exponential attack reaches 1−e^(−L/τ). The analytic law matches within 0.1 dB. |
| A DC blocker at 10 Hz reads −3.01 dB | The coherent FFT grid is 5.9 Hz wide there, so the probe was actually at 11.7 Hz. |
| Drive's linear-phase latency = peak position | Linear-phase pre-ringing legitimately precedes the peak. |
| Two independent pink-noise signals correlate within 0.05 | Pink noise has far fewer effective degrees of freedom than samples. 0.15 is the honest bound; tighter is flaky, not strict. |
| The chain "made it louder" | A 3:1 compressor at −24 dBFS takes ~6.8 dB off pink noise well above threshold; +2 dB makeup does not restore it. That was asserting a property of the test signal. |
| A clamped 40 dB band gain ends at 18 dB | It ends at 12. The clamp fires first, then the boost budget scales 18 dB to the 12 dB total. The test was asserting an intermediate state, and would have broken the moment the budget threshold moved. |
| A Q 1 bell at 1 kHz is 0 dB at 80 Hz | +0.0416 dB. That is the skirt, and a linear-phase design reproducing it as +0.046 dB is correct rather than 0.046 dB in error. The assertion was rewritten to compare against `parametric_eq` at the same settings, where the honest bound is 0.0034 dB. |
| A zero-phase filter can declare zero latency | Only if it is handed the whole signal. Under a block contract, needing R samples of the future *is* R samples of latency. |

---

## Resource and robustness gaps found by sweeping

- **Four dB parameters overflowed to Inf** at 1e5: `gain_db`, `drive_db`,
  `output_trim_db`, `makeup_gain_db`. All dB parameters now clamp to ±80.
- **Unbounded `lookahead_ms`** was a resource-exhaustion path: a 9.6 M-sample
  window at O(W) per sample. All allocation-driving time parameters are now
  bounded above. The sweep went from 20.2 s to 1.4 s.
- **`reference_delay` accepted a 100 000-sample latency.** Bounded to 48 000.
- **`std::stod` throws `std::out_of_range` on `1e400`** — a non-`JsonError`
  escaping the parser, so a caller catching `JsonError` would abort on input a
  user can type. Uses `strtod` and a finiteness check.
- **A convolver partition-order bug** produced benign-looking garbage and passed
  every functional assertion. ASan caught it instantly. Hence `make san`.

---

## Findings from the rule-based planner

### A −5 dB cut at 100 Hz on a bass guitar

The worst output this project has produced, and it came from a rule that reads
sensibly. Scoring each band against the distant 500 Hz to 4 kHz average, the
resonance rule proposed cutting 100 Hz by 5 dB on a bass — because 100 Hz sits
**+41 dB** above the midrange there.

It does. That is what a bass is.

A resonance is **locally** peaked, prominent against the bands either side of it.
A broad elevation is the instrument's range. The glossary already said this under
*boxy* — "look for a peak rather than a broad lift, because a broad lift there is
body, not box" — and the rule was written without consulting it.

### The move that won the listening test cannot be derived from measurement alone

The fixed rule cannot reproduce the change that made the guitar sound better, and
that is the honest outcome rather than a reason to loosen it.

The 4 dB I took out of the acoustic guitar's 250 Hz by hand is a **broad
plateau** — only +2.5 dB above its neighbours at its peakiest. It was a tonal
balance decision, not a resonance fix. The measurement told me *where*; knowing
what an acoustic guitar should sound like told me *whether*.

That second half is instrument knowledge. The rule-based planner has none, which
is exactly what CLASSIFY and the preset knowledge base are for — and a concrete
reason to expect a model with world knowledge to beat this floor rather than
merely match it.

It also qualifies the listening result: measurements **supported** the decision
but did not **contain** it.

### Two test sources that could not test what they claimed

A synthetic generator took a `tilt` parameter feeding a shaping exponent
directly. `tilt=-8` produced a measured rolloff of −2.5 dB/octave, nowhere near
the −12 the planner needs before it will consider brightening — so the test that
depended on it passed for the wrong reason and could not have failed.

Reparameterising it to specify the slope directly then hit a second wall: past
about −17 dB/octave the top bands fall to numerical zero and the measured fit
floors out, so the generator returned −16.8 whatever was asked for. Chasing that
would have been testing the noise generator.

The gating tests now set the report fields directly. What is under test is the
decision, and the measurement layer that produces those fields has its own suite.

## The first listening evidence

**31 July 2026.** The first human judgement of any output this project has
produced, and the first evidence of any kind that the central thesis works.

Two chains were rendered on the same acoustic guitar stem, loudness-matched:
a generic vocal chain, and one written from that file's measured spectrum.
The listener preferred the measured chain.

That thesis — *hand the model numbers rather than audio, and it makes better
decisions* — is what the entire architecture rests on. It came from three
published results about embeddings underperforming measured features, and until
now nothing in this repository had tested it on a person.

The two chains differed in ways the measurements predicted:

| | 80 Hz | 250 Hz | crest | PLR |
|---|---|---|---|---|
| source | +3.5 | **+9.2** | 24.6 | 25.4 |
| vocal chain | **+0.8** | +7.7 | 22.1 | 22.5 |
| measured chain | **+2.6** | **+5.4** | 24.0 | 24.6 |

The vocal chain's 90 Hz high-pass removed most of the guitar's bottom octave,
because it was set for a voice with nothing at 80 Hz. It also under-treated the
+9.2 dB body resonance at 250 Hz that the measured chain took twice as much out
of, and boosted air on a source already sloping at only −3.9 dB/octave.

### What this does not establish

The evidence is real and it is the weakest usable form of it:

- **One file, one listener, not blind.** The listener knew which was which.
- **I authored both chains**, knowing which was meant to win. A generic chain
  chosen to lose is not a fair opponent, and while the vocal chain was written
  in good faith before the guitar was measured, that is not the same as being
  independent.
- **Neither chain came from a model.** This tests whether measurements support
  good decisions, not whether a planner can make them.

It moves the prior. It does not substitute for the Phase 0 gate — three working
engineers, ten guitar DIs, blind, against *their own* chains — which remains the
only test that decides whether this is worth paying for.

## Findings from dynamics matching

Two of the plan's specified methods did not survive measurement. Both are
recorded here rather than quietly replaced, because the replacements look
arbitrary without them.

### Loudness range is not monotonic in ratio, so bisection on it cannot work

The plan specified finding the compressor ratio "by bisection on rendered LRA".
Bisection requires a monotonic response. Measured on the reference stems:

| | LRA against ratio 1 → 20 |
|---|---|
| vocals | 2.95 → **0.79 at ratio 4–6** → back up to 1.04 |
| stereo-guitar | 0.84 → **1.15**, rising throughout |
| bass | 1.20 → 0.47, monotonic |

Two of three would fail to converge or converge on the wrong side. The cause is
real: past a certain ratio the gain reduction *is* a modulation. Pumping is
variance, and variance is what loudness range measures. On already-flat material
— the guitar sits at 7.2 dB crest — that is the only thing compression can do
to it.

**Gain reduction at the loud percentile is monotonic on all three**, with no
violations anywhere in the sweep, because it follows the static curve. It is also
what an engineer already thinks in.

This only shows up on real material. On synthetic pink noise with a clean
envelope, loudness range falls monotonically and bisecting on it works fine — the
third time synthetic material has been more flattering than real audio. The
assertion accordingly lives in the real-stem test.

### The envelope modulation spectrum does not yield time constants

The plan specified deriving attack and release from the frequency where the
source's and reference's modulation spectra diverge by 3 dB. The reasoning is
sound — a compressor low-passes the amplitude envelope — and the measurement
rejects it.

On a source with envelope energy spread across 0.3 to 25 Hz, at 8:1:

| compressor | 4 Hz | 30 Hz |
|---|---|---|
| 1 ms / 30 ms | −0.2 dB | +1.0 dB |
| 15 ms / 150 ms | **+3.8 dB** | +3.2 dB |
| 80 ms / 800 ms | +2.3 dB | +1.3 dB |

Compression **adds** modulation at exactly the rates the method reads, because
the gain movement is itself a modulation — the same mechanism behind the loudness
range result above. There is no 3 dB corner to find, and the first implementation
duly returned 582 ms for the fast compressor and 43 ms for the slow one: a time
constant read out of noise, inverted.

Replaced with a derivation from the **source's own** modulation rate, which is
measurable and is the more useful quantity anyway — a compressor should move at
the speed of the material.

### Stem-dependent tests were silently skipping under `make`

The stems path was relative to the working directory, and `make test` runs from
`backend/`. Every test needing real audio skipped without saying so — including
the loudness-range measurement that decided the design. Resolved against the
repository root. A test that quietly does not run is worse than one that fails.

## Findings from the verify gate

### Widening decorrelated material is genuinely mono-incompatible

The gate failed a 130% stereo-width render in a test whose whole point was that
ordinary chains must pass. It was right: two *fully independent* channels widened
to 130% give correlation −0.256 by arithmetic, and that does cancel on mono.

The test source was the problem. No recording is fully decorrelated — the
reference stems measure 0.70 and 0.86 — so testing "ordinary renders are not
rejected" against the most extreme stereo field possible was testing the wrong
thing. Fixed by building a realistic mid-plus-side source.

### One Inf poisoned every measurement in the report

A single non-finite sample makes every cumulative sum downstream produce NaN, so
a file with one bad sample came back with a report full of nulls and no
indication which thing was wrong. The count is what matters and the rest of the
signal is still worth measuring, so non-finite samples are now counted first and
then replaced with zero.

## Findings from the refine stage

### A soft clamp that attenuated everything it touched

Match-EQ bounds its correction curve so it cannot chase a 20 dB difference at
40 Hz. The obvious implementation, `limit * tanh(x / limit)`, is smooth
everywhere — and attenuates everywhere.

Recovering a known 5 dB EQ against a 6 dB limit, it returned **4.09 dB**, which
is `6*tanh(5/6)` exactly. An 18% error on a correction that was never near the
limit, introduced by the safety mechanism rather than by the measurement. It was
only visible because the round-trip test compares against a filter that was
actually applied; a fit judged only against its own target curve would have
looked perfect.

Replaced with a proper soft knee: linear below 60% of the limit, bending above
it. Same shape as a compressor's soft knee and for the same reason. Recovery
error fell from 0.29 to 0.09 dB RMS, and gains now come back within 0.2 dB.

### Fitting to the band budget rather than to the material

The fit used all the bands it was allowed. Recovering a two-band EQ with a
five-band budget it returned four: the 420 Hz bell split across 335 and 470 Hz,
plus a shelf accompanied by a bell doing almost nothing.

Numerically that is a slightly better fit. As something to hand an engineer it is
worse — they cannot tell which bands matter, and the two halves of a split band
cannot be adjusted independently without breaking the shape. Every size from one
band upward is now fitted and the smallest within 0.15 dB of the best is chosen,
so a band has to earn its place. A two-band EQ now comes back as two bands.

### An optimiser seed outside its own bounds

Greedy placement subtracts each band before choosing the next, so wherever a band
overshoots, the residual there is *larger* than the original target — and a later
band could be seeded with a gain beyond the clamp that produced the target in the
first place. `least_squares` refuses to start from outside its bounds. Clipped,
since the seed is only a starting point.

### Identical match results across three different stems

Alarming at first glance and correct on inspection: when the reference is the
source with a known EQ applied, the spectral difference **is** the filter's
response, independent of the material. The match recovers the filter rather than
the source, which is what it is supposed to do.

## Findings from the planner scaffolding

### A glossary alias claimed by two terms

The duplicate-alias guard raised at import: `distant` was both a spatial term and
an alias of the tonal `buried`. Not a typo — a **real ambiguity**, since a source
reads as distant either from too much reverb or from no presence at all, and the
two want opposite answers. Silently resolving to one would have made the planner
reach for reverb on a dry source that simply lacked presence. Both entries now
state which measurement decides.

### Twenty-one glossary conditions could not be evaluated by anything

Found while taking the glossary from thirty-nine terms to sixty, by writing the
grounding test first: measure a real report, flatten its field paths, and require
every `check_first` to name one of them. Twenty-one of the original entries name
nothing measurable. Four are the interesting kind — `dry`, `washy` and `distant`
turn on a **reverb decay** and `pumping` on an **existing compressor's release**,
and the report produces neither, though `report.py`'s own docstring says "blind
reverb time" is one of its estimates. The prose was written from what a report
*should* contain.

The failure mode is not a missing move, it is a confident one: the condition
cannot be checked, so the stock move happens anyway, now with a sentence
attached that reads like evidence. The test exempts the twenty-one by name rather
than by date, so every term added from here is enrolled automatically.

### The exemption list was itself a way to make the test pass

The mutation that mattered, and the one that survived first time. Adding a new
term's name to the exemption list silences the grounding check for it, and the
suite still passed — so "every term names a measured field" was an assertion the
next author could satisfy by editing the list instead of the entry, which is
exactly the shape of every test in this repository that has passed while the
property it named was broken.

Fixed by pinning the length: the list may only shrink. Thirteen of fourteen
mutations were caught before this one was added; fourteen of fourteen after.
Worth stating plainly because the grounding test looked complete and specific
while carrying its own escape hatch.

### `presence` did not resolve

`present` was an alias of `forward`; `presence` was not. The commonest spelling
of one of the commonest requests returned `None` and the term fell through to the
model unresolved. Found by asserting the glossary answers the seven words the
project's own documentation uses as examples — a test worth having precisely
because it checks the thing everyone assumes is already true.

### A cache-boundary test that checked the documentation

The prefix assertion probed for the string `TPT`, which appears in
`docs/engine.md` and **nowhere in `ParametricEQ.h`**. The test would have passed
with the DSP reference dropped from the prompt entirely, as long as the docs still
mentioned it. Probes now come from the source files themselves, plus a size band
on the prefix — well under and the reference has silently stopped being included.

### The audition could not promise anything about the delivery

Found by the user, at the download button, which is the worst place to find it.
The excerpt render passed every gate; the full render came back
`true_peak` — *"The processed audio would clip on playback, so it was not used."*

The gate was right both times. The two renders genuinely differed:

| | source true peak | rendered | gate bar | verdict |
|---|---|---|---|---|
| excerpt, 20 s | −2.53 dBTP | −2.08 | 0.0 | passes |
| full, 250 s | **+0.52 dBTP** | +0.30 | 0.0 | fails |

The chain had no limiter, so the gate's bar was 0 dBTP — anything above that
clips on conversion whatever the graph intended. And the 20-second window the
plan was made from sat **2.5 dB below the file's own loudest moment**. A chain
that never touched level therefore cleared the bar on everything anyone had
listened to and failed on the part nobody had.

First suspicion was the limiter, since a true-peak limiter that leaks
inter-sample peaks is the classic version of this bug. It does not: driven at
0, 6, 12 and 18 dB into a −1.0 dBTP ceiling it returned −0.98, −0.96, −0.94 and
−0.92 dBTP. The limiter was never in the chain to fail.

**The excerpt is not at fault, and no choice of window fixes it.** Twenty
seconds cannot promise anything about the two hundred and thirty it excludes,
and picking the window by peak instead of by gated coverage would trade a
representative audition for an unrepresentative one — worse, because the whole
excerpt-first design rests on the audition predicting the delivery.

So the ceiling is not optional. `_check_order` now adds a −1.0 dBTP limiter to
any graph whose master section lacks one, as a **repair rather than a
rejection**, for the same reason the master ordering is repaired: there is
exactly one right answer and no intent to guess at. It is reported, not silent.
A ceiling the planner chose deliberately is left alone.

Afterwards, same file, same chain: excerpt −2.08 dBTP, full −0.98, both pass.
The limiter does nothing at all on the excerpt, which is the point — it is
insurance against the part of the file you did not audition.

Mutation-tested four ways: removing the repair, applying it without reporting
it, letting it clobber a deliberately chosen ceiling, and setting the ceiling to
0.0 instead of −1.0. Each was caught by a different assertion.

### Stage 1 was returning nothing, and stage 2 was covering for it

The worst bug found so far, because every layer built to catch it reported
success.

The planner is deliberately two calls: an unconstrained reasoning stage, then a
transcriber that serialises the decision. The split exists because constrained
decoding from token one measurably degrades reasoning. Stage 1 was called with
`max_tokens=8192`. Opus spends output tokens on thinking, thinking counts
against that ceiling, and a real call returned:

    stop_reason: max_tokens
    content blocks: [('thinking', 0)]
    usage: output_tokens=8192

**One thinking block. Zero text blocks.** The client filtered `content` for text
blocks, found none, and returned `""`. On other runs it returned a fragment that
stopped mid-sentence — 665 characters ending at `"LRA is only 1.98 LU,"`.

Stage 2 was then handed an empty decision. It cannot tell "no decision" from "a
decision with no prose", so it did the only thing it could: it invented the
chain from the measurement report. **The stage explicitly forbidden from
reasoning was doing all of it**, which is the exact failure the two-stage design
exists to prevent.

Nothing noticed. The chains were plausible, the validators passed them, the
safety gate passed the renders, the explanations read well. The only visible
trace was a single retry, which looked like ordinary schema friction.

Fixed in four parts, three of which are separate bugs found while chasing this:

**1. The budget.** `plan_max_tokens` is 32,000 and separate from the emit
budget. Stage 1 now returns 12,000–14,000 characters ending in a complete
sentence.

**2. The refusal.** `complete()` raises `TruncatedReply` on `stop_reason ==
max_tokens` or on empty text, and the pipeline stops before stage 2 rather than
handing it a partial decision. There is also a floor: a reply under 400
characters cannot be the four-section analysis that was asked for. An incomplete
decision is not a cheaper decision — it is a different one, made by the wrong
stage.

**3. The tool schema was unreadable.** Chasing the retries surfaced something
else: the model was returning the graph wrapped in a made-up key, sometimes
literally `$PARAMETER_NAME` and `$PARAMETER_VALUE` — the placeholder tokens. The
cause is that Pydantic factors repeated definitions into `$defs` and points at
them with `$ref`, which is correct JSON Schema, and the API does not resolve
them for the model. It was being shown properties whose shapes were references
it could not follow, and it did what anyone does when handed a form with no
field names.

Four trials crossing prefix size against reasoning length: the `$PARAMETER_`
spelling correlated with the larger prefix, but *every* trial wrapped. After
inlining every `$ref`, three trials for three returned
`['explanation', 'nodes', 'master']` directly. Costs about 50% more schema
characters, since a definition used twice is now written twice.

**4. A 1200-character cap on `explanation`.** Once stage 1 produced a real
analysis the transcriber wrote a fuller summary and blew the cap on two turns in
three — a whole graph discarded over prose length. Raised to 2400 and, crucially,
**stated in the description text**: `maxLength` is in the same category as
`minimum` and `maximum`, present in the schema and not enforced on the wire.

And a fifth, found while diagnosing: the parse-failure branch set the finding for
the retry prompt but never recorded it on the outcome, so a turn where every
attempt failed to parse returned `ok=False` with an **empty** `fatal` list. A
failure with no stated reason is the least useful thing the pipeline can produce,
and it cost real time here.

The lesson is not about token budgets. It is that a pipeline stage which
silently substitutes for a failed upstream stage produces output indistinguishable
from correct at every layer below it. `complete()` returning `""` was the whole
bug; everything downstream was working exactly as designed.

## Findings from the local API

### The edit endpoint wrote numbers into parameters that are names

`POST /api/projects/{id}/graphs/{gid}/edit` checked the value against the
parameter's declared range and then stored `float(value)` unconditionally. Every
`Literal` in the schema — `filter_type`, `detector`, `mode`, `curve`,
`noise_shaping`, `stereo_mode` — has no numeric range to be checked against, so
`_bounds_for` returned `None`, the bound check was skipped as "nothing to check",
and the coercion ran anyway.

Measured: `filter_type` set to `2.0` on an EQ band returned 200. The stored graph
then failed `EffectGraph.model_validate` on its own dump —

```
nodes.0.bands.0.filter_type
  Input should be 'bell', 'lowshelf', 'highshelf', 'notch', 'lowpass' or 'highpass'
  [type=literal_error, input_value=2.0, input_type=float]
```

— and `to_recipe` emitted `"filter_type": 2.0`, which the engine reads with
`str("filter_type", "bell")` and therefore silently answers with a bell. Someone
asking for a notch got a bell and was told nothing. That is the silent enum
fallback this project refuses everywhere else, arrived at through a type
coercion rather than through a fallback anyone wrote.

`detector` on a compressor behaved the same way. `inverted` on a polarity node
was a milder version: stored as `0.0` where the schema declares a bool, which the
engine happens to read identically because it reads that key as a number, but
which is the wrong type in the graph the inspector renders and the next model
turn reads.

**The fix is to key on the type already in the slot** rather than on the presence
of a bound. Every params dict got there through the per-type Pydantic model, so
the value sitting in the slot is the authority on what the slot accepts: a bool
takes a bool, a number takes a bounded number, and anything else is refused with
a message saying it is a name rather than a number. "No declared bound" and "no
check needed" are not the same statement, and treating them as the same is what
made a schema violation look like an accepted edit.

### Two of the three whole-graph rejections the endpoint's comment names are repairs

The endpoint documents that a single in-range parameter can still break the
graph, and names "a Q that rings, a boost budget now exceeded, a chain that no
longer has a delivery ceiling". Measured against `validate()`, only the third is
even a finding, and none of the three is fatal: the Q is repaired by reduction,
the budget by proportional scaling, the missing ceiling by insertion. The only
fatal a hand edit can actually reach is `gain_staging`.

So the test that claims "an edit that breaks the graph as a whole is rejected"
has to be built on gain staging. Written against the boost budget it would have
asserted a 200 and passed while proving nothing — the exact failure mode this
project keeps finding.

### Mutation results

Nineteen mutations of `backend/api/server.py`, nineteen caught. Two of them were
caught only as a crash rather than as a named failure, which stops the suite at
the first defect and reports one bug per round trip: serving a render whose file
has gone raises inside `FileResponse` after the response has started, and
removing the planner-name pattern sent an unknown planner down the model branch.
Both are now routed through a helper that compares the raised exception's name
like any other wrong answer, and the planner probes run with `ANTHROPIC_API_KEY`
removed from the environment — otherwise a regression in an enum pattern would be
"proved" by making a paid API call on every CI run.

## Frontend findings

Kept for completeness; the hero animation predates the engine work.

- **rAF never fires in a backgrounded tab** (`document.hidden === true`). I
  diagnosed a "frozen renderer" that was nothing of the kind, and confirmed it by
  measuring a busy loop at 19 ms.
- **`ease-in-out` with `alternate` spends half its cycle near-stationary**, which
  reads as no movement at all. Rewritten as closed loops on `linear` timing.
- **Overlay blending desaturates**, draining the palette to grey. Fixed with a
  `mix-blend-mode: color` tint pass.
- **Two filtered stages hazed over the detail**, because the upper stage isolates
  as its own stacking context. One warped stage plus sibling grading passes.
- **`numOctaves` 4 → 2 flattened the marbling**: low-frequency noise becomes
  near-uniform, so the displacement map shifts rather than warps.
- Chrome would not resize past 500 px via automation and `window.resizeTo` is
  blocked, so the desktop layout was never verified in-session.
