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
while the property it named was broken.** Eighteen nodes, eighteen times. The
tests were not sloppy; they were subtly measuring something adjacent to what they
claimed.

`Dither` was the first node where all nine mutations were caught on the first
pass. It took eighteen nodes of practice to write a test suite that worked
immediately.

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
