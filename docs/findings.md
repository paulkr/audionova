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

`MultibandCompressor` and `TransientDesigner` continued it: 23 of 27 mutations
caught first time, and all four survivors were the same shape as the ones
before — a defence measured where it does not bind, or a property with two
implementations so that breaking one leaves the other holding it up. Details
under "the rectification that was already being done twice" and "the crossover
spacing that binds on one input nobody had written" below.

And then a fifth, found only because the nodes were rebuilt to fix their
aliasing: **oversampling a node moves its ballistics to a different sample
rate, and nothing in either test suite measured a time constant in
milliseconds.** The multiband's timing was pinned entirely by a bit-exact
comparison against `compressor` — which runs at `oversample_factor` 1, where
the base rate and the oversampled rate are the same number, so a mutation
deriving the coefficients from the wrong one was invisible. Every attack in the
node could have been four times too fast. The fix is a test that measures the
attack in milliseconds on a burst confined to one band, which is the test that
should have existed from the start; the proxy was cheaper to write and could
not see the thing it was standing in for.

Second recurring lesson, arrived at later: **synthetic test material is more
flattering than real audio, twice in a row and in the same direction.** The codec
detector and the drive analysis both passed on synthetic signals and were wrong
on real stems.

Third, from the first mutation run on the Python stages: **a test that probes a
threshold from far outside it proves the check exists and says nothing about
where it sits.** Forty-two of seventy-two mutations survived across the verify
gate and the loudness meter, and almost all of them were constants — moved by
half their value, inside the distance the tests happened to probe from. See
"Findings from mutation-testing the Python stages" below.

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

Third time, one axis over. The contract suite's reference for the limiter was
`design(8, 100.0, 0.010)` — thirty times the filter length of the thing it
measured, and the **same 8× peak-search grid**. Filter length was the axis
everyone was watching; the grid is a second, independent error term bounded by
`cos(π/(2·factor))`, which is 0.168 dB at 8×. So the reference was
over-specified on the axis that was being discussed and identical on the one
that mattered. Now 32×, whose own grid bound is 0.010 dB — an order of
magnitude under what the node it measures is allowed.

**Rule:** "independent" has to mean independent on every axis the error has, and
you only know the axes once you have decomposed the error.

### A bound measured on stationary signals, applied to transients

`TruePeak.h` carried this, and the limiter's oversampling factor was set to
BS.1770's minimum of 4 on the strength of it:

> Oversampling FACTOR does not affect accuracy at all. 4x, 8x and 16x produce
> bit-identical errors for the same filter length.

The measurement was real and reproducible. It was also made entirely on held
tones and steady broadband noise, and for those it is true — over hundreds of
cycles some crest lands near a grid point whatever the grid is, so the grid
stops mattering. **A limiter does not exist for held tones.** On a lone
band-limited impulse placed ⅛ of a sample off the grid, against an exact
DFT-domain reconstruction:

| content to | 4× | 8× | 16× | 32× |
|---|---|---|---|---|
| 20 kHz | −0.039 dB | −0.000 | −0.000 | −0.000 |
| 40 kHz | −0.156 dB | −0.000 | −0.000 | −0.000 |
| 48 kHz | −0.224 dB | −0.017 | −0.006 | −0.001 |

Factor is worth up to 0.22 dB, in the permissive direction, on exactly the
signal class the node was built to catch. The worst case is not a corpus number
either — for anything band-limited to Nyquist the shortfall is at most
`cos(π/(2·factor))`, attained by a tone at Nyquist, so it is 0.688 dB at 4×.

**Rule:** a claim that a parameter does not matter has to be probed with the
signal class the code is for, not the one that is convenient to generate. Steady
state is the case that hides grid, alignment and phase effects; it is the worst
possible choice for measuring any of them.

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

### The rectification that was already being done twice

`TransientDesigner` half-wave rectifies its attack differential, because that
differential goes negative through a decay and an attack boost would otherwise
quietly cut the body of every note. The test for it looked right — it asserts
the body moves less than 0.05 dB while the leading edge moves 5.4 — and it
measured **+0.0000 dB** with the rectification removed.

The cause was a second copy of the same rule three lines further down. Stereo
linking took a running maximum over channels seeded at `0.0`, so at
`stereo_link = 1` the linked value was `max(0, differential)` whatever the
differential had been. **A property with two implementations is a property no
test can falsify**: breaking either one leaves the other holding it up. The
linking now seeds from channel 0 and the rectification is stated once.

The same run found the mirror-image case one line over. The SUSTAIN differential
is not rectified at all, and removing a guard there changed nothing either —
because it provably cannot go negative: `slowRelease` shares `fast`'s attack
coefficient and releases more slowly, so its detector state dominates at every
sample, by induction and bit-exactly, since IEEE rounding is monotone. That one
was not a surviving mutation, it was **dead code**, and the fix is the opposite
of the first: the guard was deleted and the invariant written down. A guard that
can never fire tells the next reader the quantity can go negative when it
cannot.

Both were replaced in the mutation list by the thing each was imagined to
protect against, and both are now caught.

### The crossover spacing that binds on one input nobody had written

`MultibandCompressor` forces two crossover corners written at the same frequency
apart by 5%. Removing it survived all 1589 checks, and the first attempt to
measure its effect produced results identical to four decimal places — because
the probe compared corners of `(1000, 1000)` against `(1000, 1050)`, and the
repair normalises the first into the second. **The probe was measuring the same
configuration twice.**

Measured properly, against a build with the spacing removed, it is worth
0.35–1.3 dB: with the mid band at +12 dB makeup, 1030 Hz reads **+5.64 dB**
spaced and **+4.77 dB** unspaced, because two stages at the same corner are the
same filter twice and the band between them is the product of a low-pass and its
own complement rather than a band. The test now writes both corners at 1 kHz,
which is the only input where the constant does anything.

### Sorting crossovers and pushing them apart are not the same repair

Out-of-order corners will arrive — the caller is a language model — and
reconstruction cannot tell the two candidate repairs apart, because the
complementary subtraction is an identity that holds for any filter placed
anywhere. Both null at −319 dB.

They are not equivalent musically. Pushing each corner above the previous one
turns `(6000, 200)` into `(6000, 6300)`: the mid band becomes a 300 Hz sliver
and everything from 200 Hz to 6 kHz is silently handled by the LOW band's
threshold, ratio and timing. Sorting turns it into `(200, 6000)`, which is what
was written. A probe at 1 kHz is the only place the two differ, and it differs
by the whole band gain — 6 dB.

The first implementation pushed. **A repair that keeps every measurable
invariant can still be the wrong repair**, and the way to tell is to ask what
the parameters the caller wrote are now attached to.

### The transient designer's steady-state limit is real, and smaller than expected

A broadband envelope detector cannot separate "the envelope is moving" from "the
waveform is moving" once a period approaches the fast envelope's release, so a
sustained tone is gain-modulated at twice its own frequency. The expectation was
that this would be bad enough to need a fix — a detector high-pass, or a
mean-square envelope.

Measured with both controls at +12 dB: a steady 1 kHz tone moves **0.033 dB**
and a steady 60 Hz tone **0.59 dB**. The reason it is that small is the
decoupled detector's release branch, which holds the peak between cycles instead
of following the rectified waveform down; the ripple that survives is what the
30 ms release leaks. 0.59 dB on sustained bass is still audible enough to route
around, so it is documented and the contract test asserts it as a ratio to the
1 kHz figure rather than as an absolute — the shape of the limit is the claim,
not the number.

### The rectifier, not the gain, was what aliased

`transient_designer` measured **−66.4 dB** in-band ASR against a −100 dB
requirement — 34 dB short, and by a distance the worst node in the engine. The
obvious diagnosis is the one the plan's guidance suggests: a gain that changes
fast has broadband content, so multiplying by it at the base rate folds
everything above Nyquist. That is true and it is not the dominant term.

The dominant term is `abs()`. A full-wave rectifier has a corner at every zero
crossing, so a tone at f0 comes out with an infinite series of even harmonics
falling only as 1/m² — and the ones past Nyquist fold **before any envelope
filter exists to remove them**. The evidence is in which sweep steps were worst:
3202 Hz and 9003 Hz, and 96000/3202 = 29.98 while 96000/9003 = 10.66. Both are
frequencies whose 30th and 32nd harmonics land within ~90 Hz of DC after
folding, where the envelope filter's response is 0 dB and it modulates the
entire signal. The good steps are the ones whose nearest fold lands hundreds of
Hz up, where the filter attenuates it.

Once the mechanism is the rectifier, the fix has to be the rate, and the
measured curve is the 1/m² law in disguise — doubling the rate doubles the order
of the lowest-folding harmonic, which is 12 dB:

| 1× | 4× | 8× | 16× |
|---|---|---|---|
| −66.4 dB | −85.0 dB | −101.8 dB | −104.7 dB |

**And the second guess was wrong too.** `findings.md` already records that for
`drive` the oversampling FILTER's stopband is the binding constraint rather than
the ratio — mutating it from 120 dB to 40 dB costs 32 dB of ASR. The natural
next move was therefore to buy the last few dB with a better filter instead of a
higher rate. Sweeping the halfband stopband 120 → 150 → 170 dB changed the
result by **0.00 dB at both 4× and 8×**, to every digit printed. The two nodes
fail for different reasons and the lever that works on one does nothing on the
other.

### ADAA2 applied to a linear function is exactly a three-point moving average

The largest surprise from building `tape` and `console`, found because
`console`'s frequency response came out 0.31 dB low at 20 kHz against an
expectation computed by hand — and the expectation was right.

Substituting f(x) = x, so F2(x) = x³/6, into the engine's ADAA2 operator:

```
(F2(a) - F2(b)) / (a - b) = (a² + ab + b²) / 6
```

and the outer difference factorises as `(x0 - x2)(x0 + x1 + x2)/6`, cancelling
the outer denominator exactly. The operator collapses to `(x0 + x1 + x2)/3` — a
rectangular three-tap FIR at the OVERSAMPLED rate, whose magnitude is
`|1 + 2cos(2π f / (factor·fs))| / 3`:

| at 20 kHz, 96 kHz base | 4× | 8× | 16× |
|---|---|---|---|
| droop | **−0.313 dB** | −0.078 | −0.019 |

This is a real high-frequency loss on **every ADAA node in the engine**,
including `drive`, and it has never been written down. It was invisible there
because a saturator's high-frequency response is not something anyone had a
prediction for; it became visible the moment a node claimed a specific
sub-decibel roll-off. At 4× it is larger than the entire frequency-dependent
shift `console` exists to produce, and it corrupted the gap-loss curve `tape`
claims to reproduce — measured −2.089 dB at 20 kHz against the sinc's −1.777.

**The fix is topological, not numerical.** Both nodes now compute `y = u + g(u)`
with `g = f − identity`, send only `g` through the oversampler and the ADAA
operator, and delay the linear term by exactly the same integer number of
base-rate samples. The round trip's errors then scale with the node's DEPARTURE
from unity instead of with the signal: a 0.31 dB tilt on a residual sitting
60 dB down is 0.0003 dB on the output. Aliasing is unaffected, because
everything that could fold is in the residual and it is still oversampled and
still antialiased — measured, `console`'s in-band ASR went from −131.9 dB to
−173.9 and `tape`'s +12 dB case from −94.6 to −96.2.

Two things fell out of writing it. The residual has to be written out in closed
form rather than computed as `f(x) - x`, because for |u| ≪ 1 the subtraction
discards every significant bit and keeps the round-off — and a console stage at
24 dB of headroom operates at exactly that amplitude, so the regime where the
cancellation is worst is the regime the node lives in. And `drive` is left
alone: 0.31 dB of high-frequency softening on a saturator is a colour rather
than an error, and changing it would move every `drive` fingerprint in the
measurement baseline for no stated benefit. It is recorded here so the next
person to predict that node's response has the number.

### The textbook bilinear transform is the wrong realisation for a pole above the band

Both new nodes model analog first-order poles that sit ABOVE the audio band — a
tape head's gap-loss knee at 28 kHz, a summing amplifier's gain-bandwidth
roll-off at 60 kHz — and that is precisely where the standard prewarp stops
describing the prototype. The bilinear map compresses the whole infinite analog
axis into [0, Nyquist), so a corner near Nyquist drags the in-band response down
with it. Measured at the 96 kHz internal rate, a 40 kHz analog corner read at
20 kHz:

| | at 20 kHz |
|---|---|
| analog prototype | −0.969 dB |
| bilinear, prewarped at the corner | **−0.180 dB** |
| prewarped at 20 kHz and scaled | −0.969 dB |

Five times too little, in the one part of the band the parameter exists to
shape. `console`'s 60 kHz pole is above the internal Nyquist and the textbook
method cannot place it at all — `tan(π f/fs)` has to be clamped, and the
roll-off very nearly disappears: measured −0.0002 dB at 20 kHz against the
prototype's −0.458.

`dsp/AnalogOnePole.h` prewarps at `min(fc, 20 kHz)` instead, which degenerates
to the textbook case when the corner is in the band. One degree of freedom can
only be exact at one frequency, so the residual error is zero at 20 kHz and at
DC and bulges in between: 0.026 dB and 1.07° worst for `console`, 0.124 dB for
`tape`'s gap loss. Putting the matched point at the top of the band puts the
error where the pole does least.

### Wow and flutter cannot be measured by anything in the measurement suite

`tape` defaults its transport modulation to zero, and the reason is worth
stating because "the default makes the number look good" is the obvious and
wrong reading of it.

Every gated measurement in `engine/tests/measurements.cpp` is a coherently
sampled tone. The whole method rests on a periodic input producing a periodic
output, so every harmonic — legitimate or folded — lands in exactly one bin with
zero leakage. Pitch modulation is exactly what breaks that. At 19.5 kHz a 0.05%
peak speed deviation moves the phase by tens of radians within one 32768-sample
analysis window, so the fundamental smears across many bins, the denominator of
the ratio collapses, and the number that comes out describes the analysis window
rather than the node. To be below −120 dB of non-periodic residual the delay
would have to be stationary to about 1.6e−6 samples, which is not a tape
machine.

The suite already records the same effect from the other side: `chorus` and
`plate_reverb` measure −74 and −85 dB of "distortion" on a stimulus neither can
possibly distort, and it correctly exempts them from the aliasing requirement
because they are LINEAR and time-varying. The modulation here is the same thing
— a linear time-varying delay generates no harmonics and has nothing to fold —
so measuring this node's aliasing with the transport perfect is not a dodge. It
is the only experiment that isolates the term the requirement is about.

What the contract suite does instead is measure the modulation on its own terms:
bit-exact across six block sizes, independently seeded per channel, and the
achieved speed deviation read out of the audio by demodulating a probe tone
rather than out of the node's own state.

### Mutation results for `tape` and `console`

Twenty-three semantic mutations across `Tape.h`, `Console.h`, `AnalogOnePole.h`
and the shaper in `Adaa.h`. **Twenty of the first twenty-one were caught on the
first pass**, which breaks the streak this document opens with — but the one
that survived is the same species as every other survivor recorded here, so the
streak did not really break.

| mutation | caught | by |
|---|---|---|
| curve's cubic term −2/3 → −0.7 | yes | h3/h1 at both amplitudes |
| curve's quintic term u⁵/5 → u⁵/4 | yes | h3, h5, and a seventh harmonic appearing |
| bias offset no longer subtracted | yes | silence in → silence out, and the latency centroid |
| bias never reaches the shaper | yes | second harmonic absent |
| head bump Q halved | yes | the bump stops being local (0.93 dB two octaves up) |
| head bump gain 5% high | yes | 3.15 dB against the 3.0 asked for |
| gap loss at the sinc's first null, not its −3 dB point | yes | corner at 14.1 kHz where the formula says 7.0 |
| gap loss stops depending on speed | yes | the speed-doubling ratio, and the corner at both speeds |
| gap loss matched at 10 kHz instead of 20 | yes | 0.39 dB from the analog prototype |
| textbook prewarp at the corner | yes | nine checks across both nodes |
| Catmull-Rom slope bound 3.5 → 2.5 | yes | flutter exceeds the speed deviation asked for |
| spline amplitude bound 1.25 → 1.0 | yes | declared base delay 10 instead of 11 |
| knot spacing 1/(2·rate) → 1/rate | yes | peak drift past the overshoot bound, and latency |
| drift evaluated once per block and held | yes | block-size invariance at 1, 7, 64, 1023 |
| PRNG stops being keyed on the channel | yes | the two channels stop drifting independently |
| the whole signal back through the oversampler | yes | console's magnitude at 10 and 20 kHz |
| base delay not declared as latency | yes | the impulse peak, the centroid, and the declared value |
| console knee half as far from 0 dBFS as stated | yes | d3 at all four headrooms |
| console coupling high-pass removed | yes | magnitude and phase at 100 Hz |
| tape oversampling floor 8× → 4× | yes | in-band ASR at +12 and +18 dB of drive |
| **`drive_db` allowed past +18 dB** | **no** | see below |
| console `drive_db` allowed past ±12 dB | yes* | the clamp checks added for the survivor |
| console `headroom_db` floor 6 dB → 2 dB | yes* | the same |

\* added after the survivor, not part of the first pass.

**The survivor is a defence probed only from inside its own range.**
`tape`'s `drive_db` is bounded above at +18 dB *because* that is where 8×
oversampling still meets the −100 dB requirement — at +24 dB it measures exactly
−100.0. Widening the bound to +30 dB walks the node straight out of its own
headline claim, and all 1943 checks passed, because every test in the suite asks
for a value inside the bound. `findings.md` already contains this exact lesson
twice, under the Kaiser window and the crossover spacing: **a defensive measure
has to be tested at the parameter value where it binds, and a bound only binds
from outside.**

The fix is bit-exactness rather than another measurement: asking for more than
the bound must return exactly what asking for the bound returns. Four such
checks were added — the two drive clamps, the headroom floor, and the
oversampling floor — and each catches its own mutation and nothing else's.

### ASan caught an off-by-one in the test, not in the node

`harmonicRatios(..., highest)` returns one entry per harmonic starting at the
fundamental, so harmonic m is at index m−1. The check that the quintic produces
no seventh harmonic read index 6 from a six-entry vector. It passed at −O2 —
the read landed inside the vector's allocated capacity and returned a plausible
small number, which is exactly the shape of "benign-looking garbage that passes
every functional assertion" this repository built `make san` for. The
sanitizer build stopped on it immediately.

### A smoother curve did not alias less

The quintic saturation curve is C2 at its knee where `drive`'s cubic is C1, so
its harmonic series falls as 1/n⁴ rather than 1/n³. The obvious prediction is
that it aliases less at the same oversampling factor and the same drive.

Measured at 4× with a 19.5 kHz probe at 0 dBFS, in-band ASR:

| | +12 dB drive |
|---|---|
| `drive`, cubic soft clip | −100.4 dB |
| `tape`, quintic | **−96.2 dB** |

Four decibels worse. The continuity argument is about the harmonic series far
above the knee and it is true; what it leaves out is that the two curves are not
at the same operating point for the same `drive_db`. The quintic's incremental
gain is `(1 − u²)²`, already −5.0 dB at half scale where the cubic is −2.5, so
at any given drive setting it is producing considerably more distortion to begin
with. A tail that decays faster from a larger starting value is not necessarily
smaller. `tape` runs at 8× for that reason and the factor was chosen from the
measured table rather than from the argument.

### Oversampling a multiband is nearly free, and oversampling a transient
designer is not

Same requirement, same fix, wildly different returns. `multiband_compressor`
went from −98.6 dB to **−152.0 dB at 4×**, and 8× bought nothing further.
`transient_designer` needed 8× to reach −101.8 dB.

The difference is how much of the folded spectrum each node's detector passes.
The multiband's fastest ballistic is a 5 ms attack — a 32 Hz corner, which
attenuates a fold landing at a few kHz by 40 dB or more. The transient
designer's reference envelope is 0.5 ms, a 318 Hz corner, and passes the same
component at −26 dB. Measured directly on the multiband: at factor 1, dropping
every band's attack from 5 ms to 0.1 ms costs **7.8 dB of ASR** (−98.6 →
−90.8), and at 4× the same change costs **0.0 dB**, because the fold has moved
above the audio band and the decimator removes it regardless.

**The general shape:** the aliasing penalty of an envelope-driven gain is set by
the detector's own bandwidth, not by how nonlinear the gain law looks. A fast
detector is a wide-open door to whatever folded.

### Oversampling a node with an exact identity has to be opt-out, not opt-in

Both nodes have a configuration in which they are provably a pure delay:
`multiband_compressor` with every band at 1:1, `transient_designer` with both
controls at zero. In both cases the gain is a constant — exactly 1.0 — and
multiplying by a constant is linear, so there is nothing for an oversampler to
prevent.

Wrapping the whole node unconditionally would have traded the multiband's
**−319 dB** unity null for the resampler's round trip, which is around −140 dB.
That is a 180 dB regression in the property the node's whole crossover design
exists to deliver, bought to fix aliasing in a configuration that has none.

The multiband resolves it without a second code path: the effective factor is 1
unless some band's ratio exceeds 1, and an `Oversampler` at factor 1 is a pair
of `std::copy` calls with zero declared latency. Same code, bit-exact result.
The transient designer does take a branch, because with nothing to shape there
is no reason to run the detectors at all — and that branch is the same rule as a
bandless EQ declaring no latency, not a special case.

### An oversampled compressor is not the same compressor

Oversampling `multiband_compressor` was expected to change its output by about
the resampler's own error — a round trip through halfband filters designed to
120 dB. Measured against the identical settings at the base rate, a single band
differs by **−31 dB**, which is four orders larger.

It is not an error. At 4× the detector reads an *upsampled* signal, and the
reconstructed inter-sample peaks of band-limited audio are genuinely higher than
the sample peaks the base-rate detector saw — the same fact that makes a
true-peak meter read above a sample-peak meter. The oversampled node therefore
compresses slightly harder, and is the more correct of the two.

Worth stating because it sets what a test can claim. "One band is bit-identical
to `compressor`" is true and worth asserting, but only at `oversample_factor` 1;
at 4× the same assertion would have to be loosened to −31 dB, at which point it
is no longer testing the identity it was written for. The check stays pinned at
factor 1 and the behavioural difference is printed rather than asserted away.

### The bed's own onset read exactly like lookahead

The check that `transient_designer`'s lookahead makes the boost arrive early
measures the four milliseconds before a hit, over a quiet tone bed, and expected
nothing there with lookahead at zero. It read **0.186 dB**.

Not a bug in the node. The bed started at sample 0 and the hit was 85 ms later,
and with `attack_ms` at 20 the slow envelope needs about five time constants to
catch up — so the window still carried the BED's own attack. Moving the hit to
340 ms into the buffer took it to 0.0001 dB. A measurement artifact that
shrinks when the measurement improves, which is the same signature as the
zero-phase EQ's analysis-window case above, with the sign the right way round.

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

**Fixed, 4 August 2026, and the obvious fix was the wrong one.** The reflex is a
shuffled-baseline control: score a random permutation of the same prediction and
report the excess. That does nothing. Permutations come in reversing pairs and
reversing a permutation negates its rank correlation, so **the shuffled control
is identically zero** and excess-over-shuffled is algebraically the raw score
again. Measured, not argued: it comes out at exactly 0.000 on every corpus, and
a test now asserts it to 1e−12 so the next person to propose it can see the
number.

What does work is two things, and the harness does both:

- **The corpus is mirrored.** Every chain scored for ordering is generated
  together with its exact reverse, sharing the same parameter draws, so a
  fixed-sequence predictor scores +ρ on one and −ρ on the other. The mirror only
  holds when the chain's node types are all distinct — reversing "eq, comp, eq"
  gives "eq, comp, eq" back — so ordering-eligible chains are built from distinct
  types and everything else is excluded from the metric outright.
- **An order-blind control runs beside the score.** The same predicted multiset,
  re-emitted in a fixed conventional sequence. Its precision and recall are
  identical by construction, so the difference between the two is ordering
  information and nothing else. On a *balanced* corpus it agrees with the raw
  score; on an unbalanced one it is the only one of the three that is honest.

The rule-based baseline now scores **+0.000 excess** on the mirrored chains,
which is what +0.498 should have read all along. The raw figure comes out at
+0.000 too, which is the corpus balance working.

### A harness whose ceiling nobody could raise

The corpus deliberately contains moves nothing can recover — a 0.2 dB shelf, a
1.05:1 compressor, a 20 Hz high-pass. Scoring those as misses puts a ceiling
below 1.0 into the harness for reasons no planner can address, and **a number
that cannot improve is a number nobody acts on**. They are excluded from the
headline instead, and counted on their own line.

Excluding them is not free either, and the first version got it wrong in the
other direction: cancelling *every* prediction of an ignored type meant a chain
with one loud EQ and one faint one scored zero for naming a single EQ — a
correct answer marked wrong by the fairness adjustment. Only a *surplus*
prediction is cancelled now.

The floors are read from `corpus.PERTURBATIONS` rather than written down: the
smallest amount of each move the sensitivity sweep ever applied is the smallest
amount anything here has evidence is detectable. Writing the numbers by hand
would have been inventing a detection threshold, which is the same class of
mistake as substituting a plausible constant for a missing measurement.

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

## Findings from mutation-testing the Python stages

The C++ nodes had all been mutation-tested; the Python measurement, verify and
plan stages never had. Seventy-two semantic mutations across
`backend/verify/gate.py`, `backend/measure/loudness.py` and
`backend/measure/gating.py`:

| | mutations | survived first pass | survived after strengthening |
|---|---|---|---|
| `verify/gate.py` | 26 | **17** | 1 |
| `measure/loudness.py` | 35 | **18** | 2 |
| `measure/gating.py` | 11 | **7** | 1 |

**Forty-two of seventy-two survived.** The four remaining are unkillable rather
than untested: three are unreachable branches and one is an oversampling factor
with no measurable effect. All four are in `open-questions.md`.

The pattern across all three modules is the same, and it is not the pattern the
C++ suites had. Nearly every check *existed* and nearly every check *fired on
the case it was given*. What was missing was the threshold: the tests probed
each check two to eight times further past its bar than the bar itself, so the
constant could move by half its value and nothing noticed.

### Every threshold in the safety gate was free to move

The gate had 36 checks and read as thorough. It tested corruption with a NaN,
mono incompatibility with a polarity flip, true peak with audio 3.3 dB over the
ceiling, a level collapse at 20 dB against a 12 dB bar, and a bandwidth collapse
at 26 dB/octave against an 18 dB/octave bar. Every one of those is the extreme
case. Measured margins:

| check | bar | what the test used | margin |
|---|---|---|---|
| true peak | ceiling + 0.1 dB | +3.3 dB over | 33x |
| correlation | −0.20 | −1.00 | the whole range |
| level collapse | 12 dB | 20 dB | 1.7x |
| bandwidth | 18 dB/oct | 26 dB/oct | 1.4x |
| loudness target | 0.5 LU | 3.0 LU | 6x |
| silence | −80 dBFS | digital zero | infinite |

The silence case is the sharpest example. An all-zero render has *no measurable
peak*, so it fails on the `peak is None` branch and never reaches the threshold
at all — the `-80.0` could have been `-200.0`, and was, and the suite passed.

The fix throughout is to probe each bar from both sides at a plausible distance:
a render 0.5 dB over its ceiling and one 0.05 dB over (inside the meter slack), a
correlation of −0.30 and one of −0.15, a 14 dB collapse and a 10 dB one. The gate
suite went from 36 checks to 66 and from 9 mutations caught to 25.

### Two gate checks were the same check written twice

`MIN_CORRELATION = -0.2` and `MAX_MONO_LOSS_DB = 1.0` are the same bar: mono loss
is `-10*log10(1 + r*k)` with `k <= 1`, which at `r = -0.2` is 0.969 dB. Since the
mono-loss test is an `elif` after the correlation one, it can never fire. See
`open-questions.md`.

### EBU case 4 does not test what its comment says

`test_loudness.py` case 4 is documented as testing the absolute gate
independently — "a meter with only the relative gate passes case 3 and fails
here". Measured: **deleting the absolute gate entirely changes cases 1, 3, 4 and
5 by nothing at all.** The quiet sections are 49 LU below the programme and the
relative gate at −10 LU discards them on its own.

The absolute gate decides exactly one thing, and nothing in the suite exercised
it: a file with **no programme in it**. A stem that is nothing but noise floor
has no loudness to report, and the relative gate cannot say so, because it
measures everything against whatever it finds. A 20 s tone at −72 LUFS must read
`-inf`; at −68 LUFS it must read −68.

### "Validated against published ground truth" was not checked against it

`loudness.py`'s docstring says its analog parameters "reproduce the published
48 kHz table exactly". Nothing compared them. Every other assertion in the suite
is either a 1 kHz reading — where the RLB high-pass is flat, and its corner
moved from 38 Hz to 60 Hz undetected — or a comparison of the filter against
*itself* at another sample rate, which a wrong filter passes as readily as a
right one.

They do match, to **8.9e−16**. That is now asserted against the coefficients as
published rather than against the derivation that produced them.

### A tolerance two hundred times looser than the measurement

The true-peak analytic case asserted 0 dBTP within 0.05 dB on a meter that
`findings.md` records as accurate to 0.0001 dB. Inside that slack, narrowing the
interpolator's passband from 0.95 to 0.85 of the corner and cutting its taps from
48 per phase to 12 both passed. Tightened to 0.002 dB, still a twenty-fold
margin over the worst reading.

**A tolerance is an assertion about the implementation, not only about the
standard.** Set at the standard's tolerance when the implementation is orders of
magnitude better, it stops constraining the implementation at all.

### Coverage is not a mask

`gating.gated_frame_mask` was checked with `0.15 < coverage < 0.60`. A range that
wide is satisfied by a mask that keeps the wrong 35% of the file — and it was:
marking only the 100 ms step of each surviving block instead of its full 400 ms
passed, and so did requiring *all four* covering blocks to survive instead of any
one, which is the specific rule the module's docstring exists to justify. The
test now asserts which frames: each phrase kept whole, the 300 ms either side
kept with it, and the gaps empty.

### The percentiles in loudness range were never the measurement

Every EBU Tech 3342 case is two flat plateaus, whose 95th and 10th percentiles
sit *on* the plateaus — and so do the maximum and the minimum. Replacing the
percentiles with extremes passed all four published cases and the steady-sine
case.

A linear ramp in dB is the distribution where the choice has an arithmetic
consequence: short-term loudness becomes roughly uniform, so the 95th minus the
10th is 0.85 of the swept range against 1.00 for max minus min. A 20 dB ramp
must read 17 LU, not 20.

## Findings from mutation-testing the planner's validators and the refine stage

The second round of the same exercise, on the three modules the first round
deferred because the schema was being edited underneath them.

| | mutations | survived first pass | survived after | checks |
|---|---|---|---|---|
| `plan/validate.py` | 62 | **43** | 3 | 203 → 280 |
| `refine/match_eq.py` | 45 | **23** | 3 | 68 → 116 |
| `refine/match_dynamics.py` | 32 | **15** | 3 | 49 → 66 |

The shape of the failure is the one the first round predicted and the rate is
worse: **81 of 139 survived.** Again almost every check existed and fired on the
case it was given, and again the missing thing was the threshold. The nine that
remain are unreachable or behaviour-neutral rather than untested, and all nine
are in `open-questions.md`.

Two patterns are new. Four checks were not merely untested but **never asserted
to exist at all** — the delivery ceiling, the bypassed-node finding, the dither
word length and the two-band cascade could each have been deleted outright
without a failure. And three constants were probed only through a wrapper that
overrides them: `fit_bands`'s `max_q`, `match_to_curve`'s smoothing and
`threshold_for_duty_cycle`'s duty cycle were each passed explicitly by every
test, so the values production actually uses were never exercised. **A default
that every test overrides is not a tested default.**

Three of the survivors were real defects rather than weak tests, and all three
are the same species: **a mechanism that reported success without delivering
it.**

### The ringing repair did not reach the bar it had just named

`_check_filter_stability` rejects a pole radius above 0.9995 and repairs it by
setting `Q = q / 4`. A fixed divisor does not know how far over the bar it
started, and the case in the function's own docstring is the case where it is not
enough:

| | Q | pole radius | over 0.9995? |
|---|---|---|---|
| as emitted | 12 | 0.999940 | yes |
| after `q / 4` | 3 | 0.999760 | **still yes** |
| solving for the bar | 1.43 | 0.999497 | no |

At 22 Hz the pole has to be pulled to Q 1.44 to clear the bar, and the repair
stopped at 3. It logged *"reduced to Q 3"* against a message saying Q 12 "rings
audibly", so the graph carried a repair note asserting it was now safe and was
not — which is worse than not repairing, because the note is what anyone reading
the log would trust.

The repair now inverts the pole-radius expression for the widest Q that sits on
the bar and rounds it down. The old test asserted only `q_factor < 12.0`, which
is true of every wrong answer including this one.

### The comment said "not the sum" and the code summed

`_check_gain_staging` projected an EQ node's contribution as

```python
projected = max(projected, projected + gain) if gain > 0 else projected
```

which for a positive gain is `projected += gain`, evaluated once per band — a
sum. Two lines below it sat the comment *"Only the largest boost, not the sum:
bands at different frequencies do not add at any single frequency"*, which is
both correct and the opposite of what the code did.

Nothing exercised it: `test_gain_staging` drove the check with a gain node and a
compressor and never with an EQ at all. Three 4 dB bands at 200 Hz, 1.5 kHz and
8 kHz were projected as a 12 dB peak that occurs at no frequency, so the check
was failing chains that are fine. Now the largest boost per node, summed across
nodes — because two EQs lifting the same region really do stack.

### The dither word length was repaired only when a measurement report existed

The dither check reads nothing from the report. It sat below
`_check_source_constraints`'s `if not report: return`, and `Project.excerpt_report`
defaults to `{}`, which is falsy — so a graph validated before the measurement
landed shipped a 20-bit dither untouched. Moved above the early return. There was
no test for the check in either direction.

### An N-point Hann window is an (N−2)-point window

`smooth_log` floors its window at 3 points to guarantee some smoothing.
`np.hanning(3)` is `[0, 1, 0]`: normalised, it is the identity. Asking for 1/12
octave or finer on a 256-point grid therefore returned the curve **completely
unsmoothed**, and the floor that existed to prevent exactly that was the thing
producing it.

The existing test compared a 1/1-octave smooth against a 1/12-octave one and
asserted the first was wider — true, and equally true when the second does
nothing. Floor raised to 5, and the test now asserts that a fine setting still
changes the curve rather than that it changes it less.

### Tolerances four to thirty times looser than the fit

`test_refine.py` allowed the recovered band to sit 0.75 octaves from the applied
one and to carry "a meaningful fraction" of its gain, defined as 40%. Measured,
the fit lands on **800.0 Hz for a filter applied at 800 Hz** and returns 4.8 dB
for 5.0:

| | asserted | measured | margin |
|---|---|---|---|
| centre frequency | 0.75 octaves | 0.000–0.005 | 150x |
| gain | 40% of applied | 96% | 5x |
| band count | at most 5 | 1 | untested |
| filter shape | not asserted | correct | — |

Inside that slack the smoothing could run at 1/1 octave, the shelf/bell boundary
could move by a factor of four, the cascade could return five bands instead of
one, and `choose_cascade` could be replaced by "always take the largest" — the
property `findings.md` records as the reason it exists. All of them passed.

### The dynamics timing was asserted only against itself

`test_match_dynamics.py` checked that a slowly-moving source gets a longer
release than a fast one and that the attack is shorter than the release. Both
are comparisons, and both survive scaling the whole derivation by any constant:
the release factor could be 0.2 instead of 0.5 and the attack 50% of the release
instead of 8%, and every assertion held. Measured, a source whose envelope
centroid sits at 4.6 Hz gets 109 ms and one at 11 Hz gets 45 ms, and the
attack/release ratio is 0.0800 exactly. Those are now the assertions.

The same shape appeared in the reachability report. `achievable_reduction_db`
was checked only as "less than the 15 dB that was asked for" — true of every
wrong answer, including probing at 8:1 instead of 20:1, which understates it by
0.7 dB. It is now compared against a render made at the top of the ratio range,
and the unachievable bar is probed 0.7 dB either side of what the material
actually yields instead of 6 dB past it.

### A confidence that is a flag wearing a float's clothes

`match` discards the derived time constants when
`modulation_summary`'s confidence is under 0.4. That function returns **0.8 or
0.0 and nothing in between** — 0.0 exactly when it measured nothing, which is
also exactly when `timing_from_source_rate` has already returned the 15/150 ms
fallback. So the gate can only fire on material that is already at the fallback,
and the assignment inside it always writes the values that are there. Raising
the gate to 0.9 is a real change (every render would silently use the default
timing, which nothing caught until now); lowering it is unobservable. See
`open-questions.md`.

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

## Findings from the per-commit DSP measurement suite

`engine/tests/measurements.cpp`. Five of the seven bugs below were in the
MEASUREMENT, not in the engine, and every one of them produced a plausible
number that a reader would have believed.

### The analysis window sat at the end of the stimulus

The first version rendered `settle + N` samples of tone and read the last `N`.
That is findings.md's own "measuring the wrong window", and it cost real
numbers: `zero_phase_eq` measured **−89 dB THD** and `chorus` **−74 dB**, on two
nodes physically incapable of producing a harmonic. Both reach BACKWARDS — a
filtfilt pass by up to its 24 000-frame reach, a lookahead limiter by its
window — so the tail of that window is the response decaying into the silence
past the signal. Every measurement now keeps 32 768 samples of live tone after
the window it reads, which is more than the largest backwards reach in the
engine.

### Homogeneity alone is not linearity

Which nodes are subject to the aliasing requirement is decided by measurement
rather than by a list, so that a new node cannot quietly escape it. The first
test was homogeneity: does `f(2x) == 2f(x)`?

`transient_designer` passes that **exactly**. Its gain is a ratio of a fast
envelope to a slow one, which is scale invariant, so it is perfectly homogeneous
and thoroughly nonlinear. It was classified "linear", excused from the aliasing
requirement, and reported as fine. Adding superposition — `f(x+y) == f(x)+f(y)`
— fails it by 60 dB and it then measures **−66 dB ASR against a −100 dB
requirement**, the single worst result in the suite.

The converse matters as much: `chorus` and `plate_reverb` are time-VARYING but
linear. They satisfy both properties exactly and are correctly left out of a
requirement that means nothing for them.

### An exactly-zero null was discarded as unmeasurable

`polarity`, `mono_to_stereo` and `stereo_width` null against themselves to the
last bit. The deviation is exactly zero, `log(0)` is not a number, the code
treated it as "could not measure", and all three were then classified
**nonlinear** and put under the aliasing requirement.

Substituting −320 dB instead would have been the `linToDb` clamp bug one row up
in this document. A measured value carries three states here, not two: absent,
present, and *exactly zero* — which serialises as `null` with an
`exactly_zero` flag, so nothing downstream can read it as "0 dB of error".

### The settle time was guessed, and the compressor was still moving

With a fixed 0.34 s settle the compressor reported **−78 dB of in-band
aliasing**. Almost none of it was aliasing. Its program release converges over
*seconds*: at 0.34 s the residual drift sat at −48 dB relative to the
fundamental, one bin either side of the tone, and its rectangular-window skirts
leaked down into the alias bins. At a 2.7 s settle the off-locus energy is
−130 dB and the same node measures **−104 dB**, which is a pass.

The settle is now measured rather than chosen. A deterministic node fed an
exactly periodic tone produces an exactly periodic output, so energy off the
harmonic loci is either noise or an unfinished transient; the probe lengthens
the settle until that residual is 20 dB below the requirement, and stops early
when a longer settle stops helping — which is what distinguishes a transient
(shrinks) from `dither`'s noise floor (does not).

### The periodicity residual counted the aliasing as evidence of itself

The first version of that convergence test defined "off the harmonic loci" as
"not a multiple of the tone's bin". **A folded harmonic is not a multiple of the
tone's bin** — that is what folding means. So the residual included the alias
energy, and for four different nodes the reported residual equalled the reported
ASR to two decimal places and said nothing at all.

### The Farina pre-window truncated the response it was measuring

Standard practice is to window off everything before the deconvolved linear
impulse, because the harmonic products land there. Implemented at 4096 samples,
which comfortably clears the second harmonic's arrival 5980 samples early.

The linear impulse of a sweep starting at 20 Hz is **band-limited, and therefore
about 50 ms wide**: still −13 dB one thousand samples from its peak and −35 dB
at four thousand. The window was cutting through the middle of it. `dc_block`
measured ±3 dB of magnitude ripple and 20° of phase where the correct answers
are 0.000 dB and 0.81°, and `linear_phase_eq` read 0.12° where its header claims
7.3e−13.

There is no window that separates them at this sweep length. The pre-window was
removed, the stimulus is at −26 dBFS so the distortion products are small enough
to leave in, and the amount left in is *measured* rather than assumed. With it
gone, `linear_phase_eq` and `zero_phase_eq` both read **0.0000° at every bin**,
which is the number that makes the rest of the table trustworthy.

### Group delay by differencing two bins is 41 722× the phase noise

`gd = −Δφ/Δω` across adjacent bins divides by `2π/M`, which at M = 262 144
multiplies the phase error by 41 722. It reported **29 samples** of residual
group delay for `linear_phase_eq` and **1486** for `dc_block` at 1 kHz — both
pure noise, both exactly the shape a reader takes for a property of the filter.
Fitting a slope across 25 bins gives −0.00002 and 0.216 samples respectively.

### Mutation results

Nine mutations, seven caught. Two of the three initially missed were failures of
the *gate*, not of the measurement, and both changed how tolerances are set:

| mutation | caught | by |
|---|---|---|
| `drive`: oversampling removed (4× → 1×) | yes | ASR −135.7 → −100.4 dB |
| `drive`: oversampling filter 120 → 40 dB | yes | ASR −100.4 → −68.6 dB, three configs stop meeting the threshold |
| limiter: ceiling raised 5% (0.42 dB overshoot) | yes* | true-peak conformance |
| limiter: gain applied 2% high | yes* | true-peak conformance, 0.17 dB |
| `parametric_eq`: bell gain +0.6 dB | yes | frequency response, group delay |
| `dither`: noise removed | yes | ASR at worst level, IMD d3 |
| compressor: knee quadratic 5% wrong | yes* | gain transfer at −18 dBFS |
| `linear_phase_eq`: symmetry broken 0.1% | **no** | see below |
| `linear_phase_eq`: symmetry broken 1 ULP | **no** | see below |

\* not caught by the first version of the gate. Three separate reasons, each a
real defect in it:

**A 0.50 dB tolerance cannot police a delivery ceiling.** Raising the limiter's
ceiling by 0.42 dB — straight overshoot on a node whose whole contract is that
it does not overshoot — slipped under it. dBTP is now 0.05 dB, which is thirty
times the reference meter's own validated error.

**An ungated metric has no good direction.** A frequency response, a
gain-transfer level and a phase are *fingerprints*. The gate compared them
directionally with "lower is better", so the compressor's knee error — which
moves the output at −18 dBFS by −0.081 dB — was recorded as an **improvement**
and waved through. Any movement in a fingerprint is now a failure.

**One flat tolerance per unit is the wrong shape.** It has to be loose enough
for a −280 dB THD residue that is nothing but FFT round-off and tight enough for
a magnitude response that should be exact to six decimals, and those differ by
five orders of magnitude. The tolerance is now derived from where the value
sits: the engine bounds cross-platform sample error at −140 dBFS, so a figure at
V dB is perturbed by at most `20·log10(1 + 10^((−140 − V)/20))` — 0.086 dB at
the −100 dB gate, 9e−7 dB for a response near unity, and unbounded below
−160 dB, where the statement "this is not a measurement" is the honest one.

**The two that survived are honest survivors.** A 0.1% break in
`linear_phase_eq`'s kernel symmetry moves every acoustic quantity in the suite
by ≤ 0.009 dB and the phase at 1 kHz by 0.0029°, and a 1-ULP break moves them by
nothing. No cross-platform-safe tolerance can resolve that, and it should not
try: the contract suite asserts symmetry **on the kernel itself**, which is
where a bit-exactness claim belongs. A 5% break is caught, with 18 regressions.

---

## Findings from expanding the recovery harness

The corpus went from six hand-written templates to 67 generated chains from a
fixed seed, and the harness gained a per-type breakdown, parameter scoring and
two controls. The floor moved **F1 0.580 → 0.509** — the corpus got harder, not
the recogniser worse.

### The headline was hiding a node that is never detected

Averaged, the six templates said "F1 0.58". Broken out by type, the same
recogniser is at **recall 0.91 on compression and 0.14 on drive**, and drive's
precision is **1.00** — when it fires it is right, and it almost never fires. The
rule needs harmonics up top *and* flattened peaks, and on already-bright material
the first condition never arrives.

Two other facts the average could not carry: `stereo_width` is recalled perfectly
and proposed five times too often, so it is a precision problem rather than a
detection one; and `plate_reverb` fails in both directions, which is the
dense-material limitation showing up as a number instead of a paragraph.

A per-type table costs twenty lines and turns one unactionable number into three
different pieces of work.

### The generated corpus is harder in a way the templates could not be

Six hand-written chains cannot vary length, type and parameter extremity
independently, and with six of them one template gaining a node moves the
headline by four points. 67 chains at 1–5 nodes, drawn from four tiers of
extremity, with duplicates allowed, moved the floor down eleven points and made
it stable enough that the per-family numbers separate: `varied` 0.58,
`ordering_pair` 0.55, `single` 0.55, `faint` 0.32.

### Nearest-neighbour parameter matching flatters, and by how much

Matching a proposed band to the nearest applied one means a proposal placed
*anywhere* lands near something. The control — same proposals, frequencies
replaced by log-uniform guesses — scores **1.80 octaves** median error against
the recogniser's **0.27**, and 11% within a third of an octave against 52%. The
signal is real, and without the control there would be no way to know.

The gain control is blunter and more useful: predicting **0 dB for every band**
scores 8.80 dB median error against the recogniser's 1.22. A metric where "say
nothing moved" is the baseline needs that baseline printed next to it.

### Mutation results

Forty semantic mutations across `backend/eval/chains.py`, `recover.py` and
`generate.py`. **Thirty-one caught first pass, nine survived**, in line with the
42-of-72 and 81-of-139 the Python stages ran before. After strengthening,
**39 of 40**.

| survived first contact | why | fix |
|---|---|---|
| excess taken over the shuffled control | on a *balanced* corpus the two controls agree, so every test case was balanced | added a one-chain corpus, the smallest unbalanced one there is: raw +1, order-blind +1, shuffled 0, honest excess 0 |
| unmirrored chains enter the ordering mean | `score_recovery` already blanks those, so `aggregate` was never tested on its own | construct a `RecoveryScore` directly and assert `aggregate` still refuses it |
| bands matched by linear frequency distance | every test case put the right answer nearest in both metrics | 400 Hz and 1900 Hz against a 1 kHz truth — 400 is nearer in hertz, 1900 in octaves |
| the gain tolerance loosened 3 dB → 12 dB | probed at 0 dB error and at 4 dB, never at the bar | 2.9 dB inside, 3.1 dB outside |
| a boost and a cut merge into one run | the two moves in the test were *distant*, so the zeros between them separated them whatever the sign rule was | adjacent bands, +5 dB at 1 kHz and −5 dB at 1.25 kHz |
| recovered gain is the run's mean, not its peak | the gain was never asserted on a case with a skirt | a bell whose skirt averages 4.0 against a 6 dB peak |
| an unknown perturbation name yields a floor | the dictionary lookup two lines later raises the same `KeyError`, so asserting the type tested nothing | assert the message says there is no evidence for a floor |
| the manifest's recorded digest is not checked | every tampered manifest also failed the *seed* check, so one of the two checks was free | a manifest with only its digest field corrupted, which regenerates perfectly |

The pattern is the one this project keeps finding: **the check existed, the check
fired, and the test sat far enough past the bar that the constant could move by
half its value.** Two of the eight are a second variant — a test whose case was
constructed so that two different implementations give the same answer.

**M08 is an equivalent mutation, not a gap.** Reading the shared-node list off a
sorted set instead of the predicted sequence changes nothing on any input: the
rank correlation is taken over *paired* index lists, and permuting both the same
way leaves it unchanged. It is written down in the code so the next person to
notice does not go looking for the test that should have caught it.

---

## Findings from the blinded listening harness

`backend/listen/`. The first piece of this project whose subject is a person
rather than a signal, and the failure modes are correspondingly different: every
bug here produces data that looks like human judgement and is something else.

### Loudness matching moves the screening stimulus by six times the bar

The plan's listener screen is a 2 dB Q=1 bell at 3 kHz. It reads as a pure
frequency-discrimination test, and it is not: measured on pink noise, the bell
alone moves integrated loudness by **0.58 LU**, against a 0.1 LU matching bar.

Unmatched, a listener with no frequency discrimination at all passes the screen
by picking the louder clip every time. The screen would then select for exactly
the bias the rest of the harness exists to remove, and it would do so
invisibly — a listener who "passed" is thereafter trusted.

The general shape: **a stimulus designed to differ in one dimension usually
differs in others too, and level is the one that decides a blind comparison.**
Anything a listener chooses between goes through the matcher, including the
material used to decide who is allowed to listen.

### Verifying the match needs a mechanism where prediction and measurement differ

Applying `target − measured` dB is exact against K-weighted loudness, so a
verification that recomputes the prediction agrees with itself perfectly and
proves nothing. The first version of the test asserted that the matcher's
reported LUFS equalled an independent re-read of the written file — which is the
right assertion and, at 24 bits, is satisfied to about 1e−12 by a matcher that
never opened the file. **A tolerance that generous does not distinguish the two
implementations.**

The mechanism that does is BS.1770's **absolute** gate. It sits at −70 LUFS and
is not relative to the programme, so a group attenuated across it loses whole
blocks from the integration. A pair at −68 LUFS pushed to a nominal −73 is
arithmetically matched and measures `-inf` on both sides. That case is now the
test, and it kills the mutation that the tolerance-based one let through.

Two other mechanisms exist and were rejected as too weak to test with: 24-bit
quantisation moves loudness by around 4e−12 dB at working levels, and PCM
clamping requires a construction where one clip of a pair clips and the other
does not, which needs two different crest factors and is fragile.

### A refusal backed up by a second refusal is a refusal no test can falsify

Three mutations survived the first round for the same reason, and it is
`findings.md`'s own "a property with two implementations" in a new place:

- substituting `-70.0` for a non-finite loudness inside the gain solver was
  invisible, because the *written* clip is still silent and `verify_matched`
  refuses it after the fact;
- accepting duplicate listener ids was invisible, because two listeners with the
  same id derive the same random stream and are caught by the
  identifier-collision check further down.

In both cases the test asserted "something was raised" and was satisfied by the
wrong check, with the wrong message, after a fabricated gain had already been
computed. Fixed two ways: the solvers are now unit-tested directly, and
`R.raises` takes a `contains=` argument so a refusal has to be the refusal it
claims. **Asserting an exception type is not asserting a reason.**

### The post-write ceiling check could not fire, and was deleted

`match_group` computes a common offset of `min(0, ceiling − highest_true_peak)`
over the scaled arrays, so every clip is at or under the ceiling before it is
written, and quantising to PCM moves a true peak by well under a thousandth of a
decibel. A guard after the write therefore had no reachable input, and widening
its threshold from 0.05 dB to 6.0 dB passed every test.

Deleted rather than tested, on the same reasoning as `TransientDesigner`'s
sustain rectification: a guard that cannot fire tells the next reader that the
quantity can go over when it cannot. The invariant is asserted from outside
instead, and at a **non-default** ceiling as well as the default one — because
"every clip is under −1.0 dBTP" is also satisfied by a matcher that ignores its
`ceiling_dbtp` argument entirely, which was a separate surviving mutation until
the −12 dBTP case was added.

The identifier-collision check in `plan_session` is the same shape and was
**kept**, which is worth stating because the two decisions look inconsistent.
The ceiling guard protects an invariant provable from three lines above it; the
collision check protects an assumption about a random number generator, and the
failure it prevents is two listeners' answers silently merging.

### The mapping was checked against the map

The strongest blinding assertions in the suite — no condition name in the
payload, byte-identical payloads when the two conditions are written the other
way round — say nothing about whether the clip a trial *serves* is the condition
the key says it is. Swapping the two sides while building the clip map is a
silent, total inversion of every result the panel would produce, and it passed
everything.

Fixed by giving each condition an unmistakable spectral signature — a marker
tone at 1 kHz or 5 kHz — and reading the served file back to ask which condition
it is. That is the only assertion in the file that does not consult the
dictionary under test.

The same trick pays twice: the screening key's `bell_side` is verified by
measuring which clip carries the boost, so a key that recorded the wrong side
produces a negative delta and an argmax in the wrong band.

### An identifier is a key, never a path fragment

The server's traversal probes were `/clip/../key.json`, `/clip/../../key.json`
and friends, all of which 404 under a resolver that joins the identifier onto
the session directory — because the key is not *above* the clips, it is beside
them. The probe that matters needs no traversal at all: **`/clip/key.json`**.

A mutation replacing the map lookup with a filesystem join served the answer key
over HTTP and passed the entire suite. The probe list now starts with the
undotted case.

### The interval a three-person panel actually earns

Ratings are not independent observations, and a t-interval over pooled ratings
treats thirty ratings from three people as thirty draws. On the suite's fixture
the pooled half-width is 0.86 against 1.43 clustered by listener — a factor of
1.7 of invented precision, on the metric that decides whether the product is
good. The clustered interval spans about 2.9 points of a 5-point scale. That is
what three listeners buys, and a narrower number is a claim about an assumption.

### Mutation results

Ninety-five semantic mutations in the final set; the first round ran
ninety-two of them against the first version of the suite, and three more were
added afterwards for code the first round's fixes had changed.

| | mutations | survived first exposure | caught only as a crash | survived after |
|---|---|---|---|---|
| `match.py` | 21 | **5** | 4 | 0 |
| `session.py` | 20 | **3** | 2 | 1 |
| `screen.py` | 15 | 0 | 1 | 0 |
| `score.py` | 18 | 0 | 1 | 0 |
| `correlate.py` | 16 | 0 | 2 | 0 |
| `serve.py` | 5 | **2** | 0 | 0 |

**Ten of ninety-five survived first exposure**, against 42 of 72 and then 81 of
139 in the project's two previous Python rounds. The improvement is not skill.
It is that those two rounds are written down, so this module's thresholds were
probed from close to the bar from the start: the loudness tolerance at 0.15 and
0.05 LU against a 0.1 bar, the screen at 8 and 7 of 10 against 80%, the
sample-size floor at 14 and 15 against 15, the length drift at 0.5 ms and 2 ms
against 1 ms, and the repeat separation at one step either side of where it
becomes infeasible. **Every mutation that survived was somewhere that lesson had
not been applied** — a constant with no assertion of its own (`BIT_DEPTH`), a
guard shadowed by a second guard, or a value only ever exercised where it is
zero.

Two of the ten were **bad mutations rather than survivors**: one re-derived the
value it was supposed to change, and one wrote a dictionary key that the next
line overwrote. Both were rewritten and both are now caught. That is the third
time a mutation script in this repository has reported a false negative, and the
running rule stands — verify the harness before trusting a negative result.

Eight more were caught only as a **crash**, which stops the suite at the first
defect and reports one bug per round trip. All eight are now routed through a
`guard()` helper that records a raise as an ordinary wrong answer, the same fix
the API suite needed for the same reason.

Final: **94 of 95 caught by a named failure, none as a crash, one survivor** —
the identifier-collision check, discussed above and recorded in
`open-questions.md` as kept-and-unkillable rather than untested.

The suite went from 198 checks to 242 over the exercise.

## Findings from taking the harness end to end

The suite above tests every property of the harness and validates none of them,
which its own closing note said in those words. This is the first run of the
whole thing as a sequence: 24 comparisons rendered from real stems by the
rule-based planner, a 30-trial session built and served, four synthetic
listeners with four different behaviours submitting over HTTP, then score and
correlate. No ears — everything mechanical, which is the half a test client can
reach. Local-only: the stems are gitignored, so none of this reproduces from a
clean checkout.

Nothing about the audio was wrong. **Every defect found was a refusal that
worked and could not be acted on, or a success that was not one.**

### The screen died on the first file anyone would reach for

`build_screening` cuts ten consecutive 8-second segments from frame zero.
`audio/stems/01-Rhythm Guitar-consolidated.wav` — the obvious guitar source for
a guitar-DI panel — opens with eight seconds of lead-in, so segment 0 is digital
silence and the build stopped on:

```
UnmeasurableLoudness: clip 0 measures -inf LUFS — it has no programme material
above the absolute gate, so there is no loudness to match it by
```

Correct, and unusable. It names neither the file, nor which of the ten trials,
nor the one thing that would fix it. The cause is structural rather than a
missing string: `GainSolver` is `Callable[[Sequence[float]], list[float]]`
precisely so it can be unit-tested without audio, so the solver genuinely has
nothing but indices — while `match_group`, one frame up, already holds both the
input paths and the caller's `what`. **A seam that exists to make something
testable will also make its errors anonymous, and the fix belongs on the far
side of the seam, not inside it.** `match_group` now re-raises the same type
with the same reason plus the context it already had, and `screen.py` passes the
segment's time range rather than its index, so the message reads
`screening trial 0 (0.0-8.0 s of the source): ...`.

The failed build also left `_unmatched/` and a partial `clips/` behind, because
the cleanup only runs on success.

### Two submissions in the same second, one file

`SessionStore.record` named files `{listener}-{int(time.time())}.json`. Two
writes inside one second — a POST retried after a timeout, a second tab, two
listeners finishing together on one machine — produced the same name, and the
first file was overwritten. Both requests got `200 {"saved": ...}`.

Reproduced in three lines, so it is not a race that needs losing:

```
first : L1-1785859875.json
second: L1-1785859875.json
files in responses/: ['L1-1785859875.json']
```

A completed session is thirty trials of somebody's afternoon and the interface
holds all of it in memory until the last one, so this is the most expensive
possible thing to drop. It is also the worst *shape* of failure available to
this process: **data loss reported as success**, which nobody looks for until
scoring, by which time the panel has left. Now a counter under a lock — a lock
because `ThreadingHTTPServer` really does serve concurrently, and an `exists()`
check without one is the same bug with a smaller window.

### The parser reports a position in a file it does not name

Two deliberately malformed bodies were POSTed to `/responses` — the server took
both — and `score` then refused the entire session with:

```
ResponseError: expected a list of responses, got str
```

`parse_responses` is right to report a position within one payload; that is all
it is given. But `_read_responses` is iterating over a directory holding one
file per listener per submission, and one bad file refuses **all** of them.
Without the path, the person who ran the panel opens files one at a time. Now
prefixed with the path, for the parse error and for a file that is not JSON at
all.

### The loudness match was doing 7.3 dB of work and reporting 0.0000

`build` prints `matched to 0.0000 LU` for every comparison, and after
re-measuring the delivered bytes off the wire the spread really is 0.00000 LU on
all 24 — the matcher is exact and the verification passes. The number is
therefore also a constant, and a constant is not evidence.

The quantity that shows the match mattered is in the key and is not printed: the
gains actually applied spanned up to **7.30 dB within a single comparison**
(worst case `[0.0, -2.27, -4.31]` and similar throughout), because the rule
planner's chain ends in a limiter to −1 dBTP while the null-render reference does
not. An unmatched version of this session would have been a level test with a
7 dB handicap, which is the entire failure `match.py` exists to prevent — and the
line that proves it is the one the CLI does not show. The protocol document now
points at `key.json` for it.

### What the wire actually carries

Re-measured from the HTTP response bodies rather than from disk, over all 360
clip fetches: spread 0.00000 LU per group against the 0.1 bar, worst true peak
−3.31 dBTP against the −1.0 ceiling, and the builder's reported LUFS agreeing
with the re-measured served bytes to 4.8e−5 — which is exactly the rounding in
`key.json`, not a measurement.

The format is 24-bit PCM with format tag **`0xFFFE`**, a 40-byte `fmt ` chunk,
`SubFormat` `KSDATAFORMAT_SUBTYPE_PCM` and `channel_mask` **0x00000000**; only
`fmt ` and `data` chunks are present, so there is no metadata channel that could
leak the blinding. `WAVE_FORMAT_EXTENSIBLE` with a zero channel mask is the
detail most likely to trip a strict `decodeAudioData`, and an HTTP client cannot
see it fail — `httpx` fetched 725 MB without ever decoding a sample.
`open-questions.md` carries this as the one thing a five-minute browser check
would settle.

### Every traversal probe was already closed, including the undotted one

Thirty-eight probes against the running server — `/key.json`,
**`/clip/key.json`**, `/clip/key`, `/clip/key.json.wav`, percent-encoded and
double-encoded traversal, `....//`, a NUL byte, uppercase hex, `.WAV`,
`/clip/<valid-id>/../key.json`, `/interface.html`, `/etc/passwd` — all 404, no
body containing a condition name. The `/clip/key.json` case that `findings.md`
records as the one that mattered stays closed. Two conclusions worth keeping:
the shape-then-membership check makes the whole class unreachable rather than
individually handled, and the probe list is now long enough that the next person
adding a route has something to run.

### Rho is right, and every proxy was refused anyway

Seven proxies with known relationships to the human scores, 48 paired points:
identity (+1.000), reversed (−1.000), a monotone cube (+1.000 — Pearson would
have said +0.885, which is the whole reason it is Spearman), a constant (NaN,
not zero), a shuffle (−0.060), noise (+0.805) and a coarsely-tied version
(+0.932). Every one agreed with `scipy.stats.spearmanr` to better than 1e−12,
including the tied series, and every interval matched a hand-computed
Bonett–Wright to the same precision.

And every single one came back `NOT USABLE`, including the two that are
correlated with the humans by construction, because the thinnest point rested on
4 ratings against `MIN_N = 15`. That is the documented behaviour working as
designed, and seeing it fire on a rho of exactly +1.000 is the clearest possible
statement of what `open-questions.md` means when it says the calibration panel is
a separate exercise: `min_ratings` is the **listener count**, so fifteen ratings
per point is fifteen people, not more files.

### The always-A listener is caught, at 2^-repeats

Four synthetic listeners, one behaviour each:

```
96 scored trial(s), 24 repeat trial(s) held back for consistency
  audionova/quality: 3.67 [3.04, 4.29] n=4 listeners, 96 ratings
  engineer/quality:  2.67 [1.49, 3.85] n=4 listeners, 96 ratings
  L1: 0/6 reversed, 0/6 changed at all      (discriminating, answers by condition)
  L2: 1/6 reversed, 4/6 changed at all      (chance)
  L3: 6/6 reversed, 6/6 changed at all      (deliberately inconsistent)
  L4: 4/6 reversed, 4/6 changed at all      (pressed A thirty times)
```

The scoring separates all four, and L4 is caught by the mechanism
`consistency_of`'s docstring claims: sides are re-randomised between the two
occurrences of a repeat, so pressing A reverses half the time. Measured over
4000 seeded plans, **P(a side-presser shows zero reversals) = 2^-(repeat
pairs)** — 0.122 at the Phase 0 shape of ten items and three repeats, 0.017 at
six. So the catch is real and probabilistic, it is indistinguishable from an
honest listener at chance, and `report()` never prints the count that would
settle it. Recorded in `open-questions.md` rather than fixed, because adding a
field to `Consistency` changes what the scorer says and that is a decision.

### Wall clock, for a session nobody had timed

Build: 60 s to render 72 clips through the engine for 24 comparisons, then 35 s
to match, verify and write them. Screen: 17 s for ten trials.

Serving is not the constraint and is worth stating because it looked like it
might be: one listener's 30 trials is 90 clip fetches and **181 MB**, delivered
in **0.47 s** over loopback — 16 ms per trial. All four listeners' 360 distinct
clips is 726 MB in 2.0 s. The interface fetches with `cache: "no-store"` and
re-fetches on every trial entry with no prefetch, which is free on loopback and
would be a visible stall per trial over a network; on the same machine, which is
what `serve` binds for by default, it does not matter.

The constraint is the person. Thirty trials at six clicks minimum each, with
switching and listening, is 30–45 minutes with no pause, no back and no resume.
That is plausible for one sitting and it is the upper end of one, and it is the
number `open-questions.md` says nobody has sat through.

### Mutation results

Seven mutations against the three fixes: dropping the re-raise entirely, keeping
the context but dropping the filenames, keeping the filenames but dropping the
reason, removing the uniqueness loop, computing a unique name and writing the
old one, and dropping the path from each of the two `_read_responses` messages.
**Seven of seven caught by a named failure, none as a crash.** The suite went
from 242 checks to 251.

The three "keep one half, drop the other" mutations are the ones worth having.
An error-message test that only asserts the exception type is the failure mode
`findings.md` already records twice; asserting that the message names *both* the
context and the reason is what makes it a test of the message rather than a test
that a message exists.

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
