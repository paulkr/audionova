# The render engine

`engine/`, C++20, header-only except the CLI. Built with a plain Makefile; moves
to CMake when the pybind11 wheel lands.

```sh
make check     # contract tests + I/O tests + ASan/UBSan build
make render    # build engine/build/anv_render
```

Compiler flags that affect **correctness** rather than speed:

- `-ffp-contract=off` — FMA contraction changes rounding, which would make the
  same source produce different audio on different compilers. Reproducibility
  outranks the free multiply.
- **No `-ffast-math`** — it permits reassociation and assumes no NaN or Inf, both
  of which break the guarantees the contract tests check.
- `-fno-math-errno` — safe, and lets libm calls inline.

## The five invariants

Everything in the engine follows from these. They are enforced by tests, not by
convention.

### 1. float64 end to end

Feedback structures accumulate f32 error into the −90 to −110 dBFS range over a
five-minute render. f64 lands below −250. The cost is memory bandwidth; the
benefit is that no reverb tail or delay feedback path has an audible noise floor
of its own.

### 2. Every node reports exact integer latency

`latencyFrames()` is a promise. The oversampling wrapper's up/down FIR pair is
provably integer at base rate, so there is **no fractional-delay compensation
anywhere** — which removes the single largest source of plugin-delay-compensation
bugs.

The latency a node declares must describe what it actually does *in the
configuration it was prepared for*. `StereoWidth` had a real bug here: it
declared crossover latency even when running as a mono pass-through, so the chain
trimmed thousands of samples off the front of the render.

### 3. Dry/wet belongs to the engine, not the node

`DryWetWrapper` delays the dry path by exactly the node's latency and
crossfades. This makes phase-correct mix universally right and impossible for a
node author to get wrong.

### 4. Block-size invariance is structural

Not tested-and-hoped — structural:

- Parameters evaluate per-sample from an absolute `globalFrameIndex`.
- LFOs are pure functions of absolute time, never accumulators. (`phase +=
  rate/fs` would work and would also make the output depend on where block
  boundaries fell, because floating-point accumulation is not associative.)
- The PRNG is **counter-based**, keyed on `(nodeSeed, channel, globalFrameIndex)`
  — stateless, so identical regardless of block partition or thread count.

`Node.h` states the rule as: *any per-block computation whose result depends on
the block size is a bug.*

### 5. Denormals are flushed identically everywhere

FTZ+DAZ is set on the render thread and in the test harness. This **changes
numeric results**, so a render that ran with different denormal handling than the
tests is not the render the tests validated. A test reads back MXCSR/FPCR and
asserts.

## The node contract

```cpp
class Node {
  virtual const char* type() const = 0;
  virtual void prepare(const PrepareSpec&) = 0;   // allocate here, never in process
  virtual void setParams(const ParamSet&) = 0;
  virtual void reset() = 0;
  virtual int  latencyFrames() const { return 0; }
  virtual int64_t tailFrames() const { return 0; }
  virtual void process(ProcessContext&) noexcept = 0;
};
```

`ParamSet` holds numbers, strings, and **exactly one** ordered sub-list (an EQ's
bands). Everything is in engineer units — Hz, dB, ms, ratio — never normalised
0–1, for two reasons: the planning model writes dB and Hz natively, and a recipe
stays reviewable by a human deciding whether to trust it.

### Defensive parameter handling

The caller is going to be a language model, so:

- Every dB parameter clamps to **±80 dB** (`clampDb`). A parameter sweep found
  four fields that produced `Inf` at 1e5.
- Every allocation-driving time parameter is bounded above. An unbounded
  `lookahead_ms` was a resource-exhaustion path: a 9.6 M-sample window with
  O(W)/sample cost.
- Unknown node types and unknown filter shapes are **hard errors, never silent
  fallbacks.** See `findings.md` — an unrecognised `filter_type` used to become a
  bell, so a planner writing `high_shelf` instead of `highshelf` got a
  plausible-sounding result that was not what it asked for.

## Node reference

22 nodes. Ranges are the clamps the engine applies; a value outside them is
clamped, not rejected.

### Utility

| Node | Parameters |
|---|---|
| `gain` | `gain_db` (±80) |
| `pan` | `pan_position` (−1…1) |
| `polarity` | `inverted` (0/1) |
| `dc_block` | `cutoff_hz` (1…200, default 10) |
| `mono_to_stereo` | `source_channel` (0…15) |
| `mid_side` | `mode`: `encode` \| `decode` |

### Tonal

**`parametric_eq`** — TPT/ZDF state-variable filters per Zavalishin. Minimum
phase, and the default for all tonal moves. Takes a `bands` list:

| Band field | Values |
|---|---|
| `filter_type` | `bell` \| `lowshelf` \| `highshelf` \| `notch` \| `lowpass` \| `highpass` |
| `center_freq_hz` | default 1000 |
| `gain_db` | ±80 |
| `q_factor` | default 0.707 |

An unknown `filter_type` is rejected by `validateRecipe`.

**`zero_phase_eq`** — the same bands run forwards and then backwards
(`filtfilt`). Phase distortion is not small, it is **zero**: the total impulse
response is the autocorrelation of the one-pass response, and an
autocorrelation is an even function. Measured across 20 Hz–20 kHz on a 1 kHz
bell: worst phase **1.2e−11 degrees**, worst group delay **2.4e−10 samples**.
No real-time plugin can offer this at any price, because a zero-phase filter's
impulse response extends backwards in time.

Two consequences that must be read before using it:

- **The magnitude is the SQUARE of the band settings.** Two passes of `H` give
  `|H|²`, so a +6 dB bell measures +12.0000 dB. Ask for half.
- **Latency is not zero.** "Zero phase" is a statement about phase, not about
  declared latency. Filtfilt is latency-free when it is handed the whole signal
  as one array; this engine hands nodes blocks and asks for output before the
  rest of the file exists, so a filter needing R samples of future input must
  declare R. The node declares `chunk + reach` exactly, the chain compensates
  it, and the phase measured after compensation is the zero above.

`reach` is measured from the cascade's own decay — the point where the
remaining tail falls below 1e−12 of its L1 norm — so a slow filter declares
proportionally more. Measured: an 8 kHz Q 0.7 bell declares 186 frames, a
400 Hz Q 2 bell 10 300, and both are exactly zero phase. Reach is capped at
24 000 frames (0.25 s), which bounds latency at 48 000.

**Past that cap the node is no longer exactly zero phase**, and the number is
in `findings.md` rather than hidden: a 20 Hz Q 12 bell needs 5.3 s of reach and
gets 0.25 s, leaving 6.6° of phase error below 100 Hz and 0.12° above it. The
same filter as `parametric_eq` is 19.3°. Below about 100 Hz at high Q, reach
for `parametric_eq` instead.

**`linear_phase_eq`** — the same bands realised as a symmetric FIR. Constant
group delay at every frequency, pre-ringing, and latency of exactly
`(taps − 1) / 2`. Measured on a 2047-tap design: worst phase residual
**7.3e−13 degrees**, worst group-delay deviation **9.4e−12 samples**, and the
kernel is symmetric to the last bit rather than to within round-off.

| Parameter | Range | Default |
|---|---|---|
| `taps` | 15…8191, forced odd | 2047 |
| `bands` | as `parametric_eq` | — |

- **`taps` is forced odd.** An even-length symmetric FIR has its centre of
  symmetry between two samples and therefore a half-integer group delay, which
  this engine has nowhere to put. 2048 becomes 2047 rather than being rejected.
- **Frequency resolution is `fs / taps`** — about 47 Hz at 2047 taps. A move
  narrower than that is not reproduced; it comes back as a broad shallow tilt.
  This is inherent to linear phase, not to this implementation.
- **Cost is linear in `taps`.** The convolution is direct rather than
  partitioned, because the partitioned convolver's latency is one partition
  *plus* the kernel's group delay and this node's contract is that its latency
  is exactly `(taps − 1) / 2`. See `open-questions.md`.

At 4095 taps the magnitude tracks `parametric_eq` to within **0.0034 dB** across
80 Hz–12 kHz, and a band list of zero bands is a bit-exact pure delay at any tap
count.

The design is frequency-sampled and Kaiser-windowed at 70 dB. The window is a
small net *loss* when `taps` is generous — 0.0034 dB against 0.0000 unwindowed
at 4095 taps, because it tapers coefficients that had already converged — and
worth 68× at 255 taps, where an unwindowed design leaks 0.238 dB of a sharp
feature across the rest of the band against 0.0035 windowed. It is kept for the
second case.

Measured on 23 s of 44.1 kHz stereo, rendered at 96 kHz internal: an empty
recipe is 0.84 s, `zero_phase_eq` adds **0.13 s**, and `linear_phase_eq` at
2047 taps adds **7.7 s**. The forward-backward EQ is essentially free; the FIR
is the expensive node in the engine.

Which of the three to use: `parametric_eq` unless there is a reason not to.
`linear_phase_eq` when two paths are summed afterwards and phase coherence
between bands matters more than transient purity. `zero_phase_eq` for surgical
mastering moves above ~100 Hz where any phase shift at all is unwanted.

**`dynamic_eq`** — same band structure plus per-band dynamics. Subsumes the
de-esser, which ships as a preset rather than a node.

### Dynamics

**`compressor`** — log-domain feedforward with a decoupled smooth peak detector
(Giannoulis / Massberg / Reiss, JAES 2012). Lookahead is free offline, so
overshoot is zero.

| Parameter | Range | Default |
|---|---|---|
| `threshold_dbfs` | ±80 | −18 |
| `ratio_to_1` | ≥ 1 | 3 |
| `knee_db` | ≥ 0 | 6 |
| `attack_ms` | 0.01…10000 | 10 |
| `release_ms` | 1…10000 | 120 |
| `lookahead_ms` | 0…200 | 5 |
| `sidechain_hp_hz` | — | 0 |
| `stereo_link` | 0…1 | 1 |
| `makeup_gain_db` | ±80 | 0 |
| `detector` | `peak` \| … | `peak` |

**`gate_expander`** — lookahead opens the gate *before* the transient, which
eliminates the clipped-attack artifact. `threshold_dbfs`, `ratio_to_1`,
`knee_db`, `range_db`, `attack_ms` (0.01…10000), `hold_ms` (0…10000),
`release_ms` (1…10000), `lookahead_ms` (0…200), `hysteresis_db`,
`sidechain_hp_hz`, `stereo_link`.

**`multiband_compressor`** — one compressor per band, over the same
**complementary linear-phase crossover** `stereo_width` uses. Each stage
low-passes what is left and forms the remainder as (input delayed by the stage's
own latency) minus that low band, so the bands sum back to a pure delay exactly,
at every frequency.

**The unity case is the specification, and it is measured, not asserted.** Three
bands at 1:1 with 0 dB makeup null against the input at **−319.1 dB** — the
arithmetic floor of float64, not a design target. That is what makes the node
safe to reach for when one part of the spectrum misbehaves and the rest is
already right: the bands nobody moved cost nothing. A Linkwitz-Riley split
cannot do this, because an LR pair sums to an allpass and leaves a phase
signature even with every band neutral.

| Parameter | Range | Default |
|---|---|---|
| `bands` | 1…6 in the engine, 2…4 in the schema | — |
| `lookahead_ms` | 0…200, shared by every band | 5 |
| `stereo_link` | 0…1 | 1 |
| `crossover_stopband_db` | 40…120 | 80 |
| `oversample_factor` | 1…16 | 4 |

Per band: `crossover_hz` (40…18000), `threshold_dbfs` (±80), `ratio_to_1` (≥ 1),
`knee_db` (≥ 0), `attack_ms` (0.01…10000), `release_ms` (1…10000),
`makeup_gain_db` (±80).

- **`crossover_hz` is the band's UPPER edge**, and the last band's is not read —
  a band that runs to Nyquist has no upper corner. Corners are **sorted**, not
  pushed upward: `(6000, 200)` becomes `(200, 6000)`, which is what was meant.
  Pushing each corner above the previous one would make it `(6000, 6300)` and
  hand everything from 200 Hz to 6 kHz to the low band's settings.
- **Align, then compress.** Each stage costs its own latency, so the bands
  emerge staggered; every band is pulled back to the common total *before* its
  detector runs. Compressing first and aligning afterwards reconstructs just as
  exactly and is wrong: the low band would be reacting to a moment thousands of
  samples away from the one the high band was reacting to. Every band would
  measure correctly on its own.
- **Lookahead is shared**, because per-band lookahead means per-band latency and
  re-aligning after compression is the mistake above.
- Latency is exactly the crossover stages plus the oversampler's round trip plus
  the lookahead, every term an integer at the base rate. Measured with two bands
  and no lookahead: a 100 Hz corner declares 4563 frames, 1 kHz 477, 4 kHz 232.
  A bandless node declares 0.
- **Cost is in the crossover, not the compressors.** The FIR's tap count grows
  as 1/f, so a 40 Hz corner is ~12 000 taps. Corners are clamped to 40 Hz for
  that reason; below that it is a rumble filter, not a band split.
- A **one-band** multiband at `oversample_factor` 1 is bit-identical to
  `compressor` at the same settings, which is the sharpest available statement
  that the detector reads the right signal.
- **Oversampled 4×, but only when a band is actually compressing.** A band at
  1:1 has a constant gain, and a constant gain is linear, so there is nothing
  for an oversampler to prevent — running one anyway would trade the −319 dB
  unity null for the resampler's round trip. The effective factor is 1 when no
  band's ratio exceeds 1, and an `Oversampler` at factor 1 is a pair of copies,
  so the transparent case runs the same code path and stays bit exact.
  Measured in-band ASR at 0 dBFS: **−98.6 dB** un-oversampled, **−152.0 dB**
  at 4×. 8× buys nothing further.

**`transient_designer`** — attack and sustain shaped by the *difference* between
envelope followers of different time constants.

**Not a compressor, and not a substitute for one.** A compressor's gain is a
function of how loud the signal is; this node's is a function of how fast the
level is *changing*. There is no threshold and adding one would be a category
error: a hit 40 dB quieter is shaped identically, measured to **4e−14 dB** on
both the peak and the body. The same two hits through a compressor differ by
9.3 dB.

| Parameter | Range | Default |
|---|---|---|
| `attack_db` | ±18 | 0 |
| `sustain_db` | ±18 | 0 |
| `attack_ms` | 1…200 | 20 |
| `sustain_ms` | 50…3000 | 250 |
| `lookahead_ms` | 0…200 | 2 |
| `stereo_link` | 0…1 | 1 |
| `oversample_factor` | 8…16 | 8 |

Three detectors, and which coefficient each pair shares is the whole design:

```
A = 1 − slowAttackEnv / fastEnv      rectified; positive only while rising
S = 1 − fastEnv / slowReleaseEnv     positive only while falling
gain_dB = attack_db * A + sustain_db * S
```

- **Both controls at zero is a bit-exact delay**, not "no measurable change":
  the gain is `pow(10, 0/20)` and multiplying by one is the identity. The node
  takes an unoversampled path there — see below — but that is a consequence of
  the identity rather than an approximation of it: inside the shaping path the
  same arithmetic produces the same one.
- **Ratios, not dB differences.** Both differentials live in [0, 1) by
  construction. A dB differential is unbounded and behaves worst exactly where
  it matters — a hit rising out of silence puts the fast envelope 20–50 dB above
  the slow one within a millisecond, so any usable control value slams into
  whatever range clamp is bolted on and stops meaning anything. As a ratio the
  same event saturates smoothly at 1, no clamp is needed, and `attack_db` means
  literally what it says.
- **A is rectified and S is not**, and the asymmetry is deliberate: `S` cannot
  go negative, because `slowRelease` shares `fast`'s attack coefficient and
  releases more slowly, so it dominates it at every sample — by induction and
  bit-exactly, since IEEE rounding is monotone. A guard that can never fire
  tells the reader the quantity can go negative when it provably cannot.
- **The fast envelope's times are fixed** (0.5 ms attack, 30 ms release). They
  are the reference the user's two times are measured against, so exposing them
  would let a planner set the reference slower than the thing being referenced
  and get silence from both controls.
- **Known limit:** the detector is broadband, so once a waveform's period
  approaches the fast envelope's release the envelope follows the *waveform*
  rather than the envelope. With both controls at +12 dB a steady 1 kHz tone
  moves **0.033 dB** and a steady 60 Hz tone moves **0.59 dB**. Send bass-led
  material to `multiband_compressor`.
- **Oversampled 8×, and that is the floor rather than a round number.** The
  gain comes from a full-wave rectifier, whose harmonic series falls only as
  1/m² and folds *before* any envelope filter exists to remove it; whichever
  harmonic lands nearest DC is then passed at full gain and modulates
  everything. Measured in-band ASR at 0 dBFS:

  | 1× | 4× | 8× | 16× |
  |---|---|---|---|
  | −66.4 dB | −85.0 dB | **−101.8 dB** | −104.7 dB |

  The requirement is −100, so 4× does not meet it and 16× costs twice as much
  for 3 dB. Unlike `drive`, the oversampling *filter* is not the binding
  constraint here — sweeping the halfband stopband 120 → 150 → 170 dB changed
  the result by 0.00 dB at both 4× and 8×. The rectifier's own series is the
  limit and the only lever on it is the rate.
- **Zero shaping is still bit exact.** With both controls at zero the gain is
  exactly one, so the node takes an unoversampled path, declares only the
  lookahead, and returns its input bit-for-bit — and costs nothing.

Measured on 10 s of 48 kHz stereo rendered at 96 kHz internal: an empty recipe
is 0.29 s, `multiband_compressor` adds **1.83 s** (three crossover FIRs plus 4×
oversampling on six band-channels), `transient_designer` adds **0.68 s** at 8×
and **0.00 s** with both controls at zero.

### Character

**`drive`** — ADAA order 2 with 4× oversampling, roughly the quality of 32×
naive. ADAA2 needs a closed-form second antiderivative, so the shapers are
piecewise polynomials rather than `tanh`.

| Parameter | Range | Default |
|---|---|---|
| `drive_db` | ±80 | 6 |
| `output_trim_db` | ±80 | 0 |
| `bias` | −1…1 | 0 |
| `curve` | `soft_clip` \| `hard_clip` | `soft_clip` |
| `oversample_factor` | — | 4 |

Aliasing-to-signal ratio is **measured** for every nonlinear node — the target
is ≤ −100 dB at 0 dBFS. Every node now meets it, and one does so by very little:
`drive` at +12 dB into 4× oversampling scrapes past at **−100.4 dB**. The two
that did not — `transient_designer` at −66.4 dB and `multiband_compressor` at
−98.6 — are now oversampled and measure −101.8 and −152.0; the thresholds were
not moved. See `docs/open-questions.md` for the `drive` margin and
`findings.md` for why a rectifier aliases the way it does.

This paragraph used to say the requirement was "asserted for every nonlinear
node". It was not, and had never been: the only node under an ASR gate was
`drive`, at −83 to −95 dB. The measurement suite that would have shown this did
not exist until it was built, and the sentence had been true of the intent
rather than of the code for as long as it had been written down. A claim about
what is enforced is worth exactly as much as the thing enforcing it.

The ASR test measures in-band and full-band separately, because counting
ultrasonic fold-down as aliasing hides the real number. Nonlinearity is
detected rather than listed — homogeneity *and* superposition at four levels —
because `transient_designer` is exactly homogeneous (its gain is a ratio of
envelopes) and the obvious test would have excused it from the requirement it
misses by the widest margin.

### Time and space

**`delay`** — Thiran allpass fractional interpolation. Chosen over Lagrange
because in a feedback path the interpolator runs on every repeat: an allpass has
*exactly* unity magnitude, so repeats keep their brightness and the only thing
dulling the tail is the damping control the user asked for.

`time_ms` (1…2000), `tempo_bpm`, `time_note_division`, `feedback_percent`
(0…95), `hf_damp_hz` (1000…20000), `lf_damp_hz` (20…800), `stereo_mode`,
`spread_ms` (0…60), `freeze`, `max_tail_s` (0.1…30).

**`plate_reverb`** — Dattorro (JAES 1997): input diffusers into a
figure-of-eight tank of two cross-coupled halves, each with a modulated allpass,
a damping lowpass and a decay gain.

`decay_s` (0.1…20), `pre_delay_ms` (0…250), `damping_hz` (500…20000),
`bandwidth_hz` (500…20000), `size` (0.4…2), `modulation_depth` (0…1).

The decay calibration is worth reading in the header. A full circuit passes
through **both** halves and each applies the decay gain **twice**, so round-trip
gain is `decay⁴` rather than `decay²` — the obvious derivation is off by a factor
of two in the exponent. The corrected form still overshoots, because the tank's
Schroeder allpasses have lossless sub-loops where energy recirculates without
crossing a decay gain. The exponent is calibrated to measurement (−1.02) with the
reasoning recorded, and the topology's ~0.52 s decay floor is documented rather
than silently clamped.

**`convolution_reverb`** — uniform-partitioned overlap-save. Uniform rather than
non-uniform partitioning because latency is free offline, so the only thing
non-uniform buys is complexity. `decay_s` (0.05…12), `pre_delay_ms` (0…250),
`hf_damp_hz` (500…20000), `lf_cut_hz` (20…800), `hf_decay_ratio` (0.1…1),
`stereo_decorrelation` (0…1).

**`chorus`** — multi-voice modulated delay, read with 4-point Lagrange. The
opposite interpolator choice from `delay`, for the opposite reason: an allpass
carries state, and changing its delay mid-render produces a transient. In a
chorus the delay moves every sample. Lagrange is a plain FIR: no state, no
transient. `voices` (1…6), `rate_hz` (0.01…12), `depth_ms` (0…20),
`base_delay_ms` (1…100), `stereo_phase_deg` (0…360), `randomize_phase`.

### Stereo

**`stereo_width`** — width on the side signal, with an optional band split so the
bass stays mono. Low-frequency side content is where width does its damage: it
costs headroom, cuts badly to vinyl, and partially cancels on single-speaker
playback.

Uses a **complementary linear-phase crossover** (high band = delayed input minus
low band), so low + high reconstructs the input as a pure delay, exactly, at
every frequency. Set both bands to unity and the node nulls against its own
input below −180 dB. An IIR crossover cannot do that — a Linkwitz-Riley pair sums
to an allpass, flat in magnitude but phase-shifted, so it leaves a signature even
when every band is neutral.

`width_percent` (0…400), `bass_mono_below_hz` (0…800), `bass_width_percent`
(0…400), `crossover_stopband_db` (40…120).

No Haas or delay-based widening anywhere. It sounds wide in isolation and
destroys mono compatibility, which is the exact failure this node exists to
avoid.

### Master only

These two are rejected in `nodes[]` — see `architecture.md` for why.

**`true_peak_limiter`** — `ceiling_dbtp` (≤ 0, default −1), `lookahead_ms`
(0.05…200), `release_ms` (1…10000), `oversample_factor` (default 4),
`safety_margin_db` (≥ 0).

**`dither`** — TPDF at 2 LSB peak-to-peak, optional first-order noise shaping,
bit-exact pass-through on float. `bit_depth` (0…32; 0 or ≥32 means no reduction),
`amount_lsb` (0…4), `noise_shaping`: `none` \| `first_order`.

Never dither a float export: a 32-bit float file has no fixed quantisation step
to decorrelate, so dither there only adds noise to a signal that had none.

## The resampler

Polyphase windowed-sinc, hand-written. **Zero net latency, exactly** — the
prototype's group delay is removed by evaluating the polyphase sum at
`m*M + halfLen`, an integer shift at the interpolated rate, so there is nothing
fractional to compensate and no drift on a long file.

Designed to **150 dB** stopband rather than the conventional 120. Kaiser gives
passband ripple roughly equal to stopband attenuation, so a 120 dB design
contributes about −120 dB of error to a tone in its own passband — which would
make the conversion, a stage that is supposed to be transparent, the loudest
error source in the render and the only one big enough to break the −140 dBFS
cross-platform contract. The extra 30 dB costs 1.27× the taps: 68 k instead of
54 k at 44.1 ↔ 96 kHz.

Edges are zero, not extrapolated. The signal is treated as silent outside the
file, so the first and last ~80 input samples fade rather than resolve. That is
the honest interpretation of a file boundary; inventing signal past the edge
would show up as a click when the render is butted against another.

## Two thresholds, deliberately

- **Bit-exact** for same-binary properties: determinism, block-size invariance,
  reset equivalence, bypass null. Any deviation is a real bug.
- **−140 dBFS** for cross-platform. libm ULP differences, SIMD summation order
  and FMA contraction make bit-exactness across x86/ARM a losing fight that
  produces flaky CI. Divergence is reduced at the source by vendoring
  transcendentals.
