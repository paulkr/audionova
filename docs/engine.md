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

18 nodes. Ranges are the clamps the engine applies; a value outside them is
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

Aliasing-to-signal ratio is asserted at ≤ −100 dB at 0 dBFS for every nonlinear
node. The ASR test measures in-band and full-band separately, because counting
ultrasonic fold-down as aliasing hides the real number.

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
