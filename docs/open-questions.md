# Open questions and known risks

Everything currently flagged, in one place. This is a living document: anything
raised as a concern during development gets an entry here rather than living only
in a commit message or a conversation.

Each entry says what the concern is, what would resolve it, and — where it
matters — what it would cost to be wrong.

Ordered by how much they could invalidate. The first section contains the things
that could make the rest of the work moot.

---

## Blocking, and not solvable by writing more code

### The Phase 0 listening gate has not been attempted

**The plan's explicit stop-or-go point.** Three working audio engineers blind-A/B
our renders against their own chains on ten guitar DIs, loudness-matched,
prefer-or-tie in at least 6 of 10.

Nothing downstream matters if this fails. Every objective harness in this
repository is a *necessary* condition — they catch what fails measurably, and
they make the listening panel affordable by doing so. None of them is sufficient,
and none can substitute for it.

Needs: real guitar DI audio, three engineers, and at least one authored preset.

**Partial, informal progress, 31 July 2026.** One listener, one file, not blind,
preferred a measurement-derived chain over a generic one on acoustic guitar — and
separately judged the vocal chain good on the vocal stem. That is the first
human judgement of any output here and it moves the prior, but it shares none of
the properties that make the Phase 0 gate meaningful: it is not blind, not
against the listener's own work, and both chains were written by the same hand
that predicted the outcome. See `findings.md`.

### Zero presets authored

The plan budgets **4–6 person-weeks for twelve** and states plainly that preset
authoring is not an engineering task. The glossary routes `vintage`, `modern` and
`lofi` to a knowledge base that does not exist.

A wrong preset is a confidently wrong answer, so twelve excellent ones beat sixty
thin ones. The first one is also where the plan's own factual error has to be
resolved: "Day Tripper" was recorded 16 Oct 1965 and ADT was invented in April
1966, so the doubled riff is manual double-tracking. The preset schema carries
citations with page numbers and confidence levels precisely so this class of
error is catchable.

### Twenty-one glossary conditions name a measurement that does not exist

The glossary is now sixty terms, but the count was never the risk. Twenty-one of
the original forty carry a `check_first` that nothing can evaluate, and four of
those are outright wrong rather than merely vague: `dry`, `washy` and `distant`
turn on **reverb decay**, and `pumping` on an **existing compressor's release**.
The measurement report produces neither. The rest are unactionable on purpose
(`vintage`, `honest`, `clean`) or true of a spectrum without naming one field
(`boxy`, `warm`, `nasal`).

This is worse than a missing term, which is the reason it is here rather than in
a backlog. A planner reading "reverb decay is long relative to the material's own
note rate" cannot check it, falls back on the stock move, and now has a
rigorous-looking sentence to justify it with. A term with no condition at least
announces itself.

Resolves either by adding a blind reverb-time estimate to the report — the
report's own docstring already claims one exists, which is how this was noticed —
or by rewriting the four conditions against what is measured. `test_planner`
holds the exemption list, and asserts it can only shrink.

Cost of being wrong: reverb and compression moves made on conditions nobody
evaluated, on exactly the material where a spatial mistake is most audible.

### The genre- and instrument-specific vocabulary is still missing

Sixty terms is the plan's number, and the twenty-one added to reach it are
general engineering vocabulary — the antonyms users type alongside the terms
already present, plus the words that separate three neighbouring complaints
(`harsh`, `shrill`, `sibilant`) by measurement. None of them is genre- or
instrument-specific, and that vocabulary still belongs with the presets, authored
by someone who mixes for a living. "Motown bass" names a record, not a region.

---

## Unverified because nothing has run yet

### The planner still has not run end to end

The two API properties the design rests on are now verified (see Resolved), but
no PLAN or EMIT call has produced an actual graph, and nothing has been scored
against the rule-based floor. That is the next thing.

### Cross-platform determinism is asserted but untested

The contract is bit-exact within `(engine_version, arch)` and −140 dBFS across
architectures. **Everything has only ever run on one machine.** The −140 dB
threshold was chosen because libm ULP differences and SIMD summation order make
bit-exactness across x86/ARM a losing fight — but that reasoning has never been
checked against an actual second architecture.

The CI matrix now targets both, so this resolves the first time the workflow
runs somewhere with a runner. What it will show is unknown: −140 dB may turn out
to be too tight, too generous, or right.

### CI has never actually run on a second machine

`ci/run.sh` exists and passes locally in about three minutes, and the workflow
targets both x86-64 and arm64. But it has only ever executed on one machine, so
the cross-architecture question below remains open until it runs somewhere else.

Still missing from the gates: an SBOM, and the prompt-cache canary asserting
`cache_read_input_tokens > 15000`, which cannot be written until there is a key
to run it against.

---

## Measurement gaps

### `drive` at +12 dB passes the aliasing requirement by 0.4 dB

`engine/tests/measurements.cpp` measures aliasing-to-signal ratio by sweeping a
coherently-sampled tone at 0 dBFS and masking the harmonic loci, which is the
plan's own method. The requirement is **ASR ≤ −100 dB at 0 dBFS for every
nonlinear node**. Every node meets it; one does so by a margin thin enough to
be worth naming:

| node | in-band ASR @ 0 dBFS | against |
|---|---|---|
| `drive` (+12 dB, 4×) | −100.4 dB | −100, **by 0.4 dB** |
| `drive` (defaults, 4×) | −105.1 dB | −100, by 5.1 dB |
| `drive` (+12 dB, 16×) | −135.7 dB | comfortable |
| `transient_designer` (8×) | −101.8 dB | −100, by 1.8 dB |
| `compressor` | −104.2 dB | passes |
| `dynamic_eq` | −107.8 dB | passes |
| `true_peak_limiter` | −124.3 dB | passes |
| `multiband_compressor` (4×) | −152.0 dB | comfortable |

**Resolved, 4 August 2026: the two dynamics nodes that failed.**
`transient_designer` was −66.4 dB and `multiband_compressor` −98.6 dB. Both are
now oversampled and both pass; the numbers and the reasoning are in
`engine.md` and `findings.md`. The thresholds were not moved. What remains
below is the `drive` observation the original entry ended on, kept because it
is the sharper half of the finding.

`transient_designer`'s 1.8 dB is thin and worth watching. 16× would buy 3 dB
more and doubles the node's cost; the decision was made on the measurement and
is recorded in the node header so it can be revisited with a number rather than
an impression.

**Also worth knowing:** `drive` barely moves between 4× and 16× at 0 dBFS
(−100.4 against −135.7 is a large gap at the worst sweep step, but the in-band
figure at most steps is dominated by the downsampling filter's stopband rather
than by the factor). The oversampling *filter* is the binding constraint, not
the oversampling *ratio* — mutating the filter's stopband from 120 dB to 40 dB
costs 32 dB of ASR, while removing the oversampling entirely costs 35 dB.

**And that does NOT generalise, which is worth knowing before applying it.**
The same experiment on `transient_designer` found the opposite: sweeping its
halfband stopband 120 → 150 → 170 dB changed its ASR by **0.00 dB** at both 4×
and 8×, while doubling the rate was worth 17 dB. A saturator's products are
generated inside the oversampled band and the filter decides how much comes
back; a rectifier's harmonic series runs to infinity and the *rate* decides
which member of it folds. Which lever works depends on where the harmonics
come from, so the diagnosis has to be redone per node rather than carried over.

### No true-peak meter here is trustworthy on content within ~2 kHz of Nyquist

The 0.02 dB limiter overshoot that used to be the entry here is fixed and has
moved to the Resolved table. Fixing it exposed a larger and less comfortable
fact about the *measurement*, which is what this entry is now about.

**The 32× / 6145-tap reference meter in `engine/tests/measurements.cpp` is
itself 0.09 dB low on the signal it is used to judge.** Ground truth was
obtained a fourth way — DFT-domain sinc interpolation, which for a buffer that
decays to zero at both ends *is* the band-limited reconstruction, with no
filter, no window and no transition band to argue about. It agrees with the
analytic answer on a unit sinusoid to 1e-4 dB at every frequency tested,
including 47.9 kHz.

On the limiter's own stress buffer, before any of this commit's changes:

| meter | reading |
|---|---|
| exact (DFT-domain, 512× grid) | **−0.886 dBTP** |
| 32× / 6145-tap Blackman-Harris reference | −0.979 dBTP |
| engine detector, 4× | −1.000 dBTP |

So the reported overshoot was 0.021 dB and the real one was **0.114 dB**. The
reference was not wrong about the direction, only about the size — but it was
wrong about the size by a factor of five, and it is the instrument the CI gate
is built on.

**The mechanism is the transition band, and it is fundamental.** Every
windowed-sinc interpolator has to put its transition somewhere, and the first
image begins exactly at base Nyquist, so a zero-width transition is not
realisable. Content in the top ~2 kHz is therefore attenuated by *some* amount
in every meter here, and the resulting peak error is **not signed** — it moves
the peak by whatever the discarded content contributed at that instant, which
was +0.032 dB on the stress input and −0.114 dB on the stress output. Measured
against the exact answer at 96 kHz, 16×:

| material | t=0.05 | t=0.02 | t=0.01 | t=0.005 |
|---|---|---|---|---|
| anything band-limited to 24 kHz, incl. hard-clipped | −0.001 | +0.000 | −0.000 | −0.000 |
| hard-clipped noise with energy to Nyquist | −0.466 | −0.219 | −0.078 | −0.055 |
| band-limited impulse at Nyquist | −0.117 | −0.046 | −0.022 | −0.010 |

Halving it costs roughly four times the runtime each time and it does not reach
zero. This is the same limit `backend/measure/loudness.py` records in
`_interpolation_filter`'s docstring for the Python meter, arrived at from a
different direction.

After this commit the limiter delivers **−1.041 dBTP** on band-limited
programme at a −1.0 ceiling — conservative, as a safety limiter should be — and
**−0.928 dBTP** on the synthetic stress signal, still 0.072 dB permissive
because the detector cannot see the content that produces it. The CI gate
passes because the reference meter reads that buffer as −1.021.

**What would resolve it:** replacing the reference meter in `measurements.cpp`
with the DFT-domain method for buffers that decay to zero at their edges, which
is most of the suite's stimuli. That makes the gate measure the true quantity
rather than a good approximation of it, and it is *cheaper* than the 6145-tap
convolution it replaces. It was deliberately not done in this commit: the
instrument and the thing being measured should not change in the same change.
**Cost of being wrong:** the engine's delivery ceiling is verified against a
meter with a known 0.09 dB blind spot on the one signal class the ceiling is
most at risk from. Nothing clips at 0.07 dB over a −1 dBTP ceiling; the cost is
that the number in the report is not the number in the file, and that gap is
invisible until somebody measures a fourth way.

**A caveat on cross-meter comparison, stated because the alternative is
overclaiming.** The engine detector and the 32× reference agree to within
0.03 dB on twenty of the twenty-four configurations measured, and diverge to
**0.10 dB on `drive`**, whose output is the most harmonically dense in the set —
there in the other direction, with the engine's meter reading *higher*. Both are
validated against sinusoids, where an analytic answer exists. Comparing two
approximations on broadband content tells you they disagree, not which is right;
that is exactly why the exact method above had to be brought in to settle it.

### The aliasing sweep cannot see transient-only nonlinearity

The ASR probe is a steady tone by construction — coherent sampling requires it.
A node whose nonlinearity only appears during an envelope transition therefore
measures clean: `gate_expander` reports **−250 dB**, which is true of a steady
tone through an open gate and says nothing about what the gate does at the
moment it opens.

Partially mitigated: the suite also runs the sweep at whichever level the
linearity probe found the node most nonlinear, which catches threshold-straddling
behaviour that 0 dBFS misses (`dynamic_eq` goes from −107.8 dB at 0 dBFS to
−71.4 dB at −40 dBFS on that probe). It does not catch anything that requires a
*transient*.

**What would resolve it:** a burst-based aliasing probe — a tone gated on and
off, analysed with a windowed transform, accepting the leakage floor that comes
with it and reporting a number that is only meaningful down to about −70 dB.
**Cost of being wrong:** gate and transient artefacts are the ones listeners
describe as "chattery", and nothing in the suite currently measures them.

### Distortion character is invisible

The report can say a source has 8 kHz energy a clean version would lack. It
cannot say whether that came from a valve, a transistor or a fuzz pedal — and for
"make this sound like that record's guitar tone", **that difference is the entire
request**.

Drive itself is detectable (through its harmonics and its effect on dynamics), so
this is narrower than first thought. What is missing is a clean-versus-driven
classifier, which the plan schedules in CLASSIFY.

### Space effects are undetectable on dense material

A delay and a plate together moved the bass stem's p10-to-p90 fill by **−0.42 dB**
— nothing. Dense sources never leave gaps for added space to show up in level
percentiles.

Consequence: the `space_only` recovery chain scores F1 0.20 against 0.70+ for the
tonal chains. The report may genuinely not carry this information for dense
material. A planner that recovers it would be demonstrating something real rather
than pattern-matching.

### The codec detector picks its wall by slope alone

Fixed separately: the *confidence* the detector reports now measures distance
from the decision (see the Resolved table). What was not touched is which
candidate it reports. The wall is still chosen by `argmax(slope)` among the
eligible bands, so a very steep cliff sitting one decibel inside the depth guard
beats a moderately steep one with 35 dB of margin. The confidence is then
honest — it comes out near zero — but the reported `bandwidth_limit_hz` is the
less informative of the two.

Measured: lifting a 3 kHz bell on the demo vocal chain's output by 10 dB moves
the reported wall from 21223 Hz to 20791 Hz while the audio above 15 kHz is
untouched. Both readings are low-confidence now, so nothing acts on either.

**What would resolve it:** select the candidate by the same weakest-link margin
the confidence uses, rather than by slope, and confirm on the real encode corpus
(lame 64/96/128/192, afconvert AAC 64/96/128 over the acoustic guitar, vocal and
bass stems) that no genuine wall moves frequency.

**What it costs to be wrong:** little today. Every case where the two rules
disagree is a case with a small depth margin, which now reports below the 0.4
the planner acts on, so a mis-chosen frequency is attached to a verdict nobody
uses. It becomes a real cost only if the 0.4 threshold is ever lowered.

### The grid and span in the bandwidth detector are not the sizes they are called

`bandwidth_limit_hz` documents "a 1/12-octave grid" and a slope "measured over a
third of an octave". The grid is 120 points from 2 kHz to 23 kHz, which is 3.53
octaves in 119 steps — a 1/33.7-octave grid — and `span = 4` makes the slope
window 0.118 octaves, about 1/8.4 of an octave.

The numbers the detector is calibrated against were measured on this grid, so
the thresholds are right and nothing is currently mis-detecting. The risk is
purely that someone reads the comment, "corrects" the grid to the 1/12 it claims,
and silently moves every threshold: the drop gate's effective slope threshold is
`min_drop_db / span_octaves`, so tripling the span width would move it from
169 dB/octave to 56 and start flagging acoustic sources again.

**What would resolve it:** decide which is intended, then either fix the comment
or fix the grid and re-derive the thresholds against the real encode corpus.

**What it costs to be wrong:** nothing until someone edits it, then a false
positive on exactly the dark acoustic material the detector was rewritten to
stop flagging.

### The reference corpus has two gaps

Six stems covering mono, dual-mono, real stereo, 44.1/48/192 kHz, and a dark bass
through a bright acoustic. Not covered:

- **percussive transient material at a normal rate** — the only drum stem is
  192 kHz and turns out to be upsampled from 44.1
- **anything genuinely wide at low frequencies**, which is where stereo width does
  its damage and where `mono_loss_db` most needs exercising

Source-separated stems were considered and rejected on both licensing and
technical grounds — see `licensing.md`.

---

## Metrics that may be measuring nothing

### The ordering score may deserve retirement

**The artifact is fixed; the underlying question is not.** The +0.498 is gone —
ordering is now scored only over mirrored chains and reported as an excess over a
matched order-blind control, and the rule-based baseline reads **+0.000** on 81
mirrored chains, which is the honest floor. See `evaluation.md`.

What remains open is whether the metric can ever read anything else. The
corrected floor is zero *by construction* for any predictor that does not reason
about order, and nothing has yet demonstrated a non-zero excess. **If no planner
can separate a chain from its reverse, measurement deltas do not carry ordering
and the metric should be retired rather than chased.**

Resolving it needs a planner that tries. Costs of being wrong in each direction:
keeping a dead metric invites optimising against noise; retiring a live one
discards the property LLM2Fx-Tools found decided perceptual preference between two
systems with identical parameter accuracy.

### Detection floors are derived from a sweep of four values

Every "is this move detectable" decision in the recovery corpus comes from
`evidence_floor`, which returns the **smallest amount the sensitivity sweep ever
applied** — 5 dB for a bell, 4 dB for a shelf, 6 dB of drive, 2:1, 0.12 reverb
mix. That is the smallest amount anything here has evidence for, and deriving it
beats writing a number down.

It is still not a *measured* floor. The true detection threshold is somewhere
between the sub-threshold tier and the demonstrated one, and the `borderline`
tier — 0.35 to 0.7 of the floor — sits in that gap with no evidence either way
and is excluded from the headline for that reason.

Resolving it means sweeping each perturbation *downward* until the report stops
tracking it, which is a sensitivity-harness run rather than new code. Cost of
being wrong: the floor excludes moves a planner could actually have found, which
makes the headline flatteringly easy, or includes ones it could not, which makes
it unfairly hard. Both are bounded by a factor of about three.

### The recovery corpus scores parameters only for EQ

Frequency in octaves and gain in dB are scored; compressor ratio, drive amount,
reverb mix and delay time are not. That is not a principled boundary — it is
where the recogniser happens to produce a number.

The risk is a planner that recovers every node type and every EQ band while
getting a 20:1 ratio where a 2:1 was applied, and the harness reporting that as a
clean run. Resolving it means the recogniser estimating those parameters at all,
which the sensitivity table says is possible for ratio (modulation centroid
tracks it at r = 0.94) and says nothing about for the rest.

### Gain staging cannot model what it is bounding

The check sums static gains, the largest EQ boost and makeup against the measured
p95. It cannot model compression or saturation, so it is a **bound rather than a
prediction**. It catches a chain arriving 15 dB hot; it will not catch a chain
that arrives 4 dB hot through a compressor's makeup interacting with a drive
stage.

### Three checks in the Python stages cannot fire at all

Found by mutation-testing `backend/verify/` and `backend/measure/`. Each of these
reads as a second line of defence and is unreachable by construction, so deleting
it is behaviour-preserving — which is why deleting it was the one mutation in
each module that no test could catch.

**1. The verify gate's mono-loss check.** `gate.verify` tests correlation first
and mono loss in an `elif`, so the mono-loss branch only ever sees material that
already cleared `MIN_CORRELATION = -0.2`. For channels of power `P_L` and `P_R`
at correlation `r`, the loss beyond the arithmetic 3 dB is
`-10*log10(1 + r*k)` with `k = 2*sqrt(P_L*P_R)/(P_L+P_R) <= 1`, maximised when
the channels are equally loud. At `r = -0.2` that is **0.969 dB against a
1.0 dB bar**, and unequal channel levels only reduce it. Confirmed empirically
over a sweep of anti-correlation and channel-balance: no signal reaches the
branch.

The two constants are two spellings of one decision, and `test_verify.py` now
says so, so loosening either without the other fails. What it does not do is
give the branch a job. **The interesting version of this check is per-band**:
`stereo.per_band` already reports mono loss per ERB band, and a mix that is
broadband-fine while cancelling in the bottom two octaves — a bass DI with a
phase-rotated double, a widened kick — is a real failure the broadband number
cannot see. Resolving this means either wiring the check to the per-band figure
or deleting it. It costs nothing to be wrong today, because the correlation bar
catches everything the mono-loss bar would.

**2. The empty-gate fallbacks in `loudness._gated_loudness` and
`gating.gated_frame_mask`.** Both read `if not np.any(keep): ...` after
`keep = above_absolute & (block_loudness > first_pass + RELATIVE_GATE)`. The
first pass is `10*log10` of the **mean** of the linear block powers and the
block loudnesses are `10*log10` of the individual ones, so the loudest block is
always at or above the first pass and therefore always clears a *negative*
relative offset. Checked over 20 000 random block sets: the minimum of
`max(block_loudness) - first_pass` is exactly 0.0. `keep` is never empty.

Harmless, and worth knowing before someone reads the fallback as evidence that
the gate handles a case it never sees.

### Nine more branches in the planner and refine stages cannot fire either

Second round of the same exercise, on `plan/validate.py`, `refine/match_eq.py`
and `refine/match_dynamics.py`. These are the mutations no test could catch
because the code they change is unreachable or behaviour-neutral, not because
the test was weak. Each is recorded rather than left as an apparently-untested
check.

**1. The Q floor in the ringing repair.** `_check_filter_stability` repairs an
over-radius pole by solving for the widest Q that clears the bar, floored at
`max(0.5, ...)`. The floor binds only where the solved Q would fall below 0.5,
which is **below 7.64 Hz**; the schema's minimum `center_freq_hz` is 20 Hz. The
floor can be 0.05 or 5.0 and no legal graph reaches it.

**2. The master section in `_check_filter_stability`.** It iterates
`graph.nodes + graph.master`, but `MASTER_ONLY` is `("true_peak_limiter",
"dither")` and neither model has a `bands` field, so the master half of that
loop can never see a band. (The equivalent loop in `_check_ranges` *is*
reachable — those nodes have bounded scalar parameters — and is now tested.)

**3. The `gate_expander` exemption in the noise-floor check.** The check skips
`gate_expander` before iterating its bands, on the reasoning that a gate is the
right answer to a noisy source. `GateExpanderParams` has no `bands` field, so
the inner loop is empty with or without the exemption. It documents an intent
correctly and does nothing.

**4. `soft_clamp`'s zero-limit guard.** `if limit_db <= 0.0: return zeros`.
Falling through at exactly zero gives `start = 0`, `headroom = 0` and
`compressed = 0 + 0 * tanh(x / 1e-9)`, which is zero for every input — verified
across the sign boundary. The guard is an early exit, not a behaviour.

**5. The greedy seeder's give-up threshold, from below.** `_greedy_seed` stops
placing bands once the largest remaining error is under 0.35 dB. Lowering it to
0.02 changes nothing observable: `choose_cascade` refits every size from one
band upward and takes the smallest within 0.15 dB, so extra seeded bands are
discarded on the way out. Only the *upper* side of that constant is observable
(at 2 dB a real 1.2 dB correction produces no bands at all), and that direction
is now tested. Lowering it costs compute and nothing else.

**6. The 800 ms cap on the derived release.** `timing_from_source_rate` clips
`period_ms * 0.5` to 30–800 ms. Reaching 800 needs an envelope centroid below
0.625 Hz. Measured across five real stems and two synthetic sources the centroid
runs **2.3 to 14.9 Hz**, giving releases of 33 to 216 ms — the cap has never
been within a factor of four. (The 30 ms floor nearly binds: drums measure
33.5 ms.)

**7. The 0.05 Hz centroid floor in the same function.** It separates material
with a measurable modulation rate from material with none. Everything measurable
sits above 2.3 Hz and everything else reads ~0, so the floor can be anywhere in
between — moving it to 1.0 Hz changes no verdict on any source available here.

**8. The low-confidence fallback in `match`.** It reassigns the time constants
to 15/150 ms when confidence is under 0.4. `modulation_summary` returns 0.8 or
0.0 and nothing else, and 0.0 is exactly the case where
`timing_from_source_rate` has already returned 15/150 — so the assignment always
writes what is already there. **The gate itself is not unreachable**: raising it
to 0.9 makes every render use the default timing, which is now tested. Only the
assignment is redundant.

This one is worth acting on rather than only recording. A two-valued flag
returned as a float invites exactly this: a threshold chosen as though the
quantity were continuous, sitting in a gap the quantity can never occupy. Either
`modulation_summary` should report a real confidence — the obvious candidate is
how peaked the modulation spectrum is around its centroid, since a broad one
means the centroid is not describing anything — or the field should be a
boolean and the gate an `if not measured`.

**9. Smoothing finer than 1/6 octave on the default grid.** `smooth_log`
converts a fraction of an octave into a window width in grid points. The default
256-point grid spans 30 Hz to 16 kHz, which is 0.0355 octaves per point, so
1/6 octave is already only 4.7 points and everything finer floors at 5. Asking
for 1/24 octave gives the identical window. The knob is live from 1/6 octave
downward in fineness and saturated above it; nothing currently asks for more,
and a caller that wants it needs a denser grid rather than a smaller fraction.

None of these is worth deleting. Each reads as a second line of defence, and the
cost of being wrong is zero in all of them — but a reader should know that six
have never executed a single time, and that #8 hides a measurement worth fixing.

### The true-peak meter's 8x oversampling at 48 kHz buys nothing measurable

`_oversample_factor` returns 8 at or below 48 kHz and 4 above, justified in the
docstring on the grounds that "the error of a 4x estimate grows toward Nyquist".
Mutating it to 4 at 48 kHz was not caught, and strengthening the test did not
change that, because **the two factors produce identical readings**: 0.0001 dB
apart or exactly equal on unit sinusoids from 997 Hz to 23 kHz, on clipped pink
noise, on a square wave and on an impulse train. The taps-per-phase is fixed at
48, so raising the factor refines where the reconstruction is *sampled* without
improving the filter, and the `max(peak, sample_peak)` floor supplies the answer
in most of the cases where it would matter.

Only one construction separated them — two tones at 20 and 23 kHz with the
sample-peak floor disabled, 4x reading 0.126 dB low against 8x at 0.074 — and
with the floor in place, as the meter actually runs, even that collapses to the
same number.

Either the justification is wrong and 4x everywhere would halve this
measurement's cost, or there is material where it binds that has not been
found. Resolving it needs a case where 8x is measurably better with the meter
in its shipping configuration. It costs little to be wrong: the error is
runtime, not accuracy.

### The true-peak meter under-reads content above about 19 kHz

Found while looking for the case above. The interpolation filter's passband
reaches 0.95 of the base Nyquist, so the top kilohertz or two is attenuated in
the reconstruction and the meter reports **less** than the true peak — the
unsafe direction for a delivery ceiling.

Two tones at 19 and 23 kHz have an exactly known continuous peak, because the
band-limited reconstruction of their samples *is* that sum. The meter reads
**0.86 dB below it**. A single tone hides this entirely: its sample peak is
close enough to its true peak that the floor supplies the answer, which is why
the analytic conformance case — which sweeps to 19 kHz and stops — reads
0.0002 dB there.

`test_loudness.py` now pins the limit so it cannot silently worsen. Whether it
matters in practice depends on whether any real delivery has that much energy
that close to Nyquist; a 44.1 or 48 kHz master usually does not, and the engine's
own conversion filter rolls off at 20 kHz. The case that would matter is a
render whose top octave was deliberately lifted.

**Largely fixed since, and the residual is now the general entry above.** The
passband edge went 0.95 → 0.99 and the filter 48 → 64 taps per phase, taking the
20+22 kHz case from −1.078 dB to −0.028; see `_interpolation_filter`'s docstring
for the measured table. What is left is the same term B the C++ detector has, and
it is the same fundamental limit in both: no finite filter reconstructs content
arbitrarily close to Nyquist, and the transition band has to go somewhere.

### `channel_weights` treats channel 4 of a 5.1 file as a surround

`loudness.channel_weights` sets `weights[3]` and `weights[4]` to 1.41 for any
layout of five channels or more. That is right for 5.0 (L, R, C, Ls, Rs) and
wrong for **5.1** (L, R, C, LFE, Ls, Rs), where index 3 is the LFE — which
BS.1770 excludes entirely — and Rs at index 5 is left at unity. The docstring
says "LFE is excluded entirely"; no line of code excludes anything.

Not reachable from the product, whose input is a mono or stereo stem, and the
5.0 case is now asserted in `test_loudness.py`. Resolving it means either
taking a channel layout as an argument or refusing more than five channels,
because a count alone cannot distinguish 5.0 from 5.1.

### No proxy metric has been calibrated against human judgement

The plan is firm: **no metric gates CI until it passes ρ > 0.6 against human MOS**,
and parameter MAE is never a quality gate — the regression baseline in LLM2Fx had
the best MAE and nearly the worst perceptual score.

Nothing here has been calibrated, because there is no human data. Every number in
`evaluation.md` should be read as "this measures what it says it measures", not as
"this predicts quality".

**The blocker is now people rather than code.** `backend/listen/` collects the
human data and `backend/listen/correlate.py` computes the ρ. Neither has ever
seen a real listener. The entries below are what is unresolved about the harness
itself.

### The listening harness has never been used by a person

Every property of it is tested and none of it is validated. The tests establish
that the mapping is not recoverable, that clips are matched, that rho is
arithmetically right — they establish nothing about whether a working engineer
can complete a session, whether the two scales mean to them what the labels say,
or whether thirty trials is past the point of fatigue.

The specific risks a test cannot reach: an 8 ms gain ramp that turns out to be
audible as a click on percussive material; a `decodeAudioData` failure on some
browser's 24-bit WAV handling; and a session length nobody has sat through.

Resolves by running one session with one colleague before booking three
engineers. Cost of skipping it: the panel is the expensive, hard-to-repeat step,
and discovering the interface is unusable during it wastes the only three
engineers who agreed.

### The request text can reveal a repeat trial

One of the two scales asks whether the render matched the request, so the request
has to be on screen. A repeated pair therefore carries the same request text
twice, and an attentive listener may notice.

Mitigated rather than solved: repeats sit at least three trials from their
original, and the A/B assignment is independently re-randomised between the two
occurrences, so noticing a repeat does not tell a listener what they answered
last time. The residual is that a listener who *knows* it is a repeat may try to
be consistent on purpose, which inflates the consistency figure — the one number
that exists to say how much to trust them.

Resolves either by using a different excerpt of the same material for the second
occurrence — which changes the stimulus and so measures something slightly
different — or by paraphrasing the request, which risks changing the question.
Cost of being wrong: `reversal_rate` reads better than the listener is, and a
noisy listener is not caught.

### Two clip identifiers resolving to one file are byte-identical over the wire

Clip identifiers are per trial occurrence, so a repeat's filenames do not give it
away. The bytes do: the server resolves both identifiers to the same file, so the
two responses have the same length and the same content, and a listener with the
network tab open could pair them.

Not fixed, because the fix is worse: padding or re-encoding a clip per occurrence
means shipping audio nobody measured, and the whole match-and-verify chain exists
so that the served bytes are the measured bytes. The threat is a colleague's
curiosity, not an attacker.

### `may_gate_ci` is stricter than the plan's literal wording

The plan says a metric may gate CI at ρ > 0.6. `may_gate_ci` requires the 95%
interval's **lower bound** to clear 0.6, not the point estimate. On twenty
paired points a sample rho of 0.62 has a lower bound of 0.28, so the two
readings disagree on exactly the cases that matter.

This is a decision, not a reading, and it is recorded here so it is argued with
rather than discovered. `use_lower_bound=False` gives the literal reading and is
tested. Cost of being wrong in this direction: a genuinely good proxy waits for a
larger panel. Cost of being wrong in the other: a proxy blessed on noise becomes
a CI gate, which is the proxy-metric capture the plan names as its top risk.

### `MIN_N = 15` applied to both axes means no Phase 0 panel produces a usable rho

`usable` is false below fifteen paired points **or** fifteen human ratings per
point. The Phase 0 gate is three engineers on ten files: ten points, three
ratings each. Both axes fail, by a factor of five on one of them.

That is the honest answer — ten points cannot support a correlation — but it
means the harness will refuse to bless anything on the data the plan actually
budgets for. Whoever runs the panel needs to know that calibrating a proxy is a
*separate*, larger exercise from passing the gate, and that the two cannot be
done with one session.

Resolves by deciding the calibration panel's size on purpose. Cost of not
deciding: someone reads `NOT USABLE` as a harness bug and lowers the constant.

### The identifier-collision check cannot be made to fire

`plan_session` refuses a session in which two trials or two clips share an
identifier. With 64 bits from a seeded Mersenne Twister that does not happen, and
mutation testing confirms the check can be deleted with no test failing.

Kept rather than removed, unlike the post-write ceiling check that was deleted in
the same round. The difference is that the ceiling check guards an invariant that
is provable from three lines above it, whereas this one guards an assumption
about a random number generator that a future change could break — and the
failure it prevents is two listeners' answers silently merging.

---

## Performance and architecture

### The cached prompt prefix grows with the node set, and its bound is not measured

Every node type adds its parameter schema to the PLAN prefix — about **1,000
estimated tokens each**. `test_planner.py` brackets that prefix, and the bracket
has now had to move twice. It was `26,000 < tokens < 36,000` and the prefix was
already at **35,388** with twenty node types, so `multiband_compressor` and
`transient_designer` would have broken it whatever their descriptions said. It
is now `< 44,000`, roughly six more nodes of headroom.

Two things are unresolved and they compound:

- **The bracket is around an ESTIMATE, not a measurement.** The live API reported
  30,938 cached tokens once; everything since is `len(prefix) / 2.4`, a constant
  fitted to that single observation. Nobody has re-measured, so the relationship
  between the number the test checks and the number that is billed is a
  four-month-old straight line through one point.
- **The prefix grows without a budget.** Nothing decides how large it is allowed
  to get; the test only notices afterwards. At some node count the cached prefix
  stops being the cheap half, and the current mechanism will report that as a
  test failure to be re-bracketed rather than as a design decision.

Resolves by making one live call and reading the `cache_read_input_tokens`
counter, then deciding a ceiling on purpose. Costs, if wrong: a per-call bill
larger than the design assumed, discovered from an invoice rather than a test.
The `SOURCE_NODES` list is the obvious lever — it is six headers chosen by hand,
and neither of the two nodes added here is in it.

### The engine is invoked per render as a subprocess

pybind11 is planned. The subprocess costs a few milliseconds against a render
measured in seconds, and buys the property that Python exercises exactly the
binary that ships. Worth revisiting when throughput matters, not before.

### Measurement is slow on long files

A four-minute stereo file takes about **17 s**, dominated by 20 zero-phase band
filters. Under the excerpt-first default of 25 s that is ~1.7 s, which is what the
product will actually run. Not worth optimising until the planner reveals which
numbers it reads.

### `linear_phase_eq` convolves directly, and the cost is linear in `taps`

The node's contract is that its declared latency is **exactly** `(taps − 1) / 2`,
which rules out the partitioned convolver `stereo_width` uses: that convolver's
latency is one partition *plus* the kernel's group delay, so a node built on it
could only ever declare `(taps − 1) / 2 + partition`. Direct convolution keeps
the contract and costs `taps` multiplies per sample per channel — at the 8191-tap
ceiling that is roughly 1.6e9 multiplies per second of stereo audio.

Two ways out, neither taken yet:

- Let the node declare `(taps − 1) / 2 + partition` and use the FFT convolver.
  Cheap to do; costs the property that the declared latency is derivable from
  the tap count alone, which is the thing that makes this node easy to reason
  about in a chain.
- A zero-latency convolver: run the first partition direct and the rest by FFT.
  This is the non-uniform partitioning `Convolver.h` deliberately avoids, and
  the header explains why — it is fiddly and a recurring source of bugs.

**Cost of being wrong:** a long render with a high tap count is slow, not
incorrect. Revisit when a real chain wants more than ~2047 taps.

### `zero_phase_eq` stops being zero-phase below ~100 Hz at high Q

The reverse pass needs the filter's whole future, approximated by a lookahead
window whose length is measured from the cascade's own decay and capped at
24 000 frames (0.25 s). A 20 Hz bell at Q 12 rings for **5.3 seconds** at the
1e−12 criterion, so it gets 4.7% of the reach it needs and comes back with 6.6°
of residual phase below 100 Hz — better than the 19.3° the same filter has as
`parametric_eq`, but not the 1e−11° the node advertises. Measured in contract
test 23; the mechanism is in `findings.md`.

What would resolve it: nothing available. Raising the cap to 48 000 halves the
error and doubles the latency of every slow filter, and only a half-million-frame
reach makes it exact. The right fix is a planner-side rule that steers narrow
low-frequency moves to `parametric_eq`, in the same family as `validate.py`'s
existing "an extreme Q at 22 Hz rings" repair — but that repair currently keys on
`parametric_eq` and `dynamic_eq` only, and does not know these two node types
exist.

**Cost of being wrong:** a mastering move at 30 Hz with Q above ~4 is quietly a
minimum-phase-ish filter with a zero-phase label. It will still sound like the
band it asked for; it will not deliver the phase coherence that was the reason
to choose the node.

### The 20 kHz passband edge is a deliberate, stated loss

The 48 → 96 → 48 kHz round trip nulls at −194 dB at 1 kHz and −168 dB at 18 kHz,
then falls off: 22 kHz returns 12 dB down. Protecting 20–24 kHz would cost two to
fourteen times the conversion compute for content nobody can hear.

Both halves are asserted, so moving the edge shows up as an intentional change.
Flagged because it has already been mistaken for a bug twice.

---

## The manual edit path

### A hand edit is refused when it is out of range and quietly rewritten when it is not

`POST .../graphs/{gid}/edit` promises, in its own docstring, that a person's
number is never quietly replaced: "they typed a specific number, and quietly
storing a different one is the sort of thing that gets noticed three renders
later". That holds for the bound check, which refuses. It does not hold for
whole-graph validation, which runs afterwards and *repairs*.

Measured: setting an EQ band to +18 dB on a graph already carrying +2 dB of
boost returns 200 with the band stored at **10.8 dB**, because the total boost
budget scaled every boost by 0.60 to preserve the shape. The response carries a
`repairs` list saying exactly that, so it is not silent at the protocol level —
but nothing forces the UI to show it, and the inspector will read 10.8 next to a
field the user typed 18 into.

The repair behaviour is right for a planner and the refusal is right for a
person, and the endpoint currently inherits the planner's. The three repairs a
hand edit can actually trigger are the boost budget, the ringing-Q reduction and
the inserted delivery ceiling.

**What would resolve it:** decide whether a repairable finding on a *user edit*
should refuse with the repair as its explanation ("18 dB would put the graph 8 dB
over its boost budget; 10.8 is the most it can take") rather than apply it. That
is a product decision, not a code one, which is why it is here rather than fixed.

**What it costs to be wrong:** low severity, high annoyance. Nothing renders
wrong; the number in the box is not the number the user typed, and the first time
an engineer notices is the moment they stop trusting the inspector.

`backend/tests/test_api.py` asserts the current behaviour — that the scaling
happens and that `repairs` names it — so whichever way this resolves, the change
is visible as a test change rather than as a silent drift.

---

## Product and legal

### The hero's desktop layout was never verified

Chrome would not resize past 500 px via automation and `window.resizeTo` is
blocked, so the landing page has only ever been seen at mobile width in-session.

### Licensing action items outstanding

- Written commercial grant from Jatin Chowdhury, who explicitly offers non-GPL
  terms. Would unlock CHOW Tape.
- Written permission from Pardo's lab for SocialFX.
- Capture our own amp models and cabinet IRs rather than scraping.
- FTO search before public launch: Fractal Audio's amp-matching patents, and
  US 9,304,988 "automatic audio production using semantic data".
- Counsel review of the reference-mode ingest question. §114(b) protects
  soundalike *output* well; analysing a copyrighted master is unsettled.
- Counsel review of the artist-alias routing table before it ships.

### A model plan takes about two minutes, and the DoD says 45 seconds

Measured through the API on a real acoustic guitar excerpt: **128.7 s** for one
model plan, request *"warm it up a little and tighten the low end"*, one attempt,
no repairs. The gate passed and the chain is sound. The problem is only the wait.

Phase 1's definition of done is **p50 ≤ 45 s, p95 ≤ 120 s** prompt to audible.
A single sample is not a p95, but one sample already exceeding the p95 target
means the budget is wrong somewhere rather than merely tight.

Where the time goes is not yet measured, and that is the next step rather than a
guess. Two stages run serially — Opus reasoning unconstrained, then Sonnet
transcribing — and the reasoning stage is high-effort by design because that is
the half that earns the model its 200-700x cost premium over the DSP.

Candidate moves, none taken yet:
- Time the two stages separately. If EMIT is a meaningful share, it is pure
  transcription and can be a smaller model or a smaller prefix.
- The known per-model cache defect below is worth roughly 31k tokens of
  prefill on the EMIT call. Fixing it is a latency win as well as a cost win.
- Stream stage 1, so the user reads the reasoning as it arrives instead of
  watching a spinner. This is the SSE work the plan already schedules, and it
  changes the *felt* latency without changing the number.

Mitigated in the UI rather than hidden: the button counts seconds up, and says
a model plan takes a minute or two before it starts. Do not let that stand in
for fixing it.


### The limiter is 0.07 dB permissive, and the fix that "verified" it did not

Reopened, with a better instrument. `true_peak_limiter` was reported fixed —
from −0.996 dBTP (permissive) to −1.041 (conservative) against a −1.0 ceiling.
That verification used the measurement suite's polyphase reference, and the
polyphase reference under-reads.

Measured against an EXACT reference — spectral zero-padding, which has no
interpolation grid and no filter, validated to **1.18e−09 dB** against the
analytic fact that a coherent unit sinusoid is exactly 0 dBTP:

| meter | reads |
|---|---|
| engine's own detector | −1.0419 dBTP |
| polyphase reference (32×, 6145 taps) | −1.0209 |
| **exact spectral reference** | **−0.9284** |

So the delivered true peak is **0.07 dB above the ceiling**, the engine's
detector is **0.11 dB low**, and the reference that was supposed to catch that
is **0.09 dB low**. Three meters, two of them agreeing with each other because
they share a grid, and the one that disagreed was the one without one.

**What is left is the transition-band term**, not the grid term. The grid term
was fixed by raising the oversampling to 16× and subtracting the analytic bound
`−20log10(cos(π/2f))`. The remainder comes from the interpolator's passband
ending at 0.475·fs, is unbounded and signal-dependent, and narrowing it costs
10–39× runtime for an improvement only on content near Nyquist.

**Resolving it needs a decision, not more measurement.** Either the limiter
subtracts a further fixed margin — cheap, but it costs every render that
loudness whether or not the material has energy near Nyquist — or the detector's
transition band narrows and the limiter gets slower. The stress signal that
shows it is deliberately adversarial: an fs/4 tone at 45°, which is the textbook
worst case, normalised so the sample peak sits at exactly 0 dBFS. Ordinary
programme material is nowhere near it.

**Cost of being wrong:** a −1.0 dBTP delivery reads −0.93 on a conformance meter
that is itself exact. Nothing clips. It matters if a user sets the ceiling to
0.0 dBTP, which nothing currently stops them doing.

Recorded as a known failure in `engine/tests/measurements.baseline.json` rather
than fixed by moving the threshold, and the conformance metric now uses the
exact reference so the number cannot go back to looking fine.

---

## Resolved

Kept so the reasoning is not re-litigated.

| Was | Resolution |
|---|---|
| The true-peak limiter overshoots its ceiling by 0.02 dB | Fixed, and the diagnosis in the original entry was only half right. The detector's error has **two** terms, not one. Term A is the peak-search grid: it evaluates the reconstruction at `factor` points per sample and returns the largest, so it can only read LOW, by at most `cos(π/(2·factor))` — a theorem about band-limited signals, 0.688 dB at the 4× the node was running. Term B is the transition band, which is unsigned and unbounded. The header's claim that "oversampling factor does not affect accuracy at all" was measured on stationary signals, where phase diversity over hundreds of cycles hides term A entirely; on an isolated transient it is worth up to 0.22 dB. Factor floor raised 4 → 16, and the limiter now subtracts `gridBoundDb()` from its ceiling unconditionally rather than leaving `safety_margin_db` at 0 and hoping. Delivered true peak against a −1.0 ceiling, measured by exact DFT-domain reconstruction: −0.996 → **−1.041 dBTP** on band-limited programme, i.e. from permissive to conservative. Cost 4× the detector, 1.02 s → 4.07 s per 30 s of stereo; the measurement suite went 29.1 s → 31.5 s. Eight mutations, eight caught after two of them exposed real gaps — an untested default path and a reference meter that shared the grid it was checking. Term B remains and has its own entry above. |
| The codec detector is unstable near its own threshold | Fixed. The confidence read the slope alone, which is the strength of the evidence and not the certainty of the verdict. It is now the weaker of two margins — slope past the gate that actually binds, and depth inside the 60 dB guard — so it falls to zero as either boundary is approached. The demo vocal chain's output went from 0.88 to 0.15; the seven real lame/AAC encodes of the acoustic guitar all still report 1.00. Ten mutations, ten caught. |
| Stereo width unmeasurable — every stem was mono or dual-mono | Two genuinely stereo files were already in the user's library. `stereo.correlation` tracks width at r = −0.99. |
| "Drive does not brighten" | Wrong. An artifact of full-band pink noise plus a summary statistic; on real stems 12 dB of drive moves the 8 kHz band by +33.5 dB on a bass. |
| "The report cannot localise mid-band EQ" | Wrong. The analysis was dropping band lists; the bell is recovered at +0.98 dB per dB, r = 1.00. |
| Bisect the compressor ratio on loudness range? | No. LRA is not monotonic in ratio on real material — it rises under heavy compression, because pumping is variance. Gain reduction at the loud percentile is monotonic and is the control variable instead. |
| Derive attack and release from the modulation-spectrum corner? | No. Compression adds modulation at those rates rather than removing it, so there is no corner. Timing comes from the source's own rate. |
| Does the prompt cache actually work? | Yes. Measured 30,938 cached tokens read on a repeated request, against a 15,000 threshold. The prefix turned out 68% larger than the character-count estimate — it is mostly C++ and JSON, which tokenise denser than prose — so the saving is bigger than budgeted, not smaller. |
| Are the schema's enums enforced on the wire? | Yes, with a caveat. Asked directly for a node type outside the closed set, the model emitted no such node. It returned an empty node list rather than being visibly blocked, so this is evidence the constraint holds rather than proof of how it is implemented. |
| The homepage promised source separation | Fixed. `app/page.tsx` said "separates, restores and remasters"; v1 does none of the separation. Found by the CI structural gate on its first run, which is what that gate is for. |
| Should we source evaluation audio from YouTube? | No. Infringement, and it would undo the licensing posture at the point where every quality claim rests. See `licensing.md`. |
