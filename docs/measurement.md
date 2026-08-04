# The measurement report

`backend/measure/`. This is the highest-leverage artefact in the system: it is
what the planner is handed *instead of* the audio.

The finding the whole design rests on is that a model given measured DSP features
scored 0.22 on EQ matching where the same model guided by audio embeddings
scored 0.55 — worse than random parameters. The model is never asked to imagine
what the audio sounds like.

```python
from backend.measure.report import measure_file
report = measure_file("stem.wav").to_dict()
```

```sh
python -m backend.measure.report stem.wav        # prints the JSON
```

## Three rules

These separate a report that helps from one that misleads.

**Units live in the key name.** `threshold_db`, not `threshold`. A model reading
a bare number will assume a unit and will sometimes assume the wrong one. Costs
nothing, removes a whole class of silent error. Asserted by a test.

**Every estimate carries a confidence.** Blind reverb time, noise floor and
bandwidth limit are *inferences*, and they are unreliable on exactly the material
where they matter most. The prompt instructs the model to disregard anything
below 0.4, and a validator enforces that low-confidence values never drive a
decision.

**The report ends in `warnings`.** Some facts need three fields read together —
a bandwidth limit at 15 kHz means an air boost does nothing; a 0.4% clipping rate
means drive will make it worse. Leaving that inference to the model leaves it to
chance. The warnings are computed deterministically and stated as instructions,
not numbers.

A missing measurement is `null`, never a substituted constant. A fabricated
number is indistinguishable from a measurement once it is in the report, and it
is exactly the kind of value a planner will do arithmetic on.

## Loudness gating governs everything time-averaged

The most common way a measurement report lies is by averaging over silence.

A vocal stem is 20–60% silence, and that silence is not empty: it is room tone,
headphone bleed, preamp hiss, the tail of the previous phrase. Averaging over
the whole file divides the programme bands by four while leaving a continuous
noise floor untouched — about **6 dB of tilt toward the bottom**. The spectrum
that comes out is the source's own, read muddy.

A planner given that prescribes a low cut for a voice with no low end, spends a
band achieving nothing, and leaves the real problem alone.

So the same BS.1770 gate that defines integrated loudness defines which parts of
the signal every other measurement may look at. One definition of "where the
programme is", used everywhere.

**What gating does not do** is remove noise that coexists with the signal.
Continuous rumble is present under the programme as well as between it; expecting
a level threshold to remove it is expecting noise reduction. What it fixes is the
*shape*.

`source.gated_coverage_percent` is reported because it is diagnostic in its own
right: a report over 12% of a stem should be read differently from one over 95%.

## Sections

### `source`

`channels`, `sample_rate_hz`, `duration_s`, `bit_depth`, `is_float`,
`gated_coverage_percent`.

### `loudness`

| Field | Notes |
|---|---|
| `integrated_lufs` | BS.1770-5, gated. `null` for silence or files under 400 ms. |
| `loudness_range_lu` | EBU Tech 3342: 95th minus 10th percentile of gated short-term loudness. |
| `true_peak_dbtp` | BS.1770 Annex 2, 8× at ≤48 kHz and 4× above. |
| `sample_peak_dbfs` | Reported alongside so the difference is visible. |
| `mono_convention_note` | Present only for mono files. |

**Mono reads 3 LU below dual-mono stereo.** BS.1770 sums weighted channel
powers, so a mono guitar DI at −23 dBFS reads −26 LUFS. This is the standard
behaving as specified, and it is the number an engineer is most likely to think
is a bug — which is why the report says so in words and a test asserts it.

**True peak is not sample peak, and the difference matters.** Between two samples
the reconstructed signal routinely exceeds both, by several dB on heavily limited
material — which is the material most likely to be sitting at a ceiling. A file
that "peaks at −0.1 dBFS" can clip on playback. One of the reference stems has a
sample peak of −0.00 dBFS and a **true peak of +0.52 dBTP**.

Validated against **EBU Tech 3341 and 3342 conformance signals** — published
inputs with the exact reading a compliant meter must produce. This is the only
part of the measurement layer whose correctness does not rest on our own
reasoning.

K-weighting is derived from analog parameters rather than the standard's 48 kHz
coefficient table, because reusing that table at 96 kHz moves both corner
frequencies by an octave. Asserted across five rates from 44.1 to 192 kHz.

### `dynamics`

| Field | Notes |
|---|---|
| `crest_factor_db` | Peak minus RMS over gated material. **See the caveat below.** |
| `peak_to_loudness_ratio_db` | True peak minus integrated. ~8 dB is a mastered pop record; 18–20 is an unprocessed acoustic performance. |
| `short_term_rms_percentiles` | p5…p99 in dBFS over 50 ms windows. The direct input to compressor threshold selection. |
| `per_erb_crest_db` | 12 ERB-spaced bands. |
| `envelope_modulation` | `centroid_hz`, `p90_hz`, `depth_db`, `confidence`. |

**Crest factor is not a reliable compression indicator on real material.** It
passes on synthetic steady noise and fails on all six reference stems. Crest is
peak minus RMS and peak is a *single sample*; a 10 ms attack deliberately passes
the initial transient, so crest **rises** on percussive material and falls on
sustained. The sign depends on the source.

Use `envelope_modulation.centroid_hz` instead: compression reduces slow, large
level excursions more than fast ones, shifting modulation energy upward
monotonically with ratio at **r = 0.94 across every stem**.

**Broadband crest hides the thing you most need to know.** A mix whose low end is
squashed flat and whose top is spiky reads as moderately dynamic overall — the
two cancel. `per_erb_crest_db` shows the actual shape, and it is the difference
between prescribing broadband compression and prescribing a multiband move.

Percentiles are used rather than mean-and-peak because a threshold is a decision
about what fraction of the material to catch. "The 90th percentile is −12 dBFS"
answers that directly.

50 ms windows sit between the two things they must not be: shorter tracks
individual waveform cycles at bass frequencies, so a 60 Hz note reads as
violently dynamic; longer averages across a whole note, hiding the attack.

### `spectrum`

| Field | Notes |
|---|---|
| `third_octave_db` | ANSI S1.11 nominal centres, 25 Hz to 20 kHz. What an engineer thinks in. |
| `erb_db` | 28 ERB-spaced bands. What the ear resolves. |
| `tilt_db_per_octave_re_pink` | **0 = pink, +3 = white.** See below. |
| `tilt_residual_db` | RMS deviation from the fitted line. |
| `bandwidth_limit_hz` | A codec wall, or `null`. |
| `bandwidth_limit_confidence` | How far the reading is from flipping, **not** how steep the cliff is. See below. |
| `hf_rolloff_db_per_octave` | Slope above 6.3 kHz. A tonal fact, not a defect. |

**The tilt sign convention is named in the key for a reason.** These are
band-integrated levels and fractional-octave bands widen with frequency, so flat
PSD (white noise) puts 3 dB more energy in each successive octave and reads as
+3. Pink — equal energy per octave, what an engineer means by neutral — reads as
0. This is the convention an RTA displays. A planner assuming the other one
inverts every tonal move it makes.

**`bandwidth_limit_confidence` measures distance from the decision.** The verdict
rests on two premises with two thresholds — the cliff is steep enough, and there
was still something above it worth cutting off — and the confidence is the weaker
of the two margins, so it falls to zero as either boundary is approached. It read
the slope alone once, and reported 0.88 on a verdict that a 1 dB midrange bell
flipped; see `findings.md`. Real lossy material is unaffected: seven lame and AAC
encodes of a full-band acoustic guitar all report 1.00. The planner acts on
anything above 0.4 by refusing every air-band move, so anything near a boundary
must not clear it.

**Residual rather than r-squared.** R-squared measures the variance the *slope*
explains, so perfectly pink noise — flat, zero slope, the best case there is —
scores near zero and would be flagged unreliable on the most common good input.

**`bandwidth_limit_hz` is `null` for most real audio, and that is the point.** A
dark source is not a bandwidth-limited one, and the difference decides whether
brightening is possible or pointless. Measured on a 1/12-octave grid:

| | steepest rolloff |
|---|---|
| bass, vocal, acoustic guitar | 3 – 37 dB/octave |
| MP3 at 128, 96, 64 kbps | 425 – 443 dB/octave |

An order of magnitude with nothing in between, because one is a source running
out of energy and the other is a brickwall filter. The threshold sits at
150 dB/octave and is validated against real LAME encodes.

Deliberately conservative: a 128 kbps encode of a *dark* vocal is not flagged,
because its cutoff sits where the vocal had nothing anyway. A missed wall above
the source's own content costs nothing; a wall reported where none exists forbids
a move that would have worked.

### `stereo`

`type` (`mono` \| `dual_mono` \| `near_mono` \| `conventional_stereo` \| `wide` \|
`out_of_phase` \| `silent`), `type_confidence`, `correlation`, `mono_loss_db`,
and `per_band` with `centre_hz`, `correlation`, `side_to_mid_db`, `mono_loss_db`
for 8 ERB-spaced bands.

**`mono_loss_db` measures loss beyond the arithmetic 3 dB.** Summing two
uncorrelated channels of equal power gives 3 dB less than their power sum, and
that 3 dB is not a loss — it is arithmetic. Reporting it as one would flag every
normally-wide mix as broken. Zero here means uncorrelated or identical; large and
positive means real cancellation.

Reported per band because the damage is never broadband. Low-frequency side
content costs headroom, cuts badly to vinyl and partially cancels on
single-speaker playback; the same side energy at 8 kHz is harmless and desirable.

A silent band returns correlation 1.0, not 0. Silence is not decorrelation, and
returning 0 would read as "extremely wide" for a band containing nothing.

### `hygiene`

`clipping` (`clipped_sample_percent`, `longest_run_samples`, `runs`),
`dc_offset_dbfs` per channel, `noise_floor_dbfs`, `noise_floor_confidence`,
`dead_channels`, `non_finite_samples`.

**Clipping requires a FLAT top, not merely a loud one.** A sine is quadratic near
its peak, so a clean 100 Hz tone at 0 dBFS holds about five consecutive samples
within 0.0005 of full scale purely from geometry. A run-length test alone reports
2% clipping on undamaged audio — and fires the "do not add drive" warning on
material that would take drive perfectly well. A run counts only if its
peak-to-peak spread is a negligible fraction of its level, measured relative to
the run's own level so material clipped and then attenuated is still caught.

**Noise floor comes with low confidence when there is no quiet passage.** A
wall-to-wall loud master genuinely does not reveal its floor, and the honest
answer is "cannot tell" rather than a number derived from the quietest moment of
a loud signal.

### `warnings`

Ordered worst-first, so a truncated read still sees what matters.

| Code | Severity | Fires when |
|---|---|---|
| `non_finite_samples` | critical | NaN or Inf present |
| `out_of_phase` | critical | strongly negative correlation |
| `no_programme_material` | critical | nothing survived the gate |
| `clipped_source` | high | >0.01% in flat-topped runs |
| `dead_channel` | high | a channel carries nothing |
| `bandwidth_limited` | high | a codec wall at confidence ≥0.4 |
| `mono_incompatible` | high | >1 dB loss beyond the arithmetic 3 |
| `dc_offset` | medium | above −50 dBFS |
| `high_noise_floor` | medium | above −55 dBFS at confidence ≥0.4 |
| `already_compressed` | medium | PLR below 8 dB |
| `sparse_material` | medium | gated coverage below 25% |
| `dual_mono` | low | channels identical |
| `tilt_incomplete` | low | band levels deviate >6 dB RMS from the fit |

Each message names what was measured *and* what to do about it. A warning the
reader has to interpret is a warning the reader can interpret wrongly.

## Audio I/O

`backend/audio/wavio.py` mirrors `engine/include/anv/io/WavFile.h` deliberately.

`soundfile` wraps libsndfile, which is LGPL — the dependency the C++ side was
written to avoid. `scipy.io.wavfile` is BSD and would be fine on licence grounds,
but then the engine and the analysis would be two independent opinions about what
a WAV file means. That disagreement does not crash: it makes the report describe
subtly different samples than the renderer processed, so every number in it is
quietly wrong about the audio the user hears. The two decoders are tested against
each other.

Integer scaling is 2^(n−1) in both directions, making read→write bit-exact for
every code. Float is **not** clamped: a mixdown handed to us hot is ordinary and
clamping would destroy headroom left deliberately.

8-bit WAV is unsigned, alone among the integer depths. Reading it signed produces
audio that is inverted and offset by half scale — and still looks like a waveform.

## Performance

A four-minute stereo file takes about 17 s, dominated by 20 zero-phase band
filters. Under the excerpt-first default of 25 s that is about 1.7 s, which is
what the product will actually run.
