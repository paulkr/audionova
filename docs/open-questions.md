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

### Twenty glossary terms are missing

Forty of the sixty the plan calls for. The missing ones are the genre- and
instrument-specific vocabulary, which should be authored alongside the presets by
someone who mixes for a living rather than guessed at here.

---

## Unverified because nothing has run yet

### No live model call has ever been made

`AnthropicClient` follows the documented API shape and has never executed. Two
things cannot be checked offline and should be the first things checked with a
key:

1. **`cache_read_input_tokens > 15000` on a repeated request.** If it is near
   zero, something per-request has leaked into the system prefix and the bill has
   quietly multiplied. The prefix is ~18.4 k tokens, so the saving is the
   difference between a viable product and one that loses money per turn.
2. **Structured output honours the schema's enums.** Numeric bounds are
   documented *not* to be enforced and we compensate for that; enums are
   documented to be enforced, and that is worth confirming rather than assuming.
   If they are not, the closed node-type set — which several safety properties
   rest on — is not actually closed.

### Cross-platform determinism is asserted but untested

The contract is bit-exact within `(engine_version, arch)` and −140 dBFS across
architectures. **Everything has only ever run on one machine.** The −140 dB
threshold was chosen because libm ULP differences and SIMD summation order make
bit-exactness across x86/ARM a losing fight — but that reasoning has never been
checked against an actual second architecture.

Resolving it needs CI on both, which does not exist.

### There is no CI

Every suite is run by hand. The plan calls for a licence-drift gate, an
undeclared-vendoring check, an SBOM, and a prompt-cache canary — none exist. The
licence-drift check is the one with teeth: it is exactly the JUCE-went-AGPL
scenario, where an upstream relicence should become a red build rather than a
legal incident found at diligence.

---

## Measurement gaps

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

The recovery baseline scores **+0.498** on ordering while carrying *no ordering
information at all* — it emits a fixed conventional sequence, so the number
measures how conventional the corpus is.

The corpus deliberately contains `eq_then_compress` and its exact reverse. **If no
planner can separate them, measurement deltas do not carry ordering and the
metric should be retired rather than chased.** That is a decision to take
deliberately when there is a planner to test, not a bug to fix now.

### Gain staging cannot model what it is bounding

The check sums static gains, the largest EQ boost and makeup against the measured
p95. It cannot model compression or saturation, so it is a **bound rather than a
prediction**. It catches a chain arriving 15 dB hot; it will not catch a chain
that arrives 4 dB hot through a compressor's makeup interacting with a drive
stage.

### No proxy metric has been calibrated against human judgement

The plan is firm: **no metric gates CI until it passes ρ > 0.6 against human MOS**,
and parameter MAE is never a quality gate — the regression baseline in LLM2Fx had
the best MAE and nearly the worst perceptual score.

Nothing here has been calibrated, because there is no human data. Every number in
`evaluation.md` should be read as "this measures what it says it measures", not as
"this predicts quality".

---

## Performance and architecture

### The engine is invoked per render as a subprocess

pybind11 is planned. The subprocess costs a few milliseconds against a render
measured in seconds, and buys the property that Python exercises exactly the
binary that ships. Worth revisiting when throughput matters, not before.

### Measurement is slow on long files

A four-minute stereo file takes about **17 s**, dominated by 20 zero-phase band
filters. Under the excerpt-first default of 25 s that is ~1.7 s, which is what the
product will actually run. Not worth optimising until the planner reveals which
numbers it reads.

### The 20 kHz passband edge is a deliberate, stated loss

The 48 → 96 → 48 kHz round trip nulls at −194 dB at 1 kHz and −168 dB at 18 kHz,
then falls off: 22 kHz returns 12 dB down. Protecting 20–24 kHz would cost two to
fourteen times the conversion compute for content nobody can hear.

Both halves are asserted, so moving the edge shows up as an intentional change.
Flagged because it has already been mistaken for a bug twice.

---

## Product and legal

### The homepage promises something v1 does not do

`app/page.tsx:57` says the product "separates, restores and remasters". **v1 does
not do source separation** — it takes single tracks and stems. The plan flags
fixing this in week 1 and it has not been fixed.

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

---

## Resolved

Kept so the reasoning is not re-litigated.

| Was | Resolution |
|---|---|
| Stereo width unmeasurable — every stem was mono or dual-mono | Two genuinely stereo files were already in the user's library. `stereo.correlation` tracks width at r = −0.99. |
| "Drive does not brighten" | Wrong. An artifact of full-band pink noise plus a summary statistic; on real stems 12 dB of drive moves the 8 kHz band by +33.5 dB on a bass. |
| "The report cannot localise mid-band EQ" | Wrong. The analysis was dropping band lists; the bell is recovered at +0.98 dB per dB, r = 1.00. |
| Should we source evaluation audio from YouTube? | No. Infringement, and it would undo the licensing posture at the point where every quality claim rests. See `licensing.md`. |
