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


---

## Resolved

Kept so the reasoning is not re-litigated.

| Was | Resolution |
|---|---|
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
