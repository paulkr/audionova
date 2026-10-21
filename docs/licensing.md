# Licensing

This is a commercial subscription product, so every dependency and every piece of
audio in the evaluation corpus has to be usable commercially. The posture is
deliberate and enforced rather than assumed.

**Allowlist:** MIT, BSD-2/3, Apache-2.0, ISC, Zlib, BSL-1.0, MPL-2.0
(unmodified), Unlicense, CC0.

**Denylist:** GPL-\*, AGPL-\*, LGPL-\*, SSPL, CC-BY-SA, CC-BY-NC, unlicensed.

## What we deliberately did not use

Several dependencies that look permissive are not, and several that are
permissive still cost more than they save.

| Rejected | Why |
|---|---|
| **libsndfile** | LGPL, *and* a parser exposed to bytes we did not create — historically the most productive category of RCE there is. WAV I/O is hand-written instead (~500 lines). Dropping it means the engine links nothing that touches untrusted input and carries no copyleft. |
| **soundfile / librosa** (Python) | Both wrap libsndfile. Same problem, other language. |
| **scipy.io.wavfile** | BSD and fine on licence grounds. Rejected for a *correctness* reason: it and the engine would be two independent opinions about what a WAV file means, and that disagreement makes the measurement report describe subtly different samples than the renderer processed. One decoder semantic, expressed twice and tested against each other. |
| **pyloudnorm** | MIT and correct for BS.1770, but it does **not** implement true peak — it reports sample peak under a name that reads like true peak, which is the most consequential measurement in the export path. |
| **Essentia** | AGPL-3.0 — the one copyleft licence whose network clause fires for SaaS. Its pretrained models are separately non-commercial. |
| **Demucs weights** | Research-only. The author states this and MUSDB18 is `other-nc`, despite the MIT badge on the code. |
| **madmom models** | CC-BY-NC-SA. |
| **open-unmix umxl** | Non-commercial weights. |
| **JUCE** | Current master is AGPL-3.0. |
| **pedalboard** | GPL-3. |
| **Text2FX, InstructFX2FX, word-eq** | No licence file at all, which means all rights reserved, not free. |
| **MUSDB18** | Non-commercial. It is the obvious evaluation corpus and cannot be used. |
| **TONE3000** | Terms explicitly prohibit commercial redistribution and building competing services. Capture our own amp models and cabinet IRs instead. |
| **r8brain** | MIT and would have been fine. Written ourselves because it is ~200 lines given a Kaiser designer we already had, and owning it means the conversion is specified by us. |
| **nlohmann/json** | MIT and excellent. Written ourselves because this parser is the boundary where model output becomes something that allocates memory and configures DSP, and that boundary is worth being able to read end to end. |

## The two character nodes were written rather than vendored

`tape` and `console` are the first nodes in this engine whose obvious
implementation already exists in public, and the two cases are different enough
to be worth separating.

**`tape`.** The best open tape emulation available is **GPL-3.0** and therefore
unusable here. Nothing from it, or from
any other implementation, was read, transcribed or paraphrased. What the node
models comes from the recording chain's own published physics: an anhysteretic
magnetisation curve, the playback head's contour effect, gap loss as the
transform of the gap's spatial window, and transport speed error. Every
derivation is written out in `engine/include/anv/nodes/Tape.h` so a reader can
check the arithmetic rather than trust the provenance.

The one place this bit is the hysteresis model. Jiles-Atherton is published
physics rather than anyone's code and could have been implemented from the
literature — and it was rejected, partly on licence-independent grounds (it is
an ODE, so it has no closed-form second antiderivative and the engine's ADAA
antialiasing is unavailable to it) and partly because its five constants are
properties of a tape formulation we have not measured. The node says plainly
that it models the anhysteretic curve and not the loop.

**`console`.** The best-known open implementation is **MIT** and could legally
have been vendored. It was not, and the reason is the same one recorded against
`r8brain` and `nlohmann/json` in the table above: this is a node whose entire
value is a sub-decibel claim about frequency response and phase, and owning it
means the claim is specified by us and testable against arithmetic we wrote
down. The node is derived from two circuit facts — an amplifier's headroom and
its two first-order poles — and both are asserted in the contract suite against
expressions expanded longhand.

**No coefficient tables anywhere.** Where a number appears in either node it is
either derived in the header (the sinc's −3 dB point at 1.3915576, the
Catmull-Rom slope bound of 3.5, the quintic's harmonic ratios of 1/6 and 1/50)
or it is an exposed parameter with a documented default. Nothing was
transcribed.

## Current dependency set

**C++:** none. The engine is header-only C++20 with no third-party code.

**Python:** `numpy` and `scipy` (BSD-3-Clause) for everything that touches
audio; `pydantic` (MIT) for the graph schema; `anthropic` (MIT) for the model
client; `fastapi` (MIT), `uvicorn` and `starlette` (BSD-3-Clause), `httpx`
(BSD-3-Clause) and `python-multipart` (Apache-2.0) for the local API. Every transitive dependency is
recorded in `ci/licences.lock.json` and gated by `ci/check_licences.py`. See
`backend/requirements.txt`, which documents the rejections above alongside the
acceptances so the reasoning travels with the file.

## Model weights need a separate register

SPDX tooling does not track them, and weights are where the landmines are —
`madmom`, `open-unmix`, Demucs and Essentia all ship permissive *code* with
non-commercial *weights*. Any classifier added in the CLASSIFY stage goes in that
register before it goes in the build.

## Planned CI gates

Not built yet. Each is a red build rather than a checkbox:

- **Undeclared-vendoring check** — nothing appears in the tree without a recorded
  licence.
- **Licence-drift check** — hash each vendored LICENSE against a recorded
  sha256. This is precisely the JUCE-went-AGPL scenario: an upstream relicence
  becomes a failing build rather than a legal incident discovered at diligence.
- ScanCode / `reuse lint` tree scan, `pip-licenses`, CycloneDX SBOM.
- Eigen-unmodified check, if Eigen is ever added (MPL-2 is file-level copyleft).

## Audio in the repository

`audio/stems/` is **gitignored**. The stems are the user's own files; their
licence status is theirs to decide, not something a commit should assume.

### Why the evaluation corpus is held to a higher standard than the rest

The evaluation set is what every future quality claim rests on. It is the last
place to introduce unlicensed material, because a licence problem there
invalidates the measurements rather than just the shipping binary.

Two consequences that came up in practice:

**No YouTube-sourced audio.** Downloading commercial recordings is infringement,
and it would undo the posture above at the point where it matters most. If more
material is needed, the legitimate sources are Freesound's CC0 tier, Wikimedia
Commons, Archive.org's public-domain collections and the Free Music Archive's
CC-BY tier — all commercially usable with attribution recorded.

**No source-separated stems**, and this one has a technical reason as strong as
the legal one. Separator output carries artifacts: one candidate vocal stem
measured as `out_of_phase` at correlation −0.32, which no real vocal recording
does, because separation produces phase-incoherent output. Material that has been
through a lossy encode, then separation, then another lossy encode is several
generations from anything a user will upload. And v1 explicitly does not do
separation — the product takes real tracks and stems — so calibrating a planner on
separator output would tune it to artifacts the real input does not have.

## The project's own licence

Audionova is released under the Apache License 2.0 (see `LICENSE` at the
repository root). It is permissive and carries an explicit patent grant, and it
is compatible with everything on the allowlist above — which is the other reason
the allowlist is enforced: a single copyleft dependency would change what that
licence means for anyone building on this code.

## Trademark safety is structural, not a prompt request

Users may type any artist name; our catalog, UI and marketing contain none.

**This is the design, and none of it is built yet**, because there are no presets
to route to. A `routing_table.yaml` will map aliases to preset IDs and be
fuzzy-matched **before** the model call. A `preset_id` field in the emitted
schema will be a closed enum, so constrained decoding makes it impossible for the
model to put an artist name into the graph. A post-generation filter will scan
the free-text explanation against a denylist.

What exists today is only the weaker half: the planner's system prompt tells the
model never to name an artist, band, producer or commercial recording
(`backend/plan/prompt.py`). That is a prompt request, which is exactly what the
heading says is not enough, and it is why the structural version has to land
before presets do.

When presets exist, the UI will show each one's display name plus its citation
list. Showing your
sources is both a product differentiator and a legal shield.
