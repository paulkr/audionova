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

## Current dependency set

**C++:** none. The engine is header-only C++20 with no third-party code.

**Python:** `numpy` and `scipy`, both BSD-3-Clause. That is the entire list. See
`backend/requirements.txt`, which documents the rejections above alongside the
two acceptances so the reasoning travels with the file.

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

## Action items

- Written commercial grant from Jatin Chowdhury, who explicitly offers non-GPL
  terms. Would unlock CHOW Tape, the best open tape emulation available.
- Written permission from Pardo's lab for SocialFX.
- Capture our own amp models and cabinet IRs.
- FTO search before public launch: Fractal Audio's amp-matching patents, and
  US 9,304,988 "automatic audio production using semantic data".
- Counsel review of the reference-mode ingest question. §114(b) protects
  soundalike *output* well; analysing a copyrighted master is unsettled.
- Counsel review of the artist-alias routing table before it ships.

## Trademark safety is structural, not a prompt request

Users may type any artist name; our catalog, UI and marketing contain none.

A `routing_table.yaml` maps aliases to preset IDs and is fuzzy-matched **before**
the model call. The `preset_id` field in the emitted schema is a closed enum, so
constrained decoding makes it impossible for the model to put an artist name into
the graph. A post-generation filter scans the free-text explanation against a
denylist.

The UI shows each preset's display name plus its citation list. Showing your
sources is both a product differentiator and a legal shield.
