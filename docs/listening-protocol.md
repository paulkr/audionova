# Running a listening panel

Everything a person needs to run a blinded panel from start to finish. No code
reading required. If you want to know *why* any of it is shaped this way, that
is `backend/listen/`'s module docstrings and the "blinded listening harness"
section of [findings.md](findings.md); this document is the recipe.

Commands are run from the repository root with the project's virtualenv:

```sh
PY=.venv/bin/python
```

Everything below needs real audio, which is gitignored (see
[licensing.md](licensing.md)), and the C++ engine built:

```sh
make -C engine render        # produces engine/build/anv_render
```

---

## 0. What you are about to run, and what it costs

| | |
|---|---|
| Panel size | three engineers is the Phase 0 gate's figure |
| Session length | 30 trials ≈ **30–45 minutes** per listener, plus the screen |
| Screen | 10 trials, 2–3 minutes, taken once per listener before the session |
| Audio served | ~180 MB per listener at 10-second stereo excerpts |
| Prep time | about 1 minute of rendering plus 35 seconds of matching per 24 comparisons |

**There is no pause, no resume and no going back.** A listener who reloads the
page loses every answer and starts the session again. Book an uninterrupted
block, and say so when you book it.

**Nothing about the panel calibrates a proxy metric.** The correlation step
refuses any result resting on fewer than fifteen ratings per point, and a
three-person panel gives three. Calibrating a proxy is a separate, larger
exercise — see the `MIN_N` entry in [open-questions.md](open-questions.md).

---

## 1. Produce the audio you want compared

**The harness does not render anything for you.** It takes files that already
exist. For every item you want judged you need three files at the same sample
rate, the same channel count and the same length:

- the **source** — use a *null render* (an empty recipe through the engine), not
  the raw file. Against the raw file, the engine's own sample-rate conversion
  ends up inside every difference a listener hears; `findings.md` records a
  −0.5 dB gain reading −0.62 dB for exactly this reason.
- **two processed versions**, one per condition. Anything: our render against an
  engineer's own chain, two planners, two settings.

Excerpts of **8–12 seconds** work well. Longer costs download time and fatigue;
shorter does not give a listener room to hear a decay.

Write them as float WAV — the harness quantises to 24-bit itself, after it has
applied its gains, so a PCM intermediate would clip a boost before the matcher
could bring it under the ceiling.

Then write a manifest, which is a JSON list:

```json
[{"item": "guitar-di-03",
  "request": "warmer, less boxy",
  "source": "work/03-null.wav",
  "conditions": {"audionova": "work/03-anv.wav",
                 "engineer":  "work/03-eng.wav"}}]
```

`request` is shown to the listener, because one of the two scales asks whether
the render did what was asked. `item` and the condition names are never shown.

Paths are used as given, so relative paths are relative to where you run the
build.

## 2. Build the session

```sh
$PY -m backend.listen.run_session build manifest.json out/ \
      --seed 20260804 --repeats 3 --listeners alice bob carol
```

- `--seed` is required and is the whole reproducibility story. Write it down.
  The same seed and the same manifest rebuild the same session in a new process.
- `--repeats N` picks N comparisons to present twice, to each listener, at least
  three trials apart and with the sides independently re-randomised. This is the
  only measurement of whether a listener is answering or clicking. **Use at
  least 3, and prefer 6** — see step 6.
- `--listeners` are ids the listeners will pick from a list. Real names are
  fine; nothing about them is secret.
- Trials per listener = comparisons + repeats. `--repeats 6` over 24 comparisons
  is a 30-trial session.

The build prints one line per comparison and refuses to write anything if any
comparison fails to loudness-match. The `matched to 0.0000 LU` it prints is the
spread **after** matching and is 0.0000 on healthy material — it is a
pass/fail indicator, not a measurement of how much work was needed. The work is
in `out/key.json` under `loudness[...].gain_db`; on a real run those gains
spanned up to **7.3 dB** within a single comparison, which is what an unmatched
session would have been measuring instead of preference.

Two files come out:

- `out/session.json` — safe. Contains no condition name, no item name, no file
  path, no repeat flag.
- `out/key.json` — **the answer key.** The server reads it and never sends it.
  Do not copy it anywhere a listener can reach, and do not put the session
  directory behind a static file server.

## 3. Build and run the screen

The screen asks which of two clips has a 2 dB Q=1 bell at 3 kHz. It exists to
separate "did not prefer it" from "could not hear it". The bar is 8 of 10.

```sh
$PY -m backend.listen.run_session screen source.wav screen/ --seed 20260804
```

It cuts ten consecutive 8-second segments **from the start of the file**, so the
source needs 80 seconds of continuous programme material with no silent lead-in.
A silent segment stops the build with a message naming the trial and its time
range; the fix is a different source or a pre-trimmed one — there is no
`--offset`.

> **Known gap.** There is no server, no interface and no CLI for *taking* or
> *scoring* the screen. `screen/screening.json` lists the trial ids and clip
> ids, `screen/clips/` holds the audio, `screen/screening_key.json` holds the
> answers, and joining those to a listener currently means writing your own
> player and calling `backend.listen.screen.score_screening` yourself. Budget
> for that, or run the screen informally and record the result by hand.

## 4. Serve the session

```sh
$PY -m backend.listen.run_session serve out/ --port 8765
```

It prints the URL, the listener ids, and a reminder that the key is not served.
Leave it running. Ctrl-C stops it.

The routes are a fixed table — the interface, the session payload for one named
listener, one clip by opaque id, and the response endpoint. There is no static
root and no filename in any URL, so `/key.json` and `/clip/key.json` are 404 for
the same reason everything else is.

Listening on `127.0.0.1` means the listener has to be at this machine. To use
another machine on the same network, `--host 0.0.0.0` — and understand that
anyone on that network can then take the test as any listener, because there is
no authentication. Do not do it on a network you do not control.

## 5. Take the session

Send each listener to:

```
http://127.0.0.1:8765/?listener=alice
```

Or send them to `http://127.0.0.1:8765/` and let them pick from the list.

Before they start, say this out loud — none of it is on screen:

- **Headphones or monitors, not laptop speakers.** Set the volume once, on the
  first trial, and do not touch it again.
- **A and B are loudness-matched to 0.1 LU.** If one sounds louder, that is a
  real difference in the audio, not a level difference.
- **"original" is the unprocessed source**, for reference. It is not one of the
  two answers.
- **The two scales are different questions and are meant to disagree.** "Does it
  do what was asked?" can be 5 while "does it sound professional?" is 2.
- **`1`, `2`, `3` switch A / B / original; space plays and pauses.** Switching
  is gapless — the clips run on one clock and never restart, so switch often and
  switch fast. Comparing across a gap measures memory.
- **You cannot go back, and you cannot reload.** If the browser is refreshed the
  session is lost and has to be taken again from trial 1.
- The same request text appearing twice is expected. Answer it as you find it.

Every trial needs both sides rated on both scales plus a preference — six clicks
minimum — before "next trial" enables.

On the last trial the button says "finish and submit", and one file appears in
`out/responses/`. **Check that the listener saw "Done — thank you"** before they
leave: a failed submission currently leaves the page saying "submitting…"
forever, with the answers still in the tab and no way to retry.

## 6. Score it

```sh
$PY -m backend.listen.run_session score out/
```

You get, per condition and per scale: the mean, a 95% interval whose *n is the
number of people*, and the counts behind it. Then a preference tally, then one
consistency line per listener.

Read the consistency lines first.

```
alice: 0/6 reversed, 0/6 changed at all; mean |change| match 1.08, quality 0.92
bob:   1/6 reversed, 4/6 changed at all; mean |change| match 1.75, quality 1.33
carol: 6/6 reversed, 6/6 changed at all; mean |change| match 2.00, quality 2.17
dave:  4/6 reversed, 4/6 changed at all; mean |change| match 2.00, quality 2.00
```

`reversed` counts genuine contradictions — said A, then said B. `changed at all`
also counts moving to or from "no preference", which on a close pair is an
honest answer rather than a mistake. A listener reversing on half their repeats
is answering at chance; a listener reversing on nearly all of them has answered
the *opposite* way twice, which is its own kind of signal.

**`dave` above pressed A on all thirty trials without listening.** He shows up as
4/6 because the sides are re-randomised between the two occurrences of a repeat,
so pressing A reverses him half the time. That detection is probabilistic and
the report does not name it: with 3 repeats a side-presser looks *perfectly
consistent* about one time in eight, and with 6 repeats about one time in sixty.
Use 6 repeats, and if a listener's numbers look wrong, open their response file
and count how many times `"preference": "A"` appears. Thirty out of thirty is
not a rating.

Nothing in `score` excludes a listener. If someone fails the screen or fails
consistency, move their file out of `out/responses/` and run `score` again.

The Phase 0 gate — prefer-or-tie on at least 6 of 10 items — is **not** what
`score` prints. `score` gives one pooled prefer-or-tie rate over every trial and
every listener. The per-item breakdown the gate needs has to be counted from
`out/responses/*.json` against `out/key.json` by hand.

## 7. Correlate a proxy, if you have the data for it

```sh
$PY -m backend.listen.run_session correlate out/ metrics.json --scale quality
```

`metrics.json` maps each metric name to a value for every item, keyed
`item::condition` — exactly the keys the human scores use, and a metric missing
any of them is refused rather than correlated on the overlap.

```json
{"spectral_distance": {"guitar-di-03::audionova": 0.42,
                       "guitar-di-03::engineer":  0.61}}
```

Expect `NOT USABLE` on a Phase 0 panel, and do not read it as a bug. It means
what it says: fifteen ratings per point is the bar, three listeners give three,
and a rho blessed on that data is the proxy-metric capture the whole project is
built to avoid.

---

## Troubleshooting

| What you see | What it means |
|---|---|
| `UnmeasurableLoudness: ... measures -inf LUFS` | one of the three files for that comparison, or one screen segment, is silent. The message names the comparison and the files. |
| `... clips span 0.31 LU after matching` | the three files are not the same excerpt of the same performance, or one is nearly silent. Loudness matching cannot fix material it cannot measure. |
| `clip lengths differ by 12.4 ms` | the renders are not the same moment. Re-render at the same length. |
| `could not place 6 repeat(s) at least 3 trials apart` | too many repeats for too few comparisons. Add comparisons or cut repeats. |
| `<path>: expected a list of responses, got str` | that response file is not a submission. Move it out of `out/responses/` and re-score. |
| `FileNotFoundError: .../session.json` from `serve` | you pointed `serve` at a screening directory, or at a directory `build` never finished writing. |
| `response names trial '...' which is not in the key` | responses and key are from different builds. The seed alone does not identify a session; the manifest has to match too. |
| `alice answered trial ... twice` | two submissions from one listener, probably a retake. Keep one file, delete the other. |
| Page stuck on `submitting…` | the POST failed and the interface does not retry. The answers are still in that tab; do not close it. Restart the server and re-submit from the browser console with the payload in `state.answers`. |

## What must not happen

- The key must never be inside anything a listener can browse. The server holds
  it in memory; a static file server pointed at `out/` would hand it out.
- Never compare clips that were not built by the same `build` run. The loudness
  match is a property of a group, not of a file.
- Never edit `out/responses/*.json` to fix a rating. Re-run the trial or drop the
  listener; a hand-edited response is indistinguishable from a real one
  afterwards.
