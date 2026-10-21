# Recipe format

A recipe is the JSON description of a render. It is four things at once, and the
design follows from that:

- the planner's output contract
- the user's editable representation of what will happen
- the render cache key
- the reproducibility record

Anything not in the recipe is not reproducible, which is why the seed and the
internal sample rate are explicit fields rather than defaults chosen by the
renderer.

Reference implementation: `engine/include/anv/io/Recipe.h`.
Python builder: `backend/engine/render.py`.

## Shape

```json
{
  "engine_version": "0.1.0",
  "internal_sample_rate": 96000,
  "seed": 20260729,
  "output": { "sample_rate": 44100, "format": "pcm24" },

  "nodes": [
    { "id": "hp", "type": "parametric_eq", "bypass": false, "mix": 1.0,
      "params": { "bands": [
        { "filter_type": "highpass", "center_freq_hz": 80, "gain_db": 0, "q_factor": 0.707 },
        { "filter_type": "bell", "center_freq_hz": 320, "gain_db": -2.5, "q_factor": 1.1 }
      ] } },
    { "id": "comp", "type": "compressor",
      "params": { "threshold_dbfs": -18, "ratio_to_1": 3.0, "attack_ms": 12,
                  "release_ms": 140, "knee_db": 6, "makeup_gain_db": 3 } },
    { "id": "plate", "type": "plate_reverb", "mix": 0.18,
      "params": { "decay_s": 1.4, "pre_delay_ms": 18, "damping_hz": 6000 } }
  ],

  "master": [
    { "id": "limiter", "type": "true_peak_limiter",
      "params": { "ceiling_dbtp": -1.0, "lookahead_ms": 5, "release_ms": 80 } },
    { "id": "dither", "type": "dither",
      "params": { "bit_depth": 24, "amount_lsb": 2.0 } }
  ]
}
```

## Top-level fields

| Field | Meaning |
|---|---|
| `engine_version` | Recorded, not enforced. Renders are bit-exact within `(engine_version, arch)`. |
| `internal_sample_rate` | Default 96000. Range 8000…768000. |
| `seed` | Drives every stochastic element. Same seed, same render, always. |
| `output.sample_rate` | `0` means "keep the input's rate". |
| `output.format` | `pcm16` \| `pcm24` \| `pcm32` \| `float32` \| `float64` |
| `nodes` | Chain at the internal rate. |
| `master` | Chain **after** the output conversion. |

## Node fields

| Field | Default | Meaning |
|---|---|---|
| `type` | required | Closed set — see `engine.md`. Unknown types are a hard error. |
| `id` | `type_index` | Must be unique. This is what a refinement names. |
| `bypass` | `false` | Bypassed nodes still report latency and get compensating delay. |
| `mix` | `1.0` | 0 = fully dry, 1 = fully wet. Handled by the engine, phase-correct. |
| `params` | `{}` | Engineer units. One list-valued key at most. |

Stable `id`s are what make multi-turn refinement a patch rather than a
regeneration. "Add slightly more delay" has to name something, and regenerating
the whole graph would churn parameters the user had implicitly accepted — which
reads as a bug.

## Parameters

Everything is in **engineer units**: Hz, dB, ms, ratio, percent. Never normalised
0–1.

Booleans are accepted and arrive as 0/1, because every node reads its flags
through the numeric accessor. Keeping a separate boolean channel would mean each
node had to guess which one the recipe used.

A node may have **at most one** list-valued parameter (an EQ's `bands`). Two
would both land in the single ordered item list and the node reading it has no
way to tell which it received, so a second is an error rather than a silent
overwrite.

## What the engine rejects

Most of this is `validateRecipe`. The seed and the recipe's sample rates are
checked while the recipe is parsed, and the input file's rate and samples by
`anv_render` when it reads the input.

Everything here is a mistake that produces plausible-sounding but wrong output.
That is the only class worth a hard error — an obviously broken render needs no
validator.

| Rejected | Why |
|---|---|
| Unknown `type` | A planner emitting a node that does not exist yet must fail loudly, not ship a render missing the effect the user asked for that otherwise sounds fine. |
| Unknown `filter_type` | Used to fall back to a bell, silently. `high_shelf` instead of `highshelf` produced a plausible wrong answer. |
| Any unknown string enum, or one sent as a number | Every string parameter every node reads is listed in `knownStringParams()` in `Recipe.h`. Unlisted values used to render as the default: `"downward"` where `dynamic_eq` expected `"down"` made every planned dynamic EQ static. |
| A `dynamic_eq` band that is a notch, lowpass or highpass, or has a negative `range_db` | The node renders the first three as a bell and clamps the last to a static band. |
| A `seed` that is negative, fractional or past 2^53 | Converting it to an integer was undefined behaviour. |
| Duplicate `id` | A refinement could not name one node unambiguously. |
| `mix` outside 0…1 | |
| `true_peak_limiter` or `dither` in `nodes[]` | Only correct after the output conversion. |
| `dither` not last in `master[]` | Anything after it re-correlates the quantisation error. |
| More than one `dither` | |
| Any node after the limiter except `dither` | The ceiling the limiter established no longer holds. |
| `internal_sample_rate`, `output.sample_rate` or the input file's rate outside 8 kHz…768 kHz | An absurd rate in a WAV header was undefined behaviour in the resampler. |
| A float input sample that is NaN or infinite | The resampler spread one into a 239-sample dropout, written as silence. |
| An output format we cannot write | |

## Rendering

```sh
anv_render --in guitar.wav --out processed.wav --recipe chain.json
```

| Flag | Meaning |
|---|---|
| `--block <n>` | Render block size, default 4096. **Output is identical at any value** — this only affects memory use. |
| `--quiet` | Suppress the progress report on stderr. |

The report on stderr names the input, both conversions with their tap counts,
the output, and the tail length. The output is **longer** than the input whenever
the chain has a tail, which is intended — truncating a reverb at the last input
sample cuts the decay off mid-air — and is reported explicitly so a file that
changed length does not look like a bug.

## Building recipes from Python

```python
from backend.engine.render import Recipe, render_array

recipe = Recipe(seed=7, output_sample_rate=48000.0, output_format="float64")
recipe.add("parametric_eq", bands=[
    {"filter_type": "highshelf", "center_freq_hz": 6000.0,
     "gain_db": 4.0, "q_factor": 0.707},
])
recipe.add("compressor", threshold_dbfs=-24.0, ratio_to_1=3.0)
recipe.add_master("true_peak_limiter", ceiling_dbtp=-1.0)

processed = render_array(samples, 48000.0, recipe)
```

`render_array` trims to the input length, which is what a comparison needs — a
null test or a band-level delta needs two arrays describing the same moments.
Use `render` when you want the tail.

`output_format="float64"` is the analysis default: anything narrower puts
quantisation noise into a measurement of the engine's own behaviour and makes a
null test impossible.
