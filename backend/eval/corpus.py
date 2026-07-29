"""Reverse-engineering corpus: known chains applied to real audio.

THE ONLY SOURCE OF OBJECTIVE GROUND TRUTH WE WILL EVER HAVE FOR FREE. Every
other way of judging whether a chain was the right one needs a human listening,
which is slow, expensive, and unavailable at three in the morning. Here the
answer is known by construction: a recipe was applied, so recovering it is
either right or wrong, and the score is a number.

The plan is emphatic that this gets built BEFORE the planner. The research
corpus it draws on is a case study in optimising against an unvalidated proxy —
CLAP-guided EQ optimisation scored 0.55 where random parameters scored 0.53,
and the papers reporting it were measuring their own metric rather than the
audio. A harness built after the planner tends to become a harness that agrees
with the planner.

WHAT THIS CANNOT MEASURE, and it matters that the limit is written down:
recovering the chain is not the same as sounding good. A recipe that recovers
every parameter exactly is by definition perfect; a recipe that recovers none
of them may still sound better than the original to a paying engineer. So this
harness is a NECESSARY condition, not a sufficient one, and it never replaces
the listening panel — it makes the listening panel affordable by catching
everything that fails objectively first.

A KNOWN GAP, FOUND BY THIS HARNESS AND WORTH STATING PLAINLY: the measurement
report cannot see nonlinearity directly. Drive is detected only through its
side effects on dynamics — crest factor and peak-to-loudness ratio — because
there is no THD, harmonic-structure or clean-versus-driven measurement in the
report at all. For a product whose headline request is "make this sound like
that record's guitar tone", distortion CHARACTER is exactly what a planner most
needs to reason about, and it is currently invisible.

That is not a defect in this harness; it is the harness doing its job. The plan
already schedules a clean-versus-driven classifier in the Classify stage, and
this is the evidence for prioritising it.

EXCERPTS, NOT WHOLE FILES. Twenty-five seconds by default, matching the plan's
excerpt-first rendering. A four-minute stem takes seventeen seconds to measure
and the corpus needs hundreds of renders; the excerpt turns a two-hour sweep
into a five-minute one, and nothing in the analysis depends on duration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from backend.audio import wavio
from backend.engine.render import Recipe
from backend.measure import gating, loudness, stereo as stereo_measure


@dataclass
class SourceContext:
    """What a perturbation needs to know about the material it will act on.

    THRESHOLDS MUST BE RELATIVE TO THE SOURCE. A compressor at a fixed
    -28 dBFS means something completely different on a bass at -11.5 LUFS than
    on an acoustic guitar at -22.6: the first is crushed, the second is barely
    touched, and a sweep of the RATIO across both is then not a sweep of one
    variable. Measured that way, compression's effect on crest factor came out
    undetectable on real stems while passing on synthetic material where every
    source had the same level.

    This is also how a planner will emit thresholds — from the measured
    percentiles of the actual audio — so a corpus built any other way would be
    testing a chain nobody will ever generate.
    """

    channels: int
    sample_rate: float
    integrated_lufs: float
    correlation: float

    @classmethod
    def measure(cls, samples: np.ndarray, sample_rate: float) -> "SourceContext":
        samples = np.atleast_2d(samples)
        lufs = loudness.integrated_lufs(samples, sample_rate)
        return cls(
            channels=int(samples.shape[0]),
            sample_rate=sample_rate,
            integrated_lufs=float(lufs) if np.isfinite(lufs) else -30.0,
            correlation=stereo_measure.broadband_correlation(samples),
        )


@dataclass
class Perturbation:
    """One node, one parameter, and the range a planner might plausibly pick.

    ``build`` returns the nodes to append. It takes the source context as well
    as the value because several of these are only meaningful relative to the
    material — see SourceContext.
    """

    name: str
    node_type: str
    parameter: str
    unit: str
    # At least three, always. A correlation computed over two points is +1 or
    # -1 by construction and carries no evidence at all.
    values: tuple[float, ...]
    build: Callable[[float, SourceContext], list[dict[str, Any]]]
    # Which report fields this SHOULD move, as the hypothesis the sensitivity
    # analysis tests. Writing it down in advance is the difference between
    # measuring detectability and going looking for whatever moved.
    expected_fields: tuple[str, ...] = ()
    master: bool = False
    # Whether this perturbation can act on this source AT ALL.
    #
    # NOT APPLICABLE IS NOT THE SAME AS NOT DETECTED, and conflating them makes
    # the harness lie. Stereo width applied to a mono or dual-mono stem changes
    # nothing — correctly, by design, since there is no side signal — and every
    # measurement delta came back as exactly 0.00 across all four stems. Scored
    # as a detection failure that reads as a hole in the measurement layer;
    # scored as inapplicable it reads as what it is, a gap in the CORPUS.
    applies_to: Callable[[SourceContext], bool] = lambda ctx: True


def _eq(filter_type: str, freq: float, gain: float, q: float = 0.707) -> list[dict[str, Any]]:
    return [{
        "type": "parametric_eq",
        "params": {"bands": [{
            "filter_type": filter_type, "center_freq_hz": freq,
            "gain_db": gain, "q_factor": q,
        }]},
    }]


# The catalogue. Ranges are what an engineer would actually reach for, not what
# the parameter accepts — a sweep over the full legal range would spend most of
# its time measuring settings nobody would ever choose.
PERTURBATIONS: tuple[Perturbation, ...] = (
    Perturbation(
        name="gain", node_type="gain", parameter="gain_db", unit="dB",
        values=(-6.0, -3.0, 3.0, 6.0),
        build=lambda v, ctx: [{"type": "gain", "params": {"gain_db": v}}],
        expected_fields=("loudness.integrated_lufs", "loudness.true_peak_dbtp"),
    ),
    Perturbation(
        name="high_shelf", node_type="parametric_eq", parameter="gain_db", unit="dB",
        values=(-8.0, -4.0, 4.0, 8.0),
        build=lambda v, ctx: _eq("highshelf", 6000.0, v),
        expected_fields=("spectrum.hf_rolloff_db_per_octave",
                         "spectrum.tilt_db_per_octave_re_pink"),
    ),
    Perturbation(
        name="low_shelf", node_type="parametric_eq", parameter="gain_db", unit="dB",
        values=(-8.0, -4.0, 4.0, 8.0),
        build=lambda v, ctx: _eq("lowshelf", 150.0, v),
        expected_fields=("spectrum.tilt_db_per_octave_re_pink",),
    ),
    Perturbation(
        name="mid_bell", node_type="parametric_eq", parameter="gain_db", unit="dB",
        values=(-9.0, -5.0, 5.0, 9.0),
        build=lambda v, ctx: _eq("bell", 1000.0, v, q=1.4),
        expected_fields=("spectrum.tilt_residual_db",),
    ),
    Perturbation(
        name="highpass", node_type="parametric_eq", parameter="center_freq_hz", unit="Hz",
        values=(60.0, 120.0, 200.0, 300.0),
        build=lambda v, ctx: _eq("highpass", v, 0.0),
        expected_fields=("spectrum.tilt_db_per_octave_re_pink",),
    ),
    Perturbation(
        name="compression", node_type="compressor", parameter="ratio_to_1", unit=":1",
        values=(2.0, 4.0, 8.0),
        # Threshold 6 dB below the source's own integrated loudness, so the
        # ratio sweep means the same thing on every stem. A fixed threshold
        # made this a sweep of two variables at once.
        build=lambda v, ctx: [{
            "type": "compressor",
            "params": {"threshold_dbfs": ctx.integrated_lufs - 6.0,
                       "ratio_to_1": v, "knee_db": 6.0,
                       "attack_ms": 10.0, "release_ms": 120.0},
        }],
        # NOT crest factor, and this was a prediction failure worth keeping.
        # It passes on synthetic steady noise and fails on all four real stems,
        # which is the signature of a measurement that only works when the
        # material is uniform.
        #
        # Crest factor is peak minus RMS, and peak is a SINGLE SAMPLE. A
        # compressor with a 10 ms attack deliberately passes the initial
        # transient, so on percussive material the peak survives while the RMS
        # falls and crest RISES; on sustained material it falls. The sign
        # depends on the source, so averaged across sources the correlation
        # goes to nothing. That is knowable in advance and should have been
        # predicted.
        #
        # The envelope modulation spectrum has no such problem. Compression
        # reduces slow, large level excursions more than fast ones, so its
        # energy shifts upward in modulation frequency monotonically with
        # ratio, whatever the material — r = 0.94 across all four stems. The
        # plan already called the modulation spectrum "the direct input to
        # attack/release estimation"; this is the evidence for it.
        expected_fields=("dynamics.envelope_modulation.centroid_hz",),
    ),
    Perturbation(
        name="drive", node_type="drive", parameter="drive_db", unit="dB",
        values=(6.0, 12.0, 18.0),
        build=lambda v, ctx: [{"type": "drive", "params": {"drive_db": v,
                                                          "output_trim_db": -v * 0.5}}],
        # NOT the HF rolloff, and the reason is a compliment to the engine.
        # Predicting that saturation would brighten the top end looked obvious
        # and is wrong: measured across 0 to 24 dB of drive, the rolloff moves
        # from -0.59 to -0.43 dB/octave, while every third-octave band rises
        # near-uniformly. The added harmonics are spread proportionally rather
        # than piling up at the top, which is exactly what ADAA plus
        # oversampling is for. A drive stage that visibly reshaped the top
        # octave would be one that was aliasing.
        #
        # What drive DOES move is dynamics: it flattens peaks and lifts the
        # quiet end. See the module note on the gap this leaves.
        expected_fields=("dynamics.crest_factor_db",
                         "dynamics.peak_to_loudness_ratio_db"),
    ),
    Perturbation(
        name="plate_reverb", node_type="plate_reverb", parameter="mix", unit="fraction",
        values=(0.12, 0.25, 0.4),
        build=lambda v, ctx: [{
            "type": "plate_reverb", "mix": v,
            "params": {"decay_s": 1.8, "pre_delay_ms": 20.0, "damping_hz": 6000.0},
        }],
        expected_fields=("dynamics.envelope_modulation.centroid_hz",
                         "dynamics.crest_factor_db"),
    ),
    Perturbation(
        name="delay", node_type="delay", parameter="mix", unit="fraction",
        values=(0.1, 0.2, 0.35),
        build=lambda v, ctx: [{
            "type": "delay", "mix": v,
            "params": {"time_ms": 320.0, "feedback_percent": 35.0},
        }],
        expected_fields=("dynamics.envelope_modulation.centroid_hz",),
    ),
    Perturbation(
        name="stereo_width", node_type="stereo_width", parameter="width_percent",
        unit="%", values=(0.0, 50.0, 160.0, 220.0),
        build=lambda v, ctx: [{"type": "stereo_width", "params": {"width_percent": v}}],
        expected_fields=("stereo.correlation",),
        # Needs a side signal to act on. All four reference stems are mono or
        # dual-mono, so without this the sweep reports a measurement failure
        # for a node that is behaving exactly as designed.
        applies_to=lambda ctx: ctx.channels >= 2 and ctx.correlation < 0.99,
    ),
)

PERTURBATIONS_BY_NAME = {p.name: p for p in PERTURBATIONS}


@dataclass
class Case:
    """One ground-truth example: a source excerpt and the chain applied to it."""

    stem: str
    perturbation: str
    value: float
    recipe: Recipe
    source: np.ndarray
    sample_rate: float
    truth: dict[str, Any] = field(default_factory=dict)


def load_excerpt(
    path: str, seconds: float = 25.0, sample_rate_hint: float | None = None
) -> tuple[np.ndarray, float]:
    """The most continuously-active window of a file.

    CHOSEN BY GATED COVERAGE, not by peak level or by taking the middle. The
    middle of a stem is frequently a rest, and the loudest window is often a
    single hit surrounded by silence. What every measurement here wants is the
    stretch with the most programme material in it, which is exactly what the
    loudness gate already computes.
    """
    audio = wavio.read_wav(path)
    rate = audio.sample_rate if sample_rate_hint is None else sample_rate_hint
    window = int(round(seconds * rate))
    if audio.frames <= window:
        return audio.samples, rate

    mask, _ = gating.gated_frame_mask(audio.samples, rate)
    covered = np.concatenate([[0], np.cumsum(mask.astype(np.int64))])

    # Hop by a second: finer would be slower for no benefit, since the answer
    # only needs to be a good window rather than the single best one.
    hop = int(round(rate))
    starts = np.arange(0, audio.frames - window + 1, hop)
    scores = covered[starts + window] - covered[starts]
    best = int(starts[int(np.argmax(scores))])
    return np.ascontiguousarray(audio.samples[:, best : best + window]), rate


def build_case(
    stem_path: str, perturbation: Perturbation, value: float,
    *, seconds: float = 25.0, seed: int = 0,
) -> Case:
    source, rate = load_excerpt(stem_path, seconds)
    context = SourceContext.measure(source, rate)

    recipe = Recipe(seed=seed, output_sample_rate=rate, output_format="float64")
    nodes = perturbation.build(value, context)
    if perturbation.master:
        recipe.master.extend(nodes)
    else:
        recipe.nodes.extend(nodes)

    return Case(
        stem=os.path.splitext(os.path.basename(stem_path))[0],
        perturbation=perturbation.name,
        value=value,
        recipe=recipe,
        source=source,
        sample_rate=rate,
        truth={
            "node_types": [n["type"] for n in nodes],
            "parameter": perturbation.parameter,
            "value": value,
            "unit": perturbation.unit,
        },
    )


def enumerate_cases(
    stem_paths: list[str], *, seconds: float = 25.0,
    only: tuple[str, ...] | None = None,
) -> list[Case]:
    """Every (stem, perturbation, value) combination.

    Exhaustive rather than sampled. The space is small enough to cover
    completely, and a random sample would make the results depend on a seed
    that has nothing to do with the audio.
    """
    cases: list[Case] = []
    for path in stem_paths:
        for perturbation in PERTURBATIONS:
            if only and perturbation.name not in only:
                continue
            for value in perturbation.values:
                cases.append(build_case(path, perturbation, value, seconds=seconds))
    return cases
