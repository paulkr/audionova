"""The listener screen: a 2 dB Q=1 bell at 3 kHz, identified at >= 80%.

WHY SCREEN AT ALL. A panel result is an average over listeners, and a listener
who cannot hear the difference contributes noise centred on chance. Three
listeners is already a thin sample; one of them answering at random doubles the
width of every interval the panel produces. The plan's screen is the cheapest
filter that separates "did not prefer it" from "could not hear it".

THE STIMULUS IS RENDERED BY THE REAL ENGINE. A screen built from a scipy
biquad would be testing whether the listener can hear a scipy biquad. What
matters is whether they can hear a difference of the size and shape the product
actually produces, so the bell goes through `anv_render` with the same
`parametric_eq` node a planner would emit.

BOTH SIDES ARE RENDERED, AND THE UNPROCESSED SIDE IS A NULL RENDER. Comparing
a rendered clip against the raw source would put the engine's own sample-rate
conversion into the difference — `findings.md` records a -0.5 dB gain reading
-0.62 dB for exactly this reason. Against a null render the only difference is
the bell.

AND THE PAIR IS LOUDNESS-MATCHED, which is not a detail. A +2 dB Q=1 bell at
3 kHz sits near the peak of the K-weighting curve and moves integrated loudness
by a few tenths of a LU. Unmatched, a listener could pass this screen with no
frequency discrimination at all by picking the louder clip every time — the
screen would then select for exactly the bias the rest of the harness exists to
remove.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.engine.render import Recipe, render_array
from backend.listen import match
from backend.listen.session import derive_rng, opaque_id
from backend.measure import spectrum

# The plan's screen, verbatim.
FREQ_HZ = 3000.0
GAIN_DB = 2.0
Q_FACTOR = 1.0
PASS_RATE = 0.80

DEFAULT_TRIALS = 10
DEFAULT_SEGMENT_S = 8.0

# 2AFC: two clips, one has the boost. Chance is one half, so 8 of 10 is a
# 5.5% result under the null and 7 of 10 is 17%.
CHANCE_RATE = 0.5

QUESTION = ("Which clip has more energy around 3 kHz? "
            "(one of the two has a 2 dB boost there)")


class ScreeningError(ValueError):
    """The screen as described cannot be built or scored."""


def bell_recipe(freq_hz: float = FREQ_HZ, gain_db: float = GAIN_DB,
                q_factor: float = Q_FACTOR) -> Recipe:
    recipe = Recipe()
    recipe.add("parametric_eq", bands=[{
        "filter_type": "bell",
        "center_freq_hz": freq_hz,
        "gain_db": gain_db,
        "q_factor": q_factor,
    }])
    return recipe


def null_recipe() -> Recipe:
    """An empty recipe: the same conversion, none of the processing."""
    return Recipe()


@dataclass(frozen=True)
class ScreeningTrial:
    trial_id: str
    index: int
    clip_a: str
    clip_b: str
    bell_side: str          # "A" or "B" — KEY SIDE ONLY


@dataclass(frozen=True)
class BuiltScreening:
    directory: str
    trials: tuple[ScreeningTrial, ...]
    clip_files: dict[str, str]
    screening_json: str
    key_json: str
    groups: tuple[match.MatchedGroup, ...]

    def public_payload(self) -> dict[str, Any]:
        return {
            "question": QUESTION,
            "pass_rate": PASS_RATE,
            "trials": [
                {"trial_id": trial.trial_id,
                 "clip_a": trial.clip_a,
                 "clip_b": trial.clip_b}
                for trial in self.trials
            ],
        }

    def key_payload(self) -> dict[str, Any]:
        return {
            "stimulus": {"freq_hz": FREQ_HZ, "gain_db": GAIN_DB,
                         "q_factor": Q_FACTOR, "shape": "bell"},
            "pass_rate": PASS_RATE,
            "chance_rate": CHANCE_RATE,
            "answers": {trial.trial_id: trial.bell_side for trial in self.trials},
            "clips": {clip_id: os.path.relpath(path, self.directory)
                      for clip_id, path in self.clip_files.items()},
        }


def build_screening(
    source: np.ndarray,
    sample_rate: float,
    directory: str,
    *,
    seed: int,
    trials: int = DEFAULT_TRIALS,
    segment_s: float = DEFAULT_SEGMENT_S,
    freq_hz: float = FREQ_HZ,
    gain_db: float = GAIN_DB,
    q_factor: float = Q_FACTOR,
    gain_solver: match.GainSolver = match.match_to_quietest,
    tolerance_lu: float = match.TOLERANCE_LU,
    ceiling_dbtp: float = match.CEILING_DBTP,
    bit_depth: int = match.BIT_DEPTH,
    binary: str | None = None,
) -> BuiltScreening:
    """Render `trials` A/B pairs from consecutive segments of `source`.

    Consecutive rather than repeated: a screen built from one segment measures
    whether the listener learned one clip.
    """
    if trials < 1:
        raise ScreeningError(f"a screen needs at least one trial, got {trials}")

    samples = np.atleast_2d(np.asarray(source, dtype=np.float64))
    segment_frames = int(round(segment_s * sample_rate))
    if segment_frames < 1:
        raise ScreeningError(f"segment_s {segment_s} is shorter than a frame")
    needed = segment_frames * trials
    if samples.shape[1] < needed:
        raise ScreeningError(
            f"{trials} trials of {segment_s} s need {needed} frames and the "
            f"source has {samples.shape[1]}. A screen that reused a segment "
            "would measure recall of one clip rather than discrimination.")

    clips_dir = os.path.join(directory, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    render_kwargs: dict[str, Any] = {}
    if binary is not None:
        render_kwargs["binary"] = binary

    rng = derive_rng(seed, "screening")
    store_rng = derive_rng(seed, "screening-store")

    built: list[ScreeningTrial] = []
    clip_files: dict[str, str] = {}
    groups: list[match.MatchedGroup] = []
    work = os.path.join(directory, "_unmatched")
    os.makedirs(work, exist_ok=True)

    for index in range(trials):
        start = index * segment_frames
        segment = samples[:, start:start + segment_frames]

        plain = render_array(segment, sample_rate, null_recipe(), **render_kwargs)
        boosted = render_array(segment, sample_rate,
                               bell_recipe(freq_hz, gain_db, q_factor),
                               **render_kwargs)

        # Float intermediates: a +2 dB boost on material already near full
        # scale goes over it, and a PCM clamp here would put the clipping
        # inside the stimulus rather than into the ceiling offset.
        raw_plain = os.path.join(work, f"{index:03d}-plain.wav")
        raw_bell = os.path.join(work, f"{index:03d}-bell.wav")
        match.write_clip(raw_plain, plain, sample_rate, 64, is_float=True)
        match.write_clip(raw_bell, boosted, sample_rate, 64, is_float=True)

        # The side bit is drawn before anything uses it, so the identifiers
        # below do not depend on which side got the bell.
        bell_on_b = bool(rng.getrandbits(1))
        trial_id = opaque_id(rng)
        clip_a = opaque_id(rng)
        clip_b = opaque_id(rng)

        out_plain = os.path.join(clips_dir, f"{opaque_id(store_rng)}.wav")
        out_bell = os.path.join(clips_dir, f"{opaque_id(store_rng)}.wav")
        group = match.match_group(
            [raw_plain, raw_bell], [out_plain, out_bell],
            gain_solver=gain_solver, tolerance_lu=tolerance_lu,
            ceiling_dbtp=ceiling_dbtp, bit_depth=bit_depth,
            what=f"screening trial {index}",
        )
        groups.append(group)

        if bell_on_b:
            clip_files[clip_a] = out_plain
            clip_files[clip_b] = out_bell
        else:
            clip_files[clip_a] = out_bell
            clip_files[clip_b] = out_plain

        built.append(ScreeningTrial(
            trial_id=trial_id, index=index, clip_a=clip_a, clip_b=clip_b,
            bell_side="B" if bell_on_b else "A",
        ))

    shutil.rmtree(work, ignore_errors=True)

    screening = BuiltScreening(
        directory=directory,
        trials=tuple(built),
        clip_files=clip_files,
        screening_json=os.path.join(directory, "screening.json"),
        key_json=os.path.join(directory, "screening_key.json"),
        groups=tuple(groups),
    )
    with open(screening.screening_json, "w") as handle:
        json.dump(screening.public_payload(), handle, indent=2, sort_keys=True)
    with open(screening.key_json, "w") as handle:
        json.dump(screening.key_payload(), handle, indent=2, sort_keys=True)
    return screening


# --- scoring ------------------------------------------------------------------


@dataclass(frozen=True)
class ScreeningResult:
    listener: str
    correct: int
    trials: int
    score: float
    passed: bool
    wrong: tuple[str, ...]

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (f"{self.listener}: {self.correct}/{self.trials} = "
                f"{self.score:.0%} — {verdict} (bar {PASS_RATE:.0%}, "
                f"chance {CHANCE_RATE:.0%})")


def score_screening(answers: dict[str, str], key: dict[str, Any],
                    listener: str = "listener") -> ScreeningResult:
    """Score one listener against the screening key.

    UNANSWERED IS AN ERROR, NOT A WRONG ANSWER. Scoring a missing trial as
    wrong silently converts a broken interface into a failed listener, and the
    two want completely different responses.
    """
    truth = key.get("answers")
    if not isinstance(truth, dict) or not truth:
        raise ScreeningError("the screening key has no answers in it")

    bar = float(key.get("pass_rate", PASS_RATE))

    unknown = sorted(set(answers) - set(truth))
    if unknown:
        raise ScreeningError(
            f"{listener}: answers name trials that are not in the key: {unknown}")
    missing = sorted(set(truth) - set(answers))
    if missing:
        raise ScreeningError(
            f"{listener}: {len(missing)} screening trial(s) unanswered "
            f"({missing[:3]}...). An unanswered trial is a broken session, not "
            "a wrong answer, and scoring it as wrong would hide that.")

    correct = 0
    wrong: list[str] = []
    for trial_id, expected in truth.items():
        given = answers[trial_id]
        if given not in ("A", "B"):
            raise ScreeningError(
                f"{listener}: trial {trial_id} answered {given!r}; a 2AFC "
                "screen takes 'A' or 'B' and nothing else")
        if given == expected:
            correct += 1
        else:
            wrong.append(trial_id)

    total = len(truth)
    score = correct / total
    return ScreeningResult(
        listener=listener,
        correct=correct,
        trials=total,
        score=score,
        # >=, not >: the plan's bar is "at least 80%", so 8 of 10 passes.
        passed=score >= bar - 1e-12,
        wrong=tuple(sorted(wrong)),
    )


def screen_panel(answers_by_listener: dict[str, dict[str, str]],
                 key: dict[str, Any]) -> dict[str, ScreeningResult]:
    return {listener: score_screening(answers, key, listener)
            for listener, answers in answers_by_listener.items()}


def binomial_tail(correct: int, trials: int, p: float = CHANCE_RATE) -> float:
    """P(at least `correct` right | guessing). Reported, never used as a gate.

    The bar is the plan's 80%. This says how surprising the result would be by
    chance, which is a different question and a useful one when someone lands
    on the bar with a short screen.
    """
    if trials < 1 or correct < 0 or correct > trials:
        raise ScreeningError(f"{correct} of {trials} is not a result")
    total = 0.0
    for k in range(correct, trials + 1):
        total += math.comb(trials, k) * (p ** k) * ((1.0 - p) ** (trials - k))
    return total


def bell_delta_db(plain: np.ndarray, boosted: np.ndarray,
                  sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Third-octave difference between the two sides, broadband tilt removed.

    The pair is loudness-matched, so a common offset sits on every band; the
    median across bands is subtracted for the same reason the chain recogniser
    subtracts it, and what is left is the shape of the filter.
    """
    freqs_a, power_a = spectrum.gated_ltas(np.atleast_2d(plain), sample_rate)
    freqs_b, power_b = spectrum.gated_ltas(np.atleast_2d(boosted), sample_rate)
    if freqs_a.size == 0 or freqs_b.size == 0:
        raise ScreeningError("clips too short to measure a spectrum")

    centres, levels_a = spectrum.third_octave_levels_db(freqs_a, power_a, sample_rate)
    _, levels_b = spectrum.third_octave_levels_db(freqs_b, power_b, sample_rate)
    delta = levels_b - levels_a
    usable = np.isfinite(delta)
    if not np.any(usable):
        raise ScreeningError("no band carries energy in both clips")
    delta = delta - float(np.median(delta[usable]))
    return centres, delta
