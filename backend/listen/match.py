"""Loudness matching for listening tests, and the verification that it worked.

LOUDER RELIABLY WINS A BLIND COMPARISON WHETHER OR NOT IT IS BETTER. That is
the single best-established artifact in audio listening tests, it is worth
somewhere around a decibel of apparent quality per decibel of level, and it
means an unmatched session produces data that looks like preference and is
loudness. Every quality claim built on such a session is a claim about gain.

So matching is not optional here, and — the part that matters — it is
VERIFIED RATHER THAN ASSUMED. Applying `target - measured` dB of gain is
arithmetically exact against K-weighted loudness, so it is tempting to write
the gain and move on. Three things can still make the written file miss:

  * BS.1770's absolute gate at -70 LUFS is not relative to the programme.
    Attenuating a whole group can push quiet blocks below it, which changes
    WHICH blocks are integrated, which moves the answer by more than the gain.
  * Quantising to the delivery word length adds noise and clamps peaks.
  * The gain solver is a parameter. A caller can pass one that does nothing,
    and the point of a verification is that it does not trust its own inputs.

So every clip is written, READ BACK FROM DISK, re-measured, and the group is
refused if the spread exceeds the tolerance. `findings.md` has a whole section
on measurements that were never checked against the thing they claimed; this
one is checked against the bytes a listener will actually hear.

THE CEILING IS A COMMON OFFSET, NOT A PER-CLIP ONE. A source stem can be over
full scale before anything touches it — `acoustic-guitar-track` in the
reference corpus measures +0.52 dBTP — and writing that to PCM clips it. The
fix has to move every clip in the group by the SAME amount, because anything
per-clip is a loudness difference reintroduced under another name.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from backend.audio import wavio
from backend.measure import loudness

# The bar. 0.1 LU is comfortably below the ~0.5-1 LU level difference that
# starts to bias a preference judgement, and comfortably above what quantising
# to 24 bits can move.
TOLERANCE_LU = 0.1

# The delivery word length for test clips. 24-bit PCM rather than float,
# because every browser decodes it and because the quantisation is 20 dB below
# anything the measurement layer resolves.
BIT_DEPTH = 24

# Written clips must sit at or under this. -1.0 dBTP is the same ceiling the
# validator inserts into every graph that lacks one.
CEILING_DBTP = -1.0

# Clips compared against each other must describe the same moment. Rendering
# can legitimately cost or gain a sample at a rate conversion; anything past
# a millisecond is a different excerpt and is refused rather than trimmed.
MAX_LENGTH_DRIFT_S = 0.001


class LoudnessMismatch(RuntimeError):
    """A group of clips that would be compared is not loudness-matched."""


class UnmeasurableLoudness(RuntimeError):
    """A clip has no integrated loudness, so it cannot be matched to anything.

    Raised rather than substituting a plausible number. A fabricated loudness
    is indistinguishable from a measured one once a gain has been derived from
    it, and the resulting session looks matched and is not.
    """


class ClipGroupError(ValueError):
    """The clips handed in cannot form an A/B comparison at all."""


GainSolver = Callable[[Sequence[float]], list[float]]


def match_to_quietest(lufs: Sequence[float]) -> list[float]:
    """The default solver: bring everything down to the quietest clip.

    DOWN, never up. Matching to the mean or to a fixed target boosts at least
    one clip, and a boost is where clipping and limiter re-engagement come
    from. Attenuation cannot introduce either.
    """
    values = list(lufs)
    if not values:
        raise ClipGroupError("no clips to match")
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise UnmeasurableLoudness(
                f"clip {index} measures {value} LUFS — it has no programme "
                "material above the absolute gate, so there is no loudness to "
                "match it by")
    target = min(values)
    return [target - value for value in values]


def match_to_target(target_lufs: float) -> GainSolver:
    """A solver that pins the group to a fixed loudness.

    Present because the sensible alternative to the default is a fixed target,
    and because it makes the solver seam an honest parameter rather than a
    test hook.
    """
    def solver(lufs: Sequence[float]) -> list[float]:
        values = list(lufs)
        for index, value in enumerate(values):
            if not math.isfinite(value):
                raise UnmeasurableLoudness(
                    f"clip {index} measures {value} LUFS and cannot be matched")
        return [target_lufs - value for value in values]
    return solver


@dataclass(frozen=True)
class MatchedClip:
    """One written clip, described by what was measured off the written file."""

    path: str
    lufs: float
    true_peak_dbtp: float
    gain_db: float          # solver gain plus the group's ceiling offset
    frames: int
    sample_rate: float
    channels: int


@dataclass(frozen=True)
class MatchedGroup:
    clips: tuple[MatchedClip, ...]
    offset_db: float        # common attenuation applied to keep the ceiling
    spread_lu: float        # measured, after writing

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(clip.path for clip in self.clips)


def verify_matched(measured_lufs: Sequence[float],
                   tolerance_lu: float = TOLERANCE_LU,
                   what: str = "clip group") -> float:
    """Refuse a group whose measured loudnesses are further apart than the bar.

    Returns the spread so a caller can record it. Raises rather than warning:
    a warning in a session build is a session that ships.
    """
    values = list(measured_lufs)
    if len(values) < 2:
        raise ClipGroupError(
            f"{what}: a match needs at least two clips, got {len(values)}")
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise UnmeasurableLoudness(
                f"{what}: clip {index} measured {value} LUFS after writing")

    spread = max(values) - min(values)
    if spread > tolerance_lu:
        raise LoudnessMismatch(
            f"{what}: clips span {spread:.4f} LU after matching, which is "
            f"over the {tolerance_lu} LU bar. Louder wins a blind comparison, "
            f"so this session would measure level rather than preference. "
            f"Measured: {[round(v, 4) for v in values]}")
    return spread


def match_group(
    inputs: Sequence[str],
    outputs: Sequence[str],
    *,
    gain_solver: GainSolver = match_to_quietest,
    tolerance_lu: float = TOLERANCE_LU,
    ceiling_dbtp: float = CEILING_DBTP,
    bit_depth: int = BIT_DEPTH,
    what: str = "clip group",
) -> MatchedGroup:
    """Load, match, write, read back, re-measure, and refuse a mismatch.

    The read-back is the whole point. Everything before it is a prediction.
    """
    if len(inputs) != len(outputs):
        raise ClipGroupError(
            f"{what}: {len(inputs)} inputs and {len(outputs)} outputs")
    if len(inputs) < 2:
        raise ClipGroupError(f"{what}: nothing to compare, {len(inputs)} clip(s)")

    audio = [wavio.read_wav(path) for path in inputs]

    rates = {clip.sample_rate for clip in audio}
    if len(rates) != 1:
        raise ClipGroupError(
            f"{what}: clips compared against each other must share a sample "
            f"rate, got {sorted(rates)}")
    sample_rate = float(next(iter(rates)))

    frames = [clip.frames for clip in audio]
    drift_s = (max(frames) - min(frames)) / sample_rate
    if drift_s > MAX_LENGTH_DRIFT_S:
        raise ClipGroupError(
            f"{what}: clip lengths differ by {drift_s * 1000:.1f} ms, so they "
            "are not the same excerpt. A/B switching only means anything when "
            "both sides are the same moment of the same performance.")
    keep = min(frames)
    trimmed = [clip.samples[:, :keep] for clip in audio]

    channels = {clip.shape[0] for clip in trimmed}
    if len(channels) != 1:
        raise ClipGroupError(
            f"{what}: clips have different channel counts {sorted(channels)}; "
            "a mono/stereo A/B is a width comparison, not the one asked for")

    before = [float(loudness.integrated_lufs(clip, sample_rate)) for clip in trimmed]
    gains = list(gain_solver(before))
    if len(gains) != len(trimmed):
        raise ClipGroupError(
            f"{what}: gain solver returned {len(gains)} gains for "
            f"{len(trimmed)} clips")

    scaled = [clip * (10.0 ** (gain / 20.0)) for clip, gain in zip(trimmed, gains)]

    # One common offset for the whole group. Per-clip headroom would be a
    # loudness difference wearing a safety mechanism's clothes.
    peaks = [float(loudness.true_peak_dbtp(clip, sample_rate)) for clip in scaled]
    finite_peaks = [peak for peak in peaks if math.isfinite(peak)]
    highest = max(finite_peaks) if finite_peaks else -math.inf
    offset_db = min(0.0, ceiling_dbtp - highest) if math.isfinite(highest) else 0.0
    if offset_db < 0.0:
        factor = 10.0 ** (offset_db / 20.0)
        scaled = [clip * factor for clip in scaled]

    for clip, path in zip(scaled, outputs):
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        wavio.write_wav(path, clip, sample_rate, bit_depth)

    written: list[MatchedClip] = []
    for path, gain in zip(outputs, gains):
        back = wavio.read_wav(path)
        written.append(MatchedClip(
            path=path,
            lufs=float(loudness.integrated_lufs(back.samples, back.sample_rate)),
            true_peak_dbtp=float(loudness.true_peak_dbtp(back.samples,
                                                         back.sample_rate)),
            gain_db=gain + offset_db,
            frames=back.frames,
            sample_rate=float(back.sample_rate),
            channels=back.channels,
        ))

    spread = verify_matched([clip.lufs for clip in written], tolerance_lu, what)

    # NO POST-WRITE CEILING CHECK, and its absence is deliberate rather than an
    # omission. `offset_db` is `min(0, ceiling - highest)` over the true peaks
    # of the scaled arrays, so every clip sits at or under the ceiling BEFORE
    # it is written, and quantising to PCM moves a true peak by well under a
    # thousandth of a decibel. A guard here could not fire, and a guard that
    # cannot fire tells the next reader the quantity can go over when it
    # cannot — the same reasoning that deleted `TransientDesigner`'s sustain
    # rectification. The invariant is asserted from outside instead, in
    # `test_listen.py`, at a non-default ceiling as well as the default one.
    return MatchedGroup(clips=tuple(written), offset_db=offset_db, spread_lu=spread)


def write_clip(path: str, samples: np.ndarray, sample_rate: float,
               bit_depth: int = BIT_DEPTH, is_float: bool = False) -> None:
    """Write a clip, for fixtures and for the matcher's own intermediates.

    INTERMEDIATES ARE WRITTEN AS FLOAT, and that is not tidiness. A render
    carrying a boost can legitimately sit over full scale before the ceiling
    offset is applied; PCM would clamp it there and the clamp would be inside
    the stimulus the listener judges. The matcher's own OUTPUT is PCM, after
    the offset has brought it under the ceiling.
    """
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    wavio.write_wav(path, np.atleast_2d(samples), sample_rate, bit_depth, is_float)
