"""Loudness gating, shared by every measurement that averages over time.

THE MOST COMMON WAY A MEASUREMENT REPORT LIES is by averaging over silence.
A vocal stem is typically 40-60% silence, and that silence is not empty — it is
room tone, headphone bleed, preamp hiss, and the tail of the previous phrase.
An ungated long-term average spectrum of such a stem describes the ROOM, not
the voice, and it does so confidently: the numbers look plausible, the shape
looks like a spectrum, and every conclusion drawn from it is about the wrong
signal.

The specific failure is worse than a wash. Room tone is low-frequency-heavy and
broadband-quiet, so an ungated LTAS reads as "muddy with a rolled-off top" for
material that is neither. A planner given that spectrum will prescribe a
low-cut and a presence boost for a track that needs neither, and the result
will sound thin and harsh.

So the same BS.1770 gate that defines integrated loudness defines which parts
of the signal every other measurement is allowed to look at. One definition of
"where the programme is", used everywhere, rather than each measurement
inventing its own threshold.
"""

from __future__ import annotations

import numpy as np

from backend.measure import loudness


def gated_frame_mask(
    samples: np.ndarray, sample_rate: float
) -> tuple[np.ndarray, float]:
    """Per-frame boolean mask of the programme material.

    Returns ``(mask, coverage)`` where coverage is the fraction of the file
    that survived. Coverage is returned rather than hidden because it is
    diagnostic in its own right: a stem at 15% coverage is mostly silence, and
    every averaged measurement over it rests on much less audio than its
    duration suggests.

    A file too short to gate (under 400 ms) comes back fully unmasked, with
    coverage 1.0. Gating there would either reject everything or accept
    everything on the strength of a single partial block, and accepting is at
    least honest about having made no judgement.
    """
    samples = np.atleast_2d(samples)
    channels, frames = samples.shape
    if frames == 0:
        return np.zeros(0, dtype=bool), 0.0

    block = int(round(loudness._BLOCK_S * sample_rate))
    step = int(round(loudness._BLOCK_S * (1.0 - loudness._OVERLAP) * sample_rate))
    if block <= 0 or step <= 0 or frames < block:
        return np.ones(frames, dtype=bool), 1.0

    weighted = loudness.k_weight(samples, sample_rate)
    weights = loudness.channel_weights(channels)
    mean_squares = loudness._block_mean_squares(
        weighted, sample_rate, loudness._BLOCK_S, loudness._BLOCK_S * (1.0 - loudness._OVERLAP)
    )
    if mean_squares.shape[0] == 0:
        return np.ones(frames, dtype=bool), 1.0

    block_loudness = loudness._blocks_to_loudness(mean_squares, weights)
    above_absolute = block_loudness > loudness._ABSOLUTE_GATE_LUFS
    if not np.any(above_absolute):
        return np.zeros(frames, dtype=bool), 0.0

    first_pass = loudness._blocks_to_loudness(
        mean_squares[above_absolute].mean(axis=0, keepdims=True), weights
    )[0]
    keep = above_absolute & (block_loudness > first_pass + loudness._RELATIVE_GATE_LU)
    if not np.any(keep):
        keep = above_absolute

    # Blocks overlap by 75%, so a frame belongs to up to four of them. It is
    # kept if ANY covering block survived: the alternative (requiring all four)
    # would erode 300 ms from each end of every phrase, which is exactly where
    # the transients that matter most to a compressor decision live.
    mask = np.zeros(frames, dtype=bool)
    starts = np.nonzero(keep)[0] * step
    for start in starts:
        mask[start : start + block] = True

    return mask, float(np.count_nonzero(mask)) / float(frames)


def gated_segments(mask: np.ndarray, min_length: int = 1) -> list[tuple[int, int]]:
    """Contiguous ``[start, end)`` runs of the mask, longest first is not implied.

    Measurements that need continuous time — envelope modulation, decay
    fitting — cannot simply index the mask, because concatenating disjoint
    regions manufactures discontinuities at every join and those look exactly
    like transients.
    """
    if mask.size == 0:
        return []
    edges = np.diff(mask.astype(np.int8))
    starts = list(np.nonzero(edges == 1)[0] + 1)
    ends = list(np.nonzero(edges == -1)[0] + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask))
    return [(s, e) for s, e in zip(starts, ends) if e - s >= min_length]
