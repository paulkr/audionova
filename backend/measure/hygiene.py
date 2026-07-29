"""Defects in the source: clipping, DC offset, noise floor, dead channels.

These are the checks that change what the planner is allowed to do, rather than
merely informing it. A clipped source cannot be fixed by EQ and will sound
worse after any drive stage. A DC offset eats headroom and makes every limiter
decision wrong. A noise floor at -45 dBFS means an upward expander or a bright
shelf will amplify hiss into the foreground.

Each one produces a warning with an explicit instruction, not just a number.
The plan calls for a deterministic ``warnings[]`` array precisely because
"clipping_percent: 0.4" requires the reader to know what 0.4 means, whereas
"0.4% of samples are clipped; do not add drive" does not.
"""

from __future__ import annotations

import numpy as np


def clipping(
    samples: np.ndarray, threshold: float = 0.999, flatness: float = 1e-4
) -> dict[str, float]:
    """Runs of consecutive samples pinned FLAT at full scale.

    RUNS, not individual samples. A single sample at 0.9999 is a sample that
    happened to land near the top; a run of them is a waveform whose peak has
    been flattened, which is what clipping actually is.

    FLAT runs, not merely loud ones, and this is the part that is easy to get
    wrong. A clean 100 Hz sine peaking at exactly 0 dBFS spends about five
    consecutive samples within 0.0005 of full scale, purely because a sine is
    quadratic near its peak and 480 samples wide at 48 kHz. A run-length test
    alone therefore reports 2% clipping on a completely undamaged file — and
    that fires the "do not add drive" warning on material that would take
    drive perfectly well.

    What distinguishes the two is curvature. A clipped peak is a horizontal
    line because every sample was clamped to the same value; a sine's peak is
    curved. So a run counts only if its peak-to-peak spread is a negligible
    fraction of its level. The tolerance is relative rather than absolute so
    that material clipped and then attenuated is still recognised.
    """
    samples = np.atleast_2d(samples)
    if samples.size == 0:
        return {"clipped_sample_percent": 0.0, "longest_run_samples": 0.0, "runs": 0.0}

    total_clipped = 0
    longest = 0
    run_count = 0

    for channel in samples:
        magnitude = np.abs(channel)
        # Relative to this channel's own peak, so a file that was clipped and
        # then turned down is still caught. An absolute threshold would only
        # ever find clipping that survived at exactly full scale.
        ceiling = float(np.max(magnitude)) if magnitude.size else 0.0
        if ceiling <= 0.0:
            continue

        at_ceiling = magnitude >= threshold * ceiling
        if not np.any(at_ceiling):
            continue

        padded = np.concatenate([[False], at_ceiling, [False]])
        edges = np.diff(padded.astype(np.int8))
        starts = np.nonzero(edges == 1)[0]
        ends = np.nonzero(edges == -1)[0]

        for start, end in zip(starts, ends):
            if end - start < 3:
                continue
            run = magnitude[start:end]
            spread = float(np.max(run) - np.min(run))
            level = float(np.max(run))
            if level <= 0.0 or spread / level > flatness:
                continue  # curved, so it is a peak rather than a clamp
            total_clipped += int(end - start)
            longest = max(longest, int(end - start))
            run_count += 1

    return {
        "clipped_sample_percent": 100.0 * total_clipped / float(samples.size),
        "longest_run_samples": float(longest),
        "runs": float(run_count),
    }


def dc_offset(samples: np.ndarray) -> list[float]:
    """Mean level per channel, in dBFS relative to full scale."""
    samples = np.atleast_2d(samples)
    if samples.shape[1] == 0:
        return [float("-inf")] * samples.shape[0]
    out = []
    for channel in samples:
        mean = abs(float(np.mean(channel)))
        out.append(20.0 * np.log10(mean) if mean > 0.0 else float("-inf"))
    return out


def noise_floor_db(
    samples: np.ndarray, sample_rate: float, window_s: float = 0.050
) -> tuple[float, float]:
    """Level of the quietest sustained passage, and a confidence.

    Estimated from the 5th percentile of short-window RMS rather than the
    minimum. The minimum finds the single quietest window, which on any file
    with a hard edit or a fade is digital silence, and reporting -inf as a
    noise floor is both true and useless.

    Confidence is low when there is no quiet passage to measure. A wall-to-wall
    loud master genuinely does not reveal its noise floor, and the honest
    answer is "cannot tell" rather than a number derived from the quietest
    moment of a loud signal.
    """
    samples = np.atleast_2d(samples)
    channels, frames = samples.shape
    window = max(1, int(round(window_s * sample_rate)))
    if frames < window * 4:
        return float("-inf"), 0.0

    mono_power = np.mean(samples * samples, axis=0)
    cumulative = np.concatenate([[0.0], np.cumsum(mono_power)])
    starts = np.arange(0, frames - window + 1, window // 2)
    mean_square = (cumulative[starts + window] - cumulative[starts]) / window

    # Windows of exact digital silence are not noise floor, they are silence.
    audible = mean_square[mean_square > 1e-24]
    if audible.size < 4:
        return float("-inf"), 0.0

    floor = float(10.0 * np.log10(np.percentile(audible, 5.0)))
    loud = float(10.0 * np.log10(np.percentile(audible, 95.0)))

    # If the quiet end is within 20 dB of the loud end, nothing in this file is
    # actually quiet and the "floor" is just the low end of the programme.
    span = loud - floor
    confidence = float(np.clip((span - 20.0) / 30.0, 0.0, 1.0))
    return floor, confidence


def dead_channels(samples: np.ndarray, threshold_db: float = -90.0) -> list[int]:
    """Channels carrying nothing.

    A stereo file with a silent right channel is extremely common — it is what
    a mono source recorded into a stereo track looks like — and processing it
    as stereo produces a hard-panned result that the user did not ask for.
    """
    samples = np.atleast_2d(samples)
    dead = []
    for index, channel in enumerate(samples):
        if channel.size == 0:
            dead.append(index)
            continue
        peak = float(np.max(np.abs(channel)))
        level = 20.0 * np.log10(peak) if peak > 0.0 else float("-inf")
        if level < threshold_db:
            dead.append(index)
    return dead


def nan_or_inf_count(samples: np.ndarray) -> int:
    samples = np.atleast_2d(samples)
    return int(np.count_nonzero(~np.isfinite(samples)))
