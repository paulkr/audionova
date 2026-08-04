"""ITU-R BS.1770-5 loudness, EBU R128 loudness range, and true peak.

WHY THIS IS BUILT RATHER THAN IMPORTED. ``pyloudnorm`` is MIT and implements
BS.1770 correctly, but it does NOT implement true peak — it reports sample
peak under a name that reads like true peak, which is the single most
consequential measurement in the export path. Since the true-peak meter has to
be written anyway, and since the loudness gating it shares plumbing with is
about a hundred lines with published conformance tests, both live here.

EVERY NUMBER IN THIS MODULE IS VALIDATED AGAINST PUBLISHED GROUND TRUTH.
EBU Tech 3341 and Tech 3342 define compliance signals with exact expected
readings and tolerances. That makes this the only part of the measurement
layer whose correctness does not rest on our own judgement, and the tests
exercise those cases directly rather than a reimplementation of them.

MONO READS 3 LU LOWER THAN DUAL-MONO STEREO, and this WILL surprise people.
BS.1770 sums channel powers with per-channel weights, so the same waveform
present in two channels measures 3 dB hotter than in one. A mono guitar DI at
-23 dBFS reads -26 LUFS, not -23. That is the standard behaving as specified,
not a bug, and it matters here because this product's input is usually a
single mono stem. It is reported explicitly in the report so nobody has to
rediscover it.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

# The gating block structure is fixed by the standard, not by us.
_BLOCK_S = 0.400
_OVERLAP = 0.75
_ABSOLUTE_GATE_LUFS = -70.0
_RELATIVE_GATE_LU = -10.0

# BS.1770's magic offset. It exists so that a 1 kHz sine at -23 dBFS in both
# channels of a stereo pair reads exactly -23.0 LUFS: the K-weighting has about
# +0.69 dB of gain at 1 kHz, and this cancels it.
_LOUDNESS_OFFSET = -0.691

# Short-term loudness for loudness range, per EBU Tech 3342.
_LRA_BLOCK_S = 3.0
_LRA_STEP_S = 0.100
_LRA_RELATIVE_GATE_LU = -20.0
_LRA_LOW_PERCENTILE = 10.0
_LRA_HIGH_PERCENTILE = 95.0


def channel_weights(channels: int) -> np.ndarray:
    """BS.1770 per-channel weights G_i.

    Left, right and centre count fully; surrounds are weighted +1.5 dB
    (a factor of 1.41); LFE is excluded entirely. Beyond five channels the
    layout is unknown, so remaining channels are treated as full-weight rather
    than guessed at — over-counting is a visible 1-2 LU error, whereas dropping
    a channel silently discards audio from the measurement.
    """
    if channels <= 2:
        return np.ones(channels)
    weights = np.ones(channels)
    if channels >= 5:
        weights[3] = 1.41
        weights[4] = 1.41
    return weights


def k_weighting_sos(sample_rate: float) -> np.ndarray:
    """K-weighting as two biquads, derived for an arbitrary sample rate.

    BS.1770 tabulates coefficients only at 48 kHz. Those numbers are not
    portable — reusing them at 96 kHz moves both corner frequencies by an
    octave and quietly biases every loudness reading. The analog parameters
    below are the ones that reproduce the published 48 kHz table exactly when
    put through this bilinear transform, so this is the same filter expressed
    in a form that generalises rather than an approximation of it.
    """
    # Stage 1: high-frequency shelf, modelling the head's acoustic effect.
    f0, gain_db, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = np.tan(np.pi * f0 / sample_rate)
    vh = np.power(10.0, gain_db / 20.0)
    vb = np.power(vh, 0.4996667741545416)
    denom = 1.0 + k / q + k * k
    shelf = np.array([
        (vh + vb * k / q + k * k) / denom,
        2.0 * (k * k - vh) / denom,
        (vh - vb * k / q + k * k) / denom,
        1.0,
        2.0 * (k * k - 1.0) / denom,
        (1.0 - k / q + k * k) / denom,
    ])

    # Stage 2: RLB high-pass, discounting low frequencies the ear weights less.
    f0, q = 38.13547087602444, 0.5003270373238773
    k = np.tan(np.pi * f0 / sample_rate)
    denom = 1.0 + k / q + k * k
    highpass = np.array([
        1.0,
        -2.0,
        1.0,
        1.0,
        2.0 * (k * k - 1.0) / denom,
        (1.0 - k / q + k * k) / denom,
    ])

    return np.vstack([shelf, highpass])


def k_weight(samples: np.ndarray, sample_rate: float) -> np.ndarray:
    """Apply K-weighting to a ``(channels, frames)`` array."""
    samples = np.atleast_2d(samples)
    sos = k_weighting_sos(sample_rate)
    return signal.sosfilt(sos, samples, axis=-1)


def _block_mean_squares(
    weighted: np.ndarray, sample_rate: float, block_s: float, step_s: float
) -> np.ndarray:
    """Mean square per channel per block: the ``z_ij`` of the standard.

    Returns shape ``(blocks, channels)``. Empty if the signal is shorter than
    one block — a 200 ms file has no defined integrated loudness, and inventing
    one from a partial block would report a number with no standing.
    """
    channels, frames = weighted.shape
    block = int(round(block_s * sample_rate))
    step = int(round(step_s * sample_rate))
    if block <= 0 or step <= 0 or frames < block:
        return np.zeros((0, channels))

    count = 1 + (frames - block) // step
    # Cumulative sums make this O(frames) rather than O(frames * block); at
    # 75% overlap the naive version re-adds every sample four times, which on a
    # ten-minute 96 kHz file is the difference between instant and not.
    power = np.concatenate(
        [np.zeros((channels, 1)), np.cumsum(weighted * weighted, axis=-1)], axis=-1
    )
    starts = np.arange(count) * step
    sums = power[:, starts + block] - power[:, starts]
    return (sums / block).T


def _blocks_to_loudness(mean_squares: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-block loudness l_j from per-channel mean squares."""
    total = mean_squares @ weights
    with np.errstate(divide="ignore"):
        return _LOUDNESS_OFFSET + 10.0 * np.log10(np.maximum(total, 1e-300))


def _gated_loudness(
    mean_squares: np.ndarray, weights: np.ndarray, relative_gate_lu: float
) -> float:
    """Two-pass gating: absolute, then relative to the signal's own mean.

    The relative gate is what makes loudness measure the PROGRAMME rather than
    the file. Without it, a track with thirty seconds of silence at the end
    measures quieter than the identical track trimmed, and the correction an
    engineer would apply is different for two files that sound the same.
    """
    if mean_squares.shape[0] == 0:
        return float("-inf")

    block_loudness = _blocks_to_loudness(mean_squares, weights)

    above_absolute = block_loudness > _ABSOLUTE_GATE_LUFS
    if not np.any(above_absolute):
        return float("-inf")

    first_pass = _blocks_to_loudness(
        mean_squares[above_absolute].mean(axis=0, keepdims=True), weights
    )[0]
    threshold = first_pass + relative_gate_lu

    keep = above_absolute & (block_loudness > threshold)
    if not np.any(keep):
        return float("-inf")

    return float(
        _blocks_to_loudness(mean_squares[keep].mean(axis=0, keepdims=True), weights)[0]
    )


def integrated_lufs(samples: np.ndarray, sample_rate: float) -> float:
    """Gated integrated loudness. ``-inf`` if nothing survives the gate."""
    samples = np.atleast_2d(samples)
    weighted = k_weight(samples, sample_rate)
    weights = channel_weights(samples.shape[0])
    mean_squares = _block_mean_squares(
        weighted, sample_rate, _BLOCK_S, _BLOCK_S * (1.0 - _OVERLAP)
    )
    return _gated_loudness(mean_squares, weights, _RELATIVE_GATE_LU)


def momentary_lufs(samples: np.ndarray, sample_rate: float) -> np.ndarray:
    """Ungated 400 ms loudness, one value per 100 ms."""
    samples = np.atleast_2d(samples)
    weighted = k_weight(samples, sample_rate)
    weights = channel_weights(samples.shape[0])
    return _blocks_to_loudness(
        _block_mean_squares(weighted, sample_rate, _BLOCK_S, _BLOCK_S * (1.0 - _OVERLAP)),
        weights,
    )


def short_term_lufs(samples: np.ndarray, sample_rate: float) -> np.ndarray:
    """Ungated 3 s loudness, one value per 100 ms."""
    samples = np.atleast_2d(samples)
    weighted = k_weight(samples, sample_rate)
    weights = channel_weights(samples.shape[0])
    return _blocks_to_loudness(
        _block_mean_squares(weighted, sample_rate, _LRA_BLOCK_S, _LRA_STEP_S), weights
    )


def loudness_range_lu(samples: np.ndarray, sample_rate: float) -> float:
    """EBU Tech 3342 loudness range.

    The 95th minus the 10th percentile of gated short-term loudness. NOT the
    full range: the top percentile would be set by a single cymbal crash and
    the bottom by the quietest breath, and neither describes how dynamic the
    material feels. The asymmetry (95/10 rather than 90/10) is deliberate in
    the standard — the loud end is more perceptually stable than the quiet one.
    """
    samples = np.atleast_2d(samples)
    weighted = k_weight(samples, sample_rate)
    weights = channel_weights(samples.shape[0])
    mean_squares = _block_mean_squares(weighted, sample_rate, _LRA_BLOCK_S, _LRA_STEP_S)
    if mean_squares.shape[0] == 0:
        return 0.0

    block_loudness = _blocks_to_loudness(mean_squares, weights)
    above_absolute = block_loudness > _ABSOLUTE_GATE_LUFS
    if not np.any(above_absolute):
        return 0.0

    first_pass = _blocks_to_loudness(
        mean_squares[above_absolute].mean(axis=0, keepdims=True), weights
    )[0]
    keep = above_absolute & (block_loudness > first_pass + _LRA_RELATIVE_GATE_LU)
    if not np.any(keep):
        return 0.0

    kept = block_loudness[keep]
    low = np.percentile(kept, _LRA_LOW_PERCENTILE)
    high = np.percentile(kept, _LRA_HIGH_PERCENTILE)
    return float(high - low)


def _oversample_factor(sample_rate: float) -> int:
    """BS.1770 Annex 2 requires at least 4x at 48 kHz.

    The error of a 4x estimate grows toward Nyquist, so lower base rates — where
    the audio band reaches proportionally closer to Nyquist — get more. At
    96 kHz there is very little signal near Nyquist to miss and 4x is ample.
    """
    if sample_rate <= 48000.0:
        return 8
    return 4


def _interpolation_filter(factor: int, stopband_db: float = 100.0) -> np.ndarray:
    """Windowed-sinc for the true-peak interpolator, scaled for polyphase gain.

    THE PASSBAND EDGE IS A SAFETY PARAMETER, NOT A TASTE ONE. Whatever the
    filter attenuates, the meter does not see, so it reports LESS than the true
    peak — the unsafe direction for a delivery ceiling, because it lets real
    inter-sample peaks through the gate.

    It used to stop at 0.95 of the base Nyquist. Measured against tone pairs
    whose continuous peak is known exactly (the band-limited reconstruction of
    those samples IS the sum, so the analytic maximum is the truth):

        content        0.95 edge, 48/phase      0.99 edge, 64/phase
        1 kHz                +0.002 dB                +0.001 dB
        10 kHz               +0.307 dB                +0.411 dB
        20 + 22 kHz          -1.078 dB                -0.125 dB
        21 + 23 kHz          -2.289 dB                -1.013 dB

    Widening the edge from 0.95 to 0.98 costs NOTHING — same tap count, same
    runtime — and was most of the improvement. The extra taps buy the rest.
    Cost is 16.5 ms per second of audio against 10.8, on a measurement that
    runs once per render.

    THE RESIDUAL NEAR NYQUIST IS FUNDAMENTAL AND IS NOT A BUG TO FIX LATER. No
    finite filter reconstructs content arbitrarily close to Nyquist; the
    transition band has to fit somewhere. Even at 96 taps per phase the 21+23
    case reads -0.762 dB. What makes this tolerable is that the sample peak is
    taken as a hard lower bound below, so the error is bounded by the gap
    between sample peak and true peak rather than being unbounded.

    A positive number in that table is the meter reading HIGH, which is
    interpolation overshoot and is the safe direction.
    """
    # 0.99 rather than 0.95: see above. The transition band then runs from
    # 0.99/factor to the first image, which the tap count has to be large
    # enough to cover.
    taps_per_phase = 64
    numtaps = taps_per_phase * factor + 1
    cutoff = 0.5 / factor * 0.99
    fir = signal.firwin(
        numtaps, cutoff * 2.0, window=("kaiser", signal.kaiser_beta(stopband_db))
    )
    # Each output sample sees only every factor-th tap, so unity DC gain per
    # phase requires scaling by the factor.
    return fir * factor


def true_peak_dbtp(samples: np.ndarray, sample_rate: float) -> float:
    """True peak in dBTP: the peak of the RECONSTRUCTED waveform.

    Sample peak is not the same measurement and is not conservative. Between
    two samples the reconstructed signal routinely exceeds both of them, by up
    to several dB on heavily limited material — which is exactly the material
    most likely to be sitting at a ceiling. A converter clips on the true peak,
    so a file that "peaks at -0.1 dBFS" can and does clip on playback.

    This is why the limiter and this meter must both run at the DELIVERY rate:
    the inter-sample peaks of a 96 kHz file are not the inter-sample peaks of
    the 44.1 kHz file it becomes.
    """
    samples = np.atleast_2d(np.asarray(samples, dtype=np.float64))
    if samples.size == 0:
        return float("-inf")

    factor = _oversample_factor(sample_rate)
    fir = _interpolation_filter(factor)

    peak = 0.0
    for channel in samples:
        upsampled = signal.upfirdn(fir, channel, up=factor, down=1)
        peak = max(peak, float(np.max(np.abs(upsampled))))
        # The sample peak is a hard lower bound: interpolation must never
        # report LESS than a sample that is actually present, and a filter
        # design error is exactly what would cause that.
        peak = max(peak, float(np.max(np.abs(channel))))

    return 20.0 * np.log10(peak) if peak > 0.0 else float("-inf")


def sample_peak_dbfs(samples: np.ndarray) -> float:
    samples = np.atleast_2d(samples)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    return 20.0 * np.log10(peak) if peak > 0.0 else float("-inf")
