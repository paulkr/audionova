"""Stereo correlation, width, and mono-compatibility loss.

MONO COMPATIBILITY IS THE MEASUREMENT THAT MATTERS, and it is the one most
often skipped. Correlation tells you the channels differ; it does not tell you
what happens when they are summed, which is what a phone speaker, a club PA,
and a laptop all do. A stereo widener can produce a beautiful correlation
number and a 4 dB hole at 200 Hz on summing.

Reported PER BAND, because the damage is never broadband. Low-frequency side
content is where width does its harm — it costs headroom, it cuts badly to
vinyl, and it partially cancels on any single-speaker playback — while the same
amount of side energy at 8 kHz is harmless and desirable. A single correlation
figure averages those two into a number that describes neither.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from backend.measure import spectrum
from backend.measure.dynamics import filtfilt_padlen


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Pearson correlation, with the degenerate cases pinned.

    A silent band is not decorrelated, it is silent — returning 0 there would
    read as "extremely wide" for a band that contains nothing at all.
    """
    if left.size == 0:
        return 1.0
    left = left - float(np.mean(left))
    right = right - float(np.mean(right))
    denom = float(np.sqrt(np.sum(left * left) * np.sum(right * right)))
    if denom <= 1e-30:
        return 1.0
    return float(np.clip(np.sum(left * right) / denom, -1.0, 1.0))


def broadband_correlation(samples: np.ndarray, mask: np.ndarray | None = None) -> float:
    samples = np.atleast_2d(samples)
    if samples.shape[0] < 2:
        return 1.0
    if mask is not None and mask.size == samples.shape[1]:
        samples = samples[:, mask]
    return _correlation(samples[0], samples[1])


def mono_compatibility_loss_db(
    samples: np.ndarray, mask: np.ndarray | None = None
) -> float:
    """How much level is lost when the channels are summed to mono.

    Measured against the energy that WOULD survive if the channels were
    uncorrelated, not against the stereo total. Summing two uncorrelated
    channels of equal power gives 3 dB less than their sum of powers, and that
    3 dB is not a loss — it is arithmetic. Reporting it as one would flag every
    normally-wide mix as broken.

    What this returns is the loss beyond that: the part caused by actual
    cancellation. Zero for uncorrelated or identical channels, large and
    positive for anything approaching polarity inversion.
    """
    samples = np.atleast_2d(samples)
    if samples.shape[0] < 2:
        return 0.0
    if mask is not None and mask.size == samples.shape[1]:
        samples = samples[:, mask]
    if samples.shape[1] == 0:
        return 0.0

    left, right = samples[0], samples[1]
    mono = 0.5 * (left + right)

    mono_power = float(np.mean(mono * mono))
    # The power a mono sum would have if the two channels shared no content:
    # 0.25 * (P_L + P_R).
    independent_power = 0.25 * (float(np.mean(left * left)) + float(np.mean(right * right)))

    if independent_power <= 1e-30:
        return 0.0
    if mono_power <= 1e-30:
        return 60.0  # complete cancellation; capped so it stays a usable number

    loss = 10.0 * np.log10(independent_power / mono_power)
    # Identical channels give mono power ABOVE the independent estimate, i.e. a
    # negative "loss". That is a gain, not a compatibility problem, so it
    # floors at zero.
    return float(max(0.0, loss))


def per_band_stereo(
    samples: np.ndarray,
    sample_rate: float,
    mask: np.ndarray | None = None,
    band_count: int = 8,
) -> list[dict[str, float]]:
    """Correlation, side-to-mid ratio and mono loss, per ERB-spaced band."""
    samples = np.atleast_2d(samples)
    if samples.shape[0] < 2:
        return []

    high = min(16000.0, sample_rate / 2.0 * 0.9)
    edges, centres = spectrum.erb_band_edges(50.0, high, band_count)

    out: list[dict[str, float]] = []
    for i in range(band_count):
        low = max(edges[i], 10.0)
        top = min(edges[i + 1], sample_rate / 2.0 * 0.99)
        if top <= low:
            continue
        sos = signal.butter(4, [low, top], btype="bandpass", fs=sample_rate, output="sos")
        if samples.shape[-1] <= filtfilt_padlen(sos):
            continue
        # Zero-phase, so the band-splitting itself cannot introduce the phase
        # difference the measurement is looking for.
        band = signal.sosfiltfilt(sos, samples[:2], axis=-1)
        if mask is not None and mask.size == band.shape[1]:
            band = band[:, mask]
        if band.shape[1] == 0:
            continue

        mid = 0.5 * (band[0] + band[1])
        side = 0.5 * (band[0] - band[1])
        mid_power = float(np.mean(mid * mid))
        side_power = float(np.mean(side * side))

        out.append({
            "centre_hz": float(centres[i]),
            "correlation": _correlation(band[0], band[1]),
            "side_to_mid_db": _side_to_mid_db(side_power, mid_power),
            "mono_loss_db": mono_compatibility_loss_db(band),
        })
    return out


def _side_to_mid_db(side_power: float, mid_power: float) -> float | None:
    """Side relative to mid, with +/-60 dB standing in at the degenerate ends.

    A band with no mid is all side — a polarity-inverted pair — and used to
    report -60, the "mono" end, exactly backwards. A band with neither has no
    stereo field to describe, so it is missing rather than either end.
    """
    if side_power <= 1e-30 and mid_power <= 1e-30:
        return None
    if side_power <= 1e-30:
        return -60.0
    if mid_power <= 1e-30:
        return 60.0
    return float(10.0 * np.log10(side_power / mid_power))


def stereo_type(samples: np.ndarray, mask: np.ndarray | None = None) -> tuple[str, float]:
    """Classify the stereo field, with a confidence.

    Deterministic rather than learned. The distinctions that matter — is this
    actually mono, is it dual-mono pretending to be stereo, is it wide, is it
    out of phase — are all decidable from correlation and side energy, and a
    model would only add a way to be confidently wrong about them.
    """
    samples = np.atleast_2d(samples)
    if samples.shape[0] < 2:
        return "mono", 1.0

    if mask is not None and mask.size == samples.shape[1]:
        samples = samples[:, mask]
    if samples.shape[1] == 0:
        return "unknown", 0.0

    left, right = samples[0], samples[1]
    side_power = float(np.mean(((left - right) * 0.5) ** 2))
    mid_power = float(np.mean(((left + right) * 0.5) ** 2))
    correlation = _correlation(left, right)

    if mid_power <= 1e-30 and side_power <= 1e-30:
        return "silent", 1.0

    # Dual mono: the side signal is not merely quiet, it is absent. The
    # threshold is far below anything a real stereo recording produces.
    if side_power <= mid_power * 1e-6:
        return "dual_mono", 1.0

    if correlation < -0.3:
        # Strongly negative correlation almost always means an inverted
        # channel rather than an artistic choice, and it is worth surfacing
        # loudly because it survives every other measurement looking fine.
        return "out_of_phase", float(min(1.0, abs(correlation)))

    side_to_mid_db = 10.0 * np.log10(max(side_power, 1e-30) / max(mid_power, 1e-30))
    if side_to_mid_db < -20.0:
        return "near_mono", float(np.clip((-side_to_mid_db - 20.0) / 10.0, 0.3, 1.0))
    if side_to_mid_db > 0.0:
        return "wide", float(np.clip(side_to_mid_db / 6.0, 0.3, 1.0))
    return "conventional_stereo", 0.8
