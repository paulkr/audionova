"""Crest factor, level percentiles, and the envelope modulation spectrum.

These are the direct inputs to compressor decisions, and they are here because
the alternative is the planner guessing them from a description of the audio.

WHY PER-BAND CREST AND NOT JUST BROADBAND. Broadband crest factor hides the
thing you most need to know. A bass-heavy mix whose low end is already squashed
flat and whose top end is spiky reads as moderately dynamic overall — the two
cancel. Per-ERB crest shows the actual shape, and it is the difference between
prescribing broadband compression (which flattens the top further) and
prescribing a multiband or dynamic-EQ move (which is what the material needs).

WHY PERCENTILES AND NOT MEAN AND PEAK. A compressor threshold is a decision
about what fraction of the material should be caught. "The 90th percentile of
short-term RMS is -12 dBFS" answers that directly; "the mean is -18 and the
peak is -3" does not, and the arithmetic to get from one to the other requires
knowing a distribution nobody measured.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from backend.measure import spectrum


def _to_db(x: np.ndarray | float) -> np.ndarray | float:
    return 20.0 * np.log10(np.maximum(np.asarray(x, dtype=float), 1e-300))


def short_term_rms_db(
    samples: np.ndarray,
    sample_rate: float,
    mask: np.ndarray | None = None,
    window_s: float = 0.050,
) -> np.ndarray:
    """RMS in dBFS over short windows, gated.

    50 ms is chosen to sit between the two things it must not be. Shorter and
    it tracks individual waveform cycles at bass frequencies, so a 60 Hz note
    reads as violently dynamic. Longer and it averages across a whole note,
    hiding the attack that a compressor's attack time exists to catch.
    """
    samples = np.atleast_2d(samples)
    channels, frames = samples.shape
    window = max(1, int(round(window_s * sample_rate)))
    hop = max(1, window // 2)
    if frames < window:
        return np.zeros(0)

    # Sum power across channels: a stereo pair is one programme.
    power = np.sum(samples * samples, axis=0)
    cumulative = np.concatenate([[0.0], np.cumsum(power)])
    starts = np.arange(0, frames - window + 1, hop)

    if mask is not None and mask.size == frames:
        covered = np.concatenate([[0], np.cumsum(mask.astype(np.int64))])
        starts = starts[(covered[starts + window] - covered[starts]) == window]
    if starts.size == 0:
        return np.zeros(0)

    mean_square = (cumulative[starts + window] - cumulative[starts]) / (window * channels)
    return _to_db(np.sqrt(mean_square))


def level_percentiles_db(
    rms_db: np.ndarray, percentiles: tuple[float, ...] = (5, 10, 25, 50, 75, 90, 95, 99)
) -> dict[str, float]:
    if rms_db.size == 0:
        return {f"p{int(p)}_dbfs": float("-inf") for p in percentiles}
    finite = rms_db[np.isfinite(rms_db)]
    if finite.size == 0:
        return {f"p{int(p)}_dbfs": float("-inf") for p in percentiles}
    return {
        f"p{int(p)}_dbfs": float(np.percentile(finite, p)) for p in percentiles
    }


def crest_factor_db(samples: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Peak minus RMS, in dB, over gated material."""
    samples = np.atleast_2d(samples)
    if mask is not None and mask.size == samples.shape[1]:
        samples = samples[:, mask]
    if samples.size == 0:
        return 0.0

    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(samples * samples)))
    if peak <= 0.0 or rms <= 0.0:
        return 0.0
    return float(_to_db(peak) - _to_db(rms))


def per_erb_crest_db(
    samples: np.ndarray,
    sample_rate: float,
    mask: np.ndarray | None = None,
    band_count: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Crest factor within each of a small number of ERB-spaced bands.

    Twelve bands rather than the twenty-eight used for the spectrum: each one
    needs a filter run over the whole signal, and the question here is "which
    part of the range is squashed", which twelve bands answer as well as
    twenty-eight would at a third of the cost.
    """
    samples = np.atleast_2d(samples)
    high = min(16000.0, sample_rate / 2.0 * 0.9)
    edges, centres = spectrum.erb_band_edges(50.0, high, band_count)

    # Sum to mono for this measurement. Per-channel crest would mostly measure
    # the difference between the channels rather than the dynamics of the
    # material, and the compression decision applies to the programme.
    mono = np.mean(samples, axis=0)

    crests = np.empty(band_count)
    for i in range(band_count):
        low = max(edges[i], 10.0)
        top = min(edges[i + 1], sample_rate / 2.0 * 0.99)
        if top <= low:
            crests[i] = 0.0
            continue
        sos = signal.butter(
            4, [low, top], btype="bandpass", fs=sample_rate, output="sos"
        )
        band = signal.sosfiltfilt(sos, mono)
        # Zero-phase filtering, so the band's transients stay where they were.
        # A minimum-phase filter would smear the attack, and the attack is
        # precisely what a crest factor measures.
        if mask is not None and mask.size == band.size:
            band = band[mask]
        if band.size == 0:
            crests[i] = 0.0
            continue
        peak = float(np.max(np.abs(band)))
        rms = float(np.sqrt(np.mean(band * band)))
        crests[i] = float(_to_db(peak) - _to_db(rms)) if peak > 0 and rms > 0 else 0.0

    return centres, crests


def envelope_modulation_spectrum(
    samples: np.ndarray,
    sample_rate: float,
    mask: np.ndarray | None = None,
    envelope_rate: float = 200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """How fast the level moves, as a spectrum of the amplitude envelope.

    THIS IS THE INPUT TO ATTACK AND RELEASE. A compressor's job is to respond
    at some rate, and the material's own rate of level change is what decides
    whether 5 ms or 50 ms is right. Comparing a source's modulation spectrum
    against a target's, and finding where their ratio crosses 3 dB, gives a
    time constant derived from the audio rather than guessed from a genre.

    Computed on the longest continuous gated segment rather than the
    concatenation of all of them: joining disjoint regions manufactures a step
    at every seam, and steps are broadband — they would fill the modulation
    spectrum with energy that came from the editing, not the performance.
    """
    from backend.measure import gating

    samples = np.atleast_2d(samples)
    mono = np.mean(samples, axis=0)

    if mask is not None and mask.size == mono.size:
        segments = gating.gated_segments(mask, min_length=int(sample_rate))
        if not segments:
            return np.zeros(0), np.zeros(0)
        start, end = max(segments, key=lambda se: se[1] - se[0])
        mono = mono[start:end]

    if mono.size < int(sample_rate):
        return np.zeros(0), np.zeros(0)

    # Rectify and low-pass to get the envelope, then decimate to a rate that
    # comfortably covers musical modulation (up to ~50 Hz) without carrying
    # tens of thousands of useless samples per second.
    decimation = max(1, int(round(sample_rate / envelope_rate)))
    sos = signal.butter(4, envelope_rate / 2.0, btype="low", fs=sample_rate, output="sos")
    envelope = signal.sosfiltfilt(sos, np.abs(mono))[::decimation]
    actual_rate = sample_rate / decimation

    if envelope.size < 64:
        return np.zeros(0), np.zeros(0)

    # Remove the mean: the DC term is the average level, which is already
    # reported elsewhere and would otherwise dominate the spectrum entirely.
    envelope = envelope - float(np.mean(envelope))

    nperseg = int(min(envelope.size, 2 ** np.floor(np.log2(envelope.size))))
    freqs, power = signal.welch(
        envelope, fs=actual_rate, nperseg=nperseg, noverlap=nperseg // 2
    )
    inside = (freqs > 0.2) & (freqs <= 50.0)
    return freqs[inside], power[inside]


def modulation_summary(freqs: np.ndarray, power: np.ndarray) -> dict[str, float]:
    """Reduce the modulation spectrum to numbers a prompt can carry.

    The full spectrum is a hundred numbers and no planner will reason over it
    usefully. These three describe it: where its energy is centred, where the
    bulk of it sits, and how much there is.
    """
    if freqs.size == 0 or float(np.sum(power)) <= 0.0:
        return {
            "centroid_hz": 0.0,
            "p90_hz": 0.0,
            "depth_db": 0.0,
            "confidence": 0.0,
        }
    total = float(np.sum(power))
    centroid = float(np.sum(freqs * power) / total)
    cumulative = np.cumsum(power) / total
    p90 = float(freqs[int(np.searchsorted(cumulative, 0.9))]) if cumulative[-1] >= 0.9 else float(freqs[-1])
    # Depth as the ratio of modulation energy to the envelope's own scale.
    depth = 10.0 * np.log10(total) if total > 0 else 0.0
    return {
        "centroid_hz": centroid,
        "p90_hz": p90,
        "depth_db": float(depth),
        "confidence": 0.8,
    }


def peak_to_loudness_ratio_db(true_peak_dbtp: float, integrated_lufs: float) -> float:
    """PLR: how much headroom sits above the programme level.

    A mastered pop record lands near 8 dB; an unprocessed acoustic
    performance is 18-20. It is the single most informative number about how
    much dynamic processing the material has already had, and therefore about
    how much more it can take before it stops sounding like a performance.
    """
    if not np.isfinite(true_peak_dbtp) or not np.isfinite(integrated_lufs):
        return float("nan")
    return float(true_peak_dbtp - integrated_lufs)
