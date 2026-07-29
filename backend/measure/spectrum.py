"""Long-term average spectrum, band summaries, tilt, and bandwidth.

Everything here is computed over the loudness-gated regions only — see
``gating.py`` for why that is not optional.

WHY BOTH 1/3-OCTAVE AND ERB. They answer different questions and the planner
needs both. Third-octave bands are what an engineer thinks in: they map onto
the frequencies written on an EQ, and "3.15 kHz is +4 dB" is directly
actionable. ERB bands are what the EAR resolves: they are narrow at the bottom
and wide at the top, so equal energy per ERB band is much closer to equal
perceived weight. A tilt computed on third-octave bands is dominated by the
dozen bands above 5 kHz that the ear barely distinguishes; computed per ERB it
is not.

DESCRIPTORS ARE NOT COMPUTED HERE. "Muddy" and "boxy" are decisions, and they
depend on the instrument, the arrangement, and what the user asked for. This
module produces the numbers those decisions are made from and stops there.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

# ANSI S1.11 nominal third-octave centres, 25 Hz to 20 kHz. Nominal rather than
# exact: these are the numbers printed on equipment and written in notes, and a
# report that says "3150 Hz" is readable where one that says "3162.3 Hz" is not.
THIRD_OCTAVE_CENTRES_HZ = np.array([
    25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
    800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
    12500, 16000, 20000,
], dtype=float)


def erb_number(freq_hz: np.ndarray | float) -> np.ndarray | float:
    """Glasberg & Moore ERB-rate scale."""
    return 21.4 * np.log10(4.37 * np.asarray(freq_hz, dtype=float) / 1000.0 + 1.0)


def erb_to_hz(erb: np.ndarray | float) -> np.ndarray | float:
    return (np.power(10.0, np.asarray(erb, dtype=float) / 21.4) - 1.0) * 1000.0 / 4.37


def erb_band_edges(low_hz: float = 50.0, high_hz: float = 16000.0, count: int = 28):
    """Equally spaced on the ERB scale, so each band is one unit of what the ear resolves."""
    edges_erb = np.linspace(erb_number(low_hz), erb_number(high_hz), count + 1)
    edges = erb_to_hz(edges_erb)
    centres = np.sqrt(edges[:-1] * edges[1:])
    return edges, centres


def gated_ltas(
    samples: np.ndarray,
    sample_rate: float,
    mask: np.ndarray | None = None,
    window_s: float = 0.200,
) -> tuple[np.ndarray, np.ndarray]:
    """Power spectral density over gated material only.

    THE WINDOW IS 200 ms, NOT 50, and the reason is band resolution rather
    than smoothing. At 48 kHz a 50 ms window puts FFT bins 23 Hz apart, while
    the 50 Hz third-octave band spans 44.5 to 56.1 Hz — 11.6 Hz wide. The band
    is narrower than the spacing, so it can contain no bin at all and integrate
    to exactly zero. 200 ms brings the spacing to 5.9 Hz, which resolves the
    third-octave bands down to about 63 Hz honestly; below that
    ``band_levels_db`` falls back to interpolating the density.

    200 ms also still fits inside the gate: gated runs are at least one 400 ms
    block, so at 50% hop every kept region contributes windows.

    Welch's method, but with window selection driven by the gate: a window is
    included only if EVERY frame it covers survived. Partial windows are
    dropped rather than included, because a window straddling the edge of a
    phrase contains a step, and a step is broadband — it would put energy at
    every frequency and read as "bright".

    Returns ``(freqs_hz, power)`` with power summed across channels.
    """
    samples = np.atleast_2d(samples)
    channels, frames = samples.shape
    nperseg = int(2 ** np.round(np.log2(max(window_s * sample_rate, 256))))
    if frames < nperseg:
        return np.zeros(0), np.zeros(0)

    hop = nperseg // 2
    starts = np.arange(0, frames - nperseg + 1, hop)
    if mask is not None and mask.size == frames:
        # Cumulative count makes "is every frame in this window kept?" O(1).
        covered = np.concatenate([[0], np.cumsum(mask.astype(np.int64))])
        full = (covered[starts + nperseg] - covered[starts]) == nperseg
        starts = starts[full]
    if starts.size == 0:
        return np.zeros(0), np.zeros(0)

    window = signal.get_window("hann", nperseg)
    # Normalising by the window's power keeps the PSD calibrated, so a sine of
    # amplitude A integrates to A^2/2 regardless of the window chosen.
    scale = 1.0 / (sample_rate * np.sum(window * window))

    accum = np.zeros(nperseg // 2 + 1)
    for channel in samples:
        frames_view = np.lib.stride_tricks.sliding_window_view(channel, nperseg)[starts]
        spectra = np.fft.rfft(frames_view * window, axis=-1)
        power = (np.abs(spectra) ** 2) * scale
        power[:, 1:-1] *= 2.0  # one-sided
        accum += power.mean(axis=0)

    freqs = np.fft.rfftfreq(nperseg, 1.0 / sample_rate)
    return freqs, accum


def band_levels_db(
    freqs: np.ndarray, power: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """Integrate a PSD into bands and express each in dB.

    Integrating rather than sampling: a band's level is the energy inside it,
    not the PSD at its centre. Sampling would make a narrow resonance sitting
    between two bin centres disappear entirely.

    A BAND NARROWER THAN THE BIN SPACING CONTAINS NO BINS, and summing over
    nothing gives exactly zero, which becomes -inf dB. That is not a quiet
    edge case: at any practical window length the bottom few third-octave
    bands are narrower than the spacing, so a naive implementation reports
    -inf for 25, 31.5 and 40 Hz on every file ever measured. Downstream that
    either propagates as -inf or gets substituted with a plausible-looking
    constant, and the second is worse — it puts a fabricated number in a
    measurement report.

    So when a band catches no bin, its energy is estimated as the interpolated
    density at its centre times its width. That is the same quantity the sum
    approximates, evaluated the only way the available resolution allows.
    """
    if freqs.size == 0:
        return np.full(len(edges) - 1, -np.inf)
    bin_width = freqs[1] - freqs[0] if freqs.size > 1 else 1.0
    levels = np.empty(len(edges) - 1)
    for i in range(len(edges) - 1):
        inside = (freqs >= edges[i]) & (freqs < edges[i + 1])
        if np.any(inside):
            energy = float(np.sum(power[inside])) * bin_width
        else:
            centre = np.sqrt(edges[i] * edges[i + 1])
            density = float(np.interp(centre, freqs, power, left=0.0, right=0.0))
            energy = density * float(edges[i + 1] - edges[i])
        levels[i] = 10.0 * np.log10(energy) if energy > 0.0 else -np.inf
    return levels


def third_octave_levels_db(
    freqs: np.ndarray, power: np.ndarray, sample_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    centres = THIRD_OCTAVE_CENTRES_HZ[THIRD_OCTAVE_CENTRES_HZ < sample_rate / 2.0]
    ratio = 2.0 ** (1.0 / 6.0)
    edges = np.concatenate([centres / ratio, [centres[-1] * ratio]])
    return centres, band_levels_db(freqs, power, edges)


def erb_levels_db(
    freqs: np.ndarray, power: np.ndarray, sample_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    high = min(16000.0, sample_rate / 2.0 * 0.9)
    edges, centres = erb_band_edges(50.0, high)
    return centres, band_levels_db(freqs, power, edges)


def spectral_tilt_db_per_octave(
    centres: np.ndarray,
    levels_db: np.ndarray,
    low_hz: float = 100.0,
    high_hz: float = 10000.0,
) -> tuple[float, float]:
    """Least-squares slope of BAND LEVEL against log2(frequency).

    ZERO MEANS PINK, NOT WHITE, and getting this backwards would invert every
    tonal decision made from it. These are band-integrated levels, and
    fractional-octave bands get wider in proportion to frequency, so a flat
    power spectral density (white noise) puts 3 dB more energy in each
    successive octave and reads as +3 dB/octave here. Pink noise — equal energy
    per octave, and what an engineer means by "neutral" — reads as 0.

    This is the convention an RTA displays and the one a mix engineer reads
    tilt in, which is why it is the one reported. The alternative (PSD slope,
    where white is 0 and pink is -3) is equally defensible in isolation and
    would be wrong here, because the number is going to a planner that reasons
    about it the way an engineer would.

    Returns ``(slope_db_per_octave_re_pink, residual_db)``, where the residual
    is the RMS deviation of the band levels from the fitted line.

    RESIDUAL RATHER THAN R-SQUARED, and the difference is not cosmetic.
    R-squared measures the fraction of variance the SLOPE explains, so a
    perfectly pink spectrum — flat, zero slope, and the single best case there
    is — scores near zero because there is no slope-explained variance to find.
    Using it would flag every neutral source as "tilt unreliable", which is a
    false alarm on the most common good input.

    The RMS residual has no such degeneracy: it is small when a line describes
    the spectrum and large when one does not, at any slope. It is also in dB,
    so the threshold for "this tilt does not describe the source" is a
    quantity an engineer can reason about rather than a unitless fraction.
    """
    inside = (centres >= low_hz) & (centres <= high_hz) & np.isfinite(levels_db)
    if np.count_nonzero(inside) < 4:
        return 0.0, 0.0

    x = np.log2(centres[inside])
    y = levels_db[inside]
    slope, intercept = np.polyfit(x, y, 1)

    predicted = slope * x + intercept
    residual_db = float(np.sqrt(np.mean((y - predicted) ** 2)))
    return float(slope), residual_db


def bandwidth_limit_hz(
    freqs: np.ndarray, power: np.ndarray, sample_rate: float, drop_db: float = 30.0
) -> tuple[float, float]:
    """Highest frequency carrying real energy, and how confident we are.

    THIS IS WHY IT MATTERS. A file that has been through a low-bitrate MP3 is
    lopped at 15 or 16 kHz with a brick wall. Prescribing an "air" shelf at
    12 kHz for such a file does nothing audible at all — the planner will have
    spent a band, and the user will hear no change and conclude the tool does
    not work. The validator uses this to reject air-band moves outright.

    Detected as a cliff rather than a threshold crossing: real music rolls off
    gently, and a codec's lowpass is a wall. Confidence reflects how wall-like
    the drop actually is.
    """
    if freqs.size == 0:
        return float(sample_rate / 2.0), 0.0

    with np.errstate(divide="ignore"):
        levels = 10.0 * np.log10(np.maximum(power, 1e-300))

    reference_band = (freqs >= 500.0) & (freqs <= 4000.0)
    if not np.any(reference_band):
        return float(sample_rate / 2.0), 0.0
    reference = float(np.mean(levels[reference_band]))

    # Smooth over a third of an octave so a single deep bin does not read as a
    # cliff. Log-spaced, so the smoothing is uniform in the domain the cliff
    # lives in.
    search = (freqs >= 4000.0) & (freqs < sample_rate / 2.0 * 0.999)
    if not np.any(search):
        return float(sample_rate / 2.0), 0.0

    search_freqs = freqs[search]
    search_levels = levels[search]
    smoothed = np.array([
        float(np.mean(search_levels[(search_freqs >= f / 1.12) & (search_freqs <= f * 1.12)]))
        for f in search_freqs
    ])

    below = np.nonzero(smoothed < reference - drop_db)[0]
    if below.size == 0:
        return float(search_freqs[-1]), 0.2

    limit = float(search_freqs[below[0]])

    # Confidence from the steepness of the drop: a codec wall loses 30 dB in
    # well under a third of an octave, whereas a dark performance takes
    # octaves to do the same thing and should not be reported as a limit.
    window = (search_freqs >= limit / 1.26) & (search_freqs <= limit * 1.26)
    if np.count_nonzero(window) >= 3:
        span_db = float(np.max(smoothed[window]) - np.min(smoothed[window]))
        confidence = float(np.clip(span_db / 40.0, 0.0, 1.0))
    else:
        confidence = 0.3

    return limit, confidence
