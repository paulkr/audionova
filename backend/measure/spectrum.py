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


def hf_rolloff_db_per_octave(
    centres: np.ndarray, levels_db: np.ndarray, low_hz: float = 6300.0
) -> float:
    """Slope of the top end, which is a tonal fact rather than a defect.

    Separated from ``bandwidth_limit_hz`` because the two answer different
    questions and conflating them produces exactly the wrong instruction. A
    dark source has a steep rolloff and CAN be brightened; a codec-limited
    source has a wall and cannot. Reporting the first as the second tells a
    planner not to touch the air band on a vocal that would respond to it.
    """
    usable = (centres >= low_hz) & np.isfinite(levels_db)
    if np.count_nonzero(usable) < 3:
        return 0.0
    x = np.log2(centres[usable])
    slope, _ = np.polyfit(x, levels_db[usable], 1)
    return float(slope)


def bandwidth_limit_hz(
    freqs: np.ndarray,
    power: np.ndarray,
    sample_rate: float,
    min_slope_db_per_octave: float = 150.0,
    min_drop_db: float = 20.0,
    max_depth_db: float = 60.0,
) -> tuple[float | None, float]:
    """The frequency of a CODEC WALL, or ``None`` if there is not one.

    THIS RETURNS None FOR MOST REAL AUDIO. The first version looked for the
    point where the level fell 30 dB below the midrange, which every dark
    source crosses eventually: measured on real stems it put a bass at 4 kHz,
    a vocal at 10.5 kHz and an acoustic guitar at 15 kHz, and warned that two
    of the three were lossily encoded. All three had continuous energy to
    20 kHz. The instruction it produced — "any shelf above that frequency will
    be inaudible; do not spend a band on air" — was wrong, high-severity, and
    aimed at exactly the material an engineer would most want to brighten.

    What distinguishes a codec is not how far the level falls but HOW FAST.
    Measured on a 1/12-octave grid over real material, the separation is not
    marginal:

        acoustic guitar, bass, vocal    steepest 3 to 37 dB/octave
        MP3 at 128, 96 and 64 kbps      steepest 425 to 443 dB/octave

    An order of magnitude, with nothing in between, because one is a source
    running out of energy and the other is a brickwall filter. The threshold
    sits at 150 dB/octave, far above anything acoustic and far below anything
    encoded.

    The fine grid matters as much as the threshold. Third-octave bands give
    only three points above 10 kHz, which cannot localise a wall at all — the
    64 kbps cutoff at 15.9 kHz falls between the 16 k and 20 k bands and
    vanishes into a single unremarkable step.

    A cliff is also only considered where there is still something to cut off.
    Once the level is 60 dB below the midrange the content is inaudible, and a
    source with no noise floor — anything heavily filtered, or produced in
    float and never dithered — keeps falling toward numerical zero, which on
    any log scale looks arbitrarily steep. Without this guard a steeply
    low-passed signal reports a wall at 21 kHz, in a region that contributes
    nothing either way.

    THE CONFIDENCE MEASURES DISTANCE FROM THE DECISION, not steepness. Both
    premises — a steep enough cliff, and a cliff with something left to cut off
    — have thresholds, and a reading that sits a decibel from either of them is
    a coin flip however impressive the slope looks. The returned value is the
    weaker of the two margins, so it falls to zero as the measurement
    approaches the boundary from either side. See the comments at the return
    for the numbers that set the scale.
    """
    if freqs.size == 0:
        return None, 0.0

    top_hz = sample_rate / 2.0 * 0.96
    if top_hz <= 5000.0:
        return None, 0.0

    # 1/12-octave grid: fine enough to localise a wall, coarse enough that a
    # single noisy bin cannot look like one.
    grid = np.exp(np.linspace(np.log(2000.0), np.log(top_hz), 120))
    levels = np.empty(grid.size)
    edge = 2.0 ** (1.0 / 24.0)
    for i, centre in enumerate(grid):
        inside = (freqs >= centre / edge) & (freqs < centre * edge)
        mean_power = float(np.mean(power[inside])) if np.any(inside) else 0.0
        levels[i] = 10.0 * np.log10(mean_power) if mean_power > 0.0 else -np.inf

    usable = np.isfinite(levels)
    if np.count_nonzero(usable) < 20:
        return None, 0.0
    grid, levels = grid[usable], levels[usable]

    # Slope measured over a third of an octave, which is wide enough to be
    # robust and narrow enough not to average the cliff away.
    span = 4
    if grid.size <= span:
        return None, 0.0
    octaves = np.log2(grid[span:] / grid[:-span])
    drops = levels[:-span] - levels[span:]
    slopes = drops / octaves

    # Only look above 5 kHz: musical structure lower down can be steep without
    # being a bandwidth limit. And only where the level is still meaningful,
    # for the reason in the docstring.
    reference_band = (grid >= 2000.0) & (grid <= 4000.0)
    reference = float(np.mean(levels[reference_band])) if np.any(reference_band) else float(np.max(levels))
    depths = reference - levels[:-span]
    candidates = (grid[:-span] >= 5000.0) & (depths < max_depth_db)
    if not np.any(candidates):
        return None, 0.0

    masked = np.where(candidates, slopes, -np.inf)
    best = int(np.argmax(masked))

    # THE TWO SLOPE GATES ARE ONE GATE IN TWO UNITS. A drop is a slope times
    # the width it was measured over, so `drops[best] < min_drop_db` is
    # `slopes[best] < min_drop_db / octaves[best]`. Over this grid the span is
    # 0.118 octaves, which makes the drop gate bind at 169 dB/octave — stricter
    # than the 150 the slope gate names, so the slope gate never decides
    # anything on its own. Both are kept because they express different
    # intentions and a future change to the grid or the span could reorder
    # them, but the margin below has to be measured from whichever one actually
    # binds. Measuring it from the nominal 150 reported 0.09 confidence at the
    # exact frequency where the verdict changed.
    effective_min_slope = max(min_slope_db_per_octave, min_drop_db / float(octaves[best]))
    if masked[best] < effective_min_slope:
        return None, 0.0

    limit = float(grid[best])

    # CONFIDENCE IS DISTANCE FROM THE DECISION, NOT STEEPNESS OF THE CLIFF, and
    # the difference is the whole point of this number. The previous version
    # read the slope alone, so it reported 0.88 on the demo vocal chain's
    # output — a verdict that a 1 dB midrange bell flips, because the cliff it
    # found sat 55.5 dB down against a 60 dB guard. A planner told 0.88 stops
    # spending a band on air; a planner told the truth does not.
    #
    # There are two independent premises, and the verdict needs both:
    #
    #   the cliff is steep enough        slope    vs the binding threshold
    #   there was something to cut off   depth    vs max_depth_db
    #
    # Confidence is the WEAKEST of the two, because a conclusion cannot be
    # better supported than its shakiest premise. Multiplying them instead
    # would punish two independently strong margins for no reason.
    #
    # The depth term is normalised by half the guard rather than all of it.
    # Measured on real encodes of a full-band acoustic guitar — lame at 64, 96,
    # 128 and 192 kbps and afconvert AAC at 64, 96 and 128 — every wall sits
    # 21 to 24 dB below the midrange, a margin of 36 to 39 dB inside the guard.
    # The unstable readings sit at margins of 0.3 to 8.8 dB. Half the guard
    # separates those two populations and saturates on the first, so real codec
    # material still reports 1.00 while anything a few dB from flipping cannot
    # clear the 0.4 the planner acts on.
    slope_margin = (masked[best] - effective_min_slope) / 200.0
    depth_margin = (max_depth_db - float(depths[best])) / (max_depth_db / 2.0)
    confidence = float(np.clip(min(slope_margin, depth_margin), 0.0, 1.0))
    return limit, confidence
