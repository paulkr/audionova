"""Match-EQ: fit an editable filter cascade to a spectral difference.

WHY THIS STAGE EXISTS. A language model chooses modules and order well and picks
parameter values badly — 0.32 MAE against a regression model's 0.20. But the
regression model that wins on parameter error has nearly the worst perceptual
score, because it produces numerically-close chains that no engineer would write.
Refine takes the model's structural decision and fixes only the numbers, which
gets the parameter accuracy without the perceptual cost.

THE OUTPUT IS A HANDFUL OF BIQUADS, NOT AN 8000-TAP FIR. A paying engineer has to
be able to read the result, trust it, disagree with it, and retype it. "-3.2 dB
at 340 Hz, Q 1.1" is all four of those things; an impulse response is none of
them. The fit is deliberately constrained to something a person would have
written by hand.

FOUR STEPS, AND THREE OF THEM ARE THE ONES IMPLEMENTATIONS SKIP.

  1. LEVEL-MATCH FIRST. Without it a match EQ is a broadband gain change wearing
     a filter costume: every band reads high or low by the same amount, the fit
     spends its bands correcting an offset, and the result is louder rather than
     closer. This is the step most implementations omit.

  2. RESAMPLE ONTO A UNIFORM LOG-FREQUENCY GRID BEFORE SMOOTHING. Fractional-
     octave smoothing applied to linearly-spaced bins is asymmetric in log
     frequency — the window covers more bins above the centre than below — which
     biases the smoothed curve upward, increasingly so toward the top end. Tylka,
     Boren and Choueiri (JAES 2017) measured it. On a linear grid a 1/3-octave
     smooth of a flat spectrum is not flat.

  3. CLAMP, WITH A SOFT KNEE. Unbounded match curves are the single commonest
     reason match EQ sounds wrong: the difference between two recordings includes
     everything that is not EQ — different performances, different rooms,
     different noise floors — and following it into a 20 dB correction at 40 Hz
     produces something that matches on paper and is unusable.

  4. FIT, THEN ROUND, THEN RE-VERIFY. Rounding to human numbers changes the
     response, so the error is measured again afterwards rather than reported
     from before.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, signal

from backend.measure import gating, spectrum

# Beyond this, the difference between two recordings stops being EQ.
DEFAULT_CLAMP_DB = 6.0

# Fit over the band where a correction is both audible and trustworthy. Below
# 30 Hz almost nothing is musical and the measurement is unreliable; above
# 16 kHz very few listeners hear the difference and the source often has nothing
# there anyway.
FIT_LOW_HZ = 30.0
FIT_HIGH_HZ = 16000.0


@dataclass
class Band:
    """One fitted band, in the same shape the engine's ParametricEQ takes."""

    filter_type: str
    center_freq_hz: float
    gain_db: float
    q_factor: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "filter_type": self.filter_type,
            "center_freq_hz": self.center_freq_hz,
            "gain_db": self.gain_db,
            "q_factor": self.q_factor,
        }


@dataclass
class MatchResult:
    bands: list[Band] = field(default_factory=list)
    # Error before and after, in dB RMS over the fit range. Both reported,
    # because "we reduced the error" is the only claim worth making and it needs
    # the starting point.
    initial_error_db: float = 0.0
    fitted_error_db: float = 0.0
    rounded_error_db: float = 0.0
    level_offset_db: float = 0.0
    clamped: bool = False
    grid_hz: np.ndarray = field(default_factory=lambda: np.zeros(0))
    target_curve_db: np.ndarray = field(default_factory=lambda: np.zeros(0))
    fitted_curve_db: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def improvement_db(self) -> float:
        return self.initial_error_db - self.rounded_error_db


def log_grid(low_hz: float = FIT_LOW_HZ, high_hz: float = FIT_HIGH_HZ,
             points: int = 256) -> np.ndarray:
    """Uniformly spaced on a log axis. Everything downstream assumes this."""
    return np.exp(np.linspace(math.log(low_hz), math.log(high_hz), points))


def _psd_on_grid(samples: np.ndarray, sample_rate: float,
                 grid: np.ndarray) -> np.ndarray:
    """Gated PSD, interpolated onto the log grid, in dB.

    Gated because an ungated average over a stem that is half silence describes
    the room. Interpolated in the POWER domain and converted afterwards, since
    averaging dB values is averaging logarithms and biases toward the quiet bins.
    """
    mask, _ = gating.gated_frame_mask(samples, sample_rate)
    freqs, power = spectrum.gated_ltas(samples, sample_rate, mask)
    if freqs.size == 0:
        return np.full(grid.size, -np.inf)

    interpolated = np.interp(grid, freqs, power, left=power[0], right=power[-1])
    return 10.0 * np.log10(np.maximum(interpolated, 1e-300))


def smooth_log(curve_db: np.ndarray, grid: np.ndarray,
               fraction_of_octave: float = 6.0) -> np.ndarray:
    """Smooth on the log grid, where the window is symmetric.

    THE GRID IS WHY THIS IS CORRECT. Applying the same smoothing to linearly-
    spaced FFT bins covers more bins above the centre frequency than below it,
    so it biases the result upward and does so increasingly toward the top end —
    a 1/3-octave smooth of a flat spectrum comes out tilted. On a uniform log
    grid a fixed-width window is symmetric in the domain that matters.
    """
    if curve_db.size < 3:
        return curve_db.copy()

    # Grid spacing in octaves, so the window width can be stated musically.
    step_octaves = math.log2(grid[1] / grid[0])
    width = max(3, int(round((1.0 / fraction_of_octave) / step_octaves)))
    if width % 2 == 0:
        width += 1

    window = np.hanning(width)
    window /= window.sum()
    padded = np.pad(curve_db, width // 2, mode="edge")
    return np.convolve(padded, window, mode="valid")


def soft_clamp(curve_db: np.ndarray, limit_db: float = DEFAULT_CLAMP_DB,
               knee: float = 0.6) -> np.ndarray:
    """Bound the curve with a soft knee: linear below, compressed above.

    A hard clamp creates a corner in the target, and a corner needs a very high
    Q to fit — so hard-clamping produces exactly the surgical filters this stage
    exists to avoid. The curve has to stay smooth to be fittable by the gentle
    bands an engineer would have written.

    LINEAR BELOW THE KNEE, and this matters more than it looks. A plain
    `limit * tanh(x / limit)` is smooth everywhere and also attenuates
    EVERYWHERE: recovering a known 5 dB EQ against a 6 dB limit, it returned
    4.09 dB, which is 6*tanh(5/6) exactly. An 18% error on a correction that was
    never near the limit, introduced by the safety mechanism rather than by the
    measurement.

    So the knee starts at 60% of the limit. Below that the correction passes
    through untouched; above it the curve bends toward the limit and never
    reaches it. Same reasoning as a compressor's soft knee, and the same shape.
    """
    if limit_db <= 0.0:
        return np.zeros_like(curve_db)

    start = limit_db * knee
    headroom = limit_db - start
    magnitude = np.abs(curve_db)

    compressed = start + headroom * np.tanh(
        np.maximum(magnitude - start, 0.0) / max(headroom, 1e-9))
    bent = np.where(magnitude <= start, magnitude, compressed)
    return np.sign(curve_db) * bent


# ---------------------------------------------------------------------------
# Filter response, in the same parameterisation the engine uses


def band_response_db(band: Band, grid: np.ndarray, sample_rate: float) -> np.ndarray:
    """Magnitude response of one band, in dB, on the grid.

    RBJ cookbook forms, matching the engine's shapes. Evaluated here rather than
    measured by rendering, because the fit needs thousands of evaluations and a
    render is milliseconds — but the result is checked against a real render in
    the tests, which is what makes this approximation safe to optimise against.
    """
    omega = 2.0 * math.pi * band.center_freq_hz / sample_rate
    cos_w, sin_w = math.cos(omega), math.sin(omega)
    q = max(band.q_factor, 1e-3)
    alpha = sin_w / (2.0 * q)
    amplitude = 10.0 ** (band.gain_db / 40.0)

    if band.filter_type == "bell":
        b = [1 + alpha * amplitude, -2 * cos_w, 1 - alpha * amplitude]
        a = [1 + alpha / amplitude, -2 * cos_w, 1 - alpha / amplitude]
    elif band.filter_type in ("lowshelf", "highshelf"):
        shelf = 2.0 * math.sqrt(amplitude) * alpha
        if band.filter_type == "lowshelf":
            b = [amplitude * ((amplitude + 1) - (amplitude - 1) * cos_w + shelf),
                 2 * amplitude * ((amplitude - 1) - (amplitude + 1) * cos_w),
                 amplitude * ((amplitude + 1) - (amplitude - 1) * cos_w - shelf)]
            a = [(amplitude + 1) + (amplitude - 1) * cos_w + shelf,
                 -2 * ((amplitude - 1) + (amplitude + 1) * cos_w),
                 (amplitude + 1) + (amplitude - 1) * cos_w - shelf]
        else:
            b = [amplitude * ((amplitude + 1) + (amplitude - 1) * cos_w + shelf),
                 -2 * amplitude * ((amplitude - 1) + (amplitude + 1) * cos_w),
                 amplitude * ((amplitude + 1) + (amplitude - 1) * cos_w - shelf)]
            a = [(amplitude + 1) - (amplitude - 1) * cos_w + shelf,
                 2 * ((amplitude - 1) - (amplitude + 1) * cos_w),
                 (amplitude + 1) - (amplitude - 1) * cos_w - shelf]
    else:
        raise ValueError(f"match-EQ does not fit {band.filter_type!r} bands")

    worN = 2.0 * np.pi * grid / sample_rate
    _, response = signal.freqz(b, a, worN=worN)
    return 20.0 * np.log10(np.maximum(np.abs(response), 1e-12))


def cascade_response_db(bands: list[Band], grid: np.ndarray,
                        sample_rate: float) -> np.ndarray:
    total = np.zeros(grid.size)
    for band in bands:
        total += band_response_db(band, grid, sample_rate)
    return total


# ---------------------------------------------------------------------------
# Fitting


def _greedy_seed(target_db: np.ndarray, grid: np.ndarray, sample_rate: float,
                 max_bands: int) -> list[Band]:
    """Place bands one at a time at the largest remaining error.

    Greedy rather than random, because the optimiser that follows is local: a
    random start lands in a different minimum on every run, and a match EQ that
    produces a different answer each time it is asked is not something an
    engineer will trust.

    Shelves at the extremes, bells in between. A bell at 40 Hz has to be very
    wide to correct a whole low end and ends up affecting the mids; a shelf is
    what an engineer would have reached for.
    """
    bands: list[Band] = []
    residual = target_db.copy()

    for _ in range(max_bands):
        index = int(np.argmax(np.abs(residual)))
        peak = float(residual[index])
        if abs(peak) < 0.35:
            break  # nothing left worth a band

        freq = float(grid[index])
        if freq <= grid[0] * 2.0:
            band = Band("lowshelf", min(freq * 2.0, 250.0), peak, 0.707)
        elif freq >= grid[-1] / 2.0:
            band = Band("highshelf", max(freq / 2.0, 4000.0), peak, 0.707)
        else:
            band = Band("bell", freq, peak, 1.2)

        bands.append(band)
        residual = target_db - cascade_response_db(bands, grid, sample_rate)

    return bands


def _pack(bands: list[Band]) -> np.ndarray:
    return np.array([v for b in bands
                     for v in (math.log(b.center_freq_hz), b.gain_db,
                               math.log(b.q_factor))])


def _unpack(vector: np.ndarray, bands: list[Band]) -> list[Band]:
    # Frequency and Q are optimised in the LOG domain so the optimiser takes
    # proportional steps. In the linear domain a step that is sensible at 8 kHz
    # is enormous at 80 Hz, and the fit either crawls at the bottom or thrashes
    # at the top.
    out = []
    for i, template in enumerate(bands):
        freq, gain, q = vector[3 * i : 3 * i + 3]
        out.append(Band(template.filter_type, float(math.exp(freq)),
                        float(gain), float(math.exp(q))))
    return out


def fit_bands(
    target_db: np.ndarray,
    grid: np.ndarray,
    sample_rate: float,
    *,
    max_bands: int = 5,
    max_gain_db: float = DEFAULT_CLAMP_DB,
    max_q: float = 4.0,
) -> list[Band]:
    """Least-squares fit of a small cascade to the target curve.

    `max_q` of 4 is the constraint that keeps the output engineer-plausible.
    Unconstrained, a least-squares fit will happily return Q 8.7 notches that
    track every wrinkle of a difference curve which is mostly not EQ. Capping it
    costs a fraction of a dB of fit error and buys a result somebody would sign
    off on.
    """
    seed = _greedy_seed(target_db, grid, sample_rate, max_bands)
    if not seed:
        return []

    low = []
    high = []
    for band in seed:
        # Each band may move within a decade of where greedy put it, which is
        # enough to find the real centre and not enough to swap two bands over
        # and land in a different minimum.
        low += [math.log(max(grid[0], band.center_freq_hz / 10.0)),
                -max_gain_db, math.log(0.3)]
        high += [math.log(min(grid[-1], band.center_freq_hz * 10.0)),
                 max_gain_db, math.log(max_q)]

    def residual(vector: np.ndarray) -> np.ndarray:
        return cascade_response_db(_unpack(vector, seed), grid, sample_rate) - target_db

    # The seed can land outside its own bounds, and least_squares refuses to
    # start if it has. Greedy placement subtracts each band before choosing the
    # next, so wherever a band overshoots, the residual there is LARGER than the
    # original target — a second band can then be seeded with a gain beyond the
    # clamp that produced the target in the first place.
    #
    # Clipping is the right response rather than rejecting: the seed is only a
    # starting point, and the optimiser will move it anyway.
    low_bound, high_bound = np.array(low), np.array(high)
    start = np.clip(_pack(seed), low_bound, high_bound)

    result = optimize.least_squares(
        residual, start, bounds=(low_bound, high_bound),
        method="trf", max_nfev=2000,
    )
    return _unpack(result.x, seed)


def round_bands(bands: list[Band]) -> list[Band]:
    """Round to numbers an engineer would type.

    Applied before the final error measurement, never after: rounding changes
    the response, so reporting the pre-rounding error would be reporting a
    result that is not what ships.
    """
    out = []
    for band in bands:
        freq = band.center_freq_hz
        if freq >= 1000.0:
            freq = round(freq / 10.0) * 10.0
        elif freq >= 100.0:
            freq = round(freq / 5.0) * 5.0
        else:
            freq = round(freq)
        out.append(Band(band.filter_type, float(freq),
                        round(band.gain_db, 1), round(band.q_factor, 2)))
    return out


def choose_cascade(
    target_db: np.ndarray,
    grid: np.ndarray,
    sample_rate: float,
    *,
    max_bands: int,
    max_gain_db: float,
    tolerance_db: float = 0.15,
) -> list[Band]:
    """The SMALLEST cascade that fits about as well as the largest.

    Fitting straight to `max_bands` uses the whole budget whether or not the
    material needs it. Recovering a two-band EQ, it returned four: the 420 Hz
    bell split across 335 Hz and 470 Hz, and the shelf accompanied by a bell
    doing almost nothing. Numerically that is a slightly better fit. As a thing
    to hand an engineer it is worse — they cannot tell which bands matter, and
    the two halves of a split band cannot be adjusted independently without
    breaking the shape.

    So each size from one band upward is fitted, and the smallest whose error is
    within `tolerance_db` of the best is chosen. A band has to earn its place by
    more than a tenth of a dB.
    """
    best_error = float("inf")
    candidates: list[tuple[int, list[Band], float]] = []

    for count in range(1, max_bands + 1):
        bands = fit_bands(target_db, grid, sample_rate,
                          max_bands=count, max_gain_db=max_gain_db)
        if not bands:
            break
        error = _rms(target_db - cascade_response_db(bands, grid, sample_rate))
        candidates.append((count, bands, error))
        best_error = min(best_error, error)

        # Greedy seeding stops on its own once nothing is left worth a band, so
        # asking for more than it placed cannot improve anything.
        if len(bands) < count:
            break

    if not candidates:
        return []

    for _, bands, error in candidates:
        if error <= best_error + tolerance_db:
            return bands
    return candidates[-1][1]


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values ** 2))) if values.size else 0.0


def match(
    source: np.ndarray,
    reference: np.ndarray,
    sample_rate: float,
    *,
    max_bands: int = 5,
    clamp_db: float = DEFAULT_CLAMP_DB,
    smoothing: float = 6.0,
    points: int = 256,
) -> MatchResult:
    """Fit an EQ that moves `source` toward `reference`'s spectral balance.

    Returns bands ready to drop into a `parametric_eq` node. The level offset is
    reported separately and deliberately NOT folded into the bands: a level
    change is a gain node's job, and hiding one inside an EQ makes the EQ look
    like it did something it did not.
    """
    grid = log_grid(points=points)

    source_db = _psd_on_grid(source, sample_rate, grid)
    reference_db = _psd_on_grid(reference, sample_rate, grid)

    # Both curves are -inf wherever a band had no energy, and -inf minus -inf is
    # nan. That is the correct answer — the difference between two silences is
    # undefined — so it is masked out rather than allowed to warn and propagate.
    with np.errstate(invalid="ignore"):
        difference = reference_db - source_db
    finite = np.isfinite(difference)
    if not np.any(finite):
        return MatchResult(grid_hz=grid)
    difference = np.where(finite, difference, 0.0)

    # STEP 1: level-match. Without this the fit spends its bands correcting a
    # broadband offset and the result is louder rather than closer.
    #
    # The offset is the median rather than the mean, so a large narrow
    # difference — a resonance in one of the two recordings — cannot drag the
    # whole curve.
    offset = float(np.median(difference))
    difference = difference - offset

    initial_error = _rms(difference)

    # STEP 2: smooth on the log grid, where the window is symmetric.
    smoothed = smooth_log(difference, grid, smoothing)

    # STEP 3: clamp with a soft knee.
    clamped = soft_clamp(smoothed, clamp_db)
    was_clamped = bool(np.any(np.abs(smoothed) > clamp_db * 1.02))

    # STEP 4: fit, round, then re-measure. The smallest cascade that fits, not
    # the largest allowed — see choose_cascade.
    fitted = choose_cascade(clamped, grid, sample_rate,
                            max_bands=max_bands, max_gain_db=clamp_db)
    fitted_curve = cascade_response_db(fitted, grid, sample_rate)

    rounded = round_bands(fitted)
    rounded_curve = cascade_response_db(rounded, grid, sample_rate)

    return MatchResult(
        bands=rounded,
        initial_error_db=initial_error,
        fitted_error_db=_rms(difference - fitted_curve),
        rounded_error_db=_rms(difference - rounded_curve),
        level_offset_db=offset,
        clamped=was_clamped,
        grid_hz=grid,
        target_curve_db=clamped,
        fitted_curve_db=rounded_curve,
    )
