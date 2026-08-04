"""Match-EQ: does the fit recover a known filter, and does the engine agree?

THE MOST IMPORTANT TEST HERE IS THE ROUND TRIP. A known EQ is applied to a source
by the real engine; match-EQ is then asked to recover it from nothing but the two
spectra. If it comes back with the frequency, gain and Q that were applied, the
whole stage works — and if it does not, no amount of curve-fitting elegance
matters.

The second most important asserts that the analytic filter response the optimiser
fits against agrees with what the engine actually renders. The optimiser
evaluates that response thousands of times, so it has to be arithmetic rather
than a render — but an approximation that has drifted from the engine would
produce a beautiful fit to the wrong target, and nothing inside the fit could
detect it.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.engine.render import DEFAULT_BINARY, Recipe, render_array  # noqa: E402
from backend.measure import gating, spectrum  # noqa: E402
from backend.refine import match_eq  # noqa: E402

SR = 48000.0
RNG = np.random.default_rng(31337)


class Results:
    def __init__(self) -> None:
        self.checks = 0
        self.failures = 0

    def check(self, ok: bool, what: str) -> None:
        self.checks += 1
        if not ok:
            self.failures += 1
            print(f"\n  FAIL  {what}")

    def near(self, got: float, want: float, tol: float, what: str) -> None:
        self.check(got is not None and math.isfinite(got) and abs(got - want) <= tol,
                   f"{what}: got {got:.3f}, want {want:.3f} +/- {tol}")


R = Results()


def pink(seconds: float = 8.0, channels: int = 1, amplitude: float = 0.3) -> np.ndarray:
    n = int(SR * seconds)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    shape = np.ones_like(freqs)
    shape[1:] = 1.0 / np.sqrt(freqs[1:])
    shape[0] = 0.0
    spec = RNG.normal(size=(channels, freqs.size)) + 1j * RNG.normal(size=(channels, freqs.size))
    out = np.fft.irfft(spec * shape, n=n, axis=-1)
    return out / np.max(np.abs(out)) * amplitude


def eq_recipe(*bands) -> Recipe:
    recipe = Recipe(output_sample_rate=SR, output_format="float64")
    recipe.add("parametric_eq", bands=list(bands))
    return recipe


def band_dict(kind: str, freq: float, gain: float, q: float) -> dict:
    return {"filter_type": kind, "center_freq_hz": freq,
            "gain_db": gain, "q_factor": q}


# ---------------------------------------------------------------------------


def test_log_grid_and_smoothing() -> None:
    print("[1] log grid and smoothing")

    # THE DEFAULT RANGE IS A DECISION AND IS ASSERTED AS ONE. Below 30 Hz almost
    # nothing is musical and the measurement is unreliable; above 16 kHz very
    # few listeners hear the difference and the source often has nothing there.
    # Every other check here passes the range in explicitly, so the constants
    # themselves were free to move.
    default_grid = match_eq.log_grid()
    R.near(float(default_grid[0]), 30.0, 1e-6, "the fit starts at 30 Hz by default")
    R.near(float(default_grid[-1]), 16000.0, 1e-3, "and ends at 16 kHz")

    grid = match_eq.log_grid(30.0, 16000.0, 256)
    R.check(grid.size == 256, "the grid has the requested resolution")
    R.near(grid[0], 30.0, 1e-6, "starting where asked")
    R.near(grid[-1], 16000.0, 1e-3, "and ending where asked")

    ratios = grid[1:] / grid[:-1]
    R.check(float(np.std(ratios)) < 1e-9,
            "spacing is uniform in the log domain, not the linear one")

    # THE BIAS THIS GRID EXISTS TO AVOID. Fractional-octave smoothing on
    # linearly-spaced bins covers more bins above the centre than below, so it
    # tilts a flat spectrum upward — increasingly toward the top end.
    flat = np.zeros(grid.size)
    R.check(float(np.max(np.abs(match_eq.smooth_log(flat, grid)))) < 1e-9,
            "smoothing a flat curve on the log grid leaves it flat")

    # And the same smoothing applied naively to linear bins does not.
    linear_freqs = np.linspace(30.0, 16000.0, 256)
    window = np.hanning(21) / np.hanning(21).sum()
    tilted = np.convolve(np.pad(np.zeros(256), 10, mode="edge"), window, mode="valid")
    R.check(tilted.size == 256, "the comparison is the same length")

    # A step must survive smoothing as a smooth transition rather than being
    # erased: the fit needs the shape, only without the wrinkles.
    step = np.where(grid > 1000.0, 6.0, 0.0)
    smoothed = match_eq.smooth_log(step, grid, 6.0)
    R.near(float(smoothed[0]), 0.0, 0.1, "a step keeps its low side")
    R.near(float(smoothed[-1]), 6.0, 0.1, "and its high side")

    # The corner has WIDTH rather than being a single jump. Deliberately not
    # asserted as a maximum slope or a point count: at 1/6 octave on a 256-point
    # grid the window is five points, so a normalised Hann convolved with a step
    # gives exactly two intermediate values. Both of those numbers are properties
    # of the grid resolution rather than of the smoothing being correct, and
    # pinning either would break the moment the grid changed size.
    intermediate = np.sum((smoothed > 0.2) & (smoothed < 5.8))
    R.check(intermediate >= 2,
            f"the corner is a transition, not a jump, {intermediate} intermediate")

    # And the smoothing parameter must actually control that width, or it is a
    # knob that does nothing.
    wide = match_eq.smooth_log(step, grid, 1.0)
    narrow = match_eq.smooth_log(step, grid, 12.0)
    wide_span = np.sum((wide > 0.2) & (wide < 5.8))
    narrow_span = np.sum((narrow > 0.2) & (narrow < 5.8))
    R.check(wide_span > narrow_span,
            f"wider smoothing gives a wider transition, {wide_span} vs {narrow_span}")
    R.check(float(np.max(np.abs(np.diff(wide)))) < float(np.max(np.abs(np.diff(narrow)))),
            "and a gentler corner")

    # AND THE FRACTION MEANS OCTAVES. Every check above is relative — wider than
    # narrower, gentler than steeper — and a window half the width it should be
    # satisfies all of them. The absolute answer is that a 1/1-octave smooth of
    # a step spreads the corner over about two thirds of an octave and a
    # 1/3-octave smooth over about a fifth of one.
    def transition_octaves(curve: np.ndarray) -> float:
        inside = np.where((curve > 0.15) & (curve < 5.85))[0]
        return math.log2(grid[inside[-1]] / grid[inside[0]]) if inside.size else 0.0

    for fraction, low, high in ((1.0, 0.45, 1.0), (3.0, 0.10, 0.30)):
        span = transition_octaves(match_eq.smooth_log(step, grid, fraction))
        R.check(low <= span <= high,
                f"1/{fraction:g}-octave smoothing spreads a step over {span:.3f} "
                f"octaves, wanted {low} to {high}")

    # THE WINDOW TAPERS. A boxcar of the same width satisfies every check above
    # — it smooths, the width tracks the fraction, a flat curve stays flat — and
    # it weights a bin an octave away exactly as heavily as the bin next door,
    # which is not smoothing so much as blurring. Under a tapered window a
    # single spike's influence falls off with distance from where it was.
    spike = np.zeros(grid.size)
    spike[grid.size // 2] = 1.0
    spread = match_eq.smooth_log(spike, grid, 6.0)
    centre = int(np.argmax(spread))
    falls_off = all(spread[centre + i] > spread[centre + i + 1] - 1e-12
                    and spread[centre - i] > spread[centre - i - 1] - 1e-12
                    for i in range(1, 3))
    R.check(falls_off and float(spread[centre]) > float(spread[centre + 1]),
            f"a spike's influence falls off with distance, got "
            f"{np.round(spread[centre - 2:centre + 3], 4)}")

    # AND THE KNOB CANNOT SILENTLY BECOME A NO-OP. A Hann window is zero at both
    # endpoints, so an N-point Hann is an (N-2)-point window and `np.hanning(3)`
    # is [0, 1, 0] — the identity. The width floor was 3, so asking for 1/12
    # octave or finer on this grid returned the curve untouched and the "narrow"
    # comparison above was passing against a smoother that did nothing.
    for fraction in (12.0, 24.0):
        fine = match_eq.smooth_log(step, grid, fraction)
        R.check(float(np.max(np.abs(fine - step))) > 0.1,
                f"1/{fraction:g}-octave smoothing still smooths something")


def test_soft_clamp() -> None:
    print("[2] soft clamp")

    # BELOW THE KNEE THE CLAMP MUST BE EXACT, not merely close. A plain
    # `limit * tanh(x / limit)` is smooth everywhere and attenuates everywhere:
    # recovering a known 5 dB EQ against a 6 dB limit it returned 4.09 dB, which
    # is 6*tanh(5/6) exactly — an 18% error introduced by the safety mechanism
    # on a correction that was never near the limit.
    small = np.array([-3.5, -2.0, -1.0, 0.0, 1.0, 2.0, 3.5])
    clamped = match_eq.soft_clamp(small, 6.0)
    R.check(float(np.max(np.abs(clamped - small))) < 1e-9,
            f"corrections below the knee are untouched, got {clamped}")

    # And a 5 dB correction, which is above the knee but well inside the limit,
    # must survive nearly intact rather than losing a fifth of itself.
    R.near(float(match_eq.soft_clamp(np.array([5.0]), 6.0)[0]), 5.0, 0.25,
           "a 5 dB correction against a 6 dB limit stays close to 5")

    # THE KNEE POINT IS PINNED FROM BOTH SIDES. Everything above is satisfied by
    # any knee from 0.6 upward, including a knee of 1.0, which is a hard clamp
    # wearing the same signature. What fixes it is the value AT the limit: with
    # the knee at 0.6 of the limit a 6 dB correction comes back as 5.43, at 0.9
    # as 5.86 (barely bent) and at 0.3 as 5.00 (half a dB thrown away on a
    # correction inside the limit).
    R.check(abs(float(match_eq.soft_clamp(np.array([3.55]), 6.0)[0]) - 3.55) < 1e-9,
            "3.55 dB is still under the knee and exact")
    at_limit = float(match_eq.soft_clamp(np.array([6.0]), 6.0)[0])
    R.check(5.2 <= at_limit <= 5.6,
            f"a correction at the limit comes back bent but not gutted, got "
            f"{at_limit:.3f}")

    # Large ones are bounded.
    large = np.array([-40.0, -20.0, 20.0, 40.0])
    bounded = match_eq.soft_clamp(large, 6.0)
    R.check(float(np.max(np.abs(bounded))) <= 6.0,
            f"large corrections are bounded by the limit, got {bounded}")

    # SOFT, not hard, and this is the part that matters. A hard clamp creates a
    # corner, and a corner needs a very high Q to fit — so hard-clamping produces
    # exactly the surgical filters this stage exists to avoid.
    ramp = np.linspace(-20.0, 20.0, 400)
    soft = match_eq.soft_clamp(ramp, 6.0)
    second = np.diff(soft, 2)
    R.check(float(np.max(np.abs(second))) < 0.05,
            "the clamped curve has no corner for a high-Q band to chase")

    R.check(float(np.max(np.abs(match_eq.soft_clamp(large, 0.0)))) == 0.0,
            "a zero limit removes the correction entirely")


def test_response_matches_the_engine() -> None:
    print("[3] the analytic response agrees with the engine")

    if not os.path.exists(DEFAULT_BINARY):
        print("  engine binary missing; skipping")
        return

    # THE ASSUMPTION THE WHOLE FIT RESTS ON. The optimiser evaluates the filter
    # response thousands of times, so it must be arithmetic rather than a render.
    # But an analytic response that had drifted from the engine would produce a
    # perfect fit to the wrong target, and nothing inside the fit could tell.
    source = pink(6.0)
    grid = match_eq.log_grid(50.0, 15000.0, 96)

    cases = [
        ("bell", 1000.0, 6.0, 1.4),
        ("bell", 250.0, -5.0, 0.9),
        ("bell", 6000.0, 4.0, 2.5),
        ("lowshelf", 150.0, 5.0, 0.707),
        ("highshelf", 5000.0, -4.0, 0.707),
    ]

    for kind, freq, gain, q in cases:
        rendered = render_array(source, SR, eq_recipe(band_dict(kind, freq, gain, q)))

        before = match_eq._psd_on_grid(source, SR, grid)
        after = match_eq._psd_on_grid(rendered, SR, grid)
        measured = after - before

        analytic = match_eq.band_response_db(
            match_eq.Band(kind, freq, gain, q), grid, SR)

        # Compared over the band where the measurement is trustworthy. The very
        # bottom of the grid has few FFT bins per log step and the estimate is
        # noisy there for reasons that have nothing to do with the filter.
        usable = (grid >= 80.0) & (grid <= 14000.0)
        worst = float(np.max(np.abs((measured - analytic)[usable])))
        R.check(worst < 0.6,
                f"{kind} at {freq:g} Hz: analytic matches the render within "
                f"{worst:.3f} dB")

    try:
        match_eq.band_response_db(
            match_eq.Band("highpass", 80.0, 0.0, 0.707), grid, SR)
        R.check(False, "should refuse a band shape it cannot fit")
    except ValueError:
        R.check(True, "refuses a band shape it cannot fit")


def test_recovers_a_known_eq() -> None:
    print("[4] recovering an EQ that was actually applied")

    if not os.path.exists(DEFAULT_BINARY):
        print("  engine binary missing; skipping")
        return

    # THE TEST THAT MATTERS. A known EQ is applied by the real engine, and
    # match-EQ is asked to recover it from the two spectra alone.
    source = pink(10.0)

    # THE TOLERANCES HERE ARE ASSERTIONS ABOUT THE FIT, not about what would be
    # acceptable. They were 0.75 octaves and "40% of the gain" against a fit
    # that lands on 800.0 Hz exactly and returns 4.8 dB for 5.0 — a margin of
    # 150x and 5x. Inside that slack the smoothing could run at 1/1 octave, the
    # shelf/bell boundary could move by a factor of four, and the cascade could
    # come back with five bands instead of one, and all of it passed.
    #
    # The 120 Hz and Q 3.0 cases are not decoration. A bell low enough to be
    # mistaken for a shelf and a bell narrow enough to be flattened by
    # over-smoothing are the two places where a plausible wrong answer looks
    # like a right one everywhere else.
    for kind, freq, gain, q, bands_wanted in (
            ("bell", 800.0, 5.0, 1.2, 1),
            ("bell", 3000.0, -6.0, 1.5, 1),
            ("bell", 120.0, 5.0, 1.2, 1),
            ("bell", 1500.0, 6.0, 3.0, 1),
            ("highshelf", 5000.0, 4.0, 0.707, 1)):
        reference = render_array(source, SR, eq_recipe(band_dict(kind, freq, gain, q)))
        result = match_eq.match(source, reference, SR, max_bands=5)

        R.check(bool(result.bands), f"{kind} at {freq:g} Hz: something was fitted")
        R.check(result.rounded_error_db < result.initial_error_db * 0.15,
                f"{kind} at {freq:g} Hz: error fell from "
                f"{result.initial_error_db:.2f} to {result.rounded_error_db:.2f} dB")

        # A BAND HAS TO EARN ITS PLACE. One filter was applied, so one band must
        # come back — not five, and not the same move split across two.
        R.check(len(result.bands) == bands_wanted,
                f"{kind} at {freq:g} Hz: {len(result.bands)} band(s) returned, "
                f"wanted {bands_wanted}: "
                f"{[(b.filter_type, b.center_freq_hz, b.gain_db) for b in result.bands]}")

        strongest = max(result.bands, key=lambda b: abs(b.gain_db))
        R.check(strongest.filter_type == kind,
                f"{kind} at {freq:g} Hz: recovered as a {strongest.filter_type}")
        octaves = abs(math.log2(strongest.center_freq_hz / freq))
        R.check(octaves < 0.1,
                f"{kind} at {freq:g} Hz: recovered at "
                f"{strongest.center_freq_hz:g} Hz, {octaves:.3f} octaves away")
        R.near(strongest.gain_db, gain, 0.6,
               f"{kind} at {freq:g} Hz: with the applied gain")
        if kind == "bell":
            R.near(strongest.q_factor, q, 0.4,
                   f"{kind} at {freq:g} Hz: and the applied Q")

    # AND A TWO-BAND EQ COMES BACK AS TWO BANDS. This is the property
    # `choose_cascade` exists for — fitting to the band budget rather than to
    # the material returned four for this case, the 420 Hz bell split across
    # 335 and 470 Hz — and nothing asserted it.
    two = render_array(source, SR, eq_recipe(
        band_dict("bell", 400.0, -5.0, 1.1),
        band_dict("highshelf", 6000.0, 4.0, 0.707)))
    pair = match_eq.match(source, two, SR, max_bands=5)
    R.check(len(pair.bands) == 2,
            f"a two-band EQ against a five-band budget returns two bands, got "
            f"{[(b.filter_type, b.center_freq_hz, b.gain_db) for b in pair.bands]}")
    R.check(sorted(b.filter_type for b in pair.bands) == ["bell", "highshelf"],
            f"one of each shape, got {[b.filter_type for b in pair.bands]}")

    # THROUGH `match_to_curve` TOO, and not as a formality. That is the entry
    # point the API uses for a stored reference curve — it measures the
    # reference once and discards the audio — so it has its own copy of every
    # default, and `match()` overrides them all on the way past. Its smoothing
    # default was free to move to 1/1 octave without a single check noticing.
    narrow = render_array(source, SR, eq_recipe(
        band_dict("bell", 1500.0, 6.0, 3.0)))
    stored = match_eq.match_to_curve(
        source, match_eq.reference_curve(narrow, SR), SR, max_bands=5)
    R.check(len(stored.bands) == 1,
            f"a stored reference curve recovers the same one band, got "
            f"{[(b.filter_type, b.center_freq_hz, b.gain_db) for b in stored.bands]}")
    if stored.bands:
        R.near(stored.bands[0].gain_db, 6.0, 0.6, "at the applied gain")
        R.near(stored.bands[0].q_factor, 3.0, 0.4, "and the applied Q")

    # A SMALL CORRECTION IS STILL A CORRECTION. Greedy placement gives up when
    # the largest remaining error is under 0.35 dB, and every case above is a
    # 5 dB move — six to seventeen times past that bar — so the threshold could
    # have been anywhere below 5 dB. At 2 dB this move produces no bands at all.
    gentle = render_array(source, SR, eq_recipe(
        band_dict("bell", 1000.0, 1.2, 1.0)))
    quiet = match_eq.match(source, gentle, SR, max_bands=3)
    R.check(len(quiet.bands) == 1,
            f"a 1.2 dB bell is still worth a band, got "
            f"{[(b.center_freq_hz, b.gain_db) for b in quiet.bands]}")
    if quiet.bands:
        R.near(quiet.bands[0].gain_db, 1.2, 0.4, "at about the applied gain")


def test_level_matching() -> None:
    print("[5] level matching happens first")

    if not os.path.exists(DEFAULT_BINARY):
        print("  engine binary missing; skipping")
        return

    # THE STEP MOST IMPLEMENTATIONS SKIP. Without it a match EQ is a broadband
    # gain change wearing a filter costume: the fit spends its bands correcting
    # an offset and the result is louder rather than closer.
    source = pink(8.0)
    louder = render_array(source, SR,
                          Recipe(output_sample_rate=SR, output_format="float64")
                          .add("gain", gain_db=6.0))

    result = match_eq.match(source, louder, SR)

    R.near(result.level_offset_db, 6.0, 0.4,
           "a pure 6 dB level difference is recognised as level")
    R.check(not result.bands or max(abs(b.gain_db) for b in result.bands) < 1.0,
            f"and produces no meaningful EQ, got "
            f"{[(b.center_freq_hz, b.gain_db) for b in result.bands]}")

    # The offset is reported separately and NOT folded into the bands. A level
    # change is a gain node's job, and hiding one inside an EQ makes the EQ look
    # like it did something it did not.
    R.check(result.level_offset_db != 0.0, "the offset is reported to the caller")

    # LEVEL-MATCH FIRST MEANS THE REPORTED ERROR IS AFTER IT. A pure 6 dB level
    # difference is not 6 dB of spectral error, and if the initial error were
    # measured before the offset came out, "we reduced the error from 6 to 0.1"
    # would be a claim about arithmetic the gain node does for free.
    R.check(result.initial_error_db < 0.05,
            f"a pure level difference leaves almost no spectral error to fit, "
            f"got {result.initial_error_db:.3f} dB")

    # THE OFFSET IS THE MEDIAN, NOT THE MEAN, so one narrow resonance in either
    # recording cannot drag the whole curve. Measured against a synthetic
    # reference curve, because a resonance narrow enough to separate the two
    # is easier to state than to render.
    grid = match_eq.log_grid()
    source_curve = match_eq._psd_on_grid(source, SR, grid)
    resonant = source_curve.copy()
    resonant[(grid > 900.0) & (grid < 1300.0)] += 24.0
    with_peak = match_eq.match_to_curve(source, resonant, SR, max_bands=3)
    R.near(with_peak.level_offset_db, 0.0, 0.3,
           "a 24 dB resonance over a fifth of an octave does not become a level "
           "change")

    # A level difference on top of a real EQ difference must not confuse it.
    both = render_array(source, SR,
                        eq_recipe(band_dict("bell", 1200.0, 6.0, 1.3)))
    both = render_array(both, SR,
                        Recipe(output_sample_rate=SR, output_format="float64")
                        .add("gain", gain_db=-8.0))
    mixed = match_eq.match(source, both, SR, max_bands=3)
    R.near(mixed.level_offset_db, -8.0, 1.0,
           "level and EQ differences are separated")
    strongest = max(mixed.bands, key=lambda b: abs(b.gain_db))
    R.check(abs(math.log2(strongest.center_freq_hz / 1200.0)) < 0.9,
            f"and the EQ move is still found at {strongest.center_freq_hz:g} Hz")


def test_stays_engineer_plausible() -> None:
    print("[6] the output is something an engineer would sign off")

    grid = match_eq.log_grid(points=192)

    # A deliberately nasty target: narrow spikes that a least-squares fit would
    # love to chase with Q 8.7 notches. Unconstrained, that is exactly what it
    # returns — numerically better and not something anybody would write.
    target = np.zeros(grid.size)
    for centre, gain in ((220.0, 9.0), (1900.0, -11.0), (7000.0, 7.0)):
        target += gain * np.exp(-((np.log2(grid / centre)) ** 2) / (2 * 0.04 ** 2))

    bands = match_eq.fit_bands(match_eq.soft_clamp(target, 6.0), grid, SR,
                               max_bands=5, max_q=4.0)

    R.check(bands, "something was fitted")
    R.check(all(b.q_factor <= 4.01 for b in bands),
            f"no band is surgical, Qs are {[round(b.q_factor, 2) for b in bands]}")
    R.check(all(abs(b.gain_db) <= 6.01 for b in bands),
            f"and none exceeds the clamp, gains are "
            f"{[round(b.gain_db, 1) for b in bands]}")
    R.check(len(bands) <= 5, f"and the cascade stays small, {len(bands)} bands")

    # THE SAME TARGET WITH NOTHING PASSED IN. Every assertion above hands
    # `fit_bands` the very constraints it is being tested for, so the defaults —
    # which are what `match()` actually uses — were never exercised: `max_q`
    # could have been 8 and this whole section would still pass.
    defaulted = match_eq.fit_bands(match_eq.soft_clamp(target, 6.0), grid, SR,
                                   max_bands=5)
    R.check(all(b.q_factor <= 4.01 for b in defaulted),
            f"the default max_q is the engineer-plausible one, Qs are "
            f"{[round(b.q_factor, 2) for b in defaulted]}")

    # And the gain bound holds against a target that has NOT been clamped. The
    # bound is never active when the target is soft-clamped to the same limit
    # first, which is how a bound of twice the limit went unnoticed.
    unclamped = match_eq.fit_bands(target, grid, SR, max_bands=5, max_gain_db=6.0)
    R.check(all(abs(b.gain_db) <= 6.01 for b in unclamped),
            f"an 11 dB target fitted against a 6 dB bound stays inside it, gains "
            f"are {[round(b.gain_db, 1) for b in unclamped]}")

    # Rounded to numbers a person would type. An engineer reading "-3.2 dB at
    # 340 Hz" trusts it; nine decimal places reads as machine output.
    #
    # ASSERTED AGAINST KNOWN INPUTS, because "is it already round?" is satisfied
    # by every coarser rounding as well: a gain rounded to whole decibels
    # equals `round(gain, 1)`, and a frequency rounded to 100 Hz is divisible by
    # 10. Rounding to 100 Hz, whole dB and whole Q all passed those checks.
    known = [match_eq.Band("bell", 7043.7, -3.246, 1.914),
             match_eq.Band("bell", 337.2, 2.349, 0.6666),
             match_eq.Band("lowshelf", 64.4, 5.57, 0.707)]
    got = match_eq.round_bands(known)
    R.check([b.center_freq_hz for b in got] == [7040.0, 335.0, 64.0],
            f"frequencies round to 10 Hz above 1 kHz, 5 Hz above 100 Hz and 1 Hz "
            f"below, got {[b.center_freq_hz for b in got]}")
    R.check([b.gain_db for b in got] == [-3.2, 2.3, 5.6],
            f"gains round to a tenth of a dB, got {[b.gain_db for b in got]}")
    R.check([b.q_factor for b in got] == [1.91, 0.67, 0.71],
            f"and Q to a hundredth, got {[b.q_factor for b in got]}")
    R.check([b.filter_type for b in got] == ["bell", "bell", "lowshelf"],
            "with the shapes carried through unchanged")

    rounded = match_eq.round_bands(bands)
    for band in rounded:
        R.check(band.gain_db == round(band.gain_db, 1),
                f"gain {band.gain_db} is readable")
        if band.center_freq_hz >= 1000.0:
            R.check(band.center_freq_hz % 10 == 0,
                    f"frequency {band.center_freq_hz} is round above 1 kHz")

    # And the fitted bands must be usable directly as engine parameters.
    from backend.schemas.graph import EqBand

    for band in rounded:
        validated = EqBand.model_validate(band.as_dict())
        R.check(validated.center_freq_hz == band.center_freq_hz,
                "a fitted band validates against the graph schema unchanged")


def test_degenerate_inputs() -> None:
    print("[7] degenerate inputs")

    silence = np.zeros((1, int(SR * 2)))
    quiet = match_eq.match(silence, silence, SR)
    R.check(quiet.bands == [], "matching silence to silence produces no bands")

    source = pink(4.0)
    identical = match_eq.match(source, source, SR)
    R.near(identical.level_offset_db, 0.0, 0.05,
           "matching a signal to itself finds no level offset")
    # NO BANDS AT ALL, not merely small ones. A band that cannot be heard is
    # worse than no band: it consumes one of the five, it appears in the
    # explanation shown to the user, and the user hears no change.
    R.check(identical.bands == [],
            f"and no bands whatsoever, got "
            f"{[(round(b.center_freq_hz), b.gain_db) for b in identical.bands]}")

    # A difference far beyond anything that is really EQ must be reported as
    # clamped rather than followed. The difference between two recordings
    # includes performances, rooms and noise floors, and chasing all of it into a
    # 20 dB correction produces something that matches on paper and is unusable.
    extreme = np.zeros(match_eq.log_grid().size) + 25.0
    R.check(float(np.max(match_eq.soft_clamp(extreme, 6.0))) <= 6.0,
            "a 25 dB difference is bounded")

    from scipy import signal as _signal

    # THE SPECTRUM IS INTERPOLATED IN POWER, NOT IN dB. Averaging logarithms
    # biases toward the quiet bins, and by Jensen's inequality the dB-domain
    # answer can only ever read LOW. On pink noise the gap is 0.009 dB RMS and
    # invisible; on a spectrum with a sharp feature it is 0.2 dB, which is more
    # than the whole fit's residual error.
    notched = _signal.sosfilt(
        _signal.butter(2, [960.0, 1040.0], btype="bandstop", fs=SR, output="sos"),
        source)
    grid = match_eq.log_grid()
    mask, _ = gating.gated_frame_mask(notched, SR)
    freqs, power = spectrum.gated_ltas(notched, SR, mask)
    power_db = 10.0 * np.log10(np.maximum(power, 1e-300))
    in_db = np.interp(grid, freqs, power_db, left=power_db[0], right=power_db[-1])
    in_power = match_eq._psd_on_grid(notched, SR, grid)
    R.check(bool(np.all(in_power >= in_db - 1e-9)),
            "a power-domain interpolation never reads below a dB-domain one")
    R.check(float(np.max(in_power - in_db)) > 0.05,
            f"and reads measurably above it where the spectrum has structure, "
            f"gap {float(np.max(in_power - in_db)):.3f} dB")

    lopped = _signal.sosfilt(
        _signal.butter(8, 2000.0, btype="low", fs=SR, output="sos"), source)
    heavy = match_eq.match(source, lopped, SR, max_bands=4)
    R.check(heavy.clamped, "a drastic spectral difference is reported as clamped")
    R.check(all(abs(b.gain_db) <= 6.01 for b in heavy.bands),
            "and no band exceeds the clamp")

    # THE CLAMPED FLAG IS PROBED EITHER SIDE OF THE LIMIT. An eighth-order
    # lowpass is 40 dB past a 6 dB clamp, and against a margin that wide the
    # flag could have been raised at any threshold up to seven times the limit.
    # A tilt is the shape where the answer is arguable and therefore worth
    # asserting: eight decibels of it is past the clamp, four is not.
    base = match_eq._psd_on_grid(source, SR, grid)
    for depth, wanted in ((8.0, True), (4.0, False)):
        tilt = base + np.linspace(-depth, depth, grid.size)
        got = match_eq.match_to_curve(source, tilt, SR, max_bands=4).clamped
        R.check(got == wanted,
                f"a +/-{depth:g} dB tilt against a 6 dB clamp reports "
                f"clamped={got}, wanted {wanted}")


def test_end_to_end_improves_the_match() -> None:
    print("[8] applying the fit actually reduces the difference")

    if not os.path.exists(DEFAULT_BINARY):
        print("  engine binary missing; skipping")
        return

    # THE CLAIM, MEASURED THROUGH THE ENGINE. Everything above measures the fit
    # against its own target curve, which is circular if the curve is wrong.
    # This applies the fitted bands with the real renderer and re-measures.
    source = pink(10.0)
    reference = render_array(source, SR, eq_recipe(
        band_dict("bell", 400.0, -5.0, 1.1),
        band_dict("highshelf", 6000.0, 4.0, 0.707),
    ))

    result = match_eq.match(source, reference, SR, max_bands=4)
    R.check(bool(result.bands), "a fit was produced")

    corrected = render_array(source, SR, eq_recipe(*[b.as_dict() for b in result.bands]))

    grid = match_eq.log_grid(60.0, 14000.0, 128)
    reference_db = match_eq._psd_on_grid(reference, SR, grid)

    def error_against_reference(signal_in: np.ndarray) -> float:
        curve = match_eq._psd_on_grid(signal_in, SR, grid)
        difference = curve - reference_db
        return float(np.sqrt(np.mean((difference - np.median(difference)) ** 2)))

    before = error_against_reference(source)
    after = error_against_reference(corrected)

    R.check(after < before * 0.5,
            f"applying the fit halves the spectral difference: "
            f"{before:.2f} -> {after:.2f} dB RMS")

    # And the reported error must agree with what the engine produced, or the
    # numbers shown to a user describe a render that did not happen.
    R.check(abs(after - result.rounded_error_db) < 0.8,
            f"the reported error {result.rounded_error_db:.2f} matches the "
            f"rendered {after:.2f} dB")

    # Rounding is applied BEFORE the reported error, never after: rounding
    # changes the response, so reporting the pre-rounding number would describe
    # something that does not ship.
    R.check(result.rounded_error_db >= result.fitted_error_db - 1e-9,
            "the reported error is the rounded one, not the pre-rounding one")

    # And the two are DIFFERENT NUMBERS. An inequality alone is satisfied by
    # reporting the pre-rounding error as both, which is exactly the mistake
    # this is here to prevent: rounding three parameters per band moves the
    # response, so if the two agree to the last bit the second was never
    # measured. The gap is small — 0.0026 dB here — which is why the assertion
    # is on the fact of it rather than on its size.
    R.check(bool(result.bands) and result.rounded_error_db != result.fitted_error_db,
            f"and it was measured again after rounding: fitted "
            f"{result.fitted_error_db:.6f}, rounded {result.rounded_error_db:.6f}")


if __name__ == "__main__":
    print("Audionova refine: match-EQ\n")
    test_log_grid_and_smoothing()
    test_soft_clamp()
    test_response_matches_the_engine()
    test_recovers_a_known_eq()
    test_level_matching()
    test_stays_engineer_plausible()
    test_degenerate_inputs()
    test_end_to_end_improves_the_match()
    print(f"\n{R.checks} checks, {R.failures} failure(s)")
    sys.exit(1 if R.failures else 0)
