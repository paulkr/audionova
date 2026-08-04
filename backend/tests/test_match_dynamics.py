"""Dynamics matching: does the loop converge, and on the right thing?

THE FIRST TEST IS THE ONE THAT DECIDED THE DESIGN. Bisection requires a monotonic
response, and the obvious control variable — loudness range, which the plan
specified — is not monotonic in ratio on real material. That is asserted here
against the engine rather than taken on trust, because if it ever became monotonic
the simpler design would be available again, and if the substitute stopped being
monotonic the loop would silently stop converging.

Everything else follows: the loop is driven through an injected renderer, so the
convergence tests can run against a cheap synthetic compressor while the
monotonicity and round-trip tests run against the real engine.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.engine.render import DEFAULT_BINARY, Recipe, render_array  # noqa: E402
from backend.eval import corpus  # noqa: E402
from backend.measure import dynamics, loudness  # noqa: E402
from backend.refine import match_dynamics as md  # noqa: E402

SR = 48000.0
RNG = np.random.default_rng(5150)

# Resolved against the repository root, not the working directory. With a
# relative path these tests skipped silently under `make test`, which runs from
# backend/ — including the loudness-range measurement that decided the whole
# design. A test that quietly does not run is worse than one that fails.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STEMS = os.path.join(REPO_ROOT, "audio", "stems")


def stem_path(name: str) -> str:
    return os.path.join(STEMS, f"{name}.wav")


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


def dynamic_source(seconds: float = 12.0, channels: int = 1) -> np.ndarray:
    """Pink noise with a slow envelope, so there are dynamics to compress."""
    n = int(SR * seconds)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    shape = np.ones_like(freqs)
    shape[1:] = 1.0 / np.sqrt(freqs[1:])
    shape[0] = 0.0
    spec = RNG.normal(size=(channels, freqs.size)) + 1j * RNG.normal(size=(channels, freqs.size))
    noise = np.fft.irfft(spec * shape, n=n, axis=-1)
    noise = noise / np.max(np.abs(noise)) * 0.5

    t = np.arange(n) / SR
    envelope = 0.15 + 0.85 * np.abs(np.sin(2.0 * np.pi * 0.4 * t)) ** 2
    return noise * envelope


def compressor_renderer(source: np.ndarray):
    return md.engine_renderer(source, SR)


def _modulated_source(rate_hz: float, seconds: float = 12.0) -> np.ndarray:
    """Pink noise whose level swings at a known rate."""
    n = int(SR * seconds)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    shape = np.ones_like(freqs)
    shape[1:] = 1.0 / np.sqrt(freqs[1:])
    shape[0] = 0.0
    spec = RNG.normal(size=(1, freqs.size)) + 1j * RNG.normal(size=(1, freqs.size))
    noise = np.fft.irfft(spec * shape, n=n, axis=-1)
    noise = noise / np.max(np.abs(noise)) * 0.5
    t = np.arange(n) / SR
    return noise * (0.15 + 0.85 * np.abs(np.sin(2.0 * np.pi * rate_hz * t)) ** 2)


# ---------------------------------------------------------------------------


def test_control_variable_is_monotonic() -> None:
    print("[1] the control variable is monotonic and LRA is not")

    if not os.path.exists(DEFAULT_BINARY):
        print("  engine binary missing; skipping")
        return

    source = dynamic_source()
    threshold = md.threshold_for_duty_cycle(source, SR, 0.3)
    source_loud = md._loud_level_db(source, SR, 90.0)

    ratios = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0)
    reductions, ranges = [], []

    for ratio in ratios:
        recipe = Recipe(output_sample_rate=SR, output_format="float64")
        recipe.add("compressor", threshold_dbfs=threshold, ratio_to_1=ratio,
                   knee_db=6.0, attack_ms=15.0, release_ms=150.0)
        rendered = render_array(source, SR, recipe)
        reductions.append(source_loud - md._loud_level_db(rendered, SR, 90.0))
        ranges.append(loudness.loudness_range_lu(rendered, SR))

    # GAIN REDUCTION IS MONOTONIC, which is what makes bisection valid. It
    # follows the static curve, so more ratio is always at least as much
    # reduction.
    violations = [(ratios[i], reductions[i], reductions[i + 1])
                  for i in range(len(reductions) - 1)
                  if reductions[i + 1] < reductions[i] - 0.05]
    R.check(not violations,
            f"gain reduction rises monotonically with ratio; violations {violations}")
    R.check(reductions[-1] > reductions[0] + 1.0,
            f"and spans a useful range, {reductions[0]:.2f} to {reductions[-1]:.2f} dB")

    # Loudness range happens to be monotonic on THIS source, which is exactly
    # why the claim that it is not belongs with the real stems rather than here.
    # Synthetic material has now been more flattering than real audio three
    # times running, so a design decision taken from real measurements is
    # asserted against real measurements. See test_lra_is_not_monotonic_on_real.
    R.check(len(ranges) == len(ratios), "loudness range was measured alongside")


def test_lra_is_not_monotonic_on_real() -> None:
    print("[1b] loudness range is not a usable control variable on real audio")

    if not os.path.exists(DEFAULT_BINARY) or not os.path.exists(stem_path("vocals")):
        print("  engine binary or stems missing; skipping")
        return

    # THE MEASUREMENT THAT DECIDED THE DESIGN, and it only shows up on real
    # material. On synthetic pink noise with a clean envelope, loudness range
    # falls monotonically with ratio and bisecting on it would work fine.
    #
    # On the reference stems it does not. Past a certain ratio the gain movement
    # is itself a modulation — pumping is variance — so loudness range turns
    # around and climbs. On already-flat material that is the only thing
    # compression can do to it.
    findings = {}
    for stem in ("vocals", "stereo-guitar"):
        source, rate = corpus.load_excerpt(stem_path(stem), 12.0)
        threshold = md.threshold_for_duty_cycle(source, rate, 0.3)
        ranges = []
        for ratio in (1.0, 2.0, 4.0, 8.0, 20.0):
            recipe = Recipe(output_sample_rate=rate, output_format="float64")
            recipe.add("compressor", threshold_dbfs=threshold, ratio_to_1=ratio,
                       knee_db=6.0, attack_ms=15.0, release_ms=150.0)
            ranges.append(loudness.loudness_range_lu(
                render_array(source, rate, recipe), rate))
        findings[stem] = ranges

        monotonic = all(ranges[i + 1] <= ranges[i] + 0.02
                        for i in range(len(ranges) - 1))
        R.check(not monotonic,
                f"{stem}: loudness range is not monotonic in ratio, "
                f"{[round(v, 2) for v in ranges]}")

    # And gain reduction is, on the same material, which is the substitute.
    for stem in ("vocals", "stereo-guitar"):
        source, rate = corpus.load_excerpt(stem_path(stem), 12.0)
        threshold = md.threshold_for_duty_cycle(source, rate, 0.3)
        loud = md._loud_level_db(source, rate, 90.0)
        reductions = []
        for ratio in (1.0, 2.0, 4.0, 8.0, 20.0):
            recipe = Recipe(output_sample_rate=rate, output_format="float64")
            recipe.add("compressor", threshold_dbfs=threshold, ratio_to_1=ratio,
                       knee_db=6.0, attack_ms=15.0, release_ms=150.0)
            reductions.append(
                loud - md._loud_level_db(render_array(source, rate, recipe), rate, 90.0))
        violations = [i for i in range(len(reductions) - 1)
                      if reductions[i + 1] < reductions[i] - 0.05]
        R.check(not violations,
                f"{stem}: gain reduction IS monotonic, "
                f"{[round(v, 2) for v in reductions]}")


def test_threshold_from_duty_cycle() -> None:
    print("[2] threshold from duty cycle")

    source = dynamic_source()
    rms = dynamics.short_term_rms_db(source, SR)
    finite = rms[np.isfinite(rms)]

    # Derived, not searched: "compress the loudest 30%" is a decision, and once
    # made the threshold follows from the measured percentile with no rendering.
    for duty in (0.1, 0.3, 0.5):
        threshold = md.threshold_for_duty_cycle(source, SR, duty)
        above = float(np.mean(finite > threshold))
        R.near(above, duty, 0.03,
               f"a duty cycle of {duty} puts that fraction above the threshold")

    # A higher duty cycle means a lower threshold, always.
    thresholds = [md.threshold_for_duty_cycle(source, SR, d) for d in (0.1, 0.3, 0.6)]
    R.check(thresholds[0] > thresholds[1] > thresholds[2],
            f"more duty cycle means a lower threshold, got {thresholds}")

    # THE DEFAULT IS 0.3, and every check above passes a duty cycle in, so the
    # default itself was never exercised — it could have been 0.5 and this
    # section would still pass while every caller that omits it compressed
    # a different fraction of the material.
    R.check(md.threshold_for_duty_cycle(source, SR)
            == md.threshold_for_duty_cycle(source, SR, 0.3),
            "the default duty cycle is 0.3")

    # And the silent-source fallback is -18 dBFS, not merely finite. A fallback
    # of -60 is finite too, and it would put a gate-like threshold under
    # material the meter could not read — the failure this branch exists to
    # avoid.
    fallback = md.threshold_for_duty_cycle(np.zeros((1, 1000)), SR)
    R.check(math.isfinite(fallback),
            "silence gets a finite fallback rather than -inf")
    R.check(fallback == -18.0, f"and it is -18 dBFS, got {fallback}")


def test_converges() -> None:
    print("[3] the loop converges")

    if not os.path.exists(DEFAULT_BINARY):
        print("  engine binary missing; skipping")
        return

    source = dynamic_source()
    render = compressor_renderer(source)

    for target in (2.0, 4.0, 6.0):
        result = md.match(source, SR, render, target_reduction_db=target)

        R.check(not result.unachievable,
                f"{target:g} dB of reduction is achievable: {result.notes}")
        R.check(result.error_db <= 0.3,
                f"converged to {result.achieved_reduction_db:.2f} dB against a "
                f"{target:g} dB target")
        R.check(result.renders <= md.DEFAULT_MAX_RENDERS + 1,
                f"within the render budget, used {result.renders}")
        R.check(md.MIN_RATIO <= result.settings.ratio_to_1 <= md.MAX_RATIO,
                f"with a ratio in range, {result.settings.ratio_to_1:.2f}")

    # More reduction must need more ratio. If this ever inverted, the bisection
    # would be converging on the wrong side of the target.
    ratios = [md.match(source, SR, render, target_reduction_db=t).settings.ratio_to_1
              for t in (2.0, 5.0)]
    R.check(ratios[1] > ratios[0],
            f"a larger target needs a larger ratio, got {ratios}")


def test_unachievable_is_reported() -> None:
    print("[4] an unreachable target is reported, not silently maxed out")

    if not os.path.exists(DEFAULT_BINARY):
        print("  engine binary missing; skipping")
        return

    # Already-flat material. "This is already too compressed for what you asked"
    # is the useful answer; quietly returning 20:1 would hide it and the user
    # would hear no change and conclude the tool does not work.
    steady = dynamic_source()
    flat_recipe = Recipe(output_sample_rate=SR, output_format="float64")
    flat_recipe.add("compressor", threshold_dbfs=-40.0, ratio_to_1=20.0,
                    knee_db=2.0, attack_ms=1.0, release_ms=50.0)
    already_flat = render_array(steady, SR, flat_recipe)

    render = compressor_renderer(already_flat)
    result = md.match(already_flat, SR, render, target_reduction_db=15.0)

    R.check(result.unachievable, "an impossible target is flagged")
    R.check(bool(result.notes) and "already too flat" in " ".join(result.notes),
            f"with an explanation, got {result.notes}")
    R.check(result.achievable_reduction_db < 15.0,
            f"and reports what IS achievable, {result.achievable_reduction_db:.2f} dB")

    # WHAT IS ACHIEVABLE IS MEASURED AT THE TOP OF THE RANGE, and "less than 15"
    # is true of every wrong answer as well. The number has to be the reduction
    # this material really gives at 20:1, checked against a render made here —
    # probing at 8:1 instead understates it by 0.7 dB, and a search ceiling of
    # 12:1 by 0.3 dB, and neither changes any verdict above.
    top = Recipe(output_sample_rate=SR, output_format="float64")
    top.add("compressor", **dict(result.settings.as_params(),
                                 ratio_to_1=20.0, makeup_gain_db=0.0))
    at_ceiling = (md._loud_level_db(already_flat, SR, 90.0)
                  - md._loud_level_db(render_array(already_flat, SR, top), SR, 90.0))
    R.near(result.achievable_reduction_db, at_ceiling, 0.2,
           "the achievable figure is the reduction at the top of the ratio range")

    # THE BAR PROBED FROM BOTH SIDES. Asking for 15 dB against 8.8 is 6 dB past
    # the bar, and inside that slack "unachievable" could have been declared
    # anywhere up to 5 dB late.
    just_past = md.match(already_flat, SR, render,
                         target_reduction_db=at_ceiling + 0.7)
    R.check(just_past.unachievable,
            f"a target 0.7 dB past what the material yields is unachievable")
    inside = md.match(already_flat, SR, render,
                      target_reduction_db=at_ceiling - 0.7)
    R.check(not inside.unachievable,
            f"and one 0.7 dB inside it is not: {inside.notes}")

    # It must not spend the whole budget converging on a bound it already knows
    # it cannot pass.
    R.check(result.renders <= 3,
            f"without wasting renders, used {result.renders}")

    # Asking for nothing leaves the compressor at 1:1 rather than finding an
    # excuse to compress.
    zero = md.match(dynamic_source(), SR, compressor_renderer(dynamic_source()),
                    target_reduction_db=0.0)
    R.near(zero.settings.ratio_to_1, 1.0, 1e-9, "a zero target leaves ratio at 1:1")


def test_makeup_is_last() -> None:
    print("[5] makeup gain is computed once, at the end")

    if not os.path.exists(DEFAULT_BINARY):
        print("  engine binary missing; skipping")
        return

    source = dynamic_source()
    result = md.match(source, SR, compressor_renderer(source),
                      target_reduction_db=4.0)

    # Compression without makeup makes things quieter, so the makeup must be
    # positive and must actually restore the level.
    R.check(result.settings.makeup_gain_db > 0.0,
            f"makeup is positive after compression, {result.settings.makeup_gain_db:+.1f} dB")

    final = Recipe(output_sample_rate=SR, output_format="float64")
    final.add("compressor", **result.settings.as_params())
    rendered = render_array(source, SR, final)

    R.near(loudness.integrated_lufs(rendered, SR),
           loudness.integrated_lufs(source, SR), 0.5,
           "and the finished chain lands back at the source's loudness")

    # Inside the loop, makeup would change the level the next iteration measures
    # and the loop would chase its own tail. Verified by confirming the reduction
    # the loop reported still holds for the settings it returned — which it only
    # can if makeup was applied after the search.
    without_makeup = dict(result.settings.as_params(), makeup_gain_db=0.0)
    plain = Recipe(output_sample_rate=SR, output_format="float64")
    plain.add("compressor", **without_makeup)
    #
    # AND THE TOLERANCE IS 0.1, not 0.3. The two numbers are the same
    # measurement of the same settings and agree to 0.001 dB, so 0.3 was three
    # hundred times looser than the quantity it bounds — wide enough for the
    # loop to control on a different percentile entirely and still pass. This is
    # the only thing that pins the loop to the 90th.
    measured = (md._loud_level_db(source, SR, 90.0)
                - md._loud_level_db(render_array(source, SR, plain), SR, 90.0))
    R.near(measured, result.achieved_reduction_db, 0.1,
           "the reported reduction describes the settings that were returned, "
           "measured at the 90th percentile")

    # The knee is a decision too: 6 dB, soft, so the ratio the loop found
    # describes a curve the material actually sits on rather than a corner.
    R.check(result.settings.knee_db == 6.0,
            f"and the knee is 6 dB, got {result.settings.knee_db}")


def test_matching_a_reference() -> None:
    print("[6] matching a reference")

    if not os.path.exists(DEFAULT_BINARY):
        print("  engine binary missing; skipping")
        return

    # A known compressor is applied by the real engine; matching should recover
    # settings that get close to the same dynamics from the same source.
    source = dynamic_source()
    known = Recipe(output_sample_rate=SR, output_format="float64")
    known.add("compressor", threshold_dbfs=md.threshold_for_duty_cycle(source, SR, 0.3),
              ratio_to_1=4.0, knee_db=6.0, attack_ms=15.0, release_ms=150.0)
    reference = render_array(source, SR, known)

    target = md.target_reduction_from_reference(source, reference, SR)
    R.check(target > 0.5,
            f"a compressed reference asks for real reduction, {target:.2f} dB")

    result = md.match(source, SR, compressor_renderer(source), reference=reference)
    R.check(result.error_db <= 0.4,
            f"and the loop hits it: {result.achieved_reduction_db:.2f} against "
            f"{result.target_reduction_db:.2f} dB")

    # A LEVEL DIFFERENCE IS NOT A DYNAMICS DIFFERENCE. A reference that is
    # simply louder must ask for no compression at all, or every quiet source
    # would get squashed on its way to matching a loud one.
    louder = render_array(source, SR,
                          Recipe(output_sample_rate=SR, output_format="float64")
                          .add("gain", gain_db=6.0))
    R.near(md.target_reduction_from_reference(source, louder, SR), 0.0, 0.3,
           "a purely louder reference asks for no reduction")

    # AND NEITHER DOES A MORE DYNAMIC ONE. The quantity is a one-sided
    # difference, not a distance: if the reference has a WIDER spread than the
    # source, the answer is that no compression is called for — an expander
    # might be, but that is a different decision. `abs()` here would compress
    # every source handed a livelier reference, which is the exact opposite of
    # what was asked.
    R.near(md.target_reduction_from_reference(reference, source, SR), 0.0, 1e-9,
           "a reference more dynamic than the source asks for no reduction either")


def test_timing_from_source_rate() -> None:
    print("[7] timing from the source's own modulation rate")

    # THE DERIVATION THE PLAN SPECIFIED DOES NOT WORK, and this records why
    # rather than leaving the replacement looking arbitrary. Comparing the
    # source's modulation spectrum against a compressed reference's was supposed
    # to reveal a 3 dB corner at the compressor's time constant. Measured on a
    # source with envelope energy spread across 0.3 to 25 Hz, compression at 8:1
    # ADDS modulation at those rates rather than removing it: +3.8 dB at 4 Hz for
    # a 15/150 ms compressor. The gain movement is itself a modulation.
    #
    # What is measurable is the source's own rate, which is the more useful
    # quantity anyway: a compressor should move at the speed of the material.
    slow_source = _modulated_source(0.4)
    fast_source = _modulated_source(4.0)

    slow_attack, slow_release, slow_confidence = md.timing_from_source_rate(slow_source, SR)
    fast_attack, fast_release, fast_confidence = md.timing_from_source_rate(fast_source, SR)

    R.check(slow_confidence > 0.4 and fast_confidence > 0.4,
            f"both sources have a measurable modulation rate, "
            f"{slow_confidence:.2f} and {fast_confidence:.2f}")
    R.check(slow_release > fast_release,
            f"a slowly-moving source gets a longer release, "
            f"{slow_release:.0f} vs {fast_release:.0f} ms")
    R.check(slow_attack >= fast_attack,
            f"and a longer attack, {slow_attack:.1f} vs {fast_attack:.1f} ms")

    # Attack stays a fraction of release, which is the conventional asymmetry
    # and what the ear expects.
    for attack, release in ((slow_attack, slow_release), (fast_attack, fast_release)):
        R.check(attack < release,
                f"attack {attack:.1f} ms stays shorter than release {release:.0f} ms")

    # A FRACTION, AND THE FRACTION IS 8%. "Shorter than the release" is true of
    # half the release as well, and half is not an asymmetry — it is a
    # compressor that closes almost as slowly as it opens. Everything else here
    # is a comparison between the two sources, so the ratio was free to be
    # anything below 1.
    for attack, release, name in ((slow_attack, slow_release, "slow"),
                                  (fast_attack, fast_release, "fast")):
        R.check(0.06 <= attack / release <= 0.10,
                f"{name}: attack is {attack / release:.3f} of the release, "
                f"wanted about 0.08")

    # AND THE RELEASE IS AN ABSOLUTE TIME, not merely longer than the other one.
    # It spans half a cycle of the source's own level movement, so a source
    # whose envelope centroid sits at 4.6 Hz gets about 110 ms and one at 11 Hz
    # about 45 ms. Scaling the whole derivation down by a factor of two
    # preserves every ordering asserted above.
    R.check(80.0 <= slow_release <= 150.0,
            f"the slowly-moving source gets about 110 ms, got {slow_release:.1f}")
    R.check(30.0 <= fast_release <= 70.0,
            f"the fast one about 45 ms, got {fast_release:.1f}")

    # The numbers that reach the engine keep their tenth of a millisecond. An
    # attack rounded to whole milliseconds is a 6% error at 8 ms, on the one
    # parameter where a few percent is audible on a transient.
    readable = md.DynamicsSettings(-18.0, 3.0, 8.73, 109.2, 6.0, 3.57).as_params()
    R.check(readable["attack_ms"] == 8.7,
            f"attack keeps a tenth of a millisecond, got {readable['attack_ms']}")
    R.check(readable["release_ms"] == 109.0,
            f"release rounds to the millisecond, got {readable['release_ms']}")
    R.check(readable["ratio_to_1"] == 3.0 and readable["makeup_gain_db"] == 3.6,
            f"and the rest round as documented, got {readable}")

    # Both stay inside what the schema will accept, since these values go
    # straight into a compressor node.
    from backend.schemas.graph import CompressorParams

    for attack, release in ((slow_attack, slow_release), (fast_attack, fast_release)):
        CompressorParams.model_validate(
            {"attack_ms": attack, "release_ms": release})
    R.check(True, "and both validate against the graph schema")

    # Material with no measurable modulation falls back rather than reading a
    # time constant out of noise.
    steady = np.ones((1, int(SR * 4))) * 0.2
    attack, release, confidence = md.timing_from_source_rate(steady, SR)
    R.check(confidence < 0.4 or (attack == 15.0 and release == 150.0),
            f"unmodulated material falls back, got {attack}/{release} "
            f"at confidence {confidence:.2f}")

    # AND THE LOOP ACTUALLY USES WHAT WAS DERIVED. Nothing asserted the
    # connection: `match` derives the timing, then discards it if the
    # confidence is under 0.4, and every source that measures at all reports
    # exactly 0.8 — so raising that gate to 0.9 silently pins every render to
    # the 15/150 ms default and no test noticed.
    if os.path.exists(DEFAULT_BINARY):
        # One source, drawn once: the pink noise is random, so deriving from one
        # draw and matching on another would compare two different signals.
        shared = dynamic_source()
        derived_attack, derived_release, _ = md.timing_from_source_rate(shared, SR)
        with_timing = md.match(shared, SR, compressor_renderer(shared),
                               target_reduction_db=2.0)
        R.near(with_timing.settings.attack_ms, derived_attack, 0.2,
               "the loop uses the attack derived from the source")
        R.near(with_timing.settings.release_ms, derived_release, 2.0,
               "and the release derived from the source")
        R.check(not any("default time constants" in n for n in with_timing.notes),
                f"without falling back, notes were {with_timing.notes}")


def test_on_real_stems() -> None:
    print("[8] on real stems")

    if not os.path.exists(DEFAULT_BINARY) or not os.path.exists(stem_path("vocals")):
        print("  engine binary or stems missing; skipping")
        return

    for stem, target in (("vocals", 4.0), ("bass", 3.0)):
        source, rate = corpus.load_excerpt(stem_path(stem), 12.0)
        render = md.engine_renderer(source, rate)
        result = md.match(source, rate, render, target_reduction_db=target)

        R.check(result.error_db <= 0.4,
                f"{stem}: converged to {result.achieved_reduction_db:.2f} dB "
                f"against {target:g}, ratio {result.settings.ratio_to_1:.2f}")
        R.check(result.renders <= md.DEFAULT_MAX_RENDERS + 1,
                f"{stem}: within the render budget, {result.renders}")

        # The settings must be usable directly, which is the whole point of
        # producing engineer-readable parameters rather than a fitted curve.
        from backend.schemas.graph import CompressorParams

        validated = CompressorParams.model_validate(result.settings.as_params())
        R.check(validated.ratio_to_1 == result.settings.as_params()["ratio_to_1"],
                f"{stem}: the settings validate against the graph schema")


if __name__ == "__main__":
    print("Audionova refine: dynamics matching\n")
    test_control_variable_is_monotonic()
    test_lra_is_not_monotonic_on_real()
    test_threshold_from_duty_cycle()
    test_converges()
    test_unachievable_is_reported()
    test_makeup_is_last()
    test_matching_a_reference()
    test_timing_from_source_rate()
    test_on_real_stems()
    print(f"\n{R.checks} checks, {R.failures} failure(s)")
    sys.exit(1 if R.failures else 0)
