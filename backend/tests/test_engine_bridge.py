"""Cross-validation between the C++ engine and the Python measurement layer.

THIS IS THE STRONGEST TEST IN THE PROJECT, and the reason is structural rather
than thorough. Every other suite checks one implementation against a reference
written by the same hand on the same day. Here two independently written
systems — DSP in C++, measurement in Python, sharing no code at all — have to
agree on a physical quantity whose value is known analytically in advance.

If the engine's gain node and the loudness meter both have a 0.5 dB error, no
test inside either half will find it. This one will, because the arithmetic
between them is fixed: attenuate by 6 dB and the integrated loudness must fall
by 6.00.

It is also the first place the two halves are wired together at all, so it
doubles as the integration test for the render bridge.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.audio import wavio  # noqa: E402
from backend.engine.render import DEFAULT_BINARY, Recipe, RenderError, render, render_array  # noqa: E402
from backend.measure import loudness, spectrum, stereo  # noqa: E402
from backend.measure import report as report_mod  # noqa: E402

SR = 48000.0
RNG = np.random.default_rng(20260728)


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
        ok = got is not None and math.isfinite(got) and abs(got - want) <= tol
        self.check(ok, f"{what}: got {got:.4f}, want {want:.4f} +/- {tol}")


R = Results()


def pink(n: int, channels: int = 1, amplitude: float = 0.3) -> np.ndarray:
    spec = RNG.normal(size=(channels, n // 2 + 1)) + 1j * RNG.normal(size=(channels, n // 2 + 1))
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    shape = np.ones_like(freqs)
    shape[1:] = 1.0 / np.sqrt(freqs[1:])
    shape[0] = 0.0
    out = np.fft.irfft(spec * shape, n=n, axis=-1)
    return out / np.max(np.abs(out)) * amplitude


def tone(n: int, freq: float, dbfs: float, channels: int = 1) -> np.ndarray:
    amp = 10.0 ** (dbfs / 20.0)
    wave = amp * np.sin(2.0 * np.pi * freq * np.arange(n) / SR)
    return np.tile(wave, (channels, 1))


def at_rate(recipe: Recipe) -> Recipe:
    """Keep the delivery rate equal to the source rate."""
    recipe.output_sample_rate = SR
    return recipe


def baseline(source: np.ndarray) -> np.ndarray:
    """The source after an EMPTY render — the only valid reference for a delta.

    COMPARING AGAINST THE RAW SOURCE IS WRONG. Every render passes through
    48 -> 96 -> 48 kHz, and the anti-imaging filter's passband edge sits at
    20 kHz, so the top third-octave band comes back about 1 dB down. On white
    or pink material that is 0.12 dB of integrated loudness, because
    K-weighting emphasises exactly that region.

    Measured against the raw source, a -0.5 dB gain therefore reads as -0.62
    and looks like a calibration error in the engine. Measured against an empty
    render — which has been through the identical conversion — it reads
    -0.5000, and the residual against an exactly scaled copy is -306 dB.

    The rule generalises to the whole evaluation harness: to isolate what a
    NODE did, the reference must have travelled the same path with that node
    absent.
    """
    return render_array(source, SR, at_rate(Recipe()))


# ---------------------------------------------------------------------------


def test_bridge() -> None:
    print("[1] render bridge")

    source = pink(int(SR * 4))

    # AN EMPTY RECIPE IS TRANSPARENT IN THE AUDIBLE BAND, and that is the whole
    # claim — not bit-exactness, which the engine does not offer and should not
    # be asked for.
    #
    # Every render goes 48 -> 96 -> 48 kHz, and the conversion filter's
    # passband edge is 20 kHz. Measured per frequency, the round trip nulls at
    # -194 dB at 1 kHz, -176 dB at 10 kHz and -168 dB at 18 kHz, then falls off
    # a cliff: a 22 kHz tone comes back 12 dB down.
    #
    # Asserting bit-exactness on full-band noise therefore fails at -32 dB and
    # looks like a catastrophic engine bug, when what it has actually found is
    # a documented design decision. Protecting 20-24 kHz would cost two to
    # fourteen times the conversion compute for content nobody can hear.
    band_limited = np.fft.rfft(source, axis=-1)
    band_limited[:, np.fft.rfftfreq(source.shape[1], 1.0 / SR) > 19000.0] = 0.0
    band_limited = np.fft.irfft(band_limited, n=source.shape[1], axis=-1)

    out = render_array(band_limited, SR, at_rate(Recipe()))
    worst = float(np.max(np.abs((out - band_limited)[:, 2000:-2000])))
    R.check(worst < 1e-7,
            f"an empty recipe is transparent below 19 kHz at "
            f"{20 * math.log10(max(worst, 1e-300)):.1f} dB")

    # And the boundary is where it is claimed to be. Asserted so that moving
    # the passband edge shows up here as an intentional change rather than
    # somewhere downstream as a mysterious fraction of a dB.
    t = np.arange(int(SR)) / SR
    above = (0.5 * np.sin(2.0 * np.pi * 22000.0 * t))[None, :]
    kept = float(np.max(np.abs(render_array(above, SR, at_rate(Recipe()))[:, 4000:-4000])))
    R.check(kept < 0.5 * 0.5,
            f"content above 20 kHz is deliberately attenuated, 22 kHz kept "
            f"{20 * math.log10(kept / 0.5):.1f} dB")

    # THE TRANSPARENCY GUARANTEE, stated as a number rather than a hope. Every
    # render converts 48 -> 96 -> 48, and the anti-imaging filter's passband
    # edge is at 20 kHz, so this is exact below that and deliberately not above
    # it. Protecting 20-24 kHz would cost two to fourteen times the conversion
    # compute for content nobody can hear, which is the wrong trade; what
    # matters is that the boundary is known and enforced rather than discovered
    # later as a mystery 0.1 dB.
    white = RNG.normal(0.0, 0.2, (1, int(SR * 8)))
    freqs, before_power = spectrum.gated_ltas(white, SR)
    centres, before_bands = spectrum.third_octave_levels_db(freqs, before_power, SR)
    freqs, after_power = spectrum.gated_ltas(render_array(white, SR, at_rate(Recipe())), SR)
    _, after_bands = spectrum.third_octave_levels_db(freqs, after_power, SR)

    audible = (centres <= 16000.0) & np.isfinite(before_bands) & np.isfinite(after_bands)
    worst_band = float(np.max(np.abs((after_bands - before_bands)[audible])))
    R.check(worst_band < 0.01,
            f"the 96 kHz round trip is transparent to {worst_band:.4f} dB at or below 16 kHz")

    # Determinism: the same recipe twice is bit-identical. The engine asserts
    # this internally; asserting it through the subprocess boundary also
    # covers the recipe serialisation and the temporary-file path.
    recipe = at_rate(Recipe(seed=99)).add("plate_reverb", decay_s=1.2, mix=0.3)
    first = render_array(source, SR, recipe)
    second = render_array(source, SR, recipe)
    R.check(np.array_equal(first, second), "the same recipe renders bit-identically twice")

    # Block size must not change the output. This is the engine's central
    # invariant, and the CLI is where a caller could most easily break it.
    small = render_array(source, SR, recipe, block_frames=257)
    R.check(np.array_equal(first, small),
            "output is identical at block sizes 4096 and 257 (a prime)")

    # The engine's structural rules must survive the bridge, and must arrive as
    # readable errors rather than exit codes.
    for bad, what in (
        ({"nodes": [{"type": "true_peak_limiter"}]}, "a limiter in the internal chain"),
        ({"master": [{"type": "dither"}, {"type": "gain"}]}, "a node after dither"),
        ({"nodes": [{"type": "no_such_node"}]}, "an unknown node type"),
        # A misspelt filter shape used to fall back to a bell, silently. A
        # planner writing "high_shelf" instead of "highshelf" would have got a
        # plausible-sounding result that was not what it asked for.
        ({"nodes": [{"type": "parametric_eq", "params": {"bands": [
            {"filter_type": "high_shelf", "center_freq_hz": 4000.0, "gain_db": 6.0}]}}]},
         "a misspelt filter_type"),
    ):
        try:
            render_array(source, SR, bad)
            R.check(False, f"should have rejected {what}")
        except RenderError as error:
            R.check(len(str(error)) > 10, f"rejects {what} with an explanation")


def test_gain_against_loudness() -> None:
    print("[2] gain vs the loudness meter")

    source = pink(int(SR * 8), channels=2)
    reference = baseline(source)
    before = loudness.integrated_lufs(reference, SR)
    before_peak = loudness.true_peak_dbtp(reference, SR)

    # THE CORE CROSS-VALIDATION. A gain change is the one operation whose
    # effect on every level measurement is exactly predictable, so any
    # disagreement here is a calibration error in one side or the other.
    for gain_db in (-12.0, -6.0, -0.5, 3.0):
        out = render_array(source, SR, at_rate(Recipe()).add("gain", gain_db=gain_db))

        R.near(loudness.integrated_lufs(out, SR) - before, gain_db, 0.02,
               f"{gain_db:+} dB of gain moves integrated loudness by {gain_db:+}")
        R.near(loudness.true_peak_dbtp(out, SR) - before_peak, gain_db, 0.02,
               f"{gain_db:+} dB of gain moves true peak by {gain_db:+}")

    # Loudness range is a property of the material, not its level, so a gain
    # change must leave it alone. A meter that failed this would be reporting
    # something about its own scaling.
    before_lra = loudness.loudness_range_lu(reference, SR)
    quiet = render_array(source, SR, at_rate(Recipe()).add("gain", gain_db=-20.0))
    R.near(loudness.loudness_range_lu(quiet, SR), before_lra, 0.1,
           "gain does not change loudness range")


def test_eq_against_spectrum() -> None:
    print("[3] EQ vs the measured spectrum")

    source = pink(int(SR * 10))
    freqs, power = spectrum.gated_ltas(baseline(source), SR)
    centres, before = spectrum.third_octave_levels_db(freqs, power, SR)

    # A high shelf has a flat asymptote, so far above its corner the change is
    # exactly its gain. A bell would be a weaker test: its measured height
    # depends on how the band's width compares to a third of an octave, so a
    # disagreement could be blamed on either side.
    for gain_db in (-8.0, 4.0, 9.0):
        recipe = at_rate(Recipe()).add(
            "parametric_eq",
            bands=[{"filter_type": "highshelf", "center_freq_hz": 4000.0,
                    "gain_db": gain_db, "q_factor": 0.707}],
        )
        out = render_array(source, SR, recipe)
        freqs, power = spectrum.gated_ltas(out, SR)
        _, after = spectrum.third_octave_levels_db(freqs, power, SR)

        settled = (centres >= 12500.0) & np.isfinite(before) & np.isfinite(after)
        measured = float(np.mean((after - before)[settled]))
        R.near(measured, gain_db, 0.35,
               f"a {gain_db:+} dB high shelf raises the bands above it by {gain_db:+}")

        # And leaves the far side alone, which is what makes it a shelf rather
        # than a gain change.
        untouched = (centres <= 250.0) & np.isfinite(before) & np.isfinite(after)
        R.near(float(np.mean((after - before)[untouched])), 0.0, 0.35,
               f"a {gain_db:+} dB high shelf leaves the low end alone")

    # A narrow bell must show up in its own third-octave band and not its
    # neighbours — the test that the frequency axes of the two halves agree.
    recipe = at_rate(Recipe()).add(
        "parametric_eq",
        bands=[{"filter_type": "bell", "center_freq_hz": 1000.0,
                "gain_db": 12.0, "q_factor": 4.0}],
    )
    out = render_array(source, SR, recipe)
    freqs, power = spectrum.gated_ltas(out, SR)
    _, after = spectrum.third_octave_levels_db(freqs, power, SR)
    delta = after - before
    peak_band = float(centres[int(np.nanargmax(np.where(np.isfinite(delta), delta, -np.inf)))])
    R.check(peak_band == 1000.0,
            f"a 1 kHz bell moves the 1 kHz band most, got {peak_band} Hz")
    two_octaves_away = np.abs(delta[(centres <= 250.0) | (centres >= 4000.0)])
    R.check(float(np.nanmax(two_octaves_away)) < 1.5,
            "and barely touches anything two octaves away")


def test_compressor_against_dynamics() -> None:
    print("[4] compressor vs the dynamics measurements")

    # A steady tone above the threshold has an exactly predictable output
    # level: the static curve of a hard-knee compressor is arithmetic. Using a
    # tone rather than music removes the attack and release behaviour from the
    # comparison, so what is left is purely the gain computer.
    source = tone(int(SR * 6), 1000.0, -10.0)
    threshold, ratio = -20.0, 4.0
    expected = threshold + (-10.0 - threshold) / ratio  # -17.5 dBFS

    recipe = at_rate(Recipe()).add(
        "compressor", threshold_dbfs=threshold, ratio_to_1=ratio, knee_db=0.0,
        attack_ms=1.0, release_ms=50.0, makeup_gain_db=0.0,
    )
    out = render_array(source, SR, recipe)

    settled = out[:, int(SR * 2):]
    measured = 20.0 * math.log10(float(np.sqrt(np.mean(settled ** 2))) * math.sqrt(2.0))
    R.near(measured, expected, 0.25,
           f"a {ratio}:1 compressor at {threshold} dBFS puts a -10 dBFS tone at {expected}")

    # On real-shaped material it must reduce crest factor and loudness range.
    # Direction rather than magnitude: the exact amount depends on the
    # programme, and asserting a number would be asserting a property of the
    # test signal.
    material = pink(int(SR * 12)) * np.concatenate([
        np.linspace(0.2, 1.0, int(SR * 6)), np.linspace(1.0, 0.2, int(SR * 6))
    ])
    squashed = render_array(
        material, SR,
        at_rate(Recipe()).add("compressor", threshold_dbfs=-30.0, ratio_to_1=6.0,
                              attack_ms=5.0, release_ms=100.0),
    )
    from backend.measure import dynamics

    R.check(dynamics.crest_factor_db(squashed) < dynamics.crest_factor_db(material) - 1.0,
            f"compression lowers crest factor "
            f"({dynamics.crest_factor_db(material):.1f} -> {dynamics.crest_factor_db(squashed):.1f} dB)")
    R.check(loudness.loudness_range_lu(squashed, SR) < loudness.loudness_range_lu(material, SR),
            "compression lowers loudness range")


def test_limiter_true_peak() -> None:
    print("[5] limiter vs an independent true-peak meter")

    # TWO INDEPENDENT TRUE-PEAK IMPLEMENTATIONS, ONE ANSWER. The engine's
    # limiter uses its own C++ oversampling meter to decide how much to pull
    # down; the Python meter here shares no code with it. Agreement means both
    # are right about BS.1770 Annex 2, and disagreement localises to whichever
    # one differs from the analytic sinusoid case its own suite pins.
    source = pink(int(SR * 6), channels=2, amplitude=0.95)

    for ceiling in (-1.0, -0.3, -6.0):
        recipe = at_rate(Recipe())
        recipe.add_master("true_peak_limiter", ceiling_dbtp=ceiling,
                          lookahead_ms=5.0, release_ms=80.0, oversample_factor=8)
        out = render(source, recipe, sample_rate=SR)

        measured = loudness.true_peak_dbtp(out.samples, SR)
        R.check(measured <= ceiling + 0.15,
                f"a {ceiling} dBTP ceiling holds, measured {measured:.3f} dBTP")
        # And it must actually be working rather than simply attenuating: the
        # result should sit near the ceiling, not far below it.
        R.check(measured > ceiling - 2.0,
                f"and the limiter reaches its {ceiling} dBTP ceiling, measured {measured:.3f}")


def test_master_ordering_is_load_bearing() -> None:
    print("[6] why master[] runs after the sample-rate conversion")

    # THE ARCHITECTURAL CLAIM, MEASURED. The engine refuses to put a limiter in
    # the internal chain, on the grounds that true peak is defined at the
    # DELIVERY rate and a ceiling met at 96 kHz can be exceeded after
    # conversion to 44.1. That rule costs real complexity — two node lists, a
    # validator, a second prepare — so it is worth confirming the effect is
    # real rather than theoretical.
    #
    # Limiting at 96 kHz is simulated by rendering the limiter with a 96 kHz
    # delivery rate, then converting that result to 44.1 separately, which is
    # exactly the order the rule forbids.
    source = pink(int(SR * 6), channels=2, amplitude=0.95)
    ceiling = -1.0

    correct = Recipe(output_sample_rate=44100.0)
    correct.add_master("true_peak_limiter", ceiling_dbtp=ceiling,
                       lookahead_ms=5.0, oversample_factor=8)
    at_delivery_rate = render(source, correct, sample_rate=SR)
    R.check(loudness.true_peak_dbtp(at_delivery_rate.samples, 44100.0) <= ceiling + 0.15,
            "limiting AFTER the conversion holds the ceiling at the delivery rate")

    wrong = Recipe(output_sample_rate=96000.0)
    wrong.add_master("true_peak_limiter", ceiling_dbtp=ceiling,
                     lookahead_ms=5.0, oversample_factor=8)
    at_internal_rate = render(source, wrong, sample_rate=SR)

    # Now convert that already-limited 96 kHz signal down, as a chain that
    # limited too early would.
    converted = render(
        at_internal_rate.samples, Recipe(output_sample_rate=44100.0),
        sample_rate=96000.0,
    )
    # THE PROPERTY IS THAT THE WRONG ORDER LOSES CONTROL, NOT THAT IT GOES UP.
    #
    # This used to assert `overshoot > 0` — that limiting early makes the
    # delivered peak exceed the ceiling. That direction was never guaranteed and
    # this test was the only thing claiming it. Measured on this signal the
    # early-limited chain comes out at -1.71 dBTP against the correct order's
    # -1.17: the 96 kHz limiter sees different inter-sample peaks, limits
    # HARDER, and the conversion does not put the level back. Quieter, not
    # louder, and still wrong.
    #
    # What the master section actually buys is that the ceiling is enforced at
    # the rate it is DEFINED at. Limiting anywhere else leaves the delivery-rate
    # peak somewhere nobody chose. So the assertion is about distance from the
    # target rather than sign, which is both the real rule and independent of
    # whatever margin the limiter carries — the previous form broke the moment
    # the limiter's own margin changed, which is a test coupled to a constant it
    # was not about.
    delivered_wrong = loudness.true_peak_dbtp(converted.samples, 44100.0)
    delivered_right = loudness.true_peak_dbtp(at_delivery_rate.samples, 44100.0)

    # DISTANCE FROM THE CEILING IS THE WRONG YARDSTICK TOO, and getting that
    # wrong twice is worth writing down. The correct order sits deliberately
    # under the ceiling — the limiter subtracts its detector's own error bound —
    # so scoring "closest to the ceiling" rewards the chain that is merely lucky
    # and penalises the one that is safe.
    #
    # The rule the master section encodes is that the two orders are NOT
    # interchangeable and only one of them carries a guarantee. So: they differ
    # measurably, and the correct one holds the ceiling. Neither clause depends
    # on any margin, which is what the previous two versions of this assertion
    # did depend on without saying so.
    R.check(abs(delivered_wrong - delivered_right) > 0.05,
            f"the two orders are not interchangeable: {delivered_wrong:+.3f} "
            f"dBTP limiting before the conversion against {delivered_right:+.3f} "
            f"limiting after")
    R.check(delivered_right <= ceiling,
            f"and only the correct order carries the guarantee "
            f"({delivered_right:+.3f} dBTP against a {ceiling:+.1f} ceiling), "
            f"which is why the master section exists")


def test_utility_nodes() -> None:
    print("[7] utility nodes vs their measurements")

    source = pink(int(SR * 4), channels=2)
    reference = baseline(source)

    # Polarity inversion must be exact, and must show as an exactly inverted
    # correlation against the untouched channel.
    flipped = render_array(source, SR, at_rate(Recipe()).add("polarity", inverted=1.0))
    R.check(np.allclose(flipped, -reference, atol=1e-12), "polarity inverts exactly")

    # Stereo width, measured as correlation. Collapsing to mono must produce
    # correlation 1; the measurement and the DSP have to agree on what width
    # means for that to hold.
    narrowed = render_array(
        source, SR, at_rate(Recipe()).add("stereo_width", width_percent=0.0)
    )
    R.near(stereo.broadband_correlation(narrowed), 1.0, 1e-6,
           "width 0% collapses to mono, correlation 1")

    widened = render_array(
        source, SR, at_rate(Recipe()).add("stereo_width", width_percent=200.0)
    )
    R.check(stereo.broadband_correlation(widened) < stereo.broadband_correlation(reference),
            "width 200% lowers correlation")

    # DC block, against the hygiene measurement.
    from backend.measure import hygiene

    offset = source + 0.05
    R.near(hygiene.dc_offset(offset)[0], 20.0 * math.log10(0.05), 0.5,
           "a 0.05 DC offset measures where it should")
    cleaned = render_array(offset, SR, at_rate(Recipe()).add("dc_block", cutoff_hz=20.0))
    R.check(hygiene.dc_offset(cleaned)[0] < -60.0,
            f"dc_block removes it, leaving {hygiene.dc_offset(cleaned)[0]:.1f} dBFS")


def test_report_survives_a_render() -> None:
    print("[8] a full report over a rendered file")

    # End to end, the way the product will actually run: measure, process,
    # measure again, and confirm the report describes what changed.
    source = pink(int(SR * 8), channels=2, amplitude=0.4)
    before = report_mod.measure(baseline(source), SR).to_dict()

    recipe = at_rate(Recipe(seed=7))
    recipe.add("parametric_eq", bands=[
        {"filter_type": "highshelf", "center_freq_hz": 6000.0, "gain_db": 5.0,
         "q_factor": 0.707},
    ])
    recipe.add("compressor", threshold_dbfs=-24.0, ratio_to_1=3.0,
               attack_ms=10.0, release_ms=120.0, makeup_gain_db=2.0)
    recipe.add_master("true_peak_limiter", ceiling_dbtp=-1.0, lookahead_ms=5.0)

    out = render(source, recipe, sample_rate=SR)
    after = report_mod.measure(out.samples, SR).to_dict()

    R.check(after["loudness"]["true_peak_dbtp"] <= -0.85,
            f"the rendered file honours the ceiling, {after['loudness']['true_peak_dbtp']} dBTP")
    # NOT "the chain made it louder". It did not, and asserting it would have
    # been asserting a property of this particular signal rather than of the
    # chain: a 3:1 compressor at -24 dBFS takes about 6.8 dB off pink noise
    # that sits well above its threshold, which +2 dB of makeup does not
    # restore. The measurable claim is that the DYNAMICS were reduced, and
    # peak-to-loudness ratio is the number that says so.
    R.check(after["dynamics"]["peak_to_loudness_ratio_db"]
            < before["dynamics"]["peak_to_loudness_ratio_db"],
            f"the chain reduced peak-to-loudness ratio "
            f"({before['dynamics']['peak_to_loudness_ratio_db']} -> "
            f"{after['dynamics']['peak_to_loudness_ratio_db']} dB)")

    # Makeup gain, isolated. Two renders differing in exactly one parameter,
    # so the result is arithmetic rather than a property of the material —
    # which is the only way to check a single knob inside a chain.
    def with_makeup(makeup_db: float) -> float:
        one = at_rate(Recipe(seed=7))
        one.add("compressor", threshold_dbfs=-24.0, ratio_to_1=3.0,
                attack_ms=10.0, release_ms=120.0, makeup_gain_db=makeup_db)
        return loudness.integrated_lufs(render_array(source, SR, one), SR)

    R.near(with_makeup(2.0) - with_makeup(0.0), 2.0, 0.02,
           "makeup gain moves loudness by exactly its own value")
    R.check(after["spectrum"]["hf_rolloff_db_per_octave"]
            > before["spectrum"]["hf_rolloff_db_per_octave"],
            "the high shelf shows as a flatter measured HF rolloff")
    R.check(after["hygiene"]["non_finite_samples"] == 0,
            "the render contains no NaN or infinite samples")
    R.check(not any(w["severity"] == "critical" for w in after["warnings"]),
            f"and raises no critical warnings, got {[w['code'] for w in after['warnings']]}")


if __name__ == "__main__":
    if not os.path.exists(DEFAULT_BINARY):
        print(f"engine binary missing at {DEFAULT_BINARY}; run `make render` in engine/")
        sys.exit(2)

    print("Audionova engine/measurement cross-validation\n")
    test_bridge()
    test_gain_against_loudness()
    test_eq_against_spectrum()
    test_compressor_against_dynamics()
    test_limiter_true_peak()
    test_master_ordering_is_load_bearing()
    test_utility_nodes()
    test_report_survives_a_render()
    print(f"\n{R.checks} checks, {R.failures} failure(s)")
    sys.exit(1 if R.failures else 0)
