"""The measurement report must be able to see what the engine does.

This is the harness's pass/fail face. ``run_sensitivity`` prints a table for a
human to read; this asserts only the hypotheses that were written down in the
perturbation catalogue BEFORE anything was run.

That ordering is the whole point. An analysis that reads whatever moved and
declares it the answer will confirm any measurement suite, including a broken
one — which is precisely how CLAP-guided optimisation came to be published as
an improvement while scoring worse than random parameters on real EQ matching.
A prediction made in advance can fail.

It runs on synthetic sources rather than the stems, so it is hermetic and lives
in CI. The stems are richer and are what ``run_sensitivity`` is for; what this
needs is material with known, controlled content across the whole band.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.engine.render import DEFAULT_BINARY  # noqa: E402
from backend.eval import corpus, sensitivity  # noqa: E402

SR = 48000.0
RNG = np.random.default_rng(4242)


class Results:
    def __init__(self) -> None:
        self.checks = 0
        self.failures = 0

    def check(self, ok: bool, what: str) -> None:
        self.checks += 1
        if not ok:
            self.failures += 1
            print(f"\n  FAIL  {what}")


R = Results()


def source(kind: str, seconds: float = 6.0) -> np.ndarray:
    """Two-channel material with content across the whole band.

    Decorrelated channels, because a width perturbation applied to dual mono
    has nothing to act on and would look undetectable for the wrong reason.
    """
    n = int(SR * seconds)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    shape = np.ones_like(freqs)
    shape[1:] = 1.0 / np.sqrt(freqs[1:])
    shape[0] = 0.0

    spec = RNG.normal(size=(2, freqs.size)) + 1j * RNG.normal(size=(2, freqs.size))
    noise = np.fft.irfft(spec * shape, n=n, axis=-1)
    noise = noise / np.max(np.abs(noise)) * 0.35

    if kind == "steady":
        return noise

    # Something with an envelope, so dynamics and modulation perturbations have
    # a target. Steady noise has no dynamics for a compressor to reduce.
    t = np.arange(n) / SR
    envelope = 0.25 + 0.75 * np.abs(np.sin(2.0 * np.pi * 0.7 * t)) ** 3
    return noise * envelope


def test_hypotheses() -> None:
    print("[1] predicted detectability")

    if not os.path.exists(DEFAULT_BINARY):
        print("  engine binary missing; skipping")
        return

    sources = {
        "steady": source("steady"),
        "dynamic": source("dynamic"),
        "quiet": source("dynamic") * 0.3,
    }

    for perturbation in corpus.PERTURBATIONS:
        by_source = {
            name: sensitivity.sweep_source(samples, SR, perturbation)
            for name, samples in sources.items()
            if perturbation.applies_to(corpus.SourceContext.measure(samples, SR))
        }
        R.check(bool(by_source),
                f"{perturbation.name}: applies to at least one synthetic source")
        if not by_source:
            continue
        effects = sensitivity.analyse(perturbation, by_source)
        ok, missing = sensitivity.hypothesis_verdict(perturbation, effects)

        # Something must move. A perturbation that changes no measurement at
        # all is invisible to the planner by construction.
        best = effects[0] if effects else None
        R.check(best is not None and best.diagnostic,
                f"{perturbation.name}: at least one field moves diagnostically")

        R.check(ok, f"{perturbation.name}: predicted fields moved "
                    f"(missing: {', '.join(missing) if missing else 'none'})")


def test_flatten() -> None:
    print("[2] report flattening")

    flat = sensitivity.flatten({
        "a": 1.0,
        "b": {"c": 2.0, "d": None, "e": True},
        "f": [{"g": 3.0}],
        "h": "text",
        "i": float("inf"),
    })
    R.check(flat == {"a": 1.0, "b.c": 2.0},
            f"flatten keeps finite scalars and drops the rest, got {flat}")

    # BAND LISTS ARE EXPANDED, keyed by centre frequency. Dropping them made a
    # 1 kHz bell look undetectable except through summary statistics that a
    # plain gain change moves identically — and the conclusion would have been
    # that the report cannot localise mid-band EQ, which is most of mixing.
    # The bands were in the report the whole time.
    bands = sensitivity.flatten({
        "spectrum": {"third_octave_db": [
            {"centre_hz": 1000.0, "level_db": -20.0},
            {"centre_hz": 2000.0, "level_db": -24.0},
            {"centre_hz": 4000.0, "level_db": None},
        ]},
    })
    R.check(bands == {"spectrum.third_octave_db.1000hz.level_db": -20.0,
                      "spectrum.third_octave_db.2000hz.level_db": -24.0},
            f"band lists expand keyed by frequency, got {bands}")

    # Keyed by frequency rather than index, so the names survive a change to
    # the band layout.
    R.check(all("hz." in k for k in bands), "band keys carry their frequency")

    # Booleans are integers in Python and would otherwise arrive as 1.0 and be
    # averaged, which is meaningless for a flag.
    R.check("b.e" not in flat, "flatten excludes booleans")
    R.check("i" not in flat, "flatten excludes non-finite values")


def test_effect_size() -> None:
    print("[3] effect size")

    class P:
        expected_fields = ()

    def sweep(pairs):
        return {"one": [(v, {"x": d}) for v, d in pairs]}

    # A field that tracks the parameter is readable, and that is the point.
    tracking = sensitivity.analyse(P(), sweep(
        [(-6.0, -6.1), (-3.0, -2.9), (3.0, 3.05), (6.0, 5.9)]))[0]
    R.check(tracking.monotonic and tracking.diagnostic,
            f"a field tracking the parameter is monotonic, r={tracking.r:+.2f}")
    R.check(abs(tracking.slope - 1.0) < 0.1,
            f"and its slope recovers the relationship, {tracking.slope:.3f}")

    # THE CASE THE FIRST METRIC GOT WRONG. Sweeping a parameter through zero
    # makes the delta change sign, which is correct behaviour and was being
    # scored as undetectable — it marked gain, the most obviously visible
    # effect there is, as invisible.
    R.check(tracking.r < -0.7 or tracking.r > 0.7,
            "a sign-changing sweep is not penalised for changing sign")

    # A symmetric effect: the delta follows |value| but not value. Still
    # useful, and calling it invisible would discard real evidence.
    symmetric = sensitivity.analyse(P(), sweep(
        [(-9.0, 4.5), (-5.0, 2.4), (5.0, 2.6), (9.0, 4.4)]))[0]
    R.check(not symmetric.monotonic and symmetric.magnitude,
            f"a symmetric effect is magnitude-diagnostic, r={symmetric.r:+.2f} "
            f"r_abs={symmetric.r_abs:+.2f}")

    # Noise must not qualify.
    noise = sensitivity.analyse(P(), sweep(
        [(-6.0, 0.4), (-3.0, -0.7), (3.0, 0.2), (6.0, -0.5)]))[0]
    R.check(not noise.diagnostic,
            f"an unrelated field is not diagnostic, r={noise.r:+.2f}")

    # A field that never moves has zero variance and must not divide its way
    # to a perfect correlation.
    flat = sensitivity.analyse(P(), sweep(
        [(-6.0, 0.0), (-3.0, 0.0), (3.0, 0.0), (6.0, 0.0)]))[0]
    R.check(not flat.diagnostic, f"a field that never moves is not, r={flat.r:+.2f}")

    # CORRELATION IS NOT ENOUGH. A 1 kHz bell moves the HF rolloff at r = -1.00
    # with a slope of 0.0085 dB/octave per dB — a flawless relationship whose
    # entire swing is under a tenth of a dB. The report rounds to two decimals,
    # so a field that small is not weak evidence, it is absent from the
    # document the planner is handed.
    tiny = sensitivity.analyse(P(), sweep(
        [(-9.0, -0.008), (-5.0, -0.004), (5.0, 0.004), (9.0, 0.008)]))[0]
    R.check(tiny.monotonic, f"a tiny effect can still correlate, r={tiny.r:+.2f}")
    R.check(not tiny.visible and not tiny.diagnostic,
            f"but it is not diagnostic, span {tiny.span:.4f}")


def test_excerpt_selection() -> None:
    print("[4] excerpt selection")

    # The chosen window must be the active one, not the middle and not the
    # loudest single moment. Built here as silence, then programme, then
    # silence, so the right answer is unambiguous.
    n = int(SR * 30)
    signal = np.zeros((1, n))
    active = slice(int(SR * 18), int(SR * 26))
    signal[:, active] = source("steady", 8.0)[:1]

    import tempfile

    from backend.audio import wavio

    with tempfile.TemporaryDirectory() as work:
        path = os.path.join(work, "sparse.wav")
        wavio.write_wav(path, signal, SR, 24)
        excerpt, rate = corpus.load_excerpt(path, seconds=8.0)

    R.check(rate == SR, "excerpt keeps the source rate")
    R.check(excerpt.shape[1] == int(SR * 8.0), "excerpt is the requested length")

    energy = float(np.mean(excerpt ** 2))
    whole = float(np.mean(signal ** 2))
    R.check(energy > whole * 2.0,
            f"the excerpt lands on the active region ({energy:.2e} vs {whole:.2e} overall)")

    # A file shorter than the window comes back whole rather than padded.
    short = np.ones((1, int(SR * 2))) * 0.1
    with tempfile.TemporaryDirectory() as work:
        path = os.path.join(work, "short.wav")
        wavio.write_wav(path, short, SR, 24)
        excerpt, _ = corpus.load_excerpt(path, seconds=25.0)
    R.check(excerpt.shape[1] == int(SR * 2), "a short file is returned whole")


def test_applicability() -> None:
    print("[6] applicability")

    # Stereo width on dual mono changes nothing, correctly — there is no side
    # signal. All four reference stems are mono or dual-mono, so without this
    # gate the sweep reported a measurement failure for a node behaving exactly
    # as designed.
    width = corpus.PERTURBATIONS_BY_NAME["stereo_width"]
    mono = corpus.SourceContext(channels=1, sample_rate=SR,
                                integrated_lufs=-18.0, correlation=1.0)
    dual_mono = corpus.SourceContext(channels=2, sample_rate=SR,
                                     integrated_lufs=-18.0, correlation=1.0)
    real_stereo = corpus.SourceContext(channels=2, sample_rate=SR,
                                       integrated_lufs=-18.0, correlation=0.4)

    R.check(not width.applies_to(mono), "width does not apply to a mono source")
    R.check(not width.applies_to(dual_mono), "width does not apply to dual mono")
    R.check(width.applies_to(real_stereo), "width applies to real stereo")

    # Everything else applies everywhere; a gate that silently excluded a
    # perturbation from the whole corpus would hide a real detection failure.
    for perturbation in corpus.PERTURBATIONS:
        if perturbation.name == "stereo_width":
            continue
        R.check(perturbation.applies_to(mono),
                f"{perturbation.name} applies to mono material")


def test_source_relative_parameters() -> None:
    print("[7] source-relative parameters")

    # A fixed compressor threshold means something different on every source.
    # Measured on real stems, the ratio sweep at a fixed -28 dBFS crushed a
    # bass at -11.5 LUFS while barely touching a guitar at -22.6, which made it
    # a sweep of two variables and lost crest factor as a detectable field.
    compression = corpus.PERTURBATIONS_BY_NAME["compression"]

    loud = corpus.SourceContext(channels=1, sample_rate=SR,
                                integrated_lufs=-11.5, correlation=1.0)
    quiet = corpus.SourceContext(channels=1, sample_rate=SR,
                                 integrated_lufs=-22.6, correlation=1.0)

    loud_threshold = compression.build(4.0, loud)[0]["params"]["threshold_dbfs"]
    quiet_threshold = compression.build(4.0, quiet)[0]["params"]["threshold_dbfs"]

    R.check(abs((loud_threshold - loud.integrated_lufs)
                - (quiet_threshold - quiet.integrated_lufs)) < 1e-9,
            "the compressor threshold sits the same distance below each source's level")
    R.check(loud_threshold > quiet_threshold,
            f"so a louder source gets a higher threshold "
            f"({loud_threshold:.1f} vs {quiet_threshold:.1f} dBFS)")


def test_catalogue_is_well_formed() -> None:
    print("[5] perturbation catalogue")

    seen: set[str] = set()
    for perturbation in corpus.PERTURBATIONS:
        R.check(perturbation.name not in seen, f"{perturbation.name} is unique")
        seen.add(perturbation.name)

        R.check(len(perturbation.values) >= 2,
                f"{perturbation.name} sweeps at least two values")
        R.check(len(perturbation.expected_fields) >= 1,
                f"{perturbation.name} states a hypothesis in advance")

        context = corpus.SourceContext(channels=2, sample_rate=SR,
                                       integrated_lufs=-18.0, correlation=0.2)
        nodes = perturbation.build(perturbation.values[0], context)
        R.check(bool(nodes) and all("type" in n for n in nodes),
                f"{perturbation.name} builds well-formed nodes")


if __name__ == "__main__":
    print("Audionova sensitivity harness\n")
    test_flatten()
    test_effect_size()
    test_excerpt_selection()
    test_catalogue_is_well_formed()
    test_applicability()
    test_source_relative_parameters()
    test_hypotheses()
    print(f"\n{R.checks} checks, {R.failures} failure(s)")
    sys.exit(1 if R.failures else 0)
