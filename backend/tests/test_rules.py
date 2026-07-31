"""The rule-based planner: does it decline as well as it acts?

A planner is judged as much by what it leaves alone. Any set of rules can find
something to do to every file; the hard part is not cutting a bass guitar's low
end because it measures loud, or brightening a source that is already bright.

So most of what follows tests refusals. The moves are checked too, but a planner
that always moves is a planner that will eventually ruin something.
"""

from __future__ import annotations

import copy
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.engine.render import DEFAULT_BINARY, render_array  # noqa: E402
from backend.measure.report import measure, measure_file  # noqa: E402
from backend.plan import rules  # noqa: E402
from backend.plan.validate import validate  # noqa: E402
from backend.verify import gate  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STEMS = os.path.join(REPO_ROOT, "audio", "stems")
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


def stem(name: str) -> str:
    return os.path.join(STEMS, f"{name}.wav")


def have_stems() -> bool:
    return os.path.exists(stem("vocals"))


def eq_bands(graph) -> list[dict]:
    return [b for n in graph.nodes if n.type == "parametric_eq"
            for b in n.params.get("bands", [])]


def synthetic(*, rolloff_db_per_octave: float = 0.0,
              seconds: float = 8.0) -> np.ndarray:
    """Noise with a MEASURED band slope, not an opaque shaping exponent.

    The first version took a `tilt` parameter that fed an exponent directly, and
    `tilt=-8` produced a measured rolloff of -2.5 dB/octave — nowhere near the
    -12 the planner needs to consider brightening. The test that depended on it
    could not distinguish anything, and passed for the wrong reason.

    Band-integrated level rises as f^(2a+1) for an amplitude spectrum f^a, which
    is 3.01 dB per octave per unit, so the exponent follows from the slope you
    actually want. Pink noise is a = -0.5 and measures 0; white is a = 0 and
    measures +3.
    """
    exponent = (rolloff_db_per_octave / 3.0103 - 1.0) / 2.0

    n = int(SR * seconds)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    shape = np.zeros_like(freqs)
    shape[1:] = freqs[1:] ** exponent
    spec = RNG.normal(size=freqs.size) + 1j * RNG.normal(size=freqs.size)
    out = np.fft.irfft(spec * shape, n=n)
    return (out / np.max(np.abs(out)) * 0.3)[None, :]


# ---------------------------------------------------------------------------


def test_always_valid() -> None:
    print("[1] every plan validates")

    if not have_stems():
        print("  stems missing; skipping")
        return

    for name in ("vocals", "acoustic-guitar-track", "bass", "drums",
                 "stereo-guitar", "acoustic-guitar-riff"):
        report = measure_file(stem(name)).to_dict()
        graph, _ = rules.plan(report)
        result = validate(graph, report)

        R.check(result.ok, f"{name}: the plan is valid — {result.summary()}")
        R.check(not result.repairs,
                f"{name}: and needs no repairs — {result.repairs}")

        # Every node must justify itself in the same field the model is asked to
        # fill. A rule that cannot say why it fired is one nobody can argue with.
        for node in list(graph.nodes) + list(graph.master):
            R.check(len(node.rationale) > 20,
                    f"{name}: {node.id} states its reason")


def test_declines_on_bass() -> None:
    print("[2] a bass keeps its low end")

    if not have_stems():
        print("  stems missing; skipping")
        return

    # THE WORST OUTPUT THIS PLANNER HAS PRODUCED, and the reason the resonance
    # rule is written the way it is. Scoring bands against the distant midrange,
    # it proposed a 5 dB cut at 100 Hz on a bass — because 100 Hz sits 41 dB
    # above the midrange there. It does. That is what a bass is.
    report = measure_file(stem("bass")).to_dict()
    graph, explanation = rules.plan(report)

    for band in eq_bands(graph):
        if band["filter_type"] == "bell" and band["center_freq_hz"] < 400.0:
            R.check(False, f"bass got a {band['gain_db']:+.1f} dB cut at "
                           f"{band['center_freq_hz']:.0f} Hz, which is its range")
    R.check(True, "no bell cut below 400 Hz on a bass")

    # And its high-pass must sit below the fundamentals, not through them.
    highpass = [b for b in eq_bands(graph) if b["filter_type"] == "highpass"]
    if highpass:
        R.check(highpass[0]["center_freq_hz"] <= 40.0,
                f"the high-pass sits at {highpass[0]['center_freq_hz']:.0f} Hz, "
                "below the fundamentals")


def test_declines_when_measurements_say_no() -> None:
    print("[3] refusals")

    if not have_stems():
        print("  stems missing; skipping")
        return

    # Already compressed: more compression removes what is left rather than
    # controlling anything.
    report = measure_file(stem("stereo-guitar")).to_dict()
    plr = report["dynamics"]["peak_to_loudness_ratio_db"]
    graph, explanation = rules.plan(report)
    R.check(plr < rules.ALREADY_COMPRESSED_PLR_DB,
            f"the stereo guitar is already compressed, PLR {plr:.1f} dB")
    R.check(not any(n.type == "compressor" for n in graph.nodes),
            "so no compressor is added")
    R.check(any("compressed" in d for d in explanation.declined),
            f"and the refusal is explained — {explanation.declined}")

    # Already bright: brightening it further is a taste decision, not a
    # measurement-backed one.
    report = measure_file(stem("acoustic-guitar-track")).to_dict()
    rolloff = report["spectrum"]["hf_rolloff_db_per_octave"]
    graph, explanation = rules.plan(report)
    R.check(rolloff > rules.DARK_ROLLOFF_DB_PER_OCTAVE,
            f"the acoustic guitar is already bright, {rolloff:.1f} dB/octave")
    R.check(not any(b["filter_type"] == "highshelf" and b["gain_db"] > 0
                    for b in eq_bands(graph)),
            "so no air is added")

    # --- the air shelf's three gates ---------------------------------------
    #
    # THE REPORT FIELDS ARE SET DIRECTLY rather than synthesised, and that is a
    # deliberate change. Two attempts to build audio dark enough to trigger the
    # shelf both measured -16.8 dB/octave whatever slope was requested: past a
    # certain steepness the top bands fall to numerical zero and the fit floors
    # out. Chasing that would have been testing the noise generator.
    #
    # What is under test here is the DECISION, so it gets its inputs directly.
    # The measurement layer that produces them has its own suite.
    def air_report(**overrides) -> dict:
        base = measure(synthetic(rolloff_db_per_octave=-6.0), SR).to_dict()
        base["spectrum"]["hf_rolloff_db_per_octave"] = -26.0
        base["spectrum"]["bandwidth_limit_hz"] = None
        base["spectrum"]["bandwidth_limit_confidence"] = 0.0
        base["hygiene"]["noise_floor_dbfs"] = -80.0
        base["hygiene"]["noise_floor_confidence"] = 0.9
        for key, value in overrides.items():
            section, field_name = key.split("__")
            base[section][field_name] = value
        return base

    # Dark, clean, unlimited: the shelf should fire. If it does not, none of the
    # refusals below prove anything.
    graph, _ = rules.plan(air_report())
    R.check(any(b["filter_type"] == "highshelf" and b["gain_db"] > 0
                for b in eq_bands(graph)),
            "a genuinely dark source with a clean floor gets an air shelf")

    # A confident codec wall makes the boost inaudible whatever the rolloff says.
    graph, _ = rules.plan(air_report(spectrum__bandwidth_limit_hz=15000.0,
                                     spectrum__bandwidth_limit_confidence=0.9))
    R.check(not any(b["filter_type"] == "highshelf" and b["gain_db"] > 0
                    for b in eq_bands(graph)),
            "a confident codec wall blocks it")

    # A low-confidence one must not, or a marginal detection silently forbids a
    # legitimate move — which is exactly the detector wobble already on file.
    graph, _ = rules.plan(air_report(spectrum__bandwidth_limit_hz=15000.0,
                                     spectrum__bandwidth_limit_confidence=0.2))
    R.check(any(b["filter_type"] == "highshelf" and b["gain_db"] > 0
                for b in eq_bands(graph)),
            "a low-confidence wall does not")

    # An audible noise floor makes brightening a way to amplify hiss.
    graph, _ = rules.plan(air_report(hygiene__noise_floor_dbfs=-40.0))
    R.check(not any(b["filter_type"] == "highshelf" and b["gain_db"] > 0
                    for b in eq_bands(graph)),
            "an audible noise floor blocks it too")

    # And a shelf smaller than a dB is not worth a band. The floor exists so the
    # planner does not spend a move on something nobody can hear.
    R.check(rules.choose_air(air_report(spectrum__hf_rolloff_db_per_octave=-16.0))
            is None,
            "a source only slightly dark yields a sub-1 dB shelf, so it declines")


def test_warnings_drive_moves() -> None:
    print("[4] the report's warnings are acted on")

    report = measure(synthetic(), SR).to_dict()

    phase = copy.deepcopy(report)
    phase["warnings"] = [{"code": "out_of_phase", "severity": "critical",
                          "message": ""}]
    graph, _ = rules.plan(phase)
    R.check(graph.nodes and graph.nodes[0].type == "polarity",
            "an out-of-phase source gets polarity fixed FIRST, before anything "
            "else is decided")

    offset = copy.deepcopy(report)
    offset["warnings"] = [{"code": "dc_offset", "severity": "medium", "message": ""}]
    graph, _ = rules.plan(offset)
    R.check(any(n.type == "dc_block" for n in graph.nodes),
            "a DC offset gets a blocker")


def test_master_section() -> None:
    print("[5] the master section")

    report = measure(synthetic(), SR).to_dict()

    graph, _ = rules.plan(report, output_bit_depth=24)
    R.check([n.type for n in graph.master] == ["true_peak_limiter", "dither"],
            f"limiter then dither, got {[n.type for n in graph.master]}")

    graph, _ = rules.plan(report, output_bit_depth=None)
    R.check([n.type for n in graph.master] == ["true_peak_limiter"],
            "no dither when the delivery format is float — dither there only "
            "adds noise to a signal that had none")

    graph, _ = rules.plan(report, output_bit_depth=32)
    R.check(not any(n.type == "dither" for n in graph.master),
            "and none at 32 bits either")


def test_reverb_is_opt_in() -> None:
    print("[6] space is requested, never inferred")

    report = measure(synthetic(), SR).to_dict()

    graph, _ = rules.plan(report)
    R.check(not any(n.type == "plate_reverb" for n in graph.nodes),
            "no reverb by default — no measurement says a dry source wants to "
            "be wet, only a listener")

    graph, _ = rules.plan(report, reverb_mix=0.15)
    plate = [n for n in graph.nodes if n.type == "plate_reverb"]
    R.check(plate and abs(plate[0].mix - 0.15) < 1e-9,
            "and exactly what was asked for when it is")


def test_explanations_are_honest() -> None:
    print("[7] the explanation records refusals")

    if not have_stems():
        print("  stems missing; skipping")
        return

    report = measure_file(stem("acoustic-guitar-track")).to_dict()
    graph, explanation = rules.plan(report)

    R.check(explanation.moves, "moves are described")
    R.check(explanation.declined, "and so are the refusals")
    R.check("left alone" in explanation.text(),
            "the graph's explanation names what was not touched")

    # Every stated move must cite a number, or the explanation is a story.
    for move in explanation.moves:
        R.check(any(ch.isdigit() for ch in move),
                f"the move cites a measurement: {move[:70]}")


def test_renders_and_passes_the_gate() -> None:
    print("[8] planned chains render and survive the safety gate")

    if not have_stems() or not os.path.exists(DEFAULT_BINARY):
        print("  stems or engine missing; skipping")
        return

    from backend.eval import corpus

    for name in ("vocals", "acoustic-guitar-track", "bass"):
        source, rate = corpus.load_excerpt(stem(name), 12.0)
        report = measure(source, rate).to_dict()

        graph, _ = rules.plan(report)
        result = validate(graph, report)
        R.check(result.ok, f"{name}: plan validates")

        recipe = result.graph.to_recipe(seed=5, output_sample_rate=rate,
                                        output_format="float64")
        rendered = render_array(source, rate, recipe)

        R.check(bool(np.all(np.isfinite(rendered))),
                f"{name}: renders without NaN or Inf")

        verdict = gate.verify(rendered, rate, graph=result.graph,
                              source_report=report)
        R.check(verdict.passed,
                f"{name}: passes the post-render gate — {verdict.summary()}")


if __name__ == "__main__":
    print("Audionova rule-based planner\n")
    test_always_valid()
    test_declines_on_bass()
    test_declines_when_measurements_say_no()
    test_warnings_drive_moves()
    test_master_section()
    test_reverb_is_opt_in()
    test_explanations_are_honest()
    test_renders_and_passes_the_gate()
    print(f"\n{R.checks} checks, {R.failures} failure(s)")
    sys.exit(1 if R.failures else 0)
