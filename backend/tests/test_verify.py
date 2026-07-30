"""The post-render safety gate.

TWO FAILURE MODES, AND THEY PULL IN OPPOSITE DIRECTIONS. A gate that misses a
broken render ships damaged audio to a paying customer. A gate that fails a good
render throws away work the user was happy with and replaces it with something
older, which is arguably worse — the first is a bug they can report, the second
is the tool quietly refusing to do its job.

So this file tests both directions for every check: the thing it must catch, and
the very similar thing it must not. A resonance the processing introduced against
a resonance the source already had; a chain that collapsed the level against one
that was supposed to; a limiter overshoot against a render sitting correctly at
its ceiling.

Most cases are built by rendering through the real engine, because a gate tested
only on hand-written measurement dictionaries would be a gate tested against my
idea of what a render looks like.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.engine.render import DEFAULT_BINARY, Recipe, render_array  # noqa: E402
from backend.measure import report as report_mod  # noqa: E402
from backend.schemas.graph import EffectGraph  # noqa: E402
from backend.verify import gate  # noqa: E402

SR = 48000.0
RNG = np.random.default_rng(90210)


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


def pink(seconds: float = 6.0, channels: int = 2, amplitude: float = 0.3) -> np.ndarray:
    n = int(SR * seconds)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    shape = np.ones_like(freqs)
    shape[1:] = 1.0 / np.sqrt(freqs[1:])
    shape[0] = 0.0
    spec = RNG.normal(size=(channels, freqs.size)) + 1j * RNG.normal(size=(channels, freqs.size))
    out = np.fft.irfft(spec * shape, n=n, axis=-1)
    return out / np.max(np.abs(out)) * amplitude


def graph_from(nodes=(), master=()) -> EffectGraph:
    return EffectGraph.model_validate({
        "schema_version": "effect_graph_v1",
        "nodes": list(nodes), "master": list(master),
    })


def node(node_id: str, node_type: str, **params) -> dict:
    return {"id": node_id, "type": node_type, "params": params}


# ---------------------------------------------------------------------------


def test_passes_a_good_render() -> None:
    print("[1] a good render passes")

    if not os.path.exists(DEFAULT_BINARY):
        print("  engine binary missing; skipping")
        return

    source = pink()
    source_report = report_mod.measure(source, SR).to_dict()

    recipe = Recipe(output_sample_rate=SR, output_format="float64")
    recipe.add("parametric_eq", bands=[
        {"filter_type": "highshelf", "center_freq_hz": 6000.0,
         "gain_db": 3.0, "q_factor": 0.707}])
    recipe.add_master("true_peak_limiter", ceiling_dbtp=-1.0, lookahead_ms=5.0)
    rendered = render_array(source, SR, recipe)

    graph = graph_from(
        [node("eq", "parametric_eq", bands=[
            {"filter_type": "highshelf", "center_freq_hz": 6000.0,
             "gain_db": 3.0, "q_factor": 0.707}])],
        [node("lim", "true_peak_limiter", ceiling_dbtp=-1.0)])

    result = gate.verify(rendered, SR, graph=graph, source_report=source_report)

    R.check(result.passed, f"an ordinary render passes: {result.summary()}")
    R.check(len(result.checked) >= 6,
            f"and enough checks actually ran, {result.checked}")

    # A passing gate must be distinguishable from a gate that skipped
    # everything, which is why the checks that ran are recorded.
    R.check("true_peak" in result.checked and "resonance" in result.checked,
            f"including the substantive ones, {result.checked}")


def test_catches_corruption() -> None:
    print("[2] corruption")

    # These two run whatever else is missing, because they are the ones that
    # catch genuine damage rather than a debatable choice.
    broken = pink(2.0)
    broken[0, 1000] = np.nan
    result = gate.verify(broken, SR)
    R.check(not result.passed and any(f.code == "non_finite" for f in result.failures),
            f"NaN samples fail the gate, got {result.summary()}")

    silent = gate.verify(np.zeros((2, int(SR * 2))), SR)
    R.check(not silent.passed and any(f.code == "silent" for f in silent.failures),
            f"a silent render fails, got {silent.summary()}")

    # Both must fail with NO graph and NO source report. A gate that only works
    # when fully informed is not a safety net.
    R.check(not gate.verify(broken, SR, graph=None, source_report=None).passed,
            "corruption is caught without a graph or a source report")

    # And every failure carries something the user can be told. A correlation
    # coefficient is not an explanation to a musician.
    for failure in result.failures + silent.failures:
        R.check(bool(failure.user_message) and "audio" in failure.user_message,
                f"{failure.code} has a user-facing message")


def test_true_peak_ceiling() -> None:
    print("[3] true peak against the ceiling")

    if not os.path.exists(DEFAULT_BINARY):
        print("  engine binary missing; skipping")
        return

    source = pink(4.0, amplitude=0.9)

    # A render that honours its ceiling passes.
    honest = Recipe(output_sample_rate=SR, output_format="float64")
    honest.add_master("true_peak_limiter", ceiling_dbtp=-1.0, lookahead_ms=5.0,
                      oversample_factor=8)
    limited = render_array(source, SR, honest)
    graph = graph_from(master=[node("lim", "true_peak_limiter", ceiling_dbtp=-1.0)])

    passing = gate.verify(limited, SR, graph=graph)
    R.check(passing.passed, f"a correctly limited render passes: {passing.summary()}")

    # THE ONE THAT MATTERS. A graph that claims a -1 dBTP ceiling, with audio
    # that does not honour it — which is what a limiter placed before the output
    # conversion produces, and exactly why the master section exists.
    hot = source * 1.4
    failing = gate.verify(hot, SR, graph=graph)
    R.check(not failing.passed and any(f.code == "true_peak" for f in failing.failures),
            f"audio above the claimed ceiling fails, got {failing.summary()}")

    # Without a limiter in the graph the bar is 0 dBTP: anything above clips on
    # conversion whatever the graph intended.
    R.check(not gate.verify(hot, SR, graph=graph_from()).passed,
            "and with no limiter at all, 0 dBTP is still enforced")
    R.check(gate.verify(source * 0.5, SR, graph=graph_from()).passed,
            "while ordinary headroom passes with no limiter")


def test_loudness_target() -> None:
    print("[4] loudness target")

    if not os.path.exists(DEFAULT_BINARY):
        print("  engine binary missing; skipping")
        return

    source = pink(8.0)
    measured = report_mod.measure(source, SR).to_dict()["loudness"]["integrated_lufs"]

    on_target = gate.verify(source, SR, target_lufs=measured)
    R.check(on_target.passed, f"a render at its target passes: {on_target.summary()}")

    off = gate.verify(source, SR, target_lufs=measured + 3.0)
    R.check(not off.passed and any(f.code == "loudness_target" for f in off.failures),
            f"3 LU off target fails, got {off.summary()}")

    # Inside the tolerance passes. LUFS is a gated average and a full render
    # whose tail differs from the excerpt it was planned on moves by a few
    # tenths for entirely correct reasons.
    R.check(gate.verify(source, SR, target_lufs=measured + 0.3).passed,
            "0.3 LU off target passes, since that is normal variation")

    # No target means no check, rather than a guessed one.
    R.check("loudness_target" not in gate.verify(source, SR).checked,
            "the check is skipped when no target was given")


def test_mono_compatibility() -> None:
    print("[5] mono compatibility")

    mono_source = pink(4.0, channels=1)[0]

    # A polarity-inverted pair: correlation -1, and the mono sum cancels
    # completely. Every other measurement of this file looks perfectly healthy,
    # which is exactly why it needs its own check.
    inverted = np.vstack([mono_source, -mono_source])
    result = gate.verify(inverted, SR)
    R.check(not result.passed and any(f.code == "out_of_phase" for f in result.failures),
            f"an inverted channel fails, got {result.summary()}")

    # Genuinely wide but mono-compatible material must pass. Two independent
    # pink sources lose the arithmetic 3 dB on summing and nothing more, and
    # failing that would reject every normally-wide mix.
    independent = np.vstack([pink(4.0, channels=1)[0], pink(4.0, channels=1)[0]])
    wide = gate.verify(independent, SR)
    R.check(wide.passed, f"independent channels pass: {wide.summary()}")

    # Dual mono passes: identical channels are perfectly mono-compatible.
    R.check(gate.verify(np.vstack([mono_source, mono_source]), SR).passed,
            "dual mono passes")

    # A mono render is not checked for stereo compatibility at all.
    R.check("mono_compatibility" not in gate.verify(mono_source[None, :], SR).checked,
            "a mono render skips the stereo check")


def test_resonance() -> None:
    print("[6] a resonance the processing introduced")

    if not os.path.exists(DEFAULT_BINARY):
        print("  engine binary missing; skipping")
        return

    source = pink(8.0)
    source_report = report_mod.measure(source, SR).to_dict()

    # A filter oscillating: a very high Q with a large boost, which is what a
    # graph produces when two moves land on the same frequency or a validator
    # bound is escaped.
    ringing = Recipe(output_sample_rate=SR, output_format="float64")
    ringing.add("parametric_eq", bands=[
        {"filter_type": "bell", "center_freq_hz": 1000.0,
         "gain_db": 18.0, "q_factor": 10.0}])
    rendered = render_array(source, SR, ringing)

    result = gate.verify(rendered, SR, source_report=source_report)
    R.check(not result.passed and any(f.code == "resonance" for f in result.failures),
            f"an introduced resonance fails, got {result.summary()}")

    # A DELIBERATE, MODERATE BOOST MUST NOT FAIL. This is the false-positive
    # direction, and it matters more than the catch: throwing away a render
    # because the planner made a reasonable tonal choice would be the tool
    # refusing to do its job.
    intended = Recipe(output_sample_rate=SR, output_format="float64")
    intended.add("parametric_eq", bands=[
        {"filter_type": "bell", "center_freq_hz": 1000.0,
         "gain_db": 6.0, "q_factor": 1.4}])
    moderate = gate.verify(render_array(source, SR, intended), SR,
                          source_report=source_report)
    R.check(moderate.passed,
            f"a deliberate 6 dB bell passes: {moderate.summary()}")

    # AND A RESONANCE THE SOURCE ALREADY HAD MUST NOT FAIL EITHER. Plenty of
    # instruments have one — a snare, a resonant guitar body — and failing a
    # render for a property of the recording would throw away good work.
    resonant_source = render_array(source, SR, ringing)
    resonant_report = report_mod.measure(resonant_source, SR).to_dict()
    passthrough = render_array(resonant_source, SR,
                               Recipe(output_sample_rate=SR, output_format="float64"))
    inherited = gate.verify(passthrough, SR, source_report=resonant_report)
    R.check(inherited.passed,
            f"a resonance inherited from the source passes: {inherited.summary()}")


def test_level_collapse() -> None:
    print("[7] an unexplained collapse in level")

    if not os.path.exists(DEFAULT_BINARY):
        print("  engine binary missing; skipping")
        return

    source = pink(8.0)
    source_report = report_mod.measure(source, SR).to_dict()

    # A big drop that the graph does NOT account for.
    quiet = source * (10 ** (-20.0 / 20.0))
    result = gate.verify(quiet, SR, graph=graph_from(), source_report=source_report)
    R.check(not result.passed and any(f.code == "level_collapse" for f in result.failures),
            f"an unexplained 20 dB drop fails, got {result.summary()}")

    # The same drop, asked for. A chain with -20 dB of gain in it is supposed to
    # come out quieter, and failing that would be the gate second-guessing the
    # user's own instruction.
    explained = gate.verify(quiet, SR,
                            graph=graph_from([node("g", "gain", gain_db=-20.0)]),
                            source_report=source_report)
    R.check(explained.passed,
            f"the same drop passes when the graph asked for it: {explained.summary()}")

    # Without a source report the check cannot run and is skipped rather than
    # guessed at.
    R.check("level_collapse" not in gate.verify(quiet, SR, graph=graph_from()).checked,
            "the check is skipped with no source report")


def test_bandwidth_collapse() -> None:
    print("[8] destroyed bandwidth")

    if not os.path.exists(DEFAULT_BINARY):
        print("  engine binary missing; skipping")
        return

    source = pink(8.0)
    source_report = report_mod.measure(source, SR).to_dict()

    from scipy import signal as _signal

    destroyed = _signal.sosfilt(
        _signal.butter(8, 900.0, btype="low", fs=SR, output="sos"), source)
    result = gate.verify(destroyed, SR, source_report=source_report)
    R.check(not result.passed
            and any(f.code == "bandwidth_collapse" for f in result.failures),
            f"a render lowpassed to 900 Hz fails, got {result.summary()}")

    # A chain is often SUPPOSED to darken a source, so an ordinary high shelf
    # cut must pass. The threshold is set where an engineer would say the source
    # had been destroyed, not where they would have made a different choice.
    darker = Recipe(output_sample_rate=SR, output_format="float64")
    darker.add("parametric_eq", bands=[
        {"filter_type": "highshelf", "center_freq_hz": 5000.0,
         "gain_db": -6.0, "q_factor": 0.707}])
    R.check(gate.verify(render_array(source, SR, darker), SR,
                        source_report=source_report).passed,
            "a deliberate 6 dB high-shelf cut passes")


def test_fallback() -> None:
    print("[9] falling back")

    good = pink(3.0)
    broken = pink(3.0)
    broken[0, 500] = np.inf

    shipped, result = gate.verify_or_fall_back(good, SR)
    R.check(result.passed and shipped is good, "a passing render is shipped")

    shipped, result = gate.verify_or_fall_back(broken, SR, previous_render=good)
    R.check(not result.passed and shipped is good,
            "a failing render is replaced by the previous one")

    # NOTHING TO FALL BACK TO MUST RETURN NOTHING, not silence and not the
    # broken render. A gate that quietly returns something is worse than one
    # that fails loudly, because the caller cannot tell the difference.
    shipped, result = gate.verify_or_fall_back(broken, SR, previous_render=None)
    R.check(not result.passed and shipped is None,
            "with no fallback available, nothing is returned")


def test_thresholds_are_not_trigger_happy() -> None:
    print("[10] ordinary renders are not rejected")

    if not os.path.exists(DEFAULT_BINARY):
        print("  engine binary missing; skipping")
        return

    # A false failure throws away work the user was happy with and replaces it
    # with something older. That is arguably worse than a miss — a miss is a bug
    # they can report, this is the tool refusing to work. So a range of entirely
    # ordinary chains is run through the gate and must all pass.
    chains = [
        ("gentle EQ", [{"type": "parametric_eq", "params": {"bands": [
            {"filter_type": "highpass", "center_freq_hz": 80.0,
             "gain_db": 0.0, "q_factor": 0.707},
            {"filter_type": "bell", "center_freq_hz": 400.0,
             "gain_db": -3.0, "q_factor": 1.2}]}}]),
        ("compression", [{"type": "compressor", "params": {
            "threshold_dbfs": -24.0, "ratio_to_1": 3.0, "attack_ms": 12.0,
            "release_ms": 140.0, "makeup_gain_db": 3.0}}]),
        ("saturation", [{"type": "drive", "params": {
            "drive_db": 8.0, "output_trim_db": -4.0}}]),
        ("space", [{"type": "plate_reverb", "mix": 0.2, "params": {
            "decay_s": 1.6, "pre_delay_ms": 20.0}}]),
        ("width", [{"type": "stereo_width", "params": {"width_percent": 130.0}}]),
    ]

    # REALISTIC STEREO, not two independent channels. Widening fully-decorrelated
    # material to 130% drives correlation to -0.256 by arithmetic, and the gate
    # is right to fail that — it genuinely does cancel on mono. But no recording
    # is fully decorrelated: the reference stems measure 0.70 and 0.86. Testing
    # "ordinary renders are not rejected" against the most extreme stereo field
    # possible was testing the wrong thing.
    mid = pink(8.0, channels=1)[0]
    side = pink(8.0, channels=1)[0] * 0.4
    source = np.vstack([mid + side, mid - side])
    source_report = report_mod.measure(source, SR).to_dict()

    for name, nodes in chains:
        recipe = Recipe(output_sample_rate=SR, output_format="float64")
        for entry in nodes:
            recipe.nodes.append(entry)
        recipe.add_master("true_peak_limiter", ceiling_dbtp=-1.0, lookahead_ms=5.0)

        rendered = render_array(source, SR, recipe)
        graph = graph_from(
            [dict(entry, id=f"n{i}") for i, entry in enumerate(nodes)],
            [node("lim", "true_peak_limiter", ceiling_dbtp=-1.0)])

        result = gate.verify(rendered, SR, graph=graph, source_report=source_report)
        R.check(result.passed, f"{name} passes the gate: {result.summary()}")


if __name__ == "__main__":
    print("Audionova post-render verify gate\n")
    test_passes_a_good_render()
    test_catches_corruption()
    test_true_peak_ceiling()
    test_loudness_target()
    test_mono_compatibility()
    test_resonance()
    test_level_collapse()
    test_bandwidth_collapse()
    test_fallback()
    test_thresholds_are_not_trigger_happy()
    print(f"\n{R.checks} checks, {R.failures} failure(s)")
    sys.exit(1 if R.failures else 0)
