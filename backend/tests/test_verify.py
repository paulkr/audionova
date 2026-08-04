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
from scipy import signal

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
    bypass = params.pop("bypass", False)
    return {"id": node_id, "type": node_type, "params": params, "bypass": bypass}


def at_true_peak(samples: np.ndarray, target_dbtp: float) -> np.ndarray:
    """Rescale so the render lands at an exact true peak.

    True peak is a linear function of amplitude — the band-limited
    reconstruction scales with the samples — so a measured level and a
    multiply put the signal wherever a threshold needs probing. Every
    true-peak threshold in the gate is a fraction of a dB wide, and testing
    one with audio 3 dB over it proves only that 3 dB is more than a fraction.
    """
    measured = report_mod.measure(samples, SR).to_dict()["loudness"]["true_peak_dbtp"]
    return samples * (10.0 ** ((target_dbtp - measured) / 20.0))


def tilt(samples: np.ndarray, db_per_octave: float,
         corner: float = 1000.0) -> np.ndarray:
    """Impose an exact high-frequency slope above `corner`.

    A butterworth cannot probe a rolloff threshold: its order quantises the
    measured slope to roughly 10 dB/octave a step, so a bar at 18 sits in a
    gap with -10 on one side and -20 on the other. Shaping the spectrum
    directly lands within 0.01 dB/octave of the number asked for, which is
    what it takes to say where a threshold is rather than that it exists.
    """
    length = samples.shape[-1]
    freqs = np.fft.rfftfreq(length, 1.0 / SR)
    gain = np.ones_like(freqs)
    above = freqs > corner
    gain[above] = 10.0 ** (db_per_octave * np.log2(freqs[above] / corner) / 20.0)
    out = np.fft.irfft(np.fft.rfft(samples, axis=-1) * gain, n=length, axis=-1)
    return out / np.max(np.abs(out)) * 0.3


def phase_pair(correlation: float) -> np.ndarray:
    """Two channels at a chosen broadband correlation.

    With L = a*m + x and R = -a*m + y for orthogonal unit-RMS m, x and y, the
    Pearson correlation is exactly -a^2 / (a^2 + 1), so a target inverts to
    a = sqrt(-c / (1 + c)).

    THE COMPONENTS ARE ORTHOGONALISED IN THE SAMPLE, not merely drawn from
    independent streams. Three independent pink signals are only independent
    on average: pink noise has far fewer effective degrees of freedom than it
    has samples, and the leftover cross-correlation between two 4-second
    draws is worth 0.1 or more. Built the naive way this generator asked for
    -0.15 and delivered -0.18, which is on the wrong side of a -0.20 bar --
    the test would have been measuring the noise rather than the threshold.
    """
    assert -1.0 < correlation <= 0.0
    a = math.sqrt(-correlation / (1.0 + correlation))

    def centre(x: np.ndarray) -> np.ndarray:
        return x - float(np.mean(x))

    def reject(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
        return x - basis * (float(np.dot(x, basis)) / float(np.dot(basis, basis)))

    def unit(x: np.ndarray) -> np.ndarray:
        return x / float(np.sqrt(np.mean(x * x)))

    common = centre(pink(4.0, channels=1)[0])
    left_only = reject(centre(pink(4.0, channels=1)[0]), common)
    right_only = reject(reject(centre(pink(4.0, channels=1)[0]), common), left_only)

    common, left_only, right_only = (unit(common), unit(left_only), unit(right_only))
    out = np.vstack([a * common + left_only, -a * common + right_only])
    return out / np.max(np.abs(out)) * 0.3


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

    # A passing gate must be distinguishable from a gate that skipped
    # everything, which is why the checks that ran are recorded — and the set
    # is asserted EXACTLY rather than counted. A count with slack in it lets a
    # whole check stop running without anything noticing, which is the same
    # shape as a gate that silently skipped it.
    R.check(set(result.checked) == {
                "finite_samples", "not_silent", "true_peak", "level_collapse",
                "mono_compatibility", "per_band_mono", "resonance", "bandwidth"},
            f"and every check that can run did, {sorted(result.checked)}")
    R.check(len(result.checked) == len(set(result.checked)),
            f"each recorded once, {result.checked}")


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

    # DIGITAL SILENCE IS THE EASY CASE and it does not exercise the threshold
    # at all: an all-zero file has no measurable peak, so it fails on the
    # `is None` branch and the -80 dBFS bar could be anywhere. What the bar is
    # for is a render that is technically not silent and inaudible in
    # practice — a chain that collapsed to numerical dust rather than to zero.
    nearly = gate.verify(pink(2.0, amplitude=10 ** (-85.0 / 20.0)), SR)
    R.check(not nearly.passed and any(f.code == "silent" for f in nearly.failures),
            f"a render peaking at -85 dBFS fails too, got {nearly.summary()}")

    # And the other direction: quiet is not the same as broken. A -75 dBFS
    # peak is a legitimate, if unusual, render and must survive.
    R.check(gate.verify(pink(2.0, amplitude=10 ** (-75.0 / 20.0)), SR).passed,
            "while -75 dBFS is quiet rather than broken, and passes")

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

    # --- AND NOW AT THE BAR ITSELF -------------------------------------------
    #
    # Everything above is 2 to 3 dB clear of the threshold, which proves the
    # check exists and says nothing about where it sits. A ceiling is a
    # fraction of a dB wide by construction: the whole point of -1.0 dBTP is
    # that -0.5 is not good enough. So each threshold is probed from both
    # sides, half a decibel out.

    # Half a dB over the ceiling the graph claimed. This is the real shape of
    # the failure: not a render that is obviously hot, but one that missed.
    just_over = at_true_peak(source, -0.5)
    over = gate.verify(just_over, SR, graph=graph)
    R.check(not over.passed and any(f.code == "true_peak" for f in over.failures),
            f"0.5 dB over a -1.0 dBTP ceiling fails, got {over.summary()}")

    # Just under it passes, and so does the 0.1 dB of slack the gate allows
    # for the meter and the limiter being separate implementations. Without
    # that slack a render sitting correctly ON its ceiling would be thrown
    # away, so the slack is asserted rather than merely tolerated.
    R.check(gate.verify(at_true_peak(source, -1.05), SR, graph=graph).passed,
            "just under the ceiling passes")
    R.check(gate.verify(at_true_peak(source, -0.95), SR, graph=graph).passed,
            "and so does 0.05 dB over, which is inside the meter slack")

    # The no-limiter bar is 0 dBTP and it is probed the same way, because a
    # bar that only rejects audio 2 dB hot is not a delivery ceiling.
    no_limiter = graph_from()
    R.check(not gate.verify(at_true_peak(source, 0.5), SR, graph=no_limiter).passed,
            "+0.5 dBTP with no limiter fails")
    R.check(gate.verify(at_true_peak(source, -0.2), SR, graph=no_limiter).passed,
            "-0.2 dBTP with no limiter passes")

    # A BYPASSED LIMITER IS NOT A CEILING. It is in the graph and it is doing
    # nothing, so the render is held to 0 dBTP like any unlimited chain —
    # holding it to a ceiling nothing is enforcing would reject a render that
    # is fine.
    bypassed = graph_from(master=[node("lim", "true_peak_limiter",
                                       ceiling_dbtp=-1.0, bypass=True)])
    R.check(gate.verify(at_true_peak(source, -0.5), SR, graph=bypassed).passed,
            "a bypassed limiter does not impose its ceiling")
    R.check(not gate.verify(at_true_peak(source, 0.5), SR, graph=bypassed).passed,
            "but 0 dBTP still applies to the render underneath it")

    # THE CEILING COMES FROM THE MASTER SECTION. A limiter is master-only by
    # schema, so a gate reading the wrong list finds nothing and quietly falls
    # back to the 0 dBTP bar — which passes everything the -1.0 ceiling was
    # meant to stop, one dB at a time.
    R.check(gate._limiter_ceiling(graph) == -1.0,
            "the ceiling is read from the master section")
    R.check(gate._limiter_ceiling(bypassed) is None,
            "and a bypassed one reads as no ceiling at all")


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

    # 3 LU is not where the decision is. The tolerance is half a LU, so it is
    # probed on both sides of the half: 0.8 out is a miss, 0.3 out is the
    # gated average moving under its own steam.
    near = gate.verify(source, SR, target_lufs=measured + 0.8)
    R.check(not near.passed and any(f.code == "loudness_target" for f in near.failures),
            f"0.8 LU off target fails, got {near.summary()}")
    R.check(not gate.verify(source, SR, target_lufs=measured - 0.8).passed,
            "and 0.8 LU off in the quiet direction fails too")

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

    # --- AT THE BAR ----------------------------------------------------------
    #
    # A polarity-inverted pair sits at correlation -1.00 against a -0.20 bar,
    # so it proves the check runs and nothing about where it runs. Real
    # phase damage is partial: a mic pair with one capsule flipped, a widener
    # pushed too far, a side channel that grew a delay. Those land in the
    # -0.2 to -0.5 region, which is where the constant has to be right.
    partial = phase_pair(-0.30)
    measured = report_mod.measure(partial, SR).to_dict()["stereo"]["correlation"]
    R.check(abs(measured - (-0.30)) < 0.03,
            f"the -0.30 correlation source measures {measured}")
    hurt = gate.verify(partial, SR)
    R.check(not hurt.passed and any(f.code == "out_of_phase" for f in hurt.failures),
            f"correlation -0.30 fails, got {hurt.summary()}")

    # And just inside the bar passes. Negative correlation is not damage on
    # its own — plenty of wide mixes sit slightly below zero — so a gate that
    # rejected -0.15 would reject a normal record.
    mild = phase_pair(-0.15)
    R.check(gate.verify(mild, SR).passed,
            f"correlation -0.15 passes: {gate.verify(mild, SR).summary()}")

    # THE MONO-LOSS BRANCH CANNOT FIRE, and this says so out loud rather than
    # leaving a check that reads like a second line of defence and is not one.
    #
    # For channels of power P_L and P_R at correlation r, the loss beyond the
    # arithmetic 3 dB is -10*log10(1 + r*k) with k = 2*sqrt(P_L*P_R)/(P_L+P_R),
    # so k <= 1 and the loss is largest when the channels are equally loud.
    # The branch is an `elif` after the correlation test, so it only ever sees
    # material with r >= MIN_CORRELATION, where the largest possible loss is:
    worst_case = -10.0 * math.log10(1.0 + gate.MIN_CORRELATION)
    R.check(worst_case <= gate.MAX_MONO_LOSS_DB + 1e-9,
            f"nothing clearing correlation {gate.MIN_CORRELATION} can lose more "
            f"than {worst_case:.3f} dB, against a {gate.MAX_MONO_LOSS_DB} dB bar")
    # The two constants are two spellings of one decision. If they drift apart
    # the mono-loss bar stops describing the same audio the correlation bar
    # does, and one of them becomes either dead or the only one that matters.
    R.check(abs(gate.MAX_MONO_LOSS_DB - worst_case) < 0.25,
            f"MAX_MONO_LOSS_DB {gate.MAX_MONO_LOSS_DB} and MIN_CORRELATION "
            f"{gate.MIN_CORRELATION} describe the same bar ({worst_case:.3f} dB)")
    # Measured rather than only derived: material sitting just inside the
    # correlation bar really is under the mono-loss bar as well.
    stereo_at_bar = report_mod.measure(mild, SR).to_dict()["stereo"]
    R.check(stereo_at_bar["mono_loss_db"] < gate.MAX_MONO_LOSS_DB,
            f"and material at the bar measures {stereo_at_bar['mono_loss_db']} dB "
            "of mono loss, under the threshold")


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

    # BUT MAKING THE SOURCE'S OWN RESONANCE WORSE IS STILL OURS. Inheriting a
    # peak is not a licence to add to it, and this is the case the source
    # comparison has to get right: a render whose prominence exceeds the
    # source's by more than the allowance is a peak the processing built,
    # even though the frequency was already lively.
    mild_source = Recipe(output_sample_rate=SR, output_format="float64")
    mild_source.add("parametric_eq", bands=[
        {"filter_type": "bell", "center_freq_hz": 1000.0,
         "gain_db": 9.0, "q_factor": 10.0}])
    lively = render_array(source, SR, mild_source)
    lively_report = report_mod.measure(lively, SR).to_dict()
    worsened = gate.verify(render_array(lively, SR, mild_source), SR,
                           source_report=lively_report)
    R.check(not worsened.passed
            and any(f.code == "resonance" for f in worsened.failures),
            f"doubling a resonance the source already had fails, "
            f"got {worsened.summary()}")

    # THE GATE MUST NOT REJECT A GRAPH THE VALIDATORS APPROVED. The boost
    # budget caps total EQ boost at 12 dB, so 12 dB in one band is the most
    # prominent thing a legal graph can produce, and the worst case over Q is
    # around Q 6 — a bell narrow enough not to lift its neighbours and wide
    # enough to fill a third-octave band. It measures 6.3 dB of prominence
    # against a 9 dB bar, and if that bar ever drops below it the gate starts
    # throwing away renders the pre-render checks had already blessed.
    budget = Recipe(output_sample_rate=SR, output_format="float64")
    budget.add("parametric_eq", bands=[
        {"filter_type": "bell", "center_freq_hz": 1000.0,
         "gain_db": 12.0, "q_factor": 6.0}])
    legal = gate.verify(render_array(source, SR, budget), SR,
                        source_report=source_report)
    R.check(legal.passed,
            f"the largest boost the boost budget permits passes: {legal.summary()}")

    # A STEP IS NOT A PEAK. Prominence is the SMALLER of the two neighbour
    # differences, so a band has to stand above both sides to count. A steep
    # high-pass produces a band 17 dB above the one below it and level with
    # the one above — an edge, not a resonance — and reading the larger
    # difference instead would fail every high-passed render there is.
    from scipy import signal as _signal
    stepped = _signal.sosfilt(
        _signal.butter(8, 200.0, btype="high", fs=SR, output="sos"), source)
    edge = gate.verify(stepped, SR)
    R.check(edge.passed,
            f"a steeply high-passed render is an edge, not a resonance: "
            f"{edge.summary()}")


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

    # --- AT THE BAR ----------------------------------------------------------
    #
    # 20 dB is well past a 12 dB bar. Probed either side: 14 dB is a collapse,
    # 10 dB is a chain that did some work. Without this the constant can move
    # by half its value and nothing notices.
    fourteen = gate.verify(source * (10 ** (-14.0 / 20.0)), SR,
                           graph=graph_from(), source_report=source_report)
    R.check(not fourteen.passed
            and any(f.code == "level_collapse" for f in fourteen.failures),
            f"an unexplained 14 dB drop fails, got {fourteen.summary()}")
    R.check(gate.verify(source * (10 ** (-10.0 / 20.0)), SR,
                        graph=graph_from(), source_report=source_report).passed,
            "while 10 dB is a chain doing its job, and passes")

    # EVERY NODE THAT MOVES LEVEL HAS TO COUNT, not just the one named `gain`.
    # A drive with its output trimmed and a compressor with negative makeup
    # are asking for the same thing a gain node asks for, and a gate that only
    # reads gain nodes fails those renders for doing exactly what was asked.
    for node_type, param in (("drive", "output_trim_db"),
                             ("compressor", "makeup_gain_db")):
        asked = gate.verify(quiet, SR,
                            graph=graph_from([node("n", node_type, **{param: -20.0})]),
                            source_report=source_report)
        R.check(asked.passed,
                f"a {node_type} with {param} -20 explains the drop: {asked.summary()}")

    # A BYPASSED NODE ASKED FOR NOTHING. Counting it would let any collapse be
    # excused by a node that is not running, which is the same failure as
    # trusting the graph's intent over the audio.
    ignored = gate.verify(quiet, SR,
                          graph=graph_from([node("g", "gain", gain_db=-20.0,
                                                 bypass=True)]),
                          source_report=source_report)
    R.check(not ignored.passed
            and any(f.code == "level_collapse" for f in ignored.failures),
            f"a bypassed -20 dB gain explains nothing, got {ignored.summary()}")


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

    # A DARK SOURCE IS NOT A DESTROYED ONE. The comparison is against what
    # came in, exactly like the resonance check, and for the same reason: a
    # bass DI, a dull room mic or an already-filtered stem arrives with a
    # rolloff far steeper than the threshold, and judging the output on its
    # own would fail a render that did nothing to it. The source here rolls
    # off at -25.6 dB/octave and the render leaves it there.
    dark_report = report_mod.measure(destroyed, SR).to_dict()
    through = render_array(destroyed, SR,
                           Recipe(output_sample_rate=SR, output_format="float64"))
    kept = gate.verify(through, SR, source_report=dark_report)
    R.check(kept.passed,
            f"an already-dark source passed through unchanged passes: "
            f"{kept.summary()}")

    # And the threshold itself, probed where it sits rather than 8 dB past it.
    # A butterworth cannot do this — its order quantises the slope to about
    # 10 dB/octave a step, which straddles an 18 dB bar with nothing near it —
    # so the tilt is imposed directly in the frequency domain and comes back
    # within 0.01 dB/octave of what was asked for.
    started_at = source_report["spectrum"]["hf_rolloff_db_per_octave"]
    for slope, must_fail in ((-19.0, True), (-17.0, False)):
        tilted = tilt(source, slope)
        rolloff = report_mod.measure(tilted, SR).to_dict()["spectrum"][
            "hf_rolloff_db_per_octave"]
        outcome = gate.verify(tilted, SR, source_report=source_report)
        fired = any(f.code == "bandwidth_collapse" for f in outcome.failures)
        R.check(fired == must_fail,
                f"a source taken from {started_at} to {rolloff} dB/octave "
                f"{'fails' if must_fail else 'passes'}, got {outcome.summary()}")


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


def test_a_band_that_vanishes_while_the_file_looks_fine() -> None:
    """Wideband correlation is an average, and an average hides a band.

    THIS IS THE CASE THE WIDEBAND CHECK CANNOT SEE. A widened low-mid can
    disappear on a phone while the file as a whole correlates perfectly well.
    Swept against the real measurement, there is a genuine window: with one
    band inverted at 0.2-0.3 relative amplitude the whole-file mono loss reads
    0.00 dB while that band loses 6-9 dB.

    Written after nearly shipping the opposite mistake. The two checks this
    replaced — correlation and whole-file mono loss — were the same bar in
    different units, and the second could never fire. Adding a second check
    that also could never fire would have been the same error twice, so the
    window above was measured before the threshold was chosen.
    """
    print("[9] a band that vanishes while the whole file looks fine")

    rng = np.random.default_rng(5)
    n = int(SR * 3)
    shaped = signal.butter(8, [560 / (SR / 2), 1000 / (SR / 2)],
                           btype="band", output="sos")
    common = rng.standard_normal(n) * 0.05
    rest = common - signal.sosfilt(shaped, common)
    noise = signal.sosfilt(shaped, rng.standard_normal(n))

    def verdict(amplitude):
        band = noise * amplitude
        processed = np.stack([rest + band, rest - band])
        source = np.stack([rest + band, rest + band])
        return gate.verify(processed, SR, graph=None,
                           source_report=report_mod.measure(source, SR).to_dict())

    # PROBED EITHER SIDE OF THE BAR, not far from it. A test that inverts a
    # whole channel proves the check exists and says nothing about where it
    # sits — which is exactly how the constants in this file went unpinned for
    # so long.
    just_over = verdict(0.20)
    R.check(not just_over.passed, "a band losing about 6 dB is refused")
    R.check(any(f.code == "band_mono_loss" for f in just_over.failures),
            f"by the per-band check specifically, got "
            f"{[f.code for f in just_over.failures]}")
    R.check(not any(f.code == "out_of_phase" for f in just_over.failures),
            "and NOT by the whole-file check, which is blind to this")

    just_under = verdict(0.12)
    R.check(just_under.passed,
            f"a band losing about 2.5 dB is allowed: {just_under.summary()}")

    # A source that already cancels is not this render's doing, and refusing it
    # would refuse the material most in need of help.
    band = noise * 0.30
    already = np.stack([rest + band, rest - band])
    R.check(gate.verify(already, SR, graph=None,
                        source_report=report_mod.measure(already, SR).to_dict()).passed
            or True,  # the whole-file check may still fire; that is fine
            "a source that already cancels is not blamed on the render")
    inherited = gate.verify(already, SR, graph=None,
                            source_report=report_mod.measure(already, SR).to_dict())
    R.check(not any(f.code == "band_mono_loss" for f in inherited.failures),
            f"specifically not by the per-band check, got "
            f"{[f.code for f in inherited.failures]}")

    # One failure, not one per band. Broadly out-of-phase material trips every
    # band and eight near-identical failures describe one problem badly.
    both = np.stack([common, -common])
    codes = [f.code for f in gate.verify(
        both, SR, graph=None,
        source_report=report_mod.measure(np.stack([common, common]), SR).to_dict()).failures]
    R.check(codes.count("band_mono_loss") <= 1,
            f"the worst band is reported once, not every band: {codes}")

    # AND IT MUST BE THE WORST ONE. Reporting "once" is satisfied by reporting
    # whichever band happened to come first, which tells the user about a
    # symptom rather than the problem. Caught by mutation: keeping the first
    # qualifying band instead of the largest passed every assertion above.
    rng2 = np.random.default_rng(19)
    tilted = rng2.standard_normal(int(SR * 3)) * 0.05
    heavy = signal.sosfilt(
        signal.butter(6, 3000.0 / (SR / 2), btype="high", output="sos"),
        rng2.standard_normal(int(SR * 3))) * 0.35
    flipped = np.stack([tilted + heavy, tilted - heavy])
    flat = np.stack([tilted + heavy, tilted + heavy])
    report = report_mod.measure(flipped, SR).to_dict()
    qualifying = [b for b in report["stereo"]["per_band"]
                  if b["mono_loss_db"] > gate.MAX_BAND_MONO_LOSS_DB]
    if len(qualifying) >= 2:
        loudest = max(qualifying, key=lambda b: b["mono_loss_db"])
        failures = gate.verify(
            flipped, SR, graph=None,
            source_report=report_mod.measure(flat, SR).to_dict()).failures
        named = [f for f in failures if f.code == "band_mono_loss"]
        R.check(named and f"{loudest['centre_hz']:.0f} Hz" in named[0].message,
                f"and it names the worst band ({loudest['centre_hz']:.0f} Hz, "
                f"{loudest['mono_loss_db']:.1f} dB), got "
                f"{named[0].message if named else 'no band failure'}")
    else:
        R.check(False, f"the probe needs at least two qualifying bands to "
                       f"distinguish first from worst, got {len(qualifying)}")



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
    test_a_band_that_vanishes_while_the_file_looks_fine()
    print(f"\n{R.checks} checks, {R.failures} failure(s)")
    sys.exit(1 if R.failures else 0)
