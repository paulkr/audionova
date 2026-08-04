"""The blinded listening-test harness.

THIS IS THE SUITE THAT GATES EVERY QUALITY CLAIM. The plan's top risk is
proxy-metric capture, and its stated defence is that no automated metric may
gate CI until it correlates with human MOS at rho > 0.6. That defence needs
human MOS, human MOS needs a blinded panel, and a blinded panel that is not
actually blinded — or not actually loudness-matched — produces numbers that
look like human judgement and are something else. So every property here is
tested from both sides:

  * the mapping is not recoverable from the payload, AND a payload that would
    leak it is caught
  * a matched pair passes, AND a pair 0.15 LU apart is REFUSED
  * 8 of 10 passes the screen, AND 7 of 10 fails it
  * rho is right on an analytically known dataset, AND on a dataset with ties,
    AND it is Spearman rather than Pearson, AND n = 14 is refused where n = 15
    is not

Thresholds are probed from close to the bar rather than from far outside it.
`findings.md` records forty-two of seventy-two mutations surviving the first
Python round and then eighty-one of a hundred and thirty-nine, and in both
rounds the dominant cause was a test that proved a check existed while saying
nothing about where it sat.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import subprocess
import sys
import tempfile
import threading

import numpy as np
import httpx
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.audio import wavio  # noqa: E402
from backend.engine.render import DEFAULT_BINARY  # noqa: E402
from backend.listen import score as listen_score  # noqa: E402
from backend.listen import correlate, match, run_session, score, screen, serve, session  # noqa: E402
from backend.measure import loudness, spectrum  # noqa: E402

SR = 48000.0
RNG = np.random.default_rng(20260804)

# Student's t at 95%, two-sided, written out rather than derived from the
# module under test. Three listeners is two degrees of freedom.
T_95_DF2 = 4.302652729911275
Z_95 = 1.959963984540054


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
        ok = (got is not None and isinstance(got, (int, float))
              and math.isfinite(got) and abs(got - want) <= tol)
        self.check(ok, f"{what}: got {got}, want {want} +/- {tol}")

    def raises(self, expected: type, call, what: str,
               contains: str | None = None) -> None:
        """`contains` matters more than it looks.

        Several refusals in this module are backed up by a SECOND refusal
        further down — a duplicate listener id also produces duplicate trial
        identifiers, for instance. Without checking which message came back, a
        test asserting the first check exists is satisfied by the second one,
        and `findings.md` calls that a property with two implementations, which
        is a property no test can falsify.
        """
        self.checks += 1
        try:
            call()
        except expected as error:
            if contains is not None and contains.lower() not in str(error).lower():
                self.failures += 1
                print(f"\n  FAIL  {what}: raised {expected.__name__} for the "
                      f"wrong reason — {str(error)[:160]!r} does not mention "
                      f"{contains!r}")
            return
        except BaseException as error:  # noqa: BLE001
            self.failures += 1
            print(f"\n  FAIL  {what}: raised {type(error).__name__}: {error}")
            return
        self.failures += 1
        print(f"\n  FAIL  {what}: returned without raising {expected.__name__}")

    def guard(self, call, what: str, default=None):
        """Run something expected to succeed, reporting a raise as a failure.

        Letting an exception escape stops the suite at the first defect and
        reports one bug per round trip. `findings.md` records that as a real
        cost in the API suite, and the fix there was this.
        """
        try:
            return call()
        except BaseException as error:  # noqa: BLE001
            self.checks += 1
            self.failures += 1
            print(f"\n  FAIL  {what}: raised {type(error).__name__}: "
                  f"{str(error)[:200]}")
            return default


R = Results()


def pink(seconds: float = 6.0, channels: int = 1, amplitude: float = 0.25,
         rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or RNG
    n = int(SR * seconds)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    shape = np.ones_like(freqs)
    shape[1:] = 1.0 / np.sqrt(freqs[1:])
    shape[0] = 0.0
    spec = (rng.normal(size=(channels, freqs.size))
            + 1j * rng.normal(size=(channels, freqs.size)))
    out = np.fft.irfft(spec * shape, n=n, axis=-1)
    return out / np.max(np.abs(out)) * amplitude


def write(path: str, samples: np.ndarray, sample_rate: float = SR) -> str:
    match.write_clip(path, samples, sample_rate, 64, is_float=True)
    return path


# Each condition carries an inaudible-to-the-test but unmistakable spectral
# signature, so the audio a trial actually serves can be traced back to a
# condition without consulting the map being tested.
MARKER_HZ = {"alpha": 1000.0, "beta": 5000.0}


def marked(base: np.ndarray, condition: str, level_db: float) -> np.ndarray:
    frames = base.shape[1]
    t = np.arange(frames) / SR
    tone = 0.2 * np.sin(2.0 * np.pi * MARKER_HZ[condition] * t)
    return (base + tone[None, :]) * (10.0 ** (level_db / 20.0))


def marker_of(samples: np.ndarray) -> str:
    """Which condition's marker tone dominates this audio."""
    freqs, power = spectrum.gated_ltas(np.atleast_2d(samples), SR)
    centres, levels = spectrum.third_octave_levels_db(freqs, power, SR)
    scores = {}
    for condition, hz in MARKER_HZ.items():
        index = int(np.argmin(np.abs(centres - hz)))
        scores[condition] = float(levels[index])
    return max(scores, key=lambda name: scores[name])


def comparison_fixture(work: str, count: int = 3) -> list:
    """`count` comparisons, each with a source and two marked conditions.

    The two conditions are deliberately at DIFFERENT levels, so a session that
    failed to match them would be trivially winnable on loudness, and carry
    DIFFERENT marker tones, so the mapping can be checked against the audio
    rather than against the map that claims to describe it.
    """
    comparisons = []
    for index in range(count):
        base = pink(3.0)
        source = write(os.path.join(work, f"src{index}.wav"), base)
        paths = []
        for which, (condition, level) in enumerate(
                (("alpha", 0.0), ("beta", -4.0))):
            paths.append(write(os.path.join(work, f"c{index}-{which}.wav"),
                               marked(base, condition, level)))
        comparisons.append(session.comparison(
            f"item-{index}", f"request number {index}", source,
            (("alpha", paths[0]), ("beta", paths[1]))))
    return comparisons


# --- 1. loudness matching -----------------------------------------------------


def test_loudness_matching() -> None:
    print("[1] loudness matching is verified, not assumed")

    with tempfile.TemporaryDirectory(prefix="anv_listen_") as work:
        base = pink(4.0)
        quiet = write(os.path.join(work, "quiet.wav"), base * 10 ** (-6.0 / 20.0))
        loud = write(os.path.join(work, "loud.wav"), base)

        out = [os.path.join(work, "m0.wav"), os.path.join(work, "m1.wav")]
        group = R.guard(lambda: match.match_group([quiet, loud], out, what="pair"),
                        "a pair 6 dB apart is matched rather than refused")
        if group is None:
            return

        # THE ASSERTION IS ON THE WRITTEN FILES, re-measured here by code that
        # is not the matcher. A matcher that returned its own prediction would
        # pass any check made against its own return value.
        remeasured = []
        for path in out:
            back = wavio.read_wav(path)
            remeasured.append(float(loudness.integrated_lufs(back.samples,
                                                             back.sample_rate)))
        R.near(abs(remeasured[0] - remeasured[1]), 0.0, 0.005,
               "the two written files measure the same loudness")
        R.near(group.clips[0].lufs, remeasured[0], 1e-9,
               "and the matcher reports what an independent read measures")

        # Matched DOWN: the quiet clip is untouched, the loud one comes down by
        # the 6 dB that separated them.
        R.near(group.clips[0].gain_db, 0.0, 1e-9,
               "the quieter clip is left alone")
        R.near(group.clips[1].gain_db, -6.0, 0.02,
               "and the louder clip is attenuated by the gap")
        R.check(group.spread_lu <= 0.005,
                f"measured spread {group.spread_lu:.5f} LU, far inside the "
                f"{match.TOLERANCE_LU} bar")
        R.near(match.TOLERANCE_LU, 0.1, 0.0,
               "and the bar itself is 0.1 LU, pinned as a literal")
        R.near(match.CEILING_DBTP, -1.0, 0.0,
               "with the same -1.0 dBTP ceiling the graph validator inserts")

        # THE WORD LENGTH IS A CHOICE, so it is pinned. Nothing else in this
        # suite can see 16 bits against 24 — the quantisation is 100 dB below
        # anything measured here — which is exactly why the constant needs an
        # assertion of its own rather than a downstream consequence.
        R.check(wavio.read_wav(out[0]).bit_depth == 24,
                "clips are written at 24 bits")
        sixteen = match.match_group(
            [quiet, loud], [os.path.join(work, "b0.wav"), os.path.join(work, "b1.wav")],
            bit_depth=16)
        R.check(wavio.read_wav(sixteen.clips[0].path).bit_depth == 16,
                "and the word length is a parameter, not a constant in disguise")

        # INTERMEDIATES MUST BE FLOAT. A render carrying a boost sits over full
        # scale before the ceiling offset is applied, and a PCM intermediate
        # would clamp it there — putting the clipping inside the stimulus
        # rather than into the offset.
        hot_path = os.path.join(work, "float.wav")
        match.write_clip(hot_path, np.array([[1.5, -1.5, 0.25]]), SR, 64,
                         is_float=True)
        R.near(float(np.max(wavio.read_wav(hot_path).samples)), 1.5, 1e-9,
               "a sample over full scale survives a float intermediate")

        # THE SOLVERS ARE TESTED DIRECTLY. Both refuse a non-finite loudness,
        # and both are backed up by `verify_matched` refusing the same thing
        # after the write — so a solver that substituted a plausible level
        # would still produce a refusal, from the wrong check, with the wrong
        # message, on a session that had already been given a fabricated gain.
        R.raises(match.UnmeasurableLoudness,
                 lambda: match.match_to_quietest([-20.0, float("-inf")]),
                 "match_to_quietest refuses a clip with no loudness",
                 contains="no loudness to match it by")
        R.raises(match.UnmeasurableLoudness,
                 lambda: match.match_to_target(-18.0)([-20.0, float("nan")]),
                 "and so does a fixed-target solver",
                 contains="cannot be matched")
        R.check(match.match_to_quietest([-20.0, -14.0]) == [0.0, -6.0],
                "and the gains a solver returns are the plain difference")


def test_mismatch_is_refused() -> None:
    print("[2] a mismatched pair is refused, and the bar is where it says")

    with tempfile.TemporaryDirectory(prefix="anv_listen_") as work:
        base = pink(4.0)

        def pair(delta_db: float) -> tuple[str, str]:
            a = write(os.path.join(work, f"a{delta_db}.wav"), base)
            b = write(os.path.join(work, f"b{delta_db}.wav"),
                      base * 10 ** (delta_db / 20.0))
            return a, b

        def nothing(lufs):
            """A solver that does not match. A verification that trusts its own
            inputs would wave this through."""
            return [0.0] * len(lufs)

        # PROBED FROM BOTH SIDES AT 1.5x AND 0.5x THE BAR, not from three
        # decibels away. Inside that distance the constant could move by half
        # its value and nothing would notice.
        over = pair(0.15)
        R.raises(match.LoudnessMismatch,
                 lambda: match.match_group(
                     list(over), [os.path.join(work, "o0.wav"),
                                  os.path.join(work, "o1.wav")],
                     gain_solver=nothing),
                 "0.15 LU apart is refused")

        under = pair(0.05)
        group = match.match_group(
            list(under), [os.path.join(work, "u0.wav"), os.path.join(work, "u1.wav")],
            gain_solver=nothing)
        R.near(group.spread_lu, 0.05, 0.005,
               "0.05 LU apart is accepted, and the spread is reported honestly")

        # And the real solver fixes the case the no-op solver failed.
        fixed = match.match_group(
            list(over), [os.path.join(work, "f0.wav"), os.path.join(work, "f1.wav")])
        R.check(fixed.spread_lu <= 0.005,
                f"the default solver takes the same pair to {fixed.spread_lu:.5f} LU")

        # A silent clip has no loudness to match by, and substituting one is
        # how a fabricated number gets into a result.
        silent = write(os.path.join(work, "silent.wav"), np.zeros((1, int(SR * 2))))
        loud = write(os.path.join(work, "loudish.wav"), pink(2.0))
        R.raises(match.UnmeasurableLoudness,
                 lambda: match.match_group(
                     [silent, loud], [os.path.join(work, "s0.wav"),
                                      os.path.join(work, "s1.wav")]),
                 "a silent clip is refused rather than given a plausible level")


def test_ceiling_and_clipping() -> None:
    print("[3] the ceiling is a common offset, and clipping is caught")

    with tempfile.TemporaryDirectory(prefix="anv_listen_") as work:
        base = pink(4.0, amplitude=0.5)
        hot = write(os.path.join(work, "hot.wav"), base * 3.0)   # over full scale
        cool = write(os.path.join(work, "cool.wav"), base * 3.0 * 10 ** (-3.0 / 20.0))

        group = R.guard(lambda: match.match_group(
            [hot, cool], [os.path.join(work, "h0.wav"), os.path.join(work, "h1.wav")]),
            "a source over full scale is brought under the ceiling")
        if group is not None:
            for clip in group.clips:
                R.check(clip.true_peak_dbtp <= match.CEILING_DBTP + 0.05,
                        f"{os.path.basename(clip.path)} sits at "
                        f"{clip.true_peak_dbtp:.2f} dBTP, at or under the ceiling")
            R.check(group.offset_db < -1.0,
                    f"a source over full scale costs a real common offset, "
                    f"{group.offset_db:.2f} dB")
            R.check(group.spread_lu <= 0.005,
                    "and the offset being COMMON leaves the pair matched, "
                    f"{group.spread_lu:.5f} LU")

            # THE REPORTED GAIN HAS TO EXPLAIN THE LEVEL CHANGE, offset
            # included. It is what the key file records and what anyone
            # reconstructing the session reads; a figure that omits the common
            # offset, or carries the wrong sign, describes a render nobody
            # made. Asserted here rather than in the matched-pair case above
            # because that one has an offset of zero, where both mistakes are
            # invisible.
            for source_path, clip in zip([hot, cool], group.clips):
                original = float(loudness.integrated_lufs(
                    wavio.read_wav(source_path).samples, SR))
                R.near(clip.lufs - original, clip.gain_db, 0.01,
                       f"{os.path.basename(clip.path)}: reported gain "
                       f"{clip.gain_db:+.3f} dB accounts for the measured move")
            R.near(group.clips[1].gain_db, group.offset_db, 1e-9,
                   "and the quieter clip moved by exactly the common offset, "
                   "its solver gain being zero")

        # AT A CEILING THAT IS NOT THE DEFAULT, so the invariant is a property
        # of the offset and not of one constant. There is no post-write guard
        # inside `match_group` — the offset makes one unreachable — which is
        # why this is asserted from outside.
        low = R.guard(lambda: match.match_group(
            [hot, cool], [os.path.join(work, "l0.wav"), os.path.join(work, "l1.wav")],
            ceiling_dbtp=-12.0), "a -12 dBTP ceiling is honoured too")
        if low is not None:
            highest = max(clip.true_peak_dbtp for clip in low.clips)
            R.check(-12.05 <= highest <= -11.95,
                    f"the loudest written clip lands ON the -12 dBTP ceiling, "
                    f"at {highest:.3f} — not merely under it, which any amount "
                    "of over-attenuation would also satisfy")
            R.check(low.spread_lu <= 0.005,
                    f"still matched at {low.spread_lu:.5f} LU")

        # THE ONE CASE THAT SEPARATES "MEASURED" FROM "PREDICTED". BS.1770's
        # absolute gate sits at -70 LUFS and is NOT relative to the programme,
        # so attenuating a whole group across it changes which blocks are
        # integrated. Here both clips are pushed to a nominal -73 LUFS: the
        # arithmetic says they are matched, and the written files have no
        # measurable loudness at all. Only a re-measurement can tell.
        quiet_source = pink(6.0)
        level = float(loudness.integrated_lufs(quiet_source, SR))
        near_gate = quiet_source * 10 ** ((-68.0 - level) / 20.0)
        a = write(os.path.join(work, "gate-a.wav"), near_gate)
        b = write(os.path.join(work, "gate-b.wav"), near_gate * 10 ** (5.0 / 20.0))
        R.raises(match.UnmeasurableLoudness,
                 lambda: match.match_group(
                     [a, b],
                     [os.path.join(work, "g0.wav"), os.path.join(work, "g1.wav")],
                     gain_solver=match.match_to_target(-73.0)),
                 "a prediction of -73 LUFS that measures nothing at all is caught")
        kept = match.match_group(
            [a, b], [os.path.join(work, "k0.wav"), os.path.join(work, "k1.wav")])
        R.near(kept.clips[0].lufs, -68.0, 0.2,
               "while the default solver keeps the group above the gate")


def test_group_refusals() -> None:
    print("[4] clips that cannot form an A/B comparison")

    with tempfile.TemporaryDirectory(prefix="anv_listen_") as work:
        base = pink(4.0)
        at_48 = write(os.path.join(work, "r48.wav"), base)
        at_44 = write(os.path.join(work, "r44.wav"), base, 44100.0)
        R.raises(match.ClipGroupError,
                 lambda: match.match_group(
                     [at_48, at_44],
                     [os.path.join(work, "x0.wav"), os.path.join(work, "x1.wav")]),
                 "two sample rates in one comparison is refused")

        # Both sides of the length bar: half a millisecond is a resampler's
        # rounding, two milliseconds is a different excerpt.
        near = write(os.path.join(work, "near.wav"), base[:, :base.shape[1] - 24])
        far = write(os.path.join(work, "far.wav"), base[:, :base.shape[1] - 96])
        ok = R.guard(lambda: match.match_group(
            [at_48, near], [os.path.join(work, "n0.wav"), os.path.join(work, "n1.wav")]),
            "a half-millisecond drift is accepted")
        R.check(ok is not None and ok.clips[0].frames == ok.clips[1].frames,
                "a half-millisecond drift is trimmed to the shorter clip")
        R.raises(match.ClipGroupError,
                 lambda: match.match_group(
                     [at_48, far],
                     [os.path.join(work, "d0.wav"), os.path.join(work, "d1.wav")]),
                 "a two-millisecond drift is a different excerpt and is refused")

        mono = at_48
        stereo = write(os.path.join(work, "stereo.wav"), np.vstack([base[0], base[0]]))
        R.raises(match.ClipGroupError,
                 lambda: match.match_group(
                     [mono, stereo],
                     [os.path.join(work, "m0.wav"), os.path.join(work, "m1.wav")]),
                 "a mono/stereo pair is a width comparison, not the one asked for")


# --- 2. blinding --------------------------------------------------------------


def test_blinding() -> None:
    print("[5] the mapping is not recoverable from the session payload")

    conditions = (("alpha", "/tmp/alpha.wav"), ("beta", "/tmp/beta.wav"))
    comparisons = [session.Comparison(
        item=f"guitar-di-{index:02d}",
        request=f"request {index}",
        source=f"/tmp/source-{index}.wav",
        conditions=conditions) for index in range(6)]

    plan = R.guard(
        lambda: session.plan_session(comparisons, ["L1", "L2", "L3"], seed=11,
                                     repeats=2),
        "a six-comparison session with two repeats can be planned")
    if plan is None:
        return
    payload = plan.public_payload()

    secrets = (["alpha", "beta"]
               + [item.item for item in comparisons]
               + [item.source for item in comparisons]
               + ["/tmp/alpha.wav", "/tmp/beta.wav", "repeat_of", "condition"])
    found = session.leaks(payload, secrets)
    R.check(found == [], f"nothing in the payload names a condition, an item, "
                         f"a file or a repeat; leaked {found}")

    # The key, by contrast, must contain all of it — otherwise the session is
    # blinded to us as well and the answers can never be scored.
    key_leaks = session.leaks(plan.key_payload(), ["alpha", "beta", "guitar-di-00"])
    R.check(key_leaks == ["alpha", "beta", "guitar-di-00"],
            f"the key holds the mapping the payload does not, got {key_leaks}")

    ids = [trial["trial_id"] for trials in payload["listeners"].values()
           for trial in trials]
    clips = [clip for trials in payload["listeners"].values() for trial in trials
             for clip in (trial["clip_a"], trial["clip_b"], trial["reference"])]
    R.check(all(len(value) == 16 and all(c in "0123456789abcdef" for c in value)
                for value in ids + clips),
            "every identifier is 16 opaque hex characters")
    R.check(len(set(clips)) == len(clips),
            "clip identifiers are per trial occurrence, so a repeat cannot be "
            "spotted by its filenames")

    # AN IDENTIFIER MUST NOT BE A FUNCTION OF WHAT IT POINTS AT. Writing the
    # same comparison with its two conditions the other way round must produce
    # a byte-identical payload and an inverted key. This is what fails if a
    # clip id is ever derived from a condition name or a file path.
    swapped = [session.Comparison(item=item.item, request=item.request,
                                  source=item.source,
                                  conditions=(item.conditions[1], item.conditions[0]))
               for item in comparisons]
    other = session.plan_session(swapped, ["L1", "L2", "L3"], seed=11, repeats=2)
    R.check(json.dumps(other.public_payload(), sort_keys=True)
            == json.dumps(payload, sort_keys=True),
            "reordering the conditions changes nothing the listener can see")
    inverted = all(a.condition_a == b.condition_b and a.condition_b == b.condition_a
                   for a, b in zip(plan.trials, other.trials))
    R.check(inverted,
            "while the key's assignment is inverted throughout — so the check "
            "above is not passing because both sides are the same session")

    # No repeat is adjacent to its original, and the two occurrences are
    # independently randomised rather than presented the same way twice.
    for listener in plan.listeners:
        trials = plan.trials_for(listener)
        positions: dict[int, list[int]] = {}
        for trial in trials:
            positions.setdefault(trial.comparison_index, []).append(trial.position)
        gaps = [max(seen) - min(seen) for seen in positions.values() if len(seen) > 1]
        R.check(len(gaps) == 2, f"{listener} sees both repeated pairs twice")
        # The literal, not the constant. A default that only ever appears as
        # itself in a test is a default nothing pins.
        R.check(all(gap >= 3 for gap in gaps),
                f"{listener}'s repeats sit {gaps} apart, at least 3")
    R.check(session.MIN_REPEAT_SEPARATION == 3,
            f"and the default separation is 3, not "
            f"{session.MIN_REPEAT_SEPARATION}")

    # WHICH pairs repeat is a property of the session. If each listener
    # repeated a different pair, their consistency numbers would be measured
    # on different material and could not be compared with each other.
    repeated_by = {listener: frozenset(
        trial.comparison_index for trial in plan.trials_for(listener)
        if trial.repeat_of) for listener in plan.listeners}
    R.check(len(set(repeated_by.values())) == 1,
            f"every listener answers the same repeated pairs, got {repeated_by}")


def test_ab_assignment_is_balanced() -> None:
    print("[6] A/B assignment is randomised per trial, not per session")

    comparisons = [session.Comparison(
        item=f"i{index}", request=f"r{index}", source=f"/s{index}.wav",
        conditions=(("alpha", f"/a{index}.wav"), ("beta", f"/b{index}.wav")))
        for index in range(40)]
    listeners = [f"L{index}" for index in range(10)]
    plan = session.plan_session(comparisons, listeners, seed=99)

    on_a = sum(1 for trial in plan.trials if trial.condition_a == "alpha")
    fraction = on_a / len(plan.trials)
    R.near(fraction, 0.5, 0.05,
           f"alpha lands on side A in {fraction:.3f} of 400 trials")

    for listener in listeners:
        sides = {trial.condition_a for trial in plan.trials_for(listener)}
        R.check(sides == {"alpha", "beta"},
                f"{listener} sees both conditions on side A at some point")

    # PER TRIAL, NOT PER COMPARISON. One bit drawn per comparison and reused
    # for the whole panel would pass the two checks above and would mean every
    # listener heard the same condition on the same side for a given item.
    varies = sum(1 for index in range(40)
                 if len({trial.condition_a for trial in plan.trials
                         if trial.comparison_index == index}) == 2)
    R.check(varies >= 36,
            f"{varies} of 40 comparisons put different conditions on side A "
            "for different listeners")

    # Presentation order is per listener. Two listeners sharing an order would
    # confound condition with fatigue for the whole panel at once.
    orders = {listener: tuple(trial.comparison_index
                              for trial in plan.trials_for(listener))
              for listener in listeners}
    R.check(len(set(orders.values())) == len(listeners),
            "every listener gets a different presentation order")


def test_seeded_and_reproducible() -> None:
    print("[7] randomisation is seeded and reproducible across processes")

    comparisons = [session.Comparison(
        item=f"i{index}", request=f"r{index}", source=f"/s{index}.wav",
        conditions=(("alpha", f"/a{index}.wav"), ("beta", f"/b{index}.wav")))
        for index in range(8)]

    first = session.plan_session(comparisons, ["L1", "L2"], seed=7, repeats=2)
    again = session.plan_session(comparisons, ["L1", "L2"], seed=7, repeats=2)
    R.check(json.dumps(first.key_payload(), sort_keys=True)
            == json.dumps(again.key_payload(), sort_keys=True),
            "the same seed rebuilds the same session, mapping included")

    different = session.plan_session(comparisons, ["L1", "L2"], seed=8, repeats=2)
    R.check(json.dumps(different.public_payload(), sort_keys=True)
            != json.dumps(first.public_payload(), sort_keys=True),
            "a different seed produces a different session")
    R.check(different.session_id != first.session_id,
            "including its identifier")

    # ACROSS PROCESSES, WITH DIFFERENT HASH SEEDS. `hash()` on a string is
    # salted per process, so a session derived from it is reproducible within
    # one run and different in the next — which looks deterministic right up
    # until someone rebuilds a session from its recorded seed.
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    program = (
        "import hashlib, json, sys;"
        f"sys.path.insert(0, {root!r});"
        "from backend.listen import session as s;"
        "c=[s.Comparison(item='i%d'%i, request='r%d'%i, source='/s%d.wav'%i,"
        " conditions=(('alpha','/a%d.wav'%i),('beta','/b%d.wav'%i)))"
        " for i in range(8)];"
        "p=s.plan_session(c, ['L1','L2'], seed=7, repeats=2);"
        "print(hashlib.sha256(json.dumps(p.key_payload(), sort_keys=True)"
        ".encode()).hexdigest())"
    )
    digests = []
    for hash_seed in ("0", "1", "12345"):
        environment = dict(os.environ, PYTHONHASHSEED=hash_seed)
        result = subprocess.run([sys.executable, "-c", program], env=environment,
                                capture_output=True, text=True, timeout=120)
        R.check(result.returncode == 0,
                f"the subprocess plan succeeds under PYTHONHASHSEED={hash_seed}: "
                f"{result.stderr[-300:]}")
        digests.append(result.stdout.strip())

    here = hashlib.sha256(json.dumps(first.key_payload(), sort_keys=True)
                          .encode()).hexdigest()
    R.check(len(set(digests)) == 1 and digests and digests[0] == here,
            f"three processes with three hash seeds rebuild the same session, "
            f"got {sorted(set(digests))} against {here}")


def test_repeat_structure() -> None:
    print("[8] repeat trials")

    comparisons = [session.Comparison(
        item=f"i{index}", request=f"r{index}", source=f"/s{index}.wav",
        conditions=(("alpha", f"/a{index}.wav"), ("beta", f"/b{index}.wav")))
        for index in range(5)]

    plan = session.plan_session(comparisons, ["L1"], seed=3, repeats=2)
    R.check(len(plan.trials) == 7, f"5 comparisons + 2 repeats = 7 trials, "
                                   f"got {len(plan.trials)}")
    repeats = [trial for trial in plan.trials if trial.repeat_of]
    R.check(len(repeats) == 2, f"two of them are marked as repeats in the KEY, "
                               f"got {len(repeats)}")
    for trial in repeats:
        original = plan.trial(trial.repeat_of)
        R.check(original.comparison_index == trial.comparison_index,
                "a repeat repeats the same comparison")
        R.check(original.position < trial.position,
                "and `repeat_of` names the earlier occurrence")

    # THE SEPARATION IS PROBED AT ONE STEP EITHER SIDE OF WHERE IT BINDS. Two
    # comparisons both repeated give four slots; every pair can be two apart
    # and no pair can be three apart.
    tight = comparisons[:2]
    ok = session.plan_session(tight, ["L1"], seed=3, repeats=2,
                              min_repeat_separation=2)
    R.check(len(ok.trials) == 4, "separation 2 is satisfiable on four slots")
    R.raises(session.SessionError,
             lambda: session.plan_session(tight, ["L1"], seed=3, repeats=2,
                                          min_repeat_separation=3),
             "separation 3 is not, and is refused rather than quietly relaxed")

    R.raises(session.SessionError,
             lambda: session.plan_session(comparisons, ["L1"], seed=3, repeats=6),
             "more repeats than comparisons is refused",
             contains="cannot be repeated more than once")
    # THE MESSAGE MATTERS. Two listeners with the same id also derive the same
    # random stream and therefore the same trial identifiers, so the collision
    # check further down refuses this too — and a test that only asserted
    # "something was raised" would be satisfied by the wrong check.
    R.raises(session.SessionError,
             lambda: session.plan_session(comparisons, ["L1", "L1"], seed=3),
             "duplicate listener ids are refused for being duplicates",
             contains="unique")
    R.raises(session.SessionError,
             lambda: session.plan_session([], ["L1"], seed=3),
             "an empty session is refused",
             contains="at least one comparison")
    R.raises(session.SessionError,
             lambda: session.Comparison(item="i", request="r", source="/s.wav",
                                        conditions=(("alpha", "/a.wav"),)),
             "a comparison with one condition is not an A/B trial")


def test_build_session_end_to_end() -> None:
    print("[9] a built session: matched audio, split documents")

    with tempfile.TemporaryDirectory(prefix="anv_listen_") as work:
        comparisons = comparison_fixture(work, count=3)
        out = os.path.join(work, "session")
        built = R.guard(lambda: session.build_session(
            comparisons, ["L1", "L2"], out, seed=5, repeats=1),
            "a three-comparison session builds")
        if built is None:
            return

        for index, group in enumerate(built.groups):
            R.check(group.spread_lu <= match.TOLERANCE_LU,
                    f"comparison {index} matched to {group.spread_lu:.5f} LU")
            R.check(group.spread_lu <= 0.005,
                    f"comparison {index} is well inside the bar, not on it")

        with open(built.session_json) as handle:
            payload = json.load(handle)
        leaked = session.leaks(payload, ["alpha", "beta", "item-0", work])
        R.check(leaked == [], f"the written session.json leaks nothing: {leaked}")

        R.check(os.path.exists(built.key_json), "the key is written")
        R.check(os.path.dirname(built.key_json) == out
                and os.path.dirname(built.clips_dir) == out,
                "key and clips are siblings, and neither is a served root")

        key = session.load_key(built.key_json)
        R.check(set(key["clips"]) == set(built.clip_files),
                "the key resolves every clip identifier in the payload")
        stored = {os.path.basename(path) for path in key["clips"].values()}
        R.check(len(stored) == 3 * len(comparisons),
                f"one written file per (comparison, role), got {len(stored)}")

        # THE SOURCE IS MATCHED TOO. A reference that jumps in level when the
        # listener switches to it biases every judgement made against it.
        for group in built.groups:
            R.check(len(group.clips) == 3,
                    f"the reference is inside the matched group, not beside "
                    f"it; group has {len(group.clips)} clips")
            levels = [clip.lufs for clip in group.clips]
            R.check(max(levels) - min(levels) <= match.TOLERANCE_LU,
                    "the unprocessed reference is matched with the pair")

        # THE MAPPING IS CHECKED AGAINST THE AUDIO, not against the map. Every
        # condition carries its own marker tone, so reading the file a trial
        # will actually serve says which condition it is without consulting
        # the dictionary under test. Swapping the two sides while building the
        # clip map is otherwise a silent, total inversion of every result the
        # panel produces, and nothing else here would notice.
        wrong = 0
        for trial in built.plan.trials:
            for clip_id, expected in ((trial.clip_a, trial.condition_a),
                                      (trial.clip_b, trial.condition_b)):
                served = wavio.read_wav(built.clip_files[clip_id])
                if marker_of(served.samples) != expected:
                    wrong += 1
        R.check(wrong == 0,
                f"every served clip carries the marker of the condition the "
                f"key says it is; {wrong} did not")

        # A comparison whose conditions cannot be matched must stop the whole
        # build before session.json exists.
        broken_dir = os.path.join(work, "broken")

        def nothing(lufs):
            return [0.0] * len(lufs)

        R.raises(match.LoudnessMismatch,
                 lambda: session.build_session(comparisons, ["L1"], broken_dir,
                                               seed=5, gain_solver=nothing),
                 "a session whose audio will not match is refused")
        R.check(not os.path.exists(os.path.join(broken_dir, "session.json")),
                "and no session.json is left behind to be run by mistake")


# --- 3. screening -------------------------------------------------------------


def test_screening_scoring() -> None:
    print("[10] the 80% screen scores correctly on both sides of the bar")

    key = {"answers": {f"t{i}": ("A" if i % 2 else "B") for i in range(10)},
           "pass_rate": screen.PASS_RATE}

    def answers(correct: int) -> dict[str, str]:
        out = {}
        for index, (trial_id, truth) in enumerate(sorted(key["answers"].items())):
            out[trial_id] = truth if index < correct else ("A" if truth == "B" else "B")
        return out

    # ONE TRIAL EITHER SIDE OF THE BAR. A screen tested at 10/10 and 0/10 would
    # pass with the threshold anywhere between.
    eight = screen.score_screening(answers(8), key, "L1")
    R.check(eight.passed and eight.correct == 8,
            f"8 of 10 passes an 80% bar, got {eight.correct} {eight.passed}")
    R.near(eight.score, 0.8, 1e-12, "and scores exactly 0.80")
    seven = screen.score_screening(answers(7), key, "L1")
    R.check(not seven.passed, "7 of 10 does not")
    R.check(len(seven.wrong) == 3, f"and names the three it got wrong, "
                                   f"got {len(seven.wrong)}")

    R.raises(screen.ScreeningError,
             lambda: screen.score_screening({"t0": "A"}, key, "L1"),
             "an unanswered trial is an error, not a wrong answer")
    R.raises(screen.ScreeningError,
             lambda: screen.score_screening({**answers(10), "t99": "A"}, key, "L1"),
             "an answer to a trial not in the key is refused")
    bad = answers(10)
    bad["t0"] = "maybe"
    R.raises(screen.ScreeningError,
             lambda: screen.score_screening(bad, key, "L1"),
             "a 2AFC screen takes A or B and nothing else")

    R.near(screen.binomial_tail(8, 10), 0.0546875, 1e-9,
           "P(>=8 of 10 | guessing) is 7/128")
    R.near(screen.binomial_tail(7, 10), 0.171875, 1e-9,
           "and P(>=7) is 11/64, which is why the bar is 8")

    panel = screen.screen_panel({"L1": answers(8), "L2": answers(5)}, key)
    R.check(panel["L1"].passed and not panel["L2"].passed,
            "a panel screens listener by listener")


def test_screening_stimulus_is_the_real_thing() -> None:
    print("[11] the screening stimulus is a 2 dB Q=1 bell at 3 kHz, from the engine")

    with tempfile.TemporaryDirectory(prefix="anv_screen_") as work:
        source = pink(14.0)
        built = R.guard(lambda: screen.build_screening(
            source, SR, work, seed=17, trials=2, segment_s=6.0),
            "the screening stimulus builds and matches")
        if built is None:
            return
        R.check(len(built.trials) == 2, "two trials were built")

        key = built.key_payload()
        for trial in built.trials:
            bell_id = trial.clip_b if trial.bell_side == "B" else trial.clip_a
            plain_id = trial.clip_a if trial.bell_side == "B" else trial.clip_b
            bell = wavio.read_wav(built.clip_files[bell_id])
            plain = wavio.read_wav(built.clip_files[plain_id])

            centres, delta = screen.bell_delta_db(plain.samples, bell.samples, SR)
            peak = int(np.nanargmax(np.where(np.isfinite(delta), delta, -np.inf)))

            def band(hz: float) -> float:
                return float(delta[int(np.argmin(np.abs(centres - hz)))])

            R.near(float(centres[peak]), 3150.0, 1.0,
                   "the boost lands in the third-octave band containing 3 kHz")

            # 1.88 dB, NOT 2.00, and the difference is the measurement rather
            # than the filter. A third-octave band is a third of an octave
            # wide and a Q=1 bell is already falling across it, so the band
            # integral of a +2 dB peak is a little under +2 dB. Asserted where
            # it measures, with a band tight enough that 1.5 dB or 2.5 dB of
            # boost would both fail.
            R.check(1.80 <= float(delta[peak]) <= 1.95,
                    f"the boost measures {float(delta[peak]):+.3f} dB in its "
                    "own band, which is a 2 dB bell seen through a "
                    "third-octave filter")

            # THE SKIRTS PIN Q. A Q=1 bell is about 1.4 octaves wide, so it is
            # still worth ~0.9 dB two thirds of an octave up and essentially
            # nothing two octaves out. A shelf would hold its gain above the
            # corner and a much lower Q would fill these brackets in.
            R.check(1.60 <= band(2500.0) <= 1.78,
                    f"2.5 kHz reads {band(2500.0):+.3f} dB")
            R.check(0.80 <= band(5000.0) <= 0.96,
                    f"5 kHz reads {band(5000.0):+.3f} dB — a bell, not a shelf")
            for hz in (800.0, 12500.0):
                R.check(abs(band(hz)) < 0.15,
                        f"{hz:.0f} Hz moves {band(hz):+.3f} dB, essentially "
                        "nothing two octaves out")

        # MATCHING THE SCREEN IS LOAD-BEARING, and by a wide margin: the bell
        # alone moves integrated loudness by about 0.58 LU on this material,
        # nearly six times the 0.1 LU bar. Unmatched, a listener with no
        # frequency discrimination at all passes by picking the louder clip.
        for index, group in enumerate(built.groups):
            R.check(group.spread_lu <= match.TOLERANCE_LU,
                    f"screening trial {index} is loudness-matched to "
                    f"{group.spread_lu:.4f} LU")
            R.check(abs(group.clips[1].gain_db - group.clips[0].gain_db) > 0.3,
                    f"and the match had real work to do — the two sides needed "
                    f"{abs(group.clips[1].gain_db - group.clips[0].gain_db):.3f}"
                    " dB of correction between them")

        # BOTH SIDES ARE RENDERED, and the unprocessed side is a NULL RENDER
        # rather than the raw source. The engine's own conversion rolls off
        # above 20 kHz, so a clip that went through it sits below its
        # neighbour at the top where the raw source is flat. Comparing against
        # the raw file would put that rolloff into the difference the listener
        # is being asked to hear.
        first = built.trials[0]
        plain_id = first.clip_a if first.bell_side == "B" else first.clip_b
        plain = wavio.read_wav(built.clip_files[plain_id])
        freqs, power = spectrum.gated_ltas(plain.samples, SR)
        centres, levels = spectrum.third_octave_levels_db(freqs, power, SR)
        top = float(levels[int(np.argmin(np.abs(centres - 20000.0)))]
                    - levels[int(np.argmin(np.abs(centres - 16000.0)))])
        R.check(top < -0.3,
                f"the unprocessed side went through the engine: its 20 kHz "
                f"band sits {top:+.3f} dB under 16 kHz, where the pink source "
                "is flat")

        # DIFFERENT SEGMENTS PER TRIAL. A screen built from one segment
        # measures whether the listener learned one clip.
        second = built.trials[1]
        other_id = second.clip_a if second.bell_side == "B" else second.clip_b
        other = wavio.read_wav(built.clip_files[other_id])
        overlap = float(np.mean(np.abs(plain.samples - other.samples)))
        R.check(overlap > 1e-3,
                f"the two trials use different material, mean |difference| "
                f"{overlap:.5f}")
        R.raises(screen.ScreeningError,
                 lambda: screen.build_screening(pink(4.0), SR, work, seed=1,
                                                trials=4, segment_s=6.0),
                 "a source too short for one segment per trial is refused "
                 "rather than quietly recycling one")

        public = built.public_payload()
        leaked = session.leaks(public, ["bell", "plain", "bell_side", "answers"])
        R.check(leaked == [], f"the screening payload does not say which side "
                              f"has the boost: {leaked}")
        R.check(set(key["answers"]) == {t.trial_id for t in built.trials},
                "and the key does")

        # The key's `bell_side` is not taken on trust anywhere above: the
        # spectral measurement resolves which clip carries the boost, so a key
        # that recorded the wrong side would have produced a NEGATIVE delta and
        # an argmax somewhere else entirely.
        sides = {trial.bell_side for trial in built.trials}
        R.check(sides == {"A", "B"},
                f"the boosted side is drawn per trial, got {sorted(sides)}")

        # The plan's screen, pinned as literals. A bar that can be moved
        # without a test failing is not a bar.
        R.near(screen.FREQ_HZ, 3000.0, 0.0, "the screen is at 3 kHz")
        R.near(screen.GAIN_DB, 2.0, 0.0, "with 2 dB of boost")
        R.near(screen.Q_FACTOR, 1.0, 0.0, "at Q = 1")
        R.near(screen.PASS_RATE, 0.80, 0.0, "and the bar is 80%")


# --- 4. scoring ---------------------------------------------------------------


def _scoring_fixture() -> tuple[dict, list[score.Response]]:
    """Three listeners, two comparisons, one repeated pair, by hand.

    Every expected number below is computed longhand in the test rather than
    read off the implementation.
    """
    def trial(listener, position, comparison, item, a, b, repeat_of=None):
        return {"listener": listener, "position": position,
                "comparison": comparison, "item": item,
                "condition_a": a, "condition_b": b,
                "clip_a": "0" * 16, "clip_b": "1" * 16, "reference": "2" * 16,
                "repeat_of": repeat_of}

    trials = {
        # L1
        "t11": trial("L1", 0, 0, "i1", "alpha", "beta"),
        "t12": trial("L1", 1, 1, "i2", "beta", "alpha"),
        "t1r": trial("L1", 4, 0, "i1", "beta", "alpha", "t11"),
        # L2
        "t21": trial("L2", 0, 0, "i1", "alpha", "beta"),
        "t22": trial("L2", 1, 1, "i2", "alpha", "beta"),
        "t2r": trial("L2", 4, 0, "i1", "alpha", "beta", "t21"),
        # L3
        "t31": trial("L3", 0, 0, "i1", "alpha", "beta"),
        "t32": trial("L3", 1, 1, "i2", "beta", "alpha"),
        "t3r": trial("L3", 4, 0, "i1", "alpha", "beta", "t31"),
    }
    key = {"trials": trials}

    def rate(alpha_quality, beta_quality):
        return {"alpha": {"match": 5, "quality": alpha_quality},
                "beta": {"match": 3, "quality": beta_quality}}

    def response(listener, trial_id, preference, values):
        record = trials[trial_id]
        return {
            "listener": listener, "trial_id": trial_id, "preference": preference,
            "ratings": {"A": values[record["condition_a"]],
                        "B": values[record["condition_b"]]},
        }

    payload = [
        response("L1", "t11", "A", rate(4, 2)),
        response("L1", "t12", "B", rate(4, 2)),
        # Same condition named, other side. The repeat's ratings are chosen so
        # that drift measured PER SIDE (1.0) differs from drift measured PER
        # CONDITION (2.0), which is the only thing separating the two.
        response("L1", "t1r", "B", rate(1, 3)),
        response("L2", "t21", "A", rate(5, 3)),
        response("L2", "t22", "tie", rate(3, 1)),
        response("L2", "t2r", "B", rate(5, 3)),   # a true reversal
        response("L3", "t31", "A", rate(3, 2)),
        response("L3", "t32", "A", rate(3, 2)),
        response("L3", "t3r", "tie", rate(3, 2)),  # a change, not a reversal
    ]
    return key, score.parse_responses(payload)


def test_scoring() -> None:
    print("[12] two scales, listener-clustered intervals")

    key, responses = _scoring_fixture()
    scores = score.score_session(responses, key)

    R.check(scores.scored_trials == 6 and scores.repeat_trials == 3,
            f"repeats are held back from the means, got "
            f"{scores.scored_trials}/{scores.repeat_trials}")

    # THE TWO SCALES ARE REPORTED SEPARATELY, asserted on the shape of the
    # result before anything indexes into it. A harness that folded them into
    # one would otherwise fail here as a KeyError, which stops the suite and
    # reports one defect per round trip.
    expected = {(condition, scale) for condition in ("alpha", "beta")
                for scale in ("match", "quality")}
    R.check(set(scores.conditions) == expected,
            f"one result per (condition, scale), got {sorted(scores.conditions)}")
    if set(scores.conditions) != expected:
        return

    alpha_quality = scores.conditions[("alpha", "quality")]
    # Per listener: L1 (4+4)/2 = 4, L2 (5+3)/2 = 4, L3 (3+3)/2 = 3.
    R.near(alpha_quality.per_listener["L1"], 4.0, 1e-12,
           "L1's alpha quality mean excludes the repeat, which rated it 1")
    R.near(alpha_quality.mean, 11.0 / 3.0, 1e-12,
           "alpha's quality is the mean of the three listener means")
    R.check(alpha_quality.n_listeners == 3 and alpha_quality.n_ratings == 6,
            f"reported over {alpha_quality.n_listeners} listeners and "
            f"{alpha_quality.n_ratings} ratings")

    # The interval, written out longhand: t(0.975, 2) * sd / sqrt(3).
    means = [4.0, 4.0, 3.0]
    centre = sum(means) / 3.0
    sd = math.sqrt(sum((value - centre) ** 2 for value in means) / 2.0)
    half = T_95_DF2 * sd / math.sqrt(3.0)
    R.near(alpha_quality.ci_low, centre - half, 1e-9, "interval low")
    R.near(alpha_quality.ci_high, centre + half, 1e-9, "interval high")
    R.check(alpha_quality.ci_high - alpha_quality.ci_low > 2.8,
            f"and it is WIDE — {alpha_quality.ci_high - alpha_quality.ci_low:.2f} "
            "on a three-person panel, which is the honest width")

    # THE TWO SCALES DISSOCIATE, and are reported apart. alpha is rated 5 for
    # matching the request on every trial while its quality varies; a harness
    # that averaged them would show neither.
    alpha_match = scores.conditions[("alpha", "match")]
    R.near(alpha_match.mean, 5.0, 1e-12, "alpha matches the request at 5.00")
    R.near(alpha_match.ci_low, 5.0, 1e-12,
           "with a zero-width interval, because every listener agreed exactly")
    R.near(scores.conditions[("beta", "quality")].mean, 2.0, 1e-12,
           "beta's quality mean")
    R.near(scores.conditions[("beta", "match")].mean, 3.0, 1e-12,
           "beta's match mean, which is a different number from its quality")

    # A pooled interval would be about sqrt(2) narrower here and would be
    # claiming six independent observations from three people.
    pooled = [4, 4, 5, 3, 3, 3]
    pooled_sd = float(np.std(pooled, ddof=1))
    pooled_half = float(stats.t.ppf(0.975, 5)) * pooled_sd / math.sqrt(6)
    R.check(pooled_half < half * 0.85,
            f"the clustered interval ({half:.3f}) is wider than the pooled one "
            f"({pooled_half:.3f}), which is the point of clustering")

    # ONE LISTENER HAS NO MEASURABLE SPREAD, so there is no interval and none
    # is invented. This is the case where a plausible width would be a
    # statement about an assumption rather than about the data.
    solo_key = {"trials": {tid: record for tid, record in key["trials"].items()
                           if record["listener"] == "L1"}}
    solo = score.score_session(
        [r for r in responses if r.listener == "L1"], solo_key)
    solo_alpha = solo.conditions[("alpha", "quality")]
    R.check(solo_alpha.ci_low is None and solo_alpha.ci_high is None,
            "a one-listener panel gets no interval at all")
    R.check(solo_alpha.n_listeners == 1 and solo_alpha.n_ratings == 2,
            f"while still reporting what it rests on, "
            f"{solo_alpha.n_listeners}/{solo_alpha.n_ratings}")

    tally = scores.preferences["alpha"]
    R.check((tally.wins, tally.ties, tally.losses) == (4, 1, 1),
            f"alpha is preferred 4, tied 1, lost 1; got "
            f"{(tally.wins, tally.ties, tally.losses)}")
    R.near(tally.prefer_or_tie, 5.0 / 6.0, 1e-12,
           "prefer-or-tie is the Phase 0 gate's own criterion")
    R.check(scores.preferences["beta"].wins == 1,
            "and the other condition's tally is its mirror")


def test_inconsistency() -> None:
    print("[13] inconsistency from the repeat trials")

    key, responses = _scoring_fixture()
    consistency = score.consistency_of(responses, key)

    # L1 SAID 'A' THEN 'B' AND WAS CONSISTENT, because the sides were
    # re-randomised and both answers name alpha. A harness comparing the raw
    # preference letters would call this a reversal.
    R.check(consistency["L1"].repeat_pairs == 1
            and consistency["L1"].reversals == 0
            and consistency["L1"].unstable == 0,
            f"L1 is consistent across the repeat: {consistency['L1']}")
    R.check(consistency["L2"].reversals == 1 and consistency["L2"].unstable == 1,
            "L2 named the other condition the second time — a true reversal")
    R.check(consistency["L3"].reversals == 0 and consistency["L3"].unstable == 1,
            "L3 moved to a tie, which is a change and not a contradiction")
    R.near(consistency["L2"].reversal_rate, 1.0, 1e-12, "L2's reversal rate")
    R.near(consistency["L3"].reversal_rate, 0.0, 1e-12, "L3's reversal rate")
    R.near(consistency["L3"].unstable_rate, 1.0, 1e-12, "L3's instability rate")

    # Rating drift is measured PER CONDITION across the two occurrences, so a
    # side flip must not show up as drift. L1 rated alpha 4 then 1 and beta 2
    # then 2, so the mean absolute change is 1.5.
    R.near(consistency["L1"].rating_drift["quality"], 2.0, 1e-12,
           "L1's quality drift over the repeated pair — 2.0 measured per "
           "condition, where per side it would read 1.0")
    R.near(consistency["L1"].rating_drift["match"], 0.0, 1e-12,
           "and no drift at all on the scale they used consistently")
    R.near(consistency["L2"].rating_drift["quality"], 0.0, 1e-12,
           "L2 rated both conditions identically and still reversed their "
           "preference, which is exactly what a repeat trial exists to catch")


def test_response_validation() -> None:
    print("[14] a malformed response is an error, never a coerced one")

    key, _ = _scoring_fixture()

    def one(**overrides):
        base = {"listener": "L1", "trial_id": "t11", "preference": "A",
                "ratings": {"A": {"match": 5, "quality": 4},
                            "B": {"match": 3, "quality": 2}}}
        base.update(overrides)
        return [base]

    R.check(len(score.parse_responses(one())) == 1, "a well-formed response parses")
    R.raises(score.ResponseError,
             lambda: score.parse_responses(one(preference="a")),
             "'a' is not 'A' — no case folding")
    R.raises(score.ResponseError,
             lambda: score.parse_responses(one(preference="either")),
             "an unknown preference is a hard error")
    R.raises(score.ResponseError,
             lambda: score.parse_responses(
                 one(ratings={"A": {"match": 5, "quality": 4}})),
             "a rating for only one side is a hole, not a zero")
    R.raises(score.ResponseError,
             lambda: score.parse_responses(one(ratings={
                 "A": {"match": 5, "quality": 4, "vibe": 3},
                 "B": {"match": 3, "quality": 2, "vibe": 1}})),
             "an unknown scale name is refused rather than dropped")
    R.raises(score.ResponseError,
             lambda: score.parse_responses(one(ratings={
                 "A": {"match": 5, "quality": 6},
                 "B": {"match": 3, "quality": 2}})),
             "a 6 on a 1-5 scale is refused rather than clamped")
    R.raises(score.ResponseError,
             lambda: score.parse_responses(one(ratings={
                 "A": {"match": 5, "quality": 0},
                 "B": {"match": 3, "quality": 2}})),
             "and so is a 0")
    R.check(len(score.parse_responses(one(ratings={
        "A": {"match": 5, "quality": 5}, "B": {"match": 1, "quality": 1}}))) == 1,
        "while the ends of the scale, 1 and 5, are accepted")
    R.raises(score.ResponseError,
             lambda: score.parse_responses(one(ratings={
                 "A": {"match": 5, "quality": "4"},
                 "B": {"match": 3, "quality": 2}})),
             "a string that looks like a number is not one")

    good = score.parse_responses(one())
    R.raises(score.ResponseError,
             lambda: score.score_session(good + good, key),
             "the same trial answered twice is a duplicate, not a repeat")
    wrong_listener = score.parse_responses(one(listener="L2"))
    R.raises(score.ResponseError,
             lambda: score.score_session(wrong_listener, key),
             "answers cannot be pooled across listeners, whose orders differ")
    R.raises(score.ResponseError,
             lambda: score.score_session(score.parse_responses(one(trial_id="nope")),
                                         key),
             "a trial that is not in the key is refused")


# --- 5. correlation -----------------------------------------------------------


def test_spearman_is_right() -> None:
    print("[15] Spearman rho against analytically known datasets")

    # rho = 1 - 6*sum(d^2)/(n*(n^2-1)); swapping the top two of five gives
    # sum(d^2) = 2, so rho = 1 - 12/120 = 0.9 exactly.
    R.near(correlate.spearman_rho([1, 2, 3, 4, 5], [1, 2, 3, 5, 4]), 0.9, 1e-12,
           "one adjacent swap in five points is rho 0.9")
    R.near(correlate.spearman_rho([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]), 1.0, 1e-12,
           "identical orderings are +1")
    R.near(correlate.spearman_rho([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]), -1.0, 1e-12,
           "reversed is -1")

    # SPEARMAN, NOT PEARSON. A metric that tracks quality through a cube is
    # perfect for a gate and mediocre to Pearson.
    x = list(range(1, 9))
    curved = [math.exp(value) for value in x]
    pearson = float(np.corrcoef(x, curved)[0, 1])
    R.near(correlate.spearman_rho(x, curved), 1.0, 1e-12,
           "a monotone nonlinear relation is rho 1.0")
    R.check(pearson < 0.85,
            f"while Pearson on the same data is {pearson:.4f}, so this is not "
            "Pearson wearing another name")

    # TIES ARE AVERAGED. x = [1,1,2,3] ranks as [1.5,1.5,3,4]; against
    # y = [1,2,3,4] the Pearson of the ranks is 0.9486832980505138.
    ties = correlate.spearman_rho([1, 1, 2, 3], [1, 2, 3, 4])
    R.near(ties, 0.9486832980505138, 1e-12, "ties take the average rank")
    R.check(abs(ties - 1.0) > 1e-6,
            "a tie the metric could not resolve is not scored as a perfect call")

    R.check(np.allclose(correlate.ranks([10.0, 10.0, 10.0]), [2.0, 2.0, 2.0]),
            "three-way ties all take the middle rank")

    # An independent implementation, on data with ties and no analytic answer.
    rng = np.random.default_rng(4242)
    a = rng.integers(0, 6, size=40).astype(float)
    b = a * 2.0 + rng.normal(size=40)
    R.near(correlate.spearman_rho(a, b),
           float(stats.spearmanr(a, b).statistic), 1e-12,
           "and it agrees with scipy, which shares no code with it")

    # A CONSTANT SERIES EXPRESSED NO ORDERING. Reporting 0.0 would read as
    # "measured, no relationship", which is a much more reassuring statement.
    R.check(math.isnan(correlate.spearman_rho([1, 2, 3, 4], [7, 7, 7, 7])),
            "a metric with one value for every item gives NaN, not zero")
    R.raises(correlate.CorrelationError,
             lambda: correlate.spearman_rho([1, 2, 3], [1, 2]),
             "unequal series are refused")
    R.raises(correlate.CorrelationError,
             lambda: correlate.ranks([1.0, float("nan")]),
             "a missing measurement has no rank")


def test_rho_interval_and_usability() -> None:
    print("[16] the interval, and the refusal to print a bare rho")

    # Bonett-Wright, written out longhand: se = sqrt((1 + rho^2/2)/(n-3)).
    rho, n = 0.5, 20
    se = math.sqrt((1.0 + rho * rho / 2.0) / (n - 3))
    expected = (math.tanh(math.atanh(rho) - Z_95 * se),
                math.tanh(math.atanh(rho) + Z_95 * se))
    low, high = correlate.rho_interval(rho, n)
    R.near(low, expected[0], 1e-12, "interval low at rho 0.5, n 20")
    R.near(high, expected[1], 1e-12, "interval high at rho 0.5, n 20")

    wider_low, wider_high = correlate.rho_interval(0.5, 8)
    R.check((wider_high - wider_low) > (high - low) * 1.5,
            f"a smaller sample gives a much wider interval, "
            f"{wider_high - wider_low:.3f} against {high - low:.3f}")

    R.check(R.guard(lambda: correlate.rho_interval(0.5, 3),
                    "n = 3 returns rather than dividing by zero",
                    default=("crashed",)) == (None, None),
            "no interval exists at n = 3, and none is invented")
    R.check(R.guard(lambda: correlate.rho_interval(0.5, 4),
                    "n = 4 returns an interval",
                    default=(None,))[0] is not None,
            "while n = 4 is the first sample that has one")
    R.check(R.guard(lambda: correlate.rho_interval(1.0, 20),
                    "rho = 1 returns rather than taking atanh of 1",
                    default=("crashed",)) == (None, None),
            "and none at rho = 1, where the Fisher transform reports a "
            "zero-width interval that is not the uncertainty")

    # THE SAMPLE-SIZE BAR, PROBED AT ONE POINT EITHER SIDE.
    below = correlate.correlate(list(range(14)), list(range(14)), name="m")
    R.check(not below.usable and "14 paired points" in below.reason,
            f"n = 14 is refused: {below.reason}")
    at_bar = correlate.correlate(list(range(15)),
                                 [0, 2, 1] + list(range(3, 15)), name="m")
    R.check(at_bar.usable, f"n = 15 is not: {at_bar.reason}")

    thin = correlate.correlate(list(range(15)), [0, 2, 1] + list(range(3, 15)),
                               name="m", min_ratings=14)
    R.check(not thin.usable and "thinnest point" in thin.reason,
            f"15 items resting on 14 ratings each is refused: {thin.reason}")
    thick = correlate.correlate(list(range(15)), [0, 2, 1] + list(range(3, 15)),
                                name="m", min_ratings=15)
    R.check(thick.usable, "15 ratings each is not")

    text = str(at_bar)
    R.check("[" in text and "]" in text and "n=15" in text,
            f"the printed form always carries its interval and n: {text}")
    R.check("NOT USABLE" in str(below),
            f"and says so when it is not usable: {below}")

    perfect = correlate.correlate(list(range(20)), list(range(20)), name="m")
    R.check(not perfect.usable and perfect.ci_low is None,
            "a perfect rho with no computable interval is not usable either")


def test_gate_rule() -> None:
    print("[17] which proxies may gate CI")

    # THE LOWER BOUND, NOT THE POINT ESTIMATE. A sample rho of 0.62 on twenty
    # points is consistent with almost no relationship at all.
    rho, n = 0.62, 20
    se = math.sqrt((1.0 + rho * rho / 2.0) / (n - 3))
    lower = math.tanh(math.atanh(rho) - Z_95 * se)
    R.check(lower < correlate.GATE_RHO,
            f"a point estimate of {rho} has a lower bound of {lower:.3f}, "
            "under the bar")

    weak = correlate.Correlation(metric="m", rho=rho, n_pairs=n,
                                 ci_low=lower, ci_high=0.85, min_ratings=20,
                                 usable=True, reason="")
    R.check(not correlate.may_gate_ci(weak),
            "so it may not gate, even though rho itself clears 0.6")
    R.check(correlate.may_gate_ci(weak, use_lower_bound=False),
            "which is precisely the reading of the plan this rejects")

    strong = correlate.Correlation(metric="m", rho=0.9, n_pairs=40,
                                   ci_low=0.81, ci_high=0.95, min_ratings=20,
                                   usable=True, reason="")
    R.check(correlate.may_gate_ci(strong), "a well-supported proxy may gate")
    R.check(not correlate.may_gate_ci(
        correlate.Correlation(metric="m", rho=0.9, n_pairs=8, ci_low=0.81,
                              ci_high=0.95, min_ratings=8, usable=False,
                              reason="too few")),
        "an unusable result may not gate whatever its interval says")

    on_bar = correlate.Correlation(metric="m", rho=0.9, n_pairs=40,
                                   ci_low=0.6, ci_high=0.95, min_ratings=20,
                                   usable=True, reason="")
    R.check(not correlate.may_gate_ci(on_bar),
            "the bar is strict: a lower bound exactly at 0.6 does not clear it")

    R.near(correlate.GATE_RHO, 0.6, 0.0, "the plan's threshold is 0.6")
    R.check(correlate.MIN_N == 15, f"and its sample-size floor is 15, "
                                   f"not {correlate.MIN_N}")


def test_correlate_proxies() -> None:
    print("[18] a listening result against a set of automated metrics")

    # A synthetic panel with a KNOWN answer: `follows` is a monotone function
    # of the human score, `inverted` is its reverse, `noise` is unrelated.
    items = {f"i{index:02d}": float(index) for index in range(20)}
    human = {name: correlate.HumanPoint(mean=value, n_ratings=20)
             for name, value in items.items()}
    scrambled = np.random.default_rng(31415).permutation(20)
    metrics = {
        "follows": {name: value ** 2 for name, value in items.items()},
        "inverted": {name: -value for name, value in items.items()},
        "noise": {name: float(scrambled[index])
                  for index, name in enumerate(sorted(items))},
        "constant": {name: 1.0 for name in items},
    }
    results = correlate.correlate_proxies(human, metrics)

    R.near(results["follows"].rho, 1.0, 1e-12, "a monotone proxy is rho 1")
    R.near(results["inverted"].rho, -1.0, 1e-12, "and its reverse is -1")
    R.check(math.isnan(results["constant"].rho),
            "a constant metric is undefined, not uncorrelated")
    R.check(abs(results["noise"].rho) < 0.35,
            f"and a shuffled one is near zero, {results['noise'].rho:+.3f}")
    R.check(not correlate.may_gate_ci(results["noise"]),
            "so it may not gate CI")

    R.check(results["follows"].min_ratings == 20,
            "the thinnest human point is carried through")

    # A METRIC MAY NOT DECLINE TO MEASURE THE CASES IT DOES BADLY ON.
    partial = {name: value for name, value in metrics["follows"].items()
               if name != "i05"}
    R.raises(correlate.CorrelationError,
             lambda: correlate.correlate_proxies(human, {"partial": partial}),
             "a metric missing an item is refused, not correlated on the overlap")
    R.raises(correlate.CorrelationError,
             lambda: correlate.correlate_proxies(
                 human, {"extra": {**metrics["follows"], "i99": 1.0}}),
             "and one covering items nobody rated is refused too")

    printed = correlate.report(results)
    R.check("may not gate" in printed and "rho" in printed,
            "the report says which proxies may gate and which may not")

    # End to end from a scored session: human points come out per rendered
    # clip, because that is the unit a metric produces a number for.
    key, responses = _scoring_fixture()
    points = score.item_points(responses, key, "quality")
    R.check(set(points) == {"i1::alpha", "i1::beta", "i2::alpha", "i2::beta"},
            f"one point per rendered clip, got {sorted(points)}")
    R.check(points["i1::alpha"].n_ratings == 6,
            f"i1 was judged by three listeners twice, got "
            f"{points['i1::alpha'].n_ratings}")
    thin = correlate.correlate_proxies(
        points, {"anything": {name: index for index, name
                              in enumerate(sorted(points))}})
    R.check(not thin["anything"].usable,
            "a four-point correlation is reported and refused, not printed bare")


# --- 6. the interface's wire ---------------------------------------------------


def test_server_blinds_at_the_wire() -> None:
    print("[19] the server never serves the key, and resolves clips by id only")

    with tempfile.TemporaryDirectory(prefix="anv_serve_") as work:
        comparisons = comparison_fixture(work, count=4)
        directory = os.path.join(work, "session")
        built = session.build_session(comparisons, ["L1", "L2"], directory,
                                      seed=21, repeats=1)

        server = serve.make_server(directory, "127.0.0.1", 0)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{port}"
        try:
            index = httpx.get(f"{base}/", timeout=10.0)
            R.check(index.status_code == 200 and "listening test" in index.text,
                    "the interface is served")

            roster = httpx.get(f"{base}/session.json", timeout=10.0).json()
            R.check(roster["listeners"] == ["L1", "L2"] and roster["trials"] is None,
                    "without a listener, only the roster is served")

            payload = httpx.get(f"{base}/session.json?listener=L1",
                                timeout=10.0).json()
            leaked = session.leaks(payload, ["alpha", "beta", "item-0",
                                             "condition", "repeat"])
            R.check(leaked == [], f"the served payload leaks nothing: {leaked}")
            R.check(len(payload["trials"]) == 5
                    and all("clip_a" in trial for trial in payload["trials"]),
                    "and carries L1's five trials")
            other = httpx.get(f"{base}/session.json?listener=L2",
                              timeout=10.0).json()
            R.check(other["trials"] != payload["trials"],
                    "L2's order is a different document")
            # AND L1'S DOCUMENT CONTAINS NONE OF L2'S. Narrowing by overwriting
            # one key rather than by excluding it leaves the whole session
            # nested somewhere in the reply, which is still a document the
            # browser received.
            l2_ids = {trial["trial_id"] for trial in other["trials"]}
            serialised = json.dumps(payload)
            R.check(not any(trial_id in serialised for trial_id in l2_ids),
                    "and none of L2's trial identifiers appear anywhere in L1's")
            R.check(httpx.get(f"{base}/session.json?listener=L9",
                              timeout=10.0).status_code == 404,
                    "an unknown listener is a 404")

            clip_id = payload["trials"][0]["clip_a"]
            audio = httpx.get(f"{base}/clip/{clip_id}.wav", timeout=30.0)
            with open(built.clip_files[clip_id], "rb") as handle:
                expected = handle.read()
            R.check(audio.status_code == 200 and audio.content == expected,
                    "a clip is served by its opaque identifier")

            R.check(httpx.get(f"{base}/clip/{'0' * 16}.wav",
                              timeout=10.0).status_code == 404,
                    "an identifier not in the map is a 404")
            R.check(httpx.get(f"{base}/key.json", timeout=10.0).status_code == 404,
                    "THE KEY IS NOT SERVED")
            R.check(httpx.get(f"{base}/clips/anything.wav",
                              timeout=10.0).status_code == 404,
                    "and neither is the clips directory by name")

            # AN IDENTIFIER IS A KEY, NEVER A PATH FRAGMENT. `/clip/key.json`
            # is the probe that matters and the one first omitted here: it
            # needs no traversal at all, because `key.json` sits in the same
            # directory as the clips a naive resolver would join against.
            for raw in ("/clip/key.json", "/clip/session.json",
                        "/clip/../key.json", "/clip/../../key.json",
                        "/clip/../session/key.json", "/clip/clips",
                        "/../key.json", "/clip/%2e%2e/key.json"):
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                connection.request("GET", raw)
                status = connection.getresponse().status
                connection.close()
                R.check(status == 404,
                        f"a traversal attempt at {raw} is a 404, got {status}")

            posted = httpx.post(f"{base}/responses", timeout=10.0, json={
                "listener": "L1",
                "responses": [{"listener": "L1",
                               "trial_id": payload["trials"][0]["trial_id"],
                               "preference": "A",
                               "ratings": {"A": {"match": 4, "quality": 4},
                                           "B": {"match": 3, "quality": 3}}}],
            })
            R.check(posted.status_code == 200, "responses are accepted")
            written = os.listdir(built.responses_dir)
            R.check(len(written) == 1 and written[0].startswith("L1-"),
                    f"and land in the responses directory, got {written}")

            with open(os.path.join(built.responses_dir, written[0])) as handle:
                saved = json.load(handle)
            parsed = score.parse_responses(saved)
            R.check(len(parsed) == 1 and parsed[0].preference == "A",
                    "what the interface posts is what the scorer reads")

            R.check(httpx.post(f"{base}/nowhere", timeout=10.0,
                               json={}).status_code == 404,
                    "no other route accepts a POST")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5.0)


# --- 7. what an operator sees when it goes wrong -------------------------------


def test_failures_name_what_broke() -> None:
    """The three refusals a first real panel actually hit.

    Every one of these was already REFUSED correctly before this test existed —
    the audio was never wrong. What was wrong is that the refusal did not carry
    enough for the person holding the laptop to act on it, or (in the third
    case) did not happen at all and reported success instead. `findings.md`
    records the same shape twice already: a check that fires with the wrong
    message is a check nobody can follow, and a failure reported as a 200 is
    worse than a crash.
    """
    print("[20] a refusal names the thing the operator has to change")

    with tempfile.TemporaryDirectory(prefix="anv_operator_") as work:
        # -- a silent segment stops a screen, and the message must say WHICH ---
        #
        # This is the first thing a real run hit: `01-Rhythm Guitar` opens with
        # eight seconds of lead-in, `build_screening` cuts its first segment
        # from frame zero, and the whole ten-trial screen died on
        # "clip 0 measures -inf LUFS" — a message that names neither the file,
        # nor the trial, nor the fact that the fix is to start later in the
        # source. The solver cannot know any of that; the caller already does.
        lead_in = write(os.path.join(work, "silent-lead-in.wav"),
                        np.zeros((1, int(SR * 4.0))))
        music = write(os.path.join(work, "music.wav"), pink(4.0))
        outputs = [os.path.join(work, "s0.wav"), os.path.join(work, "s1.wav")]

        R.raises(match.UnmeasurableLoudness,
                 lambda: match.match_group([lead_in, music], outputs,
                                           what="screening trial 7"),
                 "a silent input is refused with the caller's context",
                 contains="screening trial 7")
        R.raises(match.UnmeasurableLoudness,
                 lambda: match.match_group([lead_in, music], outputs,
                                           what="screening trial 7"),
                 "a silent input is refused naming the file that is silent",
                 contains="silent-lead-in.wav")
        R.raises(match.UnmeasurableLoudness,
                 lambda: match.match_group([lead_in, music], outputs,
                                           what="screening trial 7"),
                 "and still says why, not just where",
                 contains="above the absolute gate")

        # -- two submissions in one second are two submissions ----------------
        #
        # The response filename was `{listener}-{unix seconds}`. A retried POST
        # after a timeout, a second tab, or two listeners finishing together
        # wrote the same name, and the first file — a completed thirty-trial
        # session — disappeared behind a 200 and the word "saved". The
        # collision is forced here rather than raced for, because a test that
        # depends on two writes landing in the same second is a test that
        # passes on a slow machine.
        store_dir = os.path.join(work, "store")
        os.makedirs(os.path.join(store_dir, "clips"), exist_ok=True)
        with open(os.path.join(store_dir, "session.json"), "w") as handle:
            json.dump({"listeners": {"L1": []}, "scales": [], "preferences": []},
                      handle)
        with open(os.path.join(store_dir, "key.json"), "w") as handle:
            json.dump({"clips": {}}, handle)
        store = serve.SessionStore(store_dir)

        first = R.guard(lambda: store.record({"listener": "L1", "mark": "first"}),
                        "the first submission is written")
        second = R.guard(lambda: store.record({"listener": "L1", "mark": "second"}),
                         "the second submission is written")
        R.check(first != second,
                "two submissions in the same second get two filenames")
        written = sorted(os.listdir(store.responses_dir))
        R.check(len(written) == 2,
                f"both submissions survive on disk, got {written}")
        marks = set()
        for name in written:
            with open(os.path.join(store.responses_dir, name)) as handle:
                marks.add(json.load(handle).get("mark"))
        R.check(marks == {"first", "second"},
                f"neither submission overwrote the other, got {sorted(marks)}")

        # -- a bad response file is named ------------------------------------
        #
        # The server accepts anything JSON-shaped with a 200, so scoring is the
        # first place a broken submission is noticed — and by then the
        # directory holds one file per listener. `parse_responses` reports a
        # position within one payload, which is correct and, without the path,
        # unusable.
        scored_dir = os.path.join(work, "scored")
        responses_dir = os.path.join(scored_dir, "responses")
        os.makedirs(responses_dir, exist_ok=True)
        good = [{"listener": "L1", "trial_id": "t0", "preference": "A",
                 "ratings": {"A": {"match": 4, "quality": 4},
                             "B": {"match": 2, "quality": 2}}}]
        with open(os.path.join(responses_dir, "L1-1.json"), "w") as handle:
            json.dump({"listener": "L1", "responses": good}, handle)
        with open(os.path.join(responses_dir, "L2-2.json"), "w") as handle:
            json.dump({"listener": "L2", "responses": "nonsense"}, handle)

        R.raises(SystemExit, lambda: run_session._read_responses(scored_dir),
                 "a malformed submission names the file it came from",
                 contains="L2-2.json")
        R.raises(SystemExit, lambda: run_session._read_responses(scored_dir),
                 "and still carries the parser's own reason",
                 contains="expected a list of responses")

        # Replaced rather than added: the loop stops at the first bad file, so
        # leaving the previous one in place would test the sort order.
        os.remove(os.path.join(responses_dir, "L2-2.json"))
        with open(os.path.join(responses_dir, "L3-3.json"), "w") as handle:
            handle.write("{not json at all")
        R.raises(SystemExit, lambda: run_session._read_responses(scored_dir),
                 "a file that is not JSON names itself too",
                 contains="L3-3.json")


def test_the_phase0_gate_is_per_item() -> None:
    """A pooled prefer-or-tie rate can pass while the real criterion fails.

    THIS IS THE ENTIRE REASON THE GATE IS COMPUTED RATHER THAN EYEBALLED. The
    plan's stop-or-go point is prefer-or-tie on at least 6 of 10 FILES, and
    `score` used to print one pooled rate across every judgement. Those two
    numbers disagree exactly when preference is concentrated — a handful of
    items everyone loves carrying a majority of items nobody does — which is
    the shape a chain that works on one instrument and not others produces, and
    the shape most worth catching before booking three engineers.
    """
    print("[21] the Phase 0 gate is per item, not pooled")

    def item(name, prefer, tie, against):
        return listen_score.ItemPreference(item=name, condition="ours",
                                           prefer=prefer, tie=tie,
                                           against=against)

    # Four items unanimous FOR, six unanimous AGAINST, three listeners each.
    # Pooled: 12 of 30 judgements = 40%. Per item: 4 of 10.
    concentrated = [item(f"win{i}", 3, 0, 0) for i in range(4)]
    concentrated += [item(f"lose{i}", 0, 0, 3) for i in range(6)]
    gate = listen_score.Phase0Gate(
        condition="ours",
        required_items=listen_score.PHASE0_REQUIRED_ITEMS,
        items=tuple(concentrated))
    R.check(gate.items_passed == 4, f"four items pass, got {gate.items_passed}")
    R.check(not gate.passed, "and the gate fails, as it must")

    # Now the case that actually motivates it: a pooled rate that LOOKS fine.
    # Three items with all three listeners preferring, seven with one of three.
    # Pooled: 9 + 7 = 16 of 30 = 53%, a majority. Per item: 3 of 10.
    flattering = [item(f"win{i}", 3, 0, 0) for i in range(3)]
    flattering += [item(f"thin{i}", 1, 0, 2) for i in range(7)]
    pooled = sum(i.prefer_or_tie for i in flattering)
    total = sum(i.listeners for i in flattering)
    gate = listen_score.Phase0Gate(
        condition="ours",
        required_items=listen_score.PHASE0_REQUIRED_ITEMS,
        items=tuple(flattering))
    R.check(pooled * 2 > total,
            f"the pooled rate is a majority ({pooled}/{total})")
    R.check(gate.items_passed == 3 and not gate.passed,
            f"and the gate still FAILS at {gate.items_passed} of 10 — which is "
            f"the disagreement the pooled number hid")

    # Exactly at the bar, from both sides, because a gate probed only at 0 and
    # 10 would pass with the constant set to anything in between.
    at_bar = tuple([item(f"w{i}", 3, 0, 0) for i in range(6)]
                   + [item(f"l{i}", 0, 0, 3) for i in range(4)])
    under = tuple([item(f"w{i}", 3, 0, 0) for i in range(5)]
                  + [item(f"l{i}", 0, 0, 3) for i in range(5)])
    R.check(listen_score.Phase0Gate("ours", 6, at_bar).passed,
            "6 of 10 passes")
    R.check(not listen_score.Phase0Gate("ours", 6, under).passed,
            "5 of 10 does not")

    # A TIE COUNTS AS PREFER-OR-TIE. The plan says "prefer-or-tie", and reading
    # it as preference only would be a stricter gate than anyone agreed to.
    ties = tuple([item(f"t{i}", 0, 3, 0) for i in range(6)]
                 + [item(f"l{i}", 0, 0, 3) for i in range(4)])
    R.check(listen_score.Phase0Gate("ours", 6, ties).passed,
            "six items where every listener tied still passes")

    # THE BAR MUST BE A COUNT, NOT A PROPORTION, and at exactly ten items those
    # two are the same predicate — `passed >= 6` and `passed * 2 > 10` agree on
    # every input. A mutation swapping one for the other survived a test that
    # only ever used ten items, which is the "guard tested where it does not
    # bind" shape twice over. Eight items separates them: the count needs 6, the
    # proportion needs 5.
    eight = tuple([item(f"w{i}", 3, 0, 0) for i in range(5)]
                  + [item(f"l{i}", 0, 0, 3) for i in range(3)])
    R.check(not listen_score.Phase0Gate("ours", 6, eight).passed,
            "5 of 8 fails a bar of 6, where a proportional bar would pass it")

    # And the report must SAY when the session is not ten items rather than
    # rescaling the bar, since a proportional bar is a different criterion.
    short = listen_score.Phase0Gate("ours", 6, at_bar[:5])
    R.check("note:" in short.report(),
            f"a session with the wrong item count says so: {short.report()[-120:]}")



if __name__ == "__main__":
    if not os.path.exists(DEFAULT_BINARY):
        print(f"engine binary missing at {DEFAULT_BINARY}; run `make render` in engine/")
        sys.exit(2)

    print("Audionova blinded listening test\n")
    test_loudness_matching()
    test_mismatch_is_refused()
    test_ceiling_and_clipping()
    test_group_refusals()
    test_blinding()
    test_ab_assignment_is_balanced()
    test_seeded_and_reproducible()
    test_repeat_structure()
    test_build_session_end_to_end()
    test_screening_scoring()
    test_screening_stimulus_is_the_real_thing()
    test_scoring()
    test_inconsistency()
    test_response_validation()
    test_spearman_is_right()
    test_rho_interval_and_usability()
    test_gate_rule()
    test_correlate_proxies()
    test_server_blinds_at_the_wire()
    test_failures_name_what_broke()
    test_the_phase0_gate_is_per_item()
    print(f"\n{R.checks} checks, {R.failures} failure(s)")
    sys.exit(1 if R.failures else 0)
