"""EBU Tech 3341 / 3342 conformance for the loudness meter.

These are not tests we invented. EBU Tech 3341 and Tech 3342 publish a set of
signals together with the reading a compliant meter must produce and the
tolerance it must produce it within. That makes this the only measurement in
the project validated against an EXTERNAL authority rather than against our own
reasoning — which matters, because every other time this project has written a
test from first principles it has taken two or three attempts to get the test
itself right.

The signals are synthesised here rather than downloaded. The specification
defines them precisely enough to construct (sine frequency, level, duration),
so generating them keeps the suite dependency-free and hermetic.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.measure import loudness  # noqa: E402

SR = 48000.0


def sine(seconds: float, dbfs: float, freq: float = 1000.0, channels: int = 2) -> np.ndarray:
    """A sine at a given dBFS peak amplitude, in every channel."""
    n = int(round(seconds * SR))
    if n == 0:
        return np.zeros((channels, 0))
    amplitude = 10.0 ** (dbfs / 20.0)
    t = np.arange(n) / SR
    wave = amplitude * np.sin(2.0 * np.pi * freq * t)
    return np.tile(wave, (channels, 1))


def silence(seconds: float, channels: int = 2) -> np.ndarray:
    return np.zeros((channels, int(round(seconds * SR))))


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
        self.check(
            math.isfinite(got) and abs(got - want) <= tol,
            f"{what}: got {got:.3f}, want {want:.3f} +/- {tol}",
        )


R = Results()


# ---------------------------------------------------------------------------
# EBU Tech 3341 — integrated loudness


def test_tech3341_integrated() -> None:
    print("[1] EBU Tech 3341 integrated loudness")

    # Case 1 and 2: a stationary sine must read its own level. These are the
    # calibration cases; if the -0.691 offset or the K-weighting gain at 1 kHz
    # is wrong, everything downstream is wrong by a constant and nothing else
    # would reveal it.
    R.near(loudness.integrated_lufs(sine(20.0, -23.0), SR), -23.0, 0.1,
           "case 1: 1 kHz stereo sine at -23 dBFS")
    R.near(loudness.integrated_lufs(sine(20.0, -33.0), SR), -33.0, 0.1,
           "case 2: 1 kHz stereo sine at -33 dBFS")

    # Case 3: -36, -23, -36 LUFS in sequence with silence either side. The
    # relative gate must discard the quiet sections so the answer is the level
    # of the loud one, not an average.
    case3 = np.concatenate(
        [silence(10.0), sine(10.0, -36.0), sine(60.0, -23.0), sine(10.0, -36.0), silence(10.0)],
        axis=1,
    )
    R.near(loudness.integrated_lufs(case3, SR), -23.0, 0.1,
           "case 3: -36/-23/-36 sequence gates to the programme level")

    # Case 4: the same shape with the quiet sections at -72 LUFS, below the
    # ABSOLUTE gate. Tests the other gate independently — a meter with only the
    # relative gate passes case 3 and fails here.
    case4 = np.concatenate(
        [silence(10.0), sine(10.0, -72.0), sine(60.0, -23.0), sine(10.0, -72.0), silence(10.0)],
        axis=1,
    )
    R.near(loudness.integrated_lufs(case4, SR), -23.0, 0.1,
           "case 4: -72 dBFS sections fall below the absolute gate")

    # Case 5: 20 s at -26, 20.1 s at -20, 20 s at -26. The durations are
    # deliberately not round numbers, so a meter that mishandles the final
    # partial block reads wrong.
    case5 = np.concatenate([sine(20.0, -26.0), sine(20.1, -20.0), sine(20.0, -26.0)], axis=1)
    R.near(loudness.integrated_lufs(case5, SR), -23.0, 0.1,
           "case 5: -26/-20/-26 with a non-integer duration")

    # Mono reads 3 LU below dual-mono stereo. Asserted rather than merely
    # documented, because the product's input is usually a single mono stem and
    # this is the number an engineer is most likely to think is a bug.
    mono = sine(20.0, -23.0, channels=1)
    R.near(loudness.integrated_lufs(mono, SR), -26.0, 0.1,
           "mono sums one channel, so it reads 3 LU below dual-mono stereo")

    # Silence has no defined loudness. Returning a number would be worse than
    # returning nothing: it would flow into a planner prompt as fact.
    R.check(loudness.integrated_lufs(silence(5.0), SR) == float("-inf"),
            "digital silence has no integrated loudness")
    R.check(loudness.integrated_lufs(sine(0.1, -23.0), SR) == float("-inf"),
            "a file shorter than one gating block has no integrated loudness")

    # --- WHERE THE TWO GATES ACTUALLY SIT -----------------------------------
    #
    # Case 4 is documented as testing the absolute gate independently. It does
    # not. Its quiet sections are at -72 LUFS, 49 LU below the programme, so
    # the RELATIVE gate at -10 LU discards them on its own and the answer is
    # -23.015 with the absolute gate removed entirely. Measured: deleting the
    # absolute gate changes cases 1, 3, 4 and 5 by nothing at all.
    #
    # What the absolute gate is actually for is a file with no programme in
    # it. A stem that is nothing but noise floor has no loudness to report,
    # and the relative gate cannot say so — it measures everything relative to
    # whatever it finds, so it will confidently return the level of the hiss.
    R.check(loudness.integrated_lufs(sine(20.0, -72.0), SR) == float("-inf"),
            "a file entirely below -70 LUFS has no integrated loudness at all")
    # And just above the gate there IS a reading, so the bar is at -70 rather
    # than merely somewhere below the programme.
    R.near(loudness.integrated_lufs(sine(20.0, -68.0), SR), -68.0, 0.1,
           "while a file at -68 LUFS is quiet, not absent")

    # The relative gate, probed either side of -10 LU rather than at -13 LU
    # (case 3) where anything from -8 to -20 gives the same answer.
    #
    # With 60 s of programme at -23 the first pass lands at -24.1, so a -33
    # section is 1.1 LU INSIDE the gate and must be counted, and a -35 section
    # is 0.8 LU outside it and must not. The two readings differ by a whole dB
    # and bracket the constant to roughly -9 to -11 LU.
    near_in = np.concatenate(
        [sine(10.0, -33.0), sine(60.0, -23.0), sine(10.0, -33.0)], axis=1)
    R.near(loudness.integrated_lufs(near_in, SR), -24.086, 0.1,
           "a section 1 LU inside the relative gate is counted")
    near_out = np.concatenate(
        [sine(10.0, -35.0), sine(60.0, -23.0), sine(10.0, -35.0)], axis=1)
    R.near(loudness.integrated_lufs(near_out, SR), -23.014, 0.1,
           "and a section 1 LU outside it is not")

    # THE FIRST PASS RUNS OVER THE BLOCKS ABOVE THE ABSOLUTE GATE, not over
    # every block. Averaging the silence in as well drags the reference down
    # by however much silence the file happens to contain, and the relative
    # gate then stops excluding the things it exists to exclude. On a file
    # that is 86% digital silence it reads 2.7 dB low.
    mostly_silent = np.concatenate(
        [silence(60.0), sine(10.0, -23.0), sine(10.0, -36.0), silence(60.0)], axis=1)
    R.near(loudness.integrated_lufs(mostly_silent, SR), -23.118, 0.1,
           "silence does not dilute the reference the relative gate is measured from")

    # BS.1770 CHANNEL WEIGHTS. Left, right and centre count fully and the two
    # surrounds are weighted +1.5 dB, so the same waveform in five channels
    # reads 10*log10((3 + 2*1.41) / 2) above the stereo reading. Spelled out
    # here longhand rather than read from the module, because an expectation
    # that imports the thing under test cannot fail.
    stereo_reading = loudness.integrated_lufs(sine(20.0, -23.0), SR)
    five = loudness.integrated_lufs(sine(20.0, -23.0, channels=5), SR)
    R.near(five, stereo_reading + 10.0 * math.log10((3.0 + 2.0 * 1.41) / 2.0), 0.1,
           "five channels apply the +1.5 dB surround weighting")


# ---------------------------------------------------------------------------
# EBU Tech 3341 — momentary and short-term


def test_tech3341_windows() -> None:
    print("[2] EBU Tech 3341 momentary and short-term")

    steady = sine(20.0, -23.0)
    momentary = loudness.momentary_lufs(steady, SR)
    short = loudness.short_term_lufs(steady, SR)

    # THE WINDOW AND STEP ARE FIXED BY THE STANDARD, so they are asserted as
    # counts rather than left to be inferred from behaviour. A 20 s file gives
    # 1 + (20 - window) / step values, which pins the 400 ms and 3 s windows
    # and the 100 ms step exactly — none of which any behavioural assertion
    # here was sensitive to: halving the gating block or the overlap changed
    # nothing that was being measured.
    R.check(len(momentary) == 1 + round((20.0 - 0.400) / 0.100),
            f"momentary is a 400 ms window every 100 ms, got {len(momentary)} values")
    R.check(len(short) == 1 + round((20.0 - 3.0) / 0.100),
            f"short-term is a 3 s window every 100 ms, got {len(short)} values")

    R.check(len(momentary) > 0 and len(short) > 0, "window series are produced")
    R.near(float(np.max(momentary)), -23.0, 0.1, "momentary tracks a steady sine")
    R.near(float(np.max(short)), -23.0, 0.1, "short-term tracks a steady sine")

    # The window lengths must actually differ, and the way to see it is a
    # transient: a 400 ms window reaches full level long before a 3 s one.
    burst = np.concatenate([silence(5.0), sine(1.0, -23.0), silence(5.0)], axis=1)
    R.check(
        float(np.max(loudness.momentary_lufs(burst, SR)))
        > float(np.max(loudness.short_term_lufs(burst, SR))) + 3.0,
        "a 1 s burst reads higher on the 400 ms window than the 3 s one",
    )


# ---------------------------------------------------------------------------
# EBU Tech 3342 — loudness range


def test_tech3342_lra() -> None:
    print("[3] EBU Tech 3342 loudness range")

    # The four published cases. Each is two levels held long enough for the 3 s
    # short-term window to settle, and the expected LRA is their difference.
    cases = [
        (-20.0, -30.0, 10.0, "case 1"),
        (-20.0, -15.0, 5.0, "case 2"),
        (-40.0, -20.0, 20.0, "case 3"),
        (-50.0, -35.0, 15.0, "case 4"),
    ]
    for first_db, second_db, expected_lra, name in cases:
        audio = np.concatenate([sine(20.0, first_db), sine(20.0, second_db)], axis=1)
        R.near(loudness.loudness_range_lu(audio, SR), expected_lra, 1.0,
               f"{name}: LRA of a {first_db} / {second_db} dBFS pair")

    # A steady tone has no range at all. A meter that reported the spread of
    # its own measurement noise would put a nonzero LRA on every static signal.
    R.near(loudness.loudness_range_lu(sine(30.0, -23.0), SR), 0.0, 0.1,
           "a steady sine has zero loudness range")

    R.check(loudness.loudness_range_lu(silence(10.0), SR) == 0.0,
            "silence has zero loudness range")

    # THE PERCENTILES ARE THE MEASUREMENT, not a detail of it. Every case above
    # is two flat plateaus, whose 95th and 10th percentiles sit on the
    # plateaus — and so do the maximum and the minimum, so the four cases pass
    # unchanged if the percentiles are replaced by extremes.
    #
    # A linear ramp in dB makes short-term loudness roughly uniform over the
    # swept range, which is the one distribution where the percentile choice
    # has an arithmetic consequence: the 95th minus the 10th is 0.85 of the
    # range, against 1.00 for max minus min.
    n = int(120 * SR)
    t = np.arange(n) / SR
    level = -35.0 + 20.0 * (t / t[-1])
    ramp = np.tile(10.0 ** (level / 20.0) * np.sin(2.0 * np.pi * 1000.0 * t), (2, 1))
    R.near(loudness.loudness_range_lu(ramp, SR), 0.85 * 20.0, 0.5,
           "a 20 dB ramp has a loudness range of 0.85 * 20, not 20")

    # AND THE LRA GATE IS -20 LU, NOT THE INTEGRATED -10. A section far enough
    # below the programme is background — a fade-out, a room tone tail, a
    # count-in — and counting it would report a loudness range set by the
    # quietest thing in the file rather than by how dynamic the music is.
    # 40 s at -20 with 20 s at -42 has a 22 LU spread and a loudness range of
    # zero, because the quiet section is outside the gate.
    with_tail = np.concatenate([sine(40.0, -20.0), sine(20.0, -42.0)], axis=1)
    R.near(loudness.loudness_range_lu(with_tail, SR), 0.0, 0.1,
           "a section 20 LU below the programme is background, not range")


# ---------------------------------------------------------------------------
# BS.1770 Annex 2 — true peak


def test_true_peak() -> None:
    print("[4] BS.1770 Annex 2 true peak")

    # THE CASE THAT MAKES THE MEASUREMENT NECESSARY. A sine at exactly
    # Nyquist/2 sampled on the zero crossings has sample peak 0.0 but a real
    # peak of the full amplitude. Sample-peak metering reports -inf dBFS for a
    # signal that clips every converter it reaches.
    n = 48000
    phase_offset = np.pi / 4.0
    worst = np.sin(2.0 * np.pi * 12000.0 * np.arange(n) / SR + phase_offset)
    worst = worst / np.max(np.abs(worst))  # sample peak exactly 1.0

    tp = loudness.true_peak_dbtp(worst[None, :], SR)
    sp = loudness.sample_peak_dbfs(worst[None, :])
    R.check(tp > sp + 0.5,
            f"a 12 kHz sine reads {tp - sp:.2f} dB above sample peak (expect > 0.5)")

    # A full-scale sine at a low frequency is well described by its samples, so
    # the two measurements must nearly agree. This catches an interpolator with
    # the wrong gain, which would otherwise inflate every reading uniformly and
    # look like conservatism.
    low = sine(1.0, 0.0, freq=100.0, channels=1)
    R.near(loudness.true_peak_dbtp(low, SR), 0.0, 0.1,
           "a 100 Hz full-scale sine has true peak at 0 dBTP")

    # True peak can never be below sample peak. It is a reconstruction of a
    # signal that passes through those samples, so it passes through the peak.
    for freq in (50.0, 997.0, 5000.0, 15000.0):
        tone = sine(0.5, -3.0, freq=freq, channels=1)
        R.check(
            loudness.true_peak_dbtp(tone, SR) >= loudness.sample_peak_dbfs(tone) - 1e-9,
            f"true peak is never below sample peak at {freq:.0f} Hz",
        )

    # THE ANALYTIC REFERENCE. The band-limited reconstruction of a sampled
    # sinusoid IS that sinusoid, so its true peak is EXACTLY its amplitude
    # regardless of frequency or phase. A unit sinusoid must read 0.000 dBTP,
    # and any deviation is entirely our interpolator's error.
    #
    # This replaced a comparison against scipy's 64x resample_poly, which was
    # a worse reference than the thing it was checking: that function's default
    # Kaiser window gives roughly 1% passband ripple, or 0.086 dB — larger than
    # the error being measured. Comparing two approximations tells you they
    # disagree, not which one is right.
    fade_len = int(0.05 * SR)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade_len) / fade_len)
    for freq in (997.0, 5000.0, 11000.0, 19000.0):
        for phase in (0.0, 0.3, 1.1, 2.7):
            n = int(SR)
            tone = np.sin(2.0 * np.pi * freq * np.arange(n) / SR + phase)
            # Faded at both ends so the measurement is of the interpolator and
            # not of the file boundary — see the next case for that.
            tone[:fade_len] *= ramp
            tone[-fade_len:] *= ramp[::-1]
            # 0.05 dB was two hundred times looser than the meter's actual
            # error, which left the interpolator's design free: narrowing its
            # passband from 0.95 to 0.85 of the corner (0.006 dB) and cutting
            # its taps from 48 per phase to 12 both passed. findings.md
            # records this meter as accurate to 0.0001 dB, so the assertion
            # now says so — 0.002 dB is still a twenty-fold margin over the
            # worst reading here.
            R.near(loudness.true_peak_dbtp(tone[None, :], SR), 0.0, 0.002,
                   f"unit sinusoid at {freq:.0f} Hz, phase {phase:.1f}, reads 0 dBTP")

    # A file that begins mid-waveform has a genuine discontinuity at its edge,
    # and the reconstruction genuinely rings there. Reporting that is correct
    # rather than conservative — the ringing is in the signal a converter will
    # see — but it is worth pinning down, because it means the true peak of an
    # excerpt can exceed the true peak of the file it was cut from.
    abrupt = np.sin(2.0 * np.pi * 11000.0 * np.arange(int(SR)) / SR + 0.3)
    R.check(loudness.true_peak_dbtp(abrupt[None, :], SR) > 0.05,
            "an abrupt file boundary rings, and the meter reports it")

    R.check(loudness.true_peak_dbtp(silence(1.0), SR) == float("-inf"),
            "silence has no true peak")

    # EVERY CHANNEL IS METERED. Every case above is mono or dual-mono, so a
    # meter that read only the first channel passed all of them — and would
    # then clear a render for delivery on the strength of its quiet side.
    quiet_left = np.vstack([0.1 * np.sin(2.0 * np.pi * 1000.0 * np.arange(int(SR)) / SR),
                            0.9 * np.sin(2.0 * np.pi * 1000.0 * np.arange(int(SR)) / SR)])
    R.near(loudness.true_peak_dbtp(quiet_left, SR), -0.915, 0.02,
           "true peak is the loudest channel, not the first")
    R.near(loudness.sample_peak_dbfs(quiet_left), -0.915, 0.02,
           "and so is sample peak")

    # A PEAK IS A MAGNITUDE. Audio is not symmetric — a snare, a kick, any
    # asymmetric waveform reaches further one way than the other — so a meter
    # reading the maximum rather than the maximum magnitude misses every
    # negative-going peak there is. Sine tests cannot show it, because a sine
    # reaches the same distance both ways.
    asymmetric = np.concatenate(
        [0.3 * np.sin(2.0 * np.pi * 1000.0 * np.arange(int(SR)) / SR), [-0.8]])
    R.near(loudness.sample_peak_dbfs(asymmetric[None, :]), -1.938, 0.02,
           "a negative-going peak is a peak")
    R.check(loudness.true_peak_dbtp(asymmetric[None, :], SR) >= -1.94,
            "and true peak sees it too")

    # WHERE THE METER'S ACCURACY ENDS, asserted so it cannot quietly get
    # worse. The interpolator's passband reaches 0.95 of the base Nyquist, so
    # content in the top kilohertz or two is attenuated in the reconstruction
    # and the meter UNDER-reports — the unsafe direction for a delivery
    # ceiling. Two tones at 19 and 23 kHz have an exactly known continuous
    # peak, because the band-limited reconstruction of their samples IS that
    # sum, and the meter reads 0.86 dB below it.
    #
    # A single tone hides this: its sample peak is close enough to its true
    # peak that the sample-peak floor supplies the answer. It takes a sum
    # whose peak falls between samples to see the passband edge at all.
    # THREE POINTS, NOT ONE. A single tone pair pins one frequency and says
    # nothing about the shape of the error, and the shape is what matters: it
    # grows steeply as the content approaches Nyquist. Pinning only 19+23 would
    # have let the 21+23 case get arbitrarily worse.
    #
    # Each bound is the measured value plus a small margin, so a regression in
    # the interpolator design shows up here rather than in a delivery.
    n = int(SR)
    t = np.arange(n) / SR
    dense = np.arange(0.0, 1.0 / 1000.0, 1e-9)

    for low, high, phase, floor_db, label in (
        (19000.0, 23000.0, 0.7, -0.55, "19+23 kHz"),
        (20000.0, 22000.0, 0.3, -0.10, "20+22 kHz"),
        (21000.0, 23000.0, 0.1, -1.15, "21+23 kHz"),
    ):
        pair = 0.4 * (np.sin(2.0 * np.pi * low * t)
                      + np.sin(2.0 * np.pi * high * t + phase))
        analytic = 20.0 * math.log10(float(np.max(np.abs(
            0.4 * (np.sin(2.0 * np.pi * low * dense)
                   + np.sin(2.0 * np.pi * high * dense + phase))))))
        error = loudness.true_peak_dbtp(pair[None, :], SR) - analytic
        R.check(floor_db < error < 0.35,
                f"{label}: meter error {error:+.3f} dB, must stay within "
                f"{floor_db:+.2f} and +0.35")


# ---------------------------------------------------------------------------
# Sample-rate independence
#
# BS.1770 tabulates its filter coefficients at 48 kHz only. Reusing that table
# at another rate moves both corner frequencies, which biases every reading. A
# meter that is only correct at 48 kHz is not much use in an engine whose
# internal rate is 96.


def test_sample_rate_independence() -> None:
    print("[5] sample-rate independence")

    # THE PUBLISHED TABLE, WHICH IS THE ACTUAL EXTERNAL AUTHORITY. This
    # module's docstring says the analog parameters "reproduce the published
    # 48 kHz table exactly"; nothing checked it. Every other assertion here is
    # either a 1 kHz reading — where the RLB stage is flat and its corner can
    # move from 38 Hz to 60 Hz unnoticed — or a comparison of the filter with
    # itself at another rate, which a wrong filter passes as easily as a right
    # one.
    #
    # BS.1770-5 Tables 1 and 2, at 48 kHz.
    published = np.array([
        [1.53512485958697, -2.69169618940638, 1.19839281085285,
         1.0, -1.69065929318241, 0.73248077421585],
        [1.0, -2.0, 1.0,
         1.0, -1.99004745483398, 0.99007225036621],
    ])
    worst_coefficient = float(np.max(np.abs(
        loudness.k_weighting_sos(48000.0) - published)))
    R.check(worst_coefficient < 1e-12,
            f"the derived filter reproduces the published 48 kHz table to "
            f"{worst_coefficient:.2e}")

    for rate in (44100.0, 48000.0, 88200.0, 96000.0, 192000.0):
        n = int(round(20.0 * rate))
        t = np.arange(n) / rate
        wave = 10.0 ** (-23.0 / 20.0) * np.sin(2.0 * np.pi * 1000.0 * t)
        stereo = np.tile(wave, (2, 1))
        R.near(loudness.integrated_lufs(stereo, rate), -23.0, 0.1,
               f"a -23 dBFS sine reads -23 LUFS at {rate:.0f} Hz")

    # The K-weighting response itself must be rate-independent, not just its
    # effect on one tone. Compared at matched physical frequencies.
    from scipy import signal as _signal

    probe_hz = np.array([50.0, 200.0, 1000.0, 4000.0, 12000.0])
    responses = []
    for rate in (44100.0, 48000.0, 96000.0):
        sos = loudness.k_weighting_sos(rate)
        _, h = _signal.sosfreqz(sos, worN=2.0 * np.pi * probe_hz / rate)
        responses.append(20.0 * np.log10(np.abs(h)))

    worst = max(
        float(np.max(np.abs(responses[0] - responses[i]))) for i in (1, 2)
    )
    R.check(worst < 0.15,
            f"K-weighting response matches across rates within {worst:.3f} dB")


if __name__ == "__main__":
    print("Audionova measurement tests — loudness\n")
    test_tech3341_integrated()
    test_tech3341_windows()
    test_tech3342_lra()
    test_true_peak()
    test_sample_rate_independence()
    print(f"\n{R.checks} checks, {R.failures} failure(s)")
    sys.exit(1 if R.failures else 0)
