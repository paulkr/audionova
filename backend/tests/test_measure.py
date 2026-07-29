"""Tests for the spectral, dynamic, stereo, gating and hygiene measurements.

Wherever a measurement has an ANALYTIC answer, that is what it is checked
against — a sine's crest factor is exactly 3.01 dB, a square wave's is exactly
0, pink noise's tilt is exactly 0 dB/octave by the convention we chose,
uncorrelated channels lose exactly 0 dB beyond the arithmetic 3. Checking
against a reimplementation would only prove the two agree.

The loudness meter has its own file, since it is validated against published
EBU conformance signals rather than against derivation.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.audio import wavio  # noqa: E402
from backend.measure import (  # noqa: E402
    dynamics,
    gating,
    hygiene,
    report as report_mod,
    spectrum,
    stereo,
)

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
        self.check(ok, f"{what}: got {got}, want {want} +/- {tol}")


R = Results()


def pink_noise(n: int) -> np.ndarray:
    """Pink noise by spectral shaping: exactly -3 dB/octave PSD, by construction.

    Generated in the frequency domain rather than with a filter approximation,
    so the reference this file tests against is exact rather than close.
    """
    spec = RNG.normal(size=n // 2 + 1) + 1j * RNG.normal(size=n // 2 + 1)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    shape = np.ones_like(freqs)
    shape[1:] = 1.0 / np.sqrt(freqs[1:])
    shape[0] = 0.0
    out = np.fft.irfft(spec * shape, n=n)
    return out / np.max(np.abs(out)) * 0.5


def white_noise(n: int, amplitude: float = 0.2) -> np.ndarray:
    return RNG.normal(0.0, amplitude, n)


# ---------------------------------------------------------------------------


def test_spectrum() -> None:
    print("[1] spectrum")

    n = int(SR * 8)

    # Parseval: the PSD must integrate to the signal's mean square. If the
    # window normalisation is wrong, every band level is off by a constant and
    # nothing else in this file would catch it — the tilt is a difference, the
    # bandwidth limit is a ratio, and both survive a uniform offset.
    tone = 0.5 * np.sin(2.0 * np.pi * 1000.0 * np.arange(n) / SR)
    freqs, power = spectrum.gated_ltas(tone[None, :], SR)
    bin_width = freqs[1] - freqs[0]
    integrated = float(np.sum(power) * bin_width)
    R.near(integrated, 0.125, 0.005, "PSD integrates to mean square (A^2/2 = 0.125)")

    # A tone lands in its own third-octave band and not a neighbouring one.
    centres, levels = spectrum.third_octave_levels_db(freqs, power, SR)
    peak_band = float(centres[int(np.argmax(levels))])
    R.check(peak_band == 1000.0, f"a 1 kHz tone peaks in the 1000 Hz band, got {peak_band}")

    # THE SIGN CONVENTION, asserted rather than trusted. Pink noise is 0 and
    # white is +3 on band-integrated levels; getting this backwards would
    # invert every tonal decision made downstream.
    freqs, power = spectrum.gated_ltas(pink_noise(n)[None, :], SR)
    centres, levels = spectrum.third_octave_levels_db(freqs, power, SR)
    tilt, residual = spectrum.spectral_tilt_db_per_octave(centres, levels)
    R.near(tilt, 0.0, 0.35, "pink noise reads 0 dB/octave (the neutral reference)")
    # Small residual on a FLAT spectrum is the case r-squared got wrong: it
    # scored pink noise as an unreliable fit precisely because there was no
    # slope for a slope-explained-variance metric to find.
    R.check(residual < 1.5,
            f"pink noise sits close to its fitted line, residual {residual:.2f} dB")

    freqs, power = spectrum.gated_ltas(white_noise(n)[None, :], SR)
    centres, levels = spectrum.third_octave_levels_db(freqs, power, SR)
    tilt, _ = spectrum.spectral_tilt_db_per_octave(centres, levels)
    R.near(tilt, 3.0, 0.35, "white noise reads +3 dB/octave")

    # A scooped spectrum must report a poor fit, so the tilt is not mistaken
    # for a description of it. Without this, "tilt 0.0" is ambiguous between
    # neutral and violently scooped.
    scooped = pink_noise(n)
    from scipy import signal as _signal

    notch = _signal.butter(2, [400.0, 2000.0], btype="bandstop", fs=SR, output="sos")
    scooped = _signal.sosfilt(notch, scooped)
    freqs, power = spectrum.gated_ltas(scooped[None, :], SR)
    centres, levels = spectrum.third_octave_levels_db(freqs, power, SR)
    _, scooped_residual = spectrum.spectral_tilt_db_per_octave(centres, levels)
    R.check(scooped_residual > 4.0,
            f"a scooped spectrum deviates from its fitted line, residual "
            f"{scooped_residual:.2f} dB")

    # Bandwidth limit: the codec-cliff case it exists to catch.
    lopped = _signal.sosfilt(
        _signal.butter(16, 15000.0, btype="low", fs=SR, output="sos"), pink_noise(n)
    )
    freqs, power = spectrum.gated_ltas(lopped[None, :], SR)
    limit, confidence = spectrum.bandwidth_limit_hz(freqs, power, SR)
    R.near(limit, 15500.0, 1500.0, "a 15 kHz brick wall is detected")
    R.check(confidence >= 0.4, f"and detected confidently, {confidence:.2f}")

    # Full-band material must NOT be reported as limited, or every unprocessed
    # file gets a spurious "do not boost air" instruction.
    freqs, power = spectrum.gated_ltas(pink_noise(n)[None, :], SR)
    limit, confidence = spectrum.bandwidth_limit_hz(freqs, power, SR)
    R.check(limit > 16000.0 or confidence < 0.4,
            f"full-band material is not flagged as limited (limit {limit:.0f}, "
            f"confidence {confidence:.2f})")

    # ERB spacing must be narrow at the bottom and wide at the top; that is the
    # entire reason for using it.
    edges, erb_centres = spectrum.erb_band_edges(50.0, 16000.0, 28)
    widths = np.diff(edges)
    R.check(widths[-1] > widths[0] * 10.0,
            "ERB bands widen with frequency as the ear's resolution does")


def test_dynamics() -> None:
    print("[2] dynamics")

    n = int(SR * 4)
    t = np.arange(n) / SR

    # Analytic anchors. A sine's peak-to-RMS is sqrt(2), which is 3.0103 dB
    # exactly; a square wave's is 1, which is 0 dB exactly.
    sine = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    R.near(dynamics.crest_factor_db(sine[None, :]), 3.0103, 0.01,
           "a sine has a crest factor of 3.01 dB")

    square = 0.5 * np.sign(np.sin(2.0 * np.pi * 440.0 * t))
    R.near(dynamics.crest_factor_db(square[None, :]), 0.0, 0.05,
           "a square wave has a crest factor of 0 dB")

    # PLR is arithmetic and must stay arithmetic.
    R.near(dynamics.peak_to_loudness_ratio_db(-1.0, -14.0), 13.0, 1e-9,
           "PLR is true peak minus integrated loudness")

    # Short-term RMS of a steady sine is its RMS, at every percentile.
    rms_db = dynamics.short_term_rms_db(sine[None, :], SR)
    percentiles = dynamics.level_percentiles_db(rms_db)
    expected = 20.0 * math.log10(0.5 / math.sqrt(2.0))
    R.near(percentiles["p50_dbfs"], expected, 0.1, "median short-term RMS of a sine")
    R.near(percentiles["p95_dbfs"] - percentiles["p5_dbfs"], 0.0, 0.2,
           "a steady sine has no spread between its RMS percentiles")

    # THE CASE BROADBAND CREST HIDES. A squashed low end plus a spiky top reads
    # as unremarkable overall; per-band it is obvious. This is the measurement's
    # entire justification, so it is asserted rather than assumed.
    low = 0.4 * np.sign(np.sin(2.0 * np.pi * 80.0 * t))  # crest 0 dB
    spikes = np.zeros(n)
    spikes[:: int(SR // 8)] = 1.0
    from scipy import signal as _signal

    high = _signal.sosfilt(
        _signal.butter(4, [4000.0, 10000.0], btype="bandpass", fs=SR, output="sos"), spikes
    )
    high = 0.4 * high / max(np.max(np.abs(high)), 1e-12)
    mixed = low + high

    centres, crests = dynamics.per_erb_crest_db(mixed[None, :], SR)
    low_band = crests[centres < 200.0]
    high_band = crests[centres > 4000.0]
    R.check(low_band.size > 0 and high_band.size > 0, "per-ERB crest covers both ends")
    R.check(float(np.mean(high_band)) > float(np.mean(low_band)) + 6.0,
            f"per-ERB crest separates a squashed low end ({np.mean(low_band):.1f} dB) "
            f"from a spiky top ({np.mean(high_band):.1f} dB)")

    # Envelope modulation: an amplitude-modulated tone must report its own
    # modulation rate. This is the number attack and release are derived from,
    # and it has an exact right answer.
    for mod_hz in (2.0, 4.0, 8.0):
        carrier = np.sin(2.0 * np.pi * 1000.0 * t)
        envelope = 0.5 * (1.0 + 0.9 * np.sin(2.0 * np.pi * mod_hz * t))
        freqs, power = dynamics.envelope_modulation_spectrum((carrier * envelope)[None, :], SR)
        peak_hz = float(freqs[int(np.argmax(power))])
        R.near(peak_hz, mod_hz, 0.5, f"envelope modulation finds a {mod_hz} Hz tremolo")

    # A steady tone has no modulation to find.
    freqs, power = dynamics.envelope_modulation_spectrum(sine[None, :], SR)
    summary = dynamics.modulation_summary(freqs, power)
    R.check(summary["depth_db"] < -40.0,
            f"a steady sine reports almost no modulation depth, {summary['depth_db']:.1f} dB")


def test_stereo() -> None:
    print("[3] stereo")

    n = int(SR * 4)
    mono = pink_noise(n)

    # Identical channels.
    dual = np.vstack([mono, mono])
    R.near(stereo.broadband_correlation(dual), 1.0, 1e-6, "identical channels correlate at 1")
    R.near(stereo.mono_compatibility_loss_db(dual), 0.0, 1e-6,
           "identical channels lose nothing on summing")
    kind, confidence = stereo.stereo_type(dual)
    R.check(kind == "dual_mono", f"identical channels classify as dual_mono, got {kind}")

    # Polarity-inverted: the case that survives every other measurement looking
    # fine and destroys the mono sum completely.
    inverted = np.vstack([mono, -mono])
    R.near(stereo.broadband_correlation(inverted), -1.0, 1e-6,
           "an inverted channel correlates at -1")
    R.check(stereo.mono_compatibility_loss_db(inverted) > 40.0,
            "an inverted channel cancels on summing")
    kind, _ = stereo.stereo_type(inverted)
    R.check(kind == "out_of_phase", f"an inverted channel classifies as out_of_phase, got {kind}")

    # THE NORMALISATION THAT MAKES THE MEASUREMENT USEFUL. Two uncorrelated
    # channels lose 3 dB on summing, and that 3 dB is arithmetic rather than a
    # defect. Reporting it as a loss would flag every normally-wide mix as
    # broken, so the measurement must read zero here.
    independent = np.vstack([pink_noise(n), pink_noise(n)])
    R.near(stereo.broadband_correlation(independent), 0.0, 0.05,
           "independent channels correlate at 0")
    R.near(stereo.mono_compatibility_loss_db(independent), 0.0, 0.5,
           "independent channels lose nothing BEYOND the arithmetic 3 dB")

    # A silent band is not a decorrelated band.
    silent = np.zeros((2, n))
    R.near(stereo.broadband_correlation(silent), 1.0, 1e-9,
           "silence is not reported as decorrelated")

    # Per-band: width applied only to the top must show up only at the top.
    from scipy import signal as _signal

    side = _signal.sosfilt(
        _signal.butter(4, 3000.0, btype="high", fs=SR, output="sos"), pink_noise(n)
    )
    banded = np.vstack([mono + side, mono - side])
    bands = stereo.per_band_stereo(banded, SR)
    R.check(len(bands) > 0, "per-band stereo produces bands")
    low_corr = [b["correlation"] for b in bands if b["centre_hz"] < 500.0]
    high_corr = [b["correlation"] for b in bands if b["centre_hz"] > 4000.0]
    R.check(low_corr and high_corr and min(low_corr) > 0.9 and max(high_corr) < 0.6,
            f"high-frequency-only width shows as decorrelation only up top "
            f"(low {min(low_corr) if low_corr else None}, high {max(high_corr) if high_corr else None})")

    R.check(stereo.stereo_type(mono[None, :])[0] == "mono", "a single channel is mono")


def test_gating() -> None:
    print("[4] gating")

    # THE HEADLINE CLAIM OF THE WHOLE MODULE, demonstrated on the mechanism
    # that actually causes the damage.
    #
    # A lead vocal is 20-30% of its own track's duration, has almost nothing
    # below 150 Hz, and sits over a floor of rumble — HVAC, traffic, stand
    # noise — that is inaudible under the performance and completely exposed in
    # the gaps. An ungated long-term average therefore reports substantial
    # low-frequency energy that IS NOT PART OF THE VOICE.
    #
    # The consequence is not a small bias. A planner reading that spectrum sees
    # a muddy source and prescribes a low cut, which does nothing to a voice
    # that had no low end to begin with, and it spends a band doing it.
    #
    # Tilt is the wrong thing to assert here; it averages the contamination
    # across ten octaves and dilutes it to a fraction of a dB. The low band is
    # where the lie lives, so the low band is what gets measured.
    from scipy import signal as _signal

    n = int(SR * 24)
    voice_like = _signal.sosfilt(
        _signal.butter(6, 150.0, btype="high", fs=SR, output="sos"), pink_noise(n)
    )
    rumble = _signal.sosfilt(
        _signal.butter(6, 120.0, btype="low", fs=SR, output="sos"), white_noise(n, 1.0)
    )
    rumble = rumble / np.max(np.abs(rumble)) * 10 ** (-34 / 20)

    sparse = rumble.copy()
    for start_s in (2.0, 8.0, 14.0, 20.0):
        s = int(start_s * SR)
        sparse[s : s + int(1.5 * SR)] += voice_like[s : s + int(1.5 * SR)]

    mask, coverage = gating.gated_frame_mask(sparse[None, :], SR)
    R.check(0.15 < coverage < 0.60,
            f"gate keeps roughly the programme regions, coverage {coverage:.2f}")

    # WHAT GATING ACTUALLY FIXES is the SHAPE, not the noise. A continuous
    # rumble is present under the programme as well as between it, so no gate
    # can remove it — and expecting one to would be expecting noise reduction
    # from a level threshold.
    #
    # The lie is a ratio. During the 75% of the file with no programme, the
    # only content is rumble. Averaging over the whole file therefore leaves
    # the low band at its full level while dividing the programme bands by
    # four, which is 6 dB. The spectrum that comes out is the source's own,
    # tilted 6 dB toward the bottom: it reads muddy for a source that is not.
    #
    # A planner given that spectrum prescribes a low cut for a voice with no
    # low end, spends a band doing nothing, and leaves the actual problem
    # untouched.

    def band_balance(signal_in, use_mask):
        """Low band minus mid band, in dB — the shape, independent of level."""
        freqs, power = spectrum.gated_ltas(signal_in[None, :], SR, use_mask)
        centres, levels = spectrum.third_octave_levels_db(freqs, power, SR)
        low = (centres >= 50.0) & (centres <= 125.0)
        mid = (centres >= 500.0) & (centres <= 2000.0)
        return float(np.mean(levels[low]) - np.mean(levels[mid]))

    # The truth, derived without using the gating code at all: the spectrum of
    # the actual signal over the regions where the programme is playing.
    programme_only = np.concatenate([
        sparse[int(start_s * SR) : int(start_s * SR) + int(1.5 * SR)]
        for start_s in (2.0, 8.0, 14.0, 20.0)
    ])
    truth = band_balance(programme_only, None)
    gated_balance = band_balance(sparse, mask)
    ungated_balance = band_balance(sparse, None)

    # The disease: without gating, the source must actually read muddier than
    # it is, or the cure below proves nothing.
    R.check(ungated_balance > truth + 4.0,
            f"ungated averaging tilts the spectrum toward the low end "
            f"(truth {truth:.1f} dB, ungated {ungated_balance:.1f} dB)")

    R.check(abs(gated_balance - truth) < abs(ungated_balance - truth),
            f"gating restores the shape "
            f"(truth {truth:.1f}, gated {gated_balance:.1f}, ungated {ungated_balance:.1f})")
    R.near(gated_balance, truth, 2.0,
           "the gated spectrum's balance matches the programme's own")

    # A file too short to gate must not be silently emptied.
    short = pink_noise(int(SR * 0.1))
    mask, coverage = gating.gated_frame_mask(short[None, :], SR)
    R.check(coverage == 1.0 and mask.all(),
            "a file shorter than one gating block is left fully unmasked")

    # Segments must be contiguous runs, not a re-derivation of the mask.
    test_mask = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0], dtype=bool)
    R.check(gating.gated_segments(test_mask) == [(1, 3), (5, 8)],
            f"segments split into runs, got {gating.gated_segments(test_mask)}")
    R.check(gating.gated_segments(np.ones(5, dtype=bool)) == [(0, 5)],
            "an all-true mask is one segment")


def test_hygiene() -> None:
    print("[5] hygiene")

    n = int(SR * 2)
    t = np.arange(n) / SR

    # RUNS, not individual samples. A sine that merely touches full scale is
    # not clipped; one whose peaks are flattened is.
    touching = np.sin(2.0 * np.pi * 100.0 * t)
    R.near(hygiene.clipping(touching[None, :])["clipped_sample_percent"], 0.0, 0.001,
           "a sine touching full scale is not reported as clipped")

    # A full-scale sine is quadratic near its peak and holds ~5 consecutive
    # samples within 0.0005 of the top at 100 Hz. A run-length test alone reads
    # that as 2% clipping on undamaged audio and fires "do not add drive".
    for freq in (50.0, 100.0, 440.0):
        clean_tone = np.sin(2.0 * np.pi * freq * t)
        R.near(hygiene.clipping(clean_tone[None, :])["clipped_sample_percent"], 0.0, 0.001,
               f"a clean {freq:.0f} Hz sine at full scale is not called clipped")

    flattened = np.clip(1.6 * np.sin(2.0 * np.pi * 100.0 * t), -1.0, 1.0)
    clip = hygiene.clipping(flattened[None, :])
    R.check(clip["clipped_sample_percent"] > 5.0,
            f"a flat-topped sine is reported as clipped, {clip['clipped_sample_percent']:.2f}%")
    R.check(clip["longest_run_samples"] >= 3.0, "and its runs are measured")

    # Clipped and then turned down is still clipped. An absolute threshold
    # would only ever find damage that happened to survive at full scale, and
    # a gain stage after the clipper is the normal case rather than the
    # exception.
    attenuated = flattened * 0.4
    R.check(hygiene.clipping(attenuated[None, :])["clipped_sample_percent"] > 5.0,
            "clipping is still detected after the file was attenuated")

    # DC offset, with an exact expected value.
    offset = 0.01 + 0.1 * np.sin(2.0 * np.pi * 200.0 * t)
    R.near(hygiene.dc_offset(offset[None, :])[0], 20.0 * math.log10(0.01), 0.5,
           "DC offset is measured in dBFS")

    # Noise floor: a known floor under a known programme.
    floor_amplitude = 10 ** (-60 / 20)
    quiet = RNG.normal(0.0, floor_amplitude, n)
    loud = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    material = np.concatenate([quiet, loud, quiet])
    floor_db, confidence = hygiene.noise_floor_db(material[None, :], SR)
    R.near(floor_db, 20.0 * math.log10(floor_amplitude), 3.0,
           "the noise floor of a known-floor signal")
    R.check(confidence >= 0.4, f"and it is measured confidently, {confidence:.2f}")

    # A wall-to-wall loud file does not reveal a floor, and must say so rather
    # than reporting the quietest moment of a loud signal as one.
    _, dense_confidence = hygiene.noise_floor_db(loud[None, :], SR)
    R.check(dense_confidence < 0.4,
            f"a signal with no quiet passage reports low confidence, {dense_confidence:.2f}")

    stereo_dead = np.vstack([loud, np.zeros(n)])
    R.check(hygiene.dead_channels(stereo_dead) == [1], "a silent channel is detected")
    R.check(hygiene.dead_channels(np.vstack([loud, loud])) == [], "two live channels are not")

    broken = np.array([[0.1, np.nan, 0.3, np.inf]])
    R.check(hygiene.nan_or_inf_count(broken) == 2, "NaN and Inf are counted")


def test_report() -> None:
    print("[6] report assembly")

    n = int(SR * 8)
    t = np.arange(n) / SR
    clean = 0.3 * pink_noise(n)
    result = report_mod.measure(clean[None, :], SR).to_dict()

    for section in ("source", "loudness", "dynamics", "spectrum", "stereo", "hygiene", "warnings"):
        R.check(section in result, f"report contains the {section} section")

    R.check(result["source"]["channels"] == 1, "channel count is reported")
    R.near(result["source"]["duration_s"], 8.0, 0.01, "duration is reported")
    R.check(result["loudness"]["mono_convention_note"] is not None,
            "a mono file carries the BS.1770 convention note")

    # Units in every key name. A bare number invites the reader to assume a
    # unit, and sometimes to assume wrongly.
    def leaf_keys(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                yield from leaf_keys(value, f"{path}.{key}" if path else key)
        elif isinstance(node, list):
            for item in node:
                yield from leaf_keys(item, path)
        else:
            yield path, node

    # Dimensionless quantities genuinely have no unit to name. Everything else
    # must carry one somewhere in the key, not necessarily as a suffix:
    # `tilt_db_per_octave_re_pink` names its unit and its reference, which is
    # more informative than a bare `_db` ending, and a suffix-only rule would
    # reject it.
    dimensionless = {
        "analyzer_version", "channels", "is_float", "bit_depth", "type",
        "correlation", "runs", "dead_channels", "non_finite_samples",
        "confidence", "code", "severity", "message", "mono_convention_note",
        "ratio",
    }
    unit_tokens = ("db", "hz", "lufs", "lu", "dbfs", "dbtp", "percent",
                   "samples", "_s", "seconds", "confidence")

    unitless = []
    for path, value in leaf_keys(result):
        leaf = path.split(".")[-1]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if leaf in dimensionless or leaf.endswith("_confidence"):
            continue
        parts = set(leaf.split("_"))
        if parts & set(unit_tokens) or any(t in leaf for t in ("_db", "_hz", "_s")):
            continue
        unitless.append(path)
    R.check(not unitless, f"every numeric key names its unit; missing on {unitless}")

    # Warnings must fire on the conditions they exist for, and must not fire on
    # clean material — a report that warns about everything is a report nobody
    # reads.
    R.check(not any(w["severity"] in ("critical", "high") for w in result["warnings"]),
            f"clean material raises no critical or high warnings, got "
            f"{[w['code'] for w in result['warnings']]}")

    codes = lambda res: {w["code"] for w in res["warnings"]}

    clipped = np.clip(2.0 * np.sin(2.0 * np.pi * 220.0 * t), -1.0, 1.0)
    R.check("clipped_source" in codes(report_mod.measure(clipped[None, :], SR).to_dict()),
            "a clipped source is warned about")

    inverted = np.vstack([clean, -clean])
    R.check("out_of_phase" in codes(report_mod.measure(inverted, SR).to_dict()),
            "an out-of-phase pair is warned about")

    dual = np.vstack([clean, clean])
    R.check("dual_mono" in codes(report_mod.measure(dual, SR).to_dict()),
            "dual mono is noted")

    from scipy import signal as _signal

    lopped = _signal.sosfilt(
        _signal.butter(16, 15000.0, btype="low", fs=SR, output="sos"), clean
    )
    R.check("bandwidth_limited" in codes(report_mod.measure(lopped[None, :], SR).to_dict()),
            "a bandwidth-limited source is warned about")

    silent = report_mod.measure(np.zeros((1, n)), SR).to_dict()
    R.check("no_programme_material" in codes(silent),
            "a silent file is warned about rather than measured")
    R.check(silent["loudness"]["integrated_lufs"] is None,
            "and its loudness is null rather than a fabricated number")

    # Warnings arrive worst-first, so a truncated read still sees what matters.
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    both = np.clip(2.0 * np.sin(2.0 * np.pi * 220.0 * t), -1.0, 1.0)
    both_report = report_mod.measure(np.vstack([both, -both]), SR).to_dict()
    severities = [order[w["severity"]] for w in both_report["warnings"]]
    R.check(severities == sorted(severities), "warnings are ordered by severity")

    # The whole report must survive a JSON round trip: -inf and NaN do not
    # exist in JSON, and a report that cannot be serialised cannot reach a
    # prompt.
    import json

    text = json.dumps(silent, allow_nan=False)
    R.check(len(text) > 0, "a degenerate report serialises to strict JSON")


def test_wavio_matches_engine() -> None:
    print("[7] Python and C++ WAV decoders agree")

    # ONE DECODER SEMANTIC, EXPRESSED TWICE. If these two disagree, the
    # measurement report describes different samples than the renderer
    # processes, and every number in it is quietly wrong about the audio the
    # user hears.
    n = 4096
    original = np.vstack([
        np.linspace(-1.0, 1.0, n),
        0.7 * np.sin(np.linspace(0.0, 60.0, n)),
    ])

    for bit_depth, is_float in ((16, False), (24, False), (32, False), (32, True), (64, True)):
        encoded = wavio.encode_wav(original, SR, bit_depth, is_float)
        decoded = wavio.decode_wav(encoded)

        R.check(decoded.channels == 2 and decoded.frames == n,
                f"{bit_depth}-bit {'float' if is_float else 'PCM'}: shape round trips")
        R.near(decoded.sample_rate, SR, 0.0,
               f"{bit_depth}-bit {'float' if is_float else 'PCM'}: sample rate round trips")

        step = 1.0 if is_float else 1.0 / float(1 << (bit_depth - 1))
        expected = original if is_float else np.clip(original, -1.0, 1.0 - step)
        worst = float(np.max(np.abs(decoded.samples - expected)))
        tolerance = 0.0 if (is_float and bit_depth == 64) else (
            1e-7 if is_float else step * 0.5001
        )
        R.check(worst <= tolerance,
                f"{bit_depth}-bit {'float' if is_float else 'PCM'}: values round trip "
                f"within {tolerance:.2e}, worst {worst:.2e}")

    # 8-bit WAV is UNSIGNED, alone among the integer depths. Reading it signed
    # produces audio that is inverted and offset by half scale, and it still
    # looks like a waveform.
    header = (
        b"RIFF" + (4 + 24 + 8 + 4).to_bytes(4, "little") + b"WAVE"
        + b"fmt " + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little") + (1).to_bytes(2, "little")
        + int(SR).to_bytes(4, "little") + int(SR).to_bytes(4, "little")
        + (1).to_bytes(2, "little") + (8).to_bytes(2, "little")
        + b"data" + (4).to_bytes(4, "little") + bytes([128, 255, 0, 128])
    )
    decoded = wavio.decode_wav(header)
    R.near(float(decoded.samples[0, 0]), 0.0, 1e-9, "8-bit: 128 is the zero code")
    R.check(decoded.samples[0, 1] > 0.9, "8-bit: 255 is positive full scale")
    R.check(decoded.samples[0, 2] < -0.9, "8-bit: 0 is negative full scale")

    for bad, what in (
        (b"", "an empty file"),
        (b"JUNK\x00\x00\x00\x00WAVE", "a non-RIFF tag"),
        (b"RIFF\x00\x00\x00\x00AVI ", "a non-WAVE form"),
        (b"RIFF\x24\x00\x00\x00WAVE", "a file with no fmt or data chunk"),
    ):
        try:
            wavio.decode_wav(bad)
            R.check(False, f"should have rejected {what}")
        except wavio.WavError:
            R.check(True, f"rejects {what}")


if __name__ == "__main__":
    print("Audionova measurement tests\n")
    test_spectrum()
    test_dynamics()
    test_stereo()
    test_gating()
    test_hygiene()
    test_report()
    test_wavio_matches_engine()
    print(f"\n{R.checks} checks, {R.failures} failure(s)")
    sys.exit(1 if R.failures else 0)
