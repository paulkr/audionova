"""The measurement report: everything the planner is told about the audio.

THIS IS THE HIGHEST-LEVERAGE ARTEFACT IN THE SYSTEM. The research finding the
whole design rests on is that a model given measured DSP features scored 0.22
on EQ matching where the same model guided by audio embeddings scored 0.55 —
worse than random parameters. The model is never asked to imagine what the
audio sounds like. It is handed numbers.

Three rules govern everything here, and they are what separate a report that
helps from one that misleads:

  UNITS LIVE IN THE KEY NAME. ``threshold_db`` and ``attack_ms``, never
  ``threshold`` and ``attack``. A model reading a bare number will assume a
  unit, and it will sometimes assume the wrong one. This costs nothing and
  removes an entire class of silent error.

  EVERY ESTIMATE CARRIES A CONFIDENCE. Blind reverb time, noise floor and
  bandwidth limit are inferences, not observations, and they are unreliable on
  exactly the material where they matter most. The prompt instructs the model
  to disregard anything below 0.4, and a validator enforces that low-confidence
  values never drive a decision.

  THE REPORT ENDS IN ``warnings``. Some facts require reading three fields and
  knowing what their combination implies — a bandwidth limit at 15 kHz means an
  air-band boost does nothing; a 0.4% clipping rate means drive will make it
  worse. Leaving that inference to the model is leaving it to chance. The
  warnings are computed deterministically and stated as instructions.

NOTHING FROM THE FILE'S METADATA REACHES THIS REPORT. ID3 tags and Vorbis
comments are attacker-controlled text on a path that ends in a prompt. The
report describes samples, and only samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backend.audio import wavio
from backend.measure import dynamics, gating, hygiene, loudness, spectrum, stereo

ANALYZER_VERSION = "1.0.0"


def _finite(value: float, fallback: Any = None) -> Any:
    """JSON has no -inf. A missing measurement must read as missing, not as a
    very large negative number that arithmetic will happily consume."""
    return float(value) if np.isfinite(value) else fallback


def _round(value: Any, digits: int = 2) -> Any:
    if value is None or not isinstance(value, (int, float)):
        return value
    return round(float(value), digits) if np.isfinite(value) else None


@dataclass
class MeasurementReport:
    source: dict[str, Any] = field(default_factory=dict)
    loudness: dict[str, Any] = field(default_factory=dict)
    dynamics: dict[str, Any] = field(default_factory=dict)
    spectrum: dict[str, Any] = field(default_factory=dict)
    stereo: dict[str, Any] = field(default_factory=dict)
    hygiene: dict[str, Any] = field(default_factory=dict)
    warnings: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "analyzer_version": ANALYZER_VERSION,
            "source": self.source,
            "loudness": self.loudness,
            "dynamics": self.dynamics,
            "spectrum": self.spectrum,
            "stereo": self.stereo,
            "hygiene": self.hygiene,
            "warnings": self.warnings,
        }


def measure(samples: np.ndarray, sample_rate: float, *, bit_depth: int = 24,
            is_float: bool = False) -> MeasurementReport:
    samples = np.atleast_2d(np.asarray(samples, dtype=np.float64))
    channels, frames = samples.shape
    report = MeasurementReport()

    mask, coverage = gating.gated_frame_mask(samples, sample_rate)

    report.source = {
        "channels": int(channels),
        "sample_rate_hz": float(sample_rate),
        "duration_s": _round(frames / sample_rate if sample_rate else 0.0, 3),
        "bit_depth": int(bit_depth),
        "is_float": bool(is_float),
        # Coverage is diagnostic, not bookkeeping. Every averaged number below
        # rests on this fraction of the file, and a report over 12% of a stem
        # deserves to be read differently from one over 95%.
        "gated_coverage_percent": _round(100.0 * coverage, 1),
    }

    # --- loudness -----------------------------------------------------------
    integrated = loudness.integrated_lufs(samples, sample_rate)
    true_peak = loudness.true_peak_dbtp(samples, sample_rate)
    sample_peak = loudness.sample_peak_dbfs(samples)

    report.loudness = {
        "integrated_lufs": _round(_finite(integrated)),
        "loudness_range_lu": _round(loudness.loudness_range_lu(samples, sample_rate)),
        "true_peak_dbtp": _round(_finite(true_peak)),
        "sample_peak_dbfs": _round(_finite(sample_peak)),
        # Spelled out because it is the number people assume is a bug.
        "mono_convention_note": (
            "BS.1770 sums weighted channel powers, so a mono file measures "
            "3 LU below the same waveform as dual-mono stereo."
        ) if channels == 1 else None,
    }

    # --- dynamics -----------------------------------------------------------
    rms_db = dynamics.short_term_rms_db(samples, sample_rate, mask)
    mod_freqs, mod_power = dynamics.envelope_modulation_spectrum(samples, sample_rate, mask)
    erb_centres, erb_crests = dynamics.per_erb_crest_db(samples, sample_rate, mask)

    report.dynamics = {
        "crest_factor_db": _round(dynamics.crest_factor_db(samples, mask)),
        "peak_to_loudness_ratio_db": _round(
            dynamics.peak_to_loudness_ratio_db(true_peak, integrated)
        ),
        "short_term_rms_percentiles": {
            key: _round(value) for key, value in dynamics.level_percentiles_db(rms_db).items()
        },
        "per_erb_crest_db": [
            {"centre_hz": _round(float(f), 1), "crest_db": _round(float(c))}
            for f, c in zip(erb_centres, erb_crests)
        ],
        "envelope_modulation": {
            key: _round(value, 3) for key, value in dynamics.modulation_summary(mod_freqs, mod_power).items()
        },
    }

    # --- spectrum -----------------------------------------------------------
    freqs, power = spectrum.gated_ltas(samples, sample_rate, mask)
    third_centres, third_levels = spectrum.third_octave_levels_db(freqs, power, sample_rate)
    erb_spec_centres, erb_levels = spectrum.erb_levels_db(freqs, power, sample_rate)
    tilt, tilt_residual = spectrum.spectral_tilt_db_per_octave(third_centres, third_levels)
    limit_hz, limit_confidence = spectrum.bandwidth_limit_hz(freqs, power, sample_rate)

    report.spectrum = {
        # A band with no measurable energy reads as null, not as -120 dB. A
        # substituted constant is indistinguishable from a measurement once it
        # is in the report, and it is the kind of number a planner will
        # cheerfully do arithmetic on.
        "third_octave_db": [
            {"centre_hz": _round(float(f), 1), "level_db": _round(_finite(float(l)))}
            for f, l in zip(third_centres, third_levels)
        ],
        "erb_db": [
            {"centre_hz": _round(float(f), 1), "level_db": _round(_finite(float(l)))}
            for f, l in zip(erb_spec_centres, erb_levels)
        ],
        # Named for its reference, because the sign convention is genuinely
        # ambiguous and a planner that assumes the other one inverts every
        # tonal move it makes. 0 = pink (equal energy per octave, what an
        # engineer calls neutral); +3 = white; negative = darker than pink.
        "tilt_db_per_octave_re_pink": _round(tilt),
        # Without this, a scooped-midrange source and a genuinely flat one both
        # report a tilt near zero and are indistinguishable. RMS deviation from
        # the fitted line, in dB — see spectral_tilt_db_per_octave for why not
        # r-squared, which degenerates to zero on exactly the flat spectra that
        # are best described by a line.
        "tilt_residual_db": _round(tilt_residual),
        "bandwidth_limit_hz": _round(limit_hz, 0),
        "bandwidth_limit_confidence": _round(limit_confidence),
    }

    # --- stereo -------------------------------------------------------------
    field_type, field_confidence = stereo.stereo_type(samples, mask)
    report.stereo = {
        "type": field_type,
        "type_confidence": _round(field_confidence),
        "correlation": _round(stereo.broadband_correlation(samples, mask)),
        "mono_loss_db": _round(stereo.mono_compatibility_loss_db(samples, mask)),
        "per_band": [
            {key: _round(value) if isinstance(value, float) else value
             for key, value in band.items()}
            for band in stereo.per_band_stereo(samples, sample_rate, mask)
        ],
    }

    # --- hygiene ------------------------------------------------------------
    clip = hygiene.clipping(samples)
    floor_db, floor_confidence = hygiene.noise_floor_db(samples, sample_rate)
    report.hygiene = {
        "clipping": {key: _round(value, 4) for key, value in clip.items()},
        "dc_offset_dbfs": [_round(_finite(v, -140.0)) for v in hygiene.dc_offset(samples)],
        "noise_floor_dbfs": _round(_finite(floor_db)),
        "noise_floor_confidence": _round(floor_confidence),
        "dead_channels": hygiene.dead_channels(samples),
        "non_finite_samples": hygiene.nan_or_inf_count(samples),
    }

    report.warnings = _build_warnings(report)
    return report


def _build_warnings(report: MeasurementReport) -> list[dict[str, str]]:
    """Facts that require combining fields, stated as instructions.

    Deterministic and ordered by severity. Every entry names what was measured
    AND what to do about it, because a warning the reader has to interpret is a
    warning the reader can interpret wrongly.
    """
    warnings: list[dict[str, str]] = []

    def warn(code: str, severity: str, message: str) -> None:
        warnings.append({"code": code, "severity": severity, "message": message})

    hyg = report.hygiene
    spec = report.spectrum
    st = report.stereo
    dyn = report.dynamics
    src = report.source

    if hyg.get("non_finite_samples", 0):
        warn("non_finite_samples", "critical",
             f"{hyg['non_finite_samples']} samples are NaN or infinite. The file is "
             "damaged; do not process it.")

    clipped = (hyg.get("clipping") or {}).get("clipped_sample_percent") or 0.0
    if clipped > 0.01:
        warn("clipped_source", "high",
             f"{clipped:.3f}% of samples sit in flat-topped runs at full scale. The "
             "source is already clipped: do not add drive or saturation, and expect "
             "any upward gain to expose the distortion further.")

    dead = hyg.get("dead_channels") or []
    if dead and src.get("channels", 1) > 1:
        warn("dead_channel", "high",
             f"Channel(s) {dead} contain no audio. Treat this as a mono source; "
             "stereo processing would hard-pan the result.")

    dc_values = [v for v in (hyg.get("dc_offset_dbfs") or []) if v is not None]
    if dc_values and max(dc_values) > -50.0:
        warn("dc_offset", "medium",
             f"DC offset at {max(dc_values):.1f} dBFS. Place a dc_block first; it "
             "otherwise consumes headroom and biases every level decision downstream.")

    limit = spec.get("bandwidth_limit_hz")
    limit_confidence = spec.get("bandwidth_limit_confidence") or 0.0
    if limit is not None and limit < 16000.0 and limit_confidence >= 0.4:
        warn("bandwidth_limited", "high",
             f"Content stops at about {limit:.0f} Hz (confidence {limit_confidence:.2f}), "
             "which indicates lossy encoding. Any shelf or boost above that frequency "
             "will be inaudible; do not spend a band on air.")

    floor = hyg.get("noise_floor_dbfs")
    floor_confidence = hyg.get("noise_floor_confidence") or 0.0
    if floor is not None and floor > -55.0 and floor_confidence >= 0.4:
        warn("high_noise_floor", "medium",
             f"Noise floor at {floor:.1f} dBFS (confidence {floor_confidence:.2f}). "
             "Upward expansion, high-frequency boosts and heavy makeup gain will all "
             "lift the noise into the foreground.")

    if st.get("type") == "out_of_phase":
        warn("out_of_phase", "critical",
             f"The channels are strongly negatively correlated ({st.get('correlation')}). "
             "One channel is probably polarity-inverted; fix that before any other "
             "decision, since the mono sum currently cancels.")
    elif (st.get("mono_loss_db") or 0.0) > 1.0:
        warn("mono_incompatible", "high",
             f"Summing to mono loses {st['mono_loss_db']:.1f} dB beyond the normal 3 dB. "
             "Do not widen further; consider narrowing the low bands.")

    if st.get("type") == "dual_mono":
        warn("dual_mono", "low",
             "Both channels are identical. Stereo width has nothing to work with "
             "until something decorrelates them.")

    plr = dyn.get("peak_to_loudness_ratio_db")
    if plr is not None and plr < 8.0:
        warn("already_compressed", "medium",
             f"Peak-to-loudness ratio is {plr:.1f} dB, which is mastered-record "
             "territory. The material has already had heavy dynamic processing; "
             "further compression will flatten it audibly.")

    coverage = src.get("gated_coverage_percent")
    if coverage is not None and coverage < 25.0:
        warn("sparse_material", "medium",
             f"Only {coverage:.1f}% of the file survived loudness gating. Every "
             "averaged measurement here rests on that fraction; treat the spectrum "
             "and dynamics figures as less certain than usual.")

    residual = spec.get("tilt_residual_db")
    if residual is not None and residual > 4.0:
        warn("tilt_unreliable", "low",
             f"Band levels deviate from the fitted tilt by {residual:.1f} dB RMS, so "
             "the source is not well described by a single slope. Use the band levels "
             "directly rather than the tilt figure.")

    if report.loudness.get("integrated_lufs") is None:
        warn("no_programme_material", "critical",
             "Nothing in this file survived the loudness gate. It is silent or "
             "near-silent; there is nothing to process.")

    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    warnings.sort(key=lambda w: order.get(w["severity"], 4))
    return warnings


def measure_file(path: str) -> MeasurementReport:
    audio = wavio.read_wav(path)
    return measure(
        audio.samples,
        audio.sample_rate,
        bit_depth=audio.bit_depth,
        is_float=audio.is_float,
    )


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 2:
        print("usage: python -m backend.measure.report <file.wav>", file=sys.stderr)
        sys.exit(2)
    print(json.dumps(measure_file(sys.argv[1]).to_dict(), indent=2))
