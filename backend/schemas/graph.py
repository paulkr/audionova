"""The effect graph: the planner's output contract.

ONE DECLARATION, THREE CONSUMERS. Every parameter is declared exactly once in
`PARAMS` below, and three things are derived from that declaration:

  1. a Pydantic model that ENFORCES the range
  2. a JSON Schema handed to the model API, which describes the structure
  3. the human-readable range text inside each field's `description`

The third is not redundant, and this is the single most important thing in this
file:

    CLAUDE STRUCTURED OUTPUTS DO NOT ENFORCE minimum/maximum.

`enum`, `const`, `anyOf` and `additionalProperties: false` are honoured. Numeric
bounds are not. A schema that reads like it constrains `gain_db` to plus or minus
18 does nothing of the kind on the wire — the model may emit 400 and the API will
pass it through. So the bound has to appear in the DESCRIPTION, where the model
will actually read it, and be enforced separately in Python afterwards.

That means the same number exists in two places, which is exactly the setup where
things drift. Generating both from one declaration removes the possibility rather
than testing for it.

THE SCHEMA IS DELIBERATELY NARROWER THAN THE ENGINE. The engine's compressor
accepts twelve parameters; this exposes eight. `os_profile`, `freeze`,
`makeup_mode` and similar are omitted not because they are broken but because
every parameter offered to a planner is a parameter it can get wrong, and the
ones left out have sensible defaults that an engineer would rarely move. Ship
fewer knobs, each provably useful.

Parameter names and ranges MIRROR the engine exactly — `engine/include/anv/nodes/`
is the authority, and a test asserts that every name and node type here exists
there. A schema that drifts from the engine produces graphs that validate and
then fail to render, or worse, render with a silently defaulted parameter.
"""

from __future__ import annotations

import copy
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "effect_graph_v1"


# ---------------------------------------------------------------------------
# The single source of truth.
#
# (unit, minimum, maximum, default, description)
#
# Ranges are the ENGINE's clamps, or tighter where a tighter range is what an
# engineer would actually use. Where this file is tighter than the engine that
# is deliberate: the engine clamps defensively for any caller, while this bounds
# what a planner is allowed to ask for.

PARAMS: dict[str, tuple[str, float, float, float, str]] = {
    # gain
    "gain_db": ("dB", -24.0, 24.0, 0.0, "Level change"),
    # pan
    "pan_position": ("", -1.0, 1.0, 0.0, "-1 hard left, 0 centre, +1 hard right"),
    # dc_block
    "cutoff_hz": ("Hz", 1.0, 40.0, 10.0, "Corner of the DC-blocking high-pass"),
    # parametric_eq / dynamic_eq bands
    "center_freq_hz": ("Hz", 20.0, 20000.0, 1000.0, "Band centre or corner frequency"),
    "q_factor": ("", 0.1, 12.0, 0.707,
                 "Bandwidth. 0.707 for shelves and filters; 0.7-2 for musical "
                 "bells; above 4 is surgical and rarely what a record needs"),
    "band_gain_db": ("dB", -18.0, 18.0, 0.0,
                     "Band gain. Ignored by highpass, lowpass and notch"),
    # linear_phase_eq
    "taps": ("", 127.0, 8191.0, 2047.0,
             "FIR length, forced odd. Latency is exactly (taps-1)/2 samples and "
             "cost is linear in taps, so this is the one knob that decides "
             "whether the node is cheap. Frequency resolution is 96000/taps Hz: "
             "2047 taps resolves about 47 Hz, so a narrow move below 100 Hz "
             "needs a large value or, better, parametric_eq"),
    # compressor
    "threshold_dbfs": ("dBFS", -60.0, 0.0, -18.0,
                       "Level above which gain reduction begins. Set from the "
                       "measured short-term RMS percentiles, not from habit"),
    "ratio_to_1": ("", 1.0, 20.0, 3.0, "Compression ratio, expressed as N:1"),
    "knee_db": ("dB", 0.0, 24.0, 6.0, "Soft-knee width. 0 is a hard knee"),
    "attack_ms": ("ms", 0.1, 200.0, 10.0,
                  "Time to reach most of the gain reduction. Derive from the "
                  "measured envelope modulation spectrum where possible"),
    "release_ms": ("ms", 5.0, 3000.0, 120.0, "Time to recover after the signal falls"),
    "lookahead_ms": ("ms", 0.0, 50.0, 5.0,
                     "Free offline, so overshoot is zero. 5 ms is a sane default"),
    "sidechain_hp_hz": ("Hz", 0.0, 500.0, 0.0,
                        "High-pass on the detector only, so bass stops driving "
                        "the gain reduction. 0 disables it"),
    "stereo_link": ("", 0.0, 1.0, 1.0,
                    "1 links the channels so the image cannot wander; 0 is "
                    "fully independent"),
    "makeup_gain_db": ("dB", -24.0, 24.0, 0.0, "Applied after compression"),
    # multiband_compressor bands
    "crossover_hz": ("Hz", 40.0, 18000.0, 1000.0,
                     "Upper edge of this band. List bands lowest first; the "
                     "last band's value is not read"),
    # transient_designer
    "attack_db": ("dB", -18.0, 18.0, 0.0,
                  "Gain on a fully-developed transient. Positive sharpens the "
                  "attack, negative softens it"),
    "sustain_db": ("dB", -18.0, 18.0, 0.0,
                   "Gain on the body after the transient. Positive lengthens "
                   "the tail, negative tightens it"),
    "transient_attack_ms": ("ms", 1.0, 200.0, 20.0,
                            "How long a transient is taken to last"),
    "sustain_ms": ("ms", 50.0, 2000.0, 250.0,
                   "How long the body after a transient is taken to last; "
                   "roughly the decay time of the material"),
    # gate_expander
    "range_db": ("dB", 0.0, 80.0, 60.0, "Maximum attenuation when closed"),
    "hold_ms": ("ms", 0.0, 2000.0, 50.0, "Minimum time held open after opening"),
    "hysteresis_db": ("dB", 0.0, 24.0, 3.0,
                      "Gap between the open and close thresholds, which stops "
                      "chatter on material sitting near the threshold"),
    # drive
    "drive_db": ("dB", 0.0, 30.0, 6.0, "Input drive into the saturator"),
    "output_trim_db": ("dB", -24.0, 24.0, 0.0,
                       "Applied after saturation. Usually about half the drive, "
                       "negative, to keep the level roughly constant"),
    "bias": ("", -1.0, 1.0, 0.0,
             "Asymmetry. Non-zero adds even-order harmonics, which reads as "
             "valve-like rather than transistor-like"),
    # delay
    "time_ms": ("ms", 1.0, 2000.0, 250.0,
                "Delay time. Prefer time_note_division when the measured tempo "
                "is confident, so repeats land musically"),
    "feedback_percent": ("%", 0.0, 90.0, 30.0, "Proportion fed back for repeats"),
    "hf_damp_hz": ("Hz", 1000.0, 20000.0, 8000.0,
                   "Low-pass in the feedback path, so repeats darken"),
    "lf_damp_hz": ("Hz", 20.0, 800.0, 100.0, "High-pass in the feedback path"),
    "spread_ms": ("ms", 0.0, 60.0, 0.0, "Left/right offset for stereo modes"),
    # reverbs
    "decay_s": ("s", 0.1, 12.0, 1.8, "RT60. A plate cannot decay faster than ~0.5 s"),
    "pre_delay_ms": ("ms", 0.0, 250.0, 0.0,
                     "Gap before the reverb starts, which keeps the dry source "
                     "intelligible"),
    "damping_hz": ("Hz", 500.0, 20000.0, 6000.0, "Corner of the in-tank damping"),
    "bandwidth_hz": ("Hz", 500.0, 20000.0, 12000.0, "Input bandwidth into the tank"),
    "size": ("", 0.4, 2.0, 1.0, "Tank size multiplier"),
    "modulation_depth": ("", 0.0, 1.0, 0.6,
                         "Breaks up metallic ringing. Reduce for sustained "
                         "material where pitch movement is audible"),
    "hf_cut_hz": ("Hz", 500.0, 20000.0, 6000.0, "High-frequency damping"),
    "lf_cut_hz": ("Hz", 20.0, 800.0, 120.0, "Low cut on the reverb send"),
    "hf_decay_ratio": ("", 0.1, 1.0, 0.45,
                       "High-frequency decay relative to the low. Below 1 makes "
                       "the tail darken as it fades, which is what a real room does"),
    "stereo_decorrelation": ("", 0.0, 1.0, 1.0, "0 is mono reverb, 1 fully decorrelated"),
    # chorus
    "voices": ("", 1.0, 6.0, 3.0, "Number of modulated copies"),
    "rate_hz": ("Hz", 0.01, 12.0, 0.6, "LFO rate"),
    "depth_ms": ("ms", 0.0, 20.0, 2.0, "Modulation depth"),
    "base_delay_ms": ("ms", 1.0, 100.0, 12.0, "Centre delay the modulation moves around"),
    "stereo_phase_deg": ("deg", 0.0, 360.0, 180.0,
                         "LFO phase offset between channels. 180 gives width"),
    # stereo_width
    "width_percent": ("%", 0.0, 250.0, 100.0,
                      "100 is unchanged, 0 is mono, above 100 exaggerates the "
                      "side signal"),
    "bass_mono_below_hz": ("Hz", 0.0, 400.0, 0.0,
                           "Collapse below this to mono. Low-frequency side "
                           "content costs headroom and partially cancels on a "
                           "single speaker. 0 disables the crossover"),
    "bass_width_percent": ("%", 0.0, 250.0, 0.0, "Width applied below the crossover"),
    # true_peak_limiter
    "ceiling_dbtp": ("dBTP", -12.0, 0.0, -1.0,
                     "True-peak ceiling at the DELIVERY rate. -1.0 dBTP is the "
                     "streaming convention"),
    "limiter_release_ms": ("ms", 5.0, 1000.0, 100.0, "Limiter recovery time"),
    "safety_margin_db": ("dB", 0.0, 3.0, 0.0, "Extra headroom below the ceiling"),
    # dither
    "bit_depth": ("bits", 0.0, 32.0, 24.0,
                  "Target word length. 0 or 32 means no reduction, and float "
                  "exports must use 0 — dither on a float file only adds noise"),
    "amount_lsb": ("LSB", 0.0, 4.0, 2.0,
                   "Peak-to-peak dither amplitude. 2 LSB TPDF is the standard "
                   "and weakest fully-effective choice"),
    # dynamic_eq bands
    "static_gain_db": ("dB", -18.0, 18.0, 0.0, "Always-applied band gain"),
    "dynamic_range_db": ("dB", -18.0, 18.0, 12.0,
                         "Maximum dynamic movement. Negative compresses the "
                         "band, positive expands it"),
}


def _field(name: str, **overrides: Any) -> Any:
    """Build a Pydantic Field from the declaration, with the range in the text.

    The range is appended to the description because the API will not enforce
    `ge`/`le`. If it lived only in the JSON Schema's numeric keywords the model
    would never see it.
    """
    unit, low, high, default, text = PARAMS[name]
    suffix = f" ({unit})" if unit else ""
    return Field(
        default=overrides.pop("default", default),
        ge=overrides.pop("ge", low),
        le=overrides.pop("le", high),
        description=f"{text}{suffix}. Range {low} to {high}.",
        **overrides,
    )


class _Strict(BaseModel):
    # additionalProperties: false IS honoured by the API, so an unknown key is
    # caught at decode time rather than silently ignored on the way to a node
    # that never reads it.
    model_config = ConfigDict(extra="forbid")


FilterType = Literal["bell", "lowshelf", "highshelf", "notch", "lowpass", "highpass"]


class EqBand(_Strict):
    filter_type: FilterType = "bell"
    center_freq_hz: float = _field("center_freq_hz")
    gain_db: float = _field("band_gain_db")
    q_factor: float = _field("q_factor")

    @model_validator(mode="after")
    def _gain_is_meaningful(self) -> "EqBand":
        # A gain on a filter that has no gain is not an error, but it is a sign
        # the model misunderstood the band type, so it is normalised to zero
        # rather than passed through to be ignored.
        if self.filter_type in ("highpass", "lowpass", "notch"):
            object.__setattr__(self, "gain_db", 0.0)
        return self


class DynamicEqBand(_Strict):
    filter_type: FilterType = "bell"
    mode: Literal["static", "downward", "upward"] = "static"
    center_freq_hz: float = _field("center_freq_hz")
    q_factor: float = _field("q_factor", default=1.0)
    static_gain_db: float = _field("static_gain_db")
    threshold_dbfs: float = _field("threshold_dbfs", default=-24.0)
    ratio_to_1: float = _field("ratio_to_1", default=4.0)
    range_db: float = _field("dynamic_range_db")
    attack_ms: float = _field("attack_ms", default=5.0)
    release_ms: float = _field("release_ms", default=80.0)
    knee_db: float = _field("knee_db", default=3.0)


# ---------------------------------------------------------------------------
# Nodes


class GainParams(_Strict):
    gain_db: float = _field("gain_db")


class PanParams(_Strict):
    pan_position: float = _field("pan_position")


class PolarityParams(_Strict):
    inverted: bool = Field(default=True, description="Invert the polarity")


class DcBlockParams(_Strict):
    cutoff_hz: float = _field("cutoff_hz")


class MidSideParams(_Strict):
    mode: Literal["encode", "decode"] = "encode"


class ParametricEqParams(_Strict):
    bands: list[EqBand] = Field(
        default_factory=list, max_length=8,
        description="EQ bands, applied in series. At most 8.",
    )


class ZeroPhaseEqParams(_Strict):
    """Forward-backward EQ. Same bands, twice the dB, no phase shift at all.

    The gain doubling is not a quirk to work around: filtering twice gives
    |H|^2, so a band written as 3 dB lands as 6. Write half of what is wanted.
    """

    bands: list[EqBand] = Field(
        default_factory=list, max_length=8,
        description="EQ bands. The applied response is the SQUARE of these — a "
                    "3 dB band produces 6 dB — because the filter runs forwards "
                    "and then backwards. Halve the gain you want. At most 8.",
    )


class LinearPhaseEqParams(_Strict):
    taps: float = _field("taps")
    bands: list[EqBand] = Field(
        default_factory=list, max_length=8,
        description="EQ bands, applied in series. At most 8.",
    )


class DynamicEqParams(_Strict):
    bands: list[DynamicEqBand] = Field(
        default_factory=list, max_length=4,
        description="Dynamic EQ bands. At most 4.",
    )


class CompressorParams(_Strict):
    threshold_dbfs: float = _field("threshold_dbfs")
    ratio_to_1: float = _field("ratio_to_1")
    knee_db: float = _field("knee_db")
    attack_ms: float = _field("attack_ms")
    release_ms: float = _field("release_ms")
    lookahead_ms: float = _field("lookahead_ms")
    sidechain_hp_hz: float = _field("sidechain_hp_hz")
    stereo_link: float = _field("stereo_link")
    makeup_gain_db: float = _field("makeup_gain_db")
    detector: Literal["peak", "rms"] = "peak"


class MultibandBand(_Strict):
    """One band of the multiband compressor. `crossover_hz` is its UPPER edge."""

    crossover_hz: float = _field("crossover_hz")
    threshold_dbfs: float = _field("threshold_dbfs")
    ratio_to_1: float = _field("ratio_to_1", default=2.0)
    knee_db: float = _field("knee_db")
    attack_ms: float = _field("attack_ms")
    release_ms: float = _field("release_ms")
    makeup_gain_db: float = _field("makeup_gain_db")


class MultibandCompressorParams(_Strict):
    """Compression per band, over a crossover that is exact at unity.

    Only when one part of the spectrum misbehaves and the rest is already
    right. Every band at 1:1 with 0 dB makeup returns the input unchanged, so
    bands left alone cost nothing.
    """

    bands: list[MultibandBand] = Field(
        default_factory=list, max_length=4,
        description="Two to four bands, lowest first.",
    )
    lookahead_ms: float = _field("lookahead_ms")
    stereo_link: float = _field("stereo_link")


class TransientDesignerParams(_Strict):
    """Attack and sustain shaping from the difference between two envelopes.

    Not a compressor: the gain follows how fast the level is CHANGING, so there
    is no threshold and a quiet hit is shaped like a loud one. Broadband
    detection, so sustained low bass is modulated by its own waveform — send
    bass-led material to multiband_compressor instead.
    """

    attack_db: float = _field("attack_db")
    sustain_db: float = _field("sustain_db")
    attack_ms: float = _field("transient_attack_ms")
    sustain_ms: float = _field("sustain_ms")
    lookahead_ms: float = _field("lookahead_ms", default=2.0)
    stereo_link: float = _field("stereo_link")


class GateExpanderParams(_Strict):
    threshold_dbfs: float = _field("threshold_dbfs", default=-40.0)
    ratio_to_1: float = _field("ratio_to_1", default=4.0)
    range_db: float = _field("range_db")
    attack_ms: float = _field("attack_ms", default=1.0)
    hold_ms: float = _field("hold_ms")
    release_ms: float = _field("release_ms", default=100.0)
    lookahead_ms: float = _field("lookahead_ms", default=2.0)
    hysteresis_db: float = _field("hysteresis_db")
    knee_db: float = _field("knee_db", default=3.0)
    sidechain_hp_hz: float = _field("sidechain_hp_hz")
    stereo_link: float = _field("stereo_link")


class DriveParams(_Strict):
    drive_db: float = _field("drive_db")
    output_trim_db: float = _field("output_trim_db")
    bias: float = _field("bias")
    curve: Literal["soft_clip", "hard_clip"] = "soft_clip"


class DelayParams(_Strict):
    time_ms: float = _field("time_ms")
    time_note_division: Literal[
        "", "sixteenth", "eighth", "dotted_eighth", "triplet_eighth", "quarter"
    ] = Field(default="",
              description="Musical division. Overrides time_ms when set and a "
                          "confident tempo is available. Empty means use time_ms.")
    tempo_bpm: float = Field(default=0.0, ge=0.0, le=300.0,
                             description="Required if time_note_division is set. "
                                         "Range 0 to 300.")
    feedback_percent: float = _field("feedback_percent")
    hf_damp_hz: float = _field("hf_damp_hz")
    lf_damp_hz: float = _field("lf_damp_hz")
    stereo_mode: Literal["mono", "stereo_offset", "ping_pong"] = "mono"
    spread_ms: float = _field("spread_ms")


class PlateReverbParams(_Strict):
    decay_s: float = _field("decay_s", default=2.0)
    pre_delay_ms: float = _field("pre_delay_ms")
    damping_hz: float = _field("damping_hz")
    bandwidth_hz: float = _field("bandwidth_hz")
    size: float = _field("size")
    modulation_depth: float = _field("modulation_depth")


class ConvolutionReverbParams(_Strict):
    decay_s: float = _field("decay_s", default=1.5)
    pre_delay_ms: float = _field("pre_delay_ms")
    hf_damp_hz: float = _field("hf_cut_hz")
    lf_cut_hz: float = _field("lf_cut_hz")
    hf_decay_ratio: float = _field("hf_decay_ratio")
    stereo_decorrelation: float = _field("stereo_decorrelation")


class ChorusParams(_Strict):
    voices: float = _field("voices")
    rate_hz: float = _field("rate_hz")
    depth_ms: float = _field("depth_ms")
    base_delay_ms: float = _field("base_delay_ms")
    stereo_phase_deg: float = _field("stereo_phase_deg")


class StereoWidthParams(_Strict):
    width_percent: float = _field("width_percent")
    bass_mono_below_hz: float = _field("bass_mono_below_hz")
    bass_width_percent: float = _field("bass_width_percent")


class TruePeakLimiterParams(_Strict):
    ceiling_dbtp: float = _field("ceiling_dbtp")
    lookahead_ms: float = _field("lookahead_ms", default=5.0)
    release_ms: float = _field("limiter_release_ms")
    safety_margin_db: float = _field("safety_margin_db")


class DitherParams(_Strict):
    bit_depth: float = _field("bit_depth")
    amount_lsb: float = _field("amount_lsb")
    noise_shaping: Literal["none", "first_order"] = "none"


class MonoToStereoParams(_Strict):
    source_channel: float = Field(default=0.0, ge=0.0, le=15.0,
                                  description="Channel to duplicate. Range 0 to 15.")


# The closed set. A node type absent from here cannot be emitted, which is the
# whole point: a planner naming an effect that does not exist must fail loudly
# rather than ship a render missing the thing the user asked for.
PARAM_MODELS: dict[str, type[_Strict]] = {
    "gain": GainParams,
    "pan": PanParams,
    "polarity": PolarityParams,
    "dc_block": DcBlockParams,
    "mono_to_stereo": MonoToStereoParams,
    "mid_side": MidSideParams,
    "parametric_eq": ParametricEqParams,
    "zero_phase_eq": ZeroPhaseEqParams,
    "linear_phase_eq": LinearPhaseEqParams,
    "dynamic_eq": DynamicEqParams,
    "compressor": CompressorParams,
    "multiband_compressor": MultibandCompressorParams,
    "transient_designer": TransientDesignerParams,
    "gate_expander": GateExpanderParams,
    "drive": DriveParams,
    "delay": DelayParams,
    "plate_reverb": PlateReverbParams,
    "convolution_reverb": ConvolutionReverbParams,
    "chorus": ChorusParams,
    "stereo_width": StereoWidthParams,
    "true_peak_limiter": TruePeakLimiterParams,
    "dither": DitherParams,
}

# Only correct after the output sample-rate conversion. Mirrors the engine's
# `isMasterOnly`; a test asserts the two agree.
MASTER_ONLY = ("true_peak_limiter", "dither")

NodeType = Literal[
    "gain", "pan", "polarity", "dc_block", "mono_to_stereo", "mid_side",
    "parametric_eq", "zero_phase_eq", "linear_phase_eq", "dynamic_eq",
    "compressor", "multiband_compressor", "transient_designer",
    "gate_expander", "drive", "delay", "plate_reverb",
    "convolution_reverb", "chorus", "stereo_width", "true_peak_limiter",
    "dither",
]


class GraphNode(_Strict):
    id: Annotated[str, Field(
        min_length=1, max_length=40, pattern=r"^[a-z0-9_]+$",
        description="Stable lowercase identifier, unique within the graph. This "
                    "is what a later refinement names, so it must not change "
                    "when parameters do.",
    )]
    type: NodeType
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for this node type. See the per-type schemas.",
    )
    mix: float = Field(default=1.0, ge=0.0, le=1.0,
                       description="0 fully dry, 1 fully wet. Range 0 to 1.")
    bypass: bool = Field(default=False, description="Keep the node but do not apply it")
    rationale: str = Field(
        default="", max_length=280,
        description="One sentence on why this node is here, in terms of the "
                    "measurement report. Shown to the user.",
    )

    @model_validator(mode="after")
    def _validate_params(self) -> "GraphNode":
        # The params dict is typed loosely in the schema so the API sees one node
        # shape, then validated against the per-type model here. Doing it the
        # other way — a discriminated union of eighteen node shapes — produces a
        # schema large enough to hurt the prompt cache for no benefit.
        model = PARAM_MODELS[self.type]
        validated = model.model_validate(self.params)
        object.__setattr__(self, "params", validated.model_dump())
        return self


class EffectGraph(_Strict):
    schema_version: Literal["effect_graph_v1"] = "effect_graph_v1"
    nodes: list[GraphNode] = Field(
        default_factory=list, max_length=12,
        description="Chain at the internal rate, in the order they are applied. "
                    "Order matters and is part of the answer.",
    )
    master: list[GraphNode] = Field(
        default_factory=list, max_length=3,
        description="Runs AFTER the output sample-rate conversion. Only "
                    "true_peak_limiter and dither belong here, in that order.",
    )
    explanation: str = Field(
        # 1200 was too tight and rejected good graphs. Once the planning stage
        # started producing a full analysis rather than a truncated one, the
        # transcriber wrote a correspondingly fuller summary and blew the cap on
        # two turns out of three — a graph thrown away over prose length.
        #
        # The number is stated in the description as well, because maxLength is
        # in the same category as minimum and maximum: it appears in the schema,
        # it is NOT enforced on the wire, and the only place the model reliably
        # reads a bound is the text.
        default="", max_length=2400,
        description="Plain-language summary for the user, at most 2400 "
                    "characters — a few paragraphs, not the full analysis. "
                    "Never name an artist, band or commercial recording.",
    )

    @model_validator(mode="after")
    def _structural(self) -> "EffectGraph":
        seen: set[str] = set()
        for node in list(self.nodes) + list(self.master):
            if node.id in seen:
                raise ValueError(
                    f"duplicate node id {node.id!r} — ids must be unique so a "
                    "refinement can name exactly one node"
                )
            seen.add(node.id)

        for node in self.nodes:
            if node.type in MASTER_ONLY:
                raise ValueError(
                    f"{node.type!r} must be in master, not nodes — it is only "
                    "correct after the output sample-rate conversion"
                )
        for node in self.master:
            if node.type not in MASTER_ONLY:
                raise ValueError(
                    f"{node.type!r} does not belong in master; only "
                    f"{', '.join(MASTER_ONLY)} do"
                )
        return self

    def to_recipe(self, *, seed: int = 0, output_sample_rate: float = 0.0,
                  output_format: str = "pcm24",
                  internal_sample_rate: float = 96000.0) -> dict[str, Any]:
        """Convert to the engine's recipe JSON.

        The graph carries no rate or format: those are delivery decisions the
        planner has no business making, and keeping them out means the same graph
        can be rendered as a 25-second excerpt for audition and a full-length
        24-bit master without being regenerated.
        """
        def node(entry: GraphNode) -> dict[str, Any]:
            out: dict[str, Any] = {"id": entry.id, "type": entry.type,
                                   "params": entry.params}
            if entry.mix != 1.0:
                out["mix"] = entry.mix
            if entry.bypass:
                out["bypass"] = True
            return out

        return {
            "engine_version": "0.1.0",
            "internal_sample_rate": internal_sample_rate,
            "seed": seed,
            "output": {"sample_rate": output_sample_rate, "format": output_format},
            "nodes": [node(n) for n in self.nodes],
            "master": [node(n) for n in self.master],
        }


# ---------------------------------------------------------------------------
# Schema generation


def json_schema() -> dict[str, Any]:
    """The JSON Schema handed to the model API.

    Numeric bounds are stripped from the emitted schema on purpose. The API does
    not enforce them, so leaving them in would suggest a guarantee that does not
    exist — and every bound is already stated in the description text, which the
    model does read. Enforcement happens in Python, against the same declaration.
    """
    schema = _dereference(EffectGraph.model_json_schema())
    _strip_unenforced(schema)

    # SAY WHICH FIELDS ARE REQUIRED, or the model is told a graph with nothing
    # in it satisfies the schema. Pydantic omits `required` entirely here
    # because every field has a default — sensible for a Python constructor,
    # actively harmful as a tool schema.
    #
    # Measured: with no `required` and `additionalProperties: false`, the model
    # returned a single top-level key literally named "$PARAMETER_VALUE" with
    # the whole graph nested inside it — a filled-in template rather than an
    # answer. It cost a retry on every planning turn, which read as ordinary
    # schema friction rather than as a schema defect.
    #
    # `master` stays optional: a graph with no master section is legitimate, and
    # validation adds the delivery ceiling. `schema_version` has exactly one
    # allowed value and nothing is gained by making the model type it.
    schema["required"] = ["nodes", "explanation"]
    return schema


def _dereference(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline every `$ref`, leaving a schema with no `$defs`.

    THE MODEL CANNOT SEE THROUGH A `$ref`. Pydantic factors repeated
    definitions — `GraphNode`, `EqBand`, every params model — into `$defs` and
    points at them, which is correct JSON Schema and the right thing for a
    validator. Handed to the API as a tool's `input_schema` it is not resolved,
    so the model is shown properties whose shapes are references it cannot
    follow, and it does what anyone does when asked to fill in a form with no
    field names: it invents them.

    The symptom was a tool call whose top-level keys were `$PARAMETER_NAME` and
    `$PARAMETER_VALUE` — the literal placeholders — with the real graph
    sometimes nested one level down under a made-up key. It cost a retry on
    most turns and a total failure on some, and it read as ordinary schema
    friction rather than as the schema being unreadable.

    Measured after inlining: three trials for three, top-level keys came back
    as `['explanation', 'nodes', 'master']` with no envelope and no
    placeholders.

    Costs about 50% more characters, since a definition used twice is now
    written twice. That is the whole price, and it buys a schema the model can
    actually read.
    """
    defs = schema.get("$defs", {})

    def walk(node: Any, seen: tuple[str, ...] = ()) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].rsplit("/", 1)[-1]
                if name not in defs:
                    raise KeyError(f"schema references unknown definition {name!r}")
                if name in seen:
                    # A genuinely recursive schema cannot be inlined. Nothing
                    # here is recursive today; this refuses rather than looping
                    # forever if that ever changes.
                    raise ValueError(
                        f"cannot inline recursive definition {name!r}; the "
                        f"model API needs a flat schema and this one cycles")
                resolved = walk(copy.deepcopy(defs[name]), seen + (name,))
                resolved.update({k: v for k, v in node.items() if k != "$ref"})
                return resolved
            return {k: walk(v, seen) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(v, seen) for v in node]
        return node

    flat = walk(copy.deepcopy(schema))
    flat.pop("$defs", None)
    return flat


def _strip_unenforced(node: Any) -> None:
    if isinstance(node, dict):
        for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
            node.pop(key, None)
        for value in node.values():
            _strip_unenforced(value)
    elif isinstance(node, list):
        for item in node:
            _strip_unenforced(item)


def param_schemas() -> dict[str, dict[str, Any]]:
    """Per-node-type parameter schemas, for the prompt's reference section."""
    out = {}
    for name, model in PARAM_MODELS.items():
        schema = model.model_json_schema()
        _strip_unenforced(schema)
        out[name] = schema
    return out
