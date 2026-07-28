// Convolution reverb.
//
// The Phase 1 reverb, chosen over an algorithmic one for a specific reason:
// convolution is physically exact and trivially verifiable. Its output can be
// checked against direct time-domain convolution to the last bit, which means
// the whole node reduces to "is the convolution right?" plus "is the impulse
// response the one we intended?". An FDN reverb, by contrast, can only be
// judged by ear.
//
// IMPULSE RESPONSES ARE SYNTHESISED, not loaded. Every worthwhile IR library
// carries licensing that has to be cleared before it can ship in a commercial
// product, and shipping a node that depends on assets we do not yet have the
// rights to would block the whole engine. Synthesised IRs are parametric,
// deterministic, free of provenance questions, and good enough to validate
// the convolution engine. Loading measured IRs is an additive change later:
// the convolver does not care where its coefficients came from.
//
// The node outputs 100% WET. Dry/wet blending belongs to the chain, which
// already delays the dry path by exactly this node's latency — doing it here
// would duplicate that logic and invite it to drift.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "anv/core/Node.h"
#include "anv/core/Rng.h"
#include "anv/dsp/Convolver.h"
#include "anv/dsp/TptSvf.h"

namespace anv {

class ConvolutionReverb final : public Node {
public:
  const char* type() const override { return "convolution_reverb"; }

  void prepare(const PrepareSpec& spec) override {
    spec_ = spec;
    rebuild();
  }

  void reset() override {
    for (auto& c : conv_) c.reset();
  }

  void setParams(const ParamSet& p) override {
    // decay_s is the RT60 of the band BELOW hf_damp_hz, not the broadband
    // figure. With hf_decay_ratio < 1 the top end dies sooner, so a broadband
    // T30 measurement reads shorter — by ~10% at the default ratio, and by
    // less at both extremes (at a very fast HF ratio the high band is gone
    // before the -5 dB point and stops influencing the measurement at all).
    // Verified: with hf_decay_ratio = 1.0 the measured T30 matches decay_s to
    // within 0.1%. Anything matching a measured RT60 target must account for
    // this rather than assuming the parameter is the broadband figure.
    decayS_ = std::clamp(p.num("decay_s", 1.5), 0.05, 12.0);
    preDelayMs_ = std::clamp(p.num("pre_delay_ms", 0.0), 0.0, 250.0);
    hfDampHz_ = std::clamp(p.num("hf_damp_hz", 6000.0), 500.0, 20000.0);
    lfCutHz_ = std::clamp(p.num("lf_cut_hz", 120.0), 20.0, 800.0);
    // How much faster the top end decays than the body. Real rooms absorb
    // high frequencies far more quickly than low; a flat decay is the single
    // most obvious tell of a synthetic reverb.
    hfDecayRatio_ = std::clamp(p.num("hf_decay_ratio", 0.45), 0.1, 1.0);
    // Fully correlated L/R impulse responses collapse the reverb to the
    // centre. Decorrelating them is what makes it sound like a space.
    decorrelation_ = std::clamp(p.num("stereo_decorrelation", 1.0), 0.0, 1.0);

    if (spec_.numChannels > 0) rebuild();
  }

  int latencyFrames() const override {
    return conv_.empty() ? 0 : conv_[0].latencyFrames();
  }

  int64_t tailFrames() const override { return irLength_; }

  void process(ProcessContext& ctx) noexcept override {
    for (int c = 0; c < spec_.numChannels; ++c)
      conv_[static_cast<size_t>(c)].process(ctx.in[c], ctx.out[c], ctx.numFrames);
  }

private:
  // Two-band decaying noise. The low band decays at the requested RT60, the
  // high band proportionally faster, which is a crude but recognisable model
  // of frequency-dependent absorption.
  std::vector<Sample> synthesiseIr(uint64_t sharedSeed, uint64_t uniqueSeed) const {
    const double sr = spec_.sampleRate;
    const int preDelay = static_cast<int>(std::lround(preDelayMs_ * 0.001 * sr));
    const int body = static_cast<int>(std::lround(decayS_ * sr));
    const int len = preDelay + std::max(1, body);

    std::vector<Sample> ir(static_cast<size_t>(len), 0.0);

    // Continuous decorrelation, not a seed switch. Each channel's excitation
    // is sqrt(1-d) of a SHARED noise plus sqrt(d) of a noise unique to that
    // channel. Because the two components are independent and the weights are
    // energy-preserving, the resulting inter-channel correlation is exactly
    // (1 - d) — so the parameter means what its name says at every value, not
    // just at the endpoints.
    CounterRng shared(sharedSeed);
    CounterRng unique(uniqueSeed);
    const double wShared = std::sqrt(1.0 - decorrelation_);
    const double wUnique = std::sqrt(decorrelation_);

    TptSvf lp, hp, cut;
    lp.setSampleRate(sr);
    lp.configure(FilterShape::LowPass, hfDampHz_, 0.0, 0.707);
    hp.setSampleRate(sr);
    hp.configure(FilterShape::HighPass, hfDampHz_, 0.0, 0.707);
    cut.setSampleRate(sr);
    cut.configure(FilterShape::HighPass, lfCutHz_, 0.0, 0.707);

    // -60 dB is a factor of 1000, so the exponent constant is ln(1000).
    constexpr double kLn1000 = 6.907755278982137;
    const double kLow = kLn1000 / std::max(decayS_, 1e-6);
    const double kHigh = kLn1000 / std::max(decayS_ * hfDecayRatio_, 1e-6);

    for (int i = preDelay; i < len; ++i) {
      const double t = static_cast<double>(i - preDelay) / sr;
      const double n = wShared * shared.bipolar(0, i) + wUnique * unique.bipolar(0, i);
      const double low = lp.processSample(n) * std::exp(-kLow * t);
      const double high = hp.processSample(n) * std::exp(-kHigh * t);
      ir[static_cast<size_t>(i)] = cut.processSample(low + high);
    }

    // Unit energy, so changing the decay time does not change how loud the
    // reverb sits in the mix.
    double energy = 0.0;
    for (double v : ir) energy += v * v;
    if (energy > 0.0) {
      const double scale = 1.0 / std::sqrt(energy);
      for (double& v : ir) v *= scale;
    }
    return ir;
  }

  void rebuild() {
    const int ch = spec_.numChannels;
    if (ch <= 0) return;

    conv_.assign(static_cast<size_t>(ch), PartitionedConvolver{});
    irLength_ = 0;
    for (int c = 0; c < ch; ++c) {
      // Every channel draws on the same shared noise plus its own; the blend
      // weights above decide how much of each.
      const uint64_t uniqueSeed = splitmix64(
          spec_.seed ^ (0x9E3779B97F4A7C15ull * static_cast<uint64_t>(c + 1)));
      const std::vector<Sample> ir = synthesiseIr(spec_.seed, uniqueSeed);
      irLength_ = std::max<int64_t>(irLength_, static_cast<int64_t>(ir.size()));
      conv_[static_cast<size_t>(c)].prepare(ir);
    }
  }

  double decayS_ = 1.5, preDelayMs_ = 0.0, hfDampHz_ = 6000.0;
  double lfCutHz_ = 120.0, hfDecayRatio_ = 0.45, decorrelation_ = 1.0;
  int64_t irLength_ = 0;
  std::vector<PartitionedConvolver> conv_;
};

} // namespace anv
