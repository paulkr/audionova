// Saturation / drive — the house nonlinearity.
//
// Policy, applied to every nonlinear node in the engine: ADAA plus at least
// 4x oversampling. We are offline, so we oversample far harder than any
// real-time plugin can afford. This is the single biggest quality
// differentiator over generic "AI audio" processing, and it costs money rather
// than latency.
//
// LATENCY IS EXACTLY INTEGER, by construction:
//
//   ADAA second order           1  sample at the oversampled rate (centred)
//   integer pad             f - 1  samples
//   ------------------------------------------------------------------
//   total                       f  oversampled samples = 1 base sample
//
// so latencyFrames() = oversamplerLatency + 1, with no fractional component
// anywhere. Second-order ADAA is centred on x[n-1] rather than half a sample
// late, which is why no fractional-delay compensation is needed and why there
// is no high-frequency droop to apologise for.
#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "anv/core/Node.h"
#include "anv/dsp/Adaa.h"
#include "anv/dsp/Oversampler.h"

namespace anv {

class Drive final : public Node {
public:
  const char* type() const override { return "drive"; }

  void prepare(const PrepareSpec& spec) override {
    spec_ = spec;
    rebuild();
  }

  void reset() override {
    os_.reset();
    for (auto& s : softState_) s.reset();
    for (auto& s : clipState_) s.reset();
    for (auto& r : ring_) std::fill(r.begin(), r.end(), 0.0);
    std::fill(ringPos_.begin(), ringPos_.end(), 0);
  }

  void setParams(const ParamSet& p) override {
    driveDb_ = clampDb(p.num("drive_db", 6.0));
    trimDb_ = clampDb(p.num("output_trim_db", 0.0));
    bias_ = std::clamp(p.num("bias", 0.0), -1.0, 1.0);
    curve_ = p.str("curve", "soft_clip");
    // WIDEBAND IS THE DEFAULT, deliberately.
    //
    // The "audio" profile only protects 0-22 kHz and lets everything above
    // fold, which is inaudible but leaves roughly -20 dB of junk between
    // 22 and 48 kHz. That is invisible after an export to 44.1/48 kHz, but an
    // engineer who exports at 96 kHz and opens a spectrum analyser will see it
    // and rightly lose confidence.
    //
    // Protecting the full base band costs ~4x the filter taps (232 vs 56 at
    // 4x oversampling). That is nothing next to the economics: DSP is around
    // $0.0007 per render while a single planning call is $0.15-0.50, so
    // compute is not the constraint. Quality is the product.
    //
    // The orchestrator may opt into "audio" for bulk work when it knows both
    // the source and the delivery format are 48 kHz or below.
    profile_ = (p.str("os_profile", "wideband") == "audio") ? OsProfile::Audio
                                                            : OsProfile::Wideband;

    int f = static_cast<int>(p.num("oversample_factor", 4.0));
    // Power of two, minimum 4. 4x is the floor rather than the default
    // because that is where measured in-band aliasing crosses -85 dB on the
    // pathological test tone; see tests/contract_tests.cpp test 11.
    f = std::clamp(f, 4, 16);
    int pow2 = 4;
    while (pow2 * 2 <= f) pow2 *= 2;
    factor_ = pow2;

    driveLin_ = std::pow(10.0, driveDb_ / 20.0);
    trimLin_ = std::pow(10.0, trimDb_ / 20.0);
    softShaper_.setBias(bias_);
    clipShaper_.setBias(bias_);

    if (spec_.numChannels > 0) rebuild();
  }

  int latencyFrames() const override { return latency_; }

  void process(ProcessContext& ctx) noexcept override {
    const int n = ctx.numFrames;
    const int f = factor_;
    const bool hard = (curve_ == "hard_clip");

    for (int c = 0; c < spec_.numChannels; ++c) {
      // Apply drive at the base rate, before oversampling, so the shaper sees
      // the intended level.
      Sample* pre = pre_[static_cast<size_t>(c)].data();
      const Sample* src = ctx.in[c];
      for (int i = 0; i < n; ++i) pre[i] = src[i] * driveLin_;

      Sample* w = os_.up(pre, n, c);
      const int hi = n * f;

      if (hard) {
        auto& st = clipState_[static_cast<size_t>(c)];
        for (int i = 0; i < hi; ++i) w[i] = st.process(clipShaper_, w[i]);
      } else {
        auto& st = softState_[static_cast<size_t>(c)];
        for (int i = 0; i < hi; ++i) w[i] = st.process(softShaper_, w[i]);
      }

      // Pad ADAA2's one-sample delay out to a whole base-rate sample, so the
      // node's total latency is an exact integer.
      if (f > 1) {
        auto& ring = ring_[static_cast<size_t>(c)];
        int& pos = ringPos_[static_cast<size_t>(c)];
        const int rlen = static_cast<int>(ring.size());
        for (int i = 0; i < hi; ++i) {
          ring[static_cast<size_t>(pos)] = w[i];
          int a = pos - (f - 1);
          if (a < 0) a += rlen;
          w[i] = ring[static_cast<size_t>(a)];
          pos = (pos + 1) % rlen;
        }
      }

      os_.down(ctx.out[c], n, c);

      Sample* dst = ctx.out[c];
      for (int i = 0; i < n; ++i) dst[i] *= trimLin_;
    }
  }

private:
  void rebuild() {
    const int ch = spec_.numChannels;
    if (ch <= 0) return;

    os_.prepare(factor_, ch, spec_.maxBlockFrames, profile_, 120.0);
    latency_ = os_.latencyFrames() + 1;

    pre_.assign(static_cast<size_t>(ch),
                std::vector<Sample>(static_cast<size_t>(spec_.maxBlockFrames), 0.0));
    ring_.assign(static_cast<size_t>(ch),
                 std::vector<Sample>(static_cast<size_t>(factor_) + 1, 0.0));
    ringPos_.assign(static_cast<size_t>(ch), 0);
    softState_.resize(static_cast<size_t>(ch));
    clipState_.resize(static_cast<size_t>(ch));

    reset();
  }

  double driveDb_ = 6.0, trimDb_ = 0.0, bias_ = 0.0;
  double driveLin_ = 1.0, trimLin_ = 1.0;
  std::string curve_ = "soft_clip";
  OsProfile profile_ = OsProfile::Wideband;
  int factor_ = 4;
  int latency_ = 0;

  Oversampler os_;
  SoftClipShaper softShaper_;
  HardClipShaper clipShaper_;
  std::vector<Adaa2<SoftClipShaper>> softState_;
  std::vector<Adaa2<HardClipShaper>> clipState_;
  std::vector<std::vector<Sample>> pre_;
  std::vector<std::vector<Sample>> ring_;
  std::vector<int> ringPos_;
};

} // namespace anv
