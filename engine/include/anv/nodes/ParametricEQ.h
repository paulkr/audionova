// Minimum-phase parametric EQ — the default for all tonal moves.
//
// Minimum phase rather than linear phase is deliberate: it is analog-like, has
// no pre-ringing, and is what an engineer's ear expects from a tone control.
// Linear-phase and offline-only zero-phase variants exist as separate nodes
// for surgical and mastering work, where their pre-ringing tradeoff is
// justified and their latency is declared.
//
// Latency is exactly 0. An IIR cascade has group delay but no bulk delay, and
// reporting anything else would corrupt plugin delay compensation.
//
// The high-pass is built from cascaded SVF sections so the slope is a real
// 12/18/24 dB/oct Butterworth-ish rolloff rather than the shallow first-order
// filter that cheaper implementations ship.
#pragma once

#include <string>
#include <vector>

#include "anv/core/Node.h"
#include "anv/dsp/TptSvf.h"

namespace anv {

class ParametricEQ final : public Node {
public:
  const char* type() const override { return "parametric_eq"; }

  void prepare(const PrepareSpec& spec) override {
    spec_ = spec;
    build();
  }

  void reset() override {
    for (auto& perChannel : sections_)
      for (auto& f : perChannel) f.reset();
  }

  void setParams(const ParamSet& p) override {
    params_ = p;
    if (spec_.numChannels > 0) build();
  }

  void process(ProcessContext& ctx) noexcept override {
    for (int c = 0; c < spec_.numChannels; ++c) {
      const Sample* src = ctx.in[c];
      Sample* dst = ctx.out[c];
      auto& chain = sections_[static_cast<size_t>(c)];
      for (int i = 0; i < ctx.numFrames; ++i) {
        Sample x = src[i];
        for (auto& f : chain) x = f.processSample(x);
        dst[i] = x;
      }
    }
  }

  int latencyFrames() const override { return 0; }

private:
  static FilterShape shapeFromString(const std::string& s) {
    if (s == "lowshelf") return FilterShape::LowShelf;
    if (s == "highshelf") return FilterShape::HighShelf;
    if (s == "notch") return FilterShape::Notch;
    if (s == "lowpass") return FilterShape::LowPass;
    if (s == "highpass") return FilterShape::HighPass;
    return FilterShape::Bell;
  }

  // A cascade of N identical 2nd-order sections gives 12*N dB/oct. Q values
  // are the Butterworth pole Qs for the cascaded order.
  static std::vector<double> butterworthQs(int order) {
    std::vector<double> qs;
    const int sections = order / 2;
    for (int k = 0; k < sections; ++k) {
      const double theta = M_PI * (2.0 * k + 1.0) / (2.0 * order);
      qs.push_back(1.0 / (2.0 * std::cos(theta)));
    }
    return qs;
  }

  void build() {
    sections_.assign(static_cast<size_t>(spec_.numChannels), {});

    for (int c = 0; c < spec_.numChannels; ++c) {
      auto& chain = sections_[static_cast<size_t>(c)];

      const double hpHz = params_.num("hp_freq_hz", 0.0);
      if (hpHz > 0.0) {
        int slope = static_cast<int>(params_.num("hp_slope_db_oct", 12.0));
        if (slope < 12) slope = 12;
        const int order = (slope / 12) * 2; // 12 -> 2nd, 24 -> 4th, 36 -> 6th
        for (double q : butterworthQs(order)) {
          TptSvf f;
          f.setSampleRate(spec_.sampleRate);
          f.configure(FilterShape::HighPass, hpHz, 0.0, q);
          chain.push_back(f);
        }
      }

      const double lpHz = params_.num("lp_freq_hz", 0.0);
      if (lpHz > 0.0) {
        int slope = static_cast<int>(params_.num("lp_slope_db_oct", 12.0));
        if (slope < 12) slope = 12;
        const int order = (slope / 12) * 2;
        for (double q : butterworthQs(order)) {
          TptSvf f;
          f.setSampleRate(spec_.sampleRate);
          f.configure(FilterShape::LowPass, lpHz, 0.0, q);
          chain.push_back(f);
        }
      }

      for (const ParamSet& band : params_.items()) {
        TptSvf f;
        f.setSampleRate(spec_.sampleRate);
        f.configure(shapeFromString(band.str("filter_type", "bell")),
                    band.num("center_freq_hz", 1000.0),
                    clampDb(band.num("gain_db", 0.0)),
                    band.num("q_factor", 0.707));
        chain.push_back(f);
      }
    }
  }

  ParamSet params_;
  std::vector<std::vector<TptSvf>> sections_;
};

} // namespace anv
