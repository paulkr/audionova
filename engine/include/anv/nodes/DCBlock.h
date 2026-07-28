// DC blocker — second-order high-pass at 10 Hz.
//
// DC offset is not audible on its own but it is corrosive: it eats headroom
// that the limiter then has to give back, it biases every nonlinearity so a
// symmetric saturator starts generating even harmonics it was never asked
// for, and it makes true-peak measurement report a level the programme does
// not actually have.
//
// Second order rather than first so the 10 Hz corner is genuinely out of the
// way — a first-order blocker at 10 Hz is still 3 dB down at 10 Hz and only
// 6 dB/oct below, which noticeably thins a bass-heavy source. Latency is 0.
#pragma once

#include <algorithm>

#include "anv/core/Node.h"
#include "anv/dsp/TptSvf.h"

namespace anv {

class DCBlock final : public Node {
public:
  const char* type() const override { return "dc_block"; }

  void prepare(const PrepareSpec& spec) override {
    spec_ = spec;
    rebuild();
  }

  void reset() override {
    for (auto& f : filters_) f.reset();
  }

  void setParams(const ParamSet& p) override {
    cutoffHz_ = std::clamp(p.num("cutoff_hz", 10.0), 1.0, 200.0);
    if (spec_.numChannels > 0) rebuild();
  }

  int latencyFrames() const override { return 0; }

  void process(ProcessContext& ctx) noexcept override {
    for (int c = 0; c < spec_.numChannels; ++c) {
      const Sample* src = ctx.in[c];
      Sample* dst = ctx.out[c];
      auto& f = filters_[static_cast<size_t>(c)];
      for (int i = 0; i < ctx.numFrames; ++i) dst[i] = f.processSample(src[i]);
    }
  }

private:
  void rebuild() {
    const int ch = spec_.numChannels;
    if (ch <= 0) return;
    filters_.assign(static_cast<size_t>(ch), TptSvf{});
    for (auto& f : filters_) {
      f.setSampleRate(spec_.sampleRate);
      f.configure(FilterShape::HighPass, cutoffHz_, 0.0, 0.7071067811865476);
    }
    reset();
  }

  double cutoffHz_ = 10.0;
  std::vector<TptSvf> filters_;
};

} // namespace anv
