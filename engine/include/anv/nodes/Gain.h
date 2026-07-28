// Static gain in dB. The simplest possible node, and therefore the one that
// proves the contract-test harness works before any real DSP is written.
#pragma once

#include <cmath>

#include "anv/core/Node.h"

namespace anv {

class Gain final : public Node {
public:
  const char* type() const override { return "gain"; }

  void prepare(const PrepareSpec& spec) override { spec_ = spec; }
  void reset() override {} // stateless

  void setParams(const ParamSet& p) override {
    gainDb_ = clampDb(p.num("gain_db", 0.0));
    linear_ = std::pow(10.0, gainDb_ / 20.0);
  }

  void process(ProcessContext& ctx) noexcept override {
    for (int c = 0; c < spec_.numChannels; ++c) {
      const Sample* src = ctx.in[c];
      Sample* dst = ctx.out[c];
      for (int i = 0; i < ctx.numFrames; ++i) dst[i] = src[i] * linear_;
    }
  }

private:
  double gainDb_ = 0.0;
  double linear_ = 1.0;
};

} // namespace anv
