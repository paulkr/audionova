// Polarity inversion.
//
// Trivial, and worth having as its own node rather than a flag on Gain: a
// polarity flip is a decision an engineer makes for a reason (phase
// relationship against another mic, a mis-wired cable) and it should be
// visible in the chain as its own step rather than hidden inside a gain
// value of -1.
//
// Exactly negation — no gain change, so it is bit-reversible.
#pragma once

#include <algorithm>

#include "anv/core/Node.h"

namespace anv {

class Polarity final : public Node {
public:
  const char* type() const override { return "polarity"; }

  void prepare(const PrepareSpec& spec) override { spec_ = spec; }
  void reset() override {} // stateless

  void setParams(const ParamSet& p) override {
    inverted_ = p.num("inverted", 1.0) > 0.5;
  }

  void process(ProcessContext& ctx) noexcept override {
    const int n = ctx.numFrames;
    for (int c = 0; c < spec_.numChannels; ++c) {
      const Sample* src = ctx.in[c];
      Sample* dst = ctx.out[c];
      if (inverted_) {
        for (int i = 0; i < n; ++i) dst[i] = -src[i];
      } else {
        std::copy(src, src + n, dst);
      }
    }
  }

private:
  bool inverted_ = true;
};

} // namespace anv
