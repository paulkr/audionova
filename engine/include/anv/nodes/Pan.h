// Constant-power pan.
//
// gL = cos(theta), gR = sin(theta) with theta sweeping 0..pi/2, so
// gL^2 + gR^2 = 1 at every position. A source keeps the same apparent
// loudness as it moves across the image, which a linear pan law does not: a
// linear law dips 3 dB in the middle and the source appears to duck as it
// passes centre.
//
// The consequence engineers expect is that CENTRE IS -3 dB on each side, not
// unity. That is not a bug to be corrected — it is what keeps the power
// constant, and it is why summing a centred track to mono gives back unity.
//
// With fewer than two channels there is nothing to pan between, so the node
// passes through untouched rather than applying a centre attenuation to a
// mono signal that has nowhere to go.
#pragma once

#include <algorithm>
#include <cmath>

#include "anv/core/Node.h"

namespace anv {

class Pan final : public Node {
public:
  const char* type() const override { return "pan"; }

  void prepare(const PrepareSpec& spec) override { spec_ = spec; }
  void reset() override {} // stateless

  void setParams(const ParamSet& p) override {
    position_ = std::clamp(p.num("pan_position", 0.0), -1.0, 1.0);
    const double theta = (position_ + 1.0) * (M_PI * 0.25);
    gainL_ = std::cos(theta);
    gainR_ = std::sin(theta);
  }

  void process(ProcessContext& ctx) noexcept override {
    const int n = ctx.numFrames;
    const int ch = spec_.numChannels;

    if (ch < 2) {
      for (int c = 0; c < ch; ++c)
        std::copy(ctx.in[c], ctx.in[c] + n, ctx.out[c]);
      return;
    }

    for (int i = 0; i < n; ++i) {
      ctx.out[0][i] = ctx.in[0][i] * gainL_;
      ctx.out[1][i] = ctx.in[1][i] * gainR_;
    }
    // Anything beyond the first pair is left alone rather than silently
    // folded into the stereo image.
    for (int c = 2; c < ch; ++c)
      std::copy(ctx.in[c], ctx.in[c] + n, ctx.out[c]);
  }

private:
  double position_ = 0.0, gainL_ = 0.7071067811865476, gainR_ = 0.7071067811865476;
};

} // namespace anv
