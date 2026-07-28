// Mid/side encode and decode.
//
//   M = (L + R) / sqrt(2)      L = (M + S) / sqrt(2)
//   S = (L - R) / sqrt(2)      R = (M - S) / sqrt(2)
//
// The 1/sqrt(2) rather than the more common 1/2 makes the transform
// ENERGY-PRESERVING and its own inverse: encode followed by decode returns
// the input exactly, with no gain to remember to undo. A 1/2-on-encode,
// 1-on-decode convention round-trips too, but each direction changes level,
// so any chain that encodes, processes and decodes has a hidden 6 dB in it.
//
// Round-tripping to a null is a property the contract suite checks, because
// mid/side is almost always used as a bracket around other processing and a
// bracket that does not close is worse than no bracket at all.
#pragma once

#include <algorithm>
#include <string>

#include "anv/core/Node.h"

namespace anv {

class MidSide final : public Node {
public:
  const char* type() const override { return "mid_side"; }

  void prepare(const PrepareSpec& spec) override { spec_ = spec; }
  void reset() override {} // stateless

  // `mode` is recorded for readability only and does NOT change the maths.
  // With the 1/sqrt(2) scaling the butterfly is its own inverse, so encode and
  // decode are literally the same operation. Keeping the field means a recipe
  // reads as "encode ... process ... decode" rather than the same opaque node
  // appearing twice, which matters when the chain is shown to an engineer.
  void setParams(const ParamSet& p) override {
    decode_ = (p.str("mode", "encode") == "decode");
    (void)decode_;
  }

  void process(ProcessContext& ctx) noexcept override {
    const int n = ctx.numFrames;
    const int ch = spec_.numChannels;

    // Needs a pair to work on; anything else passes through rather than
    // silently doing something arbitrary.
    if (ch < 2) {
      for (int c = 0; c < ch; ++c)
        std::copy(ctx.in[c], ctx.in[c] + n, ctx.out[c]);
      return;
    }

    constexpr double kInvSqrt2 = 0.70710678118654752440;
    for (int i = 0; i < n; ++i) {
      const double a = ctx.in[0][i];
      const double b = ctx.in[1][i];
      // Encode and decode are the same butterfly; only the naming differs.
      ctx.out[0][i] = (a + b) * kInvSqrt2;
      ctx.out[1][i] = (a - b) * kInvSqrt2;
    }
    for (int c = 2; c < ch; ++c)
      std::copy(ctx.in[c], ctx.in[c] + n, ctx.out[c]);
  }

private:
  bool decode_ = false;
};

} // namespace anv
