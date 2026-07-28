// Distribute one channel across all channels.
//
// NAMED FOR WHAT IT DOES, not for what it sounds like it should do. A node
// cannot currently change the channel COUNT of a chain: Chain carries one
// channel count for its whole length, fixed at prepare time. So this does not
// turn a 1-channel graph into a 2-channel graph — it takes the source channel
// of an already-stereo graph and copies it to the others.
//
// That is the case that actually comes up: a mono source loaded into a stereo
// session, where the right channel would otherwise be silent. Genuine
// channel-count changes need graph support that does not exist yet, and
// pretending otherwise here would produce a node that silently does nothing
// in the situation its name implies.
//
// Explicit rather than implicit, deliberately. Silent upmixing is how a mono
// source ends up 3 dB louder than intended after summing, and how a stereo
// source quietly loses its right channel.
#pragma once

#include <algorithm>

#include "anv/core/Node.h"

namespace anv {

class MonoToStereo final : public Node {
public:
  const char* type() const override { return "mono_to_stereo"; }

  void prepare(const PrepareSpec& spec) override { spec_ = spec; }
  void reset() override {} // stateless

  void setParams(const ParamSet& p) override {
    sourceChannel_ = static_cast<int>(
        std::clamp(p.num("source_channel", 0.0), 0.0, 15.0));
  }

  void process(ProcessContext& ctx) noexcept override {
    const int n = ctx.numFrames;
    const int ch = spec_.numChannels;
    const int src = std::min(sourceChannel_, ch - 1);
    if (ch <= 0) return;

    for (int c = 0; c < ch; ++c)
      std::copy(ctx.in[src], ctx.in[src] + n, ctx.out[c]);
  }

private:
  int sourceChannel_ = 0;
};

} // namespace anv
