// A node whose only purpose is to declare and honour a fixed latency.
//
// This exists so the plugin-delay-compensation path is exercised by the
// contract suite from the very first commit. Without at least one
// latency-bearing node in the registry, the bypass null test and the latency
// test are VACUOUS — they pass trivially because there is no latency in the
// system to get wrong.
//
// It is also the reference implementation the real latency-bearing nodes
// (oversampled saturation, lookahead limiter, linear-phase crossover,
// partitioned convolution) are checked against.
#pragma once

#include <algorithm>

#include "anv/core/Node.h"
#include "anv/dsp/DelayLine.h"

namespace anv {

class ReferenceDelay final : public Node {
public:
  const char* type() const override { return "reference_delay"; }

  void prepare(const PrepareSpec& spec) override {
    spec_ = spec;
    line_.prepare(spec.numChannels, latency_);
  }

  void reset() override { line_.reset(); }

  void setParams(const ParamSet& p) override {
    // Bounded like every other node, even though this one exists only to
    // declare a latency. Declared latency sizes buffers all the way up through
    // the chain's compensation and its head trim, so an unbounded value here
    // is the same resource-exhaustion vector as an unbounded lookahead — and
    // this node is the reference the real latency-bearing nodes are checked
    // against, so it should not be the one that breaks the rule.
    latency_ = static_cast<int>(
        std::clamp(p.num("latency_frames", 64.0), 0.0, 48000.0));
    if (spec_.numChannels > 0) line_.prepare(spec_.numChannels, latency_);
  }

  int latencyFrames() const override { return latency_; }

  void process(ProcessContext& ctx) noexcept override {
    line_.process(ctx.in, ctx.out, ctx.numFrames);
  }

private:
  int latency_ = 64;
  DelayLine line_;
};

} // namespace anv
