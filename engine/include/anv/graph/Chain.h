// Serial effect chain with exact plugin delay compensation.
//
// Three behaviours here are load-bearing and deliberately live in the engine
// rather than in individual nodes:
//
// 1. A BYPASSED NODE STILL REPORTS ITS LATENCY, and the chain inserts an equal
//    compensating delay. This is why the bypass null test is the highest-value
//    test in the suite: it exercises the exact same PDC path a live render
//    uses, so a latency bug fails a test that is trivially easy to interpret
//    instead of showing up later as "it sounds phasey".
//
// 2. DRY/WET IS IMPLEMENTED BY THE CHAIN, NOT THE NODE. The dry path is
//    delayed by exactly the node's latency before the crossfade. This makes
//    phase-correct parallel processing universally right and impossible for a
//    node author to get wrong.
//
// 3. TOTAL LATENCY IS AN OUTPUT, NEVER AUTHORED. The renderer feeds
//    inputFrames + totalLatency + tail, then trims exactly totalLatency from
//    the head.
#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "anv/core/AudioBuffer.h"
#include "anv/core/Node.h"
#include "anv/core/Rng.h"
#include "anv/dsp/DelayLine.h"

namespace anv {

class Chain {
public:
  struct SlotConfig {
    std::string nodeId = "n1";
    bool bypassed = false;
    double mix = 1.0; // 0 = fully dry, 1 = fully wet
  };

  void add(std::unique_ptr<Node> node, const ParamSet& params, SlotConfig cfg) {
    Slot s;
    s.node = std::move(node);
    s.params = params;
    s.cfg = std::move(cfg);
    slots_.push_back(std::move(s));
  }

  void add(std::unique_ptr<Node> node, const ParamSet& params) {
    add(std::move(node), params, SlotConfig{});
  }

  size_t size() const { return slots_.size(); }

  void setBypassed(size_t index, bool bypassed) {
    slots_[index].cfg.bypassed = bypassed;
  }

  void prepare(const PrepareSpec& spec) {
    spec_ = spec;
    const int ch = spec.numChannels;
    const int maxBlock = spec.maxBlockFrames;

    for (auto& s : slots_) {
      PrepareSpec nodeSpec = spec;
      nodeSpec.seed = deriveNodeSeed(spec.seed, s.cfg.nodeId.c_str());
      s.node->prepare(nodeSpec);
      s.node->setParams(s.params);

      const int lat = s.node->latencyFrames();
      s.compDelay.prepare(ch, lat);
      s.dryDelay.prepare(ch, lat);
      s.dryScratch.resize(ch, maxBlock);
    }

    scratch_[0].resize(ch, maxBlock);
    scratch_[1].resize(ch, maxBlock);
    inView_.resize(static_cast<size_t>(ch));
    outView_.resize(static_cast<size_t>(ch));
    dryView_.resize(static_cast<size_t>(ch));
  }

  void reset() {
    for (auto& s : slots_) {
      s.node->reset();
      s.compDelay.reset();
      s.dryDelay.reset();
    }
    scratch_[0].clear();
    scratch_[1].clear();
  }

  int totalLatencyFrames() const {
    int total = 0;
    for (const auto& s : slots_) total += s.node->latencyFrames();
    return total;
  }

  int64_t tailFrames() const {
    int64_t total = 0;
    for (const auto& s : slots_)
      if (!s.cfg.bypassed) total += s.node->tailFrames();
    return total;
  }

  void processBlock(const Sample* const* in, Sample* const* out, int n,
                    int64_t globalFrameIndex) noexcept {
    const int ch = spec_.numChannels;

    // Seed the ping-pong with the incoming block.
    for (int c = 0; c < ch; ++c)
      std::copy(in[c], in[c] + n, scratch_[0].channel(c));

    int src = 0;
    for (auto& s : slots_) {
      const int dst = 1 - src;

      for (int c = 0; c < ch; ++c) {
        inView_[static_cast<size_t>(c)] = scratch_[src].channel(c);
        outView_[static_cast<size_t>(c)] = scratch_[dst].channel(c);
        dryView_[static_cast<size_t>(c)] = s.dryScratch.channel(c);
      }

      if (s.cfg.bypassed) {
        // Pass through, but honour the declared latency so the chain's total
        // latency — and therefore the head trim — is identical to a live run.
        s.compDelay.process(inView_.data(), outView_.data(), n);
      } else {
        const bool blend = s.cfg.mix < 0.999999;
        if (blend) {
          for (int c = 0; c < ch; ++c)
            std::copy(inView_[static_cast<size_t>(c)],
                      inView_[static_cast<size_t>(c)] + n,
                      dryView_[static_cast<size_t>(c)]);
          // Align the dry path with the wet path's latency before mixing.
          s.dryDelay.process(dryView_.data(), dryView_.data(), n);
        }

        ProcessContext ctx;
        ctx.in = inView_.data();
        ctx.out = outView_.data();
        ctx.numFrames = n;
        ctx.globalFrameIndex = globalFrameIndex;
        s.node->process(ctx);

        if (blend) {
          const double w = s.cfg.mix;
          const double d = 1.0 - w;
          for (int c = 0; c < ch; ++c) {
            Sample* o = outView_[static_cast<size_t>(c)];
            const Sample* dry = dryView_[static_cast<size_t>(c)];
            for (int i = 0; i < n; ++i) o[i] = w * o[i] + d * dry[i];
          }
        }
      }

      src = dst;
    }

    for (int c = 0; c < ch; ++c)
      std::copy(scratch_[src].channel(c), scratch_[src].channel(c) + n, out[c]);
  }

  // Offline whole-signal render. Feeds input plus enough silence to flush the
  // chain's latency and tail, then trims exactly the total latency from the
  // head so the result is sample-aligned with the input.
  void render(const AudioBuffer& input, AudioBuffer& output, int blockSize) {
    const int ch = spec_.numChannels;
    const int64_t n = input.frames();
    const int latency = totalLatencyFrames();
    const int64_t tail = tailFrames();
    const int64_t padded = n + latency + tail;

    AudioBuffer feed(ch, padded);
    for (int c = 0; c < ch; ++c)
      std::copy(input.channel(c), input.channel(c) + n, feed.channel(c));

    AudioBuffer raw(ch, padded);

    std::vector<const Sample*> blockIn(static_cast<size_t>(ch));
    std::vector<Sample*> blockOut(static_cast<size_t>(ch));

    for (int64_t pos = 0; pos < padded; pos += blockSize) {
      const int nn = static_cast<int>(std::min<int64_t>(blockSize, padded - pos));
      for (int c = 0; c < ch; ++c) {
        blockIn[static_cast<size_t>(c)] = feed.channel(c) + pos;
        blockOut[static_cast<size_t>(c)] = raw.channel(c) + pos;
      }
      processBlock(blockIn.data(), blockOut.data(), nn, pos);
    }

    output.resize(ch, n + tail);
    for (int c = 0; c < ch; ++c)
      std::copy(raw.channel(c) + latency, raw.channel(c) + latency + n + tail,
                output.channel(c));
  }

private:
  struct Slot {
    std::unique_ptr<Node> node;
    ParamSet params;
    SlotConfig cfg;
    DelayLine compDelay;
    DelayLine dryDelay;
    AudioBuffer dryScratch;
  };

  PrepareSpec spec_{};
  std::vector<Slot> slots_;
  AudioBuffer scratch_[2];
  std::vector<const Sample*> inView_;
  std::vector<Sample*> outView_;
  std::vector<Sample*> dryView_;
};

} // namespace anv
