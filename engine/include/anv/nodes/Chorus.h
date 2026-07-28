// Chorus / doubler — multi-voice modulated delay.
//
// Several taps on a short delay line, each swept by its own LFO, summed. The
// pitch wobble that modulation produces is the effect: it detunes each copy
// slightly and differently, which is what makes one source sound like several
// players rather than one player through a delay.
//
// THE LFO IS A PURE FUNCTION OF ABSOLUTE TIME, not an accumulator. Writing it
// as `phase += rate/fs` each sample would work and would also make the output
// depend on where block boundaries fell, because floating-point accumulation
// is not associative. Computing it from the global frame index instead makes
// block-size invariance structural — the same property the whole engine is
// built on.
//
// Voices are spread in quadrature around the LFO cycle, and the left and
// right channels run at opposite phase, which is what gives the effect width
// without any Haas-style delay trickery.
//
// Latency is 0: the base delay is the effect, exactly as with the delay node.
// Declaring it would have the chain compensate the chorus away.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "anv/core/Node.h"
#include "anv/core/Rng.h"
#include "anv/dsp/FractionalDelay.h"

namespace anv {

class Chorus final : public Node {
public:
  const char* type() const override { return "chorus"; }

  void prepare(const PrepareSpec& spec) override {
    spec_ = spec;
    rebuild();
  }

  void reset() override {
    for (auto& l : lines_) l.reset();
  }

  void setParams(const ParamSet& p) override {
    voices_ = static_cast<int>(std::clamp(p.num("voices", 3.0), 1.0, 6.0));
    rateHz_ = std::clamp(p.num("rate_hz", 0.6), 0.01, 12.0);
    depthMs_ = std::clamp(p.num("depth_ms", 2.0), 0.0, 20.0);
    baseDelayMs_ = std::clamp(p.num("base_delay_ms", 12.0), 1.0, 100.0);
    stereoPhaseDeg_ = std::clamp(p.num("stereo_phase_deg", 180.0), 0.0, 360.0);
    // Deterministic either way — the offset comes from the recipe seed, not a
    // clock. Turning it off makes the starting delay exactly the base delay,
    // which is what lets the contract suite pin the latency exactly.
    randomizePhase_ = p.num("randomize_phase", 1.0) > 0.5;

    if (spec_.numChannels > 0) rebuild();
  }

  int latencyFrames() const override { return 0; }

  int64_t tailFrames() const override { return maxDelaySamples_; }

  void process(ProcessContext& ctx) noexcept override {
    const int n = ctx.numFrames;
    const int ch = spec_.numChannels;
    const double invSr = 1.0 / spec_.sampleRate;
    const double norm = 1.0 / static_cast<double>(voices_);

    for (int c = 0; c < ch; ++c) {
      auto& line = lines_[static_cast<size_t>(c)];
      const double chPhase =
          (c % 2 == 1) ? (stereoPhaseDeg_ / 360.0) : 0.0;

      for (int i = 0; i < n; ++i) {
        line.write(ctx.in[c][i]);

        // Absolute position, so any block partition gives the same answer.
        const double t = static_cast<double>(ctx.globalFrameIndex + i) * invSr;
        double acc = 0.0;
        for (int v = 0; v < voices_; ++v) {
          const double phase = phase0_[static_cast<size_t>(v)] + chPhase +
                               rateHz_ * t;
          const double lfo = std::sin(2.0 * M_PI * phase);
          acc += line.read(baseSamples_ + depthSamples_ * lfo);
        }
        ctx.out[c][i] = acc * norm;
      }
    }
  }

private:
  void rebuild() {
    const int ch = spec_.numChannels;
    if (ch <= 0) return;
    const double sr = spec_.sampleRate;

    baseSamples_ = baseDelayMs_ * 0.001 * sr;
    depthSamples_ = depthMs_ * 0.001 * sr;
    // The line has to hold the deepest excursion plus interpolation headroom.
    maxDelaySamples_ =
        static_cast<int>(std::ceil(baseSamples_ + depthSamples_)) + 8;

    // Voices spread evenly around the cycle. The seeded offset keeps two
    // chorus instances in one chain from locking together, while staying
    // reproducible because it derives from the recipe seed.
    phase0_.assign(static_cast<size_t>(voices_), 0.0);
    CounterRng rng(spec_.seed);
    const double jitter = randomizePhase_ ? rng.uniform(0, 0) : 0.0;
    for (int v = 0; v < voices_; ++v)
      phase0_[static_cast<size_t>(v)] =
          static_cast<double>(v) / static_cast<double>(voices_) + jitter;

    lines_.assign(static_cast<size_t>(ch), ModulatedDelayLine{});
    for (auto& l : lines_) l.prepare(maxDelaySamples_);

    reset();
  }

  int voices_ = 3;
  double rateHz_ = 0.6, depthMs_ = 2.0, baseDelayMs_ = 12.0;
  double stereoPhaseDeg_ = 180.0;
  bool randomizePhase_ = true;

  double baseSamples_ = 0.0, depthSamples_ = 0.0;
  int maxDelaySamples_ = 0;
  std::vector<double> phase0_;
  std::vector<ModulatedDelayLine> lines_;
};

} // namespace anv
