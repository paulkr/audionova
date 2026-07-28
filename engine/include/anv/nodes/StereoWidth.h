// Stereo width, with an optional band-split so the bass can stay mono.
//
// Width works on the side signal: L = M + S*w, R = M - S*w. At w = 1 that is
// the identity, at 0 it collapses to mono, above 1 it exaggerates.
//
// THE BASS HAS TO STAY MONO, and that is the whole reason for the crossover.
// Low-frequency side content is where stereo width does its damage: it costs
// headroom, it cuts badly to vinyl, and on any single-speaker playback it
// partially cancels. Widening a mix without protecting the bottom end is the
// classic way to make something sound enormous on monitors and gutless on a
// phone.
//
// COMPLEMENTARY LINEAR-PHASE CROSSOVER, not Linkwitz-Riley. The high band is
// formed as (input delayed by the filter's group delay) minus the low band,
// so low + high reconstructs the input as a PURE DELAY — exactly, at every
// frequency. Set both bands to unity width and the node nulls against its own
// input to better than -180 dB.
//
// An IIR crossover cannot do that. An LR pair sums to an allpass: flat in
// magnitude but phase-shifted, so it leaves a signature on the signal even
// when every band is set to neutral. For a node whose job is often "leave
// this alone except below 120 Hz", that signature is the whole problem.
//
// No Haas or delay-based widening anywhere. It sounds wide in isolation and
// destroys mono compatibility, which is the exact failure this node exists
// to avoid.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "anv/core/Node.h"
#include "anv/dsp/Convolver.h"
#include "anv/dsp/DelayLine.h"
#include "anv/dsp/FirDesign.h"

namespace anv {

class StereoWidth final : public Node {
public:
  const char* type() const override { return "stereo_width"; }

  void prepare(const PrepareSpec& spec) override {
    spec_ = spec;
    rebuild();
  }

  void reset() override {
    lowBand_.reset();
    alignDelay_.reset();
  }

  void setParams(const ParamSet& p) override {
    width_ = std::clamp(p.num("width_percent", 100.0), 0.0, 400.0) / 100.0;
    bassMonoHz_ = std::clamp(p.num("bass_mono_below_hz", 0.0), 0.0, 800.0);
    // Separate from the crossover frequency so the split can be exercised
    // without also forcing the low band to mono — which is what makes the
    // perfect-reconstruction null testable.
    bassWidth_ = std::clamp(p.num("bass_width_percent", 0.0), 0.0, 400.0) / 100.0;
    stopbandDb_ = std::clamp(p.num("crossover_stopband_db", 80.0), 40.0, 120.0);

    if (spec_.numChannels > 0) rebuild();
  }

  int latencyFrames() const override { return latency_; }

  void process(ProcessContext& ctx) noexcept override {
    const int n = ctx.numFrames;
    const int ch = spec_.numChannels;

    // Nothing to widen with fewer than two channels.
    if (ch < 2) {
      for (int c = 0; c < ch; ++c) std::copy(ctx.in[c], ctx.in[c] + n, ctx.out[c]);
      return;
    }

    for (int i = 0; i < n; ++i) {
      mid_[static_cast<size_t>(i)] = 0.5 * (ctx.in[0][i] + ctx.in[1][i]);
      side_[static_cast<size_t>(i)] = 0.5 * (ctx.in[0][i] - ctx.in[1][i]);
    }

    if (!crossoverActive_) {
      for (int i = 0; i < n; ++i) {
        const double m = mid_[static_cast<size_t>(i)];
        const double s = side_[static_cast<size_t>(i)] * width_;
        ctx.out[0][i] = m + s;
        ctx.out[1][i] = m - s;
      }
    } else {
      // Low band of the side signal, and the aligned copies of mid and side.
      lowBand_.process(side_.data(), sideLow_.data(), n);

      const Sample* alignIn[2] = {mid_.data(), side_.data()};
      Sample* alignOut[2] = {midDelayed_.data(), sideDelayed_.data()};
      alignDelay_.process(alignIn, alignOut, n);

      for (int i = 0; i < n; ++i) {
        const double lo = sideLow_[static_cast<size_t>(i)];
        // Complementary: everything the low-pass did not take.
        const double hi = sideDelayed_[static_cast<size_t>(i)] - lo;
        const double s = bassWidth_ * lo + width_ * hi;
        const double m = midDelayed_[static_cast<size_t>(i)];
        ctx.out[0][i] = m + s;
        ctx.out[1][i] = m - s;
      }
    }

    for (int c = 2; c < ch; ++c)
      std::copy(ctx.in[c], ctx.in[c] + n, ctx.out[c]);
  }

private:
  void rebuild() {
    const int ch = spec_.numChannels;
    if (ch <= 0) return;
    const int maxBlock = spec_.maxBlockFrames;

    mid_.assign(static_cast<size_t>(maxBlock), 0.0);
    side_.assign(static_cast<size_t>(maxBlock), 0.0);
    sideLow_.assign(static_cast<size_t>(maxBlock), 0.0);
    midDelayed_.assign(static_cast<size_t>(maxBlock), 0.0);
    sideDelayed_.assign(static_cast<size_t>(maxBlock), 0.0);

    // Latency must describe what this node ACTUALLY does in the configuration
    // it was prepared for. With fewer than two channels process() is a
    // pass-through, so declaring the crossover's latency here would have the
    // chain trim thousands of samples off the front of a signal that was never
    // delayed — which silently deletes the beginning of the render.
    crossoverActive_ = (bassMonoHz_ > 0.0) && (ch >= 2);
    if (!crossoverActive_) {
      latency_ = 0;
      return;
    }

    // Transition band as wide as the corner frequency itself — half an octave
    // either side. A crossover this low does not need to be surgical, and a
    // narrower transition costs taps quadratically.
    const double cutoffNorm = bassMonoHz_ / spec_.sampleRate;
    const FirLowpass fir = designLowpass(cutoffNorm, cutoffNorm, stopbandDb_);

    // Run the FIR through the partitioned convolver rather than directly: at
    // this crossover frequency the filter runs to thousands of taps, and
    // direct convolution would dominate the entire render.
    lowBand_.prepare(fir.h);

    // The low band emerges delayed by the convolver's partition plus the
    // filter's own group delay; mid and the unfiltered side must be delayed to
    // match, or the complementary subtraction is meaningless.
    latency_ = lowBand_.latencyFrames() + fir.groupDelay;
    alignDelay_.prepare(2, latency_);

    reset();
  }

  double width_ = 1.0, bassMonoHz_ = 0.0, bassWidth_ = 0.0, stopbandDb_ = 80.0;
  bool crossoverActive_ = false;
  int latency_ = 0;

  PartitionedConvolver lowBand_;
  DelayLine alignDelay_;
  std::vector<Sample> mid_, side_, sideLow_, midDelayed_, sideDelayed_;
};

} // namespace anv
