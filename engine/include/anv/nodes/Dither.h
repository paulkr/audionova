// Dither and bit-depth reduction — the terminal node of an export chain.
//
// Truncating float to a fixed word length produces quantisation error that is
// CORRELATED with the signal, and correlated error is not noise: it is
// distortion that tracks the music, audible on fades and reverb tails as a
// gritty, granular edge long before the noise floor itself would be.
//
// Adding a small amount of the right noise before quantising decorrelates the
// error from the signal, converting distortion into a steady, benign hiss.
// Triangular PDF at 2 LSB peak-to-peak is the standard choice: it is the
// weakest dither that fully eliminates both the error's correlation with the
// signal AND the modulation of its level by the signal.
//
// NEVER DITHER A FLOAT EXPORT. A 32-bit float file has no fixed quantisation
// step to decorrelate, so dither there only adds noise to a signal that had
// none. bit_depth <= 0 means "no reduction" and the node is a pass-through.
//
// ORDER IS ABSOLUTE: this must come after the true-peak limiter. Anything that
// processes after dither re-correlates the error and undoes the entire point,
// and anything that changes level after limiting reintroduces the peaks the
// limiter was there to control.
//
// This is the one node in the set that adds signal-independent broadband
// noise, and the contract suite carries a named exemption for exactly that
// fact — flagged per node, with the reason written down, rather than the rules
// being quietly loosened for everyone. Two tests do not apply to it:
// silence-in/silence-out (violated by definition — adding noise to digital
// silence IS the mechanism) and the latency energy centroid (which assumes all
// output energy belongs to the impulse response). Every other contract test,
// including the strict latency peak probe, still applies.
#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "anv/core/Node.h"
#include "anv/core/Rng.h"

namespace anv {

class Dither final : public Node {
public:
  const char* type() const override { return "dither"; }

  void prepare(const PrepareSpec& spec) override {
    spec_ = spec;
    rng_.setSeed(spec.seed);
    reset();
  }

  void reset() override {
    for (auto& e : error_) e = 0.0;
  }

  void setParams(const ParamSet& p) override {
    bitDepth_ = static_cast<int>(std::clamp(p.num("bit_depth", 24.0), 0.0, 32.0));
    amountLsb_ = std::clamp(p.num("amount_lsb", 2.0), 0.0, 4.0);
    noiseShaping_ = (p.str("noise_shaping", "none") == "first_order");
    error_.assign(static_cast<size_t>(std::max(spec_.numChannels, 1)), 0.0);
  }

  int latencyFrames() const override { return 0; }

  void process(ProcessContext& ctx) noexcept override {
    const int n = ctx.numFrames;
    const int ch = spec_.numChannels;

    if (bitDepth_ <= 0 || bitDepth_ >= 32) {
      for (int c = 0; c < ch; ++c)
        std::copy(ctx.in[c], ctx.in[c] + n, ctx.out[c]);
      return;
    }

    // One LSB of a signed word of this depth, in full-scale units.
    const double step = 1.0 / std::pow(2.0, static_cast<double>(bitDepth_ - 1));
    const double invStep = 1.0 / step;

    for (int c = 0; c < ch; ++c) {
      const Sample* src = ctx.in[c];
      Sample* dst = ctx.out[c];
      double& err = error_[static_cast<size_t>(c)];

      for (int i = 0; i < n; ++i) {
        // Counter-based, so the dither is a pure function of position: the
        // same render produces the same noise every time, and it does not
        // change with block size.
        const double noise =
            amountLsb_ * 0.5 * step * rng_.tpdf(c, ctx.globalFrameIndex + i);

        // First-order noise shaping feeds the previous quantisation error
        // forward, pushing the noise spectrum up out of the ear's most
        // sensitive region. It raises TOTAL noise power while lowering
        // perceived noise, which is why it is opt-in rather than default —
        // at 24 bits there is nothing to gain and a little to lose.
        const double shaped = noiseShaping_ ? (src[i] - err) : src[i];
        const double wanted = shaped + noise;
        const double quantised = std::round(wanted * invStep) * step;

        // The fed-back error is measured against the PRE-DITHER quantiser
        // input, so it captures the dither too, not just the rounding. That
        // matters more than it looks: with 2-LSB TPDF the dither carries
        // step^2/6 of the total step^2/4, so feeding back `quantised - wanted`
        // would shape only a third of the noise and deliver about 1 dB of the
        // ~9 dB low-frequency improvement shaping is supposed to buy.
        //
        // With this definition the output is exactly x[n] + (1 - z^-1)e[n].
        err = quantised - shaped;
        dst[i] = quantised;
      }
    }
  }

private:
  int bitDepth_ = 24;
  double amountLsb_ = 2.0;
  bool noiseShaping_ = false;
  CounterRng rng_;
  std::vector<double> error_;
};

} // namespace anv
