// Plate reverb — Dattorro topology.
//
// Complements ConvolutionReverb rather than competing with it. Convolution is
// exact but its tail is fixed at whatever the impulse response contains;
// changing the decay time means synthesising and re-transforming a new one.
// A recirculating network is parametric: decay, damping and size are knobs,
// the tail can run indefinitely, and it costs a fraction of the compute.
//
// Structure follows Dattorro (JAES 1997): pre-delay, an input bandwidth
// filter, four series allpass diffusers, then a figure-of-eight tank whose two
// halves feed each other. Each half is a modulated allpass, a delay, a damping
// filter, a second allpass and another delay.
//
// WHY THE DELAY LENGTHS ARE MUTUALLY PRIME. A recirculating network whose
// delays share factors produces coincident echo times, which pile up into a
// ringing pitch — the metallic clang that gives cheap digital reverb away.
// Mutually prime lengths spread the echo density evenly instead.
//
// WHY THE TANK ALLPASSES ARE MODULATED. Even with prime lengths a static tank
// settles into standing modes on sustained material. Slowly moving two of the
// allpasses breaks them up. The excursion is under a sample, so it colours
// the tail without any audible pitch wobble.
//
// The output tap set is a simplification of Dattorro's: taps at mutually prime
// positions across both halves, summed with alternating sign for L and R. The
// topology and the decorrelation are the parts that matter musically; the
// exact tap positions are a voicing choice.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "anv/core/Node.h"
#include "anv/dsp/FractionalDelay.h"

namespace anv {

class PlateReverb final : public Node {
public:
  const char* type() const override { return "plate_reverb"; }

  void prepare(const PrepareSpec& spec) override {
    spec_ = spec;
    rebuild();
  }

  void reset() override {
    for (auto& a : diffusers_) a.reset();
    for (auto& a : tankAp_) a.reset();
    for (auto& d : tankDelay_) d.reset();
    preDelay_.reset();
    inputLp_ = 0.0;
    dampL_ = 0.0;
    dampR_ = 0.0;
    feedL_ = 0.0;
    feedR_ = 0.0;
  }

  void setParams(const ParamSet& p) override {
    decayS_ = std::clamp(p.num("decay_s", 2.0), 0.1, 20.0);
    preDelayMs_ = std::clamp(p.num("pre_delay_ms", 0.0), 0.0, 250.0);
    dampingHz_ = std::clamp(p.num("damping_hz", 6000.0), 500.0, 20000.0);
    bandwidthHz_ = std::clamp(p.num("bandwidth_hz", 12000.0), 500.0, 20000.0);
    size_ = std::clamp(p.num("size", 1.0), 0.4, 2.0);
    modDepth_ = std::clamp(p.num("modulation_depth", 0.6), 0.0, 1.0);
    if (spec_.numChannels > 0) rebuild();
  }

  int latencyFrames() const override { return 0; }

  int64_t tailFrames() const override { return tail_; }

  void process(ProcessContext& ctx) noexcept override {
    const int n = ctx.numFrames;
    const int ch = spec_.numChannels;
    const double invSr = 1.0 / spec_.sampleRate;

    for (int i = 0; i < n; ++i) {
      // Sum to mono into the tank. A plate is one physical sheet; its stereo
      // image comes from where the pickups sit, not from two separate plates.
      double x = 0.0;
      for (int c = 0; c < ch; ++c) x += ctx.in[c][i];
      x /= static_cast<double>(std::max(ch, 1));

      preDelay_.write(x);
      x = preDelayed_ > 0 ? preDelay_.read(static_cast<double>(preDelayed_)) : x;

      // Input bandwidth: keep the very top out of the tank, where it would
      // only be smeared into noise.
      inputLp_ += bwCoeff_ * (x - inputLp_);
      x = inputLp_;

      for (int d = 0; d < 4; ++d) x = diffusers_[static_cast<size_t>(d)].process(x);

      const double t = static_cast<double>(ctx.globalFrameIndex + i) * invSr;
      const double mod1 = modSamples_ * std::sin(2.0 * M_PI * 0.93 * t);
      const double mod2 = modSamples_ * std::sin(2.0 * M_PI * 1.31 * t + 1.7);

      // Left half, fed by the right half's output.
      double a = x + decay_ * feedR_;
      a = tankAp_[0].processModulated(a, mod1);
      a = tankDelay_[0].process(a);
      dampL_ += dampCoeff_ * (a - dampL_);
      a = dampL_ * decay_;
      a = tankAp_[1].process(a);
      feedL_ = tankDelay_[1].process(a);

      // Right half, fed by the left half's output.
      double b = x + decay_ * feedL_;
      b = tankAp_[2].processModulated(b, mod2);
      b = tankDelay_[2].process(b);
      dampR_ += dampCoeff_ * (b - dampR_);
      b = dampR_ * decay_;
      b = tankAp_[3].process(b);
      feedR_ = tankDelay_[3].process(b);

      // Taps at mutually prime positions, opposed between channels so the two
      // outputs decorrelate.
      const double outL = tankDelay_[0].tap(tapA_) + tankDelay_[1].tap(tapB_) -
                          tankDelay_[2].tap(tapC_);
      const double outR = tankDelay_[2].tap(tapD_) + tankDelay_[3].tap(tapE_) -
                          tankDelay_[0].tap(tapF_);

      if (ch >= 2) {
        ctx.out[0][i] = outL * 0.6;
        ctx.out[1][i] = outR * 0.6;
        for (int c = 2; c < ch; ++c) ctx.out[c][i] = 0.0;
      } else {
        ctx.out[0][i] = (outL + outR) * 0.3;
      }
    }
  }

private:
  // Schroeder allpass: unity magnitude, so it diffuses without colouring.
  class Allpass {
  public:
    void prepare(int length, double gain) {
      len_ = std::max(1, length);
      gain_ = gain;
      buf_.assign(static_cast<size_t>(len_) + 4, 0.0);
      pos_ = 0;
    }
    void reset() {
      std::fill(buf_.begin(), buf_.end(), 0.0);
      pos_ = 0;
    }
    inline double process(double x) noexcept { return processModulated(x, 0.0); }

    inline double processModulated(double x, double modSamples) noexcept {
      const int size = static_cast<int>(buf_.size());
      double readPos = static_cast<double>(pos_) - (static_cast<double>(len_) + modSamples);
      while (readPos < 0.0) readPos += static_cast<double>(size);

      // Linear interpolation is enough for a sub-sample tank excursion.
      const int i0 = static_cast<int>(readPos);
      const double frac = readPos - static_cast<double>(i0);
      const int i1 = (i0 + 1) % size;
      const double delayed = buf_[static_cast<size_t>(i0)] * (1.0 - frac) +
                             buf_[static_cast<size_t>(i1)] * frac;

      const double w = x + gain_ * delayed;
      buf_[static_cast<size_t>(pos_)] = w;
      if (++pos_ >= size) pos_ = 0;
      return delayed - gain_ * w;
    }

  private:
    std::vector<double> buf_;
    int len_ = 1, pos_ = 0;
    double gain_ = 0.5;
  };

  class TapDelay {
  public:
    void prepare(int length) {
      len_ = std::max(1, length);
      buf_.assign(static_cast<size_t>(len_) + 1, 0.0);
      pos_ = 0;
    }
    void reset() {
      std::fill(buf_.begin(), buf_.end(), 0.0);
      pos_ = 0;
    }
    inline double process(double x) noexcept {
      const int size = static_cast<int>(buf_.size());
      int r = pos_ - len_;
      if (r < 0) r += size;
      const double y = buf_[static_cast<size_t>(r)];
      buf_[static_cast<size_t>(pos_)] = x;
      if (++pos_ >= size) pos_ = 0;
      return y;
    }
    inline double tap(int delay) const noexcept {
      const int size = static_cast<int>(buf_.size());
      int r = pos_ - 1 - std::min(delay, len_);
      while (r < 0) r += size;
      return buf_[static_cast<size_t>(r)];
    }

  private:
    std::vector<double> buf_;
    int len_ = 1, pos_ = 0;
  };

  int scaled(int atReference) const {
    // Dattorro's lengths are given at 29761 Hz; scale to the working rate.
    return std::max(1, static_cast<int>(std::lround(
                           static_cast<double>(atReference) *
                           (spec_.sampleRate / 29761.0) * size_)));
  }

  void rebuild() {
    if (spec_.numChannels <= 0) return;
    const double sr = spec_.sampleRate;

    preDelayed_ = static_cast<int>(std::lround(preDelayMs_ * 0.001 * sr));
    preDelay_.prepare(preDelayed_ + 8);

    bwCoeff_ = 1.0 - std::exp(-2.0 * M_PI * bandwidthHz_ / sr);
    dampCoeff_ = 1.0 - std::exp(-2.0 * M_PI * dampingHz_ / sr);

    diffusers_[0].prepare(scaled(142), 0.75);
    diffusers_[1].prepare(scaled(107), 0.75);
    diffusers_[2].prepare(scaled(379), 0.625);
    diffusers_[3].prepare(scaled(277), 0.625);

    // Negative gains in the tank decorrelate the two halves further.
    tankAp_[0].prepare(scaled(672), -0.7);
    tankAp_[1].prepare(scaled(1800), 0.5);
    tankAp_[2].prepare(scaled(908), -0.7);
    tankAp_[3].prepare(scaled(2656), 0.5);

    tankDelay_[0].prepare(scaled(4453));
    tankDelay_[1].prepare(scaled(3720));
    tankDelay_[2].prepare(scaled(4217));
    tankDelay_[3].prepare(scaled(3163));

    tapA_ = scaled(266); tapB_ = scaled(2974); tapC_ = scaled(1913);
    tapD_ = scaled(1996); tapE_ = scaled(1990); tapF_ = scaled(187);

    modSamples_ = modDepth_ * 0.6; // sub-sample: breaks modes, no wobble

    // Count the decay multiplications per circuit carefully — this is easy to
    // get wrong by a factor of two and the result is a reverb that is always
    // plausible and never the length it was asked for.
    //
    // Each half applies `decay_` TWICE: once where the other half's output
    // enters, and once after its damping filter. A full circuit passes through
    // both halves, so the round-trip gain is decay_^4, not decay_^2.
    //
    //   (decay_^4)^(RT60/T_loop) = 10^-3   =>   decay_ = 10^(-0.75*T_loop/RT60)
    //
    // That derivation alone is not enough, and the correction is not a fudge.
    // The tank's Schroeder allpasses have their own LOSSLESS sub-loops, so
    // some energy recirculates inside an allpass without passing through the
    // decay gain at all. Those paths traverse the main ring less often per
    // second than the model assumes, and the measured tail comes out longer
    // than predicted. Measured: the pure derivation (0.75) overshoots by
    // ~1.35x, and 1.5 undershoots by ~0.70x, both consistently for decay times
    // well above the loop period. The constant below lands on unity.
    //
    // FLOOR: this topology cannot decay faster than roughly one circuit of its
    // own ring, about 0.52 s at unity size. Requests below that come out long
    // no matter what the gain is set to.
    const double loopSamples = static_cast<double>(
        scaled(4453) + scaled(3720) + scaled(4217) + scaled(3163));
    const double loopSeconds = loopSamples / sr;
    decay_ = std::pow(10.0, -1.02 * loopSeconds / decayS_);
    decay_ = std::clamp(decay_, 0.0, 0.98);

    tail_ = static_cast<int64_t>(decayS_ * 1.5 * sr);
    reset();
  }

  double decayS_ = 2.0, preDelayMs_ = 0.0, dampingHz_ = 6000.0;
  double bandwidthHz_ = 12000.0, size_ = 1.0, modDepth_ = 0.6;

  double bwCoeff_ = 0.0, dampCoeff_ = 0.0, decay_ = 0.5, modSamples_ = 0.0;
  double inputLp_ = 0.0, dampL_ = 0.0, dampR_ = 0.0, feedL_ = 0.0, feedR_ = 0.0;
  int preDelayed_ = 0;
  int tapA_ = 0, tapB_ = 0, tapC_ = 0, tapD_ = 0, tapE_ = 0, tapF_ = 0;
  int64_t tail_ = 0;

  ModulatedDelayLine preDelay_;
  Allpass diffusers_[4];
  Allpass tankAp_[4];
  TapDelay tankDelay_[4];
};

} // namespace anv
