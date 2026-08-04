// Summing-bus character.
//
// WRITTEN FROM FIRST PRINCIPLES, NOT FROM ANYONE'S SOURCE. The best-known open
// implementation of this effect is MIT and could legally have been vendored;
// it was not, and nothing from it has been read, transcribed or paraphrased
// into this file. Every line here is ours, and what it does is derived from the
// two circuit facts below rather than from a coefficient table. See
// docs/licensing.md.
//
// WHAT A SUMMING BUS ACTUALLY DOES, which is very little, on purpose.
//
// This node is the opposite of `drive`. A saturator is chosen for the sound of
// its distortion; a console stage is a thing signal passes through on the way
// somewhere else, and if any single instance of it is audible the desk is
// broken. What makes it worth modelling is that a mix passes through a lot of
// them, and small deviations that are inaudible once are not inaudible twenty
// times. So every default here is set so that ONE instance measures as almost
// nothing, and the parameters are the physical quantities that decide how much
// that "almost nothing" is.
//
// TWO MECHANISMS, BOTH FIRST-ORDER, BOTH CIRCUIT FACTS:
//
// 1. A GENTLE ODD NONLINEARITY, SET BY HEADROOM. An amplifier stage is linear
//    until it approaches its rails. `headroom_db` is the distance from 0 dBFS
//    to that knee — the level diagram every analog desk is designed around —
//    so the shaper sees
//
//        u = x * 10^(drive_db/20) / K,     K = 10^(headroom_db/20)
//
//    and the output is scaled back by K, which makes the small-signal gain
//    exactly unity. The curve is the quintic in dsp/Adaa.h, whose leading
//    error term is -2u^3/3, so a sine of peak amplitude a produces a third
//    harmonic of a^3/6 against a fundamental of a: the distortion is
//
//        d3 ~ a^2 / 6
//
//    which at the default 24 dB of headroom and a 0 dBFS peak is 6.6e-4, or
//    -63.6 dB. That is the number the defaults are chosen to produce, and it is
//    an arithmetic consequence of the curve rather than a taste setting.
//    Halving the headroom quadruples it, which is exactly how an analog stage
//    behaves as the level diagram is pushed.
//
//    ONLY THE RESIDUAL IS OVERSAMPLED. The stage computes y = u + g(u) with
//    g = f - identity, and delays the linear term by exactly the oversampler's
//    integer round trip instead of running it through. That is not an
//    optimisation, it is the difference between this node working and not:
//    ADAA2 applied to a linear function is exactly a three-point moving average
//    at the oversampled rate, which costs 0.313 dB at 20 kHz at 4x — two thirds
//    of a decibel MORE than the roll-off this node exists to model, in the same
//    part of the band, accumulating the same way over ten stages. With the
//    split it lands on a residual sitting 60 dB down and disappears. See
//    dsp/Adaa.h and docs/findings.md.
//
//    Odd-order, not even: a bus amplifier is a differential, push-pull
//    structure and its transfer is symmetric. `drive` is where asymmetry lives.
//
// 2. A FREQUENCY-DEPENDENT SHIFT AT BOTH ENDS, from AC coupling and from finite
//    gain-bandwidth product. Both are single poles and neither is a tone
//    control:
//
//        coupling_hz    a first-order high-pass, the input coupling capacitor
//        bandwidth_hz   a first-order low-pass, the amplifier's open-loop
//                       gain-bandwidth divided by the closed-loop gain
//
//    At the defaults (12 Hz and 60 kHz) one instance is -0.46 dB at 20 kHz and
//    -0.006 dB at 100 Hz — nothing. What it also is, and what actually matters,
//    is -18.4 degrees at 20 kHz and +6.8 degrees at 100 Hz, because a first-
//    order pole's phase reaches much further from its corner than its magnitude
//    does. Ten stages of that is 184 degrees at the top of the band, and phase
//    is what makes summed material sound glued or smeared. THAT is the "only
//    audible in aggregate" part, and it is the reason the poles are here at all
//    rather than being dismissed as inaudible.
//
// THE ROLL-OFF IS REALISED SO IT IS RIGHT IN THE BAND, NOT AT THE CORNER. A
// 60 kHz corner sits above the 48 kHz internal Nyquist, which is precisely
// where the textbook bilinear transform stops describing the analog prototype:
// prewarping at the corner would give -0.09 dB at 20 kHz instead of -0.46, five
// times too little, in the one part of the band the parameter exists to shape.
// See dsp/AnalogOnePole.h for the derivation and for what the realisation still
// gets wrong above 20 kHz.
//
// LATENCY is the oversampler's round trip plus ADAA2's padded sample, both
// exact integers at the base rate. The two poles are one-pole IIRs and
// contribute none.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "anv/core/Node.h"
#include "anv/dsp/Adaa.h"
#include "anv/dsp/AnalogOnePole.h"
#include "anv/dsp/Oversampler.h"

namespace anv {

class Console final : public Node {
public:
  const char* type() const override { return "console"; }

  void prepare(const PrepareSpec& spec) override {
    spec_ = spec;
    rebuild();
  }

  void reset() override {
    os_.reset();
    for (auto& s : satState_) s.reset();
    for (auto& r : ring_) std::fill(r.begin(), r.end(), 0.0);
    std::fill(ringPos_.begin(), ringPos_.end(), 0);
    for (auto& r : dryRing_) std::fill(r.begin(), r.end(), 0.0);
    std::fill(dryPos_.begin(), dryPos_.end(), 0);
    for (auto& f : coupling_) f.reset();
    for (auto& f : bandwidth_) f.reset();
  }

  void setParams(const ParamSet& p) override {
    // +/-12 dB. The stage's level diagram is `headroom_db`; this is the trim
    // in front of it, and the two together decide how far up the curve the
    // signal runs. The upper bound is the aliasing requirement: at 6 dB of
    // headroom and +12 dB of drive — the hardest configuration the node
    // accepts — in-band ASR at 0 dBFS measures -113.9 dB at 4x.
    driveDb_ = std::clamp(clampDb(p.num("drive_db", 0.0)), -12.0, 12.0);
    trimDb_ = clampDb(p.num("output_trim_db", 0.0));
    // Lower bound 6 dB rather than 0: at 0 dB of headroom the knee sits at
    // 0 dBFS and the node stops being a bus stage, which is what `drive` is
    // for. The floor is also where the aliasing figure is measured.
    headroomDb_ = std::clamp(p.num("headroom_db", 24.0), 6.0, 40.0);
    couplingHz_ = std::clamp(p.num("coupling_hz", 12.0), 1.0, 40.0);
    bandwidthHz_ = std::clamp(p.num("bandwidth_hz", 60000.0), 15000.0, 200000.0);

    // 4x is enough here and that is measured, not assumed. In-band ASR at
    // 0 dBFS, 19.5 kHz probe: -249 dB at the default 24 dB of headroom, -202
    // at the 6 dB floor, and -114 at the hardest setting the node accepts
    // (6 dB headroom, +12 dB drive). Raising the factor makes no difference
    // beyond a few dB either way, because the curve is a degree-five
    // polynomial and a console stage never reaches its knee.
    int f = static_cast<int>(p.num("oversample_factor", 4.0));
    f = std::clamp(f, 4, 16);
    int pow2 = 4;
    while (pow2 * 2 <= f) pow2 *= 2;
    factor_ = pow2;

    driveLin_ = std::pow(10.0, driveDb_ / 20.0);
    trimLin_ = std::pow(10.0, trimDb_ / 20.0);
    knee_ = std::pow(10.0, headroomDb_ / 20.0);
    invKnee_ = 1.0 / knee_;

    if (spec_.numChannels > 0) rebuild();
  }

  int latencyFrames() const override { return latency_; }

  void process(ProcessContext& ctx) noexcept override {
    const int n = ctx.numFrames;
    const int f = factor_;

    for (int c = 0; c < spec_.numChannels; ++c) {
      const size_t ci = static_cast<size_t>(c);

      // Input trim and the coupling high-pass, then normalise into the
      // shaper's own units so that |u| = 1 is the stage's knee.
      Sample* pre = pre_[ci].data();
      const Sample* src = ctx.in[c];
      auto& hp = coupling_[ci];
      const double scale = driveLin_ * invKnee_;
      for (int i = 0; i < n; ++i) pre[i] = hp.highpass(src[i]) * scale;

      Sample* w = os_.up(pre, n, c);
      const int hi = n * f;
      auto& st = satState_[ci];
      for (int i = 0; i < hi; ++i) w[i] = st.process(shaper_, w[i]);

      if (f > 1) {
        auto& ring = ring_[ci];
        int& pos = ringPos_[ci];
        const int rlen = static_cast<int>(ring.size());
        for (int i = 0; i < hi; ++i) {
          ring[static_cast<size_t>(pos)] = w[i];
          int a = pos - (f - 1);
          if (a < 0) a += rlen;
          w[i] = ring[static_cast<size_t>(a)];
          pos = (pos + 1) % rlen;
        }
      }

      os_.down(ctx.out[c], n, c);

      // The residual is in dst; the linear term arrives from the dry ring,
      // delayed by exactly the oversampler's round trip plus ADAA2's padded
      // sample. Both are in shaper units, so one scale by the knee restores
      // the level diagram.
      Sample* dst = ctx.out[c];
      auto& lp = bandwidth_[ci];
      const double post = knee_ * trimLin_;
      for (int i = 0; i < n; ++i)
        dst[i] = lp.lowpass((dst[i] + delayDry(ci, pre[i])) * post);
    }
  }

  // For the contract tests: the arithmetic third-harmonic prediction the
  // header derives, so the test can compare a rendered spectrum against the
  // formula rather than against the code that implements it.
  double kneeLinear() const { return knee_; }

private:
  // The linear half of the parallel path: an integer delay of exactly the
  // saturator's own latency, read before written so the ring's length IS the
  // delay.
  inline Sample delayDry(size_t ci, Sample x) noexcept {
    auto& r = dryRing_[ci];
    int& p = dryPos_[ci];
    const int len = static_cast<int>(r.size());
    const Sample y = r[static_cast<size_t>(p)];
    r[static_cast<size_t>(p)] = x;
    p = (p + 1 == len) ? 0 : p + 1;
    return y;
  }

  void rebuild() {
    const int ch = spec_.numChannels;
    if (ch <= 0) return;
    const double sr = spec_.sampleRate;

    os_.prepare(factor_, ch, spec_.maxBlockFrames, OsProfile::Wideband, 120.0);
    latency_ = os_.latencyFrames() + 1;

    pre_.assign(static_cast<size_t>(ch),
                std::vector<Sample>(static_cast<size_t>(spec_.maxBlockFrames), 0.0));
    ring_.assign(static_cast<size_t>(ch),
                 std::vector<Sample>(static_cast<size_t>(factor_) + 1, 0.0));
    ringPos_.assign(static_cast<size_t>(ch), 0);
    satState_.resize(static_cast<size_t>(ch));
    dryRing_.assign(static_cast<size_t>(ch),
                    std::vector<Sample>(static_cast<size_t>(latency_), 0.0));
    dryPos_.assign(static_cast<size_t>(ch), 0);

    coupling_.assign(static_cast<size_t>(ch), AnalogOnePole{});
    bandwidth_.assign(static_cast<size_t>(ch), AnalogOnePole{});
    for (auto& fl : coupling_) fl.configure(couplingHz_, sr, kReferenceHz);
    for (auto& fl : bandwidth_) fl.configure(bandwidthHz_, sr, kReferenceHz);

    reset();
  }

  // The top of the audio band. Both poles are matched to their analog
  // prototype here; for the coupling high-pass the corner is far below it, so
  // the rule degenerates to the textbook prewarp at the corner.
  static constexpr double kReferenceHz = 20000.0;

  double driveDb_ = 0.0, trimDb_ = 0.0;
  double headroomDb_ = 24.0, couplingHz_ = 12.0, bandwidthHz_ = 60000.0;
  double driveLin_ = 1.0, trimLin_ = 1.0, knee_ = 1.0, invKnee_ = 1.0;
  int factor_ = 4;
  int latency_ = 0;

  Oversampler os_;
  QuinticSatResidualShaper shaper_;
  std::vector<Adaa2<QuinticSatResidualShaper>> satState_;
  std::vector<std::vector<Sample>> pre_;
  std::vector<std::vector<Sample>> ring_;
  std::vector<int> ringPos_;
  std::vector<std::vector<Sample>> dryRing_;
  std::vector<int> dryPos_;
  std::vector<AnalogOnePole> coupling_, bandwidth_;
};

} // namespace anv
