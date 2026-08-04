// Magnetic tape character.
//
// WRITTEN FROM PUBLISHED PHYSICS, NOT FROM ANYONE'S SOURCE. Every line here is
// ours. The best open tape emulation available is GPL and therefore unusable in
// a commercial product; nothing from it, or from any other implementation, has
// been read, transcribed or paraphrased into this file. What is modelled below
// is derived from the recording chain's own mechanisms — a saturating
// magnetisation curve, the playback head's contour effect, gap loss, and
// transport speed error — each with its derivation written out so the next
// reader can check the arithmetic rather than trust the provenance. See
// docs/licensing.md.
//
// FOUR STAGES, IN THE ORDER THE SIGNAL MEETS THEM:
//
//   1. saturation   a soft, asymmetric, oversampled waveshaper
//   2. head bump    a low-frequency resonance from the playback head's contour
//   3. gap loss     a speed-dependent high-frequency roll-off
//   4. wow/flutter  slow and fast transport speed error
//
// ------------------------------------------------------------------ 1. ---
// WHICH NONLINEARITY THIS IS, AND WHY IT IS NOT JILES-ATHERTON.
//
// This is a MEMORYLESS ASYMMETRIC SHAPER with a bias control. It models the
// ANHYSTERETIC magnetisation curve — the single-valued transfer a correctly
// AC-biased recording actually traverses — and it does NOT model the hysteresis
// LOOP. Saying so plainly matters more than the parameter count: a hysteretic
// model's memory term produces a level-dependent phase shift at low frequencies
// that this node does not have.
//
// Jiles-Atherton is the physically grounded alternative and was rejected for
// two reasons, in this order:
//
//   IT NEEDS FIVE CONSTANTS WE HAVE NOT MEASURED. Ms, a, alpha, c and k are
//   properties of one tape formulation on one machine. There is no defensible
//   value for any of them here, and this project's rule is that a plausible
//   constant standing in for a missing measurement is worse than not having the
//   feature, because once it is in the code it is indistinguishable from
//   something that was measured. A shaper whose only claim is "soft, asymmetric
//   and progressive" claims exactly what it can support.
//
//   IT IS AN ODE, SO ADAA IS UNAVAILABLE. The engine's antialiasing strategy is
//   antiderivative antialiasing plus oversampling, and ADAA needs a closed-form
//   second antiderivative of the shaping function. dM/dH has none. A J-A model
//   would have to meet the -100 dB aliasing requirement by oversampling alone,
//   at a cost the measured numbers below say is unnecessary.
//
// The curve itself is `QuinticSatShaper` in dsp/Adaa.h (used here through its
// residual form, below), specified by its
// derivative f'(u) = (1 - u^2)^2 so that saturation is progressive from well
// below the ceiling rather than a rounded clip. See that header for why the
// derivative is the thing worth specifying and what the C2 continuity buys.
//
// ONLY THE RESIDUAL GOES THROUGH THE OVERSAMPLER. The stage computes
// y = u + g(u) with g = f - identity, sends g through the oversampler and the
// ADAA operator, and delays the linear term by exactly the same integer number
// of samples at the base rate. The reason is measured and is in
// docs/findings.md: ADAA2 applied to a linear function is exactly the
// three-point moving average, which at 4x oversampling costs 0.313 dB at
// 20 kHz — three times the gap loss a 30 ips machine has there, imposed on a
// node whose high-frequency response is one of the things it claims to model.
// With the split, that droop applies to the distortion products only.
//
// LEVEL CONVENTION. The shaper's knee is at |u| = 1, so `drive_db` 0 puts a
// 0 dBFS peak exactly at the knee. Programme material peaking at 0 dBFS sits
// 12-20 dB below that in RMS, which is where the curve produces a fraction of a
// percent to a couple of percent of third harmonic — the operating region a
// tape machine is actually run in. Pushing `drive_db` up walks the whole signal
// into the knee, which is what "hitting the tape harder" means.
//
// ------------------------------------------------------------------ 2. ---
// HEAD BUMP. The playback head's pole pieces are finite, so at wavelengths
// comparable to the head's contact length the flux path through the core
// interferes with itself. The result is a series of ripples in the low-frequency
// response, the first and largest of which is the familiar "head bump" — a few
// dB somewhere between 30 and 100 Hz, moving UP in frequency as the tape moves
// faster past the same head.
//
// Only the first ripple is modelled, as a peaking bell. The higher-order ones
// are progressively smaller and their spacing depends on a head geometry we do
// not have; inventing it would be inventing a measurement.
//
// `head_bump_hz` is NOT scaled by `speed_ips` automatically. It is proportional
// to speed in reality, and a hidden multiplier on a number the caller wrote in
// Hz is the kind of silent surprise this engine refuses everywhere else. The
// documented values per speed are in docs/engine.md.
//
// ------------------------------------------------------------------ 3. ---
// GAP LOSS. A playback head averages the magnetisation over its gap, which is a
// rectangular window in the space domain, so the response is that window's
// transform:
//
//     |H(f)| = |sinc(pi * g * f / v)|,     v = tape speed, g = effective gap
//
// with its first null where one full wavelength fits in the gap. This is the
// dominant high-frequency loss and the reason a 15 ips machine is brighter than
// a 7.5 ips one with the same head: halving v halves every wavelength, so the
// same gap eats twice as much of the band.
//
// Realised as a one-pole low-pass placed at the sinc's own -3 dB point, which
// is where sinc(x) = 1/sqrt(2), i.e. x = 1.39156:
//
//     f_-3dB = 1.39156 * v / (pi * g)
//
// A one-pole is not a sinc. The deviation is measured rather than assumed and
// is in docs/findings.md; it is small where the loss is small and grows as the
// loss deepens, which is the honest direction for an approximation to fail. The
// alternative, an FIR with the exact sinc magnitude, would cost latency and
// pre-ringing to reproduce a curve whose parameters are already a modelling
// choice.
//
// The corner sits ABOVE the audio band at 15 and 30 ips, which is exactly the
// case the textbook bilinear transform gets wrong — see dsp/AnalogOnePole.h.
//
// ------------------------------------------------------------------ 4. ---
// WOW AND FLUTTER, AND WHY IT IS A PURE FUNCTION OF ABSOLUTE TIME.
//
// Transport speed error reads back as pitch modulation. The obvious
// implementation — an LFO phase accumulated per block, or per sample — would
// make the output depend on where the block boundaries fell, because
// floating-point accumulation is not associative. Block-size invariance is
// structural in this engine, so the modulation is instead a PURE FUNCTION of
// the absolute frame index, built from the counter-based PRNG keyed on
// (nodeSeed, channel, knot index). Nothing is carried between blocks.
//
// It is RANDOM rather than a fixed sinusoid. The physical sources — capstan
// eccentricity, pinch-roller runout, tape-pack slip, scrape flutter against
// fixed guides — are several incommensurate rates at once, and a single
// sinusoid is audible as a warble tone on any sustained note in a way that the
// sum of them is not. `TapeDrift` below is band-limited value noise: PRNG draws
// on a knot grid, Catmull-Rom between them.
//
// THE AMPLITUDE IS DERIVED FROM THE REQUESTED SPEED DEVIATION, NOT GUESSED.
// A delay of D(n) samples produces a relative speed deviation of -D'(n), so a
// peak speed deviation of eps needs a delay whose slope is bounded by eps. For
// Catmull-Rom over knots in [-1, 1] with spacing H, the slope is bounded by
// 3.5 * A / H (the bound is derived in TapeDrift below), so
//
//     A = eps * H / 3.5
//
// which is what `rebuild()` computes. `wow_percent` therefore means the PEAK
// instantaneous speed deviation, which is what a wow-and-flutter specification
// means; the RMS is lower, and by how much is measured in the contract tests.
//
// WOW AND FLUTTER DEFAULT TO ZERO, AND THAT IS DELIBERATE.
//
// Not to make a number look good — because any nonzero value makes the headline
// number MEANINGLESS in either direction. Every measurement in
// engine/tests/measurements.cpp that could gate this node is a coherently
// sampled tone: the analysis relies on a periodic input producing a periodic
// output, so every harmonic lands in exactly one bin. Pitch modulation is
// precisely the thing that breaks that. At 19.5 kHz a 0.05% peak deviation
// smears the fundamental across several bins within one analysis window, and
// the aliasing figure computed from it describes the analysis window rather
// than the node. The suite already records the same effect on `chorus` and
// `plate_reverb`, which measure -74 and -85 dB of "distortion" they are
// physically incapable of producing.
//
// The modulation is a LINEAR, time-varying delay. It generates no harmonics and
// has nothing to fold, which is the suite's own stated reason for exempting
// those two nodes from the aliasing requirement. So measuring this node's
// aliasing with the transport perfect is not a dodge — it is the only
// experiment that isolates the term the requirement is about. The contract
// suite measures the modulation separately, on its own terms: block-size
// invariance bit-exactly, and the achieved speed deviation against the
// requested one.
//
// LATENCY. The modulation needs a base delay to swing about, and that base
// delay is NOT the effect — unlike a chorus, where the base delay is what makes
// two voices. So it is declared, the chain compensates it, and the node's mean
// group delay is zero. With both depths at zero there is no delay line at all
// and the declared latency drops to the saturator's, which is the same rule as
// a bandless EQ declaring nothing.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "anv/core/Node.h"
#include "anv/core/Rng.h"
#include "anv/dsp/Adaa.h"
#include "anv/dsp/AnalogOnePole.h"
#include "anv/dsp/FractionalDelay.h"
#include "anv/dsp/Oversampler.h"
#include "anv/dsp/TptSvf.h"

namespace anv {

// Band-limited random drift: a pure function of (channel, absolute frame).
//
// PRNG draws sit on a knot grid of `H` samples and Catmull-Rom interpolates
// between them. Nothing accumulates, so the value at frame n is the same value
// whatever block n arrived in and whatever thread computed it.
//
// TWO BOUNDS ARE USED BY THE CALLER AND BOTH ARE DERIVED HERE.
//
// AMPLITUDE. Catmull-Rom is an interpolating spline, not a convex combination,
// so it overshoots. Writing p(u) in terms of the four knots, the coefficient
// vector is
//
//     p_{-1}: -h10/2      p_0: h00 - h11/2      p_1: h01 + h10/2      p_2: h11/2
//
// and the largest attainable |p| for knots in [-1, 1] is the largest L1 norm of
// that vector over u in [0, 1], which is 1.25 at u = 0.5. So a drift of
// nominal amplitude A can reach 1.25*A, and the delay line has to be sized for
// that rather than for A.
//
// SLOPE, which is what sets the pitch deviation. Differentiating,
//
//     |p'(u)| <= 12(u - u^2) + |3u^2 - 4u + 1| + |3u^2 - 2u|   (times A / H)
//
// whose maximum over [0, 1] is 3.5, again at u = 0.5. That is the number the
// tape node inverts to turn a requested speed deviation into an amplitude.
class TapeDrift {
public:
  // Knots are placed every 1/(2*rateHz) seconds, so one full excursion and
  // return occupies one period of `rateHz`.
  void configure(double rateHz, double sampleRate) {
    const double spacing = sampleRate / (2.0 * std::max(rateHz, 1e-3));
    knot_ = std::max<int64_t>(2, static_cast<int64_t>(std::llround(spacing)));
  }

  int64_t knotSpacing() const { return knot_; }

  // Peak |value| a knot sequence in [-1, 1] can produce through this spline.
  static constexpr double kAmplitudeBound = 1.25;
  // Peak |d value / d sample| in units of amplitude per knot spacing.
  static constexpr double kSlopeBound = 3.5;

  double at(const CounterRng& rng, int channel, int64_t frame,
            uint32_t stream) const noexcept {
    // Floor division, so a negative frame index (which the engine never
    // produces, but which a test may) does not reflect the grid about zero.
    int64_t g = frame / knot_;
    if (frame < 0 && g * knot_ != frame) --g;
    const double u = static_cast<double>(frame - g * knot_) /
                     static_cast<double>(knot_);

    const double pm1 = rng.bipolar(channel, g - 1, stream);
    const double p0 = rng.bipolar(channel, g, stream);
    const double p1 = rng.bipolar(channel, g + 1, stream);
    const double p2 = rng.bipolar(channel, g + 2, stream);

    const double m0 = 0.5 * (p1 - pm1);
    const double m1 = 0.5 * (p2 - p0);
    const double u2 = u * u;
    const double u3 = u2 * u;
    const double h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
    const double h10 = u3 - 2.0 * u2 + u;
    const double h01 = -2.0 * u3 + 3.0 * u2;
    const double h11 = u3 - u2;
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1;
  }

private:
  int64_t knot_ = 1;
};

class Tape final : public Node {
public:
  const char* type() const override { return "tape"; }

  void prepare(const PrepareSpec& spec) override {
    spec_ = spec;
    rng_.setSeed(spec.seed);
    rebuild();
  }

  void reset() override {
    os_.reset();
    for (auto& s : satState_) s.reset();
    for (auto& r : ring_) std::fill(r.begin(), r.end(), 0.0);
    std::fill(ringPos_.begin(), ringPos_.end(), 0);
    for (auto& r : dryRing_) std::fill(r.begin(), r.end(), 0.0);
    std::fill(dryPos_.begin(), dryPos_.end(), 0);
    for (auto& f : bump_) f.reset();
    for (auto& f : gap_) f.reset();
    for (auto& l : lines_) l.reset();
  }

  void setParams(const ParamSet& p) override {
    // Bounded ABOVE at +18 dB, and the bound is the aliasing requirement
    // rather than taste. Measured in-band ASR at 0 dBFS, 19.5 kHz, 8x:
    //
    //     drive   0     +6     +12    +15    +18    +24
    //     ASR   -171   -143   -132   -122   -113   -100
    //
    // so +18 keeps 12 dB of margin on the -100 dB requirement across every
    // bias and speed, and +24 sits exactly on it. Past the knee the curve is
    // flat, so more drive is a limiter rather than a tape machine; there is
    // nothing there worth spending the margin on.
    driveDb_ = std::clamp(clampDb(p.num("drive_db", 0.0)), -24.0, 18.0);
    trimDb_ = clampDb(p.num("output_trim_db", 0.0));
    bias_ = std::clamp(p.num("bias", 0.0), -1.0, 1.0);

    headBumpDb_ = std::clamp(p.num("head_bump_db", 2.0), 0.0, 12.0);
    headBumpHz_ = std::clamp(p.num("head_bump_hz", 50.0), 20.0, 200.0);
    // Shape, not a measurement. The width of a real contour ripple depends on
    // head geometry we do not have, so this is exposed and documented rather
    // than baked in as though it were known.
    headBumpQ_ = std::clamp(p.num("head_bump_q", 0.8), 0.3, 4.0);

    speedIps_ = std::clamp(p.num("speed_ips", 15.0), 1.875, 30.0);
    gapUm_ = std::clamp(p.num("gap_um", 6.0), 0.5, 30.0);

    wowPercent_ = std::clamp(p.num("wow_percent", 0.0), 0.0, 1.0);
    wowRateHz_ = std::clamp(p.num("wow_rate_hz", 0.8), 0.05, 5.0);
    flutterPercent_ = std::clamp(p.num("flutter_percent", 0.0), 0.0, 1.0);
    flutterRateHz_ = std::clamp(p.num("flutter_rate_hz", 14.0), 2.0, 60.0);

    // 8x, wideband, 120 dB, and 8 is the FLOOR rather than the default.
    // Measured in-band ASR at 0 dBFS with a 19.5 kHz probe:
    //
    //     drive      0      +6     +12    +18    +24
    //     4x      -174    -113     -96    -84    -84
    //     8x      -171    -143    -132   -115   -100
    //     16x     -170    -142    -135   -133   -131
    //
    // 4x meets the -100 dB requirement only up to about +8 dB of drive, so it
    // is not offered; 16x buys 18 dB more at the top of the accepted drive
    // range and costs twice as much, and is offered for anyone who wants it.
    // The requirement is not adjusted anywhere in this file.
    int f = static_cast<int>(p.num("oversample_factor", 8.0));
    f = std::clamp(f, 8, 16);
    int pow2 = 8;
    while (pow2 * 2 <= f) pow2 *= 2;
    factor_ = pow2;

    driveLin_ = std::pow(10.0, driveDb_ / 20.0);
    trimLin_ = std::pow(10.0, trimDb_ / 20.0);
    shaper_.setBias(bias_);

    if (spec_.numChannels > 0) rebuild();
  }

  int latencyFrames() const override { return latency_; }

  void process(ProcessContext& ctx) noexcept override {
    const int n = ctx.numFrames;
    const int f = factor_;

    for (int c = 0; c < spec_.numChannels; ++c) {
      const size_t ci = static_cast<size_t>(c);

      // --- 1. saturation, at the oversampled rate ---
      //
      // Only the residual g = f - identity goes up. The linear term is delayed
      // by the same integer number of base-rate samples and added back below.
      Sample* pre = pre_[ci].data();
      const Sample* src = ctx.in[c];
      for (int i = 0; i < n; ++i) pre[i] = src[i] * driveLin_;

      Sample* w = os_.up(pre, n, c);
      const int hi = n * f;
      auto& st = satState_[ci];
      for (int i = 0; i < hi; ++i) w[i] = st.process(shaper_, w[i]);

      // Pad ADAA2's one oversampled-rate sample out to a whole base sample, so
      // the declared latency has no fractional part anywhere in it.
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

      // --- 2 and 3. head bump and gap loss, at the base rate ---
      Sample* dst = ctx.out[c];
      auto& bump = bump_[ci];
      auto& gap = gap_[ci];
      for (int i = 0; i < n; ++i)
        dst[i] = gap.lowpass(bump.processSample(dst[i] + delayDry(ci, pre[i])));

      // --- 4. transport speed error ---
      if (modulated_) {
        auto& line = lines_[ci];
        for (int i = 0; i < n; ++i) {
          line.write(dst[i]);
          const int64_t frame = ctx.globalFrameIndex + i;
          const double drift =
              wowAmp_ * wow_.at(rng_, c, frame, kWowStream) +
              flutterAmp_ * flutter_.at(rng_, c, frame, kFlutterStream);
          dst[i] = line.read(static_cast<double>(baseDelay_) + drift);
        }
      }

      for (int i = 0; i < n; ++i) dst[i] *= trimLin_;
    }
  }

  // Exposed for the contract tests, which have to be able to check that the
  // achieved speed deviation matches the requested one without reimplementing
  // the derivation they are checking.
  double wowAmplitudeSamples() const { return wowAmp_; }
  double flutterAmplitudeSamples() const { return flutterAmp_; }
  int baseDelaySamples() const { return baseDelay_; }
  double gapLossCornerHz() const { return gapCornerHz_; }

private:
  static constexpr uint32_t kWowStream = 1;
  static constexpr uint32_t kFlutterStream = 2;

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

    pre_.assign(static_cast<size_t>(ch),
                std::vector<Sample>(static_cast<size_t>(spec_.maxBlockFrames), 0.0));
    ring_.assign(static_cast<size_t>(ch),
                 std::vector<Sample>(static_cast<size_t>(factor_) + 1, 0.0));
    ringPos_.assign(static_cast<size_t>(ch), 0);
    satState_.resize(static_cast<size_t>(ch));

    const int satLatency = os_.latencyFrames() + 1;
    dryRing_.assign(static_cast<size_t>(ch),
                    std::vector<Sample>(static_cast<size_t>(satLatency), 0.0));
    dryPos_.assign(static_cast<size_t>(ch), 0);

    bump_.assign(static_cast<size_t>(ch), TptSvf{});
    for (auto& b : bump_) {
      b.setSampleRate(sr);
      b.configure(FilterShape::Bell, headBumpHz_, headBumpDb_, headBumpQ_);
    }

    // sinc(x) = 1/sqrt(2) at x = 1.3915576, so the gap loss reaches -3 dB at
    // that fraction of its first null. v is in metres per second, g in metres.
    const double v = speedIps_ * 0.0254;
    const double gapM = gapUm_ * 1e-6;
    gapCornerHz_ = 1.3915576 * v / (M_PI * gapM);
    gap_.assign(static_cast<size_t>(ch), AnalogOnePole{});
    for (auto& g : gap_) g.configure(gapCornerHz_, sr, kGapReferenceHz);

    // Peak speed deviation eps needs amplitude A = eps * H / slopeBound.
    wow_.configure(wowRateHz_, sr);
    flutter_.configure(flutterRateHz_, sr);
    wowAmp_ = (wowPercent_ * 0.01) * static_cast<double>(wow_.knotSpacing()) /
              TapeDrift::kSlopeBound;
    flutterAmp_ = (flutterPercent_ * 0.01) *
                  static_cast<double>(flutter_.knotSpacing()) /
                  TapeDrift::kSlopeBound;

    modulated_ = (wowAmp_ > 0.0 || flutterAmp_ > 0.0);
    if (modulated_) {
      // The spline overshoots its knots by up to 25%, and the 4-point Lagrange
      // read needs a sample either side of the interval, so the base delay has
      // to clear both or the read clamps and the modulation is silently
      // truncated at its extremes.
      const double swing =
          TapeDrift::kAmplitudeBound * (wowAmp_ + flutterAmp_);
      baseDelay_ = static_cast<int>(std::ceil(swing)) + 2;
      lines_.assign(static_cast<size_t>(ch), ModulatedDelayLine{});
      for (auto& l : lines_) l.prepare(baseDelay_ + static_cast<int>(std::ceil(swing)) + 8);
    } else {
      baseDelay_ = 0;
      lines_.clear();
    }

    latency_ = satLatency + baseDelay_;
    reset();
  }

  // The gap-loss corner is above the audio band at 15 ips and higher, so the
  // realisation is pinned to the analog prototype at the top of the band rather
  // than at its own corner. See dsp/AnalogOnePole.h.
  static constexpr double kGapReferenceHz = 20000.0;

  double driveDb_ = 0.0, trimDb_ = 0.0, bias_ = 0.0;
  double driveLin_ = 1.0, trimLin_ = 1.0;
  double headBumpDb_ = 2.0, headBumpHz_ = 50.0, headBumpQ_ = 0.8;
  double speedIps_ = 15.0, gapUm_ = 6.0, gapCornerHz_ = 0.0;
  double wowPercent_ = 0.0, wowRateHz_ = 0.8;
  double flutterPercent_ = 0.0, flutterRateHz_ = 14.0;
  double wowAmp_ = 0.0, flutterAmp_ = 0.0;
  bool modulated_ = false;
  int baseDelay_ = 0;
  int factor_ = 4;
  int latency_ = 0;

  Oversampler os_;
  QuinticSatResidualShaper shaper_;
  CounterRng rng_;
  TapeDrift wow_, flutter_;
  std::vector<Adaa2<QuinticSatResidualShaper>> satState_;
  std::vector<std::vector<Sample>> pre_;
  std::vector<std::vector<Sample>> ring_;
  std::vector<int> ringPos_;
  std::vector<std::vector<Sample>> dryRing_;
  std::vector<int> dryPos_;
  std::vector<TptSvf> bump_;
  std::vector<AnalogOnePole> gap_;
  std::vector<ModulatedDelayLine> lines_;
};

} // namespace anv
