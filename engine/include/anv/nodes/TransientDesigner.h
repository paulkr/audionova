// Transient designer — attack and sustain shaped by the DIFFERENCE between two
// envelope followers, so the control is level-independent.
//
// WHY THIS IS NOT A COMPRESSOR. A compressor's gain is a function of how loud
// the signal is: it acts on everything above a threshold and nothing below it,
// so a quiet ghost note and a loud accent are treated as different events even
// when they have identical shapes. This node's gain is a function of how the
// level is CHANGING. Two envelope followers watch the same signal with
// different time constants; where they disagree is where something is
// happening. A snare hit at -30 dBFS and the same hit at -6 dBFS produce the
// same envelope RATIO and therefore the same shaping, which is why there is no
// threshold parameter here and why adding one would be a category error.
//
// THE TWO DIFFERENTIALS, and why they are ratios rather than dB differences.
//
//   attack   A = 1 - slowAttackEnv / fastEnv     (positive only while rising)
//   sustain  S = 1 - fastEnv / slowReleaseEnv    (positive only while falling)
//
// Both live in [0, 1) by construction: a fully-developed transient is 1, steady
// state is 0. The obvious alternative — the difference of the two envelopes in
// dB — is unbounded and behaves badly exactly where it matters. A hit rising
// out of silence puts the fast envelope 20-50 dB above the slow one within a
// millisecond, so a dB differential scaled by any usable control value slams
// into whatever range clamp is bolted on and the control stops meaning
// anything. Expressed as a ratio the same event saturates smoothly at 1, the
// clamp is unnecessary, and `attack_db` means literally what it says: the gain
// applied to a fully-developed transient.
//
//   gain_dB = attack_db * A + sustain_db * S
//
// A IS HALF-WAVE RECTIFIED AND S IS NOT, and that asymmetry is exact rather
// than an oversight. The attack differential genuinely goes negative through a
// decay, so without `max(0, .)` an attack boost would quietly cut the body of
// every note — an attack control that is also a sustain control, which is the
// kind of thing nobody thinks to check for. The sustain differential cannot go
// negative: `slowRelease` shares `fast`'s attack coefficient and releases more
// slowly, so its detector state dominates at every sample, by induction and
// bit-exactly, because IEEE rounding is monotone. A guard there would never
// fire, and a guard that never fires tells the next reader the quantity can go
// negative when it provably cannot.
//
// THE FAST ENVELOPE'S TIMES ARE FIXED, DELIBERATELY. They are the reference the
// user's two times are measured against, so exposing them would let a planner
// set the reference slower than the thing being referenced and get silence from
// both controls. `attack_ms` says how long a transient is taken to last;
// `sustain_ms` says how long the body after it is taken to last.
//
// KNOWN LIMIT, measured rather than hedged: this is a BROADBAND detector and a
// broadband detector cannot separate "the envelope is moving" from "the
// waveform is moving" once the period approaches the fast envelope's release.
// Sustained low bass is therefore gain-modulated at twice its own frequency —
// the figure is in findings.md. Use `multiband_compressor` on bass-led
// material; this node is for drums, plucks and anything with an onset.
//
// THE NODE IS OVERSAMPLED 8x, AND THE RECTIFIER IS WHY.
//
// A time-varying gain is a multiplication, and a multiplication is a
// nonlinearity: it puts sidebands around every input partial, and any sideband
// landing above Nyquist folds back as a non-harmonic tone. That much is true of
// every dynamics node here. What makes this one worse is where the gain comes
// from. `abs()` has a corner at every zero crossing, so a tone at f0 comes out
// of the rectifier with an infinite series of even harmonics falling only as
// 1/m^2 — and the harmonics beyond Nyquist fold BEFORE any envelope filter
// exists to remove them. Whichever one lands nearest DC is passed by the
// envelope at full gain and modulates the whole signal.
//
// That is not a subtle effect. Measured in-band ASR at 0 dBFS:
//
//     1x   -66.4 dB      4x   -85.0 dB      8x  -101.8 dB     16x  -104.7 dB
//
// The requirement is -100, so 4x does not meet it and 8x is the floor. The
// steps are ~1/m^2 in disguise: doubling the rate doubles the order of the
// lowest-folding harmonic, which is 12 dB of amplitude, plus whatever the
// envelope filter's rolloff adds at the new fold frequency.
//
// 16x buys only 3 dB more and is not the default. The reason it flattens out is
// worth stating because it is the opposite of `Drive`'s: there, findings.md
// records that the oversampling FILTER's stopband is the binding constraint
// rather than the ratio. Here it is not — sweeping the halfband stopband from
// 120 dB to 150 and 170 changed the result by 0.00 dB at both 4x and 8x. The
// limit is the rectifier's own harmonic series, and the only lever on it is the
// rate.
//
// WIDEBAND PROFILE, not Audio. The Audio profile stops protecting above 22 kHz,
// which is right for a saturator whose ultrasonic products are inaudible. It is
// wrong for a rectifier, whose folded products land wherever the arithmetic
// puts them — including in the middle of the audible band.
//
// AND ZERO SHAPING IS STILL BIT EXACT. With both controls at zero the gain is
// exactly one at every sample, so the node is a pure delay; it takes the
// unoversampled path and returns its input bit-for-bit. Paying an oversampler's
// round trip and its latency to compute a multiplication by one would be
// strictly worse than not doing it. Same rule as a bandless EQ declaring no
// latency.
//
// Latency is the oversampler's round trip plus the lookahead — both exact
// integers at the base rate. The lookahead is worth having: the differentials
// peak a short way INTO a transient, so without it the boost lands slightly
// behind the hit it was computed from.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "anv/core/Node.h"
#include "anv/dsp/DelayLine.h"
#include "anv/dsp/Dynamics.h"
#include "anv/dsp/Oversampler.h"

namespace anv {

class TransientDesigner final : public Node {
public:
  const char* type() const override { return "transient_designer"; }

  void prepare(const PrepareSpec& spec) override {
    spec_ = spec;
    rebuild();
  }

  void reset() override {
    for (auto& d : fast_) d.reset(0.0);
    for (auto& d : slowAttack_) d.reset(0.0);
    for (auto& d : slowRelease_) d.reset(0.0);
    look_.reset();
    os_.reset();
    osDelay_.reset();
    for (auto& v : dly_) std::fill(v.begin(), v.end(), 0.0);
  }

  void setParams(const ParamSet& p) override {
    // Bounded well inside clampDb's +/-80. Both controls can be at their
    // extremes at once only in theory — A is large while rising and S while
    // falling — but the bound has to hold for the arithmetic, not for the
    // music, so 18 + 18 dB is the worst case the robustness test sees.
    attackDb_ = std::clamp(p.num("attack_db", 0.0), -kMaxShapeDb, kMaxShapeDb);
    sustainDb_ = std::clamp(p.num("sustain_db", 0.0), -kMaxShapeDb, kMaxShapeDb);
    attackMs_ = std::clamp(p.num("attack_ms", 20.0), 1.0, 200.0);
    sustainMs_ = std::clamp(p.num("sustain_ms", 250.0), 50.0, 3000.0);
    lookaheadMs_ = std::clamp(p.num("lookahead_ms", 2.0), 0.0, 200.0);
    stereoLink_ = std::clamp(p.num("stereo_link", 1.0), 0.0, 1.0);
    // 8x IS THE FLOOR, NOT A ROUND NUMBER. Measured in-band ASR at 0 dBFS:
    // 1x -66.4 dB, 4x -85.0, 8x -101.8, 16x -104.7. The requirement is -100,
    // so 4x does not meet it and 16x costs twice as much to buy 3 dB. See the
    // header comment for why the curve has that shape.
    int f = static_cast<int>(p.num("oversample_factor", 8.0));
    f = std::clamp(f, 8, 16);
    int pow2 = 8;
    while (pow2 * 2 <= f) pow2 *= 2;
    factor_ = pow2;
    if (spec_.numChannels > 0) rebuild();
  }

  int latencyFrames() const override { return latency_; }

  void process(ProcessContext& ctx) noexcept override {
    const int n = ctx.numFrames;
    const int ch = spec_.numChannels;

    // NOTHING TO SHAPE IS NOTHING TO OVERSAMPLE. With both controls at zero the
    // gain is pow(10, 0/20) at every sample — exactly one — so the node IS a
    // delay, algebraically and not approximately. Running the oversampler's
    // round trip anyway would replace a bit-exact identity with a -140 dB
    // approximation and charge latency for it. Same rule as a bandless EQ
    // declaring no latency: a node that does nothing declares nothing.
    if (!shaping_) {
      for (int c = 0; c < ch; ++c) {
        const size_t cc = static_cast<size_t>(c);
        inPtr_[cc] = ctx.in[c];
        outPtr_[cc] = ctx.out[c];
      }
      look_.process(inPtr_.data(), outPtr_.data(), n);
      return;
    }

    // Upsample every channel first: stereo linking needs all channels'
    // differentials at the same oversampled instant.
    const int m = n * factor_;
    for (int c = 0; c < ch; ++c) {
      const size_t cc = static_cast<size_t>(c);
      work_[cc] = os_.up(ctx.in[c], n, c);
      dlyPtr_[cc] = dly_[cc].data();
      osIn_[cc] = work_[cc];
    }
    // The lookahead is applied at the OVERSAMPLED rate, so it is exactly
    // `lookahead * factor` there and exactly `lookahead` base samples — the
    // detectors still read the undelayed signal and the gain still leads the
    // audio by the declared amount.
    osDelay_.process(osIn_.data(), dlyPtr_.data(), m);

    for (int i = 0; i < m; ++i) {
      for (int c = 0; c < ch; ++c) {
        const size_t cc = static_cast<size_t>(c);
        const double x = std::abs(work_[cc][i]);

        // `fast` and `slowAttack` share a release coefficient on purpose: the
        // decoupled detector's release branch is then bit-identical between
        // them, so A isolates the ATTACK smoothing and nothing else. Likewise
        // `fast` and `slowRelease` share an attack coefficient, so S isolates
        // the release.
        const double fast = fast_[cc].process(x, alphaFastAttack_, alphaFastRelease_);
        const double slowA =
            slowAttack_[cc].process(x, alphaSlowAttack_, alphaFastRelease_);
        const double slowR =
            slowRelease_[cc].process(x, alphaFastAttack_, alphaSlowRelease_);

        // Floored, not guarded by a branch: a branch on silence would make the
        // node's behaviour depend on how close to zero the input happened to
        // be, and kLevelFloor is already the engine's answer to that question.
        const double f = std::max(fast, kLevelFloor);
        const double r = std::max(slowR, kLevelFloor);
        chAttack_[cc] = std::max(0.0, 1.0 - std::max(slowA, 0.0) / f);
        // NO max(0, .) HERE, AND THAT ASYMMETRY IS THE POINT. The attack
        // differential genuinely goes negative through a decay and has to be
        // rectified. The sustain differential cannot: `slowRelease` has the
        // same attack coefficient as `fast` and a slower release, so its
        // detector state dominates `fast`'s at every sample — by induction, and
        // bit-exactly, because IEEE rounding is monotone. A guard here would
        // never fire, and a guard that never fires tells the next reader this
        // quantity can go negative when it provably cannot. Mutation testing
        // found it: removing it changed nothing, at any setting.
        chSustain_[cc] = 1.0 - f / r;
      }

      // SEEDED FROM CHANNEL 0, NOT FROM ZERO. Starting the running maximum at
      // zero would rectify a second time, and a property with two
      // implementations is a property no test can falsify: mutation-testing
      // this node showed that removing the `max(0, .)` above changed NOTHING at
      // stereo_link = 1, because this loop silently put it back. The
      // rectification is a statement about the differentials and it is made in
      // exactly one place.
      double linkedA = chAttack_[0], linkedS = chSustain_[0];
      for (int c = 1; c < ch; ++c) {
        const size_t cc = static_cast<size_t>(c);
        linkedA = std::max(linkedA, chAttack_[cc]);
        linkedS = std::max(linkedS, chSustain_[cc]);
      }

      for (int c = 0; c < ch; ++c) {
        const size_t cc = static_cast<size_t>(c);
        const double a = (1.0 - stereoLink_) * chAttack_[cc] + stereoLink_ * linkedA;
        const double s = (1.0 - stereoLink_) * chSustain_[cc] + stereoLink_ * linkedS;
        const double gainDb = attackDb_ * a + sustainDb_ * s;
        // Written back into the oversampler's own working buffer, which is
        // what down() reads.
        work_[cc][i] = dly_[cc][static_cast<size_t>(i)] * dbToLin(gainDb);
      }
    }

    for (int c = 0; c < ch; ++c) os_.down(ctx.out[c], n, c);
  }

private:
  static constexpr double kMaxShapeDb = 18.0;

  // The reference envelope. Fast enough to be on a transient's leading edge at
  // any musical tempo, slow enough that a 1 kHz waveform does not push it
  // around. The release is the boundary between "attack" and "sustain": above
  // it the sustain control takes over.
  static constexpr double kFastAttackMs = 0.5;
  static constexpr double kFastReleaseMs = 30.0;

  void rebuild() {
    const int ch = spec_.numChannels;
    if (ch <= 0) return;
    const size_t chs = static_cast<size_t>(ch);
    const size_t block = static_cast<size_t>(std::max(1, spec_.maxBlockFrames));

    shaping_ = (attackDb_ != 0.0) || (sustainDb_ != 0.0);

    lookaheadSamples_ = static_cast<int>(
        std::lround(lookaheadMs_ * 0.001 * spec_.sampleRate));
    if (lookaheadSamples_ < 0) lookaheadSamples_ = 0;

    fast_.assign(chs, DecoupledPeakDetector{});
    slowAttack_.assign(chs, DecoupledPeakDetector{});
    slowRelease_.assign(chs, DecoupledPeakDetector{});
    chAttack_.assign(chs, 0.0);
    chSustain_.assign(chs, 0.0);
    dlyPtr_.assign(chs, nullptr);
    inPtr_.assign(chs, nullptr);
    outPtr_.assign(chs, nullptr);
    osIn_.assign(chs, nullptr);
    work_.assign(chs, nullptr);

    if (!shaping_) {
      // A pure delay, at the base rate, bit exact.
      os_.prepare(1, ch, block > 0 ? static_cast<int>(block) : 1, OsProfile::Audio);
      look_.prepare(ch, lookaheadSamples_);
      osDelay_.prepare(ch, 0);
      dly_.assign(chs, std::vector<Sample>(block, 0.0));
      latency_ = lookaheadSamples_;
      reset();
      return;
    }

    // WIDEBAND, NOT THE AUDIO PROFILE. The Audio profile stops protecting above
    // 22 kHz, which is right for a saturator whose products up there are
    // inaudible. It is wrong for a rectifier: its harmonic series is what folds,
    // and where a fold lands is decided by arithmetic rather than by audibility,
    // so anything the decimation filter was not asked to stop comes back into
    // the middle of the band.
    os_.prepare(factor_, ch, static_cast<int>(block), OsProfile::Wideband);

    // Ballistics are coefficients per SAMPLE, so they follow the oversampled
    // rate. Deriving them from the base rate would make every time constant
    // `factor` times too fast — the node would change character with a setting
    // that is supposed to change only its aliasing.
    const double osRate = spec_.sampleRate * static_cast<double>(factor_);
    alphaFastAttack_ = timeConstantAlpha(kFastAttackMs * 0.001, osRate);
    alphaFastRelease_ = timeConstantAlpha(kFastReleaseMs * 0.001, osRate);
    alphaSlowAttack_ = timeConstantAlpha(attackMs_ * 0.001, osRate);
    alphaSlowRelease_ = timeConstantAlpha(sustainMs_ * 0.001, osRate);

    look_.prepare(ch, 0);
    osDelay_.prepare(ch, lookaheadSamples_ * factor_);
    dly_.assign(chs, std::vector<Sample>(block * static_cast<size_t>(factor_), 0.0));

    // Both terms are exact integers at the base rate: the oversampler's round
    // trip by construction (see Oversampler.h), and the lookahead because
    // `lookahead * factor` oversampled samples is `lookahead` base samples.
    latency_ = os_.latencyFrames() + lookaheadSamples_;
    reset();
  }

  double attackDb_ = 0.0, sustainDb_ = 0.0;
  double attackMs_ = 20.0, sustainMs_ = 250.0;
  double lookaheadMs_ = 2.0, stereoLink_ = 1.0;
  int factor_ = 8; // the clamp floor; see setParams for why 8 and not 4
  bool shaping_ = false;

  double alphaFastAttack_ = 0.0, alphaFastRelease_ = 0.0;
  double alphaSlowAttack_ = 0.0, alphaSlowRelease_ = 0.0;
  int lookaheadSamples_ = 0, latency_ = 0;

  std::vector<DecoupledPeakDetector> fast_, slowAttack_, slowRelease_;
  std::vector<double> chAttack_, chSustain_;
  DelayLine look_, osDelay_;
  Oversampler os_;
  std::vector<std::vector<Sample>> dly_;
  std::vector<Sample*> dlyPtr_, work_;
  std::vector<const Sample*> inPtr_, osIn_;
  std::vector<Sample*> outPtr_;
};

} // namespace anv
