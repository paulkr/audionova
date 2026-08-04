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
// A AND S ARE HALF-WAVE RECTIFIED, and that is what keeps the two controls from
// fighting. Without `max(0, .)` the attack differential goes negative through
// the decay — so an attack boost would quietly cut the sustain — and the
// sustain differential goes negative through the onset. Rectified, each control
// touches only the region it is named for, which is the property an engineer
// checks first.
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
// Latency is exactly the lookahead in samples. It is worth having: the
// differentials peak a short way INTO a transient, so without lookahead the
// boost lands slightly behind the hit it was computed from.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "anv/core/Node.h"
#include "anv/dsp/DelayLine.h"
#include "anv/dsp/Dynamics.h"

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
    if (spec_.numChannels > 0) rebuild();
  }

  int latencyFrames() const override { return lookaheadSamples_; }

  void process(ProcessContext& ctx) noexcept override {
    const int n = ctx.numFrames;
    const int ch = spec_.numChannels;

    for (int c = 0; c < ch; ++c) {
      const size_t cc = static_cast<size_t>(c);
      inPtr_[cc] = ctx.in[c];
      dlyPtr_[cc] = dly_[cc].data();
    }
    // The detectors deliberately see the UNDELAYED signal, so the gain leads
    // the audio by exactly the lookahead.
    look_.process(inPtr_.data(), dlyPtr_.data(), n);

    for (int i = 0; i < n; ++i) {
      for (int c = 0; c < ch; ++c) {
        const size_t cc = static_cast<size_t>(c);
        const double x = std::abs(ctx.in[c][i]);

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
        // BOTH CONTROLS AT ZERO IS EXACTLY UNITY. 0 * anything finite is 0,
        // 0 + 0 is 0, and pow(10, 0/20) is 1 — so a node asked to do nothing
        // returns its input bit-for-bit, delayed. There is no early-out branch
        // above precisely so that this is a property of the arithmetic rather
        // than of a special case a test could not reach.
        const double gainDb = attackDb_ * a + sustainDb_ * s;
        ctx.out[c][i] = dly_[cc][static_cast<size_t>(i)] * dbToLin(gainDb);
      }
    }
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
    const double sr = spec_.sampleRate;
    const size_t chs = static_cast<size_t>(ch);
    const size_t block = static_cast<size_t>(std::max(1, spec_.maxBlockFrames));

    alphaFastAttack_ = timeConstantAlpha(kFastAttackMs * 0.001, sr);
    alphaFastRelease_ = timeConstantAlpha(kFastReleaseMs * 0.001, sr);
    alphaSlowAttack_ = timeConstantAlpha(attackMs_ * 0.001, sr);
    alphaSlowRelease_ = timeConstantAlpha(sustainMs_ * 0.001, sr);

    lookaheadSamples_ = static_cast<int>(std::lround(lookaheadMs_ * 0.001 * sr));
    if (lookaheadSamples_ < 0) lookaheadSamples_ = 0;

    fast_.assign(chs, DecoupledPeakDetector{});
    slowAttack_.assign(chs, DecoupledPeakDetector{});
    slowRelease_.assign(chs, DecoupledPeakDetector{});
    chAttack_.assign(chs, 0.0);
    chSustain_.assign(chs, 0.0);

    look_.prepare(ch, lookaheadSamples_);
    dly_.assign(chs, std::vector<Sample>(block, 0.0));
    dlyPtr_.assign(chs, nullptr);
    inPtr_.assign(chs, nullptr);

    reset();
  }

  double attackDb_ = 0.0, sustainDb_ = 0.0;
  double attackMs_ = 20.0, sustainMs_ = 250.0;
  double lookaheadMs_ = 2.0, stereoLink_ = 1.0;

  double alphaFastAttack_ = 0.0, alphaFastRelease_ = 0.0;
  double alphaSlowAttack_ = 0.0, alphaSlowRelease_ = 0.0;
  int lookaheadSamples_ = 0;

  std::vector<DecoupledPeakDetector> fast_, slowAttack_, slowRelease_;
  std::vector<double> chAttack_, chSustain_;
  DelayLine look_;
  std::vector<std::vector<Sample>> dly_;
  std::vector<Sample*> dlyPtr_;
  std::vector<const Sample*> inPtr_;
};

} // namespace anv
