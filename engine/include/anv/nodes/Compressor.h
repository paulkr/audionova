// Compressor — log-domain, feed-forward, smooth decoupled peak detector.
//
// THE OFFLINE SUPERPOWER IS LOOKAHEAD. A real-time compressor must react after
// the fact, so a fast transient always slips through before the gain has
// finished moving; that overshoot is why plugin compressors need a limiter
// after them. We are offline, so latency is free: the detector runs on the
// UNDELAYED signal while the audio is delayed, and the running maximum over
// the lookahead window means the gain has already settled by the time the
// transient arrives. With lookahead >= attack the overshoot is mathematically
// zero, not merely small.
//
// Latency is exactly the lookahead in samples — an integer, declared, and
// verified by the contract suite.
//
// Not yet implemented, deliberately: feedback topology and an opto-style
// level-dependent release. Both are character options rather than correctness
// features, and each needs its own listening pass to be worth shipping.
#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "anv/core/Node.h"
#include "anv/dsp/DelayLine.h"
#include "anv/dsp/Dynamics.h"
#include "anv/dsp/TptSvf.h"

namespace anv {

class Compressor final : public Node {
public:
  const char* type() const override { return "compressor"; }

  void prepare(const PrepareSpec& spec) override {
    spec_ = spec;
    rebuild();
  }

  void reset() override {
    for (auto& f : scHp_) f.reset();
    for (auto& r : rmax_) r.reset();
    for (auto& d : det_) d.reset();
    for (auto& o : rmsEnv_) o.reset(0.0);
    for (auto& o : peakEnv_) o.reset(0.0);
    for (auto& o : msEnv_) o.reset(0.0);
    audioDelay_.reset();
  }

  void setParams(const ParamSet& p) override {
    comp_.thresholdDb = clampDb(p.num("threshold_dbfs", -18.0));
    comp_.ratio = std::max(1.0, p.num("ratio_to_1", 3.0));
    comp_.kneeDb = std::max(0.0, p.num("knee_db", 6.0));

    attackMs_ = std::clamp(p.num("attack_ms", 10.0), 0.01, 10000.0);
    releaseMs_ = std::clamp(p.num("release_ms", 120.0), 1.0, 10000.0);
    // Bounded above because it sizes a delay line and a running-max deque.
    lookaheadMs_ = std::clamp(p.num("lookahead_ms", 5.0), 0.0, 200.0);
    scHpHz_ = p.num("sidechain_hp_hz", 0.0);
    stereoLink_ = std::clamp(p.num("stereo_link", 1.0), 0.0, 1.0);
    detectorRms_ = (p.str("detector", "peak") == "rms");
    programRelease_ = (p.str("release_mode", "fixed") == "program");

    // Auto makeup restores the gain reduction the curve would apply to a
    // full-scale input, which is the convention every engineer expects.
    if (p.str("makeup_mode", "manual") == "auto") {
      makeupDb_ = comp_.reductionDb(0.0);
    } else {
      makeupDb_ = clampDb(p.num("makeup_gain_db", 0.0));
    }

    if (spec_.numChannels > 0) rebuild();
  }

  int latencyFrames() const override { return lookaheadSamples_; }

  void process(ProcessContext& ctx) noexcept override {
    const int n = ctx.numFrames;
    const int ch = spec_.numChannels;

    // Delay the audio; the detector deliberately sees the undelayed signal.
    for (int c = 0; c < ch; ++c) {
      delayedPtr_[static_cast<size_t>(c)] = delayed_[static_cast<size_t>(c)].data();
      inPtr_[static_cast<size_t>(c)] = ctx.in[c];
    }
    audioDelay_.process(inPtr_.data(), delayedPtr_.data(), n);

    for (int i = 0; i < n; ++i) {
      // Sidechain level per channel, then link.
      double linked = 0.0;
      for (int c = 0; c < ch; ++c) {
        double s = ctx.in[c][i];
        if (scHpHz_ > 0.0) s = scHp_[static_cast<size_t>(c)].processSample(s);

        double level;
        if (detectorRms_) {
          level = std::sqrt(std::max(0.0, rmsEnv_[static_cast<size_t>(c)].process(s * s)));
        } else {
          level = std::abs(s);
        }
        chLevel_[static_cast<size_t>(c)] = level;
        linked = std::max(linked, level);

        if (programRelease_) {
          peakEnv_[static_cast<size_t>(c)].process(std::abs(s));
          msEnv_[static_cast<size_t>(c)].process(s * s);
        }
      }

      for (int c = 0; c < ch; ++c) {
        const double level = (1.0 - stereoLink_) * chLevel_[static_cast<size_t>(c)] +
                             stereoLink_ * linked;
        const double reduction = comp_.reductionDb(linToDb(level));

        // Anticipate: the maximum reduction required across the lookahead
        // window, so the ballistics have already moved when it lands.
        const double target = rmax_[static_cast<size_t>(c)].process(reduction);

        double alphaR = alphaRelease_;
        if (programRelease_) {
          // Crest-adaptive release. Transient material (high crest) releases
          // faster to avoid audible pumping; sustained material releases
          // slower so the gain does not chatter on the body of a note.
          const double pk = peakEnv_[static_cast<size_t>(c)].value();
          const double ms = std::sqrt(std::max(msEnv_[static_cast<size_t>(c)].value(), kLevelFloor));
          const double crestDb = linToDb(std::max(pk, kLevelFloor)) - linToDb(ms);
          const double blend = std::clamp((crestDb - 6.0) / 12.0, 0.0, 1.0);
          alphaR = alphaReleaseSlow_ + blend * (alphaReleaseFast_ - alphaReleaseSlow_);
        }

        const double smoothed =
            det_[static_cast<size_t>(c)].process(target, alphaAttack_, alphaR);
        const double gain = dbToLin(makeupDb_ - smoothed);
        ctx.out[c][i] = delayed_[static_cast<size_t>(c)][static_cast<size_t>(i)] * gain;
      }
    }
  }

private:
  void rebuild() {
    const int ch = spec_.numChannels;
    if (ch <= 0) return;
    const double sr = spec_.sampleRate;

    lookaheadSamples_ = static_cast<int>(std::lround(lookaheadMs_ * 0.001 * sr));
    if (lookaheadSamples_ < 0) lookaheadSamples_ = 0;

    alphaAttack_ = timeConstantAlpha(attackMs_ * 0.001, sr);
    alphaRelease_ = timeConstantAlpha(releaseMs_ * 0.001, sr);
    alphaReleaseFast_ = timeConstantAlpha(releaseMs_ * 0.0005, sr); // half
    alphaReleaseSlow_ = timeConstantAlpha(releaseMs_ * 0.002, sr);  // double

    scHp_.assign(static_cast<size_t>(ch), TptSvf{});
    for (auto& f : scHp_) {
      f.setSampleRate(sr);
      if (scHpHz_ > 0.0) f.configure(FilterShape::HighPass, scHpHz_, 0.0, 0.707);
    }

    rmax_.assign(static_cast<size_t>(ch), RunningMax{});
    for (auto& r : rmax_) r.prepare(lookaheadSamples_ + 1);

    det_.assign(static_cast<size_t>(ch), DecoupledPeakDetector{});

    rmsEnv_.assign(static_cast<size_t>(ch), OnePole{});
    for (auto& o : rmsEnv_) o.setAlpha(timeConstantAlpha(0.010, sr)); // 10 ms RMS
    peakEnv_.assign(static_cast<size_t>(ch), OnePole{});
    for (auto& o : peakEnv_) o.setAlpha(timeConstantAlpha(0.001, sr));
    msEnv_.assign(static_cast<size_t>(ch), OnePole{});
    for (auto& o : msEnv_) o.setAlpha(timeConstantAlpha(0.150, sr));

    audioDelay_.prepare(ch, lookaheadSamples_);
    delayed_.assign(static_cast<size_t>(ch),
                    std::vector<Sample>(static_cast<size_t>(spec_.maxBlockFrames), 0.0));
    delayedPtr_.resize(static_cast<size_t>(ch));
    inPtr_.resize(static_cast<size_t>(ch));
    chLevel_.assign(static_cast<size_t>(ch), 0.0);

    reset();
  }

  GainComputer comp_;
  double attackMs_ = 10.0, releaseMs_ = 120.0, lookaheadMs_ = 5.0;
  double scHpHz_ = 0.0, stereoLink_ = 1.0, makeupDb_ = 0.0;
  bool detectorRms_ = false, programRelease_ = false;

  int lookaheadSamples_ = 0;
  double alphaAttack_ = 0.0, alphaRelease_ = 0.0;
  double alphaReleaseFast_ = 0.0, alphaReleaseSlow_ = 0.0;

  std::vector<TptSvf> scHp_;
  std::vector<RunningMax> rmax_;
  std::vector<DecoupledPeakDetector> det_;
  std::vector<OnePole> rmsEnv_, peakEnv_, msEnv_;
  DelayLine audioDelay_;
  std::vector<std::vector<Sample>> delayed_;
  std::vector<Sample*> delayedPtr_;
  std::vector<const Sample*> inPtr_;
  std::vector<double> chLevel_;
};

} // namespace anv
