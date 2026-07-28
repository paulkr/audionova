// Gate / downward expander.
//
// THE OFFLINE SUPERPOWER IS THAT THE GATE OPENS BEFORE THE TRANSIENT.
//
// Every real-time gate has the same signature flaw: it cannot know a hit is
// coming, so the gain is still down when the attack arrives and the first
// milliseconds get swallowed. Engineers work around it with absurdly fast
// attack times, which then chatter on noise. We are offline, so the detector
// runs on the undelayed signal while the audio is delayed, and a running
// maximum over the lookahead window means the gate is already fully open when
// the transient lands. The attack can then be set for smoothness rather than
// for damage control.
//
// POLARITY. The compressor feeds the shared detector its gain REDUCTION
// (rising = louder = attack). The gate feeds it GAIN (rising = opening =
// attack). Same detector, same code path, opposite sign — which is why the
// detector is documented in terms of its input rather than in terms of gain.
//
// Hysteresis and hold both exist to stop chatter, and they solve different
// halves of it: hysteresis stops a signal hovering AT the threshold from
// re-triggering, hold stops a signal DECAYING through the threshold from
// closing between cycles of a low note.
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

class GateExpander final : public Node {
public:
  const char* type() const override { return "gate_expander"; }

  void prepare(const PrepareSpec& spec) override {
    spec_ = spec;
    rebuild();
  }

  void reset() override {
    for (auto& f : scHp_) f.reset();
    for (auto& r : rmax_) r.reset();
    // Start closed. Opening at unity would let the first block through before
    // the detector settles.
    for (auto& d : det_) d.reset(-exp_.rangeDb);
    for (auto& o : rmsEnv_) o.reset(0.0);
    std::fill(holdCounter_.begin(), holdCounter_.end(), 0);
    std::fill(triggered_.begin(), triggered_.end(), 0);
    audioDelay_.reset();
  }

  void setParams(const ParamSet& p) override {
    exp_.thresholdDb = clampDb(p.num("threshold_dbfs", -40.0));
    exp_.ratio = std::max(1.0, p.num("ratio_to_1", 4.0));
    exp_.kneeDb = std::max(0.0, p.num("knee_db", 3.0));
    exp_.rangeDb = std::max(0.0, p.num("range_db", 60.0));

    attackMs_ = std::clamp(p.num("attack_ms", 1.0), 0.01, 10000.0);
    holdMs_ = std::clamp(p.num("hold_ms", 50.0), 0.0, 10000.0);
    releaseMs_ = std::clamp(p.num("release_ms", 100.0), 1.0, 10000.0);
    // Bounded above because it sizes a delay line and a running-max deque.
    lookaheadMs_ = std::clamp(p.num("lookahead_ms", 2.0), 0.0, 200.0);
    hysteresisDb_ = std::max(0.0, p.num("hysteresis_db", 3.0));
    scHpHz_ = p.num("sidechain_hp_hz", 0.0);
    stereoLink_ = std::clamp(p.num("stereo_link", 1.0), 0.0, 1.0);
    detectorRms_ = (p.str("detector", "peak") == "rms");

    if (spec_.numChannels > 0) rebuild();
  }

  int latencyFrames() const override { return lookaheadSamples_; }

  void process(ProcessContext& ctx) noexcept override {
    const int n = ctx.numFrames;
    const int ch = spec_.numChannels;

    for (int c = 0; c < ch; ++c) {
      delayedPtr_[static_cast<size_t>(c)] = delayed_[static_cast<size_t>(c)].data();
      inPtr_[static_cast<size_t>(c)] = ctx.in[c];
    }
    audioDelay_.process(inPtr_.data(), delayedPtr_.data(), n);

    const double openThreshold = exp_.thresholdDb;
    const double closeThreshold = exp_.thresholdDb - hysteresisDb_;

    for (int i = 0; i < n; ++i) {
      double linked = 0.0;
      for (int c = 0; c < ch; ++c) {
        double s = ctx.in[c][i];
        if (scHpHz_ > 0.0) s = scHp_[static_cast<size_t>(c)].processSample(s);
        const double level =
            detectorRms_
                ? std::sqrt(std::max(0.0, rmsEnv_[static_cast<size_t>(c)].process(s * s)))
                : std::abs(s);
        chLevel_[static_cast<size_t>(c)] = level;
        linked = std::max(linked, level);
      }

      for (int c = 0; c < ch; ++c) {
        const size_t ci = static_cast<size_t>(c);
        const double level =
            (1.0 - stereoLink_) * chLevel_[ci] + stereoLink_ * linked;
        const double levelDb = linToDb(level);

        // Trigger / hold state machine.
        if (levelDb >= openThreshold) {
          triggered_[ci] = 1;
          holdCounter_[ci] = holdSamples_;
        } else if (triggered_[ci] && levelDb >= closeThreshold) {
          holdCounter_[ci] = holdSamples_;
        } else if (holdCounter_[ci] > 0) {
          --holdCounter_[ci];
        } else {
          triggered_[ci] = 0;
        }

        // Hysteresis: once open, it takes a deeper drop to start closing.
        const double effThreshold = triggered_[ci] ? closeThreshold : openThreshold;
        ExpanderComputer curve = exp_;
        curve.thresholdDb = effThreshold;

        const double reduction =
            (holdCounter_[ci] > 0) ? 0.0 : curve.reductionDb(levelDb);

        // Anticipate in the GAIN domain: if any sample in the coming window
        // wants the gate open, open now.
        const double targetGainDb = -reduction;
        const double anticipated = rmax_[ci].process(targetGainDb);
        const double smoothed =
            det_[ci].process(anticipated, alphaAttack_, alphaRelease_);

        ctx.out[c][i] =
            delayed_[ci][static_cast<size_t>(i)] * dbToLin(smoothed);
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
    holdSamples_ = static_cast<int>(std::lround(holdMs_ * 0.001 * sr));

    alphaAttack_ = timeConstantAlpha(attackMs_ * 0.001, sr);
    alphaRelease_ = timeConstantAlpha(releaseMs_ * 0.001, sr);

    scHp_.assign(static_cast<size_t>(ch), TptSvf{});
    for (auto& f : scHp_) {
      f.setSampleRate(sr);
      if (scHpHz_ > 0.0) f.configure(FilterShape::HighPass, scHpHz_, 0.0, 0.707);
    }

    rmax_.assign(static_cast<size_t>(ch), RunningMax{});
    for (auto& r : rmax_) r.prepare(lookaheadSamples_ + 1);

    det_.assign(static_cast<size_t>(ch), DecoupledPeakDetector{});
    rmsEnv_.assign(static_cast<size_t>(ch), OnePole{});
    for (auto& o : rmsEnv_) o.setAlpha(timeConstantAlpha(0.010, sr));

    holdCounter_.assign(static_cast<size_t>(ch), 0);
    triggered_.assign(static_cast<size_t>(ch), 0);
    chLevel_.assign(static_cast<size_t>(ch), 0.0);

    audioDelay_.prepare(ch, lookaheadSamples_);
    delayed_.assign(static_cast<size_t>(ch),
                    std::vector<Sample>(static_cast<size_t>(spec_.maxBlockFrames), 0.0));
    delayedPtr_.resize(static_cast<size_t>(ch));
    inPtr_.resize(static_cast<size_t>(ch));

    reset();
  }

  ExpanderComputer exp_;
  double attackMs_ = 1.0, holdMs_ = 50.0, releaseMs_ = 100.0;
  double lookaheadMs_ = 2.0, hysteresisDb_ = 3.0;
  double scHpHz_ = 0.0, stereoLink_ = 1.0;
  bool detectorRms_ = false;

  int lookaheadSamples_ = 0, holdSamples_ = 0;
  double alphaAttack_ = 0.0, alphaRelease_ = 0.0;

  std::vector<TptSvf> scHp_;
  std::vector<RunningMax> rmax_;
  std::vector<DecoupledPeakDetector> det_;
  std::vector<OnePole> rmsEnv_;
  std::vector<int> holdCounter_;
  std::vector<int> triggered_;
  std::vector<double> chLevel_;
  DelayLine audioDelay_;
  std::vector<std::vector<Sample>> delayed_;
  std::vector<Sample*> delayedPtr_;
  std::vector<const Sample*> inPtr_;
};

} // namespace anv
