// Delay, with in-loop damping and stereo routing.
//
// LATENCY IS ZERO. The delay time is the EFFECT, not latency — declaring it as
// latency would make the chain compensate it away and the node would do
// nothing audible. This distinction bites people: a delay and a lookahead
// limiter both hold audio back, but only one of them is asking the rest of the
// graph to wait.
//
// The tail, on the other hand, is real and must be declared, or the chain
// stops rendering while repeats are still sounding and the delay is cut off
// mid-decay.
//
// The damping filters sit INSIDE the feedback loop, so their effect compounds
// with each repeat — which is the point. Successive echoes get progressively
// darker, the way an analog or tape delay behaves, rather than every repeat
// being an identical copy at lower level.
#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "anv/core/Node.h"
#include "anv/dsp/FractionalDelay.h"
#include "anv/dsp/TptSvf.h"

namespace anv {

class Delay final : public Node {
public:
  const char* type() const override { return "delay"; }

  void prepare(const PrepareSpec& spec) override {
    spec_ = spec;
    rebuild();
  }

  void reset() override {
    for (auto& l : lines_) l.reset();
    for (auto& f : lp_) f.reset();
    for (auto& f : hp_) f.reset();
  }

  void setParams(const ParamSet& p) override {
    timeMs_ = std::clamp(p.num("time_ms", 250.0), 1.0, 2000.0);
    tempoBpm_ = p.num("tempo_bpm", 0.0);
    noteDivision_ = p.str("time_note_division", "");

    // Feedback is capped below unity. The in-loop damping means the true loop
    // gain is already under this at most frequencies, but a delay that can be
    // driven to self-oscillation is a delay that can blow up a render.
    feedback_ = std::clamp(p.num("feedback_percent", 30.0), 0.0, 95.0) / 100.0;
    hfDampHz_ = std::clamp(p.num("hf_damp_hz", 8000.0), 1000.0, 20000.0);
    lfDampHz_ = std::clamp(p.num("lf_damp_hz", 100.0), 20.0, 800.0);
    stereoMode_ = p.str("stereo_mode", "mono");
    spreadMs_ = std::clamp(p.num("spread_ms", 0.0), 0.0, 60.0);
    freeze_ = p.num("freeze", 0.0) > 0.5;
    maxTailS_ = std::clamp(p.num("max_tail_s", 8.0), 0.1, 30.0);

    if (spec_.numChannels > 0) rebuild();
  }

  // The delay time is the effect. Nothing to compensate.
  int latencyFrames() const override { return 0; }

  int64_t tailFrames() const override { return tail_; }

  void process(ProcessContext& ctx) noexcept override {
    const int n = ctx.numFrames;
    const int ch = spec_.numChannels;
    const bool pingPong = (stereoMode_ == "ping_pong") && ch >= 2;
    const double g = freeze_ ? 1.0 : feedback_;
    const double inGain = freeze_ ? 0.0 : 1.0;

    for (int i = 0; i < n; ++i) {
      if (pingPong) {
        // Cross-coupled: each side's output feeds the other's input, so
        // repeats alternate across the image. The dry input is summed to mono
        // and injected on the left only, which is what makes the very first
        // repeat land on one side rather than in the centre.
        const double dl = lines_[0].read();
        const double dr = lines_[1].read();
        const double fl = damp(0, dr);
        const double fr = damp(1, dl);

        double mono = 0.0;
        for (int c = 0; c < ch; ++c) mono += ctx.in[c][i];
        mono /= static_cast<double>(ch);

        lines_[0].write(inGain * mono + g * fl);
        lines_[1].write(g * fr);

        ctx.out[0][i] = dl;
        ctx.out[1][i] = dr;
        for (int c = 2; c < ch; ++c) ctx.out[c][i] = 0.0;
      } else {
        for (int c = 0; c < ch; ++c) {
          const double d = lines_[static_cast<size_t>(c)].read();
          lines_[static_cast<size_t>(c)].write(inGain * ctx.in[c][i] +
                                               g * damp(c, d));
          ctx.out[c][i] = d;
        }
      }
    }
  }

private:
  inline double damp(int c, double x) noexcept {
    const size_t ci = static_cast<size_t>(c);
    return hp_[ci].processSample(lp_[ci].processSample(x));
  }

  double resolvedTimeMs() const {
    if (noteDivision_.empty() || tempoBpm_ <= 0.0) return timeMs_;
    const double quarter = 60000.0 / tempoBpm_;
    if (noteDivision_ == "quarter") return quarter;
    if (noteDivision_ == "eighth") return quarter * 0.5;
    if (noteDivision_ == "dotted_eighth") return quarter * 0.75;
    if (noteDivision_ == "triplet_eighth") return quarter / 3.0;
    if (noteDivision_ == "sixteenth") return quarter * 0.25;
    return timeMs_;
  }

  void rebuild() {
    const int ch = spec_.numChannels;
    if (ch <= 0) return;
    const double sr = spec_.sampleRate;

    const double baseMs = std::clamp(resolvedTimeMs(), 1.0, 2000.0);
    const double longestMs = baseMs + spreadMs_;
    const int maxSamples = static_cast<int>(std::ceil(longestMs * 0.001 * sr)) + 8;

    lines_.assign(static_cast<size_t>(ch), FractionalDelayLine{});
    lp_.assign(static_cast<size_t>(ch), TptSvf{});
    hp_.assign(static_cast<size_t>(ch), TptSvf{});

    double longestSamples = 0.0;
    for (int c = 0; c < ch; ++c) {
      // "stereo_offset" detunes the right-hand side so the two taps do not sit
      // on top of each other; every other mode uses the same time on all
      // channels.
      const double ms =
          (stereoMode_ == "stereo_offset" && c > 0) ? baseMs + spreadMs_ : baseMs;
      const double samples = ms * 0.001 * sr;
      longestSamples = std::max(longestSamples, samples);

      lines_[static_cast<size_t>(c)].prepare(maxSamples);
      lines_[static_cast<size_t>(c)].setDelay(samples);

      lp_[static_cast<size_t>(c)].setSampleRate(sr);
      lp_[static_cast<size_t>(c)].configure(FilterShape::LowPass, hfDampHz_, 0.0, 0.707);
      hp_[static_cast<size_t>(c)].setSampleRate(sr);
      hp_[static_cast<size_t>(c)].configure(FilterShape::HighPass, lfDampHz_, 0.0, 0.707);
    }

    // Repeats needed to fall below -140 dBFS, times the delay time. The in-loop
    // damping makes the real decay faster than this, so it is a conservative
    // bound — and it is capped, because at 95% feedback the honest answer is
    // over two minutes.
    const double g = freeze_ ? 0.95 : feedback_;
    double repeats = 1.0;
    if (g > 0.0) repeats = std::ceil(std::log(1e-7) / std::log(g));
    const double capSamples = maxTailS_ * sr;
    tail_ = static_cast<int64_t>(
        std::min(capSamples, std::max(longestSamples, repeats * longestSamples)));

    reset();
  }

  double timeMs_ = 250.0, tempoBpm_ = 0.0, feedback_ = 0.3;
  double hfDampHz_ = 8000.0, lfDampHz_ = 100.0, spreadMs_ = 0.0;
  double maxTailS_ = 8.0;
  std::string noteDivision_, stereoMode_ = "mono";
  bool freeze_ = false;
  int64_t tail_ = 0;

  std::vector<FractionalDelayLine> lines_;
  std::vector<TptSvf> lp_, hp_;
};

} // namespace anv
