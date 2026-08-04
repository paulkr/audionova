// Forward-backward (filtfilt) EQ — zero phase distortion, and impossible live.
//
// Run the filter forwards, reverse the result, run the SAME filter again, and
// reverse back. Every phase shift the first pass introduced is undone exactly
// by the second, so the total response is
//
//     H(z) * H(1/z)  =  |H(e^jw)|^2
//
// which is real and non-negative at every frequency: the magnitude is the
// SQUARE of the one-pass EQ, and the phase is identically zero. Not "nearly
// zero", not "flat within the audible band" — zero, because the impulse
// response is the autocorrelation of the one-pass response and an
// autocorrelation is even by construction.
//
// No real-time plugin can do this, and the reason is worth stating precisely
// because it also dictates the shape of the code below: a zero-phase filter's
// impulse response extends BACKWARDS in time. Output sample k depends on input
// samples after k. We render offline, so we can pay for that; a live plugin
// cannot, at any price.
//
// WHAT "ZERO PHASE" DOES NOT MEAN: zero declared latency.
//
// It is tempting to write `latencyFrames() { return 0; }` here, because
// filtfilt is famously latency-free — MATLAB's filtfilt, SciPy's filtfilt, and
// every offline EQ that advertises "zero latency" return a signal aligned with
// their input. They can, because they are handed the WHOLE SIGNAL as one
// array. This engine hands nodes blocks, in order, and asks for output before
// the rest of the file exists. Under that contract a filter that needs R
// samples of future input must declare R samples of latency. Declaring 0 would
// not make the node faster; it would make the chain trim the wrong number of
// samples off the head of the render, which is the `StereoWidth` bug in
// findings.md with the sign flipped.
//
// So: the node declares an exact integer latency, the chain compensates it,
// and the property the node is named for — phase response identically zero,
// group delay flat at zero — is what a test measures AFTER compensation. That
// is the honest reading of "zero phase", and it is the one an engineer hears.
//
// HOW THE REVERSE PASS SURVIVES BLOCK PROCESSING.
//
// The forward pass is causal and simply runs, sample by sample, forever. The
// reverse pass is the hard half: run backwards over the whole file it would
// need the whole file. Instead it runs backwards over a fixed-size window that
// extends `reach` samples PAST the block it is producing, started from rest.
// Starting from rest is wrong by exactly the influence of everything beyond
// the window — which is bounded by the filter's own impulse response at lags
// greater than `reach`, and `reach` is measured (not assumed) as the point
// where that tail sums to below 1e-12 of the response. See EqDesign.h.
//
// The window size is derived from the FILTER, never from `maxBlockFrames` —
// the same rule Convolver.h follows for its partition size, and for the same
// reason: a window derived from the caller's block size would make the output
// depend on how the host happened to chunk the audio.
//
// Cost is about three passes of the cascade per sample rather than one: one
// forward, and (chunk + reach) / chunk backward, which is exactly two at the
// chunk size chosen below. That is the price of the reverse pass and it is not
// reducible without either more latency or more error. Measured on 23 s of
// stereo it is 0.13 s — the forward-backward EQ is very nearly free, and the
// FIR sibling next door is the expensive one.
//
// KNOWN LIMIT, measured and recorded rather than hedged: past the reach cap
// below, the node is no longer exactly zero phase. A 20 Hz bell at Q 12 needs
// 5.3 s of reach and gets 0.25 s, which leaves 6.6 degrees of residual phase
// under 100 Hz against 19.3 for the same filter as `parametric_eq`. Contract
// test 23 asserts the degraded figure; findings.md explains it.
#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "anv/core/Node.h"
#include "anv/dsp/EqDesign.h"
#include "anv/dsp/TptSvf.h"

namespace anv {

class ZeroPhaseEQ final : public Node {
public:
  const char* type() const override { return "zero_phase_eq"; }

  void prepare(const PrepareSpec& spec) override {
    spec_ = spec;
    build();
  }

  void reset() override {
    for (auto& chain : forward_)
      for (auto& f : chain) f.reset();
    for (auto& f : backward_) f.reset();
    for (auto& v : window_) std::fill(v.begin(), v.end(), 0.0);
    for (auto& v : ready_) std::fill(v.begin(), v.end(), 0.0);
    std::fill(fill_.begin(), fill_.end(), 0);
    std::fill(phase_.begin(), phase_.end(), 0);
  }

  void setParams(const ParamSet& p) override {
    params_ = p;
    if (spec_.numChannels > 0) build();
  }

  // chunk + reach, exactly. Integer by construction: both are sample counts.
  int latencyFrames() const override { return latency_; }

  void process(ProcessContext& ctx) noexcept override {
    const int n = ctx.numFrames;

    if (!active_) {
      for (int c = 0; c < spec_.numChannels; ++c)
        std::copy(ctx.in[c], ctx.in[c] + n, ctx.out[c]);
      return;
    }

    for (int c = 0; c < spec_.numChannels; ++c) {
      const size_t cc = static_cast<size_t>(c);
      const Sample* src = ctx.in[c];
      Sample* dst = ctx.out[c];
      auto& chain = forward_[cc];
      auto& window = window_[cc];
      int& fill = fill_[cc];
      int& phase = phase_[cc];

      for (int i = 0; i < n; ++i) {
        // Read the input BEFORE writing the output: the chain gives separate
        // buffers, but a node that breaks when they alias is a trap for the
        // next caller.
        const Sample x = src[i];
        dst[i] = ready_[cc][static_cast<size_t>(phase)];
        if (++phase == chunk_) phase = 0;

        Sample v = x;
        for (auto& f : chain) v = f.processSample(v);
        window[static_cast<size_t>(fill)] = v;

        if (++fill == chunk_ + reach_) {
          reversePass(cc);
          // Keep the lookahead region; it becomes the head of the next window.
          for (int j = 0; j < reach_; ++j)
            window[static_cast<size_t>(j)] = window[static_cast<size_t>(chunk_ + j)];
          fill = reach_;
          phase = 0;
        }
      }
    }
  }

private:
  // The measured reach is capped here. A high-Q filter below ~40 Hz has a tail
  // that genuinely runs longer than this, and truncating it is a real
  // approximation rather than a tidy number: past the cap the reverse pass
  // starts from rest slightly too close to the output, and the node stops
  // being exactly zero phase at the very bottom of the spectrum. The cap
  // exists because latency is declared, allocated and compensated, so an
  // uncapped one is a resource-exhaustion path — the same lesson as
  // `lookahead_ms` in findings.md. 0.25 s at the canonical rate.
  static constexpr int kMaxReachFrames = 24000;

  // The tail that gets thrown away, as a fraction of the response's L1 norm.
  // -240 dB: a hundred dB below the -140 dBFS cross-platform floor, so the
  // truncation can never be the largest error in a render.
  static constexpr double kTailTolerance = 1e-12;

  std::vector<TptSvf> designCascade() const {
    std::vector<TptSvf> chain;

    const double hpHz = params_.num("hp_freq_hz", 0.0);
    if (hpHz > 0.0) {
      int slope = static_cast<int>(params_.num("hp_slope_db_oct", 12.0));
      if (slope < 12) slope = 12;
      const int order = (slope / 12) * 2;
      for (double q : eqButterworthQs(order)) {
        TptSvf f;
        f.setSampleRate(spec_.sampleRate);
        f.configure(FilterShape::HighPass, hpHz, 0.0, q);
        chain.push_back(f);
      }
    }

    const double lpHz = params_.num("lp_freq_hz", 0.0);
    if (lpHz > 0.0) {
      int slope = static_cast<int>(params_.num("lp_slope_db_oct", 12.0));
      if (slope < 12) slope = 12;
      const int order = (slope / 12) * 2;
      for (double q : eqButterworthQs(order)) {
        TptSvf f;
        f.setSampleRate(spec_.sampleRate);
        f.configure(FilterShape::LowPass, lpHz, 0.0, q);
        chain.push_back(f);
      }
    }

    for (const ParamSet& band : params_.items()) {
      TptSvf f;
      f.setSampleRate(spec_.sampleRate);
      f.configure(eqShapeFromString(band.str("filter_type", "bell")),
                  band.num("center_freq_hz", 1000.0),
                  clampDb(band.num("gain_db", 0.0)),
                  band.num("q_factor", 0.707));
      chain.push_back(f);
    }
    return chain;
  }

  void build() {
    const int ch = spec_.numChannels;
    if (ch <= 0) return;

    const std::vector<TptSvf> proto = designCascade();

    // AN EQ WITH NOTHING IN IT DECLARES NOTHING. Latency must describe what
    // the node actually does in the configuration it was prepared for; a
    // bandless EQ does nothing, so declaring a reverse-pass window here would
    // have the chain trim samples off the front of a signal that was never
    // delayed. That is `StereoWidth`'s bug, and it is cheap to not repeat.
    active_ = !proto.empty();
    if (!active_) {
      latency_ = 0;
      reach_ = 0;
      chunk_ = 0;
      forward_.clear();
      backward_.clear();
      window_.clear();
      ready_.clear();
      fill_.clear();
      phase_.clear();
      return;
    }

    const std::vector<double> h =
        cascadeImpulseResponse(proto, kMaxReachFrames, kTailTolerance);
    reach_ = static_cast<int>(h.size());

    // The window is `chunk` of output plus `reach` of lookahead, so the
    // reverse pass costs (chunk + reach) / chunk passes per output sample.
    // chunk = reach makes that exactly 2. A floor keeps very short filters
    // from turning the per-window bookkeeping into the dominant cost.
    chunk_ = std::max(reach_, 64);
    latency_ = chunk_ + reach_;

    forward_.assign(static_cast<size_t>(ch), proto);
    // One reverse cascade, not one per channel: it is started from rest at the
    // top of every window, so it carries nothing between channels to confuse.
    backward_ = proto;

    window_.assign(static_cast<size_t>(ch),
                   std::vector<Sample>(static_cast<size_t>(chunk_ + reach_), 0.0));
    ready_.assign(static_cast<size_t>(ch),
                  std::vector<Sample>(static_cast<size_t>(chunk_), 0.0));
    fill_.assign(static_cast<size_t>(ch), 0);
    phase_.assign(static_cast<size_t>(ch), 0);

    reset();
  }

  // Filter the window backwards. The last `reach` samples are warm-up and are
  // discarded; only the leading `chunk` samples become output.
  void reversePass(size_t cc) noexcept {
    for (auto& f : backward_) f.reset();
    const auto& window = window_[cc];
    auto& ready = ready_[cc];
    for (int j = chunk_ + reach_ - 1; j >= 0; --j) {
      Sample v = window[static_cast<size_t>(j)];
      for (auto& f : backward_) v = f.processSample(v);
      if (j < chunk_) ready[static_cast<size_t>(j)] = v;
    }
  }

  ParamSet params_;
  bool active_ = false;
  int reach_ = 0, chunk_ = 0, latency_ = 0;

  std::vector<std::vector<TptSvf>> forward_;
  std::vector<TptSvf> backward_;
  std::vector<std::vector<Sample>> window_, ready_;
  std::vector<int> fill_, phase_;
};

} // namespace anv
