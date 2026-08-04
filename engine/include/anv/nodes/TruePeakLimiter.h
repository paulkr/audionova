// True-peak limiter — the mandatory terminal node of every export chain.
//
// ZERO OVERSHOOT IS A PROOF HERE, NOT A HOPE.
//
// The compressor deliberately does not make this guarantee: its exponential
// attack asymptotically approaches the target and never quite arrives, which
// is musically correct for a compressor. A delivery limiter cannot work that
// way — a ceiling that is exceeded "rarely" is a ceiling that clips.
//
// The construction:
//
//   r[n]  required gain          = min(1, ceiling / truePeak[n])
//   m[n]  running MINIMUM of r over a forward window of W samples
//   s[n]  Hann-weighted average of m over W samples
//
// Claim: s[n] <= r[n-W+1], so applying s[n] to the audio sample delayed by
// W-1 can never exceed the ceiling.
//
// Proof: s[n] = sum_k h[k]*m[n-k] with h[k] >= 0 and sum h[k] = 1. Each
// m[n-k] is a minimum over the window [n-k-W+1, n-k]. For every k in [0, W-1]
// that window contains n-W+1, so m[n-k] <= r[n-W+1]. Therefore
// s[n] <= sum_k h[k]*r[n-W+1] = r[n-W+1].  QED
//
// Both halves are load-bearing. The running minimum alone would step the gain
// discontinuously; Hann smoothing alone would let the average sit above the
// instantaneous requirement. Together they give a gain that is smooth AND
// provably never too large.
//
// WHAT THE PROOF DOES NOT COVER, AND WHAT IS DONE ABOUT IT. The proof binds
// the output to `truePeak[n]`, which is what the DETECTOR reported — not to
// the true peak. The detector searches a grid of `factor` points per sample
// and so can only ever read LOW, by up to TruePeakUpsampler::gridBoundDb().
// A limiter that lands exactly on a ceiling using an instrument that reads
// 0.04 dB low delivers a peak 0.04 dB above what the user asked for, and
// permissive is the one direction a delivery ceiling may not err in. So the
// target is lowered by exactly that bound before anything else happens; see
// rebuild(). The measurement suite caught this as a 0.021 dB overshoot, and
// an exact DFT-domain reconstruction put the real figure at 0.114 dB.
//
// THE AUDIO NEVER PASSES THROUGH THE OVERSAMPLER. Only a copy is interpolated,
// purely to measure inter-sample peaks; the gain envelope is then applied to
// the untouched base-rate signal. Running the programme through an 8x SRC
// round trip to limit it would impose the resampler's own passband ripple and
// phase response on every single render, for no benefit.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "anv/core/Node.h"
#include "anv/dsp/DelayLine.h"
#include "anv/dsp/Dynamics.h"
#include "anv/dsp/TruePeak.h"

namespace anv {

class TruePeakLimiter final : public Node {
public:
  const char* type() const override { return "true_peak_limiter"; }

  void prepare(const PrepareSpec& spec) override {
    spec_ = spec;
    rebuild();
  }

  void reset() override {
    upsampler_.reset();
    rmax_.reset();
    hann_.reset(1.0);
    audioDelay_.reset();
    releaseGain_ = 1.0;
  }

  void setParams(const ParamSet& p) override {
    ceilingDbTp_ = std::min(p.num("ceiling_dbtp", -1.0), 0.0);
    // Upper bound matters as much as the lower one. These drive ALLOCATION:
    // the lookahead window sizes a running-max deque and a Hann FIR that is
    // O(window) per sample, so an unbounded value is a resource-exhaustion
    // vector, not merely a silly setting. 200 ms is already forty times any
    // sane limiter lookahead.
    lookaheadMs_ = std::clamp(p.num("lookahead_ms", 2.0), 0.05, 200.0);
    releaseMs_ = std::clamp(p.num("release_ms", 100.0), 1.0, 10000.0);

    int f = static_cast<int>(p.num("oversample_factor", 16.0));
    // 16x FLOOR, NOT 4x. BS.1770 names 4x as its minimum and this node used to
    // take it, on a measurement that said oversampling factor did not affect
    // accuracy. That measurement was made on stationary signals and did not
    // survive being repeated on transients; see the corrected table in
    // TruePeak.h. The grid a peak search runs on is bounded by
    // cos(pi/(2*factor)), which is 0.688 dB at 4x — and at the DELIVERY rate,
    // where this node runs, 20 kHz of programme sits at 0.907 of Nyquist, so
    // that bound is reachable by ordinary music rather than by a torture
    // signal. 16x brings it to 0.042 dB for four times the detector cost
    // (measured: 1.02 s -> 4.07 s per 30 s of stereo at 96 kHz).
    //
    // A caller asking for less gets more. Clamping upward is safe in the one
    // direction this node cares about, and the alternative — honouring a 4
    // that a language model wrote because it read BS.1770 — is a ceiling that
    // is quietly half a dB permissive.
    factor_ = std::clamp(f, 16, 32);
    // Optional EXTRA headroom on top of the detector's own error bound, which
    // is already subtracted unconditionally in rebuild(). Zero by default,
    // because the unconditional term is the one derived from a measurement and
    // this one is the caller's own risk appetite.
    safetyMarginDb_ = std::max(0.0, p.num("safety_margin_db", 0.0));

    if (spec_.numChannels > 0) rebuild();
  }

  int latencyFrames() const override { return latency_; }

  void process(ProcessContext& ctx) noexcept override {
    const int n = ctx.numFrames;
    const int ch = spec_.numChannels;

    for (int c = 0; c < ch; ++c) {
      delayedPtr_[static_cast<size_t>(c)] = delayed_[static_cast<size_t>(c)].data();
      inPtr_[static_cast<size_t>(c)] = ctx.in[c];
    }
    audioDelay_.process(inPtr_.data(), delayedPtr_.data(), n);

    for (int i = 0; i < n; ++i) {
      // Gain is LINKED across channels. Limiting channels independently would
      // shift the stereo image on every peak.
      double truePeak = 0.0;
      for (int c = 0; c < ch; ++c)
        truePeak = std::max(truePeak, upsampler_.processMaxAbs(ctx.in[c][i], c));

      const double required =
          (truePeak > ceilingLin_) ? (ceilingLin_ / truePeak) : 1.0;

      // Running minimum, expressed through the already brute-force-verified
      // RunningMax: min(x) == -max(-x).
      const double windowMin = -rmax_.process(-required);
      const double smoothed = hann_.process(windowMin);

      // Release. Gain falls instantly and recovers slowly, and either way it
      // may only ever end up SMALLER than s[n], so the proof above still holds.
      //
      // The final clamp is redundant and deliberately kept: on this branch
      // releaseGain_ < smoothed, and a one-pole step toward smoothed is a
      // convex combination of the two, so it cannot overshoot. Mutation
      // testing confirms deleting it changes nothing. It stays as a cheap
      // guard on the one invariant this whole node exists to provide, but it
      // is not load-bearing logic and should not be read as such.
      if (smoothed < releaseGain_) {
        releaseGain_ = smoothed;
      } else {
        releaseGain_ += (smoothed - releaseGain_) * (1.0 - alphaRelease_);
        releaseGain_ = std::min(releaseGain_, smoothed);
      }

      for (int c = 0; c < ch; ++c)
        ctx.out[c][i] =
            delayed_[static_cast<size_t>(c)][static_cast<size_t>(i)] * releaseGain_;
    }
  }

private:
  void rebuild() {
    const int ch = spec_.numChannels;
    if (ch <= 0) return;
    const double sr = spec_.sampleRate;

    alphaRelease_ = timeConstantAlpha(releaseMs_ * 0.001, sr);

    upsampler_.design(factor_, 80.0, 0.05);
    upsampler_.prepare(ch);

    // AIM UNDER THE CEILING BY WHAT THE INSTRUMENT CANNOT SEE.
    //
    // The proof in the header bounds the output against what the DETECTOR
    // reported, and that is all it can bound. The detector reads a grid rather
    // than a continuum, so it under-reports by up to gridBoundDb(); hitting
    // "-1.000 dBTP" on an instrument that reads low is how a limiter delivers
    // a peak above the number the user asked for. Subtracting the detector's
    // own bound turns the guarantee from "the ceiling on our meter" into "the
    // ceiling", for any signal band-limited to base Nyquist.
    //
    // This is unconditional rather than a default the caller can switch off.
    // Permissive is the one direction a delivery ceiling may not err in, and
    // the cost of being conservative is 0.042 dB of loudness at the 16x floor
    // — below the 0.1 dB an engineer can hear on a level match.
    //
    // What it does NOT cover is the detector's transition band, which is
    // unsigned and unbounded and which no margin of a sensible size would
    // cover; see TruePeak.h term B and docs/open-questions.md.
    detectorMarginDb_ = upsampler_.gridBoundDb();
    ceilingLin_ = dbToLin(ceilingDbTp_ - safetyMarginDb_ - detectorMarginDb_);

    window_ = std::max(2, static_cast<int>(std::lround(lookaheadMs_ * 0.001 * sr)));
    rmax_.prepare(window_);
    hann_.prepare(window_);

    // See the header proof: the gain computed at detector step n applies to
    // the audio sample W-1 earlier, and the detector itself lags the input by
    // the interpolator's group delay.
    latency_ = upsampler_.latencyFrames() + window_ - 1;

    audioDelay_.prepare(ch, latency_);
    delayed_.assign(static_cast<size_t>(ch),
                    std::vector<Sample>(static_cast<size_t>(spec_.maxBlockFrames), 0.0));
    delayedPtr_.resize(static_cast<size_t>(ch));
    inPtr_.resize(static_cast<size_t>(ch));

    reset();
  }

  double ceilingDbTp_ = -1.0, lookaheadMs_ = 2.0, releaseMs_ = 100.0;
  double ceilingLin_ = 1.0, alphaRelease_ = 0.0, safetyMarginDb_ = 0.0;
  double detectorMarginDb_ = 0.0;
  int factor_ = 16, window_ = 2, latency_ = 0;
  double releaseGain_ = 1.0;

  TruePeakUpsampler upsampler_;
  RunningMax rmax_;
  HannSmoother hann_;
  DelayLine audioDelay_;
  std::vector<std::vector<Sample>> delayed_;
  std::vector<Sample*> delayedPtr_;
  std::vector<const Sample*> inPtr_;
};

} // namespace anv
