// Multiband compressor — one compressor per band, over a crossover that adds
// nothing when the compressors are doing nothing.
//
// THE UNITY CASE IS THE SPECIFICATION. A multiband is reached for when one part
// of the spectrum misbehaves and the rest is already right. That only works if
// the bands nobody asked to move come back untouched — so with every band at
// 1:1 and 0 dB makeup this node must reduce to a PURE DELAY of its input, not
// to "close enough". Anything else means the crossover is charging rent on the
// two thirds of the spectrum the engineer never wanted to touch, and a chain
// with three multibands in it pays that rent three times.
//
// COMPLEMENTARY LINEAR-PHASE CROSSOVER, the same structure `StereoWidth` uses
// and for the same reason. Each stage low-passes what is left, then forms the
// remainder as (the input delayed by the filter's own latency) MINUS that low
// band. Low plus remainder is therefore the input delayed, exactly, at every
// frequency — magnitude and phase — because the subtraction is an identity
// rather than a filter design.
//
// A Linkwitz-Riley split cannot do this. An LR pair sums to an ALLPASS: flat in
// magnitude, phase-shifted, so a four-band LR crossover with every band at
// unity still smears the signal and still cannot null against its own input.
// That is the whole reason every "transparent" multiband compressor sounds like
// something even in bypass-per-band, and it is measurable here: the null below
// is the crossover's arithmetic floor, not a design target.
//
// ALIGN, THEN COMPRESS — NOT THE OTHER WAY ROUND.
//
// Each stage adds its own latency, so band k emerges C_k samples late where C_k
// grows with k. Every band is delayed up to the common total BEFORE its
// detector runs. Compressing first and aligning afterwards reconstructs just as
// exactly and is wrong anyway: the alignment delay would then sit between a
// band's gain and the rest of the render, so the low band would be reacting to
// a moment thousands of samples away from the one the high band was reacting
// to. The bands would be individually correct and collectively incoherent,
// which is the failure mode nobody thinks to look for because every band
// measures fine on its own.
//
// LOOKAHEAD IS SHARED, not per band. Per-band lookahead would mean per-band
// latency, and re-aligning after compression is the mistake above. One number,
// declared once, and every band's detector anticipates by the same amount.
//
// OVERSAMPLED 4x, BUT ONLY WHEN A BAND IS ACTUALLY COMPRESSING.
//
// An envelope-driven gain aliases, and this node's gain comes from a full-wave
// rectifier per band, so three bands sum three copies of the same fold-down —
// measured at -98.6 dB in-band ASR against a -100 dB requirement, which is
// almost exactly `compressor`'s own -104.2 dB plus 10*log10(3). At 4x it is
// -152.0 dB and 8x buys nothing further; the node's slowest-moving detectors
// filter out most of what folds, which is why it needs half the rate
// `transient_designer` does. See findings.md.
//
// A band at 1:1 is a different case entirely. Its gain is dbToLin(makeup) at
// every sample — a CONSTANT — and multiplying by a constant is linear, so there
// is nothing for an oversampler to prevent. Wrapping it anyway would trade the
// -319 dB unity null for the resampler's round trip at around -140 dB, which is
// a 180 dB regression in the property the whole crossover exists to deliver,
// bought to fix aliasing that a transparent configuration does not have.
//
// So the effective factor is 1 unless some band's ratio exceeds 1 — and an
// Oversampler at factor 1 is a pair of copies with zero declared latency, so
// the transparent case runs the SAME code path and stays bit exact. No second
// branch to get wrong.
//
// Latency is exactly (sum of the crossover stages) + oversampler round trip +
// lookahead. Every term is an integer sample count at the base rate by
// construction: the convolver's latency is one whole partition, the FIR's group
// delay is (N-1)/2 for an odd N, the oversampler's round trip is integer by
// design (see Oversampler.h), and `lookahead * factor` oversampled samples is
// `lookahead` base samples.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "anv/core/Node.h"
#include "anv/dsp/Convolver.h"
#include "anv/dsp/DelayLine.h"
#include "anv/dsp/Dynamics.h"
#include "anv/dsp/FirDesign.h"
#include "anv/dsp/Oversampler.h"

namespace anv {

class MultibandCompressor final : public Node {
public:
  const char* type() const override { return "multiband_compressor"; }

  void prepare(const PrepareSpec& spec) override {
    spec_ = spec;
    rebuild();
  }

  void reset() override {
    for (auto& st : stages_) {
      for (auto& cv : st.lp) cv.reset();
      st.align.reset();
    }
    os_.reset();
    for (auto& bd : bands_) {
      bd.align.reset();
      bd.look.reset();
      for (auto& d : bd.det) d.reset();
      for (auto& r : bd.rmax) r.reset();
      for (auto& v : bd.sig) std::fill(v.begin(), v.end(), 0.0);
      for (auto& v : bd.dly) std::fill(v.begin(), v.end(), 0.0);
    }
    for (auto& v : res_) std::fill(v.begin(), v.end(), 0.0);
  }

  void setParams(const ParamSet& p) override {
    // Bounded above because it sizes a delay line and a running-max deque per
    // band; an unbounded lookahead is a resource-exhaustion path rather than a
    // numerical one, so no finiteness check can see it.
    lookaheadMs_ = std::clamp(p.num("lookahead_ms", 5.0), 0.0, 200.0);
    stereoLink_ = std::clamp(p.num("stereo_link", 1.0), 0.0, 1.0);
    stopbandDb_ = std::clamp(p.num("crossover_stopband_db", 80.0), 40.0, 120.0);
    // 4x, measured: the un-oversampled node is -98.6 dB in-band ASR at 0 dBFS
    // and 4x takes it past the -100 dB requirement with room to spare. A
    // factor of 1 is allowed because it is exactly the plain `compressor`'s
    // arithmetic per band, which is what the contract test compares against.
    int f = static_cast<int>(p.num("oversample_factor", 4.0));
    f = std::clamp(f, 1, 16);
    int pow2 = 1;
    while (pow2 * 2 <= f) pow2 *= 2;
    factor_ = pow2;
    params_ = p;
    if (spec_.numChannels > 0) rebuild();
  }

  int latencyFrames() const override { return latency_; }

  // The corners actually used, after clamping, sorting and spacing. Exposed so
  // a test can assert the repair directly rather than inferring it from a
  // spectrum, where the transition band is wider than the repair.
  const std::vector<double>& crossoverFrequencies() const { return cross_; }

  void process(ProcessContext& ctx) noexcept override {
    const int n = ctx.numFrames;
    const int ch = spec_.numChannels;

    // A COMPRESSOR WITH NO BANDS DOES NOTHING, so it declares nothing. Latency
    // has to describe what the node actually does in the configuration it was
    // prepared for; declaring a crossover here would have the chain trim
    // samples off a signal that was never delayed. That is `StereoWidth`'s bug
    // in findings.md and it is cheap not to repeat.
    if (!active_) {
      for (int c = 0; c < ch; ++c) std::copy(ctx.in[c], ctx.in[c] + n, ctx.out[c]);
      return;
    }

    for (int c = 0; c < ch; ++c) {
      const size_t cc = static_cast<size_t>(c);
      std::copy(ctx.in[c], ctx.in[c] + n, res_[cc].data());
      resPtr_[cc] = res_[cc].data();
      resConst_[cc] = res_[cc].data();
    }

    // Split. Stage i takes band i off the running remainder; what is left after
    // the last stage is the top band.
    for (size_t i = 0; i < stages_.size(); ++i) {
      Stage& st = stages_[i];
      Band& bd = bands_[i];
      for (int c = 0; c < ch; ++c) {
        const size_t cc = static_cast<size_t>(c);
        st.lp[cc].process(res_[cc].data(), bd.sig[cc].data(), n);
      }
      // The remainder must be delayed by exactly what the low-pass path cost,
      // or the subtraction below is comparing two different moments and the
      // "complementary" claim is empty.
      st.align.process(resConst_.data(), resPtr_.data(), n);
      for (int c = 0; c < ch; ++c) {
        const size_t cc = static_cast<size_t>(c);
        Sample* r = res_[cc].data();
        const Sample* lo = bd.sig[cc].data();
        for (int j = 0; j < n; ++j) r[j] -= lo[j];
      }
    }
    for (int c = 0; c < ch; ++c) {
      const size_t cc = static_cast<size_t>(c);
      std::copy(res_[cc].data(), res_[cc].data() + n, bands_.back().sig[cc].data());
    }

    // Align every band to the common total, then take each band up to the
    // oversampled rate. `effFactor_` is 1 whenever no band is compressing, and
    // an Oversampler at factor 1 is a pair of copies — so the unity case stays
    // bit exact while running exactly the same code as every other case.
    const int m = n * effFactor_;
    for (size_t k = 0; k < bands_.size(); ++k) {
      Band& bd = bands_[k];
      for (int c = 0; c < ch; ++c) {
        const size_t cc = static_cast<size_t>(c);
        bd.sigPtr[cc] = bd.sig[cc].data();
        bd.sigConst[cc] = bd.sig[cc].data();
      }
      bd.align.process(bd.sigConst.data(), bd.sigPtr.data(), n);

      for (int c = 0; c < ch; ++c) {
        const size_t cc = static_cast<size_t>(c);
        bd.work[cc] = os_.up(bd.sig[cc].data(), n, osChannel(k, c));
        bd.workConst[cc] = bd.work[cc];
        bd.dlyPtr[cc] = bd.dly[cc].data();
      }
      // Lookahead at the oversampled rate: `lookahead * factor` there is
      // exactly `lookahead` base samples, so the declared latency stays an
      // integer and the detector still reads the undelayed band.
      bd.look.process(bd.workConst.data(), bd.dlyPtr.data(), m);
    }

    for (Band& bd : bands_) {
      for (int i = 0; i < m; ++i) {
        double linked = 0.0;
        for (int c = 0; c < ch; ++c) {
          const size_t cc = static_cast<size_t>(c);
          const double level = std::abs(bd.work[cc][i]);
          bd.chLevel[cc] = level;
          linked = std::max(linked, level);
        }
        for (int c = 0; c < ch; ++c) {
          const size_t cc = static_cast<size_t>(c);
          const double level =
              (1.0 - stereoLink_) * bd.chLevel[cc] + stereoLink_ * linked;
          const double reduction = bd.comp.reductionDb(linToDb(level));
          // Anticipate: the maximum reduction across the lookahead window, so
          // the ballistics have already moved when the transient lands.
          const double target = bd.rmax[cc].process(reduction);
          const double smoothed =
              bd.det[cc].process(target, bd.alphaAttack, bd.alphaRelease);
          // AT 1:1 WITH 0 dB MAKEUP THIS IS EXACTLY 1.0. The gain computer's
          // slope is 1 - 1/ratio, so a ratio of 1 gives a reduction of exactly
          // zero on every branch of the knee; the detector of a constant zero
          // is zero; and pow(10, 0/20) is 1. The unity null is therefore the
          // crossover's arithmetic floor and nothing else.
          bd.work[cc][i] =
              bd.dly[cc][static_cast<size_t>(i)] * dbToLin(bd.makeupDb - smoothed);
        }
      }
    }

    for (int c = 0; c < ch; ++c) std::fill(ctx.out[c], ctx.out[c] + n, 0.0);
    for (size_t k = 0; k < bands_.size(); ++k) {
      for (int c = 0; c < ch; ++c) {
        os_.down(bandOut_[static_cast<size_t>(c)].data(), n, osChannel(k, c));
        const Sample* src = bandOut_[static_cast<size_t>(c)].data();
        for (int i = 0; i < n; ++i) ctx.out[c][i] += src[i];
      }
    }
  }

private:
  // Crossovers below this are not a band split, they are a rumble filter, and
  // the FIR needed to place one gets very long very fast: the tap count grows
  // as 1/f for a transition band scaled to the corner. 40 Hz at the canonical
  // rate is already 12 000 taps.
  static constexpr double kMinCrossoverHz = 40.0;
  static constexpr double kMaxCrossoverHz = 18000.0;
  // Two stages at the same corner is not an error — the band between them is
  // simply empty and reconstruction is still exact — but it is never what was
  // meant, so identical corners are spaced apart after sorting.
  static constexpr double kMinCrossoverRatio = 1.05;
  // Every band costs a convolver per channel and a full crossover stage. Six is
  // already more bands than any mix engineer uses; the schema offers four.
  static constexpr int kMaxBands = 6;

  struct Stage {
    std::vector<PartitionedConvolver> lp; // one per channel
    DelayLine align;                      // the remainder, delayed by `latency`
    int latency = 0;
  };

  struct Band {
    GainComputer comp;
    double makeupDb = 0.0;
    double alphaAttack = 0.0, alphaRelease = 0.0;
    DelayLine align; // up to the common crossover total, at the base rate
    DelayLine look;  // the shared lookahead, at the oversampled rate
    std::vector<DecoupledPeakDetector> det;
    std::vector<RunningMax> rmax;
    std::vector<double> chLevel;
    std::vector<std::vector<Sample>> sig, dly;
    std::vector<Sample*> sigPtr, dlyPtr, work;
    std::vector<const Sample*> sigConst, workConst;
  };

  // The oversampler carries every (band, channel) pair as one of its channels.
  int osChannel(size_t band, int c) const {
    return static_cast<int>(band) * spec_.numChannels + c;
  }

  void rebuild() {
    const int ch = spec_.numChannels;
    if (ch <= 0) return;
    const double sr = spec_.sampleRate;
    const size_t chs = static_cast<size_t>(ch);
    const size_t block = static_cast<size_t>(std::max(1, spec_.maxBlockFrames));

    stages_.clear();
    bands_.clear();
    cross_.clear();
    latency_ = 0;
    lookaheadSamples_ = 0;

    const std::vector<ParamSet>& items = params_.items();
    int numBands = static_cast<int>(items.size());
    if (numBands > kMaxBands) numBands = kMaxBands;
    active_ = numBands > 0;
    if (!active_) return;

    lookaheadSamples_ = static_cast<int>(std::lround(lookaheadMs_ * 0.001 * sr));
    if (lookaheadSamples_ < 0) lookaheadSamples_ = 0;

    // `crossover_hz` is the UPPER edge of each band. The last band's value is
    // not read: it always runs to Nyquist, and there is no sensible reading of
    // a top edge above the top edge.
    const double maxHz = std::min(kMaxCrossoverHz, 0.45 * sr);
    const double minHz = std::min(kMinCrossoverHz, maxHz);
    std::vector<double>& cross = cross_;
    cross.clear();
    for (int i = 0; i + 1 < numBands; ++i) {
      const ParamSet& b = items[static_cast<size_t>(i)];
      cross.push_back(std::clamp(b.num("crossover_hz", 1000.0), minHz, maxHz));
    }
    // SORTED, not pushed upward. Corners out of order will happen — the caller
    // is a language model — and the two repairs are not equivalent. Pushing
    // each corner above the previous one turns (6000, 200) into
    // (6000, 6300) and leaves the middle band a 300 Hz sliver that the
    // parameters written for it can do nothing with; sorting turns it into
    // (200, 6000), which is what was meant. Every band stays non-empty, which
    // is the property worth keeping.
    std::sort(cross.begin(), cross.end());
    for (size_t i = 1; i < cross.size(); ++i)
      cross[i] = std::min(std::max(cross[i], cross[i - 1] * kMinCrossoverRatio), maxHz);

    std::vector<int> cumulative(static_cast<size_t>(numBands), 0);
    int total = 0;
    for (size_t i = 0; i < cross.size(); ++i) {
      Stage st;
      // Transition band as wide as the corner itself — half an octave either
      // side. A crossover does not need to be surgical and a narrower
      // transition costs taps quadratically. It also cannot affect the unity
      // null: reconstruction is a subtraction identity, so a bad filter splits
      // the spectrum badly and still sums back to a pure delay.
      const double cutoffNorm = cross[i] / sr;
      const FirLowpass fir = designLowpass(cutoffNorm, cutoffNorm, stopbandDb_);
      st.lp.assign(chs, PartitionedConvolver{});
      for (auto& cv : st.lp) cv.prepare(fir.h);
      // One whole partition plus the kernel's own group delay, both integers.
      st.latency = st.lp[0].latencyFrames() + fir.groupDelay;
      st.align.prepare(ch, st.latency);
      total += st.latency;
      cumulative[i] = total;
      stages_.push_back(std::move(st));
    }
    cumulative[static_cast<size_t>(numBands) - 1] = total;

    // ONLY OVERSAMPLE IF SOMETHING IS ACTUALLY COMPRESSING.
    //
    // A band at 1:1 has a gain of exactly dbToLin(makeup) at every sample — a
    // constant. Multiplying by a constant is linear, and a linear operation
    // cannot alias, so there is nothing for an oversampler to prevent. Running
    // one anyway would trade the crossover's -319 dB unity null for the
    // resampler's -140 dB round trip and charge latency for the privilege.
    //
    // `effFactor_` is therefore 1 unless some band's ratio exceeds 1. An
    // Oversampler at factor 1 is a pair of std::copy calls with zero declared
    // latency, so the transparent case runs the SAME code path as every other
    // case and is still bit exact — no second branch to get wrong.
    bool compressing = false;
    for (int k = 0; k < numBands; ++k)
      if (std::max(1.0, items[static_cast<size_t>(k)].num("ratio_to_1", 2.0)) > 1.0)
        compressing = true;
    effFactor_ = compressing ? factor_ : 1;

    // Wideband, for the same reason as `transient_designer`: what is being
    // oversampled is a rectifier, and its folded products land wherever the
    // arithmetic puts them rather than politely above 22 kHz.
    os_.prepare(effFactor_, numBands * ch, static_cast<int>(block),
                OsProfile::Wideband);

    // Every term is an exact integer at the base rate: the crossover stages by
    // construction, the oversampler's round trip by construction, and the
    // lookahead because `lookahead * factor` oversampled samples is `lookahead`
    // base samples.
    latency_ = total + os_.latencyFrames() + lookaheadSamples_;

    // Ballistics are per sample, so they follow the oversampled rate. Derived
    // from the base rate they would be `factor` times too fast, which would
    // make the oversampling change the node's sound rather than only its
    // aliasing.
    const double osRate = sr * static_cast<double>(effFactor_);

    bands_.resize(static_cast<size_t>(numBands));
    for (int k = 0; k < numBands; ++k) {
      const ParamSet& b = items[static_cast<size_t>(k)];
      Band& bd = bands_[static_cast<size_t>(k)];
      bd.comp.thresholdDb = clampDb(b.num("threshold_dbfs", -18.0));
      bd.comp.ratio = std::max(1.0, b.num("ratio_to_1", 2.0));
      bd.comp.kneeDb = std::max(0.0, b.num("knee_db", 6.0));
      bd.makeupDb = clampDb(b.num("makeup_gain_db", 0.0));
      const double attackMs = std::clamp(b.num("attack_ms", 10.0), 0.01, 10000.0);
      const double releaseMs = std::clamp(b.num("release_ms", 120.0), 1.0, 10000.0);
      bd.alphaAttack = timeConstantAlpha(attackMs * 0.001, osRate);
      bd.alphaRelease = timeConstantAlpha(releaseMs * 0.001, osRate);

      bd.align.prepare(ch, total - cumulative[static_cast<size_t>(k)]);
      bd.look.prepare(ch, lookaheadSamples_ * effFactor_);
      bd.det.assign(chs, DecoupledPeakDetector{});
      bd.rmax.assign(chs, RunningMax{});
      for (auto& r : bd.rmax) r.prepare(lookaheadSamples_ * effFactor_ + 1);
      bd.chLevel.assign(chs, 0.0);
      bd.sig.assign(chs, std::vector<Sample>(block, 0.0));
      bd.dly.assign(chs,
                    std::vector<Sample>(block * static_cast<size_t>(effFactor_), 0.0));
      bd.sigPtr.assign(chs, nullptr);
      bd.dlyPtr.assign(chs, nullptr);
      bd.work.assign(chs, nullptr);
      bd.sigConst.assign(chs, nullptr);
      bd.workConst.assign(chs, nullptr);
    }

    res_.assign(chs, std::vector<Sample>(block, 0.0));
    resPtr_.assign(chs, nullptr);
    resConst_.assign(chs, nullptr);
    bandOut_.assign(chs, std::vector<Sample>(block, 0.0));

    reset();
  }

  ParamSet params_;
  bool active_ = false;
  double lookaheadMs_ = 5.0, stereoLink_ = 1.0, stopbandDb_ = 80.0;
  int latency_ = 0, lookaheadSamples_ = 0;
  int factor_ = 4, effFactor_ = 1;

  std::vector<Stage> stages_;
  std::vector<Band> bands_;
  std::vector<double> cross_;
  Oversampler os_;
  std::vector<std::vector<Sample>> res_, bandOut_;
  std::vector<Sample*> resPtr_;
  std::vector<const Sample*> resConst_;
};

} // namespace anv
