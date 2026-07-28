// Cascaded halfband oversampling, with provably integer base-rate latency.
//
// WHY THE LATENCY IS INTEGER, which is the whole reason this design was
// chosen: each 2x stage uses the SAME symmetric Type-I halfband for both the
// up and the down direction, so its up+down pair contributes exactly 2*M
// samples at that stage's own rate. Stage i runs at 2^i times the base rate,
// so it contributes 2*M_i / 2^i = M_i / 2^(i-1) base samples. Forcing M_i to
// be a multiple of 2^(i-1) at design time (see FirDesign.h) makes every term
// an integer — so there is no fractional-delay compensation anywhere in the
// engine, which removes the single largest source of PDC bugs.
//
// The transition-band policy is a direct payoff of the 96 kHz canonical rate.
// Material that arrived at 48 kHz or below has nothing above 24 kHz, so the
// first stage only has to protect audio bandwidth and can use a very wide
// (cheap) transition band. Only the "wideband" profile, for genuinely 96 kHz
// native sources, pays for a narrow one.
#pragma once

#include <algorithm>
#include <cassert>
#include <vector>

#include "anv/core/Types.h"
#include "anv/dsp/FirDesign.h"

namespace anv {

enum class OsProfile {
  Audio,    // protect 0-22 kHz; transition band above that. Default.
  Wideband, // protect nearly to base Nyquist. For 96 kHz native sources.
};

// One 2x stage. Holds the coefficients plus per-channel history for both
// directions. A sparse tap list skips the halfband's zero taps.
class HalfbandStage {
public:
  void design(double transitionWidth, double stopbandDb, int groupDelayMultiple) {
    d_ = designHalfband(transitionWidth, stopbandDb, groupDelayMultiple);
    sparse_.clear();
    for (int i = 0; i < d_.taps(); ++i)
      if (d_.h[static_cast<size_t>(i)] != 0.0)
        sparse_.push_back({i, d_.h[static_cast<size_t>(i)]});
  }

  int groupDelay() const { return d_.groupDelay; }
  int taps() const { return d_.taps(); }

  void prepare(int channels) {
    const size_t n = static_cast<size_t>(d_.taps());
    upHist_.assign(static_cast<size_t>(channels), std::vector<Sample>(2 * n, 0.0));
    downHist_.assign(static_cast<size_t>(channels), std::vector<Sample>(2 * n, 0.0));
    upPos_.assign(static_cast<size_t>(channels), 0);
    downPos_.assign(static_cast<size_t>(channels), 0);
  }

  void reset() {
    for (auto& v : upHist_) std::fill(v.begin(), v.end(), 0.0);
    for (auto& v : downHist_) std::fill(v.begin(), v.end(), 0.0);
    std::fill(upPos_.begin(), upPos_.end(), 0);
    std::fill(downPos_.begin(), downPos_.end(), 0);
  }

  // in: nIn samples. out: 2*nIn samples. Gain-compensated so passband is unity.
  void upsample(const Sample* in, Sample* out, int nIn, int ch) noexcept {
    auto& hist = upHist_[static_cast<size_t>(ch)];
    int& pos = upPos_[static_cast<size_t>(ch)];
    const int n = d_.taps();
    for (int i = 0; i < nIn; ++i) {
      // Zero-stuffed sequence is x[0], 0, x[1], 0, ... Push each in turn and
      // evaluate the filter after each push.
      out[2 * i] = 2.0 * pushAndFilter(hist, pos, n, in[i]);
      out[2 * i + 1] = 2.0 * pushAndFilter(hist, pos, n, 0.0);
    }
  }

  // in: 2*nOut samples. out: nOut samples.
  void downsample(const Sample* in, Sample* out, int nOut, int ch) noexcept {
    auto& hist = downHist_[static_cast<size_t>(ch)];
    int& pos = downPos_[static_cast<size_t>(ch)];
    const int n = d_.taps();
    for (int i = 0; i < nOut; ++i) {
      // Decimation keeps the filtered value at the even high-rate index.
      out[i] = pushAndFilter(hist, pos, n, in[2 * i]);
      pushOnly(hist, pos, n, in[2 * i + 1]);
    }
  }

private:
  // History is stored twice back-to-back so the tap loop reads a contiguous
  // window without a modulo in the inner loop.
  inline void pushOnly(std::vector<Sample>& hist, int& pos, int n, Sample x) noexcept {
    pos = (pos == 0) ? n - 1 : pos - 1;
    hist[static_cast<size_t>(pos)] = x;
    hist[static_cast<size_t>(pos + n)] = x;
  }

  inline Sample pushAndFilter(std::vector<Sample>& hist, int& pos, int n,
                              Sample x) noexcept {
    pushOnly(hist, pos, n, x);
    const Sample* w = hist.data() + pos;
    Sample acc = 0.0;
    for (const auto& t : sparse_) acc += t.coeff * w[t.index];
    return acc;
  }

  struct Tap { int index; double coeff; };

  HalfbandDesign d_;
  std::vector<Tap> sparse_;
  std::vector<std::vector<Sample>> upHist_, downHist_;
  std::vector<int> upPos_, downPos_;
};

class Oversampler {
public:
  // `factor` must be a power of two: 1, 2, 4, 8, or 16.
  void prepare(int factor, int channels, int maxBlockFrames, OsProfile profile,
               double stopbandDb = 120.0) {
    factor_ = std::max(1, factor);
    channels_ = channels;
    maxBlock_ = maxBlockFrames;
    stages_.clear();

    int s = 0;
    for (int f = factor_; f > 1; f >>= 1) ++s;
    numStages_ = s;

    stages_.resize(static_cast<size_t>(numStages_));
    for (int i = 0; i < numStages_; ++i) {
      // Stage i (0-based) runs at 2^(i+1) x base. Its own Nyquist is
      // 2^i x baseNyquist, and it must reject everything that would fold back
      // into the protected band.
      //
      // For stage 0 in the Audio profile we only protect ~22 kHz out of a
      // 48 kHz base Nyquist, so the transition band is enormous and the filter
      // is very short. Later stages get progressively cheaper still, because
      // the protected band shrinks relative to their sample rate.
      // The protected edge must be normalized to THIS STAGE'S OWN rate
      // (2^(i+1) x base), not to the base rate. Normalizing to the base rate
      // makes every filter roughly twice as selective as required and around
      // six times longer than it needs to be.
      const double stageRate = kCanonicalSampleRate * std::pow(2.0, i + 1);
      const double protectedHz =
          (profile == OsProfile::Audio) ? 22000.0 : kCanonicalSampleRate * 0.46;
      double tw = 0.25 - (protectedHz / stageRate);
      tw = std::min(std::max(tw, 0.005), 0.2);

      // Stage i contributes M_i / 2^i base samples (0-based i), so M_i must be
      // a multiple of 2^i for the total to be an integer.
      const int multiple = 1 << i;
      stages_[static_cast<size_t>(i)].design(tw, stopbandDb, multiple);
      stages_[static_cast<size_t>(i)].prepare(channels);
    }

    // Scratch buffers, one per stage boundary, sized for the largest block.
    scratch_.assign(static_cast<size_t>(numStages_ + 1), {});
    for (int i = 0; i <= numStages_; ++i) {
      const int rate = 1 << i;
      scratch_[static_cast<size_t>(i)].assign(
          static_cast<size_t>(channels),
          std::vector<Sample>(static_cast<size_t>(maxBlockFrames * rate), 0.0));
    }

    latency_ = computeLatency();
  }

  void reset() {
    for (auto& s : stages_) s.reset();
    for (auto& per : scratch_)
      for (auto& ch : per) std::fill(ch.begin(), ch.end(), 0.0);
  }

  int factor() const { return factor_; }

  // Round-trip latency in BASE-rate samples. Always an exact integer.
  int latencyFrames() const { return latency_; }

  // Upsample one channel's block. Returns a pointer to `n * factor` samples.
  Sample* up(const Sample* in, int n, int ch) noexcept {
    if (numStages_ == 0) {
      Sample* dst = scratch_[0][static_cast<size_t>(ch)].data();
      std::copy(in, in + n, dst);
      return dst;
    }
    const Sample* src = in;
    int count = n;
    for (int i = 0; i < numStages_; ++i) {
      Sample* dst = scratch_[static_cast<size_t>(i + 1)][static_cast<size_t>(ch)].data();
      stages_[static_cast<size_t>(i)].upsample(src, dst, count, ch);
      count *= 2;
      src = dst;
    }
    return scratch_[static_cast<size_t>(numStages_)][static_cast<size_t>(ch)].data();
  }

  // Downsample `n * factor` samples held in the top scratch buffer back to n.
  void down(Sample* out, int n, int ch) noexcept {
    if (numStages_ == 0) {
      const Sample* src = scratch_[0][static_cast<size_t>(ch)].data();
      std::copy(src, src + n, out);
      return;
    }
    int count = n * factor_;
    for (int i = numStages_ - 1; i >= 0; --i) {
      const Sample* src = scratch_[static_cast<size_t>(i + 1)][static_cast<size_t>(ch)].data();
      count /= 2;
      Sample* dst = (i == 0) ? out
                             : scratch_[static_cast<size_t>(i)][static_cast<size_t>(ch)].data();
      stages_[static_cast<size_t>(i)].downsample(src, dst, count, ch);
    }
  }

  // The oversampled working buffer for a channel, valid between up() and down().
  Sample* workBuffer(int ch) {
    return scratch_[static_cast<size_t>(numStages_)][static_cast<size_t>(ch)].data();
  }

private:
  int computeLatency() const {
    // Stage i (0-based) contributes 2*M_i at rate 2^(i+1), i.e. M_i / 2^i
    // base-rate samples. The design step guarantees divisibility.
    int total = 0;
    for (int i = 0; i < numStages_; ++i) {
      const int m = stages_[static_cast<size_t>(i)].groupDelay();
      assert((m % (1 << i)) == 0 && "halfband group delay must divide evenly");
      total += m / (1 << i);
    }
    return total;
  }

  int factor_ = 1;
  int numStages_ = 0;
  int channels_ = 0;
  int maxBlock_ = 0;
  int latency_ = 0;
  std::vector<HalfbandStage> stages_;
  std::vector<std::vector<std::vector<Sample>>> scratch_;
};

} // namespace anv
