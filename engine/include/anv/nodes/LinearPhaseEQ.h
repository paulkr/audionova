// Linear-phase FIR EQ — constant group delay, pre-ringing, exact integer latency.
//
// The other half of the offline EQ story. Where `zero_phase_eq` cancels phase
// entirely by filtering twice in opposite directions, this realises the target
// magnitude directly as a SYMMETRIC FIR. A symmetric FIR of odd length N has
// phase exactly -w*(N-1)/2 at every frequency: a pure delay, identical for
// every frequency, so nothing in the signal is time-smeared relative to
// anything else. Transients keep their shape; only the whole signal moves.
//
// The price is pre-ringing. A symmetric kernel has as much energy before its
// centre as after, so a filter that cuts hard at 60 Hz puts a 60 Hz wobble
// AHEAD of the kick that triggered it. Minimum phase cannot do that and is
// usually the right default; this node is for the cases where phase coherence
// between bands matters more than transient purity — mastering moves, mid/side
// work, anything where two paths are summed afterwards.
//
// LATENCY IS EXACTLY (taps - 1) / 2, AND THAT IS WHY THE TAP COUNT IS ODD.
//
// An even-length symmetric FIR has its centre of symmetry BETWEEN two samples,
// so its group delay is a half-integer. There is no fractional-delay
// compensation anywhere in this engine — the property that removes the single
// largest class of PDC bugs — so a half-integer latency is not something the
// chain can express. `taps` is forced odd rather than rejected, because a
// planner asking for 2048 taps means "about two thousand", not "exactly 2048,
// and fail if that is impossible".
//
// DIRECT CONVOLUTION, DELIBERATELY, and it is the expensive choice.
//
// `StereoWidth` runs its FIR through the partitioned convolver because the
// filter is thousands of taps and direct convolution would dominate the
// render. That is not available here: the convolver's latency is one partition
// PLUS the kernel's own group delay, so a node using it could only declare
// (taps-1)/2 + partition. Since the declared latency is the contract this node
// exists to keep exact, the convolution is direct and the cost is real —
// roughly `taps` multiplies per sample per channel. Keep `taps` as low as the
// job allows; see open-questions.md for the way out.
//
// FREQUENCY RESOLUTION IS fs / taps, AND IT IS THE REAL LIMIT.
//
// 2047 taps at 96 kHz resolves about 47 Hz. A 3 kHz bell at Q 1 is 3 kHz wide
// and is reproduced essentially exactly; a 40 Hz bell at Q 8 is 5 Hz wide and
// is not reproduced at all — it comes back as a broad, shallow tilt. This is
// inherent to linear phase, not to this implementation: a filter whose
// response varies faster than fs/taps cannot be built from taps coefficients.
// A low, narrow move belongs in `parametric_eq`, whose IIR does it for four
// multiplies.
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <string>
#include <vector>

#include "anv/core/Node.h"
#include "anv/dsp/EqDesign.h"
#include "anv/dsp/Fft.h"
#include "anv/dsp/FirDesign.h"
#include "anv/dsp/TptSvf.h"

namespace anv {

class LinearPhaseEQ final : public Node {
public:
  const char* type() const override { return "linear_phase_eq"; }

  void prepare(const PrepareSpec& spec) override {
    spec_ = spec;
    build();
  }

  void reset() override {
    for (auto& v : history_) std::fill(v.begin(), v.end(), 0.0);
    std::fill(pos_.begin(), pos_.end(), 0);
  }

  void setParams(const ParamSet& p) override {
    params_ = p;
    // Read here rather than inside build() so the lookup is a plain `p.num`,
    // which is what the Python schema test greps for when it checks that every
    // parameter the planner can emit is a parameter the engine actually reads.
    requestedTaps_ = p.num("taps", 2047.0);
    if (spec_.numChannels > 0) build();
  }

  int latencyFrames() const override { return half_; }

  void process(ProcessContext& ctx) noexcept override {
    const int n = ctx.numFrames;
    for (int c = 0; c < spec_.numChannels; ++c) {
      const size_t cc = static_cast<size_t>(c);
      const Sample* src = ctx.in[c];
      Sample* dst = ctx.out[c];
      auto& history = history_[cc];
      int& pos = pos_[cc];

      for (int i = 0; i < n; ++i) {
        const Sample x = src[i];
        // History is stored twice back to back so the tap loop reads a
        // contiguous window with no modulo in the inner loop — the same trick
        // as the halfband stages in Oversampler.h.
        pos = (pos == 0) ? taps_ - 1 : pos - 1;
        history[static_cast<size_t>(pos)] = x;
        history[static_cast<size_t>(pos + taps_)] = x;

        const Sample* w = history.data() + pos;
        Sample acc = 0.0;
        // The kernel is exactly symmetric, so folding it would halve this
        // loop. It is written out in full on purpose: folded, the second half
        // of the kernel would never be read, and "the kernel is symmetric"
        // would stop being a property a test can falsify and become an
        // assumption the arithmetic quietly enforces.
        for (int j = 0; j < taps_; ++j)
          acc += kernel_[static_cast<size_t>(j)] * w[j];
        dst[i] = acc;
      }
    }
  }

  // Design-time access, for the tests that assert symmetry and phase.
  const std::vector<Sample>& kernel() const { return kernel_; }

private:
  // Bounds on the tap count. The lower bound keeps the Kaiser window from
  // degenerating; the upper bound is a cost and latency cap — 8191 taps is
  // 4095 samples of declared latency and 8191 multiplies per sample per
  // channel, which is already an unreasonable thing to ask for.
  static constexpr int kMinTaps = 15;
  static constexpr int kMaxTaps = 8191;

  // How long an impulse response the magnitude target is measured from, and
  // how much of its tail may be discarded. Same criterion as ZeroPhaseEQ.
  static constexpr int kMaxIrFrames = 24000;
  static constexpr double kTailTolerance = 1e-12;

  // The design window. Frequency sampling gives an ideal kernel of infinite
  // length; truncating it to `taps` without a window produces Gibbs ripple
  // across the whole response, and the ripple appears as audible pre- and
  // post-echo either side of every transient. A Kaiser designed for 70 dB
  // trades a slightly wider transition for ringing 70 dB down, which is the
  // right way round for a tone control: nobody EQs to a 5 Hz edge, and
  // everybody hears pre-echo.
  static constexpr double kWindowStopbandDb = 70.0;

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

    int requested = static_cast<int>(std::lround(requestedTaps_));
    requested = std::clamp(requested, kMinTaps, kMaxTaps);
    if ((requested % 2) == 0) --requested; // odd, so the delay is an integer
    taps_ = requested;
    half_ = (taps_ - 1) / 2;

    // The magnitude target, taken from the same cascade `parametric_eq` would
    // build. Measured through an impulse rather than evaluated analytically,
    // so the two nodes cannot drift apart: whatever the IIR does at a
    // frequency is what this node is asked to match there.
    const std::vector<double> h = cascadeImpulseResponse(
        designCascade(), kMaxIrFrames, kTailTolerance);

    // Grid four times finer than the kernel, and at least twice the measured
    // response, so the inverse transform is not the thing limiting accuracy.
    size_t nf = 1;
    const size_t needed =
        std::max<size_t>(2 * h.size(), 4 * static_cast<size_t>(taps_));
    while (nf < needed) nf <<= 1;

    std::vector<std::complex<double>> buf(nf, std::complex<double>(0.0, 0.0));
    for (size_t i = 0; i < h.size(); ++i) buf[i] = h[i];
    fftInPlace(buf);

    // THROW THE PHASE AWAY. This one line is the whole difference between this
    // node and `parametric_eq`: same magnitude, no phase at all.
    for (auto& v : buf) v = std::complex<double>(std::abs(v), 0.0);
    ifftInPlace(buf);

    const double beta = kaiserBeta(kWindowStopbandDb);
    const double denom = besselI0(beta);

    kernel_.assign(static_cast<size_t>(taps_), 0.0);
    for (int k = 0; k <= half_; ++k) {
      const double ratio = static_cast<double>(k) / static_cast<double>(half_);
      const double win =
          besselI0(beta * std::sqrt(std::max(0.0, 1.0 - ratio * ratio))) / denom;
      // The inverse transform of a real, even spectrum is real and even, so
      // g[k] and g[nf-k] are the same number twice. Averaging them makes the
      // stored kernel symmetric to the last bit rather than to within round-off
      // — the same move FirDesign.h makes when it forces the halfband's even
      // taps to exact zeros, and for the same reason: the property the latency
      // claim rests on should be exact, not nearly exact.
      const size_t mirror = (nf - static_cast<size_t>(k)) % nf;
      const double v =
          0.5 * (buf[static_cast<size_t>(k)].real() + buf[mirror].real()) * win;
      kernel_[static_cast<size_t>(half_ + k)] = v;
      kernel_[static_cast<size_t>(half_ - k)] = v;
    }

    history_.assign(static_cast<size_t>(ch),
                    std::vector<Sample>(static_cast<size_t>(2 * taps_), 0.0));
    pos_.assign(static_cast<size_t>(ch), 0);
    reset();
  }

  ParamSet params_;
  double requestedTaps_ = 2047.0;
  int taps_ = 0, half_ = 0;
  std::vector<Sample> kernel_;
  std::vector<std::vector<Sample>> history_;
  std::vector<int> pos_;
};

} // namespace anv
