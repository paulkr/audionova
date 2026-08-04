// True-peak (inter-sample peak) estimation, per ITU-R BS.1770-5 Annex 2.
//
// A signal whose SAMPLE peak sits at -1 dBFS can easily reconstruct to well
// above 0 dBFS between samples; a DAC or a lossy encoder then clips something
// that looked perfectly safe on a meter. That is what "true peak" measures and
// what a delivery ceiling has to be specified against.
//
// THE ERROR HAS EXACTLY TWO TERMS, AND THEY WANT OPPOSITE THINGS SPENT ON
// THEM. An earlier version of this comment claimed "oversampling FACTOR does
// not affect accuracy at all; 4x, 8x and 16x produce bit-identical errors for
// the same filter length", and set the default factor to 4 on that basis. THAT
// CLAIM WAS WRONG, and it was wrong in the permissive direction. It was
// measured only on long stationary signals — held tones and steady broadband
// noise — where hundreds of cycles pass under the estimator and some cycle's
// crest happens to land near a grid point whatever the grid is. An isolated
// transient, which is the thing a limiter exists to catch, gets exactly one
// crest and one grid alignment.
//
//   TERM A — PEAK-SEARCH GRID. The estimator evaluates the reconstruction at
//   only `factor` points per input sample, so it returns the grid maximum and
//   never the continuous one. It is therefore ALWAYS an under-estimate, which
//   is always the unsafe direction. For a signal band-limited to base Nyquist
//   the shortfall is at most cos(pi/(2*factor)) — attained by a tone at
//   Nyquist — which is a theorem about band-limited signals, not a corpus
//   statistic. See gridBoundDb(). Measured against an exact DFT-domain
//   reconstruction, on a lone band-limited impulse placed 1/8 of a sample off
//   the grid:
//
//     content to    4x        8x       16x       32x     bound at Nyquist
//     20 kHz     -0.039    -0.000    -0.000    -0.000     4x  -0.688 dB
//     24 kHz     -0.056    -0.000    -0.000    -0.000     8x  -0.168 dB
//     40 kHz     -0.156    -0.000    -0.000    -0.000    16x  -0.042 dB
//     48 kHz     -0.224    -0.017    -0.006    -0.001    32x  -0.010 dB
//
//   This term is why the factor floor is 16 and not 4. It matters MOST at the
//   delivery rate, which is where this node actually runs: at 44.1 kHz, 20 kHz
//   of programme sits at 0.907 of Nyquist, and 4x oversampling can under-read
//   it by 0.56 dB.
//
//   TERM B — THE TRANSITION BAND. `totalTransition` straddles base Nyquist, so
//   the passband ends at (0.5 - totalTransition/2) of the base rate and content
//   above that is attenuated. Unlike term A this one is NOT signed: it
//   perturbs the peak by whatever the discarded content happened to contribute
//   there. Measured against the exact reconstruction at 96 kHz, 16x:
//
//     material                      t=0.05   t=0.02   t=0.01   t=0.005
//     anything band-limited to      -0.001   +0.000   -0.000   -0.000
//       24 kHz (incl. hard-clipped)
//     hard-clipped noise to Nyquist -0.466   -0.219   -0.078   -0.055
//     band-limited impulse at Nyq   -0.117   -0.046   -0.022   -0.010
//
//   Driving that to zero costs 39x the runtime and still does not reach zero.
//   It is the same fundamental limit backend/measure/loudness.py records for
//   the Python meter: no finite filter reconstructs content arbitrarily close
//   to Nyquist. What makes it tolerable is the top line — for ANY signal
//   band-limited to half the base Nyquist the term vanishes, and the engine's
//   resampler has already band-limited the signal to 20 kHz before the limiter
//   sees it at the delivery rate.
//
// So: factor is bought (term A is a guaranteed under-report and it is what
// real material exercises), transition width is not (term B only bites on
// content the pipeline does not contain). The residual of term A is then
// handed to the caller as gridBoundDb() so the limiter can aim under its
// ceiling by exactly the amount its own instrument may be blind to.
//
// This is a dedicated polyphase interpolator rather than a reuse of the
// engine's halfband cascade, for one specific reason: a symmetric FIR of
// length 2*N*P+1 has a group delay of exactly N*P oversampled samples, which
// is exactly P samples at the base rate. An integer, by construction, with no
// rounding anywhere. The halfband cascade's up-only latency is not generally
// an integer number of base samples, and reconciling that against a lookahead
// window is exactly the kind of off-by-a-fraction that produces a limiter
// which "usually" holds its ceiling.
//
// The spec's -12.04 dB pre-attenuation is deliberately omitted: it exists to
// stop fixed-point accumulators overflowing, and this path is float64.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "anv/core/Types.h"
#include "anv/dsp/FirDesign.h"

namespace anv {

class TruePeakUpsampler {
public:
  // `stopbandDb` trades filter length against estimation accuracy. 80 dB is
  // comfortably beyond what BS.1770's own reference filter achieves.
  // `totalTransition` is the transition width as a fraction of the base rate,
  // straddling base Nyquist. It sets the filter length and therefore the
  // worst-case accuracy; see the measured table above.
  void design(int factor, double stopbandDb = 80.0,
              double totalTransition = 0.05) {
    factor_ = std::max(2, factor);
    transition_ = totalTransition;

    // The transition has to straddle base Nyquist because the first image
    // begins exactly there — a zero-width transition is not realisable.
    const double dw = 2.0 * M_PI * (transition_ / factor_);
    const int estimated =
        static_cast<int>(std::ceil((stopbandDb - 8.0) / (2.285 * dw))) + 1;
    P_ = std::max(1, static_cast<int>(std::ceil(
                         (estimated - 1.0) / (2.0 * factor_))));

    const int L = 2 * factor_ * P_ + 1;
    const int center = factor_ * P_;
    const double beta = kaiserBeta(stopbandDb);
    const double denom = besselI0(beta);
    const double fc = 0.5 / factor_; // cutoff at base Nyquist

    std::vector<double> h(static_cast<size_t>(L));
    for (int i = 0; i < L; ++i) {
      const double k = static_cast<double>(i - center);
      const double ideal = 2.0 * fc * sincPi(2.0 * fc * k);
      const double ratio = (2.0 * static_cast<double>(i) / (L - 1)) - 1.0;
      const double w =
          besselI0(beta * std::sqrt(std::max(0.0, 1.0 - ratio * ratio))) / denom;
      h[static_cast<size_t>(i)] = ideal * w;
    }
    phaseLen_ = (L + factor_ - 1) / factor_;
    phases_.assign(static_cast<size_t>(factor_),
                   std::vector<double>(static_cast<size_t>(phaseLen_), 0.0));
    for (int i = 0; i < L; ++i)
      phases_[static_cast<size_t>(i % factor_)][static_cast<size_t>(i / factor_)] =
          h[static_cast<size_t>(i)];

    // NORMALISE EACH POLYPHASE BRANCH SEPARATELY, not the filter as a whole.
    //
    // Normalising the whole filter gives it unity DC gain overall, but the
    // individual branches do not then each sum to 1/N — they differ by a
    // fraction of a percent. Since this estimator takes the MAXIMUM across
    // branches, that spread biases every measurement upward by the largest
    // branch's error: a constant +0.078 dB over-report on a pure sine,
    // independent of amplitude. Per-branch normalisation makes every phase
    // exactly unity at DC and removes the bias entirely.
    for (int p = 0; p < factor_; ++p) {
      double s = 0.0;
      for (double v : phases_[static_cast<size_t>(p)]) s += v;
      if (s != 0.0) {
        const double scale = 1.0 / (s * static_cast<double>(factor_));
        for (double& v : phases_[static_cast<size_t>(p)]) v *= scale;
      }
    }
  }

  int factor() const { return factor_; }

  // Exact, integer, base-rate latency.
  int latencyFrames() const { return P_; }

  // THE MOST THIS ESTIMATOR CAN BE BLIND TO, IN dB, FROM GRID SEARCH ALONE.
  //
  // processMaxAbs() reports the largest of `factor` samples of the
  // reconstruction per input sample, never the continuous maximum between
  // them, so it can only ever under-report. For y band-limited to base
  // Nyquist and sampled at factor*fs the shortfall is bounded by
  //
  //     max_t |y(t)|  <=  max_k |y[k]| / cos(pi / (2*factor))
  //
  // with equality for a tone at Nyquist, whose grid samples straddle its crest
  // by half a grid step at worst. This is a property of band-limited signals
  // rather than a measurement of this filter, so it holds at every sample
  // rate, for every signal, with no corpus behind it: 0.688 dB at 4x,
  // 0.168 at 8x, 0.042 at 16x, 0.010 at 32x.
  //
  // It is a bound on the estimator's own reconstruction, NOT on the estimator
  // against the truth — the transition band (term B in the header) is a
  // separate, unbounded, and unsigned error that no theorem covers.
  //
  // Returned as a POSITIVE number of dB, because every caller wants to
  // subtract it from a ceiling.
  double gridBoundDb() const {
    return -20.0 * std::log10(std::cos(M_PI / (2.0 * factor_)));
  }

  void prepare(int channels) {
    hist_.assign(static_cast<size_t>(channels),
                 std::vector<Sample>(static_cast<size_t>(phaseLen_) * 2, 0.0));
    pos_.assign(static_cast<size_t>(channels), 0);
  }

  void reset() {
    for (auto& h : hist_) std::fill(h.begin(), h.end(), 0.0);
    std::fill(pos_.begin(), pos_.end(), 0);
  }

  // Push one base-rate sample; return the largest |y| across the `factor`
  // interpolated outputs it produces. The value returned corresponds to the
  // base-rate sample latencyFrames() ago.
  inline double processMaxAbs(Sample x, int ch) noexcept {
    auto& hist = hist_[static_cast<size_t>(ch)];
    int& pos = pos_[static_cast<size_t>(ch)];

    pos = (pos == 0) ? phaseLen_ - 1 : pos - 1;
    hist[static_cast<size_t>(pos)] = x;
    hist[static_cast<size_t>(pos + phaseLen_)] = x;
    const Sample* w = hist.data() + pos;

    double best = 0.0;
    for (int p = 0; p < factor_; ++p) {
      const auto& e = phases_[static_cast<size_t>(p)];
      double acc = 0.0;
      for (int k = 0; k < phaseLen_; ++k)
        acc += e[static_cast<size_t>(k)] * w[k];
      best = std::max(best, std::abs(acc));
    }
    return best * static_cast<double>(factor_);
  }

private:
  double transition_ = 0.05;
  int factor_ = 8;
  int P_ = 1;
  int phaseLen_ = 1;
  std::vector<std::vector<double>> phases_;
  std::vector<std::vector<Sample>> hist_;
  std::vector<int> pos_;
};

// Hann-weighted moving average. Non-negative weights summing to one, which is
// the property the limiter's zero-overshoot proof depends on.
class HannSmoother {
public:
  void prepare(int length) {
    len_ = std::max(2, length);
    h_.assign(static_cast<size_t>(len_), 0.0);
    double sum = 0.0;
    for (int k = 0; k < len_; ++k) {
      h_[static_cast<size_t>(k)] =
          0.5 * (1.0 - std::cos(2.0 * M_PI * static_cast<double>(k) / (len_ - 1)));
      sum += h_[static_cast<size_t>(k)];
    }
    for (double& v : h_) v /= sum;
    hist_.assign(static_cast<size_t>(len_) * 2, 0.0);
    pos_ = 0;
  }

  void reset(double fill = 1.0) {
    std::fill(hist_.begin(), hist_.end(), fill);
    pos_ = 0;
  }

  int length() const { return len_; }

  inline double process(double x) noexcept {
    pos_ = (pos_ == 0) ? len_ - 1 : pos_ - 1;
    hist_[static_cast<size_t>(pos_)] = x;
    hist_[static_cast<size_t>(pos_ + len_)] = x;
    const double* w = hist_.data() + pos_;
    double acc = 0.0;
    for (int k = 0; k < len_; ++k) acc += h_[static_cast<size_t>(k)] * w[k];
    return acc;
  }

private:
  int len_ = 2;
  int pos_ = 0;
  std::vector<double> h_;
  std::vector<double> hist_;
};

} // namespace anv
