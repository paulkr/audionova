// Shared dynamics-processing primitives.
//
// Compressor, gate/expander, dynamic EQ and the true-peak limiter all need the
// same three pieces, so they live here rather than being reimplemented (and
// subtly diverging) four times.
//
// The topology follows Giannoulis, Massberg & Reiss, "Digital Dynamic Range
// Compressor Design — A Tutorial and Analysis", JAES 60(6), 2012: log-domain,
// feed-forward, with a SMOOTH DECOUPLED peak detector. That specific detector
// matters. The naive alternatives each have a well-known audible failure:
// a branching detector produces a discontinuous derivative that ticks on fast
// release, and a coupled detector cannot express attack and release
// independently, so its release time bleeds into the attack.
//
// Everything here is driven per-sample by the signal, never per block, so
// block-size invariance is structural rather than something to test for.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "anv/core/Types.h"

namespace anv {

// Floor for level -> dB conversion. Without it, digital silence maps to -inf
// and every downstream expression becomes NaN.
inline constexpr double kLevelFloor = 1e-12; // -240 dBFS

inline double linToDb(double x) {
  return 20.0 * std::log10(std::max(x, kLevelFloor));
}
inline double dbToLin(double db) { return std::pow(10.0, db / 20.0); }

// Time constant to one-pole coefficient. alpha = exp(-1 / (tau * fs)), i.e.
// tau is the time to reach 1 - 1/e of a step.
inline double timeConstantAlpha(double tauSeconds, double sampleRate) {
  if (tauSeconds <= 0.0) return 0.0;
  return std::exp(-1.0 / (tauSeconds * sampleRate));
}

// Soft-knee static curve, expressed as gain REDUCTION in dB (always >= 0).
struct GainComputer {
  double thresholdDb = -18.0;
  double ratio = 3.0;
  double kneeDb = 6.0;

  inline double reductionDb(double inputDb) const noexcept {
    const double over = inputDb - thresholdDb;
    const double slope = 1.0 - 1.0 / std::max(ratio, 1.0);

    if (kneeDb > 0.0 && 2.0 * over > -kneeDb && 2.0 * over < kneeDb) {
      // Quadratic interpolation across the knee. Continuous in value AND in
      // first derivative at both knee edges, which is what stops a soft knee
      // from sounding like a hard one with extra steps.
      const double t = over + kneeDb * 0.5;
      return slope * (t * t) / (2.0 * kneeDb);
    }
    if (2.0 * over <= -kneeDb) return 0.0;
    return slope * over;
  }
};

// Downward-expansion static curve, expressed as gain REDUCTION in dB (>= 0).
//
// Mirror image of the compressor: the compressor acts on what is ABOVE the
// threshold, the expander on what is BELOW it. A gate is simply an expander
// with a high ratio and a bounded range, which is why there is one curve here
// rather than two.
//
// `rangeDb` caps the attenuation. Capping rather than going to silence is what
// separates a usable gate from one that sounds like an edit: a drum track
// gated to -60 dB is clean, one gated to -inf sounds like the room vanished.
struct ExpanderComputer {
  double thresholdDb = -40.0;
  double ratio = 4.0;
  double kneeDb = 3.0;
  double rangeDb = 60.0;

  inline double reductionDb(double inputDb) const noexcept {
    const double over = inputDb - thresholdDb;
    const double slope = std::max(ratio, 1.0) - 1.0;

    double reduction;
    if (kneeDb > 0.0 && 2.0 * over > -kneeDb && 2.0 * over < kneeDb) {
      const double t = over - kneeDb * 0.5;
      reduction = slope * (t * t) / (2.0 * kneeDb);
    } else if (2.0 * over >= kneeDb) {
      reduction = 0.0;
    } else {
      reduction = -over * slope;
    }
    return std::min(reduction, rangeDb);
  }
};

// Smooth decoupled peak detector, operating on the gain-reduction signal.
//
//   y1 = max(x, aR*y1 + (1-aR)*x)     release branch, decoupled
//   yL = aA*yL + (1-aA)*y1            attack smoothing
//
// The release coefficient is passed per sample so callers can implement
// program-dependent release without recomputing an exponential each sample.
// The detector tracks its input upward instantly and relaxes downward at the
// release rate, then smooths the result at the attack rate. That polarity is
// stated in terms of the SIGNAL FED IN, not in terms of gain — which is what
// lets the compressor feed it reduction (rising = louder = attack) and the
// gate feed it gain (rising = opening = attack) from the same code.
class DecoupledPeakDetector {
public:
  // The initial value matters for a gate: it must start closed, or the first
  // block passes at unity before the detector has settled.
  void reset(double initial = 0.0) {
    y1_ = initial;
    yL_ = initial;
  }

  inline double process(double x, double alphaAttack, double alphaRelease) noexcept {
    y1_ = std::max(x, alphaRelease * y1_ + (1.0 - alphaRelease) * x);
    yL_ = alphaAttack * yL_ + (1.0 - alphaAttack) * y1_;
    return yL_;
  }

  double value() const { return yL_; }

private:
  double y1_ = 0.0;
  double yL_ = 0.0;
};

// Plain one-pole smoother, for RMS windows and crest estimation.
class OnePole {
public:
  void setAlpha(double a) { alpha_ = a; }
  void reset(double v = 0.0) { z_ = v; }
  inline double process(double x) noexcept {
    z_ = alpha_ * z_ + (1.0 - alpha_) * x;
    return z_;
  }
  double value() const { return z_; }

private:
  double alpha_ = 0.0;
  double z_ = 0.0;
};

// Sliding-window maximum in amortised O(1), via a monotonic deque.
//
// This is the piece that makes lookahead worth having. Feeding the ballistics
// the MAXIMUM reduction over the coming window — rather than the instantaneous
// value — means the gain has already finished moving by the time the transient
// arrives, so a lookahead of at least the attack time gives mathematically
// zero overshoot instead of "usually no overshoot".
class RunningMax {
public:
  void prepare(int window) {
    window_ = std::max(1, window);
    cap_ = window_ + 1;
    val_.assign(static_cast<size_t>(cap_), 0.0);
    idx_.assign(static_cast<size_t>(cap_), 0);
    reset();
  }

  void reset() {
    head_ = 0;
    count_ = 0;
    n_ = 0;
  }

  int window() const { return window_; }

  inline double process(double x) noexcept {
    // Drop anything at the back that this sample dominates: it can never be
    // the window maximum again.
    while (count_ > 0) {
      const int back = (head_ + count_ - 1) % cap_;
      if (val_[static_cast<size_t>(back)] <= x) --count_;
      else break;
    }
    const int slot = (head_ + count_) % cap_;
    val_[static_cast<size_t>(slot)] = x;
    idx_[static_cast<size_t>(slot)] = n_;
    ++count_;

    // Drop anything at the front that has fallen out of the window.
    while (count_ > 0 && idx_[static_cast<size_t>(head_)] <= n_ - window_) {
      head_ = (head_ + 1) % cap_;
      --count_;
    }

    ++n_;
    return val_[static_cast<size_t>(head_)];
  }

private:
  int window_ = 1;
  int cap_ = 2;
  int head_ = 0;
  int count_ = 0;
  int64_t n_ = 0;
  std::vector<double> val_;
  std::vector<int64_t> idx_;
};

} // namespace anv
