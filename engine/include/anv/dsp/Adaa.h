// Antiderivative antialiasing (Parker, Zavalishin & D'Angelo, DAFx-16).
//
// A naive waveshaper generates harmonics far above Nyquist which fold back as
// inharmonic aliasing — the single most audible flaw in cheap saturation, and
// what an engineer identifies instantly as "digital". Oversampling fights it
// with brute force; ADAA integrates the nonlinearity analytically and
// differences it, suppressing aliasing at its source.
//
// WHY SECOND ORDER, measured rather than assumed. With first-order ADAA the
// in-band aliasing of this engine plateaued around -88 dB and refused to
// improve past 8x oversampling. Driving the same node far below saturation —
// so the shaper is effectively linear and only the halfband filters remain —
// measured -108 dB at 4x and -115 dB at 16x. That gap proves the filters were
// never the limit: ADAA1's first-order residual was. More oversampling could
// not have fixed it; a better approximation order was required.
//
// SECOND ORDER ALSO FIXES THE LATENCY. ADAA1 evaluates the nonlinearity over
// [x[n-1], x[n]], so it is centred half a sample late — a fractional delay
// that has to be compensated with a 2-tap average, which costs a little high
// frequency. ADAA2 is centred on x[n-1], so its delay is EXACTLY one sample at
// the oversampled rate. No fractional compensation, no HF droop.
//
// The cost is that every shaper now needs a second antiderivative in closed
// form. That rules out tanh, whose second antiderivative is a dilogarithm, so
// the curves below are piecewise polynomials — which is no loss, since a cubic
// soft clip is a perfectly musical saturator and the polynomial form is what
// the DAFx-23 antialiased-waveshaper work recommends anyway.
#pragma once

#include <algorithm>
#include <cmath>

namespace anv {

// Cubic soft clip. C1-continuous, so the derivative never jumps and the
// harmonic series rolls off quickly — the smooth, "tube-ish" option.
//
//   f(u)  = u - u^3/3            |u| <= 1
//         = (2/3)*sign(u)        |u| >  1
//
// `bias` shifts the operating point to generate even-order harmonics (the
// asymmetry that reads as valve rather than transistor). The shaper subtracts
// the resulting offset so the node stays DC-free.
struct SoftClipShaper {
  void setBias(double b) {
    bias_ = b;
    offset_ = rawF(b);
  }

  inline double f(double x) const noexcept { return rawF(x + bias_) - offset_; }

  inline double F1(double x) const noexcept {
    return rawF1(x + bias_) - offset_ * x;
  }

  inline double F2(double x) const noexcept {
    return rawF2(x + bias_) - offset_ * 0.5 * x * x;
  }

private:
  static inline double rawF(double u) noexcept {
    if (u > 1.0) return 2.0 / 3.0;
    if (u < -1.0) return -2.0 / 3.0;
    return u - (u * u * u) / 3.0;
  }
  // Even function.
  static inline double rawF1(double u) noexcept {
    const double a = std::abs(u);
    if (a <= 1.0) {
      const double u2 = u * u;
      return 0.5 * u2 - (u2 * u2) / 12.0;
    }
    return (2.0 / 3.0) * a - 0.25;
  }
  // Odd function.
  static inline double rawF2(double u) noexcept {
    const double a = std::abs(u);
    if (a <= 1.0) {
      const double u3 = u * u * u;
      return u3 / 6.0 - (u3 * u * u) / 60.0;
    }
    const double v = (a * a) / 3.0 - a / 4.0 + 1.0 / 15.0;
    return u >= 0.0 ? v : -v;
  }

  double bias_ = 0.0;
  double offset_ = 0.0;
};

// Hard clipping. A discontinuous derivative makes its harmonics decay only as
// 1/n, so it is the worst case for aliasing and therefore the most demanding
// test of this whole path.
struct HardClipShaper {
  void setBias(double b) {
    bias_ = b;
    offset_ = rawF(b);
  }

  inline double f(double x) const noexcept { return rawF(x + bias_) - offset_; }
  inline double F1(double x) const noexcept {
    return rawF1(x + bias_) - offset_ * x;
  }
  inline double F2(double x) const noexcept {
    return rawF2(x + bias_) - offset_ * 0.5 * x * x;
  }

private:
  static inline double rawF(double u) noexcept { return std::clamp(u, -1.0, 1.0); }
  static inline double rawF1(double u) noexcept {
    const double a = std::abs(u);
    return (a <= 1.0) ? 0.5 * u * u : a - 0.5;
  }
  static inline double rawF2(double u) noexcept {
    const double a = std::abs(u);
    if (a <= 1.0) return (u * u * u) / 6.0;
    const double v = 0.5 * a * a - 0.5 * a + 1.0 / 6.0;
    return u >= 0.0 ? v : -v;
  }

  double bias_ = 0.0;
  double offset_ = 0.0;
};

// Second-order ADAA. Output is centred on x[n-1], so group delay is exactly
// one sample at whatever rate this runs at.
//
// The branches are not optional. Each difference quotient becomes
// ill-conditioned when its two arguments coincide — which happens constantly
// on real material, at every stationary point of the waveform — and without
// the fallbacks the node emits loud numerical garbage exactly where the signal
// is quietest.
template <typename Shaper>
class Adaa2 {
public:
  void reset() {
    x1_ = 0.0;
    x2_ = 0.0;
  }

  inline double process(const Shaper& s, double x0) noexcept {
    double y;
    const double outer = x0 - x2_;

    if (std::abs(outer) < kEps) {
      // x0 and x[n-2] coincide: the outer denominator collapses.
      const double xBar = 0.5 * (x0 + x2_);
      const double delta = xBar - x1_;
      if (std::abs(delta) < kEps) {
        // All three points coincide — the signal is locally flat.
        y = s.f(0.5 * (xBar + x1_));
      } else {
        y = (2.0 / delta) * (s.F1(xBar) + (s.F2(x1_) - s.F2(xBar)) / delta);
      }
    } else {
      y = (2.0 / outer) * (secondDiff(s, x0, x1_) - secondDiff(s, x1_, x2_));
    }

    x2_ = x1_;
    x1_ = x0;
    return y;
  }

private:
  // float64 lets this guard sit three orders of magnitude tighter than float32
  // would allow, so far fewer samples take the approximate branch.
  static constexpr double kEps = 1e-5;

  static inline double secondDiff(const Shaper& s, double a, double b) noexcept {
    const double d = a - b;
    if (std::abs(d) < kEps) return s.F1(0.5 * (a + b));
    return (s.F2(a) - s.F2(b)) / d;
  }

  double x1_ = 0.0;
  double x2_ = 0.0;
};

} // namespace anv
