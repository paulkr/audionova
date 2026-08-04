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

// Quintic soft saturation — the tape and console curve.
//
// WHY A SECOND CURVE RATHER THAN REUSING SoftClipShaper. The cubic above is a
// clipper with a rounded corner: its gain is 1 - u^2, so it stays near unity
// until quite late and then folds over in the last third. Magnetic saturation
// does the opposite — remanence departs from linearity gradually, from well
// below the point where it finally stops increasing. That difference is the
// entire tonal distinction between "clipped" and "tape", and it is a property
// of the derivative rather than of the ceiling.
//
// So the curve is specified BY ITS DERIVATIVE, which is the physically
// meaningful quantity (incremental gain against instantaneous level):
//
//     f'(u) = (1 - u^2)^2      |u| <= 1
//           = 0                |u| >  1
//
// Integrating, with f odd:
//
//     f(u)  = u - 2u^3/3 + u^5/5        |u| <= 1
//           = (8/15) * sign(u)          |u| >  1
//
// Two consequences fall straight out of that choice, and both are why it was
// made:
//
//   f'(0) = 1, so the curve is unity for small signals and the `drive`
//   parameter means exactly what it says.
//
//   f'(+/-1) = 0 AND f''(+/-1) = 0, so the curve is C2 rather than the cubic's
//   C1. A C1 shaper's harmonic series falls as 1/n^3; a C2 shaper's falls as
//   1/n^4. That is 6 dB per harmonic of head start on the aliasing requirement
//   before any oversampling is bought, and it costs nothing but two more terms
//   of a polynomial.
//
// Incremental gain, which is what a level diagram is drawn from:
//
//     |u|     0.25    0.50    0.71    1.00
//     f'      -1.1    -5.0    -12.0   -inf   dB
//
// `bias` is an operating-point offset, exactly as in the two shapers above: it
// moves the curve's zero, which makes the two half-cycles traverse different
// amounts of curvature and therefore generates even-order harmonics. The
// resulting DC offset is subtracted back out, so f(0) is exactly 0 at every
// bias and a silent input stays bit-exactly silent.
//
// The saturated output is 8/15, not 1. That is not a normalisation to fix:
// the shaper's small-signal gain is unity, which is the property the drive
// control is defined against, and a curve whose ceiling was scaled to 1 would
// have a small-signal gain of 15/8 and a drive control that lied by 5.5 dB.
struct QuinticSatShaper {
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
    if (u > 1.0) return 8.0 / 15.0;
    if (u < -1.0) return -8.0 / 15.0;
    const double u2 = u * u;
    return u * (1.0 - (2.0 / 3.0) * u2 + 0.2 * u2 * u2);
  }
  // Even function: the antiderivative of an odd one.
  static inline double rawF1(double u) noexcept {
    const double a = std::abs(u);
    if (a <= 1.0) {
      const double u2 = u * u;
      return u2 * (0.5 - u2 / 6.0 + (u2 * u2) / 30.0);
    }
    // 11/30 is F1(1); beyond the knee f is constant at 8/15, so F1 is a ray.
    return 11.0 / 30.0 + (8.0 / 15.0) * (a - 1.0);
  }
  // Odd function.
  static inline double rawF2(double u) noexcept {
    const double a = std::abs(u);
    if (a <= 1.0) {
      const double u3 = u * u * u;
      const double u2 = u * u;
      return u3 * (1.0 / 6.0 - u2 / 30.0 + (u2 * u2) / 210.0);
    }
    const double d = a - 1.0;
    // 29/210 is F2(1); past the knee F1 is a ray, so F2 is a parabola.
    const double v = 29.0 / 210.0 + (11.0 / 30.0) * d + (4.0 / 15.0) * d * d;
    return u >= 0.0 ? v : -v;
  }

  double bias_ = 0.0;
  double offset_ = 0.0;
};

// The same quintic curve MINUS UNITY GAIN: g(u) = f(u) - u.
//
// WHY A NODE WANTS THIS. `tape` and `console` are near-identity processors, and
// running a near-identity signal through an oversampler and an ADAA operator
// imposes the round trip's own errors on ALL of it. The largest of those is not
// the halfband ripple, it is ADAA2 itself:
//
//   ADAA2 applied to a LINEAR function is exactly the three-point moving
//   average (x0 + x1 + x2)/3.
//
//   Substituting f(x) = x, so F2(x) = x^3/6, into the operator gives
//   (F2(a)-F2(b))/(a-b) = (a^2 + ab + b^2)/6, and the outer difference
//   factorises as (x0 - x2)(x0 + x1 + x2)/6, cancelling the outer denominator
//   exactly. It is a rectangular three-tap FIR at the OVERSAMPLED rate, so its
//   magnitude is |1 + 2cos(2 pi f / (factor * fs))| / 3:
//
//       at 20 kHz, 96 kHz base:   4x  -0.313 dB    8x  -0.078    16x  -0.019
//
//   That is a real high-frequency droop on every ADAA node in the engine, and
//   at 4x it is LARGER than the entire frequency-dependent shift `console`
//   exists to produce.
//
// Sending only the DEPARTURE from unity through the oversampler makes those
// errors scale with the departure instead of with the signal: the linear path
// is a pure integer delay and is exact, and a 0.3 dB tilt on a residual sitting
// 60 dB down is a 0.0003 dB tilt on the output. Aliasing is unaffected, because
// everything that could fold is in the residual and it is still oversampled and
// still antialiased.
//
// WHY THIS IS WRITTEN OUT RATHER THAN COMPUTED AS f(x) - x. Because the
// subtraction cancels. For |u| << 1 the curve is u - 2u^3/3 + ..., so f(x) - x
// discards every significant bit of the answer and keeps the round-off:
// at u = 2e-4 the surviving relative error in the second antiderivative is
// about 1e-8, and ADAA2 divides that by differences of the same order. A
// console stage at 24 dB of headroom operates at exactly that amplitude, so the
// regime where the subtraction is worst is the regime the node lives in.
//
//     g(u)   = -2u^3/3 + u^5/5           |u| <= 1
//            = (8/15)*sign(u) - u        |u| >  1
//
// with G1 and G2 its first and second antiderivatives, and bias handled as
// g(x) = rawG(x + bias) - rawG(bias), which is the same construction as the
// full curve and is exact: the bias offset the full shaper subtracts, rawF(b),
// is b + rawG(b), and the leading b cancels against the linear term that has
// been removed.
struct QuinticSatResidualShaper {
  void setBias(double b) {
    bias_ = b;
    offset_ = rawG(b);
  }

  inline double f(double x) const noexcept { return rawG(x + bias_) - offset_; }

  inline double F1(double x) const noexcept {
    return rawG1(x + bias_) - offset_ * x;
  }

  inline double F2(double x) const noexcept {
    return rawG2(x + bias_) - offset_ * 0.5 * x * x;
  }

private:
  static inline double rawG(double u) noexcept {
    if (u > 1.0) return 8.0 / 15.0 - u;
    if (u < -1.0) return -8.0 / 15.0 - u;
    const double u2 = u * u;
    return u * u2 * (-2.0 / 3.0 + 0.2 * u2);
  }
  // Even.
  static inline double rawG1(double u) noexcept {
    const double a = std::abs(u);
    if (a <= 1.0) {
      const double u2 = u * u;
      return u2 * u2 * (-1.0 / 6.0 + u2 / 30.0);
    }
    // G1(1) = -1/6 + 1/30 = -2/15, then integrate (8/15 - t).
    const double d = a - 1.0;
    return -2.0 / 15.0 + (8.0 / 15.0) * d - (a * a - 1.0) * 0.5;
  }
  // Odd.
  static inline double rawG2(double u) noexcept {
    const double a = std::abs(u);
    if (a <= 1.0) {
      const double u2 = u * u;
      return u * u2 * u2 * (-1.0 / 30.0 + u2 / 210.0);
    }
    // G2(1) = -1/30 + 1/210 = -1/35, then integrate G1 above.
    const double d = a - 1.0;
    const double v = -1.0 / 35.0 + (11.0 / 30.0) * d + (4.0 / 15.0) * d * d -
                     (a * a * a - 1.0) / 6.0;
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
