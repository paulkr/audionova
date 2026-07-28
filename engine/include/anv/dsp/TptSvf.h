// Topology-preserving-transform state variable filter (Simper / Zavalishin).
//
// Chosen over a direct-form biquad for three reasons that matter here:
//   - Zero-delay feedback, so the response is analytically correct rather than
//     bilinear-warped at high frequencies.
//   - Numerically well-behaved at extreme Q and at low cutoff relative to the
//     sample rate, which a Direct Form I biquad is not.
//   - Per-sample by construction, with no per-block coefficient recalculation,
//     so it is block-size invariant for free.
//
// At the 96 kHz canonical rate, a 10 kHz bell sits at 0.10 x Nyquist, where
// frequency warping is inaudible — so these plain designs are correct without
// needing Orfanidis-style correction.
#pragma once

#include <cmath>

#include "anv/core/Types.h"

namespace anv {

enum class FilterShape {
  Bell,
  LowShelf,
  HighShelf,
  LowPass,
  HighPass,
  Notch,
  // Raw bandpass with k = 1/Q, deliberately independent of any gain
  // parameter. This is the building block for gain modulation that has to
  // happen per sample; see BandPass in the DynamicEQ notes.
  BandPass,
};

class TptSvf {
public:
  void setSampleRate(double sr) { sampleRate_ = sr; }

  void configure(FilterShape shape, double freqHz, double gainDb, double q) {
    // Clamp into the region where the design is valid. The Python-side
    // validator rejects out-of-range values long before this, but the engine
    // must never produce NaN even when handed nonsense.
    const double nyquist = sampleRate_ * 0.5;
    freqHz = std::fmin(std::fmax(freqHz, 1.0), nyquist * 0.995);
    q = std::fmin(std::fmax(q, 0.025), 40.0);

    const double A = std::pow(10.0, gainDb / 40.0);
    const double w = M_PI * freqHz / sampleRate_;
    double g = std::tan(w);
    double k = 1.0 / q;

    switch (shape) {
      case FilterShape::Bell:
        k = 1.0 / (q * A);
        m0_ = 1.0; m1_ = k * (A * A - 1.0); m2_ = 0.0;
        break;
      case FilterShape::LowShelf:
        g /= std::sqrt(A);
        m0_ = 1.0; m1_ = k * (A - 1.0); m2_ = A * A - 1.0;
        break;
      case FilterShape::HighShelf:
        g *= std::sqrt(A);
        m0_ = A * A; m1_ = k * (1.0 - A) * A; m2_ = 1.0 - A * A;
        break;
      case FilterShape::LowPass:
        m0_ = 0.0; m1_ = 0.0; m2_ = 1.0;
        break;
      case FilterShape::HighPass:
        m0_ = 1.0; m1_ = -k; m2_ = -1.0;
        break;
      case FilterShape::Notch:
        m0_ = 1.0; m1_ = -k; m2_ = 0.0;
        break;
      case FilterShape::BandPass:
        // Returns v1 unscaled, so its magnitude at resonance is Q. Callers
        // that want unity there multiply by k.
        m0_ = 0.0; m1_ = 1.0; m2_ = 0.0;
        break;
    }

    a1_ = 1.0 / (1.0 + g * (g + k));
    a2_ = g * a1_;
    a3_ = g * a2_;
  }

  void reset() { ic1_ = 0.0; ic2_ = 0.0; }

  inline Sample processSample(Sample v0) noexcept {
    const double v3 = v0 - ic2_;
    const double v1 = a1_ * ic1_ + a2_ * v3;
    const double v2 = ic2_ + a2_ * ic1_ + a3_ * v3;
    ic1_ = 2.0 * v1 - ic1_;
    ic2_ = 2.0 * v2 - ic2_;
    return m0_ * v0 + m1_ * v1 + m2_ * v2;
  }

private:
  double sampleRate_ = kCanonicalSampleRate;
  double a1_ = 0.0, a2_ = 0.0, a3_ = 0.0;
  double m0_ = 1.0, m1_ = 0.0, m2_ = 0.0;
  double ic1_ = 0.0, ic2_ = 0.0;
};

// First-order TPT filter, used for shelving whose gain must be modulated per
// sample. A first-order shelf decomposes exactly as
//
//     H(s) = 1 + (P - 1) * LP1(s)       low shelf, gain P at DC
//     H(s) = 1 + (P - 1) * HP1(s)       high shelf, gain P at Nyquist
//
// with LP1/HP1 independent of P — which is what makes the gain a pure output
// mix coefficient rather than something baked into the filter state.
class TptOnePole {
public:
  void setSampleRate(double sr) { sampleRate_ = sr; }

  void configure(double freqHz) {
    const double nyquist = sampleRate_ * 0.5;
    freqHz = std::fmin(std::fmax(freqHz, 1.0), nyquist * 0.995);
    const double g = std::tan(M_PI * freqHz / sampleRate_);
    gain_ = g / (1.0 + g);
  }

  void reset() { z_ = 0.0; }

  // Returns the lowpass output; the highpass is x - lowpass.
  inline Sample processLowpass(Sample x) noexcept {
    const double v = (x - z_) * gain_;
    const double y = v + z_;
    z_ = y + v;
    return y;
  }

private:
  double sampleRate_ = kCanonicalSampleRate;
  double gain_ = 0.0;
  double z_ = 0.0;
};

} // namespace anv
