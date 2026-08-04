// A first-order analog prototype, realised so its IN-BAND magnitude matches
// the prototype rather than so its corner does.
//
// WHY THIS EXISTS RATHER THAN TptOnePole.
//
// `TptOnePole` prewarps at the corner: `g = tan(pi * fc / fs)`. That is the
// textbook bilinear transform and it is exactly right when the corner sits
// inside the band, because the analog and digital responses then agree at the
// one frequency the filter is named after and, since `tan(x) ~ x` for small x,
// everywhere below it too.
//
// It is wrong when the corner sits ABOVE the band, which is the case for every
// analog corner worth modelling here: a tape head's gap-loss knee at 28 kHz, a
// summing amplifier's gain-bandwidth roll-off at 60 kHz. The bilinear map
// compresses the whole infinite analog axis into [0, Nyquist), so a corner near
// Nyquist drags the response above it down with it and the in-band droop comes
// out far SMALLER than the prototype's. Measured, at the 96 kHz internal rate,
// for a 40 kHz analog corner read at 20 kHz:
//
//     analog prototype            -0.969 dB
//     bilinear prewarped at fc    -0.180 dB     five times too little
//     this, prewarped at 20 kHz   -0.969 dB     exact by construction
//
// The whole point of the parameter is the shape it puts on the audio band, and
// the conventional realisation gets that part wrong by a factor of five.
//
// THE RULE, in one line:
//
//     prewarp at min(fc, fRef)
//
//   fc <= fRef   the corner is in the band; prewarp there, which is the
//                textbook case and unchanged.
//   fc >  fRef   the corner is above the band; prewarp at fRef instead and
//                scale, so the digital magnitude equals the analog magnitude
//                EXACTLY at fRef and closely below it.
//
// Derivation of the second case. The TPT one-pole's magnitude is
//
//     |H(f)| = 1 / sqrt(1 + (tan(pi f / fs) / g)^2)
//
// and the analog prototype's is 1 / sqrt(1 + (f / fc)^2). Setting them equal at
// f = fRef gives g = tan(pi fRef / fs) * (fc / fRef), which is what `configure`
// computes. The two expressions coincide at fc == fRef, so the rule above is
// continuous and there is no branch to get wrong at the boundary.
//
// WHAT IT DOES NOT FIX. The bilinear map still forces |H| -> 0 at Nyquist,
// which a one-pole prototype does not do. Above fRef the realisation therefore
// falls away faster than the analog filter, and for a 60 kHz corner at the
// 96 kHz internal rate that deviation reaches several dB between 30 kHz and
// Nyquist. That is stated rather than hidden; it lands entirely above the band
// the parameter exists to shape, and the alternative — pole mapping, which
// aliases the analog response instead of compressing it — is wrong in-band,
// which is the half that matters.
#pragma once

#include <algorithm>
#include <cmath>

#include "anv/core/Types.h"

namespace anv {

class AnalogOnePole {
public:
  // fcHz: the analog corner. fRefHz: the in-band frequency at which the
  // realisation must agree with the analog prototype exactly.
  void configure(double fcHz, double sampleRate, double fRefHz) {
    const double nyquist = sampleRate * 0.5;
    fcHz = std::max(fcHz, 1e-6);
    // The reference must sit inside the band, and strictly below Nyquist or
    // tan() runs away.
    fRefHz = std::clamp(fRefHz, 1.0, nyquist * 0.9);
    const double warpAt = std::min(fcHz, fRefHz);
    g_ = std::tan(M_PI * warpAt / sampleRate) * (fcHz / warpAt);
    G_ = g_ / (1.0 + g_);
  }

  void reset() { z_ = 0.0; }

  // Lowpass output. The highpass is the input minus this, exactly.
  inline Sample lowpass(Sample x) noexcept {
    const double v = (x - z_) * G_;
    const double y = v + z_;
    z_ = y + v;
    return y;
  }

  inline Sample highpass(Sample x) noexcept { return x - lowpass(x); }

  // For tests: the realised integrator gain.
  double g() const { return g_; }

private:
  double g_ = 0.0, G_ = 0.0, z_ = 0.0;
};

} // namespace anv
