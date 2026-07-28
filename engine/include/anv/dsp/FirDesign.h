// Kaiser-window halfband FIR design.
//
// Halfbands are the building block of the oversampling cascade. Their defining
// property is that every other tap is exactly zero (except the centre tap,
// which is 0.5), which halves the multiply count for free and — more
// importantly here — makes the group delay exactly (N-1)/2 samples with no
// fractional component.
//
// We are offline, so we design these far harder than any real-time plugin
// would dare: 120 dB stopband is the default, where a plugin would ship 80.
#pragma once

#include <cmath>
#include <vector>

namespace anv {

inline double besselI0(double x) {
  double sum = 1.0;
  double term = 1.0;
  const double half = x * 0.5;
  for (int k = 1; k < 80; ++k) {
    const double r = half / static_cast<double>(k);
    term *= r * r;
    sum += term;
    if (term < sum * 1e-18) break;
  }
  return sum;
}

inline double kaiserBeta(double stopbandDb) {
  if (stopbandDb > 50.0) return 0.1102 * (stopbandDb - 8.7);
  if (stopbandDb >= 21.0)
    return 0.5842 * std::pow(stopbandDb - 21.0, 0.4) + 0.07886 * (stopbandDb - 21.0);
  return 0.0;
}

inline double sincPi(double x) {
  if (std::abs(x) < 1e-12) return 1.0;
  const double a = M_PI * x;
  return std::sin(a) / a;
}

// A symmetric, odd-length, linear-phase halfband low-pass with cutoff at
// exactly fs/4.
//
// `transitionWidth` is the half-width of the transition band in normalized
// frequency (cycles/sample), measured either side of 0.25. Wider means cheaper.
// At the 96 kHz canonical rate we only need to protect audio bandwidth, so the
// first stage can afford a very wide transition band and become extremely
// cheap — a direct payoff of choosing 96 kHz as the internal rate.
//
// `groupDelayMultiple` forces the group delay M to be a multiple of the given
// value. The oversampling cascade needs this so its total round-trip latency
// lands on an exact integer at the base rate; see Oversampler.h.
struct HalfbandDesign {
  std::vector<double> h;
  int groupDelay = 0; // M, in samples at this filter's own rate

  int taps() const { return static_cast<int>(h.size()); }
};

inline HalfbandDesign designHalfband(double transitionWidth, double stopbandDb,
                                     int groupDelayMultiple = 1) {
  // Kaiser order estimate. dw is the transition width in rad/sample; the band
  // runs from 0.25-tw to 0.25+tw, so its total width is 2*tw cycles/sample.
  const double dw = 2.0 * M_PI * (2.0 * transitionWidth);
  int estimated = static_cast<int>(
      std::ceil((stopbandDb - 8.0) / (2.285 * dw))) + 1;
  if (estimated < 3) estimated = 3;

  // N must be odd so the filter is Type I and the delay is an integer.
  int M = (estimated + 1) / 2;
  if (groupDelayMultiple > 1) {
    const int r = M % groupDelayMultiple;
    if (r != 0) M += groupDelayMultiple - r;
  }
  if (M < 1) M = groupDelayMultiple > 1 ? groupDelayMultiple : 1;

  const int N = 2 * M + 1;
  const double beta = kaiserBeta(stopbandDb);
  const double denom = besselI0(beta);

  HalfbandDesign d;
  d.h.assign(static_cast<size_t>(N), 0.0);
  d.groupDelay = M;

  for (int n = 0; n < N; ++n) {
    const double k = static_cast<double>(n - M);
    // Ideal halfband impulse response: 0.5 * sinc(k/2). This is exactly zero
    // for every even non-zero k, which is the halfband property.
    const double ideal = 0.5 * sincPi(0.5 * k);
    const double ratio = (2.0 * static_cast<double>(n) / static_cast<double>(N - 1)) - 1.0;
    const double w = besselI0(beta * std::sqrt(std::max(0.0, 1.0 - ratio * ratio))) / denom;
    d.h[static_cast<size_t>(n)] = ideal * w;
  }

  // Force the even-offset taps to be exactly zero. The window is symmetric so
  // they are already zero to within rounding, but making it exact means the
  // sparse tap list below is exactly right and the delay branch is exact.
  for (int n = 0; n < N; ++n) {
    const int k = n - M;
    if (k != 0 && (k % 2) == 0) d.h[static_cast<size_t>(n)] = 0.0;
  }

  // Normalize for unity DC gain.
  double sum = 0.0;
  for (double v : d.h) sum += v;
  if (sum != 0.0)
    for (double& v : d.h) v /= sum;

  return d;
}

// General linear-phase windowed-sinc low-pass.
//
// Used to build COMPLEMENTARY crossovers. Given a linear-phase low-pass with
// group delay M, the matching high-pass is (delta delayed by M) minus the
// low-pass output — so low + high reconstructs the input as a pure delay,
// EXACTLY, with no magnitude or phase error at any frequency.
//
// That is the property an IIR crossover cannot give. A Linkwitz-Riley pair
// sums to an allpass: magnitude-flat, but phase-shifted, so it cannot null
// against its own input and any band-split processing leaves a signature even
// when every band is set to unity.
//
// Frequencies are normalised to cycles/sample (Nyquist = 0.5).
struct FirLowpass {
  std::vector<double> h;
  int groupDelay = 0;
  int taps() const { return static_cast<int>(h.size()); }
};

inline FirLowpass designLowpass(double cutoffNorm, double transitionNorm,
                                double stopbandDb) {
  cutoffNorm = std::min(std::max(cutoffNorm, 1e-5), 0.499);
  transitionNorm = std::max(transitionNorm, 1e-5);

  const double dw = 2.0 * M_PI * transitionNorm;
  int estimated =
      static_cast<int>(std::ceil((stopbandDb - 8.0) / (2.285 * dw))) + 1;
  if (estimated < 3) estimated = 3;

  // Odd length, so the filter is Type I and its group delay is an integer.
  const int M = (estimated + 1) / 2;
  const int N = 2 * M + 1;

  const double beta = kaiserBeta(stopbandDb);
  const double denom = besselI0(beta);

  FirLowpass f;
  f.h.assign(static_cast<size_t>(N), 0.0);
  f.groupDelay = M;

  for (int n = 0; n < N; ++n) {
    const double k = static_cast<double>(n - M);
    const double ideal = 2.0 * cutoffNorm * sincPi(2.0 * cutoffNorm * k);
    const double ratio = (2.0 * static_cast<double>(n) / (N - 1)) - 1.0;
    const double w =
        besselI0(beta * std::sqrt(std::max(0.0, 1.0 - ratio * ratio))) / denom;
    f.h[static_cast<size_t>(n)] = ideal * w;
  }

  double sum = 0.0;
  for (double v : f.h) sum += v;
  if (sum != 0.0)
    for (double& v : f.h) v /= sum;

  return f;
}

} // namespace anv
