// Rational sample-rate conversion by polyphase windowed-sinc interpolation.
//
// The plan named r8brain for this and r8brain would have been fine, but the
// same reasoning that applies to the rest of the DSP applies here: this is
// about two hundred lines, we already own a Kaiser designer, and owning it
// means the conversion is specified by us rather than inherited.
//
// EXACTLY TWICE PER RENDER: once on the way in to the 96 kHz canonical rate,
// once on the way out to the delivery rate. Never in the middle of a chain.
// Every intermediate conversion is another passband ripple and another
// transition band, and they accumulate.
//
// ZERO NET LATENCY, EXACTLY. A linear-phase prototype has group delay
// halfLen at the interpolated rate, and that delay is removed by evaluating
// the polyphase sum at (m*M + halfLen) rather than m*M. Because the shift is
// an integer number of interpolated-rate samples it is exact — no fractional
// delay, nothing to compensate downstream, no accumulated drift on a long
// file. Output sample m is the band-limited value of the input at time
// m*M/L, full stop.
//
// THE FILTER IS DESIGNED FROM THE SPEC, NOT THE OTHER WAY AROUND. Passband to
// 20 kHz, 150 dB down by the lower rate's Nyquist. At 44.1 <-> 96 kHz that
// works out to roughly sixty-eight thousand taps, which sounds absurd until
// you notice it is simply what a transparent conversion costs: the transition
// band is 2 kHz wide at an interpolated rate of 14.1 MHz. It is computed once
// and runs at about two hundred multiplies per output sample. A real-time
// plugin cannot afford this and ships 8 kHz of transition band instead. We are
// offline; we can, so we do. See ResamplerSpec::stopbandDb for why 150 and not
// the conventional 120.
//
// EDGES ARE ZERO, NOT EXTRAPOLATED. The signal is treated as silent outside
// the file, which means the first and last ~80 input samples fade rather than
// resolve. That is the honest interpretation of a file boundary and it is what
// every serious converter does; inventing signal past the edge would be a
// guess that shows up as a click when the render is butted against another.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include "anv/core/AudioBuffer.h"
#include "anv/core/Types.h"
#include "anv/dsp/FirDesign.h"

namespace anv {

struct ResamplerSpec {
  // Highest frequency, in Hz, that must survive unattenuated. Above the lower
  // of the two rates' Nyquist this is meaningless, so it is clamped.
  double passbandHz = 20000.0;
  // 150 dB, not the 120 a plugin would ship.
  //
  // Kaiser gives passband ripple roughly equal to stopband attenuation, so a
  // 120 dB design contributes about -120 dB of error to a tone sitting in the
  // middle of its passband. The engine's cross-platform equivalence threshold
  // is -140 dBFS, which would make the sample-rate conversion — a stage that
  // is supposed to be transparent — the loudest error source in the entire
  // render, and the only one large enough to break that contract.
  //
  // The cost of the extra 30 dB is (150-8)/(120-8) = 1.27x the taps, which at
  // 44.1 <-> 96 kHz is 68k instead of 54k and a few milliseconds per second of
  // audio. It buys the SRC's error being irrelevant rather than dominant.
  double stopbandDb = 150.0;
  // Hard ceiling on prototype length. A pathological rate pair (say 44100 to
  // 44101) reduces to L = 44100, M = 44101 and would otherwise ask for a
  // filter with billions of taps and a phase table to match. Hitting the cap
  // widens the transition band rather than failing, and `capped` records it.
  int maxTaps = 1 << 21;
};

class Resampler {
public:
  // Returns false only for a non-positive rate. Equal rates are legal and make
  // the resampler an exact pass-through.
  bool prepare(double inRate, double outRate, const ResamplerSpec& spec = {}) {
    if (!(inRate > 0.0) || !(outRate > 0.0)) return false;

    inRate_ = inRate;
    outRate_ = outRate;
    capped_ = false;

    const int64_t ri = static_cast<int64_t>(std::llround(inRate));
    const int64_t ro = static_cast<int64_t>(std::llround(outRate));
    const int64_t g = std::gcd(ri, ro);
    l_ = static_cast<int>(ro / g);   // upsample factor
    m_ = static_cast<int>(ri / g);   // downsample factor

    identity_ = (l_ == 1 && m_ == 1);
    if (identity_) return true;

    // Everything below is in cycles/sample AT THE INTERPOLATED RATE, which is
    // l_ times the input rate. The anti-imaging and anti-aliasing filters are
    // the same filter: it must stop everything above the LOWER of the two
    // Nyquists, so the cutoff is set by max(l_, m_).
    const int maxLM = std::max(l_, m_);
    const double nyquistHz = 0.5 * std::min(inRate, outRate);
    const double passHz = std::min(spec.passbandHz, 0.98 * nyquistHz);

    const double passNorm = (passHz / (inRate * l_));
    const double stopNorm = 0.5 / static_cast<double>(maxLM);
    double transition = stopNorm - passNorm;
    if (transition <= 0.0) transition = 0.05 * stopNorm;

    // Kaiser order estimate, then rounded up to an odd length so the filter is
    // Type I and its group delay is an exact integer.
    double estimated = (spec.stopbandDb - 8.0) / (2.285 * 2.0 * M_PI * transition);
    int halfLen = static_cast<int>(std::ceil(estimated * 0.5));
    if (halfLen < l_) halfLen = l_;
    if (2 * halfLen + 1 > spec.maxTaps) {
      halfLen = (spec.maxTaps - 1) / 2;
      capped_ = true;
    }
    halfLen_ = halfLen;

    const int n = 2 * halfLen + 1;
    const double cutoff = 0.5 * (passNorm + stopNorm);
    const double beta = kaiserBeta(spec.stopbandDb);
    const double denom = besselI0(beta);

    std::vector<double> h(static_cast<size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
      const double k = static_cast<double>(i - halfLen);
      const double ideal = 2.0 * cutoff * sincPi(2.0 * cutoff * k);
      const double ratio = (2.0 * static_cast<double>(i) / (n - 1)) - 1.0;
      const double w =
          besselI0(beta * std::sqrt(std::max(0.0, 1.0 - ratio * ratio))) / denom;
      h[static_cast<size_t>(i)] = ideal * w;
    }

    // Unity DC gain, then multiplied by l_. Each output sample touches only
    // every l_-th tap, so the per-output DC gain would otherwise be 1/l_ —
    // the classic interpolation gain loss, and the reason a naively built
    // upsampler comes out quiet by exactly the upsample factor.
    double sum = 0.0;
    for (double v : h) sum += v;
    if (sum != 0.0) {
      const double scale = static_cast<double>(l_) / sum;
      for (double& v : h) v *= scale;
    }

    // Split into polyphase branches. Branch p holds taps p, p+l_, p+2*l_...
    // in order, so the inner loop walks contiguous memory instead of striding
    // through the prototype by l_ and missing the cache on every tap.
    phases_.assign(static_cast<size_t>(l_), {});
    for (int p = 0; p < l_; ++p) {
      auto& branch = phases_[static_cast<size_t>(p)];
      branch.reserve(static_cast<size_t>(n / l_ + 1));
      for (int j = p; j < n; j += l_) branch.push_back(h[static_cast<size_t>(j)]);
    }
    return true;
  }

  bool isIdentity() const { return identity_; }
  bool wasCapped() const { return capped_; }
  int upFactor() const { return l_; }
  int downFactor() const { return m_; }
  int prototypeTaps() const { return identity_ ? 0 : 2 * halfLen_ + 1; }

  // Output frames produced for `inFrames` of input: every input time that
  // exists, resampled, and nothing beyond it.
  int64_t outputFrames(int64_t inFrames) const {
    if (inFrames <= 0) return 0;
    if (identity_) return inFrames;
    return ((inFrames - 1) * static_cast<int64_t>(l_)) / static_cast<int64_t>(m_) + 1;
  }

  // Whole-signal conversion. Offline, so there is no streaming variant and no
  // state to carry between calls — which is also why there is nothing here
  // that could drift over a long file.
  void process(const AudioBuffer& in, AudioBuffer& out) const {
    const int channels = in.channels();
    const int64_t inFrames = in.frames();
    const int64_t outFrames = outputFrames(inFrames);

    out.resize(channels, outFrames);
    if (identity_) {
      for (int c = 0; c < channels; ++c)
        std::copy(in.channel(c), in.channel(c) + inFrames, out.channel(c));
      return;
    }

    for (int c = 0; c < channels; ++c) {
      const Sample* x = in.channel(c);
      Sample* y = out.channel(c);

      for (int64_t frame = 0; frame < outFrames; ++frame) {
        // The +halfLen_ is the latency removal: it evaluates the interpolated
        // signal at the filter's centre rather than at its start.
        const int64_t a = frame * static_cast<int64_t>(m_) + halfLen_;
        const int p = static_cast<int>(a % static_cast<int64_t>(l_));
        const int64_t q = a / static_cast<int64_t>(l_);

        const auto& branch = phases_[static_cast<size_t>(p)];
        const int64_t taps = static_cast<int64_t>(branch.size());

        // Clip the tap range to the samples that exist rather than testing
        // bounds inside the inner loop.
        int64_t tLo = 0, tHi = taps - 1;
        if (q - tHi < 0) tHi = q;
        if (q - tLo > inFrames - 1) tLo = q - (inFrames - 1);

        double acc = 0.0;
        for (int64_t t = tLo; t <= tHi; ++t)
          acc += branch[static_cast<size_t>(t)] * x[q - t];
        y[frame] = acc;
      }
    }
  }

private:
  double inRate_ = 0.0, outRate_ = 0.0;
  int l_ = 1, m_ = 1, halfLen_ = 0;
  bool identity_ = true, capped_ = false;
  std::vector<std::vector<double>> phases_;
};

} // namespace anv
