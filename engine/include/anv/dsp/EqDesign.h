// Design pieces shared by the three EQ nodes.
//
// `parametric_eq`, `zero_phase_eq` and `linear_phase_eq` all describe the SAME
// filter — the same bands, the same shapes, the same Butterworth slopes — and
// differ only in how that filter is realised in time. Keeping the design in
// one place is not tidiness: a divergence between two of them would show up as
// "the linear-phase version sounds different from the minimum-phase one at the
// same settings", which reads as an inherent property of linear phase rather
// than as the bug it would be.
//
// Only the SHAPE MAPPING and the impulse-response measurement live here. Each
// node still reads its own band parameters by name in its own header, because
// a parameter the engine never looks up is a parameter the planner can reason
// about and the render will silently ignore — and the schema test finds those
// by grepping `nodes/` for `num()`/`str()` lookups.
#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "anv/core/Types.h"
#include "anv/dsp/TptSvf.h"

namespace anv {

// Unknown names fall through to Bell here, exactly as ParametricEQ does.
// That fallback is safe ONLY because `validateRecipe` rejects an unrecognised
// `filter_type` before any of this runs — see the note in Recipe.h. If a new
// EQ node is added and forgotten there, a planner's typo becomes a bell again.
inline FilterShape eqShapeFromString(const std::string& s) {
  if (s == "lowshelf") return FilterShape::LowShelf;
  if (s == "highshelf") return FilterShape::HighShelf;
  if (s == "notch") return FilterShape::Notch;
  if (s == "lowpass") return FilterShape::LowPass;
  if (s == "highpass") return FilterShape::HighPass;
  return FilterShape::Bell;
}

// A cascade of N identical 2nd-order sections gives 12*N dB/oct. Q values are
// the Butterworth pole Qs for the cascaded order.
inline std::vector<double> eqButterworthQs(int order) {
  std::vector<double> qs;
  const int sections = order / 2;
  for (int k = 0; k < sections; ++k) {
    const double theta = M_PI * (2.0 * static_cast<double>(k) + 1.0) /
                         (2.0 * static_cast<double>(order));
    qs.push_back(1.0 / (2.0 * std::cos(theta)));
  }
  return qs;
}

// The cascade's impulse response, truncated where the REMAINING TAIL stops
// mattering rather than where the samples get small.
//
// Both offline EQ nodes need a length that says "the filter has finished".
// ZeroPhaseEQ needs it because its reverse pass is started from rest that far
// ahead of each output sample, so the truncated tail IS the error; LinearPhaseEQ
// needs it because it takes the FFT of this response as its magnitude target.
//
// The criterion is the L1 SUM of what is thrown away, not the amplitude of the
// last sample kept. That distinction is the whole reason this function exists:
// the error a truncated reverse pass makes at one output sample is bounded by
// |w|_inf * sum_{n>L} |h[n]|, and a filter can sit far below any per-sample
// threshold while a few thousand of those samples still add up to something
// audible. A high-Q bell at 30 Hz is exactly that filter.
//
// `maxLen` is a hard cap, and hitting it is a real (documented) approximation
// rather than a value chosen to look tidy — see the reach cap in ZeroPhaseEQ.h.
inline std::vector<double> cascadeImpulseResponse(std::vector<TptSvf> chain,
                                                  int maxLen, double tailTol) {
  if (maxLen < 1) maxLen = 1;
  for (auto& f : chain) f.reset();

  std::vector<double> h;
  h.reserve(static_cast<size_t>(maxLen));
  double total = 0.0;
  for (int n = 0; n < maxLen; ++n) {
    Sample x = (n == 0) ? 1.0 : 0.0;
    for (auto& f : chain) x = f.processSample(x);
    h.push_back(x);
    total += std::abs(x);
  }

  // A cascade that produces nothing at all is not a length this can measure.
  // Report one sample rather than inventing a decay time for a filter that has
  // no decay.
  if (!(total > 0.0)) return std::vector<double>(1, 0.0);

  const double limit = tailTol * total;
  double tail = 0.0;
  int len = 1;
  for (int n = maxLen - 1; n >= 1; --n) {
    tail += std::abs(h[static_cast<size_t>(n)]);
    if (tail > limit) {
      len = n + 1;
      break;
    }
  }
  h.resize(static_cast<size_t>(len));
  return h;
}

} // namespace anv
