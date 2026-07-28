// Fractional delay line with Thiran allpass interpolation.
//
// WHY ALLPASS RATHER THAN LAGRANGE. In a feedback delay the interpolator runs
// on every single repeat, so whatever it does to the signal compounds. A
// Lagrange (FIR) interpolator has a gentle high-frequency rolloff; one pass is
// inaudible, but twenty repeats apply it twenty times and the tail audibly
// darkens. That darkening is then indistinguishable from the damping control,
// except the engineer cannot turn it off.
//
// A first-order Thiran allpass has EXACTLY unity magnitude at every frequency.
// Repeats keep their brightness, and the only thing dulling the tail is the
// damping filter the user actually asked for.
//
// The tradeoff is that an allpass carries state, so changing the delay time
// mid-render produces a transient. That is fine here — this delay's time is
// static. A modulated delay (chorus, flanger) wants Lagrange instead, for
// exactly the opposite reason.
//
// The fractional part is kept in [0.5, 1.5) by borrowing a sample from the
// integer delay: the first-order allpass is ill-conditioned as its delay
// approaches zero, where the pole migrates toward the unit circle.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "anv/core/Types.h"

namespace anv {

class ThiranAllpass1 {
public:
  // `delaySamples` should lie in [0.5, 1.5) for good conditioning.
  void setDelay(double delaySamples) {
    a_ = (1.0 - delaySamples) / (1.0 + delaySamples);
  }
  void reset() {
    x1_ = 0.0;
    y1_ = 0.0;
  }
  inline Sample process(Sample x) noexcept {
    const double y = a_ * x + x1_ - a_ * y1_;
    x1_ = x;
    y1_ = y;
    return y;
  }

private:
  double a_ = 0.0, x1_ = 0.0, y1_ = 0.0;
};

class FractionalDelayLine {
public:
  void prepare(int maxDelaySamples) {
    size_ = std::max(4, maxDelaySamples + 4);
    buf_.assign(static_cast<size_t>(size_), 0.0);
    writePos_ = 0;
    reset();
  }

  // Total delay in samples, split into an integer part and an allpass part
  // held in the well-conditioned range.
  void setDelay(double samples) {
    samples = std::clamp(samples, 1.0, static_cast<double>(size_ - 3));
    intDelay_ = static_cast<int>(std::floor(samples - 0.5));
    if (intDelay_ < 0) intDelay_ = 0;
    ap_.setDelay(samples - static_cast<double>(intDelay_));
  }

  void reset() {
    std::fill(buf_.begin(), buf_.end(), 0.0);
    ap_.reset();
    writePos_ = 0;
  }

  // Must be called exactly once per sample: the allpass advances its state.
  inline Sample read() noexcept {
    int r = writePos_ - intDelay_;
    if (r < 0) r += size_;
    return ap_.process(buf_[static_cast<size_t>(r)]);
  }

  inline void write(Sample x) noexcept {
    buf_[static_cast<size_t>(writePos_)] = x;
    if (++writePos_ >= size_) writePos_ = 0;
  }

private:
  std::vector<Sample> buf_;
  ThiranAllpass1 ap_;
  int size_ = 0, writePos_ = 0, intDelay_ = 0;
};

} // namespace anv
