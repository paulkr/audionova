// Integer delay line.
//
// Used for two things that must be exactly right or the whole engine is
// suspect: compensating a bypassed node's declared latency, and aligning the
// dry path in the engine's dry/wet wrapper. Both are integer-only by design —
// there is deliberately no fractional-delay path here, because every node in
// the engine reports integer latency.
#pragma once

#include <algorithm>
#include <vector>

#include "anv/core/Types.h"

namespace anv {

class DelayLine {
public:
  void prepare(int channels, int maxDelayFrames) {
    delay_ = maxDelayFrames;
    // +1 so a delay of exactly `maxDelayFrames` is representable.
    buf_.assign(static_cast<size_t>(channels),
                std::vector<Sample>(static_cast<size_t>(maxDelayFrames) + 1, 0.0));
    writePos_ = 0;
  }

  void reset() {
    for (auto& ch : buf_) std::fill(ch.begin(), ch.end(), 0.0);
    writePos_ = 0;
  }

  int delayFrames() const { return delay_; }

  // Push `n` frames in and pull the same count out, delayed by delayFrames().
  // Safe when in == out.
  void process(const Sample* const* in, Sample* const* out, int n) noexcept {
    if (delay_ <= 0) {
      if (in != out) {
        for (size_t c = 0; c < buf_.size(); ++c)
          std::copy(in[c], in[c] + n, out[c]);
      }
      return;
    }
    const int len = static_cast<int>(buf_[0].size());
    for (int i = 0; i < n; ++i) {
      const int w = writePos_;
      int r = w - delay_;
      if (r < 0) r += len;
      for (size_t c = 0; c < buf_.size(); ++c) {
        const Sample x = in[c][i];
        const Sample y = buf_[c][static_cast<size_t>(r)];
        buf_[c][static_cast<size_t>(w)] = x;
        out[c][i] = y;
      }
      writePos_ = (w + 1) % len;
    }
  }

private:
  std::vector<std::vector<Sample>> buf_;
  int delay_ = 0;
  int writePos_ = 0;
};

} // namespace anv
