// Planar, non-interleaved audio storage.
//
// Interleaved layout is a real-time-plugin convention with no offline benefit.
// Planar vectorizes trivially and keeps per-channel logic simple.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "anv/core/Types.h"

namespace anv {

class AudioBuffer {
public:
  AudioBuffer() = default;
  AudioBuffer(int channels, int64_t frames) { resize(channels, frames); }

  void resize(int channels, int64_t frames) {
    data_.assign(static_cast<size_t>(channels),
                 std::vector<Sample>(static_cast<size_t>(frames), 0.0));
    ptrs_.resize(static_cast<size_t>(channels));
    constPtrs_.resize(static_cast<size_t>(channels));
    for (size_t c = 0; c < data_.size(); ++c) {
      ptrs_[c] = data_[c].data();
      constPtrs_[c] = data_[c].data();
    }
    frames_ = frames;
  }

  void clear() {
    for (auto& ch : data_) std::fill(ch.begin(), ch.end(), 0.0);
  }

  int channels() const { return static_cast<int>(data_.size()); }
  int64_t frames() const { return frames_; }

  Sample* channel(int c) { return data_[static_cast<size_t>(c)].data(); }
  const Sample* channel(int c) const { return data_[static_cast<size_t>(c)].data(); }

  Sample* const* planes() { return ptrs_.data(); }
  const Sample* const* planes() const { return constPtrs_.data(); }

  // Views offset into the buffer, for feeding a chain block by block without
  // copying. The returned pointer arrays are owned by the caller's scratch.
  void planesAt(int64_t offset, std::vector<Sample*>& out) {
    out.resize(data_.size());
    for (size_t c = 0; c < data_.size(); ++c)
      out[c] = data_[c].data() + offset;
  }
  void constPlanesAt(int64_t offset, std::vector<const Sample*>& out) const {
    out.resize(data_.size());
    for (size_t c = 0; c < data_.size(); ++c)
      out[c] = data_[c].data() + offset;
  }

private:
  std::vector<std::vector<Sample>> data_;
  std::vector<Sample*> ptrs_;
  std::vector<const Sample*> constPtrs_;
  int64_t frames_ = 0;
};

} // namespace anv
