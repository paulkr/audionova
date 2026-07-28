// Uniform-partitioned overlap-save convolution.
//
// DELIBERATELY THE SIMPLE STRUCTURE. Real-time convolution reverbs use
// non-uniform partitioning — tiny partitions first, growing toward the tail —
// so the first output arrives with near-zero latency. That machinery exists
// entirely to buy back latency, and it is fiddly enough to be a recurring
// source of bugs.
//
// We are offline and we declare our latency explicitly, so we spend a whole
// partition of latency and take the structure that is easy to get right:
// every partition the same size, one FFT in, one FFT out, a frequency-domain
// delay line in between. Latency is free; complexity is not.
//
// The partition size is derived from the IR length ALONE, never from the
// caller's block size — deriving it from maxBlockFrames would make the output
// depend on how the host happened to chunk the audio, which is exactly the
// block-size dependence the whole engine is built to avoid.
#pragma once

#include <algorithm>
#include <complex>
#include <vector>

#include "anv/core/Types.h"
#include "anv/dsp/Fft.h"

namespace anv {

class PartitionedConvolver {
public:
  void prepare(const std::vector<Sample>& ir) {
    irLength_ = static_cast<int>(ir.size());
    if (irLength_ <= 0) {
      blockSize_ = 0;
      return;
    }

    // Fixed policy: a partition roughly a quarter of the IR, clamped, rounded
    // to a power of two. Short IRs stay cheap and low latency; long tails cap
    // out so the FFT does not become the dominant cost.
    int target = std::clamp(irLength_ / 4, 64, 4096);
    blockSize_ = 1;
    while (blockSize_ < target) blockSize_ <<= 1;

    fftSize_ = blockSize_ * 2;
    numParts_ = (irLength_ + blockSize_ - 1) / blockSize_;

    // Pre-transform each IR partition once.
    H_.assign(static_cast<size_t>(numParts_), {});
    for (int p = 0; p < numParts_; ++p) {
      std::vector<std::complex<double>> buf(static_cast<size_t>(fftSize_),
                                            std::complex<double>(0.0, 0.0));
      const int from = p * blockSize_;
      const int count = std::min(blockSize_, irLength_ - from);
      for (int i = 0; i < count; ++i)
        buf[static_cast<size_t>(i)] = ir[static_cast<size_t>(from + i)];
      fftInPlace(buf);
      H_[static_cast<size_t>(p)] = std::move(buf);
    }

    fdl_.assign(static_cast<size_t>(numParts_),
                std::vector<std::complex<double>>(static_cast<size_t>(fftSize_),
                                                  std::complex<double>(0.0, 0.0)));
    inBlock_.assign(static_cast<size_t>(blockSize_), 0.0);
    prevIn_.assign(static_cast<size_t>(blockSize_), 0.0);
    outReady_.assign(static_cast<size_t>(blockSize_), 0.0);
    scratch_.assign(static_cast<size_t>(fftSize_), std::complex<double>(0.0, 0.0));
    accum_.assign(static_cast<size_t>(fftSize_), std::complex<double>(0.0, 0.0));
    reset();
  }

  void reset() {
    std::fill(inBlock_.begin(), inBlock_.end(), 0.0);
    std::fill(prevIn_.begin(), prevIn_.end(), 0.0);
    std::fill(outReady_.begin(), outReady_.end(), 0.0);
    for (auto& v : fdl_)
      std::fill(v.begin(), v.end(), std::complex<double>(0.0, 0.0));
    fdlPos_ = 0;
    phase_ = 0;
  }

  // Exactly one partition. Integer by construction.
  int latencyFrames() const { return blockSize_; }
  int irLength() const { return irLength_; }

  void process(const Sample* in, Sample* out, int n) noexcept {
    if (blockSize_ == 0) {
      for (int i = 0; i < n; ++i) out[i] = 0.0;
      return;
    }
    for (int i = 0; i < n; ++i) {
      out[i] = outReady_[static_cast<size_t>(phase_)];
      inBlock_[static_cast<size_t>(phase_)] = in[i];
      if (++phase_ == blockSize_) {
        phase_ = 0;
        transformBlock();
      }
    }
  }

private:
  void transformBlock() noexcept {
    // Overlap-save: transform [previous block | current block]. The first half
    // of the circular convolution is the wrapped garbage that overlap-save
    // discards; the second half is the true linear convolution.
    for (int i = 0; i < blockSize_; ++i) {
      scratch_[static_cast<size_t>(i)] = prevIn_[static_cast<size_t>(i)];
      scratch_[static_cast<size_t>(blockSize_ + i)] = inBlock_[static_cast<size_t>(i)];
    }
    fftInPlace(scratch_);
    fdl_[static_cast<size_t>(fdlPos_)] = scratch_;

    std::fill(accum_.begin(), accum_.end(), std::complex<double>(0.0, 0.0));
    for (int p = 0; p < numParts_; ++p) {
      int idx = fdlPos_ - p;
      if (idx < 0) idx += numParts_;
      const auto& X = fdl_[static_cast<size_t>(idx)];
      const auto& Hp = H_[static_cast<size_t>(p)];
      for (int k = 0; k < fftSize_; ++k)
        accum_[static_cast<size_t>(k)] += X[static_cast<size_t>(k)] * Hp[static_cast<size_t>(k)];
    }

    ifftInPlace(accum_);
    for (int i = 0; i < blockSize_; ++i)
      outReady_[static_cast<size_t>(i)] = accum_[static_cast<size_t>(blockSize_ + i)].real();

    prevIn_ = inBlock_;
    fdlPos_ = (fdlPos_ + 1) % numParts_;
  }

  int irLength_ = 0, blockSize_ = 0, fftSize_ = 0, numParts_ = 0;
  int fdlPos_ = 0, phase_ = 0;
  std::vector<std::vector<std::complex<double>>> H_, fdl_;
  std::vector<Sample> inBlock_, prevIn_, outReady_;
  std::vector<std::complex<double>> scratch_, accum_;
};

} // namespace anv
