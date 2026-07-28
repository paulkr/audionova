// Counter-based (stateless) pseudo-random generation.
//
// A stateful LCG/xorshift advanced per block produces DIFFERENT output for
// different block partitions, which silently breaks block-size invariance and
// makes segmented parallel rendering impossible. Every stochastic element in
// the engine — dither, analog noise, modulation LFO start phase — is instead a
// pure function of (seed, node, channel, absolute frame index).
//
// That makes determinism structural rather than something to test for and hope
// about: the same recipe and seed produce bit-identical audio regardless of
// block size, thread count, or evaluation order.
#pragma once

#include <cstdint>

namespace anv {

// SplitMix64 finalizer. Strong avalanche, trivially vectorizable, no state.
inline uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

// Derives a per-node seed from the global recipe seed and the node's stable id.
// Independent of graph traversal order and thread scheduling.
inline uint64_t deriveNodeSeed(uint64_t globalSeed, const char* nodeId) {
  // FNV-1a over the node id.
  uint64_t h = 1469598103934665603ull;
  for (const char* p = nodeId; p && *p; ++p) {
    h ^= static_cast<uint64_t>(static_cast<unsigned char>(*p));
    h *= 1099511628211ull;
  }
  return splitmix64(globalSeed ^ h);
}

class CounterRng {
public:
  explicit CounterRng(uint64_t nodeSeed = 0) : seed_(nodeSeed) {}

  void setSeed(uint64_t s) { seed_ = s; }

  // Uniform in [0, 1). Pure function of its arguments — O(1) random access,
  // identical under any block partition.
  double uniform(int channel, int64_t frame, uint32_t stream = 0) const {
    uint64_t a = splitmix64(seed_ ^ (0xD6E8FEB86659FD93ull * static_cast<uint64_t>(stream + 1)));
    uint64_t b = splitmix64(a ^ (0x9E3779B97F4A7C15ull * static_cast<uint64_t>(channel + 1)));
    uint64_t c = splitmix64(b ^ static_cast<uint64_t>(frame));
    // Top 53 bits -> double in [0,1), exactly representable.
    return static_cast<double>(c >> 11) * (1.0 / 9007199254740992.0);
  }

  // Symmetric in [-1, 1).
  double bipolar(int channel, int64_t frame, uint32_t stream = 0) const {
    return uniform(channel, frame, stream) * 2.0 - 1.0;
  }

  // Triangular PDF, the correct shape for flat TPDF dither at 2 LSB
  // peak-to-peak. Two independent streams so the two rectangular draws are not
  // correlated.
  double tpdf(int channel, int64_t frame, uint32_t stream = 0) const {
    return uniform(channel, frame, stream * 2 + 1) -
           uniform(channel, frame, stream * 2 + 2);
  }

private:
  uint64_t seed_;
};

} // namespace anv
