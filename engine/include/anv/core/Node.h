// The node contract every effect implements.
//
// Five rules are enforced rather than merely requested, because each one maps
// directly to a contract test that every node is automatically enrolled in:
//
//   1. No allocation in process(). Everything is sized in prepare().
//   2. No static mutable state, no rand(), no dependence on time or thread id.
//   3. reset() must return the node to a bit-exact fresh state.
//   4. process() is noexcept — after prepare() succeeds, failure is impossible.
//   5. latencyFrames() is EXACT and INTEGER, never an estimate.
//
// Rule 5 is what makes plugin delay compensation provable. Because the
// oversampling wrapper's up/down filter pair has provably integer latency at
// the base rate, there is no fractional-delay compensation anywhere in the
// engine — removing the single largest source of PDC bugs.
//
// The central invariant, which rules 1-5 exist to protect:
//
//   ANY per-block computation whose result depends on the block size is a bug.
//
// That outlaws the standard real-time plugin idioms: per-block parameter
// smoothing, per-block envelope coefficient recalculation, LFOs advanced by
// numFrames at block start. The replacements all key off globalFrameIndex, so
// they are invariant by construction rather than by testing and hoping.
#pragma once

#include <cstdint>

#include "anv/core/ParamSet.h"
#include "anv/core/Types.h"

namespace anv {

struct PrepareSpec {
  double sampleRate = kCanonicalSampleRate;
  int maxBlockFrames = 1024; // upper bound only; process() may get any n <= this
  int numChannels = 2;
  int64_t totalFrames = -1;  // known offline; -1 when streaming
  uint64_t seed = 0;         // already derived per-node; see Rng.h
};

struct ProcessContext {
  const Sample* const* in = nullptr;
  Sample* const* out = nullptr;
  int numFrames = 0;
  // Absolute position in the render. This is what makes every time-dependent
  // computation a pure function rather than an accumulation, and therefore
  // identical under any block partition.
  int64_t globalFrameIndex = 0;
};

class Node {
public:
  virtual ~Node() = default;

  virtual const char* type() const = 0;

  // Allocate everything here. Nothing may allocate after this returns.
  virtual void prepare(const PrepareSpec& spec) = 0;

  // Return to the exact state of a freshly prepared instance. Bit-exact.
  virtual void reset() = 0;

  virtual void process(ProcessContext& ctx) noexcept = 0;

  virtual void setParams(const ParamSet& params) = 0;

  // Exact integer latency in frames, valid after prepare(). Never an estimate.
  virtual int latencyFrames() const { return 0; }

  // Frames of output after input ends before energy falls below the tail floor.
  virtual int64_t tailFrames() const { return 0; }

  // Nodes whose dry/wet is intrinsic to the algorithm opt out of the engine's
  // wrapper. Everything else gets phase-correct mix for free; see Chain.h.
  virtual bool wantsEngineMix() const { return true; }

protected:
  PrepareSpec spec_{};
};

} // namespace anv
