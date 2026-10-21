// Audionova render engine — fundamental types.
//
// The engine is float64 end-to-end. Feedback structures (reverb tails, delay
// feedback, high-Q IIR cascades) accumulate float32 error into the -90 to
// -110 dBFS range over a five-minute render; float64 lands below -250. We are
// offline, so there is no reason to accept the cheaper option.
#pragma once

#include <algorithm>
#include <cstdint>

namespace anv {

using Sample = double;

// All processing happens at one canonical internal rate. Resample exactly once
// on ingest and once on export, never mid-chain. 96 kHz specifically because
// sample-rate-dependent character processors (the Airwindows family is the
// usual example) adapt their behaviour to the rate by design, so a single
// canonical rate is the only way renders stay reproducible across whatever the
// user happened to upload. None is vendored; see docs/licensing.md.
inline constexpr double kCanonicalSampleRate = 96000.0;

// Same-binary properties (determinism, block-size invariance, reset
// equivalence, bypass null) must be BIT EXACT. Cross-platform comparison uses
// an amplitude threshold instead, because libm ULP differences, SIMD summation
// order and FMA contraction make bit-exactness across x86/ARM a losing fight.
// -140 dBFS is ~23 bits below full scale — far below the noise floor of any
// delivery format.
inline constexpr double kCrossPlatformNullDb = -140.0;

// Energy below this is silence for the purpose of tail-length measurement.
inline constexpr double kTailFloorDb = -140.0;

// Every decibel parameter is clamped to this before being converted to a
// linear gain. Without it a nonsensical value — a unit mix-up, a percent
// mistaken for dB, a model emitting 1e5 — becomes 10^5000, which is Inf, and
// a single Inf poisons every subsequent sample of the render beyond recovery.
//
// The engine clamps defensively rather than trusting its caller, because the
// caller is going to be a language model. +/-80 dB is a factor of 10,000 in
// either direction: far outside anything musical, and comfortably finite.
inline constexpr double kMaxAbsDb = 80.0;

inline double clampDb(double db) {
  return std::clamp(db, -kMaxAbsDb, kMaxAbsDb);
}

} // namespace anv
