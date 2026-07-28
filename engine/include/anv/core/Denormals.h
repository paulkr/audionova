// Flush-to-zero / denormals-are-zero control.
//
// Denormals hit hardest exactly where this engine lives: reverb tails decaying
// toward silence, feedback delays, and high-Q filters after a fade. A single
// unprotected reverb tail can cost 10-100x the nominal render time.
//
// FTZ/DAZ changes numeric results, so the test harness MUST enable it exactly
// as the render loop does — otherwise golden references drift between the two.
// `enabled()` exists so a test can read the control register back and assert
// nobody ran the suite without it.
#pragma once

#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <pmmintrin.h>
#include <xmmintrin.h>
#endif

namespace anv {

#if defined(__aarch64__) || defined(_M_ARM64)

inline uint64_t readFpControl() {
  uint64_t v;
  __asm__ __volatile__("mrs %0, fpcr" : "=r"(v));
  return v;
}
inline void writeFpControl(uint64_t v) {
  __asm__ __volatile__("msr fpcr, %0" : : "r"(v));
}
// FPCR bit 24 (FZ) flushes denormal inputs and outputs to zero.
inline constexpr uint64_t kFlushBits = 1ull << 24;

#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__)

inline uint64_t readFpControl() { return _mm_getcsr(); }
inline void writeFpControl(uint64_t v) { _mm_setcsr(static_cast<unsigned>(v)); }
// MXCSR bit 15 (FTZ) and bit 6 (DAZ). Both are required; FTZ alone still lets
// denormal *inputs* trigger the slow path.
inline constexpr uint64_t kFlushBits = (1ull << 15) | (1ull << 6);

#else

inline uint64_t readFpControl() { return 0; }
inline void writeFpControl(uint64_t) {}
inline constexpr uint64_t kFlushBits = 0;

#endif

// RAII. Save the control register, force flush-to-zero, restore on scope exit.
class ScopedNoDenormals {
public:
  ScopedNoDenormals() : saved_(readFpControl()) {
    writeFpControl(saved_ | kFlushBits);
  }
  ~ScopedNoDenormals() { writeFpControl(saved_); }

  ScopedNoDenormals(const ScopedNoDenormals&) = delete;
  ScopedNoDenormals& operator=(const ScopedNoDenormals&) = delete;

  // True when denormal flushing is currently active. Assert this in the test
  // harness so the suite cannot silently run in a different numeric mode than
  // production.
  static bool enabled() {
    if (kFlushBits == 0) return true; // platform has no such control
    return (readFpControl() & kFlushBits) == kFlushBits;
  }

private:
  uint64_t saved_;
};

} // namespace anv
