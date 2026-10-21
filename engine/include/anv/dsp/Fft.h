// Minimal iterative radix-2 FFT.
//
// Used in the audio path — the partitioned convolver and the linear-phase EQ
// both run on it — as well as by the measurement suite. It was first written
// for the measurements only, on the assumption that rendering would use an FFT
// library; it did not, and this comment said otherwise for some time. Hand-
// written so the engine has no external dependency and produces exactly
// reproducible numbers in CI.
#pragma once

#include <cmath>
#include <complex>
#include <vector>

namespace anv {

inline void fftInPlace(std::vector<std::complex<double>>& a) {
  const size_t n = a.size();
  if (n <= 1) return;

  // Bit-reversal permutation.
  for (size_t i = 1, j = 0; i < n; ++i) {
    size_t bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(a[i], a[j]);
  }

  for (size_t len = 2; len <= n; len <<= 1) {
    const double ang = -2.0 * M_PI / static_cast<double>(len);
    const std::complex<double> wlen(std::cos(ang), std::sin(ang));
    for (size_t i = 0; i < n; i += len) {
      std::complex<double> w(1.0, 0.0);
      for (size_t k = 0; k < len / 2; ++k) {
        const std::complex<double> u = a[i + k];
        const std::complex<double> v = a[i + k + len / 2] * w;
        a[i + k] = u + v;
        a[i + k + len / 2] = u - v;
        w *= wlen;
      }
    }
  }
}

// Inverse transform via the conjugation identity: ifft(x) = conj(fft(conj(x)))/N.
// Reuses the forward transform rather than duplicating the butterfly loop,
// which means there is only one place a twiddle-factor bug could live.
inline void ifftInPlace(std::vector<std::complex<double>>& a) {
  const size_t n = a.size();
  if (n <= 1) return;
  for (auto& v : a) v = std::conj(v);
  fftInPlace(a);
  const double scale = 1.0 / static_cast<double>(n);
  for (auto& v : a) v = std::conj(v) * scale;
}

// Power spectrum of a real signal, bins 0..n/2.
inline std::vector<double> powerSpectrum(const double* x, size_t n) {
  std::vector<std::complex<double>> buf(n);
  for (size_t i = 0; i < n; ++i) buf[i] = {x[i], 0.0};
  fftInPlace(buf);
  std::vector<double> p(n / 2 + 1);
  for (size_t i = 0; i <= n / 2; ++i) p[i] = std::norm(buf[i]);
  return p;
}

} // namespace anv
