// WAV / RF64 reading and writing.
//
// WHY HAND-WRITTEN. libsndfile was the obvious choice and is rejected on two
// counts, either of which alone would be decisive. It is our only LGPL
// dependency, which for a product that may one day ship a static binary is a
// distribution obligation nobody wants to discover late. And it is a PARSER
// exposed to bytes we did not create — historically the single most
// productive category of remote code execution there is. Dropping it means the
// engine links nothing that touches untrusted input and carries no copyleft.
//
// WHAT THIS IS NOT. This reader is NOT the untrusted-input path. Anything a
// user uploads goes through the sandboxed FFmpeg ingest stage first and
// arrives here as a file we wrote ourselves. That is the design, and it is the
// reason a ~500-line parser is an acceptable thing to own.
//
// It is still written as though that were not true. Every length in the file
// is attacker-controlled in the threat model even if it should not be in
// practice, so every one of them is validated against the bytes actually
// present before a single allocation is sized from it. Defence in depth costs
// about forty lines here.
//
// FLOAT SAMPLES ARE NOT NORMALISED ON READ. A 32-bit float WAV can legally
// contain values outside [-1, 1] — that is most of the point of the format,
// and a mixdown handed to us hot is a completely ordinary thing to receive.
// Clamping on read would silently destroy headroom the engineer deliberately
// left in place. Integer formats are scaled by the format's full-scale value
// and cannot exceed unity by construction.
//
// RF64 IS READ AND WRITTEN, not merely tolerated. A 3-minute stereo file is
// nowhere near 4 GB, but a 96 kHz 32-bit multichannel stem set is, and the
// failure mode of plain WAV past 4 GB is a silently wrong 32-bit size field
// rather than an error. We write RIFF when it fits and RF64 when it does not.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "anv/core/AudioBuffer.h"
#include "anv/core/Types.h"

namespace anv {

enum class SampleFormat {
  PcmInt16,
  PcmInt24,
  PcmInt32,
  Float32,
  Float64,
};

inline int bytesPerSample(SampleFormat f) {
  switch (f) {
    case SampleFormat::PcmInt16: return 2;
    case SampleFormat::PcmInt24: return 3;
    case SampleFormat::PcmInt32: return 4;
    case SampleFormat::Float32:  return 4;
    case SampleFormat::Float64:  return 8;
  }
  return 0;
}

inline int bitDepthOf(SampleFormat f) { return bytesPerSample(f) * 8; }

inline const char* formatName(SampleFormat f) {
  switch (f) {
    case SampleFormat::PcmInt16: return "16-bit PCM";
    case SampleFormat::PcmInt24: return "24-bit PCM";
    case SampleFormat::PcmInt32: return "32-bit PCM";
    case SampleFormat::Float32:  return "32-bit float";
    case SampleFormat::Float64:  return "64-bit float";
  }
  return "unknown";
}

struct AudioFile {
  AudioBuffer audio;
  double sampleRate = 0.0;
  SampleFormat format = SampleFormat::Float32;
};

class WavError : public std::runtime_error {
public:
  explicit WavError(const std::string& what) : std::runtime_error(what) {}
};

namespace detail {

// A cursor over an in-memory file. Every read is bounds-checked against the
// real buffer length; nothing trusts a size written in the file itself.
class ByteReader {
public:
  ByteReader(const uint8_t* data, size_t size) : d_(data), n_(size) {}

  size_t remaining() const { return n_ - p_; }
  size_t pos() const { return p_; }

  void require(size_t bytes, const char* what) const {
    if (remaining() < bytes)
      throw WavError(std::string("truncated file: wanted ") +
                     std::to_string(bytes) + " bytes for " + what +
                     " but only " + std::to_string(remaining()) + " remain");
  }

  void skip(size_t bytes, const char* what) {
    require(bytes, what);
    p_ += bytes;
  }

  uint32_t u32(const char* what) {
    require(4, what);
    uint32_t v;
    std::memcpy(&v, d_ + p_, 4);
    p_ += 4;
    return leToHost32(v);
  }

  uint64_t u64(const char* what) {
    require(8, what);
    uint64_t v;
    std::memcpy(&v, d_ + p_, 8);
    p_ += 8;
    return leToHost64(v);
  }

  uint16_t u16(const char* what) {
    require(2, what);
    uint16_t v;
    std::memcpy(&v, d_ + p_, 2);
    p_ += 2;
    return leToHost16(v);
  }

  void fourcc(char out[5], const char* what) {
    require(4, what);
    std::memcpy(out, d_ + p_, 4);
    out[4] = '\0';
    p_ += 4;
  }

  const uint8_t* at(size_t offset) const { return d_ + offset; }

private:
  // WAV is little-endian on every platform. Doing the swap arithmetically
  // rather than with a byte-order intrinsic keeps this correct on a big-endian
  // target without any conditional compilation, and costs nothing: every
  // compiler recognises the pattern and emits a single load on little-endian.
  static uint16_t leToHost16(uint16_t v) {
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&v);
    return static_cast<uint16_t>(b[0] | (b[1] << 8));
  }
  static uint32_t leToHost32(uint32_t v) {
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&v);
    return static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) |
           (static_cast<uint32_t>(b[2]) << 16) |
           (static_cast<uint32_t>(b[3]) << 24);
  }
  static uint64_t leToHost64(uint64_t v) {
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&v);
    uint64_t r = 0;
    for (int i = 7; i >= 0; --i) r = (r << 8) | b[i];
    return r;
  }

  const uint8_t* d_;
  size_t n_;
  size_t p_ = 0;
};

// Little-endian writers. Same reasoning as above: explicit byte assembly is
// portable and optimises away.
inline void putU16(std::vector<uint8_t>& out, uint16_t v) {
  out.push_back(static_cast<uint8_t>(v & 0xFF));
  out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
}
inline void putU32(std::vector<uint8_t>& out, uint32_t v) {
  for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}
inline void putU64(std::vector<uint8_t>& out, uint64_t v) {
  for (int i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}
inline void putTag(std::vector<uint8_t>& out, const char* tag) {
  for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>(tag[i]));
}

constexpr uint16_t kFormatPcm = 0x0001;
constexpr uint16_t kFormatFloat = 0x0003;
constexpr uint16_t kFormatExtensible = 0xFFFE;

// The leading two bytes of a WAVE_FORMAT_EXTENSIBLE GUID carry the format tag;
// the remaining fourteen are a fixed KSDATAFORMAT_SUBTYPE suffix.
constexpr uint8_t kKsdataformatSuffix[14] = {0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80,
                                             0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71};

} // namespace detail

// --------------------------------------------------------------------------
// Reading

// One sample, little-endian, to float64.
//
// The integer scaling is asymmetric on purpose. Two's complement runs from
// -32768 to +32767 at 16 bits, so dividing by 32768 maps the most negative
// code to exactly -1.0 and the most positive to 32767/32768. Dividing by
// 32767 instead would let full-scale negative read as -1.000030, which then
// clips on the way back out. Every serious implementation divides by 2^(n-1);
// the "symmetric" alternative is a bug that only shows up at full scale.
inline Sample decodeSample(const uint8_t* s, SampleFormat format) {
  switch (format) {
    case SampleFormat::PcmInt16: {
      const int16_t v = static_cast<int16_t>(s[0] | (s[1] << 8));
      return static_cast<double>(v) / 32768.0;
    }
    case SampleFormat::PcmInt24: {
      // Assemble into the TOP three bytes of a 32-bit word and arithmetic-shift
      // down, which sign-extends without a conditional.
      const int32_t v = static_cast<int32_t>(
          (static_cast<uint32_t>(s[0]) << 8) |
          (static_cast<uint32_t>(s[1]) << 16) |
          (static_cast<uint32_t>(s[2]) << 24));
      return static_cast<double>(v >> 8) / 8388608.0;
    }
    case SampleFormat::PcmInt32: {
      const int32_t v = static_cast<int32_t>(
          static_cast<uint32_t>(s[0]) | (static_cast<uint32_t>(s[1]) << 8) |
          (static_cast<uint32_t>(s[2]) << 16) |
          (static_cast<uint32_t>(s[3]) << 24));
      return static_cast<double>(v) / 2147483648.0;
    }
    case SampleFormat::Float32: {
      uint32_t bits = static_cast<uint32_t>(s[0]) |
                      (static_cast<uint32_t>(s[1]) << 8) |
                      (static_cast<uint32_t>(s[2]) << 16) |
                      (static_cast<uint32_t>(s[3]) << 24);
      float f;
      std::memcpy(&f, &bits, 4);
      return static_cast<double>(f);
    }
    case SampleFormat::Float64: {
      uint64_t bits = 0;
      for (int i = 7; i >= 0; --i) bits = (bits << 8) | s[i];
      double d;
      std::memcpy(&d, &bits, 8);
      return d;
    }
  }
  return 0.0;
}


inline std::vector<uint8_t> readWholeFile(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw WavError("cannot open " + path);

  std::vector<uint8_t> bytes;
  char chunk[65536];
  size_t got;
  while ((got = std::fread(chunk, 1, sizeof(chunk), f)) > 0)
    bytes.insert(bytes.end(), chunk, chunk + got);

  const bool bad = std::ferror(f) != 0;
  std::fclose(f);
  if (bad) throw WavError("read error on " + path);
  return bytes;
}

inline AudioFile decodeWav(const std::vector<uint8_t>& bytes) {
  using namespace detail;
  ByteReader r(bytes.data(), bytes.size());

  char tag[5];
  r.fourcc(tag, "RIFF tag");
  const bool isRf64 = (std::strcmp(tag, "RF64") == 0);
  if (!isRf64 && std::strcmp(tag, "RIFF") != 0)
    throw WavError("not a WAV file (expected RIFF or RF64, got '" +
                   std::string(tag) + "')");

  r.skip(4, "RIFF size");
  r.fourcc(tag, "WAVE tag");
  if (std::strcmp(tag, "WAVE") != 0)
    throw WavError("not a WAV file (expected WAVE, got '" + std::string(tag) + "')");

  // RF64 puts the real 64-bit sizes in a `ds64` chunk; the 32-bit fields in
  // the RIFF and data headers are then the sentinel 0xFFFFFFFF and must be
  // ignored rather than used.
  uint64_t ds64DataSize = 0;
  bool haveDs64 = false;

  bool haveFmt = false;
  uint16_t formatTag = 0, channels = 0, bitsPerSample = 0;
  uint32_t sampleRate = 0;

  size_t dataOffset = 0;
  uint64_t dataSize = 0;
  bool haveData = false;

  while (r.remaining() >= 8) {
    r.fourcc(tag, "chunk id");
    const uint32_t declared = r.u32("chunk size");

    // A chunk header may claim any size at all. Clamp to what is actually in
    // the buffer before it is used to advance the cursor or size anything.
    uint64_t size = declared;
    if (std::strcmp(tag, "data") == 0 && isRf64 && declared == 0xFFFFFFFFu)
      size = ds64DataSize;
    if (size > r.remaining()) size = r.remaining();

    if (std::strcmp(tag, "ds64") == 0) {
      r.require(size, "ds64 chunk");
      const size_t after = r.pos() + static_cast<size_t>(size);
      r.skip(8, "ds64 riff size");
      ds64DataSize = r.u64("ds64 data size");
      haveDs64 = true;
      r.skip(after - r.pos(), "ds64 remainder");
    } else if (std::strcmp(tag, "fmt ") == 0) {
      if (size < 16) throw WavError("fmt chunk too short");
      r.require(size, "fmt chunk");
      const size_t after = r.pos() + static_cast<size_t>(size);

      formatTag = r.u16("format tag");
      channels = r.u16("channel count");
      sampleRate = r.u32("sample rate");
      r.skip(4, "byte rate");
      r.skip(2, "block align");
      bitsPerSample = r.u16("bits per sample");

      if (formatTag == kFormatExtensible) {
        if (size < 40) throw WavError("WAVE_FORMAT_EXTENSIBLE fmt chunk too short");
        r.skip(2, "cbSize");
        r.skip(2, "valid bits");
        r.skip(4, "channel mask");
        // Only the first two GUID bytes carry the actual format.
        formatTag = r.u16("subformat tag");
        r.skip(14, "subformat GUID suffix");
      }

      haveFmt = true;
      r.skip(after - r.pos(), "fmt remainder");
    } else if (std::strcmp(tag, "data") == 0) {
      dataOffset = r.pos();
      dataSize = size;
      haveData = true;
      r.skip(static_cast<size_t>(size), "data chunk");
    } else {
      r.skip(static_cast<size_t>(size), "unknown chunk");
    }

    // Chunks are word-aligned: an odd size is followed by one pad byte, which
    // is NOT counted in the size field. Forgetting this misaligns every
    // subsequent chunk and is the classic WAV parser bug.
    if ((size & 1u) && r.remaining() >= 1) r.skip(1, "pad byte");
  }

  if (!haveFmt) throw WavError("no fmt chunk");
  if (!haveData) throw WavError("no data chunk");
  if (isRf64 && !haveDs64) throw WavError("RF64 file with no ds64 chunk");
  if (channels < 1) throw WavError("file declares zero channels");
  if (sampleRate < 1) throw WavError("file declares a zero sample rate");

  SampleFormat format;
  if (formatTag == kFormatPcm) {
    switch (bitsPerSample) {
      case 16: format = SampleFormat::PcmInt16; break;
      case 24: format = SampleFormat::PcmInt24; break;
      case 32: format = SampleFormat::PcmInt32; break;
      default:
        throw WavError("unsupported PCM depth: " + std::to_string(bitsPerSample) +
                       " bits");
    }
  } else if (formatTag == kFormatFloat) {
    switch (bitsPerSample) {
      case 32: format = SampleFormat::Float32; break;
      case 64: format = SampleFormat::Float64; break;
      default:
        throw WavError("unsupported float depth: " + std::to_string(bitsPerSample) +
                       " bits");
    }
  } else {
    throw WavError("unsupported WAV format tag 0x" +
                   std::to_string(formatTag) +
                   " (only uncompressed PCM and float are read here; "
                   "everything else goes through the ingest decoder)");
  }

  const int bps = bytesPerSample(format);
  const int64_t frameBytes = static_cast<int64_t>(bps) * channels;
  const int64_t frames = static_cast<int64_t>(dataSize) / frameBytes;

  AudioFile out;
  out.sampleRate = static_cast<double>(sampleRate);
  out.format = format;
  out.audio.resize(channels, frames);

  const uint8_t* src = bytes.data() + dataOffset;
  for (int64_t i = 0; i < frames; ++i) {
    const uint8_t* frame = src + i * frameBytes;
    for (int c = 0; c < channels; ++c) {
      const uint8_t* s = frame + static_cast<int64_t>(c) * bps;
      out.audio.channel(c)[i] = decodeSample(s, format);
    }
  }
  return out;
}

inline AudioFile readWav(const std::string& path) {
  return decodeWav(readWholeFile(path));
}

// --------------------------------------------------------------------------
// Writing

// Integer formats cannot represent anything past full scale, so the choice is
// between clamping and wrapping. Wrapping turns a 0.1 dB overshoot into
// full-amplitude polarity inversion — the single most destructive artifact in
// audio. Clamp.
inline double clampUnity(double v) {
  if (!(v == v)) return 0.0;  // NaN in, silence out, rather than a wrapped code
  return v < -1.0 ? -1.0 : (v > 1.0 ? 1.0 : v);
}


// One sample, float64 to the target format, little-endian.
//
// NO DITHER HERE. Bit-depth reduction is a DSP decision with audible
// consequences and it belongs to the Dither node, where it is placed correctly
// in the chain, is deterministic, and can be turned off. A writer that
// quietly dithers would apply it after the true-peak limiter's ceiling was
// already met and would be impossible to disable — so this writer rounds, and
// nothing else.
// Quantise to a signed integer of `bits`, scaling by 2^(bits-1) and clamping
// to the representable range.
//
// SCALE BY 2^(n-1), NOT 2^(n-1)-1. Decoding divides by 32768, so multiplying
// by 32768 here makes read->write BIT-EXACT for every one of the 65536 codes,
// which is a property the test suite asserts. The common alternative — scale
// by 32767 to dodge the overflow at +1.0 — makes a decode/encode round trip
// lossy at every sample and gains nothing, since the range still has to be
// clamped for the float case anyway.
inline int32_t quantise(Sample v, double scale, int32_t lo, int32_t hi) {
  const double q = std::round(clampUnity(v) * scale);
  if (q <= static_cast<double>(lo)) return lo;
  if (q >= static_cast<double>(hi)) return hi;
  return static_cast<int32_t>(q);
}

// One sample, float64 to the target format, little-endian.
//
// NO DITHER HERE. Bit-depth reduction is a DSP decision with audible
// consequences and it belongs to the Dither node, where it sits in a known
// place in the chain, is deterministic, and can be turned off. A writer that
// quietly dithered would do it after the true-peak limiter had already met its
// ceiling, and would be impossible to disable — so this writer rounds, and
// nothing else.
inline void encodeSample(std::vector<uint8_t>& out, Sample v, SampleFormat format) {
  switch (format) {
    case SampleFormat::PcmInt16: {
      const int32_t q = quantise(v, 32768.0, -32768, 32767);
      detail::putU16(out, static_cast<uint16_t>(static_cast<int16_t>(q)));
      break;
    }
    case SampleFormat::PcmInt24: {
      const int32_t q = quantise(v, 8388608.0, -8388608, 8388607);
      out.push_back(static_cast<uint8_t>(q & 0xFF));
      out.push_back(static_cast<uint8_t>((q >> 8) & 0xFF));
      out.push_back(static_cast<uint8_t>((q >> 16) & 0xFF));
      break;
    }
    case SampleFormat::PcmInt32: {
      const double q = std::round(clampUnity(v) * 2147483648.0);
      const int64_t c = q <= -2147483648.0 ? -2147483648LL
                      : q >= 2147483647.0  ?  2147483647LL
                                           : static_cast<int64_t>(q);
      detail::putU32(out, static_cast<uint32_t>(static_cast<int32_t>(c)));
      break;
    }
    case SampleFormat::Float32: {
      // Float output is NOT clamped: it can legitimately exceed unity, and
      // clamping here would destroy headroom left deliberately.
      const float f = static_cast<float>(v);
      uint32_t bits;
      std::memcpy(&bits, &f, 4);
      detail::putU32(out, bits);
      break;
    }
    case SampleFormat::Float64: {
      uint64_t bits;
      std::memcpy(&bits, &v, 8);
      detail::putU64(out, bits);
      break;
    }
  }
}

inline std::vector<uint8_t> encodeWav(const AudioBuffer& audio, double sampleRate,
                                      SampleFormat format) {
  using namespace detail;

  const int channels = audio.channels();
  if (channels < 1) throw WavError("cannot write a file with no channels");

  const int bps = bytesPerSample(format);
  const uint64_t dataBytes = static_cast<uint64_t>(audio.frames()) *
                             static_cast<uint64_t>(channels) *
                             static_cast<uint64_t>(bps);

  // RIFF's size fields are 32-bit. Past 4 GB the format cannot describe itself
  // and RF64 is not optional — writing plain RIFF would produce a file whose
  // header is silently wrong rather than one that fails to open.
  const bool needRf64 = dataBytes + 512 > 0xFFFFFFFFull;

  std::vector<uint8_t> out;
  out.reserve(static_cast<size_t>(dataBytes) + 512);

  putTag(out, needRf64 ? "RF64" : "RIFF");
  const size_t riffSizePos = out.size();
  putU32(out, needRf64 ? 0xFFFFFFFFu : 0);  // patched below for RIFF
  putTag(out, "WAVE");

  if (needRf64) {
    // ds64 carries the real 64-bit sizes. It must come first, and its own size
    // is fixed at 28 for a file with no chunk-size table.
    putTag(out, "ds64");
    putU32(out, 28);
    putU64(out, 0);           // riff size, patched below
    putU64(out, dataBytes);   // data size
    putU64(out, static_cast<uint64_t>(audio.frames()));  // sample count
    putU32(out, 0);           // table length: no extra oversized chunks
  }

  // WAVE_FORMAT_EXTENSIBLE for anything above two channels or above 16 bits.
  // Plain WAVE_FORMAT_PCM with 24-bit samples or 6 channels is technically
  // out of spec and some hosts read it as garbage; extensible is what every
  // DAW writes and every DAW reads.
  const bool extensible = channels > 2 || bps > 2;
  const uint16_t baseTag =
      (format == SampleFormat::Float32 || format == SampleFormat::Float64)
          ? kFormatFloat
          : kFormatPcm;

  putTag(out, "fmt ");
  putU32(out, extensible ? 40u : 16u);
  putU16(out, extensible ? kFormatExtensible : baseTag);
  putU16(out, static_cast<uint16_t>(channels));
  putU32(out, static_cast<uint32_t>(sampleRate + 0.5));
  putU32(out, static_cast<uint32_t>((sampleRate + 0.5)) *
                  static_cast<uint32_t>(channels) * static_cast<uint32_t>(bps));
  putU16(out, static_cast<uint16_t>(channels * bps));
  putU16(out, static_cast<uint16_t>(bps * 8));
  if (extensible) {
    // The extension is exactly 22 bytes: 2 valid-bits + 4 channel mask + a
    // 16-byte subformat GUID. The GUID's first two bytes ARE the format tag,
    // so the fixed suffix is 14 bytes, not 16 — writing a spare padding word
    // between them makes the chunk 42 bytes while the header still claims 40,
    // and every chunk after it is then read at the wrong offset.
    putU16(out, 22);                                    // cbSize
    putU16(out, static_cast<uint16_t>(bps * 8));        // valid bits
    putU32(out, 0);                                     // channel mask: unspecified
    putU16(out, baseTag);                               // GUID head
    for (uint8_t b : kKsdataformatSuffix) out.push_back(b);
  }

  putTag(out, "data");
  putU32(out, needRf64 ? 0xFFFFFFFFu : static_cast<uint32_t>(dataBytes));

  // Interleave on the way out. Planar is the engine's internal layout; the
  // file format is interleaved and this is the only place that converts.
  const int64_t frames = audio.frames();
  for (int64_t i = 0; i < frames; ++i)
    for (int c = 0; c < channels; ++c)
      encodeSample(out, audio.channel(c)[i], format);

  // An odd-sized data chunk needs a pad byte to keep the file word-aligned.
  if (dataBytes & 1ull) out.push_back(0);

  const uint64_t riffSize = out.size() - 8;
  if (needRf64) {
    // Patch the ds64 riff size: 4 bytes of "RF64" + 4 size + 4 "WAVE" = 12,
    // then the ds64 header is 8 more, so the field starts at byte 20.
    for (int i = 0; i < 8; ++i)
      out[20 + static_cast<size_t>(i)] =
          static_cast<uint8_t>((riffSize >> (8 * i)) & 0xFF);
  } else {
    for (int i = 0; i < 4; ++i)
      out[riffSizePos + static_cast<size_t>(i)] =
          static_cast<uint8_t>((riffSize >> (8 * i)) & 0xFF);
  }
  return out;
}

inline void writeWav(const std::string& path, const AudioBuffer& audio,
                     double sampleRate, SampleFormat format) {
  const auto bytes = encodeWav(audio, sampleRate, format);

  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw WavError("cannot create " + path);
  const size_t wrote = std::fwrite(bytes.data(), 1, bytes.size(), f);
  const bool bad = (wrote != bytes.size()) || std::ferror(f) != 0;
  // A close that fails means buffered data never reached the disk. Ignoring
  // it is how a "successful" render produces a truncated file.
  const bool closeBad = std::fclose(f) != 0;
  if (bad || closeBad) throw WavError("write error on " + path);
}

} // namespace anv
