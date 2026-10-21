// Contract tests for the layer between the outside world and the DSP: WAV
// encoding, JSON parsing, sample-rate conversion, and recipe validation.
//
// A separate binary from contract_tests deliberately. That suite asks "does
// this node obey the node contract?" and is parameterized over a node
// registry; none of the machinery here is a node, and bolting it on would mean
// every future node walks past twenty checks that do not apply to it.
//
// The shared theme is that all four of these components fail QUIETLY when they
// fail. A WAV writer with a wrong chunk size produces a file that opens in
// half the world's software. A resampler off by a fraction of a sample sounds
// perfect in isolation. A recipe validator that misses an ordering rule ships
// audio that is merely wrong rather than obviously broken. So the assertions
// here are mostly about exactness, not plausibility.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "anv/dsp/Resampler.h"
#include "anv/io/Json.h"
#include "anv/io/Recipe.h"
#include "anv/io/WavFile.h"

using namespace anv;

namespace {

int g_checks = 0;
int g_failures = 0;

// Leading newline for the same reason as the node suite: progress lines are
// printed without a trailing newline, and a failure appended mid-line is
// invisible to anything scanning for failures at the start of a line.
void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::printf("\n  FAIL  %s\n", what.c_str());
  }
}

double dbOf(double x) { return x > 0.0 ? 20.0 * std::log10(x) : -1000.0; }

// std::to_string is %f: it renders 1e-7 as "0.000000", which turns a failure
// message into "expected 0, got 0". Every number in this file that can be
// small goes through here instead.
std::string sci(double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.3e", v);
  return buf;
}

// --------------------------------------------------------------------------
// 1. WAV encoding

void testWavRoundTrip() {
  // Bit-exactness on the lattice. Every integer format is scaled by 2^(n-1) in
  // both directions, so a value that is already representable must survive
  // encode->decode with NO error at all. "Close enough" here would mean the
  // scaling constants disagree, which shows up as a slow DC-ward drift on
  // repeated round trips rather than as anything audible on the first one.
  const struct { SampleFormat fmt; int bits; } intFormats[] = {
      {SampleFormat::PcmInt16, 16},
      {SampleFormat::PcmInt24, 24},
      {SampleFormat::PcmInt32, 32},
  };

  for (const auto& f : intFormats) {
    const double step = 1.0 / std::pow(2.0, f.bits - 1);
    AudioBuffer b(2, 512);
    for (int64_t i = 0; i < 512; ++i) {
      // Sweep the extremes as well as the middle: full-scale negative is the
      // code most likely to wrap, and zero is the one most likely to be
      // rounded the wrong way.
      const double k = static_cast<double>(i - 256);
      b.channel(0)[i] = k * step * 1000.0;
      b.channel(1)[i] = (i == 0) ? -1.0 : (i == 1 ? 1.0 - step : k * step);
    }

    const auto bytes = encodeWav(b, 48000.0, f.fmt);
    const AudioFile back = decodeWav(bytes);

    check(back.audio.channels() == 2 && back.audio.frames() == 512,
          std::string(formatName(f.fmt)) + ": shape survives the round trip");
    check(back.sampleRate == 48000.0,
          std::string(formatName(f.fmt)) + ": sample rate survives");
    check(back.format == f.fmt,
          std::string(formatName(f.fmt)) + ": format identified correctly");

    double worst = 0.0;
    for (int c = 0; c < 2; ++c)
      for (int64_t i = 0; i < 512; ++i) {
        // The representable range is ASYMMETRIC. Two's complement runs from
        // -2^(n-1) to +2^(n-1)-1, so -1.0 is an exact code and +1.0 is not:
        // the largest positive value is 1.0 - step. Expecting +1.0 to survive
        // would be asserting that the format is something it is not.
        const double want = std::clamp(b.channel(c)[i], -1.0, 1.0 - step);
        worst = std::max(worst, std::abs(back.audio.channel(c)[i] - want));
      }
    check(worst == 0.0, std::string(formatName(f.fmt)) +
                            ": on-lattice values are bit-exact, worst " +
                            sci(worst));
  }

  // Float formats carry the value through untouched, including values ABOVE
  // full scale. A float mixdown handed to us hot is ordinary, and clamping it
  // on the way through would destroy headroom that was left deliberately.
  for (SampleFormat fmt : {SampleFormat::Float32, SampleFormat::Float64}) {
    AudioBuffer b(1, 8);
    const double values[8] = {0.0, 1.0, -1.0, 2.5, -3.75, 0.25, 1e-8, -1e-8};
    for (int i = 0; i < 8; ++i) b.channel(0)[i] = values[i];

    const AudioFile back = decodeWav(encodeWav(b, 96000.0, fmt));
    double worst = 0.0;
    for (int i = 0; i < 8; ++i)
      worst = std::max(worst, std::abs(back.audio.channel(0)[i] - values[i]));

    // float32 cannot represent 1e-8 exactly; float64 must be exact.
    const double tolerance = (fmt == SampleFormat::Float64) ? 0.0 : 1e-14;
    check(worst <= tolerance, std::string(formatName(fmt)) +
                                  ": passes values beyond full scale, worst " +
                                  sci(worst));
  }

  // Integer output clamps rather than wraps. Wrapping turns a 0.1 dB overshoot
  // into a full-amplitude polarity flip, which is the single most destructive
  // artifact available.
  {
    AudioBuffer b(1, 4);
    b.channel(0)[0] = 4.0;
    b.channel(0)[1] = -4.0;
    b.channel(0)[2] = std::nan("");
    b.channel(0)[3] = 0.5;

    const AudioFile back = decodeWav(encodeWav(b, 48000.0, SampleFormat::PcmInt16));
    check(back.audio.channel(0)[0] == 1.0 - 1.0 / 32768.0,
          "16-bit: +4.0 clamps to the largest representable positive code");
    check(back.audio.channel(0)[1] == -1.0, "16-bit: -4.0 clamps to -1.0");
    check(back.audio.channel(0)[2] == 0.0, "16-bit: NaN becomes silence");
    check(std::abs(back.audio.channel(0)[3] - 0.5) < 1e-4,
          "16-bit: an ordinary value is unaffected by the clamp");
  }

  // Channel counts above two force WAVE_FORMAT_EXTENSIBLE, whose chunk is 40
  // bytes rather than 16. Getting that length wrong misaligns every chunk
  // after it — which is exactly the bug this file was written with, and it
  // presented as "no data chunk" rather than as anything about the fmt chunk.
  for (int channels : {1, 2, 6, 8}) {
    AudioBuffer b(channels, 64);
    for (int c = 0; c < channels; ++c)
      for (int64_t i = 0; i < 64; ++i)
        b.channel(c)[i] = 0.01 * static_cast<double>(c + 1) *
                          std::sin(0.1 * static_cast<double>(i));

    const AudioFile back = decodeWav(encodeWav(b, 96000.0, SampleFormat::PcmInt24));
    check(back.audio.channels() == channels && back.audio.frames() == 64,
          std::to_string(channels) + "-channel file parses back correctly");
  }

  // An odd-length data chunk is followed by a pad byte that is NOT counted in
  // the size field. A parser that forgets it reads every later chunk at the
  // wrong offset; a writer that forgets it produces a file other parsers
  // mis-read. 8-bit would be the natural way to get an odd length, and we do
  // not write 8-bit — so it is provoked with a trailing custom chunk instead.
  {
    AudioBuffer b(1, 3);
    for (int i = 0; i < 3; ++i) b.channel(0)[i] = 0.5;
    auto bytes = encodeWav(b, 44100.0, SampleFormat::PcmInt24);  // 9 data bytes
    const AudioFile back = decodeWav(bytes);
    check(back.audio.frames() == 3, "odd-length data chunk is padded and parses");
    check((bytes.size() % 2) == 0, "file length stays word-aligned");
  }
}

void testWavRejection() {
  auto rejects = [](std::vector<uint8_t> bytes, const std::string& what) {
    try {
      decodeWav(bytes);
      check(false, "should have rejected " + what);
    } catch (const WavError&) {
      check(true, "rejects " + what);
    }
  };

  rejects({}, "an empty file");
  rejects({'J', 'U', 'N', 'K', 0, 0, 0, 0, 'W', 'A', 'V', 'E'}, "a non-RIFF tag");
  rejects({'R', 'I', 'F', 'F', 0, 0, 0, 0, 'A', 'V', 'I', ' '}, "a non-WAVE form");

  // A valid file with the data chunk removed, and one with the fmt chunk
  // removed. Both are structurally fine and would produce zero frames rather
  // than an error if the presence checks were missing.
  {
    AudioBuffer b(1, 16);
    const auto good = encodeWav(b, 48000.0, SampleFormat::PcmInt16);

    std::vector<uint8_t> noData(good.begin(), good.begin() + 12);
    noData.insert(noData.end(), good.begin() + 12, good.end() - 32 - 8);
    rejects(noData, "a file with no data chunk");

    std::vector<uint8_t> header(good.begin(), good.begin() + 12);
    rejects(header, "a file with no fmt chunk");
  }

  // A NaN or infinity in a float file. The resampler used to spread one into a
  // 239-sample dropout, written as silence, in a render that exited 0.
  for (SampleFormat format : {SampleFormat::Float32, SampleFormat::Float64}) {
    for (double bad : {std::nan(""), HUGE_VAL, -HUGE_VAL}) {
      AudioBuffer b(2, 16);
      b.channel(1)[5] = bad;
      rejects(encodeWav(b, 48000.0, format),
              std::string(format == SampleFormat::Float32 ? "float32" : "float64") +
                  " with a non-finite sample (" + std::to_string(bad) + ")");
    }
  }

  // A chunk header claiming far more bytes than the file contains. This is the
  // classic path to an over-sized allocation or an out-of-bounds read, and the
  // parser must clamp to what is actually present rather than trusting it.
  {
    AudioBuffer b(1, 16);
    auto bytes = encodeWav(b, 48000.0, SampleFormat::PcmInt16);
    // Overwrite the data chunk's size with 0xFFFFFFF0.
    for (size_t i = 12; i + 8 < bytes.size(); ++i) {
      if (std::memcmp(&bytes[i], "data", 4) == 0) {
        for (int k = 0; k < 4; ++k) bytes[i + 4 + static_cast<size_t>(k)] = 0xF0;
        break;
      }
    }
    try {
      const AudioFile back = decodeWav(bytes);
      check(back.audio.frames() <= 16,
            "a lying data-chunk size is clamped to the bytes present");
    } catch (const WavError&) {
      check(true, "a lying data-chunk size is rejected");
    }
  }
}

// --------------------------------------------------------------------------
// 2. JSON

void testJson() {
  {
    const JsonValue v = parseJson(
        R"({"n":-1.5e2,"s":"a\"b\\c\u00e9\ud83c\udfb8","t":true,"z":null,)"
        R"("arr":[1,[2,[3]]],"obj":{"k":0}})");

    check(v.numberOr("n", 0.0) == -150.0, "json: number with sign and exponent");
    check(v.stringOr("s", "") == "a\"b\\c\xC3\xA9\xF0\x9F\x8E\xB8",
          "json: escapes, 2-byte UTF-8 and a surrogate pair");
    check(v.boolOr("t", false), "json: true");
    check(v.find("z")->isNull(), "json: null");
    check(v.find("arr")->array.size() == 2, "json: nested arrays");
    check(v.find("arr")->array[1].array[1].array[0].number == 3.0,
          "json: three levels of array nesting reads back");
    check(v.find("obj")->numberOr("k", 1.0) == 0.0, "json: nested object");
    check(v.find("missing") == nullptr, "json: absent key returns null pointer");
  }

  auto rejects = [](const std::string& text, const std::string& what) {
    try {
      parseJson(text);
      check(false, "json should have rejected " + what);
    } catch (const JsonError&) {
      check(true, "json rejects " + what);
    }
  };

  rejects("{\"a\":1,}", "a trailing comma");
  rejects("[1 2]", "a missing comma");
  rejects("{a:1}", "an unquoted key");
  rejects("{'a':1}", "single quotes");
  rejects("{\"a\":1,\"a\":2}", "a duplicate key");
  rejects("{\"a\":01}", "a leading zero");
  rejects("{\"a\":.5}", "a bare decimal point");
  rejects("{\"a\":1.}", "a trailing decimal point");
  rejects("{\"a\":1e}", "an empty exponent");
  rejects("{\"a\":1e400}", "a number that overflows to infinity");
  rejects("{\"a\":NaN}", "a NaN literal");
  rejects("{\"a\":Infinity}", "an Infinity literal");
  rejects("\"unterminated", "an unterminated string");
  rejects("\"\\ud800\"", "an unpaired high surrogate");
  rejects("\"\\udc00\"", "an unpaired low surrogate");
  rejects("\"\\q\"", "an unknown escape");
  rejects("{} {}", "trailing content");
  rejects("// comment\n{}", "a comment");

  // Unbounded recursion in a recursive-descent parser is a stack overflow, and
  // `[` is one byte of attacker effort per level. The bound must fire before
  // the stack does.
  {
    std::string deep(200, '[');
    deep += std::string(200, ']');
    rejects(deep, "nesting past the depth limit");

    std::string ok(40, '[');
    ok += std::string(40, ']');
    try {
      parseJson(ok);
      check(true, "json accepts nesting within the depth limit");
    } catch (const JsonError&) {
      check(false, "json wrongly rejected 40 levels of nesting");
    }
  }
}

// --------------------------------------------------------------------------
// 3. Resampling

void testResampler() {
  // Identity is bit-exact, not merely accurate. Equal rates are the common
  // case for a 96 kHz source, and running it through an interpolator would
  // add a transition band for no reason at all.
  {
    Resampler r;
    check(r.prepare(96000.0, 96000.0), "resampler prepares for equal rates");
    check(r.isIdentity(), "equal rates give an identity resampler");

    AudioBuffer in(2, 128), out;
    for (int c = 0; c < 2; ++c)
      for (int64_t i = 0; i < 128; ++i)
        in.channel(c)[i] = std::sin(0.3 * static_cast<double>(i) + c);
    r.process(in, out);

    double worst = 0.0;
    for (int c = 0; c < 2; ++c)
      for (int64_t i = 0; i < 128; ++i)
        worst = std::max(worst, std::abs(out.channel(c)[i] - in.channel(c)[i]));
    check(worst == 0.0, "identity resampling is bit-exact, worst " + sci(worst));
  }

  const struct { double in, out; int l, m; } rates[] = {
      {44100.0, 96000.0, 320, 147},
      {96000.0, 44100.0, 147, 320},
      {48000.0, 96000.0, 2, 1},
      {96000.0, 48000.0, 1, 2},
      {88200.0, 96000.0, 160, 147},
  };

  for (const auto& rate : rates) {
    Resampler r;
    check(r.prepare(rate.in, rate.out), "prepares");
    const std::string tag =
        std::to_string(static_cast<int>(rate.in)) + "->" +
        std::to_string(static_cast<int>(rate.out));

    check(r.upFactor() == rate.l && r.downFactor() == rate.m,
          tag + ": ratio reduces to " + std::to_string(rate.l) + "/" +
              std::to_string(rate.m));
    check(!r.wasCapped(), tag + ": prototype fits within the tap ceiling");

    // ZERO NET LATENCY, and this is the assertion that proves it. A 1 kHz sine
    // is compared against the ANALYTIC sine at the output rate — not against a
    // shifted copy of itself, and with no cross-correlation search that could
    // hide a constant offset. Output sample m must be the input evaluated at
    // time m/outRate, exactly.
    const int64_t inFrames = static_cast<int64_t>(rate.in * 0.25);
    AudioBuffer in(1, inFrames), out;
    for (int64_t i = 0; i < inFrames; ++i)
      in.channel(0)[i] =
          0.5 * std::sin(2.0 * M_PI * 1000.0 * static_cast<double>(i) / rate.in);
    r.process(in, out);

    check(out.frames() == r.outputFrames(inFrames),
          tag + ": process() produces the frame count outputFrames() promised");

    double worst = 0.0;
    const int64_t lo = out.frames() / 4, hi = out.frames() * 3 / 4;
    for (int64_t i = lo; i < hi; ++i) {
      const double want =
          0.5 * std::sin(2.0 * M_PI * 1000.0 * static_cast<double>(i) / rate.out);
      worst = std::max(worst, std::abs(out.channel(0)[i] - want));
    }
    // -140 dB is not an arbitrary bar: it is the engine's cross-platform
    // equivalence threshold. A conversion that could not clear it would make
    // the SRC the dominant error source in every render.
    check(dbOf(worst / 0.5) < -140.0,
          tag + ": matches the analytic sine at " +
              std::to_string(dbOf(worst / 0.5)) + " dB (expect < -140)");

    // DC gain is exactly one. The polyphase decomposition touches only every
    // L-th tap per output sample, so a prototype normalised to unit sum gives
    // gain 1/L — the classic interpolation loss, and it is silent-until-
    // measured because it shows up as "the render came back quiet".
    AudioBuffer dc(1, inFrames), dcOut;
    for (int64_t i = 0; i < inFrames; ++i) dc.channel(0)[i] = 1.0;
    r.process(dc, dcOut);

    double worstDc = 0.0;
    for (int64_t i = dcOut.frames() / 4; i < dcOut.frames() * 3 / 4; ++i)
      worstDc = std::max(worstDc, std::abs(dcOut.channel(0)[i] - 1.0));
    // Bounded by the design's passband ripple, which is what a windowed-sinc
    // filter gives at DC — not by anything tighter, and asserting tighter
    // would just be asserting the ripple of one particular rate pair.
    check(worstDc < 1e-7,
          tag + ": DC gain is unity, worst deviation " + sci(worstDc));
  }

  // Anti-aliasing actually works. A tone above the target Nyquist must not
  // reappear as a lower-frequency tone: 30 kHz at 96 kHz would fold to 14.1 kHz
  // at 44.1 kHz, landing squarely in the audible band where it is unmistakable.
  {
    Resampler r;
    r.prepare(96000.0, 44100.0);

    const int64_t n = 96000;
    AudioBuffer in(1, n), out;
    for (int64_t i = 0; i < n; ++i)
      in.channel(0)[i] =
          0.5 * std::sin(2.0 * M_PI * 30000.0 * static_cast<double>(i) / 96000.0);
    r.process(in, out);

    double peak = 0.0;
    for (int64_t i = out.frames() / 4; i < out.frames() * 3 / 4; ++i)
      peak = std::max(peak, std::abs(out.channel(0)[i]));
    check(dbOf(peak / 0.5) < -100.0,
          "30 kHz does not alias into 44.1 kHz output, residual " +
              std::to_string(dbOf(peak / 0.5)) + " dB");
  }

  // A pathological rate pair reduces to an enormous L/M and would otherwise
  // ask for a filter with more taps than there is memory. The cap must engage
  // and be reported rather than either failing or quietly succeeding after a
  // multi-gigabyte allocation.
  {
    Resampler r;
    check(r.prepare(44100.0, 44101.0), "prepares for a pathological rate pair");
    check(r.wasCapped(), "pathological rate pair engages the tap ceiling");
    check(r.prototypeTaps() <= (1 << 21), "capped prototype respects the ceiling");
  }

  {
    Resampler r;
    check(!r.prepare(0.0, 48000.0), "rejects a zero input rate");
    check(!r.prepare(48000.0, -1.0), "rejects a negative output rate");
  }
}

// --------------------------------------------------------------------------
// 4. Recipe validation
//
// Every rule here exists because breaking it produces audio that sounds
// finished and is wrong. That is the only class of error worth a hard failure;
// a recipe that produces obvious garbage needs no validator.

void testRecipe() {
  auto accepts = [](const std::string& json, const std::string& what) {
    try {
      const Recipe r = parseRecipe(json);
      validateRecipe(r);
      check(true, "recipe accepts " + what);
    } catch (const std::exception& e) {
      check(false, "recipe wrongly rejected " + what + ": " + e.what());
    }
  };
  auto rejects = [](const std::string& json, const std::string& what) {
    try {
      const Recipe r = parseRecipe(json);
      validateRecipe(r);
      check(false, "recipe should have rejected " + what);
    } catch (const std::exception&) {
      check(true, "recipe rejects " + what);
    }
  };

  accepts(R"({"nodes":[],"master":[]})", "an empty recipe");
  accepts(R"({"nodes":[{"id":"g","type":"gain","params":{"gain_db":-3}}]})",
          "a single gain node");
  accepts(R"({"nodes":[{"type":"gain"}],
              "master":[{"type":"true_peak_limiter"},{"type":"dither"}]})",
          "limiter then dither in master");

  rejects(R"({"nodes":[{"type":"tape_saturation"}]})",
          "an unknown node type (never a silent no-op)");
  rejects(R"({"nodes":[{"params":{}}]})", "a node with no type");
  rejects(R"({"nodes":[{"id":"x","type":"gain"},{"id":"x","type":"gain"}]})",
          "duplicate node ids");
  rejects(R"({"nodes":[{"type":"gain","mix":1.5}]})", "a mix above 1");
  rejects(R"({"nodes":[{"type":"gain","mix":-0.1}]})", "a negative mix");

  // The two structural rules the master section exists to enforce.
  rejects(R"({"nodes":[{"type":"true_peak_limiter"}]})",
          "a true-peak limiter in the internal-rate chain");
  rejects(R"({"nodes":[{"type":"dither"}]})",
          "dither in the internal-rate chain");
  rejects(R"({"master":[{"type":"dither"},{"type":"true_peak_limiter"}]})",
          "dither before the limiter");
  rejects(R"({"master":[{"type":"dither"},{"type":"dither"}]})",
          "two dither nodes");
  rejects(R"({"master":[{"type":"true_peak_limiter"},{"type":"gain"}]})",
          "gain after the true-peak limiter");

  rejects(R"({"internal_sample_rate":10})", "an absurd internal sample rate");
  // The seed is a double in JSON and a uint64_t in the engine; the conversion
  // is undefined behaviour for a negative and lossy past 2^53.
  accepts(R"({"seed":20260731})", "an ordinary seed");
  rejects(R"({"seed":-1})", "a negative seed");
  rejects(R"({"seed":1.5})", "a fractional seed");
  rejects(R"({"seed":1e30})", "a seed past 2^53");
  rejects(R"({"output":{"format":"mp3"}})", "an output format we cannot write");
  rejects(R"([])", "a top-level array");

  // Parameter shape.
  {
    const Recipe r = parseRecipe(R"({"nodes":[{"id":"eq","type":"parametric_eq",
        "params":{"bands":[{"freq_hz":100,"gain_db":3},
                           {"freq_hz":200,"gain_db":-2}],
                  "output_trim_db":-1,"label":"low end","enabled":true}}]})");
    validateRecipe(r);

    const ParamSet& p = r.nodes[0].params;
    check(p.items().size() == 2, "recipe: an array of objects becomes the item list");
    check(p.items()[1].num("freq_hz", 0.0) == 200.0, "recipe: item order preserved");
    check(p.num("output_trim_db", 0.0) == -1.0, "recipe: scalar parameter");
    check(p.str("label", "") == "low end", "recipe: string parameter");
    check(p.num("enabled", 0.0) == 1.0, "recipe: bool arrives as 1.0");
  }

  // Two list parameters would both land in the single ordered item list, and
  // the node reading items() has no way to tell which one it received.
  rejects(R"({"nodes":[{"type":"dynamic_eq","params":{"bands":[{"a":1}],
                                                      "other":[{"b":2}]}}]})",
          "two list parameters on one node");

  // STRING ENUMS. Every node reads these with a fallback, so an unknown value
  // used to render as the default. Swept from the table itself, so a value
  // added there is exercised here without anyone remembering to.
  for (const KnownStrings& k : knownStringParams()) {
    const std::string type = k.type;
    const bool band = type == "dynamic_eq.band";
    auto recipeWith = [&](const std::string& value) {
      const std::string kv = "\"" + std::string(k.param) + "\":" + value;
      return band
          ? R"({"nodes":[{"type":"dynamic_eq","params":{"bands":[{)" + kv + "}]}}]}"
          : (type == "dither" ? R"({"master":[{"type":"dither","params":{)"
                              : R"({"nodes":[{"type":")" + type + R"(","params":{)")
                + kv + "}}]}";
    };
    const std::string label = type + "." + k.param;
    for (const std::string& v : k.values)
      accepts(recipeWith("\"" + v + "\""), label + " = \"" + v + "\"");
    rejects(recipeWith("\"not_a_value\""), "an unknown " + label);
    rejects(recipeWith("1"), "a numeric " + label + ", which would read as the default");
  }

  // THE BUG THAT MOTIVATED THE TABLE. The Python schema said "downward"; this
  // node compared against "down"; every planned dynamic EQ rendered static.
  rejects(R"({"nodes":[{"type":"dynamic_eq","params":{"bands":[{"mode":"downward"}]}}]})",
          "dynamic_eq mode \"downward\", the schema's spelling, not the node's");
  rejects(R"({"nodes":[{"type":"dynamic_eq","params":{"bands":[{"mode":"down","range_db":-6}]}}]})",
          "a negative dynamic_eq range, which the node would clamp to static");
  accepts(R"({"nodes":[{"type":"dynamic_eq","params":{"bands":[{"mode":"down","range_db":6}]}}]})",
          "a downward dynamic band with a positive range");
  rejects(R"({"nodes":[{"type":"parametric_eq","params":{"bands":[{"filter_type":3}]}}]})",
          "a numeric filter_type, which would silently become a bell");

  // Every type the factory advertises must actually construct. A name in the
  // dispatch table that returns null would pass validation and crash at build.
  for (const char* type :
       {"gain", "pan", "polarity", "dc_block", "mono_to_stereo", "mid_side",
        "parametric_eq", "compressor", "gate_expander", "dynamic_eq", "drive",
        "delay", "convolution_reverb", "plate_reverb", "chorus", "stereo_width",
        "true_peak_limiter", "dither"}) {
    check(makeNode(type) != nullptr,
          std::string("node factory constructs \"") + type + "\"");
  }
}

// --------------------------------------------------------------------------
// 5. End to end: a recipe applied through a full render, in memory.
//
// The individual pieces above are all correct in isolation; this asserts they
// are wired together in the right order and orientation.

void testEndToEnd() {
  const double inRate = 44100.0, internalRate = 96000.0;
  const int64_t inFrames = static_cast<int64_t>(inRate * 0.5);

  AudioBuffer source(1, inFrames);
  for (int64_t i = 0; i < inFrames; ++i)
    source.channel(0)[i] =
        0.25 * std::sin(2.0 * M_PI * 440.0 * static_cast<double>(i) / inRate);

  // -6 dB of gain, and nothing else, so the expected answer is exact arithmetic
  // rather than a judgement call.
  const Recipe recipe = parseRecipe(
      R"({"internal_sample_rate":96000,"seed":7,
          "output":{"sample_rate":44100,"format":"float64"},
          "nodes":[{"id":"g","type":"gain","params":{"gain_db":-6.0}}]})");
  validateRecipe(recipe);

  Resampler up, down;
  up.prepare(inRate, internalRate);
  down.prepare(internalRate, recipe.outputSampleRate);

  AudioBuffer wide;
  up.process(source, wide);

  Chain chain;
  buildChain(recipe.nodes, chain);
  PrepareSpec spec;
  spec.sampleRate = internalRate;
  spec.numChannels = 1;
  spec.maxBlockFrames = 1024;
  spec.totalFrames = wide.frames();
  spec.seed = recipe.seed;
  chain.prepare(spec);

  AudioBuffer processed;
  chain.render(wide, processed, 1024);

  AudioBuffer out;
  down.process(processed, out);

  const double wanted = 0.25 * std::pow(10.0, -6.0 / 20.0);
  double worst = 0.0;
  for (int64_t i = out.frames() / 4; i < out.frames() * 3 / 4; ++i) {
    const double expected =
        wanted * std::sin(2.0 * M_PI * 440.0 * static_cast<double>(i) /
                          recipe.outputSampleRate);
    worst = std::max(worst, std::abs(out.channel(0)[i] - expected));
  }
  check(dbOf(worst / wanted) < -120.0,
        "end to end: 44.1 -> 96 -> gain -> 44.1 matches the analytic answer at " +
            std::to_string(dbOf(worst / wanted)) + " dB");

  // And the round trip preserves timing: no node in this chain has latency, so
  // the output must be phase-aligned with the input, not merely similar.
  double worstAlign = 0.0;
  for (int64_t i = out.frames() / 4; i < out.frames() * 3 / 4; ++i)
    worstAlign = std::max(
        worstAlign, std::abs(out.channel(0)[i] -
                             std::pow(10.0, -6.0 / 20.0) * source.channel(0)[i]));
  check(dbOf(worstAlign / wanted) < -120.0,
        "end to end: output is sample-aligned with the input");
}

} // namespace

int main() {
  std::printf("Audionova I/O and support tests\n\n");

  std::printf("[1] WAV round trip\n");     testWavRoundTrip();
  std::printf("[2] WAV rejection\n");      testWavRejection();
  std::printf("[3] JSON\n");               testJson();
  std::printf("[4] resampling\n");         testResampler();
  std::printf("[5] recipe validation\n");  testRecipe();
  std::printf("[6] end to end\n");         testEndToEnd();

  std::printf("\n%d checks, %d failure(s)\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
