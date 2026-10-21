// anv_render — apply a recipe to an audio file.
//
//   anv_render --in guitar.wav --out processed.wav --recipe chain.json
//
// This is the whole render path in one place, and the order of the six steps
// is load-bearing:
//
//   1. read              the file, as float64
//   2. resample in       once, to the recipe's internal rate
//   3. render nodes[]    at the internal rate
//   4. resample out      once, to the delivery rate
//   5. render master[]   at the delivery rate (limiter, then dither)
//   6. write
//
// Steps 2 and 4 are the ONLY conversions. Every extra one is another
// transition band and another set of passband ripples, and unlike most DSP
// errors they compound silently — a chain that converts five times measures
// fine at every individual stage.
//
// The channel count is decided before step 3 and never changes inside a chain.
// A node that wants to widen mono to stereo declares that by being a
// mono_to_stereo node, so the decision is visible in the recipe rather than
// emergent from whichever node happened to run first.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "anv/core/Denormals.h"
#include "anv/dsp/Resampler.h"
#include "anv/graph/Chain.h"
#include "anv/io/Recipe.h"
#include "anv/io/WavFile.h"

namespace {

struct Options {
  std::string inPath, outPath, recipePath;
  int blockFrames = 4096;
  bool verbose = false;
};

void usage() {
  std::fprintf(stderr,
      "anv_render — apply an Audionova recipe to an audio file\n"
      "\n"
      "  anv_render --in <input.wav> --out <output.wav> --recipe <recipe.json>\n"
      "\n"
      "  --block <n>   render block size in frames (default 4096). Output is\n"
      "                identical at any value; this only affects memory use.\n"
      "  --quiet       suppress the progress report on stderr\n");
}

bool parseArgs(int argc, char** argv, Options& o) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* what) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "anv_render: %s needs a value\n", what);
        return {};
      }
      return argv[++i];
    };
    if (a == "--in")          o.inPath = next("--in");
    else if (a == "--out")    o.outPath = next("--out");
    else if (a == "--recipe") o.recipePath = next("--recipe");
    else if (a == "--block") {
      // strtol with an end pointer, not atoi: "12abc" was silently 12.
      const std::string v = next("--block");
      char* end = nullptr;
      const long n = std::strtol(v.c_str(), &end, 10);
      o.blockFrames = (!v.empty() && *end == '\0' && n > 0 && n <= (1 << 20))
                          ? static_cast<int>(n) : 0;
    }
    else if (a == "--quiet")  o.verbose = false;
    else if (a == "-h" || a == "--help") { usage(); return false; }
    else {
      std::fprintf(stderr, "anv_render: unknown argument '%s'\n", a.c_str());
      return false;
    }
  }
  if (o.inPath.empty() || o.outPath.empty() || o.recipePath.empty()) {
    usage();
    return false;
  }
  if (o.blockFrames < 1 || o.blockFrames > (1 << 20)) {
    std::fprintf(stderr, "anv_render: --block out of range\n");
    return false;
  }
  return true;
}

std::string readTextFile(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw anv::RecipeError("cannot open " + path);
  std::string out;
  char buf[16384];
  size_t got;
  while ((got = std::fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, got);
  const bool bad = std::ferror(f) != 0;
  std::fclose(f);
  if (bad) throw anv::RecipeError("read error on " + path);
  return out;
}

double peakDbfs(const anv::AudioBuffer& b) {
  double peak = 0.0;
  for (int c = 0; c < b.channels(); ++c)
    for (int64_t i = 0; i < b.frames(); ++i)
      peak = std::max(peak, std::abs(b.channel(c)[i]));
  return peak > 0.0 ? 20.0 * std::log10(peak) : -1000.0;
}

void runChain(const std::vector<anv::RecipeNode>& list, anv::AudioBuffer& audio,
              double sampleRate, uint64_t seed, int blockFrames) {
  if (list.empty()) return;

  anv::Chain chain;
  anv::buildChain(list, chain);

  anv::PrepareSpec spec;
  spec.sampleRate = sampleRate;
  spec.numChannels = audio.channels();
  spec.maxBlockFrames = blockFrames;
  spec.totalFrames = audio.frames();
  spec.seed = seed;
  chain.prepare(spec);

  anv::AudioBuffer out;
  chain.render(audio, out, blockFrames);
  audio = std::move(out);
}

} // namespace

int main(int argc, char** argv) {
  Options opt;
  opt.verbose = true;
  if (!parseArgs(argc, argv, opt)) return 2;

  // FTZ/DAZ for the whole render. This CHANGES NUMERIC RESULTS — denormals
  // flush to zero rather than being computed — so it is set here, once, for
  // the entire process, exactly as the test harness sets it. A render that
  // ran with different denormal handling than the tests is not the render the
  // tests validated.
  anv::ScopedNoDenormals denormalGuard;

  try {
    const anv::Recipe recipe = anv::parseRecipe(readTextFile(opt.recipePath));
    anv::validateRecipe(recipe);

    anv::AudioFile input = anv::readWav(opt.inPath);
    // The same range the recipe's own rates are held to. An absurd rate in a
    // WAV header is undefined behaviour in the resampler's ratio arithmetic,
    // and 1 Hz "rendered successfully" to a 1 Hz file.
    if (!(input.sampleRate >= 8000.0 && input.sampleRate <= 768000.0))
      throw anv::RecipeError("input sample rate " +
                             std::to_string(input.sampleRate) +
                             " Hz is outside 8000 to 768000 Hz");
    const double outRate = recipe.outputSampleRate != 0.0 ? recipe.outputSampleRate
                                                          : input.sampleRate;

    if (opt.verbose)
      std::fprintf(stderr,
          "in   %s: %d ch, %lld frames, %.0f Hz, %s (peak %.2f dBFS)\n",
          opt.inPath.c_str(), input.audio.channels(),
          (long long)input.audio.frames(), input.sampleRate,
          anv::formatName(input.format), peakDbfs(input.audio));

    // --- 2. one conversion in ---------------------------------------------
    anv::AudioBuffer audio;
    {
      anv::Resampler up;
      if (!up.prepare(input.sampleRate, recipe.internalSampleRate))
        throw anv::RecipeError("cannot resample from the input sample rate");
      if (opt.verbose && !up.isIdentity())
        std::fprintf(stderr, "src  %.0f -> %.0f Hz (%d/%d, %d taps)\n",
                     input.sampleRate, recipe.internalSampleRate,
                     up.upFactor(), up.downFactor(), up.prototypeTaps());
      up.process(input.audio, audio);
    }

    // --- 3. the chain, at the internal rate --------------------------------
    runChain(recipe.nodes, audio, recipe.internalSampleRate, recipe.seed,
             opt.blockFrames);

    // --- 4. one conversion out ---------------------------------------------
    {
      anv::Resampler down;
      if (!down.prepare(recipe.internalSampleRate, outRate))
        throw anv::RecipeError("cannot resample to the output sample rate");
      if (opt.verbose && !down.isIdentity())
        std::fprintf(stderr, "src  %.0f -> %.0f Hz (%d/%d, %d taps)\n",
                     recipe.internalSampleRate, outRate, down.upFactor(),
                     down.downFactor(), down.prototypeTaps());
      anv::AudioBuffer converted;
      down.process(audio, converted);
      audio = std::move(converted);
    }

    // --- 5. master, at the delivery rate ------------------------------------
    //
    // The seed is offset so the master chain's dither does not reuse the same
    // noise sequence as anything in the main chain. Derived from the recipe
    // seed rather than a clock, so the render stays reproducible.
    runChain(recipe.master, audio, outRate, recipe.seed ^ 0x9E3779B97F4A7C15ull,
             opt.blockFrames);

    // --- 6. write ------------------------------------------------------------
    anv::writeWav(opt.outPath, audio, outRate, recipe.outputFormat);

    if (opt.verbose) {
      std::fprintf(stderr,
          "out  %s: %d ch, %lld frames, %.0f Hz, %s (peak %.2f dBFS)\n",
          opt.outPath.c_str(), audio.channels(), (long long)audio.frames(),
          outRate, anv::formatName(recipe.outputFormat), peakDbfs(audio));

      // The output is LONGER than the input whenever the chain has a tail, and
      // that is intended: truncating a reverb at the last input sample cuts the
      // decay off mid-air. Reported explicitly because a file that silently
      // changed length would otherwise look like a bug.
      const int64_t expected = static_cast<int64_t>(std::llround(
          static_cast<double>(input.audio.frames()) * (outRate / input.sampleRate)));
      const int64_t tail = audio.frames() - expected;
      if (tail > 1)
        std::fprintf(stderr, "tail %lld frames (%.2f s) of decay past the input\n",
                     (long long)tail, static_cast<double>(tail) / outRate);
    }
    return 0;

  } catch (const std::exception& e) {
    std::fprintf(stderr, "anv_render: %s\n", e.what());
    return 1;
  }
}
