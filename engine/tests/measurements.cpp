// Per-commit DSP measurement suite.
//
// The contract tests answer "is this node still the same program?". This
// answers a different question: "what does this node actually DO to a signal,
// in numbers an engineer would recognise?" — THD, aliasing, intermodulation,
// frequency and phase response, and true-peak conformance.
//
// It emits two artefacts per run:
//
//   build/measurements.html   for a person
//   build/measurements.json   for ci/check_measurements.py, which compares
//                             against tests/measurements.baseline.json and
//                             fails the build on a regression
//
// FIVE DESIGN DECISIONS THAT ARE LOAD-BEARING:
//
// 1. COHERENT SAMPLING EVERYWHERE, NEVER A WINDOW.
//    Every single-tone measurement picks its frequency as an exact integer
//    number of cycles per FFT window (f0 = k*fs/N). The DFT of such a tone is
//    a single non-zero bin — zero leakage, exactly, not "40 dB down". That is
//    what makes harmonic levels EXACT rather than window-shape-dependent, and
//    it is the only reason a -100 dB threshold is meaningful at all: a
//    Hann-windowed measurement of the same signal has skirts at -31 dB/octave
//    and could not distinguish a clean node from a broken one at that level.
//    The harness self-validates this (see selfValidation) before trusting a
//    single number below.
//
// 2. NONLINEARITY IS MEASURED, NOT DECLARED.
//    Which nodes must meet the aliasing requirement is decided by a
//    homogeneity probe — scale the input by 2, does the output scale by 2? —
//    rather than by a hand-maintained list. A list goes stale the first time
//    somebody adds a saturator; a measurement cannot. Chorus is time-varying
//    but perfectly homogeneous, and is correctly excluded; dither is
//    homogeneous nowhere, and is correctly included.
//
// 3. THE NODE LIST COMES FROM THE ENGINE, NOT FROM THIS FILE.
//    See nodeCatalog(). Adding a node to Recipe.h's makeNode() enrols it here
//    automatically, and adding a header under include/anv/nodes/ without
//    wiring it into makeNode() is a hard error. Remembering to update a list
//    is not a mechanism.
//
// 4. EVERY COMPARISON IS AGAINST A NULL RENDER.
//    The frequency response of a node is its impulse response deconvolved and
//    divided by the SAME chain with the node bypassed — never by the raw
//    stimulus. That cancels the harness's own inverse filter, the chain's
//    delay compensation and the buffer edges, so what is left is the node.
//
// 5. A MISSING MEASUREMENT IS null, NEVER A PLAUSIBLE CONSTANT.
//    Every number below is either measured or absent. There is no "assume
//    -120 dB when the bin is empty" anywhere, because a fabricated number is
//    indistinguishable from a measurement once it is in the report.
//
// THE TRUE-PEAK REFERENCE IS DELIBERATELY NOT THE ENGINE'S.
// ReferenceTruePeak below shares no code with anv::TruePeakUpsampler or
// anv::FirDesign — different window, different length, different factor,
// written from the definition. It is then validated against the ANALYTIC
// answer (a sampled unit sinusoid reconstructs to exactly 0 dBTP at every
// frequency and phase), because agreeing with yourself proves nothing and
// two approximations agreeing proves only that they are wrong together.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "anv/core/AudioBuffer.h"
#include "anv/core/Denormals.h"
#include "anv/core/Rng.h"
#include "anv/dsp/Fft.h"
#include "anv/dsp/TruePeak.h"
#include "anv/graph/Chain.h"
#include "anv/io/Recipe.h"

using namespace anv;

namespace {

// ------------------------------------------------------------- the report --

// A number that may not exist, and a number whose logarithm does not exist.
// Both come up constantly here and they are NOT the same thing:
//
//   present == false     not measured. Serialises as JSON null.
//   negInf  == true      measured, and the ratio was EXACTLY zero — a
//                        bit-identical null. Its decibel value does not exist.
//
// The second state earns its complexity. Three linear nodes (polarity,
// mono_to_stereo, stereo_width) null against themselves to the last bit, an
// earlier version of this file discarded that as "unmeasurable", and all three
// were then misclassified as nonlinear. Substituting -320 dB instead would
// have been the `linToDb` clamp bug from findings.md, where a real -313 dB null
// read -240 because a helper said so. Exactly zero is a measurement; it is
// simply not a decibel.
struct Value {
  bool present = false;
  bool negInf = false;
  double v = 0.0; // -infinity when negInf, so comparisons need no special case

  static Value of(double x) {
    Value r;
    if (std::isinf(x) && x < 0.0) {
      r.present = true;
      r.negInf = true;
      r.v = x;
      return r;
    }
    r.present = std::isfinite(x);
    r.v = x;
    return r;
  }
  static Value none() { return Value{}; }
  bool usable() const { return present; }
};

struct Metric {
  std::string key;      // stable identifier; the baseline is keyed on it
  std::string label;    // column heading
  std::string unit;     // "dB", "%", "deg", "samples", "dBTP", "dBFS", "Hz", ""
  Value value;
  bool applicable = true;
  std::string naReason;
  bool hasThreshold = false;
  double threshold = 0.0;
  bool lowerIsBetter = true; // direction a regression moves in
  bool pass = true;
  std::string note;
};

struct ConfigReport {
  std::string configId;  // unique: node type, or type + variant
  std::string nodeType;
  std::string paramSummary;
  bool linearityKnown = false;
  bool linear = true;
  std::vector<Metric> metrics;

  const Metric* find(const std::string& key) const {
    for (const auto& m : metrics)
      if (m.key == key) return &m;
    return nullptr;
  }
  void add(Metric m) { metrics.push_back(std::move(m)); }
};

Metric plain(std::string key, std::string label, std::string unit, Value v) {
  Metric m;
  m.key = std::move(key);
  m.label = std::move(label);
  m.unit = std::move(unit);
  m.value = v;
  return m;
}

Metric gated(std::string key, std::string label, std::string unit, Value v,
             double threshold, bool lowerIsBetter = true) {
  Metric m = plain(std::move(key), std::move(label), std::move(unit), v);
  m.hasThreshold = true;
  m.threshold = threshold;
  m.lowerIsBetter = lowerIsBetter;
  // An unmeasured value cannot pass. Silence is not conformance.
  m.pass = v.usable() && (lowerIsBetter ? v.v <= threshold : v.v >= threshold);
  return m;
}

Metric notApplicable(std::string key, std::string label, std::string unit,
                     std::string why) {
  Metric m = plain(std::move(key), std::move(label), std::move(unit),
                   Value::none());
  m.applicable = false;
  m.naReason = std::move(why);
  return m;
}

// ------------------------------------------------------------ the catalog --
//
// The engine's node type set lives in exactly one place — Recipe.h's
// makeNode(), which is a closed set precisely so an unknown type is a loud
// error rather than a silent no-op. This suite reads that same list rather
// than keeping a second copy, so a node added there is measured here without
// anybody remembering to do anything.
//
// Reading a header at run time is unusual, and the alternative was worse: a
// hand-maintained list in this file would have exactly the failure mode the
// suite exists to prevent — a node that quietly is not measured, in a report
// that looks complete. The parse is defended by two hard errors: every type it
// finds must construct, and the number of types must account for every header
// under include/anv/nodes/.

std::string g_engineDir;

std::string readWholeFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return {};
  std::string text((std::istreambuf_iterator<char>(in)),
                   std::istreambuf_iterator<char>());
  return text;
}

// Node headers that exist but are deliberately not reachable from a recipe.
// Each needs a reason, because "it is not in the list" is how a node stops
// being tested.
const std::map<std::string, std::string>& catalogExemptions() {
  static const std::map<std::string, std::string> kExempt = {
      {"ReferenceDelay.h",
       "test scaffolding: a pure integer delay that exists so the bypass-null "
       "and PDC tests are not vacuous. Never reachable from a recipe."},
  };
  return kExempt;
}

std::vector<std::string> nodeCatalog(std::string& errorOut) {
  const std::string recipePath = g_engineDir + "/include/anv/io/Recipe.h";
  const std::string text = readWholeFile(recipePath);
  if (text.empty()) {
    errorOut = "cannot read " + recipePath +
               " — the node catalog is derived from it, so no measurement in "
               "this run would be known to be complete";
    return {};
  }

  const size_t begin = text.find("makeNode(const std::string& type)");
  const size_t end = begin == std::string::npos
                         ? std::string::npos
                         : text.find("return nullptr;", begin);
  if (begin == std::string::npos || end == std::string::npos) {
    errorOut = "makeNode() not found in " + recipePath +
               " in the shape this parser expects";
    return {};
  }

  std::vector<std::string> types;
  const std::string needle = "type == \"";
  for (size_t at = text.find(needle, begin);
       at != std::string::npos && at < end; at = text.find(needle, at + 1)) {
    const size_t from = at + needle.size();
    const size_t to = text.find('"', from);
    if (to == std::string::npos || to > end) break;
    types.push_back(text.substr(from, to - from));
  }

  // Defence 1: every parsed name must actually build a node. A typo in the
  // parse would otherwise silently shrink the catalog.
  for (const auto& t : types) {
    if (!makeNode(t)) {
      errorOut = "parsed node type \"" + t + "\" from makeNode() but " +
                 "makeNode() does not construct it — the parser is wrong";
      return {};
    }
  }

  // Defence 2: account for every header under include/anv/nodes/. Dropping a
  // new node header in without wiring it into makeNode() fails here rather
  // than quietly shipping an unmeasured node.
  std::vector<std::string> headers;
  const std::string nodesDir = g_engineDir + "/include/anv/nodes";
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(nodesDir, ec)) {
    if (entry.path().extension() == ".h")
      headers.push_back(entry.path().filename().string());
  }
  if (ec || headers.empty()) {
    errorOut = "cannot list " + nodesDir;
    return {};
  }
  size_t exempt = 0;
  for (const auto& h : headers)
    if (catalogExemptions().count(h)) ++exempt;
  if (headers.size() - exempt != types.size()) {
    errorOut = "include/anv/nodes/ holds " + std::to_string(headers.size()) +
               " headers (" + std::to_string(exempt) +
               " exempt) but makeNode() names " + std::to_string(types.size()) +
               " types — a node header exists that no recipe can reach, so it "
               "would never be measured";
    return {};
  }
  return types;
}

// ------------------------------------------------------ measured configs ---
//
// Parameters mirror the contract-test registry: deliberately non-trivial,
// because a node measured at unity proves nothing. They are restated here
// rather than shared, so the two suites can change their stimulus independently
// — the contract tests want awkward latencies, this one wants audible settings.
//
// A node with no entry here is measured at ITS OWN DEFAULTS, which is what
// makes a newly added node measured rather than skipped.

struct Config {
  std::string id;
  std::string type;
  std::string summary;
  ParamSet params;
};

std::map<std::string, std::pair<std::string, ParamSet>> configOverrides() {
  std::map<std::string, std::pair<std::string, ParamSet>> m;

  {
    ParamSet p;
    p.set("gain_db", -4.5);
    m["gain"] = {"gain_db -4.5", p};
  }
  {
    ParamSet p;
    p.set("hp_freq_hz", 80.0).set("hp_slope_db_oct", 24.0);
    ParamSet b1; b1.setStr("filter_type", "bell").set("center_freq_hz", 250.0)
                   .set("gain_db", -3.5).set("q_factor", 1.1);
    ParamSet b2; b2.setStr("filter_type", "bell").set("center_freq_hz", 3200.0)
                   .set("gain_db", 2.0).set("q_factor", 0.8);
    ParamSet b3; b3.setStr("filter_type", "highshelf").set("center_freq_hz", 9000.0)
                   .set("gain_db", 1.5).set("q_factor", 0.7);
    p.addItem(b1).addItem(b2).addItem(b3);
    m["parametric_eq"] = {"HP 80 Hz 24 dB/oct + 3 bands", p};
  }
  {
    ParamSet p;
    ParamSet b1; b1.setStr("filter_type", "bell").set("center_freq_hz", 3200.0)
                   .set("gain_db", 2.0).set("q_factor", 0.8);
    ParamSet b2; b2.setStr("filter_type", "bell").set("center_freq_hz", 6500.0)
                   .set("gain_db", -3.0).set("q_factor", 2.0);
    ParamSet b3; b3.setStr("filter_type", "highshelf").set("center_freq_hz", 9000.0)
                   .set("gain_db", 1.5).set("q_factor", 0.7);
    p.addItem(b1).addItem(b2).addItem(b3);
    m["zero_phase_eq"] = {"3 bands, >3 kHz (magnitude is squared)", p};
  }
  {
    ParamSet p;
    p.set("taps", 511.0);
    ParamSet b1; b1.setStr("filter_type", "bell").set("center_freq_hz", 3200.0)
                   .set("gain_db", 2.0).set("q_factor", 0.8);
    ParamSet b2; b2.setStr("filter_type", "highshelf").set("center_freq_hz", 9000.0)
                   .set("gain_db", 1.5).set("q_factor", 0.7);
    p.addItem(b1).addItem(b2);
    m["linear_phase_eq"] = {"511 taps, 2 bands", p};
  }
  {
    ParamSet p;
    p.set("drive_db", 12.0).set("output_trim_db", -6.0).set("bias", 0.15)
        .set("oversample_factor", 4.0);
    p.setStr("curve", "soft_clip");
    m["drive"] = {"soft clip, +12 dB drive, bias 0.15, 4x", p};
  }
  {
    ParamSet p;
    p.set("threshold_dbfs", -20.0).set("ratio_to_1", 4.0).set("knee_db", 6.0)
        .set("attack_ms", 5.0).set("release_ms", 150.0).set("lookahead_ms", 3.0)
        .set("sidechain_hp_hz", 90.0).set("stereo_link", 0.8)
        .set("makeup_gain_db", 3.0);
    p.setStr("detector", "peak");
    p.setStr("release_mode", "program");
    m["compressor"] = {"-20 dBFS, 4:1, 5/150 ms, 3 ms lookahead", p};
  }
  {
    ParamSet p;
    p.set("lookahead_ms", 5.0).set("stereo_link", 1.0)
        .set("crossover_stopband_db", 80.0);
    ParamSet b1; b1.set("crossover_hz", 200.0).set("threshold_dbfs", -22.0)
                   .set("ratio_to_1", 3.0).set("knee_db", 6.0)
                   .set("attack_ms", 15.0).set("release_ms", 200.0);
    ParamSet b2; b2.set("crossover_hz", 2500.0).set("threshold_dbfs", -18.0)
                   .set("ratio_to_1", 2.5).set("knee_db", 6.0)
                   .set("attack_ms", 8.0).set("release_ms", 120.0);
    ParamSet b3; b3.set("threshold_dbfs", -24.0).set("ratio_to_1", 4.0)
                   .set("knee_db", 4.0).set("attack_ms", 3.0)
                   .set("release_ms", 90.0).set("makeup_gain_db", 1.0);
    p.addItem(b1).addItem(b2).addItem(b3);
    m["multiband_compressor"] = {"3 bands @ 200 Hz / 2.5 kHz, 2.5-4:1", p};
  }
  {
    ParamSet p;
    p.set("attack_db", 6.0).set("sustain_db", -4.0).set("attack_ms", 20.0)
        .set("sustain_ms", 250.0).set("lookahead_ms", 2.0).set("stereo_link", 1.0);
    m["transient_designer"] = {"attack +6 dB, sustain -4 dB", p};
  }
  {
    ParamSet p;
    p.set("threshold_dbfs", -35.0).set("ratio_to_1", 6.0).set("knee_db", 3.0)
        .set("range_db", 45.0).set("attack_ms", 1.0).set("hold_ms", 30.0)
        .set("release_ms", 120.0).set("lookahead_ms", 2.0)
        .set("hysteresis_db", 4.0).set("sidechain_hp_hz", 120.0)
        .set("stereo_link", 1.0);
    p.setStr("detector", "peak");
    m["gate_expander"] = {"-35 dBFS, 6:1, 45 dB range", p};
  }
  {
    ParamSet p;
    ParamSet d1; d1.setStr("filter_type", "bell").setStr("mode", "down")
                   .set("center_freq_hz", 6500.0).set("q_factor", 2.5)
                   .set("threshold_dbfs", -30.0).set("ratio_to_1", 5.0)
                   .set("knee_db", 4.0).set("range_db", 10.0)
                   .set("attack_ms", 2.0).set("release_ms", 60.0);
    ParamSet d2; d2.setStr("filter_type", "bell").setStr("mode", "static")
                   .set("center_freq_hz", 320.0).set("q_factor", 1.2)
                   .set("static_gain_db", -2.5);
    ParamSet d3; d3.setStr("filter_type", "highshelf").setStr("mode", "up")
                   .set("center_freq_hz", 9000.0).set("threshold_dbfs", -40.0)
                   .set("ratio_to_1", 3.0).set("knee_db", 2.0)
                   .set("range_db", 6.0).set("attack_ms", 10.0)
                   .set("release_ms", 120.0);
    p.addItem(d1).addItem(d2).addItem(d3);
    m["dynamic_eq"] = {"de-esser + static bell + upward shelf", p};
  }
  {
    ParamSet p;
    p.set("ceiling_dbtp", -1.0).set("lookahead_ms", 2.0).set("release_ms", 100.0)
        .set("oversample_factor", 8.0);
    m["true_peak_limiter"] = {"-1 dBTP ceiling, 2 ms lookahead, 8x", p};
  }
  {
    ParamSet p;
    p.set("decay_s", 0.35).set("pre_delay_ms", 0.0).set("hf_damp_hz", 5000.0)
        .set("lf_cut_hz", 150.0).set("hf_decay_ratio", 0.5)
        .set("stereo_decorrelation", 1.0);
    m["convolution_reverb"] = {"0.35 s decay, full decorrelation", p};
  }
  {
    ParamSet p;
    p.set("pan_position", -0.4);
    m["pan"] = {"pan -0.4", p};
  }
  {
    ParamSet p;
    p.set("inverted", 1.0);
    m["polarity"] = {"inverted", p};
  }
  {
    ParamSet p;
    p.set("cutoff_hz", 10.0);
    m["dc_block"] = {"10 Hz", p};
  }
  {
    ParamSet p;
    p.set("source_channel", 0.0);
    m["mono_to_stereo"] = {"from channel 0", p};
  }
  {
    ParamSet p;
    p.setStr("mode", "encode");
    m["mid_side"] = {"encode", p};
  }
  {
    ParamSet p;
    p.set("bit_depth", 16.0).set("amount_lsb", 2.0);
    p.setStr("noise_shaping", "none");
    m["dither"] = {"16-bit, 2 LSB TPDF, no shaping", p};
  }
  {
    ParamSet p;
    p.set("decay_s", 0.4).set("pre_delay_ms", 0.0).set("damping_hz", 6000.0)
        .set("bandwidth_hz", 12000.0).set("size", 1.0)
        .set("modulation_depth", 0.6);
    m["plate_reverb"] = {"0.4 s decay, size 1.0", p};
  }
  {
    ParamSet p;
    p.set("voices", 1.0).set("rate_hz", 0.1).set("depth_ms", 0.5)
        .set("base_delay_ms", 12.0).set("randomize_phase", 0.0);
    m["chorus"] = {"1 voice, 0.1 Hz, 12 ms base", p};
  }
  {
    ParamSet p;
    p.set("width_percent", 130.0).set("bass_mono_below_hz", 120.0)
        .set("bass_width_percent", 0.0).set("crossover_stopband_db", 80.0);
    m["stereo_width"] = {"130%, bass mono below 120 Hz", p};
  }
  {
    ParamSet p;
    p.set("time_ms", 20.0).set("feedback_percent", 35.0)
        .set("hf_damp_hz", 7000.0).set("lf_damp_hz", 120.0)
        .set("spread_ms", 0.0).set("max_tail_s", 1.0);
    p.setStr("stereo_mode", "mono");
    m["delay"] = {"20 ms, 35% feedback", p};
  }

  return m;
}

// Extra configurations worth measuring beyond one per type. Only two so far,
// and both exist because the headline aliasing requirement is stated at a
// setting the documented registry does not use.
std::vector<Config> extraConfigs() {
  std::vector<Config> v;
  {
    ParamSet p;
    p.set("drive_db", 6.0).set("output_trim_db", 0.0).set("bias", 0.0)
        .set("oversample_factor", 4.0);
    p.setStr("curve", "soft_clip");
    v.push_back({"drive@defaults", "drive",
                 "soft clip, node defaults (+6 dB, no bias, 4x)", p});
  }
  {
    ParamSet p;
    p.set("drive_db", 12.0).set("output_trim_db", -6.0).set("bias", 0.15)
        .set("oversample_factor", 16.0);
    p.setStr("curve", "soft_clip");
    v.push_back({"drive@16x", "drive", "soft clip, +12 dB drive, 16x", p});
  }
  return v;
}

std::vector<Config> buildConfigs(const std::vector<std::string>& types) {
  const auto overrides = configOverrides();
  std::vector<Config> configs;
  for (const auto& t : types) {
    auto it = overrides.find(t);
    if (it == overrides.end()) {
      // A node nobody has written a stimulus for is still measured, at its own
      // defaults. That is the whole point of deriving the list from makeNode.
      configs.push_back({t, t, "node defaults (no stimulus override)", ParamSet{}});
    } else {
      configs.push_back({t, t, it->second.first, it->second.second});
    }
  }
  for (auto& e : extraConfigs()) configs.push_back(std::move(e));
  return configs;
}

// --------------------------------------------------------------- rendering --

constexpr double kFs = kCanonicalSampleRate;
constexpr int kChannels = 2;
constexpr int kBlock = 1024;

PrepareSpec makeSpec(int64_t total) {
  PrepareSpec s;
  s.sampleRate = kFs;
  s.maxBlockFrames = kBlock;
  s.numChannels = kChannels;
  s.totalFrames = total;
  s.seed = 20260728;
  return s;
}

// One node, rendered through the real Chain so latency compensation, the
// dry/wet wrapper and the block driver are all exactly what a production
// render uses. Measuring a node by calling process() directly would measure a
// program nobody ships.
AudioBuffer renderOne(const Config& cfg, const AudioBuffer& in, bool bypassed) {
  Chain chain;
  Chain::SlotConfig sc;
  sc.nodeId = cfg.id;
  sc.bypassed = bypassed;
  auto node = makeNode(cfg.type);
  chain.add(std::move(node), cfg.params, sc);
  chain.prepare(makeSpec(in.frames()));
  AudioBuffer out;
  chain.render(in, out, kBlock);
  return out;
}

AudioBuffer toneBuffer(int64_t frames, double freqHz, double amplitude,
                       double phase = 0.0) {
  AudioBuffer b(kChannels, frames);
  for (int64_t i = 0; i < frames; ++i) {
    const double s = amplitude * std::sin(2.0 * M_PI * freqHz *
                                          static_cast<double>(i) / kFs + phase);
    for (int c = 0; c < kChannels; ++c) b.channel(c)[i] = s;
  }
  return b;
}

// THE ANALYSIS WINDOW SITS IN THE MIDDLE OF THE STIMULUS, NEVER AT ITS END.
//
// This is findings.md's "measuring the wrong window", and it cost real numbers
// here before it was noticed: with the window flush against the end of the
// tone, `zero_phase_eq` measured -89 dB THD and `chorus` -74 dB, on nodes that
// are incapable of producing a harmonic. Both reach BACKWARDS — a filtfilt
// pass by up to its 24 000-frame reach, a lookahead limiter by its window —
// so the last samples of such a window describe the response decaying into the
// silence past the signal, not a steady state. Every measurement below keeps
// `kPostRoll` samples of live tone after the window it reads.
//
// 32768 frames because it exceeds the largest backwards reach in the engine
// (ZeroPhaseEQ's 24 000-frame cap) with margin. A node that ever reaches
// further would show up as harmonic distortion it cannot possibly produce,
// which is a loud and recognisable failure rather than a quiet bias.
constexpr int64_t kPostRoll = 32768;

double linToDbPower(double p) {
  // No floor. A clamp here is exactly the "floored measurement function" bug
  // in findings.md, where a -313 dB null read -240 because the helper said so.
  return 10.0 * std::log10(p);
}

// ------------------------------------------------- coherent tone analysis --
//
// N is a power of two so the radix-2 FFT applies; the tone's bin index k is
// ODD, which is what guarantees the harmonic classification below can never be
// ambiguous. m*k mod N collides with +/- m'*k only when m +/- m' is a multiple
// of N, and the harmonic orders considered here never come close, so no folded
// harmonic can ever hide inside a legitimate one.

bool isOdd(int k) { return (k % 2) != 0; }

int nearestOddBin(double targetHz, size_t n) {
  int k = static_cast<int>(std::lround(targetHz * static_cast<double>(n) / kFs));
  if (!isOdd(k)) ++k;
  return k;
}

struct ToneSpectrum {
  std::vector<double> power; // bins 0..N/2
  size_t n = 0;
  double hzPerBin = 0.0;
};

ToneSpectrum analyse(const AudioBuffer& out, int64_t offset, size_t n) {
  ToneSpectrum s;
  s.n = n;
  s.hzPerBin = kFs / static_cast<double>(n);
  if (out.frames() < offset + static_cast<int64_t>(n)) return s;
  s.power = powerSpectrum(out.channel(0) + offset, n);
  return s;
}

// THE SETTLE TIME IS MEASURED, NOT GUESSED.
//
// A deterministic node fed an exactly periodic tone produces an exactly
// periodic output, so ALL of its energy lands on multiples of the tone's bin.
// Energy anywhere else is either broadband noise or a transient that has not
// finished — and a transient is fatal here, because its rectangular-window
// skirts spread across the whole spectrum and land squarely in the bins the
// aliasing measurement reads.
//
// This is not hypothetical. With a fixed 0.34 s settle the compressor reported
// -78 dB of in-band aliasing. Almost none of it was aliasing: its "program"
// release converges over SECONDS, and at 0.34 s the residual drift sat at
// -48 dB relative to the fundamental, one bin either side of the tone, leaking
// down into the alias bins. At a 2.7 s settle the same node's off-locus energy
// is -130 dB and its aliasing measures what it actually is.
//
// So the probe lengthens the settle until the off-locus residual is 20 dB below
// the -100 dB requirement, and stops early when a longer settle stops helping —
// which distinguishes a transient (shrinks) from a noise floor (does not). The
// settle actually used is reported, because a number that needed 5 s of
// convergence is a different kind of number from one that needed none.
struct CoherentProbe {
  ToneSpectrum spec;
  int64_t settleUsed = 0;
  double offLocusDb = 0.0; // non-periodic residual, dB relative to fundamental
  bool converged = false;
  bool ok = false;
};

constexpr double kResidualTargetDb = -120.0;
constexpr int kAsrMaxHarmonic = 400;

// Every bin an exactly periodic output could occupy: harmonic m lands at
// m*k mod N, mirrored about Nyquist. NOT simply the multiples of k — a folded
// harmonic is not a multiple of k, and treating it as off-locus was a real bug
// here: it counted the aliasing itself as evidence that the output was not
// periodic, so the residual number exactly equalled the ASR number for four
// different nodes and said nothing.
std::vector<uint8_t> harmonicLocus(size_t n, int k) {
  std::vector<uint8_t> onLocus(n / 2 + 1, 0);
  for (int m = 1; m <= kAsrMaxHarmonic; ++m) {
    size_t bin = (static_cast<size_t>(m) * static_cast<size_t>(k)) % n;
    if (bin > n / 2) bin = n - bin;
    if (bin != 0) onLocus[bin] = 1;
  }
  return onLocus;
}

CoherentProbe probeTone(const Config& cfg, size_t n, int k, double amplitude) {
  const int64_t settles[] = {32768, 131072, 524288};
  CoherentProbe best;
  double previous = 0.0;
  bool havePrevious = false;

  for (int64_t settle : settles) {
    const AudioBuffer in =
        toneBuffer(settle + static_cast<int64_t>(n) + kPostRoll,
                   static_cast<double>(k) * kFs / static_cast<double>(n), amplitude);
    const AudioBuffer out = renderOne(cfg, in, false);
    const ToneSpectrum s = analyse(out, settle, n);
    if (s.power.empty()) return best;

    const double fundamental = s.power[static_cast<size_t>(k)];
    if (!(fundamental > 0.0)) return best;
    const std::vector<uint8_t> onLocus = harmonicLocus(n, k);
    double offLocus = 0.0;
    for (size_t b = 1; b <= n / 2; ++b)
      if (!onLocus[b]) offLocus += s.power[b];
    const double db = linToDbPower(offLocus / fundamental);

    best.spec = s;
    best.settleUsed = settle;
    best.offLocusDb = db;
    best.converged = db <= kResidualTargetDb;
    best.ok = true;

    if (best.converged) break;
    // A residual that stops shrinking is a noise floor, not a transient, and no
    // amount of further settling will move it. Dither's is exactly this.
    if (havePrevious && (previous - db) < 3.0) break;
    previous = db;
    havePrevious = true;
  }
  return best;
}

// -------------------------------------------------------------------- THD --
//
// Harmonic distortion at a coherently sampled tone. Two numbers, because they
// answer different questions: THD counts only the true harmonics (bins at
// m*f0 below Nyquist), THD+N counts everything in the spectrum that is not the
// fundamental. A node whose THD is clean but whose THD+N is not is adding
// noise or non-harmonic modulation, and that distinction is the difference
// between a saturator and a broken envelope follower.

struct ThdResult {
  Value thdDb = Value::none();
  Value thdPercent = Value::none();
  Value thdNDb = Value::none();
  Value fundamentalDbfs = Value::none();
  Value settleFrames = Value::none();
  double f0Hz = 0.0;
};

ThdResult measureThd(const Config& cfg, double amplitude) {
  constexpr size_t kN = 65536;
  const int k = nearestOddBin(997.0, kN); // the standard 1 kHz audio test tone
  const double f0 = static_cast<double>(k) * kFs / static_cast<double>(kN);

  ThdResult r;
  r.f0Hz = f0;

  const CoherentProbe probe = probeTone(cfg, kN, k, amplitude);
  if (!probe.ok) return r;
  const ToneSpectrum& s = probe.spec;
  r.settleFrames = Value::of(static_cast<double>(probe.settleUsed));

  const double fundamental = s.power[static_cast<size_t>(k)];
  if (!(fundamental > 0.0)) return r; // nothing to normalise against: null, not 0

  double harmonics = 0.0;
  for (int mIndex = 2;; ++mIndex) {
    const double hz = static_cast<double>(mIndex) * f0;
    if (hz >= kFs * 0.5) break;
    harmonics += s.power[static_cast<size_t>(mIndex * k)];
  }

  double total = 0.0;
  for (size_t b = 1; b <= kN / 2; ++b)
    if (b != static_cast<size_t>(k)) total += s.power[b];

  // The FFT of a real buffer has no scale-independent absolute level; the tone
  // amplitude is recovered from the bin against a full-scale reference, which
  // is 2*(N/2)^2 in this convention. Ratios below do not depend on it, but the
  // reported fundamental level does.
  const double fullScalePower = std::pow(static_cast<double>(kN) * 0.5, 2.0);
  r.fundamentalDbfs = Value::of(linToDbPower(fundamental / fullScalePower));
  r.thdDb = Value::of(linToDbPower(harmonics / fundamental));
  r.thdPercent = Value::of(100.0 * std::sqrt(harmonics / fundamental));
  r.thdNDb = Value::of(linToDbPower(total / fundamental));
  return r;
}

// ------------------------------------------------------------------- ASR ---
//
// Aliasing-to-Signal Ratio, the number that decides whether a nonlinear node
// sounds like gear or like fizz.
//
// THE TECHNIQUE. Sweep a sine upward. A true harmonic of the input moves UP
// with the fundamental; an alias — a harmonic generated above Nyquist and
// folded back — moves DOWN. So the harmonic loci are known exactly at every
// point in the sweep, and everything left after masking them is aliasing.
//
// The sweep is STEPPED rather than continuous, and that is a strengthening
// rather than a compromise. A continuous sweep has to be analysed with a
// windowed STFT, whose leakage skirts are tens of dB above the -100 dB the
// requirement is stated at; the mask then has to be a BAND around each locus,
// which either hides aliases sitting near a harmonic or lets the window's own
// leakage count as one. Stepping the sweep and choosing each step coherently
// makes every harmonic land in exactly one bin with zero leakage, so the mask
// is a single bin wide and costs nothing. Same experiment, exact arithmetic.
//
// Denominator is the FUNDAMENTAL, not the sum of all harmonics. "Alias to
// SIGNAL" — the signal is the tone that went in. Using the harmonic sum
// instead inflates the denominator by everything the nonlinearity added and
// flatters the result by several dB; that variant is reported too, because it
// is what the existing contract test gates on and the two must be comparable.

struct AsrStep {
  double f0Hz = 0.0;
  double inBandDb = 0.0;
  double fullBandDb = 0.0;
  double vsHarmonicsDb = 0.0;
  double noiseFloorDb = 0.0;
};

struct AsrResult {
  bool measured = false;
  bool collisionFree = true;
  bool allConverged = true;
  double worstInBandDb = 0.0;
  double worstFullBandDb = 0.0;
  double worstVsHarmonicsDb = 0.0;
  double worstAtHz = 0.0;
  double noiseFloorAtWorstDb = 0.0;
  double worstResidualDb = -1e300;
  int64_t longestSettle = 0;
  std::vector<AsrStep> steps;
};

constexpr size_t kAsrN = 32768;
constexpr double kAudibleHz = 20000.0;

// Classify one coherently-sampled spectrum and reduce it to one step's worth
// of numbers. Factored out so the harness can run it against a SYNTHETIC
// spectrum whose answer is known by construction — see selfValidation. That
// check is the difference between "the classifier agrees with itself" and
// "the classifier is right".
AsrStep classifyAsr(const ToneSpectrum& s, int k, bool& collisionFree) {
  const double f0 = static_cast<double>(k) * kFs / static_cast<double>(s.n);

  // 0 = untouched, 1 = a harmonic that legitimately fits below Nyquist,
  // 2 = a harmonic that folded and therefore was not in the input.
  std::vector<uint8_t> role(s.n / 2 + 1, 0);
  for (int mIndex = 1; mIndex <= kAsrMaxHarmonic; ++mIndex) {
    size_t bin = (static_cast<size_t>(mIndex) * static_cast<size_t>(k)) % s.n;
    if (bin > s.n / 2) bin = s.n - bin;
    if (bin == 0) continue;
    const bool legitimate = (static_cast<double>(mIndex) * f0) < (kFs * 0.5);
    const uint8_t want = legitimate ? uint8_t{1} : uint8_t{2};
    if (role[bin] != 0 && role[bin] != want) collisionFree = false;
    // A legitimate harmonic wins a contested bin. That moves energy into the
    // denominator rather than the numerator, so the reported ASR can only be
    // optimistic — never invented.
    if (role[bin] == 0 || want == 1) role[bin] = want;
  }

  const double fundamental = s.power[static_cast<size_t>(k)];
  double harmonicSum = 0.0, aliasFull = 0.0, aliasInBand = 0.0;
  size_t aliasBinsInBand = 0;
  std::vector<double> offLocus;
  offLocus.reserve(s.n / 2);
  for (size_t b = 1; b <= s.n / 2; ++b) {
    const double hz = static_cast<double>(b) * s.hzPerBin;
    if (role[b] == 1) {
      harmonicSum += s.power[b];
    } else if (role[b] == 2) {
      aliasFull += s.power[b];
      if (hz < kAudibleHz) {
        aliasInBand += s.power[b];
        ++aliasBinsInBand;
      }
    } else {
      offLocus.push_back(s.power[b]);
    }
  }

  AsrStep st;
  st.f0Hz = f0;
  if (!(fundamental > 0.0)) {
    st.f0Hz = -1.0; // caller's signal that nothing could be normalised
    return st;
  }

  // How much of the alias total could be nothing but the node's own noise
  // floor. A periodic input through a deterministic node puts energy ONLY on
  // multiples of k, so any energy off those bins is broadband noise; its
  // median density times the number of alias bins is what that noise would
  // contribute if it were spread evenly. Reported, never subtracted: a
  // corrected number that can go negative is worse than an honest one that is
  // noise-limited and says so.
  double noiseInAlias = 0.0;
  if (!offLocus.empty() && aliasBinsInBand > 0) {
    std::nth_element(offLocus.begin(),
                     offLocus.begin() + static_cast<long>(offLocus.size() / 2),
                     offLocus.end());
    noiseInAlias = offLocus[offLocus.size() / 2] * static_cast<double>(aliasBinsInBand);
  }

  st.inBandDb = linToDbPower(aliasInBand / fundamental);
  st.fullBandDb = linToDbPower(aliasFull / fundamental);
  st.vsHarmonicsDb = linToDbPower(aliasFull / std::max(harmonicSum, 1e-300));
  st.noiseFloorDb = linToDbPower(noiseInAlias / fundamental);
  return st;
}

AsrResult measureAsr(const Config& cfg, double amplitude) {
  constexpr size_t kN = kAsrN;

  // Roughly log-spaced from where a 4x oversampler is comfortable to where the
  // third harmonic is already above the base Nyquist. Aliasing is a function
  // of where the harmonics land, so the sweep has to reach the top.
  const double targets[] = {1000.0, 1800.0, 3200.0,  5600.0,
                            9000.0, 13000.0, 16500.0, 19500.0};

  AsrResult r;
  for (double target : targets) {
    const int k = nearestOddBin(target, kN);
    const CoherentProbe probe = probeTone(cfg, kN, k, amplitude);
    if (!probe.ok) return r;
    if (!probe.converged) r.allConverged = false;
    r.worstResidualDb = std::max(r.worstResidualDb, probe.offLocusDb);
    r.longestSettle = std::max(r.longestSettle, probe.settleUsed);

    const AsrStep st = classifyAsr(probe.spec, k, r.collisionFree);
    if (st.f0Hz < 0.0) return r; // nothing came out; do not invent a ratio
    r.steps.push_back(st);
  }

  if (r.steps.empty()) return r;
  r.measured = true;
  const AsrStep* worst = &r.steps[0];
  for (const auto& st : r.steps)
    if (st.inBandDb > worst->inBandDb) worst = &st;
  r.worstInBandDb = worst->inBandDb;
  r.worstAtHz = worst->f0Hz;
  r.noiseFloorAtWorstDb = worst->noiseFloorDb;
  r.worstFullBandDb = r.steps[0].fullBandDb;
  r.worstVsHarmonicsDb = r.steps[0].vsHarmonicsDb;
  for (const auto& st : r.steps) {
    r.worstFullBandDb = std::max(r.worstFullBandDb, st.fullBandDb);
    r.worstVsHarmonicsDb = std::max(r.worstVsHarmonicsDb, st.vsHarmonicsDb);
  }
  return r;
}

// ------------------------------------------------------------------- IMD ---
//
// Twin tone at 19 kHz + 20 kHz, the CCIF/ITU-R test. Two tones close together
// near the top of the band: any nonlinearity produces a difference product at
// 1 kHz, sitting in the most audible part of the spectrum where no harmonic of
// either tone can reach. That separation is the whole point of the test — a
// 1 kHz product cannot be anything but intermodulation.
//
// Both tones are placed on exact bins so the difference and third-order
// products land on exact bins too. Each tone is at -6 dBFS so the pair sums to
// 0 dBFS, which is the level the requirement is stated at.

struct ImdResult {
  Value d2Db = Value::none();      // difference product fA-fB
  Value d2Percent = Value::none();
  Value d3Db = Value::none();      // worst of 2fA-fB and 2fB-fA
  double fALow = 0.0, fBHigh = 0.0, fDiff = 0.0;
};

ImdResult measureImd(const Config& cfg) {
  constexpr size_t kN = 65536;
  constexpr int64_t kSettle = 65536;

  const int kA = nearestOddBin(19000.0, kN);
  const int kB = nearestOddBin(20000.0, kN);
  const int kD = kB - kA;
  const int k3lo = 2 * kA - kB;
  const int k3hi = 2 * kB - kA;

  ImdResult r;
  r.fALow = static_cast<double>(kA) * kFs / static_cast<double>(kN);
  r.fBHigh = static_cast<double>(kB) * kFs / static_cast<double>(kN);
  r.fDiff = static_cast<double>(kD) * kFs / static_cast<double>(kN);

  const int64_t total = kSettle + static_cast<int64_t>(kN) + kPostRoll;
  AudioBuffer in(kChannels, total);
  for (int64_t i = 0; i < total; ++i) {
    const double t = static_cast<double>(i) / kFs;
    const double s = 0.5 * std::sin(2.0 * M_PI * r.fALow * t) +
                     0.5 * std::sin(2.0 * M_PI * r.fBHigh * t);
    for (int c = 0; c < kChannels; ++c) in.channel(c)[i] = s;
  }

  const AudioBuffer out = renderOne(cfg, in, false);
  const ToneSpectrum s = analyse(out, kSettle, kN);
  if (s.power.empty()) return r;

  const double tones = s.power[static_cast<size_t>(kA)] + s.power[static_cast<size_t>(kB)];
  if (!(tones > 0.0)) return r;

  const double d2 = s.power[static_cast<size_t>(kD)];
  const double d3 = std::max(s.power[static_cast<size_t>(k3lo)],
                             s.power[static_cast<size_t>(k3hi)]);
  r.d2Db = Value::of(linToDbPower(d2 / tones));
  r.d2Percent = Value::of(100.0 * std::sqrt(d2 / tones));
  r.d3Db = Value::of(linToDbPower(d3 / tones));
  return r;
}

// ---------------------------------------------------------------- Farina ---
//
// Exponential swept sine plus its inverse filter (Farina, AES 2000).
// Deconvolving the response against the inverse filter collapses the sweep to
// an impulse response, and — this is the property that makes it worth the
// complexity over a plain impulse — the harmonic distortion products collapse
// to SEPARATE impulses arriving BEFORE the linear one, at times fixed by the
// sweep rate. Windowing them off leaves the linear response of a node that is
// not perfectly linear, which is exactly what "the frequency response of a
// compressor" has to mean.
//
// The stimulus is small (-26 dBFS) for the same reason the contract suite's
// latency probe is: group delay is a linear-system property, and driving a
// saturator into clipping does not measure it.

struct FarinaResult {
  bool measured = false;
  std::vector<double> freqHz;
  std::vector<double> magDb;
  std::vector<double> phaseDeg;
  std::vector<double> groupDelaySamples;
  double maxMagDb = 0.0;
  double minMagDb = 0.0;
  double maxAbsGroupDelay = 0.0;
  double preArrivalDb = 0.0;
};

std::vector<std::complex<double>> fftOfReal(const std::vector<double>& x) {
  std::vector<std::complex<double>> buf(x.size());
  for (size_t i = 0; i < x.size(); ++i) buf[i] = {x[i], 0.0};
  fftInPlace(buf);
  return buf;
}

FarinaResult measureFarina(const Config& cfg) {
  constexpr size_t kM = 262144;  // 2.73 s at 96 kHz, comfortably past every tail
  constexpr size_t kSweep = 65536; // 0.68 s
  constexpr double kF1 = 20.0;
  constexpr double kF2 = 40000.0; // below base Nyquist; reported only to 20 kHz
  constexpr double kAmp = 0.05;

  const double R = std::log(kF2 / kF1);
  const double T = static_cast<double>(kSweep) / kFs;

  AudioBuffer in(kChannels, static_cast<int64_t>(kM));
  std::vector<double> inverse(kM, 0.0);
  for (size_t i = 0; i < kSweep; ++i) {
    const double t = static_cast<double>(i) / kFs;
    const double phase = 2.0 * M_PI * kF1 * T / R * (std::exp(t * R / T) - 1.0);
    // Raised-cosine fades. Without them the sweep starts and stops on a
    // discontinuity, and the deconvolution spreads that click across the whole
    // impulse response — which shows up as phase noise, not as anything
    // recognisable. 256 samples in (the 20 Hz start is nearly at a zero
    // crossing anyway) and 1024 out, where the instantaneous frequency is
    // 40 kHz and a cycle is two samples long.
    const double fadeIn = 256.0, fadeOut = 1024.0;
    double fade = 1.0;
    if (static_cast<double>(i) < fadeIn)
      fade = 0.5 * (1.0 - std::cos(M_PI * static_cast<double>(i) / fadeIn));
    const double fromEnd = static_cast<double>(kSweep - 1 - i);
    if (fromEnd < fadeOut)
      fade *= 0.5 * (1.0 - std::cos(M_PI * fromEnd / fadeOut));
    const double s = kAmp * fade * std::sin(phase);
    for (int c = 0; c < kChannels; ++c) in.channel(c)[static_cast<int64_t>(i)] = s;
    // Time-reversed and amplitude-corrected: the sweep spends log-time at each
    // frequency, so the inverse has to undo the resulting pink tilt.
    const size_t j = kSweep - 1 - i;
    inverse[j] = s * std::exp(-t * R / T);
  }

  const AudioBuffer outNode = renderOne(cfg, in, false);
  const AudioBuffer outNull = renderOne(cfg, in, true);

  FarinaResult r;
  if (outNode.frames() < static_cast<int64_t>(kM) ||
      outNull.frames() < static_cast<int64_t>(kM))
    return r;

  const auto invSpec = fftOfReal(inverse);

  auto deconvolve = [&](const AudioBuffer& out) {
    std::vector<double> x(kM);
    for (size_t i = 0; i < kM; ++i) x[i] = out.channel(0)[static_cast<int64_t>(i)];
    auto spec = fftOfReal(x);
    for (size_t i = 0; i < kM; ++i) spec[i] *= invSpec[i];
    ifftInPlace(spec);
    std::vector<double> ir(kM);
    for (size_t i = 0; i < kM; ++i) ir[i] = spec[i].real();
    return ir;
  };

  std::vector<double> irNode = deconvolve(outNode);
  std::vector<double> irNull = deconvolve(outNull);

  // The linear arrival. Located on the NULL render, whose impulse response is
  // a single peak by construction, so a node with a diffuse response cannot
  // move the window it is measured in.
  size_t t0 = 0;
  double best = 0.0;
  for (size_t i = 0; i < kM; ++i)
    if (std::abs(irNull[i]) > best) { best = std::abs(irNull[i]); t0 = i; }

  // THERE IS NO PRE-WINDOW, AND THAT IS A DELIBERATE REVERSAL.
  //
  // The textbook move is to window off everything before the linear arrival,
  // because Farina's harmonic products land there. It was implemented, and it
  // was wrong for this sweep: the deconvolved linear impulse is band-limited to
  // the sweep's 20 Hz low edge, so it is roughly 50 ms WIDE, and it is still at
  // -13 dB one thousand samples from its peak and -35 dB four thousand. The
  // second harmonic arrives T*ln2/ln(f2/f1) = 62 ms early, which is inside
  // that. No window separates them at this sweep length.
  //
  // Cutting at 4096 samples truncated the linear response at -35 dB, and the
  // resulting ripple was +/-3 dB of magnitude and 20 degrees of phase on
  // `dc_block` — artefacts an order of magnitude larger than anything being
  // measured, and exactly the shape a reader would mistake for filter
  // behaviour. Removing it, `linear_phase_eq` reads 0.0000 degrees at every
  // bin, which is the number its header claims and the reason to trust the
  // rest of this table.
  //
  // What is left in exchange is the distortion products, and the answer to
  // those is the stimulus level: -26 dBFS keeps every node quasi-linear. The
  // residual is measured rather than assumed — see the pre-arrival metric
  // below, which is an upper bound on how much distortion this table folds in.
  // Harmonics 3 and above arrive between 12 000 and 30 000 samples before the
  // linear impulse at this sweep rate, and the linear impulse's own skirt is
  // below -50 dB across that whole span. Energy there is distortion, and its
  // ratio to the total is an upper bound on the contamination in the response.
  //
  // Measured AGAINST THE NULL RENDER, not in absolute terms: the linear
  // impulse's own skirt puts -47.9 dB into that region for every node,
  // distorting or not, and reporting that as distortion would be reporting the
  // stimulus. The excess over the null render is what the node added.
  auto preArrivalRatio = [&](const std::vector<double>& ir) {
    double pre = 0.0, total = 0.0;
    for (size_t i = 0; i < kM; ++i) {
      const double e = ir[i] * ir[i];
      total += e;
      if (i + 30000 < t0 || i + 12000 >= t0) continue;
      pre += e;
    }
    return total > 0.0 ? pre / total : std::nan("");
  };
  const double preNode = preArrivalRatio(irNode);
  const double preNull = preArrivalRatio(irNull);

  const auto specNode = fftOfReal(irNode);
  const auto specNull = fftOfReal(irNull);

  const double hzPerBin = kFs / static_cast<double>(kM);
  const double freqs[] = {31.5, 63.0, 125.0, 250.0, 500.0, 1000.0,
                          2000.0, 4000.0, 8000.0, 16000.0};

  auto transferAt = [&](size_t bin) {
    const std::complex<double> denom = specNull[bin];
    if (std::abs(denom) == 0.0)
      return std::complex<double>(std::nan(""), std::nan(""));
    return specNode[bin] / denom;
  };

  // GROUP DELAY BY LEAST SQUARES OVER A SPAN OF BINS, NOT BY DIFFERENCING TWO.
  //
  // A single-bin difference divides the phase error by 2*pi/M, which at
  // M = 262144 multiplies it by 41722. The first version of this did exactly
  // that and reported 29 samples of residual group delay for `linear_phase_eq`
  // and 1486 for `dc_block` at 1 kHz — both of which are noise amplified by
  // that factor, and both of which a reader would have taken for a property of
  // the node. Fitting a slope across +/-12 bins averages the noise down by the
  // span while still resolving any delay this engine can produce.
  //
  // The phase is unwrapped by accumulating principal-value differences within
  // the span only. A global unwrap would have to survive the stopbands, where
  // the phase is whatever round-off says it is.
  const int span = 12;
  auto groupDelayAt = [&](size_t centre) {
    if (centre < static_cast<size_t>(span) + 1) return std::nan("");
    std::vector<double> w, phi;
    double acc = std::arg(transferAt(centre - static_cast<size_t>(span)));
    double prev = acc;
    for (int d = -span; d <= span; ++d) {
      const size_t b = centre + static_cast<size_t>(d);
      const double a = std::arg(transferAt(b));
      if (d != -span) {
        double step = a - prev;
        while (step > M_PI) step -= 2.0 * M_PI;
        while (step <= -M_PI) step += 2.0 * M_PI;
        acc += step;
      }
      prev = a;
      w.push_back(static_cast<double>(d));
      phi.push_back(acc);
    }
    double sxy = 0.0, sxx = 0.0;
    const double meanPhi = std::accumulate(phi.begin(), phi.end(), 0.0) /
                           static_cast<double>(phi.size());
    for (size_t i = 0; i < w.size(); ++i) {
      sxy += w[i] * (phi[i] - meanPhi);
      sxx += w[i] * w[i];
    }
    if (sxx == 0.0) return std::nan("");
    const double slopePerBin = sxy / sxx;
    return -slopePerBin / (2.0 * M_PI / static_cast<double>(kM));
  };

  for (double f : freqs) {
    const size_t bin = static_cast<size_t>(std::lround(f / hzPerBin));
    const std::complex<double> h = transferAt(bin);
    r.freqHz.push_back(static_cast<double>(bin) * hzPerBin);
    r.magDb.push_back(20.0 * std::log10(std::abs(h)));
    r.phaseDeg.push_back(std::arg(h) * 180.0 / M_PI);
    r.groupDelaySamples.push_back(groupDelayAt(bin));
  }

  // Magnitude extremes across the audible band, as a single-number fingerprint
  // of the response shape.
  const size_t lo = static_cast<size_t>(std::lround(20.0 / hzPerBin));
  const size_t hi = static_cast<size_t>(std::lround(20000.0 / hzPerBin));
  r.maxMagDb = -1e300;
  r.minMagDb = 1e300;
  for (size_t b = lo; b <= hi; ++b) {
    const double m = 20.0 * std::log10(std::abs(transferAt(b)));
    if (std::isfinite(m)) {
      r.maxMagDb = std::max(r.maxMagDb, m);
      r.minMagDb = std::min(r.minMagDb, m);
    }
  }
  r.preArrivalDb = (preNull > 0.0) ? linToDbPower(preNode / preNull) : std::nan("");
  r.maxAbsGroupDelay = 0.0;
  for (double g : r.groupDelaySamples)
    if (std::isfinite(g)) r.maxAbsGroupDelay = std::max(r.maxAbsGroupDelay, std::abs(g));

  r.measured = true;
  return r;
}

// ------------------------------------------- independent true-peak meter ---
//
// Written from the definition, sharing nothing with anv::TruePeakUpsampler or
// anv::FirDesign: different oversampling factor (32 against 4), different
// window (4-term Blackman-Harris against Kaiser), different length (6145 taps
// against ~200). If it agreed with the engine's meter because it WAS the
// engine's meter, the comparison would be a tautology — which is exactly the
// bug findings.md records twice, once where a halfband cascade under-reported
// in the same direction as the limiter it was checking, and once where two
// approximations were compared and neither was right.
//
// Validated against the analytic answer rather than against another estimator:
// a sampled sinusoid's band-limited reconstruction IS that sinusoid, so a unit
// sinusoid's true peak is exactly 0 dBTP at every frequency and phase.
class ReferenceTruePeak {
public:
  ReferenceTruePeak() {
    const int L = 2 * kFactor * kHalfSpan + 1;
    const int centre = kFactor * kHalfSpan;
    std::vector<double> h(static_cast<size_t>(L));
    for (int i = 0; i < L; ++i) {
      const double x = static_cast<double>(i - centre) / static_cast<double>(kFactor);
      const double sinc = (x == 0.0) ? 1.0 : std::sin(M_PI * x) / (M_PI * x);
      // Blackman-Harris, 4-term, -92 dB sidelobes.
      const double u = 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(L - 1);
      const double w = 0.35875 - 0.48829 * std::cos(u) + 0.14128 * std::cos(2.0 * u) -
                       0.01168 * std::cos(3.0 * u);
      h[static_cast<size_t>(i)] = sinc * w;
    }
    branchLen_ = (L + kFactor - 1) / kFactor;
    branches_.assign(static_cast<size_t>(kFactor),
                     std::vector<double>(static_cast<size_t>(branchLen_), 0.0));
    for (int i = 0; i < L; ++i)
      branches_[static_cast<size_t>(i % kFactor)][static_cast<size_t>(i / kFactor)] =
          h[static_cast<size_t>(i)];
    // Per-branch normalisation to unity DC gain. Normalising the filter as a
    // whole leaves the branches differing by a fraction of a percent, and since
    // this estimator takes the MAXIMUM over branches that spread becomes a
    // constant positive bias on every measurement.
    for (auto& br : branches_) {
      const double s = std::accumulate(br.begin(), br.end(), 0.0);
      if (s != 0.0) for (double& v : br) v /= s;
    }
  }

  int guardSamples() const { return 4 * kHalfSpan; }

  // Peak of the band-limited reconstruction, linear. `guard` base samples are
  // excluded at each end: the buffer is zero-extended, and a signal that does
  // not decay to zero at its edges would ring against that discontinuity and
  // report a peak that is an artefact of the measurement rather than of the
  // signal.
  double measure(const double* x, int64_t n, int guard) const {
    const int pad = 2 * kHalfSpan;
    std::vector<double> buf(static_cast<size_t>(n) + 2 * static_cast<size_t>(pad), 0.0);
    std::copy(x, x + n, buf.begin() + pad);

    const int64_t first = pad + guard;
    const int64_t last = pad + n - guard + pad; // include the decaying tail
    double best = 0.0;
    for (int64_t i = first; i < last && i < static_cast<int64_t>(buf.size()); ++i) {
      for (int p = 0; p < kFactor; ++p) {
        const auto& br = branches_[static_cast<size_t>(p)];
        const int taps = static_cast<int>(std::min<int64_t>(branchLen_, i + 1));
        double acc = 0.0;
        for (int t = 0; t < taps; ++t)
          acc += br[static_cast<size_t>(t)] * buf[static_cast<size_t>(i - t)];
        best = std::max(best, std::abs(acc));
      }
    }
    return best;
  }

private:
  static constexpr int kFactor = 32;
  static constexpr int kHalfSpan = 96; // 6145 taps
  int branchLen_ = 1;
  std::vector<std::vector<double>> branches_;
};

double engineTruePeak(const double* x, int64_t n) {
  TruePeakUpsampler up;
  up.design(4, 80.0, 0.05); // exactly what TruePeakLimiter uses
  up.prepare(1);
  double best = 0.0;
  for (int64_t i = 0; i < n; ++i) best = std::max(best, up.processMaxAbs(x[i], 0));
  // Flush the filter's own latency so the last samples are actually measured.
  for (int i = 0; i < up.latencyFrames() + 1; ++i)
    best = std::max(best, up.processMaxAbs(0.0, 0));
  return best;
}

double dbFromLinear(double x) { return 20.0 * std::log10(x); }

// A signal engineered to have large inter-sample peaks: a tone at exactly fs/4
// at 45 degrees (samples land at +/-0.7071 while the waveform reaches 1.0 —
// the textbook worst case), plus content that keeps the sample peak pinned at
// full scale so a limiter has something to do. Silence at both ends so the
// reference meter's zero-extension is the truth rather than an edge artefact.
AudioBuffer truePeakStress(int64_t frames) {
  AudioBuffer b(kChannels, frames);
  CounterRng rng(4242);
  const int64_t fadeIn = 512, active = frames - 2048;
  for (int64_t i = 0; i < frames; ++i) {
    if (i < fadeIn || i >= fadeIn + active) continue;
    const double t = static_cast<double>(i) / kFs;
    const double env = std::min(1.0, static_cast<double>(i - fadeIn) / 256.0) *
                       std::min(1.0, static_cast<double>(fadeIn + active - i) / 256.0);
    double s = 0.62 * std::sin(2.0 * M_PI * (kFs * 0.25) * t + M_PI / 4.0) +
               0.22 * std::sin(2.0 * M_PI * 997.0 * t) +
               0.16 * rng.bipolar(0, i);
    s = std::clamp(s, -1.0, 1.0) * env;
    for (int c = 0; c < kChannels; ++c) b.channel(c)[i] = s;
  }
  // Normalise the SAMPLE peak to exactly 0 dBFS. The inter-sample peak is then
  // whatever it is, which is the point.
  double pk = 0.0;
  for (int64_t i = 0; i < frames; ++i) pk = std::max(pk, std::abs(b.channel(0)[i]));
  if (pk > 0.0)
    for (int c = 0; c < kChannels; ++c)
      for (int64_t i = 0; i < frames; ++i) b.channel(c)[i] /= pk;
  return b;
}

// ------------------------------------------------------------- linearity ---
//
// Linearity is BOTH properties, and it takes both to classify correctly:
//
//   homogeneity   f(a*x) == a*f(x)
//   superposition f(x+y) == f(x) + f(y)
//
// Homogeneity alone is not enough, and this is not a theoretical objection —
// it misclassified a real node here. `transient_designer` derives its gain
// from the RATIO of a fast envelope to a slow one, which is scale invariant,
// so it is exactly homogeneous while being thoroughly nonlinear. It measured
// "linear" and was excused from the aliasing requirement until superposition
// was added, at which point it failed by 60 dB.
//
// The converse matters too. A chorus and a plate reverb are time-VARYING but
// linear: they satisfy both properties exactly, generate no harmonics, and
// cannot alias. Classifying them as nonlinear would put two nodes under a
// requirement that is meaningless for them.
//
// Probed at four levels from 0 to -46 dBFS, because a threshold-based node is
// perfectly linear on either side of its threshold and only betrays itself
// across it. One level would classify the gate as linear.

struct LinearityResult {
  bool measured = false;
  double homogeneityDb = -std::numeric_limits<double>::infinity();
  double superpositionDb = -std::numeric_limits<double>::infinity();
  double timeInvarianceDb = -std::numeric_limits<double>::infinity();
  bool timeInvarianceMeasured = false;
  double worstAtDbfs = 0.0;
  double worstAmplitude = 0.0;
  // How far apart the best and worst levels were. An ARGMAX IS ONLY MEANINGFUL
  // IF SOMETHING ACTUALLY WON — see worstLevelIsMeaningful().
  double levelSpreadDb = 0.0;

  double worstDb() const { return std::max(homogeneityDb, superpositionDb); }

  // WHICH LEVEL WAS WORST IS NOT A MEASUREMENT UNLESS THE NODE IS NONLINEAR.
  // For a linear node the homogeneity error is zero at every amplitude, so the
  // winner is decided by whichever float rounded a hair differently — and that
  // differs by platform. CI reported dc_block moving from -33.979 to -20.000
  // dBFS between x86-64 and arm64: linear nodes swapping tie-break winners,
  // looking for all the world like a numeric regression.
  //
  // The first attempt at this gated on the SPREAD between levels being at
  // least 3 dB, and that was wrong for a reason worth keeping: the spread is
  // measured in dB of a quantity sitting at the float noise floor, where -320
  // and -280 dB are forty decibels apart and both mean zero. It passed
  // locally, failed on the other architecture, and failed for exactly the
  // reason it was written to fix.
  //
  // Level-dependence IS nonlinearity. So the question "which level was worst"
  // is only well posed for a node that has already been measured nonlinear,
  // and for a linear one the honest answer is that there is no worst level.
  bool worstLevelIsMeaningful(double thresholdDb) const {
    return measured && worstDb() > thresholdDb;
  }
};

// Worst |a - b| over the settled part of two renders, in dB relative to the
// larger signal. Compared over the middle so nodes with lookahead or a
// backwards reach are read in steady state, not against the edges.
double relativeDeviationDb(const AudioBuffer& a, const AudioBuffer& b) {
  const int64_t n = std::min(a.frames(), b.frames());
  if (n <= 0) return std::nan("");
  double diff = 0.0, ref = 0.0;
  for (int64_t i = n / 4; i < n - n / 4; ++i) {
    diff = std::max(diff, std::abs(a.channel(0)[i] - b.channel(0)[i]));
    ref = std::max(ref, std::abs(b.channel(0)[i]));
  }
  if (!(ref > 0.0)) return std::nan(""); // nothing came out; nothing to say
  // Negative infinity, not a floor. A bit-identical null IS the answer for a
  // node like polarity, and clamping it to a large negative number would be
  // inventing a measurement.
  if (diff == 0.0) return -std::numeric_limits<double>::infinity();
  return dbFromLinear(diff / ref);
}

AudioBuffer scaledCopy(const AudioBuffer& src, double k) {
  AudioBuffer out(src.channels(), src.frames());
  for (int c = 0; c < src.channels(); ++c)
    for (int64_t i = 0; i < src.frames(); ++i) out.channel(c)[i] = k * src.channel(c)[i];
  return out;
}

AudioBuffer sumOf(const AudioBuffer& a, const AudioBuffer& b) {
  const int64_t n = std::min(a.frames(), b.frames());
  AudioBuffer out(a.channels(), n);
  for (int c = 0; c < a.channels(); ++c)
    for (int64_t i = 0; i < n; ++i) out.channel(c)[i] = a.channel(c)[i] + b.channel(c)[i];
  return out;
}

LinearityResult measureLinearity(const Config& cfg) {
  constexpr int64_t kFrames = 32768;
  const double amps[] = {1.0, 0.1, 0.02, 0.005};
  // Two mutually inharmonic probes, so a difference between them cannot be
  // mistaken for one being a harmonic of the other.
  const double f1 = 997.0, f2 = 1523.0;

  LinearityResult r;
  double bestAcrossLevels = std::numeric_limits<double>::infinity();
  double worstAcrossLevels = -std::numeric_limits<double>::infinity();
  for (double a : amps) {
    const AudioBuffer x1 = toneBuffer(kFrames, f1, a * 0.5);
    const AudioBuffer x2 = toneBuffer(kFrames, f2, a * 0.5);
    const AudioBuffer both = sumOf(x1, x2);

    const AudioBuffer outBoth = renderOne(cfg, both, false);
    const AudioBuffer outHalfBoth = renderOne(cfg, scaledCopy(both, 0.5), false);
    const AudioBuffer out1 = renderOne(cfg, x1, false);
    const AudioBuffer out2 = renderOne(cfg, x2, false);

    const double hom = relativeDeviationDb(scaledCopy(outBoth, 0.5), outHalfBoth);
    const double sup = relativeDeviationDb(sumOf(out1, out2), outBoth);
    // NaN means the probe could not be run at this level (nothing came out).
    // -infinity means it ran and the answer was exactly zero, which is the best
    // possible result and must not be discarded.
    if (std::isnan(hom) && std::isnan(sup)) continue;

    const double neg = -std::numeric_limits<double>::infinity();
    const double atThisLevel = std::max(std::isnan(hom) ? neg : hom,
                                        std::isnan(sup) ? neg : sup);
    if (atThisLevel > r.worstDb()) {
      r.worstAmplitude = a;
      r.worstAtDbfs = dbFromLinear(a);
    }
    // Track the range across levels so a tie can be told from a real winner.
    bestAcrossLevels = std::min(bestAcrossLevels, atThisLevel);
    worstAcrossLevels = std::max(worstAcrossLevels, atThisLevel);
    if (!std::isnan(hom)) r.homogeneityDb = std::max(r.homogeneityDb, hom);
    if (!std::isnan(sup)) r.superpositionDb = std::max(r.superpositionDb, sup);
    r.measured = true;
  }
  if (std::isfinite(bestAcrossLevels) && std::isfinite(worstAcrossLevels))
    r.levelSpreadDb = worstAcrossLevels - bestAcrossLevels;

  // Time invariance, the third property. Not part of the nonlinearity verdict —
  // a chorus is linear and cannot alias — but it decides whether the THD and
  // IMD gates apply.
  //
  // A time-VARYING linear node puts its modulation sidebands at f0 +/- k*fmod,
  // and fmod is not commensurate with the analysis window, so those components
  // are the one thing in this suite that coherent sampling cannot make exact.
  // Their rectangular-window skirts land on the harmonic bins: `chorus`
  // measures -74 dB THD and `plate_reverb` -85 dB on a stimulus neither can
  // possibly distort. Gating those as harmonic distortion would be reporting
  // the analysis window.
  {
    constexpr int64_t kShift = 4096;
    constexpr int64_t kBurst = 16384;
    constexpr int64_t kAt = 4096;
    auto burstAt = [&](int64_t offset) {
      AudioBuffer b(kChannels, kFrames);
      for (int64_t i = 0; i < kBurst; ++i) {
        // Raised-cosine ends, so neither copy has an edge discontinuity that a
        // two-sided node would respond to differently.
        const double ramp = 2048.0;
        const double env = 0.5 * (1.0 - std::cos(M_PI * std::min(1.0, std::min(
            static_cast<double>(i) / ramp,
            static_cast<double>(kBurst - 1 - i) / ramp))));
        const double t = static_cast<double>(i) / kFs;
        const double s = 0.25 * env *
                         (std::sin(2.0 * M_PI * f1 * t) + std::sin(2.0 * M_PI * f2 * t));
        for (int c = 0; c < kChannels; ++c) b.channel(c)[offset + i] = s;
      }
      return b;
    };
    const AudioBuffer outA = renderOne(cfg, burstAt(kAt), false);
    const AudioBuffer outB = renderOne(cfg, burstAt(kAt + kShift), false);
    double diff = 0.0, ref = 0.0;
    const int64_t n = std::min(outA.frames() - kShift, outB.frames() - kShift);
    for (int64_t i = 0; i < n; ++i) {
      diff = std::max(diff, std::abs(outA.channel(0)[i] - outB.channel(0)[i + kShift]));
      ref = std::max(ref, std::abs(outA.channel(0)[i]));
    }
    if (ref > 0.0) {
      r.timeInvarianceMeasured = true;
      r.timeInvarianceDb = diff == 0.0
                               ? -std::numeric_limits<double>::infinity()
                               : dbFromLinear(diff / ref);
    }
  }
  return r;
}

// ------------------------------------------------- static gain transfer ---
//
// Output level against input level, on a steady tone, at nine levels from
// -60 dBFS to 0 dBFS. Not one of the five measurements the plan names, and it
// is here because MUTATION TESTING SAID SO: a 5% error in the soft-knee gain
// computer's quadratic term passed every other measurement in this file. It
// could not fail them — the knee spans -23 to -17 dBFS at these settings and
// every other stimulus here sits at 0 dBFS or below -26, so nothing ever
// entered the region the mutation changed.
//
// The levels straddle every threshold in the registry deliberately: -20 dBFS
// for the compressor, -35 for the gate, -30 and -40 for the dynamic EQ, -18
// and -22 and -24 for the multiband. A dynamics node's law is the thing it is
// FOR, and a suite that measures its distortion but not its gain is measuring
// the wrong half.
struct GainTransfer {
  bool measured = false;
  std::vector<double> inputDbfs;
  std::vector<double> outputDbfs;
};

GainTransfer measureGainTransfer(const Config& cfg) {
  const double levels[] = {-60.0, -45.0, -36.0, -30.0, -24.0,
                           -18.0, -12.0, -6.0, 0.0};
  // Long enough for every attack and release in the registry to settle, short
  // enough that nine of them cost less than one aliasing step.
  constexpr int64_t kFrames = 65536;
  const double f0 = 997.0;

  GainTransfer g;
  for (double db : levels) {
    const double amp = std::pow(10.0, db / 20.0);
    const AudioBuffer out = renderOne(cfg, toneBuffer(kFrames, f0, amp), false);
    if (out.frames() < kFrames) return g;
    // Steady-state peak, read from the middle so neither the attack nor the
    // end of the tone is in the window.
    double peak = 0.0;
    for (int64_t i = kFrames / 2; i < kFrames - kFrames / 8; ++i)
      peak = std::max(peak, std::abs(out.channel(0)[i]));
    g.inputDbfs.push_back(db);
    g.outputDbfs.push_back(peak > 0.0 ? dbFromLinear(peak)
                                      : -std::numeric_limits<double>::infinity());
  }
  g.measured = true;
  return g;
}

// -100 dB of homogeneity error is the line. Below it a node cannot be
// generating enough nonlinear product to matter at the level this suite gates
// at; above it, it is a nonlinear node whether or not anybody labelled it one.
constexpr double kLinearityThresholdDb = -100.0;

// The requirement the plan states, verbatim: aliasing at or below -100 dB at
// 0 dBFS, for every nonlinear node. It is not adjusted anywhere in this file.
constexpr double kAsrRequirementDb = -100.0;

// ------------------------------------------------------ harness self-test --
//
// Nothing below is trustworthy if these fail, so they gate the whole run. The
// rule they encode is from findings.md: verify the harness before trusting a
// negative result. Three separate mutations were once recorded as "not caught"
// because a grep in the tooling was wrong, not because the tests were.

struct SelfCheck {
  std::string label;
  std::string detail;
  Value value;
  std::string unit;
  double threshold = 0.0;
  bool pass = false;
};

std::vector<SelfCheck> selfValidation(const ReferenceTruePeak& meter) {
  std::vector<SelfCheck> checks;

  {
    SelfCheck c;
    c.label = "denormal flushing active";
    c.detail = "FTZ/DAZ changes numeric results, so a suite that measured "
               "without it measured a different program than the one that ships";
    c.value = Value::of(ScopedNoDenormals::enabled() ? 1.0 : 0.0);
    c.unit = "";
    c.threshold = 1.0;
    c.pass = ScopedNoDenormals::enabled();
    checks.push_back(c);
  }

  {
    // Coherent sampling leakage. A tone on an exact bin, through a bypassed
    // chain, must put every other bin at the arithmetic floor. If this is not
    // true then every THD and ASR number below is leakage-limited and the
    // -100 dB threshold is measuring the window rather than the node.
    constexpr size_t kN = 65536;
    const int k = nearestOddBin(997.0, kN);
    const double f0 = static_cast<double>(k) * kFs / static_cast<double>(kN);
    const AudioBuffer in = toneBuffer(static_cast<int64_t>(kN), f0, 1.0);
    const auto p = powerSpectrum(in.channel(0), kN);
    double worst = 0.0;
    for (size_t b = 1; b <= kN / 2; ++b)
      if (b != static_cast<size_t>(k)) worst = std::max(worst, p[b]);
    const double db = linToDbPower(worst / p[static_cast<size_t>(k)]);
    SelfCheck c;
    c.label = "coherent sampling leakage";
    c.detail = "worst non-fundamental bin of an exactly-coherent tone, relative "
               "to the fundamental. This is the noise floor every THD and ASR "
               "number below sits on";
    c.value = Value::of(db);
    c.unit = "dB";
    // -220 dB, and the number is chosen from what the transform can do rather
    // than from what would look impressive. This radix-2 FFT's own round-off
    // floor at N=65536 sits at about -249.5 dB, so a threshold of -250 would
    // fail on arithmetic; -220 is comfortably above that and still leaves
    // 120 dB of headroom under the -100 dB the requirement is stated at, which
    // is the only property that matters here. A leakage floor that is not at
    // least ~40 dB below the gate would mean the gate measures the window.
    c.threshold = -220.0;
    c.pass = db <= c.threshold;
    checks.push_back(c);
  }

  {
    // THE ASR CLASSIFIER, AGAINST A SPECTRUM WHOSE ANSWER IS KNOWN.
    //
    // A signal is built by hand: a coherent fundamental at amplitude 1, plus a
    // single component at amplitude 1e-5 placed exactly where the fifth
    // harmonic of that fundamental folds to. The fifth harmonic of 19.5 kHz is
    // 97.5 kHz, which folds to 1.5 kHz — inside the audible band and nowhere
    // near any legitimate harmonic. The in-band ASR of that spectrum is
    // therefore exactly -100.000 dB, by construction.
    //
    // This is the check that the bin classification, the audible-band cut and
    // the choice of denominator are all RIGHT, rather than merely
    // self-consistent. Every one of them could be wrong in a way that no node
    // measurement would reveal, because a node's true aliasing is not known
    // independently — which is precisely the trap findings.md records under
    // "insufficiently independent references".
    const int k = nearestOddBin(19500.0, kAsrN);
    const double f0 = static_cast<double>(k) * kFs / static_cast<double>(kAsrN);
    size_t foldBin = (5u * static_cast<size_t>(k)) % kAsrN;
    if (foldBin > kAsrN / 2) foldBin = kAsrN - foldBin;
    const double foldHz = static_cast<double>(foldBin) * kFs / static_cast<double>(kAsrN);
    const double injected = 1e-5; // -100 dB relative to the fundamental

    std::vector<double> x(kAsrN);
    for (size_t i = 0; i < kAsrN; ++i) {
      const double t = static_cast<double>(i) / kFs;
      x[i] = std::sin(2.0 * M_PI * f0 * t) +
             injected * std::sin(2.0 * M_PI * foldHz * t + 0.7);
    }
    ToneSpectrum s;
    s.n = kAsrN;
    s.hzPerBin = kFs / static_cast<double>(kAsrN);
    s.power = powerSpectrum(x.data(), kAsrN);
    bool collisionFree = true;
    const AsrStep st = classifyAsr(s, k, collisionFree);
    const double err = std::abs(st.inBandDb - (-100.0));

    SelfCheck c;
    c.label = "ASR classifier vs a synthetic alias";
    c.detail = "a fundamental plus one component at -100.000 dB, placed exactly "
               "where the fifth harmonic folds to. The classifier must report "
               "-100.000 dB and nothing else";
    c.value = Value::of(err);
    c.unit = "dB";
    c.threshold = 0.01;
    c.pass = collisionFree && err <= c.threshold;
    checks.push_back(c);
  }

  {
    // The reference true-peak meter, against the analytic answer. A sampled
    // unit sinusoid reconstructs to exactly 1.0, at every frequency and phase.
    const double freqs[] = {100.0, 997.0, 5000.0, 11025.0, 19000.0,
                            24000.0, 33333.0, 40000.0};
    const double phases[] = {0.0, M_PI / 4.0, M_PI / 3.0, M_PI / 2.0, 1.1};
    double worst = 0.0;
    for (double f : freqs) {
      for (double ph : phases) {
        constexpr int64_t kN = 8192;
        std::vector<double> x(kN);
        for (int64_t i = 0; i < kN; ++i)
          x[static_cast<size_t>(i)] =
              std::sin(2.0 * M_PI * f * static_cast<double>(i) / kFs + ph);
        const double db = dbFromLinear(meter.measure(x.data(), kN, meter.guardSamples()));
        worst = std::max(worst, std::abs(db));
      }
    }
    SelfCheck c;
    c.label = "reference true-peak meter vs analytic";
    c.detail = "worst |error| against 0.000 dBTP over 8 frequencies x 5 phases. "
               "The reference is independent of the engine's meter; this is what "
               "makes it a reference rather than a second opinion";
    c.value = Value::of(worst);
    c.unit = "dB";
    c.threshold = 0.01;
    c.pass = worst <= c.threshold;
    checks.push_back(c);
  }

  {
    // FFT round-trip. Cheap, and the transform is under every number here.
    constexpr size_t kN = 8192;
    std::vector<std::complex<double>> a(kN);
    CounterRng rng(7);
    std::vector<double> orig(kN);
    for (size_t i = 0; i < kN; ++i) {
      orig[i] = rng.bipolar(0, static_cast<int64_t>(i));
      a[i] = {orig[i], 0.0};
    }
    fftInPlace(a);
    ifftInPlace(a);
    double worst = 0.0;
    for (size_t i = 0; i < kN; ++i) worst = std::max(worst, std::abs(a[i].real() - orig[i]));
    SelfCheck c;
    c.label = "FFT round-trip error";
    c.detail = "forward then inverse on full-scale noise";
    c.value = Value::of(dbFromLinear(std::max(worst, 1e-300)));
    c.unit = "dBFS";
    c.threshold = -250.0;
    c.pass = c.value.usable() && c.value.v <= c.threshold;
    checks.push_back(c);
  }

  return checks;
}

// ------------------------------------------------------------- formatting --

std::string fmt(const Value& v, const std::string& unit) {
  if (!v.present) return "&mdash;";
  if (v.negInf) return "0 <span class=\"sub\">exact</span>";
  char buf[64];
  if (unit == "%") {
    std::snprintf(buf, sizeof buf, "%.4g", v.v);
  } else if (unit == "Hz") {
    std::snprintf(buf, sizeof buf, "%.0f", v.v);
  } else if (std::abs(v.v) < 0.01 && v.v != 0.0) {
    std::snprintf(buf, sizeof buf, "%.2e", v.v);
  } else {
    std::snprintf(buf, sizeof buf, "%.2f", v.v);
  }
  return std::string(buf);
}

std::string jsonNum(const Value& v) {
  // An exactly-zero ratio serialises as null and is flagged separately, so
  // nothing downstream can mistake it for "0 dB of error".
  if (!v.present || v.negInf) return "null";
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.10g", v.v);
  return std::string(buf);
}

std::string jsonEscape(const std::string& s) {
  std::string out;
  for (char ch : s) {
    switch (ch) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      default: out += ch;
    }
  }
  return out;
}

std::string htmlEscape(const std::string& s) {
  std::string out;
  for (char ch : s) {
    switch (ch) {
      case '<': out += "&lt;"; break;
      case '>': out += "&gt;"; break;
      case '&': out += "&amp;"; break;
      default: out += ch;
    }
  }
  return out;
}

// ------------------------------------------------------------------ emit ---

struct Section {
  std::string title;
  std::string blurb;
  std::vector<std::string> keys;
};

const char* kStyle = R"CSS(
:root{--bg:#fbfbfa;--fg:#1a1a1a;--dim:#6b6b6b;--line:#e2e0dc;--card:#fff;
--pass:#1a7f4b;--fail:#c02626;--warn:#a06a00;--accent:#2b5eb8;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
@media (prefers-color-scheme:dark){:root{--bg:#151516;--fg:#e8e6e3;--dim:#9a9793;
--line:#2e2e30;--card:#1d1d1f;--pass:#4ec87f;--fail:#ff6b6b;--warn:#e0a83c;--accent:#7aa5f0}}
*{box-sizing:border-box}
body{margin:0;padding:2.5rem 1.5rem 6rem;background:var(--bg);color:var(--fg);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto}
h1{font-size:1.7rem;margin:0 0 .3rem;letter-spacing:-.02em}
h2{font-size:1.12rem;margin:2.8rem 0 .35rem;letter-spacing:-.01em}
h2 .n{color:var(--dim);font-weight:400;margin-right:.5rem}
p.blurb{color:var(--dim);margin:.2rem 0 1rem;max-width:76ch;font-size:.9rem}
.meta{color:var(--dim);font-size:.85rem;margin-bottom:1.6rem}
.meta code{font-family:var(--mono)}
.cards{display:flex;flex-wrap:wrap;gap:.7rem;margin:1.4rem 0 .5rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:9px;
padding:.75rem 1.05rem;min-width:9.5rem}
.card .k{font-size:.72rem;text-transform:uppercase;letter-spacing:.07em;color:var(--dim)}
.card .v{font-size:1.45rem;font-family:var(--mono);margin-top:.15rem}
.card.bad .v{color:var(--fail)} .card.good .v{color:var(--pass)}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:9px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:.86rem}
th,td{padding:.5rem .8rem;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}
th{font-weight:600;font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;
color:var(--dim);text-align:right;position:sticky;top:0;background:var(--card)}
th:first-child,td:first-child{text-align:left}
tr:last-child td{border-bottom:none}
td.num{font-family:var(--mono)}
td .u{color:var(--dim);font-size:.78em;margin-left:.2em}
td.na{color:var(--dim)}
.badge{display:inline-block;padding:.08rem .42rem;border-radius:4px;font-size:.7rem;
font-weight:600;letter-spacing:.03em}
.badge.pass{background:color-mix(in srgb,var(--pass) 16%,transparent);color:var(--pass)}
.badge.fail{background:color-mix(in srgb,var(--fail) 16%,transparent);color:var(--fail)}
.badge.na{background:color-mix(in srgb,var(--dim) 14%,transparent);color:var(--dim)}
.sub{color:var(--dim);font-size:.78rem;font-weight:400}
.note{border-left:3px solid var(--accent);padding:.55rem .9rem;margin:1rem 0;
background:var(--card);border-radius:0 7px 7px 0;font-size:.86rem;color:var(--dim)}
.note.fail{border-color:var(--fail)}
.note b{color:var(--fg)}
footer{margin-top:4rem;color:var(--dim);font-size:.8rem;border-top:1px solid var(--line);padding-top:1rem}
)CSS";

void writeCell(std::ofstream& f, const Metric& m) {
  if (!m.applicable) {
    f << "<td class=\"na\" title=\"" << htmlEscape(m.naReason) << "\">n/a</td>";
    return;
  }
  f << "<td class=\"num\">" << fmt(m.value, m.unit);
  if (!m.unit.empty() && m.value.present && !m.value.negInf)
    f << "<span class=\"u\">" << htmlEscape(m.unit) << "</span>";
  if (m.hasThreshold)
    f << " <span class=\"badge " << (m.pass ? "pass\">PASS" : "fail\">FAIL")
      << "</span>";
  f << "</td>";
}

void writeSection(std::ofstream& f, int number, const Section& sec,
                  const std::vector<ConfigReport>& reports) {
  f << "<h2><span class=\"n\">" << number << "</span>" << htmlEscape(sec.title)
    << "</h2>\n<p class=\"blurb\">" << sec.blurb << "</p>\n";
  f << "<div class=\"scroll\"><table><thead><tr><th>node</th>";
  size_t columns = 0;
  for (const auto& key : sec.keys) {
    // The heading is taken from a config where the metric is GATED if any has
    // one. Taking the first match instead lost every threshold annotation in
    // the aliasing table, because the first node in the registry is `gain`,
    // for which the measurement does not apply.
    const Metric* heading = nullptr;
    for (const auto& r : reports) {
      const Metric* m = r.find(key);
      if (!m) continue;
      if (!heading || (m->hasThreshold && !heading->hasThreshold)) heading = m;
      if (heading->hasThreshold) break;
    }
    if (!heading) continue;
    ++columns;
    f << "<th>" << htmlEscape(heading->label);
    if (heading->hasThreshold)
      f << "<br><span class=\"sub\">threshold "
        << (heading->lowerIsBetter ? "&le; " : "&ge; ") << heading->threshold
        << " " << htmlEscape(heading->unit) << "</span>";
    f << "</th>";
  }
  f << "</tr></thead><tbody>\n";
  for (const auto& r : reports) {
    bool present = false, anyApplicable = false;
    std::string why;
    for (const auto& key : sec.keys) {
      const Metric* m = r.find(key);
      if (!m) continue;
      present = true;
      if (m->applicable) anyApplicable = true;
      else if (why.empty()) why = m->naReason;
    }
    if (!present) continue;
    f << "<tr><td>" << htmlEscape(r.configId)
      << "<br><span class=\"sub\">" << htmlEscape(r.paramSummary) << "</span></td>";
    if (!anyApplicable) {
      // One cell saying why, rather than eleven columns of em dashes. The
      // reason is the informative part: it says what was measured about the
      // node that excused it.
      f << "<td class=\"na\" colspan=\"" << columns << "\">n/a &mdash; "
        << htmlEscape(why) << "</td></tr>\n";
      continue;
    }
    for (const auto& key : sec.keys) {
      const Metric* m = r.find(key);
      if (!m) { f << "<td class=\"na\">&mdash;</td>"; continue; }
      writeCell(f, *m);
    }
    f << "</tr>\n";
  }
  f << "</tbody></table></div>\n";
}

} // namespace

// ------------------------------------------------------------------ main ---

int main(int argc, char** argv) {
  // Exactly the numeric mode the render loop runs in. Not optional: FTZ/DAZ
  // changes results, so measuring without it measures a different program.
  ScopedNoDenormals noDenormals;

  const char* envDir = std::getenv("ANV_ENGINE_DIR");
  if (envDir) {
    g_engineDir = envDir;
  } else if (std::filesystem::exists("include/anv/io/Recipe.h")) {
    g_engineDir = ".";
  } else if (std::filesystem::exists("engine/include/anv/io/Recipe.h")) {
    g_engineDir = "engine";
  } else {
    std::fprintf(stderr,
                 "measurements: cannot locate the engine source tree. Run from "
                 "engine/ or set ANV_ENGINE_DIR.\n");
    return 2;
  }
  const std::string outDir = g_engineDir + "/build";
  std::filesystem::create_directories(outDir);
  (void)argc; (void)argv;

  std::string catalogError;
  const std::vector<std::string> types = nodeCatalog(catalogError);
  if (types.empty()) {
    std::fprintf(stderr, "measurements: %s\n", catalogError.c_str());
    return 2;
  }
  std::printf("node catalog: %zu types read from Recipe.h::makeNode\n", types.size());

  const ReferenceTruePeak meter;
  const std::vector<SelfCheck> selfChecks = selfValidation(meter);
  bool harnessOk = true;
  std::printf("\nharness self-validation\n");
  for (const auto& c : selfChecks) {
    std::printf("  %-42s %10s %-4s  %s\n", c.label.c_str(),
                fmt(c.value, c.unit).c_str(), c.unit.c_str(),
                c.pass ? "ok" : "FAIL");
    if (!c.pass) harnessOk = false;
  }
  if (!harnessOk) {
    std::fprintf(stderr,
                 "\nmeasurements: the harness failed its own validation. No "
                 "number it produced would mean anything, so nothing is "
                 "written.\n");
    return 2;
  }

  const std::vector<Config> configs = buildConfigs(types);
  std::vector<ConfigReport> reports;

  // The true-peak stress signal is shared, so every node's true-peak number is
  // comparable against every other's.
  const AudioBuffer stress = truePeakStress(16384);
  const double stressRefTp = meter.measure(stress.channel(0), stress.frames(), 0);

  std::printf("\nmeasuring %zu configurations\n", configs.size());
  const auto t0 = std::chrono::steady_clock::now();

  for (const auto& cfg : configs) {
    const auto tStart = std::chrono::steady_clock::now();
    ConfigReport rep;
    rep.configId = cfg.id;
    rep.nodeType = cfg.type;
    rep.paramSummary = cfg.summary;

    // --- linearity, which decides what else applies ---
    const LinearityResult lin = measureLinearity(cfg);
    rep.linearityKnown = lin.measured;
    rep.linear = lin.measured && lin.worstDb() <= kLinearityThresholdDb;
    const bool timeInvariant =
        lin.timeInvarianceMeasured && lin.timeInvarianceDb <= kLinearityThresholdDb;
    const bool lti = rep.linear && timeInvariant;
    rep.add(plain("homogeneity_error_db", "homogeneity error", "dB",
                  lin.measured ? Value::of(lin.homogeneityDb) : Value::none()));
    rep.add(plain("superposition_error_db", "superposition error", "dB",
                  lin.measured ? Value::of(lin.superpositionDb) : Value::none()));
    rep.add(plain("time_invariance_error_db", "time-invariance error", "dB",
                  lin.timeInvarianceMeasured ? Value::of(lin.timeInvarianceDb)
                                             : Value::none()));
    rep.add(plain("homogeneity_worst_level_dbfs", "worst at", "dBFS",
                  lin.worstLevelIsMeaningful(kLinearityThresholdDb)
                      ? Value::of(lin.worstAtDbfs) : Value::none()));
    {
      Metric m = plain("classified_nonlinear", "classified", "",
                       Value::of(rep.linear ? 0.0 : 1.0));
      m.note = rep.linear ? (lti ? "LTI" : "linear, time-varying") : "nonlinear";
      rep.add(m);
    }

    // --- static gain transfer ---
    {
      const GainTransfer g = measureGainTransfer(cfg);
      for (size_t i = 0; i < g.inputDbfs.size(); ++i) {
        char key[64], label[64];
        std::snprintf(key, sizeof key, "gain_transfer_%+03.0f_dbfs", g.inputDbfs[i]);
        std::snprintf(label, sizeof label, "in %.0f dBFS", g.inputDbfs[i]);
        rep.add(plain(key, label, "dBFS", Value::of(g.outputDbfs[i])));
      }
    }

    // --- THD, coherent sampling, 0 dBFS ---
    const ThdResult thd = measureThd(cfg, 1.0);
    rep.add(plain("thd_fundamental_dbfs", "fundamental", "dBFS", thd.fundamentalDbfs));
    if (lti) {
      // An LTI node must not distort. This is the only universally defensible
      // THD threshold: an EQ is allowed to change the level of a tone and a
      // saturator is SUPPOSED to distort, but a node that satisfies linearity
      // and time invariance and still produces harmonics is broken.
      rep.add(gated("thd_db", "THD", "dB", thd.thdDb, -100.0));
    } else {
      Metric m = plain("thd_db", "THD", "dB", thd.thdDb);
      m.note = rep.linear
                   ? "linear but time-varying: this number is the leakage of "
                     "modulation sidebands, not harmonic distortion"
                   : "nonlinear by design; reported, not gated";
      rep.add(m);
    }
    rep.add(plain("thd_percent", "THD", "%", thd.thdPercent));
    rep.add(plain("thd_n_db", "THD+N", "dB", thd.thdNDb));
    rep.add(plain("thd_settle_frames", "settle used", "frames", thd.settleFrames));

    // --- ASR, the headline ---
    if (rep.linear) {
      rep.add(notApplicable("asr_inband_db", "in-band ASR", "dB",
                            "measured linear — both homogeneous and additive to "
                            "better than -100 dB — so it generates no harmonics "
                            "and has nothing that could fold"));
    } else {
      const AsrResult asr = measureAsr(cfg, 1.0);
      Metric m = gated("asr_inband_db", "in-band ASR @ 0 dBFS", "dB",
                       asr.measured ? Value::of(asr.worstInBandDb) : Value::none(),
                       kAsrRequirementDb);
      rep.add(m);
      rep.add(plain("asr_fullband_db", "full-band ASR", "dB",
                    asr.measured ? Value::of(asr.worstFullBandDb) : Value::none()));
      rep.add(plain("asr_vs_harmonics_db", "vs harmonic sum", "dB",
                    asr.measured ? Value::of(asr.worstVsHarmonicsDb) : Value::none()));
      rep.add(plain("asr_worst_f0_hz", "worst at f0", "Hz",
                    asr.measured ? Value::of(asr.worstAtHz) : Value::none()));
      rep.add(plain("asr_noise_floor_db", "noise floor in alias bins", "dB",
                    asr.measured ? Value::of(asr.noiseFloorAtWorstDb) : Value::none()));
      {
        // How much of the output was NOT exactly periodic. When this is not
        // well below the reported ASR, the ASR is an upper bound rather than a
        // measurement — leakage from the residual lands in the alias bins.
        Metric resid = plain("asr_nonperiodic_residual_db", "non-periodic residual", "dB",
                         asr.measured ? Value::of(asr.worstResidualDb) : Value::none());
        if (asr.measured && !asr.allConverged)
          resid.note = "did not converge below the -120 dB target even at the "
                       "longest settle; the ASR above is an upper bound";
        rep.add(resid);
      }
      rep.add(plain("asr_settle_frames", "settle used", "frames",
                    asr.measured ? Value::of(static_cast<double>(asr.longestSettle))
                                 : Value::none()));
      {
        Metric bins = plain("asr_collision_free", "bins unambiguous", "",
                            Value::of(asr.collisionFree ? 1.0 : 0.0));
        bins.hasThreshold = true;
        bins.threshold = 1.0;
        bins.lowerIsBetter = false;
        bins.pass = asr.collisionFree;
        rep.add(bins);
      }

      // Second probe, at whichever level the linearity sweep found the node
      // most nonlinear. A threshold-based node is nearly linear at 0 dBFS —
      // the gate is wide open there — so the requirement as literally stated
      // would certify it without ever exercising the mechanism that aliases.
      if (lin.measured && lin.worstAmplitude < 0.999) {
        const AsrResult worst = measureAsr(cfg, lin.worstAmplitude);
        rep.add(plain("asr_inband_worstlevel_db", "in-band ASR @ worst level", "dB",
                      worst.measured ? Value::of(worst.worstInBandDb) : Value::none()));
        rep.add(plain("asr_worstlevel_dbfs", "that level", "dBFS",
                      Value::of(lin.worstAtDbfs)));
      }
    }

    // --- IMD ---
    const ImdResult imd = measureImd(cfg);
    if (lti)
      rep.add(gated("imd_d2_db", "IMD d2 (1 kHz)", "dB", imd.d2Db, -100.0));
    else
      rep.add(plain("imd_d2_db", "IMD d2 (1 kHz)", "dB", imd.d2Db));
    rep.add(plain("imd_d2_percent", "IMD d2", "%", imd.d2Percent));
    rep.add(plain("imd_d3_db", "IMD d3 (18/21 kHz)", "dB", imd.d3Db));

    // --- Farina ---
    const FarinaResult fr = measureFarina(cfg);
    if (fr.measured) {
      auto at = [&](double f) {
        for (size_t i = 0; i < fr.freqHz.size(); ++i)
          if (std::abs(fr.freqHz[i] - f) < f * 0.05) return i;
        return fr.freqHz.size();
      };
      const size_t i100 = at(125.0), i1k = at(1000.0), i10k = at(8000.0);
      auto pick = [&](const std::vector<double>& v, size_t i) {
        return i < v.size() ? Value::of(v[i]) : Value::none();
      };
      rep.add(plain("fr_mag_125hz_db", "|H| 125 Hz", "dB", pick(fr.magDb, i100)));
      rep.add(plain("fr_mag_1khz_db", "|H| 1 kHz", "dB", pick(fr.magDb, i1k)));
      rep.add(plain("fr_mag_8khz_db", "|H| 8 kHz", "dB", pick(fr.magDb, i10k)));
      rep.add(plain("fr_max_mag_db", "max |H| 20 Hz-20 kHz", "dB", Value::of(fr.maxMagDb)));
      rep.add(plain("fr_min_mag_db", "min |H| 20 Hz-20 kHz", "dB", Value::of(fr.minMagDb)));
      rep.add(plain("fr_phase_1khz_deg", "phase 1 kHz", "deg", pick(fr.phaseDeg, i1k)));
      rep.add(plain("fr_gd_1khz_samples", "group delay 1 kHz", "samples",
                    pick(fr.groupDelaySamples, i1k)));
      rep.add(plain("fr_gd_max_samples", "max |group delay|", "samples",
                    Value::of(fr.maxAbsGroupDelay)));
      rep.add(plain("fr_pre_arrival_db", "pre-arrival excess vs null", "dB",
                    Value::of(fr.preArrivalDb)));
    }

    // --- true peak ---
    {
      const AudioBuffer out = renderOne(cfg, stress, false);
      const double refLin = meter.measure(out.channel(0), std::min(out.frames(), stress.frames() + 4096), 0);
      const double engLin = engineTruePeak(out.channel(0), out.frames());
      double samplePk = 0.0;
      for (int64_t i = 0; i < out.frames(); ++i)
        samplePk = std::max(samplePk, std::abs(out.channel(0)[i]));

      rep.add(plain("tp_sample_peak_dbfs", "sample peak", "dBFS",
                    Value::of(dbFromLinear(samplePk))));
      rep.add(plain("tp_reference_dbtp", "true peak (reference)", "dBTP",
                    Value::of(dbFromLinear(refLin))));
      rep.add(plain("tp_engine_meter_dbtp", "true peak (engine meter)", "dBTP",
                    Value::of(dbFromLinear(engLin))));
      rep.add(plain("tp_meter_delta_db", "engine - reference", "dB",
                    Value::of(dbFromLinear(engLin) - dbFromLinear(refLin))));

      if (cfg.type == "true_peak_limiter") {
        const double ceiling = cfg.params.num("ceiling_dbtp", -1.0);
        // The tolerance is the reference meter's own validated error, not a
        // number picked to make the node pass. It is 0.01 dB because that is
        // what selfValidation just proved the meter to.
        Metric m = gated("tp_conformance_dbtp", "output true peak vs ceiling", "dBTP",
                         Value::of(dbFromLinear(refLin)), ceiling + 0.01);
        m.note = "measured by the independent reference meter, never the "
                 "engine's own";
        rep.add(m);
      }
    }

    const auto tEnd = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(tEnd - tStart).count();
    std::printf("  %-22s %s  %5.1fs\n", cfg.id.c_str(),
                rep.linear ? "linear   " : "NONLINEAR", secs);
    reports.push_back(std::move(rep));
  }

  const double totalSecs =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  // ------------------------------------------------------------- summary --
  int failures = 0, gatedCount = 0;
  std::vector<std::string> failingRows;
  for (const auto& r : reports)
    for (const auto& m : r.metrics)
      if (m.hasThreshold && m.applicable) {
        ++gatedCount;
        if (!m.pass) {
          ++failures;
          failingRows.push_back(r.configId + " / " + m.label + " = " +
                                fmt(m.value, m.unit) + " " + m.unit +
                                " (threshold " + fmt(Value::of(m.threshold), m.unit) +
                                ")");
        }
      }

  // -------------------------------------------------------------- output --
#ifdef ANV_GIT_COMMIT
  const std::string commit = ANV_GIT_COMMIT;
#else
  const std::string commit;
#endif
  char stamp[64];
  {
    const std::time_t now = std::time(nullptr);
    std::tm tmv{};
#if defined(_WIN32)
    gmtime_s(&tmv, &now);
#else
    gmtime_r(&now, &tmv);
#endif
    std::strftime(stamp, sizeof stamp, "%Y-%m-%dT%H:%M:%SZ", &tmv);
  }

  const std::string htmlPath = outDir + "/measurements.html";
  const std::string jsonPath = outDir + "/measurements.json";

  // ---- JSON ----
  {
    std::ofstream f(jsonPath);
    f << "{\n  \"schema_version\": 1,\n";
    f << "  \"generated_at_utc\": \"" << stamp << "\",\n";
    f << "  \"git_commit\": " << (commit.empty() ? "null" : "\"" + jsonEscape(commit) + "\"") << ",\n";
    f << "  \"sample_rate_hz\": " << kFs << ",\n";
    f << "  \"asr_requirement_db\": " << kAsrRequirementDb << ",\n";
    f << "  \"node_types_in_catalog\": " << types.size() << ",\n";
    f << "  \"self_checks\": [\n";
    for (size_t i = 0; i < selfChecks.size(); ++i) {
      const auto& c = selfChecks[i];
      f << "    {\"label\": \"" << jsonEscape(c.label) << "\", \"value\": "
        << jsonNum(c.value) << ", \"unit\": \"" << jsonEscape(c.unit)
        << "\", \"threshold\": " << c.threshold << ", \"pass\": "
        << (c.pass ? "true" : "false") << "}" << (i + 1 < selfChecks.size() ? "," : "")
        << "\n";
    }
    f << "  ],\n  \"configs\": [\n";
    for (size_t i = 0; i < reports.size(); ++i) {
      const auto& r = reports[i];
      f << "    {\n      \"config\": \"" << jsonEscape(r.configId) << "\",\n";
      f << "      \"node_type\": \"" << jsonEscape(r.nodeType) << "\",\n";
      f << "      \"params\": \"" << jsonEscape(r.paramSummary) << "\",\n";
      f << "      \"linear\": " << (r.linear ? "true" : "false") << ",\n";
      f << "      \"metrics\": {\n";
      for (size_t j = 0; j < r.metrics.size(); ++j) {
        const auto& m = r.metrics[j];
        f << "        \"" << jsonEscape(m.key) << "\": {"
          << "\"value\": " << jsonNum(m.value)
          << ", \"unit\": \"" << jsonEscape(m.unit) << "\""
          << ", \"exactly_zero\": " << (m.value.negInf ? "true" : "false")
          << ", \"applicable\": " << (m.applicable ? "true" : "false");
        if (!m.applicable) f << ", \"na_reason\": \"" << jsonEscape(m.naReason) << "\"";
        if (m.hasThreshold)
          f << ", \"threshold\": " << m.threshold
            << ", \"lower_is_better\": " << (m.lowerIsBetter ? "true" : "false")
            << ", \"pass\": " << (m.pass ? "true" : "false");
        f << "}" << (j + 1 < r.metrics.size() ? "," : "") << "\n";
      }
      f << "      }\n    }" << (i + 1 < reports.size() ? "," : "") << "\n";
    }
    f << "  ]\n}\n";
  }

  // ---- HTML ----
  {
    std::ofstream f(htmlPath);
    f << "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
      << "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
      << "<title>Audionova DSP measurements</title><style>" << kStyle
      << "</style></head><body><div class=\"wrap\">\n";
    f << "<h1>Audionova DSP measurements</h1>\n";
    f << "<div class=\"meta\">" << stamp << " &middot; commit <code>"
      << (commit.empty() ? "unknown" : htmlEscape(commit)) << "</code> &middot; "
      << types.size() << " node types, " << reports.size()
      << " configurations &middot; " << static_cast<int>(totalSecs) << " s</div>\n";

    f << "<div class=\"cards\">";
    f << "<div class=\"card " << (failures ? "bad" : "good") << "\"><div class=\"k\">gated checks</div><div class=\"v\">"
      << (gatedCount - failures) << " / " << gatedCount << "</div></div>";
    int nonlinear = 0;
    for (const auto& r : reports) if (!r.linear) ++nonlinear;
    f << "<div class=\"card\"><div class=\"k\">nonlinear nodes</div><div class=\"v\">"
      << nonlinear << "</div></div>";
    f << "<div class=\"card\"><div class=\"k\">ASR requirement</div><div class=\"v\">"
      << kAsrRequirementDb << " dB</div></div>";
    f << "<div class=\"card\"><div class=\"k\">stress true peak</div><div class=\"v\">"
      << fmt(Value::of(dbFromLinear(stressRefTp)), "dBTP") << "</div></div>";
    f << "</div>\n";

    if (failures) {
      f << "<div class=\"note fail\"><b>" << failures
        << " gated measurement" << (failures == 1 ? "" : "s")
        << " below threshold.</b><ul>";
      for (const auto& s : failingRows) f << "<li>" << htmlEscape(s) << "</li>";
      f << "</ul>These are reported, not accommodated. The thresholds in this "
           "suite are the ones the plan states; none has been moved to turn a "
           "row green.</div>\n";
    }

    f << "<h2><span class=\"n\">0</span>Harness self-validation</h2>\n"
      << "<p class=\"blurb\">Nothing below is trustworthy if these fail, so the "
         "suite refuses to emit a report when they do. A measurement rig that "
         "has not been checked reports the rig, not the engine.</p>\n"
      << "<div class=\"scroll\"><table><thead><tr><th>check</th><th>measured</th>"
         "<th>threshold</th><th></th></tr></thead><tbody>\n";
    for (const auto& c : selfChecks) {
      f << "<tr><td>" << htmlEscape(c.label) << "<br><span class=\"sub\">"
        << htmlEscape(c.detail) << "</span></td><td class=\"num\">"
        << fmt(c.value, c.unit) << "<span class=\"u\">" << htmlEscape(c.unit)
        << "</span></td><td class=\"num\">" << c.threshold << "</td><td><span class=\"badge "
        << (c.pass ? "pass\">PASS" : "fail\">FAIL") << "</span></td></tr>\n";
    }
    f << "</tbody></table></div>\n";

    const std::vector<Section> sections = {
        {"Linearity classification",
         "Which nodes are subject to the aliasing requirement is <em>measured</em>, "
         "not listed. A node is probed for homogeneity &mdash; scale the input by two, "
         "does the output scale by two? &mdash; and for superposition, at four levels from "
         "0 to &minus;46 dBFS. Four levels because a threshold-based node is perfectly "
         "linear on either side of its threshold and only betrays itself across it. Both "
         "properties because neither is sufficient: <code>transient_designer</code> derives "
         "its gain from a ratio of envelopes, so it is exactly homogeneous while being "
         "thoroughly nonlinear, and measured &ldquo;linear&rdquo; here until superposition "
         "was added. Conversely a chorus and a plate reverb are time-<em>varying</em> but "
         "linear, satisfy both, and correctly stay out of the aliasing requirement. "
         "Anything worse than &minus;100 dB on either is nonlinear, whatever it is called.",
         {"homogeneity_error_db", "superposition_error_db",
          "homogeneity_worst_level_dbfs", "thd_fundamental_dbfs"}},

        {"Static gain transfer",
         "Output level against input level on a steady 997 Hz tone, at nine levels "
         "straddling every threshold in the registry. This is not one of the five "
         "measurements the plan names &mdash; it is here because mutation testing said so. "
         "A 5% error in the soft-knee gain computer&rsquo;s quadratic term passed every "
         "other measurement in this suite, and could not have failed them: the knee spans "
         "&minus;23 to &minus;17 dBFS and every other stimulus sits at 0 dBFS or below "
         "&minus;26, so nothing ever entered the region the mutation changed.",
         {"gain_transfer_-60_dbfs", "gain_transfer_-45_dbfs", "gain_transfer_-36_dbfs",
          "gain_transfer_-30_dbfs", "gain_transfer_-24_dbfs", "gain_transfer_-18_dbfs",
          "gain_transfer_-12_dbfs", "gain_transfer_-06_dbfs", "gain_transfer_+00_dbfs"}},

        {"Total harmonic distortion (coherent sampling)",
         "A 997 Hz tone at 0 dBFS, placed on an exact FFT bin so an integer number of "
         "cycles fills the window. The transform of such a tone is one non-zero bin with "
         "<em>zero</em> leakage, which is what makes each harmonic level exact rather than "
         "window-dependent. THD counts the true harmonics; THD+N counts everything that is "
         "not the fundamental, so a node that is clean harmonically but noisy shows the "
         "difference. Gated only where the node measured linear &mdash; a saturator is "
         "supposed to distort.",
         {"thd_db", "thd_percent", "thd_n_db", "thd_settle_frames"}},

        {"Aliasing-to-signal ratio",
         "Sweep a sine upward. True harmonics sweep up with it; anything generated above "
         "Nyquist folds and sweeps <em>down</em>. Masking the harmonic loci therefore "
         "leaves exactly the aliasing. Eight steps from 1 kHz to 19.5 kHz, each coherently "
         "sampled so every harmonic occupies one bin and the mask is one bin wide &mdash; "
         "a windowed continuous sweep could not resolve &minus;100 dB through its own "
         "leakage skirts. Reported against the fundamental, which is the signal that went "
         "in. <b>Requirement: &minus;100 dB at 0 dBFS for every nonlinear node.</b>",
         {"asr_inband_db", "asr_fullband_db", "asr_vs_harmonics_db", "asr_worst_f0_hz",
          "asr_noise_floor_db", "asr_nonperiodic_residual_db", "asr_settle_frames",
          "asr_collision_free", "asr_inband_worstlevel_db", "asr_worstlevel_dbfs"}},

        {"Twin-tone intermodulation, 19 kHz + 20 kHz",
         "Two equal tones at &minus;6 dBFS each, summing to 0 dBFS. Any nonlinearity "
         "produces a difference product at 1 kHz &mdash; a frequency no harmonic of either "
         "tone can reach, so a component there cannot be anything but intermodulation. "
         "Third-order products at 18 and 21 kHz are reported alongside.",
         {"imd_d2_db", "imd_d2_percent", "imd_d3_db"}},

        {"Frequency, phase and group delay (Farina swept sine)",
         "A 0.68 s exponential sweep, 20 Hz to 40 kHz, deconvolved against its inverse "
         "filter, and divided by the <em>same chain with the node bypassed</em> rather than "
         "by the raw stimulus &mdash; so the inverse filter, the delay compensation and the "
         "buffer edges all cancel and what is left is the node. Group delay is residual, "
         "after the chain has compensated the node's declared latency, and is fitted across "
         "25 bins rather than differenced across two, which otherwise multiplies the phase "
         "noise by 41 722. The textbook pre-window that discards Farina's distortion "
         "impulses is deliberately absent: the linear impulse of a sweep starting at 20 Hz "
         "is itself ~50 ms wide, so no window separates the two here, and cutting between "
         "them put &plusmn;3 dB of magnitude and 20&deg; of phase ripple into the answer. "
         "The last column is the excess energy arriving before the linear impulse, relative "
         "to the null render, which is what that decision costs.",
         {"fr_mag_125hz_db", "fr_mag_1khz_db", "fr_mag_8khz_db", "fr_max_mag_db",
          "fr_min_mag_db", "fr_phase_1khz_deg", "fr_gd_1khz_samples",
          "fr_gd_max_samples", "fr_pre_arrival_db"}},

        {"True peak, against an independent meter",
         "The reference is a 32&times; oversampling reconstruction with a 6145-tap "
         "Blackman-Harris windowed sinc, written in this harness and sharing no code with "
         "the engine's meter &mdash; then validated against the <em>analytic</em> answer, "
         "since a sampled unit sinusoid reconstructs to exactly 0 dBTP. Comparing the "
         "engine against itself would prove nothing, and comparing two approximations "
         "proves only that they disagree.",
         {"tp_sample_peak_dbfs", "tp_reference_dbtp", "tp_engine_meter_dbtp",
          "tp_meter_delta_db", "tp_conformance_dbtp"}},
    };

    int n = 1;
    for (const auto& sec : sections) writeSection(f, n++, sec, reports);

    f << "<footer>Generated by <code>engine/tests/measurements.cpp</code>. "
         "Machine-readable copy in <code>engine/build/measurements.json</code>; "
         "<code>ci/check_measurements.py</code> compares it against "
         "<code>engine/tests/measurements.baseline.json</code> and fails the build "
         "on a regression.</footer>\n";
    f << "</div></body></html>\n";
  }

  std::printf("\nwrote %s\n      %s\n", htmlPath.c_str(), jsonPath.c_str());
  std::printf("%d of %d gated measurements pass (%.1fs total)\n",
              gatedCount - failures, gatedCount, totalSecs);
  for (const auto& s : failingRows) std::printf("  BELOW THRESHOLD  %s\n", s.c_str());

  // Exit status reports whether the MEASUREMENT ran, not whether the engine is
  // perfect. A node below its threshold is a finding, and findings belong in
  // the report and in ci/check_measurements.py's gate — which knows the
  // difference between a new failure and one that is already recorded. Making
  // this binary red on a known failure would leave CI red permanently, and a
  // permanently red gate is one nobody reads.
  return 0;
}
