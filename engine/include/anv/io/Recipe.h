// Recipe: the JSON description of a render, and the code that turns it into
// chains of nodes.
//
// A recipe is four things at once — the planner's output contract, the user's
// editable representation, the render cache key, and the reproducibility
// record. Anything that is not in the recipe is not reproducible, which is why
// the seed and the internal rate are fields rather than defaults chosen here.
//
// TWO CHAINS, NOT ONE, AND THE SPLIT IS THE POINT.
//
//   nodes[]   run at the internal rate (96 kHz), where oversampling is cheap
//             and nonlinear processing has room above the audio band.
//   master[]  runs AFTER the output sample-rate conversion, at the delivery
//             rate.
//
// The master section exists because two operations are only correct at the
// delivery rate, and putting them in the main chain would quietly produce
// wrong output that still looks fine:
//
//   True peak is DEFINED at the delivery rate. A limiter that meets -1.0 dBTP
//   at 96 kHz can overshoot after conversion to 44.1 kHz, because the
//   resampler reconstructs a continuous waveform whose peaks were never
//   sample-aligned. Limiting before the final SRC does not limit the file that
//   ships.
//
//   Dither is a function of the DELIVERY word length, and anything that
//   processes after it re-correlates the quantisation error and undoes it.
//   Resampling after dither is exactly such a process.
//
// So the ordering rule from the plan — output SRC, then true-peak limiter,
// then dither, then write — is not a convention this code hopes callers
// follow. It is structural: the two stages are separate lists, and
// `validateRecipe` rejects a recipe that puts those nodes in the wrong one.
#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "anv/core/ParamSet.h"
#include "anv/graph/Chain.h"
#include "anv/io/Json.h"
#include "anv/io/WavFile.h"

#include "anv/nodes/Chorus.h"
#include "anv/nodes/Compressor.h"
#include "anv/nodes/Console.h"
#include "anv/nodes/ConvolutionReverb.h"
#include "anv/nodes/DCBlock.h"
#include "anv/nodes/Delay.h"
#include "anv/nodes/Dither.h"
#include "anv/nodes/Drive.h"
#include "anv/nodes/DynamicEQ.h"
#include "anv/nodes/Gain.h"
#include "anv/nodes/GateExpander.h"
#include "anv/nodes/LinearPhaseEQ.h"
#include "anv/nodes/MidSide.h"
#include "anv/nodes/MonoToStereo.h"
#include "anv/nodes/MultibandCompressor.h"
#include "anv/nodes/Pan.h"
#include "anv/nodes/ParametricEQ.h"
#include "anv/nodes/PlateReverb.h"
#include "anv/nodes/Polarity.h"
#include "anv/nodes/StereoWidth.h"
#include "anv/nodes/Tape.h"
#include "anv/nodes/TransientDesigner.h"
#include "anv/nodes/TruePeakLimiter.h"
#include "anv/nodes/ZeroPhaseEQ.h"

namespace anv {

class RecipeError : public std::runtime_error {
public:
  explicit RecipeError(const std::string& what) : std::runtime_error(what) {}
};

// The node type name is a CLOSED SET. An unknown type is an error, never a
// silent no-op: a planner that emits `"type": "tape_saturation"` before that
// node exists must get a loud failure, not a render that is missing the one
// effect the user asked for and sounds fine otherwise.
inline std::unique_ptr<Node> makeNode(const std::string& type) {
  if (type == "gain")               return std::make_unique<Gain>();
  if (type == "pan")                return std::make_unique<Pan>();
  if (type == "polarity")           return std::make_unique<Polarity>();
  if (type == "dc_block")           return std::make_unique<DCBlock>();
  if (type == "mono_to_stereo")     return std::make_unique<MonoToStereo>();
  if (type == "mid_side")           return std::make_unique<MidSide>();
  if (type == "parametric_eq")      return std::make_unique<ParametricEQ>();
  if (type == "zero_phase_eq")      return std::make_unique<ZeroPhaseEQ>();
  if (type == "linear_phase_eq")    return std::make_unique<LinearPhaseEQ>();
  if (type == "compressor")         return std::make_unique<Compressor>();
  if (type == "multiband_compressor")
    return std::make_unique<MultibandCompressor>();
  if (type == "transient_designer") return std::make_unique<TransientDesigner>();
  if (type == "gate_expander")      return std::make_unique<GateExpander>();
  if (type == "dynamic_eq")         return std::make_unique<DynamicEQ>();
  if (type == "drive")              return std::make_unique<Drive>();
  if (type == "tape")               return std::make_unique<Tape>();
  if (type == "console")            return std::make_unique<Console>();
  if (type == "delay")              return std::make_unique<Delay>();
  if (type == "convolution_reverb") return std::make_unique<ConvolutionReverb>();
  if (type == "plate_reverb")       return std::make_unique<PlateReverb>();
  if (type == "chorus")             return std::make_unique<Chorus>();
  if (type == "stereo_width")       return std::make_unique<StereoWidth>();
  if (type == "true_peak_limiter")  return std::make_unique<TruePeakLimiter>();
  if (type == "dither")             return std::make_unique<Dither>();
  return nullptr;
}

// Filter shapes ParametricEQ understands. Spelled here as well as in the node
// because an unrecognised name there falls back to a bell, silently — and a
// planner that writes "high_shelf" instead of "highshelf" would get a bell,
// a plausible-sounding result, and no indication anything went wrong. That is
// the same failure the closed node-type set exists to prevent, one level down.
inline bool isKnownFilterShape(const std::string& shape) {
  return shape == "bell" || shape == "lowshelf" || shape == "highshelf" ||
         shape == "notch" || shape == "lowpass" || shape == "highpass";
}

// Every string-valued parameter a node reads, with every value it acts on.
//
// Nodes read these with a fallback, so a value missing from this table — a
// typo, a newer schema's spelling — renders as the default and says nothing.
// That shipped: the Python schema spelled dynamic EQ's modes "downward" and
// "upward", the node compared against "down" and "up", and every dynamic EQ
// the planner ever wrote rendered as a static one. A node that adds a string
// parameter must add it here, for the same reason band lists are named below.
struct KnownStrings {
  const char* type;
  const char* param;
  std::vector<std::string> values;
};

inline const std::vector<KnownStrings>& knownStringParams() {
  static const std::vector<KnownStrings> table = {
      {"compressor", "detector", {"peak", "rms"}},
      {"compressor", "release_mode", {"fixed", "program"}},
      {"compressor", "makeup_mode", {"manual", "auto"}},
      {"gate_expander", "detector", {"peak", "rms"}},
      {"mid_side", "mode", {"encode", "decode"}},
      {"drive", "curve", {"soft_clip", "hard_clip"}},
      {"drive", "os_profile", {"wideband", "audio"}},
      {"delay", "time_note_division",
       {"", "sixteenth", "eighth", "dotted_eighth", "triplet_eighth", "quarter"}},
      {"delay", "stereo_mode", {"mono", "stereo_offset", "ping_pong"}},
      {"dither", "noise_shaping", {"none", "first_order"}},
      // Band-level, for the one node whose bands carry their own enums.
      {"dynamic_eq.band", "mode", {"static", "down", "up"}},
      // A dynamic band only distinguishes shelf from not-shelf, so notch,
      // lowpass and highpass would render as a bell.
      {"dynamic_eq.band", "filter_type", {"bell", "lowshelf", "highshelf"}},
  };
  return table;
}

inline void checkStringParams(const ParamSet& p, const std::string& type,
                              const std::string& where) {
  for (const KnownStrings& k : knownStringParams()) {
    if (type != k.type) continue;
    if (p.hasNum(k.param))
      throw RecipeError(where + ": " + k.param + " must be a string, not a "
                        "number — as a number it would read as the default");
    if (!p.hasStr(k.param)) continue;
    const std::string value = p.str(k.param, "");
    bool known = false;
    std::string expected;
    for (const std::string& v : k.values) {
      known = known || v == value;
      expected += (expected.empty() ? "\"" : ", \"") + v + "\"";
    }
    if (!known)
      throw RecipeError(where + ": " + k.param + " \"" + value +
                        "\" is not one of " + expected +
                        " — an unrecognised value would silently become the "
                        "default");
  }
}

// Node types that are only correct after the output sample-rate conversion.
inline bool isMasterOnly(const std::string& type) {
  return type == "true_peak_limiter" || type == "dither";
}

struct RecipeNode {
  std::string id;
  std::string type;
  ParamSet params;
  bool bypassed = false;
  double mix = 1.0;
};

struct Recipe {
  std::string engineVersion;
  double internalSampleRate = kCanonicalSampleRate;
  // 0 means "whatever the input file was". Explicit is better, but a recipe
  // that does not care should not have to guess the source rate.
  double outputSampleRate = 0.0;
  SampleFormat outputFormat = SampleFormat::PcmInt24;
  uint64_t seed = 0;

  std::vector<RecipeNode> nodes;
  std::vector<RecipeNode> master;
};

inline SampleFormat parseFormat(const std::string& name) {
  if (name == "pcm16")   return SampleFormat::PcmInt16;
  if (name == "pcm24")   return SampleFormat::PcmInt24;
  if (name == "pcm32")   return SampleFormat::PcmInt32;
  if (name == "float32") return SampleFormat::Float32;
  if (name == "float64") return SampleFormat::Float64;
  throw RecipeError("unknown output format \"" + name +
                    "\" (expected pcm16, pcm24, pcm32, float32 or float64)");
}

// JSON object to ParamSet.
//
// ParamSet holds exactly one ordered sub-list, which is what an EQ's bands or
// a dynamic EQ's bands need. An array of objects under any key becomes that
// list; a second such array is an error rather than a silent overwrite,
// because the node reading `items()` has no way to tell which one it got.
inline ParamSet paramSetFromJson(const JsonValue& v, const std::string& where) {
  ParamSet p;
  if (v.isNull()) return p;
  if (!v.isObject()) throw RecipeError(where + ": params must be an object");

  std::string listKey;
  for (const auto& kv : v.object) {
    const std::string& key = kv.first;
    const JsonValue& val = kv.second;

    if (val.isNumber()) {
      p.set(key, val.number);
    } else if (val.isString()) {
      p.setStr(key, val.string);
    } else if (val.isBool()) {
      // Booleans arrive as 0/1 because every node reads its flags through
      // num(). Keeping a separate bool channel would mean each node had to
      // guess which one the recipe used.
      p.set(key, val.boolean ? 1.0 : 0.0);
    } else if (val.isArray()) {
      if (!listKey.empty())
        throw RecipeError(where + ": two list parameters (\"" + listKey +
                          "\" and \"" + key +
                          "\") — a node can only read one ordered list");
      listKey = key;
      for (size_t i = 0; i < val.array.size(); ++i) {
        const JsonValue& item = val.array[i];
        if (!item.isObject())
          throw RecipeError(where + ": \"" + key + "\"[" + std::to_string(i) +
                            "] must be an object");
        p.addItem(paramSetFromJson(
            item, where + "." + key + "[" + std::to_string(i) + "]"));
      }
    } else if (!val.isNull()) {
      throw RecipeError(where + ": parameter \"" + key + "\" has an unsupported type");
    }
  }
  return p;
}

inline std::vector<RecipeNode> parseNodeList(const JsonValue* arr,
                                             const std::string& where) {
  std::vector<RecipeNode> out;
  if (!arr || arr->isNull()) return out;
  if (!arr->isArray()) throw RecipeError(where + " must be an array");

  for (size_t i = 0; i < arr->array.size(); ++i) {
    const JsonValue& j = arr->array[i];
    const std::string here = where + "[" + std::to_string(i) + "]";
    if (!j.isObject()) throw RecipeError(here + " must be an object");

    RecipeNode n;
    n.type = j.stringOr("type", "");
    if (n.type.empty()) throw RecipeError(here + " has no \"type\"");

    // A stable id per node is what makes multi-turn refinement a patch rather
    // than a regeneration: "adjust the reverb" has to name something.
    n.id = j.stringOr("id", n.type + "_" + std::to_string(i));
    n.bypassed = j.boolOr("bypass", false);
    n.mix = j.numberOr("mix", 1.0);
    if (!(n.mix >= 0.0 && n.mix <= 1.0))
      throw RecipeError(here + ": \"mix\" must be between 0 and 1");

    const JsonValue* params = j.find("params");
    n.params = params ? paramSetFromJson(*params, here) : ParamSet{};
    out.push_back(std::move(n));
  }
  return out;
}

inline Recipe parseRecipe(const std::string& text) {
  const JsonValue root = parseJson(text);
  if (!root.isObject()) throw RecipeError("recipe must be a JSON object");

  Recipe r;
  r.engineVersion = root.stringOr("engine_version", "");
  r.internalSampleRate = root.numberOr("internal_sample_rate", kCanonicalSampleRate);
  // JSON numbers are doubles, and converting a negative or oversized one to
  // uint64_t is undefined behaviour. 2^53 is where doubles stop representing
  // every integer, so a seed past it would silently be a different seed.
  const double seed = root.numberOr("seed", 0.0);
  if (!(seed >= 0.0 && seed <= 9007199254740992.0) || seed != std::floor(seed))
    throw RecipeError("seed must be a whole number from 0 to 2^53");
  r.seed = static_cast<uint64_t>(seed);

  if (const JsonValue* out = root.find("output")) {
    if (!out->isObject()) throw RecipeError("\"output\" must be an object");
    r.outputSampleRate = out->numberOr("sample_rate", 0.0);
    r.outputFormat = parseFormat(out->stringOr("format", "pcm24"));
  }

  if (!(r.internalSampleRate >= 8000.0 && r.internalSampleRate <= 768000.0))
    throw RecipeError("internal_sample_rate out of range");
  if (r.outputSampleRate != 0.0 &&
      !(r.outputSampleRate >= 8000.0 && r.outputSampleRate <= 768000.0))
    throw RecipeError("output.sample_rate out of range");

  r.nodes = parseNodeList(root.find("nodes"), "nodes");
  r.master = parseNodeList(root.find("master"), "master");
  return r;
}

// Structural rules that hold regardless of parameter values. Everything here
// is a mistake that produces plausible-sounding but wrong output, which is the
// only kind worth a hard error — an obviously broken render needs no validator.
inline void validateRecipe(const Recipe& r) {
  std::vector<std::string> ids;

  auto checkList = [&](const std::vector<RecipeNode>& list, bool isMaster,
                       const char* name) {
    for (size_t i = 0; i < list.size(); ++i) {
      const RecipeNode& n = list[i];
      const std::string here = std::string(name) + "[" + std::to_string(i) + "]";

      if (!makeNode(n.type))
        throw RecipeError(here + ": unknown node type \"" + n.type + "\"");

      for (const std::string& seen : ids)
        if (seen == n.id)
          throw RecipeError(here + ": duplicate node id \"" + n.id +
                            "\" — ids must be unique so a refinement can "
                            "name exactly one node");
      ids.push_back(n.id);

      // Band-level parameters are checked here rather than in the node, so a
      // typo fails before anything is rendered rather than producing audio
      // that is quietly not what was asked for.
      // Every node that reads a band list has to be named here. The node's own
      // shape lookup falls through to a bell, so a type that is added to
      // makeNode and forgotten in this condition gets the silent-fallback bug
      // back — one node at a time.
      if (n.type == "parametric_eq" || n.type == "dynamic_eq" ||
          n.type == "zero_phase_eq" || n.type == "linear_phase_eq") {
        const auto& bands = n.params.items();
        for (size_t b = 0; b < bands.size(); ++b) {
          if (bands[b].hasNum("filter_type"))
            throw RecipeError(here + ": band " + std::to_string(b) +
                              " has a numeric filter_type, which would "
                              "silently become a bell");
          if (n.type == "dynamic_eq") {
            const std::string where = here + ": band " + std::to_string(b);
            checkStringParams(bands[b], "dynamic_eq.band", where);
            // The node takes a magnitude and clamps a negative one to zero,
            // which renders as a static band. Direction is `mode`'s job.
            if (bands[b].num("range_db", 0.0) < 0.0)
              throw RecipeError(where + ": range_db is negative — it is a "
                                "magnitude, and `mode` sets the direction");
          }
          const std::string shape = bands[b].str("filter_type", "bell");
          if (!isKnownFilterShape(shape))
            throw RecipeError(here + ": band " + std::to_string(b) +
                              " has unknown filter_type \"" + shape +
                              "\" (expected bell, lowshelf, highshelf, notch, "
                              "lowpass or highpass) — an unrecognised name "
                              "would silently become a bell");
        }
      }

      checkStringParams(n.params, n.type, here);

      if (isMasterOnly(n.type) && !isMaster)
        throw RecipeError(here + ": \"" + n.type +
                          "\" must be in \"master\", not \"nodes\" — it is "
                          "only correct after the output sample-rate "
                          "conversion");
    }
  };

  checkList(r.nodes, false, "nodes");
  checkList(r.master, true, "master");

  // Dither last, and only one of it. Anything after dither re-correlates the
  // quantisation error with the signal, which is the entire thing dither
  // exists to prevent.
  int dithers = 0;
  for (size_t i = 0; i < r.master.size(); ++i) {
    if (r.master[i].type != "dither") continue;
    ++dithers;
    if (i + 1 != r.master.size())
      throw RecipeError("master[" + std::to_string(i) +
                        "]: dither must be the last node — anything after it "
                        "re-correlates the quantisation error");
  }
  if (dithers > 1) throw RecipeError("more than one dither node");

  // A limiter after dither is caught by the rule above; a limiter that is
  // never reached because dither precedes it cannot happen for the same
  // reason. What remains is the gain-after-limiting case.
  bool sawLimiter = false;
  for (const RecipeNode& n : r.master) {
    if (n.type == "true_peak_limiter") { sawLimiter = true; continue; }
    if (sawLimiter && n.type != "dither")
      throw RecipeError("master: \"" + n.type +
                        "\" runs after the true-peak limiter — nothing but "
                        "dither may, or the ceiling the limiter established "
                        "no longer holds");
  }
}

inline void buildChain(const std::vector<RecipeNode>& list, Chain& chain) {
  for (const RecipeNode& n : list) {
    auto node = makeNode(n.type);
    if (!node) throw RecipeError("unknown node type \"" + n.type + "\"");

    Chain::SlotConfig cfg;
    cfg.nodeId = n.id;
    cfg.bypassed = n.bypassed;
    cfg.mix = n.mix;
    chain.add(std::move(node), n.params, cfg);
  }
}

} // namespace anv
