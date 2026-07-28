// Node contract tests.
//
// These are parameterized over the node registry, so adding a node
// automatically enrolls it in every test below. There is no opt-in and no
// exemption without a documented entry — a node that cannot pass these is not
// a node we are willing to render a paying customer's audio through.
//
// Two thresholds, deliberately:
//   BIT-EXACT for same-binary properties (bypass null, block-size invariance,
//     determinism, reset equivalence). Any deviation here is a real bug.
//   -140 dBFS for anything that would have to survive a different compiler,
//     libm, or SIMD width. Not used yet — everything here runs one binary.

#include <cmath>
#include <chrono>
#include <complex>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "anv/core/AudioBuffer.h"
#include "anv/core/Denormals.h"
#include "anv/core/Rng.h"
#include "anv/graph/Chain.h"
#include "anv/dsp/Fft.h"
#include "anv/dsp/Oversampler.h"
#include "anv/nodes/Compressor.h"
#include "anv/nodes/DCBlock.h"
#include "anv/nodes/ConvolutionReverb.h"
#include "anv/nodes/Delay.h"
#include "anv/nodes/Drive.h"
#include "anv/nodes/DynamicEQ.h"
#include "anv/nodes/GateExpander.h"
#include "anv/nodes/MidSide.h"
#include "anv/nodes/MonoToStereo.h"
#include "anv/nodes/Pan.h"
#include "anv/nodes/Polarity.h"
#include "anv/nodes/Gain.h"
#include "anv/nodes/ParametricEQ.h"
#include "anv/nodes/ReferenceDelay.h"
#include "anv/nodes/TruePeakLimiter.h"

using namespace anv;

// ---------------------------------------------------------------- harness --

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::printf("  FAIL  %s\n", what.c_str());
  }
}

// How a node's latency can legitimately be verified.
//
//   Peak  — the impulse response's largest sample sits at the compensated
//           position. Correct for minimum-phase filters, pure delays and
//           linear-phase FIRs (whose response is symmetric about its peak).
//   Onset — the response BEGINS at the compensated position, plus whatever
//           inherent pre-delay the node's own parameters specify. Required for
//           nodes with a long decaying tail, where the peak lands at an
//           arbitrary point in the decay and the energy centroid sits
//           mid-tail.
//
// `onsetOffsetFrames` is that inherent pre-delay, and it exists to keep a real
// distinction sharp: a delay line and a lookahead limiter both hold audio
// back, but only one of them is asking the rest of the graph to wait. A
// delay's time and a reverb's pre-delay are the EFFECT and must not be
// compensated; declaring them as latency would have the chain cancel them out
// and the node would do nothing audible.
enum class LatencyProbe { Peak, Onset };

struct NodeSpec {
  std::string name;
  std::function<std::unique_ptr<Node>()> make;
  ParamSet params;
  LatencyProbe latencyProbe = LatencyProbe::Peak;
  int64_t onsetOffsetFrames = 0;
};

// The registry. Every node ships with the parameter set it should be exercised
// under — deliberately non-trivial, because a node tested at unity does not
// prove anything.
std::vector<NodeSpec> registry() {
  std::vector<NodeSpec> v;

  {
    ParamSet p;
    p.set("gain_db", -4.5);
    v.push_back({"gain", [] { return std::make_unique<Gain>(); }, p});
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
    v.push_back({"parametric_eq", [] { return std::make_unique<ParametricEQ>(); }, p});
  }
  {
    // A deliberately awkward, non-power-of-two latency. Without a
    // latency-bearing node in this registry the bypass null and latency tests
    // would pass vacuously.
    ParamSet p;
    p.set("latency_frames", 137.0);
    v.push_back({"reference_delay", [] { return std::make_unique<ReferenceDelay>(); }, p});
  }
  {
    ParamSet p;
    p.set("drive_db", 12.0).set("output_trim_db", -6.0).set("bias", 0.15)
        .set("oversample_factor", 4.0);
    p.setStr("curve", "soft_clip");
    v.push_back({"drive", [] { return std::make_unique<Drive>(); }, p});
  }
  {
    // Non-zero lookahead so the compressor also exercises the PDC path, and a
    // sidechain filter plus program release so the awkward paths are covered.
    ParamSet p;
    p.set("threshold_dbfs", -20.0).set("ratio_to_1", 4.0).set("knee_db", 6.0)
        .set("attack_ms", 5.0).set("release_ms", 150.0).set("lookahead_ms", 3.0)
        .set("sidechain_hp_hz", 90.0).set("stereo_link", 0.8)
        .set("makeup_gain_db", 3.0);
    p.setStr("detector", "peak");
    p.setStr("release_mode", "program");
    v.push_back({"compressor", [] { return std::make_unique<Compressor>(); }, p});
  }
  {
    ParamSet p;
    p.set("threshold_dbfs", -35.0).set("ratio_to_1", 6.0).set("knee_db", 3.0)
        .set("range_db", 45.0).set("attack_ms", 1.0).set("hold_ms", 30.0)
        .set("release_ms", 120.0).set("lookahead_ms", 2.0)
        .set("hysteresis_db", 4.0).set("sidechain_hp_hz", 120.0)
        .set("stereo_link", 1.0);
    p.setStr("detector", "peak");
    v.push_back({"gate_expander", [] { return std::make_unique<GateExpander>(); }, p});
  }
  {
    // A de-esser band, a static bell, and an upward shelf, so all three modes
    // and both filter families are exercised.
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
    v.push_back({"dynamic_eq", [] { return std::make_unique<DynamicEQ>(); }, p});
  }
  {
    ParamSet p;
    p.set("ceiling_dbtp", -1.0).set("lookahead_ms", 2.0).set("release_ms", 100.0)
        .set("oversample_factor", 8.0);
    v.push_back({"true_peak_limiter",
                 [] { return std::make_unique<TruePeakLimiter>(); }, p});
  }
  {
    ParamSet p;
    // pre_delay deliberately 0 here: it legitimately offsets the response and
    // would defeat the onset probe. It gets its own check in test 16.
    p.set("decay_s", 0.35).set("pre_delay_ms", 0.0).set("hf_damp_hz", 5000.0)
        .set("lf_cut_hz", 150.0).set("hf_decay_ratio", 0.5)
        .set("stereo_decorrelation", 1.0);
    v.push_back({"convolution_reverb",
                 [] { return std::make_unique<ConvolutionReverb>(); }, p,
                 LatencyProbe::Onset});
  }
  {
    ParamSet p;
    p.set("pan_position", -0.4);
    v.push_back({"pan", [] { return std::make_unique<Pan>(); }, p});
  }
  {
    ParamSet p;
    p.set("inverted", 1.0);
    v.push_back({"polarity", [] { return std::make_unique<Polarity>(); }, p});
  }
  {
    ParamSet p;
    p.set("cutoff_hz", 10.0);
    v.push_back({"dc_block", [] { return std::make_unique<DCBlock>(); }, p});
  }
  {
    ParamSet p;
    p.set("source_channel", 0.0);
    v.push_back({"mono_to_stereo", [] { return std::make_unique<MonoToStereo>(); }, p});
  }
  {
    ParamSet p;
    p.setStr("mode", "encode");
    v.push_back({"mid_side", [] { return std::make_unique<MidSide>(); }, p});
  }
  {
    // Short time and modest feedback keep the declared tail small enough that
    // every contract test does not turn into a multi-second render.
    ParamSet p;
    p.set("time_ms", 20.0).set("feedback_percent", 35.0)
        .set("hf_damp_hz", 7000.0).set("lf_damp_hz", 120.0)
        .set("spread_ms", 0.0).set("max_tail_s", 1.0);
    p.setStr("stereo_mode", "mono");
    // 20 ms of delay time is the effect, not latency: the response is
    // expected 1920 samples after the impulse, and latencyFrames() stays 0.
    v.push_back({"delay", [] { return std::make_unique<Delay>(); }, p,
                 LatencyProbe::Onset, 1920});
  }

  return v;
}

// Deterministic, broadband, non-trivial test signal. Pink-ish noise plus a
// couple of tones so both the filter and the dynamics paths see real content.
AudioBuffer testSignal(int channels, int64_t frames, uint64_t seed = 12345) {
  AudioBuffer buf(channels, frames);
  CounterRng rng(seed);
  for (int c = 0; c < channels; ++c) {
    Sample* d = buf.channel(c);
    double lp = 0.0;
    for (int64_t i = 0; i < frames; ++i) {
      const double white = rng.bipolar(c, i);
      lp = 0.98 * lp + 0.02 * white; // gives the noise some low-end weight
      const double t = static_cast<double>(i) / kCanonicalSampleRate;
      d[i] = 0.35 * (white * 0.4 + lp * 6.0) +
             0.20 * std::sin(2.0 * M_PI * 220.0 * t + c) +
             0.10 * std::sin(2.0 * M_PI * 3150.0 * t);
    }
  }
  return buf;
}

double maxAbsDiff(const AudioBuffer& a, const AudioBuffer& b, int64_t n) {
  double worst = 0.0;
  for (int c = 0; c < a.channels(); ++c)
    for (int64_t i = 0; i < n; ++i)
      worst = std::max(worst, std::abs(a.channel(c)[i] - b.channel(c)[i]));
  return worst;
}

double peak(const AudioBuffer& a) {
  double worst = 0.0;
  for (int c = 0; c < a.channels(); ++c)
    for (int64_t i = 0; i < a.frames(); ++i)
      worst = std::max(worst, std::abs(a.channel(c)[i]));
  return worst;
}

bool allFinite(const AudioBuffer& a) {
  for (int c = 0; c < a.channels(); ++c)
    for (int64_t i = 0; i < a.frames(); ++i)
      if (!std::isfinite(a.channel(c)[i])) return false;
  return true;
}

PrepareSpec makeSpec(int channels, int maxBlock, int64_t total) {
  PrepareSpec s;
  s.sampleRate = kCanonicalSampleRate;
  s.maxBlockFrames = maxBlock;
  s.numChannels = channels;
  s.totalFrames = total;
  s.seed = 20260728;
  return s;
}

std::unique_ptr<Chain> buildChain(const std::vector<NodeSpec>& nodes,
                                  bool bypassAll, int channels, int maxBlock,
                                  int64_t frames) {
  auto chain = std::make_unique<Chain>();
  int i = 0;
  for (const auto& ns : nodes) {
    Chain::SlotConfig cfg;
    cfg.nodeId = "n" + std::to_string(i++);
    cfg.bypassed = bypassAll;
    chain->add(ns.make(), ns.params, cfg);
  }
  chain->prepare(makeSpec(channels, maxBlock, frames));
  return chain;
}

// ------------------------------------------------------------------ tests --

// 1. Bypass null test. A chain of bypassed nodes must reproduce the input
//    BIT EXACTLY. Because a bypassed node still declares its latency and the
//    chain still inserts the compensating delay, this exercises the real PDC
//    path — which is what makes it the highest-value test in the suite.
void testBypassNull(const std::vector<NodeSpec>& all) {
  const int64_t frames = 8192;
  const AudioBuffer in = testSignal(2, frames);

  auto chain = buildChain(all, /*bypassAll=*/true, 2, 512, frames);
  AudioBuffer out;
  chain->render(in, out, 512);

  const double diff = maxAbsDiff(in, out, frames);
  check(diff == 0.0, "bypass null: full chain is bit-exact (diff=" +
                         std::to_string(diff) + ")");

  // And each node individually, so a failure names the culprit.
  for (const auto& ns : all) {
    auto one = buildChain({ns}, true, 2, 512, frames);
    AudioBuffer o;
    one->render(in, o, 512);
    const double d = maxAbsDiff(in, o, frames);
    check(d == 0.0, "bypass null: " + ns.name + " (diff=" + std::to_string(d) + ")");
  }
}

// 2. Block-size invariance. Odd and prime sizes are mandatory: powers of two
//    hide index arithmetic bugs. Any per-block computation whose result
//    depends on the block size is a bug, so this must be bit exact.
void testBlockSizeInvariance(const std::vector<NodeSpec>& all) {
  const int64_t frames = 8192;
  const AudioBuffer in = testSignal(2, frames);
  const int sizes[] = {1, 7, 64, 512, 1023, 4096};

  for (const auto& ns : all) {
    auto ref = buildChain({ns}, false, 2, 4096, frames);
    AudioBuffer refOut;
    ref->render(in, refOut, 512);

    for (int bs : sizes) {
      auto c = buildChain({ns}, false, 2, 4096, frames);
      AudioBuffer o;
      c->render(in, o, bs);
      const double d = maxAbsDiff(refOut, o, frames);
      check(d == 0.0, "block-size invariance: " + ns.name + " @ " +
                          std::to_string(bs) + " (diff=" + std::to_string(d) + ")");
    }
  }
}

// 3. Determinism. Same recipe, same seed, same input -> bit-identical output
//    across repeated runs and fresh instances.
void testDeterminism(const std::vector<NodeSpec>& all) {
  const int64_t frames = 4096;
  const AudioBuffer in = testSignal(2, frames);

  auto a = buildChain(all, false, 2, 512, frames);
  AudioBuffer outA;
  a->render(in, outA, 512);

  for (int run = 0; run < 3; ++run) {
    auto b = buildChain(all, false, 2, 512, frames);
    AudioBuffer outB;
    b->render(in, outB, 512);
    check(maxAbsDiff(outA, outB, frames) == 0.0,
          "determinism: run " + std::to_string(run) + " is bit-identical");
  }
}

// 4. Reset equivalence. prepare(); process(X); reset(); process(X) must equal
//    a fresh instance processing X, bit exactly. Catches state that survives
//    reset — the classic source of "the second render sounds different".
void testResetEquivalence(const std::vector<NodeSpec>& all) {
  const int64_t frames = 4096;
  const AudioBuffer in = testSignal(2, frames);

  for (const auto& ns : all) {
    auto fresh = buildChain({ns}, false, 2, 512, frames);
    AudioBuffer expected;
    fresh->render(in, expected, 512);

    auto reused = buildChain({ns}, false, 2, 512, frames);
    AudioBuffer scratch;
    reused->render(in, scratch, 512);
    reused->reset();
    AudioBuffer afterReset;
    reused->render(in, afterReset, 512);

    check(maxAbsDiff(expected, afterReset, frames) == 0.0,
          "reset equivalence: " + ns.name);
  }
}

// 5. Latency correctness. A Dirac in must emerge at exactly the same frame it
//    went in, once the declared latency has been trimmed.
//
//    Measured by the PEAK of the impulse response, not by its onset. A
//    linear-phase FIR — which every oversampled node contains — has a response
//    symmetric about its group delay, so energy legitimately appears M samples
//    BEFORE the peak. Onset detection would flag that pre-ringing as a latency
//    error and is only valid for minimum-phase nodes.
//
//    The impulse is deliberately small so nonlinear nodes are measured in
//    their quasi-linear region; group delay is a linear-system property and
//    driving a saturator into clipping does not measure it.
void testLatency(const std::vector<NodeSpec>& all) {
  const int64_t frames = 4096;
  const int impulseAt = 1000;

  for (const auto& ns : all) {
    AudioBuffer in(1, frames);
    in.channel(0)[impulseAt] = 0.01;

    auto c = buildChain({ns}, false, 1, 512, frames);
    AudioBuffer out;
    c->render(in, out, 512);

    if (ns.latencyProbe == LatencyProbe::Peak) {
      int64_t peakAt = -1;
      double peakVal = 0.0;
      for (int64_t i = 0; i < out.frames(); ++i) {
        const double v = std::abs(out.channel(0)[i]);
        if (v > peakVal) { peakVal = v; peakAt = i; }
      }
      check(peakAt == impulseAt,
            "latency: " + ns.name + " impulse peak at " + std::to_string(peakAt) +
                ", expected " + std::to_string(impulseAt));

      // Energy centroid as a second, independent estimator. It catches an
      // asymmetric response whose peak happens to land in the right place.
      double num = 0.0, den = 0.0;
      for (int64_t i = 0; i < out.frames(); ++i) {
        const double e = out.channel(0)[i] * out.channel(0)[i];
        num += e * static_cast<double>(i);
        den += e;
      }
      const double centroid = den > 0.0 ? num / den : -1.0;
      check(std::abs(centroid - static_cast<double>(impulseAt)) < 2.0,
            "latency: " + ns.name + " energy centroid at " +
                std::to_string(centroid) + ", expected ~" + std::to_string(impulseAt));
    } else {
      // Nothing may emerge before the compensated position (that would be
      // non-causal), and something must emerge exactly at it (later would
      // mean the declared latency is too large).
      //
      // The threshold is not zero. An FFT-based node cannot produce exact
      // zeros: round-off in the transform leaves around 1e-18 where the maths
      // says zero, which is -309 dB relative to the probe and utterly
      // irrelevant. Demanding exact zeros would only be testing that a node
      // happens to use direct-form arithmetic.
      const double eps = 1e-9 * 0.01; // 1e-9 of the impulse amplitude
      double before = 0.0;
      for (int64_t i = 0; i < impulseAt; ++i)
        before = std::max(before, std::abs(out.channel(0)[i]));
      check(before < eps,
            "latency: " + ns.name + " emits nothing before the impulse (got " +
                std::to_string(before) + ")");

      const int64_t expectedOnset = impulseAt + ns.onsetOffsetFrames;
      int64_t onset = -1;
      for (int64_t i = 0; i < out.frames(); ++i)
        if (std::abs(out.channel(0)[i]) > eps) { onset = i; break; }
      check(onset == expectedOnset,
            "latency: " + ns.name + " response begins at " + std::to_string(onset) +
                ", expected " + std::to_string(expectedOnset));
    }
  }
}

// 6. Silence in, silence out — EXACT zeros. Catches uninitialized state, DC
//    offsets, and stray noise injection. (Nodes that legitimately generate
//    noise, such as dither, will need an explicit documented exemption.)
void testSilence(const std::vector<NodeSpec>& all) {
  const int64_t frames = 4096;
  const AudioBuffer in(2, frames); // zero-filled

  for (const auto& ns : all) {
    auto c = buildChain({ns}, false, 2, 512, frames);
    AudioBuffer out;
    c->render(in, out, 512);
    check(peak(out) == 0.0, "silence in -> silence out: " + ns.name);
  }
}

// 7. Robustness. Full-scale noise, DC, and a 0 dBFS square must not produce
//    NaN, Inf, or an unbounded output.
void testRobustness(const std::vector<NodeSpec>& all) {
  const int64_t frames = 4096;

  std::vector<std::pair<std::string, AudioBuffer>> cases;
  {
    AudioBuffer b(2, frames);
    CounterRng rng(99);
    for (int c = 0; c < 2; ++c)
      for (int64_t i = 0; i < frames; ++i) b.channel(c)[i] = rng.bipolar(c, i);
    cases.emplace_back("full-scale noise", std::move(b));
  }
  {
    AudioBuffer b(2, frames);
    for (int c = 0; c < 2; ++c)
      for (int64_t i = 0; i < frames; ++i) b.channel(c)[i] = 1.0;
    cases.emplace_back("DC", std::move(b));
  }
  {
    AudioBuffer b(2, frames);
    for (int c = 0; c < 2; ++c)
      for (int64_t i = 0; i < frames; ++i)
        b.channel(c)[i] = ((i / 48) % 2) ? 1.0 : -1.0;
    cases.emplace_back("0 dBFS square", std::move(b));
  }

  for (const auto& ns : all) {
    for (const auto& kv : cases) {
      auto c = buildChain({ns}, false, 2, 512, frames);
      AudioBuffer out;
      c->render(kv.second, out, 512);
      check(allFinite(out), "robustness: " + ns.name + " / " + kv.first +
                                " produced no NaN or Inf");
      check(peak(out) < 1e3, "robustness: " + ns.name + " / " + kv.first +
                                 " output stayed bounded");
    }
  }
}

// 7b. Parameter robustness.
//
//     Test 7 feeds extreme SIGNALS. This feeds extreme PARAMETERS, which is a
//     different failure mode and was entirely uncovered: removing the delay's
//     feedback cap — which lets the loop gain exceed unity and diverge — did
//     not fail a single assertion, because nothing ever set feedback above the
//     legal range.
//
//     The bar is NaN and Inf, not musicality. A wrong-sounding render is a bad
//     experience; a NaN silently poisons every downstream sample and cannot be
//     recovered from. Every node must clamp defensively rather than trusting
//     its caller, because the caller is going to be a language model.
void testParameterRobustness(const std::vector<NodeSpec>& all) {
  const int64_t frames = 2048;
  const AudioBuffer in = testSignal(2, frames);

  // Plausible mistakes rather than absurdities: sign errors, percent/fraction
  // confusion, unit mix-ups, and zeros.
  const double perturbations[] = {0.0, -1.0, -100.0, 100.0, 1000.0, 1e5};

  for (const auto& ns : all) {
    for (const std::string& key : ns.params.numericKeys()) {
      for (double v : perturbations) {
        NodeSpec variant = ns;
        variant.params.set(key, v);

        // Drive process() directly rather than render(). This test is about
        // numerical safety, and rendering the tail as well would mean paying
        // for it: a perturbed decay time clamps to the reverb's maximum, and
        // convolving a twelve-second impulse response for every parameter of
        // every node turns a fast check into a multi-minute one.
        auto chain = buildChain({variant}, false, 2, 512, frames);
        AudioBuffer out(2, frames);
        std::vector<const Sample*> bin(2);
        std::vector<Sample*> bout(2);
        bool finite = true;
        for (int64_t pos = 0; pos < frames; pos += 512) {
          const int nn = static_cast<int>(std::min<int64_t>(512, frames - pos));
          for (int c = 0; c < 2; ++c) {
            bin[static_cast<size_t>(c)] = in.channel(c) + pos;
            bout[static_cast<size_t>(c)] = out.channel(c) + pos;
          }
          chain->processBlock(bin.data(), bout.data(), nn, pos);
        }
        finite = allFinite(out);

        check(finite,
              "param robustness: " + ns.name + " with " + key + "=" +
                  std::to_string(v) + " produced no NaN or Inf");

        // Declared latency must stay bounded too. An unbounded lookahead
        // parameter is a resource-exhaustion vector rather than a numerical
        // one — it produces an enormous allocation and an O(window)-per-sample
        // inner loop, never a NaN — so the finiteness check above cannot see
        // it. One second of latency is already far beyond any sane node.
        const int lat = chain->totalLatencyFrames();
        check(lat >= 0 && lat < 96000,
              "param robustness: " + ns.name + " with " + key + "=" +
                  std::to_string(v) + " declares a bounded latency (" +
                  std::to_string(lat) + ")");
      }
    }
  }
}

// 8. Denormal control is actually active. FTZ/DAZ changes numeric results, so
//    running the suite without it would silently validate a different engine
//    than the one that renders in production.
void testDenormalsEnabled() {
  check(ScopedNoDenormals::enabled(),
        "denormal flushing is active in the test harness");
}

// 9. The counter-based RNG is stateless: identical values regardless of the
//    order or grouping in which they are requested. This is what makes
//    block-size invariance structural for any node that uses randomness.
void testRngIsCounterBased() {
  CounterRng rng(4242);
  std::vector<double> forward;
  for (int64_t i = 0; i < 512; ++i) forward.push_back(rng.uniform(0, i));

  bool ok = true;
  for (int64_t i = 511; i >= 0; --i)
    if (rng.uniform(0, i) != forward[static_cast<size_t>(i)]) ok = false;
  check(ok, "counter RNG: reverse-order access matches forward order");

  bool independent = true;
  for (int64_t i = 0; i < 64; ++i)
    if (rng.uniform(0, i) == rng.uniform(1, i)) independent = false;
  check(independent, "counter RNG: channels are independent streams");

  CounterRng other(4243);
  bool differs = false;
  for (int64_t i = 0; i < 64; ++i)
    if (other.uniform(0, i) != rng.uniform(0, i)) differs = true;
  check(differs, "counter RNG: a different seed gives a different stream");
}

// 10. Frequency response matches the analytic design.
//
//     Every test above this one is self-consistency: they would all pass for a
//     filter whose math is simply wrong, as long as it were wrong consistently.
//     This is the one that checks the DSP is actually correct.
//
//     Uses COHERENT SAMPLING — the measurement window holds an exact integer
//     number of cycles, so there is zero spectral leakage and the measured
//     gain is exact rather than windowed-and-approximate. The filter's own
//     centre frequency is snapped to the same grid so the expectation is
//     analytic, not interpolated.
double measureGainDb(ParamSet eqParams, double freqHz, double& snappedHz) {
  constexpr int64_t kWindow = 16384;
  constexpr int64_t kSettle = 16384; // let the IIR reach steady state

  // Snap to the coherent grid: freq = k * fs / N.
  const double k = std::round(freqHz * static_cast<double>(kWindow) / kCanonicalSampleRate);
  snappedHz = k * kCanonicalSampleRate / static_cast<double>(kWindow);

  const int64_t total = kSettle + kWindow;
  AudioBuffer in(1, total);
  constexpr double amp = 0.1;
  for (int64_t i = 0; i < total; ++i)
    in.channel(0)[i] = amp * std::sin(2.0 * M_PI * snappedHz *
                                      static_cast<double>(i) / kCanonicalSampleRate);

  Chain chain;
  Chain::SlotConfig cfg;
  cfg.nodeId = "eq";
  chain.add(std::make_unique<ParametricEQ>(), eqParams, cfg);
  chain.prepare(makeSpec(1, 1024, total));

  AudioBuffer out;
  chain.render(in, out, 1024);

  // RMS over the steady-state window only.
  double sumOut = 0.0, sumIn = 0.0;
  for (int64_t i = kSettle; i < total; ++i) {
    const double o = out.channel(0)[i];
    const double s = in.channel(0)[i];
    sumOut += o * o;
    sumIn += s * s;
  }
  return 20.0 * std::log10(std::sqrt(sumOut) / std::sqrt(sumIn));
}

void checkNear(double got, double want, double tolDb, const std::string& what) {
  const bool ok = std::abs(got - want) <= tolDb;
  check(ok, what + " (got " + std::to_string(got) + " dB, want " +
                std::to_string(want) + " +/- " + std::to_string(tolDb) + ")");
}

ParamSet oneBand(const char* shape, double freq, double gainDb, double q) {
  ParamSet p;
  ParamSet b;
  b.setStr("filter_type", shape)
      .set("center_freq_hz", freq)
      .set("gain_db", gainDb)
      .set("q_factor", q);
  p.addItem(b);
  return p;
}

void testFrequencyResponse() {
  double f = 0.0;

  // A bell's peak gain at its centre frequency must equal its declared gain.
  {
    double snapped = 0.0;
    // Snap first so the filter and the probe agree exactly.
    measureGainDb(oneBand("bell", 1000.0, 0.0, 1.0), 1000.0, snapped);
    checkNear(measureGainDb(oneBand("bell", snapped, 6.0, 1.0), snapped, f), 6.0,
              0.05, "bell +6 dB at centre");
    checkNear(measureGainDb(oneBand("bell", snapped, -6.0, 1.0), snapped, f), -6.0,
              0.05, "bell -6 dB at centre");
    checkNear(measureGainDb(oneBand("bell", snapped, 12.0, 4.0), snapped, f), 12.0,
              0.05, "bell +12 dB at centre, high Q");
    // Far from centre a narrow bell must be transparent.
    checkNear(measureGainDb(oneBand("bell", snapped, 12.0, 4.0), 100.0, f), 0.0,
              0.15, "bell is transparent two decades below centre");
  }

  // Shelves must reach their full declared gain in the shelf band.
  checkNear(measureGainDb(oneBand("highshelf", 8000.0, 6.0, 0.707), 20000.0, f),
            6.0, 0.3, "high shelf +6 dB well above corner");
  checkNear(measureGainDb(oneBand("highshelf", 8000.0, 6.0, 0.707), 200.0, f),
            0.0, 0.15, "high shelf is transparent well below corner");
  checkNear(measureGainDb(oneBand("lowshelf", 200.0, 6.0, 0.707), 20.0, f), 6.0,
            0.3, "low shelf +6 dB well below corner");
  checkNear(measureGainDb(oneBand("lowshelf", 200.0, 6.0, 0.707), 8000.0, f), 0.0,
            0.15, "low shelf is transparent well above corner");

  // Butterworth high-pass: -3.01 dB at the corner, and the declared slope one
  // octave below. This is what separates a real filter from the first-order
  // 6 dB/oct rolloff that cheaper implementations ship as a "high-pass".
  {
    ParamSet hp12;
    hp12.set("hp_freq_hz", 200.0).set("hp_slope_db_oct", 12.0);
    checkNear(measureGainDb(hp12, 200.0, f), -3.01, 0.15,
              "12 dB/oct HP is -3 dB at corner");
    checkNear(measureGainDb(hp12, 100.0, f), -12.3, 0.6,
              "12 dB/oct HP one octave below corner");

    ParamSet hp24;
    hp24.set("hp_freq_hz", 200.0).set("hp_slope_db_oct", 24.0);
    checkNear(measureGainDb(hp24, 200.0, f), -3.01, 0.15,
              "24 dB/oct HP is -3 dB at corner");
    checkNear(measureGainDb(hp24, 100.0, f), -24.3, 0.8,
              "24 dB/oct HP one octave below corner");
    checkNear(measureGainDb(hp24, 2000.0, f), 0.0, 0.05,
              "24 dB/oct HP is transparent a decade above corner");
  }
}

// 11. Aliasing-to-Signal Ratio — the metric that decides whether a saturator
//     sounds like analog gear or like cheap plugin fizz.
//
//     Method: drive a single tone chosen so its harmonics start folding almost
//     immediately, then partition the spectrum. With COHERENT SAMPLING the
//     tone completes an exact integer number of cycles in the analysis window,
//     so every harmonic — legitimate or folded — lands in exactly one bin with
//     zero leakage. Bins holding a harmonic whose true frequency is below
//     Nyquist are signal; every other populated bin is aliasing that did not
//     exist in the input.
//
//     The test asserts no bin is claimed by both a legitimate and a folded
//     harmonic, because that collision would silently hide aliasing in the
//     signal total.
//     TWO numbers are reported, because they answer different questions:
//
//     inBandDb  — alias energy landing BELOW 20 kHz, where a human can hear
//                 it, relative to the fundamental. This is the number that
//                 decides whether the saturator sounds clean, and it is what
//                 the CI gate is set on.
//     fullBandDb— alias energy anywhere below base Nyquist. This is expected
//                 to be much worse under the "audio" oversampling profile, and
//                 that is BY DESIGN, not a defect: a 17 kHz tone's third
//                 harmonic sits at 51 kHz, above the 48 kHz base Nyquist, so
//                 it can only be filtered away or folded. The audio profile
//                 deliberately spends its filter budget protecting 0-22 kHz
//                 and lets the ultrasonic region fold, because nobody can hear
//                 a component at 45 kHz. The wideband profile buys the full
//                 band back at the cost of much longer filters.
struct AsrResult {
  double inBandDb = 0.0;
  double fullBandDb = 0.0;
  bool collisionFree = true;
};

AsrResult measureAsr(const ParamSet& driveParams, double amplitude) {
  constexpr size_t kN = 32768;
  constexpr int64_t kSettle = 32768;
  constexpr int kBin = 5803; // -> ~17.0 kHz at 96 kHz; harmonic 3 already folds

  const double f0 = static_cast<double>(kBin) * kCanonicalSampleRate /
                    static_cast<double>(kN);
  const int64_t total = kSettle + static_cast<int64_t>(kN);

  AudioBuffer in(1, total);
  for (int64_t i = 0; i < total; ++i)
    in.channel(0)[i] = amplitude * std::sin(2.0 * M_PI * f0 *
                                            static_cast<double>(i) / kCanonicalSampleRate);

  Chain chain;
  Chain::SlotConfig cfg;
  cfg.nodeId = "drv";
  chain.add(std::make_unique<Drive>(), driveParams, cfg);
  chain.prepare(makeSpec(1, 1024, total));

  AudioBuffer out;
  chain.render(in, out, 1024);

  const auto power = powerSpectrum(out.channel(0) + kSettle, kN);

  // Classify every bin: 0 = unused, 1 = legitimate harmonic, 2 = alias.
  std::vector<int> role(kN / 2 + 1, 0);
  AsrResult result;
  for (int m = 1; m <= 400; ++m) {
    size_t bin = (static_cast<size_t>(m) * static_cast<size_t>(kBin)) % kN;
    if (bin > kN / 2) bin = kN - bin;
    if (bin == 0) continue;
    const bool legitimate = (static_cast<double>(m) * f0) < (kCanonicalSampleRate * 0.5);
    const int want = legitimate ? 1 : 2;
    if (role[bin] != 0 && role[bin] != want) result.collisionFree = false;
    // A legitimate harmonic wins the bin; that is the conservative choice
    // because it moves energy into the denominator, never into the numerator.
    if (role[bin] == 0 || want == 1) role[bin] = want;
  }

  const double hzPerBin = kCanonicalSampleRate / static_cast<double>(kN);
  const double audibleLimitHz = 20000.0;

  double signalPower = 0.0, aliasFull = 0.0, aliasInBand = 0.0;
  for (size_t b = 1; b <= kN / 2; ++b) {
    const double hz = static_cast<double>(b) * hzPerBin;
    if (role[b] == 1) {
      signalPower += power[b];
    } else if (role[b] == 2) {
      aliasFull += power[b];
      if (hz < audibleLimitHz) aliasInBand += power[b];
    }
  }

  const double denom = std::max(signalPower, 1e-300);
  result.inBandDb = 10.0 * std::log10(std::max(aliasInBand, 1e-300) / denom);
  result.fullBandDb = 10.0 * std::log10(std::max(aliasFull, 1e-300) / denom);
  return result;
}

void testAliasing() {
  struct Case {
    std::string label;
    std::string curve;
    std::string profile;
    double driveDb;
    int factor;
    double gateDb; // in-band ASR must be at or below this
  };

  // Gates are set from MEASURED behaviour with ~4 dB of headroom, not from
  // aspiration. They exist to catch regressions, so a number that drifts is a
  // failure even if it still sounds fine.
  //
  // Context for the absolute values: the probe tone is deliberately
  // pathological — 17 kHz driven 18 dB into saturation, so its third harmonic
  // is already above base Nyquist. Real program material, whose fundamentals
  // sit two decades lower, aliases far less than these numbers suggest.
  const std::vector<Case> cases = {
      {"soft clip 4x  (default)", "soft_clip", "wideband", 18.0, 4, -86.0},
      {"soft clip 8x", "soft_clip", "wideband", 18.0, 8, -85.0},
      {"soft clip 16x", "soft_clip", "wideband", 18.0, 16, -85.0},
      {"soft clip 4x  audio profile", "soft_clip", "audio", 18.0, 4, -95.0},
      {"hard clip 4x", "hard_clip", "wideband", 12.0, 4, -83.0},
      {"hard clip 16x", "hard_clip", "wideband", 12.0, 16, -83.0},
  };

  std::printf("       %-30s %11s %11s\n", "configuration", "in-band", "full-band");
  for (const auto& tc : cases) {
    ParamSet p;
    p.set("drive_db", tc.driveDb).set("oversample_factor", tc.factor)
        .set("bias", 0.0).set("output_trim_db", 0.0);
    p.setStr("curve", tc.curve);
    p.setStr("os_profile", tc.profile);

    const AsrResult r = measureAsr(p, 0.7);
    std::printf("       %-30s %8.1f dB %8.1f dB\n", tc.label.c_str(),
                r.inBandDb, r.fullBandDb);
    check(r.collisionFree, "ASR bin classification is unambiguous: " + tc.label);
    check(r.inBandDb <= tc.gateDb,
          "in-band ASR " + tc.label + " = " + std::to_string(r.inBandDb) +
              " dB, gate " + std::to_string(tc.gateDb) + " dB");
  }
}

// 12. Dynamics correctness.
//
//     Tests 1-9 would all pass for a compressor that consistently applied the
//     WRONG amount of gain reduction. These check the three things that decide
//     whether it is actually a compressor: that its static curve matches the
//     analytic soft-knee law, that its ballistics run at the times it claims,
//     and that lookahead genuinely eliminates transient overshoot.

// The expected soft-knee compression law, written out longhand.
//
// This deliberately does NOT call GainComputer. Using the production gain
// computer to generate the expected value would make the test self-referential
// — it would validate the compressor against itself and pass happily even if
// the law were completely wrong. (Verified: mutating GainComputer by 10% was
// invisible to the earlier version of this test.) An independent restatement
// of the maths is the only thing that can catch an error in the maths.
double expectedReductionDb(double inputDb, double thresholdDb, double ratio,
                           double kneeDb) {
  const double over = inputDb - thresholdDb;
  const double slope = 1.0 - 1.0 / ratio;
  if (kneeDb <= 0.0) return over <= 0.0 ? 0.0 : over * slope;
  if (over <= -kneeDb / 2.0) return 0.0;
  if (over >= kneeDb / 2.0) return over * slope;
  const double t = over + kneeDb / 2.0;
  return slope * t * t / (2.0 * kneeDb);
}

ParamSet compressorParams(double thresholdDb, double ratio, double kneeDb,
                          double attackMs, double releaseMs, double lookaheadMs) {
  ParamSet p;
  p.set("threshold_dbfs", thresholdDb).set("ratio_to_1", ratio)
      .set("knee_db", kneeDb).set("attack_ms", attackMs)
      .set("release_ms", releaseMs).set("lookahead_ms", lookaheadMs)
      .set("sidechain_hp_hz", 0.0).set("stereo_link", 1.0)
      .set("makeup_gain_db", 0.0);
  p.setStr("detector", "peak");
  p.setStr("release_mode", "fixed");
  return p;
}

std::unique_ptr<Chain> compressorChain(const ParamSet& p, int channels,
                                       int64_t frames) {
  auto chain = std::make_unique<Chain>();
  Chain::SlotConfig cfg;
  cfg.nodeId = "cmp";
  chain->add(std::make_unique<Compressor>(), p, cfg);
  chain->prepare(makeSpec(channels, 1024, frames));
  return chain;
}

// Steady-state gain for a sine at a given level.
double measureStaticGainDb(const ParamSet& p, double inputDb) {
  constexpr int64_t kSettle = 96000; // 1 s, so the envelope is fully settled
  constexpr int64_t kWindow = 8192;
  const int64_t total = kSettle + kWindow;
  const double amp = dbToLin(inputDb);

  AudioBuffer in(1, total);
  for (int64_t i = 0; i < total; ++i)
    in.channel(0)[i] = amp * std::sin(2.0 * M_PI * 1000.0 *
                                      static_cast<double>(i) / kCanonicalSampleRate);

  auto chain = compressorChain(p, 1, total);
  AudioBuffer out;
  chain->render(in, out, 1024);

  double sumIn = 0.0, sumOut = 0.0;
  for (int64_t i = kSettle; i < total; ++i) {
    sumIn += in.channel(0)[i] * in.channel(0)[i];
    sumOut += out.channel(0)[i] * out.channel(0)[i];
  }
  return 20.0 * std::log10(std::sqrt(sumOut) / std::sqrt(sumIn));
}

void testDynamics() {
  // --- (0) the sliding-window maximum, against brute force ----------------
  //
  // RunningMax is shared by the compressor, gate, dynamic EQ and limiter, so
  // a subtle bug here would propagate into four nodes at once. Checked against
  // an O(n*w) reference on a signal with plenty of ties and plateaus, which is
  // exactly where a monotonic-deque implementation goes wrong.
  {
    CounterRng rng(0xBEEF);
    for (int window : {1, 2, 3, 17, 64, 129}) {
      RunningMax rm;
      rm.prepare(window);
      std::vector<double> hist;
      bool ok = true;
      for (int64_t i = 0; i < 2000; ++i) {
        // Quantised coarsely so ties and plateaus occur often.
        const double x = std::round(rng.uniform(0, i) * 6.0) / 6.0;
        hist.push_back(x);
        const double got = rm.process(x);
        double want = -1e300;
        const size_t from = hist.size() >= static_cast<size_t>(window)
                                ? hist.size() - static_cast<size_t>(window)
                                : 0;
        for (size_t j = from; j < hist.size(); ++j) want = std::max(want, hist[j]);
        if (got != want) ok = false;
      }
      check(ok, "RunningMax matches brute force, window " + std::to_string(window));
    }
  }

  // --- (a) static curve ---------------------------------------------------
  {
    const double threshold = -20.0, ratio = 4.0;
    const ParamSet p = compressorParams(threshold, ratio, 0.0, 1.0, 1000.0, 0.0);

    for (double inputDb : {-40.0, -30.0, -20.0, -12.0, -6.0, 0.0}) {
      const double got = measureStaticGainDb(p, inputDb);
      const double want = -expectedReductionDb(inputDb, threshold, ratio, 0.0);
      checkNear(got, want, 0.25,
                "static curve at " + std::to_string(static_cast<int>(inputDb)) +
                    " dBFS (hard knee, 4:1)");
    }
  }

  // A soft knee must actually round the corner: at exactly the threshold a
  // 12 dB knee should already be reducing, where a hard knee reduces nothing.
  {
    const ParamSet soft = compressorParams(-20.0, 4.0, 12.0, 1.0, 1000.0, 0.0);
    const double atThreshold = measureStaticGainDb(soft, -20.0);
    checkNear(atThreshold, -expectedReductionDb(-20.0, -20.0, 4.0, 12.0), 0.25,
              "static curve at threshold (12 dB soft knee)");
    check(atThreshold < -0.5,
          "soft knee is already reducing at the threshold (got " +
              std::to_string(atThreshold) + " dB)");
  }

  // --- (b) attack and release timing --------------------------------------
  //
  // Measured on a DC step, which is the only input that presents the detector
  // with a genuine step rather than something modulated by the waveform.
  {
    const double attackMs = 20.0, releaseMs = 200.0;
    const ParamSet p = compressorParams(-20.0, 4.0, 0.0, attackMs, releaseMs, 0.0);

    const int64_t onset = 4096;
    const int64_t offset = onset + static_cast<int64_t>(0.5 * kCanonicalSampleRate);
    const int64_t total = offset + static_cast<int64_t>(1.5 * kCanonicalSampleRate);
    const double level = 0.5;

    AudioBuffer in(1, total);
    for (int64_t i = onset; i < offset; ++i) in.channel(0)[i] = level;

    auto chain = compressorChain(p, 1, total);
    AudioBuffer out;
    chain->render(in, out, 1024);

    const double finalGr = expectedReductionDb(linToDb(level), -20.0, 4.0, 0.0);

    // Attack: time to reach 63% of the final reduction.
    int64_t attackSamples = -1;
    for (int64_t i = onset; i < offset; ++i) {
      const double grDb = -20.0 * std::log10(std::abs(out.channel(0)[i]) / level);
      if (grDb >= 0.632 * finalGr) { attackSamples = i - onset; break; }
    }
    const double measuredAttackMs =
        static_cast<double>(attackSamples) * 1000.0 / kCanonicalSampleRate;
    check(attackSamples > 0 && measuredAttackMs > attackMs * 0.6 &&
              measuredAttackMs < attackMs * 1.6,
          "attack reaches 63% in " + std::to_string(measuredAttackMs) +
              " ms, spec " + std::to_string(attackMs) + " ms");

    check(finalGr > 3.0, "DC step produces measurable gain reduction (" +
                             std::to_string(finalGr) + " dB)");
  }

  // Release, measured by dropping to a level BELOW the threshold rather than
  // to silence. Dropping to silence would leave nothing to measure the gain
  // from, since gain is only observable as a ratio against a non-zero signal.
  {
    const double attackMs = 5.0, releaseMs = 200.0;
    const ParamSet p = compressorParams(-20.0, 4.0, 0.0, attackMs, releaseMs, 0.0);

    const int64_t onset = 4096;
    const int64_t drop = onset + static_cast<int64_t>(0.5 * kCanonicalSampleRate);
    const int64_t total = drop + static_cast<int64_t>(2.0 * kCanonicalSampleRate);
    const double loud = 0.5, quiet = 0.02; // quiet is well below threshold

    AudioBuffer in(1, total);
    for (int64_t i = onset; i < drop; ++i) in.channel(0)[i] = loud;
    for (int64_t i = drop; i < total; ++i) in.channel(0)[i] = quiet;

    auto chain = compressorChain(p, 1, total);
    AudioBuffer out;
    chain->render(in, out, 1024);

    const double startGr = expectedReductionDb(linToDb(loud), -20.0, 4.0, 0.0);

    int64_t releaseSamples = -1;
    for (int64_t i = drop; i < total; ++i) {
      const double grDb = -20.0 * std::log10(std::abs(out.channel(0)[i]) / quiet);
      if (grDb <= 0.368 * startGr) { releaseSamples = i - drop; break; }
    }
    const double measuredReleaseMs =
        static_cast<double>(releaseSamples) * 1000.0 / kCanonicalSampleRate;
    check(releaseSamples > 0 && measuredReleaseMs > releaseMs * 0.6 &&
              measuredReleaseMs < releaseMs * 2.0,
          "release recovers 63% in " + std::to_string(measuredReleaseMs) +
              " ms, spec " + std::to_string(releaseMs) + " ms");
  }

  // --- (c) lookahead eliminates overshoot ---------------------------------
  //
  // This is the offline superpower and the reason latency is worth spending.
  {
    const double threshold = -20.0, ratio = 20.0, attackMs = 5.0;
    const int64_t onset = 8192;
    const int64_t total = onset + static_cast<int64_t>(0.3 * kCanonicalSampleRate);

    AudioBuffer in(1, total);
    for (int64_t i = onset; i < total; ++i)
      in.channel(0)[i] = std::sin(2.0 * M_PI * 1000.0 *
                                  static_cast<double>(i - onset) / kCanonicalSampleRate);

    auto peakDbOf = [&](double lookaheadMs) {
      const ParamSet p =
          compressorParams(threshold, ratio, 0.0, attackMs, 100.0, lookaheadMs);
      auto chain = compressorChain(p, 1, total);
      AudioBuffer out;
      chain->render(in, out, 1024);
      double pk = 0.0;
      for (int64_t i = 0; i < out.frames(); ++i)
        pk = std::max(pk, std::abs(out.channel(0)[i]));
      return linToDb(pk);
    };

    const double steadyState = -expectedReductionDb(0.0, threshold, ratio, 0.0);
    const double finalGr = -steadyState;

    const double noLook = peakDbOf(0.0);
    const double look2x = peakDbOf(2.0 * attackMs);
    const double look4x = peakDbOf(4.0 * attackMs);

    std::printf("       overshoot: none %+.2f dB, 2x attack %+.2f dB, "
                "4x attack %+.2f dB, steady state %+.2f dB\n",
                noLook, look2x, look4x, steadyState);

    check(noLook > look2x + 3.0,
          "lookahead reduces transient overshoot by more than 3 dB (" +
              std::to_string(noLook - look2x) + " dB)");

    // The residual overshoot is not a defect — an exponential attack reaches
    // 1 - exp(-L/tau) of its target in L seconds and never quite arrives. This
    // asserts the ballistics follow that law exactly, which is a far stronger
    // statement about correctness than "the peak is low enough".
    for (auto [lookMs, measured] :
         {std::pair{2.0 * attackMs, look2x}, std::pair{4.0 * attackMs, look4x}}) {
      const double predicted =
          -finalGr * (1.0 - std::exp(-lookMs / attackMs));
      checkNear(measured, predicted, 0.5,
                "overshoot at " + std::to_string(lookMs / attackMs) +
                    "x attack matches the exponential-attack law");
    }

    // Practical guidance the planner depends on: lookahead of four attack
    // time constants puts the peak within half a dB of steady state.
    check(look4x <= steadyState + 0.5,
          "lookahead of 4x attack holds the peak at steady state (peak " +
              std::to_string(look4x) + " dB, steady state " +
              std::to_string(steadyState) + " dB)");
  }
}

// 13. Gate / expander correctness.
//
//     Same discipline as test 12: the expected curve is restated here rather
//     than obtained from ExpanderComputer, so an error in the production
//     maths cannot hide behind a test that shares it.

double expectedExpansionDb(double inputDb, double thresholdDb, double ratio,
                           double kneeDb, double rangeDb) {
  const double over = inputDb - thresholdDb;
  const double slope = ratio - 1.0;
  double reduction;
  if (kneeDb <= 0.0) {
    reduction = over >= 0.0 ? 0.0 : -over * slope;
  } else if (over >= kneeDb / 2.0) {
    reduction = 0.0;
  } else if (over <= -kneeDb / 2.0) {
    reduction = -over * slope;
  } else {
    const double t = over - kneeDb / 2.0;
    reduction = slope * t * t / (2.0 * kneeDb);
  }
  return std::min(reduction, rangeDb);
}

ParamSet gateParams(double thresholdDb, double ratio, double kneeDb,
                    double rangeDb, double attackMs, double holdMs,
                    double releaseMs, double lookaheadMs, double hysteresisDb) {
  ParamSet p;
  p.set("threshold_dbfs", thresholdDb).set("ratio_to_1", ratio)
      .set("knee_db", kneeDb).set("range_db", rangeDb)
      .set("attack_ms", attackMs).set("hold_ms", holdMs)
      .set("release_ms", releaseMs).set("lookahead_ms", lookaheadMs)
      .set("hysteresis_db", hysteresisDb).set("sidechain_hp_hz", 0.0)
      .set("stereo_link", 1.0);
  p.setStr("detector", "peak");
  return p;
}

std::unique_ptr<Chain> gateChain(const ParamSet& p, int64_t frames) {
  auto chain = std::make_unique<Chain>();
  Chain::SlotConfig cfg;
  cfg.nodeId = "gate";
  chain->add(std::make_unique<GateExpander>(), p, cfg);
  chain->prepare(makeSpec(1, 1024, frames));
  return chain;
}

double measureGateGainDb(const ParamSet& p, double inputDb) {
  constexpr int64_t kSettle = 192000; // 2 s; the gate's release is slow
  constexpr int64_t kWindow = 8192;
  const int64_t total = kSettle + kWindow;
  const double amp = dbToLin(inputDb);

  AudioBuffer in(1, total);
  for (int64_t i = 0; i < total; ++i)
    in.channel(0)[i] = amp * std::sin(2.0 * M_PI * 1000.0 *
                                      static_cast<double>(i) / kCanonicalSampleRate);

  auto chain = gateChain(p, total);
  AudioBuffer out;
  chain->render(in, out, 1024);

  double sumIn = 0.0, sumOut = 0.0;
  for (int64_t i = kSettle; i < total; ++i) {
    sumIn += in.channel(0)[i] * in.channel(0)[i];
    sumOut += out.channel(0)[i] * out.channel(0)[i];
  }
  return 20.0 * std::log10(std::sqrt(sumOut) / std::sqrt(sumIn));
}

void testGate() {
  // --- (a) static expansion curve, including the range clamp --------------
  {
    const double threshold = -30.0, ratio = 4.0, range = 40.0;
    const ParamSet p =
        gateParams(threshold, ratio, 0.0, range, 0.5, 0.0, 2000.0, 0.0, 0.0);

    for (double inputDb : {-20.0, -30.0, -35.0, -40.0, -50.0, -60.0}) {
      const double got = measureGateGainDb(p, inputDb);
      const double want =
          -expectedExpansionDb(inputDb, threshold, ratio, 0.0, range);
      checkNear(got, want, 0.3,
                "expansion curve at " + std::to_string(static_cast<int>(inputDb)) +
                    " dBFS (4:1 down, 40 dB range)");
    }

    // The range clamp is what keeps a gate from sounding like an edit. At
    // 30 dB below threshold a 4:1 expander would want 90 dB of attenuation.
    const double deep = measureGateGainDb(p, -60.0);
    checkNear(deep, -range, 0.3, "range clamps attenuation at 40 dB");
  }

  // --- (b) lookahead preserves the attack transient -----------------------
  //
  // The whole point of the node being offline. A real-time gate cannot know a
  // hit is coming, so the onset is swallowed while the gain ramps up.
  {
    const double attackMs = 5.0;
    const int64_t onset = 96000; // 1 s of silence first, so the gate is shut
    const int64_t burst = static_cast<int64_t>(0.2 * kCanonicalSampleRate);
    const int64_t total = onset + burst;

    AudioBuffer in(1, total);
    for (int64_t i = onset; i < total; ++i)
      in.channel(0)[i] = std::sin(2.0 * M_PI * 1000.0 *
                                  static_cast<double>(i - onset) / kCanonicalSampleRate);

    auto onsetLossDb = [&](double lookaheadMs) {
      const ParamSet p = gateParams(-30.0, 20.0, 0.0, 60.0, attackMs, 50.0,
                                    100.0, lookaheadMs, 0.0);
      auto chain = gateChain(p, total);
      AudioBuffer out;
      chain->render(in, out, 1024);

      // Energy in the first 3 ms of the burst, versus the input's.
      const int64_t win = static_cast<int64_t>(0.003 * kCanonicalSampleRate);
      double sumIn = 0.0, sumOut = 0.0;
      for (int64_t i = onset; i < onset + win; ++i) {
        sumIn += in.channel(0)[i] * in.channel(0)[i];
        sumOut += out.channel(0)[i] * out.channel(0)[i];
      }
      return 20.0 * std::log10(std::sqrt(sumOut) / std::sqrt(sumIn));
    };

    const double noLook = onsetLossDb(0.0);
    const double withLook = onsetLossDb(4.0 * attackMs);

    std::printf("       gate onset: no lookahead %+.2f dB, 4x attack lookahead "
                "%+.2f dB\n", noLook, withLook);

    check(withLook > noLook + 6.0,
          "lookahead recovers more than 6 dB of the swallowed attack (" +
              std::to_string(withLook - noLook) + " dB)");
    check(withLook > -1.0,
          "with lookahead the attack transient passes essentially intact (" +
              std::to_string(withLook) + " dB)");
  }

  // --- (b2) the running maximum holds the gate through brief dips ---------
  //
  // Test (b) above does NOT isolate this. Delaying the audio while the
  // detector runs undelayed is by itself enough to anticipate a step onset —
  // verified by deleting the running maximum entirely and watching (b) still
  // pass at -0.82 dB. What the running maximum actually buys is immunity to
  // level dips SHORTER than the lookahead window: without it the gate starts
  // closing on a momentary drop and audibly flutters.
  //
  // Probed with hold disabled, so the hold timer cannot mask the effect.
  {
    const double threshold = -30.0;
    const double loud = dbToLin(-20.0);
    const double quiet = dbToLin(-60.0);
    const int64_t total = static_cast<int64_t>(0.5 * kCanonicalSampleRate);
    const int64_t dipStart = total / 2;
    const int64_t dipLen = static_cast<int64_t>(0.003 * kCanonicalSampleRate); // 3 ms

    AudioBuffer in(1, total);
    for (int64_t i = 0; i < total; ++i) in.channel(0)[i] = loud;
    for (int64_t i = dipStart; i < dipStart + dipLen; ++i)
      in.channel(0)[i] = quiet;

    auto minGainAfterDipDb = [&](double lookaheadMs) {
      // Release is deliberately fast here: with a slow release the gain
      // barely moves during a short dip and the test measures nothing.
      const ParamSet p = gateParams(threshold, 10.0, 0.0, 60.0, 1.0, 0.0, 10.0,
                                    lookaheadMs, 0.0);
      auto chain = gateChain(p, total);
      AudioBuffer out;
      chain->render(in, out, 1024);

      // Scan the WHOLE loud region, skipping only the genuinely quiet
      // samples. Looking just after the dip would miss the failure entirely:
      // with lookahead L the gain computed at time i+L is applied to sample i,
      // so an un-anticipated dip shows up L samples EARLIER than the audio
      // event that caused it.
      double lowest = 1e9;
      for (int64_t i = 2000; i < total - 2000; ++i) {
        if (i >= dipStart && i < dipStart + dipLen) continue; // truly quiet
        lowest = std::min(lowest, 20.0 * std::log10(std::max(
                                       std::abs(out.channel(0)[i]) / loud, 1e-12)));
      }
      return lowest;
    };

    const double without = minGainAfterDipDb(0.0);
    const double with = minGainAfterDipDb(20.0); // window far longer than the dip

    std::printf("       gate dip: no lookahead %.2f dB, 20 ms lookahead %.2f dB\n",
                without, with);

    check(without < -3.0,
          "without lookahead a 3 ms dip makes the gate start closing (" +
              std::to_string(without) + " dB)");
    check(with > -0.1,
          "the running maximum holds the gate fully open through the dip (" +
              std::to_string(with) + " dB)");
  }

  // --- (c) hysteresis stops chatter ---------------------------------------
  //
  // A signal sitting on the threshold is the classic gate failure. Probed with
  // a DC level alternating 1 dB either side of it, which no amount of
  // ballistics smoothing can fix — only hysteresis can.
  {
    const double threshold = -30.0;
    const double hi = dbToLin(threshold + 1.0);
    const double lo = dbToLin(threshold - 1.0);
    const int64_t period = static_cast<int64_t>(0.1 * kCanonicalSampleRate);
    const int64_t total = period * 20;

    AudioBuffer in(1, total);
    for (int64_t i = 0; i < total; ++i)
      in.channel(0)[i] = ((i / period) % 2 == 0) ? hi : lo;

    auto gainSwingDb = [&](double hysteresisDb) {
      const ParamSet p = gateParams(threshold, 10.0, 0.0, 60.0, 1.0, 0.0, 50.0,
                                    0.0, hysteresisDb);
      auto chain = gateChain(p, total);
      AudioBuffer out;
      chain->render(in, out, 1024);

      double lowest = 1e9, highest = -1e9;
      for (int64_t i = total / 2; i < total; ++i) {
        const double g = 20.0 * std::log10(std::max(
            std::abs(out.channel(0)[i]) / std::abs(in.channel(0)[i]), 1e-12));
        lowest = std::min(lowest, g);
        highest = std::max(highest, g);
      }
      return highest - lowest;
    };

    const double without = gainSwingDb(0.0);
    const double with = gainSwingDb(6.0);

    std::printf("       gate chatter: no hysteresis %.2f dB swing, "
                "6 dB hysteresis %.2f dB swing\n", without, with);

    check(without > 5.0,
          "without hysteresis a threshold-straddling signal makes the gain "
          "chatter (" + std::to_string(without) + " dB swing)");
    check(with < 1.0,
          "6 dB of hysteresis holds the gate steady (" + std::to_string(with) +
              " dB swing)");
  }
}

// 14. Dynamic EQ correctness.
//
//     The load-bearing claim is the gain decomposition: that y = x + k(P-1)*BP
//     really is a peaking filter of gain P, and y = x + (P-1)*LP1 really is a
//     first-order shelf of gain P. If k were wrong the node would still be
//     stable, still block-size invariant, still pass tests 1-9 — and quietly
//     apply the wrong amount of EQ. These check the response directly.

ParamSet dynEqBand(const char* filterType, const char* mode, double freqHz,
                   double q, double staticGainDb, double thresholdDb,
                   double ratio, double kneeDb, double rangeDb) {
  ParamSet b;
  b.setStr("filter_type", filterType).setStr("mode", mode)
      .set("center_freq_hz", freqHz).set("q_factor", q)
      .set("static_gain_db", staticGainDb).set("threshold_dbfs", thresholdDb)
      .set("ratio_to_1", ratio).set("knee_db", kneeDb).set("range_db", rangeDb)
      .set("attack_ms", 1.0).set("release_ms", 2000.0);
  return b;
}

double measureDynEqGainDb(const ParamSet& bands, double probeHz, double probeDb) {
  constexpr int64_t kSettle = 192000;
  constexpr int64_t kWindow = 16384;
  const int64_t total = kSettle + kWindow;

  // Snap the probe onto the coherent grid so the measurement has no leakage.
  const double k = std::round(probeHz * static_cast<double>(kWindow) / kCanonicalSampleRate);
  const double f = k * kCanonicalSampleRate / static_cast<double>(kWindow);
  const double amp = dbToLin(probeDb);

  AudioBuffer in(1, total);
  for (int64_t i = 0; i < total; ++i)
    in.channel(0)[i] = amp * std::sin(2.0 * M_PI * f *
                                      static_cast<double>(i) / kCanonicalSampleRate);

  Chain chain;
  Chain::SlotConfig cfg;
  cfg.nodeId = "deq";
  chain.add(std::make_unique<DynamicEQ>(), bands, cfg);
  chain.prepare(makeSpec(1, 1024, total));

  AudioBuffer out;
  chain.render(in, out, 1024);

  double sumIn = 0.0, sumOut = 0.0;
  for (int64_t i = kSettle; i < total; ++i) {
    sumIn += in.channel(0)[i] * in.channel(0)[i];
    sumOut += out.channel(0)[i] * out.channel(0)[i];
  }
  return 20.0 * std::log10(std::sqrt(sumOut) / std::sqrt(sumIn));
}

// Exact digital response of a first-order shelf realised by the bilinear
// transform, restated independently of the production code. The frequency
// ratio is warped through tan() because that is what the TPT structure does.
double expectedFirstOrderShelfDb(double probeHz, double cornerHz, double gainDb,
                                 bool highShelf) {
  const double P = std::pow(10.0, gainDb / 20.0);
  const double ratio = std::tan(M_PI * probeHz / kCanonicalSampleRate) /
                       std::tan(M_PI * cornerHz / kCanonicalSampleRate);
  const std::complex<double> jw(0.0, ratio);
  const std::complex<double> lp = 1.0 / (1.0 + jw);
  const std::complex<double> shape = highShelf ? (1.0 - lp) : lp;
  return 20.0 * std::log10(std::abs(1.0 + (P - 1.0) * shape));
}

void testDynamicEq() {
  // --- (a) a static bell hits its commanded gain exactly at centre --------
  {
    for (double q : {0.7, 2.0, 5.0}) {
      for (double gainDb : {-12.0, -6.0, 6.0, 12.0}) {
        ParamSet bands;
        bands.addItem(dynEqBand("bell", "static", 1000.0, q, gainDb, -100.0,
                                4.0, 0.0, 0.0));
        const double got = measureDynEqGainDb(bands, 1000.0, -20.0);
        checkNear(got, gainDb, 0.1,
                  "static bell gain at centre, Q=" + std::to_string(q));
      }
    }

    // Two decades below centre a bell must be transparent, which is what
    // proves the mix coefficient is not leaking broadband gain.
    ParamSet bands;
    bands.addItem(dynEqBand("bell", "static", 5000.0, 4.0, 12.0, -100.0, 4.0,
                            0.0, 0.0));
    checkNear(measureDynEqGainDb(bands, 50.0, -20.0), 0.0, 0.1,
              "static bell is transparent two decades below centre");
  }

  // --- (b) first-order shelves match the analytic bilinear response -------
  {
    ParamSet high;
    high.addItem(dynEqBand("highshelf", "static", 8000.0, 0.707, 6.0, -100.0,
                           4.0, 0.0, 0.0));
    for (double probe : {2000.0, 8000.0, 20000.0}) {
      checkNear(measureDynEqGainDb(high, probe, -20.0),
                expectedFirstOrderShelfDb(probe, 8000.0, 6.0, true), 0.15,
                "high shelf at " + std::to_string(static_cast<int>(probe)) + " Hz");
    }

    ParamSet low;
    low.addItem(dynEqBand("lowshelf", "static", 200.0, 0.707, -6.0, -100.0,
                          4.0, 0.0, 0.0));
    for (double probe : {30.0, 200.0, 2000.0}) {
      checkNear(measureDynEqGainDb(low, probe, -20.0),
                expectedFirstOrderShelfDb(probe, 200.0, -6.0, false), 0.15,
                "low shelf at " + std::to_string(static_cast<int>(probe)) + " Hz");
    }
  }

  // --- (c) downward mode follows the dynamic curve ------------------------
  //
  // A sine at the band centre drives the normalised bandpass to unity, so the
  // band level equals the probe level and the expected gain is analytic.
  {
    const double threshold = -30.0, ratio = 5.0, range = 10.0;
    ParamSet bands;
    bands.addItem(dynEqBand("bell", "down", 4000.0, 2.0, 0.0, threshold, ratio,
                            0.0, range));

    for (double probeDb : {-45.0, -35.0, -30.0, -24.0, -12.0, -6.0}) {
      const double got = measureDynEqGainDb(bands, 4000.0, probeDb);
      const double reduction =
          std::min(range, expectedReductionDb(probeDb, threshold, ratio, 0.0));
      checkNear(got, -reduction, 0.3,
                "downward band at " + std::to_string(static_cast<int>(probeDb)) +
                    " dBFS probe");
    }

    // The range clamp is what stops a de-esser from removing the consonant
    // entirely on a loud sibilant.
    checkNear(measureDynEqGainDb(bands, 4000.0, 0.0), -range, 0.3,
              "downward band clamps at its range");
  }

  // --- (d) the sidechain is frequency selective ---------------------------
  //
  // The detector taps the band's own filter, so out-of-band energy — however
  // loud — must not engage it. This is the property that separates a dynamic
  // EQ from a compressor with an EQ after it.
  //
  // TWO TONES AT ONCE, deliberately. Measuring the gain at a lone out-of-band
  // probe proves nothing: an 8 kHz bell does not affect 400 Hz whether it is
  // engaged or not, so that measurement reads 0 dB either way. (Verified by
  // rewiring the sidechain to the broadband signal — the single-tone version
  // of this test did not notice at all.) The interferer must be out of band
  // while the thing being MEASURED is in band.
  {
    ParamSet bands;
    bands.addItem(dynEqBand("bell", "down", 8000.0, 3.0, 0.0, -30.0, 6.0, 0.0,
                            12.0));

    constexpr size_t kN = 16384;
    constexpr int64_t kSettle = 192000;
    const int64_t total = kSettle + static_cast<int64_t>(kN);

    auto binOf = [](double hz) {
      return static_cast<size_t>(std::round(hz * static_cast<double>(kN) /
                                            kCanonicalSampleRate));
    };
    const size_t probeBin = binOf(8000.0);
    const size_t interfererBin = binOf(400.0);
    const double probeHz =
        static_cast<double>(probeBin) * kCanonicalSampleRate / static_cast<double>(kN);
    const double interfererHz =
        static_cast<double>(interfererBin) * kCanonicalSampleRate / static_cast<double>(kN);

    auto probeGainDb = [&](double interfererDb) {
      const double pa = dbToLin(-50.0); // well below threshold on its own
      const double ia = dbToLin(interfererDb);

      AudioBuffer in(1, total);
      for (int64_t i = 0; i < total; ++i) {
        const double t = static_cast<double>(i) / kCanonicalSampleRate;
        in.channel(0)[i] = pa * std::sin(2.0 * M_PI * probeHz * t) +
                           ia * std::sin(2.0 * M_PI * interfererHz * t);
      }

      Chain chain;
      Chain::SlotConfig cfg;
      cfg.nodeId = "deq";
      chain.add(std::make_unique<DynamicEQ>(), bands, cfg);
      chain.prepare(makeSpec(1, 1024, total));
      AudioBuffer out;
      chain.render(in, out, 1024);

      const auto pin = powerSpectrum(in.channel(0) + kSettle, kN);
      const auto pout = powerSpectrum(out.channel(0) + kSettle, kN);
      return 10.0 * std::log10(pout[probeBin] / pin[probeBin]);
    };

    const double quietInterferer = probeGainDb(-60.0);
    const double loudInterferer = probeGainDb(-6.0);

    std::printf("       dynamic EQ selectivity: 8 kHz probe with quiet 400 Hz "
                "%+.2f dB, with loud 400 Hz %+.2f dB\n",
                quietInterferer, loudInterferer);

    checkNear(quietInterferer, 0.0, 0.2,
              "quiet out-of-band content leaves the band idle");
    checkNear(loudInterferer, 0.0, 0.5,
              "a LOUD out-of-band tone still leaves the band idle "
              "(broadband sidechain would cut the probe here)");
  }

  // --- (e) upward mode boosts when the band is quiet ----------------------
  {
    const double threshold = -30.0, ratio = 4.0, range = 8.0;
    ParamSet bands;
    bands.addItem(dynEqBand("bell", "up", 2000.0, 1.5, 0.0, threshold, ratio,
                            0.0, range));

    for (double probeDb : {-50.0, -40.0, -34.0, -30.0, -20.0}) {
      const double got = measureDynEqGainDb(bands, 2000.0, probeDb);
      // Mirror image of the downward law: reflect level and threshold.
      const double boost = std::min(
          range, expectedReductionDb(-probeDb, -threshold, ratio, 0.0));
      checkNear(got, boost, 0.3,
                "upward band at " + std::to_string(static_cast<int>(probeDb)) +
                    " dBFS probe");
    }
  }
}

// 15. True-peak limiter correctness.
//
//     REFERENCE MEASUREMENT. The first version of this test measured the
//     limiter's output with the engine's halfband cascade, on the reasoning
//     that a different code path makes an independent check. It does not:
//     that cascade under-reports broadband true peak by about the same amount
//     the limiter's own detector does, so the two errors cancelled and the
//     test passed while the limiter was overshooting by half a dB.
//
//     The reference below is instead a deliberately over-specified version of
//     the same estimator — 30x the filter length — and it is itself validated
//     against an analytic ground truth before being trusted (see the sine
//     check in testTruePeakLimiter). Accuracy matters more than code-path
//     independence for a measurement whose error is systematic.
double tightTruePeakDb(const AudioBuffer& buf) {
  TruePeakUpsampler up;
  up.design(8, 100.0, 0.010); // ~30x the taps of the limiter's own detector
  up.prepare(buf.channels());
  up.reset();

  double best = 0.0;
  for (int c = 0; c < buf.channels(); ++c)
    for (int64_t i = 0; i < buf.frames(); ++i)
      best = std::max(best, up.processMaxAbs(buf.channel(c)[i], c));
  return linToDb(best);
}

ParamSet limiterParams(double ceilingDbTp, double lookaheadMs, double releaseMs) {
  ParamSet p;
  p.set("ceiling_dbtp", ceilingDbTp).set("lookahead_ms", lookaheadMs)
      .set("release_ms", releaseMs).set("oversample_factor", 8.0);
  return p;
}

AudioBuffer runLimiter(const ParamSet& p, const AudioBuffer& in) {
  Chain chain;
  Chain::SlotConfig cfg;
  cfg.nodeId = "lim";
  chain.add(std::make_unique<TruePeakLimiter>(), p, cfg);
  chain.prepare(makeSpec(in.channels(), 1024, in.frames()));
  AudioBuffer out;
  chain.render(in, out, 1024);
  return out;
}

void testTruePeakLimiter() {
  // --- (0) validate the reference measurement itself ----------------------
  //
  // A sine whose period does not divide the sample rate never lands a sample
  // on its crest, so its true peak is analytically its amplitude while its
  // sample peak is strictly lower. If the reference cannot recover the known
  // answer, nothing measured with it below means anything.
  //
  // THE FADE MATTERS. A sine switched on abruptly at sample 0 has a genuine
  // derivative discontinuity, and the resulting inter-sample overshoot is a
  // real +0.08 dB of true peak — not a measurement artefact. The first version
  // of this check omitted the fade and blamed the meter for correctly
  // reporting it. A raised-cosine fade removes the transient so the analytic
  // answer is the amplitude and nothing else.
  {
    const int64_t n = 96000;
    const int64_t fade = 8192;
    for (double ampDb : {-0.5, -6.0, -20.0}) {
      const double amp = dbToLin(ampDb);
      AudioBuffer b(1, n);
      double samplePeak = 0.0;
      for (int64_t i = 0; i < n; ++i) {
        double env = 1.0;
        if (i < fade)
          env = 0.5 * (1.0 - std::cos(M_PI * static_cast<double>(i) /
                                      static_cast<double>(fade)));
        else if (i >= n - fade)
          env = 0.5 * (1.0 - std::cos(M_PI * static_cast<double>(n - 1 - i) /
                                      static_cast<double>(fade)));
        b.channel(0)[i] = env * amp *
                          std::sin(2.0 * M_PI * 7333.0 *
                                   static_cast<double>(i) / kCanonicalSampleRate);
        samplePeak = std::max(samplePeak, std::abs(b.channel(0)[i]));
      }
      checkNear(tightTruePeakDb(b), ampDb, 0.02,
                "reference true-peak meter recovers a known sine amplitude at " +
                    std::to_string(static_cast<int>(ampDb)) + " dBFS");
      // No assertion on sample peak here: over a long buffer the sine's
      // phase drifts until some sample lands essentially on the crest. The
      // fs/4-at-45-degrees case below is the proper "sample peak lies" probe.
      (void)samplePeak;
    }
  }

  // --- (a) the ceiling holds on deliberately hostile material -------------
  {
    const int64_t n = 96000;
    std::vector<std::pair<std::string, AudioBuffer>> cases;

    {   // 997 Hz square at 0 dBFS: near-maximal inter-sample overshoot, and
        // the classic torture test for a delivery limiter.
      AudioBuffer b(1, n);
      for (int64_t i = 0; i < n; ++i)
        b.channel(0)[i] = std::sin(2.0 * M_PI * 997.0 *
                                   static_cast<double>(i) / kCanonicalSampleRate) >= 0.0
                              ? 1.0 : -1.0;
      cases.emplace_back("997 Hz square @ 0 dBFS", std::move(b));
    }
    {   // Full-scale white noise.
      AudioBuffer b(1, n);
      CounterRng rng(31337);
      for (int64_t i = 0; i < n; ++i) b.channel(0)[i] = rng.bipolar(0, i);
      cases.emplace_back("full-scale white noise", std::move(b));
    }
    {   // fs/4 at 45 degrees: samples land at +/-A/sqrt(2) while the
        // continuous waveform reaches A. Sample peak lies by 3 dB.
      AudioBuffer b(1, n);
      for (int64_t i = 0; i < n; ++i)
        b.channel(0)[i] = std::sin(M_PI * static_cast<double>(i) / 2.0 + M_PI / 4.0);
      cases.emplace_back("fs/4 at 45 deg (max ISP)", std::move(b));
    }
    {   // Hard transient from silence.
      AudioBuffer b(1, n);
      for (int64_t i = n / 2; i < n; ++i)
        b.channel(0)[i] = 0.98 * std::sin(2.0 * M_PI * 220.0 *
                                          static_cast<double>(i - n / 2) / kCanonicalSampleRate);
      cases.emplace_back("transient from silence", std::move(b));
    }

    // Tolerances differ by signal class, and the reason is measured rather
    // than assumed. The limiter's detector is EXACT on realistic programme
    // material (0.000 dB error against the reference) but under-reports by up
    // to ~0.5 dB on full-scale white noise, whose true peak is genuinely
    // +6.2 dBFS. That is inherent to estimating true peak with a finite
    // filter, which is why BS.1770 defines the measurement by a specific
    // filter rather than as an abstract quantity.
    for (double ceiling : {-1.0, -0.3}) {
      for (const auto& kv : cases) {
        // Content with energy all the way to Nyquist — white noise and a
        // square wave — is where a finite-length true-peak estimator loses
        // accuracy. Band-limited material is held to a tight tolerance.
        const bool broadbandToNyquist =
            (kv.first == "full-scale white noise") ||
            (kv.first == "997 Hz square @ 0 dBFS");
        const double tolerance = broadbandToNyquist ? 0.6 : 0.05;
        const AudioBuffer out = runLimiter(limiterParams(ceiling, 2.0, 100.0), kv.second);
        const double measured = tightTruePeakDb(out);
        check(measured <= ceiling + tolerance,
              "true peak <= " + std::to_string(ceiling) + " dBTP on " + kv.first +
                  " (measured " + std::to_string(measured) + " dBTP, tolerance " +
                  std::to_string(tolerance) + ")");
        check(allFinite(out), "limiter output is finite on " + kv.first);
      }
    }

    // The escape hatch has to actually work: a safety margin must buy back
    // the estimation error even on the pathological case.
    {
      AudioBuffer noise(1, n);
      CounterRng rng(31337);
      for (int64_t i = 0; i < n; ++i) noise.channel(0)[i] = rng.bipolar(0, i);
      ParamSet p = limiterParams(-1.0, 2.0, 100.0);
      p.set("safety_margin_db", 0.8);
      const double measured = tightTruePeakDb(runLimiter(p, noise));
      std::printf("       limiter safety margin: white noise at -1 dBTP ceiling "
                  "+ 0.8 dB margin -> %+.2f dBTP\n", measured);
      check(measured <= -1.0,
            "a 0.8 dB safety margin holds the true ceiling even on full-scale "
            "noise (measured " + std::to_string(measured) + " dBTP)");
    }
  }

  // --- (b) inter-sample peaks are actually detected ------------------------
  //
  // The whole reason this node exists. On fs/4-at-45-degrees the SAMPLE peak
  // sits 3 dB below the true peak, so a sample-peak limiter would see -3.5
  // dBFS against a -1 dBTP ceiling and do nothing at all.
  {
    const int64_t n = 48000;
    const double amp = dbToLin(-0.5); // true peak -0.5 dBTP
    AudioBuffer in(1, n);
    for (int64_t i = 0; i < n; ++i)
      in.channel(0)[i] = amp * std::sin(M_PI * static_cast<double>(i) / 2.0 + M_PI / 4.0);

    double samplePeakIn = 0.0;
    for (int64_t i = 0; i < n; ++i)
      samplePeakIn = std::max(samplePeakIn, std::abs(in.channel(0)[i]));

    const double truePeakIn = tightTruePeakDb(in);
    const AudioBuffer out = runLimiter(limiterParams(-1.0, 2.0, 100.0), in);
    const double truePeakOut = tightTruePeakDb(out);

    std::printf("       limiter ISP: sample peak in %+.2f dBFS, true peak in "
                "%+.2f dBTP, true peak out %+.2f dBTP\n",
                linToDb(samplePeakIn), truePeakIn, truePeakOut);

    check(linToDb(samplePeakIn) < -3.0,
          "the probe's sample peak is well below the ceiling, so only a "
          "true-peak detector can catch it");
    check(truePeakIn > -1.0, "the probe's true peak does exceed the ceiling");
    check(truePeakOut <= -1.0 + 0.1,
          "the limiter caught an inter-sample peak the sample meter could not "
          "see (out " + std::to_string(truePeakOut) + " dBTP)");
  }

  // --- (c) transparent below the ceiling ----------------------------------
  //
  // A limiter that colours quiet material is worse than no limiter. Below the
  // ceiling the gain must be identically unity.
  {
    const int64_t n = 48000;
    AudioBuffer in(1, n);
    CounterRng rng(4242);
    for (int64_t i = 0; i < n; ++i)
      in.channel(0)[i] = 0.2 * (std::sin(2.0 * M_PI * 440.0 *
                                         static_cast<double>(i) / kCanonicalSampleRate) +
                                0.3 * rng.bipolar(0, i));

    const AudioBuffer out = runLimiter(limiterParams(-1.0, 2.0, 100.0), in);
    const double diff = maxAbsDiff(in, out, n);
    const double nullDb = linToDb(diff);
    std::printf("       limiter transparency below ceiling: null at %.1f dB\n",
                nullDb);
    check(nullDb < -200.0,
          "below the ceiling the limiter is transparent (null " +
              std::to_string(nullDb) + " dB)");
  }

  // --- (c2) the gain trajectory is smooth ---------------------------------
  //
  // The running minimum ALONE already guarantees the ceiling, so every ceiling
  // assertion above passes with the Hann smoothing deleted — verified by
  // mutation. What Hann actually buys is a continuous gain: without it the
  // minimum leaves the window as a single-sample step, and a step in gain is
  // broadband distortion sprayed across the spectrum.
  //
  // Probed with a lone full-scale spike on a steady floor, so the gain has to
  // fall and recover through its whole range. The DOWNWARD edge is the one
  // that matters — the release stage smooths the upward edge regardless.
  {
    const int64_t n = 48000;
    const int64_t spikeAt = n / 2;
    AudioBuffer in(1, n);
    for (int64_t i = 0; i < n; ++i) in.channel(0)[i] = 0.5;
    in.channel(0)[spikeAt] = 1.5;

    const AudioBuffer out = runLimiter(limiterParams(-1.0, 2.0, 100.0), in);

    double maxStep = 0.0, minGain = 1e9;
    double prev = out.channel(0)[0] / in.channel(0)[0];
    for (int64_t i = 1; i < n; ++i) {
      const double g = out.channel(0)[i] / in.channel(0)[i];
      maxStep = std::max(maxStep, std::abs(g - prev));
      minGain = std::min(minGain, g);
      prev = g;
    }

    std::printf("       limiter gain smoothness: max per-sample step %.5f, "
                "min gain %.3f\n", maxStep, minGain);

    check(minGain < 0.95,
          "the spike genuinely drove the gain down (min " +
              std::to_string(minGain) + ")");
    check(maxStep < 0.01,
          "the gain moves continuously rather than stepping (max step " +
              std::to_string(maxStep) + ")");
  }

  // --- (d) gain is linked across channels ---------------------------------
  //
  // Limiting channels independently would shift the stereo image on every
  // peak. A loud left channel must attenuate the right by the same amount.
  {
    const int64_t n = 48000;
    AudioBuffer in(2, n);
    for (int64_t i = 0; i < n; ++i) {
      in.channel(0)[i] = std::sin(2.0 * M_PI * 220.0 *
                                  static_cast<double>(i) / kCanonicalSampleRate);
      in.channel(1)[i] = 0.05 * std::sin(2.0 * M_PI * 330.0 *
                                         static_cast<double>(i) / kCanonicalSampleRate);
    }

    const AudioBuffer out = runLimiter(limiterParams(-6.0, 2.0, 100.0), in);

    // Compare the gain each channel actually received, mid-render.
    double gL = 0.0, gR = 0.0;
    int64_t at = n / 2;
    for (int64_t i = at; i < at + 2000; ++i) {
      if (std::abs(in.channel(0)[i]) > 0.5)
        gL = std::abs(out.channel(0)[i]) / std::abs(in.channel(0)[i]);
      if (std::abs(in.channel(1)[i]) > 0.025)
        gR = std::abs(out.channel(1)[i]) / std::abs(in.channel(1)[i]);
    }
    checkNear(linToDb(gR), linToDb(gL), 0.01,
              "both channels receive identical gain (stereo image preserved)");
    check(gL < 0.9, "the loud channel is genuinely being limited");
  }
}

// 16. Convolution reverb correctness.
//
//     This is why a convolution reverb was chosen for Phase 1 over an
//     algorithmic one: its correctness reduces to a question with an exact
//     answer. Direct time-domain convolution is the definition; the
//     partitioned FFT implementation is an optimisation of it. If they agree
//     to the last bits, the entire overlap-save / frequency-delay-line
//     machinery is right. An FDN reverb can only be judged by ear.

ParamSet reverbParams(double decayS, double preDelayMs, double decorrelation) {
  ParamSet p;
  p.set("decay_s", decayS).set("pre_delay_ms", preDelayMs)
      .set("hf_damp_hz", 6000.0).set("lf_cut_hz", 120.0)
      .set("hf_decay_ratio", 0.5).set("stereo_decorrelation", decorrelation);
  return p;
}

std::unique_ptr<Chain> reverbChain(const ParamSet& p, int channels, int64_t frames) {
  auto chain = std::make_unique<Chain>();
  Chain::SlotConfig cfg;
  cfg.nodeId = "rv"; // fixed, so the derived IR seed is stable across renders
  chain->add(std::make_unique<ConvolutionReverb>(), p, cfg);
  chain->prepare(makeSpec(channels, 1024, frames));
  return chain;
}

// The node's own impulse response, recovered by feeding it a delta.
AudioBuffer reverbIr(const ParamSet& p, int channels) {
  const int64_t n = 64;
  AudioBuffer in(channels, n);
  for (int c = 0; c < channels; ++c) in.channel(c)[0] = 1.0;
  auto chain = reverbChain(p, channels, n);
  AudioBuffer out;
  chain->render(in, out, 1024);
  return out;
}

void testConvolutionReverb() {
  // --- (a) partitioned FFT convolution == direct convolution --------------
  {
    const ParamSet p = reverbParams(0.05, 0.0, 0.0); // short IR, cheap to brute force
    const AudioBuffer ir = reverbIr(p, 1);

    const int64_t irLen = ir.frames();
    const int64_t n = 4096;
    AudioBuffer in(1, n);
    CounterRng rng(9182);
    for (int64_t i = 0; i < n; ++i) in.channel(0)[i] = 0.5 * rng.bipolar(0, i);

    auto chain = reverbChain(p, 1, n);
    AudioBuffer out;
    chain->render(in, out, 1024);

    // Direct time-domain convolution — the definition.
    double worst = 0.0, refPeak = 0.0;
    for (int64_t i = 0; i < n; ++i) {
      double acc = 0.0;
      const int64_t kMax = std::min<int64_t>(irLen - 1, i);
      for (int64_t k = 0; k <= kMax; ++k)
        acc += ir.channel(0)[k] * in.channel(0)[i - k];
      worst = std::max(worst, std::abs(acc - out.channel(0)[i]));
      refPeak = std::max(refPeak, std::abs(acc));
    }
    const double nullDb = 20.0 * std::log10(std::max(worst / refPeak, 1e-300));
    std::printf("       reverb: partitioned FFT vs direct convolution, null at "
                "%.1f dB\n", nullDb);
    check(nullDb < -200.0,
          "partitioned FFT convolution matches direct convolution (null " +
              std::to_string(nullDb) + " dB)");
  }

  // --- (b) the decay time is the one that was asked for -------------------
  //
  // Measured as T30 (the -5 to -35 dB span of the Schroeder energy decay
  // curve, doubled), the standard estimator, robust to the IR being truncated
  // before it reaches -60 dB.
  //
  // Asserted with hf_decay_ratio = 1.0 and a TIGHT tolerance. The two-band
  // decay model deliberately makes broadband T30 read shorter than decay_s —
  // measured at ~10% for the default ratio — so testing the default with a
  // loose tolerance would hide a systematic bias behind a passing check
  // instead of pinning down the behaviour. The frequency-dependent case is
  // asserted separately below, as a direction rather than a value.
  {
    for (double decay : {0.4, 1.0, 2.0}) {
      ParamSet flat = reverbParams(decay, 0.0, 0.0);
      flat.set("hf_decay_ratio", 1.0);
      const AudioBuffer ir = reverbIr(flat, 1);
      const int64_t n = ir.frames();

      // Schroeder backward integration.
      std::vector<double> edc(static_cast<size_t>(n), 0.0);
      double acc = 0.0;
      for (int64_t i = n - 1; i >= 0; --i) {
        acc += ir.channel(0)[i] * ir.channel(0)[i];
        edc[static_cast<size_t>(i)] = acc;
      }
      const double ref = edc[0];

      auto crossing = [&](double db) {
        const double target = ref * std::pow(10.0, db / 10.0);
        for (int64_t i = 0; i < n; ++i)
          if (edc[static_cast<size_t>(i)] <= target)
            return static_cast<double>(i) / kCanonicalSampleRate;
        return -1.0;
      };
      const double t5 = crossing(-5.0), t35 = crossing(-35.0);
      const double t30 = 2.0 * (t35 - t5);

      std::printf("       reverb decay: requested %.2f s, measured T30 %.3f s\n",
                  decay, t30);
      check(t5 >= 0.0 && t35 > t5, "energy decay curve is monotonic and spans "
                                   "-5 to -35 dB");
      checkNear(t30, decay, decay * 0.02,
                "measured T30 matches the requested decay time (flat decay)");
    }
  }

  // --- (b2) frequency-dependent decay shortens the broadband tail ---------
  //
  // A flat decay across the spectrum is the most obvious tell of a synthetic
  // reverb: real rooms absorb high frequencies far faster. This asserts the
  // model is actually doing that, without pinning a specific value.
  {
    auto t30For = [&](double hfRatio) {
      ParamSet p = reverbParams(1.0, 0.0, 0.0);
      p.set("hf_decay_ratio", hfRatio);
      const AudioBuffer ir = reverbIr(p, 1);
      const int64_t n = ir.frames();
      std::vector<double> edc(static_cast<size_t>(n), 0.0);
      double acc = 0.0;
      for (int64_t i = n - 1; i >= 0; --i) {
        acc += ir.channel(0)[i] * ir.channel(0)[i];
        edc[static_cast<size_t>(i)] = acc;
      }
      auto crossing = [&](double db) {
        const double target = edc[0] * std::pow(10.0, db / 10.0);
        for (int64_t i = 0; i < n; ++i)
          if (edc[static_cast<size_t>(i)] <= target)
            return static_cast<double>(i) / kCanonicalSampleRate;
        return -1.0;
      };
      return 2.0 * (crossing(-35.0) - crossing(-5.0));
    };

    const double flat = t30For(1.0);
    const double damped = t30For(0.5);
    std::printf("       reverb HF damping: flat decay T30 %.3f s, "
                "hf_decay_ratio 0.5 -> %.3f s\n", flat, damped);
    check(damped < flat * 0.97,
          "a faster high-band decay measurably shortens the broadband tail");
  }

  // --- (b3) output level is independent of decay time ---------------------
  //
  // The impulse response is normalised to unit energy so that reaching for a
  // longer tail does not also make the reverb louder. Without it, a 3 s decay
  // carries roughly ten times the energy of a 0.3 s one and every decay change
  // becomes a level change the engineer then has to undo.
  {
    const int64_t n = 65536;
    AudioBuffer in(1, n);
    CounterRng rng(2024);
    for (int64_t i = 0; i < n; ++i) in.channel(0)[i] = 0.3 * rng.bipolar(0, i);

    std::vector<double> levels;
    std::printf("       reverb level vs decay:");
    for (double decay : {0.3, 1.0, 3.0}) {
      auto chain = reverbChain(reverbParams(decay, 0.0, 0.0), 1, n);
      AudioBuffer out;
      chain->render(in, out, 1024);
      double sum = 0.0;
      for (int64_t i = 0; i < n; ++i) sum += out.channel(0)[i] * out.channel(0)[i];
      const double rmsDb = linToDb(std::sqrt(sum / static_cast<double>(n)));
      levels.push_back(rmsDb);
      std::printf("  %.1fs -> %+.2f dB", decay, rmsDb);
    }
    std::printf("\n");

    for (size_t i = 1; i < levels.size(); ++i)
      checkNear(levels[i], levels[0], 1.0,
                "output level is independent of decay time");
  }

  // --- (c) pre-delay offsets the response by exactly the time asked for ---
  {
    const double preDelayMs = 20.0;
    const AudioBuffer ir = reverbIr(reverbParams(0.3, preDelayMs, 0.0), 1);
    const int64_t expected =
        static_cast<int64_t>(std::lround(preDelayMs * 0.001 * kCanonicalSampleRate));

    int64_t onset = -1;
    for (int64_t i = 0; i < ir.frames(); ++i)
      if (std::abs(ir.channel(0)[i]) > 1e-9) { onset = i; break; }

    check(onset == expected,
          "pre-delay of " + std::to_string(preDelayMs) + " ms places the onset at " +
              std::to_string(onset) + ", expected " + std::to_string(expected));
  }

  // --- (d) stereo decorrelation actually decorrelates ---------------------
  //
  // Identical left and right impulse responses collapse a reverb to the
  // centre of the image, which is the single most common way a synthetic
  // reverb gives itself away.
  {
    const int64_t n = 32768;
    auto correlationFor = [&](double decorrelation) {
      AudioBuffer in(2, n);
      CounterRng rng(555);
      for (int64_t i = 0; i < n; ++i) {
        const double v = 0.5 * rng.bipolar(0, i);
        in.channel(0)[i] = v; // identical mono feed into both channels
        in.channel(1)[i] = v;
      }
      auto chain = reverbChain(reverbParams(0.4, 0.0, decorrelation), 2, n);
      AudioBuffer out;
      chain->render(in, out, 1024);

      double sLL = 0.0, sRR = 0.0, sLR = 0.0;
      for (int64_t i = 0; i < out.frames(); ++i) {
        const double l = out.channel(0)[i], r = out.channel(1)[i];
        sLL += l * l; sRR += r * r; sLR += l * r;
      }
      return sLR / std::sqrt(std::max(sLL * sRR, 1e-300));
    };

    // The parameter is continuous, so check it across its range rather than
    // only at the endpoints: by construction the inter-channel correlation
    // should come out at exactly (1 - decorrelation).
    std::printf("       reverb stereo correlation:");
    for (double d : {0.0, 0.25, 0.5, 0.75, 1.0}) {
      const double corr = correlationFor(d);
      std::printf("  d=%.2f->%.3f", d, corr);
      checkNear(corr, 1.0 - d, 0.06,
                "inter-channel correlation is 1-decorrelation at d=" +
                    std::to_string(d));
    }
    std::printf("\n");
  }
}

// 17. Delay correctness.

ParamSet delayParams(double timeMs, double feedbackPct, const char* stereoMode) {
  ParamSet p;
  p.set("time_ms", timeMs).set("feedback_percent", feedbackPct)
      .set("hf_damp_hz", 20000.0).set("lf_damp_hz", 20.0)
      .set("spread_ms", 0.0).set("max_tail_s", 4.0);
  p.setStr("stereo_mode", stereoMode);
  return p;
}

AudioBuffer runDelay(const ParamSet& p, const AudioBuffer& in) {
  Chain chain;
  Chain::SlotConfig cfg;
  cfg.nodeId = "dly";
  chain.add(std::make_unique<Delay>(), p, cfg);
  chain.prepare(makeSpec(in.channels(), 1024, in.frames()));
  AudioBuffer out;
  chain.render(in, out, 1024);
  return out;
}

void testDelay() {
  // --- (a) fractional delay lands where it was asked to -------------------
  //
  // Measured from the FFT phase at a low bin rather than by looking for a
  // peak, because a fractional delay by definition does not put its energy on
  // a sample boundary. A low probe frequency also keeps the phase unambiguous
  // and is where the first-order Thiran design is most accurate.
  {
    constexpr size_t kN = 16384;
    constexpr int kBin = 4; // ~23 Hz; one wrap is 4096 samples of delay
    const double f = static_cast<double>(kBin) * kCanonicalSampleRate /
                     static_cast<double>(kN);
    constexpr int64_t kSettle = 8192;
    const int64_t n = kSettle + static_cast<int64_t>(kN);

    for (double delaySamples : {480.0, 480.25, 480.5, 480.75, 1000.37}) {
      const double timeMs = delaySamples * 1000.0 / kCanonicalSampleRate;
      AudioBuffer in(1, n);
      for (int64_t i = 0; i < n; ++i)
        in.channel(0)[i] = 0.5 * std::sin(2.0 * M_PI * f *
                                          static_cast<double>(i) / kCanonicalSampleRate);

      const AudioBuffer out = runDelay(delayParams(timeMs, 0.0, "mono"), in);

      std::vector<std::complex<double>> X(kN), Y(kN);
      for (size_t i = 0; i < kN; ++i) {
        X[i] = in.channel(0)[kSettle + static_cast<int64_t>(i)];
        Y[i] = out.channel(0)[kSettle + static_cast<int64_t>(i)];
      }
      fftInPlace(X);
      fftInPlace(Y);

      double dPhase = std::arg(Y[kBin]) - std::arg(X[kBin]);
      const double perSample = 2.0 * M_PI * static_cast<double>(kBin) /
                               static_cast<double>(kN);
      double measured = -dPhase / perSample;
      // Unwrap onto the branch nearest the requested delay.
      const double wrap = static_cast<double>(kN) / static_cast<double>(kBin);
      while (measured - delaySamples > wrap * 0.5) measured -= wrap;
      while (delaySamples - measured > wrap * 0.5) measured += wrap;

      checkNear(measured, delaySamples, 0.02,
                "fractional delay of " + std::to_string(delaySamples) +
                    " samples is realised exactly");
    }
  }

  // --- (b) the interpolator is genuinely allpass --------------------------
  //
  // THE reason Thiran was chosen over Lagrange. In a feedback delay the
  // interpolator runs on every repeat, so any magnitude error compounds: a
  // Lagrange interpolator's gentle HF rolloff applied twenty times audibly
  // darkens the tail, and the engineer cannot tell it apart from the damping
  // control. An allpass has exactly unity magnitude, so repeats stay bright.
  {
    const double delaySamples = 480.37; // deliberately fractional
    const double timeMs = delaySamples * 1000.0 / kCanonicalSampleRate;
    constexpr int64_t kSettle = 16384;
    constexpr int64_t kWindow = 16384;
    const int64_t n = kSettle + kWindow;

    double worst = 0.0;
    for (double probe : {100.0, 1000.0, 5000.0, 12000.0, 20000.0}) {
      const double k = std::round(probe * static_cast<double>(kWindow) / kCanonicalSampleRate);
      const double f = k * kCanonicalSampleRate / static_cast<double>(kWindow);

      AudioBuffer in(1, n);
      for (int64_t i = 0; i < n; ++i)
        in.channel(0)[i] = 0.5 * std::sin(2.0 * M_PI * f *
                                          static_cast<double>(i) / kCanonicalSampleRate);
      const AudioBuffer out = runDelay(delayParams(timeMs, 0.0, "mono"), in);

      double sIn = 0.0, sOut = 0.0;
      for (int64_t i = kSettle; i < n; ++i) {
        sIn += in.channel(0)[i] * in.channel(0)[i];
        sOut += out.channel(0)[i] * out.channel(0)[i];
      }
      const double db = 20.0 * std::log10(std::sqrt(sOut) / std::sqrt(sIn));
      worst = std::max(worst, std::abs(db));
    }
    std::printf("       delay interpolator: worst magnitude error across "
                "100 Hz-20 kHz = %.4f dB\n", worst);
    check(worst < 0.01,
          "fractional delay is allpass to within 0.01 dB (Lagrange would roll "
          "off at the top, and the error would compound on every repeat)");
  }

  // --- (c) feedback repeats decay at the rate asked for -------------------
  {
    const double timeMs = 50.0;
    const double feedback = 0.5;
    const int64_t period = static_cast<int64_t>(timeMs * 0.001 * kCanonicalSampleRate);
    const int64_t n = period * 8;

    // A narrowband burst, so the wide-open damping filters barely touch it and
    // the repeat amplitudes reflect the feedback alone.
    AudioBuffer in(1, n);
    const int64_t burst = period / 4;
    for (int64_t i = 0; i < burst; ++i) {
      const double w = 0.5 * (1.0 - std::cos(2.0 * M_PI * static_cast<double>(i) /
                                             static_cast<double>(burst)));
      in.channel(0)[i] = w * std::sin(2.0 * M_PI * 1000.0 *
                                      static_cast<double>(i) / kCanonicalSampleRate);
    }

    const AudioBuffer out = runDelay(delayParams(timeMs, feedback * 100.0, "mono"), in);

    std::vector<double> peaks;
    for (int rep = 1; rep <= 5; ++rep) {
      double pk = 0.0;
      for (int64_t i = rep * period; i < rep * period + burst && i < out.frames(); ++i)
        pk = std::max(pk, std::abs(out.channel(0)[i]));
      peaks.push_back(pk);
    }

    std::printf("       delay repeats:");
    for (size_t i = 0; i < peaks.size(); ++i) std::printf(" %.4f", peaks[i]);
    std::printf("\n");

    for (size_t i = 1; i < peaks.size(); ++i) {
      const double ratio = peaks[i] / peaks[i - 1];
      checkNear(ratio, feedback, 0.02,
                "repeat " + std::to_string(i + 1) + " is the feedback fraction "
                "of the one before it");
    }
  }

  // --- (c2) damping sits INSIDE the feedback loop -------------------------
  //
  // Every other delay test deliberately opens the damping filters wide so they
  // do not interfere, which left the loop placement untested — verified by
  // moving damping out of the loop and watching the whole suite still pass.
  //
  // In-loop damping is what makes each repeat darker than the last, the way an
  // analog or tape delay behaves. Applied to the output instead, every repeat
  // would be an identical copy at lower level.
  {
    const int64_t period = 4096; // power of two, so repeat windows are FFT-sized
    const double timeMs = static_cast<double>(period) * 1000.0 / kCanonicalSampleRate;
    const int64_t n = period * 6;

    ParamSet p = delayParams(timeMs, 70.0, "mono");
    p.set("hf_damp_hz", 2000.0); // deliberately audible damping

    AudioBuffer in(1, n);
    CounterRng rng(77);
    const int64_t burst = period / 4;
    for (int64_t i = 0; i < burst; ++i) in.channel(0)[i] = 0.5 * rng.bipolar(0, i);

    const AudioBuffer out = runDelay(p, in);

    // Fraction of each repeat's energy above 4 kHz.
    auto hfFraction = [&](int rep) {
      std::vector<double> w(static_cast<size_t>(period), 0.0);
      for (int64_t i = 0; i < period; ++i) {
        const int64_t idx = rep * period + i;
        if (idx < out.frames()) w[static_cast<size_t>(i)] = out.channel(0)[idx];
      }
      const auto pw = powerSpectrum(w.data(), static_cast<size_t>(period));
      const double hzPerBin = kCanonicalSampleRate / static_cast<double>(period);
      double total = 0.0, high = 0.0;
      for (size_t b = 1; b < pw.size(); ++b) {
        total += pw[b];
        if (static_cast<double>(b) * hzPerBin > 4000.0) high += pw[b];
      }
      return total > 0.0 ? high / total : 0.0;
    };

    std::printf("       delay in-loop damping, HF fraction by repeat:");
    std::vector<double> fractions;
    for (int rep = 1; rep <= 4; ++rep) {
      fractions.push_back(hfFraction(rep));
      std::printf(" %.4f", fractions.back());
    }
    std::printf("\n");

    for (size_t i = 1; i < fractions.size(); ++i)
      check(fractions[i] < fractions[i - 1] * 0.9,
            "repeat " + std::to_string(i + 1) +
                " is measurably darker than the one before (in-loop damping)");
  }

  // --- (c3) the feedback cap holds under an illegal value -----------------
  //
  // The generic parameter sweep cannot catch this. It renders 2048 frames
  // against a 1920-sample delay, so only a single loop pass happens and a
  // runaway never has time to reach Inf — verified by removing the cap and
  // watching the sweep pass. Detecting exponential growth needs enough
  // repeats for it to actually grow, so it gets a targeted test.
  //
  // Boundedness rather than finiteness is the assertion: a loop gain above
  // unity is already a bug at the first repeat, long before it overflows.
  {
    const double timeMs = 20.0;
    const int64_t period = static_cast<int64_t>(timeMs * 0.001 * kCanonicalSampleRate);
    const int64_t n = period * 40; // ~40 repeats, ample room to diverge

    AudioBuffer in(1, n);
    const int64_t burst = period / 2;
    double inPeak = 0.0;
    for (int64_t i = 0; i < burst; ++i) {
      in.channel(0)[i] = std::sin(2.0 * M_PI * 1000.0 *
                                  static_cast<double>(i) / kCanonicalSampleRate);
      inPeak = std::max(inPeak, std::abs(in.channel(0)[i]));
    }

    // 500% feedback is not a legal value; the node must clamp it rather than
    // trusting the caller, because the caller is going to be a language model.
    ParamSet p = delayParams(timeMs, 500.0, "mono");
    const AudioBuffer out = runDelay(p, in);

    double outPeak = 0.0;
    for (int64_t i = 0; i < out.frames(); ++i)
      outPeak = std::max(outPeak, std::abs(out.channel(0)[i]));

    std::printf("       delay feedback cap: 500%% requested, output peak %.3f "
                "vs input peak %.3f\n", outPeak, inPeak);
    check(allFinite(out), "illegal feedback produces no NaN or Inf");
    check(outPeak < inPeak * 10.0,
          "illegal feedback is clamped rather than allowed to diverge (peak " +
              std::to_string(outPeak) + ")");
  }

  // --- (d) tempo sync overrides the millisecond time ----------------------
  {
    ParamSet p = delayParams(123.0, 0.0, "mono"); // deliberately wrong ms
    p.set("tempo_bpm", 120.0);
    p.setStr("time_note_division", "quarter");

    const int64_t n = 96000;
    AudioBuffer in(1, n);
    in.channel(0)[0] = 1.0;
    const AudioBuffer out = runDelay(p, in);

    int64_t onset = -1;
    for (int64_t i = 0; i < out.frames(); ++i)
      if (std::abs(out.channel(0)[i]) > 1e-9) { onset = i; break; }

    // 120 BPM -> a quarter note is 500 ms -> 48000 samples at 96 kHz.
    check(onset == 48000,
          "a quarter note at 120 BPM is 48000 samples, and overrides time_ms "
          "(onset " + std::to_string(onset) + ")");
  }

  // --- (e) ping-pong alternates across the image --------------------------
  {
    const double timeMs = 50.0;
    const int64_t period = static_cast<int64_t>(timeMs * 0.001 * kCanonicalSampleRate);
    const int64_t n = period * 6;

    AudioBuffer in(2, n);
    const int64_t burst = period / 4;
    for (int64_t i = 0; i < burst; ++i) {
      const double v = std::sin(2.0 * M_PI * 1000.0 *
                                static_cast<double>(i) / kCanonicalSampleRate);
      in.channel(0)[i] = v; // identical on both: a centred mono source
      in.channel(1)[i] = v;
    }

    const AudioBuffer out = runDelay(delayParams(timeMs, 60.0, "ping_pong"), in);

    auto energy = [&](int ch, int rep) {
      double e = 0.0;
      for (int64_t i = rep * period; i < rep * period + burst && i < out.frames(); ++i)
        e += out.channel(ch)[i] * out.channel(ch)[i];
      return e;
    };

    std::printf("       ping-pong energy L/R by repeat:");
    for (int rep = 1; rep <= 4; ++rep)
      std::printf(" [%d] %.3f/%.3f", rep, energy(0, rep), energy(1, rep));
    std::printf("\n");

    // Odd repeats land left, even repeats right.
    check(energy(0, 1) > energy(1, 1) * 100.0, "repeat 1 is on the left");
    check(energy(1, 2) > energy(0, 2) * 100.0, "repeat 2 is on the right");
    check(energy(0, 3) > energy(1, 3) * 100.0, "repeat 3 is back on the left");
  }
}

// 18. Utility node correctness.
//
//     Small nodes, but the ones every chain leans on. A pan law that dips in
//     the middle, or a mid/side bracket that does not close, is the kind of
//     error that is invisible per-node and obvious across a whole mix.

// Generic single-node steady-state gain at one frequency, coherently sampled.
double measureNodeGainDb(const std::function<std::unique_ptr<Node>()>& make,
                         const ParamSet& params, double probeHz, int channel = 0,
                         int channels = 1, double* snappedHzOut = nullptr) {
  constexpr int64_t kSettle = 32768;
  constexpr int64_t kWindow = 16384;
  const int64_t total = kSettle + kWindow;
  const double k = std::round(probeHz * static_cast<double>(kWindow) / kCanonicalSampleRate);
  const double f = k * kCanonicalSampleRate / static_cast<double>(kWindow);
  if (snappedHzOut) *snappedHzOut = f;

  AudioBuffer in(channels, total);
  for (int c = 0; c < channels; ++c)
    for (int64_t i = 0; i < total; ++i)
      in.channel(c)[i] = 0.25 * std::sin(2.0 * M_PI * f *
                                         static_cast<double>(i) / kCanonicalSampleRate);

  Chain chain;
  Chain::SlotConfig cfg;
  cfg.nodeId = "u";
  chain.add(make(), params, cfg);
  chain.prepare(makeSpec(channels, 1024, total));
  AudioBuffer out;
  chain.render(in, out, 1024);

  double sIn = 0.0, sOut = 0.0;
  for (int64_t i = kSettle; i < total; ++i) {
    sIn += in.channel(channel)[i] * in.channel(channel)[i];
    sOut += out.channel(channel)[i] * out.channel(channel)[i];
  }
  return 20.0 * std::log10(std::sqrt(sOut) / std::sqrt(std::max(sIn, 1e-300)));
}

void testUtilityNodes() {
  // --- Pan: constant power across the whole sweep -------------------------
  {
    auto make = [] { return std::unique_ptr<Node>(new Pan()); };
    for (double pos : {-1.0, -0.5, 0.0, 0.5, 1.0}) {
      ParamSet p;
      p.set("pan_position", pos);
      const double gl = dbToLin(measureNodeGainDb(make, p, 1000.0, 0, 2));
      const double gr = dbToLin(measureNodeGainDb(make, p, 1000.0, 1, 2));
      checkNear(gl * gl + gr * gr, 1.0, 1e-6,
                "pan holds constant power at position " + std::to_string(pos));
    }

    // Centre is -3.01 dB per side. That is the law working, not a defect: it
    // is exactly what makes a centred source sum to unity in mono.
    ParamSet centre;
    centre.set("pan_position", 0.0);
    checkNear(measureNodeGainDb(make, centre, 1000.0, 0, 2), -3.0103, 0.01,
              "pan centre is -3.01 dB on the left");
    checkNear(measureNodeGainDb(make, centre, 1000.0, 1, 2), -3.0103, 0.01,
              "pan centre is -3.01 dB on the right");

    ParamSet hardLeft;
    hardLeft.set("pan_position", -1.0);
    checkNear(measureNodeGainDb(make, hardLeft, 1000.0, 0, 2), 0.0, 0.01,
              "hard left passes the left channel at unity");
    check(dbToLin(measureNodeGainDb(make, hardLeft, 1000.0, 1, 2)) < 1e-9,
          "hard left silences the right channel");
  }

  // --- Polarity: exact negation, bit for bit ------------------------------
  {
    const int64_t n = 4096;
    const AudioBuffer in = testSignal(2, n);
    ParamSet p;
    p.set("inverted", 1.0);
    Chain chain;
    Chain::SlotConfig cfg;
    cfg.nodeId = "pol";
    chain.add(std::unique_ptr<Node>(new Polarity()), p, cfg);
    chain.prepare(makeSpec(2, 512, n));
    AudioBuffer out;
    chain.render(in, out, 512);

    double worst = 0.0;
    for (int c = 0; c < 2; ++c)
      for (int64_t i = 0; i < n; ++i)
        worst = std::max(worst, std::abs(out.channel(c)[i] + in.channel(c)[i]));
    check(worst == 0.0, "polarity inversion is exact negation (bit for bit)");
  }

  // --- DC block: kills DC, leaves the programme alone ---------------------
  {
    auto make = [] { return std::unique_ptr<Node>(new DCBlock()); };
    ParamSet p;
    p.set("cutoff_hz", 10.0);

    // The probe frequency has to be the one actually used. At 10 Hz the
    // coherent grid is 5.9 Hz wide, so a request for 10 Hz lands on 11.7 Hz —
    // where a 10 Hz second-order Butterworth is legitimately -1.85 dB, not
    // -3.01 dB. Comparing the two was a test bug that looked like a filter bug.
    //
    // The expected magnitude is restated here from the analytic Butterworth
    // response, with the frequency ratio prewarped through tan() because the
    // filter is bilinear-transformed.
    auto butterworthHpDb = [](double f, double fc) {
      const double r = std::tan(M_PI * f / kCanonicalSampleRate) /
                       std::tan(M_PI * fc / kCanonicalSampleRate);
      const double r4 = r * r * r * r;
      return 10.0 * std::log10(r4 / (1.0 + r4));
    };

    double snapped = 0.0;
    const double atCorner = measureNodeGainDb(make, p, 10.0, 0, 1, &snapped);
    checkNear(atCorner, butterworthHpDb(snapped, 10.0), 0.05,
              "DC blocker matches the analytic Butterworth response near its corner");

    // And check the -3 dB point itself, probed high enough that the grid is
    // fine relative to the frequency.
    ParamSet p200;
    p200.set("cutoff_hz", 200.0);
    double snapped200 = 0.0;
    const double at200 = measureNodeGainDb(make, p200, 200.0, 0, 1, &snapped200);
    checkNear(at200, -3.0103, 0.1,
              "a second-order Butterworth high-pass is -3.01 dB at its corner");
    checkNear(measureNodeGainDb(make, p, 1000.0, 0, 1), 0.0, 0.01,
              "DC blocker is transparent at 1 kHz");
    checkNear(measureNodeGainDb(make, p, 40.0, 0, 1), 0.0, 0.2,
              "DC blocker is essentially transparent two octaves up");

    // Actual DC must decay away.
    const int64_t n = 96000;
    AudioBuffer dc(1, n);
    for (int64_t i = 0; i < n; ++i) dc.channel(0)[i] = 1.0;
    Chain chain;
    Chain::SlotConfig cfg;
    cfg.nodeId = "dcb";
    chain.add(make(), p, cfg);
    chain.prepare(makeSpec(1, 1024, n));
    AudioBuffer out;
    chain.render(dc, out, 1024);
    double residual = 0.0;
    for (int64_t i = n - 4096; i < n; ++i)
      residual = std::max(residual, std::abs(out.channel(0)[i]));
    check(residual < 1e-6,
          "a full-scale DC input decays to nothing (residual " +
              std::to_string(residual) + ")");
  }

  // --- Mono to stereo: every channel carries the source -------------------
  {
    const int64_t n = 4096;
    AudioBuffer in(2, n);
    CounterRng rng(4);
    for (int64_t i = 0; i < n; ++i) {
      in.channel(0)[i] = 0.4 * rng.bipolar(0, i);
      in.channel(1)[i] = 0.0; // right deliberately silent
    }
    ParamSet p;
    p.set("source_channel", 0.0);
    Chain chain;
    Chain::SlotConfig cfg;
    cfg.nodeId = "m2s";
    chain.add(std::unique_ptr<Node>(new MonoToStereo()), p, cfg);
    chain.prepare(makeSpec(2, 512, n));
    AudioBuffer out;
    chain.render(in, out, 512);

    double worst = 0.0, energy = 0.0;
    for (int64_t i = 0; i < n; ++i) {
      worst = std::max(worst, std::abs(out.channel(1)[i] - in.channel(0)[i]));
      energy += out.channel(1)[i] * out.channel(1)[i];
    }
    check(worst == 0.0, "the silent channel now carries the source exactly");
    check(energy > 0.0, "and it is not still silent");
  }

  // --- Mid/side: the bracket closes ---------------------------------------
  {
    const int64_t n = 4096;
    const AudioBuffer in = testSignal(2, n);

    ParamSet enc, dec;
    enc.setStr("mode", "encode");
    dec.setStr("mode", "decode");

    Chain chain;
    Chain::SlotConfig c1, c2;
    c1.nodeId = "ms_enc";
    c2.nodeId = "ms_dec";
    chain.add(std::unique_ptr<Node>(new MidSide()), enc, c1);
    chain.add(std::unique_ptr<Node>(new MidSide()), dec, c2);
    chain.prepare(makeSpec(2, 512, n));
    AudioBuffer out;
    chain.render(in, out, 512);

    // NOT linToDb here: its -240 dB floor exists to keep audio level maths
    // finite, and using it to report a null silently clamps the answer. The
    // first version of this check read exactly -240.0 dB and failed a -280 dB
    // assertion that the node was in fact meeting comfortably.
    const double diff = maxAbsDiff(in, out, n);
    const double nullDb = 20.0 * std::log10(std::max(diff, 1e-300));
    std::printf("       mid/side encode->decode round trip: null at %.1f dB\n",
                nullDb);
    check(nullDb < -280.0,
          "encode followed by decode returns the input (null " +
              std::to_string(nullDb) + " dB)");

    // A perfectly centred source has no side content at all.
    AudioBuffer mono(2, n);
    CounterRng rng(11);
    for (int64_t i = 0; i < n; ++i) {
      const double v = 0.5 * rng.bipolar(0, i);
      mono.channel(0)[i] = v;
      mono.channel(1)[i] = v;
    }
    Chain encChain;
    encChain.add(std::unique_ptr<Node>(new MidSide()), enc, c1);
    encChain.prepare(makeSpec(2, 512, n));
    AudioBuffer encoded;
    encChain.render(mono, encoded, 512);

    double sidePeak = 0.0;
    for (int64_t i = 0; i < n; ++i)
      sidePeak = std::max(sidePeak, std::abs(encoded.channel(1)[i]));
    check(sidePeak == 0.0,
          "a centred source encodes to exactly zero side content");
  }
}

} // namespace

// Per-section timings go to STDERR, deliberately. Stdout carries the pass/fail
// record and must stay byte-identical between runs so it can be diffed; wall
// times never are.
struct SectionTimer {
  explicit SectionTimer(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
  ~SectionTimer() {
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0).count();
    if (ms >= 200) std::fprintf(stderr, "  [%6lld ms] %s\n", (long long)ms, name);
  }
  const char* name;
  std::chrono::steady_clock::time_point t0;
};

#define TIMED(label) SectionTimer _timer_##__LINE__(label)

int main() {
  ScopedNoDenormals guard;

  const auto all = registry();
  std::printf("Audionova engine contract tests — %zu node(s)\n\n", all.size());

  std::printf("[ 1] bypass null\n");            { TIMED("bypass null"); testBypassNull(all); }
  std::printf("[ 2] block-size invariance\n");  { TIMED("block-size invariance"); testBlockSizeInvariance(all); }
  std::printf("[ 3] determinism\n");            { TIMED("determinism"); testDeterminism(all); }
  std::printf("[ 4] reset equivalence\n");      { TIMED("reset equivalence"); testResetEquivalence(all); }
  std::printf("[ 5] latency correctness\n");    { TIMED("latency"); testLatency(all); }
  std::printf("[ 6] silence -> silence\n");     { TIMED("silence"); testSilence(all); }
  std::printf("[ 7] robustness\n");             { TIMED("robustness"); testRobustness(all); }
  std::printf("[7b] parameter robustness\n");  { TIMED("parameter robustness"); testParameterRobustness(all); }
  std::printf("[ 8] denormal control\n");       testDenormalsEnabled();
  std::printf("[ 9] counter-based RNG\n");      testRngIsCounterBased();
  std::printf("[10] frequency response\n");     { TIMED("frequency response"); testFrequencyResponse(); }
  std::printf("[11] aliasing (ASR)\n");         { TIMED("aliasing"); testAliasing(); }
  std::printf("[12] dynamics correctness\n"); { TIMED("dynamics"); testDynamics(); }
  std::printf("[13] gate correctness\n");      { TIMED("gate"); testGate(); }
  std::printf("[14] dynamic EQ correctness\n"); { TIMED("dynamic EQ"); testDynamicEq(); }
  std::printf("[15] true-peak limiter\n");    { TIMED("true-peak limiter"); testTruePeakLimiter(); }
  std::printf("[16] convolution reverb\n");  { TIMED("convolution reverb"); testConvolutionReverb(); }
  std::printf("[17] delay correctness\n");   { TIMED("delay"); testDelay(); }
  std::printf("[18] utility nodes\n");        { TIMED("utility"); testUtilityNodes(); }

  std::printf("\n%d checks, %d failure(s)\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
