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
#include "anv/nodes/Chorus.h"
#include "anv/nodes/Compressor.h"
#include "anv/nodes/Console.h"
#include "anv/nodes/DCBlock.h"
#include "anv/nodes/ConvolutionReverb.h"
#include "anv/nodes/Delay.h"
#include "anv/nodes/Dither.h"
#include "anv/nodes/Drive.h"
#include "anv/nodes/DynamicEQ.h"
#include "anv/nodes/GateExpander.h"
#include "anv/nodes/LinearPhaseEQ.h"
#include "anv/nodes/MidSide.h"
#include "anv/nodes/MonoToStereo.h"
#include "anv/nodes/MultibandCompressor.h"
#include "anv/nodes/Pan.h"
#include "anv/nodes/PlateReverb.h"
#include "anv/nodes/Polarity.h"
#include "anv/nodes/StereoWidth.h"
#include "anv/nodes/Tape.h"
#include "anv/nodes/TransientDesigner.h"
#include "anv/nodes/Gain.h"
#include "anv/nodes/ParametricEQ.h"
#include "anv/nodes/ReferenceDelay.h"
#include "anv/nodes/TruePeakLimiter.h"
#include "anv/nodes/ZeroPhaseEQ.h"

using namespace anv;

// ---------------------------------------------------------------- harness --

namespace {

int g_failures = 0;
int g_checks = 0;

// The leading newline is not cosmetic. Several tests print progress
// incrementally without a trailing newline, so a failure message would
// otherwise be appended mid-line — where any tool scanning for failures at the
// start of a line silently misses it. That is exactly how three real mutations
// were recorded as "not caught" when they had in fact been detected.
void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::printf("\n  FAIL  %s\n", what.c_str());
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
  // Some nodes have an inherent delay that is not an exact integer — a chorus
  // reads its line at a continuously moving fractional position, so its
  // response lands within a sample or two of the base delay rather than
  // exactly on it. Zero (exact) for everything else.
  int64_t latencyToleranceFrames = 0;
  // ONE fact about the node, not one flag per failing test: dither adds
  // signal-independent broadband noise, and that single property is why two
  // different contract tests do not apply to it.
  //
  //   - silence in / silence out: violated by definition; adding noise to
  //     digital silence IS the mechanism.
  //   - the latency energy centroid: it assumes every joule in the buffer
  //     belongs to the impulse response. Against a 0.01 probe, a 16-bit
  //     noise floor carries ~1% of the total energy spread uniformly over
  //     4096 frames, which drags the centroid ~10 frames toward the middle.
  //     The measured 1009.5 is arithmetically correct and says nothing at
  //     all about latency.
  //
  // The peak probe still runs and is still strict: the impulse sits 80 dB
  // above the noise, so a real latency error would still be caught.
  //
  // Exempting explicitly, per node and with the reason stated, keeps both
  // rules strict for the other twenty nodes rather than quietly loosening a
  // contract everyone else must still meet.
  bool addsBroadbandNoise = false;
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
    // Bands deliberately kept above ~3 kHz. The reverse pass's reach is
    // measured from the cascade's own decay, and a low, high-Q band pushes it
    // past the 1000-frame position the latency probe puts its impulse at —
    // which would clip the response's pre-ring against the start of the buffer
    // and fail an energy-centroid check for a reason that has nothing to do
    // with latency. Low-frequency reach gets its own check in test 23.
    ParamSet p;
    ParamSet b1; b1.setStr("filter_type", "bell").set("center_freq_hz", 3200.0)
                   .set("gain_db", 2.0).set("q_factor", 0.8);
    ParamSet b2; b2.setStr("filter_type", "bell").set("center_freq_hz", 6500.0)
                   .set("gain_db", -3.0).set("q_factor", 2.0);
    ParamSet b3; b3.setStr("filter_type", "highshelf").set("center_freq_hz", 9000.0)
                   .set("gain_db", 1.5).set("q_factor", 0.7);
    p.addItem(b1).addItem(b2).addItem(b3);
    v.push_back({"zero_phase_eq", [] { return std::make_unique<ZeroPhaseEQ>(); }, p});
  }
  {
    // 511 taps, so the declared 255 samples of latency fits comfortably before
    // the latency probe's impulse and the symmetric response is not truncated
    // by either end of the buffer.
    ParamSet p;
    p.set("taps", 511.0);
    ParamSet b1; b1.setStr("filter_type", "bell").set("center_freq_hz", 3200.0)
                   .set("gain_db", 2.0).set("q_factor", 0.8);
    ParamSet b2; b2.setStr("filter_type", "highshelf").set("center_freq_hz", 9000.0)
                   .set("gain_db", 1.5).set("q_factor", 0.7);
    p.addItem(b1).addItem(b2);
    v.push_back({"linear_phase_eq", [] { return std::make_unique<LinearPhaseEQ>(); }, p});
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
    // Wow and flutter deliberately NON-ZERO here, even though they default to
    // zero, because the registry is the one place the modulated path has to be
    // exercised by every contract test. The tolerance below is why the node
    // defaults them off: at 0.05% peak speed deviation and a 1 Hz drift rate
    // the read position wanders +/-9 samples about the declared latency, which
    // is a real property of a tape transport and not something a latency probe
    // can pin to the sample.
    //
    //   wow      H = 96000/(2*1.0) = 48000 knots, A = 0.0005*48000/3.5 = 6.86
    //   flutter  H = 96000/(2*12)  =  4000 knots, A = 0.0003*4000/3.5  = 0.34
    //   swing    1.25 * (6.86 + 0.34) = 9.00, base delay 9 + 2 = 11
    //
    // 7.5 ips so the gap-loss corner (14.1 kHz) lands inside the band and the
    // filter is doing something the frequency tests can see.
    ParamSet p;
    p.set("drive_db", 6.0).set("output_trim_db", -3.0).set("bias", 0.12)
        .set("head_bump_db", 3.0).set("head_bump_hz", 55.0)
        .set("speed_ips", 7.5).set("wow_percent", 0.05).set("wow_rate_hz", 1.0)
        .set("flutter_percent", 0.03).set("flutter_rate_hz", 12.0)
        .set("oversample_factor", 8.0);
    v.push_back({"tape", [] { return std::make_unique<Tape>(); }, p,
                 LatencyProbe::Peak, 0, 10});
  }
  {
    // Headroom deliberately at 12 dB rather than the 24 dB default, so the
    // nonlinearity is large enough for the shared frequency and robustness
    // probes to see it. A console stage at its own defaults is almost the
    // identity, which would make every test here pass vacuously.
    ParamSet p;
    p.set("drive_db", 2.0).set("output_trim_db", -1.0).set("headroom_db", 12.0)
        .set("coupling_hz", 20.0).set("bandwidth_hz", 30000.0)
        .set("oversample_factor", 4.0);
    v.push_back({"console", [] { return std::make_unique<Console>(); }, p});
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
    // Crossovers deliberately HIGH (800 Hz / 5 kHz). The crossover FIR's group
    // delay grows as 1/f, and the latency probe puts its impulse at frame 1000
    // — a 120 Hz split declares 3400 frames of crossover latency, whose
    // pre-ring would be clipped against the start of the buffer and fail an
    // energy-centroid check for a reason that has nothing to do with latency.
    // The low crossover gets its own check in test 25.
    //
    // Makeup gains differ per band on purpose: at equal makeup the impulse
    // response collapses to a pure delay and the latency test would pass
    // without ever exercising the crossover's reconstruction.
    ParamSet p;
    p.set("lookahead_ms", 3.0).set("stereo_link", 0.8)
        .set("crossover_stopband_db", 80.0).set("oversample_factor", 4.0);
    ParamSet b1; b1.set("crossover_hz", 800.0).set("threshold_dbfs", -24.0)
                   .set("ratio_to_1", 3.0).set("knee_db", 6.0)
                   .set("attack_ms", 30.0).set("release_ms", 250.0)
                   .set("makeup_gain_db", 2.0);
    ParamSet b2; b2.set("crossover_hz", 5000.0).set("threshold_dbfs", -20.0)
                   .set("ratio_to_1", 2.5).set("knee_db", 4.0)
                   .set("attack_ms", 8.0).set("release_ms", 120.0)
                   .set("makeup_gain_db", 0.0);
    ParamSet b3; b3.set("crossover_hz", 12000.0).set("threshold_dbfs", -18.0)
                   .set("ratio_to_1", 2.0).set("knee_db", 3.0)
                   .set("attack_ms", 2.0).set("release_ms", 80.0)
                   .set("makeup_gain_db", -1.5);
    p.addItem(b1).addItem(b2).addItem(b3);
    v.push_back({"multiband_compressor",
                 [] { return std::make_unique<MultibandCompressor>(); }, p});
  }
  {
    // Both controls non-zero and of opposite sign, so the registry exercises a
    // boost path and a cut path at once. A node tested at zero would satisfy
    // every contract test while doing nothing at all.
    ParamSet p;
    p.set("attack_db", 6.0).set("sustain_db", -4.0).set("attack_ms", 20.0)
        .set("sustain_ms", 250.0).set("lookahead_ms", 2.0).set("stereo_link", 1.0)
        .set("oversample_factor", 8.0);
    v.push_back({"transient_designer",
                 [] { return std::make_unique<TransientDesigner>(); }, p});
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
    ParamSet p;
    p.set("bit_depth", 16.0).set("amount_lsb", 2.0);
    p.setStr("noise_shaping", "none");
    v.push_back({"dither", [] { return std::make_unique<Dither>(); }, p,
                 LatencyProbe::Peak, 0, 0, /*addsBroadbandNoise=*/true});
  }
  {
    // Short decay so the declared tail keeps the contract renders small.
    ParamSet p;
    p.set("decay_s", 0.4).set("pre_delay_ms", 0.0).set("damping_hz", 6000.0)
        .set("bandwidth_hz", 12000.0).set("size", 1.0)
        .set("modulation_depth", 0.6);
    // The first sound out of the tank arrives via the shortest output tap,
    // Dattorro's 187 samples scaled to the working rate = 603. That is the
    // plate's own structure, not latency: it is how long the sheet takes to
    // respond, and compensating it away would be wrong.
    v.push_back({"plate_reverb", [] { return std::make_unique<PlateReverb>(); }, p,
                 LatencyProbe::Onset, 603, 2});
  }
  {
    // One voice with a deterministic start phase and a very slow, shallow
    // sweep, so the base delay dominates and the latency probe can pin the
    // response position. Multi-voice behaviour is exercised in test 20.
    ParamSet p;
    p.set("voices", 1.0).set("rate_hz", 0.1).set("depth_ms", 0.5)
        .set("base_delay_ms", 12.0).set("randomize_phase", 0.0);
    v.push_back({"chorus", [] { return std::make_unique<Chorus>(); }, p,
                 LatencyProbe::Peak, 1152, 3});
  }
  {
    // Crossover active, so the node exercises its FIR path and declares a
    // non-zero latency for the PDC tests to compensate.
    ParamSet p;
    p.set("width_percent", 130.0).set("bass_mono_below_hz", 120.0)
        .set("bass_width_percent", 0.0).set("crossover_stopband_db", 80.0);
    v.push_back({"stereo_width", [] { return std::make_unique<StereoWidth>(); }, p});
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

    // The inherent-delay offset applies to BOTH probe kinds: a chorus has a
    // base delay just as a reverb has a pre-delay, and neither is latency.
    const int64_t expectedPos = impulseAt + ns.onsetOffsetFrames;
    const int64_t tol = ns.latencyToleranceFrames;

    if (ns.latencyProbe == LatencyProbe::Peak) {
      int64_t peakAt = -1;
      double peakVal = 0.0;
      for (int64_t i = 0; i < out.frames(); ++i) {
        const double v = std::abs(out.channel(0)[i]);
        if (v > peakVal) { peakVal = v; peakAt = i; }
      }
      check(std::llabs(peakAt - expectedPos) <= tol,
            "latency: " + ns.name + " impulse peak at " + std::to_string(peakAt) +
                ", expected " + std::to_string(expectedPos));

      // Energy centroid as a second, independent estimator. It catches an
      // asymmetric response whose peak happens to land in the right place.
      // Meaningless for a node that adds its own noise floor — see the
      // NodeSpec field for the arithmetic.
      if (ns.addsBroadbandNoise) continue;
      double num = 0.0, den = 0.0;
      for (int64_t i = 0; i < out.frames(); ++i) {
        const double e = out.channel(0)[i] * out.channel(0)[i];
        num += e * static_cast<double>(i);
        den += e;
      }
      const double centroid = den > 0.0 ? num / den : -1.0;
      check(std::abs(centroid - static_cast<double>(expectedPos)) <
                2.0 + static_cast<double>(tol),
            "latency: " + ns.name + " energy centroid at " +
                std::to_string(centroid) + ", expected ~" + std::to_string(expectedPos));
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

      int64_t onset = -1;
      for (int64_t i = 0; i < out.frames(); ++i)
        if (std::abs(out.channel(0)[i]) > eps) { onset = i; break; }
      const int64_t expectedOnset = expectedPos;
      check(std::llabs(onset - expectedOnset) <= tol,
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
    if (ns.addsBroadbandNoise) continue;
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
//
//     IT MUST BE OVER-SPECIFIED ON BOTH AXES, NOT ONE. This reference used to
//     be design(8, 100.0, 0.010): 30x the limiter's filter length, but the
//     SAME 8x peak-search grid the limiter then used. A grid is bounded by
//     cos(pi/(2*factor)) — 0.168 dB at 8x — so the reference shared the
//     limiter's blind spot exactly, which is the failure mode the paragraph
//     above is about, one axis over. 32x drops the reference's own grid bound
//     to 0.010 dB, an order of magnitude under the 0.042 dB the node it
//     measures is allowed.
double tightTruePeakDb(const AudioBuffer& buf) {
  TruePeakUpsampler up;
  up.design(32, 100.0, 0.010);
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

    // AND VALIDATE IT WHERE ITS OWN GRID IS EXPOSED, which the check above
    // cannot do. 7333 Hz does not divide the sample rate, so over 96000
    // samples its crest drifts across the grid until some cycle lands on a
    // sample — every oversampling factor recovers the amplitude and the check
    // says nothing about the factor. That is exactly how a reference sharing
    // the limiter's 8x grid went unnoticed.
    //
    // At fs/4 the alignment repeats every cycle instead, so whatever the first
    // cycle misses every later cycle misses identically and the grid has
    // nowhere to hide.
    //
    // THE STARTING PHASE IS SWEPT RATHER THAN CHOSEN. The worst alignment sits
    // at phase pi/(4*factor), which is a different phase for every factor — a
    // single hand-picked value is adversarial for one grid and lands near a
    // crest for the others, and a version of this check that fixed the phase
    // at pi/128 passed with an 8x reference. Sweeping finds each grid's own
    // worst case without the test having to know the factor.
    //
    // The worst reading is then cos(pi/(4*factor)): -0.042 dB at 8x, -0.010 at
    // 16x, -0.003 at 32x. A 0.006 dB tolerance admits only the last.
    {
      const int64_t n = 2048;
      const int64_t fade = 512;
      double worst = 0.0;
      for (int step = 0; step < 192; ++step) {
        const double phase = 0.5 * M_PI * static_cast<double>(step) / 192.0;
        AudioBuffer b(1, n);
        for (int64_t i = 0; i < n; ++i) {
          double env = 1.0;
          if (i < fade)
            env = 0.5 * (1.0 - std::cos(M_PI * static_cast<double>(i) /
                                        static_cast<double>(fade)));
          else if (i >= n - fade)
            env = 0.5 * (1.0 - std::cos(M_PI * static_cast<double>(n - 1 - i) /
                                        static_cast<double>(fade)));
          b.channel(0)[i] = env * std::sin(M_PI * static_cast<double>(i) / 2.0 + phase);
        }
        worst = std::min(worst, tightTruePeakDb(b));
      }
      std::printf("       reference meter on a worst-aligned fs/4 unit sine: "
                  "%+.4f dBTP (a 16x grid reads -0.0105, an 8x grid -0.0419)\n",
                  worst);
      checkNear(worst, 0.0, 0.006,
                "the reference true-peak meter's own peak-search grid is fine "
                "enough to be a reference for a 16x node");
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

  // --- (e) the detector's grid bound is real, and is what it claims --------
  //
  // TruePeakUpsampler searches a grid of `factor` points per input sample, so
  // it returns the grid maximum and never the continuous one: it can only read
  // LOW. gridBoundDb() claims that shortfall is at most cos(pi/(2*factor)),
  // and the limiter subtracts exactly that from its ceiling, so if the claim
  // is wrong the ceiling is wrong by the same amount.
  //
  // Checked against the ANALYTIC answer rather than another estimator. A
  // sampled sinusoid's band-limited reconstruction is that sinusoid, so a unit
  // sine's true peak is exactly 1.0 at every frequency and phase — no meter,
  // no reference, nothing to share a blind spot with.
  //
  // Frequencies stop at 0.44 of the sample rate deliberately. Above the
  // detector's passband edge (0.475 fs) a second and much larger error takes
  // over, the transition band, and it is neither signed nor bounded; mixing
  // the two would make this check measure something gridBoundDb() never
  // claimed. That residual is documented in TruePeak.h and open-questions.md,
  // not asserted here.
  {
    const int64_t n = 8192;
    for (int factor : {16, 32}) {
      TruePeakUpsampler up;
      up.design(factor, 80.0, 0.05);
      const double bound = up.gridBoundDb();
      checkNear(bound, -20.0 * std::log10(std::cos(M_PI / (2.0 * factor))), 1e-12,
                "gridBoundDb() at " + std::to_string(factor) + "x is the closed form");

      // A faded unit sine at frequency f, read as dB against its analytic
      // true peak of exactly 0 dBTP.
      auto reading = [&](double f, double phase) {
        up.prepare(1);
        up.reset();
        const int64_t fade = 2048;
        double best = 0.0;
        for (int64_t i = 0; i < n; ++i) {
          double env = 1.0;
          if (i < fade)
            env = 0.5 * (1.0 - std::cos(M_PI * static_cast<double>(i) /
                                        static_cast<double>(fade)));
          else if (i >= n - fade)
            env = 0.5 * (1.0 - std::cos(M_PI * static_cast<double>(n - 1 - i) /
                                        static_cast<double>(fade)));
          best = std::max(best, up.processMaxAbs(
              env * std::sin(2.0 * M_PI * f * static_cast<double>(i) /
                             kCanonicalSampleRate + phase), 0));
        }
        return linToDb(best);
      };

      // THE MECHANISM, EXACTLY. At fs/4 the oversampled grid lands on the same
      // phases every cycle, so unlike an incommensurate frequency there is no
      // phase diversity to hide behind — which is precisely why the claim this
      // bound replaces survived being measured on stationary tones. Choosing
      // the starting phase pi/(4*factor) puts every crest exactly half a grid
      // step from the nearest sample, the worst alignment there is, and the
      // reading is then cos(pi/(4*factor)) with nothing approximate about it.
      const double atQuarter = reading(0.25 * kCanonicalSampleRate,
                                       M_PI / (4.0 * factor));
      const double predicted =
          20.0 * std::log10(std::cos(M_PI / (4.0 * factor)));
      checkNear(atQuarter, predicted, 1e-4,
                "the " + std::to_string(factor) + "x grid loses exactly "
                "cos(pi/(4*factor)) on a worst-aligned fs/4 sine");

      // AND THE BOUND HOLDS EVERYWHERE IN THE PASSBAND. gridBoundDb() is the
      // Nyquist case, which is the largest of these; nothing below it may
      // exceed it.
      double worstDb = 0.0;
      for (double fRatio : {0.25, 0.33, 0.4, 0.44})
        for (int phaseStep = 0; phaseStep < 12; ++phaseStep)
          worstDb = std::min(worstDb,
                             reading(fRatio * kCanonicalSampleRate,
                                     M_PI * static_cast<double>(phaseStep) / 12.0));
      worstDb = std::min(worstDb, atQuarter);

      std::printf("       true-peak detector at %2dx: fs/4 worst alignment "
                  "%+.4f dB (predicted %+.4f), sweep worst %+.4f, bound %.4f\n",
                  factor, atQuarter, predicted, worstDb, bound);
      check(worstDb >= -bound,
            "the " + std::to_string(factor) + "x detector never under-reads a "
            "band-limited sinusoid by more than its own bound (worst " +
            std::to_string(worstDb) + " dB, bound " + std::to_string(bound) + ")");
    }
  }

  // --- (f) the delivered ceiling is the REQUESTED ceiling ------------------
  //
  // Every other ceiling check in this file measures the output with an
  // estimator. This one does not, and that is the entire point: the defect it
  // exists to catch was the limiter landing exactly on its ceiling as measured
  // by an instrument that read low, which no amount of measuring with that
  // same instrument can see.
  //
  // A unit sinusoid's true peak is EXACTLY its amplitude. Feed one at 0 dBFS,
  // and the output is that sinusoid times the limiter's gain, so the output's
  // true peak is exactly the largest steady-state gain applied — a number that
  // can be read straight off the samples with no reconstruction at all.
  //
  // fs/4 is the frequency chosen because it is where a peak search is most
  // exposed: the grid alignment repeats exactly every cycle, so unlike an
  // incommensurate frequency there is no phase diversity to rescue it, and
  // whatever the first cycle misses every later cycle misses identically.
  {
    const int64_t n = 96000;
    const int64_t settle = 24000;
    for (double ceiling : {-1.0, -0.3}) {
      double worstDeliveredDb = -1e9;
      for (int phaseStep = 0; phaseStep < 8; ++phaseStep) {
        const double phase = M_PI * static_cast<double>(phaseStep) / 8.0;
        AudioBuffer in(1, n);
        for (int64_t i = 0; i < n; ++i)
          in.channel(0)[i] = std::sin(M_PI * static_cast<double>(i) / 2.0 + phase);

        const AudioBuffer out = runLimiter(limiterParams(ceiling, 2.0, 100.0), in);

        // Gain, read where the input is far from a zero crossing so the
        // division is well conditioned. The input's true peak is exactly 1.0,
        // so this gain IS the output's true peak.
        double gain = 0.0;
        for (int64_t i = settle; i < n - settle; ++i)
          if (std::abs(in.channel(0)[i]) > 0.5)
            gain = std::max(gain, std::abs(out.channel(0)[i] / in.channel(0)[i]));
        worstDeliveredDb = std::max(worstDeliveredDb, linToDb(gain));
      }
      std::printf("       limiter delivered true peak at a %.1f dBTP ceiling: "
                  "%+.4f dBTP (analytic, fs/4 unit sine)\n",
                  ceiling, worstDeliveredDb);
      check(worstDeliveredDb <= ceiling,
            "a " + std::to_string(ceiling) + " dBTP ceiling is not exceeded on "
            "a signal whose true peak is known analytically (delivered " +
            std::to_string(worstDeliveredDb) + " dBTP)");
      // Conservative, but not so conservative that the node is just an
      // attenuator. The whole margin is the detector's grid bound at 16x.
      check(worstDeliveredDb >= ceiling - 0.15,
            "and it still reaches its ceiling rather than hiding under it "
            "(delivered " + std::to_string(worstDeliveredDb) + " dBTP)");
    }

    // THE SAME THING ON NODE DEFAULTS, with no parameters written at all.
    // Every other limiter check in this file spells out `oversample_factor`,
    // so the default path — the one a recipe that omits it takes — was
    // exercised nowhere. The default and the floor happen to coincide today,
    // which means this check is redundant today and will not be the first time
    // one of them moves.
    {
      const int64_t n = 96000;
      AudioBuffer in(1, n);
      for (int64_t i = 0; i < n; ++i)
        in.channel(0)[i] = std::sin(M_PI * static_cast<double>(i) / 2.0 +
                                    M_PI / 64.0);
      ParamSet defaults; // nothing set: ceiling defaults to -1 dBTP
      const AudioBuffer out = runLimiter(defaults, in);
      double gain = 0.0;
      for (int64_t i = 24000; i < n - 24000; ++i)
        if (std::abs(in.channel(0)[i]) > 0.5)
          gain = std::max(gain, std::abs(out.channel(0)[i] / in.channel(0)[i]));
      std::printf("       limiter on node defaults: delivered %+.4f dBTP\n",
                  linToDb(gain));
      check(linToDb(gain) <= -1.0,
            "the limiter honours its default -1 dBTP ceiling with no "
            "parameters written (delivered " + std::to_string(linToDb(gain)) +
            " dBTP)");
      check(linToDb(gain) >= -1.15,
            "and the default oversampling is fine enough not to over-limit "
            "(delivered " + std::to_string(linToDb(gain)) + " dBTP)");
    }
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

// 19. Stereo width correctness.

ParamSet widthParams(double widthPct, double bassMonoHz, double bassWidthPct) {
  ParamSet p;
  p.set("width_percent", widthPct).set("bass_mono_below_hz", bassMonoHz)
      .set("bass_width_percent", bassWidthPct)
      .set("crossover_stopband_db", 80.0);
  return p;
}

AudioBuffer runWidth(const ParamSet& p, const AudioBuffer& in) {
  Chain chain;
  Chain::SlotConfig cfg;
  cfg.nodeId = "sw";
  chain.add(std::unique_ptr<Node>(new StereoWidth()), p, cfg);
  chain.prepare(makeSpec(in.channels(), 1024, in.frames()));
  AudioBuffer out;
  chain.render(in, out, 1024);
  return out;
}

void testStereoWidth() {
  const int64_t n = 65536;
  AudioBuffer in(2, n);
  CounterRng rng(808);
  for (int64_t i = 0; i < n; ++i) {
    in.channel(0)[i] = 0.4 * rng.bipolar(0, i);
    in.channel(1)[i] = 0.4 * rng.bipolar(1, i); // genuinely decorrelated
  }

  // --- (a) the crossover reconstructs perfectly ---------------------------
  //
  // The claim that justifies a linear-phase FIR over a cheap IIR crossover:
  // with every band at unity width the node must null against its own input.
  // A Linkwitz-Riley split sums to an allpass and cannot do this — it would
  // leave a phase signature on a mix that asked for nothing to be changed.
  {
    const AudioBuffer out = runWidth(widthParams(100.0, 120.0, 100.0), in);
    const double diff = maxAbsDiff(in, out, n);
    const double nullDb = 20.0 * std::log10(std::max(diff, 1e-300));
    std::printf("       stereo width: unity through the crossover nulls at "
                "%.1f dB\n", nullDb);
    check(nullDb < -180.0,
          "complementary crossover reconstructs exactly (null " +
              std::to_string(nullDb) + " dB)");
  }

  // --- (b) width 0 collapses to mono --------------------------------------
  {
    const AudioBuffer out = runWidth(widthParams(0.0, 0.0, 0.0), in);
    double worst = 0.0;
    for (int64_t i = 0; i < n; ++i)
      worst = std::max(worst, std::abs(out.channel(0)[i] - out.channel(1)[i]));
    check(worst == 0.0, "width 0 gives identical channels (exact mono)");
  }

  // --- (c) width scales the side signal linearly --------------------------
  {
    auto sideEnergy = [&](double widthPct) {
      const AudioBuffer out = runWidth(widthParams(widthPct, 0.0, 0.0), in);
      double e = 0.0;
      for (int64_t i = 0; i < n; ++i) {
        const double s = 0.5 * (out.channel(0)[i] - out.channel(1)[i]);
        e += s * s;
      }
      return e;
    };
    const double e100 = sideEnergy(100.0);
    const double e200 = sideEnergy(200.0);
    const double e50 = sideEnergy(50.0);
    checkNear(e200 / e100, 4.0, 0.01, "width 200% doubles side amplitude");
    checkNear(e50 / e100, 0.25, 0.01, "width 50% halves side amplitude");
  }

  // --- (d) bass mono removes low side content and only low side content ---
  //
  // The reason the crossover exists. Low-frequency side content costs
  // headroom, cuts badly, and partially cancels on single-speaker playback.
  {
    constexpr size_t kN = 32768;
    constexpr int64_t kSettle = 32768;
    const int64_t total = kSettle + static_cast<int64_t>(kN);

    auto binOf = [](double hz) {
      return static_cast<size_t>(std::round(hz * static_cast<double>(kN) /
                                            kCanonicalSampleRate));
    };
    const size_t lowBin = binOf(50.0), highBin = binOf(2000.0);
    const double lowHz =
        static_cast<double>(lowBin) * kCanonicalSampleRate / static_cast<double>(kN);
    const double highHz =
        static_cast<double>(highBin) * kCanonicalSampleRate / static_cast<double>(kN);

    // Side-only content at both frequencies: equal and opposite in L and R.
    AudioBuffer tones(2, total);
    for (int64_t i = 0; i < total; ++i) {
      const double t = static_cast<double>(i) / kCanonicalSampleRate;
      const double lo = 0.3 * std::sin(2.0 * M_PI * lowHz * t);
      const double hi = 0.3 * std::sin(2.0 * M_PI * highHz * t);
      tones.channel(0)[i] = lo + hi;
      tones.channel(1)[i] = -lo - hi;
    }

    auto sideBinDb = [&](const ParamSet& p, size_t bin) {
      const AudioBuffer out = runWidth(p, tones);
      std::vector<double> side(kN);
      for (size_t i = 0; i < kN; ++i)
        side[i] = 0.5 * (out.channel(0)[kSettle + static_cast<int64_t>(i)] -
                         out.channel(1)[kSettle + static_cast<int64_t>(i)]);
      const auto pw = powerSpectrum(side.data(), kN);
      return 10.0 * std::log10(std::max(pw[bin], 1e-300));
    };

    const ParamSet passthrough = widthParams(100.0, 120.0, 100.0);
    const ParamSet bassMono = widthParams(100.0, 120.0, 0.0);

    const double lowRef = sideBinDb(passthrough, lowBin);
    const double lowMono = sideBinDb(bassMono, lowBin);
    const double highRef = sideBinDb(passthrough, highBin);
    const double highMono = sideBinDb(bassMono, highBin);

    std::printf("       bass mono: 50 Hz side %+.1f -> %+.1f dB, "
                "2 kHz side %+.1f -> %+.1f dB\n",
                lowRef, lowMono, highRef, highMono);

    check(lowMono < lowRef - 50.0,
          "bass mono removes low-frequency side content (" +
              std::to_string(lowMono - lowRef) + " dB)");
    checkNear(highMono, highRef, 0.1,
              "and leaves side content above the crossover untouched");
  }

  // --- (e) with no crossover the node is latency-free ---------------------
  {
    Chain chain;
    Chain::SlotConfig cfg;
    cfg.nodeId = "sw0";
    chain.add(std::unique_ptr<Node>(new StereoWidth()),
              widthParams(150.0, 0.0, 0.0), cfg);
    chain.prepare(makeSpec(2, 1024, 4096));
    check(chain.totalLatencyFrames() == 0,
          "with the crossover off the node declares no latency");
  }
}

// 20. Chorus correctness.

ParamSet chorusParams(int voices, double rateHz, double depthMs,
                      double baseDelayMs, double stereoPhaseDeg) {
  ParamSet p;
  p.set("voices", voices).set("rate_hz", rateHz).set("depth_ms", depthMs)
      .set("base_delay_ms", baseDelayMs).set("stereo_phase_deg", stereoPhaseDeg)
      .set("randomize_phase", 0.0);
  return p;
}

AudioBuffer runChorus(const ParamSet& p, const AudioBuffer& in) {
  Chain chain;
  Chain::SlotConfig cfg;
  cfg.nodeId = "ch";
  chain.add(std::unique_ptr<Node>(new Chorus()), p, cfg);
  chain.prepare(makeSpec(in.channels(), 1024, in.frames()));
  AudioBuffer out;
  chain.render(in, out, 1024);
  return out;
}

void testChorus() {
  // --- (a) zero depth is a bit-exact delay --------------------------------
  //
  // Lagrange coefficients at a zero fraction collapse to (0, 1, 0, 0), so an
  // unmodulated chorus at a whole-sample base delay must reproduce its input
  // exactly. That pins the interpolator's alignment: an off-by-one in the tap
  // indexing would show up here and nowhere else.
  {
    const int64_t n = 8192;
    const AudioBuffer in = testSignal(1, n);
    const double baseMs = 1000.0 * 960.0 / kCanonicalSampleRate; // exactly 960
    const AudioBuffer out = runChorus(chorusParams(1, 1.0, 0.0, baseMs, 0.0), in);

    double worst = 0.0;
    for (int64_t i = 960; i < n; ++i)
      worst = std::max(worst, std::abs(out.channel(0)[i] - in.channel(0)[i - 960]));
    check(worst == 0.0,
          "an unmodulated chorus is a bit-exact delay (worst " +
              std::to_string(worst) + ")");
  }

  // --- (b) modulation actually modulates ----------------------------------
  //
  // A swept delay is frequency modulation: the output of a pure tone spreads
  // into sidebands. If the LFO were stuck the spectrum would stay a single
  // line, and the node would be an expensive delay.
  {
    constexpr size_t kN = 32768;
    constexpr int64_t kSettle = 32768;
    const int64_t total = kSettle + static_cast<int64_t>(kN);
    const size_t carrierBin = 400; // ~1172 Hz on the coherent grid
    const double carrierHz =
        static_cast<double>(carrierBin) * kCanonicalSampleRate / static_cast<double>(kN);

    AudioBuffer in(1, total);
    for (int64_t i = 0; i < total; ++i)
      in.channel(0)[i] = 0.5 * std::sin(2.0 * M_PI * carrierHz *
                                        static_cast<double>(i) / kCanonicalSampleRate);

    auto carrierFraction = [&](double depthMs) {
      const AudioBuffer out =
          runChorus(chorusParams(1, 20.0, depthMs, 12.0, 0.0), in);
      const auto pw = powerSpectrum(out.channel(0) + kSettle, kN);
      double total_ = 0.0;
      for (size_t b = 1; b < pw.size(); ++b) total_ += pw[b];
      return total_ > 0.0 ? pw[carrierBin] / total_ : 0.0;
    };

    const double still = carrierFraction(0.0);
    const double swept = carrierFraction(2.0);
    std::printf("       chorus modulation: carrier holds %.3f of the energy "
                "unmodulated, %.3f when swept\n", still, swept);

    check(still > 0.99, "an unmodulated chorus leaves the tone as a single line");
    check(swept < 0.5,
          "sweeping the delay spreads the tone into sidebands (carrier now " +
              std::to_string(swept) + ")");
  }

  // --- (c) opposed stereo phase decorrelates ------------------------------
  //
  // Width here comes from the two channels being modulated in opposition, not
  // from any Haas-style offset — so it survives a mono sum instead of
  // partially cancelling.
  {
    const int64_t n = 65536;
    AudioBuffer in(2, n);
    CounterRng rng(1234);
    for (int64_t i = 0; i < n; ++i) {
      const double v = 0.4 * rng.bipolar(0, i);
      in.channel(0)[i] = v; // identical: a centred mono source
      in.channel(1)[i] = v;
    }

    auto correlation = [&](double phaseDeg) {
      const AudioBuffer out =
          runChorus(chorusParams(3, 0.8, 4.0, 14.0, phaseDeg), in);
      double sLL = 0.0, sRR = 0.0, sLR = 0.0;
      for (int64_t i = 0; i < out.frames(); ++i) {
        const double l = out.channel(0)[i], r = out.channel(1)[i];
        sLL += l * l; sRR += r * r; sLR += l * r;
      }
      return sLR / std::sqrt(std::max(sLL * sRR, 1e-300));
    };

    const double locked = correlation(0.0);
    const double opposed = correlation(180.0);
    std::printf("       chorus stereo: phase 0 -> correlation %.3f, "
                "phase 180 -> %.3f\n", locked, opposed);

    checkNear(locked, 1.0, 1e-9,
              "with no stereo phase offset both channels are identical");
    check(opposed < 0.9,
          "opposed modulation decorrelates the channels (" +
              std::to_string(opposed) + ")");
  }

  // --- (d) more voices, denser result -------------------------------------
  //
  // Voices are spread evenly around the LFO cycle, so adding them should
  // smooth the summed modulation rather than just making it louder.
  {
    const int64_t n = 32768;
    AudioBuffer in(1, n);
    for (int64_t i = 0; i < n; ++i)
      in.channel(0)[i] = 0.5 * std::sin(2.0 * M_PI * 1000.0 *
                                        static_cast<double>(i) / kCanonicalSampleRate);

    auto peakOf = [&](int voices) {
      const AudioBuffer out =
          runChorus(chorusParams(voices, 1.0, 3.0, 14.0, 0.0), in);
      double pk = 0.0;
      for (int64_t i = n / 2; i < n; ++i)
        pk = std::max(pk, std::abs(out.channel(0)[i]));
      return pk;
    };

    // Normalised by voice count, so more voices must not simply mean louder.
    const double one = peakOf(1);
    const double four = peakOf(4);
    check(four < one * 1.2,
          "voice summing is normalised, not additive (1 voice " +
              std::to_string(one) + ", 4 voices " + std::to_string(four) + ")");
    check(four > 0.05, "and four voices still produce real output");
  }
}

// 21. Plate reverb correctness.

ParamSet plateParams(double decayS, double dampingHz) {
  ParamSet p;
  p.set("decay_s", decayS).set("pre_delay_ms", 0.0)
      .set("damping_hz", dampingHz).set("bandwidth_hz", 14000.0)
      .set("size", 1.0).set("modulation_depth", 0.6);
  return p;
}

AudioBuffer plateIr(const ParamSet& p, int channels) {
  const int64_t n = 64;
  AudioBuffer in(channels, n);
  for (int c = 0; c < channels; ++c) in.channel(c)[0] = 1.0;
  Chain chain;
  Chain::SlotConfig cfg;
  cfg.nodeId = "plate";
  chain.add(std::unique_ptr<Node>(new PlateReverb()), p, cfg);
  chain.prepare(makeSpec(channels, 1024, n));
  AudioBuffer out;
  chain.render(in, out, 1024);
  return out;
}

double measureT30(const AudioBuffer& ir, int channel) {
  const int64_t n = ir.frames();
  std::vector<double> edc(static_cast<size_t>(n), 0.0);
  double acc = 0.0;
  for (int64_t i = n - 1; i >= 0; --i) {
    acc += ir.channel(channel)[i] * ir.channel(channel)[i];
    edc[static_cast<size_t>(i)] = acc;
  }
  auto crossing = [&](double db) {
    const double target = edc[0] * std::pow(10.0, db / 10.0);
    for (int64_t i = 0; i < n; ++i)
      if (edc[static_cast<size_t>(i)] <= target)
        return static_cast<double>(i) / kCanonicalSampleRate;
    return -1.0;
  };
  const double t5 = crossing(-5.0), t35 = crossing(-35.0);
  return (t5 >= 0.0 && t35 > t5) ? 2.0 * (t35 - t5) : -1.0;
}

void testPlateReverb() {
  // --- (a) the decay time is the one requested ----------------------------
  //
  // The tank applies its decay gain twice per circuit, so the mapping from
  // RT60 to that gain has to account for the full loop length. Getting the
  // exponent wrong gives a reverb that is plausible but never the length it
  // was asked for.
  {
    // Probed with damping wide open and well above the tank's ~0.52 s loop
    // period, so the measurement isolates the decay mapping from both the
    // damping-induced shortening and the topology's own RT60 floor.
    std::printf("       plate decay:");
    for (double decay : {1.0, 2.0, 4.0}) {
      const AudioBuffer ir = plateIr(plateParams(decay, 19000.0), 2);
      const double t30 = measureT30(ir, 0);
      std::printf("  requested %.1fs -> T30 %.2fs", decay, t30);
      check(t30 > 0.0, "plate energy decay curve spans -5 to -35 dB");
      checkNear(t30, decay, decay * 0.15,
                "plate T30 matches the requested decay time");
    }
    std::printf("\n");
  }

  // --- (b) the two outputs are decorrelated -------------------------------
  //
  // A plate is one sheet, so both outputs come from the same tank — the image
  // comes from where the pickups sit. Correlated outputs would collapse the
  // reverb to the centre and waste the whole point of a stereo return.
  {
    const int64_t n = 65536;
    AudioBuffer in(2, n);
    CounterRng rng(606);
    for (int64_t i = 0; i < n; ++i) {
      const double v = 0.4 * rng.bipolar(0, i);
      in.channel(0)[i] = v;
      in.channel(1)[i] = v;
    }
    Chain chain;
    Chain::SlotConfig cfg;
    cfg.nodeId = "plate";
    chain.add(std::unique_ptr<Node>(new PlateReverb()), plateParams(1.5, 8000.0), cfg);
    chain.prepare(makeSpec(2, 1024, n));
    AudioBuffer out;
    chain.render(in, out, 1024);

    double sLL = 0.0, sRR = 0.0, sLR = 0.0;
    for (int64_t i = 0; i < out.frames(); ++i) {
      const double l = out.channel(0)[i], r = out.channel(1)[i];
      sLL += l * l; sRR += r * r; sLR += l * r;
    }
    const double corr = sLR / std::sqrt(std::max(sLL * sRR, 1e-300));
    std::printf("       plate stereo correlation: %.3f\n", corr);
    check(std::abs(corr) < 0.5,
          "the two plate outputs are decorrelated (" + std::to_string(corr) + ")");
  }

  // --- (c) damping shortens the top end -----------------------------------
  {
    const double open = measureT30(plateIr(plateParams(2.0, 18000.0), 2), 0);
    const double damped = measureT30(plateIr(plateParams(2.0, 1500.0), 2), 0);
    std::printf("       plate damping: 18 kHz -> T30 %.2fs, 1.5 kHz -> %.2fs\n",
                open, damped);
    check(damped < open,
          "heavier damping gives a shorter tail (open " + std::to_string(open) +
              "s, damped " + std::to_string(damped) + "s)");
  }

  // --- (d) a long decay stays stable --------------------------------------
  //
  // A recirculating network with too much loop gain does not sound wrong, it
  // diverges. The decay gain is clamped below unity for exactly this reason.
  {
    const int64_t n = static_cast<int64_t>(6.0 * kCanonicalSampleRate);
    AudioBuffer in(2, n);
    for (int64_t i = 0; i < 4800; ++i) {
      in.channel(0)[i] = std::sin(2.0 * M_PI * 220.0 *
                                  static_cast<double>(i) / kCanonicalSampleRate);
      in.channel(1)[i] = in.channel(0)[i];
    }
    Chain chain;
    Chain::SlotConfig cfg;
    cfg.nodeId = "plate";
    chain.add(std::unique_ptr<Node>(new PlateReverb()), plateParams(20.0, 10000.0), cfg);
    chain.prepare(makeSpec(2, 1024, n));
    AudioBuffer out;
    chain.render(in, out, 1024);

    check(allFinite(out), "a 20 second decay stays finite");

    // Energy must be falling, not growing.
    auto windowEnergy = [&](int64_t from, int64_t len) {
      double e = 0.0;
      for (int64_t i = from; i < from + len && i < out.frames(); ++i)
        for (int c = 0; c < 2; ++c) e += out.channel(c)[i] * out.channel(c)[i];
      return e;
    };
    const double early = windowEnergy(static_cast<int64_t>(1.0 * kCanonicalSampleRate), 48000);
    const double late = windowEnergy(static_cast<int64_t>(5.0 * kCanonicalSampleRate), 48000);
    check(late < early,
          "the tank decays rather than diverging (early " + std::to_string(early) +
              ", late " + std::to_string(late) + ")");
  }
}


// ---------------------------------------------------------------------------
// 22. Dither correctness.
//
// The whole justification for dither is a claim about the STATISTICS of the
// quantisation error, so that is what gets measured. "It sounds smoother" is
// not testable; "the error's mean is independent of the input" is.

// One LSB of a signed word of `bits`, in full-scale units.
double lsbStep(int bits) { return 1.0 / std::pow(2.0, static_cast<double>(bits - 1)); }

std::vector<double> ditherRender(const ParamSet& p, const std::vector<double>& sig) {
  const int64_t n = static_cast<int64_t>(sig.size());
  AudioBuffer in(1, n);
  std::copy(sig.begin(), sig.end(), in.channel(0));

  Chain chain;
  Chain::SlotConfig cfg;
  cfg.nodeId = "dither";
  chain.add(std::unique_ptr<Node>(new Dither()), p, cfg);
  chain.prepare(makeSpec(1, 512, n));

  AudioBuffer out;
  chain.render(in, out, 512);
  return std::vector<double>(out.channel(0), out.channel(0) + n);
}

ParamSet ditherParams(double bits, double amountLsb, const char* shaping) {
  ParamSet p;
  p.set("bit_depth", bits).set("amount_lsb", amountLsb);
  p.setStr("noise_shaping", shaping);
  return p;
}

void testDither() {
  const int kBits = 16;
  const double step = lsbStep(kBits);

  // --- (a) the output really is on the quantisation lattice ---------------
  //
  // Everything below is a statement about quantisation error, and all of it
  // would be vacuous if the node were not quantising at all.
  {
    std::vector<double> sig(4096);
    for (size_t i = 0; i < sig.size(); ++i)
      sig[i] = 0.7 * std::sin(2.0 * M_PI * 11.0 * static_cast<double>(i) / 4096.0);

    const auto out = ditherRender(ditherParams(kBits, 2.0, "none"), sig);
    double worstOffLattice = 0.0;
    for (double v : out) {
      const double k = v / step;
      worstOffLattice = std::max(worstOffLattice, std::abs(k - std::round(k)));
    }
    check(worstOffLattice < 1e-9,
          "dither: output on the 16-bit lattice, worst offset " +
              std::to_string(worstOffLattice) + " LSB");
  }

  // --- (b) THE CENTRAL CLAIM: error mean stops tracking the input ---------
  //
  // Hold the input at a constant level part-way between two codes and look at
  // the average error. Undithered, the quantiser rounds every single sample
  // the same way, so the mean error is exactly minus the fractional offset —
  // the error IS the input, negated. That is not noise; it is a signal-
  // dependent DC term, and on a real signal it becomes harmonic distortion
  // that follows the music down into a fade.
  //
  // With TPDF dither the mean error is zero at every offset. That is the
  // entire point of the node, stated as a number.
  //
  // The variance is the second half of the claim and the specific reason for
  // TRIANGULAR rather than rectangular dither: RPDF removes the mean but
  // leaves the error's variance modulated by the input, which is audible as
  // a noise floor that breathes with the signal. TPDF removes both.
  {
    const int64_t n = 65536;
    const std::vector<double> offsets = {0.0, 0.1, 0.2, 0.3, 0.4,
                                         0.5, 0.6, 0.7, 0.8, 0.9};

    double worstDitheredMean = 0.0, biggestUnditheredMean = 0.0;
    double minVar = 1e300, maxVar = 0.0;

    for (double f : offsets) {
      std::vector<double> sig(static_cast<size_t>(n), f * step);

      const auto dry = ditherRender(ditherParams(kBits, 0.0, "none"), sig);
      const auto wet = ditherRender(ditherParams(kBits, 2.0, "none"), sig);

      double sumDry = 0.0, sumWet = 0.0, sumSqWet = 0.0;
      for (int64_t i = 0; i < n; ++i) {
        sumDry += dry[static_cast<size_t>(i)] - f * step;
        const double e = wet[static_cast<size_t>(i)] - f * step;
        sumWet += e;
        sumSqWet += e * e;
      }
      const double meanDry = sumDry / static_cast<double>(n);
      const double meanWet = sumWet / static_cast<double>(n);
      const double varWet = sumSqWet / static_cast<double>(n) - meanWet * meanWet;

      worstDitheredMean = std::max(worstDitheredMean, std::abs(meanWet) / step);
      biggestUnditheredMean = std::max(biggestUnditheredMean, std::abs(meanDry) / step);
      minVar = std::min(minVar, varWet);
      maxVar = std::max(maxVar, varWet);
    }

    // Sanity: the disease has to be present before the cure means anything.
    check(biggestUnditheredMean > 0.3,
          "dither: undithered error tracks the input, worst mean " +
              std::to_string(biggestUnditheredMean) + " LSB (expect ~0.5)");

    // 65536 samples at variance step^2/4 puts the standard error of the mean
    // at step/512, so 0.02 LSB is roughly ten sigma — tight, and not flaky.
    check(worstDitheredMean < 0.02,
          "dither: dithered error mean independent of input, worst " +
              std::to_string(worstDitheredMean) + " LSB");

    // No noise modulation: same variance at every position within the code.
    const double spread = (maxVar - minVar) / maxVar;
    check(spread < 0.05,
          "dither: error variance independent of input, spread " +
              std::to_string(spread * 100.0) + "%");

    // And that variance is the textbook value. Quantisation contributes
    // step^2/12 and 2-LSB TPDF contributes step^2/6, totalling step^2/4.
    const double expectedVar = step * step / 4.0;
    const double ratioDb = 10.0 * std::log10(maxVar / expectedVar);
    check(std::abs(ratioDb) < 0.2,
          "dither: noise power is step^2/4, off by " + std::to_string(ratioDb) + " dB");
  }

  // --- (c) on a real signal, nothing correlated survives ------------------
  //
  // The DC test proves the mechanism; this proves it matters on a waveform.
  //
  // MEASURED BY PERIOD-FOLDING, not by reading harmonic bins. Quantising a
  // sine only a few LSBs tall throws energy into harmonics of every order,
  // and the ones above Nyquist fold back and land wherever they like — so
  // summing a handful of named bins measures an arbitrary fraction of the
  // distortion and misses the rest. Folding sidesteps that entirely: whatever
  // is correlated with the signal repeats with the signal's period and
  // survives averaging, while anything uncorrelated averages down as
  // 1/sqrt(K). It measures the actual claim rather than a proxy for it.
  {
    const size_t kPeriod = 128;   // 750 Hz at 96 kHz
    const size_t kCycles = 1024;
    const size_t kN = kPeriod * kCycles;
    const double amp = 4.0 * step;

    std::vector<double> sig(kN);
    for (size_t i = 0; i < kN; ++i)
      sig[i] = amp * std::sin(2.0 * M_PI * static_cast<double>(i % kPeriod) /
                              static_cast<double>(kPeriod));

    auto foldedRms = [&](const std::vector<double>& out) {
      std::vector<double> fold(kPeriod, 0.0);
      for (size_t i = 0; i < kN; ++i) fold[i % kPeriod] += out[i] - sig[i];
      double acc = 0.0;
      for (double v : fold) {
        const double mean = v / static_cast<double>(kCycles);
        acc += mean * mean;
      }
      return std::sqrt(acc / static_cast<double>(kPeriod));
    };

    const double dryRms = foldedRms(ditherRender(ditherParams(kBits, 0.0, "none"), sig));
    const double wetRms = foldedRms(ditherRender(ditherParams(kBits, 2.0, "none"), sig));

    const double improvementDb = 20.0 * std::log10(dryRms / std::max(wetRms, 1e-300));
    check(improvementDb > 20.0,
          "dither: signal-correlated error drops by " +
              std::to_string(improvementDb) + " dB (expect >20)");

    // The stronger statement, and the one worth making: what is left is not
    // merely smaller, it is indistinguishable from noise. Averaging K cycles
    // of an uncorrelated floor of RMS step/2 leaves step/(2*sqrt(K)); if the
    // dithered residual sits at that level then there is no correlated
    // component left to find, only the averaging floor itself.
    const double noiseLimited = (step * 0.5) / std::sqrt(static_cast<double>(kCycles));
    check(wetRms < 1.3 * noiseLimited,
          "dither: residual is noise-limited, " +
              std::to_string(wetRms / noiseLimited) + "x the averaging floor");
  }

  // --- (d) noise shaping trades total power for spectral placement --------
  //
  // First-order shaping has error transfer function (1 - z^-1). Its magnitude
  // is 2*sin(w/2): below zero at DC, above unity past a third of Nyquist, and
  // integrating to exactly twice the flat noise power. So shaping must make
  // the TOTAL noise louder while making the LOW-FREQUENCY noise quieter. If a
  // test only checked total power it would read the improvement as a
  // regression, which is why both halves are asserted.
  {
    const size_t kN = 32768;
    const std::vector<double> silence(kN, 0.0);

    const auto flat = ditherRender(ditherParams(kBits, 2.0, "none"), silence);
    const auto shaped = ditherRender(ditherParams(kBits, 2.0, "first_order"), silence);

    auto totalPower = [](const std::vector<double>& v) {
      double acc = 0.0;
      for (double x : v) acc += x * x;
      return acc / static_cast<double>(v.size());
    };

    const double totalDb =
        10.0 * std::log10(totalPower(shaped) / totalPower(flat));
    check(totalDb > 2.0 && totalDb < 4.0,
          "dither: shaping raises total noise by " + std::to_string(totalDb) +
              " dB (expect ~3)");

    // Low band: DC to Nyquist/16, which at 96 kHz is DC to 3 kHz.
    const auto pFlat = powerSpectrum(flat.data(), kN);
    const auto pShaped = powerSpectrum(shaped.data(), kN);
    double loFlat = 0.0, loShaped = 0.0;
    for (size_t b = 1; b < pFlat.size() / 16; ++b) {
      loFlat += pFlat[b];
      loShaped += pShaped[b];
    }
    const double loDb = 10.0 * std::log10(loShaped / loFlat);
    check(loDb < -8.0,
          "dither: shaping lowers sub-3 kHz noise by " + std::to_string(-loDb) +
              " dB (expect >8)");
  }

  // --- (e) float export is left strictly alone ----------------------------
  //
  // A 32-bit float file has no fixed code spacing to decorrelate, so dither
  // there is pure added noise. Bit-exact pass-through, not "close enough":
  // anything else means the export path degrades a format it cannot improve.
  {
    std::vector<double> sig(2048);
    for (size_t i = 0; i < sig.size(); ++i)
      sig[i] = 0.31 * std::sin(0.017 * static_cast<double>(i)) +
               1e-9 * std::sin(0.4 * static_cast<double>(i));

    for (double bits : {0.0, 32.0}) {
      const auto out = ditherRender(ditherParams(bits, 2.0, "none"), sig);
      double worst = 0.0;
      for (size_t i = 0; i < sig.size(); ++i)
        worst = std::max(worst, std::abs(out[i] - sig[i]));
      check(worst == 0.0,
            "dither: bit_depth " + std::to_string(static_cast<int>(bits)) +
                " is bit-exact pass-through, worst diff " + std::to_string(worst));
    }
  }

  // --- (f) the word length is actually respected --------------------------
  //
  // Each bit removed doubles the code spacing, so the noise floor must rise
  // by exactly 6.02 dB per bit. This catches an off-by-one in the exponent,
  // which is otherwise invisible: a node quantising to 15 bits when asked for
  // 16 passes every other test in this section.
  {
    const size_t kN = 32768;
    const std::vector<double> silence(kN, 0.0);

    auto floorDb = [&](double bits) {
      const auto out = ditherRender(ditherParams(bits, 2.0, "none"), silence);
      double acc = 0.0;
      for (double x : out) acc += x * x;
      return 10.0 * std::log10(acc / static_cast<double>(out.size()));
    };

    const double f8 = floorDb(8.0), f16 = floorDb(16.0), f24 = floorDb(24.0);

    check(std::abs((f8 - f16) - 8.0 * 6.0206) < 0.2,
          "dither: 8 vs 16 bit floor differs by " + std::to_string(f8 - f16) +
              " dB (expect 48.2)");
    check(std::abs((f16 - f24) - 8.0 * 6.0206) < 0.2,
          "dither: 16 vs 24 bit floor differs by " + std::to_string(f16 - f24) +
              " dB (expect 48.2)");

    // Absolute anchor, so a consistent scaling error cannot hide behind two
    // ratios that both look right. RMS is step/2 = 2^-16 at 16 bits.
    check(std::abs(f16 - (-96.33)) < 0.2,
          "dither: 16-bit noise floor at " + std::to_string(f16) +
              " dBFS (expect -96.33)");
  }
}

// 23/24. The two offline EQs — shared measurement machinery.
//
// Both nodes make a claim about PHASE, and phase is the one property the
// contract tests cannot see: test 5 measures where the energy is, not what the
// filter did to the relationship between frequencies. So everything below
// works from the node's RAW response — driven through processBlock so the
// chain has not already removed the declared latency — and then asks whether
// the response is symmetric about exactly the position the node declared.
//
// That framing is deliberate. "Zero phase" and "linear phase" are both claims
// about the response relative to the DECLARED latency, so a test that measured
// the compensated output would be checking the chain's arithmetic as much as
// the node's.

struct RawResponse {
  std::vector<double> h;
  int64_t centre = 0; // where the response should be symmetric about
  int latency = 0;
};

RawResponse rawResponse(std::unique_ptr<Node> node, const ParamSet& p,
                        int64_t frames, int64_t impulseAt) {
  Chain chain;
  Chain::SlotConfig cfg;
  cfg.nodeId = "probe";
  chain.add(std::move(node), p, cfg);
  chain.prepare(makeSpec(1, 1024, frames));

  AudioBuffer in(1, frames);
  AudioBuffer out(1, frames);
  in.channel(0)[impulseAt] = 1.0;

  std::vector<const Sample*> bin(1);
  std::vector<Sample*> bout(1);
  for (int64_t pos = 0; pos < frames; pos += 1024) {
    const int nn = static_cast<int>(std::min<int64_t>(1024, frames - pos));
    bin[0] = in.channel(0) + pos;
    bout[0] = out.channel(0) + pos;
    chain.processBlock(bin.data(), bout.data(), nn, pos);
  }

  RawResponse r;
  r.latency = chain.totalLatencyFrames();
  r.centre = impulseAt + r.latency;
  r.h.assign(out.channel(0), out.channel(0) + frames);
  return r;
}

// Largest asymmetry either side of the claimed centre, relative to the peak.
double symmetryError(const RawResponse& r, int64_t reach) {
  double worst = 0.0, peak = 0.0;
  for (int64_t k = 0; k <= reach; ++k) {
    const int64_t lo = r.centre - k, hi = r.centre + k;
    if (lo < 0 || hi >= static_cast<int64_t>(r.h.size())) break;
    peak = std::max(peak, std::abs(r.h[static_cast<size_t>(hi)]));
    worst = std::max(worst, std::abs(r.h[static_cast<size_t>(hi)] -
                                     r.h[static_cast<size_t>(lo)]));
  }
  return peak > 0.0 ? worst / peak : 1.0;
}

struct PhaseReport {
  double worstPhaseDeg = 0.0;
  double worstGroupDelay = 0.0; // samples, relative to the declared latency
  int bins = 0;
};

// Rotate the response so the claimed centre sits at time zero, then transform.
// If the claim is right the spectrum comes back REAL: phase zero at every
// frequency, group delay zero at every frequency. Rotating first is what makes
// the two nodes measurable by the same code — a linear-phase FIR and a
// filtfilt cascade differ only in what their centre is.
PhaseReport phaseAboutCentre(const RawResponse& r, size_t window, double loHz,
                             double hiHz, double floorRel) {
  std::vector<std::complex<double>> seg(window, std::complex<double>(0.0, 0.0));
  const int64_t half = static_cast<int64_t>(window) / 2;
  for (int64_t j = -half + 1; j < half; ++j) {
    const int64_t idx = r.centre + j;
    if (idx < 0 || idx >= static_cast<int64_t>(r.h.size())) continue;
    const int64_t slot = (j + static_cast<int64_t>(window)) %
                         static_cast<int64_t>(window);
    seg[static_cast<size_t>(slot)] = r.h[static_cast<size_t>(idx)];
  }
  fftInPlace(seg);

  double maxMag = 0.0;
  for (size_t k = 0; k <= window / 2; ++k)
    maxMag = std::max(maxMag, std::abs(seg[k]));

  PhaseReport out;
  std::vector<double> phase(window / 2 + 1, 0.0);
  std::vector<bool> usable(window / 2 + 1, false);
  for (size_t k = 0; k <= window / 2; ++k) {
    const double hz = static_cast<double>(k) * kCanonicalSampleRate /
                      static_cast<double>(window);
    if (hz < loHz || hz > hiHz) continue;
    // A bin the filter has emptied carries no phase information; its angle is
    // round-off. Skipping it is not loosening the test — including it would
    // measure the arctangent of noise.
    if (std::abs(seg[k]) < floorRel * maxMag) continue;
    phase[k] = std::atan2(seg[k].imag(), seg[k].real());
    usable[k] = true;
    ++out.bins;
    out.worstPhaseDeg = std::max(out.worstPhaseDeg,
                                 std::abs(phase[k]) * 180.0 / M_PI);
  }

  // Group delay from the slope of that phase. Zero phase everywhere and flat
  // group delay are not the same statement, and a node could in principle
  // satisfy the first at the bins sampled while failing the second between
  // them, so both are measured.
  const double dw = 2.0 * M_PI / static_cast<double>(window);
  for (size_t k = 1; k <= window / 2; ++k) {
    if (!usable[k] || !usable[k - 1]) continue;
    out.worstGroupDelay =
        std::max(out.worstGroupDelay, std::abs(-(phase[k] - phase[k - 1]) / dw));
  }
  return out;
}

// Steady-state gain of a TWO-SIDED filter.
//
// `measureNodeGainDb` from section 18 analyses the LAST window of the render,
// which is correct for every causal node in the registry and wrong for both of
// these. A zero-phase or linear-phase filter reaches backwards as well as
// forwards, so the end of a render is not a steady state — it is the response
// decaying into the silence past the end of the signal. Measured there a +6 dB
// bell at 200 Hz reads +5.98 dB, and the 0.02 dB shortfall is a property of
// where the window was put rather than of the filter.
//
// A two-sided filter has TWO settling regions. This window sits in the middle,
// with settling room either side of it.
double measureTwoSidedGainDb(const std::function<std::unique_ptr<Node>()>& make,
                             const ParamSet& params, double snappedHz) {
  constexpr int64_t kSettle = 65536; // comfortably past the 24000-frame cap
  constexpr int64_t kWindow = 16384;
  const int64_t total = kSettle + kWindow + kSettle;

  AudioBuffer in(1, total);
  for (int64_t i = 0; i < total; ++i)
    in.channel(0)[i] = 0.25 * std::sin(2.0 * M_PI * snappedHz *
                                       static_cast<double>(i) / kCanonicalSampleRate);

  Chain chain;
  Chain::SlotConfig cfg;
  cfg.nodeId = "eq";
  chain.add(make(), params, cfg);
  chain.prepare(makeSpec(1, 1024, total));
  AudioBuffer out;
  chain.render(in, out, 1024);

  double sIn = 0.0, sOut = 0.0;
  for (int64_t i = kSettle; i < kSettle + kWindow; ++i) {
    sIn += in.channel(0)[i] * in.channel(0)[i];
    sOut += out.channel(0)[i] * out.channel(0)[i];
  }
  return 20.0 * std::log10(std::sqrt(sOut) / std::sqrt(sIn));
}

double snapHz(double hz, int64_t window) {
  const double k = std::round(hz * static_cast<double>(window) / kCanonicalSampleRate);
  return k * kCanonicalSampleRate / static_cast<double>(window);
}

ParamSet eqBands(std::initializer_list<ParamSet> bands) {
  ParamSet p;
  for (const ParamSet& b : bands) p.addItem(b);
  return p;
}

ParamSet eqBand(const char* shape, double freq, double gainDb, double q) {
  ParamSet b;
  b.setStr("filter_type", shape).set("center_freq_hz", freq)
      .set("gain_db", gainDb).set("q_factor", q);
  return b;
}

// 23. Zero-phase (forward-backward) EQ correctness.
void testZeroPhaseEq() {
  // A single moderate bell, so every claim below has an analytic expectation
  // rather than a number read off the implementation.
  const ParamSet oneBell = eqBands({eqBand("bell", 1000.0, 6.0, 1.0)});

  // --- (a) the phase response is zero -------------------------------------
  //
  // THE CLAIM THE NODE IS NAMED FOR. Not "small", not "flat" — zero, because
  // filtfilt's impulse response is the autocorrelation of the one-pass
  // response and an autocorrelation is an even function.
  {
    const RawResponse r = rawResponse(std::make_unique<ZeroPhaseEQ>(), oneBell,
                                      65536, 8192);
    const PhaseReport ph = phaseAboutCentre(r, 16384, 20.0, 20000.0, 1e-6);

    std::printf("       zero-phase EQ: declared latency %d, %d bins measured, "
                "worst phase %.3e deg, worst group delay %.3e samples\n",
                r.latency, ph.bins, ph.worstPhaseDeg, ph.worstGroupDelay);

    check(ph.bins > 2000, "zero-phase EQ: the phase probe saw " +
                              std::to_string(ph.bins) + " usable bins");
    check(ph.worstPhaseDeg < 1e-6,
          "zero-phase EQ: phase is zero across 20 Hz-20 kHz (worst " +
              std::to_string(ph.worstPhaseDeg) + " deg)");
    check(ph.worstGroupDelay < 1e-6,
          "zero-phase EQ: group delay is flat at 0 (worst " +
              std::to_string(ph.worstGroupDelay) + " samples)");

    // The same statement in the time domain, which fails differently: a
    // reverse pass that ran forwards would still produce a plausible spectrum
    // but an asymmetric impulse response.
    const double asym = symmetryError(r, 4096);
    std::printf("       zero-phase EQ: impulse response asymmetry about the "
                "declared latency %.3e of peak\n", asym);
    check(asym < 1e-12,
          "zero-phase EQ: the impulse response is symmetric about the declared "
          "latency (worst " + std::to_string(asym) + " of peak)");
  }

  // --- (b) a symmetric input comes back symmetric -------------------------
  //
  // The property an engineer would check by ear: nothing is smeared in one
  // direction. A minimum-phase EQ fails this and is supposed to.
  {
    const int64_t frames = 65536, centreIn = 8192, halfWidth = 400;
    Chain chain;
    Chain::SlotConfig cfg;
    cfg.nodeId = "zp";
    chain.add(std::make_unique<ZeroPhaseEQ>(), oneBell, cfg);
    chain.prepare(makeSpec(1, 1024, frames));

    // A symmetric burst: a raised-cosine-windowed tone at the bell's centre.
    AudioBuffer in(1, frames);
    for (int64_t k = -halfWidth; k <= halfWidth; ++k) {
      const double t = static_cast<double>(k) / static_cast<double>(halfWidth);
      const double env = 0.5 * (1.0 + std::cos(M_PI * t));
      in.channel(0)[centreIn + k] =
          0.3 * env * std::cos(2.0 * M_PI * 1000.0 * static_cast<double>(k) /
                               kCanonicalSampleRate);
    }

    AudioBuffer out;
    chain.render(in, out, 1024);

    double worst = 0.0, peak = 0.0;
    for (int64_t k = 0; k <= 4096; ++k) {
      const double a = out.channel(0)[centreIn + k];
      const double b = out.channel(0)[centreIn - k];
      peak = std::max(peak, std::abs(a));
      worst = std::max(worst, std::abs(a - b));
    }
    check(worst / peak < 1e-12,
          "zero-phase EQ: a symmetric burst stays symmetric (worst " +
              std::to_string(worst / peak) + " of peak)");

    // And the reference the claim is worth something against: the same bands
    // through the minimum-phase node smear the burst badly.
    Chain mp;
    Chain::SlotConfig mpCfg;
    mpCfg.nodeId = "pe";
    mp.add(std::make_unique<ParametricEQ>(), oneBell, mpCfg);
    mp.prepare(makeSpec(1, 1024, frames));
    AudioBuffer mpOut;
    mp.render(in, mpOut, 1024);

    double mpWorst = 0.0, mpPeak = 0.0;
    for (int64_t k = 0; k <= 4096; ++k) {
      const double a = mpOut.channel(0)[centreIn + k];
      const double b = mpOut.channel(0)[centreIn - k];
      mpPeak = std::max(mpPeak, std::abs(a));
      mpWorst = std::max(mpWorst, std::abs(a - b));
    }
    std::printf("       symmetric burst asymmetry: zero-phase %.2e, "
                "minimum-phase %.2e (of peak)\n", worst / peak, mpWorst / mpPeak);
    check(mpWorst / mpPeak > 0.01,
          "and the minimum-phase EQ genuinely does smear it, so the comparison "
          "is not vacuous (" + std::to_string(mpWorst / mpPeak) + ")");
  }

  // --- (c) the magnitude is the SQUARE of the one-pass EQ ------------------
  //
  // |H(e^jw)|^2, exactly: two passes of the same filter. In dB that is a clean
  // doubling, which is the number an engineer needs to know before reaching
  // for this node — a 3 dB band becomes a 6 dB band.
  {
    constexpr int64_t kWindow = 16384;
    auto zp = [] { return std::unique_ptr<Node>(new ZeroPhaseEQ()); };
    auto pe = [] { return std::unique_ptr<Node>(new ParametricEQ()); };

    double worst = 0.0;
    for (double hz : {200.0, 500.0, 1000.0, 2000.0, 5000.0}) {
      const double f = snapHz(hz, kWindow);
      const ParamSet bands = eqBands({eqBand("bell", f, 6.0, 1.0)});
      const double onePass = measureTwoSidedGainDb(pe, bands, f);
      const double twoPass = measureTwoSidedGainDb(zp, bands, f);
      worst = std::max(worst, std::abs(twoPass - 2.0 * onePass));
      check(std::abs(twoPass - 2.0 * onePass) < 0.02,
            "zero-phase EQ: at " + std::to_string(static_cast<int>(f)) +
                " Hz the response is the square (" + std::to_string(twoPass) +
                " dB vs 2 x " + std::to_string(onePass) + " dB)");
    }

    // An absolute anchor as well, so a consistently wrong pair of
    // measurements cannot satisfy the ratio. A +6 dB bell at its own centre
    // must come back at exactly +12.
    const double f = snapHz(1000.0, kWindow);
    const double centre =
        measureTwoSidedGainDb(zp, eqBands({eqBand("bell", f, 6.0, 1.0)}), f);
    std::printf("       zero-phase EQ: +6 dB bell measures %+.4f dB at centre "
                "(expect +12), worst square error %.4f dB\n", centre, worst);
    check(std::abs(centre - 12.0) < 0.02,
          "zero-phase EQ: a +6 dB bell is +12 dB at centre (got " +
              std::to_string(centre) + ")");
  }

  // --- (d) reach is measured from the filter, not assumed ------------------
  //
  // The declared latency is chunk + reach, and reach is where the cascade's
  // remaining tail falls below 1e-12 of its L1 norm. A slow filter must
  // therefore declare MORE latency than a fast one; a node that declared a
  // fixed number would pass every test above.
  {
    auto latencyOf = [](const ParamSet& p) {
      Chain chain;
      Chain::SlotConfig cfg;
      cfg.nodeId = "zp";
      chain.add(std::make_unique<ZeroPhaseEQ>(), p, cfg);
      chain.prepare(makeSpec(1, 1024, 4096));
      return chain.totalLatencyFrames();
    };

    const int fast = latencyOf(eqBands({eqBand("bell", 8000.0, 4.0, 0.7)}));
    const int slow = latencyOf(eqBands({eqBand("bell", 400.0, 4.0, 2.0)}));
    const int capped = latencyOf(eqBands({eqBand("bell", 20.0, 6.0, 12.0)}));
    std::printf("       zero-phase EQ reach: 8 kHz Q0.7 -> %d frames, "
                "400 Hz Q2 -> %d, 20 Hz Q12 -> %d (cap 48000)\n",
                fast, slow, capped);

    // Proportional to the filter's decay rate, which is 20x slower here.
    check(slow > 10 * fast,
          "zero-phase EQ: a slow filter declares a proportionally longer "
          "window (" + std::to_string(fast) + " vs " + std::to_string(slow) + ")");
    check(capped <= 48000,
          "zero-phase EQ: the reach cap bounds the declared latency (" +
              std::to_string(capped) + ")");

    // AND WHAT THE CAP COSTS, measured rather than argued. A 20 Hz bell at
    // Q 12 rings for about 5.3 SECONDS at the 1e-12 criterion; no offline node
    // can hold half a million samples of lookahead, so past the cap the
    // reverse pass starts from rest closer to the output than the filter's
    // tail warrants and the response stops being exactly even.
    //
    // The question a test can answer is what that costs relative to the
    // alternative rather than in the abstract, so the reference is the same
    // filter through `parametric_eq` — which is what the user would otherwise
    // have used.
    const ParamSet slowBand = eqBands({eqBand("bell", 20.0, 6.0, 12.0)});
    const RawResponse zpR = rawResponse(std::make_unique<ZeroPhaseEQ>(),
                                        slowBand, 1 << 19, 131072);
    const RawResponse peR = rawResponse(std::make_unique<ParametricEQ>(),
                                        slowBand, 1 << 19, 131072);

    const PhaseReport zpLow = phaseAboutCentre(zpR, 1 << 18, 20.0, 100.0, 1e-6);
    const PhaseReport zpRest = phaseAboutCentre(zpR, 1 << 18, 100.0, 20000.0, 1e-6);
    const PhaseReport peLow = phaseAboutCentre(peR, 1 << 18, 20.0, 100.0, 1e-6);

    std::printf("       zero-phase EQ past the cap (20 Hz Q12): worst phase "
                "%.2f deg below 100 Hz, %.3f deg above; the same filter "
                "minimum-phase is %.1f deg\n",
                zpLow.worstPhaseDeg, zpRest.worstPhaseDeg, peLow.worstPhaseDeg);

    check(peLow.worstPhaseDeg > 15.0,
          "the minimum-phase reference really does shift phase here, so the "
          "comparison is not vacuous (" + std::to_string(peLow.worstPhaseDeg) +
              " deg)");
    // 6.6 deg against 19.3: BETTER, NOT ZERO, and the factor is 3 rather than
    // the 1e10 this node manages on anything that fits inside the cap. The
    // assertion is written at what was measured because a looser one would let
    // a real regression through and a tighter one would be a wish. The number
    // and its cause are in findings.md; the planner-side consequence is in
    // open-questions.md.
    check(zpLow.worstPhaseDeg < 0.5 * peLow.worstPhaseDeg,
          "zero-phase EQ: past the reach cap it is degraded but still ahead of "
          "minimum phase (" + std::to_string(zpLow.worstPhaseDeg) + " deg vs " +
              std::to_string(peLow.worstPhaseDeg) + ")");
    check(zpRest.worstPhaseDeg < 0.3,
          "zero-phase EQ: and above 100 Hz the truncation is worth only " +
              std::to_string(zpRest.worstPhaseDeg) + " deg");

    // The other half of the statement, and the one that keeps the assertion
    // above from reading as "this node is approximate": a filter whose ring
    // FITS inside the cap is exact, not nearly exact.
    const RawResponse inside =
        rawResponse(std::make_unique<ZeroPhaseEQ>(),
                    eqBands({eqBand("bell", 100.0, 6.0, 1.0)}), 1 << 18, 65536);
    const PhaseReport insidePh = phaseAboutCentre(inside, 1 << 17, 20.0, 20000.0, 1e-6);
    std::printf("       zero-phase EQ inside the cap (100 Hz Q1, reach %d): "
                "worst phase %.3e deg\n",
                inside.latency, insidePh.worstPhaseDeg);
    check(inside.latency < 48000 && insidePh.worstPhaseDeg < 1e-6,
          "zero-phase EQ: a 100 Hz filter fits inside the cap and is exactly "
          "zero phase (" + std::to_string(insidePh.worstPhaseDeg) + " deg)");

    // An EQ with no bands does nothing, so it must declare nothing. This is
    // `StereoWidth`'s bug from findings.md, one node later.
    check(latencyOf(ParamSet{}) == 0,
          "zero-phase EQ: a bandless EQ declares no latency at all");
  }
}

// 24. Linear-phase EQ correctness.
void testLinearPhaseEq() {
  auto withTaps = [](double taps, std::initializer_list<ParamSet> bands) {
    ParamSet p = eqBands(bands);
    p.set("taps", taps);
    return p;
  };

  // --- (a) latency is exactly (taps - 1) / 2 ------------------------------
  //
  // Including for an even request, which is rounded down to an odd tap count
  // rather than rejected — an even-length symmetric FIR has a half-integer
  // group delay, and this engine has nowhere to put half a sample.
  {
    auto latencyOf = [&](double taps) {
      Chain chain;
      Chain::SlotConfig cfg;
      cfg.nodeId = "lp";
      chain.add(std::make_unique<LinearPhaseEQ>(),
                withTaps(taps, {eqBand("bell", 3000.0, 3.0, 1.0)}), cfg);
      chain.prepare(makeSpec(1, 1024, 4096));
      return chain.totalLatencyFrames();
    };

    check(latencyOf(511.0) == 255, "linear-phase EQ: 511 taps -> 255 frames");
    check(latencyOf(2047.0) == 1023, "linear-phase EQ: 2047 taps -> 1023 frames");
    check(latencyOf(512.0) == 255,
          "linear-phase EQ: an even 512 taps becomes 511, latency 255 (got " +
              std::to_string(latencyOf(512.0)) + ")");

    // The latency check above cannot see whether the tap count was actually
    // made odd: (512-1)/2 and (511-1)/2 are both 255, so a node that kept an
    // even kernel with a stray zero tap on the end would report the right
    // latency and pass. The kernel LENGTH is the only place that shows.
    {
      LinearPhaseEQ even;
      even.prepare(makeSpec(1, 1024, 4096));
      even.setParams(withTaps(512.0, {eqBand("bell", 3000.0, 3.0, 1.0)}));
      check(even.kernel().size() == 511,
            "linear-phase EQ: and the kernel really is 511 taps, not 512 with a "
            "zero on the end (got " + std::to_string(even.kernel().size()) + ")");
    }
    check(latencyOf(1e5) == 4095,
          "linear-phase EQ: an absurd tap count clamps to 8191, latency 4095");
    check(latencyOf(0.0) == 7,
          "linear-phase EQ: zero taps clamps to the 15-tap floor, latency 7");
  }

  // --- (b) the kernel is symmetric to the last bit ------------------------
  //
  // Checked on the coefficients rather than on the output, because this is the
  // property everything else rests on and it should be EXACT, not "within
  // round-off". The design averages each mirrored pair explicitly to make it so.
  {
    LinearPhaseEQ node;
    PrepareSpec spec = makeSpec(1, 1024, 4096);
    node.prepare(spec);
    node.setParams(withTaps(1023.0, {eqBand("bell", 400.0, -5.0, 1.5),
                                     eqBand("highshelf", 8000.0, 4.0, 0.707)}));
    const std::vector<Sample>& k = node.kernel();
    check(k.size() == 1023, "linear-phase EQ: the kernel is 1023 taps long");

    double worst = 0.0;
    const size_t half = k.size() / 2;
    for (size_t j = 0; j <= half; ++j)
      worst = std::max(worst, std::abs(k[half + j] - k[half - j]));
    check(worst == 0.0,
          "linear-phase EQ: kernel[M+k] == kernel[M-k] exactly (worst diff " +
              std::to_string(worst) + ")");
  }

  // --- (c) a Dirac lands exactly on the declared latency -------------------
  //
  // Cross-correlation against a Dirac is just the response itself, so the lag
  // that maximises it is the position of the response's peak. For a symmetric
  // kernel that is its centre, which must be the declared latency exactly —
  // not within a sample.
  {
    const RawResponse r =
        rawResponse(std::make_unique<LinearPhaseEQ>(),
                    withTaps(2047.0, {eqBand("bell", 2000.0, 5.0, 1.0)}),
                    32768, 4096);

    int64_t bestLag = -1;
    double best = 0.0;
    for (int64_t lag = 0; lag < 8192; ++lag) {
      // sum_n h[n] * dirac[n - impulseAt - lag], which is one term.
      const double c = std::abs(r.h[static_cast<size_t>(4096 + lag)]);
      if (c > best) { best = c; bestLag = lag; }
    }
    check(bestLag == r.latency,
          "linear-phase EQ: a Dirac correlates best at lag " +
              std::to_string(bestLag) + ", declared latency " +
              std::to_string(r.latency));

    const PhaseReport ph = phaseAboutCentre(r, 16384, 20.0, 20000.0, 1e-6);
    std::printf("       linear-phase EQ: declared latency %d, %d bins, worst "
                "phase %.3e deg, worst group delay error %.3e samples\n",
                r.latency, ph.bins, ph.worstPhaseDeg, ph.worstGroupDelay);
    check(ph.bins > 2000, "linear-phase EQ: the phase probe saw " +
                              std::to_string(ph.bins) + " usable bins");
    check(ph.worstPhaseDeg < 1e-8,
          "linear-phase EQ: phase is exactly linear with slope = the declared "
          "latency (residual " + std::to_string(ph.worstPhaseDeg) + " deg)");
    check(ph.worstGroupDelay < 1e-9,
          "linear-phase EQ: group delay is flat and equals the declared "
          "latency (worst deviation " + std::to_string(ph.worstGroupDelay) +
              " samples)");

    check(symmetryError(r, 1023) == 0.0,
          "linear-phase EQ: the rendered impulse response is bit-exactly "
          "symmetric about the declared latency");
  }

  // --- (d) an EQ with no bands is an exact delay --------------------------
  //
  // Frequency-sampling a flat magnitude gives a Dirac, and windowing a Dirac
  // leaves it alone. So a bandless linear-phase EQ must be a PURE DELAY, bit
  // exact — which is a much sharper statement than "roughly transparent" and
  // catches any stray normalisation in the design.
  {
    ParamSet flat;
    flat.set("taps", 255.0);
    const RawResponse r =
        rawResponse(std::make_unique<LinearPhaseEQ>(), flat, 8192, 1000);
    double worst = 0.0;
    for (size_t i = 0; i < r.h.size(); ++i) {
      const double want = (static_cast<int64_t>(i) == r.centre) ? 1.0 : 0.0;
      worst = std::max(worst, std::abs(r.h[i] - want));
    }
    check(worst == 0.0,
          "linear-phase EQ: no bands gives a bit-exact 127-sample delay (worst "
          "error " + std::to_string(worst) + ")");
  }

  // --- (e) the magnitude is the one it was asked for -----------------------
  //
  // Against the analytic expectation, not against `parametric_eq`: both nodes
  // are designed from the same cascade, so comparing them would pass for any
  // filter whose maths is consistently wrong.
  {
    constexpr int64_t kWindow = 16384;
    auto lp = [] { return std::unique_ptr<Node>(new LinearPhaseEQ()); };
    const double f = snapHz(1000.0, kWindow);

    const ParamSet fir = withTaps(4095.0, {eqBand("bell", f, 6.0, 1.0)});
    const ParamSet iir = eqBands({eqBand("bell", f, 6.0, 1.0)});
    const double centre = measureTwoSidedGainDb(lp, fir, f);
    std::printf("       linear-phase EQ: +6 dB bell at 1 kHz measures %+.4f dB "
                "at centre\n", centre);
    check(std::abs(centre - 6.0) < 0.05,
          "linear-phase EQ: a +6 dB bell is +6 dB at centre (got " +
              std::to_string(centre) + ")");

    // And the same magnitude as the minimum-phase node at every frequency —
    // which is the actual claim, and a sharper one than "transparent away from
    // centre". A Q 1 bell at 1 kHz is +0.042 dB at 80 Hz, not 0: that is the
    // skirt, and a test asserting it were zero would be asserting a number
    // that is arithmetically wrong.
    auto pe = [] { return std::unique_ptr<Node>(new ParametricEQ()); };
    double worst = 0.0;
    for (double hz : {80.0, 300.0, 3000.0, 12000.0}) {
      const double probe = snapHz(hz, kWindow);
      const double firDb = measureTwoSidedGainDb(lp, fir, probe);
      const double iirDb = measureTwoSidedGainDb(pe, iir, probe);
      worst = std::max(worst, std::abs(firDb - iirDb));
      check(std::abs(firDb - iirDb) < 0.02,
            "linear-phase EQ: at " + std::to_string(static_cast<int>(probe)) +
                " Hz it matches the minimum-phase magnitude (" +
                std::to_string(firDb) + " vs " + std::to_string(iirDb) + " dB)");
    }
    std::printf("       linear-phase vs minimum-phase magnitude: worst "
                "difference %.4f dB across 80 Hz-12 kHz\n", worst);

    // NOT the square. This is the one-line difference from `zero_phase_eq`,
    // and stating it as an assertion means the two nodes cannot be swapped
    // without something failing.
    auto zp = [] { return std::unique_ptr<Node>(new ZeroPhaseEQ()); };
    const double squared =
        measureTwoSidedGainDb(zp, eqBands({eqBand("bell", f, 6.0, 1.0)}), f);
    check(std::abs(squared - 2.0 * centre) < 0.1,
          "linear-phase EQ: the zero-phase node at the same settings is twice "
          "the dB (" + std::to_string(squared) + " vs " +
              std::to_string(centre) + ")");
  }

  // --- (f) the design window earns its place ------------------------------
  //
  // Frequency sampling produces an ideal kernel of infinite length. Truncating
  // it to `taps` with no window is the textbook Gibbs mistake, and the four
  // spot frequencies above do not see it — removing the Kaiser entirely passed
  // every other assertion in this section. So this compares the WHOLE realised
  // magnitude, bin by bin, against the minimum-phase cascade it is supposed to
  // match, on a filter demanding enough to ring: a deep narrow cut plus a
  // shelf.
  //
  // Done on the coefficients rather than by rendering. A direct convolution
  // over twenty probe tones is thirty seconds of test time to learn less than
  // one FFT of the kernel says exactly.
  //
  // 255 taps SPECIFICALLY, because the window only earns anything when the
  // ideal kernel does not fit in the tap budget. Measured at 4095 taps the
  // windowed design is 0.0034 dB from the target and the unwindowed one is
  // 0.0000 — the window is a small net LOSS there, since it tapers
  // coefficients that were already converged. At 255 taps the unwindowed
  // design leaks 0.238 dB and the windowed one 0.0035. The window is kept for
  // the second case and the first case is the price.
  {
    constexpr size_t kN = 1 << 16;
    const ParamSet bands = eqBands({eqBand("bell", 300.0, -15.0, 4.0)});

    LinearPhaseEQ node;
    node.prepare(makeSpec(1, 1024, 4096));
    ParamSet withCount = bands;
    withCount.set("taps", 255.0);
    node.setParams(withCount);

    std::vector<std::complex<double>> fir(kN, std::complex<double>(0.0, 0.0));
    for (size_t i = 0; i < node.kernel().size(); ++i) fir[i] = node.kernel()[i];
    fftInPlace(fir);

    // The reference is the minimum-phase node's own impulse response, measured
    // the same way anything else here is measured.
    const RawResponse ref = rawResponse(std::make_unique<ParametricEQ>(), bands,
                                        static_cast<int64_t>(kN), 0);
    std::vector<std::complex<double>> iir(kN, std::complex<double>(0.0, 0.0));
    for (size_t i = 0; i < kN; ++i) iir[i] = ref.h[i];
    fftInPlace(iir);

    double worst = 0.0, worstHz = 0.0;
    for (size_t k = 1; k <= kN / 2; ++k) {
      const double hz = static_cast<double>(k) * kCanonicalSampleRate /
                        static_cast<double>(kN);
      // FAR FROM THE FEATURE, where the target is flat 0 dB. Measuring at the
      // notch itself would measure the design's FREQUENCY RESOLUTION instead —
      // a 75 Hz-wide cut through a 4095-tap design comes back 3.1 dB shallow,
      // which is the documented fs/taps limit and not ripple at all. The
      // window's job is to keep that feature from splattering across the rest
      // of the spectrum, and that is what this looks at.
      if (hz < 2000.0 || hz > 20000.0) continue;
      const double d = 20.0 * std::log10(std::abs(fir[k]) / std::abs(iir[k]));
      if (std::abs(d) > std::abs(worst)) { worst = d; worstHz = hz; }
    }
    std::printf("       linear-phase EQ: a -15 dB Q4 cut at 300 Hz leaks "
                "%+.4f dB of ripple at %.0f Hz\n", worst, worstHz);
    check(std::abs(worst) < 0.02,
          "linear-phase EQ: at 255 taps the design window keeps a sharp feature "
          "from rippling across the rest of the band (" + std::to_string(worst) +
              " dB; unwindowed it is 0.238)");
  }

  // --- (g) pre-ringing exists, and is the honest cost ----------------------
  //
  // A linear-phase filter puts as much energy before a transient as after it.
  // Asserting it is present is not a bug report: it is the property that makes
  // `parametric_eq` the right default, and a node that quietly had none would
  // not be linear phase.
  {
    const RawResponse r =
        rawResponse(std::make_unique<LinearPhaseEQ>(),
                    withTaps(2047.0, {eqBand("bell", 200.0, -12.0, 2.0)}),
                    32768, 4096);
    double pre = 0.0, post = 0.0;
    for (int64_t k = 1; k <= 1023; ++k) {
      pre += r.h[static_cast<size_t>(r.centre - k)] *
             r.h[static_cast<size_t>(r.centre - k)];
      post += r.h[static_cast<size_t>(r.centre + k)] *
              r.h[static_cast<size_t>(r.centre + k)];
    }
    check(pre > 0.0 && std::abs(pre - post) / post < 1e-9,
          "linear-phase EQ: pre-ringing energy equals post-ringing energy (" +
              std::to_string(pre) + " vs " + std::to_string(post) + ")");
  }
}

// 25. Multiband compressor correctness.

ParamSet mbBand(double crossoverHz, double thresholdDb, double ratio,
                double kneeDb, double attackMs, double releaseMs,
                double makeupDb) {
  ParamSet b;
  b.set("crossover_hz", crossoverHz).set("threshold_dbfs", thresholdDb)
      .set("ratio_to_1", ratio).set("knee_db", kneeDb)
      .set("attack_ms", attackMs).set("release_ms", releaseMs)
      .set("makeup_gain_db", makeupDb);
  return b;
}

ParamSet mbParams(std::initializer_list<ParamSet> bands, double lookaheadMs,
                  double stereoLink = 1.0) {
  ParamSet p;
  p.set("lookahead_ms", lookaheadMs).set("stereo_link", stereoLink)
      .set("crossover_stopband_db", 80.0);
  for (const ParamSet& b : bands) p.addItem(b);
  return p;
}

AudioBuffer runMultiband(const ParamSet& p, const AudioBuffer& in) {
  Chain chain;
  Chain::SlotConfig cfg;
  cfg.nodeId = "mb";
  chain.add(std::unique_ptr<Node>(new MultibandCompressor()), p, cfg);
  chain.prepare(makeSpec(in.channels(), 1024, in.frames()));
  AudioBuffer out;
  chain.render(in, out, 1024);
  return out;
}

int mbLatency(const ParamSet& p, int channels = 2) {
  Chain chain;
  Chain::SlotConfig cfg;
  cfg.nodeId = "mb";
  chain.add(std::unique_ptr<Node>(new MultibandCompressor()), p, cfg);
  chain.prepare(makeSpec(channels, 1024, 4096));
  return chain.totalLatencyFrames();
}

double windowRmsDb(const AudioBuffer& b, int channel, int64_t from, int64_t to) {
  double s = 0.0;
  for (int64_t i = from; i < to; ++i)
    s += b.channel(channel)[i] * b.channel(channel)[i];
  return 10.0 * std::log10(std::max(s / static_cast<double>(to - from), 1e-300));
}

void testMultibandCompressor() {
  const int64_t n = 65536;
  AudioBuffer in(2, n);
  {
    CounterRng rng(4711);
    for (int c = 0; c < 2; ++c) {
      double lp = 0.0;
      for (int64_t i = 0; i < n; ++i) {
        const double white = rng.bipolar(c, i);
        lp = 0.995 * lp + 0.005 * white; // real weight under the crossovers
        const double t = static_cast<double>(i) / kCanonicalSampleRate;
        in.channel(c)[i] = 0.25 * (white + 8.0 * lp) +
                           0.15 * std::sin(2.0 * M_PI * 55.0 * t + c) +
                           0.10 * std::sin(2.0 * M_PI * 7000.0 * t);
      }
    }
  }

  // --- (a) THE UNITY NULL --------------------------------------------------
  //
  // The claim the whole crossover design exists to keep: three bands, every one
  // at 1:1 with 0 dB makeup, and the node must return its input DELAYED —
  // not "flat within a dB", not "phase-shifted but magnitude-flat". The bands
  // carry different attack and release times and the lookahead is non-zero, so
  // every piece of per-band machinery is running; it just has nothing to say.
  //
  // Note the last band's `crossover_hz`: it is deliberately absurd, because it
  // is the one value in the parameter set that is never read. A band that runs
  // to Nyquist has no upper corner.
  {
    const ParamSet unity = mbParams({mbBand(120.0, -18.0, 1.0, 6.0, 30.0, 250.0, 0.0),
                                     mbBand(2500.0, -18.0, 1.0, 6.0, 8.0, 120.0, 0.0),
                                     mbBand(-999.0, -18.0, 1.0, 6.0, 2.0, 80.0, 0.0)},
                                    5.0);
    const AudioBuffer out = runMultiband(unity, in);
    const double diff = maxAbsDiff(in, out, n);
    const double nullDb = 20.0 * std::log10(std::max(diff, 1e-300));
    std::printf("       multiband: three bands at 1:1 null against the input at "
                "%.1f dB\n", nullDb);
    check(nullDb < -180.0,
          "multiband: unity through the crossover reconstructs exactly (null " +
              std::to_string(nullDb) + " dB)");

    // AND THE COMPARISON IS NOT VACUOUS. If the node were a pass-through the
    // null above would also be perfect, so something has to prove the bands
    // are separable at all: 1 dB of makeup on one band alone must break it by
    // roughly that dB.
    ParamSet tilted = mbParams({mbBand(120.0, -18.0, 1.0, 6.0, 30.0, 250.0, 1.0),
                                mbBand(2500.0, -18.0, 1.0, 6.0, 8.0, 120.0, 0.0),
                                mbBand(-999.0, -18.0, 1.0, 6.0, 2.0, 80.0, 0.0)},
                               5.0);
    const AudioBuffer tiltedOut = runMultiband(tilted, in);
    const double tiltedDiff = maxAbsDiff(in, tiltedOut, n);
    check(tiltedDiff > 0.001,
          "multiband: 1 dB of makeup on the low band alone genuinely changes "
          "the output, so the null is a measurement (" +
              std::to_string(tiltedDiff) + ")");
  }

  // --- (b) each band's makeup lands in its own band -------------------------
  //
  // With every band at 1:1 the node is exactly a three-band linear-phase gain
  // stage, so the steady-state gain at a frequency inside band k must be that
  // band's makeup and nothing else. This is what says the crossover SPLITS
  // rather than merely reconstructs — an implementation that sent the whole
  // signal through band 0 would still null perfectly in (a).
  {
    auto mb = [] { return std::unique_ptr<Node>(new MultibandCompressor()); };
    const ParamSet gains =
        mbParams({mbBand(150.0, -18.0, 1.0, 0.0, 30.0, 250.0, 6.0),
                  mbBand(2500.0, -18.0, 1.0, 0.0, 8.0, 120.0, 0.0),
                  mbBand(-999.0, -18.0, 1.0, 0.0, 2.0, 80.0, -4.0)},
                 5.0);

    const double lowHz = snapHz(45.0, 16384);
    const double midHz = snapHz(800.0, 16384);
    const double highHz = snapHz(10000.0, 16384);
    const double lowDb = measureTwoSidedGainDb(mb, gains, lowHz);
    const double midDb = measureTwoSidedGainDb(mb, gains, midHz);
    const double highDb = measureTwoSidedGainDb(mb, gains, highHz);
    std::printf("       multiband band gains: %.0f Hz %+.4f dB (want +6), "
                "%.0f Hz %+.4f dB (want 0), %.0f Hz %+.4f dB (want -4)\n",
                lowHz, lowDb, midHz, midDb, highHz, highDb);

    checkNear(lowDb, 6.0, 0.05, "multiband: the low band's makeup lands below the "
                                "first crossover");
    checkNear(midDb, 0.0, 0.05, "multiband: the mid band is untouched");
    checkNear(highDb, -4.0, 0.05, "multiband: the high band's makeup lands above "
                                  "the second crossover");
  }

  // --- (c) the bands compress independently --------------------------------
  //
  // THE ONLY REASON THIS NODE EXISTS. A loud 50 Hz tone and a quiet 8 kHz tone:
  // a broadband compressor ducks the 8 kHz tone every time the bass moves,
  // which is the pumping a multiband is bought to avoid. Measured against a
  // NULL RENDER — the same node with every band at 1:1 — rather than against
  // the raw source, so the crossover's own (zero) contribution cannot be read
  // as compression.
  {
    constexpr size_t kN = 32768;
    constexpr int64_t kSettle = 65536;
    const int64_t total = kSettle + static_cast<int64_t>(kN);
    auto binOf = [](double hz) {
      return static_cast<size_t>(std::round(hz * static_cast<double>(kN) /
                                            kCanonicalSampleRate));
    };
    const size_t lowBin = binOf(50.0), highBin = binOf(8000.0);
    const double lowHz =
        static_cast<double>(lowBin) * kCanonicalSampleRate / static_cast<double>(kN);
    const double highHz =
        static_cast<double>(highBin) * kCanonicalSampleRate / static_cast<double>(kN);

    AudioBuffer tones(1, total);
    for (int64_t i = 0; i < total; ++i) {
      const double t = static_cast<double>(i) / kCanonicalSampleRate;
      tones.channel(0)[i] = 0.6 * std::sin(2.0 * M_PI * lowHz * t) +
                            0.02 * std::sin(2.0 * M_PI * highHz * t);
    }

    auto binDb = [&](const AudioBuffer& buf, size_t bin) {
      const auto pw = powerSpectrum(buf.channel(0) + kSettle, kN);
      return 10.0 * std::log10(std::max(pw[bin], 1e-300));
    };

    const ParamSet flat = mbParams({mbBand(150.0, -30.0, 1.0, 0.0, 20.0, 200.0, 0.0),
                                    mbBand(2500.0, -30.0, 1.0, 0.0, 20.0, 200.0, 0.0),
                                    mbBand(-999.0, -30.0, 1.0, 0.0, 20.0, 200.0, 0.0)},
                                   3.0);
    const ParamSet lowOnly = mbParams({mbBand(150.0, -30.0, 8.0, 0.0, 20.0, 200.0, 0.0),
                                       mbBand(2500.0, -30.0, 1.0, 0.0, 20.0, 200.0, 0.0),
                                       mbBand(-999.0, -30.0, 1.0, 0.0, 20.0, 200.0, 0.0)},
                                      3.0);

    const AudioBuffer nullRender = runMultiband(flat, tones);
    const AudioBuffer squashed = runMultiband(lowOnly, tones);

    const double lowChange = binDb(squashed, lowBin) - binDb(nullRender, lowBin);
    const double highChange = binDb(squashed, highBin) - binDb(nullRender, highBin);

    // The reference: one broadband compressor at the same threshold and ratio.
    Chain wide;
    Chain::SlotConfig wideCfg;
    wideCfg.nodeId = "wide";
    wide.add(std::make_unique<Compressor>(),
             compressorParams(-30.0, 8.0, 0.0, 20.0, 200.0, 3.0), wideCfg);
    wide.prepare(makeSpec(1, 1024, total));
    AudioBuffer wideOut;
    wide.render(tones, wideOut, 1024);
    const double wideHighChange = binDb(wideOut, highBin) - binDb(tones, highBin);

    std::printf("       multiband selectivity: 50 Hz %+.2f dB, 8 kHz %+.3f dB; "
                "the same move broadband costs the 8 kHz tone %+.2f dB\n",
                lowChange, highChange, wideHighChange);

    check(lowChange < -8.0,
          "multiband: the low band is genuinely compressed (" +
              std::to_string(lowChange) + " dB)");
    check(std::abs(highChange) < 0.1,
          "multiband: and the high band is untouched while it happens (" +
              std::to_string(highChange) + " dB)");
    check(wideHighChange < -8.0,
          "and a broadband compressor really does duck the high tone, so the "
          "comparison is not vacuous (" + std::to_string(wideHighChange) + " dB)");
  }

  // --- (d) one band is bit-identical to the plain compressor ---------------
  //
  // A one-band multiband has no crossover to run, so it must reduce to exactly
  // the node it is built out of — BIT exactly, since the arithmetic is the same
  // arithmetic. This is the sharpest available statement that the detector sees
  // the right signal: feed it the delayed audio instead of the undelayed band
  // and the lookahead stops anticipating, which no spectrum-shaped assertion
  // would notice but this does immediately.
  //
  // AT `oversample_factor` 1, deliberately. The node oversamples 4x by default
  // and the round trip is not bit-exact, so the identity would be hidden behind
  // a resampler. Factor 1 is not a special case invented for the test: it is
  // exactly `compressor`'s own arithmetic, which is the thing being claimed —
  // and `compressor` is not oversampled either, which is why it passes the
  // aliasing requirement at -104 dB while three of these summed do not.
  {
    ParamSet single = mbParams({mbBand(-999.0, -24.0, 4.0, 6.0, 5.0, 150.0, 3.0)},
                               4.0, 0.8);
    single.set("oversample_factor", 1.0);
    const AudioBuffer mbOut = runMultiband(single, in);

    ParamSet wide = compressorParams(-24.0, 4.0, 6.0, 5.0, 150.0, 4.0);
    wide.set("stereo_link", 0.8).set("makeup_gain_db", 3.0);
    Chain chain;
    Chain::SlotConfig cfg;
    cfg.nodeId = "cmp";
    chain.add(std::make_unique<Compressor>(), wide, cfg);
    chain.prepare(makeSpec(2, 1024, n));
    AudioBuffer wideOut;
    chain.render(in, wideOut, 1024);

    const double diff = maxAbsDiff(mbOut, wideOut, n);
    check(diff == 0.0,
          "multiband: a single band is bit-identical to the plain compressor "
          "(diff " + std::to_string(diff) + ")");

    // AND OVERSAMPLING IS NOT TRANSPARENT, which is worth printing rather than
    // asserting away. At 4x the detector reads an UPSAMPLED signal, whose
    // reconstructed inter-sample peaks are genuinely higher than the sample
    // peaks the base-rate detector saw — so the oversampled node compresses a
    // little harder. That is a true-peak detector behaving like one, not an
    // error, and it is why the bit-exact comparison above is pinned at
    // factor 1.
    ParamSet oversampled = single;
    oversampled.set("oversample_factor", 4.0);
    const AudioBuffer osOut = runMultiband(oversampled, in);
    std::printf("       multiband at 4x differs from the base-rate compressor by "
                "%.1f dB — inter-sample peaks the base rate could not see\n",
                20.0 * std::log10(std::max(maxAbsDiff(osOut, wideOut, n), 1e-300)));
    check(mbLatency(single) == mbLatency(single, 1) &&
              mbLatency(single) == static_cast<int>(std::lround(4.0 * 96.0)),
          "multiband: and declares only the lookahead, " +
              std::to_string(mbLatency(single)) + " frames");
  }

  // --- (d2) each band's attack time is the one it was given ----------------
  //
  // MEASURED IN MILLISECONDS, because nothing else in this section can see it.
  // Attack and release are coefficients per SAMPLE, so a detector running at 4x
  // whose coefficients were derived from the base rate has every time constant
  // four times too fast — a completely different-sounding compressor. The
  // bit-exact comparison above cannot catch it (at factor 1 the two rates are
  // the same number) and neither can any spectrum, so this measures the
  // ballistic directly: time to reach 63% of the final gain reduction on a
  // burst that lives entirely in one band.
  {
    const double attackMs = 20.0;
    const int64_t onset = 8192;
    const int64_t total = onset + static_cast<int64_t>(0.5 * kCanonicalSampleRate);
    const double toneHz = 3000.0, amp = 0.5;

    AudioBuffer burst(1, total);
    for (int64_t i = onset; i < total; ++i)
      burst.channel(0)[i] = amp * std::sin(2.0 * M_PI * toneHz *
                                           static_cast<double>(i - onset) /
                                           kCanonicalSampleRate);

    // Band 0 passive, band 1 (everything above 500 Hz) doing the work.
    const ParamSet p = mbParams({mbBand(500.0, -18.0, 1.0, 0.0, 20.0, 200.0, 0.0),
                                 mbBand(-999.0, -20.0, 4.0, 0.0, attackMs, 200.0, 0.0)},
                                0.0);
    const ParamSet flat = mbParams({mbBand(500.0, -18.0, 1.0, 0.0, 20.0, 200.0, 0.0),
                                    mbBand(-999.0, -20.0, 1.0, 0.0, attackMs, 200.0, 0.0)},
                                   0.0);
    const AudioBuffer out = runMultiband(p, burst);
    const AudioBuffer ref = runMultiband(flat, burst);

    // Gain per cycle, from the peak of each 32-sample period. A tone's
    // instantaneous sample ratio is meaningless near a zero crossing.
    const int64_t period = static_cast<int64_t>(kCanonicalSampleRate / toneHz);
    auto grAt = [&](int64_t from) {
      double a = 0.0, b = 0.0;
      for (int64_t i = from; i < from + period && i < total; ++i) {
        a = std::max(a, std::abs(out.channel(0)[i]));
        b = std::max(b, std::abs(ref.channel(0)[i]));
      }
      return (b > 1e-12) ? -20.0 * std::log10(std::max(a / b, 1e-12)) : 0.0;
    };

    const double finalGr = grAt(total - 2 * period);
    int64_t reached = -1;
    for (int64_t i = onset; i + period < total; i += period)
      if (grAt(i) >= 0.632 * finalGr) { reached = i - onset; break; }
    const double measuredMs =
        static_cast<double>(reached) * 1000.0 / kCanonicalSampleRate;
    std::printf("       multiband band attack: %.1f ms measured against %.1f ms "
                "specified, final reduction %.2f dB\n",
                measuredMs, attackMs, finalGr);

    check(finalGr > 3.0,
          "multiband: the burst produces measurable gain reduction (" +
              std::to_string(finalGr) + " dB)");
    check(reached > 0 && measuredMs > attackMs * 0.5 && measuredMs < attackMs * 1.7,
          "multiband: a band's attack takes the time it was given (" +
              std::to_string(measuredMs) + " ms against " +
              std::to_string(attackMs) + ")");
  }

  // --- (e) the bands are time-aligned with each other ----------------------
  //
  // Each crossover stage costs its own latency, so the bands emerge staggered
  // by thousands of samples and have to be pulled back together BEFORE their
  // detectors run. The evidence is the raw impulse response: every band is a
  // symmetric linear-phase kernel, so if — and only if — they all sit on the
  // same centre does their sum stay symmetric about the declared latency. A
  // band left at its own latency shifts one kernel and the sum goes lopsided.
  {
    const ParamSet gains = mbParams({mbBand(800.0, -18.0, 1.0, 0.0, 30.0, 250.0, 4.0),
                                     mbBand(5000.0, -18.0, 1.0, 0.0, 8.0, 120.0, 0.0),
                                     mbBand(-999.0, -18.0, 1.0, 0.0, 2.0, 80.0, -3.0)},
                                    2.0);
    const RawResponse r = rawResponse(std::make_unique<MultibandCompressor>(),
                                      gains, 8192, 2048);
    const double asym = symmetryError(r, 1024);
    std::printf("       multiband: impulse response asymmetry about the declared "
                "latency %d is %.3e of peak\n", r.latency, asym);
    check(asym < 1e-9,
          "multiband: the bands are aligned, so the response is symmetric about "
          "the declared latency (worst " + std::to_string(asym) + " of peak)");
  }

  // --- (f) declared latency describes what the node actually does ----------
  {
    const ParamSet none = mbParams({}, 5.0);
    check(mbLatency(none) == 0,
          "multiband: a bandless multiband declares no latency at all (" +
              std::to_string(mbLatency(none)) + ")");

    auto twoBandsAt = [](double crossoverHz, double lookaheadMs) {
      return mbParams({mbBand(crossoverHz, -18.0, 2.0, 6.0, 20.0, 200.0, 0.0),
                       mbBand(-999.0, -18.0, 2.0, 6.0, 20.0, 200.0, 0.0)},
                      lookaheadMs);
    };

    // Lookahead is added to the crossover, exactly, in whole samples.
    const int noLook = mbLatency(twoBandsAt(1000.0, 0.0));
    const int look5 = mbLatency(twoBandsAt(1000.0, 5.0));
    check(look5 - noLook == static_cast<int>(std::lround(5.0 * 0.001 * kCanonicalSampleRate)),
          "multiband: lookahead adds exactly its own sample count (" +
              std::to_string(look5 - noLook) + ")");

    // A lower crossover needs a longer FIR and must say so. A node that
    // declared a fixed number would pass every other check here.
    const int low = mbLatency(twoBandsAt(100.0, 0.0));
    const int high = mbLatency(twoBandsAt(4000.0, 0.0));
    // These totals include the oversampler's fixed round trip, which is the
    // same for all three and therefore cannot produce the ratio below.
    std::printf("       multiband declared latency: 100 Hz -> %d frames, "
                "1 kHz -> %d, 4 kHz -> %d\n", low, noLook, high);
    check(low > 5 * high,
          "multiband: a low crossover declares proportionally more latency (" +
              std::to_string(low) + " vs " + std::to_string(high) + ")");
  }

  // --- (g) crossovers out of order are sorted, not stacked -----------------
  //
  // The caller is a language model, so descending corners will happen. The
  // repair has to be the RIGHT one, and reconstruction cannot tell the two
  // candidates apart: the complementary subtraction is an identity, so it holds
  // for any filter anywhere, including one placed somewhere nobody meant.
  //
  // Pushing each corner above the previous one turns (6000, 200) into
  // (6000, 6300): the mid band becomes a 300 Hz sliver and everything from
  // 200 Hz to 6 kHz is silently handled by the LOW band's settings. Sorting
  // turns it into (200, 6000), which is what was written. The probe at 1 kHz is
  // the only place the two differ, and it differs by the whole band gain.
  {
    auto mb = [] { return std::unique_ptr<Node>(new MultibandCompressor()); };
    const ParamSet backwards =
        mbParams({mbBand(6000.0, -18.0, 1.0, 0.0, 20.0, 200.0, 6.0),
                  mbBand(200.0, -18.0, 1.0, 0.0, 20.0, 200.0, 0.0),
                  mbBand(-999.0, -18.0, 1.0, 0.0, 20.0, 200.0, -4.0)},
                 2.0);

    const AudioBuffer out = runMultiband(backwards, in);
    const double nullDb =
        20.0 * std::log10(std::max(maxAbsDiff(in, runMultiband(
                                       mbParams({mbBand(6000.0, -18.0, 1.0, 0.0, 20.0, 200.0, 0.0),
                                                 mbBand(200.0, -18.0, 1.0, 0.0, 20.0, 200.0, 0.0),
                                                 mbBand(-999.0, -18.0, 1.0, 0.0, 20.0, 200.0, 0.0)},
                                                2.0), in), n), 1e-300));
    check(nullDb < -180.0,
          "multiband: descending crossovers still reconstruct exactly (" +
              std::to_string(nullDb) + " dB)");
    check(peak(out) > 0.0, "multiband: and still produce audio");

    const double lowDb = measureTwoSidedGainDb(mb, backwards, snapHz(80.0, 16384));
    const double midDb = measureTwoSidedGainDb(mb, backwards, snapHz(1000.0, 16384));
    const double highDb = measureTwoSidedGainDb(mb, backwards, snapHz(12000.0, 16384));
    std::printf("       multiband with corners written (6000, 200): 80 Hz "
                "%+.3f dB, 1 kHz %+.3f dB, 12 kHz %+.3f dB\n",
                lowDb, midDb, highDb);
    checkNear(lowDb, 6.0, 0.05,
              "multiband: out-of-order corners are SORTED, so the low band is "
              "still below 200 Hz");
    checkNear(midDb, 0.0, 0.05,
              "multiband: and 1 kHz is still the mid band rather than being "
              "swept into the low band's settings");
    checkNear(highDb, -4.0, 0.05, "multiband: and the high band is unchanged");
  }

  // --- (h) identical corners still leave a real band between them ----------
  //
  // A defence tested where it BINDS, which is the lesson from the linear-phase
  // EQ's design window in findings.md. Removing the minimum spacing survived
  // every check above, because nothing above ever wrote two corners at the same
  // frequency — and everywhere else the spacing is a 5% shift inside a
  // transition band a full octave wide, which is nothing at all.
  //
  // Where it binds is exactly there: two stages at the same corner are the same
  // filter twice, so the band between them is the product of a low-pass and its
  // own complement rather than a band. Measured, with the mid band at +12 dB
  // makeup: 1030 Hz reads +5.64 dB spaced and +4.77 dB unspaced.
  {
    MultibandCompressor node;
    node.prepare(makeSpec(1, 1024, 4096));
    node.setParams(mbParams({mbBand(1000.0, -18.0, 1.0, 0.0, 20.0, 200.0, 0.0),
                             mbBand(1000.0, -18.0, 1.0, 0.0, 20.0, 200.0, 12.0),
                             mbBand(-999.0, -18.0, 1.0, 0.0, 20.0, 200.0, 0.0)},
                            0.0));
    const std::vector<double>& corners = node.crossoverFrequencies();
    check(corners.size() == 2 && corners[1] > corners[0],
          "multiband: two corners written at the same frequency come back "
          "strictly increasing (" + std::to_string(corners.empty() ? 0.0 : corners[0]) +
              ", " + std::to_string(corners.size() < 2 ? 0.0 : corners[1]) + ")");

    // And the same statement in the audio, so the check above cannot be
    // satisfied by a getter that is right while the filters are not.
    auto mb = [] { return std::unique_ptr<Node>(new MultibandCompressor()); };
    const ParamSet stacked =
        mbParams({mbBand(1000.0, -18.0, 1.0, 0.0, 20.0, 200.0, 0.0),
                  mbBand(1000.0, -18.0, 1.0, 0.0, 20.0, 200.0, 12.0),
                  mbBand(-999.0, -18.0, 1.0, 0.0, 20.0, 200.0, 0.0)},
                 0.0);
    const double midDb = measureTwoSidedGainDb(mb, stacked, snapHz(1030.0, 16384));
    std::printf("       multiband with both corners at 1 kHz: the +12 dB mid "
                "band still reads %+.4f dB at 1030 Hz (unspaced it is +4.77)\n",
                midDb);
    check(midDb > 5.2,
          "multiband: the band between two identical corners is narrow but real (" +
              std::to_string(midDb) + " dB)");
  }
}

// 26. Transient designer correctness.

ParamSet tdParams(double attackDb, double sustainDb, double attackMs,
                  double sustainMs, double lookaheadMs, double stereoLink = 1.0) {
  ParamSet p;
  p.set("attack_db", attackDb).set("sustain_db", sustainDb)
      .set("attack_ms", attackMs).set("sustain_ms", sustainMs)
      .set("lookahead_ms", lookaheadMs).set("stereo_link", stereoLink);
  return p;
}

AudioBuffer runTransient(const ParamSet& p, const AudioBuffer& in) {
  Chain chain;
  Chain::SlotConfig cfg;
  cfg.nodeId = "td";
  chain.add(std::unique_ptr<Node>(new TransientDesigner()), p, cfg);
  chain.prepare(makeSpec(in.channels(), 1024, in.frames()));
  AudioBuffer out;
  chain.render(in, out, 1024);
  return out;
}

// A drum-like hit: instantaneous onset, exponential decay, two partials so the
// envelope is not a pure sinusoid's. Cosines, so the very first sample is the
// peak — the onset has to be a genuine step or there is no transient to shape.
void addHit(AudioBuffer& b, int64_t onset, double amp, double decayS) {
  for (int64_t i = onset; i < b.frames(); ++i) {
    const double t = static_cast<double>(i - onset) / kCanonicalSampleRate;
    const double env = std::exp(-t / decayS);
    b.channel(0)[i] += amp * env *
                       (0.6 * std::cos(2.0 * M_PI * 180.0 * t) +
                        0.4 * std::cos(2.0 * M_PI * 1450.0 * t));
  }
}

AudioBuffer transientHit(int64_t frames, int64_t onset, double amp,
                         double decayS) {
  AudioBuffer b(1, frames);
  addHit(b, onset, amp, decayS);
  return b;
}

void testTransientDesigner() {
  const int64_t frames = 65536;
  const int64_t onset = 8192;
  const int64_t ms = static_cast<int64_t>(kCanonicalSampleRate / 1000.0);

  // Peak of the leading edge, and RMS of the body well after it.
  auto peakDb = [&](const AudioBuffer& b) {
    double pk = 0.0;
    for (int64_t i = onset; i < onset + 3 * ms; ++i)
      pk = std::max(pk, std::abs(b.channel(0)[i]));
    return 20.0 * std::log10(std::max(pk, 1e-300));
  };
  auto bodyDb = [&](const AudioBuffer& b) {
    return windowRmsDb(b, 0, onset + 60 * ms, onset + 200 * ms);
  };

  // --- (a) BOTH CONTROLS AT ZERO IS A BIT-EXACT DELAY ----------------------
  //
  // The property that decides whether this node can sit in a chain by default.
  // Not "no measurable level change" — bit exact, because the gain is
  // pow(10, 0/20) and multiplying by one is the identity. The times and the
  // lookahead are deliberately non-trivial, so all three detectors and the
  // delay line are running the whole time and still change nothing.
  {
    const AudioBuffer hit = transientHit(frames, onset, 0.5, 0.08);
    const AudioBuffer out = runTransient(tdParams(0.0, 0.0, 25.0, 300.0, 3.0), hit);
    const double diff = maxAbsDiff(hit, out, frames);
    const double levelChange = windowRmsDb(out, 0, 0, frames) -
                               windowRmsDb(hit, 0, 0, frames);
    std::printf("       transient designer at 0/0: level change %+.4f dB, worst "
                "sample difference %.3e\n", levelChange, diff);
    check(diff == 0.0,
          "transient designer: both controls at zero is a bit-exact delay (worst "
          "difference " + std::to_string(diff) + ")");
    check(levelChange == 0.0,
          "transient designer: and therefore exactly no level change (" +
              std::to_string(levelChange) + " dB)");

    // Not vacuous: the same signal with the controls up must move.
    const AudioBuffer shaped =
        runTransient(tdParams(6.0, -6.0, 25.0, 300.0, 3.0), hit);
    check(maxAbsDiff(hit, shaped, frames) > 0.01,
          "transient designer: and the controls do something when they are not "
          "zero, so the null above is a measurement");
  }

  // --- (b) attack moves the leading edge and only the leading edge ---------
  {
    const AudioBuffer hit = transientHit(frames, onset, 0.5, 0.08);
    const AudioBuffer flat = runTransient(tdParams(0.0, 0.0, 20.0, 250.0, 2.0), hit);
    const AudioBuffer up = runTransient(tdParams(6.0, 0.0, 20.0, 250.0, 2.0), hit);
    const AudioBuffer down = runTransient(tdParams(-6.0, 0.0, 20.0, 250.0, 2.0), hit);

    const double upPeak = peakDb(up) - peakDb(flat);
    const double downPeak = peakDb(down) - peakDb(flat);
    const double upBody = bodyDb(up) - bodyDb(flat);
    std::printf("       transient attack +/-6 dB: peak %+.2f / %+.2f dB, body "
                "%+.4f dB\n", upPeak, downPeak, upBody);

    // A fully-developed transient gets close to the whole control value; the
    // shortfall is the differential not having quite reached 1, which is a
    // property of the material rather than of the node.
    check(upPeak > 4.5 && upPeak <= 6.05,
          "transient designer: +6 dB of attack lifts the leading edge by " +
              std::to_string(upPeak) + " dB");
    check(downPeak < -4.5 && downPeak >= -6.05,
          "transient designer: -6 dB of attack softens it by " +
              std::to_string(downPeak) + " dB");
    // THE HALF-WAVE RECTIFICATION, measured. Through the decay the attack
    // differential is negative, so an unrectified version would quietly cut the
    // body every time the attack was boosted — an attack control that is also
    // a sustain control, which nobody would think to check for.
    check(std::abs(upBody) < 0.05,
          "transient designer: and leaves the body alone (" +
              std::to_string(upBody) + " dB)");
  }

  // --- (c) sustain moves the body and only the body ------------------------
  {
    const AudioBuffer hit = transientHit(frames, onset, 0.5, 0.08);
    const AudioBuffer flat = runTransient(tdParams(0.0, 0.0, 20.0, 250.0, 2.0), hit);
    const AudioBuffer up = runTransient(tdParams(0.0, 6.0, 20.0, 250.0, 2.0), hit);
    const AudioBuffer down = runTransient(tdParams(0.0, -6.0, 20.0, 250.0, 2.0), hit);

    const double upBody = bodyDb(up) - bodyDb(flat);
    const double downBody = bodyDb(down) - bodyDb(flat);
    const double upPeak = peakDb(up) - peakDb(flat);
    std::printf("       transient sustain +/-6 dB: body %+.2f / %+.2f dB, peak "
                "%+.3f dB\n", upBody, downBody, upPeak);

    check(upBody > 2.0, "transient designer: +6 dB of sustain lifts the body by " +
                            std::to_string(upBody) + " dB");
    check(downBody < -2.0, "transient designer: -6 dB tightens it by " +
                               std::to_string(downBody) + " dB");
    check(std::abs(upPeak) < 0.3,
          "transient designer: and leaves the leading edge alone (" +
              std::to_string(upPeak) + " dB)");
  }

  // --- (d) the shaping is LEVEL-INDEPENDENT --------------------------------
  //
  // The one property that separates this node from a compressor, and the reason
  // it has no threshold. The differentials are RATIOS of envelopes, so a hit
  // 40 dB quieter is shaped identically. A compressor cannot do that at any
  // setting — which is what the reference below establishes, so the assertion
  // is about this node rather than about arithmetic in general.
  {
    const ParamSet shape = tdParams(6.0, -6.0, 20.0, 250.0, 2.0);
    // Both halves of the shaping, against each hit's OWN null render, so the
    // hits' identical intrinsic peak-to-body ratio cannot carry the comparison.
    auto shapingDb = [&](double amp, double& bodyOut) {
      const AudioBuffer hit = transientHit(frames, onset, amp, 0.08);
      const AudioBuffer flat = runTransient(tdParams(0.0, 0.0, 20.0, 250.0, 2.0), hit);
      const AudioBuffer out = runTransient(shape, hit);
      bodyOut = bodyDb(out) - bodyDb(flat);
      return peakDb(out) - peakDb(flat);
    };
    double loudBody = 0.0, quietBody = 0.0;
    const double loudPeak = shapingDb(0.5, loudBody);
    const double quietPeak = shapingDb(0.005, quietBody); // 40 dB down

    // The reference: a compressor, measured the same way. Its threshold is a
    // fixed level, so the quiet hit sits entirely underneath it and comes back
    // untouched while the loud one is squashed. Without this the assertion
    // above would be satisfied by any processor that happened to be linear.
    auto compressorPeakDb = [&](double amp) {
      const AudioBuffer hit = transientHit(frames, onset, amp, 0.08);
      auto chain = compressorChain(compressorParams(-18.0, 10.0, 0.0, 1.0, 10.0, 2.0),
                                   1, frames);
      AudioBuffer out;
      chain->render(hit, out, 1024);
      return peakDb(out) - peakDb(hit);
    };
    const double compLoud = compressorPeakDb(0.5);
    const double compQuiet = compressorPeakDb(0.005);

    std::printf("       transient level independence: peak %+.4f / %+.4f dB and "
                "body %+.4f / %+.4f dB at -6 and -46 dBFS (deltas %.1e, %.1e); a "
                "compressor's peak change differs by %.2f dB across the same "
                "two hits\n", loudPeak, quietPeak, loudBody, quietBody,
                std::abs(loudPeak - quietPeak), std::abs(loudBody - quietBody),
                std::abs(compLoud - compQuiet));

    check(std::abs(loudPeak - quietPeak) < 0.02 &&
              std::abs(loudBody - quietBody) < 0.02,
          "transient designer: a hit 40 dB quieter is shaped identically (peak " +
              std::to_string(std::abs(loudPeak - quietPeak)) + " dB, body " +
              std::to_string(std::abs(loudBody - quietBody)) + " dB apart)");
    check(std::abs(compLoud - compQuiet) > 3.0,
          "and a compressor genuinely treats them differently, so the claim is "
          "not something every processor satisfies (" +
              std::to_string(std::abs(compLoud - compQuiet)) + " dB)");
  }

  // --- (e) steady state is left alone, and where it is not ------------------
  //
  // A tone that never changes has no transient and no decay, so both
  // differentials should be zero. They are not exactly zero, and the reason is
  // worth stating rather than hiding: the detector is BROADBAND, so once a
  // waveform's period approaches the fast envelope's release the envelope
  // follows the WAVEFORM instead of the envelope. The number below is measured
  // at both ends and recorded in findings.md.
  {
    auto steadyChangeDb = [&](double hz) {
      const int64_t total = 8 * 65536;
      AudioBuffer tone(1, total);
      for (int64_t i = 0; i < total; ++i)
        tone.channel(0)[i] = 0.3 * std::sin(2.0 * M_PI * hz *
                                            static_cast<double>(i) / kCanonicalSampleRate);
      const AudioBuffer out = runTransient(tdParams(12.0, 12.0, 20.0, 250.0, 0.0), tone);
      return windowRmsDb(out, 0, total / 2, total) -
             windowRmsDb(tone, 0, total / 2, total);
    };
    const double at1k = steadyChangeDb(1000.0);
    const double at60 = steadyChangeDb(60.0);
    std::printf("       transient designer on a steady tone, both controls at "
                "+12 dB: 1 kHz %+.3f dB, 60 Hz %+.2f dB\n", at1k, at60);
    check(std::abs(at1k) < 0.15,
          "transient designer: a steady 1 kHz tone is essentially untouched (" +
              std::to_string(at1k) + " dB)");
    // Asserted as a RATIO to the 1 kHz figure rather than as an absolute, so
    // the check is about the shape of the limit — it grows as the period
    // approaches the fast envelope's release — rather than about a number that
    // would have to be re-tuned every time a constant moved. This is the node's
    // real limit and the reason bass-led material belongs in
    // `multiband_compressor`.
    check(at60 > 5.0 * std::abs(at1k) && at60 < 2.0,
          "transient designer: and a steady 60 Hz tone is NOT, by " +
              std::to_string(at60) + " dB against " + std::to_string(at1k) +
              " at 1 kHz (the documented broadband limit)");
  }

  // --- (f) lookahead is exact, and it actually anticipates ------------------
  //
  // Latency has to be the lookahead and nothing else. Then the point of paying
  // it: the gain is computed from the UNDELAYED signal, so the boost begins
  // exactly `lookahead` samples before the hit it belongs to. Measured on a
  // quiet bed running under the hit, because a boost applied to the silence
  // before an onset multiplies zero and is invisible.
  {
    auto latencyOf = [](double lookMs, double attackDb) {
      Chain chain;
      Chain::SlotConfig cfg;
      cfg.nodeId = "td";
      chain.add(std::make_unique<TransientDesigner>(),
                tdParams(attackDb, 0.0, 20.0, 250.0, lookMs), cfg);
      chain.prepare(makeSpec(1, 1024, 4096));
      return chain.totalLatencyFrames();
    };

    // The node declares the oversampler's round trip plus the lookahead. The
    // round trip is a fixed integer, so the lookahead must show up in the
    // declared latency sample for sample on top of it.
    const int base = latencyOf(0.0, 6.0);
    for (double lookMs : {1.0, 2.5, 5.0}) {
      const int want = static_cast<int>(std::lround(lookMs * 0.001 * kCanonicalSampleRate));
      check(latencyOf(lookMs, 6.0) - base == want,
            "transient designer: " + std::to_string(lookMs) +
                " ms of lookahead adds exactly " + std::to_string(want) +
                " frames (got " + std::to_string(latencyOf(lookMs, 6.0) - base) + ")");
    }
    std::printf("       transient designer: oversampler round trip %d frames\n", base);
    check(base > 0, "transient designer: and the oversampler declares its own "
                    "round trip (" + std::to_string(base) + " frames)");

    // WITH NOTHING TO SHAPE THERE IS NO OVERSAMPLER, so the node is a bare
    // delay and declares only the lookahead. This is the same rule as a
    // bandless EQ: a node that does nothing declares nothing, and paying an
    // oversampler's round trip to multiply by one would be strictly worse than
    // not doing it.
    check(latencyOf(0.0, 0.0) == 0 && latencyOf(2.5, 0.0) == 240,
          "transient designer: with both controls at zero it declares only the "
          "lookahead (" + std::to_string(latencyOf(0.0, 0.0)) + ", " +
              std::to_string(latencyOf(2.5, 0.0)) + ")");

    // The hit sits 340 ms into the bed, not 85 ms. THE BED'S OWN ONSET IS A
    // TRANSIENT: with `attack_ms` at 20 the slow envelope needs about five time
    // constants to catch up, so measured too close to the start of the buffer
    // the window still carries 0.19 dB of the bed's own attack — which reads
    // exactly like lookahead and is nothing of the kind.
    const int64_t bedOnset = 32768;
    AudioBuffer bedded = transientHit(frames, bedOnset, 0.5, 0.08);
    for (int64_t i = 0; i < frames; ++i)
      bedded.channel(0)[i] += 0.005 * std::sin(2.0 * M_PI * 3000.0 *
                                               static_cast<double>(i) / kCanonicalSampleRate);

    auto preOnsetLiftDb = [&](double lookMs) {
      const AudioBuffer flat = runTransient(tdParams(0.0, 0.0, 20.0, 250.0, lookMs), bedded);
      const AudioBuffer out = runTransient(tdParams(12.0, 0.0, 20.0, 250.0, lookMs), bedded);
      return windowRmsDb(out, 0, bedOnset - 4 * ms, bedOnset) -
             windowRmsDb(flat, 0, bedOnset - 4 * ms, bedOnset);
    };
    const double none = preOnsetLiftDb(0.0);
    const double look4 = preOnsetLiftDb(4.0);
    std::printf("       transient lookahead: the 4 ms before the hit are lifted "
                "%+.3f dB with no lookahead and %+.2f dB with 4 ms\n", none, look4);
    // NOT ZERO WITH NO LOOKAHEAD, and the reason is the oversampler rather than
    // the node. Its halfband stages are symmetric linear-phase FIRs, so a step
    // arrives with as much pre-ring as post-ring and the detector sees the hit
    // beginning slightly before it does. That is inherent to linear-phase
    // oversampling — every oversampled plugin does it — and it is an order of
    // magnitude smaller than the effect being measured, which is the only claim
    // this needs to support.
    check(std::abs(none) < 1.0,
          "transient designer: with no lookahead only the oversampler's own "
          "pre-ring precedes the hit (" + std::to_string(none) + " dB)");
    check(look4 > 5.0 * std::abs(none) && look4 > 3.0,
          "transient designer: and 4 ms of lookahead starts the boost well "
          "before the hit lands (" + std::to_string(look4) + " dB against " +
              std::to_string(none) + ")");
  }

  // --- (g) attack_ms decides what counts as a transient --------------------
  //
  // A node that ignored the parameter and used its fixed reference for both
  // envelopes would pass everything above, because the differential would still
  // be large on a hit out of silence. What it could not do is CHANGE with the
  // setting: a longer attack window keeps the two envelopes apart for longer,
  // so the boost lasts longer.
  {
    const AudioBuffer hit = transientHit(frames, onset, 0.5, 0.08);
    auto boostedFramesFor = [&](double attackMs) {
      const AudioBuffer flat =
          runTransient(tdParams(0.0, 0.0, attackMs, 250.0, 0.0), hit);
      const AudioBuffer out =
          runTransient(tdParams(6.0, 0.0, attackMs, 250.0, 0.0), hit);
      int64_t count = 0;
      for (int64_t i = onset; i < onset + 400 * ms; ++i) {
        const double a = std::abs(out.channel(0)[i]);
        const double b = std::abs(flat.channel(0)[i]);
        if (b > 1e-9 && a / b > 1.02) ++count; // more than ~0.17 dB of lift
      }
      return count;
    };
    const int64_t shortWindow = boostedFramesFor(5.0);
    const int64_t longWindow = boostedFramesFor(80.0);
    std::printf("       transient attack_ms: 5 ms lifts %lld frames, 80 ms lifts "
                "%lld\n", (long long)shortWindow, (long long)longWindow);
    check(shortWindow > 0 && longWindow > 3 * shortWindow,
          "transient designer: a longer attack_ms holds the boost on for longer (" +
              std::to_string(shortWindow) + " vs " + std::to_string(longWindow) +
              " frames)");
  }

  // --- (h) a GHOST NOTE after an accent still gets its attack --------------
  //
  // The reason the fast and slow-attack envelopes are given the SAME release
  // coefficient, and the only place that choice is observable. On one hit in
  // isolation the two envelopes agree during the onset whatever their releases
  // are, so every check above passes with the coefficients mismatched —
  // measured, and the mutation survived the first version of this section.
  //
  // What breaks is the second hit. Give the slow-attack envelope a long
  // release and it is still sitting near the FIRST hit's peak when a quieter
  // one arrives 150 ms later; the differential goes negative, is rectified to
  // zero, and the ghost note gets no attack at all. Sharing the release is what
  // makes the shaping depend on the note being played rather than on the note
  // before it.
  {
    // A SHORT decay on both hits, so the accent has died away by the time the
    // ghost note arrives. Left ringing, the accent's own tail is louder than
    // the ghost note's onset and the measurement is of the wrong event: the
    // first version of this check used an 80 ms decay and read +2.1 dB on a
    // node that was working correctly.
    AudioBuffer two(1, frames);
    addHit(two, onset, 0.5, 0.03);
    const int64_t second = onset + 150 * ms;
    addHit(two, second, 0.15, 0.03); // 10 dB down, one slow-release time later

    const AudioBuffer flat = runTransient(tdParams(0.0, 0.0, 20.0, 250.0, 0.0), two);
    const AudioBuffer up = runTransient(tdParams(9.0, 0.0, 20.0, 250.0, 0.0), two);

    auto peakIn = [&](const AudioBuffer& b, int64_t from, int64_t to) {
      double pk = 0.0;
      for (int64_t i = from; i < to; ++i) pk = std::max(pk, std::abs(b.channel(0)[i]));
      return 20.0 * std::log10(std::max(pk, 1e-300));
    };
    const double firstLift = peakIn(up, onset, onset + 3 * ms) -
                             peakIn(flat, onset, onset + 3 * ms);
    const double secondLift = peakIn(up, second, second + 3 * ms) -
                              peakIn(flat, second, second + 3 * ms);
    std::printf("       transient ghost note: accent lifted %+.2f dB, the "
                "10 dB-quieter hit 150 ms later lifted %+.2f dB\n",
                firstLift, secondLift);
    check(firstLift > 6.0,
          "transient designer: the accent gets its attack (" +
              std::to_string(firstLift) + " dB)");
    check(secondLift > 5.0,
          "transient designer: and so does a quieter hit inside its decay (" +
              std::to_string(secondLift) + " dB)");
  }

  // --- (i) the sustain differential is non-negative by construction --------
  //
  // `slowRelease` shares `fast`'s attack coefficient and releases more slowly,
  // so it dominates `fast` at every sample and the sustain differential can
  // never go below zero. The node therefore does NOT rectify it, and this is
  // the assertion that keeps that from becoming a latent inversion the day
  // someone gives the two detectors different attack times: a positive
  // `sustain_db` must never attenuate ANY sample, anywhere, by any amount.
  // Checked with no tolerance, because a gain of at least one cannot round a
  // magnitude downwards.
  {
    AudioBuffer hard = testSignal(1, frames, 909);
    addHit(hard, 12000, 0.6, 0.05);
    addHit(hard, 20000, 0.2, 0.4);
    const AudioBuffer flat = runTransient(tdParams(0.0, 0.0, 20.0, 250.0, 0.0), hard);
    const AudioBuffer lifted = runTransient(tdParams(0.0, 6.0, 20.0, 250.0, 0.0), hard);

    // MEASURED IN WINDOWS, NOT PER SAMPLE, and the reason is worth stating
    // because the per-sample version is what a first draft writes and it is
    // wrong for a reason that has nothing to do with the property. The gain is
    // applied at the OVERSAMPLED rate and the result is then decimated, so the
    // output sample at index i is a filtered combination of oversampled samples
    // either side of it — and a linear filter of a signal that grew everywhere
    // can still produce individual samples that shrank. The claim lives in the
    // envelope, so it is checked in the envelope.
    const int64_t win = 512;
    int64_t below = 0, windows = 0;
    double worstDb = 0.0;
    for (int64_t from = 0; from + win <= frames; from += win) {
      const double a = windowRmsDb(lifted, 0, from, from + win);
      const double b = windowRmsDb(flat, 0, from, from + win);
      if (b < -200.0) continue; // nothing there to compare
      ++windows;
      if (a < b) ++below;
      worstDb = std::min(worstDb, a - b);
    }
    std::printf("       transient sustain sign: %lld windows, worst change "
                "%+.4f dB\n", (long long)windows, worstDb);
    check(windows > 50 && below == 0,
          "transient designer: a positive sustain_db never attenuates (" +
              std::to_string(below) + " of " + std::to_string(windows) +
              " windows went the wrong way, worst " + std::to_string(worstDb) +
              " dB)");
  }

  // --- (j) stereo linking holds the image still ----------------------------
  //
  // Every check above is mono, and mono cannot see linking at all: with one
  // channel the running maximum over channels is that channel. Measured —
  // inverting the link from a maximum to a minimum survived the whole suite
  // until this existed.
  //
  // A hit in the LEFT channel only, over a bed in both. Linked, the right
  // channel is shaped by the left's transient and the two move together;
  // unlinked, only the left moves and the image jumps sideways on every hit.
  {
    AudioBuffer stereo(2, frames);
    for (int64_t i = 0; i < frames; ++i) {
      const double t = static_cast<double>(i) / kCanonicalSampleRate;
      const double bed = 0.05 * std::sin(2.0 * M_PI * 900.0 * t);
      stereo.channel(0)[i] = bed;
      stereo.channel(1)[i] = bed;
    }
    for (int64_t i = onset; i < frames; ++i) {
      const double t = static_cast<double>(i - onset) / kCanonicalSampleRate;
      stereo.channel(0)[i] += 0.6 * std::exp(-t / 0.03) *
                              std::cos(2.0 * M_PI * 900.0 * t);
    }

    auto rightLiftDb = [&](double link) {
      const AudioBuffer flat =
          runTransient(tdParams(0.0, 0.0, 20.0, 250.0, 0.0, link), stereo);
      const AudioBuffer out =
          runTransient(tdParams(9.0, 0.0, 20.0, 250.0, 0.0, link), stereo);
      return windowRmsDb(out, 1, onset, onset + 3 * ms) -
             windowRmsDb(flat, 1, onset, onset + 3 * ms);
    };
    const double linked = rightLiftDb(1.0);
    const double unlinked = rightLiftDb(0.0);
    std::printf("       transient stereo link: a left-only hit lifts the RIGHT "
                "channel %+.2f dB linked, %+.3f dB unlinked\n", linked, unlinked);
    check(linked > 4.0,
          "transient designer: linked, a hit in one channel shapes both (" +
              std::to_string(linked) + " dB)");
    check(std::abs(unlinked) < 0.2,
          "transient designer: and unlinked it does not (" +
              std::to_string(unlinked) + " dB)");
  }
}

// ------------------------------------------------- shared character helpers --

AudioBuffer runCharacter(const std::function<std::unique_ptr<Node>()>& make,
                         const ParamSet& p, const AudioBuffer& in, int block,
                         int maxBlock) {
  Chain chain;
  Chain::SlotConfig cfg;
  cfg.nodeId = "chr";
  chain.add(make(), p, cfg);
  chain.prepare(makeSpec(in.channels(), maxBlock, in.frames()));
  AudioBuffer out;
  chain.render(in, out, block);
  return out;
}

// Aliasing-to-signal ratio for any node, by the same coherent-bin method as
// test 11 but parameterised over the node rather than hard-wired to `drive`.
//
// The analysis window keeps 32768 samples of live tone AFTER it, which test 11
// does not. Without that the tail of the window is the response decaying into
// the zero padding the chain feeds past the end of the input, and that decay's
// rectangular-window skirts land in exactly the bins this measurement reads —
// findings.md's "the analysis window sat at the end of the stimulus", which
// cost the measurement suite two nodes' worth of imaginary distortion.
struct CharacterAsr {
  double inBandDb = 0.0;
  double fullBandDb = 0.0;
  bool collisionFree = true;
};

CharacterAsr measureCharacterAsr(const std::function<std::unique_ptr<Node>()>& make,
                                 const ParamSet& params, double amplitude,
                                 double targetHz) {
  constexpr size_t kN = 32768;
  constexpr int64_t kSettle = 32768;
  constexpr int64_t kPostRoll = 32768;

  int bin = static_cast<int>(std::lround(targetHz * static_cast<double>(kN) /
                                         kCanonicalSampleRate));
  if (bin % 2 == 0) ++bin; // odd, so no folded harmonic can hide in a real one
  const double f0 = static_cast<double>(bin) * kCanonicalSampleRate /
                    static_cast<double>(kN);
  const int64_t total = kSettle + static_cast<int64_t>(kN) + kPostRoll;

  AudioBuffer in(1, total);
  for (int64_t i = 0; i < total; ++i)
    in.channel(0)[i] = amplitude * std::sin(2.0 * M_PI * f0 *
                                            static_cast<double>(i) / kCanonicalSampleRate);

  const AudioBuffer out = runCharacter(make, params, in, 1024, 1024);
  const auto power = powerSpectrum(out.channel(0) + kSettle, kN);

  std::vector<int> role(kN / 2 + 1, 0);
  CharacterAsr result;
  for (int m = 1; m <= 400; ++m) {
    size_t b = (static_cast<size_t>(m) * static_cast<size_t>(bin)) % kN;
    if (b > kN / 2) b = kN - b;
    if (b == 0) continue;
    const bool legitimate =
        (static_cast<double>(m) * f0) < (kCanonicalSampleRate * 0.5);
    const int want = legitimate ? 1 : 2;
    if (role[b] != 0 && role[b] != want) result.collisionFree = false;
    if (role[b] == 0 || want == 1) role[b] = want;
  }

  const double hzPerBin = kCanonicalSampleRate / static_cast<double>(kN);
  double signalPower = 0.0, aliasFull = 0.0, aliasInBand = 0.0;
  for (size_t b = 1; b <= kN / 2; ++b) {
    const double hz = static_cast<double>(b) * hzPerBin;
    if (role[b] == 1) {
      signalPower += power[b];
    } else if (role[b] == 2) {
      aliasFull += power[b];
      if (hz < 20000.0) aliasInBand += power[b];
    }
  }
  const double denom = std::max(signalPower, 1e-300);
  result.inBandDb = 10.0 * std::log10(std::max(aliasInBand, 1e-300) / denom);
  result.fullBandDb = 10.0 * std::log10(std::max(aliasFull, 1e-300) / denom);
  return result;
}

// One coherently-sampled bin, complex, so magnitude AND phase are available.
std::complex<double> binDft(const double* x, size_t n, int k) {
  std::complex<double> acc(0.0, 0.0);
  for (size_t i = 0; i < n; ++i) {
    const double th = -2.0 * M_PI * static_cast<double>(k) *
                      static_cast<double>(i) / static_cast<double>(n);
    acc += x[i] * std::complex<double>(std::cos(th), std::sin(th));
  }
  return acc;
}

// Harmonic amplitudes of a node's response to a coherently-sampled tone,
// relative to the fundamental. Index 0 is the fundamental itself.
std::vector<double> harmonicRatios(const std::function<std::unique_ptr<Node>()>& make,
                                   const ParamSet& params, double amplitude,
                                   double targetHz, int highest) {
  constexpr size_t kN = 32768;
  constexpr int64_t kSettle = 32768;
  constexpr int64_t kPostRoll = 32768;
  int bin = static_cast<int>(std::lround(targetHz * static_cast<double>(kN) /
                                         kCanonicalSampleRate));
  if (bin % 2 == 0) ++bin;
  const double f0 = static_cast<double>(bin) * kCanonicalSampleRate /
                    static_cast<double>(kN);
  const int64_t total = kSettle + static_cast<int64_t>(kN) + kPostRoll;

  AudioBuffer in(1, total);
  for (int64_t i = 0; i < total; ++i)
    in.channel(0)[i] = amplitude * std::sin(2.0 * M_PI * f0 *
                                            static_cast<double>(i) / kCanonicalSampleRate);

  const AudioBuffer out = runCharacter(make, params, in, 1024, 1024);
  const auto power = powerSpectrum(out.channel(0) + kSettle, kN);

  std::vector<double> ratios;
  const double fundamental = power[static_cast<size_t>(bin)];
  for (int m = 1; m <= highest; ++m) {
    const size_t b = static_cast<size_t>(m * bin);
    if (b > kN / 2 || fundamental <= 0.0) { ratios.push_back(0.0); continue; }
    ratios.push_back(std::sqrt(power[b] / fundamental));
  }
  return ratios;
}

// 27. Tape correctness.
//
//     Tests 1-9 pass for a "tape" node that is a gentle lowpass and nothing
//     else. These check the four mechanisms the node claims, each against the
//     arithmetic that motivated it rather than against the code that
//     implements it:
//
//       (a) the saturation curve's harmonic series, longhand from the quintic
//       (b) bias makes the curve asymmetric and nothing else does
//       (c) head bump is the bell it says it is
//       (d) gap loss lands where the sinc's -3 dB point says, at every speed,
//           and the in-band realisation matches the analog prototype rather
//           than the corner
//       (e) wow and flutter are bit-identical under any block partition, and
//           are actually doing something while that is asserted
//       (f) the achieved speed deviation is the requested one
void testTape() {
  auto tape = [] { return std::make_unique<Tape>(); };

  // --- (a) the saturation curve's harmonic series --------------------------
  //
  // f(u) = u - 2u^3/3 + u^5/5 driven by u = a*sin(theta). Expanding with
  //
  //     sin^3 = (3 sin - sin3)/4        sin^5 = (10 sin - 5 sin3 + sin5)/16
  //
  // gives, longhand and with nothing imported from the shaper:
  //
  //     h1 = a - a^3/2 + a^5/8
  //     h3 = a^3/6 - a^5/16
  //     h5 = a^5/80
  //
  // At a = 1 those are 0.625, 0.1041667 and 0.0125, so h3/h1 is exactly 1/6
  // and h5/h1 exactly 1/50. Two amplitudes, because a single one is satisfied
  // by any curve with the right value there and the whole claim is about the
  // SHAPE.
  //
  // The head bump is off and the speed is 30 ips for this probe, so the two
  // linear stages cannot colour the harmonics. Gap loss at 30 ips still takes
  // 0.02 dB off the fifth harmonic, which is inside the tolerance below.
  {
    std::printf("       tape harmonics:");
    const double amps[] = {1.0, 0.5};
    for (double a : amps) {
      ParamSet p;
      p.set("drive_db", 0.0).set("output_trim_db", 0.0).set("bias", 0.0)
          .set("head_bump_db", 0.0).set("speed_ips", 30.0)
          .set("wow_percent", 0.0).set("flutter_percent", 0.0);

      // 7, not 6: index m-1 is harmonic m, so the seventh needs seven entries.
      const auto r = harmonicRatios(tape, p, a, 997.0, 7);
      const double a3 = a * a * a, a5 = a3 * a * a;
      const double h1 = a - a3 / 2.0 + a5 / 8.0;
      const double h3 = a3 / 6.0 - a5 / 16.0;
      const double h5 = a5 / 80.0;
      const double want3 = h3 / h1, want5 = h5 / h1;

      std::printf("  a=%.2f h3/h1 %.6f (want %.6f) h5/h1 %.6f (want %.6f)",
                  a, r[2], want3, r[4], want5);
      check(std::abs(20.0 * std::log10(r[2] / want3)) < 0.03,
            "tape: third harmonic at a=" + std::to_string(a) + " is " +
                std::to_string(r[2]) + ", the quintic says " + std::to_string(want3));
      check(std::abs(20.0 * std::log10(r[4] / want5)) < 0.10,
            "tape: fifth harmonic at a=" + std::to_string(a) + " is " +
                std::to_string(r[4]) + ", the quintic says " + std::to_string(want5));
      // The curve is a polynomial of degree five, so there is no seventh.
      check(r[6] < 1e-5,
            "tape: no seventh harmonic at a=" + std::to_string(a) + " (" +
                std::to_string(r[6]) + ")");
    }
    std::printf("\n");
  }

  // --- (b) bias is the only source of asymmetry ----------------------------
  //
  // An odd curve produces odd harmonics only. Offsetting the operating point
  // makes the two half-cycles traverse different curvature, which is even
  // order. Both directions are asserted: without bias the second harmonic must
  // be absent, or the "bias adds even harmonics" claim is being read off a
  // node that produces them anyway.
  {
    auto second = [&](double bias) {
      ParamSet p;
      p.set("drive_db", 0.0).set("output_trim_db", 0.0).set("bias", bias)
          .set("head_bump_db", 0.0).set("speed_ips", 30.0);
      return harmonicRatios(tape, p, 0.7, 997.0, 3)[1];
    };
    const double none = second(0.0), biased = second(0.3);
    std::printf("       tape bias: h2/h1 %.2e at bias 0, %.4f at bias 0.3\n",
                none, biased);
    check(20.0 * std::log10(std::max(none, 1e-300)) < -140.0,
          "tape: a symmetric curve produces no second harmonic (" +
              std::to_string(20.0 * std::log10(std::max(none, 1e-300))) + " dB)");
    check(biased > 0.02,
          "tape: bias produces a second harmonic (" + std::to_string(biased) + ")");
  }

  // --- (c) the head bump is the bell it claims -----------------------------
  //
  // Measured against a NULL RENDER — the same node with the bump at 0 dB —
  // rather than against the input, so the saturation and the gap loss cancel
  // exactly instead of being subtracted approximately.
  {
    const double snapped = snapHz(52.0, 16384);
    auto atBump = [&](double bumpDb) {
      ParamSet p;
      p.set("drive_db", 0.0).set("output_trim_db", 0.0)
          .set("head_bump_db", bumpDb).set("head_bump_hz", snapped)
          .set("head_bump_q", 0.8).set("speed_ips", 15.0);
      return measureNodeGainDb(tape, p, snapped);
    };
    const double flat = atBump(0.0);
    const double lifted = atBump(3.0);
    // An octave above the bump, where a Q 0.8 bell has fallen a long way. The
    // point of the second probe is that "the bump is 3 dB" is satisfied by a
    // broadband 3 dB lift, and a broadband lift is not a head bump.
    const double snappedUp = snapHz(4.0 * snapped, 16384);
    auto atOctaves = [&](double bumpDb) {
      ParamSet p;
      p.set("drive_db", 0.0).set("output_trim_db", 0.0)
          .set("head_bump_db", bumpDb).set("head_bump_hz", snapped)
          .set("head_bump_q", 0.8).set("speed_ips", 15.0);
      return measureNodeGainDb(tape, p, snappedUp);
    };
    const double away = atOctaves(3.0) - atOctaves(0.0);
    std::printf("       tape head bump: %+.4f dB at %.1f Hz, %+.4f dB two "
                "octaves up\n", lifted - flat, snapped, away);
    check(std::abs((lifted - flat) - 3.0) < 0.02,
          "tape: head bump measures " + std::to_string(lifted - flat) +
              " dB at its centre, asked for 3.0");
    check(away < 0.5,
          "tape: the bump is local, not a broadband lift (" +
              std::to_string(away) + " dB two octaves up)");
  }

  // --- (d) gap loss --------------------------------------------------------
  //
  // The physical claim is |H| = |sinc(pi g f / v)|, realised as a one-pole at
  // the sinc's own -3 dB point, x = 1.3915576:
  //
  //     f_-3dB = 1.3915576 * v / (pi * g)
  //
  // Three things are checked, and they are different claims:
  //   1. the corner moves with speed exactly as v does
  //   2. the realisation matches the ANALOG one-pole across the audio band —
  //      this is what dsp/AnalogOnePole.h exists for
  //   3. how far the one-pole is from the sinc it stands in for, which is a
  //      modelling error rather than an implementation one, and is bounded so
  //      it cannot grow unnoticed
  {
    const double gapUm = 6.0;
    auto corner = [&](double ips) {
      return 1.3915576 * (ips * 0.0254) / (M_PI * gapUm * 1e-6);
    };

    auto normalisedResponse = [&](double ips, const std::vector<double>& freqs) {
      ParamSet p;
      // -20 dB in, so the shaper is linear to a part in 10^4 and the reading
      // is the filters and nothing else.
      p.set("drive_db", -20.0).set("output_trim_db", 0.0)
          .set("head_bump_db", 0.0).set("speed_ips", ips).set("gap_um", gapUm);
      const double ref = measureNodeGainDb(tape, p, snapHz(100.0, 16384));
      std::vector<double> out;
      for (double f : freqs) out.push_back(measureNodeGainDb(tape, p, f) - ref);
      return out;
    };

    // 1. the corner tracks speed. Only speeds whose corner lands INSIDE the
    //    audio band are probed this way: dsp/AnalogOnePole.h pins the
    //    realisation to the analog prototype at 20 kHz, so above that the
    //    digital filter falls away faster and its own -3 dB point is not where
    //    the sinc's is. Measured at 15 ips it crosses at 25.2 kHz against the
    //    sinc's 28.1 — which changes nothing anyone can hear, and asserting
    //    the formula there would be asserting a number about ultrasound.
    std::printf("       tape gap loss corner:");
    std::vector<double> corners;
    for (double ips : {3.75, 7.5}) {
      const double fc = corner(ips);
      // Search for the -3 dB crossing by bisection on the measured response.
      double lo = fc * 0.5, hi = fc * 2.0;
      for (int it = 0; it < 14; ++it) {
        const double mid = snapHz(0.5 * (lo + hi), 16384);
        const double db = normalisedResponse(ips, {mid})[0];
        if (db > -3.0103) lo = mid; else hi = mid;
      }
      const double measured = 0.5 * (lo + hi);
      corners.push_back(measured);
      std::printf("  %.2f ips -> %.0f Hz (formula %.0f)", ips, measured, fc);
      check(std::abs(measured / fc - 1.0) < 0.005,
            "tape: gap loss at " + std::to_string(ips) + " ips crosses -3 dB at " +
                std::to_string(measured) + " Hz, the sinc says " + std::to_string(fc));
    }
    // The physical claim underneath the formula: every wavelength halves when
    // the speed halves, so the corner is proportional to speed and to nothing
    // else. A constant corner with a speed-shaped fudge in front of it would
    // pass the two absolute checks above and fail this one.
    std::printf("  ratio %.4f", corners[1] / corners[0]);
    check(std::abs(corners[1] / corners[0] - 2.0) < 0.01,
          "tape: doubling the speed doubles the gap-loss corner (ratio " +
              std::to_string(corners[1] / corners[0]) + ")");
    std::printf("\n");

    // 2. the realisation against the analog prototype, and against what
    //    prewarping at the corner would have given. 15 ips puts the corner at
    //    28.1 kHz, above the band, which is the only place the two differ.
    {
      const double ips = 15.0, fc = corner(ips);
      const std::vector<double> freqs = {
          snapHz(500.0, 16384),  snapHz(2000.0, 16384), snapHz(5000.0, 16384),
          snapHz(10000.0, 16384), snapHz(16000.0, 16384), snapHz(20000.0, 16384)};
      const auto got = normalisedResponse(ips, freqs);

      double worst = 0.0, atTop = 0.0, textbookAtTop = 0.0;
      for (size_t i = 0; i < freqs.size(); ++i) {
        const double analog =
            -10.0 * std::log10(1.0 + (freqs[i] / fc) * (freqs[i] / fc));
        worst = std::max(worst, std::abs(got[i] - analog));
        if (i + 1 == freqs.size()) {
          atTop = got[i];
          // What the textbook bilinear transform would have produced:
          // g = tan(pi fc / fs), magnitude 1 / sqrt(1 + (tan(pi f / fs)/g)^2).
          const double g = std::tan(M_PI * fc / kCanonicalSampleRate);
          const double t = std::tan(M_PI * freqs[i] / kCanonicalSampleRate);
          textbookAtTop = -10.0 * std::log10(1.0 + (t / g) * (t / g));
        }
      }
      std::printf("       tape gap loss realisation: worst deviation from the "
                  "analog prototype %.4f dB; at 20 kHz %+.3f dB against "
                  "%+.3f dB prewarped at the corner\n",
                  worst, atTop, textbookAtTop);
      // 0.15 dB, and the shape of the error is the point: it is ZERO at
      // 20 kHz and at DC, both by construction, and bulges in between because
      // the bilinear map's frequency warping cannot be cancelled at more than
      // one frequency by a filter with one degree of freedom. Putting the
      // matched point at the top of the band puts the error where the pole
      // does least. It is also four times smaller than the one-pole's own
      // departure from the sinc it stands in for, measured below, so it is not
      // the dominant modelling error.
      check(worst < 0.15,
            "tape: the realised gap loss matches the analog one-pole to " +
                std::to_string(worst) + " dB across 500 Hz-20 kHz");
      check(std::abs(atTop - textbookAtTop) > 0.35,
            "tape: and it is NOT what prewarping at the corner gives (" +
                std::to_string(std::abs(atTop - textbookAtTop)) +
                " dB apart at 20 kHz)");
    }

    // 3. how far a one-pole is from the sinc it stands in for. Bounded, not
    //    asserted to be zero: it is a modelling choice, and the honest thing
    //    is to know its size and notice if it changes.
    {
      std::printf("       tape one-pole vs sinc gap loss, worst over "
                  "500 Hz-20 kHz:");
      for (double ips : {7.5, 15.0}) {
        const double v = ips * 0.0254;
        const std::vector<double> freqs = {
            snapHz(500.0, 16384),   snapHz(2000.0, 16384),
            snapHz(5000.0, 16384),  snapHz(10000.0, 16384),
            snapHz(16000.0, 16384), snapHz(20000.0, 16384)};
        const auto got = normalisedResponse(ips, freqs);
        double worst = 0.0;
        for (size_t i = 0; i < freqs.size(); ++i) {
          const double x = M_PI * gapUm * 1e-6 * freqs[i] / v;
          const double sinc = std::abs(std::sin(x) / x);
          worst = std::max(worst, std::abs(got[i] - 20.0 * std::log10(sinc)));
        }
        std::printf("  %.1f ips %.2f dB", ips, worst);
        check(worst < 2.5,
              "tape: the one-pole stands in for the sinc to within " +
                  std::to_string(worst) + " dB at " + std::to_string(ips) + " ips");
      }
      std::printf("\n");
    }
  }

  // --- (e) wow and flutter under any block partition -----------------------
  //
  // THE PROPERTY MOST LIKELY TO BE QUIETLY BROKEN, and the one an LFO
  // accumulator would break silently: `phase += rate/fs` gives the right sound
  // and a different answer for every block size, because floating-point
  // addition is not associative.
  //
  // Two halves, and the second is what stops the first from being decorative.
  // A node with the modulation switched off is trivially block-size invariant,
  // so the same stimulus is first shown to be MODULATED — a coherently sampled
  // carrier loses most of its energy out of its own bin — and only then
  // compared across partitions.
  {
    constexpr size_t kN = 32768;
    constexpr int64_t kSettle = 32768;
    // 15 kHz rather than something midrange: the phase deviation a given
    // speed error produces is proportional to frequency, so the top of the
    // band is where a transport fault is most visible and where a check that
    // the modulation is switched ON has the most to work with.
    const int carrierBin = 5121; // 15003.9 Hz at 96 kHz, odd
    const double f0 = static_cast<double>(carrierBin) * kCanonicalSampleRate /
                      static_cast<double>(kN);
    const int64_t total = kSettle + static_cast<int64_t>(kN);

    AudioBuffer in(2, total);
    for (int c = 0; c < 2; ++c)
      for (int64_t i = 0; i < total; ++i)
        in.channel(c)[i] = 0.3 * std::sin(2.0 * M_PI * f0 *
                                          static_cast<double>(i) / kCanonicalSampleRate);

    auto carrierFraction = [&](const ParamSet& p) {
      const AudioBuffer out = runCharacter(tape, p, in, 1024, 4096);
      const auto pw = powerSpectrum(out.channel(0) + kSettle, kN);
      double total2 = 0.0;
      for (size_t b = 1; b <= kN / 2; ++b) total2 += pw[b];
      return pw[static_cast<size_t>(carrierBin)] / std::max(total2, 1e-300);
    };

    ParamSet still;
    still.set("drive_db", -20.0).set("head_bump_db", 0.0).set("speed_ips", 30.0)
        .set("wow_percent", 0.0).set("flutter_percent", 0.0);
    ParamSet moving;
    moving.set("drive_db", -20.0).set("head_bump_db", 0.0).set("speed_ips", 30.0)
        .set("wow_percent", 0.1).set("wow_rate_hz", 1.0)
        .set("flutter_percent", 0.05).set("flutter_rate_hz", 14.0);

    const double fStill = carrierFraction(still);
    const double fMoving = carrierFraction(moving);
    std::printf("       tape transport: carrier holds %.4f of its energy with a "
                "perfect transport, %.4f with wow and flutter\n", fStill, fMoving);
    check(fStill > 0.999,
          "tape: a perfect transport leaves the carrier intact (" +
              std::to_string(fStill) + ")");
    check(fMoving < 0.5,
          "tape: wow and flutter smear the carrier, so the invariance check "
          "below is not vacuous (" + std::to_string(fMoving) + ")");

    // Block sizes chosen so none divides the wow knot spacing (48000 samples)
    // or the flutter's (3429): a partition boundary landing mid-knot is the
    // case an accumulator gets wrong.
    const AudioBuffer ref = runCharacter(tape, moving, in, 4096, 4096);
    for (int bs : {1, 3, 7, 511, 1024, 4096}) {
      const AudioBuffer got = runCharacter(tape, moving, in, bs, 4096);
      const double d = maxAbsDiff(ref, got, total);
      check(d == 0.0, "tape: wow and flutter are bit-identical at block size " +
                          std::to_string(bs) + " (diff=" + std::to_string(d) + ")");
    }

    // And the two channels must not share a drift. The PRNG is keyed on the
    // channel, so a key that dropped the channel would give a mono wobble on a
    // stereo tape machine, which is not what a transport does.
    {
      const AudioBuffer out = runCharacter(tape, moving, in, 1024, 4096);
      double worst = 0.0;
      for (int64_t i = kSettle; i < total; ++i)
        worst = std::max(worst,
                         std::abs(out.channel(0)[i] - out.channel(1)[i]));
      check(worst > 1e-3,
            "tape: the two channels drift independently (worst difference " +
                std::to_string(worst) + ")");
    }
  }

  // --- (f) the achieved speed deviation is the requested one ---------------
  //
  // `wow_percent` claims to be the PEAK instantaneous speed deviation. The
  // claim is derived rather than tuned: a delay of D(n) samples gives a
  // relative speed deviation of -D'(n), and the Catmull-Rom spline over knots
  // in [-1, 1] with spacing H has |D'| <= 3.5 A / H, so A = eps H / 3.5.
  //
  // Measured out of the AUDIO, by demodulating a 1 kHz probe tone, rather than
  // out of the node's own state — otherwise this checks that the code agrees
  // with itself.
  {
    // 1 kHz divides 96 kHz exactly, so a moving average of four tone periods
    // nulls the 2f0 demodulation product and every harmonic exactly.
    const double f0 = 1000.0;
    const int period = 96;
    const int smooth = period * 4;

    auto delayTrace = [&](double percent, double rateHz, bool flutter,
                          double seconds) {
      ParamSet p;
      p.set("drive_db", -20.0).set("output_trim_db", 0.0)
          .set("head_bump_db", 0.0).set("speed_ips", 30.0);
      if (flutter) {
        p.set("wow_percent", 0.0).set("flutter_percent", percent)
            .set("flutter_rate_hz", rateHz);
      } else {
        p.set("wow_percent", percent).set("wow_rate_hz", rateHz)
            .set("flutter_percent", 0.0);
      }

      const int64_t settle = 16384;
      const int64_t frames =
          static_cast<int64_t>(seconds * kCanonicalSampleRate);
      const int64_t total = settle + frames;
      AudioBuffer in(1, total);
      const double w = 2.0 * M_PI * f0 / kCanonicalSampleRate;
      for (int64_t i = 0; i < total; ++i)
        in.channel(0)[i] = 0.2 * std::sin(w * static_cast<double>(i));

      const AudioBuffer out = runCharacter(tape, p, in, 1024, 1024);

      std::vector<double> bufI(static_cast<size_t>(smooth), 0.0);
      std::vector<double> bufQ(static_cast<size_t>(smooth), 0.0);
      double sumI = 0.0, sumQ = 0.0;
      int pos = 0;
      std::vector<double> trace;
      trace.reserve(static_cast<size_t>(frames));
      double prev = 0.0, unwrapped = 0.0;
      bool have = false;
      for (int64_t i = 0; i < total; ++i) {
        const double th = w * static_cast<double>(i);
        const double vi = out.channel(0)[i] * 2.0 * std::cos(th);
        const double vq = -out.channel(0)[i] * 2.0 * std::sin(th);
        sumI += vi - bufI[static_cast<size_t>(pos)];
        sumQ += vq - bufQ[static_cast<size_t>(pos)];
        bufI[static_cast<size_t>(pos)] = vi;
        bufQ[static_cast<size_t>(pos)] = vq;
        pos = (pos + 1) % smooth;
        if (i < settle) continue;
        const double ph = std::atan2(sumI, sumQ);
        if (have) {
          double d = ph - prev;
          while (d > M_PI) d -= 2.0 * M_PI;
          while (d < -M_PI) d += 2.0 * M_PI;
          unwrapped += d;
        } else {
          unwrapped = ph;
          have = true;
        }
        prev = ph;
        // phase = 2*pi*f0*delay/fs, so delay in samples is phase * fs / (2 pi f0)
        trace.push_back(unwrapped * kCanonicalSampleRate / (2.0 * M_PI * f0));
      }
      double mean = 0.0;
      for (double v : trace) mean += v;
      mean /= static_cast<double>(trace.size());
      for (double& v : trace) v -= mean;
      return trace;
    };

    // Peak |value| and the RMS and peak of the sample-to-sample slope, the
    // slope taken as a central difference over `span` samples so the readout
    // is not dominated by the demodulator's own noise floor.
    auto stats = [](const std::vector<double>& trace, int span) {
      double peakAbs = 0.0, peakSlope = 0.0, sumSq = 0.0;
      int64_t n = 0;
      for (double v : trace) peakAbs = std::max(peakAbs, std::abs(v));
      for (size_t i = static_cast<size_t>(span);
           i + static_cast<size_t>(span) < trace.size(); ++i) {
        const double s = (trace[i + static_cast<size_t>(span)] -
                          trace[i - static_cast<size_t>(span)]) /
                         (2.0 * static_cast<double>(span));
        peakSlope = std::max(peakSlope, std::abs(s));
        sumSq += s * s;
        ++n;
      }
      return std::make_pair(peakSlope, std::sqrt(sumSq / static_cast<double>(n)));
    };

    // WOW. 0.1% peak at 1 Hz: H = 96000/2 = 48000 samples,
    // A = 0.001 * 48000 / 3.5 = 13.714 samples, and the spline may overshoot
    // its knots by 25%, so |D| <= 17.14.
    {
      const auto trace = delayTrace(0.1, 1.0, /*flutter=*/false, 20.0);
      const auto st = stats(trace, 480);
      double peakAbs = 0.0;
      for (double v : trace) peakAbs = std::max(peakAbs, std::abs(v));
      const double amp = 0.001 * 48000.0 / 3.5;
      std::printf("       tape wow at 0.1%% / 1 Hz: peak speed deviation "
                  "%.4f%% (bound 0.1000), rms %.4f%%, peak delay %.2f samples "
                  "(bound %.2f)\n",
                  st.first * 100.0, st.second * 100.0, peakAbs, 1.25 * amp);
      check(st.first <= 0.001,
            "tape: wow peak speed deviation " + std::to_string(st.first * 100.0) +
                "% never exceeds the 0.1% asked for");
      check(st.first > 0.00035,
            "tape: and it gets within a factor of three of it (" +
                std::to_string(st.first * 100.0) + "%)");
      check(peakAbs <= 1.25 * amp,
            "tape: the drift stays inside the spline's own overshoot bound (" +
                std::to_string(peakAbs) + " against " +
                std::to_string(1.25 * amp) + " samples)");
    }

    // THE DERIVATION IS LINEAR IN eps. Same rate means the same knot spacing
    // and therefore the same PRNG draws, so doubling `wow_percent` doubles the
    // drift — no statistics involved, and no dependence on which random
    // excursion happened to be the largest. This is the sharpest available
    // check that the amplitude is computed FROM the requested deviation rather
    // than from anything else that happens to grow with it.
    //
    // Read as a ratio of RMS rather than sample by sample. The two
    // configurations do not have the same base delay — it is sized from the
    // amplitude — so the fractional part the Lagrange interpolator reads at
    // differs, and the per-sample residual sits at 5.6e-4 of full scale for
    // that reason rather than because the scaling is wrong.
    {
      const auto a = stats(delayTrace(0.05, 1.0, false, 8.0), 480);
      const auto b = stats(delayTrace(0.10, 1.0, false, 8.0), 480);
      const double ratio = b.second / a.second;
      std::printf("       tape wow scaling: rms speed deviation %.5f%% at "
                  "0.05%% and %.5f%% at 0.10%%, ratio %.5f\n",
                  a.second * 100.0, b.second * 100.0, ratio);
      check(std::abs(ratio - 2.0) < 0.005,
            "tape: the drift is linear in wow_percent (ratio " +
                std::to_string(ratio) + ", expected 2)");
    }

    // FLUTTER, and the rate-invariance the derivation implies. A = eps H / 3.5
    // with H = fs / (2 rate) means the SLOPE — and therefore the speed
    // deviation — does not depend on the rate at all. Read as an RMS rather
    // than a peak, because an RMS over a few hundred knots converges and an
    // extreme value does not.
    {
      const auto slow = stats(delayTrace(0.05, 8.0, true, 10.0), 96);
      const auto fast = stats(delayTrace(0.05, 20.0, true, 10.0), 96);
      const double ratio = fast.second / slow.second;
      std::printf("       tape flutter at 0.05%%: rms speed deviation %.4f%% at "
                  "8 Hz and %.4f%% at 20 Hz, ratio %.3f\n",
                  slow.second * 100.0, fast.second * 100.0, ratio);
      check(slow.first <= 0.0005 && fast.first <= 0.0005,
            "tape: flutter peak speed deviation never exceeds the 0.05% asked "
            "for (" + std::to_string(slow.first * 100.0) + "%, " +
                std::to_string(fast.first * 100.0) + "%)");
      check(std::abs(ratio - 1.0) < 0.15,
            "tape: the speed deviation is set by wow/flutter_percent and not by "
            "the rate (ratio " + std::to_string(ratio) + ")");
    }
  }

  // --- (g) a perfect transport declares less latency and has no delay line --
  //
  // The same rule as a bandless EQ: a node's declared latency has to describe
  // what it actually does in the configuration it was prepared for. With both
  // depths at zero there is no line to read from, so declaring its base delay
  // would have the chain trim samples off the front of the render.
  {
    ParamSet still;
    still.set("wow_percent", 0.0).set("flutter_percent", 0.0);
    ParamSet moving;
    moving.set("wow_percent", 0.05).set("wow_rate_hz", 1.0)
        .set("flutter_percent", 0.03).set("flutter_rate_hz", 12.0);

    auto declared = [&](const ParamSet& p) {
      Tape node;
      node.prepare(makeSpec(2, 1024, 4096));
      node.setParams(p);
      return node.latencyFrames();
    };
    const int stillLat = declared(still), movingLat = declared(moving);
    // 1.25 * (0.0005*48000/3.5 + 0.0003*4000/3.5) = 9.00, so 9 + 2 = 11.
    std::printf("       tape latency: %d frames with a perfect transport, %d "
                "with wow and flutter\n", stillLat, movingLat);
    check(movingLat - stillLat == 11,
          "tape: the modulated path declares its base delay exactly (" +
              std::to_string(movingLat - stillLat) + ", expected 11)");
  }

  // --- (h) aliasing where the per-commit suite does not look ---------------
  //
  // engine/tests/measurements.cpp measures a node at its own DEFAULTS. The
  // defaults here are a symmetric curve at unity drive, so nothing in that
  // report exercises either the asymmetric path or a hard drive — and both
  // fold differently, the first because even harmonics land on different bins
  // and the second because there are more of them. The requirement is
  // -100 dB at 0 dBFS and it is not adjusted here.
  {
    struct Case { const char* label; double drive; double bias; };
    const Case cases[] = {
        {"defaults", 0.0, 0.0},
        {"+12 dB drive", 12.0, 0.0},
        {"+12 dB drive, bias 0.6", 12.0, 0.6},
        // The top of the accepted drive range, which is where the requirement
        // binds and where the 8x floor was chosen from.
        {"+18 dB drive", 18.0, 0.0},
        {"+18 dB drive, bias 1.0", 18.0, 1.0},
    };
    std::printf("       tape in-band ASR at 0 dBFS:");
    for (const Case& tc : cases) {
      ParamSet p;
      p.set("drive_db", tc.drive).set("output_trim_db", 0.0).set("bias", tc.bias)
          .set("wow_percent", 0.0).set("flutter_percent", 0.0);
      const CharacterAsr r = measureCharacterAsr(tape, p, 1.0, 19500.0);
      std::printf("  %s %.1f dB", tc.label, r.inBandDb);
      check(r.collisionFree, "tape ASR bin classification is unambiguous");
      check(r.inBandDb <= -100.0,
            std::string("tape in-band ASR at ") + tc.label + " = " +
                std::to_string(r.inBandDb) + " dB, requirement -100");
    }
    std::printf("\n");
  }

  // --- (i) the bounds that exist for the requirement actually bind ---------
  //
  // MUTATION TESTING ADDED THIS SECTION. Widening `drive_db`'s upper bound from
  // +18 dB to +30 — which walks the node straight out of the aliasing
  // requirement, since 8x measures -100.0 dB at +24 — survived the whole suite,
  // because every test above asks for a value INSIDE the bound. A defence
  // probed only from within its own range is not probed at all.
  //
  // The check is bit-exactness rather than a measurement: asking for more than
  // the bound must produce exactly what asking for the bound produces.
  {
    const AudioBuffer in = testSignal(2, 8192);
    auto render = [&](double drive, double factor) {
      ParamSet p;
      p.set("drive_db", drive).set("oversample_factor", factor)
          .set("wow_percent", 0.0).set("flutter_percent", 0.0);
      return runCharacter(tape, p, in, 512, 1024);
    };
    const double overDrive = maxAbsDiff(render(18.0, 8.0), render(30.0, 8.0), 8192);
    check(overDrive == 0.0,
          "tape: drive_db is clamped at +18 dB, which is where 8x still meets "
          "the aliasing requirement (diff=" + std::to_string(overDrive) + ")");
    const double underDrive = maxAbsDiff(render(-24.0, 8.0), render(-60.0, 8.0), 8192);
    check(underDrive == 0.0,
          "tape: drive_db is clamped at -24 dB (diff=" +
              std::to_string(underDrive) + ")");
    // And the oversampling floor, for the same reason: 4x measures -96 dB at
    // +12 dB of drive, so a recipe asking for it must get 8x.
    const double floorOs = maxAbsDiff(render(12.0, 4.0), render(12.0, 8.0), 8192);
    check(floorOs == 0.0,
          "tape: oversample_factor is floored at 8, where 4x does not meet the "
          "requirement (diff=" + std::to_string(floorOs) + ")");
  }
}

// 28. Console correctness.
//
//     A console stage that did nothing at all would pass tests 1-9 and sound
//     exactly right on any single instance, which is the whole difficulty with
//     this node. So every check here is against an arithmetic prediction, and
//     the last one is the only claim that matters: that what one instance does
//     is small and what ten do is ten times that.
void testConsole() {
  auto console = [] { return std::make_unique<Console>(); };

  // --- (a) distortion is set by headroom, and by the curve ----------------
  //
  // With u = a sin(theta) the quintic gives h3/h1 = (a^3/6 - a^5/16) /
  // (a - a^3/2 + a^5/8), and `headroom_db` sets a = 10^(-headroom/20) for a
  // full-scale input. Four headrooms, so the a^2 law is exercised rather than
  // one point on it: 24 dB predicts -63.6 dB of third harmonic and 6 dB
  // predicts -21.8, and nothing between them is a free parameter.
  {
    std::printf("       console d3 against headroom:");
    for (double headroom : {24.0, 18.0, 12.0, 6.0}) {
      ParamSet p;
      p.set("drive_db", 0.0).set("output_trim_db", 0.0)
          .set("headroom_db", headroom).set("coupling_hz", 12.0)
          .set("bandwidth_hz", 200000.0);
      const auto r = harmonicRatios(console, p, 1.0, 997.0, 5);

      const double a = std::pow(10.0, -headroom / 20.0);
      const double a3 = a * a * a, a5 = a3 * a * a;
      const double want = (a3 / 6.0 - a5 / 16.0) / (a - a3 / 2.0 + a5 / 8.0);
      std::printf("  %.0f dB -> %.2f dB (want %.2f)", headroom,
                  20.0 * std::log10(r[2]), 20.0 * std::log10(want));
      check(std::abs(20.0 * std::log10(r[2] / want)) < 0.1,
            "console: third harmonic at " + std::to_string(headroom) +
                " dB headroom is " + std::to_string(20.0 * std::log10(r[2])) +
                " dB, the quintic says " + std::to_string(20.0 * std::log10(want)));
      // Push-pull, so the transfer is symmetric and there is nothing even.
      check(20.0 * std::log10(std::max(r[1], 1e-300)) < -140.0,
            "console: no second harmonic at " + std::to_string(headroom) +
                " dB headroom (" +
                std::to_string(20.0 * std::log10(std::max(r[1], 1e-300))) + " dB)");
    }
    std::printf("\n");
  }

  // --- (b) the two poles are the analog prototypes ------------------------
  //
  // Both magnitude AND phase, because the magnitude is the part that does not
  // matter: at the defaults one stage is -0.46 dB at 20 kHz, which nobody
  // would hear, and -18.4 degrees, which is the part that accumulates.
  //
  // The expectations are the analog prototype written longhand:
  //
  //     low-pass    |H| = 1/sqrt(1 + (f/fc)^2)      arg = -atan(f/fc)
  //     high-pass   |H| = 1/sqrt(1 + (fc/f)^2)      arg = +atan(fc/f)
  {
    constexpr size_t kN = 16384;
    constexpr int64_t kSettle = 32768;
    constexpr int64_t kPostRoll = 32768;
    const double bandwidthHz = 60000.0, couplingHz = 12.0;

    auto probe = [&](int bin, double& magDb, double& phaseDeg) {
      const double f = static_cast<double>(bin) * kCanonicalSampleRate /
                       static_cast<double>(kN);
      const int64_t total = kSettle + static_cast<int64_t>(kN) + kPostRoll;
      AudioBuffer in(1, total);
      for (int64_t i = 0; i < total; ++i)
        in.channel(0)[i] = 0.02 * std::sin(2.0 * M_PI * f *
                                           static_cast<double>(i) / kCanonicalSampleRate);
      ParamSet p;
      // 40 dB of headroom and a tiny input, so the stage is linear to a part
      // in 10^9 and what is left is the two poles.
      p.set("drive_db", 0.0).set("output_trim_db", 0.0).set("headroom_db", 40.0)
          .set("coupling_hz", couplingHz).set("bandwidth_hz", bandwidthHz);
      const AudioBuffer out = runCharacter(console, p, in, 1024, 1024);
      const std::complex<double> a = binDft(in.channel(0) + kSettle, kN, bin);
      const std::complex<double> b = binDft(out.channel(0) + kSettle, kN, bin);
      const std::complex<double> h = b / a;
      magDb = 20.0 * std::log10(std::abs(h));
      phaseDeg = std::arg(h) * 180.0 / M_PI;
    };

    // TWO EXPECTATIONS PER POINT, AND THEY ARE DIFFERENT CLAIMS.
    //
    //   the DESIGN     what dsp/AnalogOnePole.h says it builds: a TPT one-pole
    //                  whose integrator gain is g = tan(pi fRef/fs) * fc/fRef,
    //                  giving |H| = 1/sqrt(1 + (tan(pi f/fs)/g)^2) and
    //                  arg = -atan(tan(pi f/fs)/g). Asserted tightly: this is
    //                  whether the code builds the filter it documents.
    //
    //   the PROTOTYPE  the analog first-order pole the design stands in for.
    //                  Asserted as a BOUND, because a one-pole has one degree
    //                  of freedom and the bilinear map's warping can be
    //                  cancelled at exactly one frequency. Matching at 20 kHz
    //                  makes both errors zero there and at DC, and the bulge
    //                  in between is the price.
    const int bins[] = {17, 1707, 3413}; // ~100 Hz, ~10 kHz, ~20 kHz
    const double gLp = std::tan(M_PI * 20000.0 / kCanonicalSampleRate) *
                       (bandwidthHz / 20000.0);
    const double gHp = std::tan(M_PI * couplingHz / kCanonicalSampleRate);
    double worstMagErr = 0.0, worstPhaseErr = 0.0;
    std::printf("       console poles (12 Hz / 60 kHz):\n");
    for (int bin : bins) {
      const double f = static_cast<double>(bin) * kCanonicalSampleRate /
                       static_cast<double>(kN);
      double magDb = 0.0, phaseDeg = 0.0;
      probe(bin, magDb, phaseDeg);

      const double t = std::tan(M_PI * f / kCanonicalSampleRate);
      const double designMag =
          -10.0 * std::log10(1.0 + (t / gLp) * (t / gLp)) -
          10.0 * std::log10(1.0 + (gHp / t) * (gHp / t));
      const double designPhase =
          (-std::atan(t / gLp) + std::atan(gHp / t)) * 180.0 / M_PI;

      const double protoMag =
          -10.0 * std::log10(1.0 + (f / bandwidthHz) * (f / bandwidthHz)) -
          10.0 * std::log10(1.0 + (couplingHz / f) * (couplingHz / f));
      const double protoPhase = (-std::atan(f / bandwidthHz) +
                                 std::atan(couplingHz / f)) * 180.0 / M_PI;

      std::printf("         %8.1f Hz  %+.4f dB (design %+.4f, analog %+.4f)  "
                  "%+.4f deg (design %+.4f, analog %+.4f)\n",
                  f, magDb, designMag, protoMag, phaseDeg, designPhase, protoPhase);
      check(std::abs(magDb - designMag) < 0.002,
            "console: magnitude at " + std::to_string(f) + " Hz is " +
                std::to_string(magDb) + " dB, the design says " +
                std::to_string(designMag));
      check(std::abs(phaseDeg - designPhase) < 0.02,
            "console: phase at " + std::to_string(f) + " Hz is " +
                std::to_string(phaseDeg) + " deg, the design says " +
                std::to_string(designPhase));
      worstMagErr = std::max(worstMagErr, std::abs(magDb - protoMag));
      worstPhaseErr = std::max(worstPhaseErr, std::abs(phaseDeg - protoPhase));
    }
    std::printf("         worst departure from the analog prototype: %.4f dB, "
                "%.3f deg\n", worstMagErr, worstPhaseErr);
    check(worstMagErr < 0.05 && worstPhaseErr < 1.5,
          "console: the realisation stands in for the analog pole to " +
              std::to_string(worstMagErr) + " dB and " +
              std::to_string(worstPhaseErr) + " deg across the band");
    // And it is exact where it was matched, which is the whole design claim.
    {
      double magDb = 0.0, phaseDeg = 0.0;
      probe(3413, magDb, phaseDeg);
      const double f = 3413.0 * kCanonicalSampleRate / 16384.0;
      const double protoMag =
          -10.0 * std::log10(1.0 + (f / bandwidthHz) * (f / bandwidthHz)) -
          10.0 * std::log10(1.0 + (couplingHz / f) * (couplingHz / f));
      check(std::abs(magDb - protoMag) < 0.002,
            "console: at the matched frequency the realisation IS the analog "
            "prototype (" + std::to_string(magDb) + " against " +
                std::to_string(protoMag) + " dB)");
    }

    // What prewarping at the corner would have given at 20 kHz. 60 kHz is
    // above the 48 kHz internal Nyquist, so the textbook realisation cannot
    // even place the corner — it clamps, and the roll-off it exists to model
    // very nearly disappears.
    {
      const double f = 3413.0 * kCanonicalSampleRate / 16384.0;
      const double clamped = std::min(bandwidthHz, kCanonicalSampleRate * 0.5 * 0.995);
      const double g = std::tan(M_PI * clamped / kCanonicalSampleRate);
      const double t = std::tan(M_PI * f / kCanonicalSampleRate);
      const double textbook = -10.0 * std::log10(1.0 + (t / g) * (t / g));
      const double analog = -10.0 * std::log10(1.0 + (f / bandwidthHz) * (f / bandwidthHz));
      std::printf("         prewarped at the corner it would read %+.4f dB "
                  "against the prototype's %+.4f\n", textbook, analog);
      check(std::abs(textbook - analog) > 0.4,
            "console: the corner-prewarped realisation really is a different "
            "filter in the band (" + std::to_string(std::abs(textbook - analog)) +
                " dB apart at 20 kHz)");
    }
  }

  // --- (c) one is nothing, ten is something -------------------------------
  //
  // The whole justification for this node. Ten stages in series must be ten
  // times one stage in dB and in degrees, or the effect is not the accumulating
  // one it is sold as.
  {
    constexpr size_t kN = 16384;
    constexpr int64_t kSettle = 32768;
    constexpr int64_t kPostRoll = 32768;
    const int bin = 3413; // ~20 kHz
    const double f = static_cast<double>(bin) * kCanonicalSampleRate /
                     static_cast<double>(kN);
    const int64_t total = kSettle + static_cast<int64_t>(kN) + kPostRoll;

    AudioBuffer in(1, total);
    for (int64_t i = 0; i < total; ++i)
      in.channel(0)[i] = 0.02 * std::sin(2.0 * M_PI * f *
                                         static_cast<double>(i) / kCanonicalSampleRate);

    ParamSet p;
    p.set("drive_db", 0.0).set("output_trim_db", 0.0).set("headroom_db", 40.0)
        .set("coupling_hz", 12.0).set("bandwidth_hz", 60000.0);

    auto stack = [&](int count, double& magDb, double& phaseDeg) {
      Chain chain;
      for (int i = 0; i < count; ++i) {
        Chain::SlotConfig cfg;
        cfg.nodeId = "c" + std::to_string(i);
        chain.add(std::make_unique<Console>(), p, cfg);
      }
      chain.prepare(makeSpec(1, 1024, total));
      AudioBuffer out;
      chain.render(in, out, 1024);
      const std::complex<double> a = binDft(in.channel(0) + kSettle, kN, bin);
      const std::complex<double> b = binDft(out.channel(0) + kSettle, kN, bin);
      const std::complex<double> h = b / a;
      magDb = 20.0 * std::log10(std::abs(h));
      phaseDeg = std::arg(h) * 180.0 / M_PI;
    };

    double m1 = 0.0, p1 = 0.0, m10 = 0.0, p10 = 0.0;
    stack(1, m1, p1);
    stack(10, m10, p10);
    // arg() wraps into (-180, 180]; ten stages of -18.4 degrees is -184.
    const double unwrapped10 = p10 > 0.0 ? p10 - 360.0 : p10;
    std::printf("       console in aggregate at %.0f Hz: 1 stage %+.3f dB "
                "%+.2f deg, 10 stages %+.3f dB %+.2f deg\n",
                f, m1, p1, m10, unwrapped10);
    check(std::abs(m10 - 10.0 * m1) < 0.02,
          "console: ten stages are ten times one, in dB (" +
              std::to_string(m10) + " against " + std::to_string(10.0 * m1) + ")");
    check(std::abs(unwrapped10 - 10.0 * p1) < 0.5,
          "console: ten stages are ten times one, in degrees (" +
              std::to_string(unwrapped10) + " against " +
              std::to_string(10.0 * p1) + ")");
    // And one stage is genuinely inaudible on its own, which is the other half
    // of the claim and the reason the node is allowed to exist at all.
    check(std::abs(m1) < 0.6,
          "console: one stage is under 0.6 dB at 20 kHz (" +
              std::to_string(m1) + ")");
  }

  // --- (d) aliasing at the hardest setting the node allows -----------------
  //
  // The per-commit suite measures this node at 24 dB of headroom, where it is
  // barely nonlinear. 6 dB is the floor `setParams` permits and is where the
  // aliasing requirement actually binds.
  {
    std::printf("       console in-band ASR at 0 dBFS:");
    for (double headroom : {24.0, 12.0, 6.0}) {
      ParamSet p;
      p.set("drive_db", 0.0).set("output_trim_db", 0.0)
          .set("headroom_db", headroom);
      const CharacterAsr r = measureCharacterAsr(console, p, 1.0, 19500.0);
      std::printf("  %.0f dB headroom %.1f dB", headroom, r.inBandDb);
      check(r.collisionFree, "console ASR bin classification is unambiguous");
      check(r.inBandDb <= -100.0,
            "console in-band ASR at " + std::to_string(headroom) +
                " dB headroom = " + std::to_string(r.inBandDb) +
                " dB, requirement -100");
    }
    std::printf("\n");
  }

  // --- (e) the bounds bind ------------------------------------------------
  //
  // Same reasoning as the tape section: the ASR figures above are stated at the
  // hardest configuration the node ACCEPTS, so they mean nothing unless asking
  // for something harder returns the accepted one. Bit-exact, because a clamp
  // either fires or it does not.
  {
    const AudioBuffer in = testSignal(2, 8192);
    auto render = [&](double drive, double headroom) {
      ParamSet p;
      p.set("drive_db", drive).set("headroom_db", headroom);
      return runCharacter(console, p, in, 512, 1024);
    };
    check(maxAbsDiff(render(12.0, 24.0), render(30.0, 24.0), 8192) == 0.0,
          "console: drive_db is clamped at +12 dB, which with the 6 dB headroom "
          "floor is the hardest setting the aliasing figure covers");
    check(maxAbsDiff(render(-12.0, 24.0), render(-40.0, 24.0), 8192) == 0.0,
          "console: drive_db is clamped at -12 dB");
    check(maxAbsDiff(render(0.0, 6.0), render(0.0, 2.0), 8192) == 0.0,
          "console: headroom_db is clamped at 6 dB, below which the node stops "
          "being a bus stage and becomes a saturator");
    check(maxAbsDiff(render(0.0, 40.0), render(0.0, 60.0), 8192) == 0.0,
          "console: headroom_db is clamped at 40 dB");
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
  std::printf("[19] stereo width\n");         { TIMED("stereo width"); testStereoWidth(); }
  std::printf("[20] chorus\n");               { TIMED("chorus"); testChorus(); }
  std::printf("[21] plate reverb\n");         { TIMED("plate"); testPlateReverb(); }
  std::printf("[22] dither\n");               { TIMED("dither"); testDither(); }
  std::printf("[23] zero-phase EQ\n");        { TIMED("zero-phase EQ"); testZeroPhaseEq(); }
  std::printf("[24] linear-phase EQ\n");      { TIMED("linear-phase EQ"); testLinearPhaseEq(); }
  std::printf("[25] multiband compressor\n"); { TIMED("multiband compressor"); testMultibandCompressor(); }
  std::printf("[26] transient designer\n");   { TIMED("transient designer"); testTransientDesigner(); }
  std::printf("[27] tape\n");                 { TIMED("tape"); testTape(); }
  std::printf("[28] console\n");              { TIMED("console"); testConsole(); }

  std::printf("\n%d checks, %d failure(s)\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
