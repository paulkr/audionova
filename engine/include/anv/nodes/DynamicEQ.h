// Dynamic EQ — bands whose gain is driven by the energy in their own band.
//
// This one node subsumes three products. A de-esser is a downward band at
// 5-9 kHz. Resonance suppression is a narrow downward band. "Tame the 2 kHz
// harshness" is a wide downward band. All three ship as PRESETS rather than as
// separate nodes, which is why the Phase 1 node list is 16 rather than 19.
//
// THE HARD PART IS MODULATING FILTER GAIN PER SAMPLE.
//
// The obvious approach — recompute biquad coefficients every sample — is both
// expensive (a tan and a pow per sample per band) and wrong: in a
// gain-dependent topology the state variables mean something different after
// the coefficients change, so stepping the gain steps the state and the filter
// clicks. Recomputing every N samples instead just trades clicks for zipper
// noise.
//
// Instead every band is decomposed so that gain appears ONLY as an output mix
// coefficient, never inside the filter's state evolution:
//
//     bell        y = x + k(P-1) * BP(x)      BP has k = 1/Q, fixed
//     low shelf   y = x + (P-1)  * LP1(x)     LP1 first order, fixed
//     high shelf  y = x + (P-1)  * HP1(x)
//
// where P is the linear band gain. At the bell's centre the normalised
// bandpass has magnitude 1/k, so the response there is 1 + k(P-1)/k = P
// exactly. The filter state never sees P at all, so the gain can move every
// single sample with no discontinuity and no zipper — and it costs one
// multiply.
//
// The detector taps the SAME filter that does the work, before the gain is
// applied. That makes the sidechain inherently frequency-selective — a band
// only responds to energy inside itself — and costs no extra filtering.
//
// Latency is exactly 0: everything here is minimum-phase IIR and there is no
// lookahead. Lookahead would need a second filter bank running on the undelayed
// signal; worth adding for de-essing later, but it is not free the way it was
// for the compressor and gate.
#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "anv/core/Node.h"
#include "anv/dsp/Dynamics.h"
#include "anv/dsp/TptSvf.h"

namespace anv {

class DynamicEQ final : public Node {
public:
  const char* type() const override { return "dynamic_eq"; }

  void prepare(const PrepareSpec& spec) override {
    spec_ = spec;
    rebuild();
  }

  void reset() override {
    for (auto& perBand : svf_)
      for (auto& f : perBand) f.reset();
    for (auto& perBand : onePole_)
      for (auto& f : perBand) f.reset();
    for (auto& perBand : det_)
      for (auto& d : perBand) d.reset(0.0);
  }

  void setParams(const ParamSet& p) override {
    bands_.clear();
    for (const ParamSet& b : p.items()) {
      Band band;
      const std::string type = b.str("filter_type", "bell");
      band.shelf = (type == "lowshelf" || type == "highshelf");
      band.highShelf = (type == "highshelf");
      band.freqHz = b.num("center_freq_hz", 1000.0);
      band.q = std::max(0.1, b.num("q_factor", 1.0));
      band.staticGainDb = clampDb(b.num("static_gain_db", 0.0));

      const std::string mode = b.str("mode", "static");
      band.mode = (mode == "down") ? Mode::Down
                                   : (mode == "up") ? Mode::Up : Mode::Static;
      band.thresholdDb = clampDb(b.num("threshold_dbfs", -24.0));
      band.ratio = std::max(1.0, b.num("ratio_to_1", 4.0));
      band.kneeDb = std::max(0.0, b.num("knee_db", 3.0));
      band.rangeDb = std::max(0.0, b.num("range_db", 12.0));
      band.attackMs = std::clamp(b.num("attack_ms", 5.0), 0.01, 10000.0);
      band.releaseMs = std::clamp(b.num("release_ms", 80.0), 1.0, 10000.0);
      bands_.push_back(band);
    }
    if (spec_.numChannels > 0) rebuild();
  }

  int latencyFrames() const override { return 0; }

  void process(ProcessContext& ctx) noexcept override {
    const int n = ctx.numFrames;
    const int ch = spec_.numChannels;
    const size_t nb = bands_.size();

    for (int c = 0; c < ch; ++c) {
      const size_t ci = static_cast<size_t>(c);
      const Sample* src = ctx.in[c];
      Sample* dst = ctx.out[c];

      for (int i = 0; i < n; ++i) {
        double x = src[i];

        for (size_t b = 0; b < nb; ++b) {
          const Band& band = bands_[b];

          // One filter evaluation serves both the sidechain and the output.
          double raw, normalised;
          if (band.shelf) {
            const double lp = onePole_[b][ci].processLowpass(x);
            raw = band.highShelf ? (x - lp) : lp;
            normalised = raw;
          } else {
            raw = svf_[b][ci].processSample(x); // v1, magnitude Q at centre
            normalised = raw * band.k;
          }

          double gainDb = band.staticGainDb;
          if (band.mode != Mode::Static) {
            // The ballistics smooth the LEVEL, not the resulting gain.
            //
            // Smoothing the gain instead looks equivalent and is not. The
            // detector holds the maximum of what it is fed, so feeding it a
            // gain magnitude makes downward mode latch onto the waveform's
            // peaks (right) and upward mode latch onto its ZERO CROSSINGS
            // (catastrophically wrong — every waveform is momentarily silent
            // twice a cycle, so an upward band would sit pinned at full boost
            // forever regardless of how loud the band actually is).
            //
            // Rectifying and enveloping first gives one level that means the
            // same thing in both directions.
            const double env = det_[b][ci].process(
                std::abs(normalised), band.alphaAttack, band.alphaRelease);
            const double levelDb = linToDb(env);

            // Upward mode is the mirror image of downward, so one gain
            // computer serves both: reflect the level and the threshold.
            const double magnitude =
                (band.mode == Mode::Down)
                    ? band.computer.reductionDb(levelDb)
                    : band.mirrored.reductionDb(-levelDb);
            const double clamped = std::min(magnitude, band.rangeDb);
            gainDb += (band.mode == Mode::Down) ? -clamped : clamped;
          }

          // exp2 rather than pow: same result, materially cheaper, and this
          // runs once per band per sample.
          const double P = std::exp2(gainDb * kInvDbPerOctave);
          const double mix = band.shelf ? (P - 1.0) : band.k * (P - 1.0);
          x += mix * raw;
        }

        dst[i] = x;
      }
    }
  }

private:
  enum class Mode { Static, Down, Up };

  static constexpr double kInvDbPerOctave = 1.0 / 6.020599913279624;

  struct Band {
    bool shelf = false;
    bool highShelf = false;
    double freqHz = 1000.0;
    double q = 1.0;
    double k = 1.0;
    double staticGainDb = 0.0;
    Mode mode = Mode::Static;
    double thresholdDb = -24.0;
    double ratio = 4.0;
    double kneeDb = 3.0;
    double rangeDb = 12.0;
    double attackMs = 5.0;
    double releaseMs = 80.0;
    double alphaAttack = 0.0;
    double alphaRelease = 0.0;
    GainComputer computer;
    GainComputer mirrored;
  };

  void rebuild() {
    const int ch = spec_.numChannels;
    if (ch <= 0) return;
    const double sr = spec_.sampleRate;
    const size_t nb = bands_.size();

    svf_.assign(nb, {});
    onePole_.assign(nb, {});
    det_.assign(nb, {});

    for (size_t b = 0; b < nb; ++b) {
      Band& band = bands_[b];
      band.k = 1.0 / band.q;
      band.alphaAttack = timeConstantAlpha(band.attackMs * 0.001, sr);
      band.alphaRelease = timeConstantAlpha(band.releaseMs * 0.001, sr);
      band.computer = GainComputer{band.thresholdDb, band.ratio, band.kneeDb};
      band.mirrored = GainComputer{-band.thresholdDb, band.ratio, band.kneeDb};

      svf_[b].assign(static_cast<size_t>(ch), TptSvf{});
      onePole_[b].assign(static_cast<size_t>(ch), TptOnePole{});
      det_[b].assign(static_cast<size_t>(ch), DecoupledPeakDetector{});

      for (int c = 0; c < ch; ++c) {
        svf_[b][static_cast<size_t>(c)].setSampleRate(sr);
        svf_[b][static_cast<size_t>(c)].configure(FilterShape::BandPass,
                                                  band.freqHz, 0.0, band.q);
        onePole_[b][static_cast<size_t>(c)].setSampleRate(sr);
        onePole_[b][static_cast<size_t>(c)].configure(band.freqHz);
      }
    }
    reset();
  }

  std::vector<Band> bands_;
  std::vector<std::vector<TptSvf>> svf_;
  std::vector<std::vector<TptOnePole>> onePole_;
  std::vector<std::vector<DecoupledPeakDetector>> det_;
};

} // namespace anv
