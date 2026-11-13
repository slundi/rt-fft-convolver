//! [`DriveStage`] — the integrated oversampled nonlinear-drive engine.
//!
//! One `DriveStage` wires the Phase 0 primitives into the full path a pedal
//! needs:
//!
//! ```text
//!  in ─▶ [Oversampler ↑] ─▶ ×gain ─▶ +bias ─▶ [Shaper] ─▶ [Oversampler ↓]
//!        ─▶ ×level ─▶ [DcBlocker] ─▶ [ToneStack] ─▶ out
//! ```
//!
//! The gain / bias / shaper run **inside** the oversampled region so the
//! harmonics the nonlinearity generates are created above the base Nyquist and
//! filtered back down, keeping aliasing below the roadmap's floor.
//!
//! # Latency
//!
//! The half-band cascade is linear-phase, so it has a non-zero, integer group
//! delay (reported by [`DriveStage::latency_samples`] for host compensation).
//! The DC blocker and tone stack are IIR and contribute phase, not reportable
//! latency.  The roadmap's Phase 1 note of "0 samples" is corrected here: the
//! delays on the up- and down-legs *add* (they do not cancel); "same on both
//! legs" only means the filter is symmetric.  It is measured once at
//! construction with a unit impulse.
//!
//! # Real-time safety
//!
//! [`DriveStage::process_block`] allocates nothing, never locks and does not
//! panic in release builds, provided `input.len() <= max_block`.

use crate::DcBlocker;
use crate::pedal::{PedalModel, ToneConfig};
use crate::resampler::Oversampler;
use crate::shaper::Shaper;
use crate::tone::ToneStack;
use rt_fft_convolver::db_to_linear;

/// The post-shaper DC blocker corner (Hz).
const DC_BLOCK_HZ: f32 = 20.0;

/// The fully-assembled drive engine for one audio channel.
pub struct DriveStage {
    oversampler: Oversampler,
    shaper: Shaper,
    /// Pre-shaper drive gain (linear).
    input_gain: f32,
    /// Pre-shaper DC bias.
    bias: f32,
    /// Post-shaper make-up gain (linear).
    output_gain: f32,
    dc_blocker: DcBlocker,
    tone: ToneStack,
    /// Base rate, kept so [`DriveStage::set_tone`] can redesign the biquads.
    sample_rate: f32,
    /// Reported latency in base-rate samples (linear-phase group delay).
    latency: usize,
    /// Oversampled scratch buffer, capacity `max_block * factor`.
    oversampled: Vec<f32>,
    max_block: usize,
}

impl DriveStage {
    /// Build a drive stage from a [`PedalModel`] at `sample_rate`, sized for a
    /// maximum input block of `max_block` samples, using the resampler's
    /// default filter design.
    ///
    /// **Allocates** — call off the audio thread.
    pub fn from_model(model: &PedalModel, sample_rate: f32, max_block: usize) -> Self {
        Self::with_design(
            model,
            sample_rate,
            max_block,
            Oversampler::DEFAULT_ATTEN_DB,
            Oversampler::DEFAULT_TRANSITION,
        )
    }

    /// Build a drive stage with an explicit resampler filter design.
    ///
    /// **Allocates** — call off the audio thread.
    pub fn with_design(
        model: &PedalModel,
        sample_rate: f32,
        max_block: usize,
        atten_db: f64,
        transition_frac: f64,
    ) -> Self {
        let mut oversampler = Oversampler::with_design(
            model.default_oversampling as usize,
            max_block,
            atten_db,
            transition_frac,
        );
        let factor = oversampler.factor();
        let latency = measure_oversampler_latency(&mut oversampler, max_block);

        Self {
            oversampler,
            shaper: model.shaper,
            input_gain: db_to_linear(model.input_gain_db),
            bias: model.bias,
            output_gain: db_to_linear(model.output_gain_db),
            dc_blocker: DcBlocker::new(DC_BLOCK_HZ, sample_rate),
            tone: model.tone.build(sample_rate),
            sample_rate,
            latency,
            oversampled: vec![0.0; max_block * factor],
            max_block,
        }
    }

    /// Set the pre-shaper drive gain, in dB.  **Real-time safe** — one `powf`,
    /// no rebuild.
    #[inline]
    pub fn set_input_gain_db(&mut self, gain_db: f32) {
        self.input_gain = db_to_linear(gain_db);
    }

    /// Set the post-shaper make-up gain, in dB.  **Real-time safe.**
    #[inline]
    pub fn set_output_gain_db(&mut self, gain_db: f32) {
        self.output_gain = db_to_linear(gain_db);
    }

    /// Set the pre-shaper DC bias.  **Real-time safe.**
    #[inline]
    pub fn set_bias(&mut self, bias: f32) {
        self.bias = bias;
    }

    /// Replace the tone stack from a [`ToneConfig`].
    ///
    /// Redesigns the biquad coefficients (a few transcendentals, no
    /// allocation) and resets the filter state, so call it only when the tone
    /// actually changes — a per-block call on every block would zip the tone
    /// filters' memory and can click.  Cheap enough for occasional automation.
    #[inline]
    pub fn set_tone(&mut self, tone: ToneConfig) {
        self.tone = tone.build(self.sample_rate);
    }

    /// Reported latency in base-rate samples (for host delay compensation).
    #[inline]
    pub fn latency_samples(&self) -> usize {
        self.latency
    }

    /// The active oversampling factor.
    #[inline]
    pub fn oversampling_factor(&self) -> usize {
        self.oversampler.factor()
    }

    /// The maximum input block length this stage was sized for.
    #[inline]
    pub fn max_block(&self) -> usize {
        self.max_block
    }

    /// Process one block: `output[i] = drive(input[i])`.
    ///
    /// `input` and `output` must be the same length and `<= max_block`
    /// (debug-asserted).
    ///
    /// **Real-time safe.**
    pub fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), output.len(), "process_block: length mismatch");
        debug_assert!(input.len() <= self.max_block, "block exceeds max_block");

        let block = input.len().min(output.len());
        let factor = self.oversampler.factor();
        let oversampled = &mut self.oversampled[..block * factor];

        // 1. Up-sample the raw input into the oversampled region.
        self.oversampler.upsample(&input[..block], oversampled);

        // 2. Drive, bias and shape at the oversampled rate.  Applying the
        //    linear input gain here is equivalent to applying it before
        //    up-sampling (the resampler is linear), and keeps the whole
        //    nonlinear step in one pass.
        for sample in oversampled.iter_mut() {
            *sample = self.shaper.process(*sample * self.input_gain + self.bias);
        }

        // 3. Down-sample back to the base rate.
        self.oversampler
            .downsample(oversampled, &mut output[..block]);

        // 4. Make-up gain, then remove the bias-induced DC, then tone.
        for sample in output[..block].iter_mut() {
            *sample *= self.output_gain;
        }
        self.dc_blocker.process_block(&mut output[..block]);
        self.tone.process_block(&mut output[..block]);
    }

    /// Clear all filter state (oversampler delay lines, DC blocker, tone).
    pub fn reset(&mut self) {
        self.oversampler.reset();
        self.dc_blocker.reset();
        self.tone.reset();
    }
}

/// Measure the linear-phase group delay of the oversampler round-trip by
/// pushing a unit impulse through it and locating the output peak.
///
/// Runs at construction time (allocates, non-RT); the oversampler is reset
/// afterwards so the measurement leaves no residual state.
fn measure_oversampler_latency(oversampler: &mut Oversampler, max_block: usize) -> usize {
    let factor = oversampler.factor();
    let mut input = vec![0.0f32; max_block];
    let mut oversampled = vec![0.0f32; max_block * factor];
    let mut output = vec![0.0f32; max_block];

    input[0] = 1.0;
    let mut peak_index = 0usize;
    let mut peak_value = 0.0f32;
    let mut offset = 0usize;
    // A handful of blocks is plenty for any factor ≤ 16 with the default taps.
    for _ in 0..8 {
        oversampler.upsample(&input, &mut oversampled);
        oversampler.downsample(&oversampled, &mut output);
        for (index, &sample) in output.iter().enumerate() {
            if sample.abs() > peak_value {
                peak_value = sample.abs();
                peak_index = offset + index;
            }
        }
        offset += max_block;
        input[0] = 0.0; // impulse only on the very first sample
    }
    oversampler.reset();
    peak_index
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pedal::{ToneConfig, shipped_pedals};
    use std::f32::consts::TAU;

    const SAMPLE_RATE: f32 = 48_000.0;

    /// A light resampler design keeps the impulse-latency measurement fast.
    fn light() -> (f64, f64) {
        (80.0, 0.1)
    }

    fn model(shaper: Shaper, oversampling: u8) -> PedalModel {
        PedalModel {
            name: "Test".to_string(),
            input_gain_db: 0.0,
            shaper,
            bias: 0.0,
            tone: ToneConfig::Bypass,
            output_gain_db: 0.0,
            default_oversampling: oversampling,
        }
    }

    /// Silence in → silence out at steady state, for every shipped pedal.
    ///
    /// A biased pedal (e.g. "Rodent Distortion") emits a constant DC term for
    /// a silent input — `shaper(bias)` — which the DC blocker removes over its
    /// ~20 Hz time constant.  So the invariant is *steady-state* silence: run
    /// long enough for the blocker to settle, then check the final block.
    #[test]
    fn silence_in_silence_out() {
        for pedal in &shipped_pedals().unwrap() {
            let (atten, transition) = light();
            let mut stage = DriveStage::with_design(pedal, SAMPLE_RATE, 64, atten, transition);
            let input = vec![0.0f32; 64];
            let mut output = vec![1.0f32; 64];
            for _ in 0..100 {
                stage.process_block(&input, &mut output);
            }
            assert!(
                output.iter().all(|&s| s.abs() < 1e-5),
                "{}: silence produced non-silence at steady state",
                pedal.name
            );
        }
    }

    /// Output stays finite and bounded for a hot input, for every pedal.
    #[test]
    fn hot_input_stays_bounded() {
        for pedal in &shipped_pedals().unwrap() {
            let (atten, transition) = light();
            let mut stage = DriveStage::with_design(pedal, SAMPLE_RATE, 256, atten, transition);
            let input: Vec<f32> = (0..256)
                .map(|n| 4.0 * (TAU * 0.01 * n as f32).sin())
                .collect();
            let mut output = vec![0.0f32; 256];
            stage.process_block(&input, &mut output);
            assert!(
                output.iter().all(|&s| s.is_finite() && s.abs() < 100.0),
                "{}: output not bounded",
                pedal.name
            );
        }
    }

    /// The stage-reported latency matches the impulse peak of a near-linear
    /// pass (a small impulse through `tanh`, which is ≈ identity near 0).
    #[test]
    fn reported_latency_matches_impulse_peak() {
        let (atten, transition) = light();
        for factor in [2u8, 4, 8] {
            let mut stage = DriveStage::with_design(
                &model(Shaper::Tanh, factor),
                SAMPLE_RATE,
                256,
                atten,
                transition,
            );
            let reported = stage.latency_samples();

            // Small impulse so tanh(x) ≈ x → the path is effectively linear.
            let mut collected = Vec::new();
            let mut input = vec![0.0f32; 256];
            input[0] = 0.01;
            for _ in 0..8 {
                let mut output = vec![0.0f32; 256];
                stage.process_block(&input, &mut output);
                collected.extend_from_slice(&output);
                input[0] = 0.0;
            }
            let (peak_index, _) = collected.iter().enumerate().fold(
                (0usize, 0.0f32),
                |(best_index, best), (index, &sample)| {
                    if sample.abs() > best {
                        (index, sample.abs())
                    } else {
                        (best_index, best)
                    }
                },
            );
            // Allow ±1 sample for the DC blocker's one-sample differentiator.
            let difference = (peak_index as isize - reported as isize).abs();
            assert!(
                difference <= 1,
                "factor {factor}: reported latency {reported} vs impulse peak {peak_index}"
            );
        }
    }

    /// factor = 1 (bypass oversampling) reports zero latency.
    #[test]
    fn bypass_oversampling_has_zero_latency() {
        let (atten, transition) = light();
        let stage =
            DriveStage::with_design(&model(Shaper::Tanh, 1), SAMPLE_RATE, 64, atten, transition);
        assert_eq!(stage.latency_samples(), 0);
    }

    /// Live setters change the output without a rebuild: more input gain into a
    /// saturating shaper yields a hotter (but still bounded) output.
    #[test]
    fn live_input_gain_setter_changes_output() {
        let (atten, transition) = light();
        let mut stage =
            DriveStage::with_design(&model(Shaper::Tanh, 2), SAMPLE_RATE, 128, atten, transition);
        let input: Vec<f32> = (0..128)
            .map(|n| 0.1 * (TAU * 0.02 * n as f32).sin())
            .collect();

        stage.set_input_gain_db(0.0);
        let mut quiet = vec![0.0f32; 128];
        stage.process_block(&input, &mut quiet);

        stage.reset();
        stage.set_input_gain_db(24.0);
        let mut loud = vec![0.0f32; 128];
        stage.process_block(&input, &mut loud);

        let energy = |buffer: &[f32]| buffer.iter().map(|s| (s * s) as f64).sum::<f64>();
        assert!(
            energy(&loud) > energy(&quiet) * 2.0,
            "more drive should raise output energy: quiet={}, loud={}",
            energy(&quiet),
            energy(&loud)
        );
        assert!(loud.iter().all(|&s| s.is_finite() && s.abs() <= 1.5));
    }

    /// An asymmetric shaper with bias injects DC, and the in-stage DC blocker
    /// must remove it: the settled output mean is ≈ 0.
    #[test]
    fn dc_blocker_removes_bias_offset() {
        let (atten, transition) = light();
        let pedal = PedalModel {
            name: "Biased".to_string(),
            input_gain_db: 6.0,
            shaper: Shaper::Tube,
            bias: 0.3,
            tone: ToneConfig::Bypass,
            output_gain_db: 0.0,
            default_oversampling: 2,
        };
        let mut stage = DriveStage::with_design(&pedal, SAMPLE_RATE, 512, atten, transition);
        // A quiet tone so the shaper's asymmetry produces a clear DC term.
        // Its period (64 samples) divides the block exactly, so the tone
        // contributes zero mean over the measurement window — any residual is
        // genuine DC leakage, not a partial-period artefact.
        let freq_frac = 1.0 / 64.0;
        let mut mean = 0.0f64;
        let mut count = 0.0f64;
        for block in 0..8 {
            let input: Vec<f32> = (0..512)
                .map(|n| 0.2 * (TAU * freq_frac * (block * 512 + n) as f32).sin())
                .collect();
            let mut output = vec![0.0f32; 512];
            stage.process_block(&input, &mut output);
            if block >= 4 {
                for &sample in &output {
                    mean += sample as f64;
                    count += 1.0;
                }
            }
        }
        mean /= count;
        assert!(mean.abs() < 1e-3, "residual DC not removed: mean = {mean}");
    }

    /// `reset` clears state: after driving hard then resetting, a silent block
    /// comes out silent.
    #[test]
    fn reset_clears_state() {
        let (atten, transition) = light();
        let mut stage =
            DriveStage::with_design(&model(Shaper::Tanh, 4), SAMPLE_RATE, 128, atten, transition);
        let hot = vec![5.0f32; 128];
        let mut output = vec![0.0f32; 128];
        stage.process_block(&hot, &mut output);
        stage.reset();
        let silence = vec![0.0f32; 128];
        // One block after reset may still carry the DC blocker's transient of
        // the first silent sample, so check the tail of a second block.
        stage.process_block(&silence, &mut output);
        stage.process_block(&silence, &mut output);
        assert!(
            output[64..].iter().all(|&s| s.abs() < 1e-6),
            "state not cleared after reset"
        );
    }
}
