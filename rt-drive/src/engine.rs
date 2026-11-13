//! Framework-agnostic drive engine: an off-thread [`DriveController`] and an
//! audio-thread [`DriveProcessor`] joined by the lock-free handoff from
//! `rt-ir-plugin-core`.
//!
//! # Threading
//!
//! - [`DriveController`] (control / background thread) rebuilds the per-channel
//!   [`DriveStage`]s whenever a **structural** parameter changes — the pedal
//!   model or the oversampling factor — because those reallocate the resampler.
//!   It publishes the freshly-built [`PreparedDrive`] through the wait-free
//!   handoff.
//! - [`DriveProcessor`] (audio thread) polls the handoff, applies the
//!   **continuous** parameters (drive, level, bias, tone, mix, bypass) in place
//!   — no allocation — and renders.
//!
//! Stereo is handled by building one independent [`DriveStage`] per channel
//! (roadmap Phase 2: "stereo in → stereo out, independent per channel").

use rt_drive_core::{DriveStage, PedalModel, ToneConfig};
use rt_ir_plugin_core::{IrConsumer, IrPublisher, ir_handoff};

use crate::params::{DriveParams, OversampleChoice};

/// A built, ready-to-render payload handed from the controller to the
/// processor: the active pedal plus one [`DriveStage`] per channel.
pub struct PreparedDrive {
    /// The selected pedal (its base gains / bias are the offset origin).
    pub pedal: PedalModel,
    /// One independent stage per audio channel.
    pub stages: Vec<DriveStage>,
}

/// Off-thread builder / publisher of [`PreparedDrive`] payloads.
pub struct DriveController {
    publisher: IrPublisher<PreparedDrive>,
    pedals: Vec<PedalModel>,
    sample_rate: f32,
    max_block: usize,
    channels: usize,
    model_index: usize,
    oversampling: OversampleChoice,
}

impl DriveController {
    /// Create a controller + processor pair.
    ///
    /// `pedals` is the (non-empty) preset bank; `channels` is the number of
    /// audio channels to render independently.  The initial pedal/oversampling
    /// come from `params`.  Publishes the first payload immediately so the
    /// processor has something to render on its first block.
    ///
    /// **Allocates** — call off the audio thread.
    pub fn new(
        max_block: usize,
        sample_rate: f32,
        channels: usize,
        pedals: Vec<PedalModel>,
        params: &DriveParams,
    ) -> (Self, DriveProcessor) {
        assert!(!pedals.is_empty(), "pedal bank must not be empty");
        assert!(channels > 0, "channels must be > 0");

        let (mut publisher, consumer) = ir_handoff::<PreparedDrive>();
        let model_index = params.model_index.min(pedals.len() - 1);
        let oversampling = params.oversampling;

        // Publish the first payload before the processor's first block.
        publisher.publish(build_prepared(
            &pedals,
            model_index,
            oversampling,
            sample_rate,
            max_block,
            channels,
        ));

        let controller = Self {
            publisher,
            pedals,
            sample_rate,
            max_block,
            channels,
            model_index,
            oversampling,
        };
        let processor = DriveProcessor::new(consumer, max_block);
        (controller, processor)
    }

    /// Number of pedals in the bank.
    #[inline]
    pub fn pedal_count(&self) -> usize {
        self.pedals.len()
    }

    /// The display name of pedal `index`, if in range.
    pub fn pedal_name(&self, index: usize) -> Option<&str> {
        self.pedals.get(index).map(|pedal| pedal.name.as_str())
    }

    /// Select a pedal by index (clamped to the bank) and republish.
    pub fn set_model(&mut self, index: usize) {
        self.model_index = index.min(self.pedals.len() - 1);
        self.republish();
    }

    /// Set the oversampling choice and republish.
    pub fn set_oversampling(&mut self, choice: OversampleChoice) {
        self.oversampling = choice;
        self.republish();
    }

    /// The current model index.
    #[inline]
    pub fn model_index(&self) -> usize {
        self.model_index
    }

    /// Rebuild the payload and hand it off.  **Allocates** — control thread.
    fn republish(&mut self) {
        self.publisher.publish(build_prepared(
            &self.pedals,
            self.model_index,
            self.oversampling,
            self.sample_rate,
            self.max_block,
            self.channels,
        ));
    }
}

/// Build a [`PreparedDrive`] — one [`DriveStage`] per channel — for the given
/// pedal and oversampling choice.  **Allocates** — control thread only.
fn build_prepared(
    pedals: &[PedalModel],
    model_index: usize,
    oversampling: OversampleChoice,
    sample_rate: f32,
    max_block: usize,
    channels: usize,
) -> PreparedDrive {
    let pedal = pedals[model_index].clone();
    let factor = oversampling.factor(pedal.default_oversampling);

    // Override the pedal's default oversampling with the resolved factor.
    let mut effective = pedal.clone();
    effective.default_oversampling = factor;

    let stages = (0..channels)
        .map(|_| DriveStage::from_model(&effective, sample_rate, max_block))
        .collect();
    PreparedDrive { pedal, stages }
}

/// Audio-thread renderer.  Polls the handoff and applies the continuous
/// parameters in place.
pub struct DriveProcessor {
    consumer: IrConsumer<PreparedDrive>,
    /// Scratch holding the dry (pre-drive) signal for the mix blend.
    dry: Vec<f32>,
    /// Last tone triplet applied, so the biquads are only redesigned on change.
    last_tone: Option<(f32, f32, f32)>,
    /// Number of samples processed in the last `process` call (CPU-meter proxy).
    last_process_samples: usize,
}

impl DriveProcessor {
    fn new(consumer: IrConsumer<PreparedDrive>, max_block: usize) -> Self {
        Self {
            consumer,
            dry: vec![0.0; max_block],
            last_tone: None,
            last_process_samples: 0,
        }
    }

    /// Samples rendered on the last [`process`](DriveProcessor::process) call.
    #[inline]
    pub fn last_process_samples(&self) -> usize {
        self.last_process_samples
    }

    /// Render `channels` in place using `params`.
    ///
    /// Each channel is driven by its own stage; a channel index beyond the
    /// built stage count is left untouched.  **Real-time safe** — no
    /// allocation, no locking (provided every channel length ≤ `max_block`).
    pub fn process(&mut self, channels: &mut [&mut [f32]], params: &DriveParams) {
        let Some(prepared) = self.consumer.poll() else {
            self.last_process_samples = 0;
            return;
        };
        // Disjoint borrows: read the pedal, mutate the stages.
        let PreparedDrive { pedal, stages } = prepared;

        let input_gain_db = pedal.input_gain_db + params.drive_offset_db();
        let output_gain_db = pedal.output_gain_db + params.level_db;
        let bias = pedal.bias + params.bias;
        let tone_key = (params.bass_db, params.mid_db, params.treble_db);
        let tone_changed = self.last_tone != Some(tone_key);
        let tone = ToneConfig::ThreeBand {
            bass_db: params.bass_db,
            mid_db: params.mid_db,
            treble_db: params.treble_db,
        };
        let mix = params.mix.clamp(0.0, 1.0);

        let mut rendered = 0;
        for (channel_index, channel) in channels.iter_mut().enumerate() {
            let Some(stage) = stages.get_mut(channel_index) else {
                continue;
            };
            let block = channel.len().min(self.dry.len());
            // Keep the dry signal for the mix blend / bypass.
            self.dry[..block].copy_from_slice(&channel[..block]);

            if params.bypass {
                // Leave the channel as the untouched input.
                rendered = rendered.max(block);
                continue;
            }

            stage.set_input_gain_db(input_gain_db);
            stage.set_output_gain_db(output_gain_db);
            stage.set_bias(bias);
            if tone_changed {
                stage.set_tone(tone);
            }

            stage.process_block(&self.dry[..block], &mut channel[..block]);

            // Dry / wet blend.
            for (out, &dry) in channel[..block].iter_mut().zip(self.dry[..block].iter()) {
                *out = dry * (1.0 - mix) + *out * mix;
            }
            rendered = rendered.max(block);
        }

        // Only commit the tone key once it has actually been applied to a
        // rendered (non-bypassed) channel.
        if tone_changed && !params.bypass {
            self.last_tone = Some(tone_key);
        }
        self.last_process_samples = rendered;
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rt_drive_core::shipped_pedals;

    const SAMPLE_RATE: f32 = 48_000.0;

    fn controller(channels: usize, params: &DriveParams) -> (DriveController, DriveProcessor) {
        let pedals = shipped_pedals().expect("shipped pedals parse");
        DriveController::new(64, SAMPLE_RATE, channels, pedals, params)
    }

    /// Render one channel for `blocks` blocks of `block` samples of a test tone,
    /// returning the final block's output.
    fn run(
        processor: &mut DriveProcessor,
        params: &DriveParams,
        blocks: usize,
        block: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; block];
        for block_index in 0..blocks {
            let mut samples: Vec<f32> = (0..block)
                .map(|n| 0.3 * (0.05 * (block_index * block + n) as f32).sin())
                .collect();
            {
                let mut channels: [&mut [f32]; 1] = [&mut samples];
                processor.process(&mut channels, params);
            }
            output = samples;
        }
        output
    }

    /// The controller builds one stage per channel and the processor can render.
    #[test]
    fn builds_and_renders() {
        let params = DriveParams::default();
        let (_controller, mut processor) = controller(2, &params);
        let mut left = vec![0.1f32; 64];
        let mut right = vec![0.1f32; 64];
        {
            let mut channels: [&mut [f32]; 2] = [&mut left, &mut right];
            processor.process(&mut channels, &params);
        }
        assert!(left.iter().all(|s| s.is_finite()));
        assert!(right.iter().all(|s| s.is_finite()));
        assert_eq!(processor.last_process_samples(), 64);
    }

    /// Bypass passes the input through untouched.
    #[test]
    fn bypass_is_transparent() {
        let params = DriveParams {
            bypass: true,
            ..Default::default()
        };
        let (_controller, mut processor) = controller(1, &params);
        let original: Vec<f32> = (0..64).map(|n| (n as f32 * 0.1).sin()).collect();
        let mut samples = original.clone();
        {
            let mut channels: [&mut [f32]; 1] = [&mut samples];
            processor.process(&mut channels, &params);
        }
        for (out, orig) in samples.iter().zip(original.iter()) {
            assert!((out - orig).abs() < 1e-6, "bypass altered the signal");
        }
    }

    /// mix = 0 is fully dry: output equals the input (drive path discarded).
    #[test]
    fn mix_zero_is_dry() {
        // A hot, obviously-nonlinear pedal so wet ≠ dry.
        let params = DriveParams {
            model_index: 2, // Rodent Distortion
            drive: 1.0,
            mix: 0.0,
            ..Default::default()
        };
        let (_controller, mut processor) = controller(1, &params);
        let original: Vec<f32> = (0..64).map(|n| 0.5 * (n as f32 * 0.1).sin()).collect();
        let mut samples = original.clone();
        {
            let mut channels: [&mut [f32]; 1] = [&mut samples];
            processor.process(&mut channels, &params);
        }
        for (out, orig) in samples.iter().zip(original.iter()) {
            assert!((out - orig).abs() < 1e-6, "mix=0 should be dry");
        }
    }

    /// mix = 1 with a hot drive audibly changes the signal versus dry.
    #[test]
    fn mix_one_drives_the_signal() {
        let params = DriveParams {
            model_index: 2,
            drive: 1.0,
            mix: 1.0,
            ..Default::default()
        };
        let (_controller, mut processor) = controller(1, &params);
        let original: Vec<f32> = (0..256).map(|n| 0.5 * (n as f32 * 0.05).sin()).collect();
        let mut samples = original.clone();
        {
            let mut channels: [&mut [f32]; 1] = [&mut samples];
            processor.process(&mut channels, &params);
        }
        let difference: f32 = samples
            .iter()
            .zip(original.iter())
            .map(|(out, orig)| (out - orig).abs())
            .sum();
        assert!(difference > 1e-3, "wet signal should differ from dry");
    }

    /// Silence in → silence out at steady state (bias transients settle).
    #[test]
    fn silence_is_silent_at_steady_state() {
        let params = DriveParams {
            model_index: 3, // Tube Warmth (biased)
            ..Default::default()
        };
        let (_controller, mut processor) = controller(1, &params);
        let mut output = vec![0.0f32; 64];
        for _ in 0..200 {
            let mut silence = vec![0.0f32; 64];
            {
                let mut channels: [&mut [f32]; 1] = [&mut silence];
                processor.process(&mut channels, &params);
            }
            output = silence;
        }
        assert!(
            output.iter().all(|&s| s.abs() < 1e-5),
            "steady-state silence expected, got {output:?}"
        );
    }

    /// Switching the model republishes and the processor picks up the new pedal.
    #[test]
    fn set_model_switches_pedal() {
        let params = DriveParams {
            drive: 1.0,
            ..Default::default()
        };
        let (mut controller, mut processor) = controller(1, &params);

        // Render with model 0 (Clean Boost).
        let clean = run(&mut processor, &params, 4, 64);

        // Switch to a hard distortion and render again.
        controller.set_model(2); // Rodent Distortion
        let dirty = run(&mut processor, &params, 4, 64);

        let energy = |buffer: &[f32]| buffer.iter().map(|s| (s * s) as f64).sum::<f64>();
        assert_ne!(
            controller.model_index(),
            0,
            "controller should have switched model"
        );
        // The two models should not produce identical output.
        assert!(
            (energy(&clean) - energy(&dirty)).abs() > 1e-6,
            "model switch had no audible effect"
        );
    }

    /// An out-of-range model index is clamped to the bank.
    #[test]
    fn model_index_is_clamped() {
        let params = DriveParams {
            model_index: 999,
            ..Default::default()
        };
        let (controller, _processor) = controller(1, &params);
        assert_eq!(controller.model_index(), controller.pedal_count() - 1);
    }

    /// Pedal names are exposed for the plugin's Model dropdown.
    #[test]
    fn pedal_names_available() {
        let params = DriveParams::default();
        let (controller, _processor) = controller(1, &params);
        assert_eq!(controller.pedal_name(0), Some("Clean Boost"));
        assert_eq!(controller.pedal_name(controller.pedal_count()), None);
    }
}
