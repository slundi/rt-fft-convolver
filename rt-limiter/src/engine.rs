//! Framework-agnostic limiter engine: an off-thread [`LimiterController`] and
//! an audio-thread [`LimiterProcessor`] joined by the lock-free handoff from
//! `rt-ir-plugin-core`.
//!
//! # Threading
//!
//! - [`LimiterController`] (control / background thread) rebuilds the
//!   [`Limiter`] whenever the **structural** parameter changes — the
//!   oversampling factor, which reallocates the true-peak detectors — and
//!   publishes the fresh [`PreparedLimiter`] through the wait-free handoff.
//! - [`LimiterProcessor`] (audio thread) polls the handoff and applies the
//!   **continuous** parameters (threshold, ceiling, release, curve, look-ahead,
//!   channel link, dither, bypass) in place — no allocation — then renders.
//!
//! Continuous parameters are applied with change detection so per-block setters
//! that reset engine state (look-ahead, channel link) only fire on an actual
//! move.  When a freshly-built payload arrives, *all* continuous parameters are
//! re-applied to it, since it is a new object with default-constructed state.
//!
//! # Mono handling
//!
//! The [`Limiter`] is inherently a stereo, channel-linked engine.  A mono
//! stream is limited by pairing the single channel with an internal scratch
//! copy (`L == R`), which the linked gain computer collapses to the same
//! result — the scratch output is discarded.

use rt_fft_convolver::linear_to_db;
use rt_ir_plugin_core::{IrConsumer, IrPublisher, ir_handoff};
use rt_limiter_core::Limiter;

use crate::params::LimiterParams;

/// A built, ready-to-render payload handed from the controller to the
/// processor: one stereo [`Limiter`] plus the factor it was built for.
pub struct PreparedLimiter {
    /// The stereo true-peak limiter.
    pub limiter: Limiter,
    /// The oversampling factor this limiter was built with.
    pub factor: usize,
}

/// The continuous parameters last applied to the current limiter, used for
/// change detection so state-resetting setters do not fire every block.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Applied {
    threshold_db: f32,
    ceiling_db: f32,
    release_ms: f32,
    curve: rt_limiter_core::ReleaseCurve,
    lookahead_ms: f32,
    link: rt_limiter_core::ChannelLink,
    dither: crate::params::DitherChoice,
}

impl Applied {
    fn from_params(params: &LimiterParams) -> Self {
        Self {
            threshold_db: params.threshold_db,
            ceiling_db: params.ceiling_db,
            release_ms: params.effective_release_ms(),
            curve: params.effective_curve(),
            lookahead_ms: params.lookahead_ms,
            link: params.link.to_link(),
            dither: params.dither,
        }
    }
}

/// Off-thread builder / publisher of [`PreparedLimiter`] payloads.
pub struct LimiterController {
    publisher: IrPublisher<PreparedLimiter>,
    sample_rate: f32,
    max_block: usize,
    params: LimiterParams,
}

impl LimiterController {
    /// Create a controller + processor pair, publishing the first payload
    /// immediately so the processor has something to render on its first block.
    ///
    /// **Allocates** — call off the audio thread.
    pub fn new(
        max_block: usize,
        sample_rate: f32,
        params: LimiterParams,
    ) -> (Self, LimiterProcessor) {
        let (mut publisher, consumer) = ir_handoff::<PreparedLimiter>();
        publisher.publish(build_prepared(max_block, sample_rate, &params));

        let controller = Self {
            publisher,
            sample_rate,
            max_block,
            params,
        };
        let processor = LimiterProcessor::new(consumer, max_block);
        (controller, processor)
    }

    /// The current oversampling factor.
    #[inline]
    pub fn factor(&self) -> usize {
        self.params.oversampling.factor()
    }

    /// Update the cached parameters and rebuild **only** if the structural
    /// (oversampling) parameter changed.  Returns `true` if a rebuild happened.
    ///
    /// **Allocates on rebuild** — control thread only.
    pub fn update(&mut self, params: LimiterParams) -> bool {
        let rebuild = params.oversampling != self.params.oversampling;
        self.params = params;
        if rebuild {
            self.republish();
        }
        rebuild
    }

    fn republish(&mut self) {
        self.publisher.publish(build_prepared(
            self.max_block,
            self.sample_rate,
            &self.params,
        ));
    }
}

/// Build a [`PreparedLimiter`] for the given parameters.  **Allocates** —
/// control thread only.
fn build_prepared(max_block: usize, sample_rate: f32, params: &LimiterParams) -> PreparedLimiter {
    let factor = params.oversampling.factor();
    let limiter = Limiter::new(
        sample_rate,
        max_block.max(1),
        factor,
        params.threshold_db,
        params.ceiling_db,
        params.effective_release_ms(),
        params.effective_curve(),
        params.link.to_link(),
    );
    PreparedLimiter { limiter, factor }
}

/// Audio-thread renderer.  Polls the handoff, applies continuous parameters and
/// renders in place.
pub struct LimiterProcessor {
    consumer: IrConsumer<PreparedLimiter>,
    /// Scratch right channel for limiting a mono stream (`L == R`).
    mono_scratch: Vec<f32>,
    /// Continuous parameters last applied to the current payload.
    applied: Option<Applied>,
    /// Samples rendered on the last `process` call.
    last_process_samples: usize,
    /// Reported latency of the current payload, in samples.
    last_latency: usize,
    /// Peak gain reduction (dB, ≤ 0) of the last non-bypassed block.
    last_gain_reduction_db: f32,
}

impl LimiterProcessor {
    fn new(consumer: IrConsumer<PreparedLimiter>, max_block: usize) -> Self {
        Self {
            consumer,
            mono_scratch: vec![0.0; max_block.max(1)],
            applied: None,
            last_process_samples: 0,
            last_latency: 0,
            last_gain_reduction_db: 0.0,
        }
    }

    /// Samples rendered on the last [`process`](Self::process) call.
    #[inline]
    pub fn last_process_samples(&self) -> usize {
        self.last_process_samples
    }

    /// Reported latency of the active payload, in samples (look-ahead +
    /// detector group delay).
    #[inline]
    pub fn latency_samples(&self) -> usize {
        self.last_latency
    }

    /// Peak gain reduction of the last non-bypassed block, in dB (`≤ 0`).
    #[inline]
    pub fn gain_reduction_db(&self) -> f32 {
        self.last_gain_reduction_db
    }

    /// Render `channels` in place using `params`.
    ///
    /// Handles 1 channel (mono, via a scratch pair) or ≥ 2 channels (the first
    /// two are the stereo pair; any extra channels are left untouched).
    ///
    /// **Real-time safe** — no allocation, no locking (provided every channel
    /// length ≤ `max_block`).
    pub fn process(&mut self, channels: &mut [&mut [f32]], params: &LimiterParams) {
        // Borrow the payload (disjoint field from the scratch/state below).
        let (payload, changed) = self.consumer.poll_changed();
        let Some(prepared) = payload else {
            self.last_process_samples = 0;
            return;
        };
        let limiter = &mut prepared.limiter;

        // A freshly-published payload is a new object → re-apply everything.
        if changed {
            self.applied = None;
        }
        let target = Applied::from_params(params);
        let force = self.applied.is_none();
        let previous = self.applied;

        if force || previous.map(|p| p.threshold_db) != Some(target.threshold_db) {
            limiter.set_threshold_db(target.threshold_db);
        }
        if force || previous.map(|p| p.ceiling_db) != Some(target.ceiling_db) {
            limiter.set_ceiling_db(target.ceiling_db);
        }
        if force
            || previous.map(|p| (p.release_ms, p.curve)) != Some((target.release_ms, target.curve))
        {
            limiter.set_release_ms(target.release_ms);
            limiter.set_curve(target.curve);
        }
        // These two reset engine state, so only touch them on a real change.
        if force || previous.map(|p| p.link) != Some(target.link) {
            limiter.set_link(target.link);
        }
        if force || previous.map(|p| p.lookahead_ms) != Some(target.lookahead_ms) {
            limiter.set_lookahead_ms(target.lookahead_ms);
        }
        #[cfg(feature = "dither")]
        if force || previous.map(|p| p.dither) != Some(target.dither) {
            apply_dither(limiter, target.dither);
        }

        self.applied = Some(target);
        self.last_latency = limiter.latency_samples();

        let block = channels
            .iter()
            .map(|channel| channel.len())
            .min()
            .unwrap_or(0);

        if params.bypass || block == 0 {
            // Pass-through: no reduction reported (host-visible bypass keeps the
            // dry signal; the constant reported latency still applies).
            self.last_gain_reduction_db = 0.0;
            self.last_process_samples = block;
            return;
        }

        match channels.len() {
            1 => {
                let scratch = &mut self.mono_scratch[..block];
                scratch.copy_from_slice(&channels[0][..block]);
                limiter.process_block(&mut channels[0][..block], scratch);
            }
            _ => {
                // Split off the first channel so both borrows are disjoint.
                let (first, rest) = channels.split_at_mut(1);
                limiter.process_block(&mut first[0][..block], &mut rest[0][..block]);
            }
        }

        self.last_gain_reduction_db = linear_to_db(limiter.min_gain());
        self.last_process_samples = block;
    }
}

/// Map a [`DitherChoice`](crate::params::DitherChoice) onto the engine's dither
/// stage.  Only compiled with the `dither` feature.
#[cfg(feature = "dither")]
fn apply_dither(limiter: &mut Limiter, choice: crate::params::DitherChoice) {
    use crate::params::DitherChoice;
    use rt_limiter_core::BitDepth;
    match choice {
        DitherChoice::Off => limiter.disable_dither(),
        DitherChoice::Bits16 => limiter.enable_dither(BitDepth::Bits16, true),
        DitherChoice::Bits20 => limiter.enable_dither(BitDepth::Bits20, true),
        DitherChoice::Bits24 => limiter.enable_dither(BitDepth::Bits24, true),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{LinkChoice, OversampleChoice};
    use rt_fft_convolver::db_to_linear;
    use std::f32::consts::TAU;

    const SAMPLE_RATE: f32 = 48_000.0;
    const MAX_BLOCK: usize = 256;

    fn pair(params: LimiterParams) -> (LimiterController, LimiterProcessor) {
        LimiterController::new(MAX_BLOCK, SAMPLE_RATE, params)
    }

    /// Render a mono tone for `blocks` blocks, returning the concatenated
    /// output.
    fn run_mono(
        processor: &mut LimiterProcessor,
        params: &LimiterParams,
        make: impl Fn(usize) -> f32,
        blocks: usize,
    ) -> Vec<f32> {
        let mut output = Vec::with_capacity(blocks * MAX_BLOCK);
        for block_index in 0..blocks {
            let mut samples: Vec<f32> = (0..MAX_BLOCK)
                .map(|n| make(block_index * MAX_BLOCK + n))
                .collect();
            {
                let mut channels: [&mut [f32]; 1] = [&mut samples];
                processor.process(&mut channels, params);
            }
            output.extend_from_slice(&samples);
        }
        output
    }

    /// The controller builds a limiter and the processor renders stereo.
    #[test]
    fn builds_and_renders_stereo() {
        let params = LimiterParams::default();
        let (_controller, mut processor) = pair(params);
        let mut left = vec![0.1f32; MAX_BLOCK];
        let mut right = vec![0.1f32; MAX_BLOCK];
        {
            let mut channels: [&mut [f32]; 2] = [&mut left, &mut right];
            processor.process(&mut channels, &params);
        }
        assert!(left.iter().all(|s| s.is_finite()));
        assert!(right.iter().all(|s| s.is_finite()));
        assert_eq!(processor.last_process_samples(), MAX_BLOCK);
        assert!(processor.latency_samples() > 0);
    }

    /// Bypass passes the input through untouched.
    #[test]
    fn bypass_is_transparent() {
        let params = LimiterParams {
            bypass: true,
            ..Default::default()
        };
        let (_controller, mut processor) = pair(params);
        let original: Vec<f32> = (0..MAX_BLOCK).map(|n| (n as f32 * 0.1).sin()).collect();
        let mut samples = original.clone();
        {
            let mut channels: [&mut [f32]; 1] = [&mut samples];
            processor.process(&mut channels, &params);
        }
        assert_eq!(samples, original);
    }

    /// A hot signal is limited: the output peak sits at/below the ceiling and
    /// the GR meter shows reduction.
    #[test]
    fn hot_signal_is_limited_and_metered() {
        let params = LimiterParams {
            threshold_db: -6.0,
            ceiling_db: -1.0,
            ..Default::default()
        };
        let (_controller, mut processor) = pair(params);
        let output = run_mono(
            &mut processor,
            &params,
            |n| 1.5 * (TAU * 0.05 * n as f32).sin(),
            8,
        );

        let ceiling = db_to_linear(-1.0);
        // Skip the latency ramp-up region.
        let settled = &output[MAX_BLOCK * 2..];
        let peak = settled.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        assert!(
            peak <= ceiling * 1.02,
            "output peak {peak} above ceiling {ceiling}"
        );
        assert!(
            processor.gain_reduction_db() < -0.5,
            "GR meter should show reduction, got {}",
            processor.gain_reduction_db()
        );
    }

    /// Changing the oversampling factor triggers a rebuild; other changes do
    /// not.
    #[test]
    fn only_oversampling_rebuilds() {
        let params = LimiterParams::default();
        let (mut controller, _processor) = pair(params);

        // A continuous-only change: no rebuild.
        let quieter = LimiterParams {
            threshold_db: -3.0,
            ..params
        };
        assert!(!controller.update(quieter));

        // A structural change: rebuild.
        let higher_os = LimiterParams {
            oversampling: OversampleChoice::X8,
            ..quieter
        };
        assert!(controller.update(higher_os));
        assert_eq!(controller.factor(), 8);
    }

    /// After a rebuild the processor picks up the new payload and re-applies
    /// the continuous parameters (latency reflects the new factor).
    #[test]
    fn rebuild_is_adopted_by_processor() {
        let params = LimiterParams {
            oversampling: OversampleChoice::X2,
            ..Default::default()
        };
        let (mut controller, mut processor) = pair(params);

        // Prime the processor with the initial payload.
        let mut left = vec![0.2f32; MAX_BLOCK];
        let mut right = vec![0.2f32; MAX_BLOCK];
        {
            let mut channels: [&mut [f32]; 2] = [&mut left, &mut right];
            processor.process(&mut channels, &params);
        }
        let latency_2x = processor.latency_samples();

        // Rebuild at 8× and process again — latency should grow.
        let higher = LimiterParams {
            oversampling: OversampleChoice::X8,
            ..params
        };
        assert!(controller.update(higher));
        {
            let mut channels: [&mut [f32]; 2] = [&mut left, &mut right];
            processor.process(&mut channels, &higher);
        }
        assert!(
            processor.latency_samples() > latency_2x,
            "8× latency {} should exceed 2× latency {latency_2x}",
            processor.latency_samples()
        );
    }

    /// Mid/side mode on a mono (L == R) input stays mono and bounded.
    #[test]
    fn mid_side_mono_symmetry() {
        let params = LimiterParams {
            threshold_db: -3.0,
            link: LinkChoice::MidSide,
            ..Default::default()
        };
        let (_controller, mut processor) = pair(params);
        for block in 0..6 {
            let mut left: Vec<f32> = (0..MAX_BLOCK)
                .map(|n| 1.4 * (TAU * 0.04 * (block * MAX_BLOCK + n) as f32).sin())
                .collect();
            let mut right = left.clone();
            {
                let mut channels: [&mut [f32]; 2] = [&mut left, &mut right];
                processor.process(&mut channels, &params);
            }
            for (l, r) in left.iter().zip(right.iter()) {
                assert!((l - r).abs() < 1e-4, "mono symmetry broken");
                assert!(l.is_finite());
            }
        }
    }
}
