//! Framework-agnostic audio-thread renderer for `rt-multiband`.
//!
//! # Why there is no `Controller`
//!
//! The sibling dynamics plugins split an off-thread `*Controller` (which
//! reallocates on a structural change) from an audio-thread `*Processor`.
//! `rt-multiband` v1 needs no such split: [`rt_multiband_core::MultibandEngine`]
//! performs **zero heap allocation** — its crossover, bands and smoothers are
//! all fixed-size `Copy` state — and every Phase-2 control is an in-place
//! setter.  So the engine lives **resident** in [`MultibandProcessor`] (the
//! convention's "engines the audio thread mutates stay resident in the
//! Processor"), built once in [`MultibandProcessor::new`] off the audio thread
//! when the host reports the sample rate.  A v2 band-count or linear-phase
//! change *would* reallocate and reintroduce the controller seam — hence
//! [`MultibandParams::structural_changed`](crate::params::MultibandParams::structural_changed)
//! is kept in the model.
//!
//! # Applying parameters
//!
//! The full [`MultibandParams`] is a small `Copy`+`PartialEq` struct, so the
//! processor change-detects the *whole* snapshot: when it differs from the last
//! applied one, every setter is re-applied (all cheap coefficient math, and all
//! state-preserving so a live change never clicks); otherwise only the
//! per-sample smoothers advance.  Global `Input`/`Output`/`Mix` are smoothed
//! inside the engine.
//!
//! # I/O
//!
//! Stereo in → stereo out, and mono → mono (the single channel is paired with
//! itself and the left output taken — the linked detector collapses `L == R`).

use rt_fft_convolver::DenormalGuard;
use rt_multiband_core::{MultibandEngine, NUM_BANDS};

use crate::params::MultibandParams;

/// Audio-thread renderer owning a resident [`MultibandEngine`].
pub struct MultibandProcessor {
    engine: MultibandEngine,
    /// The parameters last applied, for whole-snapshot change detection.
    applied: Option<MultibandParams>,
    /// Deepest per-band gain reduction over the last block, dB (`≤ 0`).
    band_gr_db: [f32; NUM_BANDS],
    /// Samples rendered on the last `process` call.
    last_process_samples: usize,
}

impl MultibandProcessor {
    /// Build a processor at `sample_rate` initialised to `params`.
    ///
    /// **Allocates** — off-RT only (the engine builds its fixed state; despite
    /// being heap-free it is a large struct, so construct it off the audio
    /// thread).
    pub fn new(sample_rate: f32, params: &MultibandParams) -> Self {
        let clamped = params.clamped();
        let engine = MultibandEngine::new(sample_rate, clamped.crossover_hz);
        let mut processor = Self {
            engine,
            applied: None,
            band_gr_db: [0.0; NUM_BANDS],
            last_process_samples: 0,
        };
        // Apply the initial snapshot so the first block already reflects the
        // preset (the global gain/mix smoothers ramp in over their ~10 ms
        // glide, which is inaudible).
        processor.apply(&clamped);
        processor.applied = Some(clamped);
        processor
    }

    /// Samples rendered on the last [`process`](Self::process) call.
    #[inline]
    pub fn last_process_samples(&self) -> usize {
        self.last_process_samples
    }

    /// Deepest gain reduction of band `index` over the last block, dB (`≤ 0`).
    #[inline]
    pub fn band_gain_reduction_db(&self, index: usize) -> f32 {
        self.band_gr_db[index]
    }

    /// The reported latency in samples — always `0` in v1 (zero-latency IIR
    /// crossover; the v2 linear-phase mode will report its FIR delay here).
    #[inline]
    pub fn latency_samples(&self) -> usize {
        0
    }

    /// Apply a full parameter snapshot to the engine (all setters).  RT-safe.
    fn apply(&mut self, params: &MultibandParams) {
        self.engine.set_input_gain_db(params.input_gain_db);
        self.engine.set_output_gain_db(params.output_gain_db);
        self.engine.set_mix(params.mix);
        self.engine.set_link(params.link.to_link());
        self.engine.set_frequencies(params.crossover_hz);

        for (index, band_params) in params.bands.iter().enumerate() {
            let band = self.engine.band_mut(index);
            {
                let computer = band.gain_computer_mut();
                computer.set_mode(band_params.mode.to_mode());
                computer.set_threshold_db(band_params.threshold_db);
                computer.set_ratio(band_params.ratio);
                computer.set_knee_db(band_params.knee_db);
            }
            band.set_ballistics(band_params.attack_ms, band_params.release_ms);
            band.set_blend(band_params.blend);
            band.set_makeup_db(band_params.makeup_db);
            band.sidechain_mut(0).set_hpf_hz(band_params.sc_hpf_hz);
            band.sidechain_mut(1).set_hpf_hz(band_params.sc_hpf_hz);
            band.solo = band_params.solo;
            band.mute = band_params.mute;
        }
    }

    /// Render `channels` in place using `params`.
    ///
    /// Handles 1 channel (mono, paired with itself) or ≥ 2 channels (the first
    /// two are the stereo pair; extra channels are left untouched).
    ///
    /// **Real-time safe** — no allocation, no locking, no panics.
    pub fn process(&mut self, channels: &mut [&mut [f32]], params: &MultibandParams) {
        // Flush denormals for the whole callback so the crossover biquads and
        // envelope tails settle to zero on quiet passages (this is the RT
        // boundary — guard it here rather than relying on the caller).
        let _guard = DenormalGuard::new();

        let clamped = params.clamped();
        if self.applied != Some(clamped) {
            self.apply(&clamped);
            self.applied = Some(clamped);
        }

        let block = channels
            .iter()
            .map(|channel| channel.len())
            .min()
            .unwrap_or(0);
        self.band_gr_db = [0.0; NUM_BANDS];

        // Host bypass: leave the input untouched (dry pass-through).
        if clamped.bypass || block == 0 || channels.is_empty() {
            self.last_process_samples = block;
            return;
        }

        match channels.len() {
            1 => {
                let mono = &mut channels[0][..block];
                for sample in mono.iter_mut() {
                    let (left, _right) = self.engine.process(*sample, *sample);
                    *sample = left;
                    self.track_gr();
                }
            }
            _ => {
                let (first, rest) = channels.split_at_mut(1);
                let left = &mut first[0][..block];
                let right = &mut rest[0][..block];
                for (left_sample, right_sample) in left.iter_mut().zip(right.iter_mut()) {
                    let (out_left, out_right) = self.engine.process(*left_sample, *right_sample);
                    *left_sample = out_left;
                    *right_sample = out_right;
                    self.track_gr();
                }
            }
        }

        self.last_process_samples = block;
    }

    /// Fold this sample's per-band reduction into the per-block minimum.
    #[inline]
    fn track_gr(&mut self) {
        for index in 0..NUM_BANDS {
            let reduction = self.engine.band(index).gain_reduction_db();
            self.band_gr_db[index] = self.band_gr_db[index].min(reduction);
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{BandModeChoice, MultibandParams};
    use std::f32::consts::TAU;

    const SR: f32 = 48_000.0;
    const BLOCK: usize = 256;

    /// Render a mono tone through `blocks` blocks, returning the output.
    fn run_mono(
        processor: &mut MultibandProcessor,
        params: &MultibandParams,
        freq: f32,
        amplitude: f32,
        blocks: usize,
    ) -> Vec<f32> {
        let mut output = Vec::with_capacity(blocks * BLOCK);
        for block_index in 0..blocks {
            let mut samples: Vec<f32> = (0..BLOCK)
                .map(|n| {
                    let time = (block_index * BLOCK + n) as f32 / SR;
                    amplitude * (TAU * freq * time).sin()
                })
                .collect();
            {
                let mut channels: [&mut [f32]; 1] = [&mut samples];
                processor.process(&mut channels, params);
            }
            output.extend_from_slice(&samples);
        }
        output
    }

    fn rms(signal: &[f32]) -> f32 {
        (signal.iter().map(|&s| s * s).sum::<f32>() / signal.len() as f32).sqrt()
    }

    fn bypass_all(params: &mut MultibandParams) {
        for band in &mut params.bands {
            band.mode = BandModeChoice::Bypass;
        }
    }

    #[test]
    fn builds_and_renders_stereo() {
        let params = MultibandParams::default();
        let mut processor = MultibandProcessor::new(SR, &params);
        let mut left = vec![0.1f32; BLOCK];
        let mut right = vec![0.1f32; BLOCK];
        {
            let mut channels: [&mut [f32]; 2] = [&mut left, &mut right];
            processor.process(&mut channels, &params);
        }
        assert!(left.iter().chain(right.iter()).all(|s| s.is_finite()));
        assert_eq!(processor.last_process_samples(), BLOCK);
        assert_eq!(processor.latency_samples(), 0);
    }

    #[test]
    fn bypass_is_transparent() {
        let params = MultibandParams {
            bypass: true,
            ..Default::default()
        };
        let mut processor = MultibandProcessor::new(SR, &params);
        let original: Vec<f32> = (0..BLOCK).map(|n| (n as f32 * 0.1).sin()).collect();
        let mut samples = original.clone();
        {
            let mut channels: [&mut [f32]; 1] = [&mut samples];
            processor.process(&mut channels, &params);
        }
        assert_eq!(samples, original);
    }

    #[test]
    fn all_bands_bypassed_reconstruct_the_input() {
        // With every band in Bypass mode (but mix fully wet) the crossover sum
        // reconstructs the input at flat magnitude, so the settled output RMS
        // matches the known input sine RMS (amplitude / √2).
        let mut params = MultibandParams::default();
        bypass_all(&mut params);
        let mut processor = MultibandProcessor::new(SR, &params);
        let output = run_mono(&mut processor, &params, 500.0, 0.5, 4);
        let expected = 0.5 / std::f32::consts::SQRT_2;
        let err = 20.0 * (rms(&output[BLOCK * 2..]) / expected).log10();
        assert!(err.abs() < 0.3, "bypass-mode reconstruction {err} dB");
    }

    #[test]
    fn low_band_compression_reduces_a_low_tone() {
        let mut params = MultibandParams::default();
        bypass_all(&mut params);
        params.bands[0].mode = BandModeChoice::Compressor;
        params.bands[0].threshold_db = -30.0;
        params.bands[0].ratio = 8.0;
        let mut processor = MultibandProcessor::new(SR, &params);
        // A 60 Hz tone sits in band0; hard compression should reduce it.
        let hot = run_mono(&mut processor, &params, 60.0, 0.8, 8);
        let quiet = run_mono(&mut processor, &params, 60.0, 0.02, 4);
        let hot_rms = rms(&hot[BLOCK * 4..]);
        let quiet_rms = rms(&quiet);
        // The loud tone is squashed proportionally more than the quiet one.
        assert!(hot_rms / 0.8 < quiet_rms / 0.02, "no downward compression");
        assert!(processor.band_gain_reduction_db(0) < -3.0, "GR not metered");
    }

    #[test]
    fn mono_and_default_render_is_bounded() {
        let params = MultibandParams::default();
        let mut processor = MultibandProcessor::new(SR, &params);
        let output = run_mono(&mut processor, &params, 220.0, 0.9, 8);
        assert!(output.iter().all(|s| s.is_finite() && s.abs() < 4.0));
    }

    #[test]
    fn output_gain_scales_the_result() {
        let mut params = MultibandParams::default();
        bypass_all(&mut params);
        let mut unity = MultibandProcessor::new(SR, &params);
        let flat = run_mono(&mut unity, &params, 500.0, 0.3, 8);

        params.output_gain_db = -6.0;
        let mut attenuated = MultibandProcessor::new(SR, &params);
        let lowered = run_mono(&mut attenuated, &params, 500.0, 0.3, 8);

        let ratio = rms(&lowered[BLOCK * 4..]) / rms(&flat[BLOCK * 4..]);
        assert!(
            (ratio - rt_fft_convolver::db_to_linear(-6.0)).abs() < 0.03,
            "output gain ratio {ratio}"
        );
    }

    #[test]
    fn parameter_change_between_blocks_is_applied() {
        let mut params = MultibandParams::default();
        bypass_all(&mut params);
        let mut processor = MultibandProcessor::new(SR, &params);
        run_mono(&mut processor, &params, 60.0, 0.8, 4);

        // Turn band0 into a hard compressor mid-stream.
        params.bands[0].mode = BandModeChoice::Compressor;
        params.bands[0].threshold_db = -40.0;
        params.bands[0].ratio = 10.0;
        run_mono(&mut processor, &params, 60.0, 0.8, 8);
        assert!(
            processor.band_gain_reduction_db(0) < -3.0,
            "mid-stream mode change not applied"
        );
    }

    #[test]
    fn silence_in_silence_out() {
        let params = MultibandParams::default();
        let mut processor = MultibandProcessor::new(SR, &params);
        let mut left = vec![0.0f32; BLOCK];
        let mut right = vec![0.0f32; BLOCK];
        {
            let mut channels: [&mut [f32]; 2] = [&mut left, &mut right];
            processor.process(&mut channels, &params);
        }
        assert!(left.iter().chain(right.iter()).all(|&s| s == 0.0));
    }
}
