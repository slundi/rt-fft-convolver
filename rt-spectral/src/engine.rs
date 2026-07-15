//! Framework-agnostic spectral engine: an off-thread [`SpectralController`] and
//! an audio-thread [`SpectralProcessor`] joined by two lock-free hand-offs.
//!
//! # Threading
//!
//! Two data flows cross the audio / control boundary, both wait-free and
//! allocation-free on the audio side:
//!
//! - **control → audio** — the learned [`NoiseProfile`], one
//!   [`profile_channel`] per channel.  The controller `publish`es a fresh
//!   profile; every [`SpectralEngine`] reads its own copy through the
//!   `ProfileReceiver` it was built with.
//! - **audio → control** — the *Learn* capture.  When the user arms a learn,
//!   the processor records the mono-summed input **in place** into the
//!   triple-buffer's producer slot and `publish`es once full — no allocation on
//!   the audio thread.  The controller then turns that clip into a profile
//!   ([`learn_profile`], off-RT) and publishes it back.
//!
//! # Why a per-channel engine
//!
//! [`SpectralEngine`] handles one channel (see `rt-spectral-core`); a stereo
//! instance owns two, each with its own STFT state and profile receiver.  A
//! single learned profile is published to **all** channels — a noise floor is a
//! property of the source, not the channel.
//!
//! # Latency
//!
//! Every engine shares the same `fft_size`, so the reported latency (one full
//! STFT frame plus the click despiker's fixed group delay) is constant and
//! block-size independent — read it once via
//! [`SpectralProcessor::latency_samples`].

use rt_spectral_core::{
    NoiseProfile, ProfileSender, SpectralEngine, SpectralParams, learn_profile, profile_channel,
};
use triple_buffer::{Input, Output, TripleBuffer};

use crate::params::{MAX_LEARN_MS, learn_samples};

/// Fixed-capacity mono capture clip handed audio → control for `Learn`.
///
/// `data` is pre-allocated to the maximum learn duration; `len` marks how many
/// leading samples are valid.  Cloned three times by the triple buffer at
/// construction so no publish ever reallocates.
#[derive(Clone)]
struct CaptureClip {
    data: Vec<f32>,
    len: usize,
}

/// Off-thread builder / learner.  Owns the control side of both hand-offs.
///
/// Lives behind the plugin's background-task executor: it reads a completed
/// Learn capture, runs [`learn_profile`] (which allocates), and publishes the
/// resulting [`NoiseProfile`] to every channel — never on the audio thread.
pub struct SpectralController {
    /// Consumer side of the audio → control Learn capture hand-off.
    capture: Output<CaptureClip>,
    /// One profile publisher per channel (all receive the same profile).
    profile_senders: Vec<ProfileSender>,
    fft_size: usize,
    hop: usize,
}

impl SpectralController {
    /// Turn the most recently captured Learn clip into a [`NoiseProfile`] and
    /// publish it to every channel.  Returns `true` when a non-empty capture
    /// was consumed.
    ///
    /// **Allocates** — control thread only.
    pub fn learn(&mut self) -> bool {
        // Pull the freshest published clip (the flag is discarded — the plugin
        // only calls this once per completed capture).
        self.capture.update();
        let clip = self.capture.peek_output_buffer();
        if clip.len == 0 {
            return false;
        }
        let profile = learn_profile(&clip.data[..clip.len], self.fft_size, self.hop);
        for sender in &mut self.profile_senders {
            sender.publish(profile.clone());
        }
        true
    }

    /// STFT size the engines were built for.
    #[inline]
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }
}

/// Audio-thread renderer: one [`SpectralEngine`] per channel plus the RT-safe
/// Learn recorder.
pub struct SpectralProcessor {
    /// One independent spectral engine per channel.
    engines: Vec<SpectralEngine>,
    /// Producer side of the audio → control Learn capture hand-off.
    capture: Input<CaptureClip>,
    /// Scratch input copy so each engine can render in place (`process_block`
    /// needs disjoint input/output slices).
    io_scratch: Vec<f32>,
    max_block: usize,

    // ── Learn capture state ──
    capturing: bool,
    capture_pos: usize,
    capture_target: usize,
    /// Set once a capture completes; the plugin drains it to trigger the
    /// off-thread [`SpectralController::learn`].
    pending_learn: bool,

    latency: usize,
}

impl SpectralProcessor {
    /// Reported latency in samples (constant for the built `fft_size`).
    #[inline]
    pub fn latency_samples(&self) -> usize {
        self.latency
    }

    /// Number of audio channels this processor renders.
    #[inline]
    pub fn channels(&self) -> usize {
        self.engines.len()
    }

    /// `true` while a Learn capture is in progress.
    #[inline]
    pub fn is_capturing(&self) -> bool {
        self.capturing
    }

    /// Arm a Learn capture of `samples` mono-summed samples (clamped to the
    /// capture buffer capacity).  Restarts any capture already in progress.
    pub fn begin_learn(&mut self, samples: usize) {
        let capacity = self.capture.input_buffer_mut().data.len();
        self.capture_target = samples.min(capacity);
        self.capture_pos = 0;
        self.capturing = self.capture_target > 0;
    }

    /// Take (and clear) the "capture complete" flag.  The plugin calls this
    /// after [`process`](Self::process); a `true` return means the controller
    /// should now run [`SpectralController::learn`].
    #[inline]
    pub fn take_pending_learn(&mut self) -> bool {
        std::mem::take(&mut self.pending_learn)
    }

    /// Render every channel in place using `params`.
    ///
    /// **Real-time safe** — no allocation, no lock, no panic (each channel is
    /// processed in `max_block` chunks by the underlying engine).
    pub fn process(&mut self, channels: &mut [&mut [f32]], params: &SpectralParams) {
        // Capture the *dry* input before any channel is overwritten.
        if self.capturing {
            self.capture_input(channels);
        }

        for (engine, channel) in self.engines.iter_mut().zip(channels.iter_mut()) {
            let len = channel.len();
            let mut offset = 0;
            while offset < len {
                let chunk = (len - offset).min(self.max_block);
                let scratch = &mut self.io_scratch[..chunk];
                scratch.copy_from_slice(&channel[offset..offset + chunk]);
                engine.process_block(scratch, &mut channel[offset..offset + chunk], params);
                offset += chunk;
            }
        }
    }

    /// Record the mono-summed input block into the capture producer slot.
    fn capture_input(&mut self, channels: &mut [&mut [f32]]) {
        let num_ch = channels.len();
        if num_ch == 0 {
            return;
        }
        let frames = channels[0].len();
        let clip = self.capture.input_buffer_mut();
        let inv_ch = 1.0 / num_ch as f32;

        for frame in 0..frames {
            if self.capture_pos >= self.capture_target {
                break;
            }
            let mut sum = 0.0;
            for channel in channels.iter() {
                sum += channel[frame];
            }
            clip.data[self.capture_pos] = sum * inv_ch;
            self.capture_pos += 1;
        }

        if self.capture_pos >= self.capture_target {
            clip.len = self.capture_target;
            self.capture.publish();
            self.capturing = false;
            self.pending_learn = true;
        }
    }

    /// Flush every engine to silence and cancel any Learn capture in progress.
    pub fn reset(&mut self) {
        for engine in &mut self.engines {
            engine.reset();
        }
        self.capturing = false;
        self.capture_pos = 0;
        self.pending_learn = false;
    }
}

/// Build a connected [`SpectralController`] / [`SpectralProcessor`] pair for a
/// `channels`-channel instance: an `fft_size`-point STFT at 75 % overlap,
/// `sample_rate`, host blocks up to `max_block`.
///
/// **Allocates** — call off the audio thread (e.g. in the plugin's
/// `initialize`).  The Learn capture buffer is sized for [`MAX_LEARN_MS`] so
/// arming a learn never reallocates.
///
/// # Panics
/// Panics unless `channels > 0` and `fft_size` is a multiple of 4 and `>= 8`
/// (the [`SpectralEngine`] contract).
pub fn spectral_engine(
    sample_rate: u32,
    fft_size: usize,
    max_block: usize,
    channels: usize,
) -> (SpectralController, SpectralProcessor) {
    assert!(channels > 0, "channels must be > 0");
    let hop = fft_size / 4;
    let num_bins = fft_size / 2 + 1;
    let max_block = max_block.max(1);

    // The capture buffer holds the longest learn clip; the triple buffer clones
    // it into all three slots so no publish reallocates on the audio thread.
    let max_capture = learn_samples(MAX_LEARN_MS, sample_rate).max(fft_size);
    let seed = CaptureClip {
        data: vec![0.0; max_capture],
        len: 0,
    };
    let (capture_in, capture_out) = TripleBuffer::new(&seed).split();

    let mut engines = Vec::with_capacity(channels);
    let mut profile_senders = Vec::with_capacity(channels);
    for _ in 0..channels {
        let (sender, receiver) = profile_channel(NoiseProfile::silent(num_bins));
        engines.push(SpectralEngine::new(
            sample_rate,
            fft_size,
            max_block,
            receiver,
        ));
        profile_senders.push(sender);
    }
    let latency = engines[0].latency_samples();

    let controller = SpectralController {
        capture: capture_out,
        profile_senders,
        fft_size,
        hop,
    };
    let processor = SpectralProcessor {
        engines,
        capture: capture_in,
        io_scratch: vec![0.0; max_block],
        max_block,
        capturing: false,
        capture_pos: 0,
        capture_target: 0,
        pending_learn: false,
        latency,
    };
    (controller, processor)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rt_spectral_core::DenoiseParams;
    use std::f32::consts::TAU;

    const SR: u32 = 48_000;
    const FFT: usize = 1024;

    fn rms(buf: &[f32]) -> f32 {
        (buf.iter().map(|x| x * x).sum::<f32>() / buf.len() as f32).sqrt()
    }

    /// Deterministic xorshift pseudo-noise in `[-amp, amp]`.
    fn noise_source(amp: f32) -> impl FnMut() -> f32 {
        let mut state = 0x1234_5678u32;
        move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            amp * ((state as f32 / u32::MAX as f32) * 2.0 - 1.0)
        }
    }

    /// Feed a block through the processor (single channel).
    fn run_block(processor: &mut SpectralProcessor, block: &mut [f32], params: &SpectralParams) {
        let mut channels: [&mut [f32]; 1] = [block];
        processor.process(&mut channels, params);
    }

    #[test]
    fn builds_and_renders_stereo() {
        let (_controller, mut processor) = spectral_engine(SR, FFT, 128, 2);
        assert_eq!(processor.channels(), 2);
        let params = SpectralParams::default();
        let mut left = vec![0.1f32; 128];
        let mut right = vec![0.1f32; 128];
        {
            let mut channels: [&mut [f32]; 2] = [&mut left, &mut right];
            processor.process(&mut channels, &params);
        }
        assert!(left.iter().all(|s| s.is_finite()));
        assert!(right.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn latency_is_frame_plus_click() {
        let (controller, processor) = spectral_engine(SR, 2048, 256, 2);
        assert_eq!(controller.fft_size(), 2048);
        // One full frame (2048) plus the 5-tap click despiker delay (2).
        assert_eq!(processor.latency_samples(), 2048 + 2);
    }

    #[test]
    fn bypass_passes_dry_through_delayed() {
        let params = SpectralParams {
            bypass: true,
            ..SpectralParams::default()
        };
        let (_controller, mut processor) = spectral_engine(SR, FFT, 256, 1);
        let lat = processor.latency_samples();

        let input: Vec<f32> = (0..8192)
            .map(|n| 0.4 * (TAU * 5.0 * n as f32 / 1024.0).sin())
            .collect();
        let mut output = vec![0.0f32; input.len()];
        for (chunk_in, chunk_out) in input.chunks(256).zip(output.chunks_mut(256)) {
            let mut block = chunk_in.to_vec();
            run_block(&mut processor, &mut block, &params);
            chunk_out.copy_from_slice(&block);
        }

        // Bypass emits the latency-matched dry signal untouched.
        let mut max_err = 0.0f32;
        for n in (FFT * 3)..(input.len() - FFT) {
            max_err = max_err.max((output[n] - input[n - lat]).abs());
        }
        assert!(max_err < 1e-4, "bypass altered the signal (err {max_err})");
    }

    /// A completed capture flags `pending_learn`, and running the controller's
    /// off-thread learn then engages the denoiser (pulls the noise floor down).
    #[test]
    fn learn_capture_then_denoise_reduces_noise() {
        let block_size = 256;
        let (mut controller, mut processor) = spectral_engine(SR, FFT, block_size, 1);

        let mut noise = noise_source(0.1);

        // Arm a short learn and feed noise until the capture completes.
        processor.begin_learn(FFT * 16);
        let mut completed = false;
        for _ in 0..200 {
            let mut block: Vec<f32> = (0..block_size).map(|_| noise()).collect();
            run_block(&mut processor, &mut block, &SpectralParams::default());
            if processor.take_pending_learn() {
                completed = true;
                break;
            }
        }
        assert!(completed, "learn capture never completed");
        assert!(controller.learn(), "controller found no capture to learn");

        // Denoise engaged; fresh noise of the same distribution is attenuated.
        let params = SpectralParams {
            denoise: DenoiseParams {
                enabled: true,
                reduction_db: 24.0,
                sensitivity_db: 6.0,
            },
            transient: rt_spectral_core::TransientParams {
                enabled: false,
                ..Default::default()
            },
            ..SpectralParams::default()
        };

        let input: Vec<f32> = (0..FFT * 32).map(|_| noise()).collect();
        let mut output = vec![0.0f32; input.len()];
        for (chunk_in, chunk_out) in input.chunks(block_size).zip(output.chunks_mut(block_size)) {
            let mut block = chunk_in.to_vec();
            run_block(&mut processor, &mut block, &params);
            chunk_out.copy_from_slice(&block);
        }

        let in_rms = rms(&input[FFT * 4..]);
        let out_rms = rms(&output[FFT * 4..]);
        assert!(
            out_rms < in_rms * 0.6,
            "noise floor not reduced: in {in_rms} out {out_rms}"
        );
    }

    /// Without a completed capture the controller has nothing to learn.
    #[test]
    fn learn_without_capture_is_noop() {
        let (mut controller, _processor) = spectral_engine(SR, FFT, 128, 1);
        assert!(!controller.learn());
    }

    #[test]
    fn begin_learn_clamps_to_capacity_and_completes() {
        let block_size = 512;
        let (_controller, mut processor) = spectral_engine(SR, FFT, block_size, 1);
        // Ask for far more than the buffer holds; it is clamped and still ends.
        processor.begin_learn(usize::MAX);
        assert!(processor.is_capturing());

        let mut noise = noise_source(0.2);
        let mut completed = false;
        // 2000 ms @ 48 kHz = 96 000 samples → 188 blocks of 512.
        for _ in 0..256 {
            let mut block: Vec<f32> = (0..block_size).map(|_| noise()).collect();
            run_block(&mut processor, &mut block, &SpectralParams::default());
            if processor.take_pending_learn() {
                completed = true;
                break;
            }
        }
        assert!(completed, "clamped capture never completed");
        assert!(!processor.is_capturing());
    }

    #[test]
    fn reset_cancels_capture_and_clears_flag() {
        let (_controller, mut processor) = spectral_engine(SR, FFT, 128, 1);
        processor.begin_learn(FFT * 8);
        assert!(processor.is_capturing());
        processor.reset();
        assert!(!processor.is_capturing());
        assert!(!processor.take_pending_learn());
    }

    #[test]
    fn stereo_capture_learns_from_both_channels() {
        let block_size = 256;
        let (mut controller, mut processor) = spectral_engine(SR, FFT, block_size, 2);
        processor.begin_learn(FFT * 16);

        let mut noise = noise_source(0.1);
        let mut completed = false;
        for _ in 0..200 {
            let mut left: Vec<f32> = (0..block_size).map(|_| noise()).collect();
            let mut right: Vec<f32> = (0..block_size).map(|_| noise()).collect();
            {
                let mut channels: [&mut [f32]; 2] = [&mut left, &mut right];
                processor.process(&mut channels, &SpectralParams::default());
            }
            if processor.take_pending_learn() {
                completed = true;
                break;
            }
        }
        assert!(completed);
        // Both channels receive the published profile.
        assert!(controller.learn());
    }
}
