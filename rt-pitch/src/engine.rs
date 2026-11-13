//! Framework-agnostic pitch engine: an off-thread [`PitchController`] and an
//! audio-thread [`PitchProcessor`] joined by the lock-free handoff from
//! `rt-ir-plugin-core`.
//!
//! # Threading
//!
//! - [`PitchController`] (control / background thread) rebuilds the shifter
//!   voices whenever a **structural** parameter changes — the algorithm, the
//!   quality (FFT size) or the formant configuration — because those
//!   reallocate FFT plans and buffers.  It publishes the freshly-built
//!   [`PreparedPitch`] through the wait-free handoff.
//! - [`PitchProcessor`] (audio thread) polls the handoff and applies the
//!   **continuous** parameters (preset selection, intervals, crossfade time,
//!   mix, dry-delay compensation, bypass) in place — no allocation.
//!
//! # Click-free preset changes (the two-voice crossfade)
//!
//! A single shifter cannot jump ratios without either a click or an audible
//! pitch glide.  So each channel owns **two** shifter voices and an
//! [`rt_pitch_core::PresetBank`] driving an equal-power crossfade between them
//! (the `CrossfadingConvolver` design principle from the root crate).  On a
//! preset change the incoming preset's ratio is handed to whichever physical
//! voice is currently faded out; the two voices are then blended sample-by-
//! sample.  Both voices always render so both stay warm — the incoming voice
//! never has to fade in from its silent latency warm-up.
//!
//! # Stereo & dual-voice
//!
//! One independent voice-pair is built per channel (stereo in → stereo out).
//! In `dual_voice` mode the second channel selects `voice2_preset` instead of
//! `current_preset`, giving an independent harmony line per side (harmoniser).

use rt_ir_plugin_core::{IrConsumer, IrPublisher, ir_handoff};
use rt_pitch_core::{
    FormantMethod, NUM_PRESETS, PhaseVocoder, PresetBank, Psola, Shifter, WindowKind, Wsola,
};

use crate::params::{Algorithm, MAX_PITCH_HZ, MIN_PITCH_HZ, PitchParams, Quality};

/// Dry-delay ring capacity, in samples.  A comfortable upper bound on any
/// shifter's `latency_samples()` across the exposed qualities (studio WSOLA is
/// 2048; the PSOLA tracker frame at 55 Hz is the largest) plus block headroom.
const MAX_DELAY: usize = 8192;

/// A built, ready-to-render payload handed from the controller to the
/// processor: one voice-pair per channel plus the shared reported latency.
pub struct PreparedPitch {
    /// One independent [`VoicePair`] per audio channel.
    voices: Vec<VoicePair>,
    /// Algorithmic latency in samples (identical across voices — same
    /// algorithm and quality), reported to the host for delay compensation.
    latency: usize,
}

impl PreparedPitch {
    /// Reported latency for host delay compensation.
    #[inline]
    pub fn latency(&self) -> usize {
        self.latency
    }
}

/// The two interchangeable shifter voices for one channel plus which of them
/// currently carries the *incoming* (post-crossfade) preset ratio.
struct VoicePair {
    a: Box<dyn Shifter + Send>,
    b: Box<dyn Shifter + Send>,
    /// `true` while voice `a` is the current (incoming) voice; flipped on each
    /// preset change so the outgoing voice keeps its already-settled ratio.
    current_is_a: bool,
}

impl VoicePair {
    /// `(current, previous)` voices given the ping-pong flag.
    #[inline]
    fn split(&mut self) -> (&mut Box<dyn Shifter + Send>, &mut Box<dyn Shifter + Send>) {
        if self.current_is_a {
            (&mut self.a, &mut self.b)
        } else {
            (&mut self.b, &mut self.a)
        }
    }
}

/// Build one shifter for the given algorithm / quality.  **Allocates** —
/// control thread only.
fn build_voice(
    algorithm: Algorithm,
    quality: Quality,
    sample_rate: f32,
    max_block: usize,
    formant: Option<FormantMethod>,
) -> Box<dyn Shifter + Send> {
    match algorithm {
        Algorithm::Smooth => {
            let mut pv = PhaseVocoder::new(quality.frame(), max_block, WindowKind::Hann);
            // Formant preservation is a spectral-only feature.
            pv.set_formant_preservation(formant);
            Box::new(pv)
        }
        Algorithm::TightMono => Box::new(Wsola::new(quality.frame(), max_block)),
        Algorithm::MonoVoice => Box::new(Psola::new(
            sample_rate,
            MIN_PITCH_HZ,
            MAX_PITCH_HZ,
            max_block,
        )),
    }
}

/// Build a [`PreparedPitch`] — one [`VoicePair`] per channel.  **Allocates** —
/// control thread only.
fn build_prepared(
    algorithm: Algorithm,
    quality: Quality,
    formant: Option<FormantMethod>,
    sample_rate: f32,
    max_block: usize,
    channels: usize,
) -> PreparedPitch {
    let voices: Vec<VoicePair> = (0..channels)
        .map(|_| VoicePair {
            a: build_voice(algorithm, quality, sample_rate, max_block, formant),
            b: build_voice(algorithm, quality, sample_rate, max_block, formant),
            current_is_a: true,
        })
        .collect();
    // Every voice shares the same algorithm/quality, so any one reports the
    // group latency.
    let latency = voices[0].a.latency_samples();
    PreparedPitch { voices, latency }
}

/// The formant method to build, or `None` when preservation is disabled.
#[inline]
fn formant_choice(params: &PitchParams) -> Option<FormantMethod> {
    params.formant_preservation.then_some(params.formant_method)
}

/// Off-thread builder / publisher of [`PreparedPitch`] payloads.
pub struct PitchController {
    publisher: IrPublisher<PreparedPitch>,
    sample_rate: f32,
    max_block: usize,
    channels: usize,
    algorithm: Algorithm,
    quality: Quality,
    formant: Option<FormantMethod>,
    latency: usize,
}

impl PitchController {
    /// Create a controller + processor pair, publishing the first payload
    /// immediately so the processor can render on its first block.
    ///
    /// **Allocates** — call off the audio thread.
    pub fn new(
        max_block: usize,
        sample_rate: f32,
        channels: usize,
        params: &PitchParams,
    ) -> (Self, PitchProcessor) {
        assert!(channels > 0, "channels must be > 0");
        let max_block = max_block.max(1);

        let (mut publisher, consumer) = ir_handoff::<PreparedPitch>();
        let algorithm = params.algorithm;
        let quality = params.quality;
        let formant = formant_choice(params);

        let prepared = build_prepared(
            algorithm,
            quality,
            formant,
            sample_rate,
            max_block,
            channels,
        );
        let latency = prepared.latency;
        publisher.publish(prepared);

        let controller = Self {
            publisher,
            sample_rate,
            max_block,
            channels,
            algorithm,
            quality,
            formant,
            latency,
        };
        let processor = PitchProcessor::new(consumer, sample_rate, channels, max_block, params);
        (controller, processor)
    }

    /// Latency (samples) of the most recently published build.
    #[inline]
    pub fn latency(&self) -> usize {
        self.latency
    }

    /// Apply any structural change in `params` and republish if needed.
    /// Returns `true` when a rebuild happened (the caller may re-report
    /// latency).  **Allocates on rebuild** — control thread only.
    pub fn apply_structural(&mut self, params: &PitchParams) -> bool {
        let algorithm = params.algorithm;
        let quality = params.quality;
        let formant = formant_choice(params);
        if algorithm == self.algorithm && quality == self.quality && formant == self.formant {
            return false;
        }
        self.algorithm = algorithm;
        self.quality = quality;
        self.formant = formant;
        self.republish();
        true
    }

    fn republish(&mut self) {
        let prepared = build_prepared(
            self.algorithm,
            self.quality,
            self.formant,
            self.sample_rate,
            self.max_block,
            self.channels,
        );
        self.latency = prepared.latency;
        self.publisher.publish(prepared);
    }
}

/// Fixed-capacity single-channel dry-signal delay for wet/dry alignment.
///
/// Reads `latency` samples behind the write head; a `latency` of `0` is a
/// transparent pass-through.  Sized once (off-RT) to [`MAX_DELAY`]; every
/// operation on the audio thread is allocation-free.
struct DryDelay {
    buf: Vec<f32>,
    write: usize,
    latency: usize,
}

impl DryDelay {
    fn new() -> Self {
        Self {
            buf: vec![0.0; MAX_DELAY],
            write: 0,
            latency: 0,
        }
    }

    /// Set the current delay, clamped to the buffer capacity.
    #[inline]
    fn set_latency(&mut self, latency: usize) {
        self.latency = latency.min(self.buf.len() - 1);
    }

    /// Push `x` and return the sample written `latency` samples ago.
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        if self.latency == 0 {
            return x;
        }
        let cap = self.buf.len();
        let read = (self.write + cap - self.latency) % cap;
        let delayed = self.buf[read];
        self.buf[self.write] = x;
        self.write = (self.write + 1) % cap;
        delayed
    }

    fn reset(&mut self) {
        self.buf.fill(0.0);
        self.write = 0;
    }
}

/// Audio-thread renderer.  Polls the handoff for freshly-built voices and
/// applies the continuous parameters in place.
pub struct PitchProcessor {
    consumer: IrConsumer<PreparedPitch>,
    sample_rate: f32,
    /// One crossfading preset bank per channel.
    banks: Vec<PresetBank>,
    /// One dry-delay line per channel.
    dry_delays: Vec<DryDelay>,
    /// Scratch for the current voice's wet output.
    cur_scratch: Vec<f32>,
    /// Scratch for the previous (outgoing) voice's wet output.
    prev_scratch: Vec<f32>,

    // ── Change detection so cheap bank updates only happen on a real move. ──
    last_intervals: [rt_pitch_core::PresetSlot; NUM_PRESETS],
    last_crossfade_ms: f32,

    /// Latency of the last polled build (host-reportable).
    reported_latency: usize,
    /// Samples processed in the last `process` call (CPU-meter proxy).
    last_process_samples: usize,
}

impl PitchProcessor {
    fn new(
        consumer: IrConsumer<PreparedPitch>,
        sample_rate: f32,
        channels: usize,
        max_block: usize,
        params: &PitchParams,
    ) -> Self {
        let mut banks = Vec::with_capacity(channels);
        let mut dry_delays = Vec::with_capacity(channels);
        for _ in 0..channels {
            let mut bank = PresetBank::new(sample_rate, params.crossfade_ms);
            for (index, slot) in params.intervals.iter().enumerate() {
                bank.set_slot(index, slot.semitones, slot.enabled);
            }
            bank.snap_to(params.current_preset.min(NUM_PRESETS - 1));
            banks.push(bank);
            dry_delays.push(DryDelay::new());
        }

        Self {
            consumer,
            sample_rate,
            banks,
            dry_delays,
            cur_scratch: vec![0.0; max_block],
            prev_scratch: vec![0.0; max_block],
            last_intervals: params.intervals,
            last_crossfade_ms: params.crossfade_ms,
            reported_latency: 0,
            last_process_samples: 0,
        }
    }

    /// Latency (samples) reported by the last polled build — the plugin reads
    /// this after `process` to notify the host on change.
    #[inline]
    pub fn reported_latency(&self) -> usize {
        self.reported_latency
    }

    /// Samples rendered on the last [`process`](PitchProcessor::process) call.
    #[inline]
    pub fn last_process_samples(&self) -> usize {
        self.last_process_samples
    }

    /// Render `channels` in place using `params`.
    ///
    /// **Real-time safe** — no allocation, no locking (provided every channel
    /// length ≤ the `max_block` the processor was built with).
    pub fn process(&mut self, channels: &mut [&mut [f32]], params: &PitchParams) {
        let Some(prepared) = self.consumer.poll() else {
            self.last_process_samples = 0;
            return;
        };
        self.reported_latency = prepared.latency;
        let latency = if params.dry_delay_compensation {
            prepared.latency
        } else {
            0
        };

        // ── Cheap bank updates (only on an actual move). ──
        if self.last_intervals != params.intervals {
            for bank in &mut self.banks {
                for (index, slot) in params.intervals.iter().enumerate() {
                    bank.set_slot(index, slot.semitones, slot.enabled);
                }
            }
            self.last_intervals = params.intervals;
        }
        if self.last_crossfade_ms != params.crossfade_ms {
            for bank in &mut self.banks {
                bank.set_fade_time(self.sample_rate, params.crossfade_ms);
            }
            self.last_crossfade_ms = params.crossfade_ms;
        }

        let mix = params.mix.clamp(0.0, 1.0);
        let mut rendered = 0;

        for (channel_index, channel) in channels.iter_mut().enumerate() {
            let Some(pair) = prepared.voices.get_mut(channel_index) else {
                continue;
            };
            let bank = &mut self.banks[channel_index];

            // Voice selection: channel 1 follows voice 2 in dual-voice mode.
            let target = if params.dual_voice && channel_index == 1 {
                params.voice2_preset
            } else {
                params.current_preset
            }
            .min(NUM_PRESETS - 1);

            // A change begins a fresh equal-power fade; keep the outgoing
            // voice on its settled ratio by flipping the ping-pong.
            let before = bank.current_index();
            bank.select(target);
            if bank.current_index() != before {
                pair.current_is_a = !pair.current_is_a;
            }

            let block = channel.len().min(self.cur_scratch.len());

            if params.bypass {
                rendered = rendered.max(block);
                continue;
            }

            // Render both voices (kept warm) from the same input block.
            let (cur_voice, prev_voice) = pair.split();
            cur_voice.set_ratio(bank.current_ratio());
            prev_voice.set_ratio(bank.previous_ratio());
            cur_voice.process_block(&channel[..block], &mut self.cur_scratch[..block]);
            prev_voice.process_block(&channel[..block], &mut self.prev_scratch[..block]);

            // Equal-power blend + dry/wet mix, sample by sample.
            let delay = &mut self.dry_delays[channel_index];
            delay.set_latency(latency);
            for sample in 0..block {
                let (gain_prev, gain_cur) = bank.next_gains();
                let wet =
                    gain_prev * self.prev_scratch[sample] + gain_cur * self.cur_scratch[sample];
                let dry = delay.process(channel[sample]);
                channel[sample] = dry * (1.0 - mix) + wet * mix;
            }
            rendered = rendered.max(block);
        }

        self.last_process_samples = rendered;
    }

    /// Flush every voice and bank back to silence (transport stop).
    pub fn reset(&mut self, params: &PitchParams) {
        if let Some(prepared) = self.consumer.current() {
            for pair in &mut prepared.voices {
                pair.a.reset();
                pair.b.reset();
                pair.current_is_a = true;
            }
        }
        for bank in &mut self.banks {
            bank.snap_to(params.current_preset.min(NUM_PRESETS - 1));
        }
        for delay in &mut self.dry_delays {
            delay.reset();
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f32 = 48_000.0;

    fn engine(channels: usize, params: &PitchParams) -> (PitchController, PitchProcessor) {
        PitchController::new(256, SAMPLE_RATE, channels, params)
    }

    /// Feed `blocks` blocks of a `freq`-Hz sine and return the concatenated
    /// output past the warm-up region.
    fn run_sine(
        processor: &mut PitchProcessor,
        params: &PitchParams,
        freq: f32,
        blocks: usize,
        block: usize,
    ) -> Vec<f32> {
        let mut out = Vec::with_capacity(blocks * block);
        for block_index in 0..blocks {
            let mut samples: Vec<f32> = (0..block)
                .map(|n| {
                    let t = (block_index * block + n) as f32 / SAMPLE_RATE;
                    0.5 * (std::f32::consts::TAU * freq * t).sin()
                })
                .collect();
            {
                let mut channels: [&mut [f32]; 1] = [&mut samples];
                processor.process(&mut channels, params);
            }
            out.extend_from_slice(&samples);
        }
        out
    }

    /// Estimate the dominant frequency of `signal` by counting positive
    /// zero-crossings (adequate for a strongly periodic tone).
    fn dominant_hz(signal: &[f32]) -> f32 {
        let mut crossings = 0usize;
        for pair in signal.windows(2) {
            if pair[0] <= 0.0 && pair[1] > 0.0 {
                crossings += 1;
            }
        }
        crossings as f32 * SAMPLE_RATE / signal.len() as f32
    }

    #[test]
    fn builds_and_renders_stereo() {
        let params = PitchParams::default();
        let (_controller, mut processor) = engine(2, &params);
        let mut left = vec![0.1f32; 256];
        let mut right = vec![0.1f32; 256];
        {
            let mut channels: [&mut [f32]; 2] = [&mut left, &mut right];
            processor.process(&mut channels, &params);
        }
        assert!(left.iter().all(|s| s.is_finite()));
        assert!(right.iter().all(|s| s.is_finite()));
        assert_eq!(processor.last_process_samples(), 256);
    }

    #[test]
    fn bypass_is_transparent() {
        let params = PitchParams {
            bypass: true,
            ..Default::default()
        };
        let (_controller, mut processor) = engine(1, &params);
        let original: Vec<f32> = (0..256).map(|n| (n as f32 * 0.1).sin()).collect();
        let mut samples = original.clone();
        {
            let mut channels: [&mut [f32]; 1] = [&mut samples];
            processor.process(&mut channels, &params);
        }
        for (out, orig) in samples.iter().zip(original.iter()) {
            assert!((out - orig).abs() < 1e-6, "bypass altered the signal");
        }
    }

    #[test]
    fn unison_preserves_frequency() {
        // Preset 0 is unison (ratio 1.0); the output tone matches the input.
        let params = PitchParams {
            dry_delay_compensation: false,
            mix: 1.0,
            ..Default::default()
        };
        let (_controller, mut processor) = engine(1, &params);
        let out = run_sine(&mut processor, &params, 220.0, 40, 256);
        // Skip the warm-up; measure the settled tail.
        let tail = &out[out.len() / 2..];
        let hz = dominant_hz(tail);
        assert!((hz - 220.0).abs() < 15.0, "unison frequency drifted: {hz}");
    }

    #[test]
    fn octave_up_doubles_frequency() {
        // Select preset 1 (+12 semitones → 2×).
        let params = PitchParams {
            current_preset: 1,
            crossfade_ms: 5.0,
            dry_delay_compensation: false,
            mix: 1.0,
            ..Default::default()
        };
        let (_controller, mut processor) = engine(1, &params);
        let out = run_sine(&mut processor, &params, 220.0, 80, 256);
        let tail = &out[out.len() / 2..];
        let hz = dominant_hz(tail);
        assert!(
            (hz - 440.0).abs() < 30.0,
            "octave-up should land near 440 Hz, got {hz}"
        );
    }

    #[test]
    fn mix_zero_is_dry() {
        // With no wet, the octave-up preset leaves the tone at its input pitch.
        let params = PitchParams {
            current_preset: 1,
            mix: 0.0,
            dry_delay_compensation: false,
            ..Default::default()
        };
        let (_controller, mut processor) = engine(1, &params);
        let original: Vec<f32> = (0..256).map(|n| 0.5 * (n as f32 * 0.05).sin()).collect();
        let mut samples = original.clone();
        {
            let mut channels: [&mut [f32]; 1] = [&mut samples];
            processor.process(&mut channels, &params);
        }
        for (out, orig) in samples.iter().zip(original.iter()) {
            assert!((out - orig).abs() < 1e-6, "mix=0 should be dry");
        }
    }

    #[test]
    fn preset_change_is_click_free() {
        // Switch preset mid-stream and assert no sample-to-sample jump exceeds a
        // conservative slew (−? dBFS): the equal-power crossfade is smooth.
        let mut params = PitchParams {
            current_preset: 0,
            crossfade_ms: 50.0,
            dry_delay_compensation: false,
            ..Default::default()
        };
        let (_controller, mut processor) = engine(1, &params);
        // Warm up on unison.
        let _ = run_sine(&mut processor, &params, 220.0, 20, 256);
        // Switch to the fifth and capture the transition.
        params.current_preset = 3;
        let out = run_sine(&mut processor, &params, 220.0, 40, 256);
        let max_step = out
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_step < 0.3,
            "crossfade produced a discontinuity of {max_step}"
        );
    }

    #[test]
    fn structural_change_rebuilds_and_updates_latency() {
        let params = PitchParams::default();
        let (mut controller, mut processor) = engine(1, &params);

        // Prime the processor's reported latency (Standard / Smooth).
        let mut warm = vec![0.0f32; 256];
        {
            let mut channels: [&mut [f32]; 1] = [&mut warm];
            processor.process(&mut channels, &params);
        }
        let live_latency = processor.reported_latency();

        // Switch to Studio quality: larger frame ⇒ larger latency.
        let studio = PitchParams {
            quality: Quality::Studio,
            ..Default::default()
        };
        assert!(controller.apply_structural(&studio));
        let mut block = vec![0.0f32; 256];
        {
            let mut channels: [&mut [f32]; 1] = [&mut block];
            processor.process(&mut channels, &studio);
        }
        assert!(
            processor.reported_latency() > live_latency,
            "studio latency {} should exceed standard {}",
            processor.reported_latency(),
            live_latency
        );
        assert_eq!(processor.reported_latency(), controller.latency());
    }

    #[test]
    fn dual_voice_shifts_channels_independently() {
        let params = PitchParams {
            dual_voice: true,
            current_preset: 1, // octave up on the left
            voice2_preset: 2,  // octave down on the right
            crossfade_ms: 5.0,
            dry_delay_compensation: false,
            ..Default::default()
        };
        let (_controller, mut processor) = engine(2, &params);

        let mut left = Vec::new();
        let mut right = Vec::new();
        for block_index in 0..80 {
            let make = |n: usize| {
                let t = (block_index * 256 + n) as f32 / SAMPLE_RATE;
                0.5 * (std::f32::consts::TAU * 220.0 * t).sin()
            };
            let mut l: Vec<f32> = (0..256).map(make).collect();
            let mut r: Vec<f32> = (0..256).map(make).collect();
            {
                let mut channels: [&mut [f32]; 2] = [&mut l, &mut r];
                processor.process(&mut channels, &params);
            }
            left.extend_from_slice(&l);
            right.extend_from_slice(&r);
        }
        let left_hz = dominant_hz(&left[left.len() / 2..]);
        let right_hz = dominant_hz(&right[right.len() / 2..]);
        assert!(
            left_hz > right_hz * 1.5,
            "L {left_hz} should sit above R {right_hz}"
        );
    }

    #[test]
    fn reset_clears_output_tail() {
        let params = PitchParams {
            current_preset: 1,
            dry_delay_compensation: false,
            ..Default::default()
        };
        let (_controller, mut processor) = engine(1, &params);
        let _ = run_sine(&mut processor, &params, 220.0, 20, 256);
        processor.reset(&params);
        // First block after reset over silence must be silent (no leftover tail
        // beyond the warm-up latency being re-primed with zeros).
        let mut silence = vec![0.0f32; 256];
        {
            let mut channels: [&mut [f32]; 1] = [&mut silence];
            processor.process(&mut channels, &params);
        }
        assert!(
            silence.iter().all(|s| s.abs() < 1e-3),
            "reset left an audible tail"
        );
    }
}
