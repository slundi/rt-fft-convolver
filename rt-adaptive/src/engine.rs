//! Framework-agnostic adaptive engine: an off-thread [`AdaptiveController`] and
//! an audio-thread [`AdaptiveProcessor`] joined by the lock-free handoff from
//! `rt-ir-plugin-core`.
//!
//! # Threading
//!
//! - [`AdaptiveController`] (control / background thread) rebuilds the
//!   [`EchoBank`] when the **AEC filter length** changes — the one structural
//!   control (it reallocates the tap vectors) — and publishes the fresh bank
//!   through the wait-free handoff.
//! - [`AdaptiveProcessor`] (audio thread) keeps the feedback detector, the
//!   per-channel notch banks, and the per-channel hum trackers **resident**
//!   (the audio thread mutates their state — placed notches, tracked drift — so
//!   a rebuild must never drop it), polls the handoff for the current echo bank,
//!   runs whichever engine the [`Mode`] selects, and applies output gain —
//!   allocation-free.
//!
//! # Mode routing
//!
//! - **Feedback** — the detector runs on channel 0 (feedback is a property of
//!   the room, identical across channels); a confirmed partial is notched on
//!   every channel's bank, kept in lock-step.
//! - **Hum** — each channel has its own tracker/comb (they converge on the same
//!   mains fundamental) so stereo notch state stays independent.
//! - **Echo** — a shared mono far-end reference (sidechain) drives one adaptive
//!   filter per channel; a per-channel double-talk detector freezes adaptation
//!   when the near-end talker dominates.

use rt_adaptive_core::{AdaptiveFilter, FeedbackDetector, HumTracker, NotchBank};
use rt_fft_convolver::{DenormalGuard, db_to_linear};
use rt_ir_plugin_core::{IrConsumer, IrPublisher, ir_handoff};

use crate::params::{AdaptiveParams, FilterLength, Mode};

/// Analysis-FFT size for the feedback detector (Roadmap-09 "e.g. 512").
const DETECTOR_FFT: usize = 512;

/// Root-mean-square level of a slice (block double-talk / energy metric).
fn rms(buf: &[f32]) -> f32 {
    if buf.is_empty() {
        return 0.0;
    }
    (buf.iter().map(|s| s * s).sum::<f32>() / buf.len() as f32).sqrt()
}

/// One echo canceller of a fixed tap count, reusing the core NLMS primitive.
struct Aec<const N: usize> {
    filter: AdaptiveFilter<N>,
}

impl<const N: usize> Aec<N> {
    fn new(mu: f32) -> Self {
        Self {
            filter: AdaptiveFilter::nlms(mu),
        }
    }

    /// Cancel one sample: estimate the echo of `reference`, subtract it from the
    /// near-end `mic`, and adapt toward the residual unless `adapt` is `false`
    /// (double-talk freeze).  Returns the echo-cancelled residual.
    #[inline]
    fn step(&mut self, reference: f32, mic: f32, adapt: bool) -> f32 {
        let echo = self.filter.predict(reference);
        let residual = mic - echo;
        if adapt {
            self.filter.adapt(residual);
        }
        residual
    }

    #[inline]
    fn set_step(&mut self, mu: f32) {
        self.filter.set_step(mu);
    }

    fn reset(&mut self) {
        self.filter.reset();
    }
}

/// An echo canceller sized to a discrete [`FilterLength`] (dispatched so the
/// const-generic [`AdaptiveFilter`] can be reused at each size).
///
/// Each variant is boxed so the enum stays pointer-sized rather than as large
/// as the 2048-tap array; the box is allocated off the audio thread (in
/// [`EchoBank::new`], control-thread only).
enum EchoCanceller {
    L256(Box<Aec<256>>),
    L512(Box<Aec<512>>),
    L1024(Box<Aec<1024>>),
    L2048(Box<Aec<2048>>),
}

impl EchoCanceller {
    fn new(length: FilterLength, mu: f32) -> Self {
        match length {
            FilterLength::L256 => Self::L256(Box::new(Aec::new(mu))),
            FilterLength::L512 => Self::L512(Box::new(Aec::new(mu))),
            FilterLength::L1024 => Self::L1024(Box::new(Aec::new(mu))),
            FilterLength::L2048 => Self::L2048(Box::new(Aec::new(mu))),
        }
    }

    #[inline]
    fn step(&mut self, reference: f32, mic: f32, adapt: bool) -> f32 {
        match self {
            Self::L256(aec) => aec.step(reference, mic, adapt),
            Self::L512(aec) => aec.step(reference, mic, adapt),
            Self::L1024(aec) => aec.step(reference, mic, adapt),
            Self::L2048(aec) => aec.step(reference, mic, adapt),
        }
    }

    #[inline]
    fn set_step(&mut self, mu: f32) {
        match self {
            Self::L256(aec) => aec.set_step(mu),
            Self::L512(aec) => aec.set_step(mu),
            Self::L1024(aec) => aec.set_step(mu),
            Self::L2048(aec) => aec.set_step(mu),
        }
    }

    fn reset(&mut self) {
        match self {
            Self::L256(aec) => aec.reset(),
            Self::L512(aec) => aec.reset(),
            Self::L1024(aec) => aec.reset(),
            Self::L2048(aec) => aec.reset(),
        }
    }
}

/// One echo canceller per channel — the handoff payload rebuilt on a filter
/// length change.
pub struct EchoBank {
    channels: Vec<EchoCanceller>,
}

impl EchoBank {
    /// **Allocates** — control thread only.
    fn new(length: FilterLength, mu: f32, channels: usize) -> Self {
        Self {
            channels: (0..channels)
                .map(|_| EchoCanceller::new(length, mu))
                .collect(),
        }
    }

    fn set_step(&mut self, mu: f32) {
        for canceller in &mut self.channels {
            canceller.set_step(mu);
        }
    }

    fn reset(&mut self) {
        for canceller in &mut self.channels {
            canceller.reset();
        }
    }
}

/// Off-thread builder / publisher of the current [`EchoBank`].
pub struct AdaptiveController {
    publisher: IrPublisher<EchoBank>,
    channels: usize,
    /// Filter length of the currently published bank.
    length: FilterLength,
}

impl AdaptiveController {
    /// Create a controller + processor pair, publishing the first echo bank so
    /// the processor can render on its first block.
    ///
    /// **Allocates** — call off the audio thread.
    ///
    /// # Panics
    /// Panics if `channels == 0`.
    pub fn new(
        max_block: usize,
        sample_rate: f32,
        channels: usize,
        params: &AdaptiveParams,
    ) -> (Self, AdaptiveProcessor) {
        assert!(channels > 0, "channels must be > 0");
        let max_block = max_block.max(1);

        let (mut publisher, consumer) = ir_handoff::<EchoBank>();
        publisher.publish(EchoBank::new(
            params.filter_length,
            params.convergence,
            channels,
        ));

        let controller = Self {
            publisher,
            channels,
            length: params.filter_length,
        };
        let processor = AdaptiveProcessor::new(consumer, sample_rate, channels, max_block, params);
        (controller, processor)
    }

    /// Rebuild + republish the echo bank if the filter length changed.  Returns
    /// `true` on a rebuild.  **Allocates on rebuild** — control thread only.
    pub fn apply_structural(&mut self, params: &AdaptiveParams) -> bool {
        if params.filter_length == self.length {
            return false;
        }
        self.length = params.filter_length;
        self.publisher.publish(EchoBank::new(
            params.filter_length,
            params.convergence,
            self.channels,
        ));
        true
    }
}

/// Audio-thread renderer: runs the active mode and applies output gain.
pub struct AdaptiveProcessor {
    consumer: IrConsumer<EchoBank>,
    /// Single detector (feedback is a room property, shared across channels).
    detector: FeedbackDetector,
    /// One notch bank per channel, kept structurally in lock-step.
    notch_banks: Vec<NotchBank>,
    /// One hum tracker/comb per channel.
    hum: Vec<HumTracker>,
    /// Per-channel input snapshot (dry reference for AEC / detection).
    input_copies: Vec<Vec<f32>>,

    /// Last mains fundamental pushed to the trackers (so drift isn't reset).
    last_hum_fundamental: f32,
    /// Previous `reset` flag, for rising-edge detection.
    last_reset: bool,

    // ── Telemetry (audio thread → editor). ──
    active_notches: usize,
    tracked_hum_hz: f32,
    feedback_events: u64,
}

impl AdaptiveProcessor {
    fn new(
        consumer: IrConsumer<EchoBank>,
        sample_rate: f32,
        channels: usize,
        max_block: usize,
        params: &AdaptiveParams,
    ) -> Self {
        let detector = FeedbackDetector::new(sample_rate, DETECTOR_FFT);
        let notch_banks = (0..channels)
            .map(|_| NotchBank::new(sample_rate as u32))
            .collect();
        let hum = (0..channels)
            .map(|_| HumTracker::new(sample_rate as u32, params.hum_fundamental_hz))
            .collect();
        Self {
            consumer,
            detector,
            notch_banks,
            hum,
            input_copies: (0..channels).map(|_| vec![0.0; max_block]).collect(),
            last_hum_fundamental: params.hum_fundamental_hz,
            last_reset: false,
            active_notches: 0,
            tracked_hum_hz: params.hum_fundamental_hz,
            feedback_events: 0,
        }
    }

    /// Reported latency in samples — every mode here is zero-latency (notches
    /// and time-domain NLMS add none to the direct path).
    #[inline]
    pub fn latency_samples(&self) -> usize {
        0
    }

    /// Number of active feedback notches (editor readout).
    #[inline]
    pub fn active_notches(&self) -> usize {
        self.active_notches
    }

    /// The mains fundamental currently driving the hum comb (editor readout).
    #[inline]
    pub fn tracked_hum_hz(&self) -> f32 {
        self.tracked_hum_hz
    }

    /// Monotonic count of feedback notches placed since construction — the
    /// editor flashes its detection LED when this increments.
    #[inline]
    pub fn feedback_events(&self) -> u64 {
        self.feedback_events
    }

    /// Copy the detector's latest magnitude spectrum into `out` (one value per
    /// display bin) for the editor's spectrum view.  Allocation-free.
    pub fn probe_spectrum(&self, out: &mut [f32]) {
        crate::ui::downsample_spectrum_into(
            self.detector.magnitude(),
            self.detector.bin_width_hz(),
            out,
        );
    }

    /// Copy the active notches (channel 0) into `freqs` / `ages`, zero-filling
    /// unused slots — for the editor's inventory list and highlight markers.
    /// Allocation-free.
    pub fn copy_notches(&self, freqs: &mut [f32], ages: &mut [u64]) {
        for slot in freqs.iter_mut() {
            *slot = 0.0;
        }
        for slot in ages.iter_mut() {
            *slot = 0;
        }
        for (index, (_, info)) in self.notch_banks[0].active().enumerate() {
            if index >= freqs.len() {
                break;
            }
            freqs[index] = info.freq_hz;
            if index < ages.len() {
                ages[index] = info.age_samples;
            }
        }
    }

    /// Render `main` in place using `params`; `sidechain` is the (mono) far-end
    /// reference for AEC (ignored by the other modes).
    ///
    /// **Real-time safe** — no allocation, no locking (provided every channel
    /// length ≤ the `max_block` the processor was built with).
    pub fn process(
        &mut self,
        main: &mut [&mut [f32]],
        sidechain: Option<&[f32]>,
        params: &AdaptiveParams,
    ) {
        let block = main.first().map_or(0, |channel| channel.len());
        if block == 0 {
            return;
        }
        let _guard = DenormalGuard::new();

        // Snapshot the input so the AEC reference / detector survive in-place writes.
        for (ch, channel) in main.iter().enumerate() {
            let len = channel.len().min(self.input_copies[ch].len());
            self.input_copies[ch][..len].copy_from_slice(&channel[..len]);
        }

        if params.bypass {
            return; // output already equals the untouched input
        }

        // Rising edge on `reset` clears every notch.
        if params.reset && !self.last_reset {
            for bank in &mut self.notch_banks {
                bank.reset();
            }
        }
        self.last_reset = params.reset;

        let gain = db_to_linear(params.output_gain_db);

        match params.mode {
            Mode::FeedbackSuppressor => self.process_feedback(main, params, gain, block),
            Mode::HumRemover => self.process_hum(main, params, gain, block),
            Mode::EchoCanceller => self.process_echo(main, sidechain, params, gain, block),
        }
    }

    fn process_feedback(
        &mut self,
        main: &mut [&mut [f32]],
        params: &AdaptiveParams,
        gain: f32,
        block: usize,
    ) {
        // Always analyse channel 0 so the editor's spectrum view and the
        // peak-hold stay live; only *allocate* notches when auto detection is on
        // and not frozen.  A confirmed partial is mirrored onto every channel.
        self.detector.set_sensitivity(params.sensitivity);
        let detected = self.detector.process(&self.input_copies[0][..block]);
        if params.auto && !params.freeze {
            for bank in &mut self.notch_banks {
                bank.set_max_active(params.max_notches);
            }
            if let Some(freq) = detected {
                let mut placed_new = false;
                for bank in &mut self.notch_banks {
                    let before = bank.active_count();
                    let _ = bank.allocate(freq, params.notch_q);
                    placed_new |= bank.active_count() > before;
                }
                if placed_new {
                    self.feedback_events += 1;
                }
            }
        }

        for (ch, channel) in main.iter_mut().enumerate() {
            self.notch_banks[ch].process_block(&mut channel[..block]);
            for sample in channel.iter_mut().take(block) {
                *sample *= gain;
            }
        }
        self.active_notches = self.notch_banks[0].active_count();
    }

    fn process_hum(
        &mut self,
        main: &mut [&mut [f32]],
        params: &AdaptiveParams,
        gain: f32,
        block: usize,
    ) {
        // Re-pin the nominal fundamental only when it actually changes, so
        // adaptive drift tracking isn't reset every block.
        if params.hum_fundamental_hz != self.last_hum_fundamental {
            self.last_hum_fundamental = params.hum_fundamental_hz;
            for tracker in &mut self.hum {
                tracker.set_fundamental(params.hum_fundamental_hz);
            }
        }

        for (ch, channel) in main.iter_mut().enumerate() {
            let tracker = &mut self.hum[ch];
            tracker.set_enabled(true);
            tracker.set_adaptive(params.hum_adaptive);
            tracker.set_harmonics(params.hum_harmonics);
            tracker.set_q(params.hum_q);
            tracker.process_block(&mut channel[..block]);
            for sample in channel.iter_mut().take(block) {
                *sample *= gain;
            }
        }
        self.tracked_hum_hz = self.hum[0].tracked_hz();
    }

    fn process_echo(
        &mut self,
        main: &mut [&mut [f32]],
        sidechain: Option<&[f32]>,
        params: &AdaptiveParams,
        gain: f32,
        block: usize,
    ) {
        let Some(bank) = self.consumer.poll() else {
            return;
        };
        bank.set_step(params.convergence);

        // No far-end reference patched → nothing to cancel; pass through + gain.
        let reference = match sidechain {
            Some(sc) if sc.len() >= block => &sc[..block],
            _ => {
                for channel in main.iter_mut() {
                    for sample in channel.iter_mut().take(block) {
                        *sample *= gain;
                    }
                }
                return;
            }
        };

        let threshold = db_to_linear(params.double_talk_db);
        let ref_rms = rms(reference);

        for (ch, channel) in main.iter_mut().enumerate() {
            let mic = &self.input_copies[ch][..block];
            // Double-talk: freeze adaptation when the near-end exceeds what the
            // far-end echo could plausibly produce (or the reference is silent).
            let adapt = ref_rms > 1e-6 && rms(mic) <= ref_rms * threshold;
            let canceller = &mut bank.channels[ch];
            for (sample, (&far, &near)) in channel
                .iter_mut()
                .take(block)
                .zip(reference.iter().zip(mic.iter()))
            {
                *sample = canceller.step(far, near, adapt) * gain;
            }
        }
    }

    /// Flush every engine and telemetry back to silence.
    pub fn reset(&mut self) {
        self.detector.reset();
        for bank in &mut self.notch_banks {
            bank.reset();
        }
        for tracker in &mut self.hum {
            tracker.reset();
        }
        if let Some(bank) = self.consumer.current() {
            bank.reset();
        }
        for copy in &mut self.input_copies {
            copy.fill(0.0);
        }
        self.last_reset = false;
        self.active_notches = 0;
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    const SAMPLE_RATE: f32 = 48_000.0;
    const BLOCK: usize = 256;

    fn engine(channels: usize, params: &AdaptiveParams) -> (AdaptiveController, AdaptiveProcessor) {
        AdaptiveController::new(BLOCK, SAMPLE_RATE, channels, params)
    }

    fn sine(freq: f32, samples: usize) -> Vec<f32> {
        (0..samples)
            .map(|n| (TAU * freq * n as f32 / SAMPLE_RATE).sin())
            .collect()
    }

    fn noise(len: usize, amplitude: f32) -> Vec<f32> {
        let mut state: u32 = 0x1234_5678;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                ((state as f32 / u32::MAX as f32) * 2.0 - 1.0) * amplitude
            })
            .collect()
    }

    /// Run a mono signal (with optional sidechain) block by block.
    fn run_mono(
        processor: &mut AdaptiveProcessor,
        params: &AdaptiveParams,
        input: &[f32],
        sidechain: Option<&[f32]>,
    ) -> Vec<f32> {
        let mut out = Vec::with_capacity(input.len());
        let mut offset = 0;
        for chunk in input.chunks(BLOCK) {
            let mut buffer = chunk.to_vec();
            let sc = sidechain.map(|s| s[offset..offset + chunk.len()].to_vec());
            {
                let mut main: [&mut [f32]; 1] = [&mut buffer];
                processor.process(&mut main, sc.as_deref(), params);
            }
            out.extend_from_slice(&buffer);
            offset += chunk.len();
        }
        out
    }

    #[test]
    fn builds_and_renders_finite() {
        let params = AdaptiveParams::default();
        let (_controller, mut processor) = engine(2, &params);
        let mut left = vec![0.1f32; BLOCK];
        let mut right = vec![0.1f32; BLOCK];
        {
            let mut main: [&mut [f32]; 2] = [&mut left, &mut right];
            processor.process(&mut main, None, &params);
        }
        assert!(left.iter().all(|s| s.is_finite()));
        assert!(right.iter().all(|s| s.is_finite()));
        assert_eq!(processor.latency_samples(), 0);
    }

    #[test]
    fn bypass_is_transparent() {
        let params = AdaptiveParams {
            bypass: true,
            ..Default::default()
        };
        let (_controller, mut processor) = engine(1, &params);
        let original = sine(1_000.0, BLOCK);
        let mut samples = original.clone();
        {
            let mut main: [&mut [f32]; 1] = [&mut samples];
            processor.process(&mut main, None, &params);
        }
        for (out, orig) in samples.iter().zip(&original) {
            assert!((out - orig).abs() < 1e-6, "bypass altered the signal");
        }
    }

    #[test]
    fn feedback_notches_a_sustained_howl() {
        let params = AdaptiveParams {
            mode: Mode::FeedbackSuppressor,
            sensitivity: 0.7,
            notch_q: 20.0,
            ..Default::default()
        };
        let (_controller, mut processor) = engine(1, &params);
        // A strong 2 kHz howl over a quiet floor.
        let tone = sine(2_000.0, 48_000);
        let floor = noise(tone.len(), 0.02);
        let input: Vec<f32> = tone.iter().zip(&floor).map(|(t, n)| t + n).collect();
        let out = run_mono(&mut processor, &params, &input, None);

        // A notch should have been placed and the howl attenuated by the tail.
        assert!(processor.active_notches() >= 1, "no notch placed");
        assert!(processor.feedback_events() >= 1);
        let head = rms(&out[..4096]);
        let tail = rms(&out[out.len() - 4096..]);
        assert!(tail < head * 0.5, "howl not suppressed: {head} → {tail}");
    }

    #[test]
    fn freeze_blocks_new_notches() {
        let params = AdaptiveParams {
            mode: Mode::FeedbackSuppressor,
            freeze: true,
            ..Default::default()
        };
        let (_controller, mut processor) = engine(1, &params);
        let input = sine(2_000.0, 48_000);
        let _ = run_mono(&mut processor, &params, &input, None);
        assert_eq!(processor.active_notches(), 0, "freeze let a notch through");
    }

    #[test]
    fn hum_mode_removes_mains() {
        let params = AdaptiveParams {
            mode: Mode::HumRemover,
            hum_fundamental_hz: 60.0,
            hum_adaptive: false,
            hum_harmonics: 3,
            ..Default::default()
        };
        let (_controller, mut processor) = engine(1, &params);
        let input = sine(60.0, 48_000);
        let out = run_mono(&mut processor, &params, &input, None);
        // Judge the settled tail (skip the notch warm-up).
        let tail = rms(&out[24_000..]);
        assert!(tail < 0.1, "hum not removed: {tail}");
    }

    #[test]
    fn echo_canceller_converges() {
        let params = AdaptiveParams {
            mode: Mode::EchoCanceller,
            convergence: 0.5,
            double_talk_db: 24.0, // permissive so adaptation stays enabled
            ..Default::default()
        };
        let (_controller, mut processor) = engine(1, &params);

        // Far-end reference and its echo through a short FIR path.
        let reference = noise(48_000, 0.5);
        let echo_path = [0.5f32, -0.25, 0.1];
        let mic: Vec<f32> = (0..reference.len())
            .map(|i| {
                echo_path
                    .iter()
                    .enumerate()
                    .map(|(k, h)| if i >= k { h * reference[i - k] } else { 0.0 })
                    .sum()
            })
            .collect();

        let out = run_mono(&mut processor, &params, &mic, Some(&reference));
        // The residual echo should be far quieter than the raw mic by the tail.
        let raw = rms(&mic[mic.len() - 8192..]);
        let residual = rms(&out[out.len() - 8192..]);
        assert!(
            residual < raw * 0.1,
            "echo not cancelled: {raw} → {residual}"
        );
    }

    #[test]
    fn double_talk_passes_near_end() {
        // Silent far-end + a near-end talker → adaptation frozen, voice passes.
        let params = AdaptiveParams {
            mode: Mode::EchoCanceller,
            double_talk_db: 6.0,
            ..Default::default()
        };
        let (_controller, mut processor) = engine(1, &params);
        let reference = vec![0.0f32; 48_000]; // far-end silent
        let near_end = sine(300.0, 48_000);
        let out = run_mono(&mut processor, &params, &near_end, Some(&reference));
        // With no reference to model, the near-end must survive intact.
        assert!(
            rms(&out[out.len() - 8192..]) > 0.5,
            "near-end voice was cancelled"
        );
    }

    #[test]
    fn output_gain_scales_dry_signal() {
        // Feedback mode, frozen so no notches → output = input × gain.
        let params = AdaptiveParams {
            mode: Mode::FeedbackSuppressor,
            freeze: true,
            output_gain_db: 6.0,
            ..Default::default()
        };
        let (_controller, mut processor) = engine(1, &params);
        let original = sine(1_000.0, BLOCK);
        let mut samples = original.clone();
        {
            let mut main: [&mut [f32]; 1] = [&mut samples];
            processor.process(&mut main, None, &params);
        }
        let expected = db_to_linear(6.0);
        for (out, orig) in samples.iter().zip(&original) {
            assert!((out - orig * expected).abs() < 1e-5, "gain not applied");
        }
    }

    #[test]
    fn reset_edge_clears_notches() {
        let mut params = AdaptiveParams {
            mode: Mode::FeedbackSuppressor,
            sensitivity: 0.8,
            ..Default::default()
        };
        let (_controller, mut processor) = engine(1, &params);
        let _ = run_mono(&mut processor, &params, &sine(2_000.0, 48_000), None);
        assert!(processor.active_notches() >= 1);

        // Rising edge on reset clears the bank.
        params.reset = true;
        let mut buf = vec![0.0f32; BLOCK];
        {
            let mut main: [&mut [f32]; 1] = [&mut buf];
            processor.process(&mut main, None, &params);
        }
        assert_eq!(processor.active_notches(), 0, "reset did not clear notches");
    }

    #[test]
    fn structural_change_rebuilds_echo_bank() {
        let params = AdaptiveParams::default();
        let (mut controller, mut processor) = engine(1, &params);
        assert!(!controller.apply_structural(&params));

        let longer = AdaptiveParams {
            mode: Mode::EchoCanceller,
            filter_length: FilterLength::L2048,
            ..Default::default()
        };
        assert!(controller.apply_structural(&longer));
        let reference = noise(4096, 0.4);
        let out = run_mono(
            &mut processor,
            &longer,
            &reference.clone(),
            Some(&reference),
        );
        assert!(out.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn editor_probes_expose_spectrum_and_notches() {
        let params = AdaptiveParams {
            mode: Mode::FeedbackSuppressor,
            sensitivity: 0.8,
            ..Default::default()
        };
        let (_controller, mut processor) = engine(1, &params);
        let _ = run_mono(&mut processor, &params, &sine(2_000.0, 48_000), None);

        // Spectrum probe: filled, finite, and non-trivial after a strong tone.
        let mut spectrum = vec![0.0f32; crate::ui::SPECTRUM_DISPLAY_BINS];
        processor.probe_spectrum(&mut spectrum);
        assert!(spectrum.iter().all(|v| v.is_finite()));
        assert!(spectrum.iter().any(|&v| v > 0.0), "spectrum probe empty");

        // Notch probe: at least one notch was placed at ~2 kHz.
        let mut freqs = vec![0.0f32; rt_adaptive_core::MAX_NOTCHES];
        let mut ages = vec![0u64; rt_adaptive_core::MAX_NOTCHES];
        processor.copy_notches(&mut freqs, &mut ages);
        assert!(
            freqs.iter().any(|&f| (f - 2_000.0).abs() < 150.0),
            "no 2 kHz notch reported"
        );
    }

    #[test]
    fn reset_clears_tail() {
        let params = AdaptiveParams {
            mode: Mode::HumRemover,
            ..Default::default()
        };
        let (_controller, mut processor) = engine(1, &params);
        let _ = run_mono(&mut processor, &params, &sine(50.0, 8192), None);
        processor.reset();
        let out = run_mono(&mut processor, &params, &vec![0.0f32; BLOCK * 2], None);
        assert!(rms(&out) < 1e-3, "reset left a tail: {}", rms(&out));
    }
}
