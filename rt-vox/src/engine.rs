//! Framework-agnostic vox engine: an off-thread [`VoxController`] and an
//! audio-thread [`VoxProcessor`] joined by the lock-free handoff from
//! `rt-ir-plugin-core`.
//!
//! # Threading
//!
//! - [`VoxController`] (control / background thread) rebuilds the
//!   [`Vocoder`](rt_vox_core::Vocoder) when the **band count** changes — the one
//!   structural control (it reallocates the log-band table and FFT buffers).
//!   It publishes the fresh vocoder through the wait-free handoff.
//! - [`VoxProcessor`] (audio thread) keeps the harmoniser and the per-channel
//!   formant paths **resident** (they have no structural parameters), polls the
//!   handoff for the current vocoder, runs whichever engine the [`Mode`]
//!   selects, and blends dry/wet with latency compensation — allocation-free.
//!
//! # Why the harmoniser is resident (not behind the handoff)
//!
//! MIDI note-on / note-off arrive on the audio thread and mutate the
//! harmoniser's held-note table.  Keeping the harmoniser resident means mode
//! switches and band changes never reset that table — a rebuild would drop held
//! notes mid-performance.
//!
//! # I/O and mode routing
//!
//! - **Vocoder** — main input is the modulator (voice); the carrier is the
//!   sidechain (or the input itself when no sidechain is connected).  The mono
//!   result is written to every output channel.
//! - **Harmoniser** — main channel 0 is the input voice; the wet chord is
//!   panned to stereo (summed to mono for a mono output).
//! - **Formant Shifter** — each channel is processed independently: an optional
//!   formant-preserving pitch shift, then the formant shift.

use rt_fft_convolver::{DenormalGuard, db_to_linear};
use rt_ir_plugin_core::{IrConsumer, IrPublisher, ir_handoff};
use rt_pitch_core::{
    FormantMethod, PhaseVocoder, Shifter, clamp_ratio, semitones_to_ratio, window::WindowKind,
};
use rt_vox_core::{FormantShifter, Harmoniser, MAX_VOICES, Scale, SpectralProbe, Vocoder};

use crate::params::{Mode, VoxParams};
use crate::ui::SPECTRUM_DISPLAY_BINS;

/// STFT frame size shared by the vocoder and the formant-mode paths.
const VOX_FRAME: usize = 1024;

/// Vocoder / formant STFT latency (`frame − hop`, hop = frame / 4).
const VOCODER_LATENCY: usize = VOX_FRAME - VOX_FRAME / rt_vox_core::vocoder::OVERLAP_FACTOR;

/// Dry-delay ring capacity in samples — a comfortable bound on any mode's
/// latency (harmoniser: phase vocoder + formant ≈ 1.8 k) plus block headroom.
const MAX_DELAY: usize = 8192;

/// Build a vocoder for the given band count and continuous settings.
/// **Allocates** — control thread only.
fn build_vocoder(bands: usize, sample_rate: f32, params: &VoxParams, max_block: usize) -> Vocoder {
    Vocoder::new(
        VOX_FRAME,
        bands,
        sample_rate,
        params.attack_ms,
        params.release_ms,
        max_block,
    )
}

/// Off-thread builder / publisher of the current [`Vocoder`].
pub struct VoxController {
    publisher: IrPublisher<Vocoder>,
    sample_rate: f32,
    max_block: usize,
    /// Band count of the currently published vocoder.
    bands: usize,
}

impl VoxController {
    /// Create a controller + processor pair, publishing the first vocoder so the
    /// processor can render on its first block.
    ///
    /// **Allocates** — call off the audio thread.
    ///
    /// # Panics
    /// Panics if `channels == 0`.
    pub fn new(
        max_block: usize,
        sample_rate: f32,
        channels: usize,
        params: &VoxParams,
    ) -> (Self, VoxProcessor) {
        assert!(channels > 0, "channels must be > 0");
        let max_block = max_block.max(1);

        let (mut publisher, consumer) = ir_handoff::<Vocoder>();
        let bands = params.bands.count();
        publisher.publish(build_vocoder(bands, sample_rate, params, max_block));

        let controller = Self {
            publisher,
            sample_rate,
            max_block,
            bands,
        };
        let processor = VoxProcessor::new(consumer, sample_rate, channels, max_block);
        (controller, processor)
    }

    /// Rebuild + republish the vocoder if the band count changed.  Returns
    /// `true` on a rebuild.  **Allocates on rebuild** — control thread only.
    pub fn apply_structural(&mut self, params: &VoxParams) -> bool {
        let bands = params.bands.count();
        if bands == self.bands {
            return false;
        }
        self.bands = bands;
        self.publisher.publish(build_vocoder(
            bands,
            self.sample_rate,
            params,
            self.max_block,
        ));
        true
    }
}

/// Fixed-capacity single-channel dry-signal delay for wet/dry alignment.
///
/// Reads `latency` samples behind the write head; a `latency` of `0` is a
/// transparent pass-through.  Sized once (off-RT); every operation is
/// allocation-free.
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

    #[inline]
    fn set_latency(&mut self, latency: usize) {
        self.latency = latency.min(self.buf.len() - 1);
    }

    #[inline]
    fn process(&mut self, sample: f32) -> f32 {
        if self.latency == 0 {
            return sample;
        }
        let cap = self.buf.len();
        let read = (self.write + cap - self.latency) % cap;
        let delayed = self.buf[read];
        self.buf[self.write] = sample;
        self.write = (self.write + 1) % cap;
        delayed
    }

    fn reset(&mut self) {
        self.buf.fill(0.0);
        self.write = 0;
    }
}

/// Audio-thread renderer: runs the active mode and blends dry/wet.
pub struct VoxProcessor {
    consumer: IrConsumer<Vocoder>,
    harmoniser: Harmoniser,
    /// One formant-preserving pitch shifter per channel (formant mode).
    formant_pitch: Vec<Box<dyn Shifter + Send>>,
    /// One formant shifter per channel (formant mode).
    formant_shifters: Vec<FormantShifter>,
    /// One dry-delay line per channel.
    dry_delays: Vec<DryDelay>,
    /// Per-channel input snapshot (so the dry path survives in-place writes).
    input_copies: Vec<Vec<f32>>,
    /// Wet scratch: mono / left.
    wet_a: Vec<f32>,
    /// Wet scratch: right (harmoniser).
    wet_b: Vec<f32>,
    /// Formant-mode intermediate (pitch-shift output).
    pitch_scratch: Vec<f32>,
    /// Editor-display spectral probe (driven only while the editor is open).
    probe: SpectralProbe,

    sample_rate: f32,
    harmoniser_latency: usize,
    formant_latency: usize,
    reported_latency: usize,
    last_process_samples: usize,
}

impl VoxProcessor {
    fn new(
        consumer: IrConsumer<Vocoder>,
        sample_rate: f32,
        channels: usize,
        max_block: usize,
    ) -> Self {
        let harmoniser = Harmoniser::new(sample_rate, max_block);
        let harmoniser_latency = harmoniser.latency_samples();

        let mut formant_pitch: Vec<Box<dyn Shifter + Send>> = Vec::with_capacity(channels);
        let mut formant_shifters = Vec::with_capacity(channels);
        for _ in 0..channels {
            let mut pv = PhaseVocoder::new(VOX_FRAME, max_block, WindowKind::Hann);
            pv.set_formant_preservation(Some(FormantMethod::Cepstral));
            formant_pitch.push(Box::new(pv));
            formant_shifters.push(FormantShifter::new(VOX_FRAME, sample_rate, max_block));
        }
        let formant_latency =
            formant_pitch[0].latency_samples() + formant_shifters[0].latency_samples();

        Self {
            consumer,
            harmoniser,
            formant_pitch,
            formant_shifters,
            dry_delays: (0..channels).map(|_| DryDelay::new()).collect(),
            input_copies: (0..channels).map(|_| vec![0.0; max_block]).collect(),
            wet_a: vec![0.0; max_block],
            wet_b: vec![0.0; max_block],
            pitch_scratch: vec![0.0; max_block],
            probe: SpectralProbe::new(VOX_FRAME, sample_rate, SPECTRUM_DISPLAY_BINS, max_block),
            sample_rate,
            harmoniser_latency,
            formant_latency,
            reported_latency: VOCODER_LATENCY,
            last_process_samples: 0,
        }
    }

    /// Latency (samples) the given mode reports to the host.
    #[inline]
    pub fn latency_samples(&self, mode: Mode) -> usize {
        match mode {
            Mode::Vocoder => VOCODER_LATENCY,
            Mode::Harmoniser => self.harmoniser_latency,
            Mode::FormantShifter => self.formant_latency,
        }
    }

    /// Latency reported by the last [`process`](Self::process) call.
    #[inline]
    pub fn reported_latency(&self) -> usize {
        self.reported_latency
    }

    /// Samples rendered on the last [`process`](Self::process) call.
    #[inline]
    pub fn last_process_samples(&self) -> usize {
        self.last_process_samples
    }

    /// Register a held MIDI note for the harmoniser.  RT-safe.
    #[inline]
    pub fn note_on(&mut self, note: u8) {
        self.harmoniser.note_on(note);
    }

    /// Release a held MIDI note.  RT-safe.
    #[inline]
    pub fn note_off(&mut self, note: u8) {
        self.harmoniser.note_off(note);
    }

    /// Release every held MIDI note.
    #[inline]
    pub fn all_notes_off(&mut self) {
        self.harmoniser.all_notes_off();
    }

    /// The harmoniser's most recently tracked input pitch (for the editor).
    #[inline]
    pub fn detected_hz(&self) -> Option<f32> {
        self.harmoniser.detected_hz()
    }

    /// Confidence `[0, 1]` of the most recent tracked pitch (editor readout).
    #[inline]
    pub fn detected_confidence(&self) -> f32 {
        self.harmoniser.detected_confidence()
    }

    /// Refresh the editor's formant display from the **last processed** input
    /// block, writing `SPECTRUM_DISPLAY_BINS` magnitude / envelope values.
    ///
    /// Intended to be called by the plugin only while the editor is open (so the
    /// probe's FFT + LPC cost is paid only when something is watching).  Returns
    /// `true` when the arrays were refreshed this call.  Allocation-free.
    pub fn probe_spectrum(&mut self, magnitude: &mut [f32], envelope: &mut [f32]) -> bool {
        let len = self.last_process_samples.min(self.input_copies[0].len());
        if len == 0 {
            return false;
        }
        // Split the borrow: the probe reads channel 0's snapshot.
        let (probe, input) = (&mut self.probe, &self.input_copies[0][..len]);
        probe.process_block(input, magnitude, envelope)
    }

    /// Render `main` in place using `params`; `sidechain` is the (mono) vocoder
    /// carrier (ignored by the other modes).
    ///
    /// **Real-time safe** — no allocation, no locking (provided every channel
    /// length ≤ the `max_block` the processor was built with).
    pub fn process(
        &mut self,
        main: &mut [&mut [f32]],
        sidechain: Option<&[f32]>,
        params: &VoxParams,
    ) {
        let block = main.first().map_or(0, |channel| channel.len());
        if block == 0 {
            self.last_process_samples = 0;
            return;
        }
        // Flush denormals for the whole callback.
        let _guard = DenormalGuard::new();

        // Snapshot the input so the dry path survives in-place writes.
        for (ch, channel) in main.iter().enumerate() {
            let len = channel.len().min(self.input_copies[ch].len());
            self.input_copies[ch][..len].copy_from_slice(&channel[..len]);
        }

        if params.bypass {
            // Output already equals the untouched input.
            self.last_process_samples = block;
            return;
        }

        let mix = params.mix.clamp(0.0, 1.0);
        let gain = db_to_linear(params.output_gain_db);
        let compensate = params.dry_delay_compensation;

        match params.mode {
            Mode::Vocoder => {
                self.process_vocoder(main, sidechain, params, mix, gain, compensate, block)
            }
            Mode::Harmoniser => self.process_harmoniser(main, params, mix, gain, compensate, block),
            Mode::FormantShifter => {
                self.process_formant(main, params, mix, gain, compensate, block)
            }
        }
        self.last_process_samples = block;
    }

    #[allow(clippy::too_many_arguments)]
    fn process_vocoder(
        &mut self,
        main: &mut [&mut [f32]],
        sidechain: Option<&[f32]>,
        params: &VoxParams,
        mix: f32,
        gain: f32,
        compensate: bool,
        block: usize,
    ) {
        let Some(vocoder) = self.consumer.poll() else {
            return;
        };
        vocoder.set_times(params.attack_ms, params.release_ms, self.sample_rate);
        vocoder.set_formant_shift_semitones(params.vocoder_formant_shift);
        vocoder.set_sibilance(params.sibilance, params.sibilance_amount);

        let modulator = &self.input_copies[0][..block];
        // Carrier: the connected sidechain, else the modulator itself (a
        // self-vocoding "robotise" fallback when no carrier is patched).
        let carrier: &[f32] = match sidechain {
            Some(sc) if sc.len() >= block => &sc[..block],
            _ => modulator,
        };
        vocoder.process_block(modulator, carrier, &mut self.wet_a[..block]);
        let latency = vocoder.latency_samples();
        self.reported_latency = latency;

        let comp = if compensate { latency } else { 0 };
        for (ch, channel) in main.iter_mut().enumerate() {
            self.dry_delays[ch].set_latency(comp);
            for sample in 0..block {
                let dry = self.dry_delays[ch].process(self.input_copies[ch][sample]);
                channel[sample] = (dry * (1.0 - mix) + self.wet_a[sample] * mix) * gain;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn process_harmoniser(
        &mut self,
        main: &mut [&mut [f32]],
        params: &VoxParams,
        mix: f32,
        gain: f32,
        compensate: bool,
        block: usize,
    ) {
        // Apply the harmoniser controls.
        self.harmoniser.set_snap_mode(params.snap_mode);
        self.harmoniser
            .set_scale(Scale::new(params.key, params.scale_kind));
        for index in 0..MAX_VOICES {
            let voice_params = params.voices[index];
            self.harmoniser
                .set_voice_free_interval(index, voice_params.interval);
            self.harmoniser
                .set_voice_enabled(index, voice_params.enabled);
            let voice = self.harmoniser.voice_mut(index);
            voice.set_level(voice_params.level);
            voice.set_pan(voice_params.pan);
            voice.set_detune_cents(voice_params.detune_cents);
            voice.set_formant_shift_semitones(voice_params.formant_shift);
        }

        let input = &self.input_copies[0][..block];
        self.harmoniser
            .process_block(input, &mut self.wet_a[..block], &mut self.wet_b[..block]);
        let latency = self.harmoniser.latency_samples();
        self.reported_latency = latency;

        let comp = if compensate { latency } else { 0 };
        let stereo = main.len() >= 2;
        for (ch, channel) in main.iter_mut().enumerate() {
            self.dry_delays[ch].set_latency(comp);
            for sample in 0..block {
                let wet = if !stereo {
                    // Mono output: sum the two panned halves.
                    self.wet_a[sample] + self.wet_b[sample]
                } else if ch == 0 {
                    self.wet_a[sample]
                } else {
                    self.wet_b[sample]
                };
                let dry = self.dry_delays[ch].process(self.input_copies[ch][sample]);
                channel[sample] = (dry * (1.0 - mix) + wet * mix) * gain;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn process_formant(
        &mut self,
        main: &mut [&mut [f32]],
        params: &VoxParams,
        mix: f32,
        gain: f32,
        compensate: bool,
        block: usize,
    ) {
        let ratio = clamp_ratio(semitones_to_ratio(params.formant_pitch_shift));
        let mut latency = self.formant_latency;
        for (ch, channel) in main.iter_mut().enumerate() {
            // Pitch shift (formant-preserving) → deliberate formant shift.
            self.formant_pitch[ch].set_ratio(ratio);
            self.formant_pitch[ch].process_block(
                &self.input_copies[ch][..block],
                &mut self.pitch_scratch[..block],
            );
            self.formant_shifters[ch].set_shift_semitones(params.formant_shift);
            self.formant_shifters[ch]
                .process_block(&self.pitch_scratch[..block], &mut self.wet_a[..block]);
            latency = self.formant_pitch[ch].latency_samples()
                + self.formant_shifters[ch].latency_samples();

            let comp = if compensate { latency } else { 0 };
            self.dry_delays[ch].set_latency(comp);
            for sample in 0..block {
                let dry = self.dry_delays[ch].process(self.input_copies[ch][sample]);
                channel[sample] = (dry * (1.0 - mix) + self.wet_a[sample] * mix) * gain;
            }
        }
        self.reported_latency = latency;
    }

    /// Flush every engine, delay, and scratch back to silence.
    pub fn reset(&mut self) {
        if let Some(vocoder) = self.consumer.current() {
            vocoder.reset();
        }
        self.harmoniser.reset();
        for pitch in &mut self.formant_pitch {
            pitch.reset();
        }
        for formant in &mut self.formant_shifters {
            formant.reset();
        }
        for delay in &mut self.dry_delays {
            delay.reset();
        }
        self.probe.reset();
        self.wet_a.fill(0.0);
        self.wet_b.fill(0.0);
        self.pitch_scratch.fill(0.0);
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{Bands, VoiceParams};

    const SAMPLE_RATE: f32 = 48_000.0;
    const BLOCK: usize = 256;

    fn engine(channels: usize, params: &VoxParams) -> (VoxController, VoxProcessor) {
        VoxController::new(BLOCK, SAMPLE_RATE, channels, params)
    }

    fn sawtooth(freq: f32, samples: usize) -> Vec<f32> {
        (0..samples)
            .map(|n| {
                let phase = freq * n as f32 / SAMPLE_RATE;
                2.0 * (phase - (phase + 0.5).floor())
            })
            .collect()
    }

    fn sine(freq: f32, samples: usize) -> Vec<f32> {
        use std::f32::consts::TAU;
        (0..samples)
            .map(|n| (TAU * freq * n as f32 / SAMPLE_RATE).sin())
            .collect()
    }

    fn rms(signal: &[f32]) -> f32 {
        if signal.is_empty() {
            return 0.0;
        }
        (signal.iter().map(|&s| s * s).sum::<f32>() / signal.len() as f32).sqrt()
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

    /// Run a mono signal through the processor block by block.
    fn run_mono(processor: &mut VoxProcessor, params: &VoxParams, input: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(input.len());
        for chunk in input.chunks(BLOCK) {
            let mut buffer = chunk.to_vec();
            {
                let mut main: [&mut [f32]; 1] = [&mut buffer];
                processor.process(&mut main, None, params);
            }
            out.extend_from_slice(&buffer);
        }
        out
    }

    #[test]
    fn builds_and_renders_finite() {
        let params = VoxParams::default();
        let (_controller, mut processor) = engine(2, &params);
        let mut left = vec![0.1f32; BLOCK];
        let mut right = vec![0.1f32; BLOCK];
        {
            let mut main: [&mut [f32]; 2] = [&mut left, &mut right];
            processor.process(&mut main, None, &params);
        }
        assert!(left.iter().all(|s| s.is_finite()));
        assert!(right.iter().all(|s| s.is_finite()));
        assert_eq!(processor.last_process_samples(), BLOCK);
    }

    #[test]
    fn bypass_is_transparent() {
        let params = VoxParams {
            bypass: true,
            ..Default::default()
        };
        let (_controller, mut processor) = engine(1, &params);
        let original: Vec<f32> = (0..BLOCK).map(|n| (n as f32 * 0.1).sin()).collect();
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
    fn vocoder_silent_input_is_silent() {
        // No sidechain → self-vocoding; silent input must yield silence.
        let params = VoxParams {
            mode: Mode::Vocoder,
            ..Default::default()
        };
        let (_controller, mut processor) = engine(1, &params);
        let out = run_mono(&mut processor, &params, &vec![0.0f32; 8192]);
        assert!(rms(&out) < 1e-6, "vocoder leaked signal: {}", rms(&out));
    }

    #[test]
    fn vocoder_with_sidechain_produces_output() {
        let params = VoxParams {
            mode: Mode::Vocoder,
            dry_delay_compensation: false,
            ..Default::default()
        };
        let (_controller, mut processor) = engine(1, &params);
        let modulator = sawtooth(220.0, 12_000);
        let carrier = sawtooth(110.0, 12_000);
        let mut out = Vec::new();
        for (m, c) in modulator.chunks(BLOCK).zip(carrier.chunks(BLOCK)) {
            let mut buffer = m.to_vec();
            let carrier_chunk = c.to_vec();
            {
                let mut main: [&mut [f32]; 1] = [&mut buffer];
                processor.process(&mut main, Some(&carrier_chunk), &params);
            }
            out.extend_from_slice(&buffer);
        }
        assert!(rms(&out[VOX_FRAME * 3..]) > 1e-4, "no vocoder output");
    }

    #[test]
    fn harmoniser_octave_voice_shifts_up() {
        let mut params = VoxParams {
            mode: Mode::Harmoniser,
            snap_mode: rt_vox_core::SnapMode::Free,
            mix: 1.0,
            dry_delay_compensation: false,
            ..Default::default()
        };
        // Only one unison-plus-octave voice, hard enabled.
        params.voices = [VoiceParams::default(); crate::params::NUM_VOICES];
        params.voices[0] = VoiceParams {
            interval: 12.0,
            level: 1.0,
            enabled: true,
            ..VoiceParams::default()
        };
        let (_controller, mut processor) = engine(1, &params);
        let out = run_mono(&mut processor, &params, &sine(220.0, 24_000));
        let hz = dominant_hz(&out[out.len() / 2..]);
        assert!((hz - 440.0).abs() < 15.0, "octave-up landed at {hz}");
    }

    #[test]
    fn harmoniser_midi_note_drives_voice() {
        let params = VoxParams {
            mode: Mode::Harmoniser,
            snap_mode: rt_vox_core::SnapMode::Midi,
            mix: 1.0,
            dry_delay_compensation: false,
            ..Default::default()
        };
        let (_controller, mut processor) = engine(1, &params);
        processor.note_on(69); // A4 over an A3 (220 Hz) input → +12 → 440 Hz
        let out = run_mono(&mut processor, &params, &sine(220.0, 24_000));
        let hz = dominant_hz(&out[out.len() / 2..]);
        assert!((hz - 440.0).abs() < 16.0, "midi voice landed at {hz}");
    }

    #[test]
    fn formant_mode_preserves_pitch() {
        let params = VoxParams {
            mode: Mode::FormantShifter,
            formant_shift: 7.0,
            formant_pitch_shift: 0.0,
            mix: 1.0,
            dry_delay_compensation: false,
            ..Default::default()
        };
        let (_controller, mut processor) = engine(1, &params);
        let out = run_mono(&mut processor, &params, &sine(220.0, 24_000));
        let hz = dominant_hz(&out[out.len() / 2..]);
        assert!((hz - 220.0).abs() < 14.0, "pitch drifted to {hz}");
    }

    #[test]
    fn mix_zero_is_dry() {
        // Harmoniser with an octave voice, mix 0 → output equals the input.
        let params = VoxParams {
            mode: Mode::Harmoniser,
            snap_mode: rt_vox_core::SnapMode::Free,
            mix: 0.0,
            dry_delay_compensation: false,
            ..Default::default()
        };
        let (_controller, mut processor) = engine(1, &params);
        let original: Vec<f32> = (0..BLOCK).map(|n| 0.5 * (n as f32 * 0.05).sin()).collect();
        let mut samples = original.clone();
        {
            let mut main: [&mut [f32]; 1] = [&mut samples];
            processor.process(&mut main, None, &params);
        }
        for (out, orig) in samples.iter().zip(&original) {
            assert!((out - orig).abs() < 1e-6, "mix=0 should be dry");
        }
    }

    #[test]
    fn output_gain_scales_dry() {
        let params = VoxParams {
            mode: Mode::Harmoniser,
            mix: 0.0, // pure dry so the relationship is exact
            output_gain_db: 6.0,
            dry_delay_compensation: false,
            ..Default::default()
        };
        let (_controller, mut processor) = engine(1, &params);
        let original: Vec<f32> = (0..BLOCK).map(|n| 0.25 * (n as f32 * 0.03).sin()).collect();
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
    fn latency_differs_per_mode() {
        let params = VoxParams::default();
        let (_controller, processor) = engine(1, &params);
        assert_eq!(processor.latency_samples(Mode::Vocoder), VOCODER_LATENCY);
        assert!(processor.latency_samples(Mode::Harmoniser) > 0);
        assert!(processor.latency_samples(Mode::FormantShifter) > 0);
    }

    #[test]
    fn structural_change_rebuilds_vocoder() {
        let params = VoxParams::default();
        let (mut controller, mut processor) = engine(1, &params);
        // Same band count → no rebuild.
        assert!(!controller.apply_structural(&params));
        // Change bands → rebuild, and the processor still renders.
        let more = VoxParams {
            bands: Bands::B32,
            ..Default::default()
        };
        assert!(controller.apply_structural(&more));
        let out = run_mono(&mut processor, &more, &sawtooth(220.0, 4096));
        assert!(out.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn probe_and_confidence_feed_the_editor() {
        let params = VoxParams {
            mode: Mode::Harmoniser,
            snap_mode: rt_vox_core::SnapMode::Free,
            ..Default::default()
        };
        let (_controller, mut processor) = engine(1, &params);
        let mut mag = vec![0.0f32; crate::ui::SPECTRUM_DISPLAY_BINS];
        let mut env = vec![0.0f32; crate::ui::SPECTRUM_DISPLAY_BINS];

        let input = sine(220.0, 24_000);
        let mut probed = false;
        for chunk in input.chunks(BLOCK) {
            let mut buffer = chunk.to_vec();
            {
                let mut main: [&mut [f32]; 1] = [&mut buffer];
                processor.process(&mut main, None, &params);
            }
            probed |= processor.probe_spectrum(&mut mag, &mut env);
        }
        assert!(probed, "probe never produced a frame");
        assert!(mag.iter().all(|v| v.is_finite()) && mag.iter().any(|&v| v > 0.0));
        assert!(processor.detected_confidence() > 0.5, "pitch not tracked");
    }

    #[test]
    fn reset_clears_tail() {
        let params = VoxParams {
            mode: Mode::Harmoniser,
            dry_delay_compensation: false,
            ..Default::default()
        };
        let (_controller, mut processor) = engine(1, &params);
        let _ = run_mono(&mut processor, &params, &sawtooth(220.0, 8192));
        processor.reset();
        let out = run_mono(&mut processor, &params, &vec![0.0f32; BLOCK * 2]);
        // Immediately after reset over silence the harmoniser has nothing to
        // shift → the wet output is silent (mix leaves only silent dry).
        assert!(
            rms(&out[..BLOCK]) < 1e-3,
            "reset left a tail: {}",
            rms(&out[..BLOCK])
        );
    }
}
