//! FFT-based vocoder: a modulator (voice) shapes a carrier's spectrum.
//!
//! # Topology (Roadmap-08 Phase 0 decision)
//!
//! Instead of an analog bank of state-variable band-pass filters, this engine
//! works in the STFT domain and groups FFT bins into log-spaced bands
//! ([`LogBands`]).  Per synthesis frame it:
//!
//! 1. windows and FFTs both the **modulator** and the **carrier**,
//! 2. measures the modulator's mean magnitude in each band,
//! 3. smooths that level with a per-band attack/release [`EnvelopeFollower`],
//! 4. multiplies every **carrier** bin by its band's smoothed level, and
//! 5. IFFTs and overlap-adds the reshaped carrier.
//!
//! A silent modulator drives every band gain to zero → silent output; a loud
//! modulator band boosts the same band of the carrier, imprinting the voice's
//! moving formants onto the (typically harmonically rich) carrier.
//!
//! All buffers are allocated in [`Vocoder::new`]; [`process_block`] is
//! allocation-free and safe on the audio thread.
//!
//! [`process_block`]: Vocoder::process_block

use realfft::num_complex::Complex;
use rt_fft_convolver::{DenormalGuard, FftHandler};
use rt_pitch_core::window::{Window, WindowKind};

use crate::bands::LogBands;
use crate::envelope::EnvelopeFollower;

/// Analysis/synthesis overlap factor: hop = `frame / OVERLAP_FACTOR`.  4×
/// overlap (75 %) is the usual sweet spot for a Hann-windowed STFT vocoder.
pub const OVERLAP_FACTOR: usize = 4;

/// FFT magnitude-envelope-transfer vocoder.
pub struct Vocoder {
    frame: usize,
    hop: usize,
    bins: usize,

    fft: FftHandler,
    window: Vec<f32>,

    // Two independent STFT input framers (modulator, carrier) kept in lockstep,
    // plus one output overlap-add framer.
    mod_analyzer: rt_pitch_core::stft::Analyzer,
    car_analyzer: rt_pitch_core::stft::Analyzer,
    synth: rt_pitch_core::stft::Synthesizer,

    bands: LogBands,
    followers: Vec<EnvelopeFollower>,

    // Preallocated per-frame scratch.
    mod_frame: Vec<f32>,
    car_frame: Vec<f32>,
    mod_spectrum: Vec<Complex<f32>>,
    car_spectrum: Vec<Complex<f32>>,
    out_frame: Vec<f32>,
    /// Smoothed per-band gain applied to the carrier this frame.
    band_gain: Vec<f32>,

    /// Fractional band offset applied when the carrier reads the modulator's
    /// band gains — the vocoder `Formant Shift` control.  Positive shifts the
    /// spectral character upward.
    formant_shift_bands: f32,
    /// Pass the modulator's high bins straight through (sibilance handling):
    /// consonants (`s`, `t`, `f`) live above the vocoder's band resolution and
    /// sound clearer routed direct instead of through the carrier.
    sibilance_enabled: bool,
    /// Blend `[0, 1]` of direct modulator into the sibilance region.
    sibilance_amount: f32,
    /// First FFT bin treated as sibilance (from a fixed ~5 kHz cutoff).
    sibilance_start_bin: usize,
}

impl Vocoder {
    /// Build a vocoder for `frame`-point FFTs (must divide by [`OVERLAP_FACTOR`]
    /// and be even) with `num_bands` log-spaced bands at `sample_rate`,
    /// absorbing host blocks up to `max_block`.
    ///
    /// **Allocates** — construct off the audio thread.
    ///
    /// # Panics
    /// Panics if `frame` is not a multiple of [`OVERLAP_FACTOR`], or
    /// `num_bands == 0`.
    pub fn new(
        frame: usize,
        num_bands: usize,
        sample_rate: f32,
        attack_ms: f32,
        release_ms: f32,
        max_block: usize,
    ) -> Self {
        assert!(
            frame.is_multiple_of(OVERLAP_FACTOR),
            "frame must be divisible by the overlap factor"
        );
        assert!(num_bands > 0, "need at least one band");
        let hop = frame / OVERLAP_FACTOR;
        let bins = frame / 2 + 1;
        let window = Window::new(WindowKind::Hann, frame);
        let norm = window.ola_normalisation(hop);

        let frame_rate = sample_rate / hop as f32;
        let bands = LogBands::new(num_bands, bins, frame, sample_rate, 80.0);
        let followers =
            vec![EnvelopeFollower::new(attack_ms, release_ms, frame_rate); bands.count()];

        Self {
            frame,
            hop,
            bins,
            fft: FftHandler::new(frame),
            window: window.as_slice().to_vec(),
            mod_analyzer: rt_pitch_core::stft::Analyzer::new(frame, hop, max_block),
            car_analyzer: rt_pitch_core::stft::Analyzer::new(frame, hop, max_block),
            synth: rt_pitch_core::stft::Synthesizer::new(frame, hop, max_block, norm),
            band_gain: vec![0.0; bands.count()],
            followers,
            bands,
            mod_frame: vec![0.0; frame],
            car_frame: vec![0.0; frame],
            mod_spectrum: vec![Complex::default(); bins],
            car_spectrum: vec![Complex::default(); bins],
            out_frame: vec![0.0; frame],
            formant_shift_bands: 0.0,
            sibilance_enabled: false,
            sibilance_amount: 0.0,
            // ~5 kHz: consonant energy sits above this.
            sibilance_start_bin: ((5_000.0 / sample_rate) * frame as f32).min(bins as f32 - 1.0)
                as usize,
        }
    }

    /// Algorithmic latency in samples (the STFT framing delay).
    #[inline]
    pub fn latency_samples(&self) -> usize {
        self.frame - self.hop
    }

    /// Update the envelope-follower attack/release times (off-RT).
    pub fn set_times(&mut self, attack_ms: f32, release_ms: f32, sample_rate: f32) {
        let frame_rate = sample_rate / self.hop as f32;
        for follower in &mut self.followers {
            follower.set_times(attack_ms, release_ms, frame_rate);
        }
    }

    /// Set the vocoder `Formant Shift` in semitones (± an octave is typical).
    ///
    /// Converted to a fractional band offset using the average band width in
    /// semitones, so the shift reads musically regardless of the band count.
    /// Cheap; safe per block on the audio thread.
    pub fn set_formant_shift_semitones(&mut self, semitones: f32) {
        // Bands span ~`low_hz`..Nyquist ≈ 80 Hz..24 kHz ≈ 8.2 octaves; a band is
        // therefore ~`8.2 * 12 / count` semitones wide.
        let semitones_per_band = 8.2 * 12.0 / self.bands.count() as f32;
        self.formant_shift_bands = semitones / semitones_per_band;
    }

    /// Configure sibilance handling: when `enabled`, blend `amount` of the
    /// modulator's high band directly into the output.
    pub fn set_sibilance(&mut self, enabled: bool, amount: f32) {
        self.sibilance_enabled = enabled;
        self.sibilance_amount = amount.clamp(0.0, 1.0);
    }

    /// Linear interpolation of the per-band gain array at fractional band
    /// `position`, edge-clamped.
    #[inline]
    fn band_gain_at(&self, position: f32) -> f32 {
        let last = self.band_gain.len() - 1;
        if position <= 0.0 {
            return self.band_gain[0];
        }
        if position >= last as f32 {
            return self.band_gain[last];
        }
        let base = position.floor() as usize;
        let frac = position - base as f32;
        self.band_gain[base] * (1.0 - frac) + self.band_gain[base + 1] * frac
    }

    /// Process one block: `modulator` and `carrier` shape into `output`.
    ///
    /// All three slices must be the same length (`<= max_block`).  Output is
    /// silent during the initial `latency_samples()` warm-up.
    pub fn process_block(&mut self, modulator: &[f32], carrier: &[f32], output: &mut [f32]) {
        debug_assert_eq!(modulator.len(), carrier.len());
        debug_assert_eq!(modulator.len(), output.len());
        // Denormal bins near silence would stall the FFT/IFFT arithmetic.
        let _guard = DenormalGuard::new();

        self.mod_analyzer.push_block(modulator);
        self.car_analyzer.push_block(carrier);

        // Both analyzers received identical block lengths, so they emit frames
        // in lockstep; pull them together.
        while self.mod_analyzer.next_frame(&mut self.mod_frame) {
            let had_carrier = self.car_analyzer.next_frame(&mut self.car_frame);
            debug_assert!(had_carrier, "carrier analyzer fell out of lockstep");

            self.process_frame();
            self.synth.add_frame(&self.out_frame);
        }

        self.synth.pull_block(output);
    }

    /// One STFT frame of the modulator→carrier transfer.
    fn process_frame(&mut self) {
        // Analysis windows.
        for (sample, &win) in self.mod_frame.iter_mut().zip(&self.window) {
            *sample *= win;
        }
        for (sample, &win) in self.car_frame.iter_mut().zip(&self.window) {
            *sample *= win;
        }
        self.fft.forward(&self.mod_frame, &mut self.mod_spectrum);
        self.fft.forward(&self.car_frame, &mut self.car_spectrum);

        // Per-band modulator level → smoothed gain.
        for band in 0..self.bands.count() {
            let (start, end) = self.bands.bin_range(band);
            let mut sum = 0.0f32;
            for bin in start..end {
                sum += self.mod_spectrum[bin].norm();
            }
            // Mean magnitude, normalised by frame size so the gain is ~O(1) for
            // an O(1)-amplitude modulator (absolute level is trimmed by the
            // Phase-2 output gain).
            let count = (end - start).max(1) as f32;
            let level = sum / count / self.frame as f32;
            self.band_gain[band] = self.followers[band].process(level);
        }

        // Apply each band's (formant-shifted) gain to the carrier bins, then
        // pass the modulator's highs through in the sibilance region.
        for bin in 0..self.bins {
            let band = self.bands.band_of_bin(bin) as f32;
            let gain = self.band_gain_at(band - self.formant_shift_bands);
            let mut value = self.car_spectrum[bin] * gain;
            if self.sibilance_enabled && bin >= self.sibilance_start_bin {
                // Blend toward the raw modulator bin for crisp consonants.
                value = value * (1.0 - self.sibilance_amount)
                    + self.mod_spectrum[bin] * self.sibilance_amount;
            }
            self.car_spectrum[bin] = value;
        }
        self.fft
            .inverse(&mut self.car_spectrum, &mut self.out_frame);

        // Synthesis window (w² overlap-add with the analysis window).
        for (sample, &win) in self.out_frame.iter_mut().zip(&self.window) {
            *sample *= win;
        }
    }

    /// Flush all framing / envelope state back to silence.
    pub fn reset(&mut self) {
        self.mod_analyzer.reset();
        self.car_analyzer.reset();
        self.synth.reset();
        for follower in &mut self.followers {
            follower.reset();
        }
        self.mod_frame.fill(0.0);
        self.car_frame.fill(0.0);
        self.out_frame.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{make_sawtooth, make_sine, rms};

    const SAMPLE_RATE: f32 = 48_000.0;
    const FRAME: usize = 1024;
    const MAX_BLOCK: usize = 512;

    /// Run a full signal through the vocoder block by block, returning the
    /// concatenated output.
    fn run(voc: &mut Vocoder, modulator: &[f32], carrier: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(modulator.len());
        let mut block = vec![0.0f32; MAX_BLOCK];
        for (m, c) in modulator.chunks(MAX_BLOCK).zip(carrier.chunks(MAX_BLOCK)) {
            let len = m.len();
            voc.process_block(m, c, &mut block[..len]);
            out.extend_from_slice(&block[..len]);
        }
        out
    }

    #[test]
    fn silent_modulator_gives_silence() {
        let mut voc = Vocoder::new(FRAME, 16, SAMPLE_RATE, 5.0, 50.0, MAX_BLOCK);
        let carrier = make_sawtooth(110.0, SAMPLE_RATE, 12_000);
        let modulator = vec![0.0f32; carrier.len()];
        let out = run(&mut voc, &modulator, &carrier);
        assert!(rms(&out) < 1e-6, "expected silence, rms = {}", rms(&out));
    }

    #[test]
    fn present_modulator_passes_energy() {
        let mut voc = Vocoder::new(FRAME, 16, SAMPLE_RATE, 5.0, 50.0, MAX_BLOCK);
        let carrier = make_sawtooth(110.0, SAMPLE_RATE, 12_000);
        let modulator = make_sawtooth(220.0, SAMPLE_RATE, 12_000);
        let out = run(&mut voc, &modulator, &carrier);
        // Ignore the warm-up region.
        assert!(
            rms(&out[FRAME * 3..]) > 1e-4,
            "expected audible output, rms = {}",
            rms(&out[FRAME * 3..])
        );
    }

    #[test]
    fn output_follows_the_modulator_band() {
        // Carrier: broadband noise-like sawtooth.  Modulator: a low tone then a
        // high tone.  The band carrying the modulator's energy should dominate
        // the output — verify by comparing output energy for a low- vs
        // high-frequency modulator.
        let mut voc_low = Vocoder::new(FRAME, 24, SAMPLE_RATE, 2.0, 30.0, MAX_BLOCK);
        let mut voc_high = Vocoder::new(FRAME, 24, SAMPLE_RATE, 2.0, 30.0, MAX_BLOCK);
        let carrier = make_sawtooth(110.0, SAMPLE_RATE, 12_000);
        let low_mod = make_sine(200.0, SAMPLE_RATE, 12_000);
        let high_mod = make_sine(4000.0, SAMPLE_RATE, 12_000);

        let out_low = run(&mut voc_low, &low_mod, &carrier);
        let out_high = run(&mut voc_high, &high_mod, &carrier);
        // A sawtooth carrier has far more energy in low harmonics, so a
        // low-band modulator lets more through than a high-band one.
        assert!(
            rms(&out_low[FRAME * 3..]) > rms(&out_high[FRAME * 3..]),
            "low {} should exceed high {}",
            rms(&out_low[FRAME * 3..]),
            rms(&out_high[FRAME * 3..])
        );
    }

    #[test]
    fn silent_modulator_stays_silent_with_all_controls() {
        // Formant shift + sibilance must not manufacture signal from silence.
        let mut voc = Vocoder::new(FRAME, 16, SAMPLE_RATE, 5.0, 50.0, MAX_BLOCK);
        voc.set_formant_shift_semitones(7.0);
        voc.set_sibilance(true, 1.0);
        let carrier = make_sawtooth(110.0, SAMPLE_RATE, 12_000);
        let modulator = vec![0.0f32; carrier.len()];
        let out = run(&mut voc, &modulator, &carrier);
        assert!(rms(&out) < 1e-6, "controls leaked signal: {}", rms(&out));
    }

    #[test]
    fn sibilance_injects_modulator_highs() {
        // A modulator with strong high-frequency content and a low carrier:
        // with sibilance on, high-frequency energy should reach the output.
        let carrier = make_sawtooth(110.0, SAMPLE_RATE, 16_000);
        let modulator = make_sine(9_000.0, SAMPLE_RATE, 16_000);

        let mut dry = Vocoder::new(FRAME, 16, SAMPLE_RATE, 2.0, 30.0, MAX_BLOCK);
        let plain = run(&mut dry, &modulator, &carrier);

        let mut sib = Vocoder::new(FRAME, 16, SAMPLE_RATE, 2.0, 30.0, MAX_BLOCK);
        sib.set_sibilance(true, 1.0);
        let sibilant = run(&mut sib, &modulator, &carrier);

        // The direct high-frequency modulator path adds energy the carrier
        // (which is weak up there) cannot supply on its own.
        assert!(
            rms(&sibilant[FRAME * 3..]) > rms(&plain[FRAME * 3..]),
            "sibilance {} should exceed plain {}",
            rms(&sibilant[FRAME * 3..]),
            rms(&plain[FRAME * 3..])
        );
    }

    #[test]
    fn formant_shift_changes_output() {
        let carrier = make_sawtooth(110.0, SAMPLE_RATE, 16_000);
        let modulator = make_sawtooth(220.0, SAMPLE_RATE, 16_000);

        let mut flat = Vocoder::new(FRAME, 24, SAMPLE_RATE, 2.0, 30.0, MAX_BLOCK);
        let base = run(&mut flat, &modulator, &carrier);

        let mut up = Vocoder::new(FRAME, 24, SAMPLE_RATE, 2.0, 30.0, MAX_BLOCK);
        up.set_formant_shift_semitones(12.0);
        let shifted = run(&mut up, &modulator, &carrier);

        // The two outputs must differ once the formant character is shifted.
        let diff: f32 = base[FRAME * 3..]
            .iter()
            .zip(&shifted[FRAME * 3..])
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-3, "formant shift had no audible effect: {diff}");
    }

    #[test]
    fn reset_returns_to_silence() {
        let mut voc = Vocoder::new(FRAME, 16, SAMPLE_RATE, 5.0, 50.0, MAX_BLOCK);
        let carrier = make_sawtooth(110.0, SAMPLE_RATE, 4096);
        let modulator = make_sawtooth(220.0, SAMPLE_RATE, 4096);
        run(&mut voc, &modulator, &carrier);
        voc.reset();
        // After reset, a silent modulator immediately yields silence again.
        let carrier2 = make_sawtooth(110.0, SAMPLE_RATE, 4096);
        let silence = vec![0.0f32; carrier2.len()];
        let out = run(&mut voc, &silence, &carrier2);
        assert!(rms(&out) < 1e-6, "rms after reset = {}", rms(&out));
    }
}
