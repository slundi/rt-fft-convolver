//! Independent formant shifting: move the spectral envelope, keep the pitch.
//!
//! # Source–filter separation
//!
//! A voiced sound is a harmonic **source** (the vocal-fold buzz, whose spacing
//! sets the *pitch*) shaped by a smooth **filter** — the vocal-tract resonances,
//! or *formants*, that set the perceived vowel / timbre.  Shifting formants
//! without shifting pitch means reshaping the envelope while leaving the
//! harmonic peaks exactly where they are.
//!
//! Per STFT frame this engine:
//!
//! 1. FFTs the windowed frame,
//! 2. estimates the spectral envelope with [`FormantEstimator`] (LPC by
//!    default — the voice model),
//! 3. builds a **warped** envelope by resampling the estimate along frequency by
//!    the shift factor (`env_warped[f] = env[f / factor]`),
//! 4. multiplies each bin by `env_warped / env` — flattening the original
//!    envelope and imposing the warped one, which keeps every harmonic's
//!    frequency (and phase) untouched, and
//! 5. IFFTs and overlap-adds.
//!
//! Because only smooth magnitude weights change, the fundamental is preserved
//! (Roadmap-08 Phase 4 target: ≤ 5 cents drift).  Pitch shifting *without*
//! moving formants is the complementary operation — handled by
//! `rt-pitch-core`'s formant-preserving phase vocoder — so between the two the
//! user can move either dimension independently.
//!
//! All buffers are allocated in [`FormantShifter::new`]; [`process_block`] is
//! allocation-free.
//!
//! [`process_block`]: FormantShifter::process_block

use realfft::num_complex::Complex;
use rt_fft_convolver::{DenormalGuard, FftHandler};
use rt_pitch_core::{
    FormantEstimator, FormantMethod, semitones_to_ratio,
    window::{Window, WindowKind},
};

/// Overlap factor: hop = `frame / OVERLAP_FACTOR` (75 % overlap).
pub const OVERLAP_FACTOR: usize = 4;

/// Independent (pitch-preserving) formant shifter.
pub struct FormantShifter {
    frame: usize,
    hop: usize,
    bins: usize,

    fft: FftHandler,
    window: Vec<f32>,

    analyzer: rt_pitch_core::stft::Analyzer,
    synth: rt_pitch_core::stft::Synthesizer,
    formant: FormantEstimator,

    /// Frequency-warp factor: `2^(semitones / 12)`; `1.0` = no shift.
    shift_factor: f32,

    // Preallocated per-frame scratch.
    time_frame: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    magnitude: Vec<f32>,
    envelope: Vec<f32>,
    warped: Vec<f32>,
    out_frame: Vec<f32>,
}

impl FormantShifter {
    /// Build a formant shifter for `frame`-point FFTs at `sample_rate`,
    /// absorbing host blocks up to `max_block`.
    ///
    /// The LPC order is voice-tuned (`sample_rate / 1000 + 2`).
    ///
    /// **Allocates** — construct off the audio thread.
    ///
    /// # Panics
    /// Panics if `frame` is not a multiple of [`OVERLAP_FACTOR`].
    pub fn new(frame: usize, sample_rate: f32, max_block: usize) -> Self {
        assert!(
            frame.is_multiple_of(OVERLAP_FACTOR),
            "frame must be divisible by the overlap factor"
        );
        let hop = frame / OVERLAP_FACTOR;
        let bins = frame / 2 + 1;
        let window = Window::new(WindowKind::Hann, frame);
        let norm = window.ola_normalisation(hop);
        let lpc_order = (sample_rate / 1000.0) as usize + 2;

        Self {
            frame,
            hop,
            bins,
            fft: FftHandler::new(frame),
            window: window.as_slice().to_vec(),
            analyzer: rt_pitch_core::stft::Analyzer::new(frame, hop, max_block),
            synth: rt_pitch_core::stft::Synthesizer::new(frame, hop, max_block, norm),
            formant: FormantEstimator::new(FormantMethod::Lpc, frame, lpc_order),
            shift_factor: 1.0,
            time_frame: vec![0.0; frame],
            spectrum: vec![Complex::default(); bins],
            magnitude: vec![0.0; bins],
            envelope: vec![0.0; bins],
            warped: vec![0.0; bins],
            out_frame: vec![0.0; frame],
        }
    }

    /// Algorithmic latency in samples (STFT framing delay).
    #[inline]
    pub fn latency_samples(&self) -> usize {
        self.frame - self.hop
    }

    /// Set the formant shift in semitones (positive = up, "brighter" / smaller
    /// resonator).  Cheap; safe per block on the audio thread.
    #[inline]
    pub fn set_shift_semitones(&mut self, semitones: f32) {
        self.shift_factor = semitones_to_ratio(semitones);
    }

    /// Process one block, shifting formants in place from `input` to `output`
    /// (same length, `<= max_block`).  Silent during the framing warm-up.
    pub fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), output.len());
        let _guard = DenormalGuard::new();

        self.analyzer.push_block(input);
        while self.analyzer.next_frame(&mut self.time_frame) {
            self.process_frame();
            self.synth.add_frame(&self.out_frame);
        }
        self.synth.pull_block(output);
    }

    /// One STFT frame of formant reshaping.
    fn process_frame(&mut self) {
        // Analysis window → FFT.
        for (sample, &win) in self.time_frame.iter_mut().zip(&self.window) {
            *sample *= win;
        }
        self.fft.forward(&self.time_frame, &mut self.spectrum);

        // Magnitude spectrum and its smooth envelope.
        for (mag, bin) in self.magnitude.iter_mut().zip(&self.spectrum) {
            *mag = bin.norm();
        }
        self.formant.estimate(&self.magnitude, &mut self.envelope);

        // Warp the envelope along frequency: warped[bin] = env[bin / factor],
        // linearly interpolated and edge-clamped.
        let last = self.bins - 1;
        for (bin, dst) in self.warped.iter_mut().enumerate() {
            let src = bin as f32 / self.shift_factor;
            *dst = interpolate(&self.envelope, src, last);
        }

        // Re-weight each bin by warped/original envelope (guarding silent bins),
        // keeping phase — and therefore pitch — untouched.
        for bin in 0..self.bins {
            let original = self.envelope[bin];
            let gain = if original > 1e-9 {
                self.warped[bin] / original
            } else {
                0.0
            };
            self.spectrum[bin] *= gain;
        }

        // Back to time, synthesis window (w² overlap-add).
        self.fft.inverse(&mut self.spectrum, &mut self.out_frame);
        for (sample, &win) in self.out_frame.iter_mut().zip(&self.window) {
            *sample *= win;
        }
    }

    /// Flush all framing state.
    pub fn reset(&mut self) {
        self.analyzer.reset();
        self.synth.reset();
        self.time_frame.fill(0.0);
        self.out_frame.fill(0.0);
    }
}

/// Linear interpolation of `data` at fractional index `pos`, clamped to
/// `[0, last]`.
#[inline]
fn interpolate(data: &[f32], pos: f32, last: usize) -> f32 {
    if pos <= 0.0 {
        return data[0];
    }
    if pos >= last as f32 {
        return data[last];
    }
    let base = pos.floor() as usize;
    let frac = pos - base as f32;
    data[base] * (1.0 - frac) + data[base + 1] * frac
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{dominant_frequency_hz, make_sawtooth, rms};

    const SAMPLE_RATE: f32 = 48_000.0;
    const FRAME: usize = 1024;
    const MAX_BLOCK: usize = 512;

    fn run(shifter: &mut FormantShifter, input: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(input.len());
        let mut block = vec![0.0f32; MAX_BLOCK];
        for chunk in input.chunks(MAX_BLOCK) {
            let len = chunk.len();
            shifter.process_block(chunk, &mut block[..len]);
            out.extend_from_slice(&block[..len]);
        }
        out
    }

    /// Spectral centroid (Hz) of the steady-state tail — a proxy for "how high"
    /// the formant energy sits.
    fn spectral_centroid(signal: &[f32], sample_rate: f32) -> f32 {
        let len = 4096;
        let start = signal.len() - len;
        let mut fft = FftHandler::new(len);
        let mut spectrum = vec![Complex::default(); len / 2 + 1];
        let windowed: Vec<f32> = signal[start..]
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let w = 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / len as f32).cos();
                s * w
            })
            .collect();
        fft.forward(&windowed, &mut spectrum);
        let mut weighted = 0.0f32;
        let mut total = 0.0f32;
        for (bin, value) in spectrum.iter().enumerate() {
            let mag = value.norm();
            weighted += mag * bin as f32 * sample_rate / len as f32;
            total += mag;
        }
        if total > 0.0 { weighted / total } else { 0.0 }
    }

    #[test]
    fn identity_shift_reconstructs_input() {
        let mut shifter = FormantShifter::new(FRAME, SAMPLE_RATE, MAX_BLOCK);
        shifter.set_shift_semitones(0.0);
        let input = make_sawtooth(220.0, SAMPLE_RATE, 16_000);
        let out = run(&mut shifter, &input);
        // Compare an interior region past the warm-up, allowing the framing
        // latency and windowing taper.
        let latency = shifter.latency_samples();
        let mut max_err = 0.0f32;
        for n in (FRAME * 3)..(input.len() - FRAME) {
            max_err = max_err.max((out[n] - input[n - latency]).abs());
        }
        assert!(max_err < 1e-2, "identity reconstruction error {max_err}");
    }

    #[test]
    fn preserves_pitch_when_shifting_formants() {
        let mut shifter = FormantShifter::new(FRAME, SAMPLE_RATE, MAX_BLOCK);
        shifter.set_shift_semitones(7.0); // shift formants up a fifth
        let input = make_sawtooth(220.0, SAMPLE_RATE, 20_000);
        let out = run(&mut shifter, &input);
        let freq = dominant_frequency_hz(&out, SAMPLE_RATE);
        // The fundamental must stay put (within the 4096-DFT bin resolution).
        assert!((freq - 220.0).abs() < 12.0, "pitch drifted to {freq}");
    }

    #[test]
    fn upward_shift_raises_spectral_centroid() {
        let input = make_sawtooth(220.0, SAMPLE_RATE, 20_000);

        let mut flat = FormantShifter::new(FRAME, SAMPLE_RATE, MAX_BLOCK);
        flat.set_shift_semitones(0.0);
        let base = run(&mut flat, &input);

        let mut up = FormantShifter::new(FRAME, SAMPLE_RATE, MAX_BLOCK);
        up.set_shift_semitones(12.0); // formants up an octave
        let shifted = run(&mut up, &input);

        let base_c = spectral_centroid(&base, SAMPLE_RATE);
        let up_c = spectral_centroid(&shifted, SAMPLE_RATE);
        assert!(
            up_c > base_c,
            "upward formant shift should raise centroid: {up_c} vs {base_c}"
        );
    }

    #[test]
    fn reset_returns_to_silence() {
        let mut shifter = FormantShifter::new(FRAME, SAMPLE_RATE, MAX_BLOCK);
        shifter.set_shift_semitones(5.0);
        let input = make_sawtooth(220.0, SAMPLE_RATE, 4096);
        run(&mut shifter, &input);
        shifter.reset();
        let silence = vec![0.0f32; 4096];
        let out = run(&mut shifter, &silence);
        assert!(rms(&out) < 1e-6, "rms after reset = {}", rms(&out));
    }
}
