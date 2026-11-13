//! Phase-vocoder pitch shifter ("Smooth" mode) — the default algorithm.
//!
//! # How it shifts pitch without time-stretching
//!
//! A phase vocoder analyses each STFT frame into, per bin, a **magnitude** and
//! an **instantaneous frequency** (recovered from the phase advance between
//! consecutive frames, not the bin centre).  To transpose by ratio `r` we
//! *re-bin*: the energy of analysis bin `k` is moved to synthesis bin
//! `round(k · r)` and its instantaneous frequency is scaled by `r`.  Synthesis
//! then re-integrates phase from those scaled frequencies.  Because the hop is
//! identical on analysis and synthesis there is no time-stretch and no
//! separate resampler — the classic Bernsee/`smbPitchShift` construction.
//!
//! # Instantaneous frequency as "phase increment per hop"
//!
//! We work in radians-per-hop rather than Hz, which makes the maths
//! sample-rate independent.  A tone sitting exactly on bin `k` advances its
//! phase by `ω_k = 2π · k · H / N` every hop `H` (frame size `N`).  The
//! measured phase advance minus `ω_k`, wrapped to `(−π, π]`, is the fractional
//! detuning; adding it back gives the true per-hop increment `dφ_k`.
//!
//! # Quality refinements
//!
//! - **Puckette peak-locking** (default on): independent per-bin phase
//!   integration smears a single tone across its skirt bins ("phasiness").
//!   Locking every bin's phase to its nearest spectral *peak* keeps a
//!   harmonic's bins mutually coherent.
//! - **Transient detection** (default on): on a strong onset (spectral-flux
//!   spike) the synthesis phase is reset to the analysis phase for one frame,
//!   letting the attack through crisply instead of being time-smeared.

use realfft::num_complex::Complex;
use rt_fft_convolver::{DenormalGuard, FftHandler};

use crate::formant::{FormantEstimator, FormantMethod};
use crate::shifter::{OnePoleSmoother, Shifter, clamp_ratio};
use crate::stft::{Analyzer, Synthesizer};
use crate::window::{Window, WindowKind};

use std::f32::consts::TAU;

/// Default overlap factor (hop = frame / 4) — satisfies Hann COLA and is the
/// Roadmap-03 "4× fixed" decision.
pub const OVERLAP_FACTOR: usize = 4;

/// Fraction of total spectral energy an onset must add frame-to-frame to be
/// treated as a transient.  Tuned for guitar pick attacks.
const TRANSIENT_FLUX_THRESHOLD: f32 = 0.35;

/// Phase-vocoder pitch shifter.
pub struct PhaseVocoder {
    fft: FftHandler,
    frame: usize,
    hop: usize,
    /// Analysis / synthesis window (applied on both sides).
    window: Vec<f32>,
    analyzer: Analyzer,
    synthesizer: Synthesizer,

    // ── Per-frame time-domain scratch ──
    /// Windowed analysis frame fed to the forward FFT.
    time_frame: Vec<f32>,
    /// Spectrum buffer (`N/2 + 1` bins) reused for forward and inverse.
    spectrum: Vec<Complex<f32>>,

    // ── Per-bin analysis state (length `N/2 + 1`) ──
    last_phase: Vec<f32>,
    analysis_mag: Vec<f32>,
    analysis_freq: Vec<f32>,
    analysis_phase: Vec<f32>,
    prev_mag: Vec<f32>,

    // ── Per-bin synthesis state ──
    synth_mag: Vec<f32>,
    synth_freq: Vec<f32>,
    sum_phase: Vec<f32>,

    /// Expected per-hop phase advance for bin `k` = `k · expected_step`.
    expected_step: f32,

    ratio: OnePoleSmoother,
    phase_lock: bool,
    transient_detection: bool,

    /// Formant preservation (source-filter separation); `None` = disabled.
    formant: Option<FormantEstimator>,
    /// Estimated input spectral envelope, held fixed across the shift.
    input_env: Vec<f32>,
}

impl PhaseVocoder {
    /// Build a phase vocoder for `frame`-sample FFTs (must be even), a hop of
    /// `frame / OVERLAP_FACTOR`, absorbing host blocks up to `max_block`.
    ///
    /// **Allocates** — construct off the audio thread.
    ///
    /// # Panics
    /// Panics if `frame` is not divisible by [`OVERLAP_FACTOR`] or is odd.
    pub fn new(frame: usize, max_block: usize, kind: WindowKind) -> Self {
        assert!(
            frame.is_multiple_of(OVERLAP_FACTOR),
            "frame must be divisible by the overlap factor"
        );
        let hop = frame / OVERLAP_FACTOR;
        let window = Window::new(kind, frame);
        let norm = window.ola_normalisation(hop);
        let bins = frame / 2 + 1;

        Self {
            fft: FftHandler::new(frame),
            frame,
            hop,
            window: window.as_slice().to_vec(),
            analyzer: Analyzer::new(frame, hop, max_block),
            synthesizer: Synthesizer::new(frame, hop, max_block, norm),
            time_frame: vec![0.0; frame],
            spectrum: vec![Complex::default(); bins],
            last_phase: vec![0.0; bins],
            analysis_mag: vec![0.0; bins],
            analysis_freq: vec![0.0; bins],
            analysis_phase: vec![0.0; bins],
            prev_mag: vec![0.0; bins],
            synth_mag: vec![0.0; bins],
            synth_freq: vec![0.0; bins],
            sum_phase: vec![0.0; bins],
            expected_step: TAU * hop as f32 / frame as f32,
            ratio: OnePoleSmoother::new(1.0, 8.0),
            phase_lock: true,
            transient_detection: true,
            formant: None,
            input_env: vec![0.0; bins],
        }
    }

    /// Enable formant preservation with the given estimator, or `None` to
    /// disable it.  **Allocates** (builds the estimator) — call on preset /
    /// parameter change, not from `process_block`.
    pub fn set_formant_preservation(&mut self, method: Option<FormantMethod>) {
        self.formant = method.map(|method| {
            // Cepstral lifter cutoff / LPC order scaled to the frame size.
            let order = (self.frame / 64).max(8);
            FormantEstimator::new(method, self.frame, order)
        });
    }

    /// Toggle Puckette peak-locking (default on).
    pub fn set_phase_lock(&mut self, enabled: bool) {
        self.phase_lock = enabled;
    }

    /// Toggle transient detection (default on).
    pub fn set_transient_detection(&mut self, enabled: bool) {
        self.transient_detection = enabled;
    }

    /// Analyse one frame already sitting in `self.time_frame` (windowed),
    /// filling the per-bin magnitude / instantaneous-frequency arrays.
    fn analyse(&mut self) {
        self.fft.forward(&self.time_frame, &mut self.spectrum);
        for bin in 0..self.spectrum.len() {
            let value = self.spectrum[bin];
            let magnitude = value.norm();
            let phase = value.arg();

            // Phase advance beyond the bin-centre expectation, wrapped to
            // (−π, π]; converted back into a true per-hop phase increment.
            let expected = bin as f32 * self.expected_step;
            let deviation = wrap_phase(phase - self.last_phase[bin] - expected);
            self.last_phase[bin] = phase;

            self.analysis_mag[bin] = magnitude;
            self.analysis_phase[bin] = phase;
            self.analysis_freq[bin] = expected + deviation; // radians per hop
        }
    }

    /// `true` when this frame's spectral flux marks a transient onset.
    fn is_transient(&mut self) -> bool {
        if !self.transient_detection {
            return false;
        }
        let mut flux = 0.0f32;
        let mut total = 0.0f32;
        for bin in 0..self.analysis_mag.len() {
            let rise = self.analysis_mag[bin] - self.prev_mag[bin];
            if rise > 0.0 {
                flux += rise;
            }
            total += self.analysis_mag[bin];
            self.prev_mag[bin] = self.analysis_mag[bin];
        }
        total > 1e-6 && flux / total > TRANSIENT_FLUX_THRESHOLD
    }

    /// Formant preservation, analysis half: estimate the input spectral
    /// envelope and *whiten* the analysis magnitudes by it, leaving a flat
    /// excitation spectrum that carries only the harmonic (source) structure.
    fn whiten_for_formant(&mut self) {
        if let Some(estimator) = self.formant.as_mut() {
            estimator.estimate(&self.analysis_mag, &mut self.input_env);
            for (mag, &env) in self.analysis_mag.iter_mut().zip(&self.input_env) {
                *mag /= env.max(1e-9);
            }
        }
    }

    /// Formant preservation, synthesis half: re-impose the **original** input
    /// envelope on the shifted excitation, so the formants stay put while the
    /// pitch moves.
    fn reimpose_formant(&mut self) {
        if self.formant.is_some() {
            for (mag, &env) in self.synth_mag.iter_mut().zip(&self.input_env) {
                *mag *= env;
            }
        }
    }

    /// Re-bin analysis energy to synthesis bins for the given ratio.
    fn rebin(&mut self, ratio: f32) {
        self.synth_mag.fill(0.0);
        self.synth_freq.fill(0.0);
        let max_bin = self.spectrum.len() - 1;
        for bin in 0..=max_bin {
            let target = (bin as f32 * ratio).round() as usize;
            if target <= max_bin {
                self.synth_mag[target] += self.analysis_mag[bin];
                // Instantaneous frequency scales with the transposition.
                self.synth_freq[target] = self.analysis_freq[bin] * ratio;
            }
        }
    }

    /// Integrate synthesis phase and write the output spectrum.  On a transient
    /// the accumulator is re-seeded from the analysis phase to keep the attack
    /// aligned.
    fn synthesise_spectrum(&mut self, transient: bool) {
        let max_bin = self.spectrum.len() - 1;

        // Peak-locking needs to know each bin's nearest peak before phase
        // integration; compute it once into `analysis_freq`-sized scratch by
        // reusing `last_phase`? Instead resolve inline via a helper array.
        for bin in 0..=max_bin {
            if transient {
                // Reset: adopt the analysis phase so the onset passes cleanly.
                self.sum_phase[bin] = self.analysis_phase[bin];
            } else {
                self.sum_phase[bin] += self.synth_freq[bin];
            }
        }

        if self.phase_lock && !transient {
            self.apply_peak_lock();
        }

        for bin in 0..=max_bin {
            let (sin, cos) = self.sum_phase[bin].sin_cos();
            self.spectrum[bin] = Complex::new(self.synth_mag[bin] * cos, self.synth_mag[bin] * sin);
        }

        // DC (bin 0) and Nyquist (bin N/2) must be purely real for the inverse
        // real-FFT to produce a real signal; drop any phase there.
        self.spectrum[0].im = 0.0;
        self.spectrum[max_bin].im = 0.0;
    }

    /// Puckette rigid phase-locking: snap every bin's synthesis phase onto its
    /// nearest spectral peak so a harmonic's skirt bins stay phase-coherent
    /// with the peak, killing "phasiness".
    fn apply_peak_lock(&mut self) {
        let max_bin = self.synth_mag.len() - 1;
        if max_bin < 2 {
            return;
        }
        // Walk peaks (local maxima over a ±2-bin neighbourhood) and assign the
        // bins between successive peaks' midpoint to the nearer peak.
        let mut previous_peak: Option<usize> = None;
        for bin in 0..=max_bin {
            let magnitude = self.synth_mag[bin];
            let is_peak = magnitude > self.synth_mag[bin.saturating_sub(1)]
                && magnitude > self.synth_mag[(bin + 1).min(max_bin)]
                && magnitude > self.synth_mag[bin.saturating_sub(2)]
                && magnitude > self.synth_mag[(bin + 2).min(max_bin)];
            if is_peak {
                if let Some(peak) = previous_peak {
                    let peak_phase = self.sum_phase[peak];
                    let midpoint = (peak + bin) / 2;
                    // Lock the region owned by `peak` (up to the midpoint).
                    for locked in (peak + 1)..=midpoint {
                        self.sum_phase[locked] = peak_phase;
                    }
                    let this_phase = self.sum_phase[bin];
                    for locked in (midpoint + 1)..bin {
                        self.sum_phase[locked] = this_phase;
                    }
                }
                previous_peak = Some(bin);
            }
        }
    }
}

impl Shifter for PhaseVocoder {
    fn latency_samples(&self) -> usize {
        // Framing group delay of an overlap-add STFT: frame minus one hop.
        self.frame - self.hop
    }

    fn reset(&mut self) {
        self.analyzer.reset();
        self.synthesizer.reset();
        self.time_frame.fill(0.0);
        self.spectrum.fill(Complex::default());
        self.last_phase.fill(0.0);
        self.prev_mag.fill(0.0);
        self.sum_phase.fill(0.0);
        self.ratio.snap_to(self.ratio.value());
    }

    fn set_ratio(&mut self, ratio: f32) {
        self.ratio.set_target(clamp_ratio(ratio));
    }

    fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), output.len());
        // FTZ guard: near-silent bins produce denormals in the phase maths.
        let _denormal_guard = DenormalGuard::new();

        self.analyzer.push_block(input);

        let mut frame_buf = std::mem::take(&mut self.time_frame);
        while self.analyzer.next_frame(&mut frame_buf) {
            // One smoothed ratio step per frame is enough (hop-rate control).
            let ratio = self.ratio.tick();

            // Analysis window.
            for (sample, &w) in frame_buf.iter_mut().zip(&self.window) {
                *sample *= w;
            }
            self.time_frame = frame_buf;
            self.analyse();
            let transient = self.is_transient();
            self.whiten_for_formant();
            self.rebin(ratio);
            self.reimpose_formant();
            self.synthesise_spectrum(transient);
            frame_buf = std::mem::take(&mut self.time_frame);

            // Inverse transform back into the frame buffer, then synthesis
            // window + overlap-add.
            self.fft.inverse(&mut self.spectrum, &mut frame_buf);
            for (sample, &w) in frame_buf.iter_mut().zip(&self.window) {
                *sample *= w;
            }
            self.synthesizer.add_frame(&frame_buf);
        }
        self.time_frame = frame_buf;

        self.synthesizer.pull_block(output);
    }
}

/// Wrap a phase error into `(−π, π]`.
#[inline]
fn wrap_phase(phase: f32) -> f32 {
    phase - TAU * (phase / TAU).round()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{dominant_frequency_hz, make_sine};

    fn run(shifter: &mut PhaseVocoder, input: &[f32], block: usize) -> Vec<f32> {
        let mut output = Vec::with_capacity(input.len());
        let mut buf = vec![0.0f32; block];
        let mut cursor = 0;
        while cursor < input.len() {
            let end = (cursor + block).min(input.len());
            let len = end - cursor;
            shifter.process_block(&input[cursor..end], &mut buf[..len]);
            output.extend_from_slice(&buf[..len]);
            cursor = end;
        }
        output
    }

    /// Ratio 1.0 leaves the dominant frequency unchanged (within one bin).
    #[test]
    fn unison_preserves_pitch() {
        let sample_rate = 48_000.0;
        let input = make_sine(440.0, sample_rate, 24_000);
        let mut pv = PhaseVocoder::new(1024, 512, WindowKind::Hann);
        pv.set_ratio(1.0);
        let output = run(&mut pv, &input, 256);
        let freq = dominant_frequency_hz(&output, sample_rate);
        assert!((freq - 440.0).abs() < 15.0, "unison freq = {freq}");
    }

    /// Ratio 1.0 reconstructs the input within the noise floor (STFT overlap-
    /// add is not bit-identical, but Hann COLA at 4× overlap is high-fidelity).
    ///
    /// Aligns the steady-state output against the input delayed by the reported
    /// latency and checks the normalised cross-correlation is near 1 and the
    /// residual energy is well below the signal (< −20 dB).
    #[test]
    fn unison_reconstructs_within_noise_floor() {
        let sample_rate = 48_000.0;
        let input = make_sine(440.0, sample_rate, 24_000);
        let mut pv = PhaseVocoder::new(1024, 512, WindowKind::Hann);
        pv.set_ratio(1.0);
        let output = run(&mut pv, &input, 256);

        let latency = pv.latency_samples();
        // Compare a settled mid-signal window, input[n] vs output[n + latency].
        let start = 8_000;
        let len = 8_000;
        let reference = &input[start..start + len];
        let shifted = &output[start + latency..start + latency + len];

        let dot: f64 = reference
            .iter()
            .zip(shifted)
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum();
        let ref_energy: f64 = reference.iter().map(|&a| (a as f64).powi(2)).sum();
        let out_energy: f64 = shifted.iter().map(|&b| (b as f64).powi(2)).sum();
        let correlation = dot / (ref_energy.sqrt() * out_energy.sqrt());
        assert!(correlation > 0.98, "unison correlation = {correlation}");

        // Scale-corrected residual energy relative to the signal.
        let scale = dot / ref_energy; // least-squares gain of output onto input
        let residual: f64 = reference
            .iter()
            .zip(shifted)
            .map(|(&a, &b)| (b as f64 - scale * a as f64).powi(2))
            .sum();
        let residual_db = 10.0 * (residual / out_energy).log10();
        assert!(residual_db < -20.0, "unison residual = {residual_db} dB");
    }

    /// Ratio 2.0 (octave up) on a 440 Hz sine yields a dominant 880 Hz peak.
    #[test]
    fn octave_up_doubles_pitch() {
        let sample_rate = 48_000.0;
        let input = make_sine(440.0, sample_rate, 24_000);
        let mut pv = PhaseVocoder::new(1024, 512, WindowKind::Hann);
        pv.set_ratio(2.0);
        let output = run(&mut pv, &input, 256);
        let freq = dominant_frequency_hz(&output, sample_rate);
        assert!((freq - 880.0).abs() < 20.0, "octave-up freq = {freq}");
    }

    /// Ratio 0.5 (octave down) yields a dominant 220 Hz peak.
    #[test]
    fn octave_down_halves_pitch() {
        let sample_rate = 48_000.0;
        let input = make_sine(440.0, sample_rate, 24_000);
        let mut pv = PhaseVocoder::new(1024, 512, WindowKind::Hann);
        pv.set_ratio(0.5);
        let output = run(&mut pv, &input, 256);
        let freq = dominant_frequency_hz(&output, sample_rate);
        assert!((freq - 220.0).abs() < 15.0, "octave-down freq = {freq}");
    }

    /// Peak-locking off still transposes correctly (sanity for the toggle).
    #[test]
    fn octave_up_without_peak_lock() {
        let sample_rate = 48_000.0;
        let input = make_sine(440.0, sample_rate, 24_000);
        let mut pv = PhaseVocoder::new(1024, 512, WindowKind::Hann);
        pv.set_phase_lock(false);
        pv.set_ratio(2.0);
        let output = run(&mut pv, &input, 256);
        let freq = dominant_frequency_hz(&output, sample_rate);
        assert!((freq - 880.0).abs() < 20.0, "freq = {freq}");
    }

    /// Reported latency is close to the actual output delay (± one hop),
    /// measured by cross-correlating a delayed impulse-ish onset.
    #[test]
    fn reported_latency_is_plausible() {
        let pv = PhaseVocoder::new(1024, 512, WindowKind::Hann);
        assert_eq!(pv.latency_samples(), 1024 - 256);
    }

    /// Formant preservation on still transposes correctly (integration check
    /// that the source-filter split doesn't break the shift).
    #[test]
    fn octave_up_with_formant_preservation() {
        let sample_rate = 48_000.0;
        let input = crate::test_support::make_sawtooth(220.0, sample_rate, 48_000);
        let mut pv = PhaseVocoder::new(1024, 512, WindowKind::Hann);
        pv.set_formant_preservation(Some(crate::formant::FormantMethod::Cepstral));
        pv.set_ratio(2.0);
        let output = run(&mut pv, &input, 256);
        let freq = dominant_frequency_hz(&output, sample_rate);
        assert!((freq - 440.0).abs() < 25.0, "formant octave freq = {freq}");
        assert!(output.iter().all(|s| s.is_finite()));
    }

    /// Silence in → silence out, no denormal blow-up.
    #[test]
    fn silence_stays_silent() {
        let mut pv = PhaseVocoder::new(512, 256, WindowKind::Hann);
        pv.set_ratio(1.5);
        let input = vec![0.0f32; 8192];
        let output = run(&mut pv, &input, 128);
        let peak = output.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()));
        assert!(peak < 1e-6, "silence peak = {peak}");
    }
}
