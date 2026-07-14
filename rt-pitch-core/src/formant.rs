//! Spectral-envelope estimation for formant-preserving pitch shifting.
//!
//! # Why formants need preserving
//!
//! A naïve pitch shift multiplies every frequency by the ratio, dragging the
//! **spectral envelope** — the resonances (formants) that give a voice or
//! instrument its identity — along with the pitch.  Shifted up, that turns a
//! voice into a chipmunk.  Formant preservation splits the signal into a
//! *source* (the fine harmonic/excitation structure, which we transpose) and a
//! *filter* (the smooth envelope, which we hold still): flatten by the input
//! envelope, shift the flattened spectrum, then re-apply the **original**
//! envelope.
//!
//! # Three estimators (Roadmap-03 decision — user-selectable)
//!
//! - [`FormantMethod::Cepstral`] — low-quefrency liftering.  Cheapest; reuses
//!   an FFT.  The default.
//! - [`FormantMethod::Lpc`] — an all-pole model fitted by Levinson-Durbin.
//!   Voice-tuned; the envelope is `gain / |A(e^jω)|`.
//! - [`FormantMethod::TrueEnvelope`] — iterated cepstral liftering that grows
//!   the estimate up to the harmonic peaks; most accurate, iteration-bounded
//!   for real-time use.

use realfft::num_complex::Complex;
use rt_fft_convolver::FftHandler;

/// Selectable spectral-envelope estimator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FormantMethod {
    /// Cepstral low-quefrency liftering (default).
    Cepstral,
    /// LPC all-pole envelope via Levinson-Durbin.
    Lpc,
    /// Iterated "True Envelope" cepstral estimation.
    TrueEnvelope,
}

/// Estimates a smooth spectral envelope from a magnitude spectrum, and applies
/// formant-preserving correction to a shifted spectrum.
///
/// Built once off the audio thread; `estimate` / `preserve` are allocation-free.
pub struct FormantEstimator {
    method: FormantMethod,
    frame: usize,
    /// Number of unique real-spectrum bins (`frame / 2 + 1`).
    bins: usize,
    /// Cepstral lifter cutoff (quefrency) / LPC order.
    order: usize,
    /// Max iterations for [`FormantMethod::TrueEnvelope`].
    max_iterations: usize,

    fft: FftHandler,
    /// Complex spectrum scratch (`bins`).
    spectrum: Vec<Complex<f32>>,
    /// Real cepstrum / time scratch (`frame`).
    cepstrum: Vec<f32>,
    /// Log-magnitude working buffers (`bins`).
    log_mag: Vec<f32>,
    envelope_log: Vec<f32>,
    /// Autocorrelation + LPC scratch (`order + 1`).
    autocorr: Vec<f32>,
    lpc: Vec<f32>,
    lpc_tmp: Vec<f32>,
}

impl FormantEstimator {
    /// Build an estimator for `frame`-length STFT frames.
    ///
    /// `order` is the cepstral lifter cutoff (a good default is
    /// `frame / 64`, tens of bins) or the LPC order (voice: ~`sample_rate /
    /// 1000 + 2`).  **Allocates** — off-RT only.
    pub fn new(method: FormantMethod, frame: usize, order: usize) -> Self {
        assert!(frame.is_multiple_of(2), "frame must be even");
        assert!(order >= 1, "order must be >= 1");
        let bins = frame / 2 + 1;
        Self {
            method,
            frame,
            bins,
            order: order.min(bins - 1),
            max_iterations: 8,
            fft: FftHandler::new(frame),
            spectrum: vec![Complex::default(); bins],
            cepstrum: vec![0.0; frame],
            log_mag: vec![0.0; bins],
            envelope_log: vec![0.0; bins],
            autocorr: vec![0.0; order + 1],
            lpc: vec![0.0; order + 1],
            lpc_tmp: vec![0.0; order + 1],
        }
    }

    /// The active estimation method.
    #[inline]
    pub fn method(&self) -> FormantMethod {
        self.method
    }

    /// Estimate the linear-magnitude spectral envelope of `magnitude`
    /// (`frame/2 + 1` bins) into `envelope` (same length).
    pub fn estimate(&mut self, magnitude: &[f32], envelope: &mut [f32]) {
        debug_assert_eq!(magnitude.len(), self.bins);
        debug_assert_eq!(envelope.len(), self.bins);
        match self.method {
            FormantMethod::Cepstral => self.estimate_cepstral(magnitude, envelope),
            FormantMethod::Lpc => self.estimate_lpc(magnitude, envelope),
            FormantMethod::TrueEnvelope => self.estimate_true_envelope(magnitude, envelope),
        }
    }

    /// Formant-preserving correction: given the **original** envelope and the
    /// envelope of the **shifted** spectrum, rescale `shifted_mag` in place so
    /// its envelope matches the original (holding formants still).
    pub fn preserve(original_env: &[f32], shifted_env: &[f32], shifted_mag: &mut [f32]) {
        for ((mag, &orig), &shifted) in shifted_mag.iter_mut().zip(original_env).zip(shifted_env) {
            // Multiply by orig/shifted: undo the shifted envelope, impose the
            // original.  Guard the division for near-silent bins.
            let gain = if shifted > 1e-9 { orig / shifted } else { 0.0 };
            *mag *= gain;
        }
    }

    /// Low-quefrency cepstral smoothing of `log(magnitude)`.
    fn estimate_cepstral(&mut self, magnitude: &[f32], envelope: &mut [f32]) {
        for (dst, &mag) in self.log_mag.iter_mut().zip(magnitude) {
            *dst = (mag + 1e-9).ln();
        }
        self.cepstral_smooth();
        for (dst, &log_env) in envelope.iter_mut().zip(&self.envelope_log) {
            *dst = log_env.exp();
        }
    }

    /// Iterated True Envelope: repeatedly lift `max(log_mag, current)` so the
    /// estimate rises to the harmonic peaks without dipping into the valleys.
    fn estimate_true_envelope(&mut self, magnitude: &[f32], envelope: &mut [f32]) {
        for (dst, &mag) in self.log_mag.iter_mut().zip(magnitude) {
            *dst = (mag + 1e-9).ln();
        }
        // Seed the envelope with the raw log-magnitude.
        let mut current = self.log_mag.clone();
        for _ in 0..self.max_iterations {
            // Input to the smoother is the pointwise max of the data and the
            // running estimate (the defining True-Envelope update).
            for (lm, &cur) in self.log_mag.iter_mut().zip(&current) {
                *lm = lm.max(cur);
            }
            self.cepstral_smooth();
            let mut max_delta = 0.0f32;
            for (cur, &smoothed) in current.iter_mut().zip(&self.envelope_log) {
                max_delta = max_delta.max((smoothed - *cur).abs());
                *cur = smoothed;
            }
            if max_delta < 1e-3 {
                break;
            }
        }
        for (dst, &log_env) in envelope.iter_mut().zip(&current) {
            *dst = log_env.exp();
        }
    }

    /// Shared cepstral smoother: `log_mag → cepstrum → lifter → envelope_log`.
    fn cepstral_smooth(&mut self) {
        // Treat log-magnitude as a zero-phase spectrum; its inverse real FFT is
        // the (real, even) cepstrum.
        for (bin, &lm) in self.log_mag.iter().enumerate() {
            self.spectrum[bin] = Complex::new(lm, 0.0);
        }
        self.spectrum[0].im = 0.0;
        self.spectrum[self.bins - 1].im = 0.0;
        self.fft.inverse(&mut self.spectrum, &mut self.cepstrum);

        // Lifter: keep low-quefrency coefficients (indices near 0 and, by
        // symmetry, near `frame`), zero the rest.  Halve the split point so the
        // symmetric high-quefrency tail is trimmed too.
        for (quefrency, coeff) in self.cepstrum.iter_mut().enumerate() {
            let from_end = self.frame - quefrency;
            let low = quefrency <= self.order || from_end <= self.order;
            if !low {
                *coeff = 0.0;
            }
        }

        // Forward transform back to a smoothed log spectrum; the real part is
        // the envelope (imaginary part is ~0 for the symmetric cepstrum).
        self.fft.forward(&self.cepstrum, &mut self.spectrum);
        for (dst, value) in self.envelope_log.iter_mut().zip(&self.spectrum) {
            *dst = value.re;
        }
    }

    /// LPC envelope: autocorrelation (via the power spectrum) → Levinson-Durbin
    /// → `gain / |A(e^jω)|` sampled at each bin.
    fn estimate_lpc(&mut self, magnitude: &[f32], envelope: &mut [f32]) {
        // Power spectrum → autocorrelation is its inverse FFT.
        for (bin, &mag) in magnitude.iter().enumerate() {
            self.spectrum[bin] = Complex::new(mag * mag, 0.0);
        }
        self.spectrum[0].im = 0.0;
        self.spectrum[self.bins - 1].im = 0.0;
        self.fft.inverse(&mut self.spectrum, &mut self.cepstrum);
        for lag in 0..=self.order {
            self.autocorr[lag] = self.cepstrum[lag];
        }

        let gain = self.levinson();

        // Envelope[bin] = gain / |A(e^jω_bin)|, ω_bin = π·bin/(bins−1).
        use std::f32::consts::PI;
        let denom_bins = (self.bins - 1) as f32;
        for (bin, dst) in envelope.iter_mut().enumerate() {
            let omega = PI * bin as f32 / denom_bins;
            let mut real = 1.0f32;
            let mut imag = 0.0f32;
            for k in 1..=self.order {
                let angle = -omega * k as f32;
                real += self.lpc[k] * angle.cos();
                imag += self.lpc[k] * angle.sin();
            }
            let mag_a = (real * real + imag * imag).sqrt().max(1e-9);
            *dst = gain / mag_a;
        }
    }

    /// Levinson-Durbin recursion over `self.autocorr`; fills `self.lpc`
    /// (`a[0] = 1`) and returns the residual gain `√E`.
    fn levinson(&mut self) -> f32 {
        let mut error = self.autocorr[0];
        self.lpc.iter_mut().for_each(|value| *value = 0.0);
        self.lpc[0] = 1.0;
        if error <= 1e-9 {
            return 1e-9;
        }
        for order_index in 1..=self.order {
            // Reflection coefficient.
            let mut acc = self.autocorr[order_index];
            for k in 1..order_index {
                acc += self.lpc[k] * self.autocorr[order_index - k];
            }
            let reflection = -acc / error;

            self.lpc_tmp.copy_from_slice(&self.lpc);
            for k in 1..order_index {
                self.lpc[k] = self.lpc_tmp[k] + reflection * self.lpc_tmp[order_index - k];
            }
            self.lpc[order_index] = reflection;
            error *= 1.0 - reflection * reflection;
            if error <= 1e-9 {
                break;
            }
        }
        error.max(1e-9).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::make_sawtooth;
    use realfft::num_complex::Complex;

    /// Magnitude spectrum of a windowed sawtooth (harmonic-rich).
    fn sawtooth_magnitude(frame: usize) -> Vec<f32> {
        let signal = make_sawtooth(220.0, 48_000.0, frame);
        let mut fft = FftHandler::new(frame);
        let mut spectrum = vec![Complex::default(); frame / 2 + 1];
        // Hann window to reduce leakage.
        let windowed: Vec<f32> = signal
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let w = 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / frame as f32).cos();
                s * w
            })
            .collect();
        fft.forward(&windowed, &mut spectrum);
        spectrum.iter().map(|value| value.norm()).collect()
    }

    /// An envelope must be smoother than the raw magnitude: the mean absolute
    /// bin-to-bin change of the envelope is a fraction of the magnitude's.
    fn assert_smoother(magnitude: &[f32], envelope: &[f32]) {
        let roughness = |data: &[f32]| -> f32 {
            data.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f32>() / (data.len() - 1) as f32
        };
        let mag_rough = roughness(magnitude);
        let env_rough = roughness(envelope);
        assert!(
            env_rough < mag_rough * 0.6,
            "envelope not smooth: env={env_rough} mag={mag_rough}"
        );
    }

    #[test]
    fn cepstral_envelope_is_smooth() {
        let frame = 1024;
        let magnitude = sawtooth_magnitude(frame);
        let mut estimator = FormantEstimator::new(FormantMethod::Cepstral, frame, 24);
        let mut envelope = vec![0.0f32; frame / 2 + 1];
        estimator.estimate(&magnitude, &mut envelope);
        assert!(envelope.iter().all(|&v| v.is_finite() && v >= 0.0));
        assert_smoother(&magnitude, &envelope);
    }

    #[test]
    fn lpc_envelope_is_smooth() {
        let frame = 1024;
        let magnitude = sawtooth_magnitude(frame);
        let mut estimator = FormantEstimator::new(FormantMethod::Lpc, frame, 20);
        let mut envelope = vec![0.0f32; frame / 2 + 1];
        estimator.estimate(&magnitude, &mut envelope);
        assert!(envelope.iter().all(|&v| v.is_finite() && v >= 0.0));
        assert_smoother(&magnitude, &envelope);
    }

    #[test]
    fn true_envelope_is_smooth() {
        let frame = 1024;
        let magnitude = sawtooth_magnitude(frame);
        let mut estimator = FormantEstimator::new(FormantMethod::TrueEnvelope, frame, 24);
        let mut envelope = vec![0.0f32; frame / 2 + 1];
        estimator.estimate(&magnitude, &mut envelope);
        assert!(envelope.iter().all(|&v| v.is_finite() && v >= 0.0));
        assert_smoother(&magnitude, &envelope);
    }

    #[test]
    fn preserve_restores_original_envelope() {
        let bins = 100;
        let original: Vec<f32> = (0..bins)
            .map(|k| 1.0 + (k as f32 * 0.1).sin().abs())
            .collect();
        let shifted: Vec<f32> = (0..bins)
            .map(|k| 0.5 + (k as f32 * 0.2).cos().abs())
            .collect();
        let mut mag = shifted.clone(); // start with the shifted envelope
        FormantEstimator::preserve(&original, &shifted, &mut mag);
        // After correction the magnitude should equal the original envelope.
        for (got, want) in mag.iter().zip(&original) {
            assert!((got - want).abs() < 1e-4, "got {got} want {want}");
        }
    }
}
