use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::sync::Arc;

/// Wrapper around `rustfft` for forward (real→complex) and inverse (complex→real) FFTs.
///
/// All scratch and working buffers are allocated once at construction so that
/// `forward()` and `inverse()` are **allocation-free** and safe to call from
/// a real-time audio thread.
pub struct FftHandler {
    /// Pre-planned forward FFT (size N).
    fft: Arc<dyn Fft<f32>>,
    /// Pre-planned inverse FFT (size N).
    ifft: Arc<dyn Fft<f32>>,
    /// Scratch buffer whose length satisfies both forward and inverse requirements.
    scratch: Vec<Complex<f32>>,
    /// Temporary complex buffer used to convert a real slice before in-place FFT.
    work_buf: Vec<Complex<f32>>,
}

impl FftHandler {
    /// Create a new `FftHandler` for transforms of `size` samples.
    ///
    /// # Panics
    /// Panics if `size` is 0.
    pub fn new(size: usize) -> Self {
        assert!(size > 0, "FFT size must be > 0");

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(size);
        let ifft = planner.plan_fft_inverse(size);

        // Pre-allocate scratch large enough for both directions.
        let scratch_len = fft
            .get_inplace_scratch_len()
            .max(ifft.get_inplace_scratch_len());

        Self {
            fft,
            ifft,
            scratch: vec![Complex::default(); scratch_len],
            work_buf: vec![Complex::default(); size],
        }
    }

    /// Return the FFT size this handler was created for.
    #[inline]
    pub fn size(&self) -> usize {
        self.work_buf.len()
    }

    /// Forward real-to-complex FFT.
    ///
    /// Copies `input` (real samples) into an internal complex buffer with zero
    /// imaginary parts, runs the forward FFT in place, then writes the result
    /// into `output`.
    ///
    /// `input.len()` and `output.len()` must both equal [`Self::size()`].
    ///
    /// # Real-time safety
    /// No heap allocation. Safe to call from the audio render thread.
    pub fn forward(&mut self, input: &[f32], output: &mut [Complex<f32>]) {
        let n = self.size();
        debug_assert_eq!(input.len(), n, "forward: input length mismatch");
        debug_assert_eq!(output.len(), n, "forward: output length mismatch");

        // Lift real samples into the complex work buffer (imaginary part = 0).
        for (c, &r) in self.work_buf.iter_mut().zip(input.iter()) {
            *c = Complex { re: r, im: 0.0 };
        }

        self.fft
            .process_with_scratch(&mut self.work_buf, &mut self.scratch);

        output.copy_from_slice(&self.work_buf);
    }

    /// Inverse complex-to-real FFT.
    ///
    /// Runs the inverse FFT on `input` **in place** (the slice is mutated), then
    /// writes the normalised real parts into `output`.
    ///
    /// `rustfft` computes an un-normalised IDFT, so each output sample is
    /// divided by `N` to obtain the correct amplitude.
    ///
    /// `input.len()` and `output.len()` must both equal [`Self::size()`].
    ///
    /// # Real-time safety
    /// No heap allocation. Safe to call from the audio render thread.
    pub fn inverse(&mut self, input: &mut [Complex<f32>], output: &mut [f32]) {
        let n = self.size();
        debug_assert_eq!(input.len(), n, "inverse: input length mismatch");
        debug_assert_eq!(output.len(), n, "inverse: output length mismatch");

        self.ifft.process_with_scratch(input, &mut self.scratch);

        // Normalise by 1/N and extract real part.
        let norm = (n as f32).recip();
        for (out, c) in output.iter_mut().zip(input.iter()) {
            *out = c.re * norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const EPSILON: f32 = 1e-5;

    /// A unit impulse at position 0 should have a flat spectrum (all ones).
    #[test]
    fn impulse_has_flat_spectrum() {
        let n = 16;
        let mut handler = FftHandler::new(n);

        let mut input = vec![0.0f32; n];
        input[0] = 1.0; // unit impulse

        let mut spectrum = vec![Complex::default(); n];
        handler.forward(&input, &mut spectrum);

        for bin in &spectrum {
            // |H(k)| == 1 for all k when input is a unit impulse.
            let magnitude = (bin.re * bin.re + bin.im * bin.im).sqrt();
            assert!(
                (magnitude - 1.0).abs() < EPSILON,
                "expected magnitude 1.0, got {magnitude}"
            );
        }
    }

    /// Forward followed by inverse should reconstruct the original signal.
    #[test]
    fn roundtrip_reconstruction() {
        let n = 64;
        let mut handler = FftHandler::new(n);

        // Use a simple sine wave as test signal.
        let original: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 4.0 * i as f32 / n as f32).sin())
            .collect();

        let mut spectrum = vec![Complex::default(); n];
        handler.forward(&original, &mut spectrum);

        let mut reconstructed = vec![0.0f32; n];
        handler.inverse(&mut spectrum, &mut reconstructed);

        for (orig, recon) in original.iter().zip(reconstructed.iter()) {
            assert!(
                (orig - recon).abs() < EPSILON,
                "roundtrip mismatch: {orig} != {recon}"
            );
        }
    }

    /// Linearity check: FFT(a + b) == FFT(a) + FFT(b).
    #[test]
    fn linearity() {
        let n = 32;
        let mut handler = FftHandler::new(n);

        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
        let ab: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

        let mut fa = vec![Complex::default(); n];
        let mut fb = vec![Complex::default(); n];
        let mut fab = vec![Complex::default(); n];

        handler.forward(&a, &mut fa);
        handler.forward(&b, &mut fb);
        handler.forward(&ab, &mut fab);

        for k in 0..n {
            let sum = fa[k] + fb[k];
            assert!(
                (sum.re - fab[k].re).abs() < EPSILON && (sum.im - fab[k].im).abs() < EPSILON,
                "linearity failed at bin {k}"
            );
        }
    }
}
