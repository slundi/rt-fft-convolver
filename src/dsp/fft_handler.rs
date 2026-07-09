use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

/// Wrapper around `realfft` for forward (real→complex) and inverse
/// (complex→real) FFTs.
///
/// A real-input FFT of size `N` produces a spectrum with only `N/2 + 1`
/// unique complex bins — the remaining bins are conjugate-symmetric
/// mirrors and are omitted from the stored representation.  Compared
/// with a complex-in / complex-out plan of the same size, this
///
/// - halves the memory footprint of every spectrum buffer (fewer
///   allocations, better cache utilisation), and
/// - roughly halves the FFT and IFFT arithmetic cost by exploiting the
///   redundancy of a real input.
///
/// All scratch and working buffers are allocated once at construction so
/// that [`forward`] and [`inverse`] are **allocation-free** and safe to
/// call from the audio render thread.
///
/// [`forward`]: FftHandler::forward
/// [`inverse`]: FftHandler::inverse
pub struct FftHandler {
    /// Pre-planned real-to-complex forward transform of length `N`.
    fft: Arc<dyn RealToComplex<f32>>,
    /// Pre-planned complex-to-real inverse transform of length `N`.
    ifft: Arc<dyn ComplexToReal<f32>>,
    /// Scratch buffer large enough for both directions.
    scratch: Vec<Complex<f32>>,
    /// Owned mutable buffer that receives the real input before the
    /// forward transform runs.  realfft may clobber the input buffer
    /// during processing, so we always copy the caller's slice into
    /// this internally-owned scratch.
    fwd_input_buf: Vec<f32>,
}

impl FftHandler {
    /// Create a new `FftHandler` for transforms of `size` real samples.
    ///
    /// # Panics
    /// Panics if `size == 0` or `size` is odd — realfft requires an even
    /// transform length so the spectrum has an integral Nyquist bin.
    pub fn new(size: usize) -> Self {
        assert!(size > 0, "FFT size must be > 0");
        assert!(
            size.is_multiple_of(2),
            "FFT size must be even for realfft (got {size})"
        );

        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(size);
        let ifft = planner.plan_fft_inverse(size);

        // Pre-allocate scratch large enough for both directions.
        let scratch_len = fft.get_scratch_len().max(ifft.get_scratch_len());

        Self {
            fft,
            ifft,
            scratch: vec![Complex::default(); scratch_len],
            fwd_input_buf: vec![0.0f32; size],
        }
    }

    /// Time-domain FFT length `N`.
    #[inline]
    pub fn size(&self) -> usize {
        self.fwd_input_buf.len()
    }

    /// Number of complex bins in a real-input spectrum (`N/2 + 1`).
    ///
    /// A real signal has a conjugate-symmetric FFT so only bins
    /// `0..=N/2` are unique; `bin[0]` (DC) and `bin[N/2]` (Nyquist) are
    /// purely real, the rest are general complex numbers.
    #[inline]
    pub fn spectrum_size(&self) -> usize {
        self.size() / 2 + 1
    }

    /// Forward real-to-complex FFT.
    ///
    /// Copies `input` into an internal buffer, runs the transform, and
    /// writes the `N/2 + 1` unique spectrum bins into `output`.
    ///
    /// `input.len()` must equal [`Self::size`] and `output.len()` must
    /// equal [`Self::spectrum_size`].
    ///
    /// # Real-time safety
    /// No heap allocation.  Safe to call from the audio render thread.
    pub fn forward(&mut self, input: &[f32], output: &mut [Complex<f32>]) {
        debug_assert_eq!(input.len(), self.size(), "forward: input length mismatch");
        debug_assert_eq!(
            output.len(),
            self.spectrum_size(),
            "forward: output length mismatch"
        );

        // realfft treats the input buffer as scratch — copy the caller's
        // slice into our own so their data isn't clobbered.
        self.fwd_input_buf.copy_from_slice(input);

        self.fft
            .process_with_scratch(&mut self.fwd_input_buf, output, &mut self.scratch)
            .expect("realfft forward: buffer lengths pre-asserted above");
    }

    /// Inverse complex-to-real FFT.
    ///
    /// Runs the inverse transform on `input` (which may be clobbered)
    /// and writes the normalised real samples into `output`.
    ///
    /// realfft, like rustfft, produces an **un-normalised** IDFT — each
    /// output sample is therefore divided by `N` to recover the correct
    /// amplitude.
    ///
    /// `input.len()` must equal [`Self::spectrum_size`] and
    /// `output.len()` must equal [`Self::size`].
    ///
    /// # Real-time safety
    /// No heap allocation.  Safe to call from the audio render thread.
    pub fn inverse(&mut self, input: &mut [Complex<f32>], output: &mut [f32]) {
        let n = self.size();
        debug_assert_eq!(
            input.len(),
            self.spectrum_size(),
            "inverse: input length mismatch"
        );
        debug_assert_eq!(output.len(), n, "inverse: output length mismatch");

        self.ifft
            .process_with_scratch(input, output, &mut self.scratch)
            .expect("realfft inverse: buffer lengths pre-asserted above");

        // Normalise by 1/N (rustfft/realfft both leave the IDFT un-scaled).
        let norm = (n as f32).recip();
        for sample in output.iter_mut() {
            *sample *= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const EPSILON: f32 = 1e-5;

    /// `spectrum_size()` returns `N/2 + 1` for every supported size.
    #[test]
    fn spectrum_size_is_half_plus_one() {
        for &n in &[2usize, 4, 8, 16, 32, 64, 128, 256] {
            let handler = FftHandler::new(n);
            assert_eq!(handler.size(), n, "size({n})");
            assert_eq!(handler.spectrum_size(), n / 2 + 1, "spectrum_size({n})");
        }
    }

    /// A unit impulse at position 0 should have a flat spectrum (all ones)
    /// across the `N/2 + 1` bins of the real-input FFT.
    #[test]
    fn impulse_has_flat_spectrum() {
        let n = 16;
        let mut handler = FftHandler::new(n);

        let mut input = vec![0.0f32; n];
        input[0] = 1.0; // unit impulse

        let mut spectrum = vec![Complex::default(); handler.spectrum_size()];
        handler.forward(&input, &mut spectrum);

        for bin in &spectrum {
            let magnitude = (bin.re * bin.re + bin.im * bin.im).sqrt();
            assert!(
                (magnitude - 1.0).abs() < EPSILON,
                "expected magnitude 1.0, got {magnitude}"
            );
        }
    }

    /// The DC bin of a real-input FFT equals the sum of the input samples
    /// and has zero imaginary part.
    #[test]
    fn dc_bin_matches_sum_and_is_real() {
        let n = 8;
        let mut handler = FftHandler::new(n);
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let expected_dc: f32 = input.iter().sum();

        let mut spectrum = vec![Complex::default(); handler.spectrum_size()];
        handler.forward(&input, &mut spectrum);

        assert!((spectrum[0].re - expected_dc).abs() < EPSILON);
        assert!(spectrum[0].im.abs() < EPSILON, "DC im = {}", spectrum[0].im);
    }

    /// The Nyquist bin of a real-input FFT is purely real for any real
    /// input signal.
    #[test]
    fn nyquist_bin_is_real() {
        let n = 16;
        let mut handler = FftHandler::new(n);
        // Any real-valued input — use a chirp to exercise every frequency.
        let input: Vec<f32> = (0..n).map(|i| ((i * i) as f32) * 0.05).collect();

        let mut spectrum = vec![Complex::default(); handler.spectrum_size()];
        handler.forward(&input, &mut spectrum);

        let nyquist_bin = spectrum[n / 2];
        assert!(
            nyquist_bin.im.abs() < EPSILON,
            "Nyquist im must be 0, got {}",
            nyquist_bin.im
        );
    }

    /// Forward followed by inverse reconstructs the original signal.
    #[test]
    fn roundtrip_reconstruction() {
        let n = 64;
        let mut handler = FftHandler::new(n);

        // Use a simple sine wave as test signal.
        let original: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 4.0 * i as f32 / n as f32).sin())
            .collect();

        let mut spectrum = vec![Complex::default(); handler.spectrum_size()];
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

    /// Linearity: FFT(a + b) == FFT(a) + FFT(b).
    #[test]
    fn linearity() {
        let n = 32;
        let mut handler = FftHandler::new(n);
        let sz = handler.spectrum_size();

        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
        let ab: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

        let mut fa = vec![Complex::default(); sz];
        let mut fb = vec![Complex::default(); sz];
        let mut fab = vec![Complex::default(); sz];

        handler.forward(&a, &mut fa);
        handler.forward(&b, &mut fb);
        handler.forward(&ab, &mut fab);

        for k in 0..sz {
            let sum = fa[k] + fb[k];
            assert!(
                (sum.re - fab[k].re).abs() < EPSILON && (sum.im - fab[k].im).abs() < EPSILON,
                "linearity failed at bin {k}"
            );
        }
    }

    /// Forward preserves the caller's input slice — realfft's internal
    /// clobbering happens on the private buffer only.
    #[test]
    fn forward_does_not_mutate_caller_input() {
        let n = 8;
        let mut handler = FftHandler::new(n);
        let input = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let snapshot = input;

        let mut spectrum = vec![Complex::default(); handler.spectrum_size()];
        handler.forward(&input, &mut spectrum);

        assert_eq!(input, snapshot, "forward must not mutate caller input");
    }
}
