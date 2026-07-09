//! Offline high-fidelity convolution for studio rendering.
//!
//! # When to use
//!
//! The [`UniformPartitionEngine`] optimises for **real-time** operation:
//! constant per-block cost, in-place `f32` arithmetic, and pre-allocated
//! scratch to keep the audio thread allocation-free.  Those constraints cost
//! a small amount of numerical precision — a partitioned `f32` FFT roundtrip
//! accumulates ~1e-6 relative error per partition and the sum across
//! hundreds of partitions approaches 1e-4.
//!
//! For **offline rendering** (bounce, print, master) there is no need to
//! obey real-time constraints, and the precision bump matters:
//!
//! - `f64` mantissa gives ~15 significant digits vs `f32`'s ~7.
//! - A single overlap-add pass with an IR-sized FFT touches every input
//!   sample once, without cross-partition accumulation, so numerical noise
//!   stays lower and does not depend on the audio block size.
//!
//! # Algorithm
//!
//! Standard **overlap-add** with a fixed FFT size chosen as
//! `next_power_of_two(2 · L)` where `L = ir.len()`:
//!
//! ```text
//! block_size = fft_size − L + 1
//!
//! for each chunk of block_size input samples starting at position `p`:
//!     zero-pad chunk to fft_size, convert to f64
//!     FFT(chunk)                             → X            (f64)
//!     Y = X × IR_spectrum                    (elementwise)
//!     IFFT(Y)                                → y            (length fft_size)
//!     output[p .. p + chunk_len + L − 1] += y[..chunk_len + L − 1] / fft_size
//! ```
//!
//! Output length is `input.len() + ir.len() − 1` — the full linear
//! convolution, without trimming.  All accumulation happens in `f64`; the
//! final `output` is cast to `f32` on return.
//!
//! # Not real-time safe
//!
//! Every call to [`convolve`] allocates working buffers and runs multiple
//! FFTs whose count scales with the input length.  This is fine for offline
//! rendering (typically invoked once per song/section) but must never be
//! placed on the audio render thread.
//!
//! [`UniformPartitionEngine`]: crate::engine::partition::UniformPartitionEngine
//! [`convolve`]: OfflineConvolver::convolve

use std::sync::Arc;

use rustfft::{Fft, FftPlanner, num_complex::Complex};

/// Offline, non-partitioned, `f64`-precision convolver for studio rendering.
///
/// Instantiate once for a given IR; call [`convolve`] as many times as
/// needed with arbitrary-length inputs.  The IR spectrum is pre-computed
/// once at `f64` precision so repeated convolutions only pay for the input
/// transform.
///
/// **Not real-time safe** — allocates and runs multiple FFTs per call.
///
/// [`convolve`]: OfflineConvolver::convolve
pub struct OfflineConvolver {
    /// Original IR coefficients, kept for [`Self::ir`] introspection.
    ir: Vec<f32>,

    /// FFT size — always a power of two, ≥ 2 · `ir.len()` (and at least 2 so
    /// the planner never receives an invalid length).
    fft_size: usize,

    /// Pre-computed `f64` spectrum of the zero-padded IR.
    ir_spectrum: Vec<Complex<f64>>,

    /// Forward transform used for the per-block input FFT.
    fft: Arc<dyn Fft<f64>>,

    /// Inverse transform used for the per-block output IFFT.
    ifft: Arc<dyn Fft<f64>>,
}

impl OfflineConvolver {
    /// Create an offline convolver preloaded with `ir`.
    ///
    /// Pre-computes the IR spectrum at `f64` precision so subsequent
    /// [`convolve`] calls only pay for the input transforms.  Pass `&[]` to
    /// create a no-op convolver whose [`convolve`] returns an empty vector.
    ///
    /// **Allocates** — call from a non-real-time thread.
    ///
    /// [`convolve`]: Self::convolve
    pub fn new(ir: &[f32]) -> Self {
        // Overlap-add block size is fft_size − L + 1; choose fft_size ≈ 2·L
        // so block_size ≈ L, the classical trade-off between FFT cost and
        // number of blocks.  The `.max(1)` / `.max(2)` guards keep the FFT
        // planner input valid even for an empty IR (in which case
        // `convolve()` short-circuits before touching the FFT).
        let fft_size = (2 * ir.len().max(1)).next_power_of_two().max(2);

        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(fft_size);
        let ifft = planner.plan_fft_inverse(fft_size);

        // Copy the IR into a zero-padded complex buffer in f64 and transform.
        let mut ir_spectrum = vec![Complex::default(); fft_size];
        for (bin, &sample) in ir_spectrum.iter_mut().zip(ir.iter()) {
            *bin = Complex::new(sample as f64, 0.0);
        }
        fft.process(&mut ir_spectrum);

        Self {
            ir: ir.to_vec(),
            fft_size,
            ir_spectrum,
            fft,
            ifft,
        }
    }

    /// IR coefficients this convolver was built with.
    #[inline]
    pub fn ir(&self) -> &[f32] {
        &self.ir
    }

    /// Number of IR taps (`0` for an empty IR).
    #[inline]
    pub fn ir_len(&self) -> usize {
        self.ir.len()
    }

    /// FFT size used internally (power of two, ≥ 2 · IR length).
    ///
    /// Exposed for diagnostics / benchmarks — callers do not need this to
    /// use [`convolve`].
    ///
    /// [`convolve`]: Self::convolve
    #[inline]
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Convolve `input` with the stored IR and return the full linear
    /// convolution of length `input.len() + ir.len() − 1`.
    ///
    /// Returns an empty vector if the IR **or** the input is empty.
    ///
    /// All arithmetic runs in `f64`; the result is cast to `f32` only on the
    /// final write.
    ///
    /// **Not real-time safe** — allocates and runs multiple FFTs.
    pub fn convolve(&self, input: &[f32]) -> Vec<f32> {
        if self.ir.is_empty() || input.is_empty() {
            return Vec::new();
        }

        let ir_len = self.ir.len();
        let out_len = input.len() + ir_len - 1;

        // Accumulate the overlap-add result in f64 to preserve precision;
        // convert to f32 only at the end.
        let mut output = vec![0.0f64; out_len];

        // block_size = fft_size − L + 1: the largest chunk of new input we
        // can transform per FFT without wrap-around aliasing.  With
        // fft_size = next_pow2(2·L) this is ≥ L + 1.
        let block_size = self.fft_size - ir_len + 1;

        // Working buffer reused across every overlap-add chunk.
        let mut buf = vec![Complex::<f64>::default(); self.fft_size];
        // rustfft's IDFT is un-normalised, so divide each sample by N.
        let norm = 1.0f64 / self.fft_size as f64;

        let mut pos = 0;
        while pos < input.len() {
            let end = (pos + block_size).min(input.len());
            let chunk = &input[pos..end];
            let chunk_len = chunk.len();

            // Zero-pad: copy chunk into the first slots and zero the tail.
            // Written as two loops so the second one becomes a fill pattern
            // the compiler can vectorise.
            for (bin, &sample) in buf.iter_mut().zip(chunk.iter()) {
                *bin = Complex::new(sample as f64, 0.0);
            }
            for bin in buf.iter_mut().skip(chunk_len) {
                *bin = Complex::default();
            }

            // Forward FFT of the padded input chunk.
            self.fft.process(&mut buf);

            // Frequency-domain multiply against the pre-computed IR spectrum.
            for (bin, &h) in buf.iter_mut().zip(self.ir_spectrum.iter()) {
                *bin *= h;
            }

            // Back to time domain.
            self.ifft.process(&mut buf);

            // Overlap-add: this chunk contributes `chunk_len + L − 1`
            // linearly valid samples at position `pos`.  Anything past that
            // in `buf` is the wrap-around of the circular convolution and
            // must be discarded (chunk_len + L − 1 ≤ fft_size, so the
            // truncation is only ever needed for correctness of the last,
            // possibly-short chunk).
            let contrib_len = (chunk_len + ir_len - 1).min(self.fft_size);
            for (i, bin) in buf.iter().take(contrib_len).enumerate() {
                let idx = pos + i;
                // `idx < out_len` is always true because
                // pos + chunk_len + L − 1 ≤ input.len() + L − 1 = out_len,
                // but we keep the guard so future refactors can't overflow.
                if idx < out_len {
                    output[idx] += bin.re * norm;
                }
            }

            pos += block_size;
        }

        // Cast the f64 accumulator to f32 for the return value.
        output.into_iter().map(|x| x as f32).collect()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Naïve time-domain convolution reference in f64.
    ///
    /// Used as the ground truth for correctness checks.  O(N·L) — safe here
    /// because tests use small signals.
    fn naive(ir: &[f32], input: &[f32]) -> Vec<f64> {
        if ir.is_empty() || input.is_empty() {
            return Vec::new();
        }
        let out_len = input.len() + ir.len() - 1;
        let mut out = vec![0.0f64; out_len];
        for (idx, sample) in out.iter_mut().enumerate() {
            for (tap_idx, &tap) in ir.iter().enumerate() {
                if idx >= tap_idx && idx - tap_idx < input.len() {
                    *sample += tap as f64 * input[idx - tap_idx] as f64;
                }
            }
        }
        out
    }

    fn assert_close(got: &[f32], expected: &[f64], tol: f64) {
        assert_eq!(got.len(), expected.len(), "length mismatch");
        for (idx, (&value, &target)) in got.iter().zip(expected.iter()).enumerate() {
            let err = (value as f64 - target).abs();
            assert!(
                err < tol,
                "sample {idx}: expected {target}, got {value} (abs err {err} > tol {tol})"
            );
        }
    }

    // ── Edge cases ────────────────────────────────────────────────────────────

    /// Empty IR must produce an empty output regardless of input.
    #[test]
    fn empty_ir_returns_empty() {
        let conv = OfflineConvolver::new(&[]);
        assert_eq!(conv.ir_len(), 0);
        assert!(conv.convolve(&[1.0, 2.0, 3.0]).is_empty());
    }

    /// Empty input must produce an empty output regardless of IR.
    #[test]
    fn empty_input_returns_empty() {
        let conv = OfflineConvolver::new(&[1.0, 2.0, 3.0]);
        assert!(conv.convolve(&[]).is_empty());
    }

    /// Constructor with a non-empty IR selects a power-of-two FFT size that
    /// covers at least the linear-convolution length of one block.
    #[test]
    fn fft_size_is_power_of_two_covering_ir() {
        let conv = OfflineConvolver::new(&[1.0; 10]);
        // fft_size must be ≥ 2·L and a power of two.
        assert!(conv.fft_size() >= 20);
        assert!(conv.fft_size().is_power_of_two());
    }

    // ── Correctness ───────────────────────────────────────────────────────────

    /// A unit-impulse IR passes the input through unchanged for the first
    /// `input.len()` samples; the remainder is zero.
    #[test]
    fn unit_impulse_ir_is_identity() {
        let conv = OfflineConvolver::new(&[1.0]);
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let out = conv.convolve(&input);
        assert_eq!(out.len(), input.len());
        for (idx, (&value, &target)) in out.iter().zip(input.iter()).enumerate() {
            assert!(
                (value - target).abs() < 1e-6,
                "sample {idx}: expected {target}, got {value}"
            );
        }
    }

    /// IR = [0, 1] shifts the input by one sample.  With input [1,2,3,4] the
    /// full linear convolution is [0, 1, 2, 3, 4].
    #[test]
    fn one_sample_delay_ir() {
        let conv = OfflineConvolver::new(&[0.0, 1.0]);
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let out = conv.convolve(&input);
        let expected = [0.0f32, 1.0, 2.0, 3.0, 4.0];
        assert_eq!(out.len(), expected.len());
        for (idx, (&value, &target)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (value - target).abs() < 1e-6,
                "sample {idx}: expected {target}, got {value}"
            );
        }
    }

    /// Known two-tap convolution: y[n] = x[n] + 2·x[n-1].
    /// Input [1,2,3,4] ⇒ output [1, 4, 7, 10, 8] (length M+L-1 = 5).
    #[test]
    fn known_two_tap_convolution() {
        let conv = OfflineConvolver::new(&[1.0, 2.0]);
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let out = conv.convolve(&input);
        let expected = [1.0f32, 4.0, 7.0, 10.0, 8.0];
        assert_eq!(out.len(), expected.len());
        for (idx, (&value, &target)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (value - target).abs() < 1e-6,
                "sample {idx}: expected {target}, got {value}"
            );
        }
    }

    /// Full output length is `input.len() + ir.len() − 1`.
    #[test]
    fn output_length_is_full_linear_convolution() {
        let ir = [1.0f32, 2.0, 3.0];
        let input = [0.5f32, 1.5, 2.5, 3.5, 4.5];
        let conv = OfflineConvolver::new(&ir);
        let out = conv.convolve(&input);
        assert_eq!(out.len(), input.len() + ir.len() - 1);
    }

    // ── Cross-check against naïve reference ───────────────────────────────────

    /// Matches a naïve O(N·L) time-domain reference for a non-trivial IR.
    #[test]
    fn matches_naive_reference_short_input() {
        let ir: Vec<f32> = (0..17).map(|i| ((i as f32) * 0.3).sin() * 0.5).collect();
        let input: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.17).sin()).collect();
        let conv = OfflineConvolver::new(&ir);
        let out = conv.convolve(&input);
        let reference = naive(&ir, &input);
        // f64 FFT roundtrip is ~1e-14 relative; scale by input energy to
        // account for accumulation across taps.
        assert_close(&out, &reference, 1e-6);
    }

    /// Overlap-add across multiple internal blocks stays correct — this is
    /// the main invariant of the streaming path.  With a 3-tap IR the FFT
    /// size is 8 → block_size = 6, so a 1000-sample input spans 167 blocks.
    #[test]
    fn overlap_add_covers_multi_block_input() {
        let ir = [1.0f32, -0.5, 0.25];
        let conv = OfflineConvolver::new(&ir);
        let input: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.11).cos()).collect();
        let out = conv.convolve(&input);
        let reference = naive(&ir, &input);
        assert_close(&out, &reference, 1e-5);
    }

    /// Chunk boundary edge case: the final chunk is shorter than
    /// `block_size`.  Exercises the `contrib_len` clamp for the last block.
    #[test]
    fn partial_last_chunk_is_handled() {
        let ir = [1.0f32, 0.5, 0.25, 0.125];
        let conv = OfflineConvolver::new(&ir);
        // Non-multiple of the internal block_size (which is fft_size - 3
        // = 5 here, so 13 samples is 2 full + 1 partial chunk).
        let input: Vec<f32> = (0..13).map(|i| (i + 1) as f32).collect();
        let out = conv.convolve(&input);
        let reference = naive(&ir, &input);
        assert_close(&out, &reference, 1e-5);
    }

    /// Long IR + long input: the FFT block is large so most of the input
    /// fits in a single overlap-add block, exercising the "few big FFTs"
    /// code path.
    #[test]
    fn long_ir_matches_reference() {
        let ir: Vec<f32> = (0..129)
            .map(|i| ((i as f32) * 0.05).sin() * 0.1 - ((i as f32) * 0.02).cos() * 0.05)
            .collect();
        let input: Vec<f32> = (0..300).map(|i| ((i as f32) * 0.07).sin()).collect();
        let conv = OfflineConvolver::new(&ir);
        let out = conv.convolve(&input);
        let reference = naive(&ir, &input);
        assert_close(&out, &reference, 1e-5);
    }

    // ── Introspection ─────────────────────────────────────────────────────────

    /// `ir()` returns the coefficients passed to `new`.
    #[test]
    fn ir_accessor_returns_original_coefficients() {
        let src = [0.1f32, -0.2, 0.3, -0.4];
        let conv = OfflineConvolver::new(&src);
        assert_eq!(conv.ir(), &src);
        assert_eq!(conv.ir_len(), src.len());
    }
}
