//! Uniform partitioned convolution for the IR **tail**.
//!
//! # Why partition the IR?
//!
//! A direct time-domain convolution costs O(N·L) per block, where L is the IR
//! length.  For a 4-second guitar cabinet IR at 48 kHz (L ≈ 192 000 samples)
//! that is completely impractical in real time.  The FFT-based overlap-save
//! method reduces the per-block cost to O(N · log N · P) where P = ⌈L/N⌉ is
//! the number of partitions — a dramatic saving once N ≪ L.
//!
//! # Overlap-Save algorithm
//!
//! The IR tail (everything after the zero-latency head handled by
//! [`DirectConvolver`]) is split into equal **partitions** of `block_size`
//! samples.  Each partition is zero-padded to `2 × block_size` and
//! pre-transformed via FFT, giving a frequency-domain representation stored
//! in [`ir_spectra`].
//!
//! On every audio callback (one block of `block_size` samples):
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │ 1. Slide the 2N overlap-save window:                               │
//! │      input_buf = [prev_block | new_block]                          │
//! │                                                                     │
//! │ 2. FFT(input_buf) → current input spectrum X                       │
//! │                                                                     │
//! │ 3. Store X in the frequency-domain delay line (FDL) at fdl_pos    │
//! │                                                                     │
//! │ 4. acc = Σ_{k=0}^{P-1}  FDL[(pos−k) mod P]  ⊙  IR_spectra[k]   │
//! │    (complex multiply-accumulate over all partitions)                │
//! │                                                                     │
//! │ 5. IFFT(acc) → 2N time-domain samples                             │
//! │                                                                     │
//! │ 6. Discard first N samples (circular wrap-around artifact).        │
//! │    Add second N samples to output.                                  │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Latency
//!
//! This engine produces perfectly causal output: `output[t]` uses `input[t]`
//! through `input[t - L + 1]`, with zero algorithmic latency added beyond
//! what the IR itself dictates.  Pair it with a [`DirectConvolver`] for the
//! head partition (which has zero latency) to handle the full IR.
//!
//! [`DirectConvolver`]: crate::dsp::direct::DirectConvolver

use rustfft::num_complex::Complex;

use crate::dsp::FftHandler;
use crate::engine::fdl::FreqDomainDelayLine;

/// Uniform partitioned convolution engine for the IR tail.
///
/// Split the IR tail into equal partitions of [`block_size`] samples, then
/// process each audio block using the overlap-save algorithm in the frequency
/// domain.
///
/// Construct once on a non-real-time thread, then call [`process`] from the
/// audio render thread for each block.
///
/// [`block_size`]: UniformPartitionEngine::block_size
/// [`process`]: UniformPartitionEngine::process
pub struct UniformPartitionEngine {
    /// Audio block size N (= partition size).
    block_size: usize,

    /// Pre-computed 2N-point FFT of each IR tail partition.
    /// `ir_spectra[k]` is the spectrum of IR samples `k*N .. (k+1)*N`
    /// (zero-padded to 2N).
    ir_spectra: Vec<Vec<Complex<f32>>>,

    /// Frequency-domain delay line: circular buffer of the P most recent
    /// input spectra.  Updated once per block via [`FreqDomainDelayLine::push`].
    fdl: FreqDomainDelayLine,

    /// Overlap-save buffer of length 2N.
    /// `input_buf[0..N]`  = previous block (the "overlap").
    /// `input_buf[N..2N]` = current  block (the "new" data).
    input_buf: Vec<f32>,

    /// Frequency-domain accumulator for the multiply-accumulate step.
    /// Also used as a temporary buffer for the forward FFT result.
    acc_buf: Vec<Complex<f32>>,

    /// Time-domain IFFT output buffer of length 2N.
    /// The valid output lives in `out_buf[N..2N]`.
    out_buf: Vec<f32>,

    /// Pre-planned FFT / IFFT engine for 2N-point transforms.
    fft: FftHandler,
}

impl UniformPartitionEngine {
    /// Create an engine loaded with the given IR **tail** coefficients.
    ///
    /// `ir_tail` is the portion of the IR that comes *after* the head handled
    /// by a [`DirectConvolver`] — i.e. `ir[head_len..]`.  Pass `&[]` to
    /// create a no-op (silence) engine.
    ///
    /// `block_size` must equal the number of samples per audio render callback
    /// and must be > 0.
    ///
    /// **Allocates** — call from a non-real-time thread.
    ///
    /// # Panics
    ///
    /// Panics if `block_size == 0`.
    ///
    /// [`DirectConvolver`]: crate::dsp::direct::DirectConvolver
    pub fn new(ir_tail: &[f32], block_size: usize) -> Self {
        assert!(block_size > 0, "block_size must be > 0");

        let fft_size = 2 * block_size;

        let num_partitions = if ir_tail.is_empty() {
            0
        } else {
            ir_tail.len().div_ceil(block_size)
        };

        let mut fft = FftHandler::new(fft_size);

        // ── Pre-transform each IR partition ───────────────────────────────
        // Each partition is zero-padded from block_size to fft_size before
        // the FFT so that the circular convolution matches linear convolution
        // in the valid (second-half) output region.
        let ir_spectra: Vec<Vec<Complex<f32>>> = (0..num_partitions)
            .map(|k| {
                let start = k * block_size;
                let end = (start + block_size).min(ir_tail.len());

                let mut padded = vec![0.0f32; fft_size];
                padded[..end - start].copy_from_slice(&ir_tail[start..end]);

                let mut spectrum = vec![Complex::default(); fft_size];
                fft.forward(&padded, &mut spectrum);
                spectrum
            })
            .collect();

        // FDL needs at least one slot; it is never accessed when num_partitions
        // == 0 because process() returns early in that case.
        let fdl = FreqDomainDelayLine::new(num_partitions.max(1), fft_size);

        Self {
            block_size,
            ir_spectra,
            fdl,
            input_buf: vec![0.0f32; fft_size],
            acc_buf: vec![Complex::default(); fft_size],
            out_buf: vec![0.0f32; fft_size],
            fft,
        }
    }

    /// Number of IR tail partitions.
    #[inline]
    pub fn num_partitions(&self) -> usize {
        self.ir_spectra.len()
    }

    /// Audio block size this engine was created with.
    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Clear all delay lines and input history **without reallocating**.
    ///
    /// Call between song sections or preset changes to prevent clicks from
    /// stale tail energy.
    ///
    /// **Real-time safe** — no allocation.
    pub fn reset(&mut self) {
        self.input_buf.fill(0.0);
        self.fdl.reset();
        self.acc_buf.fill(Complex::default());
        self.out_buf.fill(0.0);
    }

    /// Replace the IR tail and reset all internal state.
    ///
    /// `block_size` is preserved from construction.
    ///
    /// **Allocates** — call from a non-real-time thread.
    pub fn load_ir(&mut self, ir_tail: &[f32]) {
        *self = Self::new(ir_tail, self.block_size);
    }

    /// Process one block of `block_size` input samples, **adding** the
    /// convolved tail into `output`.
    ///
    /// `output` is *accumulated into*, not overwritten.  The typical usage is:
    ///
    /// ```text
    /// output.fill(0.0);
    /// direct.process(input, output);          // zero-latency head
    /// tail_engine.process(input, output);     // add FFT tail
    /// ```
    ///
    /// Both slices must have length equal to [`Self::block_size()`].
    ///
    /// If the IR tail is empty this is a no-op (output is not touched).
    ///
    /// **Real-time safe** — no allocation, no locking, no panics in release
    /// builds.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(
            input.len(),
            self.block_size,
            "process: input length must equal block_size"
        );
        debug_assert_eq!(
            output.len(),
            self.block_size,
            "process: output length must equal block_size"
        );

        if self.ir_spectra.is_empty() {
            return;
        }

        let n = self.block_size;

        // ── Step 1: slide the overlap-save window ─────────────────────────
        // First half ← old second half (previous block).
        // Second half ← new input block.
        self.input_buf.copy_within(n.., 0);
        self.input_buf[n..].copy_from_slice(input);

        // ── Step 2: FFT the 2N window into acc_buf ────────────────────────
        // Split field borrows so the borrow checker sees distinct locations:
        //   &mut self.fft  (for the FFT engine)
        //   &    self.input_buf
        //   &mut self.acc_buf
        {
            let fft = &mut self.fft;
            fft.forward(&self.input_buf, &mut self.acc_buf);
        }

        // ── Step 3: push current spectrum into the FDL ────────────────────
        self.fdl.push(&self.acc_buf);

        // ── Step 4: frequency-domain multiply-accumulate ──────────────────
        // acc = Σ_{k=0}^{P-1}  FDL[k]  ⊙  IR[k]
        // (FDL slot 0 = most recently pushed; managed by FreqDomainDelayLine)
        self.acc_buf.fill(Complex::default());
        {
            let fdl = &self.fdl;
            let ir = &self.ir_spectra;
            fdl.accumulate(ir, &mut self.acc_buf);
        }

        // ── Step 5: IFFT the accumulator ──────────────────────────────────
        {
            let fft = &mut self.fft;
            fft.inverse(&mut self.acc_buf, &mut self.out_buf);
        }

        // ── Step 6: accumulate the valid second half into output ──────────
        // The first N samples are the circular wrap-around artifact; the
        // second N samples are the correct linear-convolution result.
        for (y, &x) in output.iter_mut().zip(self.out_buf[n..].iter()) {
            *y += x;
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    // ── Helper ────────────────────────────────────────────────────────────────

    /// Feed `blocks` of `block_size` samples through a fresh engine and
    /// collect all output samples into one flat Vec.
    fn run_blocks(ir_tail: &[f32], block_size: usize, blocks: &[&[f32]]) -> Vec<f32> {
        let mut engine = UniformPartitionEngine::new(ir_tail, block_size);
        let mut out = Vec::new();
        for &block in blocks {
            let mut buf = vec![0.0f32; block_size];
            engine.process(block, &mut buf);
            out.extend_from_slice(&buf);
        }
        out
    }

    // ── Structural ────────────────────────────────────────────────────────────

    /// An empty IR tail must report 0 partitions.
    #[test]
    fn empty_ir_has_zero_partitions() {
        let e = UniformPartitionEngine::new(&[], 4);
        assert_eq!(e.num_partitions(), 0);
    }

    /// An IR of 7 samples with block_size=4 needs ⌈7/4⌉ = 2 partitions.
    #[test]
    fn partition_count_rounds_up() {
        let e = UniformPartitionEngine::new(&[1.0f32; 7], 4);
        assert_eq!(e.num_partitions(), 2);
    }

    // ── Silence / no-op ───────────────────────────────────────────────────────

    /// Empty IR tail must not touch the output buffer.
    #[test]
    fn empty_ir_tail_leaves_output_unchanged() {
        let mut engine = UniformPartitionEngine::new(&[], 4);
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let mut output = [99.0f32; 4]; // non-zero sentinel
        engine.process(&input, &mut output);
        // process() must be a no-op — output unchanged
        assert_eq!(output, [99.0f32; 4]);
    }

    // ── Single-partition correctness ──────────────────────────────────────────

    /// Unit-impulse tail `[1, 0, 0, 0]` is an identity: output == input.
    #[test]
    fn unit_impulse_tail_is_identity() {
        let block_size = 4;
        let ir = [1.0f32, 0.0, 0.0, 0.0];
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let out = run_blocks(&ir, block_size, &[&input]);
        for (x, y) in input.iter().zip(out.iter()) {
            assert!((x - y).abs() < EPSILON, "expected {x}, got {y}");
        }
    }

    /// IR tail `[0, 1, 0, 0]` is a one-sample delay within the block.
    /// output[0] = 0 (history empty), output[i] = input[i-1].
    #[test]
    fn one_sample_delay_tail() {
        let block_size = 4;
        let ir = [0.0f32, 1.0, 0.0, 0.0];
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let expected = [0.0f32, 1.0, 2.0, 3.0];
        let out = run_blocks(&ir, block_size, &[&input]);
        for (e, y) in expected.iter().zip(out.iter()) {
            assert!((e - y).abs() < EPSILON, "expected {e}, got {y}");
        }
    }

    // ── Multi-partition correctness ───────────────────────────────────────────

    /// IR `[1, 0, 0, 1]` with block_size=2 spans two partitions.
    ///
    /// y[n] = x[n] + x[n-3].  With impulse at x[0]:
    ///   block 1 → [y[0], y[1]] = [1, 0]
    ///   block 2 → [y[2], y[3]] = [0, 1]
    #[test]
    fn multi_partition_spans_two_blocks() {
        let block_size = 2;
        let ir = [1.0f32, 0.0, 0.0, 1.0];
        let b1 = [1.0f32, 0.0];
        let b2 = [0.0f32, 0.0];
        let out = run_blocks(&ir, block_size, &[&b1, &b2]);
        let expected = [1.0f32, 0.0, 0.0, 1.0];
        for (e, y) in expected.iter().zip(out.iter()) {
            assert!((e - y).abs() < EPSILON, "expected {e}, got {y}");
        }
    }

    /// Known two-tap convolution: input=[1,2,3,4], IR=[1,2], block_size=4.
    /// y[n] = 1*x[n] + 2*x[n-1].
    /// Expected: [1*1+2*0, 1*2+2*1, 1*3+2*2, 1*4+2*3] = [1, 4, 7, 10].
    #[test]
    fn known_two_tap_convolution() {
        let block_size = 4;
        let ir = [1.0f32, 2.0, 0.0, 0.0];
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let expected = [1.0f32, 4.0, 7.0, 10.0];
        let out = run_blocks(&ir, block_size, &[&input]);
        for (e, y) in expected.iter().zip(out.iter()) {
            assert!((e - y).abs() < EPSILON, "expected {e}, got {y}");
        }
    }

    // ── State management ──────────────────────────────────────────────────────

    /// State persists across blocks: a tail fed in block 1 appears in block 2.
    #[test]
    fn state_persists_across_blocks() {
        let block_size = 4;
        // IR spans 2 partitions: h[0]=1, h[7]=1 → y[n] = x[n] + x[n-7].
        let ir = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        // Block 1: impulse at x[0].
        let b1 = [1.0f32, 0.0, 0.0, 0.0];
        // Block 2: silence — tail from h[7] of the previous impulse appears here.
        let b2 = [0.0f32; 4];
        let out = run_blocks(&ir, block_size, &[&b1, &b2]);
        // y[0] = 1, y[7] = 1, all others = 0.
        assert!((out[0] - 1.0).abs() < EPSILON, "y[0]={}", out[0]);
        assert!((out[7] - 1.0).abs() < EPSILON, "y[7]={}", out[7]);
        for (i, val) in out.iter().enumerate().take(7).skip(1) {
            assert!(val.abs() < EPSILON, "y[{i}]={val}");
        }
    }

    /// `reset()` clears state — result matches a fresh engine.
    #[test]
    fn reset_gives_same_result_as_fresh_engine() {
        let block_size = 4;
        let ir = [1.0f32, 0.5, 0.0, 0.0];
        let mut engine = UniformPartitionEngine::new(&ir, block_size);

        // Prime with non-zero input.
        let mut scratch = vec![0.0f32; block_size];
        engine.process(&[1.0f32; 4], &mut scratch);
        engine.reset();

        let input = [1.0f32, 2.0, 3.0, 4.0];
        let mut after_reset = vec![0.0f32; block_size];
        engine.process(&input, &mut after_reset);

        let fresh = run_blocks(&ir, block_size, &[&input]);
        for (e, y) in fresh.iter().zip(after_reset.iter()) {
            assert!((e - y).abs() < EPSILON, "expected {e}, got {y}");
        }
    }

    /// `load_ir()` installs a new IR and clears all delay state.
    #[test]
    fn load_ir_installs_new_coefficients() {
        let block_size = 4;
        let mut engine = UniformPartitionEngine::new(&[1.0f32, 0.0, 0.0, 0.0], block_size);

        // Prime with non-zero input.
        let mut scratch = vec![0.0f32; block_size];
        engine.process(&[1.0f32; 4], &mut scratch);

        // Install a gain-3 IR and verify output.
        engine.load_ir(&[3.0f32, 0.0, 0.0, 0.0]);
        let mut out = vec![0.0f32; block_size];
        engine.process(&[1.0f32; 4], &mut out);
        for &y in &out {
            assert!((y - 3.0).abs() < EPSILON, "expected 3.0, got {y}");
        }
    }

    // ── Accumulation mode ─────────────────────────────────────────────────────

    /// `process()` must *add to* output, not overwrite it.
    #[test]
    fn process_accumulates_into_output() {
        let block_size = 4;
        let ir = [1.0f32, 0.0, 0.0, 0.0];
        let mut engine = UniformPartitionEngine::new(&ir, block_size);
        // Pre-fill output with 10.0.
        let mut output = [10.0f32; 4];
        engine.process(&[1.0f32; 4], &mut output);
        // Each output sample should be 10 + 1*1 = 11.
        for &y in &output {
            assert!((y - 11.0).abs() < EPSILON, "expected 11.0, got {y}");
        }
    }
}
