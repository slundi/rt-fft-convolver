/// Time-domain direct convolver for the **head** partition of an Impulse Response.
///
/// # Why direct convolution for the head?
///
/// In a partitioned convolution engine the IR is split into a short *head*
/// (processed here) and one or more longer *tail* partitions (processed via
/// FFT).  The FFT partitions introduce at least one block of algorithmic
/// latency.  By handling the first `head_len` IR coefficients with O(N²)
/// time-domain convolution we get **zero-sample latency** for the critical
/// initial attack transient — the part of a cabinet IR where the ear is most
/// sensitive.
///
/// # Doubled circular buffer
///
/// A naïve circular buffer requires a modulo operation (or branch) inside the
/// inner convolution loop to wrap the read index.  Instead we maintain a
/// buffer of length `2 * head_len` and write every incoming sample to *two*
/// positions: `write_pos` and `write_pos + head_len`.  After each write the
/// slice `history[write_pos .. write_pos + head_len]` is always a contiguous,
/// wrap-free window of the most recent `head_len` samples ordered
/// **newest → oldest**.  The inner dot-product loop then reduces to a plain
/// slice multiply-accumulate, which LLVM auto-vectorises with SIMD.
///
/// ```text
///  head_len = 4,  write_pos = 1 after writing x[5]
///
///  index : 0     1     2     3   | 4     5     6     7
///  value : x[4]  x[5]  x[2]  x[3]  x[4]  x[5]  x[2]  x[3]
///                ↑ write_pos
///  window: history[1..5] = [x[5], x[4], x[3], x[2]]  ← contiguous ✓
///  IR    :                   h[0]   h[1]  h[2]  h[3]
/// ```
pub struct DirectConvolver {
    /// IR head coefficients h[0..head_len].
    /// h[0] is multiplied by the *current* (newest) input sample, achieving
    /// zero latency.  h[1] is multiplied by the sample one step in the past,
    /// and so on.
    ir_head: Vec<f32>,

    /// Doubled circular buffer, length = 2 * head_len.
    /// Invariant: `history[write_pos .. write_pos + head_len]` always holds
    /// the last `head_len` input samples in newest-first order.
    history: Vec<f32>,

    /// Next write index, always in `[0, head_len)`.
    write_pos: usize,
}

impl DirectConvolver {
    /// Create a `DirectConvolver` loaded with the given IR head coefficients.
    ///
    /// `ir_head[0]` is applied to the *current* input sample (zero latency).
    /// Pass `&[]` to create a no-op convolver (output = silence).
    ///
    /// **Allocates** — call from a non-real-time thread.
    pub fn new(ir_head: &[f32]) -> Self {
        let head_len = ir_head.len();
        Self {
            ir_head: ir_head.to_vec(),
            // Double-length buffer, zero-initialised (silent history).
            history: vec![0.0f32; 2 * head_len.max(1)],
            write_pos: 0,
        }
    }

    /// Return the number of IR coefficients.
    #[inline]
    pub fn head_len(&self) -> usize {
        self.ir_head.len()
    }

    /// Replace the IR head coefficients with a new slice.
    ///
    /// If the new length differs from the current one the internal buffers are
    /// reallocated and the sample history is cleared.
    ///
    /// **May allocate** — call from a non-real-time thread.
    pub fn load_ir(&mut self, ir_head: &[f32]) {
        let new_len = ir_head.len();
        if new_len != self.ir_head.len() {
            self.history = vec![0.0f32; 2 * new_len.max(1)];
            self.write_pos = 0;
        } else {
            // Same length: keep history but clear it so there are no artefacts
            // from the previous IR.
            self.history.fill(0.0);
            self.write_pos = 0;
        }
        self.ir_head = ir_head.to_vec();
    }

    /// Clear the sample history without reallocating.
    ///
    /// Call between song sections or preset changes to avoid clicks.
    ///
    /// **Real-time safe** — no allocation.
    #[inline]
    pub fn reset(&mut self) {
        self.history.fill(0.0);
        self.write_pos = 0;
    }

    /// Convolve one block of `input` samples with the IR head, writing results
    /// to `output`.
    ///
    /// `input` and `output` must have the same length.  Any block size is
    /// accepted; the block size need not match `head_len`.
    ///
    /// Output sample `i` depends on `input[i]` through `input[i - head_len +
    /// 1]` (clamped to the history for past samples), so **latency is zero
    /// samples**.
    ///
    /// If the IR head is empty (`head_len == 0`) the output is filled with
    /// silence.
    ///
    /// **Real-time safe** — no allocation, no locking, no panics in release
    /// builds.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(
            input.len(),
            output.len(),
            "process: input and output must have the same length"
        );

        let head_len = self.ir_head.len();

        if head_len == 0 {
            output.fill(0.0);
            return;
        }

        for (x, y) in input.iter().zip(output.iter_mut()) {
            // --- Write new sample to both halves of the doubled buffer ---
            // This keeps both halves mirrored so the convolution window is
            // always contiguous (see struct-level doc).
            self.history[self.write_pos] = *x;
            self.history[self.write_pos + head_len] = *x;

            // --- Dot product: h ⊙ window (newest → oldest) ---
            // SAFETY: write_pos < head_len, so write_pos + head_len < 2*head_len,
            // and the slice is always within bounds.
            let window = &self.history[self.write_pos..self.write_pos + head_len];
            *y = dot(window, &self.ir_head);

            // --- Advance write cursor (wraps at head_len, not 2*head_len) ---
            self.write_pos += 1;
            if self.write_pos == head_len {
                self.write_pos = 0;
            }
        }
    }
}

/// Multiply-accumulate two equal-length slices.
///
/// Written as an explicit loop so LLVM can auto-vectorise it into SIMD
/// instructions (AVX/NEON) without `unsafe` intrinsics.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    // `zip` stops at the shorter slice; lengths are always equal here, but
    // this avoids any bounds-check overhead in the accumulation loop.
    a.iter()
        .zip(b.iter())
        .fold(0.0f32, |acc, (x, h)| acc + x * h)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    // -----------------------------------------------------------------------
    // Helper
    // -----------------------------------------------------------------------

    fn convolve_block(ir: &[f32], input: &[f32]) -> Vec<f32> {
        let mut c = DirectConvolver::new(ir);
        let mut out = vec![0.0f32; input.len()];
        c.process(input, &mut out);
        out
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    /// A unit-impulse IR {1.0} must pass the input through unchanged.
    #[test]
    fn unit_impulse_ir_is_identity() {
        let ir = [1.0f32];
        let input: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let output = convolve_block(&ir, &input);
        for (x, y) in input.iter().zip(output.iter()) {
            assert!((x - y).abs() < EPSILON, "expected {x}, got {y}");
        }
    }

    /// A two-tap IR {0.0, 1.0} is a one-sample delay.
    /// output[0] must be 0 (silence in history), output[i] = input[i-1].
    #[test]
    fn two_tap_delay_ir() {
        let ir = [0.0f32, 1.0];
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let expected = [0.0f32, 1.0, 2.0, 3.0]; // shifted by one
        let output = convolve_block(&ir, &input);
        for (e, y) in expected.iter().zip(output.iter()) {
            assert!((e - y).abs() < EPSILON, "expected {e}, got {y}");
        }
    }

    /// Zero latency: output[0] must use input[0], not a future sample.
    /// With IR = {1.0}, output[0] == input[0] exactly.
    #[test]
    fn output_zero_uses_input_zero() {
        let ir = [1.0f32];
        let input = [42.0f32, 0.0, 0.0];
        let output = convolve_block(&ir, &input);
        assert!((output[0] - 42.0).abs() < EPSILON);
    }

    /// Known convolution: input = [1, 2, 3, 4], IR = [1, 2].
    /// y[n] = h[0]*x[n] + h[1]*x[n-1]
    /// y = [1*1+2*0, 1*2+2*1, 1*3+2*2, 1*4+2*3] = [1, 4, 7, 10]
    #[test]
    fn known_two_tap_convolution() {
        let ir = [1.0f32, 2.0];
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let expected = [1.0f32, 4.0, 7.0, 10.0];
        let output = convolve_block(&ir, &input);
        for (e, y) in expected.iter().zip(output.iter()) {
            assert!((e - y).abs() < EPSILON, "expected {e}, got {y}");
        }
    }

    /// State persists correctly across consecutive block calls.
    /// Feed two blocks of {1,1,1,1} through a two-tap IR {1, 1}.
    /// Block 1 output: [1, 2, 2, 2]  (history starts empty)
    /// Block 2 output: [2, 2, 2, 2]  (history carries x[3]=1 from block 1)
    #[test]
    fn state_persists_across_blocks() {
        let ir = [1.0f32, 1.0];
        let mut conv = DirectConvolver::new(&ir);
        let block = [1.0f32; 4];

        let mut out1 = [0.0f32; 4];
        conv.process(&block, &mut out1);
        assert!((out1[0] - 1.0).abs() < EPSILON); // h[0]*1 + h[1]*0
        assert!((out1[1] - 2.0).abs() < EPSILON); // h[0]*1 + h[1]*1
        assert!((out1[2] - 2.0).abs() < EPSILON);
        assert!((out1[3] - 2.0).abs() < EPSILON);

        let mut out2 = [0.0f32; 4];
        conv.process(&block, &mut out2);
        // History carries 1.0 from the last sample of block 1.
        assert!((out2[0] - 2.0).abs() < EPSILON);
        assert!((out2[1] - 2.0).abs() < EPSILON);
    }

    /// `reset()` must clear the sample history without reallocation.
    /// After reset, the output is the same as if the convolver was freshly created.
    #[test]
    fn reset_clears_history() {
        let ir = [1.0f32, 1.0];
        let mut conv = DirectConvolver::new(&ir);

        // Prime with non-zero samples.
        let mut scratch = [0.0f32; 4];
        conv.process(&[1.0; 4], &mut scratch);

        conv.reset();

        // Output must now match a fresh convolver.
        let input = [3.0f32, 5.0, 7.0];
        let mut after_reset = [0.0f32; 3];
        conv.process(&input, &mut after_reset);

        let expected = convolve_block(&ir, &input);
        for (e, y) in expected.iter().zip(after_reset.iter()) {
            assert!((e - y).abs() < EPSILON, "expected {e}, got {y}");
        }
    }

    /// Empty IR must produce silence without panicking.
    #[test]
    fn empty_ir_produces_silence() {
        let mut conv = DirectConvolver::new(&[]);
        let input = [1.0f32, 2.0, 3.0];
        let mut output = [0.0f32; 3];
        conv.process(&input, &mut output);
        for &y in &output {
            assert_eq!(y, 0.0);
        }
    }

    /// `load_ir()` with a same-length IR clears history and uses new coefficients.
    #[test]
    fn load_ir_swaps_coefficients() {
        let mut conv = DirectConvolver::new(&[1.0f32]);
        // Prime with non-zero history (single-tap so history doesn't matter,
        // but load_ir must still clear write_pos correctly).
        let mut scratch = [0.0f32; 4];
        conv.process(&[9.0; 4], &mut scratch);

        conv.load_ir(&[2.0f32]);
        let mut out = [0.0f32; 2];
        conv.process(&[1.0, 1.0], &mut out);
        // h[0]=2 → output should be 2*1=2 each sample.
        assert!((out[0] - 2.0).abs() < EPSILON);
        assert!((out[1] - 2.0).abs() < EPSILON);
    }
}
