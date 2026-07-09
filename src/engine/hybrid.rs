//! Hybrid convolver: zero-latency direct head + partitioned FFT tail.
//!
//! # Why hybrid?
//!
//! Two primitives already exist in the crate:
//!
//! - [`DirectConvolver`] — O(N²) time-domain convolution with **zero
//!   algorithmic latency**.  Cheap for short IRs, prohibitively expensive
//!   for long ones.
//! - [`UniformPartitionEngine`] — O(N log N) FFT-based convolution that
//!   scales to arbitrary IR lengths but pays the FFT setup cost even for
//!   the first few taps.
//!
//! For a full cabinet IR the ideal split is:
//!
//! - Head (first `head_len` taps) → direct: zero latency for the initial
//!   attack transient, where the ear is most sensitive.
//! - Tail (everything after) → partitioned: efficient for the long
//!   reverb portion.
//!
//! Naïvely running both engines on the *same* IR would double-count every
//! tap.  [`HybridConvolver`] performs the split correctly:
//!
//! ```text
//! direct  ← ir[..head_len]
//! tail    ← ir[head_len..]
//! ```
//!
//! # The alignment problem
//!
//! [`UniformPartitionEngine`] documents itself as "perfectly causal, zero
//! algorithmic latency" *for its own IR*.  But its IR here is
//! `ir[head_len..]`, not the full IR — so at block `T` it computes the
//! convolution with `ir_tail` starting from **IR index 0**:
//!
//! ```text
//! P[T][i] = Σ_j ir[head_len + j] · input[T·N + i − j]
//!         = Σ_{k ≥ head_len} ir[k] · input[(T+1)·N + i − k]
//! ```
//!
//! Read the second line carefully: `P[T]` is the tail contribution to
//! **block T+1**, not block T.  The partition engine's output must
//! therefore be delayed by exactly `head_len` samples before being summed
//! with the head's output.
//!
//! Picking `head_len = block_size` makes that delay block-aligned: a
//! single `block_size`-sized carry buffer holds one block of tail output
//! and gets added on the following call.
//!
//! ```text
//! block T:
//!   1. output ← direct.process(input_T)         (head, current block)
//!   2. output += carry                          (tail contribution scheduled
//!                                                for THIS block, computed on
//!                                                the previous call)
//!   3. carry  ← tail.process(input_T)           (tail contribution for the
//!                                                NEXT block, saved to carry)
//! ```
//!
//! # Latency
//!
//! Zero.  The direct head contributes to the current block immediately,
//! and the tail is delayed to arrive on the following block — which is
//! exactly when the equivalent full-IR convolution would place those
//! taps.
//!
//! [`DirectConvolver`]: crate::dsp::direct::DirectConvolver
//! [`UniformPartitionEngine`]: crate::engine::partition::UniformPartitionEngine

use crate::dsp::direct::DirectConvolver;
use crate::engine::partition::UniformPartitionEngine;

/// Composite convolver pairing a zero-latency direct head with a
/// partitioned FFT tail so a single IR spans both engines without
/// double-counting taps.
///
/// - Head = `ir[..min(ir.len(), block_size)]` → [`DirectConvolver`]
/// - Tail = `ir[min(ir.len(), block_size)..]` → [`UniformPartitionEngine`]
///
/// The tail engine's output is delayed by one block via a carry buffer so
/// its taps land where the equivalent full-IR convolution would place
/// them.
///
/// [`DirectConvolver`]: crate::dsp::direct::DirectConvolver
/// [`UniformPartitionEngine`]: crate::engine::partition::UniformPartitionEngine
pub struct HybridConvolver {
    /// Zero-latency direct convolver for the head partition.
    direct: DirectConvolver,

    /// FFT-based tail engine for `ir[head_len..]`.  Its output is
    /// delayed one block through [`Self::carry`] before being summed
    /// with the head's output.
    tail: UniformPartitionEngine,

    /// Audio block size — must equal `head_len` for the delay to be
    /// block-aligned.
    block_size: usize,

    /// Previous call's tail output, added to the current call's output
    /// before the tail engine is re-run.  Length = `block_size`.
    carry: Vec<f32>,
}

impl HybridConvolver {
    /// Build a hybrid convolver loaded with the given IR.
    ///
    /// - `ir` — full impulse response; may be empty (creates a silence
    ///   convolver).
    /// - `block_size` — audio render block size (must be > 0).  Also
    ///   fixes the head length to `min(ir.len(), block_size)`.
    ///
    /// **Allocates** — call from a non-real-time thread.
    ///
    /// # Panics
    ///
    /// Panics if `block_size == 0`.
    pub fn new(ir: &[f32], block_size: usize) -> Self {
        assert!(block_size > 0, "block_size must be > 0");

        // Split the IR exactly at head_len.  When ir.len() < block_size,
        // tail_ir is empty and the partition engine becomes a no-op — the
        // whole convolution runs through the direct head.
        let head_len = ir.len().min(block_size);
        let (head_ir, tail_ir) = ir.split_at(head_len);

        Self {
            direct: DirectConvolver::new(head_ir),
            tail: UniformPartitionEngine::new(tail_ir, block_size),
            block_size,
            carry: vec![0.0f32; block_size],
        }
    }

    /// Audio block size this convolver was built with.
    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Number of taps handled by the direct head (`0 ..= block_size`).
    #[inline]
    pub fn head_len(&self) -> usize {
        self.direct.head_len()
    }

    /// Number of tail partitions handled by the FFT engine.
    #[inline]
    pub fn num_partitions(&self) -> usize {
        self.tail.num_partitions()
    }

    /// Clear all internal state without reallocating.
    ///
    /// **Real-time safe** — no allocation.
    pub fn reset(&mut self) {
        self.direct.reset();
        self.tail.reset();
        self.carry.fill(0.0);
    }

    /// Replace the IR and reset all internal state.
    ///
    /// **Allocates** — call from a non-real-time thread.
    pub fn load_ir(&mut self, ir: &[f32]) {
        let head_len = ir.len().min(self.block_size);
        let (head_ir, tail_ir) = ir.split_at(head_len);
        self.direct.load_ir(head_ir);
        self.tail.load_ir(tail_ir);
        self.carry.fill(0.0);
    }

    /// Convolve one block of `input` with the full IR, **overwriting**
    /// `output`.
    ///
    /// Both slices must have length equal to [`Self::block_size`].
    ///
    /// **Real-time safe** — no allocation.
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

        // 1. Head contribution — DirectConvolver overwrites `output`
        //    with the head convolution of the current input block.
        self.direct.process(input, output);

        // 2. Add the tail contribution scheduled for THIS block — it was
        //    computed on the previous call and stashed in `carry`.
        for (out, &c) in output.iter_mut().zip(self.carry.iter()) {
            *out += c;
        }

        // 3. Compute the tail contribution scheduled for the NEXT call
        //    and stash it back into `carry`.  We can reuse the buffer
        //    because step 2 has already consumed its previous contents.
        //    `UniformPartitionEngine::process` accumulates into its
        //    output — zero the carry buffer first so it holds only the
        //    current block's contribution.
        self.carry.fill(0.0);
        self.tail.process(input, &mut self.carry);
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-4;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Naïve O(N·L) time-domain convolution in f32 as the ground truth.
    fn naive(ir: &[f32], input: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; input.len()];
        for (idx, sample) in out.iter_mut().enumerate() {
            for (tap_idx, &tap) in ir.iter().enumerate() {
                if idx >= tap_idx {
                    *sample += tap * input[idx - tap_idx];
                }
            }
        }
        out
    }

    /// Feed `blocks` through a fresh hybrid convolver and collect the output.
    fn run_blocks(ir: &[f32], block_size: usize, blocks: &[&[f32]]) -> Vec<f32> {
        let mut conv = HybridConvolver::new(ir, block_size);
        let mut out = Vec::new();
        for &block in blocks {
            let mut buf = vec![0.0f32; block_size];
            conv.process(block, &mut buf);
            out.extend_from_slice(&buf);
        }
        out
    }

    fn assert_close(label: &str, got: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
        for (idx, (&value, &target)) in got.iter().zip(expected.iter()).enumerate() {
            let err = (value - target).abs();
            assert!(
                err < tol,
                "{label}[{idx}]: expected {target}, got {value} (abs err {err})"
            );
        }
    }

    // ── Structural ───────────────────────────────────────────────────────────

    /// Empty IR gives 0 head taps and 0 tail partitions.
    #[test]
    fn empty_ir_has_no_state() {
        let conv = HybridConvolver::new(&[], 4);
        assert_eq!(conv.head_len(), 0);
        assert_eq!(conv.num_partitions(), 0);
        assert_eq!(conv.block_size(), 4);
    }

    /// IR shorter than block_size lives entirely in the direct head.
    #[test]
    fn short_ir_uses_only_direct_head() {
        let conv = HybridConvolver::new(&[1.0f32, 0.5, 0.25], 8);
        assert_eq!(conv.head_len(), 3);
        assert_eq!(conv.num_partitions(), 0);
    }

    /// IR exactly `block_size` long fits in the head with no tail.
    #[test]
    fn ir_of_block_size_has_no_tail() {
        let conv = HybridConvolver::new(&[1.0f32, 0.5, 0.25, 0.125], 4);
        assert_eq!(conv.head_len(), 4);
        assert_eq!(conv.num_partitions(), 0);
    }

    /// Long IR splits into head (block_size taps) + one or more partitions.
    #[test]
    fn long_ir_splits_head_and_tail() {
        // 12-tap IR at block_size=4: head = 4 taps, tail = 8 taps = 2 partitions.
        let conv = HybridConvolver::new(&[0.5f32; 12], 4);
        assert_eq!(conv.head_len(), 4);
        assert_eq!(conv.num_partitions(), 2);
    }

    // ── Silence / no-op ──────────────────────────────────────────────────────

    /// Empty IR must produce silence.
    #[test]
    fn empty_ir_produces_silence() {
        let mut conv = HybridConvolver::new(&[], 4);
        let mut out = [99.0f32; 4]; // sentinel
        conv.process(&[1.0f32, 2.0, 3.0, 4.0], &mut out);
        assert_eq!(out, [0.0f32; 4]);
    }

    // ── Correctness against a naïve reference ────────────────────────────────

    /// Unit-impulse IR is identity within the head.
    #[test]
    fn unit_impulse_ir_is_identity() {
        let block_size = 4;
        let ir = [1.0f32];
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let out = run_blocks(&ir, block_size, &[&input]);
        assert_close("identity", &out, &input, EPSILON);
    }

    /// Head-only IR (length < block_size) matches the naïve reference.
    #[test]
    fn head_only_matches_naive() {
        let block_size = 8;
        let ir = [1.0f32, 2.0, 3.0];
        let input: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1).collect();
        let blocks: Vec<&[f32]> = input.chunks(block_size).collect();
        let out = run_blocks(&ir, block_size, &blocks);
        let reference = naive(&ir, &input);
        assert_close("head-only", &out, &reference, EPSILON);
    }

    /// Head + one tail partition matches the naïve reference.
    #[test]
    fn head_plus_one_tail_partition_matches_naive() {
        let block_size = 4;
        // 6-tap IR = head=[1,2,3,4] + tail=[5,6]
        let ir = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input: Vec<f32> = (0..12).map(|i| (i + 1) as f32).collect();
        let blocks: Vec<&[f32]> = input.chunks(block_size).collect();
        let out = run_blocks(&ir, block_size, &blocks);
        let reference = naive(&ir, &input);
        assert_close("head+1 tail", &out, &reference, EPSILON);
    }

    /// Head + several tail partitions matches the naïve reference.
    #[test]
    fn head_plus_multi_partition_matches_naive() {
        let block_size = 4;
        // 20-tap IR: 1 head (4) + 4 tail partitions (16).
        let ir: Vec<f32> = (0..20).map(|i| ((i as f32) * 0.3).sin() * 0.5).collect();
        let input: Vec<f32> = (0..40).map(|i| ((i as f32) * 0.17).sin()).collect();
        let blocks: Vec<&[f32]> = input.chunks(block_size).collect();
        let out = run_blocks(&ir, block_size, &blocks);
        let reference = naive(&ir, &input);
        // FFT contribution introduces ~1e-6 error per partition; scale tol.
        assert_close("head+multi tail", &out, &reference, 1e-4);
    }

    /// IR whose length is not a multiple of block_size — verifies the final
    /// (padded) tail partition contributes zero for its missing taps.
    #[test]
    fn non_multiple_ir_length_matches_naive() {
        let block_size = 4;
        // 11-tap IR: head (4) + 2 tail partitions [5..8], [9..11 + 1 zero-pad].
        let ir: Vec<f32> = (1..=11).map(|i| i as f32 * 0.1).collect();
        let input: Vec<f32> = (0..20).map(|i| (i as f32) * 0.05 + 0.1).collect();
        let blocks: Vec<&[f32]> = input.chunks(block_size).collect();
        let out = run_blocks(&ir, block_size, &blocks);
        let reference = naive(&ir, &input);
        assert_close("non-multiple", &out, &reference, EPSILON);
    }

    // ── Latency and alignment ────────────────────────────────────────────────

    /// Zero-latency invariant: output[0] uses input[0] × ir[0], never a
    /// delayed or future sample.  Uses a non-trivial IR spanning head+tail.
    #[test]
    fn zero_latency_at_sample_zero() {
        let block_size = 4;
        let ir = [1.0f32, 0.5, 0.25, 0.125, 0.0625, 0.03125];
        let mut conv = HybridConvolver::new(&ir, block_size);
        // Impulse at t=0 gives y[0] = ir[0] × 1 = 1.0 exactly.
        let input = [1.0f32, 0.0, 0.0, 0.0];
        let mut out = [0.0f32; 4];
        conv.process(&input, &mut out);
        assert!(
            (out[0] - 1.0).abs() < EPSILON,
            "zero-latency: out[0]={} (expected 1.0)",
            out[0]
        );
    }

    /// Tail taps fire on the CORRECT (later) block — proves the one-block
    /// delay carry is applied.  IR = [0,0,0,0,1] places its only tap in
    /// the first tail slot, so block 0 must be silent and block 1 must
    /// contain the impulse response of that tap.
    #[test]
    fn tail_tap_fires_on_next_block() {
        let block_size = 4;
        // Tap at ir[4] = first sample of the tail partition.
        let ir = [0.0f32, 0.0, 0.0, 0.0, 1.0];
        let mut conv = HybridConvolver::new(&ir, block_size);

        let mut block0 = [0.0f32; 4];
        conv.process(&[1.0f32, 0.0, 0.0, 0.0], &mut block0);
        // Head is all-zero and the carry starts empty → block 0 silent.
        for (idx, &value) in block0.iter().enumerate() {
            assert!(
                value.abs() < EPSILON,
                "block 0 out[{idx}]={value} (expected 0)"
            );
        }

        let mut block1 = [0.0f32; 4];
        conv.process(&[0.0f32; 4], &mut block1);
        // The impulse convolved with ir[4]=1 lands on block 1, sample 0.
        assert!(
            (block1[0] - 1.0).abs() < EPSILON,
            "block 1 out[0]={} (expected 1.0)",
            block1[0]
        );
        for (idx, &value) in block1.iter().enumerate().skip(1) {
            assert!(
                value.abs() < EPSILON,
                "block 1 out[{idx}]={value} (expected 0)"
            );
        }
    }

    /// Feeding a stream of samples through the hybrid must match a
    /// hand-computed impulse response over many blocks.  Guards against
    /// carry-buffer indexing bugs.
    #[test]
    fn impulse_response_across_many_blocks() {
        let block_size = 4;
        // 10-tap IR spanning 1 head + 2 tail partitions.
        let ir: Vec<f32> = (1..=10).map(|i| i as f32).collect();
        let mut conv = HybridConvolver::new(&ir, block_size);
        // Feed impulse then silence to unroll the IR from the output.
        let mut got = Vec::new();
        let blocks: [[f32; 4]; 4] = [[1.0, 0.0, 0.0, 0.0], [0.0; 4], [0.0; 4], [0.0; 4]];
        for block in &blocks {
            let mut buf = [0.0f32; 4];
            conv.process(block, &mut buf);
            got.extend_from_slice(&buf);
        }
        // Expected: the IR itself, then zeros past the IR length.
        let mut expected = vec![0.0f32; 16];
        for (i, &h) in ir.iter().enumerate() {
            expected[i] = h;
        }
        assert_close("impulse response", &got, &expected, EPSILON);
    }

    // ── State management ─────────────────────────────────────────────────────

    /// reset() clears all state — a fresh convolver with the same IR
    /// produces identical output.
    #[test]
    fn reset_gives_same_result_as_fresh() {
        let block_size = 4;
        let ir = [1.0f32, 0.5, 0.25, 0.125, 0.0625, 0.03125];

        let mut conv = HybridConvolver::new(&ir, block_size);
        let mut scratch = [0.0f32; 4];
        conv.process(&[1.0f32; 4], &mut scratch);
        conv.reset();

        let input = [1.0f32, 2.0, 3.0, 4.0];
        let mut after_reset = [0.0f32; 4];
        conv.process(&input, &mut after_reset);

        let mut fresh = HybridConvolver::new(&ir, block_size);
        let mut expected = [0.0f32; 4];
        fresh.process(&input, &mut expected);

        assert_close("reset", &after_reset, &expected, EPSILON);
    }

    /// load_ir() installs new coefficients and clears carry+FDL state.
    #[test]
    fn load_ir_installs_new_coefficients() {
        let block_size = 4;
        let mut conv = HybridConvolver::new(&[1.0f32], block_size);
        let mut scratch = [0.0f32; 4];
        conv.process(&[1.0f32; 4], &mut scratch);

        // Install a longer IR spanning head+tail: gain 2 head + one tail tap.
        let new_ir = [2.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0];
        conv.load_ir(&new_ir);
        assert_eq!(conv.head_len(), 4);
        assert_eq!(conv.num_partitions(), 1);

        // Feed impulse; the head's first tap (=2) must drive block 0
        // sample 0 exactly.  Carry must be zero (state was cleared).
        let mut out = [0.0f32; 4];
        conv.process(&[1.0f32, 0.0, 0.0, 0.0], &mut out);
        assert!(
            (out[0] - 2.0).abs() < EPSILON,
            "post-load head fires wrong: out[0]={}",
            out[0]
        );
    }

    // ── Overwrite semantics ──────────────────────────────────────────────────

    /// process() overwrites `output` — pre-existing values are discarded.
    #[test]
    fn process_overwrites_output() {
        let block_size = 4;
        let mut conv = HybridConvolver::new(&[1.0f32, 0.5], block_size);
        let mut out = [99.0f32; 4]; // sentinel
        conv.process(&[1.0f32; 4], &mut out);
        // Every sample must be overwritten by the convolution result.
        // With IR = [1.0, 0.5] the steady-state output is 1.5 ≠ 99.
        for &value in &out {
            assert!(value < 99.0, "sentinel not overwritten (sample = {value})");
        }
    }
}
