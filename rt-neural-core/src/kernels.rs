//! SIMD kernels backing the dense and conv1d layers.
//!
//! # Design
//!
//! Every hot-path arithmetic kernel is written once, twice:
//!
//! 1. A **scalar reference** that LLVM auto-vectorises well and serves as
//!    the source of truth for tests.
//! 2. A **[`pulp`]-dispatched** version that carries the same math through
//!    an explicit SIMD register width, letting `pulp` pick between
//!    SSE / AVX2 / AVX-512 / NEON at runtime.
//!
//! Both variants share signatures so callers can swap between them without
//! touching layer code.  The [`Kernels`] struct caches a [`pulp::Arch`] so
//! the ISA-selection cost is paid exactly once, at model load time —
//! never on the audio thread.

use pulp::{Arch, Simd, WithSimd};

/// A cached SIMD-dispatch handle.
///
/// Construct once at model-load time; call methods from the audio thread.
/// The handle is `Copy` and holds no state beyond a pre-selected ISA tag.
#[derive(Copy, Clone)]
pub struct Kernels {
    arch: Arch,
}

impl Kernels {
    /// Detect the best SIMD ISA available on the current CPU.
    ///
    /// **Not** real-time safe — call once, off the audio thread.
    #[inline]
    pub fn new() -> Self {
        Self { arch: Arch::new() }
    }

    /// Dot product `Σᵢ left[i] · right[i]`.
    ///
    /// Panics in debug builds if the two slices have different lengths.
    ///
    /// **Real-time safe** — no allocation, no locking.
    #[inline]
    pub fn dot(&self, left: &[f32], right: &[f32]) -> f32 {
        debug_assert_eq!(left.len(), right.len(), "dot: slice lengths must match");
        self.arch.dispatch(DotKernel { left, right })
    }

    /// Compute `output[row] = bias[row] + Σⱼ weights[row, col] · input[col]`
    /// for every `row` in `0..output.len()`.
    ///
    /// - `weights` is stored **row-major**, shape `[rows, cols]` — i.e. row
    ///   `k` starts at offset `k · cols`.
    /// - `output.len()` must equal `rows`; `input.len()` must equal `cols`;
    ///   `bias.len()` must equal `rows`.
    ///
    /// The bias is added inside the SIMD reduction so the final store is
    /// a single write per row.
    ///
    /// **Real-time safe** — no allocation, no locking.
    #[inline]
    pub fn matvec_add_bias(
        &self,
        weights: &[f32],
        input: &[f32],
        bias: &[f32],
        output: &mut [f32],
    ) {
        let cols = input.len();
        let rows = output.len();
        debug_assert_eq!(weights.len(), rows * cols, "weights shape mismatch");
        debug_assert_eq!(bias.len(), rows, "bias length mismatch");
        self.arch.dispatch(MatVecKernel {
            weights,
            input,
            bias,
            output,
            cols,
        });
    }
}

impl Default for Kernels {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Pulp kernel implementations ─────────────────────────────────────────────

/// Portable-SIMD dot product implemented via `pulp::WithSimd`.
///
/// The kernel packs both slices into full-width SIMD lanes, accumulates
/// with fused multiply-add, then reduces to a scalar and handles the
/// unaligned tail.
struct DotKernel<'left, 'right> {
    left: &'left [f32],
    right: &'right [f32],
}

impl<'left, 'right> WithSimd for DotKernel<'left, 'right> {
    type Output = f32;

    #[inline(always)]
    fn with_simd<SimdArch: Simd>(self, simd: SimdArch) -> f32 {
        let (left_lanes, left_tail) = SimdArch::as_simd_f32s(self.left);
        let (right_lanes, right_tail) = SimdArch::as_simd_f32s(self.right);

        let mut acc = simd.splat_f32s(0.0);
        for (left_lane, right_lane) in left_lanes.iter().zip(right_lanes.iter()) {
            acc = simd.mul_add_f32s(*left_lane, *right_lane, acc);
        }
        let mut scalar_acc = simd.reduce_sum_f32s(acc);

        for (left_scalar, right_scalar) in left_tail.iter().zip(right_tail.iter()) {
            scalar_acc += left_scalar * right_scalar;
        }
        scalar_acc
    }
}

/// Portable-SIMD row-major matrix × vector + bias.
struct MatVecKernel<'weights, 'input, 'bias, 'output> {
    weights: &'weights [f32],
    input: &'input [f32],
    bias: &'bias [f32],
    output: &'output mut [f32],
    cols: usize,
}

impl<'weights, 'input, 'bias, 'output> WithSimd for MatVecKernel<'weights, 'input, 'bias, 'output> {
    type Output = ();

    #[inline(always)]
    fn with_simd<SimdArch: Simd>(self, simd: SimdArch) {
        let cols = self.cols;
        let (input_lanes, input_tail) = SimdArch::as_simd_f32s(self.input);

        for (row_index, output_cell) in self.output.iter_mut().enumerate() {
            let row_start = row_index * cols;
            let row_end = row_start + cols;
            let row = &self.weights[row_start..row_end];

            let (row_lanes, row_tail) = SimdArch::as_simd_f32s(row);

            let mut acc = simd.splat_f32s(0.0);
            for (row_lane, input_lane) in row_lanes.iter().zip(input_lanes.iter()) {
                acc = simd.mul_add_f32s(*row_lane, *input_lane, acc);
            }
            let mut scalar_acc = simd.reduce_sum_f32s(acc);
            for (row_scalar, input_scalar) in row_tail.iter().zip(input_tail.iter()) {
                scalar_acc += row_scalar * input_scalar;
            }
            *output_cell = self.bias[row_index] + scalar_acc;
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_dot(left: &[f32], right: &[f32]) -> f32 {
        left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
    }

    fn scalar_matvec(
        weights: &[f32],
        input: &[f32],
        bias: &[f32],
        output: &mut [f32],
        cols: usize,
    ) {
        for (row_index, cell) in output.iter_mut().enumerate() {
            let mut sum = bias[row_index];
            for col_index in 0..cols {
                sum += weights[row_index * cols + col_index] * input[col_index];
            }
            *cell = sum;
        }
    }

    /// Dot product must match the scalar reference across a range of sizes,
    /// including sizes that are not multiples of the SIMD lane count.
    #[test]
    fn dot_matches_scalar_reference() {
        let kernels = Kernels::new();
        for length in [1usize, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 128, 129] {
            let left: Vec<f32> = (0..length).map(|index| index as f32 * 0.5).collect();
            let right: Vec<f32> = (0..length).map(|index| 1.0 - index as f32 * 0.25).collect();

            let dispatched = kernels.dot(&left, &right);
            let reference = scalar_dot(&left, &right);
            assert!(
                (dispatched - reference).abs() < 1e-4 * reference.abs().max(1.0),
                "dot length={length}: dispatched={dispatched}, reference={reference}"
            );
        }
    }

    /// Matrix-vector + bias must match the scalar reference.
    #[test]
    fn matvec_matches_scalar_reference() {
        let kernels = Kernels::new();
        for &(rows, cols) in &[(1usize, 1usize), (2, 2), (3, 4), (7, 9), (16, 17), (32, 33)] {
            let weights: Vec<f32> = (0..rows * cols)
                .map(|index| ((index * 7) % 13) as f32 - 6.0)
                .collect();
            let input: Vec<f32> = (0..cols).map(|index| index as f32 * 0.125).collect();
            let bias: Vec<f32> = (0..rows).map(|index| index as f32 * -0.5).collect();

            let mut dispatched = vec![0.0f32; rows];
            let mut reference = vec![0.0f32; rows];
            kernels.matvec_add_bias(&weights, &input, &bias, &mut dispatched);
            scalar_matvec(&weights, &input, &bias, &mut reference, cols);

            for row_index in 0..rows {
                let magnitude = reference[row_index].abs().max(1.0);
                assert!(
                    (dispatched[row_index] - reference[row_index]).abs() < 1e-4 * magnitude,
                    "matvec[{rows}×{cols}] row {row_index}: \
                     dispatched={}, reference={}",
                    dispatched[row_index],
                    reference[row_index]
                );
            }
        }
    }

    /// Dot product of a zero vector with anything is zero.
    #[test]
    fn dot_of_zero_vector_is_zero() {
        let kernels = Kernels::new();
        let zeros = vec![0.0f32; 33];
        let non_zero: Vec<f32> = (0..33).map(|index| index as f32).collect();
        assert_eq!(kernels.dot(&zeros, &non_zero), 0.0);
    }

    /// The empty dot product is zero.
    #[test]
    fn dot_of_empty_vectors_is_zero() {
        let kernels = Kernels::new();
        let empty: Vec<f32> = Vec::new();
        assert_eq!(kernels.dot(&empty, &empty), 0.0);
    }
}
