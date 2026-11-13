//! Dense (fully-connected) layer — `output = weights · input + bias`.
//!
//! # Storage layout
//!
//! Weights are stored **row-major**, shape `[output_size, input_size]`.
//! Row `k` occupies indices `[k · input_size, (k + 1) · input_size)`.  This
//! layout matches PyTorch's `nn.Linear` weight tensor, so a NAM / Proteus
//! loader can copy weights straight in without transposing them a second
//! time.
//!
//! # Real-time safety
//!
//! [`Dense::forward`] is allocation-free and delegates all math to
//! [`crate::kernels::Kernels`], which handles SIMD dispatch.

use crate::error::ModelError;
use crate::kernels::Kernels;

/// A dense (fully-connected) layer.
///
/// Owns its weight and bias buffers; construct once off-thread, then feed
/// audio-thread inputs through [`forward`](Self::forward).
pub struct Dense {
    /// Row-major, shape `[output_size, input_size]`.
    weights: Box<[f32]>,
    /// Length `output_size`.
    bias: Box<[f32]>,
    /// SIMD dispatch handle (cached from construction).
    kernels: Kernels,
    input_size: usize,
    output_size: usize,
}

impl Dense {
    /// Construct a dense layer from row-major weight + bias buffers.
    ///
    /// Validates dimensions:
    /// - `weights.len() == output_size * input_size`
    /// - `bias.len() == output_size`
    ///
    /// **Not** real-time safe (heap allocation for weight copies).
    pub fn new(
        weights: Vec<f32>,
        bias: Vec<f32>,
        input_size: usize,
        output_size: usize,
    ) -> Result<Self, ModelError> {
        let expected_weights = output_size.saturating_mul(input_size);
        if weights.len() != expected_weights {
            return Err(ModelError::WeightShapeMismatch {
                expected_rows: output_size,
                expected_cols: input_size,
                got_rows: if input_size == 0 {
                    0
                } else {
                    weights.len() / input_size.max(1)
                },
                got_cols: input_size,
            });
        }
        if bias.len() != output_size {
            return Err(ModelError::BiasLengthMismatch {
                expected: output_size,
                got: bias.len(),
            });
        }
        Ok(Self {
            weights: weights.into_boxed_slice(),
            bias: bias.into_boxed_slice(),
            kernels: Kernels::new(),
            input_size,
            output_size,
        })
    }

    /// Input width (columns of the weight matrix).
    #[inline]
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Output width (rows of the weight matrix).
    #[inline]
    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Number of stored scalar parameters (weights + bias).
    ///
    /// Cheap and RT-safe — just returns the buffer lengths.
    #[inline]
    pub fn parameter_count(&self) -> usize {
        self.weights.len() + self.bias.len()
    }

    /// Compute `output = weights · input + bias`.
    ///
    /// - `input.len()` must equal [`input_size`](Self::input_size).
    /// - `output.len()` must equal [`output_size`](Self::output_size).
    ///
    /// **Real-time safe** — no allocation, no locking.
    #[inline]
    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.input_size);
        debug_assert_eq!(output.len(), self.output_size);
        self.kernels
            .matvec_add_bias(&self.weights, input, &self.bias, output);
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity matrix + zero bias should pass input through unchanged.
    #[test]
    fn identity_weights_pass_input_through() {
        let size = 4;
        let mut weights = vec![0.0f32; size * size];
        for row_index in 0..size {
            weights[row_index * size + row_index] = 1.0;
        }
        let bias = vec![0.0f32; size];
        let dense = Dense::new(weights, bias, size, size).unwrap();

        let input = [0.5f32, -1.0, 2.0, 3.75];
        let mut output = [0.0f32; 4];
        dense.forward(&input, &mut output);

        for (produced, expected) in output.iter().zip(input.iter()) {
            assert!((produced - expected).abs() < 1e-6);
        }
    }

    /// Zero weights + non-zero bias should output the bias verbatim.
    #[test]
    fn zero_weights_output_bias_only() {
        let weights = vec![0.0f32; 3 * 4];
        let bias = vec![0.1_f32, 0.2, 0.3];
        let dense = Dense::new(weights, bias.clone(), 4, 3).unwrap();

        let input = [1.0f32, 2.0, 3.0, 4.0];
        let mut output = [0.0f32; 3];
        dense.forward(&input, &mut output);

        for (produced, expected) in output.iter().zip(bias.iter()) {
            assert!((produced - expected).abs() < 1e-6);
        }
    }

    /// Manually computed small case — checks the matmul convention.
    #[test]
    fn small_case_matches_manual_computation() {
        // Weights (2×3, row-major): [[1, 2, 3], [4, 5, 6]]
        // Bias: [10, 20]
        // Input: [1, 1, 1]
        // Expected: [1+2+3+10, 4+5+6+20] = [16, 35]
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bias = vec![10.0, 20.0];
        let dense = Dense::new(weights, bias, 3, 2).unwrap();
        let input = [1.0f32, 1.0, 1.0];
        let mut output = [0.0f32; 2];
        dense.forward(&input, &mut output);
        assert_eq!(output, [16.0, 35.0]);
    }

    /// Weight-shape mismatch produces the expected error.
    #[test]
    fn rejects_wrong_weight_length() {
        let result = Dense::new(vec![0.0f32; 5], vec![0.0f32; 2], 3, 2);
        assert!(matches!(
            result,
            Err(ModelError::WeightShapeMismatch { .. })
        ));
    }

    /// Bias-length mismatch produces the expected error.
    #[test]
    fn rejects_wrong_bias_length() {
        let result = Dense::new(vec![0.0f32; 6], vec![0.0f32; 3], 3, 2);
        assert!(matches!(result, Err(ModelError::BiasLengthMismatch { .. })));
    }
}
