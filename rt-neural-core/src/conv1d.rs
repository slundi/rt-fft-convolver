//! Causal, dilated Conv1D — the building block of WaveNet.
//!
//! # What "causal + dilated" means for audio
//!
//! - **Causal**: output at time `t` depends only on inputs at times
//!   `t, t-1, t-2, ...`.  No lookahead, so latency is zero samples.
//! - **Dilated**: the kernel skips `dilation - 1` samples between taps.
//!   A dilation-`d` kernel of size `k` covers a receptive field of
//!   `(k - 1) · d + 1` samples with the same number of multiplies as a
//!   dense kernel of size `k`.
//!
//! ```text
//! dilation=1, k=3:  [x_{t-2}, x_{t-1}, x_t]
//! dilation=2, k=3:  [x_{t-4},        x_{t-2},        x_t]
//! dilation=4, k=3:  [x_{t-8},                        x_{t-4},                        x_t]
//! ```
//!
//! # Ring-buffer history
//!
//! The layer keeps the last `(kernel_size - 1) · dilation + 1` input
//! frames in a **power-of-two ring buffer** (rounded up so the index math
//! stays cheap).  Reads are constant-time and index into the ring with a
//! bitmask.
//!
//! # Storage layout
//!
//! Weights are stored as `[out_channels, in_channels, kernel_size]`.  For a
//! given output channel `output_channel` and kernel tap `kernel_tap`, the
//! `in_channels` row starts at
//! `(output_channel · kernel_size + kernel_tap) · in_channels`.
//! Putting `in_channels` in the innermost stride makes the SIMD dot
//! product a plain contiguous load.
//!
//! # Real-time safety
//!
//! [`Conv1d::forward`] is allocation-free and lock-free.  History
//! shifts amount to a single write into the ring buffer at the current
//! index.

use crate::error::ModelError;
use crate::kernels::Kernels;

/// Causal, dilated 1-D convolution.
///
/// Streams sample-by-sample: call [`forward`](Self::forward) once per audio
/// sample with the current input frame (a slice of length `in_channels`).
pub struct Conv1d {
    /// Weights, layout `[out_channels, kernel_size, in_channels]`.
    weights: Box<[f32]>,
    /// Bias, length `out_channels`.
    bias: Box<[f32]>,
    /// Ring buffer of past input frames: length `ring_len * in_channels`.
    ///
    /// `ring_len` is the smallest power of two ≥ the receptive field size.
    /// A single frame at ring index `r` starts at `r * in_channels`.
    ring: Box<[f32]>,
    /// Bit mask (`ring_len - 1`) used to wrap ring indices without `%`.
    ring_mask: usize,
    /// Where the *next* frame will be written.
    write_index: usize,

    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    dilation: usize,

    kernels: Kernels,
}

impl Conv1d {
    /// Construct a causal-dilated 1-D convolution layer.
    ///
    /// - `weights` — row-major with layout `[out_channels, kernel_size,
    ///   in_channels]`.  Length must equal
    ///   `out_channels · kernel_size · in_channels`.
    /// - `bias` — length `out_channels`.
    /// - `kernel_size` ≥ 1, `dilation` ≥ 1.
    ///
    /// **Not** real-time safe (allocates the ring buffer).
    pub fn new(
        weights: Vec<f32>,
        bias: Vec<f32>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
    ) -> Result<Self, ModelError> {
        if kernel_size == 0 || dilation == 0 {
            return Err(ModelError::InvalidConv1dConfig {
                kernel_size,
                dilation,
            });
        }
        let expected_weights = out_channels * kernel_size * in_channels;
        if weights.len() != expected_weights {
            return Err(ModelError::WeightShapeMismatch {
                expected_rows: out_channels * kernel_size,
                expected_cols: in_channels,
                got_rows: if in_channels == 0 {
                    0
                } else {
                    weights.len() / in_channels.max(1)
                },
                got_cols: in_channels,
            });
        }
        if bias.len() != out_channels {
            return Err(ModelError::BiasLengthMismatch {
                expected: out_channels,
                got: bias.len(),
            });
        }

        // Receptive field: (kernel_size - 1) · dilation + 1 taps, but we
        // never index beyond `(kernel_size - 1) · dilation` samples in the
        // past — that's the deepest historical tap.  Round *up* to a power
        // of two so ring wrapping is a bitmask.
        let receptive_field = (kernel_size - 1) * dilation + 1;
        let ring_len = receptive_field.next_power_of_two().max(1);
        let ring = vec![0.0f32; ring_len * in_channels].into_boxed_slice();

        Ok(Self {
            weights: weights.into_boxed_slice(),
            bias: bias.into_boxed_slice(),
            ring,
            ring_mask: ring_len - 1,
            write_index: 0,
            in_channels,
            out_channels,
            kernel_size,
            dilation,
            kernels: Kernels::new(),
        })
    }

    /// Number of input channels expected per frame.
    #[inline]
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Number of output channels produced per frame.
    #[inline]
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// Number of stored scalar parameters (weights + bias).
    ///
    /// The ring-buffer history is inference state, not a parameter, and
    /// is deliberately excluded.
    #[inline]
    pub fn parameter_count(&self) -> usize {
        self.weights.len() + self.bias.len()
    }

    /// Zero the history buffer.  Used when the surrounding model resets
    /// (e.g., a bypass toggle or a preset change).
    ///
    /// **Real-time safe** — a `slice::fill(0.0)` is one contiguous store.
    #[inline]
    pub fn reset(&mut self) {
        for cell in self.ring.iter_mut() {
            *cell = 0.0;
        }
        self.write_index = 0;
    }

    /// Consume one input frame and produce one output frame.
    ///
    /// - `input.len()` must equal `in_channels`.
    /// - `output.len()` must equal `out_channels`.
    ///
    /// **Real-time safe** — the ring buffer avoids any allocation, and
    /// the SIMD dot product is dispatched through [`Kernels`].
    #[inline]
    pub fn forward(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.in_channels);
        debug_assert_eq!(output.len(), self.out_channels);

        // ── Step 1: write the incoming frame into the ring.
        let frame_offset = self.write_index * self.in_channels;
        self.ring[frame_offset..frame_offset + self.in_channels].copy_from_slice(input);

        // ── Step 2: for each output channel, dot-product all taps.
        //
        // The deepest tap is `kernel_size - 1` steps back, spaced by
        // `dilation`.  Reading order is oldest-first so it matches the
        // kernel's weight layout `[out, kernel_tap, in]`.
        let kernel_stride_in_frames = self.dilation;
        for (output_channel, output_cell) in output.iter_mut().enumerate() {
            let mut accumulator = self.bias[output_channel];
            let channel_weight_base = output_channel * self.kernel_size * self.in_channels;

            for kernel_tap in 0..self.kernel_size {
                // Tap `kernel_tap = 0` is the oldest sample; `kernel_size - 1`
                // is the current input.  Offset backward from the just-written
                // slot by `(kernel_size - 1 - kernel_tap) · dilation` frames.
                let steps_back = (self.kernel_size - 1 - kernel_tap) * kernel_stride_in_frames;
                let tap_frame_index = (self.write_index + self.ring.len() / self.in_channels
                    - steps_back)
                    & self.ring_mask;
                let tap_frame_offset = tap_frame_index * self.in_channels;
                let tap_frame = &self.ring[tap_frame_offset..tap_frame_offset + self.in_channels];

                let weight_row_start = channel_weight_base + kernel_tap * self.in_channels;
                let weight_row =
                    &self.weights[weight_row_start..weight_row_start + self.in_channels];

                accumulator += self.kernels.dot(weight_row, tap_frame);
            }
            *output_cell = accumulator;
        }

        // ── Step 3: advance the ring's write pointer.
        self.write_index = (self.write_index + 1) & self.ring_mask;
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A dilation-1, kernel-1 conv is just a per-sample matmul + bias.
    #[test]
    fn kernel_one_matches_dense() {
        // Weights (2 out × 1 tap × 3 in) = [[1,2,3], [4,5,6]] in flat form.
        let weights = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bias = vec![10.0f32, 20.0];
        let mut conv = Conv1d::new(weights, bias, 3, 2, 1, 1).unwrap();
        let input = [1.0f32, 1.0, 1.0];
        let mut output = [0.0f32; 2];
        conv.forward(&input, &mut output);
        assert_eq!(output, [16.0, 35.0]);
    }

    /// Unit-tap kernel (size 1, all-ones weights, zero bias) sums the
    /// input channels for each output channel.
    #[test]
    fn kernel_one_ones_sums_channels() {
        let in_channels = 4;
        let out_channels = 2;
        let weights = vec![1.0f32; out_channels * in_channels];
        let bias = vec![0.0f32; out_channels];
        let mut conv = Conv1d::new(weights, bias, in_channels, out_channels, 1, 1).unwrap();

        let input = [1.0f32, 2.0, 3.0, 4.0];
        let mut output = [0.0f32; 2];
        conv.forward(&input, &mut output);
        assert_eq!(output, [10.0, 10.0]);
    }

    /// Kernel-3, dilation-1, 1 in / 1 out: a moving-average delta kernel
    /// `[1, 0, -1]` computes `x_t - x_{t-2}`.
    #[test]
    fn kernel_three_dilation_one_delta() {
        // weights layout [out=1, kernel=3, in=1] = [w0, w1, w2] where
        // w0 is the OLDEST tap (t-2) and w2 is the CURRENT sample (t).
        let weights = vec![-1.0f32, 0.0, 1.0];
        let bias = vec![0.0f32];
        let mut conv = Conv1d::new(weights, bias, 1, 1, 3, 1).unwrap();

        let inputs = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut outputs = Vec::new();
        for value in inputs {
            let mut output_frame = [0.0f32; 1];
            conv.forward(&[value], &mut output_frame);
            outputs.push(output_frame[0]);
        }
        // Expected: input_t - input_{t-2}, with zero-history at start.
        assert_eq!(outputs, vec![1.0, 2.0, 2.0, 2.0, 2.0]);
    }

    /// Kernel-2, dilation-2, 1 in / 1 out: `[1, 1]` weights with skip-1
    /// spacing compute `x_t + x_{t-2}`.
    #[test]
    fn kernel_two_dilation_two_sums_current_and_two_ago() {
        // weights layout [out=1, kernel=2, in=1] = [w0, w1] where w0 is
        // the OLDEST tap (t - 2) and w1 is the CURRENT sample (t).
        let weights = vec![1.0f32, 1.0];
        let bias = vec![0.0f32];
        let mut conv = Conv1d::new(weights, bias, 1, 1, 2, 2).unwrap();

        let inputs = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let mut outputs = Vec::new();
        for value in inputs {
            let mut output_frame = [0.0f32; 1];
            conv.forward(&[value], &mut output_frame);
            outputs.push(output_frame[0]);
        }
        // Expected: x_t + x_{t-2}: [10, 20, 10+30, 20+40, 30+50]
        //                        = [10, 20, 40, 60, 80]
        assert_eq!(outputs, vec![10.0, 20.0, 40.0, 60.0, 80.0]);
    }

    /// `reset` clears history — output after reset matches the fresh state.
    #[test]
    fn reset_clears_history() {
        let weights = vec![-1.0f32, 0.0, 1.0];
        let bias = vec![0.0f32];
        let mut conv = Conv1d::new(weights, bias, 1, 1, 3, 1).unwrap();

        // Push some samples through.
        for value in [1.0f32, 2.0, 3.0, 4.0] {
            let mut output = [0.0f32; 1];
            conv.forward(&[value], &mut output);
        }

        conv.reset();

        // Reset: fresh state, first output should just be the current input.
        let mut output = [0.0f32; 1];
        conv.forward(&[9.0], &mut output);
        assert_eq!(output[0], 9.0);
    }

    /// Rejects zero kernel_size / dilation.
    #[test]
    fn rejects_zero_kernel_or_dilation() {
        assert!(matches!(
            Conv1d::new(Vec::new(), Vec::new(), 1, 1, 0, 1),
            Err(ModelError::InvalidConv1dConfig { .. }),
        ));
        assert!(matches!(
            Conv1d::new(Vec::new(), Vec::new(), 1, 1, 1, 0),
            Err(ModelError::InvalidConv1dConfig { .. }),
        ));
    }
}
