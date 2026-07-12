//! LSTM cell — the workhorse of NAM / Proteus / GuitarML amp models.
//!
//! # The cell equations
//!
//! Given the current input `x`, previous hidden state `h_prev`, and previous
//! cell state `c_prev`, an LSTM cell produces the next `(h, c)` via four
//! gates:
//!
//! ```text
//! pre-activation = W_ih · x + W_hh · h_prev + bias        (shape [4·H])
//!
//! i = σ(pre[  0 ..  H ])   input gate
//! f = σ(pre[  H .. 2H ])   forget gate
//! g = tanh(pre[ 2H .. 3H ])  cell candidate
//! o = σ(pre[ 3H .. 4H ])   output gate
//!
//! c_new = f ⊙ c_prev + i ⊙ g
//! h_new = o ⊙ tanh(c_new)
//! ```
//!
//! `H` is the hidden size, `I` the input size.
//!
//! # Storage layout: **gate-major fused**
//!
//! Both weight matrices are stored **row-major** with the gate dimension
//! as the outer axis and the input dimension as the inner axis:
//!
//! ```text
//! W_ih shape = [4·H, I]     rows 0..H = i, H..2H = f, 2H..3H = g, 3H..4H = o
//! W_hh shape = [4·H, H]     same gate order
//! ```
//!
//! Biases are pre-fused at load time (`b_ih + b_hh` folded into a single
//! `bias` vector of length `4·H`), so the RT loop pays a single matvec +
//! bias add per weight matrix.
//!
//! This layout matches PyTorch's `nn.LSTM.weight_ih_l0` / `weight_hh_l0`
//! shapes exactly — the loader can copy them straight from disk without a
//! second transpose.
//!
//! # Real-time safety
//!
//! [`LstmCell::process_sample`] is allocation-free.  All scratch buffers
//! (`pre_activation`, `tanh_c`) live inside the cell struct and are reused
//! every sample.

use crate::activations::{sigmoid, tanh};
use crate::dense::Dense;
use crate::error::ModelError;

/// A single LSTM layer (one time-step at a time).
///
/// # Fields
///
/// - `weights_ih`, `weights_hh`: gate-major fused weight matrices packaged
///   as [`Dense`] layers.  We reuse the Dense layer because the underlying
///   math is a matvec + bias — no reason to duplicate SIMD dispatch.
/// - `hidden`, `cell`: recurrent state.  Sized at construction, zeroed on
///   [`reset`](Self::reset).
/// - `pre_activation`: `4·H`-long scratch buffer holding `W_ih · x` in the
///   first pass and `W_ih · x + W_hh · h_prev + bias` after the second.
pub struct LstmCell {
    /// Input→hidden dense: shape `[4H, I]` weights, `4H` bias.
    ///
    /// The bias here holds the fused `b_ih + b_hh` sum — see module docs.
    weights_ih: Dense,
    /// Hidden→hidden dense: shape `[4H, H]` weights, `4H` bias set to zero
    /// (the fused bias is already in `weights_ih`; we just need the matvec).
    weights_hh: Dense,

    /// Recurrent hidden state `h`, length `hidden_size`.
    hidden: Box<[f32]>,
    /// Recurrent cell state `c`, length `hidden_size`.
    cell: Box<[f32]>,

    /// Scratch buffer for gate pre-activations, length `4 · hidden_size`.
    pre_activation: Box<[f32]>,
    /// Scratch buffer for the second matvec into (added to pre_activation).
    pre_activation_hh: Box<[f32]>,

    input_size: usize,
    hidden_size: usize,
}

impl LstmCell {
    /// Construct an LSTM cell from gate-major fused weights.
    ///
    /// - `weights_ih` — length `4 · hidden_size · input_size`, row-major
    ///   with gate order `[i, f, g, o]`.
    /// - `weights_hh` — length `4 · hidden_size · hidden_size`, same order.
    /// - `bias` — length `4 · hidden_size`, holds the fused `b_ih + b_hh`
    ///   sum in gate-major order.
    ///
    /// **Not** real-time safe.
    pub fn new(
        weights_ih: Vec<f32>,
        weights_hh: Vec<f32>,
        bias: Vec<f32>,
        input_size: usize,
        hidden_size: usize,
    ) -> Result<Self, ModelError> {
        if hidden_size == 0 {
            return Err(ModelError::ZeroHiddenSize { size: hidden_size });
        }
        let gate_rows = 4 * hidden_size;
        // Build the two Dense layers.  weights_hh gets a zero bias — the
        // real bias lives with weights_ih so we add it in a single pass.
        let dense_ih = Dense::new(weights_ih, bias, input_size, gate_rows)?;
        let dense_hh = Dense::new(weights_hh, vec![0.0f32; gate_rows], hidden_size, gate_rows)?;

        Ok(Self {
            weights_ih: dense_ih,
            weights_hh: dense_hh,
            hidden: vec![0.0f32; hidden_size].into_boxed_slice(),
            cell: vec![0.0f32; hidden_size].into_boxed_slice(),
            pre_activation: vec![0.0f32; gate_rows].into_boxed_slice(),
            pre_activation_hh: vec![0.0f32; gate_rows].into_boxed_slice(),
            input_size,
            hidden_size,
        })
    }

    /// Input width expected per time step.
    #[inline]
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Hidden state width.
    #[inline]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Zero `h` and `c` — call on preset changes or stream restarts.
    ///
    /// **Real-time safe** — two contiguous `slice::fill(0.0)` stores.
    #[inline]
    pub fn reset(&mut self) {
        for cell_slot in self.hidden.iter_mut() {
            *cell_slot = 0.0;
        }
        for cell_slot in self.cell.iter_mut() {
            *cell_slot = 0.0;
        }
    }

    /// Read-only view of the current hidden state (exposed so downstream
    /// stacked LSTM layers can consume it as their input, without an
    /// extra copy).
    #[inline]
    pub fn hidden(&self) -> &[f32] {
        &self.hidden
    }

    /// Consume one input frame and update `(h, c)`.
    ///
    /// - `input.len()` must equal [`input_size`](Self::input_size).
    ///
    /// **Real-time safe** — no allocation, no locking.  Uses two SIMD-
    /// dispatched matvec+bias calls plus elementwise gate maths.
    #[inline]
    pub fn process_sample(&mut self, input: &[f32]) {
        debug_assert_eq!(input.len(), self.input_size);

        // ── Pass 1: pre = W_ih · x + bias   (bias lives in weights_ih)
        self.weights_ih.forward(input, &mut self.pre_activation);
        // ── Pass 2: pre_hh = W_hh · h_prev   (no bias)
        self.weights_hh
            .forward(&self.hidden, &mut self.pre_activation_hh);
        // ── Fuse the two pre-activations in place.
        for (target, addend) in self
            .pre_activation
            .iter_mut()
            .zip(self.pre_activation_hh.iter())
        {
            *target += addend;
        }

        // ── Gate maths.
        //
        // Gate order: [i, f, g, o] — each gate occupies `hidden_size`
        // consecutive slots of the pre-activation vector.
        let hidden_size = self.hidden_size;
        let (input_gate, forget_gate, cell_gate, output_gate) = {
            let (i_slice, remainder) = self.pre_activation.split_at(hidden_size);
            let (f_slice, remainder) = remainder.split_at(hidden_size);
            let (g_slice, o_slice) = remainder.split_at(hidden_size);
            (i_slice, f_slice, g_slice, o_slice)
        };

        // c_new = f ⊙ c_prev + i ⊙ tanh(g)
        // h_new = o ⊙ tanh(c_new)
        for lane_index in 0..hidden_size {
            let input_activation = sigmoid(input_gate[lane_index]);
            let forget_activation = sigmoid(forget_gate[lane_index]);
            let cell_candidate = tanh(cell_gate[lane_index]);
            let output_activation = sigmoid(output_gate[lane_index]);

            let previous_cell = self.cell[lane_index];
            let new_cell = forget_activation * previous_cell + input_activation * cell_candidate;
            self.cell[lane_index] = new_cell;
            self.hidden[lane_index] = output_activation * tanh(new_cell);
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// An all-zero LSTM must stay at `(h, c) = (0, 0)` forever regardless
    /// of input — because `sigmoid(0) = 0.5`, `tanh(0) = 0`, so `g = 0`
    /// and `c_new = 0.5 · 0 + 0.5 · 0 = 0`, `h_new = 0.5 · tanh(0) = 0`.
    #[test]
    fn zero_weights_and_bias_produce_zero_state() {
        let hidden_size = 4;
        let input_size = 1;
        let weights_ih = vec![0.0f32; 4 * hidden_size * input_size];
        let weights_hh = vec![0.0f32; 4 * hidden_size * hidden_size];
        let bias = vec![0.0f32; 4 * hidden_size];
        let mut cell =
            LstmCell::new(weights_ih, weights_hh, bias, input_size, hidden_size).unwrap();

        for step in 0..8 {
            cell.process_sample(&[step as f32]);
        }
        for hidden_value in cell.hidden() {
            assert!(hidden_value.abs() < 1e-6);
        }
    }

    /// A cell whose forget gate is saturated to `≈1` and input gate to `≈0`
    /// must preserve its cell state across time steps — even with arbitrary
    /// inputs.  Encode this by putting a huge positive bias on `f` and a
    /// huge negative bias on `i`.
    #[test]
    fn saturated_forget_preserves_cell_state() {
        let hidden_size = 2;
        let input_size = 1;
        let weights_ih = vec![0.0f32; 4 * hidden_size * input_size];
        let weights_hh = vec![0.0f32; 4 * hidden_size * hidden_size];
        // bias layout [i, f, g, o] × H each.  Force f_gate ≈ 1, i_gate ≈ 0.
        let mut bias = vec![0.0f32; 4 * hidden_size];
        for index in 0..hidden_size {
            bias[index] = -100.0; // i-gate → sigmoid(-100) ≈ 0
            bias[hidden_size + index] = 100.0; // f-gate → sigmoid(100) ≈ 1
        }
        let mut cell =
            LstmCell::new(weights_ih, weights_hh, bias, input_size, hidden_size).unwrap();

        // Manually seed the cell state (test-only reach-around).
        for index in 0..hidden_size {
            cell.cell[index] = 0.5;
        }

        // Run several steps and verify the cell state remains ≈ 0.5.
        for step in 0..16 {
            cell.process_sample(&[step as f32]);
        }
        for index in 0..hidden_size {
            assert!(
                (cell.cell[index] - 0.5).abs() < 1e-3,
                "cell[{index}]={} drifted from 0.5",
                cell.cell[index]
            );
        }
    }

    /// `reset` zeroes both `h` and `c`, so behaviour is identical to a
    /// fresh cell right after construction.
    #[test]
    fn reset_matches_freshly_constructed_behaviour() {
        let hidden_size = 3;
        let input_size = 1;
        let build_cell = || {
            let weights_ih: Vec<f32> = (0..4 * hidden_size * input_size)
                .map(|index| (index as f32) * 0.1)
                .collect();
            let weights_hh: Vec<f32> = (0..4 * hidden_size * hidden_size)
                .map(|index| (index as f32) * 0.05 - 0.5)
                .collect();
            let bias: Vec<f32> = (0..4 * hidden_size)
                .map(|index| (index as f32) * 0.02 - 0.1)
                .collect();
            LstmCell::new(weights_ih, weights_hh, bias, input_size, hidden_size).unwrap()
        };

        let mut cell_a = build_cell();
        let mut cell_b = build_cell();

        // Warm up cell_a, then reset — it must match a fresh cell_b.
        for warm_step in 0..10 {
            cell_a.process_sample(&[warm_step as f32]);
        }
        cell_a.reset();

        for step in 0..10 {
            let sample = step as f32 * 0.1 - 0.5;
            cell_a.process_sample(&[sample]);
            cell_b.process_sample(&[sample]);
        }

        for index in 0..hidden_size {
            assert!(
                (cell_a.hidden[index] - cell_b.hidden[index]).abs() < 1e-6,
                "reset diverged at index {index}: {} vs {}",
                cell_a.hidden[index],
                cell_b.hidden[index],
            );
        }
    }

    /// The rejected `hidden_size = 0` case surfaces a clear error.
    #[test]
    fn rejects_zero_hidden_size() {
        let result = LstmCell::new(vec![], vec![], vec![], 1, 0);
        assert!(matches!(result, Err(ModelError::ZeroHiddenSize { .. })));
    }
}
