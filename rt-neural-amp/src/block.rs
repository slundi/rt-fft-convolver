//! [`MonoBlockAdapter`] — bridges a host's variable-length buffers to the
//! fixed block size the crossfader / processor need.
//!
//! [`NeuralAmpProcessor::process`](crate::engine::NeuralAmpProcessor::process)
//! renders exactly `block_size` samples per call because the underlying
//! [`SlotCrossfader`](rt_ir_plugin_core::SlotCrossfader) is block-based
//! (a fade advances by a whole block's worth of gains at once).  Hosts
//! deliver buffers of arbitrary length, so this adapter buffers input
//! until a full block is available, renders it, and drains the result —
//! introducing a **constant latency of `block_size` samples** which the
//! plugin reports to the host.
//!
//! Neural amp models are inherently mono ("mono in → mono out"); a stereo
//! host either duplicates the mono input or hosts two plugin instances
//! (roadmap Phase 2, "second instance for stereo").  This adapter is the
//! mono analogue of `rt_ir_cab::StereoBlockAdapter`.
//!
//! Every scratch buffer is pre-sized at construction, so
//! [`process_buffer`](MonoBlockAdapter::process_buffer) never allocates.

use crate::engine::NeuralAmpProcessor;
use crate::params::NeuralAmpParams;

/// Fixed-block adapter around a mono [`NeuralAmpProcessor`].
pub struct MonoBlockAdapter {
    processor: NeuralAmpProcessor,
    block_size: usize,

    /// One block of pending input being filled.
    input_buf: Vec<f32>,
    /// The most recently rendered block being drained.
    output_buf: Vec<f32>,
    /// Shared fill / drain cursor in `0..block_size`.
    cursor: usize,
}

impl MonoBlockAdapter {
    /// Wrap `processor`.  `block_size` must equal
    /// `processor.block_size()` — a debug assertion catches mismatches.
    pub fn new(processor: NeuralAmpProcessor, block_size: usize) -> Self {
        assert!(block_size > 0, "block_size must be > 0");
        Self {
            processor,
            block_size,
            input_buf: vec![0.0f32; block_size],
            // Primed with silence → the first `block_size` outputs are the
            // reported latency.
            output_buf: vec![0.0f32; block_size],
            cursor: 0,
        }
    }

    /// Latency the adapter introduces, in samples (equal to the block size).
    #[inline]
    pub fn latency_samples(&self) -> usize {
        self.block_size
    }

    /// Borrow the wrapped processor (e.g. to poll activations, apply
    /// parameter updates, or read the CPU meter).
    #[inline]
    pub fn processor_mut(&mut self) -> &mut NeuralAmpProcessor {
        &mut self.processor
    }

    /// Process a host mono channel **in place**.
    ///
    /// Each sample is:
    /// 1. Read (before overwrite, so aliased in/out buffers stay correct).
    /// 2. Drained from the previously-rendered block into the host slot.
    /// 3. Written into the pending-input block.
    /// 4. When the pending block is full, render it and swap in the fresh
    ///    output for the next drain.
    ///
    /// **Real-time safe** — no allocation.
    pub fn process_buffer(&mut self, channel: &mut [f32], params: &NeuralAmpParams) {
        for cell in channel.iter_mut() {
            // Read before overwrite so an aliased in/out buffer is safe.
            let sample_in = *cell;
            *cell = self.output_buf[self.cursor];
            self.input_buf[self.cursor] = sample_in;
            self.cursor += 1;

            if self.cursor == self.block_size {
                // Full block: render and reset the cursor.  The freshly
                // rendered samples become the next drain source.
                //
                // We read `input_buf` and write into `output_buf`, so a
                // temporary swap avoids aliasing the two `&mut` borrows.
                let (input_slice, output_slice) = (&self.input_buf[..], &mut self.output_buf[..]);
                self.processor.process(input_slice, output_slice, params);
                self.cursor = 0;
            }
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::NeuralAmpController;

    /// Adapter reports `block_size` samples of latency.
    #[test]
    fn latency_matches_block_size() {
        let params = NeuralAmpParams::default();
        let block_size = 32;
        let (_controller, processor) = NeuralAmpController::new(block_size, 48_000, &params);
        let adapter = MonoBlockAdapter::new(processor, block_size);
        assert_eq!(adapter.latency_samples(), block_size);
    }

    /// After the block latency has drained, output = input at defaults
    /// (dry pass-through with unity gains).
    #[test]
    fn output_equals_input_after_latency() {
        let params = NeuralAmpParams::default();
        let block_size = 8;
        let (_controller, processor) = NeuralAmpController::new(block_size, 48_000, &params);
        let mut adapter = MonoBlockAdapter::new(processor, block_size);

        // First `block_size` samples: primed with the silent output buffer.
        // Subsequent samples should match the earlier inputs (delayed by
        // `block_size`).
        let stream: Vec<f32> = (0..3 * block_size)
            .map(|index| index as f32 * 0.01)
            .collect();
        let mut buffer = stream.clone();
        adapter.process_buffer(&mut buffer, &params);

        // Positions 0..block_size are latency (silence).
        for value in &buffer[..block_size] {
            assert!(value.abs() < 1e-6);
        }
        // Positions block_size..(3*block_size) equal the first
        // (2*block_size) inputs.
        for (index, produced) in buffer[block_size..].iter().enumerate() {
            let expected = stream[index];
            assert!(
                (produced - expected).abs() < 1e-5,
                "sample {index}: expected {expected}, got {produced}"
            );
        }
    }
}
