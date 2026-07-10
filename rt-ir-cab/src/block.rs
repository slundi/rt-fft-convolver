//! [`StereoBlockAdapter`] — bridges a host's variable-length buffers to the
//! fixed block size the convolvers require.
//!
//! [`HybridConvolver`](rt_fft_convolver::HybridConvolver) processes exactly
//! `block_size` samples per call, but audio hosts hand out buffers of any
//! length (and the length can change block to block).  This adapter buffers
//! input until a full block is available, runs [`CabProcessor::process`], and
//! drains the result — introducing a constant latency of `block_size` samples,
//! which the plugin reports to the host.
//!
//! Output is always stereo (the cab's fixed I/O); input may be mono or stereo.
//! All buffers are pre-allocated, so [`process_buffer`] never allocates.
//!
//! [`process_buffer`]: StereoBlockAdapter::process_buffer

use crate::engine::CabProcessor;

/// Fixed-block adapter around a stereo [`CabProcessor`].
pub struct StereoBlockAdapter {
    proc: CabProcessor,
    block_size: usize,
    stereo_in: bool,

    // One block of pending input being filled (left + right).
    in_l: Vec<f32>,
    in_r: Vec<f32>,
    // The most recently processed block being drained (left + right).
    out_l: Vec<f32>,
    out_r: Vec<f32>,
    // Write/read cursor shared by fill and drain, in `0..block_size`.
    pos: usize,
}

impl StereoBlockAdapter {
    /// Wrap `proc` (which must have been built for **2 output channels**).
    ///
    /// `stereo_in` selects whether the second channel is read as an
    /// independent input or the mono left input is fanned out to both
    /// convolvers.
    pub fn new(proc: CabProcessor, stereo_in: bool) -> Self {
        let block_size = proc.block_size();
        Self {
            proc,
            block_size,
            stereo_in,
            in_l: vec![0.0; block_size],
            in_r: vec![0.0; block_size],
            // Primed with silence → the first `block_size` outputs are the
            // reported latency.
            out_l: vec![0.0; block_size],
            out_r: vec![0.0; block_size],
            pos: 0,
        }
    }

    /// Latency the adapter introduces, in samples (equal to the block size).
    #[inline]
    pub fn latency_samples(&self) -> usize {
        self.block_size
    }

    /// Borrow the wrapped processor (e.g. to apply parameter updates).
    #[inline]
    pub fn processor_mut(&mut self) -> &mut CabProcessor {
        &mut self.proc
    }

    /// Process a host buffer **in place**, one channel slice per element.
    ///
    /// `channels[0]` is the left channel and `channels[1]` (if present) the
    /// right; each is read as input and overwritten with the (block-delayed)
    /// output.  Extra channels are ignored.  Accepts any length — this is fed
    /// directly from `nih_plug::buffer::Buffer::as_slice`.
    ///
    /// **Real-time safe** — no allocation.
    pub fn process_buffer(&mut self, channels: &mut [&mut [f32]]) {
        if channels.is_empty() {
            return;
        }
        let len = channels.iter().map(|c| c.len()).min().unwrap_or(0);
        let has_right = channels.len() >= 2;

        // Index-based: each iteration touches two host channels *and* the
        // internal fill/drain cursor, which an iterator zip would not clarify.
        #[allow(clippy::needless_range_loop)]
        for i in 0..len {
            // Read the input sample(s) *before* overwriting, so an in-place
            // host buffer (input aliased with output) stays correct.
            let li = channels[0][i];
            let ri = if self.stereo_in && has_right {
                channels[1][i]
            } else {
                li
            };

            // Drain the previously-processed block into the host buffer.
            channels[0][i] = self.out_l[self.pos];
            if has_right {
                channels[1][i] = self.out_r[self.pos];
            }

            // Stage the input for the block currently being assembled.
            self.in_l[self.pos] = li;
            self.in_r[self.pos] = ri;

            self.pos += 1;
            if self.pos == self.block_size {
                // Full block staged: convolve it into the drain buffers.  `proc`
                // and the four scratch vecs are distinct fields, so these
                // simultaneous borrows are allowed and allocation-free.
                if self.stereo_in {
                    let ins: [&[f32]; 2] = [&self.in_l, &self.in_r];
                    let mut outs: [&mut [f32]; 2] = [&mut self.out_l, &mut self.out_r];
                    self.proc.process(&ins, &mut outs);
                } else {
                    let ins: [&[f32]; 1] = [&self.in_l];
                    let mut outs: [&mut [f32]; 2] = [&mut self.out_l, &mut self.out_r];
                    self.proc.process(&ins, &mut outs);
                }
                self.pos = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::CabController;
    use crate::params::CabParams;

    const BLOCK: usize = 16;
    const SR: u32 = 48_000;

    #[test]
    fn reports_block_size_latency() {
        let params = CabParams::default();
        let (_c, proc) = CabController::new(BLOCK, SR, 2, &params);
        let adapter = StereoBlockAdapter::new(proc, false);
        assert_eq!(adapter.latency_samples(), BLOCK);
    }

    #[test]
    fn passes_input_through_delayed_by_one_block() {
        // No IR loaded → dry passthrough (unity), so the adapter's only effect
        // is a `block_size` delay.  Feed a ramp and check the delay exactly,
        // processing it in-place through odd-sized chunks to exercise
        // arbitrary host block lengths.
        let params = CabParams::default();
        let (_c, proc) = CabController::new(BLOCK, SR, 2, &params);
        let mut adapter = StereoBlockAdapter::new(proc, false);

        let total = BLOCK * 3;
        let input: Vec<f32> = (0..total).map(|i| (i + 1) as f32).collect();
        let mut work_l = input.clone();
        let mut work_r = vec![0.0f32; total];

        let mut start = 0;
        for chunk in [7usize, 20, total - 27] {
            let end = start + chunk;
            {
                let mut chans: [&mut [f32]; 2] = [&mut work_l[start..end], &mut work_r[start..end]];
                adapter.process_buffer(&mut chans);
            }
            start = end;
        }

        // First block_size samples are the primed silence (the latency)…
        assert!(work_l[..BLOCK].iter().all(|&s| s == 0.0));
        // …then the input reappears, delayed by exactly block_size.
        for i in BLOCK..total {
            assert!(
                (work_l[i] - input[i - BLOCK]).abs() < 1e-6,
                "sample {i}: {} != {}",
                work_l[i],
                input[i - BLOCK]
            );
            // Mono-in fans out to both channels identically.
            assert_eq!(work_l[i], work_r[i]);
        }
    }
}
