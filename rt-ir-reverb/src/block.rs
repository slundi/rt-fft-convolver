//! [`StereoBlockAdapter`] — bridges variable-length host buffers to the fixed
//! block size the reverb convolvers require, introducing a constant
//! `block_size` latency (reported to the host).
//!
//! Same pattern as the cab's adapter, but stereo in / stereo out.  All buffers
//! are pre-allocated, so [`process_buffer`](StereoBlockAdapter::process_buffer)
//! never allocates.

use crate::engine::ReverbProcessor;

/// Fixed-block adapter around a stereo [`ReverbProcessor`].
pub struct StereoBlockAdapter {
    proc: ReverbProcessor,
    block_size: usize,
    in_l: Vec<f32>,
    in_r: Vec<f32>,
    out_l: Vec<f32>,
    out_r: Vec<f32>,
    pos: usize,
}

impl StereoBlockAdapter {
    /// Wrap a stereo [`ReverbProcessor`].
    pub fn new(proc: ReverbProcessor) -> Self {
        let block_size = proc.block_size();
        Self {
            proc,
            block_size,
            in_l: vec![0.0; block_size],
            in_r: vec![0.0; block_size],
            out_l: vec![0.0; block_size],
            out_r: vec![0.0; block_size],
            pos: 0,
        }
    }

    /// Latency the adapter introduces, in samples (the block size).
    #[inline]
    pub fn latency_samples(&self) -> usize {
        self.block_size
    }

    /// Borrow the wrapped processor (e.g. to apply parameter updates).
    #[inline]
    pub fn processor_mut(&mut self) -> &mut ReverbProcessor {
        &mut self.proc
    }

    /// Process a stereo host buffer **in place** (`channels[0]` = L,
    /// `channels[1]` = R).  Accepts any length; fed from `Buffer::as_slice`.
    ///
    /// **Real-time safe** — no allocation.
    pub fn process_buffer(&mut self, channels: &mut [&mut [f32]]) {
        if channels.len() < 2 {
            return;
        }
        let len = channels[0].len().min(channels[1].len());

        // Index-based: each iteration touches both host channels *and* the
        // internal fill/drain cursor, which an iterator zip would not clarify.
        #[allow(clippy::needless_range_loop)]
        for i in 0..len {
            // Read before overwriting so an in-place host buffer stays correct.
            let li = channels[0][i];
            let ri = channels[1][i];

            channels[0][i] = self.out_l[self.pos];
            channels[1][i] = self.out_r[self.pos];

            self.in_l[self.pos] = li;
            self.in_r[self.pos] = ri;

            self.pos += 1;
            if self.pos == self.block_size {
                // `proc` and the four scratch vecs are distinct fields → these
                // simultaneous borrows are allowed and allocation-free.
                self.proc
                    .process(&self.in_l, &self.in_r, &mut self.out_l, &mut self.out_r);
                self.pos = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ReverbController;
    use crate::params::ReverbParams;

    const BLOCK: usize = 16;
    const SR: u32 = 48_000;

    #[test]
    fn reports_block_size_latency() {
        let params = ReverbParams::default();
        let (_c, proc) = ReverbController::new(BLOCK, SR, &params);
        let adapter = StereoBlockAdapter::new(proc);
        assert_eq!(adapter.latency_samples(), BLOCK);
    }

    #[test]
    fn dry_passthrough_delayed_by_one_block() {
        // No IR + dry 0 dB + wet disabled → the only effect is a block delay.
        let params = ReverbParams::default();
        let (_c, proc) = ReverbController::new(BLOCK, SR, &params);
        let mut adapter = StereoBlockAdapter::new(proc);

        let total = BLOCK * 3;
        let input: Vec<f32> = (0..total).map(|i| (i + 1) as f32).collect();
        let mut work_l = input.clone();
        let mut work_r = input.clone();

        let mut start = 0;
        for chunk in [5usize, 19, total - 24] {
            let end = start + chunk;
            {
                let mut chans: [&mut [f32]; 2] = [&mut work_l[start..end], &mut work_r[start..end]];
                adapter.process_buffer(&mut chans);
            }
            start = end;
        }

        assert!(work_l[..BLOCK].iter().all(|&s| s == 0.0));
        for i in BLOCK..total {
            assert!((work_l[i] - input[i - BLOCK]).abs() < 1e-6, "L sample {i}");
            assert!((work_r[i] - input[i - BLOCK]).abs() < 1e-6, "R sample {i}");
        }
    }
}
