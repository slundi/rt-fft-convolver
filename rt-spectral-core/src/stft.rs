//! Streaming STFT framing: block-rate in, hop-rate frames out, and back.
//!
//! The host delivers arbitrary block sizes; the spectral engine consumes a
//! fixed *frame* every *hop* samples and produces one synthesis frame per
//! analysis frame.  [`Analyzer`] and [`Synthesizer`] bridge the two rates on
//! top of the RT-safe [`Ring`](rt_fft_convolver::Ring) FIFO from the core
//! crate — the private overlap-add framer shared with the pitch plugin
//! (Roadmap-04 prerequisite: ring-buffer utilities).
//!
//! ```text
//!   push_block ─▶ [input Ring] ─▶ next_frame (peek F, drop H) ─▶ analysis frame
//!   synthesis frame ─▶ add_frame (OLA into acc, emit H) ─▶ [output Ring] ─▶ pull_block
//! ```

use rt_fft_convolver::Ring;

/// Accumulates the host's input and emits fixed-size analysis frames every
/// `hop` samples (a sliding window with `frame − hop` overlap).
#[derive(Debug)]
pub struct Analyzer {
    ring: Ring<f32>,
    frame: usize,
    hop: usize,
}

impl Analyzer {
    /// Build an analyzer for `frame`-length windows advanced by `hop`,
    /// sized to absorb host blocks up to `max_block` samples without overflow.
    ///
    /// **Allocates** — off-RT only.
    ///
    /// # Panics
    /// Panics unless `0 < hop <= frame`.
    pub fn new(frame: usize, hop: usize, max_block: usize) -> Self {
        assert!(hop > 0 && hop <= frame, "require 0 < hop <= frame");
        // Worst case occupancy is a residual of up to `frame - 1` samples plus
        // a full incoming block; `frame + max_block` always covers it.
        Self {
            ring: Ring::with_capacity(frame + max_block.max(1)),
            frame,
            hop,
        }
    }

    /// Queue a host input block.  The caller guarantees `block.len() <=
    /// max_block`, so the push never truncates.
    #[inline]
    pub fn push_block(&mut self, block: &[f32]) {
        let written = self.ring.push_slice(block);
        debug_assert_eq!(written, block.len(), "analyzer ring overflow");
    }

    /// Copy the next analysis frame into `frame_out` and advance by one hop,
    /// returning `false` when fewer than `frame` samples are buffered.
    ///
    /// The frame is *peeked* (the oldest `frame` samples) but only `hop`
    /// samples are consumed, producing the overlap successive frames need.
    pub fn next_frame(&mut self, frame_out: &mut [f32]) -> bool {
        debug_assert_eq!(frame_out.len(), self.frame);
        if self.ring.len() < self.frame {
            return false;
        }
        // `as_slices` may split the window across the wrap point; stitch the
        // first `frame` samples back into the caller's contiguous buffer.
        let (head, tail) = self.ring.as_slices();
        let from_head = head.len().min(self.frame);
        frame_out[..from_head].copy_from_slice(&head[..from_head]);
        let remaining = self.frame - from_head;
        if remaining > 0 {
            frame_out[from_head..].copy_from_slice(&tail[..remaining]);
        }
        // Consume exactly one hop.
        drain(&mut self.ring, self.hop);
        true
    }

    /// Pre-load `count` leading zeros into the input ring.
    ///
    /// Priming with `frame − hop` zeros makes the analysis/synthesis group
    /// delay exactly `frame − hop` **independent of the host block size**: the
    /// first frame then becomes available after only `hop` real samples, so
    /// frame production tracks the input rate and the synthesis FIFO never
    /// starves (which would otherwise inject block-size-dependent extra delay).
    ///
    /// **Allocates nothing.**  Off-RT setup helper.
    #[inline]
    pub fn prime_zeros(&mut self, count: usize) {
        for _ in 0..count {
            let _ = self.ring.push(0.0);
        }
    }

    /// Drop all buffered input.
    #[inline]
    pub fn reset(&mut self) {
        self.ring.clear();
    }
}

/// Overlap-adds synthesis frames and hands the host fixed output blocks.
#[derive(Debug)]
pub struct Synthesizer {
    /// Overlap-add accumulator, length `frame`; holds the tail that future
    /// frames still contribute to.
    accumulator: Vec<f32>,
    /// Output samples produced but not yet pulled by the host.
    ready: Ring<f32>,
    frame: usize,
    hop: usize,
    /// COLA reciprocal gain applied on emission (see [`crate::window`]).
    norm: f32,
}

impl Synthesizer {
    /// Build a synthesizer matching an [`Analyzer`]'s `frame`/`hop`, applying
    /// the overlap-add normalisation `norm`, sized for `max_block` pulls.
    ///
    /// **Allocates** — off-RT only.
    pub fn new(frame: usize, hop: usize, max_block: usize, norm: f32) -> Self {
        assert!(hop > 0 && hop <= frame, "require 0 < hop <= frame");
        Self {
            accumulator: vec![0.0; frame],
            ready: Ring::with_capacity(frame + max_block.max(1)),
            frame,
            hop,
            norm,
        }
    }

    /// Overlap-add one synthesis frame and emit the next `hop` output samples.
    ///
    /// The oldest `hop` samples of the accumulator are finalised (all frames
    /// that overlap them have now been added), pushed to the output FIFO, and
    /// the accumulator slides left by one hop.
    pub fn add_frame(&mut self, frame_in: &[f32]) {
        debug_assert_eq!(frame_in.len(), self.frame);
        for (acc, &sample) in self.accumulator.iter_mut().zip(frame_in) {
            *acc += sample * self.norm;
        }
        let written = self.ready.push_slice(&self.accumulator[..self.hop]);
        debug_assert_eq!(written, self.hop, "synthesizer ring overflow");

        // Slide the accumulator left by one hop and zero the freed tail.
        self.accumulator.copy_within(self.hop.., 0);
        let tail_start = self.frame - self.hop;
        self.accumulator[tail_start..].fill(0.0);
    }

    /// Fill `block_out` from the output FIFO, zero-padding any shortfall (which
    /// happens only during the initial latency warm-up).
    #[inline]
    pub fn pull_block(&mut self, block_out: &mut [f32]) {
        let read = self.ready.pop_slice(block_out);
        block_out[read..].fill(0.0);
    }

    /// Pre-load `count` leading zeros into the output FIFO.
    ///
    /// Priming with one `hop` guarantees the FIFO never starves for **any**
    /// host block size (including blocks smaller than one hop): production only
    /// begins after the analyzer's first frame, so without this cushion a run
    /// of sub-hop blocks would pull past what has been produced and inject
    /// block-size-dependent extra latency.  With it, the group delay is a
    /// constant `frame` (analyzer prime `frame − hop` + this `hop`).
    ///
    /// **Allocates nothing.**  Off-RT setup helper.
    #[inline]
    pub fn prime_zeros(&mut self, count: usize) {
        for _ in 0..count {
            let _ = self.ready.push(0.0);
        }
    }

    /// Number of finished output samples awaiting the host.
    #[inline]
    pub fn available(&self) -> usize {
        self.ready.len()
    }

    /// Clear the overlap-add tail and any pending output.
    #[inline]
    pub fn reset(&mut self) {
        self.accumulator.fill(0.0);
        self.ready.clear();
    }
}

/// Pop and discard `count` samples from the front of `ring`.
fn drain(ring: &mut Ring<f32>, count: usize) {
    for _ in 0..count {
        ring.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::window::{Window, WindowKind};

    /// Frames appear only once `frame` samples are buffered, then every hop.
    #[test]
    fn analyzer_emits_on_hop_boundaries() {
        let (frame, hop) = (8, 2);
        let mut analyzer = Analyzer::new(frame, hop, 64);
        let mut out = vec![0.0f32; frame];

        analyzer.push_block(&[1.0, 2.0, 3.0, 4.0]); // 4 < frame → no frame yet
        assert!(!analyzer.next_frame(&mut out));

        analyzer.push_block(&[5.0, 6.0, 7.0, 8.0]); // now 8 buffered
        assert!(analyzer.next_frame(&mut out));
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        // Only one hop (2 samples) consumed → next frame slides by 2.
        analyzer.push_block(&[9.0, 10.0]);
        assert!(analyzer.next_frame(&mut out));
        assert_eq!(out, vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    }

    /// With Hann² analysis+synthesis weighting and the matching COLA norm, the
    /// framer reconstructs the input after the framing delay, to within the
    /// window taper.
    #[test]
    fn analysis_synthesis_reconstructs() {
        let (frame, hop) = (64, 16);
        let window = Window::new(WindowKind::Hann, frame);
        let norm = window.ola_normalisation(hop);
        let win = window.as_slice();

        let mut analyzer = Analyzer::new(frame, hop, 256);
        let mut synth = Synthesizer::new(frame, hop, 256, norm);

        // A slowly varying signal so the window taper barely distorts it.
        let input: Vec<f32> = (0..1024).map(|n| (n as f32 * 0.01).sin()).collect();

        let mut frame_buf = vec![0.0f32; frame];
        let mut output = Vec::new();
        let mut cursor = 0;
        while cursor < input.len() {
            let end = (cursor + hop).min(input.len());
            analyzer.push_block(&input[cursor..end]);
            cursor = end;
            while analyzer.next_frame(&mut frame_buf) {
                // Apply analysis then synthesis window (w² weighting).
                for (sample, &w) in frame_buf.iter_mut().zip(win) {
                    *sample *= w * w;
                }
                synth.add_frame(&frame_buf);
            }
            let mut block = vec![0.0f32; hop];
            synth.pull_block(&mut block);
            output.extend_from_slice(&block);
        }

        // Compare an interior region (past the warm-up) allowing the framing
        // latency of `frame - hop` samples.
        let latency = frame - hop;
        let mut max_err = 0.0f32;
        for n in (frame * 2)..(input.len() - frame) {
            let err = (output[n] - input[n - latency]).abs();
            max_err = max_err.max(err);
        }
        assert!(max_err < 1e-3, "reconstruction error {max_err}");
    }

    /// Reset returns both halves to a silent, empty state.
    #[test]
    fn reset_clears_state() {
        let (frame, hop) = (8, 2);
        let mut analyzer = Analyzer::new(frame, hop, 32);
        let mut synth = Synthesizer::new(frame, hop, 32, 1.0);
        analyzer.push_block(&[1.0; 8]);
        synth.add_frame(&[1.0; 8]);
        analyzer.reset();
        synth.reset();
        assert_eq!(synth.available(), 0);
        let mut out = vec![0.0; 8];
        assert!(!analyzer.next_frame(&mut out));
    }
}
