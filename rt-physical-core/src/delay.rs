//! Integer circular delay line — the storage element of a Karplus-Strong
//! string loop and of the pluck-position comb.
//!
//! # Why integer-only here
//!
//! In the extended Karplus-Strong model the *fractional* part of the loop
//! delay is not handled by the buffer read: it is delegated to a first-order
//! all-pass tuning filter ([`crate::allpass::TuningAllpass`]) and to the known
//! phase delay of the loop filter ([`crate::filter::LoopFilter`]).  That split
//! keeps the ring buffer read a single, branch-free array access and puts the
//! sub-sample precision where it is cheapest and most accurate — see the Phase 0
//! tuning example in `docs/Roadmap-06-physical-modeling.md`.
//!
//! # Buffer sizing
//!
//! The backing `Vec<f32>` is rounded up to a power of two at construction and
//! never reallocated, so the read index wraps with a single mask (`& (len-1)`)
//! instead of a modulo.  Capacity is fixed; the string only ever changes which
//! integer tap it reads, never the buffer size.
//!
//! # Real-time safety
//!
//! [`read`](DelayLine::read) and [`write`](DelayLine::write) allocate nothing,
//! lock nothing and never panic in release builds.  `write` flushes subnormal
//! inputs to zero so a decaying loop cannot drag the FPU into the denormal slow
//! path even without a hardware `DenormalGuard`.

use rt_fft_convolver::flush_to_zero;

/// A fixed-capacity integer circular delay line.
#[derive(Debug, Clone)]
pub struct DelayLine {
    /// Circular sample storage; length is a power of two, fixed at construction.
    buf: Vec<f32>,
    /// `len - 1`, used to wrap indices with a single bitwise AND.
    mask: usize,
    /// Index of the slot the next [`write`](DelayLine::write) will fill.
    write_idx: usize,
}

impl DelayLine {
    /// Create a delay line able to serve any delay up to `capacity` samples
    /// (rounded up to a power of two, minimum 2).
    ///
    /// **Allocates** — construct off the render thread.
    pub fn new(capacity: usize) -> Self {
        let len = capacity.next_power_of_two().max(2);
        Self {
            buf: vec![0.0; len],
            mask: len - 1,
            write_idx: 0,
        }
    }

    /// The fixed buffer capacity in samples (a power of two).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Clear the buffer and rewind the write cursor.
    #[inline]
    pub fn reset(&mut self) {
        self.buf.iter_mut().for_each(|sample| *sample = 0.0);
        self.write_idx = 0;
    }

    /// Read the sample written `delay` steps ago.  `delay == 1` returns the most
    /// recent [`write`](DelayLine::write); `delay` is clamped to the capacity.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn read(&self, delay: usize) -> f32 {
        let delay = delay.clamp(1, self.mask);
        // `write_idx` points at the *next* slot to fill, so the sample written
        // `delay` steps ago lives at `write_idx - delay` (mod len).
        let index = self.write_idx.wrapping_sub(delay) & self.mask;
        self.buf[index]
    }

    /// Store `input` at the write cursor and advance it.
    ///
    /// **Real-time safe** — subnormal inputs are flushed to zero.
    #[inline]
    pub fn write(&mut self, input: f32) {
        self.buf[self.write_idx] = flush_to_zero(input);
        self.write_idx = (self.write_idx + 1) & self.mask;
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_is_power_of_two() {
        assert_eq!(DelayLine::new(100).capacity(), 128);
        assert_eq!(DelayLine::new(1).capacity(), 2);
        assert_eq!(DelayLine::new(4096).capacity(), 4096);
    }

    #[test]
    fn integer_delay_is_exact() {
        let mut delay = DelayLine::new(64);
        // Write an impulse then zeros; it must reappear at the matching tap.
        for sample_index in 0..40 {
            delay.write(if sample_index == 0 { 1.0 } else { 0.0 });
            // After writing sample n, `read(n+1)` is the impulse at index 0.
            if sample_index == 9 {
                assert!((delay.read(10) - 1.0).abs() < 1e-9);
                assert!(delay.read(9).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn read_delay_one_is_most_recent_write() {
        let mut delay = DelayLine::new(16);
        delay.write(0.25);
        assert_eq!(delay.read(1), 0.25);
        delay.write(0.5);
        assert_eq!(delay.read(1), 0.5);
        assert_eq!(delay.read(2), 0.25);
    }

    #[test]
    fn delay_is_clamped_into_range() {
        let mut delay = DelayLine::new(8); // capacity 8, mask 7
        delay.write(1.0);
        // A delay of 0 clamps to 1 (most recent), a huge delay clamps to mask.
        assert_eq!(delay.read(0), delay.read(1));
        assert!(delay.read(10_000).is_finite());
    }

    #[test]
    fn reset_clears_history() {
        let mut delay = DelayLine::new(16);
        for _ in 0..8 {
            delay.write(1.0);
        }
        delay.reset();
        for _ in 0..16 {
            assert_eq!(delay.read(1), 0.0);
            delay.write(0.0);
        }
    }

    #[test]
    fn subnormal_writes_are_flushed() {
        let mut delay = DelayLine::new(8);
        delay.write(1.0e-40); // subnormal f32
        assert_eq!(delay.read(1), 0.0);
    }
}
