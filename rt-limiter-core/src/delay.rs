//! [`DelayLine`] — a fixed integer-sample delay for the dry-signal path.
//!
//! A look-ahead limiter must delay the audio it eventually outputs by exactly
//! the look-ahead length, so the gain reduction computed from a *future* peak
//! lands on the sample that produced it.  This is that delay: a plain ring
//! buffer with a single read tap `delay` samples behind the write head.
//!
//! # Why a dedicated type (not `rt_fft_convolver::Ring`)
//!
//! `Ring` is a FIFO *queue* (push/pop changes its length).  A delay line has a
//! constant length and needs random read access at a fixed offset, so a
//! purpose-built ring with a read tap is both simpler and faster here.
//!
//! # Real-time safety
//!
//! [`DelayLine::new`] allocates the backing buffer once, sized for the maximum
//! delay.  [`DelayLine::process`] and [`DelayLine::set_delay`] never allocate,
//! lock or panic in release builds.

/// A fixed integer-sample delay implemented as a ring buffer with one read
/// tap.  Output sample `n` is input sample `n − delay`.
pub struct DelayLine {
    /// Backing store; length is `max_delay + 1` so a delay of `max_delay`
    /// still reads a slot the current write has not yet overwritten.
    buffer: Vec<f32>,
    /// Index the next sample is written to, in `[0, capacity)`.
    write: usize,
    /// Current delay in samples, in `[0, capacity)`.
    delay: usize,
}

impl DelayLine {
    /// Create a delay line able to delay by up to `max_delay` samples,
    /// initialised to `delay` samples.
    ///
    /// **Allocates** — call off the audio thread.
    ///
    /// # Panics
    ///
    /// Panics if `delay > max_delay` — a construction-time bug.
    pub fn new(max_delay: usize, delay: usize) -> Self {
        assert!(delay <= max_delay, "initial delay exceeds max_delay");
        Self {
            // +1: a delay of `max_delay` reads the slot written `max_delay`
            // samples ago, which must not be the slot about to be written.
            buffer: vec![0.0; max_delay + 1],
            write: 0,
            delay,
        }
    }

    /// The current delay in samples.
    #[inline]
    pub fn delay(&self) -> usize {
        self.delay
    }

    /// The maximum delay this line was sized for.
    #[inline]
    pub fn max_delay(&self) -> usize {
        self.buffer.len() - 1
    }

    /// Change the delay (must be `≤ max_delay`).  Clamped defensively so a bad
    /// value can never index out of bounds on the audio thread.
    ///
    /// The buffer content is kept, so a live change crossfades through
    /// whatever samples currently sit between the old and new taps.  Callers
    /// that need a click-free change should [`reset`](Self::reset) or ramp
    /// around it.
    #[inline]
    pub fn set_delay(&mut self, delay: usize) {
        self.delay = delay.min(self.max_delay());
    }

    /// Push one input sample and return the sample `delay` samples old.
    ///
    /// With `delay == 0` this returns `sample` unchanged (identity), because
    /// the read tap lands on the slot just written.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn process(&mut self, sample: f32) -> f32 {
        let capacity = self.buffer.len();
        // Write first, then read `delay` behind, so delay 0 returns `sample`.
        self.buffer[self.write] = sample;
        let read = (self.write + capacity - self.delay) % capacity;
        let out = self.buffer[read];
        self.write = (self.write + 1) % capacity;
        out
    }

    /// Zero the delay line and rewind the write head.
    pub fn reset(&mut self) {
        self.buffer.iter_mut().for_each(|slot| *slot = 0.0);
        self.write = 0;
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Delay 0 is the identity: every sample comes straight back out.
    #[test]
    fn zero_delay_is_identity() {
        let mut line = DelayLine::new(8, 0);
        for value in [0.5, -0.25, 1.0, -1.0] {
            assert_eq!(line.process(value), value);
        }
    }

    /// A delay of `d` returns the sample fed `d` steps earlier; the first `d`
    /// outputs are the zero-initialised prefix.
    #[test]
    fn fixed_delay_shifts_by_d() {
        let delay = 3;
        let mut line = DelayLine::new(8, delay);
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = Vec::new();
        for &value in &input {
            output.push(line.process(value));
        }
        // First `delay` outputs are the initial zeros...
        assert_eq!(&output[..delay], &[0.0, 0.0, 0.0]);
        // ...then the input, delayed by `delay`.
        assert_eq!(&output[delay..], &input[..input.len() - delay]);
    }

    /// A delay equal to `max_delay` is representable (the +1 slot guarantees
    /// the read tap never collides with the write).
    #[test]
    fn max_delay_is_usable() {
        let mut line = DelayLine::new(4, 4);
        let mut last = 0.0;
        for step in 0..12 {
            last = line.process(step as f32);
        }
        // Output at step 11 is input from step 11 − 4 = 7.
        assert_eq!(last, 7.0);
    }

    /// `set_delay` clamps to `max_delay` rather than panicking or wrapping.
    #[test]
    fn set_delay_clamps() {
        let mut line = DelayLine::new(4, 0);
        line.set_delay(100);
        assert_eq!(line.delay(), 4);
    }

    /// `reset` clears buffered history.
    #[test]
    fn reset_clears_history() {
        let mut line = DelayLine::new(8, 2);
        line.process(9.0);
        line.process(9.0);
        line.reset();
        // After reset the delayed reads are zeros again.
        assert_eq!(line.process(1.0), 0.0);
        assert_eq!(line.process(2.0), 0.0);
        assert_eq!(line.process(3.0), 1.0);
    }
}
