//! Fractional, LFO-modulated circular delay line — the core storage element of
//! each FDN branch.
//!
//! # Read / write split
//!
//! Unlike a plain delay, an FDN branch must (1) read its delayed output,
//! (2) route it through the mixing matrix together with every other branch,
//! then (3) write the *fed-back* result back in.  So this type deliberately
//! separates [`read`](ModulatedDelay::read) from [`write`](ModulatedDelay::write)
//! instead of offering a single `process`.  **Contract:** call `read()` exactly
//! once and then `write()` exactly once per sample, `read` first.  `read`
//! advances the LFO; `write` advances the write cursor.
//!
//! # Fractional read
//!
//! The modulated delay length is generally non-integer, so the read is linearly
//! interpolated between the two neighbouring samples.  Linear interpolation
//! adds a gentle, delay-dependent high-frequency roll-off — inaudible here and
//! far cheaper than all-pass or cubic interpolation, which the slow LFO sweep
//! does not warrant.
//!
//! # Buffer sizing (the "cache-aligned buffers" roadmap item)
//!
//! The backing `Vec<f32>` is rounded up to a power of two at construction and
//! never reallocated afterwards, so the whole modulated read window lives in
//! one contiguous, cache-friendly block and the write cursor wraps with a
//! single compare.  Capacity is fixed at construction: all mutation
//! (`set_base_delay`, `set_mod_depth`) is clamped to fit, so the render thread
//! never allocates.
//!
//! # Real-time safety
//!
//! [`read`](ModulatedDelay::read) and [`write`](ModulatedDelay::write) allocate
//! nothing, lock nothing and never panic in release builds.  `write` runs each
//! stored sample through [`flush_to_zero`] so a decaying tail cannot poison the
//! feedback path with subnormals even if the caller forgot the hardware guard.

use crate::lfo::SineLfo;
use rt_fft_convolver::flush_to_zero;

/// A modulated fractional delay line with a fixed-capacity circular buffer.
#[derive(Debug)]
pub struct ModulatedDelay {
    /// Circular sample storage; length is a power of two, fixed at construction.
    buf: Vec<f32>,
    /// Index of the slot the next [`write`](ModulatedDelay::write) will fill.
    write_idx: usize,
    /// Nominal delay in samples (centre of the modulation sweep).
    base_delay: f32,
    /// Peak modulation deviation in samples (the LFO swings `±mod_depth`).
    mod_depth: f32,
    /// Per-line modulation oscillator.
    lfo: SineLfo,
}

impl ModulatedDelay {
    /// Create a delay line able to hold at least `capacity` samples (rounded up
    /// to a power of two), initialised to `base_delay` samples of delay with
    /// `±mod_depth` samples of LFO sweep driven by `lfo`.
    ///
    /// **Allocates** — construct off the render thread.
    pub fn new(capacity: usize, base_delay: f32, mod_depth: f32, lfo: SineLfo) -> Self {
        let len = capacity.next_power_of_two().max(2);
        let mut delay = Self {
            buf: vec![0.0; len],
            write_idx: 0,
            base_delay: 1.0,
            mod_depth: 0.0,
            lfo,
        };
        delay.set_mod_depth(mod_depth);
        delay.set_base_delay(base_delay);
        delay
    }

    /// The fixed buffer capacity in samples (a power of two).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Largest delay the buffer can serve while leaving one sample of
    /// interpolation headroom.
    #[inline]
    fn max_delay(&self) -> f32 {
        (self.buf.len() - 2) as f32
    }

    /// Set the nominal delay, clamped so that `base ± mod_depth` always stays in
    /// the readable range `[1, capacity − 2]`.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn set_base_delay(&mut self, base_delay: f32) {
        let lo = 1.0 + self.mod_depth;
        let hi = self.max_delay() - self.mod_depth;
        // If the buffer is too small for the requested depth, hi may fall below
        // lo; `max`/`min` in this order keeps the result finite and in-range.
        self.base_delay = base_delay.clamp(lo, hi.max(lo));
    }

    /// Set the peak modulation depth in samples, clamped so the swept read stays
    /// within the buffer around the current `base_delay`.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn set_mod_depth(&mut self, mod_depth: f32) {
        let ceiling = (self.max_delay() - 1.0).max(0.0) * 0.5;
        self.mod_depth = mod_depth.clamp(0.0, ceiling);
    }

    /// The current nominal delay in samples.
    #[inline]
    pub fn base_delay(&self) -> f32 {
        self.base_delay
    }

    /// Set the modulation-LFO rate in Hz (phase-continuous).
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn set_lfo_rate(&mut self, rate_hz: f32, sample_rate: f32) {
        self.lfo.set_rate(rate_hz, sample_rate);
    }

    /// Clear the buffer and rewind the LFO to its decorrelated start phase.
    pub fn reset(&mut self) {
        self.buf.iter_mut().for_each(|sample| *sample = 0.0);
        self.write_idx = 0;
        self.lfo.reset();
    }

    /// Advance the LFO and return the interpolated sample at the modulated read
    /// position.  Call **once per sample, before** [`write`](ModulatedDelay::write).
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn read(&mut self) -> f32 {
        let len = self.buf.len();
        let lfo_value = self.lfo.tick(); // [-1, 1]
        let delay = (self.base_delay + self.mod_depth * lfo_value).clamp(1.0, self.max_delay());

        // Read position measured backwards from the write cursor, wrapped into
        // [0, len) with rem_euclid so a negative intermediate is handled.
        let read_pos = (self.write_idx as f32 - delay).rem_euclid(len as f32);
        let index0 = read_pos.floor() as usize;
        let frac = read_pos - index0 as f32;
        let index1 = if index0 + 1 == len { 0 } else { index0 + 1 };

        // Linear interpolation between the two neighbouring stored samples.
        let sample0 = self.buf[index0];
        let sample1 = self.buf[index1];
        sample0 + (sample1 - sample0) * frac
    }

    /// Store `input` at the write cursor and advance it.  Call **once per
    /// sample, after** [`read`](ModulatedDelay::read).
    ///
    /// **Real-time safe** — the stored value is flushed to zero if subnormal so
    /// the feedback tail never drags the FPU into the denormal slow path.
    #[inline]
    pub fn write(&mut self, input: f32) {
        self.buf[self.write_idx] = flush_to_zero(input);
        self.write_idx += 1;
        if self.write_idx == self.buf.len() {
            self.write_idx = 0;
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    /// A stationary (no modulation) line delays an impulse by exactly its
    /// integer base delay.
    #[test]
    fn integer_delay_is_exact() {
        let lfo = SineLfo::new(0.0, SR, 0.0); // no motion
        let mut delay = ModulatedDelay::new(64, 10.0, 0.0, lfo);

        // Feed an impulse, then zeros; the impulse must reappear 10 samples on.
        let mut output = Vec::new();
        for sample_index in 0..30 {
            let value = delay.read();
            output.push(value);
            delay.write(if sample_index == 0 { 1.0 } else { 0.0 });
        }
        // read() happens before write(), so input at n reappears at n+delay.
        assert!(
            (output[10] - 1.0).abs() < 1e-6,
            "impulse not at tap 10: {output:?}"
        );
        assert!(output[9].abs() < 1e-6);
        assert!(output[11].abs() < 1e-6);
    }

    /// A half-sample fractional delay interpolates between neighbours.
    #[test]
    fn fractional_delay_interpolates() {
        let lfo = SineLfo::new(0.0, SR, 0.0);
        let mut delay = ModulatedDelay::new(64, 5.5, 0.0, lfo);

        // Write a step of 1.0 from n=0 onward; once the line fills, the 5.5-sample
        // read sits exactly between two 1.0 samples → 1.0.
        for _ in 0..40 {
            delay.read();
            delay.write(1.0);
        }
        let steady = delay.read();
        assert!(
            (steady - 1.0).abs() < 1e-6,
            "steady fractional read wrong: {steady}"
        );
    }

    /// Modulation makes the effective delay wander but never blows up or reads
    /// out of bounds (the clamp keeps the fractional read inside the buffer).
    #[test]
    fn modulation_stays_bounded() {
        let lfo = SineLfo::new(2.0, SR, 0.0);
        let mut delay = ModulatedDelay::new(256, 100.0, 20.0, lfo);
        for n in 0..96_000 {
            let value = delay.read();
            assert!(value.is_finite(), "non-finite delay output at {n}");
            delay.write((n as f32 * 0.001).sin());
        }
    }

    /// `set_mod_depth` is clamped so a huge request cannot push the read out of
    /// the buffer.
    #[test]
    fn mod_depth_is_clamped_to_buffer() {
        let lfo = SineLfo::new(1.0, SR, 0.0);
        let mut delay = ModulatedDelay::new(64, 30.0, 0.0, lfo);
        delay.set_mod_depth(10_000.0); // absurd
        // Still readable without panic / non-finite.
        assert!(delay.read().is_finite());
    }

    /// `reset` clears history: an impulse written before reset must not reappear.
    #[test]
    fn reset_clears_history() {
        let lfo = SineLfo::new(0.0, SR, 0.0);
        let mut delay = ModulatedDelay::new(64, 10.0, 0.0, lfo);
        delay.read();
        delay.write(1.0);
        delay.reset();
        for _ in 0..20 {
            let value = delay.read();
            assert!(value.abs() < 1e-6, "history leaked past reset: {value}");
            delay.write(0.0);
        }
    }

    /// Capacity is rounded up to a power of two.
    #[test]
    fn capacity_is_power_of_two() {
        let lfo = SineLfo::new(0.0, SR, 0.0);
        let delay = ModulatedDelay::new(100, 10.0, 0.0, lfo);
        assert_eq!(delay.capacity(), 128);
    }
}
