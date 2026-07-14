//! The [`Shifter`] dispatch boundary and shared pitch-ratio plumbing.
//!
//! `rt-pitch` ships three interchangeable pitch-shift algorithms (phase
//! vocoder, WSOLA, PSOLA).  They differ wildly internally — frequency vs time
//! domain, different latency, different buffers — but the host only ever needs
//! three operations, captured by the [`Shifter`] trait.  The concrete engine
//! is chosen **off** the audio thread (a `Box<dyn Shifter>` swapped at
//! preset / parameter change), so the per-call virtual dispatch is negligible
//! and the render thread never allocates or branches on algorithm identity.

use std::sync::atomic::{AtomicU32, Ordering};

/// Common interface implemented by every pitch-shift algorithm.
///
/// # Real-time contract
///
/// [`process_block`](Shifter::process_block) and [`set_ratio`](Shifter::set_ratio)
/// run on the audio thread and must not allocate, lock, or panic.  All storage
/// is reserved when the concrete shifter is constructed (off-RT).
pub trait Shifter {
    /// Algorithmic latency in samples the shifter introduces between input and
    /// output.  The host reports this via `nih_plug::latency_samples()` so
    /// other tracks can be delay-compensated.  Constant for a given
    /// construction / FFT size.
    fn latency_samples(&self) -> usize;

    /// Flush all internal state (ring buffers, phase accumulators, overlap-add
    /// tails) back to silence — called on transport stop / preset load.
    fn reset(&mut self);

    /// Update the target pitch ratio (linear frequency multiplier, `1.0` =
    /// unison).  Cheap; safe to call every block from the audio thread.  The
    /// shifter is free to smooth toward the target internally to avoid zipper
    /// noise.
    fn set_ratio(&mut self, ratio: f32);

    /// Render one block.  `input` and `output` must be the same length; the
    /// shifter writes exactly `output.len()` samples, emitting silence during
    /// the initial `latency_samples()` warm-up.
    fn process_block(&mut self, input: &[f32], output: &mut [f32]);
}

/// Convert an interval in (fractional) semitones to a linear frequency ratio.
///
/// `2^(semitones / 12)` — e.g. `+12` → `2.0` (octave up), `−12` → `0.5`.
#[inline]
pub fn semitones_to_ratio(semitones: f32) -> f32 {
    (semitones / 12.0).exp2()
}

/// Inverse of [`semitones_to_ratio`]: a linear ratio back to semitones.
///
/// # Panics
/// Never panics, but returns `NaN`/`-inf` for non-positive ratios; callers
/// pass strictly positive ratios.
#[inline]
pub fn ratio_to_semitones(ratio: f32) -> f32 {
    12.0 * ratio.log2()
}

/// Smallest / largest ratios the shifters accept (± 2 octaves), matching the
/// plugin's `−24 … +24` semitone control range.
pub const MIN_RATIO: f32 = 0.25;
/// See [`MIN_RATIO`].
pub const MAX_RATIO: f32 = 4.0;

/// Clamp an arbitrary ratio into the supported `[MIN_RATIO, MAX_RATIO]` band.
#[inline]
pub fn clamp_ratio(ratio: f32) -> f32 {
    ratio.clamp(MIN_RATIO, MAX_RATIO)
}

/// Lock-free `f32` cell for passing a pitch ratio from the GUI/host thread to
/// the audio thread.
///
/// The plugin (Phase 2) updates this from a parameter callback while the
/// render thread loads it once per block — a single `f32` fits in one atomic
/// word, so no lock is needed.  Stored as the raw `u32` bit pattern because
/// `f32` has no native atomic.
#[derive(Debug)]
pub struct SharedRatio {
    bits: AtomicU32,
}

impl SharedRatio {
    /// Create a shared ratio initialised to `ratio`.
    pub fn new(ratio: f32) -> Self {
        Self {
            bits: AtomicU32::new(ratio.to_bits()),
        }
    }

    /// Store a new target ratio (GUI/host thread).  `Relaxed` is sufficient:
    /// the value carries no dependent data, only the ratio itself matters.
    #[inline]
    pub fn store(&self, ratio: f32) {
        self.bits.store(ratio.to_bits(), Ordering::Relaxed);
    }

    /// Load the current target ratio (audio thread).
    #[inline]
    pub fn load(&self) -> f32 {
        f32::from_bits(self.bits.load(Ordering::Relaxed))
    }
}

impl Default for SharedRatio {
    fn default() -> Self {
        Self::new(1.0)
    }
}

/// One-pole smoother that ramps a control value toward its target each sample,
/// removing the zipper noise a stepped ratio would inject.
///
/// Not tied to [`Shifter`]; shared by the shifters that update pitch per hop.
#[derive(Clone, Copy, Debug)]
pub struct OnePoleSmoother {
    current: f32,
    target: f32,
    /// Per-step blend coefficient in `(0, 1]`; larger = faster tracking.
    coeff: f32,
}

impl OnePoleSmoother {
    /// Build a smoother starting at `initial`, reaching ~63 % of a step in
    /// `time_constant_steps` steps.
    pub fn new(initial: f32, time_constant_steps: f32) -> Self {
        let coeff = if time_constant_steps <= 0.0 {
            1.0
        } else {
            1.0 - (-1.0 / time_constant_steps).exp()
        };
        Self {
            current: initial,
            target: initial,
            coeff,
        }
    }

    /// Set the value the smoother ramps toward.
    #[inline]
    pub fn set_target(&mut self, target: f32) {
        self.target = target;
    }

    /// Jump immediately to `value` (used on reset / preset load).
    #[inline]
    pub fn snap_to(&mut self, value: f32) {
        self.current = value;
        self.target = value;
    }

    /// Advance one step and return the new smoothed value.
    #[inline]
    pub fn tick(&mut self) -> f32 {
        self.current += self.coeff * (self.target - self.current);
        self.current
    }

    /// Current value without advancing.
    #[inline]
    pub fn value(&self) -> f32 {
        self.current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semitone_octave_is_double() {
        assert!((semitones_to_ratio(12.0) - 2.0).abs() < 1e-5);
        assert!((semitones_to_ratio(-12.0) - 0.5).abs() < 1e-5);
        assert!((semitones_to_ratio(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn semitone_roundtrip() {
        for semi in [-24.0, -7.0, 0.0, 3.5, 12.0, 19.0] {
            let back = ratio_to_semitones(semitones_to_ratio(semi));
            assert!((back - semi).abs() < 1e-3, "semi={semi} back={back}");
        }
    }

    #[test]
    fn ratio_clamps_to_two_octaves() {
        assert_eq!(clamp_ratio(10.0), MAX_RATIO);
        assert_eq!(clamp_ratio(0.01), MIN_RATIO);
        assert_eq!(clamp_ratio(1.5), 1.5);
    }

    #[test]
    fn shared_ratio_roundtrips_across_threads() {
        let shared = SharedRatio::new(1.0);
        shared.store(1.5);
        assert!((shared.load() - 1.5).abs() < 1e-6);
    }

    #[test]
    fn smoother_converges_to_target() {
        let mut smoother = OnePoleSmoother::new(1.0, 16.0);
        smoother.set_target(2.0);
        for _ in 0..512 {
            smoother.tick();
        }
        assert!((smoother.value() - 2.0).abs() < 1e-3);
    }

    #[test]
    fn smoother_snap_is_immediate() {
        let mut smoother = OnePoleSmoother::new(1.0, 16.0);
        smoother.snap_to(0.5);
        assert_eq!(smoother.value(), 0.5);
        assert_eq!(smoother.tick(), 0.5);
    }
}
