//! One-pole exponential parameter smoother.
//!
//! # Why smooth at all?
//!
//! Host automation and UI knobs deliver a control value **once per block**, as
//! a step.  Applying that step directly to a per-sample gain, delay length or
//! filter cutoff produces *zipper noise* — an audible staircase, because the
//! coefficient jumps discontinuously between blocks.  A smoother turns each
//! step into a short exponential glide so the change is inaudible.
//!
//! # The one-pole model
//!
//! On every sample we move the current value a fixed fraction of the remaining
//! distance to the target:
//!
//! ```text
//! current += (target − current) · coeff
//! ```
//!
//! This is a first-order low-pass on the control signal.  Choosing
//!
//! ```text
//! coeff = 1 − exp(−1 / (τ · fs))
//! ```
//!
//! makes `τ` the time constant: after `τ` seconds the value has covered
//! `1 − 1/e ≈ 63 %` of a step.  A smoothing *time* of a few milliseconds is
//! enough to kill zipper noise on gains; longer (50–200 ms) suits delay-length
//! / size glides.
//!
//! # Why local to `rt-fdn-core`
//!
//! `mixer.rs` in the root crate computes equal-power gains *set-then-apply*
//! (no per-sample ramp) and `crossfade.rs` blends only for IR hot-swap —
//! neither is a reusable per-sample smoother, so there was nothing to extract
//! (see `docs/Roadmap-05-fdn-reverb.md`, Prerequisites).  This lives with the
//! engine that needs it; promote to `src/utils/` only if a second plugin does.
//!
//! # Real-time safety
//!
//! [`Smoother::tick`] is a single subtract, multiply and add — no allocation,
//! no branching hot-path beyond the settle snap, no locking.  Safe to call
//! from the audio render thread.  Construction and `set_time` call `exp` once,
//! so do them off the hot path when practical (they are still allocation-free).

/// A first-order (one-pole) exponential smoother for a single scalar control.
#[derive(Debug, Clone, Copy)]
pub struct Smoother {
    /// The value returned by the most recent [`tick`](Smoother::tick).
    current: f32,
    /// The value `current` is gliding toward.
    target: f32,
    /// Per-sample fraction of the remaining distance to close, in `(0, 1]`.
    /// `1.0` means "jump immediately" (used when the smoothing time is zero).
    coeff: f32,
}

impl Smoother {
    /// Create a smoother parked at `initial` with **instant** response
    /// (`coeff = 1`).  Call [`set_time`](Smoother::set_time) to give it a glide.
    pub fn new(initial: f32) -> Self {
        Self {
            current: initial,
            target: initial,
            coeff: 1.0,
        }
    }

    /// Create a smoother parked at `initial` with a smoothing time constant of
    /// `time_ms` milliseconds at `sample_rate` Hz.
    pub fn with_time(initial: f32, time_ms: f32, sample_rate: f32) -> Self {
        let mut smoother = Self::new(initial);
        smoother.set_time(time_ms, sample_rate);
        smoother
    }

    /// Set the smoothing time constant to `time_ms` at `sample_rate`.
    ///
    /// A non-positive time (or non-positive sample rate) yields an instant
    /// smoother (`coeff = 1`).  **Does not** disturb `current` or `target`.
    pub fn set_time(&mut self, time_ms: f32, sample_rate: f32) {
        let tau = time_ms * 1.0e-3;
        self.coeff = if tau > 0.0 && sample_rate > 0.0 {
            // coeff = 1 − exp(−1 / (τ·fs)); clamped into (0, 1] for safety.
            (1.0 - (-1.0 / (tau * sample_rate)).exp()).clamp(f32::MIN_POSITIVE, 1.0)
        } else {
            1.0
        };
    }

    /// Aim the smoother at a new `target`; the glide starts on the next
    /// [`tick`](Smoother::tick).
    #[inline]
    pub fn set_target(&mut self, target: f32) {
        self.target = target;
    }

    /// Jump both `current` and `target` to `value` immediately (no glide).
    ///
    /// Use on a hard reset / preset load where a discontinuity is acceptable
    /// (e.g. transport stopped) so the value is correct on the very first
    /// sample.
    #[inline]
    pub fn set_immediate(&mut self, value: f32) {
        self.current = value;
        self.target = value;
    }

    /// The value the smoother is gliding toward.
    #[inline]
    pub fn target(&self) -> f32 {
        self.target
    }

    /// The value most recently produced by [`tick`](Smoother::tick).
    #[inline]
    pub fn current(&self) -> f32 {
        self.current
    }

    /// `true` once `current` has reached `target` exactly.
    ///
    /// [`tick`](Smoother::tick) snaps onto the target the moment the `f32` step
    /// underflows, so this is a clean equality rather than a tolerance check.
    #[inline]
    pub fn is_settled(&self) -> bool {
        self.current == self.target
    }

    /// Advance one sample and return the new smoothed value.
    ///
    /// **Real-time safe** — subtract, compare, multiply-add.
    #[inline]
    pub fn tick(&mut self) -> f32 {
        if self.coeff >= 1.0 || self.current == self.target {
            // Instant response, or already there.
            self.current = self.target;
            return self.current;
        }
        let stepped = self.current + (self.target - self.current) * self.coeff;
        // When `f32` precision means the exponential step no longer moves the
        // value, it has stalled a few ULPs short — snap exactly onto the target
        // so the glide terminates and `is_settled` can report cleanly.
        self.current = if stepped == self.current {
            self.target
        } else {
            stepped
        };
        self.current
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    /// A fresh instant smoother returns its initial value unchanged.
    #[test]
    fn instant_smoother_holds_value() {
        let mut smoother = Smoother::new(0.5);
        assert_eq!(smoother.tick(), 0.5);
        assert_eq!(smoother.tick(), 0.5);
    }

    /// With `coeff = 1` a retargeted smoother reaches the target in one sample.
    #[test]
    fn instant_smoother_jumps_in_one_sample() {
        let mut smoother = Smoother::new(0.0);
        smoother.set_target(1.0);
        assert_eq!(smoother.tick(), 1.0);
    }

    /// A timed smoother must approach — not overshoot — the target monotonically.
    #[test]
    fn timed_smoother_approaches_monotonically() {
        let mut smoother = Smoother::with_time(0.0, 10.0, SR);
        smoother.set_target(1.0);
        let mut prev = 0.0;
        for _ in 0..48_000 {
            let value = smoother.tick();
            assert!(value >= prev, "smoother went backwards: {value} < {prev}");
            assert!(value <= 1.0 + 1e-6, "smoother overshot: {value}");
            prev = value;
        }
        assert!(smoother.is_settled(), "smoother never settled");
    }

    /// After roughly one time constant the value should have covered ~63 %.
    #[test]
    fn one_time_constant_reaches_63_percent() {
        let time_ms = 10.0;
        let mut smoother = Smoother::with_time(0.0, time_ms, SR);
        smoother.set_target(1.0);
        // Advance exactly τ seconds worth of samples.
        let samples = (time_ms * 1.0e-3 * SR).round() as usize;
        let mut value = 0.0;
        for _ in 0..samples {
            value = smoother.tick();
        }
        assert!(
            (value - 0.632).abs() < 0.02,
            "expected ≈0.632 after one τ, got {value}"
        );
    }

    /// `set_immediate` wins over any in-flight glide.
    #[test]
    fn set_immediate_overrides_glide() {
        let mut smoother = Smoother::with_time(0.0, 100.0, SR);
        smoother.set_target(1.0);
        smoother.tick();
        smoother.set_immediate(-0.25);
        assert_eq!(smoother.current(), -0.25);
        assert_eq!(smoother.tick(), -0.25);
        assert!(smoother.is_settled());
    }

    /// A zero smoothing time degrades gracefully to instant response.
    #[test]
    fn zero_time_is_instant() {
        let mut smoother = Smoother::with_time(0.0, 0.0, SR);
        smoother.set_target(2.0);
        assert_eq!(smoother.tick(), 2.0);
    }

    /// The smoother stays finite and settled across a burst of retargets.
    #[test]
    fn survives_rapid_retargets() {
        let mut smoother = Smoother::with_time(0.0, 5.0, SR);
        for step in 0..1000 {
            smoother.set_target(if step % 2 == 0 { 1.0 } else { -1.0 });
            for _ in 0..10 {
                assert!(smoother.tick().is_finite());
            }
        }
    }
}
