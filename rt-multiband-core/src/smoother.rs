//! A one-pole parameter smoother — the "keep local" Phase 0 prerequisite
//! decision.
//!
//! Crossover frequencies, thresholds and makeup gains are stepped by the host
//! on block boundaries; feeding those steps straight into the DSP would click.
//! [`ParamSmoother`] ramps the current value toward its target with a one-pole
//! filter whose time constant is set once (off-RT) from a millisecond glide
//! time:
//!
//! ```text
//! coeff       = exp(−1 / (glide_seconds · fs))
//! current[n]  = target + coeff · (current[n−1] − target)
//! ```
//!
//! Two `f32` of state per parameter, one multiply-add per sample — RT-safe.
//! (nih-plug's own `Smoother` still handles wrapper-level parameter smoothing
//! in Phase 2; this is the core-side ramp for values that feed coefficient
//! math.)

/// A one-pole smoother that glides a value toward a target.
#[derive(Debug, Clone, Copy)]
pub struct ParamSmoother {
    current: f32,
    target: f32,
    coeff: f32,
}

impl ParamSmoother {
    /// Build a smoother initialised (and targeted) at `initial`, gliding over
    /// `glide_ms` at `sample_rate`.
    pub fn new(sample_rate: f32, glide_ms: f32, initial: f32) -> Self {
        let mut smoother = Self {
            current: initial,
            target: initial,
            coeff: 0.0,
        };
        smoother.set_glide_ms(sample_rate, glide_ms);
        smoother
    }

    /// Recompute the glide time constant.  RT-safe.
    pub fn set_glide_ms(&mut self, sample_rate: f32, glide_ms: f32) {
        let seconds = (glide_ms * 1e-3).max(1e-6);
        self.coeff = (-1.0 / (seconds * sample_rate)).exp();
    }

    /// Set the value the smoother glides toward.  RT-safe.
    #[inline]
    pub fn set_target(&mut self, target: f32) {
        self.target = target;
    }

    /// Jump immediately to `value` with no glide (e.g. on `reset`).
    #[inline]
    pub fn set_immediate(&mut self, value: f32) {
        self.current = value;
        self.target = value;
    }

    /// The current smoothed value without advancing.
    #[inline]
    pub fn current(&self) -> f32 {
        self.current
    }

    /// Whether the smoother has effectively reached its target.
    #[inline]
    pub fn is_settled(&self) -> bool {
        (self.current - self.target).abs() < 1e-6
    }

    /// Advance one sample and return the new smoothed value.  RT-safe.
    #[inline]
    pub fn next_value(&mut self) -> f32 {
        self.current = self.target + self.coeff * (self.current - self.target);
        self.current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    #[test]
    fn starts_at_initial_value() {
        let mut smoother = ParamSmoother::new(SR, 10.0, 0.5);
        assert_eq!(smoother.current(), 0.5);
        assert_eq!(smoother.next_value(), 0.5);
    }

    #[test]
    fn glides_toward_target_monotonically() {
        let mut smoother = ParamSmoother::new(SR, 5.0, 0.0);
        smoother.set_target(1.0);
        let mut previous = smoother.current();
        for _ in 0..4_800 {
            let value = smoother.next_value();
            assert!(
                value >= previous - 1e-6,
                "not monotonic: {value} < {previous}"
            );
            assert!(value <= 1.0 + 1e-6);
            previous = value;
        }
        assert!((smoother.current() - 1.0).abs() < 1e-2);
    }

    #[test]
    fn reaches_within_time_constant() {
        // After ~5 time constants the one-pole is within <1% of target.
        let glide_ms = 10.0;
        let mut smoother = ParamSmoother::new(SR, glide_ms, 0.0);
        smoother.set_target(1.0);
        let samples = (5.0 * glide_ms * 1e-3 * SR) as usize;
        for _ in 0..samples {
            smoother.next_value();
        }
        assert!(
            smoother.current() > 0.99,
            "only reached {}",
            smoother.current()
        );
    }

    #[test]
    fn set_immediate_skips_the_glide() {
        let mut smoother = ParamSmoother::new(SR, 100.0, 0.0);
        smoother.set_immediate(0.75);
        assert_eq!(smoother.next_value(), 0.75);
        assert!(smoother.is_settled());
    }
}
