//! Analog-modelled envelope follower (Phase 0 decision).
//!
//! A **branching one-pole** with separate attack and release capacitor-charge
//! coefficients — the standard, cheap, well-understood dynamics follower:
//!
//! ```text
//! coeff = exp(−1 / (t · fs))            t = attack when rising, release when falling
//! env[n] = level + coeff · (env[n−1] − level)
//! ```
//!
//! The coefficient is the fraction of the old envelope retained per sample, so
//! `exp(−1 / (t·fs))` gives a time constant of `t` seconds (the envelope
//! reaches `1 − 1/e ≈ 63 %` of a step in `t`).  One branch and one multiply-add
//! per sample — RT-safe.  Both coefficients are recomputed off-RT on any
//! attack/release change.

/// A branching-one-pole attack/release envelope follower.
#[derive(Debug, Clone, Copy)]
pub struct EnvelopeFollower {
    sample_rate: f32,
    attack_coeff: f32,
    release_coeff: f32,
    env: f32,
}

impl EnvelopeFollower {
    /// Build a follower at `sample_rate` with the given attack and release
    /// times in milliseconds.
    pub fn new(sample_rate: f32, attack_ms: f32, release_ms: f32) -> Self {
        let mut follower = Self {
            sample_rate,
            attack_coeff: 0.0,
            release_coeff: 0.0,
            env: 0.0,
        };
        follower.set_attack_ms(attack_ms);
        follower.set_release_ms(release_ms);
        follower
    }

    /// One-pole coefficient for a time constant of `time_ms` at this rate.
    #[inline]
    fn coeff_for(&self, time_ms: f32) -> f32 {
        let seconds = (time_ms * 1e-3).max(1e-6);
        (-1.0 / (seconds * self.sample_rate)).exp()
    }

    /// Set the attack time in milliseconds.  RT-safe.
    pub fn set_attack_ms(&mut self, attack_ms: f32) {
        self.attack_coeff = self.coeff_for(attack_ms);
    }

    /// Set the release time in milliseconds.  RT-safe.
    pub fn set_release_ms(&mut self, release_ms: f32) {
        self.release_coeff = self.coeff_for(release_ms);
    }

    /// The current envelope value.
    #[inline]
    pub fn value(&self) -> f32 {
        self.env
    }

    /// Clear the envelope to silence.
    #[inline]
    pub fn reset(&mut self) {
        self.env = 0.0;
    }

    /// Feed a (non-negative) detector `level` and return the shaped envelope.
    ///
    /// Rising input charges at the attack rate, falling input decays at the
    /// release rate.  RT-safe.
    #[inline]
    pub fn process(&mut self, level: f32) -> f32 {
        let coeff = if level > self.env {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.env = level + coeff * (self.env - level);
        self.env
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    /// Samples the envelope takes to cross `1 − 1/e` of a unit step from rest.
    fn samples_to_time_constant(mut follower: EnvelopeFollower) -> usize {
        let target = 1.0 - std::f32::consts::E.recip();
        let mut count = 0;
        while follower.process(1.0) < target {
            count += 1;
            if count > SR as usize {
                break;
            }
        }
        count
    }

    #[test]
    fn attack_reaches_time_constant_within_ten_percent() {
        let attack_ms = 10.0;
        let follower = EnvelopeFollower::new(SR, attack_ms, 100.0);
        let samples = samples_to_time_constant(follower);
        let expected = attack_ms * 1e-3 * SR;
        let err = (samples as f32 - expected).abs() / expected;
        assert!(err < 0.1, "attack τ off by {}%", err * 100.0);
    }

    #[test]
    fn release_reaches_time_constant_within_ten_percent() {
        let release_ms = 50.0;
        let mut follower = EnvelopeFollower::new(SR, 0.01, release_ms);
        // Charge fully, then measure decay to 1/e.
        for _ in 0..SR as usize {
            follower.process(1.0);
        }
        let target = std::f32::consts::E.recip();
        let mut count = 0;
        while follower.process(0.0) > target {
            count += 1;
            if count > SR as usize {
                break;
            }
        }
        let expected = release_ms * 1e-3 * SR;
        let err = (count as f32 - expected).abs() / expected;
        assert!(err < 0.1, "release τ off by {}%", err * 100.0);
    }

    #[test]
    fn fast_attack_slow_release_is_asymmetric() {
        // A short burst charges quickly (0.1 ms attack ≈ 5 samples), then a
        // slow 200 ms release barely decays over the next few ms.
        let mut follower = EnvelopeFollower::new(SR, 0.1, 200.0);
        for _ in 0..50 {
            follower.process(1.0);
        }
        let charged = follower.value();
        assert!(charged > 0.99, "fast attack did not charge: {charged}");
        for _ in 0..100 {
            follower.process(0.0);
        }
        assert!(follower.value() > 0.95, "slow release decayed too fast");
    }
}
