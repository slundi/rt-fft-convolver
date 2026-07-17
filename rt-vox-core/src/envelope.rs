//! Attack/release envelope follower for the vocoder's per-band modulator level.
//!
//! A vocoder tracks how loud the modulator (voice) is in each frequency band
//! and uses that level to shape the carrier.  A raw per-frame magnitude jumps
//! around; an envelope follower smooths it with separate attack and release
//! time constants — fast attack to catch consonants, slower release so bands do
//! not chatter.
//!
//! The follower is updated **once per STFT frame**, so its effective sample
//! rate is `sample_rate / hop`.  Coefficients are derived from that frame rate.

/// One-pole attack/release envelope follower.
///
/// `Clone`/`Copy` so a `Vec<EnvelopeFollower>` (one per band) can be built and
/// reset cheaply.
#[derive(Clone, Copy, Debug)]
pub struct EnvelopeFollower {
    /// Per-frame blend toward a *rising* target, in `(0, 1]`.
    attack: f32,
    /// Per-frame blend toward a *falling* target, in `(0, 1]`.
    release: f32,
    /// Current envelope value.
    state: f32,
}

impl EnvelopeFollower {
    /// Build a follower with `attack_ms` / `release_ms` time constants at the
    /// given *frame rate* (`sample_rate / hop`).
    ///
    /// A time of `0` (or non-positive) means "instant" (coefficient `1.0`),
    /// which passes the input straight through in that direction.
    pub fn new(attack_ms: f32, release_ms: f32, frame_rate: f32) -> Self {
        Self {
            attack: Self::coeff(attack_ms, frame_rate),
            release: Self::coeff(release_ms, frame_rate),
            state: 0.0,
        }
    }

    /// One-pole coefficient reaching ~63 % of a step in `time_ms`
    /// milliseconds: `1 − e^(−1 / (time_ms · frame_rate / 1000))`.
    fn coeff(time_ms: f32, frame_rate: f32) -> f32 {
        let steps = time_ms * 0.001 * frame_rate;
        if steps <= 0.0 {
            1.0
        } else {
            1.0 - (-1.0 / steps).exp()
        }
    }

    /// Update the attack/release times in place (off-RT parameter change).
    pub fn set_times(&mut self, attack_ms: f32, release_ms: f32, frame_rate: f32) {
        self.attack = Self::coeff(attack_ms, frame_rate);
        self.release = Self::coeff(release_ms, frame_rate);
    }

    /// Feed the next raw band level and return the smoothed envelope.
    ///
    /// Rising inputs use the attack coefficient, falling inputs the release
    /// coefficient — the classic asymmetric follower.
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        let coeff = if input > self.state {
            self.attack
        } else {
            self.release
        };
        self.state += coeff * (input - self.state);
        self.state
    }

    /// Current envelope value without advancing.
    #[inline]
    pub fn value(&self) -> f32 {
        self.state
    }

    /// Reset the envelope back to silence.
    #[inline]
    pub fn reset(&mut self) {
        self.state = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME_RATE: f32 = 48_000.0 / 256.0; // ~187.5 frames/s

    #[test]
    fn rises_toward_a_held_input() {
        let mut follower = EnvelopeFollower::new(5.0, 50.0, FRAME_RATE);
        let mut last = 0.0;
        for _ in 0..200 {
            last = follower.process(1.0);
        }
        assert!((last - 1.0).abs() < 1e-2, "did not reach target: {last}");
    }

    #[test]
    fn attack_is_faster_than_release() {
        // From silence, a step up should climb faster than an equal step down
        // settles — verify by counting frames to cross the halfway point.
        let frame_rate = FRAME_RATE;
        let mut up = EnvelopeFollower::new(5.0, 200.0, frame_rate);
        let mut frames_to_half_up = 0;
        while up.process(1.0) < 0.5 {
            frames_to_half_up += 1;
        }

        let mut down = EnvelopeFollower::new(5.0, 200.0, frame_rate);
        for _ in 0..1000 {
            down.process(1.0); // settle at 1.0
        }
        let mut frames_to_half_down = 0;
        while down.process(0.0) > 0.5 {
            frames_to_half_down += 1;
        }

        assert!(
            frames_to_half_up < frames_to_half_down,
            "attack {frames_to_half_up} should beat release {frames_to_half_down}"
        );
    }

    #[test]
    fn zero_attack_is_instant() {
        let mut follower = EnvelopeFollower::new(0.0, 100.0, FRAME_RATE);
        assert!((follower.process(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn reset_returns_to_zero() {
        let mut follower = EnvelopeFollower::new(5.0, 50.0, FRAME_RATE);
        follower.process(1.0);
        follower.reset();
        assert_eq!(follower.value(), 0.0);
    }
}
