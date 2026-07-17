//! A single harmoniser voice: one swappable pitch [`Shifter`] plus mix state.
//!
//! Each harmony note is one [`Voice`].  It owns a boxed [`Shifter`] (chosen off
//! the audio thread — [`PhaseVocoder`](rt_pitch_core::PhaseVocoder) by default,
//! which handles the chordal, formant-preserving case), a private mono scratch
//! buffer, and its mix parameters (level, pan, detune).  The [`Harmoniser`]
//! drives the interval; the voice turns it into a ratio, shifts, and reports its
//! panned contribution.
//!
//! [`Harmoniser`]: crate::harmoniser::Harmoniser

use rt_pitch_core::{Shifter, clamp_ratio, semitones_to_ratio};

/// One pitch-shifted harmony voice.
pub struct Voice {
    shifter: Box<dyn Shifter>,
    /// Linear output level `[0, 1+]`.
    level: f32,
    /// Stereo pan `[-1, 1]` (−1 = hard left, +1 = hard right).
    pan: f32,
    /// Fine detune in cents, added to the interval before shifting.
    detune_cents: f32,
    /// Private per-block mono output; sized to `max_block` at construction.
    scratch: Vec<f32>,
}

impl Voice {
    /// Wrap `shifter` as a voice able to render blocks up to `max_block`.
    ///
    /// **Allocates** — off-RT only.
    pub fn new(shifter: Box<dyn Shifter>, max_block: usize) -> Self {
        Self {
            shifter,
            level: 1.0,
            pan: 0.0,
            detune_cents: 0.0,
            scratch: vec![0.0; max_block],
        }
    }

    /// Output level (linear).
    #[inline]
    pub fn set_level(&mut self, level: f32) {
        self.level = level.max(0.0);
    }

    /// Stereo pan, clamped to `[-1, 1]`.
    #[inline]
    pub fn set_pan(&mut self, pan: f32) {
        self.pan = pan.clamp(-1.0, 1.0);
    }

    /// Fine detune in cents (±).
    #[inline]
    pub fn set_detune_cents(&mut self, cents: f32) {
        self.detune_cents = cents;
    }

    /// The shifter's algorithmic latency in samples.
    #[inline]
    pub fn latency_samples(&self) -> usize {
        self.shifter.latency_samples()
    }

    /// Set the voice's target interval in semitones (detune is folded in), then
    /// update the shifter's ratio.  Cheap; safe per block on the audio thread.
    #[inline]
    pub fn set_interval_semitones(&mut self, semitones: f32) {
        let total = semitones + self.detune_cents * 0.01;
        self.shifter
            .set_ratio(clamp_ratio(semitones_to_ratio(total)));
    }

    /// Constant-power left/right gains for the current pan.
    ///
    /// Maps `pan ∈ [-1, 1]` to an angle in `[0, π/2]`; `cos`/`sin` keep the
    /// summed power constant across the sweep.
    fn pan_gains(&self) -> (f32, f32) {
        use std::f32::consts::FRAC_PI_4;
        let angle = (self.pan + 1.0) * FRAC_PI_4; // 0..π/2
        (angle.cos() * self.level, angle.sin() * self.level)
    }

    /// Shift `input` and accumulate the panned result into `left` / `right`.
    ///
    /// All three slices share the input length (`<= max_block`).  Allocation-
    /// free: rendering goes through the preallocated scratch buffer.
    pub fn process_add(&mut self, input: &[f32], left: &mut [f32], right: &mut [f32]) {
        debug_assert_eq!(input.len(), left.len());
        debug_assert_eq!(input.len(), right.len());
        let len = input.len();
        let (gain_l, gain_r) = self.pan_gains();
        let scratch = &mut self.scratch[..len];
        self.shifter.process_block(input, scratch);

        for ((l, r), &s) in left.iter_mut().zip(right.iter_mut()).zip(scratch.iter()) {
            *l += s * gain_l;
            *r += s * gain_r;
        }
    }

    /// Flush the shifter's internal state.
    pub fn reset(&mut self) {
        self.shifter.reset();
        self.scratch.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::rms;
    use rt_pitch_core::{PhaseVocoder, window::WindowKind};

    const MAX_BLOCK: usize = 512;

    fn make_voice() -> Voice {
        let shifter = Box::new(PhaseVocoder::new(1024, MAX_BLOCK, WindowKind::Hann));
        Voice::new(shifter, MAX_BLOCK)
    }

    #[test]
    fn centre_pan_is_equal_on_both_sides() {
        let mut voice = make_voice();
        voice.set_interval_semitones(0.0);
        let input = vec![0.5f32; MAX_BLOCK];
        let mut left = vec![0.0f32; MAX_BLOCK];
        let mut right = vec![0.0f32; MAX_BLOCK];
        // Run several blocks past the shifter warm-up.
        for _ in 0..8 {
            left.fill(0.0);
            right.fill(0.0);
            voice.process_add(&input, &mut left, &mut right);
        }
        assert!(
            (rms(&left) - rms(&right)).abs() < 1e-4,
            "centre pan should be balanced: {} vs {}",
            rms(&left),
            rms(&right)
        );
    }

    #[test]
    fn hard_left_silences_right() {
        let mut voice = make_voice();
        voice.set_pan(-1.0);
        voice.set_interval_semitones(0.0);
        let input = vec![0.5f32; MAX_BLOCK];
        let mut left = vec![0.0f32; MAX_BLOCK];
        let mut right = vec![0.0f32; MAX_BLOCK];
        for _ in 0..8 {
            left.fill(0.0);
            right.fill(0.0);
            voice.process_add(&input, &mut left, &mut right);
        }
        assert!(
            rms(&right) < 1e-6,
            "right should be silent, got {}",
            rms(&right)
        );
    }

    #[test]
    fn zero_level_is_silent() {
        let mut voice = make_voice();
        voice.set_level(0.0);
        voice.set_interval_semitones(0.0);
        let input = vec![0.5f32; MAX_BLOCK];
        let mut left = vec![0.0f32; MAX_BLOCK];
        let mut right = vec![0.0f32; MAX_BLOCK];
        for _ in 0..8 {
            voice.process_add(&input, &mut left, &mut right);
        }
        assert!(rms(&left) < 1e-6 && rms(&right) < 1e-6);
    }
}
