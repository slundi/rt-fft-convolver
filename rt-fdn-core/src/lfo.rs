//! A cheap sine LFO used to modulate the FDN delay-line read positions.
//!
//! # Why modulate the delays?
//!
//! A static Feedback Delay Network rings on a fixed set of modal frequencies;
//! sustained input makes those modes audible as a metallic, "static" tail.
//! Slowly wandering each delay length de-tunes the modes continuously, which
//! is exactly the lush, chorus-y motion of classic algorithmic reverbs
//! (Strymon / Valhalla).  Phase 0 chose **per-delay LFOs with decorrelated
//! phases** over a shared random-walk driver — see
//! `docs/Roadmap-05-fdn-reverb.md`.
//!
//! # Implementation
//!
//! A normalised phase accumulator in `[0, 1)` advanced by `rate / fs` each
//! sample, evaluated through `sin(2π·phase)`.  One `sin` per line per sample is
//! negligible next to the fractional-delay reads and the matrix mix, and keeps
//! the waveform exact (no wavetable interpolation error).
//!
//! # Real-time safety
//!
//! [`SineLfo::tick`] is one `sin`, one add and one compare — no allocation,
//! no locking.

use std::f32::consts::TAU;

/// A unit-amplitude sine low-frequency oscillator.
///
/// Output is in `[-1, 1]`.  The starting phase is remembered so
/// [`reset`](SineLfo::reset) returns to it — this is how the FDN keeps its `N`
/// LFOs decorrelated after a transport reset.
#[derive(Debug, Clone, Copy)]
pub struct SineLfo {
    /// Current phase in `[0, 1)` (turns, not radians).
    phase: f32,
    /// Phase to return to on [`reset`](SineLfo::reset).
    initial_phase: f32,
    /// Per-sample phase increment = `rate_hz / sample_rate`.
    increment: f32,
}

impl SineLfo {
    /// Create an LFO at `rate_hz`, starting at `initial_phase` turns
    /// (wrapped into `[0, 1)`).
    pub fn new(rate_hz: f32, sample_rate: f32, initial_phase: f32) -> Self {
        let wrapped = initial_phase.rem_euclid(1.0);
        let mut lfo = Self {
            phase: wrapped,
            initial_phase: wrapped,
            increment: 0.0,
        };
        lfo.set_rate(rate_hz, sample_rate);
        lfo
    }

    /// Update the oscillation rate in Hz without disturbing the current phase
    /// (so a live rate change does not jump the waveform).
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn set_rate(&mut self, rate_hz: f32, sample_rate: f32) {
        self.increment = if sample_rate > 0.0 {
            (rate_hz / sample_rate).max(0.0)
        } else {
            0.0
        };
    }

    /// Return to the starting phase (used on a hard reset so the `N` LFOs stay
    /// decorrelated and no delay jumps discontinuously — see the Phase 4
    /// "modulation reset does not click" acceptance test).
    #[inline]
    pub fn reset(&mut self) {
        self.phase = self.initial_phase;
    }

    /// Advance one sample and return the sine value in `[-1, 1]`.
    #[inline]
    pub fn tick(&mut self) -> f32 {
        let value = (self.phase * TAU).sin();
        self.phase += self.increment;
        // Wrap once; the increment is far below 1.0 for any sane LFO rate, so a
        // single subtraction always suffices (no loop needed).
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }
        value
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    /// The output must stay within `[-1, 1]` for a long run.
    #[test]
    fn output_is_bounded() {
        let mut lfo = SineLfo::new(2.0, SR, 0.0);
        for _ in 0..96_000 {
            let value = lfo.tick();
            assert!((-1.0..=1.0).contains(&value), "LFO out of range: {value}");
        }
    }

    /// A phase of 0 starts at 0 and rises (sin near 0 is increasing).
    #[test]
    fn starts_at_zero_and_rises() {
        let mut lfo = SineLfo::new(1.0, SR, 0.0);
        let first = lfo.tick();
        let second = lfo.tick();
        assert!(first.abs() < 1e-6, "expected start ≈0, got {first}");
        assert!(second > first, "LFO should be rising from phase 0");
    }

    /// One full period at `rate` Hz should take `fs / rate` samples: after that
    /// many samples we should be back near the starting value.
    #[test]
    fn completes_one_period() {
        let rate = 5.0;
        let mut lfo = SineLfo::new(rate, SR, 0.0);
        let samples = (SR / rate).round() as usize;
        for _ in 0..samples {
            lfo.tick();
        }
        assert!(
            lfo.tick().abs() < 1e-3,
            "did not return near start after a period"
        );
    }

    /// `reset` returns the oscillator to its construction phase.
    #[test]
    fn reset_restores_initial_phase() {
        let mut lfo = SineLfo::new(3.0, SR, 0.25);
        let baseline = lfo.tick();
        for _ in 0..1234 {
            lfo.tick();
        }
        lfo.reset();
        assert!(
            (lfo.tick() - baseline).abs() < 1e-6,
            "reset did not restore the initial phase"
        );
    }

    /// Decorrelated phases yield different instantaneous values.
    #[test]
    fn decorrelated_phases_differ() {
        let mut a = SineLfo::new(1.0, SR, 0.0);
        let mut b = SineLfo::new(1.0, SR, 0.25);
        assert!((a.tick() - b.tick()).abs() > 0.1);
    }

    /// A zero sample rate degrades to a stationary (DC) oscillator, not a NaN.
    #[test]
    fn zero_sample_rate_is_stationary() {
        let mut lfo = SineLfo::new(1.0, 0.0, 0.1);
        let first = lfo.tick();
        let second = lfo.tick();
        assert_eq!(first, second);
        assert!(first.is_finite());
    }
}
