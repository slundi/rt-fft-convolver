//! A 1-pole DC blocker (high-pass) for the post-shaper signal.
//!
//! # Why
//!
//! Asymmetric shapers ([`crate::shaper::Shaper::Tube`], a biased
//! [`crate::shaper::Shaper::Tape`], or an asymmetric hard-clip) push a DC
//! offset onto the output.  Left in place that offset wastes headroom, thumps
//! on bypass switching and can bias downstream stages.  A gentle high-pass at
//! ~20 Hz removes it while leaving the audible band untouched.
//!
//! # Transfer function
//!
//! The classic 1-pole DC blocker (Julius Smith, *Introduction to Digital
//! Filters*):
//!
//! ```text
//! y[n] = x[n] − x[n−1] + R · y[n−1]
//! ```
//!
//! The zero at `z = 1` forces the DC gain to zero; the pole at `z = R`
//! (with `R` just below 1) sets the corner.  For a small cutoff `fc`:
//!
//! ```text
//! R ≈ 1 − 2π · fc / fs
//! ```
//!
//! so the −3 dB point sits near `fc` and the passband gain approaches 1.

/// A 1-pole DC-blocking high-pass filter.
///
/// Construct with [`DcBlocker::new`] off the audio thread, then call
/// [`DcBlocker::process_sample`] (or [`DcBlocker::process_block`]) per sample
/// on the render thread.
#[derive(Debug, Clone, Copy)]
pub struct DcBlocker {
    /// Pole radius `R ∈ (0, 1)`.
    pole: f32,
    /// Previous input `x[n−1]`.
    prev_input: f32,
    /// Previous output `y[n−1]`.
    prev_output: f32,
}

impl DcBlocker {
    /// Create a DC blocker with a `cutoff_hz` corner at `sample_rate` Hz.
    ///
    /// `cutoff_hz` is clamped to a sane sub-Nyquist range so the pole stays
    /// strictly inside the unit circle (stable) for any caller input.
    ///
    /// **Allocates nothing**, but call it off the audio thread anyway — it is
    /// a `set`-style configuration point.
    pub fn new(cutoff_hz: f32, sample_rate: f32) -> Self {
        let mut blocker = Self {
            pole: 0.0,
            prev_input: 0.0,
            prev_output: 0.0,
        };
        blocker.set_cutoff(cutoff_hz, sample_rate);
        blocker
    }

    /// Recompute the pole for a new cutoff / sample rate without touching the
    /// filter state (so it can be retuned between blocks click-free).
    pub fn set_cutoff(&mut self, cutoff_hz: f32, sample_rate: f32) {
        // Keep fc within (0, fs/4] so R stays in a stable, useful range.
        let nyquist_quarter = sample_rate * 0.25;
        let cutoff = cutoff_hz.clamp(0.1, nyquist_quarter.max(0.1));
        let pole = 1.0 - std::f32::consts::TAU * cutoff / sample_rate;
        self.pole = pole.clamp(0.0, 0.999_99);
    }

    /// Filter one sample.
    ///
    /// **Real-time safe** — no allocation, no locking, no panics.
    #[inline]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        let output = input - self.prev_input + self.pole * self.prev_output;
        self.prev_input = input;
        self.prev_output = output;
        output
    }

    /// Filter a block in place.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn process_block(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Reset the filter memory to silence (e.g. on transport stop).
    #[inline]
    pub fn reset(&mut self) {
        self.prev_input = 0.0;
        self.prev_output = 0.0;
    }

    /// The current pole radius `R` (mostly for tests / diagnostics).
    #[inline]
    pub fn pole(&self) -> f32 {
        self.pole
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f32 = 48_000.0;

    /// A sustained DC input must be attenuated to near-zero after settling.
    /// The roadmap's Phase 4 target is > 60 dB after 500 ms.
    #[test]
    fn dc_step_is_attenuated_below_minus_60_db() {
        let mut blocker = DcBlocker::new(20.0, SAMPLE_RATE);
        // 500 ms at 48 kHz.
        let settle_samples = (SAMPLE_RATE * 0.5) as usize;
        let mut output = 0.0;
        for _ in 0..settle_samples {
            output = blocker.process_sample(1.0);
        }
        // −60 dB relative to the 1.0 DC input is 1e-3.
        assert!(
            output.abs() < 1e-3,
            "DC not attenuated enough: residual = {output}"
        );
    }

    /// A mid-band tone (1 kHz) must pass with roughly unity gain — the
    /// high-pass corner at 20 Hz is far below it.
    #[test]
    fn midband_tone_passes_at_unity_gain() {
        let mut blocker = DcBlocker::new(20.0, SAMPLE_RATE);
        let freq = 1_000.0;
        let total = 4_096;
        let mut input_energy = 0.0f64;
        let mut output_energy = 0.0f64;
        // Skip a short warm-up so the transient does not skew the RMS.
        let warmup = 512;
        for n in 0..total {
            let phase = std::f32::consts::TAU * freq * n as f32 / SAMPLE_RATE;
            let sample = phase.sin();
            let filtered = blocker.process_sample(sample);
            if n >= warmup {
                input_energy += (sample * sample) as f64;
                output_energy += (filtered * filtered) as f64;
            }
        }
        let gain = (output_energy / input_energy).sqrt();
        assert!((gain - 1.0).abs() < 0.02, "1 kHz gain not ~unity: {gain}");
    }

    /// `reset` returns the filter to a zero-state.
    #[test]
    fn reset_clears_state() {
        let mut blocker = DcBlocker::new(20.0, SAMPLE_RATE);
        for _ in 0..100 {
            blocker.process_sample(1.0);
        }
        blocker.reset();
        // First sample after reset of a fresh DC input equals the input
        // (x[0] − 0 + R·0).
        assert!((blocker.process_sample(1.0) - 1.0).abs() < 1e-6);
    }

    /// The pole must stay strictly inside the unit circle for extreme cutoffs.
    #[test]
    fn pole_stays_stable_for_extreme_cutoffs() {
        for &cutoff in &[-100.0f32, 0.0, 20.0, 1_000_000.0] {
            let blocker = DcBlocker::new(cutoff, SAMPLE_RATE);
            assert!(
                blocker.pole() >= 0.0 && blocker.pole() < 1.0,
                "unstable pole {} for cutoff {cutoff}",
                blocker.pole()
            );
        }
    }
}
