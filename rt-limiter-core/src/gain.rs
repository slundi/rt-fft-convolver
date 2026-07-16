//! [`GainComputer`] — the feed-forward gain-reduction smoother.
//!
//! # The Phase 0 topology
//!
//! Roadmap-07 Phase 0 chose a **feed-forward** gain computer: the reduction is
//! derived from a peak the audio has *not yet reached* (the look-ahead window
//! maximum, supplied by [`crate::WindowMax`]).  Per input sample the computer
//! is handed the windowed peak `p` and produces a smoothed linear gain `g`:
//!
//! ```text
//!   target = if p ≤ threshold { 1 } else { threshold / p }
//! ```
//!
//! - **Attack** is *instantaneous* here.  It does not need a ramp because the
//!   look-ahead window has already dropped `target` the moment a loud sample
//!   entered the window — i.e. `latency + look-ahead` samples *before* that
//!   sample leaves the delay line.  Snapping straight to `target` is therefore
//!   both click-free (it happens on quieter runway) and the reason the
//!   brick-wall guarantee holds: `g ≤ threshold / peak` for the sample being
//!   output.  This is the "attack = look-ahead length" from the roadmap.
//! - **Release** is the shaped recovery back toward unity once the peak has
//!   passed, per the [`ReleaseCurve`].
//!
//! Note the computer targets **`threshold`**, not the ceiling.  The ceiling is
//! the true-peak output cap enforced downstream by [`crate::CeilingClipper`]
//! and is the guaranteed absolute bound; the threshold is the level peaks are
//! pulled down to.  Set the threshold at or below the ceiling and the clipper
//! never engages.
//!
//! # Real-time safety
//!
//! [`GainComputer::process`] is one branch plus a handful of multiplies — no
//! allocation, no lock, no panic.

use rt_fft_convolver::db_to_linear;

/// The recovery shape applied once a peak has passed (Roadmap-07 Phase 0).
///
/// `Linear` and `Exponential` are the two engine primitives; `Auto` is a
/// simplified program-dependent blend of a fast and a slow one-pole (the
/// "transparent" feel).  The Phase 2 plugin layers its "punchy"/"smooth"
/// *styles* on top of these by choosing a curve and a release time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReleaseCurve {
    /// Constant linear-gain-per-sample recovery.
    Linear,
    /// One-pole exponential recovery (the natural default).
    Exponential,
    /// Program-dependent: blend of a fast and a slow exponential.
    Auto,
}

/// Feed-forward gain-reduction smoother for one (possibly channel-linked)
/// signal.
pub struct GainComputer {
    /// Limiting level as a linear amplitude — peaks are pulled down to this.
    threshold: f32,
    sample_rate: f32,
    curve: ReleaseCurve,
    release_ms: f32,
    /// One-pole release coefficient `exp(−1 / (t_r · fs))`.
    release_coeff: f32,
    /// Fast one-pole coefficient (¼ of the release time) for `Auto`.
    release_coeff_fast: f32,
    /// Per-sample linear-gain step for the `Linear` curve.
    release_step: f32,
    /// Current smoothed gain in `(0, 1]`.
    gain: f32,
}

impl GainComputer {
    /// Create a gain computer.
    ///
    /// - `threshold_db` — limiting level in dBFS (peaks pulled down to it).
    /// - `sample_rate` — host sample rate in Hz.
    /// - `release_ms` — recovery time in milliseconds.
    /// - `curve` — recovery shape.
    ///
    /// **Allocates nothing**, but calls transcendentals; fine off the audio
    /// thread and cheap enough for occasional live changes.
    pub fn new(threshold_db: f32, sample_rate: f32, release_ms: f32, curve: ReleaseCurve) -> Self {
        let mut computer = Self {
            threshold: db_to_linear(threshold_db),
            sample_rate,
            curve,
            release_ms,
            release_coeff: 0.0,
            release_coeff_fast: 0.0,
            release_step: 0.0,
            gain: 1.0,
        };
        computer.recompute_release();
        computer
    }

    /// Recompute the release coefficients from `release_ms` / `sample_rate`.
    fn recompute_release(&mut self) {
        let seconds = (self.release_ms * 1e-3).max(1e-6);
        let samples = seconds * self.sample_rate;
        // g[n] = c·g[n−1] + (1−c)·target with c = exp(−1/samples) reaches
        // ~63 % of the way to `target` in `samples` samples.
        self.release_coeff = (-1.0 / samples).exp();
        self.release_coeff_fast = (-1.0 / (0.25 * samples)).exp();
        // Linear: cross the full 0→1 range in `samples` samples.
        self.release_step = 1.0 / samples;
    }

    /// Set the limiting threshold in dBFS.  **Real-time safe** (one `powf`).
    #[inline]
    pub fn set_threshold_db(&mut self, threshold_db: f32) {
        self.threshold = db_to_linear(threshold_db);
    }

    /// Set the release time in milliseconds.  **Real-time safe.**
    #[inline]
    pub fn set_release_ms(&mut self, release_ms: f32) {
        self.release_ms = release_ms;
        self.recompute_release();
    }

    /// Set the release curve.  **Real-time safe.**
    #[inline]
    pub fn set_curve(&mut self, curve: ReleaseCurve) {
        self.curve = curve;
    }

    /// The current smoothed gain (linear, `≤ 1`).
    #[inline]
    pub fn current_gain(&self) -> f32 {
        self.gain
    }

    /// Feed the windowed peak `windowed_peak` and return the smoothed gain to
    /// apply to the delayed dry sample.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn process(&mut self, windowed_peak: f32) -> f32 {
        // Gain that pulls this window's loudest peak down to the threshold.
        let target = if windowed_peak <= self.threshold {
            1.0
        } else {
            self.threshold / windowed_peak
        };

        if target <= self.gain {
            // Attack: the look-ahead already saw this peak, so snap down.  Any
            // overshoot below `target` would only be *safer*; we take exactly
            // `target` to reduce no more than necessary.
            self.gain = target;
        } else {
            // Release: ease back up toward `target` (usually unity).
            self.gain = match self.curve {
                ReleaseCurve::Linear => (self.gain + self.release_step).min(target),
                ReleaseCurve::Exponential => {
                    self.release_coeff * self.gain + (1.0 - self.release_coeff) * target
                }
                ReleaseCurve::Auto => {
                    let fast = self.release_coeff_fast * self.gain
                        + (1.0 - self.release_coeff_fast) * target;
                    let slow = self.release_coeff * self.gain + (1.0 - self.release_coeff) * target;
                    0.5 * (fast + slow)
                }
            };
        }
        self.gain
    }

    /// Reset the smoother to unity gain.
    pub fn reset(&mut self) {
        self.gain = 1.0;
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f32 = 48_000.0;

    /// A peak at or below threshold leaves the gain at unity (bypass).
    #[test]
    fn sub_threshold_is_unity() {
        let mut computer = GainComputer::new(0.0, SAMPLE_RATE, 100.0, ReleaseCurve::Exponential);
        for _ in 0..100 {
            assert_eq!(computer.process(0.5), 1.0);
        }
    }

    /// A peak above threshold snaps the gain to `threshold / peak` in one
    /// sample (instantaneous attack).
    #[test]
    fn over_threshold_snaps_to_target() {
        // threshold 0 dB = 1.0 linear.
        let mut computer = GainComputer::new(0.0, SAMPLE_RATE, 100.0, ReleaseCurve::Exponential);
        let gain = computer.process(2.0);
        assert!((gain - 0.5).abs() < 1e-6, "expected 0.5, got {gain}");
    }

    /// −6 dB threshold pulls a peak of 1.0 down to ≈ 0.5 (−6 dB).
    #[test]
    fn threshold_scales_target() {
        let mut computer = GainComputer::new(-6.0, SAMPLE_RATE, 50.0, ReleaseCurve::Exponential);
        let gain = computer.process(1.0);
        // threshold_lin ≈ 0.501 → gain ≈ 0.501.
        assert!((gain - db_to_linear(-6.0)).abs() < 1e-6);
    }

    /// After a peak passes, an exponential release climbs monotonically back
    /// toward unity and effectively reaches it within ~the release time.
    #[test]
    fn exponential_release_recovers() {
        let release_ms = 20.0;
        let mut computer =
            GainComputer::new(0.0, SAMPLE_RATE, release_ms, ReleaseCurve::Exponential);
        computer.process(4.0); // drive gain down to 0.25
        assert!((computer.current_gain() - 0.25).abs() < 1e-6);

        let mut previous = computer.current_gain();
        let samples = (release_ms * 1e-3 * SAMPLE_RATE) as usize;
        // Run five time-constants — exp(−5) ≈ 0.0067 remaining.
        for _ in 0..(samples * 5) {
            let gain = computer.process(0.0); // sub-threshold now
            assert!(gain >= previous - 1e-7, "release must be monotonic up");
            previous = gain;
        }
        assert!(
            previous > 0.99,
            "gain should have essentially recovered, got {previous}"
        );
    }

    /// Linear release reaches unity in ~the specified time (±10 %), matching
    /// the Phase 4 acceptance criterion.
    #[test]
    fn linear_release_time_is_accurate() {
        let release_ms = 10.0;
        let mut computer = GainComputer::new(0.0, SAMPLE_RATE, release_ms, ReleaseCurve::Linear);
        computer.process(2.0); // gain → 0.5
        let mut samples_to_unity = 0usize;
        for _ in 0..100_000 {
            let gain = computer.process(0.0);
            samples_to_unity += 1;
            if gain >= 1.0 - 1e-6 {
                break;
            }
        }
        // Linear step crosses a full 0→1 in `release_ms`; from 0.5 it takes
        // half that time.
        let expected = 0.5 * release_ms * 1e-3 * SAMPLE_RATE;
        let ratio = samples_to_unity as f32 / expected;
        assert!(
            (0.9..=1.1).contains(&ratio),
            "linear release took {samples_to_unity} samples, expected ≈ {expected}"
        );
    }

    /// `Auto` recovers monotonically and settles near unity within the release
    /// window (its blend sits between the fast and slow poles).
    #[test]
    fn auto_release_recovers() {
        let release_ms = 15.0;
        let mut computer = GainComputer::new(0.0, SAMPLE_RATE, release_ms, ReleaseCurve::Auto);
        computer.process(4.0);
        let samples = (release_ms * 1e-3 * SAMPLE_RATE) as usize;
        let mut previous = computer.current_gain();
        for _ in 0..(samples * 6) {
            let gain = computer.process(0.0);
            assert!(gain >= previous - 1e-7);
            previous = gain;
        }
        assert!(previous > 0.99, "auto should recover, got {previous}");
    }

    /// Live setters take effect without a rebuild.
    #[test]
    fn setters_apply() {
        let mut computer = GainComputer::new(0.0, SAMPLE_RATE, 100.0, ReleaseCurve::Exponential);
        computer.set_threshold_db(-6.0);
        let gain = computer.process(1.0);
        assert!((gain - db_to_linear(-6.0)).abs() < 1e-6);

        computer.set_curve(ReleaseCurve::Linear);
        computer.set_release_ms(5.0);
        // A shorter release recovers faster: one step should be visible.
        let before = computer.current_gain();
        let after = computer.process(0.0);
        assert!(after > before);
    }

    /// `reset` returns to unity gain.
    #[test]
    fn reset_restores_unity() {
        let mut computer = GainComputer::new(0.0, SAMPLE_RATE, 100.0, ReleaseCurve::Exponential);
        computer.process(8.0);
        computer.reset();
        assert_eq!(computer.current_gain(), 1.0);
    }
}
