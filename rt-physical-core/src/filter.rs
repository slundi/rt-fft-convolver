//! The string loop filter (bridge / damping filter) — Phase 1 "Bridge / loop
//! filter (1-pole low-pass)" in `docs/Roadmap-06-physical-modeling.md`.
//!
//! # Two jobs in one filter
//!
//! Every pass around a Karplus-Strong loop is multiplied by this filter, so it
//! sets both **how fast** the string decays and **how bright** it stays:
//!
//! - **Decay** — a broadband gain `g < 1` applied every pass.  For a target
//!   `T60` (time to −60 dB) at a fundamental `f0`, a note completes `T60·f0`
//!   loop passes in that time, so `g = 10^(−3 / (T60·f0))` (independent of the
//!   sample rate — see the Phase 0 KS example).
//! - **Damping / brightness** — a one-zero low-pass `(1 − S) + S·z⁻¹` that rolls
//!   off highs a little each pass, so overtones decay faster than the
//!   fundamental, exactly as a real string loses its highs first.
//!
//! # Why a one-zero (FIR) filter, not a one-pole
//!
//! Tuning needs the filter's **phase delay** to be known so it can be folded
//! into the loop-length budget.  A one-zero `(1 − S) + S·z⁻¹` is trivially
//! stable and, at `S = 0.5`, is the classic linear-phase averaging filter with
//! an exact half-sample delay.  Its phase delay is available in closed form
//! ([`phase_delay`](LoopFilter::phase_delay)), which a one-pole low-pass would
//! not give as cleanly.
//!
//! # Real-time safety
//!
//! `process` is two multiply-adds; `phase_delay` calls `atan2` once and is used
//! only when retuning a note, not per sample.  No allocation, no locking.  The
//! stored input is flushed to zero so a decaying loop stays out of the denormal
//! slow path.

use rt_fft_convolver::flush_to_zero;

/// Maximum value of the damping coefficient `S`.  `0.5` is the averaging
/// filter — the darkest, most damped setting; beyond it the filter would start
/// to invert its high end.
const MAX_DAMPING_S: f32 = 0.5;

/// A one-zero low-pass with a broadband decay gain, sitting in the string loop.
#[derive(Debug, Clone, Copy)]
pub struct LoopFilter {
    /// Per-pass broadband gain `g ∈ (0, 1]` setting the decay time.
    gain: f32,
    /// One-zero coefficient `S ∈ [0, 0.5]`; larger = darker / faster HF decay.
    damping: f32,
    /// Previous input `x[n−1]`.
    x1: f32,
}

impl LoopFilter {
    /// Create a loop filter with unity decay gain and `damping ∈ [0, 1]`
    /// (mapped internally to `S ∈ [0, 0.5]`).
    pub fn new(damping: f32) -> Self {
        let mut filter = Self {
            gain: 1.0,
            damping: 0.0,
            x1: 0.0,
        };
        filter.set_damping(damping);
        filter
    }

    /// Set the brightness/damping from a normalised `[0, 1]` control.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn set_damping(&mut self, damping: f32) {
        self.damping = damping.clamp(0.0, 1.0) * MAX_DAMPING_S;
    }

    /// Set the per-pass decay gain directly.  Clamped to `(0, 1]` for stability.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn set_gain(&mut self, gain: f32) {
        self.gain = gain.clamp(f32::MIN_POSITIVE, 1.0);
    }

    /// Set the decay gain from a target `t60` (seconds) at fundamental `freq`.
    ///
    /// `g = 10^(−3 / (T60·f0))`.  An effectively-infinite `t60` sustains
    /// (gain → 1); a tiny one damps almost instantly.
    ///
    /// **Real-time safe** apart from one `powf`.
    #[inline]
    pub fn set_decay(&mut self, t60: f32, freq: f32) {
        if t60 <= 0.0 || freq <= 0.0 {
            self.gain = 0.0;
            return;
        }
        let passes = t60 * freq; // number of loop passes in one T60
        self.gain = 10.0f32.powf(-3.0 / passes).clamp(f32::MIN_POSITIVE, 1.0);
    }

    /// Phase delay (in samples) contributed by the one-zero at angular
    /// frequency `omega` (rad/sample).  This is folded into the loop-length
    /// budget so the string stays in tune.
    ///
    /// For `H(e^{jω}) = (1 − S) + S·e^{−jω}` the phase delay is
    /// `atan2(S·sinω, (1 − S) + S·cosω) / ω`, which tends to `S` as `ω → 0`.
    ///
    /// **Real-time safe** (one `atan2`); called only when retuning.
    #[inline]
    pub fn phase_delay(&self, omega: f32) -> f32 {
        if omega <= f32::MIN_POSITIVE {
            return self.damping; // low-frequency limit
        }
        let sin = (self.damping * omega.sin()).max(0.0);
        let real = (1.0 - self.damping) + self.damping * omega.cos();
        sin.atan2(real) / omega
    }

    /// The current decay gain.
    #[inline]
    pub fn gain(&self) -> f32 {
        self.gain
    }

    /// Clear the filter memory (but keep the coefficients).
    #[inline]
    pub fn reset(&mut self) {
        self.x1 = 0.0;
    }

    /// Filter one sample: `y[n] = g·[(1 − S)·x[n] + S·x[n−1]]`.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        let output = self.gain * ((1.0 - self.damping) * input + self.damping * self.x1);
        self.x1 = flush_to_zero(input);
        output
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    #[test]
    fn decay_gain_from_t60_is_below_unity() {
        let mut filter = LoopFilter::new(0.0);
        filter.set_decay(1.0, 110.0); // 1 s T60 at A2
        let gain = filter.gain();
        assert!(gain < 1.0 && gain > 0.9, "unexpected decay gain {gain}");
    }

    #[test]
    fn longer_t60_gives_gain_closer_to_one() {
        let mut short = LoopFilter::new(0.0);
        short.set_decay(0.5, 220.0);
        let mut long = LoopFilter::new(0.0);
        long.set_decay(5.0, 220.0);
        assert!(long.gain() > short.gain());
    }

    #[test]
    fn zero_t60_kills_the_loop() {
        let mut filter = LoopFilter::new(0.0);
        filter.set_decay(0.0, 220.0);
        assert_eq!(filter.gain(), 0.0);
    }

    #[test]
    fn phase_delay_tends_to_s_at_low_frequency() {
        // S = 0.5 (max damping) → half-sample delay in the low-frequency limit.
        let filter = LoopFilter::new(1.0);
        let omega = TAU * 100.0 / 48_000.0;
        assert!((filter.phase_delay(omega) - 0.5).abs() < 1e-3);
        // With no damping (S = 0) there is no phase delay.
        let bright = LoopFilter::new(0.0);
        assert!(bright.phase_delay(omega).abs() < 1e-6);
    }

    #[test]
    fn damping_rolls_off_highs_more_than_lows() {
        const SR: f32 = 48_000.0;
        let tone = |freq: f32| {
            let mut filter = LoopFilter::new(1.0); // max damping, unity gain
            let mut sum_sq = 0.0f32;
            let mut count = 0u32;
            for n in 0..48_000 {
                let x = (TAU * freq * n as f32 / SR).sin();
                let y = filter.process(x);
                if n > 24_000 {
                    sum_sq += y * y;
                    count += 1;
                }
            }
            (sum_sq / count as f32).sqrt() * std::f32::consts::SQRT_2
        };
        assert!(tone(100.0) > 0.95, "lows should pass nearly intact");
        // The averaging filter's magnitude is |cos(ω/2)|, ≈ 0.26 at 20 kHz
        // (48 kHz sr) and only reaching 0 at Nyquist — so highs are clearly
        // rolled off relative to lows without being fully nulled.
        assert!(tone(20_000.0) < 0.35, "highs should be strongly damped");
    }

    #[test]
    fn reset_clears_state() {
        let mut filter = LoopFilter::new(0.5);
        filter.process(1.0);
        filter.reset();
        // With zero history the first output is g·(1−S)·x.
        let expected = filter.gain() * (1.0 - 0.5 * 0.5);
        assert!((filter.process(1.0) - expected).abs() < 1e-6);
    }
}
