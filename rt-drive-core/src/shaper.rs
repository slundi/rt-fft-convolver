//! Nonlinear waveshaping transfer functions — the "shaper curves".
//!
//! Each [`Shaper`] maps a single input sample to a single output sample
//! through a static, memoryless nonlinearity `y = f(x)`.  Oversampling and
//! tone shaping live elsewhere ([`crate::resampler`], [`crate::tone`]); this
//! module is *only* the curve.
//!
//! # Curve families ([decision], Phase 0)
//!
//! | Variant           | Family | Character                                   |
//! |-------------------|--------|---------------------------------------------|
//! | [`Shaper::Tanh`]        | soft  | smooth, symmetric, classic "warm" clip   |
//! | [`Shaper::Reciprocal`]  | soft  | `x/√(1+x²)`, softer knee than tanh       |
//! | [`Shaper::CubicSoft`]   | soft  | cubic soft-clip, hard knee at ±1         |
//! | [`Shaper::HardClip`]    | hard  | brick-wall clip, optional asymmetry      |
//! | [`Shaper::Diode`]       | diode | exponential clipper (Rat / TS-style)     |
//! | [`Shaper::Tube`]        | tube  | asymmetric triode-style saturation       |
//! | [`Shaper::Tape`]        | tape  | bias-shifted tanh (tape saturation)      |
//!
//! # Invariants every curve upholds
//!
//! - **Passes through the origin**: `f(0.0) == 0.0`.  A curve that injects a
//!   DC offset (asymmetric [`Shaper::Tube`], biased [`Shaper::Tape`]) still
//!   returns `0` at `0`; the offset appears only for non-zero input and is
//!   removed downstream by [`crate::dc_block::DcBlocker`].
//! - **Monotonic non-decreasing**: `x₁ ≤ x₂ ⇒ f(x₁) ≤ f(x₂)`.  This keeps the
//!   transfer curve free of the fold-back distortion a naïve `x - x³/3` would
//!   produce past `|x| = 1`.
//! - **Bounded**: every curve saturates rather than growing without limit.
//!
//! The exact math and its "why" are documented on each variant.

/// A memoryless nonlinear waveshaper.
///
/// `Copy` + `PartialEq` so a host can store one inside a data-driven pedal
/// preset (see the roadmap's "Pedal model format") and compare/replace it
/// without allocation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Shaper {
    /// `y = tanh(x)` — the canonical smooth symmetric saturator.
    Tanh,

    /// `y = x / √(1 + x²)` — algebraic sigmoid with a gentler knee and
    /// cheaper evaluation than [`Shaper::Tanh`] (no transcendental).
    Reciprocal,

    /// Cubic soft-clip.  Below the knee the transfer is `1.5·(x − x³/3)`,
    /// which reaches exactly `±1` at `x = ±1`; beyond the knee the input is
    /// clamped so the curve stays flat (monotonic) instead of folding back.
    CubicSoft,

    /// Brick-wall clip to `[lower, upper]`.
    ///
    /// - `threshold` is the positive clip level (`upper`).
    /// - `asymmetry ∈ [−1, 1]` skews the negative level:
    ///   `lower = −threshold · (1 − asymmetry)`.  `0` is symmetric; a positive
    ///   value clips the negative swing earlier (even-harmonic bias).
    HardClip {
        /// Positive clip level (must be `> 0`).
        threshold: f32,
        /// Negative-side skew in `[−1, 1]`.
        asymmetry: f32,
    },

    /// Exponential diode clipper: `y = sign(x)·(1 − e^{−|x·drive|})`.
    ///
    /// Models the soft, rounded knee of a diode-to-ground clipping stage
    /// (ProCo Rat, Tube Screamer).  `drive` scales the input into the knee;
    /// larger `drive` → harder clip.  Output is bounded in `(−1, 1)`.
    Diode {
        /// Pre-gain into the exponential knee (`> 0`).
        drive: f32,
    },

    /// Simplified asymmetric triode-style saturation.
    ///
    /// The positive half saturates softly (`tanh x`) while the negative half
    /// saturates harder (`0.5·tanh 2x`), matching a triode's asymmetric
    /// grid/plate transfer and producing the even-harmonic "tube" warmth.
    ///
    /// This is intentionally *not* a full Koren plate-current model — that is
    /// circuit-level modelling and out of scope for the shared core (see the
    /// roadmap's "Out of scope"); this curve is `C¹`-continuous at the origin
    /// (unit slope from both sides) and monotonic.
    Tube,

    /// Bias-shifted tanh — tape-style saturation.
    ///
    /// `y = tanh(x + bias) − tanh(bias)`.  The `+bias` shift moves the
    /// operating point onto the asymmetric part of the tanh curve (as a tape
    /// bias signal would); subtracting `tanh(bias)` re-centres the curve so
    /// `f(0) = 0` and only the *shape* asymmetry (not a static DC offset)
    /// remains.
    Tape {
        /// Operating-point bias.
        bias: f32,
    },
}

impl Shaper {
    /// Evaluate the curve for a single sample.
    ///
    /// **Real-time safe** — no allocation, no locking, no panics.
    #[inline]
    pub fn process(self, x: f32) -> f32 {
        match self {
            Shaper::Tanh => x.tanh(),
            Shaper::Reciprocal => x / (1.0 + x * x).sqrt(),
            Shaper::CubicSoft => cubic_soft(x),
            Shaper::HardClip {
                threshold,
                asymmetry,
            } => {
                // asymmetry is clamped just short of ±1 so the two clip levels
                // never collide into a zero-width window.
                let lower = -threshold * (1.0 - asymmetry.clamp(-0.999, 0.999));
                x.clamp(lower, threshold)
            }
            Shaper::Diode { drive } => {
                let driven = x * drive;
                // sign(0) = 1 in Rust, but (1 − e⁰) = 0, so f(0) = 0 holds.
                driven.signum() * (1.0 - (-driven.abs()).exp())
            }
            Shaper::Tube => {
                if x >= 0.0 {
                    x.tanh()
                } else {
                    0.5 * (2.0 * x).tanh()
                }
            }
            Shaper::Tape { bias } => (x + bias).tanh() - bias.tanh(),
        }
    }

    /// Apply the curve to every sample of `input`, writing to `output`.
    ///
    /// Processes `min(input.len(), output.len())` samples so a length mismatch
    /// is deterministic in release builds; debug builds assert equal lengths.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn process_block(self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), output.len(), "shaper block length mismatch");
        for (sample_in, sample_out) in input.iter().zip(output.iter_mut()) {
            *sample_out = self.process(*sample_in);
        }
    }
}

/// Cubic soft-clip shared by [`Shaper::CubicSoft`] and the SIMD kernel.
///
/// The input is clamped to `[−1, 1]` *before* the polynomial so the curve is
/// monotonic everywhere (the bare `x − x³/3` turns over past `|x| = 1`).  The
/// `1.5` factor normalises the knee so `f(±1) = ±1`.
#[inline]
pub(crate) fn cubic_soft(x: f32) -> f32 {
    let clamped = x.clamp(-1.0, 1.0);
    1.5 * (clamped - clamped * clamped * clamped / 3.0)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The full set of shapers under test, with representative parameters.
    fn all_shapers() -> Vec<Shaper> {
        vec![
            Shaper::Tanh,
            Shaper::Reciprocal,
            Shaper::CubicSoft,
            Shaper::HardClip {
                threshold: 0.8,
                asymmetry: 0.0,
            },
            Shaper::HardClip {
                threshold: 0.8,
                asymmetry: 0.4,
            },
            Shaper::Diode { drive: 2.0 },
            Shaper::Tube,
            Shaper::Tape { bias: 0.3 },
        ]
    }

    /// Every curve must pass through the origin.
    #[test]
    fn all_curves_pass_through_origin() {
        for shaper in all_shapers() {
            let output = shaper.process(0.0);
            assert!(
                output.abs() < 1e-6,
                "{shaper:?} does not pass through origin: f(0) = {output}"
            );
        }
    }

    /// Every curve must be monotonic non-decreasing over a wide input sweep.
    #[test]
    fn all_curves_are_monotonic() {
        for shaper in all_shapers() {
            let mut previous = shaper.process(-8.0);
            let mut sample = -8.0;
            while sample <= 8.0 {
                let current = shaper.process(sample);
                assert!(
                    current >= previous - 1e-6,
                    "{shaper:?} not monotonic at x = {sample}: {previous} → {current}"
                );
                previous = current;
                sample += 0.01;
            }
        }
    }

    /// Every curve must stay bounded for large-magnitude input.
    #[test]
    fn all_curves_are_bounded() {
        for shaper in all_shapers() {
            for &x in &[-1000.0f32, -50.0, 50.0, 1000.0] {
                let output = shaper.process(x);
                assert!(
                    output.is_finite() && output.abs() <= 2.0,
                    "{shaper:?} unbounded at x = {x}: {output}"
                );
            }
        }
    }

    /// `tanh` reference points.
    #[test]
    fn tanh_reference_points() {
        assert!((Shaper::Tanh.process(0.0) - 0.0).abs() < 1e-6);
        assert!((Shaper::Tanh.process(10.0) - 1.0).abs() < 1e-3);
    }

    /// Cubic soft-clip must reach exactly ±1 at the knee and saturate beyond.
    #[test]
    fn cubic_soft_reaches_unity_at_knee() {
        assert!((Shaper::CubicSoft.process(1.0) - 1.0).abs() < 1e-6);
        assert!((Shaper::CubicSoft.process(-1.0) + 1.0).abs() < 1e-6);
        // Beyond the knee it stays clamped, not folded back.
        assert!((Shaper::CubicSoft.process(5.0) - 1.0).abs() < 1e-6);
    }

    /// Symmetric hard-clip clamps at ±threshold.
    #[test]
    fn hard_clip_symmetric_clamps_at_threshold() {
        let shaper = Shaper::HardClip {
            threshold: 0.5,
            asymmetry: 0.0,
        };
        assert!((shaper.process(2.0) - 0.5).abs() < 1e-6);
        assert!((shaper.process(-2.0) + 0.5).abs() < 1e-6);
        assert!((shaper.process(0.25) - 0.25).abs() < 1e-6);
    }

    /// Positive asymmetry clips the negative swing earlier than the positive.
    #[test]
    fn hard_clip_asymmetry_skews_negative_level() {
        let shaper = Shaper::HardClip {
            threshold: 1.0,
            asymmetry: 0.5,
        };
        // upper stays at +1, lower becomes -(1 - 0.5) = -0.5.
        assert!((shaper.process(2.0) - 1.0).abs() < 1e-6);
        assert!((shaper.process(-2.0) + 0.5).abs() < 1e-6);
    }

    /// The tube curve is asymmetric: |f(−x)| < |f(x)| for x > 0.
    #[test]
    fn tube_is_asymmetric() {
        let positive = Shaper::Tube.process(1.0);
        let negative = Shaper::Tube.process(-1.0);
        assert!(
            negative.abs() < positive.abs(),
            "tube not asymmetric: f(1)={positive}, f(-1)={negative}"
        );
    }

    /// Diode clipper saturates toward ±1 and is symmetric in magnitude.
    #[test]
    fn diode_saturates_toward_unity() {
        let shaper = Shaper::Diode { drive: 3.0 };
        assert!((shaper.process(10.0) - 1.0).abs() < 1e-3);
        assert!((shaper.process(-10.0) + 1.0).abs() < 1e-3);
    }

    /// `process_block` matches per-sample `process`.
    #[test]
    fn block_matches_scalar() {
        let shaper = Shaper::Tanh;
        let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let mut output = vec![0.0f32; input.len()];
        shaper.process_block(&input, &mut output);
        for (sample_in, &sample_out) in input.iter().zip(output.iter()) {
            assert!((shaper.process(*sample_in) - sample_out).abs() < 1e-7);
        }
    }
}
