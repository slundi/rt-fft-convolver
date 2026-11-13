//! Decibel ↔ linear-amplitude gain conversions.
//!
//! Audio gain knobs are specified in **decibels** (a `Level` of `+6 dB`,
//! `−24 dB`, …) but the render thread multiplies samples by a **linear**
//! coefficient.  These helpers convert between the two using the
//! *amplitude* (voltage) convention:
//!
//! ```text
//! linear = 10 ^ (db / 20)
//! db     = 20 · log10(linear)
//! ```
//!
//! The factor is **20**, not 10: samples are amplitudes, and power is
//! proportional to amplitude squared, so a doubling of amplitude
//! (`×2 ≈ +6.02 dB`) corresponds to a quadrupling of power (`+6.02 dB`
//! either way — the `20·log10` and `10·log10(·²)` forms agree).
//!
//! # Reference points
//!
//! | dB     | linear      |
//! |--------|-------------|
//! | `+6`   | `≈ 1.995`   |
//! | `0`    | `1.0`       |
//! | `−6`   | `≈ 0.501`   |
//! | `−20`  | `0.1`       |
//! | `−∞`   | `0.0`       |
//!
//! # Real-time safety
//!
//! Both functions are `#[inline]`, allocation-free, lock-free, and
//! panic-free — safe to call from the audio render thread.  Each is a
//! single transcendental (`powf` / `log10`); if a gain is applied to a
//! whole block, convert once outside the sample loop and reuse the
//! linear coefficient.

/// Convert a gain in **decibels** to a **linear** amplitude multiplier.
///
/// Uses the amplitude convention `linear = 10^(db / 20)`.
///
/// - `db_to_linear(0.0) == 1.0` (unity gain).
/// - Finite inputs always map to a finite, non-negative result.
/// - `f32::NEG_INFINITY` maps to `0.0` (silence); `f32::INFINITY` maps to
///   `f32::INFINITY`.
///
/// # Example
///
/// ```rust,ignore
/// # use rt_fft_convolver::utils::gain::db_to_linear;
/// assert!((db_to_linear(0.0) - 1.0).abs() < 1e-6);
/// assert!((db_to_linear(-20.0) - 0.1).abs() < 1e-6);
/// ```
#[inline]
pub fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

/// Convert a **linear** amplitude multiplier to a gain in **decibels**.
///
/// Uses the amplitude convention `db = 20 · log10(linear)`.
///
/// This is the inverse of [`db_to_linear`] for positive inputs.  Edge
/// cases follow IEEE 754 `log10` semantics:
///
/// - `linear_to_db(1.0) == 0.0` (unity gain).
/// - `linear_to_db(0.0) == f32::NEG_INFINITY` (true silence, `−∞ dB`).
/// - A **negative** `linear` yields `NaN` — a linear *amplitude* is
///   expected to be non-negative; pass its magnitude if you only care
///   about level.
///
/// # Example
///
/// ```rust,ignore
/// # use rt_fft_convolver::utils::gain::linear_to_db;
/// assert!((linear_to_db(1.0) - 0.0).abs() < 1e-6);
/// assert!((linear_to_db(0.1) - -20.0).abs() < 1e-4);
/// ```
#[inline]
pub fn linear_to_db(linear: f32) -> f32 {
    20.0 * linear.log10()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    fn assert_approx(label: &str, got: f32, expected: f32, tolerance: f32) {
        assert!(
            (got - expected).abs() < tolerance,
            "{label}: expected {expected}, got {got}"
        );
    }

    // ── db_to_linear ──────────────────────────────────────────────────────────

    /// 0 dB is unity gain.
    #[test]
    fn zero_db_is_unity() {
        assert_approx("db_to_linear(0)", db_to_linear(0.0), 1.0, EPSILON);
    }

    /// −20 dB is exactly a factor of 0.1 (amplitude convention).
    #[test]
    fn minus_twenty_db_is_tenth() {
        assert_approx("db_to_linear(-20)", db_to_linear(-20.0), 0.1, EPSILON);
    }

    /// +20 dB is exactly a factor of 10.
    #[test]
    fn plus_twenty_db_is_ten() {
        assert_approx("db_to_linear(20)", db_to_linear(20.0), 10.0, 1e-4);
    }

    /// +6.0206 dB ≈ doubling of amplitude.
    #[test]
    fn six_db_is_roughly_double() {
        assert_approx("db_to_linear(6.0206)", db_to_linear(6.020_6), 2.0, 1e-3);
    }

    /// −∞ dB is silence.
    #[test]
    fn neg_inf_db_is_zero() {
        assert_eq!(db_to_linear(f32::NEG_INFINITY), 0.0);
    }

    /// The full `rt-drive` `Level` range (−24…+24 dB) stays finite and positive.
    #[test]
    fn level_range_is_finite_and_positive() {
        let mut db = -24.0_f32;
        while db <= 24.0 {
            let linear = db_to_linear(db);
            assert!(
                linear.is_finite() && linear > 0.0,
                "db {db} produced non-finite/non-positive linear {linear}"
            );
            db += 0.5;
        }
    }

    // ── linear_to_db ──────────────────────────────────────────────────────────

    /// Unity gain is 0 dB.
    #[test]
    fn unity_is_zero_db() {
        assert_approx("linear_to_db(1)", linear_to_db(1.0), 0.0, EPSILON);
    }

    /// 0.1 linear is −20 dB.
    #[test]
    fn tenth_is_minus_twenty_db() {
        assert_approx("linear_to_db(0.1)", linear_to_db(0.1), -20.0, 1e-4);
    }

    /// True silence maps to −∞ dB.
    #[test]
    fn zero_linear_is_neg_inf_db() {
        assert_eq!(linear_to_db(0.0), f32::NEG_INFINITY);
    }

    /// A negative amplitude is undefined → NaN.
    #[test]
    fn negative_linear_is_nan() {
        assert!(linear_to_db(-1.0).is_nan());
    }

    // ── Round-trip ────────────────────────────────────────────────────────────

    /// `linear_to_db` is the inverse of `db_to_linear` across the useful range.
    #[test]
    fn round_trip_db_linear_db() {
        for &db in &[-60.0_f32, -24.0, -6.0, -1.0, 0.0, 1.0, 6.0, 24.0, 48.0] {
            let round_tripped = linear_to_db(db_to_linear(db));
            assert_approx(&format!("round-trip {db} dB"), round_tripped, db, 1e-3);
        }
    }

    /// `db_to_linear` is the inverse of `linear_to_db` across positive gains.
    #[test]
    fn round_trip_linear_db_linear() {
        for &linear in &[1.0e-3_f32, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0] {
            let round_tripped = db_to_linear(linear_to_db(linear));
            assert_approx(
                &format!("round-trip {linear} linear"),
                round_tripped,
                linear,
                linear * 1e-3,
            );
        }
    }
}
