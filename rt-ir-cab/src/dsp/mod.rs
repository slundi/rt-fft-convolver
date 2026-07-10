//! Framework-agnostic DSP primitives for the cabinet plugin.

pub mod biquad;

pub use biquad::{BUTTERWORTH_Q, Biquad};

/// Convert a gain in decibels to a linear amplitude multiplier.
///
/// `0 dB → 1.0`, `+6 dB → ~2.0`, `−∞`-ish values approach `0.0`.
#[inline]
pub fn db_to_gain(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn db_to_gain_reference_points() {
        assert!((db_to_gain(0.0) - 1.0).abs() < 1e-6);
        assert!((db_to_gain(6.0206) - 2.0).abs() < 1e-3);
        assert!((db_to_gain(-6.0206) - 0.5).abs() < 1e-3);
        assert!((db_to_gain(-20.0) - 0.1).abs() < 1e-6);
    }
}
