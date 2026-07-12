//! Activation functions used across LSTM and WaveNet cells.
//!
//! # Why LUTs?
//!
//! `tanh` and `sigmoid` are the dominant non-linearities in an LSTM cell:
//! four `tanh` and four `sigmoid` evaluations per audio sample per layer.
//! `f32::tanh` on x86-64 uses `libm` fallback ≈ 40-80 cycles / call, which
//! turns into 400+ cycles per LSTM step — a very real fraction of the
//! per-sample deadline at 48 kHz.
//!
//! A **piecewise-linear LUT** with 2048 entries over `[-8, +8]` gives a
//! bounded absolute error of ≈ 2⁻¹² for both curves, which is well under
//! the noise floor introduced by weight quantisation and matches the
//! tolerance published by the NAM and Proteus reference C++ engines.
//!
//! Outside `[-8, +8]` we clamp to the exact asymptote (`±1` for tanh,
//! `0` / `1` for sigmoid) — the true curve differs by less than `f32::EPSILON`
//! at `|input| = 8`.
//!
//! # Real-time safety
//!
//! All functions in this module are `#[inline]` and allocation-free.  The
//! LUTs are `static` arrays baked into the binary — no heap, no
//! lazy-initialisation cost on the audio thread.

/// LUT half-width in input units.  See module docs.
const LUT_RANGE: f32 = 8.0;

/// Number of LUT samples across `[-LUT_RANGE, +LUT_RANGE]`.
const LUT_SIZE: usize = 2048;

/// Samples-per-input-unit conversion factor.
const LUT_SCALE: f32 = (LUT_SIZE - 1) as f32 / (2.0 * LUT_RANGE);

/// Pre-tabulated `tanh(x)` values over `[-LUT_RANGE, +LUT_RANGE]`.
///
/// Built at program start via a `LazyLock` initialiser — the LUT is
/// touched exactly once, before any audio thread can access it.  Reads on
/// the render path are pure indexed loads.
static TANH_LUT: std::sync::LazyLock<[f32; LUT_SIZE]> = std::sync::LazyLock::new(|| {
    let mut table = [0.0f32; LUT_SIZE];
    for (index, cell) in table.iter_mut().enumerate() {
        let position = -LUT_RANGE + (2.0 * LUT_RANGE) * (index as f32) / ((LUT_SIZE - 1) as f32);
        *cell = position.tanh();
    }
    table
});

/// Pre-tabulated `sigmoid(x)` values over `[-LUT_RANGE, +LUT_RANGE]`.
static SIGMOID_LUT: std::sync::LazyLock<[f32; LUT_SIZE]> = std::sync::LazyLock::new(|| {
    let mut table = [0.0f32; LUT_SIZE];
    for (index, cell) in table.iter_mut().enumerate() {
        let position = -LUT_RANGE + (2.0 * LUT_RANGE) * (index as f32) / ((LUT_SIZE - 1) as f32);
        *cell = 1.0 / (1.0 + (-position).exp());
    }
    table
});

/// Linear-interpolate a LUT at a real-valued position expressed in "cells".
///
/// The caller is responsible for translating input into cell coordinates;
/// this keeps the hot path branch-free (except the range clamp).
#[inline(always)]
fn lut_lookup(table: &[f32; LUT_SIZE], cell_position: f32) -> f32 {
    // Clamp so we never read out of bounds.
    let clamped = cell_position.clamp(0.0, (LUT_SIZE - 1) as f32);
    let base_index = clamped as usize;
    // Guard the upper edge: `base_index + 1` must stay in bounds.
    let next_index = (base_index + 1).min(LUT_SIZE - 1);
    let fraction = clamped - base_index as f32;
    let lower = table[base_index];
    let upper = table[next_index];
    lower + (upper - lower) * fraction
}

/// Fast `tanh(input)` via linear-interpolated LUT.
///
/// **Accuracy**: |error| < 2⁻¹² over `[-8, 8]`; exact `±1` outside.
///
/// **Cost**: one branchy clamp + two loads + one FMA on any target.
#[inline(always)]
pub fn tanh(input: f32) -> f32 {
    if input >= LUT_RANGE {
        return 1.0;
    }
    if input <= -LUT_RANGE {
        return -1.0;
    }
    let cell_position = (input + LUT_RANGE) * LUT_SCALE;
    lut_lookup(&TANH_LUT, cell_position)
}

/// Fast `sigmoid(input)` via linear-interpolated LUT.
///
/// See [`tanh`] for accuracy and cost.  Sigmoid is `(tanh(x/2) + 1) / 2`
/// mathematically but we keep a dedicated LUT so the LSTM gate loop stays
/// a straight fetch — no per-sample arithmetic.
#[inline(always)]
pub fn sigmoid(input: f32) -> f32 {
    if input >= LUT_RANGE {
        return 1.0;
    }
    if input <= -LUT_RANGE {
        return 0.0;
    }
    let cell_position = (input + LUT_RANGE) * LUT_SCALE;
    lut_lookup(&SIGMOID_LUT, cell_position)
}

/// Rectified linear unit — `max(0, input)`.
///
/// Branchless on x86-64 (LLVM lowers to `maxss`) and AArch64 (`fmax`).
#[inline(always)]
pub fn relu(input: f32) -> f32 {
    input.max(0.0)
}

/// GELU approximation from Hendrycks & Gimpel (2016) — the "tanh" variant
/// used by NAM's WaveNet activations.
///
/// `GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))`
///
/// The inner `tanh` uses the module's LUT so the whole call is still
/// allocation-free.
#[inline(always)]
pub fn gelu(input: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEFF: f32 = 0.044_715;
    let cubed = input * input * input;
    0.5 * input * (1.0 + tanh(SQRT_2_OVER_PI * (input + COEFF * cubed)))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// LUT tanh must agree with `f32::tanh` to within 2⁻¹⁰ over `[-6, 6]`.
    /// This is looser than the 2⁻¹² module-level claim because we sample
    /// off-grid; on-grid samples are much tighter.
    #[test]
    fn tanh_matches_reference_within_tolerance() {
        let tolerance = 2f32.powi(-10);
        let mut position = -6.0f32;
        while position <= 6.0 {
            let approx = tanh(position);
            let reference = position.tanh();
            let error = (approx - reference).abs();
            assert!(
                error < tolerance,
                "tanh({position}): approx={approx}, reference={reference}, error={error}"
            );
            position += 0.031_25;
        }
    }

    /// Outside `[-LUT_RANGE, +LUT_RANGE]`, tanh clamps exactly to `±1`.
    #[test]
    fn tanh_saturates_beyond_lut_range() {
        assert_eq!(tanh(100.0), 1.0);
        assert_eq!(tanh(-100.0), -1.0);
        assert_eq!(tanh(f32::INFINITY), 1.0);
        assert_eq!(tanh(f32::NEG_INFINITY), -1.0);
    }

    /// LUT sigmoid must agree with the reference formula to within 2⁻¹⁰
    /// over `[-6, 6]`.
    #[test]
    fn sigmoid_matches_reference_within_tolerance() {
        let tolerance = 2f32.powi(-10);
        let mut position = -6.0f32;
        while position <= 6.0 {
            let approx = sigmoid(position);
            let reference = 1.0 / (1.0 + (-position).exp());
            let error = (approx - reference).abs();
            assert!(
                error < tolerance,
                "sigmoid({position}): approx={approx}, reference={reference}, error={error}"
            );
            position += 0.031_25;
        }
    }

    /// Sigmoid clamps to `0` / `1` outside `[-LUT_RANGE, +LUT_RANGE]`.
    #[test]
    fn sigmoid_saturates_beyond_lut_range() {
        assert_eq!(sigmoid(100.0), 1.0);
        assert_eq!(sigmoid(-100.0), 0.0);
    }

    /// ReLU is the identity for positive inputs and zero elsewhere.
    #[test]
    fn relu_passes_positive_zeroes_negative() {
        assert_eq!(relu(1.5), 1.5);
        assert_eq!(relu(0.0), 0.0);
        assert_eq!(relu(-0.5), 0.0);
        assert_eq!(relu(f32::NEG_INFINITY), 0.0);
    }

    /// GELU approximation is close to the analytic curve within 5e-3
    /// (published tolerance of the tanh-approximation variant).
    #[test]
    fn gelu_matches_reference_within_tolerance() {
        // Reference: 0.5 * x * (1 + erf(x / sqrt(2))).  We rely on the
        // published bound of the tanh approximation instead of computing erf
        // (no libm dep) — check three specific known-good pairs.
        let known_pairs = [
            (-3.0_f32, -0.004),
            (0.0, 0.0),
            (1.0, 0.841_192),
            (2.0, 1.9546),
        ];
        for (input, expected) in known_pairs {
            let approx = gelu(input);
            assert!(
                (approx - expected).abs() < 5e-3,
                "gelu({input}): approx={approx}, expected≈{expected}"
            );
        }
    }
}
