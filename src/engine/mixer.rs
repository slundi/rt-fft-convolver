//! Gain staging and dry/wet mixing for convolution signal paths.
//!
//! # Normalization
//!
//! Convolving a signal with a long reverb IR accumulates a large amount of
//! energy in the output — a 2-second cab IR at 48 kHz is ~96 000 taps, and
//! their combined contribution can push the output well above 0 dBFS.
//!
//! [`Mixer`] computes a **normalization gain** from the IR at construction
//! time using the reciprocal of its L2-norm (energy):
//!
//! ```text
//! ir_gain = 1 / sqrt( Σ h[n]² )
//! ```
//!
//! This ensures that a unit-energy input signal convolved with the normalized
//! IR produces a unit-energy output, preventing systematic level inflation.
//! If the IR is silent (all zeros) the gain defaults to 1.0 to avoid
//! a division-by-zero.
//!
//! # Dry / Wet
//!
//! The `dry_wet` parameter (0.0 = fully dry, 1.0 = fully wet) uses an
//! **equal-power crossfade** so that the perceived loudness stays constant
//! as the knob is swept:
//!
//! ```text
//! θ          = dry_wet × π/2
//! dry_gain   = cos(θ)              ← 1 → 0 as wet increases
//! wet_gain   = sin(θ) × ir_gain   ← 0 → ir_gain as wet increases
//! ```
//!
//! `ir_gain` is folded into `wet_gain` at `set_dry_wet` time so that
//! [`process`] needs only two multiplies per sample — no branches.
//!
//! [`process`]: Mixer::process

/// Gain staging and dry/wet mixer for a convolution signal path.
///
/// Instantiate once on a **non-real-time** thread, then call [`process`] from
/// the audio render thread for every block.
///
/// ```text
/// output[n] = dry[n] × dry_gain + wet[n] × wet_gain
/// ```
///
/// where:
/// - `dry_gain = cos(dry_wet × π/2)`
/// - `wet_gain = sin(dry_wet × π/2) × ir_gain`
/// - `ir_gain  = 1 / ‖ir‖₂`
///
/// [`process`]: Mixer::process
pub struct Mixer {
    /// Reciprocal of the IR's L2-norm, computed once at construction.
    ir_gain: f32,

    /// Equal-power dry coefficient: `cos(dry_wet × π/2)`.
    dry_gain: f32,

    /// Equal-power wet coefficient already folded with `ir_gain`:
    /// `sin(dry_wet × π/2) × ir_gain`.
    wet_gain: f32,
}

impl Mixer {
    /// Create a mixer from an IR and an initial dry/wet setting.
    ///
    /// `ir` is the **full** impulse response used to derive the normalization
    /// gain.  Pass `&[]` or a silent IR to skip normalization (gain = 1.0).
    ///
    /// `dry_wet` is clamped to `0.0..=1.0`.
    ///
    /// **Allocates** — call from a non-real-time thread.
    pub fn new(ir: &[f32], dry_wet: f32) -> Self {
        // L2-norm of the IR: sqrt( Σ h[n]² ).
        // Squaring first avoids repeated sqrt calls and stays numerically stable.
        let energy: f32 = ir.iter().map(|&x| x * x).sum();
        // Guard against silent / empty IR to avoid a divide-by-zero.
        let ir_gain = if energy > 0.0 {
            1.0 / energy.sqrt()
        } else {
            1.0
        };

        let mut m = Self {
            ir_gain,
            dry_gain: 1.0,
            wet_gain: 0.0,
        };
        m.set_dry_wet(dry_wet);
        m
    }

    /// Update the dry/wet ratio without reallocating.
    ///
    /// `dry_wet` is clamped to `0.0..=1.0`.
    ///
    /// **Real-time safe** — no allocation.
    pub fn set_dry_wet(&mut self, dry_wet: f32) {
        // Map [0, 1] → [0, π/2] for the equal-power crossfade.
        let theta = dry_wet.clamp(0.0, 1.0) * std::f32::consts::FRAC_PI_2;
        self.dry_gain = theta.cos();
        // Fold ir_gain into wet_gain so process() avoids an extra multiply.
        self.wet_gain = theta.sin() * self.ir_gain;
    }

    /// The normalization gain derived from the IR's L2-norm (`1 / ‖ir‖₂`).
    ///
    /// This is `1.0` when the IR is silent or empty.
    #[inline]
    pub fn ir_gain(&self) -> f32 {
        self.ir_gain
    }

    /// The current equal-power dry coefficient (`cos(dry_wet × π/2)`).
    #[inline]
    pub fn dry_gain(&self) -> f32 {
        self.dry_gain
    }

    /// The current equal-power wet coefficient already scaled by `ir_gain`
    /// (`sin(dry_wet × π/2) × ir_gain`).
    #[inline]
    pub fn wet_gain(&self) -> f32 {
        self.wet_gain
    }

    /// Mix `dry` and `wet` into `output`.
    ///
    /// ```text
    /// output[n] = dry[n] × dry_gain + wet[n] × wet_gain
    /// ```
    ///
    /// All three slices must have the same length.
    ///
    /// **Real-time safe** — no allocation, no locking, no panics in release
    /// builds.
    pub fn process(&self, dry: &[f32], wet: &[f32], output: &mut [f32]) {
        debug_assert_eq!(dry.len(), wet.len(), "dry and wet length mismatch");
        debug_assert_eq!(dry.len(), output.len(), "dry and output length mismatch");

        for ((&d, &w), y) in dry.iter().zip(wet.iter()).zip(output.iter_mut()) {
            *y = d * self.dry_gain + w * self.wet_gain;
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    fn assert_approx(label: &str, got: f32, expected: f32) {
        assert!(
            (got - expected).abs() < EPSILON,
            "{label}: expected {expected}, got {got}"
        );
    }

    // ── Normalization gain ────────────────────────────────────────────────────

    /// A unit-impulse IR has L2-norm = 1, so ir_gain must equal 1.0.
    #[test]
    fn unit_impulse_ir_has_gain_one() {
        let ir = [1.0f32, 0.0, 0.0, 0.0];
        let m = Mixer::new(&ir, 0.0);
        assert_approx("ir_gain", m.ir_gain(), 1.0);
    }

    /// An IR of [2, 0, 0, 0] has L2-norm = 2, so ir_gain must equal 0.5.
    #[test]
    fn gain_two_ir_has_ir_gain_half() {
        let ir = [2.0f32, 0.0, 0.0, 0.0];
        let m = Mixer::new(&ir, 0.0);
        assert_approx("ir_gain", m.ir_gain(), 0.5);
    }

    /// Silent IR must not produce NaN or Inf — ir_gain defaults to 1.0.
    #[test]
    fn silent_ir_gives_gain_one() {
        let m = Mixer::new(&[0.0f32; 8], 0.5);
        assert_approx("ir_gain", m.ir_gain(), 1.0);
        assert!(m.dry_gain().is_finite());
        assert!(m.wet_gain().is_finite());
    }

    /// Empty IR slice also defaults to ir_gain = 1.0.
    #[test]
    fn empty_ir_gives_gain_one() {
        let m = Mixer::new(&[], 0.0);
        assert_approx("ir_gain", m.ir_gain(), 1.0);
    }

    // ── Equal-power crossfade ─────────────────────────────────────────────────

    /// At dry_wet = 0.0 the output must be fully dry (dry_gain = 1, wet_gain = 0).
    #[test]
    fn fully_dry_passes_dry_only() {
        let ir = [1.0f32, 0.0, 0.0, 0.0];
        let m = Mixer::new(&ir, 0.0);
        assert_approx("dry_gain", m.dry_gain(), 1.0);
        assert_approx("wet_gain", m.wet_gain(), 0.0);

        let dry = [1.0f32, 2.0, 3.0, 4.0];
        let wet = [9.0f32; 4];
        let mut out = [0.0f32; 4];
        m.process(&dry, &wet, &mut out);
        for (&d, &y) in dry.iter().zip(out.iter()) {
            assert_approx("out", y, d);
        }
    }

    /// At dry_wet = 1.0 the output must be fully wet (dry_gain = 0).
    #[test]
    fn fully_wet_passes_wet_only() {
        // Unit-impulse IR → ir_gain = 1.0.
        let ir = [1.0f32, 0.0, 0.0, 0.0];
        let m = Mixer::new(&ir, 1.0);
        assert_approx("dry_gain", m.dry_gain(), 0.0);
        // wet_gain = sin(π/2) × 1.0 = 1.0
        assert_approx("wet_gain", m.wet_gain(), 1.0);

        let dry = [9.0f32; 4];
        let wet = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        m.process(&dry, &wet, &mut out);
        for (&w, &y) in wet.iter().zip(out.iter()) {
            assert_approx("out", y, w);
        }
    }

    /// At dry_wet = 0.5 the equal-power invariant must hold:
    /// dry_gain² + (wet_gain/ir_gain)² ≈ 1.0.
    #[test]
    fn equal_power_invariant_at_half() {
        let ir = [1.0f32, 0.0, 0.0, 0.0];
        let m = Mixer::new(&ir, 0.5);
        // Recover the raw sin component by dividing out ir_gain.
        let sin_part = m.wet_gain() / m.ir_gain();
        let power_sum = m.dry_gain().powi(2) + sin_part.powi(2);
        assert!(
            (power_sum - 1.0).abs() < 1e-6,
            "equal-power invariant violated: {power_sum}"
        );
    }

    /// `set_dry_wet` must update coefficients correctly.
    #[test]
    fn set_dry_wet_updates_gains() {
        let ir = [1.0f32, 0.0, 0.0, 0.0];
        let mut m = Mixer::new(&ir, 0.0);
        assert_approx("dry_gain before", m.dry_gain(), 1.0);

        m.set_dry_wet(1.0);
        assert_approx("dry_gain after", m.dry_gain(), 0.0);
        assert_approx("wet_gain after", m.wet_gain(), 1.0);
    }

    /// Values outside 0..=1 must be clamped, not panic or wrap.
    #[test]
    fn set_dry_wet_clamps_out_of_range() {
        let ir = [1.0f32];
        let mut m = Mixer::new(&ir, 2.0); // clamped to 1.0
        assert_approx("clamped high: dry_gain", m.dry_gain(), 0.0);

        m.set_dry_wet(-0.5); // clamped to 0.0
        assert_approx("clamped low: dry_gain", m.dry_gain(), 1.0);
        assert_approx("clamped low: wet_gain", m.wet_gain(), 0.0);
    }

    // ── process() ────────────────────────────────────────────────────────────

    /// With ir_gain=0.5 and fully wet, the output must be wet × 0.5.
    #[test]
    fn ir_normalization_scales_wet_output() {
        // IR = [2, 0] → L2-norm = 2 → ir_gain = 0.5.
        let ir = [2.0f32, 0.0];
        let m = Mixer::new(&ir, 1.0); // fully wet

        let dry = [0.0f32; 4];
        let wet = [1.0f32; 4];
        let mut out = [0.0f32; 4];
        m.process(&dry, &wet, &mut out);
        for &y in &out {
            assert_approx("out", y, 0.5);
        }
    }

    /// process() must overwrite (not accumulate into) the output slice.
    #[test]
    fn process_overwrites_output() {
        let m = Mixer::new(&[1.0f32], 0.0); // fully dry, ir_gain=1
        let dry = [3.0f32; 4];
        let wet = [0.0f32; 4];
        let mut out = [99.0f32; 4]; // non-zero sentinel
        m.process(&dry, &wet, &mut out);
        for &y in &out {
            assert_approx("out", y, 3.0);
        }
    }
}
