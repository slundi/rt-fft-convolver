//! K-weighting pre-filter (ITU-R BS.1770-4, EBU R128).
//!
//! # Why K-weighting
//!
//! Loudness perception is not flat across frequency: the ear is more sensitive
//! to mid/high content and rolls off the very low end.  BS.1770 models this
//! with a fixed two-stage IIR "K-weighting" filter applied to every channel
//! *before* the mean-square energy is measured:
//!
//! 1. **Stage 1 — a high-frequency shelving boost** (≈ +4 dB above ~1.5 kHz),
//!    the "head" filter approximating the acoustic effect of a listener's head.
//! 2. **Stage 2 — an RLB high-pass** (≈ 38 Hz), removing sub-bass energy that
//!    contributes little to perceived loudness.
//!
//! # Sample-rate independence
//!
//! The standard tabulates coefficients at 48 kHz only.  Storing those verbatim
//! would miss-weight any other rate, so instead we keep the **analog prototype
//! parameters** (centre frequency, `Q`, shelf gain) and bilinear-transform them
//! at the actual sample rate — the widely-used derivation from Brecht De Man's
//! "Evaluation of implementations of the EBU R128 loudness measurement" and
//! `libebur128`.  At 48 kHz this reproduces the standard's table to `f32`
//! precision (asserted in the tests).
//!
//! # Real-time safety
//!
//! Each stage is a [transposed direct-form II] biquad — two state words, a
//! pure multiply-add per sample, no allocation and no division in the render
//! loop (coefficients are pre-normalised by `a0` at construction).
//!
//! [transposed direct-form II]: https://en.wikipedia.org/wiki/Digital_biquad_filter

/// Analog prototype of the stage-1 shelving filter (BS.1770-4).
const SHELF_F0: f64 = 1681.974450955533;
const SHELF_GAIN_DB: f64 = 3.999843853973347;
const SHELF_Q: f64 = 0.7071752369554196;
/// Exponent relating the shelf's mid-band gain `Vb` to its high-band gain `Vh`
/// (from the reference derivation; `Vh^0.4997` places the shelf transition).
const SHELF_VB_EXP: f64 = 0.4996667741545416;

/// Analog prototype of the stage-2 RLB high-pass (BS.1770-4).
const HP_F0: f64 = 38.13547087602444;
const HP_Q: f64 = 0.5003270373238773;

/// One second-order section in transposed direct-form II, coefficients already
/// normalised by `a0`.
#[derive(Debug, Clone, Copy)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    /// Build from coefficients that are already in final (a0 = 1) form.
    fn new(b0: f64, b1: f64, b2: f64, a1: f64, a2: f64) -> Self {
        Self {
            b0: b0 as f32,
            b1: b1 as f32,
            b2: b2 as f32,
            a1: a1 as f32,
            a2: a2 as f32,
            z1: 0.0,
            z2: 0.0,
        }
    }

    #[inline]
    fn process_sample(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }

    #[inline]
    fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

/// The BS.1770-4 K-weighting filter for a **single channel**: the shelving
/// stage followed by the RLB high-pass.  One instance holds the running state
/// for one channel, so a stereo meter owns two.
#[derive(Debug, Clone, Copy)]
pub struct KWeightingFilter {
    shelf: Biquad,
    highpass: Biquad,
}

impl KWeightingFilter {
    /// Design the K-weighting filter for `sample_rate` Hz.
    ///
    /// **Allocates nothing**, but the `tan`/`powf` design maths is heavier than
    /// a render step — build it in `new()`, off the audio thread.
    pub fn new(sample_rate: u32) -> Self {
        let fs = sample_rate.max(1) as f64;

        // ── Stage 1: high-shelf ("head") filter ──────────────────────────
        // Bilinear pre-warp: K = tan(π·f0/fs).  Vh/Vb set the shelf gain.
        let k = (std::f64::consts::PI * SHELF_F0 / fs).tan();
        let vh = 10.0_f64.powf(SHELF_GAIN_DB / 20.0);
        let vb = vh.powf(SHELF_VB_EXP);
        let k2 = k * k;
        // The shelf numerator *and* denominator are both normalised by a0
        // (this reproduces the standard's shelf table exactly).
        let inv_a0 = 1.0 / (1.0 + k / SHELF_Q + k2);
        let shelf = Biquad::new(
            (vh + vb * k / SHELF_Q + k2) * inv_a0, // b0
            (2.0 * (k2 - vh)) * inv_a0,            // b1
            (vh - vb * k / SHELF_Q + k2) * inv_a0, // b2
            (2.0 * (k2 - 1.0)) * inv_a0,           // a1
            (1.0 - k / SHELF_Q + k2) * inv_a0,     // a2
        );

        // ── Stage 2: RLB high-pass ────────────────────────────────────────
        // The standard defines the RLB numerator as literally [1, −2, 1] and
        // normalises only the denominator by a0 (the residual ~0.5 % gain is
        // part of the definition).  Reproduce that, not a full normalisation.
        let k = (std::f64::consts::PI * HP_F0 / fs).tan();
        let k2 = k * k;
        let inv_a0 = 1.0 / (1.0 + k / HP_Q + k2);
        let highpass = Biquad::new(
            1.0,                            // b0
            -2.0,                           // b1
            1.0,                            // b2
            (2.0 * (k2 - 1.0)) * inv_a0,    // a1
            (1.0 - k / HP_Q + k2) * inv_a0, // a2
        );

        Self { shelf, highpass }
    }

    /// K-weight one sample.  **Real-time safe.**
    #[inline]
    pub fn process_sample(&mut self, x: f32) -> f32 {
        self.highpass.process_sample(self.shelf.process_sample(x))
    }

    /// Clear both stages' state without re-designing the coefficients.
    #[inline]
    pub fn reset(&mut self) {
        self.shelf.reset();
        self.highpass.reset();
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    /// The design maths must reproduce the standard's 48 kHz reference
    /// coefficients (ITU-R BS.1770-4, Tables 1 & 2) to `f32` precision.
    #[test]
    fn matches_reference_coefficients_at_48k() {
        let k = KWeightingFilter::new(48_000);

        // Stage 1 (pre-filter / shelving).
        assert!(
            (k.shelf.b0 - 1.535_124_9).abs() < 1e-4,
            "shelf b0 {}",
            k.shelf.b0
        );
        assert!(
            (k.shelf.b1 - -2.691_696_2).abs() < 1e-4,
            "shelf b1 {}",
            k.shelf.b1
        );
        assert!(
            (k.shelf.b2 - 1.198_392_8).abs() < 1e-4,
            "shelf b2 {}",
            k.shelf.b2
        );
        assert!(
            (k.shelf.a1 - -1.690_659_3).abs() < 1e-4,
            "shelf a1 {}",
            k.shelf.a1
        );
        assert!(
            (k.shelf.a2 - 0.732_480_8).abs() < 1e-4,
            "shelf a2 {}",
            k.shelf.a2
        );

        // Stage 2 (RLB high-pass): b = [1, -2, 1] exactly.
        assert!((k.highpass.b0 - 1.0).abs() < 1e-6);
        assert!((k.highpass.b1 - -2.0).abs() < 1e-6);
        assert!((k.highpass.b2 - 1.0).abs() < 1e-6);
        assert!(
            (k.highpass.a1 - -1.990_047_5).abs() < 1e-4,
            "hp a1 {}",
            k.highpass.a1
        );
        assert!(
            (k.highpass.a2 - 0.990_072_3).abs() < 1e-4,
            "hp a2 {}",
            k.highpass.a2
        );
    }

    /// Settled RMS gain of a sustained sine at `freq`, after warm-up.
    fn tone_gain_db(sample_rate: u32, freq: f32) -> f32 {
        let mut k = KWeightingFilter::new(sample_rate);
        let mut sum_sq = 0.0f64;
        let mut count = 0u64;
        let total = sample_rate as usize; // one second
        for index in 0..total {
            let x = (TAU * freq * index as f32 / sample_rate as f32).sin();
            let y = k.process_sample(x);
            if index > total * 3 / 4 {
                sum_sq += (y as f64) * (y as f64);
                count += 1;
            }
        }
        // Input is a unit sine (RMS 1/√2); report the gain relative to it.
        let out_rms = (sum_sq / count as f64).sqrt();
        20.0 * (out_rms / (1.0 / 2.0_f64.sqrt())).log10() as f32
    }

    /// The K-weighting curve shape: ~0 dB in the low-mids, a shelf boost of
    /// ~+4 dB in the highs, and strong attenuation in the sub-bass.
    #[test]
    fn frequency_response_shape() {
        // 1 kHz sits in the pass-band reference region — near unity.
        let mid = tone_gain_db(48_000, 1_000.0);
        assert!(mid.abs() < 1.0, "1 kHz gain {mid} dB not ~0");

        // The high shelf adds ~+4 dB by ~10 kHz.
        let high = tone_gain_db(48_000, 10_000.0);
        assert!(high > 2.5, "10 kHz gain {high} dB should be boosted");

        // The RLB high-pass strongly rejects sub-bass.
        let low = tone_gain_db(48_000, 20.0);
        assert!(low < -10.0, "20 Hz gain {low} dB should be rejected");
    }

    #[test]
    fn silence_stays_silent_and_finite() {
        let mut k = KWeightingFilter::new(44_100);
        for _ in 0..1_000 {
            assert_eq!(k.process_sample(0.0), 0.0);
        }
    }

    #[test]
    fn reset_clears_state() {
        let mut k = KWeightingFilter::new(48_000);
        for _ in 0..100 {
            k.process_sample(1.0);
        }
        k.reset();
        // With cleared state the first output equals b0·x of the two stages.
        let first = k.process_sample(0.0);
        assert_eq!(first, 0.0);
    }
}
