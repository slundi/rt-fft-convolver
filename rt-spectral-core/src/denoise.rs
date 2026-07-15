//! Wiener denoise with a decision-directed (Ephraim–Malah) a-priori SNR.
//!
//! For each bin the classic Wiener gain is `G = ξ / (1 + ξ)`, where `ξ` is the
//! *a-priori* SNR — the SNR we expect the clean signal to have.  Estimating `ξ`
//! naively from the current frame (`γ − 1`, the *a-posteriori* SNR minus one)
//! makes the gain jump around with every random noise fluctuation, and those
//! jumps are audible as *musical noise* (fluttering tones).
//!
//! The **decision-directed** estimator (Ephraim & Malah, 1984) fixes this by
//! smoothing `ξ` across frames:
//!
//! ```text
//!   γ(k)  = |Y(k)|² / N(k)                       (a-posteriori SNR)
//!   ξ(k)  = α · |X̂_prev(k)|² / N(k)
//!         + (1 − α) · max(γ(k) − 1, 0)           (a-priori SNR, DD)
//!   G(k)  = ξ(k) / (1 + ξ(k))                     (Wiener gain)
//! ```
//!
//! with `α ≈ 0.98`.  `|X̂_prev|²` is the previous frame's *estimated clean*
//! power, i.e. `(G_prev · |Y_prev|)²`.  Heavy smoothing (`α` near 1) is what
//! suppresses the musical noise.

use realfft::num_complex::Complex;
use rt_fft_convolver::db_to_linear;

use crate::profile::NoiseProfile;

/// Denoise controls (Roadmap-04 Phase 2 "Denoise controls").
#[derive(Clone, Copy, Debug)]
pub struct DenoiseParams {
    /// Whether the operator runs at all.
    pub enabled: bool,
    /// Maximum attenuation applied to a bin, in dB (0–24).  The Wiener gain is
    /// floored at `10^(−reduction/20)` so residual noise is an attenuated bed,
    /// not a hole (which would sound like musical noise / water).
    pub reduction_db: f32,
    /// How far above the learned profile a bin must sit before it is treated
    /// as signal, in dB (−30…+30).  Positive values are more aggressive
    /// (treat the noise as louder than measured).
    pub sensitivity_db: f32,
}

impl Default for DenoiseParams {
    fn default() -> Self {
        Self {
            enabled: false,
            reduction_db: 12.0,
            sensitivity_db: 0.0,
        }
    }
}

/// Per-bin decision-directed Wiener denoiser.
///
/// Holds one word of per-bin state (the previous frame's estimated clean
/// power); built off-RT, run allocation-free on the audio thread.
#[derive(Debug)]
pub struct Denoiser {
    /// Decision-directed smoothing factor `α`.
    alpha: f32,
    /// Previous frame's estimated clean power, `|X̂_prev(k)|²`, per bin.
    prev_clean_power: Vec<f32>,
}

/// Floor for the a-priori SNR so the gain never collapses fully to zero (which
/// re-introduces musical noise); ≈ −25 dB.
const XI_MIN: f32 = 3.16e-3;
/// Absolute floor on the effective noise power so `γ` stays finite when the
/// learned profile has an exactly-zero bin.
const NOISE_FLOOR: f32 = 1e-20;

impl Denoiser {
    /// Build a denoiser for `num_bins` spectrum bins with smoothing `alpha`
    /// (≈ 0.98 is the usual musical-noise-free choice; clamped to `[0, 1)`).
    ///
    /// **Allocates** — off-RT only.
    pub fn new(num_bins: usize, alpha: f32) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 0.999),
            prev_clean_power: vec![0.0; num_bins],
        }
    }

    /// Clear the inter-frame SNR memory.
    #[inline]
    pub fn reset(&mut self) {
        self.prev_clean_power.fill(0.0);
    }

    /// Apply the Wiener gain to `spectrum` in place.
    ///
    /// `depth` in `[0, 1]` scales how much attenuation is applied (the
    /// transient soft-bypass, see [`crate::transient`]): `1.0` is full
    /// denoise, `0.0` is pass-through.
    ///
    /// # Real-time safety
    /// No allocation, no lock.  `noise.num_bins()` must equal `spectrum.len()`.
    pub fn process(
        &mut self,
        spectrum: &mut [Complex<f32>],
        params: &DenoiseParams,
        noise: &NoiseProfile,
        depth: f32,
    ) {
        if !params.enabled {
            return;
        }
        debug_assert_eq!(spectrum.len(), self.prev_clean_power.len());
        debug_assert_eq!(spectrum.len(), noise.num_bins());

        // `sensitivity_db` raises/lowers the effective noise power; work in the
        // power domain, hence dB/10.
        let sensitivity_scale = db_to_linear(params.sensitivity_db * 0.5); // 10^(dB/10)
        let gain_floor = db_to_linear(-params.reduction_db);
        let depth = depth.clamp(0.0, 1.0);
        let noise_pow = noise.power();

        for ((bin, &n), prev_clean) in spectrum
            .iter_mut()
            .zip(noise_pow)
            .zip(self.prev_clean_power.iter_mut())
        {
            let y_pow = bin.re * bin.re + bin.im * bin.im;
            let n_eff = (n * sensitivity_scale).max(NOISE_FLOOR);

            // A-posteriori and decision-directed a-priori SNR.
            let gamma = y_pow / n_eff;
            let xi = (self.alpha * (*prev_clean / n_eff)
                + (1.0 - self.alpha) * (gamma - 1.0).max(0.0))
            .max(XI_MIN);

            // Wiener gain, floored, then blended by the transient depth.
            let g = (xi / (1.0 + xi)).max(gain_floor);
            let g_eff = 1.0 - depth * (1.0 - g);

            bin.re *= g_eff;
            bin.im *= g_eff;

            // Feed the *applied* clean power estimate to the next frame.
            *prev_clean = g_eff * g_eff * y_pow;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(reduction_db: f32) -> DenoiseParams {
        DenoiseParams {
            enabled: true,
            reduction_db,
            sensitivity_db: 0.0,
        }
    }

    /// Disabled operator is a no-op.
    #[test]
    fn disabled_passes_through() {
        let mut d = Denoiser::new(4, 0.98);
        let profile = NoiseProfile::from_power(vec![1.0; 4]);
        let mut spec = vec![Complex::new(0.5, 0.0); 4];
        let before = spec.clone();
        d.process(
            &mut spec,
            &DenoiseParams {
                enabled: false,
                ..params(12.0)
            },
            &profile,
            1.0,
        );
        assert_eq!(spec, before);
    }

    /// A bin far above the noise floor keeps almost all its magnitude.
    #[test]
    fn signal_bin_is_preserved() {
        let mut d = Denoiser::new(1, 0.98);
        let profile = NoiseProfile::from_power(vec![1e-4]); // very low noise
        let mut spec = vec![Complex::new(1.0, 0.0)]; // strong signal
        // A couple of frames to let the DD estimate settle.
        for _ in 0..4 {
            spec[0] = Complex::new(1.0, 0.0);
            d.process(&mut spec, &params(24.0), &profile, 1.0);
        }
        assert!(
            spec[0].norm() > 0.99,
            "signal attenuated to {}",
            spec[0].norm()
        );
    }

    /// A bin sitting at the noise floor is pushed down toward — but not below —
    /// the reduction floor.
    #[test]
    fn noise_bin_is_attenuated_to_floor() {
        let mut d = Denoiser::new(1, 0.98);
        let noise_power = 1.0f32;
        let profile = NoiseProfile::from_power(vec![noise_power]);
        let floor = db_to_linear(-12.0);
        let mut spec = vec![Complex::new(noise_power.sqrt(), 0.0)];
        let mut last = 1.0;
        for _ in 0..16 {
            spec[0] = Complex::new(noise_power.sqrt(), 0.0); // magnitude == noise
            d.process(&mut spec, &params(12.0), &profile, 1.0);
            last = spec[0].norm();
        }
        // Attenuated well below unity, but not past the −12 dB floor.
        assert!(last < 0.75, "expected attenuation, got {last}");
        assert!(
            last >= floor - 1e-3,
            "dropped below floor: {last} < {floor}"
        );
    }

    /// `depth = 0` bypasses attenuation even with a loud noise profile.
    #[test]
    fn zero_depth_bypasses() {
        let mut d = Denoiser::new(1, 0.98);
        let profile = NoiseProfile::from_power(vec![1.0]);
        let mut spec = vec![Complex::new(1.0, 0.0)];
        d.process(&mut spec, &params(24.0), &profile, 0.0);
        assert!((spec[0].norm() - 1.0).abs() < 1e-6);
    }
}
