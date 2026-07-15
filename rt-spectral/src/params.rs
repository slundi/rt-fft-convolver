//! Control ranges for the `rt-spectral` plugin and a framework-agnostic
//! clamp helper.
//!
//! The engine and its tests work directly with the core
//! [`SpectralParams`](rt_spectral_core::SpectralParams) struct — this module
//! only owns the *plugin-facing* ranges (the min/max each `nih-plug`
//! `FloatParam`/`IntParam` is built with, mirroring the Phase-2 control list in
//! `docs/Roadmap-04-spectral-processing.md`) plus [`clamp_params`], which pins a
//! host snapshot back into those ranges.  Keeping this free of `nih-plug`
//! (which lives behind the `nih` feature) lets the ranges be unit-tested
//! without the plugin framework.

use rt_spectral_core::hum::MAX_HARMONICS;
use rt_spectral_core::{HumParams, SpectralParams};

/// Longest noise clip the `Learn` capture buffer holds, in milliseconds
/// (Phase 2: "captures 500 – 2000 ms of noise profile").  Sizes the capture
/// buffer allocated off-RT in [`spectral_engine`](crate::spectral_engine).
pub const MAX_LEARN_MS: f32 = 2000.0;

/// `Learn Time` range in milliseconds (Phase 2: 500 – 2000 ms).
pub const LEARN_MS_RANGE: (f32, f32) = (500.0, MAX_LEARN_MS);

// ── Denoise ────────────────────────────────────────────────────────────────
/// `Reduction` range in dB (Phase 2: 0 – 24 dB of maximum attenuation).
pub const REDUCTION_DB_RANGE: (f32, f32) = (0.0, 24.0);
/// Denoise `Sensitivity` range in dB above the learned profile (Phase 2:
/// −30 … +30 dB).
pub const SENSITIVITY_DB_RANGE: (f32, f32) = (-30.0, 30.0);

// ── Gate ───────────────────────────────────────────────────────────────────
/// Gate `Threshold` range in dBFS.
pub const GATE_THRESHOLD_DB_RANGE: (f32, f32) = (-120.0, 0.0);
/// Gate `Range` (maximum attenuation of a closed bin) in dB.
pub const GATE_RANGE_DB_RANGE: (f32, f32) = (0.0, 96.0);
/// Gate/compressor `Attack` time-constant range in milliseconds.
pub const ATTACK_MS_RANGE: (f32, f32) = (0.1, 250.0);
/// Gate/compressor `Release` time-constant range in milliseconds.
pub const RELEASE_MS_RANGE: (f32, f32) = (1.0, 2000.0);

// ── Compressor ─────────────────────────────────────────────────────────────
/// Compressor `Threshold` range in dBFS.
pub const COMP_THRESHOLD_DB_RANGE: (f32, f32) = (-96.0, 0.0);
/// Compressor `Ratio` range (`1.0` = no-op).
pub const RATIO_RANGE: (f32, f32) = (1.0, 20.0);
/// Compressor soft-`Knee` width range in dB.
pub const KNEE_DB_RANGE: (f32, f32) = (0.0, 24.0);
/// Compressor `Makeup` gain range in dB.
pub const MAKEUP_DB_RANGE: (f32, f32) = (-24.0, 24.0);

// ── Hum ────────────────────────────────────────────────────────────────────
/// Hum `Fundamental` range in Hz (covers 50/60 Hz mains plus custom lines).
pub const HUM_FUNDAMENTAL_HZ_RANGE: (f32, f32) = (40.0, 120.0);
/// Hum `Harmonics` count range (1 … [`MAX_HARMONICS`]).
pub const HARMONICS_RANGE: (usize, usize) = (1, MAX_HARMONICS);
/// Hum notch `Q` range (higher = narrower).
pub const HUM_Q_RANGE: (f32, f32) = (1.0, 100.0);

// ── Click ──────────────────────────────────────────────────────────────────
/// Click `Sensitivity` range in robust standard deviations (higher = fewer
/// detections).
pub const CLICK_SENSITIVITY_RANGE: (f32, f32) = (2.0, 12.0);
/// Click `Max Length` (longest repaired run) range in samples.
pub const CLICK_MAX_LEN_RANGE: (usize, usize) = (1, 64);

// ── Global ─────────────────────────────────────────────────────────────────
/// `Mix` (dry/wet) range.
pub const MIX_RANGE: (f32, f32) = (0.0, 1.0);
/// `Output Gain` range in dB.
pub const OUTPUT_GAIN_DB_RANGE: (f32, f32) = (-24.0, 24.0);

/// Convert a `Learn Time` in milliseconds to a sample count at `sample_rate`,
/// clamped to the [`LEARN_MS_RANGE`].
#[inline]
pub fn learn_samples(learn_ms: f32, sample_rate: u32) -> usize {
    let ms = learn_ms.clamp(LEARN_MS_RANGE.0, LEARN_MS_RANGE.1);
    ((sample_rate as f32 / 1000.0) * ms).round() as usize
}

/// Return a copy of `params` with every continuous field clamped into its
/// plugin range.  The enable flags, `bypass`, and the transient defaults pass
/// through untouched.
pub fn clamp_params(mut params: SpectralParams) -> SpectralParams {
    params.denoise.reduction_db = params
        .denoise
        .reduction_db
        .clamp(REDUCTION_DB_RANGE.0, REDUCTION_DB_RANGE.1);
    params.denoise.sensitivity_db = params
        .denoise
        .sensitivity_db
        .clamp(SENSITIVITY_DB_RANGE.0, SENSITIVITY_DB_RANGE.1);

    params.gate.threshold_db = params
        .gate
        .threshold_db
        .clamp(GATE_THRESHOLD_DB_RANGE.0, GATE_THRESHOLD_DB_RANGE.1);
    params.gate.range_db = params
        .gate
        .range_db
        .clamp(GATE_RANGE_DB_RANGE.0, GATE_RANGE_DB_RANGE.1);
    params.gate.attack_ms = params
        .gate
        .attack_ms
        .clamp(ATTACK_MS_RANGE.0, ATTACK_MS_RANGE.1);
    params.gate.release_ms = params
        .gate
        .release_ms
        .clamp(RELEASE_MS_RANGE.0, RELEASE_MS_RANGE.1);

    params.compress.threshold_db = params
        .compress
        .threshold_db
        .clamp(COMP_THRESHOLD_DB_RANGE.0, COMP_THRESHOLD_DB_RANGE.1);
    params.compress.ratio = params.compress.ratio.clamp(RATIO_RANGE.0, RATIO_RANGE.1);
    params.compress.knee_db = params
        .compress
        .knee_db
        .clamp(KNEE_DB_RANGE.0, KNEE_DB_RANGE.1);
    params.compress.makeup_db = params
        .compress
        .makeup_db
        .clamp(MAKEUP_DB_RANGE.0, MAKEUP_DB_RANGE.1);

    params.hum = HumParams {
        enabled: params.hum.enabled,
        fundamental_hz: params
            .hum
            .fundamental_hz
            .clamp(HUM_FUNDAMENTAL_HZ_RANGE.0, HUM_FUNDAMENTAL_HZ_RANGE.1),
        harmonics: params
            .hum
            .harmonics
            .clamp(HARMONICS_RANGE.0, HARMONICS_RANGE.1),
        q: params.hum.q.clamp(HUM_Q_RANGE.0, HUM_Q_RANGE.1),
    };

    params.click.sensitivity = params
        .click
        .sensitivity
        .clamp(CLICK_SENSITIVITY_RANGE.0, CLICK_SENSITIVITY_RANGE.1);
    params.click.max_len = params
        .click
        .max_len
        .clamp(CLICK_MAX_LEN_RANGE.0, CLICK_MAX_LEN_RANGE.1);

    params.mix = params.mix.clamp(MIX_RANGE.0, MIX_RANGE.1);
    params.output_gain_db = params
        .output_gain_db
        .clamp(OUTPUT_GAIN_DB_RANGE.0, OUTPUT_GAIN_DB_RANGE.1);
    params
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_params_are_already_in_range() {
        let defaults = SpectralParams::default();
        assert_eq!(clamp_params(defaults).mix, defaults.mix);
        // Defaults must survive a clamp unchanged (idempotent on valid input).
        let clamped = clamp_params(defaults);
        assert_eq!(clamped.denoise.reduction_db, defaults.denoise.reduction_db);
        assert_eq!(clamped.gate.threshold_db, defaults.gate.threshold_db);
        assert_eq!(clamped.compress.ratio, defaults.compress.ratio);
        assert_eq!(clamped.hum.harmonics, defaults.hum.harmonics);
        assert_eq!(clamped.click.max_len, defaults.click.max_len);
    }

    #[test]
    fn clamp_pulls_out_of_range_fields_in() {
        let mut params = SpectralParams::default();
        params.denoise.reduction_db = 100.0;
        params.denoise.sensitivity_db = -100.0;
        params.gate.threshold_db = 50.0;
        params.compress.ratio = 0.1;
        params.hum.fundamental_hz = 5.0;
        params.hum.harmonics = 999;
        params.click.max_len = 9_999;
        params.mix = 5.0;
        params.output_gain_db = 200.0;

        let clamped = clamp_params(params);
        assert_eq!(clamped.denoise.reduction_db, REDUCTION_DB_RANGE.1);
        assert_eq!(clamped.denoise.sensitivity_db, SENSITIVITY_DB_RANGE.0);
        assert_eq!(clamped.gate.threshold_db, GATE_THRESHOLD_DB_RANGE.1);
        assert_eq!(clamped.compress.ratio, RATIO_RANGE.0);
        assert_eq!(clamped.hum.fundamental_hz, HUM_FUNDAMENTAL_HZ_RANGE.0);
        assert_eq!(clamped.hum.harmonics, HARMONICS_RANGE.1);
        assert_eq!(clamped.click.max_len, CLICK_MAX_LEN_RANGE.1);
        assert_eq!(clamped.mix, MIX_RANGE.1);
        assert_eq!(clamped.output_gain_db, OUTPUT_GAIN_DB_RANGE.1);
    }

    #[test]
    fn learn_samples_clamps_and_scales() {
        // 1000 ms @ 48 kHz → 48 000 samples.
        assert_eq!(learn_samples(1000.0, 48_000), 48_000);
        // Below the minimum is pinned to 500 ms.
        assert_eq!(learn_samples(0.0, 48_000), 24_000);
        // Above the maximum is pinned to 2000 ms.
        assert_eq!(learn_samples(10_000.0, 48_000), 96_000);
    }
}
