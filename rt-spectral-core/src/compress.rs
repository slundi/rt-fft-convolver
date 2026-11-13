//! Per-bin spectral compressor (static soft-knee curve).
//!
//! Each bin's magnitude is run through the standard feed-forward compressor
//! gain computer in the dB domain and the resulting gain is applied to the
//! complex bin (phase preserved).  Compressing *per bin* rather than on the
//! broadband level tames resonances and harsh partials independently — the
//! spectral-compressor behaviour of tools like Pro-DS / soothe-style
//! processors.
//!
//! Phase 0 settled on a **static** curve (no attack/release) for Phase 1; a
//! per-bin envelope follower is a straightforward later addition.
//!
//! ```text
//!   below knee:   out = in
//!   within knee:  out = in + (1/R − 1)·(in − T + W/2)² / (2W)
//!   above knee:   out = T + (in − T)/R
//! ```

use realfft::num_complex::Complex;
use rt_fft_convolver::{db_to_linear, linear_to_db};

/// Spectral-compressor controls (Roadmap-04 Phase 2 "Compressor controls").
#[derive(Clone, Copy, Debug)]
pub struct CompressParams {
    /// Whether the operator runs at all.
    pub enabled: bool,
    /// Knee centre threshold, in dBFS.
    pub threshold_db: f32,
    /// Compression ratio (`>= 1`); `1.0` is a no-op.
    pub ratio: f32,
    /// Soft-knee width, in dB (`0` = hard knee).
    pub knee_db: f32,
    /// Make-up gain applied after compression, in dB.
    pub makeup_db: f32,
}

impl Default for CompressParams {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold_db: -24.0,
            ratio: 2.0,
            knee_db: 6.0,
            makeup_db: 0.0,
        }
    }
}

/// Stateless per-bin spectral compressor.
///
/// Carries no per-bin state (the curve is static), so it is a zero-sized
/// marker; it exists as a type for symmetry with the other operators and to
/// keep the engine's pipeline uniform.
#[derive(Debug, Default)]
pub struct SpectralCompressor;

impl SpectralCompressor {
    /// A new compressor (no state to allocate).
    pub fn new() -> Self {
        Self
    }

    /// Apply the compression curve to `spectrum` in place.
    ///
    /// `depth` in `[0, 1]` scales the gain reduction (transient soft-bypass):
    /// `1.0` full compression, `0.0` pass-through.
    ///
    /// # Real-time safety
    /// No allocation.
    pub fn process(&self, spectrum: &mut [Complex<f32>], params: &CompressParams, depth: f32) {
        if !params.enabled || params.ratio <= 1.0 {
            // Make-up is still meaningful with ratio 1, but skip the whole
            // stage when disabled; a unity ratio with make-up is niche and the
            // user can use Output Gain instead.
            if !params.enabled {
                return;
            }
        }
        let ratio = params.ratio.max(1.0);
        let knee = params.knee_db.max(0.0);
        let makeup = db_to_linear(params.makeup_db);
        let depth = depth.clamp(0.0, 1.0);

        for bin in spectrum.iter_mut() {
            let mag = (bin.re * bin.re + bin.im * bin.im).sqrt();
            if mag <= 0.0 {
                // Silent bin: only make-up applies (and it scales zero anyway).
                continue;
            }
            let in_db = linear_to_db(mag);
            let out_db = gain_curve(in_db, params.threshold_db, ratio, knee);
            // Gain reduction (<= 0 dB) blended by depth, plus full make-up.
            let reduction_db = (out_db - in_db) * depth;
            let gain = db_to_linear(reduction_db) * makeup;
            bin.re *= gain;
            bin.im *= gain;
        }
    }
}

/// Soft-knee compressor output level (dB) for an input level `in_db`.
fn gain_curve(in_db: f32, threshold_db: f32, ratio: f32, knee_db: f32) -> f32 {
    let over = in_db - threshold_db;
    if knee_db > 0.0 && (2.0 * over).abs() <= knee_db {
        // Quadratic interpolation across the knee.
        let x = over + knee_db * 0.5;
        in_db + (1.0 / ratio - 1.0) * x * x / (2.0 * knee_db)
    } else if over <= -knee_db * 0.5 {
        in_db // below the knee: unaffected
    } else {
        threshold_db + over / ratio // above the knee: compressed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on() -> CompressParams {
        CompressParams {
            enabled: true,
            threshold_db: -20.0,
            ratio: 4.0,
            knee_db: 0.0, // hard knee for exact arithmetic
            makeup_db: 0.0,
        }
    }

    #[test]
    fn disabled_passes_through() {
        let c = SpectralCompressor::new();
        let mut spec = vec![Complex::new(1.0, 0.0)];
        let before = spec.clone();
        c.process(
            &mut spec,
            &CompressParams {
                enabled: false,
                ..on()
            },
            1.0,
        );
        assert_eq!(spec, before);
    }

    #[test]
    fn below_threshold_is_untouched() {
        let c = SpectralCompressor::new();
        // −40 dBFS bin, threshold −20, hard knee → below → unchanged.
        let mag = db_to_linear(-40.0);
        let mut spec = vec![Complex::new(mag, 0.0)];
        c.process(&mut spec, &on(), 1.0);
        assert!((spec[0].norm() - mag).abs() < 1e-6);
    }

    #[test]
    fn above_threshold_is_compressed_by_ratio() {
        let c = SpectralCompressor::new();
        // 0 dBFS input, threshold −20, ratio 4 → out = −20 + 20/4 = −15 dB.
        let mut spec = vec![Complex::new(1.0, 0.0)];
        c.process(&mut spec, &on(), 1.0);
        let out_db = linear_to_db(spec[0].norm());
        assert!((out_db - (-15.0)).abs() < 1e-3, "out {out_db} dB, want −15");
    }

    #[test]
    fn zero_depth_bypasses_reduction() {
        let c = SpectralCompressor::new();
        let mut spec = vec![Complex::new(1.0, 0.0)];
        c.process(&mut spec, &on(), 0.0);
        assert!((spec[0].norm() - 1.0).abs() < 1e-6);
    }
}
