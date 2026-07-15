//! Per-bin spectral gate (downward expander).
//!
//! Each bin whose magnitude sits below `threshold` is attenuated toward a
//! floor set by `range` dB.  To avoid the chattering a hard per-frame decision
//! would cause, the *gain* is smoothed with separate attack (opening) and
//! release (closing) one-pole coefficients — the standard gate envelope, but
//! applied independently per frequency bin so tonal content survives while
//! broadband hiss between notes is squelched.
//!
//! The coefficients are expressed at the STFT **frame rate** (`fs / hop`),
//! because the gate updates once per analysis frame, not once per sample.

use realfft::num_complex::Complex;
use rt_fft_convolver::db_to_linear;

/// Spectral-gate controls (Roadmap-04 Phase 2 "Gate controls").
#[derive(Clone, Copy, Debug)]
pub struct GateParams {
    /// Whether the operator runs at all.
    pub enabled: bool,
    /// Level below which a bin is gated, in dBFS.
    pub threshold_db: f32,
    /// Maximum attenuation of a closed bin, in dB (the expander floor).
    pub range_db: f32,
    /// Gain-open time constant, in milliseconds.
    pub attack_ms: f32,
    /// Gain-close time constant, in milliseconds.
    pub release_ms: f32,
}

impl Default for GateParams {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold_db: -60.0,
            range_db: 24.0,
            attack_ms: 5.0,
            release_ms: 100.0,
        }
    }
}

/// Per-bin gate with smoothed gain envelopes.
#[derive(Debug)]
pub struct SpectralGate {
    /// STFT frame rate, `fs / hop`, for converting ms → one-pole coefficients.
    frame_rate: f32,
    /// Current smoothed gain per bin.
    gain_env: Vec<f32>,
}

impl SpectralGate {
    /// Build a gate for `num_bins` bins at the given STFT `frame_rate`
    /// (`sample_rate / hop`).
    ///
    /// **Allocates** — off-RT only.
    pub fn new(num_bins: usize, frame_rate: f32) -> Self {
        Self {
            frame_rate: frame_rate.max(1.0),
            gain_env: vec![1.0; num_bins],
        }
    }

    /// Reset every bin's gain envelope to fully open.
    #[inline]
    pub fn reset(&mut self) {
        self.gain_env.fill(1.0);
    }

    /// Apply the gate to `spectrum` in place.
    ///
    /// # Real-time safety
    /// No allocation.  `spectrum.len()` must equal `num_bins`.
    pub fn process(&mut self, spectrum: &mut [Complex<f32>], params: &GateParams) {
        if !params.enabled {
            return;
        }
        debug_assert_eq!(spectrum.len(), self.gain_env.len());

        let threshold = db_to_linear(params.threshold_db);
        let floor = db_to_linear(-params.range_db);
        let attack = one_pole_coeff(params.attack_ms, self.frame_rate);
        let release = one_pole_coeff(params.release_ms, self.frame_rate);

        for (bin, env) in spectrum.iter_mut().zip(self.gain_env.iter_mut()) {
            let mag = (bin.re * bin.re + bin.im * bin.im).sqrt();
            // Target: fully open above threshold, floored below it.
            let target = if mag >= threshold { 1.0 } else { floor };
            // Opening (target > env) uses the attack coefficient; closing uses
            // release.  `coeff` is the retained fraction of the old value.
            let coeff = if target > *env { attack } else { release };
            *env = target + coeff * (*env - target);
            bin.re *= *env;
            bin.im *= *env;
        }
    }
}

/// One-pole smoothing coefficient (retained fraction per frame) for a time
/// constant `tau_ms` at `frame_rate` frames per second.
///
/// `coeff = exp(−1 / (tau_seconds · frame_rate))`; a zero/instant time
/// constant yields `0.0` (no smoothing).
fn one_pole_coeff(tau_ms: f32, frame_rate: f32) -> f32 {
    let tau_s = tau_ms * 1e-3;
    if tau_s <= 0.0 {
        return 0.0;
    }
    (-1.0 / (tau_s * frame_rate)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on(threshold_db: f32) -> GateParams {
        GateParams {
            enabled: true,
            threshold_db,
            range_db: 24.0,
            attack_ms: 0.0, // instant for deterministic tests
            release_ms: 0.0,
        }
    }

    #[test]
    fn disabled_passes_through() {
        let mut g = SpectralGate::new(2, 93.75);
        let mut spec = vec![Complex::new(0.001, 0.0); 2];
        let before = spec.clone();
        g.process(
            &mut spec,
            &GateParams {
                enabled: false,
                ..on(-20.0)
            },
        );
        assert_eq!(spec, before);
    }

    #[test]
    fn loud_bin_passes_quiet_bin_gated() {
        let mut g = SpectralGate::new(2, 93.75);
        // bin0 loud (0 dB), bin1 quiet (−80 dB), threshold −40 dB.
        let mut spec = vec![Complex::new(1.0, 0.0), Complex::new(1e-4, 0.0)];
        g.process(&mut spec, &on(-40.0));
        assert!((spec[0].norm() - 1.0).abs() < 1e-6, "loud bin should pass");
        let expected_floor = 1e-4 * db_to_linear(-24.0);
        assert!(
            (spec[1].norm() - expected_floor).abs() < 1e-8,
            "quiet bin should be floored"
        );
    }

    #[test]
    fn attack_release_coeff_bounds() {
        // Instant → 0; finite tau → in (0, 1).
        assert_eq!(one_pole_coeff(0.0, 93.75), 0.0);
        let c = one_pole_coeff(100.0, 93.75);
        assert!(c > 0.0 && c < 1.0, "coeff {c} out of range");
    }
}
