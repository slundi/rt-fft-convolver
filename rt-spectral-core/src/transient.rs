//! Spectral-flux transient detection driving a soft-bypass "depth".
//!
//! Denoise and compression smear attacks (pick transients, drum hits) because
//! they act on a whole analysis frame.  Phase 0 chose to detect onsets and
//! *crossfade the processing depth toward unity* for the offending frames
//! rather than to hard-bypass, which would click.
//!
//! Detection uses **spectral flux** — the sum of positive bin-magnitude
//! increases between successive frames — which is nearly free because the
//! magnitudes are already computed by the denoiser.  The flux is compared
//! against a slow running average; when it spikes above the average the
//! returned `depth` dips toward `0` (bypass) and then recovers.
//!
//! ```text
//!   SF = Σ_k max(|Y(k)| − |Y_prev(k)|, 0)
//!   onset if SF > mean · (1 + sensitivity)
//! ```

use realfft::num_complex::Complex;

/// Transient-detector controls.
#[derive(Clone, Copy, Debug)]
pub struct TransientParams {
    /// Whether onset detection runs; when `false`, [`TransientDetector::depth`]
    /// is always `1.0` (full processing).
    pub enabled: bool,
    /// How far above the running-average flux counts as an onset.  Higher =
    /// less sensitive (fewer frames bypassed).  Typical `0.5…4.0`.
    pub sensitivity: f32,
}

impl Default for TransientParams {
    fn default() -> Self {
        Self {
            enabled: true,
            sensitivity: 1.5,
        }
    }
}

/// Running spectral-flux onset detector.
#[derive(Debug)]
pub struct TransientDetector {
    /// Previous frame's per-bin magnitudes.
    prev_mag: Vec<f32>,
    /// Slow running average of the flux (the adaptive threshold basis).
    mean_flux: f32,
    /// Current processing depth in `[0, 1]`; dips on an onset, recovers each
    /// frame.
    depth: f32,
    /// Whether `prev_mag` holds a real previous frame yet.
    primed: bool,
}

/// One-pole coefficient for the slow flux average (retained fraction/frame).
const MEAN_SMOOTH: f32 = 0.95;
/// Per-frame recovery of the depth back toward 1.0 after a dip.
const DEPTH_RECOVERY: f32 = 0.25;

impl TransientDetector {
    /// Build a detector for `num_bins` spectrum bins.
    ///
    /// **Allocates** — off-RT only.
    pub fn new(num_bins: usize) -> Self {
        Self {
            prev_mag: vec![0.0; num_bins],
            mean_flux: 0.0,
            depth: 1.0,
            primed: false,
        }
    }

    /// Reset the running statistics and depth.
    #[inline]
    pub fn reset(&mut self) {
        self.prev_mag.fill(0.0);
        self.mean_flux = 0.0;
        self.depth = 1.0;
        self.primed = false;
    }

    /// The processing depth from the *most recent* [`Self::observe`] call:
    /// `1.0` = full processing, `0.0` = bypass.
    #[inline]
    pub fn depth(&self) -> f32 {
        self.depth
    }

    /// Update the detector from the current analysis `spectrum` and return the
    /// new depth.  Call once per frame **before** the depth-scaled operators.
    ///
    /// # Real-time safety
    /// No allocation.  `spectrum.len()` must equal `num_bins`.
    pub fn observe(&mut self, spectrum: &[Complex<f32>], params: &TransientParams) -> f32 {
        debug_assert_eq!(spectrum.len(), self.prev_mag.len());

        if !params.enabled {
            // Still track magnitudes so re-enabling doesn't spike, but hold
            // depth open.
            for (bin, prev) in spectrum.iter().zip(self.prev_mag.iter_mut()) {
                *prev = (bin.re * bin.re + bin.im * bin.im).sqrt();
            }
            self.depth = 1.0;
            return 1.0;
        }

        // Accumulate positive flux and refresh the magnitude memory.
        let mut flux = 0.0f32;
        for (bin, prev) in spectrum.iter().zip(self.prev_mag.iter_mut()) {
            let mag = (bin.re * bin.re + bin.im * bin.im).sqrt();
            flux += (mag - *prev).max(0.0);
            *prev = mag;
        }

        if !self.primed {
            // First frame: seed the average, no onset decision.
            self.mean_flux = flux;
            self.primed = true;
            self.depth = 1.0;
            return 1.0;
        }

        let threshold = self.mean_flux * (1.0 + params.sensitivity.max(0.0));
        if flux > threshold && threshold > 0.0 {
            // Onset: dip the depth proportionally to how far over threshold we
            // are, clamped so a big hit fully bypasses.
            let over = (flux / threshold - 1.0).clamp(0.0, 1.0);
            self.depth = (1.0 - over).min(self.depth);
        } else {
            // No onset: recover toward full processing.
            self.depth += DEPTH_RECOVERY * (1.0 - self.depth);
        }

        // Update the slow average last so the current spike doesn't mask
        // itself.
        self.mean_flux = MEAN_SMOOTH * self.mean_flux + (1.0 - MEAN_SMOOTH) * flux;
        self.depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mag_spectrum(mag: f32, bins: usize) -> Vec<Complex<f32>> {
        vec![Complex::new(mag, 0.0); bins]
    }

    #[test]
    fn disabled_holds_depth_open() {
        let mut t = TransientDetector::new(4);
        let params = TransientParams {
            enabled: false,
            sensitivity: 1.0,
        };
        for _ in 0..5 {
            assert_eq!(t.observe(&mag_spectrum(1.0, 4), &params), 1.0);
        }
    }

    #[test]
    fn steady_state_stays_open() {
        let mut t = TransientDetector::new(8);
        let params = TransientParams::default();
        // Constant magnitude → zero flux → depth stays at 1.
        for _ in 0..10 {
            let d = t.observe(&mag_spectrum(0.5, 8), &params);
            assert!(d > 0.99, "steady depth dipped to {d}");
        }
    }

    #[test]
    fn sudden_onset_dips_depth() {
        let mut t = TransientDetector::new(8);
        let params = TransientParams {
            enabled: true,
            sensitivity: 1.0,
        };
        // Establish a quiet baseline.
        for _ in 0..6 {
            t.observe(&mag_spectrum(0.01, 8), &params);
        }
        // A loud frame is a clear onset → depth must drop below 1.
        let d = t.observe(&mag_spectrum(1.0, 8), &params);
        assert!(d < 0.9, "onset did not dip depth (got {d})");
    }
}
