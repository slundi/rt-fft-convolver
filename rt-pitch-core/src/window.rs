//! Analysis / synthesis windows for the STFT-based shifters.
//!
//! A phase vocoder applies a window on **both** analysis and synthesis, so the
//! overlap-add of `w_analysis · w_synthesis` across successive frames must sum
//! to a constant for the reconstruction to be free of amplitude modulation —
//! the *Constant OverLap-Add* (COLA) condition.  For the identical Hann window
//! at 75 % overlap (hop = frame / 4) that sum is exactly 1.5; rather than hard-
//! code per-window constants we measure the sum numerically in
//! [`Window::ola_normalisation`] so any window / hop pairing stays correct.

use std::f32::consts::PI;

/// Which window shape to synthesise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowKind {
    /// Hann (raised cosine).  Good side-lobe roll-off, satisfies COLA at
    /// 50 % / 75 % overlap; the phase-vocoder default.
    Hann,
    /// 4-term Blackman-Harris.  Much lower side lobes (≈ −92 dB) at the cost
    /// of a wider main lobe — useful for cleaner harmonic separation on
    /// sustained material.
    BlackmanHarris,
}

/// A pre-computed window and the overlap-add normalisation for a given hop.
///
/// Built once off the audio thread; the sample slice is read-only on the
/// render thread.
#[derive(Clone, Debug)]
pub struct Window {
    /// Window coefficients, length == frame size.
    samples: Vec<f32>,
    /// Which shape was synthesised (kept for introspection / tests).
    kind: WindowKind,
}

impl Window {
    /// Synthesise a `size`-sample window of the given shape.
    ///
    /// **Allocates** — call off the real-time thread.
    ///
    /// # Panics
    /// Panics if `size < 2` (a window needs at least two points to taper).
    pub fn new(kind: WindowKind, size: usize) -> Self {
        assert!(size >= 2, "window size must be >= 2 (got {size})");
        // Periodic (DFT-even) definition — divide by `size`, not `size − 1`.
        // This is what satisfies the Constant OverLap-Add condition for STFT
        // resynthesis; the symmetric variant (`size − 1`) leaves a residual
        // amplitude modulation because its overlap sum is position-dependent.
        let denom = size as f32;
        let samples = (0..size)
            .map(|index| {
                let phase = index as f32 / denom; // 0 ..< 1 across the window
                match kind {
                    WindowKind::Hann => 0.5 - 0.5 * (2.0 * PI * phase).cos(),
                    WindowKind::BlackmanHarris => {
                        // Standard 4-term Blackman-Harris coefficients.
                        const A0: f32 = 0.35875;
                        const A1: f32 = 0.48829;
                        const A2: f32 = 0.14128;
                        const A3: f32 = 0.01168;
                        A0 - A1 * (2.0 * PI * phase).cos() + A2 * (4.0 * PI * phase).cos()
                            - A3 * (6.0 * PI * phase).cos()
                    }
                }
            })
            .collect();
        Self { samples, kind }
    }

    /// Read-only view of the window coefficients.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.samples
    }

    /// Window length (== STFT frame size).
    #[inline]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Always `false` — the constructor rejects empty windows.  Present to
    /// satisfy the `len`-without-`is_empty` clippy lint.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// The shape this window was built with.
    #[inline]
    pub fn kind(&self) -> WindowKind {
        self.kind
    }

    /// Reciprocal of the steady-state overlap-add gain for the given `hop`.
    ///
    /// Multiplying every synthesised sample by this factor makes an
    /// analysis-then-synthesis-windowed overlap-add reconstruct unity gain.
    /// The gain is `Σ_m w[n − m·hop]²`, evaluated at an interior sample so the
    /// full set of overlapping frames contributes; for a COLA-compliant
    /// window/hop the sum is position-independent.
    ///
    /// **Allocates nothing**, but is intended for off-RT setup.
    ///
    /// # Panics
    /// Panics if `hop == 0`.
    pub fn ola_normalisation(&self, hop: usize) -> f32 {
        assert!(hop > 0, "hop must be non-zero");
        let size = self.samples.len();
        // Evaluate at the window centre, summing the squared window over every
        // frame offset (± whole laps) that still overlaps this sample.
        let centre = size / 2;
        let max_laps = size / hop + 1;
        let mut gain = 0.0f32;
        for lap in -(max_laps as isize)..=(max_laps as isize) {
            let shifted = centre as isize - lap * hop as isize;
            if (0..size as isize).contains(&shifted) {
                let value = self.samples[shifted as usize];
                gain += value * value;
            }
        }
        if gain > 0.0 { gain.recip() } else { 1.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A periodic Hann window starts at zero and reaches unity at the centre.
    /// (The last sample is near — but not exactly — zero for the periodic
    /// form, which is expected and required for COLA.)
    #[test]
    fn hann_endpoints_and_centre() {
        let window = Window::new(WindowKind::Hann, 1024);
        let samples = window.as_slice();
        assert!(samples[0].abs() < 1e-6);
        assert!(samples[1023].abs() < 1e-3); // periodic tail, small but nonzero
        assert!((samples[512] - 1.0).abs() < 1e-3);
    }

    /// Blackman-Harris is strictly positive in the interior and small at the
    /// start (its first sample is the a0−a1+a2−a3 pedestal, ≈ 6e-5).
    #[test]
    fn blackman_harris_shape() {
        let window = Window::new(WindowKind::BlackmanHarris, 512);
        let samples = window.as_slice();
        assert!(samples[0].abs() < 1e-3);
        assert!((samples[256] - 1.0).abs() < 1e-2);
    }

    /// The measured Hann COLA gain at 75 % overlap is 1.5, so its reciprocal
    /// normalisation is ≈ 0.667.
    #[test]
    fn hann_cola_at_four_times_overlap() {
        let size = 1024;
        let window = Window::new(WindowKind::Hann, size);
        let norm = window.ola_normalisation(size / 4);
        assert!((norm - 1.0 / 1.5).abs() < 1e-2, "norm = {norm}");
    }

    /// Length / emptiness accessors behave.
    #[test]
    fn length_accessors() {
        let window = Window::new(WindowKind::Hann, 256);
        assert_eq!(window.len(), 256);
        assert!(!window.is_empty());
        assert_eq!(window.kind(), WindowKind::Hann);
    }
}
