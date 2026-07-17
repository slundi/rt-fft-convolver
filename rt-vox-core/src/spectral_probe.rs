//! Downsampled magnitude + formant-envelope snapshot for the editor display.
//!
//! The Phase-3 "formant display" overlays the input's spectral envelope on its
//! magnitude spectrum.  Sending a full FFT frame to the editor every block would
//! be wasteful, so [`SpectralProbe`] analyses the input on its own STFT frame
//! and reduces the result to a handful of **log-spaced display bins** (matching
//! how the plot reads on a log-frequency axis).
//!
//! It is meant to be driven only while the editor is open (the plugin gates it),
//! so the analysis FFT + LPC cost is paid only when something is watching.  Once
//! constructed it is allocation-free.

use realfft::num_complex::Complex;
use rt_fft_convolver::{DenormalGuard, FftHandler};
use rt_pitch_core::{
    FormantEstimator, FormantMethod,
    stft::Analyzer,
    window::{Window, WindowKind},
};

use crate::bands::LogBands;

/// Computes a downsampled magnitude / formant-envelope snapshot of a signal.
pub struct SpectralProbe {
    fft: FftHandler,
    window: Vec<f32>,
    analyzer: Analyzer,
    formant: FormantEstimator,
    bands: LogBands,

    frame: usize,
    time_frame: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    magnitude: Vec<f32>,
    envelope: Vec<f32>,
}

impl SpectralProbe {
    /// Build a probe analysing `frame`-point windows at `sample_rate`, reducing
    /// to `display_bins` log-spaced output bins, absorbing host blocks up to
    /// `max_block`.
    ///
    /// **Allocates** — construct off the audio thread.
    ///
    /// # Panics
    /// Panics if `frame` is odd or `display_bins == 0`.
    pub fn new(frame: usize, sample_rate: f32, display_bins: usize, max_block: usize) -> Self {
        assert!(frame.is_multiple_of(2), "frame must be even");
        assert!(display_bins > 0, "need at least one display bin");
        let bins = frame / 2 + 1;
        let window = Window::new(WindowKind::Hann, frame);
        let lpc_order = (sample_rate / 1000.0) as usize + 2;

        Self {
            fft: FftHandler::new(frame),
            window: window.as_slice().to_vec(),
            // A fresh frame every `frame` samples (no overlap) is plenty for a
            // display refresh.
            analyzer: Analyzer::new(frame, frame, max_block),
            formant: FormantEstimator::new(FormantMethod::Lpc, frame, lpc_order),
            bands: LogBands::new(display_bins, bins, frame, sample_rate, 80.0),
            frame,
            time_frame: vec![0.0; frame],
            spectrum: vec![Complex::default(); bins],
            magnitude: vec![0.0; bins],
            envelope: vec![0.0; bins],
        }
    }

    /// Feed `input` and, when a full frame completes, refresh the downsampled
    /// `magnitude` / `envelope` display arrays (each `display_bins` long).
    ///
    /// Returns `true` when the arrays were updated this call.  Allocation-free.
    ///
    /// # Panics
    /// Panics if `magnitude` / `envelope` are not `display_bins` long.
    pub fn process_block(
        &mut self,
        input: &[f32],
        magnitude: &mut [f32],
        envelope: &mut [f32],
    ) -> bool {
        debug_assert_eq!(magnitude.len(), self.bands.count());
        debug_assert_eq!(envelope.len(), self.bands.count());
        let _guard = DenormalGuard::new();

        self.analyzer.push_block(input);
        let mut updated = false;
        while self.analyzer.next_frame(&mut self.time_frame) {
            for (sample, &win) in self.time_frame.iter_mut().zip(&self.window) {
                *sample *= win;
            }
            self.fft.forward(&self.time_frame, &mut self.spectrum);
            for (mag, bin) in self.magnitude.iter_mut().zip(&self.spectrum) {
                *mag = bin.norm();
            }
            self.formant.estimate(&self.magnitude, &mut self.envelope);
            updated = true;
        }

        if updated {
            // `downsample` borrows `self` immutably alongside the source slice
            // (also `self`), so no clone / allocation is needed.
            self.downsample(&self.magnitude, magnitude);
            self.downsample(&self.envelope, envelope);
        }
        updated
    }

    /// Average `source` FFT bins into each log-spaced display band.
    fn downsample(&self, source: &[f32], out: &mut [f32]) {
        for (band, dst) in out.iter_mut().enumerate() {
            let (start, end) = self.bands.bin_range(band);
            let count = (end - start).max(1) as f32;
            *dst = source[start..end].iter().sum::<f32>() / count;
        }
    }

    /// Analysis frame length (samples).
    #[inline]
    pub fn frame_len(&self) -> usize {
        self.frame
    }

    /// Drop buffered input.
    pub fn reset(&mut self) {
        self.analyzer.reset();
        self.time_frame.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::make_sawtooth;

    const SAMPLE_RATE: f32 = 48_000.0;
    const FRAME: usize = 1024;
    const DISPLAY_BINS: usize = 48;
    const MAX_BLOCK: usize = 512;

    #[test]
    fn produces_finite_bins_after_a_frame() {
        let mut probe = SpectralProbe::new(FRAME, SAMPLE_RATE, DISPLAY_BINS, MAX_BLOCK);
        let signal = make_sawtooth(220.0, SAMPLE_RATE, 4096);
        let mut mag = vec![0.0f32; DISPLAY_BINS];
        let mut env = vec![0.0f32; DISPLAY_BINS];

        let mut ever = false;
        for block in signal.chunks(MAX_BLOCK) {
            ever |= probe.process_block(block, &mut mag, &mut env);
        }
        assert!(ever, "no frame was ever analysed");
        assert!(mag.iter().all(|&v| v.is_finite() && v >= 0.0));
        assert!(env.iter().all(|&v| v.is_finite() && v >= 0.0));
        assert!(mag.iter().any(|&v| v > 0.0), "magnitude all zero");
    }

    #[test]
    fn silence_probes_to_zero() {
        let mut probe = SpectralProbe::new(FRAME, SAMPLE_RATE, DISPLAY_BINS, MAX_BLOCK);
        let mut mag = vec![1.0f32; DISPLAY_BINS];
        let mut env = vec![1.0f32; DISPLAY_BINS];
        for _ in 0..8 {
            probe.process_block(&vec![0.0f32; MAX_BLOCK], &mut mag, &mut env);
        }
        assert!(mag.iter().all(|&v| v < 1e-6), "silent magnitude not zero");
    }

    #[test]
    fn no_frame_before_enough_input() {
        let mut probe = SpectralProbe::new(FRAME, SAMPLE_RATE, DISPLAY_BINS, MAX_BLOCK);
        let mut mag = vec![0.0f32; DISPLAY_BINS];
        let mut env = vec![0.0f32; DISPLAY_BINS];
        // One short block, far below the frame length.
        assert!(!probe.process_block(&[0.1; 128], &mut mag, &mut env));
    }
}
