//! Display-only spectrum analyser for the Phase-3 editor.
//!
//! The editor draws the input magnitude spectrum behind the draggable crossover
//! lines.  Sending a full FFT frame to the editor every block would be wasteful,
//! so [`SpectrumProbe`] analyses the input on its own non-overlapping frame and
//! reduces the result to a handful of **log-spaced display bins** (matching how
//! the plot reads on a log-frequency axis).
//!
//! It is a **UI concern**, so it lives in the plugin (not the dynamics core),
//! and the plugin runs it only while the editor is open — the analysis FFT cost
//! is paid only when something is watching.  Once constructed it is
//! allocation-free.

use std::f32::consts::TAU;

use realfft::num_complex::Complex;
use rt_fft_convolver::{DenormalGuard, FftHandler};

/// Computes a downsampled magnitude snapshot of a signal for the editor.
pub struct SpectrumProbe {
    fft: FftHandler,
    window: Vec<f32>,
    frame: usize,
    /// Samples accumulated into the current frame.
    fill: usize,
    time: Vec<f32>,
    windowed: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    magnitude: Vec<f32>,
    /// `[start, end)` FFT-bin range feeding each log-spaced display bin.
    bin_ranges: Vec<(usize, usize)>,
}

impl SpectrumProbe {
    /// Build a probe analysing `frame`-point windows at `sample_rate`, reducing
    /// to `display_bins` log-spaced output bins spanning `min_hz..nyquist`.
    ///
    /// **Allocates** — construct off the audio thread.
    ///
    /// # Panics
    /// Panics if `frame` is odd / zero or `display_bins == 0`.
    pub fn new(frame: usize, sample_rate: f32, display_bins: usize, min_hz: f32) -> Self {
        assert!(
            frame >= 2 && frame.is_multiple_of(2),
            "frame must be even ≥ 2"
        );
        assert!(display_bins > 0, "need at least one display bin");
        let bins = frame / 2 + 1;

        // Hann window (0.5 − 0.5·cos) for a clean display spectrum.
        let window = (0..frame)
            .map(|n| 0.5 - 0.5 * (TAU * n as f32 / frame as f32).cos())
            .collect();

        let nyquist = sample_rate * 0.5;
        let min_hz = min_hz.clamp(1.0, nyquist * 0.5);
        let bin_hz = sample_rate / frame as f32;
        let log_min = min_hz.ln();
        let log_max = nyquist.ln();
        let bin_ranges = (0..display_bins)
            .map(|index| {
                let lo = log_min + (log_max - log_min) * index as f32 / display_bins as f32;
                let hi = log_min + (log_max - log_min) * (index + 1) as f32 / display_bins as f32;
                let start = (lo.exp() / bin_hz).floor() as usize;
                let end = (hi.exp() / bin_hz).ceil() as usize;
                let start = start.min(bins - 1);
                let end = end.clamp(start + 1, bins);
                (start, end)
            })
            .collect();

        Self {
            fft: FftHandler::new(frame),
            window,
            frame,
            fill: 0,
            time: vec![0.0; frame],
            windowed: vec![0.0; frame],
            spectrum: vec![Complex::default(); bins],
            magnitude: vec![0.0; bins],
            bin_ranges,
        }
    }

    /// Number of log-spaced display bins produced (the `out` length).
    #[inline]
    pub fn display_bins(&self) -> usize {
        self.bin_ranges.len()
    }

    /// Feed `input` and, when a full frame completes, refresh the downsampled
    /// `out` magnitude array (`display_bins` long).  Returns `true` when `out`
    /// was updated this call.  Allocation-free.
    ///
    /// # Panics
    /// Panics if `out` is not `display_bins` long.
    pub fn push_block(&mut self, input: &[f32], out: &mut [f32]) -> bool {
        assert_eq!(out.len(), self.bin_ranges.len(), "out length mismatch");
        let _guard = DenormalGuard::new();

        let mut updated = false;
        for &sample in input {
            self.time[self.fill] = sample;
            self.fill += 1;
            if self.fill == self.frame {
                self.fill = 0;
                self.analyse(out);
                updated = true;
            }
        }
        updated
    }

    /// Window + FFT the accumulated frame and downsample into `out`.
    fn analyse(&mut self, out: &mut [f32]) {
        for ((dst, &sample), &win) in self.windowed.iter_mut().zip(&self.time).zip(&self.window) {
            *dst = sample * win;
        }
        self.fft.forward(&self.windowed, &mut self.spectrum);
        for (mag, bin) in self.magnitude.iter_mut().zip(&self.spectrum) {
            *mag = bin.norm();
        }
        for (&(start, end), dst) in self.bin_ranges.iter().zip(out.iter_mut()) {
            let count = (end - start).max(1) as f32;
            *dst = self.magnitude[start..end].iter().sum::<f32>() / count;
        }
    }

    /// Drop the accumulated frame.
    pub fn reset(&mut self) {
        self.fill = 0;
        self.time.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;
    const FRAME: usize = 1024;
    const BINS: usize = 32;
    const MIN_HZ: f32 = 30.0;

    fn make_sine(freq: f32, samples: usize) -> Vec<f32> {
        (0..samples)
            .map(|n| (TAU * freq * n as f32 / SR).sin())
            .collect()
    }

    /// The log-frequency display bin that covers `freq`.
    fn bin_for(probe: &SpectrumProbe, freq: f32) -> usize {
        let bin_hz = SR / probe.frame as f32;
        let fft_bin = (freq / bin_hz) as usize;
        probe
            .bin_ranges
            .iter()
            .position(|&(start, end)| (start..end).contains(&fft_bin))
            .unwrap()
    }

    #[test]
    fn display_bin_count_matches_request() {
        let probe = SpectrumProbe::new(FRAME, SR, BINS, MIN_HZ);
        assert_eq!(probe.display_bins(), BINS);
    }

    #[test]
    fn silence_produces_zero_spectrum() {
        let mut probe = SpectrumProbe::new(FRAME, SR, BINS, MIN_HZ);
        let mut out = vec![0.0; BINS];
        let updated = probe.push_block(&vec![0.0; FRAME], &mut out);
        assert!(updated, "a full frame should refresh");
        assert!(out.iter().all(|&m| m == 0.0), "silence not zero: {out:?}");
    }

    #[test]
    fn tone_peaks_in_its_own_display_bin() {
        let mut probe = SpectrumProbe::new(FRAME, SR, BINS, MIN_HZ);
        let target = 2_000.0;
        let mut out = vec![0.0; BINS];
        probe.push_block(&make_sine(target, FRAME), &mut out);

        let expected = bin_for(&probe, target);
        let loudest = out
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(loudest, expected, "2 kHz did not peak in its bin");
    }

    #[test]
    fn partial_frame_does_not_update() {
        let mut probe = SpectrumProbe::new(FRAME, SR, BINS, MIN_HZ);
        let mut out = vec![0.0; BINS];
        // Fewer than `frame` samples: no refresh yet.
        assert!(!probe.push_block(&vec![0.1; FRAME / 2], &mut out));
        // The rest of the frame completes it.
        assert!(probe.push_block(&vec![0.1; FRAME / 2], &mut out));
    }

    #[test]
    #[should_panic(expected = "out length mismatch")]
    fn wrong_out_length_panics() {
        let mut probe = SpectrumProbe::new(FRAME, SR, BINS, MIN_HZ);
        let mut out = vec![0.0; BINS + 1];
        probe.push_block(&[0.0; FRAME], &mut out);
    }
}
