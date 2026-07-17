//! Rolling-FFT feedback detector.
//!
//! Acoustic feedback ("howlback") shows up as a **narrow spectral spike that
//! persists and dominates its neighbours** frame after frame, unlike a musical
//! tone which sits amid harmonics and broadband energy.  This detector runs a
//! small (default 512-point) Hann-windowed FFT over the incoming signal and
//! flags a bin as feedback when its magnitude exceeds a sensitivity-controlled
//! multiple of the local spectral average for several consecutive frames
//! (Roadmap-09 Phase 0: FFT peak-tracker + peak-hold, the Sabine/Shure model).
//!
//! It is a *detector* only — it never touches the audio.  The plugin feeds it a
//! copy of the block and forwards any returned frequency to the
//! [`NotchBank`](crate::notch::NotchBank).
//!
//! # Real-time safety
//!
//! All buffers are reserved in [`FeedbackDetector::new`]; [`process`] allocates
//! nothing and is wrapped in a [`DenormalGuard`].
//!
//! [`process`]: FeedbackDetector::process

use realfft::num_complex::Complex;
use rt_fft_convolver::{DenormalGuard, FftHandler, Ring};
use std::f32::consts::TAU;

/// Frames a candidate bin must stay above threshold before it is confirmed.
const DEFAULT_PERSISTENCE: u16 = 3;

/// Peak-to-average ratio required at the least sensitive setting (`0.0`).
const RATIO_AT_MIN_SENSITIVITY: f32 = 20.0;
/// Peak-to-average ratio required at the most sensitive setting (`1.0`).
const RATIO_AT_MAX_SENSITIVITY: f32 = 3.0;

/// Lowest frequency (Hz) the detector will place a notch at — below this is the
/// mains-hum region handled by the hum tracker, not feedback.
const MIN_DETECT_HZ: f32 = 150.0;

/// FFT peak-tracking feedback detector.
pub struct FeedbackDetector {
    sample_rate: f32,
    fft: FftHandler,
    fft_size: usize,
    /// Accumulates host samples until a full analysis frame is available.
    ring: Ring<f32>,
    /// Hann window applied before the transform.
    window: Vec<f32>,
    /// One analysis frame (drained from `ring`).
    frame: Vec<f32>,
    /// Windowed copy of `frame` fed to the FFT.
    windowed: Vec<f32>,
    /// Scratch spectrum (`N/2 + 1` bins).
    spectrum: Vec<Complex<f32>>,
    /// Per-bin magnitude of the current frame.
    magnitude: Vec<f32>,
    /// Per-bin consecutive-candidate counter (peak-hold).
    persist: Vec<u16>,
    /// Peak-to-average ratio a bin must beat to count as a candidate.
    ratio_threshold: f32,
    /// Frames a candidate must persist before it is confirmed.
    persistence: u16,
    /// Inclusive bin search range `[min_bin, max_bin]`.
    min_bin: usize,
    max_bin: usize,
}

impl FeedbackDetector {
    /// Build a detector for `sample_rate` using an `fft_size`-point transform.
    ///
    /// **Allocates** — off-RT only.
    ///
    /// # Panics
    /// Panics if `fft_size` is not a positive even number (realfft's
    /// requirement).
    pub fn new(sample_rate: f32, fft_size: usize) -> Self {
        let fft = FftHandler::new(fft_size);
        let spectrum_size = fft.spectrum_size();
        // Hann window (periodic form matches the FFT bin spacing).
        let window: Vec<f32> = (0..fft_size)
            .map(|n| 0.5 - 0.5 * (TAU * n as f32 / fft_size as f32).cos())
            .collect();

        let nyquist_bin = spectrum_size - 1;
        let min_bin = ((MIN_DETECT_HZ * fft_size as f32 / sample_rate).round() as usize).max(1);
        // Ignore the top few bins (anti-alias roll-off / Nyquist).
        let max_bin = nyquist_bin.saturating_sub(1);

        let mut detector = Self {
            sample_rate,
            fft,
            fft_size,
            ring: Ring::with_capacity(fft_size),
            window,
            frame: vec![0.0; fft_size],
            windowed: vec![0.0; fft_size],
            spectrum: vec![Complex::default(); spectrum_size],
            magnitude: vec![0.0; spectrum_size],
            persist: vec![0; spectrum_size],
            ratio_threshold: 0.0,
            persistence: DEFAULT_PERSISTENCE,
            min_bin,
            max_bin,
        };
        detector.set_sensitivity(0.5);
        detector
    }

    /// Map a `0.0 ..= 1.0` sensitivity onto the peak-to-average ratio: more
    /// sensitive → a lower ratio counts as feedback.
    pub fn set_sensitivity(&mut self, sensitivity: f32) {
        let sensitivity = sensitivity.clamp(0.0, 1.0);
        self.ratio_threshold = RATIO_AT_MIN_SENSITIVITY
            + sensitivity * (RATIO_AT_MAX_SENSITIVITY - RATIO_AT_MIN_SENSITIVITY);
    }

    /// Override how many consecutive frames confirm a candidate (default 3).
    pub fn set_persistence(&mut self, frames: u16) {
        self.persistence = frames.max(1);
    }

    /// Clear the ring and all peak-hold counters (call on a transport jump).
    pub fn reset(&mut self) {
        self.ring.clear();
        for count in &mut self.persist {
            *count = 0;
        }
    }

    /// Feed a block and return the frequency (Hz) of a newly confirmed feedback
    /// partial, if any.  Only the strongest confirmed bin per analysis frame is
    /// reported; call repeatedly across blocks to catch successive howls.
    ///
    /// # Real-time safety
    /// No allocation.
    pub fn process(&mut self, block: &[f32]) -> Option<f32> {
        let _guard = DenormalGuard::new();
        let mut detected = None;
        let mut offset = 0;
        while offset < block.len() {
            offset += self.ring.push_slice(&block[offset..]);
            while self.ring.len() >= self.fft_size {
                self.ring.pop_slice(&mut self.frame);
                if let Some(freq) = self.analyse_frame() {
                    detected = Some(freq);
                }
            }
        }
        detected
    }

    /// Transform `self.frame`, update peak-hold, and return a confirmed
    /// feedback frequency if one crossed the persistence threshold this frame.
    fn analyse_frame(&mut self) -> Option<f32> {
        for (dst, (&sample, &w)) in self
            .windowed
            .iter_mut()
            .zip(self.frame.iter().zip(self.window.iter()))
        {
            *dst = sample * w;
        }
        self.fft.forward(&self.windowed, &mut self.spectrum);

        // Magnitude spectrum and mean magnitude over the search band.
        let mut sum = 0.0f32;
        for (mag, bin) in self.magnitude.iter_mut().zip(self.spectrum.iter()) {
            *mag = bin.norm();
        }
        for bin in self.min_bin..=self.max_bin {
            sum += self.magnitude[bin];
        }
        let count = (self.max_bin - self.min_bin + 1) as f32;
        let mean = sum / count;
        if mean <= 1e-9 {
            return None; // silence — nothing to detect
        }

        // Peak-hold: count consecutive frames each bin dominates the average.
        let mut confirmed: Option<(usize, f32)> = None;
        for bin in self.min_bin..=self.max_bin {
            let mag = self.magnitude[bin];
            let is_local_peak = mag >= self.magnitude[bin - 1] && mag >= self.magnitude[bin + 1];
            let dominates = mag > self.ratio_threshold * mean;
            if is_local_peak && dominates {
                self.persist[bin] = self.persist[bin].saturating_add(1);
                if self.persist[bin] >= self.persistence {
                    // Keep the strongest confirmed bin this frame.
                    match confirmed {
                        Some((_, best)) if best >= mag => {}
                        _ => confirmed = Some((bin, mag)),
                    }
                }
            } else {
                // Decay slowly so a single missed frame doesn't reset a howl.
                self.persist[bin] = self.persist[bin].saturating_sub(1);
            }
        }

        let (bin, _) = confirmed?;
        // A confirmed notch will suppress this partial; clear the counter so the
        // same bin doesn't fire again on the very next frame.
        self.persist[bin] = 0;
        Some(self.interpolated_frequency(bin))
    }

    /// Parabolic interpolation of the magnitude peak at `bin` for sub-bin
    /// frequency accuracy.
    fn interpolated_frequency(&self, bin: usize) -> f32 {
        let left = self.magnitude[bin - 1];
        let centre = self.magnitude[bin];
        let right = self.magnitude[bin + 1];
        let denom = left - 2.0 * centre + right;
        let delta = if denom.abs() > 1e-12 {
            0.5 * (left - right) / denom
        } else {
            0.0
        };
        (bin as f32 + delta.clamp(-0.5, 0.5)) * self.sample_rate / self.fft_size as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::make_sine;

    const SR: f32 = 48_000.0;

    /// Deterministic low-level broadband noise floor.
    fn noise(len: usize, amplitude: f32) -> Vec<f32> {
        let mut state: u32 = 0x9E37_79B9;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                ((state as f32 / u32::MAX as f32) * 2.0 - 1.0) * amplitude
            })
            .collect()
    }

    #[test]
    fn detects_sustained_howl_frequency() {
        let mut detector = FeedbackDetector::new(SR, 512);
        // A strong 2 kHz tone over a quiet noise floor.
        let tone = make_sine(2_000.0, SR, 512 * 8);
        let floor = noise(tone.len(), 0.02);
        let mixed: Vec<f32> = tone.iter().zip(floor.iter()).map(|(t, n)| t + n).collect();

        let mut found = None;
        for chunk in mixed.chunks(512) {
            if let Some(freq) = detector.process(chunk) {
                found = Some(freq);
                break;
            }
        }
        let freq = found.expect("feedback not detected");
        assert!((freq - 2_000.0).abs() < 100.0, "detected {freq} Hz");
    }

    #[test]
    fn broadband_noise_is_not_feedback() {
        let mut detector = FeedbackDetector::new(SR, 512);
        let noise = noise(512 * 12, 0.3);
        let mut detections = 0;
        for chunk in noise.chunks(512) {
            if detector.process(chunk).is_some() {
                detections += 1;
            }
        }
        assert_eq!(detections, 0, "noise wrongly flagged as feedback");
    }

    #[test]
    fn silence_returns_none() {
        let mut detector = FeedbackDetector::new(SR, 512);
        assert_eq!(detector.process(&vec![0.0f32; 4096]), None);
    }

    #[test]
    fn honours_arbitrary_block_sizes() {
        // A block larger than the FFT frame must still be consumed fully.
        let mut detector = FeedbackDetector::new(SR, 512);
        let tone = make_sine(3_000.0, SR, 4096);
        // One oversized block containing several frames.
        let freq = detector.process(&tone);
        // Persistence may need more than one call; feed again to confirm.
        let freq = freq.or_else(|| detector.process(&tone));
        assert!(freq.map(|f| (f - 3_000.0).abs() < 120.0).unwrap_or(false));
    }
}
