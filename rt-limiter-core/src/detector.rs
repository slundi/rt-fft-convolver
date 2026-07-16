//! [`TruePeakDetector`] — oversampled inter-sample-peak envelope.
//!
//! # Why oversample to find a peak
//!
//! A digital sample is a point on a band-limited waveform.  The *continuous*
//! signal reconstructed between two samples can rise **above** the larger of
//! the two — the "inter-sample" or "true" peak.  A signal whose sample values
//! all sit at `−0.2 dBFS` can reconstruct to `+0.5 dBTP`, which a naïve
//! sample-peak limiter would pass straight through to a D/A converter that
//! then clips.  ITU-R BS.1770-4 measures true-peak by up-sampling (≥4×) and
//! taking the peak of the oversampled signal; that is exactly what this does,
//! reusing `rt-drive-core`'s Kaiser half-band polyphase [`Oversampler`].
//!
//! # Output and latency
//!
//! [`TruePeakDetector::detect_block`] fills one **true-peak value per input
//! sample**: the maximum `|x|` over the `factor` oversampled sub-samples that
//! sit under each input sample.  The polyphase up-sampler is linear-phase, so
//! its group delay shifts that envelope by a fixed, integer number of input
//! samples — reported by [`TruePeakDetector::latency`].  The limiter adds that
//! latency to its dry-signal delay so the gain reduction still lands on the
//! sample that caused the peak.  Getting *that* alignment wrong is the classic
//! look-ahead-limiter bug, which is why the latency is measured explicitly
//! (with a unit impulse) at construction rather than assumed to be zero.
//!
//! # Real-time safety
//!
//! [`TruePeakDetector::detect_block`] allocates nothing (the oversampled
//! scratch is pre-sized) and never locks or panics for blocks `≤ max_block`.

use rt_drive_core::Oversampler;

/// Oversampled true-peak envelope generator for one audio channel.
pub struct TruePeakDetector {
    oversampler: Oversampler,
    /// Oversampled scratch, capacity `max_block · factor`.
    scratch: Vec<f32>,
    max_block: usize,
    factor: usize,
    latency: usize,
}

impl TruePeakDetector {
    /// Build a detector at oversampling `factor` (`1, 2, 4, 8, 16`) for a
    /// maximum input block of `max_block` samples, using the resampler's
    /// default filter design.
    ///
    /// **Allocates** — call off the audio thread.
    pub fn new(factor: usize, max_block: usize) -> Self {
        Self::with_design(
            factor,
            max_block,
            Oversampler::DEFAULT_ATTEN_DB,
            Oversampler::DEFAULT_TRANSITION,
        )
    }

    /// Build a detector with an explicit half-band filter design (mainly to
    /// keep the impulse-latency measurement fast in tests).
    ///
    /// **Allocates** — call off the audio thread.
    pub fn with_design(
        factor: usize,
        max_block: usize,
        atten_db: f64,
        transition_frac: f64,
    ) -> Self {
        let mut oversampler =
            Oversampler::with_design(factor, max_block, atten_db, transition_frac);
        let real_factor = oversampler.factor();
        let latency = measure_latency(&mut oversampler, real_factor, max_block);
        Self {
            oversampler,
            scratch: vec![0.0; max_block * real_factor],
            max_block,
            factor: real_factor,
            latency,
        }
    }

    /// The active oversampling factor (rounded to a power of two by the
    /// resampler).
    #[inline]
    pub fn factor(&self) -> usize {
        self.factor
    }

    /// The maximum input block length this detector was sized for.
    #[inline]
    pub fn max_block(&self) -> usize {
        self.max_block
    }

    /// The detector's group delay in **input** samples: the peak envelope is
    /// this many samples behind the input that produced it.
    #[inline]
    pub fn latency(&self) -> usize {
        self.latency
    }

    /// Fill `peaks[i]` with the true-peak magnitude around input sample `i`.
    ///
    /// `peaks.len()` must equal `input.len()`, and the block must be
    /// `≤ max_block` (both debug-asserted).
    ///
    /// **Real-time safe.**
    pub fn detect_block(&mut self, input: &[f32], peaks: &mut [f32]) {
        debug_assert_eq!(input.len(), peaks.len(), "detect_block: length mismatch");
        debug_assert!(input.len() <= self.max_block, "block exceeds max_block");

        let block = input.len().min(peaks.len());
        let factor = self.factor;
        let oversampled = &mut self.scratch[..block * factor];
        self.oversampler.upsample(&input[..block], oversampled);

        for (index, peak) in peaks[..block].iter_mut().enumerate() {
            // Max |x| over the `factor` sub-samples under input sample `index`
            // is the reconstructed inter-sample peak in that neighbourhood.
            let base = index * factor;
            let mut maximum = 0.0f32;
            for &sample in &oversampled[base..base + factor] {
                maximum = maximum.max(sample.abs());
            }
            *peak = maximum;
        }
    }

    /// Clear the oversampler's delay lines.
    pub fn reset(&mut self) {
        self.oversampler.reset();
    }
}

/// Measure the detector's group delay in input samples by pushing a unit
/// impulse and finding where the peak envelope crests.
///
/// Runs once at construction (allocates, non-RT); the oversampler is reset
/// afterwards so no state leaks into the first real block.
fn measure_latency(oversampler: &mut Oversampler, factor: usize, max_block: usize) -> usize {
    if factor == 1 {
        // factor 1 copies through: zero added delay.
        oversampler.reset();
        return 0;
    }
    let mut input = vec![0.0f32; max_block];
    let mut oversampled = vec![0.0f32; max_block * factor];
    input[0] = 1.0;

    let mut peak_index = 0usize;
    let mut peak_value = 0.0f32;
    let mut offset = 0usize;
    // Cover well past the half-band FIR delay (a few hundred input samples at
    // 96 dB) regardless of how small `max_block` is.
    let blocks = (16_384 / max_block.max(1)).max(4);
    for _ in 0..blocks {
        oversampler.upsample(&input, &mut oversampled);
        for sample_index in 0..max_block {
            let base = sample_index * factor;
            let mut maximum = 0.0f32;
            for &sample in &oversampled[base..base + factor] {
                maximum = maximum.max(sample.abs());
            }
            if maximum > peak_value {
                peak_value = maximum;
                peak_index = offset + sample_index;
            }
        }
        offset += max_block;
        input[0] = 0.0;
    }
    oversampler.reset();
    peak_index
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    /// A light design keeps the impulse-latency probe cheap.
    const LIGHT: (f64, f64) = (80.0, 0.1);

    /// factor 1 is a straight `|x|` envelope with zero latency.
    #[test]
    fn factor_one_is_abs_zero_latency() {
        let mut detector = TruePeakDetector::new(1, 8);
        assert_eq!(detector.latency(), 0);
        let input = [0.5f32, -0.7, 0.2, -1.0, 0.0, 0.3, -0.1, 0.9];
        let mut peaks = [0.0f32; 8];
        detector.detect_block(&input, &mut peaks);
        for (got, expected) in peaks.iter().zip(input.iter().map(|s| s.abs())) {
            assert!((got - expected).abs() < 1e-6);
        }
    }

    /// Oversampling reports a non-zero, finite group delay.
    #[test]
    fn oversampled_latency_is_positive() {
        for factor in [2usize, 4, 8] {
            let detector = TruePeakDetector::with_design(factor, 64, LIGHT.0, LIGHT.1);
            assert!(
                detector.latency() > 0,
                "factor {factor} should have positive latency"
            );
        }
    }

    /// The true peak of a signal that lands between samples exceeds its
    /// sample peak.  A full-scale sine sampled just off its crest is the
    /// canonical inter-sample-peak case.
    #[test]
    fn true_peak_exceeds_sample_peak() {
        // 0.25·fs sine at 45°: every sample lands at ±0.707, so the true
        // peak (≈ 1.0) lies strictly between samples — the canonical ISP case.
        let block = 256;
        let freq = 0.25f32;
        let phase = std::f32::consts::FRAC_PI_4;
        let input: Vec<f32> = (0..block)
            .map(|n| (TAU * freq * n as f32 + phase).sin())
            .collect();
        let sample_peak = input.iter().fold(0.0f32, |m, &s| m.max(s.abs()));

        let mut detector = TruePeakDetector::with_design(4, block, LIGHT.0, LIGHT.1);
        let mut peaks = vec![0.0f32; block];
        detector.detect_block(&input, &mut peaks);
        let true_peak = peaks.iter().fold(0.0f32, |m, &p| m.max(p));

        assert!(
            true_peak > sample_peak,
            "true peak {true_peak} should exceed sample peak {sample_peak}"
        );
    }

    /// `reset` clears delay-line state so a later block is not contaminated.
    #[test]
    fn reset_clears_state() {
        let mut detector = TruePeakDetector::with_design(4, 32, LIGHT.0, LIGHT.1);
        let hot = vec![1.0f32; 32];
        let mut peaks = vec![0.0f32; 32];
        detector.detect_block(&hot, &mut peaks);
        detector.reset();
        let silence = vec![0.0f32; 32];
        detector.detect_block(&silence, &mut peaks);
        // The first silent block after reset carries only the tail of the
        // *current* (silent) input, which is zero everywhere.
        assert!(peaks.iter().all(|&p| p < 1e-6), "state leaked past reset");
    }
}
