//! Monophonic pitch tracking and a polyphony guard for the PSOLA path.
//!
//! PSOLA can only reposition pitch periods it can *find*, so it needs the
//! fundamental period of a **single** voice.  [`PitchTracker`] estimates that
//! period with a normalised autocorrelation (robust, cheap, no FFT) and
//! reports a confidence in `[0, 1]`.  When the confidence is low the input is
//! either unpitched or **polyphonic** — a chord — and the caller should warn
//! and/or bypass, because PSOLA would mangle it (the Roadmap-03 "Mono Voice"
//! constraint).

/// Result of a pitch analysis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PitchEstimate {
    /// Fundamental period in (fractional) samples.
    pub period: f32,
    /// Fundamental frequency in Hz.
    pub frequency: f32,
    /// Peak normalised autocorrelation in `[0, 1]`; how strongly periodic the
    /// frame is.  ~1.0 for a clean monophonic tone.
    pub confidence: f32,
}

/// Autocorrelation-based monophonic pitch tracker.
pub struct PitchTracker {
    sample_rate: f32,
    /// Shortest lag searched (from the highest trackable frequency).
    min_period: usize,
    /// Longest lag searched (from the lowest trackable frequency).
    max_period: usize,
    /// Confidence below which a frame is treated as polyphonic / unpitched.
    poly_threshold: f32,
    /// A candidate lag is accepted as the fundamental if its normalised score
    /// is at least this fraction of the global-best score — resolving the
    /// octave error where 2·P / 3·P correlate as strongly as P.
    octave_fraction: f32,
    /// Preallocated per-lag score buffer (`max_period + 1`), so `analyse`
    /// never allocates on the audio thread.
    scores: Vec<f32>,
}

impl PitchTracker {
    /// Track fundamentals in `[min_freq, max_freq]` Hz.
    ///
    /// # Panics
    /// Panics unless `0 < min_freq < max_freq < sample_rate / 2`.
    pub fn new(sample_rate: f32, min_freq: f32, max_freq: f32) -> Self {
        assert!(
            0.0 < min_freq && min_freq < max_freq && max_freq < sample_rate * 0.5,
            "require 0 < min_freq < max_freq < Nyquist"
        );
        let max_period = (sample_rate / min_freq).ceil() as usize;
        Self {
            sample_rate,
            min_period: (sample_rate / max_freq).floor() as usize,
            max_period,
            poly_threshold: 0.8,
            octave_fraction: 0.85,
            scores: vec![0.0; max_period + 1],
        }
    }

    /// Minimum analysis-frame length required by [`analyse`](Self::analyse):
    /// two of the longest periods so the autocorrelation has a full lag range.
    #[inline]
    pub fn min_frame_len(&self) -> usize {
        2 * self.max_period
    }

    /// Override the confidence threshold used by
    /// [`is_polyphonic`](Self::is_polyphonic) (default `0.8`).
    pub fn set_polyphony_threshold(&mut self, threshold: f32) {
        self.poly_threshold = threshold;
    }

    /// Estimate the fundamental of `frame`, or `None` if the frame is too
    /// short or effectively silent.
    ///
    /// Uses the peak of the normalised cross-correlation
    /// `r(τ) / √(e₀ · e_τ)`, then parabolically interpolates the lag for
    /// sub-sample accuracy.
    pub fn analyse(&mut self, frame: &[f32]) -> Option<PitchEstimate> {
        if frame.len() < self.min_frame_len() {
            return None;
        }
        let usable = frame.len() - self.max_period; // samples with a full lag

        // Reject silence early (autocorrelation is meaningless).
        let energy0: f32 = frame[..usable].iter().map(|&s| s * s).sum();
        if energy0 < 1e-9 {
            return None;
        }

        // Pass 1: normalised autocorrelation score for every candidate lag.
        let mut best_score = f32::NEG_INFINITY;
        for lag in self.min_period..=self.max_period {
            let mut cross = 0.0f32;
            let mut energy_lag = 0.0f32;
            for i in 0..usable {
                let delayed = frame[i + lag];
                cross += frame[i] * delayed;
                energy_lag += delayed * delayed;
            }
            let denom = (energy0 * energy_lag).sqrt();
            let score = if denom > 1e-12 { cross / denom } else { 0.0 };
            self.scores[lag] = score;
            best_score = best_score.max(score);
        }
        if best_score <= 0.0 {
            return None;
        }

        // Pass 2: octave-error correction.  A periodic signal scores just as
        // high at 2·P, 3·P … as at the true period P, so pick the *shortest*
        // lag that is a local maximum reaching `octave_fraction` of the best
        // score — that is the fundamental, not one of its multiples.
        let accept = self.octave_fraction * best_score;
        let mut chosen = self.min_period;
        let mut found = false;
        for lag in self.min_period..=self.max_period {
            let score = self.scores[lag];
            let higher_than_left = lag == self.min_period || score >= self.scores[lag - 1];
            let higher_than_right = lag == self.max_period || score >= self.scores[lag + 1];
            if score >= accept && higher_than_left && higher_than_right {
                chosen = lag;
                found = true;
                break;
            }
        }
        if !found {
            return None;
        }

        let period = self.parabolic_period(chosen);
        Some(PitchEstimate {
            period,
            frequency: self.sample_rate / period,
            confidence: self.scores[chosen].clamp(0.0, 1.0),
        })
    }

    /// `true` when the frame is not confidently a single pitched voice —
    /// unpitched, noisy, or polyphonic (a chord).
    pub fn is_polyphonic(&mut self, frame: &[f32]) -> bool {
        let threshold = self.poly_threshold;
        match self.analyse(frame) {
            Some(estimate) => estimate.confidence < threshold,
            None => true,
        }
    }

    /// Parabolic interpolation of the (already computed) autocorrelation peak
    /// around `lag` for sub-sample period resolution.
    fn parabolic_period(&self, lag: usize) -> f32 {
        if lag <= self.min_period || lag >= self.max_period {
            return lag as f32;
        }
        let left = self.scores[lag - 1];
        let centre = self.scores[lag];
        let right = self.scores[lag + 1];
        let denom = left - 2.0 * centre + right;
        if denom.abs() < 1e-9 {
            lag as f32
        } else {
            let shift = 0.5 * (left - right) / denom;
            lag as f32 + shift.clamp(-1.0, 1.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{make_sawtooth, make_sine};

    #[test]
    fn detects_mono_sine_pitch() {
        let sample_rate = 48_000.0;
        let mut tracker = PitchTracker::new(sample_rate, 60.0, 1000.0);
        let frame = make_sine(220.0, sample_rate, 4096);
        let estimate = tracker.analyse(&frame).expect("pitched");
        assert!(
            (estimate.frequency - 220.0).abs() < 3.0,
            "freq = {}",
            estimate.frequency
        );
        assert!(estimate.confidence > 0.95, "conf = {}", estimate.confidence);
        assert!(!tracker.is_polyphonic(&frame));
    }

    #[test]
    fn resolves_octave_error_on_sawtooth() {
        // A sawtooth correlates as strongly at 2·P / 3·P as at P; the tracker
        // must still report the fundamental (220 Hz), not a sub-octave.
        let sample_rate = 48_000.0;
        let mut tracker = PitchTracker::new(sample_rate, 60.0, 1000.0);
        let frame = make_sawtooth(220.0, sample_rate, 1600);
        let estimate = tracker.analyse(&frame).expect("pitched");
        assert!(
            (estimate.frequency - 220.0).abs() < 4.0,
            "freq = {}",
            estimate.frequency
        );
    }

    #[test]
    fn silence_has_no_pitch() {
        let mut tracker = PitchTracker::new(48_000.0, 60.0, 1000.0);
        let frame = vec![0.0f32; 4096];
        assert_eq!(tracker.analyse(&frame), None);
        assert!(tracker.is_polyphonic(&frame)); // treated as non-mono
    }

    #[test]
    fn chord_is_flagged_polyphonic() {
        let sample_rate = 48_000.0;
        let mut tracker = PitchTracker::new(sample_rate, 60.0, 1000.0);
        // An A-major triad (A3 220, C#4 277.18, E4 329.63) is inharmonic → its
        // autocorrelation never rises to the mono-voice confidence bar.
        let a = make_sine(220.0, sample_rate, 4096);
        let cs = make_sine(277.18, sample_rate, 4096);
        let e = make_sine(329.63, sample_rate, 4096);
        let chord: Vec<f32> = (0..a.len()).map(|n| (a[n] + cs[n] + e[n]) / 3.0).collect();
        assert!(tracker.is_polyphonic(&chord));
    }

    #[test]
    fn short_frame_returns_none() {
        let mut tracker = PitchTracker::new(48_000.0, 60.0, 1000.0);
        let frame = vec![0.1f32; 8];
        assert_eq!(tracker.analyse(&frame), None);
    }
}
