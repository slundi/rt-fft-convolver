//! Streaming monophonic pitch tracking on top of [`rt_pitch_core::PitchTracker`].
//!
//! The plugin delivers arbitrary host block sizes, but the autocorrelation
//! tracker needs a fixed *analysis frame* of at least `2 · max_period` samples
//! (see [`PitchTracker::min_frame_len`]).  [`VoicePitchTracker`] bridges the two
//! rates with the same RT-safe [`Analyzer`](rt_pitch_core::stft::Analyzer) FIFO
//! the phase vocoder uses: host blocks are pushed in, and a new estimate is
//! produced every `hop` samples once a full frame is buffered.
//!
//! Everything is allocation-free after construction — the analyzer ring and the
//! tracker's per-lag score buffer are both sized up front (Roadmap-08 "off-RT
//! setup, fixed-size scratch on the audio thread").

use rt_pitch_core::PitchEstimate;
use rt_pitch_core::PitchTracker;
use rt_pitch_core::stft::Analyzer;

/// Streaming wrapper: push host blocks, pull the latest monophonic pitch.
pub struct VoicePitchTracker {
    tracker: PitchTracker,
    analyzer: Analyzer,
    /// Reusable analysis-frame scratch (`frame` samples); never reallocated.
    frame: Vec<f32>,
    /// Most recent estimate, updated whenever a full frame is analysed.  `None`
    /// until the first frame is available (or when the input is silent/atonal).
    latest: Option<PitchEstimate>,
}

impl VoicePitchTracker {
    /// Build a tracker for fundamentals in `[min_freq, max_freq]` Hz, analysing
    /// a window every `hop` samples and absorbing host blocks up to `max_block`.
    ///
    /// The analysis frame length is derived from the tracker's requirement
    /// (`2 · max_period`, rounded up so `hop` divides evenly is *not* required).
    ///
    /// **Allocates** — construct off the audio thread.
    ///
    /// # Panics
    /// Panics unless `0 < min_freq < max_freq < sample_rate / 2` and `hop > 0`.
    pub fn new(
        sample_rate: f32,
        min_freq: f32,
        max_freq: f32,
        hop: usize,
        max_block: usize,
    ) -> Self {
        assert!(hop > 0, "hop must be non-zero");
        let tracker = PitchTracker::new(sample_rate, min_freq, max_freq);
        let frame_len = tracker.min_frame_len();
        assert!(
            hop <= frame_len,
            "hop ({hop}) must not exceed the analysis frame ({frame_len})"
        );
        Self {
            tracker,
            analyzer: Analyzer::new(frame_len, hop, max_block),
            frame: vec![0.0; frame_len],
            latest: None,
        }
    }

    /// Length of the analysis frame (samples) — useful for latency accounting.
    #[inline]
    pub fn frame_len(&self) -> usize {
        self.frame.len()
    }

    /// Feed a host input block and analyse every whole frame it completes.
    ///
    /// Returns the most recent estimate produced during this call, or the
    /// previously held estimate when the block did not complete a new frame.
    /// A silent or polyphonic frame clears the estimate to `None`.
    pub fn process_block(&mut self, block: &[f32]) -> Option<PitchEstimate> {
        self.analyzer.push_block(block);
        while self.analyzer.next_frame(&mut self.frame) {
            // The tracker's `scores` buffer is preallocated, so `analyse` is
            // allocation-free; each new frame overwrites the held estimate.
            self.latest = self.tracker.analyse(&self.frame);
        }
        self.latest
    }

    /// The latest estimate without consuming any input.
    #[inline]
    pub fn latest(&self) -> Option<PitchEstimate> {
        self.latest
    }

    /// `true` when the most recent frame was not a confident single voice
    /// (silent, noisy, or a chord).  Modes that require a monophonic input
    /// (PSOLA harmoniser voices, formant tracking) should bypass in that case.
    #[inline]
    pub fn is_monophonic(&self) -> bool {
        self.latest.is_some()
    }

    /// Drop all buffered input and forget the last estimate.
    pub fn reset(&mut self) {
        self.analyzer.reset();
        self.frame.fill(0.0);
        self.latest = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::make_sine;

    const SAMPLE_RATE: f32 = 48_000.0;

    #[test]
    fn tracks_mono_sine_across_blocks() {
        let hop = 512;
        let mut tracker = VoicePitchTracker::new(SAMPLE_RATE, 60.0, 1000.0, hop, 512);
        let signal = make_sine(220.0, SAMPLE_RATE, 8192);

        let mut estimate = None;
        for block in signal.chunks(512) {
            if let Some(latest) = tracker.process_block(block) {
                estimate = Some(latest);
            }
        }
        let estimate = estimate.expect("a pitched frame should have been analysed");
        assert!(
            (estimate.frequency - 220.0).abs() < 3.0,
            "freq = {}",
            estimate.frequency
        );
        assert!(estimate.confidence > 0.9, "conf = {}", estimate.confidence);
    }

    #[test]
    fn silence_yields_no_estimate() {
        let mut tracker = VoicePitchTracker::new(SAMPLE_RATE, 60.0, 1000.0, 512, 512);
        let silence = vec![0.0f32; 4096];
        for block in silence.chunks(512) {
            tracker.process_block(block);
        }
        assert!(tracker.latest().is_none());
        assert!(!tracker.is_monophonic());
    }

    #[test]
    fn reset_clears_state() {
        let mut tracker = VoicePitchTracker::new(SAMPLE_RATE, 60.0, 1000.0, 512, 1024);
        let signal = make_sine(330.0, SAMPLE_RATE, 8192);
        for block in signal.chunks(512) {
            tracker.process_block(block);
        }
        assert!(tracker.latest().is_some());
        tracker.reset();
        assert!(tracker.latest().is_none());
    }

    #[test]
    fn short_input_before_first_frame_has_no_estimate() {
        let mut tracker = VoicePitchTracker::new(SAMPLE_RATE, 60.0, 1000.0, 512, 256);
        // One tiny block, far short of the analysis frame.
        assert!(tracker.process_block(&[0.1; 128]).is_none());
    }
}
