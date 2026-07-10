//! Framework-agnostic UI-state logic shared by both plugin editors.
//!
//! The actual rendering is `nih-plug-egui` code inside each plugin crate (see
//! `docs/UI-lv2.md`), but the *decisions* that drive it — what state a slot pad
//! is in, the down-sampled waveform to draw, the MIDI-activity blink, the info
//! line — are pure functions with no GUI dependency, so they live here and are
//! unit-tested.

/// Visual state of one slot pad (see `docs/UI-lv2.md` § Slot pad).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadState {
    /// No IR loaded — greyed out.
    Empty,
    /// Loaded and enabled, but not the active slot.
    Filled,
    /// Loaded, enabled, and currently active — accent highlight.
    Active,
    /// Loaded but disabled — plays dry; dimmed.
    Disabled,
}

/// Decide a slot pad's visual state from its flags.
///
/// Precedence: an unloaded slot is always [`Empty`](PadState::Empty); a loaded
/// but disabled slot is [`Disabled`](PadState::Disabled); otherwise the active
/// slot is [`Active`](PadState::Active) and the rest are
/// [`Filled`](PadState::Filled).
pub fn pad_state(loaded: bool, enabled: bool, active: bool) -> PadState {
    if !loaded {
        PadState::Empty
    } else if !enabled {
        PadState::Disabled
    } else if active {
        PadState::Active
    } else {
        PadState::Filled
    }
}

/// A down-sampled min/max envelope for drawing an IR waveform preview.
///
/// `peaks[i]` is the `(min, max)` of bucket `i`, **normalised** to the IR's
/// peak magnitude so the trace always fills the frame.  `peak` is the original
/// peak magnitude (for the info line / amplitude readout).
#[derive(Debug, Clone, PartialEq)]
pub struct WaveformSummary {
    /// `(min, max)` per bucket, each in `-1.0..=1.0` after normalisation.
    pub peaks: Vec<(f32, f32)>,
    /// Original peak magnitude (`max |sample|`) before normalisation.
    pub peak: f32,
}

/// Down-sample `samples` to at most `buckets` `(min, max)` pairs, normalised to
/// the signal peak.
///
/// Returns an empty summary when `samples` is empty or `buckets == 0`.  When
/// `buckets` exceeds the sample count the result has one bucket per sample.
/// **Off-thread** (allocates) — computed once when an IR is loaded, not per
/// frame.
pub fn compute_waveform(samples: &[f32], buckets: usize) -> WaveformSummary {
    if samples.is_empty() || buckets == 0 {
        return WaveformSummary {
            peaks: Vec::new(),
            peak: 0.0,
        };
    }

    let peak = samples.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
    let norm = if peak > 0.0 { 1.0 / peak } else { 0.0 };

    let n = samples.len();
    let buckets = buckets.min(n);
    let mut peaks = Vec::with_capacity(buckets);
    for b in 0..buckets {
        // Even split of the sample range across buckets.
        let start = b * n / buckets;
        let end = ((b + 1) * n / buckets).max(start + 1);
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &s in &samples[start..end] {
            lo = lo.min(s);
            hi = hi.max(s);
        }
        peaks.push((lo * norm, hi * norm));
    }

    WaveformSummary { peaks, peak }
}

/// Whether the last routed MIDI event mapped to a slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MidiStatus {
    /// No recent activity.
    Idle,
    /// The last event mapped to a slot (green LED).
    Recognised,
    /// The last event did not map to a slot (red LED).
    Unrecognised,
}

/// A decaying MIDI-activity indicator for the header LED.
///
/// The editor calls [`trigger`](Self::trigger) when a MIDI event arrives and
/// [`decay`](Self::decay) once per rendered frame; [`level`](Self::level)
/// drives the LED brightness and [`status`](Self::status) its colour.
#[derive(Debug, Clone, Copy)]
pub struct MidiActivity {
    level: f32,
    status: MidiStatus,
}

impl MidiActivity {
    /// A dark, idle LED.
    pub fn new() -> Self {
        Self {
            level: 0.0,
            status: MidiStatus::Idle,
        }
    }

    /// Flash the LED; `recognised` picks the colour.
    pub fn trigger(&mut self, recognised: bool) {
        self.level = 1.0;
        self.status = if recognised {
            MidiStatus::Recognised
        } else {
            MidiStatus::Unrecognised
        };
    }

    /// Fade the LED by `amount` (clamped at 0); returns to idle when dark.
    pub fn decay(&mut self, amount: f32) {
        self.level = (self.level - amount).max(0.0);
        if self.level == 0.0 {
            self.status = MidiStatus::Idle;
        }
    }

    /// Current brightness, `0.0..=1.0`.
    #[inline]
    pub fn level(&self) -> f32 {
        self.level
    }

    /// Current LED status/colour.
    #[inline]
    pub fn status(&self) -> MidiStatus {
        self.status
    }
}

impl Default for MidiActivity {
    fn default() -> Self {
        Self::new()
    }
}

/// IR duration in milliseconds from a tap count and sample rate.
#[inline]
pub fn ir_duration_ms(taps: usize, sample_rate: u32) -> f32 {
    if sample_rate == 0 {
        0.0
    } else {
        taps as f32 * 1000.0 / sample_rate as f32
    }
}

/// Human label for an IR's channel count (for the info line).
pub fn channel_label(channels: usize) -> &'static str {
    match channels {
        1 => "mono",
        2 => "stereo",
        4 => "true-stereo",
        _ => "multi",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_state_precedence() {
        assert_eq!(pad_state(false, true, false), PadState::Empty);
        assert_eq!(pad_state(false, true, true), PadState::Empty); // empty wins over active
        assert_eq!(pad_state(true, false, true), PadState::Disabled); // disabled wins over active
        assert_eq!(pad_state(true, true, true), PadState::Active);
        assert_eq!(pad_state(true, true, false), PadState::Filled);
    }

    #[test]
    fn waveform_empty_input_is_empty() {
        let w = compute_waveform(&[], 16);
        assert!(w.peaks.is_empty());
        assert_eq!(w.peak, 0.0);
        assert!(compute_waveform(&[1.0, 2.0], 0).peaks.is_empty());
    }

    #[test]
    fn waveform_normalises_to_peak() {
        // Peak magnitude is 4.0; normalised max must reach 1.0.
        let samples = [0.0, 2.0, -4.0, 1.0];
        let w = compute_waveform(&samples, 2);
        assert_eq!(w.peak, 4.0);
        assert_eq!(w.peaks.len(), 2);
        // Bucket 1 contains -4.0 → normalised min = -1.0.
        let global_min = w
            .peaks
            .iter()
            .map(|&(lo, _)| lo)
            .fold(f32::INFINITY, f32::min);
        let global_max = w
            .peaks
            .iter()
            .map(|&(_, hi)| hi)
            .fold(f32::NEG_INFINITY, f32::max);
        assert!((global_min + 1.0).abs() < 1e-6, "min {global_min}");
        assert!((global_max - 0.5).abs() < 1e-6, "max {global_max}"); // 2.0/4.0
    }

    #[test]
    fn waveform_buckets_capped_at_sample_count() {
        let w = compute_waveform(&[1.0, -1.0, 0.5], 16);
        assert_eq!(
            w.peaks.len(),
            3,
            "more buckets than samples → one per sample"
        );
    }

    #[test]
    fn waveform_bucket_captures_min_and_max() {
        // Single bucket over the whole signal: min/max span it.
        let w = compute_waveform(&[0.25, -0.5, 1.0, -0.75], 1);
        assert_eq!(w.peaks.len(), 1);
        let (lo, hi) = w.peaks[0];
        assert!((lo + 0.75).abs() < 1e-6); // min −0.75, peak 1.0
        assert!((hi - 1.0).abs() < 1e-6);
    }

    #[test]
    fn silent_ir_normalises_without_nan() {
        let w = compute_waveform(&[0.0, 0.0, 0.0], 3);
        assert_eq!(w.peak, 0.0);
        assert!(w.peaks.iter().all(|&(lo, hi)| lo == 0.0 && hi == 0.0));
    }

    #[test]
    fn midi_activity_triggers_and_decays() {
        let mut a = MidiActivity::new();
        assert_eq!(a.status(), MidiStatus::Idle);
        assert_eq!(a.level(), 0.0);

        a.trigger(true);
        assert_eq!(a.status(), MidiStatus::Recognised);
        assert_eq!(a.level(), 1.0);

        a.decay(0.3);
        assert!((a.level() - 0.7).abs() < 1e-6);
        assert_eq!(a.status(), MidiStatus::Recognised);

        a.decay(1.0); // over-decay clamps to 0 and returns to idle
        assert_eq!(a.level(), 0.0);
        assert_eq!(a.status(), MidiStatus::Idle);

        a.trigger(false);
        assert_eq!(a.status(), MidiStatus::Unrecognised);
    }

    #[test]
    fn ir_duration_and_labels() {
        assert!((ir_duration_ms(480, 48_000) - 10.0).abs() < 1e-4);
        assert_eq!(ir_duration_ms(100, 0), 0.0);
        assert_eq!(channel_label(1), "mono");
        assert_eq!(channel_label(2), "stereo");
        assert_eq!(channel_label(4), "true-stereo");
        assert_eq!(channel_label(3), "multi");
    }
}
