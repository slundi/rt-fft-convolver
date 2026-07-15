//! Framework-agnostic UI logic for the (Phase-3) `nih-plug-egui` editor.
//!
//! The egui drawing lives in the `gui`-gated `plugin` module; everything here
//! is pure, `egui`-free and therefore **unit-testable**:
//!
//! - [`format_interval`] — the preset-rack pad label (semitones + cents).
//! - [`latency_ms`] / [`format_latency_ms`] — the latency badge (Phase-3
//!   "latency badge showing current reported latency in ms").
//! - [`PitchTrajectory`] — the rolling, display-smoothed history line behind
//!   the real-time pitch-trajectory plot (useful for expression-pedal / whammy
//!   automation).
//!
//! The slot-pad visual state and the MIDI-activity LED reuse the shared
//! [`pad_state`](rt_ir_plugin_core::pad_state) /
//! [`MidiActivity`](rt_ir_plugin_core::MidiActivity) helpers from
//! `rt-ir-plugin-core`, re-exported here for the editor's convenience.

pub use rt_ir_plugin_core::{MidiActivity, MidiStatus, PadState, pad_state};

/// Format an interval as a semitone + cents pad label, e.g. `"+7 st"`,
/// `"+20 ¢"`, `"-7 st -25 ¢"`, or `"unison"` at zero.
///
/// Cents are rounded to the nearest whole; the sign is carried on each nonzero
/// component so a shift below a semitone reads unambiguously.
pub fn format_interval(semitones: f32) -> String {
    let total_cents = (semitones * 100.0).round() as i32;
    let st = total_cents / 100;
    let cents = total_cents % 100;
    match (st, cents) {
        (0, 0) => "unison".to_string(),
        (_, 0) => format!("{st:+} st"),
        (0, _) => format!("{cents:+} ¢"),
        _ => format!("{st:+} st {cents:+} ¢"),
    }
}

/// Convert a latency in samples to milliseconds at `sample_rate`.
///
/// Returns `0.0` for a non-positive sample rate (badge shows before the host
/// has reported one).
#[inline]
pub fn latency_ms(samples: usize, sample_rate: f32) -> f32 {
    if sample_rate <= 0.0 {
        0.0
    } else {
        samples as f32 * 1000.0 / sample_rate
    }
}

/// Badge text for a latency in milliseconds, e.g. `"21.3 ms"`.
#[inline]
pub fn format_latency_ms(ms: f32) -> String {
    format!("{ms:.1} ms")
}

/// A rolling, display-smoothed history of the current output pitch (in
/// semitones) for the real-time trajectory plot.
///
/// The audio thread publishes the selected interval each block; the editor
/// feeds it here once per frame.  A one-pole smoother eases the line between
/// stepped preset changes so the plot shows the crossfade/whammy glide rather
/// than a hard jump, and a fixed-capacity ring keeps the most recent frames in
/// draw order without reallocating.
#[derive(Debug, Clone)]
pub struct PitchTrajectory {
    /// Ring of displayed (smoothed) semitone values.
    history: Vec<f32>,
    /// Index of the next write (oldest sample when the ring is full).
    head: usize,
    /// Number of valid samples (`< capacity` only while warming up).
    len: usize,
    /// Current smoothed display value.
    display: f32,
    /// Per-frame one-pole blend coefficient in `(0, 1]`.
    coeff: f32,
}

impl PitchTrajectory {
    /// Build a trajectory holding `capacity` frames, easing toward each new
    /// target by `smoothing` (`0..=1`; `1.0` = no smoothing / instant).
    ///
    /// # Panics
    /// Panics if `capacity == 0`.
    pub fn new(capacity: usize, smoothing: f32) -> Self {
        assert!(capacity > 0, "trajectory capacity must be > 0");
        Self {
            history: vec![0.0; capacity],
            head: 0,
            len: 0,
            display: 0.0,
            coeff: smoothing.clamp(1.0e-3, 1.0),
        }
    }

    /// Advance one frame toward `target_semitones`, append the smoothed value
    /// to the history, and return it.
    pub fn push(&mut self, target_semitones: f32) -> f32 {
        self.display += self.coeff * (target_semitones - self.display);
        self.history[self.head] = self.display;
        self.head = (self.head + 1) % self.history.len();
        self.len = (self.len + 1).min(self.history.len());
        self.display
    }

    /// The history in draw order (oldest → newest).  Allocates a fresh `Vec`
    /// (editor thread, once per frame) — never called on the audio thread.
    pub fn ordered(&self) -> Vec<f32> {
        let capacity = self.history.len();
        if self.len < capacity {
            // Not yet wrapped: the valid samples are `history[0..len]`.
            self.history[..self.len].to_vec()
        } else {
            // Wrapped: oldest sample sits at `head`.
            let (tail, front) = self.history.split_at(self.head);
            front.iter().chain(tail.iter()).copied().collect()
        }
    }

    /// Number of frames currently stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether no frame has been pushed yet.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The latest smoothed value without advancing.
    #[inline]
    pub fn current(&self) -> f32 {
        self.display
    }

    /// Snap the display to `value` and forget the history (transport stop /
    /// editor open).
    pub fn reset(&mut self, value: f32) {
        self.display = value;
        self.head = 0;
        self.len = 0;
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_labels() {
        assert_eq!(format_interval(0.0), "unison");
        assert_eq!(format_interval(7.0), "+7 st");
        assert_eq!(format_interval(-12.0), "-12 st");
        assert_eq!(format_interval(0.2), "+20 ¢");
        assert_eq!(format_interval(-0.05), "-5 ¢");
        assert_eq!(format_interval(7.25), "+7 st +25 ¢");
        assert_eq!(format_interval(-7.25), "-7 st -25 ¢");
    }

    #[test]
    fn latency_conversion_and_badge() {
        // 1024 samples @ 48 kHz ≈ 21.3 ms.
        let ms = latency_ms(1024, 48_000.0);
        assert!((ms - 21.333).abs() < 1e-2, "ms = {ms}");
        assert_eq!(format_latency_ms(ms), "21.3 ms");
        // Guard against a not-yet-known sample rate.
        assert_eq!(latency_ms(1024, 0.0), 0.0);
    }

    #[test]
    fn trajectory_smooths_toward_target() {
        let mut trajectory = PitchTrajectory::new(64, 0.25);
        // Step to +12 semitones; the display eases up, never overshooting.
        let mut last = 0.0;
        for _ in 0..200 {
            let value = trajectory.push(12.0);
            assert!(value >= last - 1e-6 && value <= 12.0 + 1e-6);
            last = value;
        }
        assert!((trajectory.current() - 12.0).abs() < 1e-2);
    }

    #[test]
    fn trajectory_history_is_in_draw_order() {
        // Instant smoothing so the pushed values land verbatim.
        let mut trajectory = PitchTrajectory::new(4, 1.0);
        for value in [1.0, 2.0, 3.0] {
            trajectory.push(value);
        }
        assert_eq!(trajectory.len(), 3);
        assert_eq!(trajectory.ordered(), vec![1.0, 2.0, 3.0]);

        // Overflow the ring: oldest (1.0, 2.0) drop off, order preserved.
        for value in [4.0, 5.0] {
            trajectory.push(value);
        }
        assert_eq!(trajectory.len(), 4);
        assert_eq!(trajectory.ordered(), vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn trajectory_reset_clears_history() {
        let mut trajectory = PitchTrajectory::new(8, 0.5);
        for _ in 0..4 {
            trajectory.push(5.0);
        }
        assert!(!trajectory.is_empty());
        trajectory.reset(0.0);
        assert!(trajectory.is_empty());
        assert_eq!(trajectory.current(), 0.0);
    }

    #[test]
    fn preset_pad_state_reuse() {
        // A pitch preset is always "loaded"; map enabled / active through.
        assert_eq!(pad_state(true, true, true), PadState::Active);
        assert_eq!(pad_state(true, true, false), PadState::Filled);
        assert_eq!(pad_state(true, false, false), PadState::Disabled);
    }
}
