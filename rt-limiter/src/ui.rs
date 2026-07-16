//! Framework-agnostic UI logic for the (Phase-3) editor.
//!
//! The egui drawing lives in the `gui`-gated `plugin` module; everything here
//! is pure, `egui`-free and therefore **unit-testable**:
//!
//! - [`peak_dbfs`] — per-block peak measurement (audio thread) feeding the
//!   input / output meter atomics.
//! - [`LevelMeter`] — display-side VU ballistics (instant attack, eased
//!   release) for the true-peak meters.
//! - [`PeakHold`] — the peak-hold marker on the gain-reduction meter.
//! - [`GrHistory`] — a fixed-capacity ring of recent gain-reduction values for
//!   the scrolling history graph.
//! - [`dbfs_to_fraction`] / [`gr_to_fraction`] — dB → `[0, 1]` bar mappings.
//! - [`format_db`] / [`format_gr`] / [`format_latency_ms`] — readout text.

use rt_fft_convolver::linear_to_db;

/// The dBFS value used to represent digital silence in the meters.
pub const SILENCE_DBFS: f32 = -120.0;

/// Bottom of the input / output meter scale, in dBFS.
pub const METER_FLOOR_DBFS: f32 = -60.0;

/// Top of the gain-reduction meter scale, in dB of reduction.
pub const MAX_GR_DB: f32 = 24.0;

/// Number of samples kept in the scrolling gain-reduction history.
pub const GR_HISTORY_LEN: usize = 256;

/// Peak level of a block in dBFS, floored at [`SILENCE_DBFS`].
///
/// Called on the audio thread (cheap, no allocation) to feed a meter atomic.
pub fn peak_dbfs(samples: &[f32]) -> f32 {
    let peak = samples
        .iter()
        .fold(0.0f32, |maximum, &sample| maximum.max(sample.abs()));
    if peak <= 1.0e-6 {
        SILENCE_DBFS
    } else {
        linear_to_db(peak)
    }
}

/// Map a dBFS reading to a `[0, 1]` bar fraction over `[floor, 0] dB`.
pub fn dbfs_to_fraction(dbfs: f32, floor_db: f32) -> f32 {
    if floor_db >= 0.0 {
        return 0.0;
    }
    ((dbfs - floor_db) / -floor_db).clamp(0.0, 1.0)
}

/// Map a gain-reduction reading (in dB, `≤ 0`) to a `[0, 1]` bar fraction over
/// `[0, max_gr_db]` of reduction.
pub fn gr_to_fraction(gr_db: f32, max_gr_db: f32) -> f32 {
    if max_gr_db <= 0.0 {
        return 0.0;
    }
    (gr_db.abs() / max_gr_db).clamp(0.0, 1.0)
}

/// Display-side VU ballistics: instant attack, exponential release.
///
/// The audio thread publishes a per-block peak; the editor feeds it here each
/// frame so the bar snaps up to peaks and eases back down.
#[derive(Debug, Clone, Copy)]
pub struct LevelMeter {
    current_dbfs: f32,
}

impl LevelMeter {
    /// A meter reading digital silence.
    pub fn new() -> Self {
        Self {
            current_dbfs: SILENCE_DBFS,
        }
    }

    /// Update toward `target_dbfs`.  Rising targets are taken instantly (peak
    /// attack); falling targets ease down by `release` (`0..=1`, larger =
    /// faster).  Returns the new displayed value.
    pub fn update(&mut self, target_dbfs: f32, release: f32) -> f32 {
        if target_dbfs >= self.current_dbfs {
            self.current_dbfs = target_dbfs;
        } else {
            let release = release.clamp(0.0, 1.0);
            self.current_dbfs += (target_dbfs - self.current_dbfs) * release;
        }
        self.current_dbfs
    }

    /// The current displayed level in dBFS.
    #[inline]
    pub fn value_dbfs(&self) -> f32 {
        self.current_dbfs
    }
}

impl Default for LevelMeter {
    fn default() -> Self {
        Self::new()
    }
}

/// A peak-hold marker: latches the largest value for a number of frames, then
/// eases back down.  Used for the "max GR" marker on the reduction meter.
///
/// Values are held as *magnitudes* (larger = more prominent), so it works for
/// either a level in dBFS-above-floor or a gain-reduction magnitude in dB.
#[derive(Debug, Clone, Copy)]
pub struct PeakHold {
    held: f32,
    timer: u32,
    hold_frames: u32,
    decay: f32,
}

impl PeakHold {
    /// Hold a new peak for `hold_frames` editor frames, then ease down by
    /// `decay` (`0..=1`) per frame.
    pub fn new(hold_frames: u32, decay: f32) -> Self {
        Self {
            held: 0.0,
            timer: 0,
            hold_frames,
            decay: decay.clamp(0.0, 1.0),
        }
    }

    /// Feed the current value and return the held peak.
    pub fn update(&mut self, value: f32) -> f32 {
        if value >= self.held {
            self.held = value;
            self.timer = self.hold_frames;
        } else if self.timer > 0 {
            self.timer -= 1;
        } else {
            self.held += (value - self.held) * self.decay;
        }
        self.held
    }

    /// The current held peak.
    #[inline]
    pub fn value(&self) -> f32 {
        self.held
    }
}

/// A fixed-capacity ring of recent gain-reduction values (dB, `≤ 0`) for the
/// scrolling history graph.  Newest values overwrite the oldest.
#[derive(Debug, Clone)]
pub struct GrHistory {
    buffer: Vec<f32>,
    write: usize,
    len: usize,
}

impl GrHistory {
    /// Create a history buffer holding up to `capacity` samples (initialised
    /// empty; unfilled slots are not yielded).
    ///
    /// # Panics
    ///
    /// Panics if `capacity == 0`.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "history capacity must be non-zero");
        Self {
            buffer: vec![0.0; capacity],
            write: 0,
            len: 0,
        }
    }

    /// Capacity in samples.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Number of samples currently stored (`≤ capacity`).
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` when nothing has been pushed yet.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Append one gain-reduction sample, evicting the oldest when full.
    pub fn push(&mut self, gr_db: f32) {
        self.buffer[self.write] = gr_db;
        self.write = (self.write + 1) % self.buffer.len();
        if self.len < self.buffer.len() {
            self.len += 1;
        }
    }

    /// Iterate the stored samples oldest → newest.
    pub fn iter(&self) -> impl Iterator<Item = f32> + '_ {
        let capacity = self.buffer.len();
        // Oldest index: when full, it is `write`; otherwise the buffer filled
        // from index 0.
        let start = if self.len < capacity { 0 } else { self.write };
        (0..self.len).map(move |offset| self.buffer[(start + offset) % capacity])
    }

    /// Clear the history.
    pub fn clear(&mut self) {
        self.write = 0;
        self.len = 0;
    }
}

/// Readout text for a dB value, e.g. `-1.0 dB` (or `-∞` at the silence floor).
pub fn format_db(value_db: f32) -> String {
    if value_db <= SILENCE_DBFS {
        "-∞ dB".to_string()
    } else {
        format!("{value_db:.1} dB")
    }
}

/// Readout text for a gain-reduction value (`≤ 0`) as a positive reduction,
/// e.g. `4.5 dB` (or `0.0 dB` when not limiting).
pub fn format_gr(gr_db: f32) -> String {
    format!("{:.1} dB", gr_db.abs())
}

/// Readout text for a latency in samples at `sample_rate`, e.g. `5.2 ms`.
pub fn format_latency_ms(latency_samples: usize, sample_rate: f32) -> String {
    let milliseconds = if sample_rate > 0.0 {
        latency_samples as f32 / sample_rate * 1000.0
    } else {
        0.0
    };
    format!("{milliseconds:.1} ms")
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_dbfs_of_silence_is_floor() {
        assert_eq!(peak_dbfs(&[0.0; 16]), SILENCE_DBFS);
        assert_eq!(peak_dbfs(&[]), SILENCE_DBFS);
    }

    #[test]
    fn peak_dbfs_of_full_scale_is_zero() {
        assert!((peak_dbfs(&[0.5, -1.0, 0.25]) - 0.0).abs() < 1e-4);
    }

    #[test]
    fn dbfs_fraction_maps_floor_to_top() {
        assert!((dbfs_to_fraction(0.0, -60.0) - 1.0).abs() < 1e-6);
        assert!((dbfs_to_fraction(-60.0, -60.0) - 0.0).abs() < 1e-6);
        assert!((dbfs_to_fraction(-30.0, -60.0) - 0.5).abs() < 1e-6);
        // Out of range clamps.
        assert_eq!(dbfs_to_fraction(6.0, -60.0), 1.0);
        assert_eq!(dbfs_to_fraction(-120.0, -60.0), 0.0);
        // Degenerate floor.
        assert_eq!(dbfs_to_fraction(-6.0, 0.0), 0.0);
    }

    #[test]
    fn gr_fraction_maps_reduction() {
        assert!((gr_to_fraction(0.0, 24.0) - 0.0).abs() < 1e-6);
        assert!((gr_to_fraction(-12.0, 24.0) - 0.5).abs() < 1e-6);
        assert_eq!(gr_to_fraction(-48.0, 24.0), 1.0);
        assert_eq!(gr_to_fraction(-6.0, 0.0), 0.0);
    }

    #[test]
    fn meter_attacks_instantly_and_releases_gradually() {
        let mut meter = LevelMeter::new();
        assert!((meter.update(-6.0, 0.2) - -6.0).abs() < 1e-6);
        let after = meter.update(-24.0, 0.5);
        assert!(
            after > -24.0 && after < -6.0,
            "release not gradual: {after}"
        );
        assert!((meter.value_dbfs() - after).abs() < 1e-6);
    }

    #[test]
    fn peak_hold_latches_then_decays() {
        let mut hold = PeakHold::new(2, 0.5);
        assert!((hold.update(10.0) - 10.0).abs() < 1e-6); // latch
        assert_eq!(hold.update(0.0), 10.0); // held (frame 1)
        assert_eq!(hold.update(0.0), 10.0); // held (frame 2)
        let decayed = hold.update(0.0); // now decays
        assert!(decayed < 10.0 && decayed > 0.0, "did not decay: {decayed}");
        assert!((hold.value() - decayed).abs() < 1e-6);
    }

    #[test]
    fn peak_hold_relatches_on_higher_value() {
        let mut hold = PeakHold::new(1, 0.5);
        hold.update(5.0);
        hold.update(0.0); // hold
        hold.update(0.0); // decay a bit
        let relatched = hold.update(8.0);
        assert!((relatched - 8.0).abs() < 1e-6);
    }

    #[test]
    fn history_fills_then_scrolls() {
        let mut history = GrHistory::new(4);
        assert!(history.is_empty());
        history.push(-1.0);
        history.push(-2.0);
        assert_eq!(history.len(), 2);
        assert_eq!(history.iter().collect::<Vec<_>>(), vec![-1.0, -2.0]);

        history.push(-3.0);
        history.push(-4.0); // now full
        history.push(-5.0); // evicts -1.0
        assert_eq!(history.len(), 4);
        assert_eq!(history.capacity(), 4);
        // Oldest → newest, with the first value scrolled off.
        assert_eq!(
            history.iter().collect::<Vec<_>>(),
            vec![-2.0, -3.0, -4.0, -5.0]
        );
    }

    #[test]
    fn history_clear_empties() {
        let mut history = GrHistory::new(4);
        history.push(-1.0);
        history.clear();
        assert!(history.is_empty());
        assert_eq!(history.iter().count(), 0);
    }

    #[test]
    #[should_panic(expected = "history capacity must be non-zero")]
    fn history_zero_capacity_panics() {
        GrHistory::new(0);
    }

    #[test]
    fn formatters() {
        assert_eq!(format_db(-1.0), "-1.0 dB");
        assert_eq!(format_db(SILENCE_DBFS), "-∞ dB");
        assert_eq!(format_gr(-4.5), "4.5 dB");
        assert_eq!(format_gr(0.0), "0.0 dB");
        assert_eq!(format_latency_ms(240, 48_000.0), "5.0 ms");
        assert_eq!(format_latency_ms(100, 0.0), "0.0 ms");
    }
}
