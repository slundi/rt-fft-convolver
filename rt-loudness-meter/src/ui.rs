//! Framework-agnostic UI logic for the (Phase-3) editor.
//!
//! The egui drawing lives in the `gui`-gated `plugin` module; everything here
//! is pure, `egui`-free and therefore **unit-testable**:
//!
//! - [`format_lufs`] / [`format_lu`] / [`format_dbtp`] / [`format_dbfs`] —
//!   readout text (with a `−∞` floor for silence).
//! - [`lufs_to_fraction`] — LUFS → `[0, 1]` mapping for the scrolling graph and
//!   bar meters.
//! - [`LoudnessHistory`] — a fixed-capacity ring of recent short-term LUFS for
//!   the ~60 s scrolling graph.
//! - [`histogram_bins`] — bin recent short-term values into a display
//!   distribution (the LRA histogram panel).
//! - [`Compliance`] / [`compliance`] — the green / amber / red target LED.
//! - [`effective_running`] — fold host transport state into `Start/Stop`
//!   (hold integration while the transport is stopped).

/// LUFS value shown as digital silence in the readouts / graph floor.
pub const SILENCE_LUFS: f32 = -70.0;
/// Bottom of the loudness graph / bar scale (LUFS).
pub const GRAPH_FLOOR_LUFS: f32 = -60.0;
/// Top of the loudness graph / bar scale (LUFS).
pub const GRAPH_TOP_LUFS: f32 = 0.0;
/// Nominal seconds of short-term history shown in the scrolling graph.
pub const GRAPH_SECONDS: usize = 60;
/// Short-term samples per second the editor pushes into the graph (100 ms grid,
/// matching short-term's own cadence).
pub const GRAPH_SAMPLES_PER_SEC: usize = 10;
/// Capacity of the short-term history ring (`GRAPH_SECONDS · GRAPH_SAMPLES_PER_SEC`).
pub const GRAPH_LEN: usize = GRAPH_SECONDS * GRAPH_SAMPLES_PER_SEC;
/// Number of bars in the LRA-distribution histogram panel.
pub const HISTOGRAM_BINS: usize = 40;

/// Readout text for a LUFS value, e.g. `-23.0 LUFS` (or `-∞` at silence).
pub fn format_lufs(value_lufs: f32) -> String {
    if !value_lufs.is_finite() || value_lufs <= SILENCE_LUFS {
        "-∞ LUFS".to_string()
    } else {
        format!("{value_lufs:.1} LUFS")
    }
}

/// Readout text for a loudness-range value, e.g. `7.5 LU`.
pub fn format_lu(value_lu: f32) -> String {
    format!("{:.1} LU", value_lu.max(0.0))
}

/// Readout text for a true-peak value, e.g. `-1.0 dBTP` (or `-∞`).
pub fn format_dbtp(value_dbtp: f32) -> String {
    if value_dbtp.is_finite() {
        format!("{value_dbtp:.1} dBTP")
    } else {
        "-∞ dBTP".to_string()
    }
}

/// Readout text for a dBFS / dB value, e.g. `-3.2 dBFS` (or `-∞`).
pub fn format_dbfs(value_dbfs: f32) -> String {
    if value_dbfs.is_finite() {
        format!("{value_dbfs:.1} dBFS")
    } else {
        "-∞ dBFS".to_string()
    }
}

/// Map a LUFS value to a `[0, 1]` fraction over `[floor, top]`.  Non-finite
/// (silence) maps to `0`.
pub fn lufs_to_fraction(lufs: f32, floor: f32, top: f32) -> f32 {
    if top <= floor || !lufs.is_finite() {
        return 0.0;
    }
    ((lufs - floor) / (top - floor)).clamp(0.0, 1.0)
}

/// A fixed-capacity ring of recent short-term LUFS values for the scrolling
/// graph.  Newest values overwrite the oldest.
#[derive(Debug, Clone)]
pub struct LoudnessHistory {
    buffer: Vec<f32>,
    write: usize,
    len: usize,
}

impl LoudnessHistory {
    /// Create a history holding up to `capacity` samples.
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

    /// Append one short-term LUFS sample, evicting the oldest when full.
    pub fn push(&mut self, lufs: f32) {
        self.buffer[self.write] = lufs;
        self.write = (self.write + 1) % self.buffer.len();
        if self.len < self.buffer.len() {
            self.len += 1;
        }
    }

    /// Iterate the stored samples oldest → newest.
    pub fn iter(&self) -> impl Iterator<Item = f32> + '_ {
        let capacity = self.buffer.len();
        let start = if self.len < capacity { 0 } else { self.write };
        (0..self.len).map(move |offset| self.buffer[(start + offset) % capacity])
    }

    /// Clear the history.
    pub fn clear(&mut self) {
        self.write = 0;
        self.len = 0;
    }
}

/// Bin `values` into `num_bins` buckets over `[min, max]` (values outside the
/// range clamp to the end bins; non-finite values are skipped).  Used to draw
/// the short-term-loudness distribution (LRA histogram panel).
pub fn histogram_bins<I>(values: I, min: f32, max: f32, num_bins: usize) -> Vec<u32>
where
    I: IntoIterator<Item = f32>,
{
    let num_bins = num_bins.max(1);
    let mut bins = vec![0u32; num_bins];
    if max <= min {
        return bins;
    }
    for value in values {
        if !value.is_finite() {
            continue;
        }
        let fraction = ((value - min) / (max - min)).clamp(0.0, 1.0);
        let index = ((fraction * num_bins as f32) as usize).min(num_bins - 1);
        bins[index] = bins[index].saturating_add(1);
    }
    bins
}

/// Target-compliance state for the LED.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compliance {
    /// No integrated measurement yet.
    NoData,
    /// Within ±0.5 LU of target and under the true-peak ceiling (green).
    Compliant,
    /// Within ±2 LU of target and under the ceiling (amber).
    Close,
    /// More than 2 LU off target, or over the true-peak ceiling (red).
    Over,
}

/// Classify the current reading against the active target thresholds.
///
/// The true-peak ceiling is a hard limit — exceeding it is always [`Over`],
/// even if the integrated loudness is on target.
pub fn compliance(
    integrated_lufs: f32,
    true_peak_dbtp: f32,
    target_lufs: f32,
    ceiling_dbtp: f32,
) -> Compliance {
    if !integrated_lufs.is_finite() {
        return Compliance::NoData;
    }
    if true_peak_dbtp.is_finite() && true_peak_dbtp > ceiling_dbtp {
        return Compliance::Over;
    }
    let delta = (integrated_lufs - target_lufs).abs();
    if delta <= 0.5 {
        Compliance::Compliant
    } else if delta <= 2.0 {
        Compliance::Close
    } else {
        Compliance::Over
    }
}

/// Fold host transport state into the `Start/Stop` control: integration runs
/// only while the user has it enabled **and** the transport is playing.  When
/// the host reports no transport (`None`), the manual control decides.
#[inline]
pub fn effective_running(param_running: bool, transport_playing: Option<bool>) -> bool {
    param_running && transport_playing.unwrap_or(true)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lufs_formatting() {
        assert_eq!(format_lufs(-23.0), "-23.0 LUFS");
        assert_eq!(format_lufs(f32::NEG_INFINITY), "-∞ LUFS");
        assert_eq!(format_lufs(-80.0), "-∞ LUFS"); // below the silence floor
    }

    #[test]
    fn peak_and_lu_formatting() {
        assert_eq!(format_dbtp(-1.0), "-1.0 dBTP");
        assert_eq!(format_dbtp(f32::NEG_INFINITY), "-∞ dBTP");
        assert_eq!(format_dbfs(-3.2), "-3.2 dBFS");
        assert_eq!(format_lu(7.5), "7.5 LU");
        assert_eq!(format_lu(-1.0), "0.0 LU"); // never negative
    }

    #[test]
    fn lufs_fraction_maps_range() {
        assert!((lufs_to_fraction(0.0, -60.0, 0.0) - 1.0).abs() < 1e-6);
        assert!((lufs_to_fraction(-60.0, -60.0, 0.0) - 0.0).abs() < 1e-6);
        assert!((lufs_to_fraction(-30.0, -60.0, 0.0) - 0.5).abs() < 1e-6);
        assert_eq!(lufs_to_fraction(6.0, -60.0, 0.0), 1.0); // clamps
        assert_eq!(lufs_to_fraction(f32::NEG_INFINITY, -60.0, 0.0), 0.0);
        assert_eq!(lufs_to_fraction(-30.0, 0.0, 0.0), 0.0); // degenerate range
    }

    #[test]
    fn history_fills_then_scrolls() {
        let mut history = LoudnessHistory::new(4);
        assert!(history.is_empty());
        history.push(-10.0);
        history.push(-11.0);
        assert_eq!(history.iter().collect::<Vec<_>>(), vec![-10.0, -11.0]);
        history.push(-12.0);
        history.push(-13.0);
        history.push(-14.0); // evicts -10.0
        assert_eq!(history.len(), 4);
        assert_eq!(
            history.iter().collect::<Vec<_>>(),
            vec![-11.0, -12.0, -13.0, -14.0]
        );
    }

    #[test]
    fn history_clear_empties() {
        let mut history = LoudnessHistory::new(4);
        history.push(-10.0);
        history.clear();
        assert!(history.is_empty());
        assert_eq!(history.iter().count(), 0);
    }

    #[test]
    #[should_panic(expected = "history capacity must be non-zero")]
    fn history_zero_capacity_panics() {
        LoudnessHistory::new(0);
    }

    #[test]
    fn histogram_counts_into_bins() {
        // Values -30..-10 into 4 bins over [-30, -10].
        let values = vec![-30.0, -25.0, -20.0, -15.0, -11.0];
        let bins = histogram_bins(values, -30.0, -10.0, 4);
        assert_eq!(bins.iter().sum::<u32>(), 5);
        // -30 → bin 0, -25 → bin 1, -20 → bin 2, -15 → bin 3, -11 → bin 3.
        assert_eq!(bins, vec![1, 1, 1, 2]);
    }

    #[test]
    fn histogram_skips_non_finite_and_clamps() {
        let values = vec![f32::NEG_INFINITY, -100.0, 100.0];
        let bins = histogram_bins(values, -30.0, -10.0, 4);
        // -inf skipped; -100 clamps to bin 0; +100 clamps to last bin.
        assert_eq!(bins, vec![1, 0, 0, 1]);
    }

    #[test]
    fn histogram_degenerate_range_is_empty() {
        assert_eq!(histogram_bins(vec![1.0, 2.0], 0.0, 0.0, 3), vec![0, 0, 0]);
    }

    #[test]
    fn compliance_classifies_against_target() {
        // On target, under ceiling → green.
        assert_eq!(compliance(-23.0, -2.0, -23.0, -1.0), Compliance::Compliant);
        // 1 LU off → amber.
        assert_eq!(compliance(-24.0, -2.0, -23.0, -1.0), Compliance::Close);
        // 3 LU off → red.
        assert_eq!(compliance(-26.0, -2.0, -23.0, -1.0), Compliance::Over);
        // On target but peak over ceiling → red (hard limit).
        assert_eq!(compliance(-23.0, -0.5, -23.0, -1.0), Compliance::Over);
        // No integrated data yet.
        assert_eq!(
            compliance(f32::NEG_INFINITY, f32::NEG_INFINITY, -23.0, -1.0),
            Compliance::NoData
        );
    }

    #[test]
    fn transport_gates_integration() {
        // No transport info → manual control decides.
        assert!(effective_running(true, None));
        assert!(!effective_running(false, None));
        // With transport: hold while stopped.
        assert!(effective_running(true, Some(true)));
        assert!(!effective_running(true, Some(false)));
        assert!(!effective_running(false, Some(true)));
    }

    #[test]
    fn graph_len_is_sixty_seconds() {
        assert_eq!(GRAPH_LEN, 600);
    }
}
