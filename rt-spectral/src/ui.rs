//! Framework-agnostic UI logic for the (Phase-3) `nih-plug-egui` editor.
//!
//! The egui drawing lives in the `gui`-gated `plugin` module; everything here
//! is pure, `egui`-free and therefore **unit-testable**:
//!
//! - [`downsample_log`] — collapse a linear FFT magnitude spectrum onto
//!   log-spaced display bins (the spectrogram / spectrum row axis).
//! - [`magnitude_to_db`] / [`db_to_intensity`] — the magnitude → dB → colour
//!   intensity mapping behind the spectrogram and the noise-profile overlay.
//! - [`Spectrogram`] — the rolling column history behind the real-time
//!   spectrogram (Phase-3 "real-time spectrogram, last N seconds").
//! - [`Meter`] — peak-hold + decay ballistics for the input / output /
//!   gain-reduction meters.
//! - [`latency_ms`] / [`format_latency_ms`] / [`format_db`] — badge / readout
//!   formatting.

/// Number of log-spaced display bins per spectrogram column (and the spectrum
/// row).  Sized for the display, **never** the RT path (see the Roadmap cost
/// model: the RT path is `N/2 + 1` linear bins; this is the down-sampled view).
pub const SPECTROGRAM_BINS: usize = 128;

/// Number of columns (time slices) the editor's spectrogram retains — roughly
/// a few seconds of history at a typical editor frame rate.
pub const SPECTROGRAM_COLUMNS: usize = 256;

/// dB floor for the spectrogram / spectrum intensity mapping.
pub const DISPLAY_FLOOR_DB: f32 = -90.0;
/// dB ceiling for the spectrogram / spectrum intensity mapping.
pub const DISPLAY_CEIL_DB: f32 = 0.0;

/// Convert a linear magnitude to dBFS, floored at `floor_db` (so `0` magnitude
/// maps to the floor rather than `-inf`).
#[inline]
pub fn magnitude_to_db(magnitude: f32, floor_db: f32) -> f32 {
    (20.0 * magnitude.max(1e-9).log10()).max(floor_db)
}

/// Map a dB level onto a `0.0..=1.0` display intensity between `floor_db` and
/// `ceil_db` (values outside the window clamp to the ends).
#[inline]
pub fn db_to_intensity(db: f32, floor_db: f32, ceil_db: f32) -> f32 {
    if ceil_db <= floor_db {
        return 0.0;
    }
    ((db - floor_db) / (ceil_db - floor_db)).clamp(0.0, 1.0)
}

/// Collapse a linear FFT magnitude spectrum (`mags`, length `N/2 + 1`) onto
/// `out.len()` **log-spaced** display bins, taking the peak magnitude in each
/// band.  Log spacing over the bin index (`1 ..= K−1`, skipping DC) tracks
/// perceived pitch far better than a linear axis.
///
/// Robust to any lengths: an empty `out` is a no-op; a `mags` shorter than two
/// bins fills `out` with zeros.
pub fn downsample_log(mags: &[f32], out: &mut [f32]) {
    let num = out.len();
    if num == 0 {
        return;
    }
    let last = mags.len().saturating_sub(1);
    if last < 1 {
        out.fill(0.0);
        return;
    }
    // Log-lerp over bin indices: idx(t) = last^t, t ∈ [0, 1] (last^0 = 1 skips
    // the DC bin, last^1 = last hits Nyquist).
    let ln_last = (last as f32).ln();
    let index_at = |t: f32| -> usize {
        let idx = (t * ln_last).exp().round() as usize;
        idx.clamp(1, last)
    };
    for (bin, slot) in out.iter_mut().enumerate() {
        let lo = index_at(bin as f32 / num as f32);
        let hi = index_at((bin + 1) as f32 / num as f32).max(lo);
        let mut peak = 0.0f32;
        for &m in &mags[lo..=hi] {
            peak = peak.max(m);
        }
        *slot = peak;
    }
}

/// A rolling column history behind the real-time spectrogram.
///
/// Each pushed column is `bins` display-bin intensities (or dB values — the
/// store is agnostic); a fixed-capacity ring keeps the most recent
/// `capacity` columns in draw order without reallocating.  The editor pushes
/// one column per frame from the latest audio-thread telemetry and reads the
/// columns back oldest → newest to paint the image.
#[derive(Debug, Clone)]
pub struct Spectrogram {
    bins: usize,
    capacity: usize,
    /// `capacity × bins`, column-major: column `c` occupies `[c*bins .. )`.
    data: Vec<f32>,
    /// Index of the next column to write (oldest column when full).
    head: usize,
    /// Number of valid columns (`< capacity` only while warming up).
    len: usize,
}

impl Spectrogram {
    /// Build a spectrogram holding `capacity` columns of `bins` rows each.
    ///
    /// # Panics
    /// Panics if `bins == 0` or `capacity == 0`.
    pub fn new(bins: usize, capacity: usize) -> Self {
        assert!(bins > 0 && capacity > 0, "spectrogram dims must be > 0");
        Self {
            bins,
            capacity,
            data: vec![0.0; bins * capacity],
            head: 0,
            len: 0,
        }
    }

    /// Append `column` as the newest time slice.  Copies up to `bins` values
    /// (shorter columns zero-pad the remainder; longer columns are truncated),
    /// so a mismatched telemetry width never panics.
    pub fn push_column(&mut self, column: &[f32]) {
        let start = self.head * self.bins;
        let slot = &mut self.data[start..start + self.bins];
        let take = column.len().min(self.bins);
        slot[..take].copy_from_slice(&column[..take]);
        slot[take..].fill(0.0);
        self.head = (self.head + 1) % self.capacity;
        self.len = (self.len + 1).min(self.capacity);
    }

    /// The `ordered`-th oldest column currently stored (`0` = oldest,
    /// `len()-1` = newest).  Returns `None` when out of range.
    pub fn column(&self, ordered: usize) -> Option<&[f32]> {
        if ordered >= self.len {
            return None;
        }
        // When full, the oldest column sits at `head`; before wrap, at 0.
        let physical = if self.len < self.capacity {
            ordered
        } else {
            (self.head + ordered) % self.capacity
        };
        let start = physical * self.bins;
        Some(&self.data[start..start + self.bins])
    }

    /// Rows per column.
    #[inline]
    pub fn bins(&self) -> usize {
        self.bins
    }

    /// Maximum number of columns retained.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Columns currently stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether no column has been pushed yet.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Forget all history (transport stop / editor open).
    pub fn reset(&mut self) {
        self.data.fill(0.0);
        self.head = 0;
        self.len = 0;
    }
}

/// Peak-hold + linear-dB-decay ballistics for a single meter bar.
///
/// The audio thread hands the editor a per-block dB level; this eases the
/// displayed value so a fast peak jumps up instantly but falls back smoothly,
/// the familiar VU/PPM behaviour.  Works for both level meters (input/output)
/// and gain-reduction meters (feed the negated reduction).
#[derive(Debug, Clone)]
pub struct Meter {
    level_db: f32,
    floor_db: f32,
    decay_db: f32,
}

impl Meter {
    /// Build a meter resting at `floor_db`, decaying by `decay_db_per_frame`
    /// (a positive dB step) toward each new lower value.
    pub fn new(floor_db: f32, decay_db_per_frame: f32) -> Self {
        Self {
            level_db: floor_db,
            floor_db,
            decay_db: decay_db_per_frame.max(0.0),
        }
    }

    /// Advance one frame toward `db` and return the displayed level.  Rises to
    /// a louder value instantly; falls no faster than `decay_db` per frame.
    pub fn push(&mut self, db: f32) -> f32 {
        let target = db.max(self.floor_db);
        if target >= self.level_db {
            self.level_db = target;
        } else {
            self.level_db = (self.level_db - self.decay_db).max(target);
        }
        self.level_db
    }

    /// The current displayed level in dB.
    #[inline]
    pub fn level_db(&self) -> f32 {
        self.level_db
    }

    /// Reset to the floor.
    pub fn reset(&mut self) {
        self.level_db = self.floor_db;
    }
}

/// Convert a latency in samples to milliseconds at `sample_rate`.  Returns
/// `0.0` for a non-positive sample rate (badge shows before the host reports).
#[inline]
pub fn latency_ms(samples: usize, sample_rate: f32) -> f32 {
    if sample_rate <= 0.0 {
        0.0
    } else {
        samples as f32 * 1000.0 / sample_rate
    }
}

/// Badge text for a latency in milliseconds, e.g. `"42.7 ms"`.
#[inline]
pub fn format_latency_ms(ms: f32) -> String {
    format!("{ms:.1} ms")
}

/// Meter readout text, e.g. `"-9.0 dB"`, or `"-inf"` at/below `floor_db`.
pub fn format_db(db: f32, floor_db: f32) -> String {
    if db <= floor_db {
        "-inf".to_string()
    } else {
        format!("{db:.1} dB")
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magnitude_db_and_intensity_mapping() {
        assert!((magnitude_to_db(1.0, DISPLAY_FLOOR_DB) - 0.0).abs() < 1e-4);
        assert!((magnitude_to_db(0.1, DISPLAY_FLOOR_DB) + 20.0).abs() < 1e-3);
        // Zero magnitude floors instead of diverging.
        assert_eq!(magnitude_to_db(0.0, DISPLAY_FLOOR_DB), DISPLAY_FLOOR_DB);

        assert_eq!(db_to_intensity(0.0, -90.0, 0.0), 1.0);
        assert_eq!(db_to_intensity(-90.0, -90.0, 0.0), 0.0);
        assert!((db_to_intensity(-45.0, -90.0, 0.0) - 0.5).abs() < 1e-6);
        // Out-of-window clamps; degenerate window is safe.
        assert_eq!(db_to_intensity(10.0, -90.0, 0.0), 1.0);
        assert_eq!(db_to_intensity(0.0, 0.0, 0.0), 0.0);
    }

    #[test]
    fn downsample_log_is_monotone_and_peak_seeking() {
        // 513-bin spectrum (N = 1024) with a single spike near the top.
        let mut mags = vec![0.0f32; 513];
        mags[500] = 1.0;
        let mut out = vec![0.0f32; SPECTROGRAM_BINS];
        downsample_log(&mags, &mut out);
        // The spike lands in the last few (high-frequency) display bins.
        let peak_bin = out
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert!(
            peak_bin > SPECTROGRAM_BINS * 3 / 4,
            "spike at bin {peak_bin}"
        );
        assert_eq!(out[peak_bin], 1.0, "peak magnitude preserved");
    }

    #[test]
    fn downsample_log_low_bin_maps_low() {
        let mut mags = vec![0.0f32; 513];
        mags[2] = 1.0; // just above DC
        let mut out = vec![0.0f32; 64];
        downsample_log(&mags, &mut out);
        let peak_bin = out
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert!(peak_bin < 12, "low spike landed at display bin {peak_bin}");
    }

    #[test]
    fn downsample_log_handles_degenerate_input() {
        let mut out = vec![9.0f32; 16];
        downsample_log(&[], &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
        downsample_log(&[0.5], &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
        // Empty output never panics.
        downsample_log(&[0.1, 0.2, 0.3], &mut []);
    }

    #[test]
    fn spectrogram_orders_columns_oldest_to_newest() {
        let mut spec = Spectrogram::new(2, 3);
        assert!(spec.is_empty());
        spec.push_column(&[1.0, 1.0]);
        spec.push_column(&[2.0, 2.0]);
        assert_eq!(spec.len(), 2);
        assert_eq!(spec.column(0), Some(&[1.0, 1.0][..]));
        assert_eq!(spec.column(1), Some(&[2.0, 2.0][..]));
        assert_eq!(spec.column(2), None);

        // Overflow the ring: oldest column drops off, order preserved.
        spec.push_column(&[3.0, 3.0]);
        spec.push_column(&[4.0, 4.0]);
        assert_eq!(spec.len(), 3);
        assert_eq!(spec.column(0), Some(&[2.0, 2.0][..]));
        assert_eq!(spec.column(1), Some(&[3.0, 3.0][..]));
        assert_eq!(spec.column(2), Some(&[4.0, 4.0][..]));
    }

    #[test]
    fn spectrogram_pads_and_truncates_mismatched_columns() {
        let mut spec = Spectrogram::new(3, 2);
        spec.push_column(&[5.0]); // short → zero-padded
        assert_eq!(spec.column(0), Some(&[5.0, 0.0, 0.0][..]));
        spec.push_column(&[1.0, 2.0, 3.0, 4.0]); // long → truncated
        assert_eq!(spec.column(1), Some(&[1.0, 2.0, 3.0][..]));
    }

    #[test]
    fn spectrogram_reset_clears() {
        let mut spec = Spectrogram::new(4, 4);
        for _ in 0..6 {
            spec.push_column(&[1.0; 4]);
        }
        assert_eq!(spec.len(), 4);
        spec.reset();
        assert!(spec.is_empty());
        assert_eq!(spec.column(0), None);
    }

    #[test]
    fn meter_rises_instantly_and_decays_gradually() {
        let mut meter = Meter::new(-120.0, 3.0);
        // Jump straight up to a peak.
        assert_eq!(meter.push(-6.0), -6.0);
        // Fall back no faster than the decay step.
        assert_eq!(meter.push(-120.0), -9.0);
        assert_eq!(meter.push(-120.0), -12.0);
        // A louder value overrides the decay immediately.
        assert_eq!(meter.push(-4.0), -4.0);
    }

    #[test]
    fn meter_floors_and_resets() {
        let mut meter = Meter::new(-90.0, 6.0);
        meter.push(-10.0);
        // Never drops below the floor even with a very negative input.
        for _ in 0..100 {
            meter.push(-1000.0);
        }
        assert_eq!(meter.level_db(), -90.0);
        meter.push(-20.0);
        meter.reset();
        assert_eq!(meter.level_db(), -90.0);
    }

    #[test]
    fn latency_and_db_formatting() {
        // 2048 samples @ 48 kHz ≈ 42.7 ms.
        let ms = latency_ms(2048, 48_000.0);
        assert!((ms - 42.667).abs() < 1e-2, "ms = {ms}");
        assert_eq!(format_latency_ms(ms), "42.7 ms");
        assert_eq!(latency_ms(2048, 0.0), 0.0);

        assert_eq!(format_db(-9.04, -120.0), "-9.0 dB");
        assert_eq!(format_db(-120.0, -120.0), "-inf");
        assert_eq!(format_db(-130.0, -120.0), "-inf");
    }
}
