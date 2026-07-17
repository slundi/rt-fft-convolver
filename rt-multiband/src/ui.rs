//! Framework-agnostic editor helpers — metering maths and readout formatters,
//! `egui`-free and unit-tested.
//!
//! The Phase-3 editor (framework still a Phase-3 `[decision]`) will draw the
//! spectrum, per-band gain-reduction meters and transfer curves on top of
//! these; keeping them here lets the numeric behaviour be tested without any
//! GUI toolkit, and lets `plugin.rs` reuse [`peak_dbfs`] for its input/output
//! meters.

use rt_fft_convolver::linear_to_db;

/// dBFS value treated as digital silence for the meters.
pub const SILENCE_DBFS: f32 = -120.0;

/// Bottom of the input/output level meters, in dBFS.
pub const METER_FLOOR_DBFS: f32 = -60.0;

/// Full-scale of the gain-reduction meter, in dB of reduction.
pub const MAX_GR_DB: f32 = 24.0;

/// Peak level of a channel buffer in dBFS (`SILENCE_DBFS` for silence).
pub fn peak_dbfs(channel: &[f32]) -> f32 {
    let peak = channel
        .iter()
        .fold(0.0f32, |acc, &sample| acc.max(sample.abs()));
    if peak <= 0.0 {
        SILENCE_DBFS
    } else {
        linear_to_db(peak).max(SILENCE_DBFS)
    }
}

/// Map a dBFS level onto a `0..=1` meter fill fraction, with `floor_dbfs` at
/// the bottom and `0 dBFS` at the top.
pub fn dbfs_to_fraction(dbfs: f32, floor_dbfs: f32) -> f32 {
    ((dbfs - floor_dbfs) / -floor_dbfs).clamp(0.0, 1.0)
}

/// Map a gain reduction (dB, `≤ 0`) onto a `0..=1` fill fraction, where `0`
/// reduction is empty and `−max_gr_db` is full.
pub fn gr_to_fraction(gr_db: f32, max_gr_db: f32) -> f32 {
    (-gr_db / max_gr_db).clamp(0.0, 1.0)
}

/// Format a dB value with a sign and one decimal (e.g. `"-6.0 dB"`).
pub fn format_db(db: f32) -> String {
    if db <= SILENCE_DBFS {
        "-inf dB".to_string()
    } else {
        format!("{db:+.1} dB")
    }
}

/// Format a gain reduction for the GR readout (always shown as attenuation).
pub fn format_gr(gr_db: f32) -> String {
    format!("{:.1} dB", gr_db.min(0.0))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_of_silence_is_the_floor() {
        assert_eq!(peak_dbfs(&[0.0; 64]), SILENCE_DBFS);
        assert_eq!(peak_dbfs(&[]), SILENCE_DBFS);
    }

    #[test]
    fn peak_of_full_scale_is_zero_dbfs() {
        let buf = [0.5, -1.0, 0.25];
        assert!((peak_dbfs(&buf) - 0.0).abs() < 1e-4);
    }

    #[test]
    fn dbfs_fraction_spans_floor_to_top() {
        assert_eq!(dbfs_to_fraction(0.0, METER_FLOOR_DBFS), 1.0);
        assert_eq!(dbfs_to_fraction(METER_FLOOR_DBFS, METER_FLOOR_DBFS), 0.0);
        assert!((dbfs_to_fraction(-30.0, METER_FLOOR_DBFS) - 0.5).abs() < 1e-4);
        // Below the floor clamps to empty.
        assert_eq!(dbfs_to_fraction(-90.0, METER_FLOOR_DBFS), 0.0);
    }

    #[test]
    fn gr_fraction_grows_with_reduction() {
        assert_eq!(gr_to_fraction(0.0, MAX_GR_DB), 0.0);
        assert_eq!(gr_to_fraction(-MAX_GR_DB, MAX_GR_DB), 1.0);
        assert!((gr_to_fraction(-12.0, MAX_GR_DB) - 0.5).abs() < 1e-4);
        // Beyond full-scale clamps.
        assert_eq!(gr_to_fraction(-48.0, MAX_GR_DB), 1.0);
    }

    #[test]
    fn format_helpers_render_sign_and_silence() {
        assert_eq!(format_db(-6.0), "-6.0 dB");
        assert_eq!(format_db(3.0), "+3.0 dB");
        assert_eq!(format_db(SILENCE_DBFS), "-inf dB");
        assert_eq!(format_gr(-4.25), "-4.2 dB");
        // A positive GR is clamped to 0 (there is no upward "reduction").
        assert_eq!(format_gr(2.0), "0.0 dB");
    }
}
