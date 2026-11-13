//! Framework-agnostic editor helpers — metering maths and readout formatters,
//! `egui`-free and unit-tested.
//!
//! The Phase-3 (`nih-plug-egui`) editor draws the spectrum, per-band
//! gain-reduction meters and transfer curves on top of these; keeping the
//! numeric behaviour here lets it be tested without any GUI toolkit, and lets
//! `plugin.rs` reuse [`peak_dbfs`] for its input/output meters.

use rt_fft_convolver::linear_to_db;
use rt_multiband_core::GainComputer;

use crate::params::{BandParams, CROSSOVER_HZ_RANGE, THRESHOLD_DB_RANGE};

/// dBFS value treated as digital silence for the meters.
pub const SILENCE_DBFS: f32 = -120.0;

/// Bottom of the input/output level meters, in dBFS.
pub const METER_FLOOR_DBFS: f32 = -60.0;

/// Full-scale of the gain-reduction meter, in dB of reduction.
pub const MAX_GR_DB: f32 = 24.0;

/// Bottom of the spectrum display, in dBFS (peak-relative).
pub const SPECTRUM_FLOOR_DB: f32 = -72.0;

/// The lowest frequency shown on the editor's log axis, in Hz.
pub const DISPLAY_MIN_HZ: f32 = CROSSOVER_HZ_RANGE.0;

/// The highest frequency shown on the editor's log axis, in Hz.
pub const DISPLAY_MAX_HZ: f32 = CROSSOVER_HZ_RANGE.1;

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

/// Format a frequency for a readout (`"120 Hz"` / `"6.0 kHz"`).
pub fn format_hz(hz: f32) -> String {
    if hz >= 1_000.0 {
        format!("{:.1} kHz", hz / 1_000.0)
    } else {
        format!("{hz:.0} Hz")
    }
}

/// Map a frequency onto a `0..=1` position on a **log** axis spanning
/// `min_hz..=max_hz` (used for the spectrum x-axis and crossover lines).
pub fn hz_to_fraction(hz: f32, min_hz: f32, max_hz: f32) -> f32 {
    let hz = hz.clamp(min_hz, max_hz);
    (hz.ln() - min_hz.ln()) / (max_hz.ln() - min_hz.ln())
}

/// Inverse of [`hz_to_fraction`] — a `0..=1` axis position back to a frequency
/// (used when a dragged crossover line sets its parameter).
pub fn fraction_to_hz(fraction: f32, min_hz: f32, max_hz: f32) -> f32 {
    let fraction = fraction.clamp(0.0, 1.0);
    (min_hz.ln() + (max_hz.ln() - min_hz.ln()) * fraction).exp()
}

/// Normalise a magnitude spectrum to `0..=1` bar heights on a dB scale, with
/// the loudest bin at the top and `floor_db` below it at the bottom.
///
/// Allocates its result — an editor-frame call, never on the audio thread.
pub fn spectral_heights(magnitude: &[f32], floor_db: f32) -> Vec<f32> {
    let peak = magnitude.iter().fold(0.0f32, |acc, &m| acc.max(m));
    if peak <= 0.0 {
        return vec![0.0; magnitude.len()];
    }
    magnitude
        .iter()
        .map(|&m| {
            if m <= 0.0 {
                0.0
            } else {
                let db = 20.0 * (m / peak).log10();
                ((db - floor_db) / -floor_db).clamp(0.0, 1.0)
            }
        })
        .collect()
}

/// The output level (dBFS) a band's static curve maps `input_db` to — the
/// per-band transfer-curve plot (threshold / ratio / knee / mode).
pub fn transfer_output_db(band: &BandParams, input_db: f32) -> f32 {
    let computer = GainComputer::new(
        band.mode.to_mode(),
        band.threshold_db,
        band.ratio,
        band.knee_db,
    );
    input_db + computer.gain_db(input_db)
}

/// Map a threshold dB value onto a `0..=1` fraction for the transfer-curve
/// plot's axes (both axes share the threshold range).
pub fn threshold_to_fraction(db: f32) -> f32 {
    ((db - THRESHOLD_DB_RANGE.0) / (THRESHOLD_DB_RANGE.1 - THRESHOLD_DB_RANGE.0)).clamp(0.0, 1.0)
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

    #[test]
    fn format_hz_switches_to_khz() {
        assert_eq!(format_hz(120.0), "120 Hz");
        assert_eq!(format_hz(6_000.0), "6.0 kHz");
    }

    #[test]
    fn log_frequency_axis_round_trips() {
        let (min, max) = (DISPLAY_MIN_HZ, DISPLAY_MAX_HZ);
        assert!((hz_to_fraction(min, min, max) - 0.0).abs() < 1e-6);
        assert!((hz_to_fraction(max, min, max) - 1.0).abs() < 1e-6);
        for &hz in &[50.0, 200.0, 1_000.0, 8_000.0] {
            let round = fraction_to_hz(hz_to_fraction(hz, min, max), min, max);
            assert!((round - hz).abs() < 1.0, "round-trip {hz} → {round}");
        }
        // The geometric mid-frequency sits at the axis centre.
        let mid = fraction_to_hz(0.5, min, max);
        assert!((mid - (min * max).sqrt()).abs() < 1.0);
    }

    #[test]
    fn spectral_heights_peak_is_one_floor_is_zero() {
        // A magnitude 40 dB below peak sits just above the −72 dB floor.
        let heights = spectral_heights(&[1.0, 0.01, 0.0], SPECTRUM_FLOOR_DB);
        assert!((heights[0] - 1.0).abs() < 1e-6, "peak should top out");
        assert!(heights[1] > 0.0 && heights[1] < 1.0);
        assert_eq!(heights[2], 0.0, "silence bin is empty");
    }

    #[test]
    fn spectral_heights_of_silence_is_zero() {
        assert_eq!(
            spectral_heights(&[0.0, 0.0, 0.0], SPECTRUM_FLOOR_DB),
            vec![0.0; 3]
        );
    }

    #[test]
    fn transfer_curve_matches_the_compressor_acceptance() {
        // 20 dB above a −20 dB threshold, 2:1 hard knee → output at −10 dB.
        let band = BandParams {
            mode: crate::params::BandModeChoice::Compressor,
            threshold_db: -20.0,
            ratio: 2.0,
            knee_db: 0.0,
            ..Default::default()
        };
        assert!((transfer_output_db(&band, 0.0) - -10.0).abs() < 1e-3);
        // Below threshold the curve is 1:1 (identity).
        assert!((transfer_output_db(&band, -40.0) - -40.0).abs() < 1e-3);
    }

    #[test]
    fn transfer_curve_bypass_is_identity() {
        let band = BandParams {
            mode: crate::params::BandModeChoice::Bypass,
            ..Default::default()
        };
        for &input in &[-50.0, -10.0, 0.0] {
            assert!((transfer_output_db(&band, input) - input).abs() < 1e-6);
        }
    }

    #[test]
    fn threshold_fraction_spans_the_range() {
        assert_eq!(threshold_to_fraction(THRESHOLD_DB_RANGE.0), 0.0);
        assert_eq!(threshold_to_fraction(THRESHOLD_DB_RANGE.1), 1.0);
    }
}
