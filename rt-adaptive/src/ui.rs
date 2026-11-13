//! Framework-agnostic UI logic for the (Phase-3) `nih-plug-egui` editor.
//!
//! The egui drawing lives in the `gui`-gated `plugin` module; everything here
//! is pure, `egui`-free and therefore **unit-testable**:
//!
//! - [`format_hz`] — spectrum / hum / notch frequency labels.
//! - [`freq_fraction`] — maps a frequency to its `[0, 1]` x-position on the
//!   spectrum display (so notch markers line up with the bars).
//! - [`downsample_spectrum_into`] — collapses the detector's linear magnitude
//!   spectrum into the fixed display-bin grid (allocation-free, audio thread).
//! - [`spectral_heights`] — normalises those magnitudes to `[0, 1]` draw
//!   heights on a dB scale.
//! - [`format_q`] / [`format_age`] — the notch-inventory columns.
//!
//! The feedback-detection LED reuses the shared
//! [`MidiActivity`](rt_ir_plugin_core::MidiActivity) pulse/decay helper,
//! re-exported here for the editor's convenience.

pub use rt_ir_plugin_core::{MidiActivity, MidiStatus};

/// Number of bars in the spectrum display the audio thread publishes.
pub const SPECTRUM_DISPLAY_BINS: usize = 64;

/// Top of the spectrum display's (linear) frequency axis, in Hz — feedback
/// howls of interest live below this.
pub const SPECTRUM_MAX_HZ: f32 = 8_000.0;

/// Format a frequency, e.g. `"2.00 kHz"`, `"440 Hz"`, or `"— Hz"` when zero.
pub fn format_hz(hz: f32) -> String {
    if hz <= 0.0 {
        "— Hz".to_string()
    } else if hz >= 1_000.0 {
        format!("{:.2} kHz", hz / 1_000.0)
    } else {
        format!("{hz:.0} Hz")
    }
}

/// Fraction `[0, 1]` of the spectrum display's frequency axis for `hz` — the
/// x-position of a notch marker (clamped to the display range).
#[inline]
pub fn freq_fraction(hz: f32) -> f32 {
    (hz / SPECTRUM_MAX_HZ).clamp(0.0, 1.0)
}

/// Collapse a linear magnitude spectrum (bin `k` at `k * bin_width_hz`) into the
/// display grid spanning `[0, SPECTRUM_MAX_HZ]`, taking the peak magnitude in
/// each display band.  Writes exactly `out.len()` values and allocates nothing,
/// so it is safe to call from the audio thread.
pub fn downsample_spectrum_into(magnitude: &[f32], bin_width_hz: f32, out: &mut [f32]) {
    for value in out.iter_mut() {
        *value = 0.0;
    }
    if out.is_empty() || bin_width_hz <= 0.0 {
        return;
    }
    let bins = out.len() as f32;
    for (bin, &value) in magnitude.iter().enumerate() {
        let hz = bin as f32 * bin_width_hz;
        if hz >= SPECTRUM_MAX_HZ {
            break;
        }
        let slot = ((hz / SPECTRUM_MAX_HZ) * bins) as usize;
        let slot = slot.min(out.len() - 1);
        out[slot] = out[slot].max(value);
    }
}

/// Normalise a magnitude spectrum to `[0, 1]` draw heights on a dB scale.
///
/// Each value is converted to dB relative to the spectrum's own peak, then
/// mapped so `floor_db` → `0.0` and the peak → `1.0`.  A silent spectrum maps
/// to all zeros.  `floor_db` should be negative (e.g. `-60.0`).  Allocates a
/// fresh `Vec` — editor thread only.
pub fn spectral_heights(magnitude: &[f32], floor_db: f32) -> Vec<f32> {
    let peak = magnitude.iter().copied().fold(0.0f32, f32::max);
    if peak <= 0.0 || floor_db >= 0.0 {
        return vec![0.0; magnitude.len()];
    }
    let inv_floor = 1.0 / floor_db; // floor_db < 0
    magnitude
        .iter()
        .map(|&value| {
            if value <= 0.0 {
                return 0.0;
            }
            let db = 20.0 * (value / peak).log10();
            (1.0 - db * inv_floor).clamp(0.0, 1.0)
        })
        .collect()
}

/// Format a quality factor for the notch inventory, e.g. `"Q 20.0"`.
pub fn format_q(q: f32) -> String {
    format!("Q {q:.1}")
}

/// Format a notch age (in samples at `sample_rate`) as seconds, e.g. `"12.3 s"`.
pub fn format_age(samples: u64, sample_rate: f32) -> String {
    if sample_rate <= 0.0 {
        return "0.0 s".to_string();
    }
    format!("{:.1} s", samples as f32 / sample_rate)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hz_formatting() {
        assert_eq!(format_hz(440.0), "440 Hz");
        assert_eq!(format_hz(2_000.0), "2.00 kHz");
        assert_eq!(format_hz(0.0), "— Hz");
    }

    #[test]
    fn freq_fraction_maps_and_clamps() {
        assert!((freq_fraction(0.0) - 0.0).abs() < 1e-6);
        assert!((freq_fraction(4_000.0) - 0.5).abs() < 1e-6);
        assert_eq!(freq_fraction(20_000.0), 1.0); // clamped
    }

    #[test]
    fn downsample_places_energy_in_the_right_band() {
        // A single strong bin at 2 kHz (bin_width 100 Hz → bin 20).
        let mut magnitude = vec![0.0f32; 129];
        magnitude[20] = 5.0;
        let mut out = vec![0.0f32; SPECTRUM_DISPLAY_BINS];
        downsample_spectrum_into(&magnitude, 100.0, &mut out);
        // 2 kHz is at fraction 0.25 → slot 16 of 64.
        let expected = ((2_000.0 / SPECTRUM_MAX_HZ) * SPECTRUM_DISPLAY_BINS as f32) as usize;
        assert_eq!(out[expected], 5.0);
        assert_eq!(out.iter().filter(|&&v| v > 0.0).count(), 1);
    }

    #[test]
    fn downsample_is_allocation_free_and_safe_on_edges() {
        let mut out = vec![1.0f32; SPECTRUM_DISPLAY_BINS];
        downsample_spectrum_into(&[], 0.0, &mut out);
        assert!(out.iter().all(|&v| v == 0.0), "must clear on empty input");
    }

    #[test]
    fn spectral_heights_peak_is_one_floor_is_zero() {
        let magnitude = [1.0, 0.001, 0.0, 0.5];
        let heights = spectral_heights(&magnitude, -60.0);
        assert!((heights[0] - 1.0).abs() < 1e-4);
        assert!(heights[1] < 0.02);
        assert_eq!(heights[2], 0.0);
        assert!(heights[3] > 0.5 && heights[3] < 1.0);
    }

    #[test]
    fn spectral_heights_of_silence_is_zero() {
        let heights = spectral_heights(&[0.0, 0.0, 0.0], -60.0);
        assert!(heights.iter().all(|&h| h == 0.0));
    }

    #[test]
    fn q_and_age_labels() {
        assert_eq!(format_q(20.0), "Q 20.0");
        assert_eq!(format_age(48_000, 48_000.0), "1.0 s");
        assert_eq!(format_age(72_000, 48_000.0), "1.5 s");
        assert_eq!(format_age(1_000, 0.0), "0.0 s");
    }
}
