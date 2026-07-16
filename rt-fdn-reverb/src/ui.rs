//! Framework-agnostic display helpers for the Phase-3 editor.
//!
//! The only non-trivial piece is the **impulse-response visualiser**: given a
//! parameter snapshot it runs a virtual impulse through a throwaway
//! [`ReverbEngine`] and reduces the output to a short decay envelope (one dB
//! value per display bucket).  Keeping it here — free of `egui` — lets it be
//! unit-tested without the plugin runtime; the editor only turns the returned
//! envelope into a polyline.
//!
//! # Not real-time
//!
//! [`decay_envelope_db`] builds an engine and processes thousands of samples,
//! so it allocates and is **editor-thread only**, recomputed when the
//! parameters change.  It must run at the *host* sample rate: the in-loop
//! damping / low-cut filters lose energy per feedback *pass*, and passes-per-
//! second scale with the sample rate, so the effective decay is rate-dependent
//! — a reduced preview rate would show a much shorter tail than the plugin
//! actually produces.
//!
//! `Freeze` is deliberately **ignored** by the preview: freezing mutes the FDN
//! input, so a from-scratch impulse would show silence.  The visualiser depicts
//! the reverb's decay *shape* (set by decay / size / damping); the freeze state
//! is shown by its own LED instead.

use crate::ReverbEngine;
use crate::params::ReverbParams;

/// Number of display buckets in the decay-envelope plot.
pub const ENVELOPE_BUCKETS: usize = 128;
/// Bottom of the display dB range.
pub const DISPLAY_FLOOR_DB: f32 = -60.0;
/// Top of the display dB range.
pub const DISPLAY_CEIL_DB: f32 = 0.0;

/// Preview window length in seconds for a given decay time — long enough to
/// show the tail, clamped so a very long decay does not stall the recompute.
#[inline]
pub fn envelope_seconds(decay_s: f32) -> f32 {
    (decay_s * 1.2).clamp(0.3, 4.0)
}

/// Convert a linear magnitude to dB, flooring silence at `floor_db` instead of
/// returning `-inf`.
#[inline]
pub fn magnitude_to_db(magnitude: f32, floor_db: f32) -> f32 {
    if magnitude <= 0.0 {
        floor_db
    } else {
        (20.0 * magnitude.log10()).max(floor_db)
    }
}

/// Map a dB value to a `0.0..=1.0` display intensity between `floor_db` and
/// `ceil_db`.
#[inline]
pub fn db_to_intensity(db: f32, floor_db: f32, ceil_db: f32) -> f32 {
    ((db - floor_db) / (ceil_db - floor_db)).clamp(0.0, 1.0)
}

/// Render a virtual impulse through `params` and return a per-bucket decay
/// envelope in dB, normalised so the peak sits at `0 dB` and floored at
/// [`DISPLAY_FLOOR_DB`].
///
/// **Allocates and processes many samples** — editor thread only.
pub fn decay_envelope_db(
    params: &ReverbParams,
    sample_rate: u32,
    seconds: f32,
    buckets: usize,
) -> Vec<f32> {
    let buckets = buckets.max(1);
    let sample_rate = sample_rate.max(1);
    let total = ((seconds.max(0.01) * sample_rate as f32) as usize).max(buckets);

    // Preview the decay *shape*: freeze mutes the FDN input, which would show
    // silence, so it is ignored here (the editor shows freeze via its LED).
    let mut preview = *params;
    preview.freeze = false;
    let mut engine = ReverbEngine::new(sample_rate, &preview);
    let mut peak_per_bucket = vec![0.0f32; buckets];
    for i in 0..total {
        let (left, right) = if i == 0 {
            engine.process_sample(1.0, 1.0)
        } else {
            engine.process_sample(0.0, 0.0)
        };
        let magnitude = left.abs().max(right.abs());
        let bucket = (i * buckets / total).min(buckets - 1);
        if magnitude > peak_per_bucket[bucket] {
            peak_per_bucket[bucket] = magnitude;
        }
    }

    // Normalise to the loudest bucket so the plot always fills the range.
    let peak = peak_per_bucket
        .iter()
        .copied()
        .fold(0.0f32, f32::max)
        .max(1e-12);
    peak_per_bucket
        .iter()
        .map(|&magnitude| magnitude_to_db(magnitude / peak, DISPLAY_FLOOR_DB))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 48_000;

    fn wet_only(decay_s: f32) -> ReverbParams {
        ReverbParams {
            decay_s,
            dry_db: -60.0,
            wet_db: 0.0,
            low_cut_hz: 20.0,
            high_cut_hz: 20_000.0,
            crossfade_ms: 5.0,
            ..ReverbParams::default()
        }
    }

    #[test]
    fn envelope_has_expected_shape() {
        let env = decay_envelope_db(&wet_only(2.0), SR, 2.4, ENVELOPE_BUCKETS);
        assert_eq!(env.len(), ENVELOPE_BUCKETS);
        // Every value is finite and inside the display range.
        assert!(
            env.iter()
                .all(|&db| db.is_finite() && (DISPLAY_FLOOR_DB..=0.001).contains(&db))
        );
        // The peak (0 dB) exists somewhere, and the tail ends well below it.
        assert!(env.iter().cloned().fold(f32::MIN, f32::max) > -0.5);
        assert!(env[ENVELOPE_BUCKETS - 1] < env[0] || env[ENVELOPE_BUCKETS - 1] < -6.0);
    }

    #[test]
    fn longer_decay_has_a_higher_tail() {
        let seconds = 3.0;
        let short = decay_envelope_db(&wet_only(0.5), SR, seconds, ENVELOPE_BUCKETS);
        let long = decay_envelope_db(&wet_only(8.0), SR, seconds, ENVELOPE_BUCKETS);
        // At the same wall-clock window, the long decay is well above the short
        // one in the middle of the tail.
        let mid = ENVELOPE_BUCKETS / 2;
        assert!(
            long[mid] > short[mid] + 6.0,
            "long tail {} not clearly above short {} at bucket {mid}",
            long[mid],
            short[mid]
        );
    }

    #[test]
    fn preview_ignores_freeze() {
        // Freeze mutes the FDN input, so the preview must show the un-frozen
        // decay shape — identical whether or not freeze is engaged.
        let base = wet_only(2.0);
        let mut frozen = base;
        frozen.freeze = true;
        let env_off = decay_envelope_db(&base, SR, 3.0, ENVELOPE_BUCKETS);
        let env_on = decay_envelope_db(&frozen, SR, 3.0, ENVELOPE_BUCKETS);
        assert_eq!(env_off, env_on);
        // And it is a real decay, not silence.
        assert!(env_off[ENVELOPE_BUCKETS - 1] < env_off[0]);
    }

    #[test]
    fn db_and_intensity_round_trip_endpoints() {
        assert_eq!(magnitude_to_db(0.0, DISPLAY_FLOOR_DB), DISPLAY_FLOOR_DB);
        assert!((magnitude_to_db(1.0, DISPLAY_FLOOR_DB)).abs() < 1e-4);
        assert_eq!(
            db_to_intensity(DISPLAY_CEIL_DB, DISPLAY_FLOOR_DB, DISPLAY_CEIL_DB),
            1.0
        );
        assert_eq!(
            db_to_intensity(DISPLAY_FLOOR_DB, DISPLAY_FLOOR_DB, DISPLAY_CEIL_DB),
            0.0
        );
    }
}
