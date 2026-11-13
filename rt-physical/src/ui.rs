//! Framework-agnostic display helpers for the Phase-3 editor.
//!
//! The non-trivial piece is the **partial-spectrum visualiser**: given a
//! parameter snapshot it derives the relative amplitudes of the partials
//! (harmonics for a plucked string, inharmonic modes for the modal bank) the
//! current controls would excite, so the editor can draw a little bar chart of
//! the tone's spectral fingerprint.  Keeping the maths here — free of `egui` —
//! lets it be unit-tested without the plugin runtime; the editor only turns the
//! returned amplitudes into bars.
//!
//! # Why a *derived* spectrum rather than a rendered one
//!
//! The FDN reverb's editor previews its tail by actually running an engine
//! (a decay is a time-domain thing).  A struck / plucked instrument's timbre,
//! by contrast, is set almost entirely by two closed-form relationships, so we
//! can draw the spectrum directly and cheaply — no FFT, no throwaway engine:
//!
//! - **Plucked** — a string plucked at fractional position `p` excites the
//!   `n`-th harmonic in proportion to `|sin(n·π·p)|`: harmonics with a node at
//!   the pluck point are silent (pluck at the centre → no even harmonics).  The
//!   loop / damping filter then rolls off the highs, which we model as a
//!   per-harmonic tilt `brightness^(n−1)`.
//! - **Modal** — the partials *are* the preset's inharmonic frequency ratios,
//!   with the same gentle `1/(index+1)` amplitude taper the modal bank applies
//!   in [`rt_physical_core::ModalBank::set_voice`] so the fundamental leads.
//!
//! Both are cheap enough to recompute every time a control moves, so the editor
//! caches the result against the last parameter snapshot exactly as the reverb
//! editor caches its envelope.

use crate::params::InstrumentParams;
use rt_physical_core::Model;

/// Maximum number of partials the visualiser plots.
pub const MAX_PARTIALS: usize = 16;
/// Bottom of the display dB range for a partial bar.
pub const DISPLAY_FLOOR_DB: f32 = -48.0;
/// Top of the display dB range.
pub const DISPLAY_CEIL_DB: f32 = 0.0;

/// One partial of the derived spectrum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Partial {
    /// Frequency ratio relative to the fundamental (`1.0` = fundamental).
    pub ratio: f32,
    /// Linear amplitude, normalised so the loudest partial is `1.0`.
    pub amplitude: f32,
}

impl Partial {
    /// Amplitude in dB, floored at [`DISPLAY_FLOOR_DB`] instead of `−inf`.
    #[inline]
    pub fn amplitude_db(self) -> f32 {
        magnitude_to_db(self.amplitude, DISPLAY_FLOOR_DB)
    }
}

/// Convert a linear magnitude to dB, flooring silence at `floor_db` instead of
/// returning `−inf`.
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

/// Per-harmonic brightness factor for the plucked string: `brightness^(n−1)`
/// where `brightness ∈ (0, 1)` shrinks with `damping` and grows a little with
/// `pick_hardness` (a hard pick excites more highs).  This stands in for the
/// steady-state tilt of the one-pole loop filter without simulating it.
#[inline]
fn plucked_brightness(params: &InstrumentParams) -> f32 {
    // Hard pick claws back up to 40 % of the damping-driven darkening.
    let darkening = params.damping * (1.0 - 0.4 * params.pick_hardness);
    (1.0 - darkening).clamp(0.05, 0.999)
}

/// Derive the partial spectrum the current parameters would excite, normalised
/// so the loudest partial sits at `1.0`.  Returns fundamental-first, at most
/// [`MAX_PARTIALS`] entries.
///
/// **Not real-time** in intent (editor thread only) although it neither
/// allocates beyond the returned `Vec` nor runs any engine.
pub fn partial_spectrum(params: &InstrumentParams) -> Vec<Partial> {
    let mut partials: Vec<Partial> = match params.model {
        Model::Plucked => {
            // Pluck at fractional position `p`: the n-th harmonic is excited in
            // proportion to |sin(n·π·p)| (nodes at the pluck point stay silent),
            // then tilted down by the damping/brightness roll-off.
            let pluck = params.pluck_position.clamp(0.0, 1.0);
            let brightness = plucked_brightness(params);
            (1..=MAX_PARTIALS)
                .map(|harmonic| {
                    let comb = (harmonic as f32 * std::f32::consts::PI * pluck).sin().abs();
                    let tilt = brightness.powi(harmonic as i32 - 1);
                    Partial {
                        ratio: harmonic as f32,
                        amplitude: comb * tilt,
                    }
                })
                .collect()
        }
        Model::Modal => {
            // The partials are the preset's inharmonic ratios; amplitude follows
            // the same 1/(index+1) taper the modal bank applies so the
            // fundamental leads the timbre.
            let ratios = params.preset.ratios();
            let wanted = params.num_modes.min(ratios.len()).min(MAX_PARTIALS);
            ratios
                .iter()
                .take(wanted)
                .enumerate()
                .map(|(index, &ratio)| Partial {
                    ratio,
                    amplitude: 1.0 / (index as f32 + 1.0),
                })
                .collect()
        }
    };

    // Normalise to the loudest partial so the plot always fills the range.
    let peak = partials
        .iter()
        .map(|partial| partial.amplitude)
        .fold(0.0f32, f32::max)
        .max(1e-12);
    for partial in &mut partials {
        partial.amplitude = (partial.amplitude / peak).clamp(0.0, 1.0);
    }
    partials
}

#[cfg(test)]
mod tests {
    use super::*;
    use rt_physical_core::ModePreset;

    #[test]
    fn plucked_centre_nulls_even_harmonics() {
        // Plucking exactly at the midpoint places a node at every even
        // harmonic, so those partials must vanish while odd ones survive.
        let params = InstrumentParams {
            model: Model::Plucked,
            pluck_position: 0.5,
            damping: 0.0,
            pick_hardness: 1.0,
            ..InstrumentParams::default()
        };
        let spectrum = partial_spectrum(&params);
        for partial in &spectrum {
            let harmonic = partial.ratio.round() as i32;
            if harmonic % 2 == 0 {
                assert!(
                    partial.amplitude < 1e-4,
                    "even harmonic {harmonic} should be nulled, got {}",
                    partial.amplitude
                );
            }
        }
        // The fundamental is present and is the normalisation peak.
        assert!((spectrum[0].amplitude - 1.0).abs() < 1e-4);
    }

    #[test]
    fn more_damping_darkens_the_high_harmonics() {
        let base = InstrumentParams {
            model: Model::Plucked,
            pluck_position: 0.28, // avoid comb nulls on the harmonic we probe
            pick_hardness: 0.5,
            ..InstrumentParams::default()
        };
        let bright = partial_spectrum(&InstrumentParams {
            damping: 0.1,
            ..base
        });
        let dark = partial_spectrum(&InstrumentParams {
            damping: 0.9,
            ..base
        });
        let high = MAX_PARTIALS - 1;
        assert!(
            dark[high].amplitude < bright[high].amplitude,
            "more damping must attenuate harmonic {}: dark {} !< bright {}",
            high + 1,
            dark[high].amplitude,
            bright[high].amplitude
        );
    }

    #[test]
    fn modal_spectrum_follows_preset_ratios() {
        let params = InstrumentParams {
            model: Model::Modal,
            preset: ModePreset::Metal,
            num_modes: 4,
            ..InstrumentParams::default()
        };
        let spectrum = partial_spectrum(&params);
        assert_eq!(spectrum.len(), 4);
        // Ratios match the preset's inharmonic series, fundamental first.
        let expected = ModePreset::Metal.ratios();
        for (partial, &ratio) in spectrum.iter().zip(expected) {
            assert!((partial.ratio - ratio).abs() < 1e-4);
        }
        // Amplitude tapers monotonically (fundamental leads).
        for pair in spectrum.windows(2) {
            assert!(pair[0].amplitude >= pair[1].amplitude);
        }
        assert!((spectrum[0].amplitude - 1.0).abs() < 1e-4);
    }

    #[test]
    fn modal_mode_count_caps_partials() {
        // Asking for more modes than the preset lists yields only the listed
        // ones; asking for fewer truncates.
        let full = partial_spectrum(&InstrumentParams {
            model: Model::Modal,
            preset: ModePreset::Wood,
            num_modes: 32,
            ..InstrumentParams::default()
        });
        assert_eq!(full.len(), ModePreset::Wood.ratios().len());

        let two = partial_spectrum(&InstrumentParams {
            model: Model::Modal,
            preset: ModePreset::Wood,
            num_modes: 2,
            ..InstrumentParams::default()
        });
        assert_eq!(two.len(), 2);
    }

    #[test]
    fn every_amplitude_is_finite_and_in_range() {
        for model in [Model::Plucked, Model::Modal] {
            let spectrum = partial_spectrum(&InstrumentParams {
                model,
                ..InstrumentParams::default()
            });
            assert!(!spectrum.is_empty());
            for partial in spectrum {
                assert!(partial.amplitude.is_finite());
                assert!((0.0..=1.0).contains(&partial.amplitude));
                let db = partial.amplitude_db();
                assert!(db.is_finite() && (DISPLAY_FLOOR_DB..=0.001).contains(&db));
            }
        }
    }

    #[test]
    fn db_and_intensity_endpoints() {
        assert_eq!(magnitude_to_db(0.0, DISPLAY_FLOOR_DB), DISPLAY_FLOOR_DB);
        assert!(magnitude_to_db(1.0, DISPLAY_FLOOR_DB).abs() < 1e-4);
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
