//! The eight built-in reverb spaces.
//!
//! Each [`Preset`] is a name plus a ready-to-use [`FdnParams`] for an 8-branch
//! network.  All eight share the Phase 0 coprime delay set as their starting
//! point and differ in decay, damping, low-cut, modulation and size — the
//! knobs that actually separate a plate from a cave.  The plugin (Phase 2) will
//! expose these as the eight `Space` slots (room, hall, plate, spring, chamber,
//! cave, shimmer, non-linear).
//!
//! The delay set is quoted at 48 kHz; [`Fdn::new`](crate::Fdn::new) rescales it
//! to the host rate.

use crate::fdn::FdnParams;

/// The Phase 0 coprime delay set — eight mutually prime lengths in samples
/// @ 48 kHz (≈ 15–42 ms), the shared skeleton for every space.
pub const BASE_DELAYS: [f32; 8] = [743.0, 977.0, 1123.0, 1319.0, 1493.0, 1657.0, 1811.0, 1997.0];

/// A named reverb space: a display name plus its 8-branch FDN configuration.
#[derive(Debug, Clone, Copy)]
pub struct Preset {
    /// Human-facing name (matches the plugin `Space` dropdown).
    pub name: &'static str,
    /// The engine configuration for this space.
    pub params: FdnParams<8>,
}

/// Build one preset from the shared base delays with the distinguishing knobs.
///
/// A wide argument list is deliberate here: this is a compact table builder for
/// the const preset array, and a struct-of-args would only obscure the table.
#[allow(clippy::too_many_arguments)]
const fn space(
    name: &'static str,
    decay_s: f32,
    damping_hz: f32,
    lowcut_hz: f32,
    mod_depth: f32,
    mod_rate_hz: f32,
    diffusion: f32,
    size: f32,
) -> Preset {
    Preset {
        name,
        params: FdnParams {
            nominal_delays: BASE_DELAYS,
            decay_s,
            damping_hz,
            lowcut_hz,
            mod_depth,
            mod_rate_hz,
            diffusion,
            size,
        },
    }
}

/// The eight built-in spaces, in `Space`-dropdown order.
///
/// | # | Space       | Character                                   |
/// |---|-------------|---------------------------------------------|
/// | 0 | Room        | short, bright, tight                        |
/// | 1 | Hall        | long, smooth, gently modulated              |
/// | 2 | Plate       | dense, bright, fast build, little mod        |
/// | 3 | Spring      | short, heavily modulated, low-cut boingy    |
/// | 4 | Chamber     | medium, warm, moderate size                 |
/// | 5 | Cave        | very long, dark (heavy damping)             |
/// | 6 | Shimmer     | long, lush, deep slow modulation            |
/// | 7 | Non-linear  | short-but-diffuse, minimal damping, bright  |
pub const PRESETS: [Preset; 8] = [
    // name        decay damp    lowcut mod  rate diff size
    space("Room", 0.8, 8_000.0, 120.0, 1.5, 0.8, 0.7, 0.45),
    space("Hall", 3.2, 6_000.0, 90.0, 3.0, 0.4, 0.75, 0.85),
    space("Plate", 2.0, 9_000.0, 150.0, 1.0, 0.6, 0.8, 0.6),
    space("Spring", 1.2, 4_500.0, 200.0, 6.0, 2.5, 0.6, 0.4),
    space("Chamber", 1.8, 6_500.0, 110.0, 2.5, 0.5, 0.72, 0.65),
    space("Cave", 6.0, 3_000.0, 70.0, 4.0, 0.3, 0.7, 1.0),
    space("Shimmer", 5.0, 7_500.0, 130.0, 5.0, 0.25, 0.78, 0.9),
    space("Non-linear", 1.0, 10_000.0, 160.0, 2.0, 1.2, 0.65, 0.5),
];

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Exactly eight spaces, matching the plugin's eight `Space` slots.
    #[test]
    fn there_are_eight_presets() {
        assert_eq!(PRESETS.len(), 8);
    }

    /// The base delays are mutually prime (Phase 0 requirement) — every pair is
    /// coprime, so the network has no shared sub-period.
    #[test]
    fn base_delays_are_pairwise_coprime() {
        fn gcd(mut a: u64, mut b: u64) -> u64 {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        }
        for (i, &delay_a) in BASE_DELAYS.iter().enumerate() {
            for &delay_b in &BASE_DELAYS[i + 1..] {
                let a = delay_a as u64;
                let b = delay_b as u64;
                assert_eq!(gcd(a, b), 1, "{a} and {b} are not coprime");
            }
        }
    }

    /// Every preset's parameters are in-range and sane.
    #[test]
    fn presets_have_sane_ranges() {
        for preset in &PRESETS {
            let p = &preset.params;
            assert!(
                (0.1..=20.0).contains(&p.decay_s),
                "{}: decay {} out of range",
                preset.name,
                p.decay_s
            );
            assert!(
                p.damping_hz > 0.0 && p.damping_hz < 24_000.0,
                "{}",
                preset.name
            );
            assert!(p.lowcut_hz > 0.0, "{}", preset.name);
            assert!((0.0..=1.0).contains(&p.size), "{}", preset.name);
            assert!((0.0..=0.9).contains(&p.diffusion), "{}", preset.name);
            assert!(p.mod_depth >= 0.0, "{}", preset.name);
        }
    }

    /// Names are unique (they key the plugin dropdown).
    #[test]
    fn preset_names_are_unique() {
        for (i, first) in PRESETS.iter().enumerate() {
            for second in &PRESETS[i + 1..] {
                assert_ne!(first.name, second.name);
            }
        }
    }
}
