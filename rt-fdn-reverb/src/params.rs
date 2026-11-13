//! Plain, framework-agnostic parameter model for the FDN reverb plugin.
//!
//! Mirrors the Phase-2 control list in `docs/Roadmap-05-fdn-reverb.md`.  The
//! `nih-plug` wrapper (behind the `nih` feature) maps its automatable
//! parameters onto these fields; keeping the model framework-free lets us
//! unit-test ranges, clamping, and the percent → engine-unit mappings without
//! the plugin runtime.
//!
//! # Presets vs. global knobs
//!
//! There are two levels, exactly like the IR plugins' slot rack:
//!
//! - **Eight preset slots.**  Each slot selects one of the eight built-in
//!   [`rt_fdn_core::PRESETS`] *spaces* (`space[slot]`) and can be enabled or
//!   disabled (`enabled[slot]`).  [`current_preset`](ReverbParams::current_preset)
//!   picks the active slot; switching it equal-power **crossfades** the FDN so
//!   the outgoing tail is never cut off.
//! - **Global knobs.**  `Decay`, `Size`, `Damping`, `Diffusion`, `Modulation
//!   Depth/Rate` drive the *active* space live (they override the space's baked
//!   values), while `Pre-Delay`, `Low Cut`, `High Cut`, `Stereo Width`, `Dry`,
//!   `Wet`, `Freeze` and `Bypass` act on the wet bus around the FDN.
//!
//! The percent-valued knobs (`Size`, `Damping`, `Diffusion`, `Modulation
//! Depth`, `Stereo Width`) are `0..=1` here and converted to engine units by
//! the helpers below.

use rt_fdn_core::PRESETS;

/// Number of preset slots (roadmap: 8, matching the built-in spaces).
pub const NUM_PRESETS: usize = 8;
/// Number of built-in spaces a slot can select (`rt_fdn_core::PRESETS`).
pub const NUM_SPACES: usize = PRESETS.len();

/// Crossfade time range in ms (reverb tails need long fades, roadmap 5–500 ms).
pub const CROSSFADE_MS_RANGE: (f32, f32) = (5.0, 500.0);
/// Pre-delay range in ms.
pub const PREDELAY_MS_RANGE: (f32, f32) = (0.0, 500.0);
/// Decay (T60) range in seconds — matches the engine's own decay clamp.
pub const DECAY_S_RANGE: (f32, f32) = (0.1, 20.0);
/// Modulation-rate range in Hz.
pub const MOD_RATE_HZ_RANGE: (f32, f32) = (0.1, 5.0);
/// Low-cut (high-pass) cutoff range in Hz (wet only).
pub const LOW_CUT_RANGE: (f32, f32) = (20.0, 500.0);
/// High-cut (low-pass) cutoff range in Hz (wet only).
pub const HIGH_CUT_RANGE: (f32, f32) = (2_000.0, 20_000.0);
/// Stereo-width range (`0.0` = mono, `1.0` = 100 %, `2.0` = 200 %).
pub const WIDTH_RANGE: (f32, f32) = (0.0, 2.0);
/// Dry / Wet fader range in dB (the low end behaves as `−∞`; see `fader_gain`).
pub const FADER_DB_RANGE: (f32, f32) = (-60.0, 6.0);

/// Highest per-branch damping cutoff (0 % damping — effectively wide open).
pub const DAMPING_MAX_HZ: f32 = 18_000.0;
/// Lowest per-branch damping cutoff (100 % damping — very dark tail).
pub const DAMPING_MIN_HZ: f32 = 1_200.0;
/// Peak input-diffusion all-pass gain at 100 % (the engine caps at ~0.9).
pub const DIFFUSION_MAX_GAIN: f32 = 0.9;
/// Peak delay-modulation depth in samples @ `rt_fdn_core::REFERENCE_SR` at
/// 100 %.  Every voice is built with this headroom so the global `Modulation
/// Depth` knob has its full range on any space.
pub const MOD_DEPTH_MAX_SAMPLES: f32 = 8.0;

/// Full parameter state of one reverb instance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReverbParams {
    /// Active preset slot, `0..NUM_PRESETS`.
    pub current_preset: usize,
    /// Per-slot built-in space index (`0..NUM_SPACES`).
    pub space: [usize; NUM_PRESETS],
    /// Per-slot enable flags; the active slot being disabled plays dry.
    pub enabled: [bool; NUM_PRESETS],

    /// Equal-power crossfade duration in ms when switching preset slots.
    pub crossfade_ms: f32,
    /// Pre-delay before the wet onset, in ms (dry stays zero-latency).
    pub predelay_ms: f32,

    /// Reverb decay time (T60) in seconds.
    pub decay_s: f32,
    /// Size scale in `0..=1` (fraction of the space's delay lengths).
    pub size: f32,
    /// Damping amount in `0..=1` (0 = open, 1 = dark).
    pub damping: f32,
    /// Input-diffusion amount in `0..=1`.
    pub diffusion: f32,
    /// Modulation depth in `0..=1`.
    pub mod_depth: f32,
    /// Modulation LFO rate in Hz.
    pub mod_rate_hz: f32,

    /// Wet-only low-cut (high-pass) cutoff in Hz.
    pub low_cut_hz: f32,
    /// Wet-only high-cut (low-pass) cutoff in Hz.
    pub high_cut_hz: f32,
    /// Stereo width applied to the wet signal (`0..=2`).
    pub width: f32,
    /// Dry fader in dB (`−∞ … +6`); `−∞` mutes the dry path.
    pub dry_db: f32,
    /// Wet fader in dB (`−∞ … +6`).
    pub wet_db: f32,

    /// Freeze: unity feedback, no new input into the FDN (infinite sustain).
    pub freeze: bool,
    /// Host bypass — output equals the untouched input.
    pub bypass: bool,
}

impl Default for ReverbParams {
    fn default() -> Self {
        Self {
            current_preset: 0,
            // Slots default to spaces 0..7 in order (Room, Hall, Plate, …).
            space: core::array::from_fn(|i| i.min(NUM_SPACES - 1)),
            enabled: [true; NUM_PRESETS],
            crossfade_ms: 100.0,
            predelay_ms: 0.0,
            decay_s: 2.5,
            size: 0.7,
            damping: 0.35,
            diffusion: 0.75,
            mod_depth: 0.35,
            mod_rate_hz: 0.6,
            low_cut_hz: LOW_CUT_RANGE.0,
            high_cut_hz: HIGH_CUT_RANGE.1,
            width: 1.0,
            dry_db: 0.0,
            wet_db: -3.0,
            freeze: false,
            bypass: false,
        }
    }
}

impl ReverbParams {
    /// Clamp every field to its documented range.
    pub fn clamped(mut self) -> Self {
        self.current_preset = self.current_preset.min(NUM_PRESETS - 1);
        for space in &mut self.space {
            *space = (*space).min(NUM_SPACES - 1);
        }
        self.crossfade_ms = self
            .crossfade_ms
            .clamp(CROSSFADE_MS_RANGE.0, CROSSFADE_MS_RANGE.1);
        self.predelay_ms = self
            .predelay_ms
            .clamp(PREDELAY_MS_RANGE.0, PREDELAY_MS_RANGE.1);
        self.decay_s = self.decay_s.clamp(DECAY_S_RANGE.0, DECAY_S_RANGE.1);
        self.size = self.size.clamp(0.0, 1.0);
        self.damping = self.damping.clamp(0.0, 1.0);
        self.diffusion = self.diffusion.clamp(0.0, 1.0);
        self.mod_depth = self.mod_depth.clamp(0.0, 1.0);
        self.mod_rate_hz = self
            .mod_rate_hz
            .clamp(MOD_RATE_HZ_RANGE.0, MOD_RATE_HZ_RANGE.1);
        self.low_cut_hz = self.low_cut_hz.clamp(LOW_CUT_RANGE.0, LOW_CUT_RANGE.1);
        self.high_cut_hz = self.high_cut_hz.clamp(HIGH_CUT_RANGE.0, HIGH_CUT_RANGE.1);
        self.width = self.width.clamp(WIDTH_RANGE.0, WIDTH_RANGE.1);
        self.dry_db = self.dry_db.clamp(FADER_DB_RANGE.0, FADER_DB_RANGE.1);
        self.wet_db = self.wet_db.clamp(FADER_DB_RANGE.0, FADER_DB_RANGE.1);
        self
    }

    /// The built-in space index the active slot points at.
    #[inline]
    pub fn active_space(&self) -> usize {
        self.space[self.current_preset.min(NUM_PRESETS - 1)].min(NUM_SPACES - 1)
    }

    /// Whether the active slot is enabled (a disabled slot renders dry).
    #[inline]
    pub fn active_enabled(&self) -> bool {
        self.enabled[self.current_preset.min(NUM_PRESETS - 1)]
    }
}

/// Map a `Damping` amount (`0..=1`) to a per-branch high-cut cutoff in Hz.
///
/// 0 % is wide open ([`DAMPING_MAX_HZ`]); 100 % is very dark
/// ([`DAMPING_MIN_HZ`]).  Interpolated on a **log** frequency axis so the knob
/// feels perceptually even.
#[inline]
pub fn damping_cutoff_hz(amount: f32) -> f32 {
    let amount = amount.clamp(0.0, 1.0);
    let hi = DAMPING_MAX_HZ.ln();
    let lo = DAMPING_MIN_HZ.ln();
    (hi + (lo - hi) * amount).exp()
}

/// Map a `Diffusion` amount (`0..=1`) to the input-diffuser all-pass gain.
#[inline]
pub fn diffusion_gain(amount: f32) -> f32 {
    amount.clamp(0.0, 1.0) * DIFFUSION_MAX_GAIN
}

/// Map a `Modulation Depth` amount (`0..=1`) to a depth in samples
/// @ [`rt_fdn_core::REFERENCE_SR`].
#[inline]
pub fn mod_depth_samples(amount: f32) -> f32 {
    amount.clamp(0.0, 1.0) * MOD_DEPTH_MAX_SAMPLES
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_in_range_and_neutral() {
        let params = ReverbParams::default();
        assert_eq!(params, params.clamped(), "defaults must survive a clamp");
        assert_eq!(params.current_preset, 0);
        assert_eq!(params.width, 1.0);
        assert_eq!(params.dry_db, 0.0);
        assert!(!params.freeze);
        assert!(!params.bypass);
        assert!(params.enabled.iter().all(|&e| e));
        // Slots map to distinct spaces by default.
        assert_eq!(params.space, [0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn clamp_pulls_fields_into_range() {
        let wild = ReverbParams {
            current_preset: 99,
            space: [42; NUM_PRESETS],
            crossfade_ms: 9_999.0,
            predelay_ms: -5.0,
            decay_s: 100.0,
            size: 5.0,
            damping: -1.0,
            diffusion: 9.0,
            mod_depth: 2.0,
            mod_rate_hz: 50.0,
            low_cut_hz: 1.0,
            high_cut_hz: 99_999.0,
            width: 9.0,
            dry_db: 100.0,
            wet_db: -999.0,
            ..ReverbParams::default()
        }
        .clamped();
        assert_eq!(wild.current_preset, NUM_PRESETS - 1);
        assert!(wild.space.iter().all(|&s| s == NUM_SPACES - 1));
        assert_eq!(wild.crossfade_ms, CROSSFADE_MS_RANGE.1);
        assert_eq!(wild.predelay_ms, PREDELAY_MS_RANGE.0);
        assert_eq!(wild.decay_s, DECAY_S_RANGE.1);
        assert_eq!(wild.size, 1.0);
        assert_eq!(wild.damping, 0.0);
        assert_eq!(wild.diffusion, 1.0);
        assert_eq!(wild.mod_depth, 1.0);
        assert_eq!(wild.mod_rate_hz, MOD_RATE_HZ_RANGE.1);
        assert_eq!(wild.low_cut_hz, LOW_CUT_RANGE.0);
        assert_eq!(wild.high_cut_hz, HIGH_CUT_RANGE.1);
        assert_eq!(wild.width, WIDTH_RANGE.1);
        assert_eq!(wild.dry_db, FADER_DB_RANGE.1);
        assert_eq!(wild.wet_db, FADER_DB_RANGE.0);
    }

    #[test]
    fn damping_cutoff_is_monotonic_decreasing() {
        // 0 % → wide open; 100 % → dark; strictly decreasing in between.
        assert!((damping_cutoff_hz(0.0) - DAMPING_MAX_HZ).abs() < 1.0);
        assert!((damping_cutoff_hz(1.0) - DAMPING_MIN_HZ).abs() < 1.0);
        let mut prev = damping_cutoff_hz(0.0);
        for step in 1..=10 {
            let cut = damping_cutoff_hz(step as f32 / 10.0);
            assert!(cut < prev, "damping cutoff not decreasing at {step}");
            prev = cut;
        }
    }

    #[test]
    fn diffusion_and_mod_depth_scale_linearly() {
        assert_eq!(diffusion_gain(0.0), 0.0);
        assert!((diffusion_gain(1.0) - DIFFUSION_MAX_GAIN).abs() < 1e-6);
        assert!((diffusion_gain(0.5) - DIFFUSION_MAX_GAIN * 0.5).abs() < 1e-6);
        assert_eq!(mod_depth_samples(0.0), 0.0);
        assert!((mod_depth_samples(1.0) - MOD_DEPTH_MAX_SAMPLES).abs() < 1e-6);
    }

    #[test]
    fn active_space_and_enabled_follow_current_preset() {
        let mut params = ReverbParams {
            current_preset: 3,
            ..ReverbParams::default()
        };
        params.space[3] = 5;
        params.enabled[3] = false;
        assert_eq!(params.active_space(), 5);
        assert!(!params.active_enabled());
    }
}
