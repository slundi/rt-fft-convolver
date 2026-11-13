//! Framework-agnostic parameter model for the `rt-multiband` plugin.
//!
//! [`MultibandParams`] mirrors the Phase-2 control list from
//! `docs/Roadmap-11-multiband-compressor.md` as a plain `Copy` struct with
//! documented ranges.  Keeping it free of `nih-plug` (which lives behind the
//! `nih` feature) lets the engine and the control mappings be unit-tested
//! without the plugin framework.
//!
//! # How the controls map onto the engine
//!
//! Global `Input`/`Output`/`Mix` drive the [`MultibandEngine`]'s smoothed
//! gains; `Crossover N` retunes the shared [`Crossover`]; `Link` sets the
//! stereo detector linking on every band.  Each of the four [`BandParams`]
//! drives one [`BandProcessor`]: `Mode`/`Threshold`/`Ratio`/`Knee` configure
//! the gain computer, `Attack`/`Release` the envelope, `Blend` the detector,
//! `Makeup` the output trim, `SC HPF` the internal sidechain filter, and
//! `Solo`/`Mute` the band-sum routing.
//!
//! # v1 scope (Phase 0)
//!
//! Fixed 4 bands, LR4 IIR crossover, internal sidechain only, zero look-ahead.
//! Nothing here reallocates the engine, so [`MultibandParams::structural_changed`]
//! is always `false` in v1 (kept for family consistency and forward-compat with
//! the v2 band-count / linear-phase changes).
//!
//! [`MultibandEngine`]: rt_multiband_core::MultibandEngine
//! [`Crossover`]: rt_multiband_core::Crossover
//! [`BandProcessor`]: rt_multiband_core::BandProcessor

use rt_multiband_core::{BandMode, NUM_BANDS, NUM_CROSSOVERS, StereoLink};

/// Inclusive `Input Gain` range in dB.
pub const INPUT_GAIN_DB_RANGE: (f32, f32) = (-24.0, 24.0);
/// Inclusive `Output Gain` range in dB.
pub const OUTPUT_GAIN_DB_RANGE: (f32, f32) = (-24.0, 24.0);
/// Inclusive per-band `Threshold` range in dBFS.
pub const THRESHOLD_DB_RANGE: (f32, f32) = (-60.0, 0.0);
/// Inclusive per-band `Ratio` range (`:1`).
pub const RATIO_RANGE: (f32, f32) = (1.0, 20.0);
/// Inclusive per-band `Attack` range in milliseconds.
pub const ATTACK_MS_RANGE: (f32, f32) = (0.1, 100.0);
/// Inclusive per-band `Release` range in milliseconds.
pub const RELEASE_MS_RANGE: (f32, f32) = (5.0, 1000.0);
/// Inclusive per-band `Knee` width range in dB (`0` ⇒ hard knee).
pub const KNEE_DB_RANGE: (f32, f32) = (0.0, 24.0);
/// Inclusive per-band `Makeup Gain` range in dB.
pub const MAKEUP_DB_RANGE: (f32, f32) = (-12.0, 24.0);
/// Inclusive per-band detector `Blend` range (`0` = RMS, `1` = peak).
pub const BLEND_RANGE: (f32, f32) = (0.0, 1.0);
/// Inclusive per-band sidechain `HPF` range in Hz.
pub const SC_HPF_HZ_RANGE: (f32, f32) = (20.0, 2_000.0);
/// Inclusive `Crossover` frequency range in Hz.
pub const CROSSOVER_HZ_RANGE: (f32, f32) = (20.0, 20_000.0);

/// Default crossover edges (low|low-mid, low-mid|high-mid, high-mid|high).
pub const DEFAULT_CROSSOVER_HZ: [f32; NUM_CROSSOVERS] = [120.0, 1_000.0, 6_000.0];

/// Per-band dynamics mode (mirrors [`BandMode`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandModeChoice {
    /// Downward compression above threshold.
    Compressor,
    /// Downward expansion below threshold.
    Expander,
    /// Expander with a high ratio + deep range.
    Gate,
    /// Unity gain (band still contributes to the sum).
    Bypass,
}

impl BandModeChoice {
    /// Map to the engine's [`BandMode`].
    pub fn to_mode(self) -> BandMode {
        match self {
            BandModeChoice::Compressor => BandMode::Compressor,
            BandModeChoice::Expander => BandMode::Expander,
            BandModeChoice::Gate => BandMode::Gate,
            BandModeChoice::Bypass => BandMode::Bypass,
        }
    }
}

/// Stereo detector linking (mirrors [`StereoLink`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StereoLinkChoice {
    /// One shared gain driven by `max(|L|, |R|)` — no image shift.
    Linked,
    /// Independent per-channel gain reduction.
    Independent,
}

impl StereoLinkChoice {
    /// Map to the engine's [`StereoLink`].
    pub fn to_link(self) -> StereoLink {
        match self {
            StereoLinkChoice::Linked => StereoLink::Linked,
            StereoLinkChoice::Independent => StereoLink::Independent,
        }
    }
}

/// The parameter state of one band.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandParams {
    /// Dynamics mode.
    pub mode: BandModeChoice,
    /// Threshold in dBFS.
    pub threshold_db: f32,
    /// Ratio (`:1`).
    pub ratio: f32,
    /// Attack time in milliseconds.
    pub attack_ms: f32,
    /// Release time in milliseconds.
    pub release_ms: f32,
    /// Soft-knee width in dB (`0` ⇒ hard knee).
    pub knee_db: f32,
    /// Makeup gain in dB.
    pub makeup_db: f32,
    /// Detector peak/RMS blend (`0` = RMS, `1` = peak).
    pub blend: f32,
    /// Internal sidechain highpass corner in Hz.
    pub sc_hpf_hz: f32,
    /// Audition only this band.
    pub solo: bool,
    /// Silence this band in the sum.
    pub mute: bool,
}

impl Default for BandParams {
    fn default() -> Self {
        Self {
            mode: BandModeChoice::Compressor,
            threshold_db: -18.0,
            ratio: 2.0,
            attack_ms: 10.0,
            release_ms: 100.0,
            knee_db: 6.0,
            makeup_db: 0.0,
            blend: 0.5,
            sc_hpf_hz: SC_HPF_HZ_RANGE.0,
            solo: false,
            mute: false,
        }
    }
}

impl BandParams {
    /// Clamp every continuous field into range.
    pub fn clamped(mut self) -> Self {
        self.threshold_db = self
            .threshold_db
            .clamp(THRESHOLD_DB_RANGE.0, THRESHOLD_DB_RANGE.1);
        self.ratio = self.ratio.clamp(RATIO_RANGE.0, RATIO_RANGE.1);
        self.attack_ms = self.attack_ms.clamp(ATTACK_MS_RANGE.0, ATTACK_MS_RANGE.1);
        self.release_ms = self
            .release_ms
            .clamp(RELEASE_MS_RANGE.0, RELEASE_MS_RANGE.1);
        self.knee_db = self.knee_db.clamp(KNEE_DB_RANGE.0, KNEE_DB_RANGE.1);
        self.makeup_db = self.makeup_db.clamp(MAKEUP_DB_RANGE.0, MAKEUP_DB_RANGE.1);
        self.blend = self.blend.clamp(BLEND_RANGE.0, BLEND_RANGE.1);
        self.sc_hpf_hz = self.sc_hpf_hz.clamp(SC_HPF_HZ_RANGE.0, SC_HPF_HZ_RANGE.1);
        self
    }
}

/// The full parameter state of one `rt-multiband` instance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultibandParams {
    /// Input gain in dB (applied before the crossover).
    pub input_gain_db: f32,
    /// Output gain in dB (applied after the band sum).
    pub output_gain_db: f32,
    /// Dry/wet mix (`0` = dry, `1` = fully processed).
    pub mix: f32,
    /// Host bypass — output equals the untouched input.
    pub bypass: bool,
    /// Stereo detector linking, applied to every band.
    pub link: StereoLinkChoice,
    /// The three crossover edge frequencies (ascending) in Hz.
    pub crossover_hz: [f32; NUM_CROSSOVERS],
    /// Per-band parameters.
    pub bands: [BandParams; NUM_BANDS],
}

impl Default for MultibandParams {
    fn default() -> Self {
        Self {
            input_gain_db: 0.0,
            output_gain_db: 0.0,
            mix: 1.0,
            bypass: false,
            link: StereoLinkChoice::Linked,
            crossover_hz: DEFAULT_CROSSOVER_HZ,
            bands: [BandParams::default(); NUM_BANDS],
        }
    }
}

impl MultibandParams {
    /// Return a copy with every continuous field clamped into range.
    pub fn clamped(mut self) -> Self {
        self.input_gain_db = self
            .input_gain_db
            .clamp(INPUT_GAIN_DB_RANGE.0, INPUT_GAIN_DB_RANGE.1);
        self.output_gain_db = self
            .output_gain_db
            .clamp(OUTPUT_GAIN_DB_RANGE.0, OUTPUT_GAIN_DB_RANGE.1);
        self.mix = self.mix.clamp(0.0, 1.0);
        for freq in &mut self.crossover_hz {
            *freq = freq.clamp(CROSSOVER_HZ_RANGE.0, CROSSOVER_HZ_RANGE.1);
        }
        for band in &mut self.bands {
            *band = band.clamped();
        }
        self
    }

    /// Whether a change from `self` to `other` requires rebuilding the engine.
    ///
    /// Always `false` in v1: fixed band count and IIR crossover mean every
    /// control is applied in place.  (Reserved for the v2 band-count /
    /// linear-phase changes, which *would* reallocate.)
    #[inline]
    pub fn structural_changed(&self, _other: &Self) -> bool {
        false
    }
}

/// A MIDI-CC target: a per-band control the plugin maps continuous controllers
/// onto (Roadmap Phase 2: "CC on `Threshold` and `Makeup` per band").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CcTarget {
    /// Set band `.0`'s threshold.
    Threshold(usize),
    /// Set band `.0`'s makeup gain.
    Makeup(usize),
}

/// First CC number mapped to a band `Threshold` (bands `0..4` ⇒ CC `20..24`).
pub const THRESHOLD_CC_BASE: u8 = 20;
/// First CC number mapped to a band `Makeup` (bands `0..4` ⇒ CC `24..28`).
pub const MAKEUP_CC_BASE: u8 = 24;

/// Decode a MIDI CC number into the band control it drives, or `None` if the
/// controller is unmapped.
pub fn decode_cc(cc: u8) -> Option<CcTarget> {
    let threshold_end = THRESHOLD_CC_BASE + NUM_BANDS as u8;
    let makeup_end = MAKEUP_CC_BASE + NUM_BANDS as u8;
    if (THRESHOLD_CC_BASE..threshold_end).contains(&cc) {
        Some(CcTarget::Threshold((cc - THRESHOLD_CC_BASE) as usize))
    } else if (MAKEUP_CC_BASE..makeup_end).contains(&cc) {
        Some(CcTarget::Makeup((cc - MAKEUP_CC_BASE) as usize))
    } else {
        None
    }
}

/// Map a normalised CC value (`0..=1`) onto an inclusive `(min, max)` range.
#[inline]
pub fn cc_value_to_range(value: f32, range: (f32, f32)) -> f32 {
    range.0 + value.clamp(0.0, 1.0) * (range.1 - range.0)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_roadmap() {
        let params = MultibandParams::default();
        assert_eq!(params.mix, 1.0);
        assert!(!params.bypass);
        assert_eq!(params.link, StereoLinkChoice::Linked);
        assert_eq!(params.crossover_hz, DEFAULT_CROSSOVER_HZ);
        assert_eq!(params.bands.len(), NUM_BANDS);
        assert_eq!(params.bands[0].mode, BandModeChoice::Compressor);
    }

    #[test]
    fn mode_maps_to_engine() {
        assert_eq!(BandModeChoice::Compressor.to_mode(), BandMode::Compressor);
        assert_eq!(BandModeChoice::Expander.to_mode(), BandMode::Expander);
        assert_eq!(BandModeChoice::Gate.to_mode(), BandMode::Gate);
        assert_eq!(BandModeChoice::Bypass.to_mode(), BandMode::Bypass);
    }

    #[test]
    fn link_maps_to_engine() {
        assert_eq!(StereoLinkChoice::Linked.to_link(), StereoLink::Linked);
        assert_eq!(
            StereoLinkChoice::Independent.to_link(),
            StereoLink::Independent
        );
    }

    #[test]
    fn clamp_pulls_fields_into_range() {
        let mut params = MultibandParams {
            input_gain_db: 100.0,
            output_gain_db: -100.0,
            mix: 2.0,
            crossover_hz: [5.0, 50_000.0, 8_000.0],
            ..Default::default()
        };
        params.bands[0].threshold_db = 20.0;
        params.bands[0].ratio = 100.0;
        params.bands[0].attack_ms = 0.0;
        let params = params.clamped();
        assert_eq!(params.input_gain_db, INPUT_GAIN_DB_RANGE.1);
        assert_eq!(params.output_gain_db, OUTPUT_GAIN_DB_RANGE.0);
        assert_eq!(params.mix, 1.0);
        assert_eq!(params.crossover_hz[0], CROSSOVER_HZ_RANGE.0);
        assert_eq!(params.crossover_hz[1], CROSSOVER_HZ_RANGE.1);
        assert_eq!(params.bands[0].threshold_db, THRESHOLD_DB_RANGE.1);
        assert_eq!(params.bands[0].ratio, RATIO_RANGE.1);
        assert_eq!(params.bands[0].attack_ms, ATTACK_MS_RANGE.0);
    }

    #[test]
    fn clamp_is_idempotent_on_defaults() {
        let params = MultibandParams::default();
        assert_eq!(params.clamped(), params);
    }

    #[test]
    fn structural_change_is_never_forced_in_v1() {
        let a = MultibandParams::default();
        let mut b = a;
        b.crossover_hz = [200.0, 2_000.0, 8_000.0];
        b.bands[1].mode = BandModeChoice::Gate;
        assert!(!a.structural_changed(&b));
    }

    #[test]
    fn cc_decode_maps_thresholds_and_makeups() {
        assert_eq!(decode_cc(20), Some(CcTarget::Threshold(0)));
        assert_eq!(decode_cc(23), Some(CcTarget::Threshold(3)));
        assert_eq!(decode_cc(24), Some(CcTarget::Makeup(0)));
        assert_eq!(decode_cc(27), Some(CcTarget::Makeup(3)));
        assert_eq!(decode_cc(0), None);
        assert_eq!(decode_cc(28), None);
    }

    #[test]
    fn cc_value_maps_across_the_range() {
        assert_eq!(
            cc_value_to_range(0.0, THRESHOLD_DB_RANGE),
            THRESHOLD_DB_RANGE.0
        );
        assert_eq!(
            cc_value_to_range(1.0, THRESHOLD_DB_RANGE),
            THRESHOLD_DB_RANGE.1
        );
        let mid = cc_value_to_range(0.5, THRESHOLD_DB_RANGE);
        assert!((mid - -30.0).abs() < 1e-3);
        // Out-of-range values are clamped.
        assert_eq!(cc_value_to_range(2.0, MAKEUP_DB_RANGE), MAKEUP_DB_RANGE.1);
    }
}
