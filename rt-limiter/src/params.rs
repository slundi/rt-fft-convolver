//! Framework-agnostic parameter model for the `rt-limiter` plugin.
//!
//! [`LimiterParams`] mirrors the Phase-2 control list from
//! `docs/Roadmap-07-true-peak-limiter.md` as a plain `Copy` struct with
//! documented ranges.  Keeping it free of `nih-plug` (which lives behind the
//! `nih` feature) lets the engine and the control mappings be unit-tested
//! without the plugin framework.
//!
//! # How the controls map onto the engine
//!
//! The [`rt_limiter_core::Limiter`] takes a threshold, ceiling, release time,
//! [`ReleaseCurve`], look-ahead and [`ChannelLink`].  The plugin controls sit
//! on top:
//!
//! - `Threshold` / `Ceiling` / `Look-ahead` map straight through.
//! - `Release` is the base recovery time; `Style` scales it and picks the
//!   curve (Phase 0: "punchy/smooth are styles layered on the engine curves").
//! - `Auto Release` overrides the curve to [`ReleaseCurve::Auto`] regardless
//!   of `Style` (the program-dependent "transparent" recovery).
//! - `Oversampling` chooses the true-peak detection factor (structural — a
//!   change rebuilds the engine off the audio thread).
//! - `Channel Link` selects stereo-linked vs mid/side (Phase 0: independent
//!   4-channel deferred).
//! - `Dither` selects the output word length (only active with the crate's
//!   `dither` feature).

use rt_limiter_core::{ChannelLink, ReleaseCurve};

/// Inclusive `Threshold` range in dB — signal above this is limited down.
pub const THRESHOLD_DB_RANGE: (f32, f32) = (-60.0, 0.0);
/// Inclusive `Ceiling` range in dB — the output true-peak limit.
pub const CEILING_DB_RANGE: (f32, f32) = (-12.0, 0.0);
/// Inclusive `Release` range in milliseconds.
pub const RELEASE_MS_RANGE: (f32, f32) = (1.0, 2000.0);
/// Inclusive `Look-ahead` range in milliseconds (Phase 0: user-configurable).
pub const LOOKAHEAD_MS_RANGE: (f32, f32) = (0.1, 20.0);

/// User-selectable true-peak oversampling factor (BS.1770-4 default 4×).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OversampleChoice {
    /// No oversampling — sample-peak only (not true-peak).
    X1,
    /// 2× oversampling.
    X2,
    /// 4× oversampling (the default; BS.1770-4 "sufficient").
    X4,
    /// 8× oversampling.
    X8,
    /// 16× oversampling.
    X16,
}

impl OversampleChoice {
    /// The concrete factor this choice resolves to.
    pub fn factor(self) -> usize {
        match self {
            OversampleChoice::X1 => 1,
            OversampleChoice::X2 => 2,
            OversampleChoice::X4 => 4,
            OversampleChoice::X8 => 8,
            OversampleChoice::X16 => 16,
        }
    }
}

/// Channel-linkage mode (Phase 0: stereo-linked + M/S; independent deferred).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkChoice {
    /// One gain from `max(|L|, |R|)`, applied to both — preserves the image.
    Linked,
    /// The same, computed and applied in the mid/side domain.
    MidSide,
}

impl LinkChoice {
    /// Map to the engine's [`ChannelLink`].
    pub fn to_link(self) -> ChannelLink {
        match self {
            LinkChoice::Linked => ChannelLink::Linked,
            LinkChoice::MidSide => ChannelLink::MidSide,
        }
    }
}

/// Voiced release "style" layered on the engine curves (Phase 0).
///
/// Each style picks a [`ReleaseCurve`] and a multiplier applied to the base
/// `Release` time — so the same control code serves all four voicings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Style {
    /// Program-dependent auto release — the most transparent.
    Transparent,
    /// Fast linear release — punchy, lets transients through.
    Punch,
    /// Long exponential release — smooth, minimises pumping.
    Smooth,
    /// Very fast linear release — aggressive / loud.
    Aggressive,
}

impl Style {
    /// The release curve this style uses (before an `Auto Release` override).
    pub fn curve(self) -> ReleaseCurve {
        match self {
            Style::Transparent => ReleaseCurve::Auto,
            Style::Punch | Style::Aggressive => ReleaseCurve::Linear,
            Style::Smooth => ReleaseCurve::Exponential,
        }
    }

    /// Multiplier applied to the base release time.
    pub fn release_scale(self) -> f32 {
        match self {
            Style::Transparent => 1.0,
            Style::Punch => 0.5,
            Style::Smooth => 1.6,
            Style::Aggressive => 0.3,
        }
    }
}

/// Output dither word length (active only with the `dither` feature).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DitherChoice {
    /// No dither — 32-bit float pass-through.
    Off,
    /// Dither to 16-bit.
    Bits16,
    /// Dither to 20-bit.
    Bits20,
    /// Dither to 24-bit.
    Bits24,
}

/// The full parameter state of one `rt-limiter` instance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LimiterParams {
    /// Limiting level in dBFS — peaks are pulled down to this.
    pub threshold_db: f32,
    /// Output true-peak ceiling in dBFS.
    pub ceiling_db: f32,
    /// Base release time in milliseconds.
    pub release_ms: f32,
    /// Voiced release style.
    pub style: Style,
    /// Override the style's curve with program-dependent auto release.
    pub auto_release: bool,
    /// Look-ahead in milliseconds.
    pub lookahead_ms: f32,
    /// True-peak oversampling factor (structural).
    pub oversampling: OversampleChoice,
    /// Channel-linkage mode.
    pub link: LinkChoice,
    /// Output dither word length.
    pub dither: DitherChoice,
    /// Host bypass — output equals the untouched input.
    pub bypass: bool,
}

impl Default for LimiterParams {
    fn default() -> Self {
        Self {
            threshold_db: 0.0,
            // −1 dBTP: the standard streaming-delivery ceiling.
            ceiling_db: -1.0,
            release_ms: 100.0,
            style: Style::Transparent,
            auto_release: false,
            lookahead_ms: 5.0,
            oversampling: OversampleChoice::X4,
            link: LinkChoice::Linked,
            dither: DitherChoice::Off,
            bypass: false,
        }
    }
}

impl LimiterParams {
    /// Return a copy with every continuous field clamped into range.
    pub fn clamped(mut self) -> Self {
        self.threshold_db = self
            .threshold_db
            .clamp(THRESHOLD_DB_RANGE.0, THRESHOLD_DB_RANGE.1);
        self.ceiling_db = self
            .ceiling_db
            .clamp(CEILING_DB_RANGE.0, CEILING_DB_RANGE.1);
        self.release_ms = self
            .release_ms
            .clamp(RELEASE_MS_RANGE.0, RELEASE_MS_RANGE.1);
        self.lookahead_ms = self
            .lookahead_ms
            .clamp(LOOKAHEAD_MS_RANGE.0, LOOKAHEAD_MS_RANGE.1);
        self
    }

    /// The release curve to hand the engine: `Auto` when `auto_release` is set,
    /// otherwise the style's own curve.
    #[inline]
    pub fn effective_curve(self) -> ReleaseCurve {
        if self.auto_release {
            ReleaseCurve::Auto
        } else {
            self.style.curve()
        }
    }

    /// The release time to hand the engine: the base time scaled by the style,
    /// kept within a sane bound.
    #[inline]
    pub fn effective_release_ms(self) -> f32 {
        (self.release_ms * self.style.release_scale()).clamp(1.0, 4000.0)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_roadmap() {
        let params = LimiterParams::default();
        assert_eq!(params.threshold_db, 0.0);
        assert_eq!(params.ceiling_db, -1.0);
        assert_eq!(params.oversampling, OversampleChoice::X4);
        assert_eq!(params.link, LinkChoice::Linked);
        assert!(!params.bypass);
        assert!(!params.auto_release);
    }

    #[test]
    fn oversample_factors() {
        assert_eq!(OversampleChoice::X1.factor(), 1);
        assert_eq!(OversampleChoice::X4.factor(), 4);
        assert_eq!(OversampleChoice::X16.factor(), 16);
    }

    #[test]
    fn style_maps_curve_and_scale() {
        assert_eq!(Style::Transparent.curve(), ReleaseCurve::Auto);
        assert_eq!(Style::Punch.curve(), ReleaseCurve::Linear);
        assert_eq!(Style::Smooth.curve(), ReleaseCurve::Exponential);
        // Punch is faster than the base; Smooth is slower.
        assert!(Style::Punch.release_scale() < 1.0);
        assert!(Style::Smooth.release_scale() > 1.0);
    }

    #[test]
    fn auto_release_overrides_style_curve() {
        let params = LimiterParams {
            style: Style::Punch,
            auto_release: true,
            ..Default::default()
        };
        assert_eq!(params.effective_curve(), ReleaseCurve::Auto);
    }

    #[test]
    fn effective_release_scales_with_style() {
        let base = LimiterParams {
            release_ms: 100.0,
            style: Style::Punch,
            ..Default::default()
        };
        assert!((base.effective_release_ms() - 50.0).abs() < 1e-3);
    }

    #[test]
    fn clamp_pulls_fields_into_range() {
        let params = LimiterParams {
            threshold_db: 20.0,
            ceiling_db: 5.0,
            release_ms: 9000.0,
            lookahead_ms: 100.0,
            ..Default::default()
        }
        .clamped();
        assert_eq!(params.threshold_db, THRESHOLD_DB_RANGE.1);
        assert_eq!(params.ceiling_db, CEILING_DB_RANGE.1);
        assert_eq!(params.release_ms, RELEASE_MS_RANGE.1);
        assert_eq!(params.lookahead_ms, LOOKAHEAD_MS_RANGE.1);
    }

    #[test]
    fn clamp_is_idempotent_on_defaults() {
        let params = LimiterParams::default();
        assert_eq!(params.clamped(), params);
    }

    #[test]
    fn link_maps_to_engine() {
        assert_eq!(LinkChoice::Linked.to_link(), ChannelLink::Linked);
        assert_eq!(LinkChoice::MidSide.to_link(), ChannelLink::MidSide);
    }
}
