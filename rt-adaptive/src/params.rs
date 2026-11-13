//! Framework-agnostic parameter model for the `rt-adaptive` plugin.
//!
//! [`AdaptiveParams`] mirrors the Phase-2 control list from
//! `docs/Roadmap-09-adaptive-filters.md` as a plain `Copy` struct with
//! documented ranges.  Keeping it free of `nih-plug` (which lives behind the
//! `nih` feature) lets the engine and ranges be unit-tested without the plugin
//! framework.
//!
//! # How the controls map onto `rt-adaptive-core`
//!
//! - [`mode`](AdaptiveParams::mode) selects which engine renders — feedback
//!   suppression, hum removal, or echo cancellation.  It is **continuous**: all
//!   three engines stay resident so switching never drops placed notches or a
//!   converged echo path.
//! - [`filter_length`](AdaptiveParams::filter_length) is the one **structural**
//!   control — changing the AEC tap count rebuilds the echo canceller off the
//!   audio thread.
//! - Everything else (sensitivity, Q, harmonics, convergence speed, gain, …) is
//!   applied in place on the audio thread every block.
//!
//! # Notch depth
//!
//! The reused `rt_spectral_core::Biquad` notch is a full-depth RBJ notch, so the
//! roadmap's dB "depth" controls are expressed here as the notch **`Q`**
//! (narrowness) the primitive supports; a variable-depth notch biquad is a
//! deferred v1.1 refinement.

/// Which of the three engines is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mode {
    /// Detect howling partials and notch them out (live PA).
    #[default]
    FeedbackSuppressor,
    /// Comb-notch the mains fundamental and its harmonics.
    HumRemover,
    /// Cancel the echo of a far-end reference from the near-end signal.
    EchoCanceller,
}

/// AEC adaptive-filter length, in taps (a **structural** control).
///
/// Discrete sizes so the const-generic
/// [`AdaptiveFilter`](rt_adaptive_core::AdaptiveFilter) can be reused directly;
/// a length change rebuilds the canceller off the audio thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FilterLength {
    /// 256 taps (≈ 5.3 ms at 48 kHz) — short room echo.
    L256,
    /// 512 taps (≈ 10.7 ms) — the default.
    #[default]
    L512,
    /// 1024 taps (≈ 21.3 ms).
    L1024,
    /// 2048 taps (≈ 42.7 ms) — long tails.
    L2048,
}

impl FilterLength {
    /// Number of adaptive taps.
    #[inline]
    pub fn taps(self) -> usize {
        match self {
            FilterLength::L256 => 256,
            FilterLength::L512 => 512,
            FilterLength::L1024 => 1024,
            FilterLength::L2048 => 2048,
        }
    }
}

/// Inclusive feedback-detector sensitivity range (`0` least, `1` most).
pub const SENSITIVITY_RANGE: (f32, f32) = (0.0, 1.0);
/// Inclusive `Max Notches` range.
pub const MAX_NOTCHES_RANGE: (i32, i32) = (1, rt_adaptive_core::MAX_NOTCHES as i32);
/// Inclusive notch quality-factor range.
pub const NOTCH_Q_RANGE: (f32, f32) = (0.5, 50.0);
/// Inclusive custom mains-fundamental range in Hz (covers 50 and 60).
pub const HUM_FUNDAMENTAL_RANGE: (f32, f32) = (40.0, 70.0);
/// Inclusive number-of-harmonics range.
pub const HUM_HARMONICS_RANGE: (i32, i32) = (1, 10);
/// Inclusive hum notch quality-factor range.
pub const HUM_Q_RANGE: (f32, f32) = (1.0, 50.0);
/// Inclusive NLMS convergence-speed (`μ`) range.
pub const CONVERGENCE_RANGE: (f32, f32) = (0.05, 1.0);
/// Inclusive double-talk threshold range in dB (near-end must exceed the echo
/// estimate by this margin before adaptation freezes).
pub const DOUBLE_TALK_DB_RANGE: (f32, f32) = (0.0, 24.0);
/// Inclusive output-gain range in dB.
pub const OUTPUT_GAIN_DB_RANGE: (f32, f32) = (-24.0, 24.0);

/// The full parameter state of one `rt-adaptive` instance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveParams {
    /// Active engine.
    pub mode: Mode,

    // ── Feedback suppressor ──
    /// Detector sensitivity `[0, 1]`.
    pub sensitivity: f32,
    /// Maximum simultaneous notches (`1 ..= 24`).
    pub max_notches: usize,
    /// Notch quality factor (narrowness).
    pub notch_q: f32,
    /// Automatic detection (`true`) vs. manual notch placement (`false`).
    pub auto: bool,
    /// Stop allocating new notches, keep the existing ones.
    pub freeze: bool,
    /// Rising edge clears every notch.
    pub reset: bool,

    // ── Hum remover ──
    /// Mains fundamental in Hz (50 / 60 / custom).
    pub hum_fundamental_hz: f32,
    /// Number of harmonics to notch (including the fundamental).
    pub hum_harmonics: usize,
    /// Track mains-frequency drift.
    pub hum_adaptive: bool,
    /// Hum notch quality factor.
    pub hum_q: f32,

    // ── Echo canceller ──
    /// Adaptive-filter length (structural).
    pub filter_length: FilterLength,
    /// NLMS convergence speed (`μ`).
    pub convergence: f32,
    /// Double-talk freeze threshold in dB.
    pub double_talk_db: f32,

    // ── Global ──
    /// Output gain in dB.
    pub output_gain_db: f32,
    /// Host bypass — output equals the untouched input.
    pub bypass: bool,
}

impl Default for AdaptiveParams {
    fn default() -> Self {
        Self {
            mode: Mode::default(),
            sensitivity: 0.5,
            max_notches: 8,
            notch_q: 20.0,
            auto: true,
            freeze: false,
            reset: false,
            hum_fundamental_hz: 50.0,
            hum_harmonics: 3,
            hum_adaptive: true,
            hum_q: 30.0,
            filter_length: FilterLength::default(),
            convergence: 0.5,
            double_talk_db: 6.0,
            output_gain_db: 0.0,
            bypass: false,
        }
    }
}

impl AdaptiveParams {
    /// Return a copy with every continuous field clamped into range.
    pub fn clamped(mut self) -> Self {
        self.sensitivity = self
            .sensitivity
            .clamp(SENSITIVITY_RANGE.0, SENSITIVITY_RANGE.1);
        self.max_notches = self
            .max_notches
            .clamp(MAX_NOTCHES_RANGE.0 as usize, MAX_NOTCHES_RANGE.1 as usize);
        self.notch_q = self.notch_q.clamp(NOTCH_Q_RANGE.0, NOTCH_Q_RANGE.1);
        self.hum_fundamental_hz = self
            .hum_fundamental_hz
            .clamp(HUM_FUNDAMENTAL_RANGE.0, HUM_FUNDAMENTAL_RANGE.1);
        self.hum_harmonics = self.hum_harmonics.clamp(
            HUM_HARMONICS_RANGE.0 as usize,
            HUM_HARMONICS_RANGE.1 as usize,
        );
        self.hum_q = self.hum_q.clamp(HUM_Q_RANGE.0, HUM_Q_RANGE.1);
        self.convergence = self
            .convergence
            .clamp(CONVERGENCE_RANGE.0, CONVERGENCE_RANGE.1);
        self.double_talk_db = self
            .double_talk_db
            .clamp(DOUBLE_TALK_DB_RANGE.0, DOUBLE_TALK_DB_RANGE.1);
        self.output_gain_db = self
            .output_gain_db
            .clamp(OUTPUT_GAIN_DB_RANGE.0, OUTPUT_GAIN_DB_RANGE.1);
        self
    }

    /// `true` when `other` differs in a **structural** field — one that forces
    /// the off-thread controller to rebuild the echo canceller.  Only the AEC
    /// filter length qualifies; every other control is applied continuously.
    #[inline]
    pub fn structural_changed(&self, other: &Self) -> bool {
        self.filter_length != other.filter_length
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_roadmap() {
        let params = AdaptiveParams::default();
        assert_eq!(params.mode, Mode::FeedbackSuppressor);
        assert_eq!(params.filter_length, FilterLength::L512);
        assert!(params.auto);
        assert!(!params.freeze);
        assert!(!params.bypass);
        assert_eq!(params.hum_fundamental_hz, 50.0);
    }

    #[test]
    fn filter_lengths_map_to_taps() {
        assert_eq!(FilterLength::L256.taps(), 256);
        assert_eq!(FilterLength::L512.taps(), 512);
        assert_eq!(FilterLength::L1024.taps(), 1024);
        assert_eq!(FilterLength::L2048.taps(), 2048);
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn clamp_pulls_fields_into_range() {
        let mut params = AdaptiveParams::default();
        params.sensitivity = 5.0;
        params.max_notches = 999;
        params.notch_q = 0.0;
        params.hum_fundamental_hz = 1_000.0;
        params.hum_harmonics = 99;
        params.convergence = 10.0;
        params.output_gain_db = 100.0;
        let params = params.clamped();
        assert_eq!(params.sensitivity, SENSITIVITY_RANGE.1);
        assert_eq!(params.max_notches, MAX_NOTCHES_RANGE.1 as usize);
        assert_eq!(params.notch_q, NOTCH_Q_RANGE.0);
        assert_eq!(params.hum_fundamental_hz, HUM_FUNDAMENTAL_RANGE.1);
        assert_eq!(params.hum_harmonics, HUM_HARMONICS_RANGE.1 as usize);
        assert_eq!(params.convergence, CONVERGENCE_RANGE.1);
        assert_eq!(params.output_gain_db, OUTPUT_GAIN_DB_RANGE.1);
    }

    #[test]
    fn clamp_is_idempotent_on_defaults() {
        let params = AdaptiveParams::default();
        assert_eq!(params.clamped(), params);
    }

    #[test]
    fn only_filter_length_is_structural() {
        let base = AdaptiveParams::default();

        // Continuous changes: no rebuild.
        let mut cont = base;
        cont.mode = Mode::EchoCanceller;
        cont.sensitivity = 0.9;
        cont.convergence = 0.9;
        cont.freeze = true;
        assert!(!base.structural_changed(&cont));

        // Structural change: rebuild.
        let mut longer = base;
        longer.filter_length = FilterLength::L2048;
        assert!(base.structural_changed(&longer));
    }
}
