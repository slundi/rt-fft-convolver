//! Framework-agnostic UI logic for the (Phase-3) editor.
//!
//! The egui drawing lives in the `gui`-gated `plugin` module; everything here
//! is pure, `egui`-free and therefore **unit-testable**:
//!
//! - [`transfer_curve`] — samples the current shaper for the transfer-curve
//!   visualiser.
//! - [`peak_dbfs`] + [`LevelMeter`] — VU-meter measurement (audio thread) and
//!   display ballistics (editor thread).
//! - [`format_oversampling`] / [`format_cpu`] — the oversampling / CPU badge
//!   text.
//! - [`AbState`] + [`clipboard_string`] — A/B compare and copy-to-clipboard.

use rt_drive_core::Shaper;
use rt_fft_convolver::db_to_linear;

use crate::params::{DriveParams, OversampleChoice};

/// The dBFS value used to represent digital silence in the meters.
pub const SILENCE_DBFS: f32 = -120.0;

/// Sample the shaper transfer curve `y = f(x · gain)` over `x ∈ [-1, 1]`.
///
/// Returns `points` `(x, y)` pairs for the visualiser, where `input_gain_db`
/// is the *effective* drive (pedal base + the `Drive` knob's offset).  The
/// curve is what the plot draws; `y` is bounded because every [`Shaper`] is.
pub fn transfer_curve(shaper: Shaper, input_gain_db: f32, points: usize) -> Vec<(f32, f32)> {
    if points == 0 {
        return Vec::new();
    }
    if points == 1 {
        return vec![(0.0, shaper.process(0.0))];
    }
    let gain = db_to_linear(input_gain_db);
    let last = (points - 1) as f32;
    (0..points)
        .map(|index| {
            let x = -1.0 + 2.0 * index as f32 / last;
            (x, shaper.process(x * gain))
        })
        .collect()
}

/// Peak level of a block in dBFS, floored at [`SILENCE_DBFS`].
///
/// Called on the audio thread (cheap, no allocation) to feed the meter atomic.
pub fn peak_dbfs(samples: &[f32]) -> f32 {
    let peak = samples
        .iter()
        .fold(0.0f32, |maximum, &sample| maximum.max(sample.abs()));
    if peak <= 1.0e-6 {
        SILENCE_DBFS
    } else {
        20.0 * peak.log10()
    }
}

/// Display-side VU ballistics: instant attack, exponential release.
///
/// The audio thread publishes a per-block peak; the editor feeds it here each
/// frame so the bar snaps up to peaks and eases back down.
#[derive(Debug, Clone, Copy)]
pub struct LevelMeter {
    current_dbfs: f32,
}

impl LevelMeter {
    /// A meter reading digital silence.
    pub fn new() -> Self {
        Self {
            current_dbfs: SILENCE_DBFS,
        }
    }

    /// Update toward `target_dbfs`.  Rising targets are taken instantly (peak
    /// attack); falling targets ease down by `release` (`0..=1`, larger =
    /// faster).  Returns the new displayed value.
    pub fn update(&mut self, target_dbfs: f32, release: f32) -> f32 {
        if target_dbfs >= self.current_dbfs {
            self.current_dbfs = target_dbfs;
        } else {
            let release = release.clamp(0.0, 1.0);
            self.current_dbfs += (target_dbfs - self.current_dbfs) * release;
        }
        self.current_dbfs
    }

    /// The current displayed level in dBFS.
    #[inline]
    pub fn value_dbfs(&self) -> f32 {
        self.current_dbfs
    }
}

impl Default for LevelMeter {
    fn default() -> Self {
        Self::new()
    }
}

/// Badge text for the oversampling factor, e.g. `"4×"`.
pub fn format_oversampling(factor: u8) -> String {
    format!("{factor}×")
}

/// Badge text for a CPU load fraction (`0.12` → `"12%"`), clamped to a sane
/// display range.
pub fn format_cpu(load_fraction: f32) -> String {
    let percent = (load_fraction * 100.0).clamp(0.0, 999.0);
    format!("{percent:.0}%")
}

/// Human name of an [`OversampleChoice`] for badges / clipboard.
pub fn oversample_name(choice: OversampleChoice) -> &'static str {
    match choice {
        OversampleChoice::Preset => "Preset",
        OversampleChoice::X1 => "1×",
        OversampleChoice::X2 => "2×",
        OversampleChoice::X4 => "4×",
        OversampleChoice::X8 => "8×",
        OversampleChoice::X16 => "16×",
    }
}

/// Which of the two A/B compare slots is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbSlot {
    /// Slot A.
    A,
    /// Slot B.
    B,
}

/// A/B preset-compare state: two parameter snapshots plus the active slot.
#[derive(Debug, Clone, Copy)]
pub struct AbState {
    slot_a: DriveParams,
    slot_b: DriveParams,
    active: AbSlot,
}

impl AbState {
    /// Both slots start at `initial`, slot A active.
    pub fn new(initial: DriveParams) -> Self {
        Self {
            slot_a: initial,
            slot_b: initial,
            active: AbSlot::A,
        }
    }

    /// The active slot.
    #[inline]
    pub fn active(&self) -> AbSlot {
        self.active
    }

    /// Store `params` into the currently active slot (called as the user edits).
    pub fn store(&mut self, params: DriveParams) {
        match self.active {
            AbSlot::A => self.slot_a = params,
            AbSlot::B => self.slot_b = params,
        }
    }

    /// The parameters of the active slot.
    pub fn recall(&self) -> DriveParams {
        match self.active {
            AbSlot::A => self.slot_a,
            AbSlot::B => self.slot_b,
        }
    }

    /// Switch active slot and return the newly-active parameters (to push to
    /// the host).
    pub fn activate(&mut self, slot: AbSlot) -> DriveParams {
        self.active = slot;
        self.recall()
    }

    /// Copy the active slot's parameters into the inactive slot.
    pub fn copy_active_to_other(&mut self) {
        match self.active {
            AbSlot::A => self.slot_b = self.slot_a,
            AbSlot::B => self.slot_a = self.slot_b,
        }
    }
}

/// A readable one-preset summary for copy-to-clipboard.
///
/// `pedal_name` is the display name of the current model; the rest is the live
/// control state.
pub fn clipboard_string(params: &DriveParams, pedal_name: &str) -> String {
    format!(
        "# rt-drive preset\n\
         model = \"{name}\"\n\
         drive = {drive:.0}%\n\
         level_db = {level:.1}\n\
         bass_db = {bass:.1}\n\
         mid_db = {mid:.1}\n\
         treble_db = {treble:.1}\n\
         mix = {mix:.0}%\n\
         oversampling = \"{os}\"\n\
         bias = {bias:.2}\n\
         bypass = {bypass}\n",
        name = pedal_name,
        drive = params.drive * 100.0,
        level = params.level_db,
        bass = params.bass_db,
        mid = params.mid_db,
        treble = params.treble_db,
        mix = params.mix * 100.0,
        os = oversample_name(params.oversampling),
        bias = params.bias,
        bypass = params.bypass,
    )
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transfer_curve_has_requested_length() {
        let curve = transfer_curve(Shaper::Tanh, 12.0, 128);
        assert_eq!(curve.len(), 128);
    }

    #[test]
    fn transfer_curve_passes_through_origin_and_is_monotonic() {
        let curve = transfer_curve(Shaper::Tanh, 6.0, 65);
        // Midpoint (index 32) is x = 0 → y = 0.
        let (mid_x, mid_y) = curve[32];
        assert!(mid_x.abs() < 1e-6);
        assert!(mid_y.abs() < 1e-6);
        // Monotonic non-decreasing in x, bounded in y.
        let mut previous = f32::NEG_INFINITY;
        for &(_, y) in &curve {
            assert!(y >= previous - 1e-6, "curve not monotonic");
            assert!(y.abs() <= 1.5, "curve unbounded: {y}");
            previous = y;
        }
    }

    #[test]
    fn transfer_curve_degenerate_point_counts() {
        assert!(transfer_curve(Shaper::Tanh, 0.0, 0).is_empty());
        assert_eq!(transfer_curve(Shaper::Tanh, 0.0, 1).len(), 1);
    }

    #[test]
    fn peak_dbfs_of_silence_is_floor() {
        assert_eq!(peak_dbfs(&[0.0; 16]), SILENCE_DBFS);
        assert_eq!(peak_dbfs(&[]), SILENCE_DBFS);
    }

    #[test]
    fn peak_dbfs_of_full_scale_is_zero() {
        assert!((peak_dbfs(&[0.5, -1.0, 0.25]) - 0.0).abs() < 1e-4);
    }

    #[test]
    fn meter_attacks_instantly_and_releases_gradually() {
        let mut meter = LevelMeter::new();
        // Instant attack up to a peak.
        assert!((meter.update(-6.0, 0.2) - -6.0).abs() < 1e-6);
        // Release eases toward a lower target, not instantly.
        let after = meter.update(-24.0, 0.5);
        assert!(
            after > -24.0 && after < -6.0,
            "release not gradual: {after}"
        );
    }

    #[test]
    fn badge_formatting() {
        assert_eq!(format_oversampling(4), "4×");
        assert_eq!(format_cpu(0.12), "12%");
        assert_eq!(format_cpu(-1.0), "0%");
        assert_eq!(oversample_name(OversampleChoice::X8), "8×");
    }

    #[test]
    fn ab_store_recall_roundtrip() {
        let mut state = AbState::new(DriveParams::default());
        let edited = DriveParams {
            drive: 0.9,
            ..Default::default()
        };
        state.store(edited);
        assert_eq!(state.recall(), edited);

        // Slot B is still the initial default until activated + stored.
        let b_params = state.activate(AbSlot::B);
        assert_eq!(b_params, DriveParams::default());
        assert_eq!(state.active(), AbSlot::B);
    }

    #[test]
    fn ab_copy_makes_both_slots_equal() {
        let mut state = AbState::new(DriveParams::default());
        let edited = DriveParams {
            mix: 0.25,
            ..Default::default()
        };
        state.store(edited); // into A
        state.copy_active_to_other(); // A → B
        assert_eq!(state.activate(AbSlot::B), edited);
    }

    #[test]
    fn clipboard_string_contains_key_fields() {
        let params = DriveParams {
            drive: 0.75,
            level_db: 3.0,
            ..Default::default()
        };
        let text = clipboard_string(&params, "Green Overdrive");
        assert!(text.contains("model = \"Green Overdrive\""));
        assert!(text.contains("drive = 75%"));
        assert!(text.contains("level_db = 3.0"));
        assert!(text.contains("oversampling = \"Preset\""));
    }
}
