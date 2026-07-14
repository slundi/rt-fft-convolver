//! Eight-slot interval preset bank with click-free equal-power crossfade.
//!
//! The plugin exposes eight harmony intervals; switching between them must not
//! click.  [`PresetBank`] is the state machine driving that: it stores the
//! eight intervals, tracks the current and previous selection, and hands the
//! render thread an **equal-power** gain pair to blend two shifter voices
//! across a preset change (mirroring the `CrossfadingConvolver` design in the
//! root crate).
//!
//! Equal-power (`cos`/`sin`) rather than linear crossfade keeps the summed
//! *power* constant through the transition, so a switch between two correlated
//! pitch voices holds a steady loudness instead of dipping in the middle.

use crate::shifter::{clamp_ratio, semitones_to_ratio};

/// Number of preset slots.
pub const NUM_PRESETS: usize = 8;

/// One preset slot: an interval and whether it participates in selection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PresetSlot {
    /// Interval in semitones (fractional → cents), typically −24 … +24.
    pub semitones: f32,
    /// Whether MIDI / step selection may land on this slot.
    pub enabled: bool,
}

impl Default for PresetSlot {
    fn default() -> Self {
        Self {
            semitones: 0.0,
            enabled: true,
        }
    }
}

impl PresetSlot {
    /// This slot's pitch ratio (clamped to the shifters' supported range).
    #[inline]
    pub fn ratio(&self) -> f32 {
        clamp_ratio(semitones_to_ratio(self.semitones))
    }
}

/// Eight-slot bank plus an equal-power crossfade between the outgoing and
/// incoming preset.
#[derive(Clone, Debug)]
pub struct PresetBank {
    slots: [PresetSlot; NUM_PRESETS],
    current: usize,
    previous: usize,
    /// Crossfade position in `[0, 1]`; `1.0` = fully on `current`.
    position: f32,
    /// Per-sample increment applied to `position` while crossfading.
    increment: f32,
}

impl PresetBank {
    /// Build a bank whose preset changes crossfade over `fade_ms` at
    /// `sample_rate`.  All slots default to unison, enabled.
    ///
    /// # Panics
    /// Panics if `sample_rate <= 0` or `fade_ms < 0`.
    pub fn new(sample_rate: f32, fade_ms: f32) -> Self {
        assert!(sample_rate > 0.0 && fade_ms >= 0.0, "invalid fade config");
        let fade_samples = (sample_rate * fade_ms / 1000.0).max(1.0);
        Self {
            slots: [PresetSlot::default(); NUM_PRESETS],
            current: 0,
            previous: 0,
            position: 1.0, // start settled on preset 0
            increment: 1.0 / fade_samples,
        }
    }

    /// Configure a slot's interval and enabled flag.
    ///
    /// # Panics
    /// Panics if `index >= NUM_PRESETS`.
    pub fn set_slot(&mut self, index: usize, semitones: f32, enabled: bool) {
        assert!(index < NUM_PRESETS, "preset index out of range");
        self.slots[index] = PresetSlot { semitones, enabled };
    }

    /// Read a slot.
    #[inline]
    pub fn slot(&self, index: usize) -> PresetSlot {
        self.slots[index]
    }

    /// Currently selected slot index.
    #[inline]
    pub fn current_index(&self) -> usize {
        self.current
    }

    /// Change the fade time (e.g. from a `Crossfade Time` control).
    pub fn set_fade_time(&mut self, sample_rate: f32, fade_ms: f32) {
        let fade_samples = (sample_rate * fade_ms / 1000.0).max(1.0);
        self.increment = 1.0 / fade_samples;
    }

    /// Select a preset.  A no-op if already current; otherwise it starts a
    /// crossfade from the present blend toward `index`.  Disabled slots are
    /// ignored.
    pub fn select(&mut self, index: usize) {
        if index >= NUM_PRESETS || !self.slots[index].enabled || index == self.current {
            return;
        }
        // Begin a fresh fade from whatever is currently sounding.
        self.previous = self.current;
        self.current = index;
        self.position = 0.0;
    }

    /// `true` while a crossfade is in progress.
    #[inline]
    pub fn is_crossfading(&self) -> bool {
        self.position < 1.0
    }

    /// Pitch ratio of the outgoing (previous) preset.
    #[inline]
    pub fn previous_ratio(&self) -> f32 {
        self.slots[self.previous].ratio()
    }

    /// Pitch ratio of the incoming (current) preset.
    #[inline]
    pub fn current_ratio(&self) -> f32 {
        self.slots[self.current].ratio()
    }

    /// Advance the crossfade one sample and return the equal-power gains
    /// `(previous_gain, current_gain)` to apply to the two shifter voices.
    #[inline]
    pub fn next_gains(&mut self) -> (f32, f32) {
        use std::f32::consts::FRAC_PI_2;
        let gain_current = (self.position * FRAC_PI_2).sin();
        let gain_previous = (self.position * FRAC_PI_2).cos();
        if self.position < 1.0 {
            self.position = (self.position + self.increment).min(1.0);
        }
        (gain_previous, gain_current)
    }

    /// Reset to a settled state on `index` (transport stop / preset recall)
    /// with no crossfade in flight.
    pub fn snap_to(&mut self, index: usize) {
        let index = index.min(NUM_PRESETS - 1);
        self.current = index;
        self.previous = index;
        self.position = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_bank_is_settled_on_unison() {
        let bank = PresetBank::new(48_000.0, 20.0);
        assert_eq!(bank.current_index(), 0);
        assert!(!bank.is_crossfading());
        assert!((bank.current_ratio() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn slots_map_semitones_to_ratio() {
        let mut bank = PresetBank::new(48_000.0, 20.0);
        bank.set_slot(1, 12.0, true); // octave up
        bank.set_slot(2, -12.0, true); // octave down
        assert!((bank.slot(1).ratio() - 2.0).abs() < 1e-4);
        assert!((bank.slot(2).ratio() - 0.5).abs() < 1e-4);
    }

    #[test]
    fn select_starts_equal_power_crossfade() {
        let mut bank = PresetBank::new(48_000.0, 1.0); // 48 samples
        bank.set_slot(1, 7.0, true);
        bank.select(1);
        assert!(bank.is_crossfading());

        // At the very start the previous voice dominates.
        let (prev0, cur0) = bank.next_gains();
        assert!(prev0 > cur0);

        // Power (prev² + cur²) stays ≈ 1 throughout the fade.
        for _ in 0..48 {
            let (prev, cur) = bank.next_gains();
            let power = prev * prev + cur * cur;
            assert!((power - 1.0).abs() < 1e-4, "power = {power}");
        }
        // Fade completes.
        assert!(!bank.is_crossfading());
        let (prev_end, cur_end) = bank.next_gains();
        assert!(cur_end > 0.99 && prev_end < 0.01);
    }

    #[test]
    fn disabled_slot_is_not_selected() {
        let mut bank = PresetBank::new(48_000.0, 20.0);
        bank.set_slot(3, 5.0, false);
        bank.select(3);
        assert_eq!(bank.current_index(), 0); // unchanged
    }

    #[test]
    fn reselecting_current_is_noop() {
        let mut bank = PresetBank::new(48_000.0, 20.0);
        bank.select(0);
        assert!(!bank.is_crossfading());
    }

    #[test]
    fn snap_to_settles_immediately() {
        let mut bank = PresetBank::new(48_000.0, 50.0);
        bank.set_slot(4, 3.0, true);
        bank.select(4);
        assert!(bank.is_crossfading());
        bank.snap_to(4);
        assert!(!bank.is_crossfading());
        assert_eq!(bank.current_index(), 4);
    }
}
