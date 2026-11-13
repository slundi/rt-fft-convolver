//! Fixed-capacity bank of biquad notch filters for feedback suppression.
//!
//! The [`FeedbackDetector`](crate::detector::FeedbackDetector) locates a
//! howling partial and the plugin asks this bank to place a narrow notch there.
//! Slots are a fixed array of [`Biquad`](rt_spectral_core::Biquad) notches
//! (Roadmap-09 Phase 0 reuse) plus a per-slot `enabled` flag, so allocating and
//! releasing a notch is pure coefficient math — no heap traffic, RT-safe.
//!
//! The active notches form a **series cascade**: a sample passes through every
//! enabled notch in turn, so N notches attenuate N distinct partials.

use rt_spectral_core::Biquad;

/// Maximum number of simultaneous notches (Roadmap-09 "Max Notches (1–24)").
pub const MAX_NOTCHES: usize = 24;

/// A notch is considered a duplicate of an existing one when their centre
/// frequencies are within this fraction of each other, so the detector cannot
/// pile several notches onto the same partial before it is suppressed.
const DUPLICATE_TOLERANCE: f32 = 0.03;

/// Read-only description of one placed notch (for the Phase 3 inventory view).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NotchInfo {
    /// Centre frequency in Hz.
    pub freq_hz: f32,
    /// Quality factor (higher is narrower).
    pub q: f32,
    /// Samples elapsed since the notch was placed (for the "age" column and
    /// oldest-first recycling).
    pub age_samples: u64,
}

impl Default for NotchInfo {
    fn default() -> Self {
        Self {
            freq_hz: 0.0,
            q: 0.0,
            age_samples: 0,
        }
    }
}

/// A bank of up to [`MAX_NOTCHES`] biquad notches.
#[derive(Debug)]
pub struct NotchBank {
    sample_rate: u32,
    filters: [Biquad; MAX_NOTCHES],
    enabled: [bool; MAX_NOTCHES],
    info: [NotchInfo; MAX_NOTCHES],
    /// User cap on how many slots may be active at once (`1 ..= MAX_NOTCHES`).
    max_active: usize,
}

impl NotchBank {
    /// Build an empty bank for `sample_rate`.  Pre-allocates every slot.
    ///
    /// **Allocates** — off-RT only.
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            filters: [Biquad::identity(); MAX_NOTCHES],
            enabled: [false; MAX_NOTCHES],
            info: [NotchInfo::default(); MAX_NOTCHES],
            max_active: MAX_NOTCHES,
        }
    }

    /// Cap the number of concurrently active notches (the `Max Notches`
    /// control).  Clamped to `1 ..= MAX_NOTCHES`.  Does not retroactively
    /// release notches already above the new cap — the caller can
    /// [`reset`](Self::reset) for that.
    pub fn set_max_active(&mut self, max_active: usize) {
        self.max_active = max_active.clamp(1, MAX_NOTCHES);
    }

    /// Number of currently active notches.
    pub fn active_count(&self) -> usize {
        self.enabled.iter().filter(|&&on| on).count()
    }

    /// Place a notch at `freq_hz` with quality `q`.
    ///
    /// Returns the slot index used, or `None` when the bank is at its
    /// `max_active` cap and no near-duplicate slot exists.  If a notch already
    /// covers `freq_hz` (within [`DUPLICATE_TOLERANCE`]) its slot is returned
    /// unchanged instead of consuming a new one.
    ///
    /// # Real-time safety
    /// Coefficient math only — no allocation.
    pub fn allocate(&mut self, freq_hz: f32, q: f32) -> Option<usize> {
        // Re-use a slot already sitting on this partial.
        for (index, info) in self.info.iter().enumerate() {
            if self.enabled[index]
                && (info.freq_hz - freq_hz).abs() <= freq_hz * DUPLICATE_TOLERANCE
            {
                return Some(index);
            }
        }

        if self.active_count() >= self.max_active {
            return None;
        }

        // First free slot.
        let slot = self.enabled.iter().position(|&on| !on)?;
        self.filters[slot].set_notch(self.sample_rate, freq_hz, q);
        self.filters[slot].reset();
        self.enabled[slot] = true;
        self.info[slot] = NotchInfo {
            freq_hz,
            q,
            age_samples: 0,
        };
        Some(slot)
    }

    /// Release the notch in `slot`, freeing it for re-use.
    pub fn release(&mut self, slot: usize) {
        if slot < MAX_NOTCHES {
            self.enabled[slot] = false;
            self.info[slot] = NotchInfo::default();
            self.filters[slot].reset();
        }
    }

    /// Release every notch (the `Reset` control).
    pub fn reset(&mut self) {
        for slot in 0..MAX_NOTCHES {
            self.release(slot);
        }
    }

    /// Run every active notch over `buf` in place (series cascade) and age the
    /// active notches by the block length.
    ///
    /// # Real-time safety
    /// No allocation.
    pub fn process_block(&mut self, buf: &mut [f32]) {
        let advance = buf.len() as u64;
        for slot in 0..MAX_NOTCHES {
            if self.enabled[slot] {
                self.filters[slot].process_block(buf);
                self.info[slot].age_samples += advance;
            }
        }
    }

    /// Iterate the active notches as `(slot, NotchInfo)` for the inventory UI.
    pub fn active(&self) -> impl Iterator<Item = (usize, NotchInfo)> + '_ {
        (0..MAX_NOTCHES)
            .filter(move |&slot| self.enabled[slot])
            .map(move |slot| (slot, self.info[slot]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{make_sine, rms};

    const SR: u32 = 48_000;

    /// Peak-normalised gain a settled notch bank applies to a sustained tone.
    fn tone_attenuation_db(bank: &mut NotchBank, freq: f32) -> f32 {
        let mut buf = make_sine(freq, SR as f32, 48_000);
        let before = rms(&buf[24_000..]);
        bank.process_block(&mut buf);
        let after = rms(&buf[24_000..]);
        20.0 * (after / before).log10()
    }

    #[test]
    fn notch_attenuates_target_by_over_20_db() {
        // Phase 4 acceptance bar: a 1 kHz notch at Q=10 drops a 1 kHz sine
        // by > 20 dB.
        let mut bank = NotchBank::new(SR);
        assert_eq!(bank.allocate(1_000.0, 10.0), Some(0));
        let attenuation = tone_attenuation_db(&mut bank, 1_000.0);
        assert!(attenuation < -20.0, "only {attenuation} dB of rejection");
    }

    #[test]
    fn empty_bank_is_transparent() {
        let mut bank = NotchBank::new(SR);
        let mut buf = make_sine(1_000.0, SR as f32, 512);
        let before = buf.clone();
        bank.process_block(&mut buf);
        assert_eq!(buf, before);
    }

    #[test]
    fn duplicate_frequency_reuses_slot() {
        let mut bank = NotchBank::new(SR);
        let first = bank.allocate(1_000.0, 10.0).unwrap();
        // 1.01 kHz is within tolerance of 1 kHz → same slot, no new notch.
        let second = bank.allocate(1_010.0, 10.0).unwrap();
        assert_eq!(first, second);
        assert_eq!(bank.active_count(), 1);
    }

    #[test]
    fn max_active_cap_is_respected() {
        let mut bank = NotchBank::new(SR);
        bank.set_max_active(2);
        assert!(bank.allocate(300.0, 10.0).is_some());
        assert!(bank.allocate(900.0, 10.0).is_some());
        assert_eq!(bank.allocate(1_500.0, 10.0), None, "cap should block a 3rd");
        assert_eq!(bank.active_count(), 2);
    }

    #[test]
    fn reset_releases_all_notches() {
        let mut bank = NotchBank::new(SR);
        bank.allocate(300.0, 10.0);
        bank.allocate(900.0, 10.0);
        bank.reset();
        assert_eq!(bank.active_count(), 0);
    }

    #[test]
    fn active_ages_advance_with_processing() {
        let mut bank = NotchBank::new(SR);
        let slot = bank.allocate(1_000.0, 10.0).unwrap();
        let mut buf = vec![0.0f32; 128];
        bank.process_block(&mut buf);
        bank.process_block(&mut buf);
        let (_, info) = bank.active().find(|(s, _)| *s == slot).unwrap();
        assert_eq!(info.age_samples, 256);
    }
}
