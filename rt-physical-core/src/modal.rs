//! The modal bank — a parallel array of two-pole resonators that turns a single
//! excitation impulse into a sum of decaying sinusoids (Phase 1 "Modal bank"
//! in `docs/Roadmap-06-physical-modeling.md`).
//!
//! # The model
//!
//! A struck object (bar, bell, plate) rings at a fixed set of modal
//! frequencies, each an inharmonic multiple `ratio·f0` of a nominal fundamental
//! and each with its own decay time.  We realise each mode as a
//! [`Biquad::resonator`] and sum their outputs.  Feeding all resonators one
//! impulse yields the classic "excitation impulse → sum of decaying sinusoids".
//!
//! # Timbre presets
//!
//! [`ModePreset`] supplies the inharmonic ratio set that makes the material
//! recognisable — a tuned wooden bar (marimba: 1 : 4 : 10) sounds nothing like
//! a metal bell (≈ 1 : 2.76 : 5.40).  The `decay_slope` control then makes the
//! higher, brighter modes ring shorter or longer than the fundamental.
//!
//! # Real-time safety
//!
//! [`process`](ModalBank::process) is `active_modes` biquad steps — no
//! allocation, no locking.  Retuning ([`set_voice`](ModalBank::set_voice))
//! rebuilds the resonator coefficients and is meant for note-on, off the hot
//! loop.  The resonator array is allocated once at construction and only ever
//! reused.

use crate::biquad::Biquad;

/// Hard ceiling on modes per voice (matches the plugin's `Modes` control range,
/// Phase 2).  The resonator array is sized to this at construction.
pub const MAX_MODES: usize = 32;

/// A material timbre: the inharmonic frequency ratios (relative to the
/// fundamental) and matching amplitudes of its modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModePreset {
    /// Metal bell / free bar — strongly inharmonic, long ringing.
    Metal,
    /// Tuned wooden bar (marimba-like), roughly 1 : 4 : 10.
    Wood,
    /// Glass / rectangular plate — dense, bright, inharmonic.
    Glass,
}

impl ModePreset {
    /// Frequency ratios of the modes, fundamental first.
    pub fn ratios(self) -> &'static [f32] {
        match self {
            // Free–free bar / bell partials.
            ModePreset::Metal => &[1.0, 2.756, 5.404, 8.933, 13.34, 18.64],
            // Marimba bars are tuned so partials fall near 1 : 4 : 10.
            ModePreset::Wood => &[1.0, 4.0, 10.0, 17.6, 26.8],
            // Rectangular plate / glass partials.
            ModePreset::Glass => &[1.0, 2.32, 4.25, 6.63, 9.38, 12.6, 16.3],
        }
    }
}

/// A bank of resonators sharing one fundamental and excitation.
#[derive(Debug, Clone)]
pub struct ModalBank {
    /// Resonators, allocated to [`MAX_MODES`] once and reused.
    modes: Vec<Biquad>,
    /// Number of resonators currently sounding (`≤ modes.len()`).
    active: usize,
    /// Impulse energy queued for the next [`process`](ModalBank::process).
    pending: f32,
    sample_rate: f32,
    /// Fundamental currently tuned, for introspection.
    fundamental: f32,
}

impl ModalBank {
    /// Create a bank with capacity for [`MAX_MODES`] resonators (all silent
    /// until [`set_voice`](ModalBank::set_voice)).
    ///
    /// **Allocates** — construct off the render thread.
    pub fn new(sample_rate: f32) -> Self {
        Self {
            modes: vec![Biquad::default(); MAX_MODES],
            active: 0,
            pending: 0.0,
            sample_rate,
            fundamental: 0.0,
        }
    }

    /// (Re)tune the bank to `fundamental` Hz using `preset`, taking at most
    /// `num_modes` modes.  `base_t60` is the fundamental's decay time;
    /// `decay_slope ≥ 0` makes higher modes decay faster (`> 1`) or slower
    /// (`< 1`) than the fundamental — bright modes usually die first.
    ///
    /// Modes whose frequency would exceed Nyquist are dropped.  **Real-time
    /// safe** apart from the per-mode trig / `powf` in the resonator design.
    pub fn set_voice(
        &mut self,
        fundamental: f32,
        preset: ModePreset,
        num_modes: usize,
        base_t60: f32,
        decay_slope: f32,
    ) {
        self.fundamental = fundamental.max(1.0);
        let nyquist = self.sample_rate * 0.5;
        let ratios = preset.ratios();
        let wanted = num_modes.min(ratios.len()).min(self.modes.len());
        let slope = decay_slope.max(0.0);

        let mut built = 0;
        for (index, &ratio) in ratios.iter().take(wanted).enumerate() {
            let freq = self.fundamental * ratio;
            if freq >= nyquist {
                break; // and every higher mode too — ratios are ascending
            }
            // Higher modes ring shorter when slope > 1 (bright decays first).
            let t60 = (base_t60 / ratio.powf(slope)).max(1e-3);
            // Gently taper amplitude so the fundamental leads the timbre.
            let amp = 1.0 / (index as f32 + 1.0);
            self.modes[built] = Biquad::resonator(freq, t60, amp, self.sample_rate);
            built += 1;
        }
        self.active = built;
    }

    /// Queue an excitation impulse of the given `energy` (e.g. MIDI velocity);
    /// it is injected on the next [`process`](ModalBank::process).
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn excite(&mut self, energy: f32) {
        self.pending = energy;
    }

    /// Number of modes currently sounding.
    #[inline]
    pub fn active_modes(&self) -> usize {
        self.active
    }

    /// Clear all resonator state and any queued excitation.
    pub fn reset(&mut self) {
        for mode in &mut self.modes {
            mode.reset();
        }
        self.pending = 0.0;
    }

    /// Advance every active resonator by one sample and return their sum.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn process(&mut self) -> f32 {
        let input = self.pending;
        self.pending = 0.0;
        let mut sum = 0.0;
        for mode in &mut self.modes[..self.active] {
            sum += mode.process(input);
        }
        sum
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    fn frequency_from_zero_crossings(signal: &[f32]) -> f32 {
        let mut crossings = Vec::new();
        for (index, window) in signal.windows(2).enumerate() {
            if window[0] <= 0.0 && window[1] > 0.0 {
                let frac = -window[0] / (window[1] - window[0]);
                crossings.push(index as f32 + frac);
            }
        }
        assert!(crossings.len() >= 8, "too few crossings");
        let spacing: f32 = crossings
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .sum::<f32>()
            / (crossings.len() - 1) as f32;
        SR / spacing
    }

    /// A single-mode bank is a decaying sinusoid at the fundamental (Phase 4
    /// acceptance), validated within ±5 cents.
    #[test]
    fn single_mode_is_a_decaying_sinusoid() {
        let mut bank = ModalBank::new(SR);
        bank.set_voice(440.0, ModePreset::Metal, 1, 2.0, 1.0);
        bank.excite(1.0);
        let signal: Vec<f32> = (0..20_000).map(|_| bank.process()).collect();
        let measured = frequency_from_zero_crossings(&signal[200..]);
        let cents = 1200.0 * (measured / 440.0).log2();
        assert!(cents.abs() < 5.0, "single mode off by {cents} cents");
        // …and it decays.
        let head: f32 = signal[..2_000].iter().map(|sample| sample * sample).sum();
        let tail: f32 = signal[18_000..].iter().map(|sample| sample * sample).sum();
        assert!(tail < head, "single mode did not decay");
    }

    #[test]
    fn mode_count_is_clamped_and_reported() {
        let mut bank = ModalBank::new(SR);
        // Metal preset has 6 ratios; asking for 100 clamps to 6.
        bank.set_voice(220.0, ModePreset::Metal, 100, 2.0, 1.0);
        assert_eq!(bank.active_modes(), ModePreset::Metal.ratios().len());
    }

    #[test]
    fn modes_above_nyquist_are_dropped() {
        let mut bank = ModalBank::new(SR);
        // At a 10 kHz fundamental, metal ratios quickly exceed 24 kHz Nyquist.
        bank.set_voice(10_000.0, ModePreset::Metal, 6, 2.0, 1.0);
        assert!(bank.active_modes() < 6, "high modes were not dropped");
    }

    #[test]
    fn steeper_decay_slope_shortens_the_tail() {
        let tail_energy = |slope: f32| {
            let mut bank = ModalBank::new(SR);
            bank.set_voice(220.0, ModePreset::Metal, 6, 2.0, slope);
            bank.excite(1.0);
            (0..48_000)
                .map(|_| bank.process())
                .skip(24_000)
                .map(|sample| sample * sample)
                .sum::<f32>()
        };
        assert!(
            tail_energy(2.5) < tail_energy(0.2),
            "slope did not shorten tail"
        );
    }

    #[test]
    fn reset_prevents_state_leak() {
        let mut reused = ModalBank::new(SR);
        reused.set_voice(300.0, ModePreset::Wood, 4, 2.0, 1.0);
        reused.excite(1.0);
        for _ in 0..5_000 {
            reused.process();
        }
        reused.reset();
        reused.set_voice(196.0, ModePreset::Wood, 4, 2.0, 1.0);
        reused.excite(1.0);
        let reused_out: Vec<f32> = (0..2_000).map(|_| reused.process()).collect();

        let mut fresh = ModalBank::new(SR);
        fresh.set_voice(196.0, ModePreset::Wood, 4, 2.0, 1.0);
        fresh.excite(1.0);
        let fresh_out: Vec<f32> = (0..2_000).map(|_| fresh.process()).collect();

        assert_eq!(reused_out, fresh_out, "reused bank leaked prior state");
    }

    #[test]
    fn output_is_always_finite() {
        let mut bank = ModalBank::new(SR);
        bank.set_voice(150.0, ModePreset::Glass, 7, 20.0, 1.0);
        bank.excite(1.0);
        for _ in 0..96_000 {
            assert!(bank.process().is_finite());
        }
    }
}
