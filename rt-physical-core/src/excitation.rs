//! The excitation generator for the plucked-string model — the "pick" that
//! sets a string in motion.
//!
//! # What a pluck is, physically
//!
//! Plucking injects a short burst of energy into the string.  The extended
//! Karplus-Strong model reproduces the perceptual cues with three cheap stages
//! (Phase 0 excitation example in `docs/Roadmap-06-physical-modeling.md`):
//!
//! 1. **A short noise burst** — broadband energy, roughly one string period
//!    long, scaled by MIDI velocity (harder pluck → more energy).
//! 2. **Pick-hardness low-pass** — a soft fingertip excites fewer highs than a
//!    hard plectrum, so a one-pole low-pass whose cutoff rises with `hardness`
//!    shapes the burst.
//! 3. **Pluck-position comb** — plucking at a fraction `β` along the string
//!    cancels every harmonic whose node sits under the pick.  That is a
//!    feed-forward comb `y = x − x[n − round(β·L)]`, which notches harmonics at
//!    multiples of `1/β` and gives the pluck its characteristic timbre.
//!
//! # Audio-input excitation
//!
//! Phase 0 chose "both" excitation sources.  For the *effect* use the incoming
//! audio replaces this generator — the plugin feeds the audio sample straight
//! into the string loop instead of calling [`Excitation::next`].  This type is
//! only the *synth* path.
//!
//! # Real-time safety
//!
//! [`next`](Excitation::next) is a handful of multiply-adds plus one integer
//! PRNG step — no allocation, no locking, no `panic`.  The noise source is a
//! deterministic `xorshift` seeded per note, so a voice is reproducible for
//! tests and never touches the system RNG.

use crate::delay::DelayLine;
use crate::filter::LoopFilter;

/// Longest excitation burst, in samples, regardless of period (keeps very low
/// notes from producing an unnaturally long "scrape").
const MAX_BURST: usize = 4_096;

/// A per-voice pluck excitation source: velocity-scaled, hardness-filtered
/// noise through a pluck-position comb.
#[derive(Debug, Clone)]
pub struct Excitation {
    /// Samples of noise still to emit; `0` once the burst is done.
    remaining: usize,
    /// Peak burst amplitude (from velocity).
    amp: f32,
    /// Pick-hardness low-pass shaping the burst spectrum.
    hardness: LoopFilter,
    /// Comb delay line, sized for the lowest supported note.
    comb: DelayLine,
    /// Comb delay in samples: `round(β·L)`.
    comb_delay: usize,
    /// `xorshift32` PRNG state (never zero).
    rng: u32,
}

impl Excitation {
    /// Create an idle excitation whose comb can reach `max_delay` samples (the
    /// loop length of the lowest supported note).
    ///
    /// **Allocates** — construct off the render thread.
    pub fn new(max_delay: usize) -> Self {
        Self {
            remaining: 0,
            amp: 0.0,
            hardness: LoopFilter::new(0.0),
            comb: DelayLine::new(max_delay.max(2)),
            comb_delay: 1,
            rng: 1,
        }
    }

    /// Arm a new pluck.
    ///
    /// - `velocity` in `[0, 1]` scales the burst amplitude.
    /// - `hardness` in `[0, 1]`: 0 = soft (dull), 1 = hard (bright).
    /// - `pluck_pos` in `[0, 1]`: fractional position along the string.
    /// - `loop_len` is the string loop length in samples (`fs / f0`).
    /// - `seed` decorrelates the noise between voices.
    ///
    /// **Real-time safe.**
    pub fn trigger(
        &mut self,
        velocity: f32,
        hardness: f32,
        pluck_pos: f32,
        loop_len: f32,
        seed: u32,
    ) {
        self.amp = velocity.clamp(0.0, 1.0);
        // One period of burst, capped so sub-bass notes don't scrape forever.
        self.remaining = (loop_len as usize).clamp(1, MAX_BURST);
        // Harder pick → brighter → *less* low-pass damping.
        self.hardness.set_damping(1.0 - hardness.clamp(0.0, 1.0));
        self.hardness.set_gain(1.0);
        self.hardness.reset();
        // Keep the pluck away from the exact bridge/nut where the comb vanishes.
        let beta = pluck_pos.clamp(0.02, 0.98);
        self.comb_delay = ((beta * loop_len).round() as usize).max(1);
        self.comb.reset();
        // xorshift needs a non-zero seed.
        self.rng = seed | 1;
    }

    /// `true` while the burst (or its comb tail) may still be producing energy.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.remaining > 0
    }

    /// Silence the generator immediately: end any burst and clear the comb.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn silence(&mut self) {
        self.remaining = 0;
        self.amp = 0.0;
        self.comb.reset();
        self.hardness.reset();
    }

    /// One `xorshift32` step mapped to white noise in `[-1, 1)`.
    #[inline]
    fn white(&mut self) -> f32 {
        let mut state = self.rng;
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        self.rng = state;
        // Top 24 bits → [0, 1), then to [-1, 1).
        (state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
    }

    /// Emit the next excitation sample, or `0.0` once the pluck is spent.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn tick(&mut self) -> f32 {
        let raw = if self.remaining > 0 {
            self.remaining -= 1;
            self.white() * self.amp
        } else {
            0.0
        };
        let shaped = self.hardness.process(raw);
        // Feed-forward comb: current minus the pluck-position-delayed sample.
        let delayed = self.comb.read(self.comb_delay);
        self.comb.write(shaped);
        shaped - delayed
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn burst_ends_and_settles_to_silence() {
        let mut exc = Excitation::new(2_048);
        exc.trigger(1.0, 1.0, 0.3, 200.0, 42);
        assert!(exc.is_active());
        let mut any_energy = false;
        for _ in 0..200 {
            if exc.tick().abs() > 1e-6 {
                any_energy = true;
            }
        }
        assert!(any_energy, "burst produced no energy");
        // After the burst plus its comb tail, output must fall to silence.
        for _ in 0..MAX_BURST * 2 {
            exc.tick();
        }
        assert!(!exc.is_active());
        for _ in 0..64 {
            assert!(exc.tick().abs() < 1e-6, "energy after burst finished");
        }
    }

    #[test]
    fn velocity_scales_amplitude() {
        let peak = |velocity: f32| {
            let mut exc = Excitation::new(2_048);
            exc.trigger(velocity, 1.0, 0.3, 200.0, 7);
            let mut peak = 0.0f32;
            for _ in 0..400 {
                peak = peak.max(exc.tick().abs());
            }
            peak
        };
        assert!(
            peak(1.0) > peak(0.25) * 2.0,
            "velocity did not scale energy"
        );
    }

    #[test]
    fn hardness_controls_brightness() {
        // High-frequency content proxy: mean absolute first difference.
        let brightness = |hardness: f32| {
            let mut exc = Excitation::new(2_048);
            exc.trigger(1.0, hardness, 0.3, 400.0, 99);
            let mut prev = 0.0;
            let mut sum = 0.0f32;
            for _ in 0..400 {
                let sample = exc.tick();
                sum += (sample - prev).abs();
                prev = sample;
            }
            sum
        };
        assert!(
            brightness(1.0) > brightness(0.0),
            "hard pluck should be brighter than soft"
        );
    }

    #[test]
    fn same_seed_is_reproducible() {
        let run = || {
            let mut exc = Excitation::new(2_048);
            exc.trigger(1.0, 0.5, 0.3, 300.0, 12345);
            (0..500).map(|_| exc.tick()).collect::<Vec<_>>()
        };
        assert_eq!(run(), run(), "excitation must be deterministic per seed");
    }

    #[test]
    fn output_is_always_finite() {
        let mut exc = Excitation::new(4_096);
        exc.trigger(1.0, 0.5, 0.5, 3_000.0, 1);
        for _ in 0..10_000 {
            assert!(exc.tick().is_finite());
        }
    }
}
