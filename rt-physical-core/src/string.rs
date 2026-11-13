//! The extended Karplus-Strong plucked-string kernel — one string, monophonic.
//! A [`crate::voice::Voice`] pairs one of these with an [`crate::excitation::Excitation`].
//!
//! # Signal flow (one sample)
//!
//! ```text
//!  excitation ─▶(+)─▶ delay line ─▶ tuning all-pass ─▶ loop filter ─┐
//!                ▲                                                   │
//!                └───────────────────────────────────────────────────┘  (feedback)
//! ```
//!
//! Read the delayed string sample, nudge it by the sub-sample tuning all-pass,
//! run it through the loop (damping + decay) filter, add the current excitation
//! and write the sum back.  The delayed-and-tuned sample is the output.
//!
//! # Tuning budget (the "explain partition/loop indices" rule)
//!
//! The string is in tune only when the **total** loop delay equals `L = fs/f0`
//! samples.  Three elements share that budget:
//!
//! ```text
//! L  =  n_int   +   frac        +   lp_delay
//!       (delay)     (all-pass)      (loop filter phase delay)
//! ```
//!
//! So we compute `L`, subtract the loop-filter phase delay at the fundamental,
//! and split the remainder into an integer tap `n_int` and a fraction `frac`
//! for the all-pass.  **The classic bug** is forgetting `lp_delay`: the note
//! then plays sharp, worst in the high register where the half-sample filter
//! delay is a larger share of a short `L`.  We keep `frac ∈ [0.1, 1.1)` by
//! borrowing one integer sample when needed, so the all-pass stays well
//! conditioned (see [`crate::allpass`]).
//!
//! # Real-time safety
//!
//! [`process`](KarplusStrong::process) is a few multiply-adds across the delay,
//! all-pass and loop filter — no allocation, no locking, no `panic`.  Retuning
//! ([`set_note`](KarplusStrong::set_note)) runs a little trig / one `powf` and
//! is meant for note-on, off the hot inner loop.

use crate::allpass::TuningAllpass;
use crate::delay::DelayLine;
use crate::filter::LoopFilter;
use std::f32::consts::TAU;

/// Loop gain used while a note is released (short natural mute).
const RELEASE_T60: f32 = 0.18;

/// A single extended Karplus-Strong string.
#[derive(Debug, Clone)]
pub struct KarplusStrong {
    delay: DelayLine,
    allpass: TuningAllpass,
    loop_filter: LoopFilter,
    /// Integer part of the loop delay currently in use.
    n_int: usize,
    sample_rate: f32,
    /// Fundamental currently tuned (Hz), for the release retune.
    freq: f32,
}

impl KarplusStrong {
    /// Create a string whose delay line can reach `max_loop_len` samples (i.e.
    /// the lowest supported note).
    ///
    /// **Allocates** — construct off the render thread.
    pub fn new(sample_rate: f32, max_loop_len: usize) -> Self {
        Self {
            delay: DelayLine::new(max_loop_len),
            allpass: TuningAllpass::new(0.5),
            loop_filter: LoopFilter::new(0.5),
            n_int: 1,
            sample_rate,
            freq: 440.0,
        }
    }

    /// Tune to `freq` Hz and set the tone (`damping`) and decay (`t60`).
    ///
    /// Splits the `fs/f0` loop-delay budget across the integer delay, the
    /// all-pass fraction and the loop filter's phase delay, then derives the
    /// per-pass decay gain.  **Real-time safe** apart from a little trig.
    pub fn set_note(&mut self, freq: f32, damping: f32, t60: f32) {
        self.freq = freq.max(1.0);
        let loop_len = (self.sample_rate / self.freq).max(2.0);
        let omega = TAU * self.freq / self.sample_rate;

        self.loop_filter.set_damping(damping);
        self.loop_filter.set_decay(t60, self.freq);

        // Budget: L = n_int + frac + loop-filter phase delay.
        let lp_delay = self.loop_filter.phase_delay(omega);
        let remaining = (loop_len - lp_delay).max(1.1);
        let mut n_int = remaining.floor() as usize;
        let mut frac = remaining - n_int as f32;
        // Keep the all-pass in its well-conditioned range by borrowing a sample.
        if frac < 0.1 && n_int > 1 {
            n_int -= 1;
            frac += 1.0;
        }
        // Never exceed the buffer (leave a sample of headroom).
        self.n_int = n_int.clamp(1, self.delay.capacity() - 2);
        self.allpass.set_delay(frac);
    }

    /// Shorten the decay so a released note fades naturally instead of ringing
    /// forever.  **Real-time safe.**
    pub fn note_off(&mut self) {
        self.loop_filter.set_decay(RELEASE_T60, self.freq);
    }

    /// Clear the string state (delay, all-pass and filter memory).  Call before
    /// retriggering a reused voice so no energy leaks between notes.
    pub fn reset(&mut self) {
        self.delay.reset();
        self.allpass.reset();
        self.loop_filter.reset();
    }

    /// The integer loop delay currently in use (for tests / introspection).
    #[inline]
    pub fn integer_delay(&self) -> usize {
        self.n_int
    }

    /// Advance the string by one sample, injecting `excitation` into the loop.
    /// Returns the string output.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn process(&mut self, excitation: f32) -> f32 {
        let delayed = self.delay.read(self.n_int);
        let tuned = self.allpass.process(delayed);
        let filtered = self.loop_filter.process(tuned);
        self.delay.write(excitation + filtered);
        tuned
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::excitation::Excitation;
    use crate::pitch::max_loop_len;

    const SR: f32 = 48_000.0;

    /// Run a plucked note and collect its output.
    fn pluck(freq: f32, damping: f32, t60: f32, samples: usize) -> Vec<f32> {
        let mut string = KarplusStrong::new(SR, max_loop_len(SR));
        string.set_note(freq, damping, t60);
        let mut exc = Excitation::new(max_loop_len(SR));
        exc.trigger(1.0, 0.5, 0.3, SR / freq, 1234);
        (0..samples).map(|_| string.process(exc.tick())).collect()
    }

    /// Estimate the fundamental of a (harmonically rich) string signal by
    /// autocorrelation, searching only near `target` to avoid octave errors,
    /// then refining the peak lag with parabolic interpolation for sub-sample
    /// (few-cent) resolution — zero-crossing counting is unusable on a bright
    /// pluck because overtones add spurious crossings.
    fn fundamental_hz(signal: &[f32], target: f32) -> f32 {
        let min_lag = (SR / (target * 1.5)).floor().max(2.0) as usize;
        let max_lag = (SR / (target * 0.75)).ceil() as usize;
        let correlate = |lag: usize| -> f32 {
            signal
                .iter()
                .zip(&signal[lag..])
                .map(|(a, b)| a * b)
                .sum::<f32>()
        };
        let mut best_lag = min_lag;
        let mut best_corr = f32::MIN;
        for lag in min_lag..=max_lag {
            let corr = correlate(lag);
            if corr > best_corr {
                best_corr = corr;
                best_lag = lag;
            }
        }
        // Parabolic interpolation of the peak using its two neighbours.
        let (left, centre, right) = (
            correlate(best_lag - 1),
            correlate(best_lag),
            correlate(best_lag + 1),
        );
        let denom = left - 2.0 * centre + right;
        let delta = if denom.abs() > 1e-12 {
            0.5 * (left - right) / denom
        } else {
            0.0
        };
        SR / (best_lag as f32 + delta)
    }

    /// A plucked note matches the requested pitch — the Phase 4 acceptance
    /// target of ±5 cents, validated already at the engine level.
    #[test]
    fn pitch_is_within_five_cents() {
        for &note in &[110.0, 220.0, 440.0, 880.0] {
            let signal = pluck(note, 0.3, 3.0, 24_000);
            // Skip the noisy attack; measure the settled portion.
            let measured = fundamental_hz(&signal[2_000..], note);
            let cents = 1200.0 * (measured / note).log2();
            assert!(cents.abs() < 5.0, "{note} Hz off by {cents} cents");
        }
    }

    /// The string decays: late energy is far below early energy.
    #[test]
    fn string_decays_over_time() {
        let signal = pluck(220.0, 0.3, 1.0, 96_000);
        let head: f32 = signal[..4_800].iter().map(|sample| sample * sample).sum();
        let tail: f32 = signal[91_200..].iter().map(|sample| sample * sample).sum();
        assert!(
            tail < head * 1e-2,
            "string did not decay: head {head}, tail {tail}"
        );
    }

    /// More damping ⇒ a darker tail (less high-frequency wiggle).
    #[test]
    fn damping_darkens_the_tail() {
        let roughness = |damping: f32| {
            let signal = pluck(220.0, damping, 3.0, 24_000);
            let tail = &signal[12_000..];
            tail.windows(2)
                .map(|pair| (pair[1] - pair[0]).abs())
                .sum::<f32>()
        };
        assert!(
            roughness(0.9) < roughness(0.05),
            "damping did not darken tail"
        );
    }

    /// Output stays finite across a long, near-sustained decay.
    #[test]
    fn output_is_finite_when_sustained() {
        let signal = pluck(196.0, 0.2, 30.0, 192_000);
        assert!(signal.iter().all(|sample| sample.is_finite()));
    }

    /// `reset` erases history: a retriggered string reproduces a fresh one
    /// sample-for-sample (no state leak between reused voices).
    #[test]
    fn reset_prevents_state_leak() {
        let mut reused = KarplusStrong::new(SR, max_loop_len(SR));
        reused.set_note(330.0, 0.4, 2.0);
        let mut warm_exc = Excitation::new(max_loop_len(SR));
        warm_exc.trigger(1.0, 0.5, 0.3, SR / 330.0, 1);
        for _ in 0..5_000 {
            reused.process(warm_exc.tick()); // leave the string ringing
        }
        // Retrigger the reused string exactly like a fresh one.
        reused.reset();
        reused.set_note(247.0, 0.4, 2.0);
        let mut exc_a = Excitation::new(max_loop_len(SR));
        exc_a.trigger(1.0, 0.5, 0.3, SR / 247.0, 55);
        let reused_out: Vec<f32> = (0..2_000).map(|_| reused.process(exc_a.tick())).collect();

        let mut fresh = KarplusStrong::new(SR, max_loop_len(SR));
        fresh.set_note(247.0, 0.4, 2.0);
        let mut exc_b = Excitation::new(max_loop_len(SR));
        exc_b.trigger(1.0, 0.5, 0.3, SR / 247.0, 55);
        let fresh_out: Vec<f32> = (0..2_000).map(|_| fresh.process(exc_b.tick())).collect();

        assert_eq!(reused_out, fresh_out, "reused string leaked prior state");
    }
}
