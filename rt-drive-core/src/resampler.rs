//! Kaiser-windowed **half-band polyphase** resampling for oversampling.
//!
//! # Why half-band, why polyphase
//!
//! Oversampling by `2ᵏ` is built by cascading `k` stages that each change the
//! rate by exactly 2×.  A *half-band* low-pass (cutoff at `fs/4`) is the ideal
//! per-octave anti-imaging / anti-aliasing filter: its impulse response has a
//! zero at every even tap except the centre, so half the multiplies vanish.
//! Split into its two **polyphase** sub-filters, a 2× interpolator becomes:
//!
//! - even output phase = a pure delayed copy of the input (centre tap only);
//! - odd output phase  = one FIR of the input (the non-zero taps).
//!
//! and the 2× decimator is the transpose.  This is why the roadmap can promise
//! "group delay cancels between the up- and down-leg": both legs are the same
//! linear-phase filter, so their delays add to an integer sample count that is
//! identical on the way up and the way down.
//!
//! # Coefficient design ([decision] / deviation)
//!
//! The roadmap asks for "fixed coefficient tables at compile time (`const fn`
//! or generated at build)".  Stable Rust cannot evaluate the Kaiser window
//! (modified Bessel `I₀`, `sqrt`, `sin`) in a `const fn`, so coefficients are
//! instead computed **once in [`Oversampler::new`]** (off the audio thread)
//! and then fixed for the object's lifetime.  The render path
//! ([`Oversampler::upsample`] / [`Oversampler::downsample`]) only reads those
//! tables — no allocation, no transcendentals.  If truly const tables are
//! wanted later, a `build.rs` can emit this same design.
//!
//! # Real-time safety
//!
//! `upsample`, `downsample` and the per-stage `process` methods allocate
//! nothing and never lock or panic (release).  The maximum block size is fixed
//! at construction so all scratch buffers are pre-sized.

use crate::kernels::Kernels;
use std::f64::consts::PI;

// ─── Kaiser half-band FIR design (construction-time only) ───────────────────

/// Modified Bessel function of the first kind, order 0 — `I₀(x)`.
///
/// Series `Σ_{m≥0} ((x/2)^{2m}) / (m!)²`, summed until terms are negligible.
/// Used only for the Kaiser window at design time.
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let half_x_squared = (x / 2.0) * (x / 2.0);
    let mut m = 1.0;
    loop {
        term *= half_x_squared / (m * m);
        sum += term;
        if term < 1e-12 * sum {
            break;
        }
        m += 1.0;
    }
    sum
}

/// Normalised sinc: `sinc(t) = sin(πt)/(πt)`, with `sinc(0) = 1`.
fn sinc(t: f64) -> f64 {
    if t == 0.0 {
        1.0
    } else {
        let arg = PI * t;
        arg.sin() / arg
    }
}

/// Kaiser `β` for a target stop-band attenuation `atten_db` (Kaiser's formula).
fn kaiser_beta(atten_db: f64) -> f64 {
    if atten_db > 50.0 {
        0.1102 * (atten_db - 8.7)
    } else if atten_db >= 21.0 {
        0.5842 * (atten_db - 21.0).powf(0.4) + 0.07886 * (atten_db - 21.0)
    } else {
        0.0
    }
}

/// Half-band FIR tap count for a target attenuation and normalised transition
/// width, rounded up to `≡ 1 (mod 4)` so the centre tap sits on the even
/// polyphase phase.
fn half_band_len(atten_db: f64, transition_frac: f64) -> usize {
    let transition_rad = 2.0 * PI * transition_frac;
    let estimate = ((atten_db - 8.0) / (2.285 * transition_rad)).ceil() as usize + 1;
    let mut length = estimate.max(5);
    while length % 4 != 1 {
        length += 1;
    }
    length
}

/// Design a normalised half-band low-pass (cutoff `fs/4`, DC gain 1).
///
/// The returned length is `≡ 1 (mod 4)`; every even-offset tap except the
/// centre is forced to exactly zero and the odd-offset taps are scaled so
/// their sum is `0.5` (centre is `0.5`), giving a total DC gain of exactly 1
/// and perfectly balanced polyphase sub-filters.
fn design_half_band(num_taps: usize, beta: f64) -> Vec<f32> {
    debug_assert_eq!(num_taps % 4, 1, "half-band length must be ≡ 1 (mod 4)");
    let center = (num_taps - 1) / 2; // even by construction
    let i0_beta = bessel_i0(beta);
    let mut taps = vec![0.0f64; num_taps];

    for (index, tap) in taps.iter_mut().enumerate() {
        let offset = index as f64 - center as f64;
        // Ideal half-band impulse response: 0.5 · sinc(offset/2).
        let ideal = 0.5 * sinc(offset / 2.0);
        // Kaiser window.
        let ratio = offset / center as f64;
        let window = bessel_i0(beta * (1.0 - ratio * ratio).max(0.0).sqrt()) / i0_beta;
        *tap = ideal * window;
    }

    // Force the exact half-band zero pattern: even offsets (≠ centre) → 0.
    for (index, tap) in taps.iter_mut().enumerate() {
        let offset = index as isize - center as isize;
        if offset != 0 && offset % 2 == 0 {
            *tap = 0.0;
        }
    }
    taps[center] = 0.5;

    // Normalise the odd-offset taps so they sum to 0.5 → total DC gain = 1.
    let odd_sum: f64 = taps
        .iter()
        .enumerate()
        .filter(|(index, _)| (*index as isize - center as isize) % 2 != 0)
        .map(|(_, value)| *value)
        .sum();
    let scale = 0.5 / odd_sum;
    for (index, tap) in taps.iter_mut().enumerate() {
        if (index as isize - center as isize) % 2 != 0 {
            *tap *= scale;
        }
    }

    taps.into_iter().map(|value| value as f32).collect()
}

/// Push `sample` into a most-recent-first history buffer (shift right by one).
#[inline]
fn push_history(history: &mut [f32], sample: f32) {
    let len = history.len();
    history.copy_within(0..len - 1, 1);
    history[0] = sample;
}

// ─── 2× stages ──────────────────────────────────────────────────────────────

/// A single 2× half-band **interpolator** (up-sampler).
///
/// Produces two output samples per input sample: `out[2n]` is a delayed copy
/// of the input (even polyphase) and `out[2n+1]` is the odd-phase FIR.
#[derive(Clone)]
pub struct Upsampler2x {
    /// Even-index taps `{h[0], h[2], …}` (all zero except the centre).
    phase_even: Vec<f32>,
    /// Odd-index taps `{h[1], h[3], …}` — the interpolating filter.
    phase_odd: Vec<f32>,
    /// Most-recent-first input history, length = `phase_even.len()`.
    history: Vec<f32>,
    kernels: Kernels,
}

impl Upsampler2x {
    /// Split a half-band prototype into its two polyphase sub-filters.
    fn from_prototype(prototype: &[f32], kernels: Kernels) -> Self {
        let phase_even: Vec<f32> = prototype.iter().step_by(2).copied().collect();
        let phase_odd: Vec<f32> = prototype.iter().skip(1).step_by(2).copied().collect();
        let history = vec![0.0; phase_even.len()];
        Self {
            phase_even,
            phase_odd,
            history,
            kernels,
        }
    }

    /// Interpolate `input` into `output`, which must be exactly twice as long.
    ///
    /// **Real-time safe.**
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(output.len(), input.len() * 2, "upsample: output must be 2×");
        let odd_len = self.phase_odd.len();
        for (index, &sample) in input.iter().enumerate() {
            push_history(&mut self.history, sample);
            // ×2 compensates for the energy lost to zero-stuffing.
            let even = 2.0 * self.kernels.fir_dot(&self.phase_even, &self.history);
            let odd = 2.0
                * self
                    .kernels
                    .fir_dot(&self.phase_odd, &self.history[..odd_len]);
            output[2 * index] = even;
            output[2 * index + 1] = odd;
        }
    }

    /// Clear the delay line.
    pub fn reset(&mut self) {
        self.history.iter_mut().for_each(|sample| *sample = 0.0);
    }
}

/// A single 2× half-band **decimator** (down-sampler).
///
/// Consumes two input samples per output sample; anti-aliases with the same
/// half-band prototype then keeps one sample in two.
#[derive(Clone)]
pub struct Downsampler2x {
    phase_even: Vec<f32>,
    phase_odd: Vec<f32>,
    /// History of the even-position input samples (`u[2n], u[2n−2], …`).
    history_even: Vec<f32>,
    /// History of the odd-position input samples (`u[2n−1], u[2n−3], …`).
    history_odd: Vec<f32>,
    kernels: Kernels,
}

impl Downsampler2x {
    fn from_prototype(prototype: &[f32], kernels: Kernels) -> Self {
        let phase_even: Vec<f32> = prototype.iter().step_by(2).copied().collect();
        let phase_odd: Vec<f32> = prototype.iter().skip(1).step_by(2).copied().collect();
        let history_even = vec![0.0; phase_even.len()];
        let history_odd = vec![0.0; phase_odd.len()];
        Self {
            phase_even,
            phase_odd,
            history_even,
            history_odd,
            kernels,
        }
    }

    /// Decimate `input` into `output`; `input` must be exactly twice as long.
    ///
    /// **Real-time safe.**
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(
            input.len(),
            output.len() * 2,
            "downsample: input must be 2×"
        );
        // Decimation `y[n] = Σ_k h[k]·u[2n−k]` split by tap parity: the even
        // taps pair with the even-position samples `u[2n], u[2n−2], …` and the
        // odd taps with the odd-position samples `u[2n−1], u[2n−3], …` — all of
        // which *precede* `u[2n]`.  So we walk the high-rate stream in order,
        // routing each sample to its phase history, and emit one output as each
        // even sample completes it (the preceding odd sample is already in
        // place).  Even-length blocks keep the parity aligned across calls.
        let mut out_index = 0;
        for (high_index, &sample) in input.iter().enumerate() {
            if high_index.is_multiple_of(2) {
                push_history(&mut self.history_even, sample);
                let even = self.kernels.fir_dot(&self.phase_even, &self.history_even);
                let odd = self.kernels.fir_dot(&self.phase_odd, &self.history_odd);
                // No ×2 here: Σ(phase_even) + Σ(phase_odd) = 1 already → unity DC.
                output[out_index] = even + odd;
                out_index += 1;
            } else {
                push_history(&mut self.history_odd, sample);
            }
        }
    }

    /// Clear the delay lines.
    pub fn reset(&mut self) {
        self.history_even.iter_mut().for_each(|s| *s = 0.0);
        self.history_odd.iter_mut().for_each(|s| *s = 0.0);
    }
}

// ─── Oversampler (cascade) ──────────────────────────────────────────────────

/// A power-of-two oversampler: a cascade of 2× half-band stages.
///
/// `factor` is `1, 2, 4, 8 or 16`.  A factor of 1 is a bypass (no filtering).
/// Pre-size all scratch with the maximum input block length at construction.
pub struct Oversampler {
    factor: usize,
    upsamplers: Vec<Upsampler2x>,
    downsamplers: Vec<Downsampler2x>,
    /// Two ping-pong scratch buffers sized `max_block · factor`.
    scratch_a: Vec<f32>,
    scratch_b: Vec<f32>,
    max_block: usize,
}

impl Oversampler {
    /// Default stop-band attenuation (dB) — meets the roadmap's < −80 dBFS
    /// aliasing-floor target with margin.
    pub const DEFAULT_ATTEN_DB: f64 = 96.0;
    /// Default normalised transition width (fraction of `fs`).
    ///
    /// Chosen so the final 2×→1× decimation stage's stop-band starts *below*
    /// the 5th harmonic of a 5 kHz tone (25 kHz, just above the 24 kHz base
    /// Nyquist) — the worst-case aliasing stress from the roadmap.  A wider
    /// transition leaves that harmonic in the transition band and caps the
    /// aliasing floor around −70 dBFS regardless of oversampling factor (the
    /// final stage is identical for every factor).  See the Phase-4
    /// `spectral` tests.
    pub const DEFAULT_TRANSITION: f64 = 0.018;

    /// Create an oversampler for `factor ∈ {1, 2, 4, 8, 16}` and a maximum
    /// input block of `max_block` samples, using the default filter design.
    ///
    /// **Allocates** — call off the audio thread.
    pub fn new(factor: usize, max_block: usize) -> Self {
        Self::with_design(
            factor,
            max_block,
            Self::DEFAULT_ATTEN_DB,
            Self::DEFAULT_TRANSITION,
        )
    }

    /// Create an oversampler with an explicit half-band filter design.
    ///
    /// `factor` is rounded down to the nearest power of two in `1..=16`.
    ///
    /// **Allocates** — call off the audio thread.
    pub fn with_design(
        factor: usize,
        max_block: usize,
        atten_db: f64,
        transition_frac: f64,
    ) -> Self {
        let factor = normalise_factor(factor);
        let stages = factor.trailing_zeros() as usize; // log2(factor)
        let kernels = Kernels::new();

        let (upsamplers, downsamplers) = if stages == 0 {
            (Vec::new(), Vec::new())
        } else {
            let beta = kaiser_beta(atten_db);
            let length = half_band_len(atten_db, transition_frac);
            let prototype = design_half_band(length, beta);
            let upsamplers = (0..stages)
                .map(|_| Upsampler2x::from_prototype(&prototype, kernels))
                .collect();
            let downsamplers = (0..stages)
                .map(|_| Downsampler2x::from_prototype(&prototype, kernels))
                .collect();
            (upsamplers, downsamplers)
        };

        let capacity = max_block * factor;
        Self {
            factor,
            upsamplers,
            downsamplers,
            scratch_a: vec![0.0; capacity],
            scratch_b: vec![0.0; capacity],
            max_block,
        }
    }

    /// The oversampling factor (`1, 2, 4, 8, 16`).
    #[inline]
    pub fn factor(&self) -> usize {
        self.factor
    }

    /// The maximum input block length this instance was sized for.
    #[inline]
    pub fn max_block(&self) -> usize {
        self.max_block
    }

    /// Up-sample `input` (length `n`) into `output` (length `n · factor`).
    ///
    /// **Real-time safe** — provided `input.len() ≤ max_block`.
    pub fn upsample(&mut self, input: &[f32], output: &mut [f32]) {
        let block = input.len();
        debug_assert!(block <= self.max_block, "block exceeds max_block");
        debug_assert_eq!(output.len(), block * self.factor, "upsample output length");

        if self.factor == 1 {
            output[..block].copy_from_slice(input);
            return;
        }

        // Stage 0 reads from `input`; later stages ping-pong between scratches.
        let mut current_len = block;
        self.scratch_a[..block].copy_from_slice(input);
        for (stage, upsampler) in self.upsamplers.iter_mut().enumerate() {
            let out_len = current_len * 2;
            if stage % 2 == 0 {
                let (src, dst) = (&self.scratch_a, &mut self.scratch_b);
                upsampler.process(&src[..current_len], &mut dst[..out_len]);
            } else {
                let (src, dst) = (&self.scratch_b, &mut self.scratch_a);
                upsampler.process(&src[..current_len], &mut dst[..out_len]);
            }
            current_len = out_len;
        }
        // The final result is in scratch_a if an even number of stages ran,
        // else in scratch_b.
        let final_buffer = if self.upsamplers.len().is_multiple_of(2) {
            &self.scratch_a
        } else {
            &self.scratch_b
        };
        output.copy_from_slice(&final_buffer[..current_len]);
    }

    /// Down-sample `input` (length `n · factor`) into `output` (length `n`).
    ///
    /// **Real-time safe** — provided `output.len() ≤ max_block`.
    pub fn downsample(&mut self, input: &[f32], output: &mut [f32]) {
        let block = output.len();
        debug_assert!(block <= self.max_block, "block exceeds max_block");
        debug_assert_eq!(input.len(), block * self.factor, "downsample input length");

        if self.factor == 1 {
            output.copy_from_slice(&input[..block]);
            return;
        }

        let mut current_len = block * self.factor;
        self.scratch_a[..current_len].copy_from_slice(input);
        for (stage, downsampler) in self.downsamplers.iter_mut().enumerate() {
            let out_len = current_len / 2;
            if stage % 2 == 0 {
                let (src, dst) = (&self.scratch_a, &mut self.scratch_b);
                downsampler.process(&src[..current_len], &mut dst[..out_len]);
            } else {
                let (src, dst) = (&self.scratch_b, &mut self.scratch_a);
                downsampler.process(&src[..current_len], &mut dst[..out_len]);
            }
            current_len = out_len;
        }
        let final_buffer = if self.downsamplers.len().is_multiple_of(2) {
            &self.scratch_a
        } else {
            &self.scratch_b
        };
        output.copy_from_slice(&final_buffer[..current_len]);
    }

    /// Clear every stage's delay line.
    pub fn reset(&mut self) {
        self.upsamplers.iter_mut().for_each(Upsampler2x::reset);
        self.downsamplers.iter_mut().for_each(Downsampler2x::reset);
    }
}

/// Round `factor` down to the nearest power of two within `1..=16`.
fn normalise_factor(factor: usize) -> usize {
    match factor {
        0 | 1 => 1,
        2 | 3 => 2,
        4..=7 => 4,
        8..=15 => 8,
        _ => 16,
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    /// A light design keeps tests fast while still comfortably band-limiting.
    fn light() -> (f64, f64) {
        (80.0, 0.1)
    }

    /// The designed prototype must have unity DC gain and the half-band zero
    /// pattern (even-offset taps zero except the centre).
    #[test]
    fn prototype_is_unity_gain_half_band() {
        let (atten, transition) = light();
        let length = half_band_len(atten, transition);
        let prototype = design_half_band(length, kaiser_beta(atten));
        let sum: f32 = prototype.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "DC gain {sum} != 1");

        let center = (length - 1) / 2;
        assert!((prototype[center] - 0.5).abs() < 1e-6, "centre tap != 0.5");
        for (index, &tap) in prototype.iter().enumerate() {
            let offset = index as isize - center as isize;
            if offset != 0 && offset % 2 == 0 {
                assert_eq!(tap, 0.0, "even-offset tap {index} not zero");
            }
        }
    }

    /// factor = 1 is a transparent bypass.
    #[test]
    fn factor_one_is_bypass() {
        let mut os = Oversampler::new(1, 64);
        let input: Vec<f32> = (0..64).map(|n| (n as f32 * 0.1).sin()).collect();
        let mut up = vec![0.0f32; 64];
        os.upsample(&input, &mut up);
        assert_eq!(up, input);
        let mut down = vec![0.0f32; 64];
        os.downsample(&up, &mut down);
        assert_eq!(down, input);
    }

    /// `with_design` clamps odd factors down to a power of two.
    #[test]
    fn factor_is_normalised_to_power_of_two() {
        assert_eq!(Oversampler::new(3, 16).factor(), 2);
        assert_eq!(Oversampler::new(5, 16).factor(), 4);
        assert_eq!(Oversampler::new(100, 16).factor(), 16);
    }

    /// A constant (DC) input must survive an up→down round trip at unity gain
    /// once the filters have settled, for every factor.
    #[test]
    fn dc_round_trip_is_unity() {
        let (atten, transition) = light();
        for factor in [2usize, 4, 8, 16] {
            let block = 256;
            let mut os = Oversampler::with_design(factor, block, atten, transition);
            let input = vec![0.75f32; block];
            let mut up = vec![0.0f32; block * factor];
            let mut down = vec![0.0f32; block];
            // Run several blocks so the delay lines fill with the DC value.
            for _ in 0..8 {
                os.upsample(&input, &mut up);
                os.downsample(&up, &mut down);
            }
            // The tail of the block is fully settled.
            let settled = &down[block - 16..];
            for &sample in settled {
                assert!(
                    (sample - 0.75).abs() < 1e-3,
                    "factor {factor}: DC round-trip {sample} != 0.75"
                );
            }
        }
    }

    /// The upsampler must preserve the amplitude of an in-band tone (its even
    /// output phase is a pure delayed copy of the input).
    #[test]
    fn upsample_even_phase_is_delayed_input() {
        let (atten, transition) = light();
        let block = 512;
        let mut os = Oversampler::with_design(2, block, atten, transition);
        let freq_frac = 0.02; // well inside the pass-band
        let input: Vec<f32> = (0..block)
            .map(|n| (TAU * freq_frac * n as f32).sin())
            .collect();
        let mut up = vec![0.0f32; block * 2];
        os.upsample(&input, &mut up);
        // Even samples of the output should equal a delayed copy of the input.
        // Find the integer delay by best cross-correlation over a small range.
        let even: Vec<f32> = up.iter().step_by(2).copied().collect();
        let mut best_error = f32::INFINITY;
        for delay in 0..32usize {
            let mut error = 0.0f32;
            let mut count = 0.0f32;
            for n in delay + 8..block - 8 {
                let diff = even[n] - input[n - delay];
                error += diff * diff;
                count += 1.0;
            }
            best_error = best_error.min((error / count).sqrt());
        }
        assert!(
            best_error < 0.02,
            "even phase not a clean delay: RMS {best_error}"
        );
    }

    /// A low-frequency sine must round-trip (up→down) with low in-band error
    /// after aligning for the cascade's group delay.
    #[test]
    fn sine_round_trip_low_error() {
        let (atten, transition) = light();
        let block = 1024;
        let factor = 4;
        let mut os = Oversampler::with_design(factor, block, atten, transition);
        // Kept low: the residual is dominated by *integer*-delay alignment (the
        // downsampled output is at the input rate, so the cascade's fractional
        // group delay cannot be aligned out), and that residual scales with
        // frequency as ≈ 2·sin(π·f·τ_frac).
        let freq_frac = 0.004;
        let input: Vec<f32> = (0..block)
            .map(|n| 0.5 * (TAU * freq_frac * n as f32).sin())
            .collect();
        let mut up = vec![0.0f32; block * factor];
        let mut down = vec![0.0f32; block];
        // Prime the delay lines, then measure a fresh block.
        for _ in 0..2 {
            os.upsample(&input, &mut up);
            os.downsample(&up, &mut down);
        }
        // Align by best integer delay and compute the residual RMS ratio.
        let mut best_ratio = f32::INFINITY;
        for delay in 0..64usize {
            let mut error = 0.0f64;
            let mut signal = 0.0f64;
            for n in delay + 16..block - 16 {
                let diff = down[n] - input[n - delay];
                error += (diff * diff) as f64;
                signal += (input[n - delay] * input[n - delay]) as f64;
            }
            if signal > 0.0 {
                best_ratio = best_ratio.min((error / signal).sqrt() as f32);
            }
        }
        // Bound reflects the integer-delay-alignment residual described above,
        // not the (far lower) stop-band error of the filter itself.
        assert!(
            best_ratio < 0.01,
            "round-trip error too high: ratio {best_ratio}"
        );
    }

    /// `reset` clears the delay lines so a subsequent block starts clean.
    #[test]
    fn reset_clears_state() {
        let mut os = Oversampler::new(2, 32);
        let input = vec![1.0f32; 32];
        let mut up = vec![0.0f32; 64];
        os.upsample(&input, &mut up);
        os.reset();
        // After reset, upsampling silence must produce silence.
        let silence = vec![0.0f32; 32];
        os.upsample(&silence, &mut up);
        assert!(up.iter().all(|&s| s.abs() < 1e-9), "state not cleared");
    }
}
