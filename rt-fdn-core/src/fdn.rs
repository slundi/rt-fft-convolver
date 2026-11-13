//! The `N`-branch Feedback Delay Network reverb engine.
//!
//! # Signal flow (per sample)
//!
//! ```text
//!   in_l ─▶ diffuser_l ─┐                              ┌─▶ Σ taps ─▶ out_l
//!   in_r ─▶ diffuser_r ─┤ inject                       │
//!                       ▼                               │
//!         ┌── delay 0 ─▶ high-cut ─▶ low-cut ──▶ s[0] ──┤
//!         │   delay 1 ─▶ high-cut ─▶ low-cut ──▶ s[1] ──┤
//!         │     …                                  …    │
//!         │   delay N ─▶ high-cut ─▶ low-cut ──▶ s[N] ──┘
//!         │                     │
//!         │            Hadamard mix (lossless)
//!         │                     │
//!         └──── × feedback gᵢ ◀─┘   (gᵢ sets T60 per branch)
//! ```
//!
//! 1. **Input diffusion** — each channel passes through a Schroeder all-pass
//!    chain so transients arrive dense, not as discrete echoes.
//! 2. **Read + tone** — every branch reads its (modulated, fractional) delay
//!    output and runs it through the per-branch high-cut (damping) and low-cut.
//! 3. **Output taps** — the filtered branch outputs are summed into L / R with
//!    two decorrelated ± sign patterns to make a wide stereo image.
//! 4. **Feedback** — the branch outputs are mixed by the lossless
//!    [`hadamard_in_place`] matrix, scaled by the per-branch decay gain, have
//!    the diffused input injected, and are written back into the delays.
//!
//! # Why per-branch feedback gains
//!
//! Decay time (T60) is set by how much a signal loses per trip round a branch.
//! A branch of length `Mᵢ` samples is traversed `fs/Mᵢ` times per second, so for
//! a target T60 the per-trip gain is
//!
//! ```text
//! gᵢ = 10^( −3 · (Mᵢ / fs) / T60 )
//! ```
//!
//! (−3 because −60 dB is `10^-3` in amplitude… ×20, i.e. `10^(−60/20)` split
//! across the trips: `g^(T60·fs/Mᵢ) = 10^(−3)`).  Giving every branch its own
//! `gᵢ` makes long and short delays decay *at the same rate*, which is what
//! stops a lumpy, uneven tail.  Because the matrix is lossless, `gᵢ < 1`
//! guarantees the whole network is stable.
//!
//! # Real-time safety
//!
//! [`Fdn::process_sample_stereo`] allocates nothing, locks nothing, never
//! panics in release, and flushes feedback writes to zero.  All buffers are
//! sized at construction.  Setters only recompute scalar coefficients.

use crate::allpass::AllPassChain;
use crate::delay::ModulatedDelay;
use crate::filter::{OnePoleHighpass, OnePoleLowpass};
use crate::lfo::SineLfo;
use crate::matrix::hadamard_in_place;
use crate::smoother::Smoother;

/// Reference sample rate the nominal delay lengths are quoted at.  Lengths are
/// scaled by `sample_rate / REFERENCE_SR` at construction so a preset sounds the
/// same at any host rate.
pub const REFERENCE_SR: f32 = 48_000.0;

/// Schroeder input-diffuser delay lengths (samples @ [`REFERENCE_SR`]), left
/// channel.  Mutually prime, short, and unequal for a dense, colourless smear.
const DIFFUSER_DELAYS_L: [usize; 4] = [142, 107, 379, 277];
/// Right-channel diffuser delays — deliberately different from the left so the
/// two channels decorrelate from the very first millisecond.
const DIFFUSER_DELAYS_R: [usize; 4] = [149, 113, 383, 281];

/// Smoothing time for the `size` glide (delay-length sweeps need to be slow to
/// avoid audible pitch zips).
const SIZE_SMOOTH_MS: f32 = 100.0;
/// Smoothing time for modulation-depth changes.
const DEPTH_SMOOTH_MS: f32 = 50.0;

/// Clamp bounds for the decay time in seconds (matches the plugin `Decay`
/// range, Phase 2).
const MIN_DECAY_S: f32 = 0.1;
const MAX_DECAY_S: f32 = 20.0;

/// Configuration for an `N`-branch FDN.  Delay lengths are given at
/// [`REFERENCE_SR`]; everything else is in natural units.
#[derive(Debug, Clone, Copy)]
pub struct FdnParams<const N: usize> {
    /// Per-branch nominal delay lengths in samples @ [`REFERENCE_SR`].  Should
    /// be mutually prime to maximise modal density (Phase 0 decision).
    pub nominal_delays: [f32; N],
    /// Reverb decay time (T60) in seconds.
    pub decay_s: f32,
    /// Per-branch high-cut / damping cutoff in Hz.
    pub damping_hz: f32,
    /// Per-branch low-cut cutoff in Hz.
    pub lowcut_hz: f32,
    /// Peak delay-modulation depth in samples @ [`REFERENCE_SR`].
    pub mod_depth: f32,
    /// Modulation LFO rate in Hz.
    pub mod_rate_hz: f32,
    /// Input-diffusion all-pass coefficient in `[0, 0.9]`.
    pub diffusion: f32,
    /// Size scale in `[0, 1]` (`1.0` = the nominal delay lengths).
    pub size: f32,
}

/// An `N`-branch Feedback Delay Network reverb.  Use `N = 8` (the Phase 0
/// decision); the code is generic over any power-of-two `N`.
#[derive(Debug)]
pub struct Fdn<const N: usize> {
    sample_rate: f32,
    /// Nominal delay lengths in samples **at `sample_rate`** (already scaled
    /// from [`REFERENCE_SR`]).  `size` multiplies these.
    nominal_delays: [f32; N],
    /// Per-branch modulated delay lines.
    delays: [ModulatedDelay; N],
    /// Per-branch high-cut / damping low-pass.
    dampers: [OnePoleLowpass; N],
    /// Per-branch low-cut high-pass.
    lowcuts: [OnePoleHighpass; N],
    /// Per-branch feedback gain, derived from `decay_s`, `size` and length.
    feedback: [f32; N],
    /// Input diffusers, one chain per channel.
    diffuser_l: AllPassChain,
    diffuser_r: AllPassChain,
    /// Target decay time (T60) in seconds.
    decay_s: f32,
    /// Peak modulation depth in samples at `sample_rate`.
    mod_depth_max: f32,
    /// Smoothed `size` scale.
    size: Smoother,
    /// Smoothed modulation depth (samples).
    depth: Smoother,
    /// When `true` the per-branch feedback gains are pinned to `1.0` (infinite
    /// sustain).  Because the mixing matrix is lossless, unity feedback holds
    /// the tail energy constant — the "freeze" control.  Setters that would
    /// recompute the decay-derived gains are inert while frozen.
    frozen: bool,
}

impl<const N: usize> Fdn<N> {
    /// Build an FDN at `sample_rate` from `params`.
    ///
    /// **Allocates** — construct off the render thread.
    pub fn new(sample_rate: f32, params: FdnParams<N>) -> Self {
        assert!(
            N.is_power_of_two() && N > 0,
            "FDN size must be a power of two"
        );
        let sample_rate = sample_rate.max(1.0);
        let scale = sample_rate / REFERENCE_SR;

        let nominal_delays: [f32; N] = core::array::from_fn(|i| params.nominal_delays[i] * scale);
        let mod_depth_max = (params.mod_depth * scale).max(0.0);

        let delays: [ModulatedDelay; N] = core::array::from_fn(|i| {
            let base = nominal_delays[i];
            // Buffer must hold the longest possible read: base (at size = 1.0)
            // plus the full modulation swing, plus interpolation headroom.
            let capacity = (base + mod_depth_max).ceil() as usize + 4;
            // Spread the N start phases evenly over a full turn so the branches
            // modulate out of step (decorrelated motion).
            let phase = i as f32 / N as f32;
            let lfo = SineLfo::new(params.mod_rate_hz, sample_rate, phase);
            ModulatedDelay::new(capacity, base, mod_depth_max, lfo)
        });

        let dampers: [OnePoleLowpass; N] =
            core::array::from_fn(|_| OnePoleLowpass::new(params.damping_hz, sample_rate));
        let lowcuts: [OnePoleHighpass; N] =
            core::array::from_fn(|_| OnePoleHighpass::new(params.lowcut_hz, sample_rate));

        // Scale the diffuser delays to the host rate as well.
        let scaled_l: Vec<usize> = DIFFUSER_DELAYS_L
            .iter()
            .map(|&d| ((d as f32 * scale).round() as usize).max(1))
            .collect();
        let scaled_r: Vec<usize> = DIFFUSER_DELAYS_R
            .iter()
            .map(|&d| ((d as f32 * scale).round() as usize).max(1))
            .collect();
        let diffuser_l = AllPassChain::new(&scaled_l, params.diffusion);
        let diffuser_r = AllPassChain::new(&scaled_r, params.diffusion);

        let mut size = Smoother::with_time(params.size, SIZE_SMOOTH_MS, sample_rate);
        size.set_immediate(params.size.clamp(0.0, 1.0));
        let mut depth = Smoother::with_time(mod_depth_max, DEPTH_SMOOTH_MS, sample_rate);
        depth.set_immediate(mod_depth_max);

        let mut fdn = Self {
            sample_rate,
            nominal_delays,
            delays,
            dampers,
            lowcuts,
            feedback: [0.0; N],
            diffuser_l,
            diffuser_r,
            decay_s: params.decay_s.clamp(MIN_DECAY_S, MAX_DECAY_S),
            mod_depth_max,
            size,
            depth,
            frozen: false,
        };
        fdn.recompute_feedback();
        fdn
    }

    /// Recompute the per-branch feedback gains from the current `decay_s` and
    /// the *target* `size` (called only from setters, never per sample — it
    /// runs `N` `powf`s).
    fn recompute_feedback(&mut self) {
        // While frozen the gains are pinned to unity; do not let a decay/size
        // change clobber the sustain.
        if self.frozen {
            return;
        }
        let size = self.size.target().clamp(0.0, 1.0);
        let sample_rate = self.sample_rate;
        let decay_s = self.decay_s;
        for (gain, &nominal) in self.feedback.iter_mut().zip(self.nominal_delays.iter()) {
            let delay_samples = (nominal * size).max(1.0);
            let delay_seconds = delay_samples / sample_rate;
            // gᵢ = 10^(−3 · delaySeconds / T60); capped just below 1 for stability.
            *gain = 10.0_f32.powf(-3.0 * delay_seconds / decay_s).min(0.9999);
        }
    }

    /// Set the reverb decay time (T60) in seconds, clamped to `0.1..=20`.
    ///
    /// **Real-time safe** (recomputes `N` feedback gains, no allocation).
    pub fn set_decay(&mut self, decay_s: f32) {
        self.decay_s = decay_s.clamp(MIN_DECAY_S, MAX_DECAY_S);
        self.recompute_feedback();
    }

    /// Set the size scale in `[0, 1]`; the delay lengths glide toward the new
    /// value to avoid pitch zips.
    ///
    /// **Real-time safe.**
    pub fn set_size(&mut self, size: f32) {
        self.size.set_target(size.clamp(0.0, 1.0));
        // Decay depends on the (target) length; keep the two consistent.
        self.recompute_feedback();
    }

    /// Set the per-branch high-cut (damping) cutoff in Hz.
    ///
    /// **Real-time safe.**
    pub fn set_damping(&mut self, cutoff_hz: f32) {
        for damper in &mut self.dampers {
            damper.set_cutoff(cutoff_hz, self.sample_rate);
        }
    }

    /// Set the per-branch low-cut cutoff in Hz.
    ///
    /// **Real-time safe.**
    pub fn set_lowcut(&mut self, cutoff_hz: f32) {
        for lowcut in &mut self.lowcuts {
            lowcut.set_cutoff(cutoff_hz, self.sample_rate);
        }
    }

    /// Set the modulation depth (samples @ [`REFERENCE_SR`], scaled internally)
    /// and rate (Hz).  Depth glides; rate changes immediately (phase-continuous).
    ///
    /// **Real-time safe.**
    pub fn set_modulation(&mut self, depth_samples: f32, rate_hz: f32) {
        let scale = self.sample_rate / REFERENCE_SR;
        let scaled = (depth_samples * scale).clamp(0.0, self.mod_depth_max);
        self.depth.set_target(scaled);
        for delay in &mut self.delays {
            // The LFO lives inside the delay; expose rate via a tiny shim below.
            delay.set_lfo_rate(rate_hz, self.sample_rate);
        }
    }

    /// Set the input-diffusion coefficient in `[0, 0.9]`.
    ///
    /// **Real-time safe.**
    pub fn set_diffusion(&mut self, gain: f32) {
        self.diffuser_l.set_gain(gain);
        self.diffuser_r.set_gain(gain);
    }

    /// Freeze (or thaw) the tail.
    ///
    /// Freezing pins every branch's feedback gain to `1.0`; the lossless
    /// mixing matrix then holds the circulating energy constant, so whatever is
    /// in the delay lines sustains indefinitely.  Thawing restores the
    /// decay-derived gains for the current `decay_s` / `size`.  The caller
    /// should stop injecting new input while frozen (the plugin mutes the FDN
    /// input) so the frozen tail is not overwritten.
    ///
    /// **Real-time safe** — sets `N` gains, no allocation.
    pub fn set_freeze(&mut self, freeze: bool) {
        if freeze == self.frozen {
            return;
        }
        self.frozen = freeze;
        if freeze {
            self.feedback = [1.0; N];
        } else {
            self.recompute_feedback();
        }
    }

    /// Whether the tail is currently frozen.
    #[inline]
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// Clear all delay lines, filters and diffusers to silence and rewind the
    /// LFOs so the next sample starts from a clean, decorrelated state.
    pub fn reset(&mut self) {
        for delay in &mut self.delays {
            delay.reset();
        }
        for damper in &mut self.dampers {
            damper.reset();
        }
        for lowcut in &mut self.lowcuts {
            lowcut.reset();
        }
        self.diffuser_l.reset();
        self.diffuser_r.reset();
        // Snap the smoothers so a reset does not glide from a stale value.
        self.size.set_immediate(self.size.target());
        self.depth.set_immediate(self.depth.target());
    }

    /// The left-channel output-tap / input-injection sign for branch `i`.
    #[inline]
    fn sign_l(i: usize) -> f32 {
        if i & 1 == 0 { 1.0 } else { -1.0 }
    }

    /// The right-channel sign for branch `i` — a different pattern from
    /// [`sign_l`](Self::sign_l) so the two channels decorrelate.
    #[inline]
    fn sign_r(i: usize) -> f32 {
        if (i >> 1) & 1 == 0 { 1.0 } else { -1.0 }
    }

    /// Process one stereo sample, returning the **wet** `(left, right)` pair.
    ///
    /// The dry signal is not mixed here — that is the plugin's job (Phase 2),
    /// keeping the core a pure wet generator.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn process_sample_stereo(&mut self, in_l: f32, in_r: f32) -> (f32, f32) {
        // Advance the smoothed controls and push them into the branches.
        let size = self.size.tick();
        let depth = self.depth.tick();
        for (delay, &nominal) in self.delays.iter_mut().zip(self.nominal_delays.iter()) {
            delay.set_base_delay(nominal * size);
            delay.set_mod_depth(depth);
        }

        // 1. Input diffusion per channel.
        let diffused_l = self.diffuser_l.process(in_l);
        let diffused_r = self.diffuser_r.process(in_r);

        // 2. Read every branch and apply the in-loop tone filters.  While
        //    frozen the branch circulates *unfiltered*: the damping / low-cut
        //    would bleed energy on every pass, so at unity feedback the tail
        //    would still decay.  Bypassing them makes the orthonormal matrix
        //    preserve energy exactly, sustaining the (already-coloured) tail.
        let frozen = self.frozen;
        let mut branch = [0.0f32; N];
        for (i, slot) in branch.iter_mut().enumerate() {
            *slot = if frozen {
                // Lossless integer-tap read so the sustained tail neither decays
                // (interpolation loss) nor is coloured (in-loop filters).
                self.delays[i].read_frozen()
            } else {
                let raw = self.delays[i].read();
                let damped = self.dampers[i].process(raw);
                self.lowcuts[i].process(damped)
            };
        }

        // 3. Output taps (decorrelated ± patterns), normalised by 1/√N.
        let out_norm = 1.0 / (N as f32).sqrt();
        let mut out_l = 0.0f32;
        let mut out_r = 0.0f32;
        for (i, &sample) in branch.iter().enumerate() {
            out_l += sample * Self::sign_l(i);
            out_r += sample * Self::sign_r(i);
        }
        out_l *= out_norm;
        out_r *= out_norm;

        // 4. Lossless feedback mix, then per-branch decay + input injection.
        let mut mixed = branch;
        hadamard_in_place(&mut mixed);
        for (i, delay) in self.delays.iter_mut().enumerate() {
            // Frozen: inject nothing so the sustained tail is not overwritten.
            let injected = if frozen {
                0.0
            } else {
                diffused_l * Self::sign_l(i) + diffused_r * Self::sign_r(i)
            };
            let feedback = mixed[i] * self.feedback[i];
            delay.write(injected + feedback);
        }

        (out_l, out_r)
    }
}
