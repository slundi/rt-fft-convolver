//! Click-free IR switching via equal-power cross-fading.
//!
//! # Why cross-fade?
//!
//! Swapping the IR of a running [`UniformPartitionEngine`] mid-stream
//! creates two audible artefacts:
//!
//! 1. **Transfer-function step** — the input at time `t` is suddenly
//!    convolved against a different set of coefficients.  The output
//!    jumps, producing a click.
//! 2. **Stale-tail leak** — the frequency-domain delay line still holds
//!    input spectra from before the switch.  Those are now multiplied
//!    against the *new* IR partitions, producing several blocks of noise
//!    that don't correspond to any real acoustic response.
//!
//! Both artefacts vanish if we run **two engines in parallel** for a
//! short window and blend their outputs from one to the other.  This
//! module implements that pattern with an **equal-power** blend so the
//! perceived loudness stays constant across the fade:
//!
//! ```text
//! α           = fade_pos / (crossfade_samples − 1)   ∈ [0, 1]
//! θ           = α × π/2
//! front_gain  = cos(θ)   ← 1 → 0 across the fade
//! back_gain   = sin(θ)   ← 0 → 1 across the fade
//! output[n]   = front_gain · old_ir_out[n] + back_gain · new_ir_out[n]
//! ```
//!
//! `front_gain² + back_gain² = 1` for every sample — matches the invariant
//! used by [`Mixer`] and keeps power flat when the two cabinets are
//! uncorrelated.
//!
//! # Roles
//!
//! At any moment one engine is `front` (currently the source of audio)
//! and the other is `back` (idle, or the incoming side of an active
//! fade).  When a fade completes the two slots are swapped so `front`
//! always holds the "currently audible" IR.
//!
//! ```text
//! before set_ir(B):    front=A (audible)   back=idle
//! during fade:         front=A (fading out) back=B (fading in)
//! after fade:          front=B (audible)   back=A (idle, retained)
//! ```
//!
//! # CPU note
//!
//! While a fade is active both engines process the same input, so CPU
//! cost temporarily doubles.  The fade window is typically short
//! (10-100 ms) and the extra work is amortised by the click-free result.
//!
//! # Warm-up caveat
//!
//! When a fade starts `back`'s frequency-domain delay line is empty.  It
//! only sees input samples from the fade start onward, so for the first
//! `L` samples (`L = new IR length`) its output is a *fresh* response —
//! it lacks the reverb tail of samples that occurred before the switch.
//! In practice this is imperceptible for typical guitar cabinets (short
//! IRs, continuous signal) because the equal-power blend keeps `front`
//! dominant early in the fade.
//!
//! [`Mixer`]: crate::engine::mixer::Mixer
//! [`UniformPartitionEngine`]: crate::engine::partition::UniformPartitionEngine

use std::f32::consts::FRAC_PI_2;

use crate::engine::partition::UniformPartitionEngine;

/// Click-free IR switcher wrapping two [`UniformPartitionEngine`] slots.
///
/// The `front` slot is currently audible; the `back` slot is idle or the
/// incoming side of an active fade.  Trigger a fade with [`set_ir`] (which
/// allocates) or [`swap_engine`] (which is real-time safe).
///
/// [`set_ir`]: Self::set_ir
/// [`swap_engine`]: Self::swap_engine
/// [`UniformPartitionEngine`]: crate::engine::partition::UniformPartitionEngine
pub struct CrossfadingConvolver {
    /// Currently audible engine.  A completed fade swaps this with `back`.
    front: UniformPartitionEngine,
    /// Idle engine, or the incoming side of an active fade.
    back: UniformPartitionEngine,

    /// Audio block size — both engines share it.
    block_size: usize,

    /// Number of interpolated samples in a full fade.
    /// `0` means "instant switch" (no interpolation).
    crossfade_samples: usize,

    /// Progress through the current fade in samples.
    /// `None` = no fade in progress; `Some(k)` = fade at position `k`.
    fade_pos: Option<usize>,

    /// Pre-computed equal-power gain envelopes.
    /// Length = `crossfade_samples`; `front_gains[0] ≈ 1`, `[last] ≈ 0`.
    front_gains: Vec<f32>,
    /// `back_gains[0] ≈ 0`, `[last] ≈ 1`.
    back_gains: Vec<f32>,

    /// Scratch buffers so process() stays allocation-free even during a fade.
    front_out: Vec<f32>,
    back_out: Vec<f32>,
}

impl CrossfadingConvolver {
    /// Create a crossfader loaded with an initial IR.
    ///
    /// - `ir` — initial IR loaded into `front`; `back` starts idle.
    /// - `block_size` — audio render block size, must be > 0.
    /// - `crossfade_samples` — duration of a full fade in samples.  `0`
    ///   makes [`set_ir`] and [`swap_engine`] switch instantly (may click).
    ///
    /// **Allocates** — call from a non-real-time thread.
    ///
    /// # Panics
    ///
    /// Panics if `block_size == 0`.
    ///
    /// [`set_ir`]: Self::set_ir
    /// [`swap_engine`]: Self::swap_engine
    pub fn new(ir: &[f32], block_size: usize, crossfade_samples: usize) -> Self {
        assert!(block_size > 0, "block_size must be > 0");

        // Build the equal-power fade envelope once at construction so
        // process() can look up gains with a simple index.
        let (front_gains, back_gains) = build_gain_tables(crossfade_samples);

        Self {
            front: UniformPartitionEngine::new(ir, block_size),
            // `back` starts idle (empty IR is a no-op engine).
            back: UniformPartitionEngine::new(&[], block_size),
            block_size,
            crossfade_samples,
            fade_pos: None,
            front_gains,
            back_gains,
            front_out: vec![0.0f32; block_size],
            back_out: vec![0.0f32; block_size],
        }
    }

    /// Audio block size both internal engines share.
    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Number of samples in a full fade (`0` = instant switch).
    #[inline]
    pub fn crossfade_samples(&self) -> usize {
        self.crossfade_samples
    }

    /// Whether a fade is currently in progress.
    #[inline]
    pub fn is_crossfading(&self) -> bool {
        self.fade_pos.is_some()
    }

    /// Current position within the fade, `None` if no fade is active.
    #[inline]
    pub fn crossfade_progress(&self) -> Option<usize> {
        self.fade_pos
    }

    /// Start a fade to a new IR.
    ///
    /// Loads `new_ir` into the back slot and kicks off a fade from the
    /// currently-audible IR to the new one.  Any fade already in flight
    /// is collapsed to its completion first (may produce a small
    /// transient — real-world UIs usually debounce rapid switches).
    ///
    /// **Allocates** — call from a non-real-time thread.
    pub fn set_ir(&mut self, new_ir: &[f32]) {
        // Collapse any in-flight fade so the *currently* front-dominant
        // engine really is `front`.  Without this, chaining set_ir mid-fade
        // would treat the almost-faded-out old front as the outgoing side.
        self.finish_pending_fade();

        // load_ir may allocate; caller acknowledged by not using swap_engine.
        self.back.load_ir(new_ir);
        self.begin_fade();
    }

    /// Swap in a pre-built engine (moved by value), triggering a fade
    /// from the currently-audible IR to the new one.
    ///
    /// The engine that was displaced from the back slot is returned so
    /// the caller can drop it, pool it, or send it back to a builder
    /// thread for reuse.
    ///
    /// **Real-time safe** — no allocation.
    ///
    /// # Panics
    ///
    /// Panics if `new_engine.block_size() != self.block_size()`.
    pub fn swap_engine(&mut self, new_engine: UniformPartitionEngine) -> UniformPartitionEngine {
        assert_eq!(
            new_engine.block_size(),
            self.block_size,
            "swap_engine: block_size mismatch"
        );

        self.finish_pending_fade();
        let displaced = std::mem::replace(&mut self.back, new_engine);
        self.begin_fade();
        displaced
    }

    /// Clear both engines' delay lines and abort any active fade.
    ///
    /// **Real-time safe** — no allocation.
    pub fn reset(&mut self) {
        self.front.reset();
        self.back.reset();
        self.fade_pos = None;
    }

    /// Process one block: convolve `input` and, if a fade is active,
    /// blend the two engines' outputs into `output`.
    ///
    /// **Overwrites** `output` (unlike [`UniformPartitionEngine::process`],
    /// which accumulates).  This matches the "final render stage" role a
    /// crossfader typically plays.
    ///
    /// **Real-time safe** — no allocation.
    ///
    /// [`UniformPartitionEngine::process`]: crate::engine::partition::UniformPartitionEngine::process
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.block_size, "input length mismatch");
        debug_assert_eq!(output.len(), self.block_size, "output length mismatch");

        match self.fade_pos {
            None => {
                // No fade — render front only, overwrite output.
                output.fill(0.0);
                self.front.process(input, output);
            }
            Some(mut pos) => {
                // Fade in progress — render both engines in parallel.
                self.front_out.fill(0.0);
                self.back_out.fill(0.0);
                self.front.process(input, &mut self.front_out);
                self.back.process(input, &mut self.back_out);

                let crossfade_samples = self.crossfade_samples;
                for (i, y) in output.iter_mut().enumerate() {
                    if pos < crossfade_samples {
                        // Look up pre-computed equal-power gains.
                        let f_g = self.front_gains[pos];
                        let b_g = self.back_gains[pos];
                        *y = f_g * self.front_out[i] + b_g * self.back_out[i];
                        pos += 1;
                    } else {
                        // Fade completed mid-block: the remaining samples
                        // of this block belong to the (soon-to-be-swapped)
                        // back engine at unity gain.
                        *y = self.back_out[i];
                    }
                }

                if pos >= crossfade_samples {
                    // Fade finished within this block — swap roles so
                    // `front` holds the new IR from the next call onward.
                    std::mem::swap(&mut self.front, &mut self.back);
                    self.fade_pos = None;
                } else {
                    self.fade_pos = Some(pos);
                }
            }
        }
    }

    // ─── Internal helpers ─────────────────────────────────────────────────

    /// Kick off a fresh fade from `front` to whatever is currently in `back`.
    ///
    /// If `crossfade_samples == 0`, performs an instant swap instead.
    fn begin_fade(&mut self) {
        if self.crossfade_samples == 0 {
            std::mem::swap(&mut self.front, &mut self.back);
            self.fade_pos = None;
        } else {
            self.fade_pos = Some(0);
        }
    }

    /// If a fade is in progress, collapse it to "done": swap the engines
    /// and clear the fade state.  Used before starting a new fade so
    /// chained switches never treat the outgoing side as still being
    /// front-dominant.
    fn finish_pending_fade(&mut self) {
        if self.fade_pos.is_some() {
            std::mem::swap(&mut self.front, &mut self.back);
            self.fade_pos = None;
        }
    }
}

/// Build the equal-power fade envelope of length `crossfade_samples`.
///
/// Uses the endpoints `α = 0` and `α = 1` exactly so the transition
/// from "last fade sample" to "post-fade unity gain on the swapped
/// front" is seamless (no residual step).
///
/// Degenerate cases:
/// - `crossfade_samples == 0` → returns empty vectors (no interpolation).
/// - `crossfade_samples == 1` → returns `([0.0], [1.0])` (one-sample switch).
fn build_gain_tables(crossfade_samples: usize) -> (Vec<f32>, Vec<f32>) {
    if crossfade_samples == 0 {
        return (Vec::new(), Vec::new());
    }
    if crossfade_samples == 1 {
        return (vec![0.0], vec![1.0]);
    }

    let denom = (crossfade_samples - 1) as f32;
    let mut front_gains = Vec::with_capacity(crossfade_samples);
    let mut back_gains = Vec::with_capacity(crossfade_samples);
    for i in 0..crossfade_samples {
        let alpha = i as f32 / denom;
        let theta = alpha * FRAC_PI_2;
        front_gains.push(theta.cos());
        back_gains.push(theta.sin());
    }
    (front_gains, back_gains)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    /// Unit-impulse IR of exactly `block_size` samples: [gain, 0, 0, …].
    fn scaled_impulse(gain: f32, block_size: usize) -> Vec<f32> {
        let mut ir = vec![0.0f32; block_size];
        ir[0] = gain;
        ir
    }

    fn assert_close(label: &str, got: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
        for (i, (&value, &target)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (value - target).abs() < tol,
                "{label}[{i}]: expected {target}, got {value}"
            );
        }
    }

    // ── Gain-table generation ────────────────────────────────────────────────

    /// Empty and one-sample fades take the documented shortcut branches.
    #[test]
    fn gain_tables_handle_degenerate_lengths() {
        let (f, b) = build_gain_tables(0);
        assert!(f.is_empty() && b.is_empty());

        let (f, b) = build_gain_tables(1);
        assert_eq!(f, vec![0.0f32]);
        assert_eq!(b, vec![1.0f32]);
    }

    /// For a general fade the gains hit exact endpoints and satisfy the
    /// equal-power invariant `f² + b² = 1`.
    #[test]
    fn gain_tables_are_equal_power_and_hit_endpoints() {
        let n = 8;
        let (f_g, b_g) = build_gain_tables(n);
        assert_eq!(f_g.len(), n);
        assert_eq!(b_g.len(), n);
        // Endpoints exactly.
        assert!((f_g[0] - 1.0).abs() < EPSILON, "f[0]={}", f_g[0]);
        assert!(b_g[0].abs() < EPSILON, "b[0]={}", b_g[0]);
        assert!(f_g[n - 1].abs() < EPSILON, "f[last]={}", f_g[n - 1]);
        assert!((b_g[n - 1] - 1.0).abs() < EPSILON, "b[last]={}", b_g[n - 1]);
        // Equal-power invariant.
        for i in 0..n {
            let power = f_g[i] * f_g[i] + b_g[i] * b_g[i];
            assert!(
                (power - 1.0).abs() < EPSILON,
                "equal-power violated at {i}: {power}"
            );
        }
    }

    // ── Construction & basic behaviour ───────────────────────────────────────

    /// A freshly-created crossfader is not fading and produces the
    /// initial-IR output verbatim (unit impulse IR ⇒ identity).
    #[test]
    fn fresh_crossfader_is_not_fading() {
        let block_size = 4;
        let conv = CrossfadingConvolver::new(&scaled_impulse(1.0, block_size), block_size, 8);
        assert!(!conv.is_crossfading());
        assert!(conv.crossfade_progress().is_none());
        assert_eq!(conv.block_size(), block_size);
        assert_eq!(conv.crossfade_samples(), 8);
    }

    /// Without any set_ir call the front engine drives output alone.
    #[test]
    fn process_without_fade_uses_front_only() {
        let block_size = 4;
        // Front IR has gain 3.
        let mut conv = CrossfadingConvolver::new(&scaled_impulse(3.0, block_size), block_size, 8);
        let input = [1.0f32; 4];
        let mut output = [0.0f32; 4];
        conv.process(&input, &mut output);
        assert_close("no-fade output", &output, &[3.0, 3.0, 3.0, 3.0], EPSILON);
    }

    // ── set_ir triggers a fade ───────────────────────────────────────────────

    /// After set_ir a fade is in progress with pos = 0.
    #[test]
    fn set_ir_starts_a_fade() {
        let block_size = 4;
        let mut conv = CrossfadingConvolver::new(&scaled_impulse(1.0, block_size), block_size, 8);
        conv.set_ir(&scaled_impulse(2.0, block_size));
        assert!(conv.is_crossfading());
        assert_eq!(conv.crossfade_progress(), Some(0));
    }

    /// crossfade_samples = 0 makes set_ir switch instantly (no fade state).
    #[test]
    fn zero_length_fade_switches_instantly() {
        let block_size = 4;
        let mut conv = CrossfadingConvolver::new(&scaled_impulse(1.0, block_size), block_size, 0);
        conv.set_ir(&scaled_impulse(5.0, block_size));
        assert!(!conv.is_crossfading());
        // The new IR must be immediately audible.
        let input = [1.0f32; 4];
        let mut output = [0.0f32; 4];
        conv.process(&input, &mut output);
        assert_close("instant switch", &output, &[5.0f32; 4], EPSILON);
    }

    // ── Fade shape (equal power, correct endpoints) ──────────────────────────

    /// At the start of the fade output ≈ old IR; at the end ≈ new IR.
    /// Uses two scalar-gain IRs so the response is a clean multiple of the input.
    #[test]
    fn fade_transitions_from_old_to_new_ir() {
        let block_size = 4;
        let crossfade = 4; // one block long, easy to reason about
        let mut conv =
            CrossfadingConvolver::new(&scaled_impulse(1.0, block_size), block_size, crossfade);

        // Switch to a gain-3 IR — expect output to slide from 1×input to 3×input.
        conv.set_ir(&scaled_impulse(3.0, block_size));

        let input = [1.0f32; 4];
        let mut output = [0.0f32; 4];
        conv.process(&input, &mut output);

        // Endpoints must land exactly on the source and destination IRs.
        // - sample 0: alpha=0 ⇒ 1 * 1.0 + 0 * 3.0 = 1.0
        // - sample crossfade-1: alpha=1 ⇒ 0 * 1.0 + 1 * 3.0 = 3.0
        assert!(
            (output[0] - 1.0).abs() < EPSILON,
            "fade start: expected 1.0, got {}",
            output[0]
        );
        assert!(
            (output[3] - 3.0).abs() < EPSILON,
            "fade end: expected 3.0, got {}",
            output[3]
        );
        // Intermediate samples must stay strictly between the two IR
        // amplitudes — equal-power fades of two correlated in-phase signals
        // can overshoot the linear ramp so we don't check monotonicity.
        for (i, &sample) in output.iter().enumerate().take(3).skip(1) {
            assert!(
                sample > 1.0 && sample < 3.5,
                "fade midpoint out of bounds: output[{i}]={sample}"
            );
        }

        // Fade completed exactly at the end of the block.
        assert!(!conv.is_crossfading());
    }

    /// After the fade completes, the new IR alone drives future output.
    #[test]
    fn after_fade_new_ir_is_active() {
        let block_size = 4;
        let mut conv = CrossfadingConvolver::new(&scaled_impulse(1.0, block_size), block_size, 4);
        conv.set_ir(&scaled_impulse(2.0, block_size));

        // First block: consumed by the fade.
        let input = [1.0f32; 4];
        let mut output = [0.0f32; 4];
        conv.process(&input, &mut output);
        assert!(!conv.is_crossfading(), "fade should be done after 1 block");

        // Second block: pure new-IR output.
        conv.process(&input, &mut output);
        assert_close("post-fade output", &output, &[2.0f32; 4], EPSILON);
    }

    /// Fade spanning multiple blocks: progress advances correctly.
    #[test]
    fn multi_block_fade_progress() {
        let block_size = 4;
        let crossfade = 10;
        let mut conv =
            CrossfadingConvolver::new(&scaled_impulse(1.0, block_size), block_size, crossfade);
        conv.set_ir(&scaled_impulse(4.0, block_size));

        let input = [1.0f32; 4];
        let mut output = [0.0f32; 4];

        // Block 1: pos goes 0 → 4.
        conv.process(&input, &mut output);
        assert_eq!(conv.crossfade_progress(), Some(4));

        // Block 2: pos goes 4 → 8.
        conv.process(&input, &mut output);
        assert_eq!(conv.crossfade_progress(), Some(8));

        // Block 3: pos goes 8 → 10, fade completes; remaining 2 samples use back.
        conv.process(&input, &mut output);
        assert!(!conv.is_crossfading());
        // Last 2 samples must be the new-IR output at unity gain (4.0).
        assert!(
            (output[2] - 4.0).abs() < EPSILON,
            "post-fade sample: {}",
            output[2]
        );
        assert!(
            (output[3] - 4.0).abs() < EPSILON,
            "post-fade sample: {}",
            output[3]
        );
    }

    // ── swap_engine — RT-safe path ───────────────────────────────────────────

    /// swap_engine performs the same fade as set_ir but by moving a
    /// pre-built engine, and returns the displaced engine.
    #[test]
    fn swap_engine_starts_fade_and_returns_old_engine() {
        let block_size = 4;
        let mut conv = CrossfadingConvolver::new(&scaled_impulse(1.0, block_size), block_size, 4);

        // Build the new engine off-thread (simulated); hand it over.
        let new_engine = UniformPartitionEngine::new(&scaled_impulse(2.0, block_size), block_size);
        let displaced = conv.swap_engine(new_engine);

        // Fade starts as if set_ir had been called.
        assert!(conv.is_crossfading());
        // The displaced engine was previously the idle "back" slot — its
        // block_size must match, and it should be a valid engine.
        assert_eq!(displaced.block_size(), block_size);
    }

    /// swap_engine on block_size mismatch must panic.
    #[test]
    #[should_panic(expected = "block_size mismatch")]
    fn swap_engine_panics_on_block_size_mismatch() {
        let mut conv = CrossfadingConvolver::new(&scaled_impulse(1.0, 4), 4, 4);
        let bad = UniformPartitionEngine::new(&scaled_impulse(1.0, 8), 8);
        let _ = conv.swap_engine(bad);
    }

    // ── Chained switches ─────────────────────────────────────────────────────

    /// Calling set_ir during an active fade collapses the pending fade and
    /// starts a new one from the *incoming* IR toward the newest one.
    #[test]
    fn chained_set_ir_collapses_pending_fade() {
        let block_size = 4;
        let mut conv = CrossfadingConvolver::new(&scaled_impulse(1.0, block_size), block_size, 8);

        // Start fade A → B, but only run a few samples.
        conv.set_ir(&scaled_impulse(2.0, block_size));
        let input = [1.0f32; 4];
        let mut output = [0.0f32; 4];
        conv.process(&input, &mut output);
        // Fade still in progress (crossfade_samples=8, block_size=4).
        assert!(conv.is_crossfading());

        // Now request another switch mid-fade → the pending fade must be
        // collapsed and a new fade toward the newest IR begins.
        conv.set_ir(&scaled_impulse(4.0, block_size));
        // The new fade must be freshly started (position 0).
        assert_eq!(conv.crossfade_progress(), Some(0));
    }

    // ── reset ────────────────────────────────────────────────────────────────

    /// reset() cancels any active fade and clears both engines' state.
    #[test]
    fn reset_cancels_fade_and_clears_state() {
        let block_size = 4;
        let mut conv = CrossfadingConvolver::new(&scaled_impulse(1.0, block_size), block_size, 8);
        conv.set_ir(&scaled_impulse(3.0, block_size));
        assert!(conv.is_crossfading());

        conv.reset();
        assert!(!conv.is_crossfading());

        // After reset, the output is driven by the *original* IR (front is
        // the pre-fade engine because the pending fade was aborted).
        let input = [1.0f32; 4];
        let mut output = [0.0f32; 4];
        conv.process(&input, &mut output);
        assert_close("post-reset output", &output, &[1.0f32; 4], EPSILON);
    }

    // ── Output determinism ───────────────────────────────────────────────────

    /// process() overwrites (does not accumulate into) the output buffer.
    #[test]
    fn process_overwrites_output() {
        let block_size = 4;
        let mut conv = CrossfadingConvolver::new(&scaled_impulse(1.0, block_size), block_size, 8);
        let input = [1.0f32; 4];
        let mut output = [99.0f32; 4]; // non-zero sentinel
        conv.process(&input, &mut output);
        // Sentinels must be overwritten by the front engine's output.
        assert_close("overwrite", &output, &[1.0f32; 4], EPSILON);
    }

    // ── Fade-to-silence and fade-in ──────────────────────────────────────────

    /// Fading to an empty IR gently fades the audible signal to zero.
    #[test]
    fn fade_to_empty_ir_fades_to_silence() {
        let block_size = 4;
        let mut conv = CrossfadingConvolver::new(&scaled_impulse(1.0, block_size), block_size, 4);
        conv.set_ir(&[]);
        let input = [1.0f32; 4];
        let mut output = [0.0f32; 4];
        conv.process(&input, &mut output);
        // Endpoints: starts near 1.0 (front dominant), ends near 0.0 (back
        // is silent).
        assert!(
            (output[0] - 1.0).abs() < EPSILON,
            "fade-to-silence start: {}",
            output[0]
        );
        assert!(
            output[3].abs() < EPSILON,
            "fade-to-silence end: {}",
            output[3]
        );
    }

    /// Fading in from an empty starting IR gives a clean fade-in.
    #[test]
    fn fade_in_from_silence() {
        let block_size = 4;
        let mut conv = CrossfadingConvolver::new(&[], block_size, 4);
        conv.set_ir(&scaled_impulse(1.0, block_size));
        let input = [1.0f32; 4];
        let mut output = [0.0f32; 4];
        conv.process(&input, &mut output);
        // Starts at silence, ends at input × gain-1 IR.
        assert!(output[0].abs() < EPSILON, "fade-in start: {}", output[0]);
        assert!(
            (output[3] - 1.0).abs() < EPSILON,
            "fade-in end: {}",
            output[3]
        );
    }
}
