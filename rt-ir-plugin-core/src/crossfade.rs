//! [`SlotCrossfader`] — click-free IR switching with a **per-fade** duration.
//!
//! # Relation to the core library's `CrossfadingConvolver`
//!
//! `rt_fft_convolver::CrossfadingConvolver` bakes the fade length in at
//! construction.  The plugins expose a live `Crossfade Time` parameter (5-200 ms
//! for the cab, up to 500 ms for reverb tails), so we need the fade length
//! supplied *per switch*.  `SlotCrossfader` does exactly that and, being
//! generic over a [`MonoConvolver`], works with any engine — `HybridConvolver`
//! for the zero-latency cab, `UniformPartitionEngine` for a reverb channel.
//!
//! # Equal-power blend without a lookup table
//!
//! Because the duration is not known ahead of time we cannot pre-compute a
//! gain table (as `CrossfadingConvolver` does).  Instead we advance the fade
//! angle `θ: 0 → π/2` with a **rotation recurrence**, which needs only one
//! multiply-add pair per sample in the render loop — no `sin`/`cos` and no
//! allocation inside `process()`:
//!
//! ```text
//! step    = (π/2) / duration          (computed once, off the audio thread)
//! cos_d   = cos(step),  sin_d = sin(step)
//! (c, s)  starts at (1, 0) = (cos 0, sin 0)
//! gain    front = c,  back = s         (c² + s² = 1  → equal power)
//! rotate  c' = c·cos_d − s·sin_d
//!         s' = s·cos_d + c·sin_d       ((c, s) = (cos θ, sin θ) after n steps)
//! ```
//!
//! `front` fades `1 → 0` while `back` fades `0 → 1`; their squared gains sum to
//! one, so a switch between two uncorrelated IRs holds perceived loudness flat.
//!
//! # Real-time safety
//!
//! [`SlotCrossfader::start_fade`] takes a *prebuilt* convolver by value and
//! returns the displaced one for the caller to drop off-thread — it never
//! allocates.  [`SlotCrossfader::process`] renders into pre-allocated scratch
//! buffers.  Both are safe to call from the audio thread; only *building* the
//! incoming convolver (done by the worker) allocates.

use std::f32::consts::FRAC_PI_2;

use rt_fft_convolver::{HybridConvolver, UniformPartitionEngine};

/// A single-channel convolver `SlotCrossfader` can blend.
///
/// Implementors render one block, **overwriting** `output` with their response
/// to `input` (both slices are `block_size` long).
pub trait MonoConvolver {
    /// Render one block, overwriting `output` with the convolution of `input`.
    fn process_block(&mut self, input: &[f32], output: &mut [f32]);
    /// Clear internal state (delay lines) without reallocating.
    fn reset(&mut self);
    /// Block size this convolver was built for.
    fn block_size(&self) -> usize;
}

// `HybridConvolver::process` already overwrites its output — a direct wrap.
impl MonoConvolver for HybridConvolver {
    #[inline]
    fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        self.process(input, output);
    }
    #[inline]
    fn reset(&mut self) {
        HybridConvolver::reset(self);
    }
    #[inline]
    fn block_size(&self) -> usize {
        HybridConvolver::block_size(self)
    }
}

// `UniformPartitionEngine::process` *accumulates* into its output, so we clear
// the buffer first to honour the "overwrite" contract of `MonoConvolver`.
impl MonoConvolver for UniformPartitionEngine {
    #[inline]
    fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        output.fill(0.0);
        self.process(input, output);
    }
    #[inline]
    fn reset(&mut self) {
        UniformPartitionEngine::reset(self);
    }
    #[inline]
    fn block_size(&self) -> usize {
        UniformPartitionEngine::block_size(self)
    }
}

/// Convert a crossfade time in milliseconds to a sample count at `sample_rate`.
///
/// The plugin's `Crossfade Time` parameter is in ms; the audio thread wants
/// samples.  Rounds to nearest.
#[inline]
pub fn ms_to_samples(ms: f32, sample_rate: u32) -> usize {
    ((ms.max(0.0) * sample_rate as f32) / 1000.0).round() as usize
}

/// State of an in-flight equal-power fade (see the module docs).
struct Fade {
    /// Current `cos θ` — the front (outgoing) gain.
    c: f32,
    /// Current `sin θ` — the back (incoming) gain.
    s: f32,
    /// `cos(step)`, the per-sample rotation cosine.
    cos_d: f32,
    /// `sin(step)`, the per-sample rotation sine.
    sin_d: f32,
    /// Samples of blend left before the fade completes.
    remaining: usize,
}

/// Click-free IR switcher that blends between two convolvers over a
/// per-switch duration.
///
/// `front` is the audible engine; `back` is idle, or the incoming side of an
/// active fade.  A completed fade swaps the two so `front` always holds the
/// currently-audible IR.
pub struct SlotCrossfader<T: MonoConvolver> {
    front: T,
    /// Incoming / retired engine.  `Some` while (and after) a fade; the old
    /// front is retained here so the caller can reclaim it.
    back: Option<T>,
    block_size: usize,
    front_buf: Vec<f32>,
    back_buf: Vec<f32>,
    fade: Option<Fade>,
}

impl<T: MonoConvolver> SlotCrossfader<T> {
    /// Create a crossfader with `front` as the initial audible convolver.
    ///
    /// **Allocates** the scratch buffers — call off the audio thread.
    pub fn new(front: T) -> Self {
        let block_size = front.block_size();
        assert!(block_size > 0, "convolver block_size must be > 0");
        Self {
            front,
            back: None,
            block_size,
            front_buf: vec![0.0; block_size],
            back_buf: vec![0.0; block_size],
            fade: None,
        }
    }

    /// Block size both engines share.
    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Whether a fade is currently in progress.
    #[inline]
    pub fn is_fading(&self) -> bool {
        self.fade.is_some()
    }

    /// Samples remaining in the active fade, or `None` if not fading.
    #[inline]
    pub fn fade_remaining(&self) -> Option<usize> {
        self.fade.as_ref().map(|f| f.remaining)
    }

    /// Start fading to `new` over `duration_samples`, returning the convolver
    /// that is no longer needed (so the caller can drop it off-thread).
    ///
    /// - `duration_samples == 0` performs an instant swap (may click).
    /// - If a fade is already running it is collapsed to completion first, so
    ///   rapid switches always fade *from the currently-audible IR*.
    ///
    /// **Real-time safe** — moves convolvers, never allocates.
    ///
    /// # Panics
    ///
    /// Panics if `new.block_size()` differs from this crossfader's block size.
    pub fn start_fade(&mut self, new: T, duration_samples: usize) -> Option<T> {
        assert_eq!(
            new.block_size(),
            self.block_size,
            "start_fade: block_size mismatch"
        );

        // Collapse any in-flight fade so `front` is the truly-audible engine.
        if self.fade.is_some() {
            self.finish_fade();
        }

        if duration_samples == 0 {
            // Instant switch: `new` becomes front; the old front is retired to
            // `back` and whatever `back` held is handed back to the caller.
            let old_front = std::mem::replace(&mut self.front, new);
            return self.back.replace(old_front);
        }

        let step = FRAC_PI_2 / duration_samples as f32;
        self.fade = Some(Fade {
            c: 1.0,
            s: 0.0,
            cos_d: step.cos(),
            sin_d: step.sin(),
            remaining: duration_samples,
        });
        // Install the incoming engine; return the previous back occupant.
        self.back.replace(new)
    }

    /// Clear both engines' state and abort any active fade.
    ///
    /// **Real-time safe** — no allocation.
    pub fn reset(&mut self) {
        self.front.reset();
        if let Some(back) = self.back.as_mut() {
            back.reset();
        }
        self.fade = None;
    }

    /// Render one block into `output` (overwriting it).
    ///
    /// When idle, only `front` runs.  During a fade both engines run and their
    /// outputs are blended with the equal-power envelope.
    ///
    /// **Real-time safe** — no allocation.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.block_size, "input length mismatch");
        debug_assert_eq!(output.len(), self.block_size, "output length mismatch");

        // Fast path: no fade — render the audible engine straight to output.
        let Some(fade) = self.fade.as_mut() else {
            self.front.process_block(input, output);
            return;
        };

        // Fade active: render both engines into scratch, then blend.  `back`
        // is guaranteed `Some` here (set together with `fade` in start_fade).
        self.front.process_block(input, &mut self.front_buf);
        let back = self
            .back
            .as_mut()
            .expect("fade active without a back convolver");
        back.process_block(input, &mut self.back_buf);

        let n = output.len();
        // Samples covered by the blend before the fade completes this block.
        let blended = fade.remaining.min(n);

        let (out_blend, out_tail) = output.split_at_mut(blended);
        for ((out, &front), &back) in out_blend
            .iter_mut()
            .zip(&self.front_buf[..blended])
            .zip(&self.back_buf[..blended])
        {
            *out = fade.c * front + fade.s * back;
            // Rotate (c, s) by one step: (cos θ, sin θ) → (cos θ+step, sin θ+step).
            let next_c = fade.c * fade.cos_d - fade.s * fade.sin_d;
            let next_s = fade.s * fade.cos_d + fade.c * fade.sin_d;
            fade.c = next_c;
            fade.s = next_s;
        }
        // Any samples past fade completion belong entirely to the incoming IR.
        out_tail.copy_from_slice(&self.back_buf[blended..n]);

        fade.remaining -= blended;
        if fade.remaining == 0 {
            self.finish_fade();
        }
    }

    /// Promote the incoming (`back`) engine to `front`, retaining the old
    /// front in `back`, and clear the fade state.
    fn finish_fade(&mut self) {
        if let Some(incoming) = self.back.take() {
            let old_front = std::mem::replace(&mut self.front, incoming);
            self.back = Some(old_front);
        }
        self.fade = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-4;

    /// Test double: ignores its input and emits a constant, so blended output
    /// is a pure function of the fade gains — easy to check against theory.
    struct ConstConvolver {
        value: f32,
        block_size: usize,
        reset_count: usize,
    }

    impl ConstConvolver {
        fn new(value: f32, block_size: usize) -> Self {
            Self {
                value,
                block_size,
                reset_count: 0,
            }
        }
    }

    impl MonoConvolver for ConstConvolver {
        fn process_block(&mut self, _input: &[f32], output: &mut [f32]) {
            output.fill(self.value);
        }
        fn reset(&mut self) {
            self.reset_count += 1;
        }
        fn block_size(&self) -> usize {
            self.block_size
        }
    }

    #[test]
    fn idle_renders_front_only() {
        let mut xf = SlotCrossfader::new(ConstConvolver::new(0.75, 4));
        let mut out = [0.0; 4];
        xf.process(&[0.0; 4], &mut out);
        assert!(!xf.is_fading());
        assert_eq!(out, [0.75; 4]);
    }

    #[test]
    fn instant_swap_replaces_front_without_fading() {
        let mut xf = SlotCrossfader::new(ConstConvolver::new(1.0, 4));
        let displaced = xf.start_fade(ConstConvolver::new(2.0, 4), 0);
        // No previous back → nothing displaced.
        assert!(displaced.is_none());
        assert!(!xf.is_fading());

        let mut out = [0.0; 4];
        xf.process(&[0.0; 4], &mut out);
        assert_eq!(out, [2.0; 4]);
    }

    #[test]
    fn blend_follows_equal_power_curve() {
        // duration == block_size so the whole fade lands in one block.
        let block = 8;
        let front_value = 1.0;
        let back_value = 3.0;
        let mut xf = SlotCrossfader::new(ConstConvolver::new(front_value, block));
        let displaced = xf.start_fade(ConstConvolver::new(back_value, block), block);
        assert!(displaced.is_none());
        assert!(xf.is_fading());
        assert_eq!(xf.fade_remaining(), Some(block));

        let mut out = vec![0.0; block];
        xf.process(&vec![0.0; block], &mut out);

        // Reference: sample n uses θ = n · (π/2 / duration).
        let step = FRAC_PI_2 / block as f32;
        for (n, &got) in out.iter().enumerate() {
            let theta = n as f32 * step;
            let expected = theta.cos() * front_value + theta.sin() * back_value;
            assert!(
                (got - expected).abs() < EPSILON,
                "sample {n}: got {got}, expected {expected}"
            );
        }

        // Fade completed exactly at the block boundary.
        assert!(!xf.is_fading());
    }

    #[test]
    fn gains_stay_equal_power_across_the_fade() {
        // With front == back == 1.0, the blended output equals (cos θ + sin θ);
        // more usefully we verify cos²θ + sin²θ ≈ 1 by reconstructing the pair.
        let block = 16;
        let mut xf = SlotCrossfader::new(ConstConvolver::new(1.0, block));
        xf.start_fade(ConstConvolver::new(0.0, block), block);

        // back_value 0 → output == cos θ (the front gain) each sample.
        let mut cos_out = vec![0.0; block];
        xf.process(&vec![0.0; block], &mut cos_out);

        // Rebuild with back_value 1, front_value 0 to read the back gain sin θ.
        let mut xf2 = SlotCrossfader::new(ConstConvolver::new(0.0, block));
        xf2.start_fade(ConstConvolver::new(1.0, block), block);
        let mut sin_out = vec![0.0; block];
        xf2.process(&vec![0.0; block], &mut sin_out);

        for n in 0..block {
            let power = cos_out[n] * cos_out[n] + sin_out[n] * sin_out[n];
            assert!(
                (power - 1.0).abs() < 1e-3,
                "sample {n}: power {power} not ~1.0"
            );
        }
    }

    #[test]
    fn fade_spans_multiple_blocks() {
        let block = 8;
        let duration = 20; // 2.5 blocks
        let mut xf = SlotCrossfader::new(ConstConvolver::new(1.0, block));
        xf.start_fade(ConstConvolver::new(5.0, block), duration);

        let mut out = vec![0.0; block];
        // Block 1: still fading.
        xf.process(&vec![0.0; block], &mut out);
        assert!(xf.is_fading());
        assert_eq!(xf.fade_remaining(), Some(duration - block));
        // Block 2: still fading.
        xf.process(&vec![0.0; block], &mut out);
        assert!(xf.is_fading());
        assert_eq!(xf.fade_remaining(), Some(duration - 2 * block));
        // Block 3: fade completes mid-block; tail is pure incoming (5.0).
        xf.process(&vec![0.0; block], &mut out);
        assert!(!xf.is_fading());
        assert!((out[block - 1] - 5.0).abs() < EPSILON);

        // Steady state afterwards: the incoming IR is now the front.
        xf.process(&vec![0.0; block], &mut out);
        assert_eq!(out, vec![5.0; block]);
    }

    #[test]
    fn restarting_mid_fade_collapses_the_previous_one() {
        let block = 8;
        let mut xf = SlotCrossfader::new(ConstConvolver::new(1.0, block));
        // Start A→B, advance one block (still fading).
        xf.start_fade(ConstConvolver::new(2.0, block), block * 4);
        let mut out = vec![0.0; block];
        xf.process(&vec![0.0; block], &mut out);
        assert!(xf.is_fading());

        // Start a second fade before the first finished: the displaced engine
        // is returned so the caller can reclaim it.
        let displaced = xf.start_fade(ConstConvolver::new(3.0, block), block);
        assert!(displaced.is_some(), "collapsed fade must return an engine");
        assert!(xf.is_fading());
    }

    #[test]
    fn reset_aborts_fade_and_resets_engines() {
        let block = 4;
        let mut xf = SlotCrossfader::new(ConstConvolver::new(1.0, block));
        xf.start_fade(ConstConvolver::new(2.0, block), block * 2);
        assert!(xf.is_fading());
        xf.reset();
        assert!(!xf.is_fading());
    }

    #[test]
    fn ms_to_samples_rounds_to_nearest() {
        assert_eq!(ms_to_samples(0.0, 48_000), 0);
        assert_eq!(ms_to_samples(1.0, 48_000), 48);
        assert_eq!(ms_to_samples(10.0, 44_100), 441);
        assert_eq!(ms_to_samples(5.0, 48_000), 240);
        // Negative durations clamp to zero.
        assert_eq!(ms_to_samples(-3.0, 48_000), 0);
    }

    /// Wiring check: the crossfader drives a real `UniformPartitionEngine`
    /// through the `MonoConvolver` impl and produces finite audio.
    #[test]
    fn works_with_real_partition_engine() {
        let block = 64;
        let mut ir_a = vec![0.0f32; block];
        ir_a[0] = 1.0; // unit impulse
        let mut ir_b = vec![0.0f32; block];
        ir_b[0] = 0.5;

        let front = UniformPartitionEngine::new(&ir_a, block);
        let back = UniformPartitionEngine::new(&ir_b, block);
        let mut xf = SlotCrossfader::new(front);
        xf.start_fade(back, block);
        assert!(xf.is_fading());

        let mut input = vec![0.0f32; block];
        input[0] = 1.0;
        let mut out = vec![0.0f32; block];
        xf.process(&input, &mut out);

        assert!(out.iter().all(|s| s.is_finite()));
        // At sample 0 the fade is front-dominant (gain ~1), so ~ir_a[0] = 1.0.
        assert!((out[0] - 1.0).abs() < 1e-3, "out[0] = {}", out[0]);
        assert!(
            !xf.is_fading(),
            "single-block fade of length {block} completes"
        );
    }
}
