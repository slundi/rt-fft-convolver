//! Stereo and true-stereo convolution engines.
//!
//! # Two processing models
//!
//! ## [`StereoConvolver`] — independent L/R channels
//!
//! Each channel owns a separate [`UniformPartitionEngine`] and IR.  Left and
//! right never influence each other.  Use this when a stereo IR is stored as
//! two mono files, or when you want the same mono IR applied identically to
//! both channels.
//!
//! ```text
//! in_L ──► [engine_L] ──► out_L
//! in_R ──► [engine_R] ──► out_R
//! ```
//!
//! ## [`TrueStereoConvolver`] — 4-channel convolution matrix
//!
//! A real stereo microphone capture contains cross-channel energy: the left
//! microphone picks up room response caused by the right speaker, and vice
//! versa.  True stereo models this with four independent IR paths:
//!
//! ```text
//!          ir_ll        ir_lr
//! in_L ──►[L→L]──►(+)  [L→R]──►(+)
//!                  │              │
//!          ir_rl   ▼    ir_rr     ▼
//! in_R ──►[R→L]──► out_L  [R→R]──► out_R
//! ```
//!
//! Which gives:
//!
//! ```text
//! out_L = conv(in_L, ir_ll) + conv(in_R, ir_rl)
//! out_R = conv(in_L, ir_lr) + conv(in_R, ir_rr)
//! ```
//!
//! # Relation to the overlap-save engine
//!
//! Both types delegate per-channel processing to [`UniformPartitionEngine`].
//! That engine's `process()` *adds* to its output slice — so stereo engines
//! zero the output buffers before dispatching to each path, letting multiple
//! paths accumulate freely.
//!
//! [`UniformPartitionEngine`]: crate::engine::partition::UniformPartitionEngine

use crate::engine::partition::UniformPartitionEngine;

// ─── StereoConvolver ─────────────────────────────────────────────────────────

/// Independent left/right channel convolver.
///
/// Each channel runs its own [`UniformPartitionEngine`] with a separate IR
/// and independent FDL state.  The two channels never cross-contaminate.
///
/// # Construction
///
/// ```text
/// // Different IRs per channel:
/// let conv = StereoConvolver::new(&ir_left, &ir_right, block_size);
///
/// // Shared mono IR (applied identically to L and R):
/// let conv = StereoConvolver::new_mono_ir(&ir, block_size);
/// ```
///
/// **Allocates** on construction and [`load_ir`]; all other methods are
/// **real-time safe**.
///
/// [`load_ir`]: StereoConvolver::load_ir
/// [`UniformPartitionEngine`]: crate::engine::partition::UniformPartitionEngine
pub struct StereoConvolver {
    left: UniformPartitionEngine,
    right: UniformPartitionEngine,
}

impl StereoConvolver {
    /// Create a stereo convolver with **separate IRs** for each channel.
    ///
    /// `block_size` must equal the audio render block size and must be > 0.
    ///
    /// **Allocates** — call from a non-real-time thread.
    pub fn new(ir_left: &[f32], ir_right: &[f32], block_size: usize) -> Self {
        Self {
            left: UniformPartitionEngine::new(ir_left, block_size),
            right: UniformPartitionEngine::new(ir_right, block_size),
        }
    }

    /// Create a stereo convolver from a **single mono IR** applied to both
    /// channels identically.
    ///
    /// Each channel gets its own copy of the IR and its own FDL state, so
    /// the stereo field of the input signal is preserved.
    ///
    /// **Allocates** — call from a non-real-time thread.
    pub fn new_mono_ir(ir: &[f32], block_size: usize) -> Self {
        Self::new(ir, ir, block_size)
    }

    /// The audio block size both engines were built with.
    #[inline]
    pub fn block_size(&self) -> usize {
        self.left.block_size()
    }

    /// Replace both channel IRs and reset all delay state.
    ///
    /// **Allocates** — call from a non-real-time thread.
    pub fn load_ir(&mut self, ir_left: &[f32], ir_right: &[f32]) {
        self.left.load_ir(ir_left);
        self.right.load_ir(ir_right);
    }

    /// Clear both channels' delay history without reallocating.
    ///
    /// **Real-time safe** — no allocation.
    pub fn reset(&mut self) {
        self.left.reset();
        self.right.reset();
    }

    /// Process one stereo block.
    ///
    /// Zeroes `out_l` and `out_r`, then writes:
    ///
    /// ```text
    /// out_L = conv(in_L, ir_left)
    /// out_R = conv(in_R, ir_right)
    /// ```
    ///
    /// All four slices must have length equal to [`Self::block_size()`].
    ///
    /// **Real-time safe** — no allocation.
    pub fn process(&mut self, in_l: &[f32], in_r: &[f32], out_l: &mut [f32], out_r: &mut [f32]) {
        debug_assert_eq!(in_l.len(), self.block_size(), "in_l length mismatch");
        debug_assert_eq!(in_r.len(), self.block_size(), "in_r length mismatch");
        debug_assert_eq!(out_l.len(), self.block_size(), "out_l length mismatch");
        debug_assert_eq!(out_r.len(), self.block_size(), "out_r length mismatch");

        out_l.fill(0.0);
        out_r.fill(0.0);
        self.left.process(in_l, out_l);
        self.right.process(in_r, out_r);
    }
}

// ─── TrueStereoConvolver ─────────────────────────────────────────────────────

/// Four-channel (true-stereo) convolver.
///
/// Runs four independent [`UniformPartitionEngine`] paths and sums them
/// according to the routing matrix:
///
/// ```text
/// out_L = conv(in_L, ir_ll) + conv(in_R, ir_rl)
/// out_R = conv(in_L, ir_lr) + conv(in_R, ir_rr)
/// ```
///
/// | Field | Input | Output | Typical role                   |
/// |-------|-------|--------|--------------------------------|
/// | `ll`  | Left  | Left   | Direct left-channel response   |
/// | `lr`  | Left  | Right  | Left bleed into right channel  |
/// | `rl`  | Right | Left   | Right bleed into left channel  |
/// | `rr`  | Right | Right  | Direct right-channel response  |
///
/// For a **pure stereo** (non-crossing) IR pass `&[]` for `ir_lr` and `ir_rl`.
/// In that case the cross paths are no-ops and the engine degrades gracefully
/// to independent L/R processing.
///
/// **Allocates** on construction and [`load_ir`]; all other methods are
/// **real-time safe**.
///
/// [`load_ir`]: TrueStereoConvolver::load_ir
/// [`UniformPartitionEngine`]: crate::engine::partition::UniformPartitionEngine
pub struct TrueStereoConvolver {
    /// Left input → left output.
    ll: UniformPartitionEngine,
    /// Left input → right output.
    lr: UniformPartitionEngine,
    /// Right input → left output.
    rl: UniformPartitionEngine,
    /// Right input → right output.
    rr: UniformPartitionEngine,
}

impl TrueStereoConvolver {
    /// Create a true-stereo convolver from four IR paths.
    ///
    /// Pass `&[]` for any path that should be silent (e.g. the cross-channel
    /// paths of a non-crossing stereo IR).
    ///
    /// `block_size` must be > 0 and equal the audio render block size.
    ///
    /// **Allocates** — call from a non-real-time thread.
    pub fn new(
        ir_ll: &[f32],
        ir_lr: &[f32],
        ir_rl: &[f32],
        ir_rr: &[f32],
        block_size: usize,
    ) -> Self {
        Self {
            ll: UniformPartitionEngine::new(ir_ll, block_size),
            lr: UniformPartitionEngine::new(ir_lr, block_size),
            rl: UniformPartitionEngine::new(ir_rl, block_size),
            rr: UniformPartitionEngine::new(ir_rr, block_size),
        }
    }

    /// The audio block size all four engines were built with.
    #[inline]
    pub fn block_size(&self) -> usize {
        self.ll.block_size()
    }

    /// Replace all four IR paths and reset all internal state.
    ///
    /// **Allocates** — call from a non-real-time thread.
    pub fn load_ir(&mut self, ir_ll: &[f32], ir_lr: &[f32], ir_rl: &[f32], ir_rr: &[f32]) {
        self.ll.load_ir(ir_ll);
        self.lr.load_ir(ir_lr);
        self.rl.load_ir(ir_rl);
        self.rr.load_ir(ir_rr);
    }

    /// Clear all four convolution paths without reallocating.
    ///
    /// **Real-time safe** — no allocation.
    pub fn reset(&mut self) {
        self.ll.reset();
        self.lr.reset();
        self.rl.reset();
        self.rr.reset();
    }

    /// Process one true-stereo block.
    ///
    /// Zeroes `out_l` and `out_r`, then computes:
    ///
    /// ```text
    /// out_L = conv(in_L, ir_ll) + conv(in_R, ir_rl)
    /// out_R = conv(in_L, ir_lr) + conv(in_R, ir_rr)
    /// ```
    ///
    /// All four slices must have length equal to [`Self::block_size()`].
    ///
    /// **Real-time safe** — no allocation.
    pub fn process(&mut self, in_l: &[f32], in_r: &[f32], out_l: &mut [f32], out_r: &mut [f32]) {
        debug_assert_eq!(in_l.len(), self.block_size(), "in_l length mismatch");
        debug_assert_eq!(in_r.len(), self.block_size(), "in_r length mismatch");
        debug_assert_eq!(out_l.len(), self.block_size(), "out_l length mismatch");
        debug_assert_eq!(out_r.len(), self.block_size(), "out_r length mismatch");

        out_l.fill(0.0);
        out_r.fill(0.0);
        // Each engine *adds* to the output — see UniformPartitionEngine::process.
        self.ll.process(in_l, out_l); // L→L
        self.lr.process(in_l, out_r); // L→R
        self.rl.process(in_r, out_l); // R→L  (accumulates onto out_l)
        self.rr.process(in_r, out_r); // R→R  (accumulates onto out_r)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;
    const BLOCK: usize = 4;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Unit-impulse IR of exactly `block_size` samples: [1, 0, 0, …].
    fn unit_ir() -> Vec<f32> {
        let mut ir = vec![0.0f32; BLOCK];
        ir[0] = 1.0;
        ir
    }

    fn assert_approx(label: &str, got: &[f32], expected: &[f32]) {
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < EPSILON,
                "{label}[{i}]: expected {e}, got {g}"
            );
        }
    }

    // ── StereoConvolver ───────────────────────────────────────────────────────

    /// `new_mono_ir` with a unit impulse: each channel passes its input through
    /// unchanged.
    #[test]
    fn stereo_mono_ir_is_identity() {
        let mut conv = StereoConvolver::new_mono_ir(&unit_ir(), BLOCK);
        let in_l = [1.0f32, 2.0, 3.0, 4.0];
        let in_r = [5.0f32, 6.0, 7.0, 8.0];
        let (mut out_l, mut out_r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        conv.process(&in_l, &in_r, &mut out_l, &mut out_r);
        assert_approx("out_l", &out_l, &in_l);
        assert_approx("out_r", &out_r, &in_r);
    }

    /// Different gain IRs per channel: L×2, R×3.
    #[test]
    fn stereo_different_gains_per_channel() {
        let ir_l: Vec<f32> = std::iter::once(2.0)
            .chain(std::iter::repeat_n(0.0, 3))
            .collect();
        let ir_r: Vec<f32> = std::iter::once(3.0)
            .chain(std::iter::repeat_n(0.0, 3))
            .collect();
        let mut conv = StereoConvolver::new(&ir_l, &ir_r, BLOCK);
        let ones = [1.0f32; BLOCK];
        let (mut out_l, mut out_r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        conv.process(&ones, &ones, &mut out_l, &mut out_r);
        assert_approx("out_l", &out_l, &[2.0f32; BLOCK]);
        assert_approx("out_r", &out_r, &[3.0f32; BLOCK]);
    }

    /// Silence on R must not bleed into L output.
    #[test]
    fn stereo_channels_do_not_cross_contaminate() {
        let mut conv = StereoConvolver::new_mono_ir(&unit_ir(), BLOCK);
        let in_l = [1.0f32, 2.0, 3.0, 4.0];
        let in_r = [0.0f32; BLOCK];
        let (mut out_l, mut out_r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        conv.process(&in_l, &in_r, &mut out_l, &mut out_r);
        // out_r must be silence regardless of what in_l contains.
        assert_approx("out_r", &out_r, &[0.0f32; BLOCK]);
        // out_l must equal in_l (unit impulse IR).
        assert_approx("out_l", &out_l, &in_l);
    }

    /// `reset()` clears state — second block matches a fresh engine.
    #[test]
    fn stereo_reset_clears_state() {
        let ir = unit_ir();
        let mut conv = StereoConvolver::new_mono_ir(&ir, BLOCK);
        let noise = [9.0f32; BLOCK];
        let (mut scratch_l, mut scratch_r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        conv.process(&noise, &noise, &mut scratch_l, &mut scratch_r);
        conv.reset();

        let signal = [1.0f32, 2.0, 3.0, 4.0];
        let (mut got_l, mut got_r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        conv.process(&signal, &signal, &mut got_l, &mut got_r);

        // Must equal a freshly-constructed engine fed the same block.
        let mut fresh = StereoConvolver::new_mono_ir(&ir, BLOCK);
        let (mut exp_l, mut exp_r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        fresh.process(&signal, &signal, &mut exp_l, &mut exp_r);

        assert_approx("after_reset_l", &got_l, &exp_l);
        assert_approx("after_reset_r", &got_r, &exp_r);
    }

    /// `load_ir()` installs new coefficients and clears state.
    #[test]
    fn stereo_load_ir_replaces_coefficients() {
        let mut conv = StereoConvolver::new_mono_ir(&unit_ir(), BLOCK);
        // Prime with some state.
        let (mut s_l, mut s_r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        conv.process(&[1.0f32; BLOCK], &[1.0f32; BLOCK], &mut s_l, &mut s_r);

        // New IRs: gain-5 on L, gain-7 on R.
        let ir5: Vec<f32> = std::iter::once(5.0)
            .chain(std::iter::repeat_n(0.0, 3))
            .collect();
        let ir7: Vec<f32> = std::iter::once(7.0)
            .chain(std::iter::repeat_n(0.0, 3))
            .collect();
        conv.load_ir(&ir5, &ir7);

        let (mut out_l, mut out_r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        conv.process(&[1.0f32; BLOCK], &[1.0f32; BLOCK], &mut out_l, &mut out_r);
        assert_approx("out_l", &out_l, &[5.0f32; BLOCK]);
        assert_approx("out_r", &out_r, &[7.0f32; BLOCK]);
    }

    /// State persists correctly across blocks (tail energy shows up later).
    #[test]
    fn stereo_state_persists_across_blocks() {
        // IR: two-sample delay within one partition — h[0]=0, h[1]=1.
        let ir = [0.0f32, 1.0, 0.0, 0.0];
        let mut conv = StereoConvolver::new_mono_ir(&ir, BLOCK);

        // Block 1: impulse at position 0.
        let b1 = [1.0f32, 0.0, 0.0, 0.0];
        let (mut out_l, mut out_r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        conv.process(&b1, &b1, &mut out_l, &mut out_r);

        // y[n] = x[n-1]: expected [0, 1, 0, 0].
        assert_approx("out_l", &out_l, &[0.0, 1.0, 0.0, 0.0]);
        assert_approx("out_r", &out_r, &[0.0, 1.0, 0.0, 0.0]);
    }

    // ── TrueStereoConvolver ───────────────────────────────────────────────────

    /// Identity matrix: ll=rr=unit, lr=rl=empty.
    /// out_L = in_L, out_R = in_R.
    #[test]
    fn true_stereo_identity_matrix() {
        let unit = unit_ir();
        let mut conv = TrueStereoConvolver::new(&unit, &[], &[], &unit, BLOCK);
        let in_l = [1.0f32, 2.0, 3.0, 4.0];
        let in_r = [5.0f32, 6.0, 7.0, 8.0];
        let (mut out_l, mut out_r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        conv.process(&in_l, &in_r, &mut out_l, &mut out_r);
        assert_approx("out_l", &out_l, &in_l);
        assert_approx("out_r", &out_r, &in_r);
    }

    /// Swap matrix: ll=rr=empty, lr=rl=unit.
    /// out_L = in_R, out_R = in_L.
    #[test]
    fn true_stereo_swap_matrix() {
        let unit = unit_ir();
        let mut conv = TrueStereoConvolver::new(&[], &unit, &unit, &[], BLOCK);
        let in_l = [1.0f32, 2.0, 3.0, 4.0];
        let in_r = [5.0f32, 6.0, 7.0, 8.0];
        let (mut out_l, mut out_r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        conv.process(&in_l, &in_r, &mut out_l, &mut out_r);
        // out_L comes only from the R→L path (rl=unit), so out_L = in_R.
        assert_approx("out_l", &out_l, &in_r);
        // out_R comes only from the L→R path (lr=unit), so out_R = in_L.
        assert_approx("out_r", &out_r, &in_l);
    }

    /// All four paths contribute: verify the routing matrix numerically.
    ///
    /// IR gains: ll=1, lr=2, rl=3, rr=4.
    /// Input: in_L = in_R = [1, 0, 0, 0].
    ///
    /// Expected (first output sample):
    ///   out_L[0] = 1*in_L[0] + 3*in_R[0] = 4
    ///   out_R[0] = 2*in_L[0] + 4*in_R[0] = 6
    #[test]
    fn true_stereo_all_four_paths_contribute() {
        let make_ir = |gain: f32| -> Vec<f32> {
            std::iter::once(gain)
                .chain(std::iter::repeat_n(0.0, 3))
                .collect()
        };
        let mut conv = TrueStereoConvolver::new(
            &make_ir(1.0), // ll
            &make_ir(2.0), // lr
            &make_ir(3.0), // rl
            &make_ir(4.0), // rr
            BLOCK,
        );
        let impulse = [1.0f32, 0.0, 0.0, 0.0];
        let (mut out_l, mut out_r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        conv.process(&impulse, &impulse, &mut out_l, &mut out_r);

        assert!((out_l[0] - 4.0).abs() < EPSILON, "out_l[0]={}", out_l[0]);
        assert!((out_r[0] - 6.0).abs() < EPSILON, "out_r[0]={}", out_r[0]);
        // Remaining samples must be zero for this IR.
        for i in 1..BLOCK {
            assert!(out_l[i].abs() < EPSILON, "out_l[{i}]={}", out_l[i]);
            assert!(out_r[i].abs() < EPSILON, "out_r[{i}]={}", out_r[i]);
        }
    }

    /// Cross paths are truly independent: activating lr must not affect out_l.
    #[test]
    fn true_stereo_lr_path_does_not_affect_out_l() {
        let unit = unit_ir();
        // Only lr is active; ll, rl, rr are silent.
        let mut conv = TrueStereoConvolver::new(&[], &unit, &[], &[], BLOCK);
        let in_l = [1.0f32, 2.0, 3.0, 4.0];
        let (mut out_l, mut out_r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        conv.process(&in_l, &[0.0f32; BLOCK], &mut out_l, &mut out_r);
        // out_l must be silence (no path feeds into it).
        assert_approx("out_l", &out_l, &[0.0f32; BLOCK]);
        // out_r must equal in_l (lr path with unit IR).
        assert_approx("out_r", &out_r, &in_l);
    }

    /// `reset()` clears all four paths — result matches a fresh engine.
    #[test]
    fn true_stereo_reset_clears_all_paths() {
        let unit = unit_ir();
        let mut conv = TrueStereoConvolver::new(&unit, &unit, &unit, &unit, BLOCK);
        let noise = [9.0f32; BLOCK];
        let (mut s_l, mut s_r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        conv.process(&noise, &noise, &mut s_l, &mut s_r);
        conv.reset();

        let signal = [1.0f32, 0.0, 0.0, 0.0];
        let (mut got_l, mut got_r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        conv.process(&signal, &signal, &mut got_l, &mut got_r);

        let mut fresh = TrueStereoConvolver::new(&unit, &unit, &unit, &unit, BLOCK);
        let (mut exp_l, mut exp_r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        fresh.process(&signal, &signal, &mut exp_l, &mut exp_r);

        assert_approx("got_l", &got_l, &exp_l);
        assert_approx("got_r", &got_r, &exp_r);
    }

    /// `load_ir()` installs new IRs and clears state across all four paths.
    #[test]
    fn true_stereo_load_ir_replaces_all_paths() {
        let unit = unit_ir();
        let mut conv = TrueStereoConvolver::new(&unit, &[], &[], &unit, BLOCK);
        // Prime.
        let (mut s_l, mut s_r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        conv.process(&[1.0f32; BLOCK], &[1.0f32; BLOCK], &mut s_l, &mut s_r);

        // Install gain-2 IR on all paths.
        let ir2: Vec<f32> = std::iter::once(2.0)
            .chain(std::iter::repeat_n(0.0, 3))
            .collect();
        conv.load_ir(&ir2, &ir2, &ir2, &ir2);

        let ones = [1.0f32; BLOCK];
        let (mut out_l, mut out_r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        conv.process(&ones, &ones, &mut out_l, &mut out_r);
        // out_L = ll*1 + rl*1 = 2 + 2 = 4
        // out_R = lr*1 + rr*1 = 2 + 2 = 4
        assert_approx("out_l", &out_l, &[4.0f32; BLOCK]);
        assert_approx("out_r", &out_r, &[4.0f32; BLOCK]);
    }

    /// Multi-block tail energy propagates correctly through the rl cross-path.
    ///
    /// IR: rl = [0, 0, 0, 1] (4-sample delay inside one partition), others empty.
    /// Block 1 in_R = [1, 0, 0, 0] → out_L[3] = 1.
    #[test]
    fn true_stereo_rl_tail_persists_across_samples() {
        // rl IR is a pure delay: h[3] = 1.
        let ir_rl = [0.0f32, 0.0, 0.0, 1.0];
        let mut conv = TrueStereoConvolver::new(&[], &[], &ir_rl, &[], BLOCK);

        let in_r = [1.0f32, 0.0, 0.0, 0.0];
        let (mut out_l, mut out_r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        conv.process(&[0.0f32; BLOCK], &in_r, &mut out_l, &mut out_r);

        // out_L[3] = h[3] * in_R[0] = 1 * 1 = 1.
        assert!((out_l[3] - 1.0).abs() < EPSILON, "out_l[3]={}", out_l[3]);
        // out_R must be silence (rr is empty).
        assert_approx("out_r", &out_r, &[0.0f32; BLOCK]);
    }
}
