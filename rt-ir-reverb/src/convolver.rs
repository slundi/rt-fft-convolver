//! Stereo convolution abstraction and a stereo equal-power crossfader.
//!
//! The reverb switches between whole *stereo* impulse responses, so — unlike
//! the cab's per-channel [`MonoConvolver`](rt_ir_plugin_core::MonoConvolver) —
//! it needs a two-in/two-out primitive.  [`StereoConvolve`] unifies the crate's
//! [`StereoConvolver`] (2-channel IR) and [`TrueStereoConvolver`] (4-channel IR
//! with cross-feed), and [`StereoSlotCrossfader`] blends between two of them
//! click-free using the same equal-power rotation recurrence as the Phase-0
//! mono crossfader.

use std::f32::consts::FRAC_PI_2;

use rt_fft_convolver::{StereoConvolver, TrueStereoConvolver};

/// A two-in/two-out convolver the reverb can render and crossfade.
///
/// Implementors **overwrite** `out_l` / `out_r` with their response to
/// `in_l` / `in_r` (all four slices are `block_size` long).
pub trait StereoConvolve {
    /// Render one stereo block, overwriting the outputs.
    fn process(&mut self, in_l: &[f32], in_r: &[f32], out_l: &mut [f32], out_r: &mut [f32]);
    /// Clear internal delay state without reallocating.
    fn reset(&mut self);
    /// Block size this convolver was built for.
    fn block_size(&self) -> usize;
}

impl StereoConvolve for StereoConvolver {
    #[inline]
    fn process(&mut self, in_l: &[f32], in_r: &[f32], out_l: &mut [f32], out_r: &mut [f32]) {
        StereoConvolver::process(self, in_l, in_r, out_l, out_r);
    }
    #[inline]
    fn reset(&mut self) {
        StereoConvolver::reset(self);
    }
    #[inline]
    fn block_size(&self) -> usize {
        StereoConvolver::block_size(self)
    }
}

impl StereoConvolve for TrueStereoConvolver {
    #[inline]
    fn process(&mut self, in_l: &[f32], in_r: &[f32], out_l: &mut [f32], out_r: &mut [f32]) {
        TrueStereoConvolver::process(self, in_l, in_r, out_l, out_r);
    }
    #[inline]
    fn reset(&mut self) {
        TrueStereoConvolver::reset(self);
    }
    #[inline]
    fn block_size(&self) -> usize {
        TrueStereoConvolver::block_size(self)
    }
}

/// A prepared reverb IR: either a plain stereo or a true-stereo convolver.
///
/// The controller picks the variant from the IR's channel count (2 → stereo,
/// 4 → true-stereo, 1 → mono duplicated).
///
/// The true-stereo variant is larger (4 partition engines vs 2), but boxing it
/// would add a pointer indirection on every render block; the modest memory
/// overhead of two inline instances (front + back) is the better trade.
#[allow(clippy::large_enum_variant)]
pub enum ReverbConvolver {
    /// 1- or 2-channel IR.
    Stereo(StereoConvolver),
    /// 4-channel true-stereo IR (L→L, L→R, R→L, R→R).
    True(TrueStereoConvolver),
}

impl StereoConvolve for ReverbConvolver {
    #[inline]
    fn process(&mut self, in_l: &[f32], in_r: &[f32], out_l: &mut [f32], out_r: &mut [f32]) {
        match self {
            ReverbConvolver::Stereo(c) => c.process(in_l, in_r, out_l, out_r),
            ReverbConvolver::True(c) => c.process(in_l, in_r, out_l, out_r),
        }
    }
    #[inline]
    fn reset(&mut self) {
        match self {
            ReverbConvolver::Stereo(c) => c.reset(),
            ReverbConvolver::True(c) => c.reset(),
        }
    }
    #[inline]
    fn block_size(&self) -> usize {
        match self {
            ReverbConvolver::Stereo(c) => c.block_size(),
            ReverbConvolver::True(c) => c.block_size(),
        }
    }
}

/// In-flight equal-power fade state (see the Phase-0 mono crossfader for the
/// rotation-recurrence derivation).
struct Fade {
    c: f32,
    s: f32,
    cos_d: f32,
    sin_d: f32,
    remaining: usize,
}

/// Click-free switcher between two [`StereoConvolve`]rs with a per-fade
/// duration.  Longer fades than the cab (reverb tails need it).
pub struct StereoSlotCrossfader<T: StereoConvolve> {
    front: T,
    back: Option<T>,
    block_size: usize,
    // Per-side scratch for the two engines during a fade.
    fl: Vec<f32>,
    fr: Vec<f32>,
    bl: Vec<f32>,
    br: Vec<f32>,
    fade: Option<Fade>,
}

impl<T: StereoConvolve> StereoSlotCrossfader<T> {
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
            fl: vec![0.0; block_size],
            fr: vec![0.0; block_size],
            bl: vec![0.0; block_size],
            br: vec![0.0; block_size],
            fade: None,
        }
    }

    /// Whether a fade is currently in progress.
    #[inline]
    pub fn is_fading(&self) -> bool {
        self.fade.is_some()
    }

    /// Start fading to `new` over `duration_samples`, returning the retired
    /// convolver for off-thread disposal.  `0` = instant swap.
    ///
    /// **Real-time safe** — moves convolvers, never allocates.
    ///
    /// # Panics
    ///
    /// Panics if `new.block_size()` differs from this crossfader's.
    pub fn start_fade(&mut self, new: T, duration_samples: usize) -> Option<T> {
        assert_eq!(
            new.block_size(),
            self.block_size,
            "start_fade: block_size mismatch"
        );
        if self.fade.is_some() {
            self.finish_fade();
        }
        if duration_samples == 0 {
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
        self.back.replace(new)
    }

    /// Clear both engines and abort any active fade.  **Real-time safe.**
    pub fn reset(&mut self) {
        self.front.reset();
        if let Some(b) = self.back.as_mut() {
            b.reset();
        }
        self.fade = None;
    }

    /// Render one stereo block, blending during a fade.  **Real-time safe.**
    pub fn process(&mut self, in_l: &[f32], in_r: &[f32], out_l: &mut [f32], out_r: &mut [f32]) {
        let Some(fade) = self.fade.as_mut() else {
            self.front.process(in_l, in_r, out_l, out_r);
            return;
        };

        self.front.process(in_l, in_r, &mut self.fl, &mut self.fr);
        let back = self
            .back
            .as_mut()
            .expect("fade active without a back convolver");
        back.process(in_l, in_r, &mut self.bl, &mut self.br);

        let n = out_l.len();
        let blended = fade.remaining.min(n);
        for i in 0..blended {
            out_l[i] = fade.c * self.fl[i] + fade.s * self.bl[i];
            out_r[i] = fade.c * self.fr[i] + fade.s * self.br[i];
            let next_c = fade.c * fade.cos_d - fade.s * fade.sin_d;
            let next_s = fade.s * fade.cos_d + fade.c * fade.sin_d;
            fade.c = next_c;
            fade.s = next_s;
        }
        // Past completion: fully the incoming (back) convolver.
        out_l[blended..n].copy_from_slice(&self.bl[blended..n]);
        out_r[blended..n].copy_from_slice(&self.br[blended..n]);

        fade.remaining -= blended;
        if fade.remaining == 0 {
            self.finish_fade();
        }
    }

    /// Promote the incoming engine to `front`, retaining the old front.
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

    const BLOCK: usize = 8;
    const EPS: f32 = 1e-4;

    /// A test double that emits constants (ignores input) so blended output is
    /// a pure function of the fade gains.
    struct ConstStereo {
        vl: f32,
        vr: f32,
        block_size: usize,
    }

    impl StereoConvolve for ConstStereo {
        fn process(&mut self, _l: &[f32], _r: &[f32], out_l: &mut [f32], out_r: &mut [f32]) {
            out_l.fill(self.vl);
            out_r.fill(self.vr);
        }
        fn reset(&mut self) {}
        fn block_size(&self) -> usize {
            self.block_size
        }
    }

    fn c(vl: f32, vr: f32) -> ConstStereo {
        ConstStereo {
            vl,
            vr,
            block_size: BLOCK,
        }
    }

    #[test]
    fn idle_renders_front_only() {
        let mut xf = StereoSlotCrossfader::new(c(0.5, -0.5));
        let (mut l, mut r) = (vec![0.0; BLOCK], vec![0.0; BLOCK]);
        xf.process(&[0.0; BLOCK], &[0.0; BLOCK], &mut l, &mut r);
        assert!(l.iter().all(|&x| (x - 0.5).abs() < EPS));
        assert!(r.iter().all(|&x| (x + 0.5).abs() < EPS));
    }

    #[test]
    fn instant_swap_replaces_front() {
        let mut xf = StereoSlotCrossfader::new(c(1.0, 1.0));
        assert!(xf.start_fade(c(2.0, 3.0), 0).is_none());
        assert!(!xf.is_fading());
        let (mut l, mut r) = (vec![0.0; BLOCK], vec![0.0; BLOCK]);
        xf.process(&[0.0; BLOCK], &[0.0; BLOCK], &mut l, &mut r);
        assert!((l[0] - 2.0).abs() < EPS);
        assert!((r[0] - 3.0).abs() < EPS);
    }

    #[test]
    fn fade_follows_equal_power_curve_per_channel() {
        let mut xf = StereoSlotCrossfader::new(c(1.0, 1.0));
        xf.start_fade(c(3.0, 5.0), BLOCK);
        assert!(xf.is_fading());

        let (mut l, mut r) = (vec![0.0; BLOCK], vec![0.0; BLOCK]);
        xf.process(&[0.0; BLOCK], &[0.0; BLOCK], &mut l, &mut r);

        let step = FRAC_PI_2 / BLOCK as f32;
        for n in 0..BLOCK {
            let (cos, sin) = ((n as f32 * step).cos(), (n as f32 * step).sin());
            assert!((l[n] - (cos * 1.0 + sin * 3.0)).abs() < EPS, "L[{n}]");
            assert!((r[n] - (cos * 1.0 + sin * 5.0)).abs() < EPS, "R[{n}]");
        }
        assert!(!xf.is_fading(), "single-block fade completes");
    }

    #[test]
    fn restarting_mid_fade_returns_retired() {
        let mut xf = StereoSlotCrossfader::new(c(1.0, 1.0));
        xf.start_fade(c(2.0, 2.0), BLOCK * 4);
        let (mut l, mut r) = (vec![0.0; BLOCK], vec![0.0; BLOCK]);
        xf.process(&[0.0; BLOCK], &[0.0; BLOCK], &mut l, &mut r);
        assert!(xf.start_fade(c(3.0, 3.0), BLOCK).is_some());
    }
}
