//! `pulp`-dispatched SIMD kernels for the resampler and shaper hot paths.
//!
//! # Design
//!
//! Same two-variant pattern as `rt-neural-core::kernels`: a **scalar
//! reference** (source of truth for tests, well auto-vectorised by LLVM) and a
//! **[`pulp`]-dispatched** version carrying the identical math through an
//! explicit SIMD width, letting `pulp` pick SSE / AVX2 / AVX-512 / NEON at
//! runtime.  The [`Kernels`] handle caches a [`pulp::Arch`] so the
//! ISA-selection cost is paid once, off the audio thread.
//!
//! Two kernels live here:
//!
//! - [`Kernels::fir_dot`] — SIMD-across-taps FIR inner product, the core of
//!   the polyphase resampler.
//! - [`Kernels::soft_clip_block`] — vectorised cubic soft-clip, the batch
//!   shaper evaluation called out by the roadmap.

use pulp::{Arch, Simd, WithSimd};

/// A cached SIMD-dispatch handle.
///
/// Construct once (off the audio thread); the handle is `Copy` and holds only
/// a pre-selected ISA tag.
#[derive(Copy, Clone)]
pub struct Kernels {
    arch: Arch,
}

impl Kernels {
    /// Detect the best SIMD ISA available on the current CPU.
    ///
    /// **Not** real-time safe — call once, off the audio thread.
    #[inline]
    pub fn new() -> Self {
        Self { arch: Arch::new() }
    }

    /// FIR inner product `Σᵢ taps[i] · history[i]`.
    ///
    /// The two slices must have equal length (debug-asserted); this is the
    /// per-output-sample kernel of the polyphase resampler.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn fir_dot(&self, taps: &[f32], history: &[f32]) -> f32 {
        debug_assert_eq!(taps.len(), history.len(), "fir_dot: length mismatch");
        self.arch.dispatch(DotKernel { taps, history })
    }

    /// Cubic soft-clip a block: `output[i] = softclip(input[i] · drive)`.
    ///
    /// The transfer is the same monotonic cubic used by
    /// [`crate::shaper::Shaper::CubicSoft`] with a pre-gain `drive` folded in,
    /// evaluated a full SIMD register at a time.  `input` and `output` must
    /// have equal length (debug-asserted).
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn soft_clip_block(&self, input: &[f32], output: &mut [f32], drive: f32) {
        debug_assert_eq!(
            input.len(),
            output.len(),
            "soft_clip_block: length mismatch"
        );
        self.arch.dispatch(SoftClipKernel {
            input,
            output,
            drive,
        });
    }
}

impl Default for Kernels {
    fn default() -> Self {
        Self::new()
    }
}

/// Scalar cubic soft-clip with pre-gain — the reference for the SIMD kernel.
#[inline]
pub(crate) fn soft_clip_scalar(sample: f32, drive: f32) -> f32 {
    let clamped = (sample * drive).clamp(-1.0, 1.0);
    1.5 * (clamped - clamped * clamped * clamped / 3.0)
}

// ─── Pulp kernel implementations ─────────────────────────────────────────────

/// Portable-SIMD FIR inner product.
struct DotKernel<'taps, 'history> {
    taps: &'taps [f32],
    history: &'history [f32],
}

impl<'taps, 'history> WithSimd for DotKernel<'taps, 'history> {
    type Output = f32;

    #[inline(always)]
    fn with_simd<SimdArch: Simd>(self, simd: SimdArch) -> f32 {
        let (tap_lanes, tap_tail) = SimdArch::as_simd_f32s(self.taps);
        let (history_lanes, history_tail) = SimdArch::as_simd_f32s(self.history);

        let mut acc = simd.splat_f32s(0.0);
        for (tap_lane, history_lane) in tap_lanes.iter().zip(history_lanes.iter()) {
            acc = simd.mul_add_f32s(*tap_lane, *history_lane, acc);
        }
        let mut scalar_acc = simd.reduce_sum_f32s(acc);
        for (tap, history) in tap_tail.iter().zip(history_tail.iter()) {
            scalar_acc += tap * history;
        }
        scalar_acc
    }
}

/// Portable-SIMD cubic soft-clip block.
struct SoftClipKernel<'input, 'output> {
    input: &'input [f32],
    output: &'output mut [f32],
    drive: f32,
}

impl<'input, 'output> WithSimd for SoftClipKernel<'input, 'output> {
    type Output = ();

    #[inline(always)]
    fn with_simd<SimdArch: Simd>(self, simd: SimdArch) {
        let (input_lanes, input_tail) = SimdArch::as_simd_f32s(self.input);
        let (output_lanes, output_tail) = SimdArch::as_mut_simd_f32s(self.output);

        let drive = simd.splat_f32s(self.drive);
        let one = simd.splat_f32s(1.0);
        let minus_one = simd.splat_f32s(-1.0);
        let scale = simd.splat_f32s(1.5);
        let third = simd.splat_f32s(1.0 / 3.0);

        for (output_lane, input_lane) in output_lanes.iter_mut().zip(input_lanes.iter()) {
            let driven = simd.mul_f32s(*input_lane, drive);
            // clamp to [-1, 1]
            let clamped = simd.max_f32s(minus_one, simd.min_f32s(one, driven));
            let cubed = simd.mul_f32s(clamped, simd.mul_f32s(clamped, clamped));
            // 1.5 · (c − c³/3)
            let shaped = simd.sub_f32s(clamped, simd.mul_f32s(third, cubed));
            *output_lane = simd.mul_f32s(scale, shaped);
        }
        for (output, input) in output_tail.iter_mut().zip(input_tail.iter()) {
            *output = soft_clip_scalar(*input, self.drive);
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_dot(taps: &[f32], history: &[f32]) -> f32 {
        taps.iter().zip(history.iter()).map(|(a, b)| a * b).sum()
    }

    /// FIR dot must match the scalar reference across lengths that straddle
    /// the SIMD lane boundary.
    #[test]
    fn fir_dot_matches_scalar() {
        let kernels = Kernels::new();
        for length in [1usize, 2, 3, 4, 7, 8, 15, 16, 17, 63, 64, 65, 125] {
            let taps: Vec<f32> = (0..length).map(|i| (i as f32).sin() * 0.3).collect();
            let history: Vec<f32> = (0..length).map(|i| 1.0 - i as f32 * 0.01).collect();
            let dispatched = kernels.fir_dot(&taps, &history);
            let reference = scalar_dot(&taps, &history);
            assert!(
                (dispatched - reference).abs() < 1e-4 * reference.abs().max(1.0),
                "fir_dot length={length}: {dispatched} vs {reference}"
            );
        }
    }

    /// Dot of empty slices is zero.
    #[test]
    fn fir_dot_empty_is_zero() {
        let kernels = Kernels::new();
        assert_eq!(kernels.fir_dot(&[], &[]), 0.0);
    }

    /// Soft-clip block must match the scalar reference for every sample.
    #[test]
    fn soft_clip_block_matches_scalar() {
        let kernels = Kernels::new();
        for length in [1usize, 4, 7, 8, 16, 17, 31, 64, 100] {
            let input: Vec<f32> = (0..length).map(|i| (i as f32 - 50.0) * 0.05).collect();
            let mut dispatched = vec![0.0f32; length];
            kernels.soft_clip_block(&input, &mut dispatched, 2.0);
            for (index, &sample) in input.iter().enumerate() {
                let reference = soft_clip_scalar(sample, 2.0);
                assert!(
                    (dispatched[index] - reference).abs() < 1e-5,
                    "soft_clip len={length} idx={index}: {} vs {reference}",
                    dispatched[index]
                );
            }
        }
    }

    /// Soft-clip saturates a hot input toward ±1.
    #[test]
    fn soft_clip_saturates() {
        let kernels = Kernels::new();
        let input = vec![10.0f32; 8];
        let mut output = vec![0.0f32; 8];
        kernels.soft_clip_block(&input, &mut output, 1.0);
        for &sample in &output {
            assert!((sample - 1.0).abs() < 1e-6, "not saturated: {sample}");
        }
    }
}
