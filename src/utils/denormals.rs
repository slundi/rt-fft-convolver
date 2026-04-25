//! Anti-denormal protection for the real-time audio render thread.
//!
//! # The problem
//!
//! Denormal (subnormal) IEEE 754 floats are values too small to be represented
//! with a normalised mantissa — roughly `|x| < 1.18e-38` for `f32`.  Most FPUs
//! handle them 10–100× slower because they fall back to a microcode assist
//! path.  In audio DSP this penalty shows up silently: after a note-off, filter
//! ring-down or reverb tail drives sample values toward zero, and the CPU load
//! spikes even though "nothing is happening".
//!
//! # Solutions provided
//!
//! | Mechanism | Scope | Cost |
//! |-----------|-------|------|
//! | [`DenormalGuard`] | per-thread FPU state | one MXCSR/FPCR read+write on entry and exit |
//! | [`flush_to_zero`] | per-value software check | one branch + compare |
//!
//! ## [`DenormalGuard`] (recommended)
//!
//! A RAII guard that enables **FTZ** (Flush-To-Zero) and, on x86, also **DAZ**
//! (Denormals-Are-Zero) for the current thread.  Drop restores the previous
//! FPU state.  Wrap your entire audio render callback with it:
//!
//! ```rust,ignore
//! # use rt_fft_convolver::utils::denormals::DenormalGuard;
//! fn audio_callback(buffer: &mut [f32]) {
//!     let _guard = DenormalGuard::new(); // FTZ enabled
//!     // ... process buffer ...
//! }   // FTZ restored here
//! ```
//!
//! **FTZ**: subnormal *outputs* of FPU operations are flushed to `±0.0`.
//! **DAZ**: subnormal *inputs* are treated as `±0.0` before the operation.
//! Together they guarantee the FPU never operates on a denormal value.
//!
//! ## [`flush_to_zero`] (software fallback)
//!
//! A pure-Rust per-value check that is cross-platform but only prevents
//! *propagation* — the subnormal was already computed, so the FPU slowdown
//! already happened.  Prefer the guard; use this only when FPU-mode changes
//! are not permitted (e.g., inside a guest plugin context).
//!
//! # Platform support
//!
//! | Platform | Mechanism | Notes |
//! |----------|-----------|-------|
//! | x86 / x86_64 | MXCSR FTZ (bit 15) + DAZ (bit 6) | SSE required; always present on x86_64 |
//! | AArch64 | FPCR FZ (bit 24) | ARM FPU flush-to-zero |
//! | Other | no-op guard | safe but unprotected |

// ─── x86 / x86_64 constants ──────────────────────────────────────────────────

/// MXCSR bit 15: flush subnormal *outputs* to zero.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const MXCSR_FTZ: u32 = 1 << 15;

/// MXCSR bit 6: treat subnormal *inputs* as zero (x86 SSE only).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const MXCSR_DAZ: u32 = 1 << 6;

// ─── AArch64 constants ────────────────────────────────────────────────────────

/// FPCR bit 24: flush subnormal results to zero (ARM).
#[cfg(target_arch = "aarch64")]
const FPCR_FZ: u64 = 1 << 24;

// ─── x86 helpers ─────────────────────────────────────────────────────────────

/// Read the MXCSR register using `stmxcsr`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn read_mxcsr() -> u32 {
    let mut val: u32 = 0;
    // SAFETY: `stmxcsr` stores MXCSR into a 32-bit memory location.  The
    // address is valid (stack-allocated), no other memory is touched.
    unsafe {
        core::arch::asm!(
            "stmxcsr [{ptr}]",
            ptr = in(reg) &mut val as *mut u32,
            options(nostack, preserves_flags),
        );
    }
    val
}

/// Write the MXCSR register using `ldmxcsr`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn write_mxcsr(val: u32) {
    // SAFETY: `ldmxcsr` loads MXCSR from a 32-bit memory location.  The
    // address is valid (stack-allocated), no other memory is touched.
    unsafe {
        core::arch::asm!(
            "ldmxcsr [{ptr}]",
            ptr = in(reg) &val as *const u32,
            options(nostack, preserves_flags),
        );
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Clamps a subnormal `f32` to `0.0`; passes all other values through unchanged.
///
/// This is a **software** check — the FPU already paid the denormal penalty
/// when producing `x`.  It prevents the denormal from *propagating* into the
/// next computation.  Prefer [`DenormalGuard`] to eliminate the penalty at
/// the source.
///
/// # Example
///
/// ```rust,ignore
/// # use rt_fft_convolver::utils::denormals::flush_to_zero;
/// let tiny = 1.0e-40_f32; // subnormal
/// assert_eq!(flush_to_zero(tiny), 0.0);
/// assert_eq!(flush_to_zero(1.0_f32), 1.0);
/// ```
#[inline(always)]
pub fn flush_to_zero(x: f32) -> f32 {
    if x.is_subnormal() { 0.0 } else { x }
}

/// RAII guard that enables FTZ (+ DAZ on x86) for the **current thread**.
///
/// The previous FPU control state is saved on construction and restored on
/// drop, making guards safe to nest and safe to use inside plugin hosts that
/// may have their own FPU settings.
///
/// # Real-time safety
///
/// `new()` and `drop()` each perform **one** privileged register read and one
/// write — O(1), no allocation, no locking.  Both are safe to call from the
/// audio render thread.
///
/// # Example
///
/// ```rust,ignore
/// # use rt_fft_convolver::utils::denormals::DenormalGuard;
/// fn process(buf: &mut [f32]) {
///     let _guard = DenormalGuard::new();
///     for s in buf.iter_mut() {
///         *s *= 0.5;
///     }
/// } // guard dropped → previous FPU state restored
/// ```
pub struct DenormalGuard {
    /// Saved MXCSR value (x86 / x86_64).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    prev_mxcsr: u32,

    /// Saved FPCR value (AArch64).
    #[cfg(target_arch = "aarch64")]
    prev_fpcr: u64,
}

impl DenormalGuard {
    /// Enable FTZ (and DAZ on x86) for the current thread, returning a guard
    /// that will restore the previous state when dropped.
    #[inline]
    pub fn new() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            // SAFETY: `read_mxcsr` / `write_mxcsr` only access the MXCSR
            // register of the current thread via `stmxcsr` / `ldmxcsr`.
            let prev_mxcsr = unsafe { read_mxcsr() };
            unsafe { write_mxcsr(prev_mxcsr | MXCSR_FTZ | MXCSR_DAZ) };
            Self { prev_mxcsr }
        }

        #[cfg(target_arch = "aarch64")]
        {
            let prev_fpcr: u64;
            // SAFETY: reading FPCR is always permitted at EL0 (user space) on
            // AArch64.  Writing only the FZ bit cannot affect correctness of
            // other threads (FPCR is per-thread on all OSes we target).
            unsafe {
                core::arch::asm!("mrs {0}, fpcr", out(reg) prev_fpcr);
                core::arch::asm!("msr fpcr, {0}", in(reg) prev_fpcr | FPCR_FZ);
            }
            Self { prev_fpcr }
        }

        // On all other architectures the guard is a harmless no-op.
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        Self {}
    }
}

impl Default for DenormalGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for DenormalGuard {
    #[inline]
    fn drop(&mut self) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        // SAFETY: restoring a value we previously read from the same register.
        unsafe {
            write_mxcsr(self.prev_mxcsr);
        }

        #[cfg(target_arch = "aarch64")]
        // SAFETY: restoring a value we previously read from the same register.
        unsafe {
            core::arch::asm!("msr fpcr, {0}", in(reg) self.prev_fpcr);
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── flush_to_zero ─────────────────────────────────────────────────────────

    /// Normal positive and negative values must pass through unchanged.
    #[test]
    fn normal_values_pass_through() {
        for &v in &[
            1.0_f32,
            -1.0,
            0.5,
            -0.5,
            1.18e-38,
            f32::MAX,
            f32::MIN_POSITIVE,
        ] {
            assert_eq!(
                flush_to_zero(v),
                v,
                "normal value {v} was unexpectedly changed"
            );
        }
    }

    /// Zero (positive and negative) is not subnormal and must pass through.
    #[test]
    fn zero_passes_through() {
        assert_eq!(flush_to_zero(0.0_f32), 0.0);
        assert_eq!(flush_to_zero(-0.0_f32), -0.0_f32);
    }

    /// A subnormal value must be flushed to exactly `0.0`.
    #[test]
    fn subnormal_is_flushed() {
        // The smallest positive subnormal f32.
        let subnormal = f32::from_bits(1);
        assert!(subnormal.is_subnormal());
        assert_eq!(flush_to_zero(subnormal), 0.0);
    }

    /// A subnormal just below MIN_POSITIVE must also be flushed.
    #[test]
    fn near_min_positive_subnormal_is_flushed() {
        // One ULP below the smallest normal — still subnormal.
        let nearly_normal = f32::MIN_POSITIVE - f32::from_bits(1);
        if nearly_normal.is_subnormal() {
            assert_eq!(flush_to_zero(nearly_normal), 0.0);
        }
    }

    /// `flush_to_zero` must not alter `NaN` or `±Inf`.
    #[test]
    fn special_values_pass_through() {
        let nan = f32::NAN;
        let inf = f32::INFINITY;
        let neg_inf = f32::NEG_INFINITY;
        // NaN != NaN by IEEE, so check bits directly.
        assert!(flush_to_zero(nan).is_nan());
        assert_eq!(flush_to_zero(inf), inf);
        assert_eq!(flush_to_zero(neg_inf), neg_inf);
    }

    // ── DenormalGuard ─────────────────────────────────────────────────────────

    /// Guard must not panic on construction or drop.
    #[test]
    fn guard_constructs_and_drops() {
        let _guard = DenormalGuard::new();
        // If we reach this point the guard constructed successfully.
    }

    /// Nested guards must not corrupt the FPU state.
    #[test]
    fn nested_guards_restore_state() {
        {
            let _outer = DenormalGuard::new();
            {
                let _inner = DenormalGuard::new();
            } // inner dropped, state restored to outer's saved value
        } // outer dropped, state restored to pre-outer value
        // reaching here without a crash is sufficient verification.
    }

    /// On x86_64, verify that FTZ and DAZ bits are actually set while the guard
    /// is live, and cleared (restored) after drop.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_ftz_daz_bits_are_set_and_restored() {
        // Save a known baseline without FTZ/DAZ.
        let baseline = unsafe { read_mxcsr() } & !(MXCSR_FTZ | MXCSR_DAZ);
        unsafe { write_mxcsr(baseline) };

        {
            let _guard = DenormalGuard::new();
            let live = unsafe { read_mxcsr() };
            assert!(live & MXCSR_FTZ != 0, "FTZ bit not set while guard is live");
            assert!(live & MXCSR_DAZ != 0, "DAZ bit not set while guard is live");
        }

        let after = unsafe { read_mxcsr() };
        assert_eq!(
            after & (MXCSR_FTZ | MXCSR_DAZ),
            baseline & (MXCSR_FTZ | MXCSR_DAZ),
            "FTZ/DAZ bits not restored after guard drop"
        );
    }

    /// On x86_64, verify that a subnormal *output* is flushed to zero when
    /// FTZ is active.  We manufacture a subnormal result via bit manipulation
    /// to avoid the compiler constant-folding it away.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_subnormal_output_is_flushed_by_hardware() {
        // A very small normal float whose square is subnormal.
        // sqrt(MIN_POSITIVE) ≈ 1.08e-19, squaring gives MIN_POSITIVE ≈ 1.18e-38.
        // But MIN_POSITIVE itself is normal.  We need something whose square is
        // actually subnormal: use 1e-20, whose square ≈ 1e-40 (subnormal).
        let small: f32 = 1.0e-20;

        // Without guard: result should be subnormal (not zero).
        let without_guard = small * small;
        // (On some hosts the CPU may already have FTZ set; skip if so.)
        if without_guard == 0.0 {
            return;
        }
        assert!(
            without_guard.is_subnormal(),
            "test precondition: product must be subnormal"
        );

        // With guard: the FPU flushes subnormal outputs to zero.
        let with_guard = {
            let _guard = DenormalGuard::new();
            // Use black_box to prevent the compiler from hoisting the multiply
            // outside the guard scope.
            std::hint::black_box(small) * std::hint::black_box(small)
        };
        assert_eq!(
            with_guard, 0.0,
            "FTZ hardware flush did not zero the subnormal product"
        );
    }
}
