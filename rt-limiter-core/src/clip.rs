//! [`CeilingClipper`] — the output brick-wall safety net.
//!
//! If the gain computer is doing its job the output sample peak is already at
//! or below the ceiling and this stage is a no-op.  It exists only to catch
//! the residue the feed-forward path cannot: the tiny overshoot from finite
//! true-peak detection resolution (the oversampling factor does not resolve
//! *every* inter-sample peak exactly), and any user setting where the
//! threshold sits above the ceiling.  A hard clip at the ceiling guarantees
//! the sample-domain output never exceeds it — the last line of defence behind
//! the true-peak-aware gain reduction.
//!
//! It is deliberately a plain hard clip, not a soft knee: this runs *after*
//! limiting, on samples that should almost never reach it, so transparency is
//! irrelevant and an absolute bound is what matters.
//!
//! # Real-time safety
//!
//! [`CeilingClipper::process`] is a single `clamp` — no allocation, lock or
//! panic.

use rt_fft_convolver::db_to_linear;

/// A hard clipper at a configurable linear ceiling.
pub struct CeilingClipper {
    /// Ceiling as a positive linear amplitude; output is clamped to `±ceiling`.
    ceiling: f32,
}

impl CeilingClipper {
    /// Create a clipper with the ceiling given in dBFS.
    pub fn new(ceiling_db: f32) -> Self {
        Self {
            ceiling: db_to_linear(ceiling_db),
        }
    }

    /// Set the ceiling in dBFS.  **Real-time safe** (one `powf`).
    #[inline]
    pub fn set_ceiling_db(&mut self, ceiling_db: f32) {
        self.ceiling = db_to_linear(ceiling_db);
    }

    /// The ceiling as a linear amplitude.
    #[inline]
    pub fn ceiling(&self) -> f32 {
        self.ceiling
    }

    /// Clamp one sample to `±ceiling`.  **Real-time safe.**
    #[inline]
    pub fn process(&self, sample: f32) -> f32 {
        sample.clamp(-self.ceiling, self.ceiling)
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Samples within the ceiling pass through untouched.
    #[test]
    fn within_ceiling_is_transparent() {
        let clipper = CeilingClipper::new(0.0); // ceiling 1.0
        for value in [0.0f32, 0.5, -0.9, 1.0, -1.0] {
            assert_eq!(clipper.process(value), value);
        }
    }

    /// Samples above the ceiling are clamped symmetrically.
    #[test]
    fn above_ceiling_is_clamped() {
        let clipper = CeilingClipper::new(-6.0); // ceiling ≈ 0.501
        let ceiling = db_to_linear(-6.0);
        assert!((clipper.process(2.0) - ceiling).abs() < 1e-6);
        assert!((clipper.process(-2.0) + ceiling).abs() < 1e-6);
    }

    /// The ceiling can be changed live.
    #[test]
    fn set_ceiling_applies() {
        let mut clipper = CeilingClipper::new(0.0);
        clipper.set_ceiling_db(-20.0); // 0.1
        assert!((clipper.process(1.0) - 0.1).abs() < 1e-6);
        assert!((clipper.ceiling() - 0.1).abs() < 1e-6);
    }
}
