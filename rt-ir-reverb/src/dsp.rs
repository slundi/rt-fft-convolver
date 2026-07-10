//! Reverb-specific DSP: a pre-delay line, a stereo-width (M/S) operator, and a
//! dB-to-gain helper with an `−∞` floor for the dry/wet faders.
//!
//! The biquad EQ and plain [`db_to_gain`](rt_ir_plugin_core::db_to_gain) live
//! in the shared [`rt_ir_plugin_core::dsp`] module.

/// Below this level a fader is treated as `−∞ dB` (fully muted).
pub const FADER_FLOOR_DB: f32 = -60.0;

/// Convert a fader value in dB to linear gain, mapping anything at or below
/// [`FADER_FLOOR_DB`] to exactly `0.0` (silence) instead of a tiny non-zero
/// gain.  Used for the `Dry` / `Wet` faders (`−∞ … +6 dB`).
#[inline]
pub fn fader_gain(db: f32) -> f32 {
    if db <= FADER_FLOOR_DB {
        0.0
    } else {
        rt_ir_plugin_core::db_to_gain(db)
    }
}

/// Apply a stereo-width (mid/side) scale to a block, in place.
///
/// `width` in `0.0..=2.0`: `1.0` is unchanged, `0.0` collapses to mono, `2.0`
/// doubles the side signal.  Only the side (difference) component is scaled, so
/// the mono sum is preserved.
///
/// ```text
/// M = (L + R)/2         S = (L − R)/2
/// L' = M + width·S      R' = M − width·S
/// ```
#[inline]
pub fn apply_width(left: &mut [f32], right: &mut [f32], width: f32) {
    let n = left.len().min(right.len());
    for i in 0..n {
        let mid = 0.5 * (left[i] + right[i]);
        let side = 0.5 * (left[i] - right[i]) * width;
        left[i] = mid + side;
        right[i] = mid - side;
    }
}

/// A single-channel circular pre-delay line with a run-time-settable delay.
///
/// Capacity is fixed at construction (the maximum delay in samples), so
/// [`process_sample`](Self::process_sample) never allocates.
pub struct PreDelay {
    buf: Vec<f32>,
    write: usize,
    delay: usize,
}

impl PreDelay {
    /// Create a pre-delay able to hold up to `max_delay` samples.
    pub fn new(max_delay: usize) -> Self {
        // One extra slot so a delay of `max_delay` is representable.
        Self {
            buf: vec![0.0; max_delay + 1],
            write: 0,
            delay: 0,
        }
    }

    /// Set the current delay in samples (clamped to the capacity).
    #[inline]
    pub fn set_delay(&mut self, samples: usize) {
        self.delay = samples.min(self.buf.len() - 1);
    }

    /// Current delay in samples.
    #[inline]
    pub fn delay(&self) -> usize {
        self.delay
    }

    /// Clear the delay memory without changing the delay length.
    pub fn reset(&mut self) {
        self.buf.fill(0.0);
        self.write = 0;
    }

    /// Push one input sample and return the sample from `delay` samples ago.
    ///
    /// **Real-time safe** — no allocation.
    #[inline]
    pub fn process_sample(&mut self, x: f32) -> f32 {
        let cap = self.buf.len();
        self.buf[self.write] = x;
        // Read `delay` samples back (delay 0 → the just-written sample).
        let read = (self.write + cap - self.delay) % cap;
        let y = self.buf[read];
        self.write = (self.write + 1) % cap;
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fader_gain_floor_is_silence() {
        assert_eq!(fader_gain(FADER_FLOOR_DB), 0.0);
        assert_eq!(fader_gain(-100.0), 0.0);
        assert!((fader_gain(0.0) - 1.0).abs() < 1e-6);
        assert!((fader_gain(6.0206) - 2.0).abs() < 1e-3);
    }

    #[test]
    fn width_unity_is_identity() {
        let mut l = [0.5, -0.2, 1.0];
        let mut r = [0.1, 0.3, -1.0];
        let (l0, r0) = (l, r);
        apply_width(&mut l, &mut r, 1.0);
        for i in 0..3 {
            assert!((l[i] - l0[i]).abs() < 1e-6);
            assert!((r[i] - r0[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn width_zero_collapses_to_mono() {
        let mut l = [1.0, 0.0];
        let mut r = [-1.0, 0.4];
        apply_width(&mut l, &mut r, 0.0);
        // Both channels become the mid = (L+R)/2.
        assert!((l[0] - 0.0).abs() < 1e-6 && (r[0] - 0.0).abs() < 1e-6);
        assert!((l[1] - 0.2).abs() < 1e-6 && (r[1] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn predelay_delays_by_exact_samples() {
        let mut pd = PreDelay::new(8);
        pd.set_delay(3);
        assert_eq!(pd.delay(), 3);
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out: Vec<f32> = input.iter().map(|&x| pd.process_sample(x)).collect();
        // First 3 outputs are the primed zeros; then the input, delayed by 3.
        assert_eq!(&out[..3], &[0.0, 0.0, 0.0]);
        assert_eq!(&out[3..], &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn predelay_zero_is_passthrough() {
        let mut pd = PreDelay::new(8);
        pd.set_delay(0);
        for x in [0.3, -0.7, 0.9] {
            assert_eq!(pd.process_sample(x), x);
        }
    }

    #[test]
    fn predelay_clamps_to_capacity() {
        let mut pd = PreDelay::new(4);
        pd.set_delay(999);
        assert_eq!(pd.delay(), 4);
    }
}
