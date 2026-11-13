//! [`Dither`] — output bit-depth reducer with TPDF dither and noise shaping.
//!
//! Gated behind the `dither` Cargo feature (Roadmap-07 Phase 0: the limiter is
//! typically the last stage before export, so re-quantising to a target word
//! length belongs here, but callers that don't export a final bit depth should
//! not pay for it).
//!
//! # Why dither, why shape
//!
//! Truncating a 32-bit float to, say, 16-bit correlates the quantisation error
//! with the signal, which is audible as distortion on quiet fades.  Adding a
//! small amount of **TPDF** (triangular probability density) noise before
//! rounding decorrelates the error, turning distortion into a constant, benign
//! noise floor — the standard mastering-dither result.  A first-order
//! **noise shaper** then feeds the quantisation error back so the added noise
//! is pushed toward high frequencies where the ear is less sensitive.
//!
//! # Real-time safety
//!
//! The RNG is a per-instance xorshift (no `rand` crate, deterministic, no
//! locks).  [`Dither::process`] allocates nothing and never panics.

/// Target output word length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitDepth {
    /// 16-bit (CD / most streaming delivery).
    Bits16,
    /// 20-bit.
    Bits20,
    /// 24-bit (high-resolution delivery).
    Bits24,
}

impl BitDepth {
    /// The quantisation step for a full-scale `±1.0` signal: `2^-(bits-1)`.
    #[inline]
    fn quantum(self) -> f32 {
        let bits = match self {
            BitDepth::Bits16 => 16u32,
            BitDepth::Bits20 => 20,
            BitDepth::Bits24 => 24,
        };
        // `bits − 1` fractional bits either side of zero for signed samples.
        1.0 / (1u32 << (bits - 1)) as f32
    }
}

/// TPDF-dithered, noise-shaped quantiser to a target [`BitDepth`].
pub struct Dither {
    quantum: f32,
    shape: bool,
    /// Fed-back quantisation error (first-order shaper state).
    error: f32,
    /// xorshift32 RNG state (non-zero).
    rng: u32,
}

impl Dither {
    /// Create a dither stage for `depth`, with optional first-order noise
    /// shaping.  `seed` seeds the RNG (any value; `0` is remapped so the
    /// xorshift never gets stuck at zero).
    pub fn new(depth: BitDepth, shape: bool, seed: u32) -> Self {
        Self {
            quantum: depth.quantum(),
            shape,
            error: 0.0,
            rng: if seed == 0 { 0x1234_5678 } else { seed },
        }
    }

    /// Change the target bit depth.  **Real-time safe.**
    #[inline]
    pub fn set_depth(&mut self, depth: BitDepth) {
        self.quantum = depth.quantum();
    }

    /// Enable or disable noise shaping.  **Real-time safe.**
    #[inline]
    pub fn set_shaping(&mut self, shape: bool) {
        self.shape = shape;
    }

    /// One xorshift32 step mapped to a uniform value in `[-0.5, 0.5)`.
    #[inline]
    fn uniform(&mut self) -> f32 {
        let mut state = self.rng;
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        self.rng = state;
        // Top 24 bits → [0, 1), then centre on zero.
        (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
    }

    /// Dither, (optionally) shape, and quantise one sample.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn process(&mut self, sample: f32) -> f32 {
        // Feed back the previous quantisation error (noise shaping) so the
        // error spectrum is tilted toward high frequencies.
        let shaped = if self.shape {
            sample + self.error
        } else {
            sample
        };
        // TPDF dither: sum of two independent uniforms spans one quantum with
        // a triangular density.
        let dither = (self.uniform() + self.uniform()) * self.quantum;
        let noisy = shaped + dither;
        // Round to the nearest quantisation level.
        let quantised = (noisy / self.quantum).round() * self.quantum;
        // Remember the error we just made (pre-dither) for the next sample.
        self.error = shaped - quantised;
        quantised
    }

    /// Reset the shaper error (RNG state is left running).
    pub fn reset(&mut self) {
        self.error = 0.0;
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The output always lands on a multiple of the quantum.
    #[test]
    fn output_is_quantised() {
        let mut dither = Dither::new(BitDepth::Bits16, false, 1);
        let quantum = BitDepth::Bits16.quantum();
        for step in 0..1000 {
            let input = (step as f32 / 1000.0) * 2.0 - 1.0;
            let output = dither.process(input);
            let levels = output / quantum;
            assert!(
                (levels - levels.round()).abs() < 1e-3,
                "output {output} is not a quantum multiple"
            );
        }
    }

    /// Quantisation error stays within a couple of quanta (dither + rounding).
    #[test]
    fn error_is_bounded() {
        let mut dither = Dither::new(BitDepth::Bits16, true, 42);
        let quantum = BitDepth::Bits16.quantum();
        for step in 0..2000 {
            let input = 0.3 * (step as f32 * 0.05).sin();
            let output = dither.process(input);
            assert!(
                (output - input).abs() < 4.0 * quantum,
                "error {} exceeds 4 quanta",
                (output - input).abs()
            );
        }
    }

    /// A finer bit depth has a smaller quantum, so its error floor is lower.
    #[test]
    fn deeper_bits_are_finer() {
        assert!(BitDepth::Bits24.quantum() < BitDepth::Bits16.quantum());
        assert!(BitDepth::Bits20.quantum() < BitDepth::Bits16.quantum());
    }

    /// A zero seed is remapped so the RNG never gets stuck at zero output.
    #[test]
    fn zero_seed_still_dithers() {
        let mut dither = Dither::new(BitDepth::Bits16, false, 0);
        let first = dither.uniform();
        let second = dither.uniform();
        assert!(first != 0.0 || second != 0.0);
    }

    /// Live setters and reset work.
    #[test]
    fn setters_and_reset() {
        let mut dither = Dither::new(BitDepth::Bits16, true, 7);
        dither.process(0.5);
        dither.set_depth(BitDepth::Bits24);
        dither.set_shaping(false);
        dither.reset();
        assert_eq!(dither.error, 0.0);
    }
}
