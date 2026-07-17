//! Mean-square integrator — the momentary / short-term / integrated / LRA
//! time windows of BS.1770-4 and EBU Tech 3342.
//!
//! # The 100 ms sub-block grid
//!
//! Every window in the standard is a whole number of **100 ms** blocks:
//!
//! - **Momentary** = the last **4** sub-blocks (400 ms).
//! - **Short-term** = the last **30** sub-blocks (3 s).
//! - **Integrated** = a gated running mean of 400 ms blocks stepped every
//!   100 ms (75 % overlap).
//! - **LRA** = a gated percentile spread of the 3 s short-term values, sampled
//!   every 100 ms.
//!
//! So instead of buffering raw samples (400 ms at 48 kHz is ~19 k floats *per
//! channel*), the integrator sums each channel-weighted power sample into the
//! current sub-block, and on the 100 ms boundary rolls a fixed 30-slot ring of
//! sub-block mean powers.  Momentary and short-term are means over the tail of
//! that ring; integrated and LRA feed the gated [`LoudnessHistogram`]s.  Memory
//! is O(1) in session length — a 3-hour master costs the same as 3 seconds.
//!
//! # Real-time safety
//!
//! [`LoudnessIntegrator::push`] does a running add and, once per 100 ms, a
//! handful of ring reads plus one histogram increment — no allocation, no lock,
//! no panic.

use crate::lra::LoudnessHistogram;

/// Sub-blocks per momentary (400 ms) window.
const MOMENTARY_SUBBLOCKS: usize = 4;
/// Sub-blocks per short-term (3 s) window.
const SHORT_TERM_SUBBLOCKS: usize = 30;
/// Loudness offset of the BS.1770 formula: `L = −0.691 + 10·log10(power)`.
const REF_OFFSET_DB: f64 = -0.691;

/// Convert a mean-square power to loudness (LUFS); `−∞` for non-positive power.
#[inline]
fn loudness_from_power(power: f64) -> f32 {
    if power <= 0.0 {
        f32::NEG_INFINITY
    } else {
        (REF_OFFSET_DB + 10.0 * power.log10()) as f32
    }
}

/// Accumulates channel-weighted power into the BS.1770 time windows.
///
/// The caller (a [`crate::LoudnessMeter`]) feeds one **already channel-weighted
/// mean-square sample** per audio frame — i.e. `Σ_c G_c · y_c²` where `y_c` is
/// the K-weighted sample of channel `c` and `G_c` its BS.1770 weight.
pub struct LoudnessIntegrator {
    /// Ring of the last [`SHORT_TERM_SUBBLOCKS`] sub-block mean powers.
    subblock_power: [f64; SHORT_TERM_SUBBLOCKS],
    /// Next write slot in `subblock_power`.
    write_pos: usize,
    /// Total sub-blocks completed (saturates for the "windows full?" checks).
    subblocks_done: u64,
    /// Running sum of the current (partial) sub-block.
    current_sum: f64,
    /// Samples accumulated into the current sub-block.
    current_count: u32,
    /// Samples per 100 ms sub-block at the configured rate.
    samples_per_subblock: u32,
    /// Gated 400 ms-block-loudness histogram → integrated LUFS.
    integrated: LoudnessHistogram,
    /// Gated 3 s short-term-loudness histogram → LRA.
    lra: LoudnessHistogram,
}

impl LoudnessIntegrator {
    /// Build an integrator for `sample_rate` Hz.
    ///
    /// **Allocates** (the two histograms) — call off the audio thread.
    pub fn new(sample_rate: u32) -> Self {
        // 100 ms grid; guard against absurd rates so the divisor is never 0.
        let samples_per_subblock = ((sample_rate as f64 * 0.1).round() as u32).max(1);
        Self {
            subblock_power: [0.0; SHORT_TERM_SUBBLOCKS],
            write_pos: 0,
            subblocks_done: 0,
            current_sum: 0.0,
            current_count: 0,
            samples_per_subblock,
            integrated: LoudnessHistogram::new(),
            lra: LoudnessHistogram::new(),
        }
    }

    /// Feed one channel-weighted mean-square sample.  **Real-time safe.**
    #[inline]
    pub fn push(&mut self, weighted_power: f32) {
        self.current_sum += weighted_power as f64;
        self.current_count += 1;
        if self.current_count >= self.samples_per_subblock {
            self.finish_subblock();
        }
    }

    /// Roll the completed 100 ms sub-block into the ring and update the gated
    /// histograms for any window that is now full.
    fn finish_subblock(&mut self) {
        let mean = self.current_sum / self.current_count as f64;
        self.subblock_power[self.write_pos] = mean;
        self.write_pos = (self.write_pos + 1) % SHORT_TERM_SUBBLOCKS;
        self.subblocks_done = self.subblocks_done.saturating_add(1);
        self.current_sum = 0.0;
        self.current_count = 0;

        // A new 400 ms block completes every 100 ms once four sub-blocks exist;
        // its loudness feeds the integrated gate.
        if self.subblocks_done >= MOMENTARY_SUBBLOCKS as u64 {
            let power = self.window_mean_power(MOMENTARY_SUBBLOCKS);
            self.integrated.add(loudness_from_power(power));
        }
        // Likewise a 3 s short-term value feeds the LRA gate.
        if self.subblocks_done >= SHORT_TERM_SUBBLOCKS as u64 {
            let power = self.window_mean_power(SHORT_TERM_SUBBLOCKS);
            self.lra.add(loudness_from_power(power));
        }
    }

    /// Mean power over the last `window` completed sub-blocks (capped at the
    /// number actually available).
    fn window_mean_power(&self, window: usize) -> f64 {
        let available = (self.subblocks_done as usize).min(SHORT_TERM_SUBBLOCKS);
        let take = window.min(available);
        if take == 0 {
            return 0.0;
        }
        let mut sum = 0.0f64;
        for offset in 1..=take {
            let index = (self.write_pos + SHORT_TERM_SUBBLOCKS - offset) % SHORT_TERM_SUBBLOCKS;
            sum += self.subblock_power[index];
        }
        sum / take as f64
    }

    /// Momentary loudness (400 ms).  Before the first sub-block completes it
    /// reports the partial accumulator so a live meter is not stuck at `−∞`.
    pub fn momentary_lufs(&self) -> f32 {
        if self.subblocks_done == 0 {
            if self.current_count == 0 {
                return f32::NEG_INFINITY;
            }
            return loudness_from_power(self.current_sum / self.current_count as f64);
        }
        loudness_from_power(self.window_mean_power(MOMENTARY_SUBBLOCKS))
    }

    /// Short-term loudness (3 s), over however many sub-blocks exist so far.
    pub fn short_term_lufs(&self) -> f32 {
        if self.subblocks_done == 0 {
            return f32::NEG_INFINITY;
        }
        loudness_from_power(self.window_mean_power(SHORT_TERM_SUBBLOCKS))
    }

    /// Gated integrated loudness (session).  `−∞` until a block clears the
    /// gates.
    pub fn integrated_lufs(&self) -> f32 {
        self.integrated
            .integrated_lufs()
            .unwrap_or(f32::NEG_INFINITY)
    }

    /// Loudness range (LU), 10th–95th percentile per EBU Tech 3342.  `0.0`
    /// until enough gated short-term blocks exist.
    pub fn lra_lu(&self) -> f32 {
        self.lra.range_lu(10.0, 95.0).unwrap_or(0.0)
    }

    /// Clear every window and the gated histograms (a meter `Reset`).
    pub fn reset(&mut self) {
        self.subblock_power = [0.0; SHORT_TERM_SUBBLOCKS];
        self.write_pos = 0;
        self.subblocks_done = 0;
        self.current_sum = 0.0;
        self.current_count = 0;
        self.integrated.reset();
        self.lra.reset();
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny rate keeps sub-blocks short (10 samples each) so tests are fast.
    const SR: u32 = 100;

    /// Push `seconds` worth of a constant weighted power.
    fn push_constant(integrator: &mut LoudnessIntegrator, power: f32, seconds: f32) {
        let samples = (SR as f32 * seconds) as usize;
        for _ in 0..samples {
            integrator.push(power);
        }
    }

    #[test]
    fn empty_reports_silence() {
        let integrator = LoudnessIntegrator::new(SR);
        assert_eq!(integrator.momentary_lufs(), f32::NEG_INFINITY);
        assert_eq!(integrator.short_term_lufs(), f32::NEG_INFINITY);
        assert_eq!(integrator.integrated_lufs(), f32::NEG_INFINITY);
        assert_eq!(integrator.lra_lu(), 0.0);
    }

    #[test]
    fn constant_power_gives_expected_loudness() {
        // power = 1.0 → L = −0.691 + 10·log10(1) = −0.691 LUFS.
        let mut integrator = LoudnessIntegrator::new(SR);
        push_constant(&mut integrator, 1.0, 1.0);
        assert!((integrator.momentary_lufs() - -0.691).abs() < 0.01);
        assert!((integrator.short_term_lufs() - -0.691).abs() < 0.01);
    }

    #[test]
    fn integrated_matches_constant_after_gating() {
        // A steady tone integrates to its own block loudness.
        let mut integrator = LoudnessIntegrator::new(SR);
        // power 0.01 → L = −0.691 + 10·log10(0.01) = −20.691 LUFS.
        push_constant(&mut integrator, 0.01, 5.0);
        let integrated = integrator.integrated_lufs();
        assert!(
            (integrated - -20.691).abs() < 0.2,
            "integrated {integrated} should be ~-20.69"
        );
    }

    #[test]
    fn silence_is_excluded_from_integrated() {
        // 3 s of programme then 3 s of digital silence.  The silent blocks
        // (loudness −∞) are absolute-gated out, so the integrated value tracks
        // the programme only, not a −∞ average.
        let mut integrator = LoudnessIntegrator::new(SR);
        push_constant(&mut integrator, 0.01, 3.0);
        push_constant(&mut integrator, 0.0, 3.0);
        let integrated = integrator.integrated_lufs();
        assert!(
            (integrated - -20.691).abs() < 0.3,
            "integrated {integrated} should ignore the silence"
        );
    }

    #[test]
    fn reset_returns_to_silence() {
        let mut integrator = LoudnessIntegrator::new(SR);
        push_constant(&mut integrator, 1.0, 2.0);
        integrator.reset();
        assert_eq!(integrator.momentary_lufs(), f32::NEG_INFINITY);
        assert_eq!(integrator.integrated_lufs(), f32::NEG_INFINITY);
    }
}
