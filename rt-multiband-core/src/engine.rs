//! The integrated multi-band engine — per-channel [`Crossover`], four
//! [`BandProcessor`]s, and the **band-sum reconstruction** (Phase 1).
//!
//! Signal flow per stereo sample:
//!
//! ```text
//!   (l,r) ─▶ ×input ─┬─▶ crossover ─▶ [band0..band3].process ─▶ Σ (solo/mute) ─▶ wet
//!                    └────────────────────────────────────────────── dry
//!   out = ( mix·wet + (1−mix)·dry ) · output
//! ```
//!
//! **Band-sum reconstruction guarantee:** with every band in `Bypass` (or with
//! `mix = 0`) the Linkwitz-Riley bands recombine to a magnitude-flat copy of
//! the input — the property the crossover's all-pass correction buys and the
//! [`MultibandEngine::process`] bypass test verifies.
//!
//! # Real-time safety
//!
//! [`MultibandEngine::new`] reserves everything; [`MultibandEngine::process`]
//! and [`MultibandEngine::process_block`] allocate/lock/panic never.
//! `process_block` installs a `DenormalGuard` for the block so quiet passages
//! do not push the crossover biquad states or envelope tails into subnormal
//! arithmetic.

use rt_fft_convolver::{DenormalGuard, db_to_linear};

use crate::band::BandProcessor;
use crate::crossover::{Crossover, NUM_BANDS, NUM_CROSSOVERS};
use crate::gain_computer::{BandMode, StereoLink};
use crate::smoother::ParamSmoother;

/// Glide time for the global input/output/mix smoothers, in ms.
const GLOBAL_GLIDE_MS: f32 = 10.0;

/// A four-band stereo compressor / expander / gate engine.
#[derive(Debug, Clone)]
pub struct MultibandEngine {
    crossover: [Crossover; 2],
    bands: [BandProcessor; NUM_BANDS],
    input_gain: ParamSmoother,
    output_gain: ParamSmoother,
    mix: ParamSmoother,
}

impl MultibandEngine {
    /// Build an engine at `sample_rate` with the three crossover `freqs`.
    ///
    /// Every band starts as a gentle 2:1 compressor at −18 dBFS; the caller
    /// (Phase 2 plugin) drives the per-band setters afterwards.  Bands are
    /// unity-summing at construction (`Bypass` is available per band).
    ///
    /// **Allocates** — off-RT only.
    pub fn new(sample_rate: f32, freqs: [f32; NUM_CROSSOVERS]) -> Self {
        let make_band = || {
            BandProcessor::new(
                sample_rate,
                10.0,
                100.0,
                5.0,
                0.5,
                BandMode::Compressor,
                -18.0,
                2.0,
                6.0,
            )
        };
        Self {
            crossover: [
                Crossover::new(sample_rate, freqs),
                Crossover::new(sample_rate, freqs),
            ],
            bands: std::array::from_fn(|_| make_band()),
            input_gain: ParamSmoother::new(sample_rate, GLOBAL_GLIDE_MS, 1.0),
            output_gain: ParamSmoother::new(sample_rate, GLOBAL_GLIDE_MS, 1.0),
            mix: ParamSmoother::new(sample_rate, GLOBAL_GLIDE_MS, 1.0),
        }
    }

    /// Number of bands (fixed at 4 in v1).
    pub const fn num_bands(&self) -> usize {
        NUM_BANDS
    }

    /// Mutable access to band `index` (`0..4`) for its setters.  RT-safe.
    #[inline]
    pub fn band_mut(&mut self, index: usize) -> &mut BandProcessor {
        &mut self.bands[index]
    }

    /// Read-only access to band `index`.
    #[inline]
    pub fn band(&self, index: usize) -> &BandProcessor {
        &self.bands[index]
    }

    /// Retune the crossover frequencies on both channels.  RT-safe.
    pub fn set_frequencies(&mut self, freqs: [f32; NUM_CROSSOVERS]) {
        for crossover in &mut self.crossover {
            crossover.set_frequencies(freqs);
        }
    }

    /// Set the stereo detector linking on every band.  RT-safe.
    pub fn set_link(&mut self, link: StereoLink) {
        for band in &mut self.bands {
            band.set_link(link);
        }
    }

    /// Set the input gain (dB).  RT-safe.
    #[inline]
    pub fn set_input_gain_db(&mut self, gain_db: f32) {
        self.input_gain.set_target(db_to_linear(gain_db));
    }

    /// Set the output gain (dB).  RT-safe.
    #[inline]
    pub fn set_output_gain_db(&mut self, gain_db: f32) {
        self.output_gain.set_target(db_to_linear(gain_db));
    }

    /// Set the dry/wet mix (`0` = dry, `1` = fully processed).  RT-safe.
    #[inline]
    pub fn set_mix(&mut self, mix: f32) {
        self.mix.set_target(mix.clamp(0.0, 1.0));
    }

    /// Whether any band is soloed (only those bands then contribute).
    #[inline]
    fn any_soloed(&self) -> bool {
        self.bands.iter().any(|band| band.solo)
    }

    /// Clear all filter/envelope state and settle the smoothers.
    pub fn reset(&mut self) {
        for crossover in &mut self.crossover {
            crossover.reset();
        }
        for band in &mut self.bands {
            band.reset();
        }
        self.input_gain.set_immediate(self.input_gain.current());
        self.output_gain.set_immediate(self.output_gain.current());
        self.mix.set_immediate(self.mix.current());
    }

    /// Process one stereo sample and return `(left, right)`.  RT-safe.
    #[inline]
    pub fn process(&mut self, left: f32, right: f32) -> (f32, f32) {
        let input = self.input_gain.next_value();
        let (dry_left, dry_right) = (left * input, right * input);

        let bands_left = self.crossover[0].split(dry_left);
        let bands_right = self.crossover[1].split(dry_right);

        let soloing = self.any_soloed();
        let mut wet_left = 0.0;
        let mut wet_right = 0.0;
        for (index, band) in self.bands.iter_mut().enumerate() {
            let (band_left, band_right) = band.process(bands_left[index], bands_right[index]);
            // Solo overrides mute: when soloing, only soloed bands pass; else
            // every non-muted band passes.
            let contributes = if soloing { band.solo } else { !band.mute };
            if contributes {
                wet_left += band_left;
                wet_right += band_right;
            }
        }

        let mix = self.mix.next_value();
        let output = self.output_gain.next_value();
        let out_left = (mix * wet_left + (1.0 - mix) * dry_left) * output;
        let out_right = (mix * wet_right + (1.0 - mix) * dry_right) * output;
        (out_left, out_right)
    }

    /// Process an interleaved-by-channel stereo block in place.  Installs a
    /// `DenormalGuard` for the whole block.  RT-safe.
    pub fn process_block(&mut self, left: &mut [f32], right: &mut [f32]) {
        let _guard = DenormalGuard::new();
        let len = left.len().min(right.len());
        for index in 0..len {
            let (out_left, out_right) = self.process(left[index], right[index]);
            left[index] = out_left;
            right[index] = out_right;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{make_noise, make_sine, rms};

    const SR: f32 = 48_000.0;
    const FREQS: [f32; NUM_CROSSOVERS] = [200.0, 1_000.0, 5_000.0];

    fn bypass_all(engine: &mut MultibandEngine) {
        for index in 0..NUM_BANDS {
            engine
                .band_mut(index)
                .gain_computer_mut()
                .set_mode(BandMode::Bypass);
        }
    }

    #[test]
    fn all_bands_bypassed_reconstruct_the_input() {
        // The Phase 1 band-sum guarantee: bypassed bands sum to a magnitude-flat
        // copy of the input.
        let mut engine = MultibandEngine::new(SR, FREQS);
        bypass_all(&mut engine);
        let input = make_noise(48_000, 0xABCDEF);
        let mut left = input.clone();
        let mut right = input.clone();
        engine.process_block(&mut left, &mut right);
        let err_db = 20.0 * (rms(&left[4_000..]) / rms(&input[4_000..])).log10();
        assert!(err_db.abs() < 0.1, "bypass reconstruction {err_db} dB");
    }

    #[test]
    fn mix_zero_is_the_dry_signal() {
        // Even with the bands compressing hard, mix=0 returns dry (post input
        // gain), so the crossover all-pass phase is bypassed entirely.
        let mut engine = MultibandEngine::new(SR, FREQS);
        for index in 0..NUM_BANDS {
            let computer = engine.band_mut(index).gain_computer_mut();
            computer.set_mode(BandMode::Compressor);
            computer.set_threshold_db(-40.0);
            computer.set_ratio(8.0);
        }
        engine.set_mix(0.0);
        // Settle the mix smoother well past its glide (10 ms ≈ 480 samples).
        for _ in 0..6_000 {
            engine.process(0.0, 0.0);
        }
        let input = make_sine(500.0, SR, 4_000);
        for &x in &input {
            let (left, _) = engine.process(x, x);
            let recent = x; // dry equals input at unity input gain
            assert!(
                (left - recent).abs() < 1e-3,
                "mix=0 not dry: {left} vs {recent}"
            );
        }
    }

    #[test]
    fn low_band_compression_reduces_low_energy_only() {
        // Compress only the low band hard; a low tone is reduced, a high tone
        // passes.  Confirms bands are independently routable.
        let low_gr = {
            let mut engine = MultibandEngine::new(SR, FREQS);
            bypass_all(&mut engine);
            let computer = engine.band_mut(0).gain_computer_mut();
            computer.set_mode(BandMode::Compressor);
            computer.set_threshold_db(-30.0);
            computer.set_ratio(8.0);
            let tone = make_sine(60.0, SR, 24_000);
            let out: Vec<f32> = tone.iter().map(|&x| engine.process(x, x).0).collect();
            rms(&out[12_000..]) / rms(&tone[12_000..])
        };
        let high_gr = {
            let mut engine = MultibandEngine::new(SR, FREQS);
            bypass_all(&mut engine);
            let computer = engine.band_mut(0).gain_computer_mut();
            computer.set_mode(BandMode::Compressor);
            computer.set_threshold_db(-30.0);
            computer.set_ratio(8.0);
            let tone = make_sine(10_000.0, SR, 24_000);
            let out: Vec<f32> = tone.iter().map(|&x| engine.process(x, x).0).collect();
            rms(&out[12_000..]) / rms(&tone[12_000..])
        };
        assert!(
            low_gr < 0.7,
            "low band did not compress the low tone: {low_gr}"
        );
        assert!(high_gr > 0.9, "low band bled into the high tone: {high_gr}");
    }

    #[test]
    fn solo_isolates_a_band() {
        // Solo the high band: a low tone is (almost) silenced in the output.
        let mut engine = MultibandEngine::new(SR, FREQS);
        bypass_all(&mut engine);
        engine.band_mut(NUM_BANDS - 1).solo = true;
        let tone = make_sine(60.0, SR, 24_000);
        let out: Vec<f32> = tone.iter().map(|&x| engine.process(x, x).0).collect();
        assert!(
            rms(&out[12_000..]) < 0.1 * rms(&tone[12_000..]),
            "solo leaked lows"
        );
    }

    #[test]
    fn mute_removes_a_band() {
        // Muting the low band removes low energy from an otherwise-flat sum.
        let mut engine = MultibandEngine::new(SR, FREQS);
        bypass_all(&mut engine);
        engine.band_mut(0).mute = true;
        let tone = make_sine(60.0, SR, 24_000);
        let out: Vec<f32> = tone.iter().map(|&x| engine.process(x, x).0).collect();
        assert!(
            rms(&out[12_000..]) < 0.2 * rms(&tone[12_000..]),
            "mute kept the low band"
        );
    }

    #[test]
    fn output_gain_scales_the_result() {
        let mut engine = MultibandEngine::new(SR, FREQS);
        bypass_all(&mut engine);
        engine.set_output_gain_db(-6.0);
        for _ in 0..2_000 {
            engine.process(0.0, 0.0); // settle
        }
        let input = make_sine(500.0, SR, 8_000);
        let out: Vec<f32> = input.iter().map(|&x| engine.process(x, x).0).collect();
        let gain = rms(&out[4_000..]) / rms(&input[4_000..]);
        assert!(
            (gain - db_to_linear(-6.0)).abs() < 0.02,
            "output gain {gain}"
        );
    }

    #[test]
    fn silence_in_silence_out() {
        let mut engine = MultibandEngine::new(SR, FREQS);
        let mut left = vec![0.0f32; 512];
        let mut right = vec![0.0f32; 512];
        engine.process_block(&mut left, &mut right);
        assert!(left.iter().chain(right.iter()).all(|&s| s == 0.0));
    }
}
