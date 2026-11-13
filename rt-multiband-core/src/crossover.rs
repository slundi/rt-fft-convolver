//! Linkwitz-Riley 4th-order (24 dB/oct) crossover with **all-pass phase
//! correction**, splitting one channel into four bands that recombine to a
//! magnitude-flat sum (Phase 0 decision: IIR LR4 in v1).
//!
//! # Why LR4
//!
//! A Linkwitz-Riley 4th-order filter is a cascade of **two identical
//! Butterworth 2nd-order sections** (`Q = 1/√2`).  Each section is −3 dB at the
//! crossover, so the pair is −6 dB, and the lowpass and highpass outputs are
//! everywhere **in phase**.  Their sum is therefore an *all-pass* — flat
//! magnitude, so a two-way split reconstructs the input spectrum exactly (up to
//! the all-pass phase).
//!
//! # The >2-band subtlety
//!
//! With three crossovers `f1 < f2 < f3` the naïve tree leaves a phase error
//! between non-adjacent bands: the high band passes through *fewer* crossover
//! all-passes than the low band.  The fix (Linkwitz's own) is to run each band
//! through an **all-pass matching every crossover below it**:
//!
//! ```text
//!   x ─┬─ LP f3 ─┬─ LP f2 ─┬─ LP f1 ───────────────▶ band0 (low)
//!      │         │         └─ HP f1 ───────────────▶ band1 (low-mid)
//!      │         └─ HP f2 ─── AP f1 ───────────────▶ band2 (high-mid)
//!      └─ HP f3 ─── AP f1 ─── AP f2 ───────────────▶ band3 (high)
//! ```
//!
//! Because the LR4 crossover sum reduces algebraically to a **2nd-order
//! Butterworth all-pass** (`Q = 1/√2`), each correction is a single
//! [`Biquad::allpass`].  Summing all four bands then yields
//! `AP_f1 · AP_f2 · AP_f3 · x` — provably flat magnitude, which is the
//! [`crate::MultibandEngine`] bypass reconstruction guarantee.

use crate::biquad::Biquad;

/// Number of bands produced by the fixed v1 crossover (Phase 0: fixed 4).
pub const NUM_BANDS: usize = 4;

/// Number of crossover frequencies (`NUM_BANDS − 1`).
pub const NUM_CROSSOVERS: usize = NUM_BANDS - 1;

/// The Butterworth quality factor a Linkwitz-Riley 4th-order section is built
/// from (two of these in cascade give the LR4 response, and it is also the
/// `Q` of the matching reconstruction all-pass).
const BUTTERWORTH_Q: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// A Linkwitz-Riley 4th-order filter: two cascaded Butterworth biquads.
#[derive(Debug, Clone, Copy)]
struct Lr4 {
    first: Biquad,
    second: Biquad,
}

impl Lr4 {
    fn lowpass(sample_rate: f32, cutoff_hz: f32) -> Self {
        Self {
            first: Biquad::lowpass(sample_rate, cutoff_hz, BUTTERWORTH_Q),
            second: Biquad::lowpass(sample_rate, cutoff_hz, BUTTERWORTH_Q),
        }
    }

    fn highpass(sample_rate: f32, cutoff_hz: f32) -> Self {
        Self {
            first: Biquad::highpass(sample_rate, cutoff_hz, BUTTERWORTH_Q),
            second: Biquad::highpass(sample_rate, cutoff_hz, BUTTERWORTH_Q),
        }
    }

    fn set_lowpass(&mut self, sample_rate: f32, cutoff_hz: f32) {
        self.first
            .set_lowpass(sample_rate, cutoff_hz, BUTTERWORTH_Q);
        self.second
            .set_lowpass(sample_rate, cutoff_hz, BUTTERWORTH_Q);
    }

    fn set_highpass(&mut self, sample_rate: f32, cutoff_hz: f32) {
        self.first
            .set_highpass(sample_rate, cutoff_hz, BUTTERWORTH_Q);
        self.second
            .set_highpass(sample_rate, cutoff_hz, BUTTERWORTH_Q);
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        self.second.process_sample(self.first.process_sample(x))
    }

    fn reset(&mut self) {
        self.first.reset();
        self.second.reset();
    }
}

/// A four-band Linkwitz-Riley crossover for **one** channel.
///
/// Hold one per channel (they carry independent filter state).  Reserve it in
/// [`Crossover::new`] on a non-real-time thread; [`Crossover::split`] and
/// [`Crossover::set_frequencies`] are allocation-free and RT-safe.
#[derive(Debug, Clone)]
pub struct Crossover {
    sample_rate: f32,
    // Crossover filters, one lowpass + one highpass per split.
    low: [Lr4; NUM_CROSSOVERS],
    high: [Lr4; NUM_CROSSOVERS],
    // Phase-correction all-passes: band2 needs f1; band3 needs f1 and f2.
    ap_band2_f1: Biquad,
    ap_band3_f1: Biquad,
    ap_band3_f2: Biquad,
}

impl Crossover {
    /// Build a crossover at `sample_rate` with the three split frequencies
    /// `freqs` (ascending; they are sorted and clamped internally).
    ///
    /// **Allocates** — off-RT only.
    pub fn new(sample_rate: f32, freqs: [f32; NUM_CROSSOVERS]) -> Self {
        let ordered = Self::order(sample_rate, freqs);
        let [f1, f2, f3] = ordered;
        Self {
            sample_rate,
            low: [
                Lr4::lowpass(sample_rate, f1),
                Lr4::lowpass(sample_rate, f2),
                Lr4::lowpass(sample_rate, f3),
            ],
            high: [
                Lr4::highpass(sample_rate, f1),
                Lr4::highpass(sample_rate, f2),
                Lr4::highpass(sample_rate, f3),
            ],
            ap_band2_f1: Biquad::allpass(sample_rate, f1, BUTTERWORTH_Q),
            ap_band3_f1: Biquad::allpass(sample_rate, f1, BUTTERWORTH_Q),
            ap_band3_f2: Biquad::allpass(sample_rate, f2, BUTTERWORTH_Q),
        }
    }

    /// Clamp each frequency into `(20 Hz, 0.45·fs)` and sort ascending so the
    /// band tree is always well-ordered.
    fn order(sample_rate: f32, freqs: [f32; NUM_CROSSOVERS]) -> [f32; NUM_CROSSOVERS] {
        let ceiling = sample_rate * 0.45;
        let mut ordered = freqs;
        for freq in ordered.iter_mut() {
            *freq = freq.clamp(20.0, ceiling);
        }
        ordered.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());
        ordered
    }

    /// Retune the three crossover frequencies in place, **preserving filter
    /// state** so a live drag does not click.  RT-safe (coefficient math only).
    pub fn set_frequencies(&mut self, freqs: [f32; NUM_CROSSOVERS]) {
        let [f1, f2, f3] = Self::order(self.sample_rate, freqs);
        self.low[0].set_lowpass(self.sample_rate, f1);
        self.low[1].set_lowpass(self.sample_rate, f2);
        self.low[2].set_lowpass(self.sample_rate, f3);
        self.high[0].set_highpass(self.sample_rate, f1);
        self.high[1].set_highpass(self.sample_rate, f2);
        self.high[2].set_highpass(self.sample_rate, f3);
        self.ap_band2_f1
            .set_allpass(self.sample_rate, f1, BUTTERWORTH_Q);
        self.ap_band3_f1
            .set_allpass(self.sample_rate, f1, BUTTERWORTH_Q);
        self.ap_band3_f2
            .set_allpass(self.sample_rate, f2, BUTTERWORTH_Q);
    }

    /// Split one input sample into its four band contributions
    /// `[low, low-mid, high-mid, high]`.  Their sum is magnitude-flat with
    /// respect to the input (all-pass reconstruction).
    ///
    /// # Real-time safety
    /// No allocation, no locks, no panics.
    #[inline]
    pub fn split(&mut self, x: f32) -> [f32; NUM_BANDS] {
        // Top-down tree (see module docs).
        let low3 = self.low[2].process(x);
        let high3 = self.high[2].process(x);
        let low2 = self.low[1].process(low3);
        let high2 = self.high[1].process(low3);
        let band0 = self.low[0].process(low2);
        let band1 = self.high[0].process(low2);
        // Phase-correct the upper bands for every crossover below them.
        let band2 = self.ap_band2_f1.process_sample(high2);
        let band3 = self
            .ap_band3_f2
            .process_sample(self.ap_band3_f1.process_sample(high3));
        [band0, band1, band2, band3]
    }

    /// Clear all filter memory (e.g. on a transport relocate).
    pub fn reset(&mut self) {
        for filter in self.low.iter_mut().chain(self.high.iter_mut()) {
            filter.reset();
        }
        self.ap_band2_f1.reset();
        self.ap_band3_f1.reset();
        self.ap_band3_f2.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{make_noise, make_sine, rms};

    const SR: f32 = 48_000.0;
    const FREQS: [f32; NUM_CROSSOVERS] = [200.0, 1_000.0, 5_000.0];

    /// Sum of the four band outputs for a whole buffer.
    fn reconstruct(crossover: &mut Crossover, input: &[f32]) -> Vec<f32> {
        input
            .iter()
            .map(|&x| crossover.split(x).iter().sum())
            .collect()
    }

    #[test]
    fn bands_sum_to_flat_magnitude_broadband() {
        // All-pass reconstruction: RMS of (band0+band1+band2+band3) equals RMS
        // of the input to within a hair, for full-band noise.
        let mut crossover = Crossover::new(SR, FREQS);
        let input = make_noise(48_000, 0xC0FFEE);
        let summed = reconstruct(&mut crossover, &input);
        // Skip the filter warm-up transient.
        let in_rms = rms(&input[2_000..]);
        let out_rms = rms(&summed[2_000..]);
        let err_db = 20.0 * (out_rms / in_rms).log10();
        assert!(err_db.abs() < 0.1, "reconstruction error {err_db} dB");
    }

    #[test]
    fn reconstruction_flat_across_probe_tones() {
        // Sine at every band and every crossover edge must reconstruct at unity
        // magnitude — the all-pass correction is what makes the edges flat.
        for &freq in &[50.0, 200.0, 600.0, 1_000.0, 3_000.0, 5_000.0, 12_000.0] {
            let mut crossover = Crossover::new(SR, FREQS);
            let input = make_sine(freq, SR, 24_000);
            let summed = reconstruct(&mut crossover, &input);
            let gain = rms(&summed[12_000..]) / rms(&input[12_000..]);
            let err_db = 20.0 * gain.log10();
            assert!(
                err_db.abs() < 0.2,
                "at {freq} Hz reconstruction {err_db} dB"
            );
        }
    }

    #[test]
    fn low_band_carries_low_energy_high_band_carries_high() {
        // Sanity: a 50 Hz tone lands in band0, a 12 kHz tone lands in band3.
        let mut crossover = Crossover::new(SR, FREQS);
        let low_tone = make_sine(50.0, SR, 24_000);
        let mut low_energy = [0.0f32; NUM_BANDS];
        for &x in &low_tone[12_000..] {
            for (acc, band) in low_energy.iter_mut().zip(crossover.split(x)) {
                *acc += band * band;
            }
        }
        let low_dominant = low_energy
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(low_dominant, 0, "50 Hz did not dominate band0");

        let mut crossover = Crossover::new(SR, FREQS);
        let high_tone = make_sine(12_000.0, SR, 24_000);
        let mut high_energy = [0.0f32; NUM_BANDS];
        for &x in &high_tone[12_000..] {
            for (acc, band) in high_energy.iter_mut().zip(crossover.split(x)) {
                *acc += band * band;
            }
        }
        let high_dominant = high_energy
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(
            high_dominant,
            NUM_BANDS - 1,
            "12 kHz did not dominate band3"
        );
    }

    #[test]
    fn retune_preserves_reconstruction() {
        // After a live frequency change the sum is still flat (state preserved,
        // coefficients updated).
        let mut crossover = Crossover::new(SR, FREQS);
        let warm = make_noise(4_000, 1);
        reconstruct(&mut crossover, &warm);
        crossover.set_frequencies([120.0, 800.0, 3_000.0]);
        let input = make_noise(48_000, 2);
        let summed = reconstruct(&mut crossover, &input);
        let err_db = 20.0 * (rms(&summed[2_000..]) / rms(&input[2_000..])).log10();
        assert!(err_db.abs() < 0.1, "post-retune reconstruction {err_db} dB");
    }

    #[test]
    fn unordered_frequencies_are_sorted() {
        // Passing frequencies out of order must not break the band tree.
        let mut crossover = Crossover::new(SR, [5_000.0, 200.0, 1_000.0]);
        let input = make_noise(24_000, 7);
        let summed = reconstruct(&mut crossover, &input);
        let err_db = 20.0 * (rms(&summed[2_000..]) / rms(&input[2_000..])).log10();
        assert!(err_db.abs() < 0.1, "reconstruction {err_db} dB");
    }
}
