//! [`LoudnessMeter`] — the whole BS.1770-4 / EBU R128 measurement chain for
//! one mono or stereo signal.
//!
//! # Signal flow
//!
//! ```text
//!            ┌──────────────┐  yₖ   ┌───────────────────────────┐
//!  x_c  ───▶ │ K-weighting  │ ────▶ │ Σ_c G_c·yₖ²  (weighted)   │──▶ integrator
//!  (per ch)  │  (per ch)    │       │ Σ_c   xₖ²    (unweighted)  │──▶ RMS
//!            └──────────────┘       └───────────────────────────┘
//!  x_c  ───────────────────────────▶ true-peak / sample-peak meter
//! ```
//!
//! Each channel is K-weighted, squared, and BS.1770-weighted (`G_c`); the sum
//! over channels is the instantaneous loudness power fed to the
//! [`LoudnessIntegrator`], which owns the momentary / short-term / integrated /
//! LRA windows.  The unweighted squared sum drives a plain momentary RMS.  The
//! raw signal also feeds the [`TruePeakMeter`].
//!
//! # Channel weights (Phase 0: mono + stereo)
//!
//! BS.1770 weights are `1.0` for front L/R/C, `1.41` (+1.5 dB) for the surround
//! channels, and the LFE is excluded.  Mono and stereo — the v1 scope — use
//! unit weights throughout; the surround weighting is a v2 addition
//! ([`bs1770_weights`] is written to extend, not be rewritten).
//!
//! # Real-time safety
//!
//! Everything is sized in [`LoudnessMeter::new`] (marked **Allocates**).
//! [`LoudnessMeter::process_block`] wraps the near-silent K-weighted path in a
//! [`DenormalGuard`] and does no allocation, lock, or panic for blocks
//! `<= max_block`.

use rt_fft_convolver::DenormalGuard;

use crate::broadcast::{MeterBroadcast, MeterValues};
use crate::integrator::LoudnessIntegrator;
use crate::kweight::KWeightingFilter;
use crate::truepeak::TruePeakMeter;

/// BS.1770 channel weights for a `channels`-wide layout.
///
/// v1 (mono, stereo) is all unit weights.  When surround lands (v2) this is
/// where L/R/C = 1.0, Ls/Rs = 1.41, LFE = 0.0 go — callers never change.
fn bs1770_weights(channels: usize) -> Vec<f32> {
    vec![1.0; channels]
}

/// A minimal 400 ms sliding mean-square for the unweighted RMS readout.
///
/// The [`LoudnessIntegrator`] already tracks the *weighted, gated* windows;
/// RMS only needs the last four 100 ms sub-blocks, so this keeps a tiny
/// dedicated ring rather than a second full integrator with unused histograms.
struct MomentaryRms {
    subblocks: [f64; 4],
    write_pos: usize,
    done: u64,
    sum: f64,
    count: u32,
    samples_per_subblock: u32,
}

impl MomentaryRms {
    fn new(sample_rate: u32) -> Self {
        Self {
            subblocks: [0.0; 4],
            write_pos: 0,
            done: 0,
            sum: 0.0,
            count: 0,
            samples_per_subblock: ((sample_rate as f64 * 0.1).round() as u32).max(1),
        }
    }

    #[inline]
    fn push(&mut self, power: f32) {
        self.sum += power as f64;
        self.count += 1;
        if self.count >= self.samples_per_subblock {
            self.subblocks[self.write_pos] = self.sum / self.count as f64;
            self.write_pos = (self.write_pos + 1) % 4;
            self.done = self.done.saturating_add(1);
            self.sum = 0.0;
            self.count = 0;
        }
    }

    /// Unweighted RMS over the momentary window, in dB (`10·log10(mean-sq)`).
    fn rms_db(&self) -> f32 {
        let power = if self.done == 0 {
            if self.count == 0 {
                return f32::NEG_INFINITY;
            }
            self.sum / self.count as f64
        } else {
            let take = (self.done as usize).min(4);
            let mut acc = 0.0f64;
            for offset in 1..=take {
                acc += self.subblocks[(self.write_pos + 4 - offset) % 4];
            }
            acc / take as f64
        };
        if power <= 0.0 {
            f32::NEG_INFINITY
        } else {
            (10.0 * power.log10()) as f32
        }
    }

    fn reset(&mut self) {
        self.subblocks = [0.0; 4];
        self.write_pos = 0;
        self.done = 0;
        self.sum = 0.0;
        self.count = 0;
    }
}

/// The full loudness meter for a mono or stereo signal.
pub struct LoudnessMeter {
    kweight: Vec<KWeightingFilter>,
    weights: Vec<f32>,
    integrator: LoudnessIntegrator,
    rms: MomentaryRms,
    true_peak: TruePeakMeter,
    /// Per-sample cross-channel weighted-power accumulator (len `max_block`).
    weighted_acc: Vec<f32>,
    /// Per-sample cross-channel raw-power accumulator (len `max_block`).
    raw_acc: Vec<f32>,
    num_channels: usize,
    max_block: usize,
}

impl LoudnessMeter {
    /// Build a meter for `channels` channels at `sample_rate` Hz, sized for
    /// input blocks up to `max_block` samples, using the true-peak detector's
    /// default high-quality half-band design.
    ///
    /// **Allocates** — call off the audio thread.
    pub fn new(channels: usize, sample_rate: u32, max_block: usize) -> Self {
        Self::build(
            channels,
            sample_rate,
            TruePeakMeter::new(channels, max_block),
            max_block,
        )
    }

    /// Build with an explicit true-peak half-band design (lighter designs keep
    /// tests fast).
    ///
    /// **Allocates** — call off the audio thread.
    pub fn with_true_peak_design(
        channels: usize,
        sample_rate: u32,
        max_block: usize,
        atten_db: f64,
        transition_frac: f64,
    ) -> Self {
        Self::build(
            channels,
            sample_rate,
            TruePeakMeter::with_design(channels, max_block, atten_db, transition_frac),
            max_block,
        )
    }

    fn build(
        channels: usize,
        sample_rate: u32,
        true_peak: TruePeakMeter,
        max_block: usize,
    ) -> Self {
        Self {
            kweight: vec![KWeightingFilter::new(sample_rate); channels],
            weights: bs1770_weights(channels),
            integrator: LoudnessIntegrator::new(sample_rate),
            rms: MomentaryRms::new(sample_rate),
            true_peak,
            weighted_acc: vec![0.0; max_block],
            raw_acc: vec![0.0; max_block],
            num_channels: channels,
            max_block,
        }
    }

    /// Number of channels this meter measures.
    pub fn channels(&self) -> usize {
        self.num_channels
    }

    /// Enable or freeze integrated / LRA accumulation (the `Start/Stop`
    /// control).  Momentary, short-term and the peak holds keep updating
    /// regardless.  **Real-time safe.**
    pub fn set_integrating(&mut self, integrating: bool) {
        self.integrator.set_integrating(integrating);
    }

    /// Analyse one block of every channel (all slices the same length).
    /// The signal is **not** modified — the meter is measurement-only.
    ///
    /// **Real-time safe** for blocks `<= max_block`.
    pub fn process_block(&mut self, channels: &[&[f32]]) {
        // Flush-to-zero over the near-silent K-weighted / IIR tail.
        let _guard = DenormalGuard::new();

        if self.num_channels == 0 || channels.is_empty() {
            return;
        }

        // Peak metering is block-wise (the oversampler runs on whole blocks).
        self.true_peak.process_block(channels);

        // Loudness / RMS accumulate the cross-channel power per sample.  We go
        // channel-major (the natural order for each channel's IIR state) into
        // two per-sample accumulators, then feed the integrator sample-major.
        let usable = channels.len().min(self.num_channels);
        let block = channels[..usable]
            .iter()
            .map(|channel| channel.len())
            .min()
            .unwrap_or(0)
            .min(self.max_block);

        let weighted_acc = &mut self.weighted_acc[..block];
        let raw_acc = &mut self.raw_acc[..block];
        weighted_acc.fill(0.0);
        raw_acc.fill(0.0);

        for (channel, input) in channels[..usable].iter().enumerate() {
            let filter = &mut self.kweight[channel];
            let weight = self.weights[channel];
            for ((sample, weighted_slot), raw_slot) in input[..block]
                .iter()
                .zip(weighted_acc.iter_mut())
                .zip(raw_acc.iter_mut())
            {
                let weighted = filter.process_sample(*sample);
                *weighted_slot += weight * weighted * weighted;
                *raw_slot += sample * sample;
            }
        }

        let inv_channels = 1.0 / usable as f32;
        for (&weighted_power, &raw_power) in weighted_acc.iter().zip(raw_acc.iter()) {
            self.integrator.push(weighted_power);
            self.rms.push(raw_power * inv_channels);
        }
    }

    /// The current set of readings.
    pub fn values(&self) -> MeterValues {
        MeterValues {
            momentary_lufs: self.integrator.momentary_lufs(),
            short_term_lufs: self.integrator.short_term_lufs(),
            integrated_lufs: self.integrator.integrated_lufs(),
            lra_lu: self.integrator.lra_lu(),
            true_peak_dbtp: self.true_peak.true_peak_dbtp(),
            sample_peak_dbfs: self.true_peak.sample_peak_dbfs(),
            rms_db: self.rms.rms_db(),
        }
    }

    /// Publish the current readings to a shared broadcast.  **Real-time safe.**
    pub fn publish(&self, broadcast: &MeterBroadcast) {
        broadcast.store(&self.values());
    }

    /// Clear every window, gate, peak hold and filter state (a meter `Reset`).
    pub fn reset(&mut self) {
        for filter in self.kweight.iter_mut() {
            filter.reset();
        }
        self.integrator.reset();
        self.rms.reset();
        self.true_peak.reset();
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::make_sine;

    /// A light true-peak design keeps the impulse-latency probe cheap.
    const LIGHT: (f64, f64) = (80.0, 0.1);
    const SR: u32 = 48_000;

    fn stereo_meter(max_block: usize) -> LoudnessMeter {
        LoudnessMeter::with_true_peak_design(2, SR, max_block, LIGHT.0, LIGHT.1)
    }

    #[test]
    fn silence_reads_negative_infinity() {
        let mut meter = stereo_meter(512);
        let silence = vec![0.0f32; 512];
        meter.process_block(&[&silence, &silence]);
        let values = meter.values();
        assert_eq!(values.momentary_lufs, f32::NEG_INFINITY);
        assert_eq!(values.integrated_lufs, f32::NEG_INFINITY);
        assert_eq!(values.true_peak_dbtp, f32::NEG_INFINITY);
    }

    /// A 1 kHz sine at −20 dBFS should read close to −20 LUFS momentary: at
    /// 1 kHz the K-weighting gain is ≈ 0 dB, so LUFS ≈ dBFS for a mid tone.
    /// (BS.1770 defines 0 LUFS as a full-scale 1 kHz sine, i.e. RMS −3 dBFS →
    /// the −0.691 offset absorbs the sine's crest factor.)
    #[test]
    fn thousand_hz_sine_reads_near_its_level() {
        let block = 4_800; // 100 ms
        let level_dbfs = -20.0f32;
        let amplitude = 10.0f32.powf(level_dbfs / 20.0);
        let sine = make_sine(1_000.0, SR, block, amplitude);

        let mut meter = stereo_meter(block);
        // Feed ~1 s so momentary/short-term settle.
        for _ in 0..10 {
            meter.process_block(&[&sine, &sine]);
        }
        // Stereo dual-mono of a sine: the two channel powers sum (+3.01 LU) and
        // the −0.691 offset minus the sine's −3.01 dB crest very nearly cancel,
        // so a dual-mono 1 kHz sine reads ≈ its dBFS peak level in LUFS.
        let expected = level_dbfs - 0.691;
        let momentary = meter.values().momentary_lufs;
        assert!(
            (momentary - expected).abs() < 1.0,
            "momentary {momentary} LUFS should be ~{expected} for a −20 dBFS 1 kHz sine"
        );
    }

    #[test]
    fn mono_and_stereo_of_same_tone_match() {
        // The same tone panned dual-mono reads the same LUFS as mono: stereo
        // sums two channels of half the... no — BS.1770 sums channel powers,
        // so identical L=R is +3 LU over one channel.  Verify that relation.
        let block = 4_800;
        let sine = make_sine(1_000.0, SR, block, 0.5);

        let mut mono = LoudnessMeter::with_true_peak_design(1, SR, block, LIGHT.0, LIGHT.1);
        let mut stereo = stereo_meter(block);
        for _ in 0..10 {
            mono.process_block(&[&sine]);
            stereo.process_block(&[&sine, &sine]);
        }
        let delta = stereo.values().momentary_lufs - mono.values().momentary_lufs;
        assert!(
            (delta - 3.0103).abs() < 0.1,
            "stereo dual-mono should be +3 LU over mono, got {delta}"
        );
    }

    #[test]
    fn reset_clears_everything() {
        let mut meter = stereo_meter(512);
        let hot = vec![0.5f32; 512];
        for _ in 0..10 {
            meter.process_block(&[&hot, &hot]);
        }
        meter.reset();
        let values = meter.values();
        assert_eq!(values.momentary_lufs, f32::NEG_INFINITY);
        assert_eq!(values.integrated_lufs, f32::NEG_INFINITY);
        assert_eq!(values.true_peak_dbtp, f32::NEG_INFINITY);
    }

    #[test]
    fn publish_round_trips_through_broadcast() {
        let mut meter = stereo_meter(512);
        let hot = vec![0.3f32; 512];
        for _ in 0..10 {
            meter.process_block(&[&hot, &hot]);
        }
        let broadcast = MeterBroadcast::new();
        meter.publish(&broadcast);
        assert_eq!(broadcast.load(), meter.values());
    }
}
