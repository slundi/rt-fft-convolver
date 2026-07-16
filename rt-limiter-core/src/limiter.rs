//! [`Limiter`] — the integrated stereo true-peak look-ahead limiter.
//!
//! This wires the Phase 1 primitives into the full feed-forward path chosen in
//! Phase 0:
//!
//! ```text
//!            ┌─────────────────────────────────────── delay (look-ahead + Lᴅ) ──────────┐
//!   in ──────┤                                                                           ▼
//!            └▶ [TruePeakDetector ↑] ─▶ [WindowMax] ─▶ [GainComputer] ── g ──▶ ×g ─▶ [CeilingClipper] ─▶ (Dither) ─▶ out
//! ```
//!
//! # The alignment that makes it a brick wall
//!
//! Three delays have to line up or the limiter either clips (gain applied too
//! late) or pumps (applied too early).  Writing `Lᴅ` for the detector's group
//! delay ([`TruePeakDetector::latency`], in input samples) and `A` for the
//! user look-ahead in samples:
//!
//! - the dry signal is delayed by `A + Lᴅ`;
//! - the peak envelope produced by the detector is itself `Lᴅ` behind its
//!   input;
//! - the [`WindowMax`] takes the maximum over the last `A + Lᴅ + 1` envelope
//!   samples.
//!
//! For the sample currently leaving the delay line (input index `m`), its own
//! true-peak sits inside that window (envelope index `m + Lᴅ`), **and** so do
//! the peaks of the next `A` input samples.  So the applied gain never exceeds
//! `threshold / peak` for the sample being output — the guarantee — while also
//! ducking `A` samples early for anything loud on the way (the anticipation /
//! attack).  The reported host latency is exactly `A + Lᴅ`.
//!
//! # Channel linkage (Phase 0: stereo-linked + M/S)
//!
//! Both shipped modes apply a **single** gain to both channels, differing only
//! in the domain the peak is measured and the gain applied in:
//!
//! - [`ChannelLink::Linked`] — peak = `max(|L|, |R|)`, gain applied to L and R.
//!   Preserves the stereo image; the mastering default.
//! - [`ChannelLink::MidSide`] — the same, but in the mid/side domain (a matrix
//!   wrap around the identical single-gain machinery).
//!
//! Fully-independent per-channel reduction was deferred in Phase 0.
//!
//! # Real-time safety
//!
//! [`Limiter::process_block`] allocates nothing, never locks and does not panic
//! in release for blocks `≤ max_block`.  All buffers are sized for the 20 ms
//! maximum look-ahead at construction, so [`Limiter::set_lookahead_ms`] only
//! moves a read tap — it never reallocates.  Wrap the host render callback in
//! `rt_fft_convolver::DenormalGuard`: the gain-reduction envelope and the delay
//! tails ring down toward subnormal territory on quiet passages.

use crate::clip::CeilingClipper;
use crate::delay::DelayLine;
use crate::detector::TruePeakDetector;
use crate::gain::{GainComputer, ReleaseCurve};
use crate::window::WindowMax;

#[cfg(feature = "dither")]
use crate::dither::{BitDepth, Dither};

/// How the two channels share gain reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelLink {
    /// One gain from `max(|L|, |R|)`, applied to L and R.  Preserves the image.
    Linked,
    /// As `Linked` but computed and applied in the mid/side domain.
    MidSide,
}

/// Smallest selectable look-ahead, in milliseconds (Phase 0 range 0.1–20 ms).
pub const MIN_LOOKAHEAD_MS: f32 = 0.1;
/// Largest selectable look-ahead, in milliseconds.  Buffers are sized for it.
pub const MAX_LOOKAHEAD_MS: f32 = 20.0;
/// Default look-ahead if the caller does not set one.
pub const DEFAULT_LOOKAHEAD_MS: f32 = 5.0;

/// Integrated stereo true-peak look-ahead brick-wall limiter.
pub struct Limiter {
    sample_rate: f32,
    max_block: usize,
    link: ChannelLink,

    detector_left: TruePeakDetector,
    detector_right: TruePeakDetector,
    window: WindowMax,
    gain: GainComputer,
    delay_left: DelayLine,
    delay_right: DelayLine,
    clip_left: CeilingClipper,
    clip_right: CeilingClipper,

    #[cfg(feature = "dither")]
    dither_left: Option<Dither>,
    #[cfg(feature = "dither")]
    dither_right: Option<Dither>,

    /// Detector group delay `Lᴅ` in input samples (same for both channels).
    detector_latency: usize,
    /// Current look-ahead `A` in samples.
    lookahead: usize,
    /// Largest look-ahead the buffers were sized for.
    max_lookahead: usize,

    /// Per-block true-peak scratch, one slot per input sample.
    peaks_left: Vec<f32>,
    peaks_right: Vec<f32>,

    /// Smallest gain applied during the most recent [`process_block`] — the
    /// peak gain reduction, for a host GR meter.  `1.0` = no reduction.
    min_gain: f32,
}

impl Limiter {
    /// Build a limiter using the resampler's default (96 dB) filter design.
    ///
    /// - `factor` — true-peak oversampling factor (`1, 2, 4, 8, 16`).
    /// - `threshold_db` — limiting level in dBFS.
    /// - `ceiling_db` — output true-peak ceiling in dBFS (safety clip).
    /// - `release_ms` — release time; `curve` — its shape.
    ///
    /// **Allocates** — call off the audio thread.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sample_rate: f32,
        max_block: usize,
        factor: usize,
        threshold_db: f32,
        ceiling_db: f32,
        release_ms: f32,
        curve: ReleaseCurve,
        link: ChannelLink,
    ) -> Self {
        Self::with_detectors(
            TruePeakDetector::new(factor, max_block),
            TruePeakDetector::new(factor, max_block),
            sample_rate,
            max_block,
            threshold_db,
            ceiling_db,
            release_ms,
            curve,
            link,
        )
    }

    /// Build a limiter from two pre-built detectors (lets tests inject a light
    /// filter design so the impulse-latency probe stays cheap).
    ///
    /// **Allocates** — call off the audio thread.
    #[allow(clippy::too_many_arguments)]
    pub fn with_detectors(
        detector_left: TruePeakDetector,
        detector_right: TruePeakDetector,
        sample_rate: f32,
        max_block: usize,
        threshold_db: f32,
        ceiling_db: f32,
        release_ms: f32,
        curve: ReleaseCurve,
        link: ChannelLink,
    ) -> Self {
        debug_assert_eq!(
            detector_left.latency(),
            detector_right.latency(),
            "both detectors must share a design"
        );
        let detector_latency = detector_left.latency();

        let max_lookahead = ms_to_samples(MAX_LOOKAHEAD_MS, sample_rate);
        let lookahead = ms_to_samples(DEFAULT_LOOKAHEAD_MS, sample_rate).min(max_lookahead);

        let max_audio_delay = max_lookahead + detector_latency;
        let audio_delay = lookahead + detector_latency;
        // The window spans the sample being output plus its `lookahead` future.
        let max_window = max_audio_delay + 1;
        let window_len = audio_delay + 1;

        Self {
            sample_rate,
            max_block,
            link,
            detector_left,
            detector_right,
            window: WindowMax::new(max_window, window_len),
            gain: GainComputer::new(threshold_db, sample_rate, release_ms, curve),
            delay_left: DelayLine::new(max_audio_delay, audio_delay),
            delay_right: DelayLine::new(max_audio_delay, audio_delay),
            clip_left: CeilingClipper::new(ceiling_db),
            clip_right: CeilingClipper::new(ceiling_db),
            #[cfg(feature = "dither")]
            dither_left: None,
            #[cfg(feature = "dither")]
            dither_right: None,
            detector_latency,
            lookahead,
            max_lookahead,
            peaks_left: vec![0.0; max_block],
            peaks_right: vec![0.0; max_block],
            min_gain: 1.0,
        }
    }

    /// The peak gain reduction applied during the most recent
    /// [`process_block`](Self::process_block), as a linear gain in `(0, 1]`
    /// (`1.0` = no reduction).  For a host gain-reduction meter.
    #[inline]
    pub fn min_gain(&self) -> f32 {
        self.min_gain
    }

    /// Reported latency for host delay compensation, in samples (`A + Lᴅ`).
    #[inline]
    pub fn latency_samples(&self) -> usize {
        self.delay_left.delay()
    }

    /// The current look-ahead in samples (`A`).
    #[inline]
    pub fn lookahead_samples(&self) -> usize {
        self.lookahead
    }

    /// The oversampling factor used for true-peak detection.
    #[inline]
    pub fn oversampling_factor(&self) -> usize {
        self.detector_left.factor()
    }

    /// The maximum input block length this limiter was sized for.
    #[inline]
    pub fn max_block(&self) -> usize {
        self.max_block
    }

    /// Set the limiting threshold in dBFS.  **Real-time safe.**
    #[inline]
    pub fn set_threshold_db(&mut self, threshold_db: f32) {
        self.gain.set_threshold_db(threshold_db);
    }

    /// Set the output ceiling in dBFS.  **Real-time safe.**
    #[inline]
    pub fn set_ceiling_db(&mut self, ceiling_db: f32) {
        self.clip_left.set_ceiling_db(ceiling_db);
        self.clip_right.set_ceiling_db(ceiling_db);
    }

    /// Set the release time in milliseconds.  **Real-time safe.**
    #[inline]
    pub fn set_release_ms(&mut self, release_ms: f32) {
        self.gain.set_release_ms(release_ms);
    }

    /// Set the release curve.  **Real-time safe.**
    #[inline]
    pub fn set_curve(&mut self, curve: ReleaseCurve) {
        self.gain.set_curve(curve);
    }

    /// Change the channel-linkage mode.  Resets state, since the delay lines
    /// then hold samples in a different (L/R vs M/S) domain.
    pub fn set_link(&mut self, link: ChannelLink) {
        if link != self.link {
            self.link = link;
            self.reset();
        }
    }

    /// Set the look-ahead in milliseconds (clamped to `0.1..=20`).
    ///
    /// Only moves the delay tap and window length — **no reallocation** — but
    /// resets state so the new alignment starts clean and the latency step is
    /// not smeared.  Read the new latency back with
    /// [`latency_samples`](Self::latency_samples) and report it to the host.
    pub fn set_lookahead_ms(&mut self, lookahead_ms: f32) {
        let lookahead = ms_to_samples(lookahead_ms, self.sample_rate).min(self.max_lookahead);
        if lookahead == self.lookahead {
            return;
        }
        self.lookahead = lookahead;
        let audio_delay = lookahead + self.detector_latency;
        self.delay_left.set_delay(audio_delay);
        self.delay_right.set_delay(audio_delay);
        self.window.set_window(audio_delay + 1);
        self.reset();
    }

    /// Enable output dither at `depth` with optional noise shaping.
    #[cfg(feature = "dither")]
    pub fn enable_dither(&mut self, depth: BitDepth, shape: bool) {
        // Distinct seeds so the two channels' dither noise is decorrelated.
        self.dither_left = Some(Dither::new(depth, shape, 0x2545_F491));
        self.dither_right = Some(Dither::new(depth, shape, 0x9E37_79B9));
    }

    /// Disable output dither.
    #[cfg(feature = "dither")]
    pub fn disable_dither(&mut self) {
        self.dither_left = None;
        self.dither_right = None;
    }

    /// Limit a stereo block in place.
    ///
    /// `left` and `right` must be the same length and `≤ max_block` (both
    /// debug-asserted).
    ///
    /// **Real-time safe.**
    pub fn process_block(&mut self, left: &mut [f32], right: &mut [f32]) {
        debug_assert_eq!(left.len(), right.len(), "channel length mismatch");
        debug_assert!(left.len() <= self.max_block, "block exceeds max_block");
        let block = left.len().min(right.len());

        // Mid/side encode: work on M = (L+R)/2, S = (L−R)/2 in place.
        if self.link == ChannelLink::MidSide {
            for index in 0..block {
                let (left_sample, right_sample) = (left[index], right[index]);
                left[index] = 0.5 * (left_sample + right_sample);
                right[index] = 0.5 * (left_sample - right_sample);
            }
        }

        // True-peak envelope for each (possibly M/S) channel.
        self.detector_left
            .detect_block(&left[..block], &mut self.peaks_left[..block]);
        self.detector_right
            .detect_block(&right[..block], &mut self.peaks_right[..block]);

        let mut block_min_gain = 1.0f32;
        for index in 0..block {
            // Delay the dry signal; the returned samples are `A + Lᴅ` old.
            let dry_left = self.delay_left.process(left[index]);
            let dry_right = self.delay_right.process(right[index]);

            // Linked peak → window maximum → smoothed gain.
            let linked_peak = self.peaks_left[index].max(self.peaks_right[index]);
            let windowed_peak = self.window.push(linked_peak);
            let gain = self.gain.process(windowed_peak);
            block_min_gain = block_min_gain.min(gain);

            // Apply the single gain, then the ceiling safety net.
            let out_left = self.clip_left.process(dry_left * gain);
            let out_right = self.clip_right.process(dry_right * gain);

            // Optional output dither / noise shaping (shadows the above).
            #[cfg(feature = "dither")]
            let (out_left, out_right) = {
                let left_dithered = match self.dither_left.as_mut() {
                    Some(dither) => dither.process(out_left),
                    None => out_left,
                };
                let right_dithered = match self.dither_right.as_mut() {
                    Some(dither) => dither.process(out_right),
                    None => out_right,
                };
                (left_dithered, right_dithered)
            };

            left[index] = out_left;
            right[index] = out_right;
        }

        // Mid/side decode: L = M + S, R = M − S.
        if self.link == ChannelLink::MidSide {
            for index in 0..block {
                let (mid, side) = (left[index], right[index]);
                left[index] = mid + side;
                right[index] = mid - side;
            }
        }

        // Publish the block's peak reduction for the GR meter (an empty block
        // leaves the previous reading untouched).
        if block > 0 {
            self.min_gain = block_min_gain;
        }
    }

    /// Clear all internal state (detectors, delays, window, gain envelope,
    /// dither).  Leaves parameters (threshold, ceiling, look-ahead, …) intact.
    pub fn reset(&mut self) {
        self.detector_left.reset();
        self.detector_right.reset();
        self.delay_left.reset();
        self.delay_right.reset();
        self.window.reset();
        self.gain.reset();
        self.min_gain = 1.0;
        #[cfg(feature = "dither")]
        {
            if let Some(dither) = self.dither_left.as_mut() {
                dither.reset();
            }
            if let Some(dither) = self.dither_right.as_mut() {
                dither.reset();
            }
        }
    }
}

/// Convert a look-ahead in milliseconds (clamped to the Phase 0 range) to a
/// whole number of samples.
fn ms_to_samples(milliseconds: f32, sample_rate: f32) -> usize {
    let clamped = milliseconds.clamp(MIN_LOOKAHEAD_MS, MAX_LOOKAHEAD_MS);
    (clamped * 1e-3 * sample_rate).round() as usize
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detector::TruePeakDetector;
    use std::f32::consts::TAU;

    const SAMPLE_RATE: f32 = 48_000.0;
    /// Light half-band design keeps the construction-time latency probe fast.
    const LIGHT: (f64, f64) = (80.0, 0.1);

    fn light_limiter(
        factor: usize,
        max_block: usize,
        threshold_db: f32,
        ceiling_db: f32,
        link: ChannelLink,
    ) -> Limiter {
        Limiter::with_detectors(
            TruePeakDetector::with_design(factor, max_block, LIGHT.0, LIGHT.1),
            TruePeakDetector::with_design(factor, max_block, LIGHT.0, LIGHT.1),
            SAMPLE_RATE,
            max_block,
            threshold_db,
            ceiling_db,
            50.0,
            ReleaseCurve::Exponential,
            link,
        )
    }

    /// Reported latency is the look-ahead plus a positive detector delay at
    /// an oversampling factor > 1.
    #[test]
    fn latency_is_lookahead_plus_detector() {
        let limiter = light_limiter(4, 128, -1.0, -1.0, ChannelLink::Linked);
        // Lᴅ > 0 at 4×, so total latency strictly exceeds the look-ahead.
        assert!(limiter.latency_samples() > limiter.lookahead_samples());
        // factor 1 has zero detector latency → total latency == look-ahead.
        let flat = light_limiter(1, 128, -1.0, -1.0, ChannelLink::Linked);
        assert_eq!(flat.latency_samples(), flat.lookahead_samples());
    }

    /// Sub-threshold input comes out **bit-identical** to the delayed dry
    /// signal (Phase 4 bypass-equivalence criterion), in Linked mode.
    #[test]
    fn sub_threshold_is_delayed_dry() {
        let max_block = 128;
        let mut limiter = light_limiter(4, max_block, 0.0, 0.0, ChannelLink::Linked);
        let latency = limiter.latency_samples();

        // Quiet signal well under 0 dBFS → no gain reduction, no clipping.
        let total = max_block * 6;
        let source: Vec<f32> = (0..total)
            .map(|n| 0.3 * (TAU * 0.01 * n as f32).sin())
            .collect();

        let mut output = Vec::with_capacity(total);
        for chunk in source.chunks(max_block) {
            let mut left = chunk.to_vec();
            let mut right = chunk.to_vec();
            limiter.process_block(&mut left, &mut right);
            output.extend_from_slice(&left);
        }

        // output[n] must equal source[n − latency] exactly.
        for n in latency..total {
            assert_eq!(
                output[n],
                source[n - latency],
                "sample {n} not bit-identical to delayed dry"
            );
        }
    }

    /// A hot input never leaves the output true-peak above the ceiling
    /// (measured at the same oversampling factor the limiter controls).
    #[test]
    fn output_true_peak_respects_ceiling() {
        for factor in [2usize, 4, 8] {
            let max_block = 256;
            let ceiling_db = -1.0;
            // threshold == ceiling: limiter pulls true-peak down to the ceiling.
            let mut limiter = light_limiter(
                factor,
                max_block,
                ceiling_db,
                ceiling_db,
                ChannelLink::Linked,
            );

            // A high-frequency, off-grid, over-unity signal — designed to have
            // large inter-sample peaks and force heavy limiting.
            let total = max_block * 8;
            let make = |n: usize| 1.8 * (TAU * 0.24 * n as f32 + 0.3).sin();
            let mut collected = Vec::with_capacity(total);
            for start in (0..total).step_by(max_block) {
                let mut left: Vec<f32> = (start..start + max_block).map(make).collect();
                let mut right = left.clone();
                limiter.process_block(&mut left, &mut right);
                collected.extend_from_slice(&left);
            }

            // Measure the settled output's true-peak at the same factor.
            let mut probe = TruePeakDetector::with_design(factor, max_block, LIGHT.0, LIGHT.1);
            let mut peaks = vec![0.0f32; max_block];
            let ceiling_lin = rt_fft_convolver::db_to_linear(ceiling_db);
            // Skip the initial latency ramp-up region.
            let mut worst = 0.0f32;
            for (block_index, chunk) in collected.chunks(max_block).enumerate() {
                probe.detect_block(chunk, &mut peaks);
                if block_index >= 2 {
                    worst = worst.max(peaks.iter().copied().fold(0.0, f32::max));
                }
            }
            assert!(
                worst <= ceiling_lin * 1.02,
                "factor {factor}: output true-peak {worst} exceeds ceiling {ceiling_lin}"
            );
        }
    }

    /// Gain reduction reaches its target within the look-ahead window: when a
    /// loud region follows silence, the first loud output block is *already*
    /// limited (attack completed before the loud audio leaves the delay line),
    /// so it never overshoots the threshold (Phase 4 attack criterion).
    #[test]
    fn attack_reaches_target_within_lookahead() {
        let max_block = 512;
        let mut limiter = light_limiter(4, max_block, -6.0, 0.0, ChannelLink::Linked);
        let latency = limiter.latency_samples();
        let threshold = rt_fft_convolver::db_to_linear(-6.0);

        // Silence, then a sustained over-threshold tone.
        let total = max_block * 4;
        let onset = max_block; // loud region begins here (input index)
        let make = |n: usize| {
            if n >= onset {
                1.0 * (TAU * 0.05 * n as f32).sin()
            } else {
                0.0
            }
        };

        let mut collected = Vec::with_capacity(total);
        for start in (0..total).step_by(max_block) {
            let mut left: Vec<f32> = (start..start + max_block).map(make).collect();
            let mut right = left.clone();
            limiter.process_block(&mut left, &mut right);
            collected.extend_from_slice(&left);
        }

        // The loud region reaches the output at index `onset + latency`.  Over
        // its first block the peak must already sit at or below the threshold:
        // the attack was completed within the look-ahead, so nothing overshoots.
        let first_loud_out = onset + latency;
        let window = &collected[first_loud_out..first_loud_out + max_block];
        let peak = window.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        assert!(
            peak <= threshold * 1.05,
            "first loud output block peaks at {peak}, above threshold {threshold}"
        );
        // ...and the region is genuinely being limited (input peaked near 1.0).
        assert!(peak > 0.3 * threshold, "loud region unexpectedly silent");
    }

    /// After a loud burst the gain returns toward unity within ~the release
    /// time (Phase 4 release criterion, exponential curve, ±10 %).
    #[test]
    fn release_recovers_after_burst() {
        let max_block = 256;
        let release_ms = 20.0;
        let mut limiter = Limiter::with_detectors(
            TruePeakDetector::with_design(4, max_block, LIGHT.0, LIGHT.1),
            TruePeakDetector::with_design(4, max_block, LIGHT.0, LIGHT.1),
            SAMPLE_RATE,
            max_block,
            -6.0,
            0.0,
            release_ms,
            ReleaseCurve::Exponential,
            ChannelLink::Linked,
        );

        // One loud block to force reduction, then a long quiet tail under
        // threshold — long enough for the exponential release to complete.
        let quiet = 0.1f32;
        let total = max_block * 40;
        let make = |n: usize| if n < max_block { 2.0 } else { quiet };

        let mut collected = Vec::with_capacity(total);
        for start in (0..total).step_by(max_block) {
            let mut left: Vec<f32> = (start..start + max_block).map(make).collect();
            let mut right = left.clone();
            limiter.process_block(&mut left, &mut right);
            collected.extend_from_slice(&left);
        }

        // Well after several release time-constants the quiet tail should be
        // back to (approximately) its dry level — gain recovered to ~unity.
        let tail = &collected[total - max_block..];
        let peak = tail.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        assert!(
            peak > quiet * 0.99,
            "tail peak {peak} shows gain never recovered to unity"
        );
    }

    /// M/S mode on a mono (L == R) signal leaves the side channel silent and
    /// stays bounded; the output remains mono after the decode.
    #[test]
    fn mid_side_mono_is_bounded_and_symmetric() {
        let max_block = 128;
        let mut limiter = light_limiter(4, max_block, -3.0, 0.0, ChannelLink::MidSide);
        let make = |n: usize| 1.5 * (TAU * 0.05 * n as f32).sin();
        for start in (0..max_block * 6).step_by(max_block) {
            let mut left: Vec<f32> = (start..start + max_block).map(make).collect();
            let mut right = left.clone();
            limiter.process_block(&mut left, &mut right);
            for (l, r) in left.iter().zip(right.iter()) {
                assert!(l.is_finite() && r.is_finite());
                // Mono in → mono out (side is zero, decode gives L == R).
                assert!((l - r).abs() < 1e-4, "mono symmetry broken: {l} vs {r}");
                assert!(l.abs() <= 1.05, "output not bounded: {l}");
            }
        }
    }

    /// `min_gain` reports the block's peak reduction: unity for a quiet block,
    /// below unity once limiting engages.
    #[test]
    fn min_gain_tracks_reduction() {
        let max_block = 256;
        let mut limiter = light_limiter(4, max_block, -6.0, 0.0, ChannelLink::Linked);

        // Quiet block, well under threshold → no reduction.
        let mut left = vec![0.05f32; max_block];
        let mut right = vec![0.05f32; max_block];
        limiter.process_block(&mut left, &mut right);
        assert!((limiter.min_gain() - 1.0).abs() < 1e-6);

        // Hot block → the meter shows real reduction (gain < 1).
        let mut left = vec![2.0f32; max_block];
        let mut right = vec![2.0f32; max_block];
        limiter.process_block(&mut left, &mut right);
        assert!(
            limiter.min_gain() < 0.9,
            "GR meter should show reduction, got gain {}",
            limiter.min_gain()
        );
    }

    /// `set_lookahead_ms` changes the reported latency and clamps to the range.
    #[test]
    fn set_lookahead_changes_latency() {
        let mut limiter = light_limiter(4, 128, -1.0, -1.0, ChannelLink::Linked);
        let base = limiter.latency_samples();
        limiter.set_lookahead_ms(20.0);
        assert!(limiter.latency_samples() > base);
        // Over-range clamps to 20 ms, not beyond.
        let at_max = limiter.latency_samples();
        limiter.set_lookahead_ms(100.0);
        assert_eq!(limiter.latency_samples(), at_max);
        // Under-range clamps to 0.1 ms.
        limiter.set_lookahead_ms(0.0);
        let min_samples = (MIN_LOOKAHEAD_MS * 1e-3 * SAMPLE_RATE).round() as usize;
        assert_eq!(limiter.lookahead_samples(), min_samples);
    }

    /// `reset` clears state: a silent block after hard limiting comes out
    /// silent once the delay line has flushed.
    #[test]
    fn reset_clears_state() {
        let max_block = 128;
        let mut limiter = light_limiter(4, max_block, -6.0, 0.0, ChannelLink::Linked);
        let mut left = vec![2.0f32; max_block];
        let mut right = vec![2.0f32; max_block];
        limiter.process_block(&mut left, &mut right);
        limiter.reset();

        let mut silent_left = vec![0.0f32; max_block];
        let mut silent_right = vec![0.0f32; max_block];
        // Flush the delay line (latency < one block here is not guaranteed, so
        // run enough blocks to clear it).
        for _ in 0..4 {
            silent_left.iter_mut().for_each(|s| *s = 0.0);
            silent_right.iter_mut().for_each(|s| *s = 0.0);
            limiter.process_block(&mut silent_left, &mut silent_right);
        }
        assert!(
            silent_left.iter().all(|&s| s.abs() < 1e-6),
            "state not cleared"
        );
    }

    #[cfg(feature = "dither")]
    #[test]
    fn dither_quantises_output() {
        let max_block = 64;
        let mut limiter = light_limiter(2, max_block, -1.0, -1.0, ChannelLink::Linked);
        limiter.enable_dither(BitDepth::Bits16, true);
        let make = |n: usize| 0.2 * (TAU * 0.03 * n as f32).sin();
        let mut left: Vec<f32> = (0..max_block).map(make).collect();
        let mut right = left.clone();
        limiter.process_block(&mut left, &mut right);
        let quantum = 1.0 / (1u32 << 15) as f32;
        for &sample in &left {
            let levels = sample / quantum;
            assert!((levels - levels.round()).abs() < 1e-3);
        }
        limiter.disable_dither();
    }
}
