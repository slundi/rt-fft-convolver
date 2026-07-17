//! Gated loudness histogram — the shared machinery behind **integrated
//! loudness** and **loudness range (LRA)**.
//!
//! # Why a histogram
//!
//! Both integrated loudness and LRA are computed over *gated* distributions of
//! block-loudness values accumulated across a whole session (BS.1770-4 §5 for
//! integrated; EBU Tech 3342 for LRA).  A naïve implementation keeps a growing
//! list of every block's loudness — which reallocates without bound and is
//! therefore **not** real-time safe.  Instead we bin block-loudness into a
//! fixed histogram (0.1 LU resolution): O(1) to update, bounded memory, and
//! accurate to within half a bin — the approach `libebur128` uses and which
//! passes the reference vectors within tolerance.
//!
//! # Gating (BS.1770-4)
//!
//! Two gates suppress silence and quiet passages so they do not drag the
//! measurement down:
//!
//! - **Absolute gate** at −70 LUFS — blocks below this are ignored outright
//!   (applied on insertion, so the histogram never holds them).
//! - **Relative gate** at the gated mean minus an offset — **−10 LU** for
//!   integrated loudness, **−20 LU** for LRA.  The threshold depends on the
//!   data, so it is applied at read time, not on insertion.

/// Loudness offset of the BS.1770 formula: `L = −0.691 + 10·log10(power)`.
const REF_OFFSET_DB: f64 = -0.691;
/// Absolute gate (LUFS): blocks quieter than this never enter the histogram.
const ABSOLUTE_GATE_LUFS: f64 = -70.0;
/// Top of the histogram range (LUFS).  Louder blocks clamp to the last bin.
const MAX_LUFS: f64 = 5.0;
/// Bin width in LU.  0.1 LU keeps the quantisation error under 0.05 LU.
const BIN_STEP_LU: f64 = 0.1;
/// Number of bins spanning `[ABSOLUTE_GATE_LUFS, MAX_LUFS]` inclusive.
const NUM_BINS: usize = ((MAX_LUFS - ABSOLUTE_GATE_LUFS) / BIN_STEP_LU) as usize + 1;

/// Convert a block loudness (LUFS) to its mean-square power.
#[inline]
fn power_from_loudness(loudness: f64) -> f64 {
    10.0_f64.powf((loudness - REF_OFFSET_DB) / 10.0)
}

/// Convert a mean-square power to loudness (LUFS).
#[inline]
fn loudness_from_power(power: f64) -> f64 {
    REF_OFFSET_DB + 10.0 * power.log10()
}

/// The representative loudness at the centre of histogram bin `index`.
#[inline]
fn bin_loudness(index: usize) -> f64 {
    ABSOLUTE_GATE_LUFS + index as f64 * BIN_STEP_LU
}

/// A fixed-size histogram of absolute-gated block-loudness values.
///
/// **Memory** is `NUM_BINS` `u32` counters (≈ 3 KB), fixed for the life of the
/// meter regardless of session length.
#[derive(Debug, Clone)]
pub struct LoudnessHistogram {
    bins: Vec<u32>,
}

impl Default for LoudnessHistogram {
    fn default() -> Self {
        Self::new()
    }
}

impl LoudnessHistogram {
    /// Create an empty histogram.
    ///
    /// **Allocates** — call off the audio thread.
    pub fn new() -> Self {
        Self {
            bins: vec![0; NUM_BINS],
        }
    }

    /// Record one block-loudness value.  Blocks below the absolute −70 LUFS
    /// gate (and non-finite values, e.g. a silent block's `−∞`) are dropped.
    ///
    /// **Real-time safe** — a single indexed increment, no allocation.
    #[inline]
    pub fn add(&mut self, loudness_lufs: f32) {
        let loudness = loudness_lufs as f64;
        if !loudness.is_finite() || loudness < ABSOLUTE_GATE_LUFS {
            return;
        }
        let raw = ((loudness - ABSOLUTE_GATE_LUFS) / BIN_STEP_LU).round();
        let index = (raw as usize).min(NUM_BINS - 1);
        self.bins[index] = self.bins[index].saturating_add(1);
    }

    /// Clear all counts.
    pub fn reset(&mut self) {
        for bin in self.bins.iter_mut() {
            *bin = 0;
        }
    }

    /// Total number of blocks recorded (already absolute-gated).
    #[cfg(test)]
    fn total(&self) -> u64 {
        self.bins.iter().map(|&count| count as u64).sum()
    }

    /// Mean power of every bin whose loudness is `>= threshold_lufs`, plus the
    /// count of those blocks.
    fn mean_power_above(&self, threshold_lufs: f64) -> Option<(f64, u64)> {
        let mut sum_power = 0.0f64;
        let mut count = 0u64;
        for (index, &bin_count) in self.bins.iter().enumerate() {
            if bin_count == 0 {
                continue;
            }
            let loudness = bin_loudness(index);
            if loudness >= threshold_lufs {
                sum_power += bin_count as f64 * power_from_loudness(loudness);
                count += bin_count as u64;
            }
        }
        if count == 0 {
            None
        } else {
            Some((sum_power / count as f64, count))
        }
    }

    /// The relative gate threshold in LUFS: the loudness of the absolute-gated
    /// mean power, minus `offset_lu` (−10 for integrated, −20 for LRA).
    fn relative_threshold(&self, offset_lu: f64) -> Option<f64> {
        // The absolute gate is already applied on insertion; the mean over the
        // whole histogram is therefore the absolute-gated mean.
        let (mean_power, _) = self.mean_power_above(ABSOLUTE_GATE_LUFS)?;
        Some(loudness_from_power(mean_power) - offset_lu)
    }

    /// **Integrated loudness** (LUFS): the loudness of the mean power over
    /// blocks passing both the absolute (−70) and relative (−10 LU) gates.
    /// `None` until at least one block clears the gates.
    pub fn integrated_lufs(&self) -> Option<f32> {
        let threshold = self.relative_threshold(10.0)?;
        let (mean_power, _) = self.mean_power_above(threshold)?;
        Some(loudness_from_power(mean_power) as f32)
    }

    /// **Loudness range** (LU): the spread between the `high_pct`th and
    /// `low_pct`th percentiles of the relatively-gated (−20 LU) distribution,
    /// per EBU Tech 3342 (typically 10th–95th).  `None` until enough gated
    /// blocks exist.
    pub fn range_lu(&self, low_pct: f64, high_pct: f64) -> Option<f32> {
        let threshold = self.relative_threshold(20.0)?;

        // Total count of gated blocks.
        let mut gated_total = 0u64;
        for (index, &count) in self.bins.iter().enumerate() {
            if count > 0 && bin_loudness(index) >= threshold {
                gated_total += count as u64;
            }
        }
        if gated_total == 0 {
            return None;
        }

        let low = self.percentile(threshold, gated_total, low_pct)?;
        let high = self.percentile(threshold, gated_total, high_pct)?;
        Some((high - low) as f32)
    }

    /// Loudness at the `pct`th percentile of the gated distribution (bins with
    /// loudness `>= threshold`), by cumulative count.
    fn percentile(&self, threshold: f64, gated_total: u64, pct: f64) -> Option<f64> {
        let target = (pct / 100.0 * gated_total as f64).ceil() as u64;
        let target = target.clamp(1, gated_total);
        let mut cumulative = 0u64;
        for (index, &count) in self.bins.iter().enumerate() {
            if count == 0 || bin_loudness(index) < threshold {
                continue;
            }
            cumulative += count as u64;
            if cumulative >= target {
                return Some(bin_loudness(index));
            }
        }
        None
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_histogram_reports_nothing() {
        let hist = LoudnessHistogram::new();
        assert_eq!(hist.total(), 0);
        assert!(hist.integrated_lufs().is_none());
        assert!(hist.range_lu(10.0, 95.0).is_none());
    }

    #[test]
    fn absolute_gate_drops_silence() {
        let mut hist = LoudnessHistogram::new();
        hist.add(-80.0); // below −70 → dropped
        hist.add(f32::NEG_INFINITY); // silent block → dropped
        assert_eq!(hist.total(), 0);
        hist.add(-23.0);
        assert_eq!(hist.total(), 1);
    }

    #[test]
    fn integrated_of_constant_blocks_equals_that_loudness() {
        // A stream of identical −23 LUFS blocks integrates to −23 LUFS: the
        // relative gate (−10 LU) sits at −33 and keeps them all.
        let mut hist = LoudnessHistogram::new();
        for _ in 0..100 {
            hist.add(-23.0);
        }
        let integrated = hist.integrated_lufs().expect("some blocks");
        assert!(
            (integrated - -23.0).abs() < 0.1,
            "integrated {integrated} should be ~-23"
        );
    }

    #[test]
    fn relative_gate_excludes_quiet_tail() {
        // Loud programme (−20) plus a much quieter tail (−45).  The −45 blocks
        // fall below the relative gate (mean ≈ −20 → gate ≈ −30) and are
        // excluded, so the integrated result stays near −20, not the average.
        let mut hist = LoudnessHistogram::new();
        for _ in 0..100 {
            hist.add(-20.0);
        }
        for _ in 0..100 {
            hist.add(-45.0);
        }
        let integrated = hist.integrated_lufs().expect("some blocks");
        assert!(
            (integrated - -20.0).abs() < 0.5,
            "integrated {integrated} should stay near the loud programme"
        );
    }

    #[test]
    fn range_spans_the_distribution() {
        // Blocks spread evenly from −30 to −10 LUFS → LRA ≈ the 10–95 %
        // percentile spread, a good fraction of the 20 LU total span.
        let mut hist = LoudnessHistogram::new();
        for step in 0..=200 {
            let loudness = -30.0 + step as f32 * 0.1;
            hist.add(loudness);
        }
        let range = hist.range_lu(10.0, 95.0).expect("enough blocks");
        assert!(
            (14.0..=20.0).contains(&range),
            "LRA {range} LU outside expected 14–20 span"
        );
    }

    #[test]
    fn reset_clears_counts() {
        let mut hist = LoudnessHistogram::new();
        hist.add(-23.0);
        hist.reset();
        assert_eq!(hist.total(), 0);
        assert!(hist.integrated_lufs().is_none());
    }
}
