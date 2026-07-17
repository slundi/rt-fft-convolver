//! Log-spaced grouping of FFT bins into vocoder bands.
//!
//! A classic analog vocoder splits the spectrum into a handful of band-pass
//! channels (8 / 16 / 24 / 32).  The FFT-based engine reproduces that control
//! by grouping its linear FFT bins into the same number of **log-spaced** bands
//! — matching how we hear pitch and preserving the familiar `Bands` character
//! (Roadmap-08 Phase 0 vocoder-topology decision).
//!
//! Band edges are spaced geometrically between a low corner and Nyquist, so low
//! bands are narrow (a few bins) and high bands are wide (many bins).  DC and
//! any bins below the low corner fall into band 0.

/// Precomputed mapping from FFT bins to a fixed number of log-spaced bands.
///
/// Built once off the audio thread; [`band_of_bin`](Self::band_of_bin) and
/// [`bin_range`](Self::bin_range) are cheap lookups on the audio thread.
#[derive(Clone, Debug)]
pub struct LogBands {
    /// Number of bands.
    count: usize,
    /// `count + 1` bin boundaries; band `b` covers `edges[b]..edges[b + 1]`.
    edges: Vec<usize>,
    /// Per-bin band index (length `bins`), so a bin → band lookup is O(1).
    bin_band: Vec<usize>,
}

impl LogBands {
    /// Group the `bins` (`frame / 2 + 1`) unique real-FFT bins of a `frame`-point
    /// transform at `sample_rate` into `count` log-spaced bands starting at
    /// `low_hz`.
    ///
    /// **Allocates** — off-RT only.
    ///
    /// # Panics
    /// Panics if `count == 0`, `bins < 2`, or `low_hz` is not below Nyquist.
    pub fn new(count: usize, bins: usize, frame: usize, sample_rate: f32, low_hz: f32) -> Self {
        assert!(count > 0, "need at least one band");
        assert!(bins >= 2, "need at least two bins");
        let nyquist = sample_rate * 0.5;
        assert!(
            0.0 < low_hz && low_hz < nyquist,
            "low_hz must be in (0, Nyquist)"
        );

        let bin_hz = sample_rate / frame as f32;
        // Clamp the low corner to at least bin 1 (keep DC in band 0).
        let low_bin = ((low_hz / bin_hz).floor() as usize).clamp(1, bins - 1);
        let top_bin = bins - 1;

        // Geometric edges from `low_bin` to `top_bin`; edge 0 is always bin 0 so
        // DC and sub-corner bins join band 0.
        let ratio = (top_bin as f32 / low_bin as f32).powf(1.0 / count as f32);
        let mut edges = Vec::with_capacity(count + 1);
        edges.push(0);
        for band in 1..count {
            let edge = (low_bin as f32 * ratio.powi(band as i32)).round() as usize;
            // Keep edges strictly increasing and inside range.
            let previous = *edges.last().unwrap();
            edges.push(edge.clamp(previous + 1, top_bin).min(top_bin));
        }
        edges.push(bins); // exclusive upper bound of the last band

        // Enforce monotonicity after the clamps (narrow spectra can collide).
        for index in 1..edges.len() {
            if edges[index] <= edges[index - 1] {
                edges[index] = (edges[index - 1] + 1).min(bins);
            }
        }

        // Precompute per-bin band index.
        let mut bin_band = vec![0usize; bins];
        for band in 0..count {
            bin_band[edges[band]..edges[band + 1]].fill(band);
        }

        Self {
            count,
            edges,
            bin_band,
        }
    }

    /// Number of bands.
    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    /// The band that FFT bin `bin` belongs to.
    #[inline]
    pub fn band_of_bin(&self, bin: usize) -> usize {
        self.bin_band[bin]
    }

    /// The half-open FFT-bin range `[start, end)` of band `band`.
    #[inline]
    pub fn bin_range(&self, band: usize) -> (usize, usize) {
        (self.edges[band], self.edges[band + 1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: usize = 1024;
    const BINS: usize = FRAME / 2 + 1;
    const SAMPLE_RATE: f32 = 48_000.0;

    #[test]
    fn covers_every_bin_exactly_once() {
        let bands = LogBands::new(16, BINS, FRAME, SAMPLE_RATE, 80.0);
        let mut covered = vec![false; BINS];
        for band in 0..bands.count() {
            let (start, end) = bands.bin_range(band);
            for (offset, slot) in covered[start..end].iter_mut().enumerate() {
                assert!(!*slot, "bin {} covered twice", start + offset);
                *slot = true;
            }
        }
        assert!(covered.iter().all(|&c| c), "some bin left uncovered");
    }

    #[test]
    fn bin_lookup_matches_ranges() {
        let bands = LogBands::new(24, BINS, FRAME, SAMPLE_RATE, 80.0);
        for band in 0..bands.count() {
            let (start, end) = bands.bin_range(band);
            for bin in start..end {
                assert_eq!(bands.band_of_bin(bin), band, "bin {bin}");
            }
        }
    }

    #[test]
    fn edges_are_strictly_increasing() {
        let bands = LogBands::new(32, BINS, FRAME, SAMPLE_RATE, 80.0);
        for band in 0..bands.count() {
            let (start, end) = bands.bin_range(band);
            assert!(start < end, "band {band}: {start}..{end} not increasing");
        }
    }

    #[test]
    fn low_bands_are_narrower_than_high_bands() {
        let bands = LogBands::new(16, BINS, FRAME, SAMPLE_RATE, 80.0);
        let (lo_s, lo_e) = bands.bin_range(1);
        let (hi_s, hi_e) = bands.bin_range(bands.count() - 1);
        assert!(
            (hi_e - hi_s) > (lo_e - lo_s),
            "log spacing should widen high bands"
        );
    }

    #[test]
    fn handles_more_bands_than_bins_gracefully() {
        // Tiny frame, many bands — must stay monotonic without panicking.
        let frame = 32;
        let bins = frame / 2 + 1; // 17
        let bands = LogBands::new(16, bins, frame, SAMPLE_RATE, 200.0);
        for band in 0..bands.count() {
            let (start, end) = bands.bin_range(band);
            assert!(start < end);
        }
    }
}
