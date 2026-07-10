//! [`PreparedIr`] — an impulse response resampled to the host rate and ready
//! to be handed to a convolution engine.
//!
//! # What "prepared" means
//!
//! The expensive, non-real-time step of getting an IR ready has two parts:
//!
//! 1. **Load + resample** — decode a WAV file and convert it to the host's
//!    session sample rate.  This is what `PreparedIr` captures.
//! 2. **Partition** — split the resampled samples into FFT blocks inside a
//!    convolution engine (`UniformPartitionEngine`, `HybridConvolver`, …).
//!
//! Step 2 is *engine-specific* (a cab uses [`HybridConvolver`], a reverb uses
//! [`StereoConvolver`]/[`TrueStereoConvolver`]), so it is deliberately left to
//! the plugin: `PreparedIr` stops at the deinterleaved, resampled samples that
//! every engine needs as input.  Both steps run off the audio thread.
//!
//! [`HybridConvolver`]: rt_fft_convolver::HybridConvolver
//! [`StereoConvolver`]: rt_fft_convolver::StereoConvolver
//! [`TrueStereoConvolver`]: rt_fft_convolver::TrueStereoConvolver

use std::path::Path;

use rt_fft_convolver::{LoadError, load_ir};

/// An impulse response that has been loaded and resampled to a target rate.
///
/// Channels are deinterleaved: `channels[0]` is left/mono, `channels[1]` is
/// right, and `channels[2..4]` carry the extra channels of a 4-channel
/// true-stereo IR.  All channels share the same length and `sample_rate`.
#[derive(Debug, Clone)]
pub struct PreparedIr {
    /// One `Vec<f32>` per channel, deinterleaved and resampled.
    channels: Vec<Vec<f32>>,
    /// Sample rate of the stored samples (equals the requested host rate).
    sample_rate: u32,
}

impl PreparedIr {
    /// Load a WAV file and resample it to `target_sample_rate`.
    ///
    /// **Allocates and does file I/O** — call from the control / worker
    /// thread, never from `process()`.
    ///
    /// # Errors
    ///
    /// Returns [`LoadError`] if the file cannot be opened, decoded, or
    /// resampled.
    pub fn load(path: impl AsRef<Path>, target_sample_rate: u32) -> Result<Self, LoadError> {
        let ir = load_ir(path, target_sample_rate)?;
        Ok(Self {
            channels: ir.channels,
            sample_rate: ir.sample_rate,
        })
    }

    /// Build a `PreparedIr` directly from already-deinterleaved channel data.
    ///
    /// Useful for tests and for callers that resample themselves.  Channels
    /// shorter than the longest are left as-is (the engine zero-pads).
    ///
    /// # Panics
    ///
    /// Panics if `channels` is empty (an IR must have at least one channel).
    pub fn from_channels(channels: Vec<Vec<f32>>, sample_rate: u32) -> Self {
        assert!(
            !channels.is_empty(),
            "PreparedIr needs at least one channel"
        );
        Self {
            channels,
            sample_rate,
        }
    }

    /// Sample rate of the stored samples.
    #[inline]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Number of audio channels (1 = mono, 2 = stereo, 4 = true stereo).
    #[inline]
    pub fn num_channels(&self) -> usize {
        self.channels.len()
    }

    /// Length of the IR in frames (samples per channel).
    #[inline]
    pub fn len(&self) -> usize {
        self.channels.first().map_or(0, Vec::len)
    }

    /// Whether the IR carries no samples.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow one channel's samples, or `None` if `index` is out of range.
    #[inline]
    pub fn channel(&self, index: usize) -> Option<&[f32]> {
        self.channels.get(index).map(Vec::as_slice)
    }

    /// Borrow all channels at once.
    #[inline]
    pub fn channels(&self) -> &[Vec<f32>] {
        &self.channels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_channels_reports_shape() {
        let ir = PreparedIr::from_channels(vec![vec![1.0, 0.0, 0.0], vec![0.5, 0.0, 0.0]], 48_000);
        assert_eq!(ir.num_channels(), 2);
        assert_eq!(ir.len(), 3);
        assert_eq!(ir.sample_rate(), 48_000);
        assert!(!ir.is_empty());
        assert_eq!(ir.channel(0), Some(&[1.0, 0.0, 0.0][..]));
        assert_eq!(ir.channel(1), Some(&[0.5, 0.0, 0.0][..]));
        assert_eq!(ir.channel(2), None);
    }

    #[test]
    fn empty_channel_reports_zero_length() {
        let ir = PreparedIr::from_channels(vec![Vec::new()], 44_100);
        assert!(ir.is_empty());
        assert_eq!(ir.len(), 0);
        assert_eq!(ir.num_channels(), 1);
    }

    #[test]
    #[should_panic(expected = "at least one channel")]
    fn no_channels_panics() {
        let _ = PreparedIr::from_channels(Vec::new(), 48_000);
    }

    /// Round-trip through a real WAV file: write a known mono impulse, load it
    /// back at the same rate, and confirm the samples survive intact.
    #[test]
    fn load_reads_and_matches_wav() {
        // Unique path per test to avoid collisions under parallel test runs.
        let path = std::env::temp_dir().join("rt_ir_core_prepared_ir_load.wav");

        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let samples = [1.0f32, -0.5, 0.25, 0.0];
        {
            let mut writer = hound::WavWriter::create(&path, spec).expect("create wav");
            for &s in &samples {
                writer.write_sample(s).expect("write sample");
            }
            writer.finalize().expect("finalize wav");
        }

        let ir = PreparedIr::load(&path, 48_000).expect("load wav");
        assert_eq!(ir.num_channels(), 1);
        assert_eq!(ir.sample_rate(), 48_000);
        assert_eq!(ir.len(), samples.len());
        let loaded = ir.channel(0).expect("channel 0");
        for (i, (&got, &want)) in loaded.iter().zip(samples.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "sample {i}: got {got}, want {want}"
            );
        }

        let _ = std::fs::remove_file(&path);
    }
}
