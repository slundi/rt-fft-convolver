//! WAV file loading and sample-rate conversion for impulse responses.
//!
//! # Overview
//!
//! Impulse response (IR) files are typically recorded at a fixed sample rate
//! (44 100 Hz or 48 000 Hz) but must match the host's current session sample
//! rate before convolution.  This module provides:
//!
//! - [`load_ir`] — open a WAV file and, if necessary, resample it to the
//!   target rate in one call.
//! - [`resample`] — resample a raw `f32` slice between any two integer rates.
//! - [`IrData`] — the result type carrying deinterleaved per-channel samples.
//! - [`LoadError`] — unified error type covering I/O, WAV decoding, and
//!   resampling failures.
//!
//! # Example
//!
//! ```rust,no_run
//! # use rt_fft_convolver::{load_ir, LoadError};
//! let ir = load_ir("cabinet.wav", 48_000)?;
//! assert_eq!(ir.sample_rate, 48_000);
//! assert_eq!(ir.num_channels, 1); // mono IR
//! let samples: &[f32] = &ir.channels[0];
//! # Ok::<(), LoadError>(())
//! ```

use std::path::Path;

// rubato 2.0 re-exports `audioadapter_buffers` so we can reach it through
// the rubato namespace without adding an extra Cargo dependency.
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};

// ─── Public types ─────────────────────────────────────────────────────────────

/// Deinterleaved impulse response data returned by [`load_ir`].
///
/// Each element of `channels` holds the `f32` samples for one audio channel:
/// - index 0 = left (or mono)
/// - index 1 = right
/// - indices 2 / 3 = extra channels for true-stereo IRs recorded as 4-ch WAV
#[derive(Debug, Clone)]
pub struct IrData {
    /// One `Vec<f32>` per channel, deinterleaved from the source file.
    pub channels: Vec<Vec<f32>>,
    /// Number of audio channels (1 = mono, 2 = stereo, 4 = true stereo, …).
    pub num_channels: u16,
    /// Sample rate of the stored samples — equals the `target_sample_rate`
    /// argument passed to [`load_ir`].
    pub sample_rate: u32,
}

/// Errors that can occur while loading or resampling an impulse response.
#[derive(Debug)]
pub enum LoadError {
    /// Underlying I/O error (file not found, permissions, …).
    Io(std::io::Error),
    /// WAV parsing or encoding error from the `hound` crate.
    Hound(hound::Error),
    /// Failed to construct the resampler (bad rate pair, zero channels, …).
    Resample(rubato::ResamplerConstructionError),
    /// Runtime resampling error (buffer-size mismatch, …).
    ResampleProcess(rubato::ResampleError),
    /// The WAV file uses a bit depth or encoding that this loader does not
    /// support (e.g., 64-bit float PCM).
    UnsupportedFormat(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "I/O error: {e}"),
            LoadError::Hound(e) => write!(f, "WAV error: {e}"),
            LoadError::Resample(e) => write!(f, "resampler construction error: {e}"),
            LoadError::ResampleProcess(e) => write!(f, "resampling error: {e}"),
            LoadError::UnsupportedFormat(s) => write!(f, "unsupported WAV format: {s}"),
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LoadError::Io(e) => Some(e),
            LoadError::Hound(e) => Some(e),
            // rubato's error types do not expose a cause chain we can forward
            LoadError::Resample(_) | LoadError::ResampleProcess(_) => None,
            LoadError::UnsupportedFormat(_) => None,
        }
    }
}

impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        LoadError::Io(e)
    }
}

impl From<hound::Error> for LoadError {
    fn from(e: hound::Error) -> Self {
        LoadError::Hound(e)
    }
}

// ─── Public functions ─────────────────────────────────────────────────────────

/// Load a WAV file and resample it to `target_sample_rate` if necessary.
///
/// All common WAV variants are supported:
/// - 16-bit integer PCM (most common for guitar cab IRs)
/// - 24-bit and 32-bit integer PCM
/// - 32-bit IEEE float PCM
///
/// The returned [`IrData`] contains one `Vec<f32>` per channel, deinterleaved
/// and normalised to the `[-1.0, 1.0]` range.
///
/// # Errors
///
/// Returns [`LoadError`] if the file cannot be opened, decoded, or resampled.
pub fn load_ir(path: impl AsRef<Path>, target_sample_rate: u32) -> Result<IrData, LoadError> {
    let reader = hound::WavReader::open(path).map_err(LoadError::Hound)?;
    load_ir_from_reader(reader, target_sample_rate)
}

/// Resample `samples` from `from_rate` Hz to `to_rate` Hz.
///
/// Uses rubato's FFT-based resampler ([`rubato::Fft`] with fixed input chunks)
/// for high spectral quality.  Suitable for non-real-time use (IR
/// preparation); **not** intended for the audio render thread — allocates.
///
/// Returns the input unchanged (cloned) when `from_rate == to_rate` or when
/// the input is empty.
///
/// # Errors
///
/// Returns [`LoadError::Resample`] if rubato rejects the rate pair, or
/// [`LoadError::ResampleProcess`] on a buffer-size mismatch.
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>, LoadError> {
    if from_rate == to_rate || samples.is_empty() {
        return Ok(samples.to_vec());
    }

    // 1 024-frame chunks: good balance between quality and per-call overhead.
    // sub_chunks = 2 widens the internal FFT window for better stopband attenuation.
    const CHUNK: usize = 1024;
    let n_frames = samples.len(); // mono: 1 sample = 1 frame

    let mut resampler = Fft::<f32>::new(
        from_rate as usize,
        to_rate as usize,
        CHUNK,
        2, // sub-chunks (higher → better stopband, more memory)
        1, // mono — callers deinterleave before calling us
        FixedSync::Input,
    )
    .map_err(LoadError::Resample)?;

    // Allocate the output buffer large enough for `process_all_into_buffer`,
    // which includes the resampler's internal delay and a chunk of headroom.
    let out_buf_len = resampler.process_all_needed_output_len(n_frames);
    let mut outdata = vec![0.0_f32; out_buf_len];

    // Wrap the raw slices in adapters that the rubato Resampler trait accepts.
    let input = InterleavedSlice::new(samples, 1, n_frames)
        .map_err(|e| LoadError::UnsupportedFormat(e.to_string()))?;
    let mut output = InterleavedSlice::new_mut(&mut outdata, 1, out_buf_len)
        .map_err(|e| LoadError::UnsupportedFormat(e.to_string()))?;

    // `process_all_into_buffer` handles chunking, partial last chunk,
    // and trimming the leading silence caused by the resampler delay.
    // It returns (input_frames_consumed, output_frames_written).
    let (_, out_frames) = resampler
        .process_all_into_buffer(&input, &mut output, n_frames, None)
        .map_err(LoadError::ResampleProcess)?;

    outdata.truncate(out_frames);
    Ok(outdata)
}

// ─── Crate-internal helpers ───────────────────────────────────────────────────

/// Core implementation shared between [`load_ir`] (file path) and the
/// in-memory path used in unit tests.
pub(crate) fn load_ir_from_reader<R: std::io::Read>(
    mut reader: hound::WavReader<R>,
    target_sample_rate: u32,
) -> Result<IrData, LoadError> {
    let spec = reader.spec();
    let file_sample_rate = spec.sample_rate;
    // hound::WavSpec uses `channels` (not `num_channels`)
    let num_channels = spec.channels;
    let num_ch = num_channels as usize;

    // ── Decode to interleaved f32 ─────────────────────────────────────────────
    let interleaved = decode_samples(&mut reader, &spec)?;

    // ── Deinterleave: [L0, R0, L1, R1, …] → [[L0, L1, …], [R0, R1, …]] ─────
    let mut channels: Vec<Vec<f32>> = (0..num_ch)
        .map(|ch| {
            interleaved
                .iter()
                .skip(ch)
                .step_by(num_ch)
                .copied()
                .collect()
        })
        .collect();

    // ── Resample each channel independently if the file rate differs ──────────
    if file_sample_rate != target_sample_rate {
        for ch in channels.iter_mut() {
            *ch = resample(ch, file_sample_rate, target_sample_rate)?;
        }
    }

    Ok(IrData {
        channels,
        num_channels,
        sample_rate: target_sample_rate,
    })
}

/// Decode all samples from an open WAV reader into an interleaved `f32` vec.
///
/// Supported formats:
/// | WAV format       | hound type | normalisation          |
/// |------------------|------------|------------------------|
/// | 16-bit int PCM   | `i16`      | ÷ i16::MAX (≈ 3.1e-5)  |
/// | 24-bit int PCM   | `i32`      | ÷ 2^23                 |
/// | 32-bit int PCM   | `i32`      | ÷ 2^31                 |
/// | 32-bit float PCM | `f32`      | as-is                  |
fn decode_samples<R: std::io::Read>(
    reader: &mut hound::WavReader<R>,
    spec: &hound::WavSpec,
) -> Result<Vec<f32>, LoadError> {
    use hound::SampleFormat;

    match (spec.sample_format, spec.bits_per_sample) {
        // ── 32-bit float: no conversion needed ───────────────────────────────
        (SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .map(|s| s.map_err(LoadError::Hound))
            .collect(),

        // ── 16-bit int: divide by i16::MAX ───────────────────────────────────
        (SampleFormat::Int, 16) => {
            let scale = 1.0 / i16::MAX as f32; // ≈ 3.05e-5
            reader
                .samples::<i16>()
                .map(|s| s.map(|x| x as f32 * scale).map_err(LoadError::Hound))
                .collect()
        }

        // ── 24-bit and 32-bit int: hound packs both into i32 ─────────────────
        // 2^(bits-1) gives the positive half-range for the bit depth.
        (SampleFormat::Int, bits @ (24 | 32)) => {
            let scale = 1.0 / (1_u64 << (bits - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|x| x as f32 * scale).map_err(LoadError::Hound))
                .collect()
        }

        (fmt, bits) => Err(LoadError::UnsupportedFormat(format!("{fmt:?} {bits}-bit"))),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Build a mono 16-bit WAV in memory containing `samples`.
    fn make_wav_mono_i16(sample_rate: u32, samples: &[f32]) -> Cursor<Vec<u8>> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut buf = Cursor::new(Vec::new());
        let mut w = hound::WavWriter::new(&mut buf, spec).unwrap();
        for &s in samples {
            w.write_sample((s * i16::MAX as f32) as i16).unwrap();
        }
        w.finalize().unwrap();
        buf.set_position(0);
        buf
    }

    /// Build a stereo 16-bit WAV in memory; `l` and `r` must have equal length.
    fn make_wav_stereo_i16(sample_rate: u32, l: &[f32], r: &[f32]) -> Cursor<Vec<u8>> {
        assert_eq!(l.len(), r.len());
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut buf = Cursor::new(Vec::new());
        let mut w = hound::WavWriter::new(&mut buf, spec).unwrap();
        for (&ls, &rs) in l.iter().zip(r.iter()) {
            w.write_sample((ls * i16::MAX as f32) as i16).unwrap();
            w.write_sample((rs * i16::MAX as f32) as i16).unwrap();
        }
        w.finalize().unwrap();
        buf.set_position(0);
        buf
    }

    /// Build a mono 32-bit float WAV in memory.
    fn make_wav_mono_f32(sample_rate: u32, samples: &[f32]) -> Cursor<Vec<u8>> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut buf = Cursor::new(Vec::new());
        let mut w = hound::WavWriter::new(&mut buf, spec).unwrap();
        for &s in samples {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();
        buf.set_position(0);
        buf
    }

    // ── resample ──────────────────────────────────────────────────────────────

    /// Same rate → clone of input, no resampler constructed.
    #[test]
    fn resample_same_rate_is_identity() {
        let input: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        let output = resample(&input, 44_100, 44_100).unwrap();
        assert_eq!(output, input);
    }

    /// Empty input → empty output regardless of rates.
    #[test]
    fn resample_empty_returns_empty() {
        let out = resample(&[], 44_100, 48_000).unwrap();
        assert!(out.is_empty());
    }

    /// Upsampling 44 100 → 48 000: output length must match the ratio.
    #[test]
    fn resample_upsample_length() {
        let n = 4410_usize; // 100 ms at 44 100 Hz
        let input = vec![0.5_f32; n];
        let output = resample(&input, 44_100, 48_000).unwrap();
        let expected = (n as f64 * 48_000.0 / 44_100.0).round() as usize;
        assert_eq!(
            output.len(),
            expected,
            "upsample length mismatch: got {} expected {}",
            output.len(),
            expected
        );
    }

    /// Downsampling 48 000 → 44 100: output length must match the ratio.
    #[test]
    fn resample_downsample_length() {
        let n = 4800_usize; // 100 ms at 48 000 Hz
        let input = vec![0.5_f32; n];
        let output = resample(&input, 48_000, 44_100).unwrap();
        let expected = (n as f64 * 44_100.0 / 48_000.0).round() as usize;
        assert_eq!(
            output.len(),
            expected,
            "downsample length mismatch: got {} expected {}",
            output.len(),
            expected
        );
    }

    /// A DC signal (all 1.0) should remain approximately 1.0 after resampling.
    #[test]
    fn resample_dc_signal_preserved() {
        let input = vec![1.0_f32; 4096];
        let output = resample(&input, 44_100, 48_000).unwrap();
        // Skip the first and last portions (FIR filter transient / tail).
        let mid = &output[64..output.len().saturating_sub(64)];
        assert!(!mid.is_empty());
        for &s in mid {
            assert!(
                (s - 1.0).abs() < 0.01,
                "DC deviation too large: {s} (expected ≈ 1.0)"
            );
        }
    }

    // ── load_ir_from_reader — format decoding ─────────────────────────────────

    /// Mono 16-bit WAV loaded at the same sample rate: values round-trip.
    #[test]
    fn load_ir_mono_i16_same_rate() {
        let samples = vec![0.0_f32, 0.5, -0.5, 1.0, -1.0];
        let wav = make_wav_mono_i16(48_000, &samples);
        let reader = hound::WavReader::new(wav).unwrap();
        let ir = load_ir_from_reader(reader, 48_000).unwrap();

        assert_eq!(ir.sample_rate, 48_000);
        assert_eq!(ir.num_channels, 1);
        assert_eq!(ir.channels.len(), 1);
        assert_eq!(ir.channels[0].len(), samples.len());

        // 16-bit round-trip error is at most 1 LSB ≈ 3.1e-5
        for (&got, &expected) in ir.channels[0].iter().zip(samples.iter()) {
            assert!(
                (got - expected).abs() < 1.0 / i16::MAX as f32 + 1e-6,
                "16-bit round-trip: got {got}, expected {expected}"
            );
        }
    }

    /// Stereo 16-bit WAV: channels are correctly deinterleaved.
    #[test]
    fn load_ir_stereo_i16_deinterleaved() {
        let left = vec![1.0_f32, 0.5, 0.0];
        let right = vec![0.0_f32, -0.5, -1.0];
        let wav = make_wav_stereo_i16(44_100, &left, &right);
        let reader = hound::WavReader::new(wav).unwrap();
        let ir = load_ir_from_reader(reader, 44_100).unwrap();

        assert_eq!(ir.num_channels, 2);
        assert_eq!(ir.channels.len(), 2);
        assert_eq!(ir.channels[0].len(), left.len());
        assert_eq!(ir.channels[1].len(), right.len());

        let tol = 1.0 / i16::MAX as f32 + 1e-6;
        for i in 0..left.len() {
            assert!(
                (ir.channels[0][i] - left[i]).abs() < tol,
                "L[{i}]: got {}, expected {}",
                ir.channels[0][i],
                left[i]
            );
            assert!(
                (ir.channels[1][i] - right[i]).abs() < tol,
                "R[{i}]: got {}, expected {}",
                ir.channels[1][i],
                right[i]
            );
        }
    }

    /// 32-bit float WAV: values pass through without quantisation error.
    #[test]
    fn load_ir_float32_exact() {
        let samples = vec![0.123_456_78_f32, -0.987_654_3, 0.5];
        let wav = make_wav_mono_f32(48_000, &samples);
        let reader = hound::WavReader::new(wav).unwrap();
        let ir = load_ir_from_reader(reader, 48_000).unwrap();

        assert_eq!(ir.channels[0], samples, "float32 round-trip must be exact");
    }

    // ── load_ir_from_reader — resampling ──────────────────────────────────────

    /// WAV recorded at 44 100 Hz loaded at 48 000 Hz: output length matches ratio.
    #[test]
    fn load_ir_resamples_mono() {
        let wav_sr = 44_100_u32;
        let target_sr = 48_000_u32;
        let n = 4410_usize; // 100 ms

        let input: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let wav = make_wav_mono_i16(wav_sr, &input);
        let reader = hound::WavReader::new(wav).unwrap();
        let ir = load_ir_from_reader(reader, target_sr).unwrap();

        let expected_len = (n as f64 * target_sr as f64 / wav_sr as f64).round() as usize;
        assert_eq!(ir.sample_rate, target_sr);
        assert_eq!(ir.num_channels, 1);
        assert_eq!(
            ir.channels[0].len(),
            expected_len,
            "resampled length mismatch"
        );
    }

    /// Stereo WAV at 44 100 Hz → 48 000 Hz: both channels resampled independently.
    #[test]
    fn load_ir_resamples_stereo_both_channels() {
        let wav_sr = 44_100_u32;
        let target_sr = 48_000_u32;
        let n = 2048_usize;

        let left = vec![0.5_f32; n];
        let right = vec![-0.5_f32; n];
        let wav = make_wav_stereo_i16(wav_sr, &left, &right);
        let reader = hound::WavReader::new(wav).unwrap();
        let ir = load_ir_from_reader(reader, target_sr).unwrap();

        // rubato's process_all_into_buffer uses ceil() for the expected output length.
        let expected_len = (n as f64 * target_sr as f64 / wav_sr as f64).ceil() as usize;
        assert_eq!(ir.num_channels, 2);
        assert_eq!(ir.channels[0].len(), expected_len, "L channel length");
        assert_eq!(ir.channels[1].len(), expected_len, "R channel length");
    }

    /// When file rate equals target rate no resampler is invoked.
    #[test]
    fn load_ir_no_resample_when_rates_match() {
        let samples = vec![1.0_f32, -1.0, 0.5, -0.5];
        let wav = make_wav_mono_f32(48_000, &samples);
        let reader = hound::WavReader::new(wav).unwrap();
        let ir = load_ir_from_reader(reader, 48_000).unwrap();
        assert_eq!(ir.channels[0], samples);
    }
}
