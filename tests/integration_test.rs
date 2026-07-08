//! End-to-end integration tests.
//!
//! These tests exercise the public crate surface (`rt_fft_convolver::…`) the
//! way a downstream user would: write a real WAV file to disk, load it via
//! [`load_ir`], and drive it through the full render chain
//! (`UniformPartitionEngine` + `Mixer`).

use rt_fft_convolver::{Mixer, UniformPartitionEngine, load_ir, resample};

// ─── Local helpers ────────────────────────────────────────────────────────────

/// Tiny scoped temp dir that cleans up on drop — avoids pulling in `tempfile`
/// as a dev-dependency just for these tests.
fn tempdir() -> TempDir {
    let base = std::env::temp_dir().join(format!(
        "rt-fft-convolver-it-{}-{}",
        std::process::id(),
        rand_suffix()
    ));
    std::fs::create_dir_all(&base).expect("create tempdir");
    TempDir(base)
}

fn rand_suffix() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

struct TempDir(std::path::PathBuf);

impl TempDir {
    fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn write_wav_mono_i16(path: &std::path::Path, sample_rate: u32, samples: &[f32]) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).expect("open wav writer");
    for &s in samples {
        w.write_sample((s * i16::MAX as f32) as i16).unwrap();
    }
    w.finalize().unwrap();
}

fn write_wav_stereo_i16(path: &std::path::Path, sample_rate: u32, left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), right.len());
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).expect("open wav writer");
    for (&l, &r) in left.iter().zip(right.iter()) {
        w.write_sample((l * i16::MAX as f32) as i16).unwrap();
        w.write_sample((r * i16::MAX as f32) as i16).unwrap();
    }
    w.finalize().unwrap();
}

/// Drive `input` through `UniformPartitionEngine + Mixer` in `block_size`
/// blocks and return the concatenated output.
fn render_chain(ir: &[f32], input: &[f32], block_size: usize, dry_wet: f32) -> Vec<f32> {
    let mut engine = UniformPartitionEngine::new(ir, block_size);
    let mixer = Mixer::new(ir, dry_wet);

    let mut wet = vec![0.0f32; block_size];
    let mut mixed = vec![0.0f32; block_size];
    let mut out = Vec::with_capacity(input.len());

    for chunk in input.chunks(block_size) {
        // Pad the final chunk with silence so callers can drive tails out.
        let mut in_block = vec![0.0f32; block_size];
        in_block[..chunk.len()].copy_from_slice(chunk);

        wet.fill(0.0);
        engine.process(&in_block, &mut wet);
        mixer.process(&in_block, &wet, &mut mixed);

        out.extend_from_slice(&mixed);
    }
    out
}

// ─── Tests ────────────────────────────────────────────────────────────────────

/// Load a mono WAV at the native rate — no resampling — then drive an impulse
/// through the render chain and compare against the loaded IR scaled by
/// `ir_gain` (Mixer normalises to unit energy at fully-wet).
#[test]
fn end_to_end_mono_wav_impulse_response() {
    let dir = tempdir();
    let path = dir.path().join("test_ir.wav");
    let sample_rate = 48_000_u32;

    // 9-tap IR — deliberately spans a partition boundary at block_size=4.
    let ir_samples: Vec<f32> = vec![
        0.5, -0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0, -0.5, 0.25,
    ];
    write_wav_mono_i16(&path, sample_rate, &ir_samples);

    let ir = load_ir(&path, sample_rate).expect("load_ir");
    assert_eq!(ir.sample_rate, sample_rate);
    assert_eq!(ir.num_channels, 1);
    assert_eq!(ir.channels[0].len(), ir_samples.len());

    // Feed an impulse and read back the IR shape.
    let block_size = 4;
    let n_blocks = 4;
    let mut input = vec![0.0f32; block_size * n_blocks];
    input[0] = 1.0;
    let out = render_chain(&ir.channels[0], &input, block_size, 1.0);

    // Reference: loaded IR × ir_gain, followed by silence.
    let energy: f32 = ir.channels[0].iter().map(|&x| x * x).sum();
    let ir_gain = 1.0 / energy.sqrt();
    let mut expected = vec![0.0f32; input.len()];
    for (i, &h) in ir.channels[0].iter().enumerate() {
        expected[i] = h * ir_gain;
    }

    // Tolerance: 16-bit quantisation × FFT roundtrip, scaled by peak output.
    let peak = expected.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let tol = peak * 1e-3 + 1e-4;
    for (i, (e, y)) in expected.iter().zip(out.iter()).enumerate() {
        let err = (e - y).abs();
        assert!(
            err < tol,
            "sample {i}: expected {e}, got {y} (abs err {err} > tol {tol})"
        );
    }
}

/// Load a stereo WAV at 44.1 kHz targeting 48 kHz — exercises the resampler
/// on both channels.  Verifies that both channels decode, deinterleave, and
/// resample without corrupting each other.
#[test]
fn end_to_end_stereo_wav_resamples_both_channels() {
    let dir = tempdir();
    let path = dir.path().join("stereo_ir.wav");

    // Left = ramp, Right = negative ramp — distinct so a swap would be visible.
    let n = 2048;
    let left: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
    let right: Vec<f32> = (0..n).map(|i| -(i as f32) / n as f32).collect();
    write_wav_stereo_i16(&path, 44_100, &left, &right);

    let ir = load_ir(&path, 48_000).expect("load_ir");
    assert_eq!(ir.num_channels, 2);
    assert_eq!(ir.channels.len(), 2);
    assert_eq!(ir.sample_rate, 48_000);

    let expected_len = (n as f64 * 48_000.0 / 44_100.0).ceil() as usize;
    assert_eq!(ir.channels[0].len(), expected_len);
    assert_eq!(ir.channels[1].len(), expected_len);

    // Sanity: mid-signal averages match the source ramps within ±5%.
    let mid_l: f32 =
        ir.channels[0][100..expected_len - 100].iter().sum::<f32>() / (expected_len - 200) as f32;
    let mid_r: f32 =
        ir.channels[1][100..expected_len - 100].iter().sum::<f32>() / (expected_len - 200) as f32;
    assert!(
        (mid_l - 0.5).abs() < 0.05,
        "L average {mid_l} deviates from expected 0.5"
    );
    assert!(
        (mid_r + 0.5).abs() < 0.05,
        "R average {mid_r} deviates from expected -0.5"
    );
}

/// Round-trip: write an IR, load it, feed an impulse, and confirm energy
/// conservation — the Mixer's `ir_gain` normalisation should keep the
/// impulse-response output energy close to 1.0 regardless of IR shape.
#[test]
fn end_to_end_mixer_normalises_output_energy() {
    let dir = tempdir();
    let path = dir.path().join("energy_ir.wav");

    // Deterministic 32-tap "noisy" IR.
    let ir_samples: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.9).sin() * 0.3).collect();
    write_wav_mono_i16(&path, 48_000, &ir_samples);

    let ir = load_ir(&path, 48_000).expect("load_ir");
    let block_size = 8;
    let n_blocks = 8; // enough to flush the tail
    let mut input = vec![0.0f32; block_size * n_blocks];
    input[0] = 1.0;

    let out = render_chain(&ir.channels[0], &input, block_size, 1.0);
    let out_energy: f32 = out.iter().map(|&x| x * x).sum();

    // ir_gain uses the LOADED IR (after 16-bit quantisation), and Mixer::new
    // was called with the same loaded IR, so energy is self-consistent.
    // Expected ≈ 1.0 ± quantisation + FFT roundtrip error.
    assert!(
        (out_energy - 1.0).abs() < 5e-3,
        "output energy {out_energy} deviated from expected 1.0"
    );
}

/// The exported `resample` function is reachable through the crate root and
/// behaves consistently between the same-rate short-circuit and the actual
/// resampler.
#[test]
fn resample_is_reexported_and_functional() {
    let input: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
    let same = resample(&input, 44_100, 44_100).unwrap();
    assert_eq!(same, input, "same-rate must be identity");

    let up = resample(&input, 44_100, 48_000).unwrap();
    let ratio = 48_000.0 / 44_100.0;
    let expected_len = (input.len() as f64 * ratio).round() as usize;
    assert!(
        (up.len() as i64 - expected_len as i64).abs() <= 1,
        "up-sample length {} not within ±1 of expected {expected_len}",
        up.len()
    );
}
