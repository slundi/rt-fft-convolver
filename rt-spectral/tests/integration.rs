//! End-to-end integration tests for the `rt-spectral` plugin engine.
//!
//! These exercise the *wiring* of the framework-agnostic
//! [`SpectralProcessor`] / [`SpectralController`] — multi-channel rendering,
//! the two-hand-off Learn workflow, and the operator chain reached through the
//! plugin's snapshot path — rather than the per-operator DSP (that is covered
//! by `rt-spectral-core`'s own tests).

use rt_spectral::spectral_core::{GateParams, HumParams, SpectralParams};
use rt_spectral::{clamp_params, spectral_engine};

const SR: u32 = 48_000;
const FFT: usize = 2048;
const BLOCK: usize = 256;
const TAU: f32 = std::f32::consts::TAU;

/// RMS of a slice.
fn rms(buf: &[f32]) -> f32 {
    (buf.iter().map(|x| x * x).sum::<f32>() / buf.len() as f32).sqrt()
}

/// Run a mono signal through the processor in `BLOCK`-sized chunks.
fn run_mono(signal: &[f32], params: &SpectralParams) -> Vec<f32> {
    let (_controller, mut processor, _telemetry) = spectral_engine(SR, FFT, BLOCK, 1);
    let mut out = vec![0.0f32; signal.len()];
    for (chunk_in, chunk_out) in signal.chunks(BLOCK).zip(out.chunks_mut(BLOCK)) {
        let mut block = chunk_in.to_vec();
        {
            let mut channels: [&mut [f32]; 1] = [&mut block];
            processor.process(&mut channels, params);
        }
        chunk_out.copy_from_slice(&block);
    }
    out
}

/// A pure sine at `freq` Hz, `n` samples, amplitude `amp`.
fn sine(freq: f32, amp: f32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|k| amp * (TAU * freq * k as f32 / SR as f32).sin())
        .collect()
}

/// Enabling the hum notch strongly attenuates a tone sitting on the mains
/// fundamental while an all-pass chain otherwise reconstructs the input.
#[test]
fn hum_notch_attenuates_mains_tone() {
    let tone = sine(60.0, 0.5, FFT * 24);

    let params = SpectralParams {
        hum: HumParams {
            enabled: true,
            fundamental_hz: 60.0,
            harmonics: 1,
            q: 30.0,
        },
        ..SpectralParams::default()
    };
    let out = run_mono(&tone, &params);

    // Compare settled regions past the notch's own warm-up.
    let in_rms = rms(&tone[FFT * 8..]);
    let out_rms = rms(&out[FFT * 8..]);
    assert!(
        out_rms < in_rms * 0.1,
        "60 Hz tone not notched: in {in_rms} out {out_rms}"
    );
}

/// The bin operators (denoise/gate/compress) off + hum/click off is a pure
/// pass-through, delayed by exactly the reported latency.
#[test]
fn all_operators_off_reconstructs_input() {
    let (_controller, processor, _telemetry) = spectral_engine(SR, FFT, BLOCK, 1);
    let lat = processor.latency_samples();
    drop(processor);

    let input = sine(220.0, 0.4, FFT * 24);
    let out = run_mono(&input, &SpectralParams::default());

    let mut max_err = 0.0f32;
    for n in (FFT * 3)..(input.len() - FFT) {
        max_err = max_err.max((out[n] - input[n - lat]).abs());
    }
    assert!(
        max_err < 1e-2,
        "passthrough error {max_err} (latency {lat})"
    );
}

/// Output gain scales the wet signal predictably (−6 dB ≈ half amplitude).
#[test]
fn output_gain_scales_signal() {
    let input = sine(440.0, 0.5, FFT * 16);

    let unity = run_mono(&input, &SpectralParams::default());
    let attenuated = run_mono(
        &input,
        &SpectralParams {
            output_gain_db: -6.0,
            ..SpectralParams::default()
        },
    );

    let unity_rms = rms(&unity[FFT * 4..]);
    let atten_rms = rms(&attenuated[FFT * 4..]);
    let ratio = atten_rms / unity_rms;
    // −6 dB → 10^(−6/20) ≈ 0.501.
    assert!(
        (ratio - 0.501).abs() < 0.03,
        "−6 dB output gain gave ratio {ratio}"
    );
}

/// Every operator enabled at once renders finite, non-exploding stereo output.
#[test]
fn all_operators_on_is_stable_stereo() {
    let (_controller, mut processor, _telemetry) = spectral_engine(SR, FFT, BLOCK, 2);

    let params = clamp_params(SpectralParams {
        denoise: rt_spectral::spectral_core::DenoiseParams {
            enabled: true,
            reduction_db: 18.0,
            sensitivity_db: 3.0,
        },
        gate: GateParams {
            enabled: true,
            threshold_db: -50.0,
            range_db: 24.0,
            attack_ms: 5.0,
            release_ms: 100.0,
        },
        compress: rt_spectral::spectral_core::CompressParams {
            enabled: true,
            threshold_db: -24.0,
            ratio: 3.0,
            knee_db: 6.0,
            makeup_db: 3.0,
        },
        hum: HumParams {
            enabled: true,
            fundamental_hz: 50.0,
            harmonics: 3,
            q: 30.0,
        },
        click: rt_spectral::spectral_core::ClickParams {
            enabled: true,
            sensitivity: 5.0,
            max_len: 8,
        },
        mix: 0.8,
        output_gain_db: 2.0,
        ..SpectralParams::default()
    });

    for block_index in 0..64 {
        let make = |k: usize| {
            let t = (block_index * BLOCK + k) as f32 / SR as f32;
            0.3 * (TAU * 330.0 * t).sin() + 0.05 * (TAU * 50.0 * t).sin()
        };
        let mut left: Vec<f32> = (0..BLOCK).map(make).collect();
        let mut right: Vec<f32> = (0..BLOCK).map(make).collect();
        {
            let mut channels: [&mut [f32]; 2] = [&mut left, &mut right];
            processor.process(&mut channels, &params);
        }
        assert!(
            left.iter().chain(right.iter()).all(|s| s.is_finite()),
            "non-finite output at block {block_index}"
        );
        assert!(
            left.iter().chain(right.iter()).all(|s| s.abs() < 8.0),
            "output exploded at block {block_index}"
        );
    }
}

/// The full Learn round trip across the audio→control→audio hand-offs: capture
/// noise, learn off-thread, then denoise pulls a fresh noise floor down.
#[test]
fn learn_round_trip_denoises() {
    let (mut controller, mut processor, _telemetry) = spectral_engine(SR, FFT, BLOCK, 1);

    let mut state = 0xC0FF_EE01u32;
    let mut noise = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        0.1 * ((state as f32 / u32::MAX as f32) * 2.0 - 1.0)
    };

    processor.begin_learn(FFT * 16);
    let mut learned = false;
    for _ in 0..300 {
        let mut block: Vec<f32> = (0..BLOCK).map(|_| noise()).collect();
        {
            let mut channels: [&mut [f32]; 1] = [&mut block];
            processor.process(&mut channels, &SpectralParams::default());
        }
        if processor.take_pending_learn() {
            learned = true;
            break;
        }
    }
    assert!(learned && controller.learn());

    let params = SpectralParams {
        denoise: rt_spectral::spectral_core::DenoiseParams {
            enabled: true,
            reduction_db: 24.0,
            sensitivity_db: 6.0,
        },
        transient: rt_spectral::spectral_core::TransientParams {
            enabled: false,
            ..Default::default()
        },
        ..SpectralParams::default()
    };
    let input: Vec<f32> = (0..FFT * 32).map(|_| noise()).collect();
    let out = run_mono_with(&mut processor, &input, &params);

    let in_rms = rms(&input[FFT * 4..]);
    let out_rms = rms(&out[FFT * 4..]);
    assert!(
        out_rms < in_rms * 0.6,
        "learned denoise ineffective: in {in_rms} out {out_rms}"
    );
}

/// Run a signal through an existing processor (keeps the learned profile).
fn run_mono_with(
    processor: &mut rt_spectral::SpectralProcessor,
    signal: &[f32],
    params: &SpectralParams,
) -> Vec<f32> {
    let mut out = vec![0.0f32; signal.len()];
    for (chunk_in, chunk_out) in signal.chunks(BLOCK).zip(out.chunks_mut(BLOCK)) {
        let mut block = chunk_in.to_vec();
        {
            let mut channels: [&mut [f32]; 1] = [&mut block];
            processor.process(&mut channels, params);
        }
        chunk_out.copy_from_slice(&block);
    }
    out
}
