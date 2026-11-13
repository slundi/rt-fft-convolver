//! End-to-end integration tests for `rt-spectral-core` through its public API:
//! learn a profile off-RT, hot-swap it to the engine, and process audio with
//! the full operator chain.

use rt_spectral_core::{
    ClickParams, CompressParams, DenoiseParams, GateParams, HumParams, NoiseProfile,
    SpectralEngine, SpectralParams, TransientParams, learn_profile, profile_channel,
};
use std::f32::consts::PI;

const SR: u32 = 48_000;

/// Deterministic pseudo-noise in `[-amp, amp]` (xorshift; no `rand` dep).
fn noise_source(seed: u32) -> impl FnMut(f32) -> f32 {
    let mut state = seed;
    move |amp| {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state as f32 / u32::MAX as f32 * 2.0 - 1.0) * amp
    }
}

fn sine(freq: f32, n: usize, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|i| amp * (2.0 * PI * freq * i as f32 / SR as f32).sin())
        .collect()
}

fn rms(buf: &[f32]) -> f32 {
    (buf.iter().map(|x| x * x).sum::<f32>() / buf.len() as f32).sqrt()
}

fn run(eng: &mut SpectralEngine, input: &[f32], params: &SpectralParams, block: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    for (i, o) in input.chunks(block).zip(output.chunks_mut(block)) {
        eng.process_block(i, o, params);
    }
    output
}

/// A silent profile + all operators disabled reconstructs the input, delayed
/// by exactly the reported latency.
#[test]
fn full_passthrough_is_latency_accurate() {
    let fft = 2048;
    let (_tx, rx) = profile_channel(NoiseProfile::silent(fft / 2 + 1));
    let mut eng = SpectralEngine::new(SR, fft, 256, rx);
    let params = SpectralParams::default();

    let input = sine(220.0, 24_000, 0.5);
    let output = run(&mut eng, &input, &params, 256);

    let lat = eng.latency_samples();
    let mut max_err = 0.0f32;
    for n in (fft * 3)..(input.len() - fft) {
        max_err = max_err.max((output[n] - input[n - lat]).abs());
    }
    assert!(max_err < 1e-2, "passthrough error {max_err}");
}

/// Learn a noise profile off-RT, publish it, and confirm the engine reduces a
/// matching noise floor while a strong tone underneath largely survives.
#[test]
fn learn_then_denoise_reduces_floor_keeps_tone() {
    let fft = 2048;
    let hop = fft / 4;
    let mut ns = noise_source(0xC0FF_EE01);

    // Capture noise-only material and learn a profile off-RT.
    let noise_only: Vec<f32> = (0..fft * 32).map(|_| ns(0.05)).collect();
    let profile = learn_profile(&noise_only, fft, hop);

    // Hot-swap it into a fresh engine.
    let (mut tx, rx) = profile_channel(NoiseProfile::silent(fft / 2 + 1));
    let mut eng = SpectralEngine::new(SR, fft, 256, rx);
    tx.publish(profile);

    let params = SpectralParams {
        denoise: DenoiseParams {
            enabled: true,
            reduction_db: 24.0,
            sensitivity_db: 6.0,
        },
        transient: TransientParams {
            enabled: false,
            ..TransientParams::default()
        },
        ..SpectralParams::default()
    };

    // Tone (well above the noise) + the same-statistics noise.
    let tone = sine(1_000.0, fft * 32, 0.4);
    let input: Vec<f32> = tone.iter().map(|&t| t + ns(0.05)).collect();
    let output = run(&mut eng, &input, &params, 256);

    // Compare settled regions. Output RMS should drop (noise removed) but the
    // 1 kHz tone must remain the dominant energy.
    let seg = fft * 6..;
    assert!(
        rms(&output[seg.clone()]) < rms(&input[seg]),
        "denoise did not reduce overall level"
    );

    // The tone survives: correlation of output with a latency-shifted pure
    // 1 kHz reference is high.
    let lat = eng.latency_samples();
    let refr = sine(1_000.0, input.len(), 0.4);
    let (mut dot, mut re, mut oe) = (0.0f64, 0.0f64, 0.0f64);
    for n in (fft * 6)..(input.len() - fft) {
        let a = refr[n - lat] as f64;
        let b = output[n] as f64;
        dot += a * b;
        re += a * a;
        oe += b * b;
    }
    let corr = dot / (re.sqrt() * oe.sqrt());
    assert!(
        corr > 0.9,
        "tone not preserved through denoise (corr {corr})"
    );
}

/// The full chain (click + denoise + gate + compress + hum + transient) runs
/// without NaNs/inf and honours the dry/wet mix and bypass.
#[test]
fn full_chain_is_stable_and_respects_mix() {
    let fft = 1024;
    let (_tx, rx) = profile_channel(NoiseProfile::silent(fft / 2 + 1));
    let mut eng = SpectralEngine::new(SR, fft, 128, rx);

    let everything = SpectralParams {
        click: ClickParams {
            enabled: true,
            sensitivity: 4.0,
            max_len: 8,
        },
        denoise: DenoiseParams {
            enabled: true,
            reduction_db: 12.0,
            sensitivity_db: 0.0,
        },
        gate: GateParams {
            enabled: true,
            threshold_db: -50.0,
            range_db: 18.0,
            attack_ms: 5.0,
            release_ms: 80.0,
        },
        compress: CompressParams {
            enabled: true,
            threshold_db: -24.0,
            ratio: 3.0,
            knee_db: 6.0,
            makeup_db: 0.0,
        },
        hum: HumParams {
            enabled: true,
            fundamental_hz: 50.0,
            harmonics: 3,
            q: 30.0,
        },
        transient: TransientParams {
            enabled: true,
            sensitivity: 1.5,
        },
        mix: 1.0,
        output_gain_db: 0.0,
        bypass: false,
    };

    let mut ns = noise_source(0x1357_9BDF);
    let input: Vec<f32> = sine(440.0, 16_000, 0.3)
        .iter()
        .map(|&t| t + ns(0.02))
        .collect();

    let wet = run(&mut eng, &input, &everything, 128);
    assert!(
        wet.iter().all(|x| x.is_finite()),
        "full chain produced non-finite output"
    );

    // Bypass path: output equals the latency-delayed dry input exactly.
    eng.reset();
    let bypass = SpectralParams {
        bypass: true,
        ..everything
    };
    let out_bypass = run(&mut eng, &input, &bypass, 128);
    let lat = eng.latency_samples();
    let mut max_err = 0.0f32;
    for n in (fft * 3)..(input.len() - fft) {
        max_err = max_err.max((out_bypass[n] - input[n - lat]).abs());
    }
    assert!(
        max_err < 1e-6,
        "bypass is not a latency-accurate dry pass ({max_err})"
    );
}

/// Hum removal notches a 50 Hz tone out of a mixed signal without gutting a
/// musical 1 kHz tone (time-domain stage, exercised through the engine).
#[test]
fn hum_removal_through_engine() {
    let fft = 1024;
    let (_tx, rx) = profile_channel(NoiseProfile::silent(fft / 2 + 1));
    let mut eng = SpectralEngine::new(SR, fft, 256, rx);

    let params = SpectralParams {
        hum: HumParams {
            enabled: true,
            fundamental_hz: 50.0,
            harmonics: 1,
            q: 40.0,
        },
        ..SpectralParams::default()
    };

    let hum = sine(50.0, 48_000, 0.5);
    let music = sine(1_000.0, 48_000, 0.5);
    let input: Vec<f32> = hum.iter().zip(&music).map(|(&h, &m)| h + m).collect();
    let output = run(&mut eng, &input, &params, 256);

    // After settling, the 50 Hz component should be strongly attenuated; the
    // total energy drops toward that of the 1 kHz tone alone (~0.5 RMS/√2).
    let tail = 30_000..;
    let out_rms = rms(&output[tail.clone()]);
    let music_rms = rms(&music[tail]);
    assert!(
        (out_rms - music_rms).abs() < 0.1,
        "hum not removed cleanly: out {out_rms} vs music {music_rms}"
    );
}
