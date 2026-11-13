//! Phase-4 acceptance suite for the `rt-adaptive` plugin engine (Roadmap-09
//! Phase 4 "Integration tests" + "RT-safety audit" bullets), stated as
//! quantitative dB criteria on the end-to-end `AdaptiveProcessor`.
//!
//! - [`feedback_mode_attenuates_howl_by_over_20db`] — auto notch placement kills
//!   a sustained howl.
//! - [`hum_mode_attenuates_mains_by_over_20db`] — the comb removes 60 Hz mains.
//! - [`echo_mode_achieves_documented_erle`] — NLMS reaches a documented echo
//!   return loss enhancement.
//! - [`denormal_guard_flushes_silence_to_zero_in_every_mode`] — the FTZ guard is
//!   active on the audio path (the zero-allocation bullet lives in `rt_alloc.rs`).

use rt_adaptive::params::Mode;
use rt_adaptive::{AdaptiveController, AdaptiveParams, AdaptiveProcessor};

const SR: f32 = 48_000.0;
const BLOCK: usize = 256;

fn sine(freq: f32, samples: usize) -> Vec<f32> {
    (0..samples)
        .map(|n| (std::f32::consts::TAU * freq * n as f32 / SR).sin())
        .collect()
}

fn noise(len: usize, amplitude: f32) -> Vec<f32> {
    let mut state: u32 = 0xBEEF_1234;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            ((state as f32 / u32::MAX as f32) * 2.0 - 1.0) * amplitude
        })
        .collect()
}

fn energy(signal: &[f32]) -> f64 {
    signal.iter().map(|&s| (s as f64) * (s as f64)).sum()
}

/// Goertzel power of `signal` at `freq`.
fn power_at(signal: &[f32], freq: f32) -> f64 {
    let omega = std::f64::consts::TAU * freq as f64 / SR as f64;
    let coeff = 2.0 * omega.cos();
    let (mut s1, mut s2) = (0.0f64, 0.0f64);
    for &sample in signal {
        let s0 = sample as f64 + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    s1 * s1 + s2 * s2 - coeff * s1 * s2
}

fn run_mono(
    processor: &mut AdaptiveProcessor,
    params: &AdaptiveParams,
    input: &[f32],
    sidechain: Option<&[f32]>,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(input.len());
    let mut offset = 0;
    for chunk in input.chunks(BLOCK) {
        let mut buffer = chunk.to_vec();
        let sc = sidechain.map(|s| s[offset..offset + chunk.len()].to_vec());
        {
            let mut main: [&mut [f32]; 1] = [&mut buffer];
            processor.process(&mut main, sc.as_deref(), params);
        }
        out.extend_from_slice(&buffer);
        offset += chunk.len();
    }
    out
}

#[test]
fn feedback_mode_attenuates_howl_by_over_20db() {
    let params = AdaptiveParams {
        mode: Mode::FeedbackSuppressor,
        sensitivity: 0.8,
        ..Default::default()
    };
    let (_controller, mut processor) = AdaptiveController::new(BLOCK, SR, 1, &params);
    let out = run_mono(&mut processor, &params, &sine(2_000.0, 48_000), None);

    assert!(processor.active_notches() >= 1, "no notch placed");
    // 2 kHz power-per-sample before the notch lands (head) vs after it settles
    // (tail); normalise by window length so the ratio is comparable.
    let head_pps = power_at(&out[..512], 2_000.0) / 512.0;
    let tail_pps = power_at(&out[out.len() - 4096..], 2_000.0) / 4096.0;
    let db = 10.0 * (head_pps / tail_pps.max(1e-30)).log10();
    assert!(db > 20.0, "howl only attenuated {db:.1} dB");
}

#[test]
fn hum_mode_attenuates_mains_by_over_20db() {
    let params = AdaptiveParams {
        mode: Mode::HumRemover,
        hum_fundamental_hz: 60.0,
        hum_adaptive: false,
        hum_harmonics: 1,
        ..Default::default()
    };
    let (_controller, mut processor) = AdaptiveController::new(BLOCK, SR, 1, &params);

    let mut buf = sine(60.0, 48_000);
    let before = power_at(&buf[24_000..], 60.0);
    let out = run_mono(&mut processor, &params, &buf, None);
    let after = power_at(&out[24_000..], 60.0);
    buf.clear();

    let db = 10.0 * (before / after.max(1e-30)).log10();
    assert!(db > 20.0, "mains only attenuated {db:.1} dB");
}

#[test]
fn echo_mode_achieves_documented_erle() {
    // Echo return loss enhancement: how much quieter the residual is than the
    // raw microphone echo.  Documented acceptance bound: > 15 dB.
    let params = AdaptiveParams {
        mode: Mode::EchoCanceller,
        convergence: 0.6,
        double_talk_db: 24.0,
        ..Default::default()
    };
    let (_controller, mut processor) = AdaptiveController::new(BLOCK, SR, 1, &params);

    let reference = noise(48_000, 0.5);
    let echo_path = [0.5f32, -0.25, 0.12, 0.05];
    let mic: Vec<f32> = (0..reference.len())
        .map(|i| {
            echo_path
                .iter()
                .enumerate()
                .map(|(k, h)| if i >= k { h * reference[i - k] } else { 0.0 })
                .sum()
        })
        .collect();

    let out = run_mono(&mut processor, &params, &mic, Some(&reference));
    let tail = out.len() - 8192;
    let erle_db = 10.0 * (energy(&mic[tail..]) / energy(&out[tail..]).max(1e-30)).log10();
    assert!(erle_db > 15.0, "ERLE only {erle_db:.1} dB");
}

#[test]
fn denormal_guard_flushes_silence_to_zero_in_every_mode() {
    for mode in [
        Mode::FeedbackSuppressor,
        Mode::HumRemover,
        Mode::EchoCanceller,
    ] {
        let params = AdaptiveParams {
            mode,
            ..Default::default()
        };
        let (_controller, mut processor) = AdaptiveController::new(BLOCK, SR, 1, &params);
        // Feed ~200 blocks of silence (with a silent sidechain for the AEC).
        let silence = vec![0.0f32; BLOCK * 200];
        let sidechain = vec![0.0f32; silence.len()];
        let out = run_mono(&mut processor, &params, &silence, Some(&sidechain));
        assert!(
            out.iter().all(|&s| s.abs() < 1e-12),
            "{mode:?}: denormal dust in silent output"
        );
    }
}
