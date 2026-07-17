//! End-to-end integration tests for the `rt-adaptive` plugin engine
//! (Roadmap-09 Phase 4).
//!
//! Exercises the three modes through the public `AdaptiveController` /
//! `AdaptiveProcessor` API, including the feedback loopback simulation the
//! roadmap calls for (a synthetic acoustic path fed back into the input must
//! not be allowed to diverge).

use rt_adaptive::params::Mode;
use rt_adaptive::{AdaptiveController, AdaptiveParams, AdaptiveProcessor};

const SAMPLE_RATE: f32 = 48_000.0;
const BLOCK: usize = 128;

fn sine(freq: f32, samples: usize) -> Vec<f32> {
    use std::f32::consts::TAU;
    (0..samples)
        .map(|n| (TAU * freq * n as f32 / SAMPLE_RATE).sin())
        .collect()
}

fn noise(len: usize, amplitude: f32) -> Vec<f32> {
    let mut state: u32 = 0xC0FF_EE11;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            ((state as f32 / u32::MAX as f32) * 2.0 - 1.0) * amplitude
        })
        .collect()
}

fn rms(buf: &[f32]) -> f32 {
    if buf.is_empty() {
        return 0.0;
    }
    (buf.iter().map(|s| s * s).sum::<f32>() / buf.len() as f32).sqrt()
}

/// A constant-0 dB-peak RBJ band-pass — the resonant "acoustic path" of the
/// loop, so the loop gain exceeds unity only near 2 kHz (a single howl a notch
/// can break), and its phase is ~0 at resonance so the loop actually grows.
struct BandPass {
    b0: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl BandPass {
    fn new(freq: f32, q: f32, sample_rate: f32) -> Self {
        use std::f32::consts::TAU;
        let w0 = TAU * freq / sample_rate;
        let alpha = w0.sin() / (2.0 * q);
        let a0 = 1.0 + alpha;
        Self {
            b0: alpha / a0,
            b2: -alpha / a0,
            a1: -2.0 * w0.cos() / a0,
            a2: (1.0 - alpha) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = -self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

/// Loop block length — exactly 5 periods of 2 kHz at 48 kHz (`120 = 5 × 24`), so
/// the one-block feedback delay adds zero phase at the resonance and the loop
/// grows cleanly instead of dithering in and out of phase.
const LOOP_BLOCK: usize = 120;

/// Run a closed feedback loop for `blocks` blocks: each input block is a 2 kHz
/// drive (plus a faint floor) plus the previous output fed back through the
/// resonant path scaled by `loop_gain`.  Returns the RMS of the final block.
fn run_feedback_loop(
    processor: &mut AdaptiveProcessor,
    params: &AdaptiveParams,
    loop_gain: f32,
    blocks: usize,
) -> f32 {
    use std::f32::consts::TAU;
    let mut path = BandPass::new(2_000.0, 8.0, SAMPLE_RATE);
    let mut prev_out = vec![0.0f32; LOOP_BLOCK];
    let mut last = vec![0.0f32; LOOP_BLOCK];
    for block_index in 0..blocks {
        let floor = noise(LOOP_BLOCK, 0.01);
        let mut buffer: Vec<f32> = (0..LOOP_BLOCK)
            .map(|n| {
                let t = (block_index * LOOP_BLOCK + n) as f32 / SAMPLE_RATE;
                let drive = 0.1 * (TAU * 2_000.0 * t).sin() + floor[n];
                (drive + loop_gain * path.process(prev_out[n])).clamp(-1e6, 1e6)
            })
            .collect();
        {
            let mut main: [&mut [f32]; 1] = [&mut buffer];
            processor.process(&mut main, None, params);
        }
        prev_out.copy_from_slice(&buffer);
        if block_index == blocks - 1 {
            last.copy_from_slice(&buffer);
        }
    }
    rms(&last)
}

#[test]
fn feedback_loop_is_bounded_versus_bypass_divergence() {
    // Same unstable loop (gain > 1) driven twice: bypass lets it run away, the
    // feedback suppressor should keep it far smaller by notching the howl.
    let loop_gain = 1.05;
    let blocks = 600;

    let suppress = AdaptiveParams {
        mode: Mode::FeedbackSuppressor,
        sensitivity: 0.8,
        max_notches: 24,
        ..Default::default()
    };
    let (_c1, mut p1) = AdaptiveController::new(BLOCK, SAMPLE_RATE, 1, &suppress);
    let suppressed_rms = run_feedback_loop(&mut p1, &suppress, loop_gain, blocks);

    let bypass = AdaptiveParams {
        bypass: true,
        ..suppress
    };
    let (_c2, mut p2) = AdaptiveController::new(BLOCK, SAMPLE_RATE, 1, &bypass);
    let bypass_rms = run_feedback_loop(&mut p2, &bypass, loop_gain, blocks);

    assert!(
        p1.active_notches() >= 1,
        "no notch placed to fight feedback"
    );
    assert!(
        suppressed_rms < bypass_rms * 0.5,
        "suppressor did not tame the loop: {suppressed_rms} vs bypass {bypass_rms}"
    );
    assert!(suppressed_rms.is_finite());
}

#[test]
fn hum_mode_keeps_music_removes_hum() {
    let params = AdaptiveParams {
        mode: Mode::HumRemover,
        hum_fundamental_hz: 50.0,
        hum_adaptive: false,
        hum_harmonics: 2,
        ..Default::default()
    };
    let (_controller, mut processor) = AdaptiveController::new(BLOCK, SAMPLE_RATE, 1, &params);

    // 50 Hz hum + a 1 kHz "musical" tone.
    let hum = sine(50.0, 48_000);
    let music = sine(1_000.0, 48_000);
    let input: Vec<f32> = hum.iter().zip(&music).map(|(h, m)| h + m).collect();

    let mut out = Vec::with_capacity(input.len());
    for chunk in input.chunks(BLOCK) {
        let mut buffer = chunk.to_vec();
        {
            let mut main: [&mut [f32]; 1] = [&mut buffer];
            processor.process(&mut main, None, &params);
        }
        out.extend_from_slice(&buffer);
    }

    // The 1 kHz music (~unit RMS) survives; the 50 Hz hum is gone, so the tail
    // RMS sits near the music-only level (1/√2 ≈ 0.707), well below the mixed
    // input's ≈ 1.0.
    let tail = rms(&out[24_000..]);
    assert!(tail > 0.55 && tail < 0.85, "music/hum balance off: {tail}");
}

#[test]
fn echo_mode_cancels_far_end() {
    let params = AdaptiveParams {
        mode: Mode::EchoCanceller,
        convergence: 0.6,
        double_talk_db: 24.0,
        ..Default::default()
    };
    let (_controller, mut processor) = AdaptiveController::new(BLOCK, SAMPLE_RATE, 1, &params);

    let reference = noise(48_000, 0.5);
    let echo_path = [0.4f32, -0.2, 0.1, 0.05];
    let mic: Vec<f32> = (0..reference.len())
        .map(|i| {
            echo_path
                .iter()
                .enumerate()
                .map(|(k, h)| if i >= k { h * reference[i - k] } else { 0.0 })
                .sum()
        })
        .collect();

    let mut out = Vec::with_capacity(mic.len());
    let mut offset = 0;
    for chunk in mic.chunks(BLOCK) {
        let mut buffer = chunk.to_vec();
        let sc = reference[offset..offset + chunk.len()].to_vec();
        {
            let mut main: [&mut [f32]; 1] = [&mut buffer];
            processor.process(&mut main, Some(&sc), &params);
        }
        out.extend_from_slice(&buffer);
        offset += chunk.len();
    }

    let raw = rms(&mic[mic.len() - 8192..]);
    let residual = rms(&out[out.len() - 8192..]);
    assert!(
        residual < raw * 0.15,
        "echo not cancelled: {raw} → {residual}"
    );
}

#[test]
fn stereo_renders_all_modes_finite() {
    for mode in [
        Mode::FeedbackSuppressor,
        Mode::HumRemover,
        Mode::EchoCanceller,
    ] {
        let params = AdaptiveParams {
            mode,
            ..Default::default()
        };
        let (_controller, mut processor) = AdaptiveController::new(BLOCK, SAMPLE_RATE, 2, &params);
        let mut left = sine(440.0, BLOCK);
        let mut right = sine(660.0, BLOCK);
        let sc = noise(BLOCK, 0.3);
        {
            let mut main: [&mut [f32]; 2] = [&mut left, &mut right];
            processor.process(&mut main, Some(&sc), &params);
        }
        assert!(
            left.iter().chain(right.iter()).all(|s| s.is_finite()),
            "{mode:?}: non-finite output"
        );
    }
}
