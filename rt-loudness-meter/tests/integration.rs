//! End-to-end checks for the `rt-loudness-meter` engine: the audio-thread
//! [`MeterProcessor`], the lock-free broadcast, the parameter model, and the
//! off-RT [`MeterLogger`] working together.

use std::f32::consts::TAU;
use std::sync::Arc;
use std::time::Duration;

use rt_loudness_core::MeterBroadcast;
use rt_loudness_meter::{LogFormat, LoudnessParams, MeterLogger, MeterProcessor, Target};

const SR: u32 = 48_000;
const LIGHT: (f64, f64) = (80.0, 0.1);

fn processor(channels: usize, max_block: usize, broadcast: Arc<MeterBroadcast>) -> MeterProcessor {
    MeterProcessor::with_true_peak_design(channels, SR, max_block, broadcast, LIGHT.0, LIGHT.1)
}

fn sine(freq: f32, samples: usize, amp: f32) -> Vec<f32> {
    (0..samples)
        .map(|index| amp * (TAU * freq * index as f32 / SR as f32).sin())
        .collect()
}

/// A dual-mono −20 dBFS 1 kHz sine reads ≈ −20 LUFS integrated (the summed
/// stereo power and the sine crest cancel against the −0.691 offset).
#[test]
fn measures_known_level_through_broadcast() {
    let broadcast = Arc::new(MeterBroadcast::new());
    let mut proc = processor(2, 4_800, broadcast.clone());
    let params = LoudnessParams::default();

    let amplitude = 10.0f32.powf(-20.0 / 20.0);
    let signal = sine(1_000.0, 4_800, amplitude);
    for _ in 0..20 {
        proc.process(&[&signal, &signal], &params);
    }

    let published = broadcast.load();
    assert_eq!(published, proc.values());
    assert!(
        (published.integrated_lufs - -20.0).abs() < 1.0,
        "integrated {} LUFS, expected ~-20",
        published.integrated_lufs
    );
}

/// `Reset` on a rising edge wipes the integrated measurement, then it rebuilds.
#[test]
fn reset_then_remeasure() {
    let broadcast = Arc::new(MeterBroadcast::new());
    let mut proc = processor(2, 4_800, broadcast);
    let signal = sine(1_000.0, 4_800, 0.5);

    let mut params = LoudnessParams::default();
    for _ in 0..15 {
        proc.process(&[&signal, &signal], &params);
    }
    assert!(proc.values().integrated_lufs.is_finite());

    params.reset = true;
    let silence = vec![0.0f32; 4_800];
    proc.process(&[&silence, &silence], &params);
    assert_eq!(proc.values().integrated_lufs, f32::NEG_INFINITY);

    // Release reset and measure again — it recovers.
    params.reset = false;
    for _ in 0..15 {
        proc.process(&[&signal, &signal], &params);
    }
    assert!(proc.values().integrated_lufs.is_finite());
}

/// The chosen `Target` only moves the compliance thresholds; the measurement is
/// identical regardless of target.
#[test]
fn target_does_not_change_measurement() {
    let signal = sine(1_000.0, 4_800, 0.4);

    let measure_with = |target: Target| {
        let broadcast = Arc::new(MeterBroadcast::new());
        let mut proc = processor(2, 4_800, broadcast);
        let params = LoudnessParams {
            target,
            ..Default::default()
        };
        for _ in 0..20 {
            proc.process(&[&signal, &signal], &params);
        }
        proc.values().integrated_lufs
    };

    let r128 = measure_with(Target::EbuR128);
    let streaming = measure_with(Target::Streaming14);
    assert!(
        (r128 - streaming).abs() < 1e-4,
        "target changed the measurement: {r128} vs {streaming}"
    );
}

/// End-to-end logging: run the processor, log the broadcast to a file, and
/// confirm the file carries a header and a real measurement.
#[test]
fn logs_measurements_to_file() {
    use std::fs::File;
    use std::io::Read;

    let broadcast = Arc::new(MeterBroadcast::new());
    let mut proc = processor(2, 4_800, broadcast.clone());
    let params = LoudnessParams::default();
    let signal = sine(1_000.0, 4_800, 0.5);
    for _ in 0..15 {
        proc.process(&[&signal, &signal], &params);
    }

    let path = std::env::temp_dir().join(format!("rt-loudness-int-{}.csv", std::process::id()));
    let logger = MeterLogger::start(
        &path,
        LogFormat::Csv,
        broadcast.clone(),
        Duration::from_millis(5),
    )
    .expect("start logger");
    std::thread::sleep(Duration::from_millis(30));
    logger.stop();

    let mut contents = String::new();
    File::open(&path)
        .unwrap()
        .read_to_string(&mut contents)
        .unwrap();
    let _ = std::fs::remove_file(&path);

    assert!(contents.starts_with("elapsed_s,"), "missing CSV header");
    assert!(contents.lines().skip(1).count() >= 1, "no data rows logged");
}
