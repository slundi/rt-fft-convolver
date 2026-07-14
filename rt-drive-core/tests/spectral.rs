//! Phase-4 spectral tests: resampler round-trip fidelity and the aliasing
//! floor of the oversampled shapers.
//!
//! Both use an FFT and choose all tone frequencies on **exact FFT bins**
//! (integer cycles per window) so there is no spectral leakage and no window
//! is needed.  To reach the streaming steady state, each signal is pushed
//! through once to prime the delay lines and analysed on the second pass
//! (the signals are periodic over the window, so the second pass is periodic).

use realfft::RealFftPlanner;
use rt_drive_core::{DriveStage, Oversampler, PedalModel, Shaper, ToneConfig};

const FFT_SIZE: usize = 4096;
const SAMPLE_RATE: f32 = 48_000.0;

/// Magnitude spectrum of `signal` (length [`FFT_SIZE`]).
fn magnitude_spectrum(signal: &[f32]) -> Vec<f32> {
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let mut input = fft.make_input_vec();
    let mut spectrum = fft.make_output_vec();
    input.copy_from_slice(&signal[..FFT_SIZE]);
    fft.process(&mut input, &mut spectrum).unwrap();
    spectrum.iter().map(|bin| bin.norm()).collect()
}

/// A cosine of unit amplitude at exact bin `k`.
fn tone_at_bin(bin: usize, amplitude: f32) -> Vec<f32> {
    (0..FFT_SIZE)
        .map(|n| {
            let phase = std::f32::consts::TAU * bin as f32 * n as f32 / FFT_SIZE as f32;
            amplitude * phase.sin()
        })
        .collect()
}

// ─── Round-trip fidelity ────────────────────────────────────────────────────

/// `downsample(upsample(x)) ≈ x` — the resampler must be magnitude-transparent
/// for in-band signals within −80 dB at the highest oversampling factor.
///
/// We compare the magnitudes at several deep-passband tone bins (phase, i.e.
/// the cascade's group delay, is irrelevant to a magnitude comparison, so this
/// is delay-invariant).
#[test]
fn round_trip_magnitude_transparent_within_80db() {
    // Deep-passband bins (≈ 293 Hz … 8.2 kHz, all well below the fs/4 cutoff).
    let tone_bins = [25usize, 100, 250, 450, 700];
    let mut signal = vec![0.0f32; FFT_SIZE];
    for &bin in &tone_bins {
        for (n, sample) in signal.iter_mut().enumerate() {
            let phase = std::f32::consts::TAU * bin as f32 * n as f32 / FFT_SIZE as f32;
            *sample += 0.15 * phase.sin();
        }
    }
    let input_spectrum = magnitude_spectrum(&signal);

    for factor in [4usize, 16] {
        let mut oversampler = Oversampler::new(factor, FFT_SIZE);
        let mut high = vec![0.0f32; FFT_SIZE * factor];
        let mut output = vec![0.0f32; FFT_SIZE];
        // Prime, then analyse the periodic steady state.
        for _ in 0..2 {
            oversampler.upsample(&signal, &mut high);
            oversampler.downsample(&high, &mut output);
        }
        let output_spectrum = magnitude_spectrum(&output);

        let mut worst_db = f32::NEG_INFINITY;
        for &bin in &tone_bins {
            let reference = input_spectrum[bin];
            let ratio = output_spectrum[bin] / reference;
            let deviation_db = 20.0 * (ratio - 1.0).abs().max(1e-12).log10();
            worst_db = worst_db.max(deviation_db);
        }
        println!("round-trip factor {factor}: worst passband deviation {worst_db:.1} dB");
        assert!(
            worst_db < -80.0,
            "factor {factor}: passband deviation {worst_db:.1} dB exceeds −80 dB"
        );
    }
}

// ─── Aliasing floor ─────────────────────────────────────────────────────────

/// Measure the aliasing floor of one shaper at one oversampling factor.
///
/// Feeds a full-scale sine at bin `f0_bin`, renders through a shaper-only
/// `DriveStage`, and returns the loudest **non-harmonic** bin in dBFS
/// (relative to a full-scale sine).  Non-harmonic bins are the aliases.
fn aliasing_floor_dbfs(shaper: Shaper, factor: u8, f0_bin: usize) -> f32 {
    let pedal = PedalModel {
        name: "AliasTest".to_string(),
        input_gain_db: 0.0,
        shaper,
        bias: 0.0,
        tone: ToneConfig::Bypass,
        output_gain_db: 0.0,
        default_oversampling: factor,
    };
    let mut stage = DriveStage::from_model(&pedal, SAMPLE_RATE, FFT_SIZE);

    let signal = tone_at_bin(f0_bin, 1.0); // full scale
    let mut output = vec![0.0f32; FFT_SIZE];
    for _ in 0..2 {
        stage.process_block(&signal, &mut output);
    }
    let spectrum = magnitude_spectrum(&output);

    // 0 dBFS reference: a full-scale sine has bin magnitude FFT_SIZE/2.
    let full_scale = FFT_SIZE as f32 / 2.0;
    // Harmonic bins (± a 1-bin guard) and the lowest bins (DC-blocker skirt)
    // are excluded; the rest are aliases.
    let harmonic_guard = 1usize;
    let max_bin = FFT_SIZE / 2;
    let is_harmonic = |bin: usize| {
        (1..)
            .map(|multiple| multiple * f0_bin)
            .take_while(|&harmonic| harmonic <= max_bin + f0_bin)
            .any(|harmonic| bin.abs_diff(harmonic) <= harmonic_guard)
    };

    let mut worst = 0.0f32;
    for (bin, &magnitude) in spectrum.iter().enumerate() {
        if bin <= 4 || is_harmonic(bin) {
            continue;
        }
        worst = worst.max(magnitude);
    }
    20.0 * (worst / full_scale).max(1e-12).log10()
}

/// The full shipped shaper set, with representative parameters.
fn shapers() -> Vec<(&'static str, Shaper)> {
    vec![
        ("tanh", Shaper::Tanh),
        ("reciprocal", Shaper::Reciprocal),
        ("cubic", Shaper::CubicSoft),
        (
            "hardclip",
            Shaper::HardClip {
                threshold: 0.7,
                asymmetry: 0.2,
            },
        ),
        ("diode", Shaper::Diode { drive: 2.0 }),
        ("tube", Shaper::Tube),
        ("tape", Shaper::Tape { bias: 0.2 }),
    ]
}

/// At the highest oversampling factor (16×) the aliasing floor must sit below
/// −80 dBFS for every shaper, even a full-scale hard clip — the roadmap's
/// design target.
#[test]
fn aliasing_floor_below_80db_at_16x() {
    // ≈ 5 kHz on an exact bin (427 · 48000 / 4096 ≈ 5004 Hz).
    let f0_bin = 427;
    for (name, shaper) in shapers() {
        let floor = aliasing_floor_dbfs(shaper, 16, f0_bin);
        println!("aliasing 16× {name}: {floor:.1} dBFS");
        assert!(
            floor < -80.0,
            "{name}: aliasing floor {floor:.1} dBFS exceeds −80 dBFS at 16×"
        );
    }
}

/// Document the 4× and 8× floors (measured, not asserted at −80): a 5 kHz
/// full-scale clip has harmonics straddling the base Nyquist, so the moderate
/// per-octave filter only reaches the full −80 dB floor at higher factors.
/// This test guards against *regressions* with a loose bound and prints the
/// real numbers.
#[test]
fn aliasing_floor_improves_with_oversampling() {
    let f0_bin = 427;
    let shaper = Shaper::HardClip {
        threshold: 0.7,
        asymmetry: 0.2,
    };
    let floor_4x = aliasing_floor_dbfs(shaper, 4, f0_bin);
    let floor_8x = aliasing_floor_dbfs(shaper, 8, f0_bin);
    let floor_16x = aliasing_floor_dbfs(shaper, 16, f0_bin);
    println!("hardclip aliasing: 4×={floor_4x:.1}  8×={floor_8x:.1}  16×={floor_16x:.1} dBFS");
    // More oversampling never makes aliasing worse.
    assert!(floor_8x <= floor_4x + 1.0, "8× worse than 4×");
    assert!(floor_16x <= floor_8x + 1.0, "16× worse than 8×");
    // Even the moderate 4× path must clear a musically-safe floor.
    assert!(
        floor_4x < -55.0,
        "4× floor {floor_4x:.1} dBFS unexpectedly high"
    );
}
