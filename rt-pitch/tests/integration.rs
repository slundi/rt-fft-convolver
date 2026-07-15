//! Integration tests for the `rt-pitch` plugin engine (Roadmap-03 Phase 4).
//!
//! These drive the public `PitchController` / `PitchProcessor` end-to-end and
//! validate against a **manual DFT reference** (a Goertzel frequency probe —
//! no external `librosa` vectors required, per the Phase-4 "reference test
//! vectors from Python (librosa or manual STFT)" item) plus the MIDI
//! Program-Change / Control-Change → preset routing.

use rt_ir_plugin_core::{cc_to_slot, program_to_slot};
use rt_pitch::{Algorithm, PitchController, PitchParams, PitchProcessor, Quality};
use rt_pitch_core::NUM_PRESETS;

const SAMPLE_RATE: f32 = 48_000.0;
const BLOCK: usize = 256;

/// A unit-amplitude sine at `freq` Hz.
fn sine(freq: f32, samples: usize) -> Vec<f32> {
    (0..samples)
        .map(|n| (std::f32::consts::TAU * freq * n as f32 / SAMPLE_RATE).sin())
        .collect()
}

/// Goertzel power of `signal` at `freq` (a manual single-bin DFT).
fn power_at(signal: &[f32], freq: f32) -> f64 {
    let w = std::f64::consts::TAU * freq as f64 / SAMPLE_RATE as f64;
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (n, &x) in signal.iter().enumerate() {
        let phase = w * n as f64;
        re += x as f64 * phase.cos();
        im += x as f64 * phase.sin();
    }
    re * re + im * im
}

/// Dominant frequency of `signal` scanned over `[lo, hi]` Hz at 1 Hz steps.
fn dominant_hz(signal: &[f32], lo: u32, hi: u32) -> f32 {
    let mut best = lo;
    let mut best_power = 0.0;
    for f in lo..=hi {
        let power = power_at(signal, f as f32);
        if power > best_power {
            best_power = power;
            best = f;
        }
    }
    best as f32
}

/// Run a mono sine of `freq` Hz through the engine for `blocks` blocks and
/// return the concatenated output.
fn run_mono(
    processor: &mut PitchProcessor,
    params: &PitchParams,
    freq: f32,
    blocks: usize,
) -> Vec<f32> {
    let input = sine(freq, blocks * BLOCK);
    let mut out = Vec::with_capacity(input.len());
    for chunk in input.chunks(BLOCK) {
        let mut buf = chunk.to_vec();
        {
            let mut channels: [&mut [f32]; 1] = [&mut buf];
            processor.process(&mut channels, params);
        }
        out.extend_from_slice(&buf);
    }
    out
}

/// Build an engine settled on `preset`, pure-wet with no dry-delay so the
/// probe sees only the shifted signal.
fn engine_on(preset: usize) -> (PitchController, PitchProcessor, PitchParams) {
    let params = PitchParams {
        current_preset: preset,
        mix: 1.0,
        dry_delay_compensation: false,
        crossfade_ms: 5.0,
        ..Default::default()
    };
    let (controller, processor) = PitchController::new(BLOCK, SAMPLE_RATE, 1, &params);
    (controller, processor, params)
}

/// The settled tail past the warm-up region.
fn tail(signal: &[f32]) -> &[f32] {
    &signal[signal.len() / 2..]
}

// ─── Frequency accuracy (manual-DFT reference) ──────────────────────────────

#[test]
fn octave_up_lands_an_octave_above() {
    // Default preset 1 = +12 semitones (2×): 220 Hz → 440 Hz.
    let (_c, mut processor, params) = engine_on(1);
    let out = run_mono(&mut processor, &params, 220.0, 80);
    let hz = dominant_hz(tail(&out), 300, 600);
    assert!((hz - 440.0).abs() < 12.0, "octave-up = {hz} Hz");
}

#[test]
fn octave_down_lands_an_octave_below() {
    // Default preset 2 = −12 semitones (0.5×): 440 Hz → 220 Hz.
    let (_c, mut processor, params) = engine_on(2);
    let out = run_mono(&mut processor, &params, 440.0, 80);
    let hz = dominant_hz(tail(&out), 150, 350);
    assert!((hz - 220.0).abs() < 12.0, "octave-down = {hz} Hz");
}

#[test]
fn fifth_lands_a_fifth_above() {
    // Default preset 3 = +7 semitones (2^(7/12) ≈ 1.4983×): 220 → ~329.6 Hz.
    let (_c, mut processor, params) = engine_on(3);
    let out = run_mono(&mut processor, &params, 220.0, 80);
    let expected = 220.0 * 2f32.powf(7.0 / 12.0);
    let hz = dominant_hz(tail(&out), 250, 420);
    assert!(
        (hz - expected).abs() < 12.0,
        "fifth = {hz} Hz (want {expected})"
    );
}

// ─── MIDI PC / CC → preset routing ──────────────────────────────────────────

#[test]
fn program_change_routes_to_preset_pitch() {
    // A Program-Change of 1 maps to slot 1 (octave up); driving the engine at
    // that slot must produce the octave.
    let slot = program_to_slot(1, NUM_PRESETS).expect("program 1 is in range");
    assert_eq!(slot, 1);
    let (_c, mut processor, params) = engine_on(slot);
    let out = run_mono(&mut processor, &params, 220.0, 80);
    let hz = dominant_hz(tail(&out), 300, 600);
    assert!((hz - 440.0).abs() < 12.0, "PC-routed pitch = {hz} Hz");
}

#[test]
fn out_of_range_program_change_is_ignored() {
    // Programs past the bank size map to None (the plugin ignores them).
    assert_eq!(program_to_slot(NUM_PRESETS as u8, NUM_PRESETS), None);
    assert_eq!(program_to_slot(200, NUM_PRESETS), None);
}

#[test]
fn control_change_routes_to_preset_pitch() {
    // CC value 32 maps to slot 2 (32/16 = 2 → octave down).
    let slot = cc_to_slot(32, NUM_PRESETS);
    assert_eq!(slot, 2);
    let (_c, mut processor, params) = engine_on(slot);
    let out = run_mono(&mut processor, &params, 440.0, 80);
    let hz = dominant_hz(tail(&out), 150, 350);
    assert!((hz - 220.0).abs() < 12.0, "CC-routed pitch = {hz} Hz");

    // The CC sweep spans all eight slots and never lands out of range.
    for value in 0u8..=127 {
        assert!(cc_to_slot(value, NUM_PRESETS) < NUM_PRESETS);
    }
    assert_eq!(cc_to_slot(0, NUM_PRESETS), 0);
    assert_eq!(cc_to_slot(127, NUM_PRESETS), NUM_PRESETS - 1);
}

// ─── Click-free crossfade on preset change ──────────────────────────────────

#[test]
fn preset_change_has_no_click() {
    // Start on unison, warm up, then switch to the fifth mid-signal.  An
    // equal-power crossfade between two decorrelated (differently-pitched)
    // voices holds the *summed power* constant, so the wet RMS stays steady
    // through the transition — a click would punch a dropout hole or a spike.
    let mut params = PitchParams {
        current_preset: 0,
        mix: 1.0,
        dry_delay_compensation: false,
        crossfade_ms: 30.0,
        ..Default::default()
    };
    let (_c, mut processor) = PitchController::new(BLOCK, SAMPLE_RATE, 1, &params);

    // Warm up on unison; measure its settled RMS.
    let settled = run_mono(&mut processor, &params, 220.0, 40);
    let unison_rms = rms(tail(&settled));

    // Switch to the fifth and capture the whole transition.
    params.current_preset = 3;
    let transition = run_mono(&mut processor, &params, 220.0, 60);
    let fifth_rms = rms(tail(&transition));

    // No window may dip below half the quieter endpoint (dropout) or spike far
    // above it, and everything stays finite and bounded.
    let floor = 0.5 * unison_rms.min(fifth_rms);
    let ceil = 2.0 * unison_rms.max(fifth_rms);
    let window = 512;
    let mut min_rms = f32::INFINITY;
    let mut max_rms = 0.0f32;
    for chunk in transition.chunks(window).filter(|c| c.len() == window) {
        let r = rms(chunk);
        min_rms = min_rms.min(r);
        max_rms = max_rms.max(r);
    }
    assert!(
        transition.iter().all(|s| s.is_finite() && s.abs() < 4.0),
        "crossfade produced non-finite or clipped samples"
    );
    assert!(
        min_rms > floor,
        "crossfade dropout: min window RMS {min_rms} below floor {floor}"
    );
    assert!(
        max_rms < ceil,
        "crossfade overshoot: max window RMS {max_rms} above ceil {ceil}"
    );
}

/// Root-mean-square level of a block.
fn rms(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    let sum: f64 = signal.iter().map(|&s| (s as f64).powi(2)).sum();
    (sum / signal.len() as f64).sqrt() as f32
}

// ─── Latency warm-up matches the reported value ─────────────────────────────

#[test]
fn output_is_silent_through_the_reported_latency() {
    // The Shifter contract emits silence for the first `latency` samples; that
    // warm-up must line up with the value reported to the host.
    let params = PitchParams {
        current_preset: 1,
        quality: Quality::Standard,
        algorithm: Algorithm::Smooth,
        mix: 1.0,
        dry_delay_compensation: false,
        ..Default::default()
    };
    let (controller, mut processor) = PitchController::new(BLOCK, SAMPLE_RATE, 1, &params);
    let latency = controller.latency();
    assert_eq!(latency, 1024 - 1024 / 4, "Standard/Smooth PV latency");

    let out = run_mono(&mut processor, &params, 220.0, 40);
    assert_eq!(
        processor.reported_latency(),
        latency,
        "processor must report the controller's latency"
    );

    // Everything comfortably inside the warm-up window is silent…
    let warmup = &out[..latency / 2];
    let warmup_peak = warmup.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    assert!(
        warmup_peak < 1e-4,
        "energy leaked into warm-up: {warmup_peak}"
    );

    // …and real signal has arrived well after the reported latency.
    let settled = &out[latency + BLOCK..];
    let settled_peak = settled.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    assert!(
        settled_peak > 0.1,
        "no signal after warm-up: {settled_peak}"
    );
}
