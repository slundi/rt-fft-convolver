//! Integration tests for the `rt-vox` plugin engine (Roadmap-08 Phase 2/4).
//!
//! These drive the public [`VoxController`] / [`VoxProcessor`] end-to-end:
//! MIDI-driven harmony, sidechain vocoding, formant-preserving pitch shift, and
//! the mode / band-count switching the plugin exposes.  Frequency assertions use
//! a Goertzel power probe so no external reference vectors are needed.

use rt_vox::params::{Bands, Mode, VoiceParams};
use rt_vox::{VoxController, VoxParams, VoxProcessor};
use rt_vox_core::SnapMode;

const SAMPLE_RATE: f32 = 48_000.0;
const BLOCK: usize = 256;

/// A unit-amplitude sine at `freq` Hz.
fn sine(freq: f32, samples: usize) -> Vec<f32> {
    (0..samples)
        .map(|n| (std::f32::consts::TAU * freq * n as f32 / SAMPLE_RATE).sin())
        .collect()
}

/// A naïve sawtooth at `freq` Hz.
fn sawtooth(freq: f32, samples: usize) -> Vec<f32> {
    (0..samples)
        .map(|n| {
            let phase = freq * n as f32 / SAMPLE_RATE;
            2.0 * (phase - (phase + 0.5).floor())
        })
        .collect()
}

/// Goertzel power of `signal` at `freq` — the energy in one narrow band.
fn power_at(signal: &[f32], freq: f32) -> f64 {
    let omega = std::f64::consts::TAU * freq as f64 / SAMPLE_RATE as f64;
    let coeff = 2.0 * omega.cos();
    let (mut s1, mut s2) = (0.0f64, 0.0f64);
    for &sample in signal {
        let s0 = sample as f64 + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    s1 * s1 + s2 * s2 - coeff * s1 * s2
}

fn rms(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    (signal.iter().map(|&s| s * s).sum::<f32>() / signal.len() as f32).sqrt()
}

/// Render a mono signal through the processor, block by block.
fn run_mono(processor: &mut VoxProcessor, params: &VoxParams, input: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(input.len());
    for chunk in input.chunks(BLOCK) {
        let mut buffer = chunk.to_vec();
        {
            let mut main: [&mut [f32]; 1] = [&mut buffer];
            processor.process(&mut main, None, params);
        }
        out.extend_from_slice(&buffer);
    }
    out
}

/// The steady-state tail of a signal (past the engine warm-up).
fn tail(signal: &[f32]) -> &[f32] {
    &signal[signal.len() / 2..]
}

#[test]
fn harmoniser_midi_chord_tracks_held_notes() {
    // Input A3 (220 Hz).  Holding A4 (MIDI 69) and E5 (MIDI 76) should place
    // energy at 440 Hz (+12) and ~659 Hz (+19), above the dry input band.
    let params = VoxParams {
        mode: Mode::Harmoniser,
        snap_mode: SnapMode::Midi,
        mix: 1.0,
        dry_delay_compensation: false,
        ..Default::default()
    };
    let (_controller, mut processor) = VoxController::new(BLOCK, SAMPLE_RATE, 1, &params);
    processor.note_on(69);
    processor.note_on(76);

    let out = run_mono(&mut processor, &params, &sine(220.0, 32_000));
    let steady = tail(&out);
    let p440 = power_at(steady, 440.0);
    let p659 = power_at(steady, 659.25);
    let p220 = power_at(steady, 220.0);
    assert!(p440 > p220 * 4.0, "no +12 voice: {p440} vs {p220}");
    assert!(p659 > p220, "no +19 voice: {p659} vs {p220}");
}

#[test]
fn note_off_removes_a_harmony_voice() {
    let params = VoxParams {
        mode: Mode::Harmoniser,
        snap_mode: SnapMode::Midi,
        mix: 1.0,
        dry_delay_compensation: false,
        ..Default::default()
    };
    let (_controller, mut processor) = VoxController::new(BLOCK, SAMPLE_RATE, 1, &params);
    processor.note_on(69);
    processor.note_off(69);
    let out = run_mono(&mut processor, &params, &sine(220.0, 16_000));
    // With the only note released, the wet chord is silent.
    assert!(rms(tail(&out)) < 1e-4, "released note still sounding");
}

#[test]
fn vocoder_imprints_modulator_on_carrier() {
    // A steady carrier gated by an on/off modulator: output tracks the
    // modulator's amplitude envelope.
    let params = VoxParams {
        mode: Mode::Vocoder,
        attack_ms: 2.0,
        release_ms: 20.0,
        mix: 1.0,
        dry_delay_compensation: false,
        ..Default::default()
    };
    let (_controller, mut processor) = VoxController::new(BLOCK, SAMPLE_RATE, 1, &params);

    let carrier = sawtooth(110.0, 24_000);
    // Modulator: loud for the first half, silent for the second.
    let modulator: Vec<f32> = sawtooth(220.0, 24_000)
        .iter()
        .enumerate()
        .map(|(n, &s)| if n < 12_000 { s } else { 0.0 })
        .collect();

    let mut out = Vec::new();
    for (m, c) in modulator.chunks(BLOCK).zip(carrier.chunks(BLOCK)) {
        let mut buffer = m.to_vec();
        let carrier_chunk = c.to_vec();
        {
            let mut main: [&mut [f32]; 1] = [&mut buffer];
            processor.process(&mut main, Some(&carrier_chunk), &params);
        }
        out.extend_from_slice(&buffer);
    }
    // Loud half (skip warm-up) should have far more energy than the silent
    // half (measured past the envelope release tail).
    let loud = rms(&out[2048..11_000]);
    let quiet = rms(&out[18_000..]);
    assert!(
        loud > quiet * 4.0,
        "envelope not tracked: loud {loud} quiet {quiet}"
    );
}

#[test]
fn formant_shift_preserves_pitch() {
    let params = VoxParams {
        mode: Mode::FormantShifter,
        formant_shift: 7.0,
        formant_pitch_shift: 0.0,
        mix: 1.0,
        dry_delay_compensation: false,
        ..Default::default()
    };
    let (_controller, mut processor) = VoxController::new(BLOCK, SAMPLE_RATE, 1, &params);
    let out = run_mono(&mut processor, &params, &sine(220.0, 24_000));
    let steady = tail(&out);
    // The fundamental must dominate its neighbours despite the formant move.
    let fundamental = power_at(steady, 220.0);
    let neighbour = power_at(steady, 330.0);
    assert!(
        fundamental > neighbour * 4.0,
        "pitch not preserved: 220 {fundamental} vs 330 {neighbour}"
    );
}

#[test]
fn formant_mode_pitch_shift_moves_pitch() {
    let params = VoxParams {
        mode: Mode::FormantShifter,
        formant_shift: 0.0,
        formant_pitch_shift: 12.0, // up an octave
        mix: 1.0,
        dry_delay_compensation: false,
        ..Default::default()
    };
    let (_controller, mut processor) = VoxController::new(BLOCK, SAMPLE_RATE, 1, &params);
    let out = run_mono(&mut processor, &params, &sine(220.0, 24_000));
    let steady = tail(&out);
    let octave = power_at(steady, 440.0);
    let root = power_at(steady, 220.0);
    assert!(
        octave > root,
        "pitch shift did not raise the octave: {octave} vs {root}"
    );
}

#[test]
fn stereo_harmoniser_pans_voices() {
    // Two voices panned hard opposite: the two output channels must differ.
    let mut params = VoxParams {
        mode: Mode::Harmoniser,
        snap_mode: SnapMode::Free,
        mix: 1.0,
        dry_delay_compensation: false,
        ..Default::default()
    };
    params.voices = [VoiceParams::default(); rt_vox::params::NUM_VOICES];
    params.voices[0] = VoiceParams {
        interval: 12.0,
        pan: -1.0,
        enabled: true,
        ..VoiceParams::default()
    };
    params.voices[1] = VoiceParams {
        interval: 7.0,
        pan: 1.0,
        enabled: true,
        ..VoiceParams::default()
    };
    let (_controller, mut processor) = VoxController::new(BLOCK, SAMPLE_RATE, 2, &params);

    let input = sine(220.0, 24_000);
    let mut left = Vec::new();
    let mut right = Vec::new();
    for chunk in input.chunks(BLOCK) {
        let mut l = chunk.to_vec();
        let mut r = chunk.to_vec();
        {
            let mut main: [&mut [f32]; 2] = [&mut l, &mut r];
            processor.process(&mut main, None, &params);
        }
        left.extend_from_slice(&l);
        right.extend_from_slice(&r);
    }
    // Left carries the +12 voice (440), right the +7 voice (~330).
    assert!(power_at(tail(&left), 440.0) > power_at(tail(&right), 440.0));
    assert!(power_at(tail(&right), 329.63) > power_at(tail(&left), 329.63));
}

#[test]
fn band_count_change_keeps_rendering() {
    let params = VoxParams {
        mode: Mode::Vocoder,
        ..Default::default()
    };
    let (mut controller, mut processor) = VoxController::new(BLOCK, SAMPLE_RATE, 1, &params);
    assert!(!controller.apply_structural(&params));

    let wide = VoxParams {
        bands: Bands::B32,
        ..params
    };
    assert!(controller.apply_structural(&wide));
    let out = run_mono(&mut processor, &wide, &sawtooth(220.0, 8192));
    assert!(out.iter().all(|s| s.is_finite()));
}
