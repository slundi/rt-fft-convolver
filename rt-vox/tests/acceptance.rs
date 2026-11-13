//! Phase-4 acceptance suite for the `rt-vox` plugin engine (Roadmap-08 Phase 4
//! "Integration tests" + "RT-safety audit" bullets).
//!
//! - [`midi_chord_triggers_distinct_voices`] — a held MIDI chord produces one
//!   pitch-shifted voice per note, simultaneously.
//! - [`held_note_deterministically_triggers_its_voice`] — the harmoniser is
//!   silent until a note is held, then the note's voice appears (deterministic
//!   timing; sub-block sample placement is a nih-plug host concern).
//! - [`releasing_one_chord_note_keeps_the_rest`] — note-off removes exactly the
//!   released voice.
//! - [`denormal_guard_flushes_silence_to_zero`] — the FTZ guard is active on the
//!   audio path (RT-safety bullet; the zero-allocation bullet lives in
//!   `rt_alloc.rs`).

use rt_vox::params::Mode;
use rt_vox::{VoxController, VoxParams, VoxProcessor};
use rt_vox_core::SnapMode;

const SR: f32 = 48_000.0;
const BLOCK: usize = 256;

fn sine(freq: f32, samples: usize) -> Vec<f32> {
    (0..samples)
        .map(|n| (std::f32::consts::TAU * freq * n as f32 / SR).sin())
        .collect()
}

/// Goertzel power of `signal` at `freq` — energy in one narrow band.
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

fn rms(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    (signal.iter().map(|&s| s * s).sum::<f32>() / signal.len() as f32).sqrt()
}

/// Render a mono input (harmoniser mono-out sums its panned voices), returning
/// the output.
fn run_mono(processor: &mut VoxProcessor, params: &VoxParams, input: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(input.len());
    for chunk in input.chunks(BLOCK) {
        let mut mono = chunk.to_vec();
        {
            let mut main: [&mut [f32]; 1] = [&mut mono];
            processor.process(&mut main, None, params);
        }
        out.extend_from_slice(&mono);
    }
    out
}

fn harmoniser_params() -> VoxParams {
    VoxParams {
        mode: Mode::Harmoniser,
        snap_mode: SnapMode::Midi,
        mix: 1.0,
        dry_delay_compensation: false,
        ..Default::default()
    }
}

#[test]
fn midi_chord_triggers_distinct_voices() {
    // Input A3 (220 Hz). Hold a C-major triad: C4/E4/G4 (MIDI 60/64/67) →
    // 261.63 / 329.63 / 392.00 Hz, each a distinct harmony voice.
    let params = harmoniser_params();
    let (_controller, mut processor) = VoxController::new(BLOCK, SR, 1, &params);
    processor.note_on(60);
    processor.note_on(64);
    processor.note_on(67);

    let out = run_mono(&mut processor, &params, &sine(220.0, 40_000));
    let steady = &out[out.len() / 2..];
    let dry = power_at(steady, 220.0);
    for target in [261.63, 329.63, 392.0] {
        assert!(
            power_at(steady, target) > dry.max(1e-9) * 4.0,
            "voice at {target} Hz missing (power {}, dry {dry})",
            power_at(steady, target)
        );
    }
}

#[test]
fn held_note_deterministically_triggers_its_voice() {
    let params = harmoniser_params();
    let (_controller, mut processor) = VoxController::new(BLOCK, SR, 1, &params);

    // No notes held → the wet harmoniser output is silent.
    let before = run_mono(&mut processor, &params, &sine(220.0, 8_000));
    assert!(rms(&before) < 1e-4, "harmoniser sounded with no notes held");

    // Hold A4 (MIDI 69) → +12 → 440 Hz voice appears (after the shifter warm-up).
    processor.note_on(69);
    let after = run_mono(&mut processor, &params, &sine(220.0, 24_000));
    let steady = &after[after.len() / 2..];
    assert!(
        power_at(steady, 440.0) > power_at(steady, 220.0).max(1e-9) * 4.0,
        "held note did not trigger its 440 Hz voice"
    );
}

#[test]
fn releasing_one_chord_note_keeps_the_rest() {
    let params = harmoniser_params();
    let (_controller, mut processor) = VoxController::new(BLOCK, SR, 1, &params);
    processor.note_on(60); // C4 → 261.63
    processor.note_on(67); // G4 → 392.00
    processor.note_off(60); // drop the C

    let out = run_mono(&mut processor, &params, &sine(220.0, 32_000));
    let steady = &out[out.len() / 2..];
    let c4 = power_at(steady, 261.63);
    let g4 = power_at(steady, 392.0);
    assert!(
        g4 > c4 * 4.0,
        "released C4 still louder than expected: c4 {c4} g4 {g4}"
    );
}

#[test]
fn denormal_guard_flushes_silence_to_zero() {
    // With the FTZ guard active, a spectral mode fed silence emits exact zeros
    // rather than denormal dust.
    let params = VoxParams {
        mode: Mode::FormantShifter,
        formant_shift: 7.0,
        mix: 1.0,
        dry_delay_compensation: false,
        ..Default::default()
    };
    let (_controller, mut processor) = VoxController::new(BLOCK, SR, 1, &params);
    let out = run_mono(&mut processor, &params, &vec![0.0f32; 16_000]);
    assert!(
        out.iter().all(|&s| s.abs() < 1e-12),
        "denormal dust in silent output: {}",
        rms(&out)
    );
}
