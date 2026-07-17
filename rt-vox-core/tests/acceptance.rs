//! Phase-4 acceptance suite for `rt-vox-core` (Roadmap-08 Phase 4 "Unit tests").
//!
//! Each test maps to a bullet in the roadmap, exercised through the public
//! engine API at 48 kHz:
//!
//! - [`pitch_tracker_locks_440hz_within_2hz`] — tracker accuracy.
//! - [`vocoder_silent_modulator_is_silent`] +
//!   [`vocoder_transfers_the_modulator_envelope`] — the pass-through /
//!   envelope-transfer guarantee.
//! - [`harmoniser_unison_matches_the_dry_pitch`] — a 0-semitone voice reproduces
//!   the input pitch (bit-identity is impossible through a phase vocoder, so we
//!   assert "within noise": same pitch, audible energy).
//! - [`formant_shift_preserves_pitch_within_5_cents`] — formant move leaves the
//!   fundamental within ± 5 cents.
//!
//! Pitch is measured with `rt_pitch_core::PitchTracker` (autocorrelation +
//! parabolic interpolation) for sub-cent precision — a coarse DFT bin cannot
//! resolve 5 cents.

use rt_pitch_core::PitchTracker;
use rt_vox_core::{FormantShifter, Harmoniser, SnapMode, Vocoder, VoicePitchTracker};

const SR: f32 = 48_000.0;
const MAX_BLOCK: usize = 512;

// ─── Signal + measurement helpers ──────────────────────────────────────────────

fn make_sine(freq: f32, samples: usize) -> Vec<f32> {
    (0..samples)
        .map(|n| (std::f32::consts::TAU * freq * n as f32 / SR).sin())
        .collect()
}

fn make_sawtooth(freq: f32, samples: usize) -> Vec<f32> {
    (0..samples)
        .map(|n| {
            let phase = freq * n as f32 / SR;
            2.0 * (phase - (phase + 0.5).floor())
        })
        .collect()
}

fn rms(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    (signal.iter().map(|&s| s * s).sum::<f32>() / signal.len() as f32).sqrt()
}

/// Fundamental (Hz) of a signal's steady-state tail via autocorrelation with
/// parabolic interpolation — precise enough to compare in cents.
fn measure_pitch(signal: &[f32]) -> f32 {
    let frame_len = 4096;
    assert!(signal.len() >= frame_len, "signal too short to measure");
    let frame = &signal[signal.len() - frame_len..];
    let mut tracker = PitchTracker::new(SR, 60.0, 1000.0);
    tracker
        .analyse(frame)
        .expect("tail should be pitched")
        .frequency
}

/// Interval between two frequencies in cents.
fn cents(measured: f32, reference: f32) -> f32 {
    1200.0 * (measured / reference).log2()
}

// ─── 1. Pitch tracker ──────────────────────────────────────────────────────────

#[test]
fn pitch_tracker_locks_440hz_within_2hz() {
    let hop = 512;
    let mut tracker = VoicePitchTracker::new(SR, 60.0, 1000.0, hop, MAX_BLOCK);
    let signal = make_sine(440.0, 16_000);

    let mut estimate = None;
    for block in signal.chunks(MAX_BLOCK) {
        if let Some(latest) = tracker.process_block(block) {
            estimate = Some(latest);
        }
    }
    let estimate = estimate.expect("440 Hz sine must be tracked");
    assert!(
        (estimate.frequency - 440.0).abs() <= 2.0,
        "tracked {} Hz, want 440 ± 2",
        estimate.frequency
    );
    assert!(
        estimate.confidence > 0.9,
        "low confidence {}",
        estimate.confidence
    );
}

// ─── 2. Vocoder pass-through / envelope transfer ────────────────────────────────

fn run_vocoder(voc: &mut Vocoder, modulator: &[f32], carrier: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(modulator.len());
    let mut block = vec![0.0f32; MAX_BLOCK];
    for (m, c) in modulator.chunks(MAX_BLOCK).zip(carrier.chunks(MAX_BLOCK)) {
        let len = m.len();
        voc.process_block(m, c, &mut block[..len]);
        out.extend_from_slice(&block[..len]);
    }
    out
}

#[test]
fn vocoder_silent_modulator_is_silent() {
    // A silent modulator gates the carrier to nothing — the pass-through bound
    // at zero envelope (the roadmap's "silent modulator" case).
    let mut voc = Vocoder::new(1024, 16, SR, 5.0, 50.0, MAX_BLOCK);
    let carrier = make_sawtooth(110.0, 16_000);
    let silence = vec![0.0f32; carrier.len()];
    let out = run_vocoder(&mut voc, &silence, &carrier);
    assert!(
        rms(&out) < 1e-6,
        "carrier leaked through silence: {}",
        rms(&out)
    );
}

#[test]
fn vocoder_transfers_the_modulator_envelope() {
    // A band the modulator excites should pass more of the carrier than a band
    // it leaves quiet — the defining vocoder behaviour.  The sawtooth carrier is
    // low-harmonic-heavy, so a low-band modulator passes more than a high-band.
    let carrier = make_sawtooth(110.0, 16_000);
    let low = make_sine(200.0, 16_000);
    let high = make_sine(4_000.0, 16_000);

    let mut voc_low = Vocoder::new(1024, 24, SR, 2.0, 30.0, MAX_BLOCK);
    let mut voc_high = Vocoder::new(1024, 24, SR, 2.0, 30.0, MAX_BLOCK);
    let out_low = run_vocoder(&mut voc_low, &low, &carrier);
    let out_high = run_vocoder(&mut voc_high, &high, &carrier);

    assert!(
        rms(&out_low[4096..]) > rms(&out_high[4096..]),
        "envelope not transferred: low {} vs high {}",
        rms(&out_low[4096..]),
        rms(&out_high[4096..])
    );
}

// ─── 3. Harmoniser unison ≈ dry ─────────────────────────────────────────────────

#[test]
fn harmoniser_unison_matches_the_dry_pitch() {
    let mut harm = Harmoniser::new(SR, MAX_BLOCK);
    harm.set_snap_mode(SnapMode::Free);
    harm.set_voice_interval(0, 0.0); // one unison voice, enabled

    let input = make_sine(220.0, 24_000);
    let mut mono = Vec::with_capacity(input.len());
    let mut left = vec![0.0f32; MAX_BLOCK];
    let mut right = vec![0.0f32; MAX_BLOCK];
    for chunk in input.chunks(MAX_BLOCK) {
        let len = chunk.len();
        harm.process_block(chunk, &mut left[..len], &mut right[..len]);
        for index in 0..len {
            mono.push(left[index] + right[index]);
        }
    }

    let pitch = measure_pitch(&mono);
    assert!(
        cents(pitch, 220.0).abs() < 15.0,
        "unison drifted to {pitch} Hz ({} cents)",
        cents(pitch, 220.0)
    );
    assert!(rms(&mono[8192..]) > 1e-3, "unison voice inaudible");
}

// ─── 4. Formant shift preserves pitch (± 5 cents) ────────────────────────────────

fn run_formant(shifter: &mut FormantShifter, input: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(input.len());
    let mut block = vec![0.0f32; MAX_BLOCK];
    for chunk in input.chunks(MAX_BLOCK) {
        let len = chunk.len();
        shifter.process_block(chunk, &mut block[..len]);
        out.extend_from_slice(&block[..len]);
    }
    out
}

#[test]
fn formant_shift_preserves_pitch_within_5_cents() {
    // For several shifts and pitches, moving only the formants must leave the
    // fundamental within ± 5 cents of the input.
    for (fundamental, shift) in [(220.0, 7.0), (220.0, -5.0), (150.0, 12.0)] {
        let input = make_sawtooth(fundamental, 24_000);
        let input_pitch = measure_pitch(&input);

        let mut shifter = FormantShifter::new(1024, SR, MAX_BLOCK);
        shifter.set_shift_semitones(shift);
        let output = run_formant(&mut shifter, &input);
        let output_pitch = measure_pitch(&output);

        let drift = cents(output_pitch, input_pitch);
        assert!(
            drift.abs() <= 5.0,
            "shift {shift} st on {fundamental} Hz drifted {drift:.2} cents \
             ({input_pitch} → {output_pitch} Hz)"
        );
    }
}
