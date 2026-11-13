//! Phase-4 acceptance tests for the physical-modelling core (see
//! `docs/Roadmap-06-physical-modeling.md`, Phase 4 "Unit tests").
//!
//! These validate the measurable guarantees **end-to-end through the public
//! [`PhysicalVoices`] voice manager** — the same path the plugin drives — rather
//! than the individual DSP primitives (which are covered by the in-module unit
//! tests).  Driving the manager exercises the whole voiced chain: excitation
//! generation, per-note retune, the anti-click fade-in and voice reuse.
//!
//! - **Karplus-Strong pitch** — a `note_on` sounds within ±5 cents of the
//!   requested MIDI frequency across the guitar range.
//! - **Modal single mode** — a one-mode voice is a decaying sinusoid at the
//!   note's frequency (±5 cents) that loses energy over time.
//! - **Voice reuse / no state leak** — retriggering a voice that is still
//!   ringing yields bit-for-bit the same signal as a fresh voice (the modal
//!   kernel is deterministic — no excitation RNG — so equality is exact).
//! - **Denormal flush** — a decaying tail reaches *exact* zero, so the string
//!   loop never falls into the subnormal slow path (the RT-safety audit's
//!   denormal half; the allocation half lives in `rt-physical`'s `rt_safety`).
//!
//! The tube-stage DC-blocker test in the roadmap is intentionally absent: the
//! tube family is out of the Phase-0 v1 scope.

use rt_physical_core::{ModePreset, Model, PhysicalParams, PhysicalVoices, midi_to_freq};

const SR: f32 = 48_000.0;

/// Estimate the fundamental of a harmonically-rich signal by autocorrelation,
/// searching only near `target` to avoid octave errors, then refining the peak
/// lag with parabolic interpolation for sub-sample (few-cent) resolution.
/// Zero-crossing counting is unusable on a bright pluck — overtones add
/// spurious crossings.
fn fundamental_hz(signal: &[f32], target: f32) -> f32 {
    let min_lag = (SR / (target * 1.5)).floor().max(2.0) as usize;
    let max_lag = (SR / (target * 0.75)).ceil() as usize;
    let correlate = |lag: usize| -> f32 {
        signal
            .iter()
            .zip(&signal[lag..])
            .map(|(first, second)| first * second)
            .sum::<f32>()
    };
    let mut best_lag = min_lag;
    let mut best_corr = f32::MIN;
    for lag in min_lag..=max_lag {
        let corr = correlate(lag);
        if corr > best_corr {
            best_corr = corr;
            best_lag = lag;
        }
    }
    let (left, centre, right) = (
        correlate(best_lag - 1),
        correlate(best_lag),
        correlate(best_lag + 1),
    );
    let denom = left - 2.0 * centre + right;
    let delta = if denom.abs() > 1e-12 {
        0.5 * (left - right) / denom
    } else {
        0.0
    };
    SR / (best_lag as f32 + delta)
}

/// Cents between a measured and target frequency.
fn cents(measured: f32, target: f32) -> f32 {
    1200.0 * (measured / target).log2()
}

/// A plucked-string manager tuned to concert pitch.
fn plucked_manager() -> PhysicalVoices {
    PhysicalVoices::new(
        SR,
        PhysicalParams {
            model: Model::Plucked,
            t60: 3.0,
            damping: 0.3,
            ..PhysicalParams::default()
        },
    )
}

/// Render `samples` from `voices` (mono voice sum).
fn render(voices: &mut PhysicalVoices, samples: usize) -> Vec<f32> {
    (0..samples).map(|_| voices.process()).collect()
}

/// A `note_on` sounds within ±5 cents of the requested pitch across the useful
/// guitar range (A2 … A5), measured through the full voiced path.
#[test]
fn plucked_note_on_pitch_within_five_cents() {
    // A2, E3, A3, A4, A5 in MIDI.
    for &note in &[45u8, 52, 57, 69, 81] {
        let target = midi_to_freq(note);
        let mut voices = plucked_manager();
        voices.note_on(note, 1.0);
        let signal = render(&mut voices, 24_000);
        // Skip the fade-in and noisy attack; measure the settled portion.
        let measured = fundamental_hz(&signal[4_000..], target);
        let off = cents(measured, target);
        assert!(
            off.abs() < 5.0,
            "note {note} ({target:.1} Hz) off by {off:.2} cents (measured {measured:.1} Hz)"
        );
    }
}

/// A single-mode modal voice is a decaying sinusoid at the note frequency.
#[test]
fn modal_single_mode_is_a_decaying_sinusoid_at_pitch() {
    let note = 69u8; // A4 = 440 Hz
    let target = midi_to_freq(note);
    let mut voices = PhysicalVoices::new(
        SR,
        PhysicalParams {
            model: Model::Modal,
            num_modes: 1,
            t60: 2.0,
            preset: ModePreset::Metal,
            decay_slope: 1.0,
            ..PhysicalParams::default()
        },
    );
    voices.note_on(note, 1.0);
    let signal = render(&mut voices, 24_000);

    // Pitch: a pure decaying sinusoid has clean zero crossings.
    let measured = fundamental_hz(&signal[400..], target);
    let off = cents(measured, target);
    assert!(off.abs() < 5.0, "single mode off by {off:.2} cents");

    // Decay: late energy is well below early energy.
    let head: f32 = signal[..2_000].iter().map(|value| value * value).sum();
    let tail: f32 = signal[22_000..].iter().map(|value| value * value).sum();
    assert!(
        tail < head,
        "single mode did not decay: head {head}, tail {tail}"
    );
}

/// Retriggering a voice that is *still ringing* clears its state completely:
/// the resulting signal is bit-for-bit identical to a fresh voice playing the
/// same note.  Uses the modal family so the kernel is deterministic (the
/// plucked excitation is seeded per trigger, so it would legitimately differ).
#[test]
fn reused_voice_has_no_state_leak() {
    let params = PhysicalParams {
        model: Model::Modal,
        num_modes: 5,
        t60: 2.0,
        preset: ModePreset::Wood,
        ..PhysicalParams::default()
    };
    // Polyphony 1 forces both note-ons onto the same physical voice slot, so
    // the second trigger genuinely reuses the first note's resonators.
    let build = || {
        let mut voices = PhysicalVoices::new(SR, params);
        voices.set_polyphony(1);
        voices
    };

    // Reused: play one note, let it ring (do NOT wait for silence), then
    // retrigger a different note on the same slot.
    let mut reused = build();
    reused.note_on(50, 0.8);
    let _ringing = render(&mut reused, 2_000);
    reused.note_on(69, 1.0);
    let after_reuse = render(&mut reused, 8_000);

    // Fresh: the same second note on a pristine manager.
    let mut fresh = build();
    fresh.note_on(69, 1.0);
    let clean = render(&mut fresh, 8_000);

    assert_eq!(
        after_reuse, clean,
        "reused voice leaked state from the previous note"
    );
}

/// A decaying string tail reaches *exact* zero: the loop's flush-to-zero keeps
/// it out of the denormal slow path (RT-safety audit — denormal half).
#[test]
fn decaying_tail_flushes_to_exact_zero() {
    let mut voices = PhysicalVoices::new(
        SR,
        PhysicalParams {
            model: Model::Plucked,
            t60: 0.3, // short decay so the tail collapses quickly
            damping: 0.7,
            ..PhysicalParams::default()
        },
    );
    voices.note_on(57, 1.0); // A3
    // Let it ring briefly, then release to speed the decay.
    render(&mut voices, 2_000);
    voices.note_off(57);

    let mut reached_zero = false;
    for _ in 0..(10 * SR as usize) {
        if voices.process() == 0.0 {
            // Confirm it *stays* at zero, not a single incidental crossing.
            if (0..256).all(|_| voices.process() == 0.0) {
                reached_zero = true;
                break;
            }
        }
    }
    assert!(
        reached_zero,
        "string tail never flushed to exact zero (denormal creep)"
    );
}
