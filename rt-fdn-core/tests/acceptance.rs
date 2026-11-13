//! Phase-4 acceptance tests for the FDN engine (see
//! `docs/Roadmap-05-fdn-reverb.md`, Phase 4 "Unit tests").
//!
//! These are the measurable guarantees the engine must meet, kept apart from
//! the behavioural `integration.rs` checks:
//!
//! - **T60 accuracy** — measured reverberation time matches the requested decay
//!   within ±10 % at the *undamped reference* (damping / low-cut neutral).
//!   Heavy damping deliberately shortens the broadband tail, so T60 is specified
//!   at the reference, the standard convention.
//! - **Freeze stability** — a frozen tail holds its RMS to within 0.1 dB over
//!   10 s.
//! - **Modulation reset** — resetting clears the modulated lines without a
//!   click (silent-in → exact-silent-out).
//! - **Denormal flush** — a decaying tail reaches *exact* zero, so the feedback
//!   path never falls into the subnormal slow path.

use rt_fdn_core::{BASE_DELAYS, Fdn, FdnParams, PRESETS};

const SR: f32 = 48_000.0;

/// A near-lossless reference configuration: tone controls wide open (damping
/// just below Nyquist, low-cut near DC), no modulation, full size.  Isolates
/// the feedback-gain decay calibration from filter / interpolation loss.
fn reference_params(decay_s: f32) -> FdnParams<8> {
    FdnParams {
        nominal_delays: BASE_DELAYS,
        decay_s,
        damping_hz: 22_000.0,
        lowcut_hz: 2.0,
        mod_depth: 0.0,
        mod_rate_hz: 0.5,
        diffusion: 0.0,
        size: 1.0,
    }
}

/// Measure T60 from the backward-integrated (Schroeder) energy decay curve,
/// fitting the slope between −15 dB and −45 dB and extrapolating to −60 dB.
fn measure_t60(decay_s: f32) -> f32 {
    let mut fdn = Fdn::<8>::new(SR, reference_params(decay_s));
    let total = (decay_s * 4.5 * SR) as usize;

    let mut energy = Vec::with_capacity(total);
    fdn.process_sample_stereo(1.0, 1.0);
    for _ in 0..total {
        let (left, right) = fdn.process_sample_stereo(0.0, 0.0);
        energy.push((left * left + right * right) as f64);
    }

    // Backward Schroeder integration: edc[i] = total energy remaining from i on.
    let mut edc = vec![0.0f64; energy.len()];
    let mut acc = 0.0;
    for i in (0..energy.len()).rev() {
        acc += energy[i];
        edc[i] = acc;
    }
    let e0 = edc[0].max(1e-30);
    let time_at = |db: f64| -> f64 {
        let target = e0 * 10f64.powf(db / 10.0);
        edc.iter()
            .position(|&e| e <= target)
            .map(|i| i as f64 / SR as f64)
            .expect("EDC never reached target level")
    };
    let (t15, t45) = (time_at(-15.0), time_at(-45.0));
    let slope_db_per_s = (-45.0 - -15.0) / (t45 - t15); // negative
    (60.0 / -slope_db_per_s) as f32
}

/// Windowed RMS of `signal` in `[lo, hi)`.
fn rms(signal: &[f32]) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = signal.iter().map(|&x| (x * x) as f64).sum();
    (sum_sq / signal.len() as f64).sqrt()
}

/// T60 matches the requested decay within ±10 % across the useful range.
#[test]
fn t60_matches_requested_within_10_percent() {
    for &decay_s in &[1.0f32, 2.5, 5.0] {
        let measured = measure_t60(decay_s);
        let error = (measured / decay_s - 1.0).abs();
        assert!(
            error <= 0.10,
            "requested {decay_s}s → measured {measured:.2}s ({:+.0}%), outside ±10%",
            (measured / decay_s - 1.0) * 100.0
        );
    }
}

/// A frozen tail sustains: RMS holds within 0.1 dB over 10 s after engaging.
#[test]
fn freeze_rms_constant_within_0_1_db_over_10s() {
    // A realistic (damped, modulated) space — freeze must be flat regardless.
    let params = FdnParams {
        nominal_delays: BASE_DELAYS,
        decay_s: 3.0,
        damping_hz: 8_000.0,
        lowcut_hz: 100.0,
        mod_depth: 4.0,
        mod_rate_hz: 0.6,
        diffusion: 0.7,
        size: 0.7,
    };
    let mut fdn = Fdn::<8>::new(SR, params);

    // Build a tail, then freeze.
    fdn.process_sample_stereo(1.0, 1.0);
    for _ in 0..(SR as usize / 2) {
        fdn.process_sample_stereo(0.0, 0.0);
    }
    fdn.set_freeze(true);

    // Ten one-second RMS windows.
    let mut window = Vec::with_capacity(SR as usize);
    let mut rms_per_second = Vec::with_capacity(10);
    for _ in 0..10 {
        window.clear();
        for _ in 0..SR as usize {
            let (left, _right) = fdn.process_sample_stereo(0.0, 0.0);
            assert!(left.is_finite(), "frozen tail diverged");
            window.push(left);
        }
        rms_per_second.push(rms(&window));
    }

    let max = rms_per_second.iter().cloned().fold(0.0f64, f64::max);
    let min = rms_per_second.iter().cloned().fold(f64::MAX, f64::min);
    assert!(min > 1e-6, "frozen tail was silent");
    let span_db = 20.0 * (max / min).log10();
    assert!(
        span_db <= 0.1,
        "frozen RMS spanned {span_db:.3} dB over 10 s (>0.1 dB)"
    );
}

/// Resetting a heavily-modulated engine clears its lines cleanly: a silent
/// input immediately yields exact silence, with no residual click.
#[test]
fn modulation_reset_does_not_click() {
    let params = FdnParams {
        nominal_delays: BASE_DELAYS,
        decay_s: 4.0,
        damping_hz: 6_000.0,
        lowcut_hz: 100.0,
        mod_depth: 8.0, // deep modulation
        mod_rate_hz: 3.0,
        diffusion: 0.8,
        size: 0.8,
    };
    let mut fdn = Fdn::<8>::new(SR, params);

    // Excite and run with modulation active.
    fdn.process_sample_stereo(1.0, -1.0);
    for _ in 0..4_000 {
        fdn.process_sample_stereo(0.0, 0.0);
    }

    fdn.reset();

    // First and subsequent silent samples are exactly zero — no click, no
    // leftover tail from the pre-reset modulated state.
    for index in 0..2_000 {
        let (left, right) = fdn.process_sample_stereo(0.0, 0.0);
        assert_eq!(left, 0.0, "click/residual after reset at {index}");
        assert_eq!(right, 0.0, "click/residual after reset at {index}");
    }
}

/// A decaying tail reaches *exact* zero: `write`'s flush-to-zero keeps the
/// feedback path out of the denormal slow path (RT-safety audit).
#[test]
fn tail_flushes_to_exact_zero() {
    let mut fdn = Fdn::<8>::new(SR, PRESETS[0].params); // Room: short decay
    fdn.process_sample_stereo(1.0, 1.0);

    let mut reached_zero = false;
    for _ in 0..(15 * SR as usize) {
        let (left, right) = fdn.process_sample_stereo(0.0, 0.0);
        if left == 0.0 && right == 0.0 {
            reached_zero = true;
            break;
        }
    }
    assert!(
        reached_zero,
        "tail never flushed to exact zero (denormal creep)"
    );
}
