//! End-to-end tests for the `ReverbEngine` (the wet bus + FDN voice bank +
//! preset crossfade).  The per-primitive maths is unit-tested inside each
//! module; these front-run the Phase-4 acceptance suite: dry/wet mixing,
//! pre-delay, width, decay, freeze, bypass and click-free preset switching.

use rt_fdn_reverb::{ReverbEngine, ReverbParams};

const SR: u32 = 48_000;

/// Neutral wet-only params: dry muted, wet unity, EQ wide open, width 1, no
/// crossfade.  Isolates the reverb path for measurement.
fn wet_only() -> ReverbParams {
    ReverbParams {
        dry_db: -60.0,
        wet_db: 0.0,
        low_cut_hz: 20.0,
        high_cut_hz: 20_000.0,
        width: 1.0,
        crossfade_ms: 5.0,
        ..ReverbParams::default()
    }
}

/// Feed an impulse then `tail` samples of silence; return both wet channels.
fn impulse_response(engine: &mut ReverbEngine, tail: usize) -> (Vec<f32>, Vec<f32>) {
    let mut left = Vec::with_capacity(tail + 1);
    let mut right = Vec::with_capacity(tail + 1);
    let (l0, r0) = engine.process_sample(1.0, 1.0);
    left.push(l0);
    right.push(r0);
    for _ in 0..tail {
        let (l, r) = engine.process_sample(0.0, 0.0);
        left.push(l);
        right.push(r);
    }
    (left, right)
}

/// Windowed RMS centred on `centre`.
fn windowed_rms(signal: &[f32], centre: usize, half: usize) -> f32 {
    let lo = centre.saturating_sub(half);
    let hi = (centre + half).min(signal.len());
    if hi <= lo {
        return 0.0;
    }
    let sum_sq: f32 = signal[lo..hi].iter().map(|&x| x * x).sum();
    (sum_sq / (hi - lo) as f32).sqrt()
}

#[test]
fn silence_in_silence_out() {
    let mut engine = ReverbEngine::new(SR, &wet_only());
    for _ in 0..SR {
        let (l, r) = engine.process_sample(0.0, 0.0);
        assert_eq!(l, 0.0);
        assert_eq!(r, 0.0);
    }
}

#[test]
fn bypass_copies_input() {
    let params = ReverbParams {
        bypass: true,
        ..wet_only()
    };
    let mut engine = ReverbEngine::new(SR, &params);
    for &x in &[0.5, -0.3, 0.9, -1.0] {
        let (l, r) = engine.process_sample(x, -x);
        assert_eq!(l, x);
        assert_eq!(r, -x);
    }
}

#[test]
fn disabled_active_slot_plays_dry() {
    // Active slot disabled → wet muted; dry at 0 dB passes the input through.
    let mut params = wet_only();
    params.dry_db = 0.0;
    params.enabled[params.current_preset] = false;
    let mut engine = ReverbEngine::new(SR, &params);
    let (l, r) = engine.process_sample(0.5, -0.25);
    assert!((l - 0.5).abs() < 1e-6, "dry left not passed: {l}");
    assert!((r + 0.25).abs() < 1e-6, "dry right not passed: {r}");
    // …and the rest stays dry (silent for a silent input).
    for _ in 0..1000 {
        let (l, r) = engine.process_sample(0.0, 0.0);
        assert_eq!(l, 0.0);
        assert_eq!(r, 0.0);
    }
}

#[test]
fn wet_path_produces_a_decaying_tail() {
    let mut engine = ReverbEngine::new(SR, &wet_only());
    let (left, _) = impulse_response(&mut engine, 3 * SR as usize);
    let early = windowed_rms(&left, SR as usize / 20, 512);
    let late = windowed_rms(&left, (2.9 * SR as f32) as usize, 512);
    assert!(early > 1e-4, "wet path was silent");
    assert!(
        late < early,
        "tail did not decay: early {early}, late {late}"
    );
}

#[test]
fn longer_decay_rings_longer() {
    let short_params = ReverbParams {
        decay_s: 0.4,
        ..wet_only()
    };
    let long_params = ReverbParams {
        decay_s: 10.0,
        ..wet_only()
    };
    let mut short = ReverbEngine::new(SR, &short_params);
    let mut long = ReverbEngine::new(SR, &long_params);
    let (short_l, _) = impulse_response(&mut short, 4 * SR as usize);
    let (long_l, _) = impulse_response(&mut long, 4 * SR as usize);

    let probe = (2.5 * SR as f32) as usize;
    let short_rms = windowed_rms(&short_l, probe, 2048);
    let long_rms = windowed_rms(&long_l, probe, 2048);
    assert!(
        long_rms > short_rms * 4.0,
        "long decay ({long_rms}) not clearly louder than short ({short_rms}) at 2.5 s"
    );
}

#[test]
fn freeze_sustains_the_tail() {
    let mut engine = ReverbEngine::new(SR, &wet_only());
    // Build up a tail.
    engine.process_sample(1.0, 1.0);
    for _ in 0..(SR as usize / 2) {
        engine.process_sample(0.0, 0.0);
    }
    // Engage freeze and record several seconds.
    let mut frozen = wet_only();
    frozen.freeze = true;
    engine.apply_params(&frozen);
    let hold = 5 * SR as usize;
    let mut tail = Vec::with_capacity(hold);
    for _ in 0..hold {
        let (l, _) = engine.process_sample(0.0, 0.0);
        assert!(l.is_finite());
        tail.push(l);
    }
    let early = windowed_rms(&tail, SR as usize / 2, 4096);
    let late = windowed_rms(&tail, (4.5 * SR as f32) as usize, 4096);
    assert!(early > 1e-4, "frozen tail was silent");
    assert!(
        late > early * 0.5,
        "frozen tail decayed: early {early}, late {late}"
    );
}

#[test]
fn predelay_shifts_the_wet_onset() {
    let onset = |predelay_ms: f32| -> usize {
        let params = ReverbParams {
            predelay_ms,
            ..wet_only()
        };
        let mut engine = ReverbEngine::new(SR, &params);
        let (left, _) = impulse_response(&mut engine, SR as usize);
        left.iter()
            .position(|&x| x.abs() > 1e-6)
            .unwrap_or(usize::MAX)
    };
    let base = onset(0.0);
    let delayed = onset(2.0); // 96 samples at 48 kHz
    assert!(
        base < usize::MAX && delayed < usize::MAX,
        "no wet onset found"
    );
    let shift = delayed - base;
    assert!(
        (86..=106).contains(&shift),
        "pre-delay shift {shift} not ≈ 96 samples"
    );
}

#[test]
fn zero_width_collapses_wet_to_mono() {
    let params = ReverbParams {
        width: 0.0,
        ..wet_only()
    };
    let mut engine = ReverbEngine::new(SR, &params);
    // Asymmetric stereo impulse → the FDN decorrelates L/R, but width 0 folds
    // both to the mid, so the two outputs must be identical.
    engine.process_sample(1.0, -1.0);
    for _ in 0..SR as usize {
        let (l, r) = engine.process_sample(0.0, 0.0);
        assert!((l - r).abs() < 1e-6, "width 0 not mono: {l} vs {r}");
    }
}

#[test]
fn wet_output_is_stereo_decorrelated() {
    let mut engine = ReverbEngine::new(SR, &wet_only());
    let (left, right) = impulse_response(&mut engine, SR as usize);
    let max_diff = left
        .iter()
        .zip(right.iter())
        .map(|(l, r)| (l - r).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff > 1e-3, "channels effectively identical (mono)");
}

#[test]
fn dry_wet_faders_scale_independently() {
    // Dry only: a unit impulse comes straight through at the dry gain.
    let dry_params = ReverbParams {
        dry_db: 0.0,
        wet_db: -60.0,
        ..wet_only()
    };
    let mut engine = ReverbEngine::new(SR, &dry_params);
    let (l, _) = engine.process_sample(0.7, 0.7);
    assert!((l - 0.7).abs() < 1e-6, "dry gain wrong: {l}");
}

#[test]
fn preset_switch_crossfades_cleanly() {
    let mut params = wet_only();
    params.crossfade_ms = 50.0;
    let mut engine = ReverbEngine::new(SR, &params);
    assert_eq!(engine.active_space(), params.active_space());

    // Prime a tail on preset 0, then switch to preset 1.
    engine.process_sample(1.0, 1.0);
    for _ in 0..1000 {
        engine.process_sample(0.0, 0.0);
    }
    params.current_preset = 1;
    engine.apply_params(&params);
    assert!(engine.is_crossfading());
    assert_eq!(engine.active_space(), 1);

    // Drive the crossfade with a steady tone; output must stay finite and
    // free of gross discontinuities (equal-power blend, no click).
    let mut prev = 0.0f32;
    let fade_samples = (SR as f32 * 0.05) as usize + 100;
    for i in 0..fade_samples {
        let x = (i as f32 * 0.02).sin() * 0.3;
        let (l, _) = engine.process_sample(x, x);
        assert!(l.is_finite());
        assert!((l - prev).abs() < 0.5, "discontinuity at {i}: {prev} → {l}");
        prev = l;
    }
    assert!(!engine.is_crossfading(), "crossfade did not complete");
}

#[test]
fn stays_finite_at_extremes() {
    // Max decay + full size + freeze toggling under continuous input.
    let mut params = ReverbParams {
        decay_s: 20.0,
        size: 1.0,
        damping: 0.0,
        diffusion: 1.0,
        mod_depth: 1.0,
        mod_rate_hz: 5.0,
        ..wet_only()
    };
    let mut engine = ReverbEngine::new(SR, &params);
    for block in 0..200 {
        params.freeze = block % 50 == 0;
        params.current_preset = (block / 25) % 8;
        engine.apply_params(&params);
        for i in 0..256 {
            let x = (i as f32 * 0.05).sin() * 0.5;
            let (l, r) = engine.process_sample(x, -x);
            assert!(l.is_finite() && r.is_finite(), "diverged at block {block}");
        }
    }
}

#[test]
fn works_at_96k() {
    let mut engine = ReverbEngine::new(96_000, &wet_only());
    engine.process_sample(1.0, 1.0);
    for _ in 0..96_000 {
        let (l, r) = engine.process_sample(0.0, 0.0);
        assert!(l.is_finite() && r.is_finite());
    }
}
