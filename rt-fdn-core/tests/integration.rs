//! Engine-level integration tests for the FDN reverb.
//!
//! These exercise `Fdn` end-to-end (the per-primitive maths is unit-tested
//! inside each module).  They front-run the Phase 4 acceptance suite:
//! stability, decay ordering, stereo behaviour and clean reset.

use rt_fdn_core::{Fdn, PRESETS};

const SR: f32 = 48_000.0;

/// Feed an impulse (both channels) then `tail` samples of silence, returning
/// the wet output of both channels.
fn impulse_response(reverb: &mut Fdn<8>, tail: usize) -> (Vec<f32>, Vec<f32>) {
    let mut left = Vec::with_capacity(tail + 1);
    let mut right = Vec::with_capacity(tail + 1);
    let (first_l, first_r) = reverb.process_sample_stereo(1.0, 1.0);
    left.push(first_l);
    right.push(first_r);
    for _ in 0..tail {
        let (out_l, out_r) = reverb.process_sample_stereo(0.0, 0.0);
        left.push(out_l);
        right.push(out_r);
    }
    (left, right)
}

/// Windowed RMS of `signal` in a window centred on `centre`.
fn windowed_rms(signal: &[f32], centre: usize, half_width: usize) -> f32 {
    let lo = centre.saturating_sub(half_width);
    let hi = (centre + half_width).min(signal.len());
    if hi <= lo {
        return 0.0;
    }
    let sum_sq: f32 = signal[lo..hi].iter().map(|&x| x * x).sum();
    (sum_sq / (hi - lo) as f32).sqrt()
}

/// A silent input produces a silent output — the network never self-oscillates
/// from its zero state.
#[test]
fn silence_in_silence_out() {
    let mut reverb = Fdn::<8>::new(SR, PRESETS[1].params);
    for _ in 0..48_000 {
        let (out_l, out_r) = reverb.process_sample_stereo(0.0, 0.0);
        assert_eq!(out_l, 0.0);
        assert_eq!(out_r, 0.0);
    }
}

/// An impulse produces a decaying tail: energy late in the response is well
/// below energy early in it.
#[test]
fn impulse_response_decays() {
    let mut reverb = Fdn::<8>::new(SR, PRESETS[0].params); // Room, short decay
    let (left, _right) = impulse_response(&mut reverb, 2 * SR as usize);

    let early = windowed_rms(&left, SR as usize / 20, 512); // ~50 ms
    let late = windowed_rms(&left, (1.8 * SR) as usize, 512); // ~1.8 s
    assert!(early > 0.0, "no early energy at all");
    assert!(
        late < early * 0.1,
        "tail did not decay: early {early}, late {late}"
    );
}

/// A longer requested decay must actually ring longer than a shorter one.
#[test]
fn longer_decay_rings_longer() {
    let params = PRESETS[1].params;

    let mut short = Fdn::<8>::new(SR, params);
    short.set_decay(0.5);
    let (short_l, _) = impulse_response(&mut short, 3 * SR as usize);

    let mut long = Fdn::<8>::new(SR, params);
    long.set_decay(8.0);
    let (long_l, _) = impulse_response(&mut long, 3 * SR as usize);

    let probe = (2.0 * SR) as usize; // 2 s in
    let short_rms = windowed_rms(&short_l, probe, 1024);
    let long_rms = windowed_rms(&long_l, probe, 1024);
    assert!(
        long_rms > short_rms * 4.0,
        "long decay ({long_rms}) not clearly louder than short ({short_rms}) at 2 s"
    );
}

/// At the maximum decay the network stays bounded over a full minute — the
/// lossless matrix plus per-branch gains < 1 guarantee stability.
#[test]
fn stable_at_max_decay_over_60s() {
    let mut reverb = Fdn::<8>::new(SR, PRESETS[5].params); // Cave, long
    reverb.set_decay(20.0);
    reverb.set_size(1.0);

    // Kick it with an impulse, then run 60 s of silence.
    reverb.process_sample_stereo(1.0, 1.0);
    let mut peak = 0.0f32;
    for _ in 0..(60 * SR as usize) {
        let (out_l, out_r) = reverb.process_sample_stereo(0.0, 0.0);
        assert!(
            out_l.is_finite() && out_r.is_finite(),
            "diverged to non-finite"
        );
        peak = peak.max(out_l.abs()).max(out_r.abs());
    }
    assert!(peak < 100.0, "output grew unreasonably large: peak {peak}");
}

/// The two output channels are decorrelated (not identical), giving a real
/// stereo image even from a mono impulse.
#[test]
fn output_channels_are_decorrelated() {
    let mut reverb = Fdn::<8>::new(SR, PRESETS[1].params);
    let (left, right) = impulse_response(&mut reverb, SR as usize);
    // Somewhere in the tail the channels must differ appreciably.
    let max_diff = left
        .iter()
        .zip(right.iter())
        .map(|(l, r)| (l - r).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff > 1e-3, "channels are effectively identical (mono)");
}

/// `reset` silences the tail: after resetting mid-decay the output returns to
/// exact zero for a silent input.
#[test]
fn reset_silences_tail() {
    let mut reverb = Fdn::<8>::new(SR, PRESETS[1].params);
    reverb.process_sample_stereo(1.0, 1.0);
    for _ in 0..1_000 {
        reverb.process_sample_stereo(0.0, 0.0);
    }
    reverb.reset();
    let (out_l, out_r) = reverb.process_sample_stereo(0.0, 0.0);
    assert_eq!(out_l, 0.0);
    assert_eq!(out_r, 0.0);
}

/// Every built-in preset is stable and finite over a couple of seconds.
#[test]
fn all_presets_are_finite_and_stable() {
    for preset in PRESETS.iter() {
        let mut reverb = Fdn::<8>::new(SR, preset.params);
        reverb.process_sample_stereo(1.0, 1.0);
        let mut peak = 0.0f32;
        for _ in 0..(2 * SR as usize) {
            let (out_l, out_r) = reverb.process_sample_stereo(0.0, 0.0);
            assert!(
                out_l.is_finite() && out_r.is_finite(),
                "{} produced non-finite output",
                preset.name
            );
            peak = peak.max(out_l.abs()).max(out_r.abs());
        }
        assert!(peak < 100.0, "{} grew too large: {peak}", preset.name);
    }
}

/// A live size change does not blow up or click into non-finite territory
/// (the smoother glides the delay lengths).
#[test]
fn live_size_sweep_stays_finite() {
    let mut reverb = Fdn::<8>::new(SR, PRESETS[1].params);
    for block in 0..200 {
        // Alternate size targets to force continuous glides.
        reverb.set_size(if block % 2 == 0 { 0.2 } else { 1.0 });
        for _ in 0..256 {
            let (out_l, out_r) = reverb.process_sample_stereo(0.5, -0.5);
            assert!(out_l.is_finite() && out_r.is_finite());
        }
    }
}

/// Freezing pins the tail: after building up energy and engaging freeze (with
/// no further input), the tail sustains instead of decaying — late RMS stays
/// close to the RMS at the freeze point, unlike the un-frozen reference which
/// dies away.
#[test]
fn freeze_sustains_the_tail() {
    // Build up a tail from an impulse over ~0.5 s, then run 5 s more.
    let build = SR as usize / 2;
    let hold = 5 * SR as usize;

    let mut frozen = Fdn::<8>::new(SR, PRESETS[1].params);
    frozen.process_sample_stereo(1.0, 1.0);
    for _ in 0..build {
        frozen.process_sample_stereo(0.0, 0.0);
    }
    frozen.set_freeze(true);
    assert!(frozen.is_frozen());
    let mut tail = Vec::with_capacity(hold);
    for _ in 0..hold {
        // No new input while frozen (as the plugin mutes the FDN input).
        let (out_l, _) = frozen.process_sample_stereo(0.0, 0.0);
        assert!(out_l.is_finite(), "frozen tail diverged");
        tail.push(out_l);
    }

    let early = windowed_rms(&tail, SR as usize / 2, 4096);
    let late = windowed_rms(&tail, (4.5 * SR) as usize, 4096);
    assert!(early > 1e-4, "frozen tail was silent");
    // Energy is retained: late is within a factor of two of early (a decaying
    // tail would be orders of magnitude lower).
    assert!(
        late > early * 0.5,
        "frozen tail decayed: early {early}, late {late}"
    );

    // Thawing restores decay: after thaw, a silent input lets the tail die.
    frozen.set_freeze(false);
    assert!(!frozen.is_frozen());
    let mut thawed = Vec::with_capacity(hold);
    for _ in 0..hold {
        let (out_l, _) = frozen.process_sample_stereo(0.0, 0.0);
        thawed.push(out_l);
    }
    let thawed_late = windowed_rms(&thawed, (4.5 * SR) as usize, 4096);
    assert!(
        thawed_late < late * 0.5,
        "thawed tail did not decay: {thawed_late} vs frozen {late}"
    );
}

/// The engine works at a non-reference sample rate (delay lengths rescale).
#[test]
fn works_at_96k() {
    let mut reverb = Fdn::<8>::new(96_000.0, PRESETS[1].params);
    reverb.process_sample_stereo(1.0, 1.0);
    for _ in 0..96_000 {
        let (out_l, out_r) = reverb.process_sample_stereo(0.0, 0.0);
        assert!(out_l.is_finite() && out_r.is_finite());
    }
}
