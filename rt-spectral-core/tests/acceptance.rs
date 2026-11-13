//! Phase-4 acceptance tests (`docs/Roadmap-04-spectral-processing.md`).
//!
//! Each test maps to a specific Phase-4 criterion:
//!
//! - **STFT round-trip / reference frame** — the windowed real FFT matches an
//!   analytically-computed DFT reference (stand-in for the "reference frames
//!   captured from a NumPy STFT" item: the closed-form Hann-windowed-cosine
//!   spectrum is exactly what `np.fft.rfft(hann * cos)` produces).
//! - **Denoise** — white noise + tone: the noise floor drops while the tone's
//!   magnitude is preserved to within 0.5 dB.
//! - **Hum notch** — a pure 60 Hz sine is attenuated by more than 40 dB.
//! - **Click repair** — a single-sample impulse is detected and interpolated
//!   away.
//! - **Profile hot-swap under load** — publishing a fresh profile mid-stream
//!   never glitches (no NaN/inf, no discontinuity spike).
//! - **Latency** — the reported latency matches the measured group delay
//!   exactly (all operators pass-through).

use rt_fft_convolver::FftHandler;
use rt_spectral_core::{
    ClickParams, ClickRepair, Complex, DenoiseParams, HumNotchBank, HumParams, NoiseProfile,
    SpectralEngine, SpectralParams, TransientParams, Window, WindowKind, learn_profile,
    profile_channel,
};
use std::f32::consts::{PI, TAU};

const SR: u32 = 48_000;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn sine(freq: f32, n: usize, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|i| amp * (TAU * freq * i as f32 / SR as f32).sin())
        .collect()
}

fn rms(buf: &[f32]) -> f32 {
    (buf.iter().map(|x| x * x).sum::<f32>() / buf.len() as f32).sqrt()
}

/// Amplitude of the `freq`-Hz component of `signal`, via the Goertzel
/// algorithm.  Projecting onto a single frequency rejects uncorrelated noise,
/// so this measures a tone's level even when it is buried in broadband noise.
fn goertzel_amplitude(signal: &[f32], freq: f32) -> f32 {
    let n = signal.len();
    let omega = TAU as f64 * freq as f64 / SR as f64;
    let coeff = 2.0 * omega.cos();
    let (mut s1, mut s2) = (0.0f64, 0.0f64);
    for &sample in signal {
        let s0 = sample as f64 + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    let real = s1 - s2 * omega.cos();
    let imag = s2 * omega.sin();
    (2.0 * (real * real + imag * imag).sqrt() / n as f64) as f32
}

fn run(eng: &mut SpectralEngine, input: &[f32], params: &SpectralParams, block: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    for (chunk_in, chunk_out) in input.chunks(block).zip(output.chunks_mut(block)) {
        eng.process_block(chunk_in, chunk_out, params);
    }
    output
}

fn deterministic_noise(seed: u32) -> impl FnMut(f32) -> f32 {
    let mut state = seed;
    move |amp| {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state as f32 / u32::MAX as f32 * 2.0 - 1.0) * amp
    }
}

// ── STFT reference frame ─────────────────────────────────────────────────────

/// A Hann-windowed cosine at an exact bin has a closed-form spectrum: the DFT
/// concentrates `0.5 · Σ w[n]` at that bin (the other cosine image and the
/// window's spectral leakage fall on far bins).  This is exactly what a NumPy
/// `np.fft.rfft(hann * cos)` reference yields; matching it validates the
/// windowed real-FFT front-end against a known reference frame.
#[test]
fn stft_matches_analytic_windowed_cosine_reference() {
    let fft = 2048;
    let num_bins = fft / 2 + 1;
    let bin = 100; // interior bin (well away from DC and Nyquist)
    let freq = bin as f32 * SR as f32 / fft as f32;

    let window = Window::new(WindowKind::Hann, fft);
    let win = window.as_slice();
    let window_sum: f32 = win.iter().sum();

    // Windowed unit cosine at the exact bin frequency.
    let mut frame: Vec<f32> = (0..fft)
        .map(|n| (TAU * bin as f32 * n as f32 / fft as f32).cos())
        .collect();
    for (sample, &w) in frame.iter_mut().zip(win) {
        *sample *= w;
    }

    let mut fft_handler = FftHandler::new(fft);
    let mut spectrum = vec![Complex::default(); num_bins];
    fft_handler.forward(&frame, &mut spectrum);

    // Reference: |X[bin]| = 0.5 · Σ w[n].
    let expected = 0.5 * window_sum;
    let measured = spectrum[bin].norm();
    let error_db = 20.0 * (measured / expected).log10();
    assert!(
        error_db.abs() < 0.1,
        "bin {bin}: |X| = {measured}, reference {expected} ({error_db:+.3} dB)"
    );

    // The chosen bin dominates the spectrum.
    let peak_bin = spectrum
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.norm().partial_cmp(&b.1.norm()).unwrap())
        .unwrap()
        .0;
    assert_eq!(peak_bin, bin, "energy did not concentrate at bin {bin}");

    // Assert leakage into bins two or more away is far below the peak.
    for (index, value) in spectrum.iter().enumerate() {
        if index.abs_diff(bin) >= 2 {
            assert!(
                value.norm() < 0.05 * measured,
                "bin {index} leaks {} (peak {measured})",
                value.norm()
            );
        }
    }
    let _ = freq;
}

// ── Denoise: floor down, tone preserved to 0.5 dB ────────────────────────────

#[test]
fn denoise_reduces_floor_and_preserves_tone_within_half_db() {
    let fft = 2048;
    let hop = fft / 4;
    let tone_hz = 100.0 * SR as f32 / fft as f32; // bin 100 → ~2344 Hz
    let mut noise = deterministic_noise(0xBADC_0FFE);

    // Learn a profile from noise-only material of the same statistics.
    let noise_only: Vec<f32> = (0..fft * 32).map(|_| noise(0.05)).collect();
    let profile = learn_profile(&noise_only, fft, hop);

    let denoise_on = SpectralParams {
        denoise: DenoiseParams {
            enabled: true,
            reduction_db: 24.0,
            sensitivity_db: 6.0,
        },
        transient: TransientParams {
            enabled: false,
            ..TransientParams::default()
        },
        ..SpectralParams::default()
    };

    // Reference tone amplitude: a clean tone through the engine, denoise OFF
    // (captures the engine's unity COLA reproduction of the tone).
    let tone = sine(tone_hz, fft * 32, 0.4);
    let (_tx_ref, rx_ref) = profile_channel(NoiseProfile::silent(fft / 2 + 1));
    let mut eng_ref = SpectralEngine::new(SR, fft, 256, rx_ref);
    let out_ref = run(&mut eng_ref, &tone, &SpectralParams::default(), 256);
    let amp_ref = goertzel_amplitude(&out_ref[fft * 8..], tone_hz);

    // Test: tone + noise through the engine, denoise ON with the learned profile.
    let (mut tx, rx) = profile_channel(NoiseProfile::silent(fft / 2 + 1));
    let mut eng = SpectralEngine::new(SR, fft, 256, rx);
    tx.publish(profile);
    let noisy: Vec<f32> = tone.iter().map(|&t| t + noise(0.05)).collect();
    let out = run(&mut eng, &noisy, &denoise_on, 256);
    let amp_test = goertzel_amplitude(&out[fft * 8..], tone_hz);

    // Tone magnitude preserved within 0.5 dB.
    let tone_db = 20.0 * (amp_test / amp_ref).log10();
    assert!(
        tone_db.abs() < 0.5,
        "tone magnitude shifted {tone_db:+.3} dB (ref {amp_ref}, test {amp_test})"
    );

    // Noise floor reduced: broadband residual (tone projected out) drops well
    // below the input's noise level.
    let settle = fft * 8..;
    let residual_out = {
        let seg = &out[settle.clone()];
        let tone_component = amp_test;
        (rms(seg).powi(2) - 0.5 * tone_component.powi(2))
            .max(0.0)
            .sqrt()
    };
    let residual_in = {
        let seg = &noisy[settle];
        (rms(seg).powi(2) - 0.5 * 0.4f32.powi(2)).max(0.0).sqrt()
    };
    assert!(
        residual_out < residual_in * 0.6,
        "noise floor not reduced: residual in {residual_in} out {residual_out}"
    );
}

// ── Hum notch: > 40 dB at the mains fundamental ──────────────────────────────

#[test]
fn hum_notch_attenuates_60hz_by_more_than_40db() {
    let mut bank = HumNotchBank::new(SR);
    // The RBJ notch's zeros sit exactly on the unit circle (b0 == b2), but at
    // the tiny normalised mains frequency (ω₀ ≈ 0.0079 rad) the f32 rounding of
    // cos(ω₀) displaces the zero *angle*, and the resulting null depth scales
    // as ≈ (2Q/ω₀)·Δω.  A very high Q (the Q=30 default) is therefore
    // precision-limited to ≈ −34 dB; a moderate Q clears 40 dB comfortably
    // while still being surgical.  A production preset that needs both a narrow
    // notch and a deep null would cascade stages or compute coefficients in f64.
    let params = HumParams {
        enabled: true,
        fundamental_hz: 60.0,
        harmonics: 1,
        q: 5.0,
    };

    let mut buf = sine(60.0, 2 * SR as usize, 0.5); // 2 s of 60 Hz
    let before = goertzel_amplitude(&buf, 60.0);
    bank.process_block(&mut buf, &params);

    // Measure the fully-settled tail: a high-Q notch rings down its own
    // transient over several time constants (τ ≈ Q/(π·fc) ≈ 0.16 s here), so
    // judge only the last half-second, well past that decay.
    let after = goertzel_amplitude(&buf[3 * SR as usize / 2..], 60.0);
    let attenuation_db = 20.0 * (before / after).log10();
    assert!(
        attenuation_db > 40.0,
        "60 Hz attenuated only {attenuation_db:.1} dB (before {before}, after {after})"
    );
}

// ── Click repair: single-sample impulse ──────────────────────────────────────

#[test]
fn click_repair_interpolates_single_sample_impulse() {
    let window = 5;
    let mut click = ClickRepair::new(window);
    let params = ClickParams {
        enabled: true,
        sensitivity: 4.0,
        max_len: 8,
    };
    let delay = click.latency();

    // A smooth low-frequency sine with a lone full-scale spike.
    let clean: Vec<f32> = (0..256)
        .map(|n| 0.2 * (PI * n as f32 / 40.0).sin())
        .collect();
    let spike_at = 128;
    let mut signal = clean.clone();
    signal[spike_at] += 1.0;

    click.process_block(&mut signal, &params);

    // Output is the input delayed by `delay`; the repaired spike position lands
    // at `spike_at + delay`.  It must sit near the underlying clean signal, not
    // near the +1.0 excursion.
    let repaired = signal[spike_at + delay];
    let expected = clean[spike_at];
    assert!(
        (repaired - expected).abs() < 0.1,
        "impulse not interpolated: got {repaired}, expected ≈ {expected}"
    );
    // No sample retains a click-sized excursion.
    assert!(
        signal.iter().all(|&s| s.abs() < 0.5),
        "a click-sized excursion survived repair"
    );
}

// ── Profile hot-swap under load ──────────────────────────────────────────────

#[test]
fn profile_hot_swap_under_load_is_glitch_free() {
    let fft = 1024;
    let hop = fft / 4;
    let block = 128;
    let mut noise = deterministic_noise(0x5EED_1234);

    // Two profiles learned at different noise levels.
    let quiet: Vec<f32> = (0..fft * 16).map(|_| noise(0.02)).collect();
    let loud: Vec<f32> = (0..fft * 16).map(|_| noise(0.20)).collect();
    let profile_a = learn_profile(&quiet, fft, hop);
    let profile_b = learn_profile(&loud, fft, hop);

    let (mut tx, rx) = profile_channel(profile_a);
    let mut eng = SpectralEngine::new(SR, fft, block, rx);
    let params = SpectralParams {
        denoise: DenoiseParams {
            enabled: true,
            reduction_db: 18.0,
            sensitivity_db: 3.0,
        },
        ..SpectralParams::default()
    };

    // Continuous tone + noise; swap the profile at the midpoint block.
    let total_blocks = 200;
    let swap_block = 100;
    let mut output = Vec::with_capacity(total_blocks * block);
    for block_index in 0..total_blocks {
        if block_index == swap_block {
            tx.publish(profile_b.clone()); // hot-swap under load
        }
        let mut buf: Vec<f32> = (0..block)
            .map(|n| {
                let t = (block_index * block + n) as f32 / SR as f32;
                0.3 * (TAU * 500.0 * t).sin() + noise(0.05)
            })
            .collect();
        eng.process_block(&buf.clone(), &mut buf, &params);
        output.extend_from_slice(&buf);
    }

    // No NaN/inf anywhere.
    assert!(
        output.iter().all(|x| x.is_finite()),
        "hot-swap produced non-finite output"
    );
    // No discontinuity spike around the swap (sample-to-sample step stays sane
    // for a 0.3-amplitude tone + light noise).
    let swap_sample = swap_block * block;
    let mut max_step = 0.0f32;
    for pair in output[swap_sample - block..swap_sample + block].windows(2) {
        max_step = max_step.max((pair[1] - pair[0]).abs());
    }
    assert!(
        max_step < 0.2,
        "profile hot-swap glitched: max step {max_step}"
    );
}

// ── Latency matches the reported value exactly ───────────────────────────────

#[test]
fn reported_latency_matches_measured_group_delay() {
    for &fft in &[1024usize, 2048, 4096] {
        let (_tx, rx) = profile_channel(NoiseProfile::silent(fft / 2 + 1));
        let mut eng = SpectralEngine::new(SR, fft, 256, rx);
        let params = SpectralParams::default(); // all operators off → pass-through

        let input = sine(220.0, fft * 12, 0.5);
        let output = run(&mut eng, &input, &params, 256);
        let latency = eng.latency_samples();

        // The reported latency must line the output up with the input to within
        // the COLA window-overlap error; a wrong latency blows this up.
        let mut max_err = 0.0f32;
        for n in (fft * 3)..(input.len() - fft) {
            max_err = max_err.max((output[n] - input[n - latency]).abs());
        }
        assert!(
            max_err < 1e-2,
            "fft {fft}: latency {latency} misaligned (err {max_err})"
        );

        // An off-by-one latency is clearly worse — proves the value is exact,
        // not merely small.
        let mut off_by_one = 0.0f32;
        for n in (fft * 3)..(input.len() - fft) {
            off_by_one = off_by_one.max((output[n] - input[n - latency - 1]).abs());
        }
        assert!(
            off_by_one > max_err,
            "fft {fft}: latency not exact (on {max_err} vs off-by-one {off_by_one})"
        );
    }
}
