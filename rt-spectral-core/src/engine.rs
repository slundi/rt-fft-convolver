//! `SpectralEngine` — the per-channel signal path that wires the Phase 0
//! decisions together.
//!
//! ```text
//!   in ─▶ click repair ─▶ STFT( window·FFT ▶ denoise ▶ gate ▶ compress ▶
//!                               IFFT·window·overlap-add ) ─▶ hum notch ─▶ wet
//!   in ─▶ delay(L) ────────────────────────────────────────────────────▶ dry
//!   out = bypass ? dry : (dry·(1−mix) + wet·mix) · output_gain
//! ```
//!
//! One instance handles one channel; a stereo plugin owns two.  All buffers are
//! allocated in [`SpectralEngine::new`]; [`SpectralEngine::process_block`] is
//! allocation-, lock- and panic-free and wraps the render in a
//! [`DenormalGuard`] (spectral gains near the noise floor are the classic
//! denormal case).
//!
//! **Latency** is constant and block-size independent: one full STFT `frame`
//! (the analyzer prime `frame − hop` plus a one-`hop` synthesis-FIFO cushion
//! that keeps the delay stable even for host blocks smaller than a hop) plus
//! the click despiker's fixed group delay (always imposed, so toggling click
//! never changes the host's plugin-delay compensation).

use realfft::num_complex::Complex;
use rt_fft_convolver::{DenormalGuard, FftHandler, Ring, db_to_linear};

use crate::click::{ClickParams, ClickRepair};
use crate::compress::{CompressParams, SpectralCompressor};
use crate::denoise::{DenoiseParams, Denoiser};
use crate::gate::{GateParams, SpectralGate};
use crate::hum::{HumNotchBank, HumParams};
use crate::profile::ProfileReceiver;
use crate::stft::{Analyzer, Synthesizer};
use crate::transient::{TransientDetector, TransientParams};
use crate::window::{Window, WindowKind};

/// Decision-directed smoothing for the Wiener denoiser (Phase 0: musical-noise
/// free at ≈ 0.98).
const DD_ALPHA: f32 = 0.98;
/// Median window for the click despiker (Phase 0: order 5–9; 5 is a good
/// default with a 2-sample delay).
const CLICK_WINDOW: usize = 5;

/// The full parameter set for one processing block.  Cheap to copy; the plugin
/// snapshots its atomics into this once per block.
#[derive(Clone, Copy, Debug)]
pub struct SpectralParams {
    pub click: ClickParams,
    pub denoise: DenoiseParams,
    pub gate: GateParams,
    pub compress: CompressParams,
    pub transient: TransientParams,
    pub hum: HumParams,
    /// Dry/wet blend, `0.0` = fully dry … `1.0` = fully wet.
    pub mix: f32,
    /// Output trim in dB, applied to the mixed signal.
    pub output_gain_db: f32,
    /// Hard bypass — emits the latency-matched dry signal untouched.
    pub bypass: bool,
}

impl Default for SpectralParams {
    fn default() -> Self {
        Self {
            click: ClickParams::default(),
            denoise: DenoiseParams::default(),
            gate: GateParams::default(),
            compress: CompressParams::default(),
            transient: TransientParams::default(),
            hum: HumParams::default(),
            mix: 1.0,
            output_gain_db: 0.0,
            bypass: false,
        }
    }
}

/// Level / gain-reduction telemetry for one processed block, for the Phase-3
/// editor's meters.  All fields are in dB; gain-reduction values are `<= 0`
/// (`0` = the operator is disabled or transparent this block).  Cheap to copy.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EngineMeters {
    /// Input RMS level in dBFS over the block (`-inf`-floored to [`METER_FLOOR_DB`]).
    pub input_db: f32,
    /// Output (post-mix) RMS level in dBFS over the block.
    pub output_db: f32,
    /// Mean spectral gain reduction applied by the denoiser, in dB (`<= 0`).
    pub gr_denoise_db: f32,
    /// Mean spectral gain reduction applied by the gate, in dB (`<= 0`).
    pub gr_gate_db: f32,
    /// Mean spectral gain reduction applied by the compressor, in dB (`<= 0`).
    pub gr_compress_db: f32,
}

/// Floor for the level / gain-reduction meters — quieter than this reads as
/// "silence" and clamps here rather than diverging to `-inf`.
pub const METER_FLOOR_DB: f32 = -120.0;

/// Single-channel spectral processor.
pub struct SpectralEngine {
    sample_rate: u32,
    fft_size: usize,
    hop: usize,
    max_block: usize,
    latency: usize,

    // STFT front-end.
    fft: FftHandler,
    window: Window,
    analyzer: Analyzer,
    synth: Synthesizer,

    // Operators.
    denoiser: Denoiser,
    gate: SpectralGate,
    comp: SpectralCompressor,
    transient: TransientDetector,
    hum: HumNotchBank,
    click: ClickRepair,
    profile: ProfileReceiver,

    // Dry path delay line, primed with `latency` zeros.
    dry_delay: Ring<f32>,

    // Reusable scratch (all sized at construction).
    frame: Vec<f32>,             // time-domain analysis/synthesis frame (N)
    spectrum: Vec<Complex<f32>>, // half-spectrum (N/2 + 1)
    work: Vec<f32>,              // wet working buffer (<= max_block)
    dry: Vec<f32>,               // delayed dry buffer (<= max_block)

    // ── Telemetry (Phase-3 editor); updated every block, never read on the
    //    audio thread beyond the cheap accumulation below. ──
    /// Most recent processed frame's output magnitude spectrum (`N/2 + 1`).
    last_mag: Vec<f32>,
    /// Per-block accumulators for the input / output RMS meters.
    in_energy: f64,
    out_energy: f64,
    meter_samples: usize,
    /// Per-block accumulators for the mean per-operator gain reduction (dB).
    gr_denoise_sum: f32,
    gr_gate_sum: f32,
    gr_compress_sum: f32,
    gr_frames: usize,
}

impl SpectralEngine {
    /// Build an engine for `sample_rate`, an `fft_size`-point STFT at 75 %
    /// overlap, host blocks up to `max_block` samples, reading its noise
    /// profile from `profile`.
    ///
    /// **Allocates** — off-RT only.
    ///
    /// # Panics
    /// Panics unless `fft_size` is a multiple of 4 and `>= 8` (needed for a
    /// well-formed 75 %-overlap hop and an even real FFT).
    pub fn new(
        sample_rate: u32,
        fft_size: usize,
        max_block: usize,
        profile: ProfileReceiver,
    ) -> Self {
        assert!(
            fft_size >= 8 && fft_size.is_multiple_of(4),
            "fft_size must be a multiple of 4 and >= 8 (got {fft_size})"
        );
        let hop = fft_size / 4; // 75 % overlap (Phase 0)
        let num_bins = fft_size / 2 + 1;
        let max_block = max_block.max(1);

        let window = Window::new(WindowKind::Hann, fft_size);
        let norm = window.ola_normalisation(hop);
        let frame_rate = sample_rate as f32 / hop as f32;

        let click = ClickRepair::new(CLICK_WINDOW);
        // Block-size-independent STFT group delay is one full frame: the
        // analyzer prime (`frame − hop`) plus the synthesizer FIFO prime
        // (`hop`) — see the two `prime_zeros` docs for why one hop is the
        // minimum cushion that survives sub-hop host blocks.
        let stft_latency = fft_size;
        let latency = stft_latency + click.latency();

        // Prime the dry delay with `latency` zeros so dry and wet line up.
        let mut dry_delay = Ring::with_capacity(latency + max_block);
        for _ in 0..latency {
            dry_delay.push(0.0);
        }

        let mut analyzer = Analyzer::new(fft_size, hop, max_block);
        analyzer.prime_zeros(fft_size - hop);
        let mut synth = Synthesizer::new(fft_size, hop, max_block, norm);
        synth.prime_zeros(hop);

        Self {
            sample_rate,
            fft_size,
            hop,
            max_block,
            latency,
            fft: FftHandler::new(fft_size),
            window,
            analyzer,
            synth,
            denoiser: Denoiser::new(num_bins, DD_ALPHA),
            gate: SpectralGate::new(num_bins, frame_rate),
            comp: SpectralCompressor::new(),
            transient: TransientDetector::new(num_bins),
            hum: HumNotchBank::new(sample_rate),
            click,
            profile,
            dry_delay,
            frame: vec![0.0; fft_size],
            spectrum: vec![Complex::default(); num_bins],
            work: vec![0.0; max_block],
            dry: vec![0.0; max_block],
            last_mag: vec![0.0; num_bins],
            in_energy: 0.0,
            out_energy: 0.0,
            meter_samples: 0,
            gr_denoise_sum: 0.0,
            gr_gate_sum: 0.0,
            gr_compress_sum: 0.0,
            gr_frames: 0,
        }
    }

    /// Reported latency in samples (constant for a given `fft_size`).
    #[inline]
    pub fn latency_samples(&self) -> usize {
        self.latency
    }

    /// STFT size this engine was built for.
    #[inline]
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Hop size (`fft_size / 4`).
    #[inline]
    pub fn hop(&self) -> usize {
        self.hop
    }

    /// Sample rate this engine was built for.
    #[inline]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Number of spectrum bins (`fft_size / 2 + 1`).
    #[inline]
    pub fn num_bins(&self) -> usize {
        self.spectrum.len()
    }

    /// Most recent processed frame's **output** magnitude spectrum (`N/2 + 1`
    /// bins).  Updated once per frame; holds the last value across blocks that
    /// produced no frame.  For the Phase-3 spectrogram / spectrum display.
    #[inline]
    pub fn spectrum_magnitude(&self) -> &[f32] {
        &self.last_mag
    }

    /// Level / gain-reduction telemetry for the block just processed (all in
    /// dB).  Values are `0` before the first block.  For the Phase-3 meters.
    pub fn meters(&self) -> EngineMeters {
        let rms_db = |energy: f64, samples: usize| {
            if samples == 0 {
                return METER_FLOOR_DB;
            }
            let rms = (energy / samples as f64).sqrt() as f32;
            (20.0 * rms.max(1e-12).log10()).max(METER_FLOOR_DB)
        };
        let mean = |sum: f32, frames: usize| {
            if frames == 0 {
                0.0
            } else {
                sum / frames as f32
            }
        };
        EngineMeters {
            input_db: rms_db(self.in_energy, self.meter_samples),
            output_db: rms_db(self.out_energy, self.meter_samples),
            gr_denoise_db: mean(self.gr_denoise_sum, self.gr_frames),
            gr_gate_db: mean(self.gr_gate_sum, self.gr_frames),
            gr_compress_db: mean(self.gr_compress_sum, self.gr_frames),
        }
    }

    /// Flush every internal delay line, envelope and SNR memory to silence.
    /// The dry-delay priming is restored so latency stays correct.
    pub fn reset(&mut self) {
        self.analyzer.reset();
        self.analyzer.prime_zeros(self.fft_size - self.hop);
        self.synth.reset();
        self.synth.prime_zeros(self.hop);
        self.denoiser.reset();
        self.gate.reset();
        self.transient.reset();
        self.hum.reset();
        self.click.reset();
        self.dry_delay.clear();
        for _ in 0..self.latency {
            self.dry_delay.push(0.0);
        }
        self.last_mag.fill(0.0);
        self.reset_meters();
    }

    /// Zero the per-block meter accumulators (called at the top of every
    /// [`process_block`](Self::process_block)).
    #[inline]
    fn reset_meters(&mut self) {
        self.in_energy = 0.0;
        self.out_energy = 0.0;
        self.meter_samples = 0;
        self.gr_denoise_sum = 0.0;
        self.gr_gate_sum = 0.0;
        self.gr_compress_sum = 0.0;
        self.gr_frames = 0;
    }

    /// Process one host block: `input` → `output` (equal length, may alias
    /// different buffers).  Blocks larger than `max_block` are processed in
    /// `max_block` chunks.
    ///
    /// # Real-time safety
    /// No allocation, no lock, no panic.  Wrapped in a [`DenormalGuard`].
    pub fn process_block(&mut self, input: &[f32], output: &mut [f32], params: &SpectralParams) {
        debug_assert_eq!(input.len(), output.len());
        let _guard = DenormalGuard::new();

        // Telemetry is per-block: clear the accumulators, then each chunk adds
        // into them (see `process_chunk`).
        self.reset_meters();

        let mut offset = 0;
        while offset < input.len() {
            let n = (input.len() - offset).min(self.max_block);
            self.process_chunk(
                &input[offset..offset + n],
                &mut output[offset..offset + n],
                params,
            );
            offset += n;
        }
    }

    /// Process a single chunk no larger than `max_block`.
    fn process_chunk(&mut self, input: &[f32], output: &mut [f32], params: &SpectralParams) {
        let n = input.len();
        let win = self.window.as_slice();

        // Input level meter: accumulate the dry input energy for this chunk.
        self.in_energy += input.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>();
        self.meter_samples += n;

        // --- Wet path -------------------------------------------------------
        // Time-domain click repair (always imposes its delay; repairs only when
        // enabled) runs on a working copy of the input.
        self.work[..n].copy_from_slice(input);
        self.click.process_block(&mut self.work[..n], &params.click);

        // STFT framing → per-bin operator chain → overlap-add.
        self.analyzer.push_block(&self.work[..n]);
        while self.analyzer.next_frame(&mut self.frame) {
            // Analysis window.
            for (s, &w) in self.frame.iter_mut().zip(win) {
                *s *= w;
            }
            self.fft.forward(&self.frame, &mut self.spectrum);

            // Transient soft-bypass depth from spectral flux (Phase 0).
            let depth = self.transient.observe(&self.spectrum, &params.transient);

            // Per-bin chain: denoise → gate → compress.  Spectral energy is
            // sampled around each operator so the editor can meter per-operator
            // gain reduction (a power ratio → `10·log10`).
            let e_pre = spectrum_energy(&self.spectrum);
            self.denoiser.process(
                &mut self.spectrum,
                &params.denoise,
                self.profile.current(),
                depth,
            );
            let e_denoise = spectrum_energy(&self.spectrum);
            self.gate.process(&mut self.spectrum, &params.gate);
            let e_gate = spectrum_energy(&self.spectrum);
            self.comp
                .process(&mut self.spectrum, &params.compress, depth);
            let e_comp = spectrum_energy(&self.spectrum);

            self.gr_denoise_sum += gain_reduction_db(e_pre, e_denoise);
            self.gr_gate_sum += gain_reduction_db(e_denoise, e_gate);
            self.gr_compress_sum += gain_reduction_db(e_gate, e_comp);
            self.gr_frames += 1;

            // Latest output magnitude spectrum for the spectrogram / display.
            for (dst, bin) in self.last_mag.iter_mut().zip(self.spectrum.iter()) {
                *dst = bin.norm();
            }

            self.fft.inverse(&mut self.spectrum, &mut self.frame);
            // Synthesis window (→ Hann² weighting, compensated by the COLA norm).
            for (s, &w) in self.frame.iter_mut().zip(win) {
                *s *= w;
            }
            self.synth.add_frame(&self.frame);
        }
        // Reuse `work` to hold the wet block, then notch mains hum out of it.
        self.synth.pull_block(&mut self.work[..n]);
        self.hum.process_block(&mut self.work[..n], &params.hum);

        // --- Dry path -------------------------------------------------------
        // The original input, delayed by the engine latency to match the wet
        // signal.  Occupancy stays at `latency` (push n, pop n).
        let pushed = self.dry_delay.push_slice(input);
        debug_assert_eq!(pushed, n, "dry delay overflow");
        let popped = self.dry_delay.pop_slice(&mut self.dry[..n]);
        debug_assert_eq!(popped, n, "dry delay underflow");

        // --- Mix ------------------------------------------------------------
        let mix = params.mix.clamp(0.0, 1.0);
        let out_gain = db_to_linear(params.output_gain_db);
        let dry = &self.dry[..n];
        let wet = &self.work[..n];
        for ((out, &d), &w) in output.iter_mut().zip(dry).zip(wet) {
            *out = if params.bypass {
                d
            } else {
                (d * (1.0 - mix) + w * mix) * out_gain
            };
        }

        // Output level meter: accumulate the post-mix output energy.
        self.out_energy += output.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>();
    }
}

/// Total energy (`Σ |X(k)|²`) of a half-spectrum.
#[inline]
fn spectrum_energy(spectrum: &[Complex<f32>]) -> f32 {
    spectrum.iter().map(|bin| bin.norm_sqr()).sum()
}

/// Gain reduction from `before` → `after` spectral energy, in dB (`<= 0`),
/// floored at [`METER_FLOOR_DB`].  A non-reducing (or amplifying) step reports
/// `0`; a silent input reports `0` (nothing to reduce).
#[inline]
fn gain_reduction_db(before: f32, after: f32) -> f32 {
    if before <= 0.0 || after >= before {
        0.0
    } else {
        // Power ratio → 10·log10; already known `< 1`, so the result is `< 0`.
        (10.0 * (after / before).log10()).max(METER_FLOOR_DB)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::{NoiseProfile, learn_profile, profile_channel};
    use std::f32::consts::PI;

    const SR: u32 = 48_000;

    fn engine(fft: usize, max_block: usize) -> SpectralEngine {
        let (_tx, rx) = profile_channel(NoiseProfile::silent(fft / 2 + 1));
        SpectralEngine::new(SR, fft, max_block, rx)
    }

    /// All operators off, mix = 1.0 → the engine reconstructs its input,
    /// delayed by exactly `latency_samples()`, to within the window taper.
    #[test]
    fn passthrough_reconstructs_with_reported_latency() {
        let fft = 1024;
        let mut eng = engine(fft, 128);
        let params = SpectralParams::default(); // everything disabled, mix 1.0

        let input: Vec<f32> = (0..8192)
            .map(|n| 0.5 * (2.0 * PI * 5.0 * n as f32 / 1024.0).sin())
            .collect();
        let mut output = vec![0.0f32; input.len()];

        // Feed in 128-sample blocks.
        for (i, o) in input.chunks(128).zip(output.chunks_mut(128)) {
            eng.process_block(i, o, &params);
        }

        let lat = eng.latency_samples();
        let mut max_err = 0.0f32;
        for n in (fft * 3)..(input.len() - fft) {
            max_err = max_err.max((output[n] - input[n - lat]).abs());
        }
        assert!(
            max_err < 1e-2,
            "passthrough error {max_err} (latency {lat})"
        );
    }

    /// Reported latency is one full STFT frame (block-size independent) plus
    /// the click delay.
    #[test]
    fn latency_is_frame_plus_click() {
        let eng = engine(2048, 256);
        // frame = 2048, plus click (5-window → 2).
        assert_eq!(eng.latency_samples(), 2048 + 2);
    }

    /// The reported latency matches the actual output delay for a range of
    /// host block sizes, including blocks smaller than one hop.
    #[test]
    fn latency_is_block_size_independent() {
        let fft = 1024;
        for &bs in &[64usize, 128, 256, 512] {
            let mut eng = engine(fft, bs);
            let params = SpectralParams::default();
            let input: Vec<f32> = (0..8192)
                .map(|n| 0.5 * (2.0 * PI * 5.0 * n as f32 / 1024.0).sin())
                .collect();
            let mut output = vec![0.0f32; input.len()];
            for (i, o) in input.chunks(bs).zip(output.chunks_mut(bs)) {
                eng.process_block(i, o, &params);
            }
            let lat = eng.latency_samples();
            let mut err = 0.0f32;
            for n in 4000..6000 {
                err = err.max((output[n] - input[n - lat]).abs());
            }
            assert!(
                err < 1e-2,
                "block {bs}: latency {lat} misaligned, err {err}"
            );
        }
    }

    /// Denoise engaged with a learned profile pulls a broadband-noise input's
    /// level down; a strong tone through the same path is largely preserved.
    #[test]
    fn denoise_reduces_noise_floor() {
        let fft = 1024;
        let hop = fft / 4;
        // Deterministic pseudo-noise (xorshift → [-1, 1]).
        let mut state = 0x1234_5678u32;
        let mut noise = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let noise_buf: Vec<f32> = (0..fft * 32).map(|_| 0.1 * noise()).collect();
        let profile = learn_profile(&noise_buf, fft, hop);

        let (_tx, rx) = profile_channel(profile);
        let mut eng = SpectralEngine::new(SR, fft, 128, rx);
        let params = SpectralParams {
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

        // Fresh noise (same distribution) through the engine.
        let input: Vec<f32> = (0..fft * 32).map(|_| 0.1 * noise()).collect();
        let mut output = vec![0.0f32; input.len()];
        for (i, o) in input.chunks(128).zip(output.chunks_mut(128)) {
            eng.process_block(i, o, &params);
        }

        let rms = |b: &[f32]| (b.iter().map(|x| x * x).sum::<f32>() / b.len() as f32).sqrt();
        // Compare settled regions (past warm-up).
        let in_rms = rms(&input[fft * 4..]);
        let out_rms = rms(&output[fft * 4..]);
        assert!(
            out_rms < in_rms * 0.6,
            "noise floor not reduced: in {in_rms} out {out_rms}"
        );
    }

    /// Reset restores the primed dry-delay occupancy (latency unchanged) and
    /// produces no NaNs afterwards.
    #[test]
    fn reset_is_clean() {
        let mut eng = engine(512, 64);
        let params = SpectralParams::default();
        let block = vec![0.3f32; 64];
        let mut out = vec![0.0f32; 64];
        for _ in 0..20 {
            eng.process_block(&block, &mut out, &params);
        }
        eng.reset();
        eng.process_block(&block, &mut out, &params);
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn gain_reduction_db_is_monotone_and_bounded() {
        // No reduction (or amplification) → 0 dB.
        assert_eq!(gain_reduction_db(0.0, 5.0), 0.0);
        assert_eq!(gain_reduction_db(4.0, 4.0), 0.0);
        assert_eq!(gain_reduction_db(4.0, 8.0), 0.0);
        // Halving the power ≈ −3 dB.
        let gr = gain_reduction_db(1.0, 0.5);
        assert!((gr + 3.01).abs() < 0.05, "half-power GR = {gr}");
        // Silence floors, never diverges.
        assert_eq!(gain_reduction_db(1.0, 0.0), METER_FLOOR_DB);
    }

    /// The input meter tracks a known-RMS signal; with all operators off the
    /// output meter matches it (delayed pass-through, same long-run level) and
    /// no gain reduction is reported.
    #[test]
    fn meters_track_levels_and_zero_gr_when_disabled() {
        let mut eng = engine(1024, 256);
        let params = SpectralParams::default();

        // Full-scale 0.5-amplitude sine: RMS = 0.5/√2 ≈ 0.3536 → ≈ −9.03 dBFS.
        let block: Vec<f32> = (0..256)
            .map(|n| 0.5 * (2.0 * PI * 5.0 * n as f32 / 256.0).sin())
            .collect();
        let mut out = vec![0.0f32; 256];
        for _ in 0..64 {
            eng.process_block(&block, &mut out, &params);
        }
        let meters = eng.meters();
        assert!(
            (meters.input_db + 9.03).abs() < 0.5,
            "input meter {} dB",
            meters.input_db
        );
        // All bin operators disabled → no gain reduction.
        assert_eq!(meters.gr_denoise_db, 0.0);
        assert_eq!(meters.gr_gate_db, 0.0);
        assert_eq!(meters.gr_compress_db, 0.0);
        // Output settled to the same level as the (delayed) input.
        assert!(
            (meters.output_db - meters.input_db).abs() < 1.0,
            "in {} out {}",
            meters.input_db,
            meters.output_db
        );
    }

    /// A silent input floors both level meters.
    #[test]
    fn meters_floor_on_silence() {
        let mut eng = engine(512, 128);
        let params = SpectralParams::default();
        let silence = vec![0.0f32; 128];
        let mut out = vec![0.0f32; 128];
        eng.process_block(&silence, &mut out, &params);
        let meters = eng.meters();
        assert_eq!(meters.input_db, METER_FLOOR_DB);
        assert_eq!(meters.output_db, METER_FLOOR_DB);
    }

    /// An engaged gate on a low-level broadband input reports a negative gain
    /// reduction, and the exposed magnitude spectrum is populated.
    #[test]
    fn gate_reports_reduction_and_magnitude_is_exposed() {
        let mut eng = engine(1024, 256);
        let params = SpectralParams {
            gate: GateParams {
                enabled: true,
                threshold_db: -20.0, // above the quiet noise floor → bins gate
                range_db: 48.0,
                attack_ms: 1.0,
                release_ms: 10.0,
            },
            transient: TransientParams {
                enabled: false,
                ..TransientParams::default()
            },
            ..SpectralParams::default()
        };
        // Low-level broadband noise spreads energy across bins, each sitting
        // below the threshold so the expander attenuates them.
        let mut state = 0x9E37_79B9u32;
        let mut noise = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            0.01 * ((state as f32 / u32::MAX as f32) * 2.0 - 1.0)
        };
        let mut out = vec![0.0f32; 256];
        for _ in 0..64 {
            let block: Vec<f32> = (0..256).map(|_| noise()).collect();
            eng.process_block(&block, &mut out, &params);
        }
        assert!(
            eng.meters().gr_gate_db < -0.3,
            "gate GR = {}",
            eng.meters().gr_gate_db
        );
        // Magnitude spectrum has the expected length and holds finite values.
        assert_eq!(eng.spectrum_magnitude().len(), 1024 / 2 + 1);
        assert!(eng.spectrum_magnitude().iter().all(|m| m.is_finite()));
    }
}
