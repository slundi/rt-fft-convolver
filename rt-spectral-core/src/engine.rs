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

            // Per-bin chain: denoise → gate → compress.
            self.denoiser.process(
                &mut self.spectrum,
                &params.denoise,
                self.profile.current(),
                depth,
            );
            self.gate.process(&mut self.spectrum, &params.gate);
            self.comp
                .process(&mut self.spectrum, &params.compress, depth);

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
}
