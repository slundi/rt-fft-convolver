//! The reverb processing engine, split across the two plugin threads.
//!
//! ```text
//! control thread              lock-free handoff          audio thread
//! ─────────────────           ─────────────────          ─────────────────
//! ReverbController               IrHandoff                ReverbProcessor
//!   IrSlotBank<PreparedIr>  ── publish(PreparedReverb) ──►  StereoSlotCrossfader
//!   build Stereo/TrueStereo                                 PreDelay ×2
//!   reverse + length trim                                   wet EQ + width
//!                                                           dry/wet faders
//! ```
//!
//! # Wet signal chain (stereo)
//!
//! ```text
//! pd       = predelay(in)                     pre-delay the reverb input
//! wet      = convolve(pd)                      stereo / true-stereo IR
//! wet      = high_cut(low_cut(wet))            wet-only EQ
//! wet      = width(wet)                        stereo-width (M/S)
//! out      = dry_gain·in + wet_gain·wet        independent faders
//! ```
//!
//! An empty / disabled slot drops the wet path (`out = dry_gain·in`).  Host
//! bypass copies the input through untouched.

use std::path::Path;

use rt_fft_convolver::{LoadError, StereoConvolver, TrueStereoConvolver};
use rt_ir_plugin_core::{
    BUTTERWORTH_Q, Biquad, IrConsumer, IrPublisher, IrSlotBank, MidiRouter, PreparedIr, SlotError,
    SlotEvent, ir_handoff, ms_to_samples,
};

use crate::convolver::{ReverbConvolver, StereoSlotCrossfader};
use crate::dsp::{PreDelay, apply_width, fader_gain};
use crate::params::{NUM_SLOTS, PREDELAY_MS_RANGE, ReverbParams};

/// A slot's IR built into a ready-to-run stereo convolver.
pub struct PreparedReverb {
    /// The built convolver, or `None` for a dry (empty / disabled) activation.
    conv: Option<ReverbConvolver>,
    /// `true` when the source slot was empty or disabled → render dry.
    dry: bool,
    /// Source slot index (for read-only reporting).
    slot: usize,
    /// Tail partition count (for the CPU meter).
    partitions: usize,
}

/// Control-thread side: owns the loaded IRs and builds reverbs to hand off.
pub struct ReverbController {
    bank: IrSlotBank<PreparedIr, NUM_SLOTS>,
    slot_enabled: [bool; NUM_SLOTS],
    publisher: IrPublisher<PreparedReverb>,
    block_size: usize,
    sample_rate: u32,
    reverse: bool,
    length_frac: f32,
    router: MidiRouter,
}

impl ReverbController {
    /// Build a controller and its paired [`ReverbProcessor`] (stereo I/O).
    ///
    /// **Allocates** — call during plugin initialisation, off the audio thread.
    pub fn new(
        block_size: usize,
        sample_rate: u32,
        params: &ReverbParams,
    ) -> (Self, ReverbProcessor) {
        assert!(block_size > 0, "block_size must be > 0");
        let (publisher, consumer) = ir_handoff::<PreparedReverb>();
        let controller = Self {
            bank: IrSlotBank::new(),
            slot_enabled: params.slot_enabled,
            publisher,
            block_size,
            sample_rate,
            reverse: params.ir_reverse,
            length_frac: params.ir_length,
            router: MidiRouter::new(NUM_SLOTS),
        };
        let processor = ReverbProcessor::new(consumer, block_size, sample_rate, params);
        (controller, processor)
    }

    /// Store a prepared (resampled) IR in `index`.
    pub fn load_slot(&mut self, index: usize, ir: PreparedIr) -> Result<(), SlotError> {
        self.bank.load(index, ir).map(|_| ())
    }

    /// Load and resample a WAV file into `index` (at the host sample rate).
    ///
    /// **Does file I/O and allocates** — off the audio thread.
    pub fn load_slot_from_path(
        &mut self,
        index: usize,
        path: impl AsRef<Path>,
    ) -> Result<(), LoadError> {
        let ir = PreparedIr::load(path, self.sample_rate)?;
        let _ = self.load_slot(index, ir);
        Ok(())
    }

    /// Empty `index`.
    pub fn clear_slot(&mut self, index: usize) -> Result<(), SlotError> {
        self.bank.clear(index).map(|_| ())
    }

    /// Enable/disable a slot (disabled plays dry even when loaded).
    pub fn set_slot_enabled(&mut self, index: usize, enabled: bool) {
        if index < NUM_SLOTS {
            self.slot_enabled[index] = enabled;
        }
    }

    /// Set IR reversal for subsequently-activated slots.
    pub fn set_reverse(&mut self, reverse: bool) {
        self.reverse = reverse;
    }

    /// Set the IR length fraction (`0.01..=1.0`) for subsequent activations.
    pub fn set_length(&mut self, length_frac: f32) {
        self.length_frac = length_frac.clamp(0.01, 1.0);
    }

    /// Bind (or clear) the MIDI CC used for slot selection.
    pub fn set_midi_cc(&mut self, cc: Option<u8>) {
        self.router.set_cc(cc);
    }

    /// Host sample rate this controller loads IRs at.
    #[inline]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Whether slot `index` holds an IR.
    #[inline]
    pub fn is_loaded(&self, index: usize) -> bool {
        self.bank.is_loaded(index)
    }

    /// Down-sampled waveform preview of slot `index`'s IR (channel 0), or
    /// `None` if the slot is empty.  For the editor; **allocates** (off-RT).
    pub fn waveform(
        &self,
        index: usize,
        buckets: usize,
    ) -> Option<rt_ir_plugin_core::WaveformSummary> {
        let ir = self.bank.get(index)?;
        let ch = ir.channel(0)?;
        Some(rt_ir_plugin_core::compute_waveform(ch, buckets))
    }

    /// Route a MIDI event and activate the resulting slot, if any.
    pub fn handle_midi(&mut self, event: SlotEvent) -> Option<usize> {
        let slot = self.router.route(event)?;
        self.activate(slot);
        Some(slot)
    }

    /// Build the reverb for `slot` and publish it to the audio thread.
    ///
    /// **Allocates** — off the audio thread.
    pub fn activate(&mut self, slot: usize) {
        let slot = slot.min(NUM_SLOTS - 1);
        let prepared = self.build(slot);
        self.publisher.publish(prepared);
    }

    /// Build a [`PreparedReverb`] from the given slot.
    fn build(&self, slot: usize) -> PreparedReverb {
        let usable = self.slot_enabled[slot]
            && self
                .bank
                .get(slot)
                .is_some_and(|ir| !ir.is_empty() && ir.num_channels() > 0);

        if !usable {
            return PreparedReverb {
                conv: None,
                dry: true,
                slot,
                partitions: 0,
            };
        }

        let ir = self.bank.get(slot).expect("checked usable");
        let (conv, tail_len) = self.build_convolver(ir);
        // Uniform partitioning splits the tail into `ceil(len / block)` blocks.
        let partitions = tail_len.div_ceil(self.block_size);
        PreparedReverb {
            conv: Some(conv),
            dry: false,
            slot,
            partitions,
        }
    }

    /// Build the stereo/true-stereo convolver, applying reverse + length trim.
    /// Returns the convolver and the (post-trim) IR length in samples.
    fn build_convolver(&self, ir: &PreparedIr) -> (ReverbConvolver, usize) {
        let ch = |i: usize| self.transform(ir.channel(i).unwrap_or(&[]));
        match ir.num_channels() {
            // Mono IR → duplicated to both channels (preserves the input field).
            1 => {
                let c0 = ch(0);
                let len = c0.len();
                (
                    ReverbConvolver::Stereo(StereoConvolver::new_mono_ir(&c0, self.block_size)),
                    len,
                )
            }
            // 4+ channels → true stereo (L→L, L→R, R→L, R→R).
            n if n >= 4 => {
                let (ll, lr, rl, rr) = (ch(0), ch(1), ch(2), ch(3));
                let len = ll.len();
                (
                    ReverbConvolver::True(TrueStereoConvolver::new(
                        &ll,
                        &lr,
                        &rl,
                        &rr,
                        self.block_size,
                    )),
                    len,
                )
            }
            // 2 or 3 channels → plain stereo (independent L/R IRs).
            _ => {
                let (l, r) = (ch(0), ch(1));
                let len = l.len();
                (
                    ReverbConvolver::Stereo(StereoConvolver::new(&l, &r, self.block_size)),
                    len,
                )
            }
        }
    }

    /// Apply the IR-length trim and optional reversal to one channel.
    fn transform(&self, src: &[f32]) -> Vec<f32> {
        if src.is_empty() {
            return Vec::new();
        }
        let keep = ((src.len() as f32) * self.length_frac).round() as usize;
        let keep = keep.clamp(1, src.len());
        let mut out = src[..keep].to_vec();
        if self.reverse {
            out.reverse();
        }
        out
    }
}

/// Audio-thread side: renders the reverb and consumes activations.
pub struct ReverbProcessor {
    consumer: IrConsumer<PreparedReverb>,
    crossfader: StereoSlotCrossfader<ReverbConvolver>,
    block_size: usize,
    sample_rate: u32,

    predelay_l: PreDelay,
    predelay_r: PreDelay,
    low_cut_l: Biquad,
    low_cut_r: Biquad,
    high_cut_l: Biquad,
    high_cut_r: Biquad,

    // Cached global parameter values.
    dry_gain: f32,
    wet_gain: f32,
    width: f32,
    bypass: bool,
    crossfade_samples: usize,
    wet_enabled: bool,

    // Scratch (all `block_size`).
    pd_l: Vec<f32>,
    pd_r: Vec<f32>,
    wet_l: Vec<f32>,
    wet_r: Vec<f32>,

    // Reporting.
    active_slot: usize,
    num_partitions: usize,

    // Retired convolvers awaiting off-thread reclaim.
    retired: Vec<ReverbConvolver>,
}

impl ReverbProcessor {
    fn new(
        consumer: IrConsumer<PreparedReverb>,
        block_size: usize,
        sample_rate: u32,
        params: &ReverbParams,
    ) -> Self {
        let max_predelay = ms_to_samples(PREDELAY_MS_RANGE.1, sample_rate);
        let mut me =
            Self {
                consumer,
                // Start on a silent stereo convolver; the first activation fades in.
                crossfader: StereoSlotCrossfader::new(ReverbConvolver::Stereo(
                    StereoConvolver::new(&[], &[], block_size),
                )),
                block_size,
                sample_rate,
                predelay_l: PreDelay::new(max_predelay),
                predelay_r: PreDelay::new(max_predelay),
                low_cut_l: Biquad::high_pass(sample_rate, params.low_cut_hz, BUTTERWORTH_Q),
                low_cut_r: Biquad::high_pass(sample_rate, params.low_cut_hz, BUTTERWORTH_Q),
                high_cut_l: Biquad::low_pass(sample_rate, params.high_cut_hz, BUTTERWORTH_Q),
                high_cut_r: Biquad::low_pass(sample_rate, params.high_cut_hz, BUTTERWORTH_Q),
                dry_gain: 1.0,
                wet_gain: 1.0,
                width: params.width,
                bypass: params.bypass,
                crossfade_samples: 0,
                wet_enabled: false,
                pd_l: vec![0.0; block_size],
                pd_r: vec![0.0; block_size],
                wet_l: vec![0.0; block_size],
                wet_r: vec![0.0; block_size],
                active_slot: params.current_slot,
                num_partitions: 0,
                retired: Vec::with_capacity(4),
            };
        me.apply_params(params);
        me
    }

    /// Fixed render block size.
    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Slot index currently rendering.
    #[inline]
    pub fn active_slot(&self) -> usize {
        self.active_slot
    }

    /// Whether the crossfader is mid-fade.
    #[inline]
    pub fn is_crossfading(&self) -> bool {
        self.crossfader.is_fading()
    }

    /// CPU-load proxy: tail partitions × block size (roadmap CPU meter).
    #[inline]
    pub fn cpu_estimate(&self) -> usize {
        self.num_partitions * self.block_size
    }

    /// Update cached globals and per-channel filters/pre-delay.
    ///
    /// **Real-time safe** — no allocation.
    pub fn apply_params(&mut self, params: &ReverbParams) {
        self.dry_gain = fader_gain(params.dry_db);
        self.wet_gain = fader_gain(params.wet_db);
        self.width = params.width.clamp(0.0, 2.0);
        self.bypass = params.bypass;
        self.crossfade_samples = ms_to_samples(params.crossfade_ms, self.sample_rate);

        let pd = ms_to_samples(params.predelay_ms, self.sample_rate);
        self.predelay_l.set_delay(pd);
        self.predelay_r.set_delay(pd);

        self.low_cut_l
            .set_high_pass(self.sample_rate, params.low_cut_hz, BUTTERWORTH_Q);
        self.low_cut_r
            .set_high_pass(self.sample_rate, params.low_cut_hz, BUTTERWORTH_Q);
        self.high_cut_l
            .set_low_pass(self.sample_rate, params.high_cut_hz, BUTTERWORTH_Q);
        self.high_cut_r
            .set_low_pass(self.sample_rate, params.high_cut_hz, BUTTERWORTH_Q);
    }

    /// Reclaim retired convolvers off the audio thread; returns the count.
    pub fn drain_retired(&mut self) -> usize {
        let n = self.retired.len();
        self.retired.clear();
        n
    }

    /// Render one stereo block.  All four slices must be `block_size` long.
    ///
    /// **Real-time safe** — no allocation, no locking.
    pub fn process(&mut self, in_l: &[f32], in_r: &[f32], out_l: &mut [f32], out_r: &mut [f32]) {
        self.poll_activation();

        if self.bypass {
            out_l.copy_from_slice(in_l);
            out_r.copy_from_slice(in_r);
            return;
        }

        let n = self.block_size;
        // Pre-delay the wet input (dry uses the original signal).
        for i in 0..n {
            self.pd_l[i] = self.predelay_l.process_sample(in_l[i]);
            self.pd_r[i] = self.predelay_r.process_sample(in_r[i]);
        }

        let (dry_gain, wet_gain, width) = (self.dry_gain, self.wet_gain, self.width);

        if !self.wet_enabled {
            // Dry-only (empty / disabled slot): never silent.
            for i in 0..n {
                out_l[i] = dry_gain * in_l[i];
                out_r[i] = dry_gain * in_r[i];
            }
            return;
        }

        // Wet path: convolve → EQ → width.
        self.crossfader
            .process(&self.pd_l, &self.pd_r, &mut self.wet_l, &mut self.wet_r);
        self.low_cut_l.process_block(&mut self.wet_l);
        self.high_cut_l.process_block(&mut self.wet_l);
        self.low_cut_r.process_block(&mut self.wet_r);
        self.high_cut_r.process_block(&mut self.wet_r);
        apply_width(&mut self.wet_l, &mut self.wet_r, width);

        // Independent dry + wet faders.
        for i in 0..n {
            out_l[i] = dry_gain * in_l[i] + wet_gain * self.wet_l[i];
            out_r[i] = dry_gain * in_r[i] + wet_gain * self.wet_r[i];
        }
    }

    /// Pull a freshly-built reverb (if any) and crossfade to it.
    fn poll_activation(&mut self) {
        let activation = {
            let (payload, changed) = self.consumer.poll_changed();
            match (changed, payload) {
                (true, Some(p)) => Some((p.conv.take(), p.dry, p.slot, p.partitions)),
                _ => None,
            }
        };

        let Some((conv, dry, slot, partitions)) = activation else {
            return;
        };
        self.active_slot = slot;
        self.num_partitions = partitions;

        match conv {
            Some(c) => {
                if let Some(old) = self.crossfader.start_fade(c, self.crossfade_samples) {
                    self.retired.push(old);
                }
                self.wet_enabled = !dry;
            }
            // Dry activation: drop the wet path (instant; empty slots are rare).
            None => self.wet_enabled = false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK: usize = 64;
    const SR: u32 = 48_000;

    /// Stereo impulse IR that scales L by `gl` and R by `gr`.
    fn stereo_impulse(gl: f32, gr: f32) -> PreparedIr {
        let mut l = vec![0.0f32; BLOCK];
        l[0] = gl;
        let mut r = vec![0.0f32; BLOCK];
        r[0] = gr;
        PreparedIr::from_channels(vec![l, r], SR)
    }

    /// Neutral params: instant switch, EQ wide open, width 1, dry 0 / wet 0 dB.
    fn flat_params() -> ReverbParams {
        ReverbParams {
            crossfade_ms: 0.0,
            low_cut_hz: 20.0,
            high_cut_hz: 20_000.0,
            ..ReverbParams::default()
        }
    }

    fn run(proc: &mut ReverbProcessor, in_l: &[f32], in_r: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut out_l = vec![0.0; BLOCK];
        let mut out_r = vec![0.0; BLOCK];
        proc.process(in_l, in_r, &mut out_l, &mut out_r);
        (out_l, out_r)
    }

    #[test]
    fn empty_slot_is_dry_not_silent() {
        let params = flat_params();
        let (_c, mut proc) = ReverbController::new(BLOCK, SR, &params);
        let mut in_l = vec![0.0; BLOCK];
        in_l[0] = 0.5;
        let mut in_r = vec![0.0; BLOCK];
        in_r[0] = -0.25;
        let (l, r) = run(&mut proc, &in_l, &in_r);
        // dry 0 dB, wet disabled → out == in.
        assert!((l[0] - 0.5).abs() < 1e-6);
        assert!((r[0] + 0.25).abs() < 1e-6);
    }

    #[test]
    fn bypass_copies_input() {
        let params = ReverbParams {
            bypass: true,
            wet_db: 0.0,
            ..flat_params()
        };
        let (_c, mut proc) = ReverbController::new(BLOCK, SR, &params);
        let mut in_l = vec![0.0; BLOCK];
        in_l[3] = 0.8;
        let (l, _) = run(&mut proc, &in_l, &vec![0.0; BLOCK]);
        assert_eq!(l[3], 0.8);
    }

    #[test]
    fn wet_scales_linearly_with_ir_gain() {
        // Two setups differing only in IR gain (×1 vs ×2): the linear chain
        // makes the louder reverb's output exactly 2× the other's.
        let params = ReverbParams {
            dry_db: -60.0, // mute dry so we measure only the wet path
            ..flat_params()
        };

        let (mut c1, mut p1) = ReverbController::new(BLOCK, SR, &params);
        c1.load_slot(0, stereo_impulse(1.0, 1.0)).unwrap();
        c1.activate(0);

        let (mut c2, mut p2) = ReverbController::new(BLOCK, SR, &params);
        c2.load_slot(0, stereo_impulse(2.0, 2.0)).unwrap();
        c2.activate(0);

        let mut input = vec![0.0; BLOCK];
        input[0] = 0.5;
        let (a, _) = run(&mut p1, &input, &input);
        let (b, _) = run(&mut p2, &input, &input);

        assert!(a.iter().any(|&s| s.abs() > 1e-4), "wet path was silent");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((y - 2.0 * x).abs() < 1e-4, "sample {i}: {y} != 2·{x}");
        }
    }

    #[test]
    fn dry_muted_wet_unity_passes_reverb_only() {
        let params = ReverbParams {
            dry_db: -60.0,
            wet_db: 0.0,
            ..flat_params()
        };
        let (mut c, mut proc) = ReverbController::new(BLOCK, SR, &params);
        // Unit stereo impulse → wet == input (through the wide-open EQ).
        c.load_slot(0, stereo_impulse(1.0, 1.0)).unwrap();
        c.activate(0);

        let mut input = vec![0.0; BLOCK];
        input[0] = 0.4;
        let (l, _) = run(&mut proc, &input, &input);
        assert_eq!(proc.active_slot(), 0);
        assert!(l.iter().any(|&s| s.abs() > 1e-4));
    }

    #[test]
    fn true_stereo_ir_uses_cross_feed() {
        // 4-channel IR with only the L→R path non-zero: a left-only input must
        // produce output on the right.
        let params = ReverbParams {
            dry_db: -60.0,
            ..flat_params()
        };
        let (mut c, mut proc) = ReverbController::new(BLOCK, SR, &params);
        let z = || vec![0.0f32; BLOCK];
        let mut lr = z();
        lr[0] = 1.0; // L→R impulse
        c.load_slot(0, PreparedIr::from_channels(vec![z(), lr, z(), z()], SR))
            .unwrap();
        c.activate(0);

        let mut in_l = z();
        in_l[0] = 0.5;
        let (out_l, out_r) = run(&mut proc, &in_l, &z());
        // L→L, R→L, R→R are silent → left output ~0; L→R carries the signal.
        assert!(
            out_l.iter().all(|&s| s.abs() < 1e-4),
            "left should be silent"
        );
        assert!(
            out_r.iter().any(|&s| s.abs() > 1e-4),
            "right should have cross-feed"
        );
    }

    #[test]
    fn predelay_shifts_the_wet_onset() {
        let params = ReverbParams {
            dry_db: -60.0,
            predelay_ms: 1.0, // 48 samples at 48 kHz
            ..flat_params()
        };
        let (mut c, mut proc) = ReverbController::new(BLOCK, SR, &params);
        c.load_slot(0, stereo_impulse(1.0, 1.0)).unwrap();
        c.activate(0);

        let mut input = vec![0.0; BLOCK];
        input[0] = 1.0;
        let (l, _) = run(&mut proc, &input, &input);
        // With 48-sample pre-delay the impulse response starts ~sample 48.
        assert!(
            l[..48].iter().all(|&s| s.abs() < 1e-4),
            "onset before pre-delay"
        );
        assert!(l[48].abs() > 1e-4, "expected onset at the pre-delay point");
    }

    #[test]
    fn midi_program_change_switches_slot() {
        let params = flat_params();
        let (mut c, mut proc) = ReverbController::new(BLOCK, SR, &params);
        c.load_slot(3, stereo_impulse(1.0, 1.0)).unwrap();
        assert_eq!(c.handle_midi(SlotEvent::ProgramChange(3)), Some(3));
        run(&mut proc, &vec![0.0; BLOCK], &vec![0.0; BLOCK]);
        assert_eq!(proc.active_slot(), 3);
    }
}
