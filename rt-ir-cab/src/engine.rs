//! The cabinet processing engine, split across the two plugin threads.
//!
//! ```text
//! control thread            lock-free handoff            audio thread
//! ─────────────────         ─────────────────            ─────────────────
//! CabController                 IrHandoff                 CabProcessor
//!   IrSlotBank<PreparedIr>  ── publish(PreparedCab) ──►     SlotCrossfader × 2
//!   build HybridConvolvers                                  Biquad low/high cut
//!   (resample already done)                                 gain · mix · bypass
//! ```
//!
//! - [`CabController`] lives off the audio thread.  It owns the loaded IRs and,
//!   on activation, builds the (allocating) [`HybridConvolver`]s for the slot
//!   and publishes them as a [`PreparedCab`].
//! - [`CabProcessor`] lives on the audio thread.  Each block it polls the
//!   handoff, crossfades to any freshly-built cab, and renders the signal
//!   chain.  It never allocates: the incoming convolvers were built by the
//!   controller and the retired ones are parked in a pre-sized `retired`
//!   buffer for the controller to reclaim — never dropped on the audio thread.
//!
//! # Signal chain (per output channel, cab active)
//!
//! ```text
//! xg  = in · input_gain                 (dry reference)
//! hp  = low_cut(xg)                      high-pass, pre-convolution
//! wet = high_cut( convolve(hp) )         cab + low-pass, post-convolution
//! out = ((1−mix)·xg + mix·wet) · output_gain
//! ```
//!
//! With an empty / disabled slot the wet path is skipped and the channel
//! passes `in · input_gain · output_gain` — **dry, never silent**.  Under host
//! bypass the input is copied through untouched.

use std::path::Path;

use rt_fft_convolver::{HybridConvolver, LoadError, Mixer};
use rt_ir_plugin_core::{
    IrConsumer, IrPublisher, IrSlotBank, MidiRouter, PreparedIr, SlotError, SlotEvent, ir_handoff,
    ms_to_samples,
};

use crate::dsp::{BUTTERWORTH_Q, Biquad, db_to_gain};
use crate::params::{CabParams, NUM_SLOTS};

/// A slot's IRs built into ready-to-run convolvers, one per output channel.
///
/// Produced off the audio thread by [`CabController`] and handed to
/// [`CabProcessor`] through the lock-free handoff.
pub struct PreparedCab {
    /// One convolver per output channel (moved into the crossfaders on arrival).
    channels: Vec<HybridConvolver>,
    /// `true` when the source slot was empty or disabled → render dry.
    dry: bool,
    /// Source slot index (for the processor's read-only reporting).
    slot: usize,
}

/// Control-thread side: owns the loaded IRs and builds cabs to hand off.
pub struct CabController {
    bank: IrSlotBank<PreparedIr, NUM_SLOTS>,
    slot_enabled: [bool; NUM_SLOTS],
    publisher: IrPublisher<PreparedCab>,
    block_size: usize,
    sample_rate: u32,
    num_channels: usize,
    normalize: bool,
    /// Maps MIDI Program Change / CC to a slot index.
    router: MidiRouter,
}

impl CabController {
    /// Build a controller and its paired [`CabProcessor`].
    ///
    /// - `block_size` — the fixed render block size the convolvers use.
    /// - `sample_rate` — host sample rate (for crossfade-time conversion).
    /// - `num_channels` — number of **output** channels (2 for the cab).
    ///
    /// **Allocates** — call during plugin initialisation, off the audio thread.
    pub fn new(
        block_size: usize,
        sample_rate: u32,
        num_channels: usize,
        params: &CabParams,
    ) -> (Self, CabProcessor) {
        assert!(block_size > 0, "block_size must be > 0");
        assert!(num_channels > 0, "num_channels must be > 0");

        let (publisher, consumer) = ir_handoff::<PreparedCab>();
        let controller = Self {
            bank: IrSlotBank::new(),
            slot_enabled: params.slot_enabled,
            publisher,
            block_size,
            sample_rate,
            num_channels,
            normalize: params.ir_normalize,
            router: MidiRouter::new(NUM_SLOTS),
        };
        let processor = CabProcessor::new(consumer, block_size, sample_rate, num_channels, params);
        (controller, processor)
    }

    /// Store a prepared (resampled) IR in `index`.
    ///
    /// # Errors
    ///
    /// [`SlotError::IndexOutOfRange`] if `index >= NUM_SLOTS`.
    pub fn load_slot(&mut self, index: usize, ir: PreparedIr) -> Result<(), SlotError> {
        self.bank.load(index, ir).map(|_| ())
    }

    /// Load and resample a WAV file into `index` (at the host sample rate).
    ///
    /// Out-of-range indices are ignored.  **Does file I/O and allocates** —
    /// off the audio thread (e.g. a `nih-plug` background task).
    ///
    /// # Errors
    ///
    /// Returns [`LoadError`] if the file cannot be read, decoded, or resampled.
    pub fn load_slot_from_path(
        &mut self,
        index: usize,
        path: impl AsRef<Path>,
    ) -> Result<(), LoadError> {
        let ir = PreparedIr::load(path, self.sample_rate)?;
        let _ = self.load_slot(index, ir); // range already implied by the caller
        Ok(())
    }

    /// Host sample rate this controller loads IRs at.
    #[inline]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Empty `index` (does not change what is currently playing).
    ///
    /// # Errors
    ///
    /// [`SlotError::IndexOutOfRange`] if `index >= NUM_SLOTS`.
    pub fn clear_slot(&mut self, index: usize) -> Result<(), SlotError> {
        self.bank.clear(index).map(|_| ())
    }

    /// Enable or disable a slot.  A disabled slot plays dry even when loaded.
    pub fn set_slot_enabled(&mut self, index: usize, enabled: bool) {
        if index < NUM_SLOTS {
            self.slot_enabled[index] = enabled;
        }
    }

    /// Toggle IR normalisation for **subsequently** activated slots.
    pub fn set_normalize(&mut self, normalize: bool) {
        self.normalize = normalize;
    }

    /// Bind (or clear) the MIDI CC used for slot selection.
    pub fn set_midi_cc(&mut self, cc: Option<u8>) {
        self.router.set_cc(cc);
    }

    /// Route a MIDI event to a slot and, if it selects one, activate it.
    ///
    /// Returns the activated slot index, or `None` if the event maps nowhere.
    pub fn handle_midi(&mut self, event: SlotEvent) -> Option<usize> {
        let slot = self.router.route(event)?;
        self.activate(slot);
        Some(slot)
    }

    /// Build the cab for `slot` and publish it to the audio thread.
    ///
    /// An empty or disabled slot publishes a "dry" activation so the audio
    /// thread stops the wet path without silence.  The crossfade *duration* is
    /// applied processor-side (from the cached `crossfade_ms` parameter).
    ///
    /// **Allocates** (builds the convolvers) — off the audio thread.
    pub fn activate(&mut self, slot: usize) {
        let slot = slot.min(NUM_SLOTS - 1);
        let prepared = self.build(slot);
        self.publisher.publish(prepared);
    }

    /// Build a [`PreparedCab`] from the given slot.
    fn build(&self, slot: usize) -> PreparedCab {
        let usable = self.slot_enabled[slot]
            && self
                .bank
                .get(slot)
                .is_some_and(|ir| !ir.is_empty() && ir.num_channels() > 0);

        let mut channels = Vec::with_capacity(self.num_channels);
        if usable {
            let ir = self.bank.get(slot).expect("checked usable");
            for c in 0..self.num_channels {
                // Mono IR → same response on every output; multi-channel IR →
                // map output channel c onto IR channel c (wrapping).
                let src = ir
                    .channel(c % ir.num_channels())
                    .expect("channel index wrapped in range");
                channels.push(self.build_convolver(src));
            }
            PreparedCab {
                channels,
                dry: false,
                slot,
            }
        } else {
            // Silence convolvers keep block sizes consistent; `dry` tells the
            // processor to ignore them and pass audio through.
            for _ in 0..self.num_channels {
                channels.push(HybridConvolver::new(&[], self.block_size));
            }
            PreparedCab {
                channels,
                dry: true,
                slot,
            }
        }
    }

    /// Build one channel's convolver, applying IR normalisation if enabled.
    fn build_convolver(&self, ir: &[f32]) -> HybridConvolver {
        if self.normalize {
            // `Mixer` derives 1/‖ir‖₂; pre-scale the IR so the convolution
            // output is loudness-normalised.
            let gain = Mixer::new(ir, 1.0).ir_gain();
            let scaled: Vec<f32> = ir.iter().map(|&x| x * gain).collect();
            HybridConvolver::new(&scaled, self.block_size)
        } else {
            HybridConvolver::new(ir, self.block_size)
        }
    }

    /// Reclaim convolvers retired by the processor so they are dropped here,
    /// off the audio thread.  Call periodically from the control thread.
    pub fn reclaim(&mut self, processor: &mut CabProcessor) {
        processor.drain_retired();
    }
}

/// One output channel's audio-thread state and scratch buffers.
struct CabChannel {
    xfade: rt_ir_plugin_core::SlotCrossfader<HybridConvolver>,
    low_cut: Biquad,
    high_cut: Biquad,
    /// Whether the wet (cab) path is active; `false` = dry passthrough.
    wet_enabled: bool,
    /// `input · input_gain` (dry reference).
    xg: Vec<f32>,
    /// High-passed input feeding the convolver.
    hp: Vec<f32>,
    /// Convolver output, then high-cut filtered.
    wet: Vec<f32>,
}

impl CabChannel {
    fn new(block_size: usize, sample_rate: u32, params: &CabParams) -> Self {
        Self {
            // Start on a silent convolver; the first activation fades in.
            xfade: rt_ir_plugin_core::SlotCrossfader::new(HybridConvolver::new(&[], block_size)),
            low_cut: Biquad::high_pass(sample_rate, params.low_cut_hz, BUTTERWORTH_Q),
            high_cut: Biquad::low_pass(sample_rate, params.high_cut_hz, BUTTERWORTH_Q),
            wet_enabled: false,
            xg: vec![0.0; block_size],
            hp: vec![0.0; block_size],
            wet: vec![0.0; block_size],
        }
    }
}

/// Audio-thread side: renders the cab and consumes activations.
pub struct CabProcessor {
    consumer: IrConsumer<PreparedCab>,
    channels: Vec<CabChannel>,
    block_size: usize,
    sample_rate: u32,

    // Cached, audio-thread copies of the global parameters.
    input_gain: f32,
    output_gain: f32,
    mix: f32,
    bypass: bool,
    crossfade_samples: usize,

    // Read-only reporting.
    active_slot: usize,
    /// Tail partition count of the active cab (for the CPU meter).
    num_partitions: usize,

    /// Convolvers retired by crossfades, awaiting off-thread reclaim.  Pre-sized
    /// so `process()` does not allocate under normal switching.
    retired: Vec<HybridConvolver>,
}

impl CabProcessor {
    fn new(
        consumer: IrConsumer<PreparedCab>,
        block_size: usize,
        sample_rate: u32,
        num_channels: usize,
        params: &CabParams,
    ) -> Self {
        let channels = (0..num_channels)
            .map(|_| CabChannel::new(block_size, sample_rate, params))
            .collect();
        let mut me = Self {
            consumer,
            channels,
            block_size,
            sample_rate,
            input_gain: 1.0,
            output_gain: 1.0,
            mix: params.mix,
            bypass: params.bypass,
            crossfade_samples: ms_to_samples(params.crossfade_ms, sample_rate),
            active_slot: params.current_slot,
            num_partitions: 0,
            // Headroom for a few in-flight fades per channel before a push
            // would allocate; the controller drains this off-thread.
            retired: Vec::with_capacity(num_channels * 4),
        };
        me.apply_params(params);
        me
    }

    /// Update the cached global parameters and per-channel filter cutoffs.
    ///
    /// Recomputes biquad coefficients in place (state-preserving, click-free).
    /// **Real-time safe** — no allocation.
    pub fn apply_params(&mut self, params: &CabParams) {
        self.input_gain = db_to_gain(params.input_gain_db);
        self.output_gain = db_to_gain(params.output_gain_db);
        self.mix = params.mix.clamp(0.0, 1.0);
        self.bypass = params.bypass;
        self.crossfade_samples = ms_to_samples(params.crossfade_ms, self.sample_rate);
        for ch in &mut self.channels {
            ch.low_cut
                .set_high_pass(self.sample_rate, params.low_cut_hz, BUTTERWORTH_Q);
            ch.high_cut
                .set_low_pass(self.sample_rate, params.high_cut_hz, BUTTERWORTH_Q);
        }
    }

    /// Fixed render block size the processor and its convolvers require.
    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Slot index the processor is currently rendering.
    #[inline]
    pub fn active_slot(&self) -> usize {
        self.active_slot
    }

    /// Whether any channel is mid-crossfade.
    #[inline]
    pub fn is_crossfading(&self) -> bool {
        self.channels.iter().any(|c| c.xfade.is_fading())
    }

    /// Rough CPU-load proxy: tail partitions × block size (roadmap CPU meter).
    #[inline]
    pub fn cpu_estimate(&self) -> usize {
        self.num_partitions * self.block_size
    }

    /// Take ownership of retired convolvers so the caller can drop them
    /// off the audio thread.  Returns how many were reclaimed.
    fn drain_retired(&mut self) -> usize {
        let n = self.retired.len();
        self.retired.clear();
        n
    }

    /// Render one block.
    ///
    /// - `inputs` — one slice per input channel (mono = 1, stereo = 2).
    /// - `outputs` — one mutable slice per output channel.
    ///
    /// Every slice must be exactly `block_size` long (the convolvers require a
    /// fixed block).  **Real-time safe** — no allocation, no locking.
    pub fn process(&mut self, inputs: &[&[f32]], outputs: &mut [&mut [f32]]) {
        debug_assert!(!inputs.is_empty(), "need at least one input channel");

        // Consume a freshly-built cab if one arrived since the last block.
        self.poll_activation();

        let n = self.block_size;
        let (ig, og, mix, bypass) = (self.input_gain, self.output_gain, self.mix, self.bypass);

        for (c, out) in outputs.iter_mut().enumerate() {
            let out = &mut out[..n];
            // Mono input feeds every output; stereo input maps 1:1.
            let input = &inputs[c.min(inputs.len() - 1)][..n];

            if bypass {
                out.copy_from_slice(input);
                continue;
            }

            let ch = &mut self.channels[c];

            // Input trim → dry reference.
            for (x, &i) in ch.xg.iter_mut().zip(input.iter()) {
                *x = i * ig;
            }

            if !ch.wet_enabled {
                // Empty / disabled slot: dry passthrough with the gain trims.
                for (o, &x) in out.iter_mut().zip(ch.xg.iter()) {
                    *o = x * og;
                }
                continue;
            }

            // hp = low-cut(dry); wet = high-cut(convolve(hp)).
            ch.hp.copy_from_slice(&ch.xg);
            ch.low_cut.process_block(&mut ch.hp);
            ch.xfade.process(&ch.hp, &mut ch.wet);
            ch.high_cut.process_block(&mut ch.wet);

            // Dry/wet blend, then output trim.
            for ((o, &dry), &wet) in out.iter_mut().zip(ch.xg.iter()).zip(ch.wet.iter()) {
                *o = ((1.0 - mix) * dry + mix * wet) * og;
            }
        }
    }

    /// Pull a new [`PreparedCab`] from the handoff (if any) and crossfade to it.
    fn poll_activation(&mut self) {
        // Extract the payload (moving its convolvers out) while the handoff is
        // borrowed, then release the borrow before touching the channels.
        let activation = {
            let (payload, changed) = self.consumer.poll_changed();
            match (changed, payload) {
                (true, Some(p)) => Some((std::mem::take(&mut p.channels), p.dry, p.slot)),
                _ => None,
            }
        };

        let Some((incoming, dry, slot)) = activation else {
            return;
        };

        self.active_slot = slot;
        let cross = self.crossfade_samples;
        // Report the incoming cab's tail size before the convolvers move.
        self.num_partitions = incoming.first().map_or(0, HybridConvolver::num_partitions);

        for (i, conv) in incoming.into_iter().enumerate() {
            if let Some(ch) = self.channels.get_mut(i) {
                if let Some(old) = ch.xfade.start_fade(conv, cross) {
                    // Park the retired convolver; the controller drops it later.
                    self.retired.push(old);
                }
                ch.wet_enabled = !dry;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK: usize = 64;
    const SR: u32 = 48_000;

    /// A cab IR that is a scaled unit impulse: convolving with it just scales
    /// the input by `gain` (a clean way to assert the signal chain).
    fn impulse_ir(gain: f32) -> PreparedIr {
        let mut ir = vec![0.0f32; BLOCK];
        ir[0] = gain;
        PreparedIr::from_channels(vec![ir], SR)
    }

    /// Neutral params: no EQ effect, unity gains, instant switch, full wet.
    fn flat_params() -> CabParams {
        CabParams {
            crossfade_ms: 0.0, // instant switch → deterministic single block
            low_cut_hz: 20.0,
            high_cut_hz: 20_000.0,
            ..CabParams::default()
        }
    }

    fn run_block(proc: &mut CabProcessor, input: &[f32]) -> Vec<f32> {
        let mut out_l = vec![0.0f32; BLOCK];
        let mut out_r = vec![0.0f32; BLOCK];
        {
            let ins: [&[f32]; 1] = [input];
            let mut outs: [&mut [f32]; 2] = [&mut out_l, &mut out_r];
            proc.process(&ins, &mut outs);
        }
        out_l
    }

    #[test]
    fn empty_slot_passes_dry_not_silent() {
        let params = flat_params();
        let (_controller, mut proc) = CabController::new(BLOCK, SR, 2, &params);

        // No slot ever loaded/activated → wet path disabled → dry passthrough.
        let mut input = vec![0.0f32; BLOCK];
        input[0] = 0.5;
        input[1] = -0.25;
        let out = run_block(&mut proc, &input);
        assert_eq!(out[0], 0.5);
        assert_eq!(out[1], -0.25);
    }

    #[test]
    fn bypass_copies_input_untouched() {
        let params = CabParams {
            bypass: true,
            input_gain_db: 12.0, // would change the signal if not bypassed
            ..flat_params()
        };
        let (_controller, mut proc) = CabController::new(BLOCK, SR, 2, &params);
        let mut input = vec![0.0f32; BLOCK];
        input[3] = 0.8;
        let out = run_block(&mut proc, &input);
        assert_eq!(out[3], 0.8);
    }

    #[test]
    fn wet_output_scales_linearly_with_ir_gain() {
        // Two setups identical but for IR gain (×1 vs ×2).  The whole chain is
        // linear, so the louder cab's output must be exactly 2× the other's,
        // sample-for-sample — independent of the (always-on) EQ filters.  This
        // is the robust way to prove the wet convolution path really runs and
        // the IR gain propagates.
        let params = flat_params();

        let (mut c1, mut p1) = CabController::new(BLOCK, SR, 2, &params);
        c1.load_slot(0, impulse_ir(1.0)).unwrap();
        c1.activate(0);

        let (mut c2, mut p2) = CabController::new(BLOCK, SR, 2, &params);
        c2.load_slot(0, impulse_ir(2.0)).unwrap();
        c2.activate(0);

        let mut input = vec![0.0f32; BLOCK];
        input[0] = 0.5;
        let a = run_block(&mut p1, &input);
        let b = run_block(&mut p2, &input);

        assert_eq!(p1.active_slot(), 0);
        // Output is non-trivial (the wet path actually produced signal).
        assert!(a.iter().any(|&s| s.abs() > 1e-4), "wet output was silent");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((y - 2.0 * x).abs() < 1e-4, "sample {i}: {y} != 2·{x}");
        }
    }

    #[test]
    fn mix_zero_is_fully_dry() {
        let params = CabParams {
            mix: 0.0,
            ..flat_params()
        };
        let (mut controller, mut proc) = CabController::new(BLOCK, SR, 2, &params);
        controller.load_slot(0, impulse_ir(10.0)).unwrap(); // loud cab
        controller.activate(0);

        let mut input = vec![0.0f32; BLOCK];
        input[0] = 0.3;
        let out = run_block(&mut proc, &input);
        // mix = 0 → wet ignored → pure dry (== input at unity gain).
        assert!((out[0] - 0.3).abs() < 1e-4, "out[0] = {}", out[0]);
    }

    #[test]
    fn output_gain_scales_dry_passthrough() {
        let params = CabParams {
            output_gain_db: 6.0206, // ×2
            ..flat_params()
        };
        let (_controller, mut proc) = CabController::new(BLOCK, SR, 2, &params);
        let mut input = vec![0.0f32; BLOCK];
        input[0] = 0.25;
        let out = run_block(&mut proc, &input);
        assert!((out[0] - 0.5).abs() < 1e-3, "out[0] = {}", out[0]);
    }

    #[test]
    fn disabled_slot_plays_dry_even_when_loaded() {
        let params = flat_params();
        let (mut controller, mut proc) = CabController::new(BLOCK, SR, 2, &params);
        controller.load_slot(0, impulse_ir(4.0)).unwrap();
        controller.set_slot_enabled(0, false);
        controller.activate(0);

        let mut input = vec![0.0f32; BLOCK];
        input[0] = 0.5;
        let out = run_block(&mut proc, &input);
        // Disabled → dry, so the loud cab is not applied.
        assert!((out[0] - 0.5).abs() < 1e-4, "out[0] = {}", out[0]);
    }

    #[test]
    fn midi_program_change_switches_slot() {
        let params = flat_params();
        let (mut controller, mut proc) = CabController::new(BLOCK, SR, 2, &params);
        controller.load_slot(2, impulse_ir(1.0)).unwrap();

        let slot = controller.handle_midi(SlotEvent::ProgramChange(2));
        assert_eq!(slot, Some(2));

        let mut input = vec![0.0f32; BLOCK];
        input[0] = 0.5;
        let _ = run_block(&mut proc, &input);
        assert_eq!(proc.active_slot(), 2);
    }

    #[test]
    fn stereo_ir_maps_channels_independently() {
        let params = flat_params();
        let (mut controller, mut proc) = CabController::new(BLOCK, SR, 2, &params);
        // 2-channel IR: left scales ×1, right scales ×3.
        let mut l = vec![0.0f32; BLOCK];
        l[0] = 1.0;
        let mut r = vec![0.0f32; BLOCK];
        r[0] = 3.0;
        controller
            .load_slot(0, PreparedIr::from_channels(vec![l, r], SR))
            .unwrap();
        controller.activate(0);

        let mut input = vec![0.0f32; BLOCK];
        input[0] = 0.5;
        let mut out_l = vec![0.0f32; BLOCK];
        let mut out_r = vec![0.0f32; BLOCK];
        {
            let ins: [&[f32]; 1] = [&input];
            let mut outs: [&mut [f32]; 2] = [&mut out_l, &mut out_r];
            proc.process(&ins, &mut outs);
        }
        // Same input + same filters on both channels, IRs differ by ×3, so the
        // right output is 3× the left sample-for-sample.
        assert!(out_l.iter().any(|&s| s.abs() > 1e-4), "left was silent");
        for (i, (&l, &r)) in out_l.iter().zip(out_r.iter()).enumerate() {
            assert!((r - 3.0 * l).abs() < 1e-4, "sample {i}: R {r} != 3·L {l}");
        }
    }

    #[test]
    fn crossfade_switch_retires_old_convolver_for_reclaim() {
        let params = CabParams {
            crossfade_ms: 5.0, // real fade so both engines coexist
            ..flat_params()
        };
        let (mut controller, mut proc) = CabController::new(BLOCK, SR, 2, &params);
        controller.load_slot(0, impulse_ir(1.0)).unwrap();
        controller.load_slot(1, impulse_ir(2.0)).unwrap();

        controller.activate(0);
        let mut input = vec![0.0f32; BLOCK];
        input[0] = 0.5;
        let _ = run_block(&mut proc, &input);

        // Switch to slot 1: the fade from the silent-initial front produced a
        // retired convolver on the first activation already.
        controller.activate(1);
        let _ = run_block(&mut proc, &input);
        assert!(proc.is_crossfading());

        // Controller reclaims retired convolvers off the audio thread.
        controller.reclaim(&mut proc);
    }
}
