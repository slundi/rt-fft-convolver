//! The neural amp processing engine, split across the two plugin threads.
//!
//! ```text
//! control thread                lock-free handoff             audio thread
//! ─────────────────             ──────────────────            ─────────────────
//! NeuralAmpController           IrPublisher / IrConsumer      NeuralAmpProcessor
//!   ModelSlotBank             ── publish(PreparedNeuralAmp) ─►  SlotCrossfader<..>
//!   load / build model                                          gain · mix · bypass
//!   MIDI routing                                                CPU-time meter
//! ```
//!
//! - [`NeuralAmpController`] lives off the audio thread.  It owns the loaded
//!   [`PreparedModel`]s in a slot bank and, on activation, hands the target
//!   slot's model over to the processor as a [`PreparedNeuralAmp`].
//! - [`NeuralAmpProcessor`] lives on the audio thread.  Each block it polls
//!   the handoff, kicks off a click-free crossfade if a new model arrived,
//!   applies the gain / mix / bypass envelope, and renders the signal chain.
//!   It never allocates.
//!
//! # Signal chain (per sample, not bypassed)
//!
//! ```text
//! xg  = in · input_gain                                  (dry reference)
//! wet = process_sample(xg) · slot_trim                   (neural model)
//! out = ((1 − mix) · xg + mix · wet) · output_gain
//! ```
//!
//! With an empty / disabled slot the wet path is `xg` verbatim — the plugin
//! passes audio through dry rather than muting.  Under host bypass the input
//! is copied through untouched (no gain either).

use rt_fft_convolver::DenormalGuard;
use rt_ir_plugin_core::{
    IrConsumer, IrPublisher, MidiRouter, SlotCrossfader, SlotError, SlotEvent, db_to_gain,
    ir_handoff, ms_to_samples,
};
use rt_neural_core::{ModelSlotBank, PreparedModel};

use crate::params::{NUM_SLOTS, NeuralAmpParams};

// ─── MonoConvolver wrapper around PreparedModel ──────────────────────────────

/// Adapts a sample-by-sample [`PreparedModel`] to the block-based
/// [`rt_ir_plugin_core::MonoConvolver`] trait so we can reuse
/// [`SlotCrossfader`] for click-free model switching.
///
/// The wrapper owns the model (moved into the crossfader by value) and a
/// fixed `block_size` set at construction — this matches the trait contract
/// but has no bearing on the model itself, which is inherently sample-by-
/// sample.  A "dry" wrapper (with `model = None`) provides the pass-through
/// behaviour needed when a slot is empty or disabled.
pub struct NeuralModelWrapper {
    /// The prepared model; `None` = dry pass-through.
    model: Option<PreparedModel>,
    /// Fixed block size the processor uses (matches the audio callback).
    block_size: usize,
}

impl NeuralModelWrapper {
    /// Wrap a real model.
    pub fn new(model: PreparedModel, block_size: usize) -> Self {
        Self {
            model: Some(model),
            block_size,
        }
    }

    /// Wrap a dry pass-through (no model).  Used for empty / disabled slots.
    pub fn dry(block_size: usize) -> Self {
        Self {
            model: None,
            block_size,
        }
    }

    /// Whether this wrapper is a dry pass-through.
    #[inline]
    pub fn is_dry(&self) -> bool {
        self.model.is_none()
    }
}

impl rt_ir_plugin_core::MonoConvolver for NeuralModelWrapper {
    #[inline]
    fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), output.len());
        match self.model.as_mut() {
            Some(model) => {
                for (sample_in, sample_out) in input.iter().zip(output.iter_mut()) {
                    *sample_out = model.process_sample(*sample_in);
                }
            }
            None => {
                // Dry pass-through.
                output.copy_from_slice(input);
            }
        }
    }

    #[inline]
    fn reset(&mut self) {
        if let Some(model) = self.model.as_mut() {
            model.reset();
        }
    }

    #[inline]
    fn block_size(&self) -> usize {
        self.block_size
    }
}

// ─── Handoff payload: a slot ready to render ─────────────────────────────────

/// A slot's model prepared for the audio thread.
///
/// Produced off-thread by [`NeuralAmpController`] and moved into the
/// processor's crossfader on arrival.  Carries the slot's dB trim
/// pre-converted to a linear multiplier so the audio thread never touches
/// `powf`.
pub struct PreparedNeuralAmp {
    /// The model wrapper (real model or dry pass-through).
    wrapper: NeuralModelWrapper,
    /// Slot trim as a linear multiplier applied to the wet signal.
    trim_gain: f32,
    /// Source slot index (for the processor's read-only reporting).
    slot: usize,
}

impl PreparedNeuralAmp {
    /// Slot index this payload was built from.
    #[inline]
    pub fn slot(&self) -> usize {
        self.slot
    }

    /// Whether the payload is dry (empty / disabled slot).
    #[inline]
    pub fn is_dry(&self) -> bool {
        self.wrapper.is_dry()
    }
}

// ─── Control-thread controller ───────────────────────────────────────────────

/// Control-thread side: owns the model bank and hands built models to the
/// processor through the lock-free [`IrPublisher`].
pub struct NeuralAmpController {
    bank: ModelSlotBank<NUM_SLOTS>,
    slot_enabled: [bool; NUM_SLOTS],
    slot_trim_db: [f32; NUM_SLOTS],
    publisher: IrPublisher<PreparedNeuralAmp>,
    block_size: usize,
    /// Maps MIDI Program Change / CC to a slot index.
    router: MidiRouter,
}

impl NeuralAmpController {
    /// Build a controller and its paired [`NeuralAmpProcessor`].
    ///
    /// - `block_size` — audio callback block size.
    /// - `sample_rate` — host sample rate (used for crossfade-time conversion).
    ///
    /// **Allocates** — call during plugin initialisation, off the audio thread.
    pub fn new(
        block_size: usize,
        sample_rate: u32,
        params: &NeuralAmpParams,
    ) -> (Self, NeuralAmpProcessor) {
        assert!(block_size > 0, "block_size must be > 0");

        let (publisher, consumer) = ir_handoff::<PreparedNeuralAmp>();
        let controller = Self {
            bank: ModelSlotBank::<NUM_SLOTS>::new(),
            slot_enabled: params.slot_enabled,
            slot_trim_db: params.slot_trim_db,
            publisher,
            block_size,
            router: MidiRouter::new(NUM_SLOTS),
        };
        let processor = NeuralAmpProcessor::new(consumer, block_size, sample_rate, params);
        (controller, processor)
    }

    /// Load a prepared model into `slot_index`.  Any model previously in the
    /// slot is dropped.
    ///
    /// **Not** real-time safe — dropping the previous model may release
    /// heap allocations.
    pub fn load_slot(&mut self, slot_index: usize, model: PreparedModel) -> Result<(), SlotError> {
        self.bank.load(slot_index, model).map_err(map_slot_error)
    }

    /// Clear `slot_index` — the slot is now empty (dry pass-through).
    pub fn clear_slot(&mut self, slot_index: usize) -> Result<(), SlotError> {
        self.bank.clear(slot_index).map_err(map_slot_error)
    }

    /// Enable / disable `slot_index`.  A disabled slot renders dry.
    ///
    /// Out-of-range indices are silently ignored (matches the roadmap
    /// requirement "empty slots pass audio through dry — never panic").
    pub fn set_slot_enabled(&mut self, slot_index: usize, enabled: bool) {
        if let Some(cell) = self.slot_enabled.get_mut(slot_index) {
            *cell = enabled;
        }
    }

    /// Set the per-slot dB trim.  Out-of-range indices are ignored.
    pub fn set_slot_trim_db(&mut self, slot_index: usize, trim_db: f32) {
        if let Some(cell) = self.slot_trim_db.get_mut(slot_index) {
            *cell = trim_db;
        }
    }

    /// Configure the MIDI-learn target CC (`None` = no CC learned yet).
    pub fn set_learn_cc(&mut self, cc: Option<u8>) {
        self.router.set_cc(cc);
    }

    /// Currently learned MIDI CC, if any.
    pub fn learn_cc(&self) -> Option<u8> {
        self.router.cc()
    }

    /// Route a MIDI event to a slot activation.  Returns the resolved slot
    /// index (if the event mapped to one) so the plugin can update its
    /// automatable `current_slot` param.
    pub fn route_midi(&mut self, event: SlotEvent) -> Option<usize> {
        let slot_index = self.router.route(event)?;
        self.activate(slot_index).ok()?;
        Some(slot_index)
    }

    /// Activate `slot_index` — builds a [`PreparedNeuralAmp`] and publishes
    /// it through the handoff.  The audio thread will pick it up at the top
    /// of the next block.
    ///
    /// **Not** real-time safe (may allocate for the wrapper move).
    pub fn activate(&mut self, slot_index: usize) -> Result<(), SlotError> {
        if slot_index >= NUM_SLOTS {
            return Err(SlotError::IndexOutOfRange {
                index: slot_index,
                capacity: NUM_SLOTS,
            });
        }

        // Update the bank's own record (so `active_index` on the audio
        // side stays consistent with what we publish).
        self.bank
            .request_activation(slot_index)
            .map_err(map_slot_error)?;
        self.bank.take_pending();

        // Build the payload.  A disabled slot or an empty slot both become
        // a dry pass-through.
        let enabled = self.slot_enabled[slot_index];
        let is_loaded = self.bank.is_loaded(slot_index).unwrap_or(false);
        let wrapper = if enabled && is_loaded {
            // Move the model out of the bank into the wrapper — the wrapper
            // owns it until the crossfader hands the retired one back.  When
            // the caller wants to swap models on a slot, they call
            // `load_slot` again, which replaces the bank's copy.
            let model = self.bank.active_mut().map(|slot| {
                let mut placeholder = PreparedModel::WaveNet(
                    // Should never leave this closure — but Rust needs an
                    // initialiser for `std::mem::swap`.  A tiny sentinel is
                    // fine here.
                    build_dry_sentinel(),
                );
                std::mem::swap(&mut placeholder, slot);
                placeholder
            });
            match model {
                Some(model) => NeuralModelWrapper::new(model, self.block_size),
                None => NeuralModelWrapper::dry(self.block_size),
            }
        } else {
            NeuralModelWrapper::dry(self.block_size)
        };

        let trim_gain = db_to_gain(self.slot_trim_db[slot_index]);

        let payload = PreparedNeuralAmp {
            wrapper,
            trim_gain,
            slot: slot_index,
        };
        self.publisher.publish(payload);
        Ok(())
    }
}

/// Build a "dry sentinel" PreparedModel used only as an `std::mem::swap`
/// placeholder inside [`NeuralAmpController::activate`].
///
/// This constructor is intentionally private and produces a **valid**
/// (though empty) WaveNet model, so if it ever escaped its narrow scope it
/// would render silence rather than panic or dangle.  See the swap site
/// for why the placeholder pattern is needed at all.
fn build_dry_sentinel() -> rt_neural_core::WaveNetModel {
    use rt_neural_core::WaveNetModel;
    use rt_neural_core::conv1d::Conv1d;
    use rt_neural_core::prepared::WaveNetBlock;

    let conv = Conv1d::new(vec![0.0f32; 1], vec![0.0f32; 1], 1, 1, 1, 1)
        .expect("sentinel conv1d must build");
    let block = WaveNetBlock::new(conv, vec![0.0f32; 1], vec![0.0f32; 1]);
    WaveNetModel::new(
        vec![0.0f32; 1],
        vec![0.0f32; 1],
        vec![block],
        vec![0.0f32; 1],
        0.0,
    )
}

/// The `rt-ir-plugin-core::SlotError` variant matches the `rt-neural-core`
/// one — this shim keeps the public error type on the plugin surface a
/// single kind rather than leaking two identical enums.
fn map_slot_error(error: rt_neural_core::slot_bank::SlotError) -> SlotError {
    match error {
        rt_neural_core::slot_bank::SlotError::IndexOutOfRange { index, capacity } => {
            SlotError::IndexOutOfRange { index, capacity }
        }
    }
}

// ─── Audio-thread processor ──────────────────────────────────────────────────

/// Audio-thread side: renders the signal chain block-by-block.
pub struct NeuralAmpProcessor {
    consumer: IrConsumer<PreparedNeuralAmp>,
    crossfader: SlotCrossfader<NeuralModelWrapper>,
    /// Cached crossfade time in samples (recomputed on parameter changes).
    crossfade_samples: usize,
    /// Slot trim of the active payload, applied to the wet signal.
    trim_gain: f32,
    /// Scratch buffer holding the wet signal for one block.
    wet_scratch: Vec<f32>,
    /// Scratch buffer holding the (input × input_gain) dry reference.
    dry_scratch: Vec<f32>,
    block_size: usize,
    sample_rate: u32,
    /// Read-only inference-time report, in samples of the block deadline.
    /// A future CPU meter reads this atomically.
    last_process_samples: u32,
}

impl NeuralAmpProcessor {
    fn new(
        consumer: IrConsumer<PreparedNeuralAmp>,
        block_size: usize,
        sample_rate: u32,
        params: &NeuralAmpParams,
    ) -> Self {
        Self {
            consumer,
            // Crossfader starts with a dry wrapper — until the controller
            // publishes a real model, the plugin passes audio through dry.
            crossfader: SlotCrossfader::new(NeuralModelWrapper::dry(block_size)),
            crossfade_samples: ms_to_samples(params.crossfade_ms, sample_rate),
            trim_gain: 1.0,
            wet_scratch: vec![0.0f32; block_size],
            dry_scratch: vec![0.0f32; block_size],
            block_size,
            sample_rate,
            last_process_samples: 0,
        }
    }

    /// Update the crossfade time from a possibly-smoothed parameter value.
    ///
    /// Called at the top of each block from the plugin wrapper.  Real-time
    /// safe — recomputes the cached sample count, no allocation.
    #[inline]
    pub fn set_crossfade_ms(&mut self, crossfade_ms: f32) {
        self.crossfade_samples = ms_to_samples(crossfade_ms, self.sample_rate);
    }

    /// Report of the most recently rendered block's inference cost, in
    /// samples of the block deadline.  A CPU meter divides this by
    /// `block_size` to get a fraction.
    #[inline]
    pub fn last_process_samples(&self) -> u32 {
        self.last_process_samples
    }

    /// Poll the handoff and, if a new model arrived, start a fade to it.
    ///
    /// Returns the newly-activated slot index (if any) so the plugin can
    /// mirror it back into an editor-facing param snapshot.
    ///
    /// **Real-time safe** — the retired wrapper is stashed inside the
    /// crossfader / triple-buffer and reclaimed by the *producer* later,
    /// never dropped here.
    #[inline]
    pub fn poll_activation(&mut self) -> Option<usize> {
        let payload = self.consumer.poll()?;
        let payload = std::mem::replace(
            payload,
            PreparedNeuralAmp {
                wrapper: NeuralModelWrapper::dry(self.block_size),
                trim_gain: 1.0,
                slot: 0,
            },
        );
        let slot = payload.slot;
        self.trim_gain = payload.trim_gain;
        // The displaced wrapper is written back into the triple-buffer
        // slot — it will be dropped by the producer thread.
        let _retired = self
            .crossfader
            .start_fade(payload.wrapper, self.crossfade_samples);
        Some(slot)
    }

    /// Render one mono block: `input` → `output`, applying the signal
    /// chain described in the module docs.
    ///
    /// **Real-time safe** — no allocation.  Wraps the whole call in a
    /// [`DenormalGuard`] because model outputs decay toward zero on
    /// note-off and denormals are ~10× slower than normal FP ops.
    #[inline]
    pub fn process(&mut self, input: &[f32], output: &mut [f32], params: &NeuralAmpParams) {
        debug_assert_eq!(input.len(), self.block_size);
        debug_assert_eq!(output.len(), self.block_size);

        let _guard = DenormalGuard::new();

        if params.bypass {
            output.copy_from_slice(input);
            self.last_process_samples = 0;
            return;
        }

        // ── Signal chain, sample by sample so the model sees a scaled input.
        let input_gain = db_to_gain(params.input_gain_db);
        let output_gain = db_to_gain(params.output_gain_db);
        let mix = params.mix.clamp(0.0, 1.0);

        // Build the dry reference (input · input_gain) in dry_scratch, and
        // render the wet path through the crossfader into wet_scratch.  The
        // crossfader's process_block signature is `(input: &[f32], output: &mut [f32])`,
        // which matches our per-sample loop cleanly once we've written the
        // dry buffer.
        for (source, target) in input.iter().zip(self.dry_scratch.iter_mut()) {
            *target = *source * input_gain;
        }
        self.crossfader
            .process(&self.dry_scratch, &mut self.wet_scratch);

        // Fuse: out = ((1 - mix) · dry + mix · (wet · trim)) · output_gain
        let dry_weight = (1.0 - mix) * output_gain;
        let wet_weight = mix * self.trim_gain * output_gain;
        for (index, target) in output.iter_mut().enumerate() {
            *target = dry_weight * self.dry_scratch[index] + wet_weight * self.wet_scratch[index];
        }

        // Rough CPU-meter approximation: report the block size itself as
        // "samples used".  A real profiler-backed measure is a Phase-3 UI
        // task; for now the meter can display "≈ 100 %" when a block
        // renders, "0 %" when bypassed.
        self.last_process_samples = self.block_size as u32;
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rt_ir_plugin_core::MonoConvolver;
    use rt_neural_core::lstm::LstmCell;
    use rt_neural_core::prepared::{LstmModel, PreparedModel};

    /// Build a trivial silent LSTM model.
    fn silent_lstm_model() -> PreparedModel {
        let hidden_size = 2;
        let input_size = 1;
        let cell = LstmCell::new(
            vec![0.0f32; 4 * hidden_size * input_size],
            vec![0.0f32; 4 * hidden_size * hidden_size],
            vec![0.0f32; 4 * hidden_size],
            input_size,
            hidden_size,
        )
        .expect("LstmCell::new");
        PreparedModel::Lstm(LstmModel::new(vec![cell], vec![0.0f32; hidden_size], 0.0))
    }

    // ── NeuralModelWrapper ───────────────────────────────────────────────────

    /// A dry wrapper copies input to output verbatim.
    #[test]
    fn dry_wrapper_passes_through() {
        let mut wrapper = NeuralModelWrapper::dry(4);
        assert!(wrapper.is_dry());
        let input = [1.0f32, -0.5, 0.25, 0.0];
        let mut output = [99.0f32; 4];
        wrapper.process_block(&input, &mut output);
        assert_eq!(output, input);
    }

    /// A real (silent) LSTM model wrapper emits zeros.
    #[test]
    fn silent_lstm_wrapper_emits_zero() {
        let mut wrapper = NeuralModelWrapper::new(silent_lstm_model(), 4);
        assert!(!wrapper.is_dry());
        let input = [1.0f32, -0.5, 0.25, 0.0];
        let mut output = [99.0f32; 4];
        wrapper.process_block(&input, &mut output);
        for value in output {
            assert!(value.abs() < 1e-6);
        }
    }

    // ── Controller / Processor plumbing ──────────────────────────────────────

    /// A freshly-built processor with no loaded slot passes audio through
    /// dry (times the unit input / output gains at defaults).
    #[test]
    fn fresh_processor_is_dry_pass_through_at_defaults() {
        let params = NeuralAmpParams::default();
        let (_controller, mut processor) = NeuralAmpController::new(4, 48_000, &params);
        let input = [0.1f32, 0.2, 0.3, 0.4];
        let mut output = [0.0f32; 4];
        processor.process(&input, &mut output, &params);
        // Defaults: mix = 1.0, input/output gain 0 dB, but the crossfader
        // holds a dry wrapper → wet == dry.  So output ≈ input.
        for (produced, expected) in output.iter().zip(input.iter()) {
            assert!(
                (produced - expected).abs() < 1e-5,
                "produced={produced}, expected={expected}"
            );
        }
    }

    /// Host bypass short-circuits the chain: output == input verbatim.
    #[test]
    fn bypass_short_circuits_chain() {
        let params = NeuralAmpParams {
            bypass: true,
            input_gain_db: 12.0, // must be ignored under bypass
            output_gain_db: -12.0,
            ..NeuralAmpParams::default()
        };
        let (_controller, mut processor) = NeuralAmpController::new(4, 48_000, &params);
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let mut output = [99.0f32; 4];
        processor.process(&input, &mut output, &params);
        assert_eq!(output, input);
        assert_eq!(processor.last_process_samples(), 0);
    }

    /// mix = 0 (fully dry) gives `input · input_gain · output_gain`.
    #[test]
    fn mix_zero_gives_pure_dry_path() {
        let params = NeuralAmpParams {
            mix: 0.0,
            input_gain_db: 6.02,   // ~ ×2
            output_gain_db: -6.02, // ~ ×0.5, cancels out
            ..NeuralAmpParams::default()
        };
        let (_controller, mut processor) = NeuralAmpController::new(4, 48_000, &params);
        let input = [1.0f32; 4];
        let mut output = [0.0f32; 4];
        processor.process(&input, &mut output, &params);
        for value in output {
            // 2.0 × 0.5 = 1.0 (± tolerance for the 6.02 dB rounding)
            assert!((value - 1.0).abs() < 5e-3, "expected ≈ 1.0, got {value}");
        }
    }

    /// Loading a slot succeeds; activating that slot publishes a
    /// PreparedNeuralAmp the processor can pick up.
    #[test]
    fn load_then_activate_flows_through_handoff() {
        let params = NeuralAmpParams::default();
        let (mut controller, mut processor) = NeuralAmpController::new(4, 48_000, &params);
        controller
            .load_slot(3, silent_lstm_model())
            .expect("load_slot");
        controller.activate(3).expect("activate");
        let activated = processor.poll_activation();
        assert_eq!(activated, Some(3));
    }

    /// An empty slot activation still publishes a payload — the processor
    /// records the slot switch even though the wrapper is dry.
    #[test]
    fn activating_empty_slot_yields_dry_payload() {
        let params = NeuralAmpParams::default();
        let (mut controller, mut processor) = NeuralAmpController::new(4, 48_000, &params);
        controller.activate(5).expect("activate");
        assert_eq!(processor.poll_activation(), Some(5));
    }

    /// A disabled slot activation renders dry (wrapper is dry) even
    /// though a model was loaded.
    #[test]
    fn disabled_slot_activation_renders_dry() {
        let mut params = NeuralAmpParams::default();
        params.slot_enabled[2] = false;
        let (mut controller, mut processor) = NeuralAmpController::new(4, 48_000, &params);
        controller
            .load_slot(2, silent_lstm_model())
            .expect("load_slot");
        controller.set_slot_enabled(2, false);
        // Zero-length crossfade so the wrapper switch is instant — dry→dry
        // fades otherwise get equal-power "hump" from correlated signals.
        processor.set_crossfade_ms(0.0);
        controller.activate(2).expect("activate");
        let _ = processor.poll_activation();

        // Push samples through: bypass off, model would emit 0, but disabled ⇒
        // dry, so we should see `input · gains`.
        let input = [0.5f32; 4];
        let mut output = [0.0f32; 4];
        processor.process(&input, &mut output, &params);
        for value in output {
            assert!(
                (value - 0.5).abs() < 1e-5,
                "expected 0.5 (dry pass), got {value}"
            );
        }
    }

    /// MIDI routing dispatches PC → slot correctly and returns the slot.
    #[test]
    fn midi_program_change_routes_to_slot() {
        let params = NeuralAmpParams::default();
        let (mut controller, _processor) = NeuralAmpController::new(4, 48_000, &params);
        let slot = controller.route_midi(SlotEvent::ProgramChange(4));
        assert_eq!(slot, Some(4));
    }

    /// MIDI-learn CC round-trips.
    #[test]
    fn midi_learn_cc_round_trips() {
        let params = NeuralAmpParams::default();
        let (mut controller, _processor) = NeuralAmpController::new(4, 48_000, &params);
        assert!(controller.learn_cc().is_none());
        controller.set_learn_cc(Some(64));
        assert_eq!(controller.learn_cc(), Some(64));
    }

    /// Out-of-range slot indices are rejected by load / clear / activate.
    #[test]
    fn out_of_range_slot_operations_are_rejected() {
        let params = NeuralAmpParams::default();
        let (mut controller, _processor) = NeuralAmpController::new(4, 48_000, &params);
        assert!(
            controller
                .load_slot(NUM_SLOTS, silent_lstm_model())
                .is_err()
        );
        assert!(controller.clear_slot(NUM_SLOTS).is_err());
        assert!(controller.activate(NUM_SLOTS).is_err());
    }
}
