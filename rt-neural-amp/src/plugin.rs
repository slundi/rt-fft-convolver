//! `nih-plug` wrapper for the neural amp (compiled only with the `nih`
//! feature).
//!
//! This is the thin framework layer over the tested, framework-agnostic
//! [`engine`](crate::engine): it declares the automatable parameters, wires
//! MIDI + parameter changes into the off-thread [`NeuralAmpController`], and
//! runs the [`MonoBlockAdapter`] on the audio thread.
//!
//! # Threading
//!
//! - The [`NeuralAmpController`] lives behind a `Mutex` shared with the
//!   `nih-plug` **background task** executor.  Model loading and preparation
//!   (both allocate) happen there, never on the audio thread.
//! - `process()` only polls the lock-free handoff for freshly-published
//!   models, applies the smoothed parameters, and renders — allocation-free.
//!
//! # Known Phase-3/4 follow-ups
//!
//! - The NAM / Proteus JSON loader is stubbed in
//!   [`rt_neural_core::loader`]; model loading is triggered via
//!   [`load_slot`](NeuralAmpController::load_slot) with a
//!   pre-built [`PreparedModel`](rt_neural_core::PreparedModel) for now.
//! - The `nih-plug-egui` editor (slot rack, input meter with sweet-spot
//!   indicator, model metadata, MIDI activity LED) is Phase 3.
//! - `lv2lint` / `clap-validator` / `pluginval` host validation is Phase 4.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use nih_plug::prelude::*;
use parking_lot::Mutex as PlMutex;
use rt_ir_plugin_core::SlotEvent;

use crate::block::MonoBlockAdapter;
use crate::engine::NeuralAmpController;
use crate::params::{
    CROSSFADE_MS_RANGE, GAIN_DB_RANGE, MODEL_TRIM_DB_RANGE, NUM_SLOTS, NeuralAmpParams,
};

/// Fixed internal render block.  Latency reported to the host equals this.
///
/// 128 samples ≈ 2.7 ms at 48 kHz — small enough for live use, big enough
/// that per-block dispatch overhead is negligible compared to the
/// sample-by-sample inference cost.
const BLOCK_SIZE: usize = 128;

/// Off-thread work the audio / editor threads dispatch to the background
/// executor.  Every variant is handled on the control thread only.
///
/// `Load` and `LearnCc` are constructed by the Phase-3 egui editor
/// (drag-drop and right-click respectively); they read as dead code in
/// Phase 2 which is fine.
#[allow(dead_code)]
pub enum NeuralAmpTask {
    /// Load a model file into a slot (then activate it if it is the current slot).
    Load {
        /// Target slot index.
        slot: usize,
        /// Path to the NAM / Proteus JSON file.
        path: PathBuf,
    },
    /// Rebuild + hand off the model for a slot.
    Activate(usize),
    /// A MIDI event to route to a slot.
    Midi(SlotEvent),
    /// Enable / disable a slot.
    SetEnabled {
        /// Slot index.
        slot: usize,
        /// New enabled state.
        enabled: bool,
    },
    /// Set the per-slot dB trim.
    SetTrim {
        /// Slot index.
        slot: usize,
        /// Trim in dB.
        trim_db: f32,
    },
    /// Learn a MIDI CC on the `Current Slot` parameter.
    LearnCc(Option<u8>),
}

/// Per-slot automatable parameters.  File paths are persisted state,
/// assigned via the (Phase-3) editor.
#[derive(Params)]
struct SlotParams {
    #[id = "on"]
    enabled: BoolParam,
    /// Per-slot model level trim in dB — non-automatable per roadmap.
    #[id = "trim"]
    trim_db: FloatParam,
}

impl Default for SlotParams {
    fn default() -> Self {
        Self {
            enabled: BoolParam::new("Slot Enabled", true),
            trim_db: FloatParam::new(
                "Model Level Trim",
                0.0,
                FloatRange::Linear {
                    min: MODEL_TRIM_DB_RANGE.0,
                    max: MODEL_TRIM_DB_RANGE.1,
                },
            )
            .with_unit(" dB")
            .non_automatable(),
        }
    }
}

/// All host-visible parameters for the neural amp.
#[derive(Params)]
struct NeuralAmpPluginParams {
    #[id = "slot"]
    current_slot: IntParam,
    #[id = "xfade"]
    crossfade_ms: FloatParam,
    #[id = "ingain"]
    input_gain_db: FloatParam,
    #[id = "outgain"]
    output_gain_db: FloatParam,
    #[id = "mix"]
    mix: FloatParam,
    #[id = "byp"]
    bypass: BoolParam,

    #[nested(array, group = "Slots")]
    slots: [SlotParams; NUM_SLOTS],

    /// Model file path per slot, restored with the plugin state.
    #[persist = "model_paths"]
    model_paths: PlMutex<Vec<Option<String>>>,

    /// Learned MIDI CC for `Current Slot`, persisted across sessions.
    #[persist = "learn_cc"]
    learn_cc: PlMutex<Option<u8>>,
}

impl Default for NeuralAmpPluginParams {
    fn default() -> Self {
        Self {
            current_slot: IntParam::new(
                "Current Slot",
                0,
                IntRange::Linear {
                    min: 0,
                    max: (NUM_SLOTS - 1) as i32,
                },
            ),
            crossfade_ms: FloatParam::new(
                "Crossfade Time",
                20.0,
                FloatRange::Linear {
                    min: CROSSFADE_MS_RANGE.0,
                    max: CROSSFADE_MS_RANGE.1,
                },
            )
            .with_unit(" ms"),
            input_gain_db: FloatParam::new(
                "Input Gain",
                0.0,
                FloatRange::Linear {
                    min: GAIN_DB_RANGE.0,
                    max: GAIN_DB_RANGE.1,
                },
            )
            .with_unit(" dB"),
            output_gain_db: FloatParam::new(
                "Output Gain",
                0.0,
                FloatRange::Linear {
                    min: GAIN_DB_RANGE.0,
                    max: GAIN_DB_RANGE.1,
                },
            )
            .with_unit(" dB"),
            mix: FloatParam::new("Mix", 1.0, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_unit(" %")
                .with_value_to_string(formatters::v2s_f32_percentage(0))
                .with_string_to_value(formatters::s2v_f32_percentage()),
            bypass: BoolParam::new("Bypass", false)
                .with_value_to_string(formatters::v2s_bool_bypass()),
            slots: core::array::from_fn(|_| SlotParams::default()),
            model_paths: PlMutex::new(vec![None; NUM_SLOTS]),
            learn_cc: PlMutex::new(None),
        }
    }
}

impl NeuralAmpPluginParams {
    /// Snapshot the current values into the framework-agnostic model.
    fn snapshot(&self) -> NeuralAmpParams {
        NeuralAmpParams {
            current_slot: self.current_slot.value() as usize,
            crossfade_ms: self.crossfade_ms.value(),
            input_gain_db: self.input_gain_db.value(),
            output_gain_db: self.output_gain_db.value(),
            mix: self.mix.value(),
            bypass: self.bypass.value(),
            slot_enabled: core::array::from_fn(|index| self.slots[index].enabled.value()),
            slot_trim_db: core::array::from_fn(|index| self.slots[index].trim_db.value()),
        }
    }
}

/// The neural amp plugin instance.
pub struct NeuralAmpPlugin {
    params: Arc<NeuralAmpPluginParams>,
    /// Shared with the background-task executor; built in `initialize`.
    controller: Arc<Mutex<Option<NeuralAmpController>>>,
    /// Audio-thread renderer; built in `initialize`.
    adapter: Option<MonoBlockAdapter>,

    // Change detection so we only dispatch tasks when a value actually moves.
    last_slot: i32,
    last_enabled: [bool; NUM_SLOTS],
    last_trim_db: [f32; NUM_SLOTS],

    /// Bumped on each recognised MIDI event (Phase 3 UI reads this).
    #[allow(dead_code)]
    midi_counter: Arc<AtomicU32>,
    /// Last-block inference cost as a fraction of `BLOCK_SIZE`, ready for
    /// the CPU meter (Phase 3 reads this atomically).
    cpu_meter: Arc<AtomicU32>,
}

impl Default for NeuralAmpPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(NeuralAmpPluginParams::default()),
            controller: Arc::new(Mutex::new(None)),
            adapter: None,
            last_slot: 0,
            last_enabled: [true; NUM_SLOTS],
            last_trim_db: [0.0f32; NUM_SLOTS],
            midi_counter: Arc::new(AtomicU32::new(0)),
            cpu_meter: Arc::new(AtomicU32::new(0)),
        }
    }
}

impl Plugin for NeuralAmpPlugin {
    const NAME: &'static str = "RT Neural Amp";
    const VENDOR: &'static str = "Sébastien Lundi";
    const URL: &'static str = "https://codeberg.org/slundi/rt-fft-convolver";
    const EMAIL: &'static str = "slundi@stamus-networks.com";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    // Mono in → mono out per Phase 2 ("models are inherently mono").
    // Stereo is achieved by placing two plugin instances on L/R.
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(1),
        main_output_channels: NonZeroU32::new(1),
        ..AudioIOLayout::const_default()
    }];

    const MIDI_INPUT: MidiConfig = MidiConfig::MidiCCs;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = NeuralAmpTask;

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn task_executor(&mut self) -> TaskExecutor<Self> {
        let controller = self.controller.clone();
        Box::new(move |task| {
            let mut guard = controller.lock().unwrap();
            let Some(controller) = guard.as_mut() else {
                return;
            };
            match task {
                NeuralAmpTask::Load { slot, path: _ } => {
                    // Phase 1.5 will invoke `rt_neural_core::loader::load_nam`
                    // (or `load_proteus`) on the file bytes and hand the
                    // resulting `PreparedModel` to `controller.load_slot`.  For
                    // Phase 2 we log a clear diagnostic and leave the slot
                    // untouched — the plugin still passes audio through dry.
                    nih_error!(
                        "model loader not implemented in Phase 2; slot {slot} left unchanged"
                    );
                }
                NeuralAmpTask::Activate(slot) => {
                    if let Err(error) = controller.activate(slot) {
                        nih_error!("activate({slot}) failed: {error}");
                    }
                }
                NeuralAmpTask::Midi(event) => {
                    controller.route_midi(event);
                }
                NeuralAmpTask::SetEnabled { slot, enabled } => {
                    controller.set_slot_enabled(slot, enabled);
                }
                NeuralAmpTask::SetTrim { slot, trim_db } => {
                    controller.set_slot_trim_db(slot, trim_db);
                }
                NeuralAmpTask::LearnCc(cc) => {
                    controller.set_learn_cc(cc);
                }
            }
        })
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        let sample_rate = buffer_config.sample_rate as u32;
        let snapshot = self.params.snapshot();
        let (mut controller, processor) =
            NeuralAmpController::new(BLOCK_SIZE, sample_rate, &snapshot);

        // Restore per-slot enable / trim state from persisted params.
        for (slot_index, slot_params) in self.params.slots.iter().enumerate() {
            controller.set_slot_enabled(slot_index, slot_params.enabled.value());
            controller.set_slot_trim_db(slot_index, slot_params.trim_db.value());
        }

        // Restore the learned MIDI CC.
        controller.set_learn_cc(*self.params.learn_cc.lock());

        // Activate the current slot so the audio thread has a payload ready.
        // Even with no model loaded yet, this publishes a dry wrapper so the
        // plugin still passes audio through.
        let current_slot = self.params.current_slot.value() as usize;
        let _ = controller.activate(current_slot);

        let adapter = MonoBlockAdapter::new(processor, BLOCK_SIZE);
        context.set_latency_samples(adapter.latency_samples() as u32);

        *self.controller.lock().unwrap() = Some(controller);
        self.adapter = Some(adapter);

        self.last_slot = self.params.current_slot.value();
        self.last_enabled = core::array::from_fn(|index| self.params.slots[index].enabled.value());
        self.last_trim_db = core::array::from_fn(|index| self.params.slots[index].trim_db.value());
        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // ── 1. Route incoming MIDI to slot activations (off-thread).
        while let Some(event) = context.next_event() {
            match event {
                NoteEvent::MidiProgramChange { program, .. } => {
                    context
                        .execute_background(NeuralAmpTask::Midi(SlotEvent::ProgramChange(program)));
                    self.midi_counter.fetch_add(1, Ordering::Relaxed);
                }
                NoteEvent::MidiCC { cc, value, .. } => {
                    let raw = (value * 127.0).round().clamp(0.0, 127.0) as u8;
                    context.execute_background(NeuralAmpTask::Midi(SlotEvent::ControlChange {
                        controller: cc,
                        value: raw,
                    }));
                    self.midi_counter.fetch_add(1, Ordering::Relaxed);
                }
                _ => {}
            }
        }

        // ── 2. Dispatch parameter-driven changes (only on an actual move).
        for slot_index in 0..NUM_SLOTS {
            let enabled = self.params.slots[slot_index].enabled.value();
            if enabled != self.last_enabled[slot_index] {
                self.last_enabled[slot_index] = enabled;
                context.execute_background(NeuralAmpTask::SetEnabled {
                    slot: slot_index,
                    enabled,
                });
                if slot_index == self.params.current_slot.value() as usize {
                    context.execute_background(NeuralAmpTask::Activate(slot_index));
                }
            }
            let trim_db = self.params.slots[slot_index].trim_db.value();
            if (trim_db - self.last_trim_db[slot_index]).abs() > f32::EPSILON {
                self.last_trim_db[slot_index] = trim_db;
                context.execute_background(NeuralAmpTask::SetTrim {
                    slot: slot_index,
                    trim_db,
                });
                if slot_index == self.params.current_slot.value() as usize {
                    context.execute_background(NeuralAmpTask::Activate(slot_index));
                }
            }
        }
        let slot = self.params.current_slot.value();
        if slot != self.last_slot {
            self.last_slot = slot;
            context.execute_background(NeuralAmpTask::Activate(slot as usize));
        }

        // ── 3. Render.  Neural amp is inherently mono; take channel 0.
        let snapshot = self.params.snapshot();
        if let Some(adapter) = self.adapter.as_mut() {
            adapter
                .processor_mut()
                .set_crossfade_ms(snapshot.crossfade_ms);
            // Poll the handoff first so any freshly-published model is picked
            // up before we render this block.
            let _ = adapter.processor_mut().poll_activation();

            let channels = buffer.as_slice();
            if let Some(channel) = channels.first_mut() {
                adapter.process_buffer(channel, &snapshot);
            }

            // Update CPU meter (last-block cost / block size).
            let cost = adapter.processor_mut().last_process_samples();
            self.cpu_meter.store(cost, Ordering::Relaxed);
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for NeuralAmpPlugin {
    const CLAP_ID: &'static str = "org.codeberg.slundi.rt-neural-amp";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("Real-time neural amp / pedal modelling plugin");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Mono,
        ClapFeature::Distortion,
    ];
}

impl Vst3Plugin for NeuralAmpPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"RtNeuAmpSlnd0001";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Distortion];
}

nih_export_clap!(NeuralAmpPlugin);
nih_export_vst3!(NeuralAmpPlugin);
