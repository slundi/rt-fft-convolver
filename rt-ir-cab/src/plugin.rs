//! `nih-plug` wrapper for the cab (compiled only with the `nih` feature).
//!
//! This is the thin framework layer over the tested, framework-agnostic
//! [`engine`](crate::engine): it declares the automatable parameters, wires
//! MIDI + parameter changes into the off-thread [`CabController`], and runs the
//! [`StereoBlockAdapter`] on the audio thread.
//!
//! # Threading
//!
//! - The [`CabController`] lives behind a `Mutex` shared with the `nih-plug`
//!   **background task** executor.  IR loading and cab building (both allocate)
//!   happen there, never on the audio thread.
//! - `process()` only pulls prebuilt cabs through the lock-free handoff inside
//!   the adapter, applies the smoothed parameters, and renders — allocation-free.
//!
//! # Known Phase-3/4 follow-ups
//!
//! - IR file assignment currently comes from persisted paths (restored on
//!   load); the drag-and-drop UI is Phase 3.
//! - The CPU meter (`cpu_estimate`) and MIDI-learn UI are Phase 3.
//! - Host validation (`lv2lint`, `clap-validator`, `pluginval`) is Phase 4.
//!   `nih-plug` exports CLAP + VST3; an LV2 build needs a CLAP→LV2 bridge.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use nih_plug::prelude::*;
use parking_lot::Mutex as PlMutex;
use rt_ir_plugin_core::SlotEvent;

use crate::block::StereoBlockAdapter;
use crate::engine::CabController;
use crate::params::{
    CROSSFADE_MS_RANGE, CabParams, GAIN_DB_RANGE, HIGH_CUT_RANGE, LOW_CUT_RANGE, NUM_SLOTS,
};

/// Fixed internal render block.  The plugin reports `BLOCK_SIZE` samples of
/// latency (see [`StereoBlockAdapter`]).
const BLOCK_SIZE: usize = 512;

/// Off-thread work the audio/GUI threads dispatch to the background executor.
pub enum CabTask {
    /// Load a WAV into a slot (then activate it if it is the current slot).
    ///
    /// The executor already handles this; the Phase-3 drag-and-drop UI will be
    /// the caller (initialization restores persisted paths synchronously).
    #[allow(dead_code)]
    Load { slot: usize, path: PathBuf },
    /// Rebuild + hand off the cab for a slot.
    Activate(usize),
    /// A MIDI event to route to a slot.
    Midi(SlotEvent),
    /// Enable/disable a slot.
    SetEnabled { slot: usize, enabled: bool },
    /// Toggle IR normalisation for future activations.
    SetNormalize(bool),
}

/// Per-slot automatable parameters (just the enable toggle; file paths are
/// persisted state, assigned via the UI in Phase 3).
#[derive(Params)]
struct SlotParams {
    #[id = "on"]
    enabled: BoolParam,
}

impl Default for SlotParams {
    fn default() -> Self {
        Self {
            enabled: BoolParam::new("Slot Enabled", true),
        }
    }
}

/// All host-visible parameters for the cab.
#[derive(Params)]
struct CabPluginParams {
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
    #[id = "locut"]
    low_cut: FloatParam,
    #[id = "hicut"]
    high_cut: FloatParam,
    #[id = "norm"]
    ir_normalize: BoolParam,
    #[id = "byp"]
    bypass: BoolParam,

    #[nested(array, group = "Slots")]
    slots: [SlotParams; NUM_SLOTS],

    /// IR file path per slot, restored with the plugin state.
    #[persist = "ir_paths"]
    ir_paths: PlMutex<Vec<Option<String>>>,
}

impl Default for CabPluginParams {
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
            low_cut: FloatParam::new(
                "Low Cut",
                LOW_CUT_RANGE.0,
                FloatRange::Skewed {
                    min: LOW_CUT_RANGE.0,
                    max: LOW_CUT_RANGE.1,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            .with_unit(" Hz"),
            high_cut: FloatParam::new(
                "High Cut",
                HIGH_CUT_RANGE.1,
                FloatRange::Skewed {
                    min: HIGH_CUT_RANGE.0,
                    max: HIGH_CUT_RANGE.1,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            .with_unit(" Hz"),
            ir_normalize: BoolParam::new("IR Normalize", false),
            bypass: BoolParam::new("Bypass", false)
                .with_value_to_string(formatters::v2s_bool_bypass()),
            slots: core::array::from_fn(|_| SlotParams::default()),
            ir_paths: PlMutex::new(vec![None; NUM_SLOTS]),
        }
    }
}

impl CabPluginParams {
    /// Snapshot the current values into the framework-agnostic model.
    fn snapshot(&self) -> CabParams {
        CabParams {
            current_slot: self.current_slot.value() as usize,
            crossfade_ms: self.crossfade_ms.value(),
            input_gain_db: self.input_gain_db.value(),
            output_gain_db: self.output_gain_db.value(),
            mix: self.mix.value(),
            low_cut_hz: self.low_cut.value(),
            high_cut_hz: self.high_cut.value(),
            ir_normalize: self.ir_normalize.value(),
            bypass: self.bypass.value(),
            slot_enabled: core::array::from_fn(|i| self.slots[i].enabled.value()),
        }
    }
}

/// The cab plugin instance.
pub struct CabPlugin {
    params: Arc<CabPluginParams>,
    /// Shared with the background-task executor; built in `initialize`.
    controller: Arc<Mutex<Option<CabController>>>,
    /// Audio-thread renderer; built in `initialize`.
    adapter: Option<StereoBlockAdapter>,

    // Change detection so we only dispatch tasks when a value actually moves.
    last_slot: i32,
    last_enabled: [bool; NUM_SLOTS],
    last_normalize: bool,
}

impl Default for CabPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(CabPluginParams::default()),
            controller: Arc::new(Mutex::new(None)),
            adapter: None,
            last_slot: 0,
            last_enabled: [true; NUM_SLOTS],
            last_normalize: false,
        }
    }
}

impl Plugin for CabPlugin {
    const NAME: &'static str = "RT IR Cab";
    const VENDOR: &'static str = "Sébastien Lundi";
    const URL: &'static str = "https://codeberg.org/slundi/rt-fft-convolver";
    const EMAIL: &'static str = "slundi@stamus-networks.com";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    // Default is stereo→stereo (most host-compatible); mono→stereo is the
    // guitar use case.
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
    ];

    // Program Change + CC drive slot selection.
    const MIDI_INPUT: MidiConfig = MidiConfig::MidiCCs;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = CabTask;

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn task_executor(&mut self) -> TaskExecutor<Self> {
        let controller = self.controller.clone();
        Box::new(move |task| {
            let mut guard = controller.lock().unwrap();
            let Some(ctrl) = guard.as_mut() else {
                return;
            };
            match task {
                CabTask::Load { slot, path } => {
                    if let Err(e) = ctrl.load_slot_from_path(slot, &path) {
                        nih_error!("failed to load IR into slot {slot}: {e}");
                    }
                    ctrl.activate(slot);
                }
                CabTask::Activate(slot) => ctrl.activate(slot),
                CabTask::Midi(event) => {
                    ctrl.handle_midi(event);
                }
                CabTask::SetEnabled { slot, enabled } => ctrl.set_slot_enabled(slot, enabled),
                CabTask::SetNormalize(on) => ctrl.set_normalize(on),
            }
        })
    }

    fn initialize(
        &mut self,
        audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        let sample_rate = buffer_config.sample_rate as u32;
        let stereo_in = audio_io_layout
            .main_input_channels
            .is_some_and(|c| c.get() >= 2);

        let snapshot = self.params.snapshot();
        let (mut controller, processor) = CabController::new(BLOCK_SIZE, sample_rate, 2, &snapshot);

        // Restore any persisted IR files, then activate the current slot.  This
        // runs off the audio thread (initialization), so file I/O is fine.
        {
            let paths = self.params.ir_paths.lock();
            for (slot, path) in paths.iter().enumerate() {
                if let Some(path) = path
                    && let Err(e) = controller.load_slot_from_path(slot, path)
                {
                    nih_error!("failed to restore IR for slot {slot}: {e}");
                }
            }
        }
        for (i, slot) in self.params.slots.iter().enumerate() {
            controller.set_slot_enabled(i, slot.enabled.value());
        }
        controller.set_normalize(self.params.ir_normalize.value());
        controller.activate(self.params.current_slot.value() as usize);

        let adapter = StereoBlockAdapter::new(processor, stereo_in);
        context.set_latency_samples(adapter.latency_samples() as u32);

        *self.controller.lock().unwrap() = Some(controller);
        self.adapter = Some(adapter);

        self.last_slot = self.params.current_slot.value();
        self.last_enabled = core::array::from_fn(|i| self.params.slots[i].enabled.value());
        self.last_normalize = self.params.ir_normalize.value();
        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // 1. Route incoming MIDI to slot activations (off-thread).
        while let Some(event) = context.next_event() {
            match event {
                NoteEvent::MidiProgramChange { program, .. } => {
                    context.execute_background(CabTask::Midi(SlotEvent::ProgramChange(program)));
                }
                NoteEvent::MidiCC { cc, value, .. } => {
                    let raw = (value * 127.0).round().clamp(0.0, 127.0) as u8;
                    context.execute_background(CabTask::Midi(SlotEvent::ControlChange {
                        controller: cc,
                        value: raw,
                    }));
                }
                _ => {}
            }
        }

        // 2. Dispatch parameter-driven changes (only on an actual move).
        if self.params.ir_normalize.value() != self.last_normalize {
            self.last_normalize = self.params.ir_normalize.value();
            context.execute_background(CabTask::SetNormalize(self.last_normalize));
            // Rebuild the current slot so the change takes effect immediately.
            context
                .execute_background(CabTask::Activate(self.params.current_slot.value() as usize));
        }
        for i in 0..NUM_SLOTS {
            let enabled = self.params.slots[i].enabled.value();
            if enabled != self.last_enabled[i] {
                self.last_enabled[i] = enabled;
                context.execute_background(CabTask::SetEnabled { slot: i, enabled });
                if i == self.params.current_slot.value() as usize {
                    context.execute_background(CabTask::Activate(i));
                }
            }
        }
        let slot = self.params.current_slot.value();
        if slot != self.last_slot {
            self.last_slot = slot;
            context.execute_background(CabTask::Activate(slot as usize));
        }

        // 3. Apply smoothed globals, then render (allocation-free).
        let snapshot = self.params.snapshot();
        if let Some(adapter) = self.adapter.as_mut() {
            adapter.processor_mut().apply_params(&snapshot);
            adapter.process_buffer(buffer.as_slice());
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for CabPlugin {
    const CLAP_ID: &'static str = "org.codeberg.slundi.rt-ir-cab";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("Zero-latency guitar / bass cabinet IR loader");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Mono,
        ClapFeature::Distortion,
    ];
}

impl Vst3Plugin for CabPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"RtIrCabSlundi001";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Distortion];
}

nih_export_clap!(CabPlugin);
nih_export_vst3!(CabPlugin);
