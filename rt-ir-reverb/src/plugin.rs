//! `nih-plug` wrapper for the reverb (compiled only with the `nih` feature).
//!
//! Thin framework layer over the tested [`engine`](crate::engine): declares the
//! automatable parameters, routes MIDI + parameter changes into the off-thread
//! [`ReverbController`], and runs the [`StereoBlockAdapter`] on the audio
//! thread.  See `rt-ir-cab`'s wrapper for the shared threading rationale.
//!
//! Phase-3/4 follow-ups: drag-and-drop IR assignment + MIDI-learn UI + CPU-meter
//! readout (Phase 3); host validation and the bundled room-IR pack that the
//! presets pair with (Phase 4/5).

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use nih_plug::prelude::*;
use parking_lot::Mutex as PlMutex;
use rt_ir_plugin_core::SlotEvent;

use crate::block::StereoBlockAdapter;
use crate::engine::ReverbController;
use crate::params::{
    CROSSFADE_MS_RANGE, FADER_DB_RANGE, HIGH_CUT_RANGE, IR_LENGTH_RANGE, LOW_CUT_RANGE, NUM_SLOTS,
    PREDELAY_MS_RANGE, ReverbParams, WIDTH_RANGE,
};

/// Fixed internal render block; the plugin reports `BLOCK_SIZE` latency.
const BLOCK_SIZE: usize = 512;

/// Off-thread work dispatched to the background executor.
pub enum ReverbTask {
    /// Load a WAV into a slot (Phase-3 UI caller; init restores paths directly).
    #[allow(dead_code)]
    Load { slot: usize, path: PathBuf },
    /// Rebuild + hand off the reverb for a slot.
    Activate(usize),
    /// Route a MIDI event to a slot.
    Midi(SlotEvent),
    /// Enable/disable a slot.
    SetEnabled { slot: usize, enabled: bool },
    /// Set IR reversal for future activations.
    SetReverse(bool),
    /// Set the IR length fraction for future activations.
    SetLength(f32),
}

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

#[derive(Params)]
struct ReverbPluginParams {
    #[id = "slot"]
    current_slot: IntParam,
    #[id = "xfade"]
    crossfade_ms: FloatParam,
    #[id = "predly"]
    predelay_ms: FloatParam,
    #[id = "dry"]
    dry_db: FloatParam,
    #[id = "wet"]
    wet_db: FloatParam,
    #[id = "width"]
    width: FloatParam,
    #[id = "locut"]
    low_cut: FloatParam,
    #[id = "hicut"]
    high_cut: FloatParam,
    #[id = "irrev"]
    ir_reverse: BoolParam,
    #[id = "irlen"]
    ir_length: FloatParam,
    #[id = "byp"]
    bypass: BoolParam,

    #[nested(array, group = "Slots")]
    slots: [SlotParams; NUM_SLOTS],

    #[persist = "ir_paths"]
    ir_paths: PlMutex<Vec<Option<String>>>,
}

impl Default for ReverbPluginParams {
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
                100.0,
                FloatRange::Linear {
                    min: CROSSFADE_MS_RANGE.0,
                    max: CROSSFADE_MS_RANGE.1,
                },
            )
            .with_unit(" ms"),
            predelay_ms: FloatParam::new(
                "Pre-Delay",
                0.0,
                FloatRange::Linear {
                    min: PREDELAY_MS_RANGE.0,
                    max: PREDELAY_MS_RANGE.1,
                },
            )
            .with_unit(" ms"),
            dry_db: FloatParam::new(
                "Dry",
                0.0,
                FloatRange::Linear {
                    min: FADER_DB_RANGE.0,
                    max: FADER_DB_RANGE.1,
                },
            )
            .with_unit(" dB"),
            wet_db: FloatParam::new(
                "Wet",
                0.0,
                FloatRange::Linear {
                    min: FADER_DB_RANGE.0,
                    max: FADER_DB_RANGE.1,
                },
            )
            .with_unit(" dB"),
            width: FloatParam::new(
                "Stereo Width",
                1.0,
                FloatRange::Linear {
                    min: WIDTH_RANGE.0,
                    max: WIDTH_RANGE.1,
                },
            )
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
            ir_reverse: BoolParam::new("IR Reverse", false),
            ir_length: FloatParam::new(
                "IR Length",
                1.0,
                FloatRange::Linear {
                    min: IR_LENGTH_RANGE.0,
                    max: IR_LENGTH_RANGE.1,
                },
            )
            .with_unit(" %")
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage()),
            bypass: BoolParam::new("Bypass", false)
                .with_value_to_string(formatters::v2s_bool_bypass()),
            slots: core::array::from_fn(|_| SlotParams::default()),
            ir_paths: PlMutex::new(vec![None; NUM_SLOTS]),
        }
    }
}

impl ReverbPluginParams {
    fn snapshot(&self) -> ReverbParams {
        ReverbParams {
            current_slot: self.current_slot.value() as usize,
            crossfade_ms: self.crossfade_ms.value(),
            predelay_ms: self.predelay_ms.value(),
            dry_db: self.dry_db.value(),
            wet_db: self.wet_db.value(),
            width: self.width.value(),
            low_cut_hz: self.low_cut.value(),
            high_cut_hz: self.high_cut.value(),
            ir_reverse: self.ir_reverse.value(),
            ir_length: self.ir_length.value(),
            bypass: self.bypass.value(),
            slot_enabled: core::array::from_fn(|i| self.slots[i].enabled.value()),
        }
    }
}

/// The reverb plugin instance.
pub struct ReverbPlugin {
    params: Arc<ReverbPluginParams>,
    controller: Arc<Mutex<Option<ReverbController>>>,
    adapter: Option<StereoBlockAdapter>,

    last_slot: i32,
    last_enabled: [bool; NUM_SLOTS],
    last_reverse: bool,
    last_length: f32,
}

impl Default for ReverbPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(ReverbPluginParams::default()),
            controller: Arc::new(Mutex::new(None)),
            adapter: None,
            last_slot: 0,
            last_enabled: [true; NUM_SLOTS],
            last_reverse: false,
            last_length: 1.0,
        }
    }
}

impl Plugin for ReverbPlugin {
    const NAME: &'static str = "RT IR Reverb";
    const VENDOR: &'static str = "Sébastien Lundi";
    const URL: &'static str = "https://codeberg.org/slundi/rt-fft-convolver";
    const EMAIL: &'static str = "slundi@stamus-networks.com";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    // Stereo in → stereo out.
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),
        ..AudioIOLayout::const_default()
    }];

    const MIDI_INPUT: MidiConfig = MidiConfig::MidiCCs;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = ReverbTask;

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
                ReverbTask::Load { slot, path } => {
                    if let Err(e) = ctrl.load_slot_from_path(slot, &path) {
                        nih_error!("failed to load IR into slot {slot}: {e}");
                    }
                    ctrl.activate(slot);
                }
                ReverbTask::Activate(slot) => ctrl.activate(slot),
                ReverbTask::Midi(event) => {
                    ctrl.handle_midi(event);
                }
                ReverbTask::SetEnabled { slot, enabled } => ctrl.set_slot_enabled(slot, enabled),
                ReverbTask::SetReverse(on) => ctrl.set_reverse(on),
                ReverbTask::SetLength(frac) => ctrl.set_length(frac),
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
        let (mut controller, processor) = ReverbController::new(BLOCK_SIZE, sample_rate, &snapshot);

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
        controller.set_reverse(self.params.ir_reverse.value());
        controller.set_length(self.params.ir_length.value());
        controller.activate(self.params.current_slot.value() as usize);

        let adapter = StereoBlockAdapter::new(processor);
        context.set_latency_samples(adapter.latency_samples() as u32);

        *self.controller.lock().unwrap() = Some(controller);
        self.adapter = Some(adapter);

        self.last_slot = self.params.current_slot.value();
        self.last_enabled = core::array::from_fn(|i| self.params.slots[i].enabled.value());
        self.last_reverse = self.params.ir_reverse.value();
        self.last_length = self.params.ir_length.value();
        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // MIDI → slot activation (off-thread).
        while let Some(event) = context.next_event() {
            match event {
                NoteEvent::MidiProgramChange { program, .. } => {
                    context.execute_background(ReverbTask::Midi(SlotEvent::ProgramChange(program)));
                }
                NoteEvent::MidiCC { cc, value, .. } => {
                    let raw = (value * 127.0).round().clamp(0.0, 127.0) as u8;
                    context.execute_background(ReverbTask::Midi(SlotEvent::ControlChange {
                        controller: cc,
                        value: raw,
                    }));
                }
                _ => {}
            }
        }

        // IR-shaping changes rebuild the current slot off-thread.
        let current = self.params.current_slot.value();
        let reverse = self.params.ir_reverse.value();
        if reverse != self.last_reverse {
            self.last_reverse = reverse;
            context.execute_background(ReverbTask::SetReverse(reverse));
            context.execute_background(ReverbTask::Activate(current as usize));
        }
        let length = self.params.ir_length.value();
        if (length - self.last_length).abs() > f32::EPSILON {
            self.last_length = length;
            context.execute_background(ReverbTask::SetLength(length));
            context.execute_background(ReverbTask::Activate(current as usize));
        }
        for i in 0..NUM_SLOTS {
            let enabled = self.params.slots[i].enabled.value();
            if enabled != self.last_enabled[i] {
                self.last_enabled[i] = enabled;
                context.execute_background(ReverbTask::SetEnabled { slot: i, enabled });
                if i == current as usize {
                    context.execute_background(ReverbTask::Activate(i));
                }
            }
        }
        if current != self.last_slot {
            self.last_slot = current;
            context.execute_background(ReverbTask::Activate(current as usize));
        }

        // Apply smoothed globals, then render (allocation-free).
        let snapshot = self.params.snapshot();
        if let Some(adapter) = self.adapter.as_mut() {
            adapter.processor_mut().apply_params(&snapshot);
            adapter.process_buffer(buffer.as_slice());
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for ReverbPlugin {
    const CLAP_ID: &'static str = "org.codeberg.slundi.rt-ir-reverb";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("True-stereo convolution reverb");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Reverb,
    ];
}

impl Vst3Plugin for ReverbPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"RtIrReverbSlundi";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Reverb];
}

nih_export_clap!(ReverbPlugin);
nih_export_vst3!(ReverbPlugin);
