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
#[cfg(feature = "gui")]
use nih_plug_egui::{EguiState, create_egui_editor, egui, widgets};
use parking_lot::Mutex as PlMutex;
use rt_ir_plugin_core::SlotEvent;

#[cfg(feature = "gui")]
use rt_ir_plugin_core::{MidiStatus, PadState, WaveformSummary, pad_state};
#[cfg(feature = "gui")]
use std::sync::atomic::{AtomicU32, Ordering};

use crate::block::StereoBlockAdapter;
use crate::engine::CabController;
use crate::params::{
    CROSSFADE_MS_RANGE, CabParams, GAIN_DB_RANGE, HIGH_CUT_RANGE, LOW_CUT_RANGE, NUM_SLOTS,
};

/// Fixed internal render block.  The plugin reports `BLOCK_SIZE` samples of
/// latency (see [`StereoBlockAdapter`]).
const BLOCK_SIZE: usize = 512;

/// Waveform preview resolution (buckets across the frame width).
#[cfg(feature = "gui")]
const WAVEFORM_BUCKETS: usize = 256;

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

    #[cfg(feature = "gui")]
    #[persist = "editor-state"]
    editor_state: Arc<EguiState>,
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
            #[cfg(feature = "gui")]
            editor_state: EguiState::from_size(640, 320),
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

    /// Per-slot waveform previews for the editor, filled off-thread on load.
    #[cfg(feature = "gui")]
    waveforms: Arc<Mutex<Vec<Option<WaveformSummary>>>>,
    /// Bumped on each recognised MIDI event so the editor LED can blink.
    #[cfg(feature = "gui")]
    midi_counter: Arc<AtomicU32>,
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
            #[cfg(feature = "gui")]
            waveforms: Arc::new(Mutex::new(vec![None; NUM_SLOTS])),
            #[cfg(feature = "gui")]
            midi_counter: Arc::new(AtomicU32::new(0)),
        }
    }
}

/// Mutable per-frame editor state owned by the egui editor closure.
#[cfg(feature = "gui")]
#[derive(Default)]
struct CabEditorState {
    midi: rt_ir_plugin_core::MidiActivity,
    last_midi: u32,
}

/// Fill colour for a slot pad in each visual state.
#[cfg(feature = "gui")]
fn pad_fill(state: PadState, accent: egui::Color32) -> egui::Color32 {
    match state {
        PadState::Active => accent,
        PadState::Filled => egui::Color32::from_rgb(0x23, 0x28, 0x30),
        PadState::Disabled => egui::Color32::from_rgb(0x20, 0x22, 0x26),
        PadState::Empty => egui::Color32::from_rgb(0x1a, 0x1e, 0x24),
    }
}

/// Linear blend between two colours (`t` in `0..=1`).
#[cfg(feature = "gui")]
fn lerp_color(a: egui::Color32, b: egui::Color32, t: f32) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);
    let mix = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    egui::Color32::from_rgb(mix(a.r(), b.r()), mix(a.g(), b.g()), mix(a.b(), b.b()))
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

    #[cfg(feature = "gui")]
    fn editor(&mut self, async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        use egui::{Color32, Sense, Stroke, pos2, vec2};

        // Amber accent (docs/UI-lv2.md § cab visual identity).
        const ACCENT: Color32 = Color32::from_rgb(0xe0, 0xa1, 0x44);
        const PANEL: Color32 = Color32::from_rgb(0x1a, 0x1e, 0x24);

        let params = self.params.clone();
        let waveforms = self.waveforms.clone();
        let midi_counter = self.midi_counter.clone();
        let executor = async_executor;

        create_egui_editor(
            self.params.editor_state.clone(),
            CabEditorState::default(),
            |_ctx, _state| {},
            move |ctx, setter, state| {
                let count = midi_counter.load(Ordering::Relaxed);
                if count != state.last_midi {
                    state.last_midi = count;
                    state.midi.trigger(true);
                }
                state.midi.decay(0.08);

                egui::CentralPanel::default().show(ctx, |ui| {
                    // ── Header: name · MIDI LED · bypass ──────────────────
                    ui.horizontal(|ui| {
                        ui.heading("RT IR Cab");
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.add(widgets::ParamSlider::for_param(&params.bypass, setter));
                            let (led, _) = ui.allocate_exact_size(vec2(14.0, 14.0), Sense::hover());
                            let base = match state.midi.status() {
                                MidiStatus::Recognised => Color32::from_rgb(0x4e, 0xc3, 0x6a),
                                MidiStatus::Unrecognised => Color32::from_rgb(0xd9, 0x4a, 0x4a),
                                MidiStatus::Idle => Color32::from_gray(0x30),
                            };
                            let col =
                                lerp_color(Color32::from_gray(0x30), base, state.midi.level());
                            ui.painter().circle_filled(led.center(), 6.0, col);
                            ui.label("MIDI");
                        });
                    });
                    ui.separator();

                    // ── Slot rack: 8 pads ─────────────────────────────────
                    let current = params.current_slot.value();
                    let mut pad_rects = [egui::Rect::NOTHING; NUM_SLOTS];
                    ui.horizontal(|ui| {
                        for (i, rect) in pad_rects.iter_mut().enumerate() {
                            let loaded = params.ir_paths.lock()[i].is_some();
                            let enabled = params.slots[i].enabled.value();
                            let st = pad_state(loaded, enabled, i as i32 == current);
                            let resp = ui.add_sized(
                                vec2(56.0, 44.0),
                                egui::Button::new(format!("{}", i + 1)).fill(pad_fill(st, ACCENT)),
                            );
                            *rect = resp.rect;
                            if resp.clicked() {
                                setter.begin_set_parameter(&params.current_slot);
                                setter.set_parameter(&params.current_slot, i as i32);
                                setter.end_set_parameter(&params.current_slot);
                            }
                        }
                    });

                    // ── Drag-and-drop a file onto a pad → load that slot ──
                    let dropped = ctx.input(|i| i.raw.dropped_files.clone());
                    if !dropped.is_empty()
                        && let Some(pos) = ctx.input(|i| i.pointer.hover_pos())
                        && let Some(slot) = pad_rects.iter().position(|r| r.contains(pos))
                    {
                        for f in dropped {
                            if let Some(path) = f.path {
                                params.ir_paths.lock()[slot] =
                                    Some(path.to_string_lossy().into_owned());
                                executor.execute_background(CabTask::Load { slot, path });
                                setter.begin_set_parameter(&params.current_slot);
                                setter.set_parameter(&params.current_slot, slot as i32);
                                setter.end_set_parameter(&params.current_slot);
                            }
                        }
                    }

                    // ── Waveform preview of the active slot ───────────────
                    let (frame, _) =
                        ui.allocate_exact_size(vec2(ui.available_width(), 96.0), Sense::hover());
                    ui.painter().rect_filled(frame, 2.0, PANEL);
                    if let Some(Some(w)) = waveforms.lock().unwrap().get(current as usize) {
                        let n = w.peaks.len().max(1);
                        let cy = frame.center().y;
                        let half = frame.height() * 0.5;
                        for (k, &(lo, hi)) in w.peaks.iter().enumerate() {
                            let x = frame.left() + frame.width() * (k as f32 / n as f32);
                            ui.painter().line_segment(
                                [pos2(x, cy - hi * half), pos2(x, cy - lo * half)],
                                Stroke::new(1.0, ACCENT),
                            );
                        }
                    }
                    ui.separator();

                    // ── Global controls ───────────────────────────────────
                    egui::Grid::new("cab-controls").show(ui, |ui| {
                        for (label, param) in [
                            ("In Gain", &params.input_gain_db),
                            ("Low Cut", &params.low_cut),
                            ("High Cut", &params.high_cut),
                            ("Mix", &params.mix),
                            ("Out Gain", &params.output_gain_db),
                            ("Crossfade", &params.crossfade_ms),
                        ] {
                            ui.label(label);
                            ui.add(widgets::ParamSlider::for_param(param, setter));
                            ui.end_row();
                        }
                        ui.label("Normalize");
                        ui.add(widgets::ParamSlider::for_param(
                            &params.ir_normalize,
                            setter,
                        ));
                        ui.end_row();
                    });
                });
            },
        )
    }

    fn task_executor(&mut self) -> TaskExecutor<Self> {
        let controller = self.controller.clone();
        #[cfg(feature = "gui")]
        let waveforms = self.waveforms.clone();
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
                    #[cfg(feature = "gui")]
                    {
                        waveforms.lock().unwrap()[slot] = ctrl.waveform(slot, WAVEFORM_BUCKETS);
                    }
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

        // Pre-compute waveform previews for the restored slots.
        #[cfg(feature = "gui")]
        {
            let mut wf = self.waveforms.lock().unwrap();
            for (i, slot) in wf.iter_mut().enumerate() {
                *slot = controller.waveform(i, WAVEFORM_BUCKETS);
            }
        }

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
                    #[cfg(feature = "gui")]
                    self.midi_counter.fetch_add(1, Ordering::Relaxed);
                }
                NoteEvent::MidiCC { cc, value, .. } => {
                    let raw = (value * 127.0).round().clamp(0.0, 127.0) as u8;
                    context.execute_background(CabTask::Midi(SlotEvent::ControlChange {
                        controller: cc,
                        value: raw,
                    }));
                    #[cfg(feature = "gui")]
                    self.midi_counter.fetch_add(1, Ordering::Relaxed);
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
