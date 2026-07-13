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
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use nih_plug::prelude::*;
#[cfg(feature = "gui")]
use nih_plug_egui::{EguiState, create_egui_editor, egui, widgets};
use parking_lot::Mutex as PlMutex;
use rt_ir_plugin_core::SlotEvent;
#[cfg(feature = "gui")]
use rt_ir_plugin_core::{MidiActivity, MidiStatus, PadState, pad_state};
#[cfg(feature = "gui")]
use rt_neural_core::{DEFAULT_SWEET_SPOT_DBFS, ModelMetadata};

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
/// `Load` is only constructed under the `gui` feature (drag-drop in the
/// editor); the `#[allow]` keeps the `--features nih` build warning-free.
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

    #[cfg(feature = "gui")]
    #[persist = "editor-state"]
    editor_state: Arc<EguiState>,
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
            #[cfg(feature = "gui")]
            editor_state: EguiState::from_size(640, 380),
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

    /// Bumped on each recognised MIDI event; the editor's LED reads this.
    midi_counter: Arc<AtomicU32>,
    /// Last-block inference cost as a fraction of `BLOCK_SIZE` — used by
    /// the CPU meter (Phase 3).
    cpu_meter: Arc<AtomicU32>,
    /// Peak-hold input level in dBFS, `f32.to_bits()` packed into a u32.
    /// The audio thread writes each block; the editor reads at 30 fps for
    /// the input meter.
    input_meter_dbfs: Arc<AtomicU32>,
    /// `true` while the user is right-click-learning a MIDI CC for
    /// `Current Slot`.  The audio thread captures the next MidiCC event
    /// while this flag is set and clears it.
    learning: Arc<AtomicBool>,
    /// Editor-side mirror of the per-slot [`ModelMetadata`], populated by
    /// the background task executor after a load.
    #[cfg(feature = "gui")]
    metadata_mirror: Arc<Mutex<Vec<Option<ModelMetadata>>>>,
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
            // Prime with silence (-∞ dBFS represented as -120).
            input_meter_dbfs: Arc::new(AtomicU32::new((-120.0f32).to_bits())),
            learning: Arc::new(AtomicBool::new(false)),
            #[cfg(feature = "gui")]
            metadata_mirror: Arc::new(Mutex::new(vec![None; NUM_SLOTS])),
        }
    }
}

/// Mutable per-frame editor state owned by the egui editor closure.
#[cfg(feature = "gui")]
#[derive(Default)]
struct NeuralEditorState {
    midi: MidiActivity,
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

/// Linear blend between two colours (`fraction` in `0..=1`).
#[cfg(feature = "gui")]
fn lerp_color(source: egui::Color32, target: egui::Color32, fraction: f32) -> egui::Color32 {
    let fraction = fraction.clamp(0.0, 1.0);
    let mix = |source_channel: u8, target_channel: u8| {
        (source_channel as f32 + (target_channel as f32 - source_channel as f32) * fraction).round()
            as u8
    };
    egui::Color32::from_rgb(
        mix(source.r(), target.r()),
        mix(source.g(), target.g()),
        mix(source.b(), target.b()),
    )
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

    #[cfg(feature = "gui")]
    fn editor(&mut self, async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        use egui::{Color32, Sense, Stroke, vec2};

        // Violet accent, per UI-lv2.md § design tokens (`accent.neural`).
        const ACCENT: Color32 = Color32::from_rgb(0x7c, 0x5c, 0xff);
        const PANEL: Color32 = Color32::from_rgb(0x1a, 0x1e, 0x24);
        const METER_BG: Color32 = Color32::from_rgb(0x14, 0x18, 0x1e);
        const METER_FILL: Color32 = Color32::from_rgb(0x64, 0x74, 0xa0);
        const SWEET_SPOT_BAND: Color32 = Color32::from_rgba_premultiplied(0x7c, 0x5c, 0xff, 0x40);

        let params = self.params.clone();
        let midi_counter = self.midi_counter.clone();
        let input_meter = self.input_meter_dbfs.clone();
        let learning = self.learning.clone();
        let metadata_mirror = self.metadata_mirror.clone();
        let executor = async_executor;

        create_egui_editor(
            self.params.editor_state.clone(),
            NeuralEditorState::default(),
            |_ctx, _state| {},
            move |ctx, setter, state| {
                // ── MIDI LED decay ────────────────────────────────────────
                let count = midi_counter.load(Ordering::Relaxed);
                if count != state.last_midi {
                    state.last_midi = count;
                    state.midi.trigger(true);
                }
                state.midi.decay(0.08);

                // Escape cancels an in-flight MIDI-learn.
                if ctx.input(|input| input.key_pressed(egui::Key::Escape))
                    && learning.load(Ordering::Relaxed)
                {
                    learning.store(false, Ordering::Relaxed);
                    executor.execute_background(NeuralAmpTask::LearnCc(None));
                }

                egui::CentralPanel::default().show(ctx, |ui| {
                    // ── Header: name · MIDI LED · bypass ───────────────────
                    ui.horizontal(|ui| {
                        ui.heading("RT Neural Amp");
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.add(widgets::ParamSlider::for_param(&params.bypass, setter));
                            let (led_rect, _) =
                                ui.allocate_exact_size(vec2(14.0, 14.0), Sense::hover());
                            let base_color = match state.midi.status() {
                                MidiStatus::Recognised => Color32::from_rgb(0x4e, 0xc3, 0x6a),
                                MidiStatus::Unrecognised => Color32::from_rgb(0xd9, 0x4a, 0x4a),
                                MidiStatus::Idle => Color32::from_gray(0x30),
                            };
                            let led_color = lerp_color(
                                Color32::from_gray(0x30),
                                base_color,
                                state.midi.level(),
                            );
                            ui.painter()
                                .circle_filled(led_rect.center(), 6.0, led_color);
                            ui.label("MIDI");
                        });
                    });
                    ui.separator();

                    // ── Slot rack: 8 pads with drag-drop + right-click learn.
                    let current_slot = params.current_slot.value();
                    let mut pad_rects = [egui::Rect::NOTHING; NUM_SLOTS];
                    ui.horizontal(|ui| {
                        for (slot_index, rect) in pad_rects.iter_mut().enumerate() {
                            let loaded = params.model_paths.lock()[slot_index].is_some();
                            let enabled = params.slots[slot_index].enabled.value();
                            let visual_state =
                                pad_state(loaded, enabled, slot_index as i32 == current_slot);
                            let response = ui.add_sized(
                                vec2(56.0, 44.0),
                                egui::Button::new(format!("{}", slot_index + 1))
                                    .fill(pad_fill(visual_state, ACCENT)),
                            );
                            *rect = response.rect;
                            if response.clicked() {
                                setter.begin_set_parameter(&params.current_slot);
                                setter.set_parameter(&params.current_slot, slot_index as i32);
                                setter.end_set_parameter(&params.current_slot);
                            }
                            // Right-click: MIDI-learn on Current Slot.
                            if response.secondary_clicked() {
                                learning.store(true, Ordering::Relaxed);
                            }
                        }
                    });

                    // ── Drag-and-drop a file onto a pad → load that slot ──
                    let dropped_files = ctx.input(|input| input.raw.dropped_files.clone());
                    if !dropped_files.is_empty()
                        && let Some(cursor) = ctx.input(|input| input.pointer.hover_pos())
                        && let Some(target_slot) =
                            pad_rects.iter().position(|rect| rect.contains(cursor))
                    {
                        for file in dropped_files {
                            if let Some(path) = file.path {
                                params.model_paths.lock()[target_slot] =
                                    Some(path.to_string_lossy().into_owned());
                                executor.execute_background(NeuralAmpTask::Load {
                                    slot: target_slot,
                                    path,
                                });
                                setter.begin_set_parameter(&params.current_slot);
                                setter.set_parameter(&params.current_slot, target_slot as i32);
                                setter.end_set_parameter(&params.current_slot);
                            }
                        }
                    }
                    ui.separator();

                    // ── Input meter with sweet-spot band ──────────────────
                    ui.horizontal(|ui| {
                        ui.label("Input");
                        let (meter_rect, _) = ui.allocate_exact_size(
                            vec2(ui.available_width() - 60.0, 18.0),
                            Sense::hover(),
                        );
                        ui.painter().rect_filled(meter_rect, 2.0, METER_BG);

                        // Sweet-spot band: default -18 dBFS, overridable per
                        // the currently active slot's ModelMetadata.
                        let sweet_dbfs = metadata_mirror
                            .lock()
                            .ok()
                            .and_then(|mirror| {
                                mirror.get(current_slot as usize).and_then(|slot_meta| {
                                    slot_meta.as_ref().map(ModelMetadata::sweet_spot_dbfs)
                                })
                            })
                            .unwrap_or(DEFAULT_SWEET_SPOT_DBFS);

                        // Map dBFS ∈ [-60, 0] → [0, 1] normalised meter x.
                        let normalise = |dbfs: f32| ((dbfs + 60.0) / 60.0).clamp(0.0, 1.0);
                        let sweet_center = normalise(sweet_dbfs);
                        let band_half_width = 3.0 / 60.0; // ±3 dB band
                        let band_left = (sweet_center - band_half_width).clamp(0.0, 1.0);
                        let band_right = (sweet_center + band_half_width).clamp(0.0, 1.0);
                        let band = egui::Rect::from_min_max(
                            egui::pos2(
                                meter_rect.left() + meter_rect.width() * band_left,
                                meter_rect.top(),
                            ),
                            egui::pos2(
                                meter_rect.left() + meter_rect.width() * band_right,
                                meter_rect.bottom(),
                            ),
                        );
                        ui.painter().rect_filled(band, 0.0, SWEET_SPOT_BAND);

                        // Level bar.
                        let level_dbfs = f32::from_bits(input_meter.load(Ordering::Relaxed));
                        let fill_x = normalise(level_dbfs);
                        let fill = egui::Rect::from_min_max(
                            meter_rect.left_top(),
                            egui::pos2(
                                meter_rect.left() + meter_rect.width() * fill_x,
                                meter_rect.bottom(),
                            ),
                        );
                        ui.painter().rect_filled(fill, 2.0, METER_FILL);
                        ui.painter().rect_stroke(
                            meter_rect,
                            2.0,
                            Stroke::new(1.0, Color32::from_gray(0x40)),
                            egui::StrokeKind::Middle,
                        );
                        ui.label(format!("{level_dbfs:>5.1} dB"));
                    });

                    // ── Model metadata panel ──────────────────────────────
                    let metadata_snapshot = metadata_mirror
                        .lock()
                        .ok()
                        .and_then(|mirror| mirror.get(current_slot as usize).cloned())
                        .flatten();
                    ui.group(|ui| {
                        ui.set_min_width(ui.available_width());
                        ui.label(
                            egui::RichText::new(format!("Slot {} · Model info", current_slot + 1))
                                .strong(),
                        );
                        egui::Grid::new("metadata-grid")
                            .num_columns(2)
                            .show(ui, |ui| {
                                let (arch, params_text, sr_text, gear, format_name) =
                                    match metadata_snapshot.as_ref() {
                                        None => (
                                            "—".to_string(),
                                            "—".to_string(),
                                            "—".to_string(),
                                            "empty slot".to_string(),
                                            "—".to_string(),
                                        ),
                                        Some(meta) => (
                                            // Architecture / parameter count come from the
                                            // controller's `slot_architecture` / `slot_parameter_count`
                                            // (populated by the loader task); we surface the
                                            // metadata fields here.
                                            meta.format_name
                                                .clone()
                                                .unwrap_or_else(|| "unknown arch".into()),
                                            "—".into(),
                                            meta.training_sample_rate
                                                .map(|hz| format!("{hz} Hz"))
                                                .unwrap_or_else(|| "unknown".into()),
                                            meta.source_gear_name
                                                .clone()
                                                .unwrap_or_else(|| "unknown".into()),
                                            meta.format_name
                                                .clone()
                                                .unwrap_or_else(|| "unknown".into()),
                                        ),
                                    };
                                ui.label("Architecture");
                                ui.label(arch);
                                ui.end_row();
                                ui.label("Parameters");
                                ui.label(params_text);
                                ui.end_row();
                                ui.label("Train sr");
                                ui.label(sr_text);
                                ui.end_row();
                                ui.label("Source");
                                ui.label(gear);
                                ui.end_row();
                                ui.label("Format");
                                ui.label(format_name);
                                ui.end_row();
                            });
                    });

                    // ── Global controls ────────────────────────────────────
                    egui::Grid::new("neural-controls").show(ui, |ui| {
                        for (label, param) in [
                            ("In Gain", &params.input_gain_db),
                            ("Out Gain", &params.output_gain_db),
                            ("Mix", &params.mix),
                            ("Crossfade", &params.crossfade_ms),
                        ] {
                            ui.label(label);
                            ui.add(widgets::ParamSlider::for_param(param, setter));
                            ui.end_row();
                        }
                        ui.label("Slot Trim");
                        ui.add(widgets::ParamSlider::for_param(
                            &params.slots[current_slot as usize].trim_db,
                            setter,
                        ));
                        ui.end_row();
                    });

                    // ── Learn-mode notice ─────────────────────────────────
                    if learning.load(Ordering::Relaxed) {
                        ui.separator();
                        ui.colored_label(
                            ACCENT,
                            "🎛 MIDI-learn armed — send a CC to bind Current Slot (Esc cancels)",
                        );
                    }
                });
            },
        )
    }

    fn task_executor(&mut self) -> TaskExecutor<Self> {
        let controller = self.controller.clone();
        #[cfg(feature = "gui")]
        let metadata_mirror = self.metadata_mirror.clone();
        Box::new(move |task| {
            let mut guard = controller.lock().unwrap();
            let Some(controller) = guard.as_mut() else {
                return;
            };
            match task {
                NeuralAmpTask::Load { slot, path: _ } => {
                    // Phase 1.5 will invoke `rt_neural_core::loader::load_nam`
                    // (or `load_proteus`) on the file bytes and hand the
                    // resulting `PreparedModel` to `controller.load_slot`.
                    // For Phase 3 we log a clear diagnostic and leave the
                    // slot untouched — the plugin still passes audio through
                    // dry, and the metadata mirror keeps its existing
                    // "unknown" entry for the target slot.
                    nih_error!(
                        "model loader not implemented in Phase 3; slot {slot} left unchanged"
                    );
                    #[cfg(feature = "gui")]
                    {
                        if let Ok(mut mirror) = metadata_mirror.lock() {
                            if let Some(cell) = mirror.get_mut(slot) {
                                *cell = controller.slot_metadata(slot);
                            }
                        }
                    }
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
        //      While `learning` is set, the next MidiCC binds to
        //      `Current Slot` instead of driving a slot switch.
        while let Some(event) = context.next_event() {
            match event {
                NoteEvent::MidiProgramChange { program, .. } => {
                    context
                        .execute_background(NeuralAmpTask::Midi(SlotEvent::ProgramChange(program)));
                    self.midi_counter.fetch_add(1, Ordering::Relaxed);
                }
                NoteEvent::MidiCC { cc, value, .. } => {
                    if self.learning.swap(false, Ordering::Relaxed) {
                        context.execute_background(NeuralAmpTask::LearnCc(Some(cc)));
                    } else {
                        let raw = (value * 127.0).round().clamp(0.0, 127.0) as u8;
                        context.execute_background(NeuralAmpTask::Midi(SlotEvent::ControlChange {
                            controller: cc,
                            value: raw,
                        }));
                    }
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
                // Compute the input RMS **before** the adapter overwrites
                // the channel with its output.  Store as dBFS packed in the
                // atomic so the editor can render an input meter.
                let sample_count = channel.len().max(1) as f32;
                let sum_of_squares: f32 = channel.iter().map(|sample| sample * sample).sum();
                let rms = (sum_of_squares / sample_count).sqrt().max(1.0e-6);
                let dbfs = 20.0 * rms.log10();
                self.input_meter_dbfs
                    .store(dbfs.to_bits(), Ordering::Relaxed);

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
