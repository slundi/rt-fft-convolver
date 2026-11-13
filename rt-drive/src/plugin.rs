//! `nih-plug` wrapper for `rt-drive` (compiled only with the `nih` feature).
//!
//! A thin framework layer over the tested, framework-agnostic
//! [`engine`](crate::engine): it declares the automatable parameters, routes
//! MIDI + structural parameter changes into the off-thread
//! [`DriveController`], and runs the [`DriveProcessor`] on the audio thread.
//!
//! # Threading
//!
//! - The [`DriveController`] lives behind a `Mutex` shared with the `nih-plug`
//!   background-task executor.  Rebuilding the per-channel stages (which
//!   reallocates the resampler) happens there, never on the audio thread.
//! - `process()` polls the lock-free handoff for a freshly-built payload,
//!   applies the smoothed continuous parameters and renders — allocation-free.
//!
//! # MIDI
//!
//! - **Program Change** selects the pedal `Model` (routed through
//!   [`program_to_slot`]).
//! - **CC** control of `Drive` / `Level` is provided by exposing every
//!   continuous control as an automatable parameter, which hosts bind to any
//!   MIDI CC (host-side MIDI learn) — the idiomatic `nih-plug` path.
//!
//! # Phase-3/4 follow-ups
//!
//! - `nih-plug-egui` editor (transfer-curve view, VU meters, oversampling
//!   badge, A/B) — Phase 3.
//! - User-pedal directory discovery + hot-swap — Phase 2 leftover.
//! - `lv2lint` / `clap-validator` / `pluginval` host validation — Phase 4.

use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use nih_plug::prelude::*;
#[cfg(feature = "gui")]
use nih_plug_egui::{EguiState, create_egui_editor, egui, widgets};
#[cfg(feature = "gui")]
use rt_drive_core::PedalModel;
use rt_drive_core::shipped_pedals;
use rt_ir_plugin_core::program_to_slot;

use crate::engine::{DriveController, DriveProcessor};
use crate::params::{BIAS_RANGE, DriveParams, LEVEL_DB_RANGE, OversampleChoice, TONE_DB_RANGE};
#[cfg(feature = "gui")]
use crate::ui::{
    AbSlot, AbState, LevelMeter, clipboard_string, format_cpu, format_oversampling, transfer_curve,
};
use crate::ui::{SILENCE_DBFS, peak_dbfs};

/// Off-thread work dispatched to the background executor (control thread only).
pub enum DriveTask {
    /// Select a pedal by bank index.
    SetModel(usize),
    /// Change the oversampling factor.
    SetOversampling(OversampleChoice),
}

/// `nih-plug` oversampling enum, mirroring [`OversampleChoice`] (that type
/// stays framework-agnostic; this one carries the `#[derive(Enum)]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
enum OversampleParam {
    #[name = "Preset"]
    Preset,
    #[name = "1×"]
    X1,
    #[name = "2×"]
    X2,
    #[name = "4×"]
    X4,
    #[name = "8×"]
    X8,
    #[name = "16×"]
    X16,
}

impl OversampleParam {
    fn to_choice(self) -> OversampleChoice {
        match self {
            OversampleParam::Preset => OversampleChoice::Preset,
            OversampleParam::X1 => OversampleChoice::X1,
            OversampleParam::X2 => OversampleChoice::X2,
            OversampleParam::X4 => OversampleChoice::X4,
            OversampleParam::X8 => OversampleChoice::X8,
            OversampleParam::X16 => OversampleChoice::X16,
        }
    }

    /// Map a framework-agnostic [`OversampleChoice`] back to the param enum
    /// (used when A/B recall pushes a snapshot to the host).
    #[cfg(feature = "gui")]
    fn from_choice(choice: OversampleChoice) -> Self {
        match choice {
            OversampleChoice::Preset => OversampleParam::Preset,
            OversampleChoice::X1 => OversampleParam::X1,
            OversampleChoice::X2 => OversampleParam::X2,
            OversampleChoice::X4 => OversampleParam::X4,
            OversampleChoice::X8 => OversampleParam::X8,
            OversampleChoice::X16 => OversampleParam::X16,
        }
    }
}

/// Host-visible parameters.
#[derive(Params)]
struct DrivePluginParams {
    #[id = "model"]
    model: IntParam,
    #[id = "drive"]
    drive: FloatParam,
    #[id = "level"]
    level_db: FloatParam,
    #[id = "bass"]
    bass_db: FloatParam,
    #[id = "mid"]
    mid_db: FloatParam,
    #[id = "treble"]
    treble_db: FloatParam,
    #[id = "mix"]
    mix: FloatParam,
    #[id = "os"]
    oversampling: EnumParam<OversampleParam>,
    #[id = "bias"]
    bias: FloatParam,
    #[id = "byp"]
    bypass: BoolParam,

    #[cfg(feature = "gui")]
    #[persist = "editor-state"]
    editor_state: Arc<EguiState>,
}

impl Default for DrivePluginParams {
    fn default() -> Self {
        // The Model dropdown enumerates the shipped bank by index; the
        // formatter turns each index into its pedal name.  A parse failure
        // (guarded by a core test) degrades to a single "Default" entry.
        let names: Vec<String> = shipped_pedals()
            .map(|pedals| pedals.into_iter().map(|pedal| pedal.name).collect())
            .unwrap_or_else(|_| vec!["Default".to_string()]);
        let model_max = names.len().saturating_sub(1) as i32;
        let name_lookup = names.clone();

        Self {
            model: IntParam::new(
                "Model",
                0,
                IntRange::Linear {
                    min: 0,
                    max: model_max,
                },
            )
            .with_value_to_string(Arc::new(move |index| {
                name_lookup
                    .get(index as usize)
                    .cloned()
                    .unwrap_or_else(|| "—".to_string())
            })),
            drive: FloatParam::new("Drive", 0.5, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_unit(" %")
                .with_value_to_string(formatters::v2s_f32_percentage(0))
                .with_string_to_value(formatters::s2v_f32_percentage()),
            level_db: FloatParam::new(
                "Level",
                0.0,
                FloatRange::Linear {
                    min: LEVEL_DB_RANGE.0,
                    max: LEVEL_DB_RANGE.1,
                },
            )
            .with_unit(" dB"),
            bass_db: tone_param("Bass"),
            mid_db: tone_param("Mid"),
            treble_db: tone_param("Treble"),
            mix: FloatParam::new("Mix", 1.0, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_unit(" %")
                .with_value_to_string(formatters::v2s_f32_percentage(0))
                .with_string_to_value(formatters::s2v_f32_percentage()),
            oversampling: EnumParam::new("Oversampling", OversampleParam::Preset),
            bias: FloatParam::new(
                "Bias",
                0.0,
                FloatRange::Linear {
                    min: BIAS_RANGE.0,
                    max: BIAS_RANGE.1,
                },
            ),
            bypass: BoolParam::new("Bypass", false)
                .with_value_to_string(formatters::v2s_bool_bypass()),
            #[cfg(feature = "gui")]
            editor_state: EguiState::from_size(560, 480),
        }
    }
}

/// A `-12..+12 dB` tone-band parameter.
fn tone_param(name: &str) -> FloatParam {
    FloatParam::new(
        name,
        0.0,
        FloatRange::Linear {
            min: TONE_DB_RANGE.0,
            max: TONE_DB_RANGE.1,
        },
    )
    .with_unit(" dB")
}

impl DrivePluginParams {
    /// Snapshot into the framework-agnostic model.
    fn snapshot(&self) -> DriveParams {
        DriveParams {
            model_index: self.model.value() as usize,
            drive: self.drive.value(),
            level_db: self.level_db.value(),
            bass_db: self.bass_db.value(),
            mid_db: self.mid_db.value(),
            treble_db: self.treble_db.value(),
            mix: self.mix.value(),
            oversampling: self.oversampling.value().to_choice(),
            bias: self.bias.value(),
            bypass: self.bypass.value(),
        }
    }
}

/// The drive plugin instance.
pub struct DrivePlugin {
    params: Arc<DrivePluginParams>,
    /// Shared with the background-task executor; built in `initialize`.
    controller: Arc<Mutex<Option<DriveController>>>,
    /// Audio-thread renderer; built in `initialize`.
    processor: Option<DriveProcessor>,

    // Change detection so we only rebuild on an actual move.
    last_model: i32,
    last_oversampling: OversampleParam,

    /// Last-block CPU load as a fraction of the block's wall-clock budget,
    /// `f32.to_bits()` packed — read by the CPU badge.
    cpu_meter: Arc<AtomicU32>,
    /// Per-block input peak in dBFS, `f32.to_bits()` packed (editor VU meter).
    input_meter_dbfs: Arc<AtomicU32>,
    /// Per-block output peak in dBFS, `f32.to_bits()` packed (editor VU meter).
    output_meter_dbfs: Arc<AtomicU32>,
}

impl Default for DrivePlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(DrivePluginParams::default()),
            controller: Arc::new(Mutex::new(None)),
            processor: None,
            last_model: 0,
            last_oversampling: OversampleParam::Preset,
            cpu_meter: Arc::new(AtomicU32::new(0)),
            // Prime with silence (−120 dBFS).
            input_meter_dbfs: Arc::new(AtomicU32::new((-120.0f32).to_bits())),
            output_meter_dbfs: Arc::new(AtomicU32::new((-120.0f32).to_bits())),
        }
    }
}

/// Per-frame editor state owned by the egui closure.
#[cfg(feature = "gui")]
struct DriveEditorState {
    input_meter: LevelMeter,
    output_meter: LevelMeter,
    ab: AbState,
}

#[cfg(feature = "gui")]
impl DriveEditorState {
    fn new(initial: DriveParams) -> Self {
        Self {
            input_meter: LevelMeter::new(),
            output_meter: LevelMeter::new(),
            ab: AbState::new(initial),
        }
    }
}

/// Draw a horizontal VU bar for `dbfs`, mapping `[-60, 0] dBFS → [0, 1]`.
#[cfg(feature = "gui")]
fn draw_meter(
    ui: &mut egui::Ui,
    label: &str,
    dbfs: f32,
    background: egui::Color32,
    fill: egui::Color32,
) {
    ui.horizontal(|ui| {
        ui.monospace(label);
        let width = (ui.available_width() - 56.0).max(40.0);
        let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 14.0), egui::Sense::hover());
        ui.painter().rect_filled(rect, 2.0, background);
        let normalised = ((dbfs + 60.0) / 60.0).clamp(0.0, 1.0);
        let fill_rect = egui::Rect::from_min_max(
            rect.left_top(),
            egui::pos2(rect.left() + rect.width() * normalised, rect.bottom()),
        );
        ui.painter().rect_filled(fill_rect, 2.0, fill);
        ui.monospace(format!("{dbfs:>5.0}"));
    });
}

/// Push a full [`DriveParams`] snapshot to the host (A/B recall).
#[cfg(feature = "gui")]
fn apply_params(setter: &ParamSetter, params: &DrivePluginParams, values: &DriveParams) {
    set_param(setter, &params.model, values.model_index as i32);
    set_param(setter, &params.drive, values.drive);
    set_param(setter, &params.level_db, values.level_db);
    set_param(setter, &params.bass_db, values.bass_db);
    set_param(setter, &params.mid_db, values.mid_db);
    set_param(setter, &params.treble_db, values.treble_db);
    set_param(setter, &params.mix, values.mix);
    set_param(setter, &params.bias, values.bias);
    set_param(setter, &params.bypass, values.bypass);
    set_param(
        setter,
        &params.oversampling,
        OversampleParam::from_choice(values.oversampling),
    );
}

/// Set one parameter with a complete begin/set/end automation gesture.
#[cfg(feature = "gui")]
fn set_param<P: Param>(setter: &ParamSetter, param: &P, value: P::Plain) {
    setter.begin_set_parameter(param);
    setter.set_parameter(param, value);
    setter.end_set_parameter(param);
}

impl Plugin for DrivePlugin {
    const NAME: &'static str = "RT Drive";
    const VENDOR: &'static str = "Sébastien Lundi";
    const URL: &'static str = "https://codeberg.org/slundi/rt-fft-convolver";
    const EMAIL: &'static str = "slundi@stamus-networks.com";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    // Mono → mono and stereo → stereo (independent per channel), per Phase 2.
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(1),
            ..AudioIOLayout::const_default()
        },
    ];

    const MIDI_INPUT: MidiConfig = MidiConfig::MidiCCs;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = DriveTask;

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    #[cfg(feature = "gui")]
    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        use egui::Color32;

        // Orange accent for a drive/distortion pedal.
        const ACCENT: Color32 = Color32::from_rgb(0xff, 0x8c, 0x32);
        const PANEL: Color32 = Color32::from_rgb(0x16, 0x18, 0x1c);
        const CURVE_GRID: Color32 = Color32::from_rgb(0x2a, 0x2e, 0x36);
        const METER_BG: Color32 = Color32::from_rgb(0x14, 0x16, 0x1a);
        const IN_FILL: Color32 = Color32::from_rgb(0x5a, 0x74, 0xa0);
        const OUT_FILL: Color32 = Color32::from_rgb(0xff, 0x8c, 0x32);

        let params = self.params.clone();
        let cpu_meter = self.cpu_meter.clone();
        let input_meter = self.input_meter_dbfs.clone();
        let output_meter = self.output_meter_dbfs.clone();
        let pedals: Vec<PedalModel> = shipped_pedals().unwrap_or_default();

        create_egui_editor(
            self.params.editor_state.clone(),
            DriveEditorState::new(self.params.snapshot()),
            |_ctx, _state| {},
            move |ctx, setter, state| {
                let snapshot = params.snapshot();
                // Keep the active A/B slot in sync with the live controls.
                state.ab.store(snapshot);

                egui::CentralPanel::default().show(ctx, |ui| {
                    // ── Header: name + bypass ─────────────────────────────
                    ui.horizontal(|ui| {
                        ui.heading("RT Drive");
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.add(widgets::ParamSlider::for_param(&params.bypass, setter));
                        });
                    });
                    ui.separator();

                    // ── Model + oversampling / CPU badge ──────────────────
                    ui.horizontal(|ui| {
                        ui.label("Model");
                        ui.add(widgets::ParamSlider::for_param(&params.model, setter));
                    });
                    let model_index = snapshot.model_index;
                    let pedal = pedals.get(model_index);
                    let factor = pedal
                        .map(|pedal| snapshot.oversampling.factor(pedal.default_oversampling))
                        .unwrap_or(1);
                    let cpu = f32::from_bits(cpu_meter.load(Ordering::Relaxed));
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(format!("OS {}", format_oversampling(factor)))
                                .color(ACCENT),
                        );
                        ui.separator();
                        ui.label(format!("CPU {}", format_cpu(cpu)));
                    });

                    // ── Transfer-curve visualiser ─────────────────────────
                    if let Some(pedal) = pedal {
                        let effective_gain = pedal.input_gain_db + snapshot.drive_offset_db();
                        let curve = transfer_curve(pedal.shaper, effective_gain, 96);
                        let (rect, _) = ui.allocate_exact_size(
                            egui::vec2(ui.available_width(), 130.0),
                            egui::Sense::hover(),
                        );
                        ui.painter().rect_filled(rect, 4.0, PANEL);
                        // Zero axes.
                        ui.painter().line_segment(
                            [
                                egui::pos2(rect.center().x, rect.top()),
                                egui::pos2(rect.center().x, rect.bottom()),
                            ],
                            egui::Stroke::new(1.0, CURVE_GRID),
                        );
                        ui.painter().line_segment(
                            [
                                egui::pos2(rect.left(), rect.center().y),
                                egui::pos2(rect.right(), rect.center().y),
                            ],
                            egui::Stroke::new(1.0, CURVE_GRID),
                        );
                        let points: Vec<egui::Pos2> = curve
                            .iter()
                            .map(|&(x, y)| {
                                egui::pos2(
                                    rect.left() + (x * 0.5 + 0.5) * rect.width(),
                                    rect.center().y - y * (rect.height() * 0.45),
                                )
                            })
                            .collect();
                        ui.painter()
                            .add(egui::Shape::line(points, egui::Stroke::new(2.0, ACCENT)));
                    }

                    // ── Controls ──────────────────────────────────────────
                    ui.separator();
                    egui::Grid::new("controls").num_columns(2).show(ui, |ui| {
                        let float_controls: [(&str, &FloatParam); 7] = [
                            ("Drive", &params.drive),
                            ("Level", &params.level_db),
                            ("Bass", &params.bass_db),
                            ("Mid", &params.mid_db),
                            ("Treble", &params.treble_db),
                            ("Mix", &params.mix),
                            ("Bias", &params.bias),
                        ];
                        for (label, param) in float_controls {
                            ui.label(label);
                            ui.add(widgets::ParamSlider::for_param(param, setter));
                            ui.end_row();
                        }
                        ui.label("Oversampling");
                        ui.add(widgets::ParamSlider::for_param(
                            &params.oversampling,
                            setter,
                        ));
                        ui.end_row();
                    });

                    // ── VU meters ─────────────────────────────────────────
                    ui.separator();
                    let input_dbfs = state
                        .input_meter
                        .update(f32::from_bits(input_meter.load(Ordering::Relaxed)), 0.2);
                    draw_meter(ui, "In ", input_dbfs, METER_BG, IN_FILL);
                    let output_dbfs = state
                        .output_meter
                        .update(f32::from_bits(output_meter.load(Ordering::Relaxed)), 0.2);
                    draw_meter(ui, "Out", output_dbfs, METER_BG, OUT_FILL);

                    // ── A/B compare + clipboard ───────────────────────────
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui
                            .selectable_label(state.ab.active() == AbSlot::A, "A")
                            .clicked()
                        {
                            let recalled = state.ab.activate(AbSlot::A);
                            apply_params(setter, &params, &recalled);
                        }
                        if ui
                            .selectable_label(state.ab.active() == AbSlot::B, "B")
                            .clicked()
                        {
                            let recalled = state.ab.activate(AbSlot::B);
                            apply_params(setter, &params, &recalled);
                        }
                        if ui.button("Copy → other").clicked() {
                            state.ab.copy_active_to_other();
                        }
                        if ui.button("Copy to clipboard").clicked() {
                            let name = pedal.map(|pedal| pedal.name.as_str()).unwrap_or("—");
                            ctx.copy_text(clipboard_string(&snapshot, name));
                        }
                    });
                });
            },
        )
    }

    fn task_executor(&mut self) -> TaskExecutor<Self> {
        let controller = self.controller.clone();
        Box::new(move |task| {
            let mut guard = controller.lock().unwrap();
            let Some(controller) = guard.as_mut() else {
                return;
            };
            match task {
                DriveTask::SetModel(index) => controller.set_model(index),
                DriveTask::SetOversampling(choice) => controller.set_oversampling(choice),
            }
        })
    }

    fn initialize(
        &mut self,
        audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        let sample_rate = buffer_config.sample_rate;
        let max_block = buffer_config.max_buffer_size as usize;
        let channels = audio_io_layout
            .main_output_channels
            .map(|count| count.get() as usize)
            .unwrap_or(1);

        let pedals = match shipped_pedals() {
            Ok(pedals) => pedals,
            Err(error) => {
                nih_error!("shipped pedal bank failed to parse: {error}");
                return false;
            }
        };

        let snapshot = self.params.snapshot();
        let (controller, processor) =
            DriveController::new(max_block.max(1), sample_rate, channels, pedals, &snapshot);

        *self.controller.lock().unwrap() = Some(controller);
        self.processor = Some(processor);
        self.last_model = self.params.model.value();
        self.last_oversampling = self.params.oversampling.value();
        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // ── 1. MIDI: Program Change selects the pedal model.
        while let Some(event) = context.next_event() {
            if let NoteEvent::MidiProgramChange { program, .. } = event {
                let count = self
                    .controller
                    .lock()
                    .ok()
                    .and_then(|guard| guard.as_ref().map(DriveController::pedal_count))
                    .unwrap_or(1);
                if let Some(index) = program_to_slot(program, count) {
                    context.execute_background(DriveTask::SetModel(index));
                }
            }
        }

        // ── 2. Structural parameter changes → background rebuild.
        let model = self.params.model.value();
        if model != self.last_model {
            self.last_model = model;
            context.execute_background(DriveTask::SetModel(model as usize));
        }
        let oversampling = self.params.oversampling.value();
        if oversampling != self.last_oversampling {
            self.last_oversampling = oversampling;
            context.execute_background(DriveTask::SetOversampling(oversampling.to_choice()));
        }

        // ── 3. Render — flush denormals for the whole callback.
        let _denormal_guard = rt_fft_convolver::DenormalGuard::new();
        let snapshot = self.params.snapshot();
        let num_samples = buffer.samples();

        if let Some(processor) = self.processor.as_mut() {
            // Input peak (pre-render) for the VU meter.
            let input_dbfs = buffer
                .as_slice()
                .iter()
                .map(|channel| peak_dbfs(channel))
                .fold(SILENCE_DBFS, f32::max);
            self.input_meter_dbfs
                .store(input_dbfs.to_bits(), Ordering::Relaxed);

            let start = std::time::Instant::now();
            processor.process(buffer.as_slice(), &snapshot);
            let elapsed = start.elapsed().as_secs_f32();

            // Output peak (post-render) for the VU meter.
            let output_dbfs = buffer
                .as_slice()
                .iter()
                .map(|channel| peak_dbfs(channel))
                .fold(SILENCE_DBFS, f32::max);
            self.output_meter_dbfs
                .store(output_dbfs.to_bits(), Ordering::Relaxed);

            // CPU load = render time / block wall-clock budget.
            let sample_rate = context.transport().sample_rate;
            let budget = (num_samples as f32 / sample_rate).max(1.0e-9);
            self.cpu_meter
                .store((elapsed / budget).to_bits(), Ordering::Relaxed);
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for DrivePlugin {
    const CLAP_ID: &'static str = "org.codeberg.slundi.rt-drive";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("Oversampled nonlinear-drive / distortion plugin");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Mono,
        ClapFeature::Distortion,
    ];
}

impl Vst3Plugin for DrivePlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"RtDriveSlundi001";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Distortion];
}

nih_export_clap!(DrivePlugin);
nih_export_vst3!(DrivePlugin);
