//! `nih-plug` wrapper for `rt-pitch` (compiled only with the `nih` feature).
//!
//! A thin framework layer over the tested, framework-agnostic
//! [`engine`](crate::engine): it declares the automatable parameters, routes
//! MIDI + structural parameter changes into the off-thread
//! [`PitchController`], and runs the [`PitchProcessor`] on the audio thread.
//!
//! # Threading
//!
//! - The [`PitchController`] lives behind a `Mutex` shared with the `nih-plug`
//!   background-task executor.  Rebuilding the shifter voices (which
//!   reallocates FFT plans) happens there, never on the audio thread.
//! - `process()` polls the lock-free handoff for freshly-built voices, applies
//!   the continuous parameters and renders — allocation-free.
//!
//! # MIDI
//!
//! - **Program Change** selects the active preset (routed through
//!   [`program_to_slot`]); a subsequent host move of the `Current Preset`
//!   knob clears the override so automation always wins after the user acts.
//! - **CC** control of `Current Preset` / `Interval N` (whammy-pedal-style
//!   bends) is provided by exposing every continuous control as an automatable
//!   parameter, which hosts bind to any MIDI CC (host-side MIDI learn) — the
//!   idiomatic `nih-plug` path.
//!
//! # Phase-3/4 follow-ups
//!
//! - `nih-plug-egui` editor (preset rack, pitch-trajectory display, latency
//!   badge, MIDI activity) — Phase 3.
//! - MIDI note-in → chase target pitch — Phase 2 optional leftover.
//! - `lv2lint` / `clap-validator` / `pluginval` host validation — Phase 4.

use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use nih_plug::prelude::*;
#[cfg(feature = "gui")]
use nih_plug_egui::{EguiState, create_egui_editor, egui, widgets};
use rt_ir_plugin_core::program_to_slot;
use rt_pitch_core::{FormantMethod, NUM_PRESETS};

use crate::engine::{PitchController, PitchProcessor};
use crate::params::{Algorithm, CROSSFADE_MS_RANGE, PitchParams, Quality, SEMITONE_RANGE};
#[cfg(feature = "gui")]
use crate::ui::{
    MidiActivity, MidiStatus, PadState, PitchTrajectory, format_interval, format_latency_ms,
    latency_ms, pad_state,
};

/// Sentinel for "no MIDI Program Change override in effect".
const NO_OVERRIDE: i32 = -1;

/// MIDI-activity codes published from the audio thread to the editor LED:
/// nothing new, a recognised event, or an unrecognised one.
const MIDI_NONE: u32 = 0;
const MIDI_RECOGNISED: u32 = 1;
const MIDI_UNRECOGNISED: u32 = 2;

/// Off-thread work dispatched to the background executor (control thread only).
pub enum PitchTask {
    /// Rebuild the voices for a new algorithm / quality / formant config.
    ApplyStructural(PitchParams),
}

/// `nih-plug` algorithm enum, mirroring [`Algorithm`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
enum AlgorithmParam {
    #[name = "Smooth"]
    Smooth,
    #[name = "Tight / Mono"]
    TightMono,
    #[name = "Mono Voice"]
    MonoVoice,
}

impl AlgorithmParam {
    fn to_choice(self) -> Algorithm {
        match self {
            AlgorithmParam::Smooth => Algorithm::Smooth,
            AlgorithmParam::TightMono => Algorithm::TightMono,
            AlgorithmParam::MonoVoice => Algorithm::MonoVoice,
        }
    }
}

/// `nih-plug` quality enum, mirroring [`Quality`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
enum QualityParam {
    #[name = "Live (512)"]
    Live,
    #[name = "Standard (1024)"]
    Standard,
    #[name = "Studio (2048)"]
    Studio,
}

impl QualityParam {
    fn to_choice(self) -> Quality {
        match self {
            QualityParam::Live => Quality::Live,
            QualityParam::Standard => Quality::Standard,
            QualityParam::Studio => Quality::Studio,
        }
    }
}

/// `nih-plug` formant-method enum, mirroring [`FormantMethod`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
enum FormantParam {
    #[name = "Cepstral"]
    Cepstral,
    #[name = "LPC"]
    Lpc,
    #[name = "True Envelope"]
    TrueEnvelope,
}

impl FormantParam {
    fn to_choice(self) -> FormantMethod {
        match self {
            FormantParam::Cepstral => FormantMethod::Cepstral,
            FormantParam::Lpc => FormantMethod::Lpc,
            FormantParam::TrueEnvelope => FormantMethod::TrueEnvelope,
        }
    }
}

/// One preset slot's automatable controls (interval + enabled).  Nested as an
/// array so `nih-plug` auto-suffixes the ids (`interval1`, `enabled1`, …).
#[derive(Params)]
struct PresetParams {
    #[id = "interval"]
    interval: FloatParam,
    #[id = "enabled"]
    enabled: BoolParam,
}

impl PresetParams {
    fn new(default_semitones: f32) -> Self {
        Self {
            interval: FloatParam::new(
                "Interval",
                default_semitones,
                FloatRange::Linear {
                    min: SEMITONE_RANGE.0,
                    max: SEMITONE_RANGE.1,
                },
            )
            .with_unit(" st")
            .with_step_size(0.01),
            enabled: BoolParam::new("Enabled", true),
        }
    }
}

/// Host-visible parameters.
#[derive(Params)]
struct PitchPluginParams {
    #[nested(array, group = "Presets")]
    presets: [PresetParams; NUM_PRESETS],

    #[id = "cur"]
    current_preset: IntParam,
    #[id = "v2"]
    voice2_preset: IntParam,
    #[id = "dual"]
    dual_voice: BoolParam,
    #[id = "xfade"]
    crossfade_ms: FloatParam,
    #[id = "algo"]
    algorithm: EnumParam<AlgorithmParam>,
    #[id = "qual"]
    quality: EnumParam<QualityParam>,
    #[id = "mix"]
    mix: FloatParam,
    #[id = "fmt"]
    formant: BoolParam,
    #[id = "fmtm"]
    formant_method: EnumParam<FormantParam>,
    #[id = "drycomp"]
    dry_delay_compensation: BoolParam,
    #[id = "byp"]
    bypass: BoolParam,

    #[cfg(feature = "gui")]
    #[persist = "editor-state"]
    editor_state: Arc<EguiState>,
}

impl Default for PitchPluginParams {
    fn default() -> Self {
        let defaults = PitchParams::default();
        let presets =
            std::array::from_fn(|index| PresetParams::new(defaults.intervals[index].semitones));
        let preset_max = (NUM_PRESETS - 1) as i32;

        Self {
            presets,
            current_preset: IntParam::new(
                "Current Preset",
                defaults.current_preset as i32,
                IntRange::Linear {
                    min: 0,
                    max: preset_max,
                },
            ),
            voice2_preset: IntParam::new(
                "Voice 2 Preset",
                defaults.voice2_preset as i32,
                IntRange::Linear {
                    min: 0,
                    max: preset_max,
                },
            ),
            dual_voice: BoolParam::new("Dual Voice", defaults.dual_voice),
            crossfade_ms: FloatParam::new(
                "Crossfade Time",
                defaults.crossfade_ms,
                FloatRange::Linear {
                    min: CROSSFADE_MS_RANGE.0,
                    max: CROSSFADE_MS_RANGE.1,
                },
            )
            .with_unit(" ms"),
            algorithm: EnumParam::new("Algorithm", AlgorithmParam::Smooth),
            quality: EnumParam::new("Quality", QualityParam::Standard),
            mix: FloatParam::new(
                "Mix",
                defaults.mix,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_unit(" %")
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage()),
            formant: BoolParam::new("Formant Preservation", defaults.formant_preservation),
            formant_method: EnumParam::new("Formant Method", FormantParam::Cepstral),
            dry_delay_compensation: BoolParam::new(
                "Dry Delay Compensation",
                defaults.dry_delay_compensation,
            ),
            bypass: BoolParam::new("Bypass", false)
                .with_value_to_string(formatters::v2s_bool_bypass()),
            #[cfg(feature = "gui")]
            editor_state: EguiState::from_size(560, 520),
        }
    }
}

impl PitchPluginParams {
    /// Snapshot into the framework-agnostic model.  `preset_override` (≥ 0)
    /// replaces the `Current Preset` param when a MIDI Program Change is active.
    fn snapshot(&self, preset_override: i32) -> PitchParams {
        let mut intervals = [rt_pitch_core::PresetSlot::default(); NUM_PRESETS];
        for (slot, preset) in intervals.iter_mut().zip(self.presets.iter()) {
            slot.semitones = preset.interval.value();
            slot.enabled = preset.enabled.value();
        }
        let current_preset = if preset_override >= 0 {
            preset_override as usize
        } else {
            self.current_preset.value() as usize
        };

        PitchParams {
            intervals,
            current_preset,
            voice2_preset: self.voice2_preset.value() as usize,
            dual_voice: self.dual_voice.value(),
            crossfade_ms: self.crossfade_ms.value(),
            algorithm: self.algorithm.value().to_choice(),
            quality: self.quality.value().to_choice(),
            mix: self.mix.value(),
            formant_preservation: self.formant.value(),
            formant_method: self.formant_method.value().to_choice(),
            dry_delay_compensation: self.dry_delay_compensation.value(),
            bypass: self.bypass.value(),
        }
    }
}

/// The pitch plugin instance.
pub struct PitchPlugin {
    params: Arc<PitchPluginParams>,
    /// Shared with the background-task executor; built in `initialize`.
    controller: Arc<Mutex<Option<PitchController>>>,
    /// Audio-thread renderer; built in `initialize`.
    processor: Option<PitchProcessor>,

    /// Active MIDI Program-Change preset override, or [`NO_OVERRIDE`].
    midi_preset: i32,
    /// Last `Current Preset` param value, to clear the override on a host move.
    last_current_preset: i32,

    // Structural change detection so we only rebuild on an actual move.
    last_structural: PitchParams,
    /// Last latency reported to the host, to avoid redundant notifications.
    last_reported_latency: usize,

    // ── Editor telemetry (audio thread → egui), `f32`/`u32` packed atomics. ──
    /// Current output pitch in semitones (the selected interval) — trajectory.
    pitch_semitones: Arc<AtomicU32>,
    /// Reported latency in samples — the latency badge.
    latency_samples: Arc<AtomicU32>,
    /// Host sample rate (`f32` bits) — converts the latency badge to ms.
    sample_rate_hz: Arc<AtomicU32>,
    /// Last MIDI-activity code (see `MIDI_*`) — the header LED.
    midi_code: Arc<AtomicU32>,
}

impl Default for PitchPlugin {
    fn default() -> Self {
        let defaults = PitchParams::default();
        Self {
            params: Arc::new(PitchPluginParams::default()),
            controller: Arc::new(Mutex::new(None)),
            processor: None,
            midi_preset: NO_OVERRIDE,
            last_current_preset: defaults.current_preset as i32,
            last_structural: defaults,
            last_reported_latency: usize::MAX,
            pitch_semitones: Arc::new(AtomicU32::new(0.0f32.to_bits())),
            latency_samples: Arc::new(AtomicU32::new(0)),
            sample_rate_hz: Arc::new(AtomicU32::new(48_000.0f32.to_bits())),
            midi_code: Arc::new(AtomicU32::new(MIDI_NONE)),
        }
    }
}

/// Per-frame editor state owned by the egui closure.
#[cfg(feature = "gui")]
struct PitchEditorState {
    trajectory: PitchTrajectory,
    midi: MidiActivity,
}

#[cfg(feature = "gui")]
impl PitchEditorState {
    fn new() -> Self {
        Self {
            // ~4 s of history at a typical editor frame rate; eased for a
            // smooth whammy/crossfade line.
            trajectory: PitchTrajectory::new(240, 0.2),
            midi: MidiActivity::new(),
        }
    }
}

/// Set one `IntParam` with a complete begin/set/end automation gesture (pad
/// clicks in the preset rack).
#[cfg(feature = "gui")]
fn set_int_param(setter: &ParamSetter, param: &IntParam, value: i32) {
    setter.begin_set_parameter(param);
    setter.set_parameter(param, value);
    setter.end_set_parameter(param);
}

/// Draw the small MIDI-activity LED (green = recognised, red = unrecognised).
#[cfg(feature = "gui")]
fn draw_midi_led(ui: &mut egui::Ui, midi: &MidiActivity) {
    use egui::Color32;
    let base = match midi.status() {
        MidiStatus::Recognised => Color32::from_rgb(0x40, 0xd0, 0x60),
        MidiStatus::Unrecognised => Color32::from_rgb(0xd0, 0x50, 0x40),
        MidiStatus::Idle => Color32::from_rgb(0x30, 0x30, 0x38),
    };
    let level = midi.level().clamp(0.0, 1.0);
    // Blend toward the base colour by brightness.
    let dim = Color32::from_rgb(0x22, 0x24, 0x2a);
    let lerp = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * level) as u8;
    let colour = Color32::from_rgb(
        lerp(dim.r(), base.r()),
        lerp(dim.g(), base.g()),
        lerp(dim.b(), base.b()),
    );
    let (rect, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), 5.0, colour);
    ui.label("MIDI");
}

/// Draw the rolling pitch-trajectory line into a panel.
#[cfg(feature = "gui")]
fn draw_trajectory(
    ui: &mut egui::Ui,
    trajectory: &PitchTrajectory,
    panel: egui::Color32,
    grid: egui::Color32,
    accent: egui::Color32,
) {
    let (rect, _) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), 90.0), egui::Sense::hover());
    ui.painter().rect_filled(rect, 4.0, panel);

    // Zero line (unison) across the vertical centre.
    ui.painter().line_segment(
        [
            egui::pos2(rect.left(), rect.center().y),
            egui::pos2(rect.right(), rect.center().y),
        ],
        egui::Stroke::new(1.0, grid),
    );

    let history = trajectory.ordered();
    if history.len() < 2 {
        return;
    }
    // Map ±24 semitones to the panel height.
    const SPAN: f32 = 24.0;
    let last = (history.len() - 1) as f32;
    let points: Vec<egui::Pos2> = history
        .iter()
        .enumerate()
        .map(|(index, &semis)| {
            let x = rect.left() + index as f32 / last * rect.width();
            let norm = (semis / SPAN).clamp(-1.0, 1.0);
            let y = rect.center().y - norm * (rect.height() * 0.45);
            egui::pos2(x, y)
        })
        .collect();
    ui.painter()
        .add(egui::Shape::line(points, egui::Stroke::new(2.0, accent)));
}

impl Plugin for PitchPlugin {
    const NAME: &'static str = "RT Pitch";
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
    type BackgroundTask = PitchTask;

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    #[cfg(feature = "gui")]
    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        use egui::Color32;

        // Violet accent for a pitch/harmony plugin.
        const ACCENT: Color32 = Color32::from_rgb(0xb0, 0x8c, 0xff);
        const PANEL: Color32 = Color32::from_rgb(0x16, 0x18, 0x1c);
        const GRID: Color32 = Color32::from_rgb(0x2a, 0x2e, 0x36);
        const PAD_DISABLED: Color32 = Color32::from_rgb(0x30, 0x30, 0x38);
        const PAD_FILLED: Color32 = Color32::from_rgb(0x3a, 0x40, 0x52);

        let params = self.params.clone();
        let pitch_semitones = self.pitch_semitones.clone();
        let latency_samples = self.latency_samples.clone();
        let sample_rate_hz = self.sample_rate_hz.clone();
        let midi_code = self.midi_code.clone();

        create_egui_editor(
            self.params.editor_state.clone(),
            PitchEditorState::new(),
            |_ctx, _state| {},
            move |ctx, setter, state| {
                // Consume the latest MIDI pulse and decay the LED this frame.
                let code = midi_code.swap(MIDI_NONE, Ordering::Relaxed);
                if code != MIDI_NONE {
                    state.midi.trigger(code == MIDI_RECOGNISED);
                }
                state.midi.decay(0.08);

                // Feed the trajectory the current output pitch.
                let target = f32::from_bits(pitch_semitones.load(Ordering::Relaxed));
                state.trajectory.push(target);

                let current = params.current_preset.value() as usize;

                egui::CentralPanel::default().show(ctx, |ui| {
                    // ── Header: name + MIDI LED + bypass ──────────────────
                    ui.horizontal(|ui| {
                        ui.heading("RT Pitch");
                        draw_midi_led(ui, &state.midi);
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.add(widgets::ParamSlider::for_param(&params.bypass, setter));
                        });
                    });
                    ui.separator();

                    // ── Preset rack: 8 pads (2 rows × 4) ──────────────────
                    egui::Grid::new("preset-rack")
                        .num_columns(4)
                        .spacing(egui::vec2(6.0, 6.0))
                        .show(ui, |ui| {
                            for index in 0..NUM_PRESETS {
                                let preset = &params.presets[index];
                                let state =
                                    pad_state(true, preset.enabled.value(), index == current);
                                let fill = match state {
                                    PadState::Active => ACCENT,
                                    PadState::Disabled => PAD_DISABLED,
                                    _ => PAD_FILLED,
                                };
                                let label = format!(
                                    "{}\n{}",
                                    index + 1,
                                    format_interval(preset.interval.value())
                                );
                                let button = egui::Button::new(
                                    egui::RichText::new(label).color(Color32::WHITE),
                                )
                                .fill(fill)
                                .min_size(egui::vec2(122.0, 40.0));
                                if ui.add(button).clicked() {
                                    set_int_param(setter, &params.current_preset, index as i32);
                                }
                                if index % 4 == 3 {
                                    ui.end_row();
                                }
                            }
                        });

                    // ── Selected preset editor ────────────────────────────
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label(format!("Preset {}", current + 1));
                        ui.add(widgets::ParamSlider::for_param(
                            &params.presets[current].interval,
                            setter,
                        ));
                        ui.add(widgets::ParamSlider::for_param(
                            &params.presets[current].enabled,
                            setter,
                        ));
                    });

                    // ── Pitch-trajectory plot ─────────────────────────────
                    ui.separator();
                    draw_trajectory(ui, &state.trajectory, PANEL, GRID, ACCENT);

                    // ── Latency badge ─────────────────────────────────────
                    let samples = latency_samples.load(Ordering::Relaxed) as usize;
                    let rate = f32::from_bits(sample_rate_hz.load(Ordering::Relaxed));
                    ui.horizontal(|ui| {
                        ui.label("Latency");
                        ui.label(
                            egui::RichText::new(format_latency_ms(latency_ms(samples, rate)))
                                .color(ACCENT),
                        );
                    });

                    // ── Global controls ───────────────────────────────────
                    ui.separator();
                    egui::Grid::new("controls").num_columns(2).show(ui, |ui| {
                        let row = |ui: &mut egui::Ui, label: &str, add: &dyn Fn(&mut egui::Ui)| {
                            ui.label(label);
                            add(ui);
                            ui.end_row();
                        };
                        row(ui, "Algorithm", &|ui| {
                            ui.add(widgets::ParamSlider::for_param(&params.algorithm, setter));
                        });
                        row(ui, "Quality", &|ui| {
                            ui.add(widgets::ParamSlider::for_param(&params.quality, setter));
                        });
                        row(ui, "Mix", &|ui| {
                            ui.add(widgets::ParamSlider::for_param(&params.mix, setter));
                        });
                        row(ui, "Crossfade", &|ui| {
                            ui.add(widgets::ParamSlider::for_param(
                                &params.crossfade_ms,
                                setter,
                            ));
                        });
                        row(ui, "Current Preset", &|ui| {
                            ui.add(widgets::ParamSlider::for_param(
                                &params.current_preset,
                                setter,
                            ));
                        });
                        row(ui, "Voice 2 Preset", &|ui| {
                            ui.add(widgets::ParamSlider::for_param(
                                &params.voice2_preset,
                                setter,
                            ));
                        });
                        row(ui, "Dual Voice", &|ui| {
                            ui.add(widgets::ParamSlider::for_param(&params.dual_voice, setter));
                        });
                        row(ui, "Formant", &|ui| {
                            ui.add(widgets::ParamSlider::for_param(&params.formant, setter));
                        });
                        row(ui, "Formant Method", &|ui| {
                            ui.add(widgets::ParamSlider::for_param(
                                &params.formant_method,
                                setter,
                            ));
                        });
                        row(ui, "Dry Delay Comp.", &|ui| {
                            ui.add(widgets::ParamSlider::for_param(
                                &params.dry_delay_compensation,
                                setter,
                            ));
                        });
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
                PitchTask::ApplyStructural(params) => {
                    controller.apply_structural(&params);
                }
            }
        })
    }

    fn initialize(
        &mut self,
        audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        let sample_rate = buffer_config.sample_rate;
        let max_block = (buffer_config.max_buffer_size as usize).max(1);
        let channels = audio_io_layout
            .main_output_channels
            .map(|count| count.get() as usize)
            .unwrap_or(1);

        let snapshot = self.params.snapshot(NO_OVERRIDE);
        let (controller, processor) =
            PitchController::new(max_block, sample_rate, channels, &snapshot);

        // Report the initial latency to the host and prime the editor badges.
        let latency = controller.latency();
        context.set_latency_samples(latency as u32);
        self.last_reported_latency = latency;
        self.latency_samples
            .store(latency as u32, Ordering::Relaxed);
        self.sample_rate_hz
            .store(sample_rate.to_bits(), Ordering::Relaxed);

        *self.controller.lock().unwrap() = Some(controller);
        self.processor = Some(processor);
        self.last_structural = snapshot;
        self.last_current_preset = self.params.current_preset.value();
        self.midi_preset = NO_OVERRIDE;
        true
    }

    fn reset(&mut self) {
        if let Some(processor) = self.processor.as_mut() {
            let snapshot = self.params.snapshot(self.midi_preset);
            processor.reset(&snapshot);
        }
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // ── 1. MIDI: Program Change overrides the active preset. ──
        while let Some(event) = context.next_event() {
            if let NoteEvent::MidiProgramChange { program, .. } = event {
                match program_to_slot(program, NUM_PRESETS) {
                    Some(index) => {
                        self.midi_preset = index as i32;
                        self.midi_code.store(MIDI_RECOGNISED, Ordering::Relaxed);
                    }
                    None => self.midi_code.store(MIDI_UNRECOGNISED, Ordering::Relaxed),
                }
            }
        }

        // A host move of the Current Preset knob reclaims control from MIDI.
        let current_preset = self.params.current_preset.value();
        if current_preset != self.last_current_preset {
            self.last_current_preset = current_preset;
            self.midi_preset = NO_OVERRIDE;
        }

        let snapshot = self.params.snapshot(self.midi_preset);

        // ── 2. Structural parameter changes → background rebuild. ──
        if self.last_structural.structural_changed(&snapshot) {
            self.last_structural = snapshot;
            context.execute_background(PitchTask::ApplyStructural(snapshot));
        }

        // ── 3. Render — flush denormals for the whole callback. ──
        let _denormal_guard = rt_fft_convolver::DenormalGuard::new();

        if let Some(processor) = self.processor.as_mut() {
            processor.process(buffer.as_slice(), &snapshot);

            // Report a latency change (e.g. after a Quality rebuild lands).
            let latency = processor.reported_latency();
            if latency != self.last_reported_latency {
                self.last_reported_latency = latency;
                context.set_latency_samples(latency as u32);
            }

            // Publish editor telemetry (cheap relaxed atomics).
            let current = snapshot.current_preset.min(NUM_PRESETS - 1);
            self.pitch_semitones.store(
                snapshot.intervals[current].semitones.to_bits(),
                Ordering::Relaxed,
            );
            self.latency_samples
                .store(latency as u32, Ordering::Relaxed);
            self.sample_rate_hz
                .store(context.transport().sample_rate.to_bits(), Ordering::Relaxed);
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for PitchPlugin {
    const CLAP_ID: &'static str = "org.codeberg.slundi.rt-pitch";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("Real-time pitch-shift / harmoniser (phase-vocoder / WSOLA / PSOLA)");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Mono,
        ClapFeature::PitchShifter,
    ];
}

impl Vst3Plugin for PitchPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"RtPitchSlundi001";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::PitchShift];
}

nih_export_clap!(PitchPlugin);
nih_export_vst3!(PitchPlugin);
