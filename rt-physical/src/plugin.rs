//! `nih-plug` wrapper for the physical-modelling instrument (compiled only with
//! the `nih` feature).
//!
//! A thin framework layer over the tested, framework-agnostic
//! [`PhysicalEngine`](crate::engine::PhysicalEngine): it declares the
//! automatable parameters, maps them onto [`InstrumentParams`], routes MIDI, and
//! runs the engine on the audio thread.  The engine pre-builds both voice banks
//! at [`initialize`](PhysicalPlugin::initialize), so the audio callback only
//! ever mutates scalar coefficients and triggers voices — it never allocates.
//!
//! # MIDI (roadmap Phase 2)
//!
//! - **Note On / Off** → sample-accurate voice trigger / release, velocity →
//!   excitation energy.
//! - **CC 74 (Brightness)** and **channel pressure (aftertouch)** →
//!   `Damping`, the model's most-used tone control.  Applied as an override on
//!   top of the host value and released the moment the host moves `Damping`
//!   itself.  Because a physical model bakes tone into a note at pluck time, the
//!   override affects *subsequently* triggered notes.
//!
//! # I/O
//!
//! Stereo out.  A stereo (or no) input bus is offered: with input connected the
//! `Mix` control blends the dry audio with the synthesised voices (effect use);
//! with no input it is a pure synth.  Zero latency.

use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use nih_plug::prelude::*;
#[cfg(feature = "gui")]
use nih_plug_egui::{EguiState, create_egui_editor, egui, widgets};

use crate::engine::PhysicalEngine;
#[cfg(feature = "gui")]
use crate::params::model_to_index;
use crate::params::{
    BODY_TYPES, BodyType, DECAY_S_RANGE, DECAY_SLOPE_RANGE, DETUNE_CENTS_RANGE, GAIN_DB_RANGE,
    InstrumentParams, MODELS, MODES_RANGE, POLYPHONY_RANGE, PRESETS, model_from_index, model_label,
    preset_from_index, preset_label,
};
#[cfg(feature = "gui")]
use crate::ui::{DISPLAY_CEIL_DB, DISPLAY_FLOOR_DB, Partial, db_to_intensity, partial_spectrum};
#[cfg(feature = "gui")]
use rt_physical_core::Model;

/// CC that live-controls `Damping` (Sound Controller 5 / "Brightness").
const CC_DAMPING: u8 = 74;

/// Host-visible parameters (Phase-2 control list).
#[derive(Params)]
struct PhysicalPluginParams {
    #[id = "model"]
    model: IntParam,

    // Plucked-string controls.
    #[id = "pluckpos"]
    pluck_position: FloatParam,
    #[id = "hardness"]
    pick_hardness: FloatParam,
    #[id = "damping"]
    damping: FloatParam,
    #[id = "body"]
    body: IntParam,
    #[id = "detune"]
    detune: FloatParam,

    // Modal controls.
    #[id = "modes"]
    num_modes: IntParam,
    #[id = "preset"]
    preset: IntParam,
    #[id = "slope"]
    decay_slope: FloatParam,

    // Shared.
    #[id = "decay"]
    decay_s: FloatParam,

    // Global.
    #[id = "gain"]
    gain_db: FloatParam,
    #[id = "mix"]
    mix: FloatParam,
    #[id = "poly"]
    polyphony: IntParam,
    #[id = "byp"]
    bypass: BoolParam,

    #[cfg(feature = "gui")]
    #[persist = "editor-state"]
    editor_state: Arc<EguiState>,
}

/// Build a `0..=1` percent `FloatParam`.
fn percent(name: &str, default: f32) -> FloatParam {
    FloatParam::new(name, default, FloatRange::Linear { min: 0.0, max: 1.0 })
        .with_unit(" %")
        .with_value_to_string(formatters::v2s_f32_percentage(0))
        .with_string_to_value(formatters::s2v_f32_percentage())
}

impl Default for PhysicalPluginParams {
    fn default() -> Self {
        let defaults = InstrumentParams::default();
        Self {
            model: IntParam::new(
                "Model",
                crate::params::model_to_index(defaults.model) as i32,
                IntRange::Linear {
                    min: 0,
                    max: (MODELS.len() - 1) as i32,
                },
            )
            .with_value_to_string(Arc::new(|value| {
                model_label(model_from_index(value as usize)).to_string()
            })),
            pluck_position: percent("Pluck Position", defaults.pluck_position),
            pick_hardness: percent("Pick Hardness", defaults.pick_hardness),
            damping: percent("Damping", defaults.damping),
            body: IntParam::new(
                "Body Type",
                defaults.body.to_index() as i32,
                IntRange::Linear {
                    min: 0,
                    max: (BODY_TYPES.len() - 1) as i32,
                },
            )
            .with_value_to_string(Arc::new(|value| {
                BodyType::from_index(value as usize).label().to_string()
            })),
            detune: FloatParam::new(
                "Detune",
                defaults.detune_cents,
                FloatRange::Linear {
                    min: DETUNE_CENTS_RANGE.0,
                    max: DETUNE_CENTS_RANGE.1,
                },
            )
            .with_unit(" ct"),
            num_modes: IntParam::new(
                "Modes",
                defaults.num_modes as i32,
                IntRange::Linear {
                    min: MODES_RANGE.0 as i32,
                    max: MODES_RANGE.1 as i32,
                },
            ),
            preset: IntParam::new(
                "Frequency Ratios",
                crate::params::preset_to_index(defaults.preset) as i32,
                IntRange::Linear {
                    min: 0,
                    max: (PRESETS.len() - 1) as i32,
                },
            )
            .with_value_to_string(Arc::new(|value| {
                preset_label(preset_from_index(value as usize)).to_string()
            })),
            decay_slope: FloatParam::new(
                "Decay Slope",
                defaults.decay_slope,
                FloatRange::Linear {
                    min: DECAY_SLOPE_RANGE.0,
                    max: DECAY_SLOPE_RANGE.1,
                },
            ),
            decay_s: FloatParam::new(
                "Decay",
                defaults.decay_s,
                FloatRange::Skewed {
                    min: DECAY_S_RANGE.0,
                    max: DECAY_S_RANGE.1,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_unit(" s"),
            gain_db: FloatParam::new(
                "Gain",
                defaults.gain_db,
                FloatRange::Linear {
                    min: GAIN_DB_RANGE.0,
                    max: GAIN_DB_RANGE.1,
                },
            )
            .with_unit(" dB"),
            mix: percent("Mix", defaults.mix),
            polyphony: IntParam::new(
                "Polyphony",
                defaults.polyphony as i32,
                IntRange::Linear {
                    min: POLYPHONY_RANGE.0 as i32,
                    max: POLYPHONY_RANGE.1 as i32,
                },
            ),
            bypass: BoolParam::new("Bypass", false)
                .with_value_to_string(formatters::v2s_bool_bypass()),
            #[cfg(feature = "gui")]
            editor_state: EguiState::from_size(560, 620),
        }
    }
}

impl PhysicalPluginParams {
    /// Snapshot the host parameters into the framework-agnostic model.
    fn snapshot(&self) -> InstrumentParams {
        InstrumentParams {
            model: model_from_index(self.model.value() as usize),
            pluck_position: self.pluck_position.value(),
            pick_hardness: self.pick_hardness.value(),
            damping: self.damping.value(),
            body: BodyType::from_index(self.body.value() as usize),
            detune_cents: self.detune.value(),
            num_modes: self.num_modes.value() as usize,
            preset: preset_from_index(self.preset.value() as usize),
            decay_slope: self.decay_slope.value(),
            decay_s: self.decay_s.value(),
            gain_db: self.gain_db.value(),
            mix: self.mix.value(),
            polyphony: self.polyphony.value() as usize,
            bypass: self.bypass.value(),
        }
        .clamped()
    }
}

/// Live MIDI override of `Damping`, released once the host moves the parameter.
#[derive(Default)]
struct MidiOverrides {
    damping: Option<f32>,
    last_damping: f32,
}

/// The instrument plugin instance.
pub struct PhysicalPlugin {
    params: Arc<PhysicalPluginParams>,
    engine: Option<PhysicalEngine>,
    midi: MidiOverrides,
    /// CPU load = render time / block budget (`f32` bits), for the Phase 3 meter.
    cpu_meter: Arc<AtomicU32>,
    /// Voices sounding in the active bank last block, for the activity meter.
    #[cfg(feature = "gui")]
    voice_meter: Arc<AtomicU32>,
    /// Bumped on each note-on so the editor's note-activity LED can blink.
    #[cfg(feature = "gui")]
    note_counter: Arc<AtomicU32>,
}

impl Default for PhysicalPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(PhysicalPluginParams::default()),
            engine: None,
            midi: MidiOverrides::default(),
            cpu_meter: Arc::new(AtomicU32::new(0)),
            #[cfg(feature = "gui")]
            voice_meter: Arc::new(AtomicU32::new(0)),
            #[cfg(feature = "gui")]
            note_counter: Arc::new(AtomicU32::new(0)),
        }
    }
}

impl PhysicalPlugin {
    /// Fold the MIDI `Damping` override in, releasing it when the host moves the
    /// `Damping` parameter itself (so the DAW always wins control back).
    fn resolve_snapshot(&mut self, mut snapshot: InstrumentParams) -> InstrumentParams {
        let damping = self.params.damping.value();
        if damping != self.midi.last_damping {
            self.midi.damping = None;
        }
        self.midi.last_damping = damping;

        if let Some(value) = self.midi.damping {
            snapshot.damping = value;
        }
        snapshot.clamped()
    }
}

/// Apply one MIDI event to the engine / overrides.  A free function so it can
/// borrow the engine and the override state without re-borrowing `self`.
fn apply_event(engine: &mut PhysicalEngine, midi: &mut MidiOverrides, event: &NoteEvent<()>) {
    match *event {
        NoteEvent::NoteOn { note, velocity, .. } => engine.note_on(note, velocity),
        NoteEvent::NoteOff { note, .. } => engine.note_off(note),
        NoteEvent::Choke { note, .. } => engine.note_off(note),
        NoteEvent::MidiCC { cc, value, .. } if cc == CC_DAMPING => {
            midi.damping = Some(value.clamp(0.0, 1.0));
        }
        NoteEvent::MidiChannelPressure { pressure, .. } => {
            midi.damping = Some(pressure.clamp(0.0, 1.0));
        }
        _ => {}
    }
}

// ─── egui editor (Phase 3, `gui` feature) ──────────────────────────────────

/// Per-open editor state owned by the egui closure.
#[cfg(feature = "gui")]
#[derive(Default)]
struct PhysicalEditorState {
    /// Cached partial spectrum, recomputed only when the parameters change.
    spectrum: Vec<Partial>,
    /// Parameter snapshot the cached spectrum was computed from.
    last_params: Option<InstrumentParams>,
    /// Last observed note counter, to fire the note-activity LED.
    last_note: u32,
    /// Note-activity LED level, `0..=1`, decaying each frame.
    note_level: f32,
}

/// Draw the partial-spectrum bar chart: one bar per partial, tallest =
/// loudest, spaced by frequency ratio so inharmonic modal partials fan out.
#[cfg(feature = "gui")]
fn draw_spectrum(
    ui: &mut egui::Ui,
    spectrum: &[Partial],
    panel: egui::Color32,
    grid: egui::Color32,
    accent: egui::Color32,
) {
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), 120.0),
        egui::Sense::hover(),
    );
    ui.painter().rect_filled(rect, 4.0, panel);

    // Horizontal grid at −12 / −24 dB.
    for &db in &[-12.0f32, -24.0] {
        let intensity = db_to_intensity(db, DISPLAY_FLOOR_DB, DISPLAY_CEIL_DB);
        let y = rect.bottom() - intensity * rect.height();
        ui.painter().line_segment(
            [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
            egui::Stroke::new(1.0, grid),
        );
    }

    if spectrum.is_empty() {
        return;
    }
    // Lay the bars out along the frequency axis (ratio), so the modal presets'
    // inharmonic spacing is visible rather than evenly spaced harmonics.
    let max_ratio = spectrum
        .iter()
        .map(|partial| partial.ratio)
        .fold(1.0f32, f32::max);
    let bar_width = (rect.width() / spectrum.len() as f32 * 0.5).clamp(2.0, 14.0);
    for partial in spectrum {
        let intensity = db_to_intensity(partial.amplitude_db(), DISPLAY_FLOOR_DB, DISPLAY_CEIL_DB);
        let x = rect.left() + (partial.ratio / max_ratio) * (rect.width() - bar_width);
        let top = rect.bottom() - intensity * rect.height();
        let bar =
            egui::Rect::from_min_max(egui::pos2(x, top), egui::pos2(x + bar_width, rect.bottom()));
        ui.painter().rect_filled(bar, 1.0, accent);
    }
}

/// Draw the voice-activity meter: `maximum` pips, the first `active` lit.
#[cfg(feature = "gui")]
fn draw_voice_meter(
    ui: &mut egui::Ui,
    active: usize,
    maximum: usize,
    lit: egui::Color32,
    dim: egui::Color32,
) {
    ui.horizontal(|ui| {
        ui.label(format!("Voices {active}/{maximum}"));
        for index in 0..maximum.max(1) {
            let (pip, _) = ui.allocate_exact_size(egui::vec2(9.0, 12.0), egui::Sense::hover());
            let colour = if index < active { lit } else { dim };
            ui.painter().rect_filled(pip.shrink(1.0), 1.0, colour);
        }
    });
}

impl Plugin for PhysicalPlugin {
    const NAME: &'static str = "RT Physical";
    const VENDOR: &'static str = "Sébastien Lundi";
    const URL: &'static str = "https://codeberg.org/slundi/rt-fft-convolver";
    const EMAIL: &'static str = "slundi@stamus-networks.com";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    // Stereo out; a stereo or absent input bus (effect / pure-synth use).
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: None,
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
    ];

    const MIDI_INPUT: MidiConfig = MidiConfig::Basic;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    #[cfg(feature = "gui")]
    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        use egui::Color32;

        // Amber accent — a warm "acoustic instrument" tone, distinct from the
        // violet reverb and the teal restoration / IR plugins.
        const ACCENT: Color32 = Color32::from_rgb(0xe0, 0xa8, 0x54);
        const PANEL: Color32 = Color32::from_rgb(0x1a, 0x16, 0x10);
        const GRID: Color32 = Color32::from_rgb(0x32, 0x2c, 0x22);
        const DIM: Color32 = Color32::from_rgb(0x2a, 0x24, 0x1c);
        const LIT: Color32 = Color32::from_rgb(0x6a, 0xd0, 0x8a);

        let params = self.params.clone();
        let cpu_meter = self.cpu_meter.clone();
        let voice_meter = self.voice_meter.clone();
        let note_counter = self.note_counter.clone();

        create_egui_editor(
            self.params.editor_state.clone(),
            PhysicalEditorState::default(),
            |_ctx, _state| {},
            move |ctx, setter, state| {
                // Drive the note-activity LED from the shared counter.
                let count = note_counter.load(Ordering::Relaxed);
                if count != state.last_note {
                    state.last_note = count;
                    state.note_level = 1.0;
                }
                state.note_level = (state.note_level - 0.08).max(0.0);

                // Recompute the spectrum preview only when parameters change.
                let snapshot = params.snapshot();
                if state.last_params != Some(snapshot) {
                    state.spectrum = partial_spectrum(&snapshot);
                    state.last_params = Some(snapshot);
                }

                let model = model_from_index(params.model.value() as usize);

                egui::CentralPanel::default().show(ctx, |ui| {
                    // ── Header: title · note LED · bypass ───────────────────
                    ui.horizontal(|ui| {
                        ui.heading("RT Physical");
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.add(widgets::ParamSlider::for_param(&params.bypass, setter));
                            ui.label("Bypass");
                            let (led, _) = ui
                                .allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
                            let level = state.note_level;
                            let col = Color32::from_rgb(
                                (0x30 as f32 + (LIT.r() as f32 - 0x30 as f32) * level) as u8,
                                (0x30 as f32 + (LIT.g() as f32 - 0x30 as f32) * level) as u8,
                                (0x30 as f32 + (LIT.b() as f32 - 0x30 as f32) * level) as u8,
                            );
                            ui.painter().circle_filled(led.center(), 5.0, col);
                            ui.label("Note");
                        });
                    });
                    ui.separator();

                    // ── Model selector (dropdown switches the visible controls) ──
                    ui.horizontal(|ui| {
                        ui.label("Model");
                        for (index, &family) in MODELS.iter().enumerate() {
                            let active = model_to_index(model) == index;
                            let fill = if active { ACCENT } else { DIM };
                            if ui
                                .add(egui::Button::new(model_label(family)).fill(fill))
                                .clicked()
                            {
                                setter.begin_set_parameter(&params.model);
                                setter.set_parameter(&params.model, index as i32);
                                setter.end_set_parameter(&params.model);
                            }
                        }
                    });

                    // ── Partial-spectrum visualiser ─────────────────────────
                    ui.add_space(4.0);
                    ui.label(egui::RichText::new("Partial spectrum").weak());
                    draw_spectrum(ui, &state.spectrum, PANEL, GRID, ACCENT);

                    // ── Voice-activity meter + CPU ──────────────────────────
                    let active = voice_meter.load(Ordering::Relaxed) as usize;
                    draw_voice_meter(ui, active, params.polyphony.value() as usize, LIT, DIM);
                    let cpu = f32::from_bits(cpu_meter.load(Ordering::Relaxed));
                    ui.label(format!("CPU {:.1} %", (cpu * 100.0).clamp(0.0, 999.0)));
                    ui.separator();

                    // ── Model-specific + shared/global controls ─────────────
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        egui::Grid::new("physical-controls")
                            .num_columns(2)
                            .spacing(egui::vec2(8.0, 4.0))
                            .show(ui, |ui| {
                                let row = |ui: &mut egui::Ui, label: &str, param: &FloatParam| {
                                    ui.label(label);
                                    ui.add(widgets::ParamSlider::for_param(param, setter));
                                    ui.end_row();
                                };
                                let int_row = |ui: &mut egui::Ui, label: &str, param: &IntParam| {
                                    ui.label(label);
                                    ui.add(widgets::ParamSlider::for_param(param, setter));
                                    ui.end_row();
                                };

                                match model {
                                    Model::Plucked => {
                                        row(ui, "Pluck Position", &params.pluck_position);
                                        row(ui, "Pick Hardness", &params.pick_hardness);
                                        row(ui, "Damping", &params.damping);
                                        int_row(ui, "Body Type", &params.body);
                                        row(ui, "Detune", &params.detune);
                                    }
                                    Model::Modal => {
                                        int_row(ui, "Modes", &params.num_modes);
                                        int_row(ui, "Frequency Ratios", &params.preset);
                                        row(ui, "Decay Slope", &params.decay_slope);
                                        row(ui, "Damping", &params.damping);
                                    }
                                }

                                // Shared + global.
                                row(ui, "Decay", &params.decay_s);
                                row(ui, "Gain", &params.gain_db);
                                row(ui, "Mix", &params.mix);
                                int_row(ui, "Polyphony", &params.polyphony);
                            });
                    });
                });
            },
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        let sample_rate = buffer_config.sample_rate.max(1.0);
        let snapshot = self.params.snapshot();
        self.engine = Some(PhysicalEngine::new(sample_rate, &snapshot));
        context.set_latency_samples(0);
        self.midi.last_damping = self.params.damping.value();
        true
    }

    fn reset(&mut self) {
        if let Some(engine) = self.engine.as_mut() {
            engine.reset();
        }
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let base = self.params.snapshot();
        let snapshot = self.resolve_snapshot(base);
        let num_samples = buffer.samples();

        // Flush denormals for the whole callback — string / modal tails decay
        // through the subnormal range after note-off.
        let _denormal_guard = rt_physical_core::DenormalGuard::new();

        let midi = &mut self.midi;
        if let Some(engine) = self.engine.as_mut() {
            engine.apply_params(&snapshot);

            let start = std::time::Instant::now();
            let mut next_event = context.next_event();
            for (sample_index, mut channel_samples) in buffer.iter_samples().enumerate() {
                // Drain every event stamped for this sample (sample-accurate notes).
                while let Some(event) = next_event {
                    if event.timing() as usize != sample_index {
                        break;
                    }
                    apply_event(engine, midi, &event);
                    #[cfg(feature = "gui")]
                    if matches!(event, NoteEvent::NoteOn { .. }) {
                        self.note_counter.fetch_add(1, Ordering::Relaxed);
                    }
                    next_event = context.next_event();
                }

                let channels = channel_samples.len();
                let (in_l, in_r) = match channels {
                    0 => (0.0, 0.0),
                    1 => {
                        let mono = *channel_samples.get_mut(0).unwrap();
                        (mono, mono)
                    }
                    _ => (
                        *channel_samples.get_mut(0).unwrap(),
                        *channel_samples.get_mut(1).unwrap(),
                    ),
                };
                let (out_l, out_r) = engine.process_sample(in_l, in_r);
                match channels {
                    0 => {}
                    1 => *channel_samples.get_mut(0).unwrap() = 0.5 * (out_l + out_r),
                    _ => {
                        *channel_samples.get_mut(0).unwrap() = out_l;
                        *channel_samples.get_mut(1).unwrap() = out_r;
                    }
                }
            }
            let elapsed = start.elapsed().as_secs_f32();

            // Voices sounding this block → editor activity meter.
            #[cfg(feature = "gui")]
            self.voice_meter
                .store(engine.active_voice_count() as u32, Ordering::Relaxed);

            // Read-only CPU-load proxy (render time / block wall-clock budget).
            let sample_rate = context.transport().sample_rate;
            let budget = (num_samples as f32 / sample_rate).max(1.0e-9);
            self.cpu_meter
                .store((elapsed / budget).to_bits(), Ordering::Relaxed);
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for PhysicalPlugin {
    const CLAP_ID: &'static str = "org.codeberg.slundi.rt-physical";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("Physical-modelling instrument: plucked strings and modal resonators");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::Instrument,
        ClapFeature::Synthesizer,
        ClapFeature::Stereo,
    ];
}

impl Vst3Plugin for PhysicalPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"RtPhysiclSlundi1";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Instrument, Vst3SubCategory::Synth];
}

nih_export_clap!(PhysicalPlugin);
nih_export_vst3!(PhysicalPlugin);
