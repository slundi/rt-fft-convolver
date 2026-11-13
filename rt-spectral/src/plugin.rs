//! `nih-plug` wrapper for `rt-spectral` (compiled only with the `nih` feature).
//!
//! A thin framework layer over the tested, framework-agnostic
//! [`engine`](crate::engine): it declares the automatable parameters, maps them
//! onto the core [`SpectralParams`], drives the off-thread learn from the
//! background-task executor, and runs the [`SpectralProcessor`] on the audio
//! thread.
//!
//! # Threading
//!
//! - The [`SpectralController`] lives behind a `Mutex` shared with the
//!   background-task executor.  Turning a captured noise clip into a
//!   [`NoiseProfile`](rt_spectral_core::NoiseProfile) (which allocates) happens
//!   there, never on the audio thread.
//! - `process()` renders through the [`SpectralProcessor`] — allocation-free —
//!   captures the Learn clip in place, and dispatches
//!   [`SpectralTask::Learn`] to the background thread once a capture completes.
//!
//! # Learn workflow (Phase 2)
//!
//! The `Learn` button is edge-triggered: a rising edge arms a capture of
//! `Learn Time` milliseconds of the mono-summed input.  When the capture fills,
//! the audio thread flags it and the background thread computes + publishes the
//! profile to every channel.  The button is a plain toggle here; the momentary
//! push-button affordance and capture progress meter land with the Phase-3
//! editor.
//!
//! # Latency
//!
//! Constant for the built `fft_size` (one STFT frame + the click despiker's
//! fixed group delay); reported once in [`initialize`](SpectralPlugin::initialize).

use std::num::NonZeroU32;
#[cfg(feature = "gui")]
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use nih_plug::prelude::*;
#[cfg(feature = "gui")]
use nih_plug_egui::{EguiState, create_egui_editor, egui, widgets};
use rt_spectral_core::{
    ClickParams, CompressParams, DenoiseParams, GateParams, HumParams, SpectralParams,
    TransientParams,
};

#[cfg(feature = "gui")]
use crate::engine::SpectralTelemetry;
use crate::engine::{SpectralController, SpectralProcessor, spectral_engine};
use crate::params::{
    ATTACK_MS_RANGE, CLICK_MAX_LEN_RANGE, CLICK_SENSITIVITY_RANGE, COMP_THRESHOLD_DB_RANGE,
    GATE_RANGE_DB_RANGE, GATE_THRESHOLD_DB_RANGE, HARMONICS_RANGE, HUM_FUNDAMENTAL_HZ_RANGE,
    HUM_Q_RANGE, KNEE_DB_RANGE, LEARN_MS_RANGE, MAKEUP_DB_RANGE, OUTPUT_GAIN_DB_RANGE, RATIO_RANGE,
    REDUCTION_DB_RANGE, RELEASE_MS_RANGE, SENSITIVITY_DB_RANGE, clamp_params, learn_samples,
};
#[cfg(feature = "gui")]
use crate::ui::{
    DISPLAY_CEIL_DB, DISPLAY_FLOOR_DB, Meter, SPECTROGRAM_BINS, SPECTROGRAM_COLUMNS, Spectrogram,
    db_to_intensity, downsample_log, format_db, format_latency_ms, latency_ms, magnitude_to_db,
};

/// Fixed STFT size — the Phase-0 "Studio" default (2048).  A host-exposed
/// size selector (Live/Studio/HiRes) is a post-Phase-2 addition; keeping it
/// fixed keeps the reported latency deterministic.
const FFT_SIZE: usize = 2048;

/// Off-thread work dispatched to the background executor (control thread only).
pub enum SpectralTask {
    /// Turn the freshly-captured noise clip into a profile and publish it.
    Learn,
}

/// Denoise controls (Phase 2 "Denoise controls").
#[derive(Params)]
struct DenoiseParamGroup {
    #[id = "dn_on"]
    enabled: BoolParam,
    #[id = "dn_red"]
    reduction: FloatParam,
    #[id = "dn_sens"]
    sensitivity: FloatParam,
}

/// Spectral-gate controls (Phase 2 "Gate controls").
#[derive(Params)]
struct GateParamGroup {
    #[id = "gt_on"]
    enabled: BoolParam,
    #[id = "gt_thr"]
    threshold: FloatParam,
    #[id = "gt_rng"]
    range: FloatParam,
    #[id = "gt_atk"]
    attack: FloatParam,
    #[id = "gt_rel"]
    release: FloatParam,
}

/// Spectral-compressor controls (Phase 2 "Compressor controls").
#[derive(Params)]
struct CompressParamGroup {
    #[id = "cp_on"]
    enabled: BoolParam,
    #[id = "cp_thr"]
    threshold: FloatParam,
    #[id = "cp_ratio"]
    ratio: FloatParam,
    #[id = "cp_knee"]
    knee: FloatParam,
    #[id = "cp_mkup"]
    makeup: FloatParam,
}

/// Hum-removal controls (Phase 2 "Hum controls").
#[derive(Params)]
struct HumParamGroup {
    #[id = "hm_on"]
    enabled: BoolParam,
    #[id = "hm_fund"]
    fundamental: FloatParam,
    #[id = "hm_harm"]
    harmonics: IntParam,
    #[id = "hm_q"]
    q: FloatParam,
}

/// Click-removal controls (Phase 2 "Click controls").
#[derive(Params)]
struct ClickParamGroup {
    #[id = "ck_on"]
    enabled: BoolParam,
    #[id = "ck_sens"]
    sensitivity: FloatParam,
    #[id = "ck_len"]
    max_len: IntParam,
}

/// Host-visible parameters, grouped per operator.
#[derive(Params)]
struct SpectralPluginParams {
    #[nested(group = "Denoise")]
    denoise: DenoiseParamGroup,
    #[nested(group = "Gate")]
    gate: GateParamGroup,
    #[nested(group = "Compressor")]
    compress: CompressParamGroup,
    #[nested(group = "Hum")]
    hum: HumParamGroup,
    #[nested(group = "Click")]
    click: ClickParamGroup,

    #[id = "learn"]
    learn: BoolParam,
    #[id = "learn_ms"]
    learn_ms: FloatParam,

    #[id = "mix"]
    mix: FloatParam,
    #[id = "outgain"]
    output_gain: FloatParam,
    #[id = "byp"]
    bypass: BoolParam,

    #[cfg(feature = "gui")]
    #[persist = "editor-state"]
    editor_state: Arc<EguiState>,
}

/// Build a `FloatParam` over an inclusive `(min, max)` range.
fn float(name: &str, default: f32, range: (f32, f32)) -> FloatParam {
    FloatParam::new(
        name,
        default,
        FloatRange::Linear {
            min: range.0,
            max: range.1,
        },
    )
}

impl Default for SpectralPluginParams {
    fn default() -> Self {
        let defaults = SpectralParams::default();

        Self {
            denoise: DenoiseParamGroup {
                enabled: BoolParam::new("Denoise", defaults.denoise.enabled),
                reduction: float(
                    "Reduction",
                    defaults.denoise.reduction_db,
                    REDUCTION_DB_RANGE,
                )
                .with_unit(" dB"),
                sensitivity: float(
                    "Sensitivity",
                    defaults.denoise.sensitivity_db,
                    SENSITIVITY_DB_RANGE,
                )
                .with_unit(" dB"),
            },
            gate: GateParamGroup {
                enabled: BoolParam::new("Spectral Gate", defaults.gate.enabled),
                threshold: float(
                    "Threshold",
                    defaults.gate.threshold_db,
                    GATE_THRESHOLD_DB_RANGE,
                )
                .with_unit(" dB"),
                range: float("Range", defaults.gate.range_db, GATE_RANGE_DB_RANGE).with_unit(" dB"),
                attack: float("Attack", defaults.gate.attack_ms, ATTACK_MS_RANGE).with_unit(" ms"),
                release: float("Release", defaults.gate.release_ms, RELEASE_MS_RANGE)
                    .with_unit(" ms"),
            },
            compress: CompressParamGroup {
                enabled: BoolParam::new("Spectral Compressor", defaults.compress.enabled),
                threshold: float(
                    "Threshold",
                    defaults.compress.threshold_db,
                    COMP_THRESHOLD_DB_RANGE,
                )
                .with_unit(" dB"),
                ratio: float("Ratio", defaults.compress.ratio, RATIO_RANGE).with_unit(":1"),
                knee: float("Knee", defaults.compress.knee_db, KNEE_DB_RANGE).with_unit(" dB"),
                makeup: float("Makeup", defaults.compress.makeup_db, MAKEUP_DB_RANGE)
                    .with_unit(" dB"),
            },
            hum: HumParamGroup {
                enabled: BoolParam::new("Hum Removal", defaults.hum.enabled),
                fundamental: float(
                    "Fundamental",
                    defaults.hum.fundamental_hz,
                    HUM_FUNDAMENTAL_HZ_RANGE,
                )
                .with_unit(" Hz"),
                harmonics: IntParam::new(
                    "Harmonics",
                    defaults.hum.harmonics as i32,
                    IntRange::Linear {
                        min: HARMONICS_RANGE.0 as i32,
                        max: HARMONICS_RANGE.1 as i32,
                    },
                ),
                q: float("Notch Q", defaults.hum.q, HUM_Q_RANGE),
            },
            click: ClickParamGroup {
                enabled: BoolParam::new("Click Removal", defaults.click.enabled),
                sensitivity: float(
                    "Sensitivity",
                    defaults.click.sensitivity,
                    CLICK_SENSITIVITY_RANGE,
                ),
                max_len: IntParam::new(
                    "Max Length",
                    defaults.click.max_len as i32,
                    IntRange::Linear {
                        min: CLICK_MAX_LEN_RANGE.0 as i32,
                        max: CLICK_MAX_LEN_RANGE.1 as i32,
                    },
                )
                .with_unit(" smp"),
            },

            learn: BoolParam::new("Learn", false),
            learn_ms: float("Learn Time", 1000.0, LEARN_MS_RANGE).with_unit(" ms"),

            mix: float("Mix", defaults.mix, (0.0, 1.0))
                .with_unit(" %")
                .with_value_to_string(formatters::v2s_f32_percentage(0))
                .with_string_to_value(formatters::s2v_f32_percentage()),
            output_gain: float("Output Gain", defaults.output_gain_db, OUTPUT_GAIN_DB_RANGE)
                .with_unit(" dB"),
            bypass: BoolParam::new("Bypass", false)
                .with_value_to_string(formatters::v2s_bool_bypass()),

            #[cfg(feature = "gui")]
            editor_state: EguiState::from_size(620, 720),
        }
    }
}

impl SpectralPluginParams {
    /// Snapshot into the framework-agnostic core model, clamped into range.
    /// Transient soft-bypass runs with its Phase-0 defaults (always on).
    fn snapshot(&self) -> SpectralParams {
        clamp_params(SpectralParams {
            click: ClickParams {
                enabled: self.click.enabled.value(),
                sensitivity: self.click.sensitivity.value(),
                max_len: self.click.max_len.value() as usize,
            },
            denoise: DenoiseParams {
                enabled: self.denoise.enabled.value(),
                reduction_db: self.denoise.reduction.value(),
                sensitivity_db: self.denoise.sensitivity.value(),
            },
            gate: GateParams {
                enabled: self.gate.enabled.value(),
                threshold_db: self.gate.threshold.value(),
                range_db: self.gate.range.value(),
                attack_ms: self.gate.attack.value(),
                release_ms: self.gate.release.value(),
            },
            compress: CompressParams {
                enabled: self.compress.enabled.value(),
                threshold_db: self.compress.threshold.value(),
                ratio: self.compress.ratio.value(),
                knee_db: self.compress.knee.value(),
                makeup_db: self.compress.makeup.value(),
            },
            transient: TransientParams::default(),
            hum: HumParams {
                enabled: self.hum.enabled.value(),
                fundamental_hz: self.hum.fundamental.value(),
                harmonics: self.hum.harmonics.value() as usize,
                q: self.hum.q.value(),
            },
            mix: self.mix.value(),
            output_gain_db: self.output_gain.value(),
            bypass: self.bypass.value(),
        })
    }
}

/// The spectral plugin instance.
pub struct SpectralPlugin {
    params: Arc<SpectralPluginParams>,
    /// Shared with the background-task executor **and** the editor (profile
    /// overlay); built in `initialize`.
    controller: Arc<Mutex<Option<SpectralController>>>,
    /// Audio-thread renderer; built in `initialize`.
    processor: Option<SpectralProcessor>,
    /// Audio → editor telemetry reader, shared into the `egui` editor closure;
    /// built in `initialize` (only needed by the editor).
    #[cfg(feature = "gui")]
    telemetry: Arc<Mutex<Option<SpectralTelemetry>>>,

    /// Host sample rate, for converting `Learn Time` ms → samples.
    sample_rate: u32,
    /// Last `Learn` param value, for rising-edge detection.
    last_learn: bool,

    /// Reported latency in samples — the editor's latency badge (`f32` bits).
    #[cfg(feature = "gui")]
    latency_samples: Arc<AtomicU32>,
    /// Host sample rate (`f32` bits) — converts the latency badge to ms.
    #[cfg(feature = "gui")]
    sample_rate_hz: Arc<AtomicU32>,
}

impl Default for SpectralPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(SpectralPluginParams::default()),
            controller: Arc::new(Mutex::new(None)),
            processor: None,
            #[cfg(feature = "gui")]
            telemetry: Arc::new(Mutex::new(None)),
            sample_rate: 48_000,
            last_learn: false,
            #[cfg(feature = "gui")]
            latency_samples: Arc::new(AtomicU32::new(0)),
            #[cfg(feature = "gui")]
            sample_rate_hz: Arc::new(AtomicU32::new(48_000.0f32.to_bits())),
        }
    }
}

// ─── egui editor (Phase 3, `gui` feature) ──────────────────────────────────

/// Per-open editor state owned by the egui closure: the rolling spectrogram
/// plus the display-smoothed meters and the reusable spectrogram texture.
#[cfg(feature = "gui")]
struct SpectralEditorState {
    spectrogram: Spectrogram,
    texture: Option<egui::TextureHandle>,
    input_meter: Meter,
    output_meter: Meter,
    gr_denoise: Meter,
    gr_gate: Meter,
    gr_compress: Meter,
    /// Momentary-Learn phase: `0` idle, `1` armed (waiting for capture start),
    /// `2` capturing (waiting for completion to release the param).
    learn_phase: u8,
}

#[cfg(feature = "gui")]
impl SpectralEditorState {
    fn new() -> Self {
        Self {
            spectrogram: Spectrogram::new(SPECTROGRAM_BINS, SPECTROGRAM_COLUMNS),
            texture: None,
            input_meter: Meter::new(DISPLAY_FLOOR_DB, 3.0),
            output_meter: Meter::new(DISPLAY_FLOOR_DB, 3.0),
            gr_denoise: Meter::new(-24.0, 1.5),
            gr_gate: Meter::new(-24.0, 1.5),
            gr_compress: Meter::new(-24.0, 1.5),
            learn_phase: 0,
        }
    }
}

/// Outcome of the momentary-Learn state machine for one editor frame.
#[cfg(feature = "gui")]
enum LearnUi {
    /// Nothing to do this frame.
    Idle,
    /// The capture just finished — release the (momentary) `Learn` parameter.
    Release,
}

/// Advance the momentary-Learn state machine: once a capture we armed has
/// started and then finished, ask the caller to release the `Learn` param so a
/// later click re-triggers it.
#[cfg(feature = "gui")]
fn state_learn_phase(state: &mut SpectralEditorState, learning: bool) -> LearnUi {
    match state.learn_phase {
        1 if learning => {
            state.learn_phase = 2;
            LearnUi::Idle
        }
        2 if !learning => {
            state.learn_phase = 0;
            LearnUi::Release
        }
        _ => LearnUi::Idle,
    }
}

/// Draw the spectrum plot: the current output spectrum (accent) with the
/// learned noise profile overlaid (profile colour).  `x` is log-frequency
/// (display bins), `y` is dB between [`DISPLAY_FLOOR_DB`] and [`DISPLAY_CEIL_DB`].
#[cfg(feature = "gui")]
fn draw_spectrum(
    ui: &mut egui::Ui,
    spectrum_db: &[f32],
    profile_db: &[f32],
    panel: egui::Color32,
    grid: egui::Color32,
    accent: egui::Color32,
    profile: egui::Color32,
) {
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), 100.0),
        egui::Sense::hover(),
    );
    ui.painter().rect_filled(rect, 4.0, panel);

    // Horizontal grid at −20 / −40 / −60 dB.
    for &db in &[-20.0f32, -40.0, -60.0] {
        let intensity = db_to_intensity(db, DISPLAY_FLOOR_DB, DISPLAY_CEIL_DB);
        let y = rect.bottom() - intensity * rect.height();
        ui.painter().line_segment(
            [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
            egui::Stroke::new(1.0, grid),
        );
    }

    let to_points = |values: &[f32]| -> Vec<egui::Pos2> {
        if values.len() < 2 {
            return Vec::new();
        }
        let last = (values.len() - 1) as f32;
        values
            .iter()
            .enumerate()
            .map(|(index, &db)| {
                let x = rect.left() + index as f32 / last * rect.width();
                let intensity = db_to_intensity(db, DISPLAY_FLOOR_DB, DISPLAY_CEIL_DB);
                let y = rect.bottom() - intensity * rect.height();
                egui::pos2(x, y)
            })
            .collect()
    };

    let profile_points = to_points(profile_db);
    if !profile_points.is_empty() {
        ui.painter().add(egui::Shape::line(
            profile_points,
            egui::Stroke::new(1.5, profile),
        ));
    }
    let spectrum_points = to_points(spectrum_db);
    if !spectrum_points.is_empty() {
        ui.painter().add(egui::Shape::line(
            spectrum_points,
            egui::Stroke::new(2.0, accent),
        ));
    }
}

/// Lay out every operator's controls in labelled groups.
#[cfg(feature = "gui")]
fn operator_controls(ui: &mut egui::Ui, setter: &ParamSetter, params: &SpectralPluginParams) {
    let slider = |ui: &mut egui::Ui, label: &str, add: &dyn Fn(&mut egui::Ui)| {
        ui.label(label);
        add(ui);
        ui.end_row();
    };

    let group = |ui: &mut egui::Ui, title: &str, body: &dyn Fn(&mut egui::Ui)| {
        ui.add_space(2.0);
        ui.label(egui::RichText::new(title).strong());
        egui::Grid::new(title)
            .num_columns(2)
            .spacing(egui::vec2(8.0, 4.0))
            .show(ui, |ui| body(ui));
        ui.add_space(6.0);
    };

    group(ui, "Denoise", &|ui| {
        slider(ui, "Enabled", &|ui| {
            ui.add(widgets::ParamSlider::for_param(
                &params.denoise.enabled,
                setter,
            ));
        });
        slider(ui, "Reduction", &|ui| {
            ui.add(widgets::ParamSlider::for_param(
                &params.denoise.reduction,
                setter,
            ));
        });
        slider(ui, "Sensitivity", &|ui| {
            ui.add(widgets::ParamSlider::for_param(
                &params.denoise.sensitivity,
                setter,
            ));
        });
    });
    group(ui, "Spectral Gate", &|ui| {
        slider(ui, "Enabled", &|ui| {
            ui.add(widgets::ParamSlider::for_param(
                &params.gate.enabled,
                setter,
            ));
        });
        slider(ui, "Threshold", &|ui| {
            ui.add(widgets::ParamSlider::for_param(
                &params.gate.threshold,
                setter,
            ));
        });
        slider(ui, "Range", &|ui| {
            ui.add(widgets::ParamSlider::for_param(&params.gate.range, setter));
        });
        slider(ui, "Attack", &|ui| {
            ui.add(widgets::ParamSlider::for_param(&params.gate.attack, setter));
        });
        slider(ui, "Release", &|ui| {
            ui.add(widgets::ParamSlider::for_param(
                &params.gate.release,
                setter,
            ));
        });
    });
    group(ui, "Spectral Compressor", &|ui| {
        slider(ui, "Enabled", &|ui| {
            ui.add(widgets::ParamSlider::for_param(
                &params.compress.enabled,
                setter,
            ));
        });
        slider(ui, "Threshold", &|ui| {
            ui.add(widgets::ParamSlider::for_param(
                &params.compress.threshold,
                setter,
            ));
        });
        slider(ui, "Ratio", &|ui| {
            ui.add(widgets::ParamSlider::for_param(
                &params.compress.ratio,
                setter,
            ));
        });
        slider(ui, "Knee", &|ui| {
            ui.add(widgets::ParamSlider::for_param(
                &params.compress.knee,
                setter,
            ));
        });
        slider(ui, "Makeup", &|ui| {
            ui.add(widgets::ParamSlider::for_param(
                &params.compress.makeup,
                setter,
            ));
        });
    });
    group(ui, "Hum Removal", &|ui| {
        slider(ui, "Enabled", &|ui| {
            ui.add(widgets::ParamSlider::for_param(&params.hum.enabled, setter));
        });
        slider(ui, "Fundamental", &|ui| {
            ui.add(widgets::ParamSlider::for_param(
                &params.hum.fundamental,
                setter,
            ));
        });
        slider(ui, "Harmonics", &|ui| {
            ui.add(widgets::ParamSlider::for_param(
                &params.hum.harmonics,
                setter,
            ));
        });
        slider(ui, "Notch Q", &|ui| {
            ui.add(widgets::ParamSlider::for_param(&params.hum.q, setter));
        });
    });
    group(ui, "Click Removal", &|ui| {
        slider(ui, "Enabled", &|ui| {
            ui.add(widgets::ParamSlider::for_param(
                &params.click.enabled,
                setter,
            ));
        });
        slider(ui, "Sensitivity", &|ui| {
            ui.add(widgets::ParamSlider::for_param(
                &params.click.sensitivity,
                setter,
            ));
        });
        slider(ui, "Max Length", &|ui| {
            ui.add(widgets::ParamSlider::for_param(
                &params.click.max_len,
                setter,
            ));
        });
    });
    group(ui, "Global", &|ui| {
        slider(ui, "Mix", &|ui| {
            ui.add(widgets::ParamSlider::for_param(&params.mix, setter));
        });
        slider(ui, "Output Gain", &|ui| {
            ui.add(widgets::ParamSlider::for_param(&params.output_gain, setter));
        });
    });
}

/// Map a `0.0..=1.0` intensity to an inferno-ish heat colour `[r, g, b]`.
#[cfg(feature = "gui")]
fn intensity_color(t: f32) -> [u8; 3] {
    // Piecewise ramp: black → indigo → magenta → orange → white.
    const STOPS: [(f32, [f32; 3]); 5] = [
        (0.00, [0.0, 0.0, 0.0]),
        (0.25, [0.28, 0.05, 0.42]),
        (0.55, [0.75, 0.15, 0.45]),
        (0.82, [0.98, 0.55, 0.18]),
        (1.00, [1.0, 1.0, 0.92]),
    ];
    let t = t.clamp(0.0, 1.0);
    let mut lo = &STOPS[0];
    let mut hi = &STOPS[STOPS.len() - 1];
    for pair in STOPS.windows(2) {
        if t >= pair[0].0 && t <= pair[1].0 {
            lo = &pair[0];
            hi = &pair[1];
            break;
        }
    }
    let span = (hi.0 - lo.0).max(1e-6);
    let frac = ((t - lo.0) / span).clamp(0.0, 1.0);
    let lerp = |a: f32, b: f32| ((a + (b - a) * frac) * 255.0).round() as u8;
    [
        lerp(lo.1[0], hi.1[0]),
        lerp(lo.1[1], hi.1[1]),
        lerp(lo.1[2], hi.1[2]),
    ]
}

/// Build an RGBA spectrogram image with the newest column at the right edge and
/// low frequencies at the bottom.
#[cfg(feature = "gui")]
fn build_spectrogram_image(spec: &Spectrogram) -> egui::ColorImage {
    let cols = spec.capacity();
    let bins = spec.bins();
    let mut rgba = vec![0u8; cols * bins * 4];
    let offset = cols - spec.len(); // newest flush against the right edge
    for ordered in 0..spec.len() {
        let Some(column) = spec.column(ordered) else {
            continue;
        };
        let x = offset + ordered;
        for (bin, &magnitude) in column.iter().enumerate() {
            let db = magnitude_to_db(magnitude, DISPLAY_FLOOR_DB);
            let intensity = db_to_intensity(db, DISPLAY_FLOOR_DB, DISPLAY_CEIL_DB);
            let [r, g, b] = intensity_color(intensity);
            let y = bins - 1 - bin; // low freq at the bottom
            let index = (y * cols + x) * 4;
            rgba[index] = r;
            rgba[index + 1] = g;
            rgba[index + 2] = b;
            rgba[index + 3] = 255;
        }
    }
    egui::ColorImage::from_rgba_unmultiplied([cols, bins], &rgba)
}

/// Draw a labelled horizontal meter bar filling `fraction` of its width.
#[cfg(feature = "gui")]
fn draw_meter_bar(
    ui: &mut egui::Ui,
    label: &str,
    value: &str,
    fraction: f32,
    fill: egui::Color32,
    panel: egui::Color32,
) {
    ui.horizontal(|ui| {
        ui.add_sized(egui::vec2(46.0, 14.0), egui::Label::new(label));
        let (rect, _) = ui.allocate_exact_size(
            egui::vec2(ui.available_width() - 74.0, 14.0),
            egui::Sense::hover(),
        );
        ui.painter().rect_filled(rect, 3.0, panel);
        let mut filled = rect;
        filled.set_width(rect.width() * fraction.clamp(0.0, 1.0));
        ui.painter().rect_filled(filled, 3.0, fill);
        ui.add_sized(egui::vec2(66.0, 14.0), egui::Label::new(value));
    });
}

impl Plugin for SpectralPlugin {
    const NAME: &'static str = "RT Spectral";
    const VENDOR: &'static str = "Sébastien Lundi";
    const URL: &'static str = "https://codeberg.org/slundi/rt-fft-convolver";
    const EMAIL: &'static str = "slundi@stamus-networks.com";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    // Stereo → stereo and mono → mono (independent per channel), per Phase 2.
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

    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = SpectralTask;

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    #[cfg(feature = "gui")]
    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        use egui::Color32;

        // Teal accent for a spectral / restoration plugin.
        const ACCENT: Color32 = Color32::from_rgb(0x35, 0xc9, 0xb0);
        const PROFILE: Color32 = Color32::from_rgb(0xff, 0x9a, 0x3d);
        const PANEL: Color32 = Color32::from_rgb(0x14, 0x16, 0x1a);
        const GRID: Color32 = Color32::from_rgb(0x2a, 0x2e, 0x36);
        const BAR_BG: Color32 = Color32::from_rgb(0x22, 0x25, 0x2b);
        const GR_COLOR: Color32 = Color32::from_rgb(0xd8, 0x6a, 0x4a);

        let params = self.params.clone();
        let telemetry = self.telemetry.clone();
        let controller = self.controller.clone();
        let latency_samples = self.latency_samples.clone();
        let sample_rate_hz = self.sample_rate_hz.clone();

        create_egui_editor(
            self.params.editor_state.clone(),
            SpectralEditorState::new(),
            |_ctx, _state| {},
            move |ctx, setter, state| {
                // ── 1. Pull the freshest audio-thread telemetry. ──
                let mut spectrum_db = vec![DISPLAY_FLOOR_DB; SPECTROGRAM_BINS];
                let mut meters = rt_spectral_core::EngineMeters::default();
                let mut learning = false;
                let mut progress = 0.0f32;
                if let Ok(mut guard) = telemetry.lock() {
                    if let Some(tele) = guard.as_mut() {
                        let frame = tele.latest();
                        state.spectrogram.push_column(&frame.spectrum);
                        for (dst, &magnitude) in spectrum_db.iter_mut().zip(&frame.spectrum) {
                            *dst = magnitude_to_db(magnitude, DISPLAY_FLOOR_DB);
                        }
                        meters = frame.meters;
                        learning = frame.learning;
                        progress = frame.learn_progress;
                    }
                }

                // ── 2. Momentary Learn state machine (button → param edge). ──
                if let LearnUi::Release = state_learn_phase(state, learning) {
                    setter.begin_set_parameter(&params.learn);
                    setter.set_parameter(&params.learn, false);
                    setter.end_set_parameter(&params.learn);
                }

                // ── 3. Noise-profile overlay (editor-thread lock, never RT). ──
                let mut profile_db: Vec<f32> = Vec::new();
                if let Ok(guard) = controller.try_lock() {
                    if let Some(ctrl) = guard.as_ref() {
                        let mags: Vec<f32> =
                            ctrl.profile_power().iter().map(|&p| p.sqrt()).collect();
                        let mut downsampled = vec![0.0f32; SPECTROGRAM_BINS];
                        downsample_log(&mags, &mut downsampled);
                        if downsampled.iter().any(|&m| m > 0.0) {
                            profile_db = downsampled
                                .iter()
                                .map(|&m| magnitude_to_db(m, DISPLAY_FLOOR_DB))
                                .collect();
                        }
                    }
                }

                // Feed the display-smoothed meters.
                let in_level = state.input_meter.push(meters.input_db);
                let out_level = state.output_meter.push(meters.output_db);
                let gr_dn = state.gr_denoise.push(meters.gr_denoise_db);
                let gr_gt = state.gr_gate.push(meters.gr_gate_db);
                let gr_cp = state.gr_compress.push(meters.gr_compress_db);

                egui::CentralPanel::default().show(ctx, |ui| {
                    // ── Header ────────────────────────────────────────────
                    ui.horizontal(|ui| {
                        ui.heading("RT Spectral");
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.add(widgets::ParamSlider::for_param(&params.bypass, setter));
                            ui.label("Bypass");
                        });
                    });
                    ui.separator();

                    // ── Spectrogram (time × frequency heat-map) ───────────
                    let image = build_spectrogram_image(&state.spectrogram);
                    match &mut state.texture {
                        Some(handle) => handle.set(image, egui::TextureOptions::LINEAR),
                        none => {
                            *none = Some(ctx.load_texture(
                                "spectrogram",
                                image,
                                egui::TextureOptions::LINEAR,
                            ));
                        }
                    }
                    let (rect, _) = ui.allocate_exact_size(
                        egui::vec2(ui.available_width(), 170.0),
                        egui::Sense::hover(),
                    );
                    if let Some(handle) = &state.texture {
                        ui.painter().image(
                            handle.id(),
                            rect,
                            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                            Color32::WHITE,
                        );
                    }

                    // ── Spectrum + noise-profile overlay ──────────────────
                    ui.add_space(4.0);
                    draw_spectrum(ui, &spectrum_db, &profile_db, PANEL, GRID, ACCENT, PROFILE);

                    // ── Meters: input / output / per-operator GR ──────────
                    ui.separator();
                    let level_frac =
                        |db: f32| db_to_intensity(db, DISPLAY_FLOOR_DB, DISPLAY_CEIL_DB);
                    let gr_frac = |db: f32| (-db / 24.0).clamp(0.0, 1.0);
                    draw_meter_bar(
                        ui,
                        "In",
                        &format_db(in_level, DISPLAY_FLOOR_DB),
                        level_frac(in_level),
                        ACCENT,
                        BAR_BG,
                    );
                    draw_meter_bar(
                        ui,
                        "Out",
                        &format_db(out_level, DISPLAY_FLOOR_DB),
                        level_frac(out_level),
                        ACCENT,
                        BAR_BG,
                    );
                    draw_meter_bar(
                        ui,
                        "GR Dn",
                        &format_db(gr_dn, -120.0),
                        gr_frac(gr_dn),
                        GR_COLOR,
                        BAR_BG,
                    );
                    draw_meter_bar(
                        ui,
                        "GR Gt",
                        &format_db(gr_gt, -120.0),
                        gr_frac(gr_gt),
                        GR_COLOR,
                        BAR_BG,
                    );
                    draw_meter_bar(
                        ui,
                        "GR Cp",
                        &format_db(gr_cp, -120.0),
                        gr_frac(gr_cp),
                        GR_COLOR,
                        BAR_BG,
                    );

                    // ── Per-mode enable indicators ────────────────────────
                    ui.separator();
                    ui.horizontal(|ui| {
                        let indicator = |ui: &mut egui::Ui, name: &str, on: bool| {
                            let colour = if on { ACCENT } else { GRID };
                            let (dot, _) = ui
                                .allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                            ui.painter().circle_filled(dot.center(), 4.0, colour);
                            ui.label(name);
                        };
                        indicator(ui, "Denoise", params.denoise.enabled.value());
                        indicator(ui, "Gate", params.gate.enabled.value());
                        indicator(ui, "Comp", params.compress.enabled.value());
                        indicator(ui, "Hum", params.hum.enabled.value());
                        indicator(ui, "Click", params.click.enabled.value());
                    });

                    // ── Learn + latency badge ─────────────────────────────
                    ui.separator();
                    ui.horizontal(|ui| {
                        let learn_button = ui.add_enabled(!learning, egui::Button::new("Learn"));
                        if learn_button.clicked() {
                            setter.begin_set_parameter(&params.learn);
                            setter.set_parameter(&params.learn, true);
                            setter.end_set_parameter(&params.learn);
                            state.learn_phase = 1;
                        }
                        if learning {
                            ui.add(egui::ProgressBar::new(progress).desired_width(120.0));
                        } else {
                            ui.label("idle");
                        }
                        ui.add(widgets::ParamSlider::for_param(&params.learn_ms, setter));

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let samples = latency_samples.load(Ordering::Relaxed) as usize;
                            let rate = f32::from_bits(sample_rate_hz.load(Ordering::Relaxed));
                            ui.label(
                                egui::RichText::new(format_latency_ms(latency_ms(samples, rate)))
                                    .color(ACCENT),
                            );
                            ui.label("Latency");
                        });
                    });

                    // ── Operator controls ─────────────────────────────────
                    ui.separator();
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        operator_controls(ui, setter, &params);
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
                SpectralTask::Learn => {
                    controller.learn();
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
        let sample_rate = buffer_config.sample_rate.round().max(1.0) as u32;
        let max_block = (buffer_config.max_buffer_size as usize).max(1);
        let channels = audio_io_layout
            .main_output_channels
            .map(|count| count.get() as usize)
            .unwrap_or(1);

        let (controller, processor, telemetry) =
            spectral_engine(sample_rate, FFT_SIZE, max_block, channels);

        // Latency is constant for the fixed FFT size; report it once.
        let latency = processor.latency_samples();
        context.set_latency_samples(latency as u32);

        #[cfg(feature = "gui")]
        {
            self.latency_samples
                .store(latency as u32, Ordering::Relaxed);
            self.sample_rate_hz
                .store((sample_rate as f32).to_bits(), Ordering::Relaxed);
        }

        self.sample_rate = sample_rate;
        *self.controller.lock().unwrap() = Some(controller);
        self.processor = Some(processor);

        #[cfg(feature = "gui")]
        {
            *self.telemetry.lock().unwrap() = Some(telemetry);
        }
        #[cfg(not(feature = "gui"))]
        {
            // The telemetry stream is only consumed by the editor.
            let _ = telemetry;
        }

        self.last_learn = self.params.learn.value();
        true
    }

    fn reset(&mut self) {
        if let Some(processor) = self.processor.as_mut() {
            processor.reset();
        }
        self.last_learn = self.params.learn.value();
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let snapshot = self.params.snapshot();

        // Flush denormals for the whole callback (near-silent spectral bins).
        let _denormal_guard = rt_fft_convolver::DenormalGuard::new();

        if let Some(processor) = self.processor.as_mut() {
            // Rising edge on `Learn` arms a capture of `Learn Time` ms.
            let learn = self.params.learn.value();
            if learn && !self.last_learn {
                let samples = learn_samples(self.params.learn_ms.value(), self.sample_rate);
                processor.begin_learn(samples);
            }
            self.last_learn = learn;

            processor.process(buffer.as_slice(), &snapshot);

            // A completed capture triggers the off-thread learn + publish.
            if processor.take_pending_learn() {
                context.execute_background(SpectralTask::Learn);
            }
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for SpectralPlugin {
    const CLAP_ID: &'static str = "org.codeberg.slundi.rt-spectral";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("Real-time spectral processing (learn-seeded denoise, gate, compressor, hum, click)");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Mono,
        ClapFeature::Restoration,
        ClapFeature::Utility,
    ];
}

impl Vst3Plugin for SpectralPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"RtSpectrlSlnd001";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Restoration];
}

nih_export_clap!(SpectralPlugin);
nih_export_vst3!(SpectralPlugin);
