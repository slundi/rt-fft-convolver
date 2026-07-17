//! `nih-plug` wrapper for `rt-adaptive` (compiled only with the `nih` feature).
//!
//! A thin framework layer over the tested, framework-agnostic
//! [`engine`](crate::engine): it declares the automatable parameters, routes the
//! structural (filter-length) change into the off-thread controller, runs the
//! [`AdaptiveProcessor`](crate::engine::AdaptiveProcessor) on the audio thread,
//! and reports zero latency (every mode is zero-latency in the direct path).
//!
//! # I/O (Phase 2 decision)
//!
//! - Stereo main in → stereo out, plus a **mono sidechain** input carrying the
//!   AEC far-end reference.
//! - Mono main in → mono out (also with the sidechain).
//!
//! The sidechain is only consumed in `Echo Canceller` mode.
//!
//! # MIDI / footswitch
//!
//! `Freeze` and `Reset` are exposed as automatable `BoolParam`s; a host binds a
//! footswitch CC to them via MIDI learn (the idiomatic `nih-plug` path), so no
//! plugin-side MIDI handling is needed.
//!
//! # Threading
//!
//! The [`AdaptiveController`](crate::engine::AdaptiveController) lives behind a
//! `Mutex` shared with the `nih-plug` background-task executor; rebuilding the
//! echo bank (which reallocates the tap vectors) happens there, never on the
//! audio thread.

use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use nih_plug::prelude::*;
#[cfg(feature = "gui")]
use nih_plug_egui::{EguiState, create_egui_editor, egui, widgets};

use crate::engine::{AdaptiveController, AdaptiveProcessor};
use crate::params::{
    AdaptiveParams, CONVERGENCE_RANGE, DOUBLE_TALK_DB_RANGE, FilterLength, HUM_FUNDAMENTAL_RANGE,
    HUM_Q_RANGE, Mode, NOTCH_Q_RANGE, OUTPUT_GAIN_DB_RANGE, SENSITIVITY_RANGE,
};
#[cfg(feature = "gui")]
use crate::ui::{
    MidiActivity, MidiStatus, SPECTRUM_DISPLAY_BINS, format_age, format_hz, format_q,
    freq_fraction, spectral_heights,
};
#[cfg(feature = "gui")]
use rt_adaptive_core::MAX_NOTCHES;

/// Off-thread work dispatched to the background executor (control thread only).
pub enum AdaptiveTask {
    /// Rebuild the echo bank for a new filter length.
    ApplyStructural(AdaptiveParams),
}

/// `nih-plug` mode enum, mirroring [`Mode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
enum ModeParam {
    #[name = "Feedback Suppressor"]
    Feedback,
    #[name = "Hum Remover"]
    Hum,
    #[name = "Echo Canceller"]
    Echo,
}

impl ModeParam {
    fn to_mode(self) -> Mode {
        match self {
            ModeParam::Feedback => Mode::FeedbackSuppressor,
            ModeParam::Hum => Mode::HumRemover,
            ModeParam::Echo => Mode::EchoCanceller,
        }
    }
}

/// `nih-plug` filter-length enum, mirroring [`FilterLength`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
enum FilterLengthParam {
    #[name = "256"]
    L256,
    #[name = "512"]
    L512,
    #[name = "1024"]
    L1024,
    #[name = "2048"]
    L2048,
}

impl FilterLengthParam {
    fn to_length(self) -> FilterLength {
        match self {
            FilterLengthParam::L256 => FilterLength::L256,
            FilterLengthParam::L512 => FilterLength::L512,
            FilterLengthParam::L1024 => FilterLength::L1024,
            FilterLengthParam::L2048 => FilterLength::L2048,
        }
    }
}

/// Host-visible parameters.
#[derive(Params)]
struct AdaptivePluginParams {
    #[id = "mode"]
    mode: EnumParam<ModeParam>,

    // ── Feedback suppressor ──
    #[id = "sens"]
    sensitivity: FloatParam,
    #[id = "maxn"]
    max_notches: IntParam,
    #[id = "nq"]
    notch_q: FloatParam,
    #[id = "auto"]
    auto: BoolParam,
    #[id = "freeze"]
    freeze: BoolParam,
    #[id = "reset"]
    reset: BoolParam,

    // ── Hum remover ──
    #[id = "humhz"]
    hum_fundamental_hz: FloatParam,
    #[id = "humharm"]
    hum_harmonics: IntParam,
    #[id = "humadapt"]
    hum_adaptive: BoolParam,
    #[id = "humq"]
    hum_q: FloatParam,

    // ── Echo canceller ──
    #[id = "flen"]
    filter_length: EnumParam<FilterLengthParam>,
    #[id = "conv"]
    convergence: FloatParam,
    #[id = "dtalk"]
    double_talk_db: FloatParam,

    // ── Global ──
    #[id = "gain"]
    output_gain_db: FloatParam,
    #[id = "byp"]
    bypass: BoolParam,

    #[cfg(feature = "gui")]
    #[persist = "editor-state"]
    editor_state: Arc<EguiState>,
}

impl Default for AdaptivePluginParams {
    fn default() -> Self {
        let defaults = AdaptiveParams::default();
        Self {
            mode: EnumParam::new("Mode", ModeParam::Feedback),
            sensitivity: FloatParam::new(
                "Sensitivity",
                defaults.sensitivity,
                FloatRange::Linear {
                    min: SENSITIVITY_RANGE.0,
                    max: SENSITIVITY_RANGE.1,
                },
            )
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage()),
            max_notches: IntParam::new(
                "Max Notches",
                defaults.max_notches as i32,
                IntRange::Linear {
                    min: 1,
                    max: rt_adaptive_core::MAX_NOTCHES as i32,
                },
            ),
            notch_q: FloatParam::new(
                "Notch Q",
                defaults.notch_q,
                FloatRange::Skewed {
                    min: NOTCH_Q_RANGE.0,
                    max: NOTCH_Q_RANGE.1,
                    factor: FloatRange::skew_factor(-1.0),
                },
            ),
            auto: BoolParam::new("Auto", defaults.auto),
            freeze: BoolParam::new("Freeze", defaults.freeze),
            reset: BoolParam::new("Reset", defaults.reset),
            hum_fundamental_hz: FloatParam::new(
                "Hum Fundamental",
                defaults.hum_fundamental_hz,
                FloatRange::Linear {
                    min: HUM_FUNDAMENTAL_RANGE.0,
                    max: HUM_FUNDAMENTAL_RANGE.1,
                },
            )
            .with_unit(" Hz"),
            hum_harmonics: IntParam::new(
                "Hum Harmonics",
                defaults.hum_harmonics as i32,
                IntRange::Linear { min: 1, max: 10 },
            ),
            hum_adaptive: BoolParam::new("Hum Adaptive", defaults.hum_adaptive),
            hum_q: FloatParam::new(
                "Hum Q",
                defaults.hum_q,
                FloatRange::Skewed {
                    min: HUM_Q_RANGE.0,
                    max: HUM_Q_RANGE.1,
                    factor: FloatRange::skew_factor(-1.0),
                },
            ),
            filter_length: EnumParam::new("Filter Length", FilterLengthParam::L512),
            convergence: FloatParam::new(
                "Convergence",
                defaults.convergence,
                FloatRange::Linear {
                    min: CONVERGENCE_RANGE.0,
                    max: CONVERGENCE_RANGE.1,
                },
            ),
            double_talk_db: FloatParam::new(
                "Double-Talk Threshold",
                defaults.double_talk_db,
                FloatRange::Linear {
                    min: DOUBLE_TALK_DB_RANGE.0,
                    max: DOUBLE_TALK_DB_RANGE.1,
                },
            )
            .with_unit(" dB"),
            output_gain_db: FloatParam::new(
                "Output Gain",
                defaults.output_gain_db,
                FloatRange::Linear {
                    min: OUTPUT_GAIN_DB_RANGE.0,
                    max: OUTPUT_GAIN_DB_RANGE.1,
                },
            )
            .with_unit(" dB"),
            bypass: BoolParam::new("Bypass", false)
                .with_value_to_string(formatters::v2s_bool_bypass()),
            #[cfg(feature = "gui")]
            editor_state: EguiState::from_size(560, 600),
        }
    }
}

impl AdaptivePluginParams {
    /// Snapshot into the framework-agnostic model.
    fn snapshot(&self) -> AdaptiveParams {
        AdaptiveParams {
            mode: self.mode.value().to_mode(),
            sensitivity: self.sensitivity.value(),
            max_notches: self.max_notches.value() as usize,
            notch_q: self.notch_q.value(),
            auto: self.auto.value(),
            freeze: self.freeze.value(),
            reset: self.reset.value(),
            hum_fundamental_hz: self.hum_fundamental_hz.value(),
            hum_harmonics: self.hum_harmonics.value() as usize,
            hum_adaptive: self.hum_adaptive.value(),
            hum_q: self.hum_q.value(),
            filter_length: self.filter_length.value().to_length(),
            convergence: self.convergence.value(),
            double_talk_db: self.double_talk_db.value(),
            output_gain_db: self.output_gain_db.value(),
            bypass: self.bypass.value(),
        }
    }
}

/// The adaptive-filter plugin instance.
pub struct AdaptivePlugin {
    params: Arc<AdaptivePluginParams>,
    /// Shared with the background-task executor; built in `initialize`.
    controller: Arc<Mutex<Option<AdaptiveController>>>,
    /// Audio-thread renderer; built in `initialize`.
    processor: Option<AdaptiveProcessor>,
    /// Change detection so we only rebuild on an actual filter-length move.
    last_length: FilterLengthParam,

    // ── Editor telemetry (audio thread → egui editor), packed atomics. ──
    /// Monotonic feedback-notch-placement counter — drives the detection LED.
    feedback_events: Arc<AtomicU64>,
    /// Number of active feedback notches.
    active_notches: Arc<AtomicU32>,
    /// Mains fundamental currently driving the hum comb (`f32` bits).
    tracked_hum_hz: Arc<AtomicU32>,
    /// Host sample rate in Hz (`f32` bits) — converts notch ages to seconds.
    sample_rate_hz: Arc<AtomicU32>,

    /// Log-collapsed magnitude spectrum (`f32` bits) — the feedback display.
    #[cfg(feature = "gui")]
    spectrum: Arc<Vec<AtomicU32>>,
    /// Active-notch centre frequencies (`f32` bits, `0` = empty slot).
    #[cfg(feature = "gui")]
    notch_freqs: Arc<Vec<AtomicU32>>,
    /// Active-notch ages in samples (saturating to `u32`).
    #[cfg(feature = "gui")]
    notch_ages: Arc<Vec<AtomicU32>>,
    /// Audio-thread scratch the probes fill before publishing to the atomics.
    #[cfg(feature = "gui")]
    probe_spectrum: Vec<f32>,
    #[cfg(feature = "gui")]
    probe_freqs: Vec<f32>,
    #[cfg(feature = "gui")]
    probe_ages: Vec<u64>,
}

impl Default for AdaptivePlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(AdaptivePluginParams::default()),
            controller: Arc::new(Mutex::new(None)),
            processor: None,
            last_length: FilterLengthParam::L512,
            feedback_events: Arc::new(AtomicU64::new(0)),
            active_notches: Arc::new(AtomicU32::new(0)),
            tracked_hum_hz: Arc::new(AtomicU32::new(0)),
            sample_rate_hz: Arc::new(AtomicU32::new(48_000.0f32.to_bits())),
            #[cfg(feature = "gui")]
            spectrum: Arc::new(
                (0..SPECTRUM_DISPLAY_BINS)
                    .map(|_| AtomicU32::new(0))
                    .collect(),
            ),
            #[cfg(feature = "gui")]
            notch_freqs: Arc::new((0..MAX_NOTCHES).map(|_| AtomicU32::new(0)).collect()),
            #[cfg(feature = "gui")]
            notch_ages: Arc::new((0..MAX_NOTCHES).map(|_| AtomicU32::new(0)).collect()),
            #[cfg(feature = "gui")]
            probe_spectrum: vec![0.0; SPECTRUM_DISPLAY_BINS],
            #[cfg(feature = "gui")]
            probe_freqs: vec![0.0; MAX_NOTCHES],
            #[cfg(feature = "gui")]
            probe_ages: vec![0; MAX_NOTCHES],
        }
    }
}

/// Per-frame editor state owned by the egui closure.
#[cfg(feature = "gui")]
struct AdaptiveEditorState {
    /// Detection LED (pulses when a new notch is placed).
    led: MidiActivity,
    /// Last feedback-event count seen, to detect a fresh placement.
    last_events: u64,
}

#[cfg(feature = "gui")]
impl AdaptiveEditorState {
    fn new() -> Self {
        Self {
            led: MidiActivity::new(),
            last_events: 0,
        }
    }
}

/// Draw the feedback-detection LED (green pulse when a notch is placed).
#[cfg(feature = "gui")]
fn draw_led(ui: &mut egui::Ui, led: &MidiActivity) {
    use egui::Color32;
    let base = match led.status() {
        MidiStatus::Recognised => Color32::from_rgb(0x40, 0xd0, 0x60),
        MidiStatus::Unrecognised => Color32::from_rgb(0xd0, 0x50, 0x40),
        MidiStatus::Idle => Color32::from_rgb(0x30, 0x30, 0x38),
    };
    let level = led.level().clamp(0.0, 1.0);
    let dim = Color32::from_rgb(0x22, 0x24, 0x2a);
    let lerp = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * level) as u8;
    let colour = Color32::from_rgb(
        lerp(dim.r(), base.r()),
        lerp(dim.g(), base.g()),
        lerp(dim.b(), base.b()),
    );
    let (rect, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), 5.0, colour);
    ui.label("Detect");
}

/// Draw the spectrum bars with vertical markers at each active notch frequency.
#[cfg(feature = "gui")]
fn draw_spectrum(ui: &mut egui::Ui, heights: &[f32], notch_freqs: &[f32]) {
    use egui::Color32;
    const PANEL: Color32 = Color32::from_rgb(0x12, 0x14, 0x1a);
    const BAR: Color32 = Color32::from_rgb(0x3a, 0x46, 0x60);
    const NOTCH: Color32 = Color32::from_rgb(0xe0, 0x9a, 0x54);

    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), 120.0),
        egui::Sense::hover(),
    );
    ui.painter().rect_filled(rect, 4.0, PANEL);
    if !heights.is_empty() {
        let bin_width = rect.width() / heights.len() as f32;
        for (index, &height) in heights.iter().enumerate() {
            let x = rect.left() + index as f32 * bin_width;
            let top = rect.bottom() - height * rect.height();
            let bar = egui::Rect::from_min_max(
                egui::pos2(x, top),
                egui::pos2(x + (bin_width - 1.0).max(1.0), rect.bottom()),
            );
            ui.painter().rect_filled(bar, 0.0, BAR);
        }
    }
    // Active-notch markers.
    for &freq in notch_freqs.iter().filter(|&&f| f > 0.0) {
        let x = rect.left() + freq_fraction(freq) * rect.width();
        ui.painter().line_segment(
            [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
            egui::Stroke::new(1.5, NOTCH),
        );
    }
}

impl Plugin for AdaptivePlugin {
    const NAME: &'static str = "RT Adaptive";
    const VENDOR: &'static str = "Sébastien Lundi";
    const URL: &'static str = "https://codeberg.org/slundi/rt-fft-convolver";
    const EMAIL: &'static str = "slundi@stamus-networks.com";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    // Stereo → stereo and mono → mono, each with a mono sidechain (AEC far-end).
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            aux_input_ports: &[new_nonzero_u32(1)],
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(1),
            aux_input_ports: &[new_nonzero_u32(1)],
            ..AudioIOLayout::const_default()
        },
    ];

    // Freeze / Reset bind to a footswitch CC via host MIDI learn on the params.
    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = AdaptiveTask;

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    #[cfg(feature = "gui")]
    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        let params = self.params.clone();
        let feedback_events = self.feedback_events.clone();
        let active_notches = self.active_notches.clone();
        let tracked_hum_hz = self.tracked_hum_hz.clone();
        let sample_rate_hz = self.sample_rate_hz.clone();
        let spectrum = self.spectrum.clone();
        let notch_freqs = self.notch_freqs.clone();
        let notch_ages = self.notch_ages.clone();

        create_egui_editor(
            self.params.editor_state.clone(),
            AdaptiveEditorState::new(),
            |_ctx, _state| {},
            move |ctx, setter, state| {
                // Pulse the LED when the notch-placement counter advances.
                let events = feedback_events.load(Ordering::Relaxed);
                if events > state.last_events {
                    state.led.trigger(true);
                }
                state.last_events = events;
                state.led.decay(0.08);

                egui::CentralPanel::default().show(ctx, |ui| {
                    // ── Header: name + detection LED + bypass ──
                    ui.horizontal(|ui| {
                        ui.heading("RT Adaptive");
                        draw_led(ui, &state.led);
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.add(widgets::ParamSlider::for_param(&params.bypass, setter));
                        });
                    });
                    ui.separator();

                    // ── Mode selector ──
                    ui.horizontal(|ui| {
                        ui.label("Mode");
                        ui.add(widgets::ParamSlider::for_param(&params.mode, setter));
                    });

                    let control = |ui: &mut egui::Ui, label: &str, add: &dyn Fn(&mut egui::Ui)| {
                        ui.horizontal(|ui| {
                            ui.label(label);
                            add(ui);
                        });
                    };

                    match params.mode.value() {
                        ModeParam::Feedback => {
                            // Spectrum + active-notch markers.
                            let magnitude: Vec<f32> = spectrum
                                .iter()
                                .map(|slot| f32::from_bits(slot.load(Ordering::Relaxed)))
                                .collect();
                            let freqs: Vec<f32> = notch_freqs
                                .iter()
                                .map(|slot| f32::from_bits(slot.load(Ordering::Relaxed)))
                                .collect();
                            draw_spectrum(ui, &spectral_heights(&magnitude, -60.0), &freqs);

                            control(ui, "Sensitivity", &|ui| {
                                ui.add(widgets::ParamSlider::for_param(
                                    &params.sensitivity,
                                    setter,
                                ));
                            });
                            control(ui, "Max Notches", &|ui| {
                                ui.add(widgets::ParamSlider::for_param(
                                    &params.max_notches,
                                    setter,
                                ));
                            });
                            control(ui, "Notch Q", &|ui| {
                                ui.add(widgets::ParamSlider::for_param(&params.notch_q, setter));
                            });
                            ui.horizontal(|ui| {
                                ui.add(widgets::ParamSlider::for_param(&params.auto, setter));
                                ui.add(widgets::ParamSlider::for_param(&params.freeze, setter));
                                ui.add(widgets::ParamSlider::for_param(&params.reset, setter));
                            });

                            // ── Notch inventory ──
                            ui.separator();
                            let count = active_notches.load(Ordering::Relaxed);
                            ui.monospace(format!("Notches: {count}"));
                            let rate = f32::from_bits(sample_rate_hz.load(Ordering::Relaxed));
                            let ages: Vec<u64> = notch_ages
                                .iter()
                                .map(|slot| slot.load(Ordering::Relaxed) as u64)
                                .collect();
                            for (freq, age) in
                                freqs.iter().zip(ages.iter()).filter(|(f, _)| **f > 0.0)
                            {
                                ui.monospace(format!(
                                    "  {} · {} · {}",
                                    format_hz(*freq),
                                    format_q(params.notch_q.value()),
                                    format_age(*age, rate),
                                ));
                            }
                        }
                        ModeParam::Hum => {
                            control(ui, "Fundamental", &|ui| {
                                ui.add(widgets::ParamSlider::for_param(
                                    &params.hum_fundamental_hz,
                                    setter,
                                ));
                            });
                            control(ui, "Harmonics", &|ui| {
                                ui.add(widgets::ParamSlider::for_param(
                                    &params.hum_harmonics,
                                    setter,
                                ));
                            });
                            control(ui, "Q", &|ui| {
                                ui.add(widgets::ParamSlider::for_param(&params.hum_q, setter));
                            });
                            ui.add(widgets::ParamSlider::for_param(
                                &params.hum_adaptive,
                                setter,
                            ));
                            if params.hum_adaptive.value() {
                                let hz = f32::from_bits(tracked_hum_hz.load(Ordering::Relaxed));
                                ui.monospace(format!("Tracked: {}", format_hz(hz)));
                            }
                        }
                        ModeParam::Echo => {
                            control(ui, "Filter Length", &|ui| {
                                ui.add(widgets::ParamSlider::for_param(
                                    &params.filter_length,
                                    setter,
                                ));
                            });
                            control(ui, "Convergence", &|ui| {
                                ui.add(widgets::ParamSlider::for_param(
                                    &params.convergence,
                                    setter,
                                ));
                            });
                            control(ui, "Double-Talk", &|ui| {
                                ui.add(widgets::ParamSlider::for_param(
                                    &params.double_talk_db,
                                    setter,
                                ));
                            });
                        }
                    }

                    // ── Global ──
                    ui.separator();
                    control(ui, "Output", &|ui| {
                        ui.add(widgets::ParamSlider::for_param(
                            &params.output_gain_db,
                            setter,
                        ));
                    });

                    // Keep animating even without host automation.
                    ctx.request_repaint();
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
                AdaptiveTask::ApplyStructural(params) => {
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

        let snapshot = self.params.snapshot();
        let (controller, processor) =
            AdaptiveController::new(max_block, sample_rate, channels, &snapshot);

        // Every mode is zero-latency in the direct path.
        context.set_latency_samples(processor.latency_samples() as u32);
        self.sample_rate_hz
            .store(sample_rate.to_bits(), Ordering::Relaxed);

        *self.controller.lock().unwrap() = Some(controller);
        self.processor = Some(processor);
        self.last_length = self.params.filter_length.value();
        true
    }

    fn reset(&mut self) {
        if let Some(processor) = self.processor.as_mut() {
            processor.reset();
        }
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // Structural (filter-length) change → background rebuild.
        let length = self.params.filter_length.value();
        if length != self.last_length {
            self.last_length = length;
            _context.execute_background(AdaptiveTask::ApplyStructural(self.params.snapshot()));
        }

        // Render — flush denormals for the whole callback.
        let _denormal_guard = rt_fft_convolver::DenormalGuard::new();
        let snapshot = self.params.snapshot();

        if let Some(processor) = self.processor.as_mut() {
            // The (mono) far-end reference, if a sidechain is patched.
            let sidechain: Option<&[f32]> = aux
                .inputs
                .first_mut()
                .and_then(|reference| reference.as_slice().first().map(|channel| &**channel));

            processor.process(buffer.as_slice(), sidechain, &snapshot);

            // Publish read-only telemetry (a `.store` is a use, so these stay
            // live in the no-gui build too).
            self.feedback_events
                .store(processor.feedback_events(), Ordering::Relaxed);
            self.active_notches
                .store(processor.active_notches() as u32, Ordering::Relaxed);
            self.tracked_hum_hz
                .store(processor.tracked_hum_hz().to_bits(), Ordering::Relaxed);

            // Refresh the spectrum + notch snapshots only while the editor is
            // open (the display probes are otherwise never paid for).
            #[cfg(feature = "gui")]
            if self.params.editor_state.is_open() {
                processor.probe_spectrum(&mut self.probe_spectrum);
                processor.copy_notches(&mut self.probe_freqs, &mut self.probe_ages);
                for (slot, &value) in self.spectrum.iter().zip(&self.probe_spectrum) {
                    slot.store(value.to_bits(), Ordering::Relaxed);
                }
                for (slot, &freq) in self.notch_freqs.iter().zip(&self.probe_freqs) {
                    slot.store(freq.to_bits(), Ordering::Relaxed);
                }
                for (slot, &age) in self.notch_ages.iter().zip(&self.probe_ages) {
                    slot.store(age.min(u32::MAX as u64) as u32, Ordering::Relaxed);
                }
            }
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for AdaptivePlugin {
    const CLAP_ID: &'static str = "org.codeberg.slundi.rt-adaptive";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("Feedback suppressor / hum remover / echo canceller");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Mono,
        ClapFeature::Filter,
        ClapFeature::Restoration,
    ];
}

impl Vst3Plugin for AdaptivePlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"RtAdaptSlundi001";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Restoration];
}

nih_export_clap!(AdaptivePlugin);
nih_export_vst3!(AdaptivePlugin);
