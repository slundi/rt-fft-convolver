//! `nih-plug` wrapper for `rt-loudness-meter` (compiled only with the `nih`
//! feature).
//!
//! A thin framework layer over the tested, framework-agnostic
//! [`engine`](crate::engine): it declares the automatable parameters, runs the
//! [`MeterProcessor`](crate::engine::MeterProcessor) on the audio thread, and
//! reports zero latency (metering never delays the signal).
//!
//! # Analysis-only
//!
//! The plugin does not modify the audio: `process` reads each block into the
//! meter and returns it untouched, so `Bypass` is a formal no-op present only
//! because LV2 requires it.
//!
//! # I/O (Phase 0 decision)
//!
//! Stereo in → stereo out and mono in → mono out (pass-through).  Surround
//! (5.1 / 7.1) is deferred to v2.
//!
//! # MIDI / footswitch
//!
//! `Reset` and `Start/Stop` are automatable `BoolParam`s; a host binds a
//! footswitch / transport CC to them via MIDI learn (the idiomatic `nih-plug`
//! path), so no plugin-side MIDI parsing is needed and `MIDI_INPUT` is `None`.
//!
//! # Logging threading
//!
//! Toggling `Log` dispatches a [`LogTask`] to the background executor, which
//! starts / stops the off-RT [`MeterLogger`](crate::engine::MeterLogger).  The
//! audio thread only ever publishes to the lock-free broadcast the logger
//! polls — it never opens files or spawns threads.  (The log-file *path* picker
//! arrives with the Phase 3 editor; until a path is set, `Log` is inert.)

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use nih_plug::prelude::*;
#[cfg(feature = "gui")]
use nih_plug_egui::{EguiState, create_egui_editor, egui, widgets};
use parking_lot::Mutex as PlMutex;

use crate::engine::{LogFormat, MeterLogger, MeterProcessor};
use crate::params::{CUSTOM_DBTP_RANGE, CUSTOM_LUFS_RANGE, LoudnessParams, Target};
use crate::ui::effective_running;
#[cfg(feature = "gui")]
use crate::ui::{
    Compliance, GRAPH_FLOOR_LUFS, GRAPH_TOP_LUFS, HISTOGRAM_BINS, LoudnessHistory, compliance,
    format_dbfs, format_dbtp, format_lu, format_lufs, histogram_bins, lufs_to_fraction,
};
use rt_loudness_core::MeterBroadcast;

/// How often the off-RT logger samples the broadcast.
const LOG_PERIOD: Duration = Duration::from_millis(100);

/// `nih-plug` target enum, mirroring [`Target`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
enum TargetParam {
    #[name = "EBU R128 (−23)"]
    EbuR128,
    #[name = "Streaming (−14)"]
    Streaming14,
    #[name = "Apple / Podcast (−16)"]
    AppleMusic16,
    #[name = "Custom"]
    Custom,
}

impl TargetParam {
    fn to_target(self) -> Target {
        match self {
            TargetParam::EbuR128 => Target::EbuR128,
            TargetParam::Streaming14 => Target::Streaming14,
            TargetParam::AppleMusic16 => Target::AppleMusic16,
            TargetParam::Custom => Target::Custom,
        }
    }
}

/// Off-thread work dispatched to the background executor (control thread only).
pub enum LogTask {
    /// (Re)start the log worker for the current path + format.
    Start,
    /// Stop and join the log worker.
    Stop,
}

/// Host-visible parameters.
#[derive(Params)]
struct LoudnessPluginParams {
    #[id = "target"]
    target: EnumParam<TargetParam>,
    #[id = "clufs"]
    custom_target_lufs: FloatParam,
    #[id = "cdbtp"]
    custom_target_dbtp: FloatParam,
    #[id = "run"]
    running: BoolParam,
    #[id = "reset"]
    reset: BoolParam,
    #[id = "log"]
    log_enabled: BoolParam,
    #[id = "byp"]
    bypass: BoolParam,

    /// Log-file path, restored with the plugin state (set by the Phase 3
    /// editor's file picker).
    #[persist = "log-path"]
    log_path: PlMutex<Option<String>>,

    #[cfg(feature = "gui")]
    #[persist = "editor-state"]
    editor_state: Arc<EguiState>,
}

impl Default for LoudnessPluginParams {
    fn default() -> Self {
        let defaults = LoudnessParams::default();
        Self {
            target: EnumParam::new("Target", TargetParam::EbuR128),
            custom_target_lufs: FloatParam::new(
                "Custom Target",
                defaults.custom_target_lufs,
                FloatRange::Linear {
                    min: CUSTOM_LUFS_RANGE.0,
                    max: CUSTOM_LUFS_RANGE.1,
                },
            )
            .with_unit(" LUFS"),
            custom_target_dbtp: FloatParam::new(
                "Custom Ceiling",
                defaults.custom_target_dbtp,
                FloatRange::Linear {
                    min: CUSTOM_DBTP_RANGE.0,
                    max: CUSTOM_DBTP_RANGE.1,
                },
            )
            .with_unit(" dBTP"),
            running: BoolParam::new("Start/Stop", defaults.running),
            reset: BoolParam::new("Reset", defaults.reset),
            log_enabled: BoolParam::new("Log", defaults.log_enabled),
            bypass: BoolParam::new("Bypass", false)
                .with_value_to_string(formatters::v2s_bool_bypass()),
            log_path: PlMutex::new(None),
            #[cfg(feature = "gui")]
            editor_state: EguiState::from_size(520, 640),
        }
    }
}

impl LoudnessPluginParams {
    /// Snapshot into the framework-agnostic model (clamped).
    fn snapshot(&self) -> LoudnessParams {
        LoudnessParams {
            target: self.target.value().to_target(),
            custom_target_lufs: self.custom_target_lufs.value(),
            custom_target_dbtp: self.custom_target_dbtp.value(),
            running: self.running.value(),
            reset: self.reset.value(),
            log_enabled: self.log_enabled.value(),
            bypass: self.bypass.value(),
        }
        .clamped()
    }
}

/// Per-editor state owned by the egui closure.
#[cfg(feature = "gui")]
struct LoudnessEditorState {
    /// Scrolling short-term-LUFS history for the graph / histogram.
    history: LoudnessHistory,
    /// Last time a short-term sample was pushed (throttled to 100 ms).
    last_push: Option<std::time::Instant>,
}

#[cfg(feature = "gui")]
impl LoudnessEditorState {
    fn new() -> Self {
        Self {
            history: LoudnessHistory::new(crate::ui::GRAPH_LEN),
            last_push: None,
        }
    }
}

/// The LED colour for a compliance state.
#[cfg(feature = "gui")]
fn led_colour(state: Compliance) -> egui::Color32 {
    match state {
        Compliance::Compliant => egui::Color32::from_rgb(0x40, 0xd0, 0x60),
        Compliance::Close => egui::Color32::from_rgb(0xe0, 0xb0, 0x40),
        Compliance::Over => egui::Color32::from_rgb(0xd0, 0x50, 0x40),
        Compliance::NoData => egui::Color32::from_rgb(0x30, 0x30, 0x38),
    }
}

/// Draw the compliance LED and its label.
#[cfg(feature = "gui")]
fn draw_led(ui: &mut egui::Ui, state: Compliance) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::hover());
    ui.painter()
        .circle_filled(rect.center(), 6.0, led_colour(state));
    let label = match state {
        Compliance::Compliant => "On target",
        Compliance::Close => "Close",
        Compliance::Over => "Off target",
        Compliance::NoData => "—",
    };
    ui.label(label);
}

/// Draw one labelled numeric readout.
#[cfg(feature = "gui")]
fn draw_readout(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.vertical(|ui| {
        ui.label(egui::RichText::new(label).small().weak());
        ui.label(egui::RichText::new(value).size(20.0).monospace().strong());
    });
}

/// Draw the scrolling short-term-loudness graph with the target line.
#[cfg(feature = "gui")]
fn draw_graph(ui: &mut egui::Ui, history: &LoudnessHistory, target_lufs: f32) {
    const PANEL: egui::Color32 = egui::Color32::from_rgb(0x12, 0x14, 0x1a);
    const LINE: egui::Color32 = egui::Color32::from_rgb(0x5a, 0xc0, 0xf0);
    const TARGET: egui::Color32 = egui::Color32::from_rgb(0xe0, 0x9a, 0x54);

    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), 120.0),
        egui::Sense::hover(),
    );
    let painter = ui.painter();
    painter.rect_filled(rect, 4.0, PANEL);

    // Target reference line.
    let target_y = rect.bottom()
        - lufs_to_fraction(target_lufs, GRAPH_FLOOR_LUFS, GRAPH_TOP_LUFS) * rect.height();
    painter.line_segment(
        [
            egui::pos2(rect.left(), target_y),
            egui::pos2(rect.right(), target_y),
        ],
        egui::Stroke::new(1.0, TARGET),
    );

    // Short-term trace, right-aligned so the newest sample sits at the edge.
    let capacity = history.capacity().max(2);
    let len = history.len();
    if len >= 2 {
        let denom = (capacity - 1) as f32;
        let offset = capacity - len;
        let points: Vec<egui::Pos2> = history
            .iter()
            .enumerate()
            .map(|(index, lufs)| {
                let x = rect.left() + ((offset + index) as f32 / denom) * rect.width();
                let y = rect.bottom()
                    - lufs_to_fraction(lufs, GRAPH_FLOOR_LUFS, GRAPH_TOP_LUFS) * rect.height();
                egui::pos2(x, y)
            })
            .collect();
        painter.add(egui::Shape::line(points, egui::Stroke::new(1.5, LINE)));
    }
}

/// Draw the short-term-loudness distribution (LRA histogram panel).
#[cfg(feature = "gui")]
fn draw_histogram(ui: &mut egui::Ui, bins: &[u32]) {
    const PANEL: egui::Color32 = egui::Color32::from_rgb(0x12, 0x14, 0x1a);
    const BAR: egui::Color32 = egui::Color32::from_rgb(0x3a, 0x60, 0x50);

    let (rect, _) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), 80.0), egui::Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, 4.0, PANEL);
    if bins.is_empty() {
        return;
    }
    let peak = bins.iter().copied().max().unwrap_or(0).max(1) as f32;
    let bar_width = rect.width() / bins.len() as f32;
    for (index, &count) in bins.iter().enumerate() {
        let height = (count as f32 / peak) * rect.height();
        let left = rect.left() + index as f32 * bar_width;
        let bar = egui::Rect::from_min_max(
            egui::pos2(left, rect.bottom() - height),
            egui::pos2(left + (bar_width - 1.0).max(1.0), rect.bottom()),
        );
        painter.rect_filled(bar, 0.0, BAR);
    }
}

/// The loudness-meter plugin instance.
pub struct LoudnessMeterPlugin {
    params: Arc<LoudnessPluginParams>,
    /// Audio-thread renderer; built in `initialize`.
    processor: Option<MeterProcessor>,
    /// Shared with the editor / logger; the meter publishes here every block.
    broadcast: Arc<MeterBroadcast>,
    /// The running log worker, started / stopped off-thread.
    logger: Arc<PlMutex<Option<MeterLogger>>>,
    /// Last `Log` value, for edge detection.
    last_log_enabled: bool,
}

impl Default for LoudnessMeterPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(LoudnessPluginParams::default()),
            processor: None,
            broadcast: Arc::new(MeterBroadcast::new()),
            logger: Arc::new(PlMutex::new(None)),
            last_log_enabled: false,
        }
    }
}

impl Plugin for LoudnessMeterPlugin {
    const NAME: &'static str = "RT Loudness Meter";
    const VENDOR: &'static str = "Sébastien Lundi";
    const URL: &'static str = "https://codeberg.org/slundi/rt-fft-convolver";
    const EMAIL: &'static str = "slundi@stamus-networks.com";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    // Stereo → stereo and mono → mono, pass-through (surround is v2).
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

    // Reset / Start-Stop bind to a footswitch/transport CC via host MIDI learn.
    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = LogTask;

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    #[cfg(feature = "gui")]
    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        let params = self.params.clone();
        let broadcast = self.broadcast.clone();

        create_egui_editor(
            self.params.editor_state.clone(),
            LoudnessEditorState::new(),
            |_ctx, _state| {},
            move |ctx, setter, state| {
                let values = broadcast.load();

                // Push short-term into the scrolling history at ~10 Hz (its own
                // cadence), independent of the editor's repaint rate.
                let now = std::time::Instant::now();
                let due = state
                    .last_push
                    .map(|last| now.duration_since(last) >= Duration::from_millis(100))
                    .unwrap_or(true);
                if due {
                    state.history.push(values.short_term_lufs);
                    state.last_push = Some(now);
                }

                let thresholds = params.snapshot().thresholds();
                let led = compliance(
                    values.integrated_lufs,
                    values.true_peak_dbtp,
                    thresholds.target_lufs,
                    thresholds.ceiling_dbtp,
                );

                egui::CentralPanel::default().show(ctx, |ui| {
                    // ── Header: title + compliance LED + bypass ──
                    ui.horizontal(|ui| {
                        ui.heading("RT Loudness Meter");
                        ui.separator();
                        draw_led(ui, led);
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.add(widgets::ParamSlider::for_param(&params.bypass, setter));
                        });
                    });
                    ui.separator();

                    // ── Large numeric readouts ──
                    ui.columns(3, |cols| {
                        draw_readout(
                            &mut cols[0],
                            "Momentary",
                            &format_lufs(values.momentary_lufs),
                        );
                        draw_readout(
                            &mut cols[1],
                            "Short-term",
                            &format_lufs(values.short_term_lufs),
                        );
                        draw_readout(
                            &mut cols[2],
                            "Integrated",
                            &format_lufs(values.integrated_lufs),
                        );
                    });
                    ui.add_space(4.0);
                    ui.columns(4, |cols| {
                        draw_readout(&mut cols[0], "LRA", &format_lu(values.lra_lu));
                        draw_readout(
                            &mut cols[1],
                            "True Peak",
                            &format_dbtp(values.true_peak_dbtp),
                        );
                        draw_readout(
                            &mut cols[2],
                            "Sample Peak",
                            &format_dbfs(values.sample_peak_dbfs),
                        );
                        draw_readout(&mut cols[3], "RMS", &format_dbfs(values.rms_db));
                    });
                    ui.separator();

                    // ── Short-term graph + distribution ──
                    ui.label(egui::RichText::new("Short-term (last 60 s)").small().weak());
                    draw_graph(ui, &state.history, thresholds.target_lufs);
                    ui.add_space(4.0);
                    ui.label(egui::RichText::new("Distribution").small().weak());
                    let bins = histogram_bins(
                        state.history.iter(),
                        GRAPH_FLOOR_LUFS,
                        GRAPH_TOP_LUFS,
                        HISTOGRAM_BINS,
                    );
                    draw_histogram(ui, &bins);
                    ui.separator();

                    // ── Controls ──
                    ui.horizontal(|ui| {
                        ui.label("Target");
                        ui.add(widgets::ParamSlider::for_param(&params.target, setter));
                    });
                    if params.target.value() == TargetParam::Custom {
                        ui.horizontal(|ui| {
                            ui.label("LUFS");
                            ui.add(widgets::ParamSlider::for_param(
                                &params.custom_target_lufs,
                                setter,
                            ));
                            ui.label("Ceiling");
                            ui.add(widgets::ParamSlider::for_param(
                                &params.custom_target_dbtp,
                                setter,
                            ));
                        });
                    }
                    ui.horizontal(|ui| {
                        ui.add(widgets::ParamSlider::for_param(&params.running, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.reset, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.log_enabled, setter));
                    });

                    // Keep animating even without host automation.
                    ctx.request_repaint();
                });
            },
        )
    }

    fn task_executor(&mut self) -> TaskExecutor<Self> {
        let params = self.params.clone();
        let logger = self.logger.clone();
        let broadcast = self.broadcast.clone();
        Box::new(move |task| match task {
            LogTask::Start => {
                let path = params.log_path.lock().clone();
                let Some(path) = path.filter(|value| !value.is_empty()) else {
                    return;
                };
                if let Ok(worker) = MeterLogger::start(
                    Path::new(&path),
                    LogFormat::Csv,
                    broadcast.clone(),
                    LOG_PERIOD,
                ) {
                    // Replacing the slot drops (and joins) any previous worker.
                    *logger.lock() = Some(worker);
                }
            }
            LogTask::Stop => {
                *logger.lock() = None;
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
        let max_block = (buffer_config.max_buffer_size as usize).max(1);
        let channels = audio_io_layout
            .main_output_channels
            .map(|count| count.get() as usize)
            .unwrap_or(1);

        let processor =
            MeterProcessor::new(channels, sample_rate, max_block, self.broadcast.clone());
        context.set_latency_samples(processor.latency_samples() as u32);
        self.processor = Some(processor);
        self.last_log_enabled = self.params.log_enabled.value();
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
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // `Log` edge → start / stop the off-RT worker on the background thread.
        let log_enabled = self.params.log_enabled.value();
        if log_enabled != self.last_log_enabled {
            self.last_log_enabled = log_enabled;
            context.execute_background(if log_enabled {
                LogTask::Start
            } else {
                LogTask::Stop
            });
        }

        let mut snapshot = self.params.snapshot();

        // Playhead integration: hold integrated / LRA while the host transport
        // is stopped.  A host that reports no position has no meaningful
        // transport, so the manual `Start/Stop` decides on its own.
        let transport = context.transport();
        let playing = transport.pos_samples().map(|_| transport.playing);
        snapshot.running = effective_running(snapshot.running, playing);

        if let Some(processor) = self.processor.as_mut() {
            // Analysis-only: read the block into the meter, leave it untouched.
            processor.process_buffer(buffer.as_slice(), &snapshot);
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for LoudnessMeterPlugin {
    const CLAP_ID: &'static str = "org.codeberg.slundi.rt-loudness-meter";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("EBU R128 / BS.1770-4 loudness and true-peak meter");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Mono,
        ClapFeature::Analyzer,
        ClapFeature::Utility,
    ];
}

impl Vst3Plugin for LoudnessMeterPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"RtLoudMtrSlundi1";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Analyzer];
}

nih_export_clap!(LoudnessMeterPlugin);
nih_export_vst3!(LoudnessMeterPlugin);
