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
use parking_lot::Mutex as PlMutex;

use crate::engine::{LogFormat, MeterLogger, MeterProcessor};
use crate::params::{CUSTOM_DBTP_RANGE, CUSTOM_LUFS_RANGE, LoudnessParams, Target};
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

        let snapshot = self.params.snapshot();
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
