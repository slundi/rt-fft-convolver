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
use rt_drive_core::shipped_pedals;
use rt_ir_plugin_core::program_to_slot;

use crate::engine::{DriveController, DriveProcessor};
use crate::params::{BIAS_RANGE, DriveParams, LEVEL_DB_RANGE, OversampleChoice, TONE_DB_RANGE};

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
    /// `f32.to_bits()` packed — read by the (Phase-3) CPU meter.
    cpu_meter: Arc<AtomicU32>,
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
        }
    }
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
            let start = std::time::Instant::now();
            processor.process(buffer.as_slice(), &snapshot);
            let elapsed = start.elapsed().as_secs_f32();

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
