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
use std::sync::{Arc, Mutex};

use nih_plug::prelude::*;

use crate::engine::{AdaptiveController, AdaptiveProcessor};
use crate::params::{
    AdaptiveParams, CONVERGENCE_RANGE, DOUBLE_TALK_DB_RANGE, FilterLength, HUM_FUNDAMENTAL_RANGE,
    HUM_Q_RANGE, Mode, NOTCH_Q_RANGE, OUTPUT_GAIN_DB_RANGE, SENSITIVITY_RANGE,
};

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
}

impl Default for AdaptivePlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(AdaptivePluginParams::default()),
            controller: Arc::new(Mutex::new(None)),
            processor: None,
            last_length: FilterLengthParam::L512,
        }
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
