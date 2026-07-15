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
use std::sync::{Arc, Mutex};

use nih_plug::prelude::*;
use rt_spectral_core::{
    ClickParams, CompressParams, DenoiseParams, GateParams, HumParams, SpectralParams,
    TransientParams,
};

use crate::engine::{SpectralController, SpectralProcessor, spectral_engine};
use crate::params::{
    ATTACK_MS_RANGE, CLICK_MAX_LEN_RANGE, CLICK_SENSITIVITY_RANGE, COMP_THRESHOLD_DB_RANGE,
    GATE_RANGE_DB_RANGE, GATE_THRESHOLD_DB_RANGE, HARMONICS_RANGE, HUM_FUNDAMENTAL_HZ_RANGE,
    HUM_Q_RANGE, KNEE_DB_RANGE, LEARN_MS_RANGE, MAKEUP_DB_RANGE, OUTPUT_GAIN_DB_RANGE, RATIO_RANGE,
    REDUCTION_DB_RANGE, RELEASE_MS_RANGE, SENSITIVITY_DB_RANGE, clamp_params, learn_samples,
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
    /// Shared with the background-task executor; built in `initialize`.
    controller: Arc<Mutex<Option<SpectralController>>>,
    /// Audio-thread renderer; built in `initialize`.
    processor: Option<SpectralProcessor>,

    /// Host sample rate, for converting `Learn Time` ms → samples.
    sample_rate: u32,
    /// Last `Learn` param value, for rising-edge detection.
    last_learn: bool,
}

impl Default for SpectralPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(SpectralPluginParams::default()),
            controller: Arc::new(Mutex::new(None)),
            processor: None,
            sample_rate: 48_000,
            last_learn: false,
        }
    }
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

        let (controller, processor) = spectral_engine(sample_rate, FFT_SIZE, max_block, channels);

        // Latency is constant for the fixed FFT size; report it once.
        context.set_latency_samples(processor.latency_samples() as u32);

        self.sample_rate = sample_rate;
        *self.controller.lock().unwrap() = Some(controller);
        self.processor = Some(processor);
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
