//! `nih-plug` wrapper for `rt-limiter` (compiled only with the `nih` feature).
//!
//! A thin framework layer over the tested, framework-agnostic
//! [`engine`](crate::engine): it declares the automatable parameters, routes
//! the structural (oversampling) change into the off-thread
//! [`LimiterController`], runs the [`LimiterProcessor`] on the audio thread,
//! and reports latency + metering to the host.
//!
//! # Threading
//!
//! - The [`LimiterController`] lives behind a `Mutex` shared with the
//!   `nih-plug` background-task executor.  Rebuilding the true-peak detectors
//!   (which reallocates the resampler) happens there, never on the audio
//!   thread.
//! - `process()` polls the lock-free handoff, applies the continuous
//!   parameters and renders — allocation-free.
//!
//! # Metering (Phase 2 loudness-display decision)
//!
//! Exposed as read-only atomics for a future editor (Phase 3): input peak,
//! output peak (both dBFS), peak gain reduction (dB) and a CPU-load proxy.
//! Integrated / short-term LUFS is **deferred** to `rt-loudness-meter`
//! (Roadmap-10) to avoid duplicating a BS.1770 integrator here.
//!
//! # I/O (Phase 2 decision)
//!
//! Stereo in → stereo out and mono → mono.  The optional 4-channel /
//! fully-independent mode is deferred (consistent with the Phase 0 channel-link
//! decision).
//!
//! # MIDI
//!
//! A mastering limiter needs none; every continuous control is host-automatable
//! (and hence MIDI-learnable), so `MIDI_INPUT` is `None`.

use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use nih_plug::prelude::*;

use crate::engine::{LimiterController, LimiterProcessor};
use crate::params::{
    CEILING_DB_RANGE, DitherChoice, LOOKAHEAD_MS_RANGE, LimiterParams, LinkChoice,
    OversampleChoice, RELEASE_MS_RANGE, Style, THRESHOLD_DB_RANGE,
};

/// −∞-ish floor used when a block is silent.
const SILENCE_DBFS: f32 = -120.0;

/// Off-thread work dispatched to the background executor (control thread only).
pub enum LimiterTask {
    /// Re-apply the parameter snapshot; rebuilds only if oversampling changed.
    Reconfigure(LimiterParams),
}

/// `nih-plug` oversampling enum (mirrors [`OversampleChoice`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
enum OversampleParam {
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
            OversampleParam::X1 => OversampleChoice::X1,
            OversampleParam::X2 => OversampleChoice::X2,
            OversampleParam::X4 => OversampleChoice::X4,
            OversampleParam::X8 => OversampleChoice::X8,
            OversampleParam::X16 => OversampleChoice::X16,
        }
    }
}

/// `nih-plug` style enum (mirrors [`Style`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
enum StyleParam {
    Transparent,
    Punch,
    Smooth,
    Aggressive,
}

impl StyleParam {
    fn to_style(self) -> Style {
        match self {
            StyleParam::Transparent => Style::Transparent,
            StyleParam::Punch => Style::Punch,
            StyleParam::Smooth => Style::Smooth,
            StyleParam::Aggressive => Style::Aggressive,
        }
    }
}

/// `nih-plug` channel-link enum (mirrors [`LinkChoice`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
enum LinkParam {
    #[name = "Stereo Linked"]
    Linked,
    #[name = "Mid/Side"]
    MidSide,
}

impl LinkParam {
    fn to_choice(self) -> LinkChoice {
        match self {
            LinkParam::Linked => LinkChoice::Linked,
            LinkParam::MidSide => LinkChoice::MidSide,
        }
    }
}

/// `nih-plug` dither enum (mirrors [`DitherChoice`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
enum DitherParam {
    Off,
    #[name = "16-bit"]
    Bits16,
    #[name = "20-bit"]
    Bits20,
    #[name = "24-bit"]
    Bits24,
}

impl DitherParam {
    fn to_choice(self) -> DitherChoice {
        match self {
            DitherParam::Off => DitherChoice::Off,
            DitherParam::Bits16 => DitherChoice::Bits16,
            DitherParam::Bits20 => DitherChoice::Bits20,
            DitherParam::Bits24 => DitherChoice::Bits24,
        }
    }
}

/// Host-visible parameters.
#[derive(Params)]
struct LimiterPluginParams {
    #[id = "thresh"]
    threshold_db: FloatParam,
    #[id = "ceil"]
    ceiling_db: FloatParam,
    #[id = "release"]
    release_ms: FloatParam,
    #[id = "style"]
    style: EnumParam<StyleParam>,
    #[id = "autorel"]
    auto_release: BoolParam,
    #[id = "look"]
    lookahead_ms: FloatParam,
    #[id = "os"]
    oversampling: EnumParam<OversampleParam>,
    #[id = "link"]
    link: EnumParam<LinkParam>,
    #[id = "dither"]
    dither: EnumParam<DitherParam>,
    #[id = "byp"]
    bypass: BoolParam,
}

impl Default for LimiterPluginParams {
    fn default() -> Self {
        Self {
            threshold_db: FloatParam::new(
                "Threshold",
                0.0,
                FloatRange::Linear {
                    min: THRESHOLD_DB_RANGE.0,
                    max: THRESHOLD_DB_RANGE.1,
                },
            )
            .with_unit(" dB"),
            ceiling_db: FloatParam::new(
                "Ceiling",
                -1.0,
                FloatRange::Linear {
                    min: CEILING_DB_RANGE.0,
                    max: CEILING_DB_RANGE.1,
                },
            )
            .with_unit(" dB"),
            release_ms: FloatParam::new(
                "Release",
                100.0,
                // Skewed so the musically useful short times get more travel.
                FloatRange::Skewed {
                    min: RELEASE_MS_RANGE.0,
                    max: RELEASE_MS_RANGE.1,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_unit(" ms"),
            style: EnumParam::new("Style", StyleParam::Transparent),
            auto_release: BoolParam::new("Auto Release", false),
            lookahead_ms: FloatParam::new(
                "Look-ahead",
                5.0,
                FloatRange::Linear {
                    min: LOOKAHEAD_MS_RANGE.0,
                    max: LOOKAHEAD_MS_RANGE.1,
                },
            )
            .with_unit(" ms"),
            oversampling: EnumParam::new("Oversampling", OversampleParam::X4),
            link: EnumParam::new("Channel Link", LinkParam::Linked),
            dither: EnumParam::new("Dither", DitherParam::Off),
            bypass: BoolParam::new("Bypass", false)
                .with_value_to_string(formatters::v2s_bool_bypass()),
        }
    }
}

impl LimiterPluginParams {
    /// Snapshot into the framework-agnostic model.
    fn snapshot(&self) -> LimiterParams {
        LimiterParams {
            threshold_db: self.threshold_db.value(),
            ceiling_db: self.ceiling_db.value(),
            release_ms: self.release_ms.value(),
            style: self.style.value().to_style(),
            auto_release: self.auto_release.value(),
            lookahead_ms: self.lookahead_ms.value(),
            oversampling: self.oversampling.value().to_choice(),
            link: self.link.value().to_choice(),
            dither: self.dither.value().to_choice(),
            bypass: self.bypass.value(),
        }
    }
}

/// The limiter plugin instance.
pub struct LimiterPlugin {
    params: Arc<LimiterPluginParams>,
    /// Shared with the background-task executor; built in `initialize`.
    controller: Arc<Mutex<Option<LimiterController>>>,
    /// Audio-thread renderer; built in `initialize`.
    processor: Option<LimiterProcessor>,
    /// Change detection so we only rebuild on an actual oversampling move.
    last_oversampling: OversampleParam,
    /// Last latency reported to the host, to avoid redundant updates.
    reported_latency: u32,

    /// Peak gain reduction in dB, `f32.to_bits()` packed (editor GR meter).
    gr_meter_db: Arc<AtomicU32>,
    /// Per-block input peak in dBFS, `f32.to_bits()` packed.
    input_meter_dbfs: Arc<AtomicU32>,
    /// Per-block output peak in dBFS, `f32.to_bits()` packed.
    output_meter_dbfs: Arc<AtomicU32>,
    /// Last-block CPU load as a fraction of the block budget, `f32.to_bits()`.
    cpu_meter: Arc<AtomicU32>,
}

impl Default for LimiterPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(LimiterPluginParams::default()),
            controller: Arc::new(Mutex::new(None)),
            processor: None,
            last_oversampling: OversampleParam::X4,
            reported_latency: u32::MAX, // force a report on the first block
            gr_meter_db: Arc::new(AtomicU32::new(0.0f32.to_bits())),
            input_meter_dbfs: Arc::new(AtomicU32::new(SILENCE_DBFS.to_bits())),
            output_meter_dbfs: Arc::new(AtomicU32::new(SILENCE_DBFS.to_bits())),
            cpu_meter: Arc::new(AtomicU32::new(0)),
        }
    }
}

/// Peak level of a channel in dBFS (sample domain), or [`SILENCE_DBFS`].
fn peak_dbfs(channel: &[f32]) -> f32 {
    let peak = channel
        .iter()
        .fold(0.0f32, |maximum, &s| maximum.max(s.abs()));
    if peak > 0.0 {
        rt_fft_convolver::linear_to_db(peak)
    } else {
        SILENCE_DBFS
    }
}

impl Plugin for LimiterPlugin {
    const NAME: &'static str = "RT Limiter";
    const VENDOR: &'static str = "Sébastien Lundi";
    const URL: &'static str = "https://codeberg.org/slundi/rt-fft-convolver";
    const EMAIL: &'static str = "slundi@stamus-networks.com";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    // Stereo → stereo and mono → mono (Phase 2 I/O decision).
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

    // A mastering limiter needs no MIDI; controls are host-automatable.
    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = LimiterTask;

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
                LimiterTask::Reconfigure(params) => {
                    controller.update(params);
                }
            }
        })
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        let sample_rate = buffer_config.sample_rate;
        let max_block = (buffer_config.max_buffer_size as usize).max(1);

        let snapshot = self.params.snapshot();
        let (controller, processor) = LimiterController::new(max_block, sample_rate, snapshot);

        *self.controller.lock().unwrap() = Some(controller);
        self.processor = Some(processor);
        self.last_oversampling = self.params.oversampling.value();
        // Latency becomes known after the first render (look-ahead + detector
        // group delay); report it from `process`.
        self.reported_latency = u32::MAX;
        let _ = context;
        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // ── 1. Structural (oversampling) change → background rebuild.
        let oversampling = self.params.oversampling.value();
        if oversampling != self.last_oversampling {
            self.last_oversampling = oversampling;
            context.execute_background(LimiterTask::Reconfigure(self.params.snapshot()));
        }

        // ── 2. Render — flush denormals for the whole callback.
        let _denormal_guard = rt_fft_convolver::DenormalGuard::new();
        let snapshot = self.params.snapshot();
        let num_samples = buffer.samples();

        if let Some(processor) = self.processor.as_mut() {
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

            let output_dbfs = buffer
                .as_slice()
                .iter()
                .map(|channel| peak_dbfs(channel))
                .fold(SILENCE_DBFS, f32::max);
            self.output_meter_dbfs
                .store(output_dbfs.to_bits(), Ordering::Relaxed);
            self.gr_meter_db
                .store(processor.gain_reduction_db().to_bits(), Ordering::Relaxed);

            let sample_rate = context.transport().sample_rate;
            let budget = (num_samples as f32 / sample_rate).max(1.0e-9);
            self.cpu_meter
                .store((elapsed / budget).to_bits(), Ordering::Relaxed);

            // Report latency (constant per look-ahead / oversampling setting);
            // only when it actually changes to avoid churning the host.
            let latency = processor.latency_samples() as u32;
            if latency != self.reported_latency {
                self.reported_latency = latency;
                context.set_latency_samples(latency);
            }
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for LimiterPlugin {
    const CLAP_ID: &'static str = "org.codeberg.slundi.rt-limiter";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("True-peak look-ahead brick-wall limiter / maximiser");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Mono,
        ClapFeature::Limiter,
        ClapFeature::Mastering,
    ];
}

impl Vst3Plugin for LimiterPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"RtLimiterSlnd001";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Dynamics];
}

nih_export_clap!(LimiterPlugin);
nih_export_vst3!(LimiterPlugin);
