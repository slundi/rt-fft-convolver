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
use std::sync::{Arc, Mutex};

use nih_plug::prelude::*;
use rt_ir_plugin_core::program_to_slot;
use rt_pitch_core::{FormantMethod, NUM_PRESETS};

use crate::engine::{PitchController, PitchProcessor};
use crate::params::{Algorithm, CROSSFADE_MS_RANGE, PitchParams, Quality, SEMITONE_RANGE};

/// Sentinel for "no MIDI Program Change override in effect".
const NO_OVERRIDE: i32 = -1;

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
        }
    }
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

        // Report the initial latency to the host.
        let latency = controller.latency();
        context.set_latency_samples(latency as u32);
        self.last_reported_latency = latency;

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
            if let NoteEvent::MidiProgramChange { program, .. } = event
                && let Some(index) = program_to_slot(program, NUM_PRESETS)
            {
                self.midi_preset = index as i32;
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
