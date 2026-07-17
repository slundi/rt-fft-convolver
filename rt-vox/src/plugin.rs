//! `nih-plug` wrapper for `rt-vox` (compiled only with the `nih` feature).
//!
//! A thin framework layer over the tested, framework-agnostic
//! [`engine`](crate::engine): it declares the automatable parameters, routes
//! MIDI notes + the structural (band-count) change into the engine, runs the
//! [`VoxProcessor`](crate::engine::VoxProcessor) on the audio thread, and
//! reports the active mode's latency to the host.
//!
//! # I/O (Phase 2 decision)
//!
//! - Stereo main in → stereo out, plus a **mono sidechain** input carrying the
//!   vocoder carrier.
//! - Mono main in → mono out (also with the sidechain), for the simplest voice
//!   chains.
//!
//! The harmoniser tracks the main input's pitch and pans its voices to stereo;
//! the formant shifter processes each main channel independently.
//!
//! # MIDI
//!
//! - **Note on / off** drive the harmoniser's held-note chord
//!   ([`MidiConfig::Basic`]).
//! - **CC** control of `Formant Shift` / per-voice `Interval` is provided by
//!   exposing every continuous control as an automatable parameter, which hosts
//!   bind to any MIDI CC (host-side MIDI learn) — the idiomatic `nih-plug` path.
//!
//! # Threading
//!
//! The [`VoxController`](crate::engine::VoxController) lives behind a `Mutex`
//! shared with the `nih-plug` background-task executor; rebuilding the vocoder
//! (which reallocates FFT buffers) happens there, never on the audio thread.

use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use nih_plug::prelude::*;

use crate::engine::{VoxController, VoxProcessor};
use crate::params::{
    ATTACK_MS_RANGE, Bands, DETUNE_CENTS_RANGE, FORMANT_SHIFT_RANGE, Mode, NUM_VOICES,
    OUTPUT_GAIN_DB_RANGE, PITCH_SHIFT_RANGE, RELEASE_MS_RANGE, VOICE_INTERVAL_RANGE,
    VOICE_LEVEL_RANGE, VoiceParams, VoxParams,
};
use rt_vox_core::{ScaleKind, SnapMode};

/// Off-thread work dispatched to the background executor (control thread only).
pub enum VoxTask {
    /// Rebuild the vocoder for a new band count.
    ApplyStructural(VoxParams),
}

/// `nih-plug` mode enum, mirroring [`Mode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
enum ModeParam {
    Vocoder,
    Harmoniser,
    #[name = "Formant Shifter"]
    FormantShifter,
}

impl ModeParam {
    fn to_mode(self) -> Mode {
        match self {
            ModeParam::Vocoder => Mode::Vocoder,
            ModeParam::Harmoniser => Mode::Harmoniser,
            ModeParam::FormantShifter => Mode::FormantShifter,
        }
    }
}

/// `nih-plug` band-count enum, mirroring [`Bands`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
enum BandsParam {
    #[name = "8"]
    B8,
    #[name = "16"]
    B16,
    #[name = "24"]
    B24,
    #[name = "32"]
    B32,
}

impl BandsParam {
    fn to_bands(self) -> Bands {
        match self {
            BandsParam::B8 => Bands::B8,
            BandsParam::B16 => Bands::B16,
            BandsParam::B24 => Bands::B24,
            BandsParam::B32 => Bands::B32,
        }
    }
}

/// `nih-plug` snap-mode enum, mirroring [`SnapMode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
enum SnapModeParam {
    Free,
    Scale,
    #[name = "MIDI"]
    Midi,
}

impl SnapModeParam {
    fn to_snap(self) -> SnapMode {
        match self {
            SnapModeParam::Free => SnapMode::Free,
            SnapModeParam::Scale => SnapMode::Scale,
            SnapModeParam::Midi => SnapMode::Midi,
        }
    }
}

/// `nih-plug` scale-kind enum, mirroring [`ScaleKind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
enum ScaleKindParam {
    Major,
    #[name = "Natural Minor"]
    NaturalMinor,
    #[name = "Harmonic Minor"]
    HarmonicMinor,
    Chromatic,
}

impl ScaleKindParam {
    fn to_kind(self) -> ScaleKind {
        match self {
            ScaleKindParam::Major => ScaleKind::Major,
            ScaleKindParam::NaturalMinor => ScaleKind::NaturalMinor,
            ScaleKindParam::HarmonicMinor => ScaleKind::HarmonicMinor,
            ScaleKindParam::Chromatic => ScaleKind::Chromatic,
        }
    }
}

/// One harmony voice's automatable controls.  Nested as an array so `nih-plug`
/// auto-suffixes the ids (`vint1`, `vlvl1`, …).
#[derive(Params)]
struct VoicePluginParams {
    #[id = "vint"]
    interval: FloatParam,
    #[id = "vlvl"]
    level: FloatParam,
    #[id = "vpan"]
    pan: FloatParam,
    #[id = "vfmt"]
    formant_shift: FloatParam,
    #[id = "vdet"]
    detune: FloatParam,
    #[id = "ven"]
    enabled: BoolParam,
}

impl VoicePluginParams {
    fn new(defaults: VoiceParams) -> Self {
        Self {
            interval: FloatParam::new(
                "Interval",
                defaults.interval,
                FloatRange::Linear {
                    min: VOICE_INTERVAL_RANGE.0,
                    max: VOICE_INTERVAL_RANGE.1,
                },
            )
            .with_unit(" st")
            .with_step_size(1.0),
            level: FloatParam::new(
                "Level",
                defaults.level,
                FloatRange::Linear {
                    min: VOICE_LEVEL_RANGE.0,
                    max: VOICE_LEVEL_RANGE.1,
                },
            ),
            pan: FloatParam::new(
                "Pan",
                defaults.pan,
                FloatRange::Linear {
                    min: -1.0,
                    max: 1.0,
                },
            ),
            formant_shift: FloatParam::new(
                "Formant Shift",
                defaults.formant_shift,
                FloatRange::Linear {
                    min: FORMANT_SHIFT_RANGE.0,
                    max: FORMANT_SHIFT_RANGE.1,
                },
            )
            .with_unit(" st"),
            detune: FloatParam::new(
                "Detune",
                defaults.detune_cents,
                FloatRange::Linear {
                    min: DETUNE_CENTS_RANGE.0,
                    max: DETUNE_CENTS_RANGE.1,
                },
            )
            .with_unit(" ct"),
            enabled: BoolParam::new("Enabled", defaults.enabled),
        }
    }
}

/// Host-visible parameters.
#[derive(Params)]
struct VoxPluginParams {
    #[id = "mode"]
    mode: EnumParam<ModeParam>,

    // ── Vocoder ──
    #[id = "bands"]
    bands: EnumParam<BandsParam>,
    #[id = "atk"]
    attack_ms: FloatParam,
    #[id = "rel"]
    release_ms: FloatParam,
    #[id = "vocfmt"]
    vocoder_formant_shift: FloatParam,
    #[id = "sib"]
    sibilance: BoolParam,
    #[id = "sibamt"]
    sibilance_amount: FloatParam,

    // ── Harmoniser ──
    #[nested(array, group = "Voices")]
    voices: [VoicePluginParams; NUM_VOICES],
    #[id = "snap"]
    snap_mode: EnumParam<SnapModeParam>,
    #[id = "key"]
    key: IntParam,
    #[id = "scale"]
    scale_kind: EnumParam<ScaleKindParam>,

    // ── Formant shifter ──
    #[id = "fmt"]
    formant_shift: FloatParam,
    #[id = "fpitch"]
    formant_pitch_shift: FloatParam,

    // ── Global ──
    #[id = "mix"]
    mix: FloatParam,
    #[id = "gain"]
    output_gain_db: FloatParam,
    #[id = "drycomp"]
    dry_delay_compensation: BoolParam,
    #[id = "byp"]
    bypass: BoolParam,
}

impl Default for VoxPluginParams {
    fn default() -> Self {
        let defaults = VoxParams::default();
        let voices = std::array::from_fn(|index| VoicePluginParams::new(defaults.voices[index]));

        Self {
            mode: EnumParam::new("Mode", ModeParam::Vocoder),
            bands: EnumParam::new("Bands", BandsParam::B16),
            attack_ms: FloatParam::new(
                "Attack",
                defaults.attack_ms,
                FloatRange::Skewed {
                    min: ATTACK_MS_RANGE.0,
                    max: ATTACK_MS_RANGE.1,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_unit(" ms"),
            release_ms: FloatParam::new(
                "Release",
                defaults.release_ms,
                FloatRange::Skewed {
                    min: RELEASE_MS_RANGE.0,
                    max: RELEASE_MS_RANGE.1,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_unit(" ms"),
            vocoder_formant_shift: FloatParam::new(
                "Vocoder Formant Shift",
                defaults.vocoder_formant_shift,
                FloatRange::Linear {
                    min: FORMANT_SHIFT_RANGE.0,
                    max: FORMANT_SHIFT_RANGE.1,
                },
            )
            .with_unit(" st"),
            sibilance: BoolParam::new("Sibilance", defaults.sibilance),
            sibilance_amount: FloatParam::new(
                "Sibilance Amount",
                defaults.sibilance_amount,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage()),
            voices,
            snap_mode: EnumParam::new("Snap Mode", SnapModeParam::Midi),
            key: IntParam::new("Key", 0, IntRange::Linear { min: 0, max: 11 }),
            scale_kind: EnumParam::new("Scale", ScaleKindParam::Major),
            formant_shift: FloatParam::new(
                "Formant Shift",
                defaults.formant_shift,
                FloatRange::Linear {
                    min: FORMANT_SHIFT_RANGE.0,
                    max: FORMANT_SHIFT_RANGE.1,
                },
            )
            .with_unit(" st"),
            formant_pitch_shift: FloatParam::new(
                "Pitch Shift",
                defaults.formant_pitch_shift,
                FloatRange::Linear {
                    min: PITCH_SHIFT_RANGE.0,
                    max: PITCH_SHIFT_RANGE.1,
                },
            )
            .with_unit(" st"),
            mix: FloatParam::new(
                "Mix",
                defaults.mix,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_unit(" %")
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage()),
            output_gain_db: FloatParam::new(
                "Output Gain",
                defaults.output_gain_db,
                FloatRange::Linear {
                    min: OUTPUT_GAIN_DB_RANGE.0,
                    max: OUTPUT_GAIN_DB_RANGE.1,
                },
            )
            .with_unit(" dB"),
            dry_delay_compensation: BoolParam::new(
                "Dry Delay Compensation",
                defaults.dry_delay_compensation,
            ),
            bypass: BoolParam::new("Bypass", false)
                .with_value_to_string(formatters::v2s_bool_bypass()),
        }
    }
}

impl VoxPluginParams {
    /// Snapshot into the framework-agnostic model.
    fn snapshot(&self) -> VoxParams {
        let mut voices = [VoiceParams::default(); NUM_VOICES];
        for (dst, src) in voices.iter_mut().zip(self.voices.iter()) {
            dst.interval = src.interval.value();
            dst.level = src.level.value();
            dst.pan = src.pan.value();
            dst.formant_shift = src.formant_shift.value();
            dst.detune_cents = src.detune.value();
            dst.enabled = src.enabled.value();
        }

        VoxParams {
            mode: self.mode.value().to_mode(),
            bands: self.bands.value().to_bands(),
            attack_ms: self.attack_ms.value(),
            release_ms: self.release_ms.value(),
            vocoder_formant_shift: self.vocoder_formant_shift.value(),
            sibilance: self.sibilance.value(),
            sibilance_amount: self.sibilance_amount.value(),
            voices,
            snap_mode: self.snap_mode.value().to_snap(),
            key: self.key.value() as u8,
            scale_kind: self.scale_kind.value().to_kind(),
            formant_shift: self.formant_shift.value(),
            formant_pitch_shift: self.formant_pitch_shift.value(),
            mix: self.mix.value(),
            output_gain_db: self.output_gain_db.value(),
            dry_delay_compensation: self.dry_delay_compensation.value(),
            bypass: self.bypass.value(),
        }
    }
}

/// The vox plugin instance.
pub struct VoxPlugin {
    params: Arc<VoxPluginParams>,
    /// Shared with the background-task executor; built in `initialize`.
    controller: Arc<Mutex<Option<VoxController>>>,
    /// Audio-thread renderer; built in `initialize`.
    processor: Option<VoxProcessor>,

    /// Change detection so we only rebuild on an actual band-count move.
    last_bands: BandsParam,
    /// Last latency reported to the host, to avoid redundant notifications.
    last_reported_latency: usize,

    // ── Editor telemetry (audio thread → future egui editor). ──
    /// Detected input pitch in Hz (`f32` bits), or `0` when untracked.
    detected_hz: Arc<AtomicU32>,
    /// Current reported latency in samples.
    latency_samples: Arc<AtomicU32>,
}

impl Default for VoxPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(VoxPluginParams::default()),
            controller: Arc::new(Mutex::new(None)),
            processor: None,
            last_bands: BandsParam::B16,
            last_reported_latency: usize::MAX,
            detected_hz: Arc::new(AtomicU32::new(0)),
            latency_samples: Arc::new(AtomicU32::new(0)),
        }
    }
}

impl Plugin for VoxPlugin {
    const NAME: &'static str = "RT Vox";
    const VENDOR: &'static str = "Sébastien Lundi";
    const URL: &'static str = "https://codeberg.org/slundi/rt-fft-convolver";
    const EMAIL: &'static str = "slundi@stamus-networks.com";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    // Stereo → stereo and mono → mono, each with a mono sidechain (carrier).
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

    // Note input drives the harmoniser chord; CCs bind to params via MIDI learn.
    const MIDI_INPUT: MidiConfig = MidiConfig::Basic;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = VoxTask;

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
                VoxTask::ApplyStructural(params) => {
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
            VoxController::new(max_block, sample_rate, channels, &snapshot);

        // Report the initial latency for the current mode.
        let latency = processor.latency_samples(snapshot.mode);
        context.set_latency_samples(latency as u32);
        self.last_reported_latency = latency;
        self.latency_samples
            .store(latency as u32, Ordering::Relaxed);

        *self.controller.lock().unwrap() = Some(controller);
        self.processor = Some(processor);
        self.last_bands = self.params.bands.value();
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
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // ── 1. MIDI notes drive the harmoniser chord. ──
        while let Some(event) = context.next_event() {
            if let Some(processor) = self.processor.as_mut() {
                match event {
                    NoteEvent::NoteOn { note, .. } => processor.note_on(note),
                    NoteEvent::NoteOff { note, .. } => processor.note_off(note),
                    NoteEvent::Choke { .. } => processor.all_notes_off(),
                    _ => {}
                }
            }
        }

        // ── 2. Structural (band-count) change → background rebuild. ──
        let bands = self.params.bands.value();
        if bands != self.last_bands {
            self.last_bands = bands;
            context.execute_background(VoxTask::ApplyStructural(self.params.snapshot()));
        }

        // ── 3. Render — flush denormals for the whole callback. ──
        let _denormal_guard = rt_fft_convolver::DenormalGuard::new();
        let snapshot = self.params.snapshot();

        if let Some(processor) = self.processor.as_mut() {
            // The (mono) sidechain carrier, if one is patched.
            let sidechain: Option<&[f32]> = aux
                .inputs
                .first_mut()
                .and_then(|carrier| carrier.as_slice().first().map(|channel| &**channel));

            processor.process(buffer.as_slice(), sidechain, &snapshot);

            // Report the active mode's latency (changes with the Mode control).
            let latency = processor.reported_latency();
            if latency != self.last_reported_latency {
                self.last_reported_latency = latency;
                context.set_latency_samples(latency as u32);
            }
            self.latency_samples
                .store(latency as u32, Ordering::Relaxed);
            self.detected_hz.store(
                processor.detected_hz().unwrap_or(0.0).to_bits(),
                Ordering::Relaxed,
            );
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for VoxPlugin {
    const CLAP_ID: &'static str = "org.codeberg.slundi.rt-vox";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("Vocoder / harmoniser / formant shifter voice-processing plugin");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Mono,
        ClapFeature::PitchShifter,
        ClapFeature::Custom("vocoder"),
    ];
}

impl Vst3Plugin for VoxPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"RtVoxSlundi00001";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::PitchShift];
}

nih_export_clap!(VoxPlugin);
nih_export_vst3!(VoxPlugin);
