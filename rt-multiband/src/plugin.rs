//! `nih-plug` wrapper for `rt-multiband` (compiled only with the `nih` feature).
//!
//! A thin framework layer over the tested, framework-agnostic
//! [`engine`](crate::engine): it declares the automatable parameters (global +
//! four nested per-band groups), routes per-band `Threshold`/`Makeup` MIDI CC
//! into the snapshot, runs the resident [`MultibandProcessor`] on the audio
//! thread, and reports (zero) latency to the host.
//!
//! # Threading
//!
//! v1 has no reallocating parameter change, so there is **no background
//! controller** — the [`MultibandProcessor`] is built in `initialize` (off the
//! audio thread, where the host reports the sample rate) and applies every
//! control in place. `process()` is allocation-free.
//!
//! # I/O (Phase 2 decision)
//!
//! Stereo in → stereo out and mono → mono. The external (aux-input) sidechain
//! is deferred to v2 (Phase 0), so no auxiliary buses are declared.
//!
//! # MIDI (Phase 2)
//!
//! `Threshold` and `Makeup` are MIDI-CC-controllable per band (bands `0..4` ⇒
//! CC `20..24` for threshold, `24..28` for makeup — see
//! [`crate::params::decode_cc`]). A CC message sets an override that persists
//! until the host moves that band's knob, which reclaims control.

use std::num::NonZeroU32;
use std::sync::Arc;

use nih_plug::prelude::*;

use crate::engine::MultibandProcessor;
use crate::params::{
    ATTACK_MS_RANGE, BandModeChoice, BandParams, CROSSOVER_HZ_RANGE, CcTarget,
    DEFAULT_CROSSOVER_HZ, INPUT_GAIN_DB_RANGE, KNEE_DB_RANGE, MAKEUP_DB_RANGE, MultibandParams,
    OUTPUT_GAIN_DB_RANGE, RATIO_RANGE, RELEASE_MS_RANGE, SC_HPF_HZ_RANGE, StereoLinkChoice,
    THRESHOLD_DB_RANGE, cc_value_to_range, decode_cc,
};
use rt_multiband_core::NUM_BANDS;

/// `nih-plug` band-mode enum (mirrors [`BandModeChoice`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
enum BandModeParam {
    Compressor,
    Expander,
    Gate,
    Bypass,
}

impl BandModeParam {
    fn to_choice(self) -> BandModeChoice {
        match self {
            BandModeParam::Compressor => BandModeChoice::Compressor,
            BandModeParam::Expander => BandModeChoice::Expander,
            BandModeParam::Gate => BandModeChoice::Gate,
            BandModeParam::Bypass => BandModeChoice::Bypass,
        }
    }
}

/// `nih-plug` stereo-link enum (mirrors [`StereoLinkChoice`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
enum LinkParam {
    #[name = "Stereo Linked"]
    Linked,
    Independent,
}

impl LinkParam {
    fn to_choice(self) -> StereoLinkChoice {
        match self {
            LinkParam::Linked => StereoLinkChoice::Linked,
            LinkParam::Independent => StereoLinkChoice::Independent,
        }
    }
}

/// Host-visible per-band parameters (one nested group per band).
#[derive(Params)]
struct BandPluginParams {
    #[id = "mode"]
    mode: EnumParam<BandModeParam>,
    #[id = "thr"]
    threshold_db: FloatParam,
    #[id = "ratio"]
    ratio: FloatParam,
    #[id = "atk"]
    attack_ms: FloatParam,
    #[id = "rel"]
    release_ms: FloatParam,
    #[id = "knee"]
    knee_db: FloatParam,
    #[id = "makeup"]
    makeup_db: FloatParam,
    #[id = "blend"]
    blend: FloatParam,
    #[id = "schpf"]
    sc_hpf_hz: FloatParam,
    #[id = "solo"]
    solo: BoolParam,
    #[id = "mute"]
    mute: BoolParam,
}

impl BandPluginParams {
    fn new(defaults: BandParams) -> Self {
        Self {
            mode: EnumParam::new("Mode", BandModeParam::Compressor),
            threshold_db: FloatParam::new(
                "Threshold",
                defaults.threshold_db,
                FloatRange::Linear {
                    min: THRESHOLD_DB_RANGE.0,
                    max: THRESHOLD_DB_RANGE.1,
                },
            )
            .with_unit(" dB"),
            ratio: FloatParam::new(
                "Ratio",
                defaults.ratio,
                // Skewed so the useful low ratios get more travel.
                FloatRange::Skewed {
                    min: RATIO_RANGE.0,
                    max: RATIO_RANGE.1,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            .with_unit(":1"),
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
            knee_db: FloatParam::new(
                "Knee",
                defaults.knee_db,
                FloatRange::Linear {
                    min: KNEE_DB_RANGE.0,
                    max: KNEE_DB_RANGE.1,
                },
            )
            .with_unit(" dB"),
            makeup_db: FloatParam::new(
                "Makeup",
                defaults.makeup_db,
                FloatRange::Linear {
                    min: MAKEUP_DB_RANGE.0,
                    max: MAKEUP_DB_RANGE.1,
                },
            )
            .with_unit(" dB"),
            blend: FloatParam::new(
                "Detector Blend",
                defaults.blend,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            ),
            sc_hpf_hz: FloatParam::new(
                "SC HPF",
                defaults.sc_hpf_hz,
                FloatRange::Skewed {
                    min: SC_HPF_HZ_RANGE.0,
                    max: SC_HPF_HZ_RANGE.1,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            .with_unit(" Hz"),
            solo: BoolParam::new("Solo", defaults.solo),
            mute: BoolParam::new("Mute", defaults.mute),
        }
    }

    fn snapshot(&self) -> BandParams {
        BandParams {
            mode: self.mode.value().to_choice(),
            threshold_db: self.threshold_db.value(),
            ratio: self.ratio.value(),
            attack_ms: self.attack_ms.value(),
            release_ms: self.release_ms.value(),
            knee_db: self.knee_db.value(),
            makeup_db: self.makeup_db.value(),
            blend: self.blend.value(),
            sc_hpf_hz: self.sc_hpf_hz.value(),
            solo: self.solo.value(),
            mute: self.mute.value(),
        }
    }
}

/// Host-visible parameters.
#[derive(Params)]
struct MultibandPluginParams {
    #[id = "ingain"]
    input_gain_db: FloatParam,
    #[id = "outgain"]
    output_gain_db: FloatParam,
    #[id = "mix"]
    mix: FloatParam,
    #[id = "link"]
    link: EnumParam<LinkParam>,
    #[id = "xo1"]
    crossover1_hz: FloatParam,
    #[id = "xo2"]
    crossover2_hz: FloatParam,
    #[id = "xo3"]
    crossover3_hz: FloatParam,
    #[id = "byp"]
    bypass: BoolParam,

    #[nested(array, group = "Bands")]
    bands: [BandPluginParams; NUM_BANDS],
}

impl Default for MultibandPluginParams {
    fn default() -> Self {
        let defaults = MultibandParams::default();
        let bands = std::array::from_fn(|index| BandPluginParams::new(defaults.bands[index]));
        let crossover = |name: &str, default: f32| {
            FloatParam::new(
                name,
                default,
                FloatRange::Skewed {
                    min: CROSSOVER_HZ_RANGE.0,
                    max: CROSSOVER_HZ_RANGE.1,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_unit(" Hz")
        };
        Self {
            input_gain_db: FloatParam::new(
                "Input Gain",
                0.0,
                FloatRange::Linear {
                    min: INPUT_GAIN_DB_RANGE.0,
                    max: INPUT_GAIN_DB_RANGE.1,
                },
            )
            .with_unit(" dB"),
            output_gain_db: FloatParam::new(
                "Output Gain",
                0.0,
                FloatRange::Linear {
                    min: OUTPUT_GAIN_DB_RANGE.0,
                    max: OUTPUT_GAIN_DB_RANGE.1,
                },
            )
            .with_unit(" dB"),
            mix: FloatParam::new("Mix", 1.0, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_unit(" %")
                .with_value_to_string(formatters::v2s_f32_percentage(0))
                .with_string_to_value(formatters::s2v_f32_percentage()),
            link: EnumParam::new("Link", LinkParam::Linked),
            crossover1_hz: crossover("Crossover 1", DEFAULT_CROSSOVER_HZ[0]),
            crossover2_hz: crossover("Crossover 2", DEFAULT_CROSSOVER_HZ[1]),
            crossover3_hz: crossover("Crossover 3", DEFAULT_CROSSOVER_HZ[2]),
            bypass: BoolParam::new("Bypass", false)
                .with_value_to_string(formatters::v2s_bool_bypass()),
            bands,
        }
    }
}

impl MultibandPluginParams {
    /// Snapshot into the framework-agnostic model.
    fn snapshot(&self) -> MultibandParams {
        let bands = std::array::from_fn(|index| self.bands[index].snapshot());
        MultibandParams {
            input_gain_db: self.input_gain_db.value(),
            output_gain_db: self.output_gain_db.value(),
            mix: self.mix.value(),
            bypass: self.bypass.value(),
            link: self.link.value().to_choice(),
            crossover_hz: [
                self.crossover1_hz.value(),
                self.crossover2_hz.value(),
                self.crossover3_hz.value(),
            ],
            bands,
        }
    }
}

/// The multi-band plugin instance.
pub struct MultibandPlugin {
    params: Arc<MultibandPluginParams>,
    /// Audio-thread renderer; built in `initialize`.
    processor: Option<MultibandProcessor>,

    /// Active per-band MIDI-CC threshold overrides (dB), cleared on host move.
    midi_threshold_db: [Option<f32>; NUM_BANDS],
    /// Active per-band MIDI-CC makeup overrides (dB), cleared on host move.
    midi_makeup_db: [Option<f32>; NUM_BANDS],
    /// Last host threshold value per band, to detect a host-initiated move.
    last_threshold_db: [f32; NUM_BANDS],
    /// Last host makeup value per band, to detect a host-initiated move.
    last_makeup_db: [f32; NUM_BANDS],
}

impl Default for MultibandPlugin {
    fn default() -> Self {
        let params = MultibandPluginParams::default();
        let defaults = MultibandParams::default();
        let last_threshold_db = std::array::from_fn(|index| defaults.bands[index].threshold_db);
        let last_makeup_db = std::array::from_fn(|index| defaults.bands[index].makeup_db);
        Self {
            params: Arc::new(params),
            processor: None,
            midi_threshold_db: [None; NUM_BANDS],
            midi_makeup_db: [None; NUM_BANDS],
            last_threshold_db,
            last_makeup_db,
        }
    }
}

impl MultibandPlugin {
    /// Fold the active MIDI-CC overrides into `snapshot`, reclaiming a band's
    /// control when the host moved its knob since the last block.
    fn apply_midi(&mut self, snapshot: &mut MultibandParams) {
        for index in 0..NUM_BANDS {
            let host_threshold = snapshot.bands[index].threshold_db;
            if host_threshold != self.last_threshold_db[index] {
                // Host moved the knob → reclaim control from MIDI.
                self.midi_threshold_db[index] = None;
            }
            self.last_threshold_db[index] = host_threshold;
            if let Some(value) = self.midi_threshold_db[index] {
                snapshot.bands[index].threshold_db = value;
            }

            let host_makeup = snapshot.bands[index].makeup_db;
            if host_makeup != self.last_makeup_db[index] {
                self.midi_makeup_db[index] = None;
            }
            self.last_makeup_db[index] = host_makeup;
            if let Some(value) = self.midi_makeup_db[index] {
                snapshot.bands[index].makeup_db = value;
            }
        }
    }
}

impl Plugin for MultibandPlugin {
    const NAME: &'static str = "RT Multiband";
    const VENDOR: &'static str = "Sébastien Lundi";
    const URL: &'static str = "https://codeberg.org/slundi/rt-fft-convolver";
    const EMAIL: &'static str = "slundi@stamus-networks.com";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    // Stereo → stereo and mono → mono (Phase 2 I/O decision; external sidechain
    // deferred to v2, so no aux buses).
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

    // Per-band Threshold / Makeup are CC-controllable (Phase 2).
    const MIDI_INPUT: MidiConfig = MidiConfig::MidiCCs;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        let sample_rate = buffer_config.sample_rate;
        let snapshot = self.params.snapshot();
        self.processor = Some(MultibandProcessor::new(sample_rate, &snapshot));
        // Zero-latency IIR crossover (v1); report it explicitly.
        context.set_latency_samples(0);
        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // ── 1. MIDI: per-band Threshold / Makeup CC. ──
        while let Some(event) = context.next_event() {
            if let NoteEvent::MidiCC { cc, value, .. } = event {
                match decode_cc(cc) {
                    Some(CcTarget::Threshold(band)) => {
                        self.midi_threshold_db[band] =
                            Some(cc_value_to_range(value, THRESHOLD_DB_RANGE));
                    }
                    Some(CcTarget::Makeup(band)) => {
                        self.midi_makeup_db[band] = Some(cc_value_to_range(value, MAKEUP_DB_RANGE));
                    }
                    None => {}
                }
            }
        }

        // ── 2. Snapshot + fold MIDI overrides. ──
        let mut snapshot = self.params.snapshot();
        self.apply_midi(&mut snapshot);

        // ── 3. Render (the processor installs the DenormalGuard itself). ──
        if let Some(processor) = self.processor.as_mut() {
            processor.process(buffer.as_slice(), &snapshot);
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for MultibandPlugin {
    const CLAP_ID: &'static str = "org.codeberg.slundi.rt-multiband";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("Multi-band compressor / limiter / expander");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Mono,
        ClapFeature::Compressor,
        ClapFeature::Mastering,
    ];
}

impl Vst3Plugin for MultibandPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"RtMultibandSlnd1";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Dynamics];
}

nih_export_clap!(MultibandPlugin);
nih_export_vst3!(MultibandPlugin);
