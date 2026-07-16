//! `nih-plug` wrapper for the FDN reverb (compiled only with the `nih` feature).
//!
//! A thin framework layer over the tested, framework-agnostic
//! [`ReverbEngine`](crate::engine::ReverbEngine): it declares the automatable
//! parameters, maps them onto [`ReverbParams`], folds in live MIDI control, and
//! runs the engine on the audio thread.  There is no worker thread — the engine
//! pre-builds every voice at [`initialize`](FdnReverbPlugin::initialize), so the
//! audio callback only ever mutates scalar coefficients.
//!
//! # MIDI (roadmap Phase 2)
//!
//! - **Program Change** → active preset slot ([`program_to_slot`]).
//! - **Control Change** on three fixed controllers → `Decay` / `Size` /
//!   `Freeze`, applied as an override on top of the host value.  An override is
//!   released as soon as the host moves that parameter itself, so the DAW always
//!   wins back control.  A proper MIDI-learn binding UI lands in Phase 3.
//!
//! # Latency
//!
//! Zero — the dry path is a direct add; the pre-delay lives on the wet path
//! only, so no latency is reported to the host.

use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use nih_plug::prelude::*;
use rt_fdn_core::PRESETS;
use rt_ir_plugin_core::program_to_slot;

use crate::engine::ReverbEngine;
use crate::params::{
    CROSSFADE_MS_RANGE, DECAY_S_RANGE, FADER_DB_RANGE, HIGH_CUT_RANGE, LOW_CUT_RANGE,
    MOD_RATE_HZ_RANGE, NUM_PRESETS, NUM_SPACES, PREDELAY_MS_RANGE, ReverbParams,
};

/// CC number that live-controls `Decay`.
const CC_DECAY: u8 = 20;
/// CC number that live-controls `Size`.
const CC_SIZE: u8 = 21;
/// CC number that live-controls `Freeze` (value ≥ 0.5 = frozen).
const CC_FREEZE: u8 = 22;

/// Per-slot parameters: which built-in space, and whether the slot is enabled.
#[derive(Params)]
struct SlotParams {
    #[id = "space"]
    space: IntParam,
    #[id = "on"]
    enabled: BoolParam,
}

impl SlotParams {
    fn new(default_space: usize) -> Self {
        Self {
            space: IntParam::new(
                "Space",
                default_space as i32,
                IntRange::Linear {
                    min: 0,
                    max: (NUM_SPACES - 1) as i32,
                },
            )
            .with_value_to_string(Arc::new(|value| {
                PRESETS
                    .get(value as usize)
                    .map(|preset| preset.name.to_string())
                    .unwrap_or_else(|| value.to_string())
            })),
            enabled: BoolParam::new("Preset Enabled", true),
        }
    }
}

/// Host-visible parameters (Phase-2 control list).
#[derive(Params)]
struct FdnReverbParams {
    #[id = "preset"]
    current_preset: IntParam,

    #[nested(array, group = "Slots")]
    slots: [SlotParams; NUM_PRESETS],

    #[id = "xfade"]
    crossfade_ms: FloatParam,
    #[id = "predly"]
    predelay_ms: FloatParam,
    #[id = "decay"]
    decay_s: FloatParam,
    #[id = "size"]
    size: FloatParam,
    #[id = "damp"]
    damping: FloatParam,
    #[id = "diff"]
    diffusion: FloatParam,
    #[id = "moddep"]
    mod_depth: FloatParam,
    #[id = "modrate"]
    mod_rate: FloatParam,
    #[id = "locut"]
    low_cut: FloatParam,
    #[id = "hicut"]
    high_cut: FloatParam,
    #[id = "width"]
    width: FloatParam,
    #[id = "dry"]
    dry_db: FloatParam,
    #[id = "wet"]
    wet_db: FloatParam,
    #[id = "freeze"]
    freeze: BoolParam,
    #[id = "byp"]
    bypass: BoolParam,
}

/// Build a linear `FloatParam` over an inclusive `(min, max)` range.
fn linear(name: &str, default: f32, range: (f32, f32)) -> FloatParam {
    FloatParam::new(
        name,
        default,
        FloatRange::Linear {
            min: range.0,
            max: range.1,
        },
    )
}

/// Build a `0..=1` percent `FloatParam`.
fn percent(name: &str, default: f32) -> FloatParam {
    linear(name, default, (0.0, 1.0))
        .with_unit(" %")
        .with_value_to_string(formatters::v2s_f32_percentage(0))
        .with_string_to_value(formatters::s2v_f32_percentage())
}

impl Default for FdnReverbParams {
    fn default() -> Self {
        let defaults = ReverbParams::default();
        Self {
            current_preset: IntParam::new(
                "Current Preset",
                defaults.current_preset as i32,
                IntRange::Linear {
                    min: 0,
                    max: (NUM_PRESETS - 1) as i32,
                },
            ),
            slots: core::array::from_fn(|i| SlotParams::new(defaults.space[i])),
            crossfade_ms: linear("Crossfade Time", defaults.crossfade_ms, CROSSFADE_MS_RANGE)
                .with_unit(" ms"),
            predelay_ms: linear("Pre-Delay", defaults.predelay_ms, PREDELAY_MS_RANGE)
                .with_unit(" ms"),
            decay_s: FloatParam::new(
                "Decay",
                defaults.decay_s,
                FloatRange::Skewed {
                    min: DECAY_S_RANGE.0,
                    max: DECAY_S_RANGE.1,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_unit(" s"),
            size: percent("Size", defaults.size),
            damping: percent("Damping", defaults.damping),
            diffusion: percent("Diffusion", defaults.diffusion),
            mod_depth: percent("Modulation Depth", defaults.mod_depth),
            mod_rate: linear("Modulation Rate", defaults.mod_rate_hz, MOD_RATE_HZ_RANGE)
                .with_unit(" Hz"),
            low_cut: FloatParam::new(
                "Low Cut",
                defaults.low_cut_hz,
                FloatRange::Skewed {
                    min: LOW_CUT_RANGE.0,
                    max: LOW_CUT_RANGE.1,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            .with_unit(" Hz"),
            high_cut: FloatParam::new(
                "High Cut",
                defaults.high_cut_hz,
                FloatRange::Skewed {
                    min: HIGH_CUT_RANGE.0,
                    max: HIGH_CUT_RANGE.1,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            .with_unit(" Hz"),
            width: linear("Stereo Width", defaults.width, (0.0, 2.0))
                .with_unit(" %")
                .with_value_to_string(formatters::v2s_f32_percentage(0))
                .with_string_to_value(formatters::s2v_f32_percentage()),
            dry_db: linear("Dry", defaults.dry_db, FADER_DB_RANGE).with_unit(" dB"),
            wet_db: linear("Wet", defaults.wet_db, FADER_DB_RANGE).with_unit(" dB"),
            freeze: BoolParam::new("Freeze", defaults.freeze),
            bypass: BoolParam::new("Bypass", false)
                .with_value_to_string(formatters::v2s_bool_bypass()),
        }
    }
}

impl FdnReverbParams {
    /// Snapshot the host parameters into the framework-agnostic model.
    fn snapshot(&self) -> ReverbParams {
        ReverbParams {
            current_preset: self.current_preset.value() as usize,
            space: core::array::from_fn(|i| self.slots[i].space.value() as usize),
            enabled: core::array::from_fn(|i| self.slots[i].enabled.value()),
            crossfade_ms: self.crossfade_ms.value(),
            predelay_ms: self.predelay_ms.value(),
            decay_s: self.decay_s.value(),
            size: self.size.value(),
            damping: self.damping.value(),
            diffusion: self.diffusion.value(),
            mod_depth: self.mod_depth.value(),
            mod_rate_hz: self.mod_rate.value(),
            low_cut_hz: self.low_cut.value(),
            high_cut_hz: self.high_cut.value(),
            width: self.width.value(),
            dry_db: self.dry_db.value(),
            wet_db: self.wet_db.value(),
            freeze: self.freeze.value(),
            bypass: self.bypass.value(),
        }
        .clamped()
    }
}

/// Live MIDI overrides.  Each is released once the host moves the matching
/// parameter (tracked by comparing the parameter value block-to-block).
#[derive(Default)]
struct MidiOverrides {
    preset: Option<usize>,
    decay_s: Option<f32>,
    size: Option<f32>,
    freeze: Option<bool>,

    last_preset: i32,
    last_decay: f32,
    last_size: f32,
    last_freeze: bool,
}

/// The FDN reverb plugin instance.
pub struct FdnReverbPlugin {
    params: Arc<FdnReverbParams>,
    engine: Option<ReverbEngine>,
    midi: MidiOverrides,
    /// CPU load = render time / block budget (`f32` bits), for the Phase-3 meter.
    cpu_meter: Arc<AtomicU32>,
}

impl Default for FdnReverbPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(FdnReverbParams::default()),
            engine: None,
            midi: MidiOverrides::default(),
            cpu_meter: Arc::new(AtomicU32::new(0)),
        }
    }
}

impl FdnReverbPlugin {
    /// Fold MIDI events + host-vs-MIDI arbitration into a parameter snapshot.
    fn resolve_snapshot(&mut self, mut snapshot: ReverbParams) -> ReverbParams {
        // Release each override the moment the host moves its parameter.
        let preset = self.params.current_preset.value();
        if preset != self.midi.last_preset {
            self.midi.preset = None;
        }
        self.midi.last_preset = preset;

        let decay = self.params.decay_s.value();
        if decay != self.midi.last_decay {
            self.midi.decay_s = None;
        }
        self.midi.last_decay = decay;

        let size = self.params.size.value();
        if size != self.midi.last_size {
            self.midi.size = None;
        }
        self.midi.last_size = size;

        let freeze = self.params.freeze.value();
        if freeze != self.midi.last_freeze {
            self.midi.freeze = None;
        }
        self.midi.last_freeze = freeze;

        if let Some(preset) = self.midi.preset {
            snapshot.current_preset = preset.min(NUM_PRESETS - 1);
        }
        if let Some(decay) = self.midi.decay_s {
            snapshot.decay_s = decay;
        }
        if let Some(size) = self.midi.size {
            snapshot.size = size;
        }
        if let Some(freeze) = self.midi.freeze {
            snapshot.freeze = freeze;
        }
        snapshot
    }

    /// Decode one MIDI event into an override.
    fn handle_midi(&mut self, event: &NoteEvent<()>) {
        match *event {
            NoteEvent::MidiProgramChange { program, .. } => {
                if let Some(slot) = program_to_slot(program, NUM_PRESETS) {
                    self.midi.preset = Some(slot);
                }
            }
            NoteEvent::MidiCC { cc, value, .. } => match cc {
                CC_DECAY => {
                    self.midi.decay_s =
                        Some(DECAY_S_RANGE.0 + value * (DECAY_S_RANGE.1 - DECAY_S_RANGE.0));
                }
                CC_SIZE => self.midi.size = Some(value.clamp(0.0, 1.0)),
                CC_FREEZE => self.midi.freeze = Some(value >= 0.5),
                _ => {}
            },
            _ => {}
        }
    }
}

impl Plugin for FdnReverbPlugin {
    const NAME: &'static str = "RT FDN Reverb";
    const VENDOR: &'static str = "Sébastien Lundi";
    const URL: &'static str = "https://codeberg.org/slundi/rt-fft-convolver";
    const EMAIL: &'static str = "slundi@stamus-networks.com";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    // Stereo in → stereo out.
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),
        ..AudioIOLayout::const_default()
    }];

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
        let sample_rate = buffer_config.sample_rate.round().max(1.0) as u32;
        let snapshot = self.params.snapshot();
        self.engine = Some(ReverbEngine::new(sample_rate, &snapshot));

        // Zero-latency dry path.
        context.set_latency_samples(0);

        self.midi.last_preset = self.params.current_preset.value();
        self.midi.last_decay = self.params.decay_s.value();
        self.midi.last_size = self.params.size.value();
        self.midi.last_freeze = self.params.freeze.value();
        true
    }

    fn reset(&mut self) {
        if let Some(engine) = self.engine.as_mut() {
            engine.reset();
        }
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // Collect MIDI up front (preset / knob switches are not sample-accurate).
        while let Some(event) = context.next_event() {
            self.handle_midi(&event);
        }

        let base = self.params.snapshot();
        let snapshot = self.resolve_snapshot(base);
        let num_samples = buffer.samples();

        // Flush denormals for the whole callback — multi-second tails.
        let _denormal_guard = rt_fdn_core::DenormalGuard::new();

        if let Some(engine) = self.engine.as_mut() {
            engine.apply_params(&snapshot);

            let start = std::time::Instant::now();
            let channels = buffer.as_slice();
            match channels {
                [left, right, ..] => {
                    let n = left.len().min(right.len());
                    for i in 0..n {
                        let (out_l, out_r) = engine.process_sample(left[i], right[i]);
                        left[i] = out_l;
                        right[i] = out_r;
                    }
                }
                [mono] => {
                    for sample in mono.iter_mut() {
                        let (out_l, out_r) = engine.process_sample(*sample, *sample);
                        *sample = 0.5 * (out_l + out_r);
                    }
                }
                [] => {}
            }
            let elapsed = start.elapsed().as_secs_f32();

            // CPU load = render time / block wall-clock budget (read-only meter).
            let sample_rate = context.transport().sample_rate;
            let budget = (num_samples as f32 / sample_rate).max(1.0e-9);
            self.cpu_meter
                .store((elapsed / budget).to_bits(), Ordering::Relaxed);
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for FdnReverbPlugin {
    const CLAP_ID: &'static str = "org.codeberg.slundi.rt-fdn-reverb";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("Algorithmic FDN reverb: eight modulated true-stereo spaces with freeze");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Reverb,
    ];
}

impl Vst3Plugin for FdnReverbPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"RtFdnReverbSlnd1";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Reverb];
}

nih_export_clap!(FdnReverbPlugin);
nih_export_vst3!(FdnReverbPlugin);
