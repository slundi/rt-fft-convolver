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
//!
//! # Editor (Phase 3, `gui` feature)
//!
//! A `nih-plug-egui` editor: an input spectrum with three draggable crossover
//! lines, per-band gain-reduction meters, per-band transfer curves, solo/mute,
//! and global in/out meters. The audio thread publishes read-only telemetry
//! through packed atomics; the display-only [`SpectrumProbe`] runs **only while
//! the editor is open**.

use std::num::NonZeroU32;
use std::sync::Arc;
#[cfg(feature = "gui")]
use std::sync::atomic::{AtomicU32, Ordering};

use nih_plug::prelude::*;
#[cfg(feature = "gui")]
use nih_plug_egui::EguiState;

use crate::engine::MultibandProcessor;
use crate::params::{
    ATTACK_MS_RANGE, BandModeChoice, BandParams, CROSSOVER_HZ_RANGE, CcTarget,
    DEFAULT_CROSSOVER_HZ, INPUT_GAIN_DB_RANGE, KNEE_DB_RANGE, MAKEUP_DB_RANGE, MultibandParams,
    OUTPUT_GAIN_DB_RANGE, RATIO_RANGE, RELEASE_MS_RANGE, SC_HPF_HZ_RANGE, StereoLinkChoice,
    THRESHOLD_DB_RANGE, cc_value_to_range, decode_cc,
};
#[cfg(feature = "gui")]
use crate::spectrum::SpectrumProbe;
#[cfg(feature = "gui")]
use crate::ui::{DISPLAY_MIN_HZ, SILENCE_DBFS, peak_dbfs};
use rt_multiband_core::NUM_BANDS;

/// Analysis frame for the editor spectrum probe (display only).
#[cfg(feature = "gui")]
const SPECTRUM_FRAME: usize = 1_024;
/// Log-spaced display bins in the editor spectrum.
#[cfg(feature = "gui")]
const SPECTRUM_BINS: usize = 56;

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

    #[cfg(feature = "gui")]
    #[persist = "editor-state"]
    editor_state: Arc<EguiState>,
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
            #[cfg(feature = "gui")]
            editor_state: EguiState::from_size(680, 560),
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

    // ── Editor telemetry (packed `f32::to_bits`), audio thread → editor. ──
    /// Per-band gain reduction in dB (`≤ 0`).
    #[cfg(feature = "gui")]
    band_gr_db: [Arc<AtomicU32>; NUM_BANDS],
    /// Input peak in dBFS.
    #[cfg(feature = "gui")]
    input_dbfs: Arc<AtomicU32>,
    /// Output peak in dBFS.
    #[cfg(feature = "gui")]
    output_dbfs: Arc<AtomicU32>,
    /// Log-spaced input magnitude spectrum (one packed `f32` per display bin).
    #[cfg(feature = "gui")]
    spectrum: Arc<Vec<AtomicU32>>,
    /// Display-only spectrum analyser; runs only while the editor is open.
    #[cfg(feature = "gui")]
    probe: Option<SpectrumProbe>,
    /// Reusable scratch for the probe output (off the audio-alloc path).
    #[cfg(feature = "gui")]
    spectrum_scratch: Vec<f32>,
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
            #[cfg(feature = "gui")]
            band_gr_db: std::array::from_fn(|_| Arc::new(AtomicU32::new(0.0f32.to_bits()))),
            #[cfg(feature = "gui")]
            input_dbfs: Arc::new(AtomicU32::new(SILENCE_DBFS.to_bits())),
            #[cfg(feature = "gui")]
            output_dbfs: Arc::new(AtomicU32::new(SILENCE_DBFS.to_bits())),
            #[cfg(feature = "gui")]
            spectrum: Arc::new((0..SPECTRUM_BINS).map(|_| AtomicU32::new(0)).collect()),
            #[cfg(feature = "gui")]
            probe: None,
            #[cfg(feature = "gui")]
            spectrum_scratch: vec![0.0; SPECTRUM_BINS],
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

    #[cfg(feature = "gui")]
    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        editor::create(self)
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
        #[cfg(feature = "gui")]
        {
            self.probe = Some(SpectrumProbe::new(
                SPECTRUM_FRAME,
                sample_rate,
                SPECTRUM_BINS,
                DISPLAY_MIN_HZ,
            ));
        }
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

        // ── 3. Input metering + spectrum (gui, before the buffer is
        //       overwritten; the probe runs only while the editor is open). ──
        #[cfg(feature = "gui")]
        let editor_open = self.params.editor_state.is_open();
        #[cfg(feature = "gui")]
        if editor_open {
            let slice = buffer.as_slice();
            let input_peak = slice
                .iter()
                .map(|channel| peak_dbfs(channel))
                .fold(SILENCE_DBFS, f32::max);
            self.input_dbfs
                .store(input_peak.to_bits(), Ordering::Relaxed);
            if let (Some(probe), Some(channel)) = (self.probe.as_mut(), slice.first())
                && probe.push_block(channel, &mut self.spectrum_scratch)
            {
                for (atomic, &magnitude) in self.spectrum.iter().zip(&self.spectrum_scratch) {
                    atomic.store(magnitude.to_bits(), Ordering::Relaxed);
                }
            }
        }

        // ── 4. Render (the processor installs the DenormalGuard itself). ──
        if let Some(processor) = self.processor.as_mut() {
            processor.process(buffer.as_slice(), &snapshot);
        }

        // ── 5. Output metering + per-band gain reduction (gui). ──
        #[cfg(feature = "gui")]
        if editor_open {
            let output_peak = buffer
                .as_slice()
                .iter()
                .map(|channel| peak_dbfs(channel))
                .fold(SILENCE_DBFS, f32::max);
            self.output_dbfs
                .store(output_peak.to_bits(), Ordering::Relaxed);
            if let Some(processor) = self.processor.as_ref() {
                for (index, atomic) in self.band_gr_db.iter().enumerate() {
                    atomic.store(
                        processor.band_gain_reduction_db(index).to_bits(),
                        Ordering::Relaxed,
                    );
                }
            }
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

/// The `nih-plug-egui` editor (Phase 3), isolated behind the `gui` feature.
#[cfg(feature = "gui")]
mod editor {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    use nih_plug::prelude::{Editor, FloatParam, ParamSetter};
    use nih_plug_egui::{create_egui_editor, egui, widgets};

    use super::{MultibandPlugin, MultibandPluginParams};
    use crate::ui::{
        DISPLAY_MAX_HZ, DISPLAY_MIN_HZ, MAX_GR_DB, METER_FLOOR_DBFS, SPECTRUM_FLOOR_DB,
        dbfs_to_fraction, format_db, format_hz, fraction_to_hz, gr_to_fraction, hz_to_fraction,
        spectral_heights, threshold_to_fraction, transfer_output_db,
    };
    use rt_multiband_core::NUM_BANDS;

    use egui::Color32;

    const PANEL: Color32 = Color32::from_rgb(0x14, 0x16, 0x1a);
    const GRID: Color32 = Color32::from_rgb(0x2a, 0x2e, 0x36);
    const SPECTRUM: Color32 = Color32::from_rgb(0x4c, 0x6a, 0x92);
    const CROSSOVER: Color32 = Color32::from_rgb(0xf0, 0xb0, 0x60);
    const GR_FILL: Color32 = Color32::from_rgb(0xd8, 0x7a, 0x52);
    const IN_FILL: Color32 = Color32::from_rgb(0x5a, 0x74, 0xa0);
    const OUT_FILL: Color32 = Color32::from_rgb(0x62, 0xb0, 0x8a);
    const CURVE: Color32 = Color32::from_rgb(0x8a, 0xc0, 0xf0);

    /// Per-frame editor state: which crossover line (if any) is being dragged.
    struct EditorState {
        dragging: Option<usize>,
    }

    /// Build the editor, cloning the lock-free telemetry handles into the
    /// drawing closure.
    pub(super) fn create(plugin: &MultibandPlugin) -> Option<Box<dyn Editor>> {
        let params = plugin.params.clone();
        let band_gr = plugin.band_gr_db.clone();
        let input_dbfs = plugin.input_dbfs.clone();
        let output_dbfs = plugin.output_dbfs.clone();
        let spectrum = plugin.spectrum.clone();

        create_egui_editor(
            plugin.params.editor_state.clone(),
            EditorState { dragging: None },
            |_ctx, _state| {},
            move |ctx, setter, state| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    header(ui, &params, setter);
                    ui.separator();
                    spectrum_panel(ui, &params, setter, &spectrum, state);
                    ui.separator();
                    band_strip(ui, &params, setter, &band_gr);
                    ui.separator();
                    meters(ui, &input_dbfs, &output_dbfs);
                });
                // Keep repainting so the meters/spectrum animate without
                // parameter automation driving the frame.
                ctx.request_repaint();
            },
        )
    }

    /// Read a packed-`f32` atomic.
    fn load(atomic: &AtomicU32) -> f32 {
        f32::from_bits(atomic.load(Ordering::Relaxed))
    }

    fn header(ui: &mut egui::Ui, params: &MultibandPluginParams, setter: &ParamSetter) {
        ui.horizontal(|ui| {
            ui.heading("RT Multiband");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add(widgets::ParamSlider::for_param(&params.bypass, setter));
                ui.label("Bypass");
            });
        });
        ui.horizontal(|ui| {
            ui.label("In");
            ui.add(widgets::ParamSlider::for_param(&params.input_gain_db, setter).with_width(90.0));
            ui.label("Out");
            ui.add(
                widgets::ParamSlider::for_param(&params.output_gain_db, setter).with_width(90.0),
            );
            ui.label("Mix");
            ui.add(widgets::ParamSlider::for_param(&params.mix, setter).with_width(90.0));
            ui.label("Link");
            ui.add(widgets::ParamSlider::for_param(&params.link, setter).with_width(120.0));
        });
    }

    /// The three host crossover parameters, low → high.
    fn crossover_params(params: &MultibandPluginParams) -> [&FloatParam; NUM_BANDS - 1] {
        [
            &params.crossover1_hz,
            &params.crossover2_hz,
            &params.crossover3_hz,
        ]
    }

    fn spectrum_panel(
        ui: &mut egui::Ui,
        params: &MultibandPluginParams,
        setter: &ParamSetter,
        spectrum: &[AtomicU32],
        state: &mut EditorState,
    ) {
        let (rect, response) =
            ui.allocate_exact_size(egui::vec2(ui.available_width(), 150.0), egui::Sense::drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, PANEL);

        // Spectrum bars on the log-frequency x-axis.
        let magnitudes: Vec<f32> = spectrum.iter().map(load).collect();
        let heights = spectral_heights(&magnitudes, SPECTRUM_FLOOR_DB);
        let bin_width = rect.width() / heights.len().max(1) as f32;
        for (index, &height) in heights.iter().enumerate() {
            let x = rect.left() + index as f32 * bin_width;
            let top = rect.bottom() - rect.height() * height;
            painter.rect_filled(
                egui::Rect::from_min_max(
                    egui::pos2(x, top),
                    egui::pos2(x + bin_width * 0.9, rect.bottom()),
                ),
                0.0,
                SPECTRUM,
            );
        }

        // Draggable crossover lines.
        let lines = crossover_params(params);
        let pointer_x = response.interact_pointer_pos().map(|pos| pos.x);
        // Grab the nearest line to the pointer on drag start.
        if response.drag_started()
            && let Some(px) = pointer_x
        {
            let nearest = lines
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    let ax = line_x(rect, a.1.value());
                    let bx = line_x(rect, b.1.value());
                    (ax - px).abs().partial_cmp(&(bx - px).abs()).unwrap()
                })
                .map(|(index, _)| index);
            state.dragging = nearest;
            if let Some(index) = nearest {
                setter.begin_set_parameter(lines[index]);
            }
        }
        if response.dragged()
            && let (Some(index), Some(px)) = (state.dragging, pointer_x)
        {
            let fraction = ((px - rect.left()) / rect.width()).clamp(0.0, 1.0);
            let hz = fraction_to_hz(fraction, DISPLAY_MIN_HZ, DISPLAY_MAX_HZ);
            setter.set_parameter(lines[index], hz);
        }
        if response.drag_stopped()
            && let Some(index) = state.dragging.take()
        {
            setter.end_set_parameter(lines[index]);
        }

        for line in lines {
            let x = line_x(rect, line.value());
            painter.line_segment(
                [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                egui::Stroke::new(2.0, CROSSOVER),
            );
            painter.text(
                egui::pos2(x + 3.0, rect.top() + 2.0),
                egui::Align2::LEFT_TOP,
                format_hz(line.value()),
                egui::FontId::monospace(10.0),
                CROSSOVER,
            );
        }
    }

    /// Pixel x of a crossover line at `hz` within `rect` (log axis).
    fn line_x(rect: egui::Rect, hz: f32) -> f32 {
        rect.left() + rect.width() * hz_to_fraction(hz, DISPLAY_MIN_HZ, DISPLAY_MAX_HZ)
    }

    fn band_strip(
        ui: &mut egui::Ui,
        params: &MultibandPluginParams,
        setter: &ParamSetter,
        band_gr: &[Arc<AtomicU32>; NUM_BANDS],
    ) {
        ui.columns(NUM_BANDS, |columns| {
            for (index, ui) in columns.iter_mut().enumerate() {
                let band = &params.bands[index];
                ui.vertical_centered(|ui| {
                    ui.strong(format!("Band {}", index + 1));
                });
                // Gain-reduction meter + transfer curve side by side.
                ui.horizontal(|ui| {
                    gr_meter(ui, load(&band_gr[index]));
                    transfer_curve(ui, &band.snapshot());
                });
                ui.add(widgets::ParamSlider::for_param(&band.mode, setter));
                ui.add(widgets::ParamSlider::for_param(&band.threshold_db, setter));
                ui.add(widgets::ParamSlider::for_param(&band.ratio, setter));
                ui.add(widgets::ParamSlider::for_param(&band.attack_ms, setter));
                ui.add(widgets::ParamSlider::for_param(&band.release_ms, setter));
                ui.add(widgets::ParamSlider::for_param(&band.knee_db, setter));
                ui.add(widgets::ParamSlider::for_param(&band.makeup_db, setter));
                ui.add(widgets::ParamSlider::for_param(&band.blend, setter));
                ui.add(widgets::ParamSlider::for_param(&band.sc_hpf_hz, setter));
                ui.horizontal(|ui| {
                    ui.add(widgets::ParamSlider::for_param(&band.solo, setter));
                    ui.add(widgets::ParamSlider::for_param(&band.mute, setter));
                });
            }
        });
    }

    /// A small vertical gain-reduction meter (grows downward from the top).
    fn gr_meter(ui: &mut egui::Ui, gr_db: f32) {
        let (rect, _) = ui.allocate_exact_size(egui::vec2(14.0, 90.0), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 2.0, PANEL);
        let fraction = gr_to_fraction(gr_db, MAX_GR_DB);
        let fill = egui::Rect::from_min_max(
            rect.left_top(),
            egui::pos2(rect.right(), rect.top() + rect.height() * fraction),
        );
        painter.rect_filled(fill, 2.0, GR_FILL);
    }

    /// A small static transfer-curve plot (input dB → output dB).
    fn transfer_curve(ui: &mut egui::Ui, band: &crate::params::BandParams) {
        let (rect, _) = ui.allocate_exact_size(egui::vec2(90.0, 90.0), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 2.0, PANEL);
        // 1:1 reference diagonal.
        painter.line_segment(
            [rect.left_bottom(), rect.right_top()],
            egui::Stroke::new(1.0, GRID),
        );
        // The transfer curve sampled across the threshold range.
        let steps = 48;
        let points: Vec<egui::Pos2> = (0..=steps)
            .map(|step| {
                let input_db = -60.0 + 60.0 * step as f32 / steps as f32;
                let output_db = transfer_output_db(band, input_db);
                let x = rect.left() + rect.width() * threshold_to_fraction(input_db);
                let y = rect.bottom() - rect.height() * threshold_to_fraction(output_db);
                egui::pos2(x, y)
            })
            .collect();
        painter.add(egui::Shape::line(points, egui::Stroke::new(1.5, CURVE)));
    }

    fn meters(ui: &mut egui::Ui, input_dbfs: &AtomicU32, output_dbfs: &AtomicU32) {
        meter_bar(ui, "In ", load(input_dbfs), IN_FILL);
        meter_bar(ui, "Out", load(output_dbfs), OUT_FILL);
    }

    /// A horizontal dBFS level bar with a numeric readout.
    fn meter_bar(ui: &mut egui::Ui, label: &str, dbfs: f32, fill: Color32) {
        ui.horizontal(|ui| {
            ui.monospace(label);
            let width = (ui.available_width() - 80.0).max(40.0);
            let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 14.0), egui::Sense::hover());
            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 2.0, PANEL);
            let fraction = dbfs_to_fraction(dbfs, METER_FLOOR_DBFS);
            let filled = egui::Rect::from_min_max(
                rect.left_top(),
                egui::pos2(rect.left() + rect.width() * fraction, rect.bottom()),
            );
            painter.rect_filled(filled, 2.0, fill);
            ui.monospace(format_db(dbfs));
        });
    }
}
