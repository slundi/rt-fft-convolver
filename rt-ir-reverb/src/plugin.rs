//! `nih-plug` wrapper for the reverb (compiled only with the `nih` feature).
//!
//! Thin framework layer over the tested [`engine`](crate::engine): declares the
//! automatable parameters, routes MIDI + parameter changes into the off-thread
//! [`ReverbController`], and runs the [`StereoBlockAdapter`] on the audio
//! thread.  See `rt-ir-cab`'s wrapper for the shared threading rationale.
//!
//! Phase-3/4 follow-ups: drag-and-drop IR assignment + MIDI-learn UI + CPU-meter
//! readout (Phase 3); host validation and the bundled room-IR pack that the
//! presets pair with (Phase 4/5).

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use nih_plug::prelude::*;
#[cfg(feature = "gui")]
use nih_plug_egui::{EguiState, create_egui_editor, egui, widgets};
use parking_lot::Mutex as PlMutex;
use rt_ir_plugin_core::SlotEvent;

#[cfg(feature = "gui")]
use rt_ir_plugin_core::{MidiStatus, PadState, WaveformSummary, pad_state};
#[cfg(feature = "gui")]
use std::sync::atomic::{AtomicU32, Ordering};

/// Waveform preview resolution (buckets across the frame width).
#[cfg(feature = "gui")]
const WAVEFORM_BUCKETS: usize = 256;

use crate::block::StereoBlockAdapter;
use crate::engine::ReverbController;
use crate::params::{
    CROSSFADE_MS_RANGE, FADER_DB_RANGE, HIGH_CUT_RANGE, IR_LENGTH_RANGE, LOW_CUT_RANGE, NUM_SLOTS,
    PREDELAY_MS_RANGE, ReverbParams, WIDTH_RANGE,
};

/// Fixed internal render block; the plugin reports `BLOCK_SIZE` latency.
const BLOCK_SIZE: usize = 512;

/// Off-thread work dispatched to the background executor.
pub enum ReverbTask {
    /// Load a WAV into a slot (Phase-3 UI caller; init restores paths directly).
    #[allow(dead_code)]
    Load { slot: usize, path: PathBuf },
    /// Rebuild + hand off the reverb for a slot.
    Activate(usize),
    /// Route a MIDI event to a slot.
    Midi(SlotEvent),
    /// Enable/disable a slot.
    SetEnabled { slot: usize, enabled: bool },
    /// Set IR reversal for future activations.
    SetReverse(bool),
    /// Set the IR length fraction for future activations.
    SetLength(f32),
}

#[derive(Params)]
struct SlotParams {
    #[id = "on"]
    enabled: BoolParam,
}

impl Default for SlotParams {
    fn default() -> Self {
        Self {
            enabled: BoolParam::new("Slot Enabled", true),
        }
    }
}

#[derive(Params)]
struct ReverbPluginParams {
    #[id = "slot"]
    current_slot: IntParam,
    #[id = "xfade"]
    crossfade_ms: FloatParam,
    #[id = "predly"]
    predelay_ms: FloatParam,
    #[id = "dry"]
    dry_db: FloatParam,
    #[id = "wet"]
    wet_db: FloatParam,
    #[id = "width"]
    width: FloatParam,
    #[id = "locut"]
    low_cut: FloatParam,
    #[id = "hicut"]
    high_cut: FloatParam,
    #[id = "irrev"]
    ir_reverse: BoolParam,
    #[id = "irlen"]
    ir_length: FloatParam,
    #[id = "byp"]
    bypass: BoolParam,

    #[nested(array, group = "Slots")]
    slots: [SlotParams; NUM_SLOTS],

    #[persist = "ir_paths"]
    ir_paths: PlMutex<Vec<Option<String>>>,

    #[cfg(feature = "gui")]
    #[persist = "editor-state"]
    editor_state: Arc<EguiState>,
}

impl Default for ReverbPluginParams {
    fn default() -> Self {
        Self {
            current_slot: IntParam::new(
                "Current Slot",
                0,
                IntRange::Linear {
                    min: 0,
                    max: (NUM_SLOTS - 1) as i32,
                },
            ),
            crossfade_ms: FloatParam::new(
                "Crossfade Time",
                100.0,
                FloatRange::Linear {
                    min: CROSSFADE_MS_RANGE.0,
                    max: CROSSFADE_MS_RANGE.1,
                },
            )
            .with_unit(" ms"),
            predelay_ms: FloatParam::new(
                "Pre-Delay",
                0.0,
                FloatRange::Linear {
                    min: PREDELAY_MS_RANGE.0,
                    max: PREDELAY_MS_RANGE.1,
                },
            )
            .with_unit(" ms"),
            dry_db: FloatParam::new(
                "Dry",
                0.0,
                FloatRange::Linear {
                    min: FADER_DB_RANGE.0,
                    max: FADER_DB_RANGE.1,
                },
            )
            .with_unit(" dB"),
            wet_db: FloatParam::new(
                "Wet",
                0.0,
                FloatRange::Linear {
                    min: FADER_DB_RANGE.0,
                    max: FADER_DB_RANGE.1,
                },
            )
            .with_unit(" dB"),
            width: FloatParam::new(
                "Stereo Width",
                1.0,
                FloatRange::Linear {
                    min: WIDTH_RANGE.0,
                    max: WIDTH_RANGE.1,
                },
            )
            .with_unit(" %")
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage()),
            low_cut: FloatParam::new(
                "Low Cut",
                LOW_CUT_RANGE.0,
                FloatRange::Skewed {
                    min: LOW_CUT_RANGE.0,
                    max: LOW_CUT_RANGE.1,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            .with_unit(" Hz"),
            high_cut: FloatParam::new(
                "High Cut",
                HIGH_CUT_RANGE.1,
                FloatRange::Skewed {
                    min: HIGH_CUT_RANGE.0,
                    max: HIGH_CUT_RANGE.1,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            .with_unit(" Hz"),
            ir_reverse: BoolParam::new("IR Reverse", false),
            ir_length: FloatParam::new(
                "IR Length",
                1.0,
                FloatRange::Linear {
                    min: IR_LENGTH_RANGE.0,
                    max: IR_LENGTH_RANGE.1,
                },
            )
            .with_unit(" %")
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage()),
            bypass: BoolParam::new("Bypass", false)
                .with_value_to_string(formatters::v2s_bool_bypass()),
            slots: core::array::from_fn(|_| SlotParams::default()),
            ir_paths: PlMutex::new(vec![None; NUM_SLOTS]),
            #[cfg(feature = "gui")]
            editor_state: EguiState::from_size(720, 380),
        }
    }
}

impl ReverbPluginParams {
    fn snapshot(&self) -> ReverbParams {
        ReverbParams {
            current_slot: self.current_slot.value() as usize,
            crossfade_ms: self.crossfade_ms.value(),
            predelay_ms: self.predelay_ms.value(),
            dry_db: self.dry_db.value(),
            wet_db: self.wet_db.value(),
            width: self.width.value(),
            low_cut_hz: self.low_cut.value(),
            high_cut_hz: self.high_cut.value(),
            ir_reverse: self.ir_reverse.value(),
            ir_length: self.ir_length.value(),
            bypass: self.bypass.value(),
            slot_enabled: core::array::from_fn(|i| self.slots[i].enabled.value()),
        }
    }
}

/// The reverb plugin instance.
pub struct ReverbPlugin {
    params: Arc<ReverbPluginParams>,
    controller: Arc<Mutex<Option<ReverbController>>>,
    adapter: Option<StereoBlockAdapter>,

    last_slot: i32,
    last_enabled: [bool; NUM_SLOTS],
    last_reverse: bool,
    last_length: f32,

    /// Per-slot waveform previews for the editor, filled off-thread on load.
    #[cfg(feature = "gui")]
    waveforms: Arc<Mutex<Vec<Option<WaveformSummary>>>>,
    /// Bumped on each recognised MIDI event so the editor LED can blink.
    #[cfg(feature = "gui")]
    midi_counter: Arc<AtomicU32>,
}

impl Default for ReverbPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(ReverbPluginParams::default()),
            controller: Arc::new(Mutex::new(None)),
            adapter: None,
            last_slot: 0,
            last_enabled: [true; NUM_SLOTS],
            last_reverse: false,
            last_length: 1.0,
            #[cfg(feature = "gui")]
            waveforms: Arc::new(Mutex::new(vec![None; NUM_SLOTS])),
            #[cfg(feature = "gui")]
            midi_counter: Arc::new(AtomicU32::new(0)),
        }
    }
}

/// Mutable per-frame editor state owned by the egui editor closure.
#[cfg(feature = "gui")]
#[derive(Default)]
struct ReverbEditorState {
    midi: rt_ir_plugin_core::MidiActivity,
    last_midi: u32,
}

/// Fill colour for a slot pad in each visual state.
#[cfg(feature = "gui")]
fn pad_fill(state: PadState, accent: egui::Color32) -> egui::Color32 {
    match state {
        PadState::Active => accent,
        PadState::Filled => egui::Color32::from_rgb(0x23, 0x28, 0x30),
        PadState::Disabled => egui::Color32::from_rgb(0x20, 0x22, 0x26),
        PadState::Empty => egui::Color32::from_rgb(0x1a, 0x1e, 0x24),
    }
}

/// Linear blend between two colours (`t` in `0..=1`).
#[cfg(feature = "gui")]
fn lerp_color(a: egui::Color32, b: egui::Color32, t: f32) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);
    let mix = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    egui::Color32::from_rgb(mix(a.r(), b.r()), mix(a.g(), b.g()), mix(a.b(), b.b()))
}

impl Plugin for ReverbPlugin {
    const NAME: &'static str = "RT IR Reverb";
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
    type BackgroundTask = ReverbTask;

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    #[cfg(feature = "gui")]
    fn editor(&mut self, async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        use egui::{Color32, Sense, Stroke, pos2, vec2};

        // Teal accent (docs/UI-lv2.md § reverb visual identity).
        const ACCENT: Color32 = Color32::from_rgb(0x3a, 0xa3, 0xb5);
        const PANEL: Color32 = Color32::from_rgb(0x1a, 0x1e, 0x24);

        let params = self.params.clone();
        let waveforms = self.waveforms.clone();
        let midi_counter = self.midi_counter.clone();
        let executor = async_executor;

        create_egui_editor(
            self.params.editor_state.clone(),
            ReverbEditorState::default(),
            |_ctx, _state| {},
            move |ctx, setter, state| {
                // Drive the MIDI LED from the shared event counter.
                let count = midi_counter.load(Ordering::Relaxed);
                if count != state.last_midi {
                    state.last_midi = count;
                    state.midi.trigger(true);
                }
                state.midi.decay(0.08);

                egui::CentralPanel::default().show(ctx, |ui| {
                    // ── Header: name · MIDI LED · bypass ──────────────────
                    ui.horizontal(|ui| {
                        ui.heading("RT IR Reverb");
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.add(widgets::ParamSlider::for_param(&params.bypass, setter));
                            let (led, _) = ui.allocate_exact_size(vec2(14.0, 14.0), Sense::hover());
                            let base = match state.midi.status() {
                                MidiStatus::Recognised => Color32::from_rgb(0x4e, 0xc3, 0x6a),
                                MidiStatus::Unrecognised => Color32::from_rgb(0xd9, 0x4a, 0x4a),
                                MidiStatus::Idle => Color32::from_gray(0x30),
                            };
                            let col =
                                lerp_color(Color32::from_gray(0x30), base, state.midi.level());
                            ui.painter().circle_filled(led.center(), 6.0, col);
                            ui.label("MIDI");
                        });
                    });
                    ui.separator();

                    // ── Slot rack: 8 pads ─────────────────────────────────
                    let current = params.current_slot.value();
                    let mut pad_rects = [egui::Rect::NOTHING; NUM_SLOTS];
                    ui.horizontal(|ui| {
                        for (i, rect) in pad_rects.iter_mut().enumerate() {
                            let loaded = params.ir_paths.lock()[i].is_some();
                            let enabled = params.slots[i].enabled.value();
                            let st = pad_state(loaded, enabled, i as i32 == current);
                            let resp = ui.add_sized(
                                vec2(56.0, 44.0),
                                egui::Button::new(format!("{}", i + 1)).fill(pad_fill(st, ACCENT)),
                            );
                            *rect = resp.rect;
                            if resp.clicked() {
                                setter.begin_set_parameter(&params.current_slot);
                                setter.set_parameter(&params.current_slot, i as i32);
                                setter.end_set_parameter(&params.current_slot);
                            }
                        }
                    });

                    // ── Drag-and-drop a file onto a pad → load that slot ──
                    let dropped = ctx.input(|i| i.raw.dropped_files.clone());
                    if !dropped.is_empty()
                        && let Some(pos) = ctx.input(|i| i.pointer.hover_pos())
                        && let Some(slot) = pad_rects.iter().position(|r| r.contains(pos))
                    {
                        for f in dropped {
                            if let Some(path) = f.path {
                                params.ir_paths.lock()[slot] =
                                    Some(path.to_string_lossy().into_owned());
                                executor.execute_background(ReverbTask::Load { slot, path });
                                setter.begin_set_parameter(&params.current_slot);
                                setter.set_parameter(&params.current_slot, slot as i32);
                                setter.end_set_parameter(&params.current_slot);
                            }
                        }
                    }

                    // ── Waveform preview of the active slot ───────────────
                    let (frame, _) =
                        ui.allocate_exact_size(vec2(ui.available_width(), 96.0), Sense::hover());
                    ui.painter().rect_filled(frame, 2.0, PANEL);
                    if let Some(Some(w)) = waveforms.lock().unwrap().get(current as usize) {
                        let n = w.peaks.len().max(1);
                        let cy = frame.center().y;
                        let half = frame.height() * 0.5;
                        for (k, &(lo, hi)) in w.peaks.iter().enumerate() {
                            let x = frame.left() + frame.width() * (k as f32 / n as f32);
                            ui.painter().line_segment(
                                [pos2(x, cy - hi * half), pos2(x, cy - lo * half)],
                                Stroke::new(1.0, ACCENT),
                            );
                        }
                    }
                    ui.separator();

                    // ── Global controls ───────────────────────────────────
                    egui::Grid::new("reverb-controls").show(ui, |ui| {
                        for (label, add) in [
                            ("Pre-Delay", &params.predelay_ms),
                            ("Low Cut", &params.low_cut),
                            ("High Cut", &params.high_cut),
                            ("Width", &params.width),
                            ("Crossfade", &params.crossfade_ms),
                            ("Dry", &params.dry_db),
                            ("Wet", &params.wet_db),
                            ("IR Length", &params.ir_length),
                        ] {
                            ui.label(label);
                            ui.add(widgets::ParamSlider::for_param(add, setter));
                            ui.end_row();
                        }
                        ui.label("IR Reverse");
                        ui.add(widgets::ParamSlider::for_param(&params.ir_reverse, setter));
                        ui.end_row();
                    });
                });
            },
        )
    }

    fn task_executor(&mut self) -> TaskExecutor<Self> {
        let controller = self.controller.clone();
        #[cfg(feature = "gui")]
        let waveforms = self.waveforms.clone();
        Box::new(move |task| {
            let mut guard = controller.lock().unwrap();
            let Some(ctrl) = guard.as_mut() else {
                return;
            };
            match task {
                ReverbTask::Load { slot, path } => {
                    if let Err(e) = ctrl.load_slot_from_path(slot, &path) {
                        nih_error!("failed to load IR into slot {slot}: {e}");
                    }
                    ctrl.activate(slot);
                    #[cfg(feature = "gui")]
                    {
                        waveforms.lock().unwrap()[slot] = ctrl.waveform(slot, WAVEFORM_BUCKETS);
                    }
                }
                ReverbTask::Activate(slot) => ctrl.activate(slot),
                ReverbTask::Midi(event) => {
                    ctrl.handle_midi(event);
                }
                ReverbTask::SetEnabled { slot, enabled } => ctrl.set_slot_enabled(slot, enabled),
                ReverbTask::SetReverse(on) => ctrl.set_reverse(on),
                ReverbTask::SetLength(frac) => ctrl.set_length(frac),
            }
        })
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        let sample_rate = buffer_config.sample_rate as u32;
        let snapshot = self.params.snapshot();
        let (mut controller, processor) = ReverbController::new(BLOCK_SIZE, sample_rate, &snapshot);

        {
            let paths = self.params.ir_paths.lock();
            for (slot, path) in paths.iter().enumerate() {
                if let Some(path) = path
                    && let Err(e) = controller.load_slot_from_path(slot, path)
                {
                    nih_error!("failed to restore IR for slot {slot}: {e}");
                }
            }
        }
        for (i, slot) in self.params.slots.iter().enumerate() {
            controller.set_slot_enabled(i, slot.enabled.value());
        }
        controller.set_reverse(self.params.ir_reverse.value());
        controller.set_length(self.params.ir_length.value());
        controller.activate(self.params.current_slot.value() as usize);

        // Pre-compute waveform previews for the restored slots.
        #[cfg(feature = "gui")]
        {
            let mut wf = self.waveforms.lock().unwrap();
            for (i, slot) in wf.iter_mut().enumerate() {
                *slot = controller.waveform(i, WAVEFORM_BUCKETS);
            }
        }

        let adapter = StereoBlockAdapter::new(processor);
        context.set_latency_samples(adapter.latency_samples() as u32);

        *self.controller.lock().unwrap() = Some(controller);
        self.adapter = Some(adapter);

        self.last_slot = self.params.current_slot.value();
        self.last_enabled = core::array::from_fn(|i| self.params.slots[i].enabled.value());
        self.last_reverse = self.params.ir_reverse.value();
        self.last_length = self.params.ir_length.value();
        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // MIDI → slot activation (off-thread).
        while let Some(event) = context.next_event() {
            match event {
                NoteEvent::MidiProgramChange { program, .. } => {
                    context.execute_background(ReverbTask::Midi(SlotEvent::ProgramChange(program)));
                    #[cfg(feature = "gui")]
                    self.midi_counter.fetch_add(1, Ordering::Relaxed);
                }
                NoteEvent::MidiCC { cc, value, .. } => {
                    let raw = (value * 127.0).round().clamp(0.0, 127.0) as u8;
                    context.execute_background(ReverbTask::Midi(SlotEvent::ControlChange {
                        controller: cc,
                        value: raw,
                    }));
                    #[cfg(feature = "gui")]
                    self.midi_counter.fetch_add(1, Ordering::Relaxed);
                }
                _ => {}
            }
        }

        // IR-shaping changes rebuild the current slot off-thread.
        let current = self.params.current_slot.value();
        let reverse = self.params.ir_reverse.value();
        if reverse != self.last_reverse {
            self.last_reverse = reverse;
            context.execute_background(ReverbTask::SetReverse(reverse));
            context.execute_background(ReverbTask::Activate(current as usize));
        }
        let length = self.params.ir_length.value();
        if (length - self.last_length).abs() > f32::EPSILON {
            self.last_length = length;
            context.execute_background(ReverbTask::SetLength(length));
            context.execute_background(ReverbTask::Activate(current as usize));
        }
        for i in 0..NUM_SLOTS {
            let enabled = self.params.slots[i].enabled.value();
            if enabled != self.last_enabled[i] {
                self.last_enabled[i] = enabled;
                context.execute_background(ReverbTask::SetEnabled { slot: i, enabled });
                if i == current as usize {
                    context.execute_background(ReverbTask::Activate(i));
                }
            }
        }
        if current != self.last_slot {
            self.last_slot = current;
            context.execute_background(ReverbTask::Activate(current as usize));
        }

        // Apply smoothed globals, then render (allocation-free).
        let snapshot = self.params.snapshot();
        if let Some(adapter) = self.adapter.as_mut() {
            adapter.processor_mut().apply_params(&snapshot);
            adapter.process_buffer(buffer.as_slice());
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for ReverbPlugin {
    const CLAP_ID: &'static str = "org.codeberg.slundi.rt-ir-reverb";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("True-stereo convolution reverb");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Reverb,
    ];
}

impl Vst3Plugin for ReverbPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"RtIrReverbSlundi";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Reverb];
}

nih_export_clap!(ReverbPlugin);
nih_export_vst3!(ReverbPlugin);
