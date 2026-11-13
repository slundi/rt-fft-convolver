//! The framework-agnostic instrument engine: two pre-built polyphonic voice
//! banks (one per model family), body stereoisation, and the global gain / mix
//! / bypass around them.
//!
//! ```text
//!  MIDI note ─▶ active voice bank ─▶ ×gain ─┬─▶ body filter L ─▶ wet L ─┐
//!  (Plucked or Modal)                       └─▶ body filter R ─▶ wet R ─┤
//!  audio in ────────────────────────────────────────────────dry──────▶ mix ─▶ out
//! ```
//!
//! # Why two pre-built banks
//!
//! Switching model family must not allocate on the audio thread, and a
//! [`VoiceManager`] owns its voices' DSP buffers for life.  So both the plucked
//! and modal banks are built once at construction; the `Model` selector only
//! chooses which one renders.  Leaving a family resets its bank so a stale tail
//! can't reappear when you switch back.
//!
//! # Body stereoisation
//!
//! A single mono voice sum is coloured by two body-resonance peaking filters
//! whose centres are spread ±[`BODY_STEREO_SPREAD`], decorrelating the two
//! channels into a natural stereo image (Phase 2 "stereo out via body / bridge
//! stereoisation").  With `BodyType::None` the output is mono.
//!
//! # Real-time safety
//!
//! [`process_sample`](PhysicalEngine::process_sample),
//! [`note_on`](PhysicalEngine::note_on), [`note_off`](PhysicalEngine::note_off)
//! and [`apply_params`](PhysicalEngine::apply_params) allocate nothing, lock
//! nothing and do not panic in release.  Build the engine off the audio thread
//! with [`PhysicalEngine::new`].

use rt_physical_core::{Biquad, Model, PhysicalVoices};

use crate::params::{BodyType, InstrumentParams};

/// Fractional centre-frequency spread between the left and right body filters.
const BODY_STEREO_SPREAD: f32 = 0.03;

/// One instrument instance: both voice banks, the body stereoiser and the
/// global bus.
#[derive(Debug)]
pub struct PhysicalEngine {
    sample_rate: f32,
    /// Extended Karplus-Strong voices.
    plucked: PhysicalVoices,
    /// Modal-resonator voices.
    modal: PhysicalVoices,
    /// Family currently rendering.
    model: Model,

    /// Left/right body-resonance colour (valid only when `body != None`).
    body_l: Biquad,
    body_r: Biquad,
    body: BodyType,

    gain: f32,
    mix: f32,
    bypass: bool,
}

impl PhysicalEngine {
    /// Build the engine at `sample_rate` from `params`.
    ///
    /// **Allocates** (both voice banks) — call off the audio thread.
    pub fn new(sample_rate: f32, params: &InstrumentParams) -> Self {
        let params = params.clamped();
        let sr = sample_rate.max(1.0);

        // Build each bank locked to its own family regardless of the current
        // selection (the core `set_params` later preserves the built family).
        let mut plucked_core = params.core_params();
        plucked_core.model = Model::Plucked;
        let mut modal_core = params.core_params();
        modal_core.model = Model::Modal;

        let mut engine = Self {
            sample_rate: sr,
            plucked: PhysicalVoices::new(sr, plucked_core),
            modal: PhysicalVoices::new(sr, modal_core),
            model: params.model,
            body_l: Biquad::default(),
            body_r: Biquad::default(),
            body: BodyType::None,
            gain: params.gain(),
            mix: params.mix,
            bypass: params.bypass,
        };
        engine.rebuild_body(params.body);
        engine.apply_params(&params);
        engine
    }

    /// Mutable reference to the bank of the currently selected family.
    #[inline]
    fn active_bank(&mut self) -> &mut PhysicalVoices {
        match self.model {
            Model::Plucked => &mut self.plucked,
            Model::Modal => &mut self.modal,
        }
    }

    /// Number of voices currently sounding in the active bank (CPU-meter proxy).
    #[inline]
    pub fn active_voice_count(&self) -> usize {
        match self.model {
            Model::Plucked => self.plucked.active_count(),
            Model::Modal => self.modal.active_count(),
        }
    }

    /// The family currently rendering.
    #[inline]
    pub fn model(&self) -> Model {
        self.model
    }

    /// Rebuild the two body filters for `body` (fresh state — done only on a
    /// discrete body-type change, so no per-block click).
    fn rebuild_body(&mut self, body: BodyType) {
        self.body = body;
        if let Some((freq, q, gain_db)) = body.peak() {
            let nyquist = self.sample_rate * 0.5;
            let left = (freq * (1.0 - BODY_STEREO_SPREAD)).min(nyquist * 0.99);
            let right = (freq * (1.0 + BODY_STEREO_SPREAD)).min(nyquist * 0.99);
            self.body_l = Biquad::peaking(left, q, gain_db, self.sample_rate);
            self.body_r = Biquad::peaking(right, q, gain_db, self.sample_rate);
        }
    }

    /// Push knob values into both banks and the bus.
    ///
    /// **Real-time safe** — scalar coefficient updates only (body filters are
    /// rebuilt only when the body type actually changes).
    pub fn apply_params(&mut self, params: &InstrumentParams) {
        let params = params.clamped();

        // A model change silences the family we are leaving so its tail cannot
        // reappear on a later switch back, and clears the shared body filters so
        // the old family's coloured ring-out does not bleed into the new one.
        if params.model != self.model {
            self.active_bank().reset();
            self.body_l.reset();
            self.body_r.reset();
            self.model = params.model;
        }

        let core = params.core_params();
        self.plucked.set_params(core);
        self.modal.set_params(core);
        self.plucked.set_polyphony(params.polyphony);
        self.modal.set_polyphony(params.polyphony);

        if params.body != self.body {
            self.rebuild_body(params.body);
        }

        self.gain = params.gain();
        self.mix = params.mix;
        self.bypass = params.bypass;
    }

    /// Trigger a note (velocity `0..=1`) on the active family.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn note_on(&mut self, note: u8, velocity: f32) {
        self.active_bank().note_on(note, velocity);
    }

    /// Release a note on the active family.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn note_off(&mut self, note: u8) {
        self.active_bank().note_off(note);
    }

    /// Silence both banks and the body filters.
    pub fn reset(&mut self) {
        self.plucked.reset();
        self.modal.reset();
        self.body_l.reset();
        self.body_r.reset();
    }

    /// Render one stereo sample given the (effect-mode) audio input.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn process_sample(&mut self, in_l: f32, in_r: f32) -> (f32, f32) {
        if self.bypass {
            return (in_l, in_r);
        }

        let wet_mono = self.active_bank().process() * self.gain;
        let (wet_l, wet_r) = if self.body == BodyType::None {
            (wet_mono, wet_mono)
        } else {
            (self.body_l.process(wet_mono), self.body_r.process(wet_mono))
        };

        let dry_gain = 1.0 - self.mix;
        let wet_gain = self.mix;
        (
            dry_gain * in_l + wet_gain * wet_l,
            dry_gain * in_r + wet_gain * wet_r,
        )
    }

    /// Render a stereo block in place.  All four slices must be equal length.
    ///
    /// **Real-time safe.**
    pub fn process(&mut self, in_l: &[f32], in_r: &[f32], out_l: &mut [f32], out_r: &mut [f32]) {
        let count = in_l.len().min(in_r.len()).min(out_l.len()).min(out_r.len());
        for index in 0..count {
            let (left, right) = self.process_sample(in_l[index], in_r[index]);
            out_l[index] = left;
            out_r[index] = right;
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rt_physical_core::Model;

    const SR: f32 = 48_000.0;

    fn engine(params: InstrumentParams) -> PhysicalEngine {
        PhysicalEngine::new(SR, &params)
    }

    /// Peak absolute output over `samples`, feeding silence in (synth use).
    fn peak(engine: &mut PhysicalEngine, samples: usize) -> f32 {
        let mut peak = 0.0f32;
        for _ in 0..samples {
            let (left, right) = engine.process_sample(0.0, 0.0);
            peak = peak.max(left.abs()).max(right.abs());
        }
        peak
    }

    #[test]
    fn plucked_note_on_produces_sound() {
        let mut engine = engine(InstrumentParams {
            model: Model::Plucked,
            body: BodyType::None,
            ..InstrumentParams::default()
        });
        engine.note_on(69, 1.0);
        assert!(peak(&mut engine, 4_000) > 1e-3);
        assert_eq!(engine.active_voice_count(), 1);
    }

    #[test]
    fn modal_note_on_produces_sound() {
        let mut engine = engine(InstrumentParams {
            model: Model::Modal,
            body: BodyType::None,
            ..InstrumentParams::default()
        });
        engine.note_on(60, 1.0);
        assert!(peak(&mut engine, 4_000) > 1e-3);
    }

    #[test]
    fn bypass_passes_input_untouched() {
        let mut engine = engine(InstrumentParams {
            bypass: true,
            ..InstrumentParams::default()
        });
        engine.note_on(69, 1.0);
        let (left, right) = engine.process_sample(0.4, -0.6);
        assert_eq!((left, right), (0.4, -0.6));
    }

    #[test]
    fn mix_zero_is_dry_input_only() {
        let mut engine = engine(InstrumentParams {
            mix: 0.0,
            ..InstrumentParams::default()
        });
        engine.note_on(69, 1.0);
        // Wet is fully attenuated: output equals the dry input regardless of voices.
        for _ in 0..1_000 {
            let (left, right) = engine.process_sample(0.2, 0.2);
            assert!((left - 0.2).abs() < 1e-6 && (right - 0.2).abs() < 1e-6);
        }
    }

    #[test]
    fn gain_scales_output() {
        let loud_peak = {
            let mut loud = engine(InstrumentParams {
                gain_db: 6.0,
                body: BodyType::None,
                ..InstrumentParams::default()
            });
            loud.note_on(69, 1.0);
            peak(&mut loud, 4_000)
        };
        let quiet_peak = {
            let mut quiet = engine(InstrumentParams {
                gain_db: -6.0,
                body: BodyType::None,
                ..InstrumentParams::default()
            });
            quiet.note_on(69, 1.0);
            peak(&mut quiet, 4_000)
        };
        assert!(loud_peak > quiet_peak * 2.0, "gain did not scale output");
    }

    #[test]
    fn body_produces_stereo_decorrelation() {
        let mut engine = engine(InstrumentParams {
            model: Model::Plucked,
            body: BodyType::Dreadnought,
            ..InstrumentParams::default()
        });
        engine.note_on(45, 1.0); // low note near the body resonance
        let mut max_diff = 0.0f32;
        for _ in 0..8_000 {
            let (left, right) = engine.process_sample(0.0, 0.0);
            max_diff = max_diff.max((left - right).abs());
        }
        assert!(max_diff > 1e-4, "body filters did not decorrelate channels");
    }

    #[test]
    fn model_switch_silences_old_family() {
        let mut engine = engine(InstrumentParams {
            model: Model::Plucked,
            ..InstrumentParams::default()
        });
        engine.note_on(69, 1.0);
        assert!(peak(&mut engine, 2_000) > 1e-3);

        // Switch to modal: the plucked tail must not bleed through, and no
        // modal note has been triggered yet → silence.
        engine.apply_params(&InstrumentParams {
            model: Model::Modal,
            ..InstrumentParams::default()
        });
        assert!(
            peak(&mut engine, 2_000) < 1e-4,
            "old family bled after switch"
        );

        // A modal note now sounds.
        engine.note_on(60, 1.0);
        assert!(peak(&mut engine, 4_000) > 1e-3);
    }

    #[test]
    fn polyphony_limit_caps_active_voices() {
        let mut engine = engine(InstrumentParams {
            model: Model::Plucked,
            polyphony: 4,
            ..InstrumentParams::default()
        });
        for note in 50..62 {
            engine.note_on(note, 1.0);
            engine.process_sample(0.0, 0.0);
        }
        peak(&mut engine, 1_000);
        assert!(engine.active_voice_count() <= 4, "polyphony cap exceeded");
    }

    #[test]
    fn reset_silences_engine() {
        let mut engine = engine(InstrumentParams::default());
        for note in 60..68 {
            engine.note_on(note, 1.0);
        }
        peak(&mut engine, 500);
        engine.reset();
        assert!(peak(&mut engine, 256) < 1e-6);
    }

    #[test]
    fn output_is_finite_under_load() {
        let mut engine = engine(InstrumentParams {
            model: Model::Modal,
            decay_s: 30.0,
            ..InstrumentParams::default()
        });
        for note in 36..90 {
            engine.note_on(note, 1.0);
            let (left, right) = engine.process_sample(0.1, -0.1);
            assert!(left.is_finite() && right.is_finite());
        }
        for _ in 0..48_000 {
            let (left, right) = engine.process_sample(0.0, 0.0);
            assert!(left.is_finite() && right.is_finite());
        }
    }

    #[test]
    fn detune_is_applied_to_new_notes() {
        // A +100-cent detune (one semitone, the range maximum) raises the modal
        // pitch by 2^(1/12) ≈ 1.0595. We compare zero-crossing counts of a
        // single-mode voice over a long window for fine resolution.
        let crossings = |cents: f32| {
            let mut engine = engine(InstrumentParams {
                model: Model::Modal,
                num_modes: 1,
                body: BodyType::None,
                detune_cents: cents,
                ..InstrumentParams::default()
            });
            engine.note_on(69, 1.0); // A4, plenty of cycles for resolution
            let mut prev = 0.0;
            let mut count = 0u32;
            for _ in 0..SR as usize / 2 {
                let (left, _) = engine.process_sample(0.0, 0.0);
                if prev <= 0.0 && left > 0.0 {
                    count += 1;
                }
                prev = left;
            }
            count
        };
        let ratio = crossings(100.0) as f32 / crossings(0.0).max(1) as f32;
        let expected = 2.0f32.powf(1.0 / 12.0);
        assert!(
            (ratio - expected).abs() < 0.02,
            "semitone detune ratio {ratio}"
        );
    }
}
