//! The framework-agnostic reverb engine: a bank of pre-built FDN voices, a
//! click-free preset crossfade, and the wet bus around them.
//!
//! ```text
//!                          ┌──────────── voice[space] : Fdn<8> ───────────┐
//! in ─┬─▶ pre-delay ──────▶│  active  ──▶┐                                │
//!     │                    │  previous ─▶┤ equal-power blend ─▶ wet       │
//!     │                    └─────────────┘   (during crossfade)           │
//!     │                                       │                           │
//!     │                            low-cut ─▶ high-cut ─▶ width ─▶ wet·gain
//!     └───────────────────────────────────────────────────────▶ dry·gain
//!                                                            out = dry + wet
//! ```
//!
//! # Why pre-built voices
//!
//! One [`Fdn`] is constructed per built-in space at init (eight of them).  A
//! preset slot just *references* a space index, so re-assigning a slot's space
//! or switching the active slot never rebuilds anything — there is no worker
//! thread and nothing on the audio thread ever allocates.  Switching the active
//! space equal-power **crossfades** the outgoing voice's tail into the freshly
//! re-primed incoming voice using a per-sample rotation recurrence (no `sin`/
//! `cos` in the loop), so a live preset change never clicks.
//!
//! # Real-time safety
//!
//! [`ReverbEngine::process`] / [`ReverbEngine::process_sample`] and
//! [`ReverbEngine::apply_params`] allocate nothing, lock nothing, and never
//! panic in release.  Build the engine off the audio thread with
//! [`ReverbEngine::new`].

use std::f32::consts::FRAC_PI_2;

use rt_fdn_core::{Fdn, PRESETS};
use rt_ir_plugin_core::{BUTTERWORTH_Q, Biquad, ms_to_samples};

use crate::dsp::{PreDelay, fader_gain, width_sample};
use crate::params::{
    MOD_DEPTH_MAX_SAMPLES, NUM_SPACES, PREDELAY_MS_RANGE, ReverbParams, damping_cutoff_hz,
    diffusion_gain, mod_depth_samples,
};

/// The FDN branch count for every voice (Phase 0 decision: 8×8).
const N: usize = 8;

/// One reverb instance: the voice bank, crossfade state, and wet bus.
#[derive(Debug)]
pub struct ReverbEngine {
    sample_rate: u32,
    /// One voice per built-in space, indexed by space index.
    voices: [Fdn<N>; NUM_SPACES],

    /// Space index of the voice currently fading in / steady.
    active: usize,
    /// Space index of the voice fading out (valid only while `fading`).
    previous: usize,
    /// `true` while a preset crossfade is in progress.
    fading: bool,
    /// Total / elapsed crossfade length in samples.
    fade_len: usize,
    fade_pos: usize,
    /// Rotation-recurrence state: `(c, s) = (cos θ, sin θ)`, `θ: 0 → π/2`.
    /// `c` is the outgoing gain (`1 → 0`), `s` the incoming gain (`0 → 1`).
    cos_step: f32,
    sin_step: f32,
    cos_theta: f32,
    sin_theta: f32,

    /// Whether the active slot is enabled (a disabled slot renders dry).
    wet_enabled: bool,
    /// Cached crossfade length for the *next* switch, in samples.
    crossfade_samples: usize,

    // Wet bus.
    predelay_l: PreDelay,
    predelay_r: PreDelay,
    low_cut_l: Biquad,
    low_cut_r: Biquad,
    high_cut_l: Biquad,
    high_cut_r: Biquad,
    width: f32,
    dry_gain: f32,
    wet_gain: f32,
    bypass: bool,
}

impl ReverbEngine {
    /// Build the engine at `sample_rate` from `params`.
    ///
    /// **Allocates** (the eight voices + pre-delay buffers) — call off the
    /// audio thread.
    pub fn new(sample_rate: u32, params: &ReverbParams) -> Self {
        let params = params.clamped();
        let sr = sample_rate.max(1);
        let sr_f = sr as f32;

        // Every voice is built with full modulation headroom so the global
        // `Modulation Depth` knob has its whole range regardless of the space's
        // own baked depth.
        let voices: [Fdn<N>; NUM_SPACES] = core::array::from_fn(|space| {
            let mut voice_params = PRESETS[space].params;
            voice_params.mod_depth = MOD_DEPTH_MAX_SAMPLES;
            Fdn::<N>::new(sr_f, voice_params)
        });

        let max_predelay = ms_to_samples(PREDELAY_MS_RANGE.1, sr);
        let mut engine = Self {
            sample_rate: sr,
            voices,
            active: params.active_space(),
            previous: 0,
            fading: false,
            fade_len: 1,
            fade_pos: 0,
            cos_step: 1.0,
            sin_step: 0.0,
            cos_theta: 1.0,
            sin_theta: 0.0,
            wet_enabled: params.active_enabled(),
            crossfade_samples: ms_to_samples(params.crossfade_ms, sr),
            predelay_l: PreDelay::new(max_predelay),
            predelay_r: PreDelay::new(max_predelay),
            low_cut_l: Biquad::high_pass(sr, params.low_cut_hz, BUTTERWORTH_Q),
            low_cut_r: Biquad::high_pass(sr, params.low_cut_hz, BUTTERWORTH_Q),
            high_cut_l: Biquad::low_pass(sr, params.high_cut_hz, BUTTERWORTH_Q),
            high_cut_r: Biquad::low_pass(sr, params.high_cut_hz, BUTTERWORTH_Q),
            width: params.width,
            dry_gain: fader_gain(params.dry_db),
            wet_gain: fader_gain(params.wet_db),
            bypass: params.bypass,
        };
        // Configure the wet bus + seed the active voice's live parameters.
        // `active` already matches the target space, so this starts no fade.
        engine.apply_params(&params);
        engine
    }

    /// The space index currently rendering (fading-in / steady voice).
    #[inline]
    pub fn active_space(&self) -> usize {
        self.active
    }

    /// Whether a preset crossfade is in progress.
    #[inline]
    pub fn is_crossfading(&self) -> bool {
        self.fading
    }

    /// Number of FDN voices being rendered this sample (1, or 2 mid-crossfade).
    /// A coarse CPU-load proxy for the read-only meter.
    #[inline]
    pub fn active_voice_count(&self) -> usize {
        if self.fading { 2 } else { 1 }
    }

    /// Push the global knob values into the active voice + wet bus.
    ///
    /// **Real-time safe** — recomputes scalar coefficients only.
    pub fn apply_params(&mut self, params: &ReverbParams) {
        let params = params.clamped();

        // Crossfade length for the *next* switch must be set before we test for
        // a space change this block.
        self.crossfade_samples = ms_to_samples(params.crossfade_ms, self.sample_rate);
        self.wet_enabled = params.active_enabled();

        // A change of active space starts an equal-power crossfade.
        let target = params.active_space();
        if target != self.active {
            self.start_fade(target);
        }

        self.apply_engine_knobs(&params);

        // Wet bus.
        let predelay = ms_to_samples(params.predelay_ms, self.sample_rate);
        self.predelay_l.set_delay(predelay);
        self.predelay_r.set_delay(predelay);
        self.low_cut_l
            .set_high_pass(self.sample_rate, params.low_cut_hz, BUTTERWORTH_Q);
        self.low_cut_r
            .set_high_pass(self.sample_rate, params.low_cut_hz, BUTTERWORTH_Q);
        self.high_cut_l
            .set_low_pass(self.sample_rate, params.high_cut_hz, BUTTERWORTH_Q);
        self.high_cut_r
            .set_low_pass(self.sample_rate, params.high_cut_hz, BUTTERWORTH_Q);
        self.width = params.width;
        self.dry_gain = fader_gain(params.dry_db);
        self.wet_gain = fader_gain(params.wet_db);
        self.bypass = params.bypass;
    }

    /// Push the FDN-internal knobs onto the live voices (active, and the
    /// outgoing one while fading, so a knob move stays consistent mid-fade).
    fn apply_engine_knobs(&mut self, params: &ReverbParams) {
        let damping_hz = damping_cutoff_hz(params.damping);
        let diffusion = diffusion_gain(params.diffusion);
        let depth = mod_depth_samples(params.mod_depth);

        let set = |voice: &mut Fdn<N>| {
            voice.set_decay(params.decay_s);
            voice.set_size(params.size);
            voice.set_damping(damping_hz);
            voice.set_diffusion(diffusion);
            voice.set_modulation(depth, params.mod_rate_hz);
            voice.set_freeze(params.freeze);
        };
        set(&mut self.voices[self.active]);
        if self.fading {
            let previous = self.previous;
            set(&mut self.voices[previous]);
        }
    }

    /// Begin an equal-power crossfade to the voice for `target`.
    fn start_fade(&mut self, target: usize) {
        if target == self.active {
            return;
        }
        // Abandon a fade already in progress: retire its outgoing voice.
        if self.fading {
            self.voices[self.previous].reset();
        }
        self.previous = self.active;
        self.active = target;
        // Re-prime the incoming voice so its tail builds from the new input
        // instead of resuming stale state (which could click).
        self.voices[target].reset();

        self.fade_len = self.crossfade_samples.max(1);
        self.fade_pos = 0;
        let step = FRAC_PI_2 / self.fade_len as f32;
        self.cos_step = step.cos();
        self.sin_step = step.sin();
        self.cos_theta = 1.0; // outgoing gain starts at 1
        self.sin_theta = 0.0; // incoming gain starts at 0
        self.fading = true;
    }

    /// Clear all voices and the wet bus to silence.
    pub fn reset(&mut self) {
        for voice in &mut self.voices {
            voice.reset();
        }
        self.predelay_l.reset();
        self.predelay_r.reset();
        self.low_cut_l.reset();
        self.low_cut_r.reset();
        self.high_cut_l.reset();
        self.high_cut_r.reset();
        self.fading = false;
    }

    /// Render one stereo sample.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn process_sample(&mut self, in_l: f32, in_r: f32) -> (f32, f32) {
        if self.bypass {
            return (in_l, in_r);
        }

        // Pre-delay feeds the wet path only; the dry path is zero-latency.
        let pre_l = self.predelay_l.process_sample(in_l);
        let pre_r = self.predelay_r.process_sample(in_r);

        let (mut wet_l, mut wet_r) = if self.wet_enabled {
            self.render_voices(pre_l, pre_r)
        } else {
            (0.0, 0.0)
        };

        // Wet-only EQ: low-cut (HPF) then high-cut (LPF), per channel.
        wet_l = self
            .high_cut_l
            .process_sample(self.low_cut_l.process_sample(wet_l));
        wet_r = self
            .high_cut_r
            .process_sample(self.low_cut_r.process_sample(wet_r));
        // Stereo width (mid/side).
        let (wet_l, wet_r) = width_sample(wet_l, wet_r, self.width);

        (
            self.dry_gain * in_l + self.wet_gain * wet_l,
            self.dry_gain * in_r + self.wet_gain * wet_r,
        )
    }

    /// Render the FDN voice(s), blending during a crossfade.
    #[inline]
    fn render_voices(&mut self, pre_l: f32, pre_r: f32) -> (f32, f32) {
        let (new_l, new_r) = self.voices[self.active].process_sample_stereo(pre_l, pre_r);
        if !self.fading {
            return (new_l, new_r);
        }

        let (old_l, old_r) = self.voices[self.previous].process_sample_stereo(pre_l, pre_r);
        let (out_gain, in_gain) = (self.cos_theta, self.sin_theta);

        // Rotate (cos θ, sin θ) forward one step.
        let (c, s) = (self.cos_theta, self.sin_theta);
        self.cos_theta = c * self.cos_step - s * self.sin_step;
        self.sin_theta = s * self.cos_step + c * self.sin_step;
        self.fade_pos += 1;
        if self.fade_pos >= self.fade_len {
            self.fading = false;
            self.voices[self.previous].reset();
        }

        (
            out_gain * old_l + in_gain * new_l,
            out_gain * old_r + in_gain * new_r,
        )
    }

    /// Render a stereo block in place.  All four slices must be equal length.
    ///
    /// **Real-time safe.**
    pub fn process(&mut self, in_l: &[f32], in_r: &[f32], out_l: &mut [f32], out_r: &mut [f32]) {
        let n = in_l.len().min(in_r.len()).min(out_l.len()).min(out_r.len());
        for i in 0..n {
            let (l, r) = self.process_sample(in_l[i], in_r[i]);
            out_l[i] = l;
            out_r[i] = r;
        }
    }
}
