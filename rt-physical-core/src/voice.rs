//! A single polyphonic voice: one model kernel (plucked string *or* modal bank)
//! plus the note bookkeeping the [`crate::manager::VoiceManager`] needs to
//! allocate and steal voices.
//!
//! # One family per manager
//!
//! Phase 0 chose a single active model family at a time (the plugin's model
//! dropdown), so a voice is built for one [`Model`] and keeps it.  Switching
//! family rebuilds the manager off the render thread; it is not a per-note
//! operation.
//!
//! # Click-free voice stealing
//!
//! When the manager reuses a still-ringing voice it calls
//! [`trigger`](Voice::trigger), which resets the kernel (so no energy leaks
//! between notes) and starts a short output fade-in.  The excitation itself
//! begins from silence, so the reset + fade removes the discontinuity that a
//! hard cut would click on — the ~2 ms fade named in the Phase 0 polyphony
//! decision.
//!
//! # Real-time safety
//!
//! [`process`](Voice::process) is the kernel step plus a fade multiply and a
//! peak-follower update — no allocation, no locking, no `panic`.  All buffers
//! are allocated when the voice is constructed.

use crate::excitation::Excitation;
use crate::modal::{ModalBank, ModePreset};
use crate::pitch::midi_to_freq;
use crate::string::KarplusStrong;

/// Output fade-in applied on (re)trigger, in seconds — the anti-click ramp.
const FADE_SECONDS: f32 = 0.002;

/// Peak-follower release time constant, in seconds.  Governs how quickly a
/// decayed voice reports itself quiet (and thus reusable).
const ENV_RELEASE_SECONDS: f32 = 0.1;

/// The model family a voice runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Model {
    /// Extended Karplus-Strong plucked string.
    #[default]
    Plucked,
    /// Modal (biquad resonator) bank.
    Modal,
}

/// Per-note parameters shared by every voice in a manager.  Cheap to `Copy` so
/// the manager can hand a snapshot to each triggered voice.
#[derive(Debug, Clone, Copy)]
pub struct PhysicalParams {
    /// Active model family (fixed for a given manager).
    pub model: Model,
    /// Decay time (T60, seconds) for the plucked string and the modal
    /// fundamental.
    pub t60: f32,

    // Plucked-string controls.
    /// String tone: 0 = bright, 1 = dark.
    pub damping: f32,
    /// Pick hardness: 0 = soft, 1 = hard.
    pub pick_hardness: f32,
    /// Pluck position along the string, `(0, 1)`.
    pub pluck_position: f32,

    // Modal controls.
    /// Material timbre preset.
    pub preset: ModePreset,
    /// Number of modes to sound.
    pub num_modes: usize,
    /// Higher modes decay faster (`> 1`) or slower (`< 1`) than the fundamental.
    pub decay_slope: f32,
}

impl Default for PhysicalParams {
    fn default() -> Self {
        Self {
            model: Model::Plucked,
            t60: 3.0,
            damping: 0.3,
            pick_hardness: 0.6,
            pluck_position: 0.25,
            preset: ModePreset::Metal,
            num_modes: 5,
            decay_slope: 1.0,
        }
    }
}

/// The DSP kernel behind a voice — one variant per v1 model family.
#[derive(Debug, Clone)]
enum Kernel {
    Plucked {
        string: KarplusStrong,
        exc: Excitation,
    },
    Modal(ModalBank),
}

/// One polyphonic voice.
#[derive(Debug, Clone)]
pub struct Voice {
    kernel: Kernel,
    /// Peak follower giving the voice's current level (for stealing decisions).
    env: f32,
    /// Per-sample multiplier of the peak follower's release.
    env_release: f32,
    /// Current output fade-in gain, ramping `0 → 1` after a trigger.
    fade: f32,
    /// Per-sample fade-in increment.
    fade_inc: f32,
    /// `true` while the note is held (before note-off).
    held: bool,
    /// MIDI note currently assigned.
    note: u8,
    /// Monotonic trigger serial — smaller means older (for oldest tie-break).
    serial: u64,
}

impl Voice {
    /// Build an idle voice for `model`.  `max_loop_len` sizes the plucked
    /// string's delay line / comb for the lowest supported note.
    ///
    /// **Allocates** — construct off the render thread.
    pub fn new(sample_rate: f32, model: Model, max_loop_len: usize) -> Self {
        let kernel = match model {
            Model::Plucked => Kernel::Plucked {
                string: KarplusStrong::new(sample_rate, max_loop_len),
                exc: Excitation::new(max_loop_len),
            },
            Model::Modal => Kernel::Modal(ModalBank::new(sample_rate)),
        };
        Self {
            kernel,
            env: 0.0,
            env_release: (-1.0 / (ENV_RELEASE_SECONDS * sample_rate)).exp(),
            fade: 1.0,
            fade_inc: 1.0 / (FADE_SECONDS * sample_rate).max(1.0),
            held: false,
            note: 0,
            serial: 0,
        }
    }

    /// (Re)start this voice on `note` at `velocity ∈ [0, 1]`, tagging it with a
    /// monotonic `serial`.  Resets kernel state (no leak) and begins the
    /// anti-click fade-in.
    ///
    /// **Real-time safe** apart from the per-note kernel retune.
    pub fn trigger(
        &mut self,
        note: u8,
        velocity: f32,
        serial: u64,
        params: &PhysicalParams,
        sample_rate: f32,
    ) {
        self.note = note;
        self.serial = serial;
        self.held = true;
        self.env = 0.0;
        self.fade = 0.0;
        let freq = midi_to_freq(note);

        match &mut self.kernel {
            Kernel::Plucked { string, exc } => {
                string.reset();
                string.set_note(freq, params.damping, params.t60);
                let seed = seed_for(note, serial);
                let loop_len = sample_rate / freq;
                exc.trigger(
                    velocity,
                    params.pick_hardness,
                    params.pluck_position,
                    loop_len,
                    seed,
                );
            }
            Kernel::Modal(bank) => {
                bank.reset();
                bank.set_voice(
                    freq,
                    params.preset,
                    params.num_modes,
                    params.t60,
                    params.decay_slope,
                );
                bank.excite(velocity);
            }
        }
    }

    /// Release the note: it is no longer held and may decay faster.
    ///
    /// **Real-time safe.**
    pub fn release(&mut self) {
        self.held = false;
        if let Kernel::Plucked { string, .. } = &mut self.kernel {
            string.note_off();
        }
    }

    /// The MIDI note currently assigned.
    #[inline]
    pub fn note(&self) -> u8 {
        self.note
    }

    /// Whether the note is still held (pre note-off).
    #[inline]
    pub fn is_held(&self) -> bool {
        self.held
    }

    /// Trigger serial (smaller = older).
    #[inline]
    pub fn serial(&self) -> u64 {
        self.serial
    }

    /// Current output level from the peak follower.
    #[inline]
    pub fn level(&self) -> f32 {
        self.env
    }

    /// Whether the voice is silent and not held — i.e. freely reusable.
    #[inline]
    pub fn is_available(&self, silence: f32) -> bool {
        !self.held && self.env < silence
    }

    /// Clear all state and silence the voice.
    pub fn reset(&mut self) {
        match &mut self.kernel {
            Kernel::Plucked { string, exc } => {
                string.reset();
                exc.silence();
            }
            Kernel::Modal(bank) => bank.reset(),
        }
        self.env = 0.0;
        self.held = false;
        self.fade = 1.0;
    }

    /// Advance the voice by one sample.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn process(&mut self) -> f32 {
        let raw = match &mut self.kernel {
            Kernel::Plucked { string, exc } => {
                let excitation = exc.tick();
                string.process(excitation)
            }
            Kernel::Modal(bank) => bank.process(),
        };
        if self.fade < 1.0 {
            self.fade = (self.fade + self.fade_inc).min(1.0);
        }
        let output = raw * self.fade;
        let magnitude = output.abs();
        self.env = if magnitude > self.env {
            magnitude
        } else {
            self.env * self.env_release
        };
        output
    }
}

/// Decorrelate excitation noise between voices without touching a global RNG:
/// mix the note and serial into a non-zero `xorshift` seed.
#[inline]
fn seed_for(note: u8, serial: u64) -> u32 {
    let mixed = (serial as u32).wrapping_mul(2_654_435_761)
        ^ (note as u32).wrapping_mul(40_503).wrapping_add(0x9E37);
    mixed | 1
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pitch::max_loop_len;

    const SR: f32 = 48_000.0;

    fn plucked_params() -> PhysicalParams {
        PhysicalParams {
            model: Model::Plucked,
            ..PhysicalParams::default()
        }
    }

    #[test]
    fn triggered_voice_produces_sound_then_reports_level() {
        let mut voice = Voice::new(SR, Model::Plucked, max_loop_len(SR));
        voice.trigger(69, 1.0, 1, &plucked_params(), SR);
        let mut peak = 0.0f32;
        for _ in 0..4_000 {
            peak = peak.max(voice.process().abs());
        }
        assert!(peak > 1e-3, "voice made no sound");
        assert!(voice.level() > 0.0, "level not tracked");
        assert!(voice.is_held());
    }

    #[test]
    fn fade_in_starts_from_silence() {
        let mut voice = Voice::new(SR, Model::Modal, max_loop_len(SR));
        let params = PhysicalParams {
            model: Model::Modal,
            ..PhysicalParams::default()
        };
        voice.trigger(60, 1.0, 1, &params, SR);
        // The first output sample is scaled by a near-zero fade gain.
        let first = voice.process().abs();
        let mut later_peak = 0.0f32;
        for _ in 0..2_000 {
            later_peak = later_peak.max(voice.process().abs());
        }
        assert!(first <= later_peak, "fade-in did not ramp up");
    }

    #[test]
    fn released_voice_becomes_available_after_decay() {
        let mut voice = Voice::new(SR, Model::Plucked, max_loop_len(SR));
        voice.trigger(88, 1.0, 1, &plucked_params(), SR);
        for _ in 0..2_000 {
            voice.process();
        }
        assert!(
            !voice.is_available(1e-4),
            "held voice must not be available"
        );
        voice.release();
        assert!(!voice.is_held());
        // Ring out until the follower reports silence.
        for _ in 0..SR as usize * 2 {
            voice.process();
        }
        assert!(voice.is_available(1e-3), "decayed voice never freed");
    }

    #[test]
    fn retrigger_matches_a_fresh_voice() {
        // A reused voice must reproduce a fresh one bit-for-bit (no leak).
        let mut reused = Voice::new(SR, Model::Plucked, max_loop_len(SR));
        reused.trigger(72, 1.0, 1, &plucked_params(), SR);
        for _ in 0..3_000 {
            reused.process();
        }
        reused.trigger(64, 0.8, 2, &plucked_params(), SR);
        let reused_out: Vec<f32> = (0..2_000).map(|_| reused.process()).collect();

        let mut fresh = Voice::new(SR, Model::Plucked, max_loop_len(SR));
        fresh.trigger(64, 0.8, 2, &plucked_params(), SR);
        let fresh_out: Vec<f32> = (0..2_000).map(|_| fresh.process()).collect();

        assert_eq!(
            reused_out, fresh_out,
            "retriggered voice leaked prior state"
        );
    }

    #[test]
    fn output_is_finite() {
        let mut voice = Voice::new(SR, Model::Modal, max_loop_len(SR));
        let params = PhysicalParams {
            model: Model::Modal,
            t60: 30.0,
            ..PhysicalParams::default()
        };
        voice.trigger(48, 1.0, 1, &params, SR);
        for _ in 0..96_000 {
            assert!(voice.process().is_finite());
        }
    }
}
