//! Real-time physical-modelling engine for the `rt-physical` plugin
//! (see `docs/Roadmap-06-physical-modeling.md`).
//!
//! # Crate scope
//!
//! `rt-physical-core` is the **model core**: everything the render thread needs
//! to synthesise plucked strings and struck/metallic tones, and nothing a
//! plugin format needs.  It is framework-agnostic — no `nih-plug` / LV2 / CLAP
//! / VST3 leaks in — and reuses the root `rt-fft-convolver` crate only for the
//! shared anti-denormal helpers (`flush_to_zero`, re-exported [`DenormalGuard`]).
//!
//! # Model families (Phase 0 decisions)
//!
//! v1 ships two families, selected per manager:
//!
//! - **Extended Karplus-Strong** ([`KarplusStrong`]) — a plucked string built
//!   from a [`DelayLine`], a [`TuningAllpass`] for sub-sample pitch and a
//!   [`LoopFilter`] for decay + damping, excited by a velocity/hardness-shaped,
//!   pluck-positioned noise burst ([`Excitation`]).
//! - **Modal bank** ([`ModalBank`]) — a parallel array of [`Biquad`] resonators
//!   turning one impulse into a sum of decaying sinusoids, with material
//!   [`ModePreset`]s (metal / wood / glass).
//!
//! Both are voiced by a fixed-capacity [`VoiceManager`] with allocation-free
//! note-on / note-off and *quietest-then-oldest* voice stealing across
//! [`MAX_VOICES`] voices.
//!
//! # Real-time safety
//!
//! Every `process` method is allocation-free, lock-free and panic-free in
//! release builds, and flushes decaying state to zero so long tails never fall
//! into the denormal slow path.  Build managers, strings and banks off the
//! render thread.
//!
//! # Example
//!
//! ```
//! use rt_physical_core::{Model, PhysicalParams, PhysicalVoices};
//!
//! let params = PhysicalParams { model: Model::Plucked, ..PhysicalParams::default() };
//! let mut voices = PhysicalVoices::new(48_000.0, params);
//! voices.note_on(69, 1.0); // A4, full velocity
//! let sample = voices.process();
//! assert!(sample.is_finite());
//! ```

pub mod allpass;
pub mod biquad;
pub mod delay;
pub mod excitation;
pub mod filter;
pub mod manager;
pub mod modal;
pub mod pitch;
pub mod string;
pub mod voice;

pub use allpass::TuningAllpass;
pub use biquad::Biquad;
pub use delay::DelayLine;
pub use excitation::Excitation;
pub use filter::LoopFilter;
pub use manager::VoiceManager;
pub use modal::{MAX_MODES, ModalBank, ModePreset};
pub use pitch::{MIN_FREQ_HZ, max_loop_len, midi_to_freq};
pub use string::KarplusStrong;
pub use voice::{Model, PhysicalParams, Voice};

/// Maximum simultaneous voices — the Phase 0 polyphony decision (a guitar's six
/// strings plus ring-over headroom, well under the per-voice CPU ceiling).
pub const MAX_VOICES: usize = 16;

/// The default voice manager sized to [`MAX_VOICES`].
pub type PhysicalVoices = VoiceManager<MAX_VOICES>;

/// Re-export the shared anti-denormal guard so the plugin layer can wrap its
/// render callback without depending on the root crate directly.
pub use rt_fft_convolver::DenormalGuard;
