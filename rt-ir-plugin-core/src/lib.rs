//! Shared foundation for the `rt-ir-cab` and `rt-ir-reverb` plugins.
//!
//! This crate is **framework-agnostic**: it has no dependency on `nih-plug`
//! (or any plugin format).  It provides the reusable, real-time-safe building
//! blocks the two plugins wire together in later roadmap phases:
//!
//! | Module        | Responsibility                                               |
//! |---------------|--------------------------------------------------------------|
//! | [`ir`]        | [`PreparedIr`] — a resampled IR ready to be partitioned.     |
//! | [`slot`]      | [`IrSlotBank`] — `N` preloaded IR slots + activation intent. |
//! | [`handoff`]   | Lock-free control→audio handoff of the active convolver.     |
//! | [`midi`]      | Program-Change / CC → slot-index routing.                    |
//! | [`crossfade`] | [`SlotCrossfader`] — parametric equal-power IR crossfade.     |
//!
//! # Threading model
//!
//! Two threads cooperate (see `docs/Roadmap-lv2.md`, Phase 0):
//!
//! - **Control / worker thread** — file I/O, resampling, partitioning, and
//!   activation requests.  Allocates freely; never touches the audio buffer.
//! - **Audio thread** — pulls the currently-active convolver through the
//!   lock-free [`handoff`] and runs [`SlotCrossfader::process`].  Never
//!   allocates, locks, or blocks.

pub mod crossfade;
pub mod handoff;
pub mod ir;
pub mod midi;
pub mod slot;

pub use crossfade::{MonoConvolver, SlotCrossfader, ms_to_samples};
pub use handoff::{IrConsumer, IrPublisher, ir_handoff};
pub use ir::PreparedIr;
pub use midi::{MidiRouter, SlotEvent, cc_to_slot, program_to_slot};
pub use slot::{IrSlotBank, SlotError};
