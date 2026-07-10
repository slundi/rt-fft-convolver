//! `rt-ir-cab` — guitar / bass cabinet simulator plugin.
//!
//! **Phase 1 placeholder.** The Phase-0 shared foundation lives in
//! [`rt_ir_plugin_core`]; this crate will add the `nih-plug` integration,
//! [`HybridConvolver`](rt_fft_convolver::HybridConvolver)-based I/O, the slot
//! controls, EQ, and presets described in `docs/Roadmap-lv2.md`.

// Re-exported so the shared foundation is reachable (and the dependency is
// exercised) until the real plugin lands.
pub use rt_ir_plugin_core as core;
